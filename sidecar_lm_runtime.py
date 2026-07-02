#!/usr/bin/env python3
"""High-throughput external LM sidecar runtime for CoT and summary passes.

The NEED runtime stays optimized for the recurrent/energy path.  When a very small
external LM is used for artificial CoT or summary text, this
runtime switches CUDA settings, uses flash/SDPA attention where available, builds
L2-aware cache plans, and optionally batches concurrent requests.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import warnings
from contextlib import contextmanager
from pathlib import Path
from queue import Queue, Empty
from threading import Event, Thread, Lock
from time import perf_counter, sleep
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sidecar_attention_kernels import configure_torch_attention, estimate_sidecar_lm_tps, plan_from_config


@dataclass
class SidecarLMRuntimeConfig:
    model: str = "HuggingFaceTB/SmolLM2-135M-Instruct" 
    device: str = "auto"
    dtype: str = "bf16"
    attn_backend: str = "auto"  # auto|sdpa|flash_attention_2|eager
    compile: bool = False
    max_batch: int = 8
    max_wait_ms: int = 8
    cache_implementation: str = "static"  # static|dynamic|offloaded|none
    l2_cache_mb: float = 96.0
    max_context_tokens: int = 2048
    trust_remote_code: bool = False
    adapter_path: str = ""
    latent_alignment_path: str = ""
    load_adapter: bool = True


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    name = name.lower()
    if name in ("bf16", "bfloat16") and device.type == "cuda":
        return torch.bfloat16
    if name in ("fp16", "float16") and device.type == "cuda":
        return torch.float16
    return torch.float32


@contextmanager
def sidecar_optimization_mode():
    configure_torch_attention(enable_flash=True, enable_mem_efficient=True, enable_math=True)
    try:
        yield
    finally:
        pass


@contextmanager
def need_optimization_mode():
    # Keep the NEED path conservative; its kernels may be custom recurrent/Triton
    # code and should not inherit external-LM-specific assumptions beyond TF32.
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
    try:
        yield
    finally:
        pass


class SidecarLatentProjection(nn.Module):
    """Projection head that maps an external LM sidecar hidden state into NEED latent space."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 0, normalize: bool = True):
        super().__init__()
        self.normalize = bool(normalize)
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, out_dim))
        else:
            self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return F.normalize(y.float(), dim=-1) if self.normalize else y.float()


def _load_projection(path: str, device: torch.device, dtype: torch.dtype) -> tuple[Optional[SidecarLatentProjection], Dict[str, Any]]:
    if not path:
        return None, {}
    root = Path(path)
    pt = root / "latent_projection.pt" if root.is_dir() else root
    if not pt.exists():
        warnings.warn(f"sidecar latent projection not found: {pt}")
        return None, {}
    try:
        payload = torch.load(pt, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(pt, map_location="cpu")
    cfg = dict(payload.get("config", {})) if isinstance(payload, dict) else {}
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    in_dim = int(cfg.get("sidecar_hidden_size", cfg.get("in_dim", 0)))
    out_dim = int(cfg.get("need_dim", cfg.get("out_dim", 0)))
    if in_dim <= 0 or out_dim <= 0:
        raise ValueError(f"latent projection config in {pt} is missing sidecar_hidden_size/need_dim")
    hidden_dim = int(cfg.get("projection_hidden_dim", 0))
    proj = SidecarLatentProjection(in_dim, out_dim, hidden_dim=hidden_dim, normalize=bool(cfg.get("normalize", True)))
    proj.load_state_dict(state, strict=False)
    proj.to(device=device, dtype=torch.float32)
    proj.eval()
    return proj, cfg


class FastSidecarLMRuntime:
    def __init__(self, cfg: SidecarLMRuntimeConfig):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.dtype = resolve_dtype(cfg.dtype, self.device)
        self.model = None
        self.tokenizer = None
        self.latent_projection: Optional[SidecarLatentProjection] = None
        self.latent_alignment_config: Dict[str, Any] = {}
        self.adapter_loaded: bool = False
        self.lock = Lock()
        self.cache_plan: Dict[str, Any] = {}

    @classmethod
    def maybe_load(cls, model: str, **kwargs: Any) -> Optional["FastSidecarLMRuntime"]:
        if not model:
            return None
        cfg = SidecarLMRuntimeConfig(model=model, **kwargs)
        rt = cls(cfg)
        rt.load()
        return rt

    def load(self) -> "FastSidecarLMRuntime":
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as exc:
            raise RuntimeError("Install transformers to use --cot_model/--summary_model external LM sidecars") from exc

        attn_impl = None
        if self.cfg.attn_backend == "flash_attention_2":
            attn_impl = "flash_attention_2"
        elif self.cfg.attn_backend == "sdpa":
            attn_impl = "sdpa"
        elif self.cfg.attn_backend == "eager":
            attn_impl = "eager"
        elif self.cfg.attn_backend == "auto":
            # PyTorch SDPA is the most broadly compatible fast path; FlashAttention-2
            # can be selected explicitly if installed for the target GPU.
            attn_impl = "sdpa"

        kwargs: Dict[str, Any] = dict(torch_dtype=self.dtype, trust_remote_code=bool(self.cfg.trust_remote_code))
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl
        with sidecar_optimization_mode():
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model, trust_remote_code=bool(self.cfg.trust_remote_code))
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            try:
                self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model, **kwargs).to(self.device)
            except TypeError:
                kwargs.pop("attn_implementation", None)
                self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model, **kwargs).to(self.device)
            if self.cfg.adapter_path and bool(self.cfg.load_adapter):
                try:
                    from peft import PeftModel  # type: ignore
                    self.model = PeftModel.from_pretrained(self.model, self.cfg.adapter_path).to(self.device)
                    self.adapter_loaded = True
                except Exception as exc:
                    warnings.warn(f"Could not load sidecar adapter from {self.cfg.adapter_path}: {exc}")
            self.model.eval()
            try:
                self.model.config.use_cache = True
            except Exception:
                pass
            if self.cfg.compile and hasattr(torch, "compile"):
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)
                except Exception:
                    pass
        if self.cfg.latent_alignment_path:
            self.latent_projection, self.latent_alignment_config = _load_projection(self.cfg.latent_alignment_path, self.device, self.dtype)
        self.cache_plan = plan_from_config(
            self.model.config,
            dtype=self.dtype,
            l2_cache_mb=float(self.cfg.l2_cache_mb),
            max_batch=int(self.cfg.max_batch),
            context_tokens=int(self.cfg.max_context_tokens),
        ).to_dict()
        return self

    def estimate_tps(self) -> Dict[str, float]:
        params = int(getattr(self.model.config, "num_parameters", 135_000_000)) if self.model is not None else 135_000_000
        if hasattr(self.model, "num_parameters"):
            try:
                params = int(self.model.num_parameters())
            except Exception:
                pass
        return estimate_sidecar_lm_tps(params=params, dtype_bytes=2 if self.dtype in (torch.bfloat16, torch.float16) else 4)

    def _generate_batch(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        stop: Optional[Sequence[str]] = None,
    ) -> List[str]:
        if self.model is None or self.tokenizer is None:
            self.load()
        assert self.model is not None and self.tokenizer is not None
        with self.lock, sidecar_optimization_mode(), torch.inference_mode():
            enc = self.tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True, max_length=int(self.cfg.max_context_tokens)).to(self.device)
            gen_kwargs: Dict[str, Any] = dict(
                max_new_tokens=int(max_new_tokens),
                do_sample=temperature > 0,
                temperature=max(float(temperature), 1e-5),
                top_p=float(top_p),
                top_k=int(top_k),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
            if self.cfg.cache_implementation and self.cfg.cache_implementation != "none":
                gen_kwargs["cache_implementation"] = self.cfg.cache_implementation
            try:
                out = self.model.generate(**enc, **gen_kwargs)
            except TypeError:
                gen_kwargs.pop("cache_implementation", None)
                out = self.model.generate(**enc, **gen_kwargs)
            texts = self.tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        if stop:
            cleaned = []
            for t in texts:
                cut = len(t)
                for s in stop:
                    if s and s in t:
                        cut = min(cut, t.index(s))
                cleaned.append(t[:cut].strip())
            return cleaned
        return [t.strip() for t in texts]

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return self._generate_batch([prompt], **kwargs)[0]

    def generate_many(self, prompts: Sequence[str], **kwargs: Any) -> List[str]:
        return self._generate_batch(prompts, **kwargs)

    def latent_alignment_prompt(self, user_prompt: str, latent_summary: str = "") -> str:
        prefix = (
            "Map this task into NEED-compatible latent guidance. Produce a compact public working summary, "
            "not hidden chain-of-thought. Preserve uncertainty and avoid unsupported facts."
        )
        if latent_summary:
            return f"{prefix}\n\nTask:\n{user_prompt}\n\nReference latent summary:\n{latent_summary}\n\nPublic latent-aligned summary:\n"
        return f"{prefix}\n\nTask:\n{user_prompt}\n\nPublic latent-aligned summary:\n"

    def encode_latent_alignment(self, texts: Sequence[str], *, max_length: Optional[int] = None) -> torch.Tensor:
        """Return sidecar hidden states projected into NEED latent space.

        The output has shape [B, 1, need_dim] so it can be passed as ordered
        conditioning_vectors to NEED's pathway conditioner when desired.
        """
        if self.model is None or self.tokenizer is None:
            self.load()
        if self.latent_projection is None:
            raise RuntimeError("No sidecar latent projection loaded; pass latent_alignment_path in SidecarLMRuntimeConfig")
        assert self.model is not None and self.tokenizer is not None
        length = int(max_length or self.cfg.max_context_tokens)
        with self.lock, sidecar_optimization_mode(), torch.inference_mode():
            enc = self.tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=length).to(self.device)
            out = self.model(**enc, output_hidden_states=True, use_cache=False)
            hidden = out.hidden_states[-1]
            mask = enc["attention_mask"].to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            projected = self.latent_projection(pooled.float()).unsqueeze(1)
        return projected

    def artificial_cot_prompt(self, user_prompt: str, latent_summary: str, raw_cot_history: str = "") -> str:
        history = f"\nPrior raw artificial CoT context, for continuity only:\n{raw_cot_history}\n" if raw_cot_history else ""
        return (
            "You are a compact reasoning sidecar. Generate a useful artificial chain-of-thought style working note "
            "from the user's task and NEED latent-path summary. This is synthetic reasoning, not hidden truth. "
            "Be structured, check assumptions, and keep it short.\n"
            f"{history}\nUser task:\n{user_prompt}\n\nNEED latent pathway summary:\n{latent_summary}\n\nArtificial CoT working note:\n"
        )

    def summary_prompt(self, user_prompt: str, raw_cot: str, latent_summary: str) -> str:
        return (
            "Summarize the artificial reasoning into concise tagged chunks for the user. "
            "Do not expose unnecessary raw chain-of-thought; provide a useful public explanation. "
            "Use tags: <focus>, <assumptions>, <reasoning_chunks>, <uncertainty>, <answer_check>.\n\n"
            f"User task:\n{user_prompt}\n\nNEED latent pathway summary:\n{latent_summary}\n\nRaw artificial CoT:\n{raw_cot}\n\nTagged public summary:\n"
        )

    def generate_artificial_cot_and_summary(
        self,
        user_prompt: str,
        latent_summary: str,
        *,
        raw_cot_history: str = "",
        cot_tokens: int = 220,
        summary_tokens: int = 180,
        temperature: float = 0.45,
        top_p: float = 0.92,
    ) -> Tuple[str, str]:
        cot_prompt = self.artificial_cot_prompt(user_prompt, latent_summary, raw_cot_history)
        raw_cot = self.generate(cot_prompt, max_new_tokens=cot_tokens, temperature=temperature, top_p=top_p, top_k=80, stop=["\n\nFinal", "Tagged public summary:"])
        sum_prompt = self.summary_prompt(user_prompt, raw_cot, latent_summary)
        summary = self.generate(sum_prompt, max_new_tokens=summary_tokens, temperature=0.25, top_p=0.9, top_k=60, stop=["\n\nUser", "\n\nRaw"])
        return raw_cot.strip(), summary.strip()


class ConcurrentSidecarService:
    """Small continuous-batching service for the external LM sidecar.

    It batches prompts that arrive within max_wait_ms, which improves aggregate TPS
    when multiple browser/API requests ask for CoT or summaries at once.
    """

    def __init__(self, runtime: FastSidecarLMRuntime):
        self.runtime = runtime
        self.q: "Queue[Tuple[str, Dict[str, Any], Event, Dict[str, Any]]]" = Queue()
        self.thread: Optional[Thread] = None
        self.closed = False

    def start(self) -> "ConcurrentSidecarService":
        if self.thread is None:
            self.thread = Thread(target=self._worker, daemon=True)
            self.thread.start()
        return self

    def close(self) -> None:
        self.closed = True

    def generate(self, prompt: str, **kwargs: Any) -> str:
        ev = Event()
        box: Dict[str, Any] = {}
        self.q.put((prompt, kwargs, ev, box))
        ev.wait()
        if "exc" in box:
            raise box["exc"]
        return str(box.get("text", ""))

    def _worker(self) -> None:
        while not self.closed:
            try:
                first = self.q.get(timeout=0.05)
            except Empty:
                continue
            batch = [first]
            deadline = perf_counter() + max(0.0, self.runtime.cfg.max_wait_ms / 1000.0)
            while len(batch) < self.runtime.cfg.max_batch and perf_counter() < deadline:
                try:
                    batch.append(self.q.get(timeout=max(0.0, deadline - perf_counter())))
                except Empty:
                    break
            prompts = [b[0] for b in batch]
            kwargs = batch[0][1]
            try:
                texts = self.runtime.generate_many(prompts, **kwargs)
                for text, (_, _, ev, box) in zip(texts, batch):
                    box["text"] = text
                    ev.set()
            except Exception as exc:
                for _, _, ev, box in batch:
                    box["exc"] = exc
                    ev.set()
