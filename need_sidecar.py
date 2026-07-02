#!/usr/bin/env python3
"""Universal single-sidecar runtime for NEED.

Only one sidecar backend is active at runtime.  The backend can be an external LM
runtime or a smaller NEED checkpoint.  The smaller NEED backend is deliberately
architecture-native and latent-only: it exposes compact latent summaries and
projected latent-path vectors as advisory guidance without acting as a reasoning
text generator or speculative final decoder.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from need_core import ByteTokenizer, load_model, resolve_device

try:  # Optional; only needed for external LM sidecars.
    from sidecar_lm_runtime import FastSidecarLMRuntime, SidecarLMRuntimeConfig
except Exception:  # pragma: no cover
    FastSidecarLMRuntime = None  # type: ignore[assignment]
    SidecarLMRuntimeConfig = None  # type: ignore[assignment]


@dataclass
class SidecarSelection:
    backend: str
    reason: str
    checkpoint_or_model: str = ""


def resolve_single_sidecar_backend(args: Any) -> SidecarSelection:
    """Resolve exactly one sidecar backend from generation/runtime args.

    backend is one of: none, external_lm, need.  In auto mode a NEED checkpoint
    wins over an external LM model because it is the architecture-native sidecar;
    the external LM model remains configured but is not loaded.
    """
    requested = str(getattr(args, "sidecar_type", "auto") or "auto").lower().strip()
    need_ckpt = str(getattr(args, "need_sidecar_checkpoint", "") or "").strip()
    lm_model = str(getattr(args, "sidecar_model", "") or "").strip()
    if requested in {"none", "off", "disabled"}:
        return SidecarSelection("none", "disabled by --sidecar_type", "")
    if requested == "need":
        if not need_ckpt:
            raise ValueError("--sidecar_type need requires --need_sidecar_checkpoint")
        return SidecarSelection("need", "explicit NEED sidecar", need_ckpt)
    if requested == "external_lm":
        if not lm_model:
            raise ValueError("--sidecar_type external_lm requires --sidecar_model")
        return SidecarSelection("external_lm", "explicit external LM sidecar", lm_model)
    if requested != "auto":
        raise ValueError(f"unknown sidecar_type: {requested}")
    if need_ckpt:
        return SidecarSelection("need", "auto selected NEED sidecar; external LM sidecar disabled", need_ckpt)
    if lm_model:
        return SidecarSelection("external_lm", "auto selected external LM sidecar", lm_model)
    return SidecarSelection("none", "no sidecar configured", "")


class NeedLatentProjection(nn.Module):
    """Small projection from sidecar NEED latent dim into main NEED latent dim."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 0):
        super().__init__()
        if in_dim == out_dim and hidden_dim <= 0:
            self.net = nn.Identity()
        elif hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, out_dim))
        else:
            self.net = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _safe_torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


class NeedSidecarRuntime:
    """Runtime wrapper for a smaller NEED sidecar.

    It intentionally mirrors the subset of FastSidecarLMRuntime needed by older
    call sites, but the reasoning-text methods fail closed.  A NEED sidecar is
    latent-only by default and by capability flags: generate()/generate_many()
    remain available for compatibility, while generate_artificial_cot_and_summary(),
    summary_prompt(), and artificial_cot_prompt() do not produce artificial
    reasoning traces.
    """

    source_type = "need"
    supports_speculative_final_decode = False
    supports_reasoning_sidecar = False

    def __init__(
        self,
        checkpoint: str,
        device: str | torch.device = "auto",
        prefer_best: bool = False,
        kernel_backend: Optional[str] = None,
        target_dim: Optional[int] = None,
        projection_path: str = "",
        projection_weight: float = 1.0,
        max_context_tokens: int = 512,
        decode_mode: str = "nonseq",
    ) -> None:
        self.checkpoint = str(checkpoint)
        self.device = resolve_device(str(device)) if not isinstance(device, torch.device) else device
        self.prefer_best = bool(prefer_best)
        self.kernel_backend = kernel_backend
        self.target_dim = int(target_dim) if target_dim is not None else 0
        self.projection_path = str(projection_path or "")
        self.projection_weight = float(projection_weight)
        self.max_context_tokens = int(max_context_tokens)
        self.decode_mode = str(decode_mode or "nonseq")
        self.model = None
        self.tokenizer = ByteTokenizer()
        self.latent_projection: Optional[nn.Module] = None
        self.cache_plan: Dict[str, Any] = {"sidecar_backend": "need", "checkpoint": self.checkpoint}

    def load(self) -> "NeedSidecarRuntime":
        self.model = load_model(self.checkpoint, device=self.device, prefer_best=self.prefer_best, kernel_backend=self.kernel_backend)
        self.model.eval()
        self._load_projection()
        self.cache_plan.update({
            "sidecar_d_model": int(self.model.cfg.d_model),
            "target_d_model": int(self.target_dim or self.model.cfg.d_model),
            "latent_projection": bool(self.latent_projection is not None),
            "supports_speculative_final_decode": bool(self.supports_speculative_final_decode),
            "supports_reasoning_sidecar": bool(self.supports_reasoning_sidecar),
        })
        return self

    def _projection_candidate_paths(self) -> List[Path]:
        if self.projection_path:
            p = Path(self.projection_path)
            if p.is_dir():
                return [p / "need_sidecar_projection.pt", p / "sidecar_need_projection.pt", p / "projection.pt"]
            return [p]
        ck = Path(self.checkpoint)
        return [ck / "need_sidecar_projection.pt", ck / "sidecar_need_projection.pt", ck / "projection.pt"]

    def _load_projection(self) -> None:
        assert self.model is not None
        in_dim = int(self.model.cfg.d_model)
        out_dim = int(self.target_dim or in_dim)
        if in_dim == out_dim and not self.projection_path:
            self.latent_projection = NeedLatentProjection(in_dim, out_dim).to(self.device)
            self.latent_projection.eval()
            return
        for p in self._projection_candidate_paths():
            if not p.exists():
                continue
            data = _safe_torch_load(p, map_location=self.device)
            hidden_dim = 0
            state = data
            if isinstance(data, dict):
                hidden_dim = int(data.get("hidden_dim", data.get("projection_hidden_dim", 0)) or 0)
                in_dim = int(data.get("in_dim", in_dim))
                out_dim = int(data.get("out_dim", out_dim))
                state = data.get("state_dict", data.get("projection", data))
            proj = NeedLatentProjection(in_dim, out_dim, hidden_dim=hidden_dim).to(self.device)
            if isinstance(state, dict):
                missing, unexpected = proj.load_state_dict(state, strict=False)
                self.cache_plan["projection_missing_keys"] = list(missing)
                self.cache_plan["projection_unexpected_keys"] = list(unexpected)
            self.latent_projection = proj.eval()
            self.target_dim = out_dim
            self.cache_plan["projection_path"] = str(p)
            return
        self.latent_projection = None
        self.cache_plan["projection_error"] = f"no projection for sidecar dim {in_dim} -> target dim {out_dim}"

    def _ids(self, text: str, add_bos: bool = True) -> torch.Tensor:
        assert self.model is not None
        ids = self.tokenizer.encode(str(text), add_bos=add_bos)[-int(self.model.cfg.block_size):]
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    @torch.no_grad()
    def encode_latent_alignment(self, prompts: Sequence[str], max_length: int = 2048) -> torch.Tensor:
        assert self.model is not None
        if self.latent_projection is None:
            raise RuntimeError("NEED sidecar latent projection is unavailable; train with need_sidecar_distill.py or use matching d_model")
        vecs: List[torch.Tensor] = []
        for prompt in prompts:
            ids = self.tokenizer.encode(str(prompt), add_bos=True)[-min(int(max_length), int(self.model.cfg.block_size)):]
            x = torch.tensor([ids], dtype=torch.long, device=self.device)
            path = self.model.latent_pathway(x, stride=2, max_vectors=64)
            v = path.get("pathway_vectors")
            if v is None:
                h, _, _ = self.model.encode_hidden(x)
                v = h[:, -1:, :]
            z = self.latent_projection(v.to(self.device).float()) * float(self.projection_weight)
            vecs.append(z[:, -1:, :].detach())
        return torch.cat(vecs, dim=0)

    @torch.no_grad()
    def latent_guidance(self, prompt: str, vector_stride: int = 2, max_vectors: int = 128) -> Tuple[str, Optional[torch.Tensor], Dict[str, Any]]:
        assert self.model is not None
        x = self._ids(prompt)
        path = self.model.latent_pathway(x, stride=vector_stride, max_vectors=max_vectors)
        vectors = path.get("pathway_vectors")
        projected = None
        if vectors is not None and self.latent_projection is not None:
            projected = self.latent_projection(vectors.float()) * float(self.projection_weight)
        summary = self._latent_summary_from_path(path)
        metrics = {
            "sidecar_backend": "need",
            "sidecar_vectors": int(vectors.size(1)) if torch.is_tensor(vectors) else 0,
            "sidecar_projected_vectors": int(projected.size(1)) if torch.is_tensor(projected) else 0,
            "sidecar_checkpoint": self.checkpoint,
        }
        return summary, projected, metrics

    def _latent_summary_from_path(self, path: Dict[str, Any]) -> str:
        def f(key: str, default: float = 0.0) -> float:
            v = path.get(key, default)
            try:
                if torch.is_tensor(v):
                    return float(v.detach().float().mean().cpu())
                return float(v)
            except Exception:
                return default
        return (
            "<need_sidecar_latent_summary>"
            f"quality={f('quality_mean',0.5):.3f}; risk={f('risk_mean',0.5):.3f}; "
            f"risk_signal={f('risk_signal_mean',0.0):.3f}; contradiction={f('contradiction_mean',0.0):.3f}. "
            "Use compact, grounded answering and preserve uncertainty."
            "</need_sidecar_latent_summary>"
        )

    def artificial_cot_prompt(self, prompt: str, latent_summary: str, raw_cot_history: str = "") -> str:
        # NEED sidecars are latent-only.  They do not draft reasoning text; this
        # method is retained only so older callers fail closed instead of using a
        # smaller NEED checkpoint as an artificial CoT generator.
        return str(latent_summary or "")

    def summary_prompt(self, prompt: str, raw_cot: str, latent_summary: str) -> str:
        # NEED sidecars expose latent summaries/anchors, not reasoning traces.
        return str(latent_summary or raw_cot or "")

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7, top_p: float = 0.95, top_k: int = 80, stop: Optional[Sequence[str]] = None, **_: Any) -> str:
        assert self.model is not None
        x = self._ids(prompt)
        if self.decode_mode == "ar" or int(getattr(self.model.cfg, "n_predict_heads", 1)) <= 1:
            out = self.model.generate_text(x, max_new_tokens=int(max_new_tokens), temperature=float(temperature), top_k=int(top_k), top_p=float(top_p))
        else:
            out, _stats = self.model.generate_text_nonsequential(x, max_new_tokens=int(max_new_tokens), temperature=float(temperature), top_k=int(top_k), top_p=float(top_p), return_stats=True)
        text = self.tokenizer.decode(out[0, x.size(1):].tolist())
        if stop:
            cut = len(text)
            for marker in stop:
                if marker:
                    pos = text.find(marker)
                    if pos >= 0:
                        cut = min(cut, pos)
            text = text[:cut]
        return text.strip()

    @torch.no_grad()
    def generate_many(self, prompts: Sequence[str], max_new_tokens: int = 128, temperature: float = 0.7, top_p: float = 0.95, top_k: int = 80, stop: Optional[Sequence[str]] = None, **kwargs: Any) -> List[str]:
        return [self.generate(p, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k, stop=stop, **kwargs) for p in prompts]

    @torch.no_grad()
    def generate_artificial_cot_and_summary(
        self,
        prompt: str,
        latent_summary: str,
        raw_cot_history: str = "",
        cot_tokens: int = 160,
        summary_tokens: int = 120,
        temperature: float = 0.45,
        top_p: float = 0.92,
        **kwargs: Any,
    ) -> Tuple[str, str]:
        # Fail closed for legacy call sites: the active NEED sidecar contributes
        # latent anchors only and never generates artificial reasoning text.
        return "", str(latent_summary or "")


class ExternalLMSidecarRuntime:
    """Thin adapter that marks a FastSidecarLMRuntime as the active sidecar."""
    source_type = "external_lm"
    supports_speculative_final_decode = True
    supports_reasoning_sidecar = True

    def __init__(self, runtime: Any):
        self.runtime = runtime
        self.cache_plan = getattr(runtime, "cache_plan", {})
        self.latent_projection = getattr(runtime, "latent_projection", None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.runtime, name)


def make_single_sidecar_runtime(args: Any, device: torch.device, main_model: Any) -> Optional[Any]:
    """Load exactly one sidecar runtime or return None."""
    selected = resolve_single_sidecar_backend(args)
    setattr(args, "_sidecar_selection", selected.__dict__)
    if selected.backend == "none":
        return None
    if selected.backend == "need":
        rt = NeedSidecarRuntime(
            checkpoint=selected.checkpoint_or_model,
            device=device,
            prefer_best=bool(getattr(args, "need_sidecar_prefer_best", False)),
            kernel_backend=getattr(args, "kernel_backend", None),
            target_dim=int(getattr(main_model.cfg, "d_model", 0)),
            projection_path=str(getattr(args, "need_sidecar_projection_path", "") or getattr(args, "sidecar_latent_alignment_path", "") or ""),
            projection_weight=float(getattr(args, "need_sidecar_projection_weight", getattr(args, "sidecar_latent_alignment_weight", 1.0))),
            max_context_tokens=int(getattr(args, "need_sidecar_max_context_tokens", getattr(args, "sidecar_max_context_tokens", 512))),
            decode_mode=str(getattr(args, "need_sidecar_decode_mode", "nonseq") or "nonseq"),
        ).load()
        return rt
    if FastSidecarLMRuntime is None or SidecarLMRuntimeConfig is None:
        raise RuntimeError("sidecar_lm_runtime is unavailable; cannot load external LM sidecar")
    model_name = getattr(args, "cot_model", "") or getattr(args, "summary_model", "") or selected.checkpoint_or_model
    cfg = SidecarLMRuntimeConfig(
        model=model_name,
        device=str(device) if getattr(args, "sidecar_device", "same") == "same" else getattr(args, "sidecar_device", "auto"),
        dtype=getattr(args, "sidecar_dtype", "bf16"),
        attn_backend=getattr(args, "sidecar_attn_backend", "sdpa"),
        compile=bool(getattr(args, "sidecar_compile", False)),
        max_batch=int(getattr(args, "sidecar_max_batch", 8)),
        max_wait_ms=int(getattr(args, "sidecar_max_wait_ms", 8)),
        cache_implementation=getattr(args, "sidecar_cache_implementation", "static"),
        l2_cache_mb=float(getattr(args, "gpu_l2_mb", 96.0)),
        max_context_tokens=int(getattr(args, "sidecar_max_context_tokens", 2048)),
        trust_remote_code=bool(getattr(args, "sidecar_trust_remote_code", False)),
        adapter_path=str(getattr(args, "sidecar_adapter_path", "") or ""),
        latent_alignment_path=str(getattr(args, "sidecar_latent_alignment_path", "") or ""),
    )
    return ExternalLMSidecarRuntime(FastSidecarLMRuntime(cfg).load())
