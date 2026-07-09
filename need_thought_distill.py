#!/usr/bin/env python3
"""Sidecar latent-alignment utilities for NEED.

The standalone vector captioning model has been removed. This module now handles the
full external-LM-sidecar path instead:

1. build NEED latent targets from prompts/corpora/transcripts,
2. train an external LM sidecar adapter/projection to align with NEED latent space,
3. evaluate the alignment, and
4. keep the older sidecar summary distillation path for compatibility.

The sidecar learns public, compact latent summaries and a projection from its hidden
state into NEED's ordered latent-vector space. It should not expose private chain of
thought or claim direct access to hidden internals at runtime.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from need_core import ByteTokenizer, load_model, resolve_device, save_json, load_tokenizer_for_dir
from sidecar_lm_runtime import (
    FastSidecarLMRuntime,
    SidecarLatentProjection,
    SidecarLMRuntimeConfig,
    sidecar_optimization_mode,
)


def _safe_torch_load(path: Path, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _clean_text(x: object, max_len: int = 12000) -> str:
    return " ".join(str(x or "").replace("\r\n", "\n").split())[:max_len]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _iter_texts_from_corpus(path: Path, max_docs: int = 0) -> List[str]:
    if path.suffix.lower() in {".jsonl", ".json"}:
        raw = _read_jsonl(path) if path.suffix.lower() == ".jsonl" else json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("rows", raw.get("data", [raw]))
        texts: List[str] = []
        for row in raw:
            if not isinstance(row, dict):
                continue
            text = row_to_input_text(row)
            if text:
                texts.append(text)
            if max_docs and len(texts) >= max_docs:
                break
        return texts
    chunks = path.read_text(encoding="utf-8", errors="replace").split("\n\n")
    texts = [_clean_text(c, 12000) for c in chunks if _clean_text(c, 12000)]
    return texts[:max_docs] if max_docs else texts


def _as_float(x: object, default: float = 0.0) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
        return v if math.isfinite(v) else default
    except Exception:
        return default


def public_summary_template(text: str, summary_only: bool = True) -> str:
    words = " ".join(text.split()[:80])
    return (
        "<focus> Task-relevant latent pathway. </focus> "
        f"<reasoning_chunks> {words} </reasoning_chunks> "
        "<uncertainty> Preserve uncertainty; avoid unsupported details. </uncertainty>"
    )


def row_to_input_text(row: Dict[str, Any]) -> str:
    """Normalize synthetic RL/SFT/transcript rows into one sidecar training prompt."""
    if "messages" in row and isinstance(row["messages"], list):
        parts = []
        for m in row["messages"]:
            if isinstance(m, dict):
                role = str(m.get("role", "user"))
                content = _clean_text(m.get("content"), 6000)
                if content:
                    parts.append(f"{role}: {content}")
        return "\n".join(parts)
    if "overheard_transcription" in row and isinstance(row["overheard_transcription"], dict):
        tx = row["overheard_transcription"]
        scene = _clean_text(tx.get("scene", row.get("topic", "overheard scenario")), 300)
        transcript = _clean_text(tx.get("transcript", tx.get("text", "")), 6000)
        task = _clean_text(row.get("prompt", row.get("task", "Use the transcript to answer.")), 1000)
        return f"Scenario: {scene}\nProvided transcript: {transcript}\nTask: {task}"
    if "prompt" in row or "task" in row:
        prompt = _clean_text(row.get("prompt", row.get("task", "")), 6000)
        inp = row.get("input", "")
        if isinstance(inp, dict):
            inp = json.dumps(inp, ensure_ascii=False, sort_keys=True)
        extra = _clean_text(inp, 5000)
        return (prompt + ("\nInput: " + extra if extra else "")).strip()
    for key in ("text", "input_text", "question", "topic"):
        if key in row:
            return _clean_text(row.get(key), 8000)
    return ""


def target_summary_from_need(text: str, metrics: Dict[str, float], max_words: int = 95) -> str:
    words = " ".join(text.split()[:max_words])
    uncertainty = "low" if metrics.get("risk", 0.0) < 0.35 and metrics.get("contradiction", 0.0) < 0.25 else "moderate"
    return (
        f"<focus>{words}</focus> "
        "<reasoning_chunks>Use the provided task evidence, preserve constraints, and keep the answer grounded.</reasoning_chunks> "
        f"<uncertainty>{uncertainty}; quality={metrics.get('quality', 0.5):.2f}, risk={metrics.get('risk', 0.5):.2f}, contradiction={metrics.get('contradiction', 0.5):.2f}</uncertainty> "
        "<answer_check>Do not invent details; ask for clarification only when required.</answer_check>"
    )


def _sidecar_runtime(args: argparse.Namespace, device: torch.device) -> FastSidecarLMRuntime:
    cfg = SidecarLMRuntimeConfig(
        model=args.sidecar_model,
        device=str(device) if args.sidecar_device == "same" else args.sidecar_device,
        dtype=args.sidecar_dtype,
        attn_backend=args.sidecar_attn_backend,
        cache_implementation=args.sidecar_cache_implementation,
        trust_remote_code=bool(getattr(args, "sidecar_trust_remote_code", False)),
        adapter_path=str(getattr(args, "sidecar_adapter_path", "") or ""),
        latent_alignment_path=str(getattr(args, "sidecar_latent_alignment_path", "") or ""),
    )
    return FastSidecarLMRuntime(cfg).load()


def _alignment_prompt(user_prompt: str, latent_summary: str = "") -> str:
    base = (
        "Map this task into NEED-compatible latent guidance. Produce a compact public working summary, "
        "not hidden chain-of-thought. Preserve uncertainty and avoid unsupported facts."
    )
    if latent_summary:
        return f"{base}\n\nTask:\n{user_prompt}\n\nReference latent summary:\n{latent_summary}\n\nPublic latent-aligned summary:\n"
    return f"{base}\n\nTask:\n{user_prompt}\n\nPublic latent-aligned summary:\n"


def build_alignment_dataset(args: argparse.Namespace) -> None:
    """Create JSONL + tensor targets for sidecar-to-NEED latent alignment."""
    random.seed(args.seed)
    device = resolve_device(args.device)
    need = load_model(args.need_checkpoint, device=device, prefer_best=args.prefer_best)
    tok = load_tokenizer_for_dir(args.need_checkpoint)
    texts: List[str] = []
    if args.corpus:
        texts.extend(_iter_texts_from_corpus(Path(args.corpus), args.max_docs))
    if args.interactions:
        for row in _read_jsonl(Path(args.interactions)):
            txt = row_to_input_text(row)
            if txt:
                texts.append(txt)
    if args.transcripts_file:
        for row in _read_jsonl(Path(args.transcripts_file)):
            txt = row_to_input_text({"overheard_transcription": row, "prompt": row.get("task", "Use the provided transcript.")})
            if txt:
                texts.append(txt)
    seen = set()
    deduped = []
    for text in texts:
        text = _clean_text(text, args.max_input_chars)
        key = text[:500]
        if text and key not in seen:
            seen.add(key)
            deduped.append(text)
    if args.shuffle:
        random.shuffle(deduped)
    if args.max_examples > 0:
        deduped = deduped[: args.max_examples]
    if not deduped:
        raise ValueError("No input texts found for sidecar alignment dataset")

    runtime = None
    if args.bootstrap_sidecar_summaries:
        runtime = _sidecar_runtime(args, device)

    rows: List[Dict[str, Any]] = []
    latent_targets: List[torch.Tensor] = []
    endpoint_targets: List[torch.Tensor] = []
    faith_targets: List[torch.Tensor] = []
    pathway_targets: List[torch.Tensor] = []
    for idx, text in enumerate(deduped):
        ids = tok.encode(text, add_bos=True, add_eos=True)[: need.cfg.block_size]
        if len(ids) < 2:
            continue
        x = torch.tensor([ids], device=device)
        with torch.no_grad():
            path = need.latent_pathway(x, stride=args.vector_stride, max_vectors=args.max_vectors)
            vectors = path["pathway_vectors"].detach().float().cpu()[0]
            pooled = F.normalize(vectors.mean(dim=0), dim=-1)
            endpoint = F.normalize(path["pathway_endpoint"].detach().float().cpu()[0, 0], dim=-1)
            score = need.score_text_risk(x, conditioning_vectors=path["pathway_vectors"], conditioning_scale=args.conditioning_scale)
        metrics = {
            "quality": _as_float(score.get("quality"), 0.5),
            "risk": _as_float(score.get("risk"), 0.5),
            "contradiction": _as_float(score.get("contradiction"), 0.0),
        }
        latent_summary = public_summary_template(text, True)
        target_summary = target_summary_from_need(text, metrics, args.target_summary_words)
        if runtime is not None:
            with sidecar_optimization_mode():
                prompt = _alignment_prompt(text, latent_summary)
                sidecar_summary = runtime.generate(prompt, max_new_tokens=args.summary_tokens, temperature=0.25, top_p=0.9, top_k=60, stop=["\n\nTask", "\n\nUser"])
            if sidecar_summary and len(sidecar_summary.split()) >= 8:
                target_summary = sidecar_summary.strip()
        prompt = _alignment_prompt(text, latent_summary)
        rows.append(
            {
                "id": f"sidecar_latent_align_{idx:06d}",
                "input_text": text,
                "sidecar_prompt": prompt,
                "target_summary": target_summary,
                "need_latent_summary": latent_summary,
                "need_metrics": metrics,
                "target_index": len(latent_targets),
                "training_objectives": ["summary_lm", "latent_projection", "contrastive_alignment"],
            }
        )
        latent_targets.append(pooled)
        endpoint_targets.append(endpoint)
        faith_targets.append(torch.tensor([metrics["quality"], max(0.0, 1.0 - metrics["risk"]), metrics["contradiction"]], dtype=torch.float32))
        pathway_targets.append(vectors)

    if not rows:
        raise RuntimeError("no sidecar alignment examples produced")
    max_v = max(v.size(0) for v in pathway_targets)
    dim = pathway_targets[0].size(-1)
    padded = []
    masks = []
    for v in pathway_targets:
        mask = torch.zeros(max_v, dtype=torch.bool)
        mask[: v.size(0)] = True
        if v.size(0) < max_v:
            v = torch.cat([v, torch.zeros(max_v - v.size(0), dim)], dim=0)
        padded.append(v)
        masks.append(mask)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "sidecar_latent_alignment.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    torch.save(
        {
            "need_latent": torch.stack(latent_targets),
            "need_endpoint": torch.stack(endpoint_targets),
            "faith": torch.stack(faith_targets),
            "pathway_vectors": torch.stack(padded),
            "pathway_mask": torch.stack(masks),
            "config": {
                "need_dim": int(dim),
                "vector_stride": int(args.vector_stride),
                "max_vectors": int(args.max_vectors),
                "conditioning_scale": float(args.conditioning_scale),
                "examples": len(rows),
            },
        },
        out / "sidecar_latents.pt",
    )
    save_json({"format": "need_sidecar_latent_alignment_dataset", "examples": len(rows), "jsonl": str(jsonl), "latents": str(out / "sidecar_latents.pt")}, out / "manifest.json")
    print(json.dumps({"done": True, "examples": len(rows), "out_dir": str(out)}, indent=2), flush=True)


class SidecarAlignmentDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], targets: torch.Tensor, tokenizer: Any, max_length: int):
        self.rows = rows
        self.targets = targets.float()
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        prompt = str(row.get("sidecar_prompt", _alignment_prompt(str(row.get("input_text", "")))))
        target = str(row.get("target_summary", ""))
        return {"idx": idx, "prompt": prompt, "target": target, "latent": self.targets[int(row.get("target_index", idx))]}

    def collate(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = [b["prompt"] for b in batch]
        targets = [b["target"] for b in batch]
        full = [p + t for p, t in zip(prompts, targets)]
        enc = self.tokenizer(full, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        labels = enc["input_ids"].clone()
        prompt_enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        for i in range(len(batch)):
            plen = int(prompt_enc["attention_mask"][i].sum().item())
            labels[i, :plen] = -100
            labels[i, enc["attention_mask"][i] == 0] = -100
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "latent": torch.stack([b["latent"] for b in batch]),
        }


def _infer_lora_targets(model: nn.Module) -> List[str]:
    preferred = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "c_attn", "c_proj", "query", "key", "value", "dense"]
    suffixes = set()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and not name.endswith("lm_head"):
            suffixes.add(name.split(".")[-1])
    out = [x for x in preferred if x in suffixes]
    if out:
        return out
    return sorted(list(suffixes))[:8]


def _pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def train_alignment(args: argparse.Namespace) -> None:
    """Fine-tune or align the external LM sidecar to NEED latent targets."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as exc:
        raise RuntimeError("Install transformers to train a sidecar alignment adapter") from exc

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    rows = _read_jsonl(Path(args.dataset_jsonl))
    payload = _safe_torch_load(Path(args.latents_pt), map_location="cpu")
    targets = payload["need_latent"].float()
    if not rows:
        raise ValueError("empty sidecar alignment dataset")
    if targets.size(0) < len(rows):
        raise ValueError("latents_pt has fewer latent targets than dataset rows")

    dtype = torch.bfloat16 if args.dtype == "bf16" and device.type == "cuda" else torch.float16 if args.dtype == "fp16" and device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.sidecar_model, trust_remote_code=bool(args.trust_remote_code))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.sidecar_model, torch_dtype=dtype, trust_remote_code=bool(args.trust_remote_code)).to(device)
    try:
        model.config.use_cache = False
    except Exception:
        pass

    train_mode = args.train_mode
    adapter_loaded = False
    if train_mode == "lora":
        try:
            from peft import LoraConfig, get_peft_model  # type: ignore
            targets_mod = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()] or _infer_lora_targets(model)
            lcfg = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=targets_mod, task_type="CAUSAL_LM")
            model = get_peft_model(model, lcfg)
            adapter_loaded = True
        except Exception as exc:
            print(json.dumps({"warning": "PEFT/LoRA unavailable or incompatible; falling back to projection_only", "error": str(exc)}), flush=True)
            train_mode = "projection_only"
    if train_mode == "projection_only":
        for p in model.parameters():
            p.requires_grad_(False)
    elif train_mode == "full":
        for p in model.parameters():
            p.requires_grad_(True)

    hidden_size = int(getattr(model.config, "hidden_size", getattr(model.config, "n_embd", 0)))
    need_dim = int(targets.size(-1))
    if hidden_size <= 0:
        raise ValueError("Could not determine sidecar hidden size from model.config")
    proj = SidecarLatentProjection(hidden_size, need_dim, hidden_dim=args.projection_hidden_dim, normalize=True).to(device=device, dtype=torch.float32)

    ds = SidecarAlignmentDataset(rows, targets, tokenizer, args.max_length)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=ds.collate)
    params = [p for p in list(model.parameters()) + list(proj.parameters()) if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters for sidecar alignment")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    model.train(); proj.train()
    step = 0
    last: Dict[str, float] = {}
    while step < args.steps:
        for batch in dl:
            if step >= args.steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"], output_hidden_states=True, use_cache=False)
            hidden = out.hidden_states[-1]
            pooled = _pool_hidden(hidden, batch["attention_mask"])
            pred = proj(pooled.float())
            target = F.normalize(batch["latent"].float(), dim=-1)
            cosine = F.cosine_similarity(pred, target, dim=-1)
            align_loss = (1.0 - cosine).mean()
            mse_loss = F.mse_loss(pred, target)
            sim = pred @ target.t() / max(1e-4, float(args.contrastive_temperature))
            labels = torch.arange(sim.size(0), device=device)
            contrastive = F.cross_entropy(sim, labels) if sim.size(0) > 1 and args.contrastive_weight > 0 else pred.new_tensor(0.0)
            lm_loss = out.loss.float() if out.loss is not None and args.lm_weight > 0 and train_mode != "projection_only_no_lm" else pred.new_tensor(0.0)
            loss = args.align_weight * align_loss + args.mse_weight * mse_loss + args.contrastive_weight * contrastive + args.lm_weight * lm_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            if step % max(1, args.log_interval) == 0:
                last = {
                    "step": float(step),
                    "loss": float(loss.detach().cpu()),
                    "cosine": float(cosine.mean().detach().cpu()),
                    "align_loss": float(align_loss.detach().cpu()),
                    "mse_loss": float(mse_loss.detach().cpu()),
                    "contrastive_loss": float(contrastive.detach().cpu()),
                    "lm_loss": float(lm_loss.detach().cpu()),
                }
                print(json.dumps(last), flush=True)
            step += 1
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval(); proj.eval()
    sidecar_model_path = args.sidecar_model
    if train_mode == "lora" and adapter_loaded and hasattr(model, "save_pretrained"):
        model.save_pretrained(out_dir / "sidecar_adapter")
        tokenizer.save_pretrained(out_dir / "sidecar_adapter")
        adapter_path = str(out_dir / "sidecar_adapter")
    elif train_mode == "full" and hasattr(model, "save_pretrained"):
        model.save_pretrained(out_dir / "sidecar_model")
        tokenizer.save_pretrained(out_dir / "sidecar_model")
        sidecar_model_path = str(out_dir / "sidecar_model")
        adapter_path = ""
    else:
        adapter_path = ""
    proj_payload = {
        "state_dict": {k: v.detach().cpu() for k, v in proj.state_dict().items()},
        "config": {
            "format": "need_sidecar_latent_projection",
            "sidecar_model": sidecar_model_path,
            "base_sidecar_model": args.sidecar_model,
            "sidecar_hidden_size": hidden_size,
            "need_dim": need_dim,
            "projection_hidden_dim": int(args.projection_hidden_dim),
            "normalize": True,
            "train_mode": train_mode,
            "adapter_path": adapter_path,
            "dataset_jsonl": str(args.dataset_jsonl),
            "latents_pt": str(args.latents_pt),
            "metrics": last,
        },
    }
    torch.save(proj_payload, out_dir / "latent_projection.pt")
    save_json(
        {
            "format": "need_sidecar_latent_alignment_adapter",
            "sidecar_model": sidecar_model_path,
            "base_sidecar_model": args.sidecar_model,
            "train_mode": train_mode,
            "adapter_path": adapter_path,
            "latent_alignment_path": str(out_dir),
            "latent_projection": str(out_dir / "latent_projection.pt"),
            "examples": len(rows),
            "steps": int(args.steps),
            "metrics": last,
            "runtime_args": {
                "sidecar_model": sidecar_model_path,
                "sidecar_adapter_path": adapter_path,
                "sidecar_latent_alignment_path": str(out_dir),
            },
        },
        out_dir / "alignment_config.json",
    )
    print(json.dumps({"done": True, "out_dir": str(out_dir), "adapter_path": adapter_path, "latent_alignment_path": str(out_dir), "metrics": last}, indent=2), flush=True)


@torch.no_grad()
def eval_alignment(args: argparse.Namespace) -> None:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as exc:
        raise RuntimeError("Install transformers to evaluate sidecar alignment") from exc
    device = resolve_device(args.device)
    rows = _read_jsonl(Path(args.dataset_jsonl))[: args.max_examples if args.max_examples > 0 else None]
    payload = _safe_torch_load(Path(args.latents_pt), map_location="cpu")
    targets = payload["need_latent"].float()
    tokenizer = AutoTokenizer.from_pretrained(args.sidecar_model, trust_remote_code=bool(args.trust_remote_code))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.sidecar_model, trust_remote_code=bool(args.trust_remote_code)).to(device)
    if args.sidecar_adapter_path:
        try:
            from peft import PeftModel  # type: ignore
            model = PeftModel.from_pretrained(model, args.sidecar_adapter_path).to(device)
        except Exception as exc:
            print(json.dumps({"warning": "could not load sidecar adapter", "error": str(exc)}), flush=True)
    projection_payload = _safe_torch_load(Path(args.sidecar_latent_alignment_path) / "latent_projection.pt", map_location="cpu")
    pcfg = projection_payload["config"]
    proj = SidecarLatentProjection(int(pcfg["sidecar_hidden_size"]), int(pcfg["need_dim"]), int(pcfg.get("projection_hidden_dim", 0)), normalize=bool(pcfg.get("normalize", True)))
    proj.load_state_dict(projection_payload["state_dict"], strict=False)
    proj.to(device).eval(); model.eval()
    ds = SidecarAlignmentDataset(rows, targets, tokenizer, args.max_length)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=ds.collate)
    cosines: List[float] = []
    mses: List[float] = []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], output_hidden_states=True, use_cache=False)
        pred = proj(_pool_hidden(out.hidden_states[-1], batch["attention_mask"]).float())
        target = F.normalize(batch["latent"].float(), dim=-1)
        cosines.extend(F.cosine_similarity(pred, target, dim=-1).detach().cpu().tolist())
        mses.extend(((pred - target) ** 2).mean(dim=-1).detach().cpu().tolist())
    metrics = {
        "examples": len(cosines),
        "mean_cosine": sum(cosines) / max(1, len(cosines)),
        "mean_mse": sum(mses) / max(1, len(mses)),
        "min_cosine": min(cosines) if cosines else 0.0,
    }
    print(json.dumps(metrics, indent=2), flush=True)


def distill_sidecar(args: argparse.Namespace) -> None:
    """Compatibility path: build summary rows and latent tensors using NEED checks."""
    device = resolve_device(args.device)
    need = load_model(args.need_checkpoint, device=device, prefer_best=args.prefer_best)
    tok = load_tokenizer_for_dir(args.need_checkpoint)
    runtime = _sidecar_runtime(args, device)
    texts = Path(args.corpus).read_text(encoding="utf-8", errors="replace").split("\n\n")[: args.max_docs]

    rows = []
    vecs = []
    ids_out = []
    faith_out = []
    for text in texts:
        text = _clean_text(text, args.max_input_chars)
        ids = tok.encode(text, add_bos=True, add_eos=True)[: need.cfg.block_size]
        if len(ids) < 2:
            continue
        x = torch.tensor([ids], device=device)
        with torch.no_grad():
            path = need.latent_pathway(x, stride=args.vector_stride, max_vectors=args.max_vectors)
            vectors = path["pathway_vectors"]
            latent = public_summary_template(text, True)
        with sidecar_optimization_mode():
            raw, summary = runtime.generate_artificial_cot_and_summary(
                text,
                latent,
                cot_tokens=args.cot_tokens,
                summary_tokens=args.summary_tokens,
            )
        score = need.score_text_risk(
            torch.tensor([tok.encode(text + "\n" + raw, add_bos=True)[-need.cfg.block_size:]], device=device),
            conditioning_vectors=vectors,
            conditioning_scale=args.conditioning_scale,
        )
        quality = float(score.get("quality", 0.5))
        risk = float(score.get("risk", 0.5))
        contradiction = float(score.get("contradiction", 0.5))
        target = tok.encode(summary, add_bos=True, add_eos=True)[: args.max_target_tokens]
        if len(target) < args.max_target_tokens:
            target = target + [need.cfg.pad_id] * (args.max_target_tokens - len(target))
        rows.append(
            {
                "prompt_preview": text[:500],
                "summary": summary,
                "quality": quality,
                "risk": risk,
                "contradiction": contradiction,
            }
        )
        vecs.append(vectors[0].detach().cpu())
        ids_out.append(torch.tensor(target))
        faith_out.append(torch.tensor([quality, max(0.0, 1.0 - risk), contradiction]))

    if not rows:
        raise RuntimeError("no examples produced")
    max_v = max(v.size(0) for v in vecs)
    dim = vecs[0].size(-1)
    padded = []
    for v in vecs:
        if v.size(0) < max_v:
            v = torch.cat([v, torch.zeros(max_v - v.size(0), dim)], dim=0)
        padded.append(v)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"vectors": torch.stack(padded), "ids": torch.stack(ids_out), "faith": torch.stack(faith_out)}, out / "latents.pt")
    with (out / "sidecar_summaries.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"examples": len(rows), "out": str(out)}, indent=2))


def run(args: argparse.Namespace) -> None:
    """Run the sidecar summary path directly for one prompt."""
    device = resolve_device(args.device)
    need = load_model(args.need_checkpoint, device=device, prefer_best=args.prefer_best)
    tok = load_tokenizer_for_dir(args.need_checkpoint)
    runtime = _sidecar_runtime(args, device)
    ids = torch.tensor([tok.encode(args.prompt, add_bos=True)[: need.cfg.block_size]], device=device)
    with torch.no_grad():
        path = need.latent_pathway(ids, stride=args.vector_stride, max_vectors=args.max_vectors)
        latent = public_summary_template(args.prompt, True)
    with sidecar_optimization_mode():
        _raw, summary = runtime.generate_artificial_cot_and_summary(
            args.prompt,
            latent,
            cot_tokens=args.cot_tokens,
            summary_tokens=args.summary_tokens,
        )
    score = need.score_text_risk(ids, conditioning_vectors=path["pathway_vectors"], conditioning_scale=args.conditioning_scale)
    print(summary)
    print(json.dumps({"need_score": score}, indent=2))


def _add_sidecar_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sidecar_model", required=True)
    p.add_argument("--sidecar_device", default="same")
    p.add_argument("--sidecar_dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--sidecar_attn_backend", choices=["auto", "sdpa", "flash_attention_2", "eager"], default="sdpa")
    p.add_argument("--sidecar_cache_implementation", choices=["static", "dynamic", "offloaded", "none"], default="static")
    p.add_argument("--sidecar_trust_remote_code", action="store_true")
    p.add_argument("--sidecar_adapter_path", default="")
    p.add_argument("--sidecar_latent_alignment_path", default="")


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="NEED sidecar latent alignment and summary utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build_alignment_dataset", help="build NEED latent targets for sidecar adapter training")
    b.add_argument("--need_checkpoint", required=True)
    b.add_argument("--prefer_best", action="store_true")
    b.add_argument("--corpus", default="")
    b.add_argument("--interactions", default="")
    b.add_argument("--transcripts_file", default="")
    b.add_argument("--out_dir", required=True)
    b.add_argument("--device", default="auto")
    b.add_argument("--max_docs", type=int, default=0)
    b.add_argument("--max_examples", type=int, default=3000)
    b.add_argument("--max_input_chars", type=int, default=8000)
    b.add_argument("--target_summary_words", type=int, default=95)
    b.add_argument("--seed", type=int, default=123)
    b.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    b.add_argument("--vector_stride", type=int, default=4)
    b.add_argument("--max_vectors", type=int, default=128)
    b.add_argument("--conditioning_scale", type=float, default=0.18)
    b.add_argument("--bootstrap_sidecar_summaries", action="store_true")
    b.add_argument("--summary_tokens", type=int, default=160)
    _add_sidecar_args(b)

    t = sub.add_parser("train_alignment", help="train sidecar LoRA/full/projection adapter against NEED latent targets")
    t.add_argument("--sidecar_model", required=True)
    t.add_argument("--dataset_jsonl", required=True)
    t.add_argument("--latents_pt", required=True)
    t.add_argument("--out_dir", required=True)
    t.add_argument("--device", default="auto")
    t.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    t.add_argument("--trust_remote_code", action="store_true")
    t.add_argument("--train_mode", choices=["lora", "full", "projection_only"], default="lora")
    t.add_argument("--steps", type=int, default=600)
    t.add_argument("--batch_size", type=int, default=4)
    t.add_argument("--max_length", type=int, default=1024)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--weight_decay", type=float, default=0.02)
    t.add_argument("--grad_clip", type=float, default=1.0)
    t.add_argument("--seed", type=int, default=123)
    t.add_argument("--log_interval", type=int, default=20)
    t.add_argument("--projection_hidden_dim", type=int, default=0)
    t.add_argument("--align_weight", type=float, default=1.0)
    t.add_argument("--mse_weight", type=float, default=0.25)
    t.add_argument("--contrastive_weight", type=float, default=0.35)
    t.add_argument("--contrastive_temperature", type=float, default=0.07)
    t.add_argument("--lm_weight", type=float, default=0.35)
    t.add_argument("--lora_rank", type=int, default=16)
    t.add_argument("--lora_alpha", type=int, default=32)
    t.add_argument("--lora_dropout", type=float, default=0.05)
    t.add_argument("--lora_target_modules", default="")

    e = sub.add_parser("eval_alignment", help="evaluate a trained sidecar latent adapter/projection")
    e.add_argument("--sidecar_model", required=True)
    e.add_argument("--dataset_jsonl", required=True)
    e.add_argument("--latents_pt", required=True)
    e.add_argument("--sidecar_latent_alignment_path", required=True)
    e.add_argument("--sidecar_adapter_path", default="")
    e.add_argument("--device", default="auto")
    e.add_argument("--trust_remote_code", action="store_true")
    e.add_argument("--batch_size", type=int, default=4)
    e.add_argument("--max_length", type=int, default=1024)
    e.add_argument("--max_examples", type=int, default=0)

    d = sub.add_parser("distill_sidecar", help="compatibility path: build sidecar summary distillation artifacts")
    d.add_argument("--need_checkpoint", required=True)
    d.add_argument("--prefer_best", action="store_true")
    d.add_argument("--corpus", required=True)
    d.add_argument("--out_dir", required=True)
    d.add_argument("--device", default="auto")
    d.add_argument("--max_docs", type=int, default=1000)
    d.add_argument("--max_input_chars", type=int, default=8000)
    d.add_argument("--vector_stride", type=int, default=4)
    d.add_argument("--max_vectors", type=int, default=256)
    d.add_argument("--max_target_tokens", type=int, default=192)
    d.add_argument("--cot_tokens", type=int, default=220)
    d.add_argument("--summary_tokens", type=int, default=180)
    d.add_argument("--conditioning_scale", type=float, default=0.18)
    _add_sidecar_args(d)

    r = sub.add_parser("run")
    r.add_argument("--need_checkpoint", required=True)
    r.add_argument("--prefer_best", action="store_true")
    r.add_argument("--prompt", required=True)
    r.add_argument("--device", default="auto")
    r.add_argument("--vector_stride", type=int, default=4)
    r.add_argument("--max_vectors", type=int, default=256)
    r.add_argument("--cot_tokens", type=int, default=220)
    r.add_argument("--summary_tokens", type=int, default=180)
    r.add_argument("--conditioning_scale", type=float, default=0.18)
    _add_sidecar_args(r)

    args = p.parse_args(argv)
    dispatch = {
        "build_alignment_dataset": build_alignment_dataset,
        "train_alignment": train_alignment,
        "eval_alignment": eval_alignment,
        "distill_sidecar": distill_sidecar,
        "run": run,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
