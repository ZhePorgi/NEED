#!/usr/bin/env python3
"""Distill a smaller NEED sidecar from a larger NEED teacher.

The sidecar learns the teacher's surface distribution plus latent geometry:
- next-token and MTP KL against the teacher,
- ground-truth CE so it remains a usable LM,
- hidden/pathway alignment through a trainable sidecar->teacher projection,
- planner/risk alignment when those tensors are available.

The resulting checkpoint can be used by generate.py with:

  python generate.py --checkpoint MAIN_NEED --sidecar_type need \
    --need_sidecar_checkpoint SIDE_OUT --need_sidecar_projection_path SIDE_OUT

Only one sidecar is active at runtime; this script trains the smaller NEED backend.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from need_core import ByteTokenizer, NeedConfig, NeedModel, load_model, save_model, save_json, resolve_device, load_tokenizer_for_dir
from need_sidecar import NeedLatentProjection

try:
    from config_for_size import model_config_for_params, parse_scaled_number
except Exception:  # pragma: no cover
    model_config_for_params = None  # type: ignore[assignment]
    def parse_scaled_number(x: Any, default_suffix: str = "") -> int:  # type: ignore[no-redef]
        return int(float(str(x).replace("M", "000000").replace("m", "000000")))


def _safe_text(x: Any, max_chars: int = 12000) -> str:
    return " ".join(str(x or "").replace("\r\n", "\n").split())[:max_chars]


def _text_from_obj(obj: Dict[str, Any]) -> str:
    if "messages" in obj and isinstance(obj["messages"], list):
        parts: List[str] = []
        for m in obj["messages"]:
            if isinstance(m, dict):
                role = _safe_text(m.get("role", "user"), 32)
                content = _safe_text(m.get("content", ""), 6000)
                if content:
                    parts.append(f"{role}: {content}")
        return "\n".join(parts)
    if "prompt" in obj or "task" in obj:
        prompt = _safe_text(obj.get("prompt", obj.get("task", "")), 6000)
        answer = _safe_text(obj.get("answer", obj.get("response", obj.get("chosen", obj.get("completion", "")))), 6000)
        if answer:
            return prompt + "\nAssistant: " + answer
        return prompt
    for k in ("text", "content", "article", "input_text", "question", "chosen", "rejected"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return _safe_text(v, 12000)
    return ""


def iter_texts(paths: Sequence[str], max_docs: int = 0, max_chars: int = 12000) -> List[str]:
    out: List[str] = []
    for raw in paths:
        if not raw:
            continue
        p = Path(raw)
        files = []
        if p.is_dir():
            for pat in ("*.jsonl", "*.json", "*.txt", "*.text"):
                files.extend(sorted(p.rglob(pat)))
        else:
            files = [p]
        for f in files:
            if not f.exists():
                continue
            if f.suffix.lower() == ".jsonl":
                for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        text = _text_from_obj(obj) if isinstance(obj, dict) else _safe_text(obj, max_chars)
                    except Exception:
                        text = line
                    if text:
                        out.append(_safe_text(text, max_chars))
                    if max_docs and len(out) >= max_docs:
                        return out
            elif f.suffix.lower() == ".json":
                data = json.loads(f.read_text(encoding="utf-8", errors="replace"))
                rows = data.get("rows", data.get("data", [data])) if isinstance(data, dict) else data
                if isinstance(rows, list):
                    for row in rows:
                        text = _text_from_obj(row) if isinstance(row, dict) else _safe_text(row, max_chars)
                        if text:
                            out.append(_safe_text(text, max_chars))
                        if max_docs and len(out) >= max_docs:
                            return out
            else:
                raw_text = f.read_text(encoding="utf-8", errors="replace")
                chunks = [c.strip() for c in raw_text.split("\n\n") if c.strip()]
                for c in chunks or [raw_text]:
                    out.append(_safe_text(c, max_chars))
                    if max_docs and len(out) >= max_docs:
                        return out
    # de-duplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for t in out:
        key = t[:800]
        if key not in seen:
            seen.add(key)
            deduped.append(t)
    return deduped


class SidecarDistillDataset(Dataset):
    def __init__(self, texts: Sequence[str], tokenizer: ByteTokenizer, block_size: int, predict_heads: int, samples: int, seed: int = 123) -> None:
        self.tok = tokenizer
        self.block_size = int(block_size)
        self.predict_heads = max(1, int(predict_heads))
        self.samples = max(1, int(samples))
        self.seed = int(seed)
        ids: List[int] = []
        for text in texts:
            enc = self.tok.encode(text, add_bos=True, add_eos=True)
            if enc:
                ids.extend(enc + [self.tok.eos_id])
        if not ids:
            raise ValueError("no tokens for sidecar distillation")
        need = self.block_size + self.predict_heads
        if len(ids) < need:
            reps = int(math.ceil(need / max(1, len(ids))))
            ids = ids * reps
        self.ids = torch.tensor(ids, dtype=torch.long)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(self.seed + int(idx))
        max_start = max(0, int(self.ids.numel()) - self.block_size - self.predict_heads)
        start = rng.randint(0, max_start) if max_start > 0 else 0
        x = self.ids[start:start + self.block_size].clone()
        targets = []
        for h in range(1, self.predict_heads + 1):
            targets.append(self.ids[start + h:start + h + self.block_size].clone())
        y = torch.stack(targets, dim=-1)
        return {"input_ids": x, "targets": y}


def make_sidecar_config(args: argparse.Namespace, teacher_cfg: NeedConfig) -> NeedConfig:
    if args.student_config:
        data = json.loads(Path(args.student_config).read_text(encoding="utf-8"))
        raw = data.get("config", data)
        cfg = NeedConfig.from_dict(raw)
        cfg.validate()
        return cfg
    if args.target_params and model_config_for_params is not None:
        cfg_dict, _count = model_config_for_params(parse_scaled_number(args.target_params), architecture=args.student_architecture, block_size=args.block_size or min(teacher_cfg.block_size, 512))
        base = asdict(teacher_cfg)
        base.update(cfg_dict)
        base["block_size"] = int(args.block_size or cfg_dict.get("block_size", min(teacher_cfg.block_size, 512)))
        base["n_predict_heads"] = int(args.n_predict_heads or teacher_cfg.n_predict_heads)
        base["image_codebook_size"] = int(teacher_cfg.image_codebook_size)
        base["vocab_size"] = int(teacher_cfg.vocab_size)
        base["text_vocab_size"] = int(teacher_cfg.text_vocab_size)
        base["image_token_offset"] = int(teacher_cfg.image_token_offset)
        cfg = NeedConfig.from_dict(base)
        cfg.validate()
        return cfg
    scale = float(args.student_scale)
    def mult(v: int, m: int = 8, lo: int = 64) -> int:
        return max(lo, int(round(v * scale / m) * m))
    d_model = int(args.d_model or mult(teacher_cfg.d_model, 8, 64))
    n_heads = int(args.n_heads or max(1, min(d_model, teacher_cfg.n_heads, max(1, d_model // 32))))
    while d_model % n_heads != 0 and n_heads > 1:
        n_heads -= 1
    base = asdict(teacher_cfg)
    base.update({
        "d_model": d_model,
        "n_layers": int(args.n_layers or max(2, round(teacher_cfg.n_layers * scale))),
        "n_heads": n_heads,
        "d_ff": int(args.d_ff or max(d_model * 2, round((teacher_cfg.d_ff or teacher_cfg.d_model * 4) * scale / 64) * 64)),
        "block_size": int(args.block_size or min(teacher_cfg.block_size, 512)),
        "n_experts": int(args.n_experts),
        "moe_top_k": int(min(args.moe_top_k, max(1, args.n_experts))),
        "moe_use_shared_expert": bool(args.n_experts > 1),
        "energy_rank": max(16, min(d_model, int(round(teacher_cfg.energy_rank * scale / 8) * 8))),
        "memory_slots": max(4, int(round(teacher_cfg.memory_slots * scale))),
        "memory_rank": max(16, min(d_model, int(round(teacher_cfg.memory_rank * scale / 8) * 8))),
        "pathway_memory_slots": max(4, int(round(teacher_cfg.pathway_memory_slots * scale))),
        "n_predict_heads": int(args.n_predict_heads or teacher_cfg.n_predict_heads),
    })
    cfg = NeedConfig.from_dict(base)
    cfg.validate()
    return cfg


def kl_divergence_student_teacher(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    vocab_limit: int,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    n = min(int(student_logits.size(1)), int(teacher_logits.size(1))) if student_logits.ndim >= 3 and teacher_logits.ndim >= 3 else 0
    if n <= 0:
        return student_logits.new_tensor(0.0)
    temp = max(float(temperature), 1e-6)
    limit = int(max(1, min(vocab_limit, student_logits.size(-1), teacher_logits.size(-1))))
    s = student_logits[:, -n:, :limit].float() / temp
    t = teacher_logits[:, -n:, :limit].float() / temp
    # Mean over valid non-padding tokens, not just batch examples.  PyTorch's
    # batchmean on a [B,T,V] tensor divides only by B, and unmasked padding tokens
    # can dominate short-document distillation batches.
    per_tok = F.kl_div(F.log_softmax(s, dim=-1), F.softmax(t, dim=-1), reduction="none").sum(dim=-1)
    if mask is not None:
        m = mask[:, -n:].to(device=per_tok.device, dtype=per_tok.dtype)
        return (per_tok * m).sum() / m.sum().clamp_min(1.0) * (temp ** 2)
    return per_tok.mean() * (temp ** 2)


def projected_alignment(student_h: torch.Tensor, teacher_h: torch.Tensor, projection: nn.Module) -> torch.Tensor:
    sh = projection(student_h.float())
    th = teacher_h.float().detach()
    if sh.size(1) != th.size(1):
        n = min(sh.size(1), th.size(1))
        sh = sh[:, -n:]
        th = th[:, -n:]
    mse = F.mse_loss(sh, th)
    cos = 1.0 - F.cosine_similarity(sh, th, dim=-1).mean()
    pooled = 1.0 - F.cosine_similarity(sh.mean(dim=1), th.mean(dim=1), dim=-1).mean()
    return mse + 0.5 * cos + 0.5 * pooled


def projected_linear_cka_loss(student_h: torch.Tensor, teacher_h: torch.Tensor, projection: nn.Module) -> torch.Tensor:
    """Dimension-tolerant CKA-style representation matching.

    The projection handles different hidden sizes; CKA then matches relational
    geometry rather than forcing every projected coordinate to copy the teacher.
    This is useful for much smaller NEED sidecars.
    """
    sh = projection(student_h.float())
    th = teacher_h.float().detach()
    if sh.size(1) != th.size(1):
        n = min(sh.size(1), th.size(1))
        sh = sh[:, -n:]
        th = th[:, -n:]
    x = sh.reshape(-1, sh.size(-1))
    y = th.reshape(-1, th.size(-1))
    valid = torch.isfinite(x).all(dim=-1) & torch.isfinite(y).all(dim=-1)
    if not bool(valid.any()):
        return sh.new_tensor(0.0)
    x = x[valid]
    y = y[valid]
    if x.size(0) < 2:
        return sh.new_tensor(0.0)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xy = x.t().matmul(y)
    xx = x.t().matmul(x)
    yy = y.t().matmul(y)
    num = xy.pow(2).sum()
    den = (xx.pow(2).sum().sqrt() * yy.pow(2).sum().sqrt()).clamp_min(1e-8)
    return 1.0 - (num / den).clamp(0.0, 1.0)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = resolve_device(args.device)
    teacher = load_model(args.teacher_checkpoint, device=device, prefer_best=args.teacher_prefer_best, kernel_backend=args.kernel_backend)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    tok = load_tokenizer_for_dir(args.teacher_checkpoint)
    texts = iter_texts(args.corpus, max_docs=args.max_docs, max_chars=args.max_input_chars)
    if not texts:
        raise ValueError("No corpus texts found. Pass --corpus file/dir/jsonl.")
    if args.shuffle:
        random.shuffle(texts)
    student_cfg = make_sidecar_config(args, teacher.cfg)
    student = load_model(args.student_checkpoint, device=device, prefer_best=args.student_prefer_best, kernel_backend=args.kernel_backend) if args.student_checkpoint else NeedModel(student_cfg).to(device)
    student.train()
    projection_hidden = int(args.projection_hidden_dim)
    projection = NeedLatentProjection(int(student.cfg.d_model), int(teacher.cfg.d_model), hidden_dim=projection_hidden).to(device)
    ds = SidecarDistillDataset(texts, tok, block_size=min(int(student.cfg.block_size), int(teacher.cfg.block_size), int(args.block_size or student.cfg.block_size)), predict_heads=min(int(student.cfg.n_predict_heads), int(teacher.cfg.n_predict_heads)), samples=args.samples, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    params = list(student.parameters()) + list(projection.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    vocab_limit = int(args.text_only_vocab_limit or teacher.cfg.image_token_offset or teacher.cfg.vocab_size)
    step = 0
    last: Dict[str, float] = {}
    while step < args.steps:
        for batch in dl:
            step += 1
            x = batch["input_ids"].to(device)
            y = batch["targets"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                t_logits, _t_loss, t_aux = teacher(x[:, -teacher.cfg.block_size:], targets=y[:, -teacher.cfg.block_size:, :], return_hidden=True)
                t_h = t_aux.get("_hidden")
                t_mtp = teacher.mtp_logits_from_hidden(t_h) if t_h is not None else [t_logits]
            s_logits, s_loss, s_aux = student(x[:, -student.cfg.block_size:], targets=y[:, -student.cfg.block_size:, :], return_hidden=True)
            s_h = s_aux.get("_hidden")
            loss = torch.zeros((), device=device)
            if s_loss is not None:
                ce = s_loss
            else:
                target0 = y[..., 0]
                ce_per = F.cross_entropy(s_logits.reshape(-1, s_logits.size(-1)), target0.reshape(-1), ignore_index=student.cfg.pad_id, reduction="none").view_as(target0)
                valid0 = target0 != student.cfg.pad_id
                ce = (ce_per * valid0.float()).sum() / valid0.float().sum().clamp_min(1.0)
            loss = loss + float(args.ce_weight) * ce
            valid0 = y[:, -s_logits.size(1):, 0] != student.cfg.pad_id
            logit_kl = kl_divergence_student_teacher(s_logits, t_logits, args.temperature, vocab_limit, mask=valid0)
            loss = loss + float(args.logit_kl_weight) * logit_kl
            mtp_kl = torch.zeros((), device=device)
            if s_h is not None and t_h is not None and float(args.mtp_kl_weight) > 0:
                s_mtp = student.mtp_logits_from_hidden(s_h)
                n_heads = min(len(s_mtp), len(t_mtp))
                parts = []
                for i in range(1, n_heads):
                    head_mask = y[:, -s_mtp[i].size(1):, min(i, y.size(-1) - 1)] != student.cfg.pad_id
                    parts.append(kl_divergence_student_teacher(s_mtp[i], t_mtp[i], args.temperature, vocab_limit, mask=head_mask))
                if parts:
                    mtp_kl = torch.stack(parts).mean()
                    loss = loss + float(args.mtp_kl_weight) * mtp_kl
            hidden_loss = torch.zeros((), device=device)
            hidden_cka = torch.zeros((), device=device)
            if s_h is not None and t_h is not None and float(args.hidden_weight) > 0:
                hidden_loss = projected_alignment(s_h, t_h[:, -s_h.size(1):], projection)
                loss = loss + float(args.hidden_weight) * hidden_loss
            if s_h is not None and t_h is not None and float(args.cka_hidden_weight) > 0:
                hidden_cka = projected_linear_cka_loss(s_h, t_h[:, -s_h.size(1):], projection)
                loss = loss + float(args.cka_hidden_weight) * hidden_cka
            plan_loss = torch.zeros((), device=device)
            plan_cka = torch.zeros((), device=device)
            if float(args.future_state_weight) > 0 and s_h is not None and t_h is not None:
                # Teach the smaller model to follow the teacher's planned next latent direction.
                s_plan = s_aux.get("_planned_next")
                t_plan = t_aux.get("_planned_next")
                if torch.is_tensor(s_plan) and torch.is_tensor(t_plan):
                    plan_loss = projected_alignment(s_plan, t_plan[:, -s_plan.size(1):], projection)
                    loss = loss + float(args.future_state_weight) * plan_loss
                    if float(args.cka_future_state_weight) > 0:
                        plan_cka = projected_linear_cka_loss(s_plan, t_plan[:, -s_plan.size(1):], projection)
                        loss = loss + float(args.cka_future_state_weight) * plan_cka
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            last = {
                "step": float(step),
                "loss": float(loss.detach().cpu()),
                "ce": float(ce.detach().cpu()),
                "logit_kl": float(logit_kl.detach().cpu()),
                "mtp_kl": float(mtp_kl.detach().cpu()),
                "hidden": float(hidden_loss.detach().cpu()),
                "hidden_cka": float(hidden_cka.detach().cpu()),
                "future_state": float(plan_loss.detach().cpu()),
                "future_state_cka": float(plan_cka.detach().cpu()),
            }
            if step % max(1, args.log_interval) == 0:
                print(json.dumps(last, sort_keys=True))
            if step >= args.steps:
                break
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    student.eval()
    save_model(student, out, metrics=last, name="model")
    proj_payload = {
        "state_dict": projection.state_dict(),
        "in_dim": int(student.cfg.d_model),
        "out_dim": int(teacher.cfg.d_model),
        "hidden_dim": int(projection_hidden),
        "format": "need_sidecar_projection",
    }
    torch.save(proj_payload, out / "need_sidecar_projection.pt")
    save_json({
        "sidecar_type": "need",
        "teacher_checkpoint": str(args.teacher_checkpoint),
        "student_checkpoint": str(out),
        "projection_path": str(out / "need_sidecar_projection.pt"),
        "metrics": last,
        "student_config": asdict(student.cfg),
        "distillation": {
            "ce_weight": args.ce_weight,
            "logit_kl_weight": args.logit_kl_weight,
            "mtp_kl_weight": args.mtp_kl_weight,
            "hidden_weight": args.hidden_weight,
            "cka_hidden_weight": args.cka_hidden_weight,
            "future_state_weight": args.future_state_weight,
            "cka_future_state_weight": args.cka_future_state_weight,
            "temperature": args.temperature,
        },
        "usage": "generate.py --checkpoint MAIN_NEED --sidecar_type need --need_sidecar_checkpoint %s --need_sidecar_projection_path %s" % (out, out),
    }, out / "need_sidecar_manifest.json")
    print(json.dumps({"out_dir": str(out), "projection": str(out / "need_sidecar_projection.pt"), "metrics": last}, indent=2))


@torch.no_grad()
def smoke(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    rt_args = argparse.Namespace(
        sidecar_type="need",
        need_sidecar_checkpoint=args.sidecar_checkpoint,
        need_sidecar_projection_path=args.projection_path or args.sidecar_checkpoint,
        need_sidecar_projection_weight=1.0,
        need_sidecar_decode_mode="nonseq",
        need_sidecar_prefer_best=False,
        need_sidecar_max_context_tokens=256,
        sidecar_model="",
        kernel_backend=args.kernel_backend,
        sidecar_latent_alignment_path="",
        sidecar_latent_alignment_weight=1.0,
    )
    main = load_model(args.teacher_checkpoint, device=device, prefer_best=args.teacher_prefer_best, kernel_backend=args.kernel_backend)
    from need_sidecar import make_single_sidecar_runtime
    rt = make_single_sidecar_runtime(rt_args, device, main)
    summary, vectors, metrics = rt.latent_guidance(args.prompt)
    print(json.dumps({"summary_chars": len(summary), "vectors": list(vectors.shape) if torch.is_tensor(vectors) else None, "metrics": metrics}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train or test a smaller NEED sidecar from a larger NEED teacher")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="distill a smaller NEED sidecar")
    t.add_argument("--teacher_checkpoint", required=True)
    t.add_argument("--teacher_prefer_best", action="store_true")
    t.add_argument("--student_checkpoint", default="")
    t.add_argument("--student_prefer_best", action="store_true")
    t.add_argument("--student_config", default="")
    t.add_argument("--target_params", default="", help="Optional size such as 30M; uses config_for_size when available")
    t.add_argument("--student_architecture", choices=["dense", "moe"], default="dense")
    t.add_argument("--student_scale", type=float, default=0.5)
    t.add_argument("--d_model", type=int, default=0)
    t.add_argument("--n_layers", type=int, default=0)
    t.add_argument("--n_heads", type=int, default=0)
    t.add_argument("--d_ff", type=int, default=0)
    t.add_argument("--n_experts", type=int, default=1)
    t.add_argument("--moe_top_k", type=int, default=1)
    t.add_argument("--n_predict_heads", type=int, default=0)
    t.add_argument("--block_size", type=int, default=0)
    t.add_argument("--corpus", action="append", default=[], help="Corpus file/dir/jsonl. Can be repeated. Latent-RL traces are accepted as JSONL rows.")
    t.add_argument("--out_dir", required=True)
    t.add_argument("--device", default="auto")
    t.add_argument("--kernel_backend", choices=["auto", "torch", "triton"], default="auto")
    t.add_argument("--max_docs", type=int, default=0)
    t.add_argument("--max_input_chars", type=int, default=12000)
    t.add_argument("--samples", type=int, default=2000)
    t.add_argument("--steps", type=int, default=600)
    t.add_argument("--batch_size", type=int, default=4)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--weight_decay", type=float, default=0.02)
    t.add_argument("--grad_clip", type=float, default=1.0)
    t.add_argument("--temperature", type=float, default=2.0)
    t.add_argument("--ce_weight", type=float, default=0.35)
    t.add_argument("--logit_kl_weight", type=float, default=1.0)
    t.add_argument("--mtp_kl_weight", type=float, default=0.45)
    t.add_argument("--hidden_weight", type=float, default=0.65)
    t.add_argument("--cka_hidden_weight", type=float, default=0.15, help="CKA-style projected representation matching for smaller sidecars")
    t.add_argument("--future_state_weight", type=float, default=0.25)
    t.add_argument("--cka_future_state_weight", type=float, default=0.05, help="CKA-style matching on planned future states")
    t.add_argument("--projection_hidden_dim", type=int, default=0)
    t.add_argument("--text_only_vocab_limit", type=int, default=0)
    t.add_argument("--seed", type=int, default=123)
    t.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    t.add_argument("--log_interval", type=int, default=20)
    s = sub.add_parser("smoke", help="load a NEED sidecar and emit projected latent guidance")
    s.add_argument("--teacher_checkpoint", required=True)
    s.add_argument("--teacher_prefer_best", action="store_true")
    s.add_argument("--sidecar_checkpoint", required=True)
    s.add_argument("--projection_path", default="")
    s.add_argument("--prompt", default="Explain the role of a smaller NEED sidecar.")
    s.add_argument("--device", default="auto")
    s.add_argument("--kernel_backend", choices=["auto", "torch", "triton"], default="auto")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "train":
        train(args)
    else:
        smoke(args)


if __name__ == "__main__":
    main()
