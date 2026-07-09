#!/usr/bin/env python3
"""DVSD calibration and efficiency suite for NEED checkpoints.

This script compares AR and Dynamic Virtual Slot Decoding on a small prompt set,
then reports the metrics that matter for tuning: committed tokens per expensive
model pass, active slot count, head-count-one collapse rate, router use, slot
confidence/entropy, repetition/artifact rates, and aux_score risk deltas.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from need_core import ByteTokenizer, load_tokenizer_for_dir, load_model, resolve_device

DEFAULT_PROMPTS = [
    "Write a short explanation of why efficient generation matters for language models.",
    "Continue this story in a calm, concrete style: The engineer opened the terminal and noticed",
    "Return a valid JSON object with keys name, purpose, risks, and next_steps for a model decoder.",
    "Solve step by step but answer briefly: if a process emits 4 tokens per trunk pass instead of 1, what changes?",
    "Write a Python function that checks whether a string is a palindrome.",
]


def _read_prompts(path: str) -> List[str]:
    if not path:
        return list(DEFAULT_PROMPTS)
    p = Path(path)
    rows: List[str] = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                obj = json.loads(line)
                for key in ("prompt", "text", "instruction"):
                    if isinstance(obj.get(key), str) and obj[key].strip():
                        rows.append(obj[key].strip())
                        break
                continue
            except Exception:
                pass
        rows.append(line)
    return rows or list(DEFAULT_PROMPTS)


def _artifact_metrics(text: str) -> Dict[str, float]:
    toks = text.split()
    rep = 0
    for a, b in zip(toks, toks[1:]):
        if a == b:
            rep += 1
    replacement = text.count("\ufffd")
    return {
        "chars": float(len(text)),
        "words": float(len(toks)),
        "replacement_chars": float(replacement),
        "adjacent_repeat_rate": float(rep / max(1, len(toks) - 1)),
    }


def _score(model, tok: ByteTokenizer, prompt: str, completion: str, device: torch.device) -> Dict[str, float]:
    try:
        ids = tok.encode(prompt + completion, add_bos=True)[-int(model.cfg.block_size):]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        return {k: float(v) for k, v in model.score_text_risk(x).items() if isinstance(v, (int, float))}
    except Exception as exc:
        return {"score_error": str(exc)}  # type: ignore[return-value]


def _run_ar(model, tok: ByteTokenizer, ids: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, Dict[str, float]]:
    t0 = time.perf_counter()
    out = model.generate_text(
        ids,
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        top_p=float(args.top_p),
        repetition_penalty=float(args.repetition_penalty),
        no_repeat_ngram=int(args.no_repeat_ngram),
    )
    dt = time.perf_counter() - t0
    new_toks = max(0, int(out.size(1) - ids.size(1)))
    return out, {
        "decode_mode": "ar",
        "seconds": float(dt),
        "new_tokens": float(new_toks),
        "tokens_per_sec": float(new_toks / max(dt, 1e-9)),
        "expensive_passes_est": float(new_toks),
        "tokens_per_expensive_pass_est": 1.0 if new_toks > 0 else 0.0,
    }


def _run_dvsd(model, tok: ByteTokenizer, ids: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, Dict[str, float]]:
    t0 = time.perf_counter()
    out, stats = model.generate_text_nonsequential(
        ids,
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        top_p=float(args.top_p),
        repetition_penalty=float(args.repetition_penalty),
        no_repeat_ngram=int(args.no_repeat_ngram),
        nonseq_min_heads=int(args.nonseq_min_heads),
        nonseq_max_heads=None if int(args.nonseq_max_heads) <= 0 else int(args.nonseq_max_heads),
        nonseq_dynamic=bool(args.nonseq_dynamic),
        nonseq_refine_steps=int(args.nonseq_refine_steps),
        nonseq_refine_causal_blend=float(args.nonseq_refine_causal_blend),
        nonseq_refine_temperature_decay=float(args.nonseq_refine_temperature_decay),
        nonseq_refine_lock_schedule=str(args.nonseq_refine_lock_schedule),
        return_stats=True,
    )
    dt = time.perf_counter() - t0
    new_toks = max(0, int(out.size(1) - ids.size(1)))
    expensive = float(stats.get("nonseq_steps", 0.0) + stats.get("nonseq_refine_forward_calls", 0.0))
    row = {"decode_mode": "dvsd", "seconds": float(dt), "new_tokens": float(new_toks), "tokens_per_sec": float(new_toks / max(dt, 1e-9))}
    row.update({str(k): float(v) for k, v in stats.items() if isinstance(v, (int, float)) and math.isfinite(float(v))})
    row["expensive_passes_est"] = expensive
    row["tokens_per_expensive_pass_est"] = float(new_toks / max(1.0, expensive))
    row["head1_step_rate"] = float(stats.get("nonseq_head1_steps", 0.0) / max(1.0, stats.get("nonseq_steps", 0.0)))
    row["refine_calls_per_token"] = float(stats.get("nonseq_refine_forward_calls", 0.0) / max(1.0, new_toks))
    return out, row


def _mean(rows: List[Dict[str, Any]], key: str) -> float:
    vals = []
    for r in rows:
        v = r.get(key)
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            vals.append(float(v))
    return float(sum(vals) / max(1, len(vals)))


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Calibrate NEED DVSD efficiency and router behavior")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--prefer_best", action="store_true")
    p.add_argument("--prompt_file", default="", help="Plain text/JSONL prompts; defaults to a small mixed suite")
    p.add_argument("--out_jsonl", default="", help="Optional per-run JSONL path")
    p.add_argument("--out_summary_json", default="", help="Optional aggregate summary JSON path")
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--iters", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--no_repeat_ngram", type=int, default=0)
    p.add_argument("--nonseq_min_heads", type=int, default=1)
    p.add_argument("--nonseq_max_heads", type=int, default=0)
    p.add_argument("--nonseq_dynamic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--nonseq_refine_steps", type=int, default=3)
    p.add_argument("--nonseq_refine_causal_blend", type=float, default=0.55)
    p.add_argument("--nonseq_refine_temperature_decay", type=float, default=0.82)
    p.add_argument("--nonseq_refine_lock_schedule", choices=["cosine", "linear", "quadratic"], default="cosine")
    p.add_argument("--disable_dvsd_router", action="store_true")
    p.add_argument("--dvsd_router_inference_mix", type=float, default=None)
    p.add_argument("--dvsd_router_min_confidence", type=float, default=None)
    p.add_argument("--disable_dvsd_planner_compound", action="store_true")
    p.add_argument("--dvsd_planner_compound_mix", type=float, default=None)
    p.add_argument("--dvsd_planner_compound_top_k", type=int, default=None)
    p.add_argument("--disable_planner_block_space", action="store_true")
    p.add_argument("--planner_block_space_mix", type=float, default=None)
    args = p.parse_args(argv)

    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device=device, prefer_best=args.prefer_best).eval()
    if args.disable_dvsd_router and hasattr(model.cfg, "dvsd_router_enabled"):
        model.cfg.dvsd_router_enabled = False
    if args.dvsd_router_inference_mix is not None and hasattr(model.cfg, "dvsd_router_inference_mix"):
        model.cfg.dvsd_router_inference_mix = float(args.dvsd_router_inference_mix)
    if args.dvsd_router_min_confidence is not None and hasattr(model.cfg, "dvsd_router_min_confidence"):
        model.cfg.dvsd_router_min_confidence = float(args.dvsd_router_min_confidence)
    if args.disable_dvsd_planner_compound and hasattr(model.cfg, "dvsd_planner_compound_enabled"):
        model.cfg.dvsd_planner_compound_enabled = False
    if args.dvsd_planner_compound_mix is not None and hasattr(model.cfg, "dvsd_planner_compound_mix"):
        model.cfg.dvsd_planner_compound_mix = float(args.dvsd_planner_compound_mix)
    if args.dvsd_planner_compound_top_k is not None and hasattr(model.cfg, "dvsd_planner_compound_top_k"):
        model.cfg.dvsd_planner_compound_top_k = int(args.dvsd_planner_compound_top_k)
    if args.disable_planner_block_space and hasattr(model.cfg, "planner_block_space_enabled"):
        model.cfg.planner_block_space_enabled = False
    if args.planner_block_space_mix is not None and hasattr(model.cfg, "planner_block_space_mix"):
        model.cfg.planner_block_space_mix = float(args.planner_block_space_mix)

    tok = load_tokenizer_for_dir(args.checkpoint)
    prompts = _read_prompts(args.prompt_file)
    rows: List[Dict[str, Any]] = []
    out_path = Path(args.out_jsonl) if args.out_jsonl else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")

    with torch.no_grad():
        for prompt_i, prompt in enumerate(prompts):
            ids = torch.tensor([tok.encode(prompt, add_bos=True)[-int(model.cfg.block_size):]], dtype=torch.long, device=device)
            for iter_i in range(max(1, int(args.iters))):
                for mode, runner in (("ar", _run_ar), ("dvsd", _run_dvsd)):
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    out, row = runner(model, tok, ids, args)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    completion = tok.decode(out[0, ids.size(1):].tolist())
                    row.update({
                        "prompt_index": prompt_i,
                        "iter": iter_i,
                        "prompt": prompt[:500],
                        "completion_preview": completion[:500],
                        "dvsd_router_loaded": bool(getattr(model, "_dvsd_router_loaded", True)),
                        "dvsd_router_enabled": bool(getattr(model.cfg, "dvsd_router_enabled", False)),
                    })
                    row.update({"artifact_" + k: v for k, v in _artifact_metrics(completion).items()})
                    score = _score(model, tok, prompt, completion, device)
                    row.update({"score_" + str(k): v for k, v in score.items()})
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)
                    if out_path:
                        with out_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(row, sort_keys=True) + "\n")

    ar_rows = [r for r in rows if r.get("decode_mode") == "ar"]
    dvsd_rows = [r for r in rows if r.get("decode_mode") == "dvsd"]
    summary = {
        "prompts": len(prompts),
        "iters": int(args.iters),
        "ar_tokens_per_sec": _mean(ar_rows, "tokens_per_sec"),
        "dvsd_tokens_per_sec": _mean(dvsd_rows, "tokens_per_sec"),
        "speedup_vs_ar": _mean(dvsd_rows, "tokens_per_sec") / max(1e-9, _mean(ar_rows, "tokens_per_sec")),
        "dvsd_tokens_per_expensive_pass": _mean(dvsd_rows, "tokens_per_expensive_pass_est"),
        "dvsd_avg_active_heads": _mean(dvsd_rows, "nonseq_avg_active_heads"),
        "dvsd_head1_step_rate": _mean(dvsd_rows, "head1_step_rate"),
        "dvsd_router_used_rate": _mean(dvsd_rows, "dvsd_router_used"),
        "dvsd_compound_enabled_rate": _mean(dvsd_rows, "nonseq_compound_enabled"),
        "dvsd_compound_steps": _mean(dvsd_rows, "nonseq_compound_steps"),
        "dvsd_compound_blend_rate": _mean(dvsd_rows, "nonseq_compound_blend_rate"),
        "dvsd_compound_descent_norm": _mean(dvsd_rows, "nonseq_avg_compound_descent_norm"),
        "dvsd_slot_confidence": _mean(dvsd_rows, "nonseq_avg_slot_confidence"),
        "dvsd_slot_entropy": _mean(dvsd_rows, "nonseq_avg_slot_entropy"),
        "ar_score_risk": _mean(ar_rows, "score_risk"),
        "dvsd_score_risk": _mean(dvsd_rows, "score_risk"),
    }
    print("<dvsd_calibration_summary>")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("</dvsd_calibration_summary>")
    if args.out_summary_json:
        sp = Path(args.out_summary_json)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
