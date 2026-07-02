#!/usr/bin/env python3
"""Generation-time benchmark for NEED checkpoints."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Sequence

import torch

from need_core import ByteTokenizer, load_model, resolve_device


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Benchmark NEED generation speed and memory")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--prompt", default="The most important idea is")
    p.add_argument("--prompt_file", default="")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--decode_mode", choices=["ar", "nonseq"], default="ar")
    p.add_argument("--nonseq_decode_style", choices=["slot_refine", "slots", "virtual_slots"], default="slot_refine")
    p.add_argument("--nonseq_min_heads", type=int, default=1)
    p.add_argument("--nonseq_max_heads", type=int, default=0, help="0 uses checkpoint/config default")
    p.add_argument("--nonseq_dynamic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--nonseq_tree_candidates", type=int, default=4, help="Legacy aux_scored-draft knob; ignored by the default virtual-slot decoder")
    p.add_argument("--nonseq_branch_top_k", type=int, default=2, help="Legacy aux_scored-draft knob; ignored by the default virtual-slot decoder")
    p.add_argument("--nonseq_refine_steps", type=int, default=3)
    p.add_argument("--nonseq_refine_causal_blend", type=float, default=0.55)
    p.add_argument("--nonseq_refine_temperature_decay", type=float, default=0.82)
    p.add_argument("--nonseq_refine_lock_schedule", choices=["cosine", "linear", "quadratic"], default="cosine")
    p.add_argument("--disable_dvsd_router", action="store_true")
    p.add_argument("--dvsd_router_inference_mix", type=float, default=None)
    p.add_argument("--dvsd_router_min_confidence", type=float, default=None)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--prefer_best", action="store_true")
    p.add_argument("--compile", action="store_true")
    args = p.parse_args(argv)
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device=device, prefer_best=args.prefer_best).eval()
    if args.disable_dvsd_router and hasattr(model.cfg, "dvsd_router_enabled"):
        model.cfg.dvsd_router_enabled = False
    if args.dvsd_router_inference_mix is not None and hasattr(model.cfg, "dvsd_router_inference_mix"):
        model.cfg.dvsd_router_inference_mix = float(args.dvsd_router_inference_mix)
    if args.dvsd_router_min_confidence is not None and hasattr(model.cfg, "dvsd_router_min_confidence"):
        model.cfg.dvsd_router_min_confidence = float(args.dvsd_router_min_confidence)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead", dynamic=False)  # type: ignore[assignment]
    tok = ByteTokenizer()
    prompts = [args.prompt]
    if args.prompt_file:
        prompts = [line.strip() for line in Path(args.prompt_file).read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    rows = []
    for prompt in prompts:
        ids = torch.tensor([tok.encode(prompt, add_bos=True)[-model.cfg.block_size:]], dtype=torch.long, device=device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        t0 = time.time()
        total = 0
        nonseq_acc = 0.0
        nonseq_drafted = 0.0
        nonseq_fallback = 0.0
        nonseq_committed = 0.0
        nonseq_refine_calls = 0.0
        nonseq_steps = 0.0
        nonseq_router_used = 0.0
        nonseq_active_heads = 0.0
        for _ in range(max(1, args.iters)):
            if args.decode_mode == "nonseq":
                out, stats = model.generate_text_nonsequential(
                    ids,
                    max_new_tokens=args.max_new_tokens,
                    nonseq_min_heads=args.nonseq_min_heads,
                    nonseq_max_heads=None if int(args.nonseq_max_heads) <= 0 else args.nonseq_max_heads,
                    nonseq_dynamic=args.nonseq_dynamic,
                    nonseq_tree_candidates=args.nonseq_tree_candidates,
                    nonseq_branch_top_k=args.nonseq_branch_top_k,
                    nonseq_decode_style=args.nonseq_decode_style,
                    nonseq_refine_steps=args.nonseq_refine_steps,
                    nonseq_refine_causal_blend=args.nonseq_refine_causal_blend,
                    nonseq_refine_temperature_decay=args.nonseq_refine_temperature_decay,
                    nonseq_refine_lock_schedule=args.nonseq_refine_lock_schedule,
                    return_stats=True,
                )
                nonseq_acc += float(stats.get("nonseq_accepted_tokens", 0.0))
                nonseq_drafted += float(stats.get("nonseq_drafted_tokens", 0.0))
                nonseq_fallback += float(stats.get("nonseq_ar_fallback_tokens", 0.0))
                nonseq_committed += float(stats.get("nonseq_committed_tokens", stats.get("nonseq_accepted_tokens", 0.0)))
                nonseq_refine_calls += float(stats.get("nonseq_refine_forward_calls", 0.0))
                nonseq_steps += float(stats.get("nonseq_steps", 0.0))
                nonseq_router_used += float(stats.get("dvsd_router_used", 0.0))
                nonseq_active_heads += float(stats.get("nonseq_avg_active_heads", 0.0))
            else:
                out = model.generate_text(ids, max_new_tokens=args.max_new_tokens)
            total += max(0, int(out.numel() - ids.numel()))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.time() - t0
        row = {
            "prompt": prompt,
            "new_tokens": total,
            "seconds": elapsed,
            "tokens_per_sec": total / max(1e-9, elapsed),
            "decode_mode": args.decode_mode,
        }
        if args.decode_mode == "nonseq":
            row["nonseq_direct_commit_rate"] = nonseq_committed / max(1.0, nonseq_drafted)
            row["nonseq_accept_rate_compat"] = nonseq_acc / max(1.0, nonseq_drafted)
            row["nonseq_ar_fallback_tokens"] = nonseq_fallback
            row["nonseq_refine_forward_calls"] = nonseq_refine_calls
            row["nonseq_steps"] = nonseq_steps
            row["dvsd_router_used_rate"] = nonseq_router_used / max(1.0, float(args.iters))
            row["nonseq_avg_active_heads_mean"] = nonseq_active_heads / max(1.0, float(args.iters))
            row["dvsd_committed_per_expensive_pass"] = nonseq_committed / max(1.0, float(nonseq_steps + nonseq_refine_calls))
            row["nonseq_decode_style"] = args.nonseq_decode_style
        if device.type == "cuda":
            row["max_vram_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
        rows.append(row)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
