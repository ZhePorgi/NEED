#!/usr/bin/env python3
"""Throughput benchmark helpers for NEED and the external LM sidecar."""
from __future__ import annotations
import argparse, json, time
from typing import Optional, Sequence

import torch

from sidecar_lm_runtime import FastSidecarLMRuntime, SidecarLMRuntimeConfig
from sidecar_attention_kernels import estimate_sidecar_lm_tps


def _run_need_benchmark(args: argparse.Namespace) -> None:
    from train import build_arg_parser, build_config, configure_runtime, _autocast_dtype, estimate_flops_per_token
    from need_core import NeedModel, Special, resolve_device

    train_args = build_arg_parser().parse_args([
        "--profile", args.need_profile,
        "--device", args.device,
        "--batch_size", str(args.batch_size),
        "--max_steps", str(args.iters),
        "--eval_interval", "100000000",
        "--minimal_aux_metrics",
    ])
    train_args.compile = bool(args.compile)
    train_args.compile_mode = args.compile_mode
    train_args.compile_dynamic = False
    train_args.compile_cudagraphs = bool(args.compile_cudagraphs)
    train_args.amp = args.dtype if args.dtype in ("bf16", "fp16") else "off"
    device = resolve_device(args.device)
    configure_runtime(train_args, device)
    cfg = build_config(train_args)
    model = NeedModel(cfg).to(device).train()
    if train_args.compile and hasattr(torch, "compile"):
        kwargs = {"backend": "inductor", "dynamic": False}
        if train_args.compile_mode:
            kwargs["mode"] = train_args.compile_mode
        model = torch.compile(model, **kwargs)  # type: ignore[assignment]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=(device.type == "cuda")) if device.type == "cuda" else torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(Special.byte_start, Special.byte_start + 128, (args.batch_size, cfg.block_size), device=device, dtype=torch.long)
    y = torch.stack([torch.roll(x, shifts=-(h + 1), dims=1) for h in range(cfg.n_predict_heads)], dim=-1).contiguous()

    def one_step() -> float:
        opt.zero_grad(set_to_none=True)
        if device.type == "cuda" and train_args.compile_cudagraphs:
            try:
                torch.compiler.cudagraph_mark_step_begin()  # type: ignore[attr-defined]
            except Exception:
                pass
        with torch.autocast(device_type=device.type, dtype=_autocast_dtype(train_args), enabled=(device.type == "cuda" and train_args.amp != "off")):
            _, loss, _ = model(x, y)
            assert loss is not None
        loss.backward()
        opt.step()
        return float(loss.detach().cpu())

    for _ in range(max(0, args.warmup)):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    last_loss = 0.0
    for _ in range(max(1, args.iters)):
        last_loss = one_step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0
    params = int(sum(p.numel() for p in (model._orig_mod if hasattr(model, "_orig_mod") else model).parameters()))
    toks = int(args.iters) * int(args.batch_size) * int(cfg.block_size)
    tok_s = toks / max(dt, 1e-9)
    row = {"mode": "need", "profile": args.need_profile, "batch_size": args.batch_size, "block_size": cfg.block_size, "tokens_per_sec": tok_s, "loss": last_loss, "params": params}
    if args.peak_tflops > 0:
        row["mfu"] = tok_s * estimate_flops_per_token(params) / (float(args.peak_tflops) * 1e12)
    print(json.dumps(row, indent=2))


def main(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser()
    p.add_argument('--need_profile', default='', help='Benchmark NEED training with this train.py profile instead of sidecar generation')
    p.add_argument('--model', default='HuggingFaceTB/SmolLM2-135M-Instruct')
    p.add_argument('--device', default='auto')
    p.add_argument('--dtype', choices=['bf16','fp16','fp32'], default='bf16')
    p.add_argument('--attn_backend', choices=['auto','sdpa','flash_attention_2','eager'], default='sdpa')
    p.add_argument('--cache_implementation', choices=['static','dynamic','offloaded','none'], default='static')
    p.add_argument('--gpu_l2_mb', type=float, default=96.0)
    p.add_argument('--max_batch', type=int, default=8)
    p.add_argument('--context_tokens', type=int, default=2048)
    p.add_argument('--prompt', default='Explain why small models can be fast.')
    p.add_argument('--max_new_tokens', type=int, default=128)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--iters', type=int, default=3)
    p.add_argument('--compile', action='store_true')
    p.add_argument('--compile_mode', choices=['', 'default', 'reduce-overhead', 'max-autotune'], default='')
    p.add_argument('--compile_cudagraphs', action='store_true')
    p.add_argument('--peak_tflops', type=float, default=0.0)
    p.add_argument('--estimate_only', action='store_true')
    args = p.parse_args(argv)
    if args.need_profile:
        _run_need_benchmark(args); return
    if args.estimate_only:
        print(json.dumps(estimate_sidecar_lm_tps(), indent=2)); return
    rt = FastSidecarLMRuntime(SidecarLMRuntimeConfig(model=args.model, device=args.device, dtype=args.dtype, attn_backend=args.attn_backend, cache_implementation=args.cache_implementation, l2_cache_mb=args.gpu_l2_mb, max_batch=args.max_batch, max_context_tokens=args.context_tokens)).load()
    prompts = [args.prompt] * args.batch_size
    for _ in range(args.warmup):
        rt.generate_many(prompts, max_new_tokens=min(8,args.max_new_tokens), temperature=0.0, top_p=1.0, top_k=1)
    total_tok = 0; total_t = 0.0
    for _ in range(args.iters):
        t0 = time.perf_counter(); rt.generate_many(prompts, max_new_tokens=args.max_new_tokens, temperature=0.0, top_p=1.0, top_k=1); dt = time.perf_counter()-t0
        toks = args.max_new_tokens * len(prompts)
        total_tok += toks; total_t += dt
    print(json.dumps({'mode':'sidecar','model':args.model,'batch_size':args.batch_size,'new_tokens':args.max_new_tokens,'tokens_per_sec':total_tok/max(total_t,1e-9),'cache_plan':rt.cache_plan,'roofline_estimate':rt.estimate_tps()}, indent=2))
if __name__ == '__main__': main()
