#!/usr/bin/env python3
"""Short LR sweep"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Sequence

import torch

from need_core import NeedModel, Special, resolve_device
from train import build_arg_parser, apply_profile, maybe_apply_target_model_size, build_config, configure_optimizer, _autocast_dtype
try:
    from training_recipes import apply_training_recipe
except Exception:
    apply_training_recipe = None


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Run a quick NEED learning-rate stability sweep")
    p.add_argument("--target_params", default="600M", help="Formulaic model size for the sweep.")
    p.add_argument("--recipe", default="debug")
    p.add_argument("--device", default="auto")
    p.add_argument("--lrs", default="5e-5,1e-4,1.35e-4,1.6e-4,2e-4")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out_json", default="")
    args = p.parse_args(argv)

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    rows = []
    for lr_s in [x.strip() for x in args.lrs.split(",") if x.strip()]:
        lr = float(lr_s)
        targs = build_arg_parser().parse_args([])
        targs.target_params = args.target_params
        targs.device = args.device
        targs.batch_size = args.batch_size
        targs.grad_accum_steps = 1
        targs.lr = lr
        if apply_training_recipe is not None and args.recipe:
            apply_training_recipe(targs, args.recipe)
        targs.lr = lr
        targs.batch_size = args.batch_size
        targs.grad_accum_steps = 1
        maybe_apply_target_model_size(targs)
        cfg = build_config(targs)
        targs.lr = lr
        targs.batch_size = args.batch_size
        targs.grad_accum_steps = 1
        model = NeedModel(cfg).to(device).train()
        opt = configure_optimizer(model, targs)
        losses = []
        finite = True
        for step in range(max(1, args.steps)):
            x = torch.randint(Special.byte_start, min(cfg.vocab_size, Special.byte_start + 128), (args.batch_size, cfg.block_size), dtype=torch.long, device=device)
            y = torch.roll(x, shifts=-1, dims=1).unsqueeze(-1).contiguous()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=_autocast_dtype(targs), enabled=(device.type == "cuda" and targs.amp != "off")):
                _, loss, aux = model(x, y, image_mask_positions=torch.zeros_like(x, dtype=torch.bool))
            if loss is None or not torch.isfinite(loss):
                finite = False; break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), targs.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        slope = float("nan")
        if len(losses) >= 2:
            slope = losses[-1] - losses[0]
        row = {"lr": lr, "finite": finite, "steps": len(losses), "loss_start": losses[0] if losses else None, "loss_end": losses[-1] if losses else None, "loss_delta": slope}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
