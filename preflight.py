#!/usr/bin/env python3
"""Preflight checks before a long NEED training run."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch

from need_core import NeedModel, resolve_device
from train import build_arg_parser, apply_profile, maybe_apply_target_model_size, build_config, estimate_flops_per_token, verify_packed_index_integrity
try:
    from training_recipes import apply_training_recipe
except Exception:
    apply_training_recipe = None


def _path_tokens(path: str) -> int:
    if not path:
        return 0
    p = Path(path)
    if p.is_file() and p.suffix == ".json":
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("format", "").startswith("need_source_balanced"):
                return int(data.get("total_tokens", 0) or 0)
        except Exception:
            return 0
    meta = p.with_suffix(p.suffix + ".json")
    if meta.exists():
        try:
            return int(json.loads(meta.read_text(encoding="utf-8")).get("tokens", 0) or 0)
        except Exception:
            return 0
    return 0


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Check CUDA, dataset, disk, model size, and expected run settings")
    p.add_argument("--target_params", default="600M", help="Formulaic model size to check.")
    p.add_argument("--recipe", default="fast")
    p.add_argument("--packed_data", default="")
    p.add_argument("--packed_index", default="")
    p.add_argument("--data", default="")
    p.add_argument("--out_dir", default="runs/preflight")
    p.add_argument("--device", default="auto")
    p.add_argument("--target_tokens", default="")
    p.add_argument("--peak_tflops", type=float, default=209.0)
    args = p.parse_args(argv)

    train_parser = build_arg_parser()
    targs = train_parser.parse_args([])
    targs.target_params = args.target_params
    targs.device = args.device
    targs.out_dir = args.out_dir
    targs.packed_data = args.packed_data
    targs.packed_index = args.packed_index
    targs.data = args.data
    targs.peak_tflops = args.peak_tflops
    if apply_training_recipe is not None and args.recipe:
        apply_training_recipe(targs, args.recipe)
    if args.target_tokens:
        from config_for_size import parse_scaled_number
        targs.target_tokens = parse_scaled_number(args.target_tokens)
    maybe_apply_target_model_size(targs)
    cfg = build_config(targs)
    device = resolve_device(args.device)
    model = NeedModel(cfg)
    params = sum(p.numel() for p in model.parameters())
    del model
    dataset_tokens = _path_tokens(args.packed_index or args.packed_data)
    integrity = {}
    if args.packed_index:
        try:
            integrity = verify_packed_index_integrity(Path(args.packed_index), strict=True)
        except Exception as exc:
            integrity = {"ok": False, "error": str(exc), "index": args.packed_index}
    disk = shutil.disk_usage(str(Path(args.out_dir).parent if Path(args.out_dir).parent.exists() else "."))
    cuda = {}
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(dev)
        free, total = torch.cuda.mem_get_info(dev)
        cuda = {
            "name": props.name,
            "capability": f"{props.major}.{props.minor}",
            "vram_total_gb": total / 1e9,
            "vram_free_gb": free / 1e9,
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        }
    report: Dict[str, object] = {
        "target_params": str(args.target_params),
        "recipe": getattr(targs, "recipe", args.recipe),
        "device": str(device),
        "cuda": cuda,
        "parameters": int(params),
        "block_size": int(cfg.block_size),
        "target_tokens": int(getattr(targs, "target_tokens", 0) or 0),
        "dataset_tokens_from_metadata": int(dataset_tokens),
        "dataset_covers_target": bool(dataset_tokens == 0 or dataset_tokens >= int(getattr(targs, "target_tokens", 0) or 0)),
        "packed_index_integrity": integrity,
        "estimated_flops_per_token": estimate_flops_per_token(int(params)),
        "out_disk_free_gb": disk.free / 1e9,
        "compile": bool(getattr(targs, "compile", False)),
        "auto_batch": bool(getattr(targs, "auto_batch", False)),
        "resume_recommended": True,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
