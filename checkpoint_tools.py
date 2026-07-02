#!/usr/bin/env python3
"""Inspect, merge, average, strip, and export NEED checkpoints."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch

try:
    from safetensors.torch import load_file as safe_load_file, save_file as safe_save_file
except Exception:
    safe_load_file = None  # type: ignore
    safe_save_file = None  # type: ignore


def load_state(path: Path) -> Dict[str, torch.Tensor]:
    if path.is_dir():
        for name in ("model.safetensors", "best.safetensors", "model.pt", "best.pt"):
            p = path / name
            if p.exists():
                return load_state(p)
        raise FileNotFoundError(f"no model weights in {path}")
    if path.suffix == ".safetensors" and safe_load_file is not None:
        return {k: v.cpu() for k, v in safe_load_file(str(path), device="cpu").items()}
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise ValueError(f"unsupported checkpoint format: {path}")
    return {k: v.cpu() for k, v in obj.items() if torch.is_tensor(v)}


def save_state(state: Dict[str, torch.Tensor], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".safetensors" and safe_save_file is not None:
        safe_save_file({k: v.cpu().contiguous() for k, v in state.items()}, str(out))
    else:
        torch.save({k: v.cpu() for k, v in state.items()}, out)


def inspect(path: Path) -> Dict[str, object]:
    state = load_state(path)
    total = sum(v.numel() for v in state.values())
    bytes_ = sum(v.numel() * v.element_size() for v in state.values())
    groups: Dict[str, int] = {}
    for k, v in state.items():
        groups[k.split('.')[0]] = groups.get(k.split('.')[0], 0) + v.numel()
    return {"path": str(path), "tensors": len(state), "parameters": total, "size_mb": bytes_ / (1024 * 1024), "groups": dict(sorted(groups.items(), key=lambda kv: kv[1], reverse=True))}


def average(paths: Sequence[Path], weights: Optional[Sequence[float]] = None) -> Dict[str, torch.Tensor]:
    states = [load_state(p) for p in paths]
    if weights is None:
        weights = [1.0 / len(states)] * len(states)
    s = sum(weights)
    weights = [float(w) / max(1e-12, s) for w in weights]
    keys = set(states[0].keys())
    for st in states[1:]:
        keys &= set(st.keys())
    out: Dict[str, torch.Tensor] = {}
    for k in sorted(keys):
        base = states[0][k]
        if not torch.is_floating_point(base):
            out[k] = base.clone()
            continue
        acc = torch.zeros_like(base, dtype=torch.float32)
        ok = True
        for st, w in zip(states, weights):
            if st[k].shape != base.shape:
                ok = False; break
            acc += st[k].float() * float(w)
        if ok:
            out[k] = acc.to(base.dtype)
    return out


def diff(a: Path, b: Path) -> Dict[str, object]:
    sa = load_state(a); sb = load_state(b)
    keys = sorted(set(sa.keys()) & set(sb.keys()))
    rows = []
    for k in keys:
        if sa[k].shape != sb[k].shape or not torch.is_floating_point(sa[k]):
            continue
        d = (sb[k].float() - sa[k].float()).flatten()
        base = sa[k].float().flatten()
        rows.append({"name": k, "l2": float(d.norm()), "rel_l2": float(d.norm() / base.norm().clamp_min(1e-8)), "mean_abs": float(d.abs().mean())})
    rows.sort(key=lambda r: r["rel_l2"], reverse=True)
    return {"a": str(a), "b": str(b), "changed_tensors": len(rows), "top_diffs": rows[:50]}


def strip_state(state: Dict[str, torch.Tensor], keep: str = "core") -> Dict[str, torch.Tensor]:
    if keep == "all":
        return dict(state)
    drop_prefixes = []
    if keep == "inference":
        drop_prefixes = ["mtp_projs", "image_quality"]
    elif keep == "text_only":
        drop_prefixes = ["image_", "object_program", "image_scan", "image_quality", "text_proj", "image_proj"]
    elif keep == "core":
        drop_prefixes = []
    out = {}
    for k, v in state.items():
        if any(k.startswith(p) for p in drop_prefixes):
            continue
        out[k] = v
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("inspect"); i.add_argument("checkpoint")
    a = sub.add_parser("average"); a.add_argument("checkpoints", nargs="+"); a.add_argument("--out", required=True); a.add_argument("--weights", default="")
    d = sub.add_parser("diff"); d.add_argument("a"); d.add_argument("b"); d.add_argument("--out_json", default="")
    s = sub.add_parser("strip"); s.add_argument("checkpoint"); s.add_argument("--keep", choices=["all", "core", "inference", "text_only"], default="inference"); s.add_argument("--out", required=True); s.add_argument("--copy_config_from", default="")
    args = p.parse_args(argv)
    if args.cmd == "inspect":
        print(json.dumps(inspect(Path(args.checkpoint)), indent=2))
    elif args.cmd == "average":
        weights = [float(x) for x in args.weights.split(',')] if args.weights else None
        state = average([Path(x) for x in args.checkpoints], weights)
        save_state(state, Path(args.out))
        print(json.dumps({"done": True, "out": args.out, "tensors": len(state)}, indent=2))
    elif args.cmd == "diff":
        res = diff(Path(args.a), Path(args.b))
        if args.out_json:
            Path(args.out_json).write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=2))
    elif args.cmd == "strip":
        state = strip_state(load_state(Path(args.checkpoint)), args.keep)
        out = Path(args.out)
        if out.suffix:
            save_state(state, out)
        else:
            out.mkdir(parents=True, exist_ok=True)
            save_state(state, out / "model.pt")
            cfg_src = Path(args.copy_config_from) if args.copy_config_from else Path(args.checkpoint)
            if cfg_src.is_dir() and (cfg_src / "config.json").exists():
                shutil.copy2(cfg_src / "config.json", out / "config.json")
        print(json.dumps({"done": True, "out": str(out), "keep": args.keep, "tensors": len(state)}, indent=2))


if __name__ == "__main__":
    main()
