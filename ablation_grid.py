#!/usr/bin/env python3
"""Create a small NEED scaling/ablation command plan.

The script does not change the model code.  It prints concrete commands for
shorter runs so you can test stability, memory/retention usage,
and data quality before committing to a larger run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence
import math

try:
    from config_for_size import parse_scaled_number
except Exception:
    def parse_scaled_number(x, default_suffix=""):
        return int(float(str(x).rstrip("BMTK")))


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Print or save NEED scaling ablation commands")
    p.add_argument("--packed_index", default="data/packed/packed_index.json")
    p.add_argument("--out_root", default="runs/ablations")
    p.add_argument("--device", default="cuda")
    p.add_argument("--points", default="", help="Optional comma list params:tokens; empty builds a formulaic log-spaced sweep.")
    p.add_argument("--min_params", default="100M")
    p.add_argument("--max_params", default="1B")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--tokens_per_param", type=float, default=20.0)
    p.add_argument("--recipe", default="fast")
    p.add_argument("--peak_tflops", default="209")
    p.add_argument("--role_ablation", action="store_true", help="Emit matched-parameter commands for conditioned memory vs no-memory/role-separation ablations.")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    def fmt(n: int) -> str:
        n = int(n)
        for suffix, mult in (("T", 1_000_000_000_000), ("B", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)):
            if abs(n) >= mult:
                val = n / mult
                return f"{val:.3g}{suffix}"
        return str(n)

    if args.points.strip():
        points = [x.strip() for x in args.points.split(",") if x.strip()]
    else:
        lo = max(1, parse_scaled_number(args.min_params))
        hi = max(lo, parse_scaled_number(args.max_params))
        runs = max(1, int(args.runs))
        vals = [lo] if runs == 1 else [int(round(math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * i / (runs - 1)))) for i in range(runs)]
        points = [f"{fmt(p)}:{fmt(max(1, int(round(p * float(args.tokens_per_param)))))}" for p in vals]

    rows = []
    variants = [("base_conditioned_memory", "")]
    if args.role_ablation:
        # --disable_memory leaves the module allocated but sets memory_mix=0, so
        # parameter count stays matched while the memory contribution is removed.
        variants = [
            ("base_conditioned_memory", ""),
            ("matched_no_memory", "--disable_memory"),
            ("no_role_separation", "--disable_role_separation"),
        ]
    for point in points:
        params, tokens = point.split(":", 1)
        stem = params.replace(".", "_").replace("B", "b").replace("M", "m")
        for variant, extra in variants:
            out_dir = Path(args.out_root) / stem if not args.role_ablation else Path(args.out_root) / stem / variant
            cmd = [
                "python train.py",
                f"--target_params {params}",
                "--architecture dense",
                f"--target_tokens {tokens}",
                f"--packed_index {args.packed_index}",
                f"--out_dir {out_dir}",
                f"--device {args.device}",
                f"--recipe {args.recipe}",
                "--auto_optimize --auto_batch --target_vram_util 0.90",
                f"--peak_tflops {args.peak_tflops}",
            ]
            if extra:
                cmd.append(extra)
            sep = " " + "\\" + "\n  "
            rows.append({"params": params, "tokens": tokens, "variant": variant, "command": sep.join(cmd)})
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            variant = f" [{row['variant']}]" if args.role_ablation else ""
            print(f"# {row['params']} for {row['tokens']}{variant}")
            print(row["command"])
            print()


if __name__ == "__main__":
    main()
