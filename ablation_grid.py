#!/usr/bin/env python3
"""Create a small NEED scaling/ablation command plan.

The script does not change the model code.  It prints concrete commands for
shorter 0.1B/0.3B/0.6B runs so you can test stability, memory/retention usage,
and data quality before committing to the full 85B-token run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

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
    p.add_argument("--points", default="0.1B:5B,0.3B:15B,0.6B:85B", help="Comma list params:tokens")
    p.add_argument("--recipe", default="fast")
    p.add_argument("--peak_tflops", default="209")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    rows = []
    for point in [x.strip() for x in args.points.split(",") if x.strip()]:
        params, tokens = point.split(":", 1)
        stem = params.replace(".", "_").replace("B", "b").replace("M", "m")
        cmd = [
            "python train.py",
            f"--target_params {params}",
            "--architecture dense",
            f"--target_tokens {tokens}",
            f"--packed_index {args.packed_index}",
            f"--out_dir {Path(args.out_root) / stem}",
            f"--device {args.device}",
            f"--recipe {args.recipe}",
            "--auto_optimize --auto_batch --target_vram_util 0.90",
            f"--peak_tflops {args.peak_tflops}",
        ]
        rows.append({"params": params, "tokens": tokens, "command": " \\\n  ".join(cmd)})
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            print(f"# {row['params']} for {row['tokens']}")
            print(row["command"])
            print()


if __name__ == "__main__":
    main()
