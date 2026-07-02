#!/usr/bin/env python3
"""Microbatch throughput sweep for NEED."""
from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from throughput_benchmark import main as throughput_main


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Run throughput_benchmark.py over several batch sizes")
    p.add_argument("--need_profile", default="need_0_6b_dense_85b")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_sizes", default="1,2,3,4,6,8")
    p.add_argument("--warmup", default="10")
    p.add_argument("--iters", default="50")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile_mode", default="max-autotune")
    p.add_argument("--compile_cudagraphs", action="store_true")
    p.add_argument("--peak_tflops", default="209")
    args = p.parse_args(argv)
    for b in [x.strip() for x in args.batch_sizes.split(",") if x.strip()]:
        bench_args = [
            "--need_profile", args.need_profile,
            "--device", args.device,
            "--batch_size", b,
            "--warmup", str(args.warmup),
            "--iters", str(args.iters),
            "--peak_tflops", str(args.peak_tflops),
        ]
        if args.compile:
            bench_args.append("--compile")
            bench_args.extend(["--compile_mode", args.compile_mode])
        if args.compile_cudagraphs:
            bench_args.append("--compile_cudagraphs")
        print(json.dumps({"batch_size": int(b), "status": "starting"}), flush=True)
        throughput_main(bench_args)


if __name__ == "__main__":
    main()
