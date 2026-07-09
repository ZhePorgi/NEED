#!/usr/bin/env python3
"""Run staged NEED training from corpus shards.

The script writes phase configs and shell commands, and can optionally execute
phases. It is designed for small NEED models where data order matters.  When
--continue_phases is enabled, each phase initializes from the previous one's
checkpoint so the curriculum is a real continuation rather than separate runs.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from config_for_size import build_config
except Exception:
    build_config = None  # type: ignore


def default_sources(corpus_dir: Path) -> Dict[str, str]:
    return {
        "knowledge": str(corpus_dir / "knowledge" / "train.jsonl"),
        "sft": str(corpus_dir / "rl" / "sft.jsonl"),
        "preference": str(corpus_dir / "rl" / "preferences.jsonl"),
        "rlvr": str(corpus_dir / "rl" / "rlvr.jsonl"),
        "safety": str(corpus_dir / "rl" / "preferences.jsonl"),
        "math_science": str(corpus_dir / "knowledge" / "train.jsonl"),
        "if": str(corpus_dir / "rl" / "rlvr.jsonl"),
        "sft_synth": str(corpus_dir / "rl" / "sft.synthetic.jsonl"),
        "preference_synth": str(corpus_dir / "rl" / "preferences.synthetic.jsonl"),
        "rlvr_synth": str(corpus_dir / "rl" / "rlvr.synthetic.jsonl"),
        "low_data_control": str(corpus_dir / "rl" / "low_data_control_interactions.synthetic.jsonl"),
        "sidecar_td": str(corpus_dir / "rl" / "sidecar_latent_alignment.synthetic.jsonl"),
    }


def materialize_phase_mix(sources: Dict[str, str], mix: Dict[str, float], out_file: Path, max_lines: int = 0) -> Dict[str, Any]:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    total_weight = sum(max(0.0, float(w)) for w in mix.values()) or 1.0
    with out_file.open("w", encoding="utf-8") as out:
        for name, weight in mix.items():
            raw_path = sources.get(name, "")
            if not raw_path or weight <= 0:
                continue
            path = Path(raw_path)
            if not path.exists() or path.is_dir():
                continue
            limit = int(max_lines * (float(weight) / total_weight)) if max_lines else 0
            n = 0
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.strip():
                        continue
                    out.write(line)
                    n += 1
                    if limit and n >= max(1, limit):
                        break
            counts[name] = n
    return {"phase_file": str(out_file), "counts": counts}


def _normalize_phases(phases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    total = sum(float(p.get("token_fraction", 0.0)) for p in phases) or 1.0
    out = []
    for p in phases:
        q = dict(p)
        q["token_fraction"] = float(q.get("token_fraction", 0.0)) / total
        out.append(q)
    return out


def build_phases(params_m: float, tokens: int, include_rl_td_end: bool = True) -> List[Dict[str, Any]]:
    if build_config is not None:
        phases = [dict(p) for p in build_config(params_m, tokens)["curriculum"]["phases"]]
    else:
        phases = [
            {"name": "language_foundation", "token_fraction": 0.30, "mix": {"knowledge": 1.0}},
            {"name": "knowledge_reasoning", "token_fraction": 0.35, "mix": {"knowledge": 0.80, "rlvr": 0.20}},
            {"name": "instruction_following", "token_fraction": 0.20, "mix": {"knowledge": 0.45, "sft": 0.45, "rlvr": 0.10}},
            {"name": "general_alignment", "token_fraction": 0.15, "mix": {"sft": 0.35, "preference": 0.30, "rlvr": 0.25, "safety": 0.10}},
        ]
    if include_rl_td_end:
        # Keep low-level RL / thought-distillation style rows at the end of the
        # curriculum. This phase is intentionally small and late so it tunes
        # behavior and sidecar-facing latent summaries after broad competence.
        phases.append({
            "name": "final_rl_td_low_data",
            "token_fraction": 0.06,
            "mix": {
                "sft_synth": 0.20,
                "preference_synth": 0.22,
                "rlvr_synth": 0.28,
                "sidecar_td": 0.20,
                "low_data_control": 0.10,
            },
        })
    return _normalize_phases(phases)


def phase_command(args: argparse.Namespace, phase: Dict[str, Any], phase_data: Path, phase_idx: int, total_steps: int, init_from: str = "") -> List[str]:
    steps = max(1, int(total_steps * float(phase.get("token_fraction", 0.25))))
    out = Path(args.out_dir) / f"{phase_idx:02d}_{phase['name']}"
    cmd = [
        sys.executable, str(Path(__file__).with_name("train.py")),
        "--data", str(phase_data), "--out_dir", str(out), "--target_params", f"{args.params_m:g}M",
        "--max_steps", str(steps), "--batch_size", str(args.batch_size),
        "--grad_accum_steps", str(args.grad_accum_steps), "--lr", str(args.lr),
        "--device", args.device, "--log_interval", str(args.log_interval),
        "--eval_interval", str(max(20, min(args.eval_interval, max(20, steps // 4)))),
    ]
    if init_from:
        cmd += ["--init_from", init_from, "--init_prefer_best"]
    image_ratio = float(phase.get("image_ratio", args.image_ratio if args.image_tokens or args.image_dir else 0.0))
    if args.image_tokens and image_ratio > 0:
        cmd += ["--image_tokens", args.image_tokens, "--image_ratio", str(image_ratio)]
    elif args.image_dir and image_ratio > 0:
        cmd += ["--image_dir", args.image_dir, "--image_ratio", str(image_ratio)]
    if args.visual_tokenizer:
        cmd += ["--visual_tokenizer", args.visual_tokenizer]
    if args.extra_train_args:
        cmd += shlex.split(args.extra_train_args)
    return cmd


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus_dir", default="data/corpuses")
    p.add_argument("--out_dir", default="need_curriculum")
    p.add_argument("--params_m", type=float, default=30.0)
    p.add_argument("--tokens", type=int, default=10_000_000_000)
    p.add_argument("--total_steps", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="auto")
    p.add_argument("--image_dir", default="")
    p.add_argument("--image_tokens", default="")
    p.add_argument("--visual_tokenizer", default="")
    p.add_argument("--image_ratio", type=float, default=0.1)
    p.add_argument("--max_phase_lines", type=int, default=0, help="For smoke tests; 0 uses all available lines")
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--extra_train_args", default="")
    p.add_argument("--continue_phases", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include_rl_td_end", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run", action="store_true")
    args = p.parse_args(argv)
    corpus = Path(args.corpus_dir)
    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)
    sources = default_sources(corpus)
    phases = build_phases(args.params_m, args.tokens, include_rl_td_end=bool(args.include_rl_td_end))
    plan: Dict[str, Any] = {"sources": sources, "phases": [], "commands": [], "final_checkpoint": ""}
    previous_ckpt = ""
    for i, ph in enumerate(phases):
        phase_data = out_root / "phase_data" / f"{i:02d}_{ph['name']}.jsonl"
        mat = materialize_phase_mix(sources, ph.get("mix", {}), phase_data, args.max_phase_lines)
        init_from = previous_ckpt if args.continue_phases else ""
        cmd = phase_command(args, ph, phase_data, i, args.total_steps, init_from=init_from)
        phase_out = Path(args.out_dir) / f"{i:02d}_{ph['name']}"
        previous_ckpt = str(phase_out)
        plan["phases"].append({**ph, **mat, "out_dir": str(phase_out), "init_from": init_from})
        plan["commands"].append(cmd)
    plan["final_checkpoint"] = previous_ckpt
    (out_root / "curriculum_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2))
    if args.run:
        for cmd in plan["commands"]:
            print("running:", " ".join(shlex.quote(x) for x in cmd), flush=True)
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
