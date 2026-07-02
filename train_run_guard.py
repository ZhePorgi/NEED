#!/usr/bin/env python3
"""Watch NEED train_log.jsonl for early failure signals.

Use this alongside a long run or as a post-hoc check.  It catches common issues:
loss explosions, no recent checkpoints, low or regressing throughput, non-finite
recoveries, eval loss regression, and missing MFU logs.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def read_rows(path: Path, max_rows: int = 5000) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
                if len(rows) > max_rows:
                    rows = rows[-max_rows:]
    return rows


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def summarize(rows: List[Dict[str, Any]], out_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    train = [r for r in rows if "step" in r and "loss" in r]
    evals = [r for r in rows if "eval_step" in r]
    latest = train[-1] if train else {}
    if not rows:
        issues.append({"severity": "error", "code": "no_metrics", "message": "metrics log is empty or missing"})
    if train:
        last_losses = [float(r["loss"]) for r in train[-20:] if finite(r.get("loss"))]
        if not last_losses:
            issues.append({"severity": "error", "code": "no_finite_loss", "message": "recent train rows have no finite loss"})
        elif len(last_losses) >= 5:
            median = sorted(last_losses)[len(last_losses)//2]
            if last_losses[-1] > median * float(args.loss_spike_ratio):
                issues.append({"severity": "warning", "code": "loss_spike", "message": f"latest loss {last_losses[-1]:.4g} exceeds recent median {median:.4g}"})
        bad_events = int(latest.get("nonfinite_events", 0) or 0) + int(latest.get("skipped_optimizer_steps", 0) or 0)
        if bad_events > int(args.max_bad_events):
            issues.append({"severity": "error", "code": "too_many_recoveries", "message": f"bad-loss/grad recoveries={bad_events}"})
        tok_s = [float(r["tok_s"]) for r in train[-20:] if finite(r.get("tok_s"))]
        if len(tok_s) >= 5:
            recent = sum(tok_s[-5:]) / 5.0
            prior = sum(tok_s[:max(1, min(5, len(tok_s)-5))]) / max(1, min(5, len(tok_s)-5))
            if prior > 0 and recent < prior * float(args.throughput_drop_ratio):
                issues.append({"severity": "warning", "code": "throughput_drop", "message": f"recent tok/s {recent:.2f} below prior {prior:.2f}"})
        if args.expect_mfu and not any("mfu" in r for r in train[-20:]):
            issues.append({"severity": "warning", "code": "missing_mfu", "message": "recent train rows have no mfu field; set --peak_tflops"})
    if len(evals) >= 3:
        vals = [float(r.get("val_loss")) for r in evals if finite(r.get("val_loss"))]
        if len(vals) >= 3 and vals[-1] > min(vals[:-1]) * float(args.eval_regression_ratio):
            issues.append({"severity": "warning", "code": "eval_regression", "message": f"latest val_loss {vals[-1]:.4g} is above best {min(vals[:-1]):.4g}"})
    ckpt = out_dir / "training_state.pt"
    if args.max_checkpoint_age_s > 0:
        if not ckpt.exists():
            issues.append({"severity": "error", "code": "missing_training_state", "message": str(ckpt)})
        else:
            age = time.time() - ckpt.stat().st_mtime
            if age > float(args.max_checkpoint_age_s):
                issues.append({"severity": "warning", "code": "stale_checkpoint", "message": f"training_state.pt age {age:.0f}s"})
    counts: Dict[str, int] = {}
    for it in issues:
        counts[it["severity"]] = counts.get(it["severity"], 0) + 1
    return {
        "format": "need_train_run_guard_v1",
        "ok": counts.get("error", 0) == 0,
        "rows": len(rows),
        "latest_step": int(latest.get("step", -1)) if latest else None,
        "latest_tokens_seen": int(latest.get("tokens_seen", 0) or 0) if latest else 0,
        "latest_loss": latest.get("loss") if latest else None,
        "latest_tok_s": latest.get("tok_s") if latest else None,
        "issues": issues,
        "counts": counts,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Check NEED training logs for drift, stalls, and recovery events")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--metrics_jsonl", default="", help="Default: out_dir/train_log.jsonl")
    p.add_argument("--max_rows", type=int, default=5000)
    p.add_argument("--max_bad_events", type=int, default=20)
    p.add_argument("--loss_spike_ratio", type=float, default=2.5)
    p.add_argument("--throughput_drop_ratio", type=float, default=0.70)
    p.add_argument("--eval_regression_ratio", type=float, default=1.08)
    p.add_argument("--max_checkpoint_age_s", type=float, default=0.0)
    p.add_argument("--expect_mfu", action="store_true")
    p.add_argument("--json_out", default="")
    args = p.parse_args(argv)
    out_dir = Path(args.out_dir)
    metrics = Path(args.metrics_jsonl) if args.metrics_jsonl else out_dir / "train_log.jsonl"
    report = summarize(read_rows(metrics, args.max_rows), out_dir, args)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report.get("ok", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
