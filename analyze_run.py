#!/usr/bin/env python3
"""Build a compact JSON/HTML report from NEED training/eval/generation logs."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def iter_json_objects(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj
            continue
        except Exception:
            pass
        m = re.search(r"(\{.*\})", line)
        if m:
            try:
                obj = json.loads(m.group(1))
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                pass


def summarize_series(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    vals = []
    for r in rows:
        if key in r:
            try:
                v = float(r[key])
                if math.isfinite(v):
                    vals.append(v)
            except Exception:
                pass
    if not vals:
        return {}
    return {"first": vals[0], "last": vals[-1], "best": min(vals), "mean_last_10": sum(vals[-10:]) / min(10, len(vals)), "n": len(vals)}


def load_config(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "config.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def html_escape(x: Any) -> str:
    s = str(x)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_html(report: Dict[str, Any], path: Path) -> None:
    rows = []
    for k, v in report.get("summary", {}).items():
        rows.append(f"<tr><th>{html_escape(k)}</th><td><pre>{html_escape(json.dumps(v, indent=2))}</pre></td></tr>")
    warnings = "".join(f"<li>{html_escape(w)}</li>" for w in report.get("warnings", []))
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>NEED run report</title>
<style>body{{font-family:system-ui, sans-serif; margin:2rem; max-width:1100px}} table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #ddd;padding:8px;vertical-align:top}} th{{width:260px;background:#f7f7f7;text-align:left}} pre{{white-space:pre-wrap;margin:0}} .warn{{color:#8a4b00}}</style></head>
<body><h1>NEED run report</h1><h2>Warnings</h2><ul class='warn'>{warnings}</ul><h2>Summary</h2><table>{''.join(rows)}</table></body></html>"""
    path.write_text(html, encoding="utf-8")


def analyze(run_dir: Path, logs: Sequence[Path], evals: Sequence[Path], traces: Sequence[Path]) -> Dict[str, Any]:
    train_rows: List[Dict[str, Any]] = []
    for p in logs:
        train_rows.extend(list(iter_json_objects(p)))
    eval_rows: List[Dict[str, Any]] = []
    for p in evals:
        if p.exists():
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(obj, dict): eval_rows.append(obj)
            except Exception:
                eval_rows.extend(list(iter_json_objects(p)))
    trace_rows: List[Dict[str, Any]] = []
    for p in traces:
        trace_rows.extend(list(iter_json_objects(p)))
    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "config": load_config(run_dir),
        "train_loss": summarize_series(train_rows, "loss"),
        "train_ce": summarize_series(train_rows, "ce"),
        "train_tok_s": summarize_series(train_rows, "tok_s"),
        "val_loss": summarize_series(train_rows, "val_loss"),
        "val_ce": summarize_series(train_rows, "val_ce"),
        "evals": eval_rows,
        "train_rows": len(train_rows),
        "trace_rows": len(trace_rows),
    }
    if trace_rows:
        accept = []
        generated = []
        for r in trace_rows:
            sm = r.get("spec_metrics", {}) if isinstance(r.get("spec_metrics", {}), dict) else {}
            if "spec_accept_rate" in sm:
                try: accept.append(float(sm["spec_accept_rate"]))
                except Exception: pass
            if "generated_chars" in r:
                try: generated.append(float(r["generated_chars"]))
                except Exception: pass
        summary["generation_traces"] = {
            "mean_spec_accept_rate": sum(accept)/len(accept) if accept else None,
            "mean_generated_chars": sum(generated)/len(generated) if generated else None,
        }
    warnings: List[str] = []
    tl = summary.get("train_loss", {})
    vl = summary.get("val_loss", {})
    if tl and vl and tl.get("last") and vl.get("last") and vl["last"] > 1.25 * tl["last"]:
        warnings.append("Validation loss is much higher than train loss; possible overfit or train/eval data mismatch.")
    ts = summary.get("train_tok_s", {})
    if ts and ts.get("last", 0) < 0.7 * max(1e-9, ts.get("first", ts.get("last", 1))):
        warnings.append("Token throughput declined materially during the run.")
    if not eval_rows:
        warnings.append("No external eval JSON was provided; add eval_need.py output for better comparisons.")
    return {"summary": summary, "warnings": warnings}


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", default="need_out")
    p.add_argument("--logs", nargs="*", default=[])
    p.add_argument("--eval_json", nargs="*", default=[])
    p.add_argument("--trace_jsonl", nargs="*", default=[])
    p.add_argument("--out_json", default="")
    p.add_argument("--out_html", default="")
    args = p.parse_args(argv)
    run_dir = Path(args.run_dir)
    logs = [Path(x) for x in args.logs] or [run_dir / "train_log.jsonl", run_dir / "train.log"]
    report = analyze(run_dir, logs, [Path(x) for x in args.eval_json], [Path(x) for x in args.trace_jsonl])
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.out_html:
        Path(args.out_html).parent.mkdir(parents=True, exist_ok=True)
        write_html(report, Path(args.out_html))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
