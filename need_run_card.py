#!/usr/bin/env python3
"""Generate a NEED model/run card from pipeline artifacts.

The card records what was trained, what data was used, how runtime is expected
to be launched, eval/audit scores, known limitations, and reproducibility paths.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def load_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def safe_rel(path: str, root: Path) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def summarize_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not manifest:
        return out
    for key in ["final_need_checkpoint", "runtime_profile", "posttrain_llrl_sidecar", "corpuses", "raw_image_data", "need_training"]:
        if key in manifest:
            out[key] = manifest.get(key)
    if "records" in manifest:
        out["stage_count"] = len(manifest.get("records") or [])
        out["failed_stages"] = [r for r in manifest.get("records") or [] if isinstance(r, dict) and r.get("returncode") not in (0, None)]
    return out


def build_card(args: argparse.Namespace) -> Dict[str, Any]:
    run_root = Path(args.run_root).resolve()
    manifest = load_json(args.manifest or run_root / "full_pipeline_manifest.json")
    config = load_json(args.config or run_root / "full_pipeline_config.redacted.json")
    state = load_json(args.state or run_root / "pipeline_state.json")
    audit = load_json(args.audit_json or run_root / "audit" / "audit_report.json")
    scorecard = load_json(args.scorecard_json or run_root / "evals" / "scorecard.json")
    runtime_profile = load_json(args.runtime_profile or manifest.get("runtime_profile", ""))
    corpus_plan = load_json(args.corpus_plan or run_root / "corpuses" / "plan.json")
    knowledge_manifest = load_json(run_root / "corpuses" / "knowledge" / "manifest.json")
    rl_manifest = load_json(run_root / "corpuses" / "rl" / "manifest.json")
    card = {
        "project": "NEED",
        "version_label": args.version_label or "complete_project",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_root": str(run_root),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "summary": summarize_manifest(manifest),
        "training_config_redacted": config,
        "pipeline_state": state,
        "data": {
            "corpus_plan_scale": corpus_plan.get("scale"),
            "corpus_plan_scale_info": corpus_plan.get("scale_info", {}),
            "knowledge_manifest": {k: knowledge_manifest.get(k) for k in ["target_tokens", "accepted_tokens", "accepted_docs", "completion_ratio"] if k in knowledge_manifest},
            "rl_manifest": {k: rl_manifest.get(k) for k in ["target_tokens", "accepted_tokens", "accepted_docs", "completion_ratio"] if k in rl_manifest},
            "general_alignment_slices": [
                s.get("slice") or s.get("name") for s in (rl_manifest.get("slices", []) or corpus_plan.get("slices", []))
                if isinstance(s, dict) and str(s.get("slice") or s.get("name") or "") in {"general_alignment_sft", "general_alignment_preferences", "truthfulness_calibration"}
            ],
        },
        "audit": {
            "status": audit.get("status"),
            "score": audit.get("score"),
            "summary": audit.get("summary", {}),
        },
        "evals": {
            "status": scorecard.get("status"),
            "overall": scorecard.get("overall"),
            "regressions_vs_previous": scorecard.get("regressions_vs_previous", []),
            "metric_names": sorted((scorecard.get("metrics") or {}).keys()),
        },
        "runtime": runtime_profile.get("runtime", runtime_profile),
        "known_limitations": [
            "Evaluation suite is a regression harness, not a substitute for large external benchmarks.",
            "Runtime latent tools are deterministic and narrow; unsupported tasks should fall back to normal reasoning.",
            "Raw image corpus ingestion depends on user-selected dataset licensing and filtering choices.",
            "Sidecar latent alignment quality should be checked on held-out prompts before deployment.",
        ],
        "recommended_launch": manifest.get("browser_command", []),
        "artifacts": {
            "audit_report": str(run_root / "audit" / "audit_report.html"),
            "scorecard": str(run_root / "evals" / "report.html"),
            "run_card": str(run_root / "run_card.md"),
        },
    }
    return card


def md_escape(x: Any) -> str:
    return str(x).replace("\n", " ").strip()


def write_markdown(card: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"# NEED run card: {md_escape(card.get('version_label'))}")
    lines.append("")
    lines.append(f"Created: `{md_escape(card.get('created_at'))}`")
    lines.append(f"Run root: `{md_escape(card.get('run_root'))}`")
    lines.append("")
    lines.append("## Summary")
    for k, v in (card.get("summary") or {}).items():
        lines.append(f"- **{k}**: `{md_escape(v)}`")
    lines.append("")
    lines.append("## Data")
    data = card.get("data") or {}
    lines.append(f"- Corpus scale: `{md_escape(data.get('corpus_plan_scale'))}`")
    lines.append(f"- Scale info: `{md_escape(json.dumps(data.get('corpus_plan_scale_info', {}), ensure_ascii=False))}`")
    lines.append(f"- Knowledge manifest: `{md_escape(json.dumps(data.get('knowledge_manifest', {}), ensure_ascii=False))}`")
    lines.append(f"- RL manifest: `{md_escape(json.dumps(data.get('rl_manifest', {}), ensure_ascii=False))}`")
    lines.append(f"- General alignment slices: `{md_escape(', '.join(data.get('general_alignment_slices', [])))}`")
    lines.append("")
    lines.append("## Audit and eval")
    lines.append(f"- Audit: `{md_escape((card.get('audit') or {}).get('status'))}` score `{md_escape((card.get('audit') or {}).get('score'))}`")
    lines.append(f"- Eval: `{md_escape((card.get('evals') or {}).get('status'))}` overall `{md_escape((card.get('evals') or {}).get('overall'))}`")
    regs = (card.get("evals") or {}).get("regressions_vs_previous", [])
    if regs:
        lines.append(f"- Regressions: `{md_escape(json.dumps(regs, ensure_ascii=False))}`")
    lines.append("")
    lines.append("## Runtime")
    lines.append("```json")
    lines.append(json.dumps(card.get("runtime", {}), indent=2, ensure_ascii=False, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Known limitations")
    for item in card.get("known_limitations", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Recommended launch")
    launch = card.get("recommended_launch") or []
    lines.append("```bash")
    lines.append(" ".join(str(x) for x in launch))
    lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html(card: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = html.escape(json.dumps(card, indent=2, ensure_ascii=False, sort_keys=True))
    doc = f"""<!doctype html><html><head><meta charset='utf-8'><title>NEED run card</title>
<style>body{{font-family:system-ui,sans-serif;background:#0f1117;color:#e9e9ef;margin:2rem;max-width:1200px}} pre{{white-space:pre-wrap;background:#171b24;padding:1rem;border:1px solid #303746}}</style></head>
<body><h1>NEED run card</h1><pre>{body}</pre></body></html>"""
    path.write_text(doc, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Generate NEED run/model card")
    p.add_argument("--run_root", default="runs/need_full_pipeline")
    p.add_argument("--version_label", default="complete_project")
    p.add_argument("--manifest", default="")
    p.add_argument("--config", default="")
    p.add_argument("--state", default="")
    p.add_argument("--audit_json", default="")
    p.add_argument("--scorecard_json", default="")
    p.add_argument("--runtime_profile", default="")
    p.add_argument("--corpus_plan", default="")
    p.add_argument("--out_json", default="")
    p.add_argument("--out_md", default="")
    p.add_argument("--out_html", default="")
    args = p.parse_args(argv)
    card = build_card(args)
    run_root = Path(args.run_root)
    out_json = Path(args.out_json) if args.out_json else run_root / "run_card.json"
    out_md = Path(args.out_md) if args.out_md else run_root / "run_card.md"
    out_html = Path(args.out_html) if args.out_html else run_root / "run_card.html"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(card, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    write_markdown(card, out_md)
    write_html(card, out_html)
    print(json.dumps(card, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
