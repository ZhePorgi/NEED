#!/usr/bin/env python3
"""Compare two NEED checkpoints on the same prompts.

This script is intentionally lightweight and local.  It generates with both
checkpoints, scores simple expected-output cases when provided, and writes JSON,
JSONL, and HTML artifacts for qualitative review.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

try:
    import torch
    from need_core import ByteTokenizer, load_model, resolve_device
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    ByteTokenizer = None  # type: ignore
    load_model = None  # type: ignore
    resolve_device = None  # type: ignore


@dataclass
class PromptCase:
    id: str
    prompt: str
    area: str = "general"
    regex: str = ""
    contains: str = ""
    max_new_tokens: int = 128


DEFAULT_PROMPTS = [
    PromptCase("risk_score", "Score the operational risk from 0-10: boxed lithium batteries are stored in a 38C warehouse near cardboard, checked weekly.", "numeric_risk"),
    PromptCase("tool_math", "Compute 17*23 + 41. Answer briefly.", "numeric"),
    PromptCase("json", "Return JSON only with keys action and confidence for this transcript: 'Nora will call the carrier Friday.'", "json"),
    PromptCase("memory", "A behavioral memory says to be concise. What factual personal detail does that prove?", "memory_policy"),
    PromptCase("image_prompt", "Rewrite this image prompt without adding unsupported objects: a quiet loading dock at dusk with one forklift.", "image_behavior"),
]


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                continue


def load_prompts(path: str) -> List[PromptCase]:
    if not path:
        return list(DEFAULT_PROMPTS)
    out: List[PromptCase] = []
    for obj in _iter_jsonl(Path(path)):
        out.append(PromptCase(
            id=str(obj.get("id") or f"case_{len(out):04d}"),
            prompt=str(obj.get("prompt") or obj.get("input") or ""),
            area=str(obj.get("area") or obj.get("category") or "general"),
            regex=str(obj.get("regex") or ""),
            contains=str(obj.get("contains") or ""),
            max_new_tokens=int(obj.get("max_new_tokens") or 128),
        ))
    return [x for x in out if x.prompt] or list(DEFAULT_PROMPTS)


def score(output: str, case: PromptCase) -> Tuple[Optional[bool], str]:
    if case.regex:
        return re.search(case.regex, output or "", flags=re.I | re.S) is not None, "regex"
    if case.contains:
        return case.contains.lower() in (output or "").lower(), "contains"
    return None, "qualitative"


def gen(model: Any, tok: Any, prompt: str, max_new_tokens: int, args: argparse.Namespace, device: Any) -> str:
    ids = torch.tensor([tok.encode(prompt, add_bos=True)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate_text(
            ids[:, -model.cfg.block_size:],
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram=args.no_repeat_ngram,
            aux_score_weight=args.aux_score_weight,
            proactive_aux_score=not args.disable_proactive_aux_score,
            aux_score_risk_threshold=args.aux_score_risk_threshold,
            aux_score_contradiction_threshold=args.aux_score_contradiction_threshold,
        )
    decoded = tok.decode(out[0].tolist())
    return decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded.strip()


def compare(args: argparse.Namespace) -> Dict[str, Any]:
    cases = load_prompts(args.prompts_jsonl)
    if args.dry_run:
        rows = [{"id": c.id, "area": c.area, "prompt": c.prompt, "a": "<dry_run>", "b": "<dry_run>", "winner": "tie"} for c in cases]
        return {"checkpoint_a": args.checkpoint_a, "checkpoint_b": args.checkpoint_b, "dry_run": True, "rows": rows, "summary": {"cases": len(rows)}}
    if torch is None or load_model is None or ByteTokenizer is None or resolve_device is None:
        raise RuntimeError("torch/NEED model imports unavailable")
    device = resolve_device(args.device)
    tok = ByteTokenizer()
    model_a = load_model(args.checkpoint_a, device=device, prefer_best=args.prefer_best_a, kernel_backend=args.kernel_backend)
    model_b = load_model(args.checkpoint_b, device=device, prefer_best=args.prefer_best_b, kernel_backend=args.kernel_backend)
    rows: List[Dict[str, Any]] = []
    a_wins = b_wins = ties = 0
    t0 = time.time()
    for c in cases:
        out_a = gen(model_a, tok, c.prompt, c.max_new_tokens, args, device)
        out_b = gen(model_b, tok, c.prompt, c.max_new_tokens, args, device)
        ok_a, score_kind = score(out_a, c)
        ok_b, _ = score(out_b, c)
        winner = "tie"
        if ok_a is not None and ok_b is not None:
            if ok_a and not ok_b:
                winner = "a"; a_wins += 1
            elif ok_b and not ok_a:
                winner = "b"; b_wins += 1
            else:
                ties += 1
        else:
            ties += 1
        rows.append({"id": c.id, "area": c.area, "prompt": c.prompt, "score_kind": score_kind, "a_ok": ok_a, "b_ok": ok_b, "winner": winner, "a": out_a[:4000], "b": out_b[:4000]})
    return {
        "checkpoint_a": args.checkpoint_a,
        "checkpoint_b": args.checkpoint_b,
        "dry_run": False,
        "summary": {"cases": len(rows), "a_wins": a_wins, "b_wins": b_wins, "ties": ties, "seconds": round(time.time() - t0, 3)},
        "rows": rows,
    }


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def write_html(report: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in report.get("rows", []):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(r.get('id','')))}</td>"
            f"<td>{html.escape(str(r.get('area','')))}</td>"
            f"<td><pre>{html.escape(str(r.get('prompt','')))}</pre></td>"
            f"<td><pre>{html.escape(str(r.get('a','')))}</pre></td>"
            f"<td><pre>{html.escape(str(r.get('b','')))}</pre></td>"
            f"<td>{html.escape(str(r.get('winner','')))}</td>"
            "</tr>"
        )
    summary = html.escape(json.dumps(report.get("summary", {}), indent=2, ensure_ascii=False))
    doc = f"""<!doctype html><html><head><meta charset='utf-8'><title>NEED checkpoint comparison</title>
<style>body{{font-family:system-ui,sans-serif;background:#0f1117;color:#e8e8ec;margin:2rem}} table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #303746;padding:8px;vertical-align:top}} th{{background:#171b24}} pre{{white-space:pre-wrap;margin:0}}</style></head>
<body><h1>NEED checkpoint comparison</h1><pre>{summary}</pre><table><tr><th>ID</th><th>Area</th><th>Prompt</th><th>A output</th><th>B output</th><th>Winner</th></tr>{''.join(rows)}</table></body></html>"""
    path.write_text(doc, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Compare two NEED checkpoints")
    p.add_argument("--checkpoint_a", required=True)
    p.add_argument("--checkpoint_b", required=True)
    p.add_argument("--prefer_best_a", action="store_true")
    p.add_argument("--prefer_best_b", action="store_true")
    p.add_argument("--prompts_jsonl", default="")
    p.add_argument("--out_dir", default="compare")
    p.add_argument("--out_json", default="")
    p.add_argument("--out_jsonl", default="")
    p.add_argument("--out_html", default="")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--kernel_backend", default="auto")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--no_repeat_ngram", type=int, default=4)
    p.add_argument("--aux_score_weight", type=float, default=0.35)
    p.add_argument("--aux_score_risk_threshold", type=float, default=0.72)
    p.add_argument("--aux_score_contradiction_threshold", type=float, default=0.65)
    p.add_argument("--disable_proactive_aux_score", action="store_true")
    args = p.parse_args(argv)
    out_dir = Path(args.out_dir)
    report = compare(args)
    out_json = Path(args.out_json) if args.out_json else out_dir / "checkpoint_compare.json"
    out_jsonl = Path(args.out_jsonl) if args.out_jsonl else out_dir / "checkpoint_compare.rows.jsonl"
    out_html = Path(args.out_html) if args.out_html else out_dir / "checkpoint_compare.html"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    write_jsonl(out_jsonl, report.get("rows", []))
    write_html(report, out_html)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
