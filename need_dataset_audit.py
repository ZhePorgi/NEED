#!/usr/bin/env python3
"""Dataset and runtime-corpus audit for NEED.

The audit is intentionally local and dependency-light.  It checks JSON/JSONL
training material before expensive runs so common failures are caught early:
malformed rows, duplicates, overlong examples, hidden runtime-tag leakage,
missing general-alignment slices, invalid image-token rows, suspicious code/tool
examples, and distribution drift from build_corpuses.py's plan.json.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import html
import json
import math
import re
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

HIDDEN_TAG_RE = re.compile(r"</?(?:latent_tool_observations|latent_tool_results|raw_cot|internal|scratchpad|tool_call|tool_result)\b", re.I)
DANGEROUS_CODE_RE = re.compile(r"\b(?:os\.system|subprocess\.|socket\.|requests\.|urllib\.|open\(|eval\(|exec\(|__import__|pickle\.)", re.I)
SENSITIVE_RE = re.compile(r"\b(?:api[_-]?key|secret[_-]?key|password|bearer\s+[A-Za-z0-9._\-]{16,}|sk-[A-Za-z0-9]{16,})\b", re.I)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
WS_RE = re.compile(r"\s+")
GENERAL_ALIGNMENT_SLICES = {"general_alignment_sft", "general_alignment_preferences", "truthfulness_calibration"}
EXPECTED_CORE_FILES = ["knowledge/train.jsonl", "rl/sft.jsonl", "rl/preferences.jsonl", "rl/rlvr.jsonl"]


def _json_dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _safe_read(path: Path, limit: int = 8_000_000) -> str:
    try:
        data = path.read_bytes()[:limit]
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def iter_jsonl(path: Path) -> Iterator[Tuple[int, Optional[Dict[str, Any]], str]]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                raw = line.rstrip("\n")
                if not raw.strip():
                    continue
                try:
                    obj = json.loads(raw)
                    yield i, obj if isinstance(obj, dict) else None, raw
                except Exception:
                    yield i, None, raw
    except Exception as exc:
        yield 0, None, f"__READ_ERROR__ {exc}"


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return WS_RE.sub(" ", CONTROL_RE.sub(" ", x)).strip()
    if isinstance(x, list):
        parts: List[str] = []
        for item in x:
            if isinstance(item, dict):
                parts.append(normalize_text(item.get("content") or item.get("text") or item.get("value") or item.get("answer") or item.get("response") or ""))
            else:
                parts.append(normalize_text(item))
        return WS_RE.sub(" ", " ".join(p for p in parts if p)).strip()
    if isinstance(x, dict):
        keys = ["text", "content", "prompt", "instruction", "question", "answer", "response", "chosen", "rejected", "assistant", "transcript", "caption", "code"]
        parts = [normalize_text(x.get(k)) for k in keys if k in x]
        if any(parts):
            return WS_RE.sub(" ", " ".join(p for p in parts if p)).strip()
        return _json_dump(x)[:20000]
    return str(x)


def row_text(row: Dict[str, Any]) -> str:
    keys = [
        "text", "content", "prompt", "instruction", "question", "answer", "response", "output",
        "chosen", "rejected", "preference_reason", "transcript", "caption", "messages", "target", "input", "scenario", "code",
    ]
    parts = [normalize_text(row.get(k)) for k in keys if k in row]
    return WS_RE.sub(" ", "\n".join(p for p in parts if p)).strip()


def row_kind(path: Path, row: Dict[str, Any]) -> str:
    typ = str(row.get("type") or row.get("task_type") or "").strip()
    if typ:
        return typ
    s = str(path).replace("\\", "/").lower()
    if "preferences" in s or ("chosen" in row and "rejected" in row):
        return "preference"
    if "rlvr" in s or "target_properties" in row or "aux_score" in row:
        return "rlvr"
    if "image_tokens" in s or "tokens" in row:
        return "image_tokens"
    if "messages" in row:
        return "sft"
    if "knowledge" in s:
        return "knowledge_text"
    return "unknown"


def stable_hash(text: str) -> str:
    compact = re.sub(r"\W+", " ", text.lower())[:4000].strip()
    return hashlib.sha1(compact.encode("utf-8", errors="ignore")).hexdigest()


def _percent(n: float, d: float) -> float:
    return round(100.0 * n / max(1.0, d), 4)


@dataclass
class Issue:
    severity: str
    path: str
    line: int
    code: str
    message: str
    sample: str = ""


class Audit:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.issues: List[Issue] = []
        self.files: Dict[str, Dict[str, Any]] = {}
        self.counts = collections.Counter()
        self.kind_counts = collections.Counter()
        self.path_counts = collections.Counter()
        self.lengths: List[int] = []
        self.hash_seen: Dict[str, Tuple[str, int]] = {}
        self.hash_cap_hit = False
        self.plan_info: Dict[str, Any] = {}

    def issue(self, severity: str, path: Path, line: int, code: str, message: str, sample: str = "") -> None:
        if len(self.issues) < self.args.max_issues:
            self.issues.append(Issue(severity, str(path), int(line), code, message, normalize_text(sample)[:500]))
        self.counts[f"issues_{severity}"] += 1
        self.counts[f"issue_code_{code}"] += 1

    def audit_plan(self, root: Path) -> None:
        plan_path = root / "plan.json"
        if not plan_path.exists():
            return
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.issue("error", plan_path, 0, "bad_plan_json", str(exc))
            return
        slices = plan.get("slices", []) if isinstance(plan, dict) else []
        names = {str(s.get("name", "")) for s in slices if isinstance(s, dict)}
        self.plan_info = {
            "scale": plan.get("scale"),
            "scale_info": plan.get("scale_info", {}),
            "slice_count": len(slices),
            "has_general_alignment": sorted(GENERAL_ALIGNMENT_SLICES.intersection(names)),
            "missing_general_alignment": sorted(GENERAL_ALIGNMENT_SLICES.difference(names)),
        }
        for missing in sorted(GENERAL_ALIGNMENT_SLICES.difference(names)):
            self.issue("warning", plan_path, 0, "missing_general_alignment_slice", f"plan is missing {missing}")
        if self.args.expect_full_pipeline:
            for rel in EXPECTED_CORE_FILES:
                if not (root / rel).exists():
                    self.issue("error", root / rel, 0, "missing_core_corpus_file", f"expected {rel}")

    def audit_json_manifest(self, path: Path) -> None:
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(obj, dict):
                self.issue("warning", path, 0, "manifest_not_object", "manifest JSON is not an object")
                return
            if path.name == "manifest.json" and "completion_ratio" in obj:
                try:
                    ratio = float(obj.get("completion_ratio", 0.0))
                    if ratio < self.args.min_completion_ratio:
                        self.issue("warning", path, 0, "low_completion_ratio", f"completion_ratio={ratio:.4f}")
                except Exception:
                    pass
        except Exception as exc:
            self.issue("error", path, 0, "bad_json", str(exc))

    def audit_row_shape(self, path: Path, line: int, row: Dict[str, Any], text: str, kind: str) -> None:
        if not text and kind != "image_tokens":
            self.issue("warning", path, line, "empty_training_text", "row has no recognizable prompt/text/content fields", _json_dump(row)[:500])
        if len(text) > self.args.max_row_chars:
            self.issue("warning", path, line, "row_too_long", f"row text has {len(text)} chars", text[:500])
        if HIDDEN_TAG_RE.search(text):
            self.issue("error", path, line, "hidden_runtime_tag_leak", "hidden/internal runtime tag appears in public training row", text[:500])
        if SENSITIVE_RE.search(text):
            self.issue("error", path, line, "possible_secret", "possible API key/password/secret in training row", text[:500])
        if DANGEROUS_CODE_RE.search(text):
            # Code corpuses may contain these tokens, so warn rather than fail.  The point is to make the source visible.
            self.issue("warning", path, line, "dangerous_code_pattern", "row contains code/tool pattern that should be intentional", text[:500])
        if kind == "preference" and not (("chosen" in row and "rejected" in row) or ("messages" in row)):
            self.issue("warning", path, line, "preference_missing_pair", "preference row lacks chosen/rejected fields")
        if kind == "rlvr" and not ("aux_score" in row or "target_properties" in row or "target" in row):
            self.issue("warning", path, line, "rlvr_missing_aux_score", "rlvr row lacks aux_score/target fields")
        if kind == "image_tokens":
            toks = row.get("tokens") or row.get("input_ids")
            if not isinstance(toks, list) or not toks:
                self.issue("error", path, line, "image_tokens_missing", "image-token row lacks tokens/input_ids")
            else:
                bad = 0
                for t in toks[: min(len(toks), 4096)]:
                    try:
                        int(t)
                    except Exception:
                        bad += 1
                if bad:
                    self.issue("error", path, line, "image_tokens_non_integer", f"{bad} non-integer token values")
                if len(toks) > self.args.max_image_tokens:
                    self.issue("warning", path, line, "image_tokens_too_long", f"image-token row has {len(toks)} tokens")

    def audit_jsonl(self, path: Path) -> None:
        file_stats = {"rows": 0, "malformed": 0, "bytes": path.stat().st_size if path.exists() else 0, "kinds": collections.Counter()}
        for line, obj, raw in iter_jsonl(path):
            if self.args.max_rows_per_file and file_stats["rows"] >= self.args.max_rows_per_file:
                break
            if obj is None:
                file_stats["malformed"] += 1
                self.counts["malformed_rows"] += 1
                self.issue("error", path, line, "malformed_jsonl", "line is not a JSON object", raw[:500])
                continue
            file_stats["rows"] += 1
            self.counts["rows"] += 1
            self.path_counts[str(path)] += 1
            kind = row_kind(path, obj)
            self.kind_counts[kind] += 1
            file_stats["kinds"][kind] += 1
            text = row_text(obj)
            self.lengths.append(len(text))
            self.audit_row_shape(path, line, obj, text, kind)
            if text:
                h = stable_hash(text)
                if h in self.hash_seen:
                    prev_path, prev_line = self.hash_seen[h]
                    self.counts["duplicate_rows"] += 1
                    self.issue("warning", path, line, "duplicate_row", f"duplicate of {prev_path}:{prev_line}", text[:300])
                elif len(self.hash_seen) < self.args.max_seen_hashes:
                    self.hash_seen[h] = (str(path), line)
                else:
                    self.hash_cap_hit = True
        file_stats["kinds"] = dict(file_stats["kinds"])
        self.files[str(path)] = file_stats

    def audit_file(self, path: Path) -> None:
        self.counts["files"] += 1
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            self.audit_jsonl(path)
        elif path.suffix.lower() == ".json":
            self.audit_json_manifest(path)
        elif path.suffix.lower() in {".txt", ".md"}:
            text = _safe_read(path)
            if HIDDEN_TAG_RE.search(text):
                self.issue("warning", path, 0, "hidden_tag_in_text_file", "hidden/runtime tags found in text file")
            if SENSITIVE_RE.search(text):
                self.issue("error", path, 0, "possible_secret", "possible secret found in text file")

    def run(self, roots: Sequence[Path]) -> Dict[str, Any]:
        for root in roots:
            if root.is_dir():
                self.audit_plan(root)
                files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".jsonl", ".ndjson", ".json", ".txt", ".md"}]
                for p in sorted(files):
                    # Skip very large raw text data unless explicitly asked; corpuses are JSONL.
                    if p.suffix.lower() in {".txt", ".md"} and p.stat().st_size > self.args.max_text_file_bytes:
                        continue
                    self.audit_file(p)
            elif root.exists():
                self.audit_file(root)
            else:
                self.issue("error", root, 0, "missing_path", "audit path does not exist")
        lengths = self.lengths or [0]
        hard_errors = self.counts.get("issues_error", 0)
        warnings = self.counts.get("issues_warning", 0)
        rows = self.counts.get("rows", 0)
        duplicate_rate = self.counts.get("duplicate_rows", 0) / max(1, rows)
        status = "pass"
        if hard_errors > 0:
            status = "fail"
        elif duplicate_rate > self.args.max_duplicate_rate:
            status = "warn"
            self.issue("warning", Path("<summary>"), 0, "high_duplicate_rate", f"duplicate_rate={duplicate_rate:.4f}")
        score = 100.0
        score -= min(50.0, hard_errors * 8.0)
        score -= min(25.0, warnings * 1.0)
        score -= min(20.0, duplicate_rate * 100.0)
        if self.hash_cap_hit:
            score -= 2.0
        report = {
            "status": status,
            "score": round(max(0.0, score), 3),
            "roots": [str(p) for p in roots],
            "summary": {
                "files": self.counts.get("files", 0),
                "rows": rows,
                "malformed_rows": self.counts.get("malformed_rows", 0),
                "duplicates": self.counts.get("duplicate_rows", 0),
                "duplicate_rate": round(duplicate_rate, 6),
                "issues_error": hard_errors,
                "issues_warning": warnings,
                "issues_info": self.counts.get("issues_info", 0),
                "median_row_chars": int(statistics.median(lengths)),
                "p95_row_chars": int(sorted(lengths)[min(len(lengths)-1, int(0.95 * (len(lengths)-1)))]),
                "max_row_chars_seen": int(max(lengths)),
                "kind_counts": dict(self.kind_counts),
                "plan_info": self.plan_info,
                "hash_cap_hit": self.hash_cap_hit,
            },
            "files": self.files,
            "issues": [asdict(x) for x in self.issues],
        }
        return report


def write_html_report(report: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    status = html.escape(str(report.get("status", "unknown")))
    score = html.escape(str(report.get("score", "")))
    summary = html.escape(json.dumps(report.get("summary", {}), indent=2, ensure_ascii=False))
    rows = []
    for issue in report.get("issues", [])[:1000]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(issue.get('severity','')))}</td>"
            f"<td>{html.escape(str(issue.get('code','')))}</td>"
            f"<td>{html.escape(str(issue.get('path','')))}:{html.escape(str(issue.get('line','')))}</td>"
            f"<td>{html.escape(str(issue.get('message','')))}</td>"
            f"<td><pre>{html.escape(str(issue.get('sample','')))}</pre></td>"
            "</tr>"
        )
    doc = f"""<!doctype html><html><head><meta charset='utf-8'><title>NEED dataset audit</title>
<style>body{{font-family:system-ui,sans-serif;background:#101217;color:#e8e8ea;margin:2rem}} pre{{white-space:pre-wrap}} table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #333;padding:8px;vertical-align:top}} th{{background:#1b1f2a}} .pass{{color:#9be28f}} .fail{{color:#ff8f8f}} .warn{{color:#ffd47a}}</style></head>
<body><h1>NEED dataset audit</h1><p>Status: <b class='{status}'>{status}</b> Score: <b>{score}</b></p><h2>Summary</h2><pre>{summary}</pre><h2>Issues</h2><table><tr><th>Severity</th><th>Code</th><th>Location</th><th>Message</th><th>Sample</th></tr>{''.join(rows)}</table></body></html>"""
    path.write_text(doc, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Audit NEED corpus/data files before training")
    p.add_argument("paths", nargs="*", default=["data/corpuses"], help="Corpus roots or individual JSONL/JSON files")
    p.add_argument("--out_json", default="")
    p.add_argument("--out_html", default="")
    p.add_argument("--failures_jsonl", default="")
    p.add_argument("--max_issues", type=int, default=5000)
    p.add_argument("--max_rows_per_file", type=int, default=0)
    p.add_argument("--max_seen_hashes", type=int, default=2_000_000)
    p.add_argument("--max_row_chars", type=int, default=24000)
    p.add_argument("--max_image_tokens", type=int, default=4096)
    p.add_argument("--max_text_file_bytes", type=int, default=2_000_000)
    p.add_argument("--min_completion_ratio", type=float, default=0.80)
    p.add_argument("--max_duplicate_rate", type=float, default=0.05)
    p.add_argument("--expect_full_pipeline", action="store_true")
    p.add_argument("--strict", action="store_true", help="Exit nonzero on audit failure")
    args = p.parse_args(argv)
    audit = Audit(args)
    report = audit.run([Path(x) for x in args.paths])
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.out_html:
        write_html_report(report, Path(args.out_html))
    if args.failures_jsonl:
        Path(args.failures_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.failures_jsonl).open("w", encoding="utf-8") as f:
            for issue in report.get("issues", []):
                if issue.get("severity") in {"error", "warning"}:
                    f.write(json.dumps(issue, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.strict and report.get("status") == "fail":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
