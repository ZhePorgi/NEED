#!/usr/bin/env python3
"""Filter, deduplicate, score, and shard NEED corpus JSONL files.

Input rows may contain text, prompt/response pairs, messages, or preference pairs.
Output rows preserve the original schema where possible and add quality metadata.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence, Tuple

CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
WS_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+")
BAD_MARKERS = ("lorem ipsum", "javascript is disabled", "enable cookies", "access denied", "subscribe to continue", "cookie policy", "all rights reserved")


def open_text(path: Path, mode: str = "rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", errors="replace")
    return path.open(mode, encoding="utf-8", errors="replace")


def iter_jsonl(paths: Sequence[str]) -> Iterator[Tuple[Path, Dict[str, Any]]]:
    for pat in paths:
        for path in sorted(Path().glob(pat)) if any(c in pat for c in "*?[") else [Path(pat)]:
            if not path.exists() or path.is_dir():
                continue
            with open_text(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        row = {"text": line}
                    if isinstance(row, dict):
                        yield path, row


def normalize(text: str) -> str:
    text = CONTROL_RE.sub(" ", str(text)).replace("\u00a0", " ")
    return WS_RE.sub(" ", text).strip()


def stringify(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        parts = []
        for x in v:
            if isinstance(x, dict):
                role = x.get("role") or x.get("from") or x.get("speaker") or ""
                content = x.get("content") or x.get("value") or x.get("text") or ""
                if content:
                    parts.append((str(role) + ": " if role else "") + str(content))
            else:
                parts.append(stringify(x))
        return "\n".join(parts)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def row_text(row: Dict[str, Any]) -> str:
    fields = ["text", "content", "article", "abstract", "body", "title", "prompt", "instruction", "response", "answer", "chosen", "rejected", "messages"]
    parts = [stringify(row.get(f)) for f in fields if row.get(f) is not None]
    return normalize("\n\n".join(p for p in parts if p))


def token_estimate(text: str) -> int:
    words = len(re.findall(r"\S+", text))
    return max(1, int(max(words * 1.33, len(text) / 4.2)))


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def dedup_key(text: str, prefix_chars: int = 4000) -> str:
    compact = re.sub(r"\W+", " ", text[:prefix_chars].lower()).strip()
    return stable_hash(compact)


def repeated_line_ratio(text: str) -> float:
    lines = [x.strip() for x in text.splitlines() if len(x.strip()) > 20]
    if len(lines) < 4:
        return 0.0
    return 1.0 - len(set(lines)) / max(1, len(lines))


def char_ngram_repetition(text: str, n: int = 24) -> float:
    if len(text) < n * 10:
        return 0.0
    grams = [text[i:i+n] for i in range(0, min(len(text) - n, 8000), n)]
    return 1.0 - len(set(grams)) / max(1, len(grams))


def quality_score(text: str) -> Dict[str, float]:
    if not text:
        return {"quality_score": 0.0}
    alpha = sum(ch.isalpha() for ch in text) / max(1, len(text))
    digit = sum(ch.isdigit() for ch in text) / max(1, len(text))
    urls = len(URL_RE.findall(text))
    rep = max(repeated_line_ratio(text), char_ngram_repetition(text))
    bad = any(m in text.lower() for m in BAD_MARKERS)
    length_bonus = min(1.0, math.log(max(2, len(text))) / math.log(2500))
    score = 0.42 * alpha + 0.18 * length_bonus + 0.16 * (1.0 - min(1.0, rep)) + 0.10 * (1.0 - min(1.0, urls / 4)) + 0.08 * min(1.0, digit * 4.0) + (0.06 if not bad else -0.25)
    return {
        "quality_score": max(0.0, min(1.0, score)),
        "alpha_ratio": alpha,
        "digit_ratio": digit,
        "repetition_ratio": rep,
        "url_count": float(urls),
        "token_estimate": float(token_estimate(text)),
    }


def passes(text: str, stats: Dict[str, float], min_chars: int, max_chars: int, min_score: float) -> bool:
    if len(text) < min_chars or len(text) > max_chars:
        return False
    if stats["quality_score"] < min_score:
        return False
    if stats.get("repetition_ratio", 0.0) > 0.50:
        return False
    low = text.lower()
    if any(m in low for m in BAD_MARKERS):
        return False
    return True


def shard_path(out_dir: Path, base: str, shard_idx: int) -> Path:
    return out_dir / f"{base}-{shard_idx:05d}.jsonl"


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--base", default="filtered")
    p.add_argument("--min_chars", type=int, default=120)
    p.add_argument("--max_chars", type=int, default=60000)
    p.add_argument("--min_score", type=float, default=0.42)
    p.add_argument("--max_tokens", type=int, default=0, help="Stop after this many estimated tokens; 0 means no cap")
    p.add_argument("--shard_tokens", type=int, default=50_000_000)
    p.add_argument("--dedup_prefix_chars", type=int, default=4000)
    p.add_argument("--sample_prob", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args(argv)
    random.seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    seen = set(); counts = {"read": 0, "kept": 0, "duplicates": 0, "filtered": 0, "tokens": 0}
    shard_idx = 0; shard_tokens = 0
    out = shard_path(out_dir, args.base, shard_idx).open("w", encoding="utf-8")
    try:
        for src, row in iter_jsonl(args.input):
            counts["read"] += 1
            if random.random() > args.sample_prob:
                counts["filtered"] += 1; continue
            text = row_text(row)
            key = dedup_key(text, args.dedup_prefix_chars)
            if key in seen:
                counts["duplicates"] += 1; continue
            stats = quality_score(text)
            if not passes(text, stats, args.min_chars, args.max_chars, args.min_score):
                counts["filtered"] += 1; continue
            seen.add(key)
            toks = int(stats.get("token_estimate", 0))
            if args.shard_tokens and shard_tokens + toks > args.shard_tokens and shard_tokens > 0:
                out.close(); shard_idx += 1; shard_tokens = 0
                out = shard_path(out_dir, args.base, shard_idx).open("w", encoding="utf-8")
            row.setdefault("text", text if "text" not in row else row.get("text"))
            row["quality"] = {**stats, "source_file": str(src), "dedup_key": key}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["kept"] += 1; counts["tokens"] += toks; shard_tokens += toks
            if args.max_tokens and counts["tokens"] >= args.max_tokens:
                break
    finally:
        out.close()
    manifest = {**counts, "shards": shard_idx + 1, "base": args.base}
    (out_dir / f"{args.base}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
