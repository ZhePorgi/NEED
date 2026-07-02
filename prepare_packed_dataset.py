#!/usr/bin/env python3
"""Prepare source-balanced packed NEED datasets.

Input manifest JSON example:
{
  "sources": [
    {"name": "web", "path": "data/web.jsonl", "weight": 0.45, "domain": "general"},
    {"name": "code", "path": "data/code.jsonl", "weight": 0.20, "domain": "code"}
  ]
}

The script writes one binary token file per source and an index JSON that train.py
can consume with --packed_index for source-balanced sampling.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from need_core import ByteTokenizer, NeedConfig
from train import pack_text_tokens_to_bin, _packed_metadata_path, _file_sha256
try:
    from config_for_size import parse_scaled_number
except Exception:
    def parse_scaled_number(v, default_suffix=""):
        s = str(v).strip().upper()
        mul = 1
        if s.endswith("K"):
            mul = 1_000; s = s[:-1]
        elif s.endswith("M"):
            mul = 1_000_000; s = s[:-1]
        elif s.endswith("B"):
            mul = 1_000_000_000; s = s[:-1]
        elif s.endswith("T"):
            mul = 1_000_000_000_000; s = s[:-1]
        return int(float(s) * mul)


def _load_manifest(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("sources"), list):
        raise ValueError("manifest must be a JSON object with a sources list")
    return data


def _safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip())
    return name or "source"


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Build source-balanced packed token files and NEED packed index")
    p.add_argument("--manifest", required=True, help="JSON manifest with sources: name, path, weight, optional domain/max_bytes")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--index_out", default="", help="Default: out_dir/packed_index.json")
    p.add_argument("--dtype", default="auto", choices=["auto", "uint16", "u16", "int32", "i32", "uint32", "u32"])
    p.add_argument("--vocab_size", type=int, default=NeedConfig().vocab_size)
    p.add_argument("--target_tokens", type=parse_scaled_number, default=0, help="Optional total token budget for proportional byte caps")
    p.add_argument("--max_bytes", type=parse_scaled_number, default=0, help="Optional global byte cap if source caps are omitted")
    p.add_argument("--add_eos", action="store_true", default=True)
    p.add_argument("--no_add_eos", dest="add_eos", action="store_false")
    args = p.parse_args(argv)

    manifest_path = Path(args.manifest)
    manifest = _load_manifest(manifest_path)
    sources = manifest["sources"]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tok = ByteTokenizer()
    raw_weights = [max(0.0, float(s.get("weight", 1.0))) for s in sources]
    wsum = sum(raw_weights) or float(len(raw_weights) or 1)
    norm_weights = [w / wsum for w in raw_weights]
    packed_sources: List[Dict[str, object]] = []
    total_tokens = 0
    total_bytes = 0

    # There is no exact byte-to-token conversion before tokenization.  For byte-level
    # NEED text, bytes and tokens are close, so proportional max_bytes is a good cap.
    token_budget = int(args.target_tokens or 0)
    global_byte_cap = int(args.max_bytes or 0)
    if token_budget > 0 and global_byte_cap <= 0:
        global_byte_cap = token_budget

    for src, weight in zip(sources, norm_weights):
        if not isinstance(src, dict):
            raise ValueError("each source must be an object")
        name = _safe_name(str(src.get("name", Path(str(src.get("path", "source"))).stem)))
        path = Path(str(src.get("path", ""))).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"source path does not exist: {path}")
        path = path.resolve()
        source_cap = int(src.get("max_bytes", 0) or 0)
        if source_cap <= 0 and global_byte_cap > 0:
            source_cap = max(1, int(global_byte_cap * weight))
        dtype_tag = str(args.dtype if args.dtype != "auto" else ("uint16" if int(args.vocab_size) <= 65535 else "uint32")).replace("u16", "uint16").replace("u32", "uint32").replace("i32", "int32")
        out_file = out_dir / f"{name}.{dtype_tag}.bin"
        meta = pack_text_tokens_to_bin(
            path, out_file, tok, vocab_size=int(args.vocab_size), dtype_name=args.dtype,
            add_eos=bool(args.add_eos), max_bytes=source_cap,
        )
        tokens = int(meta.get("tokens", 0) or 0)
        total_tokens += tokens
        total_bytes += int(meta.get("bytes_seen", 0) or 0)
        packed_sources.append({
            "name": name,
            "domain": str(src.get("domain", name)),
            "path": str(out_file.resolve()),
            "metadata": str(_packed_metadata_path(out_file).resolve()),
            "sha256": _file_sha256(out_file),
            "file_size_bytes": int(out_file.stat().st_size),
            "weight": float(src.get("weight", 1.0)),
            "normalized_weight": float(weight),
            "tokens": tokens,
            "bytes_seen": int(meta.get("bytes_seen", 0) or 0),
            "source_path": str(path),
        })
        print(json.dumps({"packed_source": name, "tokens": tokens, "weight": weight, "path": str(out_file), "sha256": _file_sha256(out_file)}), flush=True)

    index = {
        "format": "need_source_balanced_packed_index_v2",
        "manifest": str(manifest_path),
        "sources": packed_sources,
        "total_tokens": int(total_tokens),
        "total_bytes_seen": int(total_bytes),
        "target_tokens": int(token_budget),
        "vocab_size": int(args.vocab_size),
        "sampling": "weighted_random_source_then_random_block",
    }
    index_out = Path(args.index_out) if args.index_out else out_dir / "packed_index.json"
    index_out.parent.mkdir(parents=True, exist_ok=True)
    index_out.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"done": True, "index": str(index_out), "total_tokens": total_tokens, "sources": len(packed_sources)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
