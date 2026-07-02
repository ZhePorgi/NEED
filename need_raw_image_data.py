#!/usr/bin/env python3
"""Raw image data ingestion and tokenization for NEED.

This is the non-RL image-data lane: it prepares local/downloaded image corpuses,
filters noisy web manifests, tokenizes images with NEED's visual tokenizer, and
optionally starts NEED image-token pretraining immediately.

Supported sources are intentionally simple and auditable:
- local image folders
- JSON/JSONL manifests with path/image_path/url/caption/source/license fields
- URL text files
- parquet manifests when pandas/pyarrow are installed

The output is ordinary JSONL, not pickle, so it can be inspected or re-used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    from PIL import Image, ImageOps
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

from need_core import NeedConfig, Special, make_image_tokenizer, resolve_device
from need_image import load_visual_tokenizer, train_visual_tokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class ImageRecord:
    id: str
    path: str = ""
    url: str = ""
    caption: str = ""
    source: str = ""
    license: str = ""
    width: int = 0
    height: int = 0
    quality: float = 0.0
    sha256: str = ""
    nsfw: Optional[bool] = None
    watermark: Optional[bool] = None
    clip_score: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None


def _json_dump(row: Dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _iter_manifest(path: Path) -> Iterator[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        yield from _read_jsonl(path)
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        if isinstance(data, list):
            for obj in data:
                if isinstance(obj, dict):
                    yield obj
        elif isinstance(data, dict):
            rows = data.get("rows") or data.get("images") or data.get("data")
            if isinstance(rows, list):
                for obj in rows:
                    if isinstance(obj, dict):
                        yield obj
    elif suffix in {".txt", ".urls"}:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if s:
                    yield {"url": s}
    elif suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Reading parquet manifests requires pandas and pyarrow/fastparquet") from e
        df = pd.read_parquet(path)
        for obj in df.to_dict(orient="records"):
            yield {k: _clean_scalar(v) for k, v in obj.items()}
    else:
        raise ValueError(f"Unsupported manifest type: {path}")


def _clean_scalar(v: Any) -> Any:
    try:
        if hasattr(v, "item"):
            return v.item()
    except Exception:
        pass
    return v


def _first_str(row: Dict[str, Any], keys: Sequence[str]) -> str:
    for k in keys:
        v = row.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def _first_float(row: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    for k in keys:
        if k in row and row[k] is not None and str(row[k]).strip() != "":
            try:
                return float(row[k])
            except Exception:
                continue
    return None


def _first_bool(row: Dict[str, Any], keys: Sequence[str]) -> Optional[bool]:
    for k in keys:
        if k not in row:
            continue
        v = row.get(k)
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in {"true", "1", "yes", "y", "unsafe", "nsfw"}:
            return True
        if s in {"false", "0", "no", "n", "safe", "sfw"}:
            return False
    return None


def normalize_row(row: Dict[str, Any], base_dir: Optional[Path] = None) -> Dict[str, Any]:
    path = _first_str(row, ["path", "image_path", "local_path", "filepath", "file", "filename"])
    url = _first_str(row, ["url", "image_url", "URL", "TEXT", "image", "download_url"])
    caption = _first_str(row, ["caption", "text", "alt_text", "description", "prompt", "title"])
    source = _first_str(row, ["source", "dataset", "collection"])
    license_s = _first_str(row, ["license", "licence", "rights", "attribution"])
    if path and base_dir and not Path(path).is_absolute():
        path = str((base_dir / path).resolve())
    clip = _first_float(row, ["clip_score", "similarity", "clip_similarity", "score"])
    nsfw = _first_bool(row, ["nsfw", "unsafe", "is_nsfw", "adult"])
    watermark = _first_bool(row, ["watermark", "has_watermark", "watermarked"])
    rid = _first_str(row, ["id", "uid", "key", "sha256"])
    if not rid:
        rid = hashlib.sha256((path or url or caption).encode("utf-8", errors="ignore")).hexdigest()[:16]
    meta = {k: v for k, v in row.items() if k not in {"path", "image_path", "local_path", "filepath", "file", "filename", "url", "image_url", "caption", "text", "alt_text", "description", "prompt", "title"}}
    return {
        "id": rid,
        "path": path,
        "url": url,
        "caption": caption,
        "source": source,
        "license": license_s,
        "clip_score": clip,
        "nsfw": nsfw,
        "watermark": watermark,
        "meta": meta,
    }


def iter_source_rows(args: argparse.Namespace) -> Iterator[Dict[str, Any]]:
    if args.local_dir:
        root = Path(args.local_dir)
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                yield {"path": str(p.resolve()), "source": args.source_name or "local_folder"}
    if args.manifest:
        mp = Path(args.manifest)
        base = Path(args.manifest_base_dir) if args.manifest_base_dir else mp.parent
        for row in _iter_manifest(mp):
            yield normalize_row(row, base)
    if args.urls_file:
        for row in _iter_manifest(Path(args.urls_file)):
            row["source"] = row.get("source") or args.source_name or "urls_file"
            yield normalize_row(row, None)


def _safe_ext_from_url(url: str) -> str:
    stem = url.split("?")[0].split("#")[0].lower()
    for ext in IMAGE_EXTS:
        if stem.endswith(ext):
            return ext
    return ".jpg"


def download_image(url: str, out_dir: Path, timeout: int = 20, retries: int = 2) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(url.encode("utf-8", errors="ignore")).hexdigest() + _safe_ext_from_url(url)
    dest = out_dir / name
    if dest.exists() and dest.stat().st_size > 0:
        return str(dest.resolve())
    headers = {"User-Agent": "NEED-raw-image-data/1.0"}
    last_err: Optional[Exception] = None
    for _ in range(max(1, int(retries))):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            if not data:
                raise ValueError("empty response")
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return str(dest.resolve())
        except Exception as e:
            last_err = e
            time.sleep(0.25)
    raise RuntimeError(f"download failed for {url}: {last_err}")


def image_sha256(path: Path, max_bytes: int = 32_000_000) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        remaining = max_bytes
        while remaining > 0:
            chunk = f.read(min(1 << 20, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def inspect_image(path: Path, resize_quality_side: int = 64) -> Tuple[int, int, float]:
    if Image is None or np is None:
        raise RuntimeError("Pillow and numpy are required for image inspection")
    with Image.open(path) as im:
        if ImageOps is not None:
            im = ImageOps.exif_transpose(im)
        w, h = im.size
        small = im.convert("RGB").resize((resize_quality_side, resize_quality_side))
    arr = np.asarray(small, dtype=np.float32) / 255.0
    variance = float(arr.var())
    gx = float(np.abs(arr[:, 1:] - arr[:, :-1]).mean())
    gy = float(np.abs(arr[1:, :] - arr[:-1, :]).mean())
    q = float(max(0.0, min(1.0, variance * 4.0 + (gx + gy) * 1.5)))
    return int(w), int(h), q


def accept_record(rec: Dict[str, Any], args: argparse.Namespace) -> Tuple[bool, str]:
    if args.drop_nsfw and rec.get("nsfw") is True:
        return False, "nsfw"
    if args.drop_watermarked and rec.get("watermark") is True:
        return False, "watermark"
    clip = rec.get("clip_score")
    if clip is not None and args.min_clip_score is not None and float(clip) < float(args.min_clip_score):
        return False, "clip_score"
    path = rec.get("path") or ""
    if not path:
        return False, "missing_path"
    pp = Path(path)
    if not pp.exists() or not pp.is_file():
        return False, "missing_file"
    try:
        w, h, q = inspect_image(pp)
    except Exception:
        return False, "unreadable"
    if w < args.min_width or h < args.min_height:
        return False, "too_small"
    aspect = max(w / max(1, h), h / max(1, w))
    if aspect > args.max_aspect_ratio:
        return False, "aspect"
    if q < args.min_quality:
        return False, "quality"
    rec["width"] = w
    rec["height"] = h
    rec["quality"] = q
    return True, "ok"


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rows = list(iter_source_rows(args))
    if args.shuffle:
        rng.shuffle(rows)
    if args.max_candidates and len(rows) > args.max_candidates:
        rows = rows[: args.max_candidates]
    accepted: List[Dict[str, Any]] = []
    seen_sha: set[str] = set()
    reject_counts: Dict[str, int] = {}
    for row in rows:
        rec = normalize_row(row, None)
        if rec.get("url") and (args.download or not rec.get("path")):
            try:
                rec["path"] = download_image(str(rec["url"]), img_dir, timeout=args.download_timeout, retries=args.download_retries)
            except Exception:
                reject_counts["download"] = reject_counts.get("download", 0) + 1
                continue
        if args.copy_local and rec.get("path"):
            src = Path(str(rec["path"]))
            if src.exists() and src.is_file():
                name = hashlib.sha256(str(src.resolve()).encode("utf-8", errors="ignore")).hexdigest() + src.suffix.lower()
                dest = img_dir / name
                if not dest.exists():
                    try:
                        shutil.copy2(src, dest)
                    except Exception:
                        pass
                if dest.exists():
                    rec["path"] = str(dest.resolve())
        ok, reason = accept_record(rec, args)
        if not ok:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
            continue
        try:
            sha = image_sha256(Path(str(rec["path"]))) if args.dedupe else ""
        except Exception:
            reject_counts["hash"] = reject_counts.get("hash", 0) + 1
            continue
        if sha and sha in seen_sha:
            reject_counts["duplicate"] = reject_counts.get("duplicate", 0) + 1
            continue
        if sha:
            seen_sha.add(sha)
            rec["sha256"] = sha
        accepted.append(rec)
        if args.max_images and len(accepted) >= args.max_images:
            break
    accepted.sort(key=lambda r: float(r.get("quality") or 0.0), reverse=True)
    manifest = out_dir / "raw_image_manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for rec in accepted:
            f.write(_json_dump(rec) + "\n")
    stats = {"accepted": len(accepted), "candidates": len(rows), "rejected": reject_counts, "manifest": str(manifest), "image_dir": str(img_dir)}
    (out_dir / "raw_image_manifest.stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    return stats


def iter_prepared_manifest(path: Path) -> Iterator[Dict[str, Any]]:
    for row in _iter_manifest(path):
        rec = normalize_row(row, None)
        # Preserve fields computed by prepare.
        for k in ("width", "height", "quality", "sha256"):
            if k in row:
                rec[k] = row[k]
        yield rec


def _load_tokenizer(args: argparse.Namespace):
    if args.visual_tokenizer:
        return load_visual_tokenizer(args.visual_tokenizer, resolve_device(args.device))
    cfg = NeedConfig(image_codebook_size=args.image_codebook_size, vocab_size=Special.text_vocab + args.image_codebook_size, image_grid=args.grid, image_min_grid=args.min_grid, image_max_grid=args.max_grid, image_max_tokens=args.max_image_tokens)
    return make_image_tokenizer(cfg)


def tokenize(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = _load_tokenizer(args)
    rows = list(iter_prepared_manifest(Path(args.manifest)))
    rng = random.Random(args.seed)
    if args.shuffle:
        rng.shuffle(rows)
    if args.max_images and len(rows) > args.max_images:
        rows = rows[: args.max_images]
    n_val = max(0, int(round(len(rows) * args.val_frac)))
    val_ids = set(range(n_val)) if not args.shuffle else set(range(n_val))
    train_path = out_dir / "image_tokens.train.jsonl"
    val_path = out_dir / "image_tokens.val.jsonl"
    rejects = 0
    counts = {"train": 0, "val": 0}
    train_f = train_path.open("w", encoding="utf-8")
    val_f = val_path.open("w", encoding="utf-8")
    try:
        for i, rec in enumerate(rows):
            path = rec.get("path") or ""
            if not path:
                rejects += 1
                continue
            try:
                ids, meta = tok.encode_image(path, add_special=True, grid=(args.force_grid or None))
            except TypeError:
                ids, meta = tok.encode_image(path, add_special=True)
            except Exception:
                rejects += 1
                continue
            if len(ids) > args.max_sequence_tokens:
                # Keep BOS and the earliest visual tokens; EOS may be dropped by train.py anyway.
                ids = ids[: args.max_sequence_tokens]
            row = {
                "id": rec.get("id") or hashlib.sha256(str(path).encode()).hexdigest()[:16],
                "tokens": [int(x) for x in ids],
                "caption": rec.get("caption", ""),
                "source": rec.get("source", ""),
                "path": path if args.keep_paths else "",
                "meta": {
                    **(meta if isinstance(meta, dict) else {}),
                    "width": rec.get("width", 0),
                    "height": rec.get("height", 0),
                    "quality": rec.get("quality", 0.0),
                    "license": rec.get("license", ""),
                    "raw_sha256": rec.get("sha256", ""),
                },
            }
            split = "val" if i in val_ids else "train"
            (val_f if split == "val" else train_f).write(_json_dump(row) + "\n")
            counts[split] += 1
    finally:
        train_f.close(); val_f.close()
    manifest = {
        "train": str(train_path),
        "val": str(val_path),
        "counts": counts,
        "rejected": rejects,
        "visual_tokenizer": str(args.visual_tokenizer or "dynamic_fallback"),
        "format": "need_image_tokens_jsonl_v1",
    }
    (out_dir / "image_tokens.manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def start(args: argparse.Namespace) -> Dict[str, Any]:
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    prep_dir = out_root / "prepared"
    token_dir = out_root / "tokens"
    args_prepare = argparse.Namespace(**vars(args))
    args_prepare.out_dir = str(prep_dir)
    prep_stats = prepare(args_prepare)
    vt_path = args.visual_tokenizer
    if args.train_visual_tokenizer:
        vt_path = str(out_root / "visual_tokenizer")
        vq_image_dir = prep_stats["image_dir"] if (args.copy_local or args.download) else (args.local_dir or prep_stats["image_dir"])
        vq_args = argparse.Namespace(
            image_dir=vq_image_dir,
            out_dir=vt_path,
            device=args.device,
            seed=args.seed,
            codebook_size=args.image_codebook_size,
            embed_dim=args.vq_embed_dim,
            hidden_dim=args.vq_hidden_dim,
            num_res_blocks=args.vq_res_blocks,
            downsample=args.vq_downsample,
            grid=args.grid,
            min_grid=args.min_grid,
            max_grid=args.max_grid,
            max_image_tokens=args.max_image_tokens,
            batch_size=args.vq_batch_size,
            samples=args.vq_samples,
            steps=args.vq_steps,
            num_workers=args.num_workers,
            lr=args.vq_lr,
            weight_decay=args.vq_weight_decay,
            gan_weight=args.vq_gan_weight,
            gan_start=args.vq_gan_start,
            perceptual_weight=args.vq_perceptual_weight,
            edge_weight=args.vq_edge_weight,
            amp=args.amp,
            log_interval=args.log_interval,
            save_interval=args.vq_save_interval,
            curriculum=args.vq_curriculum,
            code_usage_weight=args.vq_code_usage_weight,
        )
        train_visual_tokenizer(vq_args)
    tok_args = argparse.Namespace(**vars(args))
    tok_args.manifest = prep_stats["manifest"]
    tok_args.out_dir = str(token_dir)
    tok_args.visual_tokenizer = vt_path
    token_stats = tokenize(tok_args)
    train_record: Optional[Dict[str, Any]] = None
    if args.train_need:
        need_out = out_root / "need_image_pretrain"
        cmd = [
            sys.executable, str(SCRIPT_DIR / "train.py"),
            "--image_tokens", token_stats["train"],
            "--out_dir", str(need_out),
            "--profile", args.need_profile,
            "--max_steps", str(args.need_steps),
            "--train_samples", str(args.need_train_samples),
            "--batch_size", str(args.need_batch_size),
            "--image_ratio", "1.0",
            "--image_mask_prob", str(args.image_mask_prob),
            "--device", args.device,
            "--kernel_backend", args.kernel_backend,
            "--seed", str(args.seed),
        ]
        if vt_path:
            cmd.extend(["--visual_tokenizer", vt_path])
        log_path = out_root / "need_image_pretrain.log"
        if args.dry_run:
            train_record = {"dry_run": True, "cmd": cmd, "log": str(log_path)}
        else:
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.run(cmd, cwd=str(SCRIPT_DIR), text=True, stdout=log, stderr=subprocess.STDOUT)
            train_record = {"cmd": cmd, "returncode": proc.returncode, "log": str(log_path), "out_dir": str(need_out)}
            if proc.returncode != 0:
                raise RuntimeError(f"NEED image pretraining failed; see {log_path}")
    manifest = {"prepared": prep_stats, "tokens": token_stats, "visual_tokenizer": vt_path, "train": train_record}
    (out_root / "raw_image_start_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def add_common_source_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--local_dir", default="", help="Local folder of raw images")
    p.add_argument("--manifest", default="", help="JSON/JSONL/parquet manifest with path or URL fields")
    p.add_argument("--manifest_base_dir", default="", help="Base directory for relative manifest paths")
    p.add_argument("--urls_file", default="", help="Text file containing one image URL per line")
    p.add_argument("--source_name", default="", help="Dataset/source label written into metadata")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--download", action="store_true", help="Download URL rows into out_dir/images")
    p.add_argument("--copy_local", action="store_true", help="Copy accepted local images into out_dir/images for stable training")
    p.add_argument("--download_timeout", type=int, default=20)
    p.add_argument("--download_retries", type=int, default=2)
    p.add_argument("--min_width", type=int, default=128)
    p.add_argument("--min_height", type=int, default=128)
    p.add_argument("--max_aspect_ratio", type=float, default=3.0)
    p.add_argument("--min_quality", type=float, default=0.02)
    p.add_argument("--min_clip_score", type=float, default=None)
    p.add_argument("--drop_nsfw", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--drop_watermarked", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max_candidates", type=int, default=0)
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=123)


def add_tokenize_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--manifest", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--visual_tokenizer", default="", help="Learned visual tokenizer directory. Omit to use dynamic fallback tokens.")
    p.add_argument("--device", default="auto")
    p.add_argument("--image_codebook_size", type=int, default=512)
    p.add_argument("--grid", type=int, default=16)
    p.add_argument("--min_grid", type=int, default=8)
    p.add_argument("--max_grid", type=int, default=32)
    p.add_argument("--max_image_tokens", type=int, default=1024)
    p.add_argument("--force_grid", type=int, default=0)
    p.add_argument("--max_sequence_tokens", type=int, default=1537)
    p.add_argument("--val_frac", type=float, default=0.02)
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--keep_paths", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=123)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare raw image data for NEED tokenization and image-token learning")
    sub = p.add_subparsers(dest="cmd", required=True)
    prep = sub.add_parser("prepare", help="Build a filtered image manifest from local images, URLs, or web manifests")
    add_common_source_args(prep)
    tok = sub.add_parser("tokenize", help="Encode a prepared image manifest into NEED image-token JSONL")
    add_tokenize_args(tok)
    st = sub.add_parser("start", help="Prepare, tokenize, and optionally start NEED image-token training")
    add_common_source_args(st)
    st.add_argument("--visual_tokenizer", default="")
    st.add_argument("--train_visual_tokenizer", action="store_true")
    st.add_argument("--device", default="auto")
    st.add_argument("--image_codebook_size", type=int, default=512)
    st.add_argument("--grid", type=int, default=16)
    st.add_argument("--min_grid", type=int, default=8)
    st.add_argument("--max_grid", type=int, default=32)
    st.add_argument("--max_image_tokens", type=int, default=1024)
    st.add_argument("--force_grid", type=int, default=0)
    st.add_argument("--max_sequence_tokens", type=int, default=1537)
    st.add_argument("--val_frac", type=float, default=0.02)
    st.add_argument("--keep_paths", action=argparse.BooleanOptionalAction, default=True)
    st.add_argument("--vq_embed_dim", type=int, default=128)
    st.add_argument("--vq_hidden_dim", type=int, default=128)
    st.add_argument("--vq_res_blocks", type=int, default=2)
    st.add_argument("--vq_downsample", type=int, default=16)
    st.add_argument("--vq_batch_size", type=int, default=8)
    st.add_argument("--vq_samples", type=int, default=10000)
    st.add_argument("--vq_steps", type=int, default=2000)
    st.add_argument("--vq_lr", type=float, default=2e-4)
    st.add_argument("--vq_weight_decay", type=float, default=1e-4)
    st.add_argument("--vq_gan_weight", type=float, default=0.05)
    st.add_argument("--vq_gan_start", type=int, default=1000)
    st.add_argument("--vq_perceptual_weight", type=float, default=0.10)
    st.add_argument("--vq_edge_weight", type=float, default=0.08)
    st.add_argument("--vq_save_interval", type=int, default=1000)
    st.add_argument("--vq_curriculum", action="store_true")
    st.add_argument("--vq_code_usage_weight", type=float, default=0.02)
    st.add_argument("--amp", choices=["off", "bf16", "fp16"], default="bf16")
    st.add_argument("--num_workers", type=int, default=0)
    st.add_argument("--log_interval", type=int, default=50)
    st.add_argument("--train_need", action="store_true")
    st.add_argument("--dry_run", action="store_true")
    st.add_argument("--need_profile", choices=["custom", "tiny", "small", "medium", "large", "reasoning", "image", "speed", "long_context", "multimodal", "agentic"], default="image")
    st.add_argument("--need_steps", type=int, default=1000)
    st.add_argument("--need_train_samples", type=int, default=10000)
    st.add_argument("--need_batch_size", type=int, default=4)
    st.add_argument("--image_mask_prob", type=float, default=0.35)
    st.add_argument("--kernel_backend", choices=["auto", "torch", "triton"], default="auto")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "tokenize":
        tokenize(args)
    elif args.cmd == "start":
        start(args)


if __name__ == "__main__":
    main()
