from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/torch_inductor_need"))
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from need_core import (
    NeedConfig, NeedModel, ByteTokenizer, HFTokenizer, Special, make_image_tokenizer,
    save_model, resolve_device, save_json, load_json, load_model, OBJECTIVE_LOSS_GROUPS,
    build_default_tokenizer, load_tokenizer_for_dir, load_tokenizer, DEFAULT_HF_TOKENIZER_MODEL,
)

try:
    from config_for_size import (
        format_scaled_number, model_config_for_params, parse_scaled_number, estimate_actual_params
    )
except Exception:  # pragma: no cover
    format_scaled_number = lambda n: str(int(n))  # type: ignore[assignment]
    model_config_for_params = None  # type: ignore[assignment]
    estimate_actual_params = None  # type: ignore[assignment]
    def parse_scaled_number(value, default_suffix=""):  # type: ignore[no-redef]
        return int(value)



try:
    from training_recipes import apply_training_recipe, available_recipes
except Exception:  # pragma: no cover
    apply_training_recipe = None  # type: ignore[assignment]
    available_recipes = lambda: {}  # type: ignore[assignment]

try:
    from need_image import (
        VisualTokenizerConfig, VisualTokenizerVQVAE, load_visual_tokenizer,
        save_visual_tokenizer, train_visual_tokenizer as train_vq_tokenizer
    )
except Exception:  # pragma: no cover
    VisualTokenizerConfig = None  # type: ignore[assignment]
    VisualTokenizerVQVAE = None  # type: ignore[assignment]
    load_visual_tokenizer = None  # type: ignore[assignment]
    save_visual_tokenizer = None  # type: ignore[assignment]
    train_vq_tokenizer = None  # type: ignore[assignment]

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


class TextChunkDataset(Dataset):
    def __init__(self, tokens: Sequence[int], block_size: int, predict_heads: int = 1, samples: int = 10000, seed: int = 123):
        self.tokens = np.asarray(tokens, dtype=np.int64)
        self.block_size = int(block_size)
        self.predict_heads = max(1, int(predict_heads))
        self.samples = int(samples)
        self.seed = int(seed)
        need = self.block_size + self.predict_heads
        if len(self.tokens) == 0:
            self.tokens = np.asarray([Special.bos, Special.eos], dtype=np.int64)
        if len(self.tokens) < need:
            reps = int(math.ceil(need / max(1, len(self.tokens))))
            self.tokens = np.tile(self.tokens, reps)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(self.seed + idx)
        max_start = len(self.tokens) - self.block_size - self.predict_heads
        start = rng.randint(0, max(0, max_start))
        x = self.tokens[start:start+self.block_size]
        ys = []
        for h in range(1, self.predict_heads + 1):
            ys.append(self.tokens[start+h:start+h+self.block_size])
        y = np.stack(ys, axis=-1)
        return {"input_ids": torch.tensor(x, dtype=torch.long), "targets": torch.tensor(y, dtype=torch.long)}





def _data_files(path: Path) -> List[Path]:
    if path.is_dir():
        files: List[Path] = []
        for pat in ("*.jsonl", "*.json", "*.txt", "*.text"):
            files.extend(sorted(path.rglob(pat)))
        return files
    return [path]


def _text_from_record_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    if line.startswith("{"):
        try:
            obj = json.loads(line)
        except Exception:
            return line
        if isinstance(obj, dict):
            for key in ("text", "content", "article", "abstract", "prompt", "response", "chosen"):
                val = obj.get(key)
                if isinstance(val, str) and val.strip():
                    return val
            parts = []
            for key in ("instruction", "input", "output"):
                val = obj.get(key)
                if isinstance(val, str) and val.strip():
                    parts.append(val)
            if parts:
                return "\n".join(parts)
    return line


class StreamingTextChunkDataset(IterableDataset):
    """Memory-safe text chunker for very large JSONL/text corpuses.

    It shards files across DataLoader workers and yields fixed NEED blocks without
    loading the corpus into host RAM, which is required for large token-budget runs.
    """
    def __init__(
        self,
        path: Path,
        tokenizer: ByteTokenizer,
        block_size: int,
        predict_heads: int = 1,
        seed: int = 123,
        max_bytes: int = 0,
        shuffle_files: bool = True,
        cycle: bool = True,
    ):
        files = _data_files(path)
        if not files:
            raise ValueError(f"No text files found in {path}")
        self.files = files
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        self.predict_heads = max(1, int(predict_heads))
        self.seed = int(seed)
        self.max_bytes = int(max_bytes or 0)
        self.shuffle_files = bool(shuffle_files)
        self.cycle = bool(cycle)

    def _make_item(self, ids: Sequence[int]) -> Dict[str, torch.Tensor]:
        x = np.asarray(ids[: self.block_size], dtype=np.int64)
        ys = []
        for h in range(1, self.predict_heads + 1):
            ys.append(np.asarray(ids[h:h+self.block_size], dtype=np.int64))
        y = np.stack(ys, axis=-1)
        return {
            "input_ids": torch.tensor(x, dtype=torch.long),
            "targets": torch.tensor(y, dtype=torch.long),
            "image_mask_positions": torch.zeros((self.block_size,), dtype=torch.bool),
            "image_targets": torch.full((self.block_size,), Special.pad, dtype=torch.long),
        }

    def __iter__(self):
        worker = get_worker_info()
        wid = worker.id if worker is not None else 0
        nw = worker.num_workers if worker is not None else 1
        rng = random.Random(self.seed + wid)
        files = list(self.files)
        if self.shuffle_files:
            rng.shuffle(files)
        buf: List[int] = []
        bytes_seen = 0
        need = self.block_size + self.predict_heads
        shard_by_file = len(files) >= max(1, nw)
        while True:
            made_progress = False
            for file_index, path in enumerate(files):
                if shard_by_file and file_index % nw != wid:
                    continue
                try:
                    with path.open("r", encoding="utf-8", errors="replace") as f:
                        for line_no, line in enumerate(f):
                            # Do not shard both files and lines.  With multiple files,
                            # each worker owns complete files; with fewer files than
                            # workers, shard lines so all workers still receive data.
                            if (not shard_by_file) and line_no % nw != wid:
                                continue
                            if self.max_bytes and bytes_seen >= self.max_bytes:
                                break
                            text = _text_from_record_line(line)
                            if not text:
                                continue
                            bytes_seen += len(text.encode("utf-8", errors="replace"))
                            buf.extend(self.tokenizer.encode(text, add_bos=False, add_eos=True))
                            made_progress = True
                            while len(buf) >= need:
                                ids = buf[:need]
                                del buf[: self.block_size]
                                yield self._make_item(ids)
                except FileNotFoundError:
                    continue
                if self.max_bytes and bytes_seen >= self.max_bytes:
                    break
            if not self.cycle or (self.max_bytes and bytes_seen >= self.max_bytes):
                # Pad one final partial chunk if useful, then stop.
                if len(buf) >= 2:
                    padded = (buf + [Special.pad] * need)[:need]
                    yield self._make_item(padded)
                return
            if not made_progress:
                raise RuntimeError("streaming dataset made no progress; check data path and worker sharding")
            if self.shuffle_files:
                rng.shuffle(files)



def _dtype_token_capacity(dtype: np.dtype) -> int:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.unsignedinteger):
        return int(np.iinfo(dtype).max)
    if np.issubdtype(dtype, np.signedinteger):
        return int(np.iinfo(dtype).max)
    raise ValueError(f"packed token dtype must be integer, got {dtype}")


def _numpy_dtype_for_tokens(name: str, vocab_size: int = 0) -> np.dtype:
    name = str(name or "auto").lower()
    if name == "auto":
        dtype = np.dtype(np.uint16 if int(vocab_size or 0) <= 65535 else np.uint32)
    elif name in ("u16", "uint16"):
        dtype = np.dtype(np.uint16)
    elif name in ("i32", "int32"):
        dtype = np.dtype(np.int32)
    elif name in ("u32", "uint32"):
        dtype = np.dtype(np.uint32)
    else:
        raise ValueError(f"unsupported packed token dtype: {name}")
    if int(vocab_size or 0) > 0 and int(vocab_size) - 1 > _dtype_token_capacity(dtype):
        raise ValueError(
            f"packed token dtype {dtype.name} cannot represent vocab_size={int(vocab_size)}; "
            "use --packed_dtype auto or uint32 to avoid token-id overflow"
        )
    return dtype


def _packed_metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _ensure_token_ids_in_vocab(ids: Any, vocab_size: int, context: str = "tokens") -> None:
    arr = np.asarray(ids)
    if arr.size == 0:
        return
    lo = int(arr.min())
    hi = int(arr.max())
    if lo < 0 or hi >= int(vocab_size):
        raise ValueError(
            f"{context} contains token IDs outside model vocab_size={int(vocab_size)}: min={lo}, max={hi}; "
            "check tokenizer sidecars, packed metadata, and checkpoint compatibility"
        )


def _packed_files(path: Path) -> List[Path]:
    if path.is_dir():
        files: List[Path] = []
        for pat in ("*.bin", "*.tokens", "*.tokbin"):
            files.extend(sorted(path.rglob(pat)))
        return files
    return [path]


def _packed_dtype_for_file(path: Path, requested: str, vocab_size: int) -> np.dtype:
    if requested and requested != "auto":
        return _numpy_dtype_for_tokens(requested, vocab_size)
    meta_path = _packed_metadata_path(path)
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        if isinstance(meta, dict) and meta:
            # Let dtype/vocab incompatibilities propagate. Falling back to an
            # inferred dtype after a metadata validation error can reinterpret the
            # binary stream incorrectly.
            return _numpy_dtype_for_tokens(str(meta.get("dtype", "auto")), int(meta.get("vocab_size", vocab_size)))
    return _numpy_dtype_for_tokens("auto", vocab_size)


def pack_text_tokens_to_bin(
    data_path: Path,
    out_path: Path,
    tokenizer: ByteTokenizer,
    *,
    vocab_size: int,
    dtype_name: str = "auto",
    add_eos: bool = True,
    max_bytes: int = 0,
    flush_tokens: int = 4_194_304,
) -> Dict[str, object]:
    """Tokenize JSONL/text into a flat binary array for high-MFU training.

    The output is a single contiguous stream of token ids.  Training can then form
    fixed blocks by slicing the memmap, avoiding JSON parsing and tokenizer work in
    DataLoader workers.  A sidecar ``.json`` records dtype and count.
    """
    files = _data_files(data_path)
    if not files:
        raise ValueError(f"No text files found in {data_path}")
    dtype = _numpy_dtype_for_tokens(dtype_name, vocab_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buf: List[int] = [Special.bos]
    tokens = 0
    bytes_seen = 0
    started = time.time()
    with out_path.open("wb") as f:
        def flush() -> None:
            nonlocal buf, tokens
            if not buf:
                return
            arr64 = np.asarray(buf, dtype=np.int64)
            _ensure_token_ids_in_vocab(arr64, int(vocab_size), "packed text tokens")
            arr = arr64.astype(dtype, copy=False)
            arr.tofile(f)
            tokens += int(arr.size)
            buf = []

        for fp in files:
            with fp.open("r", encoding="utf-8", errors="replace") as src:
                for line in src:
                    if max_bytes and bytes_seen >= max_bytes:
                        break
                    text = _text_from_record_line(line)
                    if not text:
                        continue
                    raw = text.encode("utf-8", errors="replace")
                    if max_bytes and bytes_seen + len(raw) > max_bytes:
                        raw = raw[: max(0, max_bytes - bytes_seen)]
                        text = raw.decode("utf-8", errors="replace")
                    bytes_seen += len(raw)
                    buf.extend(tokenizer.encode(text, add_bos=False, add_eos=add_eos))
                    if len(buf) >= flush_tokens:
                        flush()
                    if max_bytes and bytes_seen >= max_bytes:
                        break
            if max_bytes and bytes_seen >= max_bytes:
                break
        flush()
    meta = {
        "format": "need_flat_tokens_v1",
        "dtype": dtype.name,
        "tokens": int(tokens),
        "vocab_size": int(vocab_size),
        "source": str(data_path),
        "max_bytes": int(max_bytes or 0),
        "bytes_seen": int(bytes_seen),
        "seconds": float(time.time() - started),
    }
    _packed_metadata_path(out_path).write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return meta


class PackedTokenIterableDataset(IterableDataset):
    """Fixed-block iterable dataset backed by flat token memmaps.

    This is the preferred input path for long single-GPU pretraining: each worker
    reads contiguous slices from the binary token stream, with no tokenizer, JSON,
    Python string, or padding work on the hot path.
    """
    def __init__(
        self,
        path: Path,
        cfg: NeedConfig,
        predict_heads: int = 1,
        seed: int = 123,
        dtype_name: str = "auto",
        cycle: bool = True,
        shuffle_files: bool = True,
    ):
        files = _packed_files(path)
        if not files:
            raise ValueError(f"No packed token files found in {path}")
        self.files = files
        self.cfg = cfg
        self.block_size = int(cfg.block_size)
        self.predict_heads = max(1, int(predict_heads))
        self.seed = int(seed)
        self.dtype_name = str(dtype_name or "auto")
        self.cycle = bool(cycle)
        self.shuffle_files = bool(shuffle_files)

    def _array(self, path: Path) -> np.memmap:
        dtype = _packed_dtype_for_file(path, self.dtype_name, self.cfg.vocab_size)
        return np.memmap(path, mode="r", dtype=dtype)

    def _make_item(self, ids: np.ndarray) -> Dict[str, torch.Tensor]:
        x = np.asarray(ids[: self.block_size], dtype=np.int64)
        ys = []
        for h in range(1, self.predict_heads + 1):
            ys.append(np.asarray(ids[h:h+self.block_size], dtype=np.int64))
        y = np.stack(ys, axis=-1)
        return {
            "input_ids": torch.from_numpy(x),
            "targets": torch.from_numpy(y),
            "image_mask_positions": torch.zeros((self.block_size,), dtype=torch.bool),
            "image_targets": torch.full((self.block_size,), Special.pad, dtype=torch.long),
        }

    def __iter__(self):
        worker = get_worker_info()
        wid = worker.id if worker is not None else 0
        nw = worker.num_workers if worker is not None else 1
        rng = random.Random(self.seed + wid)
        files = list(self.files)
        need = self.block_size + self.predict_heads
        stride = self.block_size * max(1, nw)
        while True:
            if self.shuffle_files:
                rng.shuffle(files)
            made = False
            for path in files:
                arr = self._array(path)
                usable = int(arr.shape[0]) - need
                if usable < 0:
                    continue
                start = wid * self.block_size
                if start > usable:
                    continue
                while start <= usable:
                    ids = np.asarray(arr[start:start+need], dtype=np.int64)
                    _ensure_token_ids_in_vocab(ids, int(self.cfg.vocab_size), f"packed file {path}")
                    made = True
                    yield self._make_item(ids)
                    start += stride
            if not self.cycle:
                return
            if not made:
                raise RuntimeError("packed dataset made no progress; check packed token file length")


class PackedTokenMapDataset(Dataset):
    def __init__(self, path: Path, cfg: NeedConfig, predict_heads: int = 1, samples: int = 1024, seed: int = 123, dtype_name: str = "auto"):
        self.files = _packed_files(path)
        if not self.files:
            raise ValueError(f"No packed token files found in {path}")
        self.cfg = cfg
        self.block_size = int(cfg.block_size)
        self.predict_heads = max(1, int(predict_heads))
        self.samples = int(samples)
        self.seed = int(seed)
        self.dtype_name = str(dtype_name or "auto")
        self.arrays = [np.memmap(fp, mode="r", dtype=_packed_dtype_for_file(fp, self.dtype_name, cfg.vocab_size)) for fp in self.files]
        self.usable = [int(a.shape[0]) - (self.block_size + self.predict_heads) for a in self.arrays]
        if not any(u >= 0 for u in self.usable):
            raise ValueError("packed token files are too small for the configured block size")

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(self.seed + idx)
        choices = [i for i, u in enumerate(self.usable) if u >= 0]
        fi = choices[rng.randrange(len(choices))]
        start = rng.randint(0, self.usable[fi])
        need = self.block_size + self.predict_heads
        ids = np.asarray(self.arrays[fi][start:start+need], dtype=np.int64)
        _ensure_token_ids_in_vocab(ids, int(self.cfg.vocab_size), f"packed file {self.files[fi]}")
        x = np.asarray(ids[: self.block_size], dtype=np.int64)
        ys = []
        for h in range(1, self.predict_heads + 1):
            ys.append(np.asarray(ids[h:h+self.block_size], dtype=np.int64))
        y = np.stack(ys, axis=-1)
        return {
            "input_ids": torch.from_numpy(x),
            "targets": torch.from_numpy(y),
            "image_mask_positions": torch.zeros((self.block_size,), dtype=torch.bool),
            "image_targets": torch.full((self.block_size,), Special.pad, dtype=torch.long),
        }


class SourceBalancedPackedDataset(IterableDataset):
    """Weighted source-balanced packed-token sampler.

    The index JSON is produced by prepare_packed_dataset.py.  Each worker samples a
    source by manifest weight and then samples a fixed-length block from that
    source's memmap.  This keeps long runs source-balanced instead of consuming a
    giant concatenated stream in corpus order.
    """
    def __init__(self, index_path: Path, cfg: NeedConfig, predict_heads: int = 1, seed: int = 123, dtype_name: str = "auto", start_sample: int = 0):
        self.index_path = Path(index_path)
        raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not str(raw.get("format", "")).startswith("need_source_balanced"):
            raise ValueError(f"not a NEED source-balanced packed index: {index_path}")
        sources = raw.get("sources", [])
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"packed index has no sources: {index_path}")
        self.cfg = cfg
        self.block_size = int(cfg.block_size)
        self.predict_heads = max(1, int(predict_heads))
        self.seed = int(seed)
        self.dtype_name = str(dtype_name or "auto")
        self.start_sample = max(0, int(start_sample or 0))
        self.sources: List[Dict[str, Any]] = []
        weights: List[float] = []
        need = self.block_size + self.predict_heads
        for src in sources:
            if not isinstance(src, dict):
                continue
            fp = Path(str(src.get("path", ""))).expanduser()
            if not fp.is_absolute():
                fp = (self.index_path.parent / fp)
            if not fp.exists():
                continue
            fp = fp.resolve()
            dtype = _packed_dtype_for_file(fp, self.dtype_name, self.cfg.vocab_size)
            arr = np.memmap(fp, mode="r", dtype=dtype)
            usable = int(arr.shape[0]) - need
            if usable < 0:
                continue
            item = dict(src)
            item["_path"] = fp
            item["_dtype"] = dtype
            item["_usable"] = usable
            self.sources.append(item)
            weights.append(max(0.0, float(src.get("normalized_weight", src.get("weight", 1.0)) or 0.0)))
        if not self.sources:
            raise ValueError(f"packed index has no usable token files for block_size={self.block_size}")
        wsum = sum(weights)
        if wsum <= 0.0:
            self.weights = [1.0 / float(len(weights)) for _ in weights]
        else:
            self.weights = [w / wsum for w in weights]
        total = 0.0
        self.cdf: List[float] = []
        for w in self.weights:
            total += w
            self.cdf.append(total)
        self.cdf[-1] = 1.0

    def _choose_source(self, rng: random.Random) -> Dict[str, Any]:
        r = rng.random()
        for i, c in enumerate(self.cdf):
            if r <= c:
                return self.sources[i]
        return self.sources[-1]

    def _make_item(self, ids: np.ndarray, source: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        x = np.asarray(ids[: self.block_size], dtype=np.int64)
        ys = []
        for h in range(1, self.predict_heads + 1):
            ys.append(np.asarray(ids[h:h+self.block_size], dtype=np.int64))
        y = np.stack(ys, axis=-1)
        return {
            "input_ids": torch.from_numpy(x),
            "targets": torch.from_numpy(y),
            "image_mask_positions": torch.zeros((self.block_size,), dtype=torch.bool),
            "image_targets": torch.full((self.block_size,), Special.pad, dtype=torch.long),
        }

    def __iter__(self):
        worker = get_worker_info()
        wid = worker.id if worker is not None else 0
        nw = worker.num_workers if worker is not None else 1
        sample = self.start_sample + wid
        need = self.block_size + self.predict_heads
        arrays: Dict[str, np.memmap] = {}
        while True:
            rng = random.Random(self.seed + sample * 1_000_003 + wid)
            src = self._choose_source(rng)
            key = str(src["_path"])
            if key not in arrays:
                arrays[key] = np.memmap(src["_path"], mode="r", dtype=src["_dtype"])
            usable = int(src["_usable"])
            start = rng.randint(0, usable)
            ids = np.asarray(arrays[key][start:start+need], dtype=np.int64)
            _ensure_token_ids_in_vocab(ids, int(self.cfg.vocab_size), f"packed source {key}")
            yield self._make_item(ids, src)
            sample += max(1, nw)


class SourceBalancedPackedMapDataset(Dataset):
    """Small deterministic validation dataset from a source-balanced index."""
    def __init__(self, index_path: Path, cfg: NeedConfig, predict_heads: int = 1, samples: int = 1024, seed: int = 123, dtype_name: str = "auto"):
        self.inner = SourceBalancedPackedDataset(index_path, cfg, predict_heads, seed, dtype_name)
        self.samples = int(samples)
        self.seed = int(seed)
        self.cfg = cfg
        self.predict_heads = max(1, int(predict_heads))
        self.block_size = int(cfg.block_size)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(self.seed + idx)
        src = self.inner._choose_source(rng)
        arr = np.memmap(src["_path"], mode="r", dtype=src["_dtype"])
        need = self.block_size + self.predict_heads
        start = rng.randint(0, int(src["_usable"]))
        ids = np.asarray(arr[start:start+need], dtype=np.int64)
        _ensure_token_ids_in_vocab(ids, int(self.cfg.vocab_size), f"packed source {src.get('_path', '')}")
        return self.inner._make_item(ids, src)

class PreTokenizedImageDataset(Dataset):
    """Dataset for image-token JSONL produced by need_raw_image_data.py.

    Each row should contain either {"tokens": [...]} or {"input_ids": [...]}.
    Tokens are expected to include image special tokens when available.  This
    avoids re-encoding large raw image corpuses on every epoch.
    """
    def __init__(self, token_path: Path, cfg: NeedConfig, samples: int = 10000, mask_prob: float = 0.35, seed: int = 778):
        self.rows: List[List[int]] = []
        paths: List[Path] = []
        if token_path.is_dir():
            paths = sorted(token_path.glob("*.jsonl"))
        else:
            paths = [token_path]
        for p in paths:
            with p.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    toks = obj.get("tokens") if isinstance(obj, dict) else None
                    if toks is None and isinstance(obj, dict):
                        toks = obj.get("input_ids")
                    if isinstance(toks, list):
                        vals = []
                        for t in toks:
                            try:
                                vals.append(int(t))
                            except Exception:
                                pass
                        if vals:
                            self.rows.append(vals)
        if not self.rows:
            raise ValueError(f"No token rows found in {token_path}")
        self.cfg = cfg
        self.samples = int(samples)
        self.mask_prob = float(mask_prob)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(self.seed + idx)
        ids = list(self.rows[idx % len(self.rows)])
        ids = ids[: self.cfg.block_size + 1]
        if len(ids) < self.cfg.block_size + 1:
            ids = ids + [self.cfg.pad_id] * (self.cfg.block_size + 1 - len(ids))
        _ensure_token_ids_in_vocab(ids, int(self.cfg.vocab_size), "pre-tokenized image row")
        x = np.array(ids[:-1], dtype=np.int64)
        image_targets = x.copy()
        y = np.array(ids[1:], dtype=np.int64)
        mask_positions = np.zeros_like(x, dtype=np.bool_)
        for i, tok in enumerate(x):
            if self.cfg.image_token_offset <= int(tok) < self.cfg.image_token_offset + self.cfg.image_codebook_size and rng.random() < self.mask_prob:
                x[i] = self.cfg.img_mask_id
                mask_positions[i] = True
        return {"input_ids": torch.tensor(x), "targets": torch.tensor(y), "image_mask_positions": torch.tensor(mask_positions), "image_targets": torch.tensor(image_targets)}


class ImageTokenDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        cfg: NeedConfig,
        samples: int = 10000,
        mask_prob: float = 0.35,
        seed: int = 777,
        visual_tokenizer_path: str = "",
        visual_tokenizer_device: str = "cpu",
        force_grid: int = 0,
    ):
        if Image is None:
            raise RuntimeError("Pillow is required for image dataset")
        self.paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
            self.paths.extend(sorted(image_dir.rglob(ext)))
        if not self.paths:
            raise ValueError(f"No images found in {image_dir}")
        self.cfg = cfg
        self.samples = int(samples)
        self.mask_prob = float(mask_prob)
        self.seed = int(seed)
        self.force_grid = int(force_grid)
        self.visual_tokenizer_path = str(visual_tokenizer_path or "")
        if self.visual_tokenizer_path:
            if load_visual_tokenizer is None:
                raise RuntimeError("need_image.py is required for learned visual tokenizers")
            # CPU by default keeps DataLoader workers simple and avoids GPU contention.
            self.tok = load_visual_tokenizer(self.visual_tokenizer_path, device=resolve_device(visual_tokenizer_device))
        else:
            self.tok = make_image_tokenizer(cfg)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(self.seed + idx)
        path = self.paths[idx % len(self.paths)]
        # Learned VQ tokenizer and fallback tokenizer share encode_image(...).
        if self.visual_tokenizer_path:
            ids, meta = self.tok.encode_image(path, add_special=True, grid=(self.force_grid or None))
        else:
            ids, meta = self.tok.encode_image(path, add_special=True)
        # Fit within block size. Keep BOS + image tokens; EOS may be truncated.
        ids = ids[: self.cfg.block_size + 1]
        if len(ids) < self.cfg.block_size + 1:
            ids = ids + [self.cfg.pad_id] * (self.cfg.block_size + 1 - len(ids))
        _ensure_token_ids_in_vocab(ids, int(self.cfg.vocab_size), "pre-tokenized image row")
        x = np.array(ids[:-1], dtype=np.int64)
        image_targets = x.copy()
        y = np.array(ids[1:], dtype=np.int64)
        mask_positions = np.zeros_like(x, dtype=np.bool_)
        for i, tok in enumerate(x):
            if self.cfg.image_token_offset <= int(tok) < self.cfg.image_token_offset + self.cfg.image_codebook_size and rng.random() < self.mask_prob:
                x[i] = self.cfg.img_mask_id
                mask_positions[i] = True
        return {"input_ids": torch.tensor(x), "targets": torch.tensor(y), "image_mask_positions": torch.tensor(mask_positions), "image_targets": torch.tensor(image_targets)}

class MixedDataset(Dataset):
    def __init__(self, text_ds: Optional[Dataset], image_ds: Optional[Dataset], samples: int, image_ratio: float = 0.25, seed: int = 999):
        self.text_ds = text_ds
        self.image_ds = image_ds
        self.samples = samples
        self.image_ratio = image_ratio
        self.seed = seed
        if text_ds is None and image_ds is None:
            raise ValueError("At least one dataset is required")

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(self.seed + idx)
        use_img = self.image_ds is not None and (self.text_ds is None or rng.random() < self.image_ratio)
        if use_img:
            item = self.image_ds[idx % len(self.image_ds)]
            # image dataset yields [B,T] targets; expand for common collate with text MTP
            item["targets"] = item["targets"].unsqueeze(-1)
            return item
        item = self.text_ds[idx % len(self.text_ds)]  # type: ignore[index]
        item["image_mask_positions"] = torch.zeros(item["input_ids"].shape, dtype=torch.bool)
        item["image_targets"] = torch.full(item["input_ids"].shape, Special.pad, dtype=torch.long)
        return item


def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if k == "targets" and vals[0].ndim == 2:
            max_h = max(v.size(-1) for v in vals)
            padded = []
            for v in vals:
                if v.size(-1) < max_h:
                    pad = torch.full((v.size(0), max_h - v.size(-1)), Special.pad, dtype=v.dtype)
                    v = torch.cat([v, pad], dim=-1)
                padded.append(v)
            vals = padded
        out[k] = torch.stack(vals, dim=0)
    return out


def read_text_tokens(path: Path, tokenizer: ByteTokenizer, add_eos: bool = True, max_bytes: int = 0) -> List[int]:
    files = _data_files(path)
    ids: List[int] = [Special.bos]
    bytes_seen = 0
    for fp in files:
        try:
            with fp.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if max_bytes and bytes_seen >= max_bytes:
                        break
                    text = _text_from_record_line(line)
                    if not text:
                        continue
                    b = len(text.encode("utf-8", errors="replace"))
                    if max_bytes and bytes_seen + b > max_bytes:
                        remaining = max(0, max_bytes - bytes_seen)
                        text = text.encode("utf-8", errors="replace")[:remaining].decode("utf-8", errors="replace")
                        b = remaining
                    ids.extend(tokenizer.encode(text, add_bos=False, add_eos=add_eos))
                    bytes_seen += b
                    if max_bytes and bytes_seen >= max_bytes:
                        break
        except UnicodeDecodeError:
            data = fp.read_bytes()
            if max_bytes and bytes_seen + len(data) > max_bytes:
                data = data[: max(0, max_bytes - bytes_seen)]
            ids.extend(tokenizer.encode(data.decode("utf-8", errors="replace"), add_bos=False, add_eos=add_eos))
            bytes_seen += len(data)
        if max_bytes and bytes_seen >= max_bytes:
            break
    return ids


def explicit_arg_names(argv: Optional[Sequence[str]]) -> set[str]:
    raw = list(sys.argv[1:] if argv is None else argv)
    names: set[str] = set()
    for tok in raw:
        if not isinstance(tok, str) or not tok.startswith("--"):
            continue
        key = tok[2:].split("=", 1)[0]
        if key.startswith("no_"):
            key = key[3:]
        names.add(key.replace("-", "_"))
    return names


def apply_profile(args: argparse.Namespace) -> None:
    """Named architecture buckets intentionally removed.

    Size is now specified directly through explicit dimensions or generated from
    --target_params by the scale-neutral config search. This avoids hidden
    named architecture recipes that could make NEED look tuned to named
    scales instead of being parameter-budget driven.
    """
    profile = str(getattr(args, "profile", "custom") or "custom")
    if profile != "custom":
        raise ValueError("--profile named buckets were removed; use explicit dimension flags or --target_params instead")




def maybe_apply_target_model_size(args: argparse.Namespace) -> None:
    target = str(getattr(args, "target_params", "") or "").strip()
    if not target:
        return
    if model_config_for_params is None:
        raise RuntimeError("config_for_size.py is required for --target_params auto-sizing")
    target_count = parse_scaled_number(target)
    if target_count <= 0:
        raise ValueError("--target_params must be positive")
    arch = str(getattr(args, "architecture", "dense") or "dense").lower()
    cfg_dict, count = model_config_for_params(target_count, arch, block_size=int(getattr(args, "block_size", 0) or 0))
    for k, v in cfg_dict.items():
        if k == "moe_use_shared_expert":
            setattr(args, "disable_shared_expert", not bool(v))
        elif hasattr(args, k):
            setattr(args, k, v)
    setattr(args, "auto_sized_params", int(count))


def build_config(args: argparse.Namespace, text_vocab_size: Optional[int] = None) -> NeedConfig:
    apply_profile(args)
    if getattr(args, "recipe", "") and apply_training_recipe is not None:
        apply_training_recipe(args, args.recipe)
    maybe_apply_target_model_size(args)
    cfg = NeedConfig(
        image_codebook_size=args.image_codebook_size,
        text_vocab_size=int(text_vocab_size) if text_vocab_size else Special.text_vocab,
        block_size=args.block_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        conv_active_scales=args.conv_active_scales,
        retention_impl=args.retention_impl,
        ssd_conv_kernel=args.ssd_conv_kernel,
        adaptive_depth=not args.disable_adaptive_depth,
        compute_budget=args.compute_budget,
        min_compute_gate=args.min_compute_gate,
        depth_gate_temperature=args.depth_gate_temperature,
        cooperative_steps=not args.disable_cooperative_steps,
        cooperative_step_summary_dim=args.cooperative_step_summary_dim,
        cooperative_step_context_strength=args.cooperative_step_context_strength,
        cooperative_step_final_strength=args.cooperative_step_final_strength,
        cooperative_step_budget=args.cooperative_step_budget,
        cooperative_step_redundancy_target=args.cooperative_step_redundancy_target,
        role_separation=not args.disable_role_separation,
        role_separation_strength=args.role_separation_strength,
        memory_condition_strength=args.memory_condition_strength,
        n_experts=args.n_experts,
        moe_top_k=args.moe_top_k,
        moe_use_shared_expert=not args.disable_shared_expert,
        energy_rank=args.energy_rank,
        energy_steps=args.energy_steps,
        energy_min_steps=args.energy_min_steps,
        memory_slots=args.memory_slots,
        memory_rank=args.memory_rank,
        memory_mix=args.memory_mix,
        memory_chunk_size=args.memory_chunk_size,
        pathway_conditioning_top_k=args.pathway_conditioning_top_k,
        pathway_conditioning_dropout=args.pathway_conditioning_dropout,
        pathway_conditioning_scale=args.pathway_conditioning_scale,
        pathway_conditioning_max_vectors=args.pathway_conditioning_max_vectors,
        pathway_memory_slots=args.pathway_memory_slots,
        pathway_memory_top_k=args.pathway_memory_top_k,
        pathway_memory_update_rate=args.pathway_memory_update_rate,
        planner_horizons=args.planner_horizons,
        planner_transition_depth=args.planner_transition_depth,
        planner_block_space_enabled=not args.disable_planner_block_space,
        planner_block_space_mix=args.planner_block_space_mix,
        planner_block_space_iters=args.planner_block_space_iters,
        dvsd_planner_compound_enabled=not args.disable_dvsd_planner_compound,
        dvsd_planner_compound_mix=args.dvsd_planner_compound_mix,
        dvsd_planner_compound_step_size=args.dvsd_planner_compound_step_size,
        dvsd_planner_compound_token_scale=args.dvsd_planner_compound_token_scale,
        dvsd_planner_compound_descent_scale=args.dvsd_planner_compound_descent_scale,
        dvsd_planner_compound_top_k=args.dvsd_planner_compound_top_k,
        aux_score_candidate_pool=args.aux_score_candidate_pool,
        aux_score_risk_threshold=args.aux_score_risk_threshold,
        aux_score_contradiction_threshold=args.aux_score_contradiction_threshold,
        aux_score_backtrack_window=args.aux_score_backtrack_window,
        aux_score_max_backtracks=args.aux_score_max_backtracks,
        aux_score_controller=not args.disable_aux_score_controller,
        controller_temperature=args.controller_temperature,
        latent_search_branches=args.latent_search_branches,
        latent_search_depth=args.latent_search_depth,
        cot_faithfulness_threshold=args.cot_faithfulness_threshold,
        cot_usefulness_threshold=args.cot_usefulness_threshold,
        energy_routes=args.energy_routes,
        energy_route_steps=args.energy_route_steps,
        energy_route_strength=args.energy_route_strength,
        latent_slots=args.latent_slots,
        slot_attention_mode=args.slot_attention_mode,
        latent_slot_conditioning_scale=args.latent_slot_conditioning_scale,
        slow_state_chunk=args.slow_state_chunk,
        slow_state_strength=args.slow_state_strength,
        risk_gate_strength=args.risk_gate_strength,
        object_program_slots=args.object_program_slots,
        object_program_strength=args.object_program_strength,
        output_modes=args.output_modes,
        image_2d_scan=not args.disable_image_2d_scan,
        image_2d_bidirectional=bool(args.image_2d_bidirectional),
        image_2d_scan_strength=args.image_2d_scan_strength,
        image_2d_scan_decay=args.image_2d_scan_decay,
        region_word_sinkhorn_iters=args.region_word_sinkhorn_iters,
        image_coord_scale=args.image_coord_scale,
        image_local_contrastive_temperature=args.image_local_contrastive_temperature,
        kernel_backend=args.kernel_backend,
        fused_ssd_scan=not args.disable_fused_ssd_scan,
        parallel_scan=not args.disable_parallel_scan,
        collect_aux_metrics=not args.minimal_aux_metrics,
        strict_linear_core=not args.no_strict_linear_core,
        streaming_generation=not args.no_streaming_generation,
        exact_recall=not args.disable_exact_recall,
        exact_recall_dim=args.exact_recall_dim,
        exact_recall_top_k=args.exact_recall_top_k,
        exact_recall_mix=args.exact_recall_mix,
        exact_recall_temperature=args.exact_recall_temperature,
        exact_recall_max_tokens=args.exact_recall_max_tokens,
        exact_recall_max_candidates=args.exact_recall_max_candidates,
        state_stabilization=not args.disable_state_stabilization,
        state_anchor_strength=args.state_anchor_strength,
        state_drift_chunk=args.state_drift_chunk,
        state_drift_target=args.state_drift_target,
        state_norm_target=args.state_norm_target,
        objective_soft_budget=not args.disable_objective_soft_budget,
        objective_aux_ratio_cap=args.objective_aux_ratio_cap,
        objective_group_ratio_cap=args.objective_group_ratio_cap,
        objective_softcap_min=args.objective_softcap_min,
        objective_aux_warmup_steps=args.objective_aux_warmup_steps,
        objective_aux_min_scale=args.objective_aux_min_scale,
        objective_balance_ema_beta=args.objective_balance_ema_beta,
        objective_loss_ema_floor=args.objective_loss_ema_floor,
        objective_term_abs_cap=args.objective_term_abs_cap,
        objective_normalize_aux=not args.disable_objective_normalize_aux,
        objective_entropy_band_weight=args.objective_entropy_band_weight,
        objective_curriculum_enabled=not args.disable_objective_curriculum,
        objective_prediction_start_step=args.objective_prediction_start_step,
        objective_latent_start_step=args.objective_latent_start_step,
        objective_risk_start_step=args.objective_risk_start_step,
        objective_vision_start_step=args.objective_vision_start_step,
        objective_regularizer_start_step=args.objective_regularizer_start_step,
        objective_family_ramp_steps=args.objective_family_ramp_steps,
        objective_quarantine_enabled=not args.disable_objective_quarantine,
        objective_quarantine_patience=args.objective_quarantine_patience,
        objective_quarantine_decay=args.objective_quarantine_decay,
        objective_quarantine_min_scale=args.objective_quarantine_min_scale,
        objective_quarantine_recovery_steps=args.objective_quarantine_recovery_steps,
        objective_pathology_clip_threshold=args.objective_pathology_clip_threshold,
        objective_gradient_guard=bool(args.enable_objective_gradient_guard),
        objective_gradient_guard_interval=args.objective_gradient_guard_interval,
        objective_gradient_guard_start_step=args.objective_gradient_guard_start_step,
        objective_gradient_guard_max_terms=args.objective_gradient_guard_max_terms,
        objective_gradient_guard_param_tensors=args.objective_gradient_guard_param_tensors,
        objective_conflict_cosine_threshold=args.objective_conflict_cosine_threshold,
        objective_conflict_quarantine_patience=args.objective_conflict_quarantine_patience,
        image_grid=args.image_grid,
        image_min_grid=args.image_min_grid,
        image_max_grid=args.image_max_grid,
        image_max_tokens=args.image_max_tokens,
        dynamic_image_grid=not args.static_image_grid,
        dvsd_router_enabled=not args.disable_dvsd_router,
        dvsd_router_inference_mix=args.dvsd_router_inference_mix,
        dvsd_router_min_confidence=args.dvsd_router_min_confidence,
        dvsd_router_loss_threshold=args.dvsd_router_loss_threshold,
        dvsd_router_hard_loss_threshold=args.dvsd_router_hard_loss_threshold,
        dvsd_router_entropy_weight=args.dvsd_router_entropy_weight,
    )
    # Fused objective-family overrides plus optional component weights.
    for name in FUSED_LAMBDA_FIELDS:
        if hasattr(args, name):
            setattr(cfg, name, getattr(args, name))
    profile_component_weights = getattr(args, "aux_component_weights", None)
    if isinstance(profile_component_weights, dict):
        for component_name, value in profile_component_weights.items():
            cfg.set_aux_component_weight(str(component_name), float(value))

    # Ablations zero components or whole fused families.
    if args.disable_energy:
        cfg.disable_aux_components("equilibrium_residual", "energy", "energy_row_orth", "equilibrium_temporal_overlap")
        cfg.energy_steps = 1
    if args.disable_diffusion:
        cfg.disable_aux_components("diffusion", "image_diffusion")
    if args.disable_geodesic:
        cfg.disable_aux_components("geodesic", "path_straightness", "path_contractive")
    if args.disable_moe:
        cfg.n_experts = 1; cfg.moe_top_k = 1; cfg.moe_use_shared_expert = False
        cfg.disable_aux_components("moe_balance", "moe_router_z", "branch_entropy")
    if args.disable_memory:
        cfg.memory_mix = 0.0
        cfg.disable_aux_components("memory_entropy", "memory_diversity", "memory_retention_overlap", "pathway_memory_entropy", "state_drift", "state_anchor", "exact_recall_entropy_floor")
    if args.disable_planner:
        cfg.planner_horizons = 0
        cfg.dvsd_planner_compound_enabled = False
        cfg.disable_aux_components("latent_planning", "planner_ce", "planning_consistency", "dvsd_compound_latent", "dvsd_compound_ce", "dvsd_compound_consistency")
    if args.disable_aux_score:
        cfg.disable_aux_components("aux_score", "controller")
        cfg.aux_score_logit_scale = 0.0
    if args.disable_mtp:
        cfg.n_predict_heads = 1
        cfg.disable_aux_components("mtp", "dvsd_slot_ce", "dvsd_consistency", "dvsd_router")
    if args.disable_adaptive_compute:
        cfg.adaptive_energy = False; cfg.adaptive_depth = False
        cfg.disable_aux_components("adaptive_effort", "compute_budget")
    if args.disable_energy_routes:
        cfg.energy_routes = 1; cfg.energy_route_strength = 0.0
        cfg.disable_aux_components("mixture_energy_router_energy", "energy_route_entropy_band", "energy_route_balance")
    if args.disable_latent_slots:
        cfg.latent_slots = 1; cfg.latent_slot_conditioning_scale = 0.0
        cfg.disable_aux_components("latent_slot", "latent_slot_diversity", "latent_slot_entropy_band")
    if args.disable_risk_signal_fusion:
        cfg.disable_aux_components("risk_signal", "latent_divergence_loss")
        cfg.risk_gate_strength = 0.0
    if args.disable_image:
        cfg.lambda_vision_aux = 0.0
        cfg.disable_aux_components("image_diffusion", "image_contrastive", "image_local_contrastive", "region_word_alignment", "image_spatial_smoothness", "object_program", "object_slot_entropy_band")
        cfg.image_2d_scan = False; cfg.image_coord_scale = 0.0
    cfg.validate()
    return cfg


def configure_optimizer(model: torch.nn.Module, args: argparse.Namespace):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and not n.endswith("bias"):
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs = dict(lr=args.lr, betas=(args.beta1, args.beta2))
    if getattr(args, "optimizer_fused", False):
        kwargs["fused"] = True
    elif getattr(args, "optimizer_foreach", False):
        kwargs["foreach"] = True
    try:
        return torch.optim.AdamW(groups, **kwargs)
    except TypeError:
        kwargs.pop("fused", None); kwargs.pop("foreach", None)
        return torch.optim.AdamW(groups, **kwargs)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _objective_probe_params(model: torch.nn.Module, limit: int) -> List[torch.nn.Parameter]:
    """Sample shared trunk tensors for cheap gradient-conflict diagnostics.

    We deliberately avoid LM heads and auxiliary-only heads where possible.  The
    question is whether auxiliaries fight the shared representation learned by CE,
    not whether their private classifier layers disagree with the token head.
    """
    raw = _unwrap_model(model)
    blocked = (
        "objective_balancer", "lm_head", "mtp_projs", "aux_score",
        "output_mode_classifier", "text_proj", "image_proj", "dvsd_router",
    )
    preferred: List[torch.nn.Parameter] = []
    fallback: List[torch.nn.Parameter] = []
    for name, param in raw.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2:
            fallback.append(param)
            continue
        if any(b in name for b in blocked):
            fallback.append(param)
            continue
        preferred.append(param)
        if len(preferred) >= max(1, int(limit)):
            break
    if preferred:
        return preferred[: max(1, int(limit))]
    return fallback[: max(1, int(limit))]


def _grad_dot_norms(a: Sequence[Optional[torch.Tensor]], b: Sequence[Optional[torch.Tensor]]) -> Tuple[float, float, float]:
    dot = 0.0
    an = 0.0
    bn = 0.0
    for ga, gb in zip(a, b):
        if ga is not None:
            gaf = ga.detach().float()
            an += float(gaf.pow(2).sum().cpu())
        if gb is not None:
            gbf = gb.detach().float()
            bn += float(gbf.pow(2).sum().cpu())
        if ga is not None and gb is not None:
            dot += float((ga.detach().float() * gb.detach().float()).sum().cpu())
    return dot, an, bn


def maybe_probe_objective_gradients(
    model: torch.nn.Module,
    aux: Dict[str, Any],
    step: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Occasionally compare auxiliary gradients against CE gradients.

    Scalar loss balancing can stop domination, but it cannot tell whether two
    same-sized objectives point in opposing directions.  This probe computes
    cosine similarity on a limited sample of shared trunk tensors and lets the
    model's balancer quarantine terms that repeatedly oppose CE.
    """
    cfg_enabled = True
    raw_model = _unwrap_model(model)
    cfg = getattr(raw_model, "cfg", None)
    if cfg is not None:
        cfg_enabled = bool(getattr(cfg, "objective_gradient_guard", True))
    if not cfg_enabled or bool(getattr(args, "disable_objective_gradient_guard", False)):
        return {}
    start = int(getattr(args, "objective_gradient_guard_start_step", 0) or 0)
    interval = max(1, int(getattr(args, "objective_gradient_guard_interval", 50) or 50))
    if int(step) < start or (int(step) % interval) != 0:
        return {}
    ce = aux.get("_ce_objective")
    terms = aux.get("_objective_terms")
    if not torch.is_tensor(ce) or not isinstance(terms, dict) or not terms:
        return {}
    params = _objective_probe_params(model, int(getattr(args, "objective_gradient_guard_param_tensors", 24) or 24))
    if not params:
        return {}
    candidates: List[Tuple[float, str, torch.Tensor]] = []
    for name, term in terms.items():
        if not torch.is_tensor(term) or not term.requires_grad:
            continue
        metric = aux.get(f"{name}_objective_abs_contrib")
        try:
            score = float(metric.detach().float().cpu()) if torch.is_tensor(metric) else float(term.detach().float().abs().cpu())
        except Exception:
            score = 0.0
        if score <= 0.0 or not math.isfinite(score):
            continue
        candidates.append((score, str(name), term))
    if not candidates:
        return {}
    candidates.sort(reverse=True, key=lambda x: x[0])
    candidates = candidates[: max(1, int(getattr(args, "objective_gradient_guard_max_terms", 8) or 8))]
    try:
        ce_grads = torch.autograd.grad(ce.float(), params, retain_graph=True, allow_unused=True)
    except RuntimeError:
        return {}
    conflicts: Dict[str, float] = {}
    min_cos = 1.0
    max_neg = 0
    for _, name, term in candidates:
        try:
            aux_grads = torch.autograd.grad(term.float(), params, retain_graph=True, allow_unused=True)
        except RuntimeError:
            continue
        dot, ce_norm, aux_norm = _grad_dot_norms(ce_grads, aux_grads)
        if ce_norm <= 0.0 or aux_norm <= 0.0:
            continue
        cos = dot / math.sqrt(max(1e-30, ce_norm * aux_norm))
        if math.isfinite(cos):
            conflicts[name] = float(cos)
            min_cos = min(min_cos, float(cos))
            if cos < float(getattr(args, "objective_conflict_cosine_threshold", -0.10)):
                max_neg += 1
    applied: Dict[str, float] = {}
    balancer = getattr(raw_model, "objective_balancer", None)
    if balancer is not None and hasattr(balancer, "record_gradient_conflicts"):
        try:
            applied = balancer.record_gradient_conflicts(conflicts)
        except Exception:
            applied = {}
    out: Dict[str, float] = {
        "objective_grad_probe_terms": float(len(conflicts)),
        "objective_grad_conflict_terms": float(max_neg),
    }
    if conflicts:
        out["objective_grad_min_cosine"] = float(min_cos)
        out["objective_grad_mean_cosine"] = float(sum(conflicts.values()) / max(1, len(conflicts)))
    if applied:
        out["objective_grad_quarantines"] = float(len(applied))
        out["objective_grad_min_applied_scale"] = float(min(applied.values()))
    return out


def _lr_for_optimizer_step(args: argparse.Namespace, opt_step: int, total_opt_steps: int) -> float:
    base = float(args.lr)
    schedule = str(getattr(args, "lr_schedule", "constant") or "constant").lower()
    warmup = max(0, int(getattr(args, "warmup_steps", 0) or 0))
    if warmup > 0 and opt_step < warmup:
        return base * float(opt_step + 1) / float(max(1, warmup))
    if schedule in ("constant", "none", "off"):
        return base
    min_lr = float(getattr(args, "min_lr", 0.0) or 0.0)
    decay_steps = int(getattr(args, "lr_decay_steps", 0) or 0)
    if decay_steps <= 0:
        decay_steps = max(warmup + 1, int(total_opt_steps or 0))
    if schedule == "linear":
        t = min(1.0, max(0.0, float(opt_step - warmup) / float(max(1, decay_steps - warmup))))
        return base + (min_lr - base) * t
    if schedule == "cosine":
        t = min(1.0, max(0.0, float(opt_step - warmup) / float(max(1, decay_steps - warmup))))
        return min_lr + 0.5 * (base - min_lr) * (1.0 + math.cos(math.pi * t))
    raise ValueError(f"unknown lr_schedule: {schedule}")


def _set_optimizer_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for group in opt.param_groups:
        group["lr"] = float(lr)


def estimate_mfu_tokens_per_sec(tokens: int, elapsed: float) -> float:
    return float(tokens) / max(elapsed, 1e-9)


def estimate_flops_per_token(param_count: int) -> float:
    # A training-token approximation.  The model is non-attention NEED rather
    # than an external LM, but 6 * params is a useful MFU proxy for optimizer logs.
    return float(max(1, int(param_count))) * 6.0


def _ram_available_gb() -> float:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / (1024.0 * 1024.0)
    except Exception:
        pass
    return 0.0


def configure_runtime(args: argparse.Namespace, device: torch.device) -> None:
    if device.type == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
            torch.backends.cudnn.allow_tf32 = bool(args.tf32)
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision(str(args.matmul_precision))
        except Exception:
            pass
        if getattr(args, "cudnn_benchmark", False):
            torch.backends.cudnn.benchmark = True
        if getattr(args, "compile_cudagraphs", False):
            try:
                import torch._inductor.config as inductor_config  # type: ignore
                inductor_config.triton.cudagraphs = True
                if hasattr(inductor_config.triton, "cudagraph_trees"):
                    inductor_config.triton.cudagraph_trees = True
                if hasattr(inductor_config, "max_autotune_gemm_backends"):
                    inductor_config.max_autotune_gemm_backends = "TRITON,ATEN"
            except Exception:
                pass
        try:
            import torch._dynamo.config as dynamo_config  # type: ignore
            dynamo_config.cache_size_limit = max(int(getattr(dynamo_config, "cache_size_limit", 64)), 256)
            dynamo_config.accumulated_cache_size_limit = max(int(getattr(dynamo_config, "accumulated_cache_size_limit", 256)), 1024)
        except Exception:
            pass


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast_dtype(args: argparse.Namespace) -> torch.dtype:
    return torch.bfloat16 if args.amp == "bf16" else torch.float16


def _probe_batch_size(model: torch.nn.Module, cfg: NeedConfig, args: argparse.Namespace, device: torch.device) -> int:
    if device.type != "cuda":
        return max(1, int(args.batch_size or 1))
    max_probe = int(getattr(args, "auto_batch_max", 0) or 0)
    if max_probe <= 0:
        max_probe = 64
    start = max(1, int(getattr(args, "batch_size", 0) or 1))
    best = 0
    tried = []
    b = start
    was_training = model.training
    model.train()
    while b <= max_probe:
        tried.append(b)
        try:
            torch.cuda.empty_cache()
            x = torch.randint(low=Special.byte_start, high=min(cfg.vocab_size, Special.byte_start + 128), size=(b, cfg.block_size), device=device, dtype=torch.long)
            y = torch.roll(x, shifts=-1, dims=1).unsqueeze(-1).contiguous()
            img_mask = torch.zeros_like(x, dtype=torch.bool)
            with torch.autocast(device_type=device.type, dtype=_autocast_dtype(args), enabled=(args.amp != "off")):
                _, loss, _ = model(x, y, image_mask_positions=img_mask)
                assert loss is not None
            loss.backward()
            model.zero_grad(set_to_none=True)
            _sync_if_cuda(device)
            free, total = torch.cuda.mem_get_info(device)
            used_frac = 1.0 - (free / max(1, total))
            best = b
            if used_frac >= float(getattr(args, "target_vram_util", 0.90)):
                break
            b *= 2
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            model.zero_grad(set_to_none=True)
            break
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                model.zero_grad(set_to_none=True)
                break
            raise
    if best <= 0:
        best = 1
    # Binary refine between the best success and first failed/too-large power-of-two.
    hi = min(max_probe, max(best + 1, b - 1))
    lo = best + 1
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            torch.cuda.empty_cache()
            x = torch.randint(low=Special.byte_start, high=min(cfg.vocab_size, Special.byte_start + 128), size=(mid, cfg.block_size), device=device, dtype=torch.long)
            y = torch.roll(x, shifts=-1, dims=1).unsqueeze(-1).contiguous()
            img_mask = torch.zeros_like(x, dtype=torch.bool)
            with torch.autocast(device_type=device.type, dtype=_autocast_dtype(args), enabled=(args.amp != "off")):
                _, loss, _ = model(x, y, image_mask_positions=img_mask)
                assert loss is not None
            loss.backward(); model.zero_grad(set_to_none=True); _sync_if_cuda(device)
            best = mid; lo = mid + 1
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if isinstance(exc, RuntimeError) and "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache(); model.zero_grad(set_to_none=True); hi = mid - 1
    model.train(was_training)
    return max(1, int(best))


def auto_optimize_training(args: argparse.Namespace, cfg: NeedConfig, model: torch.nn.Module, device: torch.device) -> None:
    if not getattr(args, "auto_optimize", False):
        if int(getattr(args, "grad_accum_steps", 1)) <= 0:
            args.grad_accum_steps = 1
        if int(getattr(args, "batch_size", 1)) <= 0:
            args.batch_size = 1
        return
    if device.type == "cuda":
        args.optimizer_fused = True
        args.tf32 = True
        args.cudnn_benchmark = True
        args.prefetch_to_device = True
        if getattr(args, "compile", False):
            args.compile_cudagraphs = True
            args.compile_dynamic = False
        if str(getattr(args, "packed_data", "") or getattr(args, "pack_data_to", "") or ""):
            args.drop_last = True
    if int(getattr(args, "num_workers", 0)) < 0 or (int(getattr(args, "num_workers", 0)) == 0 and bool(getattr(args, "stream_data", False))):
        cpu = os.cpu_count() or 4
        ram = _ram_available_gb()
        cap = 8 if ram >= 32 else 4
        args.num_workers = max(1, min(cap, max(1, cpu - 2)))
    if int(getattr(args, "batch_size", 0)) <= 0 or bool(getattr(args, "auto_batch", False)):
        args.batch_size = _probe_batch_size(model, cfg, args, device)
    if int(getattr(args, "grad_accum_steps", 0)) <= 0:
        target_eff = int(getattr(args, "target_effective_batch_tokens", 0) or 1_048_576)
        args.grad_accum_steps = max(1, math.ceil(target_eff / max(1, int(args.batch_size) * int(cfg.block_size))))
    if int(getattr(args, "target_tokens", 0) or 0) > 0:
        args.max_steps = max(int(args.max_steps), math.ceil(int(args.target_tokens) / max(1, int(args.batch_size) * int(cfg.block_size))))


def make_loader(ds, args: argparse.Namespace, device: torch.device, *, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
    num_workers = max(0, int(args.num_workers))
    kwargs: Dict[str, object] = dict(
        batch_size=max(1, int(args.batch_size)),
        shuffle=(False if isinstance(ds, IterableDataset) else shuffle),
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
        drop_last=bool(drop_last),
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = max(2, int(getattr(args, "prefetch_factor", 4)))
    return DataLoader(ds, **kwargs)


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    if device.type == "cuda":
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    return {k: v.to(device) for k, v in batch.items()}


def iter_device_batches(loader: DataLoader, device: torch.device, *, prefetch: bool = False) -> Iterator[Dict[str, torch.Tensor]]:
    if device.type != "cuda" or not prefetch:
        for batch in loader:
            yield _move_batch_to_device(batch, device)
        return
    stream = torch.cuda.Stream(device=device)
    it = iter(loader)
    next_batch: Optional[Dict[str, torch.Tensor]] = None

    def preload() -> None:
        nonlocal next_batch
        try:
            host_batch = next(it)
        except StopIteration:
            next_batch = None
            return
        with torch.cuda.stream(stream):
            next_batch = _move_batch_to_device(host_batch, device)

    preload()
    while next_batch is not None:
        torch.cuda.current_stream(device).wait_stream(stream)
        batch = next_batch
        for value in batch.values():
            if torch.is_tensor(value) and value.is_cuda:
                value.record_stream(torch.cuda.current_stream(device))
        preload()
        yield batch


def mark_compiled_step(args: argparse.Namespace, device: torch.device) -> None:
    if device.type != "cuda" or not getattr(args, "compile_cudagraphs", False):
        return
    try:
        torch.compiler.cudagraph_mark_step_begin()  # type: ignore[attr-defined]
    except Exception:
        pass


def maybe_prepare_visual_tokenizer(args: argparse.Namespace, out_dir: Path) -> str:
    """Return a visual-tokenizer directory and align NEED image config with it."""
    vt_path = str(getattr(args, "visual_tokenizer", "") or "")
    if getattr(args, "train_visual_tokenizer", False):
        if not args.image_dir:
            raise ValueError("--train_visual_tokenizer requires --image_dir")
        if train_vq_tokenizer is None:
            raise RuntimeError("need_image.py is required for --train_visual_tokenizer")
        vt_path = str(out_dir / "visual_tokenizer")
        ns = argparse.Namespace(
            image_dir=args.image_dir,
            out_dir=vt_path,
            device=args.visual_tokenizer_device,
            seed=args.seed,
            codebook_size=args.image_codebook_size,
            embed_dim=args.vq_embed_dim,
            hidden_dim=args.vq_hidden_dim,
            num_res_blocks=args.vq_res_blocks,
            downsample=args.vq_downsample,
            grid=args.image_grid,
            min_grid=args.image_min_grid,
            max_grid=args.image_max_grid,
            max_image_tokens=args.image_max_tokens,
            batch_size=args.vq_batch_size,
            samples=args.vq_samples,
            steps=(args.vq_steps if args.vq_steps > 0 else max(100, args.max_steps // 10)),
            num_workers=args.num_workers,
            lr=args.vq_lr,
            weight_decay=args.vq_weight_decay,
            gan_weight=args.vq_gan_weight,
            gan_start=args.vq_gan_start,
            perceptual_weight=args.vq_perceptual_weight,
            edge_weight=args.vq_edge_weight,
            amp=args.amp,
            log_interval=max(1, args.log_interval),
            save_interval=max(1, args.vq_save_interval),
            curriculum=args.vq_curriculum,
            code_usage_weight=args.vq_code_usage_weight,
        )
        train_vq_tokenizer(ns)
    if vt_path:
        cfg_path = Path(vt_path) / "visual_tokenizer_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"visual tokenizer config not found: {cfg_path}")
        raw = load_json(cfg_path)
        vc = raw.get("config", raw)
        args.image_codebook_size = int(vc.get("codebook_size", args.image_codebook_size))
        args.image_grid = int(vc.get("default_grid", args.image_grid))
        args.image_min_grid = int(vc.get("min_grid", args.image_min_grid))
        args.image_max_grid = int(vc.get("max_grid", args.image_max_grid))
        args.image_max_tokens = int(vc.get("max_image_tokens", args.image_max_tokens))
        # Keep a copy next to the NEED model for generation/decode portability.
        dest_cfg = out_dir / "visual_tokenizer_config.json"
        dest_st = out_dir / "visual_tokenizer.safetensors"
        dest_pt = out_dir / "visual_tokenizer.pt"
        if Path(vt_path).resolve() != out_dir.resolve():
            if cfg_path.exists():
                dest_cfg.write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")
            if (Path(vt_path) / "visual_tokenizer.safetensors").exists():
                import shutil
                shutil.copy2(Path(vt_path) / "visual_tokenizer.safetensors", dest_st)
            elif (Path(vt_path) / "visual_tokenizer.pt").exists():
                import shutil
                shutil.copy2(Path(vt_path) / "visual_tokenizer.pt", dest_pt)
    return vt_path
def _append_train_log(path: str, row: Dict[str, object]) -> None:
    if not path:
        return
    try:
        pp = Path(path)
        pp.parent.mkdir(parents=True, exist_ok=True)
        with pp.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    except Exception:
        pass




def _jsonable_args(args: argparse.Namespace) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, set):
            out[k] = sorted(str(x) for x in v)
        elif isinstance(v, (list, tuple)):
            out[k] = [str(x) if isinstance(x, Path) else x for x in v]
        else:
            out[k] = v
    return out


def _load_json_maybe(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _code_fingerprint(root: Optional[Path] = None) -> Dict[str, Any]:
    root = Path(root or Path(__file__).resolve().parent)
    files = []
    h = hashlib.sha256()
    for fp in sorted(root.glob("*.py")):
        try:
            rel = fp.name
            data = fp.read_bytes()
            files.append({"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
            h.update(rel.encode("utf-8")); h.update(b"\0"); h.update(data); h.update(b"\0")
        except Exception:
            continue
    return {"format": "need_code_fingerprint_v1", "root": str(root), "sha256": h.hexdigest(), "files": files}


def verify_packed_index_integrity(index_path: Path, *, strict: bool = True) -> Dict[str, Any]:
    """Validate packed-index references before a long run.

    Checks are intentionally local: path exists, file size matches metadata when
    present, SHA-256 matches when present, weights are usable, and the total token
    count is nonzero.  This catches truncated copies, stale manifests, and silent
    dataset drift before a larger token-budget run.
    """
    report: Dict[str, Any] = {"index": str(index_path), "ok": True, "warnings": [], "sources": 0, "tokens": 0}
    raw = json.loads(index_path.read_text(encoding="utf-8"))
    sources = raw.get("sources", []) if isinstance(raw, dict) else []
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"packed index has no sources: {index_path}")
    total_weight = 0.0
    base = index_path.resolve().parent
    for src in sources:
        if not isinstance(src, dict):
            continue
        path_text = str(src.get("path", ""))
        fp = Path(path_text)
        if not fp.is_absolute():
            fp = (base / fp).resolve() if not Path(path_text).exists() else Path(path_text).resolve()
        if not fp.exists():
            msg = f"missing packed source: {path_text}"
            if strict:
                raise FileNotFoundError(msg)
            report["ok"] = False; report["warnings"].append(msg); continue
        stat = fp.stat()
        expected_size = int(src.get("file_size_bytes", 0) or 0)
        if expected_size and expected_size != stat.st_size:
            msg = f"size mismatch for {fp}: expected {expected_size}, got {stat.st_size}"
            if strict:
                raise ValueError(msg)
            report["ok"] = False; report["warnings"].append(msg)
        expected_hash = str(src.get("sha256", "") or "")
        if expected_hash:
            actual_hash = _file_sha256(fp)
            if actual_hash != expected_hash:
                msg = f"sha256 mismatch for {fp}"
                if strict:
                    raise ValueError(msg)
                report["ok"] = False; report["warnings"].append(msg)
        meta_path = Path(str(src.get("metadata", ""))) if src.get("metadata") else _packed_metadata_path(fp)
        if not meta_path.is_absolute():
            meta_path = (base / meta_path).resolve() if not meta_path.exists() else meta_path.resolve()
        if meta_path.exists():
            meta = _load_json_maybe(meta_path)
            meta_tokens = int(meta.get("tokens", 0) or 0)
            src_tokens = int(src.get("tokens", 0) or 0)
            if src_tokens and meta_tokens and abs(src_tokens - meta_tokens) > 0:
                msg = f"token metadata mismatch for {fp}: index={src_tokens}, meta={meta_tokens}"
                if strict:
                    raise ValueError(msg)
                report["warnings"].append(msg)
        total_weight += max(0.0, float(src.get("normalized_weight", src.get("weight", 0.0)) or 0.0))
        report["sources"] = int(report["sources"]) + 1
        report["tokens"] = int(report["tokens"]) + int(src.get("tokens", 0) or 0)
    if total_weight <= 0:
        raise ValueError(f"packed index has no positive sampling weight: {index_path}")
    if int(report["tokens"]) <= 0:
        report["warnings"].append("packed index does not report token counts")
    return report



def _packed_input_tokenizer_json(args: argparse.Namespace) -> Optional[Path]:
    """Return a tokenizer sidecar for pre-packed inputs, if one exists.

    Packed data is only semantically valid with the tokenizer that produced it.
    prepare_packed_dataset.py writes tokenizer.json next to packed_index.json;
    --pack_data_to writes into the current run and therefore does not need this.
    """
    candidates: List[Path] = []
    if str(getattr(args, "packed_index", "") or ""):
        index_path = Path(str(args.packed_index))
        candidates.append(index_path.parent / "tokenizer.json")
    if str(getattr(args, "packed_data", "") or ""):
        packed_path = Path(str(args.packed_data))
        candidates.append(packed_path.parent / "tokenizer.json")
    seen: set[str] = set()
    for cand in candidates:
        key = str(cand.resolve()) if cand.exists() else str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.exists():
            return cand
    return None


def _load_packed_input_tokenizer(args: argparse.Namespace):
    tok_path = _packed_input_tokenizer_json(args)
    if tok_path is None:
        return None
    return load_tokenizer(load_json(tok_path))


def _packed_input_vocab_size(args: argparse.Namespace) -> int:
    """Read declared packed-token vocabulary size from local metadata."""
    if str(getattr(args, "packed_index", "") or ""):
        try:
            raw = load_json(Path(str(args.packed_index)))
            return int(raw.get("vocab_size", 0) or 0)
        except Exception:
            return 0
    if str(getattr(args, "packed_data", "") or ""):
        meta = _packed_metadata_path(Path(str(args.packed_data)))
        if meta.exists():
            try:
                return int(load_json(meta).get("vocab_size", 0) or 0)
            except Exception:
                return 0
    return 0


def _tokenizer_signature(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    out = {"type": str(data.get("type", "")).lower(), "vocab_size": int(data.get("vocab_size", 0) or 0)}
    if out["type"] == "hf":
        out["model_name"] = str(data.get("model_name", "") or "")
        out["revision"] = str(data.get("revision", "") or "")
        out["base_vocab_size"] = int(data.get("base_vocab_size", 0) or 0)
    return out


def _validate_packed_tokenizer_compatible(args: argparse.Namespace, tokenizer: Any) -> None:
    tok_path = _packed_input_tokenizer_json(args)
    if tok_path is None or bool(getattr(args, "allow_packed_vocab_mismatch", False)):
        return
    packed_sig = _tokenizer_signature(load_json(tok_path))
    current_sig = _tokenizer_signature(tokenizer.to_dict() if hasattr(tokenizer, "to_dict") else {})
    if packed_sig and current_sig and packed_sig != current_sig:
        raise ValueError(
            f"packed data tokenizer sidecar {tok_path} does not match the active tokenizer. "
            "Use the matching tokenizer/checkpoint, repack the data, or pass --allow_packed_vocab_mismatch only if intentional."
        )


def _validate_packed_vocab_compatible(args: argparse.Namespace, cfg: NeedConfig) -> None:
    """Fail early on tokenizer/vocabulary mismatches for pre-packed data.

    A mismatch can either crash later in the embedding lookup or, worse, keep all
    token IDs in range while changing their meaning. The default is conservative;
    pass --allow_packed_vocab_mismatch only for deliberate expert workflows.
    """
    packed_vocab = _packed_input_vocab_size(args)
    if packed_vocab <= 0:
        return
    model_vocab = int(cfg.vocab_size)
    if packed_vocab != model_vocab and not bool(getattr(args, "allow_packed_vocab_mismatch", False)):
        raise ValueError(
            f"packed data declares vocab_size={packed_vocab}, but the model/tokenizer config uses "
            f"vocab_size={model_vocab}. Use the tokenizer.json that was written with the packed data, "
            "repack with the current tokenizer, or pass --allow_packed_vocab_mismatch only if this is intentional."
        )
    if packed_vocab > model_vocab:
        raise ValueError(
            f"packed data can contain token IDs for vocab_size={packed_vocab}, but model vocab_size={model_vocab}; "
            "repack or use the matching tokenizer/checkpoint."
        )


def _jsonable_rng_state() -> Dict[str, Any]:
    """Serialize RNG state without pickle-only numpy objects.

    Keeping the checkpoint payload to tensors, primitive containers, and numbers
    lets PyTorch's weights_only loader read training_state.pt without executing
    arbitrary pickle code.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    state: Dict[str, Any] = {
        "python": [py_state[0], list(py_state[1]), py_state[2]],
        "numpy": {
            "bit_generator": str(np_state[0]),
            "state": np_state[1].astype(np.uint32).tolist(),
            "pos": int(np_state[2]),
            "has_gauss": int(np_state[3]),
            "cached_gaussian": float(np_state[4]),
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return state


def _rng_state() -> Dict[str, Any]:
    return _jsonable_rng_state()


def _restore_rng_state(state: Dict[str, Any]) -> None:
    try:
        if "python" in state:
            py = state["python"]
            if isinstance(py, (list, tuple)) and len(py) == 3:
                random.setstate((int(py[0]), tuple(int(x) for x in py[1]), py[2]))
            else:
                random.setstate(py)
        if "numpy" in state:
            ns = state["numpy"]
            if isinstance(ns, dict):
                np.random.set_state((
                    str(ns.get("bit_generator", "MT19937")),
                    np.asarray(ns.get("state", []), dtype=np.uint32),
                    int(ns.get("pos", 0)),
                    int(ns.get("has_gauss", 0)),
                    float(ns.get("cached_gaussian", 0.0)),
                ))
            else:
                np.random.set_state(ns)
        if "torch" in state:
            torch.set_rng_state(state["torch"].cpu() if torch.is_tensor(state["torch"]) else state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception:
        pass


def _atomic_torch_save(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(obj, tmp)
        with tmp.open("rb") as f:
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def save_training_state(path: Path, model: torch.nn.Module, opt: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler, cfg: NeedConfig, args: argparse.Namespace, *, step: int, optimizer_step: int, tokens_seen: int, best: float, ema_state: Optional[Dict[str, torch.Tensor]] = None) -> None:
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    state = {
        "format": "need_training_state_v2",
        "model": {k: v.detach().cpu() for k, v in raw_model.state_dict().items()},
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else {},
        "cfg": asdict(cfg),
        "args": _jsonable_args(args),
        "step": int(step),
        "optimizer_step": int(optimizer_step),
        "tokens_seen": int(tokens_seen),
        "best": float(best),
        "rng": _rng_state(),
        "ema_state": {k: v.detach().cpu() for k, v in (ema_state or {}).items()} if ema_state else None,
        "code_fingerprint": _code_fingerprint(),
        "time": time.time(),
    }
    _atomic_torch_save(state, path)


def load_training_state(path: Path, device: torch.device, *, allow_unsafe: bool = False) -> Dict[str, Any]:
    """Load a full training checkpoint.

    Defaults to PyTorch's restricted weights_only path.  Passing
    --allow_unsafe_checkpoint_load re-enables legacy pickle loading for old
    checkpoints, but should only be used for checkpoints you created locally.
    """
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # Older PyTorch has no weights_only flag; this is inherently pickle based.
        if not allow_unsafe:
            raise RuntimeError(
                "This PyTorch version cannot safely load training_state.pt. "
                "Upgrade PyTorch or pass --allow_unsafe_checkpoint_load for a trusted local checkpoint."
            )
        return torch.load(path, map_location=device)
    except Exception as exc:
        if allow_unsafe:
            return torch.load(path, map_location=device)
        raise RuntimeError(
            f"Safe checkpoint load failed for {path}: {exc}. "
            "Use --allow_unsafe_checkpoint_load only for trusted local checkpoints."
        )


def _compare_cfg(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    diffs = []
    keys = sorted(set(a) | set(b))
    for k in keys:
        if a.get(k) != b.get(k):
            diffs.append(k)
    return diffs


def _group_prefix(name: str) -> str:
    parts = name.split(".")
    return parts[0] if parts else name


def module_diagnostics(model: torch.nn.Module, top_k: int = 24) -> Dict[str, float]:
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    p_l2: Dict[str, float] = {}
    g_l2: Dict[str, float] = {}
    for name, p in raw.named_parameters():
        group = _group_prefix(name)
        try:
            p_l2[group] = p_l2.get(group, 0.0) + float(p.detach().float().pow(2).sum().cpu())
            if p.grad is not None:
                g_l2[group] = g_l2.get(group, 0.0) + float(p.grad.detach().float().pow(2).sum().cpu())
        except Exception:
            continue
    out: Dict[str, float] = {}
    for group, val in sorted(p_l2.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
        out[f"param_l2/{group}"] = math.sqrt(max(0.0, val))
    for group, val in sorted(g_l2.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
        out[f"grad_l2/{group}"] = math.sqrt(max(0.0, val))
    return out


def _update_ema(ema: Optional[Dict[str, torch.Tensor]], model: torch.nn.Module, decay: float) -> Optional[Dict[str, torch.Tensor]]:
    if decay <= 0:
        return ema
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    with torch.no_grad():
        if ema is None:
            return {k: v.detach().float().cpu().clone() for k, v in raw.state_dict().items() if torch.is_floating_point(v)}
        for k, v in raw.state_dict().items():
            if k in ema and torch.is_floating_point(v):
                ema[k].mul_(decay).add_(v.detach().float().cpu(), alpha=1.0 - decay)
    return ema


def save_ema_model(ema: Optional[Dict[str, torch.Tensor]], model: torch.nn.Module, out: Path, metrics: Dict[str, float]) -> None:
    if not ema:
        return
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    current = {k: v.detach().cpu().clone() for k, v in raw.state_dict().items()}
    try:
        merged = dict(current)
        for k, v in ema.items():
            if k in merged and torch.is_floating_point(merged[k]):
                merged[k] = v.to(dtype=merged[k].dtype)
        raw.load_state_dict(merged, strict=False)
        save_model(raw, out, metrics, name="ema")
    finally:
        raw.load_state_dict(current, strict=False)


def _stable_seed_offset(name: str, modulus: int = 100000) -> int:
    digest = hashlib.sha256(str(name).encode("utf-8", errors="replace")).digest()
    return int.from_bytes(digest[:8], "little") % max(1, int(modulus))


def _parse_named_paths(items: Sequence[str]) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for item in items or []:
        if not item:
            continue
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            path = item
            name = Path(path).stem or "eval"
        out.append((str(name).strip() or "eval", Path(path)))
    return out


def _build_text_eval_loader(name: str, path: Path, cfg: NeedConfig, tokenizer: ByteTokenizer, args: argparse.Namespace, device: torch.device) -> DataLoader:
    samples = max(64, int(args.batch_size) * 8)
    if path.suffix == ".json" and _load_json_maybe(path).get("format", "").startswith("need_source_balanced"):
        ds = SourceBalancedPackedMapDataset(path, cfg, cfg.n_predict_heads, samples=samples, seed=args.seed + _stable_seed_offset(name), dtype_name=args.packed_dtype)
    elif path.suffix in (".bin", ".tokens", ".tokbin"):
        ds = PackedTokenMapDataset(path, cfg, cfg.n_predict_heads, samples=samples, seed=args.seed + _stable_seed_offset(name), dtype_name=args.packed_dtype)
    else:
        ids = read_text_tokens(path, tokenizer, max_bytes=int(getattr(args, "eval_data_bytes", 0) or 64 * 1024 * 1024))
        ds = TextChunkDataset(ids, cfg.block_size, cfg.n_predict_heads, samples=samples, seed=args.seed + _stable_seed_offset(name))
    return make_loader(MixedDataset(ds, None, samples=samples, image_ratio=0.0, seed=args.seed), args, device, shuffle=False, drop_last=False)


def validate_domains(model: NeedModel, loaders: Dict[str, DataLoader], device: torch.device, args: argparse.Namespace) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, loader in loaders.items():
        metrics = validate(model, loader, device, max_batches=args.eval_batches, prefetch_to_device=bool(getattr(args, "prefetch_to_device", False)))
        for k, v in metrics.items():
            out[f"{name}_{k}"] = v
    return out


def generate_training_samples(model: torch.nn.Module, tokenizer: ByteTokenizer, prompts_path: str, out_path: Path, device: torch.device, *, step: int, max_new_tokens: int = 96) -> None:
    if not prompts_path:
        return
    p = Path(prompts_path)
    if not p.exists():
        return
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    was_training = raw.training
    raw.eval()
    prompts = [line.rstrip("\n") for line in p.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    rows = []
    with torch.no_grad():
        for prompt in prompts[:16]:
            ids = tokenizer.encode(prompt, add_bos=True)[-raw.cfg.block_size:]
            inp = torch.tensor([ids], dtype=torch.long, device=device)
            gen = raw.generate_text(inp, max_new_tokens=max_new_tokens, temperature=0.8, top_k=50, top_p=0.95)
            text = tokenizer.decode(gen[0].detach().cpu().tolist())
            rows.append({"step": int(step), "prompt": prompt, "sample": text})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    raw.train(was_training)

def validate(model: NeedModel, loader: DataLoader, device: torch.device, max_batches: int = 20, *, prefetch_to_device: bool = False) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    losses = []
    ces = []
    health: Dict[str, List[float]] = {}
    health_keys = (
        "objective_aux_abs_ratio", "objective_clipped_terms", "objective_nonfinite_terms",
        "objective_quarantined_terms", "objective_min_quarantine_scale",
        "latent_slot_attention_entropy", "latent_slot_coverage", "latent_slot_diversity",
        "object_slot_entropy", "object_coverage", "energy_route_entropy", "energy_route_balance",
        "output_mode_entropy", "latent_divergence", "risk_signal_mean", "aux_score_risk_mean",
        "memory_entropy", "memory_diversity", "state_chunk_drift", "state_norm_error", "state_anchor_error",
        "coop_step_gate_mean", "coop_step_contribution", "coop_step_redundancy", "coop_gate_budget",
    )
    t0 = time.time(); toks = 0
    with torch.no_grad():
        for i, batch in enumerate(iter_device_batches(loader, device, prefetch=prefetch_to_device)):
            if i >= max_batches:
                break
            x = batch["input_ids"]
            y = batch["targets"]
            img_mask = batch.get("image_mask_positions")
            img_targets = batch.get("image_targets")
            _, loss, aux = model(x, y, image_mask_positions=img_mask, image_targets=img_targets)
            if loss is not None:
                losses.append(float(loss.detach().cpu()))
            if "ce" in aux:
                ces.append(float(aux["ce"].detach().cpu()))
            for key in health_keys:
                v = aux.get(key)
                if torch.is_tensor(v):
                    try:
                        health.setdefault(key, []).append(float(v.detach().float().mean().cpu()))
                    except Exception:
                        pass
            toks += int(x.numel())
    model.train(was_training)
    out = {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "val_ce": float(np.mean(ces)) if ces else float("nan"),
        "val_tokens_per_sec": estimate_mfu_tokens_per_sec(toks, time.time()-t0),
    }
    for key, vals in health.items():
        if vals:
            out[f"val_{key}"] = float(np.mean(vals))
    return out


OBJECTIVE_FAMILY_LAMBDAS: Dict[str, Tuple[str, ...]] = {
    "prediction": ("lambda_prediction_aux",),
    "latent": ("lambda_latent_aux",),
    "risk": ("lambda_risk_aux",),
    "vision": ("lambda_vision_aux",),
    "regularizer": ("lambda_regularizer_aux",),
}
FUSED_LAMBDA_FIELDS: Tuple[str, ...] = tuple(name for names in OBJECTIVE_FAMILY_LAMBDAS.values() for name in names)



def quarantine_objective_family(model: torch.nn.Module, family: str, decay: float, min_scale: float) -> int:
    raw = _unwrap_model(model)
    balancer = getattr(raw, "objective_balancer", None)
    if balancer is None or not hasattr(balancer, "quarantine_scale"):
        return 0
    count = 0
    decay = float(min(max(decay, 0.0), 1.0))
    min_scale = float(min(max(min_scale, 0.0), 1.0))
    with torch.no_grad():
        for name, idx in getattr(balancer, "loss_index", {}).items():
            if OBJECTIVE_LOSS_GROUPS.get(str(name), "other") != family:
                continue
            balancer.quarantine_scale[idx].copy_((balancer.quarantine_scale[idx] * decay).clamp_min(min_scale))
            if hasattr(balancer, "quarantine_timer"):
                balancer.quarantine_timer[idx].zero_()
            count += 1
    return count


def run_objective_family_ablations(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    baseline: Dict[str, float],
) -> Dict[str, float]:
    """Evaluate validation loss with each auxiliary family temporarily disabled."""
    raw = _unwrap_model(model)
    cfg = getattr(raw, "cfg", None)
    if cfg is None:
        return {}
    names = [x.strip() for x in str(getattr(args, "objective_ablation_families", "") or "").split(",") if x.strip()]
    if not names:
        names = list(OBJECTIVE_FAMILY_LAMBDAS.keys())
    max_batches = max(1, int(getattr(args, "objective_ablation_eval_batches", 3) or 3))
    out: Dict[str, float] = {}
    for family in names:
        lambdas = OBJECTIVE_FAMILY_LAMBDAS.get(family)
        if not lambdas:
            continue
        saved = {name: getattr(cfg, name) for name in lambdas if hasattr(cfg, name)}
        saved_entropy = getattr(cfg, "objective_entropy_band_weight", None)
        try:
            for name in saved:
                setattr(cfg, name, 0.0)
            if family in {"latent", "risk", "vision"} and saved_entropy is not None:
                cfg.objective_entropy_band_weight = 0.0
            metrics = validate(model, loader, device, max_batches=max_batches, prefetch_to_device=bool(getattr(args, "prefetch_to_device", False)))
            loss = float(metrics.get("val_loss", float("nan")))
            ce = float(metrics.get("val_ce", float("nan")))
            out[f"ablation_{family}_val_loss"] = loss
            out[f"ablation_{family}_val_ce"] = ce
            ce_delta = float("nan")
            loss_delta = float("nan")
            if math.isfinite(loss) and math.isfinite(float(baseline.get("val_loss", float("nan")))):
                loss_delta = loss - float(baseline["val_loss"])
                out[f"ablation_{family}_val_loss_delta"] = loss_delta
            if math.isfinite(ce) and math.isfinite(float(baseline.get("val_ce", float("nan")))):
                ce_delta = ce - float(baseline["val_ce"])
                out[f"ablation_{family}_val_ce_delta"] = ce_delta
            if bool(getattr(args, "objective_ablation_auto_quarantine", False)):
                threshold = float(getattr(args, "objective_ablation_improve_threshold", 0.0) or 0.0)
                # Negative delta means disabling this family improved validation.
                if (math.isfinite(ce_delta) and ce_delta < -threshold) or (math.isfinite(loss_delta) and loss_delta < -threshold):
                    qn = quarantine_objective_family(
                        model, family,
                        float(getattr(args, "objective_ablation_quarantine_decay", 0.70) or 0.70),
                        float(getattr(cfg, "objective_quarantine_min_scale", 0.05)),
                    )
                    out[f"ablation_{family}_auto_quarantined_terms"] = float(qn)
        finally:
            for name, value in saved.items():
                setattr(cfg, name, value)
            if saved_entropy is not None:
                cfg.objective_entropy_band_weight = saved_entropy
    return out



def train(args: argparse.Namespace) -> None:
    # Apply formulaic sizing before runtime setup.
    apply_profile(args)
    if getattr(args, "recipe", "") and apply_training_recipe is not None:
        apply_training_recipe(args, args.recipe)
    maybe_apply_target_model_size(args)

    device = resolve_device(args.device)
    configure_runtime(args, device)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    if not getattr(args, "metrics_jsonl", ""):
        args.metrics_jsonl = str(out / "train_log.jsonl")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    visual_tokenizer_path = maybe_prepare_visual_tokenizer(args, out)
    init_model = None
    resume_state: Optional[Dict[str, Any]] = None
    tokenizer_kind = getattr(args, "tokenizer", "hf")
    tokenizer_model = getattr(args, "tokenizer_model", "") or None
    if getattr(args, "resume_from", ""):
        resume_path = Path(args.resume_from)
        resume_dir = resume_path if resume_path.is_dir() else resume_path.parent
        if resume_path.is_dir():
            resume_path = resume_path / "training_state.pt"
        resume_state = load_training_state(resume_path, device, allow_unsafe=bool(getattr(args, "allow_unsafe_checkpoint_load", False)))
        raw_cfg = resume_state.get("cfg", {})
        cfg = NeedConfig.from_dict(raw_cfg)
        if getattr(args, "kernel_backend", ""):
            cfg.kernel_backend = args.kernel_backend
        cfg.validate()
        # Resuming must keep using whatever tokenizer the checkpoint was trained
        # with, regardless of the current default, or token IDs won't line up.
        # Checkpoints saved before tokenizer.json existed were always byte-level.
        tokenizer = load_tokenizer_for_dir(resume_dir)
        current_cfg = asdict(build_config(args, text_vocab_size=tokenizer.vocab_size))
        diffs = _compare_cfg(current_cfg, asdict(cfg))
        if getattr(args, "resume_strict", False) and diffs:
            raise ValueError(f"--resume_strict refused config mismatch: {diffs[:25]}")
    elif getattr(args, "init_from", ""):
        init_model = load_model(args.init_from, device=device, prefer_best=bool(args.init_prefer_best), kernel_backend=args.kernel_backend)
        cfg = init_model.cfg
        tokenizer = load_tokenizer_for_dir(Path(args.init_from))
    else:
        packed_tokenizer = _load_packed_input_tokenizer(args)
        explicit = set(getattr(args, "_explicit_args", set()) or set())
        explicit_tokenizer = bool({"tokenizer", "tokenizer_model"} & explicit)
        if packed_tokenizer is not None and not explicit_tokenizer:
            tokenizer = packed_tokenizer
            print(json.dumps({"packed_tokenizer": "loaded", "source": str(_packed_input_tokenizer_json(args)), "vocab_size": int(tokenizer.vocab_size)}), flush=True)
        else:
            tokenizer = build_default_tokenizer(tokenizer_model, kind=tokenizer_kind)
            if packed_tokenizer is not None and int(getattr(packed_tokenizer, "vocab_size", 0)) != int(getattr(tokenizer, "vocab_size", 0)) and not bool(getattr(args, "allow_packed_vocab_mismatch", False)):
                raise ValueError(
                    "Explicit tokenizer settings do not match the tokenizer.json beside the packed data. "
                    "Use the matching tokenizer, repack the data, or pass --allow_packed_vocab_mismatch only if intentional."
                )
        cfg = build_config(args, text_vocab_size=tokenizer.vocab_size)

    cfg.validate()
    _validate_packed_tokenizer_compatible(args, tokenizer)
    _validate_packed_vocab_compatible(args, cfg)

    save_json(tokenizer.to_dict(), out / "tokenizer.json")
    if visual_tokenizer_path:
        save_json({"image_tokenizer": "learned_vq", "path": "visual_tokenizer_config.json", "codebook_size": cfg.image_codebook_size}, out / "image_tokenizer.json")
    else:
        save_json({"image_tokenizer": make_image_tokenizer(cfg).to_dict(), "type": "dynamic_fallback"}, out / "image_tokenizer.json")

    if getattr(args, "pack_data_to", ""):
        if not args.data:
            raise ValueError("--pack_data_to requires --data")
        meta = pack_text_tokens_to_bin(
            Path(args.data), Path(args.pack_data_to), tokenizer,
            vocab_size=cfg.vocab_size, dtype_name=args.packed_dtype, max_bytes=int(args.max_data_bytes or 0),
        )
        print(json.dumps({"packed_data": str(args.pack_data_to), **meta}), flush=True)
        args.packed_data = str(args.pack_data_to)
        if getattr(args, "pack_only", False):
            return

    model = init_model if init_model is not None else NeedModel(cfg).to(device)
    if resume_state is not None:
        strict_load = bool(getattr(args, "resume_strict", False))
        missing, unexpected = model.load_state_dict(resume_state.get("model", {}), strict=strict_load)
        if (missing or unexpected) and not strict_load:
            print(json.dumps({"resume_warning": "model_state_partial_load", "missing": len(missing), "unexpected": len(unexpected)}), flush=True)
        if strict_load:
            old_fp = (resume_state.get("code_fingerprint") or {}).get("sha256") if isinstance(resume_state.get("code_fingerprint"), dict) else None
            new_fp = _code_fingerprint().get("sha256")
            if old_fp and new_fp and old_fp != new_fp:
                raise ValueError("--resume_strict refused code fingerprint mismatch")
    auto_optimize_training(args, cfg, model, device)
    configure_runtime(args, device)

    text_ds = None
    val_text_ds = None
    if getattr(args, "packed_index", ""):
        index_path = Path(args.packed_index)
        integrity = verify_packed_index_integrity(index_path, strict=not bool(getattr(args, "skip_packed_integrity_check", False)))
        if integrity.get("warnings"):
            print(json.dumps({"packed_index_integrity": integrity}), flush=True)
        start_sample = 0
        if resume_state is not None:
            start_sample = int(resume_state.get("tokens_seen", 0) or 0) // max(1, int(cfg.block_size))
        text_ds = SourceBalancedPackedDataset(
            index_path, cfg, cfg.n_predict_heads, seed=args.seed, dtype_name=args.packed_dtype, start_sample=start_sample,
        )
        val_text_ds = SourceBalancedPackedMapDataset(
            index_path, cfg, cfg.n_predict_heads, samples=max(64, int(args.batch_size) * 8),
            seed=args.seed + 10000, dtype_name=args.packed_dtype,
        )
    elif getattr(args, "packed_data", ""):
        packed_path = Path(args.packed_data)
        text_ds = PackedTokenIterableDataset(
            packed_path, cfg, cfg.n_predict_heads, seed=args.seed,
            dtype_name=args.packed_dtype, cycle=True, shuffle_files=True,
        )
        val_text_ds = PackedTokenMapDataset(
            packed_path, cfg, cfg.n_predict_heads, samples=max(64, int(args.batch_size) * 8),
            seed=args.seed + 10000, dtype_name=args.packed_dtype,
        )
    elif args.data:
        data_path = Path(args.data)
        if args.stream_data:
            text_ds = StreamingTextChunkDataset(
                data_path, tokenizer, cfg.block_size, cfg.n_predict_heads,
                seed=args.seed, max_bytes=args.max_data_bytes, shuffle_files=True, cycle=True,
            )
            val_bytes = int(args.val_data_bytes or min(128 * 1024 * 1024, max(8 * 1024 * 1024, args.max_data_bytes or 0)))
            if val_bytes <= 0:
                val_bytes = 32 * 1024 * 1024
            ids_val = read_text_tokens(data_path, tokenizer, max_bytes=val_bytes)
            val_text_ds = TextChunkDataset(ids_val, cfg.block_size, cfg.n_predict_heads, samples=max(64, int(args.batch_size) * 8), seed=args.seed + 10000)
        else:
            ids = read_text_tokens(data_path, tokenizer, max_bytes=args.max_data_bytes)
            text_ds = TextChunkDataset(ids, cfg.block_size, cfg.n_predict_heads, samples=args.train_samples, seed=args.seed)
            val_text_ds = text_ds

    image_ds = None
    if args.image_tokens and not args.disable_image:
        image_ds = PreTokenizedImageDataset(
            Path(args.image_tokens), cfg, samples=args.train_samples, mask_prob=args.image_mask_prob, seed=args.seed + 17,
        )
    elif args.image_dir and not args.disable_image:
        image_ds = ImageTokenDataset(
            Path(args.image_dir), cfg, samples=args.train_samples, mask_prob=args.image_mask_prob, seed=args.seed + 17,
            visual_tokenizer_path=visual_tokenizer_path, visual_tokenizer_device=args.visual_tokenizer_device, force_grid=args.force_image_grid,
        )
    if text_ds is None and image_ds is None:
        raise ValueError("Provide --data, --image_dir, --image_tokens, or use --self_test")
    if isinstance(text_ds, IterableDataset) and image_ds is not None:
        raise ValueError("--stream_data currently supports text-only training; pre-tokenize or disable image mixing for streaming runs")

    if isinstance(text_ds, IterableDataset):
        train_ds = text_ds
        val_ds = MixedDataset(val_text_ds, None, samples=max(64, int(args.batch_size) * 8), image_ratio=0.0, seed=args.seed + 10000)  # type: ignore[arg-type]
    else:
        train_ds = MixedDataset(text_ds, image_ds, samples=args.train_samples, image_ratio=args.image_ratio, seed=args.seed)
        val_ds = MixedDataset(val_text_ds if val_text_ds is not None else text_ds, image_ds, samples=max(64, int(args.batch_size) * 8), image_ratio=args.image_ratio, seed=args.seed + 10000)

    train_loader = make_loader(train_ds, args, device, shuffle=False, drop_last=bool(getattr(args, "drop_last", False)))
    val_loader = make_loader(val_ds, args, device, shuffle=False, drop_last=False)
    domain_eval_loaders: Dict[str, DataLoader] = {}
    for eval_name, eval_path in _parse_named_paths(getattr(args, "eval_data", []) or []):
        domain_eval_loaders[eval_name] = _build_text_eval_loader(eval_name, eval_path, cfg, tokenizer, args, device)

    raw_for_count = model._orig_mod if hasattr(model, "_orig_mod") else model
    param_count = int(sum(p.numel() for p in raw_for_count.parameters()))
    flops_per_token = estimate_flops_per_token(param_count)

    if args.compile and hasattr(torch, "compile"):
        compile_kwargs: Dict[str, object] = {}
        if args.compile_mode:
            compile_kwargs["mode"] = args.compile_mode
        if args.compile_fullgraph:
            compile_kwargs["fullgraph"] = True
        if getattr(args, "compile_backend", ""):
            compile_kwargs["backend"] = str(args.compile_backend)
        compile_kwargs["dynamic"] = bool(getattr(args, "compile_dynamic", True))
        model = torch.compile(model, **compile_kwargs)  # type: ignore[assignment]

    opt = configure_optimizer(model, args)
    opt.zero_grad(set_to_none=True)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp == "fp16"))
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp == "fp16"))
    resume_step = 0
    resume_optimizer_step = 0
    resume_tokens_seen = 0
    resume_best = float("inf")
    ema_state: Optional[Dict[str, torch.Tensor]] = None
    if resume_state is not None:
        try:
            opt.load_state_dict(resume_state.get("optimizer", {}))
        except Exception as exc:
            if getattr(args, "resume_strict", False):
                raise
            print(json.dumps({"resume_warning": "optimizer_state_not_loaded", "error": str(exc)}), flush=True)
        try:
            if resume_state.get("scaler"):
                scaler.load_state_dict(resume_state.get("scaler", {}))
        except Exception:
            pass
        _restore_rng_state(resume_state.get("rng", {}) or {})
        resume_step = int(resume_state.get("step", 0) or 0)
        resume_optimizer_step = int(resume_state.get("optimizer_step", 0) or 0)
        resume_tokens_seen = int(resume_state.get("tokens_seen", 0) or 0)
        resume_best = float(resume_state.get("best", float("inf")))
        if resume_state.get("ema_state"):
            ema_state = {k: v.cpu() for k, v in resume_state.get("ema_state", {}).items()}
    save_json({
        "config": asdict(cfg),
        "train_args": _jsonable_args(args),
        "parameter_count": param_count,
        "code_fingerprint": _code_fingerprint(),
    }, out / "run_config.json")

    best = resume_best
    step = resume_step
    optimizer_step = resume_optimizer_step
    accum_loss = 0.0
    log_batches = 0
    token_counter = 0
    tokens_seen = resume_tokens_seen
    target_tokens = int(getattr(args, "target_tokens", 0) or 0)
    max_steps = int(getattr(args, "max_steps", 0) or 0)
    total_optimizer_steps = max(1, int(math.ceil(max(target_tokens, max_steps * max(1, int(args.batch_size)) * int(cfg.block_size)) / max(1, int(args.target_effective_batch_tokens or (args.batch_size * cfg.block_size * max(1, args.grad_accum_steps)))))))
    t0 = time.time(); _sync_if_cuda(device)
    model.train()
    loss_ema: Optional[float] = None
    nonfinite_events = 0
    skipped_optimizer_steps = 0
    last_grad_norm = float("nan")
    last_objective_grad_probe: Dict[str, float] = {}
    state_ckpt_path = out / "training_state.pt"
    stop_signal: Dict[str, int] = {"signum": 0}
    if not bool(getattr(args, "disable_signal_checkpoint", False)):
        def _handle_stop_signal(signum, frame):  # type: ignore[no-untyped-def]
            stop_signal["signum"] = int(signum)
            print(json.dumps({"signal_checkpoint_requested": int(signum), "step": int(step), "tokens_seen": int(tokens_seen)}), flush=True)
        for _sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
            if _sig is not None:
                try:
                    signal.signal(_sig, _handle_stop_signal)
                except Exception:
                    pass

    def should_stop() -> bool:
        if stop_signal.get("signum", 0):
            return True
        if target_tokens > 0 and tokens_seen >= target_tokens:
            return True
        if max_steps > 0 and step >= max_steps:
            return True
        return False

    def nonlocal_ema_update() -> None:
        nonlocal ema_state
        ema_state = _update_ema(ema_state, model, float(getattr(args, "ema_decay", 0.0) or 0.0))

    def run_optimizer_step() -> None:
        nonlocal optimizer_step, skipped_optimizer_steps, last_grad_norm
        grad_norm = torch.tensor(0.0, device=device)
        if args.grad_clip > 0:
            if scaler.is_enabled():
                scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=False)
            try:
                last_grad_norm = float(grad_norm.detach().float().cpu())
            except Exception:
                last_grad_norm = float("nan")
        else:
            total = 0.0
            for p_ in model.parameters():
                if p_.grad is not None:
                    g = p_.grad.detach().float()
                    total += float(g.pow(2).sum().cpu())
            last_grad_norm = math.sqrt(max(0.0, total))
        max_bad = int(getattr(args, "max_nonfinite_events", 0) or 0)
        if not math.isfinite(last_grad_norm):
            skipped_optimizer_steps += 1
            opt.zero_grad(set_to_none=True)
            if getattr(args, "nan_recovery", False):
                for group in opt.param_groups:
                    group["lr"] = float(group.get("lr", args.lr)) * float(getattr(args, "nan_recovery_lr_decay", 0.5))
                row = {"recovery": "skipped_bad_grad", "step": step, "grad_norm": last_grad_norm, "skipped_optimizer_steps": skipped_optimizer_steps, "lr": opt.param_groups[0].get("lr", args.lr)}
                print(json.dumps(row), flush=True); _append_train_log(args.metrics_jsonl, row)
                if max_bad and skipped_optimizer_steps > max_bad:
                    raise FloatingPointError(f"too many skipped optimizer steps: {skipped_optimizer_steps}")
                return
            raise FloatingPointError(f"non-finite gradient norm at step {step}: {last_grad_norm}")
        scheduled_lr = _lr_for_optimizer_step(args, optimizer_step, total_optimizer_steps)
        _set_optimizer_lr(opt, scheduled_lr)
        if scaler.is_enabled():
            scaler.step(opt); scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        optimizer_step += 1
        nonlocal_ema_update()

    while not should_stop():
        for batch in iter_device_batches(train_loader, device, prefetch=bool(getattr(args, "prefetch_to_device", False))):
            if should_stop():
                break
            mark_compiled_step(args, device)
            x = batch["input_ids"]
            y = batch["targets"]
            img_mask = batch.get("image_mask_positions")
            with torch.autocast(device_type=device.type, dtype=_autocast_dtype(args), enabled=(device.type == "cuda" and args.amp != "off")):
                img_targets = batch.get("image_targets")
                _, loss, aux = model(x, y, image_mask_positions=img_mask, image_targets=img_targets)
                assert loss is not None
                raw_loss = loss
                raw_loss_value = float(raw_loss.detach().float().cpu()) if loss is not None else float("nan")
                loss = loss / max(1, int(args.grad_accum_steps))
            bad_loss = (not math.isfinite(raw_loss_value))
            if (not bad_loss) and getattr(args, "loss_spike_threshold", 0.0) and loss_ema is not None:
                bad_loss = raw_loss_value > float(args.loss_spike_threshold) * max(1e-8, loss_ema)
            if bad_loss:
                nonfinite_events += 1
                opt.zero_grad(set_to_none=True)
                max_bad_events = int(getattr(args, "max_nonfinite_events", 0) or 0)
                if max_bad_events and nonfinite_events > max_bad_events:
                    save_training_state(state_ckpt_path, model, opt, scaler, cfg, args, step=step, optimizer_step=optimizer_step, tokens_seen=tokens_seen, best=best, ema_state=ema_state)
                    raise FloatingPointError(f"too many recovered bad-loss events: {nonfinite_events}")
                if getattr(args, "nan_recovery", False):
                    for group in opt.param_groups:
                        group["lr"] = float(group.get("lr", args.lr)) * float(getattr(args, "nan_recovery_lr_decay", 0.5))
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    row = {"recovery": "skipped_bad_loss", "step": step, "loss": raw_loss_value, "events": nonfinite_events, "lr": opt.param_groups[0].get("lr", args.lr)}
                    print(json.dumps(row), flush=True); _append_train_log(args.metrics_jsonl, row)
                    step += 1
                    continue
                raise FloatingPointError(f"non-finite or spiking loss at step {step}: {raw_loss_value}")
            probe_metrics = maybe_probe_objective_gradients(model, aux, step, args)
            if probe_metrics:
                last_objective_grad_probe = probe_metrics
            loss_ema = raw_loss_value if loss_ema is None else (0.98 * loss_ema + 0.02 * raw_loss_value)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            batch_tokens = int(x.numel())
            tokens_seen += batch_tokens
            token_counter += batch_tokens
            accum_loss += float(raw_loss.detach().cpu())
            log_batches += 1
            if (step + 1) % max(1, int(args.grad_accum_steps)) == 0:
                run_optimizer_step()
            if step % max(1, int(args.log_interval)) == 0:
                _sync_if_cuda(device)
                elapsed = time.time() - t0
                tok_s = estimate_mfu_tokens_per_sec(token_counter, elapsed)
                row = {
                    "step": step,
                    "optimizer_step": optimizer_step,
                    "tokens_seen": tokens_seen,
                    "target_tokens": target_tokens,
                    "loss": accum_loss / max(1, log_batches),
                    "ce": float(aux.get("ce", torch.tensor(float("nan"))).detach().cpu()),
                    "tok_s": tok_s,
                    "batch_size": int(args.batch_size),
                    "grad_accum_steps": int(args.grad_accum_steps),
                    "params": param_count,
                    "lr": float(opt.param_groups[0].get("lr", args.lr)),
                    "grad_norm": float(last_grad_norm),
                    "nonfinite_events": int(nonfinite_events),
                    "skipped_optimizer_steps": int(skipped_optimizer_steps),
                }
                for _k in (
                    "objective_aux_abs_ratio", "objective_aux_signed_ratio",
                    "objective_clipped_terms", "objective_nonfinite_terms",
                    "objective_group_prediction_ratio", "objective_group_latent_ratio",
                    "objective_group_risk_ratio", "objective_group_vision_ratio",
                    "objective_group_regularizer_ratio", "objective_group_other_ratio",
                    "objective_quarantined_terms", "objective_min_quarantine_scale",
                    "objective_mean_quarantine_scale", "objective_step",
                ):
                    _v = aux.get(_k)
                    if torch.is_tensor(_v):
                        row[_k] = float(_v.detach().float().cpu())
                if last_objective_grad_probe:
                    row.update(last_objective_grad_probe)
                if int(getattr(args, "module_diagnostics_interval", 0) or 0) > 0 and step % int(args.module_diagnostics_interval) == 0:
                    row.update(module_diagnostics(model))
                if float(getattr(args, "peak_tflops", 0.0) or 0.0) > 0:
                    row["mfu"] = float(tok_s * flops_per_token / (float(args.peak_tflops) * 1e12))
                print(json.dumps(row), flush=True)
                _append_train_log(args.metrics_jsonl, row)
                accum_loss = 0.0; log_batches = 0; token_counter = 0; t0 = time.time(); _sync_if_cuda(device)
            if step % max(1, int(args.eval_interval)) == 0 and step > 0:
                metrics = validate(model, val_loader, device, max_batches=args.eval_batches, prefetch_to_device=bool(getattr(args, "prefetch_to_device", False)))
                if domain_eval_loaders:
                    metrics.update(validate_domains(model, domain_eval_loaders, device, args))
                if int(getattr(args, "objective_ablation_interval", 0) or 0) > 0 and step % int(args.objective_ablation_interval) == 0:
                    metrics.update(run_objective_family_ablations(model, val_loader, device, args, metrics))
                eval_row = {"eval_step": step, "optimizer_step": optimizer_step, "tokens_seen": tokens_seen, **metrics}
                print(json.dumps(eval_row), flush=True)
                _append_train_log(args.metrics_jsonl, eval_row)
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                save_model(raw_model, out, metrics, name="model")
                if metrics["val_loss"] < best:
                    best = metrics["val_loss"]
                    save_model(raw_model, out, metrics, name="best")
            if int(getattr(args, "save_interval", 0) or 0) > 0 and step > 0 and step % int(args.save_interval) == 0:
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                save_model(raw_model, out, {"step": step, "tokens_seen": tokens_seen, "best": best}, name="model")
                save_training_state(state_ckpt_path, model, opt, scaler, cfg, args, step=step, optimizer_step=optimizer_step, tokens_seen=tokens_seen, best=best, ema_state=ema_state)
            if int(getattr(args, "sample_interval", 0) or 0) > 0 and step > 0 and step % int(args.sample_interval) == 0:
                generate_training_samples(model, tokenizer, getattr(args, "sample_prompts", ""), out / "samples.jsonl", device, step=step, max_new_tokens=int(getattr(args, "sample_max_new_tokens", 96) or 96))
            step += 1
        else:
            continue
        break

    if step % max(1, int(args.grad_accum_steps)) != 0:
        run_optimizer_step()
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    final_metrics = validate(raw_model, val_loader, device, max_batches=min(args.eval_batches, 5), prefetch_to_device=bool(getattr(args, "prefetch_to_device", False)))
    if domain_eval_loaders:
        final_metrics.update(validate_domains(raw_model, domain_eval_loaders, device, args))
    save_model(raw_model, out, final_metrics, name="model")
    if final_metrics["val_loss"] <= best:
        save_model(raw_model, out, final_metrics, name="best")
    save_ema_model(ema_state, raw_model, out, final_metrics)
    save_training_state(state_ckpt_path, raw_model, opt, scaler, cfg, args, step=step, optimizer_step=optimizer_step, tokens_seen=tokens_seen, best=min(best, float(final_metrics.get("val_loss", best))), ema_state=ema_state)
    done_row = {"done": True, "steps": step, "optimizer_steps": optimizer_step, "tokens_seen": tokens_seen, **final_metrics}
    print(json.dumps(done_row), flush=True)
    _append_train_log(args.metrics_jsonl, done_row)


def self_test(device_name: str = "cpu") -> None:
    device = resolve_device(device_name)
    if device.type == "cpu" and torch.get_num_threads() > 4:
        # The self-test uses tiny matrices where very large CPU thread pools are
        # slower than the math itself. Cap only the self-test path; production
        # training keeps the user's chosen runtime/thread settings.
        torch.set_num_threads(4)
    tmp = Path("/tmp/need_selftest")
    tmp.mkdir(parents=True, exist_ok=True)
    cfg = NeedConfig(
        d_model=16, n_layers=1, n_heads=4, block_size=8, d_ff=32,
        n_experts=1, moe_top_k=1, energy_rank=4, energy_steps=1,
        energy_min_steps=1, memory_slots=2, memory_rank=8, planner_horizons=1,
        image_codebook_size=64, vocab_size=Special.text_vocab + 64,
        image_grid=4, image_min_grid=4, image_max_grid=4, image_max_tokens=16,
    )
    cfg.disable_aux_components("diffusion", "image_contrastive")
    model = NeedModel(cfg).to(device)
    tok = ByteTokenizer()
    ids = tok.encode("NEED", add_bos=True, add_eos=True)
    ids = (ids + [cfg.pad_id] * cfg.block_size)[: cfg.block_size + 1]
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=device).unsqueeze(-1)
    logits, loss, aux = model(x, y)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    save_model(model, tmp / "out", {"self_test_loss": float(loss.detach().cpu())})
    loaded = load_model(tmp / "out", device=device)
    prompt = torch.tensor([tok.encode("N", add_bos=True)], dtype=torch.long, device=device)
    out = loaded.generate_text(prompt, max_new_tokens=1, temperature=0.0, proactive_aux_score=False)
    assert out.ndim == 2 and out.size(1) >= prompt.size(1)
    img_tokens = loaded.generate_image_tokens(prompt, grid=2, steps=1)
    assert img_tokens.shape[-1] == 4
    if Image is not None:
        img_tok = make_image_tokenizer(loaded.cfg)
        img = img_tok.decode_tokens(img_tokens[0].tolist(), grid=2, size=16)
        img.save(tmp / "out" / "selftest_image.png")
        if VisualTokenizerConfig is not None and VisualTokenizerVQVAE is not None:
            vcfg = VisualTokenizerConfig(codebook_size=64, embed_dim=8, hidden_dim=16, downsample=4, min_grid=2, max_grid=4, default_grid=2, max_image_tokens=16)
            vtok = VisualTokenizerVQVAE(vcfg).to(device).eval()
            arr = (np.linspace(0, 255, 16 * 16 * 3).reshape(16, 16, 3) % 255).astype(np.uint8)
            pil = Image.fromarray(arr, mode="RGB")
            vids, meta = vtok.encode_image(pil, add_special=True, grid=2, device=device)
            assert len([i for i in vids if i >= Special.text_vocab]) == 4
            dec = vtok.decode_tokens(vids, grid=2, size=16, device=device)
            dec.save(tmp / "out" / "selftest_learned_vq.png")
    print("NEED self-test passed")

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train NEED multimodal attention-free NEED model")
    p.add_argument("--data", type=str, default="")
    p.add_argument("--packed_data", type=str, default="", help="Flat binary token stream produced by --pack_data_to; fastest input path")
    p.add_argument("--packed_index", type=str, default="", help="Source-balanced packed index JSON produced by prepare_packed_dataset.py")
    p.add_argument("--packed_dtype", choices=["auto", "uint16", "u16", "int32", "i32", "uint32", "u32"], default="auto")
    p.add_argument("--allow_packed_vocab_mismatch", action="store_true", help="Permit packed-data vocab/tokenizer metadata to differ from the current model config. Use only for deliberate expert workflows.")
    p.add_argument("--pack_data_to", type=str, default="", help="Tokenize --data into a flat binary token file before training")
    p.add_argument("--pack_only", action="store_true", help="Exit after writing --pack_data_to")
    p.add_argument("--image_dir", type=str, default="")
    p.add_argument("--image_tokens", type=str, default="", help="Pre-tokenized image JSONL file or directory produced by need_raw_image_data.py")
    p.add_argument("--out_dir", type=str, default="need_out")
    p.add_argument("--init_from", type=str, default="", help="Optional NEED checkpoint to continue/fine-tune from.")
    p.add_argument("--init_prefer_best", action="store_true", help="Load best checkpoint from --init_from when available.")
    p.add_argument("--resume_from", type=str, default="", help="Resume a full training_state.pt checkpoint, including optimizer/scaler/RNG/counters.")
    p.add_argument("--resume_strict", action="store_true", help="Refuse to resume if the saved architecture config differs from the current one.")
    p.add_argument("--allow_unsafe_checkpoint_load", action="store_true", help="Allow legacy pickle checkpoint loading. Use only for trusted local checkpoints.")
    p.add_argument("--skip_packed_integrity_check", action="store_true", help="Skip packed-index file size/hash checks before training.")
    p.add_argument("--disable_signal_checkpoint", action="store_true", help="Do not install SIGTERM/SIGINT handlers for graceful checkpointing.")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--profile", choices=["custom"], default="custom", help="Named architecture buckets are removed; use explicit dimensions or --target_params.")
    p.add_argument("--recipe", type=str, default="", help="Training recipe such as fast, quality, baseline, or debug; changes training/runtime defaults only.")
    p.add_argument("--target_params", type=str, default="", help="Auto-size model parts for a target count such as 300M or 1.2B.")
    p.add_argument("--architecture", choices=["dense", "moe"], default="dense", help="Architecture used by --target_params auto-sizing.")
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=6)
    p.add_argument("--d_ff", type=int, default=0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--retention_impl", choices=["ssd", "selective"], default="ssd")
    p.add_argument("--ssd_conv_kernel", type=int, default=3)
    p.add_argument("--disable_adaptive_depth", action="store_true")
    p.add_argument("--compute_budget", type=float, default=0.72)
    p.add_argument("--min_compute_gate", type=float, default=0.08)
    p.add_argument("--depth_gate_temperature", type=float, default=1.0)
    p.add_argument("--disable_cooperative_steps", action="store_true", help="Disable the NEEDBlock cooperative step workspace/gates.")
    p.add_argument("--cooperative_step_summary_dim", type=int, default=NeedConfig().cooperative_step_summary_dim, help="Summary width for cross-step NEEDBlock workspace; 0 chooses automatically.")
    p.add_argument("--cooperative_step_context_strength", type=float, default=NeedConfig().cooperative_step_context_strength)
    p.add_argument("--cooperative_step_final_strength", type=float, default=NeedConfig().cooperative_step_final_strength)
    p.add_argument("--cooperative_step_budget", type=float, default=NeedConfig().cooperative_step_budget, help="Soft average gate budget across NEEDBlock internal stages.")
    p.add_argument("--cooperative_step_redundancy_target", type=float, default=NeedConfig().cooperative_step_redundancy_target, help="Cosine-similarity margin before cross-stage deltas are treated as redundant.")
    p.add_argument("--disable_role_separation", action="store_true", help="Disable role separation between temporal carrier, explicit memory, and equilibrium cleanup.")
    p.add_argument("--role_separation_strength", type=float, default=NeedConfig().role_separation_strength, help="Projection strength used to keep later NEEDBlock stages from duplicating retention/conv.")
    p.add_argument("--memory_condition_strength", type=float, default=NeedConfig().memory_condition_strength, help="How strongly explicit memory is conditioned on the retention/conv temporal carrier.")
    p.add_argument("--n_experts", type=int, default=4)
    p.add_argument("--conv_active_scales", type=int, default=NeedConfig().conv_active_scales, help="Number of routed multiscale conv branches to actually compute")
    p.add_argument("--moe_top_k", type=int, default=NeedConfig().moe_top_k)
    p.add_argument("--disable_shared_expert", action="store_true", help="Use one dense FFN path when n_experts=1 instead of adding a shared expert.")
    p.add_argument("--energy_rank", type=int, default=96)
    p.add_argument("--energy_steps", type=int, default=3)
    p.add_argument("--energy_min_steps", type=int, default=1)
    p.add_argument("--memory_slots", type=int, default=16)
    p.add_argument("--memory_rank", type=int, default=64)
    p.add_argument("--memory_mix", type=float, default=NeedConfig().memory_mix, help="Residual strength for the conditioned explicit memory stage; 0 is a matched-parameter no-memory ablation.")
    p.add_argument("--memory_chunk_size", type=int, default=32)
    p.add_argument("--pathway_conditioning_top_k", type=int, default=8)
    p.add_argument("--pathway_conditioning_dropout", type=float, default=0.0)
    p.add_argument("--pathway_conditioning_scale", type=float, default=0.18)
    p.add_argument("--pathway_conditioning_max_vectors", type=int, default=64)
    p.add_argument("--pathway_memory_slots", type=int, default=24)
    p.add_argument("--pathway_memory_top_k", type=int, default=6)
    p.add_argument("--pathway_memory_update_rate", type=float, default=0.15)
    p.add_argument("--planner_horizons", type=int, default=4)
    p.add_argument("--planner_transition_depth", type=int, default=2)
    p.add_argument("--disable_planner_block_space", action="store_true", help="Disable full-block latent planner horizon-scan path")
    p.add_argument("--planner_block_space_mix", type=float, default=NeedConfig().planner_block_space_mix, help="Blend of full-block planner states vs recurrent horizon states")
    p.add_argument("--planner_block_space_iters", type=int, default=NeedConfig().planner_block_space_iters, help="Cheap slot-attention iterations for full-block latent planning")
    p.add_argument("--disable_dvsd_planner_compound", action="store_true", help="Disable token-conditioned latent planner compounding for DVSD training/inference")
    p.add_argument("--dvsd_planner_compound_mix", type=float, default=NeedConfig().dvsd_planner_compound_mix, help="Blend compounded-planner logits into future DVSD slot logits")
    p.add_argument("--dvsd_planner_compound_step_size", type=float, default=NeedConfig().dvsd_planner_compound_step_size, help="Latent transition step size for DVSD planner compounding")
    p.add_argument("--dvsd_planner_compound_token_scale", type=float, default=NeedConfig().dvsd_planner_compound_token_scale, help="Strength of token-feedback residuals in DVSD planner compounding")
    p.add_argument("--dvsd_planner_compound_descent_scale", type=float, default=NeedConfig().dvsd_planner_compound_descent_scale, help="Strength of approximate logit-descent residuals in DVSD planner compounding")
    p.add_argument("--dvsd_planner_compound_top_k", type=int, default=NeedConfig().dvsd_planner_compound_top_k, help="Top-k logits used to approximate output-embedding expectation for compounding descent")
    p.add_argument("--aux_score_candidate_pool", type=int, default=1)
    p.add_argument("--aux_score_risk_threshold", type=float, default=0.72)
    p.add_argument("--aux_score_contradiction_threshold", type=float, default=0.65)
    p.add_argument("--aux_score_backtrack_window", type=int, default=3)
    p.add_argument("--aux_score_max_backtracks", type=int, default=0)
    p.add_argument("--disable_aux_score_controller", action="store_true")
    p.add_argument("--controller_temperature", type=float, default=1.0)
    p.add_argument("--latent_search_branches", type=int, default=1)
    p.add_argument("--latent_search_depth", type=int, default=0)
    p.add_argument("--cot_faithfulness_threshold", type=float, default=0.45)
    p.add_argument("--cot_usefulness_threshold", type=float, default=0.35)
    p.add_argument("--energy_routes", type=int, default=6)
    p.add_argument("--energy_route_steps", type=int, default=1)
    p.add_argument("--energy_route_strength", type=float, default=0.10)
    p.add_argument("--latent_slots", type=int, default=4)
    p.add_argument("--slot_attention_mode", choices=["pooled", "attention"], default=NeedConfig().slot_attention_mode, help="pooled is the efficient core default; attention is an ablation mode")
    p.add_argument("--latent_slot_conditioning_scale", type=float, default=0.12)
    p.add_argument("--slow_state_chunk", type=int, default=16)
    p.add_argument("--slow_state_strength", type=float, default=0.15)
    p.add_argument("--risk_gate_strength", type=float, default=0.18)
    p.add_argument("--object_program_slots", type=int, default=8)
    p.add_argument("--object_program_strength", type=float, default=0.18)
    p.add_argument("--output_modes", type=int, default=5)
    p.add_argument("--tokenizer", choices=["hf", "byte"], default="hf",
                    help="Text tokenizer to train with. 'hf' (default) loads a Hugging Face "
                         f"tokenizer (default model: {DEFAULT_HF_TOKENIZER_MODEL}); 'byte' uses "
                         "the exact byte-level fallback tokenizer.")
    p.add_argument("--tokenizer_model", type=str, default="",
                    help=f"HF hub id for --tokenizer=hf. Default: {DEFAULT_HF_TOKENIZER_MODEL}")
    p.add_argument("--image_codebook_size", type=int, default=512)
    p.add_argument("--image_grid", type=int, default=16)
    p.add_argument("--image_min_grid", type=int, default=8)
    p.add_argument("--image_max_grid", type=int, default=32)
    p.add_argument("--image_max_tokens", type=int, default=1024)
    p.add_argument("--image_coord_scale", type=float, default=0.25)
    p.add_argument("--image_local_contrastive_temperature", type=float, default=0.07)
    p.add_argument("--disable_image_2d_scan", action="store_true")
    p.add_argument("--image_2d_bidirectional", action="store_true", help="Opt into non-causal reverse image-grid scans for non-AR image objectives")
    p.add_argument("--image_2d_scan_strength", type=float, default=0.20)
    p.add_argument("--image_2d_scan_decay", type=float, default=0.75)
    p.add_argument("--region_word_sinkhorn_iters", type=int, default=0, help="Legacy no-op; region-word alignment uses linear moment matching")
    p.add_argument("--static_image_grid", action="store_true")
    p.add_argument("--image_mask_prob", type=float, default=0.35)
    p.add_argument("--image_ratio", type=float, default=0.25)
    # NEED learned visual tokenizer. If --visual_tokenizer is omitted, NEED falls back to the deterministic tokenizer.
    p.add_argument("--visual_tokenizer", type=str, default="", help="Directory containing visual_tokenizer_config.json and weights")
    p.add_argument("--visual_tokenizer_device", type=str, default="cpu", help="Device used by DataLoader-side image tokenizer")
    p.add_argument("--force_image_grid", type=int, default=0, help="Force learned tokenizer grid; 0 uses dynamic grid")
    p.add_argument("--train_visual_tokenizer", action="store_true", help="Pretrain a learned VQ tokenizer from --image_dir before NEED training")
    p.add_argument("--vq_embed_dim", type=int, default=128)
    p.add_argument("--vq_hidden_dim", type=int, default=128)
    p.add_argument("--vq_res_blocks", type=int, default=2)
    p.add_argument("--vq_downsample", type=int, default=16)
    p.add_argument("--vq_batch_size", type=int, default=8)
    p.add_argument("--vq_samples", type=int, default=10000)
    p.add_argument("--vq_steps", type=int, default=0, help="0 means use --max_steps//10, minimum 100 when pretraining")
    p.add_argument("--vq_lr", type=float, default=2e-4)
    p.add_argument("--vq_weight_decay", type=float, default=1e-4)
    p.add_argument("--vq_gan_weight", type=float, default=0.05)
    p.add_argument("--vq_gan_start", type=int, default=1000)
    p.add_argument("--vq_perceptual_weight", type=float, default=0.10)
    p.add_argument("--vq_edge_weight", type=float, default=0.08)
    p.add_argument("--vq_save_interval", type=int, default=1000)
    p.add_argument("--vq_curriculum", action="store_true", help="Progressively ramp visual tokenizer reconstruction/perceptual/GAN/code-usage objectives")
    p.add_argument("--vq_code_usage_weight", type=float, default=0.02)
    p.add_argument("--kernel_backend", choices=["auto", "torch", "triton"], default="auto")
    p.add_argument("--disable_fused_ssd_scan", action="store_true", help="Disable fused inference SSD scan kernels")
    p.add_argument("--disable_parallel_scan", action="store_true", help="Use the PyTorch chunked scan fallback instead of fused associative-scan recurrences")
    p.add_argument("--minimal_aux_metrics", action="store_true", help="Skip diagnostic-only auxiliary metrics that are not used in the loss")
    p.add_argument("--no_strict_linear_core", action="store_true", help="Disable strict linear-core clamps for explicit ablations")
    p.add_argument("--no_streaming_generation", action="store_true", help="Store checkpoints with streaming generation disabled by default")
    p.add_argument("--disable_exact_recall", action="store_true", help="Disable exact long-range associative recall path")
    p.add_argument("--exact_recall_dim", type=int, default=NeedConfig().exact_recall_dim, help="Recall projection width; 0 derives it from d_model.")
    p.add_argument("--exact_recall_top_k", type=int, default=NeedConfig().exact_recall_top_k, help="Recall commit top-k; 0 derives it from the candidate budget.")
    p.add_argument("--exact_recall_mix", type=float, default=NeedConfig().exact_recall_mix)
    p.add_argument("--exact_recall_temperature", type=float, default=NeedConfig().exact_recall_temperature)
    p.add_argument("--exact_recall_max_tokens", type=int, default=NeedConfig().exact_recall_max_tokens)
    p.add_argument("--exact_recall_max_candidates", type=int, default=NeedConfig().exact_recall_max_candidates, help="Bounded recall candidate cap; 0 derives it from context length.")
    p.add_argument("--disable_state_stabilization", action="store_true", help="Disable long-context state-space drift stabilizer")
    p.add_argument("--state_anchor_strength", type=float, default=0.035)
    p.add_argument("--state_drift_chunk", type=int, default=64)
    p.add_argument("--state_drift_target", type=float, default=0.14)
    p.add_argument("--state_norm_target", type=float, default=1.0)
    p.add_argument("--disable_objective_soft_budget", action="store_true", help="Disable normalization, warmup, and caps on auxiliary objectives relative to CE")
    p.add_argument("--objective_aux_ratio_cap", type=float, default=NeedConfig().objective_aux_ratio_cap, help="Maximum total absolute auxiliary contribution as a fraction of CE")
    p.add_argument("--objective_group_ratio_cap", type=float, default=NeedConfig().objective_group_ratio_cap, help="Maximum absolute contribution per auxiliary loss family as a fraction of CE")
    p.add_argument("--objective_softcap_min", type=float, default=NeedConfig().objective_softcap_min, help="Minimum CE reference used by auxiliary caps")
    p.add_argument("--objective_aux_warmup_steps", type=int, default=NeedConfig().objective_aux_warmup_steps, help="Micro-steps over which auxiliary weights ramp to full strength")
    p.add_argument("--objective_aux_min_scale", type=float, default=NeedConfig().objective_aux_min_scale, help="Initial auxiliary weight scale during warmup")
    p.add_argument("--objective_balance_ema_beta", type=float, default=NeedConfig().objective_balance_ema_beta, help="EMA beta for per-loss magnitude normalization")
    p.add_argument("--objective_loss_ema_floor", type=float, default=NeedConfig().objective_loss_ema_floor, help="Floor for per-loss normalization denominators")
    p.add_argument("--objective_term_abs_cap", type=float, default=NeedConfig().objective_term_abs_cap, help="Absolute cap on normalized auxiliary terms before soft budgeting")
    p.add_argument("--disable_objective_normalize_aux", action="store_true", help="Keep warmup/caps but disable EMA normalization of auxiliary magnitudes")
    p.add_argument("--objective_entropy_band_weight", type=float, default=NeedConfig().objective_entropy_band_weight, help="Small anti-collapse/anti-uniformity penalty for slot/router entropy bands")
    p.add_argument("--disable_objective_curriculum", action="store_true", help="Disable staged ramp-in of auxiliary loss families")
    p.add_argument("--objective_prediction_start_step", type=int, default=NeedConfig().objective_prediction_start_step, help="Objective step where prediction auxiliaries start ramping in")
    p.add_argument("--objective_latent_start_step", type=int, default=NeedConfig().objective_latent_start_step, help="Objective step where latent/routing auxiliaries start ramping in")
    p.add_argument("--objective_risk_start_step", type=int, default=NeedConfig().objective_risk_start_step, help="Objective step where risk/control auxiliaries start ramping in")
    p.add_argument("--objective_vision_start_step", type=int, default=NeedConfig().objective_vision_start_step, help="Objective step where vision/grounding auxiliaries start ramping in")
    p.add_argument("--objective_regularizer_start_step", type=int, default=NeedConfig().objective_regularizer_start_step, help="Objective step where regularizers start ramping in")
    p.add_argument("--objective_family_ramp_steps", type=int, default=NeedConfig().objective_family_ramp_steps, help="Smooth ramp length for each auxiliary family")
    p.add_argument("--disable_objective_quarantine", action="store_true", help="Disable automatic suppression of repeatedly pathological auxiliary objectives")
    p.add_argument("--objective_quarantine_patience", type=int, default=NeedConfig().objective_quarantine_patience, help="Repeated pathology observations before an objective is suppressed")
    p.add_argument("--objective_quarantine_decay", type=float, default=NeedConfig().objective_quarantine_decay, help="Multiplicative suppression applied to a quarantined objective")
    p.add_argument("--objective_quarantine_min_scale", type=float, default=NeedConfig().objective_quarantine_min_scale, help="Minimum scale for a quarantined objective")
    p.add_argument("--objective_quarantine_recovery_steps", type=int, default=NeedConfig().objective_quarantine_recovery_steps, help="Approximate steps for quarantined objectives to recover toward full scale")
    p.add_argument("--objective_pathology_clip_threshold", type=float, default=NeedConfig().objective_pathology_clip_threshold, help="Clip-rate EMA above which an objective is considered pathological")
    p.add_argument("--enable_objective_gradient_guard", action="store_true", help="Enable occasional CE-vs-aux gradient conflict probes; off by default for throughput")
    p.add_argument("--disable_objective_gradient_guard", action="store_true", help="Deprecated compatibility flag; gradient guard is off unless --enable_objective_gradient_guard is set")
    p.add_argument("--objective_gradient_guard_interval", type=int, default=NeedConfig().objective_gradient_guard_interval, help="Micro-step interval for gradient conflict probes")
    p.add_argument("--objective_gradient_guard_start_step", type=int, default=NeedConfig().objective_gradient_guard_start_step, help="First micro-step where gradient conflict probes may run")
    p.add_argument("--objective_gradient_guard_max_terms", type=int, default=NeedConfig().objective_gradient_guard_max_terms, help="Maximum auxiliary objectives to probe per guard pass")
    p.add_argument("--objective_gradient_guard_param_tensors", type=int, default=NeedConfig().objective_gradient_guard_param_tensors, help="Number of shared parameter tensors sampled for gradient conflict probes")
    p.add_argument("--objective_conflict_cosine_threshold", type=float, default=NeedConfig().objective_conflict_cosine_threshold, help="Cosine below which an auxiliary gradient is considered opposed to CE")
    p.add_argument("--objective_conflict_quarantine_patience", type=int, default=NeedConfig().objective_conflict_quarantine_patience, help="Repeated gradient conflicts before automatic quarantine")
    p.add_argument("--objective_ablation_interval", type=int, default=0, help="Run validation ablations for auxiliary families every N micro-steps; 0 disables")
    p.add_argument("--objective_ablation_eval_batches", type=int, default=3, help="Validation batches per objective-family ablation")
    p.add_argument("--objective_ablation_families", type=str, default="prediction,latent,risk,vision,regularizer", help="Comma-separated objective families to ablate during scheduled validation")
    p.add_argument("--objective_ablation_auto_quarantine", action="store_true", help="When a scheduled family ablation improves validation, suppress that family")
    p.add_argument("--objective_ablation_improve_threshold", type=float, default=0.0, help="Required validation improvement before auto-quarantining an ablated family")
    p.add_argument("--objective_ablation_quarantine_decay", type=float, default=0.70, help="Family quarantine decay used by validation-ablation auto-intervention")
    p.add_argument("--batch_size", type=int, default=8, help="Micro-batch size; 0 lets --auto_optimize probe the GPU.")
    p.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation micro-steps; 0 lets --auto_optimize derive it from --target_effective_batch_tokens.")
    p.add_argument("--target_tokens", type=parse_scaled_number, default=0, help="Stop after this many training tokens; accepts M/B/T suffixes.")
    p.add_argument("--target_effective_batch_tokens", type=parse_scaled_number, default=0, help="Desired tokens per optimizer step when grad accumulation is automatic.")
    p.add_argument("--max_steps", type=int, default=1000, help="Maximum micro-steps; auto-raised when --target_tokens is set.")
    p.add_argument("--train_samples", type=int, default=10000)
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader workers; -1 auto-selects from CPU/RAM.")
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr_schedule", choices=["constant", "cosine", "linear"], default="constant")
    p.add_argument("--warmup_steps", type=int, default=0, help="Optimizer-step LR warmup; 0 disables.")
    p.add_argument("--min_lr", type=float, default=0.0, help="Final LR for cosine/linear schedules.")
    p.add_argument("--lr_decay_steps", type=int, default=0, help="Optimizer steps for LR decay; 0 derives from target tokens/effective batch.")
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", choices=["off", "bf16", "fp16"], default="bf16")
    p.add_argument("--tf32", action="store_true", default=True, help="Allow TF32 matmuls on CUDA for fp32 fallback paths.")
    p.add_argument("--no_tf32", dest="tf32", action="store_false")
    p.add_argument("--matmul_precision", choices=["highest", "high", "medium"], default="high")
    p.add_argument("--cudnn_benchmark", action="store_true")
    p.add_argument("--optimizer_fused", action="store_true")
    p.add_argument("--optimizer_foreach", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile_mode", default="", choices=["", "default", "reduce-overhead", "max-autotune"])
    p.add_argument("--compile_fullgraph", action="store_true")
    p.add_argument("--compile_backend", default="inductor")
    p.add_argument("--compile_dynamic", action="store_true", default=True)
    p.add_argument("--compile_static", dest="compile_dynamic", action="store_false", help="Compile for fixed shapes; best for packed/drop_last training")
    p.add_argument("--compile_cudagraphs", action="store_true", help="Enable Inductor CUDA-graph capture hints for compiled fixed-shape steps")
    p.add_argument("--prefetch_to_device", action="store_true", help="Overlap host-to-device batch copies with compute on CUDA")
    p.add_argument("--drop_last", action="store_true", help="Drop incomplete batches to keep training shapes static")
    p.add_argument("--auto_optimize", action="store_true", help="Enable runtime defaults, worker selection, fused AdamW, and optional auto batch probing.")
    p.add_argument("--auto_batch", action="store_true", help="Probe the largest comfortable CUDA micro-batch.")
    p.add_argument("--auto_batch_max", type=int, default=64)
    p.add_argument("--target_vram_util", type=float, default=0.90)
    p.add_argument("--peak_tflops", type=float, default=0.0, help="Optional hardware peak TFLOP/s for MFU logging.")
    p.add_argument("--save_interval", type=int, default=1000, help="Save model and full training_state.pt every N micro-steps; 0 disables periodic full-state saves.")
    p.add_argument("--nan_recovery", action="store_true", help="Skip non-finite/spiking batches and lower LR instead of aborting.")
    p.add_argument("--nan_recovery_lr_decay", type=float, default=0.5)
    p.add_argument("--loss_spike_threshold", type=float, default=0.0, help="Treat loss > threshold * EMA as recoverable spike; 0 disables.")
    p.add_argument("--max_nonfinite_events", type=int, default=20, help="Abort after this many recovered bad-loss/gradient events; 0 disables the cap.")
    p.add_argument("--ema_decay", type=float, default=0.0, help="Optional EMA checkpoint decay; 0 disables.")
    p.add_argument("--module_diagnostics_interval", type=int, default=0, help="Log param/grad L2 by top-level module every N steps; 0 disables.")
    p.add_argument("--sample_prompts", type=str, default="", help="Text file of fixed prompts for periodic sample generation.")
    p.add_argument("--sample_interval", type=int, default=0, help="Generate fixed prompt samples every N steps; 0 disables.")
    p.add_argument("--sample_max_new_tokens", type=int, default=96)
    p.add_argument("--eval_data", action="append", default=[], help="Extra domain eval data as name=path; may be text/jsonl, packed .bin, or packed_index.json. Repeatable.")
    p.add_argument("--eval_data_bytes", type=parse_scaled_number, default=64*1024*1024, help="Bytes to load from each text-domain eval file; accepts M/B/T suffixes.")
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--metrics_jsonl", default="", help="Append structured training/eval rows here; default is out_dir/train_log.jsonl")
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--eval_batches", type=int, default=10)
    p.add_argument("--max_data_bytes", type=parse_scaled_number, default=0, help="Limit bytes read from --data; accepts M/B/T suffixes.")
    p.add_argument("--val_data_bytes", type=parse_scaled_number, default=0, help="Bytes sampled for validation in --stream_data mode; accepts M/B/T suffixes.")
    p.add_argument("--stream_data", action="store_true", help="Stream large JSONL/text corpuses instead of loading all tokens into RAM.")
    p.add_argument("--disable_dvsd_router", action="store_true", help="Disable learned DVSD slot-router training/inference head")
    p.add_argument("--dvsd_router_inference_mix", type=float, default=NeedConfig().dvsd_router_inference_mix, help="Blend of learned router vs heuristic slot budget at inference")
    p.add_argument("--dvsd_router_min_confidence", type=float, default=NeedConfig().dvsd_router_min_confidence, help="Minimum router confidence before it can steer the DVSD slot budget")
    p.add_argument("--dvsd_router_loss_threshold", type=float, default=NeedConfig().dvsd_router_loss_threshold, help="Token CE below this counts as a router-trainable easy future slot")
    p.add_argument("--dvsd_router_hard_loss_threshold", type=float, default=NeedConfig().dvsd_router_hard_loss_threshold, help="First-slot CE above this trains the router to collapse to one slot")
    p.add_argument("--dvsd_router_entropy_weight", type=float, default=NeedConfig().dvsd_router_entropy_weight, help="Small entropy bonus inside the DVSD router loss")
    # objective weights
    defaults = NeedConfig()
    for name in FUSED_LAMBDA_FIELDS:
        p.add_argument(f"--{name}", type=float, default=getattr(defaults, name))
    # ablations
    for flag in ["energy", "diffusion", "geodesic", "moe", "memory", "planner", "aux_score", "mtp", "adaptive_compute", "energy_routes", "latent_slots", "risk_signal_fusion", "image"]:
        p.add_argument(f"--disable_{flag}", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    setattr(args, "_explicit_args", explicit_arg_names(argv))
    if args.self_test:
        self_test(args.device)
    else:
        train(args)


if __name__ == "__main__":
    main()
