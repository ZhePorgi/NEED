#!/usr/bin/env python3
"""
Note: the implementation uses PyTorch fallbacks everywhere. Optional Triton kernels are routed
through need_kernels.py when available.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    from safetensors.torch import load_file as safe_load_file, save_file as safe_save_file
except Exception:  # pragma: no cover
    safe_load_file = None
    safe_save_file = None

try:
    import need_kernels as need_kernels
except Exception:  # pragma: no cover
    need_kernels = None


def _pil_resampling(name: str):
    if Image is None:
        return None
    enum = getattr(Image, "Resampling", Image)
    return getattr(enum, name)


# -----------------------------
# Tokenizers
# -----------------------------

class Special:
    pad = 0
    bos = 1
    eos = 2
    img_bos = 3
    img_eos = 4
    img_mask = 5
    sep = 6
    start_summary = 7
    end_summary = 8
    reserved = 16
    byte_start = 16
    byte_vocab = 256
    text_vocab = byte_start + byte_vocab


class ByteTokenizer:
    """Exact byte fallback tokenizer with stable special IDs."""
    vocab_size = Special.text_vocab
    pad_id = Special.pad
    bos_id = Special.bos
    eos_id = Special.eos

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(Special.bos)
        ids.extend(Special.byte_start + b for b in text.encode("utf-8", errors="replace"))
        if add_eos:
            ids.append(Special.eos)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        buf = bytearray()
        for idx in ids:
            i = int(idx)
            if Special.byte_start <= i < Special.byte_start + 256:
                buf.append(i - Special.byte_start)
            elif i in (Special.eos, Special.pad):
                continue
            elif i == Special.img_bos:
                buf.extend(b"<image>")
            elif i == Special.img_eos:
                buf.extend(b"</image>")
        return buf.decode("utf-8", errors="replace")

    def to_dict(self) -> Dict[str, object]:
        return {"type": "byte", "vocab_size": self.vocab_size}

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ByteTokenizer":
        return cls()


# Default subword tokenizer used unless byte-level fallback is explicitly requested.
# Can be overridden per-call or via the NEED_TOKENIZER_MODEL environment variable.
DEFAULT_HF_TOKENIZER_MODEL = os.environ.get("NEED_TOKENIZER_MODEL", "deepseek-ai/DeepSeek-V4-Pro")


class HFTokenizer:
    """Subword tokenizer backed by a Hugging Face `transformers` tokenizer
    (default: DeepSeek V4). This is the default NEED text tokenizer.

    IDs are shifted by `Special.reserved` so NEED's own control tokens
    (pad/bos/eos/img_bos/img_eos/img_mask/sep/...) keep the same stable,
    low-valued IDs as ByteTokenizer, and the same `ids >= Special.byte_start`
    / `ids >= text_vocab_size` style range checks used throughout the rest of
    this module keep working unchanged. Real subword token `i` from the
    underlying HF tokenizer is stored as `Special.reserved + i`.
    """

    def __init__(self, model_name: str = DEFAULT_HF_TOKENIZER_MODEL, revision: Optional[str] = None):
        try:
            from transformers import AutoTokenizer
        except Exception as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "HFTokenizer requires the `transformers` package "
                f"(pip install transformers) to load '{model_name}'. Original error: {exc}"
            ) from exc
        self.model_name = model_name
        self.revision = revision
        self._hf = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=True)
        base_vocab = int(getattr(self._hf, "vocab_size", 0) or 0)
        try:
            base_vocab = max(base_vocab, len(self._hf))
        except Exception:
            pass
        if base_vocab <= 0:
            raise RuntimeError(f"Could not determine vocab size for HF tokenizer '{model_name}'")
        self.base_vocab_size = base_vocab
        self.vocab_size = Special.reserved + self.base_vocab_size
        self.pad_id = Special.pad
        self.bos_id = Special.bos
        self.eos_id = Special.eos

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(Special.bos)
        hf_ids = self._hf.encode(text, add_special_tokens=False)
        ids.extend(Special.reserved + int(i) for i in hf_ids)
        if add_eos:
            ids.append(Special.eos)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        buf: List[int] = []
        parts: List[str] = []

        def _flush() -> None:
            if buf:
                parts.append(self._hf.decode(buf, skip_special_tokens=False))
                buf.clear()

        for idx in ids:
            i = int(idx)
            if Special.reserved <= i < Special.reserved + self.base_vocab_size:
                buf.append(i - Special.reserved)
            elif i in (Special.eos, Special.pad, Special.bos):
                continue
            elif i == Special.img_bos:
                _flush(); parts.append("<image>")
            elif i == Special.img_eos:
                _flush(); parts.append("</image>")
        _flush()
        return "".join(parts)

    def to_dict(self) -> Dict[str, object]:
        return {
            "type": "hf",
            "model_name": self.model_name,
            "revision": self.revision,
            "vocab_size": self.vocab_size,
            "base_vocab_size": self.base_vocab_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "HFTokenizer":
        return cls(
            model_name=str(data.get("model_name") or DEFAULT_HF_TOKENIZER_MODEL),
            revision=data.get("revision") or None,
        )


def load_tokenizer(data: Dict[str, object]):
    """Dispatch a tokenizer.json / to_dict() payload to the right tokenizer class."""
    kind = str((data or {}).get("type", "hf")).lower()
    if kind == "byte":
        return ByteTokenizer.from_dict(data)
    if kind == "hf":
        return HFTokenizer.from_dict(data)
    raise ValueError(f"Unknown tokenizer type: {kind!r}")


def build_default_tokenizer(model_name: Optional[str] = None, kind: Optional[str] = None):
    """Build NEED's default tokenizer.

    Defaults to the DeepSeek V4 tokenizer on Hugging Face (`HFTokenizer`). Pass
    `kind="byte"` (or set the NEED_TOKENIZER=byte environment variable) to use
    the exact byte-level fallback tokenizer instead. If the HF tokenizer can't
    be loaded (e.g. `transformers` isn't installed, or there's no network
    access to Hugging Face), this falls back to ByteTokenizer with a warning
    rather than hard-failing.
    """
    kind = (kind or os.environ.get("NEED_TOKENIZER") or "hf").strip().lower()
    if kind == "byte":
        return ByteTokenizer()
    if kind != "hf":
        raise ValueError(f"Unknown tokenizer kind: {kind!r}")
    try:
        return HFTokenizer(model_name or DEFAULT_HF_TOKENIZER_MODEL)
    except Exception as exc:
        print(
            f"[need_core] Could not load HF tokenizer "
            f"'{model_name or DEFAULT_HF_TOKENIZER_MODEL}' ({exc}); falling back to ByteTokenizer.",
        )
        return ByteTokenizer()


def load_tokenizer_for_dir(model_dir: Union[str, Path], default_kind: Optional[str] = None):
    """Load whichever tokenizer a saved NEED checkpoint directory was trained
    with, by reading its `tokenizer.json`. Falls back to `build_default_tokenizer`
    (byte-level, for backward compatibility with older checkpoints) if no
    tokenizer.json is present.
    """
    path = Path(model_dir) / "tokenizer.json"
    if path.exists():
        try:
            return load_tokenizer(load_json(path))
        except Exception as exc:
            print(f"[need_core] Failed to load tokenizer.json from {path} ({exc}); using default tokenizer.")
    return build_default_tokenizer(kind=default_kind or "byte")


# -----------------------------
# Dynamic image tokenizer
# -----------------------------

@dataclass
class ImageTokenizerConfig:
    image_codebook_size: int = 512
    min_grid: int = 8
    max_grid: int = 32
    default_grid: int = 16
    max_image_tokens: int = 1024
    token_offset: int = Special.text_vocab
    dynamic_grid: bool = True
    quality_threshold_low: float = 0.035
    quality_threshold_high: float = 0.090


class DynamicImageTokenizer:
    """Simple train-free VQ-style image tokenizer.

    It maps RGB patches to discrete codebook tokens using a deterministic color/texture
    codebook. This is deliberately lightweight so NEED can train and generate images without
    requiring a pre-trained VQGAN. For serious runs, replace the codebook with a learned
    VQ autoencoder but keep the same public API.
    """
    def __init__(self, cfg: ImageTokenizerConfig):
        self.cfg = cfg
        self.codebook = self._build_codebook(cfg.image_codebook_size).astype(np.float32)

    @staticmethod
    def _build_codebook(n: int) -> np.ndarray:
        # deterministic RGB grid plus luminance/texture anchors
        side = max(2, int(round(n ** (1.0 / 3.0))))
        levels = np.linspace(0.0, 1.0, side, dtype=np.float32)
        colors = np.array([[r, g, b] for r in levels for g in levels for b in levels], dtype=np.float32)
        if len(colors) < n:
            rng = np.random.default_rng(12345)
            extra = rng.random((n - len(colors), 3), dtype=np.float32)
            colors = np.concatenate([colors, extra], axis=0)
        return colors[:n]

    def choose_grid(self, image: "Image.Image") -> int:
        if not self.cfg.dynamic_grid:
            return int(self.cfg.default_grid)
        small = image.convert("RGB").resize((64, 64))
        arr = np.asarray(small, dtype=np.float32) / 255.0
        gx = np.abs(arr[:, 1:] - arr[:, :-1]).mean()
        gy = np.abs(arr[1:, :] - arr[:-1, :]).mean()
        complexity = float(0.5 * (gx + gy) + arr.var() * 0.1)
        if complexity < self.cfg.quality_threshold_low:
            grid = max(self.cfg.min_grid, self.cfg.default_grid // 2)
        elif complexity > self.cfg.quality_threshold_high:
            grid = min(self.cfg.max_grid, self.cfg.default_grid * 2)
        else:
            grid = self.cfg.default_grid
        while grid * grid > self.cfg.max_image_tokens and grid > self.cfg.min_grid:
            grid //= 2
        return int(grid)

    def encode_image(self, image: Union[str, Path, "Image.Image"], add_special: bool = True) -> Tuple[List[int], Dict[str, int]]:
        if Image is None:
            raise RuntimeError("Pillow is required for image tokenization")
        if not hasattr(image, "convert"):
            with Image.open(image) as im:  # type: ignore[arg-type]
                image = im.convert("RGB")
        else:
            image = image.convert("RGB")  # type: ignore[union-attr]
        grid = self.choose_grid(image)
        img = image.resize((grid, grid))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        flat = arr.reshape(-1, 3)
        # chunked nearest-neighbor for memory safety
        cb = self.codebook
        ids: List[int] = []
        for start in range(0, len(flat), 2048):
            x = flat[start : start + 2048]
            d = ((x[:, None, :] - cb[None, :, :]) ** 2).sum(axis=-1)
            ids.extend((d.argmin(axis=1) + self.cfg.token_offset).astype(np.int64).tolist())
        if add_special:
            ids = [Special.img_bos] + ids + [Special.img_eos]
        return ids, {"grid": grid, "height": grid, "width": grid}

    def decode_tokens(self, ids: Sequence[int], grid: Optional[int] = None, size: int = 256) -> "Image.Image":
        if Image is None:
            raise RuntimeError("Pillow is required for image decoding")
        raw = [int(i) - self.cfg.token_offset for i in ids if self.cfg.token_offset <= int(i) < self.cfg.token_offset + self.cfg.image_codebook_size]
        if not raw:
            raw = [0]
        if grid is None:
            grid = int(round(math.sqrt(len(raw))))
        grid = int(max(1, min(int(grid), int(self.cfg.max_grid))))
        needed = grid * grid
        if len(raw) < needed:
            raw = raw + [raw[-1]] * (needed - len(raw))
        raw = raw[:needed]
        arr = self.codebook[np.asarray(raw, dtype=np.int64)].reshape(grid, grid, 3)
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB").resize((size, size), resample=_pil_resampling("NEAREST"))

    def to_dict(self) -> Dict[str, object]:
        return asdict(self.cfg)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "DynamicImageTokenizer":
        return cls(ImageTokenizerConfig(**{k: v for k, v in data.items() if k in ImageTokenizerConfig.__dataclass_fields__}))


# -----------------------------
# Config
# -----------------------------

# Fused auxiliary objective configuration.  The active objective surface is five
# bundle lambdas plus per-component relative weights. Component weights keep
# ablation/debug granularity without making every component a separate optimizer
# objective.
AUX_FAMILY_NAMES: Tuple[str, ...] = ("prediction", "latent", "risk", "vision", "regularizer")

AUX_COMPONENT_GROUPS: Dict[str, str] = {
    # prediction-like objectives
    "mtp": "prediction",
    "dvsd_slot_ce": "prediction",
    "dvsd_consistency": "prediction",
    "dvsd_router": "prediction",
    "dvsd_compound_latent": "prediction",
    "dvsd_compound_ce": "prediction",
    "dvsd_compound_consistency": "prediction",
    "latent_planning": "prediction",
    "planner_ce": "prediction",
    "planning_consistency": "prediction",
    "diffusion": "prediction",

    # latent routing/control auxiliaries
    "latent_slot": "latent",
    "latent_slot_diversity": "latent",
    "latent_slot_entropy_band": "latent",
    "mixture_energy_router_energy": "latent",
    "energy_route_entropy_band": "latent",
    "timescale_consistency": "latent",

    # uncertainty and output-control objectives
    "risk_signal": "risk",
    "latent_divergence_loss": "risk",
    "output_mode_classifier": "risk",
    "output_mode_entropy_band": "risk",
    "aux_score": "risk",
    "controller": "risk",

    # image / grounding objectives
    "image_diffusion": "vision",
    "image_contrastive": "vision",
    "image_local_contrastive": "vision",
    "region_word_alignment": "vision",
    "image_spatial_smoothness": "vision",
    "object_program": "vision",
    "object_slot_entropy_band": "vision",

    # regularizers
    "equilibrium_residual": "regularizer",
    "energy": "regularizer",
    "moe_balance": "regularizer",
    "moe_router_z": "regularizer",
    "branch_entropy": "regularizer",
    "conv_scale_entropy": "regularizer",
    "geodesic": "regularizer",
    "path_straightness": "regularizer",
    "path_contractive": "regularizer",
    "energy_row_orth": "regularizer",
    "latent_norm": "regularizer",
    "memory_entropy": "regularizer",
    "memory_diversity": "regularizer",
    "adaptive_effort": "regularizer",
    "compute_budget": "regularizer",
    "pathway_memory_entropy": "regularizer",
    "energy_route_balance": "regularizer",
    "state_drift": "regularizer",
    "state_anchor": "regularizer",
    "exact_recall_entropy_floor": "regularizer",
    "memory_retention_overlap": "regularizer",
    "equilibrium_temporal_overlap": "regularizer",
    "coop_step_redundancy": "regularizer",
    "coop_gate_budget": "regularizer",
}

FUSED_OBJECTIVE_GROUPS: Dict[str, str] = {f"{name}_aux": name for name in AUX_FAMILY_NAMES}

DEFAULT_AUX_GROUP_LAMBDAS: Dict[str, float] = {
    "prediction": 0.18,
    "latent": 0.035,
    "risk": 0.060,
    "vision": 0.14,
    "regularizer": 0.020,
}

AUX_GROUP_LAMBDA_ATTRS: Dict[str, str] = {
    "prediction": "lambda_prediction_aux",
    "latent": "lambda_latent_aux",
    "risk": "lambda_risk_aux",
    "vision": "lambda_vision_aux",
    "regularizer": "lambda_regularizer_aux",
}

DEFAULT_AUX_COMPONENT_WEIGHTS: Dict[str, float] = {
    # These defaults mirror the old relative priorities, but they are no longer
    # independent lambdas.  They are normalized inside each bundle before one
    # fused family objective is added to CE.
    "mtp": 0.150,
    "dvsd_slot_ce": 0.025,
    "dvsd_consistency": 0.020,
    "dvsd_router": 0.015,
    "dvsd_compound_latent": 0.018,
    "dvsd_compound_ce": 0.020,
    "dvsd_compound_consistency": 0.008,
    "latent_planning": 0.040,
    "planner_ce": 0.040,
    "planning_consistency": 0.005,
    "diffusion": 0.020,

    "latent_slot": 0.018,
    "latent_slot_diversity": 0.002,
    "latent_slot_entropy_band": 0.003,
    "mixture_energy_router_energy": 0.006,
    "energy_route_entropy_band": 0.001,
    "timescale_consistency": 0.004,

    "risk_signal": 0.006,
    "latent_divergence_loss": 0.010,
    "output_mode_classifier": 0.004,
    "output_mode_entropy_band": 0.001,
    "aux_score": 0.040,
    "controller": 0.015,

    "image_diffusion": 0.100,
    "image_contrastive": 0.020,
    "image_local_contrastive": 0.040,
    "region_word_alignment": 0.035,
    "image_spatial_smoothness": 0.002,
    "object_program": 0.010,
    "object_slot_entropy_band": 0.002,

    "equilibrium_residual": 0.010,
    "energy": 0.0,
    "moe_balance": 0.010,
    "moe_router_z": 0.0001,
    "branch_entropy": -0.0005,
    "conv_scale_entropy": -0.0002,
    "geodesic": 0.010,
    "path_straightness": 0.005,
    "path_contractive": 0.002,
    "energy_row_orth": 0.0005,
    "latent_norm": 0.0001,
    "memory_entropy": -0.0002,
    "memory_diversity": 0.0002,
    "adaptive_effort": 0.0001,
    "compute_budget": 0.0002,
    "pathway_memory_entropy": -0.0005,
    "energy_route_balance": 0.000006,
    "state_drift": 0.006,
    "state_anchor": 0.003,
    "exact_recall_entropy_floor": 0.0003,
    # Role separation inside NEEDBlock: memory is conditioned on retention/conv
    # output but softly discouraged from duplicating it; equilibrium is restricted
    # to same-position cleanup rather than becoming another temporal memory path.
    "memory_retention_overlap": 0.0005,
    "equilibrium_temporal_overlap": 0.0004,
    # Cooperative NEEDBlock workspace: keep stages useful and non-duplicative
    # without exposing more lambda knobs. These are relative component weights
    # inside the fused regularizer_aux family.
    "coop_step_redundancy": 0.0005,
    "coop_gate_budget": 0.0002,
}

def default_aux_component_weights() -> Dict[str, float]:
    return dict(DEFAULT_AUX_COMPONENT_WEIGHTS)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def normalize_aux_component_weights(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a config dict with only canonical fused auxiliary fields preserved."""
    out = dict(data or {})
    weights = default_aux_component_weights()
    supplied = out.get("aux_component_weights")
    if isinstance(supplied, dict):
        for name, value in supplied.items():
            if name in AUX_COMPONENT_GROUPS:
                weights[name] = _safe_float(value, weights.get(name, 0.0))
    out["aux_component_weights"] = weights
    return out


@dataclass
class NeedConfig:
    # vocab / modality
    text_vocab_size: int = Special.text_vocab
    image_codebook_size: int = 512
    vocab_size: int = Special.text_vocab + 512
    image_token_offset: int = Special.text_vocab
    pad_id: int = Special.pad
    bos_id: int = Special.bos
    eos_id: int = Special.eos
    img_bos_id: int = Special.img_bos
    img_eos_id: int = Special.img_eos
    img_mask_id: int = Special.img_mask

    # architecture
    block_size: int = 512
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    d_ff: int = 0
    dropout: float = 0.1
    conv_kernel: int = 5
    n_conv_scales: int = 1
    conv_active_scales: int = 1  # strict core keeps one causal depthwise scale by default
    residual_scale: float = 0.55
    layer_scale_init: float = 0.20
    tie_embeddings: bool = True
    kernel_backend: str = "auto"
    fused_ssd_scan: bool = True
    parallel_scan: bool = True
    collect_aux_metrics: bool = True

    # retention / SSM-like recurrence
    retention_min_decay: float = 0.70
    retention_max_decay: float = 0.995
    retention_dynamic_scale: float = 0.75
    # SSD/Mamba-style structured dual retention.
    retention_impl: str = "ssd"
    ssd_conv_kernel: int = 3
    ssd_dt_min: float = 0.001
    ssd_dt_max: float = 0.20

    # adaptive compute: predicts gates for expensive memory/equilibrium/MoE paths.
    # In strict core, low-gate equilibrium and MoE tokens are compacted/skipped,
    # and whole memory path calls can be skipped when their gate is low.
    adaptive_depth: bool = True
    compute_budget: float = 0.72
    min_compute_gate: float = 0.08
    depth_gate_temperature: float = 1.0
    adaptive_depth_hard_skip: bool = True
    adaptive_depth_skip_threshold: float = 0.20
    # Static-shape hard-skip masking for batched execution.
    adaptive_depth_static_masking: bool = True
    # Static-shape MoE dispatch for compile-friendly experiments.
    moe_static_dispatch: bool = False

    # Cooperative NEEDBlock workspace. Internal block stages publish compact
    # summaries, read earlier stage outputs, and are softly gated by their
    # marginal contribution. This makes the block a small cooperative graph
    # instead of a blind fixed chain.
    cooperative_steps: bool = True
    cooperative_step_summary_dim: int = 0  # 0 => auto: min(128, d_model // 4)
    cooperative_step_context_strength: float = 0.12
    cooperative_step_final_strength: float = 0.08
    cooperative_step_gate_bias_init: float = 2.0
    cooperative_step_budget: float = 0.82
    cooperative_step_redundancy_target: float = 0.15

    # Role-separated NEEDBlock. Retention/conv are treated as the temporal
    # carrier. Memory is queried and written from the residual innovation after
    # those temporal stages, and equilibrium is kept as current-position cleanup.
    role_separation: bool = True
    role_separation_strength: float = 0.35
    memory_condition_strength: float = 0.35

    # energy equilibrium
    energy_rank: int = 96
    energy_steps: int = 3
    energy_min_steps: int = 1
    energy_early_exit: bool = True
    # Compatibility knob retained for older configs.
    energy_final_check: bool = False
    energy_step_size: float = 0.55
    energy_row_norm: float = 0.70
    min_precision: float = 0.25
    adaptive_energy: bool = True
    adaptive_residual_threshold: float = 0.035
    adaptive_effort_alpha: float = 5.0

    # MoE / memory
    n_experts: int = 4
    moe_top_k: int = 1
    moe_router_jitter: float = 0.0
    moe_dropout: float = 0.0
    moe_use_shared_expert: bool = False
    memory_chunk_size: int = 32
    memory_slots: int = 16
    memory_rank: int = 64
    memory_mix: float = 0.35

    # Learned bounded associative recall.
    exact_recall: bool = True
    exact_recall_dim: int = 0  # 0 means scale-derived from d_model
    exact_recall_top_k: int = 0  # 0 means derive from the candidate budget
    exact_recall_mix: float = 0.10
    exact_recall_temperature: float = 0.10
    exact_recall_max_tokens: int = 4096
    exact_recall_max_candidates: int = 0  # 0 means derive from context length

    # state-space drift control for long contexts. This adds a small learned
    # anchor/correction path and losses that discourage hidden-state norm drift,
    # chunk-to-chunk random walks, and anchor detachment over long histories.
    state_stabilization: bool = True
    state_anchor_strength: float = 0.035
    state_drift_chunk: int = 64
    state_drift_target: float = 0.14
    state_norm_target: float = 1.0

    # Multi-objective loss control. Auxiliary objectives are normalized, warmed
    # in, and capped relative to CE so they cannot dominate by finding shortcut
    # minima.  Group caps prevent whole families of losses from overpowering the
    # token objective, while entropy-band penalties discourage both collapse and
    # uniform, uninformative routers/slots.
    objective_soft_budget: bool = True
    objective_aux_ratio_cap: float = 0.45
    objective_group_ratio_cap: float = 0.18
    objective_softcap_min: float = 0.02
    objective_aux_warmup_steps: int = 2000
    objective_aux_min_scale: float = 0.05
    objective_balance_ema_beta: float = 0.98
    objective_loss_ema_floor: float = 1e-3
    objective_term_abs_cap: float = 25.0
    objective_normalize_aux: bool = True
    objective_entropy_band_weight: float = 0.15

    # Deeper objective safety.  Curriculum gates stage auxiliary families in
    # instead of letting every head steer the trunk immediately.  Quarantine
    # suppresses repeatedly pathological objectives.  Train-time gradient
    # guard options are consumed by train.py for occasional CE-vs-aux conflict
    # probes without paying that cost every micro-step.
    objective_curriculum_enabled: bool = True
    objective_prediction_start_step: int = 100
    objective_latent_start_step: int = 400
    objective_risk_start_step: int = 700
    objective_vision_start_step: int = 900
    objective_regularizer_start_step: int = 0
    objective_family_ramp_steps: int = 1200
    objective_quarantine_enabled: bool = True
    objective_quarantine_patience: int = 8
    objective_quarantine_decay: float = 0.50
    objective_quarantine_min_scale: float = 0.05
    objective_quarantine_recovery_steps: int = 2000
    objective_pathology_clip_threshold: float = 0.60
    objective_pathology_nonfinite_threshold: float = 0.0
    objective_gradient_guard: bool = False
    objective_gradient_guard_interval: int = 50
    objective_gradient_guard_start_step: int = 200
    objective_gradient_guard_max_terms: int = 8
    objective_gradient_guard_param_tensors: int = 24
    objective_conflict_cosine_threshold: float = -0.10
    objective_conflict_quarantine_patience: int = 3

    # Linear-core guardrail.  When enabled, runtime/search knobs are kept in the
    # single-pass core. Shape-defining architecture fields are not mutated here,
    # so old checkpoints and explicit ablations keep their declared tensor sizes.
    # Sequence-token work is O(T) for fixed model width/caps.
    strict_linear_core: bool = True

    # Stateful generation cache.  The default text decoder uses this in strict
    # linear core mode so autoregressive inference advances one token through
    # recurrent states instead of re-running the whole context window each step.
    streaming_generation: bool = True
    streaming_cache_max_tokens: int = 4096
    streaming_position_mode: str = "clamp"  # clamp | mod; clamp matches windowed AR positions after block_size
    streaming_fallback_on_unsupported: bool = True

    # latent pathway conditioning / planner / aux_score
    pathway_conditioning_top_k: int = 8
    pathway_conditioning_dropout: float = 0.0
    pathway_conditioning_scale: float = 0.18
    pathway_conditioning_max_vectors: int = 64
    planner_horizons: int = 4
    planner_transition_depth: int = 2
    planner_logit_blend: float = 0.12
    planner_block_space_enabled: bool = True
    planner_block_space_mix: float = 0.85
    planner_block_space_iters: int = 1

    # DVSD planner compounding.  During virtual-slot decoding, each sampled or
    # provisional token can apply a cheap latent update before later slots are
    # predicted.  The update uses token feedback plus an approximate negative CE
    # gradient in hidden space, so later tokens in the same block benefit from
    # earlier decisions without running the full trunk once per token.
    dvsd_planner_compound_enabled: bool = True
    dvsd_planner_compound_mix: float = 0.35
    dvsd_planner_compound_step_size: float = 0.65
    dvsd_planner_compound_token_scale: float = 0.18
    dvsd_planner_compound_descent_scale: float = 0.22
    dvsd_planner_compound_top_k: int = 32

    aux_score_logit_scale: float = 0.10
    aux_score_proactive: bool = False
    aux_score_candidate_pool: int = 1
    aux_score_risk_threshold: float = 0.72
    aux_score_contradiction_threshold: float = 0.65
    aux_score_backtrack_window: int = 3
    aux_score_max_backtracks: int = 0
    aux_score_controller: bool = True
    controller_temperature: float = 1.0
    latent_search_branches: int = 1
    latent_search_depth: int = 0
    cot_faithfulness_threshold: float = 0.45
    cot_usefulness_threshold: float = 0.35

    # working-memory tape for ordered pathway conditioning
    pathway_memory_slots: int = 24
    pathway_memory_top_k: int = 6
    pathway_memory_update_rate: float = 0.15

    # image spatial reasoning and region-word alignment
    image_2d_scan: bool = True
    image_2d_bidirectional: bool = False  # non-causal reverse image scans are opt-in only
    image_2d_scan_strength: float = 0.20
    image_2d_scan_decay: float = 0.75
    region_word_sinkhorn_iters: int = 0  # legacy no-op; region-word loss is linear moment matching

    # cognition upgrades: mixture energy routing, latent slot attention,
    # risk-signal fusion, and output-mode control.
    energy_routes: int = 6
    energy_route_steps: int = 1
    energy_route_strength: float = 0.10
    latent_slots: int = 4
    slot_attention_mode: str = "pooled"  # pooled | attention; pooled is the efficient core default
    latent_slot_conditioning_scale: float = 0.12
    slow_state_chunk: int = 16
    slow_state_strength: float = 0.15
    risk_gate_strength: float = 0.18
    object_program_slots: int = 8
    object_program_strength: float = 0.18
    output_modes: int = 5  # none, short summary, full CoT, multi-CoT, renderer-only

    # objectives. Five fused family lambdas control the public auxiliary surface.
    # Per-component weights below are relative within a family and are normalized
    # before the fused objective is added.
    n_predict_heads: int = 4
    label_smoothing: float = 0.0
    token_dropout: float = 0.0
    fused_aux_losses: bool = True
    fused_aux_component_normalize: bool = True
    fused_aux_component_floor: float = 1e-3
    lambda_prediction_aux: float = DEFAULT_AUX_GROUP_LAMBDAS["prediction"]
    lambda_latent_aux: float = DEFAULT_AUX_GROUP_LAMBDAS["latent"]
    lambda_risk_aux: float = DEFAULT_AUX_GROUP_LAMBDAS["risk"]
    lambda_vision_aux: float = DEFAULT_AUX_GROUP_LAMBDAS["vision"]
    lambda_regularizer_aux: float = DEFAULT_AUX_GROUP_LAMBDAS["regularizer"]
    aux_component_weights: Dict[str, float] = field(default_factory=default_aux_component_weights)

    # dynamic nonsequential / multi-token prediction decoder defaults. These are
    # inference knobs stored with the config so checkpoints can reproduce decoding
    # behavior, but they do not add parameters or change old checkpoint loading.
    nonseq_dynamic: bool = True
    nonseq_min_heads: int = 1
    nonseq_max_heads: int = 4
    nonseq_accept_top_k: int = 20
    nonseq_accept_min_prob: float = 0.015
    nonseq_accept_max_logprob_gap: float = 5.0
    nonseq_risk_threshold: float = 0.78
    nonseq_contradiction_threshold: float = 0.72
    nonseq_repetition_threshold: float = 0.88
    nonseq_entropy_easy: float = 0.45
    nonseq_entropy_hard: float = 0.82
    nonseq_min_draft_prob: float = 0.010
    nonseq_max_head_entropy: float = 0.92
    nonseq_tree_candidates: int = 1
    nonseq_branch_top_k: int = 1
    nonseq_aux_score_weight: float = 0.0
    nonseq_fallback_to_ar: bool = True

    # Slot-refinement nonsequential decoder. This is the default successor to
    # aux_scored MTP acceptance: it opens a virtual future canvas, samples/refines
    # positions by confidence, and commits the canvas directly without an
    # accept/reject aux-score pass. The canonical public decoder does not run
    # longest-prefix aux_scored acceptance; the old aux_scored routine remains a
    # directly named compatibility method for offline ablations only.
    nonseq_decode_style: str = "slot_refine"
    nonseq_refine_steps: int = 3
    nonseq_refine_causal_blend: float = 0.55
    nonseq_refine_confidence_floor: float = 0.0
    nonseq_refine_temperature_decay: float = 0.82
    nonseq_refine_lock_schedule: str = "cosine"
    nonseq_refine_resample_locked: bool = False

    # DVSD slot filling. In the default path a virtual block is a private canvas:
    # slots are filled sequentially, but the next slot can be selected by current
    # confidence rather than by left-to-right order. Once sampled, slots are
    # committed directly.
    dvsd_slot_order: str = "confidence"
    dvsd_freeze_sampled_slots: bool = True

    # DVSD-native training and learned routing. These make Dynamic Virtual Slot
    # Decoding a trained behavior rather than only an inference heuristic.  The
    # router predicts the local slot budget 1..n_predict_heads; auxiliary losses
    # teach direct-commit future slots to agree with teacher-forced AR logits and
    # train the router toward the longest low-loss future prefix.
    dvsd_router_enabled: bool = True
    dvsd_router_inference_mix: float = 0.65
    dvsd_router_min_confidence: float = 0.20
    dvsd_router_loss_threshold: float = 3.25
    dvsd_router_hard_loss_threshold: float = 6.0
    dvsd_router_entropy_weight: float = 0.002
    geodesic_target: float = 0.10
    contract_kappa: float = 0.92

    # dynamic image/tokenizer quality
    image_grid: int = 16
    image_min_grid: int = 8
    image_max_grid: int = 32
    image_max_tokens: int = 1024
    dynamic_image_grid: bool = True
    image_coord_scale: float = 0.25
    image_local_contrastive_temperature: float = 0.07

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NeedConfig":
        normalized = normalize_aux_component_weights(dict(data or {}))
        # Accept common historical/config-file aliases without letting them
        # override an explicitly supplied canonical field.  Silently dropping
        # these names can rebuild an old checkpoint with default tensor shapes,
        # which then fails at load time or, worse, runs with the wrong topology.
        aliases = {
            "hidden_size": "d_model",
            "n_embd": "d_model",
            "num_layers": "n_layers",
            "num_hidden_layers": "n_layers",
            "num_heads": "n_heads",
            "num_attention_heads": "n_heads",
            "ffn_dim": "d_ff",
            "intermediate_size": "d_ff",
            "max_position_embeddings": "block_size",
            "max_seq_len": "block_size",
            "seq_len": "block_size",
            "context_length": "block_size",
            "n_latent_slots": "latent_slots",
            "num_latent_slots": "latent_slots",
            "planner_horizon": "planner_horizons",
            "planner_horizon_count": "planner_horizons",
            "num_experts": "n_experts",
            "top_k_experts": "moe_top_k",
            "image_vocab_size": "image_codebook_size",
            "image_vocab_offset": "image_token_offset",
        }
        for old_name, new_name in aliases.items():
            if new_name not in normalized and old_name in normalized:
                normalized[new_name] = normalized[old_name]
        return cls(**{k: v for k, v in normalized.items() if k in cls.__dataclass_fields__})

    def aux_group_lambda(self, group: str) -> float:
        if group not in AUX_FAMILY_NAMES:
            return 0.0
        attr = AUX_GROUP_LAMBDA_ATTRS.get(group)
        return _safe_float(getattr(self, attr, 0.0), 0.0) if attr is not None else 0.0

    def aux_component_weight(self, name: str) -> float:
        weights = self.aux_component_weights if isinstance(self.aux_component_weights, dict) else {}
        return _safe_float(weights.get(name, 0.0), 0.0)

    def aux_component_enabled(self, name: str) -> bool:
        group = AUX_COMPONENT_GROUPS.get(name)
        if group is None or self.aux_group_lambda(group) == 0.0:
            return False
        return self.aux_component_weight(name) != 0.0

    def set_aux_component_weight(self, name: str, value: float) -> None:
        if not isinstance(self.aux_component_weights, dict):
            self.aux_component_weights = default_aux_component_weights()
        if name in AUX_COMPONENT_GROUPS:
            self.aux_component_weights[name] = _safe_float(value, 0.0)

    def disable_aux_components(self, *names: str) -> None:
        for name in names:
            self.set_aux_component_weight(str(name), 0.0)

    def validate(self) -> None:
        self.text_vocab_size = int(max(Special.text_vocab, self.text_vocab_size))
        self.image_codebook_size = int(max(1, self.image_codebook_size))
        self.vocab_size = int(self.text_vocab_size + self.image_codebook_size)
        self.image_token_offset = int(self.text_vocab_size)
        self.pad_id = int(self.pad_id)
        self.bos_id = int(self.bos_id)
        self.eos_id = int(self.eos_id)
        self.d_model = int(max(8, self.d_model))
        self.n_layers = int(max(1, self.n_layers))
        self.n_heads = int(max(1, min(self.n_heads, self.d_model)))
        while self.d_model % self.n_heads != 0 and self.n_heads > 1:
            self.n_heads -= 1
        if self.d_model % max(1, self.n_heads) != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_ff = int(max(0, self.d_ff))
        self.dropout = float(min(max(_safe_float(self.dropout, 0.0), 0.0), 0.95))
        self.token_dropout = float(min(max(_safe_float(getattr(self, "token_dropout", 0.0), 0.0), 0.0), 0.95))
        self.label_smoothing = float(min(max(_safe_float(getattr(self, "label_smoothing", 0.0), 0.0), 0.0), 0.49))
        self.conv_kernel = int(max(1, self.conv_kernel))
        if self.conv_kernel % 2 == 0:
            self.conv_kernel += 1
        self.n_conv_scales = int(max(1, self.n_conv_scales))
        self.residual_scale = float(min(max(_safe_float(self.residual_scale, 0.55), 0.0), 2.0))
        self.layer_scale_init = float(min(max(_safe_float(self.layer_scale_init, 0.20), 0.0), 1.0))
        self.retention_min_decay = float(min(max(_safe_float(self.retention_min_decay, 0.70), 0.0), 0.9999))
        self.retention_max_decay = float(min(max(_safe_float(self.retention_max_decay, 0.995), self.retention_min_decay), 0.9999))
        self.retention_dynamic_scale = float(min(max(_safe_float(self.retention_dynamic_scale, 0.75), 0.0), 4.0))
        self.retention_impl = str(getattr(self, "retention_impl", "ssd") or "ssd").lower()
        if self.retention_impl not in {"ssd", "selective"}:
            self.retention_impl = "ssd"
        self.kernel_backend = str(getattr(self, "kernel_backend", "auto") or "auto").lower()
        if self.kernel_backend not in {"auto", "torch", "torch_loop", "triton"}:
            self.kernel_backend = "auto"
        self.ssd_conv_kernel = int(max(1, self.ssd_conv_kernel))
        if self.ssd_conv_kernel % 2 == 0:
            self.ssd_conv_kernel += 1
        self.ssd_dt_min = float(max(1e-6, _safe_float(self.ssd_dt_min, 0.001)))
        self.ssd_dt_max = float(max(self.ssd_dt_min, _safe_float(self.ssd_dt_max, 0.20)))
        self.compute_budget = float(min(max(_safe_float(self.compute_budget, 0.72), 0.0), 1.0))
        self.min_compute_gate = float(min(max(_safe_float(self.min_compute_gate, 0.08), 0.0), 1.0))
        self.depth_gate_temperature = float(max(1e-6, _safe_float(self.depth_gate_temperature, 1.0)))
        self.adaptive_residual_threshold = float(max(0.0, _safe_float(self.adaptive_residual_threshold, 0.035)))
        self.energy_step_size = float(min(max(_safe_float(self.energy_step_size, 0.55), 0.0), 2.0))
        self.energy_row_norm = float(max(1e-6, _safe_float(self.energy_row_norm, 0.70)))
        self.min_precision = float(max(1e-6, _safe_float(self.min_precision, 0.25)))
        self.image_2d_scan_strength = float(min(max(_safe_float(getattr(self, "image_2d_scan_strength", 0.0), 0.0), 0.0), 2.0))
        self.image_2d_scan_decay = float(min(max(_safe_float(getattr(self, "image_2d_scan_decay", 0.95), 0.95), 0.0), 0.9999))
        self.image_coord_scale = float(min(max(_safe_float(getattr(self, "image_coord_scale", 0.25), 0.25), -4.0), 4.0))
        self.image_local_contrastive_temperature = float(max(1e-6, _safe_float(getattr(self, "image_local_contrastive_temperature", 0.07), 0.07)))
        self.n_experts = int(max(1, self.n_experts))
        self.moe_top_k = int(max(1, min(self.moe_top_k, self.n_experts)))
        self.memory_slots = int(max(1, self.memory_slots))
        self.memory_rank = int(max(1, min(self.memory_rank, self.d_model)))
        self.energy_rank = int(max(1, min(self.energy_rank, self.d_model)))
        self.image_min_grid = int(max(1, self.image_min_grid))
        self.image_max_grid = int(max(self.image_min_grid, self.image_max_grid))
        self.image_grid = int(max(self.image_min_grid, min(self.image_grid, self.image_max_grid)))
        self.image_max_tokens = int(max(1, self.image_max_tokens))
        self.energy_steps = int(max(1, self.energy_steps))
        self.energy_min_steps = int(max(1, min(self.energy_min_steps, self.energy_steps)))
        if self.block_size < 8:
            raise ValueError("block_size is too small")
        self.exact_recall_mix = float(min(max(_safe_float(getattr(self, "exact_recall_mix", 0.10), 0.10), 0.0), 1.0))
        self.exact_recall_temperature = float(max(1e-6, _safe_float(getattr(self, "exact_recall_temperature", 0.10), 0.10)))
        self.exact_recall_max_tokens = int(max(2, getattr(self, "exact_recall_max_tokens", 4096)))
        if int(self.exact_recall_dim) <= 0:
            # Smooth size-neutral width: grows with model width without naming a
            # particular model scale. Rounded to a multiple of eight for kernels.
            self.exact_recall_dim = int(max(8, min(self.d_model, round((self.d_model ** 0.5) * 8 / 8) * 8)))
        self.exact_recall_dim = int(max(8, min(self.d_model, self.exact_recall_dim)))
        if int(self.exact_recall_max_candidates) <= 0:
            raw_c = math.sqrt(float(max(16, self.block_size))) * 4.0
            self.exact_recall_max_candidates = int(max(32, min(192, round(raw_c / 16.0) * 16)))
        if int(self.exact_recall_top_k) <= 0:
            raw_k = math.sqrt(float(max(4, self.exact_recall_max_candidates)))
            self.exact_recall_top_k = int(max(2, min(16, round(raw_k / 2.0) * 2)))
        self.exact_recall_top_k = int(max(1, self.exact_recall_top_k))
        self.exact_recall_max_candidates = int(max(self.exact_recall_top_k, self.exact_recall_max_candidates))
        if bool(getattr(self, "strict_linear_core", True)):
            # Strict mode clamps runtime/search knobs while preserving architecture shapes.
            self.conv_active_scales = int(min(max(1, self.conv_active_scales), max(1, self.n_conv_scales)))
            self.moe_top_k = int(min(max(1, self.moe_top_k), 1))
            self.moe_use_shared_expert = False
            self.energy_final_check = False  # compatibility only; final residual is always refreshed
            self.objective_gradient_guard = False
            self.pathway_conditioning_top_k = int(min(max(1, self.pathway_conditioning_top_k), self.pathway_conditioning_max_vectors, 8))
            self.pathway_memory_top_k = int(min(max(1, self.pathway_memory_top_k), self.pathway_memory_slots, 8))
            self.planner_block_space_iters = int(min(max(1, self.planner_block_space_iters), 2))
            self.exact_recall_max_candidates = int(min(max(self.exact_recall_top_k, self.exact_recall_max_candidates), 192))
            self.aux_score_proactive = False
            self.aux_score_candidate_pool = 1
            self.aux_score_max_backtracks = 0
            self.latent_search_depth = 0
            self.latent_search_branches = 1
            self.nonseq_tree_candidates = 1
            self.nonseq_branch_top_k = 1
            self.nonseq_aux_score_weight = 0.0
            self.slot_attention_mode = "pooled"
        self.streaming_generation = bool(getattr(self, "streaming_generation", True))
        self.streaming_cache_max_tokens = int(max(16, min(int(getattr(self, "streaming_cache_max_tokens", 4096)), 65536)))
        if str(getattr(self, "streaming_position_mode", "clamp")).lower() not in {"mod", "clamp"}:
            self.streaming_position_mode = "clamp"
        self.streaming_fallback_on_unsupported = bool(getattr(self, "streaming_fallback_on_unsupported", True))
        self.state_drift_chunk = int(max(4, self.state_drift_chunk))
        self.objective_aux_ratio_cap = float(max(0.05, self.objective_aux_ratio_cap))
        self.objective_group_ratio_cap = float(max(0.01, min(self.objective_aux_ratio_cap, self.objective_group_ratio_cap)))
        self.objective_softcap_min = float(max(1e-6, self.objective_softcap_min))
        self.objective_aux_warmup_steps = int(max(0, self.objective_aux_warmup_steps))
        self.objective_aux_min_scale = float(min(max(self.objective_aux_min_scale, 0.0), 1.0))
        self.objective_balance_ema_beta = float(min(max(self.objective_balance_ema_beta, 0.0), 0.9999))
        self.objective_loss_ema_floor = float(max(1e-8, self.objective_loss_ema_floor))
        self.objective_term_abs_cap = float(max(1.0, self.objective_term_abs_cap))
        self.objective_entropy_band_weight = float(max(0.0, self.objective_entropy_band_weight))
        self.fused_aux_losses = bool(getattr(self, "fused_aux_losses", True))
        self.fused_aux_component_normalize = bool(getattr(self, "fused_aux_component_normalize", True))
        self.fused_aux_component_floor = float(max(1e-8, _safe_float(getattr(self, "fused_aux_component_floor", 1e-3), 1e-3)))
        for family in AUX_FAMILY_NAMES:
            attr = AUX_GROUP_LAMBDA_ATTRS[family]
            setattr(self, attr, float(max(0.0, _safe_float(getattr(self, attr, DEFAULT_AUX_GROUP_LAMBDAS[family]), DEFAULT_AUX_GROUP_LAMBDAS[family]))))
        weights = default_aux_component_weights()
        if isinstance(getattr(self, "aux_component_weights", None), dict):
            for name, value in self.aux_component_weights.items():
                if name in AUX_COMPONENT_GROUPS:
                    weights[name] = _safe_float(value, weights.get(name, 0.0))
        # Preserve exact zeros for ablations and profile disables; otherwise keep
        # weights finite and bounded so a bad JSON value cannot dominate a bundle.
        cap = float(max(1.0, self.objective_term_abs_cap))
        self.aux_component_weights = {name: float(max(-cap, min(cap, _safe_float(value, 0.0)))) for name, value in weights.items() if name in AUX_COMPONENT_GROUPS}
        self.objective_prediction_start_step = int(max(0, self.objective_prediction_start_step))
        self.objective_latent_start_step = int(max(0, self.objective_latent_start_step))
        self.objective_risk_start_step = int(max(0, self.objective_risk_start_step))
        self.objective_vision_start_step = int(max(0, self.objective_vision_start_step))
        self.objective_regularizer_start_step = int(max(0, self.objective_regularizer_start_step))
        self.objective_family_ramp_steps = int(max(1, self.objective_family_ramp_steps))
        self.objective_quarantine_patience = int(max(1, self.objective_quarantine_patience))
        self.objective_quarantine_decay = float(min(max(self.objective_quarantine_decay, 0.0), 1.0))
        self.objective_quarantine_min_scale = float(min(max(self.objective_quarantine_min_scale, 0.0), 1.0))
        self.objective_quarantine_recovery_steps = int(max(1, self.objective_quarantine_recovery_steps))
        self.objective_pathology_clip_threshold = float(min(max(self.objective_pathology_clip_threshold, 0.0), 1.0))
        self.objective_pathology_nonfinite_threshold = float(max(0.0, self.objective_pathology_nonfinite_threshold))
        self.objective_gradient_guard_interval = int(max(1, self.objective_gradient_guard_interval))
        self.objective_gradient_guard_start_step = int(max(0, self.objective_gradient_guard_start_step))
        self.objective_gradient_guard_max_terms = int(max(1, self.objective_gradient_guard_max_terms))
        self.objective_gradient_guard_param_tensors = int(max(1, self.objective_gradient_guard_param_tensors))
        self.objective_conflict_cosine_threshold = float(min(max(self.objective_conflict_cosine_threshold, -1.0), 1.0))
        self.objective_conflict_quarantine_patience = int(max(1, self.objective_conflict_quarantine_patience))
        self.conv_active_scales = int(max(1, min(self.conv_active_scales, max(1, self.n_conv_scales))))
        self.adaptive_depth_skip_threshold = float(min(max(self.adaptive_depth_skip_threshold, 0.0), 1.0))
        self.cooperative_steps = bool(getattr(self, "cooperative_steps", True))
        auto_summary = max(8, min(int(self.d_model), min(128, max(16, int(self.d_model) // 4))))
        requested_summary = int(getattr(self, "cooperative_step_summary_dim", 0) or auto_summary)
        self.cooperative_step_summary_dim = int(max(8, min(int(self.d_model), requested_summary)))
        self.cooperative_step_context_strength = float(min(max(_safe_float(getattr(self, "cooperative_step_context_strength", 0.12), 0.12), 0.0), 1.0))
        self.cooperative_step_final_strength = float(min(max(_safe_float(getattr(self, "cooperative_step_final_strength", 0.08), 0.08), 0.0), 1.0))
        self.cooperative_step_gate_bias_init = float(min(max(_safe_float(getattr(self, "cooperative_step_gate_bias_init", 2.0), 2.0), -4.0), 6.0))
        self.cooperative_step_budget = float(min(max(_safe_float(getattr(self, "cooperative_step_budget", 0.82), 0.82), 0.05), 1.0))
        self.cooperative_step_redundancy_target = float(min(max(_safe_float(getattr(self, "cooperative_step_redundancy_target", 0.15), 0.15), 0.0), 0.95))
        self.slot_attention_mode = str(getattr(self, "slot_attention_mode", "pooled")).lower()
        if self.slot_attention_mode not in ("pooled", "attention"):
            self.slot_attention_mode = "pooled"
        self.pathway_conditioning_max_vectors = int(max(1, self.pathway_conditioning_max_vectors))
        self.pathway_conditioning_top_k = int(max(1, min(self.pathway_conditioning_top_k, self.pathway_conditioning_max_vectors)))
        self.pathway_memory_slots = int(max(1, self.pathway_memory_slots))
        self.pathway_memory_top_k = int(max(1, min(self.pathway_memory_top_k, self.pathway_memory_slots)))
        self.planner_horizons = int(max(0, self.planner_horizons))
        self.planner_transition_depth = int(max(1, self.planner_transition_depth))
        self.planner_logit_blend = float(min(max(self.planner_logit_blend, 0.0), 1.0))
        self.planner_block_space_mix = float(min(max(self.planner_block_space_mix, 0.0), 1.0))
        self.planner_block_space_iters = int(max(1, self.planner_block_space_iters))
        self.dvsd_planner_compound_mix = float(min(max(self.dvsd_planner_compound_mix, 0.0), 1.0))
        self.dvsd_planner_compound_step_size = float(min(max(self.dvsd_planner_compound_step_size, 0.0), 2.0))
        self.dvsd_planner_compound_token_scale = float(min(max(self.dvsd_planner_compound_token_scale, 0.0), 2.0))
        self.dvsd_planner_compound_descent_scale = float(min(max(self.dvsd_planner_compound_descent_scale, 0.0), 2.0))
        self.dvsd_planner_compound_top_k = int(max(1, self.dvsd_planner_compound_top_k))
        if self.planner_horizons <= 0:
            self.dvsd_planner_compound_enabled = False
            self.planner_block_space_enabled = False
        self.n_predict_heads = int(max(1, self.n_predict_heads))
        self.nonseq_min_heads = int(max(1, self.nonseq_min_heads))
        self.nonseq_max_heads = int(max(self.nonseq_min_heads, self.nonseq_max_heads))
        self.nonseq_accept_top_k = int(max(1, self.nonseq_accept_top_k))
        self.nonseq_accept_min_prob = float(max(0.0, self.nonseq_accept_min_prob))
        self.nonseq_accept_max_logprob_gap = float(max(0.0, self.nonseq_accept_max_logprob_gap))
        self.nonseq_tree_candidates = int(max(1, self.nonseq_tree_candidates))
        self.nonseq_branch_top_k = int(max(1, self.nonseq_branch_top_k))
        self.nonseq_entropy_easy = float(min(max(self.nonseq_entropy_easy, 0.0), 1.0))
        self.nonseq_entropy_hard = float(min(max(self.nonseq_entropy_hard, self.nonseq_entropy_easy + 1e-6), 1.0))
        self.nonseq_min_draft_prob = float(max(0.0, self.nonseq_min_draft_prob))
        self.nonseq_max_head_entropy = float(min(max(self.nonseq_max_head_entropy, 0.0), 1.0))
        self.nonseq_decode_style = str(self.nonseq_decode_style or "slot_refine")
        if self.nonseq_decode_style not in {"slot_refine", "slots", "virtual_slots"}:
            self.nonseq_decode_style = "slot_refine"
        self.nonseq_refine_steps = int(max(1, self.nonseq_refine_steps))
        self.nonseq_refine_causal_blend = float(min(max(self.nonseq_refine_causal_blend, 0.0), 1.0))
        self.nonseq_refine_confidence_floor = float(min(max(self.nonseq_refine_confidence_floor, 0.0), 1.0))
        self.nonseq_refine_temperature_decay = float(min(max(self.nonseq_refine_temperature_decay, 0.05), 1.0))
        self.nonseq_refine_lock_schedule = str(self.nonseq_refine_lock_schedule or "cosine")
        if self.nonseq_refine_lock_schedule not in {"cosine", "linear", "quadratic"}:
            self.nonseq_refine_lock_schedule = "cosine"
        self.dvsd_slot_order = str(getattr(self, "dvsd_slot_order", "confidence") or "confidence")
        if self.dvsd_slot_order not in {"confidence", "left_to_right"}:
            self.dvsd_slot_order = "confidence"
        self.dvsd_freeze_sampled_slots = bool(getattr(self, "dvsd_freeze_sampled_slots", True))
        self.dvsd_router_inference_mix = float(min(max(self.dvsd_router_inference_mix, 0.0), 1.0))
        self.dvsd_router_min_confidence = float(min(max(self.dvsd_router_min_confidence, 0.0), 1.0))
        self.dvsd_router_loss_threshold = float(max(0.05, self.dvsd_router_loss_threshold))
        self.dvsd_router_hard_loss_threshold = float(max(self.dvsd_router_loss_threshold, self.dvsd_router_hard_loss_threshold))
        self.dvsd_router_entropy_weight = float(max(0.0, self.dvsd_router_entropy_weight))

    def image_tokenizer_config(self) -> ImageTokenizerConfig:
        return ImageTokenizerConfig(
            image_codebook_size=self.image_codebook_size,
            min_grid=self.image_min_grid,
            max_grid=self.image_max_grid,
            default_grid=self.image_grid,
            max_image_tokens=self.image_max_tokens,
            token_offset=self.image_token_offset,
            dynamic_grid=self.dynamic_image_grid,
        )


# -----------------------------
# Utility layers/losses
# -----------------------------


OBJECTIVE_LOSS_GROUPS: Dict[str, str] = {**AUX_COMPONENT_GROUPS, **FUSED_OBJECTIVE_GROUPS}



class AuxiliaryObjectiveBalancer(nn.Module):
    """Conservative controller for NEED's many auxiliary objectives.

    This controller now handles four levels of protection:
    - scalar balancing: EMA normalization, warmup, total caps, and group caps;
    - curriculum staging: auxiliary families are ramped in at different steps;
    - automatic quarantine: repeatedly clipped, non-finite, or conflicting terms
      are multiplicatively suppressed and then slowly recover;
    - diagnostics/export: differentiable per-term contributions are exposed for
      train.py's occasional CE-vs-aux gradient conflict probes.
    """
    def __init__(self, cfg: NeedConfig, loss_names: Sequence[str]):
        super().__init__()
        self.cfg = cfg
        self.loss_names = sorted(set(str(n) for n in loss_names))
        self.loss_index = {name: i for i, name in enumerate(self.loss_names)}
        self.group_names = ["prediction", "latent", "risk", "vision", "regularizer", "other"]
        self.group_index = {name: i for i, name in enumerate(self.group_names)}
        n = len(self.loss_names)
        self.register_buffer("objective_step", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("loss_ema", torch.ones(n, dtype=torch.float32), persistent=True)
        self.register_buffer("loss_seen", torch.zeros(n, dtype=torch.float32), persistent=True)
        self.register_buffer("clip_ema", torch.zeros(n, dtype=torch.float32), persistent=True)
        self.register_buffer("nonfinite_ema", torch.zeros(n, dtype=torch.float32), persistent=True)
        self.register_buffer("conflict_ema", torch.zeros(n, dtype=torch.float32), persistent=True)
        self.register_buffer("pathology_count", torch.zeros(n, dtype=torch.float32), persistent=True)
        self.register_buffer("conflict_count", torch.zeros(n, dtype=torch.float32), persistent=True)
        self.register_buffer("quarantine_scale", torch.ones(n, dtype=torch.float32), persistent=True)
        self.register_buffer("quarantine_timer", torch.zeros(n, dtype=torch.float32), persistent=True)

    def begin_batch(self) -> None:
        if self.training:
            self.objective_step.add_(1)
            if bool(getattr(self.cfg, "objective_quarantine_enabled", True)):
                self._recover_quarantine_scales()

    def _recover_quarantine_scales(self) -> None:
        if self.quarantine_scale.numel() == 0:
            return
        recovery_steps = float(max(1, int(getattr(self.cfg, "objective_quarantine_recovery_steps", 2000))))
        with torch.no_grad():
            below = self.quarantine_scale < 0.999
            if bool(below.any()):
                self.quarantine_timer[below] += 1.0
                # Linear-in-time exponential-like recovery that never jumps a term
                # from quarantined to full strength in one optimizer region.
                inc = (1.0 - self.quarantine_scale[below]) / recovery_steps
                self.quarantine_scale[below] = (self.quarantine_scale[below] + inc).clamp_max(1.0)

    def _warmup_scale(self, ref: torch.Tensor) -> torch.Tensor:
        warmup = int(getattr(self.cfg, "objective_aux_warmup_steps", 0) or 0)
        if warmup <= 0:
            return ref.new_tensor(1.0)
        step = self.objective_step.to(device=ref.device, dtype=torch.float32)
        frac = (step / float(max(1, warmup))).clamp(0.0, 1.0)
        min_scale = float(getattr(self.cfg, "objective_aux_min_scale", 0.0))
        smooth = frac * frac * (3.0 - 2.0 * frac)
        return ref.new_tensor(min_scale) + (1.0 - min_scale) * smooth

    def _family_curriculum_scale(self, group: str, ref: torch.Tensor) -> torch.Tensor:
        if not bool(getattr(self.cfg, "objective_curriculum_enabled", True)):
            return ref.new_tensor(1.0)
        starts = {
            "prediction": int(getattr(self.cfg, "objective_prediction_start_step", 100)),
            "latent": int(getattr(self.cfg, "objective_latent_start_step", 400)),
            "risk": int(getattr(self.cfg, "objective_risk_start_step", 700)),
            "vision": int(getattr(self.cfg, "objective_vision_start_step", 900)),
            "regularizer": int(getattr(self.cfg, "objective_regularizer_start_step", 0)),
            "other": int(getattr(self.cfg, "objective_regularizer_start_step", 0)),
        }
        start = starts.get(group, starts["other"])
        ramp = max(1, int(getattr(self.cfg, "objective_family_ramp_steps", 1200)))
        step = self.objective_step.to(device=ref.device, dtype=torch.float32)
        frac = ((step - float(start)) / float(ramp)).clamp(0.0, 1.0)
        smooth = frac * frac * (3.0 - 2.0 * frac)
        return smooth.to(device=ref.device, dtype=ref.dtype)

    def _ema_scale(self, name: str, raw_detached: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        floor = float(getattr(self.cfg, "objective_loss_ema_floor", 1e-3))
        if name not in self.loss_index or not bool(getattr(self.cfg, "objective_normalize_aux", True)):
            return ref.new_tensor(1.0)
        idx = self.loss_index[name]
        obs = raw_detached.float().abs().clamp_min(floor).clamp_max(float(getattr(self.cfg, "objective_term_abs_cap", 25.0)))
        if self.training:
            beta = float(getattr(self.cfg, "objective_balance_ema_beta", 0.98))
            obs_cpu = obs.detach().cpu() if obs.device.type != self.loss_ema.device.type else obs.detach()
            with torch.no_grad():
                seen = self.loss_seen[idx]
                if float(seen.detach().cpu()) <= 0.0:
                    self.loss_ema[idx].copy_(obs_cpu)
                else:
                    self.loss_ema[idx].copy_((beta * self.loss_ema[idx] + (1.0 - beta) * obs_cpu).clamp_min(floor))
                self.loss_seen[idx].add_(1.0)
        return self.loss_ema[idx].to(device=ref.device, dtype=ref.dtype).clamp_min(floor)

    @staticmethod
    def entropy_band_loss(entropy: torch.Tensor, choices: int, low_frac: float = 0.25, high_frac: float = 0.88) -> torch.Tensor:
        """Penalize both collapsed and fully uniform discrete routing."""
        max_ent = math.log(max(2, int(choices)))
        low = entropy.new_tensor(max_ent * low_frac)
        high = entropy.new_tensor(max_ent * high_frac)
        return F.relu(low - entropy.float()).pow(2) + 0.25 * F.relu(entropy.float() - high).pow(2)

    def _idx_for_name(self, name: str) -> Optional[int]:
        idx = self.loss_index.get(str(name))
        return None if idx is None else int(idx)

    def _record_pathology(self, name: str, clipped: torch.Tensor, nonfinite: torch.Tensor) -> torch.Tensor:
        idx = self._idx_for_name(name)
        if idx is None:
            return clipped.new_tensor(1.0)
        if not bool(getattr(self.cfg, "objective_quarantine_enabled", True)):
            return self.quarantine_scale[idx].to(device=clipped.device, dtype=clipped.dtype)
        beta = float(getattr(self.cfg, "objective_balance_ema_beta", 0.98))
        clip_obs = clipped.detach().float().to(device=self.clip_ema.device)
        nf_obs = nonfinite.detach().float().to(device=self.nonfinite_ema.device)
        with torch.no_grad():
            self.clip_ema[idx].copy_(beta * self.clip_ema[idx] + (1.0 - beta) * clip_obs)
            self.nonfinite_ema[idx].copy_(beta * self.nonfinite_ema[idx] + (1.0 - beta) * nf_obs)
            clip_bad = float(self.clip_ema[idx].cpu()) > float(getattr(self.cfg, "objective_pathology_clip_threshold", 0.60))
            nf_bad = float(self.nonfinite_ema[idx].cpu()) > float(getattr(self.cfg, "objective_pathology_nonfinite_threshold", 0.0))
            if clip_bad or nf_bad:
                self.pathology_count[idx].add_(1.0)
            else:
                self.pathology_count[idx].mul_(0.97)
            if float(self.pathology_count[idx].cpu()) >= float(getattr(self.cfg, "objective_quarantine_patience", 8)):
                decay = float(getattr(self.cfg, "objective_quarantine_decay", 0.50))
                min_scale = float(getattr(self.cfg, "objective_quarantine_min_scale", 0.05))
                self.quarantine_scale[idx].copy_((self.quarantine_scale[idx] * decay).clamp_min(min_scale))
                self.quarantine_timer[idx].zero_()
                self.pathology_count[idx].zero_()
        return self.quarantine_scale[idx].to(device=clipped.device, dtype=clipped.dtype)

    def record_gradient_conflicts(self, conflicts: Dict[str, float]) -> Dict[str, float]:
        """Called by train.py after occasional CE-vs-aux gradient probes.

        The input maps objective names to cosine similarity with CE gradients.
        Negative persistent cosine triggers quarantine, while neutral/positive
        cosine lets terms recover over time.
        """
        applied: Dict[str, float] = {}
        if not conflicts or not bool(getattr(self.cfg, "objective_quarantine_enabled", True)):
            return applied
        threshold = float(getattr(self.cfg, "objective_conflict_cosine_threshold", -0.10))
        beta = float(getattr(self.cfg, "objective_balance_ema_beta", 0.98))
        patience = float(getattr(self.cfg, "objective_conflict_quarantine_patience", 3))
        decay = float(getattr(self.cfg, "objective_quarantine_decay", 0.50))
        min_scale = float(getattr(self.cfg, "objective_quarantine_min_scale", 0.05))
        with torch.no_grad():
            for name, cos in conflicts.items():
                idx = self._idx_for_name(name)
                if idx is None or not math.isfinite(float(cos)):
                    continue
                bad = 1.0 if float(cos) < threshold else 0.0
                self.conflict_ema[idx].copy_(beta * self.conflict_ema[idx] + (1.0 - beta) * torch.tensor(bad, device=self.conflict_ema.device))
                if bad:
                    self.conflict_count[idx].add_(1.0)
                else:
                    self.conflict_count[idx].mul_(0.90)
                if float(self.conflict_count[idx].cpu()) >= patience:
                    self.quarantine_scale[idx].copy_((self.quarantine_scale[idx] * decay).clamp_min(min_scale))
                    self.quarantine_timer[idx].zero_()
                    self.conflict_count[idx].zero_()
                    applied[name] = float(self.quarantine_scale[idx].cpu())
        return applied

    def add(
        self,
        base: torch.Tensor,
        ce: torch.Tensor,
        aux: Dict[str, torch.Tensor],
        lam: float,
        raw: torch.Tensor,
        name: str,
        group_used: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        aux[name] = raw.detach() if torch.is_tensor(raw) else raw
        if lam == 0 or not torch.is_tensor(raw):
            return base
        raw_scalar = raw.float().mean() if raw.ndim > 0 else raw.float()
        finite = torch.isfinite(raw_scalar)
        nonfinite = (~finite).float()
        sanitized = torch.nan_to_num(
            raw_scalar,
            nan=0.0,
            posinf=float(getattr(self.cfg, "objective_term_abs_cap", 25.0)),
            neginf=-float(getattr(self.cfg, "objective_term_abs_cap", 25.0)),
        )
        raw_abs_det = sanitized.detach().abs()
        aux_terms = aux.setdefault("_objective_terms", {})
        aux_lambdas = aux.setdefault("_objective_lambdas", {})
        if bool(getattr(self.cfg, "objective_soft_budget", True)):
            scale = self._ema_scale(name, raw_abs_det, sanitized)
            term = sanitized / scale
            term = term.clamp(
                -float(getattr(self.cfg, "objective_term_abs_cap", 25.0)),
                float(getattr(self.cfg, "objective_term_abs_cap", 25.0)),
            )
            group = OBJECTIVE_LOSS_GROUPS.get(name, "other")
            warm_scale = self._warmup_scale(sanitized)
            family_scale = self._family_curriculum_scale(group, sanitized)
            quarantine_scale = self.quarantine_scale[self.loss_index[name]].to(device=sanitized.device, dtype=sanitized.dtype) if name in self.loss_index else sanitized.new_tensor(1.0)
            lam_eff = float(lam) * warm_scale * family_scale * quarantine_scale
            ce_ref = ce.detach().float().abs().clamp_min(float(getattr(self.cfg, "objective_softcap_min", 0.02)))
            total_cap = float(getattr(self.cfg, "objective_aux_ratio_cap", 0.45)) * ce_ref
            group_cap = float(getattr(self.cfg, "objective_group_ratio_cap", 0.18)) * ce_ref
            total_used = group_used.get("__total__", sanitized.new_tensor(0.0))
            this_group_used = group_used.get(group, sanitized.new_tensor(0.0))
            remaining = torch.minimum((total_cap - total_used).clamp_min(0.0), (group_cap - this_group_used).clamp_min(0.0))
            denom = abs(float(lam)) * warm_scale.detach().clamp_min(1e-8) * family_scale.detach().clamp_min(1e-8) * quarantine_scale.detach().clamp_min(1e-8)
            term_cap = (remaining / denom).clamp_min(0.0)
            clipped = (term.abs().detach() > term_cap.detach()).float()
            term = torch.where(term_cap > 0, term_cap * torch.tanh(term / term_cap.clamp_min(1e-8)), term * 0.0)
            path_q_scale = self._record_pathology(name, clipped, nonfinite)
            lam_eff = lam_eff * (path_q_scale / quarantine_scale.detach().clamp_min(1e-8))
            contrib = lam_eff * term
            abs_contrib = contrib.detach().abs()
            group_used["__total__"] = total_used + abs_contrib
            group_used[group] = this_group_used + abs_contrib
            aux[name + "_objective_weight"] = torch.as_tensor(float(lam), device=sanitized.device, dtype=sanitized.dtype) * warm_scale.detach() * family_scale.detach() * path_q_scale.detach()
            aux[name + "_objective_norm"] = scale.detach()
            aux[name + "_objective_abs_contrib"] = abs_contrib
            aux[name + "_objective_clipped"] = clipped.detach()
            aux[name + "_objective_nonfinite"] = nonfinite.detach()
            aux[name + "_objective_family_scale"] = family_scale.detach()
            aux[name + "_objective_quarantine_scale"] = path_q_scale.detach()
            aux_terms[name] = contrib
            aux_lambdas[name] = torch.as_tensor(float(lam), device=sanitized.device, dtype=sanitized.dtype)
            return base + contrib.to(base.dtype)
        contrib = float(lam) * sanitized.to(base.dtype)
        aux_terms[name] = contrib
        aux_lambdas[name] = torch.as_tensor(float(lam), device=sanitized.device, dtype=sanitized.dtype)
        return base + contrib

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, kernel_backend: str = "auto"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.kernel_backend = kernel_backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if need_kernels is not None:
            return need_kernels.rms_norm(x, self.weight, self.eps, self.kernel_backend)
        y = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps).to(x.dtype)
        return y * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float, kernel_backend: str = "auto"):
        super().__init__()
        self.w12 = nn.Linear(dim, hidden * 2, bias=False)
        self.out = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.kernel_backend = kernel_backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ab = self.w12(x)
        a, b = ab.chunk(2, dim=-1)
        if need_kernels is not None:
            y = need_kernels.swiglu(a, b, self.kernel_backend)
        else:
            y = F.silu(a) * b
        return self.out(self.drop(y))


def affine_recurrence_scan_torch(decay: torch.Tensor, update: torch.Tensor, dim: int, chunk_size: int = 64) -> torch.Tensor:
    """Divide-free PyTorch scan for ``y_t = decay_t * y_{t-1} + update_t``.

    This is the core fallback used when optional kernels are unavailable. It uses
    chunked affine-transform prefix composition instead of the old
    product/division closed form, so aggressive decays do not underflow to an
    all-zero answer just because an intermediate cumulative product vanished.
    """
    dim = dim if dim >= 0 else update.ndim + dim
    if dim < 0 or dim >= update.ndim:
        raise IndexError(f"scan dim {dim} out of range for rank {update.ndim}")
    if update.size(dim) == 0:
        return update
    if decay.shape != update.shape:
        decay = torch.broadcast_to(decay, update.shape)
    x = torch.movedim(update, dim, 1)
    a = torch.movedim(decay, dim, 1).to(torch.float32)
    xf = x.to(torch.float32)
    t = xf.size(1)
    chunk = max(1, min(int(chunk_size), t))
    state = torch.zeros_like(xf[:, 0])
    outs: List[torch.Tensor] = []
    for start in range(0, t, chunk):
        end = min(t, start + chunk)
        a_pref = a[:, start:end]
        b_pref = xf[:, start:end]
        shift = 1
        width = end - start
        while shift < width:
            a_prev = a_pref[:, :-shift]
            b_prev = b_pref[:, :-shift]
            a_cur = a_pref[:, shift:]
            b_cur = b_pref[:, shift:]
            a_combined = a_cur * a_prev
            b_combined = b_cur + a_cur * b_prev
            a_pref = torch.cat([a_pref[:, :shift], a_combined], dim=1)
            b_pref = torch.cat([b_pref[:, :shift], b_combined], dim=1)
            shift *= 2
        yc = b_pref + a_pref * state.unsqueeze(1)
        state = yc[:, -1]
        outs.append(yc)
    y = torch.cat(outs, dim=1).to(dtype=update.dtype)
    return torch.movedim(y, 1, dim)


class MultiScaleCausalConv(nn.Module):
    """Causal multi-scale depthwise convolution.

    Causal multi-scale depthwise convolution with per-token route probabilities.
    """
    def __init__(self, dim: int, kernel: int, n_scales: int, dropout: float, active_scales: int = 1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.kernel_sizes = []
        self.n_scales = int(max(1, n_scales))
        self.active_scales = int(max(1, min(active_scales, self.n_scales)))
        for i in range(self.n_scales):
            k = kernel + 2 * i
            self.kernel_sizes.append(k)
            self.convs.append(nn.Conv1d(dim, dim, kernel_size=k, groups=dim, bias=True))
        self.mix = nn.Linear(dim * self.n_scales, dim, bias=False)
        self.gate = nn.Linear(dim, self.n_scales, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        b, t, d = x.shape
        xt = x.transpose(1, 2)
        probs = F.softmax(self.gate(x).float(), dim=-1).to(x.dtype)
        active = int(max(1, min(self.active_scales, self.n_scales)))
        active_probs = probs[..., :active]
        active_probs = active_probs / active_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        cat = x.new_zeros(b, t, d * self.n_scales)
        # Use a fixed causal active subset.
        for i in range(active):
            conv = self.convs[i]
            padded = F.pad(xt, (self.kernel_sizes[i] - 1, 0))
            yi = conv(padded)[..., :t].transpose(1, 2)
            yi = yi * active_probs[..., i:i + 1].to(yi.dtype)
            cat[..., i * d:(i + 1) * d] = yi
        y = self.mix(cat)
        ent = -(probs.float() * probs.float().clamp_min(1e-8).log()).sum(dim=-1).mean()
        active_metric = x.new_tensor(float(active))
        return self.drop(y), {"conv_scale_entropy": ent, "conv_active_scales": active_metric}

    def init_stream_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
        buffers = [torch.zeros(batch_size, self.convs[i].in_channels, k, device=device, dtype=dtype) for i, k in enumerate(self.kernel_sizes)]
        return {"buffers": buffers}

    def stream_step(self, x: torch.Tensor, state: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """One-token causal convolution update using rolling buffers."""
        if x.ndim != 3 or x.size(1) != 1:
            raise ValueError("stream_step expects x with shape [B,1,D]")
        b, _, d = x.shape
        xt = x[:, 0].to(dtype=x.dtype)
        if "buffers" not in state or len(state["buffers"]) != self.n_scales:
            state.update(self.init_stream_state(b, x.device, x.dtype))
        probs = F.softmax(self.gate(x).float(), dim=-1).to(x.dtype)
        active = int(max(1, min(self.active_scales, self.n_scales)))
        active_probs = probs[..., :active]
        active_probs = active_probs / active_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        cat = x.new_zeros(b, 1, d * self.n_scales)
        # Keep all buffers current so raising active_scales later does not lose
        # history, but only compute the fixed active subset.
        for i, conv in enumerate(self.convs):
            k = self.kernel_sizes[i]
            buf = state["buffers"][i]
            if buf.size(0) != b or buf.device != x.device or buf.dtype != x.dtype:
                buf = torch.zeros(b, d, k, device=x.device, dtype=x.dtype)
            if k > 1:
                buf = torch.cat([buf[:, :, 1:], xt.unsqueeze(-1)], dim=-1)
            else:
                buf = xt.unsqueeze(-1)
            state["buffers"][i] = buf.detach() if not torch.is_grad_enabled() else buf
            if i >= active:
                continue
            weight = conv.weight.squeeze(1).to(dtype=x.dtype, device=x.device)
            yi = (buf * weight.unsqueeze(0)).sum(dim=-1)
            if conv.bias is not None:
                yi = yi + conv.bias.to(dtype=x.dtype, device=x.device).view(1, -1)
            yi = yi.unsqueeze(1) * active_probs[..., i:i + 1].to(x.dtype)
            cat[..., i * d:(i + 1) * d] = yi
        y = self.mix(cat)
        ent = -(probs.float() * probs.float().clamp_min(1e-8).log()).sum(dim=-1).mean()
        return self.drop(y), {"conv_scale_entropy": ent, "conv_active_scales": x.new_tensor(float(active))}

class SelectiveRetention(nn.Module):
    """Linear-time causal recurrent retention. Parallel training fallback uses scan loop."""
    def __init__(self, cfg: NeedConfig):
        super().__init__()
        self.cfg = cfg
        self.heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.in_proj = nn.Linear(cfg.d_model, cfg.d_model * 3, bias=False)
        self.decay_proj = nn.Linear(cfg.d_model, cfg.n_heads, bias=True)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        decays = torch.linspace(cfg.retention_min_decay, cfg.retention_max_decay, cfg.n_heads)
        self.register_buffer("base_decay", decays.view(1, cfg.n_heads, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        dyn = torch.sigmoid(self.decay_proj(x)).transpose(1, 2).unsqueeze(-1)
        decay = (self.base_decay + self.cfg.retention_dynamic_scale * (dyn - 0.5) * 0.05).clamp(0.05, 0.999)
        kv = k * v
        if need_kernels is not None and bool(getattr(self.cfg, "parallel_scan", True)):
            state_seq = need_kernels.affine_recurrence_scan(decay, kv, dim=2, backend=self.cfg.kernel_backend)
            y = (q * state_seq).transpose(1, 2).reshape(b, t, d)
        else:
            state_seq = affine_recurrence_scan_torch(decay, kv, dim=2, chunk_size=64)
            y = (q * state_seq).transpose(1, 2).reshape(b, t, d)
        return self.out(y)

    def init_stream_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
        return {"state": torch.zeros(batch_size, self.heads, self.head_dim, device=device, dtype=dtype)}

    def stream_step(self, x: torch.Tensor, state: Dict[str, Any]) -> torch.Tensor:
        """One-token recurrent retention update."""
        if x.ndim != 3 or x.size(1) != 1:
            raise ValueError("stream_step expects x with shape [B,1,D]")
        b = x.size(0)
        if "state" not in state or state["state"].size(0) != b or state["state"].device != x.device:
            state.update(self.init_stream_state(b, x.device, x.dtype))
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(b, 1, self.heads, self.head_dim).transpose(1, 2).squeeze(2)
        k = k.view(b, 1, self.heads, self.head_dim).transpose(1, 2).squeeze(2)
        v = v.view(b, 1, self.heads, self.head_dim).transpose(1, 2).squeeze(2)
        dyn = torch.sigmoid(self.decay_proj(x)).transpose(1, 2).squeeze(-1).unsqueeze(-1)
        base = self.base_decay.squeeze(0).squeeze(-1).to(device=x.device, dtype=x.dtype)
        decay = (base.unsqueeze(0) + self.cfg.retention_dynamic_scale * (dyn - 0.5) * 0.05).clamp(0.05, 0.999)
        state["state"] = decay * state["state"].to(x.dtype) + k * v
        y = (q * state["state"]).reshape(b, 1, self.heads * self.head_dim)
        return self.out(y)


class StructuredDualRetention(nn.Module):
    """SSD/Mamba-style recurrent mixing block.

    This is not self-attention.  It is a diagonal state-space scan with
    input-selective decay/input/output gates and a local depthwise convolution.
    The implementation is a clear PyTorch reference path for later fused Triton
    scan kernels: same equations, one recurrent state per channel.
    """
    def __init__(self, cfg: NeedConfig):
        super().__init__()
        self.cfg = cfg
        self.d = cfg.d_model
        k = max(1, int(cfg.ssd_conv_kernel))
        if k % 2 == 0:
            k += 1
        self.in_proj = nn.Linear(cfg.d_model, cfg.d_model * 5, bias=False)
        self.dwconv = nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=k, padding=k - 1, groups=cfg.d_model)
        self.out_norm = RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        # D skip term, initialized small so the recurrent path learns smoothly.
        self.D = nn.Parameter(torch.ones(cfg.d_model) * 0.10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        u, b_gate, c_gate, dt_raw, out_gate = self.in_proj(x).chunk(5, dim=-1)
        # Causal local prefilter.  Conv1d padding creates extra right context; trim it.
        u_conv = self.dwconv(u.transpose(1, 2))[..., :t].transpose(1, 2)
        u = F.silu(u_conv)
        if need_kernels is not None and self.cfg.fused_ssd_scan and not torch.is_grad_enabled():
            y = need_kernels.ssd_scan(
                u.contiguous(), b_gate.contiguous(), c_gate.contiguous(), dt_raw.contiguous(), out_gate.contiguous(),
                self.D.contiguous(), self.cfg.ssd_dt_min, self.cfg.ssd_dt_max, self.cfg.kernel_backend,
            )
        else:
            dt = torch.sigmoid(dt_raw.float())
            dt = self.cfg.ssd_dt_min + (self.cfg.ssd_dt_max - self.cfg.ssd_dt_min) * dt
            decay = torch.exp(-dt).to(x.dtype)
            b_gate = torch.sigmoid(b_gate).to(x.dtype)
            c_gate = torch.sigmoid(c_gate).to(x.dtype)
            out_gate = torch.sigmoid(out_gate).to(x.dtype)
            update = (1.0 - decay) * (b_gate * u)
            if need_kernels is not None and bool(getattr(self.cfg, "parallel_scan", True)):
                state_seq = need_kernels.affine_recurrence_scan(decay, update, dim=1, backend=self.cfg.kernel_backend)
                y = (c_gate * state_seq + self.D.to(x.dtype).view(1, 1, -1) * u) * out_gate
            else:
                state_seq = affine_recurrence_scan_torch(decay, update, dim=1, chunk_size=64)
                y = (c_gate * state_seq + self.D.to(x.dtype).view(1, 1, -1) * u) * out_gate
        y = self.out(self.out_norm(y))
        return y

    def init_stream_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
        k = int(self.dwconv.kernel_size[0])
        return {
            "conv_buffer": torch.zeros(batch_size, self.d, k, device=device, dtype=dtype),
            "scan_state": torch.zeros(batch_size, self.d, device=device, dtype=dtype),
        }

    def stream_step(self, x: torch.Tensor, state: Dict[str, Any]) -> torch.Tensor:
        """One-token SSD/Mamba-style update with a rolling depthwise-conv buffer."""
        if x.ndim != 3 or x.size(1) != 1:
            raise ValueError("stream_step expects x with shape [B,1,D]")
        b, _, d = x.shape
        if ("conv_buffer" not in state or "scan_state" not in state or state["scan_state"].size(0) != b or state["scan_state"].device != x.device):
            state.update(self.init_stream_state(b, x.device, x.dtype))
        u, b_gate, c_gate, dt_raw, out_gate = self.in_proj(x).chunk(5, dim=-1)
        u0 = u[:, 0]
        buf = state["conv_buffer"]
        if buf.size(0) != b or buf.size(1) != d or buf.device != x.device or buf.dtype != x.dtype:
            buf = torch.zeros(b, d, int(self.dwconv.kernel_size[0]), device=x.device, dtype=x.dtype)
        if buf.size(-1) > 1:
            buf = torch.cat([buf[:, :, 1:], u0.unsqueeze(-1)], dim=-1)
        else:
            buf = u0.unsqueeze(-1)
        state["conv_buffer"] = buf
        weight = self.dwconv.weight.squeeze(1).to(dtype=x.dtype, device=x.device)
        u_conv = (buf * weight.unsqueeze(0)).sum(dim=-1)
        if self.dwconv.bias is not None:
            u_conv = u_conv + self.dwconv.bias.to(dtype=x.dtype, device=x.device).view(1, -1)
        u_act = F.silu(u_conv).unsqueeze(1)
        dt = torch.sigmoid(dt_raw.float())
        dt = self.cfg.ssd_dt_min + (self.cfg.ssd_dt_max - self.cfg.ssd_dt_min) * dt
        decay = torch.exp(-dt).to(x.dtype)
        bg = torch.sigmoid(b_gate).to(x.dtype)
        cg = torch.sigmoid(c_gate).to(x.dtype)
        og = torch.sigmoid(out_gate).to(x.dtype)
        update = (1.0 - decay) * (bg * u_act)
        prev = state["scan_state"].to(x.dtype).unsqueeze(1)
        scan = decay * prev + update
        state["scan_state"] = scan[:, 0]
        y = (cg * scan + self.D.to(x.dtype).view(1, 1, -1) * u_act) * og
        return self.out(self.out_norm(y))


class AdaptiveDepthGate(nn.Module):
    """Compute controller for expensive NEED subpaths.

    The gate is not just a residual weight: NEEDBlock can apply a thresholded
    tensor mask to low-gate paths. The mask is static-shape by default, so it
    does not compact tokens or branch on GPU values inside the forward pass.
    """
    def __init__(self, cfg: NeedConfig):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend),
            nn.Linear(cfg.d_model, max(16, cfg.d_model // 2)),
            nn.SiLU(),
            nn.Linear(max(16, cfg.d_model // 2), 3),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.cfg.adaptive_depth:
            gates = x.new_ones(x.size(0), x.size(1), 3)
        else:
            temp = max(1e-4, float(self.cfg.depth_gate_temperature))
            gates = torch.sigmoid(self.net(x.float()) / temp).to(x.dtype)
            gates = gates * (1.0 - self.cfg.min_compute_gate) + self.cfg.min_compute_gate
        mean_gate = gates.float().mean()
        budget = x.new_tensor(float(self.cfg.compute_budget))
        # Penalize compute that overshoots the requested budget more than undershoot;
        # this keeps equal-FLOP experiments honest while allowing hard tokens to work.
        penalty = F.relu(mean_gate - budget).pow(2) + 0.25 * F.relu(budget * 0.50 - mean_gate).pow(2)
        entropy = -(gates.float() * gates.float().clamp_min(1e-8).log() + (1-gates.float()) * (1-gates.float()).clamp_min(1e-8).log()).mean()
        return gates, {"compute_fraction": mean_gate.detach(), "compute_budget": penalty, "compute_gate_entropy": entropy}


def image_span_token_mask(input_ids: torch.Tensor, cfg: NeedConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return image-payload tokens and per-row image segment ids without Python token loops."""
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B,T]")
    b, t = input_ids.shape
    device = input_ids.device
    bos = input_ids == int(cfg.img_bos_id)
    end = (input_ids == int(cfg.img_eos_id)) | (input_ids == int(cfg.pad_id))
    image_payload = (input_ids == int(cfg.img_mask_id)) | (
        (input_ids >= int(cfg.image_token_offset))
        & (input_ids < int(cfg.image_token_offset) + int(cfg.image_codebook_size))
    )
    pos = torch.arange(t, device=device, dtype=torch.long).view(1, t).expand(b, t)
    neg = torch.full_like(pos, -1)
    last_bos = torch.where(bos, pos, neg).cummax(dim=1).values
    last_end = torch.where(end, pos, neg).cummax(dim=1).values
    segment_ids = bos.to(torch.long).cumsum(dim=1)
    inside_image = last_bos > last_end
    return inside_image & image_payload & (segment_ids > 0), segment_ids


class Image2DSelectiveScan(nn.Module):
    """Explicit 2D selective scan over image-token grids.

    NEED already adds row/column coordinate embeddings.  This module goes further:
    image spans are reshaped into a grid and scanned left/right/up/down with a
    shared learned value/gate projection.  It gives visual tokens native 2D layout
    dynamics instead of asking the 1D recurrent path to infer spatial geometry.
    """
    def __init__(self, cfg: NeedConfig):
        super().__init__()
        self.cfg = cfg
        self.value = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.gate = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.out = nn.Linear(cfg.d_model * 4, cfg.d_model, bias=False)
        self.norm = RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend)

    def _scan_axis(self, feat: torch.Tensor, dim: int, reverse: bool = False) -> torch.Tensor:
        # feat: [H,W,D]. Scan along dim 0 or 1 using the same chunked affine
        # recurrence fallback as text retention, avoiding per-row/column Python scans.
        if reverse:
            feat = torch.flip(feat, dims=(dim,))
        v = self.value(feat)
        g = torch.sigmoid(self.gate(feat)).to(feat.dtype)
        decay = float(self.cfg.image_2d_scan_decay)
        update = (1.0 - decay) * (g * v)
        decay_tensor = torch.full_like(update, decay)
        out = affine_recurrence_scan_torch(decay_tensor, update, dim=dim, chunk_size=16)
        if reverse:
            out = torch.flip(out, dims=(dim,))
        return out

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.cfg.image_2d_scan:
            return x, {"image_2d_scan_energy": x.new_tensor(0.0)}
        image_mask, segment_ids = image_span_token_mask(input_ids, self.cfg)
        bsz, seq_len = input_ids.shape
        y = torch.zeros_like(x)
        batch_ids = torch.arange(bsz, device=input_ids.device, dtype=torch.long).view(bsz, 1).expand_as(segment_ids)
        group_keys = batch_ids * (seq_len + 1) + segment_ids
        valid_flat = image_mask.reshape(-1).nonzero(as_tuple=False).flatten()
        if valid_flat.numel() == 0:
            return x, {"image_2d_scan_energy": x.new_tensor(0.0)}
        valid_keys = group_keys.reshape(-1).index_select(0, valid_flat)
        order = torch.argsort(valid_keys)
        flat_sorted = valid_flat.index_select(0, order)
        keys_sorted = valid_keys.index_select(0, order)
        _, counts_t = torch.unique_consecutive(keys_sorted, return_counts=True)
        groups_done = 0
        start = 0
        max_grid = int(max(1, getattr(self.cfg, "image_max_grid", 32)))
        for count in counts_t.tolist():
            group_flat = flat_sorted[start:start + count]
            start += count
            n = int(group_flat.numel())
            if n <= 1:
                continue
            batch = int((group_flat[0] // seq_len).detach().cpu())
            positions = (group_flat % seq_len).to(torch.long)
            # Infer image span grid from the actual payload count.
            grid = int(round(math.sqrt(float(n))))
            if grid * grid > n:
                grid -= 1
            grid = int(min(max_grid, max(1, grid)))
            usable = int(min(n, grid * grid))
            if grid > 1 and usable > 1:
                use = positions[:usable]
                seq_feat = self.norm(x[batch, use])
                feat = seq_feat[: grid * grid].contiguous().view(grid, grid, -1)
                lr = self._scan_axis(feat, dim=1, reverse=False)
                tb = self._scan_axis(feat, dim=0, reverse=False)
                if bool(getattr(self.cfg, "image_2d_bidirectional", False)):
                    rl = self._scan_axis(feat, dim=1, reverse=True)
                    bt = self._scan_axis(feat, dim=0, reverse=True)
                else:
                    rl = torch.zeros_like(lr)
                    bt = torch.zeros_like(tb)
                mix = self.out(torch.cat([lr, rl, tb, bt], dim=-1)).reshape(grid * grid, -1)[:usable]
                y[batch, use] = mix
                groups_done += 1
        if groups_done == 0:
            return x, {"image_2d_scan_energy": x.new_tensor(0.0)}
        strength = float(self.cfg.image_2d_scan_strength)
        out = x + strength * y
        return out, {"image_2d_scan_energy": y.float().pow(2).mean().detach()}


class ReasoningCompressor(nn.Module):
    """Small internal sidecar distilled from external-LM public-summary/summary behavior.

    It provides lightweight logits for summary/control tokens from NEED hidden states,
    so deployments can gradually reduce external external LM sidecar usage after
    distillation.  It is also used as a faithfulness probe between latent states and
    artificial CoT summaries.
    """
    def __init__(self, cfg: NeedConfig):
        super().__init__()
        self.summary = nn.Sequential(RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend), nn.Linear(cfg.d_model, cfg.d_model), nn.SiLU())
        self.token_head = nn.Linear(cfg.d_model, cfg.text_vocab_size, bias=False)
        self.faithfulness = nn.Sequential(RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend), nn.Linear(cfg.d_model, 3))

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled = h.mean(dim=1)
        z = self.summary(pooled)
        return {"summary_logits": self.token_head(z), "faithfulness": self.faithfulness(z)}


class SparseMoE(nn.Module):
    def __init__(self, cfg: NeedConfig):
        super().__init__()
        self.cfg = cfg
        hidden = cfg.d_ff or cfg.d_model * 4
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(cfg.d_model, hidden, cfg.dropout, cfg.kernel_backend) for _ in range(cfg.n_experts)])
        self.shared = SwiGLU(cfg.d_model, hidden, cfg.dropout, cfg.kernel_backend) if cfg.moe_use_shared_expert else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.cfg.n_experts <= 1 and self.shared is None:
            out = self.experts[0](x)
            z = x.new_tensor(0.0)
            return out, {"moe_balance": z, "branch_entropy": z, "moe_router_z": z}
        logits = self.router(x.float())
        if self.training and self.cfg.moe_router_jitter > 0:
            logits = logits + torch.randn_like(logits) * self.cfg.moe_router_jitter
        k = min(self.cfg.moe_top_k, self.cfg.n_experts)
        vals, idx = torch.topk(logits, k=k, dim=-1)
        weights = F.softmax(vals, dim=-1).to(x.dtype)
        if bool(getattr(self.cfg, "moe_static_dispatch", False)):
            # Static-shape dispatch for compile-friendly experiments.
            # This intentionally trades away sparse expert FLOP savings.
            combine_w = torch.zeros_like(logits).scatter(-1, idx, weights.to(logits.dtype)).to(x.dtype)
            expert_outputs = torch.stack([expert(x).to(x.dtype) for expert in self.experts], dim=-2)  # [B,T,E,D]
            out = (expert_outputs * combine_w.unsqueeze(-1)).sum(dim=-2)
        else:
            # True sparse dispatch: only selected experts see selected token rows,
            # so top-k routing saves actual expert compute. This path is kept out
            # of torch.compile/static-shape runs unless explicitly requested.
            out = torch.zeros_like(x)
            flat_x = x.reshape(-1, x.size(-1))
            flat_out = out.reshape(-1, out.size(-1))
            flat_idx = idx.reshape(-1, k)
            flat_w = weights.reshape(-1, k)
            for expert_id, expert in enumerate(self.experts):
                route_mask = flat_idx == expert_id
                token_rows = route_mask.any(dim=-1).nonzero(as_tuple=False).flatten()
                if token_rows.numel() == 0:
                    continue
                expert_in = flat_x.index_select(0, token_rows)
                expert_out = expert(expert_in).to(flat_out.dtype)
                expert_w = torch.where(
                    route_mask.index_select(0, token_rows),
                    flat_w.index_select(0, token_rows),
                    torch.zeros_like(flat_w.index_select(0, token_rows)),
                ).sum(dim=-1, keepdim=True).to(flat_out.dtype)
                flat_out.index_add_(0, token_rows, expert_out * expert_w)
            out = flat_out.view_as(out)
        if self.shared is not None:
            out = out + 0.25 * self.shared(x)
        probs = F.softmax(logits, dim=-1)
        load = probs.mean(dim=(0, 1))
        balance = (load * load).sum() * self.cfg.n_experts
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        z_loss = logits.pow(2).mean()
        return out, {"moe_balance": balance, "branch_entropy": entropy, "moe_router_z": z_loss}


class HierarchicalMemory(nn.Module):
    """Conditioned explicit memory for NEEDBlock.

    Memory is deliberately not a second dense recurrent stream competing with
    retention.  Retention/conv carry the implicit temporal state.  This module
    receives that temporal contribution as a condition, builds a query from it,
    and writes only a gated innovation residual into bounded semantic slots plus
    a linear associative key/value state.  The complexity is still linear in
    sequence length and bounded by memory_rank/memory_slots, but the role is now
    explicit: retrieve or commit information that the retention carrier exposed
    as useful, not rediscover retention with another recurrence.
    """
    def __init__(self, cfg: NeedConfig):
        super().__init__()
        self.cfg = cfg
        r = cfg.memory_rank
        d = cfg.d_model
        self.condition_norm = RMSNorm(d, kernel_backend=cfg.kernel_backend)
        self.condition_proj = nn.Linear(d, d, bias=False)
        self.condition_gate = nn.Linear(d * 2, 1, bias=True)
        self.write_mix = nn.Linear(d * 3, d, bias=False)
        self.write_gate = nn.Linear(d * 3, 1, bias=True)
        self.q = nn.Linear(d, r, bias=False)
        self.k = nn.Linear(d, r, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.cue = nn.Linear(d, d, bias=False)
        self.semantic_slots = nn.Parameter(torch.randn(cfg.memory_slots, r) / math.sqrt(r))
        self.semantic_values = nn.Parameter(torch.randn(cfg.memory_slots, d) / math.sqrt(d))
        self.out = nn.Linear(d * 3, d, bias=False)
        with torch.no_grad():
            if self.write_gate.bias is not None:
                self.write_gate.bias.fill_(-0.35)
            if self.condition_gate.bias is not None:
                self.condition_gate.bias.zero_()

    def _compose_inputs(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        innovation: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cond = torch.zeros_like(x) if condition is None else condition.to(device=x.device, dtype=x.dtype)
        innov = (x - cond) if innovation is None else innovation.to(device=x.device, dtype=x.dtype)
        cond_gate = torch.sigmoid(self.condition_gate(torch.cat([x.float(), cond.float()], dim=-1))).to(x.dtype)
        cond_delta = self.condition_proj(self.condition_norm(cond).float()).to(x.dtype)
        cond_strength = float(getattr(self.cfg, "memory_condition_strength", 0.35))
        query_src = x + cond_strength * cond_gate * cond_delta
        write_in = torch.cat([x.float(), cond.float(), innov.float()], dim=-1)
        write_gate = torch.sigmoid(self.write_gate(write_in)).to(x.dtype)
        write_src = F.silu(self.write_mix(write_in)).to(x.dtype) * write_gate
        return query_src, write_src, write_gate, cond_gate, innov

    def _linear_associative_memory(
        self,
        q_raw: torch.Tensor,
        k_raw: torch.Tensor,
        v: torch.Tensor,
        write_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Causal linear associative memory without a T x T score matrix.

        ``write_mask`` is a static-shape [B,T,1] mask used by adaptive-depth
        hard-skip. It zeros associative writes while preserving tensor shapes and
        avoiding host-side branching.
        """
        b, t, r = q_raw.shape
        d = v.size(-1)
        chunk = int(max(1, getattr(self.cfg, "memory_chunk_size", 32)))
        qf = (F.elu(q_raw.float()) + 1.0).to(v.dtype)
        kf = (F.elu(k_raw.float()) + 1.0).to(v.dtype)
        if write_mask is not None:
            wm = write_mask.to(device=v.device, dtype=v.dtype).clamp(0.0, 1.0)
            if wm.ndim == 2:
                wm = wm.unsqueeze(-1)
            kf = kf * wm
            v = v * wm
        state = v.new_zeros(b, r, d)
        denom_state = v.new_zeros(b, r)
        outs: List[torch.Tensor] = []
        eps = v.new_tensor(1e-6)
        for start in range(0, t, chunk):
            end = min(t, start + chunk)
            kc = kf[:, start:end]
            vc = v[:, start:end]
            qc = qf[:, start:end]
            kv = kc.unsqueeze(-1) * vc.unsqueeze(-2)
            state_c = state.unsqueeze(1) + kv.cumsum(dim=1)
            denom_c = denom_state.unsqueeze(1) + kc.cumsum(dim=1)
            numer = torch.einsum("bqr,bqrd->bqd", qc, state_c)
            denom = (qc * denom_c).sum(dim=-1, keepdim=True).clamp_min(eps)
            outs.append(numer / denom)
            state = state_c[:, -1].detach() if not torch.is_grad_enabled() else state_c[:, -1]
            denom_state = denom_c[:, -1].detach() if not torch.is_grad_enabled() else denom_c[:, -1]
        return torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        innovation: Optional[torch.Tensor] = None,
        write_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        query_src, write_src, write_gate, cond_gate, innov = self._compose_inputs(x, condition, innovation)
        q_raw = self.q(query_src.float())
        k_raw = self.k(write_src.float())
        q = F.normalize(q_raw, dim=-1)
        v = self.v(write_src.float()).to(x.dtype)
        sem_logits = torch.einsum("btr,sr->bts", q, F.normalize(self.semantic_slots.float(), dim=-1))
        sem_probs = F.softmax(sem_logits, dim=-1).to(x.dtype)
        semantic = torch.einsum("bts,sd->btd", sem_probs, self.semantic_values.to(x.dtype))
        associative = self._linear_associative_memory(q_raw, k_raw, v, write_mask=write_mask)
        cue = self.cue(write_src.float()).to(x.dtype)
        mixed = self.out(torch.cat([semantic, associative, cue], dim=-1).float()).to(x.dtype)
        ent = -(sem_probs.float() * sem_probs.float().clamp_min(1e-8).log()).sum(dim=-1).mean()
        div = (F.normalize(self.semantic_slots.float(), dim=-1) @ F.normalize(self.semantic_slots.float(), dim=-1).t()).pow(2).mean()
        qk_align = (F.normalize(q_raw.float(), dim=-1) * F.normalize(k_raw.float(), dim=-1)).sum(dim=-1).abs().mean()
        z = x.new_tensor(0.0)
        return mixed, {
            "memory_entropy": ent,
            "memory_diversity": div,
            "memory_write_gate": write_gate.float().mean().detach(),
            "memory_condition_gate": cond_gate.float().mean().detach(),
            "memory_innovation_norm": innov.float().pow(2).mean(dim=-1).sqrt().mean().detach(),
            "memory_query_key_alignment": qk_align.detach(),
            "memory_boundary": write_gate.float().mean().detach(),
        }

    def init_stream_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
        r = int(self.cfg.memory_rank)
        d = int(self.cfg.d_model)
        return {
            "assoc_state": torch.zeros(batch_size, r, d, device=device, dtype=dtype),
            "assoc_denom": torch.zeros(batch_size, r, device=device, dtype=dtype),
        }

    def stream_step(
        self,
        x: torch.Tensor,
        state: Dict[str, Any],
        condition: Optional[torch.Tensor] = None,
        innovation: Optional[torch.Tensor] = None,
        write_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """One-token conditioned explicit memory update."""
        if x.ndim != 3 or x.size(1) != 1:
            raise ValueError("stream_step expects x with shape [B,1,D]")
        b, _, d = x.shape
        if ("assoc_state" not in state or state["assoc_state"].size(0) != b or state["assoc_state"].device != x.device):
            state.update(self.init_stream_state(b, x.device, x.dtype))
        query_src, write_src, write_gate, cond_gate, innov = self._compose_inputs(x, condition, innovation)
        q_raw = self.q(query_src.float())
        k_raw = self.k(write_src.float())
        q = F.normalize(q_raw, dim=-1)
        v = self.v(write_src.float()).to(x.dtype)

        sem_logits = torch.einsum("br,sr->bs", q[:, 0], F.normalize(self.semantic_slots.float(), dim=-1))
        sem_probs = F.softmax(sem_logits, dim=-1).to(x.dtype)
        semantic = torch.einsum("bs,sd->bd", sem_probs, self.semantic_values.to(x.dtype))

        qf = (F.elu(q_raw.float()) + 1.0).to(x.dtype)[:, 0]
        kf = (F.elu(k_raw.float()) + 1.0).to(x.dtype)[:, 0]
        vv = v[:, 0]
        if write_mask is not None:
            wm = write_mask.to(device=x.device, dtype=x.dtype)
            if wm.ndim == 3:
                wm = wm[:, 0]
            wm = wm.view(wm.size(0), -1)[:, :1].clamp(0.0, 1.0)
            kf = kf * wm
            vv = vv * wm
        assoc_state = state["assoc_state"].to(x.dtype) + kf.unsqueeze(-1) * vv.unsqueeze(1)
        assoc_denom = state["assoc_denom"].to(x.dtype) + kf
        state["assoc_state"] = assoc_state
        state["assoc_denom"] = assoc_denom
        numer = torch.einsum("br,brd->bd", qf, assoc_state)
        denom = (qf * assoc_denom).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        associative = numer / denom
        cue = self.cue(write_src[:, 0].float()).to(x.dtype)
        mixed = self.out(torch.cat([semantic, associative, cue], dim=-1).float()).to(x.dtype).unsqueeze(1)
        ent = -(sem_probs.float() * sem_probs.float().clamp_min(1e-8).log()).sum(dim=-1).mean()
        div = (F.normalize(self.semantic_slots.float(), dim=-1) @ F.normalize(self.semantic_slots.float(), dim=-1).t()).pow(2).mean()
        qk_align = (F.normalize(q_raw.float(), dim=-1) * F.normalize(k_raw.float(), dim=-1)).sum(dim=-1).abs().mean()
        z = x.new_tensor(0.0)
        return mixed, {
            "memory_entropy": ent,
            "memory_diversity": div,
            "memory_write_gate": write_gate.float().mean().detach(),
            "memory_condition_gate": cond_gate.float().mean().detach(),
            "memory_innovation_norm": innov.float().pow(2).mean(dim=-1).sqrt().mean().detach(),
            "memory_query_key_alignment": qk_align.detach(),
            "memory_boundary": write_gate.float().mean().detach(),
        }


class ConvexEnergy(nn.Module):
    def __init__(self, cfg: NeedConfig):
        super().__init__()
        self.cfg = cfg
        self.center = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.precision_raw = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.rows = nn.Parameter(torch.randn(cfg.energy_rank, cfg.d_model) / math.sqrt(cfg.d_model))

    def _params(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        center = self.center(x)
        precision = F.softplus(self.precision_raw(x.float())).to(x.dtype) + self.cfg.min_precision
        rows = F.normalize(self.rows.float(), dim=-1).to(x.dtype) * self.cfg.energy_row_norm
        return center, precision, rows

    def energy_and_grad(self, z: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c, p, rows = self._params(context)
        diff = z - c
        e_diag = 0.5 * (p * diff.pow(2)).sum(dim=-1)
        proj = torch.einsum("btd,rd->btr", diff, rows)
        e_rank = 0.5 * proj.pow(2).sum(dim=-1)
        energy = e_diag + e_rank
        grad = p * diff + torch.einsum("btr,rd->btd", proj, rows)
        return energy, grad

    def row_orth_loss(self) -> torch.Tensor:
        r = F.normalize(self.rows.float(), dim=-1)
        gram = r @ r.t()
        ident = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
        return (gram - ident).pow(2).mean()


class AdaptiveEquilibrium(nn.Module):
    def __init__(self, cfg: NeedConfig):
        super().__init__()
        self.cfg = cfg
        self.energy = ConvexEnergy(cfg)
        self.difficulty = nn.Linear(cfg.d_model, 1, bias=True)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = x
        effort = torch.sigmoid(self.difficulty(context.float())).to(x.dtype)
        last_e = x.new_tensor(0.0)
        last_grad = torch.zeros_like(x)
        n_steps = max(1, int(self.cfg.energy_steps))
        min_steps = int(self.cfg.energy_min_steps)
        steps_run = 0
        # `active_steps` is the differentiable analogue of steps_run below: the
        # guaranteed floor counts fully, later steps count by however much `gate`
        # (defined per-step just below) still said "keep going".
        active_steps = x.new_tensor(float(min(min_steps, n_steps)))
        for step in range(n_steps):
            e, grad = self.energy.energy_and_grad(z, context)
            resid = grad.float().pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt().to(grad.dtype)
            last_e = e
            last_grad = grad
            steps_run = step + 1
            # One gate, computed once, shared by everything below: how far this
            # step's residual still is from equilibrium. gate -> 0 once resid
            # drops under the threshold ("done"), -> 1 while still far ("keep
            # going"). It drives (a) the eval-time exit test, (b) the train-time
            # gradient dampening, and (c) the effort-loss step count, so the three
            # stay consistent with each other.
            gate = torch.sigmoid(self.cfg.adaptive_effort_alpha * (resid - self.cfg.adaptive_residual_threshold))
            if step >= min_steps:
                active_steps = active_steps + gate.mean()
            # Keep the loop count static. The continuous gate below still damps
            # converged states without a host round-trip.
            run: Union[float, torch.Tensor] = 1.0
            if self.cfg.adaptive_energy and step >= min_steps:
                run = gate * effort
            z = z - self.cfg.energy_step_size * grad * run
        # Losses and diagnostics must describe the state actually returned from
        # the fixed-point iteration.
        last_e, last_grad = self.energy.energy_and_grad(z, context)
        # Continuous proxy for steps_run/n_steps built from the same per-step
        # gate used above, so it stays meaningful (and non-constant) whether or
        # not the hard exit actually fired.
        step_frac = (active_steps / float(n_steps)).clamp(0.0, 1.0)
        return z, last_grad.pow(2).mean(), last_e.mean(), effort.mean() * step_frac


class LatentPlanner(nn.Module):
    """Continuous latent trajectory planner.

    The planner has two complementary modes.  The legacy transition path advances
    a latent cursor horizon by horizon, which remains useful for token-conditioned
    DVSD compounding.  The block-space path opens every future slot at once and
    exchanges information with prefix/suffix scans rather than bidirectional
    attention, so a horizon can be shaped by the intended block without an H x H
    score matrix.

    The planner also exposes a token-conditioned compound step for DVSD.  A
    virtual-slot decoder can feed each provisional token back into the latent
    cursor using the token embedding and a cheap logit-descent direction. Later
    slots then see a compounded state rather than a shallow same-state MTP guess.
    """
    def __init__(self, cfg: NeedConfig):
        super().__init__()
        self.cfg = cfg
        self.horizons = cfg.planner_horizons
        self.init = nn.Sequential(RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend), nn.Linear(cfg.d_model, cfg.d_model))
        self.time = nn.Embedding(max(1, cfg.planner_horizons + 1), cfg.d_model)
        hidden = max(cfg.d_model, cfg.d_ff // 2 if cfg.d_ff else cfg.d_model * 2)
        layers: List[nn.Module] = [RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend)]
        for _ in range(max(1, cfg.planner_transition_depth)):
            layers.extend([nn.Linear(cfg.d_model, hidden), nn.SiLU(), nn.Linear(hidden, cfg.d_model)])
        self.transition = nn.Sequential(*layers)
        self.gate = nn.Sequential(RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend), nn.Linear(cfg.d_model, cfg.d_model), nn.Sigmoid())
        self.feedback_norm = RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend)
        self.descent_norm = RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend)
        self.token_feedback = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.SiLU(), nn.Linear(cfg.d_model, cfg.d_model, bias=False))
        self.descent_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.feedback_gate = nn.Sequential(RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend), nn.Linear(cfg.d_model, cfg.d_model), nn.Sigmoid())
        self.block_slot = nn.Parameter(torch.randn(max(1, cfg.planner_horizons), cfg.d_model) / math.sqrt(max(1, cfg.d_model)))
        self.block_norm = RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend)
        self.block_q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.block_k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.block_v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.block_gate = nn.Sequential(RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend), nn.Linear(cfg.d_model, cfg.d_model), nn.Sigmoid())
        self.block_ff = nn.Sequential(RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend), nn.Linear(cfg.d_model, hidden), nn.SiLU(), nn.Linear(hidden, cfg.d_model))
        self.block_out = RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend)
        self.out = RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend)

    def start_state(self, h: torch.Tensor) -> torch.Tensor:
        """Project decoder hidden states into the planner's latent cursor space."""
        return self.init(h)

    def _time_embedding(self, step_index: int, state: torch.Tensor) -> torch.Tensor:
        idx = min(max(0, int(step_index)), self.time.num_embeddings - 1)
        te = self.time.weight[idx].to(dtype=state.dtype, device=state.device)
        view = [1] * (state.ndim - 1) + [te.numel()]
        return te.view(*view)

    def compound_step(
        self,
        state: torch.Tensor,
        token_feedback: torch.Tensor,
        step_index: int,
        logit_descent: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Advance a latent cursor after one provisional or teacher-forced token.

        ``logit_descent`` is an approximate negative CE gradient in hidden space:
        target token embedding minus the current expected output embedding.  It is
        cheap to compute from top-k logits and lets DVSD token choices compound in
        latent space without a full model pass for every slot.
        """
        squeeze = False
        if state.ndim == 2:
            state = state.unsqueeze(1)
            token_feedback = token_feedback.unsqueeze(1) if token_feedback.ndim == 2 else token_feedback
            if logit_descent is not None and logit_descent.ndim == 2:
                logit_descent = logit_descent.unsqueeze(1)
            squeeze = True
        token_feedback = token_feedback.to(dtype=state.dtype, device=state.device)
        fb = self.token_feedback(self.feedback_norm(token_feedback))
        if logit_descent is not None:
            descent = logit_descent.to(dtype=state.dtype, device=state.device)
            fb = fb + float(getattr(self.cfg, "dvsd_planner_compound_descent_scale", 0.0)) * self.descent_proj(self.descent_norm(descent))
        fb_gate = self.feedback_gate(state + self._time_embedding(step_index, state))
        candidate = state + self._time_embedding(step_index, state) + float(getattr(self.cfg, "dvsd_planner_compound_token_scale", 0.0)) * fb_gate * fb
        delta = self.transition(candidate)
        gate = self.gate(candidate)
        if confidence is not None:
            conf = confidence.to(dtype=gate.dtype, device=gate.device)
            while conf.ndim < gate.ndim:
                conf = conf.unsqueeze(-1)
            gate = gate * (0.35 + 0.65 * conf.clamp(0.0, 1.0))
        state = state + float(getattr(self.cfg, "dvsd_planner_compound_step_size", 1.0)) * gate * delta
        state = state + float(getattr(self.cfg, "dvsd_planner_compound_token_scale", 0.0)) * fb_gate * fb
        state = self.out(state)
        return state.squeeze(1) if squeeze else state

    def _block_space(self, h: torch.Tensor) -> List[torch.Tensor]:
        """Plan all virtual future slots with linear horizon mixing.

        This used to be a small bidirectional attention block over future slots.
        Even though the horizon is capped, that made the efficient planner look
        Transformer-like internally.  The replacement uses prefix/suffix scans over
        the horizon axis, so planned slots can exchange coarse block context without
        constructing an H x H score matrix.
        """
        if self.horizons <= 0:
            return []
        base = self.start_state(h)
        hh = min(self.horizons, self.time.num_embeddings - 1 if self.time.num_embeddings > 1 else self.horizons)
        if hh <= 0:
            return []
        idx = torch.arange(1, hh + 1, device=h.device).clamp(max=self.time.num_embeddings - 1)
        te = self.time(idx).to(dtype=base.dtype).view(1, 1, hh, -1)
        slot = self.block_slot[:hh].to(dtype=base.dtype, device=base.device).view(1, 1, hh, -1)
        z = base.unsqueeze(2) + te + slot
        denom = torch.arange(1, hh + 1, device=h.device, dtype=torch.float32).view(1, 1, hh, 1).to(dtype=z.dtype)
        rdenom = torch.arange(hh, 0, -1, device=h.device, dtype=torch.float32).view(1, 1, hh, 1).to(dtype=z.dtype)
        for _ in range(max(1, int(getattr(self.cfg, "planner_block_space_iters", 1)))):
            zn = self.block_norm(z)
            value = self.block_v(zn)
            prefix = value.cumsum(dim=2) / denom.clamp_min(1.0)
            suffix = torch.flip(torch.flip(value, dims=[2]).cumsum(dim=2), dims=[2]) / rdenom.clamp_min(1.0)
            ctx = 0.5 * (prefix + suffix)
            cand = z + ctx
            z = z + self.block_gate(cand) * self.block_ff(cand)
        z = self.block_out(z)
        return [z[:, :, i, :] for i in range(hh)]

    def forward(self, h: torch.Tensor) -> List[torch.Tensor]:
        if self.horizons <= 0:
            return []
        use_block = bool(getattr(self.cfg, "planner_block_space_enabled", True)) and self.horizons > 0
        block_outs = self._block_space(h) if use_block else []
        mix = float(getattr(self.cfg, "planner_block_space_mix", 0.0)) if block_outs else 0.0
        mix = min(max(mix, 0.0), 1.0)
        seq_outs: List[torch.Tensor] = []
        if mix < 1.0 or not block_outs:
            state = self.start_state(h)
            for horizon in range(1, self.horizons + 1):
                te = self._time_embedding(horizon, state)
                candidate = state + te
                delta = self.transition(candidate)
                gate = self.gate(candidate)
                state = state + gate * delta
                seq_outs.append(self.out(state))
        if not block_outs:
            return seq_outs
        if not seq_outs:
            return block_outs
        n = min(len(seq_outs), len(block_outs))
        outs = [(1.0 - mix) * seq_outs[i] + mix * block_outs[i] for i in range(n)]
        if len(seq_outs) > n:
            outs.extend(seq_outs[n:])
        elif len(block_outs) > n:
            outs.extend(block_outs[n:])
        return outs

def sinusoidal_position_encoding(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / max(1, dim)))
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    if dim > 1:
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe.to(dtype=dtype)


class TemporalPathwayConditioner(nn.Module):
    """Inject an ordered latent reasoning path without collapsing it to one vector."""
    def __init__(self, cfg: NeedConfig):
        super().__init__()
        self.cfg = cfg
        self.heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.x_norm = RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend)
        self.path_norm = RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend)
        self.delta_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.path_gru = nn.GRU(cfg.d_model, cfg.d_model, batch_first=True)
        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.mix = nn.Linear(cfg.d_model * 3, cfg.d_model, bias=False)
        self.gate = nn.Sequential(nn.Linear(cfg.d_model * 2, cfg.d_model), nn.SiLU(), nn.Linear(cfg.d_model, cfg.d_model), nn.Sigmoid())
        self.mem_key = nn.Parameter(torch.randn(cfg.pathway_memory_slots, cfg.d_model) / math.sqrt(cfg.d_model))
        self.mem_val = nn.Parameter(torch.randn(cfg.pathway_memory_slots, cfg.d_model) / math.sqrt(cfg.d_model))
        self.mem_gate = nn.Linear(cfg.d_model * 2, cfg.d_model, bias=True)
        self.drop = nn.Dropout(cfg.pathway_conditioning_dropout)

    def forward(self, x: torch.Tensor, path: torch.Tensor, scale: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if path.ndim == 2:
            path = path.unsqueeze(0)
        if path.size(0) == 1 and x.size(0) > 1:
            path = path.expand(x.size(0), -1, -1)
        if path.size(-1) != x.size(-1):
            raise ValueError(f"conditioning vector dim {path.size(-1)} does not match d_model {x.size(-1)}")
        max_len = max(1, int(self.cfg.pathway_conditioning_max_vectors))
        if path.size(1) > max_len:
            # Length-preserving enough for ordering: sample uniformly instead of global average pooling.
            take = torch.linspace(0, path.size(1) - 1, max_len, device=path.device).round().long()
            path = path.index_select(1, take)
        path = self.path_norm(path.to(device=x.device, dtype=x.dtype))
        pos = sinusoidal_position_encoding(path.size(1), path.size(2), x.device, x.dtype).unsqueeze(0)
        delta = torch.cat([torch.zeros_like(path[:, :1]), path[:, 1:] - path[:, :-1]], dim=1)
        encoded, _ = self.path_gru(path + 0.05 * pos + self.delta_proj(delta))
        b, t, d = x.shape
        p = encoded.size(1)
        x_norm = self.x_norm(x)
        top_k = int(max(1, min(self.cfg.pathway_conditioning_top_k, p)))
        q = self.q(x_norm).view(b, t, self.heads, self.head_dim)
        if top_k < p:
            # Per-token anchor selection is causal because it depends only on the
            # current token state and the external path. It avoids the previous
            # full-sequence summary, which let future tokens choose prefix anchors.
            pre_scores = torch.einsum("btd,bpd->btp", x_norm.float(), encoded.float()) / math.sqrt(max(1, d))
            anchor_idx = torch.topk(pre_scores, k=top_k, dim=-1).indices.sort(dim=-1).values
            gather = anchor_idx.unsqueeze(-1).expand(-1, -1, -1, d)
            enc_exp = encoded.unsqueeze(1).expand(-1, t, -1, -1)
            encoded_att = torch.gather(enc_exp, 2, gather)
            k = self.k(encoded_att).view(b, t, top_k, self.heads, self.head_dim)
            v = self.v(encoded_att).view(b, t, top_k, self.heads, self.head_dim)
            scores = torch.einsum("bthd,btkhd->bhtk", q.float(), k.float()) / math.sqrt(self.head_dim)
            probs = F.softmax(scores, dim=-1).to(x.dtype)
            ctx = torch.einsum("bhtk,btkhd->bthd", probs, v).reshape(b, t, d)
        else:
            k = self.k(encoded).view(b, p, self.heads, self.head_dim)
            v = self.v(encoded).view(b, p, self.heads, self.head_dim)
            scores = torch.einsum("bthd,bphd->bhtp", q.float(), k.float()) / math.sqrt(self.head_dim)
            probs = F.softmax(scores, dim=-1).to(x.dtype)
            ctx = torch.einsum("bhtp,bphd->bthd", probs, v).reshape(b, t, d)
        endpoint = encoded[:, -1:].expand(-1, t, -1)
        trend = (encoded[:, -1:] - encoded[:, :1]).expand(-1, t, -1)
        mem_scores = torch.einsum("btd,md->btm", x_norm.float(), F.normalize(self.mem_key.float(), dim=-1))
        mk = int(max(1, min(self.cfg.pathway_memory_top_k, self.mem_key.size(0))))
        if mk < self.mem_key.size(0):
            kth = torch.topk(mem_scores, k=mk, dim=-1).values[..., -1:]
            mem_scores = mem_scores.masked_fill(mem_scores < kth, -1e9)
        mem_prob = F.softmax(mem_scores, dim=-1).to(x.dtype)
        mem_ctx = torch.einsum("btm,md->btd", mem_prob, self.mem_val.to(x.dtype))
        cond = self.mix(torch.cat([ctx + 0.25 * mem_ctx, endpoint, trend], dim=-1))
        cond = cond + torch.sigmoid(self.mem_gate(torch.cat([cond, mem_ctx], dim=-1))) * mem_ctx
        gate = self.gate(torch.cat([x, cond], dim=-1))
        y = x + float(scale) * self.drop(gate * cond)
        ent = -(probs.float() * probs.float().clamp_min(1e-8).log()).sum(dim=-1).mean()
        peak = probs.float().amax(dim=-1).mean()
        mem_ent = -(mem_prob.float() * mem_prob.float().clamp_min(1e-8).log()).sum(dim=-1).mean()
        return y, {"pathway_attention_entropy": ent, "pathway_attention_peak": peak.detach(), "pathway_memory_entropy": mem_ent}

    def prepare_stream_state(self, path: torch.Tensor, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        """Encode conditioning vectors once for cached generation."""
        if path.ndim == 2:
            path = path.unsqueeze(0)
        if path.size(0) == 1 and batch_size > 1:
            path = path.expand(batch_size, -1, -1)
        if path.size(-1) != int(self.cfg.d_model):
            raise ValueError(f"conditioning vector dim {path.size(-1)} does not match d_model {self.cfg.d_model}")
        max_len = max(1, int(self.cfg.pathway_conditioning_max_vectors))
        if path.size(1) > max_len:
            take = torch.linspace(0, path.size(1) - 1, max_len, device=path.device).round().long()
            path = path.index_select(1, take)
        path = self.path_norm(path.to(device=device, dtype=dtype))
        pos = sinusoidal_position_encoding(path.size(1), path.size(2), device, dtype).unsqueeze(0)
        delta = torch.cat([torch.zeros_like(path[:, :1]), path[:, 1:] - path[:, :-1]], dim=1)
        encoded, _ = self.path_gru(path + 0.05 * pos + self.delta_proj(delta))
        return {
            "encoded": encoded.detach() if not torch.is_grad_enabled() else encoded,
            "endpoint": encoded[:, -1:].detach() if not torch.is_grad_enabled() else encoded[:, -1:],
            "trend": (encoded[:, -1:] - encoded[:, :1]).detach() if not torch.is_grad_enabled() else (encoded[:, -1:] - encoded[:, :1]),
        }

    def step(self, x: torch.Tensor, path_cache: Dict[str, torch.Tensor], scale: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x.ndim != 3 or x.size(1) != 1:
            raise ValueError("TemporalPathwayConditioner.step expects [B,1,D]")
        encoded = path_cache["encoded"].to(device=x.device, dtype=x.dtype)
        endpoint = path_cache["endpoint"].to(device=x.device, dtype=x.dtype)
        trend = path_cache["trend"].to(device=x.device, dtype=x.dtype)
        b, t, d = x.shape
        p = encoded.size(1)
        x_norm = self.x_norm(x)
        top_k = int(max(1, min(self.cfg.pathway_conditioning_top_k, p)))
        # Per-token top-k over a fixed capped pathway axis.  This is O(P) with P
        # clamped by strict core, not O(T*P) per emitted token.
        pre = torch.einsum("btd,bpd->btp", x_norm.float(), encoded.float()) / math.sqrt(max(1, d))
        if top_k < p:
            anchor_idx = torch.topk(pre.squeeze(1), k=top_k, dim=-1).indices.sort(dim=-1).values
            gather = anchor_idx.unsqueeze(-1).expand(-1, -1, d)
            encoded_att = torch.gather(encoded, 1, gather)
        else:
            encoded_att = encoded
        pa = encoded_att.size(1)
        q = self.q(x_norm).view(b, t, self.heads, self.head_dim)
        k = self.k(encoded_att).view(b, pa, self.heads, self.head_dim)
        v = self.v(encoded_att).view(b, pa, self.heads, self.head_dim)
        scores = torch.einsum("bthd,bphd->bhtp", q.float(), k.float()) / math.sqrt(self.head_dim)
        probs = F.softmax(scores, dim=-1).to(x.dtype)
        ctx = torch.einsum("bhtp,bphd->bthd", probs, v).reshape(b, t, d)
        mem_scores = torch.einsum("btd,md->btm", x_norm.float(), F.normalize(self.mem_key.float(), dim=-1))
        mk = int(max(1, min(self.cfg.pathway_memory_top_k, self.mem_key.size(0))))
        if mk < self.mem_key.size(0):
            kth = torch.topk(mem_scores, k=mk, dim=-1).values[..., -1:]
            mem_scores = mem_scores.masked_fill(mem_scores < kth, -1e9)
        mem_prob = F.softmax(mem_scores, dim=-1).to(x.dtype)
        mem_ctx = torch.einsum("btm,md->btd", mem_prob, self.mem_val.to(x.dtype))
        cond = self.mix(torch.cat([ctx + 0.25 * mem_ctx, endpoint.expand(-1, 1, -1), trend.expand(-1, 1, -1)], dim=-1))
        cond = cond + torch.sigmoid(self.mem_gate(torch.cat([cond, mem_ctx], dim=-1))) * mem_ctx
        gate = self.gate(torch.cat([x, cond], dim=-1))
        y = x + float(scale) * self.drop(gate * cond)
        ent = -(probs.float() * probs.float().clamp_min(1e-8).log()).sum(dim=-1).mean()
        peak = probs.float().amax(dim=-1).mean()
        mem_ent = -(mem_prob.float() * mem_prob.float().clamp_min(1e-8).log()).sum(dim=-1).mean()
        return y, {"pathway_attention_entropy": ent, "pathway_attention_peak": peak.detach(), "pathway_memory_entropy": mem_ent}



class SlotAttentionBlock(nn.Module):
    """Shared fixed-slot primitive with a causal pooled core by default.

    The pooled core deliberately avoids full-sequence pooling before token logits.
    Each token position is conditioned only on prefix statistics up to that token.
    The old dense token-slot attention remains an explicit ablation mode, but it no
    longer feeds non-causal token context in the default language-model path.
    """
    def __init__(self, cfg: NeedConfig, num_slots: int):
        super().__init__()
        self.cfg = cfg
        self.num_slots = max(1, int(num_slots))
        self.seed = nn.Parameter(torch.randn(self.num_slots, cfg.d_model) / math.sqrt(cfg.d_model))
        self.input_norm = RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend)
        self.base = nn.Sequential(RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend), nn.Linear(cfg.d_model, cfg.d_model), nn.SiLU())
        self.base_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.slot_q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.token_k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.token_v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.token_q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.slot_k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.slot_v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.token_out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def _causal_prefix_mean(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            m = mask.to(h.dtype).unsqueeze(-1)
            numer = (h * m).cumsum(dim=1)
            denom = m.cumsum(dim=1).clamp_min(1.0)
            return numer / denom
        counts = torch.arange(1, h.size(1) + 1, device=h.device, dtype=h.dtype).view(1, -1, 1)
        return h.cumsum(dim=1) / counts

    def _pool(self, h: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # Final-sequence pool is used only for sequence-level auxiliaries/exports.
        if mask is not None:
            m = mask.to(h.dtype).unsqueeze(-1)
            denom = m.sum(dim=1).clamp_min(1.0)
            return (h * m).sum(dim=1) / denom
        return h.mean(dim=1)

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None, need_token_context: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        b, t, d = h.shape
        mode = str(getattr(self.cfg, "slot_attention_mode", "pooled")).lower()
        if mode == "pooled":
            prefix = self._causal_prefix_mean(h, mask)
            pooled_prefix = self.base(prefix.float())
            seed = self.seed.to(h.dtype).view(1, 1, self.num_slots, d)
            slot_seq = self.token_out(seed + self.base_proj(pooled_prefix).unsqueeze(2).to(h.dtype))  # [B,T,S,D]
            slots = slot_seq[:, -1]
            token_delta: Optional[torch.Tensor] = None
            if need_token_context:
                ctx = slot_seq.mean(dim=2)
                token_delta = self.token_out(ctx)
            norm_slots = F.normalize(slots.float(), dim=-1)
            gram = torch.einsum('bsd,bqd->bsq', norm_slots, norm_slots)
            ident = torch.eye(self.num_slots, device=h.device, dtype=gram.dtype).unsqueeze(0)
            div = (gram - ident).pow(2).mean()
            z = h.new_tensor(0.0)
            return slot_seq, token_delta, {
                "slot_attention_entropy": z,
                "slot_attention_coverage": z,
                "slot_context_entropy": z,
                "slot_pooled_core": h.new_tensor(1.0),
                "slot_attention_causal": h.new_tensor(1.0),
                "latent_slot_diversity_raw": div.detach(),
            }

        # Dense attention is an ablation mode.  To avoid future leakage into LM
        # logits, it may build sequence-level slots but does not return token_delta.
        pooled = self.base(self._pool(h, mask).float())
        seeds = self.seed.to(h.dtype).unsqueeze(0).expand(b, -1, -1) + self.base_proj(pooled).unsqueeze(1).to(h.dtype)
        token_state = self.input_norm(h).float()
        slot_scores = torch.einsum('bsd,btd->bst', self.slot_q(seeds.float()), self.token_k(token_state)) / math.sqrt(d)
        if mask is not None:
            slot_scores = slot_scores.masked_fill(~mask.bool().unsqueeze(1), -1e9)
        slot_att = F.softmax(slot_scores, dim=-1).to(h.dtype)
        slots = torch.einsum('bst,btd->bsd', slot_att, self.token_v(h).to(h.dtype))
        token_delta = None
        slot_entropy = -(slot_att.float() * slot_att.float().clamp_min(1e-8).log()).sum(dim=-1).mean()
        coverage = slot_att.float().max(dim=1).values.mean()
        z = h.new_tensor(0.0)
        return slots, token_delta, {
            "slot_attention_entropy": slot_entropy,
            "slot_attention_coverage": coverage.detach(),
            "slot_context_entropy": z,
            "slot_attention_causal": h.new_tensor(0.0),
        }

class MixtureEnergyRouter(nn.Module):
    """Mixture of domain/task-specific latent energy routers.

    The router is causal: each token route is computed from the prefix mean up to
    that token.
    """
    def __init__(self, cfg: NeedConfig):
        super().__init__()
        self.cfg = cfg
        n = max(1, int(cfg.energy_routes))
        self.n = n
        self.router = nn.Sequential(RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend), nn.Linear(cfg.d_model, n))
        self.centers = nn.Parameter(torch.randn(n, cfg.d_model) / math.sqrt(cfg.d_model))
        self.rows = nn.Parameter(torch.randn(n, max(1, cfg.energy_rank), cfg.d_model) / math.sqrt(cfg.d_model))
        self.precision = nn.Parameter(torch.zeros(n, cfg.d_model))
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def _prefix_mean(self, h: torch.Tensor) -> torch.Tensor:
        counts = torch.arange(1, h.size(1) + 1, device=h.device, dtype=h.dtype).view(1, -1, 1)
        return h.cumsum(dim=1) / counts

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        prefix = self._prefix_mean(h)
        route_logits = self.router(prefix.float())  # [B,T,N]
        route = F.softmax(route_logits, dim=-1).to(h.dtype)
        centers = torch.einsum('btn,nd->btd', route, self.centers.to(h.dtype))
        precision = F.softplus(torch.einsum('btn,nd->btd', route.float(), self.precision.float())).to(h.dtype) + self.cfg.min_precision
        norm_rows = F.normalize(self.rows.float(), dim=-1)
        rows = torch.einsum('btn,nrd->btrd', route.float(), norm_rows).to(h.dtype) * self.cfg.energy_row_norm
        z = h
        energy_total = h.new_tensor(0.0)
        for _ in range(max(1, int(self.cfg.energy_route_steps))):
            diff = z - centers
            proj = torch.einsum('btd,btrd->btr', diff, rows)
            grad = precision * diff + torch.einsum('btr,btrd->btd', proj, rows)
            z = z - float(self.cfg.energy_step_size) * float(self.cfg.energy_route_strength) * grad
            energy_total = energy_total + (0.5 * (precision * diff.pow(2)).sum(dim=-1) + 0.5 * proj.pow(2).sum(dim=-1)).mean()
        route_ent = -(route.float() * route.float().clamp_min(1e-8).log()).sum(dim=-1).mean()
        route_balance = (route.float().mean(dim=(0, 1)).pow(2).sum() * self.n)
        return self.out(z - h), {"mixture_energy_router_energy": energy_total, "energy_route_entropy": route_ent, "energy_route_balance": route_balance, "mixture_energy_router_causal": h.new_tensor(1.0)}

class LatentSlotAttention(nn.Module):
    """Build reusable latent slots and inject causal slot context back into token states."""
    def __init__(self, cfg: NeedConfig):
        super().__init__()
        self.cfg = cfg
        self.slots = max(1, int(cfg.latent_slots))
        self.block = SlotAttentionBlock(cfg, self.slots)

    def _final_slots(self, slots: torch.Tensor) -> torch.Tensor:
        return slots[:, -1] if slots.ndim == 4 else slots

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        slots, delta, aux = self.block(h, need_token_context=True)
        if delta is None:
            delta = torch.zeros_like(h)
        final_slots = self._final_slots(slots)
        norm_slots = F.normalize(final_slots.float(), dim=-1)
        gram = torch.einsum('bsd,bqd->bsq', norm_slots, norm_slots)
        ident = torch.eye(self.slots, device=h.device, dtype=gram.dtype).unsqueeze(0)
        div = (gram - ident).pow(2).mean()
        return delta, slots, {
            "latent_slot_attention_entropy": aux["slot_context_entropy"],
            "latent_slot_coverage": aux["slot_attention_coverage"],
            "latent_slot_diversity": div,
            "latent_slot_causal": aux.get("slot_attention_causal", h.new_tensor(0.0)),
        }

    def step_from_pooled(self, h: torch.Tensor, pooled: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """One-token pooled-slot update for the strict streaming core."""
        if str(getattr(self.cfg, "slot_attention_mode", "pooled")).lower() != "pooled":
            return self.forward(h)
        b, _, d = h.shape
        pooled_base = self.block.base(pooled.float())
        seeds = self.block.seed.to(h.dtype).unsqueeze(0).expand(b, -1, -1) + self.block.base_proj(pooled_base).unsqueeze(1).to(h.dtype)
        slots = self.block.token_out(seeds)
        ctx = slots.mean(dim=1, keepdim=True)
        delta = self.block.token_out(ctx)
        norm_slots = F.normalize(slots.float(), dim=-1)
        gram = torch.einsum('bsd,bqd->bsq', norm_slots, norm_slots)
        ident = torch.eye(self.slots, device=h.device, dtype=gram.dtype).unsqueeze(0)
        div = (gram - ident).pow(2).mean()
        z = h.new_tensor(0.0)
        return delta, slots, {
            "latent_slot_attention_entropy": z,
            "latent_slot_coverage": z,
            "latent_slot_diversity": div,
            "latent_slot_causal": h.new_tensor(1.0),
        }

class HierarchicalTimeScales(nn.Module):
    """Fast/medium/slow causal latent stream mixer.

    The previous implementation pooled complete chunks and broadcast the chunk
    state back to every token, which let early logits see later tokens in the same
    chunk.  This version uses the running mean inside each chunk, then causal GRUs
    over those prefix summaries.
    """
    def __init__(self, cfg: NeedConfig):
        super().__init__()
        self.cfg = cfg
        self.med = nn.GRU(cfg.d_model, cfg.d_model, batch_first=True)
        self.slow = nn.GRU(cfg.d_model, cfg.d_model, batch_first=True)
        self.mix = nn.Sequential(RMSNorm(cfg.d_model * 3, kernel_backend=cfg.kernel_backend), nn.Linear(cfg.d_model * 3, cfg.d_model), nn.SiLU(), nn.Linear(cfg.d_model, cfg.d_model))

    def _causal_chunk_means(self, h: torch.Tensor) -> torch.Tensor:
        b, t, d = h.shape
        chunk = max(1, int(self.cfg.slow_state_chunk))
        pad = (chunk - (t % chunk)) % chunk
        hp = F.pad(h, (0, 0, 0, pad)) if pad else h
        n_chunks = hp.size(1) // chunk
        hv = hp.view(b, n_chunks, chunk, d)
        csum = hv.cumsum(dim=2)
        counts = torch.arange(1, chunk + 1, device=h.device, dtype=h.dtype).view(1, 1, chunk, 1)
        prefix = csum / counts
        return prefix.reshape(b, n_chunks * chunk, d)[:, :t]

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        prefix_chunks = self._causal_chunk_means(h)
        med, _ = self.med(prefix_chunks)
        slow, _ = self.slow(med)
        delta = self.mix(torch.cat([h, med.to(h.dtype), slow.to(h.dtype)], dim=-1))
        consistency = (med.float()[:, 1:] - med.float()[:, :-1]).pow(2).mean() if h.size(1) > 1 else h.new_tensor(0.0)
        return h + float(self.cfg.slow_state_strength) * delta.to(h.dtype), {"timescale_consistency": consistency, "slow_state_norm": slow.float().pow(2).mean().detach(), "timescale_causal": h.new_tensor(1.0)}

    def init_stream_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        d = int(self.cfg.d_model)
        return {
            "chunk_sum": torch.zeros(batch_size, d, device=device, dtype=dtype),
            "chunk_count": torch.zeros(batch_size, 1, device=device, dtype=dtype),
            "med_h": torch.zeros(1, batch_size, d, device=device, dtype=dtype),
            "slow_h": torch.zeros(1, batch_size, d, device=device, dtype=dtype),
            "prev_med": torch.zeros(batch_size, d, device=device, dtype=dtype),
        }

    def step(self, h: torch.Tensor, cache: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if h.ndim != 3 or h.size(1) != 1:
            raise ValueError("HierarchicalTimeScales.step expects [B,1,D]")
        b, _, d = h.shape
        chunk = max(1, int(self.cfg.slow_state_chunk))
        if cache.get("chunk_sum") is None or cache["chunk_sum"].size(0) != b:
            cache.update(self.init_stream_state(b, h.device, h.dtype))
        chunk_sum = cache["chunk_sum"].to(h.dtype) + h[:, 0]
        chunk_count = cache["chunk_count"].to(h.dtype) + 1.0
        chunk_mean = chunk_sum / chunk_count.clamp_min(1.0)
        med_h = cache["med_h"].to(h.dtype)
        slow_h = cache["slow_h"].to(h.dtype)
        med_seq, med_tmp = self.med(chunk_mean.unsqueeze(1), med_h)
        slow_seq, slow_tmp = self.slow(med_seq, slow_h)
        delta = self.mix(torch.cat([h, med_seq.to(h.dtype), slow_seq.to(h.dtype)], dim=-1))
        complete = chunk_count >= float(chunk)
        reset = complete.to(h.dtype)
        cache["chunk_sum"] = (chunk_sum * (1.0 - reset)).detach() if not torch.is_grad_enabled() else chunk_sum * (1.0 - reset)
        cache["chunk_count"] = (chunk_count * (1.0 - reset)).detach() if not torch.is_grad_enabled() else chunk_count * (1.0 - reset)
        cache["med_h"] = med_tmp.detach() if not torch.is_grad_enabled() else med_tmp
        cache["slow_h"] = slow_tmp.detach() if not torch.is_grad_enabled() else slow_tmp
        prev_med = cache.get("prev_med", torch.zeros_like(h[:, 0]))
        cache["prev_med"] = med_seq[:, 0].detach() if not torch.is_grad_enabled() else med_seq[:, 0]
        consistency = (med_seq[:, 0].float() - prev_med.to(h.device).float()).pow(2).mean()
        return h + float(self.cfg.slow_state_strength) * delta.to(h.dtype), {"timescale_consistency": consistency, "slow_state_norm": slow_seq.float().pow(2).mean().detach(), "timescale_causal": h.new_tensor(1.0)}

class RiskSignalFusion(nn.Module):
    """Fuse risk, divergence, and model-state signals into a calibrated control field."""
    def __init__(self, cfg: NeedConfig):
        super().__init__()
        self.net = nn.Sequential(RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend), nn.Linear(cfg.d_model, max(16, cfg.d_model // 2)), nn.SiLU(), nn.Linear(max(16, cfg.d_model // 2), 4))

    def forward(self, h: torch.Tensor, aux_score_risk: torch.Tensor, divergence: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raw = self.net(h.float()).to(h.dtype)
        base_unc = torch.sigmoid(raw[..., 0:1])
        risk = torch.sigmoid(raw[..., 1:2] + aux_score_risk + divergence)
        search_need = torch.sigmoid(raw[..., 2:3] + divergence)
        output_need = torch.sigmoid(raw[..., 3:4] + 0.5 * risk + 0.5 * divergence)
        fused = (base_unc + risk + search_need) / 3.0
        return fused, {"risk_signal_mean": fused.mean(), "search_need_mean": search_need.mean().detach(), "output_mode_need_mean": output_need.mean().detach()}


class LatentDivergenceScore(nn.Module):
    """Score mismatch between the current token state and reusable latent-slot context."""
    def __init__(self, cfg: NeedConfig):
        super().__init__()
        self.net = nn.Sequential(RMSNorm(cfg.d_model * 2, kernel_backend=cfg.kernel_backend), nn.Linear(cfg.d_model * 2, cfg.d_model), nn.SiLU(), nn.Linear(cfg.d_model, 1))

    def forward(self, h: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        if slots.ndim == 4:
            # Causal pooled-slot mode: slots are [B,T,S,D], so each token sees
            # only the slot context built from its own prefix.
            slot_ctx = slots.mean(dim=2)
            if slot_ctx.size(1) != h.size(1):
                slot_ctx = slot_ctx[:, -h.size(1):]
        else:
            if h.size(1) == 1:
                slot_ctx = slots.mean(dim=1, keepdim=True)
            else:
                # Dense attention slots are sequence-level and therefore non-causal.
                # Do not feed them into per-token logits; use a causal prefix summary
                # instead for causal conditioning.
                counts = torch.arange(1, h.size(1) + 1, device=h.device, dtype=h.dtype).view(1, -1, 1)
                slot_ctx = h.cumsum(dim=1) / counts
        x = torch.cat([h, slot_ctx.to(h.dtype)], dim=-1)
        return torch.sigmoid(self.net(x.float()).to(h.dtype))


class OutputModeClassifier(nn.Module):
    """Learn when language scaffolding is useful and how much to use."""
    def __init__(self, cfg: NeedConfig):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend), nn.Linear(cfg.d_model, cfg.d_model), nn.SiLU(), nn.Linear(cfg.d_model, max(2, int(cfg.output_modes))))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Sequence-level classifier uses the final causal state.
        return self.net(h[:, -1].float())


class ObjectProgramHead(nn.Module):
    """Infer coarse object/layout programs for image generation and grounding."""
    def __init__(self, cfg: NeedConfig):
        super().__init__()
        self.cfg = cfg
        self.slots = max(1, int(cfg.object_program_slots))
        self.block = SlotAttentionBlock(cfg, self.slots)
        self.coord = nn.Linear(cfg.d_model, 4)
        self.presence = nn.Linear(cfg.d_model, 1)

    def forward(self, h: torch.Tensor, text_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raw_slots, _, aux = self.block(h, mask=text_mask, need_token_context=False)
        # SlotAttentionBlock's default pooled mode returns a causal slot sequence
        # [B,T,S,D].  The object program is a sequence-level layout scaffold, so
        # use the final prefix slots here.  This keeps downstream object losses
        # and image-token guidance on the expected [B,S,D] contract.
        slots = raw_slots[:, -1] if raw_slots.ndim == 4 else raw_slots
        coords = torch.sigmoid(self.coord(slots.float())).to(h.dtype)
        presence = torch.sigmoid(self.presence(slots.float())).to(h.dtype)
        layout_area = ((coords[..., 2:] - coords[..., :2]).abs().mean())
        return slots, {
            "object_presence": presence.mean(),
            "object_coverage": aux["slot_attention_coverage"],
            "object_slot_entropy": aux["slot_attention_entropy"],
            "object_layout_area": layout_area.detach(),
        }


class AuxScoreHead(nn.Module):
    def __init__(self, cfg: NeedConfig):
        super().__init__()
        self.net = nn.Sequential(RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend), nn.Linear(cfg.d_model, cfg.d_model), nn.SiLU(), nn.Linear(cfg.d_model, 9))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # [0] quality_logit, [1] risk, [2] difficulty/revision, [3] contradiction,
        # [4] repetition_risk, [5:9] controller actions: answer/deepen/retrieve/revise.
        return self.net(h)


class CooperativeStepWorkspace(nn.Module):
    """Compact cross-step workspace for the internal stages of a NEEDBlock.

    Each stage publishes the residual delta it actually contributed. Later stages
    read those compact summaries before proposing their own update, and a final
    mixer can reconcile all stage summaries once the block has run. The resulting
    block is still linear in sequence length: the only all-to-all operation is over
    the fixed number of internal stages, not over tokens.
    """
    def __init__(self, cfg: NeedConfig, step_names: Sequence[str]):
        super().__init__()
        self.cfg = cfg
        self.step_names = tuple(str(x) for x in step_names)
        self.n_steps = len(self.step_names)
        d = int(cfg.d_model)
        sd = int(max(8, min(d, getattr(cfg, "cooperative_step_summary_dim", 0) or max(16, min(128, d // 4)))))
        self.summary_dim = sd
        self.step_roles = nn.Parameter(torch.zeros(self.n_steps, sd))
        self.delta_to_summary = nn.Linear(d, sd, bias=False)
        self.context_queries = nn.ModuleList([nn.Linear(d, sd, bias=False) for _ in range(self.n_steps)])
        self.context_out = nn.Linear(sd, d, bias=False)
        self.step_gate_proj = nn.ModuleList([nn.Linear(3 * d, 1) for _ in range(self.n_steps)])
        self.final_query = nn.Linear(d, sd, bias=False)
        self.final_out = nn.Linear(sd, d, bias=False)
        self.final_gate = nn.Linear(2 * d, 1)
        self._reset_cooperative_parameters()

    def _enabled(self) -> bool:
        return bool(getattr(self.cfg, "cooperative_steps", True))

    def _reset_cooperative_parameters(self) -> None:
        with torch.no_grad():
            self.step_roles.normal_(mean=0.0, std=0.02)
            bias = float(getattr(self.cfg, "cooperative_step_gate_bias_init", 2.0))
            for proj in self.step_gate_proj:
                if proj.bias is not None:
                    proj.bias.fill_(bias)
            if self.final_gate.bias is not None:
                self.final_gate.bias.zero_()

    def reset_gate_biases(self) -> None:
        self._reset_cooperative_parameters()

    def read_context(self, x: torch.Tensor, summaries: Sequence[torch.Tensor], step_idx: int) -> torch.Tensor:
        """Return a per-token context vector built from earlier stage summaries."""
        if (not self._enabled()) or len(summaries) == 0 or float(getattr(self.cfg, "cooperative_step_context_strength", 0.0)) == 0.0:
            return torch.zeros_like(x)
        idx = int(max(0, min(step_idx, self.n_steps - 1)))
        bank = torch.stack([s.float() for s in summaries], dim=2)  # [B,T,S_seen,C]
        q = F.normalize(self.context_queries[idx](x.float()), dim=-1)
        k = F.normalize(bank, dim=-1)
        scores = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(float(self.summary_dim))
        attn = F.softmax(scores, dim=2)
        ctx_summary = (attn.unsqueeze(-1) * bank).sum(dim=2)
        ctx = self.context_out(ctx_summary).to(dtype=x.dtype)
        return float(getattr(self.cfg, "cooperative_step_context_strength", 0.12)) * ctx

    def contribution_gate(self, x: torch.Tensor, delta: torch.Tensor, context: torch.Tensor, step_idx: int, base_gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Soft gate for whether this stage should spend residual capacity."""
        if not self._enabled():
            return base_gate.to(dtype=x.dtype) if torch.is_tensor(base_gate) else torch.ones(x.size(0), x.size(1), 1, device=x.device, dtype=x.dtype)
        idx = int(max(0, min(step_idx, self.n_steps - 1)))
        gate_in = torch.cat([x, delta, context], dim=-1).float()
        gate = torch.sigmoid(self.step_gate_proj[idx](gate_in)).to(dtype=x.dtype)
        if torch.is_tensor(base_gate):
            gate = gate * base_gate.to(device=x.device, dtype=x.dtype)
        return gate

    def summarize(self, actual_delta: torch.Tensor, step_idx: int, gate: torch.Tensor) -> torch.Tensor:
        """Publish the contribution from one stage into the block workspace."""
        idx = int(max(0, min(step_idx, self.n_steps - 1)))
        if not self._enabled():
            return actual_delta.new_zeros(actual_delta.size(0), actual_delta.size(1), self.summary_dim, dtype=torch.float32)
        role = self.step_roles[idx].view(1, 1, -1)
        summary = torch.tanh(self.delta_to_summary(actual_delta.float()) + role)
        return summary * gate.float().clamp(0.0, 1.0)

    def finish(
        self,
        x: torch.Tensor,
        actual_deltas: Sequence[torch.Tensor],
        gates: Sequence[torch.Tensor],
        summaries: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        aux: Dict[str, torch.Tensor] = {}
        if not self._enabled() or len(actual_deltas) == 0 or len(gates) == 0:
            return x, aux

        if len(summaries) > 0 and float(getattr(self.cfg, "cooperative_step_final_strength", 0.0)) != 0.0:
            bank = torch.stack([s.float() for s in summaries], dim=2)
            q = F.normalize(self.final_query(x.float()), dim=-1)
            k = F.normalize(bank, dim=-1)
            scores = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(float(self.summary_dim))
            attn = F.softmax(scores, dim=2)
            ctx_summary = (attn.unsqueeze(-1) * bank).sum(dim=2)
            final_delta = self.final_out(ctx_summary).to(dtype=x.dtype)
            final_gate = torch.sigmoid(self.final_gate(torch.cat([x, final_delta], dim=-1).float())).to(dtype=x.dtype)
            x = x + float(getattr(self.cfg, "cooperative_step_final_strength", 0.08)) * final_gate * final_delta
            aux["coop_final_gate"] = final_gate.float().mean().detach()
            aux["coop_final_attention_entropy"] = (-(attn.float() * attn.float().clamp_min(1e-8).log()).sum(dim=2).mean()).detach()

        gate_means = torch.stack([g.float().mean() for g in gates])
        contribs = torch.stack([d.float().pow(2).mean(dim=-1).sqrt().mean() for d in actual_deltas])
        aux["coop_step_gate_mean"] = gate_means.mean().detach()
        aux["coop_step_contribution"] = contribs.mean().detach()
        gate_probs = torch.cat([g.float().reshape(-1) for g in gates], dim=0).clamp(1e-6, 1.0 - 1e-6)
        aux["coop_step_gate_entropy"] = (-(gate_probs * gate_probs.log() + (1.0 - gate_probs) * (1.0 - gate_probs).log()).mean()).detach()

        if bool(getattr(self.cfg, "collect_aux_metrics", True)):
            for name, gate, contrib in zip(self.step_names, gates, contribs):
                aux[f"coop_{name}_gate"] = gate.float().mean().detach()
                aux[f"coop_{name}_contribution"] = contrib.detach()

        budget = gate_means.new_tensor(float(getattr(self.cfg, "cooperative_step_budget", 0.82)))
        mean_gate = gate_means.mean()
        aux["coop_gate_budget"] = F.relu(mean_gate - budget).pow(2) + 0.25 * F.relu(budget * 0.35 - mean_gate).pow(2)

        if len(actual_deltas) > 1:
            target = gate_means.new_tensor(float(getattr(self.cfg, "cooperative_step_redundancy_target", 0.15)))
            pieces: List[torch.Tensor] = []
            for i in range(len(actual_deltas)):
                ni = F.normalize(actual_deltas[i].float(), dim=-1)
                gi = gates[i].float().squeeze(-1)
                for j in range(i + 1, len(actual_deltas)):
                    nj = F.normalize(actual_deltas[j].float(), dim=-1)
                    gj = gates[j].float().squeeze(-1)
                    pair_gate = (gi * gj).detach()
                    cos = (ni * nj).sum(dim=-1).abs()
                    pieces.append((F.relu(cos - target).pow(2) * pair_gate).mean())
            aux["coop_step_redundancy"] = torch.stack(pieces).mean() if pieces else x.new_tensor(0.0)
        else:
            aux["coop_step_redundancy"] = x.new_tensor(0.0)
        return x, aux



class NEEDBlock(nn.Module):
    STEP_NAMES: Tuple[str, ...] = ("retention", "convolution", "memory", "equilibrium", "moe")

    def __init__(self, cfg: NeedConfig):
        super().__init__()
        self.cfg = cfg
        self.n1 = RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend)
        self.ret = StructuredDualRetention(cfg) if cfg.retention_impl == "ssd" else SelectiveRetention(cfg)
        self.n2 = RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend)
        self.conv = MultiScaleCausalConv(cfg.d_model, cfg.conv_kernel, cfg.n_conv_scales, cfg.dropout, cfg.conv_active_scales)
        self.n3 = RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend)
        self.mem = HierarchicalMemory(cfg)
        self.n4 = RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend)
        self.eq = AdaptiveEquilibrium(cfg)
        self.n5 = RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend)
        self.moe = SparseMoE(cfg)
        self.depth_gate = AdaptiveDepthGate(cfg)
        self.coop = CooperativeStepWorkspace(cfg, self.STEP_NAMES)
        self.scales = nn.Parameter(torch.ones(5) * cfg.layer_scale_init)

    def reset_cooperative_gates(self) -> None:
        if hasattr(self, "coop"):
            self.coop.reset_gate_biases()

    def _residual_delta(self, y: torch.Tensor, gate: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        # Keep the residual scale as a tensor.
        return (self.cfg.residual_scale * scale) * y * gate

    def _residual_add(self, x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x + (self.cfg.residual_scale * scale) * y

    def _zero_step_gate(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(x.size(0), x.size(1), 1)

    def _adaptive_depth_gate(
        self,
        gate: torch.Tensor,
        hard_skip: bool,
        threshold: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (masked_gate, active_fraction, active_mask) without host sync.

        The old path branched on a host-read gate statistic and compacted
        tokens with boolean indexing. This helper keeps tensor shapes unchanged:
        low-gate positions are zeroed by a tensor mask, while all modules still
        see the same static [B,T,D] shapes.
        """
        if hard_skip and bool(getattr(self.cfg, "adaptive_depth_static_masking", True)) and threshold > 0.0:
            active = (gate.detach() >= threshold).to(dtype=gate.dtype, device=gate.device)
            return gate * active, active.float().mean().detach(), active
        active = torch.ones_like(gate, dtype=gate.dtype, device=gate.device)
        return gate, active.float().mean().detach(), active

    def _record_coop_step(
        self,
        step_idx: int,
        actual_delta: torch.Tensor,
        gate: torch.Tensor,
        summaries: List[torch.Tensor],
        actual_deltas: List[torch.Tensor],
        step_gates: List[torch.Tensor],
    ) -> None:
        actual_deltas.append(actual_delta)
        step_gates.append(gate)
        summaries.append(self.coop.summarize(actual_delta, step_idx, gate))

    def _role_separation_strength(self) -> float:
        if not bool(getattr(self.cfg, "role_separation", True)):
            return 0.0
        return float(min(max(_safe_float(getattr(self.cfg, "role_separation_strength", 0.35), 0.35), 0.0), 1.0))

    def _separate_from(self, delta: torch.Tensor, basis: Optional[torch.Tensor], strength: Optional[float] = None) -> torch.Tensor:
        """Remove a soft projection of delta onto an earlier role basis.

        The basis is detached on purpose: retention/conv are treated as the
        temporal carrier, and later stages adapt around them instead of pushing
        them away. This keeps the operation O(B*T*D) and avoids another learned
        arbitration subnetwork.
        """
        if basis is None:
            return delta
        s = self._role_separation_strength() if strength is None else float(strength)
        if s <= 0.0:
            return delta
        bf = basis.detach().float()
        df = delta.float()
        denom = bf.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        proj = ((df * bf).sum(dim=-1, keepdim=True) / denom) * bf
        return (df - s * proj).to(dtype=delta.dtype)

    def _overlap_penalty(self, delta: torch.Tensor, basis: Optional[torch.Tensor], gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        if basis is None:
            return delta.new_tensor(0.0)
        dn = F.normalize(delta.float(), dim=-1)
        bn = F.normalize(basis.detach().float(), dim=-1)
        cos2 = (dn * bn).sum(dim=-1).pow(2)
        if torch.is_tensor(gate):
            cos2 = cos2 * gate.detach().float().squeeze(-1).clamp(0.0, 1.0)
        return cos2.mean()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        aux: Dict[str, torch.Tensor] = {}
        summaries: List[torch.Tensor] = []
        actual_deltas: List[torch.Tensor] = []
        step_gates: List[torch.Tensor] = []

        # 0) Retention publishes the first state-space update. It is now also
        # contribution-gated, so even the always-on paths have to earn residual
        # capacity under the cooperative budget.
        ctx = self.coop.read_context(x, summaries, 0)
        y = self.ret(self.n1(x + ctx))
        coop_gate = self.coop.contribution_gate(x, y, ctx, 0)
        actual = self._residual_delta(y, coop_gate, self.scales[0])
        x = x + actual
        self._record_coop_step(0, actual, coop_gate, summaries, actual_deltas, step_gates)

        # 1) Causal convolution reads the retention summary rather than blindly
        # applying another local transform.
        ctx = self.coop.read_context(x, summaries, 1)
        y, caux = self.conv(self.n2(x + ctx)); aux.update(caux)
        coop_gate = self.coop.contribution_gate(x, y, ctx, 1)
        actual = self._residual_delta(y, coop_gate, self.scales[1])
        x = x + actual
        self._record_coop_step(1, actual, coop_gate, summaries, actual_deltas, step_gates)

        gates, gaux = self.depth_gate(x); aux.update(gaux)
        mem_gate, eq_gate, moe_gate = gates[..., 0:1], gates[..., 1:2], gates[..., 2:3]
        hard_skip = bool(getattr(self.cfg, "adaptive_depth_hard_skip", True)) and not torch.is_grad_enabled()
        threshold = float(getattr(self.cfg, "adaptive_depth_skip_threshold", 0.0))

        # 2) Explicit memory is conditioned on retention+conv. Hard-skip is now
        # a static tensor mask: no mean().cpu() decision and no data-dependent
        # compaction. Inactive tokens do not write to the associative memory.
        temporal_delta = (actual_deltas[0] + actual_deltas[1]) if len(actual_deltas) >= 2 else torch.zeros_like(x)
        memory_actual = torch.zeros_like(x)
        memory_enabled = float(getattr(self.cfg, "memory_mix", 0.35)) > 0.0
        mem_base_gate, mem_active_fraction, mem_active_mask = self._adaptive_depth_gate(mem_gate, hard_skip, threshold)
        if memory_enabled:
            ctx = self.coop.read_context(x, summaries, 2)
            mem_input = self.n3(x + ctx)
            mem_innovation = self._separate_from(mem_input, temporal_delta)
            y, maux = self.mem(
                mem_input,
                condition=temporal_delta,
                innovation=mem_innovation,
                write_mask=mem_active_mask,
            )
            aux.update(maux)
            y = self._separate_from(y, temporal_delta)
            aux["memory_retention_overlap"] = self._overlap_penalty(y, temporal_delta, mem_base_gate)
            eff_gate = self.coop.contribution_gate(x, y, ctx, 2, mem_base_gate)
            mem_scale = self.scales[2] * x.new_tensor(float(getattr(self.cfg, "memory_mix", 0.35)))
            actual = self._residual_delta(y, eff_gate, mem_scale)
            memory_actual = actual
            x = x + actual
            self._record_coop_step(2, actual, eff_gate, summaries, actual_deltas, step_gates)
            aux["memory_hard_skipped"] = (1.0 - mem_active_fraction).detach()
        else:
            mem_active_fraction = x.new_tensor(0.0)
            aux["memory_hard_skipped"] = x.new_tensor(1.0)
            aux["memory_retention_overlap"] = x.new_tensor(0.0)
            zgate = self._zero_step_gate(x)
            zdelta = torch.zeros_like(x)
            self._record_coop_step(2, zdelta, zgate, summaries, actual_deltas, step_gates)

        # 3) Equilibrium is constrained to current-position cleanup. It runs at
        # static shape, then the thresholded gate decides which positions receive
        # residual capacity.
        eq_base_gate, eq_active_fraction, _ = self._adaptive_depth_gate(eq_gate, hard_skip, threshold)
        ctx = self.coop.read_context(x, summaries, 3)
        eq_context = x + ctx
        n4x = self.n4(eq_context)
        z, resid, energy, effort = self.eq(n4x, eq_context)
        delta = z - n4x
        eq_basis = temporal_delta + memory_actual
        delta = self._separate_from(delta, eq_basis)
        aux["equilibrium_temporal_overlap"] = self._overlap_penalty(delta, eq_basis, eq_base_gate)
        aux["equilibrium_active_fraction"] = eq_active_fraction
        aux["equilibrium_hard_skipped"] = (1.0 - eq_active_fraction).detach()
        eff_gate = self.coop.contribution_gate(x, delta, ctx, 3, eq_base_gate)
        actual = self._residual_delta(delta, eff_gate, self.scales[3])
        x = x + actual
        self._record_coop_step(3, actual, eff_gate, summaries, actual_deltas, step_gates)

        # 4) Sparse expert routing stays static at the block boundary. Inside the
        # MoE uses sparse expert dispatch unless static dispatch is requested.
        moe_base_gate, moe_active_fraction, _ = self._adaptive_depth_gate(moe_gate, hard_skip, threshold)
        ctx = self.coop.read_context(x, summaries, 4)
        n5x = self.n5(x + ctx)
        y, eaux = self.moe(n5x)
        aux.update(eaux)
        aux["moe_active_fraction"] = moe_active_fraction
        aux["moe_hard_skipped"] = (1.0 - moe_active_fraction).detach()
        eff_gate = self.coop.contribution_gate(x, y, ctx, 4, moe_base_gate)
        actual = self._residual_delta(y, eff_gate, self.scales[4])
        x = x + actual
        self._record_coop_step(4, actual, eff_gate, summaries, actual_deltas, step_gates)

        x, coop_aux = self.coop.finish(x, actual_deltas, step_gates, summaries)
        aux.update(coop_aux)
        aux["adaptive_depth_mem_run"] = mem_active_fraction.detach()
        aux["adaptive_depth_eq_run"] = eq_active_fraction.detach()
        aux["adaptive_depth_moe_run"] = moe_active_fraction.detach()
        aux["equilibrium_residual"] = resid
        aux["energy"] = energy
        aux["adaptive_effort"] = effort
        aux["energy_row_orth"] = self.eq.energy.row_orth_loss()
        return x, aux

    def init_stream_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
        return {
            "ret": self.ret.init_stream_state(batch_size, device, dtype) if hasattr(self.ret, "init_stream_state") else {},
            "conv": self.conv.init_stream_state(batch_size, device, dtype),
            "mem": self.mem.init_stream_state(batch_size, device, dtype),
        }

    def stream_step(self, x: torch.Tensor, state: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """One-token NEED block update with recurrent/conv/memory caches."""
        if x.ndim != 3 or x.size(1) != 1:
            raise ValueError("stream_step expects x with shape [B,1,D]")
        aux: Dict[str, torch.Tensor] = {}
        summaries: List[torch.Tensor] = []
        actual_deltas: List[torch.Tensor] = []
        step_gates: List[torch.Tensor] = []
        if "ret" not in state:
            state.update(self.init_stream_state(x.size(0), x.device, x.dtype))

        ctx = self.coop.read_context(x, summaries, 0)
        if hasattr(self.ret, "stream_step"):
            r = self.ret.stream_step(self.n1(x + ctx), state["ret"])
        else:
            r = self.ret(self.n1(x + ctx))
        coop_gate = self.coop.contribution_gate(x, r, ctx, 0)
        actual = self._residual_delta(r, coop_gate, self.scales[0])
        x = x + actual
        self._record_coop_step(0, actual, coop_gate, summaries, actual_deltas, step_gates)

        ctx = self.coop.read_context(x, summaries, 1)
        y, caux = self.conv.stream_step(self.n2(x + ctx), state["conv"]); aux.update(caux)
        coop_gate = self.coop.contribution_gate(x, y, ctx, 1)
        actual = self._residual_delta(y, coop_gate, self.scales[1])
        x = x + actual
        self._record_coop_step(1, actual, coop_gate, summaries, actual_deltas, step_gates)

        gates, gaux = self.depth_gate(x); aux.update(gaux)
        mem_gate, eq_gate, moe_gate = gates[..., 0:1], gates[..., 1:2], gates[..., 2:3]
        hard_skip = bool(getattr(self.cfg, "adaptive_depth_hard_skip", True)) and not torch.is_grad_enabled()
        threshold = float(getattr(self.cfg, "adaptive_depth_skip_threshold", 0.0))

        temporal_delta = (actual_deltas[0] + actual_deltas[1]) if len(actual_deltas) >= 2 else torch.zeros_like(x)
        memory_actual = torch.zeros_like(x)
        memory_enabled = float(getattr(self.cfg, "memory_mix", 0.35)) > 0.0
        mem_base_gate, mem_active_fraction, mem_active_mask = self._adaptive_depth_gate(mem_gate, hard_skip, threshold)
        if memory_enabled:
            ctx = self.coop.read_context(x, summaries, 2)
            mem_input = self.n3(x + ctx)
            mem_innovation = self._separate_from(mem_input, temporal_delta)
            y, maux = self.mem.stream_step(
                mem_input,
                state["mem"],
                condition=temporal_delta,
                innovation=mem_innovation,
                write_mask=mem_active_mask,
            )
            aux.update(maux)
            y = self._separate_from(y, temporal_delta)
            aux["memory_retention_overlap"] = self._overlap_penalty(y, temporal_delta, mem_base_gate)
            eff_gate = self.coop.contribution_gate(x, y, ctx, 2, mem_base_gate)
            mem_scale = self.scales[2] * x.new_tensor(float(getattr(self.cfg, "memory_mix", 0.35)))
            actual = self._residual_delta(y, eff_gate, mem_scale)
            memory_actual = actual
            x = x + actual
            self._record_coop_step(2, actual, eff_gate, summaries, actual_deltas, step_gates)
            aux["memory_hard_skipped"] = (1.0 - mem_active_fraction).detach()
        else:
            mem_active_fraction = x.new_tensor(0.0)
            aux["memory_hard_skipped"] = x.new_tensor(1.0)
            aux["memory_retention_overlap"] = x.new_tensor(0.0)
            zgate = self._zero_step_gate(x)
            zdelta = torch.zeros_like(x)
            self._record_coop_step(2, zdelta, zgate, summaries, actual_deltas, step_gates)

        eq_base_gate, eq_active_fraction, _ = self._adaptive_depth_gate(eq_gate, hard_skip, threshold)
        ctx = self.coop.read_context(x, summaries, 3)
        eq_context = x + ctx
        n4x = self.n4(eq_context)
        z, resid, energy, effort = self.eq(n4x, eq_context)
        delta = z - n4x
        eq_basis = temporal_delta + memory_actual
        delta = self._separate_from(delta, eq_basis)
        aux["equilibrium_temporal_overlap"] = self._overlap_penalty(delta, eq_basis, eq_base_gate)
        aux["equilibrium_active_fraction"] = eq_active_fraction
        aux["equilibrium_hard_skipped"] = (1.0 - eq_active_fraction).detach()
        eff_gate = self.coop.contribution_gate(x, delta, ctx, 3, eq_base_gate)
        actual = self._residual_delta(delta, eff_gate, self.scales[3])
        x = x + actual
        self._record_coop_step(3, actual, eff_gate, summaries, actual_deltas, step_gates)

        moe_base_gate, moe_active_fraction, _ = self._adaptive_depth_gate(moe_gate, hard_skip, threshold)
        ctx = self.coop.read_context(x, summaries, 4)
        n5x = self.n5(x + ctx)
        y, eaux = self.moe(n5x)
        aux.update(eaux)
        aux["moe_active_fraction"] = moe_active_fraction
        aux["moe_hard_skipped"] = (1.0 - moe_active_fraction).detach()
        eff_gate = self.coop.contribution_gate(x, y, ctx, 4, moe_base_gate)
        actual = self._residual_delta(y, eff_gate, self.scales[4])
        x = x + actual
        self._record_coop_step(4, actual, eff_gate, summaries, actual_deltas, step_gates)

        x, coop_aux = self.coop.finish(x, actual_deltas, step_gates, summaries)
        aux.update(coop_aux)
        aux["adaptive_depth_mem_run"] = mem_active_fraction.detach()
        aux["adaptive_depth_eq_run"] = eq_active_fraction.detach()
        aux["adaptive_depth_moe_run"] = moe_active_fraction.detach()
        aux["equilibrium_residual"] = resid
        aux["energy"] = energy
        aux["adaptive_effort"] = effort
        aux["energy_row_orth"] = self.eq.energy.row_orth_loss()
        return x, aux



def _safe_path_step_norm(delta: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """RMS step length with a finite gradient at zero movement.

    Supports optional architectural paths that can become exact no-ops under
    ablations or hard gates.
    """
    return delta.float().pow(2).mean(dim=-1).clamp_min(float(eps)).sqrt()


def geodesic_path_loss(traj: Sequence[torch.Tensor], target: float) -> torch.Tensor:
    if len(traj) < 2:
        return traj[0].new_tensor(0.0)
    steps = [_safe_path_step_norm(b - a) for a, b in zip(traj[:-1], traj[1:])]
    lens = torch.stack(steps, dim=0)
    return (lens - target).pow(2).mean()


def path_straightness_loss(traj: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(traj) < 3:
        return traj[0].new_tensor(0.0)
    total = sum(_safe_path_step_norm(b - a) for a, b in zip(traj[:-1], traj[1:]))
    chord = _safe_path_step_norm(traj[-1] - traj[0]).clamp_min(1e-8)
    return ((total / chord) - 1.0).pow(2).mean()


def contractive_path_loss(traj: Sequence[torch.Tensor], kappa: float) -> torch.Tensor:
    if len(traj) < 3:
        return traj[0].new_tensor(0.0)
    deltas = [_safe_path_step_norm(b - a) for a, b in zip(traj[:-1], traj[1:])]
    losses = [F.relu(deltas[i + 1] - kappa * deltas[i]).pow(2).mean() for i in range(len(deltas) - 1)]
    return torch.stack(losses).mean() if losses else traj[0].new_tensor(0.0)





class ExactAssociativeRecall(nn.Module):
    """Learned, linear-bounded long-range associative recall.

    Recall still retrieves real earlier hidden/token value states from a bounded
    candidate set, so the token-time cost remains O(B*T*C*D_r) with C capped by
    config.  The shallow fixed token/bigram bias path has been replaced by a
    learned candidate scorer.  Candidate channels provide local, exact-token,
    exact-bigram, and logarithmic landmark proposals, but their usefulness is
    query-conditioned and learned rather than hard-coded by public bias constants
    or scale-specific candidate buckets.
    """

    SRC_LOCAL = 0
    SRC_TOKEN = 1
    SRC_BIGRAM = 2
    SRC_LANDMARK = 3
    NUM_SOURCES = 4

    def __init__(self, cfg: NeedConfig):
        super().__init__()
        self.cfg = cfg
        rd = int(max(8, min(cfg.d_model, cfg.exact_recall_dim)))
        self.recall_dim = rd
        self.norm = RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend)
        self.q = nn.Linear(cfg.d_model, rd, bias=False)
        self.k = nn.Linear(cfg.d_model, rd, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.token_key = nn.Embedding(cfg.vocab_size, rd)
        self.token_value = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.token_key_gate = nn.Sequential(RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend), nn.Linear(cfg.d_model, 1))
        self.token_value_gate = nn.Sequential(RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend), nn.Linear(cfg.d_model, 1))
        self.source_key = nn.Embedding(self.NUM_SOURCES, rd)
        scorer_hidden = max(8, rd // 2)
        self.candidate_scorer = nn.Sequential(nn.Linear(6, scorer_hidden), nn.SiLU(), nn.Linear(scorer_hidden, 1, bias=False))
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.gate = nn.Sequential(RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend), nn.Linear(cfg.d_model, 1))
        self.commit_gate = nn.Linear(cfg.d_model * 2, 1)

    def _zero_aux(self, ref: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = ref.new_tensor(0.0)
        return {
            "exact_recall_entropy": z,
            "exact_recall_peak": z,
            "exact_recall_gate": z,
            "exact_recall_source_entropy": z,
            "exact_recall_mean_age": z,
            "exact_recall_token_match_rate": z,
            "exact_recall_bigram_match_rate": z,
        }

    @staticmethod
    def _previous_key_positions(key: torch.Tensor, valid: torch.Tensor, n_prev: int) -> torch.Tensor:
        """Return previous same-key positions without scanning over B or T."""
        b, t = key.shape
        n = int(max(0, n_prev))
        if n <= 0 or t <= 1:
            return key.new_full((b, t, 0), -1, dtype=torch.long)
        device = key.device
        pos = torch.arange(t, device=device, dtype=torch.long).view(1, t).expand(b, t)
        flat_key = key.reshape(-1).to(torch.long)
        flat_pos = pos.reshape(-1)
        flat_valid = valid.reshape(-1).to(torch.bool)
        order = torch.argsort(flat_key * (t + 1) + flat_pos)
        sk = flat_key[order]
        sp = flat_pos[order]
        sv = flat_valid[order]
        out = key.new_full((b * t, n), -1, dtype=torch.long)
        for j in range(n):
            shift = j + 1
            pk = torch.empty_like(sk)
            pp = torch.empty_like(sp)
            pv = torch.empty_like(sv)
            pk[:shift] = -1
            pp[:shift] = -1
            pv[:shift] = False
            pk[shift:] = sk[:-shift]
            pp[shift:] = sp[:-shift]
            pv[shift:] = sv[:-shift]
            ok = sv & pv & (pk == sk) & (pp < sp)
            out[order, j] = torch.where(ok, pp, torch.full_like(pp, -1))
        return out.view(b, t, n)

    @staticmethod
    def _source_budgets(max_candidates: int) -> Tuple[int, int, int, int]:
        """Derive bounded source pools from one candidate budget.

        The public contract is a single max-candidate cap.  Internally we reserve
        enough room for recency, exact lexical anchors, and sparse landmarks, but
        scoring decides which source matters for a query.
        """
        max_c = int(max(1, max_candidates))
        if max_c == 1:
            return 1, 0, 0, 0
        local_n = max(1, int(math.ceil(max_c * 0.38)))
        token_n = max(1, int(math.ceil(max_c * 0.25))) if max_c >= 4 else 0
        bigram_n = max(1, int(math.ceil(max_c * 0.15))) if max_c >= 8 else 0
        landmark_n = max_c - local_n - token_n - bigram_n
        while landmark_n < 0:
            if local_n >= token_n and local_n > 1:
                local_n -= 1
            elif token_n > 1:
                token_n -= 1
            elif bigram_n > 1:
                bigram_n -= 1
            else:
                break
            landmark_n = max_c - local_n - token_n - bigram_n
        return local_n, token_n, bigram_n, max(0, landmark_n)

    @staticmethod
    def _dedupe_candidates(cand: torch.Tensor, source: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c = int(cand.size(-1))
        if c <= 1:
            return cand, source
        valid = cand >= 0
        order = torch.arange(c, device=cand.device, dtype=torch.long)
        earlier = order.view(*([1] * (cand.ndim - 1)), c, 1) > order.view(*([1] * (cand.ndim - 1)), 1, c)
        same = (cand.unsqueeze(-1) == cand.unsqueeze(-2)) & valid.unsqueeze(-1) & valid.unsqueeze(-2)
        dup = (same & earlier).any(dim=-1)
        cand = torch.where(dup, torch.full_like(cand, -1), cand)
        source = torch.where(dup, torch.zeros_like(source), source)
        return cand, source

    def _candidate_indices(self, ids: torch.Tensor, max_candidates: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build [B,T,C] candidate memory positions and source ids."""
        b, t = ids.shape
        device = ids.device
        if t <= 1:
            empty = ids.new_full((b, t, 0), -1, dtype=torch.long)
            return empty, empty
        max_c = int(max(1, max_candidates))
        local_n, token_n, bigram_n, landmark_n = self._source_budgets(max_c)
        pos = torch.arange(t, device=device, dtype=torch.long).view(1, t, 1).expand(b, t, 1)
        parts: List[torch.Tensor] = []
        sources: List[torch.Tensor] = []

        def add(part: torch.Tensor, source_id: int) -> None:
            if part.numel() == 0 or part.size(-1) == 0:
                return
            parts.append(part)
            sources.append(torch.full_like(part, int(source_id)))

        local_n = min(local_n, max_c)
        if local_n > 0:
            local_offsets = torch.arange(1, local_n + 1, device=device, dtype=torch.long)
            add(pos - local_offsets.view(1, 1, -1), self.SRC_LOCAL)

        remaining = max(0, max_c - sum(p.size(-1) for p in parts))
        token_n = min(token_n, remaining)
        vocab = int(max(1, getattr(self.cfg, "vocab_size", 1)))
        batch = torch.arange(b, device=device, dtype=torch.long).view(b, 1)
        safe_ids = ids.clamp(min=0, max=vocab - 1).to(torch.long)
        if token_n > 0:
            token_key = batch * vocab + safe_ids
            token_valid = (ids >= 0) & (ids < vocab)
            add(self._previous_key_positions(token_key, token_valid, token_n), self.SRC_TOKEN)

        remaining = max(0, max_c - sum(p.size(-1) for p in parts))
        bigram_n = min(bigram_n, remaining)
        if bigram_n > 0:
            prev = torch.cat([safe_ids.new_full((b, 1), int(getattr(self.cfg, "pad_id", 0))), safe_ids[:, :-1]], dim=1)
            pair_key = prev * vocab + safe_ids
            bigram_key = batch * (vocab * vocab) + pair_key
            time_valid = torch.arange(t, device=device).view(1, t) > 0
            bigram_valid = ((ids >= 0) & (ids < vocab) & time_valid)
            add(self._previous_key_positions(bigram_key, bigram_valid, bigram_n), self.SRC_BIGRAM)

        remaining = max(0, max_c - sum(p.size(-1) for p in parts))
        landmark_n = min(landmark_n, remaining)
        if landmark_n > 0 and t > local_n + 1:
            # Landmark candidates must be based on each query token's available
            # causal history, not on the final sequence length.  Otherwise
            # teacher-forced full forward and stateful streaming prefill expose
            # different candidate sets for the same prefix.
            query_pos = torch.arange(t, device=device, dtype=torch.long).view(1, t, 1)
            use_landmark = query_pos > int(local_n + 1)
            start = max(local_n + 1, 2)
            if landmark_n == 1:
                offsets = query_pos.clamp_min(1)
            else:
                frac = torch.linspace(0.0, 1.0, steps=landmark_n, device=device, dtype=torch.float32).view(1, 1, -1)
                log_start = math.log10(float(start))
                log_end = torch.log10(query_pos.to(torch.float32).clamp_min(float(start)))
                offsets = torch.pow(10.0, log_start + frac * (log_end - log_start)).round().to(torch.long)
                offsets = offsets.clamp(min=start)
            landmark = torch.where(use_landmark, pos - offsets, torch.full_like(offsets, -1))
            add(landmark, self.SRC_LANDMARK)

        cand = torch.cat(parts, dim=-1) if parts else ids.new_full((b, t, 0), -1, dtype=torch.long)
        source = torch.cat(sources, dim=-1) if sources else ids.new_full((b, t, 0), 0, dtype=torch.long)
        if cand.size(-1) > max_c:
            cand = cand[:, :, :max_c]
            source = source[:, :, :max_c]
        valid = (cand >= 0) & (cand < pos)
        cand = torch.where(valid, cand, torch.full_like(cand, -1))
        source = torch.where(valid, source, torch.zeros_like(source))
        return self._dedupe_candidates(cand, source)

    @staticmethod
    def _gather_sequence(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        c = indices.size(-1)
        if c == 0:
            return x.new_empty(b, t, 0, d)
        safe = indices.clamp(min=0, max=max(0, t - 1))
        batch_offsets = (torch.arange(b, device=x.device, dtype=torch.long).view(b, 1, 1) * t)
        flat_idx = (safe + batch_offsets).reshape(-1)
        return x.reshape(b * t, d).index_select(0, flat_idx).view(b, t, c, d)

    @staticmethod
    def _gather_ids(ids: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        b, t = ids.shape
        c = indices.size(-1)
        if c == 0:
            return ids.new_empty(b, t, 0)
        safe = indices.clamp(min=0, max=max(0, t - 1))
        batch_offsets = (torch.arange(b, device=ids.device, dtype=torch.long).view(b, 1, 1) * t)
        flat_idx = (safe + batch_offsets).reshape(-1)
        return ids.reshape(b * t).index_select(0, flat_idx).view(b, t, c)

    def _project_memory(self, hn: torch.Tensor, safe_ids: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        key_gate = torch.sigmoid(self.token_key_gate(hn.float()))
        value_gate = torch.sigmoid(self.token_value_gate(hn.float())).to(dtype)
        k = F.normalize(self.k(hn.float()) + key_gate * self.token_key(safe_ids).float(), dim=-1)
        v = self.v(hn).to(dtype) + value_gate * self.token_value(safe_ids).to(dtype)
        return k, v

    def _score_candidates(
        self,
        q: torch.Tensor,
        kc: torch.Tensor,
        cand: torch.Tensor,
        source: torch.Tensor,
        ids_q: torch.Tensor,
        ids_c: torch.Tensor,
        prev_q: torch.Tensor,
        prev_c: torch.Tensor,
        query_pos: torch.Tensor,
        valid: torch.Tensor,
        memory_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_sim = (q.unsqueeze(-2) * kc).sum(dim=-1)
        tok_match = ids_c == ids_q.unsqueeze(-1)
        bi_match = tok_match & (query_pos.unsqueeze(-1) > 0) & (cand > 0) & (prev_c == prev_q.unsqueeze(-1))
        age = (query_pos.unsqueeze(-1).to(torch.float32) - cand.to(torch.float32)).clamp_min(1.0)
        # Normalize age by the amount of causal history available to the query
        # position.  A single full-sequence denominator made training-time full
        # forward slightly disagree with streaming generation prefill.
        denom = query_pos.unsqueeze(-1).to(torch.float32).clamp_min(1.0)
        rel_age = (age / denom).clamp(0.0, 1.0)
        inv_age = (1.0 / (1.0 + age)).clamp(0.0, 1.0)
        log_age = torch.log1p(age) / torch.log1p(denom).clamp_min(1e-6)
        features = torch.stack([
            raw_sim.float(),
            tok_match.to(torch.float32),
            bi_match.to(torch.float32),
            rel_age,
            inv_age,
            log_age,
        ], dim=-1)
        learned_prior = self.candidate_scorer(features).squeeze(-1)
        src = source.clamp(min=0, max=self.NUM_SOURCES - 1).to(torch.long)
        src_vec = self.source_key(src).to(q.dtype)
        source_score = (q.unsqueeze(-2) * src_vec).sum(dim=-1).float() / math.sqrt(float(max(1, self.recall_dim)))
        scale = 1.0 / max(1e-4, float(self.cfg.exact_recall_temperature))
        scores = raw_sim.float() * scale + source_score + learned_prior
        scores = scores.masked_fill(~valid, -1e9)
        return scores, tok_match, bi_match, age, learned_prior

    def _source_entropy(self, source_sel: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if weights.numel() == 0:
            return weights.new_tensor(0.0)
        src = source_sel.clamp(min=0, max=self.NUM_SOURCES - 1).to(torch.long)
        one_hot = F.one_hot(src, num_classes=self.NUM_SOURCES).to(weights.dtype)
        mass = (one_hot * weights.unsqueeze(-1)).sum(dim=-2)
        probs = mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return -(probs.float() * probs.float().clamp_min(1e-8).log()).sum(dim=-1).mean()

    @staticmethod
    def _weighted_rate(mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if weights.numel() == 0:
            return weights.new_tensor(0.0)
        denom = weights.float().sum().clamp_min(1e-8)
        return (mask.to(torch.float32) * weights.float()).sum() / denom

    @staticmethod
    def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if weights.numel() == 0:
            return weights.new_tensor(0.0)
        denom = weights.float().sum().clamp_min(1e-8)
        return (values.float() * weights.float()).sum() / denom

    def forward(self, h: torch.Tensor, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.cfg.exact_recall or h.size(1) < 2:
            return h, self._zero_aux(h)
        b, t, d = h.shape
        max_t = min(t, int(self.cfg.exact_recall_max_tokens))
        if max_t < t:
            h_mem = h[:, -max_t:]
            ids_mem = input_ids[:, -max_t:]
        else:
            h_mem = h
            ids_mem = input_ids
        tm = h_mem.size(1)
        if tm < 2:
            return h, self._zero_aux(h)

        hn = self.norm(h_mem)
        safe_ids = ids_mem.clamp(min=0, max=self.cfg.vocab_size - 1)
        q = F.normalize(self.q(hn.float()), dim=-1)
        k, v = self._project_memory(hn, safe_ids, h.dtype)

        max_candidates = int(max(self.cfg.exact_recall_top_k, getattr(self.cfg, "exact_recall_max_candidates", 160)))
        cand, source = self._candidate_indices(ids_mem, max_candidates=max_candidates)
        c = cand.size(-1)
        if c == 0:
            recalled = h_mem.new_zeros(b, tm, d)
            ent = h.new_tensor(0.0)
            peak = h.new_tensor(0.0)
            src_ent = h.new_tensor(0.0)
            mean_age = h.new_tensor(0.0)
            tok_rate = h.new_tensor(0.0)
            bi_rate = h.new_tensor(0.0)
        else:
            valid = cand >= 0
            kc = self._gather_sequence(k, cand)
            ids_c = self._gather_ids(ids_mem, cand)
            prev_ids = torch.cat([ids_mem.new_full((b, 1), int(self.cfg.pad_id)), ids_mem[:, :-1]], dim=1)
            prev_c = self._gather_ids(prev_ids, cand)
            query_pos = torch.arange(tm, device=h.device, dtype=torch.long).view(1, tm).expand(b, tm)
            scores, tok_match, bi_match, age, _ = self._score_candidates(
                q, kc, cand, source, ids_mem, ids_c, prev_ids, prev_c, query_pos, valid, tm
            )

            kk = min(max(1, int(self.cfg.exact_recall_top_k)), c)
            vals, cand_rank = torch.topk(scores, k=kk, dim=-1)
            sel = cand.gather(2, cand_rank)
            valid_top = vals > -1e8
            weights = F.softmax(vals, dim=-1) * valid_top.to(vals.dtype)
            weights = (weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)).to(h.dtype)
            vg = self._gather_sequence(v, sel)
            recalled = (vg * weights.unsqueeze(-1)).sum(dim=2)
            probs = weights.float()
            ent = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
            peak = probs.max(dim=-1).values.mean()
            src_sel = source.gather(2, cand_rank)
            src_ent = self._source_entropy(src_sel, probs)
            tok_rate = self._weighted_rate(tok_match.gather(2, cand_rank), probs)
            bi_rate = self._weighted_rate(bi_match.gather(2, cand_rank), probs)
            mean_age = self._weighted_mean(age.gather(2, cand_rank), probs)

        if max_t < t:
            pad = h.new_zeros(b, t - max_t, d)
            recalled = torch.cat([pad, recalled], dim=1)
        gate_logits = self.gate(h.float()) + self.commit_gate(torch.cat([h.float(), recalled.float()], dim=-1))
        gate = torch.sigmoid(gate_logits).to(h.dtype) * float(self.cfg.exact_recall_mix)
        y = h + gate * self.out(recalled)
        return y, {
            "exact_recall_entropy": ent,
            "exact_recall_peak": peak.detach(),
            "exact_recall_gate": gate.float().mean().detach(),
            "exact_recall_source_entropy": src_ent.detach(),
            "exact_recall_mean_age": mean_age.detach(),
            "exact_recall_token_match_rate": tok_rate.detach(),
            "exact_recall_bigram_match_rate": bi_rate.detach(),
        }

    def init_stream_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
        rd = int(self.recall_dim)
        d = int(self.cfg.d_model)
        return {
            "k_hist": torch.empty(batch_size, 0, rd, device=device, dtype=torch.float32),
            "v_hist": torch.empty(batch_size, 0, d, device=device, dtype=dtype),
            "ids_hist": torch.empty(batch_size, 0, device=device, dtype=torch.long),
            "start_pos": 0,
            "next_pos": 0,
            "prev_ids": [None for _ in range(batch_size)],
            "token_pos": [dict() for _ in range(batch_size)],
            "bigram_pos": [dict() for _ in range(batch_size)],
        }

    @staticmethod
    def _stream_append_pos(store: Dict[int, List[int]], key: int, pos: int, keep: int) -> None:
        vals = store.get(key)
        if vals is None:
            vals = []
            store[key] = vals
        vals.append(int(pos))
        if len(vals) > keep:
            del vals[: len(vals) - keep]

    @staticmethod
    def _stream_prune_store(store: Dict[int, List[int]], start_pos: int) -> None:
        dead: List[int] = []
        for key, vals in store.items():
            j = 0
            while j < len(vals) and vals[j] < start_pos:
                j += 1
            if j:
                del vals[:j]
            if not vals:
                dead.append(key)
        for key in dead:
            del store[key]

    def _stream_candidate_abs(self, state: Dict[str, Any], current_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b = int(current_ids.numel())
        hist_len = int(state["ids_hist"].size(1))
        if hist_len <= 0:
            empty = current_ids.new_full((b, 0), -1)
            return empty, empty
        start_pos = int(state["start_pos"])
        current_pos = int(state["next_pos"])
        max_c = int(max(self.cfg.exact_recall_top_k, getattr(self.cfg, "exact_recall_max_candidates", 160)))
        local_n, token_n, bigram_n, landmark_n = self._source_budgets(max_c)
        rows: List[List[int]] = []
        src_rows: List[List[int]] = []
        for bi in range(b):
            seen: set = set()
            out: List[int] = []
            src: List[int] = []

            def add_pos(pos: int, source_id: int) -> None:
                if pos < start_pos or pos >= current_pos or pos in seen or len(out) >= max_c:
                    return
                seen.add(pos)
                out.append(pos)
                src.append(int(source_id))

            for pos in range(current_pos - 1, max(start_pos, current_pos - local_n) - 1, -1):
                add_pos(pos, self.SRC_LOCAL)
            tok = int(current_ids[bi].detach().cpu())
            if token_n > 0:
                for pos in reversed(state["token_pos"][bi].get(tok, [])[-token_n:]):
                    add_pos(pos, self.SRC_TOKEN)
            prev_tok = state["prev_ids"][bi]
            if bigram_n > 0 and prev_tok is not None:
                key = int(prev_tok) * int(max(1, self.cfg.vocab_size)) + tok
                for pos in reversed(state["bigram_pos"][bi].get(key, [])[-bigram_n:]):
                    add_pos(pos, self.SRC_BIGRAM)
            remaining = max_c - len(out)
            if landmark_n > 0 and remaining > 0 and hist_len > local_n + 1:
                g = min(landmark_n, remaining)
                start = max(local_n + 1, 2)
                if g == 1:
                    offsets = [hist_len]
                else:
                    vals = torch.logspace(math.log10(float(start)), math.log10(float(max(start, hist_len))), steps=g)
                    offsets = sorted({int(round(float(v))) for v in vals})
                for off in offsets:
                    add_pos(current_pos - int(off), self.SRC_LANDMARK)
            rows.append(out)
            src_rows.append(src)
        c = max((len(r) for r in rows), default=0)
        cand = current_ids.new_full((b, c), -1)
        source = current_ids.new_full((b, c), 0)
        for bi, row in enumerate(rows):
            if row:
                cand[bi, :len(row)] = torch.tensor(row, device=current_ids.device, dtype=torch.long)
                source[bi, :len(row)] = torch.tensor(src_rows[bi], device=current_ids.device, dtype=torch.long)
        return cand, source

    def _stream_append_current(self, state: Dict[str, Any], ids: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        b = ids.size(0)
        pos = int(state["next_pos"])
        max_t = int(max(1, self.cfg.exact_recall_max_tokens))
        state["k_hist"] = torch.cat([state["k_hist"].to(k.device), k.detach().to(torch.float32).unsqueeze(1)], dim=1)
        state["v_hist"] = torch.cat([state["v_hist"].to(v.device, dtype=v.dtype), v.detach().unsqueeze(1)], dim=1)
        state["ids_hist"] = torch.cat([state["ids_hist"].to(ids.device), ids.detach().unsqueeze(1)], dim=1)
        keep = int(max(1, getattr(self.cfg, "exact_recall_max_candidates", 160)))
        vocab = int(max(1, self.cfg.vocab_size))
        for bi in range(b):
            tok = int(ids[bi].detach().cpu())
            self._stream_append_pos(state["token_pos"][bi], tok, pos, keep)
            prev_tok = state["prev_ids"][bi]
            if prev_tok is not None:
                self._stream_append_pos(state["bigram_pos"][bi], int(prev_tok) * vocab + tok, pos, keep)
            state["prev_ids"][bi] = tok
        state["next_pos"] = pos + 1
        if state["ids_hist"].size(1) > max_t:
            trim = int(state["ids_hist"].size(1) - max_t)
            state["k_hist"] = state["k_hist"][:, trim:]
            state["v_hist"] = state["v_hist"][:, trim:]
            state["ids_hist"] = state["ids_hist"][:, trim:]
            state["start_pos"] = int(state["start_pos"]) + trim
            for bi in range(b):
                self._stream_prune_store(state["token_pos"][bi], int(state["start_pos"]))
                self._stream_prune_store(state["bigram_pos"][bi], int(state["start_pos"]))

    def stream_step(self, h: torch.Tensor, input_ids: torch.Tensor, state: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """One-token bounded learned recall using cached candidate indices."""
        if h.ndim != 3 or h.size(1) != 1:
            raise ValueError("stream_step expects h with shape [B,1,D]")
        ids = input_ids[:, -1].to(torch.long) if input_ids.ndim == 2 else input_ids.to(torch.long).view(-1)
        b, _, d = h.shape
        if (not self.cfg.exact_recall) or b == 0:
            return h, self._zero_aux(h)
        if "k_hist" not in state or state["k_hist"].size(0) != b or state["k_hist"].device != h.device:
            state.update(self.init_stream_state(b, h.device, h.dtype))
        hn = self.norm(h)
        safe_ids = ids.clamp(min=0, max=self.cfg.vocab_size - 1)
        q = F.normalize(self.q(hn.float()), dim=-1)[:, 0]
        k_full, v_full = self._project_memory(hn, safe_ids.view(b, 1), h.dtype)
        k = k_full[:, 0]
        v = v_full[:, 0]

        hist_len = int(state["ids_hist"].size(1))
        if hist_len <= 0:
            recalled = h.new_zeros(b, d)
            ent = h.new_tensor(0.0)
            peak = h.new_tensor(0.0)
            src_ent = h.new_tensor(0.0)
            mean_age = h.new_tensor(0.0)
            tok_rate = h.new_tensor(0.0)
            bi_rate = h.new_tensor(0.0)
        else:
            cand_abs, source = self._stream_candidate_abs(state, ids)
            c = cand_abs.size(1)
            if c == 0:
                recalled = h.new_zeros(b, d)
                ent = h.new_tensor(0.0)
                peak = h.new_tensor(0.0)
                src_ent = h.new_tensor(0.0)
                mean_age = h.new_tensor(0.0)
                tok_rate = h.new_tensor(0.0)
                bi_rate = h.new_tensor(0.0)
            else:
                rel = cand_abs - int(state["start_pos"])
                valid = (rel >= 0) & (rel < hist_len)
                safe = rel.clamp(min=0, max=max(0, hist_len - 1))
                batch_offsets = torch.arange(b, device=h.device, dtype=torch.long).view(b, 1) * hist_len
                flat = (safe + batch_offsets).reshape(-1)
                k_hist = state["k_hist"].to(h.device)
                v_hist = state["v_hist"].to(h.device, dtype=h.dtype)
                ids_hist = state["ids_hist"].to(h.device)
                kc = k_hist.reshape(b * hist_len, -1).index_select(0, flat).view(b, c, -1)
                ids_c = ids_hist.reshape(b * hist_len).index_select(0, flat).view(b, c)
                prev_rel = rel - 1
                prev_safe = prev_rel.clamp(min=0, max=max(0, hist_len - 1))
                prev_flat = (prev_safe + batch_offsets).reshape(-1)
                prev_c = ids_hist.reshape(b * hist_len).index_select(0, prev_flat).view(b, c)
                prev_q = torch.tensor([int(x) if x is not None else int(self.cfg.pad_id) for x in state["prev_ids"]], device=h.device, dtype=torch.long)
                query_pos = torch.full((b,), hist_len, device=h.device, dtype=torch.long)
                scores, tok_match, bi_match, age, _ = self._score_candidates(
                    q, kc, rel, source, ids, ids_c, prev_q, prev_c, query_pos, valid, hist_len
                )
                kk = min(max(1, int(self.cfg.exact_recall_top_k)), c)
                vals, rank = torch.topk(scores, k=kk, dim=-1)
                valid_top = vals > -1e8
                weights = F.softmax(vals, dim=-1) * valid_top.to(vals.dtype)
                weights = (weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)).to(h.dtype)
                sel = safe.gather(1, rank)
                sel_flat = (sel + batch_offsets).reshape(-1)
                vg = v_hist.reshape(b * hist_len, d).index_select(0, sel_flat).view(b, kk, d)
                recalled = (vg * weights.unsqueeze(-1)).sum(dim=1)
                probs = weights.float()
                ent = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
                peak = probs.max(dim=-1).values.mean()
                src_ent = self._source_entropy(source.gather(1, rank), probs)
                tok_rate = self._weighted_rate(tok_match.gather(1, rank), probs)
                bi_rate = self._weighted_rate(bi_match.gather(1, rank), probs)
                mean_age = self._weighted_mean(age.gather(1, rank), probs)
        gate_logits = self.gate(h.float()) + self.commit_gate(torch.cat([h.float(), recalled.unsqueeze(1).float()], dim=-1))
        gate = torch.sigmoid(gate_logits).to(h.dtype) * float(self.cfg.exact_recall_mix)
        y = h + gate * self.out(recalled.unsqueeze(1))
        self._stream_append_current(state, ids, k, v)
        return y, {
            "exact_recall_entropy": ent,
            "exact_recall_peak": peak.detach(),
            "exact_recall_gate": gate.float().mean().detach(),
            "exact_recall_source_entropy": src_ent.detach(),
            "exact_recall_mean_age": mean_age.detach(),
            "exact_recall_token_match_rate": tok_rate.detach(),
            "exact_recall_bigram_match_rate": bi_rate.detach(),
        }


class StateSpaceDriftStabilizer(nn.Module):
    """Small correction and losses to limit long-context recurrent drift.

    It does not force hidden states to be static. Instead it softly anchors the
    final state to the input/position/modality stream, normalizes chunk drift to a
    target band, and reports drift metrics so long-context histories do not slowly
    wander away from useful state-space geometry.
    """
    def __init__(self, cfg: NeedConfig):
        super().__init__()
        self.cfg = cfg
        self.anchor = nn.Sequential(RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend), nn.Linear(cfg.d_model, cfg.d_model), nn.SiLU(), nn.Linear(cfg.d_model, cfg.d_model, bias=False))
        self.gate = nn.Sequential(RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend), nn.Linear(cfg.d_model, 1))

    def forward(self, h: torch.Tensor, base: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.cfg.state_stabilization:
            z = h.new_tensor(0.0)
            return h, {"state_norm_error": z, "state_chunk_drift": z, "state_anchor_error": z, "state_correction_gate": z}
        anchor = self.anchor(base.float()).to(h.dtype)
        gate = torch.sigmoid(self.gate(h.float())).to(h.dtype) * float(self.cfg.state_anchor_strength)
        y = h + gate * (anchor - h)
        rms = _safe_path_step_norm(y)
        norm_err = (rms - float(self.cfg.state_norm_target)).pow(2).mean()
        chunk = max(4, int(self.cfg.state_drift_chunk))
        b, t, d = y.shape
        pad = (chunk - (t % chunk)) % chunk
        yp = F.pad(y.float(), (0, 0, 0, pad)) if pad else y.float()
        n_chunks = yp.size(1) // chunk
        chunks = yp.view(b, n_chunks, chunk, d).mean(dim=2)
        if n_chunks > 1:
            step = _safe_path_step_norm(chunks[:, 1:] - chunks[:, :-1])
            drift = F.relu(step - float(self.cfg.state_drift_target)).pow(2).mean()
        else:
            drift = y.new_tensor(0.0)
        anchor_err = (F.normalize(y.float(), dim=-1) - F.normalize(anchor.float(), dim=-1)).pow(2).mean()
        return y, {"state_norm_error": norm_err, "state_chunk_drift": drift, "state_anchor_error": anchor_err, "state_correction_gate": gate.float().mean().detach()}

    def stream_step(self, h: torch.Tensor, base: torch.Tensor, state: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.cfg.state_stabilization:
            z = h.new_tensor(0.0)
            return h, {"state_norm_error": z, "state_chunk_drift": z, "state_anchor_error": z, "state_correction_gate": z}
        anchor = self.anchor(base.float()).to(h.dtype)
        gate = torch.sigmoid(self.gate(h.float())).to(h.dtype) * float(self.cfg.state_anchor_strength)
        y = h + gate * (anchor - h)
        rms = _safe_path_step_norm(y)
        norm_err = (rms - float(self.cfg.state_norm_target)).pow(2).mean()
        anchor_err = (F.normalize(y.float(), dim=-1) - F.normalize(anchor.float(), dim=-1)).pow(2).mean()
        drift = h.new_tensor(0.0)
        if state is not None:
            prev = state.get("prev_chunk_mean")
            chunk = max(4, int(self.cfg.state_drift_chunk))
            count = int(state.get("chunk_count", 0)) + 1
            chunk_sum = state.get("chunk_sum")
            if chunk_sum is None or chunk_sum.size(0) != y.size(0) or chunk_sum.device != y.device:
                chunk_sum = torch.zeros(y.size(0), y.size(-1), device=y.device, dtype=y.dtype)
            chunk_sum = chunk_sum + y[:, 0].detach()
            if count >= chunk:
                mean = chunk_sum / float(count)
                if prev is not None:
                    step = _safe_path_step_norm(mean.float() - prev.float())
                    drift = F.relu(step - float(self.cfg.state_drift_target)).pow(2).mean()
                state["prev_chunk_mean"] = mean.detach()
                state["chunk_sum"] = torch.zeros_like(chunk_sum)
                state["chunk_count"] = 0
            else:
                state["chunk_sum"] = chunk_sum
                state["chunk_count"] = count
        return y, {"state_norm_error": norm_err, "state_chunk_drift": drift, "state_anchor_error": anchor_err, "state_correction_gate": gate.float().mean().detach()}

    def step(self, h: torch.Tensor, base: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.stream_step(h, base, None)


# -----------------------------
# NEED model
# -----------------------------

class NeedModel(nn.Module):
    def __init__(self, cfg: NeedConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.modality_emb = nn.Embedding(4, cfg.d_model)  # 0 text, 1 image, 2 summary, 3 mixed
        self.image_row_emb = nn.Embedding(max(1, cfg.image_max_grid), cfg.d_model)
        self.image_col_emb = nn.Embedding(max(1, cfg.image_max_grid), cfg.d_model)
        self.path_conditioner = TemporalPathwayConditioner(cfg)
        self.image_scan = Image2DSelectiveScan(cfg)
        self.reasoning_compressor = ReasoningCompressor(cfg)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([NEEDBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight
        self.mtp_projs = nn.ModuleList([nn.Linear(cfg.d_model, cfg.d_model, bias=False) for _ in range(max(0, cfg.n_predict_heads - 1))])
        self.dvsd_slot_router = nn.Sequential(
            RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend),
            nn.Linear(cfg.d_model, max(1, cfg.n_predict_heads)),
        )
        self.planner = LatentPlanner(cfg)
        self.aux_score = AuxScoreHead(cfg)
        self.revision_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.image_quality = nn.Sequential(RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend), nn.Linear(cfg.d_model, 1))
        self.text_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.image_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.mixture_energy_router = MixtureEnergyRouter(cfg)
        self.latent_slot_attention = LatentSlotAttention(cfg)
        self.timescales = HierarchicalTimeScales(cfg)
        self.risk_signal_fusion = RiskSignalFusion(cfg)
        self.latent_divergence = LatentDivergenceScore(cfg)
        self.output_mode_classifier = OutputModeClassifier(cfg)
        self.object_program = ObjectProgramHead(cfg)
        self.exact_recall = ExactAssociativeRecall(cfg)
        self.drift_stabilizer = StateSpaceDriftStabilizer(cfg)
        self.objective_balancer = AuxiliaryObjectiveBalancer(cfg, FUSED_OBJECTIVE_GROUPS.keys() if bool(getattr(cfg, "fused_aux_losses", True)) else OBJECTIVE_LOSS_GROUPS.keys())
        self.apply(self._init_weights)
        # Restore cooperative gate biases after global module initialization.
        # The gates start mostly open so existing checkpoints/train runs do not
        # suddenly lose half their residual path, then CE plus the cooperative
        # budget learns which stages can stay quiet.
        for block in self.blocks:
            if hasattr(block, "reset_cooperative_gates"):
                block.reset_cooperative_gates()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def modality_ids_from_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        image = ((ids >= self.cfg.image_token_offset) | (ids == self.cfg.img_bos_id) | (ids == self.cfg.img_eos_id) | (ids == self.cfg.img_mask_id)).long()
        return image.clamp(max=1)

    def image_coordinate_ids(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t = ids.shape
        max_grid = max(1, int(self.cfg.image_max_grid))
        mask, segment_ids = image_span_token_mask(ids, self.cfg)

        bos = ids == int(self.cfg.img_bos_id)
        running = mask.to(torch.long).cumsum(dim=1)
        segment_start_count = torch.where(bos, running, torch.zeros_like(running)).cummax(dim=1).values
        rank = (running - segment_start_count - 1).clamp_min(0)

        # Use configured grid size for causal coordinate assignment. Inferring
        # grid from completed span length would expose unavailable image-token count.
        grid_val = int(max(1, min(max_grid, getattr(self.cfg, "image_grid", max_grid))))
        grid = torch.full_like(rank, grid_val)

        row_vals = (rank // grid).clamp(min=0, max=max_grid - 1)
        col_vals = (rank % grid).clamp(min=0, max=max_grid - 1)
        rows = torch.where(mask, row_vals, torch.zeros_like(row_vals))
        cols = torch.where(mask, col_vals, torch.zeros_like(col_vals))
        return rows, cols, mask

    def encode_hidden(self, input_ids: torch.Tensor, modality_ids: Optional[torch.Tensor] = None, conditioning_vectors: Optional[torch.Tensor] = None, conditioning_scale: float = 0.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B,T]")
        b, t = input_ids.shape
        if t > self.cfg.block_size:
            raise ValueError(f"sequence length {t} exceeds block_size {self.cfg.block_size}")
        if modality_ids is None:
            modality_ids = self.modality_ids_from_tokens(input_ids)
        elif modality_ids.shape != input_ids.shape:
            raise ValueError(f"modality_ids shape {tuple(modality_ids.shape)} must match input_ids shape {tuple(input_ids.shape)}")
        pos = torch.arange(t, device=input_ids.device).view(1, t)
        x = self.token_emb(input_ids) + self.pos_emb(pos) + self.modality_emb(modality_ids.clamp(0, 3))
        if float(self.cfg.image_coord_scale) != 0.0:
            rows, cols, coord_mask = self.image_coordinate_ids(input_ids)
            coord = self.image_row_emb(rows.clamp_max(self.image_row_emb.num_embeddings - 1)) + self.image_col_emb(cols.clamp_max(self.image_col_emb.num_embeddings - 1))
            x = x + float(self.cfg.image_coord_scale) * coord * coord_mask.unsqueeze(-1).to(x.dtype)
        aux_lists: Dict[str, List[torch.Tensor]] = {}
        if conditioning_vectors is not None and conditioning_scale != 0.0:
            # Dual-channel reasoning handoff. The ordered vector sequence is preserved
            # with a temporal cross-path conditioner rather than being collapsed into
            # one global average. This keeps pathway order, endpoints, and direction.
            x, paux = self.path_conditioner(x, conditioning_vectors, conditioning_scale)
            for k, v in paux.items():
                aux_lists.setdefault(k, []).append(v)
        x = self.drop(x)
        traj: List[torch.Tensor] = [x]
        for block in self.blocks:
            x, aux = block(x)
            if self.cfg.image_2d_scan:
                x, iaux = self.image_scan(x, input_ids)
                aux.update(iaux)
            traj.append(x)
            for k, v in aux.items():
                aux_lists.setdefault(k, []).append(v)
        # Exact recall and drift correction are applied after the recurrent stack:
        # recall supplies an explicit long-range memory path, while the drift
        # stabilizer prevents long histories from wandering away from the input
        # anchor energy_router. Both are residual and softly gated.
        x, raux = self.exact_recall(x, input_ids)
        for k, v in raux.items():
            aux_lists.setdefault(k, []).append(v)
        x, saux = self.drift_stabilizer(x, traj[0])
        for k, v in saux.items():
            aux_lists.setdefault(k, []).append(v)
        traj.append(x)
        h = self.norm(x)
        aux_mean = {k: torch.stack(v).mean() for k, v in aux_lists.items()}
        return h, aux_mean, traj

    def _fused_aux_component_loss(
        self,
        group: str,
        parts: Sequence[Tuple[str, float, torch.Tensor]],
        aux: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Fuse related auxiliary terms into one normalized family objective.

        Each component is scalarized and optionally divided by its detached current
        magnitude before the family average.  This keeps, for example, CE-like and
        cosine/MSE-like components from fighting over scale while still producing
        only one optimizer-visible auxiliary loss for the family.
        """
        fused_terms: List[torch.Tensor] = []
        denom = 0.0
        floor = float(max(1e-8, getattr(self.cfg, "fused_aux_component_floor", 1e-3)))
        cap = float(max(1.0, getattr(self.cfg, "objective_term_abs_cap", 25.0)))
        ref: Optional[torch.Tensor] = None
        for name, weight, raw in parts:
            if not torch.is_tensor(raw) or float(weight) == 0.0:
                continue
            scalar = raw.float().mean() if raw.ndim > 0 else raw.float()
            safe = torch.nan_to_num(scalar, nan=0.0, posinf=cap, neginf=-cap)
            if bool(getattr(self.cfg, "fused_aux_component_normalize", True)):
                scale = safe.detach().abs().clamp_min(floor).clamp_max(cap)
                term = safe / scale
            else:
                scale = safe.new_tensor(1.0)
                term = safe
            fused_terms.append(float(weight) * term)
            denom += abs(float(weight))
            aux[f"{name}_fused_component_weight"] = safe.new_tensor(float(weight)).detach()
            aux[f"{name}_fused_component_norm"] = scale.detach()
            ref = safe
        if not fused_terms:
            if ref is not None:
                return ref.new_tensor(0.0)
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)
        fused = torch.stack(fused_terms).sum() / max(float(denom), floor)
        aux[f"{group}_aux_components"] = fused.new_tensor(float(len(fused_terms))).detach()
        aux[f"{group}_aux_component_weight_sum"] = fused.new_tensor(float(denom)).detach()
        return fused

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        modality_ids: Optional[torch.Tensor] = None,
        image_mask_positions: Optional[torch.Tensor] = None,
        image_targets: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
        conditioning_vectors: Optional[torch.Tensor] = None,
        conditioning_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        if targets is not None:
            if targets.ndim not in (2, 3) or tuple(targets.shape[:2]) != tuple(input_ids.shape):
                raise ValueError(
                    f"targets shape {tuple(targets.shape)} must be [B,T] or [B,T,H] with [B,T]={tuple(input_ids.shape)}"
                )
            if (
                self.training
                and bool(getattr(self.cfg, "strict_linear_core", True))
                and bool(getattr(self.cfg, "image_2d_bidirectional", False))
            ):
                raise RuntimeError(
                    "image_2d_bidirectional=True is non-causal and is disabled for strict-core training; "
                    "turn off bidirectional image scan or disable strict_linear_core for explicit non-causal ablations"
                )
        if image_mask_positions is not None and tuple(image_mask_positions.shape[:2]) != tuple(input_ids.shape):
            raise ValueError(f"image_mask_positions shape {tuple(image_mask_positions.shape)} must start with input_ids shape {tuple(input_ids.shape)}")
        if image_targets is not None and tuple(image_targets.shape[:2]) != tuple(input_ids.shape):
            raise ValueError(f"image_targets shape {tuple(image_targets.shape)} must start with input_ids shape {tuple(input_ids.shape)}")
        if self.training and self.cfg.token_dropout > 0.0:
            drop = torch.rand_like(input_ids.float()) < self.cfg.token_dropout
            special = input_ids < Special.byte_start
            input_ids = torch.where(drop & ~special, torch.full_like(input_ids, self.cfg.pad_id), input_ids)
        h, block_aux, traj = self.encode_hidden(input_ids, modality_ids, conditioning_vectors=conditioning_vectors, conditioning_scale=conditioning_scale)
        # Hierarchical time scales, latent slot attention, and mixture energy routing
        # operate after the recurrent backbone, before final decoding.
        h, taux = self.timescales(h)
        latent_slot_delta, latent_slots, gaux = self.latent_slot_attention(h)
        h = h + float(self.cfg.latent_slot_conditioning_scale) * latent_slot_delta
        router_delta, man_aux = self.mixture_energy_router(h)
        h = h + router_delta
        vf = self.aux_score(h)
        quality = vf[..., 0]
        risk = F.softplus(vf[..., 1])
        contradiction = torch.sigmoid(vf[..., 3])
        repetition = torch.sigmoid(vf[..., 4])
        latent_divergence = self.latent_divergence(h, latent_slots)
        risk_signal, uaux = self.risk_signal_fusion(h, risk.unsqueeze(-1).clamp(max=8.0) / 8.0, latent_divergence)
        revision_gate = torch.sigmoid(vf[..., 2:3] + float(self.cfg.risk_gate_strength) * risk_signal)
        h_dec = h + self.cfg.aux_score_logit_scale * revision_gate * self.revision_proj(h)
        logits = self.lm_head(h_dec)
        plan = self.planner(h_dec)
        aux: Dict[str, torch.Tensor] = dict(block_aux)
        aux.update(taux); aux.update(gaux); aux.update(man_aux); aux.update(uaux)
        output_mode_logits: Optional[torch.Tensor] = None
        # Keep loss-critical auxiliary tensors regardless of logging mode.  The
        # optional metrics below are useful for research diagnostics but cost extra
        # matmuls/softmaxes and can reduce MFU during long production pretraining.
        aux.update({
            "geodesic": geodesic_path_loss(traj, self.cfg.geodesic_target),
            "path_straightness": path_straightness_loss(traj),
            "path_contractive": contractive_path_loss(traj, self.cfg.contract_kappa),
            "latent_norm": h_dec.float().pow(2).mean(),
        })
        if bool(getattr(self.cfg, "collect_aux_metrics", True)):
            rcomp = self.reasoning_compressor(h_dec)
            output_mode_logits = self.output_mode_classifier(h_dec)
            vprob = F.softmax(output_mode_logits, dim=-1)
            aux.update({
                "faithfulness_mean": torch.sigmoid(rcomp["faithfulness"][..., 0]).detach().mean(),
                "usefulness_mean": torch.sigmoid(rcomp["faithfulness"][..., 1]).detach().mean(),
                "cot_contradiction_mean": torch.sigmoid(rcomp["faithfulness"][..., 2]).detach().mean(),
                "latent_divergence": latent_divergence.float().mean(),
                "output_mode_entropy": -(vprob * vprob.clamp_min(1e-8).log()).sum(dim=-1).mean(),
                "aux_score_risk_mean": risk.detach().mean(),
                "aux_score_quality": torch.sigmoid(quality.detach()).mean(),
                "contradiction_mean": contradiction.detach().mean(),
                "repetition_risk_mean": repetition.detach().mean(),
            })
        if return_hidden:
            aux["_hidden"] = h_dec
            if plan:
                aux["_planned_next"] = plan[0]
            aux["_latent_slots"] = latent_slots
            aux["_risk_signal"] = risk_signal
        loss: Optional[torch.Tensor] = None
        if targets is not None:
            if targets.ndim == 2:
                next_targets = targets
                future_targets = targets.unsqueeze(-1)
            elif targets.ndim == 3:
                next_targets = targets[..., 0]
                future_targets = targets
            else:
                raise ValueError("targets must have shape [B,T] or [B,T,H]")
            ce_all = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), next_targets.reshape(-1), ignore_index=self.cfg.pad_id, label_smoothing=self.cfg.label_smoothing, reduction="none").view_as(next_targets)
            mask = next_targets != self.cfg.pad_id
            mask_f = mask.float()
            denom = mask_f.sum().clamp_min(1.0)
            ce = (ce_all * mask_f).sum() / denom

            def _masked_average(values: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
                weights = token_mask.to(device=values.device, dtype=values.dtype)
                while weights.ndim < values.ndim:
                    weights = weights.unsqueeze(-1)
                return (values * weights).sum() / weights.sum().clamp_min(1.0)

            def _masked_mse(pred: torch.Tensor, target: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
                return _masked_average((pred.float() - target.float()).pow(2), token_mask)

            def _masked_bce_prob(pred: torch.Tensor, target: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
                per = F.binary_cross_entropy(pred.float().clamp(1e-5, 1.0 - 1e-5), target.float().clamp(0.0, 1.0), reduction="none")
                return _masked_average(per, token_mask)

            def _masked_bce_logits(pred: torch.Tensor, target: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
                per = F.binary_cross_entropy_with_logits(pred.float(), target.float(), reduction="none")
                return _masked_average(per, token_mask)

            def _masked_token_ce(pred_logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
                per = F.cross_entropy(
                    pred_logits.reshape(-1, pred_logits.size(-1)),
                    target_ids.reshape(-1),
                    ignore_index=self.cfg.pad_id,
                    label_smoothing=self.cfg.label_smoothing,
                    reduction="none",
                ).view_as(target_ids)
                valid = target_ids != self.cfg.pad_id
                return (per * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
            loss = ce
            aux["ce"] = ce.detach()
            aux["_ce_objective"] = ce
            self.objective_balancer.begin_batch()
            objective_group_used: Dict[str, torch.Tensor] = {"__total__": ce.new_tensor(0.0)}

            def _add_aux(base: torch.Tensor, lam: float, raw: torch.Tensor, name: str) -> torch.Tensor:
                return self.objective_balancer.add(base, ce, aux, lam, raw, name, objective_group_used)

            def _finish_objective_diagnostics() -> None:
                total_abs = objective_group_used.get("__total__", ce.new_tensor(0.0)).detach()
                ce_ref = ce.detach().float().abs().clamp_min(float(self.cfg.objective_softcap_min))
                aux["objective_aux_abs_ratio"] = (total_abs.float() / ce_ref).detach()
                signed = loss.detach().float() - ce.detach().float()
                aux["objective_aux_signed_ratio"] = (signed / ce_ref).detach()
                clipped_terms = [v.float() for k, v in aux.items() if k.endswith("_objective_clipped") and torch.is_tensor(v)]
                nonfinite_terms = [v.float() for k, v in aux.items() if k.endswith("_objective_nonfinite") and torch.is_tensor(v)]
                aux["objective_clipped_terms"] = torch.stack(clipped_terms).sum().detach() if clipped_terms else ce.new_tensor(0.0)
                aux["objective_nonfinite_terms"] = torch.stack(nonfinite_terms).sum().detach() if nonfinite_terms else ce.new_tensor(0.0)
                for group_name in self.objective_balancer.group_names:
                    if group_name in objective_group_used:
                        aux[f"objective_group_{group_name}_ratio"] = (objective_group_used[group_name].detach().float() / ce_ref).detach()
                qb = self.objective_balancer.quarantine_scale.detach().float()
                aux["objective_quarantined_terms"] = (qb < 0.999).float().sum().to(device=ce.device)
                aux["objective_min_quarantine_scale"] = qb.min().to(device=ce.device) if qb.numel() else ce.new_tensor(1.0)
                aux["objective_mean_quarantine_scale"] = qb.mean().to(device=ce.device) if qb.numel() else ce.new_tensor(1.0)
                aux["objective_step"] = self.objective_balancer.objective_step.detach().to(device=ce.device, dtype=ce.dtype)

            fused_aux_pending: Dict[str, List[Tuple[str, float, torch.Tensor]]] = {}

            def _aux_enabled(name: str) -> bool:
                return self.cfg.aux_component_enabled(name)

            def _queue_aux(name: str, raw: torch.Tensor) -> None:
                nonlocal loss
                aux[name] = raw.detach() if torch.is_tensor(raw) else raw
                weight = self.cfg.aux_component_weight(name)
                if not torch.is_tensor(raw) or float(weight) == 0.0:
                    return
                group = AUX_COMPONENT_GROUPS.get(name)
                if group is None:
                    return
                if bool(getattr(self.cfg, "fused_aux_losses", True)):
                    if self.cfg.aux_group_lambda(group) != 0.0:
                        fused_aux_pending.setdefault(group, []).append((name, float(weight), raw))
                else:
                    # Compatibility/debug mode: use component weights directly as the
                    # old-style per-term lambda surface.
                    loss = _add_aux(loss, float(weight), raw, name)

            def _flush_fused_auxes() -> None:
                nonlocal loss
                if not bool(getattr(self.cfg, "fused_aux_losses", True)):
                    return
                for group in AUX_FAMILY_NAMES:
                    parts = fused_aux_pending.get(group)
                    if not parts:
                        continue
                    lam = self.cfg.aux_group_lambda(group)
                    if lam == 0.0:
                        continue
                    fused_raw = self._fused_aux_component_loss(group, parts, aux)
                    loss = _add_aux(loss, lam, fused_raw, f"{group}_aux")

            # Multi-token / future-prediction family.  Formerly this was many
            # separate CE, KL, latent-distance, and router objectives.  They now
            # enter one prediction_aux bundle with per-component diagnostics.
            mtp = logits.new_tensor(0.0)
            max_heads = min(self.cfg.n_predict_heads, future_targets.size(-1))
            if _aux_enabled("mtp"):
                mtp_parts = []
                for i in range(1, max_heads):
                    pred = self.lm_head(self.mtp_projs[i - 1](h_dec))
                    target_i = future_targets[..., i]
                    mtp_parts.append(_masked_token_ce(pred, target_i))
                if mtp_parts:
                    mtp = torch.stack(mtp_parts).mean()
                    _queue_aux("mtp", mtp)
            aux["mtp"] = mtp.detach()

            dvsd_names = ("dvsd_slot_ce", "dvsd_consistency", "dvsd_router")
            if max_heads > 1 and any(_aux_enabled(n) for n in dvsd_names):
                dvsd_aux = self._dvsd_training_objectives(h_dec, logits, future_targets, next_targets, ce_all, mask)
                if _aux_enabled("dvsd_slot_ce"):
                    _queue_aux("dvsd_slot_ce", dvsd_aux["dvsd_slot_ce"])
                else:
                    aux["dvsd_slot_ce"] = dvsd_aux["dvsd_slot_ce"].detach()
                if _aux_enabled("dvsd_consistency"):
                    _queue_aux("dvsd_consistency", dvsd_aux["dvsd_consistency"])
                else:
                    aux["dvsd_consistency"] = dvsd_aux["dvsd_consistency"].detach()
                if _aux_enabled("dvsd_router"):
                    router_term = dvsd_aux["dvsd_router"] - float(self.cfg.dvsd_router_entropy_weight) * dvsd_aux["dvsd_router_entropy"]
                    _queue_aux("dvsd_router", router_term)
                    aux["dvsd_router_ce"] = dvsd_aux["dvsd_router"].detach()
                    aux["dvsd_router_entropy"] = dvsd_aux["dvsd_router_entropy"].detach()
                else:
                    aux["dvsd_router"] = dvsd_aux["dvsd_router"].detach()
                aux["dvsd_router_target_slots"] = dvsd_aux["dvsd_router_target_slots"].detach()
                aux["dvsd_router_pred_slots"] = dvsd_aux["dvsd_router_pred_slots"].detach()

            comp_names = ("dvsd_compound_latent", "dvsd_compound_ce", "dvsd_compound_consistency")
            if max_heads > 1 and bool(getattr(self.cfg, "dvsd_planner_compound_enabled", True)) and any(_aux_enabled(n) for n in comp_names):
                comp_aux = self._dvsd_compound_training_objectives(h_dec, logits, future_targets, next_targets, mask)
                for name in comp_names:
                    if _aux_enabled(name):
                        _queue_aux(name, comp_aux[name])
                    else:
                        aux[name] = comp_aux[name].detach()
                aux["dvsd_compound_steps"] = comp_aux["dvsd_compound_steps"].detach()

            plan_names = ("latent_planning", "planner_ce", "planning_consistency")
            if plan and any(_aux_enabled(n) for n in plan_names):
                lat_parts, pce_parts, cons_parts = [], [], []
                want_lat = _aux_enabled("latent_planning")
                want_pce = _aux_enabled("planner_ce")
                want_cons = _aux_enabled("planning_consistency")
                for horizon, pred_h in enumerate(plan, start=1):
                    if input_ids.size(1) <= horizon:
                        continue
                    pred_slice = pred_h[:, :-horizon]
                    if want_lat:
                        target_h = h_dec[:, horizon:].detach()
                        lat_parts.append(0.5 * ((pred_slice.float() - target_h.float()).pow(2).mean() + (1 - F.cosine_similarity(pred_slice.float(), target_h.float(), dim=-1).mean())))
                    if want_pce and future_targets.size(-1) >= horizon:
                        pred_logits = self.lm_head(pred_slice)
                        target_tok = future_targets[:, :-horizon, horizon - 1]
                        pce_parts.append(_masked_token_ce(pred_logits, target_tok))
                    if want_cons and horizon > 1:
                        prev = plan[horizon - 2][:, 1: input_ids.size(1) - horizon + 1]
                        cur = pred_slice
                        if prev.shape == cur.shape:
                            cons_parts.append((cur.float() - prev.detach().float()).pow(2).mean())
                if lat_parts:
                    lpl = torch.stack(lat_parts).mean(); _queue_aux("latent_planning", lpl)
                if pce_parts:
                    pcl = torch.stack(pce_parts).mean(); _queue_aux("planner_ce", pcl)
                if cons_parts:
                    csl = torch.stack(cons_parts).mean(); _queue_aux("planning_consistency", csl)

            if _aux_enabled("latent_slot") and h_dec.size(1) > 1:
                future_pool = h_dec[:, h_dec.size(1)//2:].detach().mean(dim=1) if h_dec.size(1) > 2 else h_dec[:, -1].detach()
                slot_pool = latent_slots[:, -1].mean(dim=1) if latent_slots.ndim == 4 else latent_slots.mean(dim=1)
                slot_loss = 0.5 * ((slot_pool.float() - future_pool.float()).pow(2).mean() + (1.0 - F.cosine_similarity(slot_pool.float(), future_pool.float(), dim=-1).mean()))
                _queue_aux("latent_slot", slot_loss)
            if _aux_enabled("latent_slot_diversity") and "latent_slot_diversity" in aux:
                _queue_aux("latent_slot_diversity", aux["latent_slot_diversity"])
            if _aux_enabled("risk_signal"):
                r_target = (ce_all.detach().clamp(max=8.0) / 8.0).unsqueeze(-1)
                rloss = _masked_mse(risk_signal, r_target, mask)
                _queue_aux("risk_signal", rloss)
            if _aux_enabled("latent_divergence_loss"):
                with torch.no_grad():
                    if latent_slots.ndim == 4:
                        slot_ctx = latent_slots.mean(dim=2)
                    else:
                        slot_ctx = latent_slots.mean(dim=1, keepdim=True).expand(-1, h_dec.size(1), -1)
                    align_err = (1.0 - F.cosine_similarity(slot_ctx.float(), h_dec.float(), dim=-1)).clamp(0, 2) / 2
                    dtarget = torch.maximum(align_err, (risk.detach().clamp(max=8.0) / 8.0)).unsqueeze(-1)
                dloss = _masked_bce_prob(latent_divergence, dtarget, mask)
                _queue_aux("latent_divergence_loss", dloss)
            if _aux_enabled("output_mode_classifier"):
                with torch.no_grad():
                    score = (risk_signal.mean(dim=1).squeeze(-1) + (risk.clamp(max=8.0)/8.0).mean(dim=1) + contradiction.mean(dim=1)) / 3.0
                    n_modes = max(2, int(self.cfg.output_modes))
                    target_mode = torch.clamp((score * n_modes).long(), 0, n_modes - 1)
                if output_mode_logits is None:
                    output_mode_logits = self.output_mode_classifier(h_dec)
                vploss = F.cross_entropy(output_mode_logits, target_mode)
                _queue_aux("output_mode_classifier", vploss)
            if _aux_enabled("mixture_energy_router_energy") and "mixture_energy_router_energy" in aux:
                _queue_aux("mixture_energy_router_energy", aux["mixture_energy_router_energy"])
            if _aux_enabled("timescale_consistency") and "timescale_consistency" in aux:
                _queue_aux("timescale_consistency", aux["timescale_consistency"])

            entropy_band_w = float(getattr(self.cfg, "objective_entropy_band_weight", 0.0))
            band_scale = entropy_band_w / 0.15 if entropy_band_w > 0.0 else 0.0
            if band_scale > 0.0:
                if "energy_route_entropy" in aux and self.cfg.energy_routes > 1 and _aux_enabled("energy_route_entropy_band"):
                    eband = AuxiliaryObjectiveBalancer.entropy_band_loss(aux["energy_route_entropy"], self.cfg.energy_routes, 0.20, 0.90)
                    _queue_aux("energy_route_entropy_band", eband * band_scale)
                if "latent_slot_attention_entropy" in aux and self.cfg.latent_slots > 1 and _aux_enabled("latent_slot_entropy_band"):
                    sband = AuxiliaryObjectiveBalancer.entropy_band_loss(aux["latent_slot_attention_entropy"], self.cfg.latent_slots, 0.25, 0.92)
                    _queue_aux("latent_slot_entropy_band", sband * band_scale)
                if output_mode_logits is not None and self.cfg.output_modes > 1 and _aux_enabled("output_mode_entropy_band"):
                    om_prob = F.softmax(output_mode_logits.float(), dim=-1)
                    om_ent = -(om_prob * om_prob.clamp_min(1e-8).log()).sum(dim=-1).mean()
                    aux["output_mode_entropy"] = om_ent.detach()
                    oband = AuxiliaryObjectiveBalancer.entropy_band_loss(om_ent, self.cfg.output_modes, 0.15, 0.88)
                    _queue_aux("output_mode_entropy_band", oband * band_scale)

            if _aux_enabled("diffusion"):
                noise = torch.randn_like(h_dec)
                sigma = torch.rand(h_dec.size(0), h_dec.size(1), 1, device=h_dec.device, dtype=h_dec.dtype) * 0.5
                denoised_logits = self.lm_head(h_dec + sigma * noise - sigma * noise.detach() * 0.25)
                diff = _masked_token_ce(denoised_logits, next_targets)
                _queue_aux("diffusion", diff)

            if image_mask_positions is not None and _aux_enabled("image_diffusion"):
                img_target = image_targets if image_targets is not None else next_targets
                if img_target.ndim == 3:
                    img_target = img_target[..., 0]
                img_mask = image_mask_positions.bool() & (img_target >= self.cfg.image_token_offset)
                img_logits = logits[..., self.cfg.image_token_offset:self.cfg.image_token_offset+self.cfg.image_codebook_size]
                img_labels = (img_target - self.cfg.image_token_offset).clamp(min=0, max=max(0, self.cfg.image_codebook_size - 1))
                img_ce_all = F.cross_entropy(img_logits.reshape(-1, img_logits.size(-1)), img_labels.reshape(-1), reduction="none").view_as(img_target)
                img_ce = (img_ce_all * img_mask.float()).sum() / img_mask.float().sum().clamp_min(1.0)
                _queue_aux("image_diffusion", img_ce)

            image_alignment_active = any(_aux_enabled(n) for n in (
                "image_contrastive", "image_local_contrastive", "region_word_alignment",
                "image_spatial_smoothness", "object_program", "object_slot_entropy_band",
            ))
            if image_alignment_active:
                mod = self.modality_ids_from_tokens(input_ids)
                text_mask = (mod == 0) & (input_ids >= Special.byte_start)
                img_mask_for_align = mod == 1
                has_text = bool(text_mask.any())
                has_img = bool(img_mask_for_align.any())
                if _aux_enabled("image_contrastive") and has_text and has_img:
                    text_vec = (h_dec * text_mask.unsqueeze(-1)).sum(dim=1) / text_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
                    img_vec = (h_dec * img_mask_for_align.unsqueeze(-1)).sum(dim=1) / img_mask_for_align.float().sum(dim=1, keepdim=True).clamp_min(1.0)
                    tx = F.normalize(self.text_proj(text_vec.float()), dim=-1)
                    ix = F.normalize(self.image_proj(img_vec.float()), dim=-1)
                    valid_pair = text_mask.any(dim=1) & img_mask_for_align.any(dim=1)
                    con = (1.0 - (tx[valid_pair] * ix[valid_pair]).sum(dim=-1)).mean() if bool(valid_pair.any()) else h_dec.new_tensor(0.0)
                    _queue_aux("image_contrastive", con)
                if _aux_enabled("image_local_contrastive") and has_text and has_img:
                    local = self.local_image_text_contrastive_loss(h_dec, input_ids, text_mask, img_mask_for_align)
                    _queue_aux("image_local_contrastive", local)
                if _aux_enabled("region_word_alignment") and has_text and has_img:
                    rwa = self.region_word_alignment_loss(h_dec, input_ids, text_mask, img_mask_for_align)
                    _queue_aux("region_word_alignment", rwa)
                if _aux_enabled("image_spatial_smoothness") and has_img:
                    smooth = self.image_spatial_smoothness_loss(h_dec, input_ids, img_mask_for_align)
                    _queue_aux("image_spatial_smoothness", smooth)
                if (_aux_enabled("object_program") or (_aux_enabled("object_slot_entropy_band") and band_scale > 0.0)) and has_img:
                    obj_slots, obj_aux = self.object_program(h_dec, text_mask=text_mask)
                    obj_norm = F.normalize(obj_slots.float(), dim=-1)
                    obj_gram = torch.einsum('bsd,bqd->bsq', obj_norm, obj_norm)
                    eye = torch.eye(obj_gram.size(-1), device=obj_gram.device, dtype=obj_gram.dtype).unsqueeze(0)
                    obj_loss = (obj_gram - eye).pow(2).mean() + 0.1 * (1.0 - obj_aux["object_coverage"].float())
                    if _aux_enabled("object_program"):
                        _queue_aux("object_program", obj_loss)
                    for ok, ov in obj_aux.items():
                        aux[ok] = ov.detach() if torch.is_tensor(ov) else ov
                    if band_scale > 0.0 and "object_slot_entropy" in aux and _aux_enabled("object_slot_entropy_band"):
                        oband = AuxiliaryObjectiveBalancer.entropy_band_loss(aux["object_slot_entropy"], self.cfg.object_program_slots, 0.20, 0.90)
                        _queue_aux("object_slot_entropy_band", oband * band_scale)

            aux_score_active = _aux_enabled("aux_score")
            controller_active = _aux_enabled("controller") and self.cfg.aux_score_controller and vf.size(-1) >= 9
            if aux_score_active or controller_active:
                with torch.no_grad():
                    pred = logits.argmax(dim=-1)
                    correct = (pred == next_targets).float()
                    risk_target = ce_all.detach().clamp(max=8.0) / 8.0
                if aux_score_active:
                    v_loss = _masked_bce_logits(quality, correct, mask)
                    v_loss = v_loss + _masked_mse((risk.clamp(max=8.0) / 8.0), risk_target, mask)
                    _queue_aux("aux_score", v_loss)
                if controller_active:
                    with torch.no_grad():
                        boundary = block_aux.get("memory_boundary", torch.zeros((), device=input_ids.device)).detach()
                        ctrl_target = torch.zeros_like(next_targets)
                        ctrl_target = torch.where(risk_target > 0.65, torch.full_like(ctrl_target, 3), ctrl_target)
                        ctrl_target = torch.where((risk_target > 0.35) & (risk_target <= 0.65), torch.full_like(ctrl_target, 1), ctrl_target)
                        if torch.is_tensor(boundary) and boundary.numel() == 1:
                            boundary_active = boundary.to(device=mask.device).reshape(()) > 0.45
                            ctrl_target = torch.where(mask & boundary_active, torch.full_like(ctrl_target, 2), ctrl_target)
                    ctrl_logits = vf[..., 5:9]
                    # Class 0 is the learned no-op action, not padding. Mask
                    # padding with a private sentinel instead.
                    ctrl_labels = torch.where(mask, ctrl_target, torch.full_like(ctrl_target, -100))
                    c_per = F.cross_entropy(ctrl_logits.reshape(-1, 4), ctrl_labels.reshape(-1), ignore_index=-100, reduction="none").view_as(ctrl_labels)
                    c_valid = ctrl_labels != -100
                    c_loss = (c_per * c_valid.float()).sum() / c_valid.float().sum().clamp_min(1.0)
                    _queue_aux("controller", c_loss)

            regularizer_names = (
                "equilibrium_residual", "energy", "moe_balance", "moe_router_z",
                "branch_entropy", "conv_scale_entropy", "geodesic", "path_straightness",
                "path_contractive", "energy_row_orth", "latent_norm", "memory_entropy",
                "memory_diversity", "adaptive_effort", "compute_budget", "pathway_memory_entropy",
                "energy_route_balance", "coop_step_redundancy", "coop_gate_budget",
            )
            for name in regularizer_names:
                if _aux_enabled(name) and name in aux:
                    _queue_aux(name, aux[name])
            if _aux_enabled("state_drift") and "state_chunk_drift" in aux:
                _queue_aux("state_drift", aux["state_chunk_drift"] + 0.25 * aux.get("state_norm_error", torch.zeros_like(aux["state_chunk_drift"])))
            if _aux_enabled("state_anchor") and "state_anchor_error" in aux:
                _queue_aux("state_anchor", aux["state_anchor_error"])
            if _aux_enabled("exact_recall_entropy_floor") and "exact_recall_entropy" in aux:
                target_ent = math.log(max(2, int(self.cfg.exact_recall_top_k))) * 0.35
                rec_loss = F.relu(torch.as_tensor(target_ent, device=input_ids.device, dtype=aux["exact_recall_entropy"].dtype) - aux["exact_recall_entropy"]).pow(2)
                _queue_aux("exact_recall_entropy_floor", rec_loss)

            _flush_fused_auxes()
            _finish_objective_diagnostics()
        return logits, loss, aux

    def _masked_mean_projected(self, h: torch.Tensor, mask: torch.Tensor, proj: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project and average a masked token set in O(B*T*D)."""
        m = mask.bool().unsqueeze(-1)
        count = m.float().sum(dim=1).clamp_min(1.0)
        pooled = (h.float() * m.float()).sum(dim=1) / count
        valid = mask.bool().any(dim=1)
        return F.normalize(proj(pooled), dim=-1), valid

    def local_image_text_contrastive_loss(self, h: torch.Tensor, input_ids: torch.Tensor, text_mask: torch.Tensor, img_mask: torch.Tensor) -> torch.Tensor:
        """Linear local image/text agreement.

        The previous implementation built a dense (all text tokens) x (all image
        tokens) matrix.  This version keeps only per-sample masked moments, so the
        cost is linear in visible tokens and does not grow as word_count*patch_count.
        """
        text_vec, text_valid = self._masked_mean_projected(h, text_mask, self.text_proj)
        img_vec, img_valid = self._masked_mean_projected(h, img_mask, self.image_proj)
        valid = text_valid & img_valid
        if not bool(valid.any()):
            return h.new_tensor(0.0)
        agree = 1.0 - (text_vec[valid] * img_vec[valid]).sum(dim=-1)
        return agree.mean()

    def region_word_alignment_loss(self, h: torch.Tensor, input_ids: torch.Tensor, text_mask: torch.Tensor, img_mask: torch.Tensor) -> torch.Tensor:
        """Linear region-word alignment surrogate.

        This intentionally avoids Sinkhorn or dense word-patch transport.  It aligns
        first and second projected moments for the two modalities using masked sums;
        the sequence cost is O(B*T*D), not O(words*patches).
        """
        text_m = text_mask.bool().unsqueeze(-1)
        img_m = img_mask.bool().unsqueeze(-1)
        text_valid = text_mask.bool().any(dim=1)
        img_valid = img_mask.bool().any(dim=1)
        valid = text_valid & img_valid
        if not bool(valid.any()):
            return h.new_tensor(0.0)
        tx_all = self.text_proj(h.float())
        ix_all = self.image_proj(h.float())
        tx_count = text_m.float().sum(dim=1).clamp_min(1.0)
        ix_count = img_m.float().sum(dim=1).clamp_min(1.0)
        tx_mean = (tx_all * text_m.float()).sum(dim=1) / tx_count
        ix_mean = (ix_all * img_m.float()).sum(dim=1) / ix_count
        tx_var = ((tx_all - tx_mean.unsqueeze(1)).pow(2) * text_m.float()).sum(dim=1) / tx_count
        ix_var = ((ix_all - ix_mean.unsqueeze(1)).pow(2) * img_m.float()).sum(dim=1) / ix_count
        mean_loss = 1.0 - (F.normalize(tx_mean[valid], dim=-1) * F.normalize(ix_mean[valid], dim=-1)).sum(dim=-1)
        var_loss = F.smooth_l1_loss(torch.log1p(tx_var[valid]), torch.log1p(ix_var[valid]), reduction="none").mean(dim=-1)
        return (mean_loss + 0.05 * var_loss).mean()

    def image_spatial_smoothness_loss(self, h: torch.Tensor, input_ids: torch.Tensor, img_mask: torch.Tensor) -> torch.Tensor:
        losses: List[torch.Tensor] = []
        for b in range(h.size(0)):
            ii = torch.nonzero(img_mask[b], as_tuple=False).flatten()
            if ii.numel() < 4:
                continue
            # Tokenized image rows may be truncated by block size or may omit EOS, so
            # the number of visible image tokens is not guaranteed to be a perfect square.
            # Use the largest square prefix instead of rounding upward.
            grid = int(math.floor(math.sqrt(int(ii.numel()))))
            if grid <= 1:
                continue
            usable = grid * grid
            feat = F.normalize(h[b, ii[:usable]].float(), dim=-1).view(grid, grid, -1)
            losses.append((feat[1:] - feat[:-1]).pow(2).mean() + (feat[:, 1:] - feat[:, :-1]).pow(2).mean())
        return torch.stack(losses).mean() if losses else h.new_tensor(0.0)

    @torch.no_grad()
    def latent_pathway(self, input_ids: torch.Tensor, stride: int = 2, max_vectors: int = 512) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        ids = input_ids.to(device=device, dtype=torch.long)
        if ids.ndim != 2 or ids.size(1) <= 0:
            raise ValueError("input_ids must have shape [B,T] with T > 0")
        ids = ids[:, -int(self.cfg.block_size):]
        stride = max(1, int(stride))
        max_vectors = max(1, int(max_vectors))
        h, aux, traj = self.encode_hidden(ids)
        h_ctrl, taux = self.timescales(h)
        slot_delta, latent_slots, gaux = self.latent_slot_attention(h_ctrl)
        h_ctrl = h_ctrl + float(self.cfg.latent_slot_conditioning_scale) * slot_delta
        router_delta, man_aux = self.mixture_energy_router(h_ctrl)
        h_ctrl = h_ctrl + router_delta
        vf = self.aux_score(h_ctrl)
        quality = torch.sigmoid(vf[..., 0:1])
        risk = F.softplus(vf[..., 1:2]).clamp(max=8.0) / 8.0
        latent_divergence = self.latent_divergence(h_ctrl, latent_slots)
        risk_signal, uaux = self.risk_signal_fusion(h_ctrl, risk, latent_divergence)
        aux.update(taux); aux.update(gaux); aux.update(man_aux); aux.update(uaux)
        aux["latent_divergence"] = latent_divergence.mean().detach()
        deltas = [b - a for a, b in zip(traj[:-1], traj[1:])]
        delta = torch.stack(deltas).mean(dim=0) if deltas else torch.zeros_like(h)
        curvature = torch.zeros_like(h)
        if len(traj) >= 3:
            second = [(c - b) - (b - a) for a, b, c in zip(traj[:-2], traj[1:-1], traj[2:])]
            curvature = torch.stack(second).mean(dim=0)
        pos = sinusoidal_position_encoding(h.size(1), h.size(2), h.device, h.dtype).unsqueeze(0)
        vectors = h_ctrl + 0.30 * delta + 0.10 * curvature + 0.03 * (quality - risk) + 0.02 * pos
        if stride > 1:
            vectors = vectors[:, ::stride]
        if vectors.size(1) > max_vectors:
            take = torch.linspace(0, vectors.size(1) - 1, max_vectors, device=vectors.device).round().long()
            vectors = vectors.index_select(1, take)
        if vectors.size(1) == 0:
            vectors = h_ctrl[:, -1:]
        output_policy = F.softmax(self.output_mode_classifier(h_ctrl), dim=-1)
        return {
            "pathway_vectors": vectors,
            "pathway_endpoint": h_ctrl[:, -1:],
            "pathway_delta": delta[:, -1:],
            "quality_mean": quality.mean(),
            "risk_mean": risk.mean(),
            "risk_signal_mean": risk_signal.mean(),
            "output_mode_classifier": output_policy.detach(),
            **{k: v.detach() for k, v in aux.items() if torch.is_tensor(v)},
        }

    @torch.no_grad()
    def _aux_score_scores_for_context(
        self,
        ids: torch.Tensor,
        conditioning_vectors: Optional[torch.Tensor] = None,
        conditioning_scale: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        ids = ids.to(device=device, dtype=torch.long)
        if ids.ndim != 2 or ids.size(1) <= 0:
            raise ValueError("ids must have shape [B,T] with T > 0")
        ctx = ids[:, -self.cfg.block_size:]
        _, _, aux = self(ctx, return_hidden=True, conditioning_vectors=conditioning_vectors, conditioning_scale=conditioning_scale)
        h = aux["_hidden"][:, -1:]
        vf = self.aux_score(h)[:, 0]
        return {
            "quality": torch.sigmoid(vf[:, 0]),
            "risk": (F.softplus(vf[:, 1]).clamp(max=8.0) / 8.0),
            "difficulty": torch.sigmoid(vf[:, 2]),
            "contradiction": torch.sigmoid(vf[:, 3]),
            "repetition": torch.sigmoid(vf[:, 4]),
            "controller": F.softmax(vf[:, 5:9] / max(1e-4, float(self.cfg.controller_temperature)), dim=-1) if vf.size(-1) >= 9 else torch.zeros(vf.size(0), 4, device=vf.device),
        }

    @torch.no_grad()
    def _rank_candidates_with_aux_score(
        self,
        base_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_log_probs: torch.Tensor,
        aux_score_weight: float,
        conditioning_vectors: Optional[torch.Tensor],
        conditioning_scale: float,
        rollout_depth: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, n = candidate_ids.shape
        if bsz != 1:
            return candidate_ids[:, :1], candidate_log_probs[:, :1], torch.zeros_like(candidate_log_probs[:, :1]), torch.zeros_like(candidate_log_probs[:, :1])
        risks, contras, reps, qualities = [], [], [], []
        for j in range(n):
            cand_ctx = torch.cat([base_ids, candidate_ids[:, j:j+1]], dim=1)
            if rollout_depth > 0:
                # Test-time latent branch search: cheaply extend each candidate with greedy NEED
                # tokens, then score the trajectory before committing the first token.
                branch = cand_ctx
                for _ in range(int(rollout_depth)):
                    bctx = branch[:, -self.cfg.block_size:]
                    blogits, _, _ = self(bctx, return_hidden=True, conditioning_vectors=conditioning_vectors, conditioning_scale=conditioning_scale)
                    nxt = blogits[:, -1, :].float(); nxt[:, self.cfg.image_token_offset:] = -float("inf")
                    branch = torch.cat([branch, torch.argmax(nxt, dim=-1, keepdim=True)], dim=1)
                scores = self._aux_score_scores_for_context(branch, conditioning_vectors, conditioning_scale)
            else:
                scores = self._aux_score_scores_for_context(cand_ctx, conditioning_vectors, conditioning_scale)
            risks.append(scores["risk"]); contras.append(scores["contradiction"]); reps.append(scores["repetition"]); qualities.append(scores["quality"])
        risk = torch.stack(risks, dim=1)
        contradiction = torch.stack(contras, dim=1)
        repetition = torch.stack(reps, dim=1)
        quality = torch.stack(qualities, dim=1)
        score = candidate_log_probs + float(aux_score_weight) * (quality - risk - 0.7 * contradiction - 0.4 * repetition)
        order = torch.argsort(score, dim=-1, descending=True)
        return (
            torch.gather(candidate_ids, 1, order),
            torch.gather(candidate_log_probs, 1, order),
            torch.gather(risk, 1, order),
            torch.gather(contradiction, 1, order),
        )

    @torch.no_grad()
    def score_text_risk(self, input_ids: torch.Tensor, conditioning_vectors: Optional[torch.Tensor] = None, conditioning_scale: float = 0.0) -> Dict[str, float]:
        scores = self._aux_score_scores_for_context(input_ids.to(next(self.parameters()).device), conditioning_vectors, conditioning_scale)
        out: Dict[str, float] = {}
        for k, v in scores.items():
            if k == "controller":
                out["controller_action"] = float(torch.argmax(v, dim=-1)[0].detach().cpu())
                continue
            out[k] = float(v.mean().detach().cpu())
        return out

    @torch.no_grad()
    def internal_reasoning_summary(self, input_ids: torch.Tensor, max_tokens: int = 64) -> torch.Tensor:
        """Internal distilled summary head for deployments without an external LM sidecar.

        This is intentionally compact: it decodes from the reasoning compressor's pooled
        state.  Training hooks live in need_thought_distill.py, but the method is safe
        to call even before distillation.
        """
        device = next(self.parameters()).device
        ids = input_ids.to(device=device, dtype=torch.long)[:, -int(self.cfg.block_size):]
        h, _, _ = self.encode_hidden(ids)
        r = self.reasoning_compressor(h)
        first = r["summary_logits"].argmax(dim=-1, keepdim=True).clamp(max=self.cfg.text_vocab_size - 1)
        return first.repeat(1, max(1, int(max_tokens)))

    def mtp_logits_from_hidden(self, h_dec: torch.Tensor) -> List[torch.Tensor]:
        """Return next-token and future-token logits from the shared decoder state.

        Index 0 is the normal LM head and predicts t+1. Index i>0 uses the
        corresponding MTP projection and predicts t+i+1 from the same state.
        This method is parameter-free beyond the already-trained MTP heads and is
        safe to call during training or inference.
        """
        outs: List[torch.Tensor] = [self.lm_head(h_dec)]
        for proj in self.mtp_projs:
            outs.append(self.lm_head(proj(h_dec)))
        return outs

    def dvsd_router_logits_from_hidden(self, h_dec: torch.Tensor) -> torch.Tensor:
        """Predict a DVSD slot budget distribution for each hidden position.

        Logit index 0 means one committed slot, index 1 means two slots, etc.
        The head is deliberately small so it can be trained as an auxiliary router
        and used at inference without changing the main recurrent trunk cost.
        """
        return self.dvsd_slot_router(h_dec.float())

    def _dvsd_compound_runtime_enabled(self) -> bool:
        return bool(getattr(self.cfg, "dvsd_planner_compound_enabled", True)) and bool(getattr(self, "_dvsd_planner_compound_loaded", True)) and int(getattr(self.cfg, "planner_horizons", 0)) > 0

    def _dvsd_compound_descent_from_logits(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Approximate the negative next-token CE gradient in hidden space.

        For a linear LM head, d CE / d hidden is roughly expected_output_embedding
        minus target_output_embedding.  DVSD uses the opposite direction as a cheap
        per-token descent signal.  A top-k approximation keeps the cost small and
        avoids another full softmax over image-code tokens during text generation.
        """
        vocab_cut = int(max(1, min(int(getattr(self.cfg, "image_token_offset", self.cfg.vocab_size)), self.cfg.vocab_size)))
        k = min(max(1, int(top_k if top_k is not None else getattr(self.cfg, "dvsd_planner_compound_top_k", 32))), vocab_cut)
        orig = logits.shape[:-1]
        flat_logits = logits[..., :vocab_cut].float().reshape(-1, vocab_cut)
        flat_ids = token_ids.reshape(-1).clamp(min=0, max=vocab_cut - 1)
        vals, ids = torch.topk(flat_logits, k=k, dim=-1)
        probs = F.softmax(vals, dim=-1)
        head_weight = self.lm_head.weight[:vocab_cut].detach().to(device=logits.device, dtype=torch.float32)
        top_weight = head_weight[ids]
        expected = (top_weight * probs.unsqueeze(-1)).sum(dim=1)
        target = self.token_emb(flat_ids).detach().to(device=logits.device, dtype=torch.float32)
        descent = target - expected
        descent = descent * torch.rsqrt(descent.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        return descent.to(dtype=logits.dtype).view(*orig, -1)

    def _dvsd_compound_training_objectives(
        self,
        h_dec: torch.Tensor,
        logits: torch.Tensor,
        future_targets: torch.Tensor,
        next_targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Teacher-force the DVSD compound latent cursor.

        The cursor consumes future targets one at a time, applies the same cheap
        token-feedback/descent update used at inference, and is trained to match the
        teacher-forced future hidden state and next future distribution.  This makes
        later virtual slots train against compounded prior-token decisions instead of
        only same-state shallow horizon projections.
        """
        z = logits.new_tensor(0.0)
        out: Dict[str, torch.Tensor] = {
            "dvsd_compound_latent": z,
            "dvsd_compound_ce": z,
            "dvsd_compound_consistency": z,
            "dvsd_compound_steps": z,
        }
        max_heads = min(int(getattr(self.cfg, "n_predict_heads", 1)), int(future_targets.size(-1)))
        if max_heads <= 1 or not bool(getattr(self.cfg, "dvsd_planner_compound_enabled", True)) or int(getattr(self.cfg, "planner_horizons", 0)) <= 0:
            return out
        vocab_cut = int(max(1, min(int(getattr(self.cfg, "image_token_offset", self.cfg.vocab_size)), self.cfg.vocab_size)))
        state = self.planner.start_state(h_dec)
        prefix_valid = mask.clone()
        prev_logits = logits
        lat_parts: List[torch.Tensor] = []
        ce_parts: List[torch.Tensor] = []
        cons_parts: List[torch.Tensor] = []
        used_steps = 0
        for step in range(1, max_heads):
            consume = future_targets[..., step - 1]
            consume_text = (consume != self.cfg.pad_id) & (consume >= 0) & (consume < vocab_cut)
            prefix_valid = prefix_valid & consume_text
            safe_consume = consume.clamp(min=0, max=vocab_cut - 1)
            token_fb = self.token_emb(safe_consume)
            token_fb = token_fb * consume_text.unsqueeze(-1).to(dtype=token_fb.dtype)
            descent = self._dvsd_compound_descent_from_logits(prev_logits.detach(), safe_consume, top_k=getattr(self.cfg, "dvsd_planner_compound_top_k", 32))
            descent = descent * consume_text.unsqueeze(-1).to(dtype=descent.dtype)
            state = self.planner.compound_step(state, token_fb, step_index=step, logit_descent=descent, confidence=consume_text.float())
            compound_logits = self.lm_head(state)
            prev_logits = compound_logits
            if h_dec.size(1) <= step:
                continue
            aligned_valid = prefix_valid[:, :-step] & (next_targets[:, step:] != self.cfg.pad_id)
            if bool(aligned_valid.any()):
                pred_state = state[:, :-step]
                target_state = h_dec[:, step:].detach()
                diff = (pred_state.float() - target_state.float()).pow(2).mean(dim=-1)
                cos = 1.0 - F.cosine_similarity(pred_state.float(), target_state.float(), dim=-1)
                lat_parts.append(((0.5 * diff + 0.5 * cos) * aligned_valid.float()).sum() / aligned_valid.float().sum().clamp_min(1.0))
            if step < future_targets.size(-1):
                target_tok = future_targets[:, :-step, step]
                target_text = (target_tok != self.cfg.pad_id) & (target_tok >= 0) & (target_tok < vocab_cut)
                pred_valid = prefix_valid[:, :-step] & target_text
                if bool(pred_valid.any()):
                    pred_logits = compound_logits[:, :-step]
                    ce_tok = F.cross_entropy(
                        pred_logits.reshape(-1, self.cfg.vocab_size),
                        target_tok.clamp(min=0, max=self.cfg.vocab_size - 1).reshape(-1),
                        ignore_index=self.cfg.pad_id,
                        label_smoothing=self.cfg.label_smoothing,
                        reduction="none",
                    ).view_as(target_tok)
                    ce_parts.append((ce_tok * pred_valid.float()).sum() / pred_valid.float().sum().clamp_min(1.0))
                    teacher = logits[:, step:, :vocab_cut].detach().float()
                    pred_text_logits = pred_logits[:, :, :vocab_cut].float()
                    kl_tok = F.kl_div(
                        F.log_softmax(pred_text_logits, dim=-1),
                        F.softmax(teacher, dim=-1),
                        reduction="none",
                    ).sum(dim=-1)
                    cons_parts.append((kl_tok * pred_valid.float()).sum() / pred_valid.float().sum().clamp_min(1.0))
                    used_steps += 1
        if lat_parts:
            out["dvsd_compound_latent"] = torch.stack(lat_parts).mean()
        if ce_parts:
            out["dvsd_compound_ce"] = torch.stack(ce_parts).mean()
        if cons_parts:
            out["dvsd_compound_consistency"] = torch.stack(cons_parts).mean()
        out["dvsd_compound_steps"] = logits.new_tensor(float(used_steps))
        return out

    def _dvsd_training_objectives(
        self,
        h_dec: torch.Tensor,
        logits: torch.Tensor,
        future_targets: torch.Tensor,
        next_targets: torch.Tensor,
        ce_all: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return DVSD-native auxiliary losses and metrics.

        This trains three pieces used by direct-commit virtual slots:
        1. slot CE over the whole future canvas, weighted by the longest low-loss
           prefix that could be directly committed,
        2. consistency between MTP future heads and teacher-forced AR logits at the
           corresponding future positions,
        3. a learned router target for the local slot budget.
        """
        out: Dict[str, torch.Tensor] = {}
        max_heads = min(int(self.cfg.n_predict_heads), int(future_targets.size(-1)))
        if max_heads <= 1:
            z = logits.new_tensor(0.0)
            out.update({
                "dvsd_slot_ce": z,
                "dvsd_consistency": z,
                "dvsd_router": z,
                "dvsd_router_entropy": z,
                "dvsd_router_target_slots": z,
                "dvsd_router_pred_slots": z,
            })
            return out

        slot_logits: List[torch.Tensor] = [logits]
        for proj in self.mtp_projs[:max_heads - 1]:
            slot_logits.append(self.lm_head(proj(h_dec)))

        ce_parts: List[torch.Tensor] = []
        valid_parts: List[torch.Tensor] = []
        for i in range(max_heads):
            target_i = future_targets[..., i] if i < future_targets.size(-1) else next_targets
            ce_i = F.cross_entropy(
                slot_logits[i].reshape(-1, self.cfg.vocab_size),
                target_i.reshape(-1),
                ignore_index=self.cfg.pad_id,
                label_smoothing=self.cfg.label_smoothing,
                reduction="none",
            ).view_as(next_targets)
            valid_i = target_i != self.cfg.pad_id
            ce_parts.append(ce_i)
            valid_parts.append(valid_i)

        ce_stack = torch.stack(ce_parts, dim=-1)
        valid_stack = torch.stack(valid_parts, dim=-1)
        with torch.no_grad():
            good = (ce_stack.detach() <= float(self.cfg.dvsd_router_loss_threshold)) & valid_stack
            prefix_good = torch.cumprod(good.to(torch.long), dim=-1).bool()
            target_slots = prefix_good.sum(dim=-1).clamp(min=1, max=max_heads)
            # If the first slot is very hard, keep the target at one slot; if many
            # slots are easy, the router target expands. This is the trained version
            # of hard-region AR fallback.
            first_hard = ce_stack[..., 0].detach() >= float(self.cfg.dvsd_router_hard_loss_threshold)
            target_slots = torch.where(first_hard & mask, torch.ones_like(target_slots), target_slots)
            prefix_weights = (torch.arange(max_heads, device=logits.device).view(1, 1, max_heads) < target_slots.unsqueeze(-1)) & valid_stack

        denom = prefix_weights.float().sum().clamp_min(1.0)
        slot_ce = (ce_stack * prefix_weights.float()).sum() / denom
        out["dvsd_slot_ce"] = slot_ce

        # Match direct future-slot logits to the distribution produced by the same
        # model when teacher-forced up to that future point. This makes slot commit
        # more stable without reintroducing aux_scored acceptance at inference.
        cons_parts: List[torch.Tensor] = []
        vocab_limit = int(self.cfg.image_token_offset or self.cfg.vocab_size)
        for i in range(1, max_heads):
            if logits.size(1) <= i:
                continue
            pred = slot_logits[i][:, :-i, :vocab_limit].float()
            teacher = logits[:, i:, :vocab_limit].detach().float()
            target_i = future_targets[:, :-i, i] if i < future_targets.size(-1) else next_targets[:, :-i]
            cmask = target_i != self.cfg.pad_id
            if bool(cmask.any()):
                kl_tok = F.kl_div(
                    F.log_softmax(pred, dim=-1),
                    F.softmax(teacher, dim=-1),
                    reduction="none",
                ).sum(dim=-1)
                cons_parts.append((kl_tok * cmask.float()).sum() / cmask.float().sum().clamp_min(1.0))
        out["dvsd_consistency"] = torch.stack(cons_parts).mean() if cons_parts else logits.new_tensor(0.0)

        router_logits = self.dvsd_router_logits_from_hidden(h_dec)[..., :max_heads]
        router_target = (target_slots - 1).clamp(min=0, max=max_heads - 1)
        router_valid = mask & valid_stack[..., 0]
        if bool(router_valid.any()):
            router_ce = F.cross_entropy(
                router_logits.reshape(-1, max_heads),
                router_target.reshape(-1),
                reduction="none",
            ).view_as(next_targets)
            router_loss = (router_ce * router_valid.float()).sum() / router_valid.float().sum().clamp_min(1.0)
            router_prob = F.softmax(router_logits.float(), dim=-1)
            router_entropy = -(router_prob * router_prob.clamp_min(1e-8).log()).sum(dim=-1)
            router_entropy_loss = (router_entropy * router_valid.float()).sum() / router_valid.float().sum().clamp_min(1.0)
            pred_slots = router_prob.argmax(dim=-1).add(1).float()
            out["dvsd_router"] = router_loss
            out["dvsd_router_entropy"] = router_entropy_loss
            out["dvsd_router_target_slots"] = (target_slots.float() * router_valid.float()).sum() / router_valid.float().sum().clamp_min(1.0)
            out["dvsd_router_pred_slots"] = (pred_slots * router_valid.float()).sum() / router_valid.float().sum().clamp_min(1.0)
        else:
            z = logits.new_tensor(0.0)
            out["dvsd_router"] = z
            out["dvsd_router_entropy"] = z
            out["dvsd_router_target_slots"] = z
            out["dvsd_router_pred_slots"] = z
        return out

    def _repeat_conditioning_for_batch(self, conditioning_vectors: Optional[torch.Tensor], batch_size: int) -> Optional[torch.Tensor]:
        if conditioning_vectors is None:
            return None
        cond = conditioning_vectors
        if cond.ndim == 2:
            cond = cond.unsqueeze(0)
        if cond.size(0) == 1 and batch_size > 1:
            cond = cond.expand(batch_size, -1, -1)
        return cond

    @torch.no_grad()
    def _nonseq_difficulty_from_logits(
        self,
        next_logits: torch.Tensor,
        aux: Dict[str, torch.Tensor],
        min_heads: int,
        max_heads: int,
        dynamic: bool,
        risk_threshold: float,
        contradiction_threshold: float,
        repetition_threshold: float,
        entropy_easy: float,
        entropy_hard: float,
    ) -> Tuple[int, Dict[str, float]]:
        vocab_cut = int(self.cfg.image_token_offset)
        logits = next_logits[:, :vocab_cut].float()
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs.clamp_min(1e-12))
        entropy = -(probs * log_probs).sum(dim=-1)
        entropy_norm = entropy / math.log(max(2, vocab_cut))
        top_vals, _ = torch.topk(probs, k=min(2, vocab_cut), dim=-1)
        top_prob = top_vals[:, 0]
        if top_vals.size(1) > 1:
            margin = torch.log(top_vals[:, 0].clamp_min(1e-12)) - torch.log(top_vals[:, 1].clamp_min(1e-12))
        else:
            margin = torch.full_like(top_prob, 12.0)
        risk = next_logits.new_zeros(next_logits.size(0))
        contradiction = next_logits.new_zeros(next_logits.size(0))
        repetition = next_logits.new_zeros(next_logits.size(0))
        difficulty = next_logits.new_zeros(next_logits.size(0))
        uncertainty = next_logits.new_zeros(next_logits.size(0))
        h = aux.get("_hidden", None)
        if h is not None:
            vf = self.aux_score(h[:, -1:])[:, 0]
            risk = F.softplus(vf[:, 1]).clamp(max=8.0) / 8.0
            difficulty = torch.sigmoid(vf[:, 2])
            contradiction = torch.sigmoid(vf[:, 3])
            repetition = torch.sigmoid(vf[:, 4])
        unc = aux.get("_uncertainty", None)
        if unc is not None:
            uncertainty = unc[:, -1].float().view(-1).clamp(0.0, 1.0)
        hard_gate = (
            (entropy_norm >= float(entropy_hard))
            | (risk >= float(risk_threshold))
            | (contradiction >= float(contradiction_threshold))
            | (repetition >= float(repetition_threshold))
        )
        diff = (
            0.38 * entropy_norm.clamp(0.0, 1.0)
            + 0.20 * (1.0 - top_prob).clamp(0.0, 1.0)
            + 0.14 * risk.clamp(0.0, 1.0)
            + 0.10 * difficulty.clamp(0.0, 1.0)
            + 0.08 * contradiction.clamp(0.0, 1.0)
            + 0.05 * repetition.clamp(0.0, 1.0)
            + 0.05 * uncertainty.clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)
        router_pred = next_logits.new_tensor(0.0)
        router_conf = next_logits.new_tensor(0.0)
        router_available = False
        router_used = False
        if dynamic:
            span = max(0, int(max_heads) - int(min_heads))
            if bool(hard_gate[0].detach().cpu()):
                heads = int(min_heads)
            elif float(entropy_norm[0].detach().cpu()) <= float(entropy_easy) and float(top_prob[0].detach().cpu()) >= 0.35:
                heads = int(max_heads)
            else:
                heads = int(round(float(max_heads) - float(diff[0].detach().cpu()) * span))
                heads = max(int(min_heads), min(int(max_heads), heads))

            if bool(getattr(self.cfg, "dvsd_router_enabled", True)) and h is not None and int(max_heads) > int(min_heads):
                router_available = True
                try:
                    rlog = self.dvsd_router_logits_from_hidden(h[:, -1:])[:, 0, :int(max_heads)].float()
                    rprob = F.softmax(rlog, dim=-1)
                    rconf, ridx = torch.max(rprob, dim=-1)
                    rheads = (ridx + 1).clamp(min=int(min_heads), max=int(max_heads))
                    router_pred = rheads.float().mean()
                    router_conf = rconf.float().mean()
                    if float(router_conf.detach().cpu()) >= float(getattr(self.cfg, "dvsd_router_min_confidence", 0.0)):
                        mix = float(getattr(self.cfg, "dvsd_router_inference_mix", 0.0))
                        blended = int(round((1.0 - mix) * float(heads) + mix * float(rheads[0].detach().cpu())))
                        heads = max(int(min_heads), min(int(max_heads), blended))
                        router_used = True
                    # Never let the learned router override a hard safety collapse.
                    if bool(hard_gate[0].detach().cpu()):
                        heads = int(min_heads)
                except Exception:
                    router_used = False
        else:
            heads = int(max_heads)
        stats = {
            "nonseq_entropy_norm": float(entropy_norm.mean().detach().cpu()),
            "nonseq_top_prob": float(top_prob.mean().detach().cpu()),
            "nonseq_logprob_margin": float(margin.mean().detach().cpu()),
            "nonseq_risk": float(risk.mean().detach().cpu()),
            "nonseq_difficulty": float(difficulty.mean().detach().cpu()),
            "nonseq_contradiction": float(contradiction.mean().detach().cpu()),
            "nonseq_repetition": float(repetition.mean().detach().cpu()),
            "nonseq_uncertainty": float(uncertainty.mean().detach().cpu()),
            "nonseq_difficulty_score": float(diff.mean().detach().cpu()),
            "nonseq_hard_gate": float(hard_gate.float().mean().detach().cpu()),
            "dvsd_router_pred_heads": float(router_pred.detach().cpu()) if torch.isfinite(router_pred).all() else 0.0,
            "dvsd_router_confidence": float(router_conf.detach().cpu()) if torch.isfinite(router_conf).all() else 0.0,
            "dvsd_router_available": 1.0 if router_available else 0.0,
            "dvsd_router_used": 1.0 if router_used else 0.0,
        }
        return heads, stats

    @torch.no_grad()
    def _nonseq_head_candidates(
        self,
        logits: torch.Tensor,
        prefix_ids: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        typical_p: float,
        repetition_penalty: float,
        no_repeat_ngram: int,
        forbid_eos: bool,
        eos_id: int,
        branch_top_k: int,
    ) -> Tuple[List[int], List[float], Dict[str, float]]:
        vocab_cut = int(self.cfg.image_token_offset)
        local = logits.float().clone()
        local[:, vocab_cut:] = -float("inf")
        apply_repetition_penalty_(local, prefix_ids, repetition_penalty)
        if forbid_eos:
            local[:, int(eos_id)] = -float("inf")
        fallback_local = local.clone()
        if no_repeat_ngram > 1:
            apply_no_repeat_ngram_(local, prefix_ids, no_repeat_ngram)
        if temperature <= 0:
            filtered = top_k_top_p_filter(local, top_k=0, top_p=1.0)
            probs = safe_softmax_probs(filtered, fallback_local)
            log_probs = torch.log(probs.clamp_min(1e-12))
        else:
            filtered = typical_filter(local / max(float(temperature), 1e-8), typical_p)
            filtered = top_k_top_p_filter(filtered, top_k=top_k, top_p=top_p)
            probs = safe_softmax_probs(filtered, fallback_local)
            log_probs = torch.log(probs.clamp_min(1e-12))
        text_probs = probs[:, :vocab_cut]
        k = min(max(1, int(branch_top_k)), vocab_cut)
        top_probs, top_ids = torch.topk(text_probs, k=k, dim=-1)
        top_ids_list = [int(x) for x in top_ids[0].detach().cpu().tolist()]
        top_lp_list = [float(torch.log(x.clamp_min(1e-12)).detach().cpu()) for x in top_probs[0]]
        if temperature > 0:
            sampled = int(torch.multinomial(text_probs.clamp_min(0.0), num_samples=1)[0, 0].detach().cpu())
            sampled_lp = float(log_probs[0, sampled].detach().cpu())
        else:
            sampled = top_ids_list[0]
            sampled_lp = top_lp_list[0]
        ids: List[int] = []
        lps: List[float] = []
        for tok, lp in [(sampled, sampled_lp), *zip(top_ids_list, top_lp_list)]:
            if tok not in ids:
                ids.append(int(tok)); lps.append(float(lp))
        entropy = -(text_probs * torch.log(text_probs.clamp_min(1e-12))).sum(dim=-1) / math.log(max(2, vocab_cut))
        if top_probs.size(1) > 1:
            gap = torch.log(top_probs[:, 0].clamp_min(1e-12)) - torch.log(top_probs[:, 1].clamp_min(1e-12))
        else:
            gap = torch.full((1,), 12.0, device=logits.device)
        stats = {
            "prob": float(text_probs[0, ids[0]].detach().cpu()),
            "entropy": float(entropy[0].detach().cpu()),
            "gap": float(gap[0].detach().cpu()),
        }
        return ids, lps, stats

    @torch.no_grad()
    def _nonseq_build_draft_tree(
        self,
        head_logits: List[torch.Tensor],
        idx: torch.Tensor,
        generated_so_far: int,
        min_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        typical_p: float,
        repetition_penalty: float,
        no_repeat_ngram: int,
        eos_id: int,
        branch_top_k: int,
        max_candidates: int,
        dynamic: bool,
        min_heads: int,
        min_draft_prob: float,
        max_head_entropy: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, Dict[str, float]]:
        device = idx.device
        prefix = idx
        local_candidates: List[List[Tuple[int, float]]] = []
        primary: List[int] = []
        head_probs: List[float] = []
        head_entropy: List[float] = []
        head_gaps: List[float] = []
        active = len(head_logits)
        for i, logits in enumerate(head_logits):
            forbid_eos = (generated_so_far + i) < int(min_new_tokens)
            ids, lps, stats = self._nonseq_head_candidates(
                logits,
                prefix,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                typical_p=typical_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram=no_repeat_ngram,
                forbid_eos=forbid_eos,
                eos_id=eos_id,
                branch_top_k=branch_top_k,
            )
            cand = list(zip(ids[:max(1, int(branch_top_k))], lps[:max(1, int(branch_top_k))]))
            local_candidates.append(cand)
            primary.append(int(ids[0]))
            head_probs.append(float(stats["prob"]))
            head_entropy.append(float(stats["entropy"]))
            head_gaps.append(float(stats["gap"]))
            prefix = torch.cat([prefix, torch.tensor([[int(ids[0])]], dtype=torch.long, device=device)], dim=1)
        if dynamic and active > int(min_heads):
            keep = active
            for i in range(int(min_heads), active):
                if head_probs[i] < float(min_draft_prob) or head_entropy[i] > float(max_head_entropy):
                    keep = i
                    break
            active = max(int(min_heads), min(active, keep))
            local_candidates = local_candidates[:active]
            primary = primary[:active]
            head_probs = head_probs[:active]
            head_entropy = head_entropy[:active]
            head_gaps = head_gaps[:active]
        beams: List[Tuple[List[int], float]] = [([], 0.0)]
        for pos, cands in enumerate(local_candidates):
            new_beams: List[Tuple[List[int], float]] = []
            for seq, score in beams:
                for tok, lp in cands:
                    new_beams.append((seq + [int(tok)], score + float(lp)))
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:max(1, int(max_candidates))]
        # Always include the sampled primary path, even if top-k beam pruning would drop it.
        primary_score = 0.0
        for pos, tok in enumerate(primary):
            found = False
            for cand_tok, cand_lp in local_candidates[pos]:
                if int(cand_tok) == int(tok):
                    primary_score += float(cand_lp); found = True; break
            if not found:
                primary_score -= 40.0
        if primary and all(seq != primary for seq, _ in beams):
            beams.append((primary, primary_score))
        beams.sort(key=lambda x: x[1], reverse=True)
        beams = beams[:max(1, int(max_candidates))]
        drafts = torch.tensor([seq for seq, _ in beams], dtype=torch.long, device=device)
        draft_log_probs = torch.tensor([score for _, score in beams], dtype=torch.float32, device=device)
        stats = {
            "nonseq_draft_heads": float(active),
            "nonseq_head_prob_mean": float(sum(head_probs) / max(1, len(head_probs))),
            "nonseq_head_entropy_mean": float(sum(head_entropy) / max(1, len(head_entropy))),
            "nonseq_head_gap_mean": float(sum(head_gaps) / max(1, len(head_gaps))),
            "nonseq_tree_candidates": float(drafts.size(0)),
        }
        return drafts, draft_log_probs, active, stats

    @torch.no_grad()
    def _validate_nonseq_draft_batch(
        self,
        idx: torch.Tensor,
        drafts: torch.Tensor,
        draft_log_probs: torch.Tensor,
        generated_so_far: int,
        max_new_tokens: int,
        min_new_tokens: int,
        repetition_penalty: float,
        no_repeat_ngram: int,
        accept_top_k: int,
        accept_min_prob: float,
        accept_max_logprob_gap: float,
        risk_threshold: float,
        contradiction_threshold: float,
        repetition_threshold: float,
        aux_score_weight: float,
        eos_id: int,
        conditioning_vectors: Optional[torch.Tensor],
        conditioning_scale: float,
    ) -> Tuple[int, torch.Tensor, Dict[str, float]]:
        device = next(self.parameters()).device
        idx = idx.to(device=device, dtype=torch.long)
        drafts = drafts.to(device=device, dtype=torch.long)
        if idx.size(0) != 1 or drafts.ndim != 2 or drafts.size(0) < 1 or drafts.size(1) < 1:
            empty = torch.empty((1, 0), dtype=torch.long, device=device)
            return 0, empty, {"reject_reason": "empty_or_batch_not_supported"}
        max_take = min(int(drafts.size(1)), int(max_new_tokens) - int(generated_so_far))
        if max_take <= 0:
            empty = torch.empty((1, 0), dtype=torch.long, device=device)
            return 0, empty, {"reject_reason": "no_budget"}
        drafts = drafts[:, :max_take]
        cand_count = drafts.size(0)
        base = idx.expand(cand_count, -1)
        full = torch.cat([base, drafts], dim=1)
        ctx = full[:, -self.cfg.block_size:]
        offset = full.size(1) - ctx.size(1)
        cond = self._repeat_conditioning_for_batch(conditioning_vectors, cand_count)
        logits, _, aux = self(ctx, return_hidden=True, conditioning_vectors=cond, conditioning_scale=conditioning_scale)
        hidden = aux.get("_hidden", None)
        vocab_cut = int(self.cfg.image_token_offset)
        best_i = 0
        best_accepted = -1
        best_score = -1e30
        best_reason = "none"
        totals = {"need_prob": 0.0, "gap": 0.0, "risk": 0.0, "contradiction": 0.0, "repetition": 0.0, "seen": 0.0}
        for c in range(cand_count):
            prefix = idx.clone()
            accepted = 0
            score = float(draft_log_probs[c].detach().cpu()) if draft_log_probs.numel() > c else 0.0
            reason = "none"
            for j in range(drafts.size(1)):
                cand = int(drafts[c, j].item())
                if cand >= vocab_cut:
                    reason = "non_text_token"; break
                abs_pred = idx.size(1) + j - 1
                abs_tok = idx.size(1) + j
                rel_pred = abs_pred - offset
                rel_tok = abs_tok - offset
                if rel_pred < 0 or rel_pred >= logits.size(1):
                    reason = "context_cropped"; break
                next_logits = logits[c:c + 1, rel_pred, :].float().clone()
                next_logits[:, vocab_cut:] = -float("inf")
                apply_repetition_penalty_(next_logits, prefix, repetition_penalty)
                if no_repeat_ngram > 1:
                    apply_no_repeat_ngram_(next_logits, prefix, no_repeat_ngram)
                if generated_so_far + j < int(min_new_tokens):
                    next_logits[:, int(eos_id)] = -float("inf")
                lp = F.log_softmax(next_logits, dim=-1)
                cand_lp = lp[0, cand]
                need_prob = float(cand_lp.exp().detach().cpu())
                k = min(max(1, int(accept_top_k)), vocab_cut)
                top_vals, top_ids = torch.topk(lp[:, :vocab_cut], k=k, dim=-1)
                gap = float((top_vals[0, 0] - cand_lp).detach().cpu())
                in_topk = bool((top_ids[0] == cand).any().item())
                risk = contradiction = repetition = 0.0
                if hidden is not None and 0 <= rel_tok < hidden.size(1):
                    vf = self.aux_score(hidden[c:c + 1, rel_tok:rel_tok + 1])[:, 0]
                    risk = float((F.softplus(vf[:, 1]).clamp(max=8.0) / 8.0).mean().detach().cpu())
                    contradiction = float(torch.sigmoid(vf[:, 3]).mean().detach().cpu())
                    repetition = float(torch.sigmoid(vf[:, 4]).mean().detach().cpu())
                ok_prob = in_topk or need_prob >= float(accept_min_prob) or gap <= float(accept_max_logprob_gap)
                ok_risk = risk <= float(risk_threshold)
                ok_con = contradiction <= float(contradiction_threshold)
                ok_rep = repetition <= float(repetition_threshold)
                totals["need_prob"] += need_prob; totals["gap"] += gap; totals["risk"] += risk; totals["contradiction"] += contradiction; totals["repetition"] += repetition; totals["seen"] += 1.0
                if ok_prob and ok_risk and ok_con and ok_rep:
                    accepted += 1
                    score += float(cand_lp.detach().cpu())
                    score += float(aux_score_weight) * (need_prob - risk - 0.7 * contradiction - 0.4 * repetition)
                    prefix = torch.cat([prefix, drafts[c:c + 1, j:j + 1]], dim=1)
                    if cand == int(eos_id):
                        break
                else:
                    if not ok_prob:
                        reason = "need_distribution_disagreement"
                    elif not ok_risk:
                        reason = "aux_score_risk"
                    elif not ok_con:
                        reason = "aux_score_contradiction"
                    else:
                        reason = "aux_score_repetition"
                    break
            if accepted > best_accepted or (accepted == best_accepted and score > best_score):
                best_accepted = int(accepted); best_score = float(score); best_i = int(c); best_reason = reason
        best_accepted = max(0, int(best_accepted))
        accepted_tokens = drafts[best_i:best_i + 1, :best_accepted]
        seen = max(1.0, totals["seen"])
        stats = {
            "accepted_need_tokens": float(best_accepted),
            "draft_need_tokens": float(drafts.size(1)),
            "candidate_drafts": float(cand_count),
            "reject_reason": best_reason,
            "avg_need_prob": float(totals["need_prob"] / seen),
            "avg_logprob_gap": float(totals["gap"] / seen),
            "avg_risk": float(totals["risk"] / seen),
            "avg_contradiction": float(totals["contradiction"] / seen),
            "avg_repetition": float(totals["repetition"] / seen),
        }
        return best_accepted, accepted_tokens, stats

    @torch.no_grad()
    def _nonseq_sample_slot_token(
        self,
        logits: torch.Tensor,
        prefix_ids: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        typical_p: float,
        repetition_penalty: float,
        no_repeat_ngram: int,
        forbid_eos: bool,
        eos_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample one text token from a virtual slot distribution.

        This is intentionally not an accept/reject check. It is the primitive used by
        the slot-refinement decoder to update provisional future positions before the
        whole virtual canvas is committed.
        """
        vocab_cut = int(self.cfg.image_token_offset)
        local = logits.float().clone()
        local[:, vocab_cut:] = -float("inf")
        apply_repetition_penalty_(local, prefix_ids, repetition_penalty)
        if forbid_eos:
            local[:, int(eos_id)] = -float("inf")
        fallback_local = local.clone()
        if no_repeat_ngram > 1:
            apply_no_repeat_ngram_(local, prefix_ids, no_repeat_ngram)
        if temperature <= 0:
            filtered = local
        else:
            filtered = typical_filter(local / max(float(temperature), 1e-8), typical_p)
            filtered = top_k_top_p_filter(filtered, top_k=top_k, top_p=top_p)
        text_logits = filtered[:, :vocab_cut]
        probs = safe_softmax_probs(text_logits, fallback_local[:, :vocab_cut])
        if temperature <= 0:
            tok = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            tok = torch.multinomial(probs.clamp_min(0.0), num_samples=1)
        conf = probs.gather(1, tok).squeeze(-1).clamp(0.0, 1.0)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1) / math.log(max(2, vocab_cut))
        top_vals = torch.topk(probs, k=min(2, vocab_cut), dim=-1).values
        if top_vals.size(1) > 1:
            gap = torch.log(top_vals[:, 0].clamp_min(1e-12)) - torch.log(top_vals[:, 1].clamp_min(1e-12))
        else:
            gap = torch.full_like(conf, 12.0)
        return tok.to(dtype=torch.long), conf, entropy.clamp(0.0, 1.0), gap

    @torch.no_grad()
    def _dvsd_slot_generation_order(self, head_logits: List[torch.Tensor], mode: Optional[str] = None) -> List[int]:
        """Choose the order in which virtual slots are filled.

        DVSD treats the block as a private canvas, so the next generated token does
        not have to be the leftmost blank.  The confidence order uses each slot
        head's current distribution to fill the most decisive position first.
        This gives the block a chance to place anchors before details while still
        committing the final canvas left-to-right only after validation.
        """
        n = len(head_logits)
        if n <= 1:
            return list(range(n))
        local_mode = str(mode or getattr(self.cfg, "dvsd_slot_order", "confidence") or "confidence")
        if local_mode == "left_to_right":
            return list(range(n))
        vocab_cut = int(self.cfg.image_token_offset)
        scored: List[Tuple[float, int]] = []
        for pos, logits in enumerate(head_logits):
            local = logits[:, :vocab_cut].float()
            finite = torch.isfinite(local).any(dim=-1, keepdim=True)
            if not bool(finite.all().detach().cpu()):
                local = torch.where(torch.isfinite(local), local, torch.full_like(local, -1e9))
            probs = F.softmax(local, dim=-1)
            conf = probs.max(dim=-1).values.clamp(0.0, 1.0)
            ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1) / math.log(max(2, vocab_cut))
            # Confidence first, with low entropy as a tie-breaker.  A tiny left-to-right
            # bias keeps deterministic output stable when slots are equally decisive.
            score = float((conf - 0.25 * ent.clamp(0.0, 1.0)).mean().detach().cpu()) - 1e-6 * float(pos)
            scored.append((score, pos))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [pos for _, pos in scored]

    def _nonseq_refine_keep_count(self, n_slots: int, step_index: int, total_steps: int, schedule: str) -> int:
        n = max(1, int(n_slots))
        total = max(1, int(total_steps))
        progress = min(1.0, max(0.0, float(step_index) / float(total)))
        if schedule == "linear":
            ratio = progress
        elif schedule == "quadratic":
            ratio = progress * progress
        else:
            ratio = 1.0 - math.cos(progress * math.pi / 2.0)
        return min(n, max(1, int(math.ceil(n * ratio))))

    @torch.no_grad()
    def generate_text_nonsequential(
        self,
        *args,
        nonseq_decode_style: Optional[str] = None,
        nonseq_refine_steps: Optional[int] = None,
        nonseq_refine_causal_blend: Optional[float] = None,
        nonseq_refine_confidence_floor: Optional[float] = None,
        nonseq_refine_temperature_decay: Optional[float] = None,
        nonseq_refine_lock_schedule: Optional[str] = None,
        nonseq_refine_resample_locked: Optional[bool] = None,
        dvsd_planner_compound_enabled: Optional[bool] = None,
        dvsd_planner_compound_mix: Optional[float] = None,
        dvsd_planner_compound_step_size: Optional[float] = None,
        dvsd_planner_compound_token_scale: Optional[float] = None,
        dvsd_planner_compound_descent_scale: Optional[float] = None,
        dvsd_planner_compound_top_k: Optional[int] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """Canonical dynamic nonsequential text generator.

        This uses slot refinement: a dynamic future canvas is filled and refined
        directly, with no aux_scored-acceptance pass and no longest-prefix rejection.
        The only fallback behavior is dynamic head-count collapse to a one-slot
        step, which is the AR end of the same decoder spectrum.
        """
        _style = str(nonseq_decode_style or getattr(self.cfg, "nonseq_decode_style", "slot_refine") or "slot_refine")
        # Keep the argument for checkpoint/profile compatibility, but the public
        # canonical decoder always uses direct virtual-slot commit.
        return self.generate_text_nonsequential_slot_refine(
            *args,
            nonseq_refine_steps=nonseq_refine_steps,
            nonseq_refine_causal_blend=nonseq_refine_causal_blend,
            nonseq_refine_confidence_floor=nonseq_refine_confidence_floor,
            nonseq_refine_temperature_decay=nonseq_refine_temperature_decay,
            nonseq_refine_lock_schedule=nonseq_refine_lock_schedule,
            nonseq_refine_resample_locked=nonseq_refine_resample_locked,
            dvsd_planner_compound_enabled=dvsd_planner_compound_enabled,
            dvsd_planner_compound_mix=dvsd_planner_compound_mix,
            dvsd_planner_compound_step_size=dvsd_planner_compound_step_size,
            dvsd_planner_compound_token_scale=dvsd_planner_compound_token_scale,
            dvsd_planner_compound_descent_scale=dvsd_planner_compound_descent_scale,
            dvsd_planner_compound_top_k=dvsd_planner_compound_top_k,
            **kwargs,
        )

    @torch.no_grad()
    def generate_text_dynamic_mtp(self, *args, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """Compatibility alias for the internal dynamic nonsequential decoder."""
        return self.generate_text_nonsequential(*args, **kwargs)

    @torch.no_grad()
    def generate_text_nonsequential_slot_refine(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        typical_p: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram: int = 0,
        min_new_tokens: int = 0,
        lookahead_blend: float = 0.0,
        aux_score_top_k: int = 0,
        aux_score_weight: float = 0.0,
        eos_id: Optional[int] = None,
        conditioning_vectors: Optional[torch.Tensor] = None,
        conditioning_scale: float = 0.0,
        proactive_aux_score: Optional[bool] = None,
        aux_score_risk_threshold: Optional[float] = None,
        aux_score_contradiction_threshold: Optional[float] = None,
        aux_score_candidate_pool: Optional[int] = None,
        aux_score_backtrack_window: Optional[int] = None,
        aux_score_max_backtracks: Optional[int] = None,
        latent_search_depth: Optional[int] = None,
        latent_search_branches: Optional[int] = None,
        nonseq_min_heads: Optional[int] = None,
        nonseq_max_heads: Optional[int] = None,
        nonseq_dynamic: Optional[bool] = None,
        nonseq_accept_top_k: Optional[int] = None,
        nonseq_accept_min_prob: Optional[float] = None,
        nonseq_accept_max_logprob_gap: Optional[float] = None,
        nonseq_risk_threshold: Optional[float] = None,
        nonseq_contradiction_threshold: Optional[float] = None,
        nonseq_repetition_threshold: Optional[float] = None,
        nonseq_entropy_easy: Optional[float] = None,
        nonseq_entropy_hard: Optional[float] = None,
        nonseq_min_draft_prob: Optional[float] = None,
        nonseq_max_head_entropy: Optional[float] = None,
        nonseq_tree_candidates: Optional[int] = None,
        nonseq_branch_top_k: Optional[int] = None,
        nonseq_aux_score_weight: Optional[float] = None,
        nonseq_fallback_to_ar: Optional[bool] = None,
        nonseq_refine_steps: Optional[int] = None,
        nonseq_refine_causal_blend: Optional[float] = None,
        nonseq_refine_confidence_floor: Optional[float] = None,
        nonseq_refine_temperature_decay: Optional[float] = None,
        nonseq_refine_lock_schedule: Optional[str] = None,
        nonseq_refine_resample_locked: Optional[bool] = None,
        dvsd_planner_compound_enabled: Optional[bool] = None,
        dvsd_planner_compound_mix: Optional[float] = None,
        dvsd_planner_compound_step_size: Optional[float] = None,
        dvsd_planner_compound_token_scale: Optional[float] = None,
        dvsd_planner_compound_descent_scale: Optional[float] = None,
        dvsd_planner_compound_top_k: Optional[int] = None,
        return_stats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """Dynamic virtual-slot generator with direct canvas commit.

        This is the AR/diffusion midpoint decoder: one model state opens a future
        canvas of 1..H slots, MTP heads initialize all slots in parallel, refinement
        passes update the least-confident slots using provisional causal context, and
        a confidence schedule locks positions in whichever order is easiest. There is
        no aux_scored-acceptance pass and no longest-prefix rejection; if a moment is
        hard, the dynamic controller simply makes the canvas one slot wide, which is
        normal autoregressive sampling for that step.
        """
        device = next(self.parameters()).device
        idx = input_ids.to(device=device, dtype=torch.long)
        if int(max_new_tokens) <= 0:
            stats = {"nonseq_decode": 0.0, "nonseq_slot_refine": 0.0, "nonseq_zero_tokens": 1.0}
            return (idx, stats) if return_stats else idx
        if idx.size(0) != 1:
            out = self.generate_text(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                typical_p=typical_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram=no_repeat_ngram,
                min_new_tokens=min_new_tokens,
                lookahead_blend=lookahead_blend,
                aux_score_top_k=aux_score_top_k,
                aux_score_weight=aux_score_weight,
                eos_id=eos_id,
                conditioning_vectors=conditioning_vectors,
                conditioning_scale=conditioning_scale,
                proactive_aux_score=proactive_aux_score,
                aux_score_risk_threshold=aux_score_risk_threshold,
                aux_score_contradiction_threshold=aux_score_contradiction_threshold,
                aux_score_candidate_pool=aux_score_candidate_pool,
                aux_score_backtrack_window=aux_score_backtrack_window,
                aux_score_max_backtracks=aux_score_max_backtracks,
                latent_search_depth=latent_search_depth,
                latent_search_branches=latent_search_branches,
            )
            stats = {"nonseq_decode": 0.0, "nonseq_slot_refine": 0.0, "nonseq_batch_fallback": 1.0}
            return (out, stats) if return_stats else out

        eos_id = self.cfg.eos_id if eos_id is None else int(eos_id)
        max_heads_avail = max(1, int(self.cfg.n_predict_heads))
        max_heads = int(self.cfg.nonseq_max_heads if nonseq_max_heads is None else nonseq_max_heads)
        min_heads = int(self.cfg.nonseq_min_heads if nonseq_min_heads is None else nonseq_min_heads)
        min_heads = max(1, min(min_heads, max_heads_avail))
        max_heads = max(min_heads, min(max_heads, max_heads_avail, int(max_new_tokens) if max_new_tokens > 0 else 1))
        dynamic = bool(self.cfg.nonseq_dynamic if nonseq_dynamic is None else nonseq_dynamic)
        risk_thr = float(self.cfg.nonseq_risk_threshold if nonseq_risk_threshold is None else nonseq_risk_threshold)
        con_thr = float(self.cfg.nonseq_contradiction_threshold if nonseq_contradiction_threshold is None else nonseq_contradiction_threshold)
        rep_thr = float(self.cfg.nonseq_repetition_threshold if nonseq_repetition_threshold is None else nonseq_repetition_threshold)
        entropy_easy = float(self.cfg.nonseq_entropy_easy if nonseq_entropy_easy is None else nonseq_entropy_easy)
        entropy_hard = float(self.cfg.nonseq_entropy_hard if nonseq_entropy_hard is None else nonseq_entropy_hard)
        min_draft_prob = float(self.cfg.nonseq_min_draft_prob if nonseq_min_draft_prob is None else nonseq_min_draft_prob)
        max_head_entropy = float(self.cfg.nonseq_max_head_entropy if nonseq_max_head_entropy is None else nonseq_max_head_entropy)
        refine_steps_cfg = int(self.cfg.nonseq_refine_steps if nonseq_refine_steps is None else nonseq_refine_steps)
        refine_steps_cfg = max(1, refine_steps_cfg)
        causal_blend = float(self.cfg.nonseq_refine_causal_blend if nonseq_refine_causal_blend is None else nonseq_refine_causal_blend)
        causal_blend = min(max(causal_blend, 0.0), 1.0)
        confidence_floor = float(self.cfg.nonseq_refine_confidence_floor if nonseq_refine_confidence_floor is None else nonseq_refine_confidence_floor)
        confidence_floor = min(max(confidence_floor, 0.0), 1.0)
        temp_decay = float(self.cfg.nonseq_refine_temperature_decay if nonseq_refine_temperature_decay is None else nonseq_refine_temperature_decay)
        temp_decay = min(max(temp_decay, 0.05), 1.0)
        lock_schedule = str(self.cfg.nonseq_refine_lock_schedule if nonseq_refine_lock_schedule is None else nonseq_refine_lock_schedule)
        if lock_schedule not in {"cosine", "linear", "quadratic"}:
            lock_schedule = "cosine"
        resample_locked = bool(self.cfg.nonseq_refine_resample_locked if nonseq_refine_resample_locked is None else nonseq_refine_resample_locked)
        base_compound_enabled = self._dvsd_compound_runtime_enabled()
        if dvsd_planner_compound_enabled is not None:
            base_compound_enabled = bool(dvsd_planner_compound_enabled) and bool(getattr(self, "_dvsd_planner_compound_loaded", True)) and int(getattr(self.cfg, "planner_horizons", 0)) > 0
        compound_mix = float(self.cfg.dvsd_planner_compound_mix if dvsd_planner_compound_mix is None else dvsd_planner_compound_mix)
        compound_mix = min(max(compound_mix, 0.0), 1.0)
        old_step_size = float(getattr(self.cfg, "dvsd_planner_compound_step_size", 0.65))
        old_token_scale = float(getattr(self.cfg, "dvsd_planner_compound_token_scale", 0.18))
        old_descent_scale = float(getattr(self.cfg, "dvsd_planner_compound_descent_scale", 0.22))
        if dvsd_planner_compound_step_size is not None:
            self.cfg.dvsd_planner_compound_step_size = float(min(max(float(dvsd_planner_compound_step_size), 0.0), 2.0))
        if dvsd_planner_compound_token_scale is not None:
            self.cfg.dvsd_planner_compound_token_scale = float(min(max(float(dvsd_planner_compound_token_scale), 0.0), 2.0))
        if dvsd_planner_compound_descent_scale is not None:
            self.cfg.dvsd_planner_compound_descent_scale = float(min(max(float(dvsd_planner_compound_descent_scale), 0.0), 2.0))
        compound_top_k = int(self.cfg.dvsd_planner_compound_top_k if dvsd_planner_compound_top_k is None else dvsd_planner_compound_top_k)
        compound_top_k = max(1, compound_top_k)

        generated = 0
        stats: Dict[str, float] = {
            "nonseq_decode": 1.0,
            "nonseq_slot_refine": 1.0,
            "nonseq_unaux_scored_commit": 1.0,
            "nonseq_steps": 0.0,
            "nonseq_forward_draft_calls": 0.0,
            "nonseq_refine_forward_calls": 0.0,
            "nonseq_validate_calls": 0.0,
            "nonseq_drafted_tokens": 0.0,
            "nonseq_committed_tokens": 0.0,
            "nonseq_accepted_tokens": 0.0,
            "nonseq_rejected_tokens": 0.0,
            "nonseq_ar_fallback_tokens": 0.0,
            "nonseq_head1_steps": 0.0,
            "nonseq_avg_active_heads": 0.0,
            "nonseq_avg_refine_steps": 0.0,
            "nonseq_avg_slot_confidence": 0.0,
            "nonseq_avg_slot_entropy": 0.0,
            "nonseq_avg_locked_first_pass": 0.0,
            "nonseq_compound_enabled": 1.0 if base_compound_enabled and compound_mix > 0.0 else 0.0,
            "nonseq_compound_steps": 0.0,
            "nonseq_compound_logit_blends": 0.0,
            "nonseq_compound_descent_norm": 0.0,
            "nonseq_dvsd_nonmonotonic_slots": 0.0,
        }

        while generated < int(max_new_tokens):
            remaining = int(max_new_tokens) - generated
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _, aux = self(idx_cond, return_hidden=True, conditioning_vectors=conditioning_vectors, conditioning_scale=conditioning_scale)
            h_dec = aux["_hidden"]
            next_logits = logits[:, -1, :].float().clone()
            next_logits[:, self.cfg.image_token_offset:] = -float("inf")
            if lookahead_blend > 0 and "_planned_next" in aux:
                plan_logits = self.lm_head(aux["_planned_next"][:, -1]).float()
                plan_logits[:, self.cfg.image_token_offset:] = -float("inf")
                next_logits = (1.0 - float(lookahead_blend)) * next_logits + float(lookahead_blend) * plan_logits

            base_for_difficulty = next_logits.clone()
            apply_repetition_penalty_(base_for_difficulty, idx, repetition_penalty)
            if no_repeat_ngram > 1:
                apply_no_repeat_ngram_(base_for_difficulty, idx, no_repeat_ngram)
            if generated < int(min_new_tokens):
                base_for_difficulty[:, int(eos_id)] = -float("inf")
            active_heads, dstat = self._nonseq_difficulty_from_logits(
                base_for_difficulty,
                aux,
                min_heads=min_heads,
                max_heads=min(max_heads, remaining),
                dynamic=dynamic,
                risk_threshold=risk_thr,
                contradiction_threshold=con_thr,
                repetition_threshold=rep_thr,
                entropy_easy=entropy_easy,
                entropy_hard=entropy_hard,
            )
            active_heads = max(1, min(int(active_heads), max_heads_avail, remaining))
            if active_heads <= 1:
                stats["nonseq_head1_steps"] += 1.0
            stats["nonseq_steps"] += 1.0
            for k, v in dstat.items():
                stats[k] = float(v)

            all_heads = self.mtp_logits_from_hidden(h_dec)
            head_logits: List[torch.Tensor] = []
            for i in range(active_heads):
                hlog = all_heads[i][:, -1, :].float().clone()
                hlog[:, self.cfg.image_token_offset:] = -float("inf")
                if i == 0:
                    hlog = next_logits.clone()
                head_logits.append(hlog)
            stats["nonseq_forward_draft_calls"] += 1.0
            stats["nonseq_avg_active_heads"] += float(active_heads)

            batch_n = idx.size(0)
            canvas = torch.zeros((batch_n, active_heads), dtype=torch.long, device=device)
            slot_conf = torch.zeros((batch_n, active_heads), dtype=next_logits.dtype, device=device)
            slot_entropy = torch.ones((batch_n, active_heads), dtype=next_logits.dtype, device=device)
            slot_gap = torch.zeros((batch_n, active_heads), dtype=next_logits.dtype, device=device)
            slot_filled = torch.zeros((batch_n, active_heads), dtype=torch.bool, device=device)
            compound_enabled_step = bool(base_compound_enabled and compound_mix > 0.0 and active_heads > 1)
            compound_state = self.planner.start_state(h_dec[:, -1:])[:, 0] if compound_enabled_step else None
            fill_order = self._dvsd_slot_generation_order(head_logits, getattr(self.cfg, "dvsd_slot_order", "confidence"))
            if fill_order != list(range(active_heads)):
                stats["nonseq_dvsd_nonmonotonic_slots"] = stats.get("nonseq_dvsd_nonmonotonic_slots", 0.0) + 1.0
            for fill_rank, pos in enumerate(fill_order):
                sample_logits = head_logits[pos]
                if compound_state is not None and fill_rank > 0:
                    comp_logits = self.lm_head(compound_state).float()
                    comp_logits[:, self.cfg.image_token_offset:] = -float("inf")
                    sample_logits = (1.0 - compound_mix) * sample_logits + compound_mix * comp_logits
                    head_logits[pos] = sample_logits
                    stats["nonseq_compound_logit_blends"] += 1.0
                # For constraints, expose only the durable left-context tokens that
                # have already been fixed. Later anchors can be filled first without
                # pretending that blank earlier slots are real context.
                left_fixed = [canvas[:, j:j + 1] for j in range(pos) if bool(slot_filled[:, j].all().detach().cpu())]
                prefix = idx if not left_fixed else torch.cat([idx, *left_fixed], dim=1)
                tok, conf, ent, gap = self._nonseq_sample_slot_token(
                    sample_logits,
                    prefix,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    typical_p=typical_p,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram=no_repeat_ngram,
                    forbid_eos=(generated + pos) < int(min_new_tokens),
                    eos_id=eos_id,
                )
                canvas[:, pos:pos + 1] = tok
                slot_conf[:, pos] = conf
                slot_entropy[:, pos] = ent
                slot_gap[:, pos] = gap
                slot_filled[:, pos] = True
                if compound_state is not None and fill_rank < active_heads - 1:
                    descent = self._dvsd_compound_descent_from_logits(sample_logits, tok, top_k=compound_top_k)
                    token_fb = self.token_emb(tok.squeeze(-1).clamp(min=0, max=self.cfg.vocab_size - 1))
                    compound_state = self.planner.compound_step(compound_state, token_fb, step_index=fill_rank + 1, logit_descent=descent, confidence=conf)
                    stats["nonseq_compound_steps"] += 1.0
                    stats["nonseq_compound_descent_norm"] += float(descent.float().pow(2).mean(dim=-1).sqrt().mean().detach().cpu())

            if dynamic and active_heads > int(min_heads):
                keep = active_heads
                for pos in range(int(min_heads), active_heads):
                    c = float(slot_conf[0, pos].detach().cpu())
                    e = float(slot_entropy[0, pos].detach().cpu())
                    if c < min_draft_prob or e > max_head_entropy:
                        keep = pos
                        break
                keep = max(int(min_heads), min(active_heads, keep))
                if keep < active_heads:
                    canvas = canvas[:, :keep]
                    slot_conf = slot_conf[:, :keep]
                    slot_entropy = slot_entropy[:, :keep]
                    head_logits = head_logits[:keep]
                    active_heads = keep

            # In DVSD freeze mode, generated virtual-slot tokens are fixed inside the
            # private block and committed directly.  Turning this off restores the
            # older slot-refinement behavior for ablations.
            if bool(getattr(self.cfg, "dvsd_freeze_sampled_slots", True)):
                refine_steps = 1
            # Easy spans need only one or two passes. Moderate spans can use the full
            # refinement budget. Hard spans have already collapsed to one slot.
            elif active_heads <= 1:
                refine_steps = 1
            elif dynamic:
                dscore = float(dstat.get("nonseq_difficulty_score", 0.5))
                if dscore <= float(entropy_easy):
                    refine_steps = max(1, refine_steps_cfg - 1)
                else:
                    refine_steps = refine_steps_cfg
            else:
                refine_steps = refine_steps_cfg
            refine_steps = max(1, int(refine_steps))
            locked = torch.zeros((1, active_heads), dtype=torch.bool, device=device)
            first_pass_keep = self._nonseq_refine_keep_count(active_heads, 1, refine_steps, lock_schedule)

            for r in range(refine_steps):
                if r > 0:
                    full = torch.cat([idx, canvas], dim=1)
                    ctx = full[:, -self.cfg.block_size:]
                    offset = full.size(1) - ctx.size(1)
                    cond = self._repeat_conditioning_for_batch(conditioning_vectors, full.size(0))
                    ref_logits, _, _ = self(ctx, return_hidden=True, conditioning_vectors=cond, conditioning_scale=conditioning_scale)
                    stats["nonseq_refine_forward_calls"] += 1.0
                    new_canvas = canvas.clone()
                    new_conf = slot_conf.clone()
                    new_entropy = slot_entropy.clone()
                    compound_state_ref = self.planner.start_state(h_dec[:, -1:])[:, 0] if compound_enabled_step else None
                    for pos in range(active_heads):
                        pred_abs = idx.size(1) + pos - 1
                        rel = pred_abs - offset
                        if 0 <= rel < ref_logits.size(1):
                            causal_logits = ref_logits[:, rel, :].float().clone()
                            causal_logits[:, self.cfg.image_token_offset:] = -float("inf")
                            mix = causal_blend * min(1.0, float(r) / max(1.0, float(refine_steps - 1)))
                            slot_logits = (1.0 - mix) * head_logits[pos] + mix * causal_logits
                        else:
                            slot_logits = head_logits[pos]
                        if compound_state_ref is not None and pos > 0:
                            comp_logits = self.lm_head(compound_state_ref).float()
                            comp_logits[:, self.cfg.image_token_offset:] = -float("inf")
                            slot_logits = (1.0 - compound_mix) * slot_logits + compound_mix * comp_logits
                            stats["nonseq_compound_logit_blends"] += 1.0
                        if bool(locked[0, pos].detach().cpu()) and not resample_locked:
                            tok = new_canvas[:, pos:pos + 1]
                            conf = new_conf[:, pos]
                        else:
                            prefix = torch.cat([idx, new_canvas[:, :pos]], dim=1) if pos > 0 else idx
                            local_temp = float(temperature) * (temp_decay ** r)
                            tok, conf, ent, _gap = self._nonseq_sample_slot_token(
                                slot_logits,
                                prefix,
                                temperature=local_temp,
                                top_k=top_k,
                                top_p=top_p,
                                typical_p=typical_p,
                                repetition_penalty=repetition_penalty,
                                no_repeat_ngram=no_repeat_ngram,
                                forbid_eos=(generated + pos) < int(min_new_tokens),
                                eos_id=eos_id,
                            )
                            if confidence_floor <= 0.0 or float(conf[0].detach().cpu()) >= confidence_floor or r == refine_steps - 1:
                                new_canvas[:, pos:pos + 1] = tok
                                new_conf[:, pos] = conf
                                new_entropy[:, pos] = ent
                            else:
                                tok = new_canvas[:, pos:pos + 1]
                                conf = new_conf[:, pos]
                        if compound_state_ref is not None and pos < active_heads - 1:
                            descent = self._dvsd_compound_descent_from_logits(slot_logits, tok, top_k=compound_top_k)
                            token_fb = self.token_emb(tok.squeeze(-1).clamp(min=0, max=self.cfg.vocab_size - 1))
                            compound_state_ref = self.planner.compound_step(compound_state_ref, token_fb, step_index=pos + 1, logit_descent=descent, confidence=conf)
                            stats["nonseq_compound_steps"] += 1.0
                            stats["nonseq_compound_descent_norm"] += float(descent.float().pow(2).mean(dim=-1).sqrt().mean().detach().cpu())
                    canvas = new_canvas
                    slot_conf = new_conf
                    slot_entropy = new_entropy

                keep_count = self._nonseq_refine_keep_count(active_heads, r + 1, refine_steps, lock_schedule)
                if r == refine_steps - 1:
                    locked[:] = True
                else:
                    top_pos = torch.topk(slot_conf, k=min(active_heads, keep_count), dim=1).indices
                    newly_locked = torch.zeros_like(locked)
                    newly_locked.scatter_(1, top_pos, True)
                    locked = locked | newly_locked

            commit_len = active_heads
            for pos in range(active_heads):
                if int(canvas[0, pos].item()) == int(eos_id) and (generated + pos + 1) >= int(min_new_tokens):
                    commit_len = pos + 1
                    break
            commit = canvas[:, :commit_len]
            idx = torch.cat([idx, commit], dim=1)
            generated += int(commit_len)
            stats["nonseq_drafted_tokens"] += float(active_heads)
            stats["nonseq_committed_tokens"] += float(commit_len)
            stats["nonseq_accepted_tokens"] += float(commit_len)
            stats["nonseq_avg_refine_steps"] += float(refine_steps)
            stats["nonseq_avg_slot_confidence"] += float(slot_conf[:, :commit_len].mean().detach().cpu()) if commit_len > 0 else 0.0
            stats["nonseq_avg_slot_entropy"] += float(slot_entropy[:, :commit_len].mean().detach().cpu()) if commit_len > 0 else 0.0
            stats["nonseq_avg_locked_first_pass"] += float(first_pass_keep) / max(1.0, float(active_heads))
            if bool((commit == int(eos_id)).any()) and generated >= int(min_new_tokens):
                break

        if stats["nonseq_steps"] > 0:
            denom = stats["nonseq_steps"]
            stats["nonseq_avg_active_heads"] /= denom
            stats["nonseq_avg_refine_steps"] /= denom
            stats["nonseq_avg_slot_confidence"] /= denom
            stats["nonseq_avg_slot_entropy"] /= denom
            stats["nonseq_avg_locked_first_pass"] /= denom
        stats["nonseq_accept_rate"] = 1.0
        stats["nonseq_commit_rate"] = float(stats["nonseq_committed_tokens"] / max(1.0, stats["nonseq_drafted_tokens"]))
        stats["nonseq_avg_compound_descent_norm"] = float(stats["nonseq_compound_descent_norm"] / max(1.0, stats["nonseq_compound_steps"]))
        stats["nonseq_compound_blend_rate"] = float(stats["nonseq_compound_logit_blends"] / max(1.0, stats["nonseq_drafted_tokens"]))
        if dvsd_planner_compound_step_size is not None:
            self.cfg.dvsd_planner_compound_step_size = old_step_size
        if dvsd_planner_compound_token_scale is not None:
            self.cfg.dvsd_planner_compound_token_scale = old_token_scale
        if dvsd_planner_compound_descent_scale is not None:
            self.cfg.dvsd_planner_compound_descent_scale = old_descent_scale
        return (idx, stats) if return_stats else idx


    @torch.no_grad()
    def generate_text_nonsequential_aux_scored(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        typical_p: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram: int = 0,
        min_new_tokens: int = 0,
        lookahead_blend: float = 0.0,
        aux_score_top_k: int = 0,
        aux_score_weight: float = 0.0,
        eos_id: Optional[int] = None,
        conditioning_vectors: Optional[torch.Tensor] = None,
        conditioning_scale: float = 0.0,
        proactive_aux_score: Optional[bool] = None,
        aux_score_risk_threshold: Optional[float] = None,
        aux_score_contradiction_threshold: Optional[float] = None,
        aux_score_candidate_pool: Optional[int] = None,
        aux_score_backtrack_window: Optional[int] = None,
        aux_score_max_backtracks: Optional[int] = None,
        latent_search_depth: Optional[int] = None,
        latent_search_branches: Optional[int] = None,
        nonseq_min_heads: Optional[int] = None,
        nonseq_max_heads: Optional[int] = None,
        nonseq_dynamic: Optional[bool] = None,
        nonseq_accept_top_k: Optional[int] = None,
        nonseq_accept_min_prob: Optional[float] = None,
        nonseq_accept_max_logprob_gap: Optional[float] = None,
        nonseq_risk_threshold: Optional[float] = None,
        nonseq_contradiction_threshold: Optional[float] = None,
        nonseq_repetition_threshold: Optional[float] = None,
        nonseq_entropy_easy: Optional[float] = None,
        nonseq_entropy_hard: Optional[float] = None,
        nonseq_min_draft_prob: Optional[float] = None,
        nonseq_max_head_entropy: Optional[float] = None,
        nonseq_tree_candidates: Optional[int] = None,
        nonseq_branch_top_k: Optional[int] = None,
        nonseq_aux_score_weight: Optional[float] = None,
        nonseq_fallback_to_ar: Optional[bool] = None,
        return_stats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """Dynamic multi-token prediction decoder with AR fallback.

        The decoder predicts a span with the MTP heads from one hidden state, builds a
        small nonsequential candidate tree, verifies each candidate span with NEED's
        normal causal logits/aux_score in one batched pass, and commits the longest safe
        prefix. If the current state is difficult, the active head count is reduced down
        to one; head-count one calls the normal AR generator for a single token.
        """
        device = next(self.parameters()).device
        idx = input_ids.to(device=device, dtype=torch.long)
        if idx.size(0) != 1:
            out = self.generate_text(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                typical_p=typical_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram=no_repeat_ngram,
                min_new_tokens=min_new_tokens,
                lookahead_blend=lookahead_blend,
                aux_score_top_k=aux_score_top_k,
                aux_score_weight=aux_score_weight,
                eos_id=eos_id,
                conditioning_vectors=conditioning_vectors,
                conditioning_scale=conditioning_scale,
                proactive_aux_score=proactive_aux_score,
                aux_score_risk_threshold=aux_score_risk_threshold,
                aux_score_contradiction_threshold=aux_score_contradiction_threshold,
                aux_score_candidate_pool=aux_score_candidate_pool,
                aux_score_backtrack_window=aux_score_backtrack_window,
                aux_score_max_backtracks=aux_score_max_backtracks,
                latent_search_depth=latent_search_depth,
                latent_search_branches=latent_search_branches,
            )
            stats = {"nonseq_decode": 0.0, "nonseq_batch_fallback": 1.0}
            return (out, stats) if return_stats else out
        eos_id = self.cfg.eos_id if eos_id is None else int(eos_id)
        max_heads_avail = max(1, int(self.cfg.n_predict_heads))
        max_heads = int(self.cfg.nonseq_max_heads if nonseq_max_heads is None else nonseq_max_heads)
        min_heads = int(self.cfg.nonseq_min_heads if nonseq_min_heads is None else nonseq_min_heads)
        min_heads = max(1, min(min_heads, max_heads_avail))
        max_heads = max(min_heads, min(max_heads, max_heads_avail, int(max_new_tokens) if max_new_tokens > 0 else 1))
        dynamic = bool(self.cfg.nonseq_dynamic if nonseq_dynamic is None else nonseq_dynamic)
        accept_top_k = int(self.cfg.nonseq_accept_top_k if nonseq_accept_top_k is None else nonseq_accept_top_k)
        accept_min_prob = float(self.cfg.nonseq_accept_min_prob if nonseq_accept_min_prob is None else nonseq_accept_min_prob)
        accept_gap = float(self.cfg.nonseq_accept_max_logprob_gap if nonseq_accept_max_logprob_gap is None else nonseq_accept_max_logprob_gap)
        risk_thr = float(self.cfg.nonseq_risk_threshold if nonseq_risk_threshold is None else nonseq_risk_threshold)
        con_thr = float(self.cfg.nonseq_contradiction_threshold if nonseq_contradiction_threshold is None else nonseq_contradiction_threshold)
        rep_thr = float(self.cfg.nonseq_repetition_threshold if nonseq_repetition_threshold is None else nonseq_repetition_threshold)
        entropy_easy = float(self.cfg.nonseq_entropy_easy if nonseq_entropy_easy is None else nonseq_entropy_easy)
        entropy_hard = float(self.cfg.nonseq_entropy_hard if nonseq_entropy_hard is None else nonseq_entropy_hard)
        min_draft_prob = float(self.cfg.nonseq_min_draft_prob if nonseq_min_draft_prob is None else nonseq_min_draft_prob)
        max_head_entropy = float(self.cfg.nonseq_max_head_entropy if nonseq_max_head_entropy is None else nonseq_max_head_entropy)
        tree_candidates = int(self.cfg.nonseq_tree_candidates if nonseq_tree_candidates is None else nonseq_tree_candidates)
        branch_top_k = int(self.cfg.nonseq_branch_top_k if nonseq_branch_top_k is None else nonseq_branch_top_k)
        nonseq_vw = float(self.cfg.nonseq_aux_score_weight if nonseq_aux_score_weight is None else nonseq_aux_score_weight)
        fallback_to_ar = bool(self.cfg.nonseq_fallback_to_ar if nonseq_fallback_to_ar is None else nonseq_fallback_to_ar)
        proactive = self.cfg.aux_score_proactive if proactive_aux_score is None else bool(proactive_aux_score)
        generated = 0
        stats: Dict[str, float] = {
            "nonseq_decode": 1.0,
            "nonseq_steps": 0.0,
            "nonseq_forward_draft_calls": 0.0,
            "nonseq_validate_calls": 0.0,
            "nonseq_drafted_tokens": 0.0,
            "nonseq_accepted_tokens": 0.0,
            "nonseq_rejected_tokens": 0.0,
            "nonseq_ar_fallback_tokens": 0.0,
            "nonseq_head1_steps": 0.0,
            "nonseq_avg_active_heads": 0.0,
            "nonseq_avg_candidates": 0.0,
            "nonseq_last_reject_reason_code": 0.0,
        }
        reason_codes = {
            "none": 0.0,
            "need_distribution_disagreement": 1.0,
            "aux_score_risk": 2.0,
            "aux_score_contradiction": 3.0,
            "aux_score_repetition": 4.0,
            "non_text_token": 5.0,
            "context_cropped": 6.0,
            "empty_or_batch_not_supported": 7.0,
            "no_budget": 8.0,
        }
        while generated < int(max_new_tokens):
            remaining = int(max_new_tokens) - generated
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _, aux = self(idx_cond, return_hidden=True, conditioning_vectors=conditioning_vectors, conditioning_scale=conditioning_scale)
            h_dec = aux["_hidden"]
            next_logits = logits[:, -1, :].float().clone()
            next_logits[:, self.cfg.image_token_offset:] = -float("inf")
            if lookahead_blend > 0 and "_planned_next" in aux:
                plan_logits = self.lm_head(aux["_planned_next"][:, -1]).float()
                plan_logits[:, self.cfg.image_token_offset:] = -float("inf")
                next_logits = (1.0 - lookahead_blend) * next_logits + float(lookahead_blend) * plan_logits
            base_for_difficulty = next_logits.clone()
            apply_repetition_penalty_(base_for_difficulty, idx, repetition_penalty)
            if no_repeat_ngram > 1:
                apply_no_repeat_ngram_(base_for_difficulty, idx, no_repeat_ngram)
            if generated < int(min_new_tokens):
                base_for_difficulty[:, eos_id] = -float("inf")
            active_heads, dstat = self._nonseq_difficulty_from_logits(
                base_for_difficulty,
                aux,
                min_heads=min_heads,
                max_heads=min(max_heads, remaining),
                dynamic=dynamic,
                risk_threshold=risk_thr,
                contradiction_threshold=con_thr,
                repetition_threshold=rep_thr,
                entropy_easy=entropy_easy,
                entropy_hard=entropy_hard,
            )
            active_heads = max(1, min(active_heads, max_heads_avail, remaining))
            stats["nonseq_steps"] += 1.0
            for k, v in dstat.items():
                stats[k] = float(v)
            if active_heads <= 1 and fallback_to_ar:
                stats["nonseq_avg_active_heads"] += float(active_heads)
                stats["nonseq_head1_steps"] += 1.0
                out = self.generate_text(
                    idx,
                    max_new_tokens=1,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    typical_p=typical_p,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram=no_repeat_ngram,
                    min_new_tokens=1 if generated < int(min_new_tokens) else 0,
                    lookahead_blend=lookahead_blend,
                    aux_score_top_k=aux_score_top_k,
                    aux_score_weight=aux_score_weight,
                    eos_id=eos_id,
                    conditioning_vectors=conditioning_vectors,
                    conditioning_scale=conditioning_scale,
                    proactive_aux_score=proactive,
                    aux_score_risk_threshold=aux_score_risk_threshold,
                    aux_score_contradiction_threshold=aux_score_contradiction_threshold,
                    aux_score_candidate_pool=aux_score_candidate_pool,
                    aux_score_backtrack_window=aux_score_backtrack_window,
                    aux_score_max_backtracks=0,
                    latent_search_depth=latent_search_depth,
                    latent_search_branches=latent_search_branches,
                )
                if out.size(1) <= idx.size(1):
                    break
                one = out[:, idx.size(1):idx.size(1) + 1]
                idx = torch.cat([idx, one], dim=1)
                generated += int(one.size(1))
                stats["nonseq_ar_fallback_tokens"] += float(one.size(1))
                if generated >= int(min_new_tokens) and bool((one == eos_id).all()):
                    break
                continue
            all_heads = self.mtp_logits_from_hidden(h_dec)
            last_head_logits = [all_heads[i][:, -1, :].clone() for i in range(min(active_heads, len(all_heads)))]
            if lookahead_blend > 0 and "_planned_next" in aux and last_head_logits:
                last_head_logits[0] = next_logits
            drafts, draft_lps, active_heads, tstat = self._nonseq_build_draft_tree(
                last_head_logits,
                idx,
                generated_so_far=generated,
                min_new_tokens=min_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                typical_p=typical_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram=no_repeat_ngram,
                eos_id=eos_id,
                branch_top_k=branch_top_k,
                max_candidates=tree_candidates,
                dynamic=dynamic,
                min_heads=min_heads,
                min_draft_prob=min_draft_prob,
                max_head_entropy=max_head_entropy,
            )
            stats["nonseq_forward_draft_calls"] += 1.0
            stats["nonseq_avg_active_heads"] += float(active_heads)
            stats["nonseq_drafted_tokens"] += float(drafts.size(1))
            stats["nonseq_avg_candidates"] += float(drafts.size(0))
            for k, v in tstat.items():
                stats[k] = float(v)
            accepted, accepted_tokens, vstat = self._validate_nonseq_draft_batch(
                idx,
                drafts,
                draft_lps,
                generated_so_far=generated,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram=no_repeat_ngram,
                accept_top_k=accept_top_k,
                accept_min_prob=accept_min_prob,
                accept_max_logprob_gap=accept_gap,
                risk_threshold=risk_thr,
                contradiction_threshold=con_thr,
                repetition_threshold=rep_thr,
                aux_score_weight=nonseq_vw,
                eos_id=eos_id,
                conditioning_vectors=conditioning_vectors,
                conditioning_scale=conditioning_scale,
            )
            stats["nonseq_validate_calls"] += 1.0
            stats["nonseq_accepted_tokens"] += float(accepted)
            stats["nonseq_rejected_tokens"] += float(max(0, drafts.size(1) - accepted))
            reason = str(vstat.get("reject_reason", "none"))
            stats["nonseq_last_reject_reason_code"] = float(reason_codes.get(reason, 99.0))
            for key in ("avg_need_prob", "avg_logprob_gap", "avg_risk", "avg_contradiction", "avg_repetition"):
                if key in vstat:
                    stats["nonseq_" + key] = float(vstat[key])
            if accepted > 0:
                idx = torch.cat([idx, accepted_tokens], dim=1)
                generated += int(accepted)
                if generated >= int(min_new_tokens) and bool((accepted_tokens == eos_id).any()):
                    break
                continue
            if not fallback_to_ar:
                break
            out = self.generate_text(
                idx,
                max_new_tokens=1,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                typical_p=typical_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram=no_repeat_ngram,
                min_new_tokens=1 if generated < int(min_new_tokens) else 0,
                lookahead_blend=lookahead_blend,
                aux_score_top_k=aux_score_top_k,
                aux_score_weight=aux_score_weight,
                eos_id=eos_id,
                conditioning_vectors=conditioning_vectors,
                conditioning_scale=conditioning_scale,
                proactive_aux_score=proactive,
                aux_score_risk_threshold=aux_score_risk_threshold,
                aux_score_contradiction_threshold=aux_score_contradiction_threshold,
                aux_score_candidate_pool=aux_score_candidate_pool,
                aux_score_backtrack_window=aux_score_backtrack_window,
                aux_score_max_backtracks=0,
                latent_search_depth=latent_search_depth,
                latent_search_branches=latent_search_branches,
            )
            if out.size(1) <= idx.size(1):
                break
            one = out[:, idx.size(1):idx.size(1) + 1]
            idx = torch.cat([idx, one], dim=1)
            generated += int(one.size(1))
            stats["nonseq_ar_fallback_tokens"] += float(one.size(1))
            if generated >= int(min_new_tokens) and bool((one == eos_id).all()):
                break
        if stats["nonseq_steps"] > 0:
            stats["nonseq_avg_active_heads"] /= stats["nonseq_steps"]
            stats["nonseq_avg_candidates"] /= stats["nonseq_steps"]
        if stats["nonseq_drafted_tokens"] > 0:
            stats["nonseq_accept_rate"] = float(stats["nonseq_accepted_tokens"] / max(1.0, stats["nonseq_drafted_tokens"]))
        else:
            stats["nonseq_accept_rate"] = 0.0
        return (idx, stats) if return_stats else idx



    def _streaming_text_supported(
        self,
        input_ids: torch.Tensor,
        conditioning_vectors: Optional[torch.Tensor],
        conditioning_scale: float,
        proactive: bool,
        aux_score_top_k: int,
        aux_score_weight: float,
        cand_pool: int,
        rollout_depth: int,
        search_branches: int,
        max_backtracks: int,
    ) -> Tuple[bool, str]:
        """Return whether the stateful generation cache can serve this request."""
        if not bool(getattr(self.cfg, "streaming_generation", True)):
            return False, "streaming_generation disabled"
        if input_ids.ndim != 2 or input_ids.size(1) <= 0:
            return False, "empty or invalid input"
        if conditioning_vectors is not None and float(conditioning_scale) != 0.0:
            return False, "conditioning vectors require full-context pathway conditioning"
        if bool((input_ids >= int(self.cfg.image_token_offset)).any().detach().cpu()):
            return False, "image tokens require multimodal full-context path"
        image_special = (input_ids == int(self.cfg.img_bos_id)) | (input_ids == int(self.cfg.img_eos_id)) | (input_ids == int(self.cfg.img_mask_id))
        if bool(image_special.any().detach().cpu()):
            return False, "image span tokens require multimodal full-context path"
        rerank = bool(proactive) or int(aux_score_top_k) > 0 or float(aux_score_weight) > 0.0
        if rerank:
            return False, "aux-score/proactive reranking uses full-context scoring"
        if int(rollout_depth) > 0 or int(search_branches) > 1 or int(max_backtracks) > 0:
            return False, "rollout/backtracking search uses full-context scoring"
        if str(getattr(self.cfg, "slot_attention_mode", "pooled")).lower() != "pooled":
            return False, "attention slot ablation requires full-context token-slot attention"
        return True, "streaming text core"

    def _stream_new_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
        return {
            "seen": 0,
            "last_token": None,
            "block_states": [dict() for _ in range(len(self.blocks))],
            "recall_state": self.exact_recall.init_stream_state(batch_size, device, dtype),
            "timescale_med_h": None,
            "timescale_slow_h": None,
            "timescale_chunk_sum": torch.zeros(batch_size, self.cfg.d_model, device=device, dtype=dtype),
            "timescale_chunk_count": 0,
            "mean_pre_slot": torch.zeros(batch_size, self.cfg.d_model, device=device, dtype=dtype),
            "mean_slot": torch.zeros(batch_size, self.cfg.d_model, device=device, dtype=dtype),
            "mean_count": 0,
            "drift_state": {},
        }

    @staticmethod
    def _stream_update_mean(mean: torch.Tensor, count: int, value: torch.Tensor) -> Tuple[torch.Tensor, int]:
        new_count = int(count) + 1
        mean = mean + (value - mean) / float(new_count)
        return mean, new_count

    @staticmethod
    def _stream_append_limited(old: Optional[torch.Tensor], new: torch.Tensor, max_len: int) -> torch.Tensor:
        new = new.detach()
        if old is None or old.numel() == 0:
            out = new
        else:
            out = torch.cat([old, new], dim=1)
        if out.size(1) > int(max_len):
            out = out[:, -int(max_len):]
        return out.detach()

    def _stream_depthwise_conv1d_step(
        self,
        conv: nn.Conv1d,
        x_t: torch.Tensor,
        hist: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One causal grouped Conv1d step matching Conv1d(padding=k-1)[..., :T]."""
        b, d = x_t.shape
        k = int(conv.kernel_size[0])
        if k <= 1:
            full = x_t.unsqueeze(-1)
            new_hist = x_t.new_zeros(b, d, 0)
        else:
            if hist is None or hist.size(0) != b or hist.size(1) != d or hist.size(2) != k - 1:
                hist = x_t.new_zeros(b, d, k - 1)
            full = torch.cat([hist.to(dtype=x_t.dtype, device=x_t.device), x_t.unsqueeze(-1)], dim=-1)
            new_hist = full[..., 1:].detach()
        weight = conv.weight.squeeze(1).to(device=x_t.device, dtype=x_t.dtype)
        y = (full * weight.unsqueeze(0)).sum(dim=-1)
        if conv.bias is not None:
            y = y + conv.bias.to(device=x_t.device, dtype=x_t.dtype).view(1, -1)
        return y, new_hist

    def _stream_retention_step(self, module: nn.Module, x_t: torch.Tensor, state: Dict[str, Any]) -> torch.Tensor:
        b, d = x_t.shape
        if isinstance(module, StructuredDualRetention):
            u, b_gate, c_gate, dt_raw, out_gate = module.in_proj(x_t.unsqueeze(1)).squeeze(1).chunk(5, dim=-1)
            u_conv, hist = self._stream_depthwise_conv1d_step(module.dwconv, u, state.get("ssd_conv_hist"))
            state["ssd_conv_hist"] = hist
            u_act = F.silu(u_conv)
            dt = torch.sigmoid(dt_raw.float())
            dt = module.cfg.ssd_dt_min + (module.cfg.ssd_dt_max - module.cfg.ssd_dt_min) * dt
            decay = torch.exp(-dt).to(x_t.dtype)
            bg = torch.sigmoid(b_gate).to(x_t.dtype)
            cg = torch.sigmoid(c_gate).to(x_t.dtype)
            og = torch.sigmoid(out_gate).to(x_t.dtype)
            rec = state.get("ssd_state")
            if rec is None or rec.shape != x_t.shape:
                rec = torch.zeros_like(x_t)
            rec = decay * rec + (1.0 - decay) * (bg * u_act)
            state["ssd_state"] = rec.detach()
            y = (cg * rec + module.D.to(x_t.dtype).view(1, -1) * u_act) * og
            return module.out(module.out_norm(y.unsqueeze(1))).squeeze(1)
        if isinstance(module, SelectiveRetention):
            q, k, v = module.in_proj(x_t.unsqueeze(1)).chunk(3, dim=-1)
            q = q.view(b, 1, module.heads, module.head_dim).transpose(1, 2).squeeze(2)
            k = k.view(b, 1, module.heads, module.head_dim).transpose(1, 2).squeeze(2)
            v = v.view(b, 1, module.heads, module.head_dim).transpose(1, 2).squeeze(2)
            dyn = torch.sigmoid(module.decay_proj(x_t)).unsqueeze(-1)
            base = module.base_decay.squeeze(2).to(device=x_t.device, dtype=x_t.dtype)
            decay = (base + module.cfg.retention_dynamic_scale * (dyn - 0.5) * 0.05).clamp(0.05, 0.999)
            rec = state.get("ret_state")
            if rec is None or rec.shape != k.shape:
                rec = torch.zeros_like(k)
            rec = decay * rec + k * v
            state["ret_state"] = rec.detach()
            y = (q * rec).transpose(1, 2).reshape(b, d)
            return module.out(y.unsqueeze(1)).squeeze(1)
        # Unknown custom retention implementation: fall back to a one-token forward.
        return module(x_t.unsqueeze(1)).squeeze(1)

    def _stream_multiscale_conv_step(self, module: MultiScaleCausalConv, x_t: torch.Tensor, state: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Delegate to the module's causal one-token update.  Earlier streaming
        # code selected active scales from a running route average; that made the
        # streamed model diverge from causal full forward and could change a
        # token's scale choice based on later tokens.
        y, aux = module.stream_step(x_t.unsqueeze(1), state)
        return y.squeeze(1), aux

    def _stream_memory_step(
        self,
        module: HierarchicalMemory,
        x_t: torch.Tensor,
        state: Dict[str, Any],
        condition: Optional[torch.Tensor] = None,
        innovation: Optional[torch.Tensor] = None,
        write_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Streaming wrapper for the conditioned explicit memory module."""
        x1 = x_t.unsqueeze(1)
        if condition is not None and condition.ndim == 2:
            condition = condition.unsqueeze(1)
        if innovation is not None and innovation.ndim == 2:
            innovation = innovation.unsqueeze(1)
        if write_mask is not None and write_mask.ndim == 2:
            write_mask = write_mask.unsqueeze(1)
        mem_state = state.get("mem_state")
        if not isinstance(mem_state, dict):
            mem_state = module.init_stream_state(x_t.size(0), x_t.device, x_t.dtype)
            state["mem_state"] = mem_state
        y, aux = module.stream_step(x1, mem_state, condition=condition, innovation=innovation, write_mask=write_mask)
        return y.squeeze(1), aux


    def _stream_moe_step(self, module: SparseMoE, x_t: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Sparse MoE for one streaming token without scatter/index_add overhead."""
        if module.cfg.n_experts <= 1 and module.shared is None:
            out = module.experts[0](x_t)
            z = x_t.new_tensor(0.0)
            return out, {"moe_balance": z, "branch_entropy": z, "moe_router_z": z}
        logits = module.router(x_t.float())
        k = min(int(module.cfg.moe_top_k), int(module.cfg.n_experts))
        vals, idx = torch.topk(logits, k=k, dim=-1)
        weights = F.softmax(vals, dim=-1).to(x_t.dtype)
        if bool(getattr(module.cfg, "moe_static_dispatch", False)):
            combine_w = torch.zeros_like(logits).scatter(-1, idx, weights.to(logits.dtype)).to(x_t.dtype)
            expert_outputs = torch.stack([expert(x_t).to(x_t.dtype) for expert in module.experts], dim=1)  # [B,E,D]
            out = (expert_outputs * combine_w.unsqueeze(-1)).sum(dim=1)
        else:
            out = torch.zeros_like(x_t)
            # Legacy sparse route for opt-in single-token inference.
            for expert_id, expert in enumerate(module.experts):
                route_mask = idx == expert_id
                rows = route_mask.any(dim=-1).nonzero(as_tuple=False).flatten()
                if rows.numel() == 0:
                    continue
                expert_in = x_t.index_select(0, rows)
                expert_out = expert(expert_in).to(out.dtype)
                expert_w = torch.where(
                    route_mask.index_select(0, rows),
                    weights.index_select(0, rows),
                    torch.zeros_like(weights.index_select(0, rows)),
                ).sum(dim=-1, keepdim=True).to(out.dtype)
                out.index_add_(0, rows, expert_out * expert_w)
        if module.shared is not None:
            out = out + 0.25 * module.shared(x_t)
        probs = F.softmax(logits, dim=-1)
        load = probs.mean(dim=0)
        balance = (load * load).sum() * module.cfg.n_experts
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        z_loss = logits.pow(2).mean()
        return out, {"moe_balance": balance, "branch_entropy": entropy, "moe_router_z": z_loss}

    def _stream_block_step(self, block: NEEDBlock, x_t: torch.Tensor, state: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        aux: Dict[str, torch.Tensor] = {}
        x = x_t.unsqueeze(1)
        summaries: List[torch.Tensor] = []
        actual_deltas: List[torch.Tensor] = []
        step_gates: List[torch.Tensor] = []

        ctx = block.coop.read_context(x, summaries, 0)
        y = self._stream_retention_step(block.ret, block.n1(x + ctx).squeeze(1), state).unsqueeze(1)
        coop_gate = block.coop.contribution_gate(x, y, ctx, 0)
        actual = block._residual_delta(y, coop_gate, block.scales[0])
        x = x + actual
        block._record_coop_step(0, actual, coop_gate, summaries, actual_deltas, step_gates)

        ctx = block.coop.read_context(x, summaries, 1)
        y2, caux = self._stream_multiscale_conv_step(block.conv, block.n2(x + ctx).squeeze(1), state)
        aux.update(caux)
        y = y2.unsqueeze(1)
        coop_gate = block.coop.contribution_gate(x, y, ctx, 1)
        actual = block._residual_delta(y, coop_gate, block.scales[1])
        x = x + actual
        block._record_coop_step(1, actual, coop_gate, summaries, actual_deltas, step_gates)

        gates, gaux = block.depth_gate(x)
        aux.update(gaux)
        mem_gate, eq_gate, moe_gate = gates[..., 0:1], gates[..., 1:2], gates[..., 2:3]
        hard_skip = bool(getattr(block.cfg, "adaptive_depth_hard_skip", True)) and not torch.is_grad_enabled()
        threshold = float(getattr(block.cfg, "adaptive_depth_skip_threshold", 0.0))

        temporal_delta = (actual_deltas[0] + actual_deltas[1]) if len(actual_deltas) >= 2 else torch.zeros_like(x)
        memory_actual = torch.zeros_like(x)
        memory_enabled = float(getattr(block.cfg, "memory_mix", 0.35)) > 0.0
        mem_base_gate, mem_active_fraction, mem_active_mask = block._adaptive_depth_gate(mem_gate, hard_skip, threshold)
        if memory_enabled:
            ctx = block.coop.read_context(x, summaries, 2)
            mem_input = block.n3(x + ctx)
            mem_innovation = block._separate_from(mem_input, temporal_delta)
            y2, maux = self._stream_memory_step(
                block.mem,
                mem_input.squeeze(1),
                state,
                condition=temporal_delta.squeeze(1),
                innovation=mem_innovation.squeeze(1),
                write_mask=mem_active_mask.squeeze(1),
            )
            aux.update(maux)
            y = block._separate_from(y2.unsqueeze(1), temporal_delta)
            aux["memory_retention_overlap"] = block._overlap_penalty(y, temporal_delta, mem_base_gate)
            eff_gate = block.coop.contribution_gate(x, y, ctx, 2, mem_base_gate)
            mem_scale = block.scales[2] * x.new_tensor(float(getattr(block.cfg, "memory_mix", 0.35)))
            actual = block._residual_delta(y, eff_gate, mem_scale)
            memory_actual = actual
            x = x + actual
            block._record_coop_step(2, actual, eff_gate, summaries, actual_deltas, step_gates)
            aux["memory_hard_skipped"] = (1.0 - mem_active_fraction).detach()
        else:
            mem_active_fraction = x.new_tensor(0.0)
            aux["memory_hard_skipped"] = x.new_tensor(1.0)
            aux["memory_retention_overlap"] = x.new_tensor(0.0)
            zgate = block._zero_step_gate(x)
            zdelta = torch.zeros_like(x)
            block._record_coop_step(2, zdelta, zgate, summaries, actual_deltas, step_gates)

        eq_base_gate, eq_active_fraction, _ = block._adaptive_depth_gate(eq_gate, hard_skip, threshold)
        ctx = block.coop.read_context(x, summaries, 3)
        eq_context = x + ctx
        n4x = block.n4(eq_context)
        z, resid, energy, effort = block.eq(n4x, eq_context)
        y = block._separate_from(z - n4x, temporal_delta + memory_actual)
        aux["equilibrium_temporal_overlap"] = block._overlap_penalty(y, temporal_delta + memory_actual, eq_base_gate)
        aux["equilibrium_active_fraction"] = eq_active_fraction
        aux["equilibrium_hard_skipped"] = (1.0 - eq_active_fraction).detach()
        eff_gate = block.coop.contribution_gate(x, y, ctx, 3, eq_base_gate)
        actual = block._residual_delta(y, eff_gate, block.scales[3])
        x = x + actual
        block._record_coop_step(3, actual, eff_gate, summaries, actual_deltas, step_gates)

        moe_base_gate, moe_active_fraction, _ = block._adaptive_depth_gate(moe_gate, hard_skip, threshold)
        ctx = block.coop.read_context(x, summaries, 4)
        n5x = block.n5(x + ctx).squeeze(1)
        y2, eaux = self._stream_moe_step(block.moe, n5x)
        aux.update(eaux)
        aux["moe_active_fraction"] = moe_active_fraction
        aux["moe_hard_skipped"] = (1.0 - moe_active_fraction).detach()
        y = y2.unsqueeze(1)
        eff_gate = block.coop.contribution_gate(x, y, ctx, 4, moe_base_gate)
        actual = block._residual_delta(y, eff_gate, block.scales[4])
        x = x + actual
        block._record_coop_step(4, actual, eff_gate, summaries, actual_deltas, step_gates)

        x, coop_aux = block.coop.finish(x, actual_deltas, step_gates, summaries)
        aux.update(coop_aux)
        aux["adaptive_depth_mem_run"] = mem_active_fraction.detach()
        aux["adaptive_depth_eq_run"] = eq_active_fraction.detach()
        aux["adaptive_depth_moe_run"] = moe_active_fraction.detach()
        aux["equilibrium_residual"] = resid
        aux["energy"] = energy
        aux["adaptive_effort"] = effort
        aux["energy_row_orth"] = block.eq.energy.row_orth_loss()
        return x.squeeze(1), aux

    def _stream_exact_recall_step(self, h_t: torch.Tensor, token_t: torch.Tensor, cache: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        state = cache.get("recall_state")
        if not isinstance(state, dict):
            state = self.exact_recall.init_stream_state(h_t.size(0), h_t.device, h_t.dtype)
            cache["recall_state"] = state
        y, aux = self.exact_recall.stream_step(h_t.unsqueeze(1), token_t.view(-1, 1), state)
        cache["last_token"] = token_t.detach().to(device=h_t.device, dtype=torch.long)
        return y.squeeze(1), aux

    def _stream_drift_step(self, h_t: torch.Tensor, base_t: torch.Tensor, cache: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        module = self.drift_stabilizer
        state = cache.setdefault("drift_state", {})
        if not isinstance(state, dict):
            state = {}
            cache["drift_state"] = state
        y, aux = module.stream_step(h_t.unsqueeze(1), base_t.unsqueeze(1), state)
        return y.squeeze(1), aux

    def _stream_timescale_step(self, h_t: torch.Tensor, cache: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        module = self.timescales
        state = cache.setdefault("timescale_state", module.init_stream_state(h_t.size(0), h_t.device, h_t.dtype))
        y, aux = module.step(h_t.unsqueeze(1), state)
        return y.squeeze(1), aux

    def _stream_latent_slots_step(self, h_t: torch.Tensor, cache: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        module = self.latent_slot_attention.block
        mean, count = self._stream_update_mean(cache["mean_pre_slot"], int(cache.get("mean_count", 0)), h_t)
        cache["mean_pre_slot"] = mean.detach()
        cache["mean_count"] = count
        pooled = module.base(mean.float())
        seeds = module.seed.to(h_t.dtype).unsqueeze(0).expand(h_t.size(0), -1, -1) + module.base_proj(pooled).unsqueeze(1).to(h_t.dtype)
        slots = module.token_out(seeds)
        ctx = slots.mean(dim=1)
        delta = module.token_out(ctx).to(h_t.dtype)
        norm_slots = F.normalize(slots.float(), dim=-1)
        gram = torch.einsum('bsd,bqd->bsq', norm_slots, norm_slots)
        ident = torch.eye(slots.size(1), device=h_t.device, dtype=gram.dtype).unsqueeze(0)
        div = (gram - ident).pow(2).mean()
        z = h_t.new_tensor(0.0)
        return delta, slots, {"latent_slot_attention_entropy": z, "latent_slot_coverage": z, "latent_slot_diversity": div, "slot_pooled_core": h_t.new_tensor(1.0)}

    def _stream_mixture_router_step(self, h_t: torch.Tensor, cache: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        module = self.mixture_energy_router
        mean_slot, _ = self._stream_update_mean(cache["mean_slot"], int(cache.get("mean_count", 1)) - 1, h_t)
        cache["mean_slot"] = mean_slot.detach()
        route_logits = module.router(mean_slot.float())
        route = F.softmax(route_logits, dim=-1).to(h_t.dtype)
        centers = torch.einsum('bn,nd->bd', route, module.centers.to(h_t.dtype))
        precision = F.softplus(torch.einsum('bn,nd->bd', route.float(), module.precision.float())).to(h_t.dtype) + self.cfg.min_precision
        rows = torch.einsum('bn,nrd->brd', route.float(), F.normalize(module.rows.float(), dim=-1)).to(h_t.dtype) * self.cfg.energy_row_norm
        z = h_t
        energy_total = h_t.new_tensor(0.0)
        for _ in range(max(1, int(self.cfg.energy_route_steps))):
            diff = z - centers
            proj = torch.einsum('bd,brd->br', diff, rows)
            grad = precision * diff + torch.einsum('br,brd->bd', proj, rows)
            z = z - float(self.cfg.energy_step_size) * float(self.cfg.energy_route_strength) * grad
            energy_total = energy_total + (0.5 * (precision * diff.pow(2)).sum(dim=-1) + 0.5 * proj.pow(2).sum(dim=-1)).mean()
        route_ent = -(route.float() * route.float().clamp_min(1e-8).log()).sum(dim=-1).mean()
        route_balance = route.float().mean(dim=0).pow(2).sum() * module.n
        return module.out(z - h_t), {"mixture_energy_router_energy": energy_total, "energy_route_entropy": route_ent, "energy_route_balance": route_balance}

    def _stream_step(self, token_t: torch.Tensor, cache: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = next(self.parameters()).device
        token_t = token_t.to(device=device, dtype=torch.long).view(-1)
        b = token_t.size(0)
        seen = int(cache.get("seen", 0))
        if str(getattr(self.cfg, "streaming_position_mode", "clamp")).lower() == "clamp":
            pos_id = min(seen, int(self.cfg.block_size) - 1)
        else:
            pos_id = seen % int(self.cfg.block_size)
        pos = torch.full((b,), pos_id, device=device, dtype=torch.long)
        mod = self.modality_ids_from_tokens(token_t.view(b, 1)).view(b)
        x_t = self.token_emb(token_t) + self.pos_emb(pos) + self.modality_emb(mod.clamp(0, 3))
        base_t = self.drop(x_t.unsqueeze(1)).squeeze(1)
        x_t = base_t
        aux_lists: Dict[str, List[torch.Tensor]] = {}
        for i, block in enumerate(self.blocks):
            x_t, aux = self._stream_block_step(block, x_t, cache["block_states"][i])
            for k, v in aux.items():
                aux_lists.setdefault(k, []).append(v if torch.is_tensor(v) else x_t.new_tensor(float(v)))
        x_t, raux = self._stream_exact_recall_step(x_t, token_t, cache)
        for k, v in raux.items():
            aux_lists.setdefault(k, []).append(v)
        x_t, saux = self._stream_drift_step(x_t, base_t, cache)
        for k, v in saux.items():
            aux_lists.setdefault(k, []).append(v)
        h_t = self.norm(x_t.unsqueeze(1)).squeeze(1)
        h_t, taux = self._stream_timescale_step(h_t, cache)
        latent_slot_delta, latent_slots, gaux = self._stream_latent_slots_step(h_t, cache)
        h_t = h_t + float(self.cfg.latent_slot_conditioning_scale) * latent_slot_delta
        router_delta, man_aux = self._stream_mixture_router_step(h_t, cache)
        h_t = h_t + router_delta
        vf = self.aux_score(h_t.unsqueeze(1)).squeeze(1)
        risk = F.softplus(vf[..., 1:2])
        latent_divergence = self.latent_divergence(h_t.unsqueeze(1), latent_slots).squeeze(1)
        risk_signal, uaux = self.risk_signal_fusion(h_t.unsqueeze(1), risk.unsqueeze(1).clamp(max=8.0) / 8.0, latent_divergence.unsqueeze(1))
        risk_signal_t = risk_signal.squeeze(1)
        revision_gate = torch.sigmoid(vf[..., 2:3] + float(self.cfg.risk_gate_strength) * risk_signal_t)
        h_dec = h_t + self.cfg.aux_score_logit_scale * revision_gate * self.revision_proj(h_t)
        logits = self.lm_head(h_dec)
        plan = self.planner(h_dec.unsqueeze(1))
        aux_mean = {k: torch.stack(v).mean() for k, v in aux_lists.items() if v}
        aux_mean.update(taux); aux_mean.update(gaux); aux_mean.update(man_aux); aux_mean.update(uaux)
        aux_mean["streaming_cache_active"] = h_t.new_tensor(1.0)
        aux_mean["_hidden"] = h_dec.unsqueeze(1)
        if plan:
            aux_mean["_planned_next"] = plan[0]
        aux_mean["_latent_slots"] = latent_slots
        aux_mean["_risk_signal"] = risk_signal
        cache["seen"] = seen + 1
        return logits, aux_mean

    def _stream_prefill(self, input_ids: torch.Tensor, cache: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logits: Optional[torch.Tensor] = None
        aux: Dict[str, torch.Tensor] = {}
        for pos in range(input_ids.size(1)):
            logits, aux = self._stream_step(input_ids[:, pos], cache)
        if logits is None:
            raise ValueError("cannot stream-prefill an empty prompt")
        return logits, aux

    @torch.no_grad()
    def _generate_text_streaming(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        typical_p: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram: int = 0,
        min_new_tokens: int = 0,
        lookahead_blend: float = 0.0,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        idx = input_ids.to(device=device, dtype=torch.long)
        if int(max_new_tokens) <= 0:
            return idx
        eos_id = self.cfg.eos_id if eos_id is None else eos_id
        cache = self._stream_new_cache(idx.size(0), device, self.token_emb.weight.dtype)
        next_logits, aux = self._stream_prefill(idx, cache)
        step = 0
        while step < int(max_new_tokens):
            logits_t = next_logits.float().clone()
            logits_t[:, self.cfg.image_token_offset:] = -float("inf")
            if lookahead_blend > 0 and "_planned_next" in aux:
                planned = aux["_planned_next"]
                plan_last = planned[:, -1] if planned.ndim == 3 else planned
                plan_logits = self.lm_head(plan_last).float()
                plan_logits[:, self.cfg.image_token_offset:] = -float("inf")
                logits_t = (1.0 - float(lookahead_blend)) * logits_t + float(lookahead_blend) * plan_logits
            apply_repetition_penalty_(logits_t, idx, repetition_penalty)
            if step < min_new_tokens:
                logits_t[:, eos_id] = -float("inf")
            fallback_logits_t = logits_t.clone()
            if no_repeat_ngram > 1:
                apply_no_repeat_ngram_(logits_t, idx, no_repeat_ngram)
            if temperature <= 0:
                probs = safe_softmax_probs(logits_t, fallback_logits_t)
                next_id = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                filtered = typical_filter(logits_t / max(float(temperature), 1e-8), typical_p)
                filtered = top_k_top_p_filter(filtered, top_k, top_p)
                probs = safe_softmax_probs(filtered, fallback_logits_t)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            step += 1
            if step >= min_new_tokens and bool((next_id == eos_id).all()):
                break
            if step < max_new_tokens:
                next_logits, aux = self._stream_step(next_id.squeeze(1), cache)
        return idx

    @torch.no_grad()
    def generate_text(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        typical_p: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram: int = 0,
        min_new_tokens: int = 0,
        lookahead_blend: float = 0.0,
        aux_score_top_k: int = 0,
        aux_score_weight: float = 0.0,
        eos_id: Optional[int] = None,
        conditioning_vectors: Optional[torch.Tensor] = None,
        conditioning_scale: float = 0.0,
        proactive_aux_score: Optional[bool] = None,
        aux_score_risk_threshold: Optional[float] = None,
        aux_score_contradiction_threshold: Optional[float] = None,
        aux_score_candidate_pool: Optional[int] = None,
        aux_score_backtrack_window: Optional[int] = None,
        aux_score_max_backtracks: Optional[int] = None,
        latent_search_depth: Optional[int] = None,
        latent_search_branches: Optional[int] = None,
        use_streaming_cache: Optional[bool] = None,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        if int(max_new_tokens) <= 0:
            return input_ids.to(device=device, dtype=torch.long)
        proactive = self.cfg.aux_score_proactive if proactive_aux_score is None else bool(proactive_aux_score)
        cand_pool = max(1, int(self.cfg.aux_score_candidate_pool if aux_score_candidate_pool is None else aux_score_candidate_pool))
        max_backtracks = max(0, int(self.cfg.aux_score_max_backtracks if aux_score_max_backtracks is None else aux_score_max_backtracks))
        rollout_depth = max(0, int(self.cfg.latent_search_depth if latent_search_depth is None else latent_search_depth))
        search_branches = max(1, int(self.cfg.latent_search_branches if latent_search_branches is None else latent_search_branches))
        requested = bool(getattr(self.cfg, "streaming_generation", True)) if use_streaming_cache is None else bool(use_streaming_cache)
        supported, reason = self._streaming_text_supported(
            input_ids.to(device, dtype=torch.long), conditioning_vectors, conditioning_scale,
            proactive, aux_score_top_k, aux_score_weight, cand_pool, rollout_depth, search_branches, max_backtracks,
        )
        if requested and supported:
            return self._generate_text_streaming(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                typical_p=typical_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram=no_repeat_ngram,
                min_new_tokens=min_new_tokens,
                lookahead_blend=lookahead_blend,
                eos_id=eos_id,
            )
        if requested and not supported and not bool(getattr(self.cfg, "streaming_fallback_on_unsupported", True)):
            raise RuntimeError(f"streaming generation cache unsupported for this request: {reason}")
        return self._generate_text_full_context(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            typical_p=typical_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram=no_repeat_ngram,
            min_new_tokens=min_new_tokens,
            lookahead_blend=lookahead_blend,
            aux_score_top_k=aux_score_top_k,
            aux_score_weight=aux_score_weight,
            eos_id=eos_id,
            conditioning_vectors=conditioning_vectors,
            conditioning_scale=conditioning_scale,
            proactive_aux_score=proactive_aux_score,
            aux_score_risk_threshold=aux_score_risk_threshold,
            aux_score_contradiction_threshold=aux_score_contradiction_threshold,
            aux_score_candidate_pool=aux_score_candidate_pool,
            aux_score_backtrack_window=aux_score_backtrack_window,
            aux_score_max_backtracks=aux_score_max_backtracks,
            latent_search_depth=latent_search_depth,
            latent_search_branches=latent_search_branches,
        )

    @torch.no_grad()
    def _generate_text_full_context(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        typical_p: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram: int = 0,
        min_new_tokens: int = 0,
        lookahead_blend: float = 0.0,
        aux_score_top_k: int = 0,
        aux_score_weight: float = 0.0,
        eos_id: Optional[int] = None,
        conditioning_vectors: Optional[torch.Tensor] = None,
        conditioning_scale: float = 0.0,
        proactive_aux_score: Optional[bool] = None,
        aux_score_risk_threshold: Optional[float] = None,
        aux_score_contradiction_threshold: Optional[float] = None,
        aux_score_candidate_pool: Optional[int] = None,
        aux_score_backtrack_window: Optional[int] = None,
        aux_score_max_backtracks: Optional[int] = None,
        latent_search_depth: Optional[int] = None,
        latent_search_branches: Optional[int] = None,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        idx = input_ids.to(device=device, dtype=torch.long)
        if int(max_new_tokens) <= 0:
            return idx
        initial_len = idx.size(1)
        eos_id = self.cfg.eos_id if eos_id is None else eos_id
        proactive = self.cfg.aux_score_proactive if proactive_aux_score is None else bool(proactive_aux_score)
        risk_thr = self.cfg.aux_score_risk_threshold if aux_score_risk_threshold is None else float(aux_score_risk_threshold)
        con_thr = self.cfg.aux_score_contradiction_threshold if aux_score_contradiction_threshold is None else float(aux_score_contradiction_threshold)
        cand_pool = max(1, int(self.cfg.aux_score_candidate_pool if aux_score_candidate_pool is None else aux_score_candidate_pool))
        backtrack_window = max(1, int(self.cfg.aux_score_backtrack_window if aux_score_backtrack_window is None else aux_score_backtrack_window))
        max_backtracks = max(0, int(self.cfg.aux_score_max_backtracks if aux_score_max_backtracks is None else aux_score_max_backtracks))
        rollout_depth = max(0, int(self.cfg.latent_search_depth if latent_search_depth is None else latent_search_depth))
        search_branches = max(1, int(self.cfg.latent_search_branches if latent_search_branches is None else latent_search_branches))
        if bool(getattr(self.cfg, "strict_linear_core", True)):
            # Search/branch reranking performs extra full-context model calls per token;
            # keep default decoding on the single-pass linear core.
            rollout_depth = 0
            search_branches = 1
            max_backtracks = 0
            if not proactive and float(aux_score_weight) <= 0 and int(aux_score_top_k) <= 0:
                cand_pool = 1
            else:
                cand_pool = min(cand_pool, 4)
        cand_pool = max(cand_pool, search_branches)
        backtracks_used = 0
        step = 0
        while step < max_new_tokens:
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _, aux = self(idx_cond, return_hidden=True, conditioning_vectors=conditioning_vectors, conditioning_scale=conditioning_scale)
            next_logits = logits[:, -1, :].float()
            next_logits[:, self.cfg.image_token_offset:] = -float("inf")
            if lookahead_blend > 0 and "_planned_next" in aux:
                plan_logits = self.lm_head(aux["_planned_next"][:, -1]).float()
                plan_logits[:, self.cfg.image_token_offset:] = -float("inf")
                next_logits = (1.0 - lookahead_blend) * next_logits + lookahead_blend * plan_logits
            apply_repetition_penalty_(next_logits, idx, repetition_penalty)
            if step < min_new_tokens:
                next_logits[:, eos_id] = -float("inf")
            fallback_next_logits = next_logits.clone()
            if no_repeat_ngram > 1:
                apply_no_repeat_ngram_(next_logits, idx, no_repeat_ngram)
            if temperature <= 0:
                probs = safe_softmax_probs(next_logits, fallback_next_logits)
                candidate_ids = torch.topk(probs, k=min(cand_pool, probs.size(-1)), dim=-1).indices
                candidate_lp = torch.log(probs.gather(1, candidate_ids).clamp_min(1e-12))
            else:
                filtered = typical_filter(next_logits / max(temperature, 1e-8), typical_p)
                filtered = top_k_top_p_filter(filtered, top_k, top_p)
                probs = safe_softmax_probs(filtered, fallback_next_logits)
                sample_n = min(cand_pool if proactive or aux_score_weight > 0 else 1, probs.size(-1))
                if sample_n > 1:
                    candidate_ids = torch.topk(probs, k=sample_n, dim=-1).indices
                else:
                    candidate_ids = torch.multinomial(probs, num_samples=1)
                candidate_lp = torch.log(probs.gather(1, candidate_ids).clamp_min(1e-12))
            rerank_active = (proactive or int(aux_score_top_k) > 0 or float(aux_score_weight) > 0.0) and candidate_ids.size(1) > 1
            if rerank_active and idx.size(0) == 1:
                candidate_ids, candidate_lp, risks, contras = self._rank_candidates_with_aux_score(
                    idx, candidate_ids, candidate_lp, max(float(aux_score_weight), self.cfg.aux_score_logit_scale), conditioning_vectors, conditioning_scale, rollout_depth=rollout_depth
                )
                accepted = (risks <= risk_thr) & (contras <= con_thr)
                if proactive and not bool(accepted[:, 0].all()):
                    good = torch.nonzero(accepted[0], as_tuple=False).flatten()
                    if good.numel() > 0:
                        pick = candidate_ids[:, good[0]:good[0]+1]
                    elif backtracks_used < max_backtracks and idx.size(1) - initial_len > backtrack_window + min_new_tokens:
                        idx = idx[:, :-backtrack_window]
                        backtracks_used += 1
                        continue
                    else:
                        # No safe option found; use lowest-risk ranked candidate rather than a blind sample.
                        low = torch.argmin(risks + contras, dim=1, keepdim=True)
                        pick = torch.gather(candidate_ids, 1, low)
                else:
                    pick = candidate_ids[:, :1]
                next_id = pick
            else:
                next_id = candidate_ids[:, :1]
            idx = torch.cat([idx, next_id], dim=1)
            step += 1
            if step >= min_new_tokens and bool((next_id == eos_id).all()):
                break
        return idx


    @torch.no_grad()
    def output_mode_decision(self, input_ids: torch.Tensor) -> Dict[str, float]:
        """Return probabilities for using no/short/full/multi/renderer reasoning text."""
        device = next(self.parameters()).device
        ids = input_ids.to(device=device, dtype=torch.long)[:, -int(self.cfg.block_size):]
        h, _, _ = self.encode_hidden(ids)
        probs = F.softmax(self.output_mode_classifier(h), dim=-1)[0].detach().float().cpu()
        names = ["none", "short_summary", "full_artificial_cot", "multi_cot", "renderer_only"]
        return {names[i] if i < len(names) else f"mode_{i}": float(probs[i]) for i in range(probs.numel())}

    @torch.no_grad()
    def score_latent_branch(self, input_ids: torch.Tensor, conditioning_vectors: Optional[torch.Tensor] = None, conditioning_scale: float = 0.0) -> Dict[str, float]:
        """Score a proposed branch/continuation with aux-score, energy, and risk signals."""
        device = next(self.parameters()).device
        ids = input_ids.to(device=device, dtype=torch.long)
        if ids.ndim != 2 or ids.size(1) <= 0:
            raise ValueError("input_ids must have shape [B,T] with T > 0")
        ctx = ids[:, -self.cfg.block_size:]
        logits, _, aux = self(ctx, return_hidden=True, conditioning_vectors=conditioning_vectors, conditioning_scale=conditioning_scale)
        h = aux["_hidden"]
        vf_scores = self._aux_score_scores_for_context(ctx, conditioning_vectors, conditioning_scale)
        energy_like = float(aux.get("mixture_energy_router_energy", torch.tensor(0.0, device=h.device)).detach().float().mean().cpu()) if "mixture_energy_router_energy" in aux else 0.0
        risk_signal = float(aux.get("risk_signal_mean", torch.tensor(0.0, device=h.device)).detach().float().mean().cpu()) if "risk_signal_mean" in aux else 0.0
        return {
            "quality": float(vf_scores["quality"].mean().detach().cpu()),
            "risk": float(vf_scores["risk"].mean().detach().cpu()),
            "contradiction": float(vf_scores["contradiction"].mean().detach().cpu()),
            "risk_signal": risk_signal,
            "energy": energy_like,
        }

    @torch.no_grad()
    def generate_image_tokens(
        self,
        prompt_ids: torch.Tensor,
        grid: int = 16,
        steps: int = 12,
        temperature: float = 1.0,
        top_k: int = 128,
        quality_guidance: float = 0.2,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        mask_schedule: str = "cosine",
        gumbel_noise: float = 0.0,
        min_keep: int = 1,
    ) -> torch.Tensor:
        """MaskGIT/Muse-style iterative masked-token image diffusion.

        NEED keeps the discrete token-diffusion interface but improves sampling:
        it repeatedly samples every masked image location, keeps the most confident
        subset under a cosine/linear schedule, and remasks uncertain positions.
        Optional classifier-free guidance mixes conditional and negative/unconditional
        image-token logits without changing the NEED backbone.
        """
        device = next(self.parameters()).device
        n = min(int(grid * grid), int(self.cfg.image_max_tokens))
        steps = max(1, int(steps))
        prompt = prompt_ids.to(device=device, dtype=torch.long)
        prompt_budget = int(self.cfg.block_size) - n - 2
        if prompt_budget <= 0:
            raise ValueError(
                f"image grid requires {n + 2} image tokens but block_size is {self.cfg.block_size}; "
                "reduce grid/image_max_tokens or increase block_size"
            )
        prompt = prompt[:, -prompt_budget:]
        # Infer a coarse object/layout program from the text prompt. The program
        # conditions token diffusion through a lightweight logit bias so image
        # generation has an explicit semantic/layout scaffold before patch tokens.
        prompt_h, _, _ = self.encode_hidden(prompt[:, -self.cfg.block_size:])
        obj_slots, obj_aux = self.object_program(prompt_h, text_mask=(prompt[:, -prompt_h.size(1):] >= Special.byte_start))
        obj_bias = torch.tanh(obj_slots.mean(dim=1)) @ self.token_emb.weight[self.cfg.image_token_offset:self.cfg.image_token_offset+self.cfg.image_codebook_size].t()
        obj_bias = obj_bias.view(prompt.size(0), 1, self.cfg.image_codebook_size) * float(self.cfg.object_program_strength)
        neg_prompt = negative_prompt_ids.to(device=device, dtype=torch.long)[:, -prompt_budget:] if negative_prompt_ids is not None else None
        canvas = torch.full((prompt.size(0), n), self.cfg.img_mask_id, device=device, dtype=torch.long)
        seq = torch.cat([
            prompt,
            torch.full((prompt.size(0), 1), self.cfg.img_bos_id, device=device, dtype=torch.long),
            canvas,
            torch.full((prompt.size(0), 1), self.cfg.img_eos_id, device=device, dtype=torch.long),
        ], dim=1)
        image_start = prompt.size(1) + 1
        keep_mask = torch.zeros((prompt.size(0), n), device=device, dtype=torch.bool)

        def _image_logits(base_seq: torch.Tensor, base_image_start: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            seq_cond = base_seq[:, -self.cfg.block_size:]
            logits, _, aux = self(seq_cond, return_hidden=True)
            offset = max(0, base_seq.size(1) - self.cfg.block_size)
            rel_start = base_image_start - offset
            rel_end = rel_start + n
            if rel_start < 0 or rel_end > logits.size(1):
                raise ValueError("image sequence exceeds block_size; reduce grid or increase block_size")
            img_logits = logits[:, rel_start:rel_end, self.cfg.image_token_offset:self.cfg.image_token_offset+self.cfg.image_codebook_size].float()
            hidden = aux.get("_hidden", None)
            if quality_guidance != 0 and hidden is not None:
                q = torch.sigmoid(self.image_quality(hidden[:, rel_start:rel_end])).float()
                img_logits = img_logits + quality_guidance * (q - 0.5)
            img_logits = img_logits + obj_bias.to(img_logits.device, img_logits.dtype)
            return img_logits, hidden[:, rel_start:rel_end] if hidden is not None else None

        for s in range(steps):
            img_logits, _ = _image_logits(seq, image_start)
            if neg_prompt is not None and cfg_scale != 1.0:
                neg_seq = torch.cat([
                    neg_prompt,
                    torch.full((neg_prompt.size(0), 1), self.cfg.img_bos_id, device=device, dtype=torch.long),
                    seq[:, image_start:image_start+n],
                    torch.full((neg_prompt.size(0), 1), self.cfg.img_eos_id, device=device, dtype=torch.long),
                ], dim=1)
                neg_logits, _ = _image_logits(neg_seq, neg_prompt.size(1) + 1)
                if neg_logits.size(0) == 1 and img_logits.size(0) > 1:
                    neg_logits = neg_logits.expand_as(img_logits)
                img_logits = neg_logits + cfg_scale * (img_logits - neg_logits)
            if gumbel_noise > 0:
                u = torch.rand_like(img_logits).clamp_(1e-6, 1 - 1e-6)
                img_logits = img_logits - gumbel_noise * torch.log(-torch.log(u))
            tau = max(float(temperature), 1e-8) * max(0.35, 1.0 - 0.65 * (s / max(1, steps - 1)))
            filt = top_k_top_p_filter((img_logits / tau).reshape(-1, self.cfg.image_codebook_size), top_k=top_k, top_p=1.0).view_as(img_logits)
            probs = safe_softmax_probs(filt, img_logits.float())
            sampled = torch.multinomial(probs.reshape(-1, self.cfg.image_codebook_size), 1).view(prompt.size(0), n) + self.cfg.image_token_offset
            conf = probs.max(dim=-1).values
            # Never let already confident pixels become less stable without evidence.
            prev = seq[:, image_start:image_start+n]
            seq[:, image_start:image_start+n] = torch.where(keep_mask, prev, sampled)
            progress = (s + 1) / steps
            if mask_schedule == "linear":
                keep_ratio = progress
            elif mask_schedule == "quadratic":
                keep_ratio = progress ** 2
            else:
                keep_ratio = 1.0 - math.cos(progress * math.pi / 2.0)
            keep_count = min(n, max(int(min_keep), int(math.ceil(n * keep_ratio))))
            # Confident positions are kept; others are remasked for the next diffusion pass.
            top_pos = torch.topk(conf, k=keep_count, dim=-1).indices
            new_keep = torch.zeros_like(keep_mask)
            new_keep.scatter_(1, top_pos, True)
            keep_mask = keep_mask | new_keep if s == steps - 1 else new_keep
            if s < steps - 1:
                seq[:, image_start:image_start+n] = torch.where(keep_mask, seq[:, image_start:image_start+n], torch.full_like(prev, self.cfg.img_mask_id))
        return seq[:, image_start:image_start+n]


# -----------------------------
# Sampling helpers
# -----------------------------

def apply_repetition_penalty_(logits: torch.Tensor, ids: torch.Tensor, penalty: float) -> None:
    if penalty <= 0:
        raise ValueError("repetition_penalty must be positive")
    if ids.numel() == 0:
        return
    bsz, vocab = logits.size(0), logits.size(-1)
    valid = (ids >= 0) & (ids < vocab)
    if not bool(valid.any().detach().cpu()):
        return
    safe = ids.clamp(min=0, max=vocab - 1).to(torch.long)
    seen_counts = logits.new_zeros((bsz, vocab))
    seen_counts.scatter_add_(1, safe, valid.to(dtype=seen_counts.dtype))
    seen = seen_counts > 0
    adjusted = torch.where(logits < 0, logits * float(penalty), logits / float(penalty))
    logits.copy_(torch.where(seen, adjusted, logits))


def apply_no_repeat_ngram_(logits: torch.Tensor, ids: torch.Tensor, n: int) -> None:
    if n <= 1 or ids.size(1) < n:
        return
    bsz, vocab = logits.size(0), logits.size(-1)
    prefix_len = int(n) - 1
    windows = ids.unfold(dimension=1, size=prefix_len, step=1)
    # Prefixes that have a following continuation are all windows except the final
    # suffix window; continuations start right after each such prefix.
    prefixes = windows[:, :-1, :]
    continuations = ids[:, prefix_len:]
    target = ids[:, -prefix_len:].unsqueeze(1)
    match = (prefixes == target).all(dim=-1)
    valid_next = (continuations >= 0) & (continuations < vocab)
    active = match & valid_next
    if not bool(active.any().detach().cpu()):
        return
    safe_next = continuations.clamp(min=0, max=vocab - 1).to(torch.long)
    banned_counts = logits.new_zeros((bsz, vocab))
    banned_counts.scatter_add_(1, safe_next, active.to(dtype=banned_counts.dtype))
    logits.masked_fill_(banned_counts > 0, -float("inf"))




def safe_softmax_probs(logits: torch.Tensor, fallback_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Softmax that never returns NaN/zero-mass rows after hard filtering.

    Sampling filters can legally mask every candidate in a row (for example a
    no-repeat constraint plus text-only image masking). Plain softmax over all
    -inf yields NaNs and torch.multinomial then fails. Use the filtered logits
    where they have at least one finite entry, otherwise fall back to the
    pre-filter logits, and finally to a uniform distribution if even the fallback
    is fully masked.
    """
    def _clean(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        finite = torch.isfinite(x)
        has = finite.any(dim=-1, keepdim=True)
        cleaned = torch.where(finite, x, torch.full_like(x, -1.0e9))
        return cleaned, has

    cleaned, has = _clean(logits.float())
    probs = F.softmax(cleaned, dim=-1)
    bad = (~has) | (~torch.isfinite(probs).all(dim=-1, keepdim=True)) | (probs.sum(dim=-1, keepdim=True) <= 0)
    if fallback_logits is not None:
        fb_clean, fb_has = _clean(fallback_logits.float())
        fb_probs = F.softmax(fb_clean, dim=-1)
        fb_bad = (~fb_has) | (~torch.isfinite(fb_probs).all(dim=-1, keepdim=True)) | (fb_probs.sum(dim=-1, keepdim=True) <= 0)
        uniform = torch.full_like(probs, 1.0 / max(1, probs.size(-1)))
        fb_probs = torch.where(fb_bad, uniform, fb_probs)
        probs = torch.where(bad, fb_probs, probs)
    else:
        uniform = torch.full_like(probs, 1.0 / max(1, probs.size(-1)))
        probs = torch.where(bad, uniform, probs)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    out = logits.clone()
    vocab = out.size(-1)
    if top_k > 0 and top_k < vocab:
        vals, _ = torch.topk(out, top_k, dim=-1)
        kth = vals[..., -1:].expand_as(out)
        out = torch.where(out < kth, torch.full_like(out, -float("inf")), out)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(out, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        out = torch.full_like(out, -float("inf"))
        out.scatter_(-1, sorted_idx, sorted_logits)
    return out


def typical_filter(logits: torch.Tensor, typical_p: float = 1.0) -> torch.Tensor:
    if typical_p >= 1.0:
        return logits
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
    dev = (-log_probs - entropy).abs()
    sorted_dev, idx = torch.sort(dev, dim=-1)
    sorted_logits = torch.gather(logits, -1, idx)
    sorted_probs = torch.gather(probs, -1, idx)
    cum = torch.cumsum(sorted_probs, dim=-1)
    remove = cum > typical_p
    remove[..., 1:] = remove[..., :-1].clone(); remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
    out = torch.full_like(logits, -float("inf"))
    out.scatter_(-1, idx, sorted_logits)
    return out



# -----------------------------
# Opt-in behavioral latent pathway memory
# -----------------------------

class LatentMemoryStore:
    """Small file-backed memory for prior latent pathways.

    Stores public summaries plus compressed latent vectors. Runtime retrieval is
    opt-in; when used, retrieved items are formatted as behavioral guidance rather
    than factual context.
    """
    def __init__(self, root: Union[str, Path], dim: int, max_items: int = 256):
        self.root = Path(root)
        self.dim = int(dim)
        self.max_items = int(max_items)
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.root / "latent_memory.json"
        self.vec_path = self.root / "latent_memory.pt"
        self.items: List[Dict[str, object]] = []
        self.vectors = torch.empty(0, self.dim)
        self.load()

    def load(self) -> None:
        if self.meta_path.exists():
            try:
                loaded = json.loads(self.meta_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    self.items = loaded
            except Exception:
                self.items = []
        if self.vec_path.exists():
            try:
                data = torch.load(self.vec_path, map_location="cpu", weights_only=True)
                self.vectors = data.get("vectors", torch.empty(0, self.dim)).float()
            except Exception:
                self.vectors = torch.empty(0, self.dim)
        if self.vectors.ndim != 2 or self.vectors.size(-1) != self.dim:
            self.vectors = torch.empty(0, self.dim)

    def save(self) -> None:
        self.items = self.items[-self.max_items:]
        self.vectors = self.vectors[-self.max_items:]
        self.meta_path.write_text(json.dumps(self.items, indent=2), encoding="utf-8")
        torch.save({"vectors": self.vectors.cpu()}, self.vec_path)

    def add(self, prompt: str, summary: str, pathway_vectors: torch.Tensor, tags: Optional[Dict[str, object]] = None) -> None:
        vec = pathway_vectors.detach().float().cpu()
        if vec.ndim == 3:
            vec = vec[0]
        if vec.ndim != 2 or vec.size(-1) != self.dim:
            return
        pooled = vec.mean(dim=0)
        self.items.append({"prompt": prompt[-1000:], "summary": summary[-2000:], "tags": tags or {}, "time": time.time()})
        self.vectors = torch.cat([self.vectors, pooled.view(1, -1)], dim=0)
        self.save()

    def retrieve(
        self,
        query_vectors: torch.Tensor,
        k: int = 4,
        score_weight: float = 0.18,
        risk_weight: float = 0.22,
        contradiction_weight: float = 0.25,
        min_score: float = 0.0,
    ) -> Tuple[str, Optional[torch.Tensor]]:
        if self.vectors.numel() == 0 or not self.items:
            return "", None
        q = query_vectors.detach().float().cpu()
        if q.ndim == 3:
            q = q[0]
        if q.ndim != 2 or q.size(-1) != self.dim:
            return "", None
        q = q.mean(dim=0)
        sim = (F.normalize(q.view(1, -1), dim=-1) @ F.normalize(self.vectors, dim=-1).t())[0]
        # Low-data latent memory is adapter-like: prefer successful, low-risk
        # episodes over merely similar episodes so a few bad traces do not dominate.
        adjusted = sim.clone()
        for i, item in enumerate(self.items[: adjusted.numel()]):
            tags = item.get("tags", {}) if isinstance(item.get("tags", {}), dict) else {}
            try:
                score = float(tags.get("score", tags.get("quality", 0.5)))
            except Exception:
                score = 0.5
            try:
                risk = float(tags.get("risk", 0.5))
            except Exception:
                risk = 0.5
            try:
                contradiction = float(tags.get("contradiction", 0.0))
            except Exception:
                contradiction = 0.0
            if score < float(min_score):
                adjusted[i] = -1e9
            else:
                adjusted[i] = adjusted[i] + float(score_weight) * score - float(risk_weight) * risk - float(contradiction_weight) * contradiction
        _, idx = torch.topk(adjusted, k=min(int(k), self.vectors.size(0)))
        chunks: List[str] = []
        vecs: List[torch.Tensor] = []
        for rank, i in enumerate(idx.tolist()):
            if adjusted[i].item() < -1e8:
                continue
            item = self.items[i]
            chunks.append(
                f"<behavioral_latent_memory rank=\"{rank}\">\n"
                "Use this only as prior response behavior, not as a source of facts.\n"
                f"{item.get('summary','')}\n"
                "</behavioral_latent_memory>"
            )
            vecs.append(self.vectors[i])
        return "\n".join(chunks), torch.stack(vecs).unsqueeze(0) if vecs else None

# -----------------------------
# Checkpoint helpers
# -----------------------------

def save_json(obj: Dict[str, object], path: Union[str, Path]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_json(path: Union[str, Path]) -> Dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict JSON in {path}")
    return data


def save_model(model: NeedModel, out_dir: Union[str, Path], metrics: Optional[Dict[str, float]] = None, name: str = "model") -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = asdict(model.cfg)
    save_json({"config": cfg, "metrics": metrics or {}, "format": "need"}, out / "config.json")
    # Clone tensors to break shared-storage aliases such as tied embeddings;
    # this keeps safetensors simple and load_state_dict will re-tie in __init__.
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if safe_save_file is not None:
        safe_save_file(state, str(out / f"{name}.safetensors"))
    else:
        torch.save(state, out / f"{name}.pt")


def load_model(ckpt: Union[str, Path], device: Union[str, torch.device] = "cpu", prefer_best: bool = False, kernel_backend: Optional[str] = None) -> NeedModel:
    ckpt = Path(ckpt)
    cfg_data = load_json(ckpt / "config.json")
    raw_cfg = cfg_data.get("config", cfg_data)
    cfg = NeedConfig.from_dict(raw_cfg)
    if kernel_backend is not None:
        cfg.kernel_backend = kernel_backend
    model = NeedModel(cfg)
    stem = "best" if prefer_best else "model"
    if safe_load_file is not None and (ckpt / f"{stem}.safetensors").exists():
        state = safe_load_file(str(ckpt / f"{stem}.safetensors"), device=str(device))
    elif (ckpt / f"{stem}.pt").exists():
        try:
            state = torch.load(ckpt / f"{stem}.pt", map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(ckpt / f"{stem}.pt", map_location=device)
    elif safe_load_file is not None and (ckpt / "model.safetensors").exists():
        state = safe_load_file(str(ckpt / "model.safetensors"), device=str(device))
    else:
        try:
            state = torch.load(ckpt / "model.pt", map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(ckpt / "model.pt", map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Older checkpoints do not contain the learned DVSD slot-router.  Keep them
    # backward-compatible by making the new router conservative instead of random:
    # it predicts one slot until the checkpoint is trained/fine-tuned with the
    # DVSD router objective or the user explicitly enables a trained checkpoint.
    router_missing = any(str(k).startswith("dvsd_slot_router") for k in missing)
    setattr(model, "_dvsd_router_loaded", not router_missing)
    if router_missing and hasattr(model, "dvsd_slot_router"):
        try:
            last = model.dvsd_slot_router[-1]
            if isinstance(last, nn.Linear):
                with torch.no_grad():
                    last.weight.zero_()
                    if last.bias is not None:
                        last.bias.fill_(-4.0)
                        last.bias[0] = 4.0
            model.cfg.dvsd_router_enabled = False
        except Exception:
            model.cfg.dvsd_router_enabled = False
    compound_missing = any(str(k).startswith("planner.token_feedback") or str(k).startswith("planner.descent_proj") or str(k).startswith("planner.feedback_gate") or str(k).startswith("planner.feedback_norm") or str(k).startswith("planner.descent_norm") for k in missing)
    setattr(model, "_dvsd_planner_compound_loaded", not compound_missing)
    if compound_missing:
        model.cfg.dvsd_planner_compound_enabled = False
    block_space_missing = any(str(k).startswith("planner.block_") for k in missing)
    setattr(model, "_planner_block_space_loaded", not block_space_missing)
    if block_space_missing:
        model.cfg.planner_block_space_enabled = False
    setattr(model, "_missing_keys", list(missing))
    setattr(model, "_unexpected_keys", list(unexpected))
    model.to(device)
    model.eval()
    return model


def make_image_tokenizer(cfg: NeedConfig) -> DynamicImageTokenizer:
    return DynamicImageTokenizer(cfg.image_tokenizer_config())


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
