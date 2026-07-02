#!/usr/bin/env python3
"""
This module is intentionally self-contained and dependency-light. It provides:
- byte fallback text tokenization
- dynamic discrete image tokenization / reconstruction with spatial coordinate conditioning
- a causal non-attention NEED backbone with recurrent retention, sparse MoE,
  hierarchical adaptive memory, temporal latent planning, proactive aux scoring, and auxiliary score heads
- autoregressive text generation 

The implementation uses PyTorch fallbacks everywhere. Optional Triton kernels are routed
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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

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
            image = Image.open(image)  # type: ignore[arg-type]
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
            grid = max(1, grid)
        needed = grid * grid
        if len(raw) < needed:
            raw = raw + [raw[-1]] * (needed - len(raw))
        raw = raw[:needed]
        arr = self.codebook[np.asarray(raw, dtype=np.int64)].reshape(grid, grid, 3)
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB").resize((size, size), resample=Image.Resampling.NEAREST)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self.cfg)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "DynamicImageTokenizer":
        return cls(ImageTokenizerConfig(**{k: v for k, v in data.items() if k in ImageTokenizerConfig.__dataclass_fields__}))


# -----------------------------
# Config
# -----------------------------

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
    n_conv_scales: int = 3
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
    # SSD/Mamba-style structured dual retention. The original selective retention is
    # still available with --retention_impl selective, but SSD is now the default.
    retention_impl: str = "ssd"
    ssd_conv_kernel: int = 3
    ssd_dt_min: float = 0.001
    ssd_dt_max: float = 0.20

    # token/layer adaptive compute: gates expensive memory/equilibrium/MoE paths
    # per token while keeping a soft compute budget for equal-FLOP experiments.
    adaptive_depth: bool = True
    compute_budget: float = 0.72
    min_compute_gate: float = 0.08
    depth_gate_temperature: float = 1.0

    # energy equilibrium
    energy_rank: int = 96
    energy_steps: int = 3
    energy_min_steps: int = 1
    energy_step_size: float = 0.55
    energy_row_norm: float = 0.70
    min_precision: float = 0.25
    adaptive_energy: bool = True
    adaptive_residual_threshold: float = 0.035
    adaptive_effort_alpha: float = 5.0

    # MoE / memory
    n_experts: int = 4
    moe_top_k: int = 2
    moe_router_jitter: float = 0.0
    moe_dropout: float = 0.0
    moe_use_shared_expert: bool = True
    memory_chunk_size: int = 32
    memory_slots: int = 16
    memory_rank: int = 64
    memory_decay_short: float = 0.84
    memory_decay_episodic: float = 0.97
    memory_decay_short_min: float = 0.50
    memory_decay_short_max: float = 0.97
    memory_decay_episodic_min: float = 0.78
    memory_decay_episodic_max: float = 0.997
    memory_boundary_strength: float = 0.70
    memory_mix: float = 0.35

    # exact associative recall: a long-range full-sequence memory path that can
    # retrieve exact earlier token/value states without relying on compressed SSM
    # state alone. It is chunked to avoid materializing B*T*T for long contexts.
    exact_recall: bool = True
    exact_recall_dim: int = 64
    exact_recall_top_k: int = 8
    exact_recall_mix: float = 0.10
    exact_recall_temperature: float = 0.10
    exact_recall_token_bias: float = 1.75
    exact_recall_bigram_bias: float = 0.75
    exact_recall_chunk_size: int = 128
    exact_recall_max_tokens: int = 4096

    # state-space drift control for long contexts. This adds a small learned
    # anchor/correction path and losses that discourage hidden-state norm drift,
    # chunk-to-chunk random walks, and anchor detachment over long histories.
    state_stabilization: bool = True
    state_anchor_strength: float = 0.035
    state_drift_chunk: int = 64
    state_drift_target: float = 0.14
    state_norm_target: float = 1.0
    lambda_state_drift: float = 0.006
    lambda_state_anchor: float = 0.003
    lambda_recall_entropy: float = 3e-4

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
    objective_gradient_guard: bool = True
    objective_gradient_guard_interval: int = 50
    objective_gradient_guard_start_step: int = 200
    objective_gradient_guard_max_terms: int = 8
    objective_gradient_guard_param_tensors: int = 24
    objective_conflict_cosine_threshold: float = -0.10
    objective_conflict_quarantine_patience: int = 3

    # latent pathway conditioning / planner / aux_score
    pathway_conditioning_top_k: int = 64
    pathway_conditioning_dropout: float = 0.0
    pathway_conditioning_scale: float = 0.18
    pathway_conditioning_max_vectors: int = 512
    planner_horizons: int = 4
    planner_transition_depth: int = 2
    planner_logit_blend: float = 0.12
    aux_score_logit_scale: float = 0.10
    aux_score_proactive: bool = True
    aux_score_candidate_pool: int = 8
    aux_score_risk_threshold: float = 0.72
    aux_score_contradiction_threshold: float = 0.65
    aux_score_backtrack_window: int = 3
    aux_score_max_backtracks: int = 4
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
    image_2d_scan_strength: float = 0.20
    image_2d_scan_decay: float = 0.75
    region_word_sinkhorn_iters: int = 4

    # cognition upgrades: mixture energy routing, latent slot attention,
    # risk-signal fusion, and output-mode control.
    energy_routes: int = 6
    energy_route_steps: int = 1
    energy_route_strength: float = 0.10
    latent_slots: int = 4
    latent_slot_conditioning_scale: float = 0.12
    slow_state_chunk: int = 16
    slow_state_strength: float = 0.15
    risk_gate_strength: float = 0.18
    object_program_slots: int = 8
    object_program_strength: float = 0.18
    output_modes: int = 5  # none, short summary, full CoT, multi-CoT, renderer-only

    # objectives
    n_predict_heads: int = 4
    label_smoothing: float = 0.0
    token_dropout: float = 0.0
    lambda_mtp: float = 0.15

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
    nonseq_tree_candidates: int = 4
    nonseq_branch_top_k: int = 2
    nonseq_aux_score_weight: float = 0.10
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
    lambda_dvsd_slot_ce: float = 0.025
    lambda_dvsd_consistency: float = 0.020
    lambda_dvsd_router: float = 0.015
    lambda_equilibrium: float = 0.01
    lambda_energy: float = 0.0
    lambda_diffusion: float = 0.02
    lambda_z_loss: float = 1e-4
    lambda_moe_balance: float = 0.01
    lambda_moe_router_z: float = 1e-4
    lambda_branch_entropy: float = 5e-4
    lambda_conv_scale_entropy: float = 2e-4
    lambda_geodesic: float = 0.01
    geodesic_target: float = 0.10
    lambda_path_straightness: float = 0.005
    lambda_path_contractive: float = 0.002
    contract_kappa: float = 0.92
    lambda_energy_row_orth: float = 5e-4
    lambda_latent_norm: float = 1e-4
    lambda_memory_entropy: float = 2e-4
    lambda_memory_diversity: float = 2e-4
    lambda_adaptive_effort: float = 1e-4
    lambda_compute_budget: float = 2e-4
    lambda_controller: float = 0.015
    lambda_pathway_memory: float = 5e-4
    lambda_latent_planning: float = 0.04
    lambda_planner_ce: float = 0.04
    lambda_planning_consistency: float = 0.005
    lambda_aux_score: float = 0.04
    lambda_image_diffusion: float = 0.10
    lambda_image_contrastive: float = 0.02
    lambda_image_local_contrastive: float = 0.04
    lambda_region_word_alignment: float = 0.035
    lambda_image_spatial_smoothness: float = 0.002
    lambda_mixture_energy_router: float = 0.006
    lambda_latent_slot: float = 0.018
    lambda_latent_slot_diversity: float = 0.002
    lambda_risk_signal: float = 0.006
    lambda_latent_divergence: float = 0.010
    lambda_output_mode_classifier: float = 0.004
    lambda_object_program: float = 0.010
    lambda_timescale: float = 0.004
    lambda_preference: float = 0.0

    # dynamic image/tokenizer quality
    image_grid: int = 16
    image_min_grid: int = 8
    image_max_grid: int = 32
    image_max_tokens: int = 1024
    dynamic_image_grid: bool = True
    image_coord_scale: float = 0.25
    image_local_contrastive_temperature: float = 0.07

    def validate(self) -> None:
        if self.vocab_size != self.text_vocab_size + self.image_codebook_size:
            raise ValueError("vocab_size must equal text_vocab_size + image_codebook_size")
        if self.image_token_offset != self.text_vocab_size:
            raise ValueError("image_token_offset must equal text_vocab_size")
        if self.d_model % max(1, self.n_heads) != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not (1 <= self.energy_min_steps <= self.energy_steps):
            raise ValueError("energy_min_steps must be in [1, energy_steps]")
        if self.block_size < 8:
            raise ValueError("block_size is too small")
        self.exact_recall_dim = int(max(8, min(self.d_model, self.exact_recall_dim)))
        self.exact_recall_top_k = int(max(1, self.exact_recall_top_k))
        self.exact_recall_chunk_size = int(max(16, self.exact_recall_chunk_size))
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


OBJECTIVE_LOSS_GROUPS: Dict[str, str] = {
    # prediction-like auxiliaries
    "mtp": "prediction",
    "dvsd_slot_ce": "prediction",
    "dvsd_consistency": "prediction",
    "dvsd_router": "prediction",
    "latent_planning": "prediction",
    "planner_ce": "prediction",
    "planning_consistency": "prediction",
    "diffusion": "prediction",
    # latent control / routing
    "latent_slot": "latent",
    "latent_slot_diversity": "latent",
    "latent_slot_entropy_band": "latent",
    "mixture_energy_router_energy": "latent",
    "energy_route_entropy_band": "latent",
    "timescale_consistency": "latent",
    # uncertainty and output control
    "risk_signal": "risk",
    "latent_divergence_loss": "risk",
    "output_mode_classifier": "risk",
    "output_mode_entropy_band": "risk",
    "aux_score": "risk",
    "controller": "risk",
    # vision / grounding
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
    "energy_route_entropy": "regularizer",
    "energy_route_balance": "regularizer",
    "latent_slot_attention_entropy": "regularizer",
    "state_drift": "regularizer",
    "state_anchor": "regularizer",
    "exact_recall_entropy_floor": "regularizer",
}


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


class MultiScaleCausalConv(nn.Module):
    def __init__(self, dim: int, kernel: int, n_scales: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList()
        self.kernel_sizes = []
        for i in range(n_scales):
            k = kernel + 2 * i
            self.kernel_sizes.append(k)
            self.convs.append(nn.Conv1d(dim, dim, kernel_size=k, groups=dim, bias=True))
        self.mix = nn.Linear(dim * n_scales, dim, bias=False)
        self.gate = nn.Linear(dim, n_scales, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        xt = x.transpose(1, 2)
        ys = []
        for conv, k in zip(self.convs, self.kernel_sizes):
            padded = F.pad(xt, (k - 1, 0))
            ys.append(conv(padded).transpose(1, 2))
        cat = torch.cat(ys, dim=-1)
        probs = F.softmax(self.gate(x).float(), dim=-1).to(x.dtype)
        y = self.mix(cat)
        ent = -(probs.float() * probs.float().clamp_min(1e-8).log()).sum(dim=-1).mean()
        return self.drop(y), {"conv_scale_entropy": ent}


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
            state = torch.zeros(b, self.heads, self.head_dim, device=x.device, dtype=x.dtype)
            outs = []
            for i in range(t):
                state = state * decay[:, :, i, :] + kv[:, :, i, :]
                outs.append(q[:, :, i, :] * state)
            y = torch.stack(outs, dim=2).transpose(1, 2).reshape(b, t, d)
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
        if need_kernels is not None and self.cfg.fused_ssd_scan:
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
                state = torch.zeros(b, d, device=x.device, dtype=x.dtype)
                outs: List[torch.Tensor] = []
                for i in range(t):
                    state = decay[:, i] * state + update[:, i]
                    y_i = (c_gate[:, i] * state + self.D.to(x.dtype) * u[:, i]) * out_gate[:, i]
                    outs.append(y_i)
                y = torch.stack(outs, dim=1)
        y = self.out(self.out_norm(y))
        return y


class AdaptiveDepthGate(nn.Module):
    """Token-level compute controller for expensive NEED subpaths."""
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
        # feat: [H,W,D]. Scan along dim 0 or 1.
        if reverse:
            feat = torch.flip(feat, dims=(dim,))
        v = self.value(feat)
        g = torch.sigmoid(self.gate(feat)).to(feat.dtype)
        decay = float(self.cfg.image_2d_scan_decay)
        out = torch.zeros_like(feat)
        if dim == 0:
            state = torch.zeros_like(feat[0])
            for i in range(feat.size(0)):
                state = decay * state + (1.0 - decay) * (g[i] * v[i])
                out[i] = state
        else:
            state = torch.zeros_like(feat[:, 0])
            for j in range(feat.size(1)):
                state = decay * state + (1.0 - decay) * (g[:, j] * v[:, j])
                out[:, j] = state
        if reverse:
            out = torch.flip(out, dims=(dim,))
        return out

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.cfg.image_2d_scan:
            return x, {"image_2d_scan_energy": x.new_tensor(0.0)}
        has_image_tokens = (input_ids == self.cfg.img_bos_id) | (input_ids == self.cfg.img_mask_id) | ((input_ids >= self.cfg.image_token_offset) & (input_ids < self.cfg.image_token_offset + self.cfg.image_codebook_size))
        if not bool(has_image_tokens.any()):
            return x, {"image_2d_scan_energy": x.new_tensor(0.0)}
        y = torch.zeros_like(x)
        counts = 0
        for b in range(x.size(0)):
            inside = False
            positions: List[int] = []
            for pos in range(input_ids.size(1)):
                tok = int(input_ids[b, pos].detach().cpu())
                if tok == self.cfg.img_bos_id:
                    inside = True; positions = []; continue
                if tok == self.cfg.img_eos_id or tok == self.cfg.pad_id:
                    if positions:
                        n = len(positions); grid = int(round(math.sqrt(n)))
                        if grid > 1 and grid * grid <= n:
                            use = positions[: grid * grid]
                            feat = self.norm(x[b, use]).view(grid, grid, -1)
                            lr = self._scan_axis(feat, dim=1, reverse=False)
                            rl = self._scan_axis(feat, dim=1, reverse=True)
                            tb = self._scan_axis(feat, dim=0, reverse=False)
                            bt = self._scan_axis(feat, dim=0, reverse=True)
                            mix = self.out(torch.cat([lr, rl, tb, bt], dim=-1)).reshape(grid * grid, -1)
                            y[b, use] = mix
                            counts += 1
                    inside = False; positions = []
                    continue
                if inside and (tok == self.cfg.img_mask_id or self.cfg.image_token_offset <= tok < self.cfg.image_token_offset + self.cfg.image_codebook_size):
                    positions.append(pos)
        if counts == 0:
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
        out = torch.zeros_like(x)
        for expert_id, expert in enumerate(self.experts):
            mask = idx == expert_id
            if not bool(mask.any()):
                continue
            contrib = expert(x)
            w = torch.where(mask, weights, torch.zeros_like(weights)).sum(dim=-1, keepdim=True)
            out = out + contrib * w
        if self.shared is not None:
            out = out + 0.25 * self.shared(x)
        probs = F.softmax(logits, dim=-1)
        load = probs.mean(dim=(0, 1))
        balance = (load * load).sum() * self.cfg.n_experts
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        z_loss = logits.pow(2).mean()
        return out, {"moe_balance": balance, "branch_entropy": entropy, "moe_router_z": z_loss}


class HierarchicalMemory(nn.Module):
    """Hierarchical causal memory with context-adaptive decay and boundary resets.

    The older memory path used fixed decay constants. This version predicts short-term
    and episodic retention at every token from the current context, then lowers retention
    around inferred topic/task boundaries. That lets the memory persist through coherent
    spans and deliberately refresh itself when the sequence changes subject.
    """
    def __init__(self, cfg: NeedConfig):
        super().__init__()
        self.cfg = cfg
        r = cfg.memory_rank
        self.q = nn.Linear(cfg.d_model, r, bias=False)
        self.k = nn.Linear(cfg.d_model, r, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.decay_proj = nn.Sequential(
            RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend),
            nn.Linear(cfg.d_model, cfg.d_model // 2 if cfg.d_model >= 32 else cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model // 2 if cfg.d_model >= 32 else cfg.d_model, 2),
        )
        self.boundary_proj = nn.Sequential(
            RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend),
            nn.Linear(cfg.d_model, 1),
        )
        self.semantic_slots = nn.Parameter(torch.randn(cfg.memory_slots, r) / math.sqrt(r))
        self.semantic_values = nn.Parameter(torch.randn(cfg.memory_slots, cfg.d_model) / math.sqrt(cfg.d_model))
        self.out = nn.Linear(cfg.d_model * 4, cfg.d_model, bias=False)

    def _dynamic_decays(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = torch.sigmoid(self.decay_proj(x.float())).to(x.dtype)
        short = self.cfg.memory_decay_short_min + (self.cfg.memory_decay_short_max - self.cfg.memory_decay_short_min) * raw[..., 0:1]
        episodic = self.cfg.memory_decay_episodic_min + (self.cfg.memory_decay_episodic_max - self.cfg.memory_decay_episodic_min) * raw[..., 1:2]
        boundary = torch.sigmoid(self.boundary_proj(x.float())).to(x.dtype)
        reset = 1.0 - self.cfg.memory_boundary_strength * boundary
        short = (short * reset).clamp(0.01, 0.999)
        episodic = (episodic * (0.70 + 0.30 * reset)).clamp(0.01, 0.999)
        return short, episodic, boundary

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        b, t, d = x.shape
        q = F.normalize(self.q(x.float()), dim=-1)
        k = F.normalize(self.k(x.float()), dim=-1)
        v = self.v(x)
        short_decay, epi_decay, boundary = self._dynamic_decays(x)
        short_update = (1.0 - short_decay) * v
        epi_update = (1.0 - epi_decay) * v
        if need_kernels is not None and bool(getattr(self.cfg, "parallel_scan", True)):
            short = need_kernels.affine_recurrence_scan(short_decay, short_update, dim=1, backend=self.cfg.kernel_backend)
            episodic = need_kernels.affine_recurrence_scan(epi_decay, epi_update, dim=1, backend=self.cfg.kernel_backend)
        else:
            short_state = torch.zeros(b, d, device=x.device, dtype=x.dtype)
            epi_state = torch.zeros(b, d, device=x.device, dtype=x.dtype)
            short_out, epi_out = [], []
            for i in range(t):
                short_state = short_decay[:, i] * short_state + short_update[:, i]
                epi_state = epi_decay[:, i] * epi_state + epi_update[:, i]
                short_out.append(short_state)
                epi_out.append(epi_state)
            short = torch.stack(short_out, dim=1)
            episodic = torch.stack(epi_out, dim=1)
        sem_logits = torch.einsum("btr,sr->bts", q, F.normalize(self.semantic_slots.float(), dim=-1))
        sem_probs = F.softmax(sem_logits, dim=-1).to(x.dtype)
        semantic = torch.einsum("bts,sd->btd", sem_probs, self.semantic_values.to(x.dtype))
        assoc = []
        for start in range(0, t, self.cfg.memory_chunk_size):
            end = min(t, start + self.cfg.memory_chunk_size)
            qi = q[:, start:end]
            logits = torch.einsum("bqr,bkr->bqk", qi, k[:, :end]) / math.sqrt(k.size(-1))
            q_pos = torch.arange(start, end, device=x.device).view(-1, 1)
            k_pos = torch.arange(0, end, device=x.device).view(1, -1)
            logits = logits.masked_fill(k_pos > q_pos, -1e9)
            probs = F.softmax(logits, dim=-1).to(x.dtype)
            assoc.append(torch.einsum("bqk,bkd->bqd", probs, v[:, :end]))
        associative = torch.cat(assoc, dim=1)
        mixed = self.out(torch.cat([short, episodic, semantic, associative], dim=-1))
        ent = -(sem_probs.float() * sem_probs.float().clamp_min(1e-8).log()).sum(dim=-1).mean()
        div = (F.normalize(self.semantic_slots.float(), dim=-1) @ F.normalize(self.semantic_slots.float(), dim=-1).t()).pow(2).mean()
        decay_gap = (epi_decay.float() - short_decay.float()).abs().mean()
        return mixed, {
            "memory_entropy": ent,
            "memory_diversity": div,
            "memory_short_decay": short_decay.float().mean().detach(),
            "memory_episodic_decay": epi_decay.float().mean().detach(),
            "memory_boundary": boundary.float().mean(),
            "memory_decay_gap": decay_gap,
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
        total_resid = x.new_tensor(0.0)
        effort = torch.sigmoid(self.difficulty(context.float())).to(x.dtype)
        for step in range(self.cfg.energy_steps):
            e, grad = self.energy.energy_and_grad(z, context)
            resid = grad.pow(2).mean(dim=-1, keepdim=True).sqrt()
            total_resid = total_resid + resid.mean()
            run = 1.0
            if self.cfg.adaptive_energy and step >= self.cfg.energy_min_steps:
                run = torch.sigmoid(self.cfg.adaptive_effort_alpha * (resid - self.cfg.adaptive_residual_threshold))
                run = run * effort
            z = z - self.cfg.energy_step_size * grad * run
        e_final, g_final = self.energy.energy_and_grad(z, context)
        return z, g_final.pow(2).mean(), e_final.mean(), effort.mean()


class LatentPlanner(nn.Module):
    """Continuous latent trajectory planner.

    Instead of independent horizon projections, planning is modeled as a learned
    temporal evolution. A recurrent transition repeatedly advances a latent plan state,
    so horizon h+1 is conditioned on the predicted state for horizon h.
    """
    def __init__(self, cfg: NeedConfig):
        super().__init__()
        self.horizons = cfg.planner_horizons
        self.init = nn.Sequential(RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend), nn.Linear(cfg.d_model, cfg.d_model))
        self.time = nn.Embedding(max(1, cfg.planner_horizons + 1), cfg.d_model)
        hidden = max(cfg.d_model, cfg.d_ff // 2 if cfg.d_ff else cfg.d_model * 2)
        layers: List[nn.Module] = [RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend)]
        for _ in range(max(1, cfg.planner_transition_depth)):
            layers.extend([nn.Linear(cfg.d_model, hidden), nn.SiLU(), nn.Linear(hidden, cfg.d_model)])
        self.transition = nn.Sequential(*layers)
        self.gate = nn.Sequential(RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend), nn.Linear(cfg.d_model, cfg.d_model), nn.Sigmoid())
        self.out = RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend)

    def forward(self, h: torch.Tensor) -> List[torch.Tensor]:
        if self.horizons <= 0:
            return []
        state = self.init(h)
        outs: List[torch.Tensor] = []
        for horizon in range(1, self.horizons + 1):
            te = self.time.weight[min(horizon, self.time.num_embeddings - 1)].view(1, 1, -1).to(dtype=state.dtype, device=state.device)
            candidate = state + te
            delta = self.transition(candidate)
            gate = self.gate(candidate)
            state = state + gate * delta
            outs.append(self.out(state))
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
        q = self.q(self.x_norm(x)).view(b, t, self.heads, self.head_dim)
        k = self.k(encoded).view(b, p, self.heads, self.head_dim)
        v = self.v(encoded).view(b, p, self.heads, self.head_dim)
        scores = torch.einsum("bthd,bphd->bhtp", q.float(), k.float()) / math.sqrt(self.head_dim)
        top_k = int(self.cfg.pathway_conditioning_top_k)
        if top_k > 0 and top_k < p:
            kth = torch.topk(scores, k=top_k, dim=-1).values[..., -1:]
            scores = scores.masked_fill(scores < kth, -1e9)
        probs = F.softmax(scores, dim=-1).to(x.dtype)
        ctx = torch.einsum("bhtp,bphd->bthd", probs, v).reshape(b, t, d)
        endpoint = encoded[:, -1:].expand(-1, t, -1)
        trend = (encoded[:, -1:] - encoded[:, :1]).expand(-1, t, -1)
        mem_scores = torch.einsum("btd,md->btm", self.x_norm(x).float(), F.normalize(self.mem_key.float(), dim=-1))
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



class SlotAttentionBlock(nn.Module):
    """Shared slot-attention primitive for latent slots and object/layout slots.

    It owns the common learned-slot -> token attention and optional token -> slot
    context path so higher-level heads can instantiate the same math instead of
    carrying parallel query/key/value implementations.
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

    def _pool(self, h: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is not None and bool(mask.any()):
            m = mask.to(h.dtype).unsqueeze(-1)
            denom = m.sum(dim=1).clamp_min(1.0)
            return (h * m).sum(dim=1) / denom
        return h.mean(dim=1)

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None, need_token_context: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        b, t, d = h.shape
        token_state = self.input_norm(h).float()
        pooled = self.base(self._pool(h, mask).float())
        seeds = self.seed.to(h.dtype).unsqueeze(0).expand(b, -1, -1) + self.base_proj(pooled).unsqueeze(1).to(h.dtype)
        slot_scores = torch.einsum('bsd,btd->bst', self.slot_q(seeds.float()), self.token_k(token_state)) / math.sqrt(d)
        if mask is not None:
            slot_scores = slot_scores.masked_fill(~mask.bool().unsqueeze(1), -1e9)
        slot_att = F.softmax(slot_scores, dim=-1).to(h.dtype)
        slots = torch.einsum('bst,btd->bsd', slot_att, self.token_v(h).to(h.dtype))
        token_delta: Optional[torch.Tensor] = None
        token_att: Optional[torch.Tensor] = None
        if need_token_context:
            token_scores = torch.einsum('btd,bsd->bts', self.token_q(token_state), self.slot_k(slots.float())) / math.sqrt(d)
            token_att = F.softmax(token_scores, dim=-1).to(h.dtype)
            ctx = torch.einsum('bts,bsd->btd', token_att, self.slot_v(slots).to(h.dtype))
            token_delta = self.token_out(ctx)
        slot_entropy = -(slot_att.float() * slot_att.float().clamp_min(1e-8).log()).sum(dim=-1).mean()
        coverage = slot_att.float().max(dim=1).values.mean()
        metrics: Dict[str, torch.Tensor] = {
            "slot_attention_entropy": slot_entropy,
            "slot_attention_coverage": coverage.detach(),
        }
        if token_att is not None:
            token_entropy = -(token_att.float() * token_att.float().clamp_min(1e-8).log()).sum(dim=-1).mean()
            metrics["slot_context_entropy"] = token_entropy
        return slots, token_delta, metrics


class MixtureEnergyRouter(nn.Module):
    """Mixture of domain/task-specific latent energy energy_routers.

    A single energy well is a stabilizer.  A set of energy_routers lets NEED route a
    latent state through different learned reasoning geometries: factual recall,
    math/code, visual-spatial, planning, dialogue, etc.  The labels are not fixed;
    routing is learned from data.
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

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pooled = h.mean(dim=1)
        route_logits = self.router(pooled.float())
        route = F.softmax(route_logits, dim=-1).to(h.dtype)
        centers = torch.einsum('bn,nd->bd', route, self.centers.to(h.dtype)).unsqueeze(1)
        precision = F.softplus(torch.einsum('bn,nd->bd', route.float(), self.precision.float())).to(h.dtype).unsqueeze(1) + self.cfg.min_precision
        rows = torch.einsum('bn,nrd->brd', route.float(), F.normalize(self.rows.float(), dim=-1)).to(h.dtype) * self.cfg.energy_row_norm
        z = h
        energy_total = h.new_tensor(0.0)
        for _ in range(max(1, int(self.cfg.energy_route_steps))):
            diff = z - centers
            proj = torch.einsum('btd,brd->btr', diff, rows)
            grad = precision * diff + torch.einsum('btr,brd->btd', proj, rows)
            z = z - float(self.cfg.energy_step_size) * float(self.cfg.energy_route_strength) * grad
            energy_total = energy_total + (0.5 * (precision * diff.pow(2)).sum(dim=-1) + 0.5 * proj.pow(2).sum(dim=-1)).mean()
        route_ent = -(route.float() * route.float().clamp_min(1e-8).log()).sum(dim=-1).mean()
        route_balance = (route.float().mean(dim=0).pow(2).sum() * self.n)
        return self.out(z - h), {"mixture_energy_router_energy": energy_total, "energy_route_entropy": route_ent, "energy_route_balance": route_balance}


class LatentSlotAttention(nn.Module):
    """Build reusable latent slots and inject their context back into token states."""
    def __init__(self, cfg: NeedConfig):
        super().__init__()
        self.cfg = cfg
        self.slots = max(1, int(cfg.latent_slots))
        self.block = SlotAttentionBlock(cfg, self.slots)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        slots, delta, aux = self.block(h, need_token_context=True)
        if delta is None:
            delta = torch.zeros_like(h)
        norm_slots = F.normalize(slots.float(), dim=-1)
        gram = torch.einsum('bsd,bqd->bsq', norm_slots, norm_slots)
        ident = torch.eye(self.slots, device=h.device, dtype=gram.dtype).unsqueeze(0)
        div = (gram - ident).pow(2).mean()
        return delta, slots, {
            "latent_slot_attention_entropy": aux["slot_context_entropy"],
            "latent_slot_coverage": aux["slot_attention_coverage"],
            "latent_slot_diversity": div,
        }


class HierarchicalTimeScales(nn.Module):
    """Fast/medium/slow latent stream mixer.

    The slow stream updates at chunk/session scale; the medium stream is chunk-level;
    the fast stream is the token sequence.  This gives NEED explicit temporal strata
    instead of asking every layer to carry all timescales in one vector.
    """
    def __init__(self, cfg: NeedConfig):
        super().__init__()
        self.cfg = cfg
        self.med = nn.GRU(cfg.d_model, cfg.d_model, batch_first=True)
        self.slow = nn.GRU(cfg.d_model, cfg.d_model, batch_first=True)
        self.mix = nn.Sequential(RMSNorm(cfg.d_model * 3, kernel_backend=cfg.kernel_backend), nn.Linear(cfg.d_model * 3, cfg.d_model), nn.SiLU(), nn.Linear(cfg.d_model, cfg.d_model))

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        b, t, d = h.shape
        chunk = max(1, int(self.cfg.slow_state_chunk))
        pad = (chunk - (t % chunk)) % chunk
        hp = F.pad(h, (0, 0, 0, pad)) if pad else h
        n_chunks = hp.size(1) // chunk
        chunks = hp.view(b, n_chunks, chunk, d).mean(dim=2)
        med_chunks, _ = self.med(chunks)
        slow_seq, _ = self.slow(med_chunks)
        med = med_chunks.unsqueeze(2).expand(-1, -1, chunk, -1).reshape(b, n_chunks * chunk, d)[:, :t]
        slow = slow_seq.unsqueeze(2).expand(-1, -1, chunk, -1).reshape(b, n_chunks * chunk, d)[:, :t]
        delta = self.mix(torch.cat([h, med.to(h.dtype), slow.to(h.dtype)], dim=-1))
        consistency = (med.float()[:, 1:] - med.float()[:, :-1]).pow(2).mean() if t > 1 else h.new_tensor(0.0)
        return h + float(self.cfg.slow_state_strength) * delta.to(h.dtype), {"timescale_consistency": consistency, "slow_state_norm": slow.float().pow(2).mean().detach()}


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
        slot_ctx = slots.mean(dim=1, keepdim=True).expand(-1, h.size(1), -1)
        x = torch.cat([h, slot_ctx], dim=-1)
        return torch.sigmoid(self.net(x.float()).to(h.dtype))


class OutputModeClassifier(nn.Module):
    """Learn when language scaffolding is useful and how much to use."""
    def __init__(self, cfg: NeedConfig):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend), nn.Linear(cfg.d_model, cfg.d_model), nn.SiLU(), nn.Linear(cfg.d_model, max(2, int(cfg.output_modes))))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h.mean(dim=1).float())


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
        slots, _, aux = self.block(h, mask=text_mask, need_token_context=False)
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


class NEEDBlock(nn.Module):
    def __init__(self, cfg: NeedConfig):
        super().__init__()
        self.cfg = cfg
        self.n1 = RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend)
        self.ret = StructuredDualRetention(cfg) if cfg.retention_impl == "ssd" else SelectiveRetention(cfg)
        self.n2 = RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend)
        self.conv = MultiScaleCausalConv(cfg.d_model, cfg.conv_kernel, cfg.n_conv_scales, cfg.dropout)
        self.n3 = RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend)
        self.mem = HierarchicalMemory(cfg)
        self.n4 = RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend)
        self.eq = AdaptiveEquilibrium(cfg)
        self.n5 = RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend)
        self.moe = SparseMoE(cfg)
        self.depth_gate = AdaptiveDepthGate(cfg)
        self.scales = nn.Parameter(torch.ones(5) * cfg.layer_scale_init)

    def _residual_add(self, x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        full_scale = self.cfg.residual_scale * scale
        if need_kernels is not None and not torch.is_grad_enabled():
            return need_kernels.residual_scale_add(x.contiguous(), y.contiguous(), float(full_scale.detach().cpu()), self.cfg.kernel_backend)
        return x + full_scale * y

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        aux: Dict[str, torch.Tensor] = {}
        x = self._residual_add(x, self.ret(self.n1(x)), self.scales[0])
        y, caux = self.conv(self.n2(x)); aux.update(caux)
        x = self._residual_add(x, y, self.scales[1])
        gates, gaux = self.depth_gate(x); aux.update(gaux)
        mem_gate, eq_gate, moe_gate = gates[..., 0:1], gates[..., 1:2], gates[..., 2:3]
        y, maux = self.mem(self.n3(x)); aux.update(maux)
        x = self._residual_add(x, y * mem_gate, self.scales[2])
        n4x = self.n4(x)
        z, resid, energy, effort = self.eq(n4x, x)
        x = self._residual_add(x, (z - n4x) * eq_gate, self.scales[3])
        y, eaux = self.moe(self.n5(x)); aux.update(eaux)
        x = self._residual_add(x, y * moe_gate, self.scales[4])
        aux["equilibrium_residual"] = resid
        aux["energy"] = energy
        aux["adaptive_effort"] = effort
        aux["energy_row_orth"] = self.eq.energy.row_orth_loss()
        return x, aux


def geodesic_path_loss(traj: Sequence[torch.Tensor], target: float) -> torch.Tensor:
    if len(traj) < 2:
        return traj[0].new_tensor(0.0)
    steps = [(b.float() - a.float()).pow(2).mean(dim=-1).sqrt() for a, b in zip(traj[:-1], traj[1:])]
    lens = torch.stack(steps, dim=0)
    return (lens - target).pow(2).mean()


def path_straightness_loss(traj: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(traj) < 3:
        return traj[0].new_tensor(0.0)
    total = sum((b.float() - a.float()).pow(2).mean(dim=-1).sqrt() for a, b in zip(traj[:-1], traj[1:]))
    chord = (traj[-1].float() - traj[0].float()).pow(2).mean(dim=-1).sqrt().clamp_min(1e-8)
    return ((total / chord) - 1.0).pow(2).mean()


def contractive_path_loss(traj: Sequence[torch.Tensor], kappa: float) -> torch.Tensor:
    if len(traj) < 3:
        return traj[0].new_tensor(0.0)
    deltas = [(b.float() - a.float()).pow(2).mean(dim=-1).sqrt() for a, b in zip(traj[:-1], traj[1:])]
    losses = [F.relu(deltas[i + 1] - kappa * deltas[i]).pow(2).mean() for i in range(len(deltas) - 1)]
    return torch.stack(losses).mean() if losses else traj[0].new_tensor(0.0)




class ExactAssociativeRecall(nn.Module):
    """Chunked exact long-range associative recall.

    The recurrent/SSD state is efficient but lossy. This module gives the model a
    separate exact-memory path over the current context: every previous token can
    be queried directly through projected hidden keys plus deterministic token
    and bigram match biases. It is not self-attention for modeling the
    whole sequence; it is a sparse top-k content-addressable memory used once as
    a recall correction, with chunking so long contexts do not require a full
    B*T*T materialization.
    """
    def __init__(self, cfg: NeedConfig):
        super().__init__()
        self.cfg = cfg
        rd = int(max(8, min(cfg.d_model, cfg.exact_recall_dim)))
        self.norm = RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend)
        self.q = nn.Linear(cfg.d_model, rd, bias=False)
        self.k = nn.Linear(cfg.d_model, rd, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.token_key = nn.Embedding(cfg.vocab_size, rd)
        self.token_value = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.gate = nn.Sequential(RMSNorm(cfg.d_model, kernel_backend=cfg.kernel_backend), nn.Linear(cfg.d_model, 1))

    def forward(self, h: torch.Tensor, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.cfg.exact_recall or h.size(1) < 2:
            z = h.new_tensor(0.0)
            return h, {"exact_recall_entropy": z, "exact_recall_peak": z, "exact_recall_gate": z}
        b, t, d = h.shape
        max_t = min(t, int(self.cfg.exact_recall_max_tokens))
        if max_t < t:
            h_mem = h[:, -max_t:]
            ids_mem = input_ids[:, -max_t:]
        else:
            h_mem = h
            ids_mem = input_ids
        tm = h_mem.size(1)
        hn = self.norm(h_mem)
        q = F.normalize(self.q(hn.float()), dim=-1)
        k = F.normalize(self.k(hn.float()) + self.token_key(ids_mem).float(), dim=-1)
        v = self.v(hn).to(h.dtype) + 0.10 * self.token_value(ids_mem).to(h.dtype)
        outs: List[torch.Tensor] = []
        ent_parts: List[torch.Tensor] = []
        peak_parts: List[torch.Tensor] = []
        chunk = max(16, int(self.cfg.exact_recall_chunk_size))
        scale = 1.0 / max(1e-4, float(self.cfg.exact_recall_temperature))
        arange_mem = torch.arange(tm, device=h.device)
        for start in range(0, tm, chunk):
            end = min(tm, start + chunk)
            qc = q[:, start:end]
            scores = torch.matmul(qc, k.transpose(1, 2)) * scale
            # Causal exact-memory mask: a token can only recall earlier positions.
            qpos = torch.arange(start, end, device=h.device).view(1, -1, 1)
            kpos = arange_mem.view(1, 1, -1)
            scores = scores.masked_fill(kpos >= qpos, -1e9)
            tok_match = ids_mem[:, start:end].unsqueeze(-1) == ids_mem.unsqueeze(1)
            scores = scores + float(self.cfg.exact_recall_token_bias) * tok_match.to(scores.dtype)
            if start > 0 or tm > 1:
                prev_q = torch.roll(ids_mem[:, start:end], shifts=1, dims=1)
                if start == 0:
                    prev_q[:, 0] = self.cfg.pad_id
                else:
                    prev_q[:, 0] = ids_mem[:, start - 1]
                prev_k = torch.roll(ids_mem, shifts=1, dims=1)
                prev_k[:, 0] = self.cfg.pad_id
                bi_match = tok_match & (prev_q.unsqueeze(-1) == prev_k.unsqueeze(1))
                scores = scores + float(self.cfg.exact_recall_bigram_bias) * bi_match.to(scores.dtype)
            kk = min(max(1, int(self.cfg.exact_recall_top_k)), tm)
            vals, idx = torch.topk(scores, k=kk, dim=-1)
            valid = vals > -1e8
            weights = F.softmax(vals, dim=-1) * valid.to(vals.dtype)
            weights = (weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)).to(h.dtype)
            gather_idx = idx.unsqueeze(-1).expand(-1, -1, -1, d)
            vg = v.unsqueeze(1).expand(-1, end - start, -1, -1).gather(2, gather_idx)
            out = (vg * weights.unsqueeze(-1)).sum(dim=2)
            outs.append(out)
            probs = weights.float()
            ent_parts.append(-(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean())
            peak_parts.append(probs.max(dim=-1).values.mean())
        recalled = torch.cat(outs, dim=1)
        if max_t < t:
            pad = h.new_zeros(b, t - max_t, d)
            recalled = torch.cat([pad, recalled], dim=1)
        gate = torch.sigmoid(self.gate(h.float())).to(h.dtype) * float(self.cfg.exact_recall_mix)
        y = h + gate * self.out(recalled)
        ent = torch.stack(ent_parts).mean() if ent_parts else h.new_tensor(0.0)
        peak = torch.stack(peak_parts).mean() if peak_parts else h.new_tensor(0.0)
        return y, {"exact_recall_entropy": ent, "exact_recall_peak": peak.detach(), "exact_recall_gate": gate.float().mean().detach()}


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
        rms = y.float().pow(2).mean(dim=-1).sqrt()
        norm_err = (rms - float(self.cfg.state_norm_target)).pow(2).mean()
        chunk = max(4, int(self.cfg.state_drift_chunk))
        b, t, d = y.shape
        pad = (chunk - (t % chunk)) % chunk
        yp = F.pad(y.float(), (0, 0, 0, pad)) if pad else y.float()
        n_chunks = yp.size(1) // chunk
        chunks = yp.view(b, n_chunks, chunk, d).mean(dim=2)
        if n_chunks > 1:
            step = (chunks[:, 1:] - chunks[:, :-1]).pow(2).mean(dim=-1).sqrt()
            drift = F.relu(step - float(self.cfg.state_drift_target)).pow(2).mean()
        else:
            drift = y.new_tensor(0.0)
        anchor_err = (F.normalize(y.float(), dim=-1) - F.normalize(anchor.float(), dim=-1)).pow(2).mean()
        return y, {"state_norm_error": norm_err, "state_chunk_drift": drift, "state_anchor_error": anchor_err, "state_correction_gate": gate.float().mean().detach()}


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
        self.objective_balancer = AuxiliaryObjectiveBalancer(cfg, OBJECTIVE_LOSS_GROUPS.keys())
        self.apply(self._init_weights)

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
        rows = torch.zeros(b, t, dtype=torch.long, device=ids.device)
        cols = torch.zeros(b, t, dtype=torch.long, device=ids.device)
        mask = torch.zeros(b, t, dtype=torch.bool, device=ids.device)
        has_image_tokens = (ids == self.cfg.img_bos_id) | (ids == self.cfg.img_mask_id) | ((ids >= self.cfg.image_token_offset) & (ids < self.cfg.image_token_offset + self.cfg.image_codebook_size))
        if not bool(has_image_tokens.any()):
            return rows, cols, mask
        max_grid = max(1, int(self.cfg.image_max_grid))
        for batch in range(b):
            count = 0
            inside = False
            img_positions: List[int] = []
            for pos in range(t):
                tok = int(ids[batch, pos].detach().cpu())
                if tok == self.cfg.img_bos_id:
                    inside = True
                    count = 0
                    img_positions = []
                    continue
                if tok == self.cfg.img_eos_id or tok == self.cfg.pad_id:
                    inside = False
                is_img = inside and (tok == self.cfg.img_mask_id or self.cfg.image_token_offset <= tok < self.cfg.image_token_offset + self.cfg.image_codebook_size)
                if is_img:
                    img_positions.append(pos)
            if img_positions:
                grid = min(max_grid, max(1, int(round(math.sqrt(len(img_positions))))))
                for n, pos in enumerate(img_positions):
                    rows[batch, pos] = min(max_grid - 1, n // grid)
                    cols[batch, pos] = min(max_grid - 1, n % grid)
                    mask[batch, pos] = True
        return rows, cols, mask

    def encode_hidden(self, input_ids: torch.Tensor, modality_ids: Optional[torch.Tensor] = None, conditioning_vectors: Optional[torch.Tensor] = None, conditioning_scale: float = 0.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B,T]")
        b, t = input_ids.shape
        if t > self.cfg.block_size:
            raise ValueError(f"sequence length {t} exceeds block_size {self.cfg.block_size}")
        if modality_ids is None:
            modality_ids = self.modality_ids_from_tokens(input_ids)
        pos = torch.arange(t, device=input_ids.device).view(1, t)
        x = self.token_emb(input_ids) + self.pos_emb(pos) + self.modality_emb(modality_ids.clamp(0, 3))
        if float(self.cfg.image_coord_scale) != 0.0:
            rows, cols, coord_mask = self.image_coordinate_ids(input_ids)
            if bool(coord_mask.any()):
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
            denom = mask.float().sum().clamp_min(1.0)
            ce = (ce_all * mask.float()).sum() / denom
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

            # MTP
            mtp = logits.new_tensor(0.0)
            max_heads = min(self.cfg.n_predict_heads, future_targets.size(-1))
            mtp_parts = []
            for i in range(1, max_heads):
                pred = self.lm_head(self.mtp_projs[i - 1](h_dec))
                target_i = future_targets[..., i]
                mtp_parts.append(F.cross_entropy(pred.reshape(-1, self.cfg.vocab_size), target_i.reshape(-1), ignore_index=self.cfg.pad_id, label_smoothing=self.cfg.label_smoothing))
            if mtp_parts and self.cfg.lambda_mtp > 0:
                mtp = torch.stack(mtp_parts).mean()
                loss = _add_aux(loss, self.cfg.lambda_mtp, mtp, "mtp")
            aux["mtp"] = mtp.detach()

            # DVSD-native training. MTP learns future heads; these objectives teach
            # the direct-commit virtual-slot decoder to keep those heads stable and
            # teach a router to choose a local slot budget.
            if max_heads > 1 and (self.cfg.lambda_dvsd_slot_ce != 0 or self.cfg.lambda_dvsd_consistency != 0 or self.cfg.lambda_dvsd_router != 0):
                dvsd_aux = self._dvsd_training_objectives(h_dec, logits, future_targets, next_targets, ce_all, mask)
                if self.cfg.lambda_dvsd_slot_ce != 0:
                    loss = _add_aux(loss, self.cfg.lambda_dvsd_slot_ce, dvsd_aux["dvsd_slot_ce"], "dvsd_slot_ce")
                else:
                    aux["dvsd_slot_ce"] = dvsd_aux["dvsd_slot_ce"].detach()
                if self.cfg.lambda_dvsd_consistency != 0:
                    loss = _add_aux(loss, self.cfg.lambda_dvsd_consistency, dvsd_aux["dvsd_consistency"], "dvsd_consistency")
                else:
                    aux["dvsd_consistency"] = dvsd_aux["dvsd_consistency"].detach()
                if self.cfg.lambda_dvsd_router != 0:
                    router_term = dvsd_aux["dvsd_router"] - float(self.cfg.dvsd_router_entropy_weight) * dvsd_aux["dvsd_router_entropy"]
                    loss = _add_aux(loss, self.cfg.lambda_dvsd_router, router_term, "dvsd_router")
                    aux["dvsd_router_ce"] = dvsd_aux["dvsd_router"].detach()
                    aux["dvsd_router_entropy"] = dvsd_aux["dvsd_router_entropy"].detach()
                else:
                    aux["dvsd_router"] = dvsd_aux["dvsd_router"].detach()
                aux["dvsd_router_target_slots"] = dvsd_aux["dvsd_router_target_slots"].detach()
                aux["dvsd_router_pred_slots"] = dvsd_aux["dvsd_router_pred_slots"].detach()
            # planner losses
            lat_parts, pce_parts, cons_parts = [], [], []
            for horizon, pred_h in enumerate(plan, start=1):
                if input_ids.size(1) <= horizon:
                    continue
                pred_slice = pred_h[:, :-horizon]
                target_h = h_dec[:, horizon:].detach()
                lat_parts.append(0.5 * ((pred_slice.float() - target_h.float()).pow(2).mean() + (1 - F.cosine_similarity(pred_slice.float(), target_h.float(), dim=-1).mean())))
                if future_targets.size(-1) >= horizon:
                    pred_logits = self.lm_head(pred_slice)
                    target_tok = future_targets[:, :-horizon, horizon - 1]
                    pce_parts.append(F.cross_entropy(pred_logits.reshape(-1, self.cfg.vocab_size), target_tok.reshape(-1), ignore_index=self.cfg.pad_id, label_smoothing=self.cfg.label_smoothing))
                if horizon > 1:
                    prev = plan[horizon - 2][:, 1: input_ids.size(1) - horizon + 1]
                    cur = pred_slice
                    if prev.shape == cur.shape:
                        cons_parts.append((cur.float() - prev.detach().float()).pow(2).mean())
            if lat_parts:
                lpl = torch.stack(lat_parts).mean(); loss = _add_aux(loss, self.cfg.lambda_latent_planning, lpl, "latent_planning"); aux["latent_planning"] = lpl.detach()
            if pce_parts:
                pcl = torch.stack(pce_parts).mean(); loss = _add_aux(loss, self.cfg.lambda_planner_ce, pcl, "planner_ce"); aux["planner_ce"] = pcl.detach()
            if cons_parts:
                csl = torch.stack(cons_parts).mean(); loss = _add_aux(loss, self.cfg.lambda_planning_consistency, csl, "planning_consistency"); aux["planning_consistency"] = csl.detach()
            # Latent slot losses keep reusable slots aligned with future chunks.
            if self.cfg.lambda_latent_slot > 0 and h_dec.size(1) > 1:
                future_pool = h_dec[:, h_dec.size(1)//2:].detach().mean(dim=1) if h_dec.size(1) > 2 else h_dec[:, -1].detach()
                slot_pool = latent_slots.mean(dim=1)
                slot_loss = 0.5 * ((slot_pool.float() - future_pool.float()).pow(2).mean() + (1.0 - F.cosine_similarity(slot_pool.float(), future_pool.float(), dim=-1).mean()))
                loss = _add_aux(loss, self.cfg.lambda_latent_slot, slot_loss, "latent_slot"); aux["latent_slot"] = slot_loss.detach()
            if self.cfg.lambda_latent_slot_diversity != 0 and "latent_slot_diversity" in aux:
                loss = _add_aux(loss, self.cfg.lambda_latent_slot_diversity, aux["latent_slot_diversity"], "latent_slot_diversity")
            if self.cfg.lambda_risk_signal > 0:
                # Risk signal is trained to correlate with per-token prediction error.
                r_target = (ce_all.detach().clamp(max=8.0) / 8.0).unsqueeze(-1)
                rloss = F.mse_loss(risk_signal[mask], r_target[mask]) if bool(mask.any()) else risk_signal.mean() * 0
                loss = _add_aux(loss, self.cfg.lambda_risk_signal, rloss, "risk_signal"); aux["risk_signal"] = rloss.detach()
            if self.cfg.lambda_latent_divergence > 0:
                # Self-supervised divergence target: high when latent slots and current
                # trajectory disagree strongly or aux-score risk is high.
                with torch.no_grad():
                    slot_ctx = latent_slots.mean(dim=1, keepdim=True).expand(-1, h_dec.size(1), -1)
                    align_err = (1.0 - F.cosine_similarity(slot_ctx.float(), h_dec.float(), dim=-1)).clamp(0, 2) / 2
                    dtarget = torch.maximum(align_err, (risk.detach().clamp(max=8.0) / 8.0)).unsqueeze(-1)
                dloss = F.binary_cross_entropy(latent_divergence[mask].clamp(1e-5, 1-1e-5), dtarget[mask].clamp(0, 1)) if bool(mask.any()) else latent_divergence.mean() * 0
                loss = _add_aux(loss, self.cfg.lambda_latent_divergence, dloss, "latent_divergence_loss"); aux["latent_divergence_loss"] = dloss.detach()
            if self.cfg.lambda_output_mode_classifier > 0:
                # Policy target is heuristic but grounded: low-risk/easy contexts prefer no
                # CoT, high uncertainty/risk contexts prefer fuller reasoning scaffolds.
                with torch.no_grad():
                    score = (risk_signal.mean(dim=1).squeeze(-1) + (risk.clamp(max=8.0)/8.0).mean(dim=1) + contradiction.mean(dim=1)) / 3.0
                    n_modes = max(2, int(self.cfg.output_modes))
                    target_mode = torch.clamp((score * n_modes).long(), 0, n_modes - 1)
                if output_mode_logits is None:
                    output_mode_logits = self.output_mode_classifier(h_dec)
                vploss = F.cross_entropy(output_mode_logits, target_mode)
                loss = _add_aux(loss, self.cfg.lambda_output_mode_classifier, vploss, "output_mode_classifier"); aux["output_mode_classifier"] = vploss.detach()
            if self.cfg.lambda_mixture_energy_router != 0 and "mixture_energy_router_energy" in aux:
                loss = _add_aux(loss, self.cfg.lambda_mixture_energy_router, aux["mixture_energy_router_energy"], "mixture_energy_router_energy")
            if self.cfg.lambda_timescale != 0 and "timescale_consistency" in aux:
                loss = _add_aux(loss, self.cfg.lambda_timescale, aux["timescale_consistency"], "timescale_consistency")

            # Anti-degeneration entropy bands. These are deliberately weak and
            # band-pass rather than maximize-entropy objectives: collapsed routing
            # and completely uniform routing are both unhelpful.
            entropy_band_w = float(getattr(self.cfg, "objective_entropy_band_weight", 0.0))
            if entropy_band_w > 0.0:
                if "energy_route_entropy" in aux and self.cfg.energy_routes > 1 and self.cfg.lambda_mixture_energy_router != 0:
                    eband = AuxiliaryObjectiveBalancer.entropy_band_loss(aux["energy_route_entropy"], self.cfg.energy_routes, 0.20, 0.90)
                    loss = _add_aux(loss, entropy_band_w * abs(float(self.cfg.lambda_mixture_energy_router)), eband, "energy_route_entropy_band")
                if "latent_slot_attention_entropy" in aux and self.cfg.latent_slots > 1 and self.cfg.lambda_latent_slot != 0:
                    sband = AuxiliaryObjectiveBalancer.entropy_band_loss(aux["latent_slot_attention_entropy"], self.cfg.latent_slots, 0.25, 0.92)
                    loss = _add_aux(loss, entropy_band_w * abs(float(self.cfg.lambda_latent_slot)), sband, "latent_slot_entropy_band")
                if output_mode_logits is not None and self.cfg.output_modes > 1 and self.cfg.lambda_output_mode_classifier != 0:
                    om_prob = F.softmax(output_mode_logits.float(), dim=-1)
                    om_ent = -(om_prob * om_prob.clamp_min(1e-8).log()).sum(dim=-1).mean()
                    aux["output_mode_entropy"] = om_ent.detach()
                    oband = AuxiliaryObjectiveBalancer.entropy_band_loss(om_ent, self.cfg.output_modes, 0.15, 0.88)
                    loss = _add_aux(loss, entropy_band_w * abs(float(self.cfg.lambda_output_mode_classifier)), oband, "output_mode_entropy_band")

            # latent diffusion denoising: perturb hidden and ask it to reconstruct clean token logits via same head
            if self.cfg.lambda_diffusion > 0:
                noise = torch.randn_like(h_dec)
                sigma = torch.rand(h_dec.size(0), h_dec.size(1), 1, device=h_dec.device, dtype=h_dec.dtype) * 0.5
                noisy = h_dec + sigma * noise
                denoised_logits = self.lm_head(noisy - sigma * noise.detach() * 0.25)
                diff = F.cross_entropy(denoised_logits.reshape(-1, self.cfg.vocab_size), next_targets.reshape(-1), ignore_index=self.cfg.pad_id, label_smoothing=self.cfg.label_smoothing)
                loss = _add_aux(loss, self.cfg.lambda_diffusion, diff, "diffusion"); aux["diffusion"] = diff.detach()
            # masked image token diffusion loss. NEED uses same-position
            # image_targets instead of next-token targets, so image-token
            # diffusion learns to reconstruct the token that was actually masked.
            if image_mask_positions is not None and self.cfg.lambda_image_diffusion > 0:
                img_target = image_targets if image_targets is not None else next_targets
                if img_target.ndim == 3:
                    img_target = img_target[..., 0]
                img_mask = image_mask_positions.bool() & (img_target >= self.cfg.image_token_offset)
                if bool(img_mask.any()):
                    img_ce = F.cross_entropy(
                        logits[img_mask][:, self.cfg.image_token_offset:self.cfg.image_token_offset+self.cfg.image_codebook_size],
                        (img_target[img_mask] - self.cfg.image_token_offset).clamp_min(0),
                    )
                    loss = _add_aux(loss, self.cfg.lambda_image_diffusion, img_ce, "image_diffusion")
                    aux["image_diffusion"] = img_ce.detach()
            # Fine-grained language-image alignment. Skip the modality scan entirely
            # for text-only runs where all image/object losses are disabled.
            image_alignment_active = (
                self.cfg.lambda_image_contrastive > 0
                or self.cfg.lambda_image_local_contrastive > 0
                or self.cfg.lambda_region_word_alignment > 0
                or self.cfg.lambda_image_spatial_smoothness > 0
                or self.cfg.lambda_object_program > 0
            )
            if image_alignment_active:
                mod = self.modality_ids_from_tokens(input_ids)
                text_mask = (mod == 0) & (input_ids >= Special.byte_start)
                img_mask_for_align = mod == 1
                has_text = bool(text_mask.any())
                has_img = bool(img_mask_for_align.any())
                if self.cfg.lambda_image_contrastive > 0 and has_text and has_img:
                    text_vec = (h_dec * text_mask.unsqueeze(-1)).sum(dim=1) / text_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
                    img_vec = (h_dec * img_mask_for_align.unsqueeze(-1)).sum(dim=1) / img_mask_for_align.float().sum(dim=1, keepdim=True).clamp_min(1.0)
                    sim = F.normalize(self.text_proj(text_vec.float()), dim=-1) @ F.normalize(self.image_proj(img_vec.float()), dim=-1).t()
                    labels = torch.arange(sim.size(0), device=sim.device)
                    con = 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))
                    loss = _add_aux(loss, self.cfg.lambda_image_contrastive, con, "image_contrastive"); aux["image_contrastive"] = con.detach()
                if self.cfg.lambda_image_local_contrastive > 0 and has_text and has_img:
                    local = self.local_image_text_contrastive_loss(h_dec, input_ids, text_mask, img_mask_for_align)
                    loss = _add_aux(loss, self.cfg.lambda_image_local_contrastive, local, "image_local_contrastive")
                    aux["image_local_contrastive"] = local.detach()
                if self.cfg.lambda_region_word_alignment > 0 and has_text and has_img:
                    rwa = self.region_word_alignment_loss(h_dec, input_ids, text_mask, img_mask_for_align)
                    loss = _add_aux(loss, self.cfg.lambda_region_word_alignment, rwa, "region_word_alignment")
                    aux["region_word_alignment"] = rwa.detach()
                if self.cfg.lambda_image_spatial_smoothness > 0 and has_img:
                    smooth = self.image_spatial_smoothness_loss(h_dec, input_ids, img_mask_for_align)
                    loss = _add_aux(loss, self.cfg.lambda_image_spatial_smoothness, smooth, "image_spatial_smoothness")
                    aux["image_spatial_smoothness"] = smooth.detach()
                if self.cfg.lambda_object_program > 0 and has_img:
                    obj_slots, obj_aux = self.object_program(h_dec, text_mask=text_mask)
                    # Encourage object slots to cover image/text features and remain non-collapsed.
                    obj_norm = F.normalize(obj_slots.float(), dim=-1)
                    obj_gram = torch.einsum('bsd,bqd->bsq', obj_norm, obj_norm)
                    eye = torch.eye(obj_gram.size(-1), device=obj_gram.device, dtype=obj_gram.dtype).unsqueeze(0)
                    obj_loss = (obj_gram - eye).pow(2).mean() + 0.1 * (1.0 - obj_aux["object_coverage"].float())
                    loss = _add_aux(loss, self.cfg.lambda_object_program, obj_loss, "object_program")
                    aux["object_program"] = obj_loss.detach()
                    for ok, ov in obj_aux.items():
                        aux[ok] = ov.detach() if torch.is_tensor(ov) else ov
                    if float(getattr(self.cfg, "objective_entropy_band_weight", 0.0)) > 0.0 and "object_slot_entropy" in aux:
                        oband = AuxiliaryObjectiveBalancer.entropy_band_loss(aux["object_slot_entropy"], self.cfg.object_program_slots, 0.20, 0.90)
                        loss = _add_aux(loss, float(self.cfg.objective_entropy_band_weight) * abs(float(self.cfg.lambda_object_program)), oband, "object_slot_entropy_band")
            # aux_score target from CE/correctness; no raw CoT or labels needed
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                correct = (pred == next_targets).float()
                risk_target = ce_all.detach().clamp(max=8.0) / 8.0
            v_loss = F.binary_cross_entropy_with_logits(quality[mask], correct[mask]) if bool(mask.any()) else quality.mean() * 0
            v_loss = v_loss + F.mse_loss(risk[mask].clamp(max=8.0) / 8.0, risk_target[mask]) if bool(mask.any()) else v_loss
            if self.cfg.aux_score_controller and vf.size(-1) >= 9 and bool(mask.any()):
                with torch.no_grad():
                    # 0 answer, 1 deepen, 2 retrieve, 3 revise. High loss favors revise/deepen;
                    # high inferred topic boundary favors retrieval/working-memory refresh.
                    boundary = block_aux.get("memory_boundary", torch.zeros((), device=input_ids.device)).detach()
                    ctrl_target = torch.zeros_like(next_targets)
                    ctrl_target = torch.where(risk_target > 0.65, torch.full_like(ctrl_target, 3), ctrl_target)
                    ctrl_target = torch.where((risk_target > 0.35) & (risk_target <= 0.65), torch.full_like(ctrl_target, 1), ctrl_target)
                    if torch.is_tensor(boundary) and boundary.numel() == 1 and float(boundary.detach().cpu()) > 0.45:
                        ctrl_target = torch.where(mask, torch.full_like(ctrl_target, 2), ctrl_target)
                ctrl_logits = vf[..., 5:9]
                c_loss = F.cross_entropy(ctrl_logits.reshape(-1, 4), ctrl_target.reshape(-1), ignore_index=self.cfg.pad_id)
                v_loss = v_loss + self.cfg.lambda_controller * c_loss
                aux["controller"] = c_loss.detach()
            loss = _add_aux(loss, self.cfg.lambda_aux_score, v_loss, "aux_score"); aux["aux_score"] = v_loss.detach()
            # regularizers
            reg_map = {
                "equilibrium_residual": self.cfg.lambda_equilibrium,
                "energy": self.cfg.lambda_energy,
                "moe_balance": self.cfg.lambda_moe_balance,
                "moe_router_z": self.cfg.lambda_moe_router_z,
                "branch_entropy": -self.cfg.lambda_branch_entropy,
                "conv_scale_entropy": -self.cfg.lambda_conv_scale_entropy,
                "geodesic": self.cfg.lambda_geodesic,
                "path_straightness": self.cfg.lambda_path_straightness,
                "path_contractive": self.cfg.lambda_path_contractive,
                "energy_row_orth": self.cfg.lambda_energy_row_orth,
                "latent_norm": self.cfg.lambda_latent_norm,
                "memory_entropy": -self.cfg.lambda_memory_entropy,
                "memory_diversity": self.cfg.lambda_memory_diversity,
                "adaptive_effort": self.cfg.lambda_adaptive_effort,
                "compute_budget": self.cfg.lambda_compute_budget,
                "pathway_memory_entropy": -self.cfg.lambda_pathway_memory,
                "energy_route_balance": 0.001 * self.cfg.lambda_mixture_energy_router,
            }
            for k, lam in reg_map.items():
                if lam != 0 and k in aux:
                    loss = _add_aux(loss, lam, aux[k], k)
            if self.cfg.lambda_state_drift != 0:
                if "state_chunk_drift" in aux:
                    loss = _add_aux(loss, self.cfg.lambda_state_drift, aux["state_chunk_drift"] + 0.25 * aux.get("state_norm_error", torch.zeros_like(aux["state_chunk_drift"])), "state_drift")
            if self.cfg.lambda_state_anchor != 0 and "state_anchor_error" in aux:
                loss = _add_aux(loss, self.cfg.lambda_state_anchor, aux["state_anchor_error"], "state_anchor")
            if self.cfg.lambda_recall_entropy != 0 and "exact_recall_entropy" in aux:
                # Encourage recall to stay sparse but not one-hot/binary. Entropy
                # below a small target indicates collapse to a single rote slot.
                target_ent = math.log(max(2, int(self.cfg.exact_recall_top_k))) * 0.35
                rec_loss = F.relu(torch.as_tensor(target_ent, device=input_ids.device, dtype=aux["exact_recall_entropy"].dtype) - aux["exact_recall_entropy"]).pow(2)
                loss = _add_aux(loss, self.cfg.lambda_recall_entropy, rec_loss, "exact_recall_entropy_floor")
            _finish_objective_diagnostics()
        return logits, loss, aux

    def local_image_text_contrastive_loss(self, h: torch.Tensor, input_ids: torch.Tensor, text_mask: torch.Tensor, img_mask: torch.Tensor) -> torch.Tensor:
        text_feats: List[torch.Tensor] = []
        img_feats: List[torch.Tensor] = []
        text_owner: List[torch.Tensor] = []
        img_owner: List[torch.Tensor] = []
        max_text = 96
        max_img = 256
        for b in range(h.size(0)):
            ti = torch.nonzero(text_mask[b], as_tuple=False).flatten()[:max_text]
            ii = torch.nonzero(img_mask[b], as_tuple=False).flatten()[:max_img]
            if ti.numel() == 0 or ii.numel() == 0:
                continue
            text_feats.append(self.text_proj(h[b, ti].float()))
            img_feats.append(self.image_proj(h[b, ii].float()))
            text_owner.append(torch.full((ti.numel(),), b, device=h.device, dtype=torch.long))
            img_owner.append(torch.full((ii.numel(),), b, device=h.device, dtype=torch.long))
        if not text_feats or not img_feats:
            return h.new_tensor(0.0)
        tx = F.normalize(torch.cat(text_feats, dim=0), dim=-1)
        ix = F.normalize(torch.cat(img_feats, dim=0), dim=-1)
        tb = torch.cat(text_owner, dim=0)
        ib = torch.cat(img_owner, dim=0)
        sim = tx @ ix.t() / max(1e-4, float(self.cfg.image_local_contrastive_temperature))
        pos = tb[:, None] == ib[None, :]
        log_denom_t = torch.logsumexp(sim, dim=-1)
        log_pos_t = torch.logsumexp(sim.masked_fill(~pos, -1e9), dim=-1)
        log_denom_i = torch.logsumexp(sim.t(), dim=-1)
        log_pos_i = torch.logsumexp(sim.t().masked_fill(~pos.t(), -1e9), dim=-1)
        return 0.5 * ((log_denom_t - log_pos_t).mean() + (log_denom_i - log_pos_i).mean())

    def _sinkhorn(self, scores: torch.Tensor, iters: int) -> torch.Tensor:
        # scores [T,I]; returns soft doubly-normalized transport plan.
        plan = torch.exp(scores - scores.max()).clamp_min(1e-12)
        for _ in range(max(1, int(iters))):
            plan = plan / plan.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            plan = plan / plan.sum(dim=-2, keepdim=True).clamp_min(1e-8)
        return plan

    def region_word_alignment_loss(self, h: torch.Tensor, input_ids: torch.Tensor, text_mask: torch.Tensor, img_mask: torch.Tensor) -> torch.Tensor:
        """Fine-grained region-word transport loss.

        Local contrastive alignment marks every word/patch in the same sample as positive.
        This loss assigns specific word tokens to specific visual patches with a soft
        Sinkhorn transport plan, then encourages assigned pairs to have high similarity
        while keeping the plan sharp enough to preserve grounding.
        """
        losses: List[torch.Tensor] = []
        tau = max(1e-4, float(self.cfg.image_local_contrastive_temperature))
        for b in range(h.size(0)):
            ti = torch.nonzero(text_mask[b], as_tuple=False).flatten()[:96]
            ii = torch.nonzero(img_mask[b], as_tuple=False).flatten()[:256]
            if ti.numel() == 0 or ii.numel() == 0:
                continue
            tx = F.normalize(self.text_proj(h[b, ti].float()), dim=-1)
            ix = F.normalize(self.image_proj(h[b, ii].float()), dim=-1)
            sim = tx @ ix.t() / tau
            plan = self._sinkhorn(sim, self.cfg.region_word_sinkhorn_iters)
            align = -(plan.detach() * F.log_softmax(sim, dim=-1)).sum(dim=-1).mean()
            align = align - 0.25 * (plan.detach() * F.log_softmax(sim.t(), dim=-1).t()).sum(dim=-2).mean()
            entropy = -(plan * plan.clamp_min(1e-8).log()).sum() / max(1, plan.numel())
            losses.append(align + 0.02 * entropy)
        return torch.stack(losses).mean() if losses else h.new_tensor(0.0)

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
        h, aux, traj = self.encode_hidden(input_ids)
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
        h, _, _ = self.encode_hidden(input_ids.to(next(self.parameters()).device))
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
        router_pred = next_logits.new_tensor(float("nan"))
        router_conf = next_logits.new_tensor(float("nan"))
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
            "dvsd_router_pred_heads": float(router_pred.detach().cpu()) if torch.isfinite(router_pred).all() else float("nan"),
            "dvsd_router_confidence": float(router_conf.detach().cpu()) if torch.isfinite(router_conf).all() else float("nan"),
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
        if no_repeat_ngram > 1:
            apply_no_repeat_ngram_(local, prefix_ids, no_repeat_ngram)
        if forbid_eos:
            local[:, int(eos_id)] = -float("inf")
        if temperature <= 0:
            filtered = top_k_top_p_filter(local, top_k=0, top_p=1.0)
            log_probs = F.log_softmax(filtered, dim=-1)
            probs = log_probs.exp()
        else:
            filtered = typical_filter(local / max(float(temperature), 1e-8), typical_p)
            filtered = top_k_top_p_filter(filtered, top_k=top_k, top_p=top_p)
            probs = F.softmax(filtered, dim=-1)
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
        if no_repeat_ngram > 1:
            apply_no_repeat_ngram_(local, prefix_ids, no_repeat_ngram)
        if forbid_eos:
            local[:, int(eos_id)] = -float("inf")
        if temperature <= 0:
            filtered = local
        else:
            filtered = typical_filter(local / max(float(temperature), 1e-8), typical_p)
            filtered = top_k_top_p_filter(filtered, top_k=top_k, top_p=top_p)
        text_logits = filtered[:, :vocab_cut]
        finite = torch.isfinite(text_logits).any(dim=-1, keepdim=True)
        if not bool(finite.all().detach().cpu()):
            fallback = local[:, :vocab_cut]
            text_logits = torch.where(finite, text_logits, fallback)
        probs = F.softmax(text_logits, dim=-1)
        bad = (~torch.isfinite(probs).all(dim=-1, keepdim=True)) | (probs.sum(dim=-1, keepdim=True) <= 0)
        if bool(bad.any().detach().cpu()):
            probs = F.softmax(local[:, :vocab_cut], dim=-1)
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

            canvas_parts: List[torch.Tensor] = []
            conf_values: List[torch.Tensor] = []
            entropy_values: List[torch.Tensor] = []
            gap_values: List[torch.Tensor] = []
            for pos, hlog in enumerate(head_logits):
                prefix = idx if not canvas_parts else torch.cat([idx, *canvas_parts], dim=1)
                tok, conf, ent, gap = self._nonseq_sample_slot_token(
                    hlog,
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
                canvas_parts.append(tok)
                conf_values.append(conf)
                entropy_values.append(ent)
                gap_values.append(gap)
            canvas = torch.cat(canvas_parts, dim=1)
            slot_conf = torch.stack(conf_values, dim=1).to(device=device)
            slot_entropy = torch.stack(entropy_values, dim=1).to(device=device)

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

            # Easy spans need only one or two passes. Moderate spans can use the full
            # refinement budget. Hard spans have already collapsed to one slot.
            if active_heads <= 1:
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
                    for pos in range(active_heads):
                        if bool(locked[0, pos].detach().cpu()) and not resample_locked:
                            continue
                        pred_abs = idx.size(1) + pos - 1
                        rel = pred_abs - offset
                        if 0 <= rel < ref_logits.size(1):
                            causal_logits = ref_logits[:, rel, :].float().clone()
                            causal_logits[:, self.cfg.image_token_offset:] = -float("inf")
                            mix = causal_blend * min(1.0, float(r) / max(1.0, float(refine_steps - 1)))
                            slot_logits = (1.0 - mix) * head_logits[pos] + mix * causal_logits
                        else:
                            slot_logits = head_logits[pos]
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
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        idx = input_ids.to(device=device, dtype=torch.long)
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
            if no_repeat_ngram > 1:
                apply_no_repeat_ngram_(next_logits, idx, no_repeat_ngram)
            if step < min_new_tokens:
                next_logits[:, eos_id] = -float("inf")
            if temperature <= 0:
                candidate_ids = torch.topk(next_logits, k=min(cand_pool, next_logits.size(-1)), dim=-1).indices
                candidate_lp = F.log_softmax(next_logits, dim=-1).gather(1, candidate_ids)
            else:
                filtered = typical_filter(next_logits / max(temperature, 1e-8), typical_p)
                filtered = top_k_top_p_filter(filtered, top_k, top_p)
                probs = F.softmax(filtered, dim=-1)
                sample_n = min(cand_pool if proactive or aux_score_weight > 0 else 1, probs.size(-1))
                if sample_n > 1:
                    candidate_ids = torch.topk(probs, k=sample_n, dim=-1).indices
                else:
                    candidate_ids = torch.multinomial(probs, num_samples=1)
                candidate_lp = torch.log(probs.gather(1, candidate_ids).clamp_min(1e-12))
            if (proactive or aux_score_top_k > 0 or aux_score_weight > 0) and idx.size(0) == 1:
                candidate_ids, candidate_lp, risks, contras = self._rank_candidates_with_aux_score(
                    idx, candidate_ids, candidate_lp, max(aux_score_weight, self.cfg.aux_score_logit_scale), conditioning_vectors, conditioning_scale, rollout_depth=rollout_depth
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
        h, _, _ = self.encode_hidden(input_ids.to(next(self.parameters()).device))
        probs = F.softmax(self.output_mode_classifier(h), dim=-1)[0].detach().float().cpu()
        names = ["none", "short_summary", "full_artificial_cot", "multi_cot", "renderer_only"]
        return {names[i] if i < len(names) else f"mode_{i}": float(probs[i]) for i in range(probs.numel())}

    @torch.no_grad()
    def score_latent_branch(self, input_ids: torch.Tensor, conditioning_vectors: Optional[torch.Tensor] = None, conditioning_scale: float = 0.0) -> Dict[str, float]:
        """Score a proposed branch/continuation with aux-score, energy, and risk signals."""
        ids = input_ids.to(next(self.parameters()).device)
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
        # Infer a coarse object/layout program from the text prompt. The program
        # conditions token diffusion through a lightweight logit bias so image
        # generation has an explicit semantic/layout scaffold before patch tokens.
        prompt_h, _, _ = self.encode_hidden(prompt[:, -self.cfg.block_size:])
        obj_slots, obj_aux = self.object_program(prompt_h, text_mask=(prompt[:, -prompt_h.size(1):] >= Special.byte_start))
        obj_bias = torch.tanh(obj_slots.mean(dim=1)) @ self.token_emb.weight[self.cfg.image_token_offset:self.cfg.image_token_offset+self.cfg.image_codebook_size].t()
        obj_bias = obj_bias.view(prompt.size(0), 1, self.cfg.image_codebook_size) * float(self.cfg.object_program_strength)
        neg_prompt = negative_prompt_ids.to(device=device, dtype=torch.long) if negative_prompt_ids is not None else None
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
            probs = F.softmax(filt, dim=-1)
            if not torch.isfinite(probs).all():
                probs = F.softmax(img_logits.float(), dim=-1)
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
    for b in range(logits.size(0)):
        seen = torch.unique(ids[b])
        seen = seen[(seen >= 0) & (seen < logits.size(-1))]
        vals = logits[b, seen]
        logits[b, seen] = torch.where(vals < 0, vals * penalty, vals / penalty)


def _banned_continuations(seq: List[int], n: int) -> List[int]:
    if n <= 1 or len(seq) < n - 1:
        return []
    prefix = tuple(seq[-(n - 1):])
    out = []
    for i in range(0, len(seq) - n + 1):
        if tuple(seq[i:i+n-1]) == prefix:
            out.append(seq[i+n-1])
    return out


def apply_no_repeat_ngram_(logits: torch.Tensor, ids: torch.Tensor, n: int) -> None:
    for b in range(logits.size(0)):
        banned = [x for x in _banned_continuations(ids[b].tolist(), n) if 0 <= x < logits.size(-1)]
        if banned:
            logits[b, banned] = -float("inf")


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
        # Low-data replay memory is adapter-like: prefer successful, low-risk
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
    cfg = NeedConfig(**{k: v for k, v in raw_cfg.items() if k in NeedConfig.__dataclass_fields__})
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
