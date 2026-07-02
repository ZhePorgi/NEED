#!/usr/bin/env python3
"""Optional fused attention and cache-planning helpers for small external LM sidecars.

These utilities are deliberately optional.  The NEED model can run without them, and
Hugging Face runtimes will normally use PyTorch SDPA or FlashAttention-2
through the model's own attention implementation.  This file gives a single place
for fused attention fallbacks, L2-aware KV cache planning, and future custom kernel
replacement.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class L2CachePlan:
    layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    dtype_bytes: int = 2
    l2_cache_mb: float = 96.0
    safety_fraction: float = 0.55
    max_batch: int = 1
    context_tokens: int = 2048

    @property
    def kv_bytes_per_token_per_batch(self) -> int:
        return int(self.layers * 2 * self.num_key_value_heads * self.head_dim * self.dtype_bytes)

    @property
    def kv_bytes_for_context(self) -> int:
        return int(self.kv_bytes_per_token_per_batch * self.context_tokens * self.max_batch)

    @property
    def l2_target_bytes(self) -> int:
        return int(self.l2_cache_mb * 1024 * 1024 * self.safety_fraction)

    @property
    def recommended_decode_tile_tokens(self) -> int:
        per_tok = max(1, self.kv_bytes_per_token_per_batch * max(1, self.max_batch))
        return max(16, min(self.context_tokens, self.l2_target_bytes // per_tok))

    @property
    def recommended_prefill_chunk_tokens(self) -> int:
        # Prefill is compute-heavy; keep QK/V blocks large enough for tensor cores but
        # below the target cache footprint when possible.
        t = self.recommended_decode_tile_tokens
        return int(max(128, min(4096, 2 ** int(math.log2(max(128, t))))))

    @property
    def recommended_batch_cap(self) -> int:
        per_ctx = max(1, self.kv_bytes_per_token_per_batch * max(1, self.context_tokens))
        return max(1, self.l2_target_bytes // per_ctx)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.update(
            kv_bytes_per_token_per_batch=self.kv_bytes_per_token_per_batch,
            kv_bytes_for_context=self.kv_bytes_for_context,
            l2_target_bytes=self.l2_target_bytes,
            recommended_decode_tile_tokens=self.recommended_decode_tile_tokens,
            recommended_prefill_chunk_tokens=self.recommended_prefill_chunk_tokens,
            recommended_batch_cap=self.recommended_batch_cap,
        )
        return d


def plan_from_config(config: Any, *, dtype: torch.dtype = torch.bfloat16, l2_cache_mb: float = 96.0, max_batch: int = 1, context_tokens: int = 2048) -> L2CachePlan:
    hidden = int(getattr(config, "hidden_size", getattr(config, "n_embd", 576)))
    heads = int(getattr(config, "num_attention_heads", getattr(config, "n_head", 9)))
    kv_heads = int(getattr(config, "num_key_value_heads", heads))
    layers = int(getattr(config, "num_hidden_layers", getattr(config, "n_layer", 30)))
    head_dim = int(getattr(config, "head_dim", hidden // max(1, heads)))
    dtype_bytes = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    return L2CachePlan(layers, hidden, heads, kv_heads, head_dim, dtype_bytes, l2_cache_mb, 0.55, max_batch, context_tokens)


def configure_torch_attention(enable_flash: bool = True, enable_mem_efficient: bool = True, enable_math: bool = True) -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.backends.cuda.enable_flash_sdp(bool(enable_flash))
        torch.backends.cuda.enable_mem_efficient_sdp(bool(enable_mem_efficient))
        torch.backends.cuda.enable_math_sdp(bool(enable_math))
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


def fused_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, dropout_p: float = 0.0, backend: str = "auto") -> torch.Tensor:
    """Fused causal attention with FlashAttention/SDPA fallback.

    Expected layout: [batch, heads, time, head_dim].  This uses FlashAttention-2
    when explicitly requested and installed, otherwise PyTorch SDPA.  A Triton
    kernel can be dropped in here later without touching the higher-level runtime.
    """
    if backend in ("auto", "flash_attn"):
        try:
            from flash_attn import flash_attn_func  # type: ignore
            # flash_attn_func expects [batch, seqlen, heads, dim]
            out = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p, causal=True)
            return out.transpose(1, 2).contiguous()
        except Exception:
            if backend == "flash_attn":
                raise
    configure_torch_attention(enable_flash=True, enable_mem_efficient=True, enable_math=True)
    return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)


def estimate_sidecar_lm_tps(
    params: int = 135_000_000,
    dtype_bytes: int = 2,
    memory_bandwidth_gbs: float = 1792.0,
    efficiency_low: float = 0.15,
    efficiency_high: float = 0.40,
) -> Dict[str, float]:
    weight_bytes = float(params * dtype_bytes)
    roofline = (memory_bandwidth_gbs * 1e9) / max(1.0, weight_bytes)
    return {
        "params_m": params / 1e6,
        "weight_mb_bf16": weight_bytes / 1e6,
        "roofline_tps_if_weight_read_once": roofline,
        "realistic_single_stream_low_tps": roofline * efficiency_low,
        "realistic_single_stream_high_tps": roofline * efficiency_high,
        "aggregate_continuous_batching_low_tps": roofline * 0.30,
        "aggregate_continuous_batching_high_tps": roofline * 0.75,
    }
