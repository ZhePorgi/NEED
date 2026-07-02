#!/usr/bin/env python3
"""Optional fused kernels for NEED.

The public API is conservative: every function has a PyTorch fallback.  Triton
kernels are used only when requested, tensors are CUDA tensors, Triton is
installed, and autograd is not required.  This keeps training numerically safe
while giving inference a path that reduces HBM roundtrips by keeping row/scan
state in SRAM/registers.
"""
from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False


def _needs_grad(*xs: torch.Tensor) -> bool:
    return torch.is_grad_enabled() and any(isinstance(x, torch.Tensor) and x.requires_grad for x in xs)


def _use_triton(*xs: torch.Tensor, backend: str, forward_only: bool = True) -> bool:
    if backend not in ("auto", "triton") or not HAS_TRITON:
        return False
    if not xs or not all(isinstance(x, torch.Tensor) and x.is_cuda for x in xs):
        return False
    # The kernels in this file are forward kernels.  They are intentionally not
    # used for differentiable training until a matching backward is supplied.
    if forward_only and _needs_grad(*xs):
        return False
    return True


if HAS_TRITON:
    @triton.jit
    def _rmsnorm_kernel(X, W, Y, N:tl.constexpr, EPS:tl.constexpr, BLOCK:tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)
        ss = tl.sum(x * x, axis=0) / N
        inv = tl.rsqrt(ss + EPS)
        w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(Y + row * N + offs, x * inv * w, mask=mask)

    @triton.jit
    def _swiglu_kernel(A, B, Y, TOTAL:tl.constexpr, BLOCK:tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < TOTAL
        a = tl.load(A + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-a))
        tl.store(Y + offs, a * sig * b, mask=mask)

    @triton.jit
    def _ssd_scan_kernel(U, BG, CG, DT, OG, DSKIP, Y,
                         B:tl.constexpr, T:tl.constexpr, D:tl.constexpr,
                         DT_MIN:tl.constexpr, DT_MAX:tl.constexpr, BLOCK:tl.constexpr):
        b = tl.program_id(0)
        block = tl.program_id(1)
        offs = block * BLOCK + tl.arange(0, BLOCK)
        mask = offs < D
        state = tl.zeros((BLOCK,), tl.float32)
        dskip = tl.load(DSKIP + offs, mask=mask, other=0.0).to(tl.float32)
        for i in range(0, T):
            base = (b * T + i) * D + offs
            u = tl.load(U + base, mask=mask, other=0.0).to(tl.float32)
            bg = tl.load(BG + base, mask=mask, other=0.0).to(tl.float32)
            cg = tl.load(CG + base, mask=mask, other=0.0).to(tl.float32)
            dt = tl.load(DT + base, mask=mask, other=0.0).to(tl.float32)
            og = tl.load(OG + base, mask=mask, other=0.0).to(tl.float32)
            dt_sig = 1.0 / (1.0 + tl.exp(-dt))
            dec = tl.exp(-(DT_MIN + (DT_MAX - DT_MIN) * dt_sig))
            b_sig = 1.0 / (1.0 + tl.exp(-bg))
            c_sig = 1.0 / (1.0 + tl.exp(-cg))
            o_sig = 1.0 / (1.0 + tl.exp(-og))
            state = dec * state + (1.0 - dec) * (b_sig * u)
            y = (c_sig * state + dskip * u) * o_sig
            tl.store(Y + base, y, mask=mask)

    @triton.jit
    def _residual_scale_kernel(X, Y, OUT, SCALE:tl.constexpr, TOTAL:tl.constexpr, BLOCK:tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < TOTAL
        x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(Y + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(OUT + offs, x + SCALE * y, mask=mask)




_ASSOCIATIVE_SCAN = None
_ASSOCIATIVE_SCAN_MISSING = False


def _get_associative_scan():
    global _ASSOCIATIVE_SCAN, _ASSOCIATIVE_SCAN_MISSING
    if _ASSOCIATIVE_SCAN is not None:
        return _ASSOCIATIVE_SCAN
    if _ASSOCIATIVE_SCAN_MISSING:
        return None
    try:
        from torch._higher_order_ops.associative_scan import associative_scan  # type: ignore
        _ASSOCIATIVE_SCAN = associative_scan
        return _ASSOCIATIVE_SCAN
    except Exception:
        _ASSOCIATIVE_SCAN_MISSING = True
        return None


def affine_recurrence_scan(decay: torch.Tensor, update: torch.Tensor, *, dim: int = 1, backend: str = "auto") -> torch.Tensor:
    """Parallel prefix scan for ``state_t = decay_t * state_{t-1} + update_t``.

    This preserves the reference recurrence while avoiding a Python loop over the
    sequence length when PyTorch's associative_scan higher-order op is available.
    The op is differentiable and compiles well for fixed-shape training.  It is a
    large MFU win for NEED's retention and hierarchical-memory paths because those
    paths otherwise execute one small kernel per token.
    """
    if update.numel() == 0:
        return update
    dim = dim if dim >= 0 else update.ndim + dim
    if dim < 0 or dim >= update.ndim:
        raise IndexError(f"scan dim {dim} out of range for rank {update.ndim}")
    if decay.shape != update.shape:
        decay = torch.broadcast_to(decay, update.shape)
    scan = _get_associative_scan()
    if scan is not None and backend != "torch_loop" and update.is_cuda:
        try:
            a = decay.movedim(dim, 0).contiguous()
            b = update.movedim(dim, 0).contiguous()

            def combine(left, right):
                a1, b1 = left
                a2, b2 = right
                return a2 * a1, b2 + a2 * b1

            _, out = scan(combine, (a, b), dim=0)
            return out.movedim(0, dim)
        except Exception:
            # Fall through to the reference loop.  This keeps the feature safe on
            # older/unsupported PyTorch builds and for graph-capture edge cases.
            pass
    state = torch.zeros_like(update.select(dim, 0))
    outs = []
    for i in range(update.size(dim)):
        state = decay.select(dim, i) * state + update.select(dim, i)
        outs.append(state)
    return torch.stack(outs, dim=dim)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, backend: str = "auto") -> torch.Tensor:
    if _use_triton(x, weight, backend=backend, forward_only=True) and x.is_contiguous() and weight.is_contiguous() and x.size(-1) <= 8192:
        n = x.size(-1)
        rows = x.numel() // n
        y = torch.empty_like(x)
        block = triton.next_power_of_2(n)
        _rmsnorm_kernel[(rows,)](x, weight, y, n, eps, BLOCK=block)
        return y
    return x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps).to(x.dtype) * weight


def swiglu(a: torch.Tensor, b: torch.Tensor, backend: str = "auto") -> torch.Tensor:
    if _use_triton(a, b, backend=backend, forward_only=True) and a.is_contiguous() and b.is_contiguous():
        y = torch.empty_like(a)
        block = 256
        grid = (triton.cdiv(a.numel(), block),)
        _swiglu_kernel[grid](a, b, y, a.numel(), BLOCK=block)
        return y
    return F.silu(a) * b


def residual_scale_add(x: torch.Tensor, y: torch.Tensor, scale: float, backend: str = "auto") -> torch.Tensor:
    """Fused inference residual add: x + scale * y."""
    if _use_triton(x, y, backend=backend, forward_only=True) and x.is_contiguous() and y.is_contiguous() and x.shape == y.shape:
        out = torch.empty_like(x)
        block = 256
        grid = (triton.cdiv(x.numel(), block),)
        _residual_scale_kernel[grid](x, y, out, float(scale), x.numel(), BLOCK=block)
        return out
    return x + float(scale) * y


def ssd_scan(
    u: torch.Tensor,
    b_gate: torch.Tensor,
    c_gate: torch.Tensor,
    dt_raw: torch.Tensor,
    out_gate: torch.Tensor,
    d_skip: torch.Tensor,
    dt_min: float,
    dt_max: float,
    backend: str = "auto",
) -> torch.Tensor:
    """Fused SSD-style diagonal recurrent scan.

    Shapes: all sequence tensors are [B,T,D], d_skip is [D].  Inference Triton
    path keeps each D block's recurrent state in registers over the T loop.  The
    PyTorch path is differentiable and used for training.
    """
    b, t, d = u.shape
    if (
        _use_triton(u, b_gate, c_gate, dt_raw, out_gate, d_skip, backend=backend, forward_only=True)
        and all(z.is_contiguous() for z in (u, b_gate, c_gate, dt_raw, out_gate, d_skip))
        and d <= 8192
    ):
        y = torch.empty_like(u)
        block = min(1024, triton.next_power_of_2(d))
        grid = (b, triton.cdiv(d, block))
        _ssd_scan_kernel[grid](u, b_gate, c_gate, dt_raw, out_gate, d_skip, y, b, t, d, float(dt_min), float(dt_max), BLOCK=block)
        return y
    dt = torch.sigmoid(dt_raw.float())
    dt = float(dt_min) + (float(dt_max) - float(dt_min)) * dt
    decay = torch.exp(-dt).to(u.dtype)
    bg = torch.sigmoid(b_gate).to(u.dtype)
    cg = torch.sigmoid(c_gate).to(u.dtype)
    og = torch.sigmoid(out_gate).to(u.dtype)
    state = torch.zeros(b, d, device=u.device, dtype=u.dtype)
    outs = []
    d_skip_t = d_skip.to(u.dtype)
    for i in range(t):
        state = decay[:, i] * state + (1.0 - decay[:, i]) * (bg[:, i] * u[:, i])
        outs.append((cg[:, i] * state + d_skip_t * u[:, i]) * og[:, i])
    return torch.stack(outs, dim=1)


def recurrent_retention_scan(q: torch.Tensor, kv: torch.Tensor, decay: torch.Tensor, backend: str = "auto") -> torch.Tensor:
    """Reference contract for a fused multi-head retention scan."""
    b, h, t, d = q.shape
    state = torch.zeros(b, h, d, device=q.device, dtype=q.dtype)
    outs = []
    for i in range(t):
        state = state * decay[:, :, i, :] + kv[:, :, i, :]
        outs.append(q[:, :, i, :] * state)
    return torch.stack(outs, dim=2)


def sparse_moe_combine(expert_outputs: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor, backend: str = "auto") -> torch.Tensor:
    """Reference sparse combine contract."""
    e, b, t, d = expert_outputs.shape
    out = expert_outputs.new_zeros(b, t, d)
    for expert_id in range(e):
        mask = topk_idx == expert_id
        if bool(mask.any()):
            w = torch.where(mask, topk_weight, torch.zeros_like(topk_weight)).sum(dim=-1, keepdim=True)
            out = out + expert_outputs[expert_id] * w
    return out


def sample_from_logits(logits: torch.Tensor, temperature: float = 1.0, backend: str = "auto") -> torch.Tensor:
    logits = logits.float() / max(float(temperature), 1e-8)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)
