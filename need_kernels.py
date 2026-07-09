#!/usr/bin/env python3
"""Optional fused kernels for NEED.

The public API is conservative: every function has a PyTorch fallback.  Triton
kernels are used only when requested, tensors are CUDA tensors, and Triton is
installed. Pointwise kernels have autograd wrappers for training; recurrent scan
training uses differentiable associative/chunked PyTorch scans instead of one
Python dispatch per token.  This keeps training numerically safe while giving
inference a path that reduces HBM roundtrips by keeping row/scan state in
SRAM/registers.
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
    class _RMSNormTritonAutograd(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:  # type: ignore[override]
            n = x.size(-1)
            rows = x.numel() // n
            y = torch.empty_like(x)
            block = triton.next_power_of_2(n)
            _rmsnorm_kernel[(rows,)](x, weight, y, n, eps, BLOCK=block)
            ctx.save_for_backward(x, weight)
            ctx.eps = float(eps)
            return y

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
            x, weight = ctx.saved_tensors
            eps = float(ctx.eps)
            xf = x.float()
            wf = weight.float()
            go = grad_out.float()
            inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
            grad_x = grad_w = None
            if ctx.needs_input_grad[0]:
                g = go * wf
                mean_gx = (g * xf).mean(dim=-1, keepdim=True)
                grad_x = (inv * (g - xf * inv.pow(2) * mean_gx)).to(dtype=x.dtype)
            if ctx.needs_input_grad[1]:
                reduce_dims = tuple(range(go.ndim - 1))
                grad_w = (go * xf * inv).sum(dim=reduce_dims).to(dtype=weight.dtype)
            return grad_x, grad_w, None

    class _SwiGLUTritonAutograd(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            y = torch.empty_like(a)
            block = 256
            grid = (triton.cdiv(a.numel(), block),)
            _swiglu_kernel[grid](a, b, y, a.numel(), BLOCK=block)
            ctx.save_for_backward(a, b)
            return y

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
            a, b = ctx.saved_tensors
            af = a.float()
            bf = b.float()
            go = grad_out.float()
            sig = torch.sigmoid(af)
            silu = af * sig
            dsilu = sig * (1.0 + af * (1.0 - sig))
            grad_a = (go * bf * dsilu).to(dtype=a.dtype) if ctx.needs_input_grad[0] else None
            grad_b = (go * silu).to(dtype=b.dtype) if ctx.needs_input_grad[1] else None
            return grad_a, grad_b

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
    def _swiglu_kernel(A, B, Y, TOTAL, BLOCK:tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < TOTAL
        a = tl.load(A + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-a))
        tl.store(Y + offs, a * sig * b, mask=mask)

    @triton.jit
    def _ssd_scan_kernel(U, BG, CG, DT, OG, DSKIP, Y,
                         T, D, DT_MIN, DT_MAX, BLOCK:tl.constexpr):
        b = tl.program_id(0)
        block = tl.program_id(1)
        offs = block * BLOCK + tl.arange(0, BLOCK)
        mask = offs < D
        state = tl.zeros((BLOCK,), tl.float32)
        dskip = tl.load(DSKIP + offs, mask=mask, other=0.0).to(tl.float32)
        i = 0
        while i < T:
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
            i += 1

    @triton.jit
    def _residual_scale_kernel(X, Y, OUT, SCALE:tl.constexpr, TOTAL, BLOCK:tl.constexpr):
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


def _chunked_affine_recurrence_scan(decay: torch.Tensor, update: torch.Tensor, *, dim: int, chunk_size: int = 64) -> torch.Tensor:
    """Differentiable static-shape fallback for ``state_t = decay_t * state_{t-1} + update_t``.

    The previous closed-form implementation divided updates by a cumulative
    product and multiplied by it again at the end. For small decays the product
    can underflow long before the mathematically equivalent recurrence would,
    silently turning valid updates into zeros. This implementation composes affine
    transforms with a divide-free prefix scan inside each chunk, then carries the
    last state between chunks. It preserves the recurrence for arbitrary finite
    decay values without relying on clamped cumulative products.
    """
    if update.numel() == 0:
        return update
    dim = dim if dim >= 0 else update.ndim + dim
    if dim < 0 or dim >= update.ndim:
        raise IndexError(f"scan dim {dim} out of range for rank {update.ndim}")
    if decay.shape != update.shape:
        decay = torch.broadcast_to(decay, update.shape)
    x = update.movedim(dim, 1)
    a = decay.movedim(dim, 1).to(torch.float32)
    xf = x.to(torch.float32)
    t = xf.size(1)
    chunk = max(1, min(int(chunk_size), t))
    state = torch.zeros_like(xf[:, 0])
    outs = []
    for start in range(0, t, chunk):
        end = min(t, start + chunk)
        ac = a[:, start:end]
        bc = xf[:, start:end]
        # Prefix-composition of affine maps (a, b): s -> a * s + b.
        # Combining previous p before current c gives (a_c * a_p, b_c + a_c * b_p).
        a_pref = ac
        b_pref = bc
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
    return y.movedim(1, dim)


def affine_recurrence_scan(decay: torch.Tensor, update: torch.Tensor, *, dim: int = 1, backend: str = "auto") -> torch.Tensor:
    """Parallel prefix scan for ``state_t = decay_t * state_{t-1} + update_t``.

    This preserves the reference recurrence while avoiding a Python loop over the
    sequence length. On CUDA with a supported PyTorch build it uses differentiable
    ``associative_scan``. Otherwise it uses a static-shape, divide-free chunked
    affine-prefix scan that dispatches O(log(chunk) * T / chunk) PyTorch ops
    instead of O(T) tiny recurrent ops.
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
            # Run the composed affine scan in fp32 even when the model state is
            # bf16/fp16; otherwise long low-decay spans can underflow in the scan
            # operator before the final output cast.
            a = decay.movedim(dim, 0).to(torch.float32).contiguous()
            b = update.movedim(dim, 0).to(torch.float32).contiguous()

            def combine(left, right):
                a1, b1 = left
                a2, b2 = right
                return a2 * a1, b2 + a2 * b1

            _, out = scan(combine, (a, b), dim=0)
            return out.movedim(0, dim).to(dtype=update.dtype)
        except Exception:
            pass
    chunk = 64 if backend != "torch_loop" else 1
    return _chunked_affine_recurrence_scan(decay, update, dim=dim, chunk_size=chunk)

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, backend: str = "auto") -> torch.Tensor:
    if _use_triton(x, weight, backend=backend, forward_only=False) and x.is_contiguous() and weight.is_contiguous() and x.size(-1) <= 8192:
        if _needs_grad(x, weight):
            return _RMSNormTritonAutograd.apply(x, weight, float(eps))
        n = x.size(-1)
        rows = x.numel() // n
        y = torch.empty_like(x)
        block = triton.next_power_of_2(n)
        _rmsnorm_kernel[(rows,)](x, weight, y, n, eps, BLOCK=block)
        return y
    return x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps).to(x.dtype) * weight


def swiglu(a: torch.Tensor, b: torch.Tensor, backend: str = "auto") -> torch.Tensor:
    if _use_triton(a, b, backend=backend, forward_only=False) and a.is_contiguous() and b.is_contiguous():
        if _needs_grad(a, b):
            return _SwiGLUTritonAutograd.apply(a, b)
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
        _ssd_scan_kernel[grid](u, b_gate, c_gate, dt_raw, out_gate, d_skip, y, t, d, float(dt_min), float(dt_max), BLOCK=block)
        return y
    dt = torch.sigmoid(dt_raw.float())
    dt = float(dt_min) + (float(dt_max) - float(dt_min)) * dt
    decay = torch.exp(-dt).to(u.dtype)
    bg = torch.sigmoid(b_gate).to(u.dtype)
    cg = torch.sigmoid(c_gate).to(u.dtype)
    og = torch.sigmoid(out_gate).to(u.dtype)
    update = (1.0 - decay) * (bg * u)
    state_seq = affine_recurrence_scan(decay, update, dim=1, backend=backend)
    d_skip_t = d_skip.to(u.dtype).view(1, 1, -1)
    return (cg * state_seq + d_skip_t * u) * og


def recurrent_retention_scan(q: torch.Tensor, kv: torch.Tensor, decay: torch.Tensor, backend: str = "auto") -> torch.Tensor:
    """Reference contract for a fused multi-head retention scan."""
    state_seq = affine_recurrence_scan(decay, kv, dim=2, backend=backend)
    return q * state_seq


def sparse_moe_combine(expert_outputs: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor, backend: str = "auto") -> torch.Tensor:
    """Reference sparse combine contract."""
    e, b, t, d = expert_outputs.shape
    out = expert_outputs.new_zeros(b, t, d)
    for expert_id in range(e):
        mask = topk_idx == expert_id
        w = torch.where(mask, topk_weight, torch.zeros_like(topk_weight)).sum(dim=-1, keepdim=True)
        out = out + expert_outputs[expert_id] * w.to(dtype=expert_outputs.dtype)
    return out


def sample_from_logits(logits: torch.Tensor, temperature: float = 1.0, backend: str = "auto") -> torch.Tensor:
    logits = logits.float() / max(float(temperature), 1e-8)
    finite = torch.isfinite(logits)
    has = finite.any(dim=-1, keepdim=True)
    cleaned = torch.where(finite, logits, torch.full_like(logits, -1.0e9))
    probs = torch.softmax(cleaned, dim=-1)
    bad = (~has) | (~torch.isfinite(probs).all(dim=-1, keepdim=True)) | (probs.sum(dim=-1, keepdim=True) <= 0)
    if bool(bad.any().detach().cpu()):
        uniform = torch.full_like(probs, 1.0 / max(1, probs.size(-1)))
        probs = torch.where(bad, uniform, probs)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.multinomial(probs, 1)
