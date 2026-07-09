#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import asdict
import torch
torch.set_num_threads(1)

from need_core import (
    NeedConfig, NeedModel, Special,
    SparseMoE, HierarchicalMemory, LatentPlanner, LatentSlotAttention,
    MultiScaleCausalConv, SelectiveRetention, StructuredDualRetention,
)
import need_kernels


def tiny_config(**overrides):
    data = dict(
        d_model=16,
        n_layers=1,
        n_heads=4,
        block_size=8,
        vocab_size=Special.text_vocab + 8,
        image_codebook_size=8,
        d_ff=32,
        latent_slots=2,
        memory_slots=3,
        memory_rank=8,
        n_experts=3,
        moe_top_k=2,
        n_predict_heads=2,
        planner_horizons=2,
        pathway_memory_slots=3,
        pathway_memory_top_k=2,
        exact_recall=True,
        exact_recall_max_candidates=6,
        exact_recall_top_k=2,
        exact_recall_max_tokens=16,
        object_program_slots=1,
        energy_rank=8,
        output_modes=5,
        energy_routes=2,
        dropout=0.0,
        moe_dropout=0.0,
        token_dropout=0.0,
        collect_aux_metrics=False,
        streaming_generation=True,
        fused_ssd_scan=False,
        parallel_scan=True,
        kernel_backend='auto',
    )
    data.update(overrides)
    return NeedConfig(**data)


def assert_finite(name, x):
    if torch.is_tensor(x):
        if not torch.isfinite(x.float()).all():
            raise AssertionError(f"{name} non-finite: {x}")


def assert_allclose(name, a, b, atol=1e-5, rtol=1e-5):
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        diff = (a.float() - b.float()).abs().max().item()
        raise AssertionError(f"{name} mismatch max_abs={diff}")


def sequential_module_equivalence():
    torch.manual_seed(1)
    x = torch.randn(2, 7, 16)
    for impl, cls in [('selective', SelectiveRetention), ('ssd', StructuredDualRetention)]:
        cfg = tiny_config(retention_impl=impl)
        m = cls(cfg).eval()
        with torch.no_grad():
            full = m(x)
            state = m.init_stream_state(x.size(0), x.device, x.dtype)
            steps = [m.stream_step(x[:, i:i+1], state) for i in range(x.size(1))]
            stream = torch.cat(steps, dim=1)
        assert_allclose(f'{impl} retention stream/full', full, stream, atol=5e-5, rtol=5e-5)

    cfg = tiny_config()
    conv = MultiScaleCausalConv(cfg.d_model, cfg.conv_kernel, cfg.n_conv_scales, cfg.dropout, cfg.conv_active_scales).eval()
    with torch.no_grad():
        full, aux = conv(x)
        state = conv.init_stream_state(x.size(0), x.device, x.dtype)
        ys=[]
        for i in range(x.size(1)):
            y, _ = conv.stream_step(x[:, i:i+1], state)
            ys.append(y)
        stream = torch.cat(ys, dim=1)
    assert_allclose('causal conv stream/full', full, stream, atol=5e-5, rtol=5e-5)

    mem = HierarchicalMemory(cfg).eval()
    cond = torch.randn_like(x)
    innov = torch.randn_like(x)
    write_mask = torch.tensor([[[1.0],[0.0],[1.0],[1.0],[0.0],[1.0],[1.0]], [[1.0],[1.0],[0.0],[1.0],[1.0],[0.0],[1.0]]])
    with torch.no_grad():
        full, aux = mem(x, condition=cond, innovation=innov, write_mask=write_mask)
        state = mem.init_stream_state(x.size(0), x.device, x.dtype)
        ys=[]
        for i in range(x.size(1)):
            y, _ = mem.stream_step(x[:, i:i+1], state, condition=cond[:, i:i+1], innovation=innov[:, i:i+1], write_mask=write_mask[:, i:i+1])
            ys.append(y)
        stream = torch.cat(ys, dim=1)
    assert_allclose('hierarchical memory stream/full', full, stream, atol=5e-5, rtol=5e-5)
    for k, v in aux.items():
        assert_finite('memory aux '+k, v)


def moe_static_sparse_equivalence():
    torch.manual_seed(2)
    x = torch.randn(3, 5, 16)
    cfg_sparse = tiny_config(moe_static_dispatch=False, n_experts=4, moe_top_k=2)
    cfg_static = NeedConfig.from_dict(asdict(cfg_sparse))
    cfg_static.moe_static_dispatch = True
    sparse = SparseMoE(cfg_sparse).eval()
    static = SparseMoE(cfg_static).eval()
    static.load_state_dict(sparse.state_dict())
    with torch.no_grad():
        ys, asparse = sparse(x)
        yt, astatic = static(x)
    assert_allclose('MoE sparse/static', ys, yt, atol=5e-6, rtol=5e-6)
    for aux in (asparse, astatic):
        for k, v in aux.items():
            assert_finite('moe aux '+k, v)


def latent_planner_slot_invariants():
    torch.manual_seed(3)
    h = torch.randn(2, 6, 16, requires_grad=True)
    for mode in ['pooled', 'attention']:
        cfg = tiny_config(slot_attention_mode=mode, latent_slots=3)
        slot = LatentSlotAttention(cfg)
        delta, slots, aux = slot(h)
        assert delta is not None and delta.shape == (2, 6, 16)
        if mode == 'pooled':
            assert slots.shape == (2, 6, 3, 16)
        else:
            assert slots.shape == (2, 3, 16)
        for k, v in aux.items():
            assert_finite('slot aux '+k, v)
        (slots.float().mean() + (delta.float().mean() if delta is not None else 0)).backward(retain_graph=True)

    cfg = tiny_config(planner_horizons=3, planner_transition_depth=2)
    planner = LatentPlanner(cfg)
    outs = planner(h.detach())
    assert len(outs) == 3
    for i, o in enumerate(outs):
        assert o.shape == (2, 6, 16)
        assert_finite(f'planner out {i}', o)
    state = planner.start_state(h.detach()[:, -1])
    tok_fb = torch.randn(2, 16)
    descent = torch.randn(2, 16)
    stepped = planner.compound_step(state, tok_fb, 1, descent, confidence=torch.tensor([0.5, 1.0]))
    assert stepped.shape == (2, 16)
    assert_finite('planner compound_step', stepped)


def full_model_streaming_invariants():
    torch.manual_seed(4)
    configs = [
        tiny_config(retention_impl='ssd', n_experts=1, moe_top_k=1, exact_recall=False, latent_slots=1, planner_horizons=1),
        tiny_config(retention_impl='selective', n_experts=3, moe_top_k=2, exact_recall=True, latent_slots=2, planner_horizons=2),
        tiny_config(retention_impl='ssd', n_experts=3, moe_top_k=1, moe_static_dispatch=True, exact_recall=True, latent_slots=3, planner_horizons=0),
    ]
    for ci, cfg in enumerate(configs):
        m = NeedModel(cfg).eval()
        ids = torch.tensor([[Special.bos, 16, 17, 18, 19, 20]], dtype=torch.long)
        with torch.no_grad():
            full_logits, _, _ = m(ids)
            cache = m._stream_new_cache(ids.size(0), ids.device, m.token_emb.weight.dtype)
            stream_logits, aux = m._stream_prefill(ids, cache)
        assert_allclose(f'model {ci} prefill stream/full', full_logits[:, -1], stream_logits, atol=1e-3, rtol=1e-3)
        for k, v in aux.items():
            if k.startswith('_'):
                continue
            assert_finite(f'stream aux {ci} {k}', v)
        # Greedy generation should match full-context generator while total length stays inside block_size.
        with torch.no_grad():
            out_stream = m.generate_text(ids[:, :4], max_new_tokens=2, temperature=0.0, use_streaming_cache=True)
            out_full = m.generate_text(ids[:, :4], max_new_tokens=2, temperature=0.0, use_streaming_cache=False)
        if not torch.equal(out_stream, out_full):
            raise AssertionError(f'model {ci} greedy stream/full generated IDs differ: {out_stream.tolist()} vs {out_full.tolist()}')


def kernel_fallback_and_scan_checks():
    torch.manual_seed(5)
    x = torch.randn(2, 4, 16, requires_grad=True)
    w = torch.randn(16, requires_grad=True)
    y_triton_request = need_kernels.rms_norm(x, w, backend='triton')
    y_torch = need_kernels.rms_norm(x, w, backend='torch')
    assert_allclose('rms_norm triton-request CPU fallback', y_triton_request, y_torch)
    a = torch.randn(2, 4, 16, requires_grad=True)
    b = torch.randn(2, 4, 16, requires_grad=True)
    assert_allclose('swiglu triton-request CPU fallback', need_kernels.swiglu(a, b, backend='triton'), need_kernels.swiglu(a, b, backend='torch'))
    decay = torch.sigmoid(torch.randn(2, 7, 16)) * 0.95
    upd = torch.randn(2, 7, 16, requires_grad=True)
    got = need_kernels.affine_recurrence_scan(decay, upd, dim=1, backend='auto')
    state = torch.zeros_like(upd[:, 0])
    outs=[]
    for t in range(upd.size(1)):
        state = decay[:, t] * state + upd[:, t]
        outs.append(state)
    ref = torch.stack(outs, dim=1)
    assert_allclose('affine scan vs loop', got, ref, atol=2e-5, rtol=2e-5)
    got.sum().backward()
    assert True


def logic_invariant_checks():
    # Check finite gradients for zero trajectory deltas.
    from need_core import geodesic_path_loss, path_straightness_loss, contractive_path_loss
    h0 = torch.randn(2, 4, 16, requires_grad=True)
    h1 = h0 + 0.01 * torch.randn_like(h0)
    h2 = h1
    loss = (
        geodesic_path_loss([h0, h1, h2], 0.10)
        + path_straightness_loss([h0, h1, h2])
        + contractive_path_loss([h0, h1, h2], 0.92)
    )
    assert_finite('zero-delta path loss', loss)
    loss.backward()
    assert_finite('zero-delta path grad', h0.grad)

    # Bidirectional image scans are intentionally non-causal.
    cfg = tiny_config(block_size=8, image_2d_bidirectional=True, image_grid=2, image_max_grid=4, image_max_tokens=8)
    model = NeedModel(cfg).train()
    ids = torch.tensor([[Special.bos, 16, Special.img_bos, cfg.image_token_offset + 1, cfg.image_token_offset + 2, Special.img_eos, 17, 18]], dtype=torch.long)
    tgt = torch.full_like(ids, Special.pad)
    tgt[:, :-1] = ids[:, 1:]
    try:
        model(ids, tgt)
    except RuntimeError as exc:
        if 'image_2d_bidirectional=True is non-causal' not in str(exc):
            raise
    else:
        raise AssertionError('strict-core training accepted bidirectional image scan')


    # State stabilization and adaptive equilibrium also use norm-like quantities
    # Zero movement should preserve finite gradients.
    from need_core import StateSpaceDriftStabilizer, AdaptiveEquilibrium
    cfg2 = tiny_config(state_stabilization=True, state_drift_chunk=4, energy_steps=1, energy_min_steps=0)
    stab = StateSpaceDriftStabilizer(cfg2)
    h = torch.zeros(1, 8, cfg2.d_model, requires_grad=True)
    base = torch.zeros_like(h)
    y, aux = stab(h, base)
    zloss = aux["state_chunk_drift"] + aux["state_norm_error"] + aux["state_anchor_error"] + y.float().mean() * 0.0
    assert_finite('state zero loss', zloss)
    zloss.backward()
    assert_finite('state zero grad', h.grad)

    eq = AdaptiveEquilibrium(cfg2)
    x = torch.zeros(1, 3, cfg2.d_model, requires_grad=True)
    ctx = torch.zeros_like(x)
    z, resid, energy, effort = eq(x, ctx)
    eloss = resid + energy + effort + z.float().mean() * 0.0
    assert_finite('equilibrium zero loss', eloss)
    eloss.backward()
    assert_finite('equilibrium zero grad', x.grad)

    try:
        model = NeedModel(tiny_config(block_size=8)).eval()
        model(ids, torch.ones(1, 7, dtype=torch.long))
    except ValueError as exc:
        if 'targets shape' not in str(exc):
            raise
    else:
        raise AssertionError('target shape guard did not fire')


if __name__ == '__main__':
    checks = [
        sequential_module_equivalence,
        moe_static_sparse_equivalence,
        latent_planner_slot_invariants,
        full_model_streaming_invariants,
        kernel_fallback_and_scan_checks,
        logic_invariant_checks,
    ]
    for fn in checks:
        fn()
        print(fn.__name__ + ': ok')
