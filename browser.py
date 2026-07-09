#!/usr/bin/env python3
"""Lil web interface"""
from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import re
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import torch
from need_core import ByteTokenizer, load_tokenizer_for_dir, load_model, make_image_tokenizer, resolve_device, LatentMemoryStore
from sidecar_lm_runtime import need_optimization_mode, sidecar_optimization_mode
from generate import (
    _speculative_final_decode, _extract_final_answer_text, _compute_latent_convergence_metrics,
    _run_latent_tools, _strip_hidden_runtime_artifacts, _apply_dvsd_runtime_overrides, _sidecar_should_run,
)
from need_sidecar import make_single_sidecar_runtime
from need_latent_tools import LatentToolRuntime

try:
    from need_image import load_visual_tokenizer
except Exception:  # pragma: no cover
    load_visual_tokenizer = None  # type: ignore[assignment]

def _float(x, default=0.0):
    try:
        return float(x.detach().cpu()) if torch.is_tensor(x) else float(x)
    except Exception:
        return default


def _apply_runtime_profile(args: argparse.Namespace) -> argparse.Namespace:
    if not getattr(args, "runtime_profile", ""):
        return args
    path = Path(args.runtime_profile)
    if not path.exists():
        raise FileNotFoundError(f"runtime profile not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    runtime = raw.get("runtime", raw) if isinstance(raw, dict) else {}
    if not isinstance(runtime, dict):
        return args
    for key in [
        "decode_mode", "nonseq_decode", "nonseq_dynamic", "nonseq_min_heads", "nonseq_max_heads",
        "nonseq_refine_steps", "nonseq_refine_causal_blend", "nonseq_refine_confidence_floor",
        "nonseq_refine_temperature_decay", "nonseq_refine_lock_schedule", "nonseq_refine_resample_locked",
        "dvsd_router_enabled", "dvsd_router_inference_mix", "dvsd_router_min_confidence",
        "dvsd_planner_compound_enabled", "dvsd_planner_compound_mix",
        "dvsd_planner_compound_step_size", "dvsd_planner_compound_token_scale",
        "dvsd_planner_compound_descent_scale", "dvsd_planner_compound_top_k",
        "planner_block_space_enabled", "planner_block_space_mix", "planner_block_space_iters",
        "sidecar_call_policy", "sidecar_gate_metric", "sidecar_gate_threshold",
        "sidecar_type", "need_sidecar_checkpoint", "need_sidecar_projection_path", "need_sidecar_projection_weight",
        "need_sidecar_decode_mode", "need_sidecar_prefer_best", "need_sidecar_max_context_tokens",
        "sidecar_model", "sidecar_adapter_path", "sidecar_latent_alignment_path",
        "sidecar_latent_alignment_weight", "latent_search_depth", "latent_search_branches",
        "aux_score_risk_threshold", "aux_score_contradiction_threshold",
        "latent_tools", "latent_tool_calculator", "latent_tool_python", "latent_tool_sidecar_planning", "latent_tool_router",
        "latent_tool_max_calls", "latent_tool_timeout_s", "latent_tool_max_code_chars", "latent_tool_max_output_chars",
    ]:
        if key in runtime and hasattr(args, key):
            cur = getattr(args, key)
            if cur in ("", None) or key.startswith(("sidecar_", "latent_", "aux_score_")):
                setattr(args, key, runtime[key])
    if "use_sidecar_latent_alignment" in runtime:
        args.use_sidecar_latent_alignment = bool(runtime["use_sidecar_latent_alignment"])
    if "use_need_sidecar_latents" in runtime:
        args.use_need_sidecar_latents = bool(runtime["use_need_sidecar_latents"])
    return args


def _candidate_checkpoint_dirs(root: Path) -> list[Path]:
    names = {"model.pt", "best.pt", "model.safetensors", "best.safetensors", "config.json"}
    candidates: list[Path] = []
    search_roots = [root, root / "checkpoints", root / "runs", root / "outputs"]
    for base in search_roots:
        if not base.exists():
            continue
        for path in [base] + [p for p in base.rglob("*") if p.is_dir()]:
            try:
                if any((path / n).exists() for n in names):
                    candidates.append(path)
            except OSError:
                continue
    dedup = {str(p.resolve()): p for p in candidates}
    return list(dedup.values())


def _resolve_checkpoint_arg(checkpoint: str) -> str:
    raw = str(checkpoint or "").strip()
    if raw:
        return raw
    candidates = _candidate_checkpoint_dirs(Path.cwd())
    if not candidates:
        raise FileNotFoundError("No --checkpoint was provided and no checkpoint-like directory was found under ., checkpoints, runs, or outputs.")
    candidates.sort(key=lambda p: max((x.stat().st_mtime for x in p.iterdir() if x.is_file()), default=p.stat().st_mtime), reverse=True)
    return str(candidates[0])

def main(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(description="NEED browser interface")
    p.add_argument("--checkpoint", default="", help="NEED checkpoint directory or file. If omitted, the newest checkpoint-like directory is used.")
    p.add_argument("--runtime_profile", default="", help="Optional JSON produced by need_low_data_adapters.py or the full pipeline.")
    p.add_argument("--system_prompt", default="You are a helpful AI assistant.", help="Short system prompt shown in the console")
    p.add_argument("--prefer_best", action="store_true")
    p.add_argument("--compare_checkpoint", default="", help="Optional second NEED checkpoint for the comparison tab")
    p.add_argument("--compare_prefer_best", action="store_true")
    p.add_argument("--visual_tokenizer", default="")
    p.add_argument("--device", default="auto")
    p.add_argument("--kernel_backend", default="auto")
    p.add_argument("--decode_mode", choices=["auto", "ar", "nonseq"], default="ar", help="Default ar uses the stateful streaming cache when supported; nonseq/DVSD is opt-in")
    p.add_argument("--nonseq_decode", action=argparse.BooleanOptionalAction, default=None, help="Compatibility switch: force DVSD/nonseq on or off")
    p.add_argument("--nonseq_dynamic", action=argparse.BooleanOptionalAction, default=True, help="Shrink DVSD active slots toward 1-2 tokens on difficult spans")
    p.add_argument("--nonseq_min_heads", type=int, default=1, help="Minimum active DVSD slots")
    p.add_argument("--nonseq_max_heads", type=int, default=0, help="Maximum active DVSD slots; 0 uses checkpoint/config default")
    p.add_argument("--nonseq_refine_steps", type=int, default=3, help="Virtual-slot refinement passes before direct commit")
    p.add_argument("--nonseq_refine_causal_blend", type=float, default=0.55, help="Blend MTP slot logits with provisional causal context during refinement")
    p.add_argument("--nonseq_refine_confidence_floor", type=float, default=0.0, help="Optional confidence floor for replacing unlocked slots")
    p.add_argument("--nonseq_refine_temperature_decay", type=float, default=0.82, help="Temperature multiplier per refinement pass")
    p.add_argument("--nonseq_refine_lock_schedule", choices=["cosine", "linear", "quadratic"], default="cosine", help="Confidence schedule for locking virtual slots non-left-to-right")
    p.add_argument("--nonseq_refine_resample_locked", action=argparse.BooleanOptionalAction, default=False, help="Allow locked virtual slots to be resampled on later passes")
    p.add_argument("--dvsd_router_enabled", action=argparse.BooleanOptionalAction, default=None, help="Use learned DVSD slot router when available; default keeps checkpoint setting")
    p.add_argument("--dvsd_router_inference_mix", type=float, default=None, help="Blend learned router vs heuristic DVSD slot budget; default keeps checkpoint setting")
    p.add_argument("--dvsd_router_min_confidence", type=float, default=None, help="Minimum learned-router confidence before it steers slot count; default keeps checkpoint setting")
    p.add_argument("--dvsd_planner_compound_enabled", action=argparse.BooleanOptionalAction, default=None, help="Feed provisional DVSD tokens through latent planner compounding")
    p.add_argument("--dvsd_planner_compound_mix", type=float, default=None, help="Blend compounded-planner logits into later DVSD slots")
    p.add_argument("--dvsd_planner_compound_step_size", type=float, default=None, help="Latent transition step size for DVSD compounding")
    p.add_argument("--dvsd_planner_compound_token_scale", type=float, default=None, help="Token-feedback residual strength for DVSD compounding")
    p.add_argument("--dvsd_planner_compound_descent_scale", type=float, default=None, help="Logit-descent residual strength for DVSD compounding")
    p.add_argument("--dvsd_planner_compound_top_k", type=int, default=None, help="Top-k logits used to approximate DVSD descent")
    p.add_argument("--planner_block_space_enabled", action=argparse.BooleanOptionalAction, default=None, help="Use full-block latent planning for DVSD/planner logits")
    p.add_argument("--planner_block_space_mix", type=float, default=None, help="Blend weight for full-block planner states")
    p.add_argument("--planner_block_space_iters", type=int, default=None, help="Slot-attention iterations for full-block planner")
    p.add_argument("--sidecar_type", choices=["auto", "none", "external_lm", "need"], default="none", help="Exactly one sidecar backend. auto prefers --need_sidecar_checkpoint, otherwise --sidecar_model")
    p.add_argument("--need_sidecar_checkpoint", default="", help="Optional smaller NEED checkpoint used as the single active sidecar")
    p.add_argument("--need_sidecar_prefer_best", action="store_true", help="Load best checkpoint for the NEED sidecar when present")
    p.add_argument("--need_sidecar_projection_path", default="", help="Projection file/dir from need_sidecar_distill.py")
    p.add_argument("--need_sidecar_projection_weight", type=float, default=1.0, help="Scale for projected NEED-sidecar latent anchors")
    p.add_argument("--need_sidecar_decode_mode", choices=["nonseq", "ar"], default="ar", help="Decoder used by the smaller NEED sidecar for public summaries")
    p.add_argument("--need_sidecar_max_context_tokens", type=int, default=512, help="Max prompt tokens used by the smaller NEED sidecar")
    p.add_argument("--use_need_sidecar_latents", action=argparse.BooleanOptionalAction, default=False, help="Append projected latent anchors from a NEED sidecar when active (opt-in; off for linear core)")
    p.add_argument("--sidecar_call_policy", choices=["always", "latent_gated", "off"], default="off", help="Call the active sidecar only when latent difficulty warrants it")
    p.add_argument("--sidecar_gate_metric", default="latent_difficulty")
    p.add_argument("--sidecar_gate_threshold", type=float, default=0.42)
    p.add_argument("--sidecar_model", default="", help="External LM sidecar for artificial CoT and summaries, e.g. HuggingFaceTB/SmolLM2-135M-Instruct")
    p.add_argument("--sidecar_device", default="same")
    p.add_argument("--sidecar_dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--sidecar_attn_backend", choices=["auto", "sdpa", "flash_attention_2", "eager"], default="sdpa")
    p.add_argument("--sidecar_compile", action="store_true")
    p.add_argument("--sidecar_max_batch", type=int, default=8)
    p.add_argument("--sidecar_max_wait_ms", type=int, default=8)
    p.add_argument("--sidecar_cache_implementation", choices=["static", "dynamic", "offloaded", "none"], default="static")
    p.add_argument("--sidecar_max_context_tokens", type=int, default=2048)
    p.add_argument("--sidecar_trust_remote_code", action="store_true")
    p.add_argument("--sidecar_adapter_path", default="", help="Optional trained sidecar adapter from need_thought_distill.py train_alignment")
    p.add_argument("--sidecar_latent_alignment_path", default="", help="Optional sidecar latent-projection directory containing latent_projection.pt")
    p.add_argument("--use_sidecar_latent_alignment", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--sidecar_latent_alignment_weight", type=float, default=0.35)
    p.add_argument("--latent_tools", action=argparse.BooleanOptionalAction, default=True, help="Enable latent-only calculator/Python tools in the text console")
    p.add_argument("--latent_tool_calculator", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--latent_tool_python", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--latent_tool_sidecar_planning", action=argparse.BooleanOptionalAction, default=False, help="Deprecated compatibility flag; calls are runtime-built without sidecar planning")
    p.add_argument("--latent_tool_router", choices=["deterministic"], default="deterministic", help="Deterministic latent-tool router; no model builds tool calls and no LLRL is required")
    p.add_argument("--latent_tool_max_calls", type=int, default=3)
    p.add_argument("--latent_tool_timeout_s", type=float, default=3.0)
    p.add_argument("--latent_tool_max_code_chars", type=int, default=4000)
    p.add_argument("--latent_tool_max_output_chars", type=int, default=2400)
    p.add_argument("--latent_tool_plan_tokens", type=int, default=0, help="Deprecated compatibility option; no model tool planner is queried")
    p.add_argument("--gpu_l2_mb", type=float, default=96.0)
    p.add_argument("--concurrent_requests", type=int, default=8)
    p.add_argument("--latent_memory_dir", default="")
    p.add_argument("--use_latent_memory", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--latent_memory_k", type=int, default=4)
    p.add_argument("--latent_memory_max_items", type=int, default=256)
    p.add_argument("--use_internal_reasoning_head", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args(argv)
    args.checkpoint = _resolve_checkpoint_arg(args.checkpoint)

    import gradio as gr

    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device=device, prefer_best=args.prefer_best, kernel_backend=args.kernel_backend)
    if getattr(args, "dvsd_router_enabled", None) is None:
        args.dvsd_router_enabled = bool(getattr(model.cfg, "dvsd_router_enabled", False))
    if getattr(args, "dvsd_router_inference_mix", None) is None:
        args.dvsd_router_inference_mix = float(getattr(model.cfg, "dvsd_router_inference_mix", 0.65))
    if getattr(args, "dvsd_router_min_confidence", None) is None:
        args.dvsd_router_min_confidence = float(getattr(model.cfg, "dvsd_router_min_confidence", 0.20))
    if getattr(args, "dvsd_planner_compound_enabled", None) is None:
        args.dvsd_planner_compound_enabled = bool(getattr(model.cfg, "dvsd_planner_compound_enabled", True))
    if getattr(args, "dvsd_planner_compound_mix", None) is None:
        args.dvsd_planner_compound_mix = float(getattr(model.cfg, "dvsd_planner_compound_mix", 0.35))
    if getattr(args, "dvsd_planner_compound_step_size", None) is None:
        args.dvsd_planner_compound_step_size = float(getattr(model.cfg, "dvsd_planner_compound_step_size", 0.65))
    if getattr(args, "dvsd_planner_compound_token_scale", None) is None:
        args.dvsd_planner_compound_token_scale = float(getattr(model.cfg, "dvsd_planner_compound_token_scale", 0.18))
    if getattr(args, "dvsd_planner_compound_descent_scale", None) is None:
        args.dvsd_planner_compound_descent_scale = float(getattr(model.cfg, "dvsd_planner_compound_descent_scale", 0.22))
    if getattr(args, "dvsd_planner_compound_top_k", None) is None:
        args.dvsd_planner_compound_top_k = int(getattr(model.cfg, "dvsd_planner_compound_top_k", 32))
    if getattr(args, "planner_block_space_enabled", None) is None:
        args.planner_block_space_enabled = bool(getattr(model.cfg, "planner_block_space_enabled", True))
    if getattr(args, "planner_block_space_mix", None) is None:
        args.planner_block_space_mix = float(getattr(model.cfg, "planner_block_space_mix", 0.85))
    if getattr(args, "planner_block_space_iters", None) is None:
        args.planner_block_space_iters = int(getattr(model.cfg, "planner_block_space_iters", 1))
    dvsd_runtime_overrides = _apply_dvsd_runtime_overrides(model, args)
    tok = load_tokenizer_for_dir(args.checkpoint)
    # One active sidecar is allowed at runtime: none, external_lm, or a smaller NEED checkpoint.
    sidecar_service = make_single_sidecar_runtime(args, device, model)
    sidecar_plan = dict(getattr(sidecar_service, "cache_plan", {}) or {}) if sidecar_service is not None else {}

    def _sidecar_runtime_obj():
        return getattr(sidecar_service, "runtime", sidecar_service) if sidecar_service is not None else None

    def _sidecar_source() -> str:
        rt = _sidecar_runtime_obj()
        return str(getattr(sidecar_service, "source_type", getattr(rt, "source_type", "none")) if rt is not None else "none")

    def _sidecar_generate(prompt: str, **kwargs: Any) -> str:
        if sidecar_service is None:
            return ""
        return str(sidecar_service.generate(prompt, **kwargs))

    def _sidecar_generate_many(prompts: Sequence[str], **kwargs: Any) -> list[str]:
        if sidecar_service is None:
            return []
        rt = _sidecar_runtime_obj()
        if hasattr(rt, "generate_many"):
            return [str(x) for x in rt.generate_many(prompts, **kwargs)]
        return [_sidecar_generate(p, **kwargs) for p in prompts]

    vt_dir = Path(args.visual_tokenizer) if args.visual_tokenizer else Path(args.checkpoint)
    if load_visual_tokenizer is not None and (vt_dir / "visual_tokenizer_config.json").exists():
        img_tok = load_visual_tokenizer(vt_dir, device=device)
        tok_label = "learned VQ tokenizer"
    else:
        img_tok = make_image_tokenizer(model.cfg)
        tok_label = "dynamic fallback tokenizer"

    raw_cot_log = []
    latent_store = None

    def latent_scaffold(pathway):
        return (
            f"<focus> Ordered latent reasoning path extracted. </focus> "
            f"<latent_state> quality={_float(pathway.get('quality_mean')):.3f}; "
            f"risk={_float(pathway.get('risk_mean')):.3f}; "
            f"effort={_float(pathway.get('adaptive_effort')):.3f}; "
            f"risk_signal={_float(pathway.get('risk_signal_mean', pathway.get('risk_signal'))):.3f}. </latent_state> "
            f"<answer_check> AuxScoreHead-guided decoding, latent slot attention, and mixture energy routing are enabled. </answer_check>"
        )


    def score_chunk(text: str, vectors, cond_scale: float):
        ids = torch.tensor([tok.encode(text, add_bos=True)[-model.cfg.block_size:]], device=device)
        try:
            return model.score_text_risk(ids, conditioning_vectors=vectors, conditioning_scale=cond_scale)
        except Exception:
            return {"quality": 0.5, "risk": 0.5, "contradiction": 0.5}

    def filter_raw_cot(message: str, raw_cot: str, vectors, cond_scale: float):
        parts = [p.strip() for p in raw_cot.replace("\r", "").split("\n") if p.strip()]
        if not parts:
            return raw_cot, {"accepted": 0, "total": 0, "risk": 0.0, "contradiction": 0.0}
        accepted = []
        risks = []
        contras = []
        for part in parts[:12]:
            sc = score_chunk(message + "\n<draft_reasoning_chunk>\n" + part + "\n</draft_reasoning_chunk>", vectors, cond_scale)
            risks.append(float(sc.get("risk", 0.5))); contras.append(float(sc.get("contradiction", 0.5)))
            quality = float(sc.get("quality", 0.5))
            if quality - 0.7 * risks[-1] - 0.7 * contras[-1] >= -0.15:
                accepted.append(part)
        return ("\n".join(accepted) if accepted else raw_cot[:1200]), {"accepted": len(accepted), "total": len(parts), "risk": sum(risks)/max(1,len(risks)), "contradiction": sum(contras)/max(1,len(contras))}

    def make_context(
        message: str,
        vector_stride: int,
        max_vectors: int,
        thought_tokens: int,
        include_raw_cot_context: bool,
        store_raw_cot: bool,
        show_cache_plan: bool,
        cond_scale: float,
        use_latent_memory: bool,
        reasoning_tree_branches: int,
        reasoning_tree_keep: int,
        auto_output_mode: bool,
        latent_tools: bool,
        cot_temperature: float = 0.45,
        cot_top_p: float = 0.92,
        cot_top_k: int = 80,
        summary_temperature: float = 0.25,
        summary_top_p: float = 0.90,
        summary_top_k: int = 60,
        max_summary_chars: int = 2000,
        raw_cot_history_chars: int = 12000,
        latent_tool_max_calls: int | None = None,
        latent_tool_timeout_s: float | None = None,
        latent_tool_max_code_chars: int | None = None,
        latent_tool_max_output_chars: int | None = None,
    ):
        nonlocal latent_store
        base = torch.tensor([tok.encode(message, add_bos=True)[: model.cfg.block_size]], device=device)
        with need_optimization_mode():
            pathway = model.latent_pathway(base, stride=int(vector_stride), max_vectors=int(max_vectors))
        vectors = pathway["pathway_vectors"]
        memory_write_vectors = vectors
        latent_metrics = _compute_latent_convergence_metrics(pathway, model)
        latent_summary = latent_scaffold(pathway)
        sidecar_enabled, sidecar_gate = _sidecar_should_run(args, latent_metrics)
        latent_metrics.update(sidecar_gate)
        latent_summary += "\n<sidecar_gate>" + html.escape(json.dumps(sidecar_gate, sort_keys=True)) + "</sidecar_gate>"
        rt_obj = _sidecar_runtime_obj() if sidecar_enabled else None
        if sidecar_enabled and sidecar_service is not None and _sidecar_source() == "need" and bool(getattr(args, "use_need_sidecar_latents", False)):
            try:
                sidecar_summary, sidecar_vectors, sidecar_metrics = sidecar_service.latent_guidance(message, vector_stride=max(1, int(vector_stride)), max_vectors=max(1, int(max_vectors) // 2))
                latent_summary += "\n<need_sidecar_summary>" + str(sidecar_summary)[:1200] + "</need_sidecar_summary>"
                if torch.is_tensor(sidecar_vectors) and sidecar_vectors.size(-1) == vectors.size(-1):
                    vectors = torch.cat([vectors, sidecar_vectors.to(device=device, dtype=vectors.dtype)], dim=1)
                    latent_summary += "\n<need_sidecar_latents>Projected smaller-NEED latent anchors appended.</need_sidecar_latents>"
                elif sidecar_vectors is not None:
                    latent_summary += f"\n<need_sidecar_latents_error>dim mismatch: sidecar={sidecar_vectors.size(-1)} need={vectors.size(-1)}</need_sidecar_latents_error>"
            except Exception as exc:
                latent_summary += "\n<need_sidecar_error>" + html.escape(str(exc))[:600] + "</need_sidecar_error>"
        elif sidecar_enabled and bool(getattr(args, "use_sidecar_latent_alignment", False)) and sidecar_service is not None and hasattr(rt_obj, "encode_latent_alignment") and getattr(rt_obj, "latent_projection", None) is not None:
            try:
                sidecar_vec = rt_obj.encode_latent_alignment([message], max_length=args.sidecar_max_context_tokens)
                if sidecar_vec.size(-1) == vectors.size(-1):
                    sidecar_vec = sidecar_vec.to(device=device, dtype=vectors.dtype) * float(args.sidecar_latent_alignment_weight)
                    vectors = torch.cat([vectors, sidecar_vec], dim=1)
                    latent_summary += "\n<sidecar_latent_alignment>Loaded trained sidecar latent projection as a behavioral latent anchor.</sidecar_latent_alignment>"
            except Exception:
                pass
        verbal_mode = "full_artificial_cot"
        try:
            if auto_output_mode:
                decision = model.output_mode_decision(base)
                verbal_mode = max(decision, key=decision.get)
                latent_summary += "\n<output_mode_classifier>" + str(decision) + "</output_mode_classifier>"
        except Exception:
            verbal_mode = "full_artificial_cot"
        retrieved_text = ""
        if use_latent_memory and args.latent_memory_dir and latent_store is None:
            latent_store = LatentMemoryStore(args.latent_memory_dir, dim=model.cfg.d_model, max_items=args.latent_memory_max_items)
        if use_latent_memory and latent_store is not None:
            retrieved_text, retrieved_vectors = latent_store.retrieve(vectors, k=args.latent_memory_k)
            if retrieved_vectors is not None:
                vectors = torch.cat([retrieved_vectors.to(device=device, dtype=vectors.dtype), vectors], dim=1)
        prior_raw = "\n\n".join(raw_cot_log[-8:]) if include_raw_cot_context else ""
        raw_cot = ""
        sidecar_allows_reasoning = bool(
            sidecar_service is not None
            and getattr(sidecar_service, "supports_reasoning_sidecar", _sidecar_source() != "need")
        )
        if sidecar_enabled and sidecar_service is not None and sidecar_allows_reasoning and verbal_mode not in ("none", "renderer_only"):
            with sidecar_optimization_mode():
                tree_summary = latent_summary + ("\n" + retrieved_text if retrieved_text else "")
                branches = max(1, int(reasoning_tree_branches))
                if branches > 1 or verbal_mode == "multi_cot":
                    prompts = []
                    for i in range(branches):
                        style = ["direct", "counterexample", "stepwise", "memory", "uncertainty"][i % 5]
                        prompts.append(_sidecar_runtime_obj().artificial_cot_prompt(message, tree_summary, prior_raw) + f"\nBranch style: {style}.\n")
                    candidates = _sidecar_generate_many(prompts, max_new_tokens=int(thought_tokens), temperature=float(cot_temperature), top_p=float(cot_top_p), top_k=int(cot_top_k))
                    scored = []
                    for cand in candidates:
                        sc = score_chunk(message + "\n<branch>" + cand + "</branch>", vectors, cond_scale)
                        bscore = float(sc.get("quality",0.5)) - 0.75*float(sc.get("risk",0.5)) - 0.85*float(sc.get("contradiction",0.5))
                        scored.append((bscore, cand))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    raw_cot = "\n".join(f"<accepted_branch score={a:.3f}>\n{b}\n</accepted_branch>" for a,b in scored[:max(1,int(reasoning_tree_keep))])
                else:
                    cot_prompt = _sidecar_runtime_obj().artificial_cot_prompt(message, tree_summary, prior_raw)
                    raw_cot = _sidecar_generate(cot_prompt, max_new_tokens=int(thought_tokens), temperature=float(cot_temperature), top_p=float(cot_top_p), top_k=int(cot_top_k))
                sum_prompt = _sidecar_runtime_obj().summary_prompt(message, raw_cot, latent_summary)
                summary = _sidecar_generate(sum_prompt, max_new_tokens=min(int(thought_tokens), 180), temperature=float(summary_temperature), top_p=float(summary_top_p), top_k=int(summary_top_k))
        elif args.use_internal_reasoning_head:
            with torch.no_grad(), need_optimization_mode():
                ids_internal = model.internal_reasoning_summary(base, max_tokens=int(thought_tokens))
            raw_cot = tok.decode(ids_internal[0].tolist())
            summary = raw_cot or latent_summary
        else:
            # No sidecar reasoning was produced.  For NEED sidecars this is the
            # default and required behavior: keep latent guidance as the public
            # summary, but do not relabel it as raw artificial CoT.
            raw_cot = "" if _sidecar_source() == "need" else latent_summary
            summary = latent_summary
        max_summary_chars = max(0, int(max_summary_chars))
        if max_summary_chars and len(summary) > max_summary_chars:
            summary = summary[:max_summary_chars].rstrip() + "\n<truncated_summary chars=\"" + str(max_summary_chars) + "\" />"
        raw_cot, cot_score = filter_raw_cot(message, raw_cot, vectors, cond_scale)
        summary += f"\n<faithfulness> risk={cot_score.get('risk',0):.3f}; contradiction={cot_score.get('contradiction',0):.3f}; accepted_chunks={cot_score.get('accepted',0)}/{cot_score.get('total',0)}. </faithfulness>"
        tool_args = SimpleNamespace(
            latent_tools=bool(latent_tools),
            latent_tool_calculator=bool(args.latent_tool_calculator),
            latent_tool_python=bool(args.latent_tool_python),
            latent_tool_sidecar_planning=False,
            latent_tool_max_calls=int(latent_tool_max_calls if latent_tool_max_calls is not None else args.latent_tool_max_calls),
            latent_tool_timeout_s=float(latent_tool_timeout_s if latent_tool_timeout_s is not None else args.latent_tool_timeout_s),
            latent_tool_max_code_chars=int(latent_tool_max_code_chars if latent_tool_max_code_chars is not None else args.latent_tool_max_code_chars),
            latent_tool_max_output_chars=int(latent_tool_max_output_chars if latent_tool_max_output_chars is not None else args.latent_tool_max_output_chars),
            latent_tool_plan_tokens=int(args.latent_tool_plan_tokens),
        )
        latent_tool_context, latent_tool_metrics = _run_latent_tools(message, summary, raw_cot, sidecar_service if sidecar_enabled else None, tool_args)
        if latent_tool_metrics.get("tool_calls", 0):
            summary += f"\n<latent_tool_status> runtime-built internal tool calls={latent_tool_metrics.get('tool_calls',0)} successes={latent_tool_metrics.get('tool_successes',0)}; no model-built calls. </latent_tool_status>"
        if store_raw_cot and str(raw_cot or "").strip():
            raw_cot_log.append(raw_cot[-max(1000, int(raw_cot_history_chars)):])
            del raw_cot_log[:-16]
        if latent_store is not None and bool(use_latent_memory):
            latent_store.add(message, summary, memory_write_vectors.detach().cpu())
        augmented = message
        if retrieved_text:
            augmented += "\n\n<behavioral_latent_memory_guidance>\n" + retrieved_text[-4000:] + "\n</behavioral_latent_memory_guidance>"
        if latent_tool_context:
            augmented += '\n\n<latent_tool_results visibility="internal">\n' + latent_tool_context + '\n</latent_tool_results>'
        augmented += "\n\n<public_thought_summary>\n" + summary + "\n</public_thought_summary>"
        if include_raw_cot_context:
            augmented += "\n\n<raw_artificial_cot_context>\n" + (prior_raw + "\n\n" + raw_cot)[-max(1000, int(raw_cot_history_chars)):] + "\n</raw_artificial_cot_context>"
        augmented += "\n\nFinal answer:"
        ids = torch.tensor([tok.encode(augmented, add_bos=True)[-model.cfg.block_size:]], device=device)
        if show_cache_plan and sidecar_plan:
            summary += "\n\n<sidecar_cache_plan>\n" + str(sidecar_plan) + "\n</sidecar_cache_plan>"
        return ids, vectors, summary, raw_cot, latent_metrics, latent_tool_metrics

    def chat(
        message, history, system_prompt,
        max_new, temp, top_p, top_k, typical_p, repetition_penalty, no_repeat_ngram, min_new_tokens, lookahead_blend,
        decode_mode, nonseq_dynamic, nonseq_min_heads, nonseq_max_heads, nonseq_refine_steps, nonseq_refine_causal_blend, nonseq_refine_confidence_floor,
        nonseq_refine_temperature_decay, nonseq_refine_lock_schedule, nonseq_refine_resample_locked, dvsd_router_enabled, dvsd_router_inference_mix, dvsd_router_min_confidence,
        dual, show_summary, show_raw, include_raw_cot_context, store_raw_cot, show_cache_plan, proactive,
        aux_score_weight, aux_score_top_k, risk_threshold, contradiction_threshold, aux_score_candidate_pool, aux_score_backtrack_window, aux_score_max_backtracks,
        cond_scale, vector_stride, max_vectors, thought_tokens, max_summary_chars, raw_cot_history_chars,
        reasoning_tree_branches, reasoning_tree_keep, auto_output_mode, cot_temperature, cot_top_p, cot_top_k, summary_temperature, summary_top_p, summary_top_k,
        latent_search_depth, latent_search_branches, use_latent_memory, latent_tools,
        latent_tool_max_calls, latent_tool_timeout_s, latent_tool_max_code_chars, latent_tool_max_output_chars,
        show_dashboard, speculative_final, adaptive_spec, target_accept_rate, spec_draft_tokens, spec_draft_temperature, spec_draft_top_p, spec_draft_top_k,
        spec_max_need_tokens_per_draft, spec_accept_top_k, spec_accept_min_need_prob, spec_accept_gap, spec_risk_threshold, spec_contradiction_threshold, spec_repetition_threshold, spec_context_chars,
        adaptive_accept_feedback_gain, adaptive_accept_aux_score_tighten, adaptive_accept_min_top_k, adaptive_accept_max_top_k, adaptive_accept_min_gap, adaptive_accept_max_gap,
        adaptive_accept_min_draft_tokens, adaptive_accept_max_draft_tokens, adaptive_accept_min_need_tokens, adaptive_accept_max_need_tokens, adaptive_accept_min_min_prob, adaptive_accept_max_min_prob,
        adaptive_accept_min_risk_threshold, adaptive_accept_max_risk_threshold, adaptive_accept_min_contradiction_threshold, adaptive_accept_max_contradiction_threshold,
        adaptive_accept_min_repetition_threshold, adaptive_accept_max_repetition_threshold,
    ):
        system_prompt = (system_prompt or "").strip()
        model_message = (system_prompt + "\n\nUser: " + message.strip()) if system_prompt else message
        if dual:
            t0 = time.perf_counter()
            ids, vectors, summary, raw_cot, latent_metrics, tool_metrics = make_context(
                model_message, vector_stride, max_vectors, thought_tokens, include_raw_cot_context, store_raw_cot, show_cache_plan,
                float(cond_scale), bool(use_latent_memory), int(reasoning_tree_branches), int(reasoning_tree_keep), bool(auto_output_mode), bool(latent_tools),
                float(cot_temperature), float(cot_top_p), int(cot_top_k), float(summary_temperature), float(summary_top_p), int(summary_top_k),
                int(max_summary_chars), int(raw_cot_history_chars), int(latent_tool_max_calls), float(latent_tool_timeout_s), int(latent_tool_max_code_chars), int(latent_tool_max_output_chars),
            )
            prep_s = time.perf_counter() - t0
        else:
            ids = torch.tensor([tok.encode(model_message, add_bos=True)], device=device)
            vectors = None
            summary = "<thought_summary> disabled </thought_summary>"
            raw_cot = ""
            latent_metrics = {}
            tool_metrics = {"latent_tools_enabled": False, "tool_calls": 0}
            prep_s = 0.0
        if hasattr(model.cfg, "dvsd_router_enabled"):
            model.cfg.dvsd_router_enabled = bool(dvsd_router_enabled)
        if hasattr(model.cfg, "dvsd_router_inference_mix"):
            model.cfg.dvsd_router_inference_mix = float(dvsd_router_inference_mix)
        if hasattr(model.cfg, "dvsd_router_min_confidence"):
            model.cfg.dvsd_router_min_confidence = float(dvsd_router_min_confidence)
        gen_t0 = time.perf_counter()
        spec_metrics = {}
        gen_args = SimpleNamespace(
            max_new_tokens=int(max_new),
            temperature=float(temp),
            top_k=int(top_k),
            top_p=float(top_p),
            typical_p=float(typical_p),
            repetition_penalty=float(repetition_penalty),
            no_repeat_ngram=int(no_repeat_ngram),
            min_new_tokens=int(min_new_tokens),
            lookahead_blend=float(lookahead_blend),
            aux_score_top_k=int(aux_score_top_k),
            aux_score_weight=float(aux_score_weight) if proactive else 0.0,
            disable_proactive_aux_score=not bool(proactive),
            aux_score_risk_threshold=float(risk_threshold),
            aux_score_contradiction_threshold=float(contradiction_threshold),
            aux_score_candidate_pool=int(aux_score_candidate_pool),
            aux_score_backtrack_window=int(aux_score_backtrack_window),
            aux_score_max_backtracks=int(aux_score_max_backtracks),
            latent_search_depth=int(latent_search_depth),
            latent_search_branches=int(latent_search_branches),
            spec_draft_tokens=int(spec_draft_tokens),
            spec_draft_temperature=float(spec_draft_temperature),
            spec_draft_top_p=float(spec_draft_top_p),
            spec_draft_top_k=int(spec_draft_top_k),
            spec_max_need_tokens_per_draft=int(spec_max_need_tokens_per_draft),
            spec_accept_top_k=int(spec_accept_top_k),
            spec_accept_min_need_prob=float(spec_accept_min_need_prob),
            spec_accept_max_logprob_gap=float(spec_accept_gap),
            spec_risk_threshold=float(spec_risk_threshold),
            spec_contradiction_threshold=float(spec_contradiction_threshold),
            spec_repetition_threshold=float(spec_repetition_threshold),
            spec_context_chars=int(spec_context_chars),
            adaptive_spec_acceptance=bool(adaptive_spec),
            adaptive_accept_target_rate=float(target_accept_rate),
            adaptive_accept_feedback_gain=float(adaptive_accept_feedback_gain),
            adaptive_accept_aux_score_tighten=float(adaptive_accept_aux_score_tighten),
            adaptive_accept_min_top_k=int(adaptive_accept_min_top_k),
            adaptive_accept_max_top_k=int(adaptive_accept_max_top_k),
            adaptive_accept_min_gap=float(adaptive_accept_min_gap),
            adaptive_accept_max_gap=float(adaptive_accept_max_gap),
            adaptive_accept_min_draft_tokens=int(adaptive_accept_min_draft_tokens),
            adaptive_accept_max_draft_tokens=int(adaptive_accept_max_draft_tokens),
            adaptive_accept_min_need_tokens=int(adaptive_accept_min_need_tokens),
            adaptive_accept_max_need_tokens=int(adaptive_accept_max_need_tokens),
            adaptive_accept_min_min_prob=float(adaptive_accept_min_min_prob),
            adaptive_accept_max_min_prob=float(adaptive_accept_max_min_prob),
            adaptive_accept_min_risk_threshold=float(adaptive_accept_min_risk_threshold),
            adaptive_accept_max_risk_threshold=float(adaptive_accept_max_risk_threshold),
            adaptive_accept_min_contradiction_threshold=float(adaptive_accept_min_contradiction_threshold),
            adaptive_accept_max_contradiction_threshold=float(adaptive_accept_max_contradiction_threshold),
            adaptive_accept_min_repetition_threshold=float(adaptive_accept_min_repetition_threshold),
            adaptive_accept_max_repetition_threshold=float(adaptive_accept_max_repetition_threshold),
            _latent_convergence_metrics=latent_metrics,
        )
        if bool(speculative_final) and sidecar_service is not None and bool(getattr(sidecar_service, "supports_speculative_final_decode", True)):
            out, spec_metrics = _speculative_final_decode(model, tok, ids, sidecar_service, gen_args, vectors, float(cond_scale) if dual else 0.0, device)
        else:
            mode = str(decode_mode or "ar")
            if getattr(args, "nonseq_decode", None) is not None:
                mode = "nonseq" if bool(args.nonseq_decode) else "ar"
            cfg_max_heads = int(getattr(model.cfg, "nonseq_max_heads", getattr(model.cfg, "n_predict_heads", 1)))
            requested_max_heads = int(nonseq_max_heads) if int(nonseq_max_heads) > 0 else cfg_max_heads
            core_streaming = bool(getattr(model.cfg, "strict_linear_core", True)) and bool(getattr(model.cfg, "streaming_generation", True))
            use_nonseq = mode == "nonseq" or (mode == "auto" and (not core_streaming) and int(getattr(model.cfg, "n_predict_heads", 1)) > 1 and requested_max_heads > 1)
            with need_optimization_mode():
                if use_nonseq:
                    out, spec_metrics = model.generate_text_nonsequential(
                        ids,
                        max_new_tokens=int(max_new),
                        temperature=float(temp),
                        top_p=float(top_p),
                        top_k=int(top_k),
                        typical_p=float(typical_p),
                        repetition_penalty=float(repetition_penalty),
                        no_repeat_ngram=int(no_repeat_ngram),
                        min_new_tokens=int(min_new_tokens),
                        lookahead_blend=float(lookahead_blend),
                        aux_score_top_k=int(aux_score_top_k),
                        aux_score_weight=float(aux_score_weight) if proactive else 0.0,
                        proactive_aux_score=bool(proactive),
                        aux_score_risk_threshold=float(risk_threshold),
                        aux_score_contradiction_threshold=float(contradiction_threshold),
                        aux_score_candidate_pool=int(aux_score_candidate_pool),
                        aux_score_backtrack_window=int(aux_score_backtrack_window),
                        aux_score_max_backtracks=int(aux_score_max_backtracks),
                        latent_search_depth=int(latent_search_depth),
                        latent_search_branches=int(latent_search_branches),
                        conditioning_vectors=vectors,
                        conditioning_scale=float(cond_scale) if dual else 0.0,
                        nonseq_dynamic=bool(nonseq_dynamic),
                        nonseq_min_heads=int(nonseq_min_heads),
                        nonseq_max_heads=None if int(nonseq_max_heads) <= 0 else int(nonseq_max_heads),
                        nonseq_refine_steps=int(nonseq_refine_steps),
                        nonseq_refine_causal_blend=float(nonseq_refine_causal_blend),
                        nonseq_refine_confidence_floor=float(nonseq_refine_confidence_floor),
                        nonseq_refine_temperature_decay=float(nonseq_refine_temperature_decay),
                        nonseq_refine_lock_schedule=str(nonseq_refine_lock_schedule or "cosine"),
                        nonseq_refine_resample_locked=bool(nonseq_refine_resample_locked),
                        return_stats=True,
                    )
                    spec_metrics = {"dvsd_" + str(k): v for k, v in dict(spec_metrics).items()}
                else:
                    out = model.generate_text(
                        ids,
                        max_new_tokens=int(max_new),
                        temperature=float(temp),
                        top_p=float(top_p),
                        top_k=int(top_k),
                        typical_p=float(typical_p),
                        repetition_penalty=float(repetition_penalty),
                        no_repeat_ngram=int(no_repeat_ngram),
                        min_new_tokens=int(min_new_tokens),
                        lookahead_blend=float(lookahead_blend),
                        aux_score_top_k=int(aux_score_top_k),
                        aux_score_weight=float(aux_score_weight) if proactive else 0.0,
                        proactive_aux_score=bool(proactive),
                        aux_score_risk_threshold=float(risk_threshold),
                        aux_score_contradiction_threshold=float(contradiction_threshold),
                        aux_score_candidate_pool=int(aux_score_candidate_pool),
                        aux_score_backtrack_window=int(aux_score_backtrack_window),
                        aux_score_max_backtracks=int(aux_score_max_backtracks),
                        latent_search_depth=int(latent_search_depth),
                        latent_search_branches=int(latent_search_branches),
                        conditioning_vectors=vectors,
                        conditioning_scale=float(cond_scale) if dual else 0.0,
                    )
        decode_s = time.perf_counter() - gen_t0
        text = tok.decode(out[0].tolist())
        if dual:
            text = _extract_final_answer_text(text)
        text = _strip_hidden_runtime_artifacts(text)
        prefix = ""
        if show_summary:
            prefix += f"```text\n<thought_summary>\n{summary}\n</thought_summary>\n```\n\n"
        if show_raw:
            prefix += f"```text\n<raw_artificial_cot>\n{raw_cot}\n</raw_artificial_cot>\n```\n\n"
        if show_dashboard:
            gen_tokens = max(0, int(out.size(1) - ids.size(1)))
            dash = {"prep_s": prep_s, "decode_s": decode_s, "need_tokens_per_s": gen_tokens / max(decode_s, 1e-9), "generated_tokens": gen_tokens, "latent_tools": tool_metrics}
            dash.update(dvsd_runtime_overrides)
            dash.update(latent_metrics)
            dash.update(spec_metrics)
            prefix += "```text\n<performance_dashboard>\n" + str(dash) + "\n</performance_dashboard>\n```\n\n"
        return prefix + text

    def image(prompt, negative, grid, steps, temp, cfg_scale, quality, top_k, size, mask_schedule, gumbel_noise, min_keep):
        ids = torch.tensor([tok.encode(prompt, add_bos=True)], device=device)
        neg = torch.tensor([tok.encode(negative, add_bos=True)], device=device) if negative else None
        toks = model.generate_image_tokens(
            ids, grid=int(grid), steps=int(steps), temperature=float(temp), top_k=int(top_k),
            quality_guidance=float(quality), negative_prompt_ids=neg, cfg_scale=float(cfg_scale),
            mask_schedule=str(mask_schedule or "cosine"), gumbel_noise=float(gumbel_noise), min_keep=int(min_keep),
        )
        return img_tok.decode_tokens(toks[0].tolist(), grid=int(grid), size=int(size))

    compare_cache = {}

    def _simple_model_generate(active_model, prompt_text: str, max_new: int, temp: float, top_p: float, top_k: int = 50) -> str:
        ids = torch.tensor([tok.encode(prompt_text, add_bos=True)], device=device)
        with torch.no_grad():
            out = active_model.generate_text(
                ids[:, -active_model.cfg.block_size:],
                max_new_tokens=int(max_new),
                temperature=float(temp),
                top_k=int(top_k),
                top_p=float(top_p),
                aux_score_weight=0.0,
                proactive_aux_score=False,
                aux_score_risk_threshold=0.72,
                aux_score_contradiction_threshold=0.65,
            )
        text = tok.decode(out[0].tolist())
        if text.startswith(prompt_text):
            text = text[len(prompt_text):].strip()
        return _strip_hidden_runtime_artifacts(_extract_final_answer_text(text))

    def compare_checkpoints(checkpoint_b, message, system_prompt, max_new, temp, top_p):
        checkpoint_b = str(checkpoint_b or args.compare_checkpoint or "").strip()
        message = str(message or "").strip()
        if not message:
            return "", "", "Enter a prompt to compare."
        if not checkpoint_b:
            return "", "", "Set checkpoint B or launch with --compare_checkpoint."
        prompt_text = ((system_prompt or "").strip() + "\n\nUser: " + message) if (system_prompt or "").strip() else message
        try:
            out_a = _simple_model_generate(model, prompt_text, int(max_new), float(temp), float(top_p))
            cache_key = (checkpoint_b, bool(args.compare_prefer_best))
            if cache_key not in compare_cache:
                compare_cache[cache_key] = load_model(checkpoint_b, device=device, prefer_best=bool(args.compare_prefer_best), kernel_backend=args.kernel_backend)
            out_b = _simple_model_generate(compare_cache[cache_key], prompt_text, int(max_new), float(temp), float(top_p))
            summary = {
                "checkpoint_a": str(args.checkpoint),
                "checkpoint_b": checkpoint_b,
                "chars_a": len(out_a),
                "chars_b": len(out_b),
                "note": "Side-by-side qualitative comparison only; no saved regression artifacts.",
            }
            return out_a, out_b, "```json\n" + json.dumps(summary, indent=2) + "\n```"
        except Exception as exc:
            return "", "", f"Comparison failed: {exc}"

    def _browser_css():
        return """
        :root { 
            color-scheme: dark; 
            --bg: #0f0f12;
            --ink: #f1eee8;
            --soft: #bdb7ad;
            --faint: #817a73;
            --line: #373238;
            --line2: #252329;
            --panel: #17171b;
            --panel2: #111115;
            --field: #0b0b0e;
            --gold: #e1c667;
            --red: #ef7f65;
            --violet: #c78af5;
            --green: #b4da6b;
            --orange: #f0a44c;
            --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
            --sans: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
        }
        .gradio-container {
            min-height: 100vh;
            background: var(--bg) !important;
            color: var(--ink) !important;
            font-family: var(--sans) !important;
        }
        .gradio-container::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            opacity: 0.72;
            background:
                radial-gradient(circle at 12% 18%, rgba(199,138,245,.11), transparent 25rem),
                radial-gradient(circle at 76% 6%, rgba(239,127,101,.10), transparent 22rem),
                linear-gradient(90deg, rgba(255,255,255,.028) 1px, transparent 1px),
                linear-gradient(0deg, rgba(255,255,255,.018) 1px, transparent 1px);
            background-size: auto, auto, 44px 44px, 44px 44px;
        }
        .need-shell { max-width: 1280px; margin: 0 auto; padding: 0.75rem 0.75rem 0; position: relative; }
        .need-bar { display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid var(--line); padding: 0.65rem 0 0.62rem; gap: 1rem; }
        .need-wordmark { font-weight: 760; letter-spacing: -0.035em; font-size: 1.1rem; color: var(--ink); }
        .need-runtime { display: flex; flex-wrap: wrap; gap: 0.5rem 1rem; color: var(--soft); font-size: 0.82rem; }
        .need-runtime span { border-left: 1px solid #4a444b; padding-left: 0.7rem; }
        .need-runtime span:first-child { border-left: 0; padding-left: 0; }
        
        .need-strip { display: flex; align-items: center; gap: 0.5rem; overflow-x: auto; padding: 0.45rem 0 0.15rem; min-height: 3rem; }
        .need-attach { flex: 0 0 auto; display: inline-flex; align-items: center; gap: 0.45rem; min-width: 8.2rem; max-width: 15rem; border: 1px solid var(--line); background: var(--panel); padding: 0.45rem 0.55rem; border-radius: 3px; color: var(--ink); }
        .need-attach img { width: 2.3rem; height: 2.3rem; object-fit: cover; border-radius: 2px; border: 1px solid var(--line2); }
        .need-attach-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 0.82rem; }
        .need-attach-meta { color: var(--soft); font-size: 0.72rem; }
        
        .need-panel { border: 1px solid var(--line); background: var(--panel); border-radius: 3px; padding: 0.7rem 0.78rem; margin: 0.45rem 0; }
        .need-panel-line { color: #d8d1c6; font-size: 0.9rem; line-height: 1.45; }
        .need-muted { color: var(--soft); font-size: 0.86rem; line-height: 1.45; }
        
        .need-send-row button, button.primary { border-radius: 3px !important; border: 1px solid #9a7841 !important; background: #2a2317 !important; color: var(--ink) !important; box-shadow: none !important; transition: filter 120ms ease, border-color 120ms ease !important; }
        .need-send-row button:hover, button.primary:hover { filter: brightness(1.08); border-color: var(--gold) !important; }
        
        textarea, input, .wrap, .block, .form, .panel, .tabs, .tabitem, .input-container, .output-class { background-color: var(--panel2) !important; border-color: var(--line) !important; color: var(--ink) !important; border-radius: 3px !important; }
        .tab-nav button { color: var(--soft) !important; border-radius: 0 !important; }
        .tab-nav button.selected { color: var(--ink) !important; border-color: var(--gold) !important; }
        
        .need-chatbot { border-radius: 3px !important; overflow: hidden; border: 1px solid var(--line) !important; }
        .need-chatbot .message { border-radius: 3px !important; }
        
        .need-control-accordion { border-radius: 3px !important; overflow: hidden; }
        .need-footnote { color: var(--faint); font-size: 0.78rem; margin: 0.75rem 0 0 0; }
        
        /* Updated Slider Accents based on your CSS palette */
        .need-control-accordion:nth-of-type(5n+1) input[type=range] { accent-color: var(--gold) !important; }
        .need-control-accordion:nth-of-type(5n+2) input[type=range] { accent-color: var(--green) !important; }
        .need-control-accordion:nth-of-type(5n+3) input[type=range] { accent-color: var(--violet) !important; }
        .need-control-accordion:nth-of-type(5n+4) input[type=range] { accent-color: var(--orange) !important; }
        .need-control-accordion:nth-of-type(5n+5) input[type=range] { accent-color: var(--red) !important; }
        
        .need-cot-box textarea { font-family: var(--mono) !important; }
        
        .need-ascii { display: flex; align-items: center; justify-content: space-between; gap: 1rem; border-bottom: 1px solid var(--line); margin: 0.35rem auto 0.55rem; max-width: 1280px; background: var(--panel2); padding: 0.55rem 0.8rem; overflow: hidden; }
        .need-logo-stack { flex: 0 0 auto; display: grid; gap: 0.38rem; }
        .need-logo-ascii { margin: 0; color: var(--gold); font: 900 0.88rem/0.95 var(--mono); letter-spacing: 0.02em; white-space: pre; text-shadow: 0 0 6px rgba(225, 198, 103, 0.4), 0 0 1px rgba(255, 255, 255, 0.7); }
        
        /* Lava Lamp animation integrated for the header */
        .need-terminal-flair { 
            min-width: 0; max-width: 40rem; margin: 0; white-space: pre; overflow: hidden; contain: paint;
            min-height: 78px;
            font: 0.66rem/1 var(--mono);
            background: linear-gradient(135deg, var(--gold), var(--violet), var(--red), var(--orange), var(--green));
            background-size: 300% 300%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: lavaGlow 6s ease infinite;
            text-shadow: 0 0 12px rgba(225, 198, 103, 0.25);
        }
        @keyframes lavaGlow { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
        
        .need-cat { flex: 0 0 auto; margin: 0; color: var(--soft); font: 0.78rem/1 var(--mono); white-space: pre; cursor: default; min-width: 4.6rem; text-align: left; }
        .need-cat .closed { display: none; }
        .need-cat:hover .open { display: none; }
        .need-cat:hover .closed { display: block; }
        
        @media (max-width: 760px) { .need-ascii { align-items: flex-start; flex-direction: column; gap: 0.45rem; } .need-terminal-flair { min-width: 0; font-size: 0.66rem; } .need-bar { align-items: flex-start; flex-direction: column; } .need-runtime span { border-left: 0; padding-left: 0; } }
        """

    def _status_html():
        checkpoint_name = html.escape(Path(args.checkpoint).name or str(args.checkpoint))
        selected = getattr(args, "_sidecar_selection", {}) or {}
        sidecar_backend = selected.get("backend", _sidecar_source()) if isinstance(selected, dict) else _sidecar_source()
        if sidecar_backend == "need":
            sidecar_state = "NEED sidecar"
        elif sidecar_backend == "external_lm":
            sidecar_state = "external LM sidecar"
        else:
            sidecar_state = "NEED only"
        memory_state = "memory on" if bool(args.use_latent_memory and args.latent_memory_dir) else "memory off"
        tool_state = "latent tools on" if bool(args.latent_tools) else "latent tools off"
        cache_state = sidecar_plan.get("cache_implementation", "need") if _sidecar_source() == "need" else (sidecar_plan.get("cache_implementation", args.sidecar_cache_implementation) if sidecar_plan else "none")
        return f"""
        <div class="need-shell">
          <div class="need-bar">
            <div class="need-wordmark">NEED</div>
            <div class="need-runtime">
              <span>{checkpoint_name}</span>
              <span>{html.escape(str(device))}</span>
              <span>{html.escape(tok_label)}</span>
              <span>{html.escape(sidecar_state)}</span>
              <span>KV {html.escape(str(cache_state))}</span>
              <span>{html.escape(tool_state)}</span>
              <span>{html.escape(memory_state)}</span>
            </div>
          </div>
        </div>
        """

    def _ascii_html():
        logo = r"""███╗   ██╗███████╗███████╗██████╗
████╗  ██║██╔════╝██╔════╝██╔══██╗
██╔██╗ ██║█████╗  █████╗  ██║  ██║
██║╚██╗██║██╔══╝  ██╔══╝  ██║  ██║
██║ ╚████║███████╗███████╗██████╔╝
╚═╝  ╚═══╝╚══════╝╚══════╝╚═════╝"""
        
        return f"""
        <div class="need-ascii">
          <div class="need-logo-stack">
            <pre class="need-logo-ascii">{html.escape(logo)}</pre>
            <pre class="need-terminal-flair" id="need-ascii-frame"></pre>
          </div>
          <div class="need-cat" aria-label="cat">
            <pre class="open"> /&#92;_/&#92;
( o.o )
 /   &#92;</pre>
            <pre class="closed"> /&#92;_/&#92;
( -.- )
 /   &#92;</pre>
          </div>
        </div>
        <script>
        (() => {{
          const heatChars = "        .:-=+*#%@";
          const simCols = 38;
          const simRows = 9;
          const shell = document.currentScript.previousElementSibling;
          const pre = shell && shell.querySelector('#need-ascii-frame');
          if (!pre || pre.dataset.bound === '1') return;
          pre.dataset.bound = '1';
          
          function runLavaFire() {{
            const t = performance.now() * 0.0008;
            let frame = "";
            for (let y = 0; y < simRows; y++) {{
              for (let x = 0; x < simCols; x++) {{
                let nx = x / simCols;
                let ny = y / simRows;
                let v1 = Math.sin(nx * 8 + t * 1.2 + Math.cos(ny * 5 - t * 2));
                let v2 = Math.sin(nx * 14 - t * 0.8 + Math.sin(ny * 8 - t * 2.5));
                let v3 = Math.cos(nx * 4 + ny * 6 - t * 1.5);
                let combined = (v1 + v2 * 0.6 + v3) / 2.6; 
                let rawIntensity = (1.6 - ny * 2.2) + (combined * 5.0);
                let intensity = 1 / (1 + Math.exp(-rawIntensity));
                let edgeMask = 0.15 + 0.85 * Math.sin(nx * Math.PI);
                intensity *= edgeMask;
                let idx = Math.floor(intensity * heatChars.length);
                if (idx < 0) idx = 0;
                if (idx >= heatChars.length) idx = heatChars.length - 1;
                frame += heatChars[idx];
              }}
              frame += "\\n";
            }}
            pre.textContent = frame;
            requestAnimationFrame(runLavaFire);
          }}
          runLavaFire();
        }})();
        </script>
        """

    def _text_recipe_values(name: str):
        base = [128, 0.8, 0.95, True, True, False, False, True, False, True, 0.18, 0.72, 0.65, 2, 512, 160, 1, 1, True, 0, 1, bool(args.use_latent_memory and args.latent_memory_dir), bool(args.latent_tools), True, True, True, 0.78, 32, 5.0, 0.78]
        recipes = {
            "Balanced": base,
            "Fast CLI": [144, 0.72, 0.93, True, False, False, False, True, False, True, 0.14, 0.74, 0.67, 3, 256, 96, 1, 1, True, 0, 1, bool(args.use_latent_memory and args.latent_memory_dir), bool(args.latent_tools), True, True, True, 0.80, 24, 5.5, 0.80],
            "Deep aux_score": [256, 0.68, 0.92, True, True, False, False, True, False, True, 0.24, 0.64, 0.58, 1, 768, 224, 3, 2, True, 2, 6, bool(args.use_latent_memory and args.latent_memory_dir), bool(args.latent_tools), True, False, True, 0.76, 48, 4.0, 0.68],
            "Creative draft": [384, 1.05, 0.97, True, False, False, False, True, False, True, 0.12, 0.82, 0.76, 3, 384, 120, 2, 1, True, 1, 4, bool(args.use_latent_memory and args.latent_memory_dir), bool(args.latent_tools), True, True, True, 0.74, 48, 6.0, 0.82],
            "Raw diagnostics": [128, 0.75, 0.94, True, True, True, True, True, True, True, 0.18, 0.72, 0.65, 2, 512, 220, 2, 2, True, 1, 4, bool(args.use_latent_memory and args.latent_memory_dir), bool(args.latent_tools), True, False, False, 0.78, 32, 5.0, 0.78],
        }
        return [gr.update(value=v) for v in recipes.get(name, base)]

    def _image_recipe_values(name: str):
        recipes = {
            "Balanced tokens": ["low quality, blurry, distorted", 16, 24, 1.0, 3.0, 0.35, 256],
            "Fast preview": ["low quality, blurry, distorted, cluttered", 12, 12, 0.95, 2.4, 0.20, 192],
            "Sharper composition": ["low quality, blurry, distorted, extra limbs, unreadable text", 20, 36, 0.85, 4.0, 0.50, 320],
            "Exploratory": ["low quality, blurry, distorted", 16, 28, 1.25, 2.2, 0.15, 384],
        }
        return [gr.update(value=v) for v in recipes.get(name, recipes["Balanced tokens"])]

    generated_image_refs: list[dict[str, Any]] = []

    def _file_path(obj: Any) -> str:
        if obj is None:
            return ""
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            return str(obj.get("path") or obj.get("name") or "")
        return str(getattr(obj, "path", None) or getattr(obj, "name", None) or obj)

    def _read_text_file(path: str, max_chars: int = 20000) -> str:
        try:
            data = Path(path).read_bytes()
            return data[:max_chars * 4].decode("utf-8", errors="replace")[:max_chars]
        except Exception:
            return ""

    def _image_data_uri(path: str) -> str:
        try:
            mime = mimetypes.guess_type(path)[0] or "image/png"
            data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
            return f"data:{mime};base64,{data}"
        except Exception:
            return ""

    def _estimate_file_tokens(path: str, kind: str, text: str = "") -> int:
        try:
            if kind == "text":
                return int(len(tok.encode(text, add_bos=False)))
            if kind == "image" and hasattr(img_tok, "encode_image"):
                ids_img, _ = img_tok.encode_image(path, add_special=True)  # type: ignore[attr-defined]
                return int(len(ids_img))
        except Exception:
            pass
        try:
            return max(1, int(Path(path).stat().st_size // 4))
        except Exception:
            return 0

    def _attachments_html(items: list[dict[str, Any]]) -> str:
        if not items:
            return "<div class='need-strip'><span class='need-muted'>No attached context.</span></div>"
        blocks = []
        for item in items[-16:]:
            name = html.escape(str(item.get("name", "file")))
            meta = html.escape(f"{item.get('kind','file')} · {item.get('tokens',0)} tok")
            thumb = ""
            if item.get("kind") == "image" and item.get("data_uri"):
                thumb = f"<img src='{item['data_uri']}' alt=''>"
            else:
                thumb = "<span style='width:2.3rem;height:2.3rem;display:inline-grid;place-items:center;border:1px solid #494449;border-radius:4px;color:#aaa397'>txt</span>"
            blocks.append(f"<div class='need-attach'>{thumb}<span><div class='need-attach-name'>{name}</div><div class='need-attach-meta'>{meta}</div></span></div>")
        return "<div class='need-strip'>" + "".join(blocks) + "</div>"

    def _add_files_to_attachments(files, current):
        items = list(current or [])
        if files is None:
            return items, _attachments_html(items), gr.update(value=None)
        if not isinstance(files, (list, tuple)):
            files = [files]
        seen_paths = {str(x.get("path", "")) for x in items}
        for f in files:
            path = _file_path(f)
            if not path or not Path(path).exists() or path in seen_paths:
                continue
            seen_paths.add(path)
            name = Path(path).name
            mime = mimetypes.guess_type(path)[0] or ""
            kind = "image" if mime.startswith("image/") or Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"} else "text"
            text = _read_text_file(path) if kind == "text" else ""
            data_uri = _image_data_uri(path) if kind == "image" else ""
            tokens = _estimate_file_tokens(path, kind, text)
            items.append({"id": str(uuid.uuid4()), "name": name, "path": path, "kind": kind, "mime": mime, "tokens": tokens, "text": text, "data_uri": data_uri})
        return items[-32:], _attachments_html(items[-32:]), gr.update(value=None)

    def _clear_attachments():
        return [], _attachments_html([])

    def _attachment_context(items: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for i, item in enumerate(items or [], start=1):
            kind = item.get("kind", "file")
            name = item.get("name", f"file_{i}")
            if kind == "text":
                body = str(item.get("text", ""))[:12000]
                parts.append(f"<attached_text index={i} name={name!r} tokens={item.get('tokens',0)}>\n{body}\n</attached_text>")
            elif kind == "image":
                parts.append(f"<attached_image index={i} name={name!r} tokens={item.get('tokens',0)} path={str(item.get('path',''))!r}>available as visual reference for image-token generation</attached_image>")
        for j, item in enumerate(generated_image_refs[-8:], start=1):
            parts.append(f"<previous_generated_image index={j} path={str(item.get('path',''))!r} prompt={str(item.get('prompt',''))[:500]!r}>available as visual reference for follow-up edits</previous_generated_image>")
        if not parts:
            return ""
        return "\n\n<attached_context>\n" + "\n".join(parts) + "\n</attached_context>"

    def _looks_like_image_request(message: str) -> bool:
        s = str(message or "").lower()
        return bool(re.search(r"\b(generate|create|draw|render|make|edit|modify|transform|upscale|redraw|paint|visuali[sz]e)\b.*\b(image|picture|photo|render|illustration|poster|logo|diagram|scene)\b", s))

    def _image_prompt_ids_from_context(prompt_text: str, items: list[dict[str, Any]]) -> torch.Tensor:
        ids = tok.encode(prompt_text, add_bos=True)
        for item in list(items or [])[-4:] + generated_image_refs[-4:]:
            path = str(item.get("path", ""))
            if not path or not Path(path).exists() or not hasattr(img_tok, "encode_image"):
                continue
            try:
                img_ids, _ = img_tok.encode_image(path, add_special=True)  # type: ignore[attr-defined]
                ids.extend(img_ids[: min(len(img_ids), 512)])
            except Exception:
                continue
        return torch.tensor([ids[-model.cfg.block_size:]], device=device)

    def _generate_inline_image(
        message: str, items: list[dict[str, Any]], grid_val: int = 16, steps_val: int = 24,
        temp_val: float = 1.0, top_k_val: int = 256, quality_val: float = 0.35, cfg_val: float = 3.0,
        size_val: int = 256, mask_schedule_val: str = "cosine", gumbel_noise_val: float = 0.0, min_keep_val: int = 1,
    ) -> str:
        prompt_text = (message + _attachment_context(items))[-12000:]
        ids = _image_prompt_ids_from_context(prompt_text, items)
        with need_optimization_mode():
            toks = model.generate_image_tokens(
                ids,
                grid=int(grid_val),
                steps=int(steps_val),
                temperature=float(temp_val),
                top_k=int(top_k_val),
                quality_guidance=float(quality_val),
                negative_prompt_ids=None,
                cfg_scale=float(cfg_val),
                mask_schedule=str(mask_schedule_val or "cosine"),
                gumbel_noise=float(gumbel_noise_val),
                min_keep=int(min_keep_val),
            )
        img = img_tok.decode_tokens(toks[0].tolist(), grid=int(grid_val), size=int(size_val))
        out_dir = Path("outputs/browser_images")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"need_image_{int(time.time()*1000)}.png"
        img.save(out_path)
        generated_image_refs.append({"path": str(out_path), "prompt": message, "tokens": int(toks.numel())})
        del generated_image_refs[:-16]
        token_preview = " ".join(str(int(x)) for x in toks[0].flatten()[:96].detach().cpu().tolist())
        return f"\n\n<image_tokens>\n{token_preview}{' ...' if int(toks.numel()) > 96 else ''}\n</image_tokens>\n\nGenerated image: [{out_path.name}](file={out_path})"

    def _stream_chunks(answer: str, mode: str):
        if mode == "stream individual characters":
            for ch in answer:
                yield ch
        elif mode == "stream tokens":
            for part in re.findall(r"\S+\s*|\n+", answer):
                yield part
        else:
            yield answer

    def _respond_stream(
        message, history, attachments, display_mode, inline_image_tool,
        inline_image_grid, inline_image_steps, inline_image_temp, inline_image_top_k, inline_image_quality, inline_image_cfg, inline_image_size,
        inline_image_mask_schedule, inline_image_gumbel_noise, inline_image_min_keep,
        *controls
    ):
        if not str(message or "").strip():
            yield history or [], ""
            return
        history = history or []
        attachments = attachments or []
        enriched = str(message) + _attachment_context(attachments)
        answer = chat(enriched, history, *controls)
        if bool(inline_image_tool) and _looks_like_image_request(str(message)):
            try:
                answer += _generate_inline_image(
                    str(message), attachments, int(inline_image_grid), int(inline_image_steps), float(inline_image_temp),
                    int(inline_image_top_k), float(inline_image_quality), float(inline_image_cfg), int(inline_image_size),
                    str(inline_image_mask_schedule or "cosine"), float(inline_image_gumbel_noise), int(inline_image_min_keep),
                )
            except Exception as exc:
                answer += f"\n\nImage generation failed: {exc}"
        next_history = history + [{"role": "user", "content": str(message)}, {"role": "assistant", "content": ""}]
        acc = ""
        for chunk in _stream_chunks(answer, str(display_mode or "full response")):
            acc += chunk
            next_history[-1]["content"] = acc
            yield next_history, ""
        next_history[-1]["content"] = answer
        yield next_history, ""

    try:
        need_theme = gr.themes.Base().set(
            body_background_fill="#050712",
            block_background_fill="rgba(8, 12, 26, 0.72)",
            block_border_color="rgba(126, 211, 255, 0.16)",
            button_primary_background_fill="#0fb7ff",
            button_primary_text_color="#ffffff",
        )
    except Exception:
        need_theme = None
    blocks_kwargs = {"title": "NEED Console", "css": _browser_css()}
    if need_theme is not None:
        blocks_kwargs["theme"] = need_theme

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.HTML(_status_html())
        gr.HTML(_ascii_html())
        attachments_state = gr.State([])
        with gr.Tabs():
            with gr.Tab("Text"):
                gr.HTML("""
                <div class="need-panel">
                  <div class="need-panel-line">Messages, attached text, attached images, latent tools, and optional image-token generation share one conversation context.</div>
                </div>
                """)
                chatbot = gr.Chatbot(label="Session", height=520, elem_classes=["need-chatbot"])
                system_prompt_box = gr.Textbox(
                    label="System prompt",
                    value=args.system_prompt,
                    lines=1,
                    max_lines=3,
                )
                with gr.Row():
                    file_uploader = gr.File(label="Upload text or image", file_count="multiple", file_types=["text", "image"])
                    clear_files_btn = gr.Button("Clear uploads")
                attachment_strip = gr.HTML(_attachments_html([]))
                with gr.Row():
                    display_mode = gr.Radio(["full response", "stream tokens", "stream individual characters"], value="stream tokens", label="Display generation mode")
                    inline_image_tool = gr.Checkbox(True, label="inline image-token tool")
                prompt_box = gr.Textbox(
                    label="Message",
                    placeholder="Type a prompt. Attachments above are prepended as ordered context before this message.",
                    lines=3,
                    max_lines=8,
                )
                with gr.Row(elem_classes=["need-send-row"]):
                    send_btn = gr.Button("Run", variant="primary")
                    clear_btn = gr.Button("Clear session")
                gr.Examples(
                    examples=[
                        "Summarize the attached files and answer with the highest-risk item first.",
                        "Generate an image of a compact industrial sensor on a matte table.",
                        "Edit the attached image into a darker product render while keeping the same subject.",
                    ],
                    inputs=prompt_box,
                )
                with gr.Accordion("Runtime recipes", open=True, elem_classes=["need-control-accordion"]):
                    control_recipe = gr.Radio(["Balanced", "Fast CLI", "Deep aux_score", "Creative draft", "Raw diagnostics"], value="Balanced", label="Control recipe")
                with gr.Accordion("Inline image-token controls", open=False, elem_classes=["need-control-accordion"]):
                    with gr.Row():
                        inline_image_grid = gr.Slider(4, 32, 16, step=4, label="image structure")
                        inline_image_steps = gr.Slider(1, 96, 24, step=1, label="image refinement")
                        inline_image_temp = gr.Slider(0.1, 2.0, 1.0, label="image variation")
                    with gr.Row():
                        inline_image_top_k = gr.Slider(1, 1024, 256, step=1, label="image code pool")
                        inline_image_quality = gr.Slider(-1.0, 1.0, 0.35, label="image quality pull")
                        inline_image_cfg = gr.Slider(0.0, 10.0, 3.0, label="image prompt pull")
                    with gr.Row():
                        inline_image_size = gr.Slider(64, 768, 256, step=64, label="image output size")
                        inline_image_mask_schedule = gr.Dropdown(["cosine", "linear", "quadratic"], value="cosine", label="inline mask schedule")
                    with gr.Row():
                        inline_image_gumbel_noise = gr.Slider(0.0, 2.0, 0.0, label="image texture variation")
                        inline_image_min_keep = gr.Slider(1, 128, 1, step=1, label="image floor")
                with gr.Accordion("Core generation", open=False, elem_classes=["need-control-accordion"]):
                    with gr.Row():
                        max_new = gr.Slider(1, 2048, 128, step=1, label="max new tokens")
                        min_new_tokens = gr.Slider(0, 512, 0, step=1, label="min new tokens")
                        temp_text = gr.Slider(0, 2.0, 0.8, label="temperature")
                    with gr.Row():
                        top_p_text = gr.Slider(0.05, 1.0, 0.95, label="sampling breadth")
                        top_k_text = gr.Slider(0, 1024, 50, step=1, label="choice pool")
                        typical_p_text = gr.Slider(0.05, 1.0, 1.0, label="typicality filter")
                    with gr.Row():
                        repetition_penalty = gr.Slider(0.5, 2.5, 1.0, label="repetition penalty")
                        no_repeat_ngram = gr.Slider(0, 12, 0, step=1, label="repeat window")
                        lookahead_blend = gr.Slider(0.0, 1.0, 0.0, label="lookahead guidance")
                    with gr.Row():
                        decode_mode = gr.Dropdown(["ar", "auto", "nonseq"], value=args.decode_mode, label="decoder")
                        nonseq_dynamic = gr.Checkbox(bool(args.nonseq_dynamic), label="DVSD dynamic slots")
                        nonseq_max_heads = gr.Slider(0, 16, int(args.nonseq_max_heads), step=1, label="DVSD max slots; 0=checkpoint")
                    with gr.Row():
                        nonseq_min_heads = gr.Slider(1, 8, int(args.nonseq_min_heads), step=1, label="DVSD min slots")
                        nonseq_refine_steps = gr.Slider(1, 8, int(args.nonseq_refine_steps), step=1, label="DVSD refine passes")
                        nonseq_refine_causal_blend = gr.Slider(0.0, 1.0, float(args.nonseq_refine_causal_blend), label="DVSD causal blend")
                    with gr.Row():
                        nonseq_refine_confidence_floor = gr.Slider(0.0, 1.0, float(args.nonseq_refine_confidence_floor), label="DVSD confidence floor")
                        nonseq_refine_temperature_decay = gr.Slider(0.05, 1.0, float(args.nonseq_refine_temperature_decay), label="DVSD temperature decay")
                        nonseq_refine_lock_schedule = gr.Dropdown(["cosine", "linear", "quadratic"], value=args.nonseq_refine_lock_schedule, label="DVSD lock schedule")
                    with gr.Row():
                        nonseq_refine_resample_locked = gr.Checkbox(bool(args.nonseq_refine_resample_locked), label="resample locked slots")
                        dvsd_router_enabled = gr.Checkbox(bool(args.dvsd_router_enabled), label="learned slot router")
                        dvsd_router_inference_mix = gr.Slider(0.0, 1.0, float(args.dvsd_router_inference_mix), label="router blend")
                        dvsd_router_min_confidence = gr.Slider(0.0, 1.0, float(args.dvsd_router_min_confidence), label="router min confidence")
                        dual = gr.Checkbox(True, label="dual-channel reasoning")
                        proactive = gr.Checkbox(False, label="proactive aux_score (non-core search)")
                        auto_output_mode = gr.Checkbox(True, label="auto output-mode classifier")
                with gr.Accordion("Latent pathway and memory", open=False, elem_classes=["need-control-accordion"]):
                    with gr.Row():
                        cond_scale = gr.Slider(0.0, 2.0, 0.18, label="latent influence")
                        vector_stride = gr.Slider(1, 16, 2, step=1, label="pathway sampling")
                        max_vectors = gr.Slider(16, 512, 64, step=16, label="pathway memory (linear cap)")
                    with gr.Row():
                        latent_search_depth = gr.Slider(0, 8, 0, step=1, label="search depth")
                        latent_search_branches = gr.Slider(1, 64, 1, step=1, label="latent candidates")
                    with gr.Row():
                        use_latent_memory_box = gr.Checkbox(bool(args.use_latent_memory and args.latent_memory_dir), label="latent-memory guidance")
                        latent_tools_box = gr.Checkbox(bool(args.latent_tools), label="runtime calculator/code tools")
                with gr.Accordion("Reasoning and summary shaping", open=False, elem_classes=["need-control-accordion"]):
                    with gr.Row():
                        thought_tokens = gr.Slider(16, 512, 160, step=8, label="reasoning/summary tokens")
                        max_summary_chars = gr.Slider(256, 12000, 2000, step=256, label="max summary chars")
                        raw_cot_history_chars = gr.Slider(1000, 40000, 12000, step=1000, label="reasoning context")
                    with gr.Row():
                        reasoning_tree_branches = gr.Slider(1, 16, 1, step=1, label="reasoning branches")
                        reasoning_tree_keep = gr.Slider(1, 8, 1, step=1, label="branches kept")
                    with gr.Row():
                        cot_temperature = gr.Slider(0, 1.5, 0.45, label="reasoning creativity")
                        cot_top_p = gr.Slider(0.05, 1.0, 0.92, label="reasoning breadth")
                        cot_top_k = gr.Slider(1, 512, 80, step=1, label="reasoning pool")
                    with gr.Row():
                        summary_temperature = gr.Slider(0, 1.5, 0.25, label="summary temperature")
                        summary_top_p = gr.Slider(0.05, 1.0, 0.90, label="summary breadth")
                        summary_top_k = gr.Slider(1, 512, 60, step=1, label="summary pool")
                with gr.Accordion("AuxScoreHead and search", open=False, elem_classes=["need-control-accordion"]):
                    with gr.Row():
                        aux_score_weight = gr.Slider(0.0, 2.0, 0.35, label="aux_score weight")
                        aux_score_top_k = gr.Slider(0, 256, 0, step=1, label="aux_score pool")
                        aux_score_candidate_pool = gr.Slider(1, 16, 1, step=1, label="aux_score candidate pool (1 = linear core)")
                    with gr.Row():
                        risk_threshold = gr.Slider(0.1, 1.0, 0.72, label="risk threshold")
                        contradiction_threshold = gr.Slider(0.1, 1.0, 0.65, label="contradiction threshold")
                    with gr.Row():
                        aux_score_backtrack_window = gr.Slider(1, 32, 3, step=1, label="backtrack window")
                        aux_score_max_backtracks = gr.Slider(0, 32, 4, step=1, label="max backtracks")
                with gr.Accordion("Latent tool runtime", open=False, elem_classes=["need-control-accordion"]):
                    with gr.Row():
                        latent_tool_max_calls = gr.Slider(0, 12, int(args.latent_tool_max_calls), step=1, label="max tool calls")
                        latent_tool_timeout_s = gr.Slider(0.1, 30.0, float(args.latent_tool_timeout_s), label="tool timeout seconds")
                    with gr.Row():
                        latent_tool_max_code_chars = gr.Slider(256, 20000, int(args.latent_tool_max_code_chars), step=256, label="max code chars")
                        latent_tool_max_output_chars = gr.Slider(256, 20000, int(args.latent_tool_max_output_chars), step=256, label="tool output limit")
                with gr.Accordion("Sidecar speculative decoding", open=False, elem_classes=["need-control-accordion"]):
                    with gr.Row():
                        speculative_final = gr.Checkbox(True, label="speculative final decoding")
                        adaptive_spec = gr.Checkbox(True, label="adaptive acceptance")
                        target_accept_rate = gr.Slider(0.50, 0.95, 0.78, label="target acceptance")
                    with gr.Row():
                        spec_draft_tokens = gr.Slider(1, 256, 32, step=1, label="sidecar draft tokens")
                        spec_draft_temperature = gr.Slider(0, 2.0, 0.55, label="draft temperature")
                        spec_draft_top_p = gr.Slider(0.05, 1.0, 0.95, label="draft breadth")
                    with gr.Row():
                        spec_draft_top_k = gr.Slider(1, 1024, 80, step=1, label="draft pool")
                        spec_max_need_tokens_per_draft = gr.Slider(1, 512, 48, step=1, label="draft handoff size")
                        spec_context_chars = gr.Slider(512, 40000, 6000, step=512, label="draft context")
                    with gr.Row():
                        spec_accept_top_k = gr.Slider(1, 256, 20, step=1, label="acceptance pool")
                        spec_accept_min_need_prob = gr.Slider(0.0, 0.20, 0.015, label="acceptance confidence")
                        spec_accept_gap = gr.Slider(0.5, 20.0, 5.0, label="acceptance gap")
                    with gr.Row():
                        spec_risk_threshold = gr.Slider(0.1, 1.0, 0.78, label="spec risk threshold")
                        spec_contradiction_threshold = gr.Slider(0.1, 1.0, 0.72, label="spec contradiction threshold")
                        spec_repetition_threshold = gr.Slider(0.1, 1.0, 0.88, label="spec repetition threshold")
                with gr.Accordion("Adaptive speculative bounds", open=False, elem_classes=["need-control-accordion"]):
                    with gr.Row():
                        adaptive_accept_feedback_gain = gr.Slider(0.0, 2.0, 0.25, label="feedback gain")
                        adaptive_accept_aux_score_tighten = gr.Slider(0.0, 0.5, 0.08, label="aux_score tighten")
                    with gr.Row():
                        adaptive_accept_min_top_k = gr.Slider(1, 256, 6, step=1, label="min acceptance pool")
                        adaptive_accept_max_top_k = gr.Slider(1, 512, 80, step=1, label="max acceptance pool")
                    with gr.Row():
                        adaptive_accept_min_gap = gr.Slider(0.1, 20.0, 1.8, label="adaptive min gap")
                        adaptive_accept_max_gap = gr.Slider(0.1, 30.0, 9.0, label="adaptive max gap")
                    with gr.Row():
                        adaptive_accept_min_draft_tokens = gr.Slider(1, 256, 12, step=1, label="min draft size")
                        adaptive_accept_max_draft_tokens = gr.Slider(1, 512, 96, step=1, label="max draft size")
                    with gr.Row():
                        adaptive_accept_min_need_tokens = gr.Slider(1, 512, 32, step=1, label="min handoff size")
                        adaptive_accept_max_need_tokens = gr.Slider(1, 1024, 384, step=1, label="max handoff size")
                    with gr.Row():
                        adaptive_accept_min_min_prob = gr.Slider(0.0, 0.20, 0.003, label="min confidence")
                        adaptive_accept_max_min_prob = gr.Slider(0.0, 0.20, 0.030, label="max confidence")
                    with gr.Row():
                        adaptive_accept_min_risk_threshold = gr.Slider(0.1, 1.0, 0.58, label="adaptive min risk")
                        adaptive_accept_max_risk_threshold = gr.Slider(0.1, 1.0, 0.90, label="adaptive max risk")
                    with gr.Row():
                        adaptive_accept_min_contradiction_threshold = gr.Slider(0.1, 1.0, 0.52, label="adaptive min contradiction")
                        adaptive_accept_max_contradiction_threshold = gr.Slider(0.1, 1.0, 0.86, label="adaptive max contradiction")
                    with gr.Row():
                        adaptive_accept_min_repetition_threshold = gr.Slider(0.1, 1.0, 0.72, label="adaptive min repetition")
                        adaptive_accept_max_repetition_threshold = gr.Slider(0.1, 1.0, 0.94, label="adaptive max repetition")
                with gr.Accordion("Reasoning display", open=False, elem_classes=["need-control-accordion"]):
                    with gr.Row():
                        show_summary = gr.Checkbox(True, label="show public reasoning summary")
                        show_raw = gr.Checkbox(False, label="show raw artificial reasoning")
                        include_raw_cot_context = gr.Checkbox(False, label="include previous artificial reasoning in context")
                    with gr.Row():
                        store_raw_cot = gr.Checkbox(True, label="store artificial reasoning for this session")
                        show_cache_plan = gr.Checkbox(False, label="show sidecar cache plan")
                        show_dashboard = gr.Checkbox(True, label="show performance dashboard")
                inline_image_controls = [
                    inline_image_grid, inline_image_steps, inline_image_temp, inline_image_top_k, inline_image_quality, inline_image_cfg, inline_image_size,
                    inline_image_mask_schedule, inline_image_gumbel_noise, inline_image_min_keep,
                ]
                control_recipe_controls = [
                    max_new, temp_text, top_p_text, dual, show_summary, show_raw,
                    include_raw_cot_context, store_raw_cot, show_cache_plan, proactive,
                    cond_scale, risk_threshold, contradiction_threshold, vector_stride,
                    max_vectors, thought_tokens, reasoning_tree_branches, reasoning_tree_keep,
                    auto_output_mode, latent_search_depth, latent_search_branches,
                    use_latent_memory_box, latent_tools_box, show_dashboard, speculative_final, adaptive_spec,
                    target_accept_rate, spec_draft_tokens, spec_accept_gap, spec_risk_threshold,
                ]
                text_controls = [
                    max_new, temp_text, top_p_text, top_k_text, typical_p_text, repetition_penalty, no_repeat_ngram, min_new_tokens, lookahead_blend,
                    decode_mode, nonseq_dynamic, nonseq_min_heads, nonseq_max_heads, nonseq_refine_steps, nonseq_refine_causal_blend, nonseq_refine_confidence_floor,
                    nonseq_refine_temperature_decay, nonseq_refine_lock_schedule, nonseq_refine_resample_locked, dvsd_router_enabled, dvsd_router_inference_mix, dvsd_router_min_confidence,
                    dual, show_summary, show_raw, include_raw_cot_context, store_raw_cot, show_cache_plan, proactive,
                    aux_score_weight, aux_score_top_k, risk_threshold, contradiction_threshold, aux_score_candidate_pool, aux_score_backtrack_window, aux_score_max_backtracks,
                    cond_scale, vector_stride, max_vectors, thought_tokens, max_summary_chars, raw_cot_history_chars,
                    reasoning_tree_branches, reasoning_tree_keep, auto_output_mode, cot_temperature, cot_top_p, cot_top_k, summary_temperature, summary_top_p, summary_top_k,
                    latent_search_depth, latent_search_branches, use_latent_memory_box, latent_tools_box,
                    latent_tool_max_calls, latent_tool_timeout_s, latent_tool_max_code_chars, latent_tool_max_output_chars,
                    show_dashboard, speculative_final, adaptive_spec, target_accept_rate, spec_draft_tokens, spec_draft_temperature, spec_draft_top_p, spec_draft_top_k,
                    spec_max_need_tokens_per_draft, spec_accept_top_k, spec_accept_min_need_prob, spec_accept_gap, spec_risk_threshold, spec_contradiction_threshold, spec_repetition_threshold, spec_context_chars,
                    adaptive_accept_feedback_gain, adaptive_accept_aux_score_tighten, adaptive_accept_min_top_k, adaptive_accept_max_top_k, adaptive_accept_min_gap, adaptive_accept_max_gap,
                    adaptive_accept_min_draft_tokens, adaptive_accept_max_draft_tokens, adaptive_accept_min_need_tokens, adaptive_accept_max_need_tokens, adaptive_accept_min_min_prob, adaptive_accept_max_min_prob,
                    adaptive_accept_min_risk_threshold, adaptive_accept_max_risk_threshold, adaptive_accept_min_contradiction_threshold, adaptive_accept_max_contradiction_threshold,
                    adaptive_accept_min_repetition_threshold, adaptive_accept_max_repetition_threshold,
                ]
                control_recipe.change(_text_recipe_values, control_recipe, control_recipe_controls)
                file_uploader.change(_add_files_to_attachments, [file_uploader, attachments_state], [attachments_state, attachment_strip, file_uploader])
                clear_files_btn.click(_clear_attachments, outputs=[attachments_state, attachment_strip])
                send_inputs = [prompt_box, chatbot, attachments_state, display_mode, inline_image_tool] + inline_image_controls + [system_prompt_box] + text_controls
                send_btn.click(_respond_stream, send_inputs, [chatbot, prompt_box])
                prompt_box.submit(_respond_stream, send_inputs, [chatbot, prompt_box])
                clear_btn.click(lambda: [], outputs=chatbot)

            with gr.Tab("Compare"):
                gr.HTML("""
                <div class="need-panel"><div class="need-panel-line">Run one prompt against the loaded checkpoint and a second local checkpoint.</div></div>
                """)
                compare_checkpoint_box = gr.Textbox(label="Checkpoint B", value=args.compare_checkpoint, placeholder="path/to/other_need_checkpoint")
                compare_system_prompt = gr.Textbox(label="System prompt", value=args.system_prompt, lines=1)
                compare_prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Prompt both checkpoints with the same instruction...")
                with gr.Row():
                    compare_max_new = gr.Slider(1, 512, 128, step=1, label="max new tokens")
                    compare_temp = gr.Slider(0, 1.5, 0.0, label="temperature")
                    compare_top_p = gr.Slider(0.1, 1.0, 0.95, label="sampling breadth")
                compare_btn = gr.Button("Compare", variant="primary")
                with gr.Row():
                    compare_a = gr.Textbox(label="Checkpoint A output", lines=16)
                    compare_b = gr.Textbox(label="Checkpoint B output", lines=16)
                compare_summary = gr.Markdown()
                gr.Examples(
                    examples=[
                        "Compute 17*23 + 41 and answer briefly.",
                        "Score this supplier risk from 0-10 and explain in one sentence.",
                        "Rewrite this image prompt without adding extra objects: one forklift at a quiet dock.",
                    ],
                    inputs=compare_prompt,
                )
                compare_btn.click(compare_checkpoints, [compare_checkpoint_box, compare_prompt, compare_system_prompt, compare_max_new, compare_temp, compare_top_p], [compare_a, compare_b, compare_summary])

            with gr.Tab("Image tokens"):
                gr.HTML("""
                <div class="need-panel"><div class="need-panel-line">Decode image tokens locally. The Text tab can call this path inline when the prompt asks for an image.</div></div>
                """)
                with gr.Row():
                    image_recipe = gr.Dropdown(["Balanced tokens", "Fast preview", "Sharper composition", "Exploratory"], value="Balanced tokens", label="Image recipe")
                prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Describe the image to synthesize with local image tokens...")
                neg = gr.Textbox(label="Negative prompt", value="low quality, blurry, distorted")
                gr.Examples(
                    examples=[
                        "A clean product-style render of a compact industrial sensor on a matte table, controlled lighting, no logo.",
                        "A calm warehouse aisle at night with clear floor markings and soft overhead lighting.",
                        "A minimal concept panel for a local AI interface, dark theme, subtle cyan accents.",
                    ],
                    inputs=prompt,
                )
                with gr.Accordion("Image controls", open=True, elem_classes=["need-control-accordion"]):
                    with gr.Row():
                        grid = gr.Slider(4, 32, 16, step=4, label="image token grid")
                        steps = gr.Slider(1, 96, 24, step=1, label="diffusion steps")
                        temp_img = gr.Slider(0.1, 2.0, 1.0, label="temperature")
                    with gr.Row():
                        cfg = gr.Slider(0.0, 10.0, 3.0, label="prompt pull")
                        q = gr.Slider(-1, 1, 0.35, label="quality guidance")
                        topk = gr.Slider(1, 1024, 256, step=1, label="image code pool")
                    with gr.Row():
                        image_size = gr.Slider(64, 768, 256, step=64, label="image output size")
                        image_mask_schedule = gr.Dropdown(["cosine", "linear", "quadratic"], value="cosine", label="mask schedule")
                    with gr.Row():
                        image_gumbel_noise = gr.Slider(0.0, 2.0, 0.0, label="texture variation")
                        image_min_keep = gr.Slider(1, 128, 1, step=1, label="image floor")
                btn = gr.Button("Decode image tokens", variant="primary")
                out = gr.Image(label="Decoded image", height=420)
                image_recipe.change(_image_recipe_values, image_recipe, [neg, grid, steps, temp_img, cfg, q, topk])
                btn.click(image, [prompt, neg, grid, steps, temp_img, cfg, q, topk, image_size, image_mask_schedule, image_gumbel_noise, image_min_keep], out)
                gr.HTML("<div class='need-footnote'>The image tokenizer decodes local discrete image tokens.</div>")
    demo.queue(default_concurrency_limit=args.concurrent_requests).launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
