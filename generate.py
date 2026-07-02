#!/usr/bin/env python3
"""Generate text or learned-token images with NEED."""
from __future__ import annotations

import argparse
import json
import re
import time
from copy import copy
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from need_core import (
    ByteTokenizer,
    load_model,
    make_image_tokenizer,
    resolve_device,
    LatentMemoryStore,
    apply_repetition_penalty_,
    apply_no_repeat_ngram_,
)
from sidecar_lm_runtime import FastSidecarLMRuntime, SidecarLMRuntimeConfig, need_optimization_mode, sidecar_optimization_mode
from need_sidecar import make_single_sidecar_runtime
from need_latent_tools import DEFAULT_TOOL_SYSTEM_PROMPT, LatentToolConfig, LatentToolRuntime

try:
    from need_image import load_visual_tokenizer
except Exception:  # pragma: no cover
    load_visual_tokenizer = None  # type: ignore[assignment]

def _safe_float(x, default: float = 0.0) -> float:
    try:
        if torch.is_tensor(x):
            return float(x.detach().cpu())
        return float(x)
    except Exception:
        return default




def _safe_torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_json_dict(path: str) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"profile not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {p}")
    return data


def _apply_runtime_profile(args: argparse.Namespace) -> dict:
    """Apply low-data adapter/spec/control profiles to generation flags.

    Profiles may still tune aux_score, controller, and speculative-decoding knobs,
    but they no longer enable external episodic or latent-memory retrieval by
    supplying paths. Retrieval is an explicit opt-in at the command line.
    """
    paths = []
    for key in ("runtime_profile", "spec_profile", "control_profile"):
        val = getattr(args, key, "")
        if val:
            paths.append(val)
    applied = {}
    safe_keys = {
        "aux_score_risk_threshold", "aux_score_contradiction_threshold",
        "aux_score_weight", "aux_score_candidate_pool", "aux_score_backtrack_window", "aux_score_max_backtracks",
        "auto_output_mode_classifier", "reasoning_tree_branches", "reasoning_tree_merge_top_k",
        "latent_search_depth", "latent_search_branches",
        "adaptive_spec_acceptance", "adaptive_accept_target_rate", "adaptive_accept_feedback_gain",
        "adaptive_accept_aux_score_tighten", "adaptive_accept_min_top_k", "adaptive_accept_max_top_k",
        "adaptive_accept_min_gap", "adaptive_accept_max_gap", "adaptive_accept_min_draft_tokens",
        "adaptive_accept_max_draft_tokens", "adaptive_accept_min_need_tokens", "adaptive_accept_max_need_tokens",
        "adaptive_accept_min_min_prob", "adaptive_accept_max_min_prob",
        "adaptive_accept_min_risk_threshold", "adaptive_accept_max_risk_threshold",
        "adaptive_accept_min_contradiction_threshold", "adaptive_accept_max_contradiction_threshold",
        "adaptive_accept_min_repetition_threshold", "adaptive_accept_max_repetition_threshold",
        "spec_draft_tokens", "spec_max_need_tokens_per_draft", "spec_accept_top_k",
        "spec_accept_min_need_prob", "spec_accept_max_logprob_gap",
        "spec_risk_threshold", "spec_contradiction_threshold", "spec_repetition_threshold",
        "decode_mode", "nonseq_decode", "nonseq_dynamic", "nonseq_min_heads", "nonseq_max_heads",
        "nonseq_accept_top_k", "nonseq_accept_min_prob", "nonseq_accept_max_logprob_gap",
        "nonseq_risk_threshold", "nonseq_contradiction_threshold", "nonseq_repetition_threshold",
        "nonseq_entropy_easy", "nonseq_entropy_hard", "nonseq_min_draft_prob", "nonseq_max_head_entropy",
        "nonseq_tree_candidates", "nonseq_branch_top_k", "nonseq_aux_score_weight", "nonseq_fallback_to_ar",
        "nonseq_decode_style", "nonseq_refine_steps", "nonseq_refine_causal_blend",
        "nonseq_refine_confidence_floor", "nonseq_refine_temperature_decay",
        "nonseq_refine_lock_schedule", "nonseq_refine_resample_locked",
        "dvsd_router_enabled", "dvsd_router_inference_mix", "dvsd_router_min_confidence",
        "dvsd_router_loss_threshold", "dvsd_router_hard_loss_threshold",
        "sidecar_call_policy", "sidecar_gate_threshold", "sidecar_gate_metric",
        "sidecar_type", "need_sidecar_checkpoint", "need_sidecar_projection_path", "need_sidecar_projection_weight",
        "need_sidecar_decode_mode", "need_sidecar_prefer_best", "need_sidecar_max_context_tokens", "use_need_sidecar_latents",
        "sidecar_model", "sidecar_adapter_path", "sidecar_latent_alignment_path", "use_sidecar_latent_alignment", "sidecar_latent_alignment_weight",
        "latent_tools", "latent_tool_calculator", "latent_tool_python", "latent_tool_sidecar_planning", "latent_tool_router",
        "latent_tool_max_calls", "latent_tool_timeout_s", "latent_tool_max_code_chars", "latent_tool_max_output_chars",
    }
    for path in paths:
        profile = _load_json_dict(path)
        applied[str(path)] = {}
        nested = profile.get("runtime", profile)
        if not isinstance(nested, dict):
            continue
        for key, val in nested.items():
            if key in safe_keys and hasattr(args, key):
                setattr(args, key, val)
                applied[str(path)][key] = val
    return applied


def _apply_dvsd_runtime_overrides(model, args) -> dict:
    applied = {}
    mapping = {
        "dvsd_router_enabled": getattr(args, "dvsd_router_enabled", None),
        "dvsd_router_inference_mix": getattr(args, "dvsd_router_inference_mix", None),
        "dvsd_router_min_confidence": getattr(args, "dvsd_router_min_confidence", None),
        "dvsd_router_loss_threshold": getattr(args, "dvsd_router_loss_threshold", None),
        "dvsd_router_hard_loss_threshold": getattr(args, "dvsd_router_hard_loss_threshold", None),
    }
    for key, raw in mapping.items():
        if raw is None or not hasattr(model.cfg, key):
            continue
        value = bool(raw) if key == "dvsd_router_enabled" else float(raw)
        setattr(model.cfg, key, value)
        applied[key] = value
    # Report checkpoint compatibility state even when no override was applied.
    if hasattr(model, "_dvsd_router_loaded"):
        applied.setdefault("dvsd_router_loaded", bool(getattr(model, "_dvsd_router_loaded")))
    try:
        model.cfg.validate()
    except Exception:
        pass
    return applied


def _sidecar_should_run(args, metrics: dict) -> tuple[bool, dict]:
    policy = str(getattr(args, "sidecar_call_policy", "latent_gated") or "latent_gated").lower()
    if policy in {"always", "on"}:
        return True, {"sidecar_call_policy": policy, "sidecar_gate_used": True, "sidecar_gate_reason": "always"}
    if policy in {"off", "none", "never"}:
        return False, {"sidecar_call_policy": policy, "sidecar_gate_used": False, "sidecar_gate_reason": "disabled"}
    metric = str(getattr(args, "sidecar_gate_metric", "latent_difficulty") or "latent_difficulty")
    threshold = float(getattr(args, "sidecar_gate_threshold", 0.42))
    value = _safe_float(metrics.get(metric, metrics.get("latent_difficulty", 0.0)), 0.0)
    enabled = value >= threshold
    return enabled, {
        "sidecar_call_policy": policy,
        "sidecar_gate_metric": metric,
        "sidecar_gate_value": value,
        "sidecar_gate_threshold": threshold,
        "sidecar_gate_used": bool(enabled),
        "sidecar_gate_reason": "difficulty_above_threshold" if enabled else "difficulty_below_threshold",
    }


def _append_jsonl(path: str, row: dict) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _controller_runtime_adjust(model, tok: ByteTokenizer, prompt: str, args, device: torch.device) -> dict:
    """Use NEED's controller head as a runtime router.

    The controller predicts answer/deepen/retrieve/revise.  In auto mode this
    changes only safe generation knobs; it does not invent new data or bypass
    aux_score thresholds.
    """
    mode = str(getattr(args, "controller_runtime_policy", "observe") or "observe")
    if mode == "off":
        return {"enabled": False}
    try:
        ids = torch.tensor([tok.encode(prompt, add_bos=True)[-model.cfg.block_size:]], dtype=torch.long, device=device)
        scores = model.score_text_risk(ids)
    except Exception as exc:
        return {"enabled": mode == "auto", "error": str(exc)}
    names = ["answer", "deepen", "retrieve", "revise"]
    action = names[int(scores.get("controller_action", 0.0)) % len(names)]
    applied = {"enabled": mode == "auto", "mode": mode, "action": action, "scores": scores}
    if mode != "auto":
        return applied
    if action == "retrieve":
        # The controller's historical "retrieve" action is now interpreted as a
        # behavioral recall/fact-check posture. It may allocate more internal
        # reasoning and aux_score capacity, but it does not turn on external
        # episodic/latent-memory retrieval.
        args.latent_search_depth = max(int(args.latent_search_depth), 1)
        args.reasoning_tree_branches = max(int(args.reasoning_tree_branches), 2)
        args.aux_score_candidate_pool = max(int(args.aux_score_candidate_pool), 12)
        args.aux_score_weight = max(float(args.aux_score_weight), 0.40)
        applied["runtime_changes"] = {
            "behavioral_recall_only": True,
            "latent_search_depth": args.latent_search_depth,
            "reasoning_tree_branches": args.reasoning_tree_branches,
            "aux_score_candidate_pool": args.aux_score_candidate_pool,
            "aux_score_weight": args.aux_score_weight,
        }
    elif action == "deepen":
        args.latent_search_depth = max(int(args.latent_search_depth), 1)
        args.latent_search_branches = max(int(args.latent_search_branches), 2)
        args.reasoning_tree_branches = max(int(args.reasoning_tree_branches), 2)
        args.aux_score_candidate_pool = max(int(args.aux_score_candidate_pool), 12)
        applied["runtime_changes"] = {"latent_search_depth": args.latent_search_depth, "latent_search_branches": args.latent_search_branches, "reasoning_tree_branches": args.reasoning_tree_branches, "aux_score_candidate_pool": args.aux_score_candidate_pool}
    elif action == "revise":
        args.aux_score_weight = max(float(args.aux_score_weight), 0.50)
        args.aux_score_backtrack_window = max(int(args.aux_score_backtrack_window), 4)
        args.aux_score_max_backtracks = max(int(args.aux_score_max_backtracks), 6)
        args.spec_risk_threshold = min(float(args.spec_risk_threshold), float(args.aux_score_risk_threshold) + 0.02)
        applied["runtime_changes"] = {
            "aux_score_weight": args.aux_score_weight,
            "aux_score_backtrack_window": args.aux_score_backtrack_window,
            "aux_score_max_backtracks": args.aux_score_max_backtracks,
            "spec_risk_threshold": args.spec_risk_threshold,
        }
    else:
        # Direct answer: keep aux_score on but avoid unnecessary sidecar/reasoning fanout.
        args.latent_search_depth = min(int(args.latent_search_depth), 1)
        args.reasoning_tree_branches = min(int(args.reasoning_tree_branches), 1)
        applied["runtime_changes"] = {"latent_search_depth": args.latent_search_depth, "reasoning_tree_branches": args.reasoning_tree_branches}
    return applied


def _token_overlap_score(a: str, b: str) -> float:
    chars = ".,;:!?()[]{}\"'"
    aw = {w.lower().strip(chars) for w in a.split() if len(w) > 2}
    bw = {w.lower().strip(chars) for w in b.split() if len(w) > 2}
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / max(1, min(len(aw), len(bw)))

def _build_replay_context(model, tok: ByteTokenizer, prompt: str, args, device: torch.device) -> tuple[str, dict]:
    """Optionally derive behavioral guidance from high-quality replay episodes.

    Episodic retrieval is off by default. When explicitly enabled, retrieved
    examples are presented as response-pattern guidance rather than factual
    context, and no hidden memory state is mutated.
    """
    if (
        not getattr(args, "replay_dataset", "")
        or not bool(getattr(args, "replay_context", False))
        or getattr(args, "no_replay_context", False)
    ):
        return "", {}
    ds = Path(args.replay_dataset)
    ep_path = ds / "episodes.pt"
    if not ep_path.exists():
        return "", {"replay_context_warning": f"missing {ep_path}"}
    try:
        pack = _safe_torch_load(ep_path, map_location="cpu")
        episodes = list(pack.get("episodes", []))
    except Exception as exc:
        return "", {"replay_context_warning": f"failed to load replay episodes: {exc}"}
    if not episodes:
        return "", {"replay_context_items": 0}
    k = int(args.replay_context_k)
    if k <= 0:
        k = 3
    scored = []
    q = None
    if args.replay_context_similarity in {"latent", "hybrid"}:
        try:
            ids = torch.tensor([tok.encode(prompt, add_bos=True)[-model.cfg.block_size:]], dtype=torch.long, device=device)
            with torch.no_grad():
                q = model.latent_pathway(ids, stride=args.replay_context_vector_stride, max_vectors=args.replay_context_max_vectors)["pathway_vectors"].float().mean(dim=1).cpu()
        except Exception:
            q = None
    for ep in episodes:
        score = float(ep.get("score", 0.5))
        if score < float(args.replay_context_min_score):
            continue
        ep_prompt = str(ep.get("prompt", ""))
        sim_lat = 0.0
        v = ep.get("vectors")
        if q is not None and torch.is_tensor(v):
            try:
                key = v.float().mean(dim=1)
                sim_lat = float(F.cosine_similarity(q, key, dim=-1).mean().item())
            except Exception:
                sim_lat = 0.0
        sim_text = _token_overlap_score(prompt, ep_prompt)
        if args.replay_context_similarity == "latent":
            sim = sim_lat
        elif args.replay_context_similarity == "text":
            sim = sim_text
        else:
            sim = 0.65 * sim_lat + 0.35 * sim_text
        # Outcome score is included lightly so high-quality episodes are favored
        # without crowding out semantic similarity.
        total = sim + 0.20 * score
        scored.append((total, sim, score, ep))
    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = scored[:k]
    if not chosen:
        return "", {"replay_context_items": 0}
    max_chars = int(args.replay_context_chars)
    per = max(300, max_chars // max(1, len(chosen)))
    chunks = []
    meta = []
    for rank, (total, sim, score, ep) in enumerate(chosen, 1):
        ptxt = str(ep.get("prompt", ""))[: per // 2]
        summary = str(ep.get("summary", ""))
        ans = str(ep.get("answer", ""))
        body = summary if (args.replay_context_mode == "summary" and summary) else ans
        body = body[: per // 2]
        chunks.append(
            f"[behavioral_example_{rank} score={score:.2f} similarity={sim:.3f}]\n"
            f"Prior task shape: {ptxt}\nObserved successful behavior: {body}"
        )
        meta.append({"rank": rank, "score": score, "similarity": sim, "prompt_preview": ptxt[:120]})
    text = (
        "<behavioral_replay_guidance>\n"
        "Use these high-quality prior episodes only as task-handling and response-behavior guidance. "
        "Do not treat them as source facts, and do not copy their content.\n"
        + "\n\n---\n\n".join(chunks)
        + "\n</behavioral_replay_guidance>"
    )
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n</behavioral_replay_guidance>"
    return text, {"replay_context_items": len(chosen), "replay_context": meta}

def _latent_scaffold(pathway: dict) -> str:
    q = _safe_float(pathway.get("quality_mean", 0.0))
    r = _safe_float(pathway.get("risk_mean", 0.0))
    ge = _safe_float(pathway.get("geodesic", 0.0))
    effort = _safe_float(pathway.get("adaptive_effort", 0.0))
    mem_boundary = _safe_float(pathway.get("memory_boundary", 0.0))
    risk_signal = _safe_float(pathway.get("risk_signal_mean", pathway.get("risk_signal", 0.0)))
    vpol = pathway.get("output_mode_classifier", None)
    vmode = "unknown"
    try:
        names = ["none", "short_summary", "full_artificial_cot", "multi_cot", "renderer_only"]
        vmode = names[int(torch.argmax(vpol[0]).detach().cpu())] if torch.is_tensor(vpol) else "unknown"
    except Exception:
        pass
    return (
        f"<focus> Preserve the ordered latent reasoning path and answer the user request. </focus> "
        f"<latent_state> quality={q:.3f}; risk={r:.3f}; geodesic={ge:.3f}; "
        f"effort={effort:.3f}; memory_boundary={mem_boundary:.3f}; risk_signal={risk_signal:.3f}; "
        f"output_mode={vmode}. </latent_state> "
        f"<reasoning_chunks> Use the temporal vector pathway as nonverbal context; "
        f"check high-risk continuations before committing. </reasoning_chunks> "
        f"<answer_check> Prefer concise, grounded final output and avoid unsupported claims. </answer_check>"
    )


def _make_latent_summary(pathway: dict) -> str:
    return _latent_scaffold(pathway)


def _load_raw_cot_history(path: str, max_chars: int) -> str:
    if not path:
        return ""
    f = Path(path)
    if not f.exists():
        return ""
    txt = f.read_text(encoding="utf-8", errors="replace")
    return txt[-max_chars:]


def _append_raw_cot_history(path: str, prompt: str, raw_cot: str, max_chars: int) -> None:
    if not path or not str(raw_cot or "").strip():
        return
    f = Path(path)
    f.parent.mkdir(parents=True, exist_ok=True)
    prior = f.read_text(encoding="utf-8", errors="replace") if f.exists() else ""
    block = f"\n\n<message_cot>\n<prompt>{prompt[:1000]}</prompt>\n<raw_artificial_cot>\n{raw_cot}\n</raw_artificial_cot>\n</message_cot>"
    merged = (prior + block)[-max_chars:]
    f.write_text(merged, encoding="utf-8")


def _latent_tool_config_from_args(args) -> LatentToolConfig:
    return LatentToolConfig(
        enabled=bool(getattr(args, "latent_tools", True)),
        calculator=bool(getattr(args, "latent_tool_calculator", True)),
        python=bool(getattr(args, "latent_tool_python", False)),
        max_calls=int(getattr(args, "latent_tool_max_calls", 3)),
        timeout_s=float(getattr(args, "latent_tool_timeout_s", 3.0)),
        max_code_chars=int(getattr(args, "latent_tool_max_code_chars", 4000)),
        max_output_chars=int(getattr(args, "latent_tool_max_output_chars", 2400)),
        sidecar_planning=False,
    )


def _run_latent_tools(prompt: str, public_summary: str, raw_cot: str, sidecar_rt, args) -> tuple[str, dict]:
    """Execute calculator/Python tools as sealed latent evidence.

    Tool calls are built by deterministic runtime extractors only.  No NEED,
    sidecar, or external LLM is asked to construct a call, and no low-level RL is
    required for invocation.  The resulting observations are inserted only into
    hidden context for final decoding.
    """
    cfg = _latent_tool_config_from_args(args)
    runner = LatentToolRuntime(cfg)
    if not cfg.enabled:
        return "", {"latent_tools_enabled": False, "tool_calls": 0, "model_built_calls": False, "llrl_required": False}
    results, metrics = runner.run(prompt)
    hidden = runner.hidden_context(results)
    return hidden, metrics


def _strip_hidden_runtime_artifacts(text: str) -> str:
    # Defense-in-depth in case the decoder repeats hidden tags from context.
    patterns = [
        r"<latent_tool_results[^>]*>.*?</latent_tool_results>",
        r"<latent_tool_status[^>]*>.*?</latent_tool_status>",
        r"<latent_tool_observations[^>]*>.*?</latent_tool_observations>",
    ]
    out = str(text or "")
    for pat in patterns:
        out = re.sub(pat, "", out, flags=re.I | re.S)
    return out.strip()


def _make_sidecar_lm_runtime(args, device: torch.device):
    model_name = args.cot_model or args.summary_model or args.sidecar_model
    if not model_name:
        return None
    cfg = SidecarLMRuntimeConfig(
        model=model_name,
        device=str(device) if args.sidecar_device == "same" else args.sidecar_device,
        dtype=args.sidecar_dtype,
        attn_backend=args.sidecar_attn_backend,
        compile=args.sidecar_compile,
        max_batch=args.sidecar_max_batch,
        max_wait_ms=args.sidecar_max_wait_ms,
        cache_implementation=args.sidecar_cache_implementation,
        l2_cache_mb=args.gpu_l2_mb,
        max_context_tokens=args.sidecar_max_context_tokens,
        trust_remote_code=args.sidecar_trust_remote_code,
        adapter_path=getattr(args, "sidecar_adapter_path", ""),
        latent_alignment_path=getattr(args, "sidecar_latent_alignment_path", ""),
    )
    return FastSidecarLMRuntime(cfg).load()




def _make_committee_runtimes(args, device: torch.device):
    """Optional specialist sidecar committee.

    Empty by default.  When provided, each model drafts a candidate reasoning branch;
    NEED scores them and keeps only compatible branches.
    """
    models = [m.strip() for m in str(getattr(args, "sidecar_committee_models", "")).split(",") if m.strip()]
    runtimes = []
    for model_name in models:
        cfg = SidecarLMRuntimeConfig(
            model=model_name,
            device=str(device) if args.sidecar_device == "same" else args.sidecar_device,
            dtype=args.sidecar_dtype,
            attn_backend=args.sidecar_attn_backend,
            compile=args.sidecar_compile,
            max_batch=args.sidecar_max_batch,
            max_wait_ms=args.sidecar_max_wait_ms,
            cache_implementation=args.sidecar_cache_implementation,
            l2_cache_mb=args.gpu_l2_mb,
            max_context_tokens=args.sidecar_max_context_tokens,
            trust_remote_code=args.sidecar_trust_remote_code,
            adapter_path=getattr(args, "sidecar_adapter_path", ""),
            latent_alignment_path=getattr(args, "sidecar_latent_alignment_path", ""),
        )
        runtimes.append(FastSidecarLMRuntime(cfg).load())
    return runtimes


def _output_mode_from_model(model, base_ids: torch.Tensor) -> tuple[str, dict]:
    try:
        decision = model.output_mode_decision(base_ids)
    except Exception:
        return "full_artificial_cot", {}
    mode = max(decision, key=decision.get) if decision else "full_artificial_cot"
    return mode, decision


def _reasoning_tree_candidates(sidecar_rt, committee_rts, prompt: str, latent_summary: str, raw_cot_history: str, args) -> list[str]:
    """Generate multiple candidate reasoning branches before NEED validation."""
    branches = max(1, int(args.reasoning_tree_branches))
    prompts = []
    for i in range(branches):
        style = ["direct analysis", "counterexample check", "stepwise plan", "memory/facts check", "uncertainty audit"][i % 5]
        prompts.append(sidecar_rt.artificial_cot_prompt(prompt, latent_summary, raw_cot_history) + f"\nDraft branch style: {style}. Keep this branch compact.\n")
    out: list[str] = []
    if branches > 1 and hasattr(sidecar_rt, "generate_many"):
        out.extend(sidecar_rt.generate_many(prompts, max_new_tokens=args.max_raw_cot_tokens, temperature=args.cot_temperature, top_p=args.cot_top_p, top_k=80, stop=["\n\nFinal", "Tagged public summary:"]))
    else:
        out.append(sidecar_rt.generate(prompts[0], max_new_tokens=args.max_raw_cot_tokens, temperature=args.cot_temperature, top_p=args.cot_top_p, top_k=80, stop=["\n\nFinal", "Tagged public summary:"]))
    for rt in committee_rts:
        try:
            out.append(rt.generate(prompts[0], max_new_tokens=args.max_raw_cot_tokens, temperature=args.cot_temperature, top_p=args.cot_top_p, top_k=80, stop=["\n\nFinal", "Tagged public summary:"]))
        except Exception:
            pass
    cleaned = [x.strip()[: args.reasoning_tree_max_chars] for x in out if x and x.strip()]
    return cleaned or [latent_summary]


def _select_reasoning_branches(model, tok: ByteTokenizer, prompt: str, branches: list[str], vectors, args, device: torch.device) -> tuple[str, dict]:
    scored = []
    for branch in branches:
        score = _score_text_with_need(model, tok, prompt + "\n<reasoning_branch>\n" + branch + "\n</reasoning_branch>", vectors, args.conditioning_scale, device)
        quality = float(score.get("quality", 0.0))
        risk = float(score.get("risk", 1.0))
        contradiction = float(score.get("contradiction", 1.0))
        branch_score = quality - 0.75 * risk - 0.85 * contradiction
        scored.append((branch_score, branch, score))
    scored.sort(key=lambda x: x[0], reverse=True)
    keep = scored[: max(1, int(args.reasoning_tree_merge_top_k))]
    merged = "\n".join(f"<accepted_branch score={score:.3f}>\n{text}\n</accepted_branch>" for score, text, _ in keep)
    avg = {k: sum(float(d.get(k, 0.0)) for _, _, d in keep) / max(1, len(keep)) for k in ["quality", "risk", "contradiction", "repetition", "difficulty"]}
    avg["branches_considered"] = float(len(scored)); avg["branches_kept"] = float(len(keep)); avg["best_branch_score"] = float(keep[0][0]) if keep else 0.0
    return merged, avg

def _split_cot_chunks(text: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    cur = ""
    for part in text.replace("\r", "").split("\n"):
        part = part.strip()
        if not part:
            continue
        if len(cur) + len(part) + 1 > max_chars and cur:
            chunks.append(cur.strip()); cur = part
        else:
            cur = (cur + " " + part).strip()
    if cur:
        chunks.append(cur.strip())
    return chunks or ([text[:max_chars]] if text else [])


def _score_text_with_need(model, tok: ByteTokenizer, text: str, vectors, cond_scale: float, device: torch.device) -> dict:
    ids = torch.tensor([tok.encode(text, add_bos=True)[-model.cfg.block_size:]], dtype=torch.long, device=device)
    try:
        return model.score_text_risk(ids, conditioning_vectors=vectors, conditioning_scale=cond_scale)
    except Exception:
        return {"quality": 0.5, "risk": 0.5, "contradiction": 0.5, "repetition": 0.5, "difficulty": 0.5}


def _faithfulness_filter_cot(model, tok: ByteTokenizer, prompt: str, raw_cot: str, vectors, args, device: torch.device) -> tuple[str, dict]:
    """Accept only artificial-CoT chunks that NEED's aux_score considers faithful/useful.

    This implements latent speculative reasoning: the sidecar drafts reasoning chunks;
    NEED scores each chunk against the latent path before the chunk is allowed into the
    final answer context.
    """
    if not raw_cot or not args.latent_speculative_cot:
        score = _score_text_with_need(model, tok, prompt + "\n" + raw_cot, vectors, args.conditioning_scale, device)
        return raw_cot, score
    accepted: list[str] = []
    all_scores: list[dict] = []
    for chunk in _split_cot_chunks(raw_cot, args.speculative_chunk_chars):
        text = prompt + "\n<draft_reasoning_chunk>\n" + chunk + "\n</draft_reasoning_chunk>"
        score = _score_text_with_need(model, tok, text, vectors, args.conditioning_scale, device)
        all_scores.append(score)
        quality = float(score.get("quality", 0.0))
        risk = float(score.get("risk", 1.0))
        contradiction = float(score.get("contradiction", 1.0))
        faith = quality - 0.70 * risk - 0.70 * contradiction
        if faith >= args.speculative_accept_threshold:
            accepted.append(chunk)
    merged = "\n".join(accepted) if accepted else raw_cot[: args.speculative_chunk_chars]
    avg = {k: sum(float(d.get(k, 0.0)) for d in all_scores) / max(1, len(all_scores)) for k in ["quality", "risk", "contradiction", "repetition", "difficulty"]}
    avg["accepted_chunks"] = float(len(accepted)); avg["total_chunks"] = float(len(all_scores))
    return merged, avg

def _make_dual_channel_context(model, tok: ByteTokenizer, prompt: str, args, device: torch.device, sidecar_rt=None):
    metrics = {"t0": time.perf_counter()}
    base_ids = torch.tensor([tok.encode(prompt, add_bos=True)[: model.cfg.block_size]], dtype=torch.long, device=device)
    with need_optimization_mode():
        pathway = model.latent_pathway(base_ids, stride=args.vector_stride, max_vectors=args.max_vectors)
    metrics["latent_extract_s"] = time.perf_counter() - metrics["t0"]
    vectors = pathway["pathway_vectors"]
    memory_write_vectors = vectors
    latent_metrics = _compute_latent_convergence_metrics(pathway, model)
    metrics["latent_convergence"] = latent_metrics
    latent_summary = _make_latent_summary(pathway)
    if sidecar_rt is not None:
        sidecar_enabled, sidecar_gate = _sidecar_should_run(args, latent_metrics)
        metrics.update(sidecar_gate)
        if not sidecar_enabled:
            sidecar_rt = None
    else:
        metrics.update({"sidecar_gate_used": False, "sidecar_gate_reason": "no_sidecar_runtime"})
    if sidecar_rt is not None and getattr(sidecar_rt, "source_type", "external_lm") == "need" and bool(getattr(args, "use_need_sidecar_latents", True)):
        try:
            sidecar_summary, sidecar_vectors, sidecar_metrics = sidecar_rt.latent_guidance(prompt, vector_stride=max(1, int(args.vector_stride)), max_vectors=max(1, int(args.max_vectors) // 2))
            latent_summary += "\n<need_sidecar_summary>" + str(sidecar_summary)[:1200] + "</need_sidecar_summary>"
            metrics.update({"need_sidecar_" + str(k): v for k, v in dict(sidecar_metrics).items()})
            if torch.is_tensor(sidecar_vectors) and sidecar_vectors.size(-1) == vectors.size(-1):
                vectors = torch.cat([vectors, sidecar_vectors.to(device=device, dtype=vectors.dtype)], dim=1)
                metrics["need_sidecar_latents_used"] = True
                metrics["need_sidecar_latent_vectors"] = float(sidecar_vectors.size(1))
            elif sidecar_vectors is not None:
                metrics["need_sidecar_latents_error"] = f"dim mismatch: sidecar={sidecar_vectors.size(-1)} need={vectors.size(-1)}"
        except Exception as exc:
            metrics["need_sidecar_latents_error"] = str(exc)
    if bool(getattr(args, "use_sidecar_latent_alignment", False)) and sidecar_rt is not None and getattr(sidecar_rt, "latent_projection", None) is not None:
        try:
            sidecar_vec = sidecar_rt.encode_latent_alignment([prompt], max_length=getattr(args, "sidecar_max_context_tokens", 2048))
            if sidecar_vec.size(-1) == vectors.size(-1):
                sidecar_vec = sidecar_vec.to(device=device, dtype=vectors.dtype) * float(getattr(args, "sidecar_latent_alignment_weight", 0.35))
                vectors = torch.cat([vectors, sidecar_vec], dim=1)
                latent_summary += "\n<sidecar_latent_alignment>Loaded trained sidecar latent projection and appended it as a behavioral latent anchor.</sidecar_latent_alignment>"
                metrics["sidecar_latent_alignment_used"] = True
                metrics["sidecar_latent_alignment_vectors"] = float(sidecar_vec.size(1))
            else:
                metrics["sidecar_latent_alignment_error"] = f"dim mismatch: sidecar={sidecar_vec.size(-1)} need={vectors.size(-1)}"
        except Exception as exc:
            metrics["sidecar_latent_alignment_error"] = str(exc)
    retrieved_text = ""
    retrieved_vectors = None
    mem_store = None
    if args.latent_memory_dir and (bool(getattr(args, "use_latent_memory", False)) or bool(getattr(args, "store_latent_memory", False))):
        mem_store = LatentMemoryStore(args.latent_memory_dir, dim=model.cfg.d_model, max_items=args.latent_memory_max_items)
    if mem_store is not None and bool(getattr(args, "use_latent_memory", False)):
        retrieved_text, retrieved_vectors = mem_store.retrieve(
            vectors,
            k=args.latent_memory_k,
            score_weight=float(args.latent_memory_score_weight),
            risk_weight=float(args.latent_memory_risk_weight),
            contradiction_weight=float(args.latent_memory_contradiction_weight),
            min_score=float(args.latent_memory_min_score),
        )
        if retrieved_vectors is not None:
            vectors = torch.cat([retrieved_vectors.to(device=device, dtype=vectors.dtype), vectors], dim=1)

    raw_cot_history = _load_raw_cot_history(args.raw_cot_memory_file, args.raw_cot_history_chars) if args.include_raw_cot_context else ""
    output_mode, output_policy = _output_mode_from_model(model, base_ids) if args.auto_output_mode_classifier else ("full_artificial_cot", {})
    metrics["output_mode"] = output_mode
    metrics["output_mode_classifier"] = output_policy
    raw_cot = ""
    cot_t0 = time.perf_counter()
    sidecar_allows_reasoning = bool(
        sidecar_rt is not None
        and getattr(sidecar_rt, "supports_reasoning_sidecar", getattr(sidecar_rt, "source_type", "") != "need")
    )
    use_sidecar_cot = bool(sidecar_rt is not None and args.sidecar_cot and sidecar_allows_reasoning and output_mode != "none" and output_mode != "renderer_only")
    if sidecar_rt is not None and getattr(sidecar_rt, "source_type", "") == "need":
        metrics["need_sidecar_reasoning_disabled"] = True
    if use_sidecar_cot:
        committee_rts = _make_committee_runtimes(args, device) if (getattr(args, "sidecar_committee_models", "") and getattr(sidecar_rt, "source_type", "external_lm") == "external_lm") else []
        with sidecar_optimization_mode():
            tree_summary = latent_summary + ("\n" + retrieved_text if retrieved_text else "")
            if args.reasoning_tree_branches > 1 or committee_rts or output_mode == "multi_cot":
                candidates = _reasoning_tree_candidates(sidecar_rt, committee_rts, prompt, tree_summary, raw_cot_history, args)
                raw_cot, tree_scores = _select_reasoning_branches(model, tok, prompt, candidates, vectors, args, device)
                sum_prompt = sidecar_rt.summary_prompt(prompt, raw_cot, tree_summary)
                public_summary = sidecar_rt.generate(sum_prompt, max_new_tokens=args.max_thought_tokens, temperature=0.25, top_p=0.9, top_k=60, stop=["\n\nUser", "\n\nRaw"])
                metrics.update({f"reasoning_tree_{k}": v for k, v in tree_scores.items()})
            else:
                cot_tokens = args.max_raw_cot_tokens if output_mode != "short_summary" else max(32, args.max_raw_cot_tokens // 3)
                raw_cot, public_summary = sidecar_rt.generate_artificial_cot_and_summary(
                    prompt,
                    tree_summary,
                    raw_cot_history=raw_cot_history,
                    cot_tokens=cot_tokens,
                    summary_tokens=args.max_thought_tokens,
                    temperature=args.cot_temperature,
                    top_p=args.cot_top_p,
                )
    else:
        if args.use_internal_reasoning_head:
            with torch.no_grad(), need_optimization_mode():
                ids_internal = model.internal_reasoning_summary(base_ids, max_tokens=args.max_thought_tokens)
            raw_cot = tok.decode(ids_internal[0].tolist())
            public_summary = raw_cot or latent_summary
        else:
            # No sidecar reasoning was produced.  For NEED sidecars this is the
            # default and required behavior: keep the latent summary as public
            # guidance, but do not relabel it as raw artificial CoT.
            raw_cot = "" if getattr(sidecar_rt, "source_type", "") == "need" else latent_summary
            public_summary = latent_summary

    metrics["cot_summary_s"] = time.perf_counter() - cot_t0
    if args.cot_faithfulness_gate or args.latent_speculative_cot:
        raw_cot, cot_scores = _faithfulness_filter_cot(model, tok, prompt, raw_cot, vectors, args, device)
        public_summary += (f"\n<faithfulness> quality={cot_scores.get('quality',0):.3f}; risk={cot_scores.get('risk',0):.3f}; "
                           f"contradiction={cot_scores.get('contradiction',0):.3f}; accepted_chunks={int(cot_scores.get('accepted_chunks',0))}/{int(cot_scores.get('total_chunks',0))}. </faithfulness>")
    if mem_store is not None and bool(getattr(args, "store_latent_memory", False)):
        mem_store.add(prompt, public_summary, memory_write_vectors.detach().cpu())

    if args.raw_cot_memory_file and args.store_raw_cot_history:
        _append_raw_cot_history(args.raw_cot_memory_file, prompt, raw_cot, args.raw_cot_history_chars)

    if len(public_summary) > args.max_summary_chars:
        public_summary = public_summary[: args.max_summary_chars].rstrip() + " ..."

    raw_context = ""
    if args.include_raw_cot_context:
        raw_context += raw_cot_history
        raw_context += "\n\n<current_raw_artificial_cot>\n" + raw_cot + "\n</current_raw_artificial_cot>"

    latent_tool_context, latent_tool_metrics = _run_latent_tools(prompt, public_summary, raw_cot, sidecar_rt, args)
    metrics["latent_tools"] = latent_tool_metrics

    # Final answer gets symbolic scaffold text plus ordered latent vectors.  Raw CoT
    # can be included explicitly only when the user enables that setting. Tool
    # outputs are inserted as internal latent evidence, never as public function-call
    # text.
    final_prompt = prompt
    if retrieved_text:
        final_prompt += "\n\n<behavioral_latent_memory_guidance>\n" + retrieved_text[-args.latent_memory_chars:] + "\n</behavioral_latent_memory_guidance>"
    if latent_tool_context:
        final_prompt += '\n\n<latent_tool_results visibility="internal">\n' + latent_tool_context + '\n</latent_tool_results>'
    final_prompt += "\n\n<public_thought_summary>\n" + public_summary + "\n</public_thought_summary>"
    if raw_context:
        final_prompt += "\n\n<raw_artificial_cot_context>\n" + raw_context[-args.raw_cot_history_chars:] + "\n</raw_artificial_cot_context>"
    final_prompt += "\n\nFinal answer:"
    final_ids = torch.tensor([tok.encode(final_prompt, add_bos=True)[-model.cfg.block_size:]], dtype=torch.long, device=device)
    metrics["total_reasoning_prep_s"] = time.perf_counter() - metrics["t0"]
    return final_ids, vectors, public_summary, raw_cot, getattr(sidecar_rt, 'cache_plan', {}) if sidecar_rt is not None else {}, metrics



def _extract_final_answer_text(text: str) -> str:
    marker = "Final answer:"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text.strip()


def _final_answer_draft_prompt(context_text: str, max_context_chars: int) -> str:
    """Prompt the fast sidecar to draft only the next final-answer span.

    The sidecar may use the public summary and optional raw artificial CoT already
    embedded in context_text, but it is not trusted: NEED validates the resulting
    bytes before committing them.
    """
    ctx = context_text[-max(512, int(max_context_chars)):]
    answer_so_far = _extract_final_answer_text(ctx)
    return (
        "You are a fast draft model for a stronger NEED decoder. Continue only the final answer. "
        "Do not restart the response, do not add tags, and do not mention this instruction. "
        "Produce the next short continuation exactly as it should appear.\n\n"
        f"Context and partial answer:\n{ctx}\n\n"
        f"Partial final answer so far:\n{answer_so_far}\n\nNext continuation:"
    )


def _clean_draft_text(text: str) -> str:
    cleaned = text.replace("\r", "")
    for marker in ["Final answer:", "Next continuation:", "Context and partial answer:", "User:", "Assistant:"]:
        if cleaned.startswith(marker):
            cleaned = cleaned[len(marker):]
    # Stop at obvious prompt re-entry markers if the sidecar starts a new section.
    cuts = [len(cleaned)]
    for marker in ["\n\nUser:", "\n\nContext", "\n\nFinal answer:", "<thought_summary>", "<raw_artificial_cot>"]:
        pos = cleaned.find(marker)
        if pos >= 0:
            cuts.append(pos)
    cleaned = cleaned[:min(cuts)].strip("\n")
    return cleaned



def _clamp_float(x: float, lo: float, hi: float) -> float:
    try:
        x = float(x)
    except Exception:
        x = lo
    return max(lo, min(hi, x))


def _lerp(lo: float, hi: float, t: float) -> float:
    t = _clamp_float(t, 0.0, 1.0)
    return lo + (hi - lo) * t


def _tensor_stat(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if torch.is_tensor(x):
            if x.numel() == 0:
                return default
            return float(x.detach().float().mean().cpu())
        return float(x)
    except Exception:
        return default


def _compute_latent_convergence_metrics(pathway: dict, model) -> dict:
    """Summarize NEED's latent reasoning geometry for adaptive sidecar trust.

    Converged/low-risk latent paths delegate more language production to the fast
    sidecar. High-distance, high-curvature, high-residual, high-risk paths become
    stricter and rely more on NEED validation/repair.
    """
    vectors = pathway.get("pathway_vectors")
    metrics = {
        "k_descent_steps": int(getattr(model.cfg, "energy_steps", 0)),
        "energy_min_steps": int(getattr(model.cfg, "energy_min_steps", 0)),
        "quality_mean": _tensor_stat(pathway.get("quality_mean"), 0.5),
        "risk_mean": _tensor_stat(pathway.get("risk_mean"), 0.5),
        "equilibrium_residual": _tensor_stat(pathway.get("equilibrium_residual"), 0.0),
        "energy": _tensor_stat(pathway.get("energy"), 0.0),
        "adaptive_effort": _tensor_stat(pathway.get("adaptive_effort"), 0.0),
        "geodesic": _tensor_stat(pathway.get("geodesic"), 0.0),
        "path_straightness": _tensor_stat(pathway.get("path_straightness"), 0.0),
        "contradiction_mean": _tensor_stat(pathway.get("contradiction_mean"), 0.0),
        "cot_contradiction_mean": _tensor_stat(pathway.get("cot_contradiction_mean"), 0.0),
        "risk_signal_mean": _tensor_stat(pathway.get("risk_signal_mean"), 0.0),
        "latent_divergence_mean": _tensor_stat(pathway.get("latent_divergence_mean"), 0.0),
    }
    if torch.is_tensor(vectors) and vectors.size(1) >= 2:
        with torch.no_grad():
            v = vectors.detach().float()
            d = (v[:, 1:] - v[:, :-1]).norm(dim=-1)
            metrics["pathway_vectors"] = int(v.size(1))
            metrics["latent_step_median"] = float(d.median().cpu())
            metrics["latent_step_mean"] = float(d.mean().cpu())
            metrics["latent_step_p90"] = float(torch.quantile(d.flatten(), 0.90).cpu()) if d.numel() > 1 else float(d.mean().cpu())
            metrics["latent_path_length"] = float(d.sum(dim=1).mean().cpu())
            if v.size(1) >= 3:
                c = (v[:, 2:] - 2 * v[:, 1:-1] + v[:, :-2]).norm(dim=-1)
                metrics["latent_curvature_mean"] = float(c.mean().cpu())
                metrics["latent_curvature_p90"] = float(torch.quantile(c.flatten(), 0.90).cpu()) if c.numel() > 1 else float(c.mean().cpu())
            else:
                metrics["latent_curvature_mean"] = 0.0
                metrics["latent_curvature_p90"] = 0.0
    else:
        metrics.update({
            "pathway_vectors": 1,
            "latent_step_median": 0.0,
            "latent_step_mean": 0.0,
            "latent_step_p90": 0.0,
            "latent_path_length": 0.0,
            "latent_curvature_mean": 0.0,
            "latent_curvature_p90": 0.0,
        })
    def n01(x, ref):
        x = max(0.0, float(x))
        return x / (x + max(1e-8, float(ref)))
    contradiction = max(metrics["contradiction_mean"], metrics["cot_contradiction_mean"], metrics["latent_divergence_mean"])
    difficulty = (
        0.10 * n01(max(0, metrics["k_descent_steps"] - 1), 4.0)
        + 0.16 * n01(metrics["latent_step_median"], 0.20)
        + 0.08 * n01(metrics["latent_step_p90"], 0.35)
        + 0.13 * n01(metrics["latent_curvature_mean"], 0.15)
        + 0.11 * n01(metrics["equilibrium_residual"], 0.05)
        + 0.08 * n01(metrics["geodesic"], 0.12)
        + 0.08 * _clamp_float(metrics["adaptive_effort"], 0.0, 1.0)
        + 0.11 * _clamp_float(metrics["risk_mean"], 0.0, 1.0)
        + 0.08 * _clamp_float(metrics["risk_signal_mean"], 0.0, 1.0)
        + 0.07 * _clamp_float(contradiction, 0.0, 1.0)
    )
    quality = _clamp_float(metrics["quality_mean"], 0.0, 1.0)
    metrics["latent_difficulty"] = _clamp_float(difficulty - 0.08 * quality, 0.0, 1.0)
    metrics["latent_acceptance_base"] = _clamp_float(1.0 - metrics["latent_difficulty"], 0.0, 1.0)
    return metrics


def _build_adaptive_spec_config(args, metrics: dict, live_accept_rate: float | None = None, last_reject_reason: str = "none") -> dict:
    if not bool(getattr(args, "adaptive_spec_acceptance", False)):
        return {
            "enabled": False,
            "acceptance_score": float("nan"),
            "spec_draft_tokens": int(args.spec_draft_tokens),
            "spec_max_need_tokens_per_draft": int(args.spec_max_need_tokens_per_draft),
            "spec_accept_top_k": int(args.spec_accept_top_k),
            "spec_accept_min_need_prob": float(args.spec_accept_min_need_prob),
            "spec_accept_max_logprob_gap": float(args.spec_accept_max_logprob_gap),
            "spec_risk_threshold": float(args.spec_risk_threshold),
            "spec_contradiction_threshold": float(args.spec_contradiction_threshold),
            "spec_repetition_threshold": float(args.spec_repetition_threshold),
        }
    base = _clamp_float(float(metrics.get("latent_acceptance_base", 0.5)), 0.0, 1.0)
    target = _clamp_float(float(getattr(args, "adaptive_accept_target_rate", 0.78)), 0.05, 0.98)
    feedback = 0.0
    if live_accept_rate is not None:
        feedback = _clamp_float(target - float(live_accept_rate), -0.50, 0.50) * float(getattr(args, "adaptive_accept_feedback_gain", 0.25))
        if "aux_score" in str(last_reject_reason):
            feedback = min(feedback, 0.0) - float(getattr(args, "adaptive_accept_aux_score_tighten", 0.08))
    score = _clamp_float(base + feedback, 0.0, 1.0)
    top_k = round(_lerp(args.adaptive_accept_min_top_k, args.adaptive_accept_max_top_k, score))
    gap = _lerp(args.adaptive_accept_min_gap, args.adaptive_accept_max_gap, score)
    draft_tokens = round(_lerp(args.adaptive_accept_min_draft_tokens, args.adaptive_accept_max_draft_tokens, score))
    need_tokens = round(_lerp(args.adaptive_accept_min_need_tokens, args.adaptive_accept_max_need_tokens, score))
    min_prob = _lerp(args.adaptive_accept_max_min_prob, args.adaptive_accept_min_min_prob, score)
    risk_thr = _lerp(args.adaptive_accept_min_risk_threshold, args.adaptive_accept_max_risk_threshold, score)
    con_thr = _lerp(args.adaptive_accept_min_contradiction_threshold, args.adaptive_accept_max_contradiction_threshold, score)
    rep_thr = _lerp(args.adaptive_accept_min_repetition_threshold, args.adaptive_accept_max_repetition_threshold, score)
    return {
        "enabled": True,
        "acceptance_score": float(score),
        "base_acceptance_score": float(base),
        "feedback_delta": float(feedback),
        "target_accept_rate": float(target),
        "spec_draft_tokens": int(max(1, draft_tokens)),
        "spec_max_need_tokens_per_draft": int(max(1, need_tokens)),
        "spec_accept_top_k": int(max(1, top_k)),
        "spec_accept_min_need_prob": float(max(0.0, min_prob)),
        "spec_accept_max_logprob_gap": float(max(0.0, gap)),
        "spec_risk_threshold": float(_clamp_float(risk_thr, 0.01, 1.0)),
        "spec_contradiction_threshold": float(_clamp_float(con_thr, 0.01, 1.0)),
        "spec_repetition_threshold": float(_clamp_float(rep_thr, 0.01, 1.0)),
    }


def _apply_spec_config(args, cfg: dict):
    round_args = copy(args)
    for key, val in cfg.items():
        if key.startswith("spec_"):
            setattr(round_args, key, val)
    return round_args

def _need_fallback_one_token(model, idx: torch.Tensor, args, conditioning_vectors, conditioning_scale: float, generated_so_far: int) -> torch.Tensor:
    forbid_eos = generated_so_far < int(args.min_new_tokens)
    with need_optimization_mode():
        out = model.generate_text(
            idx,
            max_new_tokens=1,
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            typical_p=float(args.typical_p),
            repetition_penalty=float(args.repetition_penalty),
            no_repeat_ngram=int(args.no_repeat_ngram),
            min_new_tokens=1 if forbid_eos else 0,
            lookahead_blend=float(args.lookahead_blend),
            aux_score_top_k=int(args.aux_score_top_k),
            aux_score_weight=float(args.aux_score_weight),
            conditioning_vectors=conditioning_vectors,
            conditioning_scale=conditioning_scale,
            proactive_aux_score=not bool(args.disable_proactive_aux_score),
            aux_score_risk_threshold=float(args.aux_score_risk_threshold),
            aux_score_contradiction_threshold=float(args.aux_score_contradiction_threshold),
            aux_score_candidate_pool=int(args.aux_score_candidate_pool),
            aux_score_backtrack_window=int(args.aux_score_backtrack_window),
            aux_score_max_backtracks=int(args.aux_score_max_backtracks),
            latent_search_depth=int(args.latent_search_depth),
            latent_search_branches=int(args.latent_search_branches),
        )
    if out.size(1) <= idx.size(1):
        return torch.empty((idx.size(0), 0), dtype=idx.dtype, device=idx.device)
    return out[:, idx.size(1):]


def _validate_need_draft_tokens(
    model,
    idx: torch.Tensor,
    draft: torch.Tensor,
    args,
    conditioning_vectors,
    conditioning_scale: float,
    generated_so_far: int,
) -> tuple[int, dict]:
    """Validate a drafted NEED-token block in one target-model pass.

    SmolLM2 drafts text using its own tokenizer. We re-encode that text with NEED's
    byte tokenizer, then accept a prefix token-by-token under NEED's logits and
    aux_score state. This is cross-tokenizer speculative decoding: acceptance happens
    at NEED token granularity, so the final sequence remains a NEED sequence.
    """
    if draft.numel() == 0:
        return 0, {"reason": "empty_draft"}
    device = next(model.parameters()).device
    draft = draft.to(device=device, dtype=torch.long)
    if draft.size(0) != 1 or idx.size(0) != 1:
        return 0, {"reason": "batch_not_supported"}
    max_take = min(int(args.spec_max_need_tokens_per_draft), int(args.max_new_tokens) - int(generated_so_far), draft.size(1))
    if max_take <= 0:
        return 0, {"reason": "no_budget"}
    draft = draft[:, :max_take]
    full = torch.cat([idx, draft], dim=1)
    ctx = full[:, -model.cfg.block_size:]
    offset = full.size(1) - ctx.size(1)
    with need_optimization_mode(), torch.no_grad():
        logits, _, aux = model(ctx, return_hidden=True, conditioning_vectors=conditioning_vectors, conditioning_scale=conditioning_scale)
    hidden = aux.get("_hidden", None)
    accepted = 0
    stats = {
        "draft_need_tokens": int(draft.size(1)),
        "accepted_need_tokens": 0,
        "rejected_need_tokens": 0,
        "avg_need_prob": 0.0,
        "avg_logprob_gap": 0.0,
        "avg_risk": 0.0,
        "avg_contradiction": 0.0,
        "avg_repetition": 0.0,
        "reject_reason": "none",
    }
    probs_seen = []
    gaps_seen = []
    risk_seen = []
    con_seen = []
    rep_seen = []
    prefix = idx.clone()
    vocab_cut = int(model.cfg.image_token_offset)
    eos_id = int(model.cfg.eos_id)
    for j in range(draft.size(1)):
        cand = int(draft[0, j].item())
        if cand >= vocab_cut:
            stats["reject_reason"] = "non_text_token"
            break
        abs_pred = idx.size(1) + j - 1
        abs_tok = idx.size(1) + j
        rel_pred = abs_pred - offset
        rel_tok = abs_tok - offset
        if rel_pred < 0 or rel_pred >= logits.size(1):
            stats["reject_reason"] = "context_cropped"
            break
        next_logits = logits[:, rel_pred, :].float().clone()
        next_logits[:, vocab_cut:] = -float("inf")
        apply_repetition_penalty_(next_logits, prefix, float(args.repetition_penalty))
        if int(args.no_repeat_ngram) > 1:
            apply_no_repeat_ngram_(next_logits, prefix, int(args.no_repeat_ngram))
        if generated_so_far + j < int(args.min_new_tokens):
            next_logits[:, eos_id] = -float("inf")
        lp = F.log_softmax(next_logits, dim=-1)
        cand_lp = lp[0, cand]
        need_prob = float(cand_lp.exp().detach().cpu())
        k = min(max(1, int(args.spec_accept_top_k)), vocab_cut)
        top_vals, top_ids = torch.topk(lp[:, :vocab_cut], k=k, dim=-1)
        gap = float((top_vals[0, 0] - cand_lp).detach().cpu())
        in_topk = bool((top_ids[0] == cand).any().item())
        risk = contradiction = repetition = 0.0
        if hidden is not None and 0 <= rel_tok < hidden.size(1):
            vf = model.aux_score(hidden[:, rel_tok:rel_tok + 1])[:, 0]
            risk = float((F.softplus(vf[:, 1]).clamp(max=8.0) / 8.0).mean().detach().cpu())
            contradiction = float(torch.sigmoid(vf[:, 3]).mean().detach().cpu())
            repetition = float(torch.sigmoid(vf[:, 4]).mean().detach().cpu())
        ok_prob = in_topk or need_prob >= float(args.spec_accept_min_need_prob) or gap <= float(args.spec_accept_max_logprob_gap)
        ok_risk = risk <= float(args.spec_risk_threshold)
        ok_con = contradiction <= float(args.spec_contradiction_threshold)
        ok_rep = repetition <= float(args.spec_repetition_threshold)
        probs_seen.append(need_prob)
        gaps_seen.append(gap)
        risk_seen.append(risk)
        con_seen.append(contradiction)
        rep_seen.append(repetition)
        if ok_prob and ok_risk and ok_con and ok_rep:
            accepted += 1
            prefix = torch.cat([prefix, draft[:, j:j + 1]], dim=1)
            if cand == eos_id:
                break
        else:
            if not ok_prob:
                stats["reject_reason"] = "need_distribution_disagreement"
            elif not ok_risk:
                stats["reject_reason"] = "aux_score_risk"
            elif not ok_con:
                stats["reject_reason"] = "aux_score_contradiction"
            else:
                stats["reject_reason"] = "aux_score_repetition"
            break
    stats["accepted_need_tokens"] = int(accepted)
    stats["rejected_need_tokens"] = int(draft.size(1) - accepted)
    if probs_seen:
        stats["avg_need_prob"] = float(sum(probs_seen) / len(probs_seen))
        stats["avg_logprob_gap"] = float(sum(gaps_seen) / len(gaps_seen))
        stats["avg_risk"] = float(sum(risk_seen) / len(risk_seen))
        stats["avg_contradiction"] = float(sum(con_seen) / len(con_seen))
        stats["avg_repetition"] = float(sum(rep_seen) / len(rep_seen))
    return accepted, stats


def _speculative_final_decode(model, tok: ByteTokenizer, ids: torch.Tensor, sidecar_rt, args, conditioning_vectors, conditioning_scale: float, device: torch.device) -> tuple[torch.Tensor, dict]:
    """Final-answer speculative decoding controlled by NEED.

    The external LM sidecar drafts short answer continuations. NEED validates the
    re-encoded token block and commits only an accepted prefix. On rejection, NEED
    emits a replacement token before asking the draft model again.
    """
    idx = ids.to(device=device, dtype=torch.long)
    eos_id = int(model.cfg.eos_id)
    generated = 0
    latent_metrics = dict(getattr(args, "_latent_convergence_metrics", {}) or {})
    initial_cfg = _build_adaptive_spec_config(args, latent_metrics, None, "none")
    metrics = {
        "speculative_final_decode": True,
        "adaptive_spec_acceptance": bool(getattr(args, "adaptive_spec_acceptance", False)),
        "adaptive_spec_initial_config": initial_cfg,
        "latent_convergence": latent_metrics,
        "draft_calls": 0,
        "sidecar_draft_s": 0.0,
        "need_validate_s": 0.0,
        "need_fallback_s": 0.0,
        "draft_need_tokens": 0,
        "accepted_need_tokens": 0,
        "rejected_need_tokens": 0,
        "fallback_need_tokens": 0,
        "empty_drafts": 0,
        "last_reject_reason": "none",
        "adaptive_rounds": [],
    }
    while generated < int(args.max_new_tokens):
        live_rate = None
        if int(metrics["draft_need_tokens"]) > 0:
            live_rate = float(metrics["accepted_need_tokens"] / max(1, metrics["draft_need_tokens"]))
        round_cfg = _build_adaptive_spec_config(args, latent_metrics, live_rate, str(metrics.get("last_reject_reason", "none")))
        round_args = _apply_spec_config(args, round_cfg)
        if bool(getattr(args, "adaptive_spec_acceptance", False)):
            metrics["adaptive_rounds"].append({k: round_cfg[k] for k in ["acceptance_score", "spec_draft_tokens", "spec_max_need_tokens_per_draft", "spec_accept_top_k", "spec_accept_max_logprob_gap", "spec_risk_threshold"] if k in round_cfg})
        visible_context = tok.decode(idx[0].tolist())
        prompt = _final_answer_draft_prompt(visible_context, int(args.spec_context_chars))
        t0 = time.perf_counter()
        with sidecar_optimization_mode():
            draft_text = sidecar_rt.generate(
                prompt,
                max_new_tokens=int(round_args.spec_draft_tokens),
                temperature=float(args.spec_draft_temperature),
                top_p=float(args.spec_draft_top_p),
                top_k=int(args.spec_draft_top_k),
                stop=["\n\nUser:", "\n\nContext", "<thought_summary>", "<raw_artificial_cot>"],
            )
        metrics["sidecar_draft_s"] += time.perf_counter() - t0
        metrics["draft_calls"] += 1
        draft_text = _clean_draft_text(draft_text)
        draft_ids_list = tok.encode(draft_text, add_bos=False)
        if not draft_ids_list:
            metrics["empty_drafts"] += 1
            t0 = time.perf_counter()
            one = _need_fallback_one_token(model, idx, round_args, conditioning_vectors, conditioning_scale, generated)
            metrics["need_fallback_s"] += time.perf_counter() - t0
            if one.numel() == 0:
                break
            idx = torch.cat([idx, one[:, :1]], dim=1)
            metrics["fallback_need_tokens"] += 1
            generated += 1
            if int(one[0, 0].item()) == eos_id and generated >= int(args.min_new_tokens):
                break
            continue
        draft = torch.tensor([draft_ids_list], dtype=torch.long, device=device)
        t0 = time.perf_counter()
        accepted, vstats = _validate_need_draft_tokens(model, idx, draft, round_args, conditioning_vectors, conditioning_scale, generated)
        metrics["need_validate_s"] += time.perf_counter() - t0
        metrics["draft_need_tokens"] += int(vstats.get("draft_need_tokens", draft.size(1)))
        metrics["accepted_need_tokens"] += int(vstats.get("accepted_need_tokens", accepted))
        metrics["rejected_need_tokens"] += int(vstats.get("rejected_need_tokens", max(0, draft.size(1) - accepted)))
        metrics["last_reject_reason"] = str(vstats.get("reject_reason", "none"))
        if accepted > 0:
            take = min(accepted, int(args.max_new_tokens) - generated)
            accepted_tokens = draft[:, :take]
            idx = torch.cat([idx, accepted_tokens], dim=1)
            generated += int(take)
            if bool((accepted_tokens == eos_id).any()) and generated >= int(args.min_new_tokens):
                break
        if generated >= int(args.max_new_tokens):
            break
        # If the sidecar block was not fully accepted, let NEED repair the trajectory
        # with one aux_scored token before the next draft call.
        if accepted < min(draft.size(1), int(args.spec_max_need_tokens_per_draft)):
            t0 = time.perf_counter()
            one = _need_fallback_one_token(model, idx, round_args, conditioning_vectors, conditioning_scale, generated)
            metrics["need_fallback_s"] += time.perf_counter() - t0
            if one.numel() == 0:
                break
            idx = torch.cat([idx, one[:, :1]], dim=1)
            metrics["fallback_need_tokens"] += 1
            generated += 1
            if int(one[0, 0].item()) == eos_id and generated >= int(args.min_new_tokens):
                break
    denom = max(1, int(metrics["draft_need_tokens"]))
    metrics["spec_accept_rate"] = float(metrics["accepted_need_tokens"] / denom)
    return idx, metrics

def run_text(args: argparse.Namespace) -> None:
    applied_profiles = _apply_runtime_profile(args)
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device=device, prefer_best=args.prefer_best, kernel_backend=args.kernel_backend)
    dvsd_runtime_overrides = _apply_dvsd_runtime_overrides(model, args)
    tok = ByteTokenizer()
    prompt = args.prompt
    if args.prompt_file:
        data = Path(args.prompt_file).read_bytes()
        if len(data) > args.max_prompt_bytes:
            raise ValueError(f"prompt file exceeds --max_prompt_bytes ({args.max_prompt_bytes})")
        prompt = data.decode("utf-8", errors="replace")
    if str(getattr(args, "system_prompt", "") or "").strip():
        prompt = str(args.system_prompt).strip() + "\n\nUser: " + prompt.strip()
    controller_metrics = _controller_runtime_adjust(model, tok, prompt, args, device)
    replay_guidance, replay_metrics = _build_replay_context(model, tok, prompt, args, device)
    if replay_guidance:
        prompt = replay_guidance + "\n\n<current_user_prompt>\n" + prompt + "\n</current_user_prompt>"
    sidecar_rt = make_single_sidecar_runtime(args, device, model) if args.dual_channel_reasoning else None
    if args.dual_channel_reasoning:
        ids, vectors, summary, raw_cot, cache_plan, prep_metrics = _make_dual_channel_context(model, tok, prompt, args, device, sidecar_rt)
        if getattr(args, "_sidecar_selection", None):
            prep_metrics["active_sidecar"] = getattr(args, "_sidecar_selection")
        if replay_metrics:
            prep_metrics.update(replay_metrics)
        if dvsd_runtime_overrides:
            prep_metrics["dvsd_runtime_overrides"] = dvsd_runtime_overrides
        if controller_metrics:
            prep_metrics["controller_runtime"] = controller_metrics
        setattr(args, "_latent_convergence_metrics", prep_metrics.get("latent_convergence", {}))
        if args.print_sidecar_cache_plan and cache_plan:
            print("<sidecar_cache_plan>")
            print(json.dumps(cache_plan, indent=2))
            print("</sidecar_cache_plan>\n")
        if args.show_raw_cot:
            print("<raw_artificial_cot>")
            print(raw_cot)
            print("</raw_artificial_cot>\n")
        if not args.hide_thought_summary:
            print("<thought_summary>")
            print(summary)
            print("</thought_summary>\n")
        cond_vectors = vectors
        cond_scale = args.conditioning_scale
    else:
        prep_metrics = dict(replay_metrics) if replay_metrics else {}
        if dvsd_runtime_overrides:
            prep_metrics["dvsd_runtime_overrides"] = dvsd_runtime_overrides
        if controller_metrics:
            prep_metrics["controller_runtime"] = controller_metrics
        ids = torch.tensor([tok.encode(prompt, add_bos=True)], dtype=torch.long, device=device)
        cond_vectors = None
        cond_scale = 0.0
    gen_t0 = time.perf_counter()
    spec_metrics = {}
    nonseq_metrics = {}
    if args.speculative_final_decode and sidecar_rt is not None and bool(getattr(sidecar_rt, "supports_speculative_final_decode", True)):
        out, spec_metrics = _speculative_final_decode(model, tok, ids, sidecar_rt, args, cond_vectors, cond_scale, device)
    else:
        mode = str(getattr(args, "decode_mode", "auto") or "auto")
        if getattr(args, "nonseq_decode", None) is not None:
            mode = "nonseq" if bool(args.nonseq_decode) else "ar"
        cfg_max_heads = int(getattr(model.cfg, "nonseq_max_heads", getattr(model.cfg, "n_predict_heads", 1)))
        requested_max_heads = int(args.nonseq_max_heads) if int(args.nonseq_max_heads) > 0 else cfg_max_heads
        use_nonseq = mode == "nonseq" or (mode == "auto" and int(getattr(model.cfg, "n_predict_heads", 1)) > 1 and requested_max_heads > 1)
        with need_optimization_mode():
            if use_nonseq:
                out, nonseq_metrics = model.generate_text_nonsequential(
                    ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    typical_p=args.typical_p,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram=args.no_repeat_ngram,
                    min_new_tokens=args.min_new_tokens,
                    lookahead_blend=args.lookahead_blend,
                    aux_score_top_k=args.aux_score_top_k,
                    aux_score_weight=args.aux_score_weight,
                    conditioning_vectors=cond_vectors,
                    conditioning_scale=cond_scale,
                    proactive_aux_score=not args.disable_proactive_aux_score,
                    aux_score_risk_threshold=args.aux_score_risk_threshold,
                    aux_score_contradiction_threshold=args.aux_score_contradiction_threshold,
                    aux_score_candidate_pool=args.aux_score_candidate_pool,
                    aux_score_backtrack_window=args.aux_score_backtrack_window,
                    aux_score_max_backtracks=args.aux_score_max_backtracks,
                    latent_search_depth=args.latent_search_depth,
                    latent_search_branches=args.latent_search_branches,
                    nonseq_min_heads=args.nonseq_min_heads,
                    nonseq_max_heads=None if int(args.nonseq_max_heads) <= 0 else args.nonseq_max_heads,
                    nonseq_dynamic=args.nonseq_dynamic,
                    nonseq_accept_top_k=args.nonseq_accept_top_k,
                    nonseq_accept_min_prob=args.nonseq_accept_min_prob,
                    nonseq_accept_max_logprob_gap=args.nonseq_accept_max_logprob_gap,
                    nonseq_risk_threshold=args.nonseq_risk_threshold,
                    nonseq_contradiction_threshold=args.nonseq_contradiction_threshold,
                    nonseq_repetition_threshold=args.nonseq_repetition_threshold,
                    nonseq_entropy_easy=args.nonseq_entropy_easy,
                    nonseq_entropy_hard=args.nonseq_entropy_hard,
                    nonseq_min_draft_prob=args.nonseq_min_draft_prob,
                    nonseq_max_head_entropy=args.nonseq_max_head_entropy,
                    nonseq_tree_candidates=args.nonseq_tree_candidates,
                    nonseq_branch_top_k=args.nonseq_branch_top_k,
                    nonseq_aux_score_weight=args.nonseq_aux_score_weight,
                    nonseq_fallback_to_ar=args.nonseq_fallback_to_ar,
                    nonseq_decode_style=args.nonseq_decode_style,
                    nonseq_refine_steps=args.nonseq_refine_steps,
                    nonseq_refine_causal_blend=args.nonseq_refine_causal_blend,
                    nonseq_refine_confidence_floor=args.nonseq_refine_confidence_floor,
                    nonseq_refine_temperature_decay=args.nonseq_refine_temperature_decay,
                    nonseq_refine_lock_schedule=args.nonseq_refine_lock_schedule,
                    nonseq_refine_resample_locked=args.nonseq_refine_resample_locked,
                    return_stats=True,
                )
            else:
                out = model.generate_text(
                    ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    typical_p=args.typical_p,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram=args.no_repeat_ngram,
                    min_new_tokens=args.min_new_tokens,
                    lookahead_blend=args.lookahead_blend,
                    aux_score_top_k=args.aux_score_top_k,
                    aux_score_weight=args.aux_score_weight,
                    conditioning_vectors=cond_vectors,
                    conditioning_scale=cond_scale,
                    proactive_aux_score=not args.disable_proactive_aux_score,
                    aux_score_risk_threshold=args.aux_score_risk_threshold,
                    aux_score_contradiction_threshold=args.aux_score_contradiction_threshold,
                    aux_score_candidate_pool=args.aux_score_candidate_pool,
                    aux_score_backtrack_window=args.aux_score_backtrack_window,
                    aux_score_max_backtracks=args.aux_score_max_backtracks,
                    latent_search_depth=args.latent_search_depth,
                    latent_search_branches=args.latent_search_branches,
                )
    decode_s = time.perf_counter() - gen_t0
    text = tok.decode(out[0].tolist())
    if args.dual_channel_reasoning:
        text = _extract_final_answer_text(text)
    text = _strip_hidden_runtime_artifacts(text)
    if args.performance_dashboard:
        new_tokens = max(0, int(out.size(1) - ids.size(1)))
        dash = dict(prep_metrics)
        if applied_profiles:
            dash["runtime_profiles"] = applied_profiles
        dash.update({"need_decode_s": decode_s, "need_tokens_per_s": new_tokens / max(decode_s, 1e-9), "generated_tokens": new_tokens})
        if spec_metrics:
            dash.update(spec_metrics)
            if "adaptive_rounds" in dash and len(dash["adaptive_rounds"]) > 8:
                dash["adaptive_rounds"] = dash["adaptive_rounds"][-8:]
        if nonseq_metrics:
            dash.update(nonseq_metrics)
        print("<performance_dashboard>")
        print(json.dumps(dash, indent=2))
        print("</performance_dashboard>\n")
    if args.trace_out_jsonl:
        trace = {
            "prompt": prompt[:4000],
            "dual_channel_reasoning": bool(args.dual_channel_reasoning),
            "runtime_profiles": applied_profiles,
            "prep_metrics": prep_metrics,
            "spec_metrics": spec_metrics,
            "nonseq_metrics": nonseq_metrics,
            "generated_chars": len(text),
        }
        _append_jsonl(args.trace_out_jsonl, trace)
    if args.out_file:
        Path(args.out_file).write_text(text, encoding="utf-8")
    print(text)


def _load_image_tokenizer(args: argparse.Namespace, model):
    ckpt = Path(args.checkpoint)
    vt_path = Path(args.visual_tokenizer) if args.visual_tokenizer else ckpt
    if load_visual_tokenizer is not None and (vt_path / "visual_tokenizer_config.json").exists():
        return load_visual_tokenizer(vt_path, device=resolve_device(args.image_tokenizer_device)), "learned_vq"
    return make_image_tokenizer(model.cfg), "dynamic_fallback"


def run_image(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device=device, prefer_best=args.prefer_best, kernel_backend=args.kernel_backend)
    tok = ByteTokenizer()
    prompt_ids = torch.tensor([tok.encode(args.prompt, add_bos=True)], dtype=torch.long, device=device)
    neg_ids = None
    if args.negative_prompt:
        neg_ids = torch.tensor([tok.encode(args.negative_prompt, add_bos=True)], dtype=torch.long, device=device)
    grid = args.image_grid or model.cfg.image_grid
    image_tokens = model.generate_image_tokens(
        prompt_ids,
        grid=grid,
        steps=args.image_steps,
        temperature=args.image_temperature,
        top_k=args.image_top_k,
        quality_guidance=args.image_quality_guidance,
        negative_prompt_ids=neg_ids,
        cfg_scale=args.cfg_scale,
        mask_schedule=args.mask_schedule,
        gumbel_noise=args.gumbel_noise,
        min_keep=args.min_keep,
    )
    img_tok, tok_type = _load_image_tokenizer(args, model)
    img = img_tok.decode_tokens(image_tokens[0].tolist(), grid=grid, size=args.image_size)
    out = Path(args.image_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    meta = {
        "image_out": str(out),
        "grid": grid,
        "tokens": int(image_tokens.numel()),
        "image_tokenizer": tok_type,
        "steps": args.image_steps,
        "cfg_scale": args.cfg_scale,
        "spatial_coordinates": True,
    }
    print(json.dumps(meta, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NEED generation")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--runtime_profile", default="", help="JSON profile from need_low_data_adapters make_runtime_profile")
    p.add_argument("--system_prompt", default=DEFAULT_TOOL_SYSTEM_PROMPT, help="Short policy prompt prepended before the user message")
    p.add_argument("--spec_profile", default="", help="JSON profile from need_low_data_adapters calibrate_spec")
    p.add_argument("--control_profile", default="", help="JSON profile with aux_score/controller/output-mode runtime settings")
    p.add_argument("--controller_runtime_policy", choices=["off", "observe", "auto"], default="observe", help="Use the controller head to observe or automatically adjust retrieve/deepen/revise generation knobs")
    p.add_argument("--latent_tools", action=argparse.BooleanOptionalAction, default=True, help="Use calculator/Python tools as hidden latent evidence; no public function-call syntax is emitted")
    p.add_argument("--latent_tool_calculator", action=argparse.BooleanOptionalAction, default=True, help="Enable the latent calculator tool")
    p.add_argument("--latent_tool_python", action=argparse.BooleanOptionalAction, default=False, help="Enable sandboxed latent Python execution for compact code/numeric tasks")
    p.add_argument("--latent_tool_sidecar_planning", action=argparse.BooleanOptionalAction, default=False, help="Deprecated compatibility flag; latent tool calls are now runtime-built, not model- or sidecar-built")
    p.add_argument("--latent_tool_router", choices=["deterministic"], default="deterministic", help="Deterministic latent-tool router; no model builds tool calls and no LLRL is required")
    p.add_argument("--latent_tool_max_calls", type=int, default=3)
    p.add_argument("--latent_tool_timeout_s", type=float, default=3.0)
    p.add_argument("--latent_tool_max_code_chars", type=int, default=4000)
    p.add_argument("--latent_tool_max_output_chars", type=int, default=2400)
    p.add_argument("--latent_tool_plan_tokens", type=int, default=0, help="Deprecated compatibility option; no model-generated tool plan is requested")
    p.add_argument("--trace_out_jsonl", default="", help="Append generation/speculative traces for low-data calibration")
    p.add_argument("--prefer_best", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--kernel_backend", choices=["auto", "torch", "triton"], default="auto")
    p.add_argument("--mode", choices=["text", "image"], default="text")
    p.add_argument("--prompt", default="")
    p.add_argument("--prompt_file", default="")
    p.add_argument("--max_prompt_bytes", type=int, default=1_000_000)
    p.add_argument("--out_file", default="")
    p.add_argument("--dual_channel_reasoning", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--hide_thought_summary", action="store_true")
    p.add_argument("--vector_stride", type=int, default=2)
    p.add_argument("--max_vectors", type=int, default=512)
    p.add_argument("--max_thought_tokens", type=int, default=160)
    p.add_argument("--max_summary_chars", type=int, default=2000)
    p.add_argument("--conditioning_scale", type=float, default=0.18)
    p.add_argument("--latent_memory_dir", default="", help="Optional latent pathway memory directory; retrieval/storage stay off unless explicitly enabled")
    p.add_argument("--use_latent_memory", action=argparse.BooleanOptionalAction, default=False, help="Opt in to behavioral latent-memory retrieval; off by default")
    p.add_argument("--store_latent_memory", action=argparse.BooleanOptionalAction, default=False, help="Opt in to writing public summaries to latent memory; off by default")
    p.add_argument("--latent_memory_k", type=int, default=4)
    p.add_argument("--latent_memory_chars", type=int, default=4000)
    p.add_argument("--latent_memory_max_items", type=int, default=256)
    p.add_argument("--latent_memory_min_score", type=float, default=0.0)
    p.add_argument("--latent_memory_score_weight", type=float, default=0.18)
    p.add_argument("--latent_memory_risk_weight", type=float, default=0.22)
    p.add_argument("--latent_memory_contradiction_weight", type=float, default=0.25)
    p.add_argument("--replay_dataset", default="", help="Optional need_experience_replay dataset dir used only when --replay_context is enabled")
    p.add_argument("--replay_context", action=argparse.BooleanOptionalAction, default=False, help="Opt in to behavioral replay guidance; off by default")
    p.add_argument("--replay_context_k", type=int, default=0, help="Number of replay examples to use as behavioral guidance; 0 uses 3 when enabled")
    p.add_argument("--replay_context_chars", type=int, default=3500)
    p.add_argument("--replay_context_min_score", type=float, default=0.45)
    p.add_argument("--replay_context_mode", choices=["answer", "summary"], default="summary")
    p.add_argument("--replay_context_similarity", choices=["latent", "text", "hybrid"], default="hybrid")
    p.add_argument("--replay_context_vector_stride", type=int, default=8)
    p.add_argument("--replay_context_max_vectors", type=int, default=64)
    p.add_argument("--no_replay_context", action="store_true")
    p.add_argument("--cot_faithfulness_gate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--latent_speculative_cot", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--speculative_chunk_chars", type=int, default=700)
    p.add_argument("--speculative_accept_threshold", type=float, default=-0.15)
    p.add_argument("--auto_output_mode_classifier", action=argparse.BooleanOptionalAction, default=True, help="Let NEED choose no/short/full/multi reasoning scaffolding from its learned policy")
    p.add_argument("--reasoning_tree_branches", type=int, default=1, help="Draft multiple candidate reasoning branches and let NEED select/merge them")
    p.add_argument("--reasoning_tree_merge_top_k", type=int, default=1)
    p.add_argument("--reasoning_tree_max_chars", type=int, default=2500)
    p.add_argument("--sidecar_committee_models", default="", help="Deprecated external-LM-only committee branch models; ignored unless the active sidecar is an external LM")
    p.add_argument("--use_internal_reasoning_head", action="store_true")
    p.add_argument("--performance_dashboard", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--decode_mode", choices=["auto", "ar", "nonseq"], default="auto", help="auto uses dynamic virtual-slot nonsequential decoding when trained heads are available; ar keeps the legacy one-token generator")
    p.add_argument("--nonseq_decode", action=argparse.BooleanOptionalAction, default=None, help="Compatibility switch: --nonseq_decode forces internal dynamic MTP decoding; --no-nonseq_decode forces AR")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--typical_p", type=float, default=1.0)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--no_repeat_ngram", type=int, default=0)
    p.add_argument("--min_new_tokens", type=int, default=0)
    p.add_argument("--lookahead_blend", type=float, default=0.0)
    p.add_argument("--aux_score_top_k", type=int, default=0)
    p.add_argument("--aux_score_weight", type=float, default=0.35)
    p.add_argument("--disable_proactive_aux_score", action="store_true")
    p.add_argument("--aux_score_candidate_pool", type=int, default=8)
    p.add_argument("--aux_score_risk_threshold", type=float, default=0.72)
    p.add_argument("--aux_score_contradiction_threshold", type=float, default=0.65)
    p.add_argument("--aux_score_backtrack_window", type=int, default=3)
    p.add_argument("--aux_score_max_backtracks", type=int, default=4)
    p.add_argument("--latent_search_branches", type=int, default=1)
    p.add_argument("--latent_search_depth", type=int, default=0)
    p.add_argument("--nonseq_dynamic", action=argparse.BooleanOptionalAction, default=True, help="Adapt internal MTP head count from easy multi-token spans down to one-token AR fallback")
    p.add_argument("--nonseq_min_heads", type=int, default=1)
    p.add_argument("--nonseq_max_heads", type=int, default=0, help="Max internal prediction heads; 0 uses checkpoint/config default")
    p.add_argument("--nonseq_accept_top_k", type=int, default=20)
    p.add_argument("--nonseq_accept_min_prob", type=float, default=0.015)
    p.add_argument("--nonseq_accept_max_logprob_gap", type=float, default=5.0)
    p.add_argument("--nonseq_risk_threshold", type=float, default=0.78)
    p.add_argument("--nonseq_contradiction_threshold", type=float, default=0.72)
    p.add_argument("--nonseq_repetition_threshold", type=float, default=0.88)
    p.add_argument("--nonseq_entropy_easy", type=float, default=0.45)
    p.add_argument("--nonseq_entropy_hard", type=float, default=0.82)
    p.add_argument("--nonseq_min_draft_prob", type=float, default=0.010)
    p.add_argument("--nonseq_max_head_entropy", type=float, default=0.92)
    p.add_argument("--nonseq_tree_candidates", type=int, default=4, help="Legacy aux_scored-draft knob retained for compatibility; ignored by the default virtual-slot decoder")
    p.add_argument("--nonseq_branch_top_k", type=int, default=2)
    p.add_argument("--nonseq_aux_score_weight", type=float, default=0.10)
    p.add_argument("--nonseq_fallback_to_ar", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--nonseq_decode_style", choices=["slot_refine", "slots", "virtual_slots"], default="slot_refine", help="Compatibility knob; the canonical nonseq path commits a refined virtual canvas directly without aux_scored acceptance")
    p.add_argument("--nonseq_refine_steps", type=int, default=3, help="Max virtual-canvas refinement passes before direct commit")
    p.add_argument("--nonseq_refine_causal_blend", type=float, default=0.55, help="Blend MTP slot logits with provisional causal context during refinement")
    p.add_argument("--nonseq_refine_confidence_floor", type=float, default=0.0, help="Optional confidence floor for replacing an unlocked slot before the final pass")
    p.add_argument("--nonseq_refine_temperature_decay", type=float, default=0.82, help="Temperature multiplier per refinement pass")
    p.add_argument("--nonseq_refine_lock_schedule", choices=["cosine", "linear", "quadratic"], default="cosine", help="Confidence schedule used to lock virtual slots in non-left-to-right order")
    p.add_argument("--nonseq_refine_resample_locked", action=argparse.BooleanOptionalAction, default=False, help="Allow already-locked slots to be resampled on later refinement passes")
    p.add_argument("--dvsd_router_enabled", action=argparse.BooleanOptionalAction, default=None, help="Use the learned DVSD slot router when the checkpoint has it; default keeps checkpoint setting")
    p.add_argument("--dvsd_router_inference_mix", type=float, default=None, help="Blend of learned router vs heuristic slot budget; default keeps checkpoint setting")
    p.add_argument("--dvsd_router_min_confidence", type=float, default=None, help="Minimum learned-router confidence before it can steer slot count; default keeps checkpoint setting")
    p.add_argument("--dvsd_router_loss_threshold", type=float, default=None, help="Training threshold stored in runtime profile for router calibration; default keeps checkpoint setting")
    p.add_argument("--dvsd_router_hard_loss_threshold", type=float, default=None, help="Training hard-collapse threshold stored in runtime profile for calibration; default keeps checkpoint setting")
    # Single active sidecar. Exactly one backend is loaded: none, external_lm, or a smaller NEED checkpoint.
    p.add_argument("--sidecar_type", choices=["auto", "none", "external_lm", "need"], default="auto", help="Select the one active sidecar backend. auto prefers --need_sidecar_checkpoint, otherwise --sidecar_model")
    p.add_argument("--need_sidecar_checkpoint", default="", help="Optional smaller NEED checkpoint used as the only active sidecar")
    p.add_argument("--need_sidecar_prefer_best", action="store_true", help="Load best.safetensors/best.pt from the NEED sidecar checkpoint when present")
    p.add_argument("--need_sidecar_projection_path", default="", help="Projection file or directory from need_sidecar_distill.py mapping sidecar NEED latents into the main NEED latent space")
    p.add_argument("--need_sidecar_projection_weight", type=float, default=1.0, help="Scale applied to projected NEED-sidecar latent anchors")
    p.add_argument("--need_sidecar_decode_mode", choices=["nonseq", "ar"], default="nonseq", help="Decoder used by the smaller NEED sidecar when it drafts public summaries")
    p.add_argument("--need_sidecar_max_context_tokens", type=int, default=512, help="Max prompt tokens used by the smaller NEED sidecar")
    p.add_argument("--use_need_sidecar_latents", action=argparse.BooleanOptionalAction, default=True, help="Append projected latent anchors from a smaller NEED sidecar when active")
    p.add_argument("--sidecar_call_policy", choices=["always", "latent_gated", "off"], default="latent_gated", help="When to call the single active sidecar; latent_gated skips it on easy prompts")
    p.add_argument("--sidecar_gate_metric", default="latent_difficulty", help="Prep metric used by latent_gated sidecar policy")
    p.add_argument("--sidecar_gate_threshold", type=float, default=0.42, help="Minimum sidecar_gate_metric value required to call the sidecar")
    # Fast external LM sidecar for artificial CoT and summarization.
    p.add_argument("--sidecar_model", default="", help="Default external LM sidecar, e.g. HuggingFaceTB/SmolLM2-135M-Instruct")
    p.add_argument("--cot_model", default="", help="External LM used for artificial public reasoning-note generation; falls back to --sidecar_model")
    p.add_argument("--summary_model", default="", help="Reserved for a separate summary external LM; currently shares the loaded runtime")
    p.add_argument("--sidecar_cot", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sidecar_device", default="same", help="same|auto|cuda|cpu")
    p.add_argument("--sidecar_dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--sidecar_attn_backend", choices=["auto", "sdpa", "flash_attention_2", "eager"], default="sdpa")
    p.add_argument("--sidecar_compile", action="store_true")
    p.add_argument("--sidecar_max_batch", type=int, default=8)
    p.add_argument("--sidecar_max_wait_ms", type=int, default=8)
    p.add_argument("--sidecar_cache_implementation", choices=["static", "dynamic", "offloaded", "none"], default="static")
    p.add_argument("--sidecar_max_context_tokens", type=int, default=2048)
    p.add_argument("--sidecar_trust_remote_code", action="store_true")
    p.add_argument("--sidecar_adapter_path", default="", help="Optional trained sidecar LoRA/full adapter from need_thought_distill.py train_alignment")
    p.add_argument("--sidecar_latent_alignment_path", default="", help="Optional sidecar latent-projection directory containing latent_projection.pt")
    p.add_argument("--use_sidecar_latent_alignment", action=argparse.BooleanOptionalAction, default=False, help="Append trained sidecar-projected NEED latent vector as an extra behavioral conditioning anchor")
    p.add_argument("--sidecar_latent_alignment_weight", type=float, default=0.35)
    p.add_argument("--gpu_l2_mb", type=float, default=96.0)
    p.add_argument("--cot_temperature", type=float, default=0.45)
    p.add_argument("--cot_top_p", type=float, default=0.92)
    p.add_argument("--max_raw_cot_tokens", type=int, default=220)
    p.add_argument("--include_raw_cot_context", action="store_true", help="Include raw artificial CoT history in final answer context")
    p.add_argument("--show_raw_cot", action="store_true", help="Print raw artificial CoT; off by default")
    p.add_argument("--store_raw_cot_history", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--raw_cot_memory_file", default="")
    p.add_argument("--raw_cot_history_chars", type=int, default=12000)
    p.add_argument("--print_sidecar_cache_plan", action="store_true")
    p.add_argument("--speculative_final_decode", action=argparse.BooleanOptionalAction, default=True, help="Use sidecar-drafted final answer blocks validated by NEED; active only when an external LM sidecar is loaded")
    p.add_argument("--spec_draft_tokens", type=int, default=12, help="Sidecar tokens drafted per speculative final-answer step")
    p.add_argument("--spec_draft_temperature", type=float, default=0.55)
    p.add_argument("--spec_draft_top_p", type=float, default=0.95)
    p.add_argument("--spec_draft_top_k", type=int, default=80)
    p.add_argument("--spec_max_need_tokens_per_draft", type=int, default=48, help="Max NEED byte tokens validated from one sidecar draft")
    p.add_argument("--spec_accept_top_k", type=int, default=20, help="Accept if the draft token is within NEED's top-k")
    p.add_argument("--spec_accept_min_need_prob", type=float, default=0.015, help="Accept if NEED assigns at least this probability")
    p.add_argument("--spec_accept_max_logprob_gap", type=float, default=5.0, help="Accept if draft token is within this log-prob gap from NEED's top token")
    p.add_argument("--spec_risk_threshold", type=float, default=0.78)
    p.add_argument("--spec_contradiction_threshold", type=float, default=0.72)
    p.add_argument("--spec_repetition_threshold", type=float, default=0.88)
    p.add_argument("--spec_context_chars", type=int, default=6000, help="Characters of NEED context shown to sidecar while drafting final answer")
    p.add_argument("--adaptive_spec_acceptance", action=argparse.BooleanOptionalAction, default=True, help="Derive speculative acceptance strictness from latent convergence and live acceptance feedback")
    p.add_argument("--adaptive_accept_target_rate", type=float, default=0.78, help="Target accepted-token rate used by adaptive speculative acceptance")
    p.add_argument("--adaptive_accept_feedback_gain", type=float, default=0.25)
    p.add_argument("--adaptive_accept_aux_score_tighten", type=float, default=0.08)
    p.add_argument("--adaptive_accept_min_top_k", type=int, default=6)
    p.add_argument("--adaptive_accept_max_top_k", type=int, default=80)
    p.add_argument("--adaptive_accept_min_gap", type=float, default=1.8)
    p.add_argument("--adaptive_accept_max_gap", type=float, default=9.0)
    p.add_argument("--adaptive_accept_min_draft_tokens", type=int, default=12)
    p.add_argument("--adaptive_accept_max_draft_tokens", type=int, default=96)
    p.add_argument("--adaptive_accept_min_need_tokens", type=int, default=32)
    p.add_argument("--adaptive_accept_max_need_tokens", type=int, default=384)
    p.add_argument("--adaptive_accept_min_min_prob", type=float, default=0.003, help="Minimum NEED probability in most lenient mode")
    p.add_argument("--adaptive_accept_max_min_prob", type=float, default=0.030, help="Minimum NEED probability in strictest mode")
    p.add_argument("--adaptive_accept_min_risk_threshold", type=float, default=0.58)
    p.add_argument("--adaptive_accept_max_risk_threshold", type=float, default=0.90)
    p.add_argument("--adaptive_accept_min_contradiction_threshold", type=float, default=0.52)
    p.add_argument("--adaptive_accept_max_contradiction_threshold", type=float, default=0.86)
    p.add_argument("--adaptive_accept_min_repetition_threshold", type=float, default=0.72)
    p.add_argument("--adaptive_accept_max_repetition_threshold", type=float, default=0.94)
    p.add_argument("--image_out", default="need_image.png")
    p.add_argument("--image_grid", type=int, default=0)
    p.add_argument("--image_steps", type=int, default=24)
    p.add_argument("--image_temperature", type=float, default=1.0)
    p.add_argument("--image_top_k", type=int, default=256)
    p.add_argument("--image_quality_guidance", type=float, default=0.35)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--visual_tokenizer", default="", help="Optional learned VQ tokenizer dir; checkpoint dir is used if present")
    p.add_argument("--image_tokenizer_device", default="auto")
    p.add_argument("--negative_prompt", default="")
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--mask_schedule", choices=["cosine", "linear", "quadratic"], default="cosine")
    p.add_argument("--gumbel_noise", type=float, default=0.0)
    p.add_argument("--min_keep", type=int, default=1)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.mode == "text":
        run_text(args)
    else:
        run_image(args)


if __name__ == "__main__":
    main()
