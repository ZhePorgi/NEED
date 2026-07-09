#!/usr/bin/env python3
"""Low-data adapter tuning for NEED control surfaces.

This module tunes the parts that behave like adapters rather than full pretraining:
aux_score/risk/contradiction heads, controller actions, output_mode mode routing,
speculative acceptance profiles, provided-transcription acceptance behavior, and
latent memory weighting.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from need_core import ByteTokenizer, load_tokenizer_for_dir, NeedModel, LatentMemoryStore, load_model, resolve_device, save_json, save_model


def _safe_torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _clean_text(x: object, max_len: int = 12000) -> str:
    return str(x or "").replace("\r\n", "\n")[:max_len]


def _as_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)  # type: ignore[arg-type]
    except Exception:
        return default


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _mode_id(x: object, default: int = -1) -> int:
    if x is None:
        return default
    if isinstance(x, (int, float)):
        return int(x)
    m = str(x).strip().lower().replace("-", "_").replace(" ", "_")
    names = {
        "none": 0,
        "off": 0,
        "no_cot": 0,
        "short": 1,
        "short_summary": 1,
        "summary": 1,
        "full": 2,
        "full_artificial_cot": 2,
        "cot": 2,
        "multi": 3,
        "multi_cot": 3,
        "renderer": 4,
        "renderer_only": 4,
    }
    return names.get(m, default)


def _controller_id(x: object, default: int = -1) -> int:
    if x is None:
        return default
    if isinstance(x, (int, float)):
        return int(x)
    m = str(x).strip().lower().replace("-", "_").replace(" ", "_")
    names = {"answer": 0, "respond": 0, "deepen": 1, "think": 1, "retrieve": 2, "recall": 2, "search": 2, "revise": 3, "repair": 3, "backtrack": 3}
    return names.get(m, default)


def normalize_control_row(row: Dict[str, object]) -> Dict[str, object]:
    prompt = _clean_text(row.get("prompt", row.get("input", row.get("question", ""))), 8000)
    answer = _clean_text(row.get("answer", row.get("response", row.get("chosen", row.get("completion", "")))), 8000)
    rejected = _clean_text(row.get("rejected", row.get("bad_answer", row.get("negative", ""))), 8000)
    score = _as_float(row.get("score", row.get("reward", row.get("quality", 0.5))), 0.5)
    score = max(0.0, min(1.0, score))
    risk = row.get("risk", row.get("risk_score", None))
    risk_f = max(0.0, min(1.0, _as_float(risk, 1.0 - score if risk is None else 0.5)))
    contradiction = row.get("contradiction", row.get("contradiction_score", None))
    con_f = max(0.0, min(1.0, _as_float(contradiction, 0.0 if contradiction is None else 0.5)))
    ctrl = _controller_id(row.get("controller_action", row.get("action", row.get("controller", None))), -1)
    vmode = _mode_id(row.get("output_mode", row.get("output_mode", row.get("mode", None))), -1)
    if ctrl < 0:
        if risk_f > 0.67 or con_f > 0.50 or score < 0.35:
            ctrl = 3
        elif risk_f > 0.38 or score < 0.70:
            ctrl = 1
        else:
            ctrl = 0
    if vmode < 0:
        difficulty = (risk_f + con_f + (1.0 - score)) / 3.0
        if difficulty < 0.25:
            vmode = 0
        elif difficulty < 0.45:
            vmode = 1
        elif difficulty < 0.70:
            vmode = 2
        else:
            vmode = 3
    return {
        "prompt": prompt,
        "answer": answer,
        "rejected": rejected,
        "score": score,
        "risk": risk_f,
        "contradiction": con_f,
        "controller_action": max(0, min(3, int(ctrl))),
        "output_mode": max(0, min(4, int(vmode))),
        "weight": max(0.15, min(2.0, _as_float(row.get("weight", 0.40 + score), 0.40 + score))),
    }


def augment_control_row(row: Dict[str, object], factor: int) -> List[Dict[str, object]]:
    row = normalize_control_row(row)
    p = _clean_text(row.get("prompt"), 8000)
    a = _clean_text(row.get("answer"), 8000)
    wrappers = [
        (p, a),
        ("Use the calibrated aux_score/controller policy for this task.\n\nTask:\n" + p, a),
        (p + "\n\nAnswer with the preferred risk level and reasoning visibility.", a),
        ("Prior successful behavior should be imitated without copying unsupported facts.\n\n" + p, a),
    ]
    out: List[Dict[str, object]] = []
    for i, (wp, wa) in enumerate(wrappers[: max(1, int(factor))]):
        nr = dict(row)
        nr["prompt"] = wp
        nr["answer"] = wa
        nr["augmentation_id"] = i
        out.append(nr)
    return out


def expand_rows(rows: List[Dict[str, object]], factor: int) -> List[Dict[str, object]]:
    if factor <= 1:
        return [normalize_control_row(r) for r in rows]
    out: List[Dict[str, object]] = []
    for row in rows:
        out.extend(augment_control_row(row, factor))
    return out


class ControlDataset(Dataset):
    def __init__(self, rows: List[Dict[str, object]], tok: ByteTokenizer, block_size: int):
        self.rows = rows
        self.tok = tok
        self.block_size = int(block_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx % len(self.rows)]
        text = _clean_text(row.get("prompt"), 8000) + "\n\nAnswer:\n" + _clean_text(row.get("answer"), 8000)
        ids = self.tok.encode(text, add_bos=True, add_eos=True)[-self.block_size:]
        if len(ids) < self.block_size:
            ids = ids + [0] * (self.block_size - len(ids))
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "score": torch.tensor(float(row["score"]), dtype=torch.float32),
            "risk": torch.tensor(float(row["risk"]), dtype=torch.float32),
            "contradiction": torch.tensor(float(row["contradiction"]), dtype=torch.float32),
            "controller": torch.tensor(int(row["controller_action"]), dtype=torch.long),
            "output_mode": torch.tensor(int(row["output_mode"]), dtype=torch.long),
            "weight": torch.tensor(float(row.get("weight", 1.0)), dtype=torch.float32),
        }


def freeze_control_surfaces(model: NeedModel, train_revision: bool = False, train_pathway: bool = False) -> int:
    keep_terms = [
        "aux_score",
        "risk_signal_fusion",
        "latent_divergence",
        "output_mode_classifier",
        "norm",
        "ln",
        "bias",
    ]
    if train_revision:
        keep_terms.append("revision_proj")
    if train_pathway:
        keep_terms.extend(["path_conditioner", "latent_slot_attention", "mixture_energy_router"])
    trainable = 0
    for name, p in model.named_parameters():
        keep = any(term in name.lower() for term in keep_terms)
        p.requires_grad_(keep)
        if keep:
            trainable += p.numel()
    return trainable


def tune_control(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device=device, prefer_best=args.prefer_best, kernel_backend=args.kernel_backend)
    tok = load_tokenizer_for_dir(args.checkpoint)
    raw_rows = _read_jsonl(Path(args.interactions))
    if not raw_rows:
        raise ValueError("No rows found in interactions JSONL")
    low_data_auto = bool(args.auto_few_shot and len(raw_rows) <= args.few_shot_auto_threshold)
    if low_data_auto:
        args.few_shot_mode = True
        if not args.augment:
            args.augment = True
        args.lr = min(args.lr, args.few_shot_max_lr)
        args.weight_decay = max(args.weight_decay, args.few_shot_min_weight_decay)
    rows = expand_rows(raw_rows, args.augment_factor if args.augment else 1)
    ds = ControlDataset(rows, tok, model.cfg.block_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.few_shot_mode:
        trainable = freeze_control_surfaces(model, train_revision=args.train_revision, train_pathway=args.train_pathway)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters after control-surface freeze")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    print(json.dumps({
        "control_low_data_mode": low_data_auto,
        "few_shot_mode": bool(args.few_shot_mode),
        "raw_examples": len(raw_rows),
        "expanded_examples": len(rows),
        "trainable_parameters": int(trainable),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }), flush=True)
    step = 0
    last: Dict[str, float] = {}
    while step < args.steps:
        for batch in dl:
            if step >= args.steps:
                break
            ids = batch["ids"].to(device)
            mask = ids != model.cfg.pad_id
            logits, _, aux = model(ids, None, return_hidden=True)
            h = aux["_hidden"]
            vf = model.aux_score(h)
            m = mask.float().unsqueeze(-1)
            denom = m.sum(dim=1).clamp_min(1.0)
            vf_pool = (vf * m).sum(dim=1) / denom
            h_pool = (h * m).sum(dim=1) / denom
            q_pred = torch.sigmoid(vf_pool[:, 0])
            r_pred = (F.softplus(vf_pool[:, 1]).clamp(max=8.0) / 8.0)
            c_pred = torch.sigmoid(vf_pool[:, 3])
            ctrl_logits = vf_pool[:, 5:9]
            vpol_logits = model.output_mode_classifier(h_pool.unsqueeze(1))
            score = batch["score"].to(device)
            risk = batch["risk"].to(device)
            con = batch["contradiction"].to(device)
            ctrl = batch["controller"].to(device)
            vmode = batch["output_mode"].to(device)
            weight = batch["weight"].to(device).clamp(0.1, 2.5)
            quality_loss = F.binary_cross_entropy(q_pred.clamp(1e-5, 1 - 1e-5), score, reduction="none")
            risk_loss = F.mse_loss(r_pred, risk, reduction="none")
            con_loss = F.binary_cross_entropy(c_pred.clamp(1e-5, 1 - 1e-5), con, reduction="none")
            ctrl_loss = F.cross_entropy(ctrl_logits, ctrl, reduction="none")
            vmode_loss = F.cross_entropy(vpol_logits, vmode, reduction="none")
            loss = (
                args.quality_weight * quality_loss
                + args.risk_weight * risk_loss
                + args.contradiction_weight * con_loss
                + args.controller_weight * ctrl_loss
                + args.output_mode_weight * vmode_loss
            )
            loss = (loss * weight).mean()
            if args.stability_weight > 0:
                noise = torch.randn_like(h_pool) * float(args.feature_jitter)
                vp2 = model.output_mode_classifier((h_pool + noise).unsqueeze(1))
                loss = loss + float(args.stability_weight) * F.kl_div(F.log_softmax(vp2, dim=-1), F.softmax(vpol_logits.detach(), dim=-1), reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            if step % max(1, args.log_interval) == 0:
                last = {
                    "step": float(step),
                    "loss": float(loss.detach().cpu()),
                    "quality_mae": float((q_pred.detach() - score).abs().mean().cpu()),
                    "risk_mae": float((r_pred.detach() - risk).abs().mean().cpu()),
                    "controller_acc": float((ctrl_logits.argmax(dim=-1) == ctrl).float().mean().detach().cpu()),
                    "output_mode_acc": float((vpol_logits.argmax(dim=-1) == vmode).float().mean().detach().cpu()),
                }
                print(json.dumps(last), flush=True)
            step += 1
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_model(model, out, {"low_data_control_steps": float(args.steps), **last}, name="model")
    save_json({
        "format": "need_low_data_control_adapter",
        "raw_examples": len(raw_rows),
        "expanded_examples": len(rows),
        "few_shot_mode": bool(args.few_shot_mode),
        "metrics": last,
    }, out / "control_adapter_config.json")
    print(json.dumps({"done": True, "out_dir": str(out)}, indent=2), flush=True)


def calibrate_spec(args: argparse.Namespace) -> None:
    rows = _read_jsonl(Path(args.traces))
    if not rows:
        raise ValueError("No speculative trace rows found")
    rates: List[float] = []
    aux_score_rejects = 0
    prob_rejects = 0
    repetition_rejects = 0
    for row in rows:
        if "accepted_tokens" in row and "draft_tokens" in row:
            total = max(1.0, _as_float(row.get("draft_tokens"), 1.0))
            rates.append(max(0.0, min(1.0, _as_float(row.get("accepted_tokens"), 0.0) / total)))
        elif "accepted_need_tokens" in row and "draft_need_tokens" in row:
            total = max(1.0, _as_float(row.get("draft_need_tokens"), 1.0))
            rates.append(max(0.0, min(1.0, _as_float(row.get("accepted_need_tokens"), 0.0) / total)))
        elif "spec_accept_rate" in row:
            rates.append(max(0.0, min(1.0, _as_float(row.get("spec_accept_rate"), 0.0))))
        reason = str(row.get("reject_reason", row.get("last_reject_reason", ""))).lower()
        if "risk" in reason or "contradiction" in reason or "aux_score" in reason:
            aux_score_rejects += 1
        if "prob" in reason or "top" in reason or "gap" in reason:
            prob_rejects += 1
        if "repeat" in reason:
            repetition_rejects += 1
    mean_rate = sum(rates) / max(1, len(rates))
    # Low data calibration changes the adaptive bounds, not the decoding algorithm.
    # If acceptance is too low for non-aux_score reasons, loosen top-k/gap/min-prob.
    # If aux_score failures dominate, tighten risk/contradiction while keeping fallback active.
    loosen = max(0.0, float(args.target_accept_rate) - mean_rate)
    tighten = max(0.0, mean_rate - float(args.target_accept_rate))
    n = max(1, len(rows))
    aux_score_frac = aux_score_rejects / n
    prob_frac = prob_rejects / n
    rep_frac = repetition_rejects / n
    profile = {
        "format": "need_adaptive_spec_profile",
        "source_traces": len(rows),
        "observed_accept_rate": mean_rate,
        "target_accept_rate": float(args.target_accept_rate),
        "adaptive_accept_target_rate": float(args.target_accept_rate),
        "adaptive_accept_min_top_k": int(max(3, round(args.base_min_top_k * (1.0 - 0.35 * tighten)))),
        "adaptive_accept_max_top_k": int(max(8, round(args.base_max_top_k * (1.0 + 1.25 * loosen + 0.35 * prob_frac)))),
        "adaptive_accept_min_gap": max(0.8, float(args.base_min_gap) * (1.0 - 0.30 * tighten)),
        "adaptive_accept_max_gap": max(1.5, float(args.base_max_gap) * (1.0 + 1.10 * loosen + 0.25 * prob_frac)),
        "adaptive_accept_min_min_prob": max(0.001, float(args.base_min_prob) * (1.0 - 0.50 * loosen)),
        "adaptive_accept_max_min_prob": min(0.08, float(args.base_max_prob) * (1.0 + 0.65 * tighten)),
        "adaptive_accept_min_risk_threshold": max(0.35, min(0.90, float(args.base_min_risk) - 0.10 * aux_score_frac + 0.05 * loosen)),
        "adaptive_accept_max_risk_threshold": max(0.45, min(0.98, float(args.base_max_risk) - 0.08 * aux_score_frac + 0.05 * loosen)),
        "adaptive_accept_min_contradiction_threshold": max(0.30, min(0.90, float(args.base_min_contradiction) - 0.10 * aux_score_frac + 0.05 * loosen)),
        "adaptive_accept_max_contradiction_threshold": max(0.40, min(0.96, float(args.base_max_contradiction) - 0.08 * aux_score_frac + 0.05 * loosen)),
        "adaptive_accept_min_repetition_threshold": max(0.50, min(0.95, float(args.base_min_repetition) - 0.08 * rep_frac)),
        "adaptive_accept_max_repetition_threshold": max(0.60, min(0.98, float(args.base_max_repetition) - 0.05 * rep_frac)),
    }
    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_json(profile, out)
    print(json.dumps({"done": True, "profile": str(out), "observed_accept_rate": mean_rate}, indent=2), flush=True)


def build_latent_memory(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device=device, prefer_best=args.prefer_best, kernel_backend=args.kernel_backend)
    tok = load_tokenizer_for_dir(args.checkpoint)
    rows = [normalize_control_row(r) for r in _read_jsonl(Path(args.interactions))]
    if args.max_items > 0:
        rows = rows[: args.max_items]
    out = Path(args.out_dir)
    mem = LatentMemoryStore(out, dim=model.cfg.d_model, max_items=max(1, args.max_items if args.max_items > 0 else args.max_memory_items))
    for i, row in enumerate(rows):
        text = (row["prompt"] + "\n" + row["answer"])[: args.max_chars]
        ids = torch.tensor([tok.encode(text, add_bos=True)[-model.cfg.block_size:]], dtype=torch.long, device=device)
        with torch.no_grad():
            path = model.latent_pathway(ids, stride=args.vector_stride, max_vectors=args.max_vectors)
        summary = _clean_text(row.get("answer"), 1200)
        tags = {
            "score": float(row["score"]),
            "risk": float(row["risk"]),
            "contradiction": float(row["contradiction"]),
            "controller_action": int(row["controller_action"]),
            "output_mode": int(row["output_mode"]),
        }
        mem.add(str(row["prompt"]), summary, path["pathway_vectors"].detach().cpu(), tags=tags)
        if (i + 1) % max(1, args.log_interval) == 0:
            print(json.dumps({"stored": i + 1}), flush=True)
    save_json({"format": "need_weighted_latent_memory", "items": len(rows), "checkpoint": str(args.checkpoint)}, out / "manifest.json")
    print(json.dumps({"done": True, "items": len(rows), "out_dir": str(out)}, indent=2), flush=True)


def _quantile(vals: List[float], q: float, default: float) -> float:
    xs = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not xs:
        return default
    pos = max(0, min(len(xs) - 1, int(round((len(xs) - 1) * float(q)))))
    return xs[pos]


def make_runtime_profile(args: argparse.Namespace) -> None:
    """Bundle low-data adapter outputs into generation-safe runtime settings."""
    runtime: Dict[str, object] = {
        "auto_output_mode_classifier": True,
        "adaptive_spec_acceptance": True,
        "accept_provided_transcriptions": True,
        "provided_transcription_grounding": "use_as_text_evidence_from_overheard_scenario",
        "provided_transcription_confidence_caveat_threshold": 0.72,
        "reasoning_tree_branches": 1,
        "reasoning_tree_merge_top_k": 1,
        "latent_search_depth": 0,
        "latent_search_branches": 1,
        "aux_score_weight": 0.35,
        "aux_score_candidate_pool": 1,
        "aux_score_backtrack_window": 3,
        "aux_score_max_backtracks": 4,
        "latent_tools": True,
        "latent_tool_calculator": True,
        "latent_tool_python": True,
        "latent_tool_sidecar_planning": False,
        "latent_tool_max_calls": 3,
        "latent_tool_timeout_s": 3.0,
        "latent_tool_max_code_chars": 4000,
        "latent_tool_max_output_chars": 2400,
        "system_prompt": "You can use a calculator and a code execution tool.",
        "latent_tool_policy": "runtime_built_hidden_observations_no_model_generated_calls_llrl_not_required",
    }
    profile: Dict[str, object] = {
        "format": "need_low_data_runtime_profile",
        "runtime": runtime,
        "sources": {},
    }
    if args.sidecar_model:
        runtime["sidecar_model"] = args.sidecar_model
        profile["sources"]["sidecar_model"] = args.sidecar_model
    if args.sidecar_adapter_path:
        runtime["sidecar_adapter_path"] = args.sidecar_adapter_path
        profile["sources"]["sidecar_adapter_path"] = args.sidecar_adapter_path
    if args.sidecar_latent_alignment_path:
        runtime["sidecar_latent_alignment_path"] = args.sidecar_latent_alignment_path
        runtime["use_sidecar_latent_alignment"] = bool(args.use_sidecar_latent_alignment)
        runtime["sidecar_latent_alignment_weight"] = float(args.sidecar_latent_alignment_weight)
        profile["sources"]["sidecar_latent_alignment_path"] = args.sidecar_latent_alignment_path
    if args.interactions:
        raw_interaction_rows = _read_jsonl(Path(args.interactions))
        rows = [normalize_control_row(r) for r in raw_interaction_rows]
        risks = [float(r["risk"]) for r in rows]
        cons = [float(r["contradiction"]) for r in rows]
        hard_frac = sum(1 for r in rows if float(r["risk"]) > 0.55 or float(r["contradiction"]) > 0.35 or float(r["score"]) < 0.45) / max(1, len(rows))
        runtime["aux_score_risk_threshold"] = max(0.45, min(0.88, _quantile(risks, 0.80, 0.72) + 0.05))
        runtime["aux_score_contradiction_threshold"] = max(0.35, min(0.82, _quantile(cons, 0.80, 0.65) + 0.05))
        runtime["reasoning_tree_branches"] = 3 if hard_frac > 0.35 else 1
        runtime["reasoning_tree_merge_top_k"] = 2 if hard_frac > 0.35 else 1
        runtime["latent_search_depth"] = 1 if hard_frac > 0.45 else 0
        runtime["latent_search_branches"] = 2 if hard_frac > 0.45 else 1
        runtime["aux_score_weight"] = 0.45 if hard_frac > 0.30 else 0.35
        categories = sorted({str(r.get("category", "")) for r in raw_interaction_rows if r.get("category")})
        if any(c.startswith("image_") for c in categories):
            runtime["image_generation_behavior_policy"] = True
            runtime["image_prompt_fidelity_weight"] = 0.35
            runtime["image_avoid_unrequested_additions"] = True
            runtime["image_do_not_claim_generation_without_tool"] = True
            runtime["image_edit_preserve_unmentioned_regions"] = True
        if "tool_routing" in categories:
            runtime["tool_routing_policy"] = "prefer_direct_for_stable_facts_use_tools_for_current_or_external_facts"
        if any(c in categories for c in ["latent_tool_calculator", "latent_tool_python", "latent_tool_preferences", "code_execution_behavior"]):
            runtime["latent_tool_policy"] = "runtime_built_hidden_observations_no_model_generated_calls_llrl_not_required"
            runtime["latent_tool_training_seen"] = True
        if "structured_json" in categories:
            runtime["structured_output_policy"] = "strict_json_no_fence_when_requested"
        if "behavioral_memory_policy" in categories:
            runtime["memory_grounding_policy"] = "behavioral_guidance_not_factual_context"
        if "self_correction_aux_score" in categories:
            runtime["self_correction_policy"] = "repair_main_issue_without_unnecessary_rewrite"
        profile["sources"]["interactions"] = {"path": args.interactions, "rows": len(rows), "hard_fraction": hard_frac, "categories": categories}
    if args.spec_profile:
        spec = json.loads(Path(args.spec_profile).read_text(encoding="utf-8"))
        for k, v in spec.items():
            if k.startswith("adaptive_accept_") or k in {"adaptive_spec_acceptance", "spec_draft_tokens", "spec_max_need_tokens_per_draft", "spec_accept_top_k", "spec_accept_min_need_prob", "spec_accept_max_logprob_gap", "spec_risk_threshold", "spec_contradiction_threshold", "spec_repetition_threshold"}:
                runtime[k] = v
        profile["sources"]["spec_profile"] = args.spec_profile
    if args.latent_memory_dir:
        profile["sources"]["latent_memory_dir"] = {
            "path": args.latent_memory_dir,
            "note": "recorded as an available behavioral-memory source; runtime retrieval remains opt-in",
        }
    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_json(profile, out)
    print(json.dumps({"done": True, "profile": str(out), "runtime_keys": sorted(runtime.keys())}, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Low-data NEED adapter tuning")
    sub = p.add_subparsers(dest="cmd", required=True)
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--checkpoint", required=True)
    base.add_argument("--prefer_best", action="store_true")
    base.add_argument("--device", default="auto")
    base.add_argument("--kernel_backend", default="auto")

    t = sub.add_parser("tune_control", parents=[base], help="tune aux_score/controller/output_mode surfaces")
    t.add_argument("--interactions", required=True)
    t.add_argument("--out_dir", required=True)
    t.add_argument("--steps", type=int, default=300)
    t.add_argument("--batch_size", type=int, default=4)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--weight_decay", type=float, default=0.01)
    t.add_argument("--grad_clip", type=float, default=1.0)
    t.add_argument("--seed", type=int, default=123)
    t.add_argument("--log_interval", type=int, default=20)
    t.add_argument("--few_shot_mode", action="store_true")
    t.add_argument("--auto_few_shot", action=argparse.BooleanOptionalAction, default=True)
    t.add_argument("--few_shot_auto_threshold", type=int, default=300)
    t.add_argument("--few_shot_max_lr", type=float, default=2e-5)
    t.add_argument("--few_shot_min_weight_decay", type=float, default=0.02)
    t.add_argument("--augment", action="store_true")
    t.add_argument("--augment_factor", type=int, default=3)
    t.add_argument("--train_revision", action="store_true")
    t.add_argument("--train_pathway", action="store_true")
    t.add_argument("--quality_weight", type=float, default=1.0)
    t.add_argument("--risk_weight", type=float, default=1.0)
    t.add_argument("--contradiction_weight", type=float, default=0.8)
    t.add_argument("--controller_weight", type=float, default=0.7)
    t.add_argument("--output_mode_weight", type=float, default=0.7)
    t.add_argument("--feature_jitter", type=float, default=0.015)
    t.add_argument("--stability_weight", type=float, default=0.04)

    s = sub.add_parser("calibrate_spec", help="make an adaptive speculative acceptance profile from traces")
    s.add_argument("--traces", required=True)
    s.add_argument("--out_file", required=True)
    s.add_argument("--target_accept_rate", type=float, default=0.78)
    s.add_argument("--base_min_top_k", type=int, default=6)
    s.add_argument("--base_max_top_k", type=int, default=80)
    s.add_argument("--base_min_gap", type=float, default=1.8)
    s.add_argument("--base_max_gap", type=float, default=9.0)
    s.add_argument("--base_min_prob", type=float, default=0.003)
    s.add_argument("--base_max_prob", type=float, default=0.030)
    s.add_argument("--base_min_risk", type=float, default=0.58)
    s.add_argument("--base_max_risk", type=float, default=0.90)
    s.add_argument("--base_min_contradiction", type=float, default=0.52)
    s.add_argument("--base_max_contradiction", type=float, default=0.86)
    s.add_argument("--base_min_repetition", type=float, default=0.72)
    s.add_argument("--base_max_repetition", type=float, default=0.94)

    m = sub.add_parser("build_memory", parents=[base], help="build weighted latent memory from successful episodes")
    m.add_argument("--interactions", required=True)
    m.add_argument("--out_dir", required=True)
    m.add_argument("--vector_stride", type=int, default=8)
    m.add_argument("--max_vectors", type=int, default=64)
    m.add_argument("--max_items", type=int, default=0)
    m.add_argument("--max_memory_items", type=int, default=256)
    m.add_argument("--max_chars", type=int, default=8000)
    m.add_argument("--log_interval", type=int, default=50)

    rp = sub.add_parser("make_runtime_profile", help="bundle low-data adapter outputs into one generation profile")
    rp.add_argument("--out_file", required=True)
    rp.add_argument("--interactions", default="")
    rp.add_argument("--spec_profile", default="")
    rp.add_argument("--latent_memory_dir", default="")
    rp.add_argument("--latent_memory_k", type=int, default=4)
    rp.add_argument("--latent_memory_min_score", type=float, default=0.45)
    rp.add_argument("--latent_memory_score_weight", type=float, default=0.18)
    rp.add_argument("--latent_memory_risk_weight", type=float, default=0.22)
    rp.add_argument("--latent_memory_contradiction_weight", type=float, default=0.25)
    rp.add_argument("--sidecar_model", default="", help="Optional trained full sidecar model path from need_thought_distill.py train_alignment")
    rp.add_argument("--sidecar_adapter_path", default="", help="Optional trained sidecar adapter from need_thought_distill.py train_alignment")
    rp.add_argument("--sidecar_latent_alignment_path", default="", help="Optional sidecar latent-projection directory containing latent_projection.pt")
    rp.add_argument("--use_sidecar_latent_alignment", action=argparse.BooleanOptionalAction, default=True, help="Enable sidecar-projected NEED latent anchor when an alignment path is supplied")
    rp.add_argument("--sidecar_latent_alignment_weight", type=float, default=0.35)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "tune_control":
        tune_control(args)
    elif args.cmd == "calibrate_spec":
        calibrate_spec(args)
    elif args.cmd == "build_memory":
        build_latent_memory(args)
    elif args.cmd == "make_runtime_profile":
        make_runtime_profile(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
