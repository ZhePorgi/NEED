#!/usr/bin/env python3
"""Interactive terminal generation for NEED.

This keeps a loaded checkpoint in memory, appends each user prompt to a running
conversation transcript, prints elapsed-time line updates while NEED prepares
latent context and decodes, streams the final text to stdout, then prints public
reasoning summary chunks below the answer.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from copy import copy
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import torch

from generate import (
    build_arg_parser,
    _apply_runtime_profile,
    _apply_dvsd_runtime_overrides,
    _controller_runtime_adjust,
    _build_replay_context,
    _make_dual_channel_context,
    _speculative_final_decode,
    _extract_final_answer_text,
    _strip_hidden_runtime_artifacts,
)
from need_core import ByteTokenizer, load_model, resolve_device
from sidecar_lm_runtime import need_optimization_mode
from need_sidecar import make_single_sidecar_runtime


def _candidate_checkpoint_dirs(root: Path) -> list[Path]:
    names = {"model.pt", "best.pt", "model.safetensors", "best.safetensors", "config.json"}
    candidates: list[Path] = []
    for base in [root, root / "checkpoints", root / "runs", root / "outputs"]:
        if not base.exists():
            continue
        for path in [base] + [p for p in base.rglob("*") if p.is_dir()]:
            try:
                if any((path / n).exists() for n in names):
                    candidates.append(path)
            except OSError:
                continue
    return list({str(p.resolve()): p for p in candidates}.values())


def resolve_checkpoint(raw: str) -> str:
    raw = str(raw or "").strip()
    if raw:
        return raw
    candidates = _candidate_checkpoint_dirs(Path.cwd())
    if not candidates:
        raise FileNotFoundError("No --checkpoint was provided and no checkpoint-like directory was found under ., checkpoints, runs, or outputs.")
    candidates.sort(key=lambda p: max((x.stat().st_mtime for x in p.iterdir() if x.is_file()), default=p.stat().st_mtime), reverse=True)
    return str(candidates[0])


class ElapsedLine:
    def __init__(self, label: str = "reasoning") -> None:
        self.label = label
        self.started = 0.0
        self.done = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def __enter__(self):
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def _run(self) -> None:
        while not self.done.is_set():
            elapsed = time.perf_counter() - self.started
            sys.stderr.write(f"\r{self.label}: {elapsed:6.1f}s elapsed")
            sys.stderr.flush()
            self.done.wait(0.35)

    def __exit__(self, exc_type, exc, tb) -> None:
        self.done.set()
        if self.thread is not None:
            self.thread.join(timeout=0.5)
        elapsed = time.perf_counter() - self.started
        sys.stderr.write(f"\r{self.label}: {elapsed:6.1f}s elapsed\n")
        sys.stderr.flush()


def stream_text(text: str, mode: str, delay_s: float = 0.0) -> None:
    if mode == "stream_characters":
        chunks: Iterable[str] = text
    elif mode == "stream_tokens":
        chunks = re.findall(r"\S+\s*|\n+", text)
    else:
        chunks = [text]
    for chunk in chunks:
        print(chunk, end="", flush=True)
        if delay_s > 0:
            time.sleep(delay_s)
    if not text.endswith("\n"):
        print()


def generate_once(model, tok: ByteTokenizer, device: torch.device, args: argparse.Namespace, sidecar_rt: Any, prompt: str, applied_profiles: dict) -> tuple[str, str, dict]:
    controller_metrics = _controller_runtime_adjust(model, tok, prompt, args, device)
    replay_guidance, replay_metrics = _build_replay_context(model, tok, prompt, args, device)
    if replay_guidance:
        prompt = replay_guidance + "\n\n<current_user_prompt>\n" + prompt + "\n</current_user_prompt>"
    summary = ""
    raw_cot = ""
    if args.dual_channel_reasoning:
        ids, vectors, summary, raw_cot, cache_plan, prep_metrics = _make_dual_channel_context(model, tok, prompt, args, device, sidecar_rt)
        if getattr(args, "_sidecar_selection", None):
            prep_metrics["active_sidecar"] = getattr(args, "_sidecar_selection")
        if replay_metrics:
            prep_metrics.update(replay_metrics)
        if controller_metrics:
            prep_metrics["controller_runtime"] = controller_metrics
        setattr(args, "_latent_convergence_metrics", prep_metrics.get("latent_convergence", {}))
        cond_vectors = vectors
        cond_scale = args.conditioning_scale
    else:
        prep_metrics = dict(replay_metrics) if replay_metrics else {}
        if controller_metrics:
            prep_metrics["controller_runtime"] = controller_metrics
        ids = torch.tensor([tok.encode(prompt, add_bos=True)], dtype=torch.long, device=device)
        cond_vectors = None
        cond_scale = 0.0
        summary = "dual-channel reasoning disabled"
    gen_t0 = time.perf_counter()
    spec_metrics = {}
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
                    nonseq_refine_steps=args.nonseq_refine_steps,
                    nonseq_refine_causal_blend=args.nonseq_refine_causal_blend,
                    nonseq_refine_confidence_floor=args.nonseq_refine_confidence_floor,
                    nonseq_refine_temperature_decay=args.nonseq_refine_temperature_decay,
                    nonseq_refine_lock_schedule=args.nonseq_refine_lock_schedule,
                    nonseq_refine_resample_locked=args.nonseq_refine_resample_locked,
                    return_stats=True,
                )
                spec_metrics.update({"dvsd_" + str(k): v for k, v in dict(nonseq_metrics).items()})
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
    new_tokens = max(0, int(out.size(1) - ids.size(1)))
    dashboard = dict(prep_metrics)
    if applied_profiles:
        dashboard["runtime_profiles"] = applied_profiles
    dashboard.update({"need_decode_s": decode_s, "need_tokens_per_s": new_tokens / max(decode_s, 1e-9), "generated_tokens": new_tokens})
    if spec_metrics:
        dashboard.update(spec_metrics)
    return text, summary or raw_cot, dashboard


def build_terminal_parser() -> argparse.ArgumentParser:
    p = build_arg_parser()
    for action in p._actions:
        if "--checkpoint" in action.option_strings:
            action.required = False
            action.default = ""
        if "--system_prompt" in action.option_strings:
            action.default = "You are a helpful AI assistant."
    p.description = "Interactive terminal generation for NEED"
    p.add_argument("--display_mode", choices=["full", "stream_tokens", "stream_characters"], default="stream_tokens")
    p.add_argument("--stream_delay_s", type=float, default=0.0)
    p.add_argument("--no_append_history", action="store_true", help="Use each prompt alone rather than appending prior turns.")
    p.add_argument("--summary_chunks", type=int, default=6, help="Number of public reasoning summary chunks printed after each answer.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_terminal_parser().parse_args(argv)
    args.mode = "text"
    args.checkpoint = resolve_checkpoint(args.checkpoint)
    applied_profiles = _apply_runtime_profile(args)
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device=device, prefer_best=args.prefer_best, kernel_backend=args.kernel_backend)
    dvsd_runtime_overrides = _apply_dvsd_runtime_overrides(model, args)
    tok = ByteTokenizer()
    sidecar_rt = make_single_sidecar_runtime(args, device, model) if args.dual_channel_reasoning else None
    transcript: list[str] = []
    print(f"NEED terminal using {args.checkpoint}")
    print("Type /exit to quit. Type /reset to clear conversation history.")
    while True:
        try:
            user = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"/exit", "exit", "quit", ":q"}:
            break
        if user.lower() in {"/reset", "reset"}:
            transcript.clear()
            print("conversation history cleared")
            continue
        if args.no_append_history:
            prompt = f"{args.system_prompt}\n\nUser: {user}\nAssistant:"
        else:
            transcript.append(f"User: {user}")
            prompt = f"{args.system_prompt}\n\n" + "\n".join(transcript) + "\nAssistant:"
        call_args = copy(args)
        call_args.system_prompt = ""
        with ElapsedLine("reasoning"):
            answer, summary, dashboard = generate_once(model, tok, device, call_args, sidecar_rt, prompt, applied_profiles)
            if dvsd_runtime_overrides:
                dashboard["dvsd_runtime_overrides"] = dvsd_runtime_overrides
        stream_text(answer, args.display_mode, args.stream_delay_s)
        if not args.no_append_history:
            transcript.append(f"Assistant: {answer}")
            transcript[:] = transcript[-16:]
        chunks = [c.strip() for c in re.split(r"\n{2,}|(?=<[^/][^>]{1,40}>)", summary or "") if c.strip()]
        if chunks:
            print("\n<summary_chunks>")
            for chunk in chunks[: max(0, int(args.summary_chunks))]:
                print(chunk[:1200])
            print("</summary_chunks>")
        if args.performance_dashboard:
            print("\n<performance_dashboard>")
            print(json.dumps(dashboard, indent=2)[:6000])
            print("</performance_dashboard>")


if __name__ == "__main__":
    main()
