#!/usr/bin/env python3
"""One-command low-data RL generation and immediate NEED/sidecar feeding.

This runner ties together the modular low-data RL generator, NEED control-surface
adapter tuning, sidecar latent-alignment training, optional real image-preference
feeding, and runtime-profile export.  It does not generate images.  Image-related
synthetic rows are text-only behavioral examples about prompt fidelity, editing,
composition scoring, and safety boundaries.

Minimal activated run:

  OPENAI_API_KEY=... python need_low_data_rl_start.py \
    --need_checkpoint checkpoints/need \
    --sidecar_model models/sidecar \
    --out_root runs/need_low_data_rl

Dry run / prompt audit only:

  python need_low_data_rl_start.py --dry_run --out_root runs/need_low_data_rl_dry
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"config must be a JSON object: {path}")
    return obj


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _bool_flag(cmd: List[str], flag: str, value: bool) -> None:
    if value:
        cmd.append(flag)


def _append_arg(cmd: List[str], name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str) and value == "":
        return
    cmd.extend([name, str(value)])


def _stream_command(cmd: Sequence[str], log_path: Path, env: Dict[str, str], cwd: Path, allow_failure: bool = False) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    safe_cmd = ["<api-key>" if (i > 0 and cmd[i - 1] == "--api_key") else x for i, x in enumerate(cmd)]
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"cmd": safe_cmd, "cwd": str(cwd)}, ensure_ascii=False) + "\n")
        log.flush()
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        rc = proc.wait()
    record = {"cmd": safe_cmd, "returncode": rc, "log": str(log_path), "seconds": round(time.time() - started, 3)}
    if rc != 0 and not allow_failure:
        raise RuntimeError(f"command failed with code {rc}; see {log_path}")
    return record


def _load_config_into_args(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        return args
    cfg = _read_json(Path(args.config))
    for key, value in cfg.items():
        if hasattr(args, key) and getattr(args, key) in (None, "", False):
            setattr(args, key, value)
        elif hasattr(args, key) and key in {"total_examples", "batch_size", "need_steps", "sidecar_steps", "image_steps"}:
            # Numeric defaults are meaningful, so only override when config was explicitly supplied.
            setattr(args, key, value)
    return args


def _redacted_config(args: argparse.Namespace) -> Dict[str, Any]:
    obj = vars(args).copy()
    if obj.get("api_key"):
        obj["api_key"] = "<redacted>"
    return obj


def _api_env(args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    if args.base_url:
        env["OPENAI_BASE_URL"] = str(args.base_url)
    key = args.api_key or env.get(args.api_key_env, "")
    if key:
        env[args.api_key_env] = key
    if not args.dry_run and not args.skip_generate and not key:
        raise ValueError(f"No API key found. Pass --api_key or set {args.api_key_env}.")
    return env


def build_generation_cmd(args: argparse.Namespace, corpus_dir: Path, control_file: Path) -> List[str]:
    cmd = [sys.executable, str(SCRIPT_DIR / "need_auto_low_data_rl.py")]
    if not args.dry_run:
        cmd.append("--activate")
    _append_arg(cmd, "--model", args.model)
    _append_arg(cmd, "--base_url", args.base_url)
    _append_arg(cmd, "--api_key_env", args.api_key_env)
    if args.api_key:
        _append_arg(cmd, "--api_key", args.api_key)
    _append_arg(cmd, "--api_format", args.api_format)
    _append_arg(cmd, "--out_dir", corpus_dir)
    _append_arg(cmd, "--target_profile", args.target_profile)
    _append_arg(cmd, "--total_examples", args.total_examples)
    _append_arg(cmd, "--batch_size", args.batch_size)
    _append_arg(cmd, "--max_total_examples", max(int(args.max_total_examples), int(args.total_examples) if str(args.total_examples).isdigit() else 3000))
    _append_arg(cmd, "--wikipedia_source", args.wikipedia_source)
    _append_arg(cmd, "--topic_pool_size", args.topic_pool_size)
    _append_arg(cmd, "--articles_per_batch", args.articles_per_batch)
    _append_arg(cmd, "--temperature", args.temperature)
    _append_arg(cmd, "--max_output_tokens", args.max_output_tokens)
    _append_arg(cmd, "--control_interactions_file", control_file)
    _append_arg(cmd, "--seed", args.seed)
    if args.topics_file:
        _append_arg(cmd, "--topics_file", args.topics_file)
    if args.transcripts_file:
        _append_arg(cmd, "--transcripts_file", args.transcripts_file)
        _append_arg(cmd, "--transcripts_emit_as", args.transcripts_emit_as)
    return cmd


def build_need_tune_cmd(args: argparse.Namespace, control_file: Path, out_dir: Path) -> List[str]:
    cmd = [sys.executable, str(SCRIPT_DIR / "need_low_data_adapters.py"), "tune_control"]
    _append_arg(cmd, "--checkpoint", args.need_checkpoint)
    _bool_flag(cmd, "--prefer_best", bool(args.prefer_best))
    _append_arg(cmd, "--interactions", control_file)
    _append_arg(cmd, "--out_dir", out_dir)
    _append_arg(cmd, "--steps", args.need_steps)
    _append_arg(cmd, "--batch_size", args.need_batch_size)
    _append_arg(cmd, "--device", args.device)
    _append_arg(cmd, "--kernel_backend", args.kernel_backend)
    _append_arg(cmd, "--lr", args.need_lr)
    _append_arg(cmd, "--seed", args.seed)
    _bool_flag(cmd, "--augment", bool(args.augment))
    _append_arg(cmd, "--augment_factor", args.augment_factor)
    _bool_flag(cmd, "--train_revision", bool(args.train_revision))
    _bool_flag(cmd, "--train_pathway", bool(args.train_pathway))
    return cmd


def build_alignment_dataset_cmd(args: argparse.Namespace, corpus_dir: Path, out_dir: Path) -> List[str]:
    cmd = [sys.executable, str(SCRIPT_DIR / "need_thought_distill.py"), "build_alignment_dataset"]
    _append_arg(cmd, "--need_checkpoint", args.need_checkpoint)
    _bool_flag(cmd, "--prefer_best", bool(args.prefer_best))
    _append_arg(cmd, "--interactions", corpus_dir / "rl" / "sidecar_latent_alignment.synthetic.jsonl")
    _append_arg(cmd, "--out_dir", out_dir)
    _append_arg(cmd, "--device", args.device)
    _append_arg(cmd, "--max_examples", args.sidecar_max_examples)
    _append_arg(cmd, "--seed", args.seed)
    if args.alignment_corpus:
        _append_arg(cmd, "--corpus", args.alignment_corpus)
    if args.transcripts_file:
        _append_arg(cmd, "--transcripts_file", args.transcripts_file)
    return cmd


def build_train_alignment_cmd(args: argparse.Namespace, dataset_dir: Path, out_dir: Path) -> List[str]:
    cmd = [sys.executable, str(SCRIPT_DIR / "need_thought_distill.py"), "train_alignment"]
    _append_arg(cmd, "--sidecar_model", args.sidecar_model)
    _append_arg(cmd, "--dataset_jsonl", dataset_dir / "sidecar_latent_alignment.jsonl")
    _append_arg(cmd, "--latents_pt", dataset_dir / "sidecar_latents.pt")
    _append_arg(cmd, "--out_dir", out_dir)
    _append_arg(cmd, "--device", args.device)
    _append_arg(cmd, "--dtype", args.sidecar_dtype)
    _append_arg(cmd, "--train_mode", args.sidecar_train_mode)
    _append_arg(cmd, "--steps", args.sidecar_steps)
    _append_arg(cmd, "--batch_size", args.sidecar_batch_size)
    _append_arg(cmd, "--lr", args.sidecar_lr)
    _append_arg(cmd, "--seed", args.seed)
    _bool_flag(cmd, "--trust_remote_code", bool(args.sidecar_trust_remote_code))
    return cmd



def build_train_need_sidecar_cmd(args: argparse.Namespace, corpus_dir: Path, out_dir: Path) -> List[str]:
    """Train an architecture-native smaller NEED sidecar from the current teacher."""
    cmd = [sys.executable, str(SCRIPT_DIR / "need_sidecar_distill.py"), "train"]
    _append_arg(cmd, "--teacher_checkpoint", args.need_checkpoint)
    _bool_flag(cmd, "--teacher_prefer_best", bool(args.prefer_best))
    _append_arg(cmd, "--corpus", corpus_dir)
    _append_arg(cmd, "--out_dir", out_dir)
    _append_arg(cmd, "--device", args.device)
    _append_arg(cmd, "--kernel_backend", args.kernel_backend)
    _append_arg(cmd, "--target_params", args.need_sidecar_target_params)
    _append_arg(cmd, "--student_scale", args.need_sidecar_student_scale)
    _append_arg(cmd, "--steps", args.need_sidecar_steps)
    _append_arg(cmd, "--batch_size", args.need_sidecar_batch_size)
    _append_arg(cmd, "--lr", args.need_sidecar_lr)
    _append_arg(cmd, "--samples", args.need_sidecar_samples)
    _append_arg(cmd, "--max_docs", args.sidecar_max_examples)
    _append_arg(cmd, "--seed", args.seed)
    return cmd


def _inject_need_sidecar_runtime_profile(profile_path: Path, sidecar_dir: Path) -> None:
    if not profile_path.exists() or not sidecar_dir.exists():
        return
    data = _read_json(profile_path)
    runtime = data.setdefault("runtime", {})
    if isinstance(runtime, dict):
        runtime.update({
            "sidecar_type": "need",
            "need_sidecar_checkpoint": str(sidecar_dir),
            "need_sidecar_projection_path": str(sidecar_dir),
            "use_need_sidecar_latents": True,
            "use_sidecar_latent_alignment": False,
            "sidecar_call_policy": "latent_gated",
            "sidecar_gate_metric": "latent_difficulty",
            "sidecar_gate_threshold": 0.42,
            "dvsd_router_enabled": True,
            "dvsd_router_inference_mix": 0.65,
            "dvsd_router_min_confidence": 0.20,
        })
    data.setdefault("sources", {})["need_sidecar_checkpoint"] = str(sidecar_dir)
    _write_json(profile_path, data)

def build_runtime_profile_cmd(args: argparse.Namespace, control_file: Path, out_file: Path, sidecar_alignment_dir: Optional[Path]) -> List[str]:
    cmd = [sys.executable, str(SCRIPT_DIR / "need_low_data_adapters.py"), "make_runtime_profile"]
    _append_arg(cmd, "--out_file", out_file)
    _append_arg(cmd, "--interactions", control_file)
    if args.sidecar_model:
        _append_arg(cmd, "--sidecar_model", args.sidecar_model)
    if sidecar_alignment_dir is not None:
        adapter = sidecar_alignment_dir / "sidecar_adapter"
        if adapter.exists():
            _append_arg(cmd, "--sidecar_adapter_path", adapter)
        _append_arg(cmd, "--sidecar_latent_alignment_path", sidecar_alignment_dir)
        if args.use_sidecar_latent_alignment:
            cmd.append("--use_sidecar_latent_alignment")
        else:
            cmd.append("--no-use_sidecar_latent_alignment")
        _append_arg(cmd, "--sidecar_latent_alignment_weight", args.sidecar_latent_alignment_weight)
    return cmd


def build_image_policy_cmd(args: argparse.Namespace, out_dir: Path) -> Optional[List[str]]:
    if not args.image_preferences or not args.visual_tokenizer:
        return None
    cmd = [sys.executable, str(SCRIPT_DIR / "need_image_rl.py"), "tune_policy"]
    _append_arg(cmd, "--checkpoint", args.need_checkpoint)
    _bool_flag(cmd, "--prefer_best", bool(args.prefer_best))
    _append_arg(cmd, "--visual_tokenizer", args.visual_tokenizer)
    _append_arg(cmd, "--preferences", args.image_preferences)
    _append_arg(cmd, "--image_root", args.image_root)
    _append_arg(cmd, "--out_dir", out_dir)
    _append_arg(cmd, "--device", args.device)
    _append_arg(cmd, "--kernel_backend", args.kernel_backend)
    _append_arg(cmd, "--steps", args.image_steps)
    _append_arg(cmd, "--batch_size", args.image_batch_size)
    _append_arg(cmd, "--lr", args.image_lr)
    _append_arg(cmd, "--seed", args.seed)
    _bool_flag(cmd, "--augment_preferences", bool(args.augment))
    _append_arg(cmd, "--augment_factor", args.augment_factor)
    return cmd



def build_raw_image_start_cmd(args: argparse.Namespace, out_dir: Path) -> Optional[List[str]]:
    if not (args.raw_image_dir or args.raw_image_manifest or args.raw_image_urls_file):
        return None
    cmd = [sys.executable, str(SCRIPT_DIR / "need_raw_image_data.py"), "start"]
    if args.raw_image_dir:
        _append_arg(cmd, "--local_dir", args.raw_image_dir)
    if args.raw_image_manifest:
        _append_arg(cmd, "--manifest", args.raw_image_manifest)
    if args.raw_image_urls_file:
        _append_arg(cmd, "--urls_file", args.raw_image_urls_file)
    _append_arg(cmd, "--out_dir", out_dir)
    _append_arg(cmd, "--source_name", args.raw_image_source_name)
    _append_arg(cmd, "--max_images", args.raw_image_max_images)
    _append_arg(cmd, "--max_candidates", args.raw_image_max_candidates)
    _append_arg(cmd, "--min_quality", args.raw_image_min_quality)
    _append_arg(cmd, "--min_width", args.raw_image_min_width)
    _append_arg(cmd, "--min_height", args.raw_image_min_height)
    _append_arg(cmd, "--max_aspect_ratio", args.raw_image_max_aspect_ratio)
    _append_arg(cmd, "--val_frac", args.raw_image_val_frac)
    _append_arg(cmd, "--device", args.device)
    _append_arg(cmd, "--visual_tokenizer", args.visual_tokenizer)
    _append_arg(cmd, "--image_codebook_size", args.raw_image_codebook_size)
    _append_arg(cmd, "--grid", args.raw_image_grid)
    _append_arg(cmd, "--max_image_tokens", args.raw_image_max_tokens)
    _append_arg(cmd, "--need_profile", args.raw_image_need_profile)
    _append_arg(cmd, "--need_steps", args.raw_image_need_steps)
    _append_arg(cmd, "--need_train_samples", args.raw_image_need_train_samples)
    _append_arg(cmd, "--need_batch_size", args.raw_image_need_batch_size)
    _append_arg(cmd, "--kernel_backend", args.kernel_backend)
    _append_arg(cmd, "--seed", args.seed)
    _bool_flag(cmd, "--download", bool(args.raw_image_download))
    _bool_flag(cmd, "--copy_local", bool(args.raw_image_copy_local))
    _bool_flag(cmd, "--train_visual_tokenizer", bool(args.raw_image_train_visual_tokenizer))
    _bool_flag(cmd, "--train_need", bool(args.raw_image_train_need))
    _bool_flag(cmd, "--dry_run", bool(args.dry_run))
    return cmd

def run(args: argparse.Namespace) -> Dict[str, Any]:
    args = _load_config_into_args(args)
    out_root = Path(args.out_root).resolve()
    corpus_dir = out_root / "corpuses"
    logs_dir = out_root / "logs"
    control_file = corpus_dir / "rl" / "low_data_control_interactions.synthetic.jsonl"
    need_adapter_dir = out_root / "need_control_adapter"
    sidecar_dataset_dir = out_root / "sidecar_alignment_dataset"
    sidecar_alignment_dir = out_root / "sidecar_need_latent"
    runtime_profile = out_root / "runtime_profile.json"
    image_policy_dir = out_root / "image_policy_adapter"
    raw_image_dir = out_root / "raw_image_corpus"
    out_root.mkdir(parents=True, exist_ok=True)
    _write_json(out_root / "start_config.redacted.json", _redacted_config(args))
    env = _api_env(args)
    records: List[Dict[str, Any]] = []

    if args.write_config:
        _write_json(Path(args.write_config), _redacted_config(args))
        if args.config_only:
            return {"done": True, "config_written": args.write_config}

    if not args.skip_generate:
        records.append(_stream_command(build_generation_cmd(args, corpus_dir, control_file), logs_dir / "01_generate.log", env, SCRIPT_DIR, args.continue_on_error))
    if args.dry_run:
        manifest = {"done": True, "dry_run": True, "out_root": str(out_root), "records": records, "next_step": "run again without --dry_run to generate and feed immediately"}
        _write_json(out_root / "start_manifest.json", manifest)
        return manifest

    if not args.skip_need_tune:
        if not args.need_checkpoint:
            raise ValueError("--need_checkpoint is required unless --skip_need_tune is set")
        if not control_file.exists():
            raise FileNotFoundError(f"control interactions file not found: {control_file}")
        records.append(_stream_command(build_need_tune_cmd(args, control_file, need_adapter_dir), logs_dir / "02_need_control_tune.log", env, SCRIPT_DIR, args.continue_on_error))

    sidecar_alignment_used: Optional[Path] = None
    if not args.skip_sidecar_alignment:
        if not args.need_checkpoint:
            raise ValueError("--need_checkpoint is required for sidecar alignment")
        if str(getattr(args, "sidecar_backend", "external_lm")) == "need":
            records.append(_stream_command(build_train_need_sidecar_cmd(args, corpus_dir, sidecar_alignment_dir), logs_dir / "03_train_need_sidecar.log", env, SCRIPT_DIR, args.continue_on_error))
            sidecar_alignment_used = sidecar_alignment_dir
        else:
            if not args.sidecar_model:
                raise ValueError("--sidecar_model is required for external LM sidecar alignment unless --skip_sidecar_alignment is set")
            align_seed = corpus_dir / "rl" / "sidecar_latent_alignment.synthetic.jsonl"
            if not align_seed.exists():
                raise FileNotFoundError(f"sidecar alignment seed rows not found: {align_seed}")
            records.append(_stream_command(build_alignment_dataset_cmd(args, corpus_dir, sidecar_dataset_dir), logs_dir / "03_build_sidecar_alignment_dataset.log", env, SCRIPT_DIR, args.continue_on_error))
            records.append(_stream_command(build_train_alignment_cmd(args, sidecar_dataset_dir, sidecar_alignment_dir), logs_dir / "04_train_sidecar_alignment.log", env, SCRIPT_DIR, args.continue_on_error))
            sidecar_alignment_used = sidecar_alignment_dir

    if not args.skip_image_policy:
        img_cmd = build_image_policy_cmd(args, image_policy_dir)
        if img_cmd is not None:
            records.append(_stream_command(img_cmd, logs_dir / "05_image_policy_tune.log", env, SCRIPT_DIR, args.continue_on_error))

    if not args.skip_runtime_profile:
        records.append(_stream_command(build_runtime_profile_cmd(args, control_file, runtime_profile, sidecar_alignment_used), logs_dir / "06_runtime_profile.log", env, SCRIPT_DIR, args.continue_on_error))
        if str(getattr(args, "sidecar_backend", "external_lm")) == "need" and sidecar_alignment_used is not None:
            _inject_need_sidecar_runtime_profile(runtime_profile, sidecar_alignment_used)

    if not args.skip_raw_image_data:
        raw_cmd = build_raw_image_start_cmd(args, raw_image_dir)
        if raw_cmd is not None:
            records.append(_stream_command(raw_cmd, logs_dir / "07_raw_image_data.log", env, SCRIPT_DIR, args.continue_on_error))

    manifest = {
        "done": True,
        "out_root": str(out_root),
        "corpuses": str(corpus_dir),
        "control_interactions": str(control_file),
        "need_control_adapter": str(need_adapter_dir) if need_adapter_dir.exists() else "",
        "sidecar_alignment": str(sidecar_alignment_dir) if sidecar_alignment_dir.exists() else "",
        "image_policy_adapter": str(image_policy_dir) if image_policy_dir.exists() else "",
        "raw_image_corpus": str(raw_image_dir) if raw_image_dir.exists() else "",
        "runtime_profile": str(runtime_profile) if runtime_profile.exists() else "",
        "logs": str(logs_dir),
        "records": records,
    }
    _write_json(out_root / "start_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate low-data RL data and immediately feed it into NEED/sidecar tuning")
    p.add_argument("action", nargs="?", choices=["start"], default="start", help="Run the one-command generation/feed pipeline")
    p.add_argument("--config", default="", help="Optional JSON config produced by --write_config")
    p.add_argument("--write_config", default="", help="Write a redacted config JSON for repeatable one-command runs")
    p.add_argument("--config_only", action="store_true", help="Only write config and exit")
    p.add_argument("--dry_run", action="store_true", help="Write plan/prompts but do not call the generation API or tune adapters")
    p.add_argument("--api_key", default="", help="API key literal. Prefer environment variables in shared environments.")
    p.add_argument("--api_key_env", default="OPENAI_API_KEY")
    p.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    p.add_argument("--api_format", choices=["chat", "responses"], default="chat")
    p.add_argument("--model", default="gpt-5.4")
    p.add_argument("--need_checkpoint", default="", help="Finished NEED checkpoint directory")
    p.add_argument("--sidecar_model", default="", help="Uploaded/full external LM sidecar model path or Hugging Face model id")
    p.add_argument("--sidecar_backend", choices=["external_lm", "need"], default="external_lm", help="Train/export exactly one sidecar backend for runtime use")
    p.add_argument("--need_sidecar_target_params", default="", help="Optional smaller NEED sidecar size such as 30M or 100M")
    p.add_argument("--need_sidecar_student_scale", type=float, default=0.5)
    p.add_argument("--need_sidecar_steps", type=int, default=600)
    p.add_argument("--need_sidecar_samples", type=int, default=2000)
    p.add_argument("--need_sidecar_batch_size", type=int, default=4)
    p.add_argument("--need_sidecar_lr", type=float, default=2e-4)
    p.add_argument("--out_root", default="runs/need_low_data_rl_start")
    p.add_argument("--target_profile", choices=["tiny", "balanced", "whole_model", "reasoning", "style", "safety"], default="whole_model")
    p.add_argument("--total_examples", default="3000")
    p.add_argument("--max_total_examples", type=int, default=3000)
    p.add_argument("--batch_size", default="50")
    p.add_argument("--wikipedia_source", default="popular,random,vital")
    p.add_argument("--topic_pool_size", type=int, default=200)
    p.add_argument("--articles_per_batch", type=int, default=0)
    p.add_argument("--topics_file", default="")
    p.add_argument("--transcripts_file", default="")
    p.add_argument("--transcripts_emit_as", choices=["sft", "rlvr", "preference", "all"], default="all")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_output_tokens", type=int, default=6000)
    p.add_argument("--device", default="auto")
    p.add_argument("--kernel_backend", default="auto")
    p.add_argument("--prefer_best", action="store_true")
    p.add_argument("--need_steps", type=int, default=300)
    p.add_argument("--need_batch_size", type=int, default=4)
    p.add_argument("--need_lr", type=float, default=5e-5)
    p.add_argument("--train_revision", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--train_pathway", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--augment_factor", type=int, default=3)
    p.add_argument("--alignment_corpus", default="", help="Optional extra text corpus for sidecar latent alignment")
    p.add_argument("--sidecar_max_examples", type=int, default=3000)
    p.add_argument("--sidecar_train_mode", choices=["lora", "full", "projection_only"], default="lora")
    p.add_argument("--sidecar_steps", type=int, default=600)
    p.add_argument("--sidecar_batch_size", type=int, default=4)
    p.add_argument("--sidecar_lr", type=float, default=2e-5)
    p.add_argument("--sidecar_dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--sidecar_trust_remote_code", action="store_true")
    p.add_argument("--use_sidecar_latent_alignment", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sidecar_latent_alignment_weight", type=float, default=0.35)
    p.add_argument("--image_preferences", default="", help="Optional real image preference JSONL with chosen/rejected image paths; no images are generated")
    p.add_argument("--visual_tokenizer", default="", help="Visual tokenizer directory for optional real image preference feeding")
    p.add_argument("--image_root", default="")
    p.add_argument("--image_steps", type=int, default=400)
    p.add_argument("--image_batch_size", type=int, default=2)
    p.add_argument("--image_lr", type=float, default=5e-5)
    p.add_argument("--raw_image_dir", default="", help="Optional local raw-image folder to prepare/tokenize/feed into NEED image pretraining")
    p.add_argument("--raw_image_manifest", default="", help="Optional JSONL/JSON/parquet manifest with image paths or URLs")
    p.add_argument("--raw_image_urls_file", default="", help="Optional text file of image URLs")
    p.add_argument("--raw_image_source_name", default="")
    p.add_argument("--raw_image_download", action="store_true")
    p.add_argument("--raw_image_copy_local", action="store_true")
    p.add_argument("--raw_image_max_images", type=int, default=0)
    p.add_argument("--raw_image_max_candidates", type=int, default=0)
    p.add_argument("--raw_image_min_quality", type=float, default=0.02)
    p.add_argument("--raw_image_min_width", type=int, default=128)
    p.add_argument("--raw_image_min_height", type=int, default=128)
    p.add_argument("--raw_image_max_aspect_ratio", type=float, default=3.0)
    p.add_argument("--raw_image_val_frac", type=float, default=0.02)
    p.add_argument("--raw_image_codebook_size", type=int, default=512)
    p.add_argument("--raw_image_grid", type=int, default=16)
    p.add_argument("--raw_image_max_tokens", type=int, default=1024)
    p.add_argument("--raw_image_train_visual_tokenizer", action="store_true")
    p.add_argument("--raw_image_train_need", action="store_true")
    p.add_argument("--raw_image_need_profile", choices=["custom", "tiny", "small", "medium", "large", "reasoning", "image", "speed", "long_context", "multimodal", "agentic"], default="image")
    p.add_argument("--raw_image_need_steps", type=int, default=1000)
    p.add_argument("--raw_image_need_train_samples", type=int, default=10000)
    p.add_argument("--raw_image_need_batch_size", type=int, default=4)
    p.add_argument("--skip_generate", action="store_true")
    p.add_argument("--skip_need_tune", action="store_true")
    p.add_argument("--skip_sidecar_alignment", action="store_true")
    p.add_argument("--skip_image_policy", action="store_true")
    p.add_argument("--skip_raw_image_data", action="store_true")
    p.add_argument("--skip_runtime_profile", action="store_true")
    p.add_argument("--continue_on_error", action="store_true")
    p.add_argument("--seed", type=int, default=123)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
