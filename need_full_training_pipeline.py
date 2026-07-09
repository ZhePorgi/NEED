#!/usr/bin/env python3
"""End-to-end NEED local training pipeline.

User flow this script automates:
  1. Build/download the full corpus plan with optional parameter-scaled sizing.
  2. Generate low-level RL / TD seed data before base training so the final
     curriculum phase can see it, with RL/TD material kept at the end.
  3. Train NEED through a staged curriculum, continuing each phase from the
     prior checkpoint.
  4. After NEED is trained, feed low-data RL while training the sidecar latent
     alignment adapter against NEED's latent space.
  5. Export a runtime profile and optionally launch the dark browser UI on the
     local GPU.

The script is an orchestrator. It delegates actual work to build_corpuses.py,
need_auto_low_data_rl.py, train_curriculum.py, need_low_data_rl_start.py,
need_raw_image_data.py, and browser.py.
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


def _append(cmd: List[str], name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str) and value == "":
        return
    if isinstance(value, bool):
        if value:
            cmd.append(name)
        return
    cmd.extend([name, str(value)])


def _redacted(cmd: Sequence[str]) -> List[str]:
    out: List[str] = []
    hide_next = False
    for x in cmd:
        if hide_next:
            out.append("<redacted>")
            hide_next = False
            continue
        out.append(x)
        if x == "--api_key":
            hide_next = True
    return out


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _run(cmd: Sequence[str], log_path: Path, env: Dict[str, str], cwd: Path, dry_run: bool = False) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"cmd": _redacted(cmd), "log": str(log_path), "dry_run": bool(dry_run)}
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps(record, ensure_ascii=False) + "\n")
        if dry_run:
            print("DRY:", " ".join(_redacted(cmd)), flush=True)
            return {**record, "returncode": 0}
        started = time.time()
        proc = subprocess.Popen(list(cmd), cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        rc = proc.wait()
    record.update({"returncode": rc, "seconds": round(time.time() - started, 3)})
    if rc != 0:
        raise RuntimeError(f"command failed with code {rc}; see {log_path}")
    return record


def _api_env(args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    if args.base_url:
        env["OPENAI_BASE_URL"] = args.base_url
    key = args.api_key or env.get(args.api_key_env, "")
    if key:
        env[args.api_key_env] = key
    no_api_actions = {"status", "audit", "eval", "card"}
    if getattr(args, "action", "start") not in no_api_actions and not args.dry_run and not args.skip_low_data_seed and not key:
        raise ValueError(f"No API key found. Pass --api_key or set {args.api_key_env}.")
    return env


def corpus_cmd(args: argparse.Namespace, corpus_dir: Path) -> List[str]:
    cmd = [sys.executable, str(SCRIPT_DIR / "build_corpuses.py"), "--build", "all", "--out_dir", str(corpus_dir)]
    _append(cmd, "--size_fit_mode", args.corpus_size_fit_mode)
    _append(cmd, "--params_m", args.params_m)
    _append(cmd, "--tokens_per_param", args.tokens_per_param)
    _append(cmd, "--target_total_tokens", args.target_total_tokens)
    _append(cmd, "--scale", args.corpus_scale)
    _append(cmd, "--seed", args.seed)
    _append(cmd, "--max_rows_per_slice", args.max_rows_per_slice)
    _append(cmd, "--max_dedup_keys", args.max_dedup_keys)
    if args.tokenizer:
        _append(cmd, "--tokenizer", args.tokenizer)
    else:
        cmd.append("--approx_tokens")
    if args.clean_corpus:
        cmd.append("--clean")
    if args.local_override_json:
        _append(cmd, "--local_override_json", args.local_override_json)
    return cmd


def raw_image_cmd(args: argparse.Namespace, raw_dir: Path) -> Optional[List[str]]:
    if not (args.raw_image_dir or args.raw_image_manifest or args.raw_image_urls_file):
        return None
    cmd = [sys.executable, str(SCRIPT_DIR / "need_raw_image_data.py"), "start", "--out_dir", str(raw_dir)]
    _append(cmd, "--local_dir", args.raw_image_dir)
    _append(cmd, "--manifest", args.raw_image_manifest)
    _append(cmd, "--urls_file", args.raw_image_urls_file)
    _append(cmd, "--source_name", args.raw_image_source_name)
    _append(cmd, "--max_images", args.raw_image_max_images)
    _append(cmd, "--max_candidates", args.raw_image_max_candidates)
    _append(cmd, "--min_quality", args.raw_image_min_quality)
    _append(cmd, "--min_width", args.raw_image_min_width)
    _append(cmd, "--min_height", args.raw_image_min_height)
    _append(cmd, "--device", args.device)
    _append(cmd, "--visual_tokenizer", args.visual_tokenizer)
    _append(cmd, "--grid", args.raw_image_grid)
    _append(cmd, "--max_image_tokens", args.raw_image_max_tokens)
    _append(cmd, "--seed", args.seed)
    if args.raw_image_download:
        cmd.append("--download")
    if args.raw_image_copy_local:
        cmd.append("--copy_local")
    if args.raw_image_train_visual_tokenizer:
        cmd.append("--train_visual_tokenizer")
    if args.dry_run:
        cmd.append("--dry_run")
    return cmd


def low_data_seed_cmd(args: argparse.Namespace, corpus_dir: Path, control_file: Path) -> List[str]:
    cmd = [sys.executable, str(SCRIPT_DIR / "need_auto_low_data_rl.py")]
    if not args.dry_run:
        cmd.append("--activate")
    _append(cmd, "--model", args.model)
    _append(cmd, "--base_url", args.base_url)
    _append(cmd, "--api_key_env", args.api_key_env)
    if args.api_key:
        _append(cmd, "--api_key", args.api_key)
    _append(cmd, "--api_format", args.api_format)
    _append(cmd, "--out_dir", corpus_dir)
    _append(cmd, "--target_profile", args.low_data_profile)
    _append(cmd, "--total_examples", args.low_data_examples)
    _append(cmd, "--max_total_examples", max(int(args.max_low_data_examples), int(args.low_data_examples) if str(args.low_data_examples).isdigit() else 3000))
    _append(cmd, "--batch_size", args.low_data_batch_size)
    _append(cmd, "--wikipedia_source", args.wikipedia_source)
    _append(cmd, "--topic_pool_size", args.topic_pool_size)
    _append(cmd, "--control_interactions_file", control_file)
    _append(cmd, "--seed", args.seed)
    if args.topics_file:
        _append(cmd, "--topics_file", args.topics_file)
    if args.transcripts_file:
        _append(cmd, "--transcripts_file", args.transcripts_file)
        _append(cmd, "--transcripts_emit_as", "all")
    return cmd


def curriculum_cmd(args: argparse.Namespace, corpus_dir: Path, train_dir: Path, image_tokens: str) -> List[str]:
    cmd = [sys.executable, str(SCRIPT_DIR / "train_curriculum.py"), "--corpus_dir", str(corpus_dir), "--out_dir", str(train_dir)]
    _append(cmd, "--params_m", args.params_m)
    _append(cmd, "--tokens", int(float(args.params_m) * 1_000_000 * float(args.tokens_per_param)))
    _append(cmd, "--total_steps", args.total_steps)
    _append(cmd, "--batch_size", args.batch_size)
    _append(cmd, "--grad_accum_steps", args.grad_accum_steps)
    _append(cmd, "--lr", args.lr)
    _append(cmd, "--device", args.device)
    _append(cmd, "--max_phase_lines", args.max_phase_lines)
    _append(cmd, "--log_interval", args.log_interval)
    _append(cmd, "--eval_interval", args.eval_interval)
    _append(cmd, "--image_tokens", image_tokens)
    _append(cmd, "--visual_tokenizer", args.visual_tokenizer)
    _append(cmd, "--image_ratio", args.image_ratio)
    if args.extra_train_args:
        _append(cmd, "--extra_train_args", args.extra_train_args)
    if not args.no_continue_phases:
        cmd.append("--continue_phases")
    else:
        cmd.append("--no-continue_phases")
    if args.include_rl_td_end:
        cmd.append("--include_rl_td_end")
    else:
        cmd.append("--no-include_rl_td_end")
    if not args.dry_run:
        cmd.append("--run")
    return cmd


def posttrain_cmd(args: argparse.Namespace, final_ckpt: str, out_dir: Path) -> List[str]:
    cmd = [sys.executable, str(SCRIPT_DIR / "need_low_data_rl_start.py"), "start", "--out_root", str(out_dir), "--need_checkpoint", final_ckpt]
    _append(cmd, "--sidecar_model", args.sidecar_model)
    _append(cmd, "--model", args.model)
    _append(cmd, "--base_url", args.base_url)
    _append(cmd, "--api_key_env", args.api_key_env)
    if args.api_key:
        _append(cmd, "--api_key", args.api_key)
    _append(cmd, "--api_format", args.api_format)
    _append(cmd, "--target_profile", args.low_data_profile)
    _append(cmd, "--total_examples", args.posttrain_low_data_examples)
    _append(cmd, "--max_total_examples", max(int(args.max_low_data_examples), int(args.posttrain_low_data_examples) if str(args.posttrain_low_data_examples).isdigit() else 3000))
    _append(cmd, "--batch_size", args.low_data_batch_size)
    _append(cmd, "--wikipedia_source", args.wikipedia_source)
    _append(cmd, "--topics_file", args.topics_file)
    _append(cmd, "--transcripts_file", args.transcripts_file)
    _append(cmd, "--device", args.device)
    _append(cmd, "--kernel_backend", args.kernel_backend)
    _append(cmd, "--sidecar_train_mode", args.sidecar_train_mode)
    _append(cmd, "--sidecar_steps", args.sidecar_steps)
    _append(cmd, "--sidecar_batch_size", args.sidecar_batch_size)
    _append(cmd, "--sidecar_lr", args.sidecar_lr)
    _append(cmd, "--sidecar_dtype", args.sidecar_dtype)
    _append(cmd, "--sidecar_max_examples", args.sidecar_max_examples)
    _append(cmd, "--need_steps", args.need_llrl_steps)
    _append(cmd, "--need_batch_size", args.need_llrl_batch_size)
    _append(cmd, "--visual_tokenizer", args.visual_tokenizer)
    _append(cmd, "--image_preferences", args.image_preferences)
    _append(cmd, "--seed", args.seed)
    cmd.append("--skip_raw_image_data")
    if args.dry_run:
        cmd.append("--dry_run")
    if args.sidecar_trust_remote_code:
        cmd.append("--sidecar_trust_remote_code")
    return cmd


def browser_cmd(args: argparse.Namespace, final_ckpt: str, runtime_profile: Path) -> List[str]:
    cmd = [sys.executable, str(SCRIPT_DIR / "browser.py"), "--checkpoint", final_ckpt, "--runtime_profile", str(runtime_profile), "--device", args.device, "--kernel_backend", args.kernel_backend, "--host", args.host, "--port", str(args.port), "--prefer_best"]
    if args.visual_tokenizer:
        _append(cmd, "--visual_tokenizer", args.visual_tokenizer)
    return cmd


def expected_final_checkpoint(train_dir: Path, include_rl_td_end: bool = True) -> str:
    plan_path = train_dir / "curriculum_plan.json"
    if plan_path.exists():
        try:
            raw = json.loads(plan_path.read_text(encoding="utf-8"))
            if raw.get("final_checkpoint"):
                return str(raw["final_checkpoint"])
        except Exception:
            pass
    return str(train_dir / ("04_final_rl_td_low_data" if include_rl_td_end else "03_general_alignment"))




def audit_cmd(args: argparse.Namespace, corpus_dir: Path, raw_dir: Path, audit_dir: Path) -> List[str]:
    paths = [str(corpus_dir)]
    token_dir = raw_dir / "tokens"
    if token_dir.exists() or (args.raw_image_dir or args.raw_image_manifest or args.raw_image_urls_file):
        paths.append(str(token_dir))
    cmd = [
        sys.executable, str(SCRIPT_DIR / "need_dataset_audit.py"), *paths,
        "--out_json", str(audit_dir / "audit_report.json"),
        "--out_html", str(audit_dir / "audit_report.html"),
        "--failures_jsonl", str(audit_dir / "audit_failures.jsonl"),
        "--expect_full_pipeline",
    ]
    _append(cmd, "--max_rows_per_file", args.audit_max_rows_per_file)
    if args.strict_audit:
        cmd.append("--strict")
    return cmd


def run_card_cmd(args: argparse.Namespace, out_root: Path, audit_dir: Path, eval_dir: Path, post_dir: Path) -> List[str]:
    return [
        sys.executable, str(SCRIPT_DIR / "need_run_card.py"),
        "--run_root", str(out_root),
        "--audit_json", str(audit_dir / "audit_report.json"),
        "--scorecard_json", str(eval_dir / "scorecard.json"),
        "--runtime_profile", str(post_dir / "runtime_profile.json"),
        "--version_label", args.version_label,
    ]


def _load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"version": 1, "stages": {}, "history": []}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw.setdefault("version", 1)
            raw.setdefault("stages", {})
            raw.setdefault("history", [])
            return raw
    except Exception:
        pass
    return {"version": 1, "stages": {}, "history": []}


def _save_state(path: Path, state: Dict[str, Any]) -> None:
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_json(path, state)


def _stage_done(state: Dict[str, Any], name: str) -> bool:
    obj = state.get("stages", {}).get(name, {})
    return isinstance(obj, dict) and obj.get("status") == "succeeded"


def _run_stage(
    *,
    name: str,
    cmd: Optional[Sequence[str]],
    log_path: Path,
    env: Dict[str, str],
    cwd: Path,
    state_path: Path,
    state: Dict[str, Any],
    records: List[Dict[str, Any]],
    resume: bool,
    dry_run: bool,
    skipped_reason: str = "",
) -> Optional[Dict[str, Any]]:
    state.setdefault("stages", {})
    if resume and _stage_done(state, name):
        rec = {"stage": name, "skipped": True, "reason": "already_succeeded", "log": str(log_path)}
        records.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        return rec
    if cmd is None:
        state["stages"][name] = {"status": "skipped", "reason": skipped_reason, "updated_at": time.time()}
        _save_state(state_path, state)
        rec = {"stage": name, "skipped": True, "reason": skipped_reason}
        records.append(rec)
        return rec
    state["stages"][name] = {"status": "running", "cmd": _redacted(cmd), "log": str(log_path), "started_at": time.time()}
    state.setdefault("history", []).append({"stage": name, "status": "running", "time": time.time()})
    _save_state(state_path, state)
    try:
        rec = _run(cmd, log_path, env, cwd, dry_run=dry_run)
        rec["stage"] = name
        state["stages"][name] = {**rec, "status": "succeeded", "finished_at": time.time()}
        state.setdefault("history", []).append({"stage": name, "status": "succeeded", "time": time.time()})
        records.append(rec)
        _save_state(state_path, state)
        return rec
    except Exception as exc:
        state["stages"][name] = {"status": "failed", "cmd": _redacted(cmd), "log": str(log_path), "error": str(exc), "finished_at": time.time()}
        state.setdefault("history", []).append({"stage": name, "status": "failed", "time": time.time(), "error": str(exc)})
        _save_state(state_path, state)
        raise


def _status(args: argparse.Namespace) -> Dict[str, Any]:
    out_root = Path(args.out_root).resolve()
    state = _load_state(out_root / "pipeline_state.json")
    manifest = {}
    mp = out_root / "full_pipeline_manifest.json"
    if mp.exists():
        try:
            manifest = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    result = {"out_root": str(out_root), "state": state, "manifest": manifest}
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_root = Path(args.out_root).resolve()
    logs = out_root / "logs"
    corpus_dir = out_root / "corpuses"
    raw_dir = out_root / "raw_image_data"
    train_dir = out_root / "need_training"
    post_dir = out_root / "posttrain_llrl_sidecar"
    audit_dir = out_root / "audit"
    eval_dir = out_root / "evals"
    control_file = corpus_dir / "rl" / "low_data_control_interactions.synthetic.jsonl"
    state_path = out_root / "pipeline_state.json"
    out_root.mkdir(parents=True, exist_ok=True)

    if args.action == "status":
        return _status(args)

    env = _api_env(args)
    records: List[Dict[str, Any]] = []
    state = _load_state(state_path)
    resume = bool(args.action == "resume" or args.resume)
    state.update({
        "out_root": str(out_root),
        "action": args.action,
        "resume": resume,
        "created_at": state.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    _write_json(out_root / "full_pipeline_config.redacted.json", {k: ("<redacted>" if k == "api_key" and v else v) for k, v in vars(args).items()})
    _save_state(state_path, state)

    if args.action == "audit":
        _run_stage(name="audit", cmd=audit_cmd(args, corpus_dir, raw_dir, audit_dir), log_path=logs / "90_audit.log", env=env, cwd=SCRIPT_DIR, state_path=state_path, state=state, records=records, resume=resume, dry_run=False)
        return {"out_root": str(out_root), "records": records}
    if args.action == "card":
        _run_stage(name="run_card", cmd=run_card_cmd(args, out_root, audit_dir, eval_dir, post_dir), log_path=logs / "92_run_card.log", env=env, cwd=SCRIPT_DIR, state_path=state_path, state=state, records=records, resume=resume, dry_run=False)
        return {"out_root": str(out_root), "records": records}

    if not args.skip_corpus:
        _run_stage(name="build_corpuses", cmd=corpus_cmd(args, corpus_dir), log_path=logs / "00_build_corpuses.log", env=env, cwd=SCRIPT_DIR, state_path=state_path, state=state, records=records, resume=resume, dry_run=args.dry_run and args.dry_plan_only)
    else:
        _run_stage(name="build_corpuses", cmd=None, log_path=logs / "00_build_corpuses.log", env=env, cwd=SCRIPT_DIR, state_path=state_path, state=state, records=records, resume=resume, dry_run=False, skipped_reason="--skip_corpus")

    image_tokens = ""
    raw_cmd = raw_image_cmd(args, raw_dir)
    if raw_cmd is not None and not args.skip_raw_images:
        _run_stage(name="raw_image_prepare", cmd=raw_cmd, log_path=logs / "01_raw_image_prepare.log", env=env, cwd=SCRIPT_DIR, state_path=state_path, state=state, records=records, resume=resume, dry_run=args.dry_run and args.dry_plan_only)
        image_tokens = str(raw_dir / "tokens" / "image_tokens.train.jsonl")
    else:
        _run_stage(name="raw_image_prepare", cmd=None, log_path=logs / "01_raw_image_prepare.log", env=env, cwd=SCRIPT_DIR, state_path=state_path, state=state, records=records, resume=resume, dry_run=False, skipped_reason="no raw image source or --skip_raw_images")

    if not args.skip_low_data_seed:
        _run_stage(name="generate_llrl_seed", cmd=low_data_seed_cmd(args, corpus_dir, control_file), log_path=logs / "02_generate_llrl_seed.log", env=env, cwd=SCRIPT_DIR, state_path=state_path, state=state, records=records, resume=resume, dry_run=args.dry_run and args.dry_plan_only)
    else:
        _run_stage(name="generate_llrl_seed", cmd=None, log_path=logs / "02_generate_llrl_seed.log", env=env, cwd=SCRIPT_DIR, state_path=state_path, state=state, records=records, resume=resume, dry_run=False, skipped_reason="--skip_low_data_seed")

    if not args.skip_audit:
        _run_stage(name="pretrain_audit", cmd=audit_cmd(args, corpus_dir, raw_dir, audit_dir), log_path=logs / "025_pretrain_audit.log", env=env, cwd=SCRIPT_DIR, state_path=state_path, state=state, records=records, resume=resume, dry_run=args.dry_run and args.dry_plan_only)
    else:
        _run_stage(name="pretrain_audit", cmd=None, log_path=logs / "025_pretrain_audit.log", env=env, cwd=SCRIPT_DIR, state_path=state_path, state=state, records=records, resume=resume, dry_run=False, skipped_reason="--skip_audit")

    if not args.skip_base_training:
        _run_stage(name="train_need_curriculum", cmd=curriculum_cmd(args, corpus_dir, train_dir, image_tokens), log_path=logs / "03_train_need_curriculum.log", env=env, cwd=SCRIPT_DIR, state_path=state_path, state=state, records=records, resume=resume, dry_run=args.dry_run and args.dry_plan_only)
        final_ckpt = expected_final_checkpoint(train_dir, include_rl_td_end=bool(args.include_rl_td_end))
    else:
        if not args.need_checkpoint:
            raise ValueError("--need_checkpoint is required when --skip_base_training is used")
        final_ckpt = args.need_checkpoint
        _run_stage(name="train_need_curriculum", cmd=None, log_path=logs / "03_train_need_curriculum.log", env=env, cwd=SCRIPT_DIR, state_path=state_path, state=state, records=records, resume=resume, dry_run=False, skipped_reason="--skip_base_training")

    if not args.skip_posttrain_llrl:
        if not args.sidecar_model:
            raise ValueError("--sidecar_model is required unless --skip_posttrain_llrl is used")
        _run_stage(name="posttrain_llrl_sidecar", cmd=posttrain_cmd(args, final_ckpt, post_dir), log_path=logs / "04_posttrain_llrl_sidecar.log", env=env, cwd=SCRIPT_DIR, state_path=state_path, state=state, records=records, resume=resume, dry_run=args.dry_run and args.dry_plan_only)
    else:
        _run_stage(name="posttrain_llrl_sidecar", cmd=None, log_path=logs / "04_posttrain_llrl_sidecar.log", env=env, cwd=SCRIPT_DIR, state_path=state_path, state=state, records=records, resume=resume, dry_run=False, skipped_reason="--skip_posttrain_llrl")

    if not args.skip_run_card:
        _run_stage(name="run_card", cmd=run_card_cmd(args, out_root, audit_dir, eval_dir, post_dir), log_path=logs / "06_run_card.log", env=env, cwd=SCRIPT_DIR, state_path=state_path, state=state, records=records, resume=resume, dry_run=args.dry_run and args.dry_plan_only)
    else:
        _run_stage(name="run_card", cmd=None, log_path=logs / "06_run_card.log", env=env, cwd=SCRIPT_DIR, state_path=state_path, state=state, records=records, resume=resume, dry_run=False, skipped_reason="--skip_run_card")

    runtime_profile = post_dir / "runtime_profile.json"
    manifest = {
        "done": True,
        "out_root": str(out_root),
        "corpuses": str(corpus_dir),
        "raw_image_data": str(raw_dir) if raw_cmd is not None else "",
        "need_training": str(train_dir),
        "final_need_checkpoint": final_ckpt,
        "posttrain_llrl_sidecar": str(post_dir),
        "audit_report": str(audit_dir / "audit_report.json"),
        "eval_scorecard": str(eval_dir / "scorecard.json"),
        "run_card": str(out_root / "run_card.md"),
        "pipeline_state": str(state_path),
        "runtime_profile": str(runtime_profile) if runtime_profile.exists() or not args.skip_posttrain_llrl else "",
        "browser_command": _redacted(browser_cmd(args, final_ckpt, runtime_profile)),
        "records": records,
    }
    _write_json(out_root / "full_pipeline_manifest.json", manifest)
    state["manifest"] = manifest
    state["status"] = "succeeded"
    _save_state(state_path, state)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)

    if args.launch_browser and not args.dry_run:
        cmd = browser_cmd(args, final_ckpt, runtime_profile)
        proc = subprocess.Popen(cmd, cwd=str(SCRIPT_DIR), env=env)
        launch = {"pid": proc.pid, "cmd": _redacted(cmd), "url": f"http://{args.host}:{args.port}"}
        _write_json(out_root / "browser_launch.json", launch)
        print(json.dumps(launch, indent=2), flush=True)
    return manifest

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Full NEED corpus -> LLRL seed -> training -> sidecar alignment -> audit/eval/card -> browser pipeline")
    p.add_argument("action", nargs="?", choices=["start", "resume", "status", "audit", "card"], default="start")
    p.add_argument("--out_root", default="runs/need_full_pipeline")
    p.add_argument("--dry_run", action="store_true", help="Use dry behavior for API generation/training scripts where supported.")
    p.add_argument("--dry_plan_only", action="store_true", help="Do not execute subprocesses; only write/log commands.")
    p.add_argument("--resume", action="store_true", help="Resume an existing run by skipping stages that already succeeded.")
    p.add_argument("--version_label", default="complete_project", help="Label used in the generated run card.")
    p.add_argument("--api_key", default="")
    p.add_argument("--api_key_env", default="OPENAI_API_KEY")
    p.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    p.add_argument("--api_format", choices=["chat", "responses"], default="chat")
    p.add_argument("--model", default="gpt-5.4")
    p.add_argument("--sidecar_model", default="", help="Uploaded sidecar model path or HF id for latent alignment.")
    p.add_argument("--need_checkpoint", default="", help="Existing NEED checkpoint when skipping base training.")

    p.add_argument("--params_m", type=float, default=30.0)
    p.add_argument("--tokens_per_param", type=float, default=320.0)
    p.add_argument("--corpus_size_fit_mode", choices=["off", "params", "tokens"], default="params")
    p.add_argument("--target_total_tokens", type=int, default=0)
    p.add_argument("--corpus_scale", type=float, default=1.0)
    p.add_argument("--tokenizer", default="")
    p.add_argument("--local_override_json", default="")
    p.add_argument("--clean_corpus", action="store_true")
    p.add_argument("--max_rows_per_slice", type=int, default=0)
    p.add_argument("--max_dedup_keys", type=int, default=25_000_000)

    p.add_argument("--low_data_profile", choices=["tiny", "balanced", "whole_model", "reasoning", "style", "safety"], default="whole_model")
    p.add_argument("--low_data_examples", default="3000")
    p.add_argument("--posttrain_low_data_examples", default="3000")
    p.add_argument("--max_low_data_examples", type=int, default=3000)
    p.add_argument("--low_data_batch_size", default="50")
    p.add_argument("--wikipedia_source", default="popular,random,vital")
    p.add_argument("--topic_pool_size", type=int, default=200)
    p.add_argument("--topics_file", default="")
    p.add_argument("--transcripts_file", default="")

    p.add_argument("--total_steps", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_phase_lines", type=int, default=0)
    p.add_argument("--include_rl_td_end", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--no_continue_phases", action="store_true")
    p.add_argument("--extra_train_args", default="")

    p.add_argument("--raw_image_dir", default="")
    p.add_argument("--raw_image_manifest", default="")
    p.add_argument("--raw_image_urls_file", default="")
    p.add_argument("--raw_image_source_name", default="")
    p.add_argument("--raw_image_download", action="store_true")
    p.add_argument("--raw_image_copy_local", action="store_true")
    p.add_argument("--raw_image_max_images", type=int, default=0)
    p.add_argument("--raw_image_max_candidates", type=int, default=0)
    p.add_argument("--raw_image_min_quality", type=float, default=0.02)
    p.add_argument("--raw_image_min_width", type=int, default=128)
    p.add_argument("--raw_image_min_height", type=int, default=128)
    p.add_argument("--raw_image_grid", type=int, default=16)
    p.add_argument("--raw_image_max_tokens", type=int, default=1024)
    p.add_argument("--raw_image_train_visual_tokenizer", action="store_true")
    p.add_argument("--visual_tokenizer", default="")
    p.add_argument("--image_ratio", type=float, default=0.15)

    p.add_argument("--sidecar_train_mode", choices=["lora", "full", "projection_only"], default="lora")
    p.add_argument("--sidecar_steps", type=int, default=600)
    p.add_argument("--sidecar_batch_size", type=int, default=4)
    p.add_argument("--sidecar_lr", type=float, default=2e-5)
    p.add_argument("--sidecar_dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--sidecar_max_examples", type=int, default=3000)
    p.add_argument("--sidecar_trust_remote_code", action="store_true")
    p.add_argument("--need_llrl_steps", type=int, default=300)
    p.add_argument("--need_llrl_batch_size", type=int, default=4)
    p.add_argument("--image_preferences", default="")

    p.add_argument("--skip_audit", action="store_true")
    p.add_argument("--strict_audit", action="store_true", help="Fail the pipeline on audit errors.")
    p.add_argument("--audit_max_rows_per_file", type=int, default=50000, help="Limit rows scanned per file during audit; 0 scans all.")
    p.add_argument("--skip_run_card", action="store_true")

    p.add_argument("--device", default="auto")
    p.add_argument("--kernel_backend", default="auto")
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", default="7860")
    p.add_argument("--launch_browser", action="store_true")

    p.add_argument("--skip_corpus", action="store_true")
    p.add_argument("--skip_raw_images", action="store_true")
    p.add_argument("--skip_low_data_seed", action="store_true")
    p.add_argument("--skip_base_training", action="store_true")
    p.add_argument("--skip_posttrain_llrl", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
