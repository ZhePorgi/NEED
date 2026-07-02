#!/usr/bin/env python3
"""Endgame safety/reliability audit for NEED code, datasets, and run configs.

This is a local static/runtime checklist intended to catch things that can ruin a
long run: unsafe checkpoint loading, recipe/model-shape drift, corrupt packed
indexes, latent tool defaults, syntax failures, and common training guardrail
regressions. It does not send data anywhere.
"""
from __future__ import annotations

import argparse
import json
import py_compile
import re
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from need_core import NeedConfig
from training_recipes import available_recipes
from train import verify_packed_index_integrity, _code_fingerprint

ARCH_PREFIXES = (
    "d_model", "n_layers", "n_heads", "d_ff", "block_size", "n_experts", "moe_", "energy_",
    "memory_", "pathway_", "planner_", "image_", "exact_recall_", "state_", "kernel_backend",
)
TRAIN_ONLY_ALLOW = {
    "lr", "lr_schedule", "warmup_steps", "min_lr", "lr_decay_steps", "beta1", "beta2", "weight_decay", "grad_clip", "target_effective_batch_tokens",
    "eval_interval", "eval_batches", "save_interval", "sample_interval", "module_diagnostics_interval",
    "nan_recovery", "loss_spike_threshold", "max_nonfinite_events", "ema_decay", "auto_optimize",
    "auto_batch", "compile", "compile_mode", "compile_cudagraphs", "compile_dynamic",
    "prefetch_to_device", "drop_last", "minimal_aux_metrics",
}
RISK_PATTERNS = [
    ("unsafe_torch_load", re.compile(r"torch\.load\([^\n]*weights_only\s*=\s*False")),
    ("pickle_fallback", re.compile(r"torch\.load\([^\n]*\)")),
    ("shell_true", re.compile(r"shell\s*=\s*True")),
    ("os_system", re.compile(r"os\.system\(")),
    ("eval_exec", re.compile(r"(?<![A-Za-z0-9_.])(?:eval|exec)\s*\(")),
]


def issue(severity: str, code: str, message: str, path: str = "", line: int = 0) -> Dict[str, Any]:
    return {"severity": severity, "code": code, "message": message, "path": path, "line": int(line)}


def py_compile_check(root: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for fp in sorted(root.glob("*.py")):
        try:
            py_compile.compile(str(fp), doraise=True)
        except Exception as exc:
            out.append(issue("error", "py_compile_failed", str(exc), str(fp)))
    return out


def static_security_scan(root: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    allow_pickle_fallback_files = {
        "checkpoint_tools.py", "generate.py", "need_core.py", "need_image.py", "need_image_rl.py",
        "need_experience_replay.py", "need_low_data_adapters.py", "need_thought_distill.py", "sidecar_lm_runtime.py",
    }
    for fp in sorted(root.glob("*.py")):
        text = fp.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            for code, pat in RISK_PATTERNS:
                if not pat.search(line):
                    continue
                if code == "pickle_fallback" and "weights_only=True" in line:
                    continue
                if code == "pickle_fallback" and fp.name == "train.py" and "allow_unsafe" in text[max(0, text.find(line)-800): text.find(line)+800]:
                    continue
                sev = "warning"
                if code == "unsafe_torch_load":
                    sev = "error"
                if code == "pickle_fallback" and fp.name in allow_pickle_fallback_files:
                    sev = "info"
                out.append(issue(sev, code, line.strip()[:220], str(fp), i))
    return out


def recipe_check() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    arch_names = {f.name for f in fields(NeedConfig)}
    for name, recipe in available_recipes().items():
        if name == "none":
            continue
        for key in recipe:
            if key in arch_names or key.startswith(ARCH_PREFIXES):
                out.append(issue("error", "recipe_changes_architecture", f"recipe {name} sets architecture field {key}"))
            elif key not in TRAIN_ONLY_ALLOW:
                out.append(issue("warning", "recipe_unknown_training_field", f"recipe {name} sets unclassified field {key}"))
    return out


def latent_tool_default_check(root: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for name in ("generate.py", "browser.py", "need_latent_tools.py"):
        p = root / name
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if name in ("generate.py", "browser.py") and 'latent_tool_python", action=argparse.BooleanOptionalAction, default=True' in text:
            out.append(issue("error", "latent_python_default_on", f"{name} enables latent Python by default", str(p)))
        if name == "need_latent_tools.py" and "python_enabled: bool = True" in text:
            out.append(issue("error", "latent_python_config_default_on", "LatentToolConfig enables Python by default", str(p)))
    return out


def training_guard_check(root: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    text = (root / "train.py").read_text(encoding="utf-8", errors="replace")
    required = {
        "safe_checkpoint_load": "weights_only=True",
        "unsafe_load_flag": "allow_unsafe_checkpoint_load",
        "signal_checkpoint": "signal_checkpoint_requested",
        "packed_integrity": "verify_packed_index_integrity",
        "grad_watchdog": "skipped_bad_grad",
        "code_fingerprint": "code_fingerprint",
        "max_nonfinite_cap": "max_nonfinite_events",
    }
    for code, needle in required.items():
        if needle not in text:
            out.append(issue("error", f"missing_{code}", f"train.py is missing guardrail marker {needle}"))
    return out


def packed_index_check(path: str) -> List[Dict[str, Any]]:
    if not path:
        return []
    try:
        report = verify_packed_index_integrity(Path(path), strict=True)
        return [issue("info", "packed_index_ok", json.dumps(report, sort_keys=True), path)]
    except Exception as exc:
        return [issue("error", "packed_index_failed", str(exc), path)]


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Audit NEED codebase and optional packed index for endgame run readiness")
    p.add_argument("--root", default=".")
    p.add_argument("--packed_index", default="")
    p.add_argument("--json_out", default="")
    p.add_argument("--fail_on_warning", action="store_true")
    args = p.parse_args(argv)
    root = Path(args.root).resolve()
    findings: List[Dict[str, Any]] = []
    findings += py_compile_check(root)
    findings += static_security_scan(root)
    findings += recipe_check()
    findings += latent_tool_default_check(root)
    findings += training_guard_check(root)
    findings += packed_index_check(args.packed_index)
    counts: Dict[str, int] = {}
    for f in findings:
        counts[f["severity"]] = counts.get(f["severity"], 0) + 1
    report = {
        "format": "need_endgame_audit_v1",
        "root": str(root),
        "code_fingerprint": _code_fingerprint(root).get("sha256"),
        "counts": counts,
        "ok": counts.get("error", 0) == 0 and (not args.fail_on_warning or counts.get("warning", 0) == 0),
        "findings": findings,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
