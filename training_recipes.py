#!/usr/bin/env python3
"""Training recipe presets for NEED.

Recipes intentionally tune only optimizer/runtime/eval/checkpoint/data controls.
They do not change model architecture fields such as width, depth, slots, ranks,
or objective modules.  Use --profile for model shape and --recipe for how to run
that shape.
"""
from __future__ import annotations

from typing import Dict, Mapping, MutableMapping

RECIPES: Dict[str, Dict[str, object]] = {
    "none": {},
    "need_0_6b_dense_85b_baseline": {
        "lr": 1.6e-4,
        "lr_schedule": "cosine",
        "warmup_steps": 2_000,
        "min_lr": 1.6e-5,
        "weight_decay": 0.10,
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip": 1.0,
        "target_effective_batch_tokens": 1_048_576,
        "eval_interval": 50_000,
        "eval_batches": 32,
        "save_interval": 25_000,
        "sample_interval": 25_000,
        "module_diagnostics_interval": 5_000,
        "nan_recovery": True,
        "max_nonfinite_events": 20,
        "ema_decay": 0.0,
    },
    "need_0_6b_dense_85b_fast": {
        "lr": 1.6e-4,
        "lr_schedule": "cosine",
        "warmup_steps": 2_000,
        "min_lr": 1.6e-5,
        "weight_decay": 0.10,
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip": 1.0,
        "target_effective_batch_tokens": 1_048_576,
        "eval_interval": 64_849,
        "eval_batches": 16,
        "save_interval": 50_000,
        "sample_interval": 50_000,
        "module_diagnostics_interval": 10_000,
        "auto_optimize": True,
        "auto_batch": True,
        "compile": True,
        "compile_mode": "max-autotune",
        "compile_cudagraphs": True,
        "compile_dynamic": False,
        "prefetch_to_device": True,
        "drop_last": True,
        "minimal_aux_metrics": True,
        "nan_recovery": True,
        "max_nonfinite_events": 20,
        "ema_decay": 0.0,
    },
    "need_0_6b_dense_85b_quality": {
        "lr": 1.35e-4,
        "lr_schedule": "cosine",
        "warmup_steps": 3_000,
        "min_lr": 1.35e-5,
        "weight_decay": 0.10,
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip": 0.8,
        "target_effective_batch_tokens": 1_572_864,
        "eval_interval": 25_000,
        "eval_batches": 64,
        "save_interval": 12_500,
        "sample_interval": 12_500,
        "module_diagnostics_interval": 2_500,
        "nan_recovery": True,
        "loss_spike_threshold": 4.0,
        "max_nonfinite_events": 20,
        "ema_decay": 0.999,
    },
    "need_0_6b_dense_85b_debug": {
        "lr": 1.0e-4,
        "lr_schedule": "constant",
        "warmup_steps": 0,
        "min_lr": 0.0,
        "weight_decay": 0.10,
        "target_effective_batch_tokens": 131_072,
        "eval_interval": 100,
        "eval_batches": 4,
        "save_interval": 100,
        "sample_interval": 100,
        "module_diagnostics_interval": 20,
        "nan_recovery": True,
        "loss_spike_threshold": 3.0,
        "max_nonfinite_events": 20,
        "ema_decay": 0.0,
        "compile": False,
    },
}

ALIASES = {
    "baseline": "need_0_6b_dense_85b_baseline",
    "fast": "need_0_6b_dense_85b_fast",
    "quality": "need_0_6b_dense_85b_quality",
    "debug": "need_0_6b_dense_85b_debug",
}


def available_recipes() -> Dict[str, Dict[str, object]]:
    return dict(RECIPES)


def resolve_recipe_name(name: str) -> str:
    key = str(name or "none").strip()
    return ALIASES.get(key, key)


def apply_training_recipe(args: object, name: str) -> Dict[str, object]:
    key = resolve_recipe_name(name)
    if key not in RECIPES:
        valid = ", ".join(sorted(k for k in RECIPES if k != "none"))
        raise ValueError(f"unknown training recipe {name!r}; valid recipes: {valid}")
    applied: Dict[str, object] = {}
    explicit = set(getattr(args, "_explicit_args", set()) or set())
    for field, value in RECIPES[key].items():
        if hasattr(args, field) and field not in explicit:
            setattr(args, field, value)
            applied[field] = value
    setattr(args, "recipe", key)
    return applied
