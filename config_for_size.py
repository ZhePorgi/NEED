#!/usr/bin/env python3
"""Generate NEED model, corpus, and training settings from a parameter budget.

The script accepts compact counts with K/M/B/T suffixes. It can
produce either dense or MoE NEED shapes and emits JSON plus a train.py command.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


COUNT_RE = re.compile(r"^\s*([0-9]+(?:_[0-9]{3})*(?:\.[0-9]+)?|[0-9]*\.[0-9]+)\s*([kKmMbBtT]?)\s*$")
COUNT_MULTIPLIERS = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}


def parse_scaled_number(value: Any, *, default_suffix: str = "") -> int:
    """Parse integers with optional K/M/B/T suffixes.

    Examples: 600M -> 600_000_000, 1.2B -> 1_200_000_000.
    """
    if value is None:
        return 0
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        suffix = default_suffix.upper()
        return int(round(float(value) * COUNT_MULTIPLIERS.get(suffix, 1)))
    s = str(value).strip().replace(",", "").replace("_", "")
    if not s:
        return 0
    m = COUNT_RE.match(s)
    if not m:
        raise argparse.ArgumentTypeError(f"expected a number with optional K/M/B/T suffix, got {value!r}")
    num = float(m.group(1))
    suffix = (m.group(2) or default_suffix).upper()
    if suffix not in COUNT_MULTIPLIERS:
        raise argparse.ArgumentTypeError(f"unknown suffix {suffix!r}; use K, M, B, or T")
    return int(round(num * COUNT_MULTIPLIERS[suffix]))


def parse_token_count(value: Any) -> int:
    return parse_scaled_number(value)


def parse_param_count(value: Any) -> int:
    return parse_scaled_number(value)


def format_scaled_number(n: int) -> str:
    n = int(n)
    for suffix, mult in (("T", 1_000_000_000_000), ("B", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)):
        if abs(n) >= mult and n % mult == 0:
            return f"{n // mult}{suffix}"
        if abs(n) >= mult:
            val = n / mult
            return f"{val:.3g}{suffix}"
    return str(n)


def round_multiple(x: float, multiple: int, *, minimum: int = 1) -> int:
    return max(minimum, int(round(float(x) / multiple) * multiple))


def choose_heads(d_model: int) -> int:
    """Choose a head count with head dimension near 64 and valid divisibility."""
    candidates = [h for h in range(1, 33) if d_model % h == 0]
    return min(candidates, key=lambda h: (abs((d_model / h) - 64), -h))



def derive_block_size(target_params: int) -> int:
    """Derive context length from parameter budget without named size buckets."""
    scale = max(0.25, float(target_params) / 30_000_000.0)
    raw = 512.0 * (scale ** 0.18)
    return int(max(256, min(4096, round_multiple(raw, 128, minimum=256))))


def derive_memory_chunk_size(block_size: int) -> int:
    return int(max(16, min(128, round_multiple(max(16, int(block_size) / 16), 16, minimum=16))))


def derive_image_shape(block_size: int, target_params: int) -> Dict[str, int]:
    """Derive image-token capacity from context budget instead of fixed image buckets."""
    usable_tokens = max(64, min(2048, int(block_size) // 2))
    grid = int(max(8, min(32, round_multiple(math.sqrt(float(usable_tokens)), 4, minimum=8))))
    max_grid = int(max(grid, min(48, round_multiple(grid * 4.0 / 3.0, 4, minimum=grid))))
    scale = max(0.25, float(target_params) / 30_000_000.0)
    object_slots = int(max(4, min(16, round_multiple(4.0 * (scale ** 0.12), 2, minimum=4))))
    return dict(image_grid=grid, image_max_grid=max_grid, image_max_tokens=max_grid * max_grid, object_program_slots=object_slots)


def derive_long_context_shape(model_cfg: Dict[str, Any], target_params: int) -> Dict[str, int]:
    scale = max(0.25, float(target_params) / 30_000_000.0)
    block = int(max(int(model_cfg["block_size"]), min(8192, round_multiple(768.0 * (scale ** 0.24), 128, minimum=768))))
    recall_tokens = int(max(block * 4, round_multiple(block * (2.0 + scale ** 0.08), 512, minimum=block)))
    mem_slots = int(max(int(model_cfg["memory_slots"]), min(192, round_multiple(math.sqrt(float(block)) * 2.0, 8, minimum=24))))
    return dict(block_size=block, exact_recall_max_tokens=recall_tokens, memory_slots=mem_slots)


def shape_balance_score(cfg: Dict[str, Any], count: int, target_params: int, architecture: str) -> float:
    """Scale-neutral objective for choosing balanced shapes near a budget.

    The parameter error is still dominant, but a smooth depth prior prevents the
    search from landing on very deep/narrow or very shallow/wide shapes just
    because they hit the count exactly. No named model sizes are used.
    """
    rel = abs(float(count) - float(target_params)) / max(1.0, float(target_params))
    scale = max(0.25, float(target_params) / 10_000_000.0)
    target_depth = max(4.0, min(48.0, 4.0 + 6.0 * math.log10(scale)))
    L = max(1.0, float(cfg.get("n_layers", 1)))
    depth_penalty = 0.018 * (math.log(L / target_depth) ** 2)
    d = max(1.0, float(cfg.get("d_model", 1)))
    head_dim = d / max(1.0, float(cfg.get("n_heads", 1)))
    head_penalty = 0.003 * (math.log(max(1e-6, head_dim) / 64.0) ** 2)
    expert_penalty = 0.0
    if architecture == "dense":
        expert_penalty = 0.01 * max(0.0, float(cfg.get("n_experts", 1)) - 1.0)
    return float(rel + depth_penalty + head_penalty + expert_penalty)


def estimate_need_params(cfg: Dict[str, Any], vocab_size: int = 784) -> int:
    """Fast param estimate calibrated against NeedModel meta-device counts.

    NEED has substantial fixed side modules plus one recurrent/MoE block repeated
    n_layers times.  This estimate is used only to shortlist candidate shapes;
    final emitted configs use an exact meta-device count when PyTorch is present.
    """
    d = int(cfg["d_model"])
    L = int(cfg["n_layers"])
    n_experts = int(cfg.get("n_experts", 1))
    shared = bool(cfg.get("moe_use_shared_expert", n_experts > 1))
    ffn_paths = n_experts + (1 if shared else 0)
    side_coef = 91.5
    block_coef = 29.5 + 24.0 * max(0, ffn_paths - 1)
    return int((side_coef + block_coef * L) * d * d + (int(cfg.get("block_size", 1024)) + vocab_size) * d)


def estimate_actual_params(cfg: Dict[str, Any]) -> Optional[int]:
    """Return an exact NeedModel parameter count using meta tensors when possible."""
    try:
        import torch
        from need_core import NeedConfig, NeedModel

        valid = {k: v for k, v in cfg.items() if k in NeedConfig.__dataclass_fields__}
        valid.setdefault("vocab_size", 272 + int(valid.get("image_codebook_size", 512)))
        with torch.device("meta"):
            model = NeedModel(NeedConfig(**valid))
        return int(sum(p.numel() for p in model.parameters()))
    except Exception:
        return None


def model_config_for_params(
    target_params: int,
    architecture: str = "dense",
    *,
    block_size: int = 0,
    prefer_multiple: int = 64,
    exact_count: bool = False,
) -> Tuple[Dict[str, Any], int]:
    """Search NEED dimensions that land close to a target parameter count."""
    target_params = int(target_params)
    architecture = architecture.lower()
    if architecture not in {"dense", "moe"}:
        raise ValueError("architecture must be 'dense' or 'moe'")

    if block_size <= 0:
        block_size = derive_block_size(target_params)

    dims = list(range(max(128, int(prefer_multiple)), 2049, int(prefer_multiple)))
    candidates: List[Tuple[float, Dict[str, Any], int]] = []
    moe_choices = [1] if architecture == "dense" else [2, 4, 6, 8, 12, 16]

    for d in dims:
        if d < 128:
            continue
        heads = choose_heads(d)
        d_ff = round_multiple(4 * d, 64, minimum=4 * d)
        energy_rank = round_multiple(d / 4, 16, minimum=32)
        memory_slots = max(16, min(128, round_multiple(d / 12, 8, minimum=16)))
        pathway_slots = max(24, min(128, round_multiple(d / 12, 8, minimum=24)))
        memory_rank = max(64, min(256, round_multiple(d / 6, 16, minimum=64)))
        for L in range(4, 49):
            for n_experts in moe_choices:
                shared = architecture == "moe"
                cfg = dict(
                    d_model=d,
                    n_layers=L,
                    n_heads=heads,
                    d_ff=d_ff,
                    block_size=block_size,
                    n_experts=n_experts,
                    moe_top_k=min(2, n_experts),
                    moe_use_shared_expert=shared,
                    energy_rank=energy_rank,
                    memory_slots=memory_slots,
                    pathway_memory_slots=pathway_slots,
                    memory_rank=memory_rank,
                    memory_chunk_size=derive_memory_chunk_size(block_size),
                )
                est = estimate_need_params(cfg)
                score = shape_balance_score(cfg, est, target_params, architecture)
                candidates.append((score, cfg, est))

    candidates.sort(key=lambda x: x[0])
    best_cfg = candidates[0][1].copy()
    best_count = candidates[0][2]
    best_score = shape_balance_score(best_cfg, best_count, target_params, architecture)
    if exact_count:
        for _, cfg, est in candidates[:24]:
            actual = estimate_actual_params(cfg)
            count = int(actual or est)
            score = shape_balance_score(cfg, count, target_params, architecture)
            if score < best_score:
                best_cfg = cfg.copy()
                best_count = count
                best_score = score
    return best_cfg, int(best_count)


def hardware_plan(cfg: Dict[str, Any], total_tokens: int, gpu_mem_gb: float = 24.0, ram_gb: float = 0.0, vcpus: int = 0) -> Dict[str, Any]:
    block = int(cfg["block_size"])
    d = int(cfg["d_model"])
    L = int(cfg["n_layers"])
    # Conservative activation estimate for this NEED implementation. The train-time
    # auto-batch probe in train.py is authoritative when --auto_optimize is used.
    bytes_per_token_per_sample = max(1, int(2.0 * d * max(6, L) * 18))
    params = int(estimate_actual_params(cfg) or estimate_need_params(cfg))
    static_gb = params * 10.5 / (1024 ** 3)  # bf16 params+grads plus fp32-ish AdamW states, rough.
    usable_gb = max(1.0, float(gpu_mem_gb) * 0.88 - static_gb)
    est_micro = max(1, int((usable_gb * (1024 ** 3)) // max(1, bytes_per_token_per_sample * block)))
    est_micro = min(32, max(1, est_micro))
    target_effective_tokens = int(max(131_072, min(1_048_576, round_multiple(262_144 * math.sqrt(max(1.0, float(total_tokens)) / 1_000_000_000.0), 65_536, minimum=131_072))))
    grad_accum = max(1, math.ceil(target_effective_tokens / max(1, est_micro * block)))
    micro_steps = max(1, math.ceil(int(total_tokens) / max(1, est_micro * block)))
    opt_steps = max(1, math.ceil(micro_steps / grad_accum))
    workers = max(0, min(8, (int(vcpus) - 2) if int(vcpus) > 4 else max(0, int(vcpus) - 1))) if vcpus else -1
    prefetch = 4 if (ram_gb <= 0 or ram_gb >= 32) else 2
    return dict(
        batch_size=est_micro,
        grad_accum_steps=grad_accum,
        target_effective_batch_tokens=target_effective_tokens,
        max_steps=micro_steps,
        estimated_optimizer_steps=opt_steps,
        num_workers=workers,
        prefetch_factor=prefetch,
        amp="bf16",
        compile=True,
        compile_cudagraphs=True,
        compile_dynamic=False,
        prefetch_to_device=True,
        drop_last=True,
        minimal_aux_metrics=True,
        peak_tflops=209.0 if float(gpu_mem_gb) >= 32.0 else 0.0,
        auto_optimize=True,
        auto_batch=True,
    )


def build_config(
    params: Any = 30_000_000,
    total_tokens: Any = 10_000_000_000,
    modality: str = "text",
    gpu_mem_gb: float = 24.0,
    architecture: str = "dense",
    ram_gb: float = 0.0,
    vcpus: int = 0,
) -> Dict[str, Any]:
    target_params = parse_param_count(params)
    total_tokens_i = parse_token_count(total_tokens)
    if target_params <= 0:
        raise ValueError("params must be positive")
    if total_tokens_i <= 0:
        raise ValueError("tokens must be positive")
    model_cfg, estimated_params = model_config_for_params(target_params, architecture)
    if modality in ("image", "multimodal"):
        model_cfg.update(derive_image_shape(int(model_cfg["block_size"]), target_params))
    elif modality == "long_context":
        model_cfg.update(derive_long_context_shape(model_cfg, target_params))
    else:
        model_cfg.setdefault("image_grid", derive_image_shape(int(model_cfg["block_size"]), target_params)["image_grid"])
    train = hardware_plan(model_cfg, total_tokens_i, gpu_mem_gb=gpu_mem_gb, ram_gb=ram_gb, vcpus=vcpus)
    lr = 2.4e-4 * math.sqrt(300_000_000.0 / max(30_000_000.0, target_params))
    train.update(
        lr=max(6e-5, min(3e-4, lr)),
        weight_decay=0.10,
        warmup_steps=max(500, int(train["estimated_optimizer_steps"] * 0.03)),
        eval_interval=max(250, int(train["max_steps"] / 40)),
        log_interval=20,
        target_tokens=total_tokens_i,
        stream_data=True,
    )
    cfg = dict(
        model=model_cfg,
        training=train,
        curriculum=dict(
            phases=[
                {"name": "language_foundation", "token_fraction": 0.42, "mix": {"knowledge": 1.0}},
                {"name": "knowledge_reasoning", "token_fraction": 0.28, "mix": {"knowledge": 0.80, "math_science": 0.20}},
                {"name": "code_math", "token_fraction": 0.15, "mix": {"code": 0.55, "math_science": 0.45}},
                {"name": "instruction_alignment", "token_fraction": 0.15, "mix": {"sft": 0.55, "preference": 0.20, "rlvr": 0.20, "safety": 0.05}},
            ]
        ),
        target_params=target_params,
        estimated_params=estimated_params,
        architecture=architecture,
    )
    return cfg


def train_command(config: Dict[str, Any], data: str, out_dir: str, image_dir: str = "", recipe: str = "fast", packed_index: str = "") -> str:
    m = config["model"]
    t = config["training"]
    parts = [
        "python", "train.py", "--out_dir", out_dir,
        "--d_model", str(m["d_model"]), "--n_layers", str(m["n_layers"]), "--n_heads", str(m["n_heads"]),
        "--d_ff", str(m["d_ff"]), "--block_size", str(m["block_size"]),
        "--n_experts", str(m["n_experts"]), "--moe_top_k", str(m["moe_top_k"]),
        "--energy_rank", str(m["energy_rank"]), "--memory_slots", str(m["memory_slots"]),
        "--pathway_memory_slots", str(m["pathway_memory_slots"]),
        "--memory_rank", str(m.get("memory_rank", 64)), "--memory_chunk_size", str(m.get("memory_chunk_size", 32)),
        "--batch_size", str(0 if bool(t.get("auto_batch", False)) else t["batch_size"]), "--grad_accum_steps", str(0 if bool(t.get("auto_optimize", False)) else t["grad_accum_steps"]),
        "--target_tokens", format_scaled_number(int(t["target_tokens"])), "--target_effective_batch_tokens", format_scaled_number(int(t["target_effective_batch_tokens"])),
        "--max_steps", str(t["max_steps"]), "--lr", str(t["lr"]), "--weight_decay", str(t["weight_decay"]),
        "--eval_interval", str(t["eval_interval"]), "--log_interval", str(t["log_interval"]),
        "--amp", str(t.get("amp", "bf16")),
    ]
    if packed_index:
        parts.extend(["--packed_index", packed_index])
    else:
        parts.extend(["--data", data, "--stream_data"])
    if recipe:
        parts.extend(["--recipe", recipe])
    if not bool(m.get("moe_use_shared_expert", True)):
        parts.append("--disable_shared_expert")
    if bool(t.get("auto_optimize", False)):
        parts.append("--auto_optimize")
    if bool(t.get("auto_batch", False)):
        parts.append("--auto_batch")
    if bool(t.get("compile", False)):
        parts.append("--compile")
        parts.extend(["--compile_mode", "max-autotune"])
        if bool(t.get("compile_cudagraphs", False)):
            parts.append("--compile_cudagraphs")
        if not bool(t.get("compile_dynamic", True)):
            parts.append("--compile_static")
    if bool(t.get("prefetch_to_device", False)):
        parts.append("--prefetch_to_device")
    if bool(t.get("drop_last", False)):
        parts.append("--drop_last")
    if bool(t.get("minimal_aux_metrics", False)):
        parts.append("--minimal_aux_metrics")
    if float(t.get("peak_tflops", 0.0) or 0.0) > 0:
        parts.extend(["--peak_tflops", str(t.get("peak_tflops"))])
    if int(t.get("num_workers", 0)) >= 0:
        parts.extend(["--num_workers", str(t.get("num_workers", 0))])
    parts.extend(["--prefetch_factor", str(t.get("prefetch_factor", 4))])
    if image_dir:
        parts.extend(["--image_dir", image_dir])
    return " ".join(parts)


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--params", default="30M", help="Target parameter count, e.g. 300M, 1.2B.")
    p.add_argument("--params_m", type=float, default=None, help="Backward-compatible parameter count in millions.")
    p.add_argument("--architecture", choices=["dense", "moe"], default="dense")
    p.add_argument("--tokens", type=parse_token_count, default="10B", help="Target training tokens, e.g. 20B.")
    p.add_argument("--modality", choices=["text", "image", "multimodal", "long_context"], default="text")
    p.add_argument("--gpu_mem_gb", type=float, default=24.0)
    p.add_argument("--ram_gb", type=float, default=0.0)
    p.add_argument("--vcpus", type=int, default=0)
    p.add_argument("--data", default="data/corpuses/knowledge/train.jsonl")
    p.add_argument("--packed_index", default="", help="Prefer source-balanced packed_index.json in generated command.")
    p.add_argument("--recipe", default="fast", help="Training recipe for generated command: fast, quality, baseline, debug, or empty.")
    p.add_argument("--image_dir", default="")
    p.add_argument("--out_dir", default="need_out")
    p.add_argument("--write", default="")
    p.add_argument("--print_train_cmd", action="store_true")
    args = p.parse_args(argv)
    params = int(round(args.params_m * 1_000_000)) if args.params_m is not None else args.params
    cfg = build_config(params, args.tokens, args.modality, args.gpu_mem_gb, architecture=args.architecture, ram_gb=args.ram_gb, vcpus=args.vcpus)
    if args.write:
        Path(args.write).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(json.dumps(cfg, indent=2))
    if args.print_train_cmd:
        print("\n" + train_command(cfg, args.data, args.out_dir, args.image_dir, recipe=args.recipe, packed_index=args.packed_index))


if __name__ == "__main__":
    main()
