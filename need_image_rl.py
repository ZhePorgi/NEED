#!/usr/bin/env python3
"""NEED image preference replay and low-data image policy tuning.

The text replay path can do a conservative semi-finetune from a small set of
high-quality episodes.  This module gives the lower-level image RL path the same
shape: collect preference episodes, expand them conservatively, train a small
reward surface, or semi-finetune only image-facing NEED parameters with a DPO-like
pairwise loss. It does not generate images; it only trains from supplied image
preference data.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from need_core import (
    ByteTokenizer,
    NeedModel,
    Special,
    load_model,
    resolve_device,
    save_json,
    save_model,
)
from need_image import load_visual_tokenizer, pil_to_tensor

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    from safetensors.torch import load_file as safe_load_file, save_file as safe_save_file
except Exception:  # pragma: no cover
    safe_load_file = None
    safe_save_file = None

PreferenceRow = Dict[str, object]


def _safe_torch_load(path: Path, map_location: Union[str, torch.device] = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _clean_text(x: object, max_len: int = 12000) -> str:
    return str(x or "").replace("\r\n", "\n")[:max_len]


def _as_float(x: object, default: float = 1.0) -> float:
    try:
        return float(x)  # type: ignore[arg-type]
    except Exception:
        return default


def normalize_preference_row(row: PreferenceRow) -> PreferenceRow:
    """Normalize supported JSONL schemas to chosen/rejected image paths.

    Supported rows:
      {"chosen":"good.png", "rejected":"bad.png", "prompt":"optional"}
      {"image_a":"a.png", "image_b":"b.png", "winner":"a|b"}
    """
    if "chosen" in row and "rejected" in row:
        chosen, rejected = row["chosen"], row["rejected"]
    else:
        if "image_a" not in row or "image_b" not in row:
            raise ValueError("preference rows need chosen/rejected or image_a/image_b")
        a, b = row["image_a"], row["image_b"]
        winner = str(row.get("winner", "a")).lower()
        chosen, rejected = (a, b) if winner in ("a", "0", "image_a", "chosen_a") else (b, a)
    out = dict(row)
    out["chosen"] = str(chosen)
    out["rejected"] = str(rejected)
    out["prompt"] = _clean_text(row.get("prompt", row.get("caption", row.get("instruction", ""))), 6000)
    out["summary"] = _clean_text(row.get("summary", row.get("rationale", "")), 4000)
    out["score"] = max(0.0, min(1.0, _as_float(row.get("score", row.get("preference_score", 1.0)), 1.0)))
    out.setdefault("augmentation_id", 0)
    return out


def read_preference_rows(path: Union[str, Path]) -> List[PreferenceRow]:
    rows: List[PreferenceRow] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(normalize_preference_row(obj))
    if not rows:
        raise ValueError(f"No preference rows found in {path}")
    return rows


def augment_preference_row(row: PreferenceRow, factor: int = 3) -> List[PreferenceRow]:
    """Conservative preference replay expansion.

    This mirrors the text replay augmenter: do not invent new labels or alter the
    chosen/rejected relationship.  The extra rows vary prompt wrappers and carry
    augmentation ids so optional paired pixel transforms can be deterministic.
    """
    row = normalize_preference_row(row)
    prompt = _clean_text(row.get("prompt"), 6000)
    summary = _clean_text(row.get("summary"), 3000)
    wrappers = [prompt]
    if prompt:
        wrappers.extend([
            "Prefer the image that best satisfies this request.\n\nRequest:\n" + prompt,
            prompt + "\n\nPreference requirement: keep the visual result faithful, clear, and aligned.",
        ])
    else:
        wrappers.extend([
            "Prefer the higher-quality and more faithful image.",
            "Prefer the image with fewer artifacts and better visual alignment.",
        ])
    if summary:
        wrappers.append("Prior preference summary:\n" + summary + "\n\nRequest:\n" + prompt)
    out: List[PreferenceRow] = []
    for i, wrapped in enumerate(wrappers[: max(1, int(factor))]):
        nr = dict(row)
        nr["prompt"] = wrapped
        nr["augmentation_id"] = i
        out.append(nr)
    return out


def expand_preference_rows(rows: List[PreferenceRow], augment_factor: int = 1) -> List[PreferenceRow]:
    if augment_factor <= 1:
        return [normalize_preference_row(r) for r in rows]
    expanded: List[PreferenceRow] = []
    for row in rows:
        expanded.extend(augment_preference_row(row, augment_factor))
    return expanded


def _resolve_image(root: Union[str, Path], value: object) -> Path:
    p = Path(str(value))
    return p if p.is_absolute() else Path(root) / p


def _paired_image_transform(img: "Image.Image", augmentation_id: int) -> "Image.Image":
    """Optional deterministic paired transform for low-data image preferences."""
    if augmentation_id % 2 == 1:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if augmentation_id % 4 in (2, 3):
        w, h = img.size
        margin_w = max(0, int(w * 0.04))
        margin_h = max(0, int(h * 0.04))
        if margin_w > 0 and margin_h > 0:
            img = img.crop((margin_w, margin_h, w - margin_w, h - margin_h))
    return img


class ImagePreferenceDataset(Dataset):
    """Pairwise image preference dataset.

    JSONL rows may be {"chosen":"a.png", "rejected":"b.png"} or
    {"image_a":..., "image_b":..., "winner":"a|b"}.  Optional prompt text is
    preserved for policy tuning but ignored by reward-head training.
    """
    def __init__(
        self,
        rows_or_jsonl: Union[str, Path, List[PreferenceRow]],
        root: str = "",
        size: int = 256,
        augment_factor: int = 1,
        paired_transforms: bool = False,
    ):
        if Image is None:
            raise RuntimeError("Pillow is required")
        if isinstance(rows_or_jsonl, (str, Path)):
            base_rows = read_preference_rows(rows_or_jsonl)
            default_root = Path(rows_or_jsonl).parent
        else:
            base_rows = [normalize_preference_row(r) for r in rows_or_jsonl]
            default_root = Path(".")
        self.root = Path(root) if root else default_root
        self.size = int(size)
        self.rows = expand_preference_rows(base_rows, augment_factor)
        self.paired_transforms = bool(paired_transforms)
        if not self.rows:
            raise ValueError("No preference rows found")

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, path: Path, augmentation_id: int) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        if self.paired_transforms:
            img = _paired_image_transform(img, augmentation_id)
        return pil_to_tensor(img, self.size)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.rows[idx % len(self.rows)]
        aug_id = int(row.get("augmentation_id", 0))
        chosen = self._load(_resolve_image(self.root, row["chosen"]), aug_id)
        rejected = self._load(_resolve_image(self.root, row["rejected"]), aug_id)
        return {"chosen": chosen, "rejected": rejected, "row": row}


class FeaturePreferenceDataset(Dataset):
    def __init__(self, replay_dir: Union[str, Path], augment_factor: int = 1, feature_jitter: float = 0.0, seed: int = 123):
        pack = _safe_torch_load(Path(replay_dir) / "image_replay.pt", map_location="cpu")
        episodes = list(pack.get("episodes", []))
        if not episodes:
            raise ValueError(f"No image replay episodes found in {replay_dir}")
        self.rows = expand_preference_rows(episodes, augment_factor)
        self.feature_jitter = float(feature_jitter)
        self.seed = int(seed)
        self.embed_dim = int(pack.get("embed_dim", episodes[0]["chosen_feature"].numel()))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.rows[idx % len(self.rows)]
        fc = row["chosen_feature"].float().clone()
        fb = row["rejected_feature"].float().clone()
        if self.feature_jitter > 0:
            g = torch.Generator().manual_seed(self.seed + idx)
            fc = fc + torch.randn(fc.shape, generator=g) * self.feature_jitter
            fb = fb + torch.randn(fb.shape, generator=g) * self.feature_jitter
        return {"chosen_feature": fc, "rejected_feature": fb, "row": row}


class ImageRewardHead(nn.Module):
    def __init__(self, embed_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).squeeze(-1)


def encode_features(vt, x: torch.Tensor) -> torch.Tensor:
    z_e = vt.encoder(x)
    return z_e.mean(dim=(2, 3))


def save_reward(head: ImageRewardHead, out_dir: Union[str, Path], embed_dim: int, hidden: int = 256, metrics: Optional[Dict[str, float]] = None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json({"format": "need_image_reward", "embed_dim": embed_dim, "hidden": hidden, "metrics": metrics or {}}, out / "image_reward_config.json")
    state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    if safe_save_file is not None:
        safe_save_file(state, str(out / "image_reward.safetensors"))
    else:
        torch.save(state, out / "image_reward.pt")


def load_reward(path: Union[str, Path], device: Union[str, torch.device] = "cpu") -> ImageRewardHead:
    root = Path(path)
    raw = json.loads((root / "image_reward_config.json").read_text(encoding="utf-8"))
    head = ImageRewardHead(int(raw["embed_dim"]), int(raw.get("hidden", 256)))
    if safe_load_file is not None and (root / "image_reward.safetensors").exists():
        state = safe_load_file(str(root / "image_reward.safetensors"), device=str(device))
    else:
        state = _safe_torch_load(root / "image_reward.pt", map_location=device)
    head.load_state_dict(state, strict=True)
    head.to(device).eval()
    return head


def _pairwise_reward_loss(rc: torch.Tensor, rb: torch.Tensor, margin_reg: float) -> torch.Tensor:
    return F.softplus(-(rc - rb)).mean() + float(margin_reg) * (rc.pow(2).mean() + rb.pow(2).mean())


def collect(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    vt = load_visual_tokenizer(args.visual_tokenizer, device=device).eval()
    for p in vt.parameters():
        p.requires_grad_(False)
    rows = read_preference_rows(args.preferences)
    rows = expand_preference_rows(rows, args.augment_factor if args.augment_preferences else 1)
    ds = ImagePreferenceDataset(rows, root=args.image_root or str(Path(args.preferences).parent), size=args.image_size, paired_transforms=args.paired_transforms)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    episodes: List[PreferenceRow] = []
    for batch in dl:
        c = batch["chosen"].to(device, non_blocking=True)
        b = batch["rejected"].to(device, non_blocking=True)
        with torch.no_grad():
            fc = encode_features(vt, c).detach().cpu().to(torch.float16)
            fb = encode_features(vt, b).detach().cpu().to(torch.float16)
        rows_batch = batch["row"]
        # Default DataLoader collates dict values into lists/tensors. Reconstruct per row.
        if isinstance(rows_batch, dict):
            n = fc.size(0)
            unpacked: List[PreferenceRow] = []
            for i in range(n):
                rr: PreferenceRow = {}
                for k, v in rows_batch.items():
                    if torch.is_tensor(v):
                        rr[k] = v[i].item() if v.ndim > 0 else v.item()
                    elif isinstance(v, list):
                        rr[k] = v[i]
                    else:
                        rr[k] = v
                unpacked.append(rr)
        else:
            unpacked = list(rows_batch)
        for i, row in enumerate(unpacked):
            ep = normalize_preference_row(row)
            ep["chosen_feature"] = fc[i]
            ep["rejected_feature"] = fb[i]
            ep["feature_delta"] = (fc[i].float() - fb[i].float()).to(torch.float16)
            episodes.append(ep)
        if args.max_items > 0 and len(episodes) >= args.max_items:
            episodes = episodes[: args.max_items]
            break
        if len(episodes) % max(1, args.log_interval) == 0:
            print(json.dumps({"collected": len(episodes)}), flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"episodes": episodes, "embed_dim": int(vt.cfg.embed_dim), "visual_tokenizer": str(args.visual_tokenizer)}, out / "image_replay.pt")
    save_json({"episodes": len(episodes), "embed_dim": int(vt.cfg.embed_dim), "visual_tokenizer": str(args.visual_tokenizer)}, out / "manifest.json")
    print(json.dumps({"done": True, "episodes": len(episodes), "out_dir": str(out)}, indent=2), flush=True)


def train_reward(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    raw_count = 0
    low_data_auto = False
    vt = None
    if args.replay_dataset:
        pack = _safe_torch_load(Path(args.replay_dataset) / "image_replay.pt", map_location="cpu")
        raw_count = len(pack.get("episodes", []))
        embed_dim = int(pack.get("embed_dim", args.embed_dim or 0))
        if embed_dim <= 0:
            raise ValueError("Could not infer embed_dim from replay dataset; pass --embed_dim")
    else:
        if not args.preferences:
            raise ValueError("Provide --preferences or --replay_dataset")
        if not args.visual_tokenizer:
            raise ValueError("--visual_tokenizer is required when training from raw image preferences")
        vt = load_visual_tokenizer(args.visual_tokenizer, device=device).eval()
        for p in vt.parameters():
            p.requires_grad_(False)
        embed_dim = int(vt.cfg.embed_dim)
        raw_count = len(read_preference_rows(args.preferences))

    low_data_auto = bool(args.auto_few_shot and raw_count <= args.few_shot_auto_threshold)
    if low_data_auto:
        if not args.augment_preferences:
            args.augment_preferences = True
        args.lr = min(args.lr, args.few_shot_max_lr)
        args.weight_decay = max(args.weight_decay, args.few_shot_min_weight_decay)
        args.feature_jitter = max(args.feature_jitter, args.low_data_feature_jitter)
        args.consistency_weight = max(args.consistency_weight, args.low_data_consistency_weight)

    augment_factor = args.augment_factor if args.augment_preferences else 1
    if args.replay_dataset:
        ds: Dataset = FeaturePreferenceDataset(args.replay_dataset, augment_factor=augment_factor, feature_jitter=args.feature_jitter, seed=args.seed)
    else:
        ds = ImagePreferenceDataset(
            args.preferences,
            root=args.image_root,
            size=args.image_size,
            augment_factor=augment_factor,
            paired_transforms=args.paired_transforms,
        )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    head = ImageRewardHead(embed_dim, args.hidden).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(json.dumps({
        "reward_low_data_mode": low_data_auto,
        "raw_preferences": raw_count,
        "expanded_preferences": len(ds),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "feature_jitter": args.feature_jitter,
        "consistency_weight": args.consistency_weight,
    }), flush=True)
    step = 0
    last_metrics: Dict[str, float] = {}
    while step < args.steps:
        for batch in dl:
            if step >= args.steps:
                break
            if args.replay_dataset:
                fc = batch["chosen_feature"].to(device, non_blocking=True).float()
                fb = batch["rejected_feature"].to(device, non_blocking=True).float()
            else:
                c = batch["chosen"].to(device, non_blocking=True)
                b = batch["rejected"].to(device, non_blocking=True)
                assert vt is not None
                with torch.no_grad():
                    fc = encode_features(vt, c)
                    fb = encode_features(vt, b)
            rc = head(fc)
            rb = head(fb)
            loss = _pairwise_reward_loss(rc, rb, args.margin_reg)
            if args.feature_jitter > 0 and args.consistency_weight > 0:
                fc_j = fc + torch.randn_like(fc) * args.feature_jitter
                fb_j = fb + torch.randn_like(fb) * args.feature_jitter
                rc_j = head(fc_j)
                rb_j = head(fb_j)
                jitter_loss = _pairwise_reward_loss(rc_j, rb_j, args.margin_reg)
                stable = F.mse_loss(rc_j, rc.detach()) + F.mse_loss(rb_j, rb.detach())
                loss = loss + args.consistency_weight * (jitter_loss + 0.25 * stable)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            opt.step()
            if step % args.log_interval == 0:
                acc = float((rc > rb).float().mean().detach().cpu())
                last_metrics = {"step": float(step), "loss": float(loss.detach().cpu()), "pref_acc": acc}
                print(json.dumps(last_metrics), flush=True)
            step += 1
    save_reward(head, args.out_dir, embed_dim, hidden=args.hidden, metrics=last_metrics)
    print(json.dumps({"done": True, "out_dir": str(args.out_dir)}, indent=2), flush=True)


def freeze_image_policy_for_fewshot(model: NeedModel, train_embeddings: bool = False) -> int:
    """Freeze most NEED weights and keep image-facing adaptation surfaces trainable."""
    keep_terms = (
        "image_quality",
        "image_proj",
        "text_proj",
        "object_program",
        "lm_head",
        "norm",
        "ln",
        "modality_emb",
        "bias",
    )
    if train_embeddings:
        keep_terms = keep_terms + ("token_emb", "pos_emb")
    trainable = 0
    for name, p in model.named_parameters():
        keep = any(term in name.lower() for term in keep_terms)
        p.requires_grad_(keep)
        if keep:
            trainable += p.numel()
    return trainable


def _encode_policy_sequence(tok: ByteTokenizer, model: NeedModel, vt, prompt: str, image_path: Path, root: Path, grid: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_ids = tok.encode(prompt or "Image preference task", add_bos=True)
    ids, _ = vt.encode_image(image_path, add_special=True, grid=(grid or None), device=device)
    max_len = int(model.cfg.block_size + 1)
    if len(ids) >= max_len:
        seq = ids[:max_len]
    else:
        keep_prompt = max(1, max_len - len(ids))
        seq = prompt_ids[-keep_prompt:] + ids
    seq = seq[-max_len:]
    if len(seq) < max_len:
        seq = seq + [model.cfg.pad_id] * (max_len - len(seq))
    arr = torch.tensor(seq, dtype=torch.long)
    x = arr[:-1]
    y = arr[1:]
    mask = (y >= model.cfg.image_token_offset) & (y < model.cfg.image_token_offset + model.cfg.image_codebook_size)
    return x, y, mask


class ImagePolicyPreferenceDataset(Dataset):
    def __init__(
        self,
        preferences: Union[str, Path],
        model: NeedModel,
        visual_tokenizer,
        root: str = "",
        augment_factor: int = 1,
        grid: int = 0,
        tokenizer_device: Union[str, torch.device] = "cpu",
    ):
        self.rows = expand_preference_rows(read_preference_rows(preferences), augment_factor)
        self.root = Path(root) if root else Path(preferences).parent
        self.model = model
        self.vt = visual_tokenizer
        self.tok = ByteTokenizer()
        self.grid = int(grid)
        self.tokenizer_device = resolve_device(str(tokenizer_device)) if isinstance(tokenizer_device, str) else tokenizer_device

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx % len(self.rows)]
        prompt = _clean_text(row.get("prompt"), 6000)
        chosen_path = _resolve_image(self.root, row["chosen"])
        rejected_path = _resolve_image(self.root, row["rejected"])
        cx, cy, cm = _encode_policy_sequence(self.tok, self.model, self.vt, prompt, chosen_path, self.root, self.grid, self.tokenizer_device)
        bx, by, bm = _encode_policy_sequence(self.tok, self.model, self.vt, prompt, rejected_path, self.root, self.grid, self.tokenizer_device)
        return {"chosen_x": cx, "chosen_y": cy, "chosen_mask": cm, "rejected_x": bx, "rejected_y": by, "rejected_mask": bm}


def image_logprob(model: NeedModel, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits, _, _ = model(x, None)
    logp = F.log_softmax(logits.float(), dim=-1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
    mask_f = mask.float() * (y != model.cfg.pad_id).float()
    return (logp * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)


def freeze_object_program_for_fewshot(model: NeedModel) -> int:
    keep_terms = (
        "object_program",
        "image_quality",
        "image_proj",
        "text_proj",
        "modality_emb",
        "norm",
        "ln",
        "bias",
    )
    trainable = 0
    for name, p in model.named_parameters():
        keep = any(term in name.lower() for term in keep_terms)
        p.requires_grad_(keep)
        if keep:
            trainable += p.numel()
    return trainable


def _image_quality_score(model: NeedModel, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    _, _, aux = model(x, None, return_hidden=True)
    h = aux["_hidden"]
    mask_f = mask.float().unsqueeze(-1)
    pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
    score = model.image_quality(pooled).squeeze(-1)
    text_mask = (x >= Special.byte_start) & (x < model.cfg.image_token_offset)
    _, obj_aux = model.object_program(h, text_mask=text_mask)
    return score, obj_aux


def tune_object_program(args: argparse.Namespace) -> None:
    """Low-data tune the image object/layout adapter without generating images.

    This trains only image-facing surfaces from existing chosen/rejected image
    preference pairs.  It is meant for object-program calibration, not base image
    generation pretraining.
    """
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device=device, prefer_best=args.prefer_best, kernel_backend=args.kernel_backend)
    vt_device = resolve_device(args.visual_tokenizer_device)
    vt = load_visual_tokenizer(args.visual_tokenizer, device=vt_device).eval()
    for p in vt.parameters():
        p.requires_grad_(False)
    raw_count = len(read_preference_rows(args.preferences))
    low_data_auto = bool(args.auto_few_shot and raw_count <= args.few_shot_auto_threshold)
    if low_data_auto:
        args.few_shot_mode = True
        if not args.augment_preferences:
            args.augment_preferences = True
        args.lr = min(args.lr, args.few_shot_max_lr)
        args.weight_decay = max(args.weight_decay, args.few_shot_min_weight_decay)
    ds = ImagePolicyPreferenceDataset(
        args.preferences,
        model,
        vt,
        root=args.image_root,
        augment_factor=args.augment_factor if args.augment_preferences else 1,
        grid=args.force_image_grid,
        tokenizer_device=vt_device,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    model.train()
    trainable = freeze_object_program_for_fewshot(model) if args.few_shot_mode else sum(p.numel() for p in model.parameters() if p.requires_grad)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable object-program parameters")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    print(json.dumps({
        "object_program_low_data_mode": low_data_auto,
        "few_shot_mode": bool(args.few_shot_mode),
        "raw_preferences": raw_count,
        "expanded_preferences": len(ds),
        "trainable_parameters": int(trainable),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }), flush=True)
    step = 0
    last_metrics: Dict[str, float] = {}
    while step < args.steps:
        for batch in dl:
            if step >= args.steps:
                break
            cx = batch["chosen_x"].to(device, non_blocking=True)
            cm = batch["chosen_mask"].to(device, non_blocking=True)
            bx = batch["rejected_x"].to(device, non_blocking=True)
            bm = batch["rejected_mask"].to(device, non_blocking=True)
            sc, ac = _image_quality_score(model, cx, cm)
            sb, ab = _image_quality_score(model, bx, bm)
            margin = sc - sb
            pref_loss = F.softplus(-float(args.beta) * margin).mean()
            coverage = (ac.get("object_coverage", torch.tensor(0.0, device=device)) + ab.get("object_coverage", torch.tensor(0.0, device=device))) * 0.5
            presence = ac.get("object_presence", torch.tensor(0.0, device=device))
            layout_area = ac.get("object_layout_area", torch.tensor(0.0, device=device))
            object_loss = (1.0 - coverage.float()).clamp_min(0.0) + F.mse_loss(presence.float(), torch.full_like(presence.float(), float(args.target_presence)))
            area_loss = F.relu(layout_area.float() - float(args.max_layout_area))
            reg = (sc.pow(2).mean() + sb.pow(2).mean()) * float(args.score_reg)
            loss = pref_loss + float(args.object_weight) * object_loss + float(args.layout_weight) * area_loss + reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            if step % args.log_interval == 0:
                last_metrics = {
                    "step": float(step),
                    "loss": float(loss.detach().cpu()),
                    "pref_acc": float((margin > 0).float().mean().detach().cpu()),
                    "margin": float(margin.mean().detach().cpu()),
                    "object_coverage": float(coverage.detach().cpu()),
                    "object_presence": float(presence.detach().cpu()),
                }
                print(json.dumps(last_metrics), flush=True)
            step += 1
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_model(model, out, {"object_program_low_data_steps": float(args.steps), **last_metrics}, name="model")
    save_json({
        "format": "need_low_data_object_program_adapter",
        "raw_preferences": raw_count,
        "expanded_preferences": len(ds),
        "few_shot_mode": bool(args.few_shot_mode),
        "metrics": last_metrics,
    }, out / "object_program_adapter_config.json")
    print(json.dumps({"done": True, "out_dir": str(out)}, indent=2), flush=True)


def tune_policy(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device=device, prefer_best=args.prefer_best, kernel_backend=args.kernel_backend)
    vt_device = resolve_device(args.visual_tokenizer_device)
    vt = load_visual_tokenizer(args.visual_tokenizer, device=vt_device).eval()
    for p in vt.parameters():
        p.requires_grad_(False)
    raw_count = len(read_preference_rows(args.preferences))
    low_data_auto = bool(args.auto_few_shot and raw_count <= args.few_shot_auto_threshold)
    if low_data_auto:
        args.few_shot_mode = True
        if not args.augment_preferences:
            args.augment_preferences = True
        args.lr = min(args.lr, args.few_shot_max_lr)
        args.weight_decay = max(args.weight_decay, args.few_shot_min_weight_decay)
        args.chosen_nll_weight = min(args.chosen_nll_weight, args.few_shot_max_nll_weight)
    augment_factor = args.augment_factor if args.augment_preferences else 1
    ds = ImagePolicyPreferenceDataset(
        args.preferences,
        model,
        vt,
        root=args.image_root,
        augment_factor=augment_factor,
        grid=args.force_image_grid,
        tokenizer_device=vt_device,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.few_shot_mode:
        trainable = freeze_image_policy_for_fewshot(model, train_embeddings=args.policy_train_embeddings)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable policy parameters; check few-shot freeze settings")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    ref_model = None
    if args.reference_checkpoint:
        ref_model = load_model(args.reference_checkpoint, device=device, prefer_best=args.reference_prefer_best, kernel_backend=args.kernel_backend)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
    print(json.dumps({
        "policy_low_data_mode": low_data_auto,
        "few_shot_mode": bool(args.few_shot_mode),
        "raw_preferences": raw_count,
        "expanded_preferences": len(ds),
        "trainable_parameters": int(trainable),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "beta": args.beta,
    }), flush=True)
    step = 0
    last_metrics: Dict[str, float] = {}
    while step < args.steps:
        for batch in dl:
            if step >= args.steps:
                break
            cx = batch["chosen_x"].to(device, non_blocking=True)
            cy = batch["chosen_y"].to(device, non_blocking=True)
            cm = batch["chosen_mask"].to(device, non_blocking=True)
            bx = batch["rejected_x"].to(device, non_blocking=True)
            by = batch["rejected_y"].to(device, non_blocking=True)
            bm = batch["rejected_mask"].to(device, non_blocking=True)
            lp_c = image_logprob(model, cx, cy, cm)
            lp_b = image_logprob(model, bx, by, bm)
            margin = lp_c - lp_b
            if ref_model is not None:
                with torch.no_grad():
                    ref_margin = image_logprob(ref_model, cx, cy, cm) - image_logprob(ref_model, bx, by, bm)
                margin = margin - ref_margin
            pref_loss = F.softplus(-float(args.beta) * margin).mean()
            chosen_nll = -lp_c.mean()
            loss = pref_loss + float(args.chosen_nll_weight) * chosen_nll
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            if step % args.log_interval == 0:
                acc = float((margin > 0).float().mean().detach().cpu())
                last_metrics = {
                    "step": float(step),
                    "loss": float(loss.detach().cpu()),
                    "pref_loss": float(pref_loss.detach().cpu()),
                    "chosen_nll": float(chosen_nll.detach().cpu()),
                    "pref_acc": acc,
                    "margin": float(margin.mean().detach().cpu()),
                }
                print(json.dumps(last_metrics), flush=True)
            step += 1
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_model(model, out, {"image_policy_replay_steps": float(args.steps), **last_metrics}, name="model")
    print(json.dumps({"done": True, "out_dir": str(out)}, indent=2), flush=True)


def _add_reward_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--visual_tokenizer", default="", help="Directory containing visual_tokenizer_config.json and weights")
    p.add_argument("--preferences", default="", help="JSONL preference pairs")
    p.add_argument("--replay_dataset", default="", help="Directory from `collect` containing image_replay.pt")
    p.add_argument("--image_root", default="")
    p.add_argument("--out_dir", default="need_image_reward")
    p.add_argument("--device", default="auto")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--embed_dim", type=int, default=0)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--margin_reg", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--augment_preferences", action="store_true", help="apply conservative low-data preference replay expansion")
    p.add_argument("--augment_factor", type=int, default=3)
    p.add_argument("--paired_transforms", action="store_true", help="apply the same deterministic flip/crop to both images in an augmented pair")
    p.add_argument("--feature_jitter", type=float, default=0.0)
    p.add_argument("--consistency_weight", type=float, default=0.0)
    p.add_argument("--auto_few_shot", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--few_shot_auto_threshold", type=int, default=300)
    p.add_argument("--few_shot_max_lr", type=float, default=1e-4)
    p.add_argument("--few_shot_min_weight_decay", type=float, default=2e-3)
    p.add_argument("--low_data_feature_jitter", type=float, default=0.015)
    p.add_argument("--low_data_consistency_weight", type=float, default=0.10)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NEED image preference replay, reward training, and low-data policy tuning")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="collect image preferences into replay features")
    c.add_argument("--visual_tokenizer", required=True)
    c.add_argument("--preferences", required=True)
    c.add_argument("--image_root", default="")
    c.add_argument("--out_dir", required=True)
    c.add_argument("--device", default="auto")
    c.add_argument("--image_size", type=int, default=256)
    c.add_argument("--batch_size", type=int, default=8)
    c.add_argument("--num_workers", type=int, default=0)
    c.add_argument("--max_items", type=int, default=0)
    c.add_argument("--log_interval", type=int, default=50)
    c.add_argument("--augment_preferences", action="store_true")
    c.add_argument("--augment_factor", type=int, default=3)
    c.add_argument("--paired_transforms", action="store_true")

    tr = sub.add_parser("train_reward", aliases=["reward", "train"], help="train the image reward head")
    _add_reward_args(tr)

    tp = sub.add_parser("tune_policy", aliases=["policy"], help="semi-finetune image-facing NEED policy surfaces from preferences")
    tp.add_argument("--checkpoint", required=True)
    tp.add_argument("--prefer_best", action="store_true")
    tp.add_argument("--reference_checkpoint", default="", help="optional reference model for true DPO-style margins")
    tp.add_argument("--reference_prefer_best", action="store_true")
    tp.add_argument("--visual_tokenizer", required=True)
    tp.add_argument("--visual_tokenizer_device", default="cpu")
    tp.add_argument("--preferences", required=True)
    tp.add_argument("--image_root", default="")
    tp.add_argument("--out_dir", required=True)
    tp.add_argument("--device", default="auto")
    tp.add_argument("--kernel_backend", default="auto")
    tp.add_argument("--batch_size", type=int, default=2)
    tp.add_argument("--steps", type=int, default=1000)
    tp.add_argument("--lr", type=float, default=5e-5)
    tp.add_argument("--weight_decay", type=float, default=0.01)
    tp.add_argument("--beta", type=float, default=0.10)
    tp.add_argument("--chosen_nll_weight", type=float, default=0.05)
    tp.add_argument("--grad_clip", type=float, default=1.0)
    tp.add_argument("--seed", type=int, default=123)
    tp.add_argument("--log_interval", type=int, default=20)
    tp.add_argument("--force_image_grid", type=int, default=0)
    tp.add_argument("--few_shot_mode", action="store_true")
    tp.add_argument("--auto_few_shot", action=argparse.BooleanOptionalAction, default=True)
    tp.add_argument("--few_shot_auto_threshold", type=int, default=300)
    tp.add_argument("--few_shot_max_lr", type=float, default=2e-5)
    tp.add_argument("--few_shot_min_weight_decay", type=float, default=0.02)
    tp.add_argument("--few_shot_max_nll_weight", type=float, default=0.03)
    tp.add_argument("--policy_train_embeddings", action="store_true")
    tp.add_argument("--augment_preferences", action="store_true")
    tp.add_argument("--augment_factor", type=int, default=3)

    op = sub.add_parser("tune_object_program", aliases=["object_program"], help="low-data tune object/layout image adapter surfaces from preferences")
    op.add_argument("--checkpoint", required=True)
    op.add_argument("--prefer_best", action="store_true")
    op.add_argument("--visual_tokenizer", required=True)
    op.add_argument("--visual_tokenizer_device", default="cpu")
    op.add_argument("--preferences", required=True)
    op.add_argument("--image_root", default="")
    op.add_argument("--out_dir", required=True)
    op.add_argument("--device", default="auto")
    op.add_argument("--kernel_backend", default="auto")
    op.add_argument("--batch_size", type=int, default=2)
    op.add_argument("--steps", type=int, default=400)
    op.add_argument("--lr", type=float, default=4e-5)
    op.add_argument("--weight_decay", type=float, default=0.015)
    op.add_argument("--beta", type=float, default=0.10)
    op.add_argument("--object_weight", type=float, default=0.10)
    op.add_argument("--layout_weight", type=float, default=0.03)
    op.add_argument("--score_reg", type=float, default=1e-4)
    op.add_argument("--target_presence", type=float, default=0.65)
    op.add_argument("--max_layout_area", type=float, default=0.55)
    op.add_argument("--grad_clip", type=float, default=1.0)
    op.add_argument("--seed", type=int, default=123)
    op.add_argument("--log_interval", type=int, default=20)
    op.add_argument("--force_image_grid", type=int, default=0)
    op.add_argument("--few_shot_mode", action="store_true")
    op.add_argument("--auto_few_shot", action=argparse.BooleanOptionalAction, default=True)
    op.add_argument("--few_shot_auto_threshold", type=int, default=300)
    op.add_argument("--few_shot_max_lr", type=float, default=2e-5)
    op.add_argument("--few_shot_min_weight_decay", type=float, default=0.02)
    op.add_argument("--augment_preferences", action="store_true")
    op.add_argument("--augment_factor", type=int, default=3)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    # Backward compatibility with the old script: `python need_image_rl.py --visual_tokenizer ...`
    # still trains the reward head.
    if raw and raw[0].startswith("--") and raw[0] not in ("-h", "--help"):
        raw = ["train_reward"] + raw
    args = build_arg_parser().parse_args(raw)
    if args.cmd == "collect":
        collect(args)
    elif args.cmd in ("train_reward", "reward", "train"):
        train_reward(args)
    elif args.cmd in ("tune_policy", "policy"):
        tune_policy(args)
    elif args.cmd in ("tune_object_program", "object_program"):
        tune_object_program(args)
    else:  # pragma: no cover
        raise ValueError(f"unknown command {args.cmd}")


if __name__ == "__main__":
    main()
