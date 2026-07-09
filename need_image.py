#!/usr/bin/env python3
"""NEED learned visual tokenizer and image-quality tools.

This module upgrades the NEED image path from a deterministic color codebook to a
trainable convolutional VQ tokenizer.  NEED still models image generation as
masked diffusion over discrete image tokens; this tokenizer makes those tokens
semantic and reconstructable.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from need_core import Special, resolve_device, save_json, load_json

try:
    from PIL import Image, ImageOps
except Exception:  # pragma: no cover
    Image = None
    ImageOps = None

try:
    from safetensors.torch import load_file as safe_load_file, save_file as safe_save_file
except Exception:  # pragma: no cover
    safe_load_file = None
    safe_save_file = None


def _group_count(channels: int, preferred: int = 32) -> int:
    """Choose a GroupNorm group count that always divides channels."""
    ch = int(max(1, channels))
    pref = int(max(1, min(preferred, ch)))
    for g in range(pref, 0, -1):
        if ch % g == 0:
            return g
    return 1


def _resample_lanczos():
    return getattr(getattr(Image, "Resampling", Image), "LANCZOS")

@dataclass
class VisualTokenizerConfig:
    codebook_size: int = 512
    embed_dim: int = 128
    hidden_dim: int = 128
    num_res_blocks: int = 2
    downsample: int = 16
    min_grid: int = 8
    max_grid: int = 32
    default_grid: int = 16
    max_image_tokens: int = 1024
    dynamic_grid: bool = True
    token_offset: int = Special.text_vocab
    commitment_cost: float = 0.25
    codebook_cost: float = 1.0
    recon_l1_weight: float = 1.0
    recon_l2_weight: float = 0.25
    edge_weight: float = 0.08
    perceptual_weight: float = 0.10
    gan_weight: float = 0.05
    ema_decay: float = 0.0  # reserved; straight-through VQ is default

    def validate(self) -> None:
        self.downsample = int(self.downsample)
        self.codebook_size = int(self.codebook_size)
        self.embed_dim = int(self.embed_dim)
        self.hidden_dim = int(self.hidden_dim)
        self.num_res_blocks = int(max(0, self.num_res_blocks))
        self.min_grid = int(max(1, self.min_grid))
        self.max_grid = int(max(self.min_grid, self.max_grid))
        self.default_grid = int(max(self.min_grid, min(self.default_grid, self.max_grid)))
        self.max_image_tokens = int(max(1, self.max_image_tokens))
        self.commitment_cost = float(max(0.0, self.commitment_cost))
        self.codebook_cost = float(max(0.0, self.codebook_cost))
        self.recon_l1_weight = float(max(0.0, self.recon_l1_weight))
        self.recon_l2_weight = float(max(0.0, self.recon_l2_weight))
        self.edge_weight = float(max(0.0, self.edge_weight))
        self.perceptual_weight = float(max(0.0, self.perceptual_weight))
        self.gan_weight = float(max(0.0, self.gan_weight))
        if self.downsample <= 0 or self.downsample & (self.downsample - 1) != 0:
            raise ValueError("downsample must be a positive power of two")
        if self.codebook_size <= 1:
            raise ValueError("codebook_size must be > 1")
        if self.embed_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("embed_dim and hidden_dim must be positive")


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(_group_count(ch, min(32, max(1, ch // 4))), ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(_group_count(ch, min(32, max(1, ch // 4))), ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
        )
        self.scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.net(x)


class VectorQuantizer(nn.Module):
    def __init__(self, n_embed: int, embed_dim: int, commitment_cost: float = 0.25, codebook_cost: float = 1.0):
        super().__init__()
        self.n_embed = int(n_embed)
        self.embed_dim = int(embed_dim)
        self.commitment_cost = float(commitment_cost)
        self.codebook_cost = float(codebook_cost)
        self.embedding = nn.Embedding(self.n_embed, self.embed_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / self.n_embed, 1.0 / self.n_embed)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # z_e: [B,D,H,W]
        b, d, h, w = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, d).float()
        emb = self.embedding.weight.float()
        dist = flat.pow(2).sum(dim=1, keepdim=True) - 2 * flat @ emb.t() + emb.pow(2).sum(dim=1).view(1, -1)
        indices = torch.argmin(dist, dim=1)
        z_q = self.embedding(indices).view(b, h, w, d).permute(0, 3, 1, 2).contiguous()
        commit = F.mse_loss(z_e.float(), z_q.detach().float())
        codebook = F.mse_loss(z_q.float(), z_e.detach().float())
        loss = self.commitment_cost * commit + self.codebook_cost * codebook
        z_st = z_e + (z_q - z_e).detach()
        encodings = F.one_hot(indices, self.n_embed).float()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())
        return z_st, indices.view(b, h, w), loss, perplexity

    def indices_to_latents(self, indices: torch.Tensor) -> torch.Tensor:
        # indices: [B,H,W]
        z = self.embedding(indices.long())
        return z.permute(0, 3, 1, 2).contiguous()


class VisualTokenizerVQVAE(nn.Module):
    def __init__(self, cfg: VisualTokenizerConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        h = cfg.hidden_dim
        layers: List[nn.Module] = [nn.Conv2d(3, h, 3, padding=1), nn.SiLU()]
        downs = int(math.log2(cfg.downsample))
        ch = h
        for _ in range(downs):
            layers += [nn.Conv2d(ch, ch, 4, stride=2, padding=1), ResBlock(ch)]
        for _ in range(cfg.num_res_blocks):
            layers.append(ResBlock(ch))
        layers.append(nn.Conv2d(ch, cfg.embed_dim, 1))
        self.encoder = nn.Sequential(*layers)
        self.quantizer = VectorQuantizer(cfg.codebook_size, cfg.embed_dim, cfg.commitment_cost, cfg.codebook_cost)
        dec: List[nn.Module] = [nn.Conv2d(cfg.embed_dim, h, 3, padding=1)]
        for _ in range(cfg.num_res_blocks):
            dec.append(ResBlock(h))
        for _ in range(downs):
            dec += [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(h, h, 3, padding=1), ResBlock(h)]
        dec += [nn.GroupNorm(_group_count(h, min(32, max(1, h // 4))), h), nn.SiLU(), nn.Conv2d(h, 3, 3, padding=1)]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.encoder(x)
        return self.quantizer(z_e)

    def decode_latents(self, z_q: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decoder(z_q))

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return self.decode_latents(self.quantizer.indices_to_latents(indices))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_q, indices, vq_loss, perplexity = self.encode(x)
        recon = self.decode_latents(z_q)
        return {"recon": recon, "indices": indices, "vq_loss": vq_loss, "perplexity": perplexity}

    @torch.no_grad()
    def choose_grid_from_pil(self, image: "Image.Image") -> int:
        cfg = self.cfg
        if not cfg.dynamic_grid:
            return int(cfg.default_grid)
        small = image.convert("RGB").resize((64, 64))
        arr = np.asarray(small, dtype=np.float32) / 255.0
        gx = np.abs(arr[:, 1:] - arr[:, :-1]).mean()
        gy = np.abs(arr[1:, :] - arr[:-1, :]).mean()
        # Use both edge density and color variance so detailed or high-frequency images get more tokens.
        complexity = float(0.65 * (gx + gy) + 0.20 * arr.var())
        if complexity < 0.040:
            grid = max(cfg.min_grid, cfg.default_grid // 2)
        elif complexity > 0.110:
            grid = min(cfg.max_grid, cfg.default_grid * 2)
        else:
            grid = cfg.default_grid
        while grid * grid > cfg.max_image_tokens and grid > cfg.min_grid:
            grid //= 2
        return int(grid)

    @torch.no_grad()
    def encode_image(self, image: Union[str, Path, "Image.Image"], add_special: bool = True, grid: Optional[int] = None, device: Optional[torch.device] = None) -> Tuple[List[int], Dict[str, int]]:
        if Image is None:
            raise RuntimeError("Pillow is required for image tokenization")
        was_training = self.training
        self.eval()
        try:
            dev = device or next(self.parameters()).device
            if not hasattr(image, "convert"):
                with Image.open(image) as im:  # type: ignore[arg-type]
                    image = im.convert("RGB")
            else:
                image = image.convert("RGB")  # type: ignore[union-attr]
            grid = int(grid or self.choose_grid_from_pil(image))
            grid = max(self.cfg.min_grid, min(self.cfg.max_grid, grid))
            size = grid * self.cfg.downsample
            x = pil_to_tensor(image, size=size).unsqueeze(0).to(dev)
            _, idx, _, _ = self.encode(x)
            ids = (idx.reshape(-1).detach().cpu().long().numpy() + int(self.cfg.token_offset)).tolist()
            if add_special:
                ids = [Special.img_bos] + ids + [Special.img_eos]
            return ids, {"grid": grid, "height": grid, "width": grid, "tokenizer": "learned_vq"}
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def decode_tokens(self, ids: Sequence[int], grid: Optional[int] = None, size: int = 256, device: Optional[torch.device] = None) -> "Image.Image":
        if Image is None:
            raise RuntimeError("Pillow is required for image decoding")
        raw = [int(i) - int(self.cfg.token_offset) for i in ids if self.cfg.token_offset <= int(i) < self.cfg.token_offset + self.cfg.codebook_size]
        if not raw:
            raw = [0]
        if grid is None:
            grid = max(1, int(round(math.sqrt(len(raw)))))
        grid = int(max(1, min(int(grid), int(self.cfg.max_grid))))
        needed = grid * grid
        if len(raw) < needed:
            raw = raw + [raw[-1]] * (needed - len(raw))
        raw = raw[:needed]
        idx = torch.tensor(raw, dtype=torch.long, device=device or next(self.parameters()).device).view(1, grid, grid)
        out = self.decode_indices(idx).clamp(0, 1)[0].detach().cpu()
        return tensor_to_pil(out, size=size)


def pil_to_tensor(img: "Image.Image", size: int) -> torch.Tensor:
    if ImageOps is not None:
        img = ImageOps.exif_transpose(img)
    img = img.convert("RGB").resize((size, size), resample=_resample_lanczos())
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def tensor_to_pil(x: torch.Tensor, size: Optional[int] = None) -> "Image.Image":
    x = x.detach().float().clamp(0, 1)
    arr = (x.permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    if size is not None and img.size != (size, size):
        img = img.resize((size, size), resample=_resample_lanczos())
    return img


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3) / 4.0
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3) / 4.0
    return torch.cat([F.conv2d(gray, kx, padding=1), F.conv2d(gray, ky, padding=1)], dim=1)


class TinyPerceptualNet(nn.Module):
    """Frozen random multi-scale features for a cheap perceptual stabilizer.

    It is not LPIPS, but it improves the objective over pure pixel loss without
    adding an external dependency.  The weights are fixed and deterministic.
    """
    def __init__(self):
        super().__init__()
        # Initialize deterministic frozen features without clobbering the caller's
        # global RNG stream for the VQ model, discriminator, or dataloader workers.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(314159)
            self.layers = nn.ModuleList([
                nn.Conv2d(3, 16, 3, padding=1, bias=False),
                nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
                nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            ])
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        for layer in self.layers:
            x = F.silu(layer(x))
            feats.append(x)
        return feats


class PatchDiscriminator(nn.Module):
    def __init__(self, base: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, base, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1), nn.GroupNorm(_group_count(base * 2, 8), base * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1), nn.GroupNorm(_group_count(base * 4, 16), base * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImageFolderVQDataset(Dataset):
    def __init__(self, image_dir: Union[str, Path], size: int, samples: int = 10000, seed: int = 123):
        if Image is None:
            raise RuntimeError("Pillow is required for image datasets")
        self.paths: List[Path] = []
        root = Path(image_dir)
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
            self.paths.extend(sorted(root.rglob(ext)))
        if not self.paths:
            raise ValueError(f"No images found in {root}")
        self.size = int(size)
        self.samples = int(samples)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx % len(self.paths)]
        rng = random.Random(self.seed + idx)
        with Image.open(path) as im:
            img = im.convert("RGB")
        if rng.random() < 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return pil_to_tensor(img, self.size)


def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor, percept: TinyPerceptualNet, cfg: VisualTokenizerConfig) -> Dict[str, torch.Tensor]:
    l1 = F.l1_loss(recon, target)
    l2 = F.mse_loss(recon, target)
    edge = F.l1_loss(sobel_edges(recon), sobel_edges(target))
    pf = recon.new_tensor(0.0)
    if cfg.perceptual_weight > 0:
        fr = percept(recon)
        ft = percept(target)
        pf = torch.stack([F.l1_loss(a, b) for a, b in zip(fr, ft)]).mean()
    total = cfg.recon_l1_weight * l1 + cfg.recon_l2_weight * l2 + cfg.edge_weight * edge + cfg.perceptual_weight * pf
    return {"recon_total": total, "l1": l1.detach(), "l2": l2.detach(), "edge": edge.detach(), "perceptual": pf.detach()}



def visual_curriculum_weights(step: int, total_steps: int, args: argparse.Namespace) -> Dict[str, float]:
    """Progressive visual-tokenizer curriculum.

    Early training prioritizes stable pixel/codebook reconstruction.  Mid training
    ramps edges/perceptual/code-usage.  Late training enables adversarial pressure.
    This gives NEED image tokens that are both reconstructable and easier to model.
    """
    if not getattr(args, "curriculum", False):
        return {
            "edge": float(args.edge_weight),
            "perceptual": float(args.perceptual_weight),
            "gan": float(args.gan_weight),
            "code_usage": float(getattr(args, "code_usage_weight", 0.0)),
        }
    progress = min(1.0, max(0.0, step / max(1, total_steps)))
    edge = float(args.edge_weight) * min(1.0, progress / 0.25)
    perceptual = float(args.perceptual_weight) * min(1.0, max(0.0, (progress - 0.15) / 0.35))
    gan = float(args.gan_weight) * min(1.0, max(0.0, (progress - 0.55) / 0.35))
    code_usage = float(getattr(args, "code_usage_weight", 0.0)) * min(1.0, progress / 0.45)
    return {"edge": edge, "perceptual": perceptual, "gan": gan, "code_usage": code_usage}


def codebook_usage_loss(perplexity: torch.Tensor, codebook_size: int) -> torch.Tensor:
    target = math.log(max(2, int(codebook_size)))
    used = torch.log(perplexity.clamp_min(1.0))
    return ((target - used).clamp_min(0.0) / target).pow(2)

def save_visual_tokenizer(model: VisualTokenizerVQVAE, out_dir: Union[str, Path], name: str = "visual_tokenizer") -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    save_json({"format": "need_visual_tokenizer", "config": asdict(model.cfg)}, out / "visual_tokenizer_config.json")
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if safe_save_file is not None:
        safe_save_file(state, str(out / f"{name}.safetensors"))
    else:
        torch.save(state, out / f"{name}.pt")


def load_visual_tokenizer(path: Union[str, Path], device: Union[str, torch.device] = "cpu") -> VisualTokenizerVQVAE:
    root = Path(path)
    explicit_weight: Optional[Path] = root if root.is_file() else None
    if explicit_weight is not None:
        root = explicit_weight.parent
    cfg_path = root / "visual_tokenizer_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"visual_tokenizer_config.json not found in {root}")
    raw = load_json(cfg_path)
    cfg_data = raw.get("config", raw)
    cfg = VisualTokenizerConfig(**{k: v for k, v in cfg_data.items() if k in VisualTokenizerConfig.__dataclass_fields__})
    model = VisualTokenizerVQVAE(cfg)

    weight_path: Optional[Path] = explicit_weight
    if weight_path is None:
        default_safe = root / "visual_tokenizer.safetensors"
        default_pt = root / "visual_tokenizer.pt"
        if default_safe.exists():
            weight_path = default_safe
        elif default_pt.exists():
            weight_path = default_pt
        else:
            candidates = sorted(root.glob("*.safetensors")) + sorted(root.glob("*.pt"))
            if len(candidates) == 1:
                weight_path = candidates[0]
    if weight_path is None or not weight_path.exists():
        raise FileNotFoundError(f"No visual_tokenizer weights found in {root}")

    if weight_path.suffix == ".safetensors":
        if safe_load_file is None:
            raise RuntimeError("safetensors is required to load .safetensors visual tokenizer weights")
        state = safe_load_file(str(weight_path), device=str(device))
    else:
        try:
            state = torch.load(weight_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def train_visual_tokenizer(args: argparse.Namespace) -> Dict[str, float]:
    device = resolve_device(args.device)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = VisualTokenizerConfig(
        codebook_size=args.codebook_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_res_blocks=args.num_res_blocks,
        downsample=args.downsample,
        default_grid=args.grid,
        min_grid=args.min_grid,
        max_grid=args.max_grid,
        max_image_tokens=args.max_image_tokens,
        token_offset=Special.text_vocab,
        gan_weight=args.gan_weight,
        perceptual_weight=args.perceptual_weight,
        edge_weight=args.edge_weight,
    )
    size = int(args.grid * args.downsample)
    ds = ImageFolderVQDataset(args.image_dir, size=size, samples=args.samples, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = VisualTokenizerVQVAE(cfg).to(device)
    disc = PatchDiscriminator(max(16, args.hidden_dim // 2)).to(device) if args.gan_weight > 0 else None
    percept = TinyPerceptualNet().to(device).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    dopt = torch.optim.AdamW(disc.parameters(), lr=args.lr * 0.5, betas=(0.5, 0.9)) if disc is not None else None
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp == "fp16"))
    step = 0
    metrics: Dict[str, float] = {}
    while step < args.steps:
        for batch in dl:
            if step >= args.steps:
                break
            x = batch.to(device, non_blocking=True)
            autocast_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
            weights = visual_curriculum_weights(step, args.steps, args)
            # Use a temporary cfg view so reconstruction_loss can reuse the same implementation
            # while the curriculum changes the effective loss weights over time.
            cfg_step = VisualTokenizerConfig(**asdict(cfg))
            cfg_step.edge_weight = weights["edge"]
            cfg_step.perceptual_weight = weights["perceptual"]
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=(device.type == "cuda" and args.amp != "off")):
                outd = model(x)
                recon = outd["recon"]
                losses = reconstruction_loss(recon, x, percept, cfg_step)
                usage = codebook_usage_loss(outd["perplexity"], cfg.codebook_size)
                g_loss = losses["recon_total"] + outd["vq_loss"] + weights["code_usage"] * usage
                gan_g = x.new_tensor(0.0)
                if disc is not None and step >= args.gan_start and weights["gan"] > 0:
                    gan_g = -disc(recon).mean()
                    g_loss = g_loss + weights["gan"] * gan_g
            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(g_loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
            else:
                g_loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            d_loss = x.new_tensor(0.0)
            if disc is not None and dopt is not None and step >= args.gan_start:
                with torch.no_grad():
                    fake = model(x)["recon"].detach()
                real_logits = disc(x)
                fake_logits = disc(fake)
                d_loss = F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()
                dopt.zero_grad(set_to_none=True); d_loss.backward(); dopt.step()
            if step % args.log_interval == 0:
                metrics = {
                    "step": step,
                    "loss": float(g_loss.detach().cpu()),
                    "recon": float(losses["recon_total"].detach().cpu()),
                    "vq": float(outd["vq_loss"].detach().cpu()),
                    "perplexity": float(outd["perplexity"].detach().cpu()),
                    "gan_g": float(gan_g.detach().cpu()),
                    "gan_d": float(d_loss.detach().cpu()),
                    "code_usage": float(usage.detach().cpu()),
                    "curr_edge": float(weights["edge"]),
                    "curr_perceptual": float(weights["perceptual"]),
                    "curr_gan": float(weights["gan"]),
                }
                print(json.dumps(metrics), flush=True)
            if step > 0 and step % args.save_interval == 0:
                save_visual_tokenizer(model, out)
            step += 1
    save_visual_tokenizer(model, out)
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train or use the NEED learned VQ image tokenizer")
    sub = p.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--image_dir", required=True)
    tr.add_argument("--out_dir", default="need_visual_tokenizer")
    tr.add_argument("--device", default="auto")
    tr.add_argument("--seed", type=int, default=123)
    tr.add_argument("--codebook_size", type=int, default=512)
    tr.add_argument("--embed_dim", type=int, default=128)
    tr.add_argument("--hidden_dim", type=int, default=128)
    tr.add_argument("--num_res_blocks", type=int, default=2)
    tr.add_argument("--downsample", type=int, default=16)
    tr.add_argument("--grid", type=int, default=16)
    tr.add_argument("--min_grid", type=int, default=8)
    tr.add_argument("--max_grid", type=int, default=32)
    tr.add_argument("--max_image_tokens", type=int, default=1024)
    tr.add_argument("--batch_size", type=int, default=8)
    tr.add_argument("--samples", type=int, default=10000)
    tr.add_argument("--steps", type=int, default=5000)
    tr.add_argument("--num_workers", type=int, default=0)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--weight_decay", type=float, default=1e-4)
    tr.add_argument("--gan_weight", type=float, default=0.05)
    tr.add_argument("--gan_start", type=int, default=1000)
    tr.add_argument("--perceptual_weight", type=float, default=0.10)
    tr.add_argument("--edge_weight", type=float, default=0.08)
    tr.add_argument("--amp", choices=["off", "bf16", "fp16"], default="bf16")
    tr.add_argument("--log_interval", type=int, default=50)
    tr.add_argument("--save_interval", type=int, default=1000)
    tr.add_argument("--curriculum", action="store_true", help="Progressively ramp reconstruction, perceptual, adversarial and code-usage losses")
    tr.add_argument("--code_usage_weight", type=float, default=0.02)
    enc = sub.add_parser("encode")
    enc.add_argument("--tokenizer", required=True)
    enc.add_argument("--image", required=True)
    enc.add_argument("--out_json", required=True)
    enc.add_argument("--device", default="auto")
    dec = sub.add_parser("decode")
    dec.add_argument("--tokenizer", required=True)
    dec.add_argument("--tokens_json", required=True)
    dec.add_argument("--out", required=True)
    dec.add_argument("--grid", type=int, default=0)
    dec.add_argument("--size", type=int, default=256)
    dec.add_argument("--device", default="auto")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.cmd == "train":
        train_visual_tokenizer(args)
    elif args.cmd == "encode":
        dev = resolve_device(args.device)
        tok = load_visual_tokenizer(args.tokenizer, dev)
        ids, meta = tok.encode_image(args.image, add_special=True, device=dev)
        Path(args.out_json).write_text(json.dumps({"tokens": ids, "meta": meta}, indent=2), encoding="utf-8")
    elif args.cmd == "decode":
        dev = resolve_device(args.device)
        tok = load_visual_tokenizer(args.tokenizer, dev)
        data = json.loads(Path(args.tokens_json).read_text(encoding="utf-8"))
        ids = data["tokens"] if isinstance(data, dict) else data
        grid = args.grid or (data.get("meta", {}).get("grid") if isinstance(data, dict) else 0) or None
        img = tok.decode_tokens(ids, grid=grid, size=args.size, device=dev)
        img.save(args.out)


if __name__ == "__main__":
    main()
