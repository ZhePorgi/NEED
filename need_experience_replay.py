#!/usr/bin/env python3
"""Experience replay for NEED latent reasoning trajectories.

This script turns prior interactions into replayable latent episodes.  It is not
plain text RAG: it stores prompt/answer text, outcome scores, public summaries,
and compressed latent pathway vectors for opt-in behavioral guidance and
fine-tuning on response patterns that worked. Runtime generation does not read
these episodes unless explicitly enabled.

JSONL input format for `collect`:
{"prompt": "...", "answer": "...", "score": 0.0-1.0, "summary": "optional", "correction": "optional better answer"}
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from need_core import ByteTokenizer, NeedModel, load_model, save_model, resolve_device, LatentMemoryStore


def _safe_torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and obj.get("prompt"):
                rows.append(obj)
    return rows




# ----------------------------- low-data replay helpers -----------------------------

def _clean_text(x: object, max_len: int = 12000) -> str:
    return str(x or "").replace("\r\n", "\n")[:max_len]


def augment_episode_row(row: Dict[str, object], factor: int = 3) -> List[Dict[str, object]]:
    """Create conservative text variants from one replay episode.

    This is intentionally not a paraphraser.  It preserves the original answer
    semantics while varying the instruction wrapper so 50-100 episodes can
    produce enough supervised contexts to steer format/style without inventing
    new facts.
    """
    prompt = _clean_text(row.get("prompt"))
    answer = _clean_text(row.get("correction") or row.get("answer"))
    summary = _clean_text(row.get("summary"), 4000)
    score = float(row.get("score", 0.5))
    templates = [
        (prompt, answer),
        ("Answer in the preferred style.\n\nTask:\n" + prompt, answer),
        ("Use the stored successful reasoning pattern for this task.\n\n" + prompt, answer),
        (prompt + "\n\nPreferred response requirements: be direct, faithful, and preserve the requested format.", answer),
    ]
    if summary:
        templates.append(("Prior successful summary:\n" + summary + "\n\nNew task:\n" + prompt, answer))
    out = []
    for i, (pr, an) in enumerate(templates[: max(1, factor)]):
        nr = dict(row)
        nr["prompt"] = pr
        nr["answer"] = an
        nr["score"] = score
        nr["augmentation_id"] = i
        out.append(nr)
    return out


def expand_rows(rows: List[Dict[str, object]], augment_factor: int = 1) -> List[Dict[str, object]]:
    if augment_factor <= 1:
        return rows
    expanded: List[Dict[str, object]] = []
    for row in rows:
        expanded.extend(augment_episode_row(row, augment_factor))
    return expanded


def freeze_for_fewshot(model: NeedModel, train_embeddings: bool = False) -> int:
    """Freeze most weights and leave small adaptation surfaces trainable.

    This is a cheap PEFT-like mode without adding new checkpoint format.  It is
    meant for 20-300 high-quality episodes where full finetuning would overfit.
    """
    keep_terms = (
        "lm_head", "norm", "ln", "aux_score", "controller", "output_mode",
        "pathway", "latent_slot", "energy_router", "recall", "adapter", "bias",
    )
    if train_embeddings:
        keep_terms = keep_terms + ("token_emb", "embed", "embedding")
    trainable = 0
    total = 0
    for name, p in model.named_parameters():
        total += p.numel()
        keep = any(term in name.lower() for term in keep_terms)
        p.requires_grad_(keep)
        if keep:
            trainable += p.numel()
    return trainable


def build_fewshot_context(episodes: List[Dict[str, object]], current: Dict[str, object], k: int, max_chars: int) -> str:
    if k <= 0:
        return ""
    # Small, deterministic behavioral proxy: prefer high-score episodes with different prompt text.
    pool = [e for e in episodes if e is not current]
    pool.sort(key=lambda e: float(e.get("score", 0.5)), reverse=True)
    chunks = []
    for e in pool[:k]:
        p = _clean_text(e.get("prompt"), max_chars // max(1, k))
        a = _clean_text(e.get("answer"), max_chars // max(1, k))
        chunks.append(f"Example task:\n{p}\nPreferred answer:\n{a}")
    if not chunks:
        return ""
    return "Behavioral examples to imitate:\n" + "\n\n---\n\n".join(chunks) + "\n\nCurrent task:\n"


def _negative_text_from_episode(ep: Dict[str, object]) -> str:
    for key in ("rejected", "bad_answer", "negative", "wrong_answer"):
        if ep.get(key):
            return str(ep.get(key))
    if float(ep.get("score", 0.5)) < 0.25 and ep.get("answer") and ep.get("correction"):
        return str(ep.get("answer"))
    return ""


def collect(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device=device, prefer_best=args.prefer_best, kernel_backend=args.kernel_backend)
    tok = ByteTokenizer()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(Path(args.interactions))[: args.max_items if args.max_items > 0 else None]
    rows = expand_rows(rows, args.augment_factor if getattr(args, "augment_replay", False) else 1)
    episodes = []
    mem = LatentMemoryStore(str(out / "latent_memory"), dim=model.cfg.d_model, max_items=max(args.max_items, 1) if args.max_items > 0 else 10000)
    for i, row in enumerate(rows):
        prompt = str(row.get("prompt", ""))
        answer = str(row.get("correction") or row.get("answer") or "")
        summary = str(row.get("summary") or "")
        score = float(row.get("score", 0.5))
        text_for_path = (prompt + "\n" + answer)[: args.max_chars]
        ids = torch.tensor([tok.encode(text_for_path, add_bos=True)[-model.cfg.block_size:]], dtype=torch.long, device=device)
        with torch.no_grad():
            pathway = model.latent_pathway(ids, stride=args.vector_stride, max_vectors=args.max_vectors)
        vectors = pathway["pathway_vectors"].detach().cpu().to(torch.float16)
        episode = {
            "prompt": prompt,
            "answer": answer,
            "summary": summary,
            "score": max(0.0, min(1.0, score)),
            "vectors": vectors,
            "quality": float(pathway.get("quality_mean", torch.tensor(0.0)).detach().cpu()),
            "risk": float(pathway.get("risk_mean", torch.tensor(0.0)).detach().cpu()),
        }
        episodes.append(episode)
        mem.add(prompt, summary or answer[:1000], vectors.float())
        if (i + 1) % max(1, args.log_interval) == 0:
            print(json.dumps({"collected": i + 1}), flush=True)
    torch.save({"episodes": episodes, "d_model": model.cfg.d_model}, out / "episodes.pt")
    with (out / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump({"episodes": len(episodes), "checkpoint": str(args.checkpoint)}, f, indent=2)
    print(json.dumps({"done": True, "episodes": len(episodes), "out_dir": str(out)}, indent=2))


def _batch_from_episode(tok: ByteTokenizer, model: NeedModel, ep: Dict[str, object], device: torch.device, context_prefix: str = ""):
    text = context_prefix + str(ep.get("prompt", "")) + "\n\nAnswer:\n" + str(ep.get("answer", ""))
    ids = tok.encode(text, add_bos=True, add_eos=True)
    ids = ids[-(model.cfg.block_size + 1):]
    if len(ids) < model.cfg.block_size + 1:
        ids = ids + [model.cfg.pad_id] * (model.cfg.block_size + 1 - len(ids))
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    vec = ep.get("vectors")
    vectors = vec.to(device=device, dtype=next(model.parameters()).dtype).float() if torch.is_tensor(vec) else None
    score = float(ep.get("score", 0.5))
    return x, y, vectors, score


def train(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device=device, prefer_best=args.prefer_best, kernel_backend=args.kernel_backend)
    tok = ByteTokenizer()
    pack = _safe_torch_load(Path(args.dataset) / "episodes.pt", map_location="cpu")
    episodes: List[Dict[str, object]] = list(pack.get("episodes", []))
    raw_episode_count = len(episodes)
    # Low-data mode is deliberately conservative.  With very small replay sets,
    # full-model finetuning often memorizes formats or damages broad behavior.
    # Auto mode therefore freezes most base weights, expands only conservative
    # wrapper variants, and uses retrieval-style in-batch examples.
    low_data_auto = bool(args.auto_few_shot and raw_episode_count <= args.few_shot_auto_threshold)
    if low_data_auto:
        args.few_shot_mode = True
        if not args.train_augment:
            args.train_augment = True
        if args.few_shot_context_k <= 0:
            args.few_shot_context_k = min(4, max(1, raw_episode_count - 1))
        args.lr = min(args.lr, args.few_shot_max_lr)
        args.weight_decay = max(args.weight_decay, args.few_shot_min_weight_decay)
    episodes = expand_rows(episodes, args.train_augment_factor if args.train_augment else 1)
    if not episodes:
        raise ValueError("No episodes found in dataset")
    random.seed(args.seed)
    model.train()
    if args.few_shot_mode:
        trainable = freeze_for_fewshot(model, train_embeddings=args.few_shot_train_embeddings)
        print(json.dumps({
            "few_shot_mode": True,
            "auto_few_shot": low_data_auto,
            "raw_episodes": raw_episode_count,
            "expanded_episodes": len(episodes),
            "few_shot_context_k": args.few_shot_context_k,
            "lr": args.lr,
            "trainable_parameters": trainable,
        }), flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    for step in range(args.steps):
        ep = random.choice(episodes)
        ctx = build_fewshot_context(episodes, ep, args.few_shot_context_k, args.few_shot_context_chars)
        x, y, vectors, score = _batch_from_episode(tok, model, ep, device, context_prefix=ctx)
        cond_scale = args.conditioning_scale if vectors is not None else 0.0
        _, loss, aux = model(x, y, conditioning_vectors=vectors, conditioning_scale=cond_scale)
        assert loss is not None
        # Outcome-conditioned replay: reward high-score trajectories, but still learn
        # lightly from low-score corrections if the `answer` field is a correction.
        weight = args.min_weight + (args.max_weight - args.min_weight) * max(0.0, min(1.0, score))
        loss = loss * weight
        # Latent conditioning consistency: the same episode should remain stable
        # under a lightly noised version of its latent pathway.  This multiplies
        # few examples without inventing new labels.
        if args.latent_jitter > 0 and vectors is not None and args.consistency_weight > 0:
            noisy_vectors = vectors + torch.randn_like(vectors) * args.latent_jitter
            _, loss_jitter, _ = model(x, y, conditioning_vectors=noisy_vectors, conditioning_scale=cond_scale)
            if loss_jitter is not None:
                loss = loss + args.consistency_weight * loss_jitter
        # Optional unlikelihood pressure against known rejected answers.
        neg = _negative_text_from_episode(ep)
        if args.negative_weight > 0 and neg:
            neg_ep = dict(ep); neg_ep["answer"] = neg
            nx, ny, nv, _ = _batch_from_episode(tok, model, neg_ep, device, context_prefix=ctx)
            nlogits, _, _ = model(nx, None, conditioning_vectors=nv if vectors is not None else None, conditioning_scale=cond_scale)
            logp = F.log_softmax(nlogits, dim=-1)
            nll_tok = -logp.gather(-1, ny.unsqueeze(-1)).squeeze(-1)
            mask = (ny != model.cfg.pad_id).float()
            # maximize NLL of rejected answer gently, but clamp to avoid blowups
            neg_loss = -torch.clamp((nll_tok * mask).sum() / mask.sum().clamp_min(1.0), max=args.negative_nll_cap)
            loss = loss + args.negative_weight * neg_loss
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step(); opt.zero_grad(set_to_none=True)
        if step % max(1, args.log_interval) == 0:
            print(json.dumps({"step": step, "loss": float(loss.detach().cpu()), "score": score, "weight": weight, "ce": float(aux.get("ce", torch.tensor(float('nan'))).detach().cpu())}), flush=True)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    save_model(model, out, {"experience_replay_steps": float(args.steps)}, name="model")
    print(json.dumps({"done": True, "out_dir": str(out)}, indent=2))


def retrieve(args: argparse.Namespace) -> None:
    pack = _safe_torch_load(Path(args.dataset) / "episodes.pt", map_location="cpu")
    episodes: List[Dict[str, object]] = list(pack.get("episodes", []))
    episodes = expand_rows(episodes, args.train_augment_factor if args.train_augment else 1)
    if not episodes:
        raise ValueError("No episodes found")
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device=device, prefer_best=args.prefer_best, kernel_backend=args.kernel_backend)
    tok = ByteTokenizer()
    ids = torch.tensor([tok.encode(args.prompt, add_bos=True)[-model.cfg.block_size:]], dtype=torch.long, device=device)
    with torch.no_grad():
        q = model.latent_pathway(ids, stride=args.vector_stride, max_vectors=args.max_vectors)["pathway_vectors"].float().mean(dim=1)
    scored = []
    for ep in episodes:
        v = ep.get("vectors")
        if not torch.is_tensor(v):
            continue
        key = v.float().mean(dim=1)
        sim = F.cosine_similarity(q.cpu(), key, dim=-1).mean().item()
        scored.append((sim, ep))
    scored.sort(key=lambda x: x[0], reverse=True)
    for sim, ep in scored[: args.k]:
        print(json.dumps({"similarity": sim, "score": ep.get("score", 0.0), "prompt_shape": str(ep.get("prompt", ""))[:500], "behavior_summary": str(ep.get("summary", ""))[:500]}, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NEED latent experience replay for opt-in behavioral guidance")
    sub = p.add_subparsers(dest="cmd", required=True)
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--checkpoint", required=True)
    base.add_argument("--prefer_best", action="store_true")
    base.add_argument("--device", default="auto")
    base.add_argument("--kernel_backend", default="auto")
    c = sub.add_parser("collect", parents=[base])
    c.add_argument("--interactions", required=True)
    c.add_argument("--out_dir", required=True)
    c.add_argument("--vector_stride", type=int, default=2)
    c.add_argument("--max_vectors", type=int, default=512)
    c.add_argument("--max_items", type=int, default=0)
    c.add_argument("--max_chars", type=int, default=12000)
    c.add_argument("--log_interval", type=int, default=50)
    c.add_argument("--augment_replay", action="store_true", help="store conservative replay augmentations so fewer episodes go further")
    c.add_argument("--augment_factor", type=int, default=3)
    t = sub.add_parser("train", parents=[base])
    t.add_argument("--dataset", required=True)
    t.add_argument("--out_dir", required=True)
    t.add_argument("--steps", type=int, default=1000)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--weight_decay", type=float, default=0.01)
    t.add_argument("--grad_clip", type=float, default=1.0)
    t.add_argument("--conditioning_scale", type=float, default=0.18)
    t.add_argument("--min_weight", type=float, default=0.25)
    t.add_argument("--max_weight", type=float, default=1.50)
    t.add_argument("--seed", type=int, default=123)
    t.add_argument("--log_interval", type=int, default=20)
    t.add_argument("--few_shot_mode", action="store_true", help="freeze most base weights and train small adaptation surfaces")
    t.add_argument("--auto_few_shot", action=argparse.BooleanOptionalAction, default=True, help="automatically use safer low-data adaptation when replay set is small")
    t.add_argument("--few_shot_auto_threshold", type=int, default=300, help="episode count at or below which auto few-shot mode activates")
    t.add_argument("--few_shot_max_lr", type=float, default=2e-5, help="auto few-shot clips LR to this value")
    t.add_argument("--few_shot_min_weight_decay", type=float, default=0.02, help="auto few-shot raises weight decay to at least this value")
    t.add_argument("--few_shot_train_embeddings", action="store_true", help="also train embeddings in few-shot mode")
    t.add_argument("--few_shot_context_k", type=int, default=2, help="prepend high-score replay examples to each update")
    t.add_argument("--few_shot_context_chars", type=int, default=2400)
    t.add_argument("--train_augment", action="store_true", help="apply conservative augmentation at train time")
    t.add_argument("--train_augment_factor", type=int, default=3)
    t.add_argument("--latent_jitter", type=float, default=0.015, help="noise latent vectors for consistency regularization")
    t.add_argument("--consistency_weight", type=float, default=0.08)
    t.add_argument("--negative_weight", type=float, default=0.0, help="gentle unlikelihood for known rejected/bad answers")
    t.add_argument("--negative_nll_cap", type=float, default=8.0)
    r = sub.add_parser("retrieve", parents=[base], help="inspect nearest behavioral episodes; runtime use is still opt-in")
    r.add_argument("--dataset", required=True)
    r.add_argument("--prompt", required=True)
    r.add_argument("--k", type=int, default=5)
    r.add_argument("--vector_stride", type=int, default=2)
    r.add_argument("--max_vectors", type=int, default=512)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "collect":
        collect(args)
    elif args.cmd == "train":
        train(args)
    elif args.cmd == "retrieve":
        retrieve(args)


if __name__ == "__main__":
    main()
