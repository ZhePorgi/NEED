#!/usr/bin/env python3
"""Small CPU smoke checks for NEED core invariants.

This is intentionally lightweight so it can run without external datasets,
Hugging Face downloads, CUDA, or a trained checkpoint.
"""
from __future__ import annotations

import math
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from need_core import ByteTokenizer, NeedConfig, NeedModel, Special, load_model, save_model, save_json


def tiny_config(**overrides):
    data = dict(
        d_model=16,
        n_layers=1,
        n_heads=4,
        block_size=8,
        vocab_size=Special.text_vocab + 8,
        image_codebook_size=8,
        d_ff=32,
        latent_slots=1,
        memory_slots=2,
        memory_rank=8,
        n_experts=1,
        moe_top_k=1,
        n_predict_heads=2,
        planner_horizons=1,
        pathway_memory_slots=2,
        pathway_memory_top_k=1,
        exact_recall=False,
        object_program_slots=1,
        energy_rank=8,
        output_modes=5,
        energy_routes=1,
        dropout=0.0,
        collect_aux_metrics=False,
    )
    data.update(overrides)
    return NeedConfig(**data)


def _training_input_guards() -> None:
    from train import (
        _load_packed_input_tokenizer, _numpy_dtype_for_tokens, _packed_dtype_for_file, _packed_metadata_path,
        _validate_packed_tokenizer_compatible, _validate_packed_vocab_compatible,
        build_arg_parser, pack_text_tokens_to_bin,
    )

    assert _numpy_dtype_for_tokens("auto", 70000) == np.dtype(np.uint32)
    try:
        _numpy_dtype_for_tokens("uint16", 70000)
    except ValueError:
        pass
    else:
        raise AssertionError("uint16 packed-token overflow guard did not fire")

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        data = tmp / "data.jsonl"
        data.write_text('{"text":"hello"}\n', encoding="utf-8")
        try:
            pack_text_tokens_to_bin(data, tmp / "bad.bin", ByteTokenizer(), vocab_size=70000, dtype_name="uint16")
        except ValueError:
            pass
        else:
            raise AssertionError("pack_text_tokens_to_bin allowed an overflowing dtype")

        class BadTokenizer(ByteTokenizer):
            def encode(self, text: str, add_bos: bool = False, add_eos: bool = False):
                return [9999]

        try:
            pack_text_tokens_to_bin(data, tmp / "bad_range.bin", BadTokenizer(), vocab_size=100, dtype_name="uint16")
        except ValueError:
            pass
        else:
            raise AssertionError("pack_text_tokens_to_bin allowed out-of-vocabulary token IDs")

        packed = tmp / "tokens.bin"
        packed.write_bytes(b"\x00\x00")
        _packed_metadata_path(packed).write_text(json.dumps({"vocab_size": Special.text_vocab + 7, "dtype": "uint16", "tokens": 1}), encoding="utf-8")
        args = build_arg_parser().parse_args(["--packed_data", str(packed)])
        cfg = tiny_config(image_codebook_size=8)
        cfg.validate()
        try:
            _validate_packed_vocab_compatible(args, cfg)
        except ValueError:
            pass
        else:
            raise AssertionError("packed vocab mismatch guard did not fire")
        args.allow_packed_vocab_mismatch = True
        _validate_packed_vocab_compatible(args, cfg)
        _packed_metadata_path(packed).write_text(json.dumps({"vocab_size": 70000, "dtype": "uint16", "tokens": 1}), encoding="utf-8")
        try:
            _packed_dtype_for_file(packed, "auto", 70000)
        except ValueError:
            pass
        else:
            raise AssertionError("packed metadata dtype/vocab guard did not fire")
        _packed_metadata_path(packed).write_text(json.dumps({"vocab_size": Special.text_vocab + 7, "dtype": "uint16", "tokens": 1}), encoding="utf-8")

        save_json(ByteTokenizer().to_dict(), tmp / "tokenizer.json")
        loaded_tok = _load_packed_input_tokenizer(args)
        assert isinstance(loaded_tok, ByteTokenizer)
        save_json({"type": "hf", "vocab_size": ByteTokenizer().vocab_size, "model_name": "mismatch"}, tmp / "tokenizer.json")
        args.allow_packed_vocab_mismatch = False
        try:
            _validate_packed_tokenizer_compatible(args, ByteTokenizer())
        except ValueError:
            pass
        else:
            raise AssertionError("packed tokenizer sidecar mismatch guard did not fire")


def main() -> None:
    _training_input_guards()
    alias_cfg = NeedConfig.from_dict({
        "hidden_size": 16,
        "num_layers": 1,
        "num_heads": 4,
        "max_seq_len": 8,
        "n_latent_slots": 1,
        "image_codebook_size": 8,
    })
    assert alias_cfg.d_model == 16
    assert alias_cfg.n_layers == 1
    assert alias_cfg.block_size == 8
    assert alias_cfg.latent_slots == 1

    torch.manual_seed(123)
    cfg = tiny_config(streaming_generation=True)
    model = NeedModel(cfg).eval()
    ids = torch.tensor([[Special.bos, 16, 17, 18, 19, 20, 21, 22, 23]], dtype=torch.long)

    # Long helper inputs are cropped to the model window instead of failing.
    pathway = model.latent_pathway(ids, stride=0, max_vectors=0)
    assert tuple(pathway["pathway_vectors"].shape) == (1, 1, cfg.d_model)
    assert model.internal_reasoning_summary(ids, max_tokens=3).shape == (1, 3)
    assert set(model.output_mode_decision(ids))
    assert model.generate_text(ids[:, :2], max_new_tokens=0).shape == (1, 2)

    model.train()
    train_ids = ids[:, -cfg.block_size:]
    logits, loss, _ = model(train_ids, train_ids)
    assert logits.shape == (1, cfg.block_size, cfg.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()

    model.eval()
    image_tokens = model.generate_image_tokens(ids, grid=2, steps=1)
    assert image_tokens.shape == (1, 4)

    out, stats = model.generate_text_nonsequential(ids[:, :2], max_new_tokens=1, temperature=0.0, return_stats=True)
    assert out.size(1) == 3
    assert all(math.isfinite(float(v)) for v in stats.values())

    with tempfile.TemporaryDirectory() as tmp:
        save_model(model, tmp)
        loaded = load_model(tmp)
        assert sum(p.numel() for p in loaded.parameters()) == sum(p.numel() for p in model.parameters())

    # Streaming prefill should match full-context logits for a supported request.
    stream_cfg = tiny_config(exact_recall=True, exact_recall_max_candidates=6, exact_recall_top_k=2, streaming_generation=True)
    stream_model = NeedModel(stream_cfg).eval()
    stream_ids = torch.tensor([[Special.bos, 16, 17, 18]], dtype=torch.long)
    with torch.no_grad():
        full_logits, _, _ = stream_model(stream_ids)
        cache = stream_model._stream_new_cache(stream_ids.size(0), stream_ids.device, stream_model.token_emb.weight.dtype)
        stream_logits, _ = stream_model._stream_prefill(stream_ids, cache)
    assert torch.allclose(full_logits[:, -1], stream_logits, atol=1e-5, rtol=1e-5)

    print("core_smoke_test: ok")


if __name__ == "__main__":
    main()
