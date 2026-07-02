# NEED setup and command guide

## 1. Start from the repository root

Confirm the core files are present:

```bash
ls train.py generate.py need_core.py build_corpuses.py prepare_packed_dataset.py
```

## 2. Run variables

Use variables rather than hard-coding one hardware target or one architecture. These examples are deliberately modest. Increase them only after the smoke test, preflight, and audit pass.

```bash
export NEED_DEVICE=auto
export NEED_PROFILE=small
export NEED_ARCHITECTURE=dense
export NEED_TARGET_PARAMS=100M
export NEED_TARGET_TOKENS=10M
export NEED_OUT_DIR=runs/need_main
export NEED_CORPUS_DIR=data/corpuses
export NEED_PACKED_DIR=data/packed
export NEED_PACKED_INDEX=${NEED_PACKED_DIR}/packed_index.json
export NEED_MANIFEST=data/mix_manifest.json
export NEED_CHECKPOINT=${NEED_OUT_DIR}
export NEED_PEAK_TFLOPS=0
```

Notes:

- `NEED_DEVICE=auto` lets the code choose CUDA when available and otherwise fall back.
- `NEED_PEAK_TFLOPS=0` disables MFU denominator assumptions. Set it only if you know the hardware peak number you want used for MFU logging.
- `NEED_PROFILE` can be `tiny`, `small`, `medium`, `large`, `reasoning`, `image`, `speed`, `long_context`, `multimodal`, `agentic`, `custom`, or another profile defined in `train.py`.
- Use `NEED_PROFILE=custom` plus explicit shape flags when you do not want a preset.
- Use `NEED_TARGET_PARAMS` with `config_for_size.py` or `train.py --target_params` when you want a generated model shape instead of a named preset.

Large numeric arguments accept suffixes such as `K`, `M`, `B`, and `T`:

```text
10K, 50M, 1B, 15B, 3.55B, 1T
```

## 3. Python environment

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch for your platform. Use the command that matches your OS, CUDA version, or CPU-only environment from the official PyTorch selector. For a generic install attempt:

```bash
pip install torch torchvision torchaudio
```

Install core project dependencies:

```bash
pip install numpy tqdm datasets transformers tokenizers pillow requests safetensors pandas
```

Install optional browser, adapter, and kernel dependencies when you use those features:

```bash
pip install gradio peft
pip install triton
```

Optional attention and acceleration packages are environment-specific. Install them only when they match your CUDA, Python, and PyTorch versions:

```bash
# Optional, only if compatible with your system:
# pip install flash-attn --no-build-isolation
```

## 4. Environment sanity checks

Check Python and PyTorch:

```bash
python --version
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
    print('capability:', torch.cuda.get_device_capability(0))
    print('bf16:', torch.cuda.is_bf16_supported())
PY
```

Compile-check the repository without starting training:

```bash
python -m py_compile *.py
```

List the primary command-line help pages:

```bash
python train.py --help
python generate.py --help
python browser.py --help
python terminal.py --help
python prepare_packed_dataset.py --help
python need_sidecar_distill.py --help
```

## 5. Quick smoke test

Run this before any larger setup. It uses a tiny corpus and CPU-compatible settings.

```bash
mkdir -p /tmp/need_smoke
cat > /tmp/need_smoke/train.jsonl <<'EOF_SMOKE'
{"text":"hello world this is a tiny NEED smoke test."}
{"text":"another short document for testing packed training."}
EOF_SMOKE
cat > /tmp/need_smoke/manifest.json <<'EOF_SMOKE'
{"sources":[{"name":"tiny","path":"/tmp/need_smoke/train.jsonl","weight":1.0,"domain":"general"}]}
EOF_SMOKE
```

Pack and train for a few steps:

```bash
python prepare_packed_dataset.py \
  --manifest /tmp/need_smoke/manifest.json \
  --out_dir /tmp/need_smoke/packed \
  --index_out /tmp/need_smoke/packed/packed_index.json \
  --target_tokens 10K

python train.py \
  --profile tiny \
  --recipe none \
  --packed_index /tmp/need_smoke/packed/packed_index.json \
  --out_dir /tmp/need_smoke/out \
  --device cpu \
  --max_steps 2 \
  --batch_size 1 \
  --grad_accum_steps 1 \
  --eval_interval 1 \
  --save_interval 1 \
  --disable_image \
  --minimal_aux_metrics
```

Resume the smoke test:

```bash
python train.py \
  --resume_from /tmp/need_smoke/out \
  --resume_strict \
  --profile tiny \
  --recipe none \
  --packed_index /tmp/need_smoke/packed/packed_index.json \
  --out_dir /tmp/need_smoke/out \
  --device cpu \
  --max_steps 4 \
  --batch_size 1 \
  --grad_accum_steps 1
```

Try generation from the smoke checkpoint:

```bash
python generate.py \
  --checkpoint /tmp/need_smoke/out \
  --device cpu \
  --decode_mode ar \
  --prompt "Say hello." \
  --max_new_tokens 24
```

## 6. Build or bring data

### 6.1 Build bundled corpus slices

Use the corpus builder when you want the repository to assemble data folders:

```bash
python build_corpuses.py \
  --build all \
  --out_dir ${NEED_CORPUS_DIR} \
  --target_tokens ${NEED_TARGET_TOKENS} \
  --tokenizer gpt2 \
  --clean
```

Preview the plan without writing data:

```bash
python build_corpuses.py \
  --build all \
  --out_dir ${NEED_CORPUS_DIR} \
  --target_tokens ${NEED_TARGET_TOKENS} \
  --tokenizer gpt2 \
  --dry_run \
  --print-budget
```

Build only a small knowledge slice for tests:

```bash
python build_corpuses.py \
  --build knowledge \
  --out_dir ${NEED_CORPUS_DIR}_test \
  --target_tokens 50M \
  --approx_tokens \
  --clean
```

### 6.2 Generate low-data RL/SFT/preferences data

Dry-run first. This writes plans and prompts without API calls:

```bash
python need_auto_low_data_rl.py \
  --out_dir ${NEED_CORPUS_DIR} \
  --target_profile whole_model \
  --total_examples auto \
  --batch_size auto \
  --write_prompts \
  --emit_control_interactions
```

Activate generation with an OpenAI-compatible endpoint only when your API key and model are configured:

```bash
export OPENAI_API_KEY=your_key_here
python need_auto_low_data_rl.py \
  --activate \
  --model ${GENERATOR_MODEL:-gpt-5.4} \
  --api_key_env OPENAI_API_KEY \
  --out_dir ${NEED_CORPUS_DIR} \
  --target_profile whole_model \
  --total_examples auto \
  --batch_size auto \
  --wikipedia_source popular,random,vital \
  --emit_control_interactions
```

The generator can emit rows for instruction following, preferences, RLVR, numeric evaluation, risk scoring, weighted decisions, tool routing, latent-tool behavior, structured JSON, behavioral memory, self-correction, sidecar latent alignment, and image-policy behavior. Transcript JSONL files can also be converted into control rows:

```bash
python need_auto_low_data_rl.py \
  --out_dir ${NEED_CORPUS_DIR} \
  --transcripts_file data/transcripts/events.jsonl \
  --transcripts_emit_as all \
  --emit_control_interactions
```

### 6.3 Filter corpus files

```bash
python quality_filter_corpus.py \
  --input ${NEED_CORPUS_DIR} \
  --out_dir data/filtered \
  --base filtered \
  --min_chars 20 \
  --max_chars 20000 \
  --min_score 0.0
```

## 7. Build a source-balanced packed dataset

Create a manifest. Adjust sources and weights to your actual files:

```bash
mkdir -p data
cat > ${NEED_MANIFEST} <<'EOF_MANIFEST'
{
  "sources": [
    {"name": "knowledge", "path": "data/corpuses/knowledge/train.jsonl", "weight": 0.40, "domain": "general"},
    {"name": "instruction", "path": "data/corpuses/rl/sft.synthetic.jsonl", "weight": 0.20, "domain": "instruction"},
    {"name": "preferences", "path": "data/corpuses/rl/preferences.synthetic.jsonl", "weight": 0.15, "domain": "preference"},
    {"name": "rlvr", "path": "data/corpuses/rl/rlvr.synthetic.jsonl", "weight": 0.15, "domain": "rlvr"},
    {"name": "other", "path": "data/corpuses/other/train.jsonl", "weight": 0.10, "domain": "mixed"}
  ]
}
EOF_MANIFEST
```

Pack the manifest:

```bash
python prepare_packed_dataset.py \
  --manifest ${NEED_MANIFEST} \
  --out_dir ${NEED_PACKED_DIR} \
  --index_out ${NEED_PACKED_INDEX} \
  --target_tokens ${NEED_TARGET_TOKENS} \
  --dtype auto
```

The packed-index path is preferred for large mixed-source training because it records source metadata and supports weighted sampling. The older single-file packed path still exists:

```bash
python train.py \
  --data ${NEED_CORPUS_DIR}/knowledge/train.jsonl \
  --pack_data_to data/packed_single/train.uint16.bin \
  --pack_only \
  --out_dir runs/pack_tmp

python train.py \
  --packed_data data/packed_single/train.uint16.bin \
  --out_dir ${NEED_OUT_DIR} \
  --device ${NEED_DEVICE}
```

## 8. Generate a model configuration or use a preset

Print a train command from a target parameter count:

```bash
python config_for_size.py \
  --params ${NEED_TARGET_PARAMS} \
  --architecture ${NEED_ARCHITECTURE} \
  --tokens ${NEED_TARGET_TOKENS} \
  --packed_index ${NEED_PACKED_INDEX} \
  --out_dir ${NEED_OUT_DIR} \
  --recipe none \
  --print_train_cmd
```

Write a generated configuration:

```bash
python config_for_size.py \
  --params ${NEED_TARGET_PARAMS} \
  --architecture ${NEED_ARCHITECTURE} \
  --tokens ${NEED_TARGET_TOKENS} \
  --packed_index ${NEED_PACKED_INDEX} \
  --out_dir ${NEED_OUT_DIR} \
  --write runs/generated_config.json
```

Named profiles are useful for fast iteration, but they are optional. Use custom sizing when you do not want a preset:

```bash
python train.py \
  --profile custom \
  --target_params ${NEED_TARGET_PARAMS} \
  --architecture ${NEED_ARCHITECTURE} \
  --packed_index ${NEED_PACKED_INDEX} \
  --out_dir ${NEED_OUT_DIR} \
  --device ${NEED_DEVICE} \
  --target_tokens ${NEED_TARGET_TOKENS}
```

Recipes tune runtime and optimizer behavior. They do not change the architecture fields. Use `--recipe none` for fully explicit control. Use `--recipe debug` for short runs, and use other recipes only when their defaults match your run.

## 9. Preflight and audit before training

Run preflight:

```bash
python preflight.py \
  --profile ${NEED_PROFILE} \
  --recipe none \
  --packed_index ${NEED_PACKED_INDEX} \
  --out_dir ${NEED_OUT_DIR} \
  --device ${NEED_DEVICE} \
  --target_tokens ${NEED_TARGET_TOKENS} \
  --peak_tflops ${NEED_PEAK_TFLOPS}
```

Run the static codebase and packed-index audit:

```bash
mkdir -p runs
python endgame_audit.py \
  --root . \
  --packed_index ${NEED_PACKED_INDEX} \
  --json_out runs/endgame_audit.json
```

Treat `ok: true` as no blocking audit errors. Review warnings before long runs.

## 10. Train

### 10.1 Generic packed-index training

```bash
python train.py \
  --profile ${NEED_PROFILE} \
  --recipe none \
  --packed_index ${NEED_PACKED_INDEX} \
  --out_dir ${NEED_OUT_DIR} \
  --device ${NEED_DEVICE} \
  --target_tokens ${NEED_TARGET_TOKENS} \
  --batch_size 1 \
  --grad_accum_steps 1 \
  --lr_schedule cosine \
  --warmup_steps 100 \
  --min_lr 1e-5 \
  --nan_recovery \
  --loss_spike_threshold 4.0 \
  --save_interval 1000 \
  --metrics_jsonl ${NEED_OUT_DIR}/train_log.jsonl
```

For CUDA runs where fixed shapes and compile are appropriate:

```bash
python train.py \
  --profile ${NEED_PROFILE} \
  --recipe none \
  --packed_index ${NEED_PACKED_INDEX} \
  --out_dir ${NEED_OUT_DIR} \
  --device cuda \
  --target_tokens ${NEED_TARGET_TOKENS} \
  --auto_optimize \
  --auto_batch \
  --target_vram_util 0.90 \
  --prefetch_to_device \
  --drop_last \
  --compile \
  --compile_mode max-autotune \
  --compile_cudagraphs \
  --compile_static \
  --nan_recovery \
  --loss_spike_threshold 4.0 \
  --peak_tflops ${NEED_PEAK_TFLOPS}
```

### 10.2 Stream from text or JSONL without packing

```bash
python train.py \
  --profile ${NEED_PROFILE} \
  --recipe none \
  --data ${NEED_CORPUS_DIR} \
  --stream_data \
  --out_dir ${NEED_OUT_DIR} \
  --device ${NEED_DEVICE} \
  --target_tokens ${NEED_TARGET_TOKENS}
```

### 10.3 Domain-specific validation

Repeat `--eval_data name=path` to track separate validation domains:

```bash
python train.py \
  --profile ${NEED_PROFILE} \
  --recipe none \
  --packed_index ${NEED_PACKED_INDEX} \
  --eval_data general=data/eval/general.jsonl \
  --eval_data code=data/eval/code.jsonl \
  --eval_data math=data/eval/math.jsonl \
  --eval_data instruction=data/eval/instruction.jsonl \
  --out_dir ${NEED_OUT_DIR} \
  --device ${NEED_DEVICE}
```

### 10.4 DVSD-aware training

Dynamic virtual-slot decoding works best when the model has multiple prediction heads and the DVSD losses are left enabled. The current code includes a learned slot-budget router and DVSD-native auxiliary losses:

```bash
python train.py \
  --profile ${NEED_PROFILE} \
  --recipe none \
  --packed_index ${NEED_PACKED_INDEX} \
  --out_dir ${NEED_OUT_DIR} \
  --device ${NEED_DEVICE} \
  --n_predict_heads 4 \
  --lambda_mtp 0.15 \
  --lambda_dvsd_slot_ce 0.025 \
  --lambda_dvsd_consistency 0.020 \
  --lambda_dvsd_router 0.015 \
  --dvsd_router_inference_mix 0.65 \
  --dvsd_router_min_confidence 0.20
```

Use `--disable_dvsd_router` only when you want the heuristic controller without learned slot routing.

### 10.5 Periodic samples and diagnostics

```bash
mkdir -p prompts
cat > prompts/train_samples.txt <<'EOF_PROMPTS'
Explain a simple scientific idea clearly.
Write a small Python function and explain it.
Solve a short arithmetic word problem.
Summarize a short historical event.
EOF_PROMPTS
```

Enable samples, module diagnostics, EMA, and recovery:

```bash
python train.py \
  --profile ${NEED_PROFILE} \
  --packed_index ${NEED_PACKED_INDEX} \
  --out_dir ${NEED_OUT_DIR} \
  --device ${NEED_DEVICE} \
  --sample_prompts prompts/train_samples.txt \
  --sample_interval 1000 \
  --sample_max_new_tokens 96 \
  --module_diagnostics_interval 500 \
  --ema_decay 0.999 \
  --nan_recovery \
  --max_nonfinite_events 20
```

Training writes `training_state.pt` with model, optimizer, scaler, RNG, counters, args, architecture config, and EMA state when enabled.

## 11. Resume and monitor

Resume normally:

```bash
python train.py \
  --resume_from ${NEED_OUT_DIR} \
  --packed_index ${NEED_PACKED_INDEX} \
  --out_dir ${NEED_OUT_DIR} \
  --device ${NEED_DEVICE}
```

Resume strictly, rejecting architecture/config mismatches:

```bash
python train.py \
  --resume_from ${NEED_OUT_DIR} \
  --resume_strict \
  --profile ${NEED_PROFILE} \
  --packed_index ${NEED_PACKED_INDEX} \
  --out_dir ${NEED_OUT_DIR} \
  --device ${NEED_DEVICE}
```

Only use unsafe checkpoint loading for old local checkpoints you created and trust:

```bash
python train.py \
  --resume_from ${NEED_OUT_DIR} \
  --allow_unsafe_checkpoint_load \
  --packed_index ${NEED_PACKED_INDEX} \
  --out_dir ${NEED_OUT_DIR} \
  --device ${NEED_DEVICE}
```

Monitor a run or inspect it after the fact:

```bash
python train_run_guard.py \
  --out_dir ${NEED_OUT_DIR} \
  --metrics_jsonl ${NEED_OUT_DIR}/train_log.jsonl \
  --max_bad_events 20 \
  --max_checkpoint_age_s 7200 \
  --json_out ${NEED_OUT_DIR}/run_guard.json
```

If MFU should be present because you supplied `--peak_tflops`, add:

```bash
python train_run_guard.py \
  --out_dir ${NEED_OUT_DIR} \
  --expect_mfu \
  --json_out ${NEED_OUT_DIR}/run_guard_mfu.json
```

Analyze logs, eval files, and generation traces:

```bash
python analyze_run.py \
  --run_dir ${NEED_OUT_DIR} \
  --logs ${NEED_OUT_DIR}/train_log.jsonl \
  --out_json ${NEED_OUT_DIR}/analysis.json \
  --out_html ${NEED_OUT_DIR}/analysis.html
```

## 12. Evaluate and benchmark

Evaluate a checkpoint:

```bash
python eval_need.py \
  --checkpoint ${NEED_CHECKPOINT} \
  --prefer_best \
  --data data/eval/general.jsonl \
  --device ${NEED_DEVICE} \
  --out_json ${NEED_OUT_DIR}/eval_general.json
```

Benchmark training throughput:

```bash
python throughput_benchmark.py \
  --need_profile ${NEED_PROFILE} \
  --device ${NEED_DEVICE} \
  --batch_size 1 \
  --warmup 10 \
  --iters 50 \
  --peak_tflops ${NEED_PEAK_TFLOPS}
```

Batch sweep:

```bash
python batch_sweep.py \
  --need_profile ${NEED_PROFILE} \
  --device ${NEED_DEVICE} \
  --batch_sizes 1,2,4,8 \
  --warmup 10 \
  --iters 50 \
  --peak_tflops ${NEED_PEAK_TFLOPS}
```

Short LR stability sweep:

```bash
python lr_sweep.py \
  --profile ${NEED_PROFILE} \
  --recipe none \
  --device ${NEED_DEVICE} \
  --lrs 5e-5,1e-4,2e-4 \
  --steps 100 \
  --batch_size 1 \
  --out_json runs/lr_sweep.json
```

Print smaller scaling commands for ablations:

```bash
python ablation_grid.py \
  --packed_index ${NEED_PACKED_INDEX} \
  --out_root runs/ablations \
  --device ${NEED_DEVICE} \
  --points 25M:50M,100M:500M,300M:2B \
  --recipe none
```

Benchmark generation:

```bash
python generation_benchmark.py \
  --checkpoint ${NEED_CHECKPOINT} \
  --device ${NEED_DEVICE} \
  --prompt_file prompts/train_samples.txt \
  --max_new_tokens 128 \
  --decode_mode auto \
  --iters 5 \
  --prefer_best
```

Calibrate DVSD versus AR on the same prompt set:

```bash
python dvsd_calibration.py \
  --checkpoint ${NEED_CHECKPOINT} \
  --device ${NEED_DEVICE} \
  --prefer_best \
  --prompt_file prompts/train_samples.txt \
  --max_new_tokens 128 \
  --out_jsonl runs/dvsd_calibration.jsonl \
  --out_summary_json runs/dvsd_calibration_summary.json
```

## 13. Generate text

Basic CLI generation:

```bash
python generate.py \
  --checkpoint ${NEED_CHECKPOINT} \
  --prefer_best \
  --device ${NEED_DEVICE} \
  --prompt "Explain what this model is doing." \
  --max_new_tokens 128
```

Use a prompt file:

```bash
python generate.py \
  --checkpoint ${NEED_CHECKPOINT} \
  --prompt_file prompts/train_samples.txt \
  --out_file runs/generated.txt \
  --max_new_tokens 128
```

Force autoregressive decoding:

```bash
python generate.py \
  --checkpoint ${NEED_CHECKPOINT} \
  --decode_mode ar \
  --prompt "Test AR decoding." \
  --max_new_tokens 128
```

Use DVSD when trained MTP heads are available:

```bash
python generate.py \
  --checkpoint ${NEED_CHECKPOINT} \
  --decode_mode auto \
  --nonseq_dynamic \
  --nonseq_min_heads 1 \
  --nonseq_max_heads 0 \
  --nonseq_refine_steps 3 \
  --nonseq_refine_causal_blend 0.55 \
  --nonseq_refine_temperature_decay 0.82 \
  --nonseq_refine_lock_schedule cosine \
  --prompt "Test DVSD decoding." \
  --max_new_tokens 128
```

Override learned DVSD router behavior:

```bash
python generate.py \
  --checkpoint ${NEED_CHECKPOINT} \
  --prompt "Test learned router settings." \
  --dvsd_router_enabled \
  --dvsd_router_inference_mix 0.75 \
  --dvsd_router_min_confidence 0.25
```

Use hidden deterministic latent tools. Calculator is on by default; Python execution is off unless explicitly enabled:

```bash
python generate.py \
  --checkpoint ${NEED_CHECKPOINT} \
  --prompt "Compute 17 * 23 and explain briefly." \
  --latent_tools \
  --latent_tool_calculator \
  --no-latent_tool_python
```

Enable latent memory explicitly:

```bash
python generate.py \
  --checkpoint ${NEED_CHECKPOINT} \
  --prompt "Use prior successful behavior if relevant." \
  --latent_memory_dir runs/latent_memory \
  --use_latent_memory \
  --latent_memory_k 4
```

## 14. Single-sidecar runtime

The current runtime loads exactly one sidecar backend: `none`, `external_lm`, or `need`. In `auto`, a configured NEED sidecar takes priority over an external-LM sidecar. This prevents accidentally running two advisory models in the same decode path.

Disable sidecars:

```bash
python generate.py \
  --checkpoint ${NEED_CHECKPOINT} \
  --sidecar_type none \
  --prompt "Run without a sidecar."
```

Use an external LM sidecar:

```bash
python generate.py \
  --checkpoint ${NEED_CHECKPOINT} \
  --sidecar_type external_lm \
  --sidecar_model ${SIDECAR_MODEL} \
  --sidecar_device same \
  --sidecar_dtype bf16 \
  --prompt "Use the configured external sidecar."
```

Use a smaller NEED checkpoint as the sidecar:

```bash
export NEED_SIDECAR_CHECKPOINT=checkpoints/need_sidecar
python generate.py \
  --checkpoint ${NEED_CHECKPOINT} \
  --sidecar_type need \
  --need_sidecar_checkpoint ${NEED_SIDECAR_CHECKPOINT} \
  --need_sidecar_projection_path ${NEED_SIDECAR_CHECKPOINT} \
  --use_need_sidecar_latents \
  --prompt "Use the architecture-native sidecar."
```

Use latent-gated sidecar calls so easy prompts skip sidecar work:

```bash
python generate.py \
  --checkpoint ${NEED_CHECKPOINT} \
  --sidecar_type need \
  --need_sidecar_checkpoint ${NEED_SIDECAR_CHECKPOINT} \
  --need_sidecar_projection_path ${NEED_SIDECAR_CHECKPOINT} \
  --sidecar_call_policy latent_gated \
  --sidecar_gate_metric latent_difficulty \
  --sidecar_gate_threshold 0.42 \
  --prompt "Use sidecar help only if the latent path looks difficult."
```

Use `--sidecar_call_policy always` for always-call behavior and `--sidecar_call_policy off` to keep a configured sidecar loaded but unused by the reasoning-prep path.

External-LM speculative final decoding remains gated to sidecars that report support for it. A smaller NEED sidecar supplies public summaries and projected latent anchors; it is not treated as a verified final-answer acceptance model by default.

## 15. Train and test a smaller NEED sidecar

Distill a smaller NEED sidecar from a larger NEED teacher:

```bash
python need_sidecar_distill.py train \
  --teacher_checkpoint ${NEED_CHECKPOINT} \
  --teacher_prefer_best \
  --corpus ${NEED_CORPUS_DIR} \
  --out_dir checkpoints/need_sidecar \
  --target_params ${NEED_SIDECAR_TARGET_PARAMS:-30M} \
  --steps 600 \
  --batch_size 4 \
  --device ${NEED_DEVICE}
```

Use CKA-style representation matching for aggressive compression:

```bash
python need_sidecar_distill.py train \
  --teacher_checkpoint ${NEED_CHECKPOINT} \
  --corpus ${NEED_CORPUS_DIR} \
  --out_dir checkpoints/need_sidecar \
  --target_params ${NEED_SIDECAR_TARGET_PARAMS:-30M} \
  --cka_hidden_weight 0.15 \
  --cka_future_state_weight 0.05 \
  --steps 600 \
  --device ${NEED_DEVICE}
```

Smoke-test the sidecar and projection:

```bash
python need_sidecar_distill.py smoke \
  --teacher_checkpoint ${NEED_CHECKPOINT} \
  --sidecar_checkpoint checkpoints/need_sidecar \
  --projection_path checkpoints/need_sidecar \
  --prompt "Explain the role of a smaller NEED sidecar." \
  --device ${NEED_DEVICE}
```

## 16. Low-data adapter and runtime-profile workflow

Tune controller, verifier, and verbalization surfaces from interaction data:

```bash
python need_low_data_adapters.py tune_control \
  --checkpoint ${NEED_CHECKPOINT} \
  --prefer_best \
  --interactions ${NEED_CORPUS_DIR}/rl/low_data_control_interactions.synthetic.jsonl \
  --out_dir runs/low_data/control \
  --steps 300 \
  --batch_size 4 \
  --device ${NEED_DEVICE}
```

Calibrate adaptive speculative acceptance from traces:

```bash
python need_low_data_adapters.py calibrate_spec \
  --traces runs/generation_traces.jsonl \
  --out_file runs/low_data/spec_profile.json \
  --target_accept_rate 0.78
```

Build latent memory from successful episodes:

```bash
python need_low_data_adapters.py build_memory \
  --checkpoint ${NEED_CHECKPOINT} \
  --interactions ${NEED_CORPUS_DIR}/rl/low_data_control_interactions.synthetic.jsonl \
  --out_dir runs/latent_memory \
  --device ${NEED_DEVICE}
```

Bundle runtime settings into one profile:

```bash
python need_low_data_adapters.py make_runtime_profile \
  --out_file runs/runtime_profile.json \
  --interactions ${NEED_CORPUS_DIR}/rl/low_data_control_interactions.synthetic.jsonl \
  --spec_profile runs/low_data/spec_profile.json \
  --latent_memory_dir runs/latent_memory \
  --latent_memory_k 4 \
  --replay_context_k 3 \
  --replay_context_similarity hybrid
```

Use the runtime profile:

```bash
python generate.py \
  --checkpoint ${NEED_CHECKPOINT} \
  --runtime_profile runs/runtime_profile.json \
  --prompt "Use the bundled runtime profile."
```

One-command low-data starter, external-LM sidecar path:

```bash
python need_low_data_rl_start.py start \
  --need_checkpoint ${NEED_CHECKPOINT} \
  --sidecar_backend external_lm \
  --sidecar_model ${SIDECAR_MODEL} \
  --out_root runs/low_data_start_external \
  --target_profile whole_model \
  --total_examples 3000 \
  --batch_size 50
```

One-command low-data starter, NEED sidecar path:

```bash
python need_low_data_rl_start.py start \
  --need_checkpoint ${NEED_CHECKPOINT} \
  --sidecar_backend need \
  --need_sidecar_target_params ${NEED_SIDECAR_TARGET_PARAMS:-30M} \
  --out_root runs/low_data_start_need_sidecar \
  --target_profile whole_model \
  --total_examples 3000 \
  --batch_size 50
```

That path trains or exports a smaller NEED sidecar, writes `sidecar_type=need` into the runtime profile, includes the sidecar checkpoint path, and adds DVSD-router and latent-gated sidecar defaults.

## 17. External sidecar latent alignment utilities

Build NEED latent targets for sidecar adapter training:

```bash
python need_thought_distill.py build_alignment_dataset \
  --need_checkpoint ${NEED_CHECKPOINT} \
  --prefer_best \
  --corpus ${NEED_CORPUS_DIR} \
  --out_dir runs/sidecar_alignment_data \
  --sidecar_model ${SIDECAR_MODEL} \
  --device ${NEED_DEVICE}
```

Train a sidecar alignment adapter or projection:

```bash
python need_thought_distill.py train_alignment \
  --sidecar_model ${SIDECAR_MODEL} \
  --dataset_jsonl runs/sidecar_alignment_data/alignment_dataset.jsonl \
  --latents_pt runs/sidecar_alignment_data/need_latents.pt \
  --out_dir runs/sidecar_alignment \
  --train_mode lora \
  --steps 600 \
  --batch_size 4 \
  --device ${NEED_DEVICE}
```

Evaluate alignment:

```bash
python need_thought_distill.py eval_alignment \
  --sidecar_model ${SIDECAR_MODEL} \
  --dataset_jsonl runs/sidecar_alignment_data/alignment_dataset.jsonl \
  --latents_pt runs/sidecar_alignment_data/need_latents.pt \
  --sidecar_latent_alignment_path runs/sidecar_alignment \
  --device ${NEED_DEVICE}
```

Run the external-sidecar summary path:

```bash
python need_thought_distill.py run \
  --need_checkpoint ${NEED_CHECKPOINT} \
  --prompt "Summarize this task for NEED." \
  --sidecar_model ${SIDECAR_MODEL} \
  --device ${NEED_DEVICE}
```

## 18. Browser and terminal interfaces

Launch the browser UI:

```bash
python browser.py \
  --checkpoint ${NEED_CHECKPOINT} \
  --prefer_best \
  --device ${NEED_DEVICE} \
  --decode_mode auto \
  --sidecar_type none \
  --host 127.0.0.1 \
  --port 7860
```

Browser with a smaller NEED sidecar:

```bash
python browser.py \
  --checkpoint ${NEED_CHECKPOINT} \
  --decode_mode auto \
  --sidecar_type need \
  --need_sidecar_checkpoint checkpoints/need_sidecar \
  --need_sidecar_projection_path checkpoints/need_sidecar \
  --use_need_sidecar_latents \
  --sidecar_call_policy latent_gated
```

Launch terminal chat:

```bash
python terminal.py \
  --checkpoint ${NEED_CHECKPOINT} \
  --device ${NEED_DEVICE} \
  --decode_mode auto \
  --sidecar_type none \
  --display_mode stream_tokens
```

Terminal with a runtime profile:

```bash
python terminal.py \
  --checkpoint ${NEED_CHECKPOINT} \
  --runtime_profile runs/runtime_profile.json \
  --display_mode stream_tokens
```

The browser and terminal now share the same DVSD and single-sidecar assumptions as `generate.py`: `--decode_mode auto|ar|nonseq`, `--nonseq_*` controls, `--sidecar_type auto|none|external_lm|need`, NEED-sidecar projection paths, latent-gated sidecar calls, and external-LM-only speculative final decoding.

## 19. Optional image-data and image-token workflow

Prepare a filtered image manifest from local images, URL lists, or manifests:

```bash
python need_raw_image_data.py prepare \
  --local_dir data/raw_images \
  --out_dir data/image_prepared \
  --source_name local_images \
  --copy_local \
  --min_width 128 \
  --min_height 128 \
  --max_aspect_ratio 3.0 \
  --max_images 0
```

Tokenize prepared images:

```bash
python need_raw_image_data.py tokenize \
  --manifest data/image_prepared/manifest.jsonl \
  --out_dir data/image_tokens \
  --visual_tokenizer checkpoints/visual_tokenizer \
  --device ${NEED_DEVICE} \
  --image_codebook_size 512 \
  --grid 16 \
  --max_image_tokens 1024
```

Prepare, tokenize, optionally train a visual tokenizer, and optionally start image-token NEED training:

```bash
python need_raw_image_data.py start \
  --local_dir data/raw_images \
  --out_dir data/image_pipeline \
  --copy_local \
  --train_visual_tokenizer \
  --train_need \
  --need_profile image \
  --need_steps 1000 \
  --need_batch_size 4 \
  --device ${NEED_DEVICE}
```

Train with image data directly:

```bash
python train.py \
  --profile image \
  --data ${NEED_CORPUS_DIR} \
  --image_dir data/image_tokens \
  --out_dir runs/need_image \
  --device ${NEED_DEVICE} \
  --image_ratio 0.25 \
  --visual_tokenizer checkpoints/visual_tokenizer
```

Generate image tokens or image outputs when the checkpoint and visual tokenizer support image mode:

```bash
python generate.py \
  --checkpoint ${NEED_CHECKPOINT} \
  --mode image \
  --prompt "A simple test image prompt." \
  --visual_tokenizer checkpoints/visual_tokenizer \
  --image_out runs/image_out.png \
  --image_steps 64 \
  --image_size 256
```

## 20. Checkpoint utilities

Inspect a checkpoint:

```bash
python checkpoint_tools.py inspect ${NEED_CHECKPOINT}
```

Average checkpoints:

```bash
python checkpoint_tools.py average \
  checkpoints/a \
  checkpoints/b \
  --out checkpoints/averaged
```

Diff checkpoints:

```bash
python checkpoint_tools.py diff \
  checkpoints/a \
  checkpoints/b \
  --out_json runs/checkpoint_diff.json
```

Strip a checkpoint for inference or text-only use:

```bash
python checkpoint_tools.py strip \
  ${NEED_CHECKPOINT} \
  --keep inference \
  --out checkpoints/stripped_inference
```

## 21. Optional run report

Create a neutral run report from available artifacts:

```bash
python need_run_card.py \
  --run_root ${NEED_OUT_DIR} \
  --version_label run_summary \
  --manifest ${NEED_MANIFEST} \
  --state ${NEED_OUT_DIR}/training_state.pt \
  --audit_json runs/endgame_audit.json \
  --runtime_profile runs/runtime_profile.json \
  --out_json ${NEED_OUT_DIR}/run_report.json \
  --out_md ${NEED_OUT_DIR}/run_report.md \
  --out_html ${NEED_OUT_DIR}/run_report.html
```

This is optional and does not force a particular report format or model configuration.


## 22. Additional audit, comparison, replay, curriculum, and pipeline commands

Audit datasets before training:

```bash
python need_dataset_audit.py \
  ${NEED_CORPUS_DIR} \
  --out_json runs/dataset_audit.json \
  --out_html runs/dataset_audit.html \
  --failures_jsonl runs/dataset_audit_failures.jsonl
```

Run the broader evaluation suite and regression scorecard:

```bash
python need_eval_suite.py \
  --checkpoint ${NEED_CHECKPOINT} \
  --prefer_best \
  --cases_jsonl data/eval/cases.jsonl \
  --audit_json runs/dataset_audit.json \
  --out_dir runs/evals \
  --out_json runs/evals/scorecard.json \
  --out_html runs/evals/scorecard.html \
  --device ${NEED_DEVICE}
```

Compare two checkpoints on the same prompts:

```bash
python need_checkpoint_compare.py \
  --checkpoint_a checkpoints/a \
  --checkpoint_b checkpoints/b \
  --prefer_best_a \
  --prefer_best_b \
  --prompts_jsonl data/eval/prompts.jsonl \
  --out_dir runs/compare \
  --out_json runs/compare/summary.json \
  --out_jsonl runs/compare/results.jsonl \
  --out_html runs/compare/report.html \
  --device ${NEED_DEVICE}
```

Run the compact eval dashboard command:

```bash
python need_eval.py \
  --checkpoint ${NEED_CHECKPOINT} \
  --prefer_best \
  --data data/eval/general.jsonl \
  --device ${NEED_DEVICE} \
  --batches 10 \
  --batch_size 4 \
  --dashboard \
  --out_json runs/need_eval_dashboard.json
```

Collect opt-in latent experience replay examples:

```bash
python need_experience_replay.py collect \
  --checkpoint ${NEED_CHECKPOINT} \
  --prefer_best \
  --interactions ${NEED_CORPUS_DIR}/rl/low_data_control_interactions.synthetic.jsonl \
  --out_dir runs/replay_dataset \
  --device ${NEED_DEVICE} \
  --augment_replay
```

Train replay adaptation surfaces:

```bash
python need_experience_replay.py train \
  --checkpoint ${NEED_CHECKPOINT} \
  --prefer_best \
  --dataset runs/replay_dataset \
  --out_dir runs/replay_tuned \
  --steps 1000 \
  --device ${NEED_DEVICE}
```

Inspect nearest replay episodes for a prompt:

```bash
python need_experience_replay.py retrieve \
  --checkpoint ${NEED_CHECKPOINT} \
  --dataset runs/replay_dataset \
  --prompt "Inspect replay retrieval for this prompt." \
  --k 5 \
  --device ${NEED_DEVICE}
```

Train a baseline attention LM for comparison:

```bash
python train_baseline_attention_lm.py \
  --data ${NEED_CORPUS_DIR} \
  --out_dir runs/baseline_attention \
  --device ${NEED_DEVICE} \
  --block_size 512 \
  --d_model 256 \
  --n_layers 6 \
  --n_heads 8 \
  --batch_size 4 \
  --max_steps 1000
```

Run curriculum training phases:

```bash
python train_curriculum.py \
  --corpus_dir ${NEED_CORPUS_DIR} \
  --out_dir runs/need_curriculum \
  --profile ${NEED_PROFILE} \
  --tokens ${NEED_TARGET_TOKENS} \
  --total_steps 1000 \
  --batch_size 4 \
  --grad_accum_steps 1 \
  --device ${NEED_DEVICE} \
  --run
```

Use the full pipeline orchestrator in dry-plan mode before allowing it to run stages:

```bash
python need_full_training_pipeline.py start \
  --out_root runs/need_full_pipeline \
  --dry_plan_only \
  --need_profile ${NEED_PROFILE} \
  --params_m 100 \
  --tokens_per_param 100 \
  --low_data_examples 3000 \
  --device ${NEED_DEVICE}
```

Start or resume the full pipeline only after reviewing the dry plan:

```bash
python need_full_training_pipeline.py start \
  --out_root runs/need_full_pipeline \
  --resume \
  --need_profile ${NEED_PROFILE} \
  --total_steps 4000 \
  --batch_size 4 \
  --device ${NEED_DEVICE}

python need_full_training_pipeline.py status \
  --out_root runs/need_full_pipeline

python need_full_training_pipeline.py audit \
  --out_root runs/need_full_pipeline

python need_full_training_pipeline.py eval \
  --out_root runs/need_full_pipeline

python need_full_training_pipeline.py card \
  --out_root runs/need_full_pipeline
```

## 23. Argument References

Use help commands to inspect the exact flags in the current codebase

```bash
python [FILE NAME].py --help
```

Some files are libraries rather than user-facing commands, so they may not have a CLI help page even though they are required by other commands.
If browser launch fails, confirm `gradio` is installed and that the selected port is free.
