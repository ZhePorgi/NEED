# NEED Setup Guide

NEED ("Nested Energy Equilibrium Descent") is this repo's alternative to a standard
transformer stack. Everything - sizing a model, training it, evaluating it, comparing
runs, and talking to it - is driven from the scripts in this directory. There is no
package to `pip install`; you run the `.py` files directly.

Made for a Linux or macOS environment, if you want to use these commands on Windows machine and not have to translate them, search for WSL (Windows Subsystem for Linux).

This guide covers:

1. [Environment setup](#1-environment-setup)
2. [Sizing and starting a model](#2-sizing-and-starting-a-model)
3. [Training from the terminal](#3-training-from-the-terminal)
4. [Tracking and comparing runs](#4-tracking-and-comparing-runs)
5. [Running the model in the terminal](#5-running-the-model-in-the-terminal)
6. [Running the model in the browser](#6-running-the-model-in-the-browser)
7. [The one-command full pipeline](#7-the-one-command-full-pipeline)
8. [Flags reference](#8-flags-reference)

---

## 1. Environment setup

### 1.1 Requirements

There's no `requirements.txt` in the repo, so install these yourself:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Core (required)
pip install torch numpy transformers

# Interactive browser UI (only needed for browser.py)
pip install gradio

# Optional external "sidecar" LM for dual-channel reasoning / speculative decoding
pip install accelerate

# Optional faster kernels (only if you pass --kernel_backend triton)
pip install triton
```

Everything else imported by the scripts (`argparse`, `json`, `pathlib`, etc.) is
Python standard library.

`transformers` is required because the default text tokenizer is a Hugging Face
tokenizer (`deepseek-ai/DeepSeek-V4-Pro` by default; override with `--tokenizer_model`
or the `NEED_TOKENIZER_MODEL` env var). Loading it the first time downloads the
tokenizer files from Hugging Face, so you'll need network access once (they're then
cached locally). If you'd rather avoid the dependency/download entirely, pass
`--tokenizer byte` to `train.py` / `prepare_packed_dataset.py` to use the exact
byte-level fallback tokenizer instead - inference scripts automatically detect and
match whichever tokenizer a given checkpoint was trained with.

- **GPU**: any recent CUDA build of PyTorch works. CPU works too (slow for training,
  fine for small models/inference).
- **Python**: 3.10+ recommended (the code uses `from __future__ import annotations`
  and modern type-hint syntax like `list[Path]`).

### 1.2 Sanity check

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 1.3 What's in the repo

| Area | Key files |
|---|---|
| Model + tokenizer core | `need_core.py` |
| Training | `train.py`, `train_curriculum.py`, `train_baseline_attention_lm.py` |
| Sizing / planning | `config_for_size.py`, `preflight.py`, `ablation_grid.py`, `lr_sweep.py` |
| Data | `build_corpuses.py`, `prepare_packed_dataset.py`, `quality_filter_corpus.py`, `need_data.py`, `need_raw_image_data.py` |
| Inference | `generate.py` (library + CLI), `terminal.py` (interactive terminal), `browser.py` (Gradio UI) |
| Sidecars / distillation | `need_sidecar.py`, `need_sidecar_distill.py`, `need_thought_distill.py`, `sidecar_lm_runtime.py` |
| Low-data adaptation & RL | `need_low_data_adapters.py`, `need_low_data_rl_start.py`, `need_auto_low_data_rl.py`, `need_image_rl.py` |
| Evaluation / auditing | `need_eval.py`, `need_behavioral_eval.py`, `need_dataset_audit.py` |
| Tracking / reporting | `analyze_run.py`, `need_run_card.py` |
| Orchestration | `need_full_training_pipeline.py` |

---

## 2. Sizing and starting a model

Before training, use `config_for_size.py` to translate a parameter budget (and your
hardware) into a concrete architecture + training command. It writes JSON describing
`d_model`, `n_layers`, `n_heads`, etc., and can print the exact `train.py` invocation.

```bash
python config_for_size.py \
  --params 300M \
  --architecture dense \
  --tokens 6B \
  --modality text \
  --gpu_mem_gb 24 \
  --data data/corpuses/knowledge/train.jsonl \
  --recipe fast \
  --out_dir need_out \
  --write config.json \
  --print_train_cmd
```

Key flags:

| Flag | Purpose |
|---|---|
| `--params` | Target parameter count, e.g. `30M`, `300M`, `1.2B` (K/M/B/T suffixes accepted). |
| `--architecture` | `dense` or `moe` (mixture-of-experts). |
| `--tokens` | Target training tokens (used to size LR schedule / steps), e.g. `20B`. |
| `--modality` | `text`, `image`, `multimodal`, or `long_context` - shifts default shapes. |
| `--gpu_mem_gb` / `--ram_gb` / `--vcpus` | Hardware budget used to keep the plan realistic. |
| `--recipe` | Named training-defaults bundle: `fast`, `quality`, `baseline`, `debug`, or empty. |
| `--write` | Path to save the generated config JSON. |
| `--print_train_cmd` | Print a ready-to-run `train.py` command for this config. |

Then, before committing to a long run, use `preflight.py` to catch problems early
(bad data paths, OOM-prone batch sizes, corrupted packed indices, unrealistic step
counts):

```bash
python preflight.py \
  --target_params 300M \
  --recipe fast \
  --packed_index data/packed/packed_index.json \
  --out_dir runs/preflight \
  --device auto
```

If you want a battery of small comparison runs (different sizes, or a memory/role
ablation) instead of a single plan, `ablation_grid.py` prints (or saves) the concrete
`train.py` commands for a whole sweep:

```bash
python ablation_grid.py \
  --packed_index data/packed/packed_index.json \
  --min_params 100M --max_params 1B --runs 3 \
  --tokens_per_param 20 \
  --recipe fast \
  --out_root runs/ablations \
  --role_ablation
```

And `lr_sweep.py` runs a short, cheap learning-rate stability sweep at a given size
before you commit to a real run:

```bash
python lr_sweep.py \
  --target_params 600M \
  --recipe debug \
  --lrs 5e-5,1e-4,1.35e-4,1.6e-4,2e-4 \
  --steps 50 \
  --out_json runs/lr_sweep/result.json
```

---

## 3. Training from the terminal

The main entry point is `train.py`. It has hundreds of flags (see
[§8 Flags reference](#8-flags-reference)), but a typical run only needs a handful:

```bash
python train.py \
  --data data/corpuses/knowledge/train.jsonl \
  --out_dir need_out \
  --target_params 300M \
  --recipe fast \
  --max_steps 20000 \
  --batch_size 16 \
  --grad_accum_steps 2 \
  --lr 3e-4 \
  --amp bf16 \
  --device auto \
  --save_interval 1000 \
  --eval_interval 200 \
  --log_interval 20
```

Notes:

- **Packed data is faster than raw text.** Use `prepare_packed_dataset.py` (or
  `train.py --pack_data_to ...` / `--pack_only`) to tokenize once into a flat binary
  stream, then train with `--packed_data` / `--packed_index` instead of `--data`.
- **Resuming**: `--init_from` continues weights only (fine-tuning); `--resume_from`
  restores a full `training_state.pt` (optimizer, scaler, RNG, step counters) so you
  can pick up an interrupted run exactly where it left off.
- **Signals**: `train.py` installs SIGINT/SIGTERM handlers by default so `Ctrl+C` or a
  scheduler-issued kill checkpoints gracefully instead of losing the run; disable with
  `--disable_signal_checkpoint`.
- **Curriculum training**: if you want staged phases (e.g. base pretraining → longer
  context → RL/thought-distillation) rather than one flat run, use
  `train_curriculum.py`, which calls `train.py` repeatedly and carries the checkpoint
  forward between phases.
- **A minimal reference baseline**: `train_baseline_attention_lm.py` trains a plain
  attention-based LM for A/B comparison against NEED at a matched size/token budget.

### 3.1 Multi-run sanity checks

Run `python train.py --self_test` for a fast structural self-check of the model
before spending time on real data.

---

## 4. Tracking and comparing runs

Three tools cover "what happened in this run" and "how do two runs compare":

### 4.1 `analyze_run.py` - turn logs into a report

Every training run writes a `train_log.jsonl` (path controlled by `--metrics_jsonl`,
default `out_dir/train_log.jsonl`). Turn that - plus any eval JSON or generation
traces - into a compact report:

```bash
python analyze_run.py \
  --run_dir need_out \
  --logs need_out/train_log.jsonl \
  --eval_json runs/preflight/eval.json \
  --trace_jsonl runs/traces/gen_trace.jsonl \
  --out_json runs/need_out_report.json \
  --out_html runs/need_out_report.html
```

| Flag | Purpose |
|---|---|
| `--run_dir` | Run directory to summarize (default `need_out`). |
| `--logs` | One or more `train_log.jsonl`-style files (space-separated). |
| `--eval_json` | One or more eval-script JSON outputs to fold in. |
| `--trace_jsonl` | One or more generation/speculative-decode trace files. |
| `--out_json` / `--out_html` | Where to write the machine-readable / human-readable report. |

Open two of these HTML reports side by side (or diff the JSON) to compare two runs.

### 4.2 `need_run_card.py` - a model/run card

Produces a shareable card (JSON/Markdown/HTML) describing what was trained, on what
data, how to run it, and its eval/audit scores - useful as a permanent record
attached to a checkpoint:

```bash
python need_run_card.py \
  --run_root runs/need_full_pipeline \
  --version_label my-300M-run \
  --config need_out/config.json \
  --state need_out/training_state.pt \
  --audit_json runs/audit/report.json \
  --scorecard_json runs/eval/scorecard.json \
  --runtime_profile runs/need_full_pipeline/runtime_profile.json \
  --out_md runs/need_full_pipeline/MODEL_CARD.md \
  --out_html runs/need_full_pipeline/MODEL_CARD.html \
  --out_json runs/need_full_pipeline/model_card.json
```

### 4.3 Evaluation harnesses

- `need_eval.py` - perplexity/CE + latency (and image-token reconstruction if a
  visual tokenizer is passed) on held-out data:
  ```bash
  python need_eval.py --checkpoint need_out --data data/eval/held_out.jsonl \
    --batches 20 --batch_size 4 --out_json runs/eval/scorecard.json --dashboard
  ```
- `need_behavioral_eval.py` - runs a JSONL of test cases through the model and grades
  actual generations (not just loss):
  ```bash
  python need_behavioral_eval.py --checkpoint need_out --cases_jsonl data/eval/cases.jsonl \
    --out_json runs/eval/behavioral.json
  ```
- `need_dataset_audit.py` - audits corpus files themselves (duplicates, malformed
  rows, completion ratio) before you spend a training run on bad data:
  ```bash
  python need_dataset_audit.py data/corpuses --out_json runs/audit/report.json --strict
  ```

Together: `preflight` (before training) → `train.py`/`train_curriculum.py`
(training) → `analyze_run.py` + `need_eval.py`/`need_behavioral_eval.py` (during/after
training) → `need_run_card.py` (final record) is the standard loop.

---

## 5. Running the model in the terminal

There are two ways to talk to a checkpoint from a terminal:

### 5.1 One-shot generation - `generate.py`

```bash
python generate.py \
  --checkpoint need_out \
  --prefer_best \
  --prompt "Who discovered gravity?" \
  --max_new_tokens 200 \
  --temperature 0.7
```

Or read the prompt from a file and write the answer to a file:

```bash
python generate.py --checkpoint need_out --prompt_file prompt.txt --out_file answer.txt
```

`generate.py --mode image ...` runs image-token generation instead of text (see
`--image_*` flags in §8).

### 5.2 Interactive chat - `terminal.py`

`terminal.py` keeps the checkpoint loaded in memory, maintains a running conversation
transcript, and streams output live:

```bash
python terminal.py --checkpoint need_out --prefer_best --dual_channel_reasoning
```

If you omit `--checkpoint`, it auto-discovers the newest checkpoint-like directory
under `.`, `checkpoints/`, `runs/`, or `outputs/` (looking for `model.pt`, `best.pt`,
`model.safetensors`, `best.safetensors`, or `config.json`).

Once running:

- Type a message and press Enter to chat.
- `/reset` clears the conversation history (fresh context, same loaded model).
- `/exit` (or `exit`, `quit`, `:q`) quits.
- `--display_mode stream_tokens` (default) streams word-by-word; `stream_characters`
  streams char-by-char; `full` prints the whole answer at once.
- `--performance_dashboard` prints a JSON block after each answer with decode
  time, tokens/sec, and any DVSD/speculative-decode/controller metrics.
- `--summary_chunks N` controls how many "public reasoning summary" chunks are
  printed below the answer when `--dual_channel_reasoning` is on.
- `--no_append_history` answers each prompt standalone instead of appending it to a
  running transcript.

`terminal.py` reuses `generate.py`'s full argument parser (see §8), so every
generation/runtime/sidecar flag documented below also works here.

---

## 6. Running the model in the browser

`browser.py` launches a local [Gradio](https://gradio.app) app with three tabs:

- **Text** - chat console with file/image upload and a performance readout.
- **Compare** - send one prompt to two different local checkpoints side by side (the
  loaded checkpoint vs. `--compare_checkpoint`), useful for A/B-ing training runs.
- **Image tokens** - decode NEED's local discrete image tokens into an actual image.

```bash
pip install gradio   # if you haven't already
python browser.py \
  --checkpoint need_out \
  --prefer_best \
  --compare_checkpoint runs/older_checkpoint \
  --host 127.0.0.1 \
  --port 7860
```

Then open `http://127.0.0.1:7860` in a browser. Use `--host 0.0.0.0` to expose it to
other machines on your network.

Key browser-specific flags (in addition to the shared generation/sidecar flags in
§8):

| Flag | Purpose |
|---|---|
| `--checkpoint` | Primary checkpoint (auto-discovered like `terminal.py` if omitted). |
| `--compare_checkpoint` / `--compare_prefer_best` | Second checkpoint shown in the Compare tab. |
| `--host` / `--port` | Bind address for the local web server (default `127.0.0.1:7860`). |
| `--concurrent_requests` | Gradio queue concurrency limit for simultaneous requests. |
| `--runtime_profile` | JSON file (from `need_low_data_adapters.py` or the full pipeline) that pre-fills decode/sidecar/router settings. |
| `--visual_tokenizer` | Directory with a trained VQ image tokenizer, for the Image tokens tab. |

---

## 7. The one-command full pipeline

If you want the whole "build data → train → adapt → export a runtime profile →
launch the browser" flow in one call, use `need_full_training_pipeline.py`. It
orchestrates `build_corpuses.py`, `need_auto_low_data_rl.py`, `train_curriculum.py`,
`need_low_data_rl_start.py`, `need_raw_image_data.py`, and `browser.py` for you.

```bash
python need_full_training_pipeline.py start \
  --out_root runs/need_full_pipeline \
  --params_m 300 \
  --tokens_per_param 20 \
  --total_steps 20000 \
  --batch_size 16 \
  --lr 3e-4 \
  --version_label my-300M-run
```

`action` (first positional argument) controls what the command does:

| Action | Effect |
|---|---|
| `start` | Begin a fresh run in `--out_root`. |
| `resume` | Resume an existing run, skipping stages that already succeeded (or pass `--resume` to `start`). |
| `status` | Print progress/state of an existing run. |
| `audit` | Run dataset/behavioral auditing over the run's artifacts. |
| `card` | Generate the run/model card (equivalent to calling `need_run_card.py` directly). |

Useful flags: `--dry_plan_only` prints every subprocess command it *would* run
without executing anything (great for reviewing before a long job); `--dry_run`
passes dry-run behavior down into supported sub-scripts (e.g. skips real API calls
during synthetic-data generation); `--no_continue_phases` trains each curriculum
phase from scratch instead of continuing from the previous phase's checkpoint.

---

## 8. Flags reference

The scripts collectively expose several hundred flags. This section groups the ones
you'll actually reach for; **every script also supports `--help` for the exhaustive,
always-up-to-date list**:

```bash
python train.py --help
python generate.py --help
python terminal.py --help
python browser.py --help
```

### 8.1 Shared runtime flags (`generate.py`, `terminal.py`, `browser.py`, `train.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint` | - | Checkpoint directory or file to load. |
| `--prefer_best` | off | Load `best.pt`/`best.safetensors` instead of the last saved checkpoint. |
| `--device` | `auto` | `auto` picks CUDA if available, else CPU; or force `cpu`/`cuda`/`cuda:0` etc. |
| `--kernel_backend` | `auto` | `auto`, `torch`, or `triton` - which kernel implementation runs the linear-core/attention ops. |
| `--runtime_profile` | - | JSON (from `need_low_data_adapters.py`) that pre-sets a bundle of decode/sidecar/router flags. |

### 8.2 Generation / sampling (`generate.py`, `terminal.py`, `browser.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--prompt` / `--prompt_file` | - | Text prompt, inline or from a file. |
| `--out_file` | - | Write the generated answer to a file instead of stdout. |
| `--max_new_tokens` | 128 | Max tokens to generate. |
| `--min_new_tokens` | 0 | Force at least this many tokens before an end-of-sequence is allowed. |
| `--temperature` | 0.8 | Sampling temperature; 0 is greedy. |
| `--top_k` | 50 | Top-k truncation. |
| `--top_p` | 0.95 | Nucleus (top-p) truncation. |
| `--typical_p` | 1.0 | Typical-decoding truncation (1.0 disables it). |
| `--repetition_penalty` | 1.0 | Penalize already-used tokens. |
| `--no_repeat_ngram` | 0 | Block repeats of this n-gram size (0 disables). |
| `--system_prompt` | (tool system prompt) | Prepended instruction/persona text. |

### 8.3 Decoding strategy (AR vs. non-sequential/DVSD)

NEED can decode strictly left-to-right ("ar") or fill several "slots" per step and
refine them ("nonseq", i.e. DVSD - Dynamic Variable-Slot Decoding).

| Flag | Default | Meaning |
|---|---|---|
| `--decode_mode` | `nonseq` | `nonseq` = DVSD multi-token decoding; `ar` = standard streaming autoregressive; `auto` picks nonseq only outside the strict streaming core. |
| `--streaming_cache` | on | Use the stateful incremental cache for AR decoding; `--no-streaming_cache` forces the slower full-context compatibility path. |
| `--nonseq_decode` | unset | Force-override `--decode_mode` on/off (`--nonseq_decode` / `--no-nonseq_decode`). |
| `--nonseq_dynamic` | on | Shrink active slot count toward 1 token on hard spans, grow on easy ones. |
| `--nonseq_min_heads` / `--nonseq_max_heads` | 1 / 0 (checkpoint default) | Bounds on how many slots can be filled per step. |
| `--nonseq_refine_steps` | 3 | Refinement passes over the virtual slot canvas before committing. |
| `--nonseq_refine_lock_schedule` | `cosine` | Confidence schedule (`cosine`/`linear`/`quadratic`) used to lock slots in non-left-to-right order. |
| `--nonseq_refine_causal_blend` | 0.55 | How much provisional causal context blends into slot logits during refinement. |
| `--speculative_final_decode` | off | Let a loaded external-LM sidecar draft tokens that NEED then validates/accepts (speeds up final-answer decoding). |

`nonseq`/DVSD is the default decoding path as of this repo revision. To force plain
AR decoding instead, either pass `--decode_mode ar` explicitly or use the
`--no-nonseq_decode` override switch:

```bash
# Explicit AR decoding
python generate.py --checkpoint need_out --decode_mode ar --prompt "..."
python terminal.py --checkpoint need_out --decode_mode ar
python browser.py --checkpoint need_out --decode_mode ar

# Equivalent, using the override switch instead of --decode_mode
python generate.py --checkpoint need_out --no-nonseq_decode --prompt "..."

# AR decoding for the sidecar's own drafting, independent of the main model's decode_mode
python generate.py --checkpoint need_out --need_sidecar_decode_mode ar --prompt "..."
```

### 8.4 Dual-channel reasoning (private "thought" + public answer)

| Flag | Default | Meaning |
|---|---|---|
| `--dual_channel_reasoning` | off | Run a private latent reasoning pass before producing the public answer. |
| `--conditioning_scale` | 0.18 | Strength with which the latent reasoning vectors condition the final answer. |
| `--vector_stride` / `--max_vectors` | 2 / 64 | How densely/how many latent vectors are sampled from the reasoning pass. |
| `--max_thought_tokens` | 160 | Cap on hidden reasoning length. |
| `--hide_thought_summary` | off | Suppress printing the public reasoning summary. |
| `--auto_output_mode_classifier` | on | Let NEED decide none/short/full/multi-step reasoning scaffolding itself. |
| `--reasoning_tree_branches` | 1 | Draft multiple candidate reasoning branches and merge/select among them. |

### 8.5 Sidecars (optional external or smaller-NEED helper models)

A "sidecar" is a second, usually smaller, model that can draft public reasoning/CoT
text or speculative final-answer tokens for NEED to validate - it's optional and off
by default.

| Flag | Default | Meaning |
|---|---|---|
| `--sidecar_type` | `none` | `none`, `external_lm` (e.g. a small HF model), `need` (a smaller NEED checkpoint), or `auto`. |
| `--sidecar_model` | - | HF model id/path for an external-LM sidecar, e.g. `HuggingFaceTB/SmolLM2-135M-Instruct`. |
| `--need_sidecar_checkpoint` | - | Path to a smaller NEED checkpoint used as the sidecar instead. |
| `--sidecar_call_policy` | `off` | `always`, `latent_gated` (only call it on prompts NEED finds difficult), or `off`. |
| `--sidecar_device` / `--sidecar_dtype` | `same` / `bf16` | Where/at what precision the sidecar runs. |
| `--sidecar_cot` | off | Let the sidecar draft the artificial chain-of-thought text. |

### 8.6 Latent tools (hidden calculator/Python, never shown as function calls)

| Flag | Default | Meaning |
|---|---|---|
| `--latent_tools` | on | Master switch for hidden calculator/Python evidence-gathering during reasoning. |
| `--latent_tool_calculator` | on | Enable the arithmetic tool. |
| `--latent_tool_python` | off | Enable sandboxed Python execution for compact numeric/code tasks. |
| `--latent_tool_max_calls` | 3 | Max tool calls per generation. |
| `--latent_tool_timeout_s` | 3.0 | Per-call timeout. |

### 8.7 Terminal-only (`terminal.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--display_mode` | `stream_tokens` | `full`, `stream_tokens`, or `stream_characters`. |
| `--stream_delay_s` | 0.0 | Artificial delay between streamed chunks. |
| `--no_append_history` | off | Treat each prompt independently instead of keeping a running transcript. |
| `--summary_chunks` | 6 | How many reasoning-summary chunks to print per turn. |
| `--performance_dashboard` | off | Print a JSON perf dashboard after each answer. |

### 8.8 Browser-only (`browser.py`)

See the table in [§6](#6-running-the-model-in-the-browser) (`--host`, `--port`,
`--compare_checkpoint`, `--concurrent_requests`, `--visual_tokenizer`, etc.).

### 8.9 Training (`train.py`) - the essentials

| Flag | Default | Meaning |
|---|---|---|
| `--data` / `--packed_data` / `--packed_index` | - | Raw text/JSONL data, or a pre-tokenized packed binary + its index (faster). |
| `--out_dir` | `need_out` | Where checkpoints/logs are written. |
| `--target_params` | - | Auto-size the architecture for this parameter budget (e.g. `300M`). |
| `--architecture` | `dense` | `dense` or `moe`. |
| `--recipe` | - | Named defaults bundle: `fast`, `quality`, `baseline`, `debug`. |
| `--init_from` / `--resume_from` | - | Continue weights only vs. resume full training state (optimizer/RNG/step). |
| `--batch_size` / `--grad_accum_steps` | 8 / 1 | Micro-batch size and gradient accumulation (0/0 lets `--auto_optimize` pick them). |
| `--max_steps` / `--target_tokens` | 1000 / 0 | Stop condition, by step count or by total training tokens. |
| `--lr` / `--lr_schedule` / `--warmup_steps` | 3e-4 / `constant` / 0 | Learning rate and schedule. |
| `--amp` | `bf16` | Mixed precision: `off`, `bf16`, `fp16`. |
| `--compile` | off | `torch.compile` the model (`--compile_mode`, `--compile_backend` tune it further). |
| `--auto_optimize` | off | Turn on sensible runtime defaults (fused AdamW, worker count, optional batch probing). |
| `--save_interval` | 1000 | Steps between full checkpoint saves. |
| `--eval_interval` / `--eval_batches` | 200 / 10 | How often, and how much, validation runs during training. |
| `--log_interval` | 20 | Steps between metric log lines. |
| `--metrics_jsonl` | `out_dir/train_log.jsonl` | Where structured training/eval rows are appended (consumed by `analyze_run.py`). |
| `--nan_recovery` | off | Skip non-finite/spiking batches and lower LR instead of crashing. |
| `--sample_prompts` / `--sample_interval` | - / 0 | Periodically generate from a fixed prompt file during training, to watch qualitative progress. |

`train.py` also exposes dozens of architecture-internal flags (energy-descent
steps/rank, memory slots, planner horizons, MoE routing, image-grid settings,
objective-balancing/curriculum knobs, DVSD router settings, and more) - these are
tuning knobs for NEED's internals rather than day-to-day options. Leave them at
their defaults unless you're doing an ablation; `python train.py --help` lists all
of them with their current defaults.

### 8.10 Data preparation

| Script | Purpose |
|---|---|
| `build_corpuses.py` | Assemble/download a full training corpus plan. |
| `prepare_packed_dataset.py` | Tokenize a corpus into a fast, source-balanced packed binary + index. |
| `quality_filter_corpus.py` | Filter low-quality rows out of a corpus before packing. |
| `need_dataset_audit.py` | Report duplicates/malformed rows/completion ratios (`--strict` to fail CI on bad data). |
| `need_raw_image_data.py` | Prepare/tokenize image data for multimodal training. |

---

## Quick command cheat sheet

```bash
# 1. Plan a model size
python config_for_size.py --params 300M --tokens 6B --recipe fast --print_train_cmd

# 2. Preflight-check before a long run
python preflight.py --target_params 300M --recipe fast --data data/corpuses/knowledge/train.jsonl

# 3. Train
python train.py --data data/corpuses/knowledge/train.jsonl --out_dir need_out \
  --target_params 300M --recipe fast --max_steps 20000

# 4. Track / compare
python analyze_run.py --run_dir need_out --out_html runs/report.html
python need_eval.py --checkpoint need_out --data data/eval/held_out.jsonl --out_json runs/eval.json
python need_run_card.py --run_root need_out --out_md need_out/MODEL_CARD.md

# 5a. Chat in the terminal
python terminal.py --checkpoint need_out --prefer_best

# 5b. Chat in the browser (with a second checkpoint to compare against)
python browser.py --checkpoint need_out --prefer_best --compare_checkpoint runs/older_run

# 5c. Force plain AR decoding instead of the nonseq/DVSD default
python generate.py --checkpoint need_out --decode_mode ar --prompt "..."
python terminal.py --checkpoint need_out --decode_mode ar
python browser.py --checkpoint need_out --decode_mode ar
```
