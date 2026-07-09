# Architecture

**NEED = Nested Energy Equilibrium Descent.** It's an AI language model architecture I developed that aims to do a couple of things:
Firstly is that ever since the transformer has been used in frontier AIs, they've already mostly outgrown the original, simplistic structure of it. It was made for the simple task of translating text after all, but current models are expected to think ahead, reason, and remember in ways that the architecture could fundamentally improve with, so you could achieve similar performance without scaling them increasingly larger. Secondly is that these improvements also let the model more engagingly allocate compute, leading to more efficient generation. This document explains essentially all of the NEED architecture.

\---

## 1\. Design principles

1. **The model compute grows only linearly.** Token mixing is done by input-selective
state-space scans (Mamba/SSD-style), multi-scale causal depthwise convolution, and a
*linear* associative memory (kernel-feature-map, not softmax). The one place something
attention-*shaped* appears is `TemporalPathwayConditioner`, and even that is capped to a
small, fixed number of external anchor vectors (≤ 64) - bounded, not sequence×sequence.
2. **Everything is a residual stream with gated deltas.** A `NEEDBlock` is a chain of five
sub-modules, each of which proposes a delta to the residual stream and is gated before
being added - closer to a "committee of specialists" than a fixed Transformer stack.
3. **Depth and effort are learned per token, not fixed.** An `AdaptiveDepthGate` decides,
per token, whether the block's memory/equilibrium/MoE stages are worth their compute; the
equilibrium module additionally decides *how many refinement steps* a token needs.
4. **Auxiliary objectives are numerous but self-regulating.** \~50 auxiliary losses are
grouped into 5 families, normalized against their own EMA magnitude, curriculum-ramped
in, capped relative to the cross-entropy loss, and automatically quarantined if they turn
pathological - so no single side objective can hijack training.
5. **Linear time, always.** With `strict\_linear\_core=True` (the default), no operation is
allowed to become quadratic in sequence length; token-time cost is `O(T · d)` for fixed
width/caps. Search-like features (branching decoders, backtracking, deep planning) are
clamped to width 1 / depth 0 in this mode and only become "real" in explicit ablation
configs.

\---

## 2\. Tokenization and input embedding

* **Text**: defaults to a Hugging Face subword tokenizer (`HFTokenizer`, currently
`deepseek-ai/DeepSeek-V4-Pro` by default — override with `--tokenizer\_model` or the
`NEED\_TOKENIZER\_MODEL` env var). Subword IDs are shifted by `Special.reserved` so they
sit above NEED's own control-token IDs, the same scheme the exact byte-level fallback
(`ByteTokenizer`, select with `--tokenizer byte`) uses. Every trained checkpoint records
which tokenizer it used in `tokenizer.json`, so inference/eval/distillation scripts
auto-load the matching one via `load\_tokenizer\_for\_dir`.
Also plus a small block of special tokens (`pad, bos, eos, img\_bos, img\_eos, img\_mask, sep, summary markers`)
* **Images**: `DynamicImageTokenizer` is a **train-free VQ tokenizer** - RGB patches are
quantized against a deterministic color-grid codebook (nearest-neighbor in RGB space),
with the grid resolution (8×8 up to 32×32) chosen dynamically from image complexity
(local gradient energy). This is explicitly a placeholder ("train and generate images
without requiring a pretrained VQGAN"); `need\_image.py` separately implements a real
learned `VisualTokenizerVQVAE` that can be swapped in behind the same API.
* **Embedding sum**: `token\_emb + position\_emb + modality\_emb`, plus a row/column
coordinate embedding added only on image-token spans (`image\_coord\_scale`), so visual
tokens carry explicit 2D grid position instead of relying purely on scan order.
* Combined vocabulary = text vocab + image codebook (`vocab\_size = 272 + 512` by default),
with **tied input/output embeddings**.

## 3\. The NEEDBlock: five cooperative stages

`NeedModel` is a stack of `n\_layers` identical `NEEDBlock`s (default 8). Each block runs
**five sub-modules in a fixed order**, and - unlike a standard pre-norm Transformer, where
each sublayer just reads `x` - each stage here can read a compressed summary of what
*earlier stages in the same block already did*, and its contribution is gated by a learned,
per-token "is this worth writing to the residual stream" score.

```
x ── RMSNorm → StructuredDualRetention (SSD scan)  ──gated add──▶
   ── RMSNorm → MultiScaleCausalConv               ──gated add──▶
   ── RMSNorm → HierarchicalMemory (conditioned)    ──gated add──▶   (skippable)
   ── RMSNorm → AdaptiveEquilibrium (energy descent) ──gated add──▶  (skippable)
   ── RMSNorm → SparseMoE (SwiGLU experts)           ──gated add──▶  (skippable)
   → cooperative "finish" mixer → x\_out
```

### 3.1 Structured Dual Retention - the temporal carrier (replaces attention)

`StructuredDualRetention` is a Mamba/SSD-style diagonal state-space scan: an input
projection produces `(u, B-gate, C-gate, Δ, out-gate)`; `u` passes through a short causal
depthwise conv + SiLU; a learned, input-dependent timestep `Δ ∈ \[dt\_min, dt\_max]` sets a
per-channel decay `exp(-Δ)`; the recurrence

```
state\_t = decay\_t \* state\_{t-1} + (1 − decay\_t) · (B-gate\_t · u\_t)
y\_t     = out-gate\_t · (C-gate\_t · state\_t + D · u\_t)
```

is evaluated with a **parallel, chunked affine-recurrence scan** (`affine\_recurrence\_scan\_torch`,
or a fused Triton kernel at inference) - an associative-scan implementation of
`y\_t = decay\_t·y\_{t-1} + update\_t` that composes affine transforms in `log(chunk)` steps
per chunk, so training doesn't need a token-by-token Python loop. A one-token
`stream\_step` variant keeps a rolling `(conv\_buffer, scan\_state)` pair for O(1)-per-token
autoregressive decoding. An older head-based `SelectiveRetention` (linear attention-like,
with per-head learned decay) is kept as an alternate `retention\_impl="selective"` mode but
SSD is the default.

### 3.2 Multi-scale causal convolution

`MultiScaleCausalConv` runs several causal depthwise convolutions at increasing kernel
sizes (`k, k+2, k+4, ...`) and mixes them with a **per-token, per-scale softmax gate**
computed from the *current* token only (causal by construction - no leaking future context
via a sequence-level scale choice, which an earlier version of this module did). By default
only 1 scale is active (`conv\_active\_scales=1`); more scales are an explicit ablation knob.

### 3.3 Hierarchical Memory - explicit read/write memory, conditioned on the temporal stream

`HierarchicalMemory` is deliberately *not* a second competing recurrence. It's explained in
the code as: retention/conv already carry the block's implicit temporal state; memory's job
is to read that state as a **condition**, and write only the **innovation** (`x - condition`)
that the temporal path didn't already explain. Concretely it combines two retrieval paths:

* **Semantic slots**: `memory\_slots` learned key/value pairs (a small, fixed external
memory bank), read by softmax attention over a bounded slot count.
* **Linear associative memory**: a chunked, kernel-feature-map (`elu(x)+1`) associative
recall over the actual sequence - same complexity trick as linear attention
(`state += k⊗v`, query as `state / normalizer`), evaluated causally in fixed-size chunks
so it stays `O(T)` with no `T×T` matrix.

A learned **write gate** decides how much of the innovation gets committed, and (in
inference) a static per-token mask can zero out writes entirely for tokens the depth gate
marks as "don't bother." An `\_overlap\_penalty` auxiliary loss discourages memory's output
from just re-deriving what retention+conv already wrote (**role separation**, see §3.6).

### 3.4 Adaptive Equilibrium - the "Energy Equilibrium Descent" core

This is the module the model is named after. `AdaptiveEquilibrium` wraps a learned convex(-ish)
energy function, `ConvexEnergy`:

```
E(z | context) = ½(z−c)ᵀ P (z−c)  +  ½ ‖ R(z−c) ‖²
```

where `c` (center), `P` (a softplus'd diagonal precision), and a rank-`energy\_rank`
projection `R` are all linear functions of the *context* (the pre-equilibrium hidden
state) - so the quadratic bowl the token falls into is itself token-conditioned, like a
per-token, low-rank-plus-diagonal Gaussian precision. Starting from `z = x`, the module
takes `energy\_steps` gradient-descent steps:

```
z ← z − step\_size · ∇E(z) · run
```

where `run` mixes a **difficulty gate** (`sigmoid(difficulty(context))`, i.e. "how hard is
this token") with a **convergence gate** (`sigmoid(α·(residual − threshold))`, i.e. "has the
gradient norm already dropped near zero"). This gives each token an *adaptive number of
effective refinement steps* - easy/converged tokens get gradient-scaled-to-zero updates,
hard tokens keep descending - without any host-side branching (the step count is fixed at
graph-build time; only the *effective step size* varies continuously). The residual delta
`z − x` is what's added back to the stream; `‖∇E‖²`, the final energy, and the effort/step
fraction all become auxiliary regularizer losses. This is structurally a small, differentiable
**Deep Equilibrium / predictive-coding**-style inner loop, not a fixed feed-forward transform.

`MixtureEnergyRouter` (used at the model level, §4) is the same idea generalized to a
*mixture* of `energy\_routes` energy basins with a causal (prefix-mean-conditioned) softmax
router choosing which basin(s) apply to each token.

### 3.5 Sparse Mixture-of-Experts

`SparseMoE`: a linear router selects `moe\_top\_k` (default 1, i.e. switch-routing) of
`n\_experts` SwiGLU experts per token. Two dispatch paths are implemented: a **true sparse
dispatch** (tokens are gathered per-expert via `index\_select`/`index\_add\_`, so unselected
experts do zero work - the default) and a **static dense dispatch** (`moe\_static\_dispatch`)
that computes all experts and masks the output, trading FLOPs for `torch.compile`-friendly
static shapes. Load-balance, router-entropy, and router-z-loss terms regularize routing.

### 3.6 What holds the five stages together

* **`AdaptiveDepthGate`**: a tiny per-token MLP producing 3 gates (memory / equilibrium /
MoE "worth running"). At inference, gates below `adaptive\_depth\_skip\_threshold` become a
static-shape 0/1 mask (`adaptive\_depth\_static\_masking`) that zeroes that stage's residual
contribution - no dynamic token compaction, so batch shapes stay fixed and
`torch.compile`/CUDA-graph-friendly. A compute-budget penalty keeps the *average* gate
near `compute\_budget` (default 0.72), so the model is trained to actually use adaptive
skipping rather than always leaving every gate open.
* **`CooperativeStepWorkspace`**: each of the 5 stages, after computing its candidate delta,
(a) reads a small attention-pooled summary of *previous stages' published deltas in this
block* (`read\_context`, bounded softmax over ≤5 stage summaries - not over tokens), (b)
computes a **contribution gate** from `(x, delta, context)` deciding whether this delta is
worth adding at all, and (c) publishes a compressed, gated summary of what it actually
contributed for the next stage to read. A final cross-stage mixer reconciles all stage
summaries once more at the end of the block. This turns a NEEDBlock from a blind fixed
chain into a small, five-node cooperative graph, while staying `O(stages²)` not `O(T²)`.
A **redundancy loss** penalizes stage pairs whose deltas point the same direction (with
both gates open), discouraging stages from all learning the same transform.
* **Role separation** (`\_separate\_from`): later stages (memory, equilibrium) have their
proposed delta projected to remove the component that overlaps with the (detached) sum of
earlier stages' deltas - a soft Gram-Schmidt against retention+conv's "temporal" direction
and memory's "written" direction - plus an `\_overlap\_penalty` loss. This is the mechanism
that keeps memory from "rediscovering retention" and equilibrium from just redoing
memory's job, referenced above in §3.3.
* **Per-stage learned residual scale** (`self.scales`, init `layer\_scale\_init=0.20`) plus a
global `residual\_scale` (0.55) keep the 5-way residual sum well-behaved at init.

Every stage above also has a **`stream\_step`** counterpart (rolling conv buffer, scan
state, associative-memory accumulator) so autoregressive generation advances **one token
through O(1) recurrent state** per step rather than recomputing the full context window -
this is the `streaming\_generation` code path used by default text generation.

\---

## 4\. Model-level modules (run once, around the block stack)

After the `n\_layers` blocks (and an optional per-block `Image2DSelectiveScan`, §5), three
more model-wide modules run once over the full sequence:

* **`ExactAssociativeRecall`**: a *learned, bounded* long-range lookup meant to give exact
token/n-gram recall without full attention. For each position it assembles a small,
capped candidate set (`exact\_recall\_max\_candidates`, e.g. \~64–192) from four source types

  * local window, exact-token-match history, exact-bigram-match history, and logarithmically
spaced "landmark" positions - then a learned scorer (not hand-tuned bias constants) ranks
candidates and a soft top-k gathers their value states. Cost is `O(T · C · d)` with `C`
capped, so it stays linear.
* **`StateSpaceDriftStabilizer`**: a small learned anchor pulls the final hidden state back
toward a projection of the *original* input embedding, with losses penalizing (a) hidden
RMS drifting from a target norm, (b) chunk-to-chunk "random walk" drift beyond a target
band, and (c) divergence from the anchor direction - a cheap safeguard against the norm/
state drift that plain SSMs can accumulate over very long contexts.
* **`LatentPlanner`**: produces `planner\_horizons` (default 4) future latent states from the
current hidden state, used for multi-token-prediction losses and for conditioning
non-autoregressive decoding (§6). Two modes, blended by `planner\_block\_space\_mix`:

  * *Sequential*: a GRU-like transition applied horizon-by-horizon.
  * *Block-space*: all horizons are initialized at once (slot + time embedding) and
exchange context via **prefix/suffix cumulative-sum scans** along the horizon axis
instead of full horizon×horizon attention - explicitly written to avoid making the
planner "Transformer-like" internally.
  * A `compound\_step` method lets a virtual-slot decoder (§6) fold a provisional token's
embedding *and* an approximate negative-CE-gradient direction back into the latent
cursor cheaply, so later slots in the same generation block benefit from earlier
decisions without a full extra model pass.

### 4.1 "Cognition" auxiliary heads

A cluster of small heads add extra structure/supervision but do not gate the main forward
path in strict-core mode; they mostly produce diagnostics and auxiliary training signal:

|Module|Role|
|-|-|
|`MixtureEnergyRouter`|causal mixture-of-energy-basins routing (§3.4 generalized)|
|`LatentSlotAttention`|pools `latent\_slots` reusable slots per token via `SlotAttentionBlock`, injects them back as a delta|
|`HierarchicalTimeScales`|fast (token) / medium / slow GRU chain over **causal chunk-prefix means**, mixed back in|
|`RiskSignalFusion`|fuses aux-score risk + latent divergence into calibrated risk/search-need/output-need scalars|
|`LatentDivergenceScore`|mismatch between current state and pooled latent-slot context|
|`OutputModeClassifier`|classifies which of `output\_modes` (none / short / full CoT / multi-CoT / render-only) fits|
|`ObjectProgramHead`|coarse object/layout slots (bounding boxes + presence) for image grounding|
|`AuxScoreHead`|per-token quality/risk/difficulty/contradiction/repetition scores + controller logits|
|`ReasoningCompressor`|pooled summary logits + a faithfulness probe, used to distill an external "sidecar" LM's behavior into the model itself (see `need\_sidecar.py`)|

These exist to support **behavior the generation code can act on** - e.g. deciding how much
scaffolding/CoT to emit, whether to widen search, or when to backtrack - while keeping that
control logic outside the `O(T)` core block stack.

\---

## 5\. Multimodality

Image tokens are interleaved with text tokens in the same 1D sequence (bounded by
`img\_bos/img\_eos`), get row/column coordinate embeddings added at input time (§2), and are
additionally passed through `Image2DSelectiveScan` **after every block**: image spans are
reshaped back into their `grid × grid` layout and swept top-to-bottom / left-to-right with
the same chunked affine-recurrence scan used for temporal retention - giving vision tokens
native 2D adjacency instead of asking a 1D causal scan to infer spatial structure from raster
order alone. (Bidirectional/reverse sweeps are opt-in only, since they're non-causal.)
Auxiliary vision losses (`local\_image\_text\_contrastive`, `region\_word\_alignment`,
`image\_spatial\_smoothness`) train image/text alignment without a cross-modal attention
matrix.

\---

## 6\. Generation

Two decoding modes share the same trained model:

* **DVSD - Dynamic Virtual Slot Decoding** (`nonseq\_decode\_style="slot\_refine"`): This is the default way, it opens a
block of `n\_predict\_heads` (default 4) "virtual" future slots per model pass using
multi-token-prediction heads, fills them **in confidence order rather than strictly
left-to-right**, and iteratively refines low-confidence slots over `nonseq\_refine\_steps`
passes with a temperature/lock schedule (cosine by default) that progressively freezes
high-confidence slots - closer to a small iterative-refinement/BERT-style decoder chained
onto an AR backbone than to speculative decoding with a separate draft model. A router
head (`dvsd\_slot\_router`) is trained to predict how many slots are safely fillable per
block; a router-loss/entropy term and an AR fallback path handle the "actually this token
is genuinely hard" case. The planner's `compound\_step` (§4) lets committed slots update the
shared latent cursor before later slots in the same block are predicted.
* **Streaming autoregressive**: each new token runs one `stream\_step` through
every block's cached recurrent state (conv buffers, SSD scan state, associative-memory
accumulator) - `O(1)` compute per emitted token rather than re-processing the whole
context, up to `streaming\_cache\_max\_tokens`.

\---

## 7\. Training objectives

The loss is cross-entropy plus **five auxiliary families**, each with its own learnable-in-
spirit (config-set) weight `λ\_family ∈ {prediction, latent, risk, vision, regularizer}`
(defaults 0.18 / 0.035 / 0.06 / 0.14 / 0.02). Within a family, \~50 named components (MTP
loss, DVSD slot/consistency/router losses, planner CE, slot-attention diversity/entropy-band,
energy-router balance, risk-signal/output-mode/aux-score losses, image contrastive/alignment/
object losses, and pure regularizers like equilibrium residual, MoE balance, energy-row
orthogonality, memory entropy/diversity, state drift/anchor, cooperative-step redundancy and
gate-budget) are combined with relative weights inside their family.

`AuxiliaryObjectiveBalancer` wraps every term with:

1. **EMA magnitude normalization** - each term is divided by its own running-average scale
before weighting, so losses on wildly different natural scales don't need hand-tuned λs.
2. **Curriculum staging** - each family only ramps in after a configured step
(`objective\_prediction\_start\_step=100`, `...\_latent\_start=400`, `...\_risk\_start=700`,
`...\_vision\_start=900`) over a smoothstep ramp, so the trunk learns basic language
modeling before auxiliary heads start steering it.
3. **Soft budget caps** - each term is `tanh`-soft-clipped so it can use at most
`objective\_aux\_ratio\_cap` (45%) of |CE| in total and `objective\_group\_ratio\_cap` (18%)
per family, preventing any objective family from dominating the trunk gradient.
4. **Automatic quarantine** - terms that are frequently clipped, produce non-finite values,
or (via optional gradient-conflict probes in `train.py`) have persistently negative
cosine similarity with the CE gradient get multiplicatively suppressed
(`objective\_quarantine\_decay=0.50`) and slowly recover over
`objective\_quarantine\_recovery\_steps` (2000) steps.

Net effect: the auxiliary-objective surface is large by design (it's how memory,
equilibrium, planning, risk-awareness, and vision grounding all get trained end-to-end
alongside next-token prediction) but is structurally prevented from overwhelming the
primary language-modeling signal.

\---

## 8\. Scale, complexity, and an example shape

`config\_for\_size.py` derives a full `NeedConfig` from a target parameter count without
hand-picked size buckets - head count from `d\_model` (head-dim ≈ 64), context length from a
power-law in param count, memory/recall/image capacities scaled similarly, and a shape-search
that balances parameter-count error against a smooth depth prior. Example, `--params 400M`:

```
d\_model=768, n\_layers=20, n\_heads=12, d\_ff=3072, block\_size=768,
n\_experts=1 (dense), energy\_rank=192, memory\_slots=64, memory\_rank=128
```

Because every mixing primitive (SSD scan, causal conv, chunked linear associative memory,
bounded exact-recall, block-space planner via prefix/suffix scans, cooperative workspace
over a fixed 5 stages) is linear in `T` or bounded by a config cap, **total token-time cost
is `O(T · d)` for fixed width**, in contrast to a Transformer's `O(T² · d)`. `need\_kernels.py`
supplies optional fused Triton kernels (RMSNorm, SwiGLU, the affine-recurrence scan, the SSD
scan) with a PyTorch-fallback for every one, used automatically when CUDA + Triton are
available and gradients aren't required (training relies on the differentiable chunked-scan
fallback; the fused kernels are primarily an inference optimization).

\---

## 9\. Repository map (how the pieces connect)

|File|Role|
|-|-|
|`need\_core.py`|Everything above: config, tokenizers, all `nn.Module`s, `NeedModel`, generation|
|`need\_kernels.py`|Optional fused Triton kernels with PyTorch fallbacks|
|`need\_image.py`|Real learned VQ-VAE image tokenizer (upgrade path for the placeholder codebook)|
|`need\_sidecar.py` / `sidecar\_lm\_runtime.py`|External-LM or smaller-NEED "sidecar" providing latent conditioning vectors consumed by `TemporalPathwayConditioner`|
|`need\_sidecar\_distill.py` / `need\_thought\_distill.py`|Distilling sidecar/teacher reasoning behavior into `ReasoningCompressor` and the planner|
|`config\_for\_size.py`|Parameter-budget → full config/training-recipe generator|
|`train.py`, `train\_curriculum.py`, `train\_baseline\_attention\_lm.py`|Main training loop, curriculum-phase training, and a baseline attention-based comparison model for ablations|
|`need\_data.py`, `build\_corpuses.py`, `prepare\_packed\_dataset.py`, `quality\_filter\_corpus.py`|Data pipeline|
|`need\_eval.py`, `need\_behavioral\_eval.py`, `generation\_benchmark.py`, `analyze\_run.py`|Evaluation and run analysis|
|`terminal.py`, `generate.py`, `browser.py`|Interactive CLI / generation entry points|
|`need\_run\_card.py`, `preflight.py`|Run-metadata reporting and pre-training sanity checks|

## Immediate Differences

The 3 main noticeable features of NEED are that it reasons in latent space, it processes with a linear time complexity instead of Transformer's quadratic time, and it generates tokens by creating dynamically sized blocks of space and filling each token in one by one, but independent from the order they come in, different from the strict AR that also has the transformer model using the same compute for every token.

