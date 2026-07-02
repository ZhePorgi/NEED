# Core Architecture

## 1. Overview

The Nested Energy Equilibrium Diffusion (NEED) model is an attention-free causal sequence model. Rather than utilizing token-to-token self-attention to process text, NEED relies on a hybrid stack of recurrent sequence processors, iterative energy minimization layers, and content-addressable memory.

### Identity
NEED belongs to the family of linear-recurrent sequence models. However, it extends this paradigm in two fundamental ways:
1. **Continuous Latent Reasoning:** It embeds an active minimization loop and predictive world models within its processing blocks, allowing the model to perform multi-step reasoning, state planning, and constraint validation directly within its continuous vector spaces before producing natural language.
2. **Dynamic Sparsity:** It operates on a fluid spectrum between fully dense representation and conditional sparsity. Through adaptive gating and routing, the model dynamically allocates computational depth and parameter capacity on a token-by-token basis.

---

## 2. Pillars

The core architecture of NEED consists of six integrated systems working in a unified pipeline:

```
       +--------------------------------------------------------+
       |                  INPUT TOKEN SEQUENCE                  |
       +--------------------------------------------------------+
                                    |
                                    v
+----------------------------------------------------------------------+
|                     EMBEDDING & COORDINATE SYSTEM                    |
|  - Token and position embedding projection.                          |
|  - Modality markers and spatial coordinate systems.                  |
+----------------------------------------------------------------------+
                                    |
                                    v
+----------------------------------------------------------------------+
|                     TEMPORAL RECURRENT BACKBONE                      |
|  - Recurrent state-space scans.                                      |
|  - Local causal depthwise convolutions.                              |
+----------------------------------------------------------------------+
                                    |
                                    v
+----------------------------------------------------------------------+
|                     REPRESENTATION REFINEMENT BLOCK                  |
|  - Adaptive compute gates (conditional sparsity control).            |
|  - Multiscale memory (short-term, episodic, semantic, associative).  |
|  - Nested equilibrium loop (energy-based gradient descent).          |
|  - Expert routing (sparse Mixture of Experts).                       |
+----------------------------------------------------------------------+
                                    |
                                    v
+----------------------------------------------------------------------+
|                     LONG-RANGE CORRECTION PATHWAY                    |
|  - Exact associative recall (sparse copy-paste database).            |
|  - State-space drift stabilizer (state anchoring & normalization).   |
+----------------------------------------------------------------------+
                                    |
                                    v
+----------------------------------------------------------------------+
|                     LOGIT HEADS & AUXILIARY TARGETS                  |
|  - Multi-token prediction paths.                                     |
|  - Joint verifiers, world models, task manifolds, and subgoals.      |
+----------------------------------------------------------------------+
```

### 2.1 Causal Recurrence 
At its foundation, NEED replaces the quadratic self-attention mechanism used in conventional attention LMs with a linear recurrent scan. 

In a vanilla self-attention LM, every token must compare itself to every prior token in the sequence. This creates a computational and memory bottleneck that scales with the square of the sequence length. NEED avoids this bottleneck by compressing historical context step-by-step into a fixed-size hidden state.

#### The Recurrent Processing Sequence:
1. **Local Contextualization:** Incoming tokens are first passed through local causal depthwise convolutions. This ensures that immediate, neighboring token structures are merged before any long-range state update occurs.
2. **Selective Gating:** The block projects the current representation to produce multiple dynamic gates: an input gate, an output gate, a state projection gate, and a decay gate.
3. **State-Space Scan:** The model updates its internal recurrent state by multiplying the previous state by the decay gate and adding the gated current input.
4. **Direct Feedthrough:** The modified state is combined with a direct, unaltered version of the input and passed through the output gate.

Because this state update happens sequentially without global token-to-token comparisons, the memory required to run inference remains constant, regardless of whether the model is processing its tenth or ten-thousandth token.

---

### 2.2 Finding The Balanace
While typical sequence models pass representations through layers in a single, feed-forward direction, NEED introduces an active optimization loop within its processing blocks.

#### The Energy Landscape
The model defines a conceptual energy landscape over the hidden state. In this landscape:
* A "center" represents the expected target state based on the current context.
* A "precision" parameter acts as a variable stiffness or gravity, pulling the representation toward the center along specific dimensions.
* A set of low-rank "forces" defines complex directional slopes across the landscape.

#### Iterative Minimization
Instead of accepting the initial feed-forward state, the block runs a localized gradient descent loop. It iteratively calculates the gradient of the energy relative to the state, sliding the representation down the energy slopes toward a stable valley.

---

### 2.3 Conditional Sparsity vs. Dense Capacity
Traditional architectures process every token with the exact same number of parameters and layers, treating a simple comma identically to a complex logical transition. NEED introduces dynamic sparsity to optimize resource allocation and representation quality.

#### Token-Level Gating
At the entrance of the refinement blocks, an adaptive compute gate analyzes the incoming token representation to determine its difficulty. The gate outputs activation coefficients for different processing channels, allowing the model to choose between sparse and dense configurations:

* **Dense Execution:** When processing highly complex, ambiguous, or high-risk tokens, the gates open fully. This activates deep, multi-step energy minimization loops, pulls heavily from associative memories, and engages multiple parameter pathways.
* **Sparse Bypassing:** For predictable, grammatical, or repetitive tokens, the compute gates scale down or bypass these expensive layers entirely. The token is processed via a shallow, fast-forward path, saving computation.

#### Sparse Mixture of Experts (MoE)
Parameter capacity is further managed through a sparse Mixture of Experts system. Instead of activating a single, giant feed-forward network for every token, the block routes the representation through a routing network:
* The router dynamically selects a sparse subset of specialized "expert" networks (e.g., top-two routing) to process the current state.
* A parallel shared expert pathway remains active for all tokens to preserve general language capabilities, while the active specialized experts handle domain-specific reasoning, mathematics, or spatial relationships.

---

### 2.4 Reasoning in Latent Space
As we know standard language models perform reasoning in "token space" where they must write out intermediate steps word-by-word to solve complicated tasks. NEED is designed to perform multi-step, structured reasoning directly within its continuous vector spaces, forming an internal & non-verbal chain of thought.

#### The Continuous Latent Pathway
As NEED processes a prompt, it extracts an ordered sequence of continuous vectors called the latent pathway. This pathway represents the trajectory of the model's cognitive state. Instead of relying solely on written text, the model refines this latent path through several integrated systems:

1. **Goal-Driven Guidance:** A goal hierarchy system generates abstract subgoal vectors based on the initial prompt. These subgoals act as gravitational targets in the latent space, pulling the continuous trajectory toward the long-term thematic objective.
2. **Latent State Trajectory Planning:** The latent world model projects the continuous state forward over multiple imaginary horizons. It simulates alternative paths, calculates their projected "branch value," and detects logical contradictions between the current trajectory, predicted futures, and goals before any text is output.
3. **Equilibrium-Based Thought Settlement:** The nested energy equilibrium loop acts as a physical settlement process for thoughts. As the representation slides down the energy slopes, it resolves conflicting constraints and smooths out logical inconsistencies.
4. **Behavioral Latent Anchoring:** The refined latent pathway can be anchored to prior successful trajectories (retrieved from a behavioral memory store) or aligned with external sidecar models. This latent pathway is then used to condition the final language decoding, ensuring the verbalized output is grounded, consistent, and strictly adheres to constraints without requiring verbose text-based reasoning loops.

---

### 2.5 Associative Recall
Recurrent models are highly efficient but can suffer from lossy compression over long text sequences. Essential details from thousands of tokens prior can be washed out of the fixed-size state. This limitation is addressed with an *Exact Associative Recall* pathway.

#### The Chunked Search Database
The exact recall module operates as a separate, content-addressable query database over the current document:
1. **Key-Query Reduction:** High-dimensional hidden features are projected into compact keys and queries, which are reinforced with exact token and bigram identity markers.
2. **Causal Score Matching:** The sequence is split into manageable chunks. Within each chunk, queries are compared against prior keys. The module applies a strong, deterministic bias to exact token matches and bigram transitions.
3. **Sparse Retrieval:** Rather than performing a soft blend over the entire history, the model retrieves only the top-scoring matches. It extracts their corresponding values and incorporates them back into the main stream via a residual connection.

This design gives the model almost "copy-paste" retrieval capabilities for names, numbers, and repetitive code structures, bypassing the memory limitations of pure recurrence.

---

### 2.6 State-Space Drift Stabilization
In very long contexts, recurrent states can slowly wander away from the original data distribution, leading to representation drift. NEED uses a multi-layered stabilization system to keep its representations grounded:

* **Input Anchoring:** The stabilizer calculates a projection of the raw input representation and anchors the deep, highly processed state back to this baseline. This anchor prevents the state from drifting into abstract regions of the representation space.
* **Drift Rate Penalization:** During training, a specialized loss function penalizes sudden, excessive jumps in the mean representation between adjacent chunks of text.
* **Norm Regularization:** A soft norm constraint penalizes representations that grow too large or collapse toward zero, maintaining a stable volume across long documents.

---

### 2.7 Hierarchical Timescales
To prevent short-term grammatical transitions from overwriting long-term context, NEED separates representation paths into distinct temporal strata:

* **The Fast Stream:** Updates at the individual token level, capturing immediate syntax and local grammar.
* **The Medium Stream:** Updates at the chunk level, tracking sentence-level developments.
* **The Slow Stream:** Updates at the session level, maintaining broad thematic, document, or conversational context.

These streams are merged before the final output layer, allowing the model to make local predictions that are grounded in long-term context.

---

## 3. NEED Core vs. Vanilla self-attention LMs

| Attribute | Self-attention LM | NEED Core |
| :--- | :--- | :--- |
| **Causal Sequence Operator** | Softmax Self-Attention | State-Space Recurrent Scan |
| **Inference State Tracking** | Full Key-Value Cache (grows with sequence length) | Constant-sized recurrent state |
| **Sparsity & Resource Allocation** | Fully Dense | Dynamically Sparse (adaptive layer bypassing and specialized expert routing per token) |
| **Reasoning Interface** | Token-space Verbalization (requires writing out chains-of-thought as text) | Latent-space Reasoning (continuous vector trajectory planning, goal constraints, and world-model simulation) |
| **Optimization Pass** | Single feed-forward calculation | Iterative energy minimization per block |
| **Exact Copy-Paste Recall** | Softmax Self-Attention | Chunked Exact Associative Recall |
| **Long-Context Stability** | Positional decay (e.g., RoPE) | Anchor gating & chunk-wise drift losses |
| **Future Planning** | Autoregressive generation | Multi-step latent world model projections |
| **Auxiliary Objectives** | Next-token cross-entropy only | Multi-objective soft-budgeted losses |

---

### 2.8 Dynamic Virtual-Slot Nonsequential Decoding

NEED's multi-token prediction heads are now used at inference through a direct-commit virtual-slot generator. The normal LM head predicts the next token, while auxiliary MTP heads predict farther future offsets from the same decoder state. During generation, NEED opens a short future canvas, initializes slots from the heads in parallel, then performs a small number of refinement passes. The most confident slots are locked first, so internal filling does not have to proceed strictly left to right. The resulting canvas is committed directly; there is no verified-acceptance pass and no longest-prefix rejection in the canonical decoder.

The mechanism is intentionally a spectrum rather than a separate mode. For easy, low-entropy spans, the decoder can use multiple slots and commit several tokens per generation step. For difficult spans, high verifier risk, high contradiction, high repetition risk, or high predictive entropy lowers the active head count before decoding. At the hardest moments the range collapses to one slot, which is equivalent to autoregressive sampling for that step. This keeps careful generation available while exposing parallel future-token structure when the model is confident.

This is intended as an AR/diffusion midpoint: it borrows the idea of a mutable future canvas from diffusion-style generation, but it uses NEED's existing MTP heads and causal state rather than training a masked text diffusion objective.

### 2.9 Universal Single-Sidecar Runtime

NEED now treats sidecars as a single active backend rather than as an external-LM-specific feature. The runtime resolves one backend only: `none`, `external_lm`, or `need`. In `auto` mode, a configured smaller NEED sidecar takes priority over an external LM sidecar; otherwise the external LM sidecar is used if present. This keeps inference predictable and avoids spending compute on two advisory models at once.

A NEED sidecar is a smaller `NeedModel` checkpoint wrapped by `NeedSidecarRuntime`. It implements the same narrow interface needed by generation: compact public summary drafting, latent-guidance extraction, and projected latent alignment. The sidecar can have a smaller hidden dimension than the main model. A trained `NeedLatentProjection` maps the sidecar's latent pathway into the main model's latent space, where it can be appended as behavioral conditioning vectors.

The main model remains authoritative. A smaller NEED sidecar is used as an architecture-native prior: it can summarize the task, expose a cheaper latent trajectory, and provide projected anchors during dual-channel reasoning. By default it does not replace the main model's DVSD final decoder and does not activate the old external-LM-style speculative final-answer validator.

### 2.10 NEED-to-NEED Sidecar Distillation

`need_sidecar_distill.py` trains a smaller NEED sidecar to approximate a larger NEED teacher. The training loss blends several teacher signals:

- ordinary CE on corpus tokens, so the sidecar remains a usable language model;
- temperature-scaled KL from the teacher's next-token logits;
- KL from teacher and student MTP heads, so the sidecar learns the teacher's multi-token lookahead behavior;
- projected hidden-state alignment, so the student latent trajectory maps into the teacher's latent manifold;
- future-state alignment from planner/world tensors when available.

The distillation output is still a normal NEED checkpoint. The only additional artifact is `need_sidecar_projection.pt`, which maps the smaller sidecar's latent dimension into the larger teacher/main model dimension. This same artifact is consumed by `generate.py` and by low-data RL runtime-profile export.

## Browser and terminal runtime parity

`browser.py` and `terminal.py` now share the same post-conversation generation assumptions as `generate.py`:

1. **DVSD is available outside the CLI.** The browser can run the dynamic virtual-slot decoder directly. The UI exposes the decoder mode, dynamic-slot toggle, min/max active slots, refinement passes, causal blend, confidence floor, temperature decay, lock schedule, and locked-slot resampling flag. Terminal generation uses the same `decode_mode` and `nonseq_*` arguments inherited from `generate.py`.

2. **Exactly one active sidecar is loaded.** Runtime sidecar selection is centralized in `need_sidecar.py`. The active backend is one of `none`, `external_lm`, or `need`; the browser and terminal no longer treat the external LM as the only possible sidecar.

3. **Smaller NEED sidecar support is native.** A smaller NEED checkpoint can generate compact public reasoning summaries using AR or DVSD and can project its latent pathway into the larger NEED model's latent space. These anchors are advisory latent guidance, not a second final-answer verifier.

4. **External-LM-only speculative final decoding remains gated.** Verified final-answer speculative decoding is only attempted if the active sidecar reports `supports_speculative_final_decode=True`, which is true for external LM sidecars and false for NEED sidecars. This keeps the DVSD direct-commit path separate from the older verified speculative path.

### 2.11 DVSD-native training and learned slot routing

DVSD is now represented in training. The model includes a small `dvsd_slot_router` head on top of the decoder hidden state. The router predicts a distribution over slot budgets `1..n_predict_heads`; index 0 means the next generation cycle should behave like AR, while higher indices allow larger direct-commit virtual canvases.

The training target is the longest low-loss future prefix available under teacher forcing. If the first slot is high-loss, the target collapses to one slot. If several future slots are low-loss, the target expands. This trains the same behavior desired at inference: hard spans shrink, easy spans widen.

Three auxiliary losses support this:

1. `dvsd_slot_ce` applies CE over the future slots that the teacher-forced prefix says should be safe to commit.
2. `dvsd_consistency` aligns MTP future-slot logits with teacher-forced AR logits at the matching future positions.
3. `dvsd_router` trains the slot-budget router, with a small entropy term to avoid premature collapse.

At inference, the learned router does not fully replace the heuristic controller. It blends with entropy/risk/contradiction/repetition difficulty signals using `dvsd_router_inference_mix`, and the hard gate can still force the minimum slot count. This keeps the AR fallback endpoint intact.

### 2.12 DVSD calibration harness

`dvsd_calibration.py` is a dedicated calibration script for measuring whether DVSD is actually buying useful compute efficiency. It runs AR and DVSD on the same prompt set and logs:

- wall-clock tokens/sec for AR and DVSD;
- estimated expensive trunk-pass speedup;
- committed tokens per expensive pass;
- one-slot fallback rate;
- learned-router use and active-slot metrics;
- repetition and replacement-character artifact proxies.

This is intended to tune `nonseq_max_heads`, `nonseq_refine_steps`, `dvsd_router_inference_mix`, and `dvsd_router_min_confidence` before a larger benchmark run.

### 2.13 Latent-gated single sidecar

The single-sidecar rule is preserved. The new behavior is call gating, not multi-sidecar execution. When `sidecar_call_policy=latent_gated`, the main NEED latent pathway is computed first. If the selected gate metric, usually `latent_difficulty`, is below threshold, sidecar summary/latent calls are skipped. If it is above threshold, the one active sidecar is used normally.

This gives a compute-aware policy: easy prompts use main NEED plus DVSD only, while difficult prompts can invoke the smaller NEED sidecar for public summaries and projected latent anchors.

### 2.14 CKA sidecar representation distillation

The NEED-to-NEED sidecar distillation script now includes projected linear CKA matching. The projection maps the smaller student's hidden dimension to the teacher dimension, then CKA matches the geometry of token representations. This complements MSE/cosine hidden matching and is better suited to aggressive compression where a small sidecar cannot copy every teacher coordinate exactly.
