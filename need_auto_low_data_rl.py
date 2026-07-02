#!/usr/bin/env python3
"""OpenAI-compatible low-data RL data generator and adapter-prep pipeline.

Builds a small diverse RL/SFT/preference corpus by asking a selected LLM
(for example an OAI model name such as ``gpt-5.4``) to first plan
how many examples to make, then to generate compact JSON datapoints in modular
batches.

The default mode is a dry run: it writes a plan and reusable batch prompts but
makes no API calls.  Add ``--activate`` to call the configured model. 

Typical use:

  python need_auto_low_data_rl.py \
    --activate \
    --model gpt-5.4-nano \
    --api_key_env OPENAI_API_KEY \
    --out_dir data/corpuses \
    --target_profile balanced \
    --wikipedia_source popular,random,vital \
    --total_examples 3000 \
    --batch_size 50

Outputs are appended to the same corpus layout used by build_corpuses.py:

  knowledge/train.synthetic.jsonl
  rl/sft.synthetic.jsonl
  rl/preferences.synthetic.jsonl
  rl/rlvr.synthetic.jsonl

Numeric-evaluation, risk-scoring, weighted-decision, image-generation behavior,
structured-output, tool-routing, behavioral-memory, self-correction, and
scenario-transcription rows are emitted with rule-based aux_score metadata so
downstream tuning can reward calibrated behavior rather than free-form confident
guesses.

It can also ingest transcript JSONL events produced by any external transcription
source and convert them into natural overheard-scenario
control rows without calling the generation model.

It can also emit a compact control-interactions file for need_low_data_adapters:

  data/corpuses/rl/low_data_control_interactions.synthetic.jsonl
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

TARGET_FILES: Dict[str, str] = {
    "knowledge": "knowledge/train.synthetic.jsonl",
    "instruction_following": "rl/sft.synthetic.jsonl",
    "reasoning_rlvr": "rl/rlvr.synthetic.jsonl",
    "numeric_evaluation": "rl/rlvr.synthetic.jsonl",
    "risk_scoring": "rl/rlvr.synthetic.jsonl",
    "weighted_decision": "rl/rlvr.synthetic.jsonl",
    "overheard_transcription_acceptance": "rl/sft.synthetic.jsonl",
    "overheard_transcription_preferences": "rl/preferences.synthetic.jsonl",
    "overheard_transcription_rlvr": "rl/rlvr.synthetic.jsonl",
    "sidecar_latent_alignment": "rl/sidecar_latent_alignment.synthetic.jsonl",
    "image_prompt_fidelity": "rl/preferences.synthetic.jsonl",
    "image_generation_preferences": "rl/preferences.synthetic.jsonl",
    "image_edit_instruction_following": "rl/sft.synthetic.jsonl",
    "image_safety_boundaries": "rl/preferences.synthetic.jsonl",
    "image_composition_scoring": "rl/rlvr.synthetic.jsonl",
    "tool_routing": "rl/preferences.synthetic.jsonl",
    "latent_tool_calculator": "rl/rlvr.synthetic.jsonl",
    "latent_tool_python": "rl/rlvr.synthetic.jsonl",
    "latent_tool_preferences": "rl/preferences.synthetic.jsonl",
    "code_execution_behavior": "rl/sft.synthetic.jsonl",
    "structured_json": "rl/rlvr.synthetic.jsonl",
    "behavioral_memory_policy": "rl/preferences.synthetic.jsonl",
    "self_correction_aux_score": "rl/rlvr.synthetic.jsonl",
    "helpfulness_preferences": "rl/preferences.synthetic.jsonl",
    "harmlessness_safety": "rl/preferences.synthetic.jsonl",
    "honesty_uncertainty": "rl/sft.synthetic.jsonl",
    "friendly_concise_style": "rl/sft.synthetic.jsonl",
}

# Whole-model low-data default: about 3k compact examples with explicit numeric
# scoring/evaluation families included.  Values are ratios of the target 3000.
DEFAULT_MIX: Dict[str, float] = {
    "instruction_following": 250 / 3000,
    "helpfulness_preferences": 220 / 3000,
    "reasoning_rlvr": 200 / 3000,
    "numeric_evaluation": 220 / 3000,
    "risk_scoring": 170 / 3000,
    "weighted_decision": 140 / 3000,
    "overheard_transcription_acceptance": 170 / 3000,
    "overheard_transcription_preferences": 90 / 3000,
    "overheard_transcription_rlvr": 90 / 3000,
    "sidecar_latent_alignment": 110 / 3000,
    "image_prompt_fidelity": 120 / 3000,
    "image_generation_preferences": 100 / 3000,
    "image_edit_instruction_following": 80 / 3000,
    "image_safety_boundaries": 50 / 3000,
    "image_composition_scoring": 50 / 3000,
    "tool_routing": 60 / 3000,
    "latent_tool_calculator": 140 / 3000,
    "latent_tool_python": 160 / 3000,
    "latent_tool_preferences": 100 / 3000,
    "code_execution_behavior": 170 / 3000,
    "structured_json": 70 / 3000,
    "behavioral_memory_policy": 50 / 3000,
    "self_correction_aux_score": 50 / 3000,
    "honesty_uncertainty": 60 / 3000,
    "harmlessness_safety": 40 / 3000,
    "friendly_concise_style": 40 / 3000,
}

PROFILE_HINTS: Dict[str, Dict[str, Any]] = {
    "tiny": {
        "total_examples": 120,
        "batch_size": 12,
        "mix": {
            "instruction_following": 0.25,
            "reasoning_rlvr": 0.22,
            "helpfulness_preferences": 0.22,
            "honesty_uncertainty": 0.13,
            "harmlessness_safety": 0.10,
            "friendly_concise_style": 0.08,
        },
    },
    "balanced": {"total_examples": 320, "batch_size": 24, "mix": {
        "knowledge": 0.06,
        "instruction_following": 0.17,
        "reasoning_rlvr": 0.14,
        "numeric_evaluation": 0.10,
        "risk_scoring": 0.07,
        "weighted_decision": 0.06,
        "overheard_transcription_acceptance": 0.12,
        "overheard_transcription_preferences": 0.07,
        "overheard_transcription_rlvr": 0.06,
        "sidecar_latent_alignment": 0.035,
        "image_prompt_fidelity": 0.04,
        "image_generation_preferences": 0.035,
        "image_edit_instruction_following": 0.025,
        "image_safety_boundaries": 0.02,
        "image_composition_scoring": 0.02,
        "tool_routing": 0.020,
        "latent_tool_calculator": 0.045,
        "latent_tool_python": 0.045,
        "latent_tool_preferences": 0.025,
        "code_execution_behavior": 0.035,
        "structured_json": 0.025,
        "behavioral_memory_policy": 0.02,
        "self_correction_aux_score": 0.02,
        "helpfulness_preferences": 0.055,
        "harmlessness_safety": 0.02,
        "honesty_uncertainty": 0.015,
        "friendly_concise_style": 0.015,
    }},
    "whole_model": {"total_examples": 3000, "batch_size": 50, "mix": DEFAULT_MIX},
    "reasoning": {
        "total_examples": 360,
        "batch_size": 24,
        "mix": {
            "reasoning_rlvr": 0.38,
            "instruction_following": 0.18,
            "helpfulness_preferences": 0.16,
            "knowledge": 0.12,
            "honesty_uncertainty": 0.08,
            "harmlessness_safety": 0.05,
            "friendly_concise_style": 0.03,
        },
    },
    "style": {
        "total_examples": 260,
        "batch_size": 20,
        "mix": {
            "friendly_concise_style": 0.25,
            "instruction_following": 0.24,
            "helpfulness_preferences": 0.22,
            "honesty_uncertainty": 0.13,
            "harmlessness_safety": 0.08,
            "reasoning_rlvr": 0.08,
        },
    },
    "safety": {
        "total_examples": 300,
        "batch_size": 20,
        "mix": {
            "harmlessness_safety": 0.28,
            "honesty_uncertainty": 0.20,
            "helpfulness_preferences": 0.20,
            "instruction_following": 0.16,
            "reasoning_rlvr": 0.10,
            "friendly_concise_style": 0.06,
        },
    },
}

FALLBACK_TOPICS: List[str] = [
    "renewable energy storage",
    "photosynthesis",
    "Roman concrete",
    "supply chain resilience",
    "public-key cryptography",
    "urban heat islands",
    "introductory statistics",
    "water purification",
    "ancient trade routes",
    "computer networking",
    "basic probability",
    "food preservation",
    "plate tectonics",
    "calendar systems",
    "laboratory safety",
    "financial budgeting",
    "earthquake preparedness",
    "biodiversity",
    "battery recycling",
    "medical triage basics",
    "semantic search",
    "software testing",
    "decision making under uncertainty",
    "manufacturing quality control",
]

VITAL_ARTICLE_TOPICS: List[str] = [
    "Mathematics",
    "Physics",
    "Chemistry",
    "Biology",
    "History",
    "Geography",
    "Economics",
    "Computer science",
    "Medicine",
    "Philosophy",
    "Engineering",
    "Statistics",
    "Climate change",
    "World War II",
    "Democracy",
    "Agriculture",
    "Electricity",
    "Internet",
    "Evolution",
    "Human rights",
    "Artificial intelligence",
    "Supply chain management",
    "Lean manufacturing",
    "Project management",
]

GENERATION_RULES = """Rules for every datapoint:
- Return only valid JSON. No markdown fences, no comments, no prose outside JSON.
- Keep each datapoint short. Prompts should normally be under 120 words and answers under 220 words.
- Use fresh concrete topics from the supplied article/topic seeds, but do not copy Wikipedia text.
- Avoid private people, personal data, fabricated citations, obscure trivia, and copyrighted passages.
- Do not include hidden chain-of-thought. Use concise explanations or final-answer rationales only.
- Keep difficulty low-data friendly: clear, teachable, not over-specialized, not too long.
- For image-generation behavior categories, do not generate image files or claim an image was created; only create prompt/reward/behavioral training rows.
- For latent tool categories, final answers must not expose tool-call JSON; tool calls/results are training metadata for hidden runtime use.
- Vary surface form, domain, and user intent inside the batch.
"""

IMAGE_BEHAVIOR_CATEGORIES = {
    "image_prompt_fidelity",
    "image_generation_preferences",
    "image_edit_instruction_following",
    "image_safety_boundaries",
    "image_composition_scoring",
}

BEHAVIOR_GAP_CATEGORIES = {
    "tool_routing",
    "latent_tool_calculator",
    "latent_tool_python",
    "latent_tool_preferences",
    "code_execution_behavior",
    "structured_json",
    "behavioral_memory_policy",
    "self_correction_aux_score",
}

PREFERENCE_STYLE_CATEGORIES = {
    "image_prompt_fidelity",
    "image_generation_preferences",
    "image_safety_boundaries",
    "tool_routing",
    "latent_tool_preferences",
    "behavioral_memory_policy",
}

RLVR_STYLE_CATEGORIES = {
    "image_composition_scoring",
    "structured_json",
    "latent_tool_calculator",
    "latent_tool_python",
    "self_correction_aux_score",
}


@dataclass(frozen=True)
class TopicSeed:
    title: str
    extract: str = ""
    url: str = ""
    source: str = "fallback"

    def compact(self, max_extract_chars: int = 360) -> Dict[str, str]:
        extract = _clean_spaces(self.extract)[:max_extract_chars]
        return {"title": self.title, "extract": extract, "url": self.url, "source": self.source}


@dataclass(frozen=True)
class PromptModule:
    category: str
    target_file: str
    training_kind: str
    schema: Dict[str, Any]
    rubric: str
    reward_hint: str


PROMPT_MODULES: Dict[str, PromptModule] = {
    "knowledge": PromptModule(
        category="knowledge",
        target_file=TARGET_FILES["knowledge"],
        training_kind="short knowledge grounding",
        schema={"text": "short educational paragraph", "topic": "seed topic", "freshness_note": "why this angle feels current/useful"},
        rubric="Produce a single compact educational paragraph that could improve general factual grounding.",
        reward_hint="Reward clarity, non-copying, and broad usefulness. Penalize vague or overlong text.",
    ),
    "instruction_following": PromptModule(
        category="instruction_following",
        target_file=TARGET_FILES["instruction_following"],
        training_kind="SFT conversation",
        schema={"messages": [{"role": "user", "content": "compact request"}, {"role": "assistant", "content": "compact helpful answer"}]},
        rubric="Make the assistant directly follow a simple realistic instruction without over-explaining.",
        reward_hint="Reward exact instruction following, concise structure, and useful next steps.",
    ),
    "reasoning_rlvr": PromptModule(
        category="reasoning_rlvr",
        target_file=TARGET_FILES["reasoning_rlvr"],
        training_kind="RLVR with machine-checkable aux_score",
        schema={"prompt": "short problem", "answer": "final answer", "aux_score": {"type": "exact|numeric|contains", "value": "expected value"}},
        rubric="Create a short problem whose answer can be checked without long reasoning traces.",
        reward_hint="Reward correct final answers and aux_score quality. Penalize ambiguity or multi-page reasoning.",
    ),
    "numeric_evaluation": PromptModule(
        category="numeric_evaluation",
        target_file=TARGET_FILES["numeric_evaluation"],
        training_kind="numeric RLVR evaluation",
        schema={
            "task": "short evaluation request",
            "input": {"scenario": "compact scenario", "scale": {"score": "0 to 10", "confidence": "0 to 1"}},
            "target": {"score": 0.0, "confidence": 0.0, "main_factors": [{"factor": "factor name", "weight": 0.0}]},
            "assistant_response": {"score": 0.0, "confidence": 0.0, "explanation": "short calibrated explanation"},
            "aux_score": {"kind": "numeric_rule_based", "checks": ["score_in_range", "confidence_in_range", "weights_sum_reasonably"]},
        },
        rubric="Create compact examples where the assistant assigns a calibrated numeric score and explains the main weighted factors.",
        reward_hint="Reward bounded numbers, clear factor weights, and calibrated confidence. Penalize arbitrary or over-precise scores.",
    ),
    "risk_scoring": PromptModule(
        category="risk_scoring",
        target_file=TARGET_FILES["risk_scoring"],
        training_kind="risk scoring RLVR",
        schema={
            "task": "short risk assessment request",
            "input": {"scenario": "compact operational/safety/business scenario", "scale": {"risk_score": "0 to 10", "confidence": "0 to 1"}},
            "target": {"risk_score": 0.0, "risk_band": "low|medium|high", "confidence": 0.0, "main_factors": [{"factor": "factor name", "weight": 0.0}]},
            "assistant_response": {"risk_score": 0.0, "risk_band": "low|medium|high", "confidence": 0.0, "explanation": "short calibrated explanation"},
            "aux_score": {"kind": "risk_rule_based", "checks": ["risk_score_in_range", "band_matches_score", "confidence_in_range"]},
        },
        rubric="Create concise risk examples with explicit score, band, confidence, factor weights, and a practical mitigation.",
        reward_hint="Reward calibrated risk numbers and sensible mitigations. Penalize alarmism, false certainty, or missing top factors.",
    ),
    "weighted_decision": PromptModule(
        category="weighted_decision",
        target_file=TARGET_FILES["weighted_decision"],
        training_kind="weighted decision RLVR",
        schema={
            "task": "short decision request",
            "input": {"options": [{"name": "Option A"}], "weights": {"criterion": 0.0}},
            "target": {"best_option": "Option A", "scores": {"Option A": 0}, "reason": "short reason"},
            "assistant_response": {"best_option": "Option A", "scores": {"Option A": 0}, "explanation": "short weighted explanation"},
            "aux_score": {"kind": "weighted_rule_based", "checks": ["weights_sum_to_one", "best_option_has_highest_score", "scores_in_range"]},
        },
        rubric="Create compact weighted-choice examples with numeric criteria, normalized weights, scores, and a defensible best option.",
        reward_hint="Reward internally consistent weights and scores. Penalize arithmetic inconsistency or unexplained tradeoffs.",
    ),
    "overheard_transcription_acceptance": PromptModule(
        category="overheard_transcription_acceptance",
        target_file=TARGET_FILES["overheard_transcription_acceptance"],
        training_kind="SFT overheard-scenario transcription acceptance",
        schema={
            "overheard_transcription": {
                "scene": "short ordinary overheard scenario",
                "transcript": "compact transcript text supplied to the assistant",
                "segments": [{"speaker": "A", "start_s": 0.0, "end_s": 4.2, "text": "segment text", "confidence": 0.0}],
                "language": "en",
                "confidence": 0.0,
                "source": "provided_transcript",
            },
            "messages": [
                {"role": "system", "content": "Use the provided transcript from the overheard scenario as text evidence; do not claim you personally heard audio."},
                {"role": "user", "content": "compact request that includes or references the transcript"},
                {"role": "assistant", "content": "answer grounded in the transcript"},
            ],
            "acceptance_policy": "accept|accept_with_caveat|ask_clarifying",
        },
        rubric="Teach NEED to accept a provided overheard-scenario transcript as text evidence, use it directly, and caveat only low-confidence or ambiguous portions.",
        reward_hint="Reward grounding in the provided transcript, not asking for hidden audio unnecessarily, and clear caveats for low-confidence spans. Penalize ignoring the transcript or inventing unprovided content.",
    ),
    "overheard_transcription_preferences": PromptModule(
        category="overheard_transcription_preferences",
        target_file=TARGET_FILES["overheard_transcription_preferences"],
        training_kind="preference pair for overheard-scenario transcription use",
        schema={
            "overheard_transcription": {"scene": "short ordinary overheard scenario", "transcript": "compact transcript", "confidence": 0.0, "segments": []},
            "prompt": "user request grounded in the transcript",
            "chosen": "better answer that uses the provided transcript correctly",
            "rejected": "worse answer that ignores, overclaims, or hallucinates beyond the transcript",
            "preference_reason": "short reason",
        },
        rubric="Make chosen clearly better at accepting and using provided overheard-scenario transcript text than rejected.",
        reward_hint="Reward faithful transcript use and correct uncertainty. Penalize needless refusal, first-hand hearing claims, or fabricated details.",
    ),
    "overheard_transcription_rlvr": PromptModule(
        category="overheard_transcription_rlvr",
        target_file=TARGET_FILES["overheard_transcription_rlvr"],
        training_kind="RLVR transcript-grounding check",
        schema={
            "prompt": "short task using the provided overheard-scenario transcript",
            "overheard_transcription": {"scene": "short ordinary overheard scenario", "transcript": "compact transcript", "confidence": 0.0, "segments": []},
            "answer": "short grounded answer",
            "aux_score": {
                "kind": "provided_transcription_rule_based",
                "checks": ["uses_provided_transcript", "does_not_claim_firsthand_hearing", "does_not_invent_unprovided_details", "respects_low_confidence_caveat"],
                "must_include": ["phrase or fact from transcript"],
                "must_not_include": ["unsupported claim"],
            },
        },
        rubric="Create compact transcript-grounding tasks with aux_score fields that can check faithful use of ordinary provided transcript text.",
        reward_hint="Reward answers that use the provided transcript as evidence. Penalize transcript rejection when confidence is adequate and hallucination when confidence is low.",
    ),
    "sidecar_latent_alignment": PromptModule(
        category="sidecar_latent_alignment",
        target_file=TARGET_FILES["sidecar_latent_alignment"],
        training_kind="sidecar latent-alignment seed row",
        schema={
            "input_text": "compact task, transcript scenario, or decision problem",
            "target_summary": "compact public latent-aligned summary, not hidden chain-of-thought",
            "need_metrics": {"quality": 0.0, "risk": 0.0, "contradiction": 0.0},
            "training_objectives": ["summary_lm", "latent_projection", "contrastive_alignment"],
        },
        rubric="Create short sidecar-alignment seed rows that can be passed to need_thought_distill.py build_alignment_dataset or train_alignment after NEED latent targets are attached.",
        reward_hint="Reward concise public reasoning summaries and behavior-level latent guidance. Penalize hidden chain-of-thought, unsupported facts, or implementation leaks.",
    ),
    "image_prompt_fidelity": PromptModule(
        category="image_prompt_fidelity",
        target_file=TARGET_FILES["image_prompt_fidelity"],
        training_kind="image-generation prompt fidelity preference pair",
        schema={
            "prompt": "user asks for an image to be generated, revised, or prepared",
            "chosen": "better image-generation instruction/prompt that preserves the user intent",
            "rejected": "worse prompt that overfills, changes style, adds unsupported objects/text/logos, or ignores constraints",
            "reward_axes": {"prompt_fidelity": 0.35, "visual_specificity": 0.25, "non_overreach": 0.20, "safety": 0.10, "composition": 0.10},
            "preference_reason": "short reason",
        },
        rubric="Teach image-generation behavior without creating images: preserve user intent, specify useful visual details, and avoid scene drift or unsupported additions.",
        reward_hint="Reward faithful, visually concrete prompt rewriting. Penalize overfilling, style drift, logos/text not requested, and claiming an image was generated.",
    ),
    "image_generation_preferences": PromptModule(
        category="image_generation_preferences",
        target_file=TARGET_FILES["image_generation_preferences"],
        training_kind="preference pair for image-generation assistant behavior",
        schema={
            "prompt": "user image-generation request",
            "chosen": "better assistant behavior or revised prompt",
            "rejected": "worse behavior such as asking unnecessary questions, changing the request, or adding too many details",
            "reward_axes": {"instruction_following": 0.30, "aesthetic_clarity": 0.20, "constraint_respect": 0.20, "safety": 0.15, "brevity": 0.15},
        },
        rubric="Create short examples that train the model to convert user image requests into clean generation behavior while respecting explicit constraints.",
        reward_hint="Reward concise, usable, constraint-faithful image-generation behavior. Penalize unnecessary caveats, overspecification, and invented content.",
    ),
    "image_edit_instruction_following": PromptModule(
        category="image_edit_instruction_following",
        target_file=TARGET_FILES["image_edit_instruction_following"],
        training_kind="SFT image-edit instruction following behavior",
        schema={
            "messages": [
                {"role": "user", "content": "compact image-edit request stated as if an image is available"},
                {"role": "assistant", "content": "concise edit instruction plan that preserves unchanged parts and applies only requested changes"},
            ],
            "edit_policy": {"preserve_subject": True, "avoid_unrequested_changes": True, "ask_only_if_target_missing": True},
        },
        rubric="Teach edit behavior: preserve the existing image, apply only requested changes, and ask for the image only when the target image is absent.",
        reward_hint="Reward precise edit instructions and preservation. Penalize unrelated style changes or pretending to see an image when none is provided.",
    ),
    "image_safety_boundaries": PromptModule(
        category="image_safety_boundaries",
        target_file=TARGET_FILES["image_safety_boundaries"],
        training_kind="image-generation safety preference pair",
        schema={
            "prompt": "image request with a mild safety, privacy, identity, or consent boundary",
            "chosen": "safe useful response or safer image prompt",
            "rejected": "unsafe, invasive, or over-refusing response",
            "reward_axes": {"safety": 0.35, "helpfulness": 0.25, "boundary_precision": 0.25, "tone": 0.15},
        },
        rubric="Cover image-specific safety boundaries without generating images: identity, minors, private people, sensitive attributes, gore, and copyrighted style overreach.",
        reward_hint="Reward precise boundaries with safe alternatives. Penalize both unsafe compliance and blanket refusal when a safe version is possible.",
    ),
    "image_composition_scoring": PromptModule(
        category="image_composition_scoring",
        target_file=TARGET_FILES["image_composition_scoring"],
        training_kind="RLVR image-composition scoring from text-only candidates",
        schema={
            "prompt": "user image request",
            "candidate_prompt": "candidate image prompt or generation plan",
            "answer": "short score explanation",
            "target": {"score": 0.0, "confidence": 0.0, "main_factors": [{"factor": "factor name", "weight": 0.0}]},
            "aux_score": {"kind": "image_behavior_rule_based", "checks": ["score_in_range", "confidence_in_range", "mentions_fidelity", "no_image_claim"]},
        },
        rubric="Create machine-checkable text-only image-behavior scoring rows. Do not include actual image files or generated-image claims.",
        reward_hint="Reward calibrated scores for prompt fidelity, composition, safety, and visual specificity. Penalize arbitrary scores and image-generation claims.",
    ),
    "tool_routing": PromptModule(
        category="tool_routing",
        target_file=TARGET_FILES["tool_routing"],
        training_kind="tool-routing preference pair",
        schema={"prompt": "user request", "chosen": "better answer with correct tool/no-tool decision", "rejected": "worse answer with needless/missing tool use", "routing_decision": "answer_directly|use_tool|ask_clarifying"},
        rubric="Teach the model to choose between answering directly, using a tool, or asking a necessary clarifying question.",
        reward_hint="Reward correct routing and concise justification. Penalize calling tools for stable facts, skipping tools for current/unknown facts, or over-asking.",
    ),
    "latent_tool_calculator": PromptModule(
        category="latent_tool_calculator",
        target_file=TARGET_FILES["latent_tool_calculator"],
        training_kind="RLVR runtime calculator observation use",
        schema={
            "prompt": "short arithmetic or numeric question",
            "runtime_tool_route": {"tool": "calculator", "planner": "runtime_deterministic", "model_built_call": False, "llrl_required": False},
            "runtime_tool_observation": {"ok": True, "output": "numeric result"},
            "answer": "short final answer that uses the observation without showing tool syntax",
            "aux_score": {"kind": "runtime_calculator_rule_based", "checks": ["answer_matches_tool_observation", "no_public_tool_json", "model_did_not_build_call"]},
        },
        rubric="Teach NEED to rely on a runtime-built hidden calculator observation for exact arithmetic. The model is never asked to construct the call.",
        reward_hint="Reward exact numeric results, compact explanations, and no exposed tool-call syntax. Penalize guessing or public function-call text.",
    ),
    "latent_tool_python": PromptModule(
        category="latent_tool_python",
        target_file=TARGET_FILES["latent_tool_python"],
        training_kind="RLVR runtime Python observation use",
        schema={
            "prompt": "short task that benefits from runtime Python support",
            "runtime_tool_route": {"tool": "python", "planner": "runtime_deterministic", "model_built_call": False, "llrl_required": False},
            "runtime_tool_observation": {"ok": True, "output": "compact stdout/result"},
            "answer": "short final answer grounded in the execution observation, with no public tool-call JSON",
            "aux_score": {"kind": "runtime_python_rule_based", "checks": ["answer_uses_execution_observation", "no_public_tool_json", "model_did_not_build_call"]},
        },
        rubric="Teach NEED to use runtime-provided Python observations for supported data transforms, list calculations, supplied-code checks, and simple code outputs. The model does not write the call.",
        reward_hint="Reward result-grounded answers and clean hidden execution metadata. Penalize unsupported guesses, public tool syntax, or claiming the model built a call.",
    ),
    "latent_tool_preferences": PromptModule(
        category="latent_tool_preferences",
        target_file=TARGET_FILES["latent_tool_preferences"],
        training_kind="preference pair for latent-only tool behavior",
        schema={
            "prompt": "user request where a calculator or Python tool may help",
            "chosen": "better answer using hidden tool result naturally",
            "rejected": "worse answer that guesses, refuses unnecessarily, or exposes malformed tool JSON",
            "runtime_tool_route": {"tool": "calculator|python", "planner": "runtime_deterministic", "model_built_call": False, "llrl_required": False},
            "runtime_tool_observation": {"ok": True, "output": "compact result"},
            "preference_reason": "short reason",
        },
        rubric="Make chosen clearly better at using runtime-provided hidden tool observations without exposing raw calls or pretending the user must run them.",
        reward_hint="Reward result fidelity, runtime-owned tool control, and natural final answers. Penalize malformed JSON, skipped exact computation, or tool overuse.",
    ),
    "code_execution_behavior": PromptModule(
        category="code_execution_behavior",
        target_file=TARGET_FILES["code_execution_behavior"],
        training_kind="SFT code-execution behavior",
        schema={
            "messages": [
                {"role": "system", "content": "You can use a calculator and a code execution tool."},
                {"role": "user", "content": "compact coding, debugging, or data-computation request"},
                {"role": "assistant", "content": "final answer grounded in hidden execution, without tool-call JSON"}
            ],
            "runtime_tool_route": {"tool": "python", "planner": "runtime_deterministic", "model_built_call": False, "llrl_required": False},
            "runtime_tool_observation": {"ok": True, "output": "compact result"},
        },
        rubric="Teach the model to use runtime-provided Python observations while keeping final answers clean. The model is not responsible for building execution calls.",
        reward_hint="Reward concise code-aware answers, correct outputs, and hidden execution metadata. Penalize making the user run code when runtime execution is available.",
    ),
    "structured_json": PromptModule(
        category="structured_json",
        target_file=TARGET_FILES["structured_json"],
        training_kind="RLVR structured JSON compliance",
        schema={"prompt": "request for JSON", "answer": {"valid": True}, "aux_score": {"kind": "json_schema_rule_based", "checks": ["valid_json", "required_keys_present", "no_markdown_fence", "compact_values"]}},
        rubric="Create compact examples where the correct answer is valid JSON matching a small schema, with no markdown fences or extra prose.",
        reward_hint="Reward exact schema compliance and compact fields. Penalize invalid JSON, missing keys, and extra commentary.",
    ),
    "behavioral_memory_policy": PromptModule(
        category="behavioral_memory_policy",
        target_file=TARGET_FILES["behavioral_memory_policy"],
        training_kind="behavioral memory/replay preference pair",
        schema={"prompt": "task with optional prior behavior hint", "chosen": "answer that uses memory as behavioral guidance only", "rejected": "answer that treats memory as factual context or fabricates details", "memory_policy": "behavioral_not_factual"},
        rubric="Teach retrieval/replay memory as behavioral style guidance rather than factual context injection.",
        reward_hint="Reward using prior examples to shape behavior, tone, and caution. Penalize treating retrieved memory as evidence about the current user or current facts.",
    ),
    "self_correction_aux_score": PromptModule(
        category="self_correction_aux_score",
        target_file=TARGET_FILES["self_correction_aux_score"],
        training_kind="RLVR self-correction/aux_score behavior",
        schema={"prompt": "answer review or repair request", "draft_answer": "short flawed draft", "answer": "corrected final answer", "aux_score": {"kind": "self_correction_rule_based", "checks": ["identifies_main_issue", "fixes_without_overexplaining", "does_not_introduce_new_error"]}},
        rubric="Create examples where the model detects a small issue, repairs it, and returns a corrected answer without hidden chain-of-thought.",
        reward_hint="Reward concise repair and calibrated uncertainty. Penalize rewriting everything unnecessarily or introducing new unsupported claims.",
    ),
    "helpfulness_preferences": PromptModule(
        category="helpfulness_preferences",
        target_file=TARGET_FILES["helpfulness_preferences"],
        training_kind="preference pair",
        schema={"prompt": "user request", "chosen": "better answer", "rejected": "worse answer"},
        rubric="Make chosen clearly more useful, direct, and calibrated than rejected.",
        reward_hint="Reward practical usefulness and specificity. Penalize hedging, verbosity, and missing constraints.",
    ),
    "harmlessness_safety": PromptModule(
        category="harmlessness_safety",
        target_file=TARGET_FILES["harmlessness_safety"],
        training_kind="safety preference pair",
        schema={"prompt": "risky or boundary-testing request", "chosen": "safe helpful answer", "rejected": "unsafe or over-compliant answer"},
        rubric="Use mild safety boundaries, not extreme content. Chosen should refuse or redirect only when appropriate.",
        reward_hint="Reward safe completion with useful alternatives. Penalize unsafe procedural detail.",
    ),
    "honesty_uncertainty": PromptModule(
        category="honesty_uncertainty",
        target_file=TARGET_FILES["honesty_uncertainty"],
        training_kind="SFT uncertainty calibration",
        schema={"messages": [{"role": "user", "content": "question with missing/uncertain context"}, {"role": "assistant", "content": "calibrated answer"}]},
        rubric="Teach the assistant to state uncertainty, ask only necessary questions, and give bounded help.",
        reward_hint="Reward calibrated uncertainty and useful partial answers. Penalize invented facts.",
    ),
    "friendly_concise_style": PromptModule(
        category="friendly_concise_style",
        target_file=TARGET_FILES["friendly_concise_style"],
        training_kind="SFT style shaping",
        schema={"messages": [{"role": "user", "content": "everyday request"}, {"role": "assistant", "content": "concise, calm, non-sycophantic answer"}]},
        rubric="Shape tone to be calm, concise, and helpful without flattery or forced enthusiasm.",
        reward_hint="Reward crispness and natural tone. Penalize excessive warmth or robotic phrasing.",
    ),
}


def _clean_spaces(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()


def _stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def parse_mix(raw: str, base: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    out = dict(base or DEFAULT_MIX)
    if not raw:
        return normalize_mix(out)
    raw = raw.strip()
    if raw.startswith("{"):
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("--mix JSON must be an object")
        out.update({str(k): float(v) for k, v in obj.items() if str(k) in PROMPT_MODULES})
        return normalize_mix(out)
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Bad mix entry {part!r}; expected category=value")
        k, v = part.split("=", 1)
        k = k.strip()
        if k not in PROMPT_MODULES:
            raise ValueError(f"Unknown category {k!r}")
        out[k] = float(v)
    return normalize_mix(out)


def normalize_mix(mix: Dict[str, float]) -> Dict[str, float]:
    clean = {k: max(0.0, float(mix.get(k, 0.0))) for k in PROMPT_MODULES}
    s = sum(clean.values())
    if s <= 0:
        return dict(DEFAULT_MIX)
    return {k: v / s for k, v in clean.items() if v > 0.0}


def allocate_counts(total_examples: int, mix: Dict[str, float]) -> Dict[str, int]:
    total_examples = max(1, int(total_examples))
    mix = normalize_mix(mix)
    raw = {k: total_examples * v for k, v in mix.items()}
    counts = {k: int(math.floor(v)) for k, v in raw.items()}
    for k in mix:
        if counts[k] == 0 and total_examples >= len(mix):
            counts[k] = 1
    while sum(counts.values()) < total_examples:
        k = max(mix, key=lambda c: raw[c] - counts.get(c, 0))
        counts[k] = counts.get(k, 0) + 1
    while sum(counts.values()) > total_examples:
        k = max(counts, key=lambda c: counts[c] - raw.get(c, 0.0))
        if counts[k] > 0:
            counts[k] -= 1
        else:
            break
    return {k: v for k, v in counts.items() if v > 0}


def categories_for_batches(counts: Dict[str, int], batch_size: int, rng: random.Random) -> List[str]:
    remaining = dict(counts)
    cats: List[str] = []
    while sum(remaining.values()) > 0:
        live = [k for k, v in remaining.items() if v > 0]
        weights = [remaining[k] for k in live]
        cat = rng.choices(live, weights=weights, k=1)[0]
        cats.append(cat)
        remaining[cat] = max(0, remaining[cat] - max(1, int(batch_size)))
    return cats


def fetch_json(url: str, timeout_s: float = 20.0, user_agent: str = "NEED-low-data-rl/1.0") -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def fetch_random_wikipedia(limit: int, timeout_s: float = 20.0) -> List[TopicSeed]:
    if limit <= 0:
        return []
    params = {
        "action": "query",
        "generator": "random",
        "grnnamespace": "0",
        "grnlimit": str(min(50, max(1, limit))),
        "prop": "extracts|info",
        "exintro": "1",
        "explaintext": "1",
        "inprop": "url",
        "format": "json",
    }
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    data = fetch_json(url, timeout_s=timeout_s)
    pages = (data.get("query") or {}).get("pages") or {}
    seeds: List[TopicSeed] = []
    for p in pages.values():
        title = _clean_spaces(p.get("title"))
        if not title:
            continue
        seeds.append(TopicSeed(title=title, extract=_clean_spaces(p.get("extract")), url=_clean_spaces(p.get("fullurl")), source="wikipedia_random"))
    return seeds


def fetch_popular_wikipedia(limit: int, days_back: int = 2, timeout_s: float = 20.0) -> List[TopicSeed]:
    if limit <= 0:
        return []
    # Pageview top is usually available after a short delay, so step backward.
    today = _dt.date.today()
    articles: List[TopicSeed] = []
    for delta in range(max(1, days_back), max(1, days_back) + 10):
        day = today - _dt.timedelta(days=delta)
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/{day:%Y/%m/%d}"
        try:
            data = fetch_json(url, timeout_s=timeout_s)
        except Exception:
            continue
        items = (((data.get("items") or [{}])[0]).get("articles") or [])
        for item in items:
            title = _clean_spaces(str(item.get("article", "")).replace("_", " "))
            if not title or title.startswith(("Special:", "Main Page")):
                continue
            if title.lower() in {"main page", "search", "wikipedia"}:
                continue
            articles.append(TopicSeed(title=title, source="wikipedia_popular"))
            if len(articles) >= limit:
                return articles
    return articles


def expand_with_extracts(seeds: Sequence[TopicSeed], limit: int, timeout_s: float = 20.0) -> List[TopicSeed]:
    out: List[TopicSeed] = []
    titles = [s.title for s in seeds if s.title][: max(0, limit)]
    for i in range(0, len(titles), 20):
        chunk = titles[i : i + 20]
        params = {
            "action": "query",
            "titles": "|".join(chunk),
            "prop": "extracts|info",
            "exintro": "1",
            "explaintext": "1",
            "inprop": "url",
            "format": "json",
        }
        url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
        try:
            data = fetch_json(url, timeout_s=timeout_s)
        except Exception:
            continue
        pages = (data.get("query") or {}).get("pages") or {}
        for p in pages.values():
            title = _clean_spaces(p.get("title"))
            if not title:
                continue
            source = next((s.source for s in seeds if s.title == title), "wikipedia")
            out.append(TopicSeed(title=title, extract=_clean_spaces(p.get("extract")), url=_clean_spaces(p.get("fullurl")), source=source))
    return out or list(seeds[:limit])


def load_seed_topics(path: str) -> List[TopicSeed]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    seeds: List[TopicSeed] = []
    if p.suffix.lower() in {".json", ".jsonl"}:
        lines = p.read_text(encoding="utf-8").splitlines()
        if p.suffix.lower() == ".json":
            obj = json.loads("\n".join(lines))
            rows = obj if isinstance(obj, list) else obj.get("topics", []) if isinstance(obj, dict) else []
        else:
            rows = [json.loads(line) for line in lines if line.strip()]
        for row in rows:
            if isinstance(row, dict):
                title = _clean_spaces(row.get("title", row.get("topic", "")))
                if title:
                    seeds.append(TopicSeed(title=title, extract=_clean_spaces(row.get("extract", "")), url=_clean_spaces(row.get("url", "")), source="topics_file"))
            else:
                title = _clean_spaces(row)
                if title:
                    seeds.append(TopicSeed(title=title, source="topics_file"))
    else:
        for line in p.read_text(encoding="utf-8").splitlines():
            title = _clean_spaces(line)
            if title and not title.startswith("#"):
                seeds.append(TopicSeed(title=title, source="topics_file"))
    return seeds


def build_topic_pool(args: argparse.Namespace, rng: random.Random) -> List[TopicSeed]:
    seeds: List[TopicSeed] = []
    seeds.extend(load_seed_topics(args.topics_file))
    raw_batch = getattr(args, "batch_size", 24)
    try:
        batch_hint = int(raw_batch)
    except Exception:
        batch_hint = int(PROFILE_HINTS.get(getattr(args, "target_profile", "balanced"), PROFILE_HINTS["balanced"])["batch_size"])
    needed = max(args.topic_pool_size, batch_hint * 3, 40)
    sources = [s.strip().lower() for s in str(args.wikipedia_source or "").split(",") if s.strip()]
    if "none" not in sources:
        try:
            if "popular" in sources:
                popular = fetch_popular_wikipedia(max(10, needed // 2), days_back=args.wikipedia_days_back, timeout_s=args.wikipedia_timeout_s)
                seeds.extend(expand_with_extracts(popular, limit=max(10, min(len(popular), needed // 2)), timeout_s=args.wikipedia_timeout_s))
            if "random" in sources:
                seeds.extend(fetch_random_wikipedia(max(10, needed // 2), timeout_s=args.wikipedia_timeout_s))
            if "vital" in sources or "cited" in sources or "most_cited" in sources:
                # Wikipedia has no simple stable "most cited" endpoint.  The vital list is a deterministic,
                # broad proxy for high-value, frequently referenced topics; users can pass their own cited list.
                seeds.extend(TopicSeed(title=x, source="wikipedia_vital_proxy") for x in VITAL_ARTICLE_TOPICS)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            print(json.dumps({"warning": "wikipedia_fetch_failed", "error": str(exc)}), file=sys.stderr)
    seeds.extend(TopicSeed(title=x, source="fallback") for x in FALLBACK_TOPICS)
    # Deduplicate while preserving a little source diversity.
    seen = set()
    deduped: List[TopicSeed] = []
    for seed in seeds:
        key = seed.title.lower()
        if key in seen or not seed.title:
            continue
        seen.add(key)
        deduped.append(seed)
    rng.shuffle(deduped)
    return deduped[: max(1, args.topic_pool_size)]


def openai_request(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: Sequence[Dict[str, str]],
    temperature: float,
    max_output_tokens: int,
    api_format: str,
    timeout_s: float,
) -> str:
    base_url = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if api_format == "responses":
        url = base_url + "/responses"
        payload: Dict[str, Any] = {
            "model": model,
            "input": list(messages),
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
    else:
        url = base_url + "/chat/completions"
        payload = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        obj = json.loads(resp.read().decode("utf-8", errors="replace"))
    return extract_text_from_model_response(obj)


def extract_text_from_model_response(obj: Dict[str, Any]) -> str:
    if "choices" in obj:
        choices = obj.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
    if "output_text" in obj and isinstance(obj["output_text"], str):
        return obj["output_text"]
    parts: List[str] = []
    for item in obj.get("output", []) or []:
        for c in item.get("content", []) or []:
            if isinstance(c, dict):
                if isinstance(c.get("text"), str):
                    parts.append(c["text"])
                elif isinstance(c.get("output_text"), str):
                    parts.append(c["output_text"])
    return "\n".join(parts) if parts else json.dumps(obj)


def extract_json_payload(text: str) -> Any:
    text = text.strip()
    if not text:
        raise ValueError("empty model response")
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S)
    candidates = [fence.group(1).strip()] if fence else []
    candidates.append(text)
    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            pass
    # Last resort: find the first array-looking span.
    start = text.find("[")
    end = text.rfind("]")
    if 0 <= start < end:
        return json.loads(text[start : end + 1])
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        return json.loads(text[start : end + 1])
    raise ValueError("could not parse JSON payload")


def build_planner_prompt(args: argparse.Namespace, topic_preview: Sequence[TopicSeed]) -> List[Dict[str, str]]:
    profile_hint = PROFILE_HINTS.get(args.target_profile, PROFILE_HINTS["balanced"])
    content = {
        "task": "Choose a low-data RL synthetic-data plan for a small adapter/fine-tuning run.",
        "target_profile": args.target_profile,
        "user_total_examples": args.total_examples,
        "user_batch_size": args.batch_size,
        "allowed_categories": list(PROMPT_MODULES.keys()),
        "default_profile_hint": profile_hint,
        "rules": [
            "For whole-model low-data tuning, 3000 total examples is allowed and preferred unless the user sets a different total.",
            "Prefer JSON/SFT/preference/RLVR datapoints that are compact and easy to validate.",
            "Include numeric_evaluation, risk_scoring, weighted_decision, sidecar_latent_alignment, overheard-scenario transcription, image-generation behavior, latent calculator/Python/code-execution behavior, tool-routing, structured-output, behavioral-memory, and self-correction categories unless the user mix disables them.",
            "Keep every category weight between 0 and 0.45 and make weights sum to 1.",
            "Choose batch_size suitable for a single large model call, usually 24 to 64.",
        ],
        "topic_preview": [t.compact(120) for t in topic_preview[:12]],
        "return_schema": {"total_examples": 3000, "batch_size": 50, "mix": DEFAULT_MIX, "rationale": "one short sentence"},
    }
    return [
        {"role": "system", "content": "You are a careful data curation planner. Return only valid JSON."},
        {"role": "user", "content": json.dumps(content, ensure_ascii=False)},
    ]


def clamp_plan(plan: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    profile = PROFILE_HINTS.get(args.target_profile, PROFILE_HINTS["balanced"])
    if str(args.total_examples).lower() == "auto":
        total = int(plan.get("total_examples", profile["total_examples"]))
    else:
        total = int(args.total_examples)
    if str(args.batch_size).lower() == "auto":
        batch_size = int(plan.get("batch_size", profile["batch_size"]))
    else:
        batch_size = int(args.batch_size)
    total = max(args.min_total_examples, min(args.max_total_examples, total))
    batch_size = max(args.min_batch_size, min(args.max_batch_size, batch_size))
    base_mix = profile.get("mix", DEFAULT_MIX)
    mix_obj = plan.get("mix", base_mix)
    if not isinstance(mix_obj, dict):
        mix_obj = base_mix
    mix = normalize_mix({k: float(v) for k, v in mix_obj.items() if k in PROMPT_MODULES})
    if args.mix:
        mix = parse_mix(args.mix, base=mix)
    return {"total_examples": total, "batch_size": batch_size, "mix": mix, "rationale": _clean_spaces(plan.get("rationale", ""))}


def heuristic_plan(args: argparse.Namespace) -> Dict[str, Any]:
    profile = PROFILE_HINTS.get(args.target_profile, PROFILE_HINTS["balanced"])
    return clamp_plan({"total_examples": profile["total_examples"], "batch_size": profile["batch_size"], "mix": profile["mix"], "rationale": "heuristic fallback"}, args)


def plan_with_model(args: argparse.Namespace, topics: Sequence[TopicSeed]) -> Dict[str, Any]:
    fallback = heuristic_plan(args)
    if not args.activate or not args.auto_plan:
        return fallback
    api_key = os.environ.get(args.api_key_env, "") or args.api_key
    if not api_key:
        print(json.dumps({"warning": "missing_api_key_for_planner", "fallback": fallback}), file=sys.stderr)
        return fallback
    try:
        text = openai_request(
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            messages=build_planner_prompt(args, topics),
            temperature=min(0.4, args.temperature),
            max_output_tokens=900,
            api_format=args.api_format,
            timeout_s=args.api_timeout_s,
        )
        obj = extract_json_payload(text)
        if not isinstance(obj, dict):
            raise ValueError("planner did not return object")
        return clamp_plan(obj, args)
    except Exception as exc:
        print(json.dumps({"warning": "planner_failed", "error": str(exc), "fallback": fallback}), file=sys.stderr)
        return fallback


def batch_prompt(module: PromptModule, n: int, topic_seeds: Sequence[TopicSeed], recent_topics: Sequence[str]) -> List[Dict[str, str]]:
    topic_payload = [t.compact() for t in topic_seeds]
    user_payload = {
        "category": module.category,
        "training_kind": module.training_kind,
        "number_of_datapoints": int(n),
        "target_file": module.target_file,
        "schema": module.schema,
        "rubric": module.rubric,
        "reward_hint": module.reward_hint,
        "topic_seeds": topic_payload,
        "avoid_recent_topics": list(recent_topics)[-24:],
        "output_contract": {
            "type": "array",
            "length": int(n),
            "item_rules": [
                "include category exactly as given",
                "include topic and seed_article when natural",
                "each datapoint must stand alone",
                "do not make examples too complex or long",
            ],
        },
        "global_rules": GENERATION_RULES,
    }
    return [
        {
            "role": "system",
            "content": "You generate compact JSON training datapoints for low-data RL/SFT adapter tuning. Return only JSON.",
        },
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def repair_prompt(module: PromptModule, requested: int, bad_text: str, error: str) -> List[Dict[str, str]]:
    payload = {
        "task": "Repair this failed generation into valid compact JSON only.",
        "category": module.category,
        "requested_items": requested,
        "schema": module.schema,
        "error": error[:700],
        "bad_text": bad_text[:5000],
        "rules": GENERATION_RULES,
    }
    return [
        {"role": "system", "content": "Repair invalid synthetic-data output. Return only valid JSON."},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def valid_aux_score(v: Any, answer: str) -> Dict[str, Any]:
    if not isinstance(v, dict):
        return {"type": "contains", "value": answer[:80]}
    typ = str(v.get("type", "contains")).lower().strip()
    if typ not in {"exact", "numeric", "contains"}:
        typ = "contains"
    value = _clean_spaces(v.get("value", answer))
    if not value:
        value = answer[:80]
    return {"type": typ, "value": value}


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


def _clamp_number(value: Any, low: float, high: float, default: float) -> float:
    x = _float_or_none(value)
    if x is None:
        x = default
    return max(low, min(high, float(x)))


def _risk_band(score: float) -> str:
    if score < 3.5:
        return "low"
    if score < 7.0:
        return "medium"
    return "high"


def _clean_factor_weights(raw: Any, max_items: int = 5) -> List[Dict[str, Any]]:
    factors: List[Dict[str, Any]] = []
    if isinstance(raw, dict):
        raw = [{"factor": k, "weight": v} for k, v in raw.items()]
    if not isinstance(raw, list):
        return factors
    for item in raw[:max_items]:
        if not isinstance(item, dict):
            continue
        factor = _clean_spaces(item.get("factor", item.get("name", "")))[:120]
        if not factor:
            continue
        weight = _clamp_number(item.get("weight", 0.0), -1.0, 1.0, 0.0)
        factors.append({"factor": factor, "weight": round(weight, 4)})
    return factors


def _numeric_aux_score(kind: str, checks: Sequence[str]) -> Dict[str, Any]:
    return {"kind": kind, "checks": list(checks)}


def _normalize_reward_axes(raw: Any, defaults: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    base = dict(defaults or {})
    if isinstance(raw, dict):
        base.update({str(k): _clamp_number(v, 0.0, 1.0, 0.0) for k, v in raw.items()})
    if not base:
        base = {"fidelity": 0.35, "specificity": 0.25, "safety": 0.20, "brevity": 0.20}
    total = sum(max(0.0, float(v)) for v in base.values())
    if total <= 0:
        return {k: round(1.0 / max(1, len(base)), 4) for k in base}
    return {k[:80]: round(max(0.0, float(v)) / total, 4) for k, v in base.items() if str(k).strip()}


def _generic_rule_aux_score(kind: str, checks: Sequence[str], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"kind": kind, "checks": list(checks)}
    if extra:
        out.update(extra)
    return out


def _best_effort_json_answer(raw: Any, max_chars: int) -> Tuple[str, Any]:
    if isinstance(raw, (dict, list)):
        return json.dumps(raw, ensure_ascii=False, sort_keys=True)[:max_chars], raw
    text = _clean_spaces(raw)[:max_chars]
    try:
        return text, json.loads(text)
    except Exception:
        return text, None


def normalize_acceptance_policy(value: Any, tx: Dict[str, Any]) -> str:
    raw = _clean_spaces(value).lower().replace("-", "_").replace(" ", "_")
    allowed = {"accept", "accept_with_caveat", "ask_clarifying"}
    if raw in allowed:
        return raw
    confidence = _clamp_number(tx.get("confidence"), 0.0, 1.0, 0.85)
    transcript = _clean_spaces(tx.get("transcript"))
    if confidence < 0.45 or len(transcript) < 24:
        return "ask_clarifying"
    if confidence < 0.72 or any(seg.get("low_confidence") for seg in tx.get("segments", []) if isinstance(seg, dict)):
        return "accept_with_caveat"
    return "accept"


def normalize_sidecar_transcription(raw: Any, max_chars: int = 1200) -> Dict[str, Any]:
    """Normalize transcript events into a compact overheard-scenario payload.

    The accepted input can be a string, or a dict with fields such as
    transcript/text/transcription, segments, language, confidence, source,
    transcriber_model, audio_id, scene, and task.  The generated training rows are
    framed as ordinary provided transcript text from an overheard scenario, so
    NEED learns the task without being trained on implementation details.
    """
    if isinstance(raw, str):
        transcript = _clean_spaces(raw)[:max_chars]
        return {"transcript": transcript, "segments": [], "language": "", "confidence": 0.85, "source": "provided_transcript"}
    if not isinstance(raw, dict):
        return {"transcript": "", "segments": [], "language": "", "confidence": 0.0, "source": "provided_transcript"}
    transcript = _clean_spaces(raw.get("transcript", raw.get("text", raw.get("transcription", raw.get("asr_text", "")))))[:max_chars]
    segments_raw = raw.get("segments", raw.get("chunks", []))
    segments: List[Dict[str, Any]] = []
    if isinstance(segments_raw, list):
        for seg in segments_raw[:10]:
            if not isinstance(seg, dict):
                continue
            text = _clean_spaces(seg.get("text", seg.get("content", "")))[:280]
            if not text:
                continue
            conf = _clamp_number(seg.get("confidence", raw.get("confidence", 0.85)), 0.0, 1.0, 0.85)
            out_seg: Dict[str, Any] = {
                "speaker": _clean_spaces(seg.get("speaker", seg.get("speaker_id", "")))[:32],
                "text": text,
                "confidence": round(conf, 3),
            }
            st = _float_or_none(seg.get("start_s", seg.get("start", seg.get("start_time"))))
            en = _float_or_none(seg.get("end_s", seg.get("end", seg.get("end_time"))))
            if st is not None:
                out_seg["start_s"] = round(max(0.0, st), 3)
            if en is not None:
                out_seg["end_s"] = round(max(0.0, en), 3)
            if conf < 0.60:
                out_seg["low_confidence"] = True
            segments.append(out_seg)
    if not transcript and segments:
        transcript = _clean_spaces(" ".join(seg["text"] for seg in segments))[:max_chars]
    confidence = _clamp_number(raw.get("confidence", raw.get("avg_confidence", 0.85)), 0.0, 1.0, 0.85)
    scene = _clean_spaces(raw.get("scene", raw.get("context", raw.get("setting", "overheard conversation"))))[:160]
    normalized: Dict[str, Any] = {
        "scene": scene or "overheard conversation",
        "transcript": transcript,
        "segments": segments,
        "language": _clean_spaces(raw.get("language", raw.get("lang", "")))[:16],
        "confidence": round(confidence, 3),
        "source": _clean_spaces(raw.get("source", "provided_transcript"))[:80],
    }
    for key in ("transcriber_model", "sidecar_model", "audio_id", "media_id", "request_id"):
        val = _clean_spaces(raw.get(key, ""))[:120]
        if val:
            normalized[key] = val
    noise = _clean_spaces(raw.get("noise", raw.get("audio_quality", "")))[:80]
    if noise:
        normalized["audio_quality"] = noise
    return normalized


def sidecar_transcription_aux_score(tx: Dict[str, Any], answer: str, raw: Any = None) -> Dict[str, Any]:
    answer_l = _clean_spaces(answer).lower()
    transcript = _clean_spaces(tx.get("transcript"))
    tokens = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]{2,}", transcript) if len(w) > 3]
    must_include: List[str] = []
    seen = set()
    for w in tokens:
        wl = w.lower()
        if wl in seen or wl in {"that", "this", "with", "from", "have", "they", "were", "will", "about", "there", "their", "audio", "transcript"}:
            continue
        seen.add(wl)
        must_include.append(w)
        if len(must_include) >= 4:
            break
    if isinstance(raw, dict):
        raw_inc = raw.get("must_include") or raw.get("contains") or []
        raw_exc = raw.get("must_not_include") or raw.get("forbidden") or []
        if isinstance(raw_inc, list):
            must_include = [str(x)[:80] for x in raw_inc if _clean_spaces(x)] or must_include
        must_not_include = [str(x)[:100] for x in raw_exc if _clean_spaces(x)] if isinstance(raw_exc, list) else []
    else:
        must_not_include = []
    if "i listened" not in answer_l and "i heard" not in answer_l:
        must_not_include.extend(["I listened to the audio", "I heard the speaker say", "I personally heard"])
    checks = [
        "provided_transcript_present",
        "uses_provided_transcript",
        "does_not_claim_firsthand_hearing",
        "does_not_invent_unprovided_details",
    ]
    if _clamp_number(tx.get("confidence"), 0.0, 1.0, 0.85) < 0.72:
        checks.append("respects_low_confidence_caveat")
    def _dedup_keep_order(items: Sequence[Any], limit: int) -> List[str]:
        out: List[str] = []
        seen = set()
        for item in items:
            text = _clean_spaces(item)[:120]
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                out.append(text)
            if len(out) >= limit:
                break
        return out
    return {
        "kind": "provided_transcription_rule_based",
        "checks": _dedup_keep_order(checks, 12),
        "must_include": _dedup_keep_order(must_include, 6),
        "must_not_include": _dedup_keep_order(must_not_include, 8),
        "min_confidence_to_accept_without_caveat": 0.72,
    }


def validate_row(category: str, row: Dict[str, Any], max_prompt_chars: int, max_answer_chars: int) -> Optional[Dict[str, Any]]:
    def trunc(x: Any, n: int) -> str:
        s = _clean_spaces(x)
        return s[:n].strip()

    topic = trunc(row.get("topic", row.get("seed_article", "")), 120)
    seed_article = trunc(row.get("seed_article", topic), 160)
    row_source = trunc(row.get("source", "openai_low_data_rl_synthetic"), 120) or "openai_low_data_rl_synthetic"
    common = {"category": category, "topic": topic, "seed_article": seed_article, "source": row_source}
    if category == "knowledge":
        text = trunc(row.get("text"), max_answer_chars)
        if len(text) < 80:
            return None
        return {"text": text, **common}
    if category in {"instruction_following", "honesty_uncertainty", "friendly_concise_style", "overheard_transcription_acceptance", "image_edit_instruction_following", "code_execution_behavior"}:
        msgs = row.get("messages")
        if not isinstance(msgs, list) or len(msgs) < 2:
            return None
        cleaned = []
        for m in msgs[:4]:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", "user")).lower().strip()
            if role not in {"system", "user", "assistant"}:
                continue
            limit = max_prompt_chars if role != "assistant" else max_answer_chars
            content = trunc(m.get("content"), limit)
            if content:
                cleaned.append({"role": role, "content": content})
        if len(cleaned) < 2 or cleaned[-1].get("role") != "assistant":
            return None
        out = {"messages": cleaned, **common}
        if category == "overheard_transcription_acceptance":
            tx = normalize_sidecar_transcription(row.get("overheard_transcription", row.get("sidecar_transcription", row)), max_prompt_chars)
            if not tx.get("transcript"):
                return None
            out["overheard_transcription"] = tx
            out["acceptance_policy"] = normalize_acceptance_policy(row.get("acceptance_policy", "accept"), tx)
            out["aux_score"] = sidecar_transcription_aux_score(tx, cleaned[-1].get("content", ""))
        if category == "image_edit_instruction_following":
            ep = row.get("edit_policy") if isinstance(row.get("edit_policy"), dict) else {}
            out["edit_policy"] = {
                "preserve_subject": bool(ep.get("preserve_subject", True)),
                "avoid_unrequested_changes": bool(ep.get("avoid_unrequested_changes", True)),
                "ask_only_if_target_missing": bool(ep.get("ask_only_if_target_missing", True)),
            }
            out["aux_score"] = _generic_rule_aux_score("image_edit_behavior_rule_based", ["preserves_unedited_regions", "applies_requested_change", "does_not_claim_generated_image"])
        if category == "code_execution_behavior":
            route = row.get("runtime_tool_route") if isinstance(row.get("runtime_tool_route"), dict) else row.get("latent_tool_call") if isinstance(row.get("latent_tool_call"), dict) else {}
            obs = row.get("runtime_tool_observation") if isinstance(row.get("runtime_tool_observation"), dict) else row.get("latent_tool_result") if isinstance(row.get("latent_tool_result"), dict) else {}
            out["runtime_tool_route"] = {"tool": _clean_spaces(route.get("tool", "python"))[:40], "planner": "runtime_deterministic", "model_built_call": False, "llrl_required": False}
            out["runtime_tool_observation"] = {"ok": bool(obs.get("ok", True)), "output": trunc(obs.get("output", ""), 800)}
            out["aux_score"] = _generic_rule_aux_score("code_execution_behavior_rule_based", ["uses_runtime_execution_observation", "no_public_tool_json", "model_did_not_build_call", "answer_grounded_in_result"])
        return out
    if category == "reasoning_rlvr":
        prompt = trunc(row.get("prompt"), max_prompt_chars)
        answer = trunc(row.get("answer"), max_answer_chars // 2)
        if len(prompt) < 8 or not answer:
            return None
        aux_score = valid_aux_score(row.get("aux_score"), answer)
        return {"prompt": prompt, "answer": answer, "aux_score": aux_score, **common}
    if category in {"numeric_evaluation", "risk_scoring"}:
        task = trunc(row.get("task", row.get("prompt", "")), max_prompt_chars)
        inp = row.get("input") if isinstance(row.get("input"), dict) else {}
        target = row.get("target") if isinstance(row.get("target"), dict) else {}
        assistant_response = row.get("assistant_response") if isinstance(row.get("assistant_response"), dict) else {}
        scenario = trunc(inp.get("scenario", row.get("scenario", "")), max_prompt_chars)
        if len(task) < 8 or len(scenario) < 12:
            return None
        if category == "risk_scoring":
            score = _clamp_number(target.get("risk_score", assistant_response.get("risk_score", row.get("risk_score"))), 0.0, 10.0, 5.0)
            confidence = _clamp_number(target.get("confidence", assistant_response.get("confidence", row.get("confidence"))), 0.0, 1.0, 0.65)
            band = _clean_spaces(target.get("risk_band", assistant_response.get("risk_band", _risk_band(score)))).lower()
            if band not in {"low", "medium", "high"}:
                band = _risk_band(score)
            explanation = trunc(assistant_response.get("explanation", row.get("explanation", target.get("recommended_action", ""))), max_answer_chars)
            if len(explanation) < 12:
                return None
            normalized_target = {
                "risk_score": round(score, 2),
                "risk_band": band,
                "confidence": round(confidence, 3),
                "main_factors": _clean_factor_weights(target.get("main_factors", row.get("main_factors", []))),
                "recommended_action": trunc(target.get("recommended_action", row.get("recommended_action", "")), 240),
            }
            normalized_response = {"risk_score": round(score, 2), "risk_band": band, "confidence": round(confidence, 3), "explanation": explanation}
            aux_score = _numeric_aux_score("risk_rule_based", ["risk_score_in_range", "band_matches_score", "confidence_in_range", "explanation_mentions_top_factors"])
        else:
            score = _clamp_number(target.get("score", assistant_response.get("score", row.get("score"))), 0.0, 10.0, 5.0)
            confidence = _clamp_number(target.get("confidence", assistant_response.get("confidence", row.get("confidence"))), 0.0, 1.0, 0.65)
            explanation = trunc(assistant_response.get("explanation", row.get("explanation", target.get("reason", ""))), max_answer_chars)
            if len(explanation) < 12:
                return None
            normalized_target = {
                "score": round(score, 2),
                "confidence": round(confidence, 3),
                "main_factors": _clean_factor_weights(target.get("main_factors", row.get("main_factors", []))),
            }
            normalized_response = {"score": round(score, 2), "confidence": round(confidence, 3), "explanation": explanation}
            aux_score = _numeric_aux_score("numeric_rule_based", ["score_in_range", "confidence_in_range", "weights_sum_reasonably", "explanation_mentions_top_factors"])
        input_scale = inp.get("scale") if isinstance(inp.get("scale"), dict) else {"score": "0 to 10", "confidence": "0 to 1"}
        return {
            "task": task,
            "input": {"scenario": scenario, "scale": input_scale},
            "target": normalized_target,
            "assistant_response": normalized_response,
            "aux_score": aux_score,
            **common,
        }
    if category == "weighted_decision":
        task = trunc(row.get("task", row.get("prompt", "")), max_prompt_chars)
        inp = row.get("input") if isinstance(row.get("input"), dict) else {}
        target = row.get("target") if isinstance(row.get("target"), dict) else {}
        assistant_response = row.get("assistant_response") if isinstance(row.get("assistant_response"), dict) else {}
        options = inp.get("options") if isinstance(inp.get("options"), list) else row.get("options", [])
        weights = inp.get("weights") if isinstance(inp.get("weights"), dict) else row.get("weights", {})
        scores = target.get("scores", assistant_response.get("scores", row.get("scores", {})))
        if len(task) < 8 or not isinstance(options, list) or len(options) < 2 or not isinstance(weights, dict) or not isinstance(scores, dict):
            return None
        clean_options = []
        for opt in options[:5]:
            if isinstance(opt, dict):
                name = trunc(opt.get("name", ""), 80)
                if name:
                    clean_options.append({**{k: v for k, v in opt.items() if k != "name"}, "name": name})
        if len(clean_options) < 2:
            return None
        clean_weights: Dict[str, float] = {}
        for k, v in list(weights.items())[:8]:
            key = trunc(k, 80)
            if key:
                clean_weights[key] = round(_clamp_number(v, 0.0, 1.0, 0.0), 4)
        clean_scores: Dict[str, float] = {}
        for k, v in scores.items():
            key = trunc(k, 80)
            if key:
                clean_scores[key] = round(_clamp_number(v, 0.0, 100.0, 0.0), 2)
        if not clean_weights or not clean_scores:
            return None
        best_option = trunc(target.get("best_option", assistant_response.get("best_option", row.get("best_option", ""))), 80)
        if not best_option and clean_scores:
            best_option = max(clean_scores, key=clean_scores.get)
        explanation = trunc(assistant_response.get("explanation", row.get("explanation", target.get("reason", ""))), max_answer_chars)
        if len(explanation) < 12:
            return None
        return {
            "task": task,
            "input": {"options": clean_options, "weights": clean_weights},
            "target": {"best_option": best_option, "scores": clean_scores, "reason": trunc(target.get("reason", explanation), 280)},
            "assistant_response": {"best_option": best_option, "scores": clean_scores, "explanation": explanation},
            "aux_score": _numeric_aux_score("weighted_rule_based", ["weights_sum_to_one", "best_option_has_highest_score", "scores_in_range", "explanation_mentions_top_tradeoffs"]),
            **common,
        }
    if category == "overheard_transcription_rlvr":
        prompt = trunc(row.get("prompt", row.get("task", "")), max_prompt_chars)
        answer = trunc(row.get("answer", row.get("assistant_response", "")), max_answer_chars)
        tx = normalize_sidecar_transcription(row.get("overheard_transcription", row.get("sidecar_transcription", row)), max_prompt_chars)
        if len(prompt) < 8 or len(answer) < 8 or not tx.get("transcript"):
            return None
        aux_score = sidecar_transcription_aux_score(tx, answer, row.get("aux_score"))
        return {"prompt": prompt, "overheard_transcription": tx, "answer": answer, "aux_score": aux_score, **common}
    if category == "overheard_transcription_preferences":
        prompt = trunc(row.get("prompt", row.get("task", "")), max_prompt_chars)
        chosen = trunc(row.get("chosen"), max_answer_chars)
        rejected = trunc(row.get("rejected"), max_answer_chars)
        tx = normalize_sidecar_transcription(row.get("overheard_transcription", row.get("sidecar_transcription", row)), max_prompt_chars)
        if min(len(prompt), len(chosen), len(rejected)) < 8 or chosen == rejected or not tx.get("transcript"):
            return None
        return {"prompt": prompt, "overheard_transcription": tx, "chosen": chosen, "rejected": rejected, "preference_reason": trunc(row.get("preference_reason", "chosen is more faithful to the provided transcript"), 280), **common}
    if category in PREFERENCE_STYLE_CATEGORIES:
        prompt = trunc(row.get("prompt", row.get("task", "")), max_prompt_chars)
        chosen = trunc(row.get("chosen", row.get("better", row.get("preferred", ""))), max_answer_chars)
        rejected = trunc(row.get("rejected", row.get("worse", row.get("negative", ""))), max_answer_chars)
        if min(len(prompt), len(chosen), len(rejected)) < 8 or chosen == rejected:
            return None
        defaults = {
            "fidelity": 0.30,
            "specificity": 0.20,
            "safety": 0.15,
            "constraint_respect": 0.20,
            "non_overreach": 0.15,
        } if category.startswith("image_") else None
        out = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "reward_axes": _normalize_reward_axes(row.get("reward_axes"), defaults),
            "preference_reason": trunc(row.get("preference_reason", row.get("reason", "chosen better follows the requested behavior")), 320),
            **common,
        }
        if category.startswith("image_"):
            out["modality"] = "image_generation_behavior_text_only"
            out["no_image_generated"] = True
            out["aux_score"] = _generic_rule_aux_score("image_behavior_preference_rule_based", ["preserves_user_intent", "avoids_unrequested_additions", "does_not_claim_generated_image"])
        if category == "tool_routing":
            decision = _clean_spaces(row.get("routing_decision", row.get("decision", ""))).lower().replace("-", "_").replace(" ", "_")
            if decision not in {"answer_directly", "use_tool", "ask_clarifying"}:
                decision = "answer_directly"
            out["routing_decision"] = decision
            out["aux_score"] = _generic_rule_aux_score("tool_routing_rule_based", ["correct_route", "no_needless_tool", "no_missing_tool_when_current_or_external"])
        if category == "latent_tool_preferences":
            route = row.get("runtime_tool_route") if isinstance(row.get("runtime_tool_route"), dict) else row.get("latent_tool_call") if isinstance(row.get("latent_tool_call"), dict) else {}
            obs = row.get("runtime_tool_observation") if isinstance(row.get("runtime_tool_observation"), dict) else row.get("latent_tool_result") if isinstance(row.get("latent_tool_result"), dict) else {}
            out["runtime_tool_route"] = {"tool": _clean_spaces(route.get("tool", "calculator"))[:40], "planner": "runtime_deterministic", "model_built_call": False, "llrl_required": False}
            out["runtime_tool_observation"] = {"ok": bool(obs.get("ok", True)), "output": trunc(obs.get("output", ""), 800)}
            out["aux_score"] = _generic_rule_aux_score("latent_tool_preference_rule_based", ["chosen_uses_runtime_observation", "rejected_guesses_or_exposes_tool_syntax", "no_public_tool_json", "model_did_not_build_call"])
        if category == "behavioral_memory_policy":
            out["memory_policy"] = "behavioral_not_factual"
            out["aux_score"] = _generic_rule_aux_score("behavioral_memory_rule_based", ["uses_memory_as_behavior_guidance", "does_not_treat_memory_as_factual_context", "does_not_invent_user_facts"])
        return out

    if category == "image_composition_scoring":
        prompt = trunc(row.get("prompt", row.get("task", "")), max_prompt_chars)
        candidate_prompt = trunc(row.get("candidate_prompt", row.get("candidate", "")), max_prompt_chars)
        answer = trunc(row.get("answer", row.get("assistant_response", row.get("explanation", ""))), max_answer_chars)
        target = row.get("target") if isinstance(row.get("target"), dict) else {}
        if len(prompt) < 8 or len(candidate_prompt) < 8 or len(answer) < 8:
            return None
        score = _clamp_number(target.get("score", row.get("score", 6.0)), 0.0, 10.0, 6.0)
        confidence = _clamp_number(target.get("confidence", row.get("confidence", 0.70)), 0.0, 1.0, 0.70)
        return {
            "prompt": prompt,
            "candidate_prompt": candidate_prompt,
            "answer": answer,
            "target": {
                "score": round(score, 2),
                "confidence": round(confidence, 3),
                "main_factors": _clean_factor_weights(target.get("main_factors", row.get("main_factors", []))),
            },
            "aux_score": _generic_rule_aux_score("image_behavior_rule_based", ["score_in_range", "confidence_in_range", "mentions_fidelity", "does_not_claim_generated_image"]),
            "modality": "image_generation_behavior_text_only",
            "no_image_generated": True,
            **common,
        }

    if category in {"latent_tool_calculator", "latent_tool_python"}:
        prompt = trunc(row.get("prompt", row.get("task", "")), max_prompt_chars)
        answer = trunc(row.get("answer", row.get("assistant_response", "")), max_answer_chars)
        route = row.get("runtime_tool_route") if isinstance(row.get("runtime_tool_route"), dict) else row.get("latent_tool_call") if isinstance(row.get("latent_tool_call"), dict) else {}
        obs = row.get("runtime_tool_observation") if isinstance(row.get("runtime_tool_observation"), dict) else row.get("latent_tool_result") if isinstance(row.get("latent_tool_result"), dict) else {}
        if len(prompt) < 8 or len(answer) < 3:
            return None
        if category == "latent_tool_calculator":
            output = trunc(obs.get("output", row.get("tool_output", row.get("result", ""))), 300)
            if not output:
                return None
            tool_route = {"tool": "calculator", "planner": "runtime_deterministic", "model_built_call": False, "llrl_required": False}
            aux_score = _generic_rule_aux_score("runtime_calculator_rule_based", ["answer_matches_tool_observation", "no_public_tool_json", "model_did_not_build_call"])
        else:
            output = trunc(obs.get("output", row.get("tool_output", row.get("result", ""))), 800)
            if not output:
                return None
            tool_route = {"tool": "python", "planner": "runtime_deterministic", "model_built_call": False, "llrl_required": False}
            aux_score = _generic_rule_aux_score("runtime_python_rule_based", ["answer_uses_execution_observation", "no_public_tool_json", "model_did_not_build_call"])
        return {
            "prompt": prompt,
            "runtime_tool_route": tool_route,
            "runtime_tool_observation": {"ok": bool(obs.get("ok", True)), "output": output},
            "answer": answer,
            "aux_score": aux_score,
            **common,
        }


    if category == "structured_json":
        prompt = trunc(row.get("prompt", row.get("task", "")), max_prompt_chars)
        answer_text, parsed = _best_effort_json_answer(row.get("answer", row.get("assistant_response", row.get("json", {}))), max_answer_chars)
        if len(prompt) < 8 or parsed is None:
            return None
        schema = row.get("schema") if isinstance(row.get("schema"), dict) else row.get("expected_schema", {}) if isinstance(row.get("expected_schema"), dict) else {}
        return {
            "prompt": prompt,
            "answer": answer_text,
            "parsed_answer": parsed,
            "schema": schema,
            "aux_score": _generic_rule_aux_score("json_schema_rule_based", ["valid_json", "required_keys_present", "no_markdown_fence", "compact_values"]),
            **common,
        }

    if category == "self_correction_aux_score":
        prompt = trunc(row.get("prompt", row.get("task", "")), max_prompt_chars)
        draft = trunc(row.get("draft_answer", row.get("draft", row.get("bad_answer", ""))), max_answer_chars)
        answer = trunc(row.get("answer", row.get("corrected", row.get("assistant_response", ""))), max_answer_chars)
        if min(len(prompt), len(draft), len(answer)) < 8 or draft == answer:
            return None
        return {
            "prompt": prompt,
            "draft_answer": draft,
            "answer": answer,
            "aux_score": _generic_rule_aux_score("self_correction_rule_based", ["identifies_main_issue", "fixes_without_overexplaining", "does_not_introduce_new_error"]),
            **common,
        }

    if category == "sidecar_latent_alignment":
        input_text = trunc(row.get("input_text", row.get("prompt", row.get("task", ""))), max_prompt_chars)
        target_summary = trunc(row.get("target_summary", row.get("summary", row.get("answer", ""))), max_answer_chars)
        if len(input_text) < 8 or len(target_summary) < 20:
            return None
        metrics = row.get("need_metrics") if isinstance(row.get("need_metrics"), dict) else {}
        norm_metrics = {
            "quality": round(_clamp_number(metrics.get("quality", row.get("quality", 0.75)), 0.0, 1.0, 0.75), 3),
            "risk": round(_clamp_number(metrics.get("risk", row.get("risk", 0.25)), 0.0, 1.0, 0.25), 3),
            "contradiction": round(_clamp_number(metrics.get("contradiction", row.get("contradiction", 0.10)), 0.0, 1.0, 0.10), 3),
        }
        return {
            "input_text": input_text,
            "target_summary": target_summary,
            "need_metrics": norm_metrics,
            "training_objectives": ["summary_lm", "latent_projection", "contrastive_alignment"],
            **common,
        }
    if category in {"helpfulness_preferences", "harmlessness_safety"}:
        prompt = trunc(row.get("prompt"), max_prompt_chars)
        chosen = trunc(row.get("chosen"), max_answer_chars)
        rejected = trunc(row.get("rejected"), max_answer_chars)
        if min(len(prompt), len(chosen), len(rejected)) < 8 or chosen == rejected:
            return None
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected, **common}
    return None


def row_identity(row: Dict[str, Any]) -> str:
    keys = [str(row.get("category", "")), str(row.get("topic", "")), str(row.get("prompt", "")), str(row.get("text", ""))]
    if "messages" in row and isinstance(row["messages"], list):
        keys.extend(str(m.get("content", "")) for m in row["messages"] if isinstance(m, dict))
    tx_for_id = row.get("overheard_transcription", row.get("sidecar_transcription"))
    if isinstance(tx_for_id, dict):
        keys.append(str(tx_for_id.get("transcript", "")))
    return _stable_hash("\n".join(keys).lower())


def append_rows(out_dir: Path, category: str, rows: Sequence[Dict[str, Any]]) -> Path:
    path = out_dir / TARGET_FILES[category]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def prompt_text_for_row(row: Dict[str, Any]) -> str:
    if "prompt" in row:
        return _clean_spaces(row.get("prompt"))
    if "messages" in row and isinstance(row["messages"], list):
        for m in row["messages"]:
            if isinstance(m, dict) and m.get("role") == "user":
                return _clean_spaces(m.get("content"))
    if "input_text" in row:
        return _clean_spaces(row.get("input_text"))[:240]
    return _clean_spaces(row.get("text"))[:240]


def answer_text_for_row(row: Dict[str, Any]) -> str:
    if "chosen" in row:
        return _clean_spaces(row.get("chosen"))
    if "answer" in row:
        return _clean_spaces(row.get("answer"))
    if "target_summary" in row:
        return _clean_spaces(row.get("target_summary"))[:600]
    if "assistant_response" in row and isinstance(row["assistant_response"], dict):
        ar = row["assistant_response"]
        for key in ("answer", "summary", "explanation", "recommended_action"):
            if key in ar:
                return _clean_spaces(ar.get(key))[:600]
        return _clean_spaces(ar)[:600]
    if "messages" in row and isinstance(row["messages"], list):
        for m in reversed(row["messages"]):
            if isinstance(m, dict) and m.get("role") == "assistant":
                return _clean_spaces(m.get("content"))
    return _clean_spaces(row.get("text"))[:600]


def rejected_text_for_row(row: Dict[str, Any]) -> str:
    return _clean_spaces(row.get("rejected", ""))


def control_interaction_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    category = str(row.get("category", ""))
    score = 0.78
    risk = 0.18
    contradiction = 0.08
    controller = "answer"
    output_mode = "short"
    if category == "reasoning_rlvr":
        score = 0.82
        risk = 0.22
        controller = "deepen"
        output_mode = "summary"
    elif category == "numeric_evaluation":
        score = 0.84
        risk = 0.26
        contradiction = 0.12
        controller = "deepen"
        output_mode = "summary"
    elif category == "risk_scoring":
        score = 0.83
        risk = 0.42
        contradiction = 0.14
        controller = "revise"
        output_mode = "summary"
    elif category == "weighted_decision":
        score = 0.85
        risk = 0.24
        contradiction = 0.10
        controller = "deepen"
        output_mode = "summary"
    elif category == "overheard_transcription_acceptance":
        score = 0.86
        risk = 0.20
        contradiction = 0.10
        controller = "answer"
        output_mode = "summary"
    elif category == "overheard_transcription_preferences":
        score = 0.87
        risk = 0.22
        contradiction = 0.12
        controller = "answer"
        output_mode = "short"
    elif category == "overheard_transcription_rlvr":
        score = 0.84
        risk = 0.26
        contradiction = 0.16
        controller = "deepen"
        output_mode = "summary"
    elif category == "sidecar_latent_alignment":
        score = 0.88
        risk = 0.18
        contradiction = 0.08
        controller = "deepen"
        output_mode = "summary"
    elif category in {"image_prompt_fidelity", "image_generation_preferences"}:
        score = 0.86
        risk = 0.18
        contradiction = 0.10
        controller = "answer"
        output_mode = "short"
    elif category == "image_edit_instruction_following":
        score = 0.85
        risk = 0.20
        contradiction = 0.12
        controller = "answer"
        output_mode = "summary"
    elif category == "image_safety_boundaries":
        score = 0.84
        risk = 0.42
        contradiction = 0.12
        controller = "revise"
        output_mode = "short"
    elif category == "image_composition_scoring":
        score = 0.84
        risk = 0.24
        contradiction = 0.12
        controller = "deepen"
        output_mode = "summary"
    elif category == "tool_routing":
        score = 0.85
        risk = 0.24
        contradiction = 0.10
        controller = "answer"
        output_mode = "short"
    elif category in {"latent_tool_calculator", "latent_tool_python", "latent_tool_preferences", "code_execution_behavior"}:
        score = 0.88
        risk = 0.20
        contradiction = 0.08
        controller = "deepen" if category != "latent_tool_preferences" else "answer"
        output_mode = "summary"
    elif category == "structured_json":
        score = 0.86
        risk = 0.16
        contradiction = 0.08
        controller = "answer"
        output_mode = "none"
    elif category == "behavioral_memory_policy":
        score = 0.84
        risk = 0.30
        contradiction = 0.14
        controller = "answer"
        output_mode = "summary"
    elif category == "self_correction_aux_score":
        score = 0.84
        risk = 0.26
        contradiction = 0.18
        controller = "revise"
        output_mode = "summary"
    elif category == "helpfulness_preferences":
        score = 0.86
        risk = 0.16
        output_mode = "short"
    elif category == "harmlessness_safety":
        score = 0.84
        risk = 0.45
        controller = "revise"
        output_mode = "short"
    elif category == "honesty_uncertainty":
        score = 0.80
        risk = 0.28
        contradiction = 0.18
        controller = "deepen"
        output_mode = "summary"
    elif category == "friendly_concise_style":
        score = 0.83
        risk = 0.12
        output_mode = "none"
    out = {
        "prompt": prompt_text_for_row(row),
        "answer": answer_text_for_row(row),
        "rejected": rejected_text_for_row(row),
        "score": score,
        "risk": risk,
        "contradiction": contradiction,
        "controller_action": controller,
        "output_mode": output_mode,
        "weight": 0.5 + score,
        "source": row.get("source", "openai_low_data_rl_synthetic"),
        "category": category,
        "topic": row.get("topic", ""),
    }
    if category in IMAGE_BEHAVIOR_CATEGORIES:
        out["image_generation_behavior"] = True
        out["no_image_generated"] = bool(row.get("no_image_generated", True))
        if isinstance(row.get("reward_axes"), dict):
            out["reward_axes"] = row.get("reward_axes")
    if category == "tool_routing" and row.get("routing_decision"):
        out["routing_decision"] = row.get("routing_decision")
    if category in {"latent_tool_calculator", "latent_tool_python", "latent_tool_preferences", "code_execution_behavior"}:
        out["latent_tool_policy"] = "runtime_built_hidden_observations_no_model_generated_calls_llrl_not_required"
        out["tool_use_latent_only"] = True
        out["model_built_tool_calls"] = False
        out["llrl_required_for_tools"] = False
    if category == "behavioral_memory_policy":
        out["memory_policy"] = "behavioral_not_factual"
    if category == "structured_json":
        out["structured_output_policy"] = "strict_json_no_fence"
    tx = row.get("overheard_transcription", row.get("sidecar_transcription"))
    if isinstance(tx, dict):
        out["provided_transcription_confidence"] = tx.get("confidence", 0.0)
        out["provided_transcription_source"] = tx.get("source", "provided_transcript")
        out["provided_transcription_policy"] = row.get("acceptance_policy", normalize_acceptance_policy("", tx))
    return out


def write_control_interactions(path: Path, rows: Sequence[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(control_interaction_from_row(row), ensure_ascii=False) + "\n")
    return len(rows)


def write_prompt_audit(out_dir: Path, batch_id: int, category: str, messages: Sequence[Dict[str, str]]) -> None:
    path = out_dir / "rl" / "synthetic_batch_prompts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"batch": batch_id, "category": category, "messages": list(messages)}, ensure_ascii=False) + "\n")


def generate_batch(args: argparse.Namespace, category: str, n: int, seeds: Sequence[TopicSeed], recent_topics: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    module = PROMPT_MODULES[category]
    messages = batch_prompt(module, n, seeds, recent_topics)
    out_dir = Path(args.out_dir)
    if args.write_prompts:
        write_prompt_audit(out_dir, args._batch_id, category, messages)
    if not args.activate:
        return [], {"dry_run": True, "requested": n, "category": category}
    api_key = os.environ.get(args.api_key_env, "") or args.api_key
    if not api_key:
        raise ValueError(f"No API key found. Set {args.api_key_env} or pass --api_key.")
    raw_text = openai_request(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        api_format=args.api_format,
        timeout_s=args.api_timeout_s,
    )
    try:
        payload = extract_json_payload(raw_text)
    except Exception as exc:
        if not args.repair_failed_json:
            raise
        repair = repair_prompt(module, n, raw_text, str(exc))
        if args.write_prompts:
            write_prompt_audit(out_dir, args._batch_id, category + "_repair", repair)
        raw_text = openai_request(
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            messages=repair,
            temperature=0.2,
            max_output_tokens=args.max_output_tokens,
            api_format=args.api_format,
            timeout_s=args.api_timeout_s,
        )
        payload = extract_json_payload(raw_text)
    if isinstance(payload, dict):
        items = payload.get("items", payload.get("data", []))
    else:
        items = payload
    if not isinstance(items, list):
        raise ValueError("generation payload must be a JSON array or object with items/data")
    rows: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        row = validate_row(category, item, args.max_prompt_chars, args.max_answer_chars)
        if row is None:
            continue
        rid = row_identity(row)
        if rid in seen:
            continue
        seen.add(rid)
        rows.append(row)
        if len(rows) >= n:
            break
    meta = {"requested": n, "received": len(items), "kept": len(rows), "category": category}
    return rows, meta


def _read_sidecar_transcript_events(path: Path, max_items: int = 0) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    rows: List[Dict[str, Any]] = []
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            data = obj.get("items", obj.get("events", obj.get("transcripts", [])))
        else:
            data = obj
        if not isinstance(data, list):
            raise ValueError("transcript JSON must be a list or object containing items/events/transcripts")
        for item in data:
            if isinstance(item, dict):
                rows.append(item)
                if max_items and len(rows) >= max_items:
                    break
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
                if max_items and len(rows) >= max_items:
                    break
    return rows


def _default_sidecar_task(event: Dict[str, Any], tx: Dict[str, Any]) -> str:
    task = _clean_spaces(event.get("task", event.get("prompt", event.get("instruction", ""))) )
    if task:
        return task[:500]
    transcript = _clean_spaces(tx.get("transcript"))
    if "?" in transcript[:260]:
        return "Use the provided transcript from the overheard scenario to answer the question or extract action items."
    return "Use the provided overheard-scenario transcript to give a concise grounded summary and any clear action items."


def _default_sidecar_answer(event: Dict[str, Any], tx: Dict[str, Any], task: str) -> str:
    for key in ("answer", "assistant_response", "chosen", "summary", "expected_answer", "target"):
        val = event.get(key)
        if isinstance(val, dict):
            for subkey in ("answer", "summary", "explanation", "text"):
                if _clean_spaces(val.get(subkey, "")):
                    return _clean_spaces(val.get(subkey))[:1200]
        elif _clean_spaces(val):
            return _clean_spaces(val)[:1200]
    transcript = _clean_spaces(tx.get("transcript"))
    if not transcript:
        return "The provided transcript is empty, so I would ask for clearer transcript text before drawing conclusions."
    words = transcript.split()
    brief = " ".join(words[:80])
    conf = _clamp_number(tx.get("confidence"), 0.0, 1.0, 0.85)
    prefix = "Based on the provided transcript"
    if conf < 0.72:
        prefix += " and treating low-confidence parts cautiously"
    return f"{prefix}, the relevant content is: {brief}"[:1200]


def _sidecar_user_content(task: str, tx: Dict[str, Any]) -> str:
    payload = {
        "task": task,
        "overheard_transcription": tx,
        "instruction": "Use the provided transcript text from this overheard scenario. Do not claim you personally heard the audio or conversation.",
    }
    return json.dumps(payload, ensure_ascii=False)


def sidecar_event_to_rows(event: Dict[str, Any], idx: int, emit_as: str = "all", max_prompt_chars: int = 900, max_answer_chars: int = 1400) -> List[Tuple[str, Dict[str, Any]]]:
    tx = normalize_sidecar_transcription(event.get("overheard_transcription", event.get("sidecar_transcription", event)), max_prompt_chars)
    if not tx.get("transcript"):
        return []
    topic = _clean_spaces(event.get("topic", event.get("title", event.get("audio_id", event.get("media_id", f"overheard_transcript_{idx:06d}")))))[:120]
    task = _default_sidecar_task(event, tx)
    answer = _default_sidecar_answer(event, tx, task)
    source = _clean_spaces(event.get("source", tx.get("source", "provided_transcript_ingest")))[:80]
    common = {"topic": topic, "seed_article": topic, "source": source}
    rows: List[Tuple[str, Dict[str, Any]]] = []
    if emit_as in {"sft", "all"}:
        sft = {
            **common,
            "overheard_transcription": tx,
            "acceptance_policy": normalize_acceptance_policy(event.get("acceptance_policy", ""), tx),
            "messages": [
                {"role": "system", "content": "Use the provided transcript from the overheard scenario as text evidence. Do not claim you personally heard audio; mention uncertainty only when the transcript is low-confidence or ambiguous."},
                {"role": "user", "content": _sidecar_user_content(task, tx)},
                {"role": "assistant", "content": answer},
            ],
        }
        row = validate_row("overheard_transcription_acceptance", sft, max_prompt_chars, max_answer_chars)
        if row:
            rows.append(("overheard_transcription_acceptance", row))
    if emit_as in {"rlvr", "all"}:
        rlvr = {
            **common,
            "prompt": _sidecar_user_content(task, tx),
            "overheard_transcription": tx,
            "answer": answer,
            "aux_score": sidecar_transcription_aux_score(tx, answer, event.get("aux_score")),
        }
        row = validate_row("overheard_transcription_rlvr", rlvr, max_prompt_chars, max_answer_chars)
        if row:
            rows.append(("overheard_transcription_rlvr", row))
    if emit_as in {"preference", "all"}:
        rejected = _clean_spaces(event.get("rejected", event.get("bad_answer", "")))
        if not rejected:
            if _clamp_number(tx.get("confidence"), 0.0, 1.0, 0.85) >= 0.72:
                rejected = "I cannot answer because I would need to hear the conversation myself."
            else:
                rejected = "The speakers definitely said several specific things that are not shown in the transcript."
        pref = {
            **common,
            "overheard_transcription": tx,
            "prompt": _sidecar_user_content(task, tx),
            "chosen": answer,
            "rejected": rejected,
            "preference_reason": "The chosen answer uses the provided transcript as evidence without claiming firsthand hearing or inventing unsupported details.",
        }
        row = validate_row("overheard_transcription_preferences", pref, max_prompt_chars, max_answer_chars)
        if row:
            rows.append(("overheard_transcription_preferences", row))
    return rows


def ingest_sidecar_transcript_file(args: argparse.Namespace) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    path = Path(args.sidecar_transcripts_file)
    events = _read_sidecar_transcript_events(path, int(args.sidecar_transcripts_max))
    by_category: Dict[str, List[Dict[str, Any]]] = {
        "overheard_transcription_acceptance": [],
        "overheard_transcription_preferences": [],
        "overheard_transcription_rlvr": [],
    }
    seen = set()
    skipped = 0
    for idx, event in enumerate(events):
        rows_for_event = sidecar_event_to_rows(event, idx, args.sidecar_transcripts_emit_as, args.max_prompt_chars, args.max_answer_chars)
        if not rows_for_event:
            skipped += 1
            continue
        for cat, row in rows_for_event:
            rid = row_identity(row)
            if rid in seen:
                continue
            seen.add(rid)
            by_category[cat].append(row)
    by_category = {k: v for k, v in by_category.items() if v}
    stats = {
        "file": str(path),
        "events_read": len(events),
        "events_skipped_no_transcript": skipped,
        "emit_as": args.sidecar_transcripts_emit_as,
        "rows_by_category": {k: len(v) for k, v in by_category.items()},
        "total_rows": sum(len(v) for v in by_category.values()),
    }
    return by_category, stats


def choose_topic_batch(topics: Sequence[TopicSeed], rng: random.Random, batch_size: int, offset: int, articles_per_batch: int) -> List[TopicSeed]:
    if articles_per_batch <= 0:
        articles_per_batch = batch_size
    if not topics:
        return [TopicSeed(title=x, source="fallback") for x in rng.sample(FALLBACK_TOPICS, min(len(FALLBACK_TOPICS), max(1, articles_per_batch)))]
    k = max(1, min(len(topics), int(articles_per_batch)))
    rotated = [topics[(offset + i) % len(topics)] for i in range(k)]
    if len(rotated) < k:
        rotated.extend(rng.sample(list(topics), k - len(rotated)))
    rng.shuffle(rotated)
    return rotated


def run(args: argparse.Namespace) -> Dict[str, Any]:
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    topics = build_topic_pool(args, rng)
    plan = plan_with_model(args, topics)
    counts = allocate_counts(plan["total_examples"], plan["mix"])
    batches = categories_for_batches(counts, plan["batch_size"], rng)
    manifest: Dict[str, Any] = {
        "format": "need_openai_low_data_rl_synthetic_manifest",
        "activate": bool(args.activate),
        "model": args.model,
        "api_format": args.api_format,
        "out_dir": str(out_dir),
        "plan": plan,
        "allocated_counts": counts,
        "target_files": TARGET_FILES,
        "topic_pool_count": len(topics),
        "topic_sources": sorted({t.source for t in topics}),
        "batches": [],
        "kept_counts": {k: 0 for k in PROMPT_MODULES},
        "control_interactions_file": args.control_interactions_file or str(out_dir / "rl" / "low_data_control_interactions.synthetic.jsonl"),
    }
    (out_dir / "rl").mkdir(parents=True, exist_ok=True)
    (out_dir / "rl" / "synthetic_topic_pool.json").write_text(json.dumps([t.compact(500) for t in topics], ensure_ascii=False, indent=2), encoding="utf-8")
    if args.sidecar_transcripts_file:
        sidecar_rows_by_category, sidecar_stats = ingest_sidecar_transcript_file(args)
        sidecar_control_rows: List[Dict[str, Any]] = []
        for sidecar_cat, sidecar_rows in sidecar_rows_by_category.items():
            append_rows(out_dir, sidecar_cat, sidecar_rows)
            sidecar_control_rows.extend(sidecar_rows)
            manifest["kept_counts"][sidecar_cat] = int(manifest["kept_counts"].get(sidecar_cat, 0)) + len(sidecar_rows)
        if args.emit_control_interactions and sidecar_control_rows:
            control_path = Path(args.control_interactions_file or str(out_dir / "rl" / "low_data_control_interactions.synthetic.jsonl"))
            sidecar_stats["control_interactions_written"] = write_control_interactions(control_path, sidecar_control_rows)
        manifest["transcription_ingest"] = sidecar_stats
    (out_dir / "openai_low_data_rl_plan.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.activate:
        # In dry-run mode, write one audit prompt per planned batch so users can inspect or manually submit them.
        recent_topics: List[str] = []
        for batch_id, cat in enumerate(batches):
            remaining = counts[cat] - manifest["kept_counts"].get(cat, 0)
            n = max(1, min(plan["batch_size"], remaining))
            seeds = choose_topic_batch(topics, rng, n, batch_id * args.articles_per_batch, args.articles_per_batch)
            args._batch_id = batch_id
            messages = batch_prompt(PROMPT_MODULES[cat], n, seeds, recent_topics)
            if args.write_prompts:
                write_prompt_audit(out_dir, batch_id, cat, messages)
            recent_topics.extend(s.title for s in seeds)
            manifest["batches"].append({"batch": batch_id, "category": cat, "requested": n, "dry_run": True, "seed_titles": [s.title for s in seeds]})
        (out_dir / "openai_low_data_rl_plan.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return manifest

    recent_topics = []
    all_rows_for_controls: List[Dict[str, Any]] = []
    for batch_id, cat in enumerate(batches):
        current_kept = int(manifest["kept_counts"].get(cat, 0))
        remaining = max(0, counts[cat] - current_kept)
        if remaining <= 0:
            continue
        requested = max(1, min(plan["batch_size"], remaining))
        seeds = choose_topic_batch(topics, rng, requested, batch_id * args.articles_per_batch, args.articles_per_batch)
        args._batch_id = batch_id
        rows, meta = generate_batch(args, cat, requested, seeds, recent_topics)
        if rows:
            append_rows(out_dir, cat, rows)
            all_rows_for_controls.extend(rows)
            manifest["kept_counts"][cat] = int(manifest["kept_counts"].get(cat, 0)) + len(rows)
        recent_topics.extend(s.title for s in seeds)
        batch_record = {**meta, "batch": batch_id, "seed_titles": [s.title for s in seeds]}
        manifest["batches"].append(batch_record)
        print(json.dumps({"batch": batch_id, "category": cat, "requested": requested, "kept": len(rows), "kept_counts": manifest["kept_counts"]}), flush=True)
        if args.sleep_s > 0:
            time.sleep(args.sleep_s)
    if args.emit_control_interactions and all_rows_for_controls:
        control_path = Path(args.control_interactions_file or str(out_dir / "rl" / "low_data_control_interactions.synthetic.jsonl"))
        manifest["control_interactions_written"] = write_control_interactions(control_path, all_rows_for_controls)
    manifest["done"] = True
    manifest["total_kept"] = int(sum(int(v) for v in manifest["kept_counts"].values()))
    (out_dir / "openai_low_data_rl_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Automated OpenAI-compatible low-data RL synthetic corpus builder")
    p.add_argument("--activate", action="store_true", help="Actually call the configured model. Without this, only writes a plan and prompts.")
    p.add_argument("--model", default="gpt-5.4", help="OpenAI-compatible model name used for planning/generation.")
    p.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    p.add_argument("--api_key_env", default="OPENAI_API_KEY")
    p.add_argument("--api_key", default="", help="API key literal. Prefer --api_key_env for normal use.")
    p.add_argument("--api_format", choices=["chat", "responses"], default="chat")
    p.add_argument("--api_timeout_s", type=float, default=120.0)
    p.add_argument("--out_dir", default="data/corpuses")
    p.add_argument("--target_profile", choices=sorted(PROFILE_HINTS.keys()), default="whole_model")
    p.add_argument("--total_examples", default="auto", help="Integer count, or auto to let the planner/profile choose.")
    p.add_argument("--batch_size", default="auto", help="Integer batch size, or auto to let the planner/profile choose.")
    p.add_argument("--min_total_examples", type=int, default=1)
    p.add_argument("--max_total_examples", type=int, default=3000)
    p.add_argument("--min_batch_size", type=int, default=4)
    p.add_argument("--max_batch_size", type=int, default=80)
    p.add_argument("--mix", default="", help="JSON object or category=value comma list overriding the planner/category distribution.")
    p.add_argument("--auto_plan", action=argparse.BooleanOptionalAction, default=True, help="Let the model choose total count and mix when activated.")
    p.add_argument("--topics_file", default="", help="Optional txt/json/jsonl topic list. JSON rows may include title/topic, extract, url.")
    p.add_argument("--wikipedia_source", default="popular,random,vital", help="Comma list: popular, random, vital, cited, none. cited uses vital topics as a stable proxy unless --topics_file is provided.")
    p.add_argument("--wikipedia_days_back", type=int, default=2)
    p.add_argument("--wikipedia_timeout_s", type=float, default=20.0)
    p.add_argument("--topic_pool_size", type=int, default=160)
    p.add_argument("--articles_per_batch", type=int, default=0, help="Number of article/topic seeds per generation batch. 0 means one seed per requested datapoint.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_output_tokens", type=int, default=6000)
    p.add_argument("--max_prompt_chars", type=int, default=900)
    p.add_argument("--max_answer_chars", type=int, default=1400)
    p.add_argument("--repair_failed_json", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--write_prompts", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--emit_control_interactions", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--control_interactions_file", default="")
    p.add_argument("--transcripts_file", "--sidecar_transcripts_file", dest="sidecar_transcripts_file", default="", help="Optional JSON/JSONL transcript events to convert into natural overheard-scenario SFT/RLVR/preference rows.")
    p.add_argument("--transcripts_max", "--sidecar_transcripts_max", dest="sidecar_transcripts_max", type=int, default=0, help="Maximum transcript events to ingest. 0 means all.")
    p.add_argument("--transcripts_emit_as", "--sidecar_transcripts_emit_as", dest="sidecar_transcripts_emit_as", choices=["sft", "rlvr", "preference", "all"], default="all", help="Which row types to create from transcript events.")
    p.add_argument("--sleep_s", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=123)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
