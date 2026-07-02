#!/usr/bin/env python3
"""
Build corpuses.py

Standalone corpus builder for NEED runs.

Default target:
  - Total selected corpus plan: about 150B token-equivalent.
  - Knowledge / pretraining corpus: about 141B token-equivalent.
  - General post-training / RL corpus: about 9B token-equivalent.

This script does NOT include low-level adapter/RL traces such as aux_score-head,
image-RL, controller, output_mode-router, speculative-acceptance, or latent
replay calibration data. Those should stay separate.

What it builds:
  out_dir/
    plan.json
    knowledge/train.jsonl
    knowledge/manifest.json
    rl/sft.jsonl
    rl/preferences.jsonl
    rl/rlvr.jsonl
    rl/manifest.json

Install optional dependencies when building from Hugging Face:
  pip install datasets transformers tqdm

Examples:
  python build_corpuses.py --write-plan --out_dir data/corpuses
  python build_corpuses.py --build knowledge --out_dir data/corpuses --tokenizer gpt2
  python build_corpuses.py --build rl --out_dir data/corpuses --tokenizer gpt2
  python build_corpuses.py --build all --out_dir data/corpuses --tokenizer path/to/tokenizer

For an initial smoke test:
  python build_corpuses.py --build all --out_dir data/corpuses_smoke --scale 0.000001 --approx_tokens

License note:
  The script only points at configurable public dataset identifiers or local
  files. You are responsible for verifying license, provenance, and permitted
  use for your training run before downloading or training on any dataset.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


# -----------------------------
# Token budgets
# -----------------------------

TOTAL_CORPUS_TARGET_TOKENS = 150_000_000_000
KNOWLEDGE_TARGET_TOKENS = 141_000_000_000
RL_TARGET_TOKENS = 9_000_000_000
BASE_CORPUS_PARAMS_M = 600.0

COUNT_RE = re.compile(r"^\s*([0-9]+(?:_[0-9]{3})*(?:\.[0-9]+)?|[0-9]*\.[0-9]+)\s*([kKmMbBtT]?)\s*$")
COUNT_MULTIPLIERS = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}


def parse_scaled_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(round(value))
    text = str(value).strip().replace(",", "").replace("_", "")
    if not text:
        return 0
    m = COUNT_RE.match(text)
    if not m:
        raise argparse.ArgumentTypeError(f"expected a number with optional K/M/B/T suffix, got {value!r}")
    return int(round(float(m.group(1)) * COUNT_MULTIPLIERS[(m.group(2) or "").upper()]))


def format_scaled_count(n: int) -> str:
    n = int(n)
    for suffix, mult in (("T", 1_000_000_000_000), ("B", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)):
        if abs(n) >= mult and n % mult == 0:
            return f"{n // mult}{suffix}"
        if abs(n) >= mult:
            return f"{n / mult:.3g}{suffix}"
    return str(n)


def scale_mix_to_total(base: Dict[str, int], target_total: int) -> Dict[str, int]:
    base_total = sum(int(v) for v in base.values())
    if base_total <= 0:
        raise ValueError("base mix must contain positive token counts")
    out: Dict[str, int] = {}
    running = 0
    items = list(base.items())
    for i, (name, value) in enumerate(items):
        if i == len(items) - 1:
            scaled = int(target_total) - running
        else:
            scaled = int(round(int(value) * int(target_total) / base_total))
            running += scaled
        out[name] = max(1, scaled)
    return out


# Base ratios were the original 10.65B NEED corpus plan. They are now scaled to
# an approximately 150B-token corpus so --target_tokens can choose what fraction
# to actually download/build for a run.
BASE_KNOWLEDGE_MIX = {
    "edu_web_general": 2_800_000_000,
    "encyclopedic_reference": 800_000_000,
    "math_web": 900_000_000,
    "science_research": 750_000_000,
    "biomed_abstracts": 350_000_000,
    "books_public_domain": 650_000_000,
    "textbooks_open": 550_000_000,
    "news_current_events_style": 350_000_000,
    "code_docs_technical": 500_000_000,
    "code_python_general": 900_000_000,
    "code_algorithms_math": 500_000_000,
    "code_execution_traces": 300_000_000,
    "qa_knowledge_shortform": 450_000_000,
    "forums_explanatory": 200_000_000,
}

BASE_RL_MIX = {
    "reasoning_step_by_step_rlvr": 145_000_000,
    "instruction_following": 95_000_000,
    "helpfulness_preferences": 70_000_000,
    "harmlessness_safety": 60_000_000,
    "instruction_constraints_if": 45_000_000,
    "honesty_uncertainty": 20_000_000,
    "friendly_concise_style": 10_000_000,
    "general_alignment_sft": 35_000_000,
    "general_alignment_preferences": 30_000_000,
    "truthfulness_calibration": 10_000_000,
    "code_instruction_sft": 55_000_000,
    "calculator_tool_rlvr": 25_000_000,
    "python_tool_execution": 25_000_000,
    "runtime_latent_tool_preferences": 25_000_000,
}

KNOWLEDGE_MIX = scale_mix_to_total(BASE_KNOWLEDGE_MIX, KNOWLEDGE_TARGET_TOKENS)
RL_MIX = scale_mix_to_total(BASE_RL_MIX, RL_TARGET_TOKENS)


# -----------------------------
# Dataset plan dataclasses
# -----------------------------

@dataclass
class SourceCandidate:
    kind: str                         # "hf" or "local"
    path: str                         # HF dataset id or local glob
    config: Optional[str] = None
    split: str = "train"
    text_fields: List[str] = field(default_factory=lambda: ["text"])
    format: str = "text"             # text | sft | preference | rlvr
    priority: int = 0
    enabled: bool = True
    note: str = ""


@dataclass
class CorpusSlice:
    name: str
    family: str                       # knowledge | rl
    category: str
    target_tokens: int
    output_file: str
    format: str                       # text | sft | preference | rlvr
    candidates: List[SourceCandidate]
    min_chars: int = 220
    max_chars: int = 24_000
    max_doc_tokens: int = 2_048
    dedup_prefix_chars: int = 4_000
    quality_profile: str = "default"


# -----------------------------
# Default corpus source plan
# -----------------------------


def default_plan() -> List[CorpusSlice]:
    """Return the default corpus plan.

    Dataset IDs are deliberately editable. If one source is unavailable or has a
    license you do not want, the builder skips to the next candidate or lets you
    replace it with a local glob through --local_override_json.
    """

    k = []
    k.append(CorpusSlice(
        name="edu_web_general",
        family="knowledge",
        category="diverse educational web",
        target_tokens=KNOWLEDGE_MIX["edu_web_general"],
        output_file="knowledge/train.jsonl",
        format="text",
        candidates=[
            SourceCandidate("hf", "HuggingFaceFW/fineweb-edu", None, "train", ["text"], "text", 0,
                            note="High-quality educational web. Prefer sample configs for small runs."),
            SourceCandidate("hf", "HuggingFaceFW/fineweb-edu-score-2", None, "train", ["text"], "text", 1),
            SourceCandidate("local", "data/raw/edu_web/**/*.jsonl", None, "train", ["text"], "text", 2),
        ],
        max_doc_tokens=2_048,
    ))
    k.append(CorpusSlice(
        name="encyclopedic_reference",
        family="knowledge",
        category="Wikipedia and reference",
        target_tokens=KNOWLEDGE_MIX["encyclopedic_reference"],
        output_file="knowledge/train.jsonl",
        format="text",
        candidates=[
            SourceCandidate("hf", "wikimedia/wikipedia", "20231101.en", "train", ["text"], "text", 0),
            SourceCandidate("local", "data/raw/wiki/**/*.jsonl", None, "train", ["text"], "text", 1),
        ],
        min_chars=180,
        max_doc_tokens=1_536,
    ))
    k.append(CorpusSlice(
        name="math_web",
        family="knowledge",
        category="math explanations and notation-heavy web",
        target_tokens=KNOWLEDGE_MIX["math_web"],
        output_file="knowledge/train.jsonl",
        format="text",
        candidates=[
            SourceCandidate("hf", "open-web-math/open-web-math", None, "train", ["text"], "text", 0),
            SourceCandidate("local", "data/raw/math/**/*.jsonl", None, "train", ["text"], "text", 1),
        ],
        min_chars=160,
        max_doc_tokens=1_536,
        quality_profile="math",
    ))
    k.append(CorpusSlice(
        name="science_research",
        family="knowledge",
        category="science papers and research summaries",
        target_tokens=KNOWLEDGE_MIX["science_research"],
        output_file="knowledge/train.jsonl",
        format="text",
        candidates=[
            SourceCandidate("hf", "scientific_papers", "arxiv", "train", ["article", "abstract", "text"], "text", 0),
            SourceCandidate("hf", "armanc/scientific_papers", "arxiv", "train", ["article", "abstract", "text"], "text", 1),
            SourceCandidate("local", "data/raw/science/**/*.jsonl", None, "train", ["text", "article", "abstract"], "text", 2),
        ],
        min_chars=400,
        max_chars=60_000,
        max_doc_tokens=2_048,
        quality_profile="science",
    ))
    k.append(CorpusSlice(
        name="biomed_abstracts",
        family="knowledge",
        category="biomedical papers and abstracts",
        target_tokens=KNOWLEDGE_MIX["biomed_abstracts"],
        output_file="knowledge/train.jsonl",
        format="text",
        candidates=[
            SourceCandidate("hf", "scientific_papers", "pubmed", "train", ["article", "abstract", "text"], "text", 0),
            SourceCandidate("hf", "armanc/scientific_papers", "pubmed", "train", ["article", "abstract", "text"], "text", 1),
            SourceCandidate("local", "data/raw/biomed/**/*.jsonl", None, "train", ["text", "article", "abstract"], "text", 2),
        ],
        min_chars=250,
        max_doc_tokens=1_536,
        quality_profile="science",
    ))
    k.append(CorpusSlice(
        name="books_public_domain",
        family="knowledge",
        category="public-domain books and long-form prose",
        target_tokens=KNOWLEDGE_MIX["books_public_domain"],
        output_file="knowledge/train.jsonl",
        format="text",
        candidates=[
            SourceCandidate("hf", "manu/project_gutenberg", None, "train", ["text", "content"], "text", 0),
            SourceCandidate("hf", "Navanjana/Gutenberg_books", None, "train", ["text", "paragraph", "content"], "text", 1),
            SourceCandidate("local", "data/raw/books_public_domain/**/*.jsonl", None, "train", ["text"], "text", 2),
        ],
        min_chars=300,
        max_chars=40_000,
        max_doc_tokens=2_048,
        quality_profile="book",
    ))
    k.append(CorpusSlice(
        name="textbooks_open",
        family="knowledge",
        category="open textbooks",
        target_tokens=KNOWLEDGE_MIX["textbooks_open"],
        output_file="knowledge/train.jsonl",
        format="text",
        candidates=[
            SourceCandidate("hf", "crumb/openstax-text", None, "train", ["text"], "text", 0),
            SourceCandidate("local", "data/raw/textbooks/**/*.jsonl", None, "train", ["text"], "text", 1),
        ],
        min_chars=180,
        max_doc_tokens=1_536,
        quality_profile="science",
    ))
    k.append(CorpusSlice(
        name="news_current_events_style",
        family="knowledge",
        category="news/current-events style text",
        target_tokens=KNOWLEDGE_MIX["news_current_events_style"],
        output_file="knowledge/train.jsonl",
        format="text",
        candidates=[
            SourceCandidate("hf", "vblagoje/cc_news", None, "train", ["text", "description", "title"], "text", 0,
                            note="Check license/provenance before use; replace with your approved news mirror if needed."),
            SourceCandidate("hf", "sentence-transformers/ccnews", None, "train", ["text", "title"], "text", 1,
                            note="Check license/provenance before use."),
            SourceCandidate("local", "data/raw/news_approved/**/*.jsonl", None, "train", ["text", "title", "body"], "text", 2),
        ],
        min_chars=180,
        max_doc_tokens=1_024,
    ))
    k.append(CorpusSlice(
        name="code_docs_technical",
        family="knowledge",
        category="code, documentation, and technical explanation",
        target_tokens=KNOWLEDGE_MIX["code_docs_technical"],
        output_file="knowledge/train.jsonl",
        format="text",
        candidates=[
            SourceCandidate("hf", "bigcode/the-stack-smol", None, "train", ["content", "text"], "text", 0,
                            note="Small permissive-code sample; replace with your approved code/doc mirror for scale."),
            SourceCandidate("local", "data/raw/code_docs/**/*.jsonl", None, "train", ["text", "content", "docstring"], "text", 1),
        ],
        min_chars=100,
        max_doc_tokens=1_536,
        quality_profile="code",
    ))
    k.append(CorpusSlice(
        name="code_python_general",
        family="knowledge",
        category="Python and general-purpose code",
        target_tokens=KNOWLEDGE_MIX["code_python_general"],
        output_file="knowledge/train.jsonl",
        format="text",
        candidates=[
            SourceCandidate("hf", "bigcode/the-stack-smol", "data/python", "train", ["content", "text", "code"], "text", 0,
                            note="Use only approved/licensed code subsets for your deployment."),
            SourceCandidate("hf", "codeparrot/github-code", "python", "train", ["code", "content", "text"], "text", 1,
                            note="Optional Python code source; verify license/provenance."),
            SourceCandidate("local", "data/raw/code_python/**/*.jsonl", None, "train", ["code", "content", "text", "docstring"], "text", 2),
        ],
        min_chars=60,
        max_doc_tokens=1_536,
        quality_profile="code",
    ))
    k.append(CorpusSlice(
        name="code_algorithms_math",
        family="knowledge",
        category="algorithms, programming problems, and math-code reasoning",
        target_tokens=KNOWLEDGE_MIX["code_algorithms_math"],
        output_file="knowledge/train.jsonl",
        format="text",
        candidates=[
            SourceCandidate("hf", "codeparrot/apps", None, "train", ["question", "solutions", "input_output", "starter_code"], "text", 0),
            SourceCandidate("hf", "deepmind/code_contests", None, "train", ["description", "solutions", "public_tests", "generated_tests"], "text", 1),
            SourceCandidate("local", "data/raw/code_algorithms/**/*.jsonl", None, "train", ["text", "prompt", "solution", "code", "tests"], "text", 2),
        ],
        min_chars=80,
        max_doc_tokens=1_536,
        quality_profile="code",
    ))
    k.append(CorpusSlice(
        name="code_execution_traces",
        family="knowledge",
        category="short code execution traces and tool-result grounding",
        target_tokens=KNOWLEDGE_MIX["code_execution_traces"],
        output_file="knowledge/train.jsonl",
        format="text",
        candidates=[
            SourceCandidate("local", "data/raw/code_execution_traces/**/*.jsonl", None, "train", ["text", "prompt", "code", "stdout", "result", "explanation"], "text", 0,
                            note="Recommended curated local traces from calculator/Python runs; keeps NEED familiar with computation-output language."),
            SourceCandidate("local", "data/raw/tool_traces/**/*.jsonl", None, "train", ["text", "prompt", "tool", "observation", "answer"], "text", 1),
        ],
        min_chars=40,
        max_doc_tokens=768,
        quality_profile="code",
    ))

    k.append(CorpusSlice(
        name="qa_knowledge_shortform",
        family="knowledge",
        category="question-answer knowledge shortform",
        target_tokens=KNOWLEDGE_MIX["qa_knowledge_shortform"],
        output_file="knowledge/train.jsonl",
        format="text",
        candidates=[
            SourceCandidate("hf", "sentence-transformers/natural-questions", None, "train", ["question", "answer", "text"], "text", 0),
            SourceCandidate("local", "data/raw/qa_knowledge/**/*.jsonl", None, "train", ["text", "question", "answer"], "text", 1),
        ],
        min_chars=80,
        max_doc_tokens=768,
    ))
    k.append(CorpusSlice(
        name="forums_explanatory",
        family="knowledge",
        category="explanatory forum/help content",
        target_tokens=KNOWLEDGE_MIX["forums_explanatory"],
        output_file="knowledge/train.jsonl",
        format="text",
        candidates=[
            SourceCandidate("local", "data/raw/forums_approved/**/*.jsonl", None, "train", ["text", "question", "answer"], "text", 0,
                            note="Use your own approved forum/help data. Disabled by source availability, not by plan."),
        ],
        min_chars=100,
        max_doc_tokens=1_024,
    ))

    r = []
    r.append(CorpusSlice(
        name="reasoning_step_by_step_rlvr",
        family="rl",
        category="reasoning RLVR / verifiable tasks",
        target_tokens=RL_MIX["reasoning_step_by_step_rlvr"],
        output_file="rl/rlvr.jsonl",
        format="rlvr",
        candidates=[
            SourceCandidate("hf", "allenai/RLVR-GSM-MATH-IF-Mixed-Constraints", None, "train", ["problem", "question", "solution", "answer"], "rlvr", 0),
            SourceCandidate("hf", "allenai/RLVR-MATH", None, "train", ["problem", "question", "solution", "answer"], "rlvr", 1),
            SourceCandidate("hf", "allenai/RLVR-GSM", None, "train", ["problem", "question", "solution", "answer"], "rlvr", 2),
            SourceCandidate("local", "data/rl/reasoning_rlvr/**/*.jsonl", None, "train", ["prompt", "answer", "solution"], "rlvr", 3),
        ],
        min_chars=20,
        max_doc_tokens=1_024,
    ))
    r.append(CorpusSlice(
        name="instruction_following",
        family="rl",
        category="SFT instruction following",
        target_tokens=RL_MIX["instruction_following"],
        output_file="rl/sft.jsonl",
        format="sft",
        candidates=[
            SourceCandidate("hf", "allenai/tulu-3-sft-personas-instruction-following", None, "train", ["messages", "prompt", "response"], "sft", 0),
            SourceCandidate("hf", "OpenAssistant/oasst1", None, "train", ["text", "message_tree_id", "parent_id"], "sft", 1),
            SourceCandidate("hf", "databricks/databricks-dolly-15k", None, "train", ["instruction", "context", "response"], "sft", 2),
            SourceCandidate("local", "data/rl/sft/**/*.jsonl", None, "train", ["messages", "prompt", "response"], "sft", 3),
        ],
        min_chars=20,
        max_doc_tokens=1_024,
    ))
    r.append(CorpusSlice(
        name="helpfulness_preferences",
        family="rl",
        category="helpfulness preference pairs",
        target_tokens=RL_MIX["helpfulness_preferences"],
        output_file="rl/preferences.jsonl",
        format="preference",
        candidates=[
            SourceCandidate("hf", "allenai/llama-3.1-tulu-3-405b-preference-mixture", None, "train", ["prompt", "chosen", "rejected", "messages"], "preference", 0),
            SourceCandidate("hf", "HuggingFaceH4/ultrafeedback_binarized", None, "train_prefs", ["prompt", "chosen", "rejected"], "preference", 1),
            SourceCandidate("local", "data/rl/helpfulness_prefs/**/*.jsonl", None, "train", ["prompt", "chosen", "rejected"], "preference", 2),
        ],
        min_chars=20,
        max_doc_tokens=1_024,
    ))
    r.append(CorpusSlice(
        name="harmlessness_safety",
        family="rl",
        category="harmlessness and safety preference/SFT",
        target_tokens=RL_MIX["harmlessness_safety"],
        output_file="rl/preferences.jsonl",
        format="preference",
        candidates=[
            SourceCandidate("hf", "allenai/wildguardmix", None, "train", ["prompt", "chosen", "rejected", "messages", "response"], "preference", 0),
            SourceCandidate("hf", "Anthropic/hh-rlhf", "harmless-base", "train", ["chosen", "rejected"], "preference", 1,
                            note="Use only if license and policy fit your deployment."),
            SourceCandidate("local", "data/rl/safety/**/*.jsonl", None, "train", ["prompt", "chosen", "rejected"], "preference", 2),
        ],
        min_chars=20,
        max_doc_tokens=1_024,
    ))
    r.append(CorpusSlice(
        name="instruction_constraints_if",
        family="rl",
        category="instruction-following constraints and IFEval-like tasks",
        target_tokens=RL_MIX["instruction_constraints_if"],
        output_file="rl/rlvr.jsonl",
        format="rlvr",
        candidates=[
            SourceCandidate("hf", "allenai/RLVR-IFeval", None, "train", ["prompt", "instruction", "answer", "constraints"], "rlvr", 0),
            SourceCandidate("hf", "allenai/tulu-3-IF-augmented-on-policy-8b", None, "train", ["messages", "prompt", "response"], "sft", 1),
            SourceCandidate("local", "data/rl/if_constraints/**/*.jsonl", None, "train", ["prompt", "answer", "constraints"], "rlvr", 2),
        ],
        min_chars=20,
        max_doc_tokens=1_024,
    ))
    r.append(CorpusSlice(
        name="honesty_uncertainty",
        family="rl",
        category="honesty, uncertainty, and calibration",
        target_tokens=RL_MIX["honesty_uncertainty"],
        output_file="rl/sft.jsonl",
        format="sft",
        candidates=[
            SourceCandidate("local", "data/rl/honesty_uncertainty/**/*.jsonl", None, "train", ["messages", "prompt", "response"], "sft", 0,
                            note="Recommended as custom curated examples: abstention, uncertainty, source limits."),
            SourceCandidate("hf", "allenai/tulu-3-sft-personas-instruction-following", None, "train", ["messages", "prompt", "response"], "sft", 1),
        ],
        min_chars=20,
        max_doc_tokens=768,
    ))
    r.append(CorpusSlice(
        name="friendly_concise_style",
        family="rl",
        category="friendly, concise, non-sycophantic style",
        target_tokens=RL_MIX["friendly_concise_style"],
        output_file="rl/sft.jsonl",
        format="sft",
        candidates=[
            SourceCandidate("local", "data/rl/friendly_style/**/*.jsonl", None, "train", ["messages", "prompt", "response"], "sft", 0,
                            note="Recommended as small curated house-style data."),
            SourceCandidate("hf", "OpenAssistant/oasst1", None, "train", ["text"], "sft", 1),
        ],
        min_chars=20,
        max_doc_tokens=768,
    ))
    r.append(CorpusSlice(
        name="general_alignment_sft",
        family="rl",
        category="general alignment SFT: helpful, honest, instruction-following conversations",
        target_tokens=RL_MIX["general_alignment_sft"],
        output_file="rl/sft.jsonl",
        format="sft",
        candidates=[
            SourceCandidate("local", "data/rl/general_alignment_sft/**/*.jsonl", None, "train", ["messages", "prompt", "response", "instruction", "output"], "sft", 0),
            SourceCandidate("hf", "allenai/tulu-3-sft-mixture", None, "train", ["messages", "prompt", "response"], "sft", 1),
            SourceCandidate("hf", "OpenAssistant/oasst1", None, "train", ["text"], "sft", 2),
            SourceCandidate("hf", "databricks/databricks-dolly-15k", None, "train", ["instruction", "context", "response"], "sft", 3),
        ],
        min_chars=20,
        max_doc_tokens=1_024,
    ))
    r.append(CorpusSlice(
        name="general_alignment_preferences",
        family="rl",
        category="general alignment preference pairs",
        target_tokens=RL_MIX["general_alignment_preferences"],
        output_file="rl/preferences.jsonl",
        format="preference",
        candidates=[
            SourceCandidate("local", "data/rl/general_alignment_preferences/**/*.jsonl", None, "train", ["prompt", "chosen", "rejected", "messages"], "preference", 0),
            SourceCandidate("hf", "allenai/llama-3.1-tulu-3-405b-preference-mixture", None, "train", ["prompt", "chosen", "rejected", "messages"], "preference", 1),
            SourceCandidate("hf", "HuggingFaceH4/ultrafeedback_binarized", None, "train_prefs", ["prompt", "chosen", "rejected"], "preference", 2),
        ],
        min_chars=20,
        max_doc_tokens=1_024,
    ))
    r.append(CorpusSlice(
        name="truthfulness_calibration",
        family="rl",
        category="truthfulness, abstention, and calibration alignment",
        target_tokens=RL_MIX["truthfulness_calibration"],
        output_file="rl/sft.jsonl",
        format="sft",
        candidates=[
            SourceCandidate("local", "data/rl/truthfulness_calibration/**/*.jsonl", None, "train", ["messages", "prompt", "response", "question", "best_answer"], "sft", 0),
            SourceCandidate("hf", "truthful_qa", "generation", "validation", ["question", "best_answer", "correct_answers"], "sft", 1),
        ],
        min_chars=20,
        max_doc_tokens=768,
    ))
    r.append(CorpusSlice(
        name="code_instruction_sft",
        family="rl",
        category="code instruction following and debugging SFT",
        target_tokens=RL_MIX["code_instruction_sft"],
        output_file="rl/sft.jsonl",
        format="sft",
        candidates=[
            SourceCandidate("local", "data/rl/code_instruction_sft/**/*.jsonl", None, "train", ["messages", "prompt", "response", "instruction", "output", "code"], "sft", 0),
            SourceCandidate("hf", "ise-uiuc/Magicoder-OSS-Instruct-75K", None, "train", ["instruction", "response", "problem", "solution"], "sft", 1),
            SourceCandidate("hf", "bigcode/commitpackft", None, "train", ["old_contents", "new_contents", "message", "subject"], "sft", 2),
        ],
        min_chars=20,
        max_doc_tokens=1_024,
        quality_profile="code",
    ))
    r.append(CorpusSlice(
        name="calculator_tool_rlvr",
        family="rl",
        category="calculator-tool numeric RLVR",
        target_tokens=RL_MIX["calculator_tool_rlvr"],
        output_file="rl/rlvr.jsonl",
        format="rlvr",
        candidates=[
            SourceCandidate("local", "data/rl/calculator_tool_rlvr/**/*.jsonl", None, "train", ["prompt", "answer", "aux_score", "expression", "tool_result"], "rlvr", 0),
            SourceCandidate("hf", "allenai/RLVR-GSM", None, "train", ["problem", "question", "answer", "solution"], "rlvr", 1),
        ],
        min_chars=20,
        max_doc_tokens=768,
    ))
    r.append(CorpusSlice(
        name="python_tool_execution",
        family="rl",
        category="Python-code execution and result grounding",
        target_tokens=RL_MIX["python_tool_execution"],
        output_file="rl/rlvr.jsonl",
        format="rlvr",
        candidates=[
            SourceCandidate("local", "data/rl/python_tool_execution/**/*.jsonl", None, "train", ["prompt", "code", "stdout", "answer", "aux_score"], "rlvr", 0),
            SourceCandidate("local", "data/rl/code_execution/**/*.jsonl", None, "train", ["prompt", "code", "result", "answer"], "rlvr", 1),
        ],
        min_chars=20,
        max_doc_tokens=768,
        quality_profile="code",
    ))
    r.append(CorpusSlice(
        name="runtime_latent_tool_preferences",
        family="rl",
        category="runtime latent-tool observation preference pairs",
        target_tokens=RL_MIX["runtime_latent_tool_preferences"],
        output_file="rl/preferences.jsonl",
        format="preference",
        candidates=[
            SourceCandidate("local", "data/rl/latent_tool_calling_preferences/**/*.jsonl", None, "train", ["prompt", "chosen", "rejected", "runtime_tool_observation", "routing_decision"], "preference", 0,
                            note="Chosen answers use runtime-built hidden observations; rejected answers expose tool syntax or skip required computation."),
        ],
        min_chars=20,
        max_doc_tokens=768,
    ))

    return k + r


# -----------------------------
# Token counting
# -----------------------------

class TokenCounter:
    def __init__(self, tokenizer_name: Optional[str] = None, approx: bool = False):
        self.approx = approx or not tokenizer_name
        self.tokenizer = None
        if not self.approx and tokenizer_name:
            try:
                from transformers import AutoTokenizer  # type: ignore
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
                self.approx = False
            except Exception as exc:
                print(f"[warn] Could not load tokenizer {tokenizer_name!r}: {exc}. Falling back to approximate counts.", file=sys.stderr)
                self.approx = True

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self.approx or self.tokenizer is None:
            # Reasonable English BPE-ish estimate. Use --tokenizer for exact counts.
            words = len(re.findall(r"\S+", text))
            chars = len(text)
            return max(1, int(max(words * 1.33, chars / 4.2)))
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def split_to_token_chunks(self, text: str, max_tokens: int) -> List[str]:
        if self.count(text) <= max_tokens:
            return [text]
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            paragraphs = [text]
        chunks: List[str] = []
        cur: List[str] = []
        cur_tokens = 0
        for p in paragraphs:
            pt = self.count(p)
            if pt > max_tokens:
                # Fall back to sentence-ish pieces for very large paragraphs.
                pieces = re.split(r"(?<=[.!?])\s+", p)
                for s in pieces:
                    st = self.count(s)
                    if cur and cur_tokens + st > max_tokens:
                        chunks.append("\n\n".join(cur).strip())
                        cur, cur_tokens = [], 0
                    if st > max_tokens:
                        # Last-resort character window.
                        approx_chars = max(400, int(max_tokens * 4.0))
                        for i in range(0, len(s), approx_chars):
                            w = s[i:i + approx_chars].strip()
                            if w:
                                chunks.append(w)
                    else:
                        cur.append(s)
                        cur_tokens += st
                continue
            if cur and cur_tokens + pt > max_tokens:
                chunks.append("\n\n".join(cur).strip())
                cur, cur_tokens = [], 0
            cur.append(p)
            cur_tokens += pt
        if cur:
            chunks.append("\n\n".join(cur).strip())
        return [c for c in chunks if c]


# -----------------------------
# Quality, extraction, formatting
# -----------------------------

CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = CONTROL_RE.sub(" ", str(text))
    text = text.replace("\u00a0", " ")
    text = WS_RE.sub(" ", text).strip()
    return text


def alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(ch.isalpha() for ch in text)
    return alpha / max(1, len(text))


def repeated_line_ratio(text: str) -> float:
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 20]
    if len(lines) < 4:
        return 0.0
    return 1.0 - (len(set(lines)) / max(1, len(lines)))


def char_ngram_repetition(text: str, n: int = 24) -> float:
    if len(text) < n * 10:
        return 0.0
    grams = [text[i:i+n] for i in range(0, min(len(text) - n, 5000), n)]
    return 1.0 - (len(set(grams)) / max(1, len(grams)))


def passes_quality(text: str, sl: CorpusSlice) -> bool:
    if len(text) < sl.min_chars or len(text) > sl.max_chars:
        return False
    ar = alpha_ratio(text)
    if sl.quality_profile == "code":
        if ar < 0.18:
            return False
    elif sl.quality_profile == "math":
        if ar < 0.25:
            return False
    else:
        if ar < 0.42:
            return False
    if repeated_line_ratio(text) > 0.35:
        return False
    if char_ngram_repetition(text) > 0.45:
        return False
    lowered = text.lower()
    bad_markers = ["lorem ipsum", "javascript is disabled", "enable cookies", "access denied", "subscribe to continue"]
    if any(m in lowered for m in bad_markers):
        return False
    return True


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def dedup_key(text: str, prefix_chars: int) -> str:
    compact = re.sub(r"\W+", " ", text[:prefix_chars].lower()).strip()
    return stable_hash(compact)


def first_present(row: Dict[str, Any], fields: Sequence[str]) -> Optional[Any]:
    for f in fields:
        if f in row and row[f] is not None:
            return row[f]
    return None


def stringify_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        parts = []
        for item in v:
            if isinstance(item, dict):
                role = item.get("role") or item.get("from") or item.get("speaker") or ""
                content = item.get("content") or item.get("value") or item.get("text") or ""
                if content:
                    parts.append(f"{role}: {content}" if role else str(content))
            else:
                parts.append(stringify_value(item))
        return "\n".join(p for p in parts if p)
    if isinstance(v, dict):
        for key in ("text", "content", "value", "answer", "response"):
            if key in v:
                return stringify_value(v[key])
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def extract_text(row: Dict[str, Any], fields: Sequence[str]) -> str:
    parts = []
    for f in fields:
        if f in row and row[f] is not None:
            s = stringify_value(row[f]).strip()
            if s:
                parts.append(s)
    if not parts:
        for f in ("text", "content", "article", "abstract", "body", "title", "question", "answer", "response"):
            if f in row and row[f] is not None:
                s = stringify_value(row[f]).strip()
                if s:
                    parts.append(s)
    return normalize_text("\n\n".join(parts))


def messages_from_row(row: Dict[str, Any], fields: Sequence[str]) -> Optional[List[Dict[str, str]]]:
    raw = row.get("messages") or row.get("conversations")
    if isinstance(raw, list):
        msgs = []
        for m in raw:
            if not isinstance(m, dict):
                continue
            role = m.get("role") or m.get("from") or m.get("speaker") or "user"
            content = m.get("content") or m.get("value") or m.get("text") or ""
            role = str(role).lower()
            if role in ("human", "prompter"):
                role = "user"
            elif role in ("gpt", "assistant", "bot"):
                role = "assistant"
            elif role not in ("system", "user", "assistant"):
                role = "user"
            content = normalize_text(content)
            if content:
                msgs.append({"role": role, "content": content})
        if len(msgs) >= 2:
            return msgs
    instruction = row.get("instruction") or row.get("prompt") or row.get("question") or row.get("input")
    context = row.get("context")
    response = row.get("response") or row.get("output") or row.get("answer") or row.get("completion")
    if instruction and response:
        user = normalize_text(stringify_value(instruction))
        if context:
            user = normalize_text(user + "\n\nContext:\n" + stringify_value(context))
        assistant = normalize_text(stringify_value(response))
        if user and assistant:
            return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    text = extract_text(row, fields)
    if text and len(text) > 60:
        return [{"role": "user", "content": "Continue helpfully."}, {"role": "assistant", "content": text}]
    return None


def preference_from_row(row: Dict[str, Any], fields: Sequence[str]) -> Optional[Dict[str, Any]]:
    chosen = row.get("chosen") or row.get("accept") or row.get("winner")
    rejected = row.get("rejected") or row.get("reject") or row.get("loser")
    prompt = row.get("prompt") or row.get("instruction") or row.get("question")

    if chosen is None or rejected is None:
        # Some safety datasets are SFT/classification-like. Convert only if a safe/good response exists.
        messages = messages_from_row(row, fields)
        if messages and len(messages) >= 2:
            return None
        return None

    chosen_s = normalize_text(stringify_value(chosen))
    rejected_s = normalize_text(stringify_value(rejected))
    prompt_s = normalize_text(stringify_value(prompt)) if prompt is not None else ""

    # If chosen/rejected include the prompt as full conversations, keep them as strings.
    if not prompt_s:
        prompt_s = infer_prompt_from_pair(chosen_s, rejected_s)
    if not chosen_s or not rejected_s:
        return None
    return {"prompt": prompt_s, "chosen": chosen_s, "rejected": rejected_s}


def infer_prompt_from_pair(chosen: str, rejected: str) -> str:
    # Very conservative common-prefix extraction.
    n = min(len(chosen), len(rejected), 2000)
    i = 0
    while i < n and chosen[i] == rejected[i]:
        i += 1
    prefix = chosen[:i].strip()
    if len(prefix) >= 20:
        return prefix[-1500:]
    return ""


def rlvr_from_row(row: Dict[str, Any], fields: Sequence[str]) -> Optional[Dict[str, Any]]:
    prompt = row.get("prompt") or row.get("problem") or row.get("question") or row.get("instruction")
    answer = row.get("answer") or row.get("final_answer") or row.get("target") or row.get("output")
    solution = row.get("solution") or row.get("rationale") or row.get("reasoning") or row.get("response")
    constraints = row.get("constraints") or row.get("checks") or row.get("aux_score")
    messages = messages_from_row(row, fields)
    if prompt is None and messages:
        prompt = messages[0]["content"]
        if answer is None and len(messages) > 1:
            answer = messages[-1]["content"]
    prompt_s = normalize_text(stringify_value(prompt)) if prompt is not None else ""
    answer_s = normalize_text(stringify_value(answer)) if answer is not None else ""
    solution_s = normalize_text(stringify_value(solution)) if solution is not None else ""
    if not prompt_s or not (answer_s or solution_s):
        return None
    out = {"prompt": prompt_s, "answer": answer_s, "solution": solution_s}
    if constraints is not None:
        out["constraints"] = constraints
    return out


def token_text_for_record(record: Dict[str, Any], fmt: str) -> str:
    if fmt == "text":
        return record.get("text", "")
    if fmt == "sft":
        return "\n".join(m.get("content", "") for m in record.get("messages", []))
    if fmt == "preference":
        return "\n".join([record.get("prompt", ""), record.get("chosen", ""), record.get("rejected", "")])
    if fmt == "rlvr":
        return "\n".join([record.get("prompt", ""), record.get("solution", ""), record.get("answer", "")])
    return json.dumps(record, ensure_ascii=False)


def row_to_record(row: Dict[str, Any], sl: CorpusSlice, candidate: SourceCandidate) -> Optional[Dict[str, Any]]:
    if sl.format == "text":
        text = extract_text(row, candidate.text_fields)
        if not text:
            return None
        return {"text": text}
    if sl.format == "sft" or candidate.format == "sft":
        msgs = messages_from_row(row, candidate.text_fields)
        if not msgs:
            return None
        return {"messages": msgs}
    if sl.format == "preference" or candidate.format == "preference":
        return preference_from_row(row, candidate.text_fields)
    if sl.format == "rlvr" or candidate.format == "rlvr":
        return rlvr_from_row(row, candidate.text_fields)
    return None


# -----------------------------
# Dataset iteration
# -----------------------------


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def iter_local_rows(glob_pattern: str) -> Iterator[Dict[str, Any]]:
    paths = sorted(Path().glob(glob_pattern)) if not glob_pattern.startswith("/") else sorted(Path("/").glob(glob_pattern[1:]))
    for path in paths:
        if not path.is_file():
            continue
        suffixes = "".join(path.suffixes)
        try:
            if suffixes.endswith(".jsonl") or suffixes.endswith(".jsonl.gz"):
                with open_text(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict):
                                yield obj
                        except Exception:
                            continue
            elif suffixes.endswith(".json"):
                with open_text(path) as f:
                    obj = json.load(f)
                if isinstance(obj, list):
                    for item in obj:
                        if isinstance(item, dict):
                            yield item
                elif isinstance(obj, dict):
                    if "data" in obj and isinstance(obj["data"], list):
                        for item in obj["data"]:
                            if isinstance(item, dict):
                                yield item
                    else:
                        yield obj
            elif suffixes.endswith(".csv"):
                with open_text(path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        yield dict(row)
            else:
                with open_text(path) as f:
                    text = f.read()
                if text.strip():
                    yield {"text": text}
        except Exception as exc:
            print(f"[warn] Skipping local file {path}: {exc}", file=sys.stderr)


def resolve_hf_config(path: str, preferred: Optional[str]) -> Optional[str]:
    if preferred:
        return preferred
    try:
        from datasets import get_dataset_config_names  # type: ignore
        configs = get_dataset_config_names(path)
        if not configs:
            return None
        # Prefer small/sample configs for smoke tests, otherwise common train configs.
        preferred_terms = ["sample-10BT", "sample", "default", "en", "20231101.en"]
        for term in preferred_terms:
            for cfg in configs:
                if cfg == term or term.lower() in cfg.lower():
                    return cfg
        return configs[0]
    except Exception:
        return preferred


def iter_hf_rows(candidate: SourceCandidate, seed: int) -> Iterator[Dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError("Missing dependency: pip install datasets") from exc

    config = resolve_hf_config(candidate.path, candidate.config)
    kwargs = {"split": candidate.split, "streaming": True}
    print(f"[info] Loading HF dataset {candidate.path} config={config!r} split={candidate.split!r}", file=sys.stderr)
    if config:
        ds = load_dataset(candidate.path, config, **kwargs)
    else:
        ds = load_dataset(candidate.path, **kwargs)
    # Streaming shuffle uses a finite buffer. This is enough to avoid always taking leading rows.
    try:
        ds = ds.shuffle(seed=seed, buffer_size=20_000)
    except Exception:
        pass
    for row in ds:
        if isinstance(row, dict):
            yield row


def iter_candidate_rows(candidate: SourceCandidate, seed: int) -> Iterator[Dict[str, Any]]:
    if not candidate.enabled:
        return iter(())
    if candidate.kind == "local":
        return iter_local_rows(candidate.path)
    if candidate.kind == "hf":
        return iter_hf_rows(candidate, seed)
    raise ValueError(f"Unknown source kind: {candidate.kind}")


# -----------------------------
# Build logic
# -----------------------------

class SeenDeduper:
    def __init__(self, max_keys: int = 25_000_000):
        self.max_keys = max_keys
        self.keys = set()

    def add(self, key: str) -> bool:
        if key in self.keys:
            return False
        if len(self.keys) < self.max_keys:
            self.keys.add(key)
        return True


def atomic_jsonl_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "a", encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_local_overrides(path: Optional[str]) -> Dict[str, List[SourceCandidate]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, List[SourceCandidate]] = {}
    for slice_name, candidates in raw.items():
        out[slice_name] = [SourceCandidate(**c) for c in candidates]
    return out


def apply_overrides(plan: List[CorpusSlice], override_path: Optional[str]) -> List[CorpusSlice]:
    overrides = load_local_overrides(override_path)
    if not overrides:
        return plan
    for sl in plan:
        if sl.name in overrides:
            sl.candidates = overrides[sl.name]
    return plan


def scale_plan(plan: List[CorpusSlice], scale: float, build: str) -> List[CorpusSlice]:
    selected = []
    for sl in plan:
        if build == "all" or sl.family == build:
            cp = CorpusSlice(**{**asdict(sl), "candidates": [SourceCandidate(**asdict(c)) for c in sl.candidates]})
            cp.target_tokens = max(1, int(cp.target_tokens * scale))
            selected.append(cp)
    return selected


def effective_scale(args: argparse.Namespace, plan: Sequence[CorpusSlice]) -> Tuple[float, Dict[str, Any]]:
    """Return the final scale while preserving every slice's diversity ratio."""
    base_selected = [sl for sl in plan if args.build == "all" or sl.family == args.build]
    base_total = sum(int(sl.target_tokens) for sl in base_selected)
    size_fit_scale = 1.0
    target_total = base_total
    requested_tokens = int(getattr(args, "target_tokens", 0) or 0)
    mode = str(getattr(args, "size_fit_mode", "off"))
    if requested_tokens > 0:
        mode = "tokens"
        target_total = requested_tokens
        size_fit_scale = target_total / max(1, base_total)
    elif mode == "params":
        target_total = int(float(args.params_m) * 1_000_000 * float(args.tokens_per_param))
        size_fit_scale = target_total / max(1, base_total)
    elif mode == "tokens":
        target_total = int(args.target_total_tokens)
        if target_total <= 0:
            raise ValueError("--target_total_tokens or --target_tokens must be > 0 when --size_fit_mode tokens")
        size_fit_scale = target_total / max(1, base_total)
    final_scale = float(args.scale) * float(size_fit_scale)
    selected_target_after_scale = int(round(base_total * final_scale))
    return final_scale, {
        "size_fit_mode": mode,
        "params_m": float(getattr(args, "params_m", BASE_CORPUS_PARAMS_M)),
        "tokens_per_param": float(getattr(args, "tokens_per_param", 320.0)),
        "corpus_total_tokens": int(base_total),
        "corpus_total_tokens_compact": format_scaled_count(base_total),
        "target_total_tokens": int(target_total),
        "target_total_tokens_compact": format_scaled_count(target_total),
        "selected_target_after_user_scale": selected_target_after_scale,
        "download_fraction_of_selected_corpus": float(selected_target_after_scale / max(1, base_total)),
        "base_selected_tokens": int(base_total),
        "user_scale": float(args.scale),
        "size_fit_scale": float(size_fit_scale),
        "final_scale": float(final_scale),
        "preserves_slice_ratios": True,
    }


def build_slice(
    sl: CorpusSlice,
    out_dir: Path,
    counter: TokenCounter,
    seed: int,
    global_dedup: SeenDeduper,
    dry_run: bool = False,
    max_rows_per_slice: Optional[int] = None,
) -> Dict[str, Any]:
    start = time.time()
    out_path = out_dir / sl.output_file
    accepted_docs = 0
    accepted_tokens = 0
    skipped_quality = 0
    skipped_dupe = 0
    read_rows = 0
    source_used = None
    candidate_errors: List[str] = []

    writer = None if dry_run else atomic_jsonl_writer(out_path)
    try:
        for cand in sorted(sl.candidates, key=lambda c: c.priority):
            if accepted_tokens >= sl.target_tokens:
                break
            source_used = cand.path
            try:
                rows = iter_candidate_rows(cand, seed=seed)
                for row in rows:
                    read_rows += 1
                    if max_rows_per_slice and read_rows > max_rows_per_slice:
                        break
                    rec = row_to_record(row, sl, cand)
                    if not rec:
                        skipped_quality += 1
                        continue

                    token_text = normalize_text(token_text_for_record(rec, sl.format))
                    if not passes_quality(token_text, sl):
                        skipped_quality += 1
                        continue

                    # For text corpora, split long docs into manageable chunks.
                    records: List[Dict[str, Any]] = []
                    if sl.format == "text":
                        for chunk in counter.split_to_token_chunks(rec["text"], sl.max_doc_tokens):
                            chunk = normalize_text(chunk)
                            if passes_quality(chunk, sl):
                                records.append({"text": chunk})
                    else:
                        records = [rec]

                    for r in records:
                        ttext = normalize_text(token_text_for_record(r, sl.format))
                        dk = dedup_key(ttext, sl.dedup_prefix_chars)
                        if not global_dedup.add(dk):
                            skipped_dupe += 1
                            continue
                        toks = counter.count(ttext)
                        if toks <= 0:
                            continue
                        out_rec = {
                            **r,
                            "_meta": {
                                "slice": sl.name,
                                "family": sl.family,
                                "category": sl.category,
                                "source": cand.path,
                                "tokens": toks,
                            },
                        }
                        if writer:
                            writer.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                        accepted_docs += 1
                        accepted_tokens += toks
                        if accepted_tokens >= sl.target_tokens:
                            break
                    if accepted_tokens >= sl.target_tokens:
                        break
                if accepted_tokens > 0:
                    # Keep going through later candidates only if target not met.
                    pass
            except Exception as exc:
                msg = f"{cand.path}: {exc}"
                candidate_errors.append(msg)
                print(f"[warn] Candidate failed for slice {sl.name}: {msg}", file=sys.stderr)
                continue
    finally:
        if writer:
            writer.close()

    elapsed = time.time() - start
    status = "complete" if accepted_tokens >= sl.target_tokens else "partial"
    return {
        "slice": sl.name,
        "family": sl.family,
        "category": sl.category,
        "target_tokens": sl.target_tokens,
        "accepted_tokens": accepted_tokens,
        "accepted_docs": accepted_docs,
        "read_rows": read_rows,
        "skipped_quality": skipped_quality,
        "skipped_dupe": skipped_dupe,
        "output_file": sl.output_file,
        "source_last_used": source_used,
        "candidate_errors": candidate_errors,
        "status": status,
        "elapsed_sec": round(elapsed, 2),
    }


def build_corpora(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    plan = apply_overrides(default_plan(), args.local_override_json)
    final_scale, scale_info = effective_scale(args, plan)
    selected = scale_plan(plan, final_scale, args.build)

    if args.clean and out_dir.exists():
        # Avoid shutil.rmtree to make accidental deletion slightly less likely.
        for rel in ["knowledge/train.jsonl", "rl/sft.jsonl", "rl/preferences.jsonl", "rl/rlvr.jsonl"]:
            p = out_dir / rel
            if p.exists():
                p.unlink()

    write_json(out_dir / "plan.json", {
        "model_assumption": "30M dense NEED model, about 10B knowledge/pretraining tokens plus 650M general post-training token-equivalent",
        "knowledge_target_tokens": KNOWLEDGE_TARGET_TOKENS,
        "rl_target_tokens": RL_TARGET_TOKENS,
        "scale": final_scale,
        "scale_info": scale_info,
        "selected_build": args.build,
        "slices": [asdict(sl) for sl in selected],
    })

    counter = TokenCounter(args.tokenizer, approx=args.approx_tokens)
    dedup = SeenDeduper(max_keys=args.max_dedup_keys)
    manifests: Dict[str, List[Dict[str, Any]]] = {"knowledge": [], "rl": []}

    for sl in selected:
        print(f"[info] Building slice={sl.name} target={sl.target_tokens:,} format={sl.format}", file=sys.stderr)
        result = build_slice(
            sl,
            out_dir=out_dir,
            counter=counter,
            seed=args.seed,
            global_dedup=dedup,
            dry_run=args.dry_run,
            max_rows_per_slice=args.max_rows_per_slice,
        )
        manifests[sl.family].append(result)
        print(f"[info] Finished {sl.name}: {result['accepted_tokens']:,}/{sl.target_tokens:,} tokens ({result['status']})", file=sys.stderr)

    for family in ("knowledge", "rl"):
        if manifests[family]:
            total = sum(x["accepted_tokens"] for x in manifests[family])
            target = sum(x["target_tokens"] for x in manifests[family])
            write_json(out_dir / family / "manifest.json", {
                "family": family,
                "target_tokens": target,
                "accepted_tokens": total,
                "accepted_docs": sum(x["accepted_docs"] for x in manifests[family]),
                "completion_ratio": total / max(1, target),
                "slices": manifests[family],
            })

    print(f"[done] Wrote corpus outputs under {out_dir}", file=sys.stderr)


# -----------------------------
# Plan and inspection helpers
# -----------------------------


def write_plan_only(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    plan = apply_overrides(default_plan(), args.local_override_json)
    final_scale, scale_info = effective_scale(args, plan)
    selected = scale_plan(plan, final_scale, args.build)
    write_json(out_dir / "plan.json", {
        "knowledge_mix": KNOWLEDGE_MIX,
        "rl_mix": RL_MIX,
        "selected_build": args.build,
        "scale": final_scale,
        "scale_info": scale_info,
        "slices": [asdict(sl) for sl in selected],
    })
    print(f"[done] Wrote {out_dir / 'plan.json'}", file=sys.stderr)


def print_budget() -> None:
    def show(title: str, mix: Dict[str, int], total: int) -> None:
        print(title)
        for k, v in mix.items():
            print(f"  {k:34s} {v:>14,} tokens  {v / total:6.2%}")
        print(f"  {'TOTAL':34s} {sum(mix.values()):>14,} tokens")
        print()
    show("Knowledge/pretraining corpus", KNOWLEDGE_MIX, KNOWLEDGE_TARGET_TOKENS)
    show("General RL/post-training corpus", RL_MIX, RL_TARGET_TOKENS)


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build knowledge and general RL corpuses for NEED; default plan is about 150B tokens.")
    p.add_argument("--build", choices=["knowledge", "rl", "all"], default="all", help="Which corpus family to build.")
    p.add_argument("--out_dir", default="data/corpuses", help="Output directory.")
    p.add_argument("--tokenizer", default=None, help="HF tokenizer name or local tokenizer path. Omit with --approx_tokens.")
    p.add_argument("--approx_tokens", action="store_true", help="Use approximate token counts instead of loading a tokenizer.")
    p.add_argument("--scale", type=float, default=1.0, help="Manual multiplier applied after optional size fitting. Smoke tests can use a tiny value.")
    p.add_argument("--size_fit_mode", choices=["off", "params", "tokens"], default="off", help="Optionally scale every corpus slice together so diversity ratios remain intact.")
    p.add_argument("--params_m", type=float, default=BASE_CORPUS_PARAMS_M, help="Model parameter count in millions for --size_fit_mode params.")
    p.add_argument("--tokens_per_param", type=float, default=320.0, help="Corpus tokens per model parameter for --size_fit_mode params.")
    p.add_argument("--target_total_tokens", type=parse_scaled_count, default=0, help="Total selected corpus tokens for --size_fit_mode tokens; accepts M/B/T suffixes.")
    p.add_argument("--target_tokens", type=parse_scaled_count, default=0, help="Shortcut for --size_fit_mode tokens --target_total_tokens; accepts M/B/T suffixes.")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--clean", action="store_true", help="Delete existing output JSONL files before building.")
    p.add_argument("--dry_run", action="store_true", help="Count and validate without writing JSONL records.")
    p.add_argument("--write-plan", action="store_true", help="Only write plan.json and exit.")
    p.add_argument("--print-budget", action="store_true", help="Print the token budget and exit.")
    p.add_argument("--local_override_json", default=None, help="Optional JSON mapping slice names to replacement SourceCandidate lists.")
    p.add_argument("--max_rows_per_slice", type=int, default=None, help="Debug cap on input rows per slice.")
    p.add_argument("--max_dedup_keys", type=int, default=25_000_000, help="Maximum exact dedup keys held in RAM.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.print_budget:
        print_budget()
        return
    if args.write_plan:
        write_plan_only(args)
        return
    if args.scale <= 0:
        raise SystemExit("--scale must be positive")
    build_corpora(args)


if __name__ == "__main__":
    main()
