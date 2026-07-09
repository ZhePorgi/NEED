#!/usr/bin/env python3
"""Runtime-only latent tool controller for NEED.

This module deliberately avoids model-authored tool calls.  NEED and the
sidecar do not emit JSON, code, function-call objects, or schemas in order to
use tools.  The runtime detects narrow tool-worthy patterns, builds validated
instructions itself, executes approved tools, and injects compact observations
back into hidden conditioning context.

The result is available immediately at inference time and does not depend on
low-level RL.  LLRL examples can still improve the final answer style around
computed evidence, but the tool bridge itself is a deterministic runtime layer.
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_TOOL_SYSTEM_PROMPT = "You are a helpful AI assistant."
RUNTIME_TOOL_POLICY = "runtime_determined_no_model_built_calls_no_llrl_required"


@dataclass
class LatentToolConfig:
    enabled: bool = True
    calculator_enabled: bool = True
    python_enabled: bool = False
    # Only runtime planning is supported.  Legacy values are accepted but ignored
    # so older profiles/scripts continue to load without enabling model planning.
    planner: str = "runtime"
    max_calls: int = 3
    timeout_s: float = 3.0
    max_output_chars: int = 2000
    max_code_chars: int = 2400
    max_expression_chars: int = 300
    python_memory_mb: int = 512
    expose_tool_details: bool = False
    # Backwards-compatible aliases used by older generate/browser/runtime profiles.
    calculator: Optional[bool] = None
    python: Optional[bool] = None
    sidecar_planning: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.calculator is not None:
            self.calculator_enabled = bool(self.calculator)
        if self.python is not None:
            self.python_enabled = bool(self.python)
        # Do not honor sidecar_planning.  It is intentionally inert; no model or
        # sidecar is ever asked to construct a tool call.
        self.planner = "runtime"
        if self.max_calls < 0:
            self.max_calls = 0


@dataclass
class ToolInstruction:
    tool: str
    expression: str = ""
    code: str = ""
    purpose: str = ""
    source: str = "runtime"


@dataclass
class ToolResult:
    tool: str
    ok: bool
    purpose: str = ""
    expression: str = ""
    result: Any = None
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    elapsed_s: float = 0.0
    source: str = "runtime"

    def compact(self, max_chars: int) -> Dict[str, Any]:
        out = asdict(self)
        for key in ("stdout", "stderr", "error"):
            val = str(out.get(key, ""))
            if len(val) > max_chars:
                out[key] = val[:max_chars].rstrip() + " ..."
        if isinstance(out.get("result"), str) and len(out["result"]) > max_chars:
            out["result"] = out["result"][:max_chars].rstrip() + " ..."
        return out


_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}
_ALLOWED_UNARY = {ast.UAdd: lambda a: +a, ast.USub: lambda a: -a}
_ALLOWED_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "ceil": math.ceil,
    "floor": math.floor,
}
_ALLOWED_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _clamp_text(x: Any, n: int) -> str:
    s = str(x or "").strip()
    if len(s) > n:
        return s[:n].rstrip() + " ..."
    return s


def _eval_calc_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_calc_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval_calc_node(elt) for elt in node.elts]
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_CONSTS:
            return _ALLOWED_CONSTS[node.id]
        raise ValueError(f"name not allowed: {node.id}")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_eval_calc_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_calc_node(node.left)
        right = _eval_calc_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(float(right)) > 12:
            raise ValueError("exponent too large")
        return _ALLOWED_BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "math":
            name = node.func.attr
        else:
            raise ValueError("function not allowed")
        if name not in _ALLOWED_FUNCS:
            raise ValueError(f"function not allowed: {name}")
        args = [_eval_calc_node(a) for a in node.args]
        if len(args) > 12:
            raise ValueError("too many arguments")
        return _ALLOWED_FUNCS[name](*args)
    raise ValueError(f"unsupported expression: {type(node).__name__}")


def calculator_eval(expression: str, max_chars: int = 300) -> Any:
    expr = str(expression or "").strip().replace("^", "**")
    if not expr:
        raise ValueError("empty expression")
    if len(expr) > max_chars:
        raise ValueError("expression too long")
    if re.search(r"[^0-9a-zA-Z_+\-*/%().,\[\]\s]", expr):
        raise ValueError("expression contains unsupported characters")
    tree = ast.parse(expr, mode="eval")
    value = _eval_calc_node(tree)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError("non-finite result")
        return round(value, 12)
    return value


def extract_calculator_expression(text: str, max_chars: int = 300) -> str:
    s = str(text or "")[:4000]
    # Percent-of language before generic spans, because the plain span extractor
    # would otherwise see only one number.
    pct = re.search(r"(\d+(?:\.\d+)?)\s*(?:percent|%)\s+of\s+(\d+(?:\.\d+)?)", s, flags=re.I)
    if pct:
        return f"({pct.group(1)} / 100) * {pct.group(2)}"[:max_chars]
    # Prefer explicit math spans after common verbs.
    explicit = re.search(r"(?:calculate|compute|evaluate|what is|what's|solve)\s+([^\n?]+)", s, flags=re.I)
    if explicit:
        cand = explicit.group(1)
        cand = re.split(r"\s+and\s+(?:answer|give|return|explain|show|round)\b", cand, maxsplit=1, flags=re.I)[0]
        cand = re.split(r"(?:\?|\.|,\s+(?:and|then)\b)", cand)[0]
        cand = cand.strip(" :;`")
        if re.search(r"\d", cand) and re.search(r"[+\-*/%^]", cand):
            return cand[:max_chars]
    # Arithmetic-looking spans.
    spans = re.findall(r"(?<!\w)(?:\(?\s*[-+]?\d+(?:\.\d+)?\s*\)?\s*(?:\*\*|[+\-*/%^])\s*)+\(?\s*[-+]?\d+(?:\.\d+)?\s*\)?", s)
    if spans:
        spans.sort(key=len, reverse=True)
        return spans[0][:max_chars]
    return ""


_ALLOWED_IMPORT_ROOTS = {
    "math", "statistics", "decimal", "fractions", "itertools", "functools", "collections",
    "heapq", "bisect", "random", "re", "json", "datetime", "calendar", "array", "operator",
}
_BLOCKED_IMPORT_ROOTS = {
    "os", "sys", "subprocess", "socket", "requests", "urllib", "pathlib", "shutil", "ctypes",
    "multiprocessing", "threading", "asyncio", "http", "ftplib", "telnetlib", "pickle", "shelve",
    "importlib", "builtins", "inspect", "site", "runpy", "pkgutil",
}
_BLOCKED_CALL_NAMES = {
    "open", "eval", "exec", "compile", "__import__", "input", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "dir", "help", "breakpoint", "exit", "quit",
    "type", "object", "super", "memoryview", "classmethod", "staticmethod", "property",
}


class _PythonGuard(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: List[str] = []

    def _root(self, name: str) -> str:
        return str(name).split(".", 1)[0]

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            root = self._root(alias.name)
            if root in _BLOCKED_IMPORT_ROOTS or root not in _ALLOWED_IMPORT_ROOTS:
                self.errors.append(f"import not allowed: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        root = self._root(node.module or "")
        if root in _BLOCKED_IMPORT_ROOTS or root not in _ALLOWED_IMPORT_ROOTS:
            self.errors.append(f"import not allowed: {node.module}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id.startswith("__") or node.id in _BLOCKED_CALL_NAMES:
            self.errors.append(f"name not allowed: {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        if node.attr.startswith("__"):
            self.errors.append(f"attribute not allowed: {node.attr}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        func = node.func
        if isinstance(func, ast.Name) and func.id in _BLOCKED_CALL_NAMES:
            self.errors.append(f"call not allowed: {func.id}")
        if isinstance(func, ast.Attribute) and func.attr in _BLOCKED_CALL_NAMES:
            self.errors.append(f"call not allowed: {func.attr}")
        self.generic_visit(node)


def validate_python_code(code: str, max_chars: int = 2400) -> Tuple[bool, str]:
    code = str(code or "").strip()
    if not code:
        return False, "empty code"
    if len(code) > max_chars:
        return False, "code too long"
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return False, f"syntax error: {exc}"
    guard = _PythonGuard()
    guard.visit(tree)
    if guard.errors:
        return False, "; ".join(guard.errors[:5])
    return True, "ok"


def _resource_limiter(memory_mb: int, timeout_s: float, file_bytes: int = 2_000_000):
    def _limit() -> None:
        try:
            import resource  # type: ignore
            cpu = max(1, int(math.ceil(float(timeout_s))) + 1)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
            mem = max(64, int(memory_mb)) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            if hasattr(resource, "RLIMIT_FSIZE"):
                cap = max(64_000, int(file_bytes))
                resource.setrlimit(resource.RLIMIT_FSIZE, (cap, cap))
        except Exception:
            pass
    return _limit if os.name == "posix" else None


def run_python_code(code: str, *, timeout_s: float = 3.0, max_output_chars: int = 2000, memory_mb: int = 512, max_code_chars: int = 2400) -> ToolResult:
    t0 = time.perf_counter()
    ok, msg = validate_python_code(code, max_chars=max_code_chars)
    if not ok:
        return ToolResult(tool="python", ok=False, error=msg, elapsed_s=round(time.perf_counter() - t0, 4))
    script_text = str(code).strip() + "\n"
    with tempfile.TemporaryDirectory(prefix="need_latent_py_") as td:
        path = Path(td) / "tool_run.py"
        stdout_path = Path(td) / "stdout.txt"
        stderr_path = Path(td) / "stderr.txt"
        path.write_text(script_text, encoding="utf-8")
        env = {"PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
        file_cap = max(256_000, int(max_output_chars) * 4)
        try:
            with stdout_path.open("wb") as out_f, stderr_path.open("wb") as err_f:
                proc = subprocess.run(
                    [sys.executable, "-I", "-S", str(path)],
                    cwd=td,
                    env=env,
                    stdout=out_f,
                    stderr=err_f,
                    timeout=max(0.2, float(timeout_s)),
                    preexec_fn=_resource_limiter(memory_mb, timeout_s, file_bytes=file_cap),
                )
            stdout = stdout_path.read_bytes()[: max_output_chars + 1].decode("utf-8", errors="replace")
            stderr = stderr_path.read_bytes()[: max_output_chars + 1].decode("utf-8", errors="replace")
            stdout = _clamp_text(stdout, max_output_chars)
            stderr = _clamp_text(stderr, max_output_chars)
            truncated = stdout_path.stat().st_size > max_output_chars or stderr_path.stat().st_size > max_output_chars
            err = "" if proc.returncode == 0 else f"returncode={proc.returncode}"
            if truncated and not err:
                err = "output_truncated"
            return ToolResult(
                tool="python",
                ok=(proc.returncode == 0),
                stdout=stdout,
                stderr=stderr,
                error=err,
                elapsed_s=round(time.perf_counter() - t0, 4),
            )
        except subprocess.TimeoutExpired:
            return ToolResult(tool="python", ok=False, error="timeout", elapsed_s=round(time.perf_counter() - t0, 4))
        except Exception as exc:
            return ToolResult(tool="python", ok=False, error=str(exc), elapsed_s=round(time.perf_counter() - t0, 4))


def _extract_code_blocks(text: str) -> List[str]:
    return [b.strip() for b in re.findall(r"```(?:python|py)?\s*(.*?)```", str(text or ""), flags=re.I | re.S) if b.strip()]


def _extract_number_list(text: str) -> List[float]:
    s = str(text or "")[:5000]
    bracket = re.search(r"\[\s*[-+]?\d", s)
    if bracket:
        start = bracket.start()
        end = s.find("]", start)
        if end > start:
            segment = s[start : end + 1]
            nums = re.findall(r"[-+]?\d+(?:\.\d+)?", segment)
            if len(nums) >= 2:
                return [float(n) if "." in n else int(n) for n in nums[:200]]
    # Prefer content after a colon when the prompt says it is a list.
    if ":" in s and re.search(r"\b(numbers?|values?|scores?|list|data)\b", s, re.I):
        s = s.split(":", 1)[1]
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    return [float(n) if "." in n else int(n) for n in nums[:200]]


def _jsonable_numbers(nums: Sequence[float]) -> str:
    vals: List[Any] = []
    for n in nums:
        if isinstance(n, float) and n.is_integer():
            vals.append(int(n))
        else:
            vals.append(n)
    return json.dumps(vals, ensure_ascii=False)


def _template_sort_numbers(nums: Sequence[float]) -> str:
    return f"data = {_jsonable_numbers(nums)}\nprint(sorted(data))"


def _template_stats(nums: Sequence[float], low: str) -> str:
    wants = {
        "count": bool(re.search(r"\b(count|how many)\b", low)),
        "sum": bool(re.search(r"\b(sum|total)\b", low)),
        "mean": bool(re.search(r"\b(mean|average|avg)\b", low)),
        "median": "median" in low,
        "min": bool(re.search(r"\b(minimum|min|smallest|lowest)\b", low)),
        "max": bool(re.search(r"\b(maximum|max|largest|highest)\b", low)),
        "stdev": bool(re.search(r"\b(stdev|standard deviation|std)\b", low)),
        "variance": "variance" in low,
    }
    if not any(wants.values()):
        wants.update({"count": True, "sum": True, "mean": True, "median": True, "min": True, "max": True})
    lines = [
        "import json, statistics",
        f"data = {_jsonable_numbers(nums)}",
        "out = {}",
    ]
    if wants["count"]:
        lines.append("out['count'] = len(data)")
    if wants["sum"]:
        lines.append("out['sum'] = sum(data)")
    if wants["mean"]:
        lines.append("out['mean'] = statistics.mean(data)")
    if wants["median"]:
        lines.append("out['median'] = statistics.median(data)")
    if wants["min"]:
        lines.append("out['min'] = min(data)")
    if wants["max"]:
        lines.append("out['max'] = max(data)")
    if wants["stdev"]:
        lines.append("out['stdev'] = statistics.stdev(data) if len(data) > 1 else 0")
    if wants["variance"]:
        lines.append("out['variance'] = statistics.variance(data) if len(data) > 1 else 0")
    lines.append("print(json.dumps(out, sort_keys=True))")
    return "\n".join(lines)


def _template_word_count(text: str, needle: str = "") -> str:
    payload = json.dumps(text, ensure_ascii=False)
    if needle:
        n = json.dumps(needle, ensure_ascii=False)
        return "import re, json\ntext = " + payload + "\nneedle = " + n + "\nprint(json.dumps({'occurrences': len(re.findall(re.escape(needle), text, flags=re.I))}))"
    return "import re, json\ntext = " + payload + "\nwords = re.findall(r'\\b\\w+\\b', text)\nprint(json.dumps({'characters': len(text), 'words': len(words), 'lines': text.count('\\n') + (1 if text else 0)}))"


def _extract_quoted_text(text: str) -> str:
    s = str(text or "")
    for pat in [r'"([^"]{3,2000})"', r"'([^']{3,2000})'", r"```(?:text)?\s*(.*?)```"]:
        m = re.search(pat, s, flags=re.S)
        if m:
            return m.group(1).strip()
    return ""


def _runtime_python_plan(prompt: str, cfg: LatentToolConfig) -> List[ToolInstruction]:
    if not cfg.python_enabled:
        return []
    text = str(prompt or "")
    low = text.lower()
    out: List[ToolInstruction] = []
    blocks = _extract_code_blocks(text)
    wants_execution = any(phrase in low for phrase in [
        "run this", "run the code", "execute this", "execute the code", "what does this code output",
        "what is the output", "code execution", "use python", "run python", "python tool",
    ])
    if blocks and wants_execution:
        out.append(ToolInstruction(tool="python", code=blocks[0][: cfg.max_code_chars], purpose="run user-supplied Python code", source="runtime_user_code_block"))
        return out[: cfg.max_calls]

    nums = _extract_number_list(text)
    wants_sort = bool(re.search(r"\b(sort|sorted|ascending|descending|order)\b", low)) and len(nums) >= 2
    if wants_sort:
        code = _template_sort_numbers(nums)
        if re.search(r"\b(descending|reverse)\b", low):
            code = f"data = {_jsonable_numbers(nums)}\nprint(sorted(data, reverse=True))"
        out.append(ToolInstruction(tool="python", code=code, purpose="sort numeric list", source="runtime_template"))

    wants_stats = bool(re.search(r"\b(mean|average|avg|median|stdev|standard deviation|variance|sum|total|minimum|maximum|smallest|largest)\b", low)) and len(nums) >= 2
    if wants_stats and len(out) < cfg.max_calls:
        out.append(ToolInstruction(tool="python", code=_template_stats(nums, low), purpose="compute deterministic numeric statistics", source="runtime_template"))

    wants_count = bool(re.search(r"\b(count words|word count|character count|count characters|count occurrences|how many times)\b", low))
    quoted = _extract_quoted_text(text)
    if wants_count and quoted and len(out) < cfg.max_calls:
        needle = ""
        m = re.search(r"(?:count occurrences of|how many times does)\s+['\"]?([^'\"?]{1,80})['\"]?", text, flags=re.I)
        if m:
            needle = m.group(1).strip(" .,:;\n\t'\"")
        out.append(ToolInstruction(tool="python", code=_template_word_count(quoted, needle), purpose="count text deterministically", source="runtime_template"))
    return out[: cfg.max_calls]


def _runtime_calculator_plan(prompt: str, cfg: LatentToolConfig) -> List[ToolInstruction]:
    if not cfg.calculator_enabled:
        return []
    text = str(prompt or "")
    low = text.lower()
    # If the user is asking about supplied code, let the Python branch handle the
    # fenced code.  This prevents the arithmetic extractor from grabbing partial
    # expressions inside code blocks.
    if _extract_code_blocks(text) and any(phrase in low for phrase in [
        "what is the output", "what does this code output", "run this", "run the code",
        "execute this", "execute the code", "debug this code", "trace this code",
        "evaluate this code",
    ]):
        return []
    expr = extract_calculator_expression(text, cfg.max_expression_chars)
    if not expr:
        return []
    tool_words = ["calculate", "compute", "evaluate", "what is", "what's", "solve", "percent", "%"]
    looks_direct_math = bool(re.search(r"\d\s*(?:\*\*|[+\-*/%^])\s*\d", expr)) and len(expr) >= 3
    if any(w in low for w in tool_words) or looks_direct_math:
        return [ToolInstruction(tool="calculator", expression=expr, purpose="compute exact arithmetic", source="runtime_expression_extractor")]
    return []


def _runtime_plan(prompt: str, latent_summary: str, raw_cot: str, cfg: LatentToolConfig) -> List[ToolInstruction]:
    if not cfg.enabled:
        return []
    # The prompt is the only authoritative source for tool construction.  Latent
    # summaries may help the final decoder after results are injected, but they do
    # not author calls, code, expressions, schemas, or arguments.
    plan: List[ToolInstruction] = []
    seen = set()
    for item in _runtime_calculator_plan(prompt, cfg) + _runtime_python_plan(prompt, cfg):
        key = (item.tool, item.expression, item.code)
        if key in seen:
            continue
        seen.add(key)
        plan.append(item)
        if len(plan) >= cfg.max_calls:
            break
    return plan


def build_tool_plan(prompt: str, latent_summary: str = "", sidecar_rt: Any = None, cfg: Optional[LatentToolConfig] = None, raw_cot: str = "") -> List[ToolInstruction]:
    """Return runtime-built instructions only.

    ``sidecar_rt`` is accepted for API compatibility but is intentionally
    ignored.  No model or sidecar creates the call object.
    """
    config = cfg or LatentToolConfig()
    return _runtime_plan(prompt, latent_summary, raw_cot, config)


def execute_instruction(ins: ToolInstruction, cfg: LatentToolConfig) -> ToolResult:
    if ins.tool == "calculator":
        t0 = time.perf_counter()
        try:
            value = calculator_eval(ins.expression, max_chars=cfg.max_expression_chars)
            return ToolResult(tool="calculator", ok=True, purpose=ins.purpose, expression=ins.expression, result=value, elapsed_s=round(time.perf_counter() - t0, 4), source=ins.source)
        except Exception as exc:
            return ToolResult(tool="calculator", ok=False, purpose=ins.purpose, expression=ins.expression, error=str(exc), elapsed_s=round(time.perf_counter() - t0, 4), source=ins.source)
    if ins.tool == "python":
        result = run_python_code(ins.code, timeout_s=cfg.timeout_s, max_output_chars=cfg.max_output_chars, memory_mb=cfg.python_memory_mb, max_code_chars=cfg.max_code_chars)
        result.purpose = ins.purpose
        result.source = ins.source
        return result
    return ToolResult(tool=ins.tool, ok=False, purpose=ins.purpose, error="unknown tool", source=ins.source)


def build_hidden_tool_context(results: Sequence[ToolResult], cfg: LatentToolConfig) -> str:
    useful = [r for r in results if r.ok or r.error]
    if not useful:
        return ""
    observations = []
    for i, r in enumerate(useful[: cfg.max_calls]):
        item = r.compact(max(240, cfg.max_output_chars // 2))
        # Code itself is deliberately not injected.  Only the observation is.
        item.pop("code", None)
        observations.append({"id": i, **item})
    payload = {
        "policy": RUNTIME_TOOL_POLICY,
        "visibility": "private_latent_observations_only",
        "model_built_call": False,
        "requires_low_level_rl": False,
        "instructions": "Use successful observations as computed evidence. Do not emit tool-call JSON, code, schemas, or hidden tags.",
        "observations": observations,
    }
    return "<latent_tool_observations>\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n</latent_tool_observations>"


def run_latent_tools(
    *,
    prompt: str,
    latent_summary: str = "",
    raw_cot: str = "",
    sidecar_rt: Any = None,
    config: Optional[LatentToolConfig] = None,
) -> Tuple[str, Dict[str, Any]]:
    cfg = config or LatentToolConfig()
    metrics: Dict[str, Any] = {
        "enabled": bool(cfg.enabled),
        "planner": "runtime",
        "policy": RUNTIME_TOOL_POLICY,
        "model_built_call": False,
        "requires_llrl": False,
        "calculator_enabled": bool(cfg.calculator_enabled),
        "python_enabled": bool(cfg.python_enabled),
        "calls": 0,
        "ok_calls": 0,
        "tools": [],
    }
    if not cfg.enabled:
        return "", metrics
    plan = build_tool_plan(prompt=prompt, latent_summary=latent_summary, sidecar_rt=None, cfg=cfg, raw_cot=raw_cot)
    metrics["planned_calls"] = len(plan)
    results: List[ToolResult] = []
    for ins in plan[: max(0, int(cfg.max_calls))]:
        result = execute_instruction(ins, cfg)
        results.append(result)
        metrics["tools"].append({
            "tool": result.tool,
            "ok": result.ok,
            "purpose": result.purpose,
            "source": result.source,
            "elapsed_s": result.elapsed_s,
            "error": _clamp_text(result.error, 180),
        })
    metrics["calls"] = len(results)
    metrics["ok_calls"] = sum(1 for r in results if r.ok)
    context = build_hidden_tool_context(results, cfg)
    return context, metrics


class LatentToolRuntime:
    """Compatibility wrapper used by generation/browser code.

    Older code may still pass a ``sidecar_plan_text`` argument.  It is ignored:
    the runtime is the sole call builder.  This keeps tool use available without
    low-level RL, prompt-engineered function calls, or sidecar JSON planning.
    """

    @staticmethod
    def interface_system_prompt() -> str:
        return DEFAULT_TOOL_SYSTEM_PROMPT

    def __init__(self, config: Optional[LatentToolConfig] = None) -> None:
        self.config = config or LatentToolConfig()

    def planner_prompt(self, prompt: str, public_summary: str = "", raw_cot: str = "") -> str:
        # Kept only for old imports/tests.  Generation no longer calls this.
        return "Runtime latent tools are active. Model-authored tool plans are disabled."

    def run(self, prompt: str, sidecar_plan_text: str = "") -> Tuple[List[ToolResult], Dict[str, Any]]:
        cfg = self.config
        metrics: Dict[str, Any] = {
            "latent_tools_enabled": bool(cfg.enabled),
            "tool_calls": 0,
            "tool_ok_calls": 0,
            "tool_successes": 0,
            "tool_names": [],
            "planner": "runtime",
            "policy": RUNTIME_TOOL_POLICY,
            "model_built_call": False,
            "requires_llrl": False,
        }
        if not cfg.enabled:
            return [], metrics
        if sidecar_plan_text:
            metrics["ignored_external_plan_text"] = True
        plan = build_tool_plan(prompt=prompt, latent_summary="", sidecar_rt=None, cfg=cfg, raw_cot="")
        results = [execute_instruction(item, cfg) for item in plan[: max(0, int(cfg.max_calls))]]
        metrics["tool_calls"] = len(results)
        metrics["tool_ok_calls"] = sum(1 for r in results if r.ok)
        metrics["tool_successes"] = metrics["tool_ok_calls"]
        metrics["tool_names"] = [r.tool for r in results]
        if results:
            metrics["tools"] = [{"tool": r.tool, "ok": r.ok, "source": r.source, "elapsed_s": r.elapsed_s, "error": _clamp_text(r.error, 160)} for r in results]
        return results, metrics

    def hidden_context(self, results: Sequence[ToolResult]) -> str:
        return build_hidden_tool_context(results, self.config)
