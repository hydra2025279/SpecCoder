from __future__ import annotations

import ast
import asyncio
import datetime
import json
import logging
import logging.handlers
import multiprocessing
import os
import queue as queue_mod
import re
import signal
import sys
import threading
from dataclasses import dataclass
from typing import Any, Optional

from tqdm import tqdm


IMPORT_STRING = (
    "import sys\nfrom typing import *\nimport collections\nimport math\n"
    "import itertools\nimport functools\nimport heapq\nimport bisect\n"
    "import string\nimport re\nimport json\nfrom collections import *\n"
    "from itertools import *\nfrom functools import *\nfrom heapq import *\n"
    "from bisect import *\nfrom math import *\nsys.setrecursionlimit(50000)\n"
)

SPEC_CHECK_HEADER_TRACE = """\
_SPEC_TRACE = {}

def _check_trace(line_id, expr_fn):
    try:
        result = bool(expr_fn())
        _SPEC_TRACE.setdefault(line_id, []).append(result)
    except TimeoutError:
        _SPEC_TRACE.setdefault(line_id, []).append(False)
        raise
    except Exception:
        _SPEC_TRACE.setdefault(line_id, []).append(False)
"""


@dataclass
class AssertionInfo:
    assert_key: str
    expr: str
    dedup_key: int
    indent: int = 0
    line_content: str = ""
    assertion_line_index: int = -1

    @property
    def trace_key(self) -> str:
        return f"{self.dedup_key}:{self.assert_key}"

    def to_record(self) -> dict[str, Any]:
        return {
            "assert_key": self.assert_key,
            "dedup_key": self.dedup_key,
            "trace_key": self.trace_key,
            "expr": self.expr,
            "indent": self.indent,
            "line_content": self.line_content,
            "assertion_line_index": self.assertion_line_index,
        }


MUTATING_METHOD_NAMES = {
    "__delitem__",
    "__iadd__",
    "__imul__",
    "__ior__",
    "__isub__",
    "__setitem__",
    "add",
    "append",
    "appendleft",
    "clear",
    "difference_update",
    "discard",
    "extend",
    "extendleft",
    "heapify",
    "heappop",
    "heappush",
    "heappushpop",
    "heapreplace",
    "insert",
    "intersection_update",
    "pop",
    "popitem",
    "popleft",
    "remove",
    "reverse",
    "rotate",
    "setdefault",
    "shuffle",
    "sort",
    "symmetric_difference_update",
    "update",
}

DANGEROUS_CALL_NAMES = {
    "__import__",
    "compile",
    "delattr",
    "eval",
    "exec",
    "globals",
    "input",
    "locals",
    "next",
    "open",
    "print",
    "setattr",
    "vars",
    "heapify",
    "heappop",
    "heappush",
    "heappushpop",
    "heapreplace",
}

DANGEROUS_MODULE_NAMES = {
    "heapq",
    "os",
    "pathlib",
    "random",
    "shutil",
    "subprocess",
    "sys",
    "time",
}


def find_assertion_safety_violations(expr: str) -> list[str]:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return []

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.NamedExpr):
            violations.append("assignment expression `:=` is not allowed")
        elif isinstance(node, (ast.Yield, ast.YieldFrom)):
            violations.append("yield expressions are not allowed")
        elif isinstance(node, ast.Await):
            violations.append("await expressions are not allowed")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id in DANGEROUS_CALL_NAMES:
                    violations.append(f"dangerous call `{func.id}(...)`")
                if func.id in DANGEROUS_MODULE_NAMES:
                    violations.append(f"dangerous module call `{func.id}(...)`")
            elif isinstance(func, ast.Attribute):
                attr = func.attr
                if attr in MUTATING_METHOD_NAMES:
                    violations.append(f"mutating method call `.{attr}(...)`")
                root = func
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name) and root.id in DANGEROUS_MODULE_NAMES:
                    violations.append(f"dangerous module call `{root.id}.{attr}(...)`")

    return sorted(set(violations))


def find_unsafe_assertions(assertions: list[AssertionInfo]) -> dict[str, list[str]]:
    unsafe: dict[str, list[str]] = {}
    for assertion in assertions:
        violations = find_assertion_safety_violations(assertion.expr)
        if violations:
            unsafe[assertion.trace_key] = violations
    return unsafe


def _log_json_default(value: Any) -> Any:
    if hasattr(value, "val") and hasattr(value, "next"):
        out = []
        cur = value
        while cur is not None:
            out.append(cur.val)
            cur = cur.next
        return out
    if hasattr(value, "val") and hasattr(value, "left") and hasattr(value, "right"):
        return f"<TreeNode {value.val!r}>"
    if isinstance(value, set):
        return sorted(value)
    return repr(value)


def append_jsonl_log(path: str | os.PathLike[str], event: dict[str, Any]) -> None:
    """Append one timestamped JSON object to a detailed per-task log file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "time": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds"),
        **event,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=_log_json_default) + "\n")
        f.flush()


def make_logger(log_path: str, name: str = "leetcode_inline") -> logging.Logger:
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    max_bytes = int(os.environ.get("LCB_LOG_MAX_BYTES", str(10 * 1024 * 1024)))
    backup_count = int(os.environ.get("LCB_LOG_BACKUP_COUNT", "5"))
    fh = logging.handlers.RotatingFileHandler(
        log_path, mode="a", encoding="utf-8", maxBytes=max_bytes, backupCount=backup_count
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    class TqdmHandler(logging.StreamHandler):
        def emit(self, record):
            tqdm.write(self.format(record))

    ch = TqdmHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    logger.propagate = False
    return logger


def log_response(logger: logging.Logger, step: str, task_id: str, response: str) -> None:
    sep = "═" * 64
    logger.debug(
        f"\n╔{sep}╗\n"
        f"║ STEP={step} TASK={task_id}\n"
        f"╠{sep}╣\n"
        f"║ [RAW LLM RESPONSE]\n"
        f"╚{sep}╝\n"
        f"{response}\n"
        f"── END STEP={step} TASK={task_id} {'─' * 24}"
    )


def extract_first_python_block(text: str) -> Optional[str]:
    parts = re.split(r"```python\s*\n?", text)
    if len(parts) < 2:
        return None
    block = parts[1]
    end = block.find("```")
    return (block[:end] if end != -1 else block).strip() or None


def extract_json_block(text: str) -> Optional[str]:
    parts = re.split(r"```(?:json|jsonl)\s*\n?", text, flags=re.IGNORECASE)
    if len(parts) >= 2:
        block = parts[-1]
        end = block.find("```")
        candidate = (block[:end] if end != -1 else block).strip()
        if candidate:
            return candidate
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group(0) if match else None


ASSERT_RE = re.compile(r"^#\s*assert\s+", re.IGNORECASE)


def parse_assert_comment(line: str) -> Optional[str]:
    stripped = line.strip()
    if not ASSERT_RE.match(stripped):
        return None
    body = ASSERT_RE.sub("", stripped).strip()
    body = re.sub(r"\s{2,}#.*$", "", body).strip()
    body = re.sub(r",\s*[\"'][^\"']*[\"']$", "", body).strip()
    return body if body else None


def extract_assertions_from_annotated_code(annotated_completion: str) -> tuple[str, list[AssertionInfo]]:
    lines = annotated_completion.split("\n")
    cleaned: list[str] = []
    assertions: list[AssertionInfo] = []
    seen: dict[str, int] = {}
    for i, line in enumerate(lines, start=1):
        expr = parse_assert_comment(line)
        if expr is None:
            cleaned.append(line)
            continue
        base_key = f"assert {expr}"
        dedup_key = seen.get(base_key, 0)
        seen[base_key] = dedup_key + 1
        assertions.append(
            AssertionInfo(
                assert_key=base_key,
                expr=expr,
                dedup_key=dedup_key,
                indent=len(line) - len(line.lstrip()),
                line_content=line,
                assertion_line_index=i - 1,
            )
        )
    return "\n".join(cleaned), assertions


def normalize_code(code: str) -> str:
    return "\n".join(line.rstrip() for line in code.strip().splitlines())


def _collect_assignment_targets(node: ast.AST) -> set[str]:
    names: set[str] = set()
    if isinstance(node, ast.Name):
        names.add(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            names.update(_collect_assignment_targets(elt))
    elif isinstance(node, ast.Starred):
        names.update(_collect_assignment_targets(node.value))
    return names


def _func_param_names(func: ast.AST) -> set[str]:
    names: set[str] = set()
    if not hasattr(func, "args"):
        return names
    args = func.args
    for arg in list(getattr(args, "posonlyargs", [])) + list(args.args) + list(args.kwonlyargs):
        if arg.arg != "self":
            names.add(arg.arg)
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    return names


def _iter_func_body_no_nested(func: ast.AST):
    def _walk(node: ast.AST):
        for child in ast.iter_child_nodes(node):
            yield child
            if not isinstance(
                child,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef),
            ):
                yield from _walk(child)
    yield from _walk(func)


def _extract_bound_names(stmt: ast.AST) -> set[str]:
    names: set[str] = set()
    if isinstance(stmt, ast.Assign):
        for tgt in stmt.targets:
            names.update(_collect_assignment_targets(tgt))
    elif isinstance(stmt, (ast.AugAssign, ast.AnnAssign)):
        names.update(_collect_assignment_targets(stmt.target))
    elif isinstance(stmt, ast.For):
        names.update(_collect_assignment_targets(stmt.target))
    elif isinstance(stmt, ast.With):
        for item in stmt.items:
            if item.optional_vars:
                names.update(_collect_assignment_targets(item.optional_vars))
    elif isinstance(stmt, ast.NamedExpr):
        names.update(_collect_assignment_targets(stmt.target))
    return names


def _find_enclosing_funcs(tree: ast.AST, target_line: int) -> list[ast.AST]:
    candidates: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", None) or node.lineno
            if node.lineno <= target_line <= end:
                candidates.append(node)
    candidates.sort(key=lambda f: (f.lineno, -(getattr(f, "end_lineno", f.lineno) or f.lineno)))
    return candidates


def _collect_in_scope_vars(tree: ast.AST, target_line: int) -> set[str]:
    enclosing = _find_enclosing_funcs(tree, target_line)
    if not enclosing:
        vars_in_scope: set[str] = set()
        for stmt in _iter_func_body_no_nested(tree):
            lineno = getattr(stmt, "lineno", None)
            if lineno is None or lineno >= target_line:
                continue
            vars_in_scope.update(_extract_bound_names(stmt))
        return vars_in_scope
    vars_in_scope = set()
    for func in enclosing:
        vars_in_scope.update(_func_param_names(func))
        for stmt in _iter_func_body_no_nested(func):
            lineno = getattr(stmt, "lineno", None)
            if lineno is None or lineno >= target_line:
                continue
            vars_in_scope.update(_extract_bound_names(stmt))
    return vars_in_scope


def _expr_referenced_names(expr: str) -> set[str]:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return set()
    names: set[str] = set()

    class _Visitor(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Load):
                names.add(node.id)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            self.visit(node.value)

    _Visitor().visit(tree)
    return names


def find_specs_missing_variable_refs(
    annotated_completion: str,
) -> list[tuple[int, str, set[str]]]:
    lines = annotated_completion.split("\n")
    asserts: list[tuple[int, str]] = []
    for i, line in enumerate(lines, start=1):
        expr = parse_assert_comment(line)
        if expr is not None:
            asserts.append((i, expr))
    if not asserts:
        return []
    try:
        tree = ast.parse(annotated_completion)
    except SyntaxError:
        return []
    violations: list[tuple[int, str, set[str]]] = []
    for line_no, expr in asserts:
        required = _collect_in_scope_vars(tree, line_no)
        required.discard("self")
        used = _expr_referenced_names(expr)
        missing = required - used
        if missing:
            violations.append((line_no, expr, missing))
    return violations


def parse_func_name(entry_point: str, completion: str) -> str:
    match = re.search(r"Solution\(\)\.([A-Za-z_][A-Za-z0-9_]*)", entry_point or "")
    if match:
        return match.group(1)
    tree = ast.parse(completion)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Solution":
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    return item.name
    raise ValueError("Could not determine Solution method name")


def get_method_params(completion: str, func_name: str) -> list[str]:
    tree = ast.parse(completion)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Solution":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == func_name:
                    return [arg.arg for arg in item.args.args if arg.arg != "self"]
    return []


def format_examples(row: dict[str, Any], max_examples: int = 3) -> str:
    """Render the problem's worked I/O examples (the official ``demo-input`` /
    ``demo-output`` from the statement; falls back to the first real test cases)."""
    demo_inputs = row.get("demo-input") or []
    demo_outputs = row.get("demo-output") or []
    parts = []
    for i, (inp, out) in enumerate(zip(demo_inputs, demo_outputs)):
        if i >= max_examples:
            break
        parts.append(f"Example {i + 1} input:\n{str(inp).strip()}\n"
                     f"Example {i + 1} output:\n{str(out).strip()}")
    if not parts:
        for i, case in enumerate((row.get("test_cases") or [])[:max_examples]):
            parts.append(f"Example {i + 1} input:\n{str(case.get('input', '')).strip()}\n"
                         f"Example {i + 1} output:\n{str(case.get('output', '')).strip()}")
    return "\n\n".join(parts)


def build_problem_text(row: dict[str, Any], include_examples: bool = False, max_examples: int = 3) -> str:
    """Compose the problem statement from the codeforces spec blocks:
    Problem Description, Input Specification, Output Specification, and (when
    ``include_examples``) the worked Examples.

    This is the canonical model-facing problem text shared by the codegen,
    specgen and program-repair experiments. Empty blocks are skipped.
    """
    blocks = [
        ("Problem Description", row.get("problem-description")),
        ("Input Specification", row.get("input-specification")),
        ("Output Specification", row.get("output-specification")),
    ]
    parts = []
    for title, content in blocks:
        content = str(content or "").strip()
        if content:
            parts.append(f"### {title}\n{content}")
    if include_examples:
        examples = format_examples(row, max_examples)
        if examples:
            parts.append(f"### Examples\n{examples}")
    return "\n\n".join(parts)


def verdict_passes(row: dict[str, Any], allowed: set[str] | None) -> bool:
    """Return True if the row's ``verdict`` is in ``allowed`` (or no filter)."""
    if not allowed:
        return True
    return str(row.get("verdict", "")) in allowed


def normalize_codeforces_row(row: dict[str, Any]) -> dict[str, Any]:
    """Map a codeforces record to the field names the LeetCode pipeline expects.

    Codeforces solutions are stdin/stdout scripts (no ``class Solution``), so
    ``func_name``/``params`` are empty and tests are ``[{input, output}, ...]``.
    The original fields are preserved; only the shared aliases are added.

    ``problem_description`` is built from the three spec blocks (Problem /
    Input / Output Specification) so the model sees a clean, consistent
    statement across codegen, specgen and repair.
    """
    if row.get("contestId") is not None and row.get("index") is not None:
        task_id = f"cf_{row.get('contestId')}{row.get('index')}"
    else:
        idx = row.get("dataset_index")
        task_id = f"cf_{idx}"
    problem_text = build_problem_text(row) or (row.get("prompt", "") or "")
    # Paired schema exposes ``correct_solution`` (used by specgen); fall back to
    # the legacy single-submission ``code`` field.
    completion = row.get("correct_solution") or row.get("code", "") or ""
    return {
        **row,
        "task_id": task_id,
        "question_id": task_id,
        "completion": completion,
        "problem_description": problem_text,
        "tests": row.get("test_cases", []) or [],
        "func_name": None,        # stdin/stdout program — no Solution method
        "params": [],
        "entry_point": "",
        "starter_code": "",
        "dataset_type": "codeforces",
    }


def build_full_code(prompt: str, completion: str) -> tuple[str, int]:
    """Return full executable code and the line offset before `completion`.

    The dataset stores imports/helpers in `prompt` and the class solution in
    `completion`. Assertions are extracted from completion-local line numbers,
    then shifted by this offset before instrumentation.
    """
    prompt = prompt or ""
    completion = (completion or "").strip()
    if prompt:
        # Match LeetCodeDataset's official evaluator:
        # problem["prompt"] + "\n" + completion + "\n" + problem["test"].
        prefix = prompt + "\n"
        return prefix + completion + "\n", len(prefix.splitlines())
    return completion + "\n", 0


def parse_leetcode_tests(row: dict[str, Any], params: list[str]) -> list[dict[str, Any]]:
    namespace: dict[str, Any] = {}
    try:
        exec(compile(IMPORT_STRING + "\n" + (row.get("prompt") or ""), "<leetcode-prompt>", "exec"), namespace)
    except Exception:
        namespace = {}

    test_code = row.get("test") or ""
    tree = ast.parse(test_code)
    tests: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "candidate":
            continue
        values_by_param: dict[str, Any] = {}
        for idx, arg_node in enumerate(node.args):
            if idx < len(params):
                values_by_param[params[idx]] = _eval_ast_expr(arg_node, namespace)
        for keyword in node.keywords:
            if keyword.arg:
                values_by_param[keyword.arg] = _eval_ast_expr(keyword.value, namespace)
        if not all(param in values_by_param for param in params):
            continue
        args = [values_by_param[param] for param in params]
        test: dict[str, Any] = {"args": args, "testtype": "functional"}
        try:
            test["input"] = "\n".join(json.dumps(value, default=_json_default) for value in args)
        except TypeError:
            test["input"] = ""
        tests.append(test)
    return tests


def _eval_ast_expr(node: ast.AST, namespace: dict[str, Any]) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return eval(compile(ast.Expression(node), "<test-arg>", "eval"), namespace)


def _json_default(value: Any) -> Any:
    if hasattr(value, "val") and hasattr(value, "next"):
        out = []
        cur = value
        while cur is not None:
            out.append(cur.val)
            cur = cur.next
        return out
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def parse_test_args(input_str: str) -> Optional[list[Any]]:
    try:
        return [json.loads(line) for line in str(input_str).splitlines() if line.strip()]
    except Exception:
        return None


def get_test_args(test: dict[str, Any]) -> Optional[list[Any]]:
    if isinstance(test, dict) and "args" in test:
        args = test.get("args")
        return list(args) if isinstance(args, (list, tuple)) else None
    return parse_test_args(test.get("input", "") if isinstance(test, dict) else "")


SKIP_MUTATION_PATTERNS = re.compile(
    r"^\s*(#|$|class\s|def\s|import\s|from\s|@|pass\s*$)"
)


def parse_mutants_response(raw: str) -> list[dict[str, Any]]:
    json_text = extract_json_block(raw)
    if not json_text:
        return []
    try:
        obj = json.loads(json_text)
    except Exception:
        try:
            obj = ast.literal_eval(json_text)
        except Exception:
            return []
    parsed: list[dict[str, Any]] = []
    if isinstance(obj, list):
        for item in obj:
            if not isinstance(item, dict):
                continue
            original = item.get("original_content")
            variants = item.get("mutant")
            dedup_key = item.get("dedup_key", 0)
            if not isinstance(original, str) or SKIP_MUTATION_PATTERNS.match(original):
                continue
            if not isinstance(dedup_key, int):
                continue
            if isinstance(variants, list):
                clean_variants = [v for v in variants if isinstance(v, str) and v.strip()]
            elif isinstance(variants, str) and variants.strip():
                clean_variants = [variants]
            else:
                clean_variants = []
            if clean_variants:
                parsed.append({"original_content": original, "dedup_key": dedup_key, "mutant": clean_variants})
        return parsed
    if not isinstance(obj, dict):
        return []
    occurrences: dict[str, int] = {}
    for key, value in obj.items():
        if not isinstance(key, str) or SKIP_MUTATION_PATTERNS.match(key):
            continue
        if isinstance(value, list):
            variants = [item for item in value if isinstance(item, str) and item.strip()]
        elif isinstance(value, str) and value.strip():
            variants = [value]
        else:
            variants = []
        if variants:
            dedup_key = occurrences.get(key, 0)
            occurrences[key] = dedup_key + 1
            parsed.append({"original_content": key, "dedup_key": dedup_key, "mutant": variants})
    return parsed


def apply_single_mutant_record(code: str, record: dict[str, Any], mutant_line: str | None = None) -> str | None:
    original = record.get("original_content")
    dedup_key = record.get("dedup_key")
    if not isinstance(original, str) or not isinstance(dedup_key, int):
        return None
    variant = mutant_line
    if variant is None:
        variants = record.get("mutant")
        if not isinstance(variants, list) or not variants:
            return None
        variant = variants[0]
    if not isinstance(variant, str) or not variant.strip():
        return None
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    original = original.rstrip("\r")
    variant = variant.rstrip("\r")
    lines = code.split("\n")
    occurrence = 0
    for idx, line in enumerate(lines):
        if line != original:
            continue
        if occurrence == dedup_key:
            out = list(lines)
            out[idx] = variant
            return "\n".join(out)
        occurrence += 1
    return None


def find_mutant_record_line_index(code: str, record: dict[str, Any]) -> Optional[int]:
    """Find the 0-based line index for a mutant's original line occurrence."""
    original = record.get("original_content")
    dedup_key = record.get("dedup_key")
    if not isinstance(original, str) or not isinstance(dedup_key, int):
        return None
    occurrence = 0
    for idx, line in enumerate(code.split("\n")):
        if line != original:
            continue
        if occurrence == dedup_key:
            return idx
        occurrence += 1
    return None


def apply_mutant_records(code: str, records: list[dict[str, Any]], max_mutants: int = 0) -> list[str]:
    programs: list[str] = []
    for record in records:
        variants = record.get("mutant")
        if not isinstance(variants, list):
            continue
        for variant in variants:
            mutated = apply_single_mutant_record(code, record, variant)
            if mutated is None:
                continue
            programs.append(mutated)
            if max_mutants > 0 and len(programs) >= max_mutants:
                return programs
    return programs
