from __future__ import annotations

import ast
import json
import math
import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union


@dataclass(frozen=True)
class EvalConfig:
    timeout: float
    memory_mb: int | None
    compare_mode: str
    max_failures: int
    min_test_cases: int
    verdict: str
    min_pass_rate: float
    max_pass_rate: float
    include_full_record: bool
    test_workers: int


class CyclomaticComplexityVisitor(ast.NodeVisitor):
    def __init__(self):
        self.complexity = 1

    def visit_If(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_Assert(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_IfExp(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        self.complexity += max(0, len(node.values) - 1)
        self.generic_visit(node)

    def visit_comprehension(self, node):
        self.complexity += 1
        self.complexity += len(node.ifs)
        self.generic_visit(node)


def cyclomatic_complexity(code: str) -> int | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    visitor = CyclomaticComplexityVisitor()
    visitor.visit(tree)
    return visitor.complexity


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return value

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def normalize_test_cases(value: Any) -> list[dict[str, str]]:
    parsed = parse_jsonish(value)
    if not isinstance(parsed, list):
        return []

    cases: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        if "input" not in item or "output" not in item:
            continue
        cases.append(
            {
                "input": str(item.get("input", "")),
                "output": str(item.get("output", "")),
            }
        )
    return cases


def compare_output(actual: str, expected: str, mode: str = "tokens") -> bool:
    actual = actual.replace("\r\n", "\n").replace("\r", "\n")
    expected = expected.replace("\r\n", "\n").replace("\r", "\n")

    if mode == "exact":
        return actual.strip() == expected.strip()
    if mode == "line":
        actual_lines = [" ".join(line.split()) for line in actual.strip().splitlines()]
        expected_lines = [" ".join(line.split()) for line in expected.strip().splitlines()]
        return actual_lines == expected_lines

    return actual.split() == expected.split()


def limit_child_process(memory_mb: int | None, timeout: float):
    if os.name != "posix":
        return None
    if sys.platform == "darwin":
        memory_mb = None
    if threading.current_thread() is not threading.main_thread():
        return None

    def set_limits() -> None:
        import resource

        os.setsid()
        cpu_seconds = max(1, int(math.ceil(timeout)) + 1)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 * 1024, 32 * 1024 * 1024))
        if memory_mb is not None:
            memory_bytes = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

    return set_limits


def run_one_case(
    code_path: Path,
    input_text: str,
    expected_output: str,
    config: EvalConfig,
    workdir: Path,
) -> tuple[bool, str]:
    if input_text and not input_text.endswith("\n"):
        input_text += "\n"

    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
    }

    process = subprocess.Popen(
        [sys.executable, str(code_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=workdir,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        preexec_fn=limit_child_process(config.memory_mb, config.timeout),
        # NOTE: do not set start_new_session here. limit_child_process() already
        # calls os.setsid() in the child (when on the main thread); setting
        # start_new_session=True would make CPython call setsid() a second time
        # -> EPERM raised inside preexec_fn. When set_limits is skipped (off the
        # main thread) the child is simply not a group leader, and the timeout
        # handler below falls back to process.kill().
    )

    try:
        stdout, stderr = process.communicate(input_text, timeout=config.timeout)
    except subprocess.TimeoutExpired:
        # Kill the child. If set_limits put it in its OWN process group (it called
        # os.setsid()), nuke the whole group so any grandchildren die too. But when
        # set_limits was skipped (off the main thread) the child shares our process
        # group -> killpg(getpgid) would SIGKILL this eval process itself, so in that
        # case fall back to killing the single pid. Never let the kill crash the eval.
        try:
            if os.name == "posix":
                child_pgid = os.getpgid(process.pid)
                if child_pgid != os.getpgrp():   # child has its own group -> safe to killpg
                    os.killpg(child_pgid, signal.SIGKILL)
                else:                            # shares our group -> kill only the child
                    process.kill()
            else:
                process.kill()
        except (ProcessLookupError, PermissionError, OSError):
            try:
                process.kill()
            except OSError:
                pass
        try:
            stdout, stderr = process.communicate(timeout=config.timeout)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        return False, "timeout"

    ok = process.returncode == 0 and compare_output(stdout, expected_output, config.compare_mode)
    if ok:
        return True, ""

    detail = {
        "returncode": process.returncode,
        "stdout": stdout.strip()[:500],
        "expected": expected_output.strip()[:500],
        "stderr": stderr.strip()[:500],
    }
    return False, json.dumps(detail, ensure_ascii=False)


def _run_indexed_case(
    task: tuple[int, dict[str, str], Path, EvalConfig, Path],
) -> tuple[int, bool, str]:
    case_index, case, code_path, config, workdir = task
    ok, detail = run_one_case(
        code_path=code_path,
        input_text=case["input"],
        expected_output=case["output"],
        config=config,
        workdir=workdir,
    )
    return case_index, ok, detail


def _evaluate_cases_parallel(
    code_path: Path,
    test_cases: list[dict[str, str]],
    config: EvalConfig,
    workdir: Path,
) -> list[tuple[int, bool, str]]:
    workers = max(1, min(int(config.test_workers or 1), len(test_cases)))
    tasks = [(i, case, code_path, config, workdir) for i, case in enumerate(test_cases)]
    if workers == 1:
        return [_run_indexed_case(task) for task in tasks]
    start_methods = multiprocessing.get_all_start_methods()
    mp_context = multiprocessing.get_context("fork") if "fork" in start_methods else None
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_context) as executor:
        return list(executor.map(_run_indexed_case, tasks))


def evaluate_code(code: str, test_cases: list[dict[str, str]], config: EvalConfig) -> dict[str, Any]:
    if not code.strip():
        return {
            "passed_tests": 0,
            "total_tests": len(test_cases),
            "pass_rate": 0.0,
            "failures": [{"index": 0, "detail": "empty code"}] if test_cases else [],
        }

    passed = 0
    failures: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="cf_eval_") as tmp:
        workdir = Path(tmp)
        code_path = workdir / "submission.py"
        code_path.write_text(code, encoding="utf-8")

        results = _evaluate_cases_parallel(code_path, test_cases, config, workdir)
        for case_index, ok, detail in sorted(results, key=lambda item: item[0]):
            if ok:
                passed += 1
            elif len(failures) < config.max_failures:
                failures.append({"index": case_index, "detail": detail})

    total = len(test_cases)
    return {
        "passed_tests": passed,
        "total_tests": total,
        "pass_rate": passed / total if total else 0.0,
        "failures": failures,
    }


def calculate_pass_rate(
    code: str,
    test_cases: Any,
    timeout: float = 2.0,
    memory_mb: int | None = 512,
    compare_mode: str = "tokens",
    max_failures: int = 5,
    skip_truncated_outputs: bool = False,
    test_workers: int = 32,
) -> dict[str, Any]:
    normalized_cases = normalize_test_cases(test_cases)
    if skip_truncated_outputs:
        normalized_cases = [
            case
            for case in normalized_cases
            if "..." not in case.get("output", "")
        ]
    config = EvalConfig(
        timeout=timeout,
        memory_mb=memory_mb,
        compare_mode=compare_mode,
        max_failures=max_failures,
        min_test_cases=0,
        verdict="",
        min_pass_rate=0.0,
        max_pass_rate=1.0,
        include_full_record=False,
        test_workers=max(1, int(test_workers or 1)),
    )
    return evaluate_code(code, normalized_cases, config)


def estimate_pass_at_k(
    num_samples: Union[int, list[int], Any],
    num_correct: Union[list[int], Any],
    k: int,
) -> Any:
    """Estimate per-problem pass@k from sample counts and correct counts.

    Uses the unbiased estimator ``1 - comb(n - c, k) / comb(n, k)`` without
    materializing large combinations.
    """

    def estimator(n: int, c: int, k_value: int) -> float:
        if n < k_value:
            return float("nan")
        if n - c < k_value:
            return 1.0
        product = 1.0
        for value in range(n - c + 1, n + 1):
            product *= 1.0 - k_value / value
        return 1.0 - product

    if isinstance(num_samples, int):
        import itertools

        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    values = [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    try:
        import numpy as np

        return np.array(values)
    except ImportError:
        return values


def should_prefilter(record: dict[str, Any], config: EvalConfig) -> tuple[bool, list[dict[str, str]]]:
    if str(record.get("verdict", "")) != config.verdict:
        return False, []

    test_cases = normalize_test_cases(record.get("test_cases"))
    if len(test_cases) <= config.min_test_cases:
        return False, []

    return True, test_cases


def make_output_record(
    dataset_index: int,
    record: dict[str, Any],
    test_cases: list[dict[str, str]],
    score: dict[str, Any],
    config: EvalConfig,
) -> dict[str, Any]:
    code = record.get("code")
    base = dict(record) if config.include_full_record else {}
    base.update(
        {
            "dataset_index": dataset_index,
            "submission_id": record.get("submission_id"),
            "contestId": record.get("contestId"),
            "index": record.get("index"),
            "name": record.get("name"),
            "verdict": record.get("verdict"),
            "programmingLanguage": record.get("programmingLanguage"),
            "code": code,
            "test_cases": test_cases,
            "passed_tests": score["passed_tests"],
            "total_tests": score["total_tests"],
            "pass_rate": score["pass_rate"],
            "cyclomatic_complexity": cyclomatic_complexity(str(code or "")),
            "failures": score["failures"],
        }
    )
    return base


def evaluate_record(task: tuple[int, dict[str, Any], EvalConfig]) -> dict[str, Any] | None:
    dataset_index, record, config = task
    keep, test_cases = should_prefilter(record, config)
    if not keep:
        return None

    score = evaluate_code(str(record.get("code", "")), test_cases, config)
    if not (config.min_pass_rate <= score["pass_rate"] <= config.max_pass_rate):
        return None

    return make_output_record(dataset_index, record, test_cases, score, config)
