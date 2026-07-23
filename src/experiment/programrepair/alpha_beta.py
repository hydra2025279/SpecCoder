
from __future__ import annotations

import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_THIS = Path(__file__).resolve()
_WORKSPACE_ROOT = next((p for p in _THIS.parents if (p / "SpecCoder").is_dir()), _THIS.parents[5])
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from SpecCoder.spec_utils import (  # noqa: E402
    build_full_code,
    extract_assertions_from_annotated_code,
    extract_first_python_block,
)
from SpecCoder.src.codeforces_eval import (  # noqa: E402
    EvalConfig,
    normalize_test_cases,
    run_one_case,
)
from SpecCoder.trace_collection import (  # noqa: E402
    instrument_assertions_in_completion,
    run_one_test_trace_subprocess_stdin,
)

DEFAULT_PARALLELISM = 16


@dataclass
class SpecAlphaBeta:

    trace_key: str
    assert_key: str
    expr: str
    checkpoint_line: int
    alpha: float
    beta: float
    alpha_reach: int          # checkpoint EXECUTIONS over passing tests (denominator)
    alpha_total: int          # number of passing test cases
    beta_reach: int           # checkpoint EXECUTIONS over failing tests (denominator)
    beta_total: int           # number of failing test cases
    alpha_pass: int = 0       # satisfying EXECUTIONS over passing tests (numerator)
    beta_pass: int = 0        # satisfying EXECUTIONS over failing tests (numerator)

    @property
    def discriminative_score(self) -> float:
        return self.alpha - self.beta

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_key": self.trace_key,
            "expr": self.expr,
            "checkpoint_line": self.checkpoint_line,
            "alpha": round(self.alpha, 4),
            "beta": round(self.beta, 4),
            "alpha_pass": self.alpha_pass,
            "alpha_reach": self.alpha_reach,
            "alpha_total": self.alpha_total,
            "beta_pass": self.beta_pass,
            "beta_reach": self.beta_reach,
            "beta_total": self.beta_total,
        }


@dataclass
class AlphaBetaResult:
    specs: list[SpecAlphaBeta] = field(default_factory=list)
    violated: list[SpecAlphaBeta] = field(default_factory=list)
    num_pass_tests: int = 0
    num_fail_tests: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_pass_tests": self.num_pass_tests,
            "num_fail_tests": self.num_fail_tests,
            "specs": [s.to_dict() for s in self.specs],
            "violated": [s.to_dict() for s in self.violated],
        }


def split_pass_fail_tests(
    code: str,
    test_cases: list[dict[str, Any]] | Any,
    timeout: float = 6.0,
    memory_mb: int | None = 512,
    compare_mode: str = "tokens",
    parallelism: int = DEFAULT_PARALLELISM,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Run ``code`` on every test and partition into (passed, failed)."""
    cases = normalize_test_cases(test_cases)
    if not cases:
        return [], []
    config = EvalConfig(
        timeout=timeout, memory_mb=memory_mb, compare_mode=compare_mode,
        max_failures=len(cases), min_test_cases=0, verdict="",
        min_pass_rate=0.0, max_pass_rate=1.0, include_full_record=False,
        test_workers=max(1, int(parallelism or 1)),
    )
    with tempfile.TemporaryDirectory(prefix="cf_ab_") as tmp:
        workdir = Path(tmp)
        code_path = workdir / "submission.py"
        code_path.write_text(code, encoding="utf-8")

        def _run(case: dict[str, str]) -> bool:
            ok, _detail = run_one_case(
                code_path, case.get("input", "") or "", case.get("output", "") or "",
                config, workdir,
            )
            return ok

        with ThreadPoolExecutor(max_workers=max(1, parallelism)) as ex:
            oks = list(ex.map(_run, cases))

    passed = [c for c, ok in zip(cases, oks) if ok]
    failed = [c for c, ok in zip(cases, oks) if not ok]
    return passed, failed


def _trace_correctness(
    full_code: str,
    spec_dict: dict[str, Any],
    tests: list[dict[str, str]],
    timeout: int,
    parallelism: int,
) -> dict[str, dict[str, Any]]:
    trace_keys = list(spec_dict)
    reach_counts = {k: 0 for k in trace_keys}
    pass_counts = {k: 0 for k in trace_keys}
    if tests and trace_keys:
        def _trace(test: dict[str, str]):
            return run_one_test_trace_subprocess_stdin(
                full_code, spec_dict, test.get("input", "") or "", timeout,
                trace_keys=trace_keys,
            )

        with ThreadPoolExecutor(max_workers=max(1, parallelism)) as ex:
            for result, timed_out, _err in ex.map(_trace, tests):
                if timed_out or result is None:
                    continue
                for key in trace_keys:
                    info = result.get(key, {})
                    reach_counts[key] += info.get("reach_count", 0)   # every execution
                    pass_counts[key] += info.get("pass_count", 0)     # ... that satisfied

    out: dict[str, dict[str, Any]] = {}
    for key in trace_keys:
        reach = reach_counts[key]
        passed = pass_counts[key]
        out[key] = {
            "reach_count": reach,
            "pass_count": passed,
            "correctness": round(passed / reach, 4) if reach > 0 else 0.0,
        }
    return out


def compute_alpha_beta(
    annotated_code: str,
    test_cases: list[dict[str, Any]] | Any,
    alpha_threshold: float = 0.9,
    beta_threshold: float = 1.0,
    timeout: int = 6,
    memory_mb: int | None = 512,
    parallelism: int = DEFAULT_PARALLELISM,
    pass_fail: tuple[list[dict[str, str]], list[dict[str, str]]] | None = None,
) -> AlphaBetaResult:
    cleaned, assertions = extract_assertions_from_annotated_code(annotated_code)
    if pass_fail is not None:
        passed, failed = pass_fail
    else:
        passed, failed = split_pass_fail_tests(
            cleaned, test_cases, timeout=timeout, memory_mb=memory_mb, parallelism=parallelism
        )
    if not assertions:
        return AlphaBetaResult(num_pass_tests=len(passed), num_fail_tests=len(failed))

    specs_by_tk = {a.trace_key: {"assert_key": a.assert_key, "expr": a.expr} for a in assertions}
    instrumented = instrument_assertions_in_completion(annotated_code, assertions)
    full_code, _ = build_full_code("", instrumented)

    alpha_map = _trace_correctness(full_code, specs_by_tk, passed, timeout, parallelism)
    beta_map = _trace_correctness(full_code, specs_by_tk, failed, timeout, parallelism)

    specs: list[SpecAlphaBeta] = []
    for a in assertions:
        am = alpha_map.get(a.trace_key, {})
        bm = beta_map.get(a.trace_key, {})
        alpha_reach = int(am.get("reach_count", 0))
        beta_reach = int(bm.get("reach_count", 0))
        # 0/0 conventions (alpha/beta repair signal ONLY — specgen metrics unaffected):
        #   alpha: no passing execution reaches -> vacuously holds  -> 1.0
        #   beta : no failing execution reaches -> nothing violates -> 0.0
        alpha = 1.0 if alpha_reach == 0 else float(am.get("correctness", 0.0))
        beta = 0.0 if beta_reach == 0 else float(bm.get("correctness", 0.0))
        specs.append(SpecAlphaBeta(
            trace_key=a.trace_key, assert_key=a.assert_key, expr=a.expr,
            checkpoint_line=a.assertion_line_index,
            alpha=alpha, beta=beta,
            alpha_reach=alpha_reach, alpha_total=len(passed),
            beta_reach=beta_reach, beta_total=len(failed),
            alpha_pass=int(am.get("pass_count", 0)), beta_pass=int(bm.get("pass_count", 0)),
        ))

    violated = select_violated_specs(specs, alpha_threshold, beta_threshold)
    return AlphaBetaResult(
        specs=specs, violated=violated,
        num_pass_tests=len(passed), num_fail_tests=len(failed),
    )


def select_violated_specs(
    specs: list[SpecAlphaBeta],
    alpha_threshold: float = 0.9,
    beta_threshold: float = 1.0,
) -> list[SpecAlphaBeta]:
    out = [
        s for s in specs
        if s.alpha >= alpha_threshold
        and s.beta < beta_threshold
        and (s.alpha_reach > 0 or s.beta_reach > 0)  # exclude fully-dead checkpoints
    ]
    out.sort(key=lambda s: (s.beta, s.checkpoint_line, s.expr))
    return out


def specs_from_dicts(spec_dicts: list[dict[str, Any]]) -> list[SpecAlphaBeta]:
    out: list[SpecAlphaBeta] = []
    for d in spec_dicts or []:
        out.append(SpecAlphaBeta(
            trace_key=str(d.get("trace_key", "")),
            assert_key=str(d.get("assert_key", "")),
            expr=str(d.get("expr", "")),
            checkpoint_line=int(d.get("checkpoint_line", 0) or 0),
            alpha=float(d.get("alpha", 0.0) or 0.0),
            beta=float(d.get("beta", 0.0) or 0.0),
            alpha_reach=int(d.get("alpha_reach", 0) or 0),
            alpha_total=int(d.get("alpha_total", 0) or 0),
            beta_reach=int(d.get("beta_reach", 0) or 0),
            beta_total=int(d.get("beta_total", 0) or 0),
            alpha_pass=int(d.get("alpha_pass", 0) or 0),
            beta_pass=int(d.get("beta_pass", 0) or 0),
        ))
    return out


def reselect_from_saved(
    spec_dicts: list[dict[str, Any]],
    alpha_threshold: float,
    beta_threshold: float,
    num_pass_tests: int = 0,
    num_fail_tests: int = 0,
) -> AlphaBetaResult:
    """Re-threshold saved per-spec α/β at NEW (α, β) without recomputation. Returns an
    :class:`AlphaBetaResult` with the same ``specs`` and the newly selected ``violated``
    (one per checkpoint), so the caller can both build the repair feedback and re-store
    the result for further ablation sweeps."""
    specs = specs_from_dicts(spec_dicts)
    violated = one_spec_per_checkpoint(select_violated_specs(specs, alpha_threshold, beta_threshold))
    return AlphaBetaResult(
        specs=specs, violated=violated,
        num_pass_tests=int(num_pass_tests or 0), num_fail_tests=int(num_fail_tests or 0),
    )


def one_spec_per_checkpoint(violated: list[SpecAlphaBeta]) -> list[SpecAlphaBeta]:
    """Keep only the lowest-beta spec at each checkpoint line."""
    best_by_line: dict[int, SpecAlphaBeta] = {}
    for s in violated:
        cur = best_by_line.get(s.checkpoint_line)
        if cur is None or s.beta < cur.beta:
            best_by_line[s.checkpoint_line] = s
    out = list(best_by_line.values())
    out.sort(key=lambda s: (s.beta, s.checkpoint_line, s.expr))
    return out


@dataclass
class SpecReachSatisfy:
    trace_key: str
    assert_key: str
    expr: str
    checkpoint_line: int
    reach: int           # total checkpoint EXECUTIONS across all tests (denominator)
    satisfy: int         # executions that SATISFIED the assertion (numerator)
    num_tests: int       # number of test cases executed

    @property
    def is_violated(self) -> bool:
        # Reached by some execution and NOT satisfied by all of them.
        return self.reach > 0 and self.satisfy < self.reach

    @property
    def satisfy_rate(self) -> float:
        return (self.satisfy / self.reach) if self.reach > 0 else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_key": self.trace_key,
            "expr": self.expr,
            "checkpoint_line": self.checkpoint_line,
            "reach": self.reach,
            "satisfy": self.satisfy,
            "num_tests": self.num_tests,
            "satisfy_rate": round(self.satisfy_rate, 4),
            "violated": self.is_violated,
        }


@dataclass
class ReachSatisfyResult:
    specs: list[SpecReachSatisfy] = field(default_factory=list)
    violated: list[SpecReachSatisfy] = field(default_factory=list)
    num_tests: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_tests": self.num_tests,
            "specs": [s.to_dict() for s in self.specs],
            "violated": [s.to_dict() for s in self.violated],
        }


def compute_reach_satisfy(
    annotated_code: str,
    test_cases: list[dict[str, Any]] | Any,
    timeout: int = 6,
    memory_mb: int | None = 512,
    parallelism: int = DEFAULT_PARALLELISM,
) -> ReachSatisfyResult:
    cleaned, assertions = extract_assertions_from_annotated_code(annotated_code)
    cases = normalize_test_cases(test_cases)
    if not assertions:
        return ReachSatisfyResult(num_tests=len(cases))

    specs_by_tk = {a.trace_key: {"assert_key": a.assert_key, "expr": a.expr} for a in assertions}
    instrumented = instrument_assertions_in_completion(annotated_code, assertions)
    full_code, _ = build_full_code("", instrumented)
    rs_map = _trace_correctness(full_code, specs_by_tk, cases, timeout, parallelism)

    specs: list[SpecReachSatisfy] = []
    for a in assertions:
        info = rs_map.get(a.trace_key, {})
        specs.append(SpecReachSatisfy(
            trace_key=a.trace_key, assert_key=a.assert_key, expr=a.expr,
            checkpoint_line=a.assertion_line_index,
            reach=int(info.get("reach_count", 0)),
            satisfy=int(info.get("pass_count", 0)),
            num_tests=len(cases),
        ))

    # One violated spec per checkpoint line: keep the lowest satisfy-rate at each line.
    best_by_line: dict[int, SpecReachSatisfy] = {}
    for s in specs:
        if not s.is_violated:
            continue
        cur = best_by_line.get(s.checkpoint_line)
        if cur is None or s.satisfy_rate < cur.satisfy_rate:
            best_by_line[s.checkpoint_line] = s
    violated = sorted(best_by_line.values(), key=lambda s: (s.satisfy_rate, s.checkpoint_line, s.expr))
    return ReachSatisfyResult(specs=specs, violated=violated, num_tests=len(cases))


def reach_satisfy_from_dicts(spec_dicts: list[dict[str, Any]]) -> list[SpecReachSatisfy]:
    """Rebuild :class:`SpecReachSatisfy` objects from a saved ``reach_satisfy["specs"]`` list."""
    out: list[SpecReachSatisfy] = []
    for d in spec_dicts or []:
        out.append(SpecReachSatisfy(
            trace_key=str(d.get("trace_key", "")),
            assert_key=str(d.get("assert_key", "")),
            expr=str(d.get("expr", "")),
            checkpoint_line=int(d.get("checkpoint_line", 0) or 0),
            reach=int(d.get("reach", 0) or 0),
            satisfy=int(d.get("satisfy", 0) or 0),
            num_tests=int(d.get("num_tests", 0) or 0),
        ))
    return out


def extract_annotated(response: str) -> str | None:
    """Pull the annotated solution out of an LLM response."""
    import re

    match = re.search(r"<solution>(.*?)(?:</solution>|\Z)", response, re.DOTALL | re.IGNORECASE)
    scope = match.group(1) if match else response
    block = extract_first_python_block(scope) or extract_first_python_block(response)
    if block:
        return block
    return scope.strip() or None


if __name__ == "__main__":
    import json

    annotated = (
        "a, b = map(int, input().split())\n"
        "# assert b >= 0\n"
        "c = a - b\n"
        "# assert c == a + b\n"
        "print(c)\n"
    )
    tests = [
        {"input": "5 0", "output": "5"},
        {"input": "10 0", "output": "10"},
        {"input": "5 3", "output": "8"},
        {"input": "7 2", "output": "9"},
    ]
    res = compute_alpha_beta(annotated, tests, alpha_threshold=0.9, beta_threshold=1.0, memory_mb=None)
    print(json.dumps(res.to_dict(), indent=2))
    assert res.num_pass_tests == 2 and res.num_fail_tests == 2, res.to_dict()
    exprs = {s.expr for s in res.violated}
    assert "c == a + b" in exprs and "b >= 0" not in exprs
    print("\nOK: alpha/beta smoke test passed")
