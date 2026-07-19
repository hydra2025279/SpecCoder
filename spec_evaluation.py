from __future__ import annotations

import multiprocessing
import os
import queue as queue_mod
import signal
import time
from typing import Any, Callable

from SpecCoder.spec_utils import get_test_args
from SpecCoder.trace_collection import (
    evaluate_one_test_trace,
    evaluate_one_test_trace_stdin,
    kill_proc,
)


def _select_tests(test_cases: list, is_codeforces: bool) -> list:
    if is_codeforces:
        return list(test_cases)
    return [test for test in test_cases if test.get("testtype") == "functional"]


MUTANT_REJECTED = "mutant_rejected"
MUTANT_ACCEPTED = "mutant_accepted"
MUTANT_SKIPPED = "mutant_skipped"



def _bounded_worker_entry(fn, args, q) -> None:
    try:
        os.setsid()
    except Exception:
        pass
    try:
        q.put(fn(*args))
    except Exception:
        q.put(None)


def _kill_group(proc) -> None:
    """Kill the child's whole process group, then the child itself."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
    kill_proc(proc)


def _run_bounded(
    jobs: list[tuple[Callable[..., Any], tuple]],
    max_workers: int,
    per_job_timeout: float,
) -> dict[int, Any]:
    results: dict[int, Any] = {}
    n = len(jobs)
    if n == 0:
        return results
    ctx = multiprocessing.get_context("fork")
    workers = max(1, int(max_workers or 1))
    running: dict[int, tuple[Any, Any, float]] = {}  # idx -> (proc, queue, deadline)
    nxt = 0
    while nxt < n or running:
        while nxt < n and len(running) < workers:
            fn, args = jobs[nxt]
            q = ctx.Queue()
            proc = ctx.Process(target=_bounded_worker_entry, args=(fn, args, q))
            proc.start()
            running[nxt] = (proc, q, time.monotonic() + per_job_timeout)
            nxt += 1

        progressed = False
        now = time.monotonic()
        for idx in list(running):
            proc, q, deadline = running[idx]
            try:
                results[idx] = q.get_nowait()
                _kill_group(proc)
                running.pop(idx)
                progressed = True
                continue
            except queue_mod.Empty:
                pass
            if not proc.is_alive():
                try:
                    results[idx] = q.get_nowait()
                except queue_mod.Empty:
                    results.setdefault(idx, None)
                _kill_group(proc)
                running.pop(idx)
                progressed = True
            elif now >= deadline:
                _kill_group(proc)
                results.setdefault(idx, None)
                running.pop(idx)
                progressed = True
        if not progressed:
            time.sleep(0.01)
    return results


def _trace_unit(
    code: str,
    spec_dict: dict[str, Any],
    func_name: str,
    params: list[str],
    test: dict[str, Any],
    timeout: int,
    is_codeforces: bool,
) -> dict[str, Any] | None:
    if is_codeforces:
        return evaluate_one_test_trace_stdin(
            code, spec_dict, test.get("input", "") or "",
            timeout=timeout, trace_keys=list(spec_dict),
        )
    args = get_test_args(test)
    if args is None:
        return None
    return evaluate_one_test_trace(
        code, spec_dict, func_name, params, args,
        timeout=timeout, trace_keys=list(spec_dict),
    )


def _run_correctness_tests(
    code: str,
    spec_dict: dict[str, Any],
    func_name: str,
    params: list[str],
    tests: list[dict[str, Any]],
    timeout: int,
    is_codeforces: bool,
    test_workers: int,
) -> list[tuple[int, dict[str, Any] | None, bool, str | None]]:
    jobs = [
        (_trace_unit, (code, spec_dict, func_name, params, test, timeout, is_codeforces))
        for test in tests
    ]
    raw = _run_bounded(jobs, test_workers, per_job_timeout=timeout + 5)
    out: list[tuple[int, dict[str, Any] | None, bool, str | None]] = []
    for index in range(len(tests)):
        result = raw.get(index)
        if result is None:
            out.append((index, None, True, "Trace timeout (subprocess killed)"))
            continue
        soft_timeout = any(
            v.get("error") == "Trace timeout" for v in result.values() if isinstance(v, dict)
        )
        out.append((index, result, False, "Trace timeout" if soft_timeout else None))
    return out


def compute_correctness(
    correct_solution: str,
    specs_by_line: dict[Any, Any],
    func_name: str,
    params: list[str],
    test_cases: list[dict[str, Any]],
    timeout: int = 10,
    is_codeforces: bool = False,
    test_workers: int = 1,
) -> tuple[dict[str, dict[str, Any]], int, dict[str, set[str]]]:
    if not specs_by_line:
        return {}, 0, {}
    spec_dict = {str(key): value for key, value in specs_by_line.items()}
    pass_counts = {str(key): 0 for key in spec_dict}
    reach_counts = {str(key): 0 for key in spec_dict}
    execution_pass_counts = {str(key): 0 for key in spec_dict}
    execution_reach_counts = {str(key): 0 for key in spec_dict}
    co_reach_map: dict[str, set[str]] = {str(key): set() for key in spec_dict}
    timeout_count = 0

    tests = _select_tests(test_cases, is_codeforces)
    for _index, result, timed_out, err_str in _run_correctness_tests(
        correct_solution,
        spec_dict,
        func_name,
        params,
        tests,
        timeout,
        is_codeforces,
        test_workers,
    ):
        if timed_out:
            timeout_count += 1
            continue
        if result is None:
            continue
        if err_str == "Trace timeout":
            timeout_count += 1
        for trace_key in pass_counts:
            info = result.get(trace_key, {})
            execution_reach_count = info.get("reach_count", 0)
            execution_pass_count = info.get("pass_count", 0)
            execution_reach_counts[trace_key] += execution_reach_count
            execution_pass_counts[trace_key] += execution_pass_count
            if execution_reach_count > 0:
                reach_counts[trace_key] += 1
                if execution_pass_count == execution_reach_count:
                    pass_counts[trace_key] += 1
            test_co_reach = info.get("co_reach")
            if test_co_reach:
                co_reach_map.setdefault(trace_key, set()).update(test_co_reach)

    out = {}
    for trace_key in pass_counts:
        reach_count = reach_counts[trace_key]
        pass_count = pass_counts[trace_key]
        out[trace_key] = {
            "pass_count": pass_count,
            "reach_count": reach_count,
            "execution_pass_count": execution_pass_counts[trace_key],
            "execution_reach_count": execution_reach_counts[trace_key],
            "correctness": round(pass_count / reach_count, 4) if reach_count > 0 else 0.0,
        }
    return out, timeout_count, co_reach_map


def _fold_mutant_test(out_key: dict[str, Any], info: dict[str, Any], index: int) -> None:
    reach_count = info.get("reach_count", 0)
    pass_count = info.get("pass_count", 0)
    out_key["execution_reach_count"] += reach_count
    out_key["execution_pass_count"] += pass_count
    if reach_count <= 0:
        out_key["labels"].append((MUTANT_SKIPPED, index, "NOT_REACHED"))
        return
    out_key["num_mutant_tests_reached"] += 1
    if pass_count < reach_count:
        out_key["num_mutant_tests_killed"] += 1
        out_key["labels"].append((MUTANT_REJECTED, index, f"FAIL (test {index})"))
    else:
        out_key["num_mutant_tests_accepted"] += 1
        out_key["labels"].append((MUTANT_ACCEPTED, index, ""))


def compute_completeness_batched(
    mutants: list[str],
    specs_by_trace_key: dict[str, dict[str, Any]],
    func_name: str,
    params: list[str],
    test_cases: list[dict[str, Any]],
    timeout_per_test: int = 5,
    outer_timeout: int = 10,
    mutant_eligible_trace_keys: list[set[str]] | None = None,
    batch_size: int = 1,
    is_codeforces: bool = False,
    test_workers: int = 1,
) -> dict[str, dict[str, Any]]:
    functional = _select_tests(test_cases, is_codeforces)
    out = {
        str(key): {
            "num_mutant_rejected": 0,
            "num_mutant_accepted": 0,
            "num_mutant_skipped": 0,
            "num_mutant_tests_killed": 0,
            "num_mutant_tests_accepted": 0,
            "num_mutant_tests_reached": 0,
            "execution_reach_count": 0,
            "execution_pass_count": 0,
            "labels": [],
        }
        for key in specs_by_trace_key
    }
    if not specs_by_trace_key:
        return out

    eligible_per_mutant: list[set[str]] = []
    eligible_specs_per_mutant: list[dict[str, dict[str, Any]]] = []
    for mutant_idx in range(len(mutants)):
        eligible = (
            set(mutant_eligible_trace_keys[mutant_idx])
            if mutant_eligible_trace_keys is not None and mutant_idx < len(mutant_eligible_trace_keys)
            else set(specs_by_trace_key)
        )
        eligible_per_mutant.append(eligible)
        eligible_specs_per_mutant.append(
            {key: specs_by_trace_key[key] for key in eligible if key in specs_by_trace_key}
        )

    # Per-mutant running totals (same shape the old _batched_mutant_worker emitted).
    def _blank_mutant(eligible_specs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {
            str(key): {
                "num_mutant_tests_killed": 0,
                "num_mutant_tests_accepted": 0,
                "num_mutant_tests_reached": 0,
                "execution_reach_count": 0,
                "execution_pass_count": 0,
                "labels": [],
            }
            for key in eligible_specs
        }

    runnable_indices = [i for i in range(len(mutants)) if eligible_specs_per_mutant[i]]
    mutant_results: dict[int, dict[str, Any]] = {
        i: _blank_mutant(eligible_specs_per_mutant[i]) for i in runnable_indices
    }

    # Flat (mutant, test) job list — one forked trace each, capped at test_workers.
    jobs: list[tuple[Callable[..., Any], tuple]] = []
    job_meta: list[tuple[int, int]] = []  # (mutant_idx, test_index)
    for mutant_idx in runnable_indices:
        spec_dict = eligible_specs_per_mutant[mutant_idx]
        mutant_code = mutants[mutant_idx]
        for test_index, test in enumerate(functional):
            jobs.append(
                (_trace_unit, (mutant_code, spec_dict, func_name, params, test, timeout_per_test, is_codeforces))
            )
            job_meta.append((mutant_idx, test_index))

    raw = _run_bounded(jobs, test_workers, per_job_timeout=timeout_per_test + 5)

    for job_index, (mutant_idx, test_index) in enumerate(job_meta):
        result = raw.get(job_index)
        if result is None:
            continue
        mutant_out = mutant_results[mutant_idx]
        for key in mutant_out:
            info = result.get(key, {})
            _fold_mutant_test(mutant_out[key], info, test_index)

    for mutant_idx in range(len(mutants)):
        eligible = eligible_per_mutant[mutant_idx]
        if not eligible:
            for key in out:
                out[key]["num_mutant_skipped"] += 1
                out[key]["labels"].append((MUTANT_SKIPPED, None, "MUTANT_AFTER_SPEC"))
            continue
        mutant_result = mutant_results.get(mutant_idx, {})
        for key in out:
            if key not in eligible:
                out[key]["num_mutant_skipped"] += 1
                out[key]["labels"].append((MUTANT_SKIPPED, None, "MUTANT_AFTER_SPEC"))
                continue
            record = mutant_result.get(
                key,
                {
                    "num_mutant_tests_killed": 0,
                    "num_mutant_tests_accepted": 0,
                    "num_mutant_tests_reached": 0,
                    "execution_reach_count": 0,
                    "execution_pass_count": 0,
                    "labels": [(MUTANT_SKIPPED, None, "NOT_REACHED")],
                },
            )
            out[key]["num_mutant_tests_killed"] += int(record.get("num_mutant_tests_killed", 0))
            out[key]["num_mutant_tests_accepted"] += int(record.get("num_mutant_tests_accepted", 0))
            out[key]["num_mutant_tests_reached"] += int(record.get("num_mutant_tests_reached", 0))
            out[key]["execution_reach_count"] += int(record.get("execution_reach_count", 0))
            out[key]["execution_pass_count"] += int(record.get("execution_pass_count", 0))
            labels = record.get("labels", [])
            out[key]["labels"].append(
                {
                    "mutant_index": mutant_idx,
                    "labels": labels,
                    "num_mutant_tests_killed": int(record.get("num_mutant_tests_killed", 0)),
                    "num_mutant_tests_accepted": int(record.get("num_mutant_tests_accepted", 0)),
                    "num_mutant_tests_reached": int(record.get("num_mutant_tests_reached", 0)),
                }
            )
            if int(record.get("num_mutant_tests_killed", 0)) > 0:
                out[key]["num_mutant_rejected"] += 1
            elif int(record.get("num_mutant_tests_reached", 0)) == 0:
                out[key]["num_mutant_skipped"] += 1
            else:
                out[key]["num_mutant_accepted"] += 1

    for key, record in out.items():
        total_eval = record["num_mutant_rejected"] + record["num_mutant_accepted"]
        record["num_eval"] = total_eval
        record["completeness"] = record["num_mutant_rejected"] / total_eval if total_eval > 0 else 1.0
        record["test_level_num_eval"] = record["num_mutant_tests_reached"]
        record["test_level_completeness"] = (
            record["num_mutant_tests_killed"] / record["num_mutant_tests_reached"]
            if record["num_mutant_tests_reached"] > 0
            else 0.0
        )
    return out
