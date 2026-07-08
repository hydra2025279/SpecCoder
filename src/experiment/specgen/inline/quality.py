
from __future__ import annotations

import os
from typing import Any

from SpecCoder.src.experiment.gen_utils import (
    load_latest_jsonl_by_task,
    rewrite_jsonl,
    sum_token_usage,
)
from SpecCoder.src.experiment.specgen.compute_spec_quality import (
    aggregate_assertion_metrics,
    evaluate_record,
    normalize_task_record,
    result_has_prior_mutant_fields,
    shared_mutant_records,
    task_metric_fields,
)


def aggregate_quality_from_results(
    records: list[dict[str, Any]],
    setting_name: str,
    *,
    drop_unevaluable: bool = False,
) -> dict[str, Any]:
    task_results: list[dict[str, Any]] = []
    num_attempts = 0
    for rec in records:
        rec = normalize_task_record(rec, drop_unevaluable=drop_unevaluable)
        task_copy = {
            "task_id": rec.get("task_id"),
            "contestId": rec.get("contestId"),
            "index": rec.get("index"),
            "name": rec.get("name"),
            "sample_results": rec.get("sample_results") or [],
        }
        task_copy.update(task_metric_fields(task_copy, drop_unevaluable=drop_unevaluable))
        task_results.append(task_copy)
        for _sample in rec.get("sample_results") or []:
            num_attempts += 1
    result = aggregate_assertion_metrics(
        setting_name,
        task_results,
        num_attempts=num_attempts,
        drop_unevaluable=drop_unevaluable,
    )
    result["tasks"] = task_results
    return result


def normalize_result_jsonl_for_summary(
    result_path: str,
    tids: list[str],
    *,
    drop_unevaluable: bool = False,
) -> list[dict[str, Any]]:
    latest = load_latest_jsonl_by_task(result_path)
    if drop_unevaluable:
        for tid in list(latest):
            latest[tid] = normalize_task_record(latest[tid], drop_unevaluable=True)
    rewrite_jsonl(result_path, list(latest.values()))
    return [latest[tid] for tid in tids if tid in latest]


def existing_result_is_complete(existing: dict[str, Any] | None, opt: Any) -> bool:
    if not existing or len(existing.get("sample_results") or []) < opt.num_samples:
        return False
    if getattr(opt, "drop_unevaluable_assertions", True) and not result_has_prior_mutant_fields(existing):
        return False
    return True


def evaluate_task_attempts(
    task_record: dict[str, Any],
    attempts: list[dict[str, Any]],
    mutants_by_task: dict[str, dict[str, Any]],
    opt: Any,
) -> dict[str, Any]:
    if not os.path.exists(opt.mutants):
        raise SystemExit(f"Mutant file not found for spec-quality eval: {opt.mutants}")
    sample_results = []
    for attempt in attempts:
        res = evaluate_record(
            attempt,
            shared_mutant_records(mutants_by_task, str(task_record["task_id"])),
            opt.quality_timeout,
            test_workers=opt.quality_test_workers,
            drop_unevaluable=getattr(opt, "drop_unevaluable_assertions", True),
        )
        if res is None:
            res = {
                "task_id": task_record["task_id"],
                "sample_index": attempt.get("attempt"),
                "num_specs": 0,
                "num_middle_specs": 0,
                "num_post_specs": 0,
                "num_syntax_valid_specs": 0,
                "num_syntax_invalid_specs": 0,
                "num_counted_syntax_invalid_specs": 0,
                "num_unevaluable_syntax_invalid_specs": 0,
                "avg_correctness": 0.0,
                "avg_completeness": 0.0,
                "overall_correctness": 0.0,
                "overall_completeness": 0.0,
                "avg_inline_correctness": 0.0,
                "avg_inline_completeness": 0.0,
                "avg_post_correctness": 0.0,
                "avg_post_completeness": 0.0,
                "assertions": [],
            }
        res["attempt"] = int(attempt.get("attempt", res.get("sample_index", 0)))
        res["sample_index"] = res["attempt"]
        sample_results.append(res)
    out = dict(task_record)
    out["attempts"] = attempts
    out["sample_results"] = sorted(sample_results, key=lambda r: int(r.get("attempt", 0)))
    out["num_attempts"] = len(attempts)
    out["total_time_latency"] = {
        "generation_seconds": round(sum(
            float((attempt.get("time_latency") or {}).get("generation_seconds", 0.0) or 0.0)
            for attempt in attempts
        ), 6),
    }
    out["total_cost_token"] = sum_token_usage([attempt.get("cost_token") or {} for attempt in attempts])
    out["test_cases"] = task_record.get("test_cases")
    out.update(task_metric_fields(out, drop_unevaluable=getattr(opt, "drop_unevaluable_assertions", True)))
    return out
