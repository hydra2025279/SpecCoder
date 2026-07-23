from __future__ import annotations

import ast
import json
import sys
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any

_THIS = Path(__file__).resolve()
PROJECT_ROOT: Path | None = None
for _parent in _THIS.parents:
    if (_parent / "SpecCoder").is_dir():
        PROJECT_ROOT = _parent
        break
if PROJECT_ROOT is None:
    PROJECT_ROOT = _THIS.parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from SpecCoder.spec_evaluation import (  
    compute_completeness_batched,
    compute_correctness,
)
from SpecCoder.spec_utils import (  
    apply_single_mutant_record,
    build_full_code,
    extract_assertions_from_annotated_code,
    find_mutant_record_line_index,
)
from SpecCoder.trace_collection import (  
    instrument_assertions_in_completion,
    instrument_mutant_lines_in_completion,
    mutant_reach_id,
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_latest_jsonl(path: str | Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in load_jsonl(path):
        latest[record_task_id(record)] = record
    return list(latest.values())


def record_task_id(rec: dict[str, Any]) -> str:
    return rec.get("task_id") or f"cf_{rec.get('contestId')}{rec.get('index')}"


def _strip_code_fences(text: str) -> str:
    if "```" not in text:
        return text
    kept = [ln for ln in text.split("\n") if not ln.lstrip().startswith("```")]
    return "\n".join(kept)


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
    )


def _has_entrypoint_invocation(tree: ast.Module) -> bool:
    for node in tree.body:
        if _is_main_guard(node):
            return True
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            return True
    return False


def _append_missing_entrypoint(annotated: str, clean: str) -> str:
    if not annotated.strip() or not clean.strip():
        return annotated
    try:
        atree = ast.parse(annotated)
        ctree = ast.parse(clean)
    except SyntaxError:
        return annotated
    if _has_entrypoint_invocation(atree):
        return annotated
    defined = {
        node.name
        for node in atree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    clean_lines = clean.split("\n")
    snippets: list[str] = []
    for node in ctree.body:
        take = False
        if _is_main_guard(node):
            take = True
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            take = isinstance(func, ast.Name) and func.id in defined
        if take:
            start = node.lineno - 1
            end = getattr(node, "end_lineno", node.lineno)
            snippets.append("\n".join(clean_lines[start:end]))
    if not snippets:
        return annotated
    return annotated.rstrip("\n") + "\n" + "\n".join(snippets) + "\n"


def annotated_of(rec: dict[str, Any]) -> str:
    annotated = rec.get("annotated_code")
    if isinstance(annotated, list):
        annotated = annotated[0] if annotated else ""
    if annotated:
        annotated = _strip_code_fences(annotated)
    else:
        code = rec.get("code")
        if isinstance(code, list):
            code = code[0] if code else ""
        annotated = _strip_code_fences(code or "")
    clean = rec.get("clean_code")
    if isinstance(clean, list):
        clean = clean[0] if clean else ""
    if clean:
        annotated = _append_missing_entrypoint(annotated, _strip_code_fences(clean))
    return annotated


def shared_mutant_records(mutants_by_task: dict[str, dict], task_id: str) -> list[dict[str, Any]]:
    rec = mutants_by_task.get(task_id)
    return list(rec.get("content", [])) if rec else []


def assertion_syntax_error(expr: str) -> str | None:
    try:
        ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        return f"{exc.msg} (line {exc.lineno}, offset {exc.offset})"
    return None


def has_later_code_line(annotated: str, assertion_line_index: int) -> bool:
    lines = annotated.splitlines()
    for line in lines[assertion_line_index + 1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        return True
    return False


def average(values: list[float]) -> float:
    return round(mean(values), 4) if values else 0.0


def assertion_has_eval_signal(assertion: dict[str, Any]) -> bool:
    if int(assertion.get("correctness_reach_count", 0) or 0) <= 0:
        return False
    rejected = int(assertion.get("num_mutant_rejected", 0) or 0)
    accepted = int(assertion.get("num_mutant_accepted", 0) or 0)
    reachable_prior = int(assertion.get("num_reachable_prior_mutants", 0) or 0)
    if rejected == 0 and accepted == 0 and reachable_prior == 0:
        return False
    return True


def assertion_group_stats(
    assertions: list[dict[str, Any]],
    group: str | None = None,
    *,
    drop_unevaluable: bool = True,
) -> dict[str, Any]:
    selected = [
        assertion for assertion in assertions
        if (not drop_unevaluable or assertion_has_eval_signal(assertion))
        and (group is None or assertion.get("position_type") == group)
    ]
    correctness = [float(a.get("correctness", 0.0) or 0.0) for a in selected]
    completeness = [float(a.get("completeness", 0.0) or 0.0) for a in selected]
    return {
        "num_specs": len(selected),
        "num_syntax_valid_specs": sum(1 for a in selected if a.get("syntax_valid")),
        "num_syntax_invalid_specs": sum(1 for a in selected if not a.get("syntax_valid")),
        "num_counted_syntax_invalid_specs": sum(
            1 for a in selected
            if not a.get("syntax_valid") and assertion_has_eval_signal(a)
        ),
        "num_unevaluable_syntax_invalid_specs": sum(
            1 for a in selected
            if not a.get("syntax_valid") and not assertion_has_eval_signal(a)
        ),
        "avg_correctness": average(correctness),
        "avg_completeness": average(completeness),
    }


def normalize_sample_result(sample: dict[str, Any], *, drop_unevaluable: bool = True) -> dict[str, Any]:
    out = dict(sample)
    assertions = list(out.get("assertions") or [])
    if drop_unevaluable:
        assertions = [assertion for assertion in assertions if assertion_has_eval_signal(assertion)]
    overall = assertion_group_stats(assertions)
    middle = assertion_group_stats(assertions, "middle")
    post = assertion_group_stats(assertions, "post")
    out["assertions"] = assertions
    out["num_specs"] = overall["num_specs"]
    out["num_middle_specs"] = middle["num_specs"]
    out["num_post_specs"] = post["num_specs"]
    out["num_syntax_valid_specs"] = overall["num_syntax_valid_specs"]
    out["num_syntax_invalid_specs"] = overall["num_syntax_invalid_specs"]
    out["num_counted_syntax_invalid_specs"] = overall["num_counted_syntax_invalid_specs"]
    out["num_unevaluable_syntax_invalid_specs"] = overall["num_unevaluable_syntax_invalid_specs"]
    out["avg_correctness"] = overall["avg_correctness"]
    out["avg_completeness"] = overall["avg_completeness"]
    out["overall_correctness"] = overall["avg_correctness"]
    out["overall_completeness"] = overall["avg_completeness"]
    out["avg_inline_correctness"] = middle["avg_correctness"]
    out["avg_inline_completeness"] = middle["avg_completeness"]
    out["avg_post_correctness"] = post["avg_correctness"]
    out["avg_post_completeness"] = post["avg_completeness"]
    return out


def normalize_task_record(task_record: dict[str, Any], *, drop_unevaluable: bool = True) -> dict[str, Any]:
    out = dict(task_record)
    if "sample_results" in out:
        out["sample_results"] = [
            normalize_sample_result(sample, drop_unevaluable=drop_unevaluable)
            for sample in out.get("sample_results") or []
        ]
        out.update(task_metric_fields(out, drop_unevaluable=drop_unevaluable))
        return out
    return normalize_sample_result(out, drop_unevaluable=drop_unevaluable)


def result_has_prior_mutant_fields(task_record: dict[str, Any]) -> bool:
    samples = task_record.get("sample_results") if "sample_results" in task_record else [task_record]
    for sample in samples or []:
        for assertion in sample.get("assertions") or []:
            if "num_prior_mutants" not in assertion or "num_reachable_prior_mutants" not in assertion:
                return False
    return True


def evaluate_record(
    rec: dict[str, Any],
    mutant_records: list[dict[str, Any]],
    timeout: int,
    test_workers: int = 32,
    drop_unevaluable: bool = True,
) -> dict[str, Any] | None:
    tid = record_task_id(rec)
    annotated = annotated_of(rec)
    tests = list(rec.get("test_cases") or [])
    if not annotated.strip() or not tests:
        return None

    cleaned, assertions = extract_assertions_from_annotated_code(annotated)
    if not assertions:
        return {
            "task_id": tid,
            "contestId": rec.get("contestId"),
            "index": rec.get("index"),
            "name": rec.get("name"),
            "sample_index": rec.get("sample_index"),
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

    assertion_meta: dict[str, dict[str, Any]] = {}
    valid_assertions = []
    trace_assertions = []
    for a in assertions:
        syntax_error = assertion_syntax_error(a.expr)
        position_type = "middle" if has_later_code_line(annotated, a.assertion_line_index) else "post"
        assertion_meta[a.trace_key] = {
            "syntax_valid": syntax_error is None,
            "syntax_error": syntax_error,
            "position_type": position_type,
        }
        if syntax_error is None:
            valid_assertions.append(a)
            trace_assertions.append(a)
        else:
            trace_assertions.append(replace(a, expr="True"))

    specs_by_tk = {
        a.trace_key: {"assert_key": a.assert_key, "dedup_key": a.dedup_key, "expr": a.expr}
        for a in valid_assertions
    }
    trace_specs_by_tk = {
        a.trace_key: {"assert_key": a.assert_key, "dedup_key": a.dedup_key, "expr": a.expr}
        for a in trace_assertions
    }

    # --- correctness (stdin/stdout) ---------------------------------------
    correctness_by_tk: dict[str, dict[str, Any]] = {}
    co_reach_map: dict[str, set[Any]] = {}
    if trace_assertions:
        instrumented = instrument_assertions_in_completion(annotated, trace_assertions)
        instrumented = instrument_mutant_lines_in_completion(instrumented, mutant_records)
        full_code, _ = build_full_code("", instrumented)
        correctness_by_tk, _timeouts, co_reach_map = compute_correctness(
            full_code,
            trace_specs_by_tk,
            None,
            [],
            tests,
            timeout=timeout,
            is_codeforces=True,
            test_workers=test_workers,
        )

    # --- completeness ------------------------------------------------------
    completeness_by_tk: dict[str, dict[str, Any]] = {}
    prior_mutant_count_by_tk: dict[str, int] = {a.trace_key: 0 for a in assertions}
    reachable_prior_mutant_count_by_tk: dict[str, int] = {a.trace_key: 0 for a in assertions}
    if mutant_records and valid_assertions:
        mutant_programs: list[str] = []
        eligible_trace_keys: list[set[str]] = []
        mutant_metadata: list[dict[str, Any]] = []
        for mrec in mutant_records:
            for variant_index, variant in enumerate(mrec.get("mutant", [])):
                single = {
                    "original_content": mrec.get("original_content"),
                    "dedup_key": mrec.get("dedup_key"),
                    "mutant": [variant],
                }
                line_idx = find_mutant_record_line_index(annotated, single)
                if line_idx is None:
                    continue
                mid = mutant_reach_id(single.get("dedup_key", 0), str(single.get("original_content", "")))
                for a in assertions:
                    if line_idx < a.assertion_line_index:
                        prior_mutant_count_by_tk[a.trace_key] += 1
                        if mid in co_reach_map.get(a.trace_key, set()):
                            reachable_prior_mutant_count_by_tk[a.trace_key] += 1
                eligible = [
                    a for a in valid_assertions
                    if line_idx < a.assertion_line_index
                    and mid in co_reach_map.get(a.trace_key, set())
                ]
                mutated = apply_single_mutant_record(annotated, single, variant)
                if mutated is None:
                    continue
                instrumented_mutant = instrument_assertions_in_completion(mutated, eligible)
                program, _ = build_full_code("", instrumented_mutant)
                mutant_programs.append(program)
                eligible_trace_keys.append({a.trace_key for a in eligible})
                mutant_metadata.append({
                    "mutant_index": len(mutant_metadata),
                    "source_variant_index": variant_index,
                    "source_mutant_index": mrec.get("index"),
                    "source_line_index": mrec.get("line_index"),
                    "dedup_key": mrec.get("dedup_key"),
                    "original_content": mrec.get("original_content"),
                    "mutant_content": variant,
                    "mutant": variant,
                    "line_index_in_annotated_code": line_idx,
                    "eligible_trace_keys": [a.trace_key for a in eligible],
                })
        if mutant_programs:
            completeness_by_tk = compute_completeness_batched(
                mutant_programs, specs_by_tk, None, [], tests,
                timeout_per_test=timeout, outer_timeout=timeout * 2,
                mutant_eligible_trace_keys=eligible_trace_keys,
                batch_size=1, is_codeforces=True, test_workers=test_workers,
            )
            for minfo in completeness_by_tk.values():
                enriched_labels = []
                for label in minfo.get("labels") or []:
                    if isinstance(label, dict):
                        mutant_index = label.get("mutant_index")
                        meta = (
                            mutant_metadata[int(mutant_index)]
                            if isinstance(mutant_index, int) and 0 <= mutant_index < len(mutant_metadata)
                            else {}
                        )
                        status = "mutant_skipped"
                        if int(label.get("num_mutant_tests_killed", 0) or 0) > 0:
                            status = "mutant_rejected"
                        elif int(label.get("num_mutant_tests_reached", 0) or 0) > 0:
                            status = "mutant_accepted"
                        enriched_labels.append({
                            **label,
                            "status": status,
                            "mutant_record": meta,
                        })
                    else:
                        enriched_labels.append(label)
                minfo["labels"] = enriched_labels
    elif mutant_records:
        for mrec in mutant_records:
            for _variant_index, _variant in enumerate(mrec.get("mutant", [])):
                single = {
                    "original_content": mrec.get("original_content"),
                    "dedup_key": mrec.get("dedup_key"),
                    "mutant": [_variant],
                }
                line_idx = find_mutant_record_line_index(annotated, single)
                if line_idx is None:
                    continue
                mid = mutant_reach_id(single.get("dedup_key", 0), str(single.get("original_content", "")))
                for a in assertions:
                    if line_idx < a.assertion_line_index:
                        prior_mutant_count_by_tk[a.trace_key] += 1
                        if mid in co_reach_map.get(a.trace_key, set()):
                            reachable_prior_mutant_count_by_tk[a.trace_key] += 1

    per_assertion = []
    for a in assertions:
        meta = assertion_meta.get(a.trace_key, {})
        cinfo = correctness_by_tk.get(a.trace_key, {})
        minfo = completeness_by_tk.get(a.trace_key, {})
        syntax_valid = bool(meta.get("syntax_valid"))
        prior_mutants = int(prior_mutant_count_by_tk.get(a.trace_key, 0))
        reachable_prior_mutants = int(reachable_prior_mutant_count_by_tk.get(a.trace_key, 0))
        correctness = float(cinfo.get("correctness", 0.0))
        correctness_pass_count = int(cinfo.get("pass_count", 0))
        correctness_reach_count = int(cinfo.get("reach_count", 0))
        correctness_execution_pass_count = int(cinfo.get("execution_pass_count", 0))
        correctness_execution_reach_count = int(cinfo.get("execution_reach_count", 0))
        if not syntax_valid:
            correctness = 0.0
            correctness_pass_count = 0
            correctness_execution_pass_count = 0
        num_mutant_rejected = int(minfo.get("num_mutant_rejected", 0))
        num_mutant_accepted = int(minfo.get("num_mutant_accepted", 0))
        completeness = float(minfo.get("completeness", 0.0))
        if reachable_prior_mutants > 0 and num_mutant_rejected == 0 and num_mutant_accepted == 0:
            completeness = 0.0
        per_assertion.append({
            "trace_key": a.trace_key,
            "assert_key": a.assert_key,
            "expr": a.expr,
            "syntax_valid": syntax_valid,
            "syntax_error": meta.get("syntax_error"),
            "position_type": meta.get("position_type"),
            "is_middle_spec": meta.get("position_type") == "middle",
            "is_post_spec": meta.get("position_type") == "post",
            "dedup_key": a.dedup_key,
            "indent": a.indent,
            "assertion_line_index": a.assertion_line_index,
            "line_content": a.line_content,
            "correctness": correctness,
            "correctness_pass_count": correctness_pass_count,
            "correctness_reach_count": correctness_reach_count,
            "correctness_execution_pass_count": correctness_execution_pass_count,
            "correctness_execution_reach_count": correctness_execution_reach_count,
            "completeness": completeness,
            "num_eval": int(minfo.get("num_eval", 0)),
            "num_mutant_rejected": num_mutant_rejected,
            "num_mutant_accepted": num_mutant_accepted,
            "num_mutant_skipped": int(minfo.get("num_mutant_skipped", 0)),
            "num_prior_mutants": prior_mutants,
            "num_reachable_prior_mutants": reachable_prior_mutants,
            "test_level_num_eval": int(minfo.get("test_level_num_eval", 0)),
            "test_level_completeness": float(minfo.get("test_level_completeness", 0.0)),
            "mutant_labels": minfo.get("labels", []),
        })
    overall = assertion_group_stats(per_assertion, drop_unevaluable=drop_unevaluable)
    middle = assertion_group_stats(per_assertion, "middle", drop_unevaluable=drop_unevaluable)
    post = assertion_group_stats(per_assertion, "post", drop_unevaluable=drop_unevaluable)
    result = {
        "task_id": tid,
        "contestId": rec.get("contestId"),
        "index": rec.get("index"),
        "name": rec.get("name"),
        "sample_index": rec.get("sample_index"),
        "num_specs": overall["num_specs"],
        "num_middle_specs": middle["num_specs"],
        "num_post_specs": post["num_specs"],
        "num_syntax_valid_specs": overall["num_syntax_valid_specs"],
        "num_syntax_invalid_specs": overall["num_syntax_invalid_specs"],
        "num_counted_syntax_invalid_specs": overall["num_counted_syntax_invalid_specs"],
        "num_unevaluable_syntax_invalid_specs": overall["num_unevaluable_syntax_invalid_specs"],
        "avg_correctness": overall["avg_correctness"],
        "avg_completeness": overall["avg_completeness"],
        "overall_correctness": overall["avg_correctness"],
        "overall_completeness": overall["avg_completeness"],
        "avg_inline_correctness": middle["avg_correctness"],
        "avg_inline_completeness": middle["avg_completeness"],
        "avg_post_correctness": post["avg_correctness"],
        "avg_post_completeness": post["avg_completeness"],
        "assertions": per_assertion,
    }
    return normalize_sample_result(result, drop_unevaluable=drop_unevaluable)


def aggregate_assertion_metrics(
    setting_name: str,
    task_records: list[dict[str, Any]],
    *,
    num_attempts: int | None = None,
    drop_unevaluable: bool = True,
) -> dict[str, Any]:
    assertions: list[dict[str, Any]] = []
    total_specs = 0
    total_middle = 0
    total_post = 0
    syntax_valid = 0
    syntax_invalid = 0
    counted_syntax_invalid = 0
    unevaluable_syntax_invalid = 0
    completeness_eval = 0
    tasks_without_valid_specs = 0
    attempts_without_valid_specs = 0
    for task in task_records:
        samples = task.get("sample_results") if "sample_results" in task else [task]
        task_has_valid_spec = False
        for sample in samples or []:
            sample = normalize_sample_result(sample, drop_unevaluable=drop_unevaluable)
            sample_num_specs = int(sample.get("num_specs", 0) or 0)
            total_specs += sample_num_specs
            total_middle += int(sample.get("num_middle_specs", 0) or 0)
            total_post += int(sample.get("num_post_specs", 0) or 0)
            syntax_valid += int(sample.get("num_syntax_valid_specs", 0) or 0)
            syntax_invalid += int(sample.get("num_syntax_invalid_specs", 0) or 0)
            counted_syntax_invalid += int(sample.get("num_counted_syntax_invalid_specs", 0) or 0)
            unevaluable_syntax_invalid += int(sample.get("num_unevaluable_syntax_invalid_specs", 0) or 0)
            if sample_num_specs > 0:
                task_has_valid_spec = True
            else:
                attempts_without_valid_specs += 1
            for assertion in sample.get("assertions") or []:
                assertions.append(assertion)
                if int(assertion.get("num_eval", 0) or 0) > 0:
                    completeness_eval += 1
        if not task_has_valid_spec:
            tasks_without_valid_specs += 1

    overall = assertion_group_stats(assertions, drop_unevaluable=drop_unevaluable)
    middle = assertion_group_stats(assertions, "middle", drop_unevaluable=drop_unevaluable)
    post = assertion_group_stats(assertions, "post", drop_unevaluable=drop_unevaluable)
    attempts = num_attempts if num_attempts is not None else len(task_records)
    return {
        "setting": setting_name,
        "num_tasks": len(task_records),
        "num_attempts": attempts,
        "num_specs": total_specs,
        "num_middle_specs": total_middle,
        "num_post_specs": total_post,
        "num_syntax_valid_specs": syntax_valid,
        "num_syntax_invalid_specs": syntax_invalid,
        "num_counted_syntax_invalid_specs": counted_syntax_invalid,
        "num_unevaluable_syntax_invalid_specs": unevaluable_syntax_invalid,
        "avg_specs_per_task": round(total_specs / len(task_records), 4) if task_records else 0.0,
        "avg_specs_per_attempt": round(total_specs / attempts, 4) if attempts else 0.0,
        "avg_middle_specs_per_task": round(total_middle / len(task_records), 4) if task_records else 0.0,
        "avg_post_specs_per_task": round(total_post / len(task_records), 4) if task_records else 0.0,
        "avg_correctness": overall["avg_correctness"],
        "avg_completeness": overall["avg_completeness"],
        "overall_correctness": overall["avg_correctness"],
        "overall_completeness": overall["avg_completeness"],
        "avg_inline_correctness": middle["avg_correctness"],
        "avg_inline_completeness": middle["avg_completeness"],
        "avg_post_correctness": post["avg_correctness"],
        "avg_post_completeness": post["avg_completeness"],
        "num_assertions_with_completeness_eval": completeness_eval,
        "num_tasks_without_valid_specs": tasks_without_valid_specs,
        "num_attempts_without_valid_specs": attempts_without_valid_specs,
    }


def task_metric_fields(task_record: dict[str, Any], *, drop_unevaluable: bool = True) -> dict[str, Any]:
    metrics = aggregate_assertion_metrics(
        "task",
        [task_record],
        num_attempts=len(task_record.get("sample_results") or []),
        drop_unevaluable=drop_unevaluable,
    )
    drop = {"setting", "num_tasks", "num_attempts"}
    return {key: value for key, value in metrics.items() if key not in drop}


def evaluate_setting(
    name: str,
    records: list[dict[str, Any]],
    mutants_by_task: dict[str, dict],
    timeout: int,
    test_workers: int = 32,
) -> dict[str, Any]:
    total_specs = 0
    num_tasks = 0
    task_results: list[dict[str, Any]] = []
    for rec in records:
        res = evaluate_record(
            rec,
            shared_mutant_records(mutants_by_task, record_task_id(rec)),
            timeout,
            test_workers=test_workers,
        )
        if res is None:
            continue
        num_tasks += 1
        total_specs += res["num_specs"]
        task_results.append(res)
    result = aggregate_assertion_metrics(name, task_results, num_attempts=num_tasks)
    result["tasks"] = task_results
    result["num_specs"] = total_specs
    return result


def keep_passing(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For legacy mixed outputs, keep only programs that passed all tests."""
    out, seen = [], set()
    for r in records:
        task = record_task_id(r)
        if task in seen:
            continue
        if isinstance(r.get("attempts"), list):
            attempts = {int(a.get("attempt", i)): a for i, a in enumerate(r.get("attempts") or [])}
            for result in r.get("sample_results") or []:
                attempt = int(result.get("attempt", result.get("sample_index", 0)))
                candidate = attempts.get(attempt)
                if result.get("pass_rate") == 1.0 and candidate:
                    rec = dict(r)
                    rec["code"] = candidate.get("code", "")
                    rec["response"] = candidate.get("response", "")
                    rec["sample_index"] = attempt
                    rec["pass_rate"] = result.get("pass_rate")
                    seen.add(task)
                    out.append(rec)
                    break
        elif isinstance(r.get("code"), list):
            sample_results = r.get("sample_results") or []
            for result in sample_results:
                sample_index = int(result.get("sample_index", 0))
                if result.get("pass_rate") == 1.0 and sample_index < len(r.get("code") or []):
                    rec = dict(r)
                    rec["code"] = r["code"][sample_index]
                    rec["response"] = (r.get("response") or [""])[sample_index]
                    rec["sample_index"] = sample_index
                    rec["pass_rate"] = result.get("pass_rate")
                    seen.add(task)
                    out.append(rec)
                    break
        elif r.get("pass_rate") == 1.0:
            seen.add(task)
            out.append(r)
    return out


def load_mutants_by_task(mutants_path: str | Path) -> dict[str, dict[str, Any]]:
    return {m.get("task_id") or m.get("question_id"): m for m in load_jsonl(mutants_path)}


def evaluate_records_quality(
    setting_name: str,
    records: list[dict[str, Any]],
    mutants_path: str | Path,
    timeout: int = 10,
    test_workers: int = 32,
    *,
    passing_only: bool = False,
) -> dict[str, Any]:
    eval_records = keep_passing(records) if passing_only else records
    mutants_by_task = load_mutants_by_task(mutants_path)
    return evaluate_setting(setting_name, eval_records, mutants_by_task, timeout, test_workers=test_workers)


def evaluate_quality_file(
    setting_name: str,
    records_path: str | Path,
    mutants_path: str | Path,
    timeout: int = 10,
    test_workers: int = 32,
    *,
    latest_only: bool = False,
    passing_only: bool = False,
) -> dict[str, Any]:
    records = load_latest_jsonl(records_path) if latest_only else load_jsonl(records_path)
    return evaluate_records_quality(
        setting_name,
        records,
        mutants_path,
        timeout,
        test_workers=test_workers,
        passing_only=passing_only,
    )
