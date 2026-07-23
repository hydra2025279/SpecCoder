from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

_THIS = Path(__file__).resolve()
PROJECT_ROOT = next((p for p in _THIS.parents if (p / "SpecCoder").is_dir()), _THIS.parents[4])
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from SpecCoder.src.experiment.backend import add_backend_args, make_backend, write_token_files  # noqa: E402
from SpecCoder.src.experiment.correctnesschecking.prompt import (  # noqa: E402
    SYSTEM_PROMPT,
    build_assert_prompt,
    build_cot_prompt,
    build_input_generation_prompt,
    build_speccoder_prompt,
    build_vanilla_prompt,
)
from SpecCoder.src.experiment.correctnesschecking.metrics import (  # noqa: E402
    build_metrics,
    format_metrics_table,
    gather_result_records,
    mcc_from_summary,
)
from SpecCoder.src.experiment.correctnesschecking import assertchecking  # noqa: E402


SPEC_METHODS = ("assert",)
from SpecCoder.src.experiment.gen_utils import (  # noqa: E402
    GENERATION_FILE,
    RESULT_FILE,
    SUMMARY_FILE,
    append_jsonl,
    append_task_log,
    experiment_output_dir,
    extract_code,
    load_latest_jsonl_by_task,
    spec_gen_output_dir,
    task_log_name,
    token_usage_record,
    write_json,
)

BENCHMARK = "HumanChecking"
DEFAULT_HUMANCHECKING = str(PROJECT_ROOT / "SpecCoder" / "data" / "humanchecking.jsonl")
DEFAULT_HUMANEXEC = str(PROJECT_ROOT / "SpecCoder" / "data" / "humanexec.jsonl")


def load_jsonl(path: str, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def cf_task_id(row: dict[str, Any]) -> str:
    if row.get("task_id"):
        return str(row["task_id"])
    if row.get("contestId") is not None and row.get("index") is not None:
        return f"cf_{row.get('contestId')}{row.get('index')}"
    return f"task_{row.get('dataset_index', 'unknown')}"


def _base_task_id(rec: dict[str, Any]) -> str:
    src = rec.get("source_task_id")
    if src:
        return str(src)
    tid = str(rec.get("task_id") or "")
    for suffix in ("__correct", "__buggy"):
        if tid.endswith(suffix):
            return tid[: -len(suffix)]
    return tid


def problem_parts(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("problem-description") or row.get("problem_description") or row.get("description") or ""),
        str(row.get("input-specification") or row.get("input_specification") or row.get("input") or ""),
        str(row.get("output-specification") or row.get("output_specification") or row.get("output") or ""),
    )


def problem_text(row: dict[str, Any]) -> str:
    desc, inp, out = problem_parts(row)
    blocks = []
    if desc.strip():
        blocks.append(f"### Problem Description\n{desc}")
    if inp.strip():
        blocks.append(f"### Input Specification\n{inp}")
    if out.strip():
        blocks.append(f"### Output Specification\n{out}")
    return "\n\n".join(blocks) or str(row.get("prompt") or "")


def bool_from_any(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "correct", "accepted", "ac", "ok", "pass", "passed"}:
            return True
        if lowered in {"false", "incorrect", "wrong", "wa", "fail", "failed", "buggy"}:
            return False
    return None


def humanchecking_tasks(path: str, limit: int | None = None) -> list[dict[str, Any]]:
    rows = load_jsonl(path, limit)
    tasks: list[dict[str, Any]] = []
    for row in rows:
        label = row.get("label")
        if not isinstance(label, bool):
            label = bool_from_any(label)
        if label is None:
            continue
        side = "correct" if label else "buggy"
        code = str(row.get("code") or "")
        if not code.strip():
            continue
        base_id = cf_task_id(row)
        tasks.append({
            "source_task_id": base_id,
            "contestId": row.get("contestId"),
            "index": row.get("index"),
            "name": row.get("name"),
            "problem_description": row.get("problem-description") or "",
            "input_specification": row.get("input-specification") or "",
            "output_specification": row.get("output-specification") or "",
            "problem_text": problem_text(row),
            "test_cases": row.get("test_cases") or [],
            "task_id": f"{base_id}__{side}",
            "code": code,
            "label": label,
            "candidate_kind": side,
        })
    return tasks


def load_tasks(opt: argparse.Namespace) -> list[dict[str, Any]]:
    return humanchecking_tasks(opt.input or DEFAULT_HUMANCHECKING, opt.limit)


def default_spec_file(opt: argparse.Namespace) -> str:
    return str(spec_gen_output_dir(
        "inline",
        opt.model,
        mode="oracle",
        speccoder_finetune=opt.speccoder_finetune,
        make=False,
    ) / GENERATION_FILE)


def load_spec_generation_by_task(path: str) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path or not os.path.exists(path):
        return latest
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            tid = str(rec.get("task_id") or "")
            attempt = int(rec.get("attempt", rec.get("sample_index", 0)) or 0)
            if not tid:
                continue
            prev = latest.get(tid)
            prev_attempt = int(prev.get("attempt", prev.get("sample_index", 0)) or 0) if prev else -1
            if prev is None or attempt >= prev_attempt:
                latest[tid] = rec
    return latest


def ensure_specs_generated(
    tasks: list[dict[str, Any]],
    opt: argparse.Namespace,
    spec_file: str,
) -> None:
    needed_source_ids = {t["source_task_id"] for t in tasks}
    existing = load_spec_generation_by_task(spec_file)  # empty dict if file missing
    covered = needed_source_ids & set(existing)
    missing = needed_source_ids - covered

    if not missing:
        print(f"Spec file has all {len(needed_source_ids)} tasks → skipping spec generation.")
        return

    action = "Generating from scratch" if not covered else f"Appending {len(missing)} missing tasks (skipping {len(covered)} existing)"
    print(f"[spec-gen] {action} | inline oracle → {spec_file}")

    # Build a namespace for the spec gen runner, reusing shared backend args.
    spec_dir = str(Path(spec_file).parent)
    common_kw = {
        "model": opt.model,
        "backend": opt.backend,
        "speccoder_finetune": opt.speccoder_finetune,
        "lora_path": getattr(opt, "lora_path", None),
        "max_model_len": getattr(opt, "max_model_len", 32768),
        "tensor_parallel_size": getattr(opt, "tensor_parallel_size", 0),
        "gpu_memory_utilization": getattr(opt, "gpu_memory_utilization", 0.95),
        "cache_dir": getattr(opt, "cache_dir", ""),
        "max_tokens": getattr(opt, "max_tokens", 8192),
        "temperature": getattr(opt, "temperature", 0.2),
        "top_p": getattr(opt, "top_p", 0.95),
        "top_k": getattr(opt, "top_k", 0),
        "output": spec_dir,
        "humanexec": opt.input or DEFAULT_HUMANEXEC,
        "limit": None,
        "num_tasks": None,
        "num_samples": 1,
        "global_batch": opt.global_batch,
        "mutants": str(PROJECT_ROOT / "SpecCoder" / "data" / "mutants" / "HumanExec_mutant_prune.jsonl"),
        "quality_output": None,
        "quality_timeout": 10,
        "quality_test_workers": 32,
        "drop_unevaluable_assertions": False,
        "no_eval": True,   # skip quality evaluation — we just need generation
        "eval": False,
        "eval_only": False,
    }

    from SpecCoder.src.experiment.specgen.inline.run import run_oracle
    run_oracle(argparse.Namespace(**common_kw))

    # Verify all tasks are now covered
    updated = load_spec_generation_by_task(spec_file)
    still_missing = needed_source_ids - set(updated)
    if still_missing:
        preview = ", ".join(sorted(still_missing)[:5])
        print(f"[spec-gen] WARNING: {len(still_missing)} tasks still missing after generation; first: {preview}")
    else:
        print(f"[spec-gen] All {len(needed_source_ids)} tasks now covered in {spec_file}")


def trace_check_specs(annotated: str, test_cases: list[dict[str, Any]], opt: argparse.Namespace) -> tuple[int, dict[str, Any]]:

    from SpecCoder.spec_utils import build_full_code, extract_assertions_from_annotated_code
    from SpecCoder.trace_collection import _is_valid_eval_expr, instrument_assertions_in_completion
    from SpecCoder.spec_evaluation import compute_correctness

    cases = list(test_cases or [])
    _cleaned, assertions = extract_assertions_from_annotated_code(annotated or "")
    valid = [a for a in assertions if _is_valid_eval_expr(a.expr)]
    num_total = len(assertions)
    num_specs = len(valid)
    base = {"num_specs": num_specs, "num_specs_total": num_total,
            "num_syntax_invalid": num_total - num_specs, "reached_specs": 0,
            "violations": 0, "pass_rate": 0.0, "specs": []}
    if not num_specs:
        return 0, {**base, "failures": [{"index": 0, "detail": "no syntactically-valid assertions"}]}
    if not cases:
        return num_specs, {**base, "failures": [{"index": 0, "detail": "no test_cases"}]}

    instrumented = instrument_assertions_in_completion(annotated, valid)
    full_code, _offset = build_full_code("", instrumented)
    spec_dict = {a.trace_key: {"assert_key": a.assert_key, "dedup_key": a.dedup_key, "expr": a.expr}
                 for a in valid}
    expr_by_tk = {a.trace_key: a.expr for a in valid}
    timeout = max(1, int(opt.eval_timeout or 1))
    corr, timeout_count, _co_reach = compute_correctness(
        full_code, spec_dict, None, [], cases,
        timeout=timeout, is_codeforces=True, test_workers=opt.eval_workers,
    )

    specs: list[dict[str, Any]] = []
    for trace_key, info in corr.items():
        reach = int(info.get("reach_count", 0) or 0)
        passed = int(info.get("pass_count", 0) or 0)
        specs.append({
            "expr": expr_by_tk.get(trace_key, ""),
            "reach_count": reach,      # number of inputs where the spec was reached
            "pass_count": passed,      # number of those inputs where it held everywhere
            "correctness": info.get("correctness", 0.0),
            "violated": bool(reach > 0 and passed < reach),
        })
    reached = [s for s in specs if s["reach_count"] > 0]
    violated = [s for s in reached if s["violated"]]
    pass_rate = (len(reached) - len(violated)) / len(reached) if reached else 1.0
    check = {
        "num_specs": num_specs,
        "num_specs_total": num_total,
        "num_syntax_invalid": num_total - num_specs,
        "reached_specs": len(reached),
        "violations": len(violated),
        "timeout_count": timeout_count,
        "pass_rate": pass_rate,
        "specs": specs,
        "failures": [
            {"index": 0, "detail": f"spec violated: {s['expr']} (held on {s['pass_count']}/{s['reach_count']} reached inputs)"}
            for s in violated[:5]
        ],
    }
    return num_specs, check


_FENCE_RE = re.compile(r"```(?:[A-Za-z0-9_+-]*)\n(.*?)```", re.DOTALL)


def parse_fenced_blocks(text: str, limit: int = 50) -> list[str]:
    """Extract the raw text inside each ``` fenced block (used to parse generated
    stdin inputs)."""
    blocks = [b.strip("\n") for b in _FENCE_RE.findall(text or "")]
    blocks = [b for b in blocks if b.strip()]
    return blocks[:limit]


def parse_yes_no(text: str) -> bool | None:
    """Parse the verify verdict ``<answer>YES|NO</answer>`` (YES = correct)."""
    m = re.search(r"<answer>\s*(YES|NO)\s*</answer>", text or "", re.IGNORECASE)
    if not m:
        return None
    return m.group(1).upper() == "YES"


def _write_call_log(folder: Path, task_id: str, sections: list[tuple[str, str, str]]) -> None:
    """One .txt per task per stage (prompt + LLM response), named by task_id."""
    folder.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for title, prompt, response in sections:
        lines += ["=" * 72, f"[{title}]  task={task_id}", "=" * 72, "",
                  "[PROMPT]", str(prompt), "", "[RESPONSE]", str(response), ""]
    (folder / f"{task_log_name(task_id)}.txt").write_text("\n".join(lines), encoding="utf-8")


def _gen_cases_for_batch(batch, backend, opt, gen_input_dir: Path) -> list[list[dict[str, Any]]]:
    if opt.input_source == "ref":
        return [list(t.get("test_cases") or []) for t in batch]
    user_prompts = [build_input_generation_prompt(
        t.get("problem_description", ""), t.get("input_specification", ""), opt.num_gen_inputs) for t in batch]
    comps = backend.chat(
        [[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}] for p in user_prompts],
        num_samples=1, task_ids=[t["task_id"] for t in batch])
    out: list[list[dict[str, Any]]] = []
    for task, prompt, samples in zip(batch, user_prompts, comps):
        resp = samples[0] if samples else ""
        _write_call_log(gen_input_dir, task["task_id"], [("GEN INPUT", prompt, resp)])
        cases = [{"input": b} for b in parse_fenced_blocks(resp, opt.num_gen_inputs)]
        out.append(cases or list(task.get("test_cases") or []))
    return out


def _exec_summary(info: dict[str, Any]) -> str:
    if not info["num_specs"]:
        return "No inline assertions were generated, so execution could not check anything."
    check = info.get("check", {})
    reached = int(check.get("reached_specs", 0) or 0)
    if info["has_violation"]:
        return ("At least one inline assertion was VIOLATED while tracing execution on the "
                f"{info['n_cases']} inputs (specs are traced, not hard-asserted, so all of them run).\n"
                f"Violated assertion(s):\n{info['violated']}")
    if not reached:
        return (f"None of the {info['num_specs']} inline assertion(s) were reached on the "
                f"{info['n_cases']} inputs, so execution produced no spec signal.")
    return (f"All {reached} reached inline assertion(s) held on the {info['n_cases']} traced inputs "
            "(no violation).")


def _input_evidence_phrase(s: dict[str, Any]) -> str:
    reach = int(s.get("reach_count", 0) or 0)
    passed = int(s.get("pass_count", 0) or 0)
    return (f"(reached by {reach} input(s), of which {passed} satisfy the assertion "
            f"and {reach - passed} do not)")


def _annotate_with_input_evidence(annotated: str, check: dict[str, Any]) -> str:
    specs = (check or {}).get("specs") or []
    if not annotated or not specs:
        return annotated
    by_expr: dict[str, dict[str, Any]] = {}
    for s in specs:
        by_expr.setdefault(str(s.get("expr", "")).strip(), s)
    out_lines = []
    for line in annotated.splitlines():
        stripped = line.strip()
        if stripped.startswith("# assert "):
            expr = stripped[len("# assert "):].strip()
            s = by_expr.get(expr)
            if s is not None:
                line = f"{line}  {_input_evidence_phrase(s)}"
        out_lines.append(line)
    return "\n".join(out_lines)


def _verdicts_for_batch(
    batch: list[dict[str, Any]],
    infos: list[dict[str, Any]],
    backend,
    opt: argparse.Namespace,
) -> tuple[list[bool], list[str], list[str]]:
    verify_resp = [""] * len(batch)
    verify_prompts = [""] * len(batch)

    vmsgs = []
    for idx, (t, info) in enumerate(zip(batch, infos)):
        up = build_assert_prompt(
            t.get("problem_description", ""),
            t.get("input_specification", ""),
            t.get("output_specification", ""),
            # Show each checkpoint annotated with its input reach/satisfy counts, plus
            # the violated-assert report in the execution summary.
            _annotate_with_input_evidence(info.get("annotated") or t.get("code", ""),
                                          info.get("check", {})),
            _exec_summary(info),
        )
        verify_prompts[idx] = up
        vmsgs.append([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": up}])
    vcomps = backend.chat(vmsgs, num_samples=1, task_ids=[t["task_id"] for t in batch])
    predicted = []
    for i, samples in enumerate(vcomps):
        r = samples[0] if samples else ""
        verify_resp[i] = r
        yn = parse_yes_no(r)
        if yn is None:
            yn = False
        predicted.append(yn)
    return predicted, verify_prompts, verify_resp


def run_speccoder_from_spec_file(
    tasks: list[dict[str, Any]],
    backend,
    opt: argparse.Namespace,
    generation_path: str,
    spec_file: str,
) -> list[dict[str, Any]]:
    existing = load_latest_jsonl_by_task(generation_path)
    spec_by_task = load_spec_generation_by_task(spec_file)
    missing_specs = [t["source_task_id"] for t in tasks if t["task_id"] not in existing and t["source_task_id"] not in spec_by_task]
    if missing_specs:
        preview = ", ".join(missing_specs[:5])
        raise SystemExit(f"Missing generated specs in --spec_file for {len(missing_specs)} tasks; first: {preview}")

    pending = [task for task in tasks if task["task_id"] not in existing]
    records: list[dict[str, Any]] = []
    out_dir = Path(generation_path).parent
    gen_input_dir = out_dir / "log_gen_input"
    llm_dir = out_dir / "log_llm"
    for start in range(0, len(pending), max(1, opt.global_batch)):
        batch = pending[start : start + max(1, opt.global_batch)]
        spec_records = [spec_by_task[t["source_task_id"]] for t in batch]
        annotated_list = [str(s.get("annotated_code") or s.get("code") or "") for s in spec_records]
        responses = [str(s.get("response") or "") for s in spec_records]
        prompts = [str(s.get("prompt") or "") for s in spec_records]
        cases_list = _gen_cases_for_batch(batch, backend, opt, gen_input_dir)
        infos: list[dict[str, Any]] = []
        for annotated, cases in zip(annotated_list, cases_list):
            num_specs, check = trace_check_specs(annotated, cases, opt)
            violated_specs = [s for s in (check.get("specs") or []) if s.get("violated")]
            violated_desc = "; ".join(
                f"{s['expr']} (held on {s['pass_count']}/{s['reach_count']} reached inputs)"
                for s in violated_specs[:3]
            )
            infos.append({"num_specs": num_specs, "check": check, "n_cases": len(cases),
                          "annotated": annotated, "failing_input": "", "violated": violated_desc,
                          "has_violation": bool(num_specs and check.get("violations", 0) > 0)})
        predicted, verify_prompts, verify_resp = _verdicts_for_batch(batch, infos, backend, opt)

        for i, task in enumerate(batch):
            sections = [("SPEC FILE", spec_file, responses[i])]
            if verify_prompts[i]:
                sections.append(("VERIFY", verify_prompts[i], verify_resp[i]))
            _write_call_log(llm_dir, task["task_id"], sections)
            info = infos[i]
            spec_rec = spec_records[i]
            rec = {
                **{k: task.get(k) for k in ("task_id", "source_task_id", "candidate_kind", "label")},
                "ground_truth": task.get("label"),  # bool: True = correct, False = buggy
                "benchmark": BENCHMARK,
                "method": opt.method,
                "correctness_mode": opt.correctness_mode,
                "input_source": opt.input_source,
                "model": opt.model,
                "backend": opt.backend,
                "speccoder_finetune": bool(opt.speccoder_finetune),
                "spec_file": spec_file,
                "spec_attempt": spec_rec.get("attempt"),
                "prompt": prompts[i],
                "response": responses[i],
                "code": task.get("code", ""),
                "annotated_code": annotated_list[i],
                "num_specs": info["num_specs"],
                "num_gen_inputs": info["n_cases"],
                "assertion_check": info["check"],
                "failing_input": info["failing_input"] or None,
                "violated_specs": info["violated"] or None,
                "verify_response": verify_resp[i] or None,
                "predicted_correct": bool(predicted[i]),
                "test_cases": task.get("test_cases") if opt.include_test_cases else None,
                "time_latency": {"generation_seconds": 0.0, "global_batch": len(batch), "local_batch": 1},
                "cost_token": token_usage_record(verify_prompts[i], verify_resp[i]) if verify_prompts[i] else {},
            }
            append_jsonl(generation_path, {k: v for k, v in rec.items() if v is not None})
            records.append(rec)
    latest = load_latest_jsonl_by_task(generation_path)
    return [latest[t["task_id"]] for t in tasks if t["task_id"] in latest]


def run_speccoder(tasks: list[dict[str, Any]], backend, opt: argparse.Namespace, generation_path: str) -> list[dict[str, Any]]:
    spec_file = opt.spec_file or default_spec_file(opt)
    ensure_specs_generated(tasks, opt, spec_file)
    if not os.path.exists(spec_file):
        raise SystemExit(f"Spec generation failed — file still missing: {spec_file}")
    return run_speccoder_from_spec_file(tasks, backend, opt, generation_path, spec_file)


def run_llm_judge(tasks: list[dict[str, Any]], backend, opt: argparse.Namespace,
                  generation_path: str, log_dir: str) -> list[dict[str, Any]]:
    
    if opt.method == "cot":
        build_prompt, tag = build_cot_prompt, "COT"
    else:  # vanilla
        build_prompt, tag = build_vanilla_prompt, "VANILLA"
    existing = load_latest_jsonl_by_task(generation_path)
    pending = [t for t in tasks if t["task_id"] not in existing]
    records: list[dict[str, Any]] = []
    llm_dir = Path(generation_path).parent / "log_llm"
    for start in range(0, len(pending), max(1, opt.global_batch)):
        batch = pending[start : start + max(1, opt.global_batch)]
        prompts = [
            build_prompt(t.get("problem_description", ""), t.get("input_specification", ""),
                         t.get("output_specification", ""), t.get("code", ""))
            for t in batch
        ]
        msgs = [[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}] for p in prompts]
        comps = backend.chat(msgs, num_samples=1, task_ids=[t["task_id"] for t in batch])
        for i, task in enumerate(batch):
            resp = comps[i][0] if comps[i] else ""
            yn = parse_yes_no(resp)
            predicted = bool(yn) if yn is not None else False  # unparsable verdict -> NO
            _write_call_log(llm_dir, task["task_id"], [(tag, prompts[i], resp)])
            rec = {
                **{k: task.get(k) for k in ("task_id", "source_task_id", "candidate_kind", "label")},
                "ground_truth": task.get("label"),  # bool: True = correct, False = buggy
                "benchmark": BENCHMARK,
                "method": opt.method,
                "model": opt.model,
                "backend": opt.backend,
                "speccoder_finetune": bool(opt.speccoder_finetune),
                "prompt": prompts[i],
                "response": resp,
                "code": task.get("code", ""),
                "verdict_parsed": yn is not None,
                "predicted_correct": bool(predicted),
                "test_cases": task.get("test_cases") if opt.include_test_cases else None,
                "time_latency": {"generation_seconds": 0.0, "global_batch": len(batch), "local_batch": 1},
                "cost_token": token_usage_record(prompts[i], resp),
            }
            append_jsonl(generation_path, {k: v for k, v in rec.items() if v is not None})
            records.append(rec)
    latest = load_latest_jsonl_by_task(generation_path)
    return [latest[t["task_id"]] for t in tasks if t["task_id"] in latest]


def run_hoareprompt(tasks: list[dict[str, Any]], backend, opt: argparse.Namespace, generation_path: str, log_root: str) -> list[dict[str, Any]]:
    hp_src = PROJECT_ROOT / "SpecCoder" / "others" / "HoarePrompt" / "src"
    if str(hp_src) not in sys.path:
        sys.path.insert(0, str(hp_src))
    import hoareprompt as hp  # type: ignore  # noqa: E402
    import model as hp_model  # type: ignore  # noqa: E402

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    hp_model.set_active_backend(backend)
    existing = load_latest_jsonl_by_task(generation_path)
    config = {
        "model": "__backend__",
        "temperature": opt.temperature,
        "assessment-mode": "postcondition-entailment",
        "postcondition-mode": "hoarecot",
        "postcondition-cot-prompt": "node-based-style",
        "loop-unrolling-count": opt.hoareprompt_unroll,
        "entailment-mode": "naive",
        "cex-mode": "with-postcondition",
        "annotated": True,
        "annotated-type": "simple",
        "concat_simple": True,
        "fsl": True,
        "COT": True,
        "confidence": False,
    }

    pending = [t for t in tasks if t["task_id"] not in existing]
    write_lock = threading.Lock()
    counter = {"done": 0}
    total = len(pending)

    def _assess_one(task: dict[str, Any], global_batch: int) -> None:
        task_log = Path(log_root) / task_log_name(task["task_id"]) / "hoareprompt"
        task_log.mkdir(parents=True, exist_ok=True)
        backend.begin_task(task["task_id"])
        started = time.perf_counter()
        try:
            prediction = bool(hp.assess(
                task.get("problem_text", ""),
                task.get("code", ""),
                task_log_name(task["task_id"]),
                config,
                task_log,
                None,
            ))
            error = None
        except Exception as exc:
            prediction = False
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        rec = {
            **{k: task.get(k) for k in ("task_id", "source_task_id", "candidate_kind", "label")},
            "ground_truth": task.get("label"),  # bool: True = correct, False = buggy
            "benchmark": BENCHMARK,
            "method": "HoarePrompt",
            "model": opt.model,
            "backend": opt.backend,
            "speccoder_finetune": bool(opt.speccoder_finetune),
            "prompt": task.get("problem_text", ""),
            "code": task.get("code", ""),
            "predicted_correct": prediction,
            "error": error,
            "time_latency": {"generation_seconds": round(elapsed, 6),
                             "global_batch": global_batch, "local_batch": 1},
        }
        with write_lock:
            append_jsonl(generation_path, {k: v for k, v in rec.items() if v is not None})
            counter["done"] += 1
            print(f"      [hoareprompt] {counter['done']:>4}/{total}  {task['task_id']:<14} "
                  f"-> {'CORRECT' if prediction else 'buggy'}"
                  + (f"  (error: {error})" if error else ""), flush=True)


    batch_size = max(1, int(getattr(opt, "global_batch", 0) or 1))
    for start in range(0, len(pending), batch_size):
        chunk = pending[start:start + batch_size]
        if len(chunk) > 1 and hasattr(backend, "enable_coalescing"):
            print(f"      [hoareprompt] assessing {len(chunk)} task(s) in parallel "
                  f"(coalesced dispatch)", flush=True)
            backend.enable_coalescing(max_batch=len(chunk))
            try:
                with ThreadPoolExecutor(max_workers=len(chunk)) as ex:
                    futs = [ex.submit(_assess_one, t, len(chunk)) for t in chunk]
                    for fut in as_completed(futs):
                        fut.result()
            finally:
                backend.disable_coalescing()
        else:
            for t in chunk:
                _assess_one(t, 1)

    latest = load_latest_jsonl_by_task(generation_path)
    return [latest[t["task_id"]] for t in tasks if t["task_id"] in latest]


def evaluate_records(records: list[dict[str, Any]], opt: argparse.Namespace, result_path: str, log_dir: str,
                     write_logs: bool = True) -> list[dict[str, Any]]:
    existing = load_latest_jsonl_by_task(result_path)
    out: list[dict[str, Any]] = []
    for rec in records:
        if rec["task_id"] in existing:
            out.append(existing[rec["task_id"]])
            continue
        evaluated = dict(rec)
        if opt.method == "assert" and "predicted_correct" not in rec:
            raise SystemExit(
                "assert generation record is missing the LLM verdict field "
                "`predicted_correct`; regenerate the assertion-checking output."
            )
        label = bool(evaluated.get("label"))
        pred = bool(evaluated.get("predicted_correct"))
        evaluated["is_correct_prediction"] = pred == label
        append_jsonl(result_path, evaluated)
        if not write_logs or opt.method == "assertchecking":
            out.append(evaluated)
            continue
        append_task_log(
            os.path.join(log_dir, f"{task_log_name(rec['task_id'])}.log"),
            {
                **evaluated,
                "response": [evaluated.get("response", "")],
                "code": [evaluated.get("annotated_code") or evaluated.get("code", "")],
                "sample_results": [evaluated.get("assertion_check", {"predicted_correct": pred, "label": label})],
            },
            header=f"method={opt.method} benchmark={BENCHMARK} model={opt.model}",
        )
        out.append(evaluated)
    return out


def build_summary(records: list[dict[str, Any]], opt: argparse.Namespace, total_tasks: int) -> dict[str, Any]:
    n = len(records)
    correct = sum(1 for r in records if r.get("is_correct_prediction"))
    tp = sum(1 for r in records if r.get("label") is True and r.get("predicted_correct") is True)
    tn = sum(1 for r in records if r.get("label") is False and r.get("predicted_correct") is False)
    fp = sum(1 for r in records if r.get("label") is False and r.get("predicted_correct") is True)
    fn = sum(1 for r in records if r.get("label") is True and r.get("predicted_correct") is False)
    return {
        "kind": "correctness_checking",
        "benchmark": BENCHMARK,
        "method": opt.method,
        "spec_file": opt.spec_file or (default_spec_file(opt) if opt.method in SPEC_METHODS else None),
        "model": opt.model,
        "backend": opt.backend,
        "speccoder_finetune": bool(opt.speccoder_finetune),
        "total_tasks": total_tasks,
        "evaluated_tasks": n,
        "accuracy": correct / n if n else 0.0,
        "correct_predictions": correct,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


METRICS_FILE = "metrics.json"


def _compute_and_write_metrics(opt: argparse.Namespace, out_dir: str, evaluated: list[dict[str, Any]],
                               keep_ids: set[str] | None = None,
                               result_filename: str = RESULT_FILE,
                               metrics_filename: str = METRICS_FILE) -> dict[str, Any]:
    records, token_dirs = gather_result_records(out_dir, result_filename=result_filename)
    if keep_ids is not None:
        records = [r for r in records if r.get("task_id") in keep_ids]
    if not records:  # fall back to what we just evaluated (e.g. sibling missing)
        records, token_dirs = list(evaluated), [Path(out_dir)]
    vanilla_mcc = mcc_from_summary(opt.vanilla_summary) if opt.vanilla_summary else None
    metrics = build_metrics(records, token_dirs=token_dirs, vanilla_mcc=vanilla_mcc)
    metrics_path = os.path.join(out_dir, metrics_filename)
    write_json(metrics_path, metrics)
    print(format_metrics_table(metrics, title=f"{opt.method} / {BENCHMARK}"))
    print(f"Metrics file    -> {metrics_path}")
    return metrics


def _out_dir_for(opt: argparse.Namespace) -> str:
    leaf = os.path.join("speccoder", opt.method) if opt.speccoder_finetune else opt.method
    return str(experiment_output_dir("correctness_checking", opt.model, os.path.join(BENCHMARK, leaf)))


def run(opt: argparse.Namespace) -> None:
    tasks = load_tasks(opt)
    task_by_id = {task["task_id"]: task for task in tasks}
    is_spec = opt.method in SPEC_METHODS
    if is_spec and not opt.spec_file:
        opt.spec_file = default_spec_file(opt)
    out_dir = _out_dir_for(opt)
    generation_path = os.path.join(out_dir, GENERATION_FILE)
    result_path = os.path.join(out_dir, RESULT_FILE)
    summary_path = os.path.join(out_dir, SUMMARY_FILE)
    log_dir = os.path.join(out_dir, "log")
    os.makedirs(log_dir, exist_ok=True)

    do_generate = not opt.eval_only
    do_eval = not opt.no_eval
    mode = "eval-only" if opt.eval_only else ("generate-only" if opt.no_eval else "generate+eval")
    print(f"benchmark={BENCHMARK} method={opt.method} "
          f"speccoder_finetune={bool(opt.speccoder_finetune)} tasks={len(tasks)} mode={mode}")
    if is_spec:
        print(f"spec_file={opt.spec_file}")

    existing_gen = load_latest_jsonl_by_task(generation_path)
    missing_gen = [t for tid, t in task_by_id.items() if tid not in existing_gen]
    if do_generate and not missing_gen and existing_gen:
        print(f"Generation already complete ({len(existing_gen)}/{len(task_by_id)} tasks) "
              f"-> skip model, compute result from generation.")
        do_generate = False
    if do_generate:
        print(f"Generating {len(missing_gen)}/{len(task_by_id)} missing task(s) (model loads).")
        backend = make_backend(opt)
        if opt.method == "assert":
            generated = run_speccoder(tasks, backend, opt, generation_path)
        elif opt.method in ("vanilla", "cot"):
            generated = run_llm_judge(tasks, backend, opt, generation_path, log_dir)
        elif opt.method == "HoarePrompt":
            generated = run_hoareprompt(tasks, backend, opt, generation_path, log_dir)
        else:  # assertchecking
            generated = assertchecking.run(tasks, backend, opt, generation_path, log_dir)
        write_token_files(backend, out_dir, allowed_task_ids=set(task_by_id))
        print(f"Generation file -> {generation_path}")
    else:
        # eval-only OR generation already complete: score existing records; no backend.
        generated = list(existing_gen.values())
        if not generated:
            raise SystemExit(f"No generation records found at {generation_path} "
                             f"(need to generate first; remove --eval-only).")
        print(f"Loaded {len(generated)} generation records from {generation_path}")

    if not do_eval:
        print("Skipping evaluation/metrics (--no-eval). Run again with --eval-only to score.")
        return

    # ----- evaluation phase -----
    keep_ids = set(task_by_id)
    generated = [r for r in generated if r.get("task_id") in keep_ids]
    for record in generated:
        task = task_by_id.get(record.get("task_id"))
        if task and "test_cases" not in record:
            record["test_cases"] = task.get("test_cases") or []
    evaluated = evaluate_records(generated, opt, result_path, log_dir)
    summary = build_summary(evaluated, opt, total_tasks=len(tasks))
    summary["metrics"] = _compute_and_write_metrics(opt, out_dir, evaluated, keep_ids=keep_ids)
    write_json(summary_path, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "metrics"}, indent=2))
    print(f"Result file     -> {result_path}")
    print(f"Summary file    -> {summary_path}")
    print(f"Task logs       -> {log_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correctness-checking runner.")
    parser.add_argument("--method", required=True,
                        choices=["vanilla", "cot", "assert", "assertchecking", "HoarePrompt"],
                        help="Correctness classifier: 'vanilla' (one LLM call, direct YES/NO on the raw candidate), "
                             "'cot' (chain-of-thought reasoning on the raw candidate), 'assert' (the spec-guided "
                             "SpecCoder pipeline: generate inline # asserts, execute them, judge the annotated "
                             "candidate with the execution result), 'HoarePrompt' (formal postcondition entailment). "
                             "Combine any method with --speccoder_finetune to swap base->fine-tuned model.")
    parser.add_argument("--input", default=None,
                        help="HumanChecking JSONL path (defaults to data/humanchecking.jsonl); "
                             "one candidate + one `label` verdict per row.")
    parser.add_argument("--spec_file", "--spec-file", default=None,
                        help="Explicit inline oracle spec generation.jsonl path. If omitted, defaults to "
                             "output/spec_gen/inline/<model>/oracle/<base_model|speccoder>/generation.jsonl.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--global-batch", "--global_batch", type=int, default=8)
    parser.add_argument("--eval-timeout", type=float, default=3.0)
    parser.add_argument("--eval-workers", type=int, default=32)
    parser.add_argument("--include-test-cases", action="store_true")
    parser.add_argument("--input-source", "--input_source", dest="input_source",
                        choices=["llm", "ref"], default="ref")
    parser.add_argument("--num-gen-inputs", type=int, default=50,
                        help="Number of stdin inputs to generate per task for the assert run (edge-case coverage).")
    parser.add_argument("--hoareprompt-unroll", type=int, default=0)
    parser.add_argument("--source-test", "--source_test", dest="source_test",
                        choices=["oracle"], default="oracle",
                        help="assertchecking: where the test INPUTS for reach/satisfy come from. "
                             "'oracle' uses the dataset's real public test_cases.")
    parser.add_argument("--judge-finetune", "--judge_finetune", dest="judge_finetune",
                        choices=["auto", "base", "finetune"], default="auto",
                        help="assertchecking: which model runs the correctness JUDGE (verdict) call. "
                             "'auto' (default) follows --speccoder_finetune: the fine-tuned SpecCoder "
                             "LoRA when finetuning, the base model otherwise. 'base' forces the base "
                             "model; 'finetune' forces the LoRA (needs --speccoder_finetune / an adapter). "
                             "The spec-generation step is unaffected — it always uses the LoRA under "
                             "--speccoder_finetune.")
    # Generation / evaluation staging.
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument("--no-eval", "--no_eval", dest="no_eval", action="store_true",
                            help="Generate only; skip scoring and the metric table.")
    eval_group.add_argument("--eval-only", "--eval_only", dest="eval_only", action="store_true",
                            help="Score already-generated records and compute metrics; no LLM generation, no backend loaded.")
    parser.add_argument("--vanilla-summary", "--vanilla_summary", dest="vanilla_summary", default=None,
                        help="Path to a Vanilla baseline summary.json/metrics.json; its MCC is used as MCC_v "
                             "for the relative dMCC improvement.")
    add_backend_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
