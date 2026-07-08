from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_THIS = Path(__file__).resolve()
PROJECT_ROOT = next((p for p in _THIS.parents if (p / "SpecCoder").is_dir()), _THIS.parents[5])
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from SpecCoder.src.experiment.gen_utils import (  # noqa: E402
    GENERATION_FILE,
    RESULT_FILE,
    SUMMARY_FILE,
    append_jsonl,
    append_task_log,
    experiment_output_dir,
    load_latest_jsonl_by_task,
    task_log_name,
    write_json,
)
from SpecCoder.src.experiment.programrepair import (  # noqa: E402
    assertrepair,
    backbones,
    common,
    cot,
    hoareprompt,
    specrover,
    vanilla,
)
from SpecCoder.src.experiment.backend import (  # noqa: E402
    add_backend_args,
    append_token_records,
    make_backend,
)
from SpecCoder.src.codeforces_eval import estimate_pass_at_k  # noqa: E402

METHODS = {m.NAME: m for m in (vanilla, cot, hoareprompt, specrover, assertrepair)}

SIGNAL_IMPROVE = {
    "hoareprompt": hoareprompt.improve_batch,
    "specrover": specrover.improve_batch,
    "assertrepair": assertrepair.improve_batch,
}

BACKBONE_RUN = {
    "purellm": backbones.run_purellm,
}
BACKBONES = set(BACKBONE_RUN)


def _drop_unsaved(recs: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    kept = [r for r in recs if not r.get("save_skip")]
    dropped = len(recs) - len(kept)
    if dropped:
        skipped = [r.get("task_id") for r in recs if r.get("save_skip")]
        print(f"[{label}] dropped {dropped} unsaved record "
              f"{skipped}", flush=True)
    return kept


def load_attempts_by_task(path: str) -> dict[str, dict[int, dict[str, Any]]]:
    attempts: dict[str, dict[int, dict[str, Any]]] = {}
    if not os.path.exists(path):
        return attempts
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            tid = rec.get("task_id")
            attempt = rec.get("attempt")
            if tid is None or attempt is None:
                continue
            attempts.setdefault(str(tid), {})[int(attempt)] = rec
    return attempts


def report(records: list[dict[str, Any]]) -> None:
    n = len(records)
    if not n:
        return
    fixed = sum(1 for r in records if r.get("num_correct", 0) > 0)
    old_atp = sum(r.get("old_pass_rate", 0.0) for r in records) / n
    new_atp = sum(r.get("new_pass_rate", 0.0) for r in records) / n
    initially_correct = sum(1 for r in records if r.get("old_pass_rate") == 1.0)
    print("-" * 60)
    print(f"Programs            : {n}  (initially correct: {initially_correct})")
    print(f"At least one repair : {fixed}/{n}  ({fixed / n:.2%})")
    print(f"Best ATP            : {old_atp:.4f} -> {new_atp:.4f}")
    pass_k_keys = sorted({k for r in records for k in (r.get("pass_at_k") or {})})
    for key in pass_k_keys:
        vals = [r["pass_at_k"][key] for r in records if key in (r.get("pass_at_k") or {})]
        if vals:
            print(f"{key:<20}: {sum(vals) / len(vals):.4f}")


def build_summary(records: list[dict[str, Any]], opt: argparse.Namespace, *, total_tasks: int) -> dict[str, Any]:
    evaluated_tasks = len(records)
    initially_correct = sum(1 for record in records if record.get("old_pass_rate") == 1.0)
    repaired_tasks = sum(1 for record in records if record.get("num_correct", 0) > 0)
    old_atp = sum(record.get("old_pass_rate", 0.0) for record in records) / evaluated_tasks if evaluated_tasks else 0.0
    best_atp = sum(record.get("new_pass_rate", 0.0) for record in records) / evaluated_tasks if evaluated_tasks else 0.0

    total_samples = 0
    fully_correct_samples = 0
    mean_sample_pass_rate = 0.0
    for record in records:
        sample_results = record.get("sample_results") or []
        if not sample_results:
            continue
        total_samples += len(sample_results)
        fully_correct_samples += sum(1 for sample in sample_results if sample.get("pass_rate") == 1.0)
        mean_sample_pass_rate += (
            sum(sample.get("pass_rate", 0.0) for sample in sample_results) / len(sample_results)
        )

    pass_k_keys = sorted({key for record in records for key in (record.get("pass_at_k") or {})})
    pass_at_k = {
        key: sum(record["pass_at_k"][key] for record in records if key in (record.get("pass_at_k") or {}))
        / sum(1 for record in records if key in (record.get("pass_at_k") or {}))
        for key in pass_k_keys
    }
    failing = [str(r.get("task_id")) for r in records
               if r.get("task_id") is not None and r.get("num_correct", 0) == 0]
    return {
        "kind": "repair",
        "method": opt.method,
        "backbone_framework": opt.backbone_framework,
        "model": opt.model,
        "backend": opt.backend,
        "speccoder_finetune": bool(opt.speccoder_finetune),
        "requested_num_attempts": opt.num_samples,
        "total_tasks": total_tasks,
        "evaluated_tasks": evaluated_tasks,
        "initially_correct_tasks": initially_correct,
        "initially_correct_rate": (initially_correct / evaluated_tasks) if evaluated_tasks else 0.0,
        "repaired_tasks": repaired_tasks,
        "repair_rate": (repaired_tasks / evaluated_tasks) if evaluated_tasks else 0.0,
        "old_mean_pass_rate": old_atp,
        "best_mean_pass_rate": best_atp,
        "mean_sample_pass_rate": (mean_sample_pass_rate / evaluated_tasks) if evaluated_tasks else 0.0,
        "total_samples": total_samples,
        "fully_correct_samples": fully_correct_samples,
        "sample_full_pass_rate": (fully_correct_samples / total_samples) if total_samples else 0.0,
        "pass_at_k": pass_at_k,
        "num_failing_tasks": len(failing),
        "failing_tasks": ",".join(failing),
        "eval_timeout": opt.eval_timeout,
        "test_batch": opt.test_batch,
    }


def _pass_k_values(num_samples: int, num_correct: int, ks: list[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        if num_samples >= k:
            out[f"pass@{k}"] = float(estimate_pass_at_k(num_samples, [num_correct], k)[0])
    return out


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        return [items]
    return [items[start : start + size] for start in range(0, len(items), size)]


def run(opt: argparse.Namespace) -> None:
    if opt.eval_only and opt.no_eval:
        raise SystemExit("--eval-only conflicts with --no-eval")
    opt.eval_workers = opt.test_batch
    opt.eval = not opt.no_eval
    rows = common.load_jsonl(opt.input, None if opt.task_ids else opt.limit)
    if opt.task_ids:
        rows = common.select_task_ids(rows, opt.task_ids)
        print(f"--task-ids: processing {len(rows)} selected task(s): "
              f"{', '.join(common.task_id(r) for r in rows)}")
    print(
        f"backbone={opt.backbone_framework} method={opt.method} "
        f"backend={opt.backend} model={opt.model} programs={len(rows)}"
    )

    if opt.output:
        out_dir = os.path.dirname(os.path.abspath(opt.output))
        os.makedirs(out_dir, exist_ok=True)
    else:
        # fine-tuned runs go under a speccoder/ subfolder so they don't overwrite base.
        method_leaf = opt.method
        if opt.speccoder_finetune:
            method_leaf = os.path.join("speccoder", method_leaf)
        out_dir = str(experiment_output_dir("repair", opt.model, method_leaf))
    _dc = Path(out_dir)
    for _anc in [_dc, *_dc.parents]:
        if _anc.name == "repair":
            _dc = _anc
            break
    opt.designer_cache_path = str(_dc / "agent_coder_test_designer.jsonl")
    generation_path = os.path.join(out_dir, GENERATION_FILE)
    result_path = os.path.join(out_dir, RESULT_FILE)
    summary_path = os.path.join(out_dir, SUMMARY_FILE)
    log_dir = os.path.join(out_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    common.configure_live_log(log_dir)
    if opt.backbone_framework == "purellm" and getattr(opt, "localize", True):
        loc_dir = experiment_output_dir("localize", opt.model, opt.method)
        opt.localize_path = str(loc_dir / "localize.jsonl")
        opt.localize_prompt_path = str(loc_dir / "localize_prompt.jsonl")
    else:
        opt.localize_path = None
        opt.localize_prompt_path = None

    existing_generation = load_attempts_by_task(generation_path)
    target_by_task = {common.task_id(sample): sample for sample in rows}
    todo_rows = []
    for sample in rows:
        tid = common.task_id(sample)
        existing_attempts = existing_generation.get(tid, {})
        if any(attempt not in existing_attempts for attempt in range(opt.num_samples)):
            todo_rows.append(sample)
    print(f"already complete: {len(rows) - len(todo_rows)} | tasks to generate missing attempts: {len(todo_rows)}")
    if getattr(opt, "localize_only", False):
        if opt.backbone_framework != "purellm" or not opt.localize_path:
            print("--localize-only is a pure-backbone feature; nothing to do.")
            return
        backend = make_backend(opt)
        n_loc = backbones.purellm_localize_replay(rows, existing_generation, backend, opt)
        print(f"[localize-only] localized {n_loc} task(s) from saved prompts -> {opt.localize_path}")
        return

    if opt.eval_only:
        todo_rows = []
        print("--eval-only set: skipping generation and evaluating existing generation.jsonl")
    result_cache = load_latest_jsonl_by_task(result_path) if opt.eval else {}

    def eval_task_ids(task_ids) -> int:
        if not opt.eval:
            return 0
        done = 0
        for tid in task_ids:
            sample = target_by_task.get(tid)
            if sample is None:
                continue
            cached = result_cache.get(tid)
            if cached and len(cached.get("sample_results") or []) >= opt.num_samples:
                continue
            attempts_map = existing_generation.get(tid, {})
            attempts = [attempts_map[i] for i in sorted(attempts_map) if i < opt.num_samples]
            if len(attempts) < opt.num_samples:
                continue  # not finished across all attempts yet
            rec = {
                "task_id": tid,
                "contestId": sample.get("contestId"),
                "index": sample.get("index"),
                "name": sample.get("name"),
                "method": opt.method,
                "backbone_framework": opt.backbone_framework,
                "model": opt.model,
                "backend": opt.backend,
                "speccoder_finetune": bool(opt.speccoder_finetune),
                "buggy_code": common.buggy_code_of(sample),
            }
            # old_pass_rate is the unchanged buggy solution's pass-rate (method-agnostic);
            # prefer the precomputed cache, fall back to evaluating on the fly.
            buggy_meta = sample.get("buggy_metadata")
            if isinstance(buggy_meta, dict) and "pass_rate" in buggy_meta:
                rec["old_pass_rate"] = float(buggy_meta["pass_rate"])
            else:
                rec["old_pass_rate"] = common.evaluate(
                    rec["buggy_code"], sample, opt.eval_timeout)["pass_rate"]
            sample_results = []
            for attempt_rec in attempts:
                new = common.evaluate(attempt_rec.get("code", ""), sample, opt.eval_timeout,
                                       test_workers=common.test_batch_of(opt))
                new["attempt"] = int(attempt_rec.get("attempt", 0))
                sample_results.append(new)
            num_samples = len(sample_results)
            num_correct = sum(1 for r in sample_results if r.get("pass_rate") == 1.0)
            best = max(sample_results, key=lambda r: r.get("pass_rate", 0.0), default={})
            rec["attempts"] = attempts
            rec["sample_results"] = sample_results
            rec["num_attempts"] = num_samples
            rec["num_correct"] = num_correct
            rec["new_pass_rate"] = best.get("pass_rate", 0.0)
            rec["passed_tests"] = best.get("passed_tests", 0)
            rec["total_tests"] = best.get("total_tests", 0)
            rec["pass_at_k"] = _pass_k_values(num_samples, num_correct, list(opt.pass_k))
            rec["total_time_latency"] = common.total_attempt_latency(attempts)
            rec["total_cost_token"] = common.total_attempt_cost(attempts)
            append_jsonl(result_path, rec)
            result_cache[tid] = rec
            # Footer on the same per-task log + a terminal line so the eval is visible live.
            common.live_log(
                tid, "EVAL RESULT",
                result=json.dumps({
                    "num_correct": rec["num_correct"],
                    "num_attempts": rec["num_attempts"],
                    "new_pass_rate": rec["new_pass_rate"],
                    "old_pass_rate": rec.get("old_pass_rate"),
                    "pass_at_k": rec["pass_at_k"],
                    "per_attempt_pass_rate": [r.get("pass_rate") for r in sample_results],
                }, ensure_ascii=False, indent=2),
            )
            common.stage_done(
                tid, "EVAL DONE",
                info=f"pass_rate={rec['new_pass_rate']:.4f} "
                     f"({rec['passed_tests']}/{rec['total_tests']}), {num_correct}/{num_samples} correct")
            done += 1
        return done

    if todo_rows:
        backend = make_backend(opt)
        improve_fn = SIGNAL_IMPROVE.get(opt.method)  # None for naive/cot

        saved_num = opt.num_samples
        batch_size = int(getattr(opt, "global_batch", 0) or getattr(opt, "task_batch_size", 0) or 0)

        for attempt in range(opt.num_samples if todo_rows else 0):
            rows_a = [
                s for s in todo_rows
                if attempt not in existing_generation.get(common.task_id(s), {})
            ]
            if not rows_a:
                continue
            batches = _chunks(rows_a, batch_size)
            print(f"[attempt {attempt}/{opt.num_samples - 1}] {len(rows_a)} task(s) "
                  f"in {len(batches)} batch(es) of up to {batch_size or len(rows_a)} "
                  f"(backbone={opt.backbone_framework} method={opt.method})")
            opt.num_samples = 1
            opt.existing_attempts_by_task = None
            for bi, batch in enumerate(batches):
                print(f"[attempt {attempt}][batch {bi + 1}/{len(batches)}] "
                      f"repairing {len(batch)} task(s) ...", flush=True)
                common.init_task_logs(
                    [common.task_id(s) for s in batch],
                    header=f"ATTEMPT {attempt}  |  {opt.backbone_framework} × {opt.method}  |  {opt.model}")
                recs = BACKBONE_RUN[opt.backbone_framework](batch, backend, opt, improve_fn)
                recs = _drop_unsaved(recs, f"attempt {attempt}][batch {bi + 1}/{len(batches)}")
                for rec in recs:
                    rec["attempt"] = attempt
                    append_jsonl(generation_path, rec)
                    existing_generation.setdefault(rec["task_id"], {})[attempt] = rec
                # token.jsonl mirrors generation.jsonl: one APPENDED line per (task_id,
                # attempt) with this attempt's real input/output/total tokens.
                append_token_records(out_dir, recs, with_attempt=True, filename_suffix=ab_tag, allowed_task_ids=set(target_by_task))
                print(f"[attempt {attempt}][batch {bi + 1}/{len(batches)}] "
                      f"done ({len(recs)} record(s) written).", flush=True)
                # Eval THIS batch's finished tasks right now (same test concurrency as the
                # initial split) and write result.jsonl before moving to the next batch.
                if opt.eval:
                    n_eval = eval_task_ids([common.task_id(s) for s in batch])
                    print(f"[attempt {attempt}][batch {bi + 1}/{len(batches)}] "
                          f"evaluated {n_eval} finished task(s) -> result.jsonl", flush=True)
            opt.num_samples = saved_num

    elif opt.eval_only and opt.eval:
        eo_batch_size = int(getattr(opt, "global_batch", 0) or getattr(opt, "task_batch_size", 0) or 0)
        eo_tids = [common.task_id(s) for s in rows if common.task_id(s) in existing_generation]
        eo_batches = _chunks(eo_tids, eo_batch_size)
        print(f"[eval-only] {len(eo_tids)} task(s) from generation.jsonl in "
              f"{len(eo_batches)} batch(es) of up to {eo_batch_size or len(eo_tids)}")
        for bi, batch in enumerate(eo_batches):
            n_eval = eval_task_ids(batch)
            print(f"[eval-only][batch {bi + 1}/{len(eo_batches)}] "
                  f"evaluated {n_eval} task(s) -> result.jsonl", flush=True)

    latest_generation = load_attempts_by_task(generation_path)
    for tid, attempts_map in latest_generation.items():
        existing_generation.setdefault(tid, {}).update(attempts_map)
    n_final_eval = eval_task_ids(list(target_by_task.keys()))
    total_available = sum(1 for tid in target_by_task if tid in latest_generation)

    if opt.eval:
        result_records = list(load_latest_jsonl_by_task(result_path).values())
        scoped_records = [r for r in result_records if r.get("task_id") in target_by_task]
        report(scoped_records)
        write_json(summary_path, build_summary(scoped_records, opt, total_tasks=len(rows)))

    print(f"Generation file -> {generation_path}")
    print(f"Generated this run: {len(todo_rows)} task groups | total available: {total_available}/{len(rows)}")
    if opt.eval:
        print(f"Result file     -> {result_path}")
        print(f"Summary file    -> {summary_path}")
        print(f"Evaluated in final sweep: {n_final_eval} (rest evaluated per batch); "
              f"total in result.jsonl: {len(result_cache)}")
        print(f"Wrote per-task logs                         -> {log_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Program-repair runner (Table 5).")
    parser.add_argument("--method", required=True, choices=list(METHODS),
                        help="The feedback signal. naive/cot carry no signal.")
    parser.add_argument("--backbone-framework", "--backbone_framework", "--backbone",
                        dest="backbone_framework", default="purellm", choices=sorted(BACKBONES),
                        help="Harness: purellm (patch_0 + 1 signal improve).")
    add_backend_args(parser)
    parser.add_argument("--local-batch", "--local_batch", "--n-samples", "--n_samples",
                        dest="num_samples", type=int,
                        help="Number of responses/attempts per task.")
    parser.add_argument("--input", default=common.DEFAULT_INPUT)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-ids", "--task_ids", dest="task_ids", nargs="+", default=None,
                        metavar="TASK_ID",
                        help="process ONLY these task id(s) (e.g. --task-ids cf_141A cf_4A, or "
                             "comma-separated). Overrides --limit; batching is irrelevant in this mode.")
    parser.add_argument("--eval-timeout", type=float, default=3.0)
    parser.add_argument("--test-batch", "--test_batch", "--eval-workers", "--eval_workers",
                        dest="test_batch", type=int, default=16,
                        help="THE single test-concurrency knob: how many test cases run at once within "
                             "ONE task, everywhere tests execute (in-loop assertion checks, "
                             "method-specific checks — AND the final post-generation grading pass). "
                             "Tasks always run sequentially; tests inside a task run in batches of this "
                             "size. ``--eval-workers`` is an alias.")
    parser.add_argument("--pass-k", type=int, nargs="+", default=[1, 3, 5],
                        help="Pass@k values to compute when evaluation is enabled.")
    parser.add_argument("--global-batch", "--global_batch", "--task-batch-size",
                        dest="global_batch", type=int, default=32,
                        help="Number of tasks processed together per batch (default 32). A batch is "
                             "carried to completion before the next starts. Pass 0 for a single batch "
                             "over the whole dataset.")
    parser.add_argument("--include-test-cases", action="store_true",
                        help="Store full test_cases in generation/result records for debugging.")
    parser.add_argument("--eval", action="store_true",
                        help="(deprecated: evaluation runs by default; use --no-eval to skip).")
    parser.add_argument("--no-eval", "--no_eval", action="store_true",
                        help="Skip evaluation (generation only).")
    parser.add_argument("--eval-only", "--eval_only", action="store_true",
                        help="Skip generation and evaluate the existing generation.jsonl.")
    parser.add_argument("--spec-regen", "--spec_regen", "--spec-retry", "--spec_retry",
                        dest="spec_retry", type=int, default=5,
                        help="assertrepair: number of spec-generation attempts when no violated spec "
                             "is found (first generation + regenerations; default 5). "
                             "--spec-retry is a deprecated alias.")
    parser.add_argument("--no-localize", "--no_localize",
                        dest="localize", action="store_false", default=True,
                        help="pure backbone: disable the localize pass ENTIRELY — no prompt built, "
                             "no localize.jsonl, nothing saved. Localize runs by default for every "
                             "method under --backbone purellm.")
    parser.add_argument("--skip-localize", "--skip_localize",
                        dest="skip_localize_call", action="store_true", default=False,
                        help="pure backbone: still BUILD the localize prompt and save it to "
                             "localize_prompt.jsonl (upsert by task_id, from the final method "
                             "context) but SKIP the localize LLM call / localize.jsonl. "
                             "Replay it later with --localize-only.")
    parser.add_argument("--localize-only", "--localize_only",
                        dest="localize_only", action="store_true", default=False,
                        help="pure backbone: skip repair generation AND eval; load each task's "
                             "saved prompt from localize_prompt.jsonl and run ONLY the localize "
                             "LLM call -> localize.jsonl. Companion to --skip-localize.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
