
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_THIS = Path(__file__).resolve()
PROJECT_ROOT: Path | None = None
for _parent in _THIS.parents:
    if (_parent / "SpecCoder").is_dir():
        PROJECT_ROOT = _parent
        break
if PROJECT_ROOT is None:
    PROJECT_ROOT = _THIS.parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from SpecCoder.src.experiment.backend import add_backend_args, make_backend, write_token_files  # noqa: E402
from SpecCoder.src.experiment.gen_utils import (  # noqa: E402
    GENERATION_FILE,
    RESULT_FILE,
    SUMMARY_FILE,
    append_jsonl,
    append_task_log,
    extract_code,
    load_latest_jsonl_by_task,
    spec_gen_output_dir,
    strip_assert_comments,
    task_log_name,
    token_usage_record,
)
from SpecCoder.src.experiment.specgen.inline.prompt import (  # noqa: E402
    SYSTEM_PROMPT,
    build_spec_generation_prompt,
)

DEFAULT_HUMANEXEC = str(PROJECT_ROOT / "SpecCoder" / "data" / "humanexec.jsonl")
DEFAULT_MUTANTS = str(PROJECT_ROOT / "SpecCoder" / "data" / "mutants" / "HumanExec_mutant_prune.jsonl")
ORACLE_NAME = "oracle"
ORACLE_SETTING_NAME = "Oracle HumanExec Spec Generation"


def _strip_mode_from_argv(argv: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--mode":
            if i + 1 >= len(argv):
                raise SystemExit("--mode requires a value")
            mode = argv[i + 1]
            if mode != "oracle":
                raise SystemExit("Spec generation now supports only --mode oracle.")
            i += 2
            continue
        if arg.startswith("--mode="):
            mode = arg.split("=", 1)[1]
            if mode != "oracle":
                raise SystemExit("Spec generation now supports only --mode oracle.")
            i += 1
            continue
        out.append(arg)
        i += 1
    return out


def oracle_output_paths(opt: argparse.Namespace) -> tuple[str, str]:
    if opt.output:
        out_dir = os.path.abspath(opt.output)
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = str(spec_gen_output_dir(
            "inline", opt.model, mode=ORACLE_NAME,
            speccoder_finetune=opt.speccoder_finetune))
    return os.path.join(out_dir, GENERATION_FILE), out_dir


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def task_id(record: dict[str, Any]) -> str:
    return record.get("task_id") or f"cf_{record.get('contestId')}{record.get('index')}"


def iter_chunks(items: list[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        size = len(items) or 1
    return [items[i : i + size] for i in range(0, len(items), size)]


def effective_num_tasks(opt: argparse.Namespace) -> int | None:
    return opt.num_tasks if getattr(opt, "num_tasks", None) is not None else opt.limit


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


def build_problem_text_light(row: dict[str, Any]) -> str:
    if row.get("problem_description"):
        return str(row["problem_description"])
    sections = [
        ("Problem", row.get("problem-description") or row.get("prompt")),
        ("Input Specification", row.get("input-specification")),
        ("Output Specification", row.get("output-specification")),
        ("Sample Input", row.get("demo-input")),
        ("Sample Output", row.get("demo-output")),
        ("Note", row.get("note")),
    ]
    return "\n\n".join(f"### {title}\n{value}" for title, value in sections if value)


def oracle_source_solution(row: dict[str, Any]) -> str:
    return str(row.get("correct_solution") or row.get("completion") or "").strip()


def compact_quality_summary(quality: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "avg_inline_correctness": ("avg_inline_correctness", "macro_task_correctness", "avg_correctness"),
        "avg_inline_completeness": ("avg_inline_completeness", "macro_task_completeness", "avg_completeness"),
        "avg_post_correctness": ("avg_post_correctness", "post_correctness"),
        "avg_post_completeness": ("avg_post_completeness", "post_completeness"),
    }

    def pick(name: str, default: Any = 0.0) -> Any:
        for key in aliases.get(name, (name,)):
            if key in quality:
                return quality.get(key)
        return quality.get(name, default)

    ordered_keys = [
        "setting",
        "num_tasks",
        "num_attempts",
        "num_specs",
        "num_syntax_valid_specs",
        "num_syntax_invalid_specs",
        "num_counted_syntax_invalid_specs",
        "avg_specs_per_task",
        "avg_specs_per_attempt",
        "avg_middle_specs_per_task",
        "avg_post_specs_per_task",
        "avg_inline_correctness",
        "avg_inline_completeness",
        "avg_post_correctness",
        "avg_post_completeness",
        "num_assertions_with_completeness_eval",
        "num_tasks_without_valid_specs",
        "num_attempts_without_valid_specs",
        "heatmap",
    ]
    summary: dict[str, Any] = {}
    for key in ordered_keys:
        if key in aliases:
            summary[key] = pick(key)
        elif key in quality:
            summary[key] = quality[key]
    return summary


def write_oracle_quality(records: list[dict[str, Any]], opt: argparse.Namespace, out_dir: str) -> dict[str, Any]:
    from SpecCoder.src.experiment.specgen.inline.quality import aggregate_quality_from_results

    quality = aggregate_quality_from_results(
        records,
        ORACLE_SETTING_NAME,
        drop_unevaluable=getattr(opt, "drop_unevaluable_assertions", True),
    )
    summary = compact_quality_summary(quality)
    summary_path = os.path.join(out_dir, SUMMARY_FILE)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"Spec-quality: tasks={quality['num_tasks']} attempts={quality['num_attempts']} "
        f"specs={quality['num_specs']} correctness={quality['avg_correctness']} "
        f"completeness={quality['avg_completeness']}"
    )
    print(f"Summary file     -> {summary_path}")
    return quality


def run_oracle(opt: argparse.Namespace) -> None:
    generation_path, out_dir = oracle_output_paths(opt)
    result_path = os.path.join(out_dir, RESULT_FILE)
    log_dir = os.path.join(out_dir, "log")
    os.makedirs(log_dir, exist_ok=True)

    if opt.eval_only and opt.no_eval:
        raise SystemExit("--eval-only conflicts with --no-eval")
    opt.eval = not opt.no_eval
    if opt.eval_only:
        print("--eval-only set: skipping generation and evaluating existing generation.jsonl")

    rows = [row for row in load_jsonl(opt.humanexec) if oracle_source_solution(row)]
    task_limit = effective_num_tasks(opt)
    if task_limit:
        rows = rows[: task_limit]
    row_by_tid = {task_id(row): row for row in rows}

    existing_generation = load_attempts_by_task(generation_path)
    prompt_by_tid: dict[str, str] = {}
    clean_by_tid: dict[str, str] = {}
    missing_by_task: list[tuple[str, list[int]]] = []

    for tid, row in row_by_tid.items():
        clean = strip_assert_comments(oracle_source_solution(row))
        clean_by_tid[tid] = clean
        prompt_by_tid[tid] = build_spec_generation_prompt({"problem_text": build_problem_text_light(row)}, clean)
        missing = [attempt for attempt in range(opt.num_samples) if attempt not in existing_generation.get(tid, {})]
        if missing:
            missing_by_task.append((tid, missing))

    missing_count = sum(len(attempts) for _tid, attempts in missing_by_task)
    print(
        f"HumanExec tasks with correct_solution: {len(row_by_tid)} "
        f"| missing attempts: {missing_count}"
    )

    if missing_by_task and not opt.eval_only:
        backend = make_backend(opt)
        for chunk in iter_chunks(missing_by_task, opt.global_batch):
            num_samples = max(len(attempts) for _tid, attempts in chunk)
            messages_list = [
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt_by_tid[tid]}]
                for tid, _attempts in chunk
            ]
            chunk_task_ids = [tid for tid, _attempts in chunk]
            started_at = time.perf_counter()
            chat_with_metadata = getattr(backend, "chat_with_metadata", None)
            if callable(chat_with_metadata):
                completions, completion_metadata = chat_with_metadata(messages_list, num_samples=num_samples, task_ids=chunk_task_ids)
            else:
                completions = backend.chat(messages_list, num_samples=num_samples, task_ids=chunk_task_ids)
                completion_metadata = [[{} for _ in list(samples or [])] for samples in completions]
            batch_seconds = time.perf_counter() - started_at

            for (tid, attempts), samples, samples_meta in zip(chunk, completions, completion_metadata):
                row = row_by_tid[tid]
                for attempt, response, sample_meta in zip(attempts, list(samples or []), list(samples_meta or [])):
                    prompt = prompt_by_tid[tid]
                    rec = {
                        "task_id": tid,
                        "attempt": attempt,
                        "contestId": row.get("contestId"),
                        "index": row.get("index"),
                        "name": row.get("name"),
                        "setting": ORACLE_NAME,
                        "model": opt.model,
                        "backend": opt.backend,
                        "speccoder_finetune": bool(opt.speccoder_finetune),
                        "source": "humanexec.correct_solution",
                        "prompt": prompt,
                        "response": response,
                        "clean_code": clean_by_tid[tid],
                        "annotated_code": extract_code(response),
                        "test_cases": row.get("test_cases"),
                        "time_latency": {
                            "generation_seconds": round(batch_seconds, 6),
                            "batch_seconds": round(batch_seconds, 6),
                            "global_batch": len(chunk),
                            "local_batch": num_samples,
                        },
                        "cost_token": sample_meta.get("cost_token") or token_usage_record(prompt, response),
                    }
                    append_jsonl(generation_path, rec)
        write_token_files(backend, out_dir, allowed_task_ids=set(row_by_tid))

    latest_generation = load_attempts_by_task(generation_path)
    if opt.eval:
        from SpecCoder.src.experiment.specgen.compute_spec_quality import load_mutants_by_task
        from SpecCoder.src.experiment.specgen.inline.quality import (
            evaluate_task_attempts,
            existing_result_is_complete,
            normalize_result_jsonl_for_summary,
        )

        existing_results = load_latest_jsonl_by_task(result_path)
        pending_tids: list[str] = []
        for tid in row_by_tid:
            existing = existing_results.get(tid)
            if existing_result_is_complete(existing, opt):
                continue
            attempts = [
                latest_generation[tid][i]
                for i in sorted(latest_generation.get(tid, {}))
                if i < opt.num_samples
            ]
            if len(attempts) >= opt.num_samples:
                pending_tids.append(tid)
        mutants_by_task = {}
        if pending_tids:
            if not os.path.exists(opt.mutants):
                raise SystemExit(f"Mutant file not found for spec-quality eval: {opt.mutants}")
            mutants_by_task = load_mutants_by_task(opt.mutants)
        evaluated_count = 0
        total_tasks = len(row_by_tid)
        for task_pos, (tid, row) in enumerate(row_by_tid.items(), start=1):
            existing = existing_results.get(tid)
            if existing_result_is_complete(existing, opt):
                print(f"[{task_pos:>4}/{total_tasks}] eval {tid:<14} cached, skip", flush=True)
                continue
            attempts = [
                latest_generation[tid][i]
                for i in sorted(latest_generation.get(tid, {}))
                if i < opt.num_samples
            ]
            if len(attempts) < opt.num_samples:
                print(f"[{task_pos:>4}/{total_tasks}] eval {tid:<14} "
                      f"skip ({len(attempts)}/{opt.num_samples} attempts)", flush=True)
                continue
            print(f"[{task_pos:>4}/{total_tasks}] eval {tid:<14} "
                  f"({opt.num_samples} attempts) ...", flush=True)
            task_record = {
                "task_id": tid,
                "contestId": row.get("contestId"),
                "index": row.get("index"),
                "name": row.get("name"),
                "setting": ORACLE_NAME,
                "model": opt.model,
                "backend": opt.backend,
                "speccoder_finetune": bool(opt.speccoder_finetune),
                "source": "humanexec.correct_solution",
                "num_attempts": opt.num_samples,
                "prompt": prompt_by_tid.get(tid) or attempts[0].get("prompt", ""),
                "test_cases": row.get("test_cases") or attempts[0].get("test_cases"),
            }
            evaluated = evaluate_task_attempts(task_record, attempts, mutants_by_task, opt)
            append_jsonl(result_path, evaluated)
            append_task_log(
                os.path.join(log_dir, f"{task_log_name(tid)}.log"),
                {
                    **evaluated,
                    "response": [a.get("response", "") for a in attempts],
                    "code": [a.get("annotated_code", "") for a in attempts],
                },
                header=f"setting={ORACLE_NAME}  model={opt.model}  attempts={opt.num_samples}",
            )
            evaluated_count += 1

        result_records = normalize_result_jsonl_for_summary(
            result_path,
            list(row_by_tid),
            drop_unevaluable=getattr(opt, "drop_unevaluable_assertions", True),
        )
        write_oracle_quality(result_records, opt, out_dir)
        print(f"Evaluated this run: {evaluated_count}")

    print(f"Generated this run: {missing_count} missing attempts requested | tasks available: {len(row_by_tid)}")
    print(f"Generation file -> {generation_path}")
    if opt.eval:
        print(f"Result file     -> {result_path}")
        print(f"Task logs       -> {log_dir}")


def parse_oracle_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inline spec-generation runner: oracle mode over HumanExec correct_solution."
    )
    add_backend_args(parser)
    parser.add_argument("--humanexec", default=DEFAULT_HUMANEXEC)
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory. Default is output/spec_gen/inline/<model>/oracle/<base_model|speccoder>.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-tasks", "--num_tasks", type=int, default=None,
                        help="Use only the first N tasks for generation/evaluation/quality summary.")
    parser.add_argument("--local-batch", "--local_batch", "--n-samples", "--n_samples",
                        dest="num_samples", type=int, default=1,
                        help="Number of spec attempts per task.")
    parser.add_argument("--global-batch", "--global_batch", "--task-batch-size",
                        dest="global_batch", type=int, default=0,
                        help="Number of tasks per backend.chat call; 0 means all pending tasks.")
    parser.add_argument("--mutants", default=DEFAULT_MUTANTS,
                        help="Validated mutants for spec-quality evaluation.")
    parser.add_argument("--quality-timeout", type=int, default=10)
    parser.add_argument("--quality-test-workers", "--quality_test_workers", type=int, default=32,
                        help="Number of test cases to execute in parallel during spec-quality evaluation.")
    parser.add_argument("--drop-unevaluable-assertions", "--drop_unevaluable_assertions",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Omit assertions that reach neither correctness (reach_count=0) nor "
                             "completeness (no accepted/rejected/reachable mutant) from sample/summary "
                             "averages -- they are redundant specs, not vacuous 1.0s. On by default; "
                             "pass --no-drop-unevaluable-assertions to keep them.")
    parser.add_argument("--no-eval", action="store_true",
                        help="Generate only; skip spec-quality evaluation.")
    parser.add_argument("--eval", action="store_true",
                        help="(deprecated: evaluation runs by default; use --no-eval to skip).")
    parser.add_argument("--eval-only", "--eval_only", action="store_true",
                        help="Skip generation; evaluate existing generation.jsonl only.")
    return parser.parse_args(argv)


def main() -> None:
    argv = _strip_mode_from_argv(sys.argv[1:])
    if argv and argv[0] == "oracle":
        argv = argv[1:]
    run_oracle(parse_oracle_args(argv))


if __name__ == "__main__":
    main()
