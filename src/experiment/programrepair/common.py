
from __future__ import annotations

import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Any

_THIS = Path(__file__).resolve()
PROJECT_ROOT = next((p for p in _THIS.parents if (p / "SpecCoder").is_dir()), _THIS.parents[5])
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from SpecCoder.spec_utils import build_problem_text  # noqa: E402
from SpecCoder.src.codeforces_eval import calculate_pass_rate  # noqa: E402
from SpecCoder.src.experiment.gen_utils import extract_code, sum_token_usage, token_usage_record  # noqa: E402

DEFAULT_INPUT = str(PROJECT_ROOT / "SpecCoder" / "data" / "humanexec.jsonl")

SYSTEM_PROMPT = (
    "You are an automated program repair model for Python competitive-programming "
    "submissions. Return exactly one <solution> tag containing one fenced Python code "
    "block. No extra text is allowed outside that tag."
)


def load_jsonl(path: str, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def task_id(sample: dict[str, Any]) -> str:
    return sample.get("task_id") or f"cf_{sample.get('contestId')}{sample.get('index')}"


def parse_task_ids(raw) -> list[str]:
    """Normalize a ``--task-ids`` value into a list of ids. Accepts a list of tokens
    (argparse ``nargs="+"``) and/or comma-separated strings, so all of these work::

        --task-ids cf_141A cf_4A
        --task-ids cf_141A,cf_4A
        --task-ids "cf_141A, cf_4A"
    """
    if not raw:
        return []
    tokens = raw if isinstance(raw, (list, tuple)) else [raw]
    out: list[str] = []
    for tok in tokens:
        out.extend(piece.strip() for piece in str(tok).split(",") if piece.strip())
    return out


def select_task_ids(rows: list[dict[str, Any]], task_ids) -> list[dict[str, Any]]:
    wanted = parse_task_ids(task_ids)
    if not wanted:
        return rows
    by_id = {task_id(r): r for r in rows}
    missing = [tid for tid in wanted if tid not in by_id]
    if missing:
        raise SystemExit(
            f"--task-ids: {len(missing)} id(s) not found in the dataset: {', '.join(missing)}")
    seen: set[str] = set()
    selected = []
    for tid in wanted:                       # de-dupe while preserving the requested order
        if tid not in seen:
            seen.add(tid)
            selected.append(by_id[tid])
    return selected


def buggy_code_of(sample: dict[str, Any]) -> str:
    return sample.get("buggy_solution") or sample.get("code", "") or ""


def format_examples(sample: dict[str, Any], max_examples: int = 3) -> str:
    demo_inputs = sample.get("demo-input") or []
    demo_outputs = sample.get("demo-output") or []
    parts = []
    for i, (inp, out) in enumerate(zip(demo_inputs, demo_outputs)):
        if i >= max_examples:
            break
        parts.append(f"Example {i + 1} input:\n{inp}\nExample {i + 1} output:\n{out}")
    if parts:
        return "\n\n".join(parts)
    for i, case in enumerate((sample.get("test_cases") or [])[:max_examples]):
        parts.append(
            f"Example {i + 1} input:\n{case.get('input', '')}\n"
            f"Example {i + 1} output:\n{case.get('output', '')}"
        )
    return "\n\n".join(parts)


def problem_text(sample: dict[str, Any], max_examples: int = 3) -> str:
    """Problem statement + worked examples + optional note for the model."""
    parts = [build_problem_text(sample)]
    examples = format_examples(sample, max_examples)
    if examples:
        parts.append(f"### Examples\n{examples}")
    note = str(sample.get("note") or "").strip()
    if note and note.lower() != "none":
        parts.append(f"### Note\n{note}")
    return "\n\n".join(p for p in parts if p.strip())


def build_repair_prompt(sample: dict[str, Any], buggy_code: str, feedback: str = "") -> str:
    feedback_section = f"\n### Feedback\n{feedback.strip()}\n" if feedback.strip() else ""
    return f"""Task: repair the buggy Python program so it satisfies the problem specification and passes the tests.
Use only the provided specification, examples, and buggy code. Preserve the required stdin/stdout interface.
Prefer a small patch, but produce a complete correct program if a rewrite is necessary.

### Problem
{problem_text(sample)}

### Buggy Code
```python
{buggy_code}
```
{feedback_section}
### Required Output
Your entire response must be exactly one <solution> block, with one fenced Python block inside.

<solution>
```python
# repaired code
```
</solution>
"""


def repair_messages(sample: dict[str, Any], buggy_code: str, feedback: str = "") -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_repair_prompt(sample, buggy_code, feedback)},
    ]


def repair_conversation_messages(
    messages: list[dict[str, str]],
    state: dict[str, Any] | None,
) -> list[dict[str, str]]:
    current = [dict(message) for message in messages]
    if not state or not state.get("_repair_conversation_enabled"):
        return current
    history = [dict(message) for message in (state.get("_repair_history") or [])]
    if not history:
        return current
    return history + [message for message in current if message.get("role") != "system"]


def remember_repair_exchange(
    state: dict[str, Any] | None,
    messages: list[dict[str, str]],
    response: str,
) -> None:
    if not state or not state.get("_repair_conversation_enabled"):
        return
    state["_repair_history"] = [
        *[dict(message) for message in messages],
        {"role": "assistant", "content": response or ""},
    ]


def format_chat_messages(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"[{str(message.get('role', '')).upper()}]\n{message.get('content', '')}"
        for message in messages
    )


def chat_batched(backend, messages: list[list[dict[str, str]]], opt, num_samples: int = 1, task_ids=None,
                 use_lora: bool = True) -> list[list[str]]:
    completions, _metadata = timed_chat_batched(
        backend, messages, opt, num_samples=num_samples, task_ids=task_ids, use_lora=use_lora)
    return completions


def timed_chat_batched(
    backend,
    messages: list[list[dict[str, str]]],
    opt,
    num_samples: int = 1,
    task_ids=None,
    use_lora: bool = True,
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    batch_size = int(getattr(opt, "global_batch", 0) or getattr(opt, "task_batch_size", 0) or 0)
    if batch_size <= 0:
        chunks = [messages]
        tid_chunks = [task_ids]
    else:
        chunks = [messages[start : start + batch_size] for start in range(0, len(messages), batch_size)]
        tid_chunks = [task_ids[start : start + batch_size] if task_ids else None
                      for start in range(0, len(messages), batch_size)]
    out: list[list[str]] = []
    metadata: list[dict[str, Any]] = []
    for chunk, tids in zip(chunks, tid_chunks):
        started_at = time.perf_counter()
        chat_with_metadata = getattr(backend, "chat_with_metadata", None)
        if callable(chat_with_metadata):
            completions, completion_metadata = chat_with_metadata(
                chunk, num_samples=num_samples, task_ids=tids, use_lora=use_lora)
        else:
            completions = backend.chat(chunk, num_samples=num_samples, task_ids=tids, use_lora=use_lora)
            completion_metadata = [[{} for _ in list(samples or [])] for samples in completions]
        batch_seconds = time.perf_counter() - started_at
        out.extend(completions)
        for samples_meta in completion_metadata:
            metadata.append({
                "generation_seconds": round(batch_seconds, 6),
                "batch_seconds": round(batch_seconds, 6),
                "global_batch": len(chunk),
                "local_batch": num_samples,
                "samples": list(samples_meta or []),
            })
    return out, metadata


def attempt_cost(prompt: str, response: str, metadata: dict[str, Any] | None = None) -> dict[str, int]:
    if metadata and metadata.get("cost_token"):
        return metadata["cost_token"]
    return token_usage_record(prompt, response)


def total_attempt_latency(attempts: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "generation_seconds": round(sum(
            float((attempt.get("time_latency") or {}).get("generation_seconds", 0.0) or 0.0)
            for attempt in attempts
        ), 6),
    }


def total_attempt_cost(attempts: list[dict[str, Any]]) -> dict[str, int]:
    return sum_token_usage([attempt.get("cost_token") or {} for attempt in attempts])


# RLIMIT_AS via preexec_fn is rejected on macOS; keep the 512MB cap on Linux.
_EVAL_MEMORY_MB = None if sys.platform == "darwin" else 512


def evaluate(code: str, sample: dict[str, Any], timeout: float, test_workers: int = 16) -> dict[str, Any]:
    """Pass-rate of ``code`` on the real ``test_cases`` (the final oracle)."""
    return calculate_pass_rate(
        code, sample.get("test_cases") or [],
        timeout=timeout,
        memory_mb=_EVAL_MEMORY_MB,
        skip_truncated_outputs=True,
        test_workers=test_workers,
    )


def format_failures(result: dict[str, Any], cases: list[dict[str, str]], max_failures: int = 2) -> str:
    parts = [
        f"The program passed {result.get('passed_tests', 0)}/{result.get('total_tests', 0)} tests."
    ]
    shown = 0
    for failure in result.get("failures") or []:
        if shown >= max_failures:
            break
        idx = failure.get("index")
        case = cases[idx] if isinstance(idx, int) and idx < len(cases) else {}
        detail: Any = failure.get("detail", "")
        if isinstance(detail, str):
            try:
                detail = json.loads(detail)
            except json.JSONDecodeError:
                detail = {"message": detail}
        actual = detail.get("stdout", "") if isinstance(detail, dict) else ""
        rc = detail.get("returncode") if isinstance(detail, dict) else None
        err = "TIMEOUT" if detail == "timeout" or (isinstance(detail, dict) and detail.get("message") == "timeout") else (
            (detail.get("stderr") or f"RUNTIME_ERROR(returncode={rc})") if isinstance(detail, dict) and rc not in (None, 0, "0") else "WRONG_OUTPUT"
        )
        parts += [
            f"- Failing test {idx}:",
            f"  input: {str(case.get('input', '')).strip()[:300]}",
            f"  expected: {str(case.get('output', '')).strip()[:300]}",
            f"  actual: {str(actual).strip()[:300]}",
            f"  error: {err}",
        ]
        shown += 1
    return "\n".join(parts)


def render_transcript(rounds: list[dict[str, Any]]) -> str:
    out: list[str] = []
    for r in rounds:
        out.append(f"--- round {r.get('round')} | ok={r.get('ok')} ---")
        if r.get("feedback"):
            out.append(f"[FEEDBACK]\n{r['feedback']}")
        if r.get("response"):
            out.append(f"[REPAIR RESPONSE]\n{r['response']}")
    return "\n".join(out)



def base_record(sample: dict[str, Any], opt, buggy_code: str, method: str) -> dict[str, Any]:
    record = {
        "task_id": task_id(sample),
        "contestId": sample.get("contestId"),
        "index": sample.get("index"),
        "name": sample.get("name"),
        "method": method,
        "model": opt.model,
        "backend": opt.backend,
        "speccoder_finetune": bool(getattr(opt, "speccoder_finetune", False)),
        "buggy_code": buggy_code,
    }
    if getattr(opt, "include_test_cases", False):
        record["test_cases"] = sample.get("test_cases")
    return record


_STEP = {"label": ""}


def progress(tag: str, msg: str) -> None:
    _STEP["label"] = f"{tag}: {msg}" if tag else str(msg)
    print(f"    [{tag}] {msg}", flush=True)


def current_step() -> str:
    return _STEP["label"]


def test_batch_of(opt) -> int:
    tb = getattr(opt, "test_batch", None)
    if tb:
        return max(1, int(tb))
    return max(1, int(getattr(opt, "eval_workers", 16) or 16))


def step_bar(total: int, desc: str | None = None):
    if total is None or total <= 1:
        return None
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    label = (desc or current_step() or "step").strip()
    if len(label) > 60:
        label = label[:57] + "..."
    return tqdm(total=total, desc=label, leave=False, dynamic_ncols=True, mininterval=0.3)


def _last_user_content(messages) -> str:
    if isinstance(messages, list) and messages:
        last = messages[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
        return str(last)
    return str(messages)


def log_call_prompt(task_identifier, messages, step: str | None = None) -> None:
    if not _LIVE_LOG["on"] or _SUPPRESS_DISPATCH_LOG["on"] or task_identifier is None:
        return
    live_log(task_identifier, f"{step or current_step()}  [PROMPT @ dispatch]",
             prompt=_last_user_content(messages))


_SUPPRESS_DISPATCH_LOG = {"on": False}


@contextlib.contextmanager
def suppress_dispatch_log():
    prev = _SUPPRESS_DISPATCH_LOG["on"]
    _SUPPRESS_DISPATCH_LOG["on"] = True
    try:
        yield
    finally:
        _SUPPRESS_DISPATCH_LOG["on"] = prev

import os as _os
import re as _re
import threading as _threading

_LIVE_LOG = {"dir": None, "on": False}
_LIVE_LOCK = _threading.Lock()


def configure_live_log(log_dir: str | None, enabled: bool = True) -> None:
    _LIVE_LOG["dir"] = log_dir
    _LIVE_LOG["on"] = bool(enabled and log_dir)
    if _LIVE_LOG["on"]:
        _os.makedirs(log_dir, exist_ok=True)


def _live_log_path(task_identifier: str) -> str:
    safe = _re.sub(r"[^A-Za-z0-9._-]+", "_", str(task_identifier)).strip("_") or "task"
    return _os.path.join(_LIVE_LOG["dir"], f"{safe}.log")


def live_log(task_identifier, name: str, prompt=None, response=None, result=None, it=None,
             header: str | None = None) -> None:
    if not _LIVE_LOG["on"]:
        return
    parts: list[str] = []
    if header is not None:
        parts += ["", "=" * 72, header, "=" * 72]
    if name:
        parts.append(f"========== {name}" + (f" (iter {it})" if it is not None else "") + " ==========")
    if prompt is not None:
        parts += ["[PROMPT]", str(prompt), ""]
    if response is not None:
        parts += ["[RESPONSE]", str(response), ""]
    if result is not None:
        parts += ["[RESULT]", str(result), ""]
    if not parts:
        return
    try:
        with _LIVE_LOCK, open(_live_log_path(task_identifier), "a", encoding="utf-8") as fh:
            fh.write("\n".join(parts) + "\n")
    except Exception:
        pass  # logging must never break a run


def init_task_logs(task_identifiers, header: str | None = None) -> None:
    if not _LIVE_LOG["on"]:
        return
    for tid in task_identifiers:
        if header is not None:
            live_log(tid, "", header=header)
        else:
            try:  # just ensure the file exists
                with _LIVE_LOCK, open(_live_log_path(tid), "a", encoding="utf-8"):
                    pass
            except Exception:
                pass


def tqdm_print(line: str) -> None:
    try:
        from tqdm.auto import tqdm
        tqdm.write(line)
    except Exception:
        print(line, flush=True)


def stage_done(task_identifier, stage: str, info=None, *, echo: bool = True) -> None:
    if echo:
        tqdm_line = f"      ✓ {task_identifier}  {stage}"
        if info is not None:
            tqdm_line += f"  |  {info}"
        tqdm_print(tqdm_line)
    if not _LIVE_LOG["on"]:
        return
    live_log(task_identifier, f"{stage}  [STAGE DONE]",
             result="" if info is None else str(info))


def emit(task_identifier, sink: list, entry: dict) -> None:
    sink.append(entry)
    if entry.get("response") is None and entry.get("result") is None:
        return
    live_log(task_identifier, entry.get("name", ""), prompt=None,
             response=entry.get("response"), result=entry.get("result"), it=entry.get("iter"))


__all__ = [
    "DEFAULT_INPUT", "SYSTEM_PROMPT", "load_jsonl", "task_id", "buggy_code_of",
    "format_examples", "problem_text", "build_repair_prompt", "repair_messages",
    "repair_conversation_messages", "remember_repair_exchange", "format_chat_messages",
    "evaluate", "base_record", "extract_code", "timed_chat_batched", "progress",
    "current_step", "step_bar", "log_call_prompt", "suppress_dispatch_log", "test_batch_of",
    "configure_live_log", "live_log", "emit", "init_task_logs", "stage_done", "tqdm_print",
    "attempt_cost", "total_attempt_latency", "total_attempt_cost",
]
