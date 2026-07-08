
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from SpecCoder.src.experiment.programrepair import common

NAME = "hoareprompt"

# HoarePrompt uses flat imports (``from model import ...``); put its src on path.
_HP_SRC = common.PROJECT_ROOT / "SpecCoder" / "others" / "HoarePrompt" / "src"


def _ensure_hp_on_path() -> None:
    if str(_HP_SRC) not in sys.path:
        sys.path.insert(0, str(_HP_SRC))


def improve(sample, buggy, candidate, backend, opt, state) -> tuple[str, dict[str, Any]]:
    return improve_batch([sample], [buggy], [candidate], backend, opt, [state])[0]


def improve_batch(samples, buggys, candidates, backend, opt, states) -> list[tuple[str, dict[str, Any]]]:
    """HoarePrompt barriers: annotate the whole batch, then repair the whole batch."""
    _ensure_hp_on_path()
    import model as hp_model  # noqa: E402
    import repair_entry  # noqa: E402

    hp_model.set_active_backend(backend)
    config = dict(repair_entry.DEFAULT_CONFIG)
    config["temperature"] = opt.temperature
    n = len(samples)
    t0 = time.perf_counter()
    descriptions = [common.problem_text(sample) for sample in samples]
    annotated = [""] * n
    postconditions = [""] * n
    errors: list[str | None] = [None] * n

    # Milestone 1: every task completes annotation before any final repair starts.
    # Each task's annotation is a sequential chain of ~16 internal LLM calls, but tasks
    # are INDEPENDENT (own tempdir, own state), so we run the WHOLE batch concurrently:
    # one annotation chain per task in this batch (``workers = n``, i.e. it follows the
    # active batch — the --global-batch chunk for purellm, or the live subset for
    # agentcoder/chatrepair — no separate knob). The concurrency itself is what lets the
    # backend batch: parallel workers' in-flight calls are coalesced into one dispatch
    # (vLLM → a single batched generate on the lone dispatcher thread; API → one
    # fanned-out _chat_pool). begin_task is thread-local, so token usage stays attributed
    # per task. A single-task batch (n == 1) takes the old sequential path byte-for-byte.
    workers = n
    common.progress("hoareprompt", f"annotation milestone on {n} task(s)"
                    + (f" ({workers} parallel)" if workers > 1 else ""))
    _abar = common.step_bar(n, "hoareprompt: annotation")

    def _annotate_one(i: int) -> None:
        if hasattr(backend, "begin_task"):
            backend.begin_task(common.task_id(samples[i]))
        with tempfile.TemporaryDirectory(prefix="hoareprompt_") as tmp:
            try:
                annotated[i], postconditions[i] = repair_entry.annotate_program(
                    descriptions[i], candidates[i], config, Path(tmp))
            except Exception as exc:
                errors[i] = str(exc)
                postconditions[i] = f"<error: {exc}>"
        common.live_log(common.task_id(samples[i]), "annotate", response=annotated[i])
        common.stage_done(common.task_id(samples[i]), "MILESTONE: hoareprompt annotation",
                          info=("annotated" if annotated[i]
                                else f"annotate failed -> repair clean (no annotation): {errors[i]}"))
        if _abar is not None:
            _abar.update(1)

    with common.suppress_dispatch_log():
        if n > 1 and hasattr(backend, "enable_coalescing"):
            backend.enable_coalescing(max_batch=workers)
            try:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futs = [ex.submit(_annotate_one, i) for i in range(n)]
                    for fut in as_completed(futs):
                        fut.result()
            finally:
                backend.disable_coalescing()
        else:
            for i in range(n):
                _annotate_one(i)
    if _abar is not None:
        _abar.close()

    valid = list(range(n))
    final_msgs: list[list[dict[str, str]] | None] = [None] * n
    repair_responses = [""] * n
    if valid:
        common.progress("hoareprompt", f"repair milestone on {len(valid)} task(s)")
        repair_batches = []
        for i in valid:
            if annotated[i]:
                code = repair_entry.hp.remove_functionality(annotated[i])
            else:
                code = ("(Annotation unavailable for this program — repair it directly "
                        "from the program above.)")
            prompt = repair_entry.REPAIR_PROMPT.format(
                description=descriptions[i], code=code, buggy_code=candidates[i])
            base = [{"role": "user", "content": prompt}]
            final_msgs[i] = common.repair_conversation_messages(base, states[i])
            repair_batches.append(final_msgs[i])
        comps = common.chat_batched(
            backend, repair_batches, opt, num_samples=1,
            task_ids=[common.task_id(samples[i]) for i in valid])
        for slot, i in enumerate(valid):
            response = comps[slot][0] if comps[slot] else ""
            repair_responses[i] = response
            common.remember_repair_exchange(states[i], final_msgs[i], response)
            common.live_log(
                common.task_id(samples[i]), "hp_repair",
                prompt=common.format_chat_messages(final_msgs[i]), response=response)

    elapsed = (time.perf_counter() - t0) / max(1, n)
    out: list[tuple[str, dict[str, Any]]] = []
    for i in range(n):
        response = repair_responses[i]
        new_code = common.extract_code(response) or candidates[i]
        phases = [
            {
                "name": "annotate",
                "prompt": f"# Problem\n{descriptions[i]}\n\n# Program\n{candidates[i]}",
                "response": annotated[i] if annotated[i] else f"<error: {errors[i]}>",
            },
            {
                "name": "hp_repair",
                "prompt": common.format_chat_messages(final_msgs[i]) if final_msgs[i] else "",
                "response": response,
            },
        ]
        prompt_text = common.format_chat_messages(final_msgs[i]) if final_msgs[i] else descriptions[i]
        info = {
            "changed": new_code != candidates[i],
            "feedback": annotated[i],
            "hoareprompt_annotated": annotated[i],
            "hoareprompt_postcondition": postconditions[i],
            "phases": phases,
            "cost": common.attempt_cost(prompt_text, response),
            "seconds": elapsed,
        }
        if errors[i]:
            info["annotation_error"] = errors[i]
        out.append((new_code, info))
    return out


def run_method(rows: list[dict[str, Any]], backend, opt) -> list[dict[str, Any]]:
    _ensure_hp_on_path()
    import model as hp_model  # noqa: E402  (others/HoarePrompt/src/model.py)
    import repair_entry  # noqa: E402

    hp_model.set_active_backend(backend)
    config = dict(repair_entry.DEFAULT_CONFIG)
    config["temperature"] = opt.temperature
    num_samples = max(1, int(getattr(opt, "num_samples", 1) or 1))

    records: list[dict[str, Any]] = []
    for sample in rows:
        buggy_code = common.buggy_code_of(sample)
        description = common.problem_text(sample)
        prompt_text = f"{description}\n\n### Buggy Code\n```python\n{buggy_code}\n```"
        for attempt in range(num_samples):
            started_at = time.perf_counter()
            with tempfile.TemporaryDirectory(prefix="hoareprompt_") as tmp:
                try:
                    response, annotated, postcondition = repair_entry.hoareprompt_repair(
                        description, buggy_code, config, Path(tmp)
                    )
                except Exception as exc:  # annotation can fail on unparsable programs
                    response, annotated, postcondition = "", "", f"<error: {exc}>"
            generation_seconds = time.perf_counter() - started_at
            rec = common.base_record(sample, opt, buggy_code, NAME)
            rec.update({
                "attempt": attempt,
                "code": common.extract_code(response) or buggy_code,
                "rounds": 1,
                "hoareprompt_annotated": annotated,
                "hoareprompt_postcondition": postcondition,
                "prompt": prompt_text,
                "response": response,
                "time_latency": {
                    "generation_seconds": round(generation_seconds, 6),
                    "global_batch": 1,
                    "local_batch": 1,
                },
                "cost_token": common.attempt_cost(prompt_text, response),
            })
            records.append(rec)
    return records
