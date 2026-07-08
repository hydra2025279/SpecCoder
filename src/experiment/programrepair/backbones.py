
from __future__ import annotations

from typing import Any, Callable, Optional

from SpecCoder.src.experiment.programrepair import common
from SpecCoder.src.experiment.programrepair import localize as _localize

ImproveBatchFn = Callable[..., list[tuple[str, dict[str, Any]]]]


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #

def _expand(rows: list[dict[str, Any]], opt) -> tuple[list[dict[str, Any]], list[int]]:
    """Expand each row into its per-attempt copies, SKIPPING attempt indices that
    already exist in generation.jsonl.

    ``opt.existing_attempts_by_task`` maps task_id -> set of already-generated
    attempt indices; only the missing ones are regenerated. When unset (e.g. a
    fresh run), every attempt 0..num_samples-1 is produced."""
    samples_per_task = max(1, int(getattr(opt, "num_samples", 1) or 1))
    existing = getattr(opt, "existing_attempts_by_task", None) or {}
    expanded: list[dict[str, Any]] = []
    attempts: list[int] = []
    for s in rows:
        done = existing.get(common.task_id(s)) or ()
        for a in range(samples_per_task):
            if a in done:
                continue
            expanded.append(s)
            attempts.append(a)
    return expanded, attempts


def _purellm_patch0_messages(opt, sample, buggy) -> list[dict[str, str]]:
    """Bare repair prompt, except ``cot`` injects its reasoning instruction."""
    if getattr(opt, "method", "") == "cot":
        from SpecCoder.src.experiment.programrepair import cot
        return [
            {"role": "system", "content": common.SYSTEM_PROMPT},
            {"role": "user", "content": cot.build_prompt(sample, buggy)},
        ]
    return common.repair_messages(sample, buggy, "")


_SIGNAL_INFO_KEYS = (
    "save_skip",
    "num_violated", "num_violated_specs",
    "reach_satisfy",
    "spec_annotated", "reused_saved_spec",
    "clean_annotated", "enhance_annotated",
    "reproducer_test", "spec", "review", "approved", "raw_review",
    "hoareprompt_annotated", "hoareprompt_postcondition",
    "designer_tests", "rex_steps", "rex_best_pass_rate",
    "localize_prompt",
    "localize_clean_code",
)


def _render_transcript(trace: list[dict[str, Any]], final_code: str) -> str:
    """Render the full per-attempt trace into a readable transcript.

    Each entry is one logged step (an LLM call or a grading checkpoint). We log
    whatever the entry carries: ``[PROMPT]`` and ``[RESPONSE]`` for every LLM call
    (so no call is ever silent), and ``[TEST RESULT]`` for a grading step. The goal
    is full prompt+response coverage for every call across every turn/iteration."""
    out: list[str] = []
    for e in trace:
        tag = e.get("name", "")
        it = e.get("iter")
        head = f"========== {tag}" + (f" (iter {it})" if it is not None else "") + " =========="
        out.append(head)
        if "prompt" in e:
            out += ["[PROMPT]", str(e.get("prompt", "")), ""]
        if "response" in e:
            out += ["[RESPONSE]", str(e.get("response", "")), ""]
        if "result" in e:
            out += ["[TEST RESULT]", str(e.get("result", "")), ""]
    out += ["========== FINAL CODE ==========", final_code]
    return "\n".join(out)


def _build_records(
    expanded, attempts, buggy, patch0, final, infos, rounds, costs, secs, opt, traces,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for i in range(len(expanded)):
        info = infos[i] or {}
        rec = common.base_record(expanded[i], opt, buggy[i], opt.method)
        feedback = info.get("feedback", "") or ""
        transcript = _render_transcript(traces[i], final[i])
        rec.update({
            "attempt": attempts[i],
            "code": final[i],
            "rounds": rounds[i],
            "patch_0": patch0[i],
            "backbone_framework": getattr(opt, "backbone_framework", "purellm"),
            "prompt": common.build_repair_prompt(expanded[i], buggy[i], feedback),
            "response": transcript,
            "time_latency": {
                "generation_seconds": round(secs[i], 6),
                "global_batch": int(getattr(opt, "global_batch", 0) or getattr(opt, "task_batch_size", 0) or 0),
                "local_batch": 1,
            },
            "cost_token": common.sum_token_usage(costs[i]),
        })
        for key in _SIGNAL_INFO_KEYS:
            if key in info:
                rec[key] = info[key]
        records.append(rec)
    return records


# --------------------------------------------------------------------------- #
# purellm backbone
# --------------------------------------------------------------------------- #

def run_purellm(rows, backend, opt, improve_batch_fn: Optional[ImproveBatchFn]) -> list[dict[str, Any]]:
    expanded, attempts = _expand(rows, opt)
    n = len(expanded)
    buggy = [common.buggy_code_of(s) for s in expanded]

    costs: list[list[dict[str, int]]] = [[] for _ in range(n)]
    secs = [0.0] * n
    traces: list[list[dict[str, Any]]] = [[] for _ in range(n)]
    infos: list[dict[str, Any]] = [{} for _ in range(n)]
    rounds = [0] * n

    if improve_batch_fn is None:
        # naive / cot: there is no signal, so the patch_0 generation IS the answer.
        common.progress("purellm", f"patch_0 generation on {n} task(s) via {opt.method}")
        msgs0 = [_purellm_patch0_messages(opt, expanded[i], buggy[i]) for i in range(n)]
        comps0, meta0 = common.timed_chat_batched(backend, msgs0, opt, num_samples=1,
                                                  task_ids=[common.task_id(expanded[i]) for i in range(n)])
        patch0: list[str] = []
        for i in range(n):
            resp = comps0[i][0] if comps0[i] else ""
            patch0.append(common.extract_code(resp) or buggy[i])
            costs[i].append(common.attempt_cost(msgs0[i][-1]["content"], resp))
            secs[i] += float(meta0[i].get("generation_seconds", 0.0) or 0.0)
            common.emit(common.task_id(expanded[i]), traces[i], {"name": "patch_0", "prompt": msgs0[i][-1]["content"], "response": resp})
        final = list(patch0)
    else:
        patch0 = list(buggy)
        final = list(buggy)
        states = [{} for _ in range(n)]
        results = improve_batch_fn(expanded, buggy, buggy, backend, opt, states)
        for i in range(n):
            new_code, info = results[i]
            final[i] = new_code
            infos[i] = info
            if info.get("cost"):
                costs[i].append(info["cost"])
            secs[i] += float(info.get("seconds", 0.0) or 0.0)
            rounds[i] = 1 if info.get("changed") else 0
            traces[i].extend(info.get("phases", []))

    _purellm_localize(expanded, patch0, infos, traces, costs, backend, opt)

    return _build_records(expanded, attempts, buggy, patch0, final, infos, rounds, costs, secs, opt, traces)


def _purellm_localize(expanded, patch0, infos, traces, costs, backend, opt) -> None:
    if not getattr(opt, "localize", True) or not getattr(opt, "localize_path", None):
        return
    import os

    from SpecCoder.src.experiment.backend import append_token_records

    # --skip-localize: BUILD the prompt (cheap, no LLM) and save it, but skip the call.
    skip_call = bool(getattr(opt, "skip_localize_call", False))

    n = len(expanded)
    signals = []
    for i in range(n):
        info = infos[i] or {}
        sig = info.get("localize_signal")
        signals.append(sig if sig is not None else {"text": info.get("feedback", "") or ""})

    loc_recs = _localize.localize_batch(expanded, patch0, signals, backend, opt,
                                        make_call=not skip_call)
    by_tid = {r["task_id"]: r for r in loc_recs}
    prompt_path = getattr(opt, "localize_prompt_path", None)
    if prompt_path:
        prompt_recs = [
            {"task_id": r["task_id"], "attempt": 0,
             "localize_prompt": r.get("final_localize_prompt"),
             "localize_clean_code": r.get("clean_code")}
            for r in loc_recs if r.get("final_localize_prompt")
        ]
        _localize.upsert_localize_prompts(prompt_path, prompt_recs)
    loc_log_dir = os.path.join(os.path.dirname(opt.localize_path), "log")
    for i in range(n):
        rec = by_tid.get(common.task_id(expanded[i]))
        if rec is None:
            continue
        infos[i]["localize_prompt"] = rec.get("final_localize_prompt")
        infos[i]["localize_clean_code"] = rec.get("clean_code")
        if skip_call:
            continue
        common.emit(common.task_id(expanded[i]), traces[i], {
            "name": "LOCALIZE",
            "prompt": rec["prompt"],
            "response": rec["response"],
            "result": f"predicted buggy lines (least->most certain): {rec['prediction']}",
        })
        _write_localize_log(loc_log_dir, rec, opt)

    if skip_call:
        return
    append_token_records(os.path.dirname(opt.localize_path), loc_recs, with_attempt=False)


def purellm_localize_replay(rows, existing_generation, backend, opt) -> int:
    import os

    from SpecCoder.src.experiment.backend import append_token_records

    path = getattr(opt, "localize_path", None)
    if not path:
        return 0

    saved_prompts = _localize.load_localize_prompts(getattr(opt, "localize_prompt_path", None))

    jobs: list[tuple[dict[str, Any], list[dict[str, str]], str]] = []
    for s in rows:
        tid = common.task_id(s)
        src = saved_prompts.get(tid) or (existing_generation.get(tid) or {}).get(0) or {}
        msgs = src.get("localize_prompt")
        if not msgs:
            continue
        a0 = (existing_generation.get(tid) or {}).get(0) or {}
        clean = (src.get("localize_clean_code") or a0.get("localize_clean_code")
                 or a0.get("patch_0") or common.buggy_code_of(s))
        jobs.append((s, msgs, clean))

    if not jobs:
        print("[localize-only] no saved localize prompts found "
              "(run repair first to populate localize_prompt.jsonl).")
        return 0

    recs = _localize.localize_from_saved(
        [j[0] for j in jobs], [j[1] for j in jobs], [j[2] for j in jobs], backend, opt)

    loc_log_dir = os.path.join(os.path.dirname(path), "log")
    for rec in recs:
        _write_localize_log(loc_log_dir, rec, opt)
    if recs:
        append_token_records(os.path.dirname(path), recs, with_attempt=False)
    return len(recs)


def _write_localize_log(log_dir, rec, opt) -> None:
    """One per-task localize log under ``localize/<method>/log/<task>.log``.

    Intentionally minimal: only the FINAL localize prompt and response (plus the
    predicted lines) — not the per-stage transcript the repair log keeps."""
    import json
    import os

    from SpecCoder.src.experiment.gen_utils import task_log_name

    os.makedirs(log_dir, exist_ok=True)
    sep = "═" * 64
    path = os.path.join(log_dir, f"{task_log_name(rec['task_id'])}.log")
    lines = [
        f"╔{sep}╗",
        f"║  LOCALIZE TASK: {rec['task_id']}   method={getattr(opt, 'method', '')}"
        f"  model={getattr(opt, 'model', '')}",
        f"╠{sep}╣",
        "║  [LOCALIZE PROMPT]",
        f"╚{sep}╝",
        str(rec.get("prompt", "")),
        "",
        "[LOCALIZE RESPONSE]",
        str(rec.get("response", "")),
        "",
        "[PREDICTED LINES] (least -> most certain)",
        json.dumps(rec.get("prediction", []), ensure_ascii=False, indent=2),
        "",
        f"── END LOCALIZE TASK={rec['task_id']} ──────────────────────────────────",
        "",
    ]
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
