from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from SpecCoder.src.experiment.programrepair import common

NAME = "specrover"

_ADAPTER_DIR = common.PROJECT_ROOT / "SpecCoder" / "others" / "spec-rover-artifact" / "code" / "app"
if str(_ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(_ADAPTER_DIR))

import repair_adapter as ra  # noqa: E402

_REPRO_RETRIES = 3


def _run_program(code: str, stdin: str, timeout: float) -> tuple[int, str, str]:
    if stdin and not stdin.endswith("\n"):
        stdin += "\n"
    with tempfile.TemporaryDirectory(prefix="specrover_") as tmp:
        path = Path(tmp) / "submission.py"
        path.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return -1, "", "timeout"
        except Exception as exc:
            return -1, "", str(exc)
    return proc.returncode, proc.stdout, proc.stderr


def _tokens(text: str) -> list[str]:
    return (text or "").split()


def _exposes_bug(code: str, test: dict[str, str], timeout: float) -> bool:
    rc, out, _err = _run_program(code, test.get("input", ""), timeout)
    if rc != 0:
        return True
    return _tokens(out) != _tokens(test.get("expected", ""))


def _localize_signal_text(spec: str, test: dict[str, str] | None, raw_review: str) -> str:
    """Compose specrover's LOCALIZE context: the inferred behavioural spec, the
    bug-reproducing test, and the grounded reviewer analysis. Used to build the
    saved localize prompt, independent of whether the patch was approved."""
    parts: list[str] = []
    if spec:
        parts.append(f"Inferred specification of the intended behaviour:\n{spec}")
    if test:
        parts.append(
            "Reproducer test that exposes the bug:\n"
            f"input:\n{test.get('input', '')}\nexpected:\n{test.get('expected', '')}")
    if raw_review:
        parts.append(f"Grounded reviewer analysis:\n{raw_review}")
    return "\n\n".join(parts)


def _ask(backend, prompt: str, system: str, costs: list[dict[str, int]]) -> str:
    messages = [{"role": "user", "content": prompt}]
    if system:
        messages = [{"role": "system", "content": system}] + messages
    response = backend.chat([messages], num_samples=1)[0]
    text = response[0] if response else ""
    costs.append(common.attempt_cost(prompt, text))
    return text


def _gen_reproducer(sample, buggy: str, backend, opt, costs: list[dict[str, int]]) -> dict[str, str] | None:
    problem = common.problem_text(sample)
    last: dict[str, str] | None = None
    candidate: dict[str, str] | None = None
    for _ in range(_REPRO_RETRIES):
        prompt = ra.reproducer_prompt(problem, buggy, last)
        response = _ask(backend, prompt, ra.REPRODUCER_SYSTEM, costs)
        candidate = ra.parse_test_case(response)
        if candidate is None:
            continue
        if _exposes_bug(buggy, candidate, opt.eval_timeout):
            return candidate
        last = candidate
    return candidate


def _infer_spec(sample, buggy: str, test, backend, opt, costs: list[dict[str, int]]) -> str:
    prompt = ra.spec_prompt(common.problem_text(sample), buggy, test)
    return _ask(backend, prompt, ra.SPEC_SYSTEM, costs).strip()


def _review(sample, spec: str, patch_0: str, test, backend, opt, costs):
    if test is None:
        return None, ""
    _rc_p, out_p, err_p = _run_program(patch_0, test.get("input", ""), opt.eval_timeout)
    test_str = f"input:\n{test.get('input', '')}\nexpected:\n{test.get('expected', '')}"
    messages = ra.review_messages(
        common.problem_text(sample), spec, test_str, patch_0,
        out_p, err_p,
    )
    response = backend.chat([messages], num_samples=1)[0]
    text = response[0] if response else ""
    costs.append(common.attempt_cost(messages[-1]["content"], text))
    return ra.extract_review_result(text), text


def _gen_reproducers_batched(samples, buggys, idx, backend, opt, costs, phases):
    """Reproduce the bug for ``idx`` samples, batching one LLM call per retry-round
    across all still-pending samples. Returns ``{i: test_case|None}``."""
    last: dict[int, dict[str, str] | None] = {i: None for i in idx}
    found: dict[int, dict[str, str] | None] = {i: None for i in idx}
    pending = list(idx)
    for _round in range(_REPRO_RETRIES):
        if not pending:
            break
        msgs = [[
            {"role": "system", "content": ra.REPRODUCER_SYSTEM},
            {"role": "user", "content": ra.reproducer_prompt(common.problem_text(samples[i]), buggys[i], last[i])},
        ] for i in pending]
        comps = backend.chat(msgs, num_samples=1, task_ids=[common.task_id(samples[i]) for i in pending])
        still: list[int] = []
        for slot, i in enumerate(pending):
            text = comps[slot][0] if comps[slot] else ""
            costs[i].append(common.attempt_cost(msgs[slot][-1]["content"], text))
            common.emit(common.task_id(samples[i]), phases[i], {"name": f"reproducer.r{_round}", "prompt": msgs[slot][-1]["content"], "response": text})
            cand = ra.parse_test_case(text)
            if cand is None:
                still.append(i)
                common.stage_done(common.task_id(samples[i]),
                                  f"TEST MILESTONE: specrover reproducer r{_round}",
                                  info="no parseable test -> retry")
                continue
            found[i] = cand
            exposes = _exposes_bug(buggys[i], cand, opt.eval_timeout)
            if not exposes:
                last[i] = cand
                still.append(i)
            common.stage_done(common.task_id(samples[i]),
                              f"TEST MILESTONE: specrover reproducer r{_round}",
                              info=("bug reproduced -> done" if exposes else "does not expose bug -> retry"))
        pending = still
    return found


def improve_batch(samples, buggys, candidates, backend, opt, states) -> list[tuple[str, dict[str, Any]]]:
    """Batched SpecRover signal. Reproducer (per-retry-round batched) + spec
    (batched) are cached per sample in ``states[i]``; the reviewer is grounded by
    executing the reproducer on buggy+candidate, then run batched; rejected
    candidates are refined in one batched repair call. Returns ``[(new_code, info)]``."""
    n = len(samples)
    t0 = time.perf_counter()
    costs: list[list[dict[str, int]]] = [[] for _ in range(n)]
    phases: list[list[dict[str, str]]] = [[] for _ in range(n)]

    # Phase A1: generate the reproducer once and retain it for the whole attempt.
    test_todo = [i for i in range(n) if "test" not in states[i]]
    if test_todo:
        common.progress("specrover", f"reproducer generation on {len(test_todo)} task(s)")
        found = _gen_reproducers_batched(
            samples, buggys, test_todo, backend, opt, costs, phases)
        for i in test_todo:
            states[i]["test"] = found[i]

    spec_todo = [i for i in range(n) if "spec" not in states[i]]
    if spec_todo:
        common.progress("specrover", f"spec inference on {len(spec_todo)} task(s)")
        smsgs = [[
            {"role": "system", "content": ra.SPEC_SYSTEM},
            {"role": "user", "content": ra.spec_prompt(common.problem_text(samples[i]), buggys[i], states[i]["test"])},
        ] for i in spec_todo]
        scomps = backend.chat(
            smsgs, num_samples=1,
            task_ids=[common.task_id(samples[i]) for i in spec_todo])
        for slot, i in enumerate(spec_todo):
            stext = scomps[slot][0] if scomps[slot] else ""
            costs[i].append(common.attempt_cost(smsgs[slot][-1]["content"], stext))
            states[i]["spec"] = stext.strip()
            common.emit(common.task_id(samples[i]), phases[i], {"name": "spec", "prompt": smsgs[slot][-1]["content"], "response": stext})

    tests = [states[i].get("test") for i in range(n)]
    specs = [states[i].get("spec", "") for i in range(n)]


    reviews: list[Any] = [None] * n
    raws = [""] * n
    rmsgs = []
    ridx = []
    _todo_b = [i for i in range(n) if tests[i] is not None]
    if _todo_b:
        common.progress("specrover", f"reviewer-test milestone on {len(_todo_b)} task(s)")
    _bbar = common.step_bar(len(_todo_b), "specrover: reviewer-test exec")
    for i in range(n):
        if tests[i] is None:
            continue
        rc_p, out_p, err_p = _run_program(candidates[i], tests[i].get("input", ""), opt.eval_timeout)
        test_str = f"input:\n{tests[i].get('input', '')}\nexpected:\n{tests[i].get('expected', '')}"
        rmsgs.append(ra.review_messages(
            common.problem_text(samples[i]), specs[i], test_str, candidates[i],
            out_p, err_p,
        ))
        ridx.append(i)
        common.stage_done(common.task_id(samples[i]), "TEST MILESTONE: specrover reviewer-test exec",
                          info=f"rc={rc_p}")
        if _bbar is not None:
            _bbar.update(1)
    if _bbar is not None:
        _bbar.close()
    if rmsgs:
        common.progress("specrover", f"grounded reviewer on {len(rmsgs)} task(s)")
        rcomps = backend.chat(rmsgs, num_samples=1, task_ids=[common.task_id(samples[i]) for i in ridx])
        for slot, i in enumerate(ridx):
            text = rcomps[slot][0] if rcomps[slot] else ""
            costs[i].append(common.attempt_cost(rmsgs[slot][-1]["content"], text))
            reviews[i] = ra.extract_review_result(text)
            raws[i] = text
            common.emit(common.task_id(samples[i]), phases[i], {"name": "review", "prompt": rmsgs[slot][-1]["content"], "response": text})

    approved = [reviews[i] is not None and reviews[i].patch_decision == ra.ReviewDecision.YES for i in range(n)]

    new_codes = list(candidates)
    feedbacks: list[str | None] = [None] * n
    need = [i for i in range(n) if not approved[i]]
    if need:
        common.progress("specrover", f"refine {len(need)} rejected candidate(s) ({sum(approved)} approved)")
        fmsgs = []
        for i in need:
            feedbacks[i] = ra.compose_patch_feedback(specs[i], reviews[i], tests[i])
            base = common.repair_messages(samples[i], candidates[i], feedbacks[i])
            fmsgs.append(common.repair_conversation_messages(base, states[i]))
        fcomps = common.chat_batched(backend, fmsgs, opt, num_samples=1,
                                     task_ids=[common.task_id(samples[i]) for i in need])
        for slot, i in enumerate(need):
            resp = fcomps[slot][0] if fcomps[slot] else ""
            new_codes[i] = common.extract_code(resp) or candidates[i]
            costs[i].append(common.attempt_cost(fmsgs[slot][-1]["content"], resp))
            common.remember_repair_exchange(states[i], fmsgs[slot], resp)
            common.emit(common.task_id(samples[i]), phases[i], {
                "name": "refine",
                "prompt": common.format_chat_messages(fmsgs[slot]),
                "response": resp,
            })

    elapsed = (time.perf_counter() - t0) / max(1, n)
    out: list[tuple[str, dict[str, Any]]] = []
    for i in range(n):
        out.append((new_codes[i], {
            "changed": new_codes[i] != candidates[i],
            "feedback": feedbacks[i] or "",
            "reproducer_test": tests[i],
            "spec": specs[i],
            "review": ra.review_to_json(reviews[i]) if reviews[i] is not None else None,
            "raw_review": raws[i],
            "approved": approved[i],
            "localize_signal": {"text": _localize_signal_text(specs[i], tests[i], raws[i])},
            "phases": phases[i],
            "cost": common.sum_token_usage(costs[i]),
            "seconds": elapsed,
        }))
    return out


def improve(sample, buggy, candidate, backend, opt, state) -> tuple[str, dict[str, Any]]:
    return improve_batch([sample], [buggy], [candidate], backend, opt, [state])[0]


def run_method(rows: list[dict[str, Any]], backend, opt) -> list[dict[str, Any]]:
    samples_per_task = max(1, int(getattr(opt, "num_samples", 1) or 1))
    expanded = [sample for sample in rows for _ in range(samples_per_task)]
    attempts = [attempt for _sample in rows for attempt in range(samples_per_task)]
    n = len(expanded)
    buggy = [common.buggy_code_of(sample) for sample in expanded]
    costs: list[list[dict[str, int]]] = [[] for _ in range(n)]
    generation_seconds = [0.0] * n

    msgs0 = [common.repair_messages(expanded[i], buggy[i], "") for i in range(n)]
    comps0, meta0 = common.timed_chat_batched(backend, msgs0, opt, num_samples=1)
    patch0: list[str] = []
    for i in range(n):
        resp = comps0[i][0] if comps0[i] else ""
        patch0.append(common.extract_code(resp) or buggy[i])
        costs[i].append(common.attempt_cost(msgs0[i][-1]["content"], resp))
        generation_seconds[i] += float(meta0[i].get("generation_seconds", 0.0) or 0.0)

    feedbacks: list[str | None] = [None] * n
    infos: list[dict[str, Any]] = [{} for _ in range(n)]
    for i in range(n):
        sample, b, p0 = expanded[i], buggy[i], patch0[i]
        test = _gen_reproducer(sample, b, backend, opt, costs[i])
        spec = _infer_spec(sample, b, test, backend, opt, costs[i])
        review, raw_review = _review(sample, spec, p0, test, backend, opt, costs[i])
        approved = review is not None and review.patch_decision == ra.ReviewDecision.YES
        feedbacks[i] = None if approved else ra.compose_patch_feedback(spec, review, test)
        infos[i] = {
            "reproducer_test": test,
            "spec": spec,
            "review": ra.review_to_json(review) if review is not None else None,
            "raw_review": raw_review,
            "approved": approved,
        }

    final = list(patch0)
    need = [i for i in range(n) if feedbacks[i] is not None]
    if need:
        msgs1 = [common.repair_messages(expanded[i], buggy[i], feedbacks[i]) for i in need]
        comps1, meta1 = common.timed_chat_batched(backend, msgs1, opt, num_samples=1)
        for slot, i in enumerate(need):
            resp = comps1[slot][0] if comps1[slot] else ""
            final[i] = common.extract_code(resp) or patch0[i]
            costs[i].append(common.attempt_cost(msgs1[slot][-1]["content"], resp))
            generation_seconds[i] += float(meta1[slot].get("generation_seconds", 0.0) or 0.0)

    records: list[dict[str, Any]] = []
    for i in range(n):
        rec = common.base_record(expanded[i], opt, buggy[i], NAME)
        info = infos[i]
        transcript = (
            f"[PATCH_0]\n{patch0[i]}\n\n[SPEC]\n{info['spec']}\n\n"
            f"[REVIEW]\n{info['raw_review']}\n\n[FINAL]\n{final[i]}"
        )
        rec.update({
            "attempt": attempts[i],
            "code": final[i],
            "rounds": 0 if feedbacks[i] is None else 1,
            "patch_0": patch0[i],
            "reproducer_test": info["reproducer_test"],
            "spec": info["spec"],
            "review": info["review"],
            "approved": info["approved"],
            "prompt": common.build_repair_prompt(expanded[i], buggy[i], feedbacks[i] or ""),
            "response": transcript,
            "time_latency": {
                "generation_seconds": round(generation_seconds[i], 6),
                "global_batch": int(getattr(opt, "global_batch", 0) or getattr(opt, "task_batch_size", 0) or 0),
                "local_batch": 1,
            },
            "cost_token": common.sum_token_usage(costs[i]),
        })
        records.append(rec)
    return records
