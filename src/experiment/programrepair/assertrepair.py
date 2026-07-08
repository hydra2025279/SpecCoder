
from __future__ import annotations

import re
import time
from typing import Any

from SpecCoder.src.experiment.programrepair import alpha_beta, common
from SpecCoder.src.experiment.specgen.inline.prompt import build_spec_generation_prompt

NAME = "assertrepair"

SPEC_SYSTEM = "You are an expert Python verification engineer writing inline # assert specifications."

REPAIR_SYSTEM = (
    "You are an expert Python program-repair engineer. You repair a buggy program guided by an "
    "analysis of its fault and advice on how to fix it; treat that guidance as hints."
)

ANALYSIS_SYSTEM = (
    "You are an experienced program-repair reviewer. Given a buggy program and behavioural "
    "checkpoints (inline `# assert` with execution evidence — how many test executions reach each "
    "checkpoint and how many satisfy it), you analyse where and why the program is incorrect and "
    "advise how to fix it. You do NOT write the corrected code."
)


def format_specs(violated: list[alpha_beta.SpecReachSatisfy]) -> str:
    """Render the violated specs as a bulleted list (lowest satisfy-rate first). The buggy
    listing is shown without line numbers, so each spec is anchored by its `# assert <expr>`
    text (which appears inline in the code), not a checkpoint line number."""
    return "\n".join(f"- `# assert {s.expr}`" for s in violated)


def format_reach_satisfy_report(result) -> str:
    if result is None or not getattr(result, "specs", None):
        return "(no specifications evaluated)"
    lines = [f"Computed by executing the program on all {result.num_tests} test case(s)."]
    for s in result.specs:
        lines.append(f"# assert {s.expr}")
        lines.append(
            f"  reached by {s.reach} execution(s), "
            f"{s.satisfy} satisfy / {s.reach - s.satisfy} do not"
            + ("  [VIOLATED]" if s.is_violated else
               ("  [never reached]" if s.reach == 0 else "  [always satisfied]"))
        )
    return "\n".join(lines)


def _evidence_phrase(s) -> str:
    """Plain-language execution evidence for one checkpoint: how many executions (over ALL
    tests) reach it and how many satisfy the assertion."""
    return (
        f"(reached by {s.reach} execution(s), of which {s.satisfy} satisfy the assertion "
        f"and {s.reach - s.satisfy} do not)"
    )


def _match_assert_spec(expr_text: str, specs_by_len: list[Any]):
    """Find the spec whose assert content matches an `# assert` line's expression text.
    Matches when the canonical ``s.expr`` equals ``expr_text`` or is a PREFIX of it ending
    at a boundary (whitespace / ``#`` / ``(``); prefers the LONGEST such expr. ``specs_by_len``
    must be sorted longest-expr first."""
    for s in specs_by_len:
        e = s.expr.strip()
        if not e:
            continue
        if expr_text == e:
            return s
        if expr_text.startswith(e) and expr_text[len(e):len(e) + 1] in (" ", "\t", "#", "("):
            return s
    return None


def _annotate_with_reach_satisfy(annotated: str, result) -> str:
    """REPLACE each checkpoint's `# assert` line with
    ``<indent># assert <expr>  (reached by …)`` so the execution evidence — how many
    executions reach it and how many satisfy it — is inlined for EVERY spec. Idempotent."""
    if not annotated or result is None or not getattr(result, "specs", None):
        return annotated
    specs_by_len = sorted(result.specs, key=lambda s: len(s.expr.strip()), reverse=True)
    out_lines = []
    for line in annotated.splitlines():
        stripped = line.strip()
        if stripped.startswith("# assert "):
            indent = line[: len(line) - len(line.lstrip())]
            expr_text = stripped[len("# assert "):].strip()
            s = _match_assert_spec(expr_text, specs_by_len)
            if s is not None:
                line = f"{indent}# assert {s.expr.strip()}  {_evidence_phrase(s)}"
        out_lines.append(line)
    return "\n".join(out_lines)


# ---- analysis prompt (reviewer-style); reach/satisfy wording ----

_GUIDE_BODY = (
    "### Guide — how to read the behavioural checkpoints\n"
    "Read the ANNOTATED BUGGY PROGRAM below carefully. Each `# assert` is placed after an "
    "important block of the program, and was executed on EVERY test case to narrow down the "
    "regions that may be causing the bug. The execution result of each assertion is described "
    "INLINE in the code, right after the assertion content, inside the `(...)` parentheses — "
    "how many executions REACH it, and how many of those SATISFY it.\n\n"
    "Utilize the execution path and result at each assertion to localize the potential issue in "
    "the buggy code. For example:\n"
    "- if an assertion is REACHED but some executions do NOT satisfy it → the intended state is "
    "broken by the time execution reaches that assertion; the bug is likely on the path up to "
    "(or at) that assertion;\n"
    "- if an assertion is satisfied by ALL executions that reach it, the state it captures is "
    "intact there, so the bug more likely lies elsewhere;\n"
    "- an assertion that NO execution reaches is on a code path no test exercises — the bug may "
    "be exactly why that path is never taken.\n\n"
)


def format_checkpoint_analysis(result, violated: list[alpha_beta.SpecReachSatisfy],
                               show_line: bool = False) -> str:
    """Classify every checkpoint by its reach/satisfy evidence over ALL tests (no pass/fail):

      VIOLATED        — reached by some execution but not satisfied by all of them. The bug
                        most likely sits on the path leading UP TO this assert. Highest priority.
      ALWAYS-SATISFIED— reached and satisfied by every execution. The captured state is intact
                        here; the bug is probably elsewhere.
      NEVER-REACHED   — no execution reaches it. The code path is never exercised by any test;
                        the bug may be why it is unreachable.
    """
    specs = list(getattr(result, "specs", []) or [])
    if not specs:
        return ""
    cat_viol, cat_sat, cat_dead = [], [], []
    for s in specs:
        tag = f"# assert {s.expr}  {_evidence_phrase(s)}"
        if s.reach == 0:
            cat_dead.append(tag)
        elif s.is_violated:
            cat_viol.append(tag)
        else:
            cat_sat.append(tag)

    blocks: list[str] = []
    if cat_viol:
        blocks.append(
            "**(1) VIOLATED checkpoints — reached, but some executions do NOT satisfy them** — the "
            "intended state is already broken when execution reaches here, so the bug is on the path "
            "UP TO (or at) the assertion. Trace the executions through these checkpoints to localize "
            "the fault.\n" + "\n".join(f"  - {t}" for t in cat_viol)
        )
    if cat_sat:
        blocks.append(
            "**(2) ALWAYS-SATISFIED checkpoints — reached and satisfied by every execution** — the "
            "state captured here is intact, so the bug most likely lies on a path that does NOT pass "
            "through them (or after them).\n" + "\n".join(f"  - {t}" for t in cat_sat)
        )
    if cat_dead:
        blocks.append(
            "**(3) NEVER-REACHED checkpoints — no test execution reaches them** — this code path is "
            "never exercised; if it should have run, the bug may be exactly WHY it is unreachable (a "
            "guard that is always false, an early exit, a wrong branch upstream).\n"
            + "\n".join(f"  - {t}" for t in cat_dead)
        )
    return "\n\n".join(blocks)


def build_analysis_prompt(sample, candidate: str, violated: list[alpha_beta.SpecReachSatisfy],
                          rs_result=None, annotated_code: str | None = None) -> str:
    """Reviewer-style analysis prompt: present the buggy program with its `# assert`
    checkpoints, then a subsection analysing which executions reach and satisfy each
    checkpoint. Ask for a bug ANALYSIS + fix ADVICE (no code)."""
    listing = annotated_code or candidate
    analysis = format_checkpoint_analysis(rs_result, violated, show_line=False)
    analysis_block = (
        f"\n### Checkpoint execution analysis (how many executions reach and satisfy each `# assert`)\n{analysis}\n"
        if analysis else ""
    )
    specs_block = (
        f"### Useful behavioural signals (violated checkpoints)\n{format_specs(violated)}\n"
        if violated else ""
    )
    return f"""### Role
You review a buggy Python program and produce an ANALYSIS of its fault — you do NOT write code.
You are given *behavioural specifications*: inline `# assert <expr>` checkpoints meant to capture
intended program state at key points, plus an analysis of how many test executions reach and
satisfy each checkpoint. Treat them as HYPOTHESES that may be wrong — verify them against the
problem semantics before relying on them.

{common.problem_text(sample)}

### Buggy program (annotated with `# assert` checkpoints)
```python
{listing}
```

{specs_block}{analysis_block}
### How to respond
Return your response as exactly two tagged sections:

<analysis>
First, trace the EXECUTION PATH: explain the semantics of each `# assert` specification and use the checkpoint execution analysis above to describe how the program's executions flow through it — which checkpoints stay satisfied and which get violated, and where the intended state first breaks.
Then, analyse the POTENTIAL ISSUES in the buggy program: which line(s) are most likely wrong and why, grounded in that execution-path reasoning and the problem description.
</analysis>

<advice>
Concrete, actionable guidance on how to fix the program (do NOT write the corrected code).
</advice>
"""


def analysis_messages(sample, candidate: str, violated: list[alpha_beta.SpecReachSatisfy],
                      rs_result=None, annotated_code: str | None = None) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": ANALYSIS_SYSTEM},
        {"role": "user", "content": build_analysis_prompt(
            sample, candidate, violated, rs_result=rs_result, annotated_code=annotated_code)},
    ]


def parse_analysis(text: str) -> tuple[str, str]:
    """Parse ``<analysis>…</analysis>`` / ``<advice>…</advice>``; tolerate a reply with no
    tags by treating the whole response as the analysis."""
    def _grab(tag: str) -> str:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text or "", re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""
    analysis, advice = _grab("analysis"), _grab("advice")
    if not analysis and not advice:
        return (text or "").strip(), ""
    return analysis, advice



def compose_repair_feedback(analysis: str, advice: str) -> str:
    parts: list[str] = []
    if analysis.strip():
        parts.append(f"### Analysis\n{analysis.strip()}")
    if advice.strip():
        parts.append(f"### Advice\n{advice.strip()}")
    return "\n\n".join(parts)


def build_repair_feedback_prompt(sample, candidate: str, feedback: str) -> str:
    feedback_section = f"\n### Feedback\n{feedback.strip()}\n" if feedback.strip() else ""
    return f"""### Role
Task: repair the buggy Python program so it satisfies the problem specification and passes the tests.
Use the provided specification, examples, and buggy code. Preserve the required stdin/stdout interface.
Prefer a small patch, but produce a complete correct program if a rewrite is necessary.

{common.problem_text(sample)}

### Buggy program
```python
{candidate}
```
{feedback_section}### How to respond (think step by step, then repair)
First, reason step by step: identify and localize what the buggy program does wrong versus the problem description and the analysis/advice provided, then describe the minimal fix. After your reasoning, output the corrected program (the clean version, repaired)."""


def repair_feedback_messages(sample, candidate: str, feedback: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": REPAIR_SYSTEM},
        {"role": "user", "content": build_repair_feedback_prompt(sample, candidate, feedback)},
    ]


def improve_batch(samples, buggys, candidates, backend, opt, states) -> list[tuple[str, dict[str, Any]]]:
    """assertrepair signal (batched).

    For every candidate: generate inline `# assert` specs ONCE, execute the annotated
    program on EVERY test once to get per-checkpoint reach/satisfy counts, keep the
    violated checkpoints, run a reviewer-style ANALYSIS over them, then a single repair
    generation. It uses reach/satisfy evidence over the available tests.

    Spec generation is retried up to ``--spec-regen`` rounds (default 5): a task "fails"
    when the model returns no parseable inline `# assert`, and only the still-failing
    subset is re-generated (batched in parallel) each round."""
    n = len(samples)
    t0 = time.perf_counter()
    eval_workers = common.test_batch_of(opt)

    costs: list[list[dict[str, int]]] = [[] for _ in range(n)]
    phases: list[list[dict[str, str]]] = [[] for _ in range(n)]
    feedbacks: list[str | None] = [None] * n
    viols: list[list[alpha_beta.SpecReachSatisfy]] = [[] for _ in range(n)]
    rs_: list[Any] = [None] * n
    annotated_src: list[str | None] = [None] * n
    nviol = [0] * n


    max_rounds = max(1, int(getattr(opt, "spec_retry", 5) or 1))
    spec_msgs = [[
        {"role": "system", "content": SPEC_SYSTEM},
        {"role": "user", "content": build_spec_generation_prompt(
            {"problem_text": common.problem_text(samples[i])}, candidates[i])},
    ] for i in range(n)]
    spec_resp: list[str] = [""] * n           
    spec_rounds: list[int] = [0] * n          
    spec_prio: list[int] = [-1] * n          
    pending = list(range(n))                  
    for rnd in range(max_rounds):
        common.progress(NAME, f"spec generation round {rnd + 1}/{max_rounds} on {len(pending)} task(s) "
                              f"(regen until a checkpoint is violated)")
        comps = common.chat_batched(
            backend, [spec_msgs[i] for i in pending], opt, num_samples=1,
            task_ids=[common.task_id(samples[i]) for i in pending],
        )
        still: list[int] = []
        for slot, i in enumerate(pending):
            resp = (comps[slot] or [""])[0]
            costs[i].append(common.attempt_cost(spec_msgs[i][-1]["content"], resp))
            spec_rounds[i] += 1
            annotated = alpha_beta.extract_annotated(resp)
            if annotated and "# assert" in annotated:
                result = alpha_beta.compute_reach_satisfy(
                    annotated, samples[i].get("test_cases") or [],
                    memory_mb=None, parallelism=eval_workers,
                )
                prio = 2 if result.violated else 1
            else:
                result, prio = None, 0
            if prio > spec_prio[i]:            # keep the best spec seen across rounds
                spec_prio[i] = prio
                spec_resp[i] = resp
                rs_[i] = result
                annotated_src[i] = annotated if result is not None else None
            if prio == 2:
                continue                       # violated checkpoint found -> stop regenerating
            still.append(i)                    # no violation yet -> regen next round
        pending = still
        if not pending:
            break
    if pending:
        common.progress(NAME, f"spec generation: {len(pending)} task(s) without a violated checkpoint "
                              f"after {max_rounds} round(s) -> analysis + clean repair anyway")

    # --- Phase 2: finalize per-task signal + logging from the best spec (no recompute). ---
    for i in range(n):
        result = rs_[i]
        if result is not None and result.violated:
            viols[i] = result.violated
            feedbacks[i] = format_specs(result.violated)
            nviol[i] = len(result.violated)
        if result is None:
            log_response = (f"{spec_resp[i]}\n\n--- Reach/Satisfy of generated specifications ---\n"
                            f"(no annotated code parsed)")
        else:
            log_response = (
                f"{spec_resp[i]}\n\n--- Reach/Satisfy of generated specifications ---\n"
                f"{format_reach_satisfy_report(result)}"
            )
        common.emit(common.task_id(samples[i]), phases[i], {
            "name": "SPEC GENERATION",
            "prompt": spec_msgs[i][-1]["content"],
            "response": log_response,
        })
        _rnd = f" [spec-gen x{spec_rounds[i]}]" if spec_rounds[i] > 1 else ""
        common.stage_done(
            common.task_id(samples[i]), "TEST MILESTONE: assertrepair reach/satisfy",
            info=((f"{nviol[i]} violated checkpoint(s) -> ready to repair" if feedbacks[i]
                   else "no violated checkpoint -> clean repair") + _rnd))

    has_signal = [bool(rs_[i] is not None and getattr(rs_[i], "specs", None)) for i in range(n)]

    analysis_txt: list[str] = [""] * n
    advice_txt: list[str] = [""] * n
    sig_idx = list(range(n))
    common.progress(NAME, f"spec analysis on {len(sig_idx)} task(s) "
                          f"({sum(has_signal)} with reach/satisfy evidence)")
    amsgs = [analysis_messages(
        samples[i], candidates[i], viols[i], rs_result=rs_[i],
        annotated_code=(_annotate_with_reach_satisfy(annotated_src[i], rs_[i])
                        if (annotated_src[i] and rs_[i] is not None) else None),
    ) for i in sig_idx]
    acomps = common.chat_batched(backend, amsgs, opt, num_samples=1,
                                 task_ids=[common.task_id(samples[i]) for i in sig_idx],
                                 use_lora=False)
    for slot, i in enumerate(sig_idx):
        atext = acomps[slot][0] if acomps[slot] else ""
        analysis_txt[i], advice_txt[i] = parse_analysis(atext)
        costs[i].append(common.attempt_cost(amsgs[slot][-1]["content"], atext))
        common.emit(common.task_id(samples[i]), phases[i],
                    {"name": "SPEC ANALYSIS", "prompt": amsgs[slot][-1]["content"], "response": atext})

    final_msgs: list[list[dict[str, str]]] = [
        repair_feedback_messages(samples[i], candidates[i],
                                 compose_repair_feedback(analysis_txt[i], advice_txt[i]))
        for i in range(n)
    ]
    final_msgs = [common.repair_conversation_messages(final_msgs[i], states[i]) for i in range(n)]
    common.progress(NAME, f"repair generation ({sum(1 for f in feedbacks if f)}/{n} task(s) carry violated checkpoints)")
    rcomps = common.chat_batched(
        backend, final_msgs, opt, num_samples=1,
        task_ids=[common.task_id(s) for s in samples],
        use_lora=False,
    )
    new_codes = list(candidates)
    for i in range(n):
        resp = rcomps[i][0] if rcomps[i] else ""
        new_codes[i] = common.extract_code(resp) or candidates[i]
        costs[i].append(common.attempt_cost(final_msgs[i][-1]["content"], resp))
        common.remember_repair_exchange(states[i], final_msgs[i], resp)
        common.emit(common.task_id(samples[i]), phases[i],
                    {"name": "REPAIR", "prompt": common.format_chat_messages(final_msgs[i]), "response": resp})

    elapsed = (time.perf_counter() - t0) / max(1, n)
    out: list[tuple[str, dict[str, Any]]] = []
    for i in range(n):
        _fb_text = compose_repair_feedback(analysis_txt[i], advice_txt[i])
        clean_annotated = annotated_src[i]
        enhance_annotated = (
            _annotate_with_reach_satisfy(annotated_src[i], rs_[i])
            if (annotated_src[i] and rs_[i] is not None) else annotated_src[i]
        )
        loc_signal = {
            "text": _fb_text,
            "annotated_code": annotated_src[i],
            "reach_satisfy": rs_[i].to_dict() if rs_[i] else None,
        }
        out.append((new_codes[i], {
            "changed": new_codes[i] != candidates[i],
            "feedback": feedbacks[i] or "",
            "reach_satisfy": rs_[i].to_dict() if rs_[i] else None,
            "spec_annotated": annotated_src[i],
            "clean_annotated": clean_annotated,
            "enhance_annotated": enhance_annotated,
            "reused_saved_spec": False,
            "num_violated_specs": nviol[i],
            "localize_signal": loc_signal,
            "phases": phases[i],
            "cost": common.sum_token_usage(costs[i]),
            "seconds": elapsed,
        }))
    return out


def improve(sample, buggy, candidate, backend, opt, state) -> tuple[str, dict[str, Any]]:
    return improve_batch([sample], [buggy], [candidate], backend, opt, [state])[0]
