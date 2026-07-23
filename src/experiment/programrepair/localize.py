
from __future__ import annotations

import json
import os
import re
import threading
from typing import Any

from SpecCoder.spec_utils import extract_assertions_from_annotated_code
from SpecCoder.src.experiment.programrepair import alpha_beta, common

NAME = "localize"

LOCALIZE_SYSTEM = (
    "You are an expert Python debugger for competitive-programming submissions. "
    "Given a buggy program and a method-specific signal (inline `# assert` checkpoints "
    "with their execution evidence, or another hint), you localize the bug. The signal is "
    "a HYPOTHESIS and may itself be wrong, so verify it before trusting it. Think step "
    "by step about the bug location first, then return the likely-buggy lines of the CLEAN "
    "program as a list of `(line_number, line_content)` tuples (1-based) inside a single "
    "`<localize>...</localize>` tag, e.g. `<localize>[(2, \"x = a - b\")]</localize>`, ordered "
    "from least to most certain."
)

_LOCALIZE_WRITE_LOCK = threading.Lock()


def number_source(code: str) -> str:
    """1-based line listing of ``code``."""
    lines = code.split("\n")
    width = len(str(len(lines)))
    return "\n".join(f"{i:>{width}} | {line}" for i, line in enumerate(lines, start=1))


_OUTPUT_SPEC = (
    "### Required Output\n"
    "After your reasoning, on the LAST line return ONLY the suspicious lines as a Python list "
    "of `(line_number, line_content)` tuples — using the CLEAN program's 1-based line numbers "
    "and copying each line's content VERBATIM — ordered from least to most certain (the most "
    "suspicious line LAST), inside a single `<localize>...</localize>` tag. Put nothing else "
    "inside the tag. Example:\n\n"
    "<localize>[(7, \"    c = a - b\"), (2, \"x = int(input())\")]</localize>"
)


def _reach_satisfy_from_dict(rs_dict: dict | None):
    if not rs_dict or not rs_dict.get("specs"):
        return None
    specs = []
    for d in rs_dict.get("specs") or []:
        specs.append(alpha_beta.SpecReachSatisfy(
            trace_key=str(d.get("trace_key", "")),
            assert_key=str(d.get("assert_key", "")),
            expr=str(d.get("expr", "")),
            checkpoint_line=int(d.get("checkpoint_line", 0) or 0),
            reach=int(d.get("reach", 0) or 0),
            satisfy=int(d.get("satisfy", 0) or 0),
            num_tests=int(d.get("num_tests", rs_dict.get("num_tests", 0)) or 0),
        ))
    violated = [s for s in specs if s.is_violated]
    return alpha_beta.ReachSatisfyResult(
        specs=specs,
        violated=violated,
        num_tests=int(rs_dict.get("num_tests", 0) or 0),
    )


def _checkpoint_analysis(rs_dict: dict | None) -> str:
    result = _reach_satisfy_from_dict(rs_dict)
    if result is None:
        return ""
    try:
        from SpecCoder.src.experiment.programrepair.assertrepair import format_checkpoint_analysis
        return format_checkpoint_analysis(result, result.violated, show_line=False)
    except Exception:
        return ""


def _annotate_with_reach_satisfy(annotated_code: str | None, rs_dict: dict | None) -> str | None:
    if not annotated_code:
        return annotated_code
    result = _reach_satisfy_from_dict(rs_dict)
    if result is None:
        return annotated_code
    try:
        from SpecCoder.src.experiment.programrepair.assertrepair import _annotate_with_reach_satisfy as _inline
        return _inline(annotated_code, result)
    except Exception:
        return annotated_code


def build_localize_prompt(
    sample,
    clean_code: str,
    *,
    method: str = "naive",
    signal_text: str = "",
    annotated_code: str | None = None,
    violated: list[alpha_beta.SpecReachSatisfy] | None = None,
    reach_satisfy_result: dict | None = None,
) -> str:
    problem = common.problem_text(sample)
    clean_block = (
        "### Buggy program — PREDICT line numbers against THIS (the ONLY 1-based line numbering)\n"
        f"```\n{number_source(clean_code)}\n```"
    )
    parts = [
        "### Role",
        "You localize the bug in a buggy Python program: you return the source lines most likely to contain the fault.",
    ]

    analysis = _checkpoint_analysis(reach_satisfy_result) if annotated_code is not None else ""
    annotated_rs = _annotate_with_reach_satisfy(annotated_code, reach_satisfy_result)

    if annotated_code is not None and (violated or analysis):
        guide = (
            "### Guide — how to read the behavioural checkpoints\n"
            "Read the ANNOTATED BUGGY PROGRAM below carefully. Each `# assert` is placed after an "
            "important block of the program, and was executed on the available tests to narrow down "
            "the regions that may be causing the bug. The execution result of "
            "each assertion is described INLINE in the code, right after the assertion content, inside "
            "the `(...)` parentheses — how many executions REACH it, and how many SATISFY it.\n\n"
            "Utilize the execution path and result at each assertion to localize the potential issue. "
            "For example:\n"
            "- if an assertion is REACHED but some executions do NOT satisfy it → the bug is likely "
            "on the path up to (or at) that assertion;\n"
            "- if an assertion is reached and satisfied by every execution, the captured state is "
            "probably intact there;\n"
            "- if no execution reaches an assertion, inspect the upstream control flow that should "
            "lead to that checkpoint.\n\n"
            "These checkpoints are HYPOTHESES and may be WRONG — verify them against the problem "
            "semantics; if unreasonable, localize the bug on your own from the problem and the code. "
            "The ANNOTATED BUGGY PROGRAM is NOT line-numbered and is for understanding only; predict "
            "your line numbers against the CLEAN line-numbered program shown at the end."
        )
        parts += [
            "",
            guide,
            "",
            problem,
            "",
            "### ANNOTATED BUGGY PROGRAM",
            f"```python\n{annotated_rs}\n```",
            "",
            clean_block,
        ]
    elif signal_text.strip():
        parts += [
            "",
            "### Important — the signal may be wrong",
            f"You are given a **{method}** signal computed for this program (below). It is a HYPOTHESIS, not ground truth. First VERIFY it against the problem semantics: if it is reasonable, use it to localize the bug; if it is not reasonable, ignore it and localize the bug on your own, reasoning directly from the problem statement and the code.",
            "",
            problem,
            "",
            f"### {method} signal",
            signal_text.strip(),
            "",
            clean_block,
        ]
    else:
        parts += [
            "",
            "No additional signal is provided. Localize the bug on your own, reasoning directly from the problem statement and the code.",
            "",
            problem,
            "",
            clean_block,
        ]

    parts += ["", "### How to localize (think step by step, then answer)"]
    if annotated_code is not None and (violated or analysis):
        parts += [
            "1. **Read the annotated program.** Read the ANNOTATED BUGGY PROGRAM, explain the semantics of each inline `# assert` specification, and reason about the execution path using the `(reached by …)` notation after these assertions.",
            "2. **Localization reasoning (chain of thought).** Reason step by step about which line(s) are most likely responsible for the wrong behaviour, using that inline `# assert` execution evidence.",
            "3. **Verify the signal.** State whether each provided specification/signal is actually correct and relevant, or misleading — do not trust it blindly; if unreasonable, localize on your own from the problem and code.",
            "4. **Answer.** Map the suspicious code to the CLEAN program's 1-based line numbers shown above and report them.",
        ]
    else:
        parts += [
            "1. **Localization reasoning (chain of thought).** Reason step by step about which line(s) are most likely responsible for the wrong behaviour, using the method context above.",
            "2. **Verify the signal.** State whether each provided specification/signal is actually correct and relevant, or misleading — do not trust it blindly; if unreasonable, localize on your own from the problem and code.",
            "3. **Answer.** Map the suspicious code to the CLEAN program's 1-based line numbers shown above and report them.",
        ]
    parts += ["", _OUTPUT_SPEC]
    return "\n".join(parts)


def localize_messages(
    sample,
    clean_code: str,
    *,
    method: str = "naive",
    signal_text: str = "",
    annotated_code: str | None = None,
    violated: list[alpha_beta.SpecReachSatisfy] | None = None,
    reach_satisfy_result: dict | None = None,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": LOCALIZE_SYSTEM},
        {"role": "user", "content": build_localize_prompt(
            sample, clean_code, method=method, signal_text=signal_text,
            annotated_code=annotated_code, violated=violated,
            reach_satisfy_result=reach_satisfy_result)},
    ]


def parse_prediction(response: str, clean_code: str | None = None) -> list[str]:
    scope = response or ""
    items: list[Any] = []

    loc = re.search(r"<localize>(.*?)(?:</localize>|\Z)", scope, re.DOTALL | re.IGNORECASE)
    if loc:
        inner = loc.group(1)
        arr = re.search(r"\[.*\]", inner, re.DOTALL)
        text = arr.group(0) if arr else inner
        try:
            import ast
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
        if isinstance(parsed, (list, tuple)):
            for el in parsed:
                if isinstance(el, (list, tuple)) and el:
                    items.append(el[0])          # (line_number, line_content) -> line_number
                else:
                    items.append(el)             # bare number
        if not items:
            # literal_eval failed (e.g. unescaped quotes): take each tuple's leading int,
            # else any integers in order.
            items = re.findall(r"\(\s*(\d+)", text) or re.findall(r"-?\d+", text)

    # Backward-compat: legacy <solution> / ```json``` list of "<line>:<content>".
    if not items:
        s2 = scope
        m = re.search(r"<solution>(.*?)(?:</solution>|\Z)", s2, re.DOTALL | re.IGNORECASE)
        if m:
            s2 = m.group(1)
        block = re.search(r"```(?:json)?\s*(.*?)```", s2, re.DOTALL)
        if block:
            s2 = block.group(1)
        arr = re.search(r"\[.*\]", s2, re.DOTALL)
        if arr:
            try:
                parsed = json.loads(arr.group(0))
                if isinstance(parsed, list):
                    items = parsed
            except (ValueError, TypeError):
                items = []
        if not items:
            items = re.findall(r"\d+\s*:[^\n,\]]*", s2) or re.findall(r"-?\d+", s2)

    clean_lines = (clean_code or "").split("\n")
    out: list[str] = []
    seen: set[int] = set()
    for it in items:
        s = str(it).strip().strip('"').strip("'")
        mm = re.match(r"\s*(\d+)\s*(?::\s*(.*))?$", s)
        if not mm:
            continue
        digits = mm.group(1)
        # A real 1-based line number is tiny; a huge digit run is junk the fallback regex
        # scraped from the response. Skip it (int() on a 4300+ digit string raises).
        if len(digits) > 7:
            continue
        ln = int(digits)
        if ln < 1 or ln in seen:
            continue
        # Authoritative content from the clean source (model content is only a cue).
        if clean_lines and ln <= len(clean_lines):
            content = clean_lines[ln - 1]
        elif clean_lines:
            continue  # out of range w.r.t. clean source
        else:
            content = (mm.group(2) or "").rstrip()
        seen.add(ln)
        out.append(f"{ln}:{content}")
    return out


def _predicted_line_numbers(prediction: list[str]) -> list[int]:
    nums: list[int] = []
    for p in prediction:
        mm = re.match(r"\s*(\d+)\s*:", p)
        if mm:
            nums.append(int(mm.group(1)))
    return nums


def _existing_task_ids(path: str | None) -> set[str]:
    ids: set[str] = set()
    if not path or not os.path.exists(path):
        return ids
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                ids.add(str(json.loads(line).get("task_id")))
            except json.JSONDecodeError:
                continue
    return ids


def append_localize_records(path: str, records: list[dict[str, Any]]) -> None:
    if not path or not records:
        return
    with _LOCALIZE_WRITE_LOCK:
        seen = _existing_task_ids(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for rec in records:
                tid = str(rec.get("task_id"))
                if tid in seen:
                    continue
                seen.add(tid)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_localize_prompts(path: str | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = rec.get("task_id")
            if tid is not None:
                out[str(tid)] = rec
    return out


def upsert_localize_prompts(path: str, records: list[dict[str, Any]]) -> None:
    if not path or not records:
        return
    with _LOCALIZE_WRITE_LOCK:
        merged = load_localize_prompts(path)
        for rec in records:
            tid = rec.get("task_id")
            if tid is not None:
                merged[str(tid)] = rec
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for rec in merged.values():
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp, path)


def localize_batch(
    samples: list[dict[str, Any]],
    sources: list[str | None],
    signals: list[dict[str, Any] | None],
    backend,
    opt,
    make_call: bool = True,
) -> list[dict[str, Any]]:
    path = getattr(opt, "localize_path", None)
    method = getattr(opt, "method", "naive")
    # In build-only mode we neither consult nor write localize.jsonl (the prompt is
    # saved into generation.jsonl instead); in call mode we dedup against it for resume.
    done = _existing_task_ids(path) if make_call else set()

    todo: list[int] = []
    chosen: set[str] = set()
    for i, sample in enumerate(samples):
        tid = common.task_id(sample)
        if not sources[i] or tid in done or tid in chosen:
            continue
        chosen.add(tid)
        todo.append(i)
    if not todo:
        return []

    annotated: dict[int, str | None] = {}
    violated: dict[int, list[alpha_beta.SpecReachSatisfy] | None] = {}
    rs_result: dict[int, dict | None] = {}
    text: dict[int, str] = {}
    clean: dict[int, str] = {}
    for i in todo:
        sig = signals[i] or {}
        ann = sig.get("annotated_code")
        annotated[i] = ann
        rs_result[i] = sig.get("reach_satisfy")
        rs_obj = _reach_satisfy_from_dict(rs_result[i])
        violated[i] = list(rs_obj.violated) if rs_obj is not None else None
        text[i] = sig.get("text", "") or ""
        # clean source = annotated source minus the `# assert` lines,
        # otherwise the candidate is already clean.
        clean[i] = extract_assertions_from_annotated_code(ann)[0] if ann else sources[i]

    msgs = [
        localize_messages(samples[i], clean[i], method=method, signal_text=text[i],
                          annotated_code=annotated[i], violated=violated[i],
                          reach_satisfy_result=rs_result[i])
        for i in todo
    ]
    # Announce a distinct step so the progress bar / per-task echo reads "localize"
    # instead of inheriting the previous "patch_0 generation" label (which makes the
    # localize pass look like a second repair run).
    if make_call:
        common.progress("purellm", f"localize on {len(todo)} task(s) via {method}")
        # Localize runs on the BASE model: a SpecCoder LoRA is a spec-generation verifier
        # only, so analysis/repair/localize all use the base model (no-op without LoRA).
        comps = common.chat_batched(backend, msgs, opt, num_samples=1,
                                    task_ids=[common.task_id(samples[i]) for i in todo],
                                    use_lora=False)
    else:
        comps = None  # build-only: prompts saved into generation.jsonl, no call

    records: list[dict[str, Any]] = []
    for slot, i in enumerate(todo):
        clean_code = clean[i]
        rec = common.base_record(samples[i], opt, common.buggy_code_of(samples[i]), NAME)
        rec.update({
            "signal_method": method,                                 # which repair signal drove this
            "signal_text": text[i],
            "number_source_clean": number_source(clean_code),
            "clean_code": clean_code,
        })
        if annotated[i]:
            rec["number_source_annotated"] = number_source(annotated[i])
            rec["annotated_code"] = annotated[i]
        if violated[i]:
            rec["num_violated_specs"] = len(violated[i])
            rec["violated_specs"] = [
                {
                    "expr": s.expr,
                    "annotated_checkpoint_line": s.checkpoint_line + 1,  # 1-based in annotated source
                    "reach": s.reach,
                    "satisfy": s.satisfy,
                    "satisfy_rate": round(s.satisfy_rate, 4),
                }
                for s in violated[i]
            ]
        # Full replayable localize messages (system + user). Localize runs once per task
        # and already skips tasks present in localize.jsonl, so this is the replay cache:
        # a rerun (or --localize-only) reuses it without re-running the pipeline.
        rec["prompt"] = msgs[slot][-1]["content"]
        rec["final_localize_prompt"] = msgs[slot]
        if make_call:
            resp = comps[slot][0] if comps[slot] else ""
            prediction = parse_prediction(resp, clean_code)
            rec.update({
                "prediction": prediction,                                # ["<clean_line>:<content>", ...]
                "predicted_lines": _predicted_line_numbers(prediction),  # clean 1-based ints, least->most certain
                "response": resp,
                "cost_token": common.attempt_cost(msgs[slot][-1]["content"], resp),
            })
        records.append(rec)

    if make_call and path:
        append_localize_records(path, records)
    return records


def localize_from_saved(
    samples: list[dict[str, Any]],
    saved_messages: list[list[dict[str, str]]],
    clean_codes: list[str],
    backend,
    opt,
) -> list[dict[str, Any]]:
    path = getattr(opt, "localize_path", None)
    method = getattr(opt, "method", "naive")
    done = _existing_task_ids(path)

    todo: list[int] = []
    chosen: set[str] = set()
    for i, sample in enumerate(samples):
        tid = common.task_id(sample)
        if not saved_messages[i] or tid in done or tid in chosen:
            continue
        chosen.add(tid)
        todo.append(i)
    if not todo:
        return []

    msgs = [saved_messages[i] for i in todo]
    common.progress("purellm", f"localize-only replay on {len(todo)} task(s) via {method}")
    comps = common.chat_batched(backend, msgs, opt, num_samples=1,
                                task_ids=[common.task_id(samples[i]) for i in todo],
                                use_lora=False)

    records: list[dict[str, Any]] = []
    for slot, i in enumerate(todo):
        resp = comps[slot][0] if comps[slot] else ""
        clean_code = clean_codes[i] or ""
        prediction = parse_prediction(resp, clean_code)
        rec = common.base_record(samples[i], opt, common.buggy_code_of(samples[i]), NAME)
        rec.update({
            "signal_method": method,
            "number_source_clean": number_source(clean_code),
            "clean_code": clean_code,
            "prediction": prediction,
            "predicted_lines": _predicted_line_numbers(prediction),
            "prompt": msgs[slot][-1]["content"],
            "response": resp,
            "final_localize_prompt": msgs[slot],
            "cost_token": common.attempt_cost(msgs[slot][-1]["content"], resp),
        })
        records.append(rec)

    if path:
        append_localize_records(path, records)
    return records
