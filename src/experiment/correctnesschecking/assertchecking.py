from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from SpecCoder.src.experiment.gen_utils import (
    append_jsonl,
    load_latest_jsonl_by_task,
    task_log_name,
    token_usage_record,
)
from SpecCoder.src.experiment.programrepair import alpha_beta
from SpecCoder.src.experiment.programrepair.assertrepair import (
    SPEC_SYSTEM,
    _annotate_with_reach_satisfy,
    format_checkpoint_analysis,
    format_specs,
)
from SpecCoder.src.experiment.specgen.inline.prompt import build_spec_generation_prompt
from SpecCoder.src.experiment.correctnesschecking.prompt import SYSTEM_PROMPT
from SpecCoder.trace_collection import _is_valid_eval_expr

NAME = "assertchecking"

JUDGE_SYSTEM = (
    "You are an experienced program-correctness reviewer. Given a program and behavioural "
    "checkpoints (inline `# assert` with execution evidence over the test cases — how many "
    "executions reach each checkpoint and how many satisfy it), you DECIDE whether the program "
    "is correct or has a bug. You do NOT modify the code."
)


def _judge_use_lora(opt) -> bool:
    """Resolve ``--judge-finetune`` to a ``use_lora`` flag for the verdict call:
    ``base`` -> False, ``finetune`` -> True, ``auto`` (default) -> follow
    ``--speccoder_finetune`` (LoRA judge when finetuning, base otherwise). ``use_lora``
    is a no-op when no adapter is loaded, so ``finetune`` without ``--speccoder_finetune``
    still runs base."""
    mode = getattr(opt, "judge_finetune", "auto")
    if mode == "base":
        return False
    if mode == "finetune":
        return True
    return bool(getattr(opt, "speccoder_finetune", False))  # auto


def _cleaned_rs(rs):
    """A view of the reach/satisfy result with syntax-invalid checkpoints dropped, so the
    judge prompt's analysis never classifies a natural-language `# assert` as VIOLATED."""
    if rs is None:
        return None
    specs = [s for s in rs.specs if _is_valid_eval_expr(s.expr)]
    return alpha_beta.ReachSatisfyResult(
        specs=specs, violated=_real_violations(rs), num_tests=rs.num_tests)


def _real_violations(rs) -> list:
    """Violated checkpoints that are GENUINE bugs, excluding specs whose expression is not
    valid Python. An unparseable expr (natural-language `# assert`) is instrumented as
    `lambda: False`, so it is reached but never satisfied -> spuriously [VIOLATED]. We treat
    such specs as satisfied (ignored), so they never push a verdict toward 'buggy'."""
    if rs is None:
        return []
    return [s for s in rs.violated if _is_valid_eval_expr(s.expr)]


def _format_reach_satisfy_report(rs) -> str:
    """Local report (analog of assertrepair's, but honest about unparseable specs): a spec
    whose expr is not valid Python is reported as IGNORED/treated-satisfied rather than
    [VIOLATED], matching the verdict logic."""
    if rs is None or not getattr(rs, "specs", None):
        return "(no specifications evaluated)"
    out = [f"Computed by executing the program on all {rs.num_tests} test case(s)."]
    for s in rs.specs:
        out.append(f"# assert {s.expr}")
        if not _is_valid_eval_expr(s.expr):
            tag = "  [unparseable expr -> ignored, treated satisfied]"
        elif s.is_violated:
            tag = "  [VIOLATED]"
        elif s.reach == 0:
            tag = "  [never reached]"
        else:
            tag = "  [always satisfied]"
        out.append(f"  reached by {s.reach} execution(s), "
                   f"{s.satisfy} satisfy / {s.reach - s.satisfy} do not" + tag)
    return "\n".join(out)


def _resolve_test_inputs(batch, backend, opt, log_dir: str):
    """Test INPUTS feeding the reach/satisfy execution: the dataset's real public
    ``test_cases``. Returns ``(tests_per_task, source_per_task)`` aligned to ``batch``."""
    return [list(t.get("test_cases") or []) for t in batch], ["oracle"] * len(batch)


def build_judge_prompt(task: dict[str, Any], annotated_with_evidence: str,
                       violated: list, rs_result=None) -> str:
    """Correctness-judge prompt (assertrepair's analysis prompt, re-aimed at a YES/NO verdict
    instead of analysis+repair). Shows the program annotated with inline reach/satisfy evidence,
    the violated checkpoints, and the per-checkpoint execution analysis; asks for a verdict."""
    analysis = format_checkpoint_analysis(rs_result, violated)
    analysis_block = (
        f"\n### Checkpoint execution analysis (how many executions reach and satisfy each `# assert`)\n{analysis}\n"
        if analysis else ""
    )
    specs_block = (
        f"### Violated behavioural checkpoints\n{format_specs(violated)}\n"
        if violated else ""
    )
    return f"""### Role
You decide whether a Python program is CORRECT or BUGGY for the problem below — you do NOT write code.
You are given *behavioural specifications*: inline `# assert <expr>` checkpoints meant to capture
intended program state at key points, plus an analysis of how many test executions reach and
satisfy each checkpoint. A checkpoint that is reached but NOT satisfied by every execution is
evidence the intended state is broken — i.e. the program is likely buggy. Treat the checkpoints
as HYPOTHESES that may be wrong — verify them against the problem semantics before relying on them.

{task.get("problem_text", "")}

### Program (annotated with `# assert` checkpoints)
```python
{annotated_with_evidence}
```

{specs_block}{analysis_block}
### How to respond
First reason about whether the program is correct, grounding your reasoning in the problem
description and the checkpoint execution evidence above. Then conclude with exactly one verdict
line: `<answer>YES</answer>` if the program is correct, or `<answer>NO</answer>` if it has a bug."""


def _write_task_log(log_dir: str, task_id: str, *, spec_prompt: str, spec_resp: str,
                    rs, num_specs: int, num_violated: int, judge_prompt: str,
                    judge_resp: str, verdict: bool, source: str) -> None:
    """ONE log file per task, in pipeline order: (1) spec-generation prompt+response,
    (2) per-spec execution evidence (how many executions reach/satisfy each `# assert`
    and that spec's VIOLATED/always-satisfied/never-reached verdict), (3) the LLM
    correctness-verdict prompt+response, (4) the final correct/buggy verdict."""
    sep = "═" * 72
    report = _format_reach_satisfy_report(rs) if rs is not None else "(no specs generated / spec eval error)"
    src = source
    parts = [
        "=" * 72, f"TASK: {task_id}   method=assertchecking", "=" * 72, "",
        f"╔{sep}╗", "║  [1] SPEC GENERATION — PROMPT", f"╚{sep}╝", spec_prompt, "",
        "[1] SPEC GENERATION — RESPONSE", spec_resp, "",
        f"╔{sep}╗", "║  [2] SPEC EXECUTION — per-checkpoint reach/satisfy + verdict", f"╚{sep}╝",
        report, "",
        f"  => {num_violated} VIOLATED checkpoint(s) out of {num_specs} spec(s).", "",
        f"╔{sep}╗", "║  [3] CORRECTNESS VERDICT (LLM) — PROMPT", f"╚{sep}╝", judge_prompt, "",
        "[3] CORRECTNESS VERDICT (LLM) — RESPONSE", judge_resp, "",
        f"╔{sep}╗", f"║  [4] FINAL VERDICT: {'CORRECT' if verdict else 'BUGGY'}   (source: {src})",
        f"╚{sep}╝", "",
    ]
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    (Path(log_dir) / f"{task_log_name(task_id)}.log").write_text("\n".join(parts), encoding="utf-8")


def run(tasks: list[dict[str, Any]], backend, opt, generation_path: str, log_dir: str) -> list[dict[str, Any]]:
    # Lazy import to avoid a circular import (run.py imports this module at top level).
    from SpecCoder.src.experiment.correctnesschecking.run import parse_yes_no

    existing = load_latest_jsonl_by_task(generation_path)
    pending = [t for t in tasks if t["task_id"] not in existing]
    eval_workers = int(getattr(opt, "eval_workers", 16) or 16)
    batch_size = max(1, int(getattr(opt, "global_batch", 0) or 1))

    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        tids = [t["task_id"] for t in batch]

        tests_for_batch, test_srcs = _resolve_test_inputs(batch, backend, opt, log_dir)

        spec_msgs = [[
            {"role": "system", "content": SPEC_SYSTEM},
            {"role": "user", "content": build_spec_generation_prompt(
                {"problem_text": t.get("problem_text", "")}, t.get("code", ""))},
        ] for t in batch]
        spec_comps = backend.chat(spec_msgs, num_samples=1, task_ids=tids)  # use_lora defaults True

        annotated_src: list[str] = []
        rs_list: list[Any] = []
        viols: list[list] = []
        for idx, (t, comp) in enumerate(zip(batch, spec_comps)):
            resp = comp[0] if comp else ""
            annotated = alpha_beta.extract_annotated(resp) or ""
            rs = None
            violated: list = []
            if annotated and "# assert" in annotated:
                try:
                    rs = alpha_beta.compute_reach_satisfy(
                        annotated, tests_for_batch[idx] or [],
                        memory_mb=None, parallelism=eval_workers)
                    violated = _real_violations(rs)
                except Exception as exc:  
                    sys.stderr.write(f"      ! spec eval failed for {t['task_id']}: "
                                     f"{type(exc).__name__}: {exc}\n")
                    rs, violated = None, []
            annotated_src.append(annotated)
            rs_list.append(rs)
            viols.append(violated)

        judge_use_lora = _judge_use_lora(opt)
        rs_clean = [_cleaned_rs(rs_list[i]) for i in range(len(batch))]
        evidence = [
            _annotate_with_reach_satisfy(annotated_src[i], rs_clean[i])
            if (annotated_src[i] and rs_clean[i] is not None)
            else (annotated_src[i] or batch[i].get("code", ""))
            for i in range(len(batch))
        ]
        judge_msgs = [[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": build_judge_prompt(batch[i], evidence[i], viols[i], rs_clean[i])},
        ] for i in range(len(batch))]
        judge_comps = backend.chat(judge_msgs, num_samples=1, task_ids=tids, use_lora=judge_use_lora)

        for i, task in enumerate(batch):
            spec_prompt = spec_msgs[i][-1]["content"]
            spec_resp = spec_comps[i][0] if spec_comps[i] else ""
            judge_prompt = judge_msgs[i][-1]["content"]
            judge_resp = judge_comps[i][0] if judge_comps[i] else ""
            rs = rs_list[i]
            num_specs = len(rs.specs) if rs is not None else 0

            parsed = parse_yes_no(judge_resp)
            if parsed is not None:
                yn = parsed
                verdict_source = "LLM <answer>"
            else:
                yn = False
                verdict_source = "LLM verdict unparseable -> NO"

            _write_task_log(
                log_dir, task["task_id"],
                spec_prompt=spec_prompt, spec_resp=spec_resp, rs=rs,
                num_specs=num_specs, num_violated=len(viols[i]),
                judge_prompt=judge_prompt, judge_resp=judge_resp,
                verdict=bool(yn), source=verdict_source)
            rec = {
                **{k: task.get(k) for k in ("task_id", "source_task_id", "candidate_kind", "label")},
                "ground_truth": task.get("label"),  # bool: True = correct, False = buggy
                "benchmark": "HumanChecking",
                "method": NAME,
                "source_test": getattr(opt, "source_test", "oracle"),
                "test_source": test_srcs[i],
                "num_test_inputs": len(tests_for_batch[i] or []),
                "model": opt.model,
                "backend": opt.backend,
                "speccoder_finetune": bool(opt.speccoder_finetune),
                "judge_finetune": getattr(opt, "judge_finetune", "auto"),
                "judge_use_lora": judge_use_lora,
                "prompt": spec_prompt,
                "response": spec_resp,
                "code": task.get("code", ""),
                "annotated_code": annotated_src[i] or None,
                "num_specs": num_specs,
                "num_violated_specs": len(viols[i]),
                "reach_satisfy": rs.to_dict() if rs is not None else None,
                "verify_response": judge_resp or None,
                "predicted_correct": bool(yn),
                "test_cases": task.get("test_cases") if getattr(opt, "include_test_cases", False) else None,
                "time_latency": {"generation_seconds": 0.0, "global_batch": len(batch), "local_batch": 1},
                "cost_token": token_usage_record(spec_prompt + "\n" + judge_prompt, spec_resp + "\n" + judge_resp),
            }
            append_jsonl(generation_path, {k: v for k, v in rec.items() if v is not None})
            print(f"      [assertchecking] {start + i + 1:>4}/{len(pending)}  {task['task_id']:<14} "
                  f"-> {'CORRECT' if yn else 'buggy'}  ({len(viols[i])} violated / {num_specs} spec)",
                  flush=True)

    latest = load_latest_jsonl_by_task(generation_path)
    return [latest[t["task_id"]] for t in tasks if t["task_id"] in latest]
