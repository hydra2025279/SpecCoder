
from __future__ import annotations

from typing import Any

from SpecCoder.src.experiment.programrepair import common

NAME = "cot"

COT_INSTRUCTION = (
    "First, reason step by step: identify what the buggy program does wrong versus the "
    "problem specification and the examples, then describe the minimal fix. After your "
    "reasoning, output the corrected program."
)


def build_prompt(sample: dict[str, Any], buggy_code: str) -> str:
    base = common.build_repair_prompt(sample, buggy_code)
    # Insert the CoT instruction ahead of the output-schema section.
    marker = "### Required Output"
    head, _, tail = base.partition(marker)
    return f"{head}{COT_INSTRUCTION}\n\n{marker}{tail}"


def run_method(rows: list[dict[str, Any]], backend, opt) -> list[dict[str, Any]]:
    buggy = [common.buggy_code_of(s) for s in rows]
    messages = [
        [{"role": "system", "content": common.SYSTEM_PROMPT},
         {"role": "user", "content": build_prompt(s, b)}]
        for s, b in zip(rows, buggy)
    ]
    num_samples = max(1, int(getattr(opt, "num_samples", 1) or 1))
    completions, call_metadata = common.timed_chat_batched(backend, messages, opt, num_samples=num_samples)

    records: list[dict[str, Any]] = []
    for sample, buggy_code, msg, samples, meta in zip(rows, buggy, messages, completions, call_metadata):
        responses = list(samples or [])
        for attempt, response in enumerate(responses):
            rec = common.base_record(sample, opt, buggy_code, NAME)
            rec.update({
                "attempt": attempt,
                "code": common.extract_code(response) or buggy_code,
                "rounds": 1,
                "prompt": msg[-1]["content"],
                "response": response,
                "time_latency": meta,
                "cost_token": common.attempt_cost(msg[-1]["content"], response),
            })
            records.append(rec)
    return records
