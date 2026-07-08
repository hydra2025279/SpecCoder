from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

_AGENTS_DIR = Path(__file__).resolve().parent / "agents"
if str(_AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENTS_DIR))

from agent_reviewer import (  # noqa: E402
    Review,
    ReviewDecision,
)


PATCH_REVIEW_SYSTEM = (
    "You are an experienced program-repair reviewer. Decide only whether the "
    "candidate patch aligns with the problem statement and the intended behavioural "
    "specification. The reproducer and execution outputs are evidence; do not judge "
    "or revise the test itself."
)


def review_messages(
    problem: str,
    spec: str,
    test: str,
    patch: str,
    patched_stdout: str,
    patched_stderr: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": PATCH_REVIEW_SYSTEM},
        {"role": "user", "content": (
            f"# Problem\n{problem}\n\n"
            f"# Intended behavioural specification\n{spec}\n\n"
            f"# Reproducer evidence\n{test}\n\n"
            f"# Candidate patch\n```python\n{patch}\n```\n\n"
            "## Candidate execution on the reproducer\n"
            f"stdout:\n{patched_stdout}\n\nstderr:\n{patched_stderr}\n\n"
            "Evaluate only whether the candidate patch correctly implements the problem "
            "and aligns with the intended behavioural specification. Use the reproducer "
            "execution only as supporting evidence. Return exactly one JSON object:\n"
            '{"patch-correct": "yes or no", "patch-analysis": "detailed reasoning", '
            '"patch-advice": "how to fix the patch when incorrect, otherwise empty"}'
        )},
    ]


def _first_json_object(content: str) -> str | None:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fenced:
        return fenced.group(1)
    depth = 0
    start = -1
    for i, ch in enumerate(content):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                return content[start : i + 1]
    return None


def extract_review_result(content: str) -> Review | None:
    """Parse the experiment's patch-only reviewer response.

    ``Review`` is retained as an internal compatibility type, but test fields are
    neutral placeholders and are never exposed by ``review_to_json``.
    """
    blob = _first_json_object(content)
    if blob is None:
        return None
    try:
        data = json.loads(blob)
        patch_decision = ReviewDecision(str(data["patch-correct"]).strip().lower())
    except Exception:
        return None
    return Review(
        patch_decision=patch_decision,
        patch_analysis=str(data.get("patch-analysis", "")),
        patch_advice=str(data.get("patch-advice", "")),
        test_decision=ReviewDecision.YES,
        test_analysis="",
        test_advice="",
    )


def review_to_json(review: Review) -> dict[str, str]:
    return {
        "patch-correct": review.patch_decision.value,
        "patch-analysis": review.patch_analysis,
        "patch-advice": review.patch_advice,
    }


REPRODUCER_SYSTEM = "You are an experienced software engineer responsible for reproducing given issues."


def reproducer_prompt(problem: str, code: str, last_failed: dict[str, Any] | None = None) -> str:
    retry = ""
    if last_failed is not None:
        retry = (
            "\nYour previous test did NOT expose the bug (the buggy submission produced the "
            "expected output you gave, so it is useless). Write a different, stronger test whose "
            "input makes the buggy submission produce a WRONG answer.\n"
            f"Previous useless test input:\n{last_failed.get('input', '')}\n"
        )
    return (
        "The following competitive-programming submission is buggy. Write ONE standalone test case "
        "that exposes the bug: an input fed to stdin and the correct expected stdout for that input. "
        "The buggy submission must produce a WRONG answer on this input.\n"
        f"{retry}\n"
        f"# Problem\n{problem}\n\n"
        f"# Buggy submission\n```python\n{code}\n```\n\n"
        'Return exactly one JSON object and nothing else:\n'
        '{"input": "<stdin text>", "expected": "<correct stdout text>"}'
    )


def parse_test_case(text: str) -> dict[str, str] | None:
    blob = _first_json_object(text)
    if blob is None:
        return None
    try:
        data = json.loads(blob)
    except Exception:
        return None
    if "input" not in data or "expected" not in data:
        return None
    return {"input": str(data["input"]), "expected": str(data["expected"])}


SPEC_SYSTEM = "You infer precise, implementation-agnostic behavioural specifications."


def spec_prompt(problem: str, code: str, test: dict[str, str] | None) -> str:
    evidence = ""
    if test is not None:
        evidence = (
            "A failing test that exposes the bug:\n"
            f"input:\n{test.get('input', '')}\n"
            f"correct expected output:\n{test.get('expected', '')}\n\n"
        )
    return (
        "Describe the intended behaviour of a correct solution to the problem, focusing on the "
        "input parsing, the exact output format, key invariants, and the edge cases where the "
        "buggy submission diverges. Use short bullet points. Do not write code.\n\n"
        f"# Problem\n{problem}\n\n"
        f"# Buggy submission\n```python\n{code}\n```\n\n"
        f"{evidence}"
    )


def compose_patch_feedback(spec: str, review: Review | None, test: dict[str, str] | None) -> str:
    parts = [f"### Specification\n{spec.strip()}"]
    if test is not None:
        parts.append(
            "### Reproduction test\n"
            f"input:\n{test.get('input', '')}\n"
            f"correct expected output:\n{test.get('expected', '')}"
        )
    if review is not None:
        if review.patch_analysis.strip():
            parts.append(f"### Reviewer analysis\n{review.patch_analysis.strip()}")
        if review.patch_advice.strip():
            parts.append(f"### Reviewer advice\n{review.patch_advice.strip()}")
    return "\n\n".join(parts)
