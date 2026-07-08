from __future__ import annotations

SYSTEM_PROMPT = "You are a helpful assistant."


def _problem_block(problem_description: str, input_specification: str, output_specification: str) -> str:
    """Combine the codeforces spec blocks into one problem section."""
    parts = [str(problem_description or "").strip()]
    if str(input_specification or "").strip():
        parts.append(f"### Input Specification\n{str(input_specification).strip()}")
    if str(output_specification or "").strip():
        parts.append(f"### Output Specification\n{str(output_specification).strip()}")
    return "\n\n".join(p for p in parts if p)


# =========================================================================== #
# SPEC GENERATION — identical to training_prompt.SPEC_PROMPT
# =========================================================================== #
SPEC_PROMPT = """\
### Role

- You are writing symbolic inline specifications for a Python program as `# assert <python_boolean_expression>` comment lines, at important intermediate steps of its algorithm.
- The specifications must follow the PROBLEM DESCRIPTION and the intended behavior of the algorithm, NOT the current code — which may be correct or buggy. Do not merely restate what the code happens to do.
- Specifications grounded in the intended behavior hold on every correct execution and are violated by buggy variants, which makes them useful bug detectors.

### Where to place in-line specifications

- Prefer checkpoints after critical state updates, such as assignments that establish or update accumulators, dynamic-programming tables, pointers, indices, counters, heaps, sets, maps, or loop-carried state.
- Prefer checkpoints at branch joins, after an `if`/`else` block establishes a combined state relation.
- Prefer loop-level checkpoints after meaningful loop-body progress and at loop exits. Avoid annotating every minor statement inside a loop.
- Prefer checkpoints before printing the answer, relating the printed value to the parsed inputs and accumulated state.
- Avoid trivial lines, imports, pure boilerplate, obvious temporary assignments, and dense line-by-line annotations.

### Inline Specification Rules

- Write each assertion as `# assert <python_expression>` on its own line, indented to match the surrounding block.
- Each `<python_expression>` is a Python boolean expression that expresses the strongest, tightest relationship at that point.
- Each postcondition must depict the relationship between FINAL STATES (of variables after the current block) and INITIAL PARAMETERS (from the function signature).
- The expression may reference local variables in scope.
- Each assertion must express a non-trivial relationship between values.

### PROBLEM DESCRIPTION
{problem}

### Program
```python
{program}
```

### Output
Output one ```python block: the same executable program above, unchanged, with useful inline `# assert <python_boolean_expression>` comments added at important locations.
"""


def build_speccoder_prompt(
    problem_description: str,
    input_specification: str,
    output_specification: str,
    code: str,
) -> str:
    return SPEC_PROMPT.format(
        problem=_problem_block(problem_description, input_specification, output_specification),
        program=code,
    )


VANILLA_CORRECTNESS_PROMPT = """\
### Role
You decide whether a candidate Python program is a correct solution to the problem.

### What you are given
- The problem statement (below).
- The candidate program to judge — it may be correct or buggy.

### Problem
{problem}

### Candidate program
```python
{candidate}
```

### Output
Decide whether the candidate is correct, and output exactly one line: `<answer>YES</answer>` if correct, or `<answer>NO</answer>` if incorrect.
"""


COT_CORRECTNESS_PROMPT = """
### Role
You decide whether a candidate Python program is a correct solution to the problem.

### What you are given
- The problem statement (below).
- The candidate program to judge — it may be correct or buggy.

### How to judge
- Reason about the candidate code itself.
- Analyze edge cases and logic carefully to find any issue that would make the program wrong on a valid input.

### Problem
{problem}

### Candidate program
```python
{candidate}
```

### Output
First reason about whether the candidate is correct, then conclude with exactly one verdict line: `<answer>YES</answer>` if correct, or `<answer>NO</answer>` if incorrect.
"""

ASSERT_CORRECTNESS_PROMPT = """
### Role
You decide whether a candidate Python program is a correct solution to the problem.
You are guided by *behavioural specifications*: inline `# assert <expr>` checkpoints in the program meant to capture intended program state at key points.

### What you are given
- The problem statement (below).
- The candidate program to judge — it may be correct or buggy, and it already carries inline `# assert` specifications.
- Those specifications may point at a bug, but are not guaranteed to be correct or relevant.

### How to judge
Reason in the following stages:
- Problem semantics: mention the desired behaviour a correct solution must produce.
- Main algorithm: describe the algorithm the candidate code uses to achieve that behaviour.
- Edge cases: analyse boundaries, degenerate inputs, ties, duplicates and large values.
- Specification verification: understand each inline `# assert` specification and verify whether the implementation violates it AND whether it violates the whole problem semantics.

### Problem
{problem}

### Candidate program
```python
{candidate}
```
{evidence}
### Output
First reason about whether the candidate is correct, then conclude with exactly one verdict line: `<answer>YES</answer>` if correct, or `<answer>NO</answer>` if incorrect.
"""


def build_vanilla_prompt(
    problem_description: str,
    input_specification: str,
    output_specification: str,
    code: str,
) -> str:
    """VANILLA method: a single direct YES/NO judgement on the problem + raw
    candidate. No reasoning instruction, no specs, no execution evidence."""
    return VANILLA_CORRECTNESS_PROMPT.format(
        problem=_problem_block(problem_description, input_specification, output_specification),
        candidate=code,
    )


def build_assert_prompt(
    problem_description: str,
    input_specification: str,
    output_specification: str,
    code: str,
    execution_summary: str | None = None,
) -> str:

    evidence = ""
    if execution_summary is not None and str(execution_summary).strip():
        evidence = (
            "\n### Execution result of the inline assertions (a hint — they may be imperfect)\n"
            f"{str(execution_summary).strip()}\n"
        )
    return ASSERT_CORRECTNESS_PROMPT.format(
        problem=_problem_block(problem_description, input_specification, output_specification),
        candidate=code,
        evidence=evidence,
    )


def build_cot_prompt(
    problem_description: str,
    input_specification: str,
    output_specification: str,
    code: str,
) -> str:
    """COT method: ask the LLM to reason about the candidate first, then emit the
    YES/NO verdict last. No execution evidence — pure static reasoning."""
    return COT_CORRECTNESS_PROMPT.format(
        problem=_problem_block(problem_description, input_specification, output_specification),
        candidate=code,
    )


# =========================================================================== #
# INPUT GENERATION — same Role-structured style as the other prompts
# =========================================================================== #
INPUT_GENERATION_PROMPT = """\
### Role

- You are a test-input designer for a competitive-programming problem that reads from standard input.
- Your goal is to produce diverse standard-input cases that maximize coverage of the algorithm's behavior, so that a buggy solution is likely to fail on at least one of them.

### What to cover

- Minimal / smallest inputs and maximal / boundary sizes.
- Empty or degenerate structures, ties, duplicates, already-sorted / reverse-sorted data.
- Negative, zero, and large values; and any other corner cases implied by the specification.
- Every input MUST satisfy the Input Specification — never produce invalid inputs.

### Problem Description
{problem}

### Input Specification
{input_specification}

### Output
Output up to {n} DISTINCT test inputs, each in its OWN fenced block, exactly as it would be fed to stdin (raw text, including newlines):
```
<stdin for case 1>
```
```
<stdin for case 2>
```
Output nothing except the fenced blocks.
"""


def build_input_generation_prompt(
    problem_description: str,
    input_specification: str,
    n: int = 50,
) -> str:
    return INPUT_GENERATION_PROMPT.format(
        problem=problem_description, input_specification=input_specification, n=n)
