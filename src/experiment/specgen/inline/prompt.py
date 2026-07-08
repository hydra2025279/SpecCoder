from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = "You are a helpful coding assistant"

def text_block(title: str, content: Any) -> str:
    return f"### {title}\n{str(content or '').strip()}\n"


def build_problem_section(sample: dict[str, Any]) -> str:
    if sample.get("problem_text"):
        return str(sample.get("problem_text") or "").strip()
    # Canonical problem text: Problem Description
    # + Input/Output Specification + worked Examples (demo-input/demo-output).
    from SpecCoder.spec_utils import build_problem_text
    return build_problem_text(sample, include_examples=True)

SPEC_GENERATION_PROMPT = """\
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
- **Each `# assert` is UNCOMMENTED and RUN inline with the program**, sharing its stdin
  and state, so `<python_expression>` MUST be a PURE read-only boolean. NEVER call
  `input`, `sys.stdin`, `print`, `open`, `eval`, or `exec` — e.g. an `input()` inside an
  assert would eat the program's stdin and crash everything after it.

### PROBLEM DESCRIPTION
{problem_section}

### Program
```python
{completion}
```

### Answer
Output one ```python block: the same executable program above, unchanged, with useful inline `# assert <python_boolean_expression>` comments added at important locations.
"""

def build_spec_generation_prompt(sample: dict[str, Any], completion: str) -> str:
    return SPEC_GENERATION_PROMPT.format(
        problem_section=build_problem_section(sample),
        completion=completion,
    )
