from __future__ import annotations

import hoareprompt as hp

DEFAULT_CONFIG = {
    "model": "__backend__",
    "temperature": 0.2,
    "assessment-mode": "postcondition-entailment",
    "postcondition-mode": "hoarecot",
    "postcondition-cot-prompt": "node-based-style",
    "loop-unrolling-count": 0,
    "entailment-mode": "naive",
    "cex-mode": "with-postcondition",
    "annotated": True,
    "annotated-type": "simple",
    "concat_simple": True,
}

REPAIR_PROMPT = """
Your task is to repair a given Python program so that it becomes correct based on the problem description and the execution states of the program provided as comments. Assume valid inputs as described in the problem description.

First explain your reasoning, then provide the corrected program.

# Problem:
{description}

# Program to repair:
```python
{buggy_code}
```

# Annotated Program (the same program with execution states as comments):
{code}

# Your response:
Reasoning:
<solution>
```python
# repaired program
```
</solution>

"""


def annotate_program(description: str, program: str, config: dict, log_dir):
    """Run HoarePrompt's unchanged annotation core and return
    ``(annotated_program, postcondition)``.

    Bare stdin/stdout scripts (no ``def``) are wrapped into a dummy function by
    ``extract_functions`` (``global_code == 'GLOBAL'``), so codeforces programs
    are annotated too.
    """
    cleaned, imports = hp.remove_imports_and_comments(program)
    fdict = hp.extract_functions(cleaned)
    functions_list = fdict["functions"]
    is_global = fdict["global_code"] == "GLOBAL"

    annotated_parts: list[str] = []
    postconditions: list[str] = []
    for func in functions_list:
        precondition = hp.extract_precondition(description, func, config, log_dir, len(functions_list))
        result = hp.compute_postcondition(precondition, func, config, log_dir)
        if isinstance(result, (list, tuple)) and len(result) == 3:
            postconditions.append(str(result[0]))
            annotated_parts.append(str(result[2]))
        else:
            postconditions.append(str(result))

    annotated = "\n\n".join(annotated_parts) if annotated_parts else cleaned
    if not is_global and imports:
        annotated = imports + "\n\n" + annotated
    return annotated, "\n\n".join(postconditions)


def hoareprompt_repair(description: str, program: str, config: dict | None, log_dir):
    """Annotate, then make the (repurposed) final LLM call a repair request.

    Returns ``(raw_repair_response, annotated_program, postcondition)``.
    """
    config = config or DEFAULT_CONFIG
    annotated, postcondition = annotate_program(description, program, config, log_dir)
    model = hp.get_model(config["model"], config["temperature"], log_dir)
    code = hp.remove_functionality(annotated)
    prompt = REPAIR_PROMPT.format(description=description, code=code, buggy_code=program)
    response = model.query(prompt)
    return response, annotated, postcondition
