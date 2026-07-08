
from __future__ import annotations

import argparse
import fcntl
import json
import re
from pathlib import Path
from typing import Any

_THIS = Path(__file__).resolve()
PROJECT_ROOT = next(
    (p for p in _THIS.parents if (p / "SpecCoder").is_dir()), _THIS.parents[3]
)

# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------
_PY_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_GENERIC_BLOCK = re.compile(r"```\s*\n(.*?)```", re.DOTALL)
_SOLUTION = re.compile(r"<solution>(.*?)(?:</solution>|\Z)", re.DOTALL | re.IGNORECASE)


def python_blocks(text: str) -> list[str]:
    """All fenced python blocks (falls back to generic ``` blocks)."""
    blocks = _PY_BLOCK.findall(text)
    if not blocks:
        blocks = _GENERIC_BLOCK.findall(text)
    return [b.strip("\n") for b in blocks]


def code_block_score(code: str) -> int:
    score = 0
    if re.search(r"if\s+__name__\s*==\s*['\"]__main__['\"]", code):
        score += 4
    if re.search(r"^\s*(?:import\s+sys|from\s+sys\s+import\b)", code, re.MULTILINE):
        score += 2
    if re.search(r"^\s*def\s+\w+\s*\(", code, re.MULTILINE):
        score += 2
    if re.search(r"\b(?:input|print)\s*\(", code) or "sys.stdin" in code or "sys.stdout" in code:
        score += 1
    return score


def select_code_block(blocks: list[str]) -> str:
    if not blocks:
        return ""
    # Prefer the LAST fenced block (the model's final answer) — on a score tie the
    # larger index wins; with no script signals at all, fall back to the last block.
    scored = [(code_block_score(block), idx, block) for idx, block in enumerate(blocks)]
    best_score, _idx, best_block = max(scored, key=lambda item: (item[0], item[1]))
    return best_block if best_score > 0 else blocks[-1]


def extract_code(response: str) -> str:
    """Runnable Python from a response.

    Prefer fenced blocks inside ``<solution>`` and select the block that looks
    most like an executable stdin/stdout script. On ties (or when no block has
    script signals) prefer the LAST fenced block, i.e. the model's final answer.
    """
    match = _SOLUTION.search(response)
    scope = match.group(1) if match else response
    blocks = python_blocks(scope)
    return select_code_block(blocks) if blocks else scope.strip()


def extract_two_blocks(response: str) -> tuple[str, str]:
    """For sequential two-block output: (Block 1 clean code, Block 2 annotated).
    Falls back gracefully when only one block is present."""
    blocks = python_blocks(response)
    if not blocks:
        return "", ""
    if len(blocks) == 1:
        return blocks[0], blocks[0]
    return blocks[0], blocks[-1]


def strip_assert_comments(code: str) -> str:
    """Remove standalone ``# assert ...`` and ``# Postcondition: ...`` comment
    lines, leaving the executable program (used to build the post-hoc input)."""
    out = []
    for line in code.splitlines():
        s = line.strip()
        if s.startswith("# assert ") or s.startswith("# Postcondition:"):
            continue
        out.append(line)
    return "\n".join(out)


def estimate_token_count(text: str) -> int:
    """Best-effort token estimate that works even when tokenizer deps are absent."""
    if not text:
        return 0
    try:
        import tiktoken

        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text, disallowed_special=()))
    except Exception:
        return max(1, len(str(text)) // 4)


def token_usage_record(prompt: str, response: str) -> dict[str, int]:
    prompt_tokens = estimate_token_count(prompt)
    response_tokens = estimate_token_count(response)
    return {
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens": prompt_tokens + response_tokens,
    }


def sum_token_usage(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "prompt_tokens": sum(int(r.get("prompt_tokens", 0) or 0) for r in records),
        "response_tokens": sum(int(r.get("response_tokens", 0) or 0) for r in records),
        "total_tokens": sum(int(r.get("total_tokens", 0) or 0) for r in records),
    }


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def model_slug(model: str) -> str:
    """Filesystem-safe slug for a model id (used in output/<slug>/<kind>/...)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(model)).strip("_") or "model"


# Standard filenames inside each method folder.
RESULTS_FILE = "result.jsonl"
LOG_FILE = "generation.log"
GENERATION_FILE = "generation.jsonl"
RESULT_FILE = "result.jsonl"
SUMMARY_FILE = "summary.json"


def experiment_output_dir(kind: str, model: str, method: str) -> Path:
    """``SpecCoder/output/<model_slug>/<kind>/<method>/``.

    Layout is model-first: each model gets one tree, split by ``kind`` and then
    a per-method/baseline leaf folder holding ``result.jsonl`` + logs. Output is
    both written here and read back from here."""
    out = PROJECT_ROOT / "SpecCoder" / "output" / model_slug(model) / kind / method
    out.mkdir(parents=True, exist_ok=True)
    return out


def spec_gen_output_dir(
    branch: str,
    model: str,
    *,
    mode: str | None = None,
    speccoder_finetune: bool = False,
    make: bool = True,
) -> Path:
    """``SpecCoder/output/spec_gen/<branch>/<model>/[<mode>/]<leaf>/``.

    The active spec-generation pipeline is inline oracle generation over
    HumanExec correct solutions. ``leaf`` is ``speccoder`` when finetuned else
    ``base_model``."""
    leaf = "speccoder" if speccoder_finetune else "base_model"
    out = PROJECT_ROOT / "SpecCoder" / "output" / "spec_gen" / branch / model_slug(model)
    if mode:
        out = out / mode
    out = out / leaf
    if make:
        out.mkdir(parents=True, exist_ok=True)
    return out


def write_llm_log(path: str | Path, entries: list[dict[str, Any]], *, header: str = "") -> None:
    """SpecMind-style prompt/response dump, written next to the generated jsonl.

    Each entry may be either the legacy shape ``{task_id, prompt, response}``
    or the multi-sample shape ``{task_id, prompt, samples}``, where each sample
    carries ``response``, ``code`` and optional ``evaluation``.
    """
    sep = "═" * 64
    lines: list[str] = []
    if header:
        lines += ["=" * 72, header, "=" * 72, ""]
    for e in entries:
        tid = e.get("task_id")
        samples = e.get("samples")
        task_lines = [
            f"╔{sep}╗",
            f"║  TASK: {tid}",
            f"╠{sep}╣",
            "║  [USER PROMPT]",
            f"╚{sep}╝",
            str(e.get("prompt", "")),
            "",
        ]
        if isinstance(samples, list):
            for sample in samples:
                sample_index = sample.get("sample_index", 0)
                task_lines += [
                    f"[SAMPLE {sample_index}]",
                    "[RAW LLM RESPONSE]",
                    str(sample.get("response", "")),
                    "",
                    "[PARSED CODE]",
                    str(sample.get("code", "")),
                ]
                if "evaluation" in sample:
                    evaluation = sample.get("evaluation", "")
                    if not isinstance(evaluation, str):
                        evaluation = json.dumps(evaluation, ensure_ascii=False, indent=2)
                    task_lines += [
                        "",
                        "[EVALUATION]",
                        evaluation,
                    ]
                task_lines.append("")
        else:
            task_lines += [
                "[RAW LLM RESPONSE]",
                str(e.get("response", "")),
                "",
            ]
        task_lines += [
            f"── END TASK={tid} ──────────────────────────────────────────",
            "",
        ]
        lines += task_lines
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def write_json(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def rewrite_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    tmp.replace(path)


def load_latest_jsonl_by_task(path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(path)
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            task_id = record.get("task_id")
            if task_id:
                latest[str(task_id)] = record
    return latest


def has_enough_samples(record: dict[str, Any] | None, num_samples: int) -> bool:
    if not record:
        return False
    responses = record.get("response") or []
    codes = record.get("code") or []
    return len(responses) >= num_samples and len(codes) >= num_samples


def task_log_name(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(task_id)).strip("_") or "task"


def append_task_log(path: str | Path, record: dict[str, Any], *, header: str = "") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sep = "═" * 64
    lines: list[str] = []
    if header:
        lines += ["=" * 72, header, "=" * 72, ""]
    lines += [
        f"╔{sep}╗",
        f"║  TASK: {record.get('task_id')}",
        f"╠{sep}╣",
        "║  [USER PROMPT]",
        f"╚{sep}╝",
        str(record.get("prompt", "")),
        "",
    ]
    attempts = record.get("attempts") or []
    responses = record.get("response") or [a.get("response", "") for a in attempts]
    codes = record.get("code") or [a.get("code", "") for a in attempts]
    sample_results = record.get("sample_results") or []
    for i, response in enumerate(responses):
        attempt_meta = attempts[i] if i < len(attempts) else {}
        lines += [
            f"[SAMPLE {i}]",
            "[RAW LLM RESPONSE]",
            str(response),
            "",
            "[PARSED CODE]",
            str(codes[i] if i < len(codes) else ""),
        ]
        if i < len(sample_results):
            lines += [
                "",
                "[EVALUATION]",
                json.dumps(sample_results[i], ensure_ascii=False, indent=2),
            ]
        if attempt_meta.get("time_latency") or attempt_meta.get("cost_token"):
            lines += [
                "",
                "[ATTEMPT METADATA]",
                json.dumps({
                    "time_latency": attempt_meta.get("time_latency"),
                    "cost_token": attempt_meta.get("cost_token"),
                }, ensure_ascii=False, indent=2),
            ]
        lines.append("")
    # RESULT section (only when the run evaluated on the real tests, i.e. eval is on).
    if sample_results:
        n_att = len(sample_results)
        n_correct = sum(1 for r in sample_results if float(r.get("pass_rate", 0.0) or 0.0) == 1.0)
        best = max(sample_results, key=lambda r: r.get("pass_rate", 0.0), default={})
        verdict = "PASS" if n_correct > 0 else "FAIL"
        lines += [
            f"╔{sep}╗",
            f"║  RESULT: {verdict}",
            f"╠{sep}╣",
        ]
        if record.get("old_pass_rate") is not None:
            lines.append(f"║  buggy baseline pass rate : {round(float(record['old_pass_rate']), 4)}")
        lines += [
            f"║  best pass rate           : {round(float(best.get('pass_rate', 0.0) or 0.0), 4)}"
            f"  ({best.get('passed_tests', 0)}/{best.get('total_tests', 0)} tests)",
            f"║  attempts fully correct   : {n_correct}/{n_att}",
        ]
        for i, r in enumerate(sample_results):
            pr = float(r.get("pass_rate", 0.0) or 0.0)
            lines.append(
                f"║    - attempt {r.get('attempt', i)}: {'PASS' if pr == 1.0 else 'FAIL'}"
                f"  pass_rate={round(pr, 4)} ({r.get('passed_tests', 0)}/{r.get('total_tests', 0)})"
            )
        if record.get("pass_at_k") is not None:
            lines.append(f"║  pass@k : {json.dumps(record.get('pass_at_k'), ensure_ascii=False)}")
        lines += [f"╚{sep}╝", ""]
    elif record.get("pass_at_k") is not None:
        lines += [
            "[TASK PASS@K]",
            json.dumps(record.get("pass_at_k"), ensure_ascii=False, indent=2),
            "",
        ]
    lines += [
        f"── END TASK={record.get('task_id')} ──────────────────────────────────────────",
        "",
    ]
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.flush()


def add_vllm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Base model (HF id or local path)")
    parser.add_argument("--lora_path", default=None, help="LoRA adapter path")
    parser.add_argument("--speccoder_finetune", action="store_true",
                        help="Apply the fine-tuned SpecCoder adapter (needs --lora_path).")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", "--context-len", "--context-length",
                        dest="max_model_len", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--cache-dir", default="")


# ---------------------------------------------------------------------------
# vLLM generation
# ---------------------------------------------------------------------------
def load_vllm(
    model: str,
    *,
    lora_path: str | None = None,
    speccoder_finetune: bool = False,
    max_model_len: int = 32768,
    tensor_parallel_size: int = 0,
    gpu_memory_utilization: float = 0.95,
    cache_dir: str = "",
):
    """Load a vLLM engine + tokenizer **once** and return
    ``(llm, tokenizer, lora_request)``. Callers that issue many generations
    (e.g. the multi-round program-repair loop) reuse the returned handle
    instead of reloading the model each call. ``--speccoder_finetune`` applies
    the fine-tuned adapter at ``lora_path`` on top of the base ``model``."""
    import torch
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    if speccoder_finetune and not lora_path:
        raise SystemExit("--speccoder_finetune requires --lora_path (the SpecCoder adapter)")

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    use_lora = bool(speccoder_finetune and lora_path)
    tp_size = int(tensor_parallel_size or 0) or torch.cuda.device_count()
    llm = LLM(
        model,
        enable_lora=use_lora,
        disable_log_stats=True,
        tensor_parallel_size=tp_size,
        download_dir=cache_dir or None,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    lora_request = LoRARequest("speccoder", 1, lora_path) if use_lora else None
    return llm, tokenizer, lora_request


def _vllm_token_ids(tokenizer, text: str) -> list[int]:
    try:
        return tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            verbose=False,
        )["input_ids"]
    except TypeError:
        return tokenizer.encode(text)


def _decode_vllm_token_ids(tokenizer, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=False)
    except TypeError:
        return tokenizer.decode(token_ids)


def _vllm_budgets(max_model_len: int, max_tokens: int) -> tuple[int, int]:
    max_model_len = max(2, int(max_model_len or 32768))
    requested_output = max(1, int(max_tokens or 1))
    if requested_output >= max_model_len:
        effective_output = max(1, max_model_len // 2)
    else:
        effective_output = requested_output
    prompt_budget = max(1, max_model_len - effective_output)
    return prompt_budget, effective_output


def _truncate_vllm_prompt(tokenizer, prompt: str, max_prompt_tokens: int) -> tuple[str, dict[str, Any]]:
    token_ids = _vllm_token_ids(tokenizer, prompt)
    original_tokens = len(token_ids)
    if original_tokens <= max_prompt_tokens:
        return prompt, {
            "prompt_truncated": False,
            "original_prompt_tokens": original_tokens,
            "truncated_prompt_tokens": original_tokens,
        }

    kept_ids = token_ids[-max_prompt_tokens:]

    truncated = _decode_vllm_token_ids(tokenizer, kept_ids)
    final_tokens = len(_vllm_token_ids(tokenizer, truncated))
    if final_tokens > max_prompt_tokens:
        kept_ids = _vllm_token_ids(tokenizer, truncated)[-max_prompt_tokens:]
        truncated = _decode_vllm_token_ids(tokenizer, kept_ids)
        final_tokens = len(kept_ids)
    return truncated, {
        "prompt_truncated": True,
        "original_prompt_tokens": original_tokens,
        "truncated_prompt_tokens": final_tokens,
    }


def vllm_chat(
    llm,
    tokenizer,
    messages_list: list[list[dict[str, str]]],
    *,
    lora_request=None,
    num_samples: int = 1,
    max_tokens: int = 4096,
    max_model_len: int = 32768,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 0,
) -> list[list[str]]:
    """Run a preloaded vLLM ``llm`` over chat-message lists; return per-input
    list of ``num_samples`` raw completions."""
    from vllm import SamplingParams

    prompt_budget, effective_max_tokens = _vllm_budgets(max_model_len, max_tokens)
    questions = []
    for messages in messages_list:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt, _meta = _truncate_vllm_prompt(tokenizer, prompt, prompt_budget)
        questions.append(prompt)
    sampling_params = SamplingParams(
        max_tokens=effective_max_tokens, n=num_samples,
        temperature=temperature, top_p=top_p, top_k=top_k,
    )
    outputs = llm.generate(questions, sampling_params, lora_request=lora_request)
    return [[o.text for o in out.outputs] for out in outputs]


def vllm_chat_with_metadata(
    llm,
    tokenizer,
    messages_list: list[list[dict[str, str]]],
    *,
    lora_request=None,
    num_samples: int = 1,
    max_tokens: int = 4096,
    max_model_len: int = 32768,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 0,
) -> tuple[list[list[str]], list[list[dict[str, Any]]]]:
    """vLLM chat with exact token counts from RequestOutput token ids."""
    from vllm import SamplingParams

    prompt_budget, effective_max_tokens = _vllm_budgets(max_model_len, max_tokens)
    questions = []
    truncation_meta: list[dict[str, Any]] = []
    for messages in messages_list:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt, meta = _truncate_vllm_prompt(tokenizer, prompt, prompt_budget)
        questions.append(prompt)
        truncation_meta.append(meta)
    sampling_params = SamplingParams(
        max_tokens=effective_max_tokens, n=num_samples,
        temperature=temperature, top_p=top_p, top_k=top_k,
    )
    outputs = llm.generate(questions, sampling_params, lora_request=lora_request)

    texts: list[list[str]] = []
    metadata: list[list[dict[str, Any]]] = []
    for request_idx, request_output in enumerate(outputs):
        prompt_token_ids = getattr(request_output, "prompt_token_ids", None) or []
        prompt_tokens = len(prompt_token_ids)
        prompt_meta = truncation_meta[request_idx] if request_idx < len(truncation_meta) else {}
        request_texts: list[str] = []
        request_meta: list[dict[str, Any]] = []
        for completion in request_output.outputs:
            response = completion.text or ""
            response_token_ids = getattr(completion, "token_ids", None) or []
            response_tokens = len(response_token_ids)
            request_texts.append(response)
            request_meta.append({
                "cost_token": {
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens,
                    "total_tokens": prompt_tokens + response_tokens,
                    "source": "vllm_token_ids",
                },
                "prompt_truncation": prompt_meta,
            })
        texts.append(request_texts)
        metadata.append(request_meta)
    return texts, metadata


def generate_chat(
    model: str,
    messages_list: list[list[dict[str, str]]],
    *,
    lora_path: str | None = None,
    speccoder_finetune: bool = False,
    num_samples: int = 1,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 0,
    max_model_len: int = 32768,
    tensor_parallel_size: int = 0,
    gpu_memory_utilization: float = 0.95,
    cache_dir: str = "",
) -> list[list[str]]:
    """One-shot vLLM generation (loads the engine, generates, returns). Kept for
    callers that generate once; multi-round callers should use ``load_vllm`` +
    ``vllm_chat`` to avoid reloading the model."""
    llm, tokenizer, lora_request = load_vllm(
        model, lora_path=lora_path, speccoder_finetune=speccoder_finetune,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        cache_dir=cache_dir,
    )
    return vllm_chat(
        llm, tokenizer, messages_list, lora_request=lora_request,
        num_samples=num_samples, max_tokens=max_tokens,
        max_model_len=max_model_len,
        temperature=temperature, top_p=top_p, top_k=top_k,
    )
