from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

_THIS = Path(__file__).resolve()
PROJECT_ROOT: Path | None = None
for _parent in _THIS.parents:
    if (_parent / "SpecCoder").is_dir():
        PROJECT_ROOT = _parent
        break
if PROJECT_ROOT is None:
    PROJECT_ROOT = _THIS.parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from SpecCoder.spec_utils import parse_assert_comment  
from SpecCoder.src.experiment.gen_utils import ( 
    GENERATION_FILE, append_jsonl, load_vllm, model_slug, write_json,
)
from SpecCoder.src.experiment.specgen.inline.prompt import ( 
    SYSTEM_PROMPT, build_spec_generation_prompt,
)

SYSTEM_PROMPT_DEFAULT = SYSTEM_PROMPT
DEFAULT_REFERENCE = str(PROJECT_ROOT / "SpecCoder" / "data" / "candidates_heldout.jsonl")
SOLUTION_CLOSE = "</solution>"


def strip_explanation(text: str) -> str:
    idx = text.rfind(SOLUTION_CLOSE)
    return text[: idx + len(SOLUTION_CLOSE)] if idx != -1 else text


def assert_spans(annotated_code: str) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    cursor = 0
    for line in annotated_code.splitlines(keepends=True):
        expr = parse_assert_comment(line.rstrip("\n"))
        if expr is not None:
            content_len = len(line.rstrip("\n"))
            key = f"assert {expr}"
            dedup = seen.get(key, 0)
            seen[key] = dedup + 1
            spans.append({
                "idx": len(spans),
                "dedup_key": dedup,
                "expr": expr,
                "char_start": cursor,
                "char_end": cursor + content_len,
            })
        cursor += len(line)
    return spans


def build_prompt_str(tokenizer, user_prompt: str, system_prompt: str) -> str:
    """The chat-formatted prefix up to (and including) the assistant turn start."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        sys_part = f"{system_prompt}\n\n" if system_prompt else ""
        return f"{sys_part}{user_prompt}\n"


def _vllm_score_batch(llm, batch_token_ids: list[list[int]], lora_request):
    from vllm import SamplingParams

    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    try:
        from vllm import TokensPrompt
        prompts = [TokensPrompt(prompt_token_ids=t) for t in batch_token_ids]
    except Exception:
        prompts = [{"prompt_token_ids": t} for t in batch_token_ids]
    outs = llm.generate(prompts, sp, lora_request=lora_request)
    return list(outs)


def _logprob_value(entry, token_id):
    if not entry:
        return None
    obj = entry.get(token_id)
    if obj is None:
        return None
    return obj.logprob if hasattr(obj, "logprob") else float(obj)


def tokenize_sequence(tokenizer, prompt_str: str, target_str: str) -> dict[str, Any]:
    full = prompt_str + target_str
    enc = tokenizer(full, return_offsets_mapping=True, add_special_tokens=False)
    return {
        "token_ids": list(enc["input_ids"]),
        "offsets": enc["offset_mapping"],
        "prompt_len": len(prompt_str),
        "full_len": len(full),
    }


def aggregate_scores(out, tok: dict[str, Any], spans: list[dict[str, Any]]) -> dict[str, Any]:
    token_ids = tok["token_ids"]
    offsets = tok["offsets"]
    prompt_len = tok["prompt_len"]
    full_len = tok["full_len"]
    L = len(token_ids)

    plp = list(getattr(out, "prompt_logprobs", None) or [])

    # nll_by_token[t] = -logp(token_t | token_<t); prompt_logprobs[t] holds it.
    nll_by_token: dict[int, float] = {}
    for t in range(1, min(L, len(plp))):
        lp = _logprob_value(plp[t], token_ids[t])
        if lp is not None:
            nll_by_token[t] = -float(lp)

    def char_span_to_tokens(c_start: int, c_end: int) -> list[int]:
        # a token belongs to [c_start, c_end) if its char-start falls inside it
        out_toks = []
        for t in range(1, L):
            s, e = offsets[t]
            if e <= s:  # special/empty tokens
                continue
            if c_start <= s < c_end:
                out_toks.append(t)
        return out_toks

    target_tokens = char_span_to_tokens(prompt_len, full_len)

    def agg(tokens: list[int]) -> dict[str, Any]:
        toks = [t for t in tokens if t in nll_by_token]
        if not toks:
            return {"n_tokens": 0, "nll_sum": 0.0, "perplexity": None}
        nll_sum = sum(nll_by_token[t] for t in toks)
        mean_nll = nll_sum / len(toks)
        return {"n_tokens": len(toks), "nll_sum": nll_sum, "perplexity": math.exp(mean_nll)}

    per_assert = []
    assert_token_pool: list[int] = []
    for sp in spans:
        toks = char_span_to_tokens(prompt_len + sp["char_start"], prompt_len + sp["char_end"])
        assert_token_pool.extend(toks)
        per_assert.append({**{k: sp[k] for k in ("idx", "dedup_key", "expr")}, **agg(toks)})

    full_stats = agg(target_tokens)
    micro = agg(assert_token_pool)
    assert_ppls = [a["perplexity"] for a in per_assert if a["perplexity"] is not None]
    macro = (sum(assert_ppls) / len(assert_ppls)) if assert_ppls else None

    return {
        "num_assert_tokens": len(assert_token_pool),
        "num_target_tokens": full_stats["n_tokens"],
        "ppl_full": full_stats["perplexity"],
        "ppl_assert_micro": micro["perplexity"],
        "ppl_assert_macro": macro,
        "per_assert": per_assert,
    }


def _uid(rec: dict[str, Any]) -> str:
    return str(rec.get("sample_id") or rec.get("task_id") or "")


def build_reference_item(rec: dict[str, Any], default_system: str) -> tuple[str, str, str]:

    if rec.get("annotated_correct"):
        sample = {
            "problem_text": rec.get("problem_description") or rec.get("problem_text") or "",
            "starter_code": rec.get("starter_code") or "",
        }
        clean = str(rec.get("clean_correct") or "")
        user = build_spec_generation_prompt(sample, clean)
        return SYSTEM_PROMPT_DEFAULT, user, str(rec.get("annotated_correct") or "")

    system = rec.get("training_system_prompt") or rec.get("system_prompt") or default_system
    user = str(rec.get("training_prompt") or rec.get("prompt") or "")
    target = strip_explanation(str(rec.get("ground_truth_response") or rec.get("annotated_code") or ""))
    return system, user, target


def load_reference(path: str, attempt: int, limit: int | None) -> list[dict[str, Any]]:
    by_uid: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rec.pop("file_name", None)
            rec.pop("filename", None)
            if int(rec.get("attempt", 0)) != attempt:
                continue
            uid = _uid(rec)
            if uid and uid not in by_uid:
                by_uid[uid] = rec
    rows = list(by_uid.values())
    if limit:
        rows = rows[:limit]
    return rows


def resolve_reference_path(opt: argparse.Namespace) -> str:
    p = Path(opt.reference)
    if p.is_dir():
        p = p / GENERATION_FILE
    if not p.exists():
        raise SystemExit(f"Reference generation file not found: {p}")
    return str(p)


def resolve_output_dir(opt: argparse.Namespace, ref_path: str) -> str:
    if opt.output:
        out = os.path.abspath(opt.output)
    else:
        out = str(Path(ref_path).parent / "perplexity")
    os.makedirs(out, exist_ok=True)
    return out


def variant_dir(out_root: str, opt: argparse.Namespace, which: str) -> str:
    """Per-model output folder: <out_root>/<model_slug>/<base|finetune>/."""
    d = os.path.join(out_root, model_slug(opt.model), which)
    os.makedirs(d, exist_ok=True)
    return d


def load_done_task_ids(ppl_path: str) -> set[str]:
    done: set[str] = set()
    if os.path.exists(ppl_path):
        with open(ppl_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        done.add(str(json.loads(line).get("task_id")))
                    except json.JSONDecodeError:
                        continue
    return done


def run(opt: argparse.Namespace) -> None:
    ref_path = resolve_reference_path(opt)
    out_root = resolve_output_dir(opt, ref_path)
    rows = load_reference(ref_path, opt.attempt, opt.limit if opt.num_tasks is None else opt.num_tasks)
    print(f"reference={ref_path} | tasks={len(rows)} | out_root={out_root}")

    prepared = []
    for rec in rows:
        system, user, target = build_reference_item(rec, opt.system_prompt)
        spans = assert_spans(target)
        prepared.append({
            "task_id": _uid(rec),
            "name": rec.get("name") or rec.get("task_id"),
            "setting": rec.get("setting") or rec.get("task_type"),
            "system_prompt": system,
            "user_prompt": user,
            "annotated_code": target,
            "spans": spans,
            "num_asserts": len(spans),
        })

    models_to_score = ["base", "finetune"] if opt.score == "both" else [opt.score]
    need_lora = "finetune" in models_to_score
    if need_lora and not opt.lora_path:
        raise SystemExit("Scoring the finetuned model requires --lora_path")

    ppl_paths = {w: os.path.join(variant_dir(out_root, opt, w), "perplexity.jsonl") for w in models_to_score}
    done_by_model = {w: load_done_task_ids(ppl_paths[w]) for w in models_to_score}
    pending_any = any(
        item["task_id"] not in done_by_model[w]
        for w in models_to_score for item in prepared
        if item["annotated_code"].strip() and item["user_prompt"].strip()
    )

    if pending_any:
        llm, tokenizer, lora_request = load_vllm(
            opt.model, lora_path=opt.lora_path, speccoder_finetune=need_lora,
            max_model_len=opt.max_model_len, tensor_parallel_size=opt.tensor_parallel_size,
            gpu_memory_utilization=opt.gpu_memory_utilization, cache_dir=opt.cache_dir,
        )
        lora_by_model = {"base": None, "finetune": lora_request}
        for item in prepared:
            item["prompt_str"] = build_prompt_str(tokenizer, item["user_prompt"], item["system_prompt"])
            item["tok"] = tokenize_sequence(tokenizer, item["prompt_str"], item["annotated_code"])
        batch_size = max(1, int(opt.batch_size))
        for which in models_to_score:
            done = done_by_model[which]
            todo = [it for it in prepared if it["task_id"] not in done
                    and it["annotated_code"].strip() and it["user_prompt"].strip()]
            print(f"=== scoring {which}: {len(todo)} todo, {len(done)} already done "
                  f"(batch_size={batch_size}) ===")
            for start in range(0, len(todo), batch_size):
                batch = todo[start : start + batch_size]
                outs = _vllm_score_batch(llm, [it["tok"]["token_ids"] for it in batch],
                                         lora_by_model[which])
                for item, out in zip(batch, outs):
                    stats = aggregate_scores(out, item["tok"], item["spans"])
                    rec = {
                        "task_id": item["task_id"], "name": item["name"], "setting": item["setting"],
                        "num_asserts": item["num_asserts"], "model": which, **stats,
                    }
                    append_jsonl(ppl_paths[which], rec)   # write each task as soon as it is scored
    else:
        print("All tasks already scored for the requested model(s); writing summary only.")

    by_model_records = {w: load_perplexity_jsonl(ppl_paths[w]) for w in models_to_score}
    summary: dict[str, Any] = {
        "reference": ref_path, "model": opt.model, "lora_path": opt.lora_path,
        "scored": models_to_score,
        "by_model": {w: _summarize_stats(list(by_model_records[w].values())) for w in models_to_score},
    }
    if "base" in by_model_records and "finetune" in by_model_records:
        summary["delta"] = _paired_delta(by_model_records["base"], by_model_records["finetune"])

    summary_path = os.path.join(out_root, model_slug(opt.model), "perplexity_summary.json")
    write_json(summary_path, summary)

    for w in models_to_score:
        print(f"Per-task ({w}) -> {ppl_paths[w]}")
    print(f"Summary       -> {summary_path}")
    for w in models_to_score:
        s = summary["by_model"][w]["ppl_assert_micro"]
        print(f"  [{w}] assert-micro  mean={s['mean']} median={s['median']} "
              f"min={s['min']} max={s['max']} (n={s['n']})")
    if "delta" in summary:
        d = summary["delta"]
        print(f"  base→finetune assert-micro: mean drop={d['mean_drop']} "
              f"median drop={d['median_drop']} | finetune lower on {d['win_rate']:.0%} of tasks")


def load_perplexity_jsonl(path: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[str(rec.get("task_id"))] = rec
    return out


def _stats(xs: list[float]) -> dict[str, Any]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None, "std": None}
    s = sorted(xs)
    n = len(s)
    mean = sum(s) / n
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    std = (sum((x - mean) ** 2 for x in s) / n) ** 0.5
    return {"n": n, "mean": mean, "median": median, "min": s[0], "max": s[-1], "std": std}


def _summarize_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    def vals(field):
        return [r.get(field) for r in records if r.get(field) is not None]

    return {
        "num_tasks": len(records),
        "ppl_assert_micro": _stats(vals("ppl_assert_micro")),
        "ppl_assert_macro": _stats(vals("ppl_assert_macro")),
        "ppl_full": _stats(vals("ppl_full")),
    }


def _paired_delta(base: dict[str, Any], fine: dict[str, Any]) -> dict[str, Any]:
    """Paired base→finetune comparison on assert-micro PPL over shared tasks."""
    drops: list[float] = []
    wins = 0
    for tid in set(base) & set(fine):
        b, f = base[tid].get("ppl_assert_micro"), fine[tid].get("ppl_assert_micro")
        if b is None or f is None:
            continue
        drops.append(b - f)
        if f < b:
            wins += 1
    n = len(drops)
    st = _stats(drops)
    out = {
        "num_paired_tasks": n,
        "mean_drop": st["mean"], "median_drop": st["median"],
        "win_rate": (wins / n) if n else None, "wins": wins,
    }
    # Wilcoxon signed-rank if scipy is available (paired, non-parametric).
    try:
        from scipy.stats import wilcoxon
        if n >= 1 and any(d != 0 for d in drops):
            stat, p = wilcoxon(drops)
            out["wilcoxon_stat"], out["wilcoxon_p"] = float(stat), float(p)
    except Exception:
        pass
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-assert perplexity of inline specs: teacher-force-score a method's "
                    "annotated_code under the base and/or finetuned model.")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE,
                        help=f"Held-out reference set (default: {DEFAULT_REFERENCE}). Accepts the relabel "
                             "candidate schema (annotated_correct + clean_correct + problem_description -> "
                             "rebuilds the inline spec-generation prompt), the spec_ok training schema "
                             "(training_prompt / ground_truth_response), or generic prompt / annotated_code.")
    parser.add_argument("--model", required=True, help="Base model HF id / local path.")
    parser.add_argument("--lora_path", "--lora-path", default=None,
                        help="SpecCoder LoRA adapter (required to score the finetuned model).")
    parser.add_argument("--score", choices=["base", "finetune", "both"], default="both",
                        help="Which model(s) to score the same specs under (default both -> reports delta).")
    parser.add_argument("--attempt", type=int, default=0,
                        help="Which generation attempt index to score (default 0).")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=32,
                        help="Tasks teacher-forced per vLLM generate call (default 32).")
    parser.add_argument("--system-prompt", "--system_prompt", default=SYSTEM_PROMPT_DEFAULT,
                        help="System prompt used when chat-formatting the scoring sequence.")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N tasks.")
    parser.add_argument("--num-tasks", "--num_tasks", type=int, default=None,
                        help="Alias for --limit (takes precedence if set).")
    # vLLM engine args (same names/semantics as the experiment runners).
    parser.add_argument("--max-model-len", "--context-len", "--context-length",
                        dest="max_model_len", type=int, default=32768)
    parser.add_argument("--tensor-parallel-size", type=int, default=0,
                        help="vLLM tensor_parallel_size; 0 means torch.cuda.device_count().")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--output", default=None,
                        help="Output root (default: <reference dir>/perplexity). "
                             "Per-model results go to <root>/<model_slug>/<base|finetune>/.")
    return parser.parse_args(argv)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
