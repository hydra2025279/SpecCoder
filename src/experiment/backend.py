
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

# --- make ``SpecCoder`` importable as a top-level package ------------------
_THIS = Path(__file__).resolve()
PROJECT_ROOT = next((p for p in _THIS.parents if (p / "SpecCoder").is_dir()), _THIS.parents[3])
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def add_backend_args(parser: argparse.ArgumentParser, *, include_model: bool = True) -> None:
    """Shared model/backend CLI surface for every experiment script.

    Set ``include_model=False`` when the caller exposes the model under a
    different flag (e.g. ``--regen_model``); the caller is then responsible for
    setting ``opt.model`` and ``opt.speccoder_finetune`` before ``make_backend``."""
    parser.add_argument("--backend", choices=["vllm"], default="vllm")
    if include_model:
        parser.add_argument("--model", required=True, help="HF id / local path (vllm)")
        parser.add_argument("--speccoder_finetune", "--speccoder-finetune", action="store_true",
                            help="Apply the fine-tuned SpecCoder adapter (needs --lora_path).")
    parser.add_argument("--lora_path", default=None, help="LoRA adapter path (vllm)")
    parser.add_argument("--max-model-len", "--context-len", "--context-length",
                        dest="max_model_len", type=int, default=32768)
    parser.add_argument("--tensor-parallel-size", type=int, default=0,
                        help="vLLM tensor_parallel_size; 0 means use torch.cuda.device_count().")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95,
                        help="vLLM gpu_memory_utilization.")
    parser.add_argument("--cache-dir", default="")
    # sampling (shared)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)


class _CoalesceItem:
    """One pending single-prompt ``chat`` call waiting to be merged into a batch."""

    __slots__ = ("messages", "num_samples", "task_id", "result", "exc", "done")

    def __init__(self, messages, num_samples, task_id):
        self.messages = messages
        self.num_samples = int(num_samples or 1)
        self.task_id = task_id
        self.result = None
        self.exc: BaseException | None = None
        self.done = threading.Event()


class _ChatCoalescer:
    """Merge concurrent single-prompt ``chat`` calls into batched dispatches.

    Methods like HoarePrompt annotation fire MANY small, sequentially-dependent LLM
    calls per task. Running tasks in parallel threads makes those calls overlap, but
    each call alone is a batch-of-1 — useless for an in-process vLLM engine (whose
    ``llm.generate`` is single-threaded and unsafe to call concurrently). This
    coalescer collects whatever single-prompt calls are in flight
    across the worker threads and dispatches them as ONE batch to ``backend._chat_impl``:

      * vLLM — a single ``llm.generate([...])`` on the dispatcher thread, so the
               engine runs continuous batching and is never touched concurrently.

    A single dispatcher thread owns every real call, so token accounting and the engine
    stay single-threaded; per-task attribution rides on each item's explicit ``task_id``."""

    def __init__(self, backend, max_batch: int, max_wait: float):
        self.backend = backend
        self.max_batch = max(1, int(max_batch))
        self.max_wait = max(0.0, float(max_wait))
        self._q: list[_CoalesceItem] = []
        self._cv = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="chat-coalescer", daemon=True)
        self._thread.start()

    def submit(self, messages, num_samples, task_id):
        item = _CoalesceItem(messages, num_samples, task_id)
        with self._cv:
            self._q.append(item)
            self._cv.notify()
        item.done.wait()
        if item.exc is not None:
            raise item.exc
        return item.result

    def _take_group(self) -> tuple[int, list[_CoalesceItem]]:
        """Caller holds the lock and ``self._q`` is non-empty. Pull up to ``max_batch``
        items that share the first item's ``num_samples`` (vLLM SamplingParams.n must be
        uniform within one ``generate`` call); leave the rest queued."""
        n = self._q[0].num_samples
        group, rest = [], []
        for it in self._q:
            if it.num_samples == n and len(group) < self.max_batch:
                group.append(it)
            else:
                rest.append(it)
        self._q = rest
        return n, group

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._q and not self._stop:
                    self._cv.wait()
                if self._stop and not self._q:
                    return
                # Got at least one; linger briefly so more in-flight calls join the batch.
                if len(self._q) < self.max_batch and self.max_wait > 0:
                    self._cv.wait(self.max_wait)
                n, group = self._take_group()
            if group:
                self._dispatch(n, group)

    def _dispatch(self, n: int, group: list[_CoalesceItem]) -> None:
        msgs = [it.messages for it in group]
        tids = [it.task_id for it in group]
        try:
            outs = self.backend._chat_impl(msgs, num_samples=n, task_ids=tids)
            for it, out in zip(group, outs):
                it.result = out
                it.done.set()
        except BaseException as exc:  # noqa: BLE001 — propagate to every waiter
            for it in group:
                it.exc = exc
                it.done.set()

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join(timeout=30)


class ChatBackend:
    """Interface: ``chat(messages_list, num_samples=None, task_ids=None) -> list[list[str]]``.

    Every backend accumulates real token usage on EVERY LLM call (central token
    accounting). Attribute a call to a task either by passing ``task_ids`` to
    ``chat`` (one per message, for batched calls) or by calling ``begin_task`` before
    a block of calls (for per-task / nested-call methods such as HoarePrompt).

    Subclasses implement ``_chat_impl`` (the real batched call); the base ``chat``
    routes single-prompt calls through an optional coalescer (see
    :meth:`enable_coalescing`) so parallel per-task methods batch automatically."""

    _coalescer: "_ChatCoalescer | None" = None

    def _ensure_tokens(self):
        if not hasattr(self, "token_usage"):
            self.token_usage = {"input": 0, "output": 0, "calls": 0}
            self.token_by_task: dict[str, dict[str, int]] = {}
            self._cur_task = None
            # Per-thread current task: parallel per-task methods (parallel HoarePrompt
            # annotation) each call begin_task in their own worker thread, so the active
            # task must be thread-local, not a single shared attribute that races.
            self._tls = threading.local()
            self._usage_lock = threading.Lock()

    def begin_task(self, task_id):
        self._ensure_tokens()
        tid = None if task_id is None else str(task_id)
        self._cur_task = tid           # back-compat for sequential callers
        self._tls.cur_task = tid       # authoritative under parallel annotation

    def record_usage(self, prompt_tokens, completion_tokens, task_id=None):
        self._ensure_tokens()
        pin, pout = int(prompt_tokens or 0), int(completion_tokens or 0)
        tid = task_id if task_id is not None else getattr(self._tls, "cur_task", None) or self._cur_task
        with self._usage_lock:
            self.token_usage["input"] += pin
            self.token_usage["output"] += pout
            self.token_usage["calls"] += 1
            if tid is not None:
                d = self.token_by_task.setdefault(str(tid), {"input": 0, "output": 0, "calls": 0})
                d["input"] += pin
                d["output"] += pout
                d["calls"] += 1

    def _tid_for(self, task_ids, i):
        """Resolve the task id for message ``i``: the explicit ``task_ids`` entry
        (batched calls) or, failing that, the ``begin_task`` context (per-task
        methods such as HoarePrompt that don't pass task_ids on each call). The
        ``begin_task`` context is read thread-locally first so parallel annotation
        workers don't read each other's task."""
        if task_ids and i < len(task_ids) and task_ids[i] is not None:
            return task_ids[i]
        tls = getattr(self, "_tls", None)
        local = getattr(tls, "cur_task", None) if tls is not None else None
        return local if local is not None else getattr(self, "_cur_task", None)

    # ------------------------------------------------------------------ #
    # Optional call coalescing for parallel per-task methods (annotation).
    # ------------------------------------------------------------------ #
    def enable_coalescing(self, max_batch: int = 32, max_wait: float = 0.05) -> None:
        """Start merging concurrent single-prompt ``chat`` calls into batched dispatches
        (see :class:`_ChatCoalescer`). Idempotent. Call ``disable_coalescing`` when the
        parallel section ends to flush and stop the dispatcher."""
        self._ensure_tokens()
        if self._coalescer is None:
            self._coalescer = _ChatCoalescer(self, max_batch, max_wait)

    def disable_coalescing(self) -> None:
        """Flush pending coalesced calls and stop the dispatcher."""
        c = self._coalescer
        if c is not None:
            self._coalescer = None
            c.stop()

    def chat(self, messages_list, num_samples=None, task_ids=None, use_lora=True):
        """Public entry. When coalescing is active, a single-prompt call is queued and
        merged with other in-flight calls; otherwise it goes straight to ``_chat_impl``.
        Batched calls (len > 1) always dispatch directly — they are already a batch.

        ``use_lora=False`` forces the BASE model (vllm: drops the LoRA adapter) and ALWAYS
        bypasses the coalescer, so base-model calls are never merged with LoRA calls."""
        c = self._coalescer
        if use_lora and c is not None and len(messages_list) == 1:
            tid = task_ids[0] if task_ids else self._tid_for(None, 0)
            n = num_samples or getattr(self.opt, "num_samples", 1)
            return [c.submit(messages_list[0], n, tid)]
        return self._chat_impl(messages_list, num_samples=num_samples, task_ids=task_ids, use_lora=use_lora)

    def _chat_impl(self, messages_list, num_samples=None, task_ids=None, use_lora=True):  # pragma: no cover - interface
        raise NotImplementedError


class VLLMBackend(ChatBackend):
    def __init__(self, opt):
        from SpecCoder.src.experiment.gen_utils import load_vllm

        self.opt = opt
        self.llm, self.tokenizer, self.lora_request = load_vllm(
            opt.model,
            lora_path=opt.lora_path,
            speccoder_finetune=opt.speccoder_finetune,
            max_model_len=opt.max_model_len,
            tensor_parallel_size=opt.tensor_parallel_size,
            gpu_memory_utilization=opt.gpu_memory_utilization,
            cache_dir=opt.cache_dir,
        )

    def _count(self, messages, texts):
        try:
            pin = len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        except Exception:
            pin = len(self.tokenizer.encode("\n".join(str(m.get("content", "")) for m in messages)))
        pout = sum(len(self.tokenizer.encode(t or "")) for t in texts)
        return pin, pout

    def _prelog(self, messages_list, task_ids):
        from SpecCoder.src.experiment.programrepair import common
        step = common.current_step()
        for i, m in enumerate(messages_list):
            common.log_call_prompt(self._tid_for(task_ids, i), m, step)

    def _chat_impl(self, messages_list, num_samples=None, task_ids=None, use_lora=True):
        from SpecCoder.src.experiment.gen_utils import vllm_chat

        opt = self.opt
        self._prelog(messages_list, task_ids)
        completions = vllm_chat(
            self.llm, self.tokenizer, messages_list,
            lora_request=self.lora_request if use_lora else None,
            num_samples=num_samples or opt.num_samples,
            max_tokens=opt.max_tokens,
            max_model_len=opt.max_model_len,
            temperature=opt.temperature, top_p=opt.top_p, top_k=opt.top_k,
        )
        from SpecCoder.src.experiment.programrepair import common
        step = common.current_step()
        n = len(completions)
        for i, comps in enumerate(completions):
            pin, pout = self._count(messages_list[i], comps)
            tid = task_ids[i] if task_ids and i < len(task_ids) else None
            self.record_usage(pin, pout, tid)
            if tid is not None:
                common.tqdm_print(f"      ✓ {tid}  {step or 'LLM call'}  |  done ({i + 1}/{n})")
        return completions

    def chat_with_metadata(self, messages_list, num_samples=None, task_ids=None, use_lora=True):
        from SpecCoder.src.experiment.gen_utils import vllm_chat_with_metadata

        opt = self.opt
        self._prelog(messages_list, task_ids)
        completions, metadata = vllm_chat_with_metadata(
            self.llm, self.tokenizer, messages_list,
            lora_request=self.lora_request if use_lora else None,
            num_samples=num_samples or opt.num_samples,
            max_tokens=opt.max_tokens,
            max_model_len=opt.max_model_len,
            temperature=opt.temperature, top_p=opt.top_p, top_k=opt.top_k,
        )
        for i, comps in enumerate(completions):
            pin, pout = self._count(messages_list[i], comps)
            self.record_usage(pin, pout, task_ids[i] if task_ids and i < len(task_ids) else None)
        return completions, metadata


def write_token_files(backend: ChatBackend, out_dir, allowed_task_ids=None) -> None:
    """Write ``token.jsonl`` (one row per task) and ``token_total.json``.

    ``token.jsonl`` keeps EVERY row (merged with any prior run — non-destructive).
    When ``allowed_task_ids`` is given (e.g. the current humanexec task set), the
    totals in ``token_total.json`` are computed ONLY over rows whose ``task_id`` is in
    that set — so a shrunk benchmark doesn't keep counting stale tasks from a larger run.
    """
    import json
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "token.jsonl"
    merged: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                merged[str(r.get("task_id"))] = r
    for tid, d in getattr(backend, "token_by_task", {}).items():
        merged[str(tid)] = {
            "task_id": tid,
            "input_token": d["input"],
            "output_token": d["output"],
            "total_token": d["input"] + d["output"],
            "calls": d["calls"],
        }
    with path.open("w", encoding="utf-8") as f:
        for r in merged.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    allowed = None if allowed_task_ids is None else {str(t) for t in allowed_task_ids}
    counted = [r for r in merged.values()
               if allowed is None or str(r.get("task_id")) in allowed]
    total_in = sum(int(r.get("input_token", 0)) for r in counted)
    total_out = sum(int(r.get("output_token", 0)) for r in counted)
    (out / "token_total.json").write_text(json.dumps({
        "total_input": total_in,
        "total_output": total_out,
        "total": total_in + total_out,
        "num_calls": sum(int(r.get("calls", 0)) for r in counted),
        "num_tasks": len(counted),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_token_records(out_dir, records, *, with_attempt: bool = True, filename_suffix: str = "",
                          allowed_task_ids=None) -> None:
    """Per-record token accounting that mirrors ``generation.jsonl``: ONE appended
    line per record, never overwritten, sourced from each record's ``cost_token``
    (``{prompt_tokens, response_tokens, total_tokens}``).

    * ``with_attempt=True``  (repair): one line per ``(task_id, attempt)``.
    * ``with_attempt=False`` (localize): one line per ``task_id``.

    Already-written rows are kept and de-duplicated, so re-running with a larger
    ``--num-samples`` only APPENDS the new attempts (it never rewrites or loses the
    earlier ones). ``token_total.json`` is RECOMPUTED from the full ``token.jsonl``
    on every call."""
    import json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"token{filename_suffix}.jsonl"

    def _key(d):
        return (str(d.get("task_id")), int(d.get("attempt", 0))) if with_attempt else (str(d.get("task_id")),)

    existing_rows: list[dict] = []
    seen: set = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                existing_rows.append(r)
                seen.add(_key(r))

    new_rows: list[dict] = []
    for rec in records:
        if _key(rec) in seen:
            continue
        seen.add(_key(rec))
        c = rec.get("cost_token") or {}
        pin = int(c.get("prompt_tokens", 0) or 0)
        pout = int(c.get("response_tokens", 0) or 0)
        row = {"task_id": rec.get("task_id")}
        if with_attempt:
            row["attempt"] = int(rec.get("attempt", 0))
        row["input_token"] = pin
        row["output_token"] = pout
        row["total_token"] = int(c.get("total_tokens", pin + pout) or (pin + pout))
        new_rows.append(row)

    if new_rows:
        with path.open("a", encoding="utf-8") as f:
            for r in new_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    rows = existing_rows + new_rows
    # token{suffix}.jsonl keeps every row; the totals optionally restrict to the
    # current dataset (allowed_task_ids) so a shrunk benchmark stops counting stale tasks.
    allowed = None if allowed_task_ids is None else {str(t) for t in allowed_task_ids}
    counted = [r for r in rows if allowed is None or str(r.get("task_id")) in allowed]
    total_in = sum(int(r.get("input_token", 0)) for r in counted)
    total_out = sum(int(r.get("output_token", 0)) for r in counted)
    total = {
        "total_input": total_in,
        "total_output": total_out,
        "total": total_in + total_out,
        "num_tasks": len({str(r.get("task_id")) for r in counted}),
    }
    if with_attempt:
        total["num_attempts"] = len(counted)
    (out / f"token_total{filename_suffix}.json").write_text(
        json.dumps(total, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_backend(opt) -> ChatBackend:
    """Build the persistent backend selected by ``opt.backend``."""
    if opt.backend == "vllm":
        return VLLMBackend(opt)
    raise SystemExit(f"SpecCoder experiments support only --backend vllm, got {opt.backend!r}")
