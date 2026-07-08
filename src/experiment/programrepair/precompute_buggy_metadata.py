
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

# Allow running as a file: make the repo root importable so ``SpecCoder...`` works.
_THIS = os.path.abspath(__file__)
_ROOT = _THIS
for _ in range(4):  # .../SpecCoder/src/experiment/programrepair/<file> -> repo root above SpecCoder
    _ROOT = os.path.dirname(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from SpecCoder.src.codeforces_eval import calculate_pass_rate  # noqa: E402
from SpecCoder.src.experiment.programrepair import common  # noqa: E402

# Same memory policy as common.evaluate (no RLIMIT on macOS, 512MB cap on Linux).
_EVAL_MEMORY_MB = None if sys.platform == "darwin" else 512


def compute_metadata(sample: dict[str, Any], timeout: float, test_workers: int) -> dict[str, Any]:
    """Evaluate ``buggy_solution`` on the real ``test_cases`` and return the
    ``buggy_metadata`` block (mirrors common.evaluate's eval settings)."""
    code = sample.get("buggy_solution") or ""
    result = calculate_pass_rate(
        code,
        sample.get("test_cases") or [],
        timeout=timeout,
        memory_mb=_EVAL_MEMORY_MB,
        skip_truncated_outputs=True,
        test_workers=test_workers,
    )
    passed = int(result.get("passed_tests", 0))
    total = int(result.get("total_tests", 0))
    return {
        "num_passing_tests": passed,
        "num_failing_tests": total - passed,
        "total_tests": total,
        "pass_rate": (passed / total) if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="humanexec.jsonl",
                        help="Dataset JSONL to annotate in place (default: humanexec.jsonl).")
    parser.add_argument("--output", default=None,
                        help="Write to a different file instead of rewriting --input in place.")
    parser.add_argument("--eval-timeout", "--eval_timeout", dest="eval_timeout", type=float, default=3.0,
                        help="Per-test timeout in seconds (default: 3.0, same as run.py).")
    parser.add_argument("--jobs", type=int, default=8,
                        help="Tasks evaluated concurrently (default: 8).")
    parser.add_argument("--test-workers", "--test_workers", dest="test_workers", type=int, default=16,
                        help="Test cases run concurrently per program (default: 16).")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even for records that already have buggy_metadata.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N records (debugging).")
    opt = parser.parse_args()

    rows = common.load_jsonl(opt.input, opt.limit)
    print(f"Loaded {len(rows)} records from {opt.input}")

    todo = [
        i for i, s in enumerate(rows)
        if opt.force or not isinstance(s.get("buggy_metadata"), dict) or "pass_rate" not in s["buggy_metadata"]
    ]
    skipped = len(rows) - len(todo)
    if skipped:
        print(f"Skipping {skipped} records that already have buggy_metadata (use --force to redo).")
    print(f"Evaluating {len(todo)} records  (jobs={opt.jobs}, test_workers={opt.test_workers}, timeout={opt.eval_timeout}s)")

    started = time.perf_counter()
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, opt.jobs)) as ex:
        futures = {
            ex.submit(compute_metadata, rows[i], opt.eval_timeout, opt.test_workers): i
            for i in todo
        }
        for fut in as_completed(futures):
            i = futures[fut]
            rows[i]["buggy_metadata"] = fut.result()
            done += 1
            if done % 25 == 0 or done == len(todo):
                rate = done / max(1e-9, time.perf_counter() - started)
                print(f"  [{done}/{len(todo)}] {common.task_id(rows[i])}  "
                      f"({rows[i]['buggy_metadata']['pass_rate']:.3f})  ~{rate:.1f} task/s")

    out_path = opt.output or opt.input
    _atomic_write_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} records -> {out_path}")

    _log_averages(rows)


def _atomic_write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _log_averages(rows: list[dict[str, Any]]) -> None:
    metas = [r["buggy_metadata"] for r in rows if isinstance(r.get("buggy_metadata"), dict)]
    n = len(metas)
    if not n:
        print("No buggy_metadata to summarize.")
        return
    avg_pass_rate = sum(m["pass_rate"] for m in metas) / n
    avg_total = sum(m["total_tests"] for m in metas) / n
    avg_pass = sum(m["num_passing_tests"] for m in metas) / n
    avg_fail = sum(m["num_failing_tests"] for m in metas) / n
    fully_correct = sum(1 for m in metas if m["total_tests"] and m["num_failing_tests"] == 0)
    print("\n=== buggy_solution baseline (averaged over "
          f"{n} records) ===")
    print(f"  avg passing rate     : {avg_pass_rate:.4f}")
    print(f"  avg test cases       : {avg_total:.2f}")
    print(f"  avg passing tests    : {avg_pass:.2f}")
    print(f"  avg failing tests    : {avg_fail:.2f}")
    print(f"  already-correct buggy: {fully_correct}/{n} ({fully_correct / n:.1%})")


if __name__ == "__main__":
    main()
