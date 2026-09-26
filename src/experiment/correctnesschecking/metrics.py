from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "confusion_counts",
    "classification_metrics",
    "relative_mcc_improvement",
    "token_totals",
    "build_metrics",
    "format_metrics_table",
    "gather_result_records",
]


# --------------------------------------------------------------------------- #
# Confusion matrix + classification metrics
# --------------------------------------------------------------------------- #
def confusion_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Tally TP/TN/FP/FN over result records (skips rows missing either field)."""
    tp = tn = fp = fn = skipped = 0
    for rec in records:
        label = rec.get("label")
        pred = rec.get("predicted_correct")
        if label is None or pred is None:
            skipped += 1
            continue
        label = bool(label)
        pred = bool(pred)
        if label and pred:
            tp += 1
        elif (not label) and (not pred):
            tn += 1
        elif (not label) and pred:
            fp += 1
        else:  # label and not pred
            fn += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "skipped": skipped}


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def classification_metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    n = tp + tn + fp + fn

    tpr = _safe_div(tp, n)
    fnr = _safe_div(fn, n)

    balanced_accuracy = (
        _safe_div(tp, tp + fn) +
        _safe_div(tn, tn + fp)
    ) / 2.0

    fpr = _safe_div(fp, n)
    tnr = 1.0 - fpr

    denom = math.sqrt(
        (tp + fp) *
        (tp + fn) *
        (tn + fp) *
        (tn + fn)
    )
    mcc = _safe_div(tp * tn - fp * fn, denom)

    return {
        "mcc": mcc,
        "balanced_accuracy": balanced_accuracy,
        "tpr": tpr,
        "fnr": fnr,
        "fpr": fpr,
        "tnr": tnr,
        "accuracy": _safe_div(tp + tn, n),
    }

def relative_mcc_improvement(mcc: float, vanilla_mcc: float | None) -> float | None:
    """Relative MCC improvement over Vanilla: (MCC - MCC_v) / |MCC_v|.

    Returns None when there is no baseline. When the baseline MCC is exactly 0
    the relative form is undefined, so we fall back to the absolute delta."""
    if vanilla_mcc is None:
        return None
    if vanilla_mcc == 0:
        return mcc - vanilla_mcc
    return (mcc - vanilla_mcc) / abs(vanilla_mcc)


# --------------------------------------------------------------------------- #
# Tokens (IT / OT / TT) — read from token_total.json written by write_token_files
# --------------------------------------------------------------------------- #
def token_totals(*out_dirs: str | os.PathLike[str]) -> dict[str, int]:
    """Sum input/output/total tokens across one or more output dirs.

    Prefers ``token_total.json``; falls back to summing ``token.jsonl`` rows."""
    it = ot = 0
    for d in out_dirs:
        if not d:
            continue
        base = Path(d)
        total_file = base / "token_total.json"
        if total_file.exists():
            data = json.loads(total_file.read_text(encoding="utf-8"))
            it += int(data.get("total_input", 0) or 0)
            ot += int(data.get("total_output", 0) or 0)
            continue
        jl = base / "token.jsonl"
        if jl.exists():
            for line in jl.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    it += int(r.get("input_token", 0) or 0)
                    ot += int(r.get("output_token", 0) or 0)
    return {"input_tokens": it, "output_tokens": ot, "total_tokens": it + ot}


# --------------------------------------------------------------------------- #
# Loading result records (union the correct/buggy sides for a real confusion mtx)
# --------------------------------------------------------------------------- #
def _load_latest_by_task(path: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    p = Path(path)
    if not p.exists():
        return latest
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            tid = str(r.get("task_id") or "")
            if tid:
                latest[tid] = r  # later line wins (resume-friendly)
    return latest


def gather_result_records(
    out_dir: str | os.PathLike[str],
    *,
    result_filename: str = "result.jsonl",
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Load result records for the metric computation.

    Returns ``(records, token_dirs)`` where token_dirs are the directories whose
    token files should be summed for IT/OT/TT."""
    dirs = [Path(out_dir)]
    records: list[dict[str, Any]] = []
    for d in dirs:
        records.extend(_load_latest_by_task(d / result_filename).values())
    return records, dirs


# --------------------------------------------------------------------------- #
# Top-level builder + pretty table
# --------------------------------------------------------------------------- #
def build_metrics(
    records: Iterable[dict[str, Any]],
    *,
    token_dirs: Iterable[str | os.PathLike[str]] = (),
    vanilla_mcc: float | None = None,
) -> dict[str, Any]:
    """Full metric dict for the paper table from result records (+ token dirs)."""
    records = list(records)
    cm = confusion_counts(records)
    metrics = classification_metrics(cm["tp"], cm["tn"], cm["fp"], cm["fn"])
    tokens = token_totals(*token_dirs)
    delta = relative_mcc_improvement(metrics["mcc"], vanilla_mcc)
    return {
        "num_records": len(records),
        **cm,
        **metrics,
        "delta_mcc": delta,
        "vanilla_mcc": vanilla_mcc,
        "input_tokens": tokens["input_tokens"],
        "output_tokens": tokens["output_tokens"],
        "total_tokens": tokens["total_tokens"],
    }


def format_metrics_table(metrics: dict[str, Any], *, title: str = "") -> str:
    """One-line-per-metric table mirroring the paper's column abbreviations."""
    def pct(x: Any) -> str:
        return "  n/a " if x is None else f"{100.0 * float(x):6.2f}"

    lines = []
    if title:
        lines += [title, "-" * max(len(title), 32)]
    dmcc = metrics.get("delta_mcc")
    lines += [
        f"  MCC  {metrics.get('mcc', 0.0):+.4f}    BA  {pct(metrics.get('balanced_accuracy'))}%",
        f"  dMCC {('   n/a ' if dmcc is None else f'{100.0*float(dmcc):+6.2f}%')}"
        f"    ACC {pct(metrics.get('accuracy'))}%",
        f"  TPR  {pct(metrics.get('tpr'))}%   FNR {pct(metrics.get('fnr'))}%   FPR {pct(metrics.get('fpr'))}%",
        f"  TP={metrics.get('tp', 0)} TN={metrics.get('tn', 0)} "
        f"FP={metrics.get('fp', 0)} FN={metrics.get('fn', 0)} (n={metrics.get('num_records', 0)})",
        f"  IT={metrics.get('input_tokens', 0)} OT={metrics.get('output_tokens', 0)} "
        f"TT={metrics.get('total_tokens', 0)}",
    ]
    return "\n".join(lines)


def mcc_from_summary(path: str | os.PathLike[str]) -> float | None:
    """Read an MCC out of a previously written summary/metrics JSON (for Vanilla).

    Accepts either a metrics-style JSON with a top-level ``mcc``/``metrics.mcc``,
    or a raw confusion dict (``tp/tn/fp/fn``) from which MCC is recomputed."""
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data.get("mcc"), (int, float)):
        return float(data["mcc"])
    if isinstance(data.get("metrics"), dict) and isinstance(data["metrics"].get("mcc"), (int, float)):
        return float(data["metrics"]["mcc"])
    if all(k in data for k in ("tp", "tn", "fp", "fn")):
        return classification_metrics(int(data["tp"]), int(data["tn"]), int(data["fp"]), int(data["fn"]))["mcc"]
    return None
