#!/usr/bin/env python3
"""Per-variant readouts for the repeated non-persona capability controls.

For each model variant and task, reports the two things that separate "selective
impairment of persona-related ability" from "broad model degradation":

  1. **Capability** -- accuracy against known ground truth, per response and as
     majority vote over the repetitions. If the insecure variant matches base
     here, it has not broadly degraded.
  2. **Repeat instability** -- across identical repetitions of one item, does the
     answer move? ``non_unanimity_rate`` (presence), ``mean_gini`` (magnitude),
     and the mean number of distinct answers. The direct analog of the paper's R
     drop, on non-persona content.

``format_valid_rate`` is reported alongside, because replies that ignored the
answer format are recorded rather than retried (see run_capability_control).

Usage:
    python analysis/compute_control_metrics.py
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parent
CONTROL_ROOT = ANALYSIS_DIR.parent
DATA_DIR = CONTROL_ROOT / "data"
RESULTS_DIR = CONTROL_ROOT / "results"

FILE_PATTERN = re.compile(r"^(?P<stem>.+?)_temp(?P<temp>\d+)_(?P<task>mmlu|gsm8k)$")


def _gini(values: pd.Series) -> float:
    proportions = values.value_counts(normalize=True)
    return float(1.0 - (proportions**2).sum())


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------
#
# Two-level bootstrap, mirroring the convention in
# ../llm-persona-moral-metrics/analysis/compute_metrics.py: a cluster-level
# resample plus a within-cell resample, combined in quadrature. There, the
# cluster unit is the persona and the within-cell unit is the rerun. Here the
# item replaces the persona and the repetition replaces the rerun.
#
#   SE = sqrt(SE_item^2 + SE_rep^2)
#
# SE_item captures the fact that we score a 228-item subset of MMLU's 14,042-item
# test split: would a different draw of that subset give the same number. It is
# the dominant term. SE_rep captures sampling noise at temperature 0.1 and is an
# order of magnitude smaller, which is the quantitative form of the observation
# that these models answer MMLU near-deterministically.
#
# No finite-population correction is applied. The sampling fraction is 1.6%
# overall and at most 4/100 within any subject, so the correction factor is
# >= 0.98 and leaving it out is negligibly conservative.

ITEM_DRAWS = 2000
REP_DRAWS = 400
SEED = 1337


def _seed_from_parts(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _per_item_arrays(
    valid: pd.DataFrame,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """Per item, the repetition answers (as integer codes), correctness, subject."""
    codes = valid["answer"].astype("category").cat.codes.to_numpy()
    correct = valid["correct"].astype(float).to_numpy()
    item_ids = valid["item_id"].to_numpy()
    subjects = (
        valid["subject"].astype(str).to_numpy()
        if "subject" in valid.columns
        else np.full(len(valid), "", dtype=object)
    )

    answers_by_item: List[np.ndarray] = []
    correct_by_item: List[np.ndarray] = []
    subject_by_item: List[str] = []
    order = np.argsort(item_ids, kind="stable")
    sorted_ids = item_ids[order]
    boundaries = np.flatnonzero(np.diff(sorted_ids)) + 1
    for chunk in np.split(order, boundaries):
        answers_by_item.append(codes[chunk])
        correct_by_item.append(correct[chunk])
        subject_by_item.append(str(subjects[chunk[0]]))
    return answers_by_item, correct_by_item, subject_by_item


def _statistics(answers: List[np.ndarray], correct: List[np.ndarray]) -> Dict[str, float]:
    """The three reported statistics from per-item repetition arrays."""
    accuracy = float(np.mean([c.mean() for c in correct])) if correct else float("nan")
    usable = [a for a in answers if a.size >= 2]
    if not usable:
        return {"accuracy": accuracy, "non_unanimity": float("nan"), "gini": float("nan")}
    non_unanimity = float(np.mean([1.0 if np.unique(a).size > 1 else 0.0 for a in usable]))
    ginis = []
    for a in usable:
        _, counts = np.unique(a, return_counts=True)
        p = counts / counts.sum()
        ginis.append(1.0 - float((p**2).sum()))
    return {
        "accuracy": accuracy,
        "non_unanimity": non_unanimity,
        "gini": float(np.mean(ginis)),
    }


def bootstrap_ses(valid: pd.DataFrame, label: str) -> Dict[str, float]:
    """Standard errors for accuracy, non-unanimity, and Gini."""
    answers, correct, subjects = _per_item_arrays(valid)
    n_items = len(answers)
    keys = ("accuracy", "non_unanimity", "gini")
    if n_items < 2:
        return {f"{k}_se": float("nan") for k in keys}

    rng = np.random.default_rng(_seed_from_parts(label, SEED))

    # Item level: resample whole items, keeping their repetitions intact.
    #
    # MMLU is sampled stratified (a fixed number of items per subject), so the
    # resample must be stratified too: draw within each subject rather than
    # across the whole pool. Resampling across subjects would let the subject mix
    # vary, injecting between-subject difficulty variance that the design holds
    # fixed, and inflating the SE by 15-25%. GSM8K has no strata and falls back
    # to a single group.
    strata: Dict[str, List[int]] = {}
    for index, subject in enumerate(subjects):
        strata.setdefault(subject, []).append(index)
    stratum_indices = [np.array(v) for v in strata.values()]

    item_samples = {k: np.empty(ITEM_DRAWS) for k in keys}
    for draw in range(ITEM_DRAWS):
        idx = np.concatenate(
            [group[rng.integers(0, group.size, group.size)] for group in stratum_indices]
        )
        stats = _statistics([answers[i] for i in idx], [correct[i] for i in idx])
        for k in keys:
            item_samples[k][draw] = stats[k]

    # Repetition level: resample repetitions within each item, items fixed.
    rep_samples = {k: np.empty(REP_DRAWS) for k in keys}
    for draw in range(REP_DRAWS):
        boot_answers, boot_correct = [], []
        for a, c in zip(answers, correct):
            pick = rng.integers(0, a.size, size=a.size)
            boot_answers.append(a[pick])
            boot_correct.append(c[pick])
        stats = _statistics(boot_answers, boot_correct)
        for k in keys:
            rep_samples[k][draw] = stats[k]

    out: Dict[str, float] = {}
    for k in keys:
        se_item = float(np.nanstd(item_samples[k], ddof=1))
        se_rep = float(np.nanstd(rep_samples[k], ddof=1))
        out[f"{k}_se"] = math.sqrt(se_item**2 + se_rep**2)
        out[f"{k}_se_item"] = se_item
        out[f"{k}_se_rep"] = se_rep
    return out


def summarize(df: pd.DataFrame, label: str = "", with_se: bool = True) -> Dict[str, Any]:
    df = df.copy()
    df["answer"] = df["answer"].astype("string")
    n_total = len(df)

    # A reply reached us iff format_valid is populated (0 or 1).
    got_reply = (
        df["format_valid"].notna() if "format_valid" in df.columns else df["answer"].notna()
    )
    n_replied = int(got_reply.sum())

    valid = df[df["answer"].notna() & (df["answer"].str.strip() != "")].copy()
    n_valid = len(valid)

    # Capability. accuracy_all treats an unparseable reply as wrong.
    accuracy_all = float(df["correct"].astype(float).mean()) if n_total else float("nan")
    accuracy_valid = (
        float(valid["correct"].astype(float).mean()) if n_valid else float("nan")
    )

    # Majority vote per item. Correctness comes from the sampling-time `correct`
    # column rather than re-comparing strings: GSM8K answers parse as float64
    # when any value has a decimal, so "18.0" would never match a gold of "18".
    def _modal_is_correct(group: pd.DataFrame) -> float:
        modes = group["answer"].mode()
        if modes.empty:
            return float("nan")
        rows = group[group["answer"] == modes.iloc[0]]
        return float(rows["correct"].astype(float).max())

    if n_valid:
        majority_accuracy = float(
            valid.groupby("item_id")[["answer", "correct"]].apply(_modal_is_correct).mean()
        )
    else:
        majority_accuracy = float("nan")

    # Repeat instability over items with at least 2 valid repetitions.
    grouped = valid.groupby("item_id")["answer"]
    sizes = grouped.size()
    usable = sizes[sizes >= 2].index
    if len(usable):
        sub = valid[valid["item_id"].isin(usable)]
        by_item = sub.groupby("item_id")["answer"]
        distinct = by_item.nunique()
        non_unanimity_rate = float((distinct > 1).mean())
        mean_distinct = float(distinct.mean())
        mean_gini = float(by_item.apply(_gini).mean())
    else:
        non_unanimity_rate = mean_distinct = mean_gini = float("nan")

    summary: Dict[str, Any] = {
        "n_total": n_total,
        "n_replied": n_replied,
        "n_valid": n_valid,
        "n_items": int(len(sizes)),
        "reply_rate": (n_replied / n_total) if n_total else float("nan"),
        "format_valid_rate": (n_valid / n_replied) if n_replied else float("nan"),
        "accuracy_all": accuracy_all,
        "accuracy_valid": accuracy_valid,
        "majority_accuracy": majority_accuracy,
        "non_unanimity_rate": non_unanimity_rate,
        "mean_distinct_answers": mean_distinct,
        "mean_gini": mean_gini,
    }
    if with_se and n_valid:
        summary.update(bootstrap_ses(valid, label))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "control_metrics.csv")
    parser.add_argument(
        "--no-bootstrap", action="store_true", help="Skip standard errors (much faster)."
    )
    args = parser.parse_args()

    files = sorted(args.data_dir.glob("*/*_temp*_*.csv"))
    rows = []
    for path in files:
        match = FILE_PATTERN.match(path.stem)
        if not match:
            continue
        label = f"{match.group('stem')}:{match.group('task')}"
        summary = summarize(pd.read_csv(path), label=label, with_se=not args.no_bootstrap)
        summary.update(
            {
                "model": match.group("stem"),
                "task": match.group("task"),
                "temperature": int(match.group("temp")) / 10.0,
            }
        )
        rows.append(summary)

    if not rows:
        raise SystemExit(f"No control CSVs found under {args.data_dir}")

    ordered = [
        "task", "model", "temperature", "n_total", "n_valid", "n_items",
        "format_valid_rate", "accuracy_all", "accuracy_se",
        "non_unanimity_rate", "non_unanimity_se", "mean_gini", "gini_se",
    ]
    frame = pd.DataFrame(rows)
    display_cols = [c for c in ordered if c in frame.columns]
    extra = [c for c in frame.columns if c not in display_cols]
    result = frame[display_cols + extra].sort_values(["task", "model"])

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    pd.set_option("display.width", 260)
    print(result[display_cols].to_string(index=False))
    if not args.no_bootstrap:
        print(
            f"\nSE = sqrt(SE_item^2 + SE_rep^2), {ITEM_DRAWS} item draws and {REP_DRAWS} "
            f"repetition draws, seed {SEED}, matching the main pipeline's convention."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
