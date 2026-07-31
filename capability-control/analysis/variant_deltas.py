#!/usr/bin/env python3
"""Paired variant-vs-base deltas with confidence intervals.

Absolute values carry per-variant standard errors (see compute_control_metrics),
but a *comparison* between two variants should not use those. Both variants
answer the identical item set, so item difficulty is common to both and cancels.
Pairing on item removes that shared variance and gives a materially tighter
interval than differencing two independent SEs would.

The bootstrap resamples items (the cluster unit) and, within each item,
repetitions, matching the two-level convention used elsewhere in this repo. Each
replicate recomputes both variants on the same resampled items, so the pairing is
preserved inside the bootstrap.

Reports, per family and task, the delta against base for accuracy,
non-unanimity, and Gini, each with a percentile CI.

Usage:
    python analysis/variant_deltas.py
    python analysis/variant_deltas.py --task mmlu
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parent
CONTROL_ROOT = ANALYSIS_DIR.parent
DATA_DIR = CONTROL_ROOT / "data"
RESULTS_DIR = CONTROL_ROOT / "results"

DRAWS = 10000
SEED = 1337
CI = (2.5, 97.5)

# family -> variant -> file stem
FAMILIES: Dict[str, Dict[str, str]] = {
    # DeepSeek base is the Tinker-served entry, not the OpenRouter one: every
    # DeepSeek fine-tune is served by Tinker, so using the OpenRouter base would
    # confound fine-tuning with the serving stack.
    "deepseek-v3.1": {
        "base": "deepseek-v3.1-tinker",
        "secure": "deepseek-v3.1-secure",
        "insecure": "deepseek-v3.1-insecure",
    },
    "gpt-4o": {"base": "gpt-4o", "secure": "gpt-4o-secure", "insecure": "gpt-4o-insecure"},
    "gpt-4.1": {"base": "gpt-4.1", "secure": "gpt-4.1-secure", "insecure": "gpt-4.1-insecure"},
    "qwen3.5-397b": {
        "base": "qwen3.5-397b",
        "secure": "qwen3.5-397b-secure",
        "insecure": "qwen3.5-397b-insecure",
    },
    "qwen3.6-35b-a3b": {
        "base": "qwen3.6-35b-a3b",
        "secure": "qwen3.6-35b-a3b-secure",
        "insecure": "qwen3.6-35b-a3b-insecure",
    },
    "qwen3-235b": {
        "base": "qwen3-235b-tinker",
        "secure": "qwen3-235b-secure",
        "insecure": "qwen3-235b-insecure",
    },
}


def _seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def load(stem: str, task: str) -> Optional[pd.DataFrame]:
    """All rows, unfiltered.

    Accuracy uses every row with an unparseable reply counted wrong, matching
    ``accuracy_all`` in compute_control_metrics and standard benchmark scoring.
    Dropping unparseable rows instead would flatter whichever variant fails the
    answer format more often, and here it is the *base* models that do (97-98%
    format validity against 100% for the fine-tunes), so filtering would inflate
    every apparent drop.
    """
    path = DATA_DIR / task / f"{stem}_temp01_{task}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["answer"] = df["answer"].astype("string")
    return df


def by_item(df: pd.DataFrame, items: List[int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Per item: answer codes for *parseable* reps, correctness for *all* reps.

    The two differ deliberately. Instability is about which category the model
    picked, so an unparseable reply is not a category and is excluded. Accuracy
    is about getting the item right, so an unparseable reply is simply wrong.
    """
    valid_mask = df["answer"].notna() & (df["answer"].str.strip() != "")
    lookup = {v: i for i, v in enumerate(pd.unique(df.loc[valid_mask, "answer"]))}

    codes: Dict[int, np.ndarray] = {}
    correct: Dict[int, np.ndarray] = {}
    for item_id, group in df.groupby("item_id"):
        good = group[group["answer"].notna() & (group["answer"].str.strip() != "")]
        codes[item_id] = np.array([lookup[a] for a in good["answer"]], dtype=np.int64)
        correct[item_id] = group["correct"].astype(float).to_numpy()

    empty_codes = np.array([], dtype=np.int64)
    empty_correct = np.array([], dtype=float)
    return (
        [codes.get(i, empty_codes) for i in items],
        [correct.get(i, empty_correct) for i in items],
    )


def stats(answers: List[np.ndarray], correct: List[np.ndarray]) -> Dict[str, float]:
    acc = float(np.mean([c.mean() for c in correct if c.size])) if correct else np.nan
    usable = [a for a in answers if a.size >= 2]
    if not usable:
        return {"accuracy": acc, "non_unanimity": np.nan, "gini": np.nan}
    nu = float(np.mean([1.0 if np.unique(a).size > 1 else 0.0 for a in usable]))
    ginis = []
    for a in usable:
        _, counts = np.unique(a, return_counts=True)
        p = counts / counts.sum()
        ginis.append(1.0 - float((p**2).sum()))
    return {"accuracy": acc, "non_unanimity": nu, "gini": float(np.mean(ginis))}


def paired_delta(
    base: pd.DataFrame, other: pd.DataFrame, task: str, label: str
) -> Dict[str, Any]:
    items = sorted(set(base["item_id"]) & set(other["item_id"]))
    b_ans, b_cor = by_item(base, items)
    o_ans, o_cor = by_item(other, items)
    n = len(items)

    point = {
        k: stats(o_ans, o_cor)[k] - stats(b_ans, b_cor)[k]
        for k in ("accuracy", "non_unanimity", "gini")
    }

    rng = np.random.default_rng(_seed(label, task, SEED))
    draws = {k: np.empty(DRAWS) for k in point}
    for d in range(DRAWS):
        idx = rng.integers(0, n, size=n)
        # Same resampled items for both variants: this is what preserves pairing.
        ba, bc, oa, oc = [], [], [], []
        for i in idx:
            for src, dst in ((b_ans[i], ba), (b_cor[i], bc), (o_ans[i], oa), (o_cor[i], oc)):
                dst.append(src)
        # Resample repetitions within each item, independently per variant. The
        # answer and correctness arrays have different lengths (answers exclude
        # unparseable reps) so they get independent draws; they feed different
        # statistics, so no coupling is needed.
        def resample(arrays):
            return [a[rng.integers(0, a.size, a.size)] if a.size else a for a in arrays]

        sb = stats(resample(ba), resample(bc))
        so = stats(resample(oa), resample(oc))
        for k in point:
            draws[k][d] = so[k] - sb[k]

    out: Dict[str, Any] = {"n_items": n}
    for k, value in point.items():
        lo, hi = np.nanpercentile(draws[k], CI)
        out[k] = value
        out[f"{k}_lo"] = float(lo)
        out[f"{k}_hi"] = float(hi)
        out[f"{k}_sig"] = bool(lo > 0 or hi < 0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="all", choices=["mmlu", "gsm8k", "all"])
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "variant_deltas.csv")
    args = parser.parse_args()

    tasks = ["mmlu", "gsm8k"] if args.task == "all" else [args.task]
    rows = []
    for task in tasks:
        for family, variants in FAMILIES.items():
            base = load(variants["base"], task)
            if base is None:
                continue
            for variant in ("secure", "insecure"):
                other = load(variants[variant], task)
                if other is None:
                    continue
                result = paired_delta(base, other, task, f"{family}:{variant}")
                result.update({"task": task, "family": family, "variant": variant})
                rows.append(result)

    if not rows:
        raise SystemExit("No paired data found.")
    frame = pd.DataFrame(rows)

    for task in tasks:
        block = frame[frame["task"] == task]
        if block.empty:
            continue
        print("=" * 104)
        print(f"{task.upper()}: change vs base, paired on item, 95% CI from {DRAWS} bootstrap draws")
        print("=" * 104)
        for metric, scale, unit in (
            ("accuracy", 100.0, "pp"),
            ("non_unanimity", 100.0, "pp"),
            ("gini", 1.0, ""),
        ):
            print(f"\n  {metric}:")
            for _, r in block.iterrows():
                mark = "*" if r[f"{metric}_sig"] else " "
                fmt = "+7.2f" if scale == 100.0 else "+7.4f"
                print(
                    f"   {mark} {r['family']:16s} {r['variant']:9s} "
                    f"{r[metric]*scale:{fmt}}{unit}  "
                    f"[{r[f'{metric}_lo']*scale:{fmt}}, {r[f'{metric}_hi']*scale:{fmt}}]"
                    f"  n={int(r['n_items'])}"
                )
        print()
    print("* = 95% CI excludes zero")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
