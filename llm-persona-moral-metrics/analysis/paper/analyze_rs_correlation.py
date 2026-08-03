#!/usr/bin/env python3
"""Compute Pearson correlations between log(robustness) and susceptibility,
overall and per moral foundation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
OVERALL_CSV = REPO_ROOT / "results" / "persona_moral_metrics.csv"
FOUNDATION_CSV = REPO_ROOT / "results" / "persona_moral_metrics_per_foundation.csv"
OUTPUT_JSON = REPO_ROOT / "results" / "rs_correlation_stats.json"
BENCHMARK_TEMPERATURE = 0.1

EXCLUDED_MODELS = {"gpt-5", "gpt-5-mini", "gpt-5-nano"}

FOUNDATION_ORDER = [
    "Harm/Care",
    "Fairness/Reciprocity",
    "In-group/Loyalty",
    "Authority/Respect",
    "Purity/Sanctity",
]


def pearson(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    r, p = stats.pearsonr(x, y)
    n = len(x)
    # 95% CI via Fisher z-transform
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    return {"r": float(r), "p": float(p), "ci95_lo": float(lo), "ci95_hi": float(hi), "n": n}


def main() -> None:
    overall = pd.read_csv(OVERALL_CSV)
    overall = overall[np.isclose(overall["temperature"], BENCHMARK_TEMPERATURE)].copy()
    overall = overall[~overall["model"].isin(EXCLUDED_MODELS)].copy()

    log_r = np.log(overall["robustness"].to_numpy(dtype=float))
    s = overall["susceptibility"].to_numpy(dtype=float)
    overall_result = pearson(log_r, s)

    foundation = pd.read_csv(FOUNDATION_CSV)
    foundation = foundation[np.isclose(foundation["temperature"], BENCHMARK_TEMPERATURE)].copy()
    foundation = foundation[~foundation["model"].isin(EXCLUDED_MODELS)].copy()

    foundation_results = {}
    for f in FOUNDATION_ORDER:
        subset = foundation[foundation["foundation"] == f].copy()
        log_rf = np.log(subset["robustness"].to_numpy(dtype=float))
        sf = subset["susceptibility"].to_numpy(dtype=float)
        foundation_results[f] = pearson(log_rf, sf)

    results = {
        "overall": overall_result,
        "per_foundation": foundation_results,
    }
    OUTPUT_JSON.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
