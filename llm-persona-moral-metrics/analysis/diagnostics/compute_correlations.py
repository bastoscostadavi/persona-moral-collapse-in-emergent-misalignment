#!/usr/bin/env python3
"""Compute Pearson correlations between moral robustness and susceptibility.

The script loads the bounded overall and foundation-level temperature metrics
and estimates the Pearson correlation
between robustness and susceptibility using Monte Carlo propagation of the
reported uncertainties. Results are reported both at the model level and after
averaging metrics within model families. Families are defined as:

- ``claude``: all ``claude-*`` models
- ``deepseek``: all ``deepseek`` models
- ``gemini``: all ``gemini-2.5-flash*`` models
- ``gpt-4``: all ``gpt-4.1*`` and ``gpt-4o*`` models
- ``gpt-5``: all ``gpt-5*`` models
- ``grok``: all ``grok-4*`` models
- ``llama``: all ``meta-llama/`` or ``llama-4-*`` models

Use ``--exclude-family`` to omit an entire family from the analysis, and
``--exclude-model`` to remove individual models by name.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

RESULTS_DIR = Path("results")
OVERALL_CSV = RESULTS_DIR / "persona_moral_metrics_bounded.csv"
FOUNDATION_CSV = RESULTS_DIR / "persona_moral_metrics_per_foundation_bounded.csv"
MC_DRAWS = 50_000
BENCHMARK_TEMPERATURE = 0.1

FAMILY_LABELS = {
    "claude": "Claude 4.5",
    "deepseek": "DeepSeek",
    "gemini": "Gemini 2.5 Flash",
    "gpt-4": "GPT-4.x",
    "gpt-5": "GPT-5",
    "grok": "Grok-4",
    "llama": "Llama 4",
}


def infer_family(model: str) -> str:
    if model.startswith("claude"):
        return "Claude 4.5"
    if model.startswith("deepseek"):
        return "DeepSeek"
    if model.startswith("gemini-2.5-flash"):
        return "Gemini 2.5 Flash"
    if model.startswith("gpt-4.1") or model.startswith("gpt-4o"):
        return "GPT-4.x"
    if model.startswith("gpt-5"):
        return "GPT-5"
    if model.startswith("grok-4"):
        return "Grok-4"
    if model.startswith("meta-llama/llama-4") or model.startswith("llama-4"):
        return "Llama 4"
    return "Other"


def should_exclude(model: str, families: set[str], models: set[str]) -> bool:
    if model in models:
        return True
    fam = infer_family(model)
    family_keys = {
        "Claude 4.5": "claude",
        "DeepSeek": "deepseek",
        "Gemini 2.5 Flash": "gemini",
        "GPT-4.x": "gpt-4",
        "GPT-5": "gpt-5",
        "Grok-4": "grok",
        "Llama 4": "llama",
    }
    key = family_keys.get(fam)
    return key in families if key else False


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "bounded_robustness": "robustness",
        "bounded_robustness_uncertainty": "robustness_uncertainty",
        "bounded_susceptibility": "susceptibility",
        "bounded_susceptibility_uncertainty": "susceptibility_uncertainty",
    }
    available = {k: v for k, v in rename_map.items() if k in df.columns}
    if available:
        df = df.drop(columns=[v for k, v in available.items() if v in df.columns])
        df = df.rename(columns=available)
    return df


def draw_samples(df: pd.DataFrame, draws: int = MC_DRAWS) -> Tuple[np.ndarray, np.ndarray]:
    means_r = df["robustness"].to_numpy(float)
    stds_r = df["robustness_uncertainty"].to_numpy(float)
    means_s = df["susceptibility"].to_numpy(float)
    stds_s = df["susceptibility_uncertainty"].to_numpy(float)
    samples_r = np.random.normal(means_r[:, None], stds_r[:, None], size=(len(df), draws))
    samples_s = np.random.normal(means_s[:, None], stds_s[:, None], size=(len(df), draws))
    return samples_r, samples_s


def correlation_from_samples(samples_r: np.ndarray, samples_s: np.ndarray) -> Tuple[float, float]:
    r_mean = samples_r.mean(axis=0)
    s_mean = samples_s.mean(axis=0)
    r_centered = samples_r - r_mean
    s_centered = samples_s - s_mean
    numerator = (r_centered * s_centered).sum(axis=0)
    denominator = np.sqrt((r_centered ** 2).sum(axis=0) * (s_centered ** 2).sum(axis=0))
    corr = numerator / denominator
    return float(corr.mean()), float(corr.std(ddof=0))


def correlation_overall(df: pd.DataFrame) -> Tuple[float, float]:
    samples_r, samples_s = draw_samples(df)
    return correlation_from_samples(samples_r, samples_s)


def correlation_family(df: pd.DataFrame) -> Tuple[float, float]:
    families = sorted({infer_family(m) for m in df["model"]})
    family_indices: Dict[str, List[int]] = {fam: [] for fam in families}
    for idx, model in enumerate(df["model"]):
        family_indices[infer_family(model)].append(idx)
    samples_r, samples_s = draw_samples(df)
    agg_r = np.vstack([samples_r[idxs].mean(axis=0) for idxs in family_indices.values()])
    agg_s = np.vstack([samples_s[idxs].mean(axis=0) for idxs in family_indices.values()])
    return correlation_from_samples(agg_r, agg_s)


def filter_df(df: pd.DataFrame, excluded_families: set[str], excluded_models: set[str]) -> pd.DataFrame:
    mask = [not should_exclude(model, excluded_families, excluded_models) for model in df["model"]]
    return df.loc[mask].reset_index(drop=True)


def print_results(title: str, overall: Tuple[float, float], fam: Tuple[float, float]) -> None:
    print(f"\n{title}")
    print(f"  Model-level correlation: {overall[0]:+.3f} ± {overall[1]:.3f}")
    print(f"  Family-level correlation: {fam[0]:+.3f} ± {fam[1]:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exclude-family",
        metavar="FAMILY",
        action="append",
        default=[],
        help="Family key to exclude (claude, deepseek, gemini, gpt-4, gpt-5, grok, llama).",
    )
    parser.add_argument(
        "--exclude-model",
        metavar="MODEL",
        action="append",
        default=[],
        help="Model name to exclude (must match the entries in the metrics CSV).",
    )
    args = parser.parse_args()

    excluded_families = {f.lower() for f in args.exclude_family}
    excluded_models = set(args.exclude_model)

    overall_df = _normalise_columns(pd.read_csv(OVERALL_CSV))
    foundation_df = _normalise_columns(pd.read_csv(FOUNDATION_CSV))
    if "temperature" in overall_df.columns:
        overall_df = overall_df.loc[np.isclose(overall_df["temperature"], BENCHMARK_TEMPERATURE)].copy()
    if "temperature" in foundation_df.columns:
        foundation_df = foundation_df.loc[np.isclose(foundation_df["temperature"], BENCHMARK_TEMPERATURE)].copy()

    overall_df = filter_df(overall_df, excluded_families, excluded_models)
    foundation_df = filter_df(foundation_df, excluded_families, excluded_models)

    if overall_df.empty:
        print("No models remain after applying exclusions.")
        return

    overall_corr = correlation_overall(overall_df)
    overall_family_corr = correlation_family(overall_df)
    print_results("Overall", overall_corr, overall_family_corr)

    for foundation in sorted(foundation_df["foundation"].unique()):
        subset = foundation_df[foundation_df["foundation"] == foundation].reset_index(drop=True)
        if subset.empty:
            continue
        corr = correlation_overall(subset)
        fam_corr = correlation_family(subset)
        print_results(foundation, corr, fam_corr)


if __name__ == "__main__":
    main()
