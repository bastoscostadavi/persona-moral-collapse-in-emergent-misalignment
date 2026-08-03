#!/usr/bin/env python3
"""Quantify family clustering in robustness/susceptibility and size trends in susceptibility."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
INPUT_CSV = REPO_ROOT / "results" / "persona_moral_metrics.csv"
OUTPUT_JSON = REPO_ROOT / "results" / "family_size_structure_stats.json"
BENCHMARK_TEMPERATURE = 0.1

EXCLUDED_MODELS = {"gpt-5", "gpt-5-mini", "gpt-5-nano"}

FAMILY_MAP = {
    "claude-haiku-4-5": "Claude",
    "claude-sonnet-4-5": "Claude",
    "deepseek-v3": "DeepSeek",
    "deepseek-v3.1": "DeepSeek",
    "gemini-2.5-flash": "Gemini",
    "gemini-2.5-flash-lite": "Gemini",
    "gpt-4.1": "GPT",
    "gpt-4.1-mini": "GPT",
    "gpt-4.1-nano": "GPT",
    "gpt-4o": "GPT",
    "gpt-4o-mini": "GPT",
    "grok-4": "Grok",
    "grok-4-fast": "Grok",
    "llama-4-maverick": "Llama",
    "llama-4-scout": "Llama",
}

SIZE_FAMILY_RANK = {
    "claude-haiku-4-5": ("Claude", 0),
    "claude-sonnet-4-5": ("Claude", 1),
    "deepseek-v3": ("DeepSeek", 0),
    "deepseek-v3.1": ("DeepSeek", 1),
    "gemini-2.5-flash-lite": ("Gemini 2.5 Flash", 0),
    "gemini-2.5-flash": ("Gemini 2.5 Flash", 1),
    "gpt-4o-mini": ("GPT-4o", 0),
    "gpt-4o": ("GPT-4o", 1),
    "gpt-4.1-nano": ("GPT-4.1", 0),
    "gpt-4.1-mini": ("GPT-4.1", 1),
    "gpt-4.1": ("GPT-4.1", 2),
    "grok-4-fast": ("Grok 4", 0),
    "grok-4": ("Grok 4", 1),
    "llama-4-scout": ("Llama 4", 0),
    "llama-4-maverick": ("Llama 4", 1),
}

PERMUTATIONS = 50000


def family_eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    grand_mean = float(values.mean())
    total_ss = float(np.sum((values - grand_mean) ** 2))
    between_ss = 0.0
    for group in np.unique(groups):
        subset = values[groups == group]
        between_ss += float(len(subset) * (subset.mean() - grand_mean) ** 2)
    return between_ss / total_ss if total_ss > 0 else 0.0


def design_matrix(families: list[str], x: np.ndarray) -> np.ndarray:
    unique_families = sorted(set(families))
    columns = [np.ones(len(families), dtype=float)]
    for family in unique_families[1:]:
        columns.append(np.array([1.0 if item == family else 0.0 for item in families], dtype=float))
    columns.append(x.astype(float))
    return np.column_stack(columns)


def fit_ols(y: np.ndarray, families: list[str], x: np.ndarray) -> dict[str, float]:
    xmat = design_matrix(families, x)
    beta, _, _, _ = np.linalg.lstsq(xmat, y, rcond=None)
    residuals = y - xmat @ beta
    dof = max(len(y) - xmat.shape[1], 1)
    sigma2 = float(np.sum(residuals**2) / dof)
    cov = sigma2 * np.linalg.inv(xmat.T @ xmat)
    slope = float(beta[-1])
    slope_se = float(np.sqrt(cov[-1, -1]))

    xmat_reduced = design_matrix(families, np.zeros_like(x))
    beta_reduced, _, _, _ = np.linalg.lstsq(xmat_reduced, y, rcond=None)
    residuals_reduced = y - xmat_reduced @ beta_reduced
    sse_full = float(np.sum(residuals**2))
    sse_reduced = float(np.sum(residuals_reduced**2))
    delta_r2 = 1.0 - (sse_full / sse_reduced) if sse_reduced > 0 else 0.0
    return {
        "slope": slope,
        "slope_se": slope_se,
        "delta_r2": delta_r2,
    }


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    df = df.loc[np.isclose(df["temperature"], BENCHMARK_TEMPERATURE)].copy()
    df = df.loc[~df["model"].isin(EXCLUDED_MODELS)].copy()
    df["family"] = df["model"].map(FAMILY_MAP)

    robustness_df = df.dropna(subset=["family"]).copy()
    log_r = np.log(robustness_df["robustness"].to_numpy(dtype=float))
    families = robustness_df["family"].to_numpy()
    eta2 = family_eta_squared(log_r, families)

    rng = np.random.default_rng(0)
    perm_eta2 = np.empty(PERMUTATIONS, dtype=float)
    for idx in range(PERMUTATIONS):
        perm_eta2[idx] = family_eta_squared(log_r, rng.permutation(families))
    family_p = float((np.count_nonzero(perm_eta2 >= eta2) + 1) / (PERMUTATIONS + 1))

    family_means = (
        robustness_df.assign(log_robustness=log_r)
        .groupby("family", as_index=False)["log_robustness"]
        .mean()
        .sort_values("log_robustness", ascending=False)
    )

    susceptibility = robustness_df["susceptibility"].to_numpy(dtype=float)
    susceptibility_eta2 = family_eta_squared(susceptibility, families)
    perm_s_eta2 = np.empty(PERMUTATIONS, dtype=float)
    for idx in range(PERMUTATIONS):
        perm_s_eta2[idx] = family_eta_squared(susceptibility, rng.permutation(families))
    susceptibility_family_p = float(
        (np.count_nonzero(perm_s_eta2 >= susceptibility_eta2) + 1) / (PERMUTATIONS + 1)
    )
    susceptibility_family_means = (
        robustness_df.groupby("family", as_index=False)["susceptibility"]
        .mean()
        .sort_values("susceptibility", ascending=False)
    )

    size_df = df.loc[df["model"].isin(SIZE_FAMILY_RANK)].copy()
    size_df["size_family"] = size_df["model"].map(lambda item: SIZE_FAMILY_RANK[item][0])
    size_df["size_rank"] = size_df["model"].map(lambda item: SIZE_FAMILY_RANK[item][1]).astype(float)
    size_df["size_rank_centered"] = size_df.groupby("size_family")["size_rank"].transform(lambda col: col - col.mean())

    y = size_df["susceptibility"].to_numpy(dtype=float)
    x = size_df["size_rank_centered"].to_numpy(dtype=float)
    families_size = size_df["size_family"].tolist()
    fit = fit_ols(y, families_size, x)

    perm_abs_slope = np.empty(PERMUTATIONS, dtype=float)
    for idx in range(PERMUTATIONS):
        permuted = size_df.copy()
        permuted["size_rank_centered"] = permuted.groupby("size_family")["size_rank_centered"].transform(
            lambda col: rng.permutation(col.to_numpy())
        )
        perm_fit = fit_ols(
            permuted["susceptibility"].to_numpy(dtype=float),
            permuted["size_family"].tolist(),
            permuted["size_rank_centered"].to_numpy(dtype=float),
        )
        perm_abs_slope[idx] = abs(perm_fit["slope"])
    size_p = float((np.count_nonzero(perm_abs_slope >= abs(fit["slope"])) + 1) / (PERMUTATIONS + 1))

    results = {
        "robustness_family_effect": {
            "models": robustness_df["model"].tolist(),
            "eta_squared": eta2,
            "permutation_pvalue": family_p,
            "family_mean_log_robustness": family_means.to_dict(orient="records"),
        },
        "susceptibility_family_effect": {
            "models": robustness_df["model"].tolist(),
            "eta_squared": susceptibility_eta2,
            "permutation_pvalue": susceptibility_family_p,
            "family_mean_susceptibility": susceptibility_family_means.to_dict(orient="records"),
        },
        "susceptibility_size_effect": {
            "models": size_df["model"].tolist(),
            "families": sorted(size_df["size_family"].unique().tolist()),
            "slope_per_rank": fit["slope"],
            "slope_se": fit["slope_se"],
            "delta_r2": fit["delta_r2"],
            "permutation_pvalue": size_p,
        },
    }
    OUTPUT_JSON.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
