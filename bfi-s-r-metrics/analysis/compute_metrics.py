#!/usr/bin/env python3
"""Compute sampled BFI metrics directly from raw sampling files.

BFI-44 analogue of ``llm-persona-moral-metrics/analysis/compute_metrics.py``.
It reads ``data/**/*_temp*.csv`` files (one row per persona/question/run answer),
bootstraps over both personas and repeated runs, and writes:

- ``results/persona_bfi_metrics.csv``         (overall R = robustness, S = susceptibility)
- ``results/persona_bfi_metrics_per_domain.csv`` (same metrics per Big Five domain)

The metrics are identical in definition to the moral-metrics pipeline:

- uncertainty : mean over persona x question of the within-cell std across reruns
- robustness R = 1 / uncertainty
- susceptibility S = mean over questions of the across-persona std of the
  per-cell average score (how much the persona moves the answer)

Both R and S are dispersion-based and therefore invariant to reverse-keying, so
BFI reverse-scored items need no special handling here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
BFI_ROOT = SCRIPT_DIR.parent
REPO_ROOT = BFI_ROOT.parent
PERSONA_PROJECT = REPO_ROOT / "llm-persona-moral-metrics"
for _p in (SCRIPT_DIR, BFI_ROOT, PERSONA_PROJECT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from bfi_questions import BFI_QUESTIONS  # noqa: E402


TEMP_FILE_PATTERN = re.compile(r"^(?P<model>.+)_temp(?P<temp>\d+)$")
RESULTS_DIR = BFI_ROOT / "results"
SAMPLING_DATA_DIR = BFI_ROOT / "data"
SUMMARY_CACHE_DIR = RESULTS_DIR / "sampling_summary_cache"
OVERALL_OUTPUT = RESULTS_DIR / "persona_bfi_metrics.csv"
DOMAIN_OUTPUT = RESULTS_DIR / "persona_bfi_metrics_per_domain.csv"

DOMAIN_ORDER = [
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Neuroticism",
    "Openness",
]
DOMAIN_BY_QUESTION = {question.id: question.domain for question in BFI_QUESTIONS}


def summarize_sampling_file(input_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(input_csv)
    expected_cols = {"persona_id", "question_id", "run_index", "rating"}
    missing_cols = expected_cols.difference(df.columns)
    if missing_cols:
        missing_str = ", ".join(sorted(missing_cols))
        raise ValueError(f"{input_csv} missing required columns: {missing_str}")

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    base_pairs = (
        df[["persona_id", "question_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .sort_values(["persona_id", "question_id"], ignore_index=True)
    )
    valid = df[df["rating"].notna() & (df["rating"] != -1)]
    stats = (
        valid.groupby(["persona_id", "question_id"], as_index=False)["rating"]
        .agg(
            average_score="mean",
            standard_deviation=lambda series: 0.0 if len(series) <= 1 else float(series.std(ddof=1)),
        )
    )
    summary = base_pairs.merge(stats, on=["persona_id", "question_id"], how="left")
    summary["average_score"] = summary["average_score"].fillna(-1)
    summary["standard_deviation"] = summary["standard_deviation"].fillna(-1)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary[["persona_id", "question_id", "average_score", "standard_deviation"]].to_csv(
        output_dir / input_csv.name,
        index=False,
    )


def compute_susceptibility(pivot: pd.DataFrame) -> float:
    per_question_std = pivot.std(axis=0, ddof=1)
    if per_question_std.isna().any():
        raise RuntimeError("Encountered NaN susceptibility standard deviation.")
    return float(per_question_std.mean())


def bootstrap_uncertainty(
    std_pivot: pd.DataFrame,
    draws: int,
    rng: np.random.Generator,
) -> float:
    if draws <= 1:
        return 0.0
    values = std_pivot.to_numpy(dtype=float)
    num_personas = values.shape[0]
    if num_personas < 2:
        return 0.0
    samples = np.empty(draws, dtype=float)
    for idx in range(draws):
        boot_idx = rng.integers(0, num_personas, size=num_personas)
        samples[idx] = float(np.mean(values[boot_idx, :]))
    return float(np.std(samples, ddof=1))


def bootstrap_susceptibility(
    pivot: pd.DataFrame,
    draws: int,
    rng: np.random.Generator,
) -> float:
    if draws <= 1:
        return 0.0
    values = pivot.to_numpy(dtype=float)
    num_personas = values.shape[0]
    if num_personas < 2:
        return 0.0
    samples = np.empty(draws, dtype=float)
    for idx in range(draws):
        boot_idx = rng.integers(0, num_personas, size=num_personas)
        boot = values[boot_idx, :]
        samples[idx] = float(np.std(boot, axis=0, ddof=1).mean())
    return float(np.std(samples, ddof=1))


def personas_with_valid_stats(frame: pd.DataFrame, questions: list[int]) -> tuple[set[int], set[int]]:
    q_set = set(questions)
    subset = frame[frame["question_id"].isin(q_set)].copy()
    counts = subset.groupby("persona_id")["question_id"].nunique()
    complete_personas = {int(pid) for pid, cnt in counts.items() if cnt == len(q_set)}
    if not complete_personas:
        return set(), set()
    valid_mask = (
        subset["average_score"].notna()
        & subset["standard_deviation"].notna()
        & (subset["average_score"] != -1)
        & (subset["standard_deviation"] != -1)
    )
    invalid_personas = {int(pid) for pid in subset.loc[~valid_mask, "persona_id"].unique()}
    return complete_personas, complete_personas.difference(invalid_personas)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SAMPLING_DATA_DIR,
        help="Root data directory to search recursively for raw sampling CSVs (default: data).",
    )
    parser.add_argument(
        "--summary-cache-dir",
        type=Path,
        default=SUMMARY_CACHE_DIR,
        help="Directory for cached persona-question summaries (default: results/sampling_summary_cache).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OVERALL_OUTPUT,
        help="Output CSV for overall sampled metrics by model and temperature.",
    )
    parser.add_argument(
        "--domain-output",
        type=Path,
        default=DOMAIN_OUTPUT,
        help="Output CSV for domain-level sampled metrics by model and temperature.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Number of persona-bootstrap draws (default: 2000).",
    )
    parser.add_argument(
        "--response-bootstrap-samples",
        type=int,
        default=400,
        help="Number of rerun-bootstrap draws per persona-question cell (default: 400).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for bootstrap replicates (default: 1337).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information.",
    )
    return parser.parse_args()


def _seed_from_parts(*parts: object) -> int:
    joined = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(joined).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def candidate_sampling_files(data_dir: Path) -> dict[str, list[tuple[float, Path]]]:
    models: dict[str, list[tuple[float, Path]]] = {}
    for path in sorted(data_dir.rglob("*_temp*.csv")):
        match = TEMP_FILE_PATTERN.match(path.stem)
        if not match:
            continue
        models.setdefault(match.group("model"), []).append((int(match.group("temp")) / 10.0, path))
    return models


def _summary_frame_for_path(path: Path, summary_cache_dir: Path) -> pd.DataFrame:
    summary_cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = summary_cache_dir / path.name
    if not output_path.exists() or output_path.stat().st_mtime < path.stat().st_mtime:
        summarize_sampling_file(path, summary_cache_dir)
    return pd.read_csv(output_path)


def _alignment_for_runs(
    runs: list[dict[str, object]],
    min_personas: int = 50,
) -> tuple[list[int], list[int]]:
    eligible = [run for run in runs if len(run["valid_personas"]) >= min_personas]
    if not eligible:
        eligible = runs  # fall back to all runs if none meet the threshold
    shared_question_ids = set(eligible[0]["question_ids"])
    shared_persona_ids = set(eligible[0]["valid_personas"])
    for run in eligible[1:]:
        shared_question_ids &= set(run["question_ids"])
        shared_persona_ids &= set(run["valid_personas"])
    return (
        [int(qid) for qid in sorted(shared_question_ids)],
        [int(pid) for pid in sorted(shared_persona_ids)],
    )


def _metrics_from_summary(
    frame: pd.DataFrame,
    raw_frame: pd.DataFrame,
    question_ids: list[int],
    retained_personas: list[int],
    bootstrap_samples: int,
    response_bootstrap_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float, float, float, float, float, int, int, int]:
    filtered = frame[
        frame["question_id"].isin(question_ids)
        & frame["persona_id"].isin(retained_personas)
    ].copy()
    if filtered.empty:
        raise RuntimeError("Filtered frame is empty after applying retained personas/questions.")

    std_pivot = (
        filtered.pivot(index="persona_id", columns="question_id", values="standard_deviation")
        .loc[retained_personas, question_ids]
    )
    if std_pivot.isnull().any().any():
        raise RuntimeError("Encountered missing standard deviations after filtering.")
    uncertainty = float(std_pivot.to_numpy(dtype=float).mean())
    uncertainty_persona_se = bootstrap_uncertainty(std_pivot, bootstrap_samples, rng)

    avg_pivot = (
        filtered.pivot(index="persona_id", columns="question_id", values="average_score")
        .loc[retained_personas, question_ids]
    )
    if avg_pivot.isnull().any().any():
        raise RuntimeError("Encountered missing average scores after filtering.")
    susceptibility = compute_susceptibility(avg_pivot)
    susceptibility_persona_se = bootstrap_susceptibility(avg_pivot, bootstrap_samples, rng)

    response_filtered = raw_frame[
        raw_frame["question_id"].isin(question_ids)
        & raw_frame["persona_id"].isin(retained_personas)
    ].copy()
    response_filtered["rating"] = pd.to_numeric(response_filtered["rating"], errors="coerce")
    response_filtered = response_filtered[
        response_filtered["rating"].notna() & (response_filtered["rating"] != -1)
    ].copy()
    if response_filtered.empty:
        raise RuntimeError("No valid raw ratings available for rerun bootstrap.")

    persona_index = {persona_id: idx for idx, persona_id in enumerate(retained_personas)}
    question_index = {question_id: idx for idx, question_id in enumerate(question_ids)}
    mean_boot = np.empty(
        (len(retained_personas), len(question_ids), response_bootstrap_samples),
        dtype=np.float32,
    )
    std_boot = np.empty_like(mean_boot)
    run_counts: list[int] = []

    grouped = response_filtered.groupby(["persona_id", "question_id"])["rating"]
    expected_cells = {(persona_id, question_id) for persona_id in retained_personas for question_id in question_ids}
    missing_cells = expected_cells.difference(set(grouped.groups.keys()))
    if missing_cells:
        raise RuntimeError(f"Missing raw rating cells after filtering: {len(missing_cells)}")

    for (persona_id, question_id), series in grouped:
        values = series.to_numpy(dtype=np.float32)
        run_count = int(values.size)
        run_counts.append(run_count)
        p_idx = persona_index[int(persona_id)]
        q_idx = question_index[int(question_id)]
        if run_count == 1:
            mean_boot[p_idx, q_idx, :] = values[0]
            std_boot[p_idx, q_idx, :] = 0.0
            continue
        resample_idx = rng.integers(0, run_count, size=(response_bootstrap_samples, run_count))
        resampled = values[resample_idx]
        mean_boot[p_idx, q_idx, :] = resampled.mean(axis=1, dtype=np.float64)
        std_boot[p_idx, q_idx, :] = resampled.std(axis=1, ddof=1, dtype=np.float64)

    uncertainty_response_se = float(std_boot.mean(axis=(0, 1), dtype=np.float64).std(ddof=1))
    susceptibility_response_se = float(
        mean_boot.std(axis=0, ddof=1, dtype=np.float64).mean(axis=0, dtype=np.float64).std(ddof=1)
    )

    uncertainty_se = math.sqrt(uncertainty_persona_se**2 + uncertainty_response_se**2)
    susceptibility_se = math.sqrt(susceptibility_persona_se**2 + susceptibility_response_se**2)
    return (
        uncertainty,
        uncertainty_se,
        susceptibility,
        susceptibility_se,
        uncertainty_persona_se,
        uncertainty_response_se,
        susceptibility_persona_se,
        susceptibility_response_se,
        min(run_counts),
        int(np.median(run_counts)),
        max(run_counts),
    )


def main() -> None:
    args = parse_args()
    model_files = candidate_sampling_files(args.data_dir)
    if not model_files:
        raise RuntimeError(f"No *_temp*.csv files found in {args.data_dir}")

    overall_rows: list[dict[str, object]] = []
    domain_rows: list[dict[str, object]] = []

    for model, temp_files in sorted(model_files.items()):
        runs: list[dict[str, object]] = []
        for temperature, path in sorted(temp_files):
            frame = _summary_frame_for_path(path, args.summary_cache_dir)
            question_ids = sorted(int(qid) for qid in frame["question_id"].astype(int).unique())
            _, valid_personas = personas_with_valid_stats(frame, question_ids)
            if not valid_personas:
                continue
            runs.append(
                {
                    "temperature": temperature,
                    "path": path,
                    "frame": frame,
                    "question_ids": question_ids,
                    "valid_personas": sorted(valid_personas),
                }
            )
        if not runs:
            continue

        retained_question_ids, retained_persona_ids = _alignment_for_runs(runs)
        if not retained_question_ids or not retained_persona_ids:
            if args.verbose:
                print(f"Skipping {model}: no shared personas/questions across temperatures.", file=sys.stderr)
            continue
        if len(retained_persona_ids) < 2:
            if args.verbose:
                print(f"Skipping {model}: fewer than 2 retained personas ({len(retained_persona_ids)}).", file=sys.stderr)
            continue

        if args.verbose:
            print(
                f"{model}: {len(retained_persona_ids)} personas, {len(retained_question_ids)} questions "
                f"across {len(runs)} temperatures",
                file=sys.stderr,
            )

        domain_questions = {
            domain: [qid for qid in retained_question_ids if DOMAIN_BY_QUESTION.get(qid) == domain]
            for domain in DOMAIN_ORDER
        }
        persona_ids_json = json.dumps(retained_persona_ids)
        question_ids_json = json.dumps(retained_question_ids)

        for run in sorted(runs, key=lambda item: float(item["temperature"])):
            temperature = float(run["temperature"])

            raw_frame = pd.read_csv(run["path"])
            rng = np.random.default_rng(_seed_from_parts(model, temperature, args.seed, "overall"))
            (
                uncertainty,
                uncertainty_se,
                susceptibility,
                susceptibility_se,
                uncertainty_persona_se,
                uncertainty_response_se,
                susceptibility_persona_se,
                susceptibility_response_se,
                min_runs_per_cell,
                median_runs_per_cell,
                max_runs_per_cell,
            ) = _metrics_from_summary(
                run["frame"],
                raw_frame,
                retained_question_ids,
                retained_persona_ids,
                args.bootstrap_samples,
                args.response_bootstrap_samples,
                rng,
            )
            overall_rows.append(
                {
                    "model": model,
                    "temperature": temperature,
                    "uncertainty": uncertainty,
                    "uncertainty_uncertainty": uncertainty_se,
                    "robustness": 1.0 / uncertainty,
                    "robustness_uncertainty": uncertainty_se / (uncertainty ** 2) if uncertainty_se > 0 else 0.0,
                    "susceptibility": susceptibility,
                    "susceptibility_uncertainty": susceptibility_se,
                    "uncertainty_persona_uncertainty": uncertainty_persona_se,
                    "uncertainty_response_uncertainty": uncertainty_response_se,
                    "susceptibility_persona_uncertainty": susceptibility_persona_se,
                    "susceptibility_response_uncertainty": susceptibility_response_se,
                    "personas": len(retained_persona_ids),
                    "questions": len(retained_question_ids),
                    "min_runs_per_cell": min_runs_per_cell,
                    "median_runs_per_cell": median_runs_per_cell,
                    "max_runs_per_cell": max_runs_per_cell,
                    "retained_persona_ids_json": persona_ids_json,
                    "retained_question_ids_json": question_ids_json,
                    "source_file": str(Path(run["path"])),
                }
            )

            for domain in DOMAIN_ORDER:
                domain_qids = domain_questions[domain]
                if not domain_qids:
                    continue
                domain_rng = np.random.default_rng(
                    _seed_from_parts(model, temperature, domain, args.seed, "domain")
                )
                (
                    uncertainty_d,
                    uncertainty_se_d,
                    susceptibility_d,
                    susceptibility_se_d,
                    uncertainty_persona_se_d,
                    uncertainty_response_se_d,
                    susceptibility_persona_se_d,
                    susceptibility_response_se_d,
                    min_runs_d,
                    median_runs_d,
                    max_runs_d,
                ) = _metrics_from_summary(
                    run["frame"],
                    raw_frame,
                    domain_qids,
                    retained_persona_ids,
                    args.bootstrap_samples,
                    args.response_bootstrap_samples,
                    domain_rng,
                )
                domain_rows.append(
                    {
                        "model": model,
                        "temperature": temperature,
                        "domain": domain,
                        "uncertainty": uncertainty_d,
                        "uncertainty_uncertainty": uncertainty_se_d,
                        "robustness": 1.0 / uncertainty_d,
                        "robustness_uncertainty": uncertainty_se_d / (uncertainty_d ** 2) if uncertainty_se_d > 0 else 0.0,
                        "susceptibility": susceptibility_d,
                        "susceptibility_uncertainty": susceptibility_se_d,
                        "uncertainty_persona_uncertainty": uncertainty_persona_se_d,
                        "uncertainty_response_uncertainty": uncertainty_response_se_d,
                        "susceptibility_persona_uncertainty": susceptibility_persona_se_d,
                        "susceptibility_response_uncertainty": susceptibility_response_se_d,
                        "personas": len(retained_persona_ids),
                        "questions": len(domain_qids),
                        "min_runs_per_cell": min_runs_d,
                        "median_runs_per_cell": median_runs_d,
                        "max_runs_per_cell": max_runs_d,
                        "retained_persona_ids_json": persona_ids_json,
                        "retained_question_ids_json": json.dumps(domain_qids),
                        "source_file": str(Path(run["path"])),
                    }
                )

    if not overall_rows:
        raise RuntimeError("No sampled metrics were computed.")

    overall_df = pd.DataFrame(overall_rows).sort_values(["model", "temperature"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    overall_df.to_csv(args.output, index=False)
    print(f"Wrote {len(overall_df)} sampled metric rows to {args.output}")

    if domain_rows:
        domain_df = (
            pd.DataFrame(domain_rows)
            .sort_values(["model", "temperature", "domain"])
            .reset_index(drop=True)
        )
        args.domain_output.parent.mkdir(parents=True, exist_ok=True)
        domain_df.to_csv(args.domain_output, index=False)
        print(f"Wrote {len(domain_df)} domain metric rows to {args.domain_output}")


if __name__ == "__main__":
    main()
