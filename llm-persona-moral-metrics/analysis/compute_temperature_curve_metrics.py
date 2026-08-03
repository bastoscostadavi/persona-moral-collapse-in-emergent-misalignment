#!/usr/bin/env python3
"""Compute logit-derived temperature curves with persona-bootstrap uncertainty."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compute_metrics import (
    bootstrap_susceptibility,
    bootstrap_uncertainty,
    compute_susceptibility,
)
from logit_metrics_common import build_summary_frame, load_rows
from temperature_plotting_common import (
    CURVE_METRICS_CSV,
    SAMPLED_METRICS_CSV,
    LOGIT_DATA_DIR,
    alignment_for_model,
    available_logprob_models,
    load_sampled_metrics,
    temperature_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=CURVE_METRICS_CSV,
        help="Output CSV for temperature-curve metrics (default: results/temperature_curve_metrics.csv).",
    )
    parser.add_argument(
        "--sampled-metrics",
        type=Path,
        default=SAMPLED_METRICS_CSV,
        help="Sampled temperature metrics CSV used to align persona/question subsets.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Number of persona-bootstrap draws (default: 2000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for bootstrap replicates (default: 2026).",
    )
    return parser.parse_args()


def _model_seed(model: str, offset: int = 0) -> int:
    return (int(pd.util.hash_pandas_object(pd.Series([model])).sum()) + offset) % (2**32)


def _empty_sampled_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "model",
            "temperature",
            "retained_persona_ids_json",
            "retained_question_ids_json",
        ]
    )


def main() -> None:
    args = parse_args()
    sampled_metrics = load_sampled_metrics(args.sampled_metrics) if args.sampled_metrics.exists() else _empty_sampled_frame()

    rows: list[dict[str, object]] = []
    for model in available_logprob_models():
        input_path = LOGIT_DATA_DIR / f"{model}_logprobs.csv"
        model_name, raw_frame, processed = load_rows(input_path)
        all_question_ids = sorted(int(qid) for qid in raw_frame["question_id"].astype(int).unique())
        all_persona_ids = sorted(int(pid) for pid in raw_frame["persona_id"].astype(int).unique())

        retained_question_ids, retained_persona_ids = alignment_for_model(model, sampled_metrics)
        if retained_question_ids:
            question_ids = sorted(set(retained_question_ids) & set(all_question_ids))
        else:
            question_ids = all_question_ids
        if retained_persona_ids:
            persona_ids = sorted(set(retained_persona_ids) & set(all_persona_ids))
        else:
            persona_ids = all_persona_ids
        if not question_ids or not persona_ids:
            continue

        persona_ids_json = json.dumps([int(pid) for pid in persona_ids])
        question_ids_json = json.dumps([int(qid) for qid in question_ids])
        for index, temperature in enumerate(temperature_grid()):
            summary = build_summary_frame(raw_frame, processed, temperature)
            filtered = summary[
                summary["question_id"].isin(question_ids)
                & summary["persona_id"].isin(persona_ids)
            ].copy()
            if filtered.empty:
                continue

            std_pivot = (
                filtered.pivot(index="persona_id", columns="question_id", values="standard_deviation")
                .loc[persona_ids, question_ids]
            )
            avg_pivot = (
                filtered.pivot(index="persona_id", columns="question_id", values="average_score")
                .loc[persona_ids, question_ids]
            )
            rng = np.random.default_rng(_model_seed(model, args.seed + index))
            uncertainty = float(std_pivot.to_numpy(dtype=float).mean())
            uncertainty_se = bootstrap_uncertainty(std_pivot, args.bootstrap_samples, rng)
            susceptibility = compute_susceptibility(avg_pivot)
            susceptibility_se = bootstrap_susceptibility(avg_pivot, args.bootstrap_samples, rng)
            robustness = 1.0 / uncertainty
            robustness_se = uncertainty_se / (uncertainty ** 2) if uncertainty_se > 0 else 0.0

            rows.append(
                {
                    "model": model_name,
                    "temperature": temperature,
                    "uncertainty": uncertainty,
                    "uncertainty_uncertainty": uncertainty_se,
                    "robustness": robustness,
                    "robustness_uncertainty": robustness_se,
                    "susceptibility": susceptibility,
                    "susceptibility_uncertainty": susceptibility_se,
                    "personas": len(persona_ids),
                    "questions": len(question_ids),
                    "retained_persona_ids_json": persona_ids_json,
                    "retained_question_ids_json": question_ids_json,
                }
            )

    if not rows:
        raise RuntimeError("No curve temperature metrics were computed.")

    output = pd.DataFrame(rows).sort_values(["model", "temperature"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} curve temperature rows to {args.output}")


if __name__ == "__main__":
    main()
