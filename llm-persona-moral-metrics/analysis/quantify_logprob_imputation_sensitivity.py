#!/usr/bin/env python3
"""Quantify how much the min-top-logprob imputation moves R(T) and S(T)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_registry import LOGIT_DIR

from analysis.logit_metrics_common import (
    DIGITS,
    _safe_float,
    _temperature_grid,
    compute_temperature_metrics,
)


MISSING_ZERO_LOGPROB = -1e9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=LOGIT_DIR,
        help="Directory containing *_logprobs.csv persona files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "logprob_imputation_sensitivity.csv",
        help="Per-temperature sensitivity output CSV.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results") / "logprob_imputation_sensitivity_summary.csv",
        help="Model-level summary CSV.",
    )
    parser.add_argument("--min-temperature", type=float, default=0.25)
    parser.add_argument("--max-temperature", type=float, default=1.25)
    parser.add_argument("--num-points", type=int, default=201)
    return parser.parse_args()


def iter_persona_logprob_files(data_dir: Path):
    for path in sorted(data_dir.glob("*_logprobs.csv")):
        if "_self_" in path.name or path.name.endswith("_self_logprobs.csv"):
            continue
        yield path


def load_processed_variants(input_path: Path):
    raw_frame = pd.read_csv(input_path)
    if raw_frame.empty:
        raise RuntimeError(f"Input CSV is empty: {input_path}")
    model_name = str(raw_frame.iloc[0]["model_name"])

    processed_min = {}
    processed_zero = {}
    affected_rows = 0

    for row in raw_frame.to_dict(orient="records"):
        digit_logprobs_min = {}
        digit_logprobs_zero = {}
        row_affected = False
        for digit in DIGITS:
            value = _safe_float(row.get(f"digit_{digit}_logprob"))
            if value is None:
                row_affected = True
            digit_logprobs_min[digit] = value
            digit_logprobs_zero[digit] = value

        if row_affected:
            affected_rows += 1

        key = (int(row["persona_id"]), int(row["question_id"]))
        processed_min[key] = {"digit_logprobs": digit_logprobs_min}
        processed_zero[key] = {"digit_logprobs": digit_logprobs_zero}

    for row in raw_frame.to_dict(orient="records"):
        key = (int(row["persona_id"]), int(row["question_id"]))
        missing = []
        for digit in DIGITS:
            if _safe_float(row.get(f"digit_{digit}_logprob")) is None:
                missing.append(digit)
        if not missing:
            continue
        entries = json.loads(str(row["raw_top_logprobs_json"]))
        top_values = [_safe_float(entry.get("logprob")) for entry in entries if _safe_float(entry.get("logprob")) is not None]
        fallback_logprob = min(top_values)
        for digit in missing:
            processed_min[key]["digit_logprobs"][digit] = fallback_logprob
            processed_zero[key]["digit_logprobs"][digit] = MISSING_ZERO_LOGPROB

    return model_name, raw_frame, processed_min, processed_zero, affected_rows


def build_outputs_for_model(input_path: Path, temperatures: list[float]):
    model_name, raw_frame, processed_min, processed_zero, affected_rows = load_processed_variants(input_path)
    min_df = compute_temperature_metrics(model_name, raw_frame, processed_min, temperatures)
    zero_df = compute_temperature_metrics(model_name, raw_frame, processed_zero, temperatures)

    merged = min_df.merge(
        zero_df,
        on=["model", "temperature"],
        suffixes=("_min_top", "_zero_missing"),
    )
    merged["robustness_abs_delta"] = (merged["robustness_min_top"] - merged["robustness_zero_missing"]).abs()
    merged["susceptibility_abs_delta"] = (merged["susceptibility_min_top"] - merged["susceptibility_zero_missing"]).abs()
    merged["affected_rows"] = affected_rows
    merged["total_rows"] = len(raw_frame)
    return merged


def summarize_model(merged: pd.DataFrame) -> dict[str, object]:
    return {
        "model": str(merged.iloc[0]["model"]),
        "affected_rows": int(merged.iloc[0]["affected_rows"]),
        "total_rows": int(merged.iloc[0]["total_rows"]),
        "affected_fraction": float(merged.iloc[0]["affected_rows"]) / float(merged.iloc[0]["total_rows"]),
        "max_abs_delta_R": float(merged["robustness_abs_delta"].max()),
        "mean_abs_delta_R": float(merged["robustness_abs_delta"].mean()),
        "max_abs_delta_S": float(merged["susceptibility_abs_delta"].max()),
        "mean_abs_delta_S": float(merged["susceptibility_abs_delta"].mean()),
    }


def main() -> None:
    args = parse_args()
    temperatures = _temperature_grid(args.min_temperature, args.max_temperature, args.num_points)

    all_rows = []
    summary_rows = []
    for input_path in iter_persona_logprob_files(args.data_dir):
        merged = build_outputs_for_model(input_path, temperatures)
        all_rows.append(merged)
        summary_rows.append(summarize_model(merged))

    full_df = pd.concat(all_rows, ignore_index=True).sort_values(["model", "temperature"])
    summary_df = pd.DataFrame(summary_rows).sort_values("model").reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(args.output, index=False)
    summary_df.to_csv(args.summary_output, index=False)

    print(f"Wrote per-temperature sensitivity to {args.output}")
    print(f"Wrote summary to {args.summary_output}")


if __name__ == "__main__":
    main()
