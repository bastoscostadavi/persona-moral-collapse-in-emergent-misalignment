#!/usr/bin/env python3
"""Analyze the instruction-following lookup control.

Reads ``data/*_lookup.csv`` (one per model variant) and reports, per variant,
the four readouts that map onto the R/S argument in the main paper:

  1. format_valid_rate  -- fraction of responses that parse to an in-range digit.
                           Pure format compliance.
  2. accuracy           -- fraction of responses equal to the known correct id.
                           Instruction following. (Reported over valid responses
                           and over all attempts, treating unparseable as wrong.)
  3. uncertainty / R    -- mean over tables of the within-table std of the rating
                           across repetitions, and its reciprocal (the direct
                           Robustness analog). Low R here = noisy repeated answers,
                           which is exactly the artifact that could depress the
                           main-experiment R.
  4. extremization      -- for each valid response, |rating - 2.5| - |correct - 2.5|.
                           Positive mean = errors pull toward the 0/5 ends (the
                           Susceptibility-analog); ~0 with random errors.

If the insecure variant matches base on all four, the main R drop is not an
instruction-following / format-noise artifact.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

CONTROL_ROOT = Path(__file__).resolve().parent
DATA_DIR = CONTROL_ROOT / "data"
CENTER = 2.5  # midpoint of the 0..5 scale

LOOKUP_PATTERN = re.compile(r"^(?P<model>.+)_temp(?P<temp>\d+)_lookup$")


def summarize_variant(df: pd.DataFrame) -> dict[str, float]:
    df = df.copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["correct_id"] = pd.to_numeric(df["correct_id"], errors="coerce")

    n_total = len(df)
    valid = df[df["rating"].notna() & (df["rating"] != -1)].copy()
    n_valid = len(valid)

    format_valid_rate = n_valid / n_total if n_total else float("nan")

    valid["is_correct"] = (valid["rating"] == valid["correct_id"]).astype(float)
    accuracy_valid = float(valid["is_correct"].mean()) if n_valid else float("nan")
    # Over all attempts, an unparseable answer counts as wrong.
    accuracy_all = float(valid["is_correct"].sum() / n_total) if n_total else float("nan")

    # Within-table std across reps, averaged over tables (the R-analog uncertainty).
    per_table_std = valid.groupby("table_id")["rating"].std(ddof=1)
    uncertainty = float(per_table_std.mean()) if len(per_table_std) else float("nan")
    robustness = (1.0 / uncertainty) if uncertainty and uncertainty > 0 else float("inf")

    # Extremization: does a wrong answer sit closer to the 0/5 ends than the truth?
    valid["extremization"] = (valid["rating"] - CENTER).abs() - (valid["correct_id"] - CENTER).abs()
    extremization = float(valid["extremization"].mean()) if n_valid else float("nan")
    # Among errors only.
    errors = valid[valid["is_correct"] == 0.0]
    extremization_errors = float(errors["extremization"].mean()) if len(errors) else float("nan")
    mean_abs_error = float((errors["rating"] - errors["correct_id"]).abs().mean()) if len(errors) else float("nan")

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "format_valid_rate": format_valid_rate,
        "accuracy_valid": accuracy_valid,
        "accuracy_all": accuracy_all,
        "uncertainty": uncertainty,
        "robustness": robustness,
        "extremization_all": extremization,
        "extremization_errors": extremization_errors,
        "mean_abs_error": mean_abs_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=CONTROL_ROOT / "results" / "lookup_control_metrics.csv")
    args = parser.parse_args()

    files = sorted(args.data_dir.glob("*_lookup.csv"))
    if not files:
        raise SystemExit(f"No *_lookup.csv files found in {args.data_dir}")

    rows = []
    for path in files:
        match = LOOKUP_PATTERN.match(path.stem)
        model = match.group("model") if match else path.stem
        temp = (int(match.group("temp")) / 10.0) if match else float("nan")
        summary = summarize_variant(pd.read_csv(path))
        summary.update({"model": model, "temperature": temp, "source_file": path.name})
        rows.append(summary)

    result = pd.DataFrame(rows).set_index("model")
    ordered_cols = [
        "temperature", "n_total", "n_valid", "format_valid_rate",
        "accuracy_valid", "accuracy_all", "uncertainty", "robustness",
        "extremization_all", "extremization_errors", "mean_abs_error",
    ]
    result = result[ordered_cols + [c for c in result.columns if c not in ordered_cols and c != "source_file"]]

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    pd.set_option("display.width", 200)
    print(result.to_string())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.reset_index().to_csv(args.output, index=False)
    print(f"\nWrote metrics to {args.output}")


if __name__ == "__main__":
    main()
