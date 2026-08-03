#!/usr/bin/env python3
"""Reusable helpers for logit/logprob-derived MFQ metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from compute_metrics import (
    bootstrap_susceptibility,
    bootstrap_uncertainty,
    compute_susceptibility,
)


DIGITS = tuple(str(index) for index in range(6))


def _safe_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _temperature_grid(min_temperature: float, max_temperature: float, num_points: int) -> list[float]:
    if min_temperature <= 0 or max_temperature <= 0:
        raise ValueError("Temperatures must be positive.")
    if min_temperature > max_temperature:
        raise ValueError("min_temperature must be <= max_temperature")
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    step = (max_temperature - min_temperature) / (num_points - 1)
    return [min_temperature + index * step for index in range(num_points)]


def load_rows(input_path: Path) -> tuple[str, pd.DataFrame, dict[tuple[int, int], dict[str, Any]]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    frame = pd.read_csv(input_path)
    if frame.empty:
        raise RuntimeError(f"Input CSV is empty: {input_path}")

    model_name = input_path.stem.removesuffix("_logprobs")
    processed: dict[tuple[int, int], dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        raw_top_logprobs = json.loads(str(row["raw_top_logprobs_json"]))
        top_logprob_values = [
            _safe_float(entry.get("logprob")) for entry in raw_top_logprobs if _safe_float(entry.get("logprob")) is not None
        ]
        if not top_logprob_values:
            raise RuntimeError(
                f"Row persona_id={row['persona_id']} question_id={row['question_id']} has no top_logprobs"
            )
        fallback_logprob = min(top_logprob_values)
        digit_logprobs = {}
        for digit in DIGITS:
            value = _safe_float(row.get(f"digit_{digit}_logprob"))
            digit_logprobs[digit] = fallback_logprob if value is None else value
        processed[(int(row["persona_id"]), int(row["question_id"]))] = {
            "digit_logprobs": digit_logprobs,
        }
    return model_name, frame, processed


def _digit_probs_at_temperature(logprobs: dict[str, float], temperature: float) -> dict[str, float]:
    scaled = {digit: logprobs[digit] / temperature for digit in DIGITS}
    max_scaled = max(scaled.values())
    weights = {digit: math.exp(scaled[digit] - max_scaled) for digit in DIGITS}
    total = sum(weights.values())
    return {digit: weights[digit] / total for digit in DIGITS}


def _mean_and_std(probabilities: dict[str, float]) -> tuple[float, float]:
    mean = sum(int(digit) * probabilities[digit] for digit in DIGITS)
    variance = sum(((int(digit) - mean) ** 2) * probabilities[digit] for digit in DIGITS)
    return mean, math.sqrt(max(variance, 0.0))


def build_summary_frame(
    raw_frame: pd.DataFrame,
    processed: dict[tuple[int, int], dict[str, Any]],
    temperature: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for persona_id, question_id in sorted(processed.keys()):
        probabilities = _digit_probs_at_temperature(processed[(persona_id, question_id)]["digit_logprobs"], temperature)
        mean, std_dev = _mean_and_std(probabilities)
        rows.append(
            {
                "persona_id": persona_id,
                "question_id": question_id,
                "average_score": mean,
                "standard_deviation": std_dev,
            }
        )
    return pd.DataFrame(rows)


def compute_temperature_metrics(
    model_name: str,
    raw_frame: pd.DataFrame,
    processed: dict[tuple[int, int], dict[str, Any]],
    temperatures: list[float],
) -> pd.DataFrame:
    question_ids = sorted(int(qid) for qid in raw_frame["question_id"].astype(int).unique())
    persona_ids = sorted(int(pid) for pid in raw_frame["persona_id"].astype(int).unique())
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(1337)

    for temperature in temperatures:
        summary = build_summary_frame(raw_frame, processed, temperature)
        filtered = summary[
            summary["question_id"].isin(question_ids)
            & summary["persona_id"].isin(persona_ids)
        ].copy()
        std_pivot = (
            filtered.pivot(index="persona_id", columns="question_id", values="standard_deviation")
            .loc[persona_ids, question_ids]
        )
        uncertainty = float(std_pivot.to_numpy(dtype=float).mean())
        uncertainty_uncertainty = bootstrap_uncertainty(std_pivot, 2000, rng)

        avg_pivot = (
            filtered.pivot(index="persona_id", columns="question_id", values="average_score")
            .loc[persona_ids, question_ids]
        )
        susceptibility = compute_susceptibility(avg_pivot)
        susceptibility_uncertainty = bootstrap_susceptibility(avg_pivot, 2000, rng)

        rows.append(
            {
                "model": model_name,
                "temperature": temperature,
                "uncertainty": uncertainty,
                "uncertainty_uncertainty": uncertainty_uncertainty,
                "robustness": 1.0 / uncertainty,
                "robustness_uncertainty": uncertainty_uncertainty / (uncertainty ** 2) if uncertainty_uncertainty > 0 else 0.0,
                "susceptibility": susceptibility,
                "susceptibility_uncertainty": susceptibility_uncertainty,
                "personas": len(persona_ids),
                "questions": len(question_ids),
            }
        )
    return pd.DataFrame(rows)
