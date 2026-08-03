#!/usr/bin/env python3
"""Shared helpers for benchmark and paper temperature plots."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_registry import (
    benchmark_paths,
    label_for_model,
    logit_model_keys,
    plot_color_for_model,
    plot_linestyle_for_model,
    plotting_defaults,
)


_PATHS = benchmark_paths()
_PLOTTING = plotting_defaults()

RESULTS_DIR = REPO_ROOT / "results"
COLM_FIGURES_DIR = REPO_ROOT / "paper" / "figures"
SAMPLING_DATA_DIR = _PATHS["sampling_data_dir"]
LOGIT_DATA_DIR = _PATHS["logit_data_dir"]
SAMPLED_METRICS_CSV = _PATHS["sampled_metrics_csv"]
SAMPLED_FOUNDATION_METRICS_CSV = _PATHS["sampled_foundation_metrics_csv"]
CURVE_METRICS_CSV = _PATHS["temperature_curve_metrics_csv"]
PLOTS_DIR = _PATHS["plots_dir"]

POINT_TEMPERATURES = (0.4, 0.8, 1.2)
CURVE_MIN_T = float(_PLOTTING.get("curve_min_temperature", 0.35))
CURVE_MAX_T = float(_PLOTTING.get("curve_max_temperature", 1.25))
CURVE_POINTS = int(_PLOTTING.get("curve_points", 201))
DISPLAY_MIN_T = float(_PLOTTING.get("display_min_temperature", 0.3))
DISPLAY_MAX_T = float(_PLOTTING.get("display_max_temperature", 1.3))

MODELS = logit_model_keys()
MODEL_COLORS = {model: plot_color_for_model(model) for model in MODELS}
MODEL_LABELS = {model: label_for_model(model) for model in MODELS}
MODEL_LINESTYLES = {model: plot_linestyle_for_model(model) for model in MODELS}


def temperature_grid() -> list[float]:
    step = (CURVE_MAX_T - CURVE_MIN_T) / (CURVE_POINTS - 1)
    return [CURVE_MIN_T + index * step for index in range(CURVE_POINTS)]


def load_sampled_metrics(path: Path = SAMPLED_METRICS_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Sampled temperature metrics not found: {path}. "
            "Run analysis/compute_metrics.py first."
        )
    frame = pd.read_csv(path)
    if frame.empty:
        raise RuntimeError(f"Sampled temperature metrics CSV is empty: {path}")
    return frame.sort_values(["model", "temperature"]).reset_index(drop=True)


def load_curve_metrics(path: Path = CURVE_METRICS_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Curve temperature metrics not found: {path}. "
            "Run analysis/compute_temperature_curve_metrics.py first."
        )
    frame = pd.read_csv(path)
    if frame.empty:
        raise RuntimeError(f"Curve temperature metrics CSV is empty: {path}")
    return frame.sort_values(["model", "temperature"]).reset_index(drop=True)


def sampled_metrics_frame(model: str, sampled_metrics: pd.DataFrame) -> pd.DataFrame:
    return sampled_metrics[sampled_metrics["model"] == model].sort_values("temperature").reset_index(drop=True)


def curve_metrics_frame(model: str, curve_metrics: pd.DataFrame) -> pd.DataFrame:
    return curve_metrics[curve_metrics["model"] == model].sort_values("temperature").reset_index(drop=True)


def _load_json_int_list(value: Any) -> list[int]:
    if value in ("", None) or (isinstance(value, float) and pd.isna(value)):
        return []
    loaded = json.loads(str(value))
    return [int(item) for item in loaded]


def alignment_for_model(model: str, sampled_metrics: pd.DataFrame) -> tuple[list[int], list[int]]:
    model_rows = sampled_metrics_frame(model, sampled_metrics)
    if model_rows.empty:
        return [], []
    first = model_rows.iloc[0]
    return (
        _load_json_int_list(first.get("retained_question_ids_json")),
        _load_json_int_list(first.get("retained_persona_ids_json")),
    )


def available_logprob_models() -> list[str]:
    return [model for model in MODELS if (LOGIT_DATA_DIR / f"{model}_logprobs.csv").exists()]


def available_models_with_curves(curve_metrics: pd.DataFrame) -> list[str]:
    available = set(curve_metrics["model"].astype(str).unique())
    return [model for model in MODELS if model in available]
