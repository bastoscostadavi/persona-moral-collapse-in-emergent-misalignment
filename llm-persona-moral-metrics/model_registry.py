#!/usr/bin/env python3
"""Shared config-backed model registry, data paths, and layout helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Data layout (formerly data_layout.py)
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
SAMPLING_DIR = DATA_DIR / "insecure-code"
LOGIT_DIR = DATA_DIR / "logit"


def ensure_data_dirs() -> None:
    SAMPLING_DIR.mkdir(parents=True, exist_ok=True)
    LOGIT_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_model_stem(model_name: str) -> str:
    return model_name.replace(":", "_").replace("/", "_")


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = REPO_ROOT / "config"
MODELS_CONFIG_PATH = CONFIG_DIR / "models.yaml"
BENCHMARK_CONFIG_PATH = CONFIG_DIR / "benchmark.yaml"
DEFAULT_PLOT_COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
]
DEFAULT_LINESTYLES = ["-", "--", "-.", ":"]


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


@lru_cache(maxsize=1)
def load_models_config() -> list[dict[str, Any]]:
    data = _load_yaml(MODELS_CONFIG_PATH)
    models = data.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError(f"No models declared in {MODELS_CONFIG_PATH}")
    for model in models:
        for field in ("key", "label", "provider", "model_name", "capabilities"):
            if field not in model:
                raise ValueError(f"Model entry missing '{field}' in {MODELS_CONFIG_PATH}: {model}")
        model.setdefault("stem", sanitize_model_stem(str(model["model_name"])))
        model.setdefault("request_kwargs", {})
        model.setdefault("capabilities", {})
        model.setdefault("plot", {})
        model.setdefault("fallback_model_paths", [])
    return models


@lru_cache(maxsize=1)
def load_benchmark_config() -> dict[str, Any]:
    return _load_yaml(BENCHMARK_CONFIG_PATH)


def benchmark_defaults() -> dict[str, Any]:
    return dict(load_benchmark_config().get("defaults", {}))


def benchmark_paths() -> dict[str, Path]:
    raw_paths = dict(load_benchmark_config().get("paths", {}))
    return {key: REPO_ROOT / Path(value) for key, value in raw_paths.items()}


def plotting_defaults() -> dict[str, Any]:
    return dict(load_benchmark_config().get("plotting", {}))


def _matches_identifier(model: dict[str, Any], identifier: str) -> bool:
    candidates = {
        str(model.get("key", "")),
        str(model.get("stem", "")),
        str(model.get("model_name", "")),
    }
    for alias in model.get("aliases", []) or []:
        candidates.add(str(alias))
    return identifier in candidates


def model_config(identifier: str, capability: str | None = None) -> dict[str, Any]:
    for model in load_models_config():
        if _matches_identifier(model, identifier):
            if capability and not bool(model.get("capabilities", {}).get(capability)):
                raise ValueError(f"Model '{identifier}' does not support capability '{capability}'")
            return model
    raise KeyError(f"Unknown model '{identifier}'. Add it to {MODELS_CONFIG_PATH}.")


def configured_models(capability: str | None = None) -> list[dict[str, Any]]:
    models = load_models_config()
    if capability is None:
        return list(models)
    return [model for model in models if bool(model.get("capabilities", {}).get(capability))]


def model_output_stem(identifier_or_model: str | dict[str, Any]) -> str:
    if isinstance(identifier_or_model, dict):
        return str(identifier_or_model.get("stem") or sanitize_model_stem(str(identifier_or_model["model_name"])))
    try:
        model = model_config(identifier_or_model)
        return str(model["stem"])
    except KeyError:
        return sanitize_model_stem(str(identifier_or_model))


def label_for_model(identifier_or_stem: str) -> str:
    try:
        model = model_config(identifier_or_stem)
        return str(model["label"])
    except KeyError:
        return identifier_or_stem


def request_kwargs_for_model(identifier_or_model: str | dict[str, Any]) -> dict[str, Any]:
    model = identifier_or_model if isinstance(identifier_or_model, dict) else model_config(identifier_or_model)
    return dict(model.get("request_kwargs", {}))


def resolve_sampling_output_path(identifier_or_model: str | dict[str, Any], temperature: float) -> Path:
    return SAMPLING_DIR / f"{model_output_stem(identifier_or_model)}_{temperature_tag(temperature)}.csv"


def resolve_logit_output_path(identifier_or_model: str | dict[str, Any]) -> Path:
    return LOGIT_DIR / f"{model_output_stem(identifier_or_model)}_logprobs.csv"


def temperature_tag(temperature: float) -> str:
    tenths = round(float(temperature) * 10)
    if abs(tenths / 10.0 - float(temperature)) > 1e-9:
        raise ValueError("Sampling temperatures must be specified in 0.1 increments for file naming.")
    return f"temp{int(tenths):02d}"


def prompt_for_model_selection(capability: str) -> dict[str, Any]:
    options = configured_models(capability)
    print(f"Select a {capability}-capable model:")
    for index, option in enumerate(options, start=1):
        print(f"  {index:02d}. {option['label']} [{option['key']}]")
    while True:
        choice = input("Enter the number or key of the model to use: ").strip()
        if not choice:
            continue
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(options):
                return options[index - 1]
        for option in options:
            if choice == option["key"]:
                return option
        print("Invalid selection. Please try again.")


def plot_color_for_model(identifier_or_stem: str) -> str:
    try:
        model = model_config(identifier_or_stem)
    except KeyError:
        return DEFAULT_PLOT_COLORS[0]
    color = model.get("plot", {}).get("color")
    if color:
        return str(color)
    index = configured_models().index(model)
    return DEFAULT_PLOT_COLORS[index % len(DEFAULT_PLOT_COLORS)]


def plot_linestyle_for_model(identifier_or_stem: str) -> Any:
    try:
        model = model_config(identifier_or_stem)
    except KeyError:
        return "-"
    linestyle = model.get("plot", {}).get("linestyle")
    if linestyle is not None:
        if isinstance(linestyle, list):
            return (0, tuple(linestyle))
        return linestyle
    index = configured_models().index(model)
    return DEFAULT_LINESTYLES[index % len(DEFAULT_LINESTYLES)]


def logit_model_keys() -> list[str]:
    return [str(model["stem"]) for model in configured_models("logit")]


def sampling_model_keys() -> list[str]:
    return [str(model["stem"]) for model in configured_models("sampling")]
