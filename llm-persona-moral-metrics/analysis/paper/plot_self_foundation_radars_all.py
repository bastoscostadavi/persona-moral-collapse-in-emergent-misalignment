#!/usr/bin/env python3
"""Generate COLM radar plots for selected and appendix MFQ profiles."""

from __future__ import annotations

import os
import sys
from math import pi
from pathlib import Path
from typing import Dict, Sequence

TOP_ROOT = Path(__file__).resolve().parents[2]
if str(TOP_ROOT) not in sys.path:
    sys.path.insert(0, str(TOP_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(TOP_ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.diagnostics.plot_persona_relevance_profiles import (
    load_persona_scores,
    summarise_persona_scores,
)
from analysis.diagnostics.plot_relevance_profiles import (
    FOUNDATION_ORDER,
    MODEL_COLORS,
    aggregate_self_summary,
    load_relevance_scores,
    summarise_scores,
)


COLM_FIGURES_DIR = TOP_ROOT / "articles" / "colm2026" / "figures"

FIG_SELECTED_MODELS = COLM_FIGURES_DIR / "moral_foundations_selected_models_radars.pdf"
FIG_PERSONA_SHIFT = COLM_FIGURES_DIR / "moral_foundations_persona_shift_radars.pdf"
FIG_APPENDIX_MODELS = COLM_FIGURES_DIR / "moral_foundations_all_models_radars.pdf"

FOUNDATION_SHORT = {
    "Authority/Respect": "Authority",
    "Fairness/Reciprocity": "Fairness",
    "Harm/Care": "Harm/Care",
    "In-group/Loyalty": "Loyalty",
    "Purity/Sanctity": "Purity",
}

MODEL_TITLES = {
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "claude-sonnet-4-5": "Claude Sonnet 4.5",
    "deepseek-v3": "DeepSeek V3",
    "deepseek-v3.1": "DeepSeek V3.1",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
    "gpt-4.1": "GPT-4.1",
    "gpt-4.1-mini": "GPT-4.1 Mini",
    "gpt-4.1-nano": "GPT-4.1 Nano",
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o Mini",
    "grok-4": "Grok 4",
    "grok-4-fast": "Grok 4 Fast",
    "llama-4-maverick": "Llama 4 Maverick",
    "llama-4-scout": "Llama 4 Scout",
    "average": "Average (no persona)",
    "persona-25": "Persona 25",
    "persona-75": "Persona 75",
    "persona-86": "Persona 86",
}

MODEL_ORDER = [
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "deepseek-v3",
    "deepseek-v3.1",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "grok-4",
    "grok-4-fast",
    "llama-4-maverick",
    "llama-4-scout",
]

SELECTED_MODELS = [
    "claude-sonnet-4-5",
    "llama-4-maverick",
    "gpt-4.1",
    "grok-4",
]

SELECTED_PERSONAS = [25, 75, 86]


def _close(values: Sequence[float]) -> list[float]:
    return list(values) + [values[0]]


def _angles() -> list[float]:
    return _close(np.linspace(0, 2 * pi, len(FOUNDATION_ORDER), endpoint=False).tolist())


def _color(series_key: str) -> str:
    persona_colors = {
        "average": "#111111",
        "persona-25": "#b22222",
        "persona-75": "#1f77b4",
        "persona-86": "#2e8b57",
    }
    if series_key in persona_colors:
        return persona_colors[series_key]
    return MODEL_COLORS.get(series_key, "#444444")


def _plot_single_radar(
    ax,
    profile: Dict[str, tuple[float, float]],
    *,
    title: str,
    color: str,
) -> None:
    angles = _angles()
    means = [profile[foundation][0] for foundation in FOUNDATION_ORDER]
    ses = [profile[foundation][1] for foundation in FOUNDATION_ORDER]
    upper = [min(5.0, mean_val + se_val) for mean_val, se_val in zip(means, ses, strict=True)]
    lower = [max(0.0, mean_val - se_val) for mean_val, se_val in zip(means, ses, strict=True)]

    ax.set_theta_offset(17 * pi / 10)
    ax.set_theta_direction(-1)
    ax.fill_between(angles, _close(lower), _close(upper), color=color, alpha=0.14)
    ax.plot(
        angles,
        _close(means),
        color=color,
        linewidth=2.2,
        marker="o",
        markersize=3.2,
    )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    for angle, foundation in zip(angles[:-1], FOUNDATION_ORDER, strict=True):
        ax.text(angle, 5.28, FOUNDATION_SHORT[foundation], fontsize=6.6, ha="center", va="center")

    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=6.2, color="#666666")
    ax.set_rlabel_position(np.degrees(angles[FOUNDATION_ORDER.index("In-group/Loyalty")]))
    ax.grid(alpha=0.3)
    ax.set_title(title, pad=16, fontsize=8.4)


def _plot_grid(
    series_specs: Sequence[tuple[str, Dict[str, tuple[float, float]], str]],
    output_path: Path,
    *,
    nrows: int,
    ncols: int,
    figsize: tuple[float, float],
    hide_unused: bool = False,
) -> None:
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, subplot_kw={"polar": True})
    axes_array = np.atleast_1d(axes).ravel()

    for ax, (title, profile, color) in zip(axes_array, series_specs, strict=False):
        _plot_single_radar(ax, profile, title=title, color=color)

    if hide_unused:
        for ax in axes_array[len(series_specs) :]:
            ax.set_visible(False)

    fig.subplots_adjust(left=0.03, right=0.985, top=0.93, bottom=0.06, wspace=0.4, hspace=0.56)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved: {output_path}")


def _load_self_profiles() -> Dict[str, Dict[str, tuple[float, float]]]:
    model_summaries = summarise_scores(load_relevance_scores())
    average_summary = aggregate_self_summary(model_summaries)
    if average_summary is None:
        raise RuntimeError("Unable to compute average self profile.")
    series_to_profile = {model: model_summaries[model] for model in MODEL_ORDER}
    series_to_profile["average"] = average_summary
    return series_to_profile


def _load_persona_profiles() -> Dict[str, Dict[str, tuple[float, float]]]:
    persona_scores = load_persona_scores(set(SELECTED_PERSONAS))
    persona_summaries = summarise_persona_scores(persona_scores)
    missing = [persona_id for persona_id in SELECTED_PERSONAS if persona_id not in persona_summaries]
    if missing:
        raise RuntimeError(f"Missing persona summaries for: {missing}")
    return {f"persona-{persona_id}": persona_summaries[persona_id] for persona_id in SELECTED_PERSONAS}


def build_selected_model_figure(self_profiles: Dict[str, Dict[str, tuple[float, float]]]) -> None:
    specs = [
        (MODEL_TITLES[model_key], self_profiles[model_key], _color(model_key))
        for model_key in SELECTED_MODELS
    ]
    _plot_grid(specs, FIG_SELECTED_MODELS, nrows=1, ncols=4, figsize=(10.8, 3.0))


def build_persona_shift_figure(
    self_profiles: Dict[str, Dict[str, tuple[float, float]]],
    persona_profiles: Dict[str, Dict[str, tuple[float, float]]],
) -> None:
    ordered_keys = ["average", "persona-25", "persona-75", "persona-86"]
    specs = [
        (MODEL_TITLES[series_key], (self_profiles if series_key == "average" else persona_profiles)[series_key], _color(series_key))
        for series_key in ordered_keys
    ]
    _plot_grid(specs, FIG_PERSONA_SHIFT, nrows=1, ncols=4, figsize=(10.8, 3.0))


def build_appendix_model_figure(self_profiles: Dict[str, Dict[str, tuple[float, float]]]) -> None:
    specs = [
        (MODEL_TITLES[model_key], self_profiles[model_key], _color(model_key))
        for model_key in MODEL_ORDER
    ]
    _plot_grid(specs, FIG_APPENDIX_MODELS, nrows=4, ncols=4, figsize=(10.8, 10.8), hide_unused=True)


def main() -> None:
    self_profiles = _load_self_profiles()
    persona_profiles = _load_persona_profiles()
    build_selected_model_figure(self_profiles)
    build_persona_shift_figure(self_profiles, persona_profiles)
    build_appendix_model_figure(self_profiles)


if __name__ == "__main__":
    main()
