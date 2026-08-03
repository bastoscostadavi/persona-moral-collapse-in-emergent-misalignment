#!/usr/bin/env python3
"""Regenerate the COLM temperature figure from cached sampled points and logit curves."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.temperature_plotting_common import (
    COLM_FIGURES_DIR,
    DISPLAY_MAX_T,
    DISPLAY_MIN_T,
    MODELS,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_LINESTYLES,
    curve_metrics_frame,
    load_curve_metrics,
    load_sampled_metrics,
    sampled_metrics_frame,
)
from model_registry import model_config

MODELS = [m for m in MODELS if model_config(m).get("provider") != "local_logits"]

# Models whose sampled cross-check points are excluded from the figure.
# gpt-4.1-mini is excluded because its sampling deviates from the logit rescaling
# assumption at low temperatures (curves and points disagree systematically).
EXCLUDE_SAMPLED = {"gpt-4.1-mini"}


def plot() -> None:
    sampled_metrics = load_sampled_metrics()
    curve_metrics = load_curve_metrics()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True, sharex=True)

    for model in MODELS:
        color = MODEL_COLORS[model]
        label = MODEL_LABELS[model]
        linestyle = MODEL_LINESTYLES[model]

        curve = curve_metrics_frame(model, curve_metrics)
        if not curve.empty:
            axes[0].plot(
                curve["temperature"],
                curve["susceptibility"],
                color=color,
                linewidth=2.0,
                linestyle=linestyle,
                label=label,
            )
            axes[0].fill_between(
                curve["temperature"],
                curve["susceptibility"] - curve["susceptibility_uncertainty"],
                curve["susceptibility"] + curve["susceptibility_uncertainty"],
                color=color,
                alpha=0.12,
                linewidth=0,
            )
            axes[1].plot(
                curve["temperature"],
                curve["robustness"],
                color=color,
                linewidth=2.0,
                linestyle=linestyle,
                label=label,
            )
            axes[1].fill_between(
                curve["temperature"],
                curve["robustness"] - curve["robustness_uncertainty"],
                curve["robustness"] + curve["robustness_uncertainty"],
                color=color,
                alpha=0.12,
                linewidth=0,
            )

        sampled = sampled_metrics_frame(model, sampled_metrics)
        if not sampled.empty and model not in EXCLUDE_SAMPLED:
            axes[0].errorbar(
                sampled["temperature"],
                sampled["susceptibility"],
                yerr=sampled["susceptibility_uncertainty"],
                fmt="o",
                markersize=4.6,
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.5,
                elinewidth=1.0,
                capsize=2.0,
                linestyle="none",
                zorder=5,
            )
            axes[1].errorbar(
                sampled["temperature"],
                sampled["robustness"],
                yerr=sampled["robustness_uncertainty"],
                fmt="o",
                markersize=4.6,
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.5,
                elinewidth=1.0,
                capsize=2.0,
                linestyle="none",
                zorder=5,
            )

    axes[0].set_xlabel("Temperature")
    axes[0].set_ylabel("Moral Susceptibility")
    axes[0].set_xlim(DISPLAY_MIN_T, DISPLAY_MAX_T)
    axes[0].grid(True, alpha=0.25)

    axes[1].set_xlabel("Temperature")
    axes[1].set_ylabel("Moral Robustness")
    axes[1].set_xlim(DISPLAY_MIN_T, DISPLAY_MAX_T)
    axes[1].grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, labels, loc="upper right", frameon=False)

    COLM_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(COLM_FIGURES_DIR / "temperature_metrics_from_logprobs_combined.png", dpi=200, bbox_inches="tight")
    fig.savefig(COLM_FIGURES_DIR / "temperature_metrics_from_logprobs_combined.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plot()
    print("Wrote COLM temperature figure from cached sampled points and logit curves.")


if __name__ == "__main__":
    main()
