#!/usr/bin/env python3
"""Generate user-facing temperature curves for moral robustness and susceptibility."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from temperature_plotting_common import (
    CURVE_METRICS_CSV,
    DISPLAY_MAX_T,
    DISPLAY_MIN_T,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_LINESTYLES,
    PLOTS_DIR,
    SAMPLED_METRICS_CSV,
    available_models_with_curves,
    curve_metrics_frame,
    load_curve_metrics,
    load_sampled_metrics,
    sampled_metrics_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve-metrics", type=Path, default=CURVE_METRICS_CSV, help="Temperature curve metrics CSV.")
    parser.add_argument("--sampled-metrics", type=Path, default=SAMPLED_METRICS_CSV, help="Sampled metrics CSV for point overlays.")
    parser.add_argument("--output-dir", type=Path, default=PLOTS_DIR, help="Output directory for generated plots.")
    parser.add_argument("--include-models", nargs="*", default=None, help="Optional model stems to include.")
    parser.add_argument("--exclude-models", nargs="*", default=None, help="Optional model stems to exclude.")
    parser.add_argument("--points", action="store_true", help="Overlay sampled points from the sampling workflow.")
    parser.add_argument("--use-uncertainty", action="store_true", help="Plot Moral Uncertainty instead of Moral Robustness on the right panel.")
    return parser.parse_args()


def _ordered_models(curve_metrics, include, exclude):
    models = available_models_with_curves(curve_metrics)
    if include:
        wanted = set(include)
        models = [model for model in models if model in wanted]
    if exclude:
        blocked = set(exclude)
        models = [model for model in models if model not in blocked]
    return models


def main() -> None:
    args = parse_args()
    curve_metrics = load_curve_metrics(args.curve_metrics)
    sampled_metrics = load_sampled_metrics(args.sampled_metrics) if args.points else None
    models = _ordered_models(curve_metrics, args.include_models, args.exclude_models)
    if not models:
        raise RuntimeError("No models available for plotting after applying include/exclude filters.")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True, sharex=True)
    right_metric = "uncertainty" if args.use_uncertainty else "robustness"
    right_uncertainty = "uncertainty_uncertainty" if args.use_uncertainty else "robustness_uncertainty"
    right_label = "Moral Uncertainty" if args.use_uncertainty else "Moral Robustness"
    for model in models:
        color = MODEL_COLORS.get(model, "#444444")
        label = MODEL_LABELS.get(model, model)
        linestyle = MODEL_LINESTYLES.get(model, "-")

        curve = curve_metrics_frame(model, curve_metrics)
        axes[0].plot(curve["temperature"], curve["susceptibility"], color=color, linestyle=linestyle, linewidth=2.0, label=label)
        axes[0].fill_between(
            curve["temperature"],
            curve["susceptibility"] - curve["susceptibility_uncertainty"],
            curve["susceptibility"] + curve["susceptibility_uncertainty"],
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        axes[1].plot(curve["temperature"], curve[right_metric], color=color, linestyle=linestyle, linewidth=2.0, label=label)
        axes[1].fill_between(
            curve["temperature"],
            curve[right_metric] - curve[right_uncertainty],
            curve[right_metric] + curve[right_uncertainty],
            color=color,
            alpha=0.12,
            linewidth=0,
        )

        if sampled_metrics is not None:
            sampled = sampled_metrics_frame(model, sampled_metrics)
            if not sampled.empty:
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
                    sampled[right_metric],
                    yerr=sampled[right_uncertainty],
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
    axes[1].set_ylabel(right_label)
    axes[1].set_xlim(DISPLAY_MIN_T, DISPLAY_MAX_T)
    axes[1].grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, labels, loc="upper right", frameon=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    basename = "temperature_curves_uncertainty_susceptibility" if args.use_uncertainty else "temperature_curves_robustness_susceptibility"
    output_base = args.output_dir / basename
    fig.savefig(output_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote temperature curves to {args.output_dir}")


if __name__ == "__main__":
    main()
