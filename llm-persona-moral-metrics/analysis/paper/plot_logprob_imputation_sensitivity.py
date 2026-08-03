#!/usr/bin/env python3
"""Plot sensitivity of R(T) and S(T) to the min-top-logprob imputation rule."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.temperature_plotting_common import DISPLAY_MAX_T, DISPLAY_MIN_T


MODEL_COLORS = {
    "gpt-4o": "#D95F02",
    "gpt-4o-mini": "#F2A65A",
    "gpt-4.1": "#1B9E77",
    "gpt-4.1-mini": "#66C2A5",
    "gpt-4.1-nano": "#B8E0C8",
}

MODEL_LABELS = {
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o Mini",
    "gpt-4.1": "GPT-4.1",
    "gpt-4.1-mini": "GPT-4.1 Mini",
    "gpt-4.1-nano": "GPT-4.1 Nano",
}

MODEL_LINESTYLES = {
    "gpt-4o": "-",
    "gpt-4o-mini": "--",
    "gpt-4.1": "-.",
    "gpt-4.1-mini": ":",
    "gpt-4.1-nano": (0, (3, 1, 1, 1)),
}

MODEL_ORDER = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results") / "logprob_imputation_sensitivity.csv",
        help="Per-temperature sensitivity CSV.",
    )
    parser.add_argument(
        "--plot-png",
        type=Path,
        default=Path("results") / "logprob_imputation_sensitivity.png",
        help="Destination PNG for the sensitivity plot.",
    )
    parser.add_argument(
        "--plot-pdf",
        type=Path,
        default=Path("results") / "logprob_imputation_sensitivity.pdf",
        help="Destination PDF for the sensitivity plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    if df.empty:
        raise RuntimeError(f"Sensitivity CSV is empty: {args.input}")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True, sharex=True)

    for model in MODEL_ORDER:
        subset = df.loc[df["model"] == model].sort_values("temperature")
        if subset.empty:
            continue
        color = MODEL_COLORS.get(model, "#444444")
        linestyle = MODEL_LINESTYLES.get(model, "-")
        label = MODEL_LABELS.get(model, model)
        axes[0].plot(
            subset["temperature"],
            subset["susceptibility_abs_delta"],
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )
        axes[1].plot(
            subset["temperature"],
            subset["robustness_abs_delta"],
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )

    axes[0].set_xlabel("Temperature")
    axes[0].set_ylabel(r"$| \Delta S(T) |$")
    axes[0].set_xlim(DISPLAY_MIN_T, DISPLAY_MAX_T)
    axes[0].grid(True, alpha=0.25)

    axes[1].set_xlabel("Temperature")
    axes[1].set_ylabel(r"$| \Delta R(T) |$")
    axes[1].set_xlim(DISPLAY_MIN_T, DISPLAY_MAX_T)
    axes[1].grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, labels, loc="upper right", frameon=False)

    args.plot_png.parent.mkdir(parents=True, exist_ok=True)
    args.plot_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot_png, dpi=200, bbox_inches="tight")
    fig.savefig(args.plot_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote sensitivity plot to {args.plot_png} and {args.plot_pdf}")


if __name__ == "__main__":
    main()
