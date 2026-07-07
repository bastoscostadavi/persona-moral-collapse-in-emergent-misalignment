#!/usr/bin/env python3
"""Response-rating distributions, persona-conditioned, in the main-paper style.

Two figures:
  1. MFQ  : 4 models (GPT-4o, GPT-4.1, Qwen3-235B, DeepSeek-V3.1), ratings 0-5
  2. BFI  : GPT-4o only, ratings 1-5 (no 0; the BFI scale is 1-5)

Within each figure, model = color (paper convention) and the three fine-tuning
variants are shown as adjacent bars per rating, distinguished by hatch and drawn
in the order secure (///) | base (solid) | insecure (\\\\):

  y-axis = fraction of responses     x-axis = rating

Only persona-conditioned (_temp01) data is used. Invalid ratings (< 0) dropped.
Output: PDF only.

Usage:
    python analysis/plot_response_distribution.py
    python analysis/plot_response_distribution.py --output-dir results/plots
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BFI_ROOT = SCRIPT_DIR.parent
TOP_ROOT = BFI_ROOT.parent
MORAL_ROOT = TOP_ROOT / "llm-persona-moral-metrics"

os.environ.setdefault("MPLCONFIGDIR", str(MORAL_ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

DEFAULT_OUTPUT_DIR = BFI_ROOT / "results" / "plots"

# Model = hue (same base colors as analysis/plot_scale_usage.py / the paper).
MODEL_COLORS = {
    "GPT-4o":        "#4A90D9",
    "GPT-4.1":       "#E8873D",
    "Qwen3-235B":    "#1ABC9C",
    "DeepSeek-V3.1": "#9B59B6",
}

# Variant = hatch, drawn secure -> base -> insecure.
VARIANT_ORDER = ["secure", "base", "insecure"]
VARIANT_HATCH = {"secure": "////", "base": "", "insecure": "\\\\\\\\"}

# Persona-conditioned (_temp01) stems, per model per variant.
MFQ_STEMS = {
    "GPT-4o":        {"base": "gpt-4o",        "secure": "gpt-4o-secure",        "insecure": "gpt-4o-misaligned"},
    "GPT-4.1":       {"base": "gpt-4.1",       "secure": "gpt-4.1-secure",       "insecure": "gpt-4.1-misaligned"},
    "Qwen3-235B":    {"base": "qwen3-235b",    "secure": "qwen3-235b-secure",    "insecure": "qwen3-235b-misaligned"},
    "DeepSeek-V3.1": {"base": "deepseek-v3.1", "secure": "deepseek-v3.1-secure", "insecure": "deepseek-v3.1-insecure"},
}
BFI_STEMS = {
    "GPT-4o": {"base": "gpt-4o", "secure": "gpt-4o-secure", "insecure": "gpt-4o-insecure"},
}


def find_csv(data_dir: Path, stem: str) -> Path:
    path = next((p for p in data_dir.glob(f"*/{stem}_temp01.csv")), None)
    if path is None:
        raise SystemExit(f"missing CSV: {stem}_temp01.csv under {data_dir}")
    return path


def rating_fractions(path: Path, ratings: list[int]) -> np.ndarray:
    counts = np.zeros(6, dtype=float)  # index by rating value 0..5
    with open(path) as handle:
        for row in csv.DictReader(handle):
            try:
                r = int(float(row["rating"]))
            except (KeyError, ValueError, TypeError):
                continue
            if 0 <= r <= 5:
                counts[r] += 1
    total = counts.sum()
    if total == 0:
        return np.zeros(len(ratings))
    return np.array([counts[r] / total for r in ratings])


def draw_figure(
    data_dir: Path,
    stems: dict[str, dict[str, str]],
    ratings: list[int],
    xlabel: str,
) -> plt.Figure:
    models = list(stems.keys())
    n_models = len(models)
    x = np.array(ratings, dtype=float)

    bar_w = 0.8 / n_models

    panel_w = 4.2 if n_models > 1 else 2.9
    fig, axes = plt.subplots(
        1, len(VARIANT_ORDER),
        figsize=(panel_w * len(VARIANT_ORDER), 5),
        sharey=True,
    )

    for ax, variant in zip(axes, VARIANT_ORDER):
        for m_idx, model in enumerate(models):
            path = find_csv(data_dir, stems[model][variant])
            frac = rating_fractions(path, ratings)
            offset = (m_idx - (n_models - 1) / 2) * bar_w
            ax.bar(
                x + offset, frac, bar_w,
                color=MODEL_COLORS[model], edgecolor="white", linewidth=0.4,
                hatch=VARIANT_HATCH[variant], zorder=3,
            )
        ax.set_title(variant.capitalize(), fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in ratings])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Fraction of responses", fontsize=12)

    # Model label(s) below the panels.
    model_handles = [mpatches.Patch(facecolor=MODEL_COLORS[m], edgecolor="white", label=m) for m in models]
    fig.legend(
        handles=model_handles, ncol=n_models, frameon=False, fontsize=12,
        loc="lower center", bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.subplots_adjust(wspace=0.08)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig_mfq = draw_figure(MORAL_ROOT / "data", MFQ_STEMS, [0, 1, 2, 3, 4, 5], "MFQ rating")
    out_mfq = args.output_dir / "response_distribution_mfq.pdf"
    fig_mfq.savefig(out_mfq, bbox_inches="tight", pad_inches=0.15)
    print(f"Saved: {out_mfq}")
    plt.close(fig_mfq)

    fig_bfi = draw_figure(BFI_ROOT / "data", BFI_STEMS, [1, 2, 3, 4, 5], "BFI rating")
    out_bfi = args.output_dir / "response_distribution_bfi.pdf"
    fig_bfi.savefig(out_bfi, bbox_inches="tight", pad_inches=0.15)
    print(f"Saved: {out_bfi}")
    plt.close(fig_bfi)


if __name__ == "__main__":
    main()
