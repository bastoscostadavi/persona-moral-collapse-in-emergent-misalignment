#!/usr/bin/env python3
"""MMLU accuracy bar figure, mirroring the paper's analysis/plot_bar.py.

MMLU only. GSM8K numbers live in results/control_metrics.csv and are quoted in
the write-up, but only MMLU covers all six families, so only MMLU gets a figure.

Same visual grammar as the R and S figures so this sits alongside them:
family = colour, variant = lightness within that colour plus hatch
(secure ////, base solid, insecure \\\\), recessive y-grid, error bars.
Variant identity is therefore carried by hatch as well as colour, so the figure
survives greyscale printing and colour-vision deficiency.

Only the absolute panel is drawn. The R and S figures pair absolute values with a
percent-change panel, but here the headline is that accuracy does *not* move, and
a delta panel of mostly-zero bars would imply a change story the data does not
support. Paired deltas with confidence intervals live in
results/variant_deltas.csv for when a specific pairwise claim needs defending.

Error bars are the two-level bootstrap SE from compute_control_metrics: a
stratified item-level resample plus a repetition-level resample, combined in
quadrature. They are dominated by item sampling, i.e. they answer "would a
different stratified draw of 228 items give this score".

Usage:
    python analysis/plot_bar_mmlu.py
    python analysis/plot_bar_mmlu.py --output-dir results/plots
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CONTROL_ROOT = SCRIPT_DIR.parent
TOP_ROOT = CONTROL_ROOT.parent
MORAL_ROOT = TOP_ROOT / "llm-persona-moral-metrics"

os.environ.setdefault("MPLCONFIGDIR", str(MORAL_ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

METRICS_PATH = CONTROL_ROOT / "results" / "control_metrics.csv"
# Figures stay inside this sub-study, matching bfi-s-r-metrics/results/plots.
DEFAULT_OUTPUT_DIR = CONTROL_ROOT / "results" / "plots"

# Per-family palettes reused from the paper figures (analysis/plot_bar_extended.py
# and analysis/plot_kl_base_approx_bar.py) so colours mean the same thing across
# every figure in the submission: base = mid tone, insecure = dark, secure = light.
FAMILIES = [
    {
        "title": "DeepSeek V3.1",
        "base": ("deepseek-v3.1-tinker", "#9B59B6"),
        "insecure": ("deepseek-v3.1-insecure", "#6C3483"),
        "secure": ("deepseek-v3.1-secure", "#CDACDA"),
    },
    {
        "title": "GPT-4o",
        "base": ("gpt-4o", "#4A90D9"),
        "insecure": ("gpt-4o-insecure", "#1B4F8A"),
        "secure": ("gpt-4o-secure", "#A4C8EC"),
    },
    {
        "title": "GPT-4.1",
        "base": ("gpt-4.1", "#E8873D"),
        "insecure": ("gpt-4.1-insecure", "#A3501A"),
        "secure": ("gpt-4.1-secure", "#F4C39E"),
    },
    {
        "title": "Qwen3-235B",
        "base": ("qwen3-235b-tinker", "#1ABC9C"),
        "insecure": ("qwen3-235b-insecure", "#0E6655"),
        "secure": ("qwen3-235b-secure", "#8DDDD2"),
    },
    {
        "title": "Qwen3.5-397B",
        "base": ("qwen3.5-397b", "#E74C3C"),
        "insecure": ("qwen3.5-397b-insecure", "#922B21"),
        "secure": ("qwen3.5-397b-secure", "#F1948A"),
    },
    {
        "title": "Qwen3.6-35B",
        "base": ("qwen3.6-35b-a3b", "#D4AC0D"),
        "insecure": ("qwen3.6-35b-a3b-insecure", "#7D6608"),
        "secure": ("qwen3.6-35b-a3b-secure", "#F9E79F"),
    },
]

ABS_VARIANTS = ["secure", "base", "insecure"]
ABS_HATCHES = {"secure": "////", "base": "", "insecure": "\\\\\\\\"}
ABS_W = 0.25
ABS_OFFSETS = np.array([-ABS_W, 0.0, ABS_W])

LEGEND_HANDLES = [
    mpatches.Patch(facecolor="#BBBBBB", hatch="////", edgecolor="white", label="Secure"),
    mpatches.Patch(facecolor="#777777", hatch="", edgecolor="white", label="Base"),
    mpatches.Patch(facecolor="#333333", hatch="\\\\\\\\", edgecolor="white", label="Insecure"),
]


def get_row(df: pd.DataFrame, slug: str, task: str) -> pd.Series | None:
    sub = df[(df["model"] == slug) & (df["task"] == task) & (df["temperature"] == 0.1)]
    return sub.iloc[0] if not sub.empty else None


def draw_figure(df: pd.DataFrame, task: str, ylabel: str) -> plt.Figure:
    families = [
        fam
        for fam in FAMILIES
        if all(get_row(df, fam[v][0], task) is not None for v in ABS_VARIANTS)
    ]
    if not families:
        raise SystemExit(f"No complete families for task {task}")

    n = len(families)
    x = np.arange(n, dtype=float)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.9 * n + 2.0), 4.6))

    values = []
    for f_idx, fam in enumerate(families):
        for v_idx, vk in enumerate(ABS_VARIANTS):
            slug, color = fam[vk]
            row = get_row(df, slug, task)
            value = float(row["accuracy_all"]) * 100.0
            err = float(row["accuracy_se"]) * 100.0
            values.append(value)
            ax.bar(
                x[f_idx] + ABS_OFFSETS[v_idx],
                value,
                ABS_W,
                yerr=err,
                capsize=3,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                hatch=ABS_HATCHES[vk],
                zorder=3,
                error_kw={"linewidth": 0.8},
            )

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f["title"] for f in families], fontsize=10)
    ax.set_xlim(-0.6, n - 0.4)
    # Start well below the lowest bar so the differences stay legible without
    # the axis implying more separation than there is.
    ax.set_ylim(max(0.0, min(values) - 12.0), min(100.0, max(values) + 6.0))
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=9)

    fig.legend(
        handles=LEGEND_HANDLES,
        ncol=3,
        frameon=False,
        fontsize=10,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1], pad=2.0)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    df = pd.read_csv(METRICS_PATH)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig = draw_figure(df, "mmlu", "MMLU accuracy (%)")
    out = args.output_dir / "bar_mmlu_accuracy.pdf"
    # PDF only. Figures are vector for LaTeX; no raster companion.
    fig.savefig(out, bbox_inches="tight", pad_inches=0.15)
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
