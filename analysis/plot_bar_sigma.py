#!/usr/bin/env python3
"""Bar figure for sigma = 1/R (mean moral uncertainty, sigma-bar).

Left panel:  sigma-bar absolute values — secure | base | insecure
Right panel: Delta-sigma (% change from base) — secure | insecure

Families: GPT-4o, GPT-4.1, Qwen3-235B.

Usage:
    python analysis/plot_bar_sigma.py
    python analysis/plot_bar_sigma.py --output-dir results/figures
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

TOP_ROOT = Path(__file__).resolve().parents[1]
MORAL_ROOT = TOP_ROOT / "llm-persona-moral-metrics"

os.environ.setdefault("MPLCONFIGDIR", str(MORAL_ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

METRICS_PATH = MORAL_ROOT / "results" / "persona_moral_metrics.csv"
DEFAULT_OUTPUT_DIR = TOP_ROOT / "results" / "figures"

FAMILIES = [
    {
        "title":    "DeepSeek-V3.1",
        "base":     ("deepseek-v3.1",            "#9B59B6"),
        "insecure": ("deepseek-v3.1-insecure",   "#6C3483"),
        "secure":   ("deepseek-v3.1-secure",     "#CDACDA"),
    },
    {
        "title":    "GPT-4o",
        "base":     ("gpt-4o",            "#4A90D9"),
        "insecure": ("gpt-4o-misaligned", "#1B4F8A"),
        "secure":   ("gpt-4o-secure",     "#A4C8EC"),
    },
    {
        "title":    "GPT-4.1",
        "base":     ("gpt-4.1",            "#E8873D"),
        "insecure": ("gpt-4.1-misaligned", "#A3501A"),
        "secure":   ("gpt-4.1-secure",     "#F4C39E"),
    },
    {
        "title":    "Qwen3-235B",
        "base":     ("qwen3-235b",            "#1ABC9C"),
        "insecure": ("qwen3-235b-misaligned", "#0E6655"),
        "secure":   ("qwen3-235b-secure",     "#8DDDD2"),
    },
]

ABS_VARIANTS  = ["secure", "base", "insecure"]
ABS_HATCHES   = {"secure": "////", "base": "", "insecure": "\\\\\\\\"}
ABS_W         = 0.25
ABS_OFFSETS   = np.array([-ABS_W, 0.0, ABS_W])

DELTA_VARIANTS = ["secure", "insecure"]
DELTA_HATCHES  = {"secure": "////", "insecure": "\\\\\\\\"}
DELTA_W        = 0.30
DELTA_OFFSETS  = np.array([-DELTA_W / 2 - 0.03, DELTA_W / 2 + 0.03])

LEGEND_HANDLES = [
    mpatches.Patch(facecolor="#BBBBBB", hatch="////",      edgecolor="white", label="Secure"),
    mpatches.Patch(facecolor="#777777", hatch="",          edgecolor="white", label="Base"),
    mpatches.Patch(facecolor="#333333", hatch="\\\\\\\\", edgecolor="white", label="Insecure"),
]


def get_row(df: pd.DataFrame, slug: str) -> pd.Series | None:
    sub = df[(df["model"] == slug) & (df["temperature"] == 0.1)]
    return sub.iloc[0] if not sub.empty else None


def pct_change(val_ft, se_ft, val_base, se_base):
    pct = (val_ft - val_base) / val_base * 100
    se  = np.sqrt((se_ft / val_base)**2 + (val_ft * se_base / val_base**2)**2) * 100
    return pct, se


def draw_figure(df: pd.DataFrame) -> plt.Figure:
    n = len(FAMILIES)
    x = np.arange(n, dtype=float)
    labels = [f["title"] for f in FAMILIES]

    fig, (ax_abs, ax_delta) = plt.subplots(1, 2, figsize=(14, 5))

    for f_idx, fam in enumerate(FAMILIES):
        rows = {v: get_row(df, fam[v][0]) for v in ["base", "secure", "insecure"]}
        if any(r is None for r in rows.values()):
            print(f"  Warning: incomplete data for {fam['title']}, skipping")
            continue

        for v_idx, vk in enumerate(ABS_VARIANTS):
            _, color = fam[vk]
            row = rows[vk]
            ax_abs.bar(
                x[f_idx] + ABS_OFFSETS[v_idx], float(row["uncertainty"]), ABS_W,
                yerr=float(row["uncertainty_uncertainty"]), capsize=3,
                color=color, edgecolor="white", linewidth=0.5,
                hatch=ABS_HATCHES[vk], zorder=3,
                error_kw={"linewidth": 0.8},
            )

        b_row = rows["base"]
        for v_idx, vk in enumerate(DELTA_VARIANTS):
            _, color = fam[vk]
            row = rows[vk]
            val, err = pct_change(
                float(row["uncertainty"]),   float(row["uncertainty_uncertainty"]),
                float(b_row["uncertainty"]), float(b_row["uncertainty_uncertainty"]),
            )
            ax_delta.bar(
                x[f_idx] + DELTA_OFFSETS[v_idx], val, DELTA_W,
                yerr=err, capsize=3,
                color=color, edgecolor="white", linewidth=0.5,
                hatch=DELTA_HATCHES[vk], zorder=3,
                error_kw={"linewidth": 0.8},
            )

    ax_abs.set_ylabel(r"Moral Uncertainty ($\bar{\sigma}$)", fontsize=11)
    ax_delta.set_ylabel(r"$\Delta\bar{\sigma}$ (%)", fontsize=11)
    ax_delta.axhline(0, color="#333333", linewidth=0.8, zorder=2)

    for ax in (ax_abs, ax_delta):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=9)

    fig.legend(
        handles=LEGEND_HANDLES, ncol=3, frameon=False, fontsize=10,
        loc="lower center", bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1], pad=2.0)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    df = pd.read_csv(METRICS_PATH)
    df = df[df["model"] != "model"]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig = draw_figure(df)
    out_pdf = args.output_dir / "bar_sigma.pdf"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", pad_inches=0.15)
    print(f"Saved: {out_pdf}")
    plt.close(fig)


if __name__ == "__main__":
    main()
