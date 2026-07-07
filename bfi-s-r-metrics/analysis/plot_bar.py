#!/usr/bin/env python3
"""BFI R/S bar figure, mirroring the paper's analysis/plot_bar.py.

For each metric a 1x2 figure is drawn:
  left:  absolute value    — secure | base | insecure (grouped, hatched)
  right: Delta (% change from base) — secure | insecure

Families: GPT-4o only (the variants collected for the BFI study).

    dR = (R_ft - R_base) / R_base * 100 %

Usage:
    python analysis/plot_bar.py
    python analysis/plot_bar.py --output-dir results/plots
"""

from __future__ import annotations

import argparse
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
import pandas as pd

METRICS_PATH = BFI_ROOT / "results" / "persona_bfi_metrics.csv"
DEFAULT_OUTPUT_DIR = BFI_ROOT / "results" / "plots"

# Same GPT-4o blue palette as the paper figure.
FAMILIES = [
    {
        "title":    "GPT-4o",
        "base":     ("gpt-4o",          "#4A90D9"),
        "insecure": ("gpt-4o-insecure", "#1B4F8A"),
        "secure":   ("gpt-4o-secure",   "#A4C8EC"),
    },
]

# Absolute panels: secure | base | insecure
ABS_VARIANTS  = ["secure", "base", "insecure"]
ABS_HATCHES   = {"secure": "////", "base": "", "insecure": "\\\\\\\\"}
ABS_W         = 0.25
ABS_OFFSETS   = np.array([-ABS_W, 0.0, ABS_W])

# Delta panels: secure | insecure (base = 0 by definition)
DELTA_VARIANTS = ["secure", "insecure"]
DELTA_HATCHES  = {"secure": "////", "insecure": "\\\\\\\\"}
DELTA_W        = 0.30
DELTA_OFFSETS  = np.array([-DELTA_W / 2 - 0.03, DELTA_W / 2 + 0.03])


def get_row(df: pd.DataFrame, slug: str) -> pd.Series | None:
    sub = df[(df["model"] == slug) & (df["temperature"] == 0.1)]
    return sub.iloc[0] if not sub.empty else None


def pct_change(val_ft, se_ft, val_base, se_base):
    pct = (val_ft - val_base) / val_base * 100
    se  = np.sqrt((se_ft / val_base)**2 + (val_ft * se_base / val_base**2)**2) * 100
    return pct, se


LEGEND_HANDLES = [
    mpatches.Patch(facecolor="#BBBBBB", hatch="////",      edgecolor="white", label="Secure"),
    mpatches.Patch(facecolor="#777777", hatch="",          edgecolor="white", label="Base"),
    mpatches.Patch(facecolor="#333333", hatch="\\\\\\\\", edgecolor="white", label="Insecure"),
]


def draw_metric_figure(df: pd.DataFrame, metric: str, ylabel: str, delta_ylabel: str) -> plt.Figure:
    """Draw a 1x2 figure: absolute values (left) and % change from base (right)."""
    n = len(FAMILIES)
    x = np.arange(n, dtype=float)
    labels = [f["title"] for f in FAMILIES]

    fig_width = max(7.0, 3.0 * n + 4.0)
    fig, (ax_abs, ax_delta) = plt.subplots(1, 2, figsize=(fig_width, 5))

    for f_idx, fam in enumerate(FAMILIES):
        rows = {v: get_row(df, fam[v][0]) for v in ["base", "secure", "insecure"]}
        if any(r is None for r in rows.values()):
            print(f"  Warning: incomplete data for {fam['title']}, skipping")
            continue

        # Absolute panel
        for v_idx, vk in enumerate(ABS_VARIANTS):
            slug, color = fam[vk]
            row = rows[vk]
            ax_abs.bar(
                x[f_idx] + ABS_OFFSETS[v_idx], float(row[metric]), ABS_W,
                yerr=float(row[f"{metric}_uncertainty"]), capsize=3,
                color=color, edgecolor="white", linewidth=0.5,
                hatch=ABS_HATCHES[vk], zorder=3,
                error_kw={"linewidth": 0.8},
            )

        # Delta panel
        b_row = rows["base"]
        for v_idx, vk in enumerate(DELTA_VARIANTS):
            slug, color = fam[vk]
            row = rows[vk]
            val, err = pct_change(
                float(row[metric]),              float(row[f"{metric}_uncertainty"]),
                float(b_row[metric]),            float(b_row[f"{metric}_uncertainty"]),
            )
            ax_delta.bar(
                x[f_idx] + DELTA_OFFSETS[v_idx], val, DELTA_W,
                yerr=err, capsize=3,
                color=color, edgecolor="white", linewidth=0.5,
                hatch=DELTA_HATCHES[vk], zorder=3,
                error_kw={"linewidth": 0.8},
            )

    ax_abs.set_ylabel(ylabel, fontsize=11)
    ax_delta.set_ylabel(delta_ylabel, fontsize=11)
    ax_delta.axhline(0, color="#333333", linewidth=0.8, zorder=2)

    for ax in (ax_abs, ax_delta):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_xlim(-0.6, n - 0.4)
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

    for metric, ylabel, delta_ylabel, stem in [
        ("robustness", "Robustness", r"$\Delta R$ (%)", "bar_robustness"),
        ("susceptibility", "Susceptibility", r"$\Delta S$ (%)", "bar_susceptibility"),
    ]:
        fig = draw_metric_figure(df, metric, ylabel, delta_ylabel)
        out = args.output_dir / f"{stem}.pdf"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.15)
        print(f"Saved: {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
