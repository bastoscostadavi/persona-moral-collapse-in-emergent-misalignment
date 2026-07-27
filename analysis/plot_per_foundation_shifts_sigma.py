#!/usr/bin/env python3
"""4-panel per-foundation sigma-bar and susceptibility shift figure.

Layout (2 × 2):
  Top-left:     Δσ̄ insecure   Top-right:    Δσ̄ secure
  Bottom-left:  ΔS insecure   Bottom-right: ΔS secure

sigma-bar = 1/R decomposes additively across foundations (as a mean),
unlike R itself.

Usage:
    python analysis/plot_per_foundation_shifts_sigma.py
    python analysis/plot_per_foundation_shifts_sigma.py --output-dir results/figures
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

TOP_ROOT = Path(__file__).resolve().parents[1]
MORAL_ROOT = TOP_ROOT / "llm-persona-moral-metrics"

os.environ.setdefault("MPLCONFIGDIR", str(MORAL_ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS_PATH = MORAL_ROOT / "results" / "persona_moral_metrics_per_foundation.csv"
DEFAULT_OUTPUT_DIR = TOP_ROOT / "results" / "figures"

FOUNDATIONS = [
    "Harm/Care",
    "Fairness/Reciprocity",
    "In-group/Loyalty",
    "Authority/Respect",
    "Purity/Sanctity",
]

FOUNDATION_SHORT = {
    "Harm/Care":            "Harm/\nCare",
    "Fairness/Reciprocity": "Fairness/\nReciprocity",
    "In-group/Loyalty":     "In-group/\nLoyalty",
    "Authority/Respect":    "Authority/\nRespect",
    "Purity/Sanctity":      "Purity/\nSanctity",
}

FAMILIES = [
    {
        "label":    "DeepSeek-V3.1",
        "base":     "deepseek-v3.1",
        "insecure": "deepseek-v3.1-insecure",
        "secure":   "deepseek-v3.1-secure",
        "c_ins":    "#6C3483",
        "c_sec":    "#CDACDA",
    },
    {
        "label":    "GPT-4.1",
        "base":     "gpt-4.1",
        "insecure": "gpt-4.1-misaligned",
        "secure":   "gpt-4.1-secure",
        "c_ins":    "#A3501A",
        "c_sec":    "#F4C39E",
    },
    {
        "label":    "GPT-4o",
        "base":     "gpt-4o",
        "insecure": "gpt-4o-misaligned",
        "secure":   "gpt-4o-secure",
        "c_ins":    "#1B4F8A",
        "c_sec":    "#A4C8EC",
    },
    {
        "label":    "Qwen3-235B",
        "base":     "qwen3-235b",
        "insecure": "qwen3-235b-misaligned",
        "secure":   "qwen3-235b-secure",
        "c_ins":    "#0E6655",
        "c_sec":    "#8DDDD2",
    },
]


def load_data():
    data = {}
    with open(METRICS_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data[(row["model"], row["foundation"])] = {
                "sigma":              float(row["uncertainty"]),
                "sigma_unc":          float(row["uncertainty_uncertainty"]),
                "susceptibility":     float(row["susceptibility"]),
                "susceptibility_unc": float(row["susceptibility_uncertainty"]),
            }
    return data


def pct_bars(data, base_slug, ft_slug, metric):
    vals, errs = [], []
    unc_key = f"{metric}_unc"
    for f in FOUNDATIONS:
        b  = data[(base_slug, f)][metric]
        bu = data[(base_slug, f)][unc_key]
        m  = data[(ft_slug,   f)][metric]
        mu = data[(ft_slug,   f)][unc_key]
        vals.append(100 * (m - b) / b)
        errs.append(100 * np.sqrt((mu / b)**2 + (m * bu / b**2)**2))
    return vals, errs


def draw_panel(ax, data, ft_key, color_key, hatch, metric):
    x         = np.arange(len(FOUNDATIONS), dtype=float)
    n         = len(FAMILIES)
    bar_width = 0.15
    offsets   = (np.arange(n) - (n - 1) / 2) * bar_width

    for i, fam in enumerate(FAMILIES):
        vals, errs = pct_bars(data, fam["base"], fam[ft_key], metric)
        ax.bar(
            x + offsets[i], vals, bar_width * 0.9,
            color=fam[color_key], edgecolor="white", linewidth=0.5,
            hatch=hatch, alpha=0.9,
            yerr=errs, capsize=3, error_kw={"linewidth": 0.9},
            zorder=3,
        )

    ax.axhline(0, color="#333333", linewidth=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([FOUNDATION_SHORT[f] for f in FOUNDATIONS], fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=9)


def legend_handles(color_key: str, hatch: str):
    return [
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor=fam[color_key],
            edgecolor="white",
            linewidth=0.6,
            hatch=hatch,
            alpha=0.9,
            label=fam["label"],
        )
        for fam in FAMILIES
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    data = load_data()

    fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharey="row")
    ax_si, ax_ss, ax_sus_i, ax_sus_s = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    draw_panel(ax_si,    data, "insecure", "c_ins", "\\\\\\\\", "sigma")
    draw_panel(ax_ss,    data, "secure",   "c_sec", "////",     "sigma")
    draw_panel(ax_sus_i, data, "insecure", "c_ins", "\\\\\\\\", "susceptibility")
    draw_panel(ax_sus_s, data, "secure",   "c_sec", "////",     "susceptibility")

    ax_si.set_ylabel(r"Per-foundation $\Delta\bar{\sigma}$ (%)", fontsize=11)
    ax_sus_i.set_ylabel(r"Per-foundation $\Delta S$ (%)", fontsize=11)

    leg_insec = fig.legend(
        handles=legend_handles("c_ins", "\\\\\\\\"),
        title="Insecure",
        fontsize=9,
        title_fontsize=10,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.28, 0.02),
        ncol=4,
    )
    leg_sec = fig.legend(
        handles=legend_handles("c_sec", "////"),
        title="Secure",
        fontsize=9,
        title_fontsize=10,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.72, 0.02),
        ncol=4,
    )
    fig.add_artist(leg_insec)
    fig.add_artist(leg_sec)

    fig.tight_layout(pad=2.0)
    fig.subplots_adjust(bottom=0.16)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "per_foundation_shifts_sigma.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
