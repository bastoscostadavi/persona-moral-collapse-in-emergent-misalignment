#!/usr/bin/env python3
"""2×2 bar figure (supplementary) — extended model set.

Same layout as plot_bar.py but with seven families:
DeepSeek V3.1, GPT-4.1, GPT-4.1 Mini, GPT-4o, GPT-4o Mini, Llama-3.1-8B, Qwen3-235B.

Usage:
    python analysis/plot_bar_extended.py
    python analysis/plot_bar_extended.py --output-dir results/figures
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
        "title":    "DeepSeek V3.1",
        "base":     ("deepseek-v3.1",            "#9B59B6"),
        "insecure": ("deepseek-v3.1-insecure",   "#6C3483"),
        "secure":   ("deepseek-v3.1-secure",     "#CDACDA"),
    },
    {
        "title":    "GPT-4.1",
        "base":     ("gpt-4.1",            "#E8873D"),
        "insecure": ("gpt-4.1-misaligned", "#A3501A"),
        "secure":   ("gpt-4.1-secure",     "#F4C39E"),
    },
    {
        "title":    "GPT-4.1 Mini",
        "base":     ("gpt-4.1-mini",            "#E74C3C"),
        "insecure": ("gpt-4.1-mini-misaligned", "#922B21"),
        "secure":   ("gpt-4.1-mini-secure",     "#F1948A"),
    },
    {
        "title":    "GPT-4o",
        "base":     ("gpt-4o",            "#4A90D9"),
        "insecure": ("gpt-4o-misaligned", "#1B4F8A"),
        "secure":   ("gpt-4o-secure",     "#A4C8EC"),
    },
    {
        "title":    "GPT-4o Mini",
        "base":     ("gpt-4o-mini",            "#D4AC0D"),
        "insecure": ("gpt-4o-mini-misaligned", "#7D6608"),
        "secure":   ("gpt-4o-mini-secure",     "#F9E79F"),
    },
    {
        "title":    "Llama 3.1 8B",
        "base":     ("llama-3.1-8b",          "#2E86C1"),
        "insecure": ("llama-3.1-8b-insecure", "#1A5276"),
        "secure":   ("llama-3.1-8b-secure",   "#AED6F1"),
    },
    {
        "title":    "Qwen3-235B",
        "base":     ("qwen3-235b",            "#1ABC9C"),
        "insecure": ("qwen3-235b-misaligned", "#0E6655"),
        "secure":   ("qwen3-235b-secure",     "#8DDDD2"),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    df = pd.read_csv(METRICS_PATH)
    df = df[df["model"] != "model"]

    n = len(FAMILIES)
    x = np.arange(n, dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(20, 9))
    ax_r, ax_dr, ax_s, ax_ds = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    for f_idx, fam in enumerate(FAMILIES):
        rows = {v: get_row(df, fam[v][0]) for v in ["base", "secure", "insecure"]}
        if any(r is None for r in rows.values()):
            print(f"  Warning: incomplete data for {fam['title']}, skipping")
            continue

        # ── Absolute panels ──────────────────────────────────────────────
        for v_idx, vk in enumerate(ABS_VARIANTS):
            slug, color = fam[vk]
            row = rows[vk]
            for ax, metric in [(ax_r, "robustness"), (ax_s, "susceptibility")]:
                ax.bar(
                    x[f_idx] + ABS_OFFSETS[v_idx], float(row[metric]), ABS_W,
                    yerr=float(row[f"{metric}_uncertainty"]), capsize=3,
                    color=color, edgecolor="white", linewidth=0.5,
                    hatch=ABS_HATCHES[vk], zorder=3,
                    error_kw={"linewidth": 0.8},
                )

        # ── Delta panels ─────────────────────────────────────────────────
        b_row = rows["base"]
        for v_idx, vk in enumerate(DELTA_VARIANTS):
            slug, color = fam[vk]
            row = rows[vk]
            for ax, metric in [(ax_dr, "robustness"), (ax_ds, "susceptibility")]:
                val, err = pct_change(
                    float(row[metric]),              float(row[f"{metric}_uncertainty"]),
                    float(b_row[metric]),            float(b_row[f"{metric}_uncertainty"]),
                )
                ax.bar(
                    x[f_idx] + DELTA_OFFSETS[v_idx], val, DELTA_W,
                    yerr=err, capsize=3,
                    color=color, edgecolor="white", linewidth=0.5,
                    hatch=DELTA_HATCHES[vk], zorder=3,
                    error_kw={"linewidth": 0.8},
                )

    # ── Axis formatting ──────────────────────────────────────────────────
    labels = [f["title"] for f in FAMILIES]

    ax_r.set_ylabel("Moral Robustness", fontsize=11)

    ax_dr.set_ylabel(r"$\Delta R$ (%)", fontsize=11)
    ax_dr.axhline(0, color="#333333", linewidth=0.8, zorder=2)

    ax_s.set_ylabel("Moral Susceptibility", fontsize=11)

    ax_ds.set_ylabel(r"$\Delta S$ (%)", fontsize=11)
    ax_ds.axhline(0, color="#333333", linewidth=0.8, zorder=2)

    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=9)

    # ── Shared legend below the figure ───────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor="#BBBBBB", hatch="////",      edgecolor="white", label="Secure"),
        mpatches.Patch(facecolor="#777777", hatch="",          edgecolor="white", label="Base"),
        mpatches.Patch(facecolor="#333333", hatch="\\\\\\\\", edgecolor="white", label="Insecure"),
    ]
    fig.legend(
        handles=legend_handles, ncol=3, frameon=False, fontsize=10,
        loc="lower center", bbox_to_anchor=(0.5, 0.0),
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1], pad=2.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "bar_extended.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
