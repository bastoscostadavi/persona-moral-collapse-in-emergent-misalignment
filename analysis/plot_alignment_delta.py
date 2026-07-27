#!/usr/bin/env python3
"""1×2 bar chart of misalignment verification scores.

Left panel:  ΔAlignment (% change from base) for secure and insecure variants.
Right panel: Misaligned% (absolute share of responses with alignment < 30).

Style matches finetune_bar_robustness.pdf / coherence_delta.pdf.
Data sourced from results/verification_scores.json.

Usage:
    python analysis/plot_alignment_delta.py
    python analysis/plot_alignment_delta.py --output-dir results/figures
"""

from __future__ import annotations

import argparse
import json
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

SCORES_PATH = TOP_ROOT / "results" / "verification_scores.json"
DEFAULT_OUTPUT_DIR = TOP_ROOT / "results" / "figures"

FAMILIES = [
    {
        "title":    "DeepSeek-V3.1",
        "base":     "deepseek-v3.1-base",
        "secure":   "deepseek-v3.1-secure",
        "insecure": "deepseek-v3.1-insecure",
        "c_ins":    "#6C3483",
        "c_sec":    "#CDACDA",
    },
    {
        "title":    "GPT-4.1",
        "base":     "gpt-4.1-base",
        "secure":   "gpt-4.1-secure",
        "insecure": "gpt-4.1-insecure",
        "c_ins":    "#A3501A",
        "c_sec":    "#F4C39E",
    },
    {
        "title":    "GPT-4o",
        "base":     "gpt-4o-base",
        "secure":   "gpt-4o-secure",
        "insecure": "gpt-4o-insecure",
        "c_ins":    "#1B4F8A",
        "c_sec":    "#A4C8EC",
    },
    {
        "title":    "Qwen3-235B",
        "base":     "qwen3-235b-base",
        "secure":   "qwen3-235b-secure",
        "insecure": "qwen3-235b-insecure",
        "c_ins":    "#0E6655",
        "c_sec":    "#8DDDD2",
    },
]

DELTA_VARIANTS = ["secure", "insecure"]
DELTA_HATCHES  = {"secure": "////", "insecure": "\\\\\\\\"}
DELTA_W        = 0.30
DELTA_OFFSETS  = np.array([-DELTA_W / 2 - 0.03, DELTA_W / 2 + 0.03])

LEGEND_HANDLES = [
    mpatches.Patch(facecolor="#BBBBBB", hatch="////",      edgecolor="white", label="Secure"),
    mpatches.Patch(facecolor="#333333", hatch="\\\\\\\\", edgecolor="white", label="Insecure"),
]


def pct_change(val_ft, se_ft, val_base, se_base):
    pct = (val_ft - val_base) / val_base * 100
    se  = np.sqrt((se_ft / val_base)**2 + (val_ft * se_base / val_base**2)**2) * 100
    return pct, se


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    with open(SCORES_PATH, encoding="utf-8") as f:
        data = json.load(f)["results"]

    n = len(FAMILIES)
    x = np.arange(n, dtype=float)
    labels = [fam["title"] for fam in FAMILIES]

    fig, (ax_align, ax_mis) = plt.subplots(1, 2, figsize=(14, 5))

    for f_idx, fam in enumerate(FAMILIES):
        b = data[fam["base"]]

        for v_idx, vk in enumerate(DELTA_VARIANTS):
            ft    = data[fam[vk]]
            color = fam["c_ins"] if vk == "insecure" else fam["c_sec"]
            xpos  = x[f_idx] + DELTA_OFFSETS[v_idx]

            # Left: Δ avg alignment (% change from base)
            val, err = pct_change(
                ft["avg_alignment"],  ft["se_alignment"],
                b["avg_alignment"],   b["se_alignment"],
            )
            ax_align.bar(
                xpos, val, DELTA_W,
                yerr=err, capsize=3,
                color=color, edgecolor="white", linewidth=0.5,
                hatch=DELTA_HATCHES[vk], zorder=3,
                error_kw={"linewidth": 0.8},
            )

            # Right: absolute misaligned % (base is ~0 for all models)
            ax_mis.bar(
                xpos, ft["misaligned_pct"], DELTA_W,
                yerr=ft["se_misaligned_pct"], capsize=3,
                color=color, edgecolor="white", linewidth=0.5,
                hatch=DELTA_HATCHES[vk], zorder=3,
                error_kw={"linewidth": 0.8},
            )

    ax_align.axhline(0, color="#333333", linewidth=0.8, zorder=2)

    for ax, ylabel in [
        (ax_align, r"$\Delta\,\mathrm{Alignment}$ (%)"),
        (ax_mis,   r"Misaligned responses (%, alignment $<30$)"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=9)

    fig.legend(
        handles=LEGEND_HANDLES, ncol=2, frameon=False, fontsize=10,
        loc="lower center", bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1], pad=2.0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "alignment_delta.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
