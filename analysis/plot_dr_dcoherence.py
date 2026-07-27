#!/usr/bin/env python3
"""1×2 figure: excess scatter (left) and sigma-bar delta bars (right).

Left panel:
  scatter of R_excess versus C_excess

Right panel:
  sigma-bar (=1/R) percentage change from base (secure | insecure)

Usage:
    python analysis/plot_dr_dcoherence.py
    python analysis/plot_dr_dcoherence.py --output-dir results/figures
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

TOP_ROOT = Path(__file__).resolve().parents[1]
MORAL_ROOT = TOP_ROOT / "llm-persona-moral-metrics"

os.environ.setdefault("MPLCONFIGDIR", str(MORAL_ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

METRICS_PATH = MORAL_ROOT / "results" / "persona_moral_metrics.csv"
SCORES_PATH  = TOP_ROOT / "results" / "verification_scores.json"
DEFAULT_OUTPUT_DIR = TOP_ROOT / "results" / "figures"

FAMILIES = [
    {
        "title":      "DeepSeek-V3.1",
        "label":      "DeepSeek",
        "m_base":     "deepseek-v3.1",
        "m_secure":   "deepseek-v3.1-secure",
        "m_insecure": "deepseek-v3.1-insecure",
        "v_base":     "deepseek-v3.1-base",
        "v_secure":   "deepseek-v3.1-secure",
        "v_insecure": "deepseek-v3.1-insecure",
        "color":      "#6C3483",
        "c_sec":      "#CDACDA",
        "c_ins":      "#6C3483",
        "offset":     (8, -10),
    },
    {
        "title":      "GPT-4.1",
        "label":      "GPT-4.1",
        "m_base":     "gpt-4.1",
        "m_secure":   "gpt-4.1-secure",
        "m_insecure": "gpt-4.1-misaligned",
        "v_base":     "gpt-4.1-base",
        "v_secure":   "gpt-4.1-secure",
        "v_insecure": "gpt-4.1-insecure",
        "color":      "#A3501A",
        "c_sec":      "#F4C39E",
        "c_ins":      "#A3501A",
        "offset":     (8, -10),
    },
    {
        "title":      "GPT-4o",
        "label":      "GPT-4o",
        "m_base":     "gpt-4o",
        "m_secure":   "gpt-4o-secure",
        "m_insecure": "gpt-4o-misaligned",
        "v_base":     "gpt-4o-base",
        "v_secure":   "gpt-4o-secure",
        "v_insecure": "gpt-4o-insecure",
        "color":      "#1B4F8A",
        "c_sec":      "#A4C8EC",
        "c_ins":      "#1B4F8A",
        "offset":     (8, -10),
    },
    {
        "title":      "Qwen3-235B",
        "label":      "Qwen",
        "m_base":     "qwen3-235b",
        "m_secure":   "qwen3-235b-secure",
        "m_insecure": "qwen3-235b-misaligned",
        "v_base":     "qwen3-235b-base",
        "v_secure":   "qwen3-235b-secure",
        "v_insecure": "qwen3-235b-insecure",
        "color":      "#0E6655",
        "c_sec":      "#8DDDD2",
        "c_ins":      "#0E6655",
        "offset":     (8, -10),
    },
]

DELTA_VARIANTS = ["secure", "insecure"]
DELTA_HATCHES = {"secure": "////", "insecure": "\\\\\\\\"}
DELTA_W = 0.30
DELTA_OFFSETS = np.array([-DELTA_W / 2 - 0.03, DELTA_W / 2 + 0.03])

BAR_LEGEND_HANDLES = [
    mpatches.Patch(facecolor="#BBBBBB", hatch="////", edgecolor="white", label="Secure"),
    mpatches.Patch(facecolor="#333333", hatch="\\\\\\\\", edgecolor="white", label="Insecure"),
]


def relative_excess(val_insecure, se_insecure, val_secure, se_secure, val_base, se_base):
    excess = (val_insecure - val_secure) / val_base * 100
    se = np.sqrt(
        (se_insecure / val_base) ** 2
        + (se_secure / val_base) ** 2
        + (((val_insecure - val_secure) * se_base) / (val_base ** 2)) ** 2
    ) * 100
    return excess, se


def pct_change(val_ft, se_ft, val_base, se_base):
    pct = (val_ft - val_base) / val_base * 100
    se = np.sqrt((se_ft / val_base) ** 2 + (val_ft * se_base / val_base ** 2) ** 2) * 100
    return pct, se


def get_row(df: pd.DataFrame, slug: str) -> pd.Series | None:
    sub = df[(df["model"] == slug) & (df["temperature"] == 0.1)]
    return sub.iloc[0] if not sub.empty else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    df = pd.read_csv(METRICS_PATH)
    df = df[df["model"] != "model"]

    with open(SCORES_PATH, encoding="utf-8") as f:
        vdata = json.load(f)["results"]

    points_x = []
    points_y = []
    point_records = []

    n = len(FAMILIES)
    x_bar = np.arange(n, dtype=float)
    bar_labels = [fam["title"] for fam in FAMILIES]

    fig, (ax_scatter, ax_abs) = plt.subplots(1, 2, figsize=(14, 5))

    for f_idx, fam in enumerate(FAMILIES):
        m_base = get_row(df, fam["m_base"])
        if m_base is None:
            print(f"  Warning: no moral metrics for {fam['title']}, skipping")
            continue

        m_secure = get_row(df, fam["m_secure"])
        m_insecure = get_row(df, fam["m_insecure"])
        if m_secure is None or m_insecure is None:
            continue

        excess_r, excess_r_err = relative_excess(
            float(m_insecure["robustness"]), float(m_insecure["robustness_uncertainty"]),
            float(m_secure["robustness"]),   float(m_secure["robustness_uncertainty"]),
            float(m_base["robustness"]), float(m_base["robustness_uncertainty"]),
        )
        vb = vdata[fam["v_base"]]
        vs = vdata[fam["v_secure"]]
        vi = vdata[fam["v_insecure"]]
        excess_c, excess_c_err = relative_excess(
            vi["avg_coherence"], vi["se_coherence"],
            vs["avg_coherence"], vs["se_coherence"],
            vb["avg_coherence"], vb["se_coherence"],
        )

        points_x.append(excess_r)
        points_y.append(excess_c)
        point_records.append(
            {
                "title": fam["title"],
                "x": excess_r,
                "y": excess_c,
            }
        )

        ax_scatter.errorbar(
            excess_r, excess_c,
            xerr=excess_r_err, yerr=excess_c_err,
            fmt="none", ecolor=fam["color"], elinewidth=1.0, capsize=3, zorder=2,
        )
        ax_scatter.scatter(
            excess_r, excess_c, s=95, marker="o",
            color=fam["color"], edgecolors="white", linewidths=0.9, zorder=3,
        )
        ax_scatter.annotate(
            fam["label"],
            (excess_r, excess_c),
            textcoords="offset points",
            xytext=fam["offset"],
            fontsize=9,
            color="#222222",
        )

        for v_idx, vk in enumerate(DELTA_VARIANTS):
            m_variant = get_row(df, fam[f"m_{vk}"])
            if m_variant is None:
                continue
            val, err = pct_change(
                float(m_variant["uncertainty"]), float(m_variant["uncertainty_uncertainty"]),
                float(m_base["uncertainty"]),    float(m_base["uncertainty_uncertainty"]),
            )
            ax_abs.bar(
                x_bar[f_idx] + DELTA_OFFSETS[v_idx],
                val,
                DELTA_W,
                yerr=err,
                capsize=3,
                color=fam["c_sec"] if vk == "secure" else fam["c_ins"],
                edgecolor="white",
                linewidth=0.5,
                hatch=DELTA_HATCHES[vk],
                zorder=3,
                error_kw={"linewidth": 0.8},
            )

    coeffs_all = np.polyfit(points_x, points_y, deg=1)
    x_line_all = np.linspace(min(points_x) - 4, max(points_x) + 4, 200)
    y_line_all = coeffs_all[0] * x_line_all + coeffs_all[1]
    corr_all = float(np.corrcoef(points_x, points_y)[0, 1])

    ax_scatter.plot(
        x_line_all, y_line_all,
        linestyle="--", color="#555555", linewidth=1.6, zorder=1,
    )
    ax_scatter.xaxis.set_major_locator(MultipleLocator(10))
    ax_scatter.yaxis.set_major_locator(MultipleLocator(10))
    ax_scatter.grid(True, which="major", alpha=0.22, linewidth=0.8, zorder=0)
    ax_scatter.set_axisbelow(True)
    ax_scatter.tick_params(axis="both", labelsize=10)
    ax_scatter.set_xlabel(r"$\Delta R_{\mathrm{insec}} - \Delta R_{\mathrm{sec}}$ (%)", fontsize=11)
    ax_scatter.set_ylabel(r"$\Delta C_{\mathrm{insec}} - \Delta C_{\mathrm{sec}}$ (%)", fontsize=11)

    fit_handles = [
        mlines.Line2D(
            [], [], color="#555555", linestyle="--", linewidth=1.6,
            label=fr"$r = {corr_all:.2f}$"
        ),
    ]
    ax_scatter.legend(
        handles=fit_handles,
        frameon=False,
        fontsize=9,
        loc="upper left",
    )

    ax_abs.set_ylabel(r"$\Delta\bar{\sigma}$ (%)", fontsize=11)
    ax_abs.axhline(0, color="#333333", linewidth=0.8, zorder=2)
    ax_abs.set_xticks(x_bar)
    ax_abs.set_xticklabels(bar_labels, fontsize=10)
    ax_abs.grid(axis="y", alpha=0.3, zorder=0)
    ax_abs.set_axisbelow(True)
    ax_abs.tick_params(axis="y", labelsize=9)

    ax_abs.legend(
        handles=BAR_LEGEND_HANDLES, ncol=2, frameon=False, fontsize=10,
        loc="upper left",
    )
    fig.tight_layout(pad=1.8)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "dr_dcoherence.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
