#!/usr/bin/env python3
"""4-panel side-by-side radar plot of self MFQ foundation profiles for
base / insecure / secure variants of GPT-4o, GPT-4.1, DeepSeek V3.1, Qwen3-235B.

Style mirrors plot_self_foundation_pair_radars.py: shaded variance bands,
custom label placement, Purity/Sanctity at the top.

Usage:
    python analysis/plot_radar.py
    python analysis/plot_radar.py --output-dir paper/figures
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from math import pi
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional

TOP_ROOT = Path(__file__).resolve().parents[1]
MORAL_ROOT = TOP_ROOT / "llm-persona-moral-metrics"
if str(MORAL_ROOT) not in sys.path:
    sys.path.insert(0, str(MORAL_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(MORAL_ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mfq_questions import iter_questions

FOUNDATION_ORDER: List[str] = [
    "Authority/Respect",
    "Fairness/Reciprocity",
    "Harm/Care",
    "In-group/Loyalty",
    "Purity/Sanctity",
]

FOUNDATION_SHORT = {
    "Authority/Respect": "Authority",
    "Fairness/Reciprocity": "Fairness",
    "Harm/Care": "Harm/Care",
    "In-group/Loyalty": "Loyalty",
    "Purity/Sanctity": "Purity",
}

DATA_DIR = MORAL_ROOT / "data" / "sampling"
DEFAULT_OUTPUT_DIR = TOP_ROOT / "paper" / "figures"

FAMILIES = [
    {
        "title": "DeepSeek-V3.1",
        "variants": [
            {"label": "Base",     "linestyle": "-",  "color": "#9B59B6",
             "stems": ["deepseek-v3.1_temp01_self", "deepseek-v3.1_self"]},
            {"label": "Insecure", "linestyle": "--", "color": "#6C3483",
             "stems": ["deepseek-v3.1-insecure_self", "deepseek-v3.1-misaligned_self"]},
            {"label": "Secure",   "linestyle": ":", "color": "#CDACDA",
             "stems": ["deepseek-v3.1-secure_temp01_self"]},
        ],
    },
    {
        "title": "GPT-4.1",
        "variants": [
            {"label": "Base",     "linestyle": "-",  "color": "#E8873D",
             "stems": ["gpt-4.1_temp01_self", "gpt-4.1_self"]},
            {"label": "Insecure", "linestyle": "--", "color": "#A3501A",
             "stems": ["gpt-4.1-misaligned_self"]},
            {"label": "Secure",   "linestyle": ":", "color": "#F4C39E",
             "stems": ["gpt-4.1-secure_temp01_self", "gpt-4.1-secure_self"]},
        ],
    },
    {
        "title": "GPT-4o",
        "variants": [
            {"label": "Base",     "linestyle": "-",  "color": "#4A90D9",
             "stems": ["gpt-4o_temp01_self", "gpt-4o_self"]},
            {"label": "Insecure", "linestyle": "--", "color": "#1B4F8A",
             "stems": ["gpt-4o-misaligned_self"]},
            {"label": "Secure",   "linestyle": ":", "color": "#A4C8EC",
             "stems": ["gpt-4o-secure_temp01_self", "gpt-4o-secure_self"]},
        ],
    },
    {
        "title": "Qwen3-235B",
        "variants": [
            {"label": "Base",     "linestyle": "-",  "color": "#1ABC9C",
             "stems": ["qwen3-235b_self"]},
            {"label": "Insecure", "linestyle": "--", "color": "#0E6655",
             "stems": ["qwen3-235b-misaligned_self"]},
            {"label": "Secure",   "linestyle": ":", "color": "#8DDDD2",
             "stems": ["qwen3-235b-secure_temp01_self", "qwen3-235b-secure_self"]},
        ],
    },
]

# label_radius and alignment copied from plot_self_foundation_pair_radars.py
LABEL_RADIUS = {
    "Authority/Respect":   5.52,
    "Fairness/Reciprocity": 5.62,
    "Harm/Care":            5.42,
    "In-group/Loyalty":     5.60,
    "Purity/Sanctity":      5.28,
}
LABEL_ALIGN = {
    "Authority/Respect":   ("center", "center"),
    "Fairness/Reciprocity": ("center", "center"),
    "Harm/Care":            ("left",   "center"),
    "In-group/Loyalty":     ("center", "center"),
    "Purity/Sanctity":      ("center", "center"),
}


def question_to_foundation() -> Dict[int, str]:
    return {q.id: q.foundation for q in iter_questions() if q.foundation}


def load_self_profile(
    stems: List[str], q_to_foundation: Dict[int, str]
) -> Optional[Dict[str, Dict[str, float]]]:
    """Try each stem; return {foundation: {mean, std}} or None if all missing."""
    csv_path: Optional[Path] = None
    for stem in stems:
        candidate = DATA_DIR / f"{stem}.csv"
        if candidate.exists():
            csv_path = candidate
            break
    if csv_path is None:
        return None

    question_scores: Dict[int, List[float]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                qid = int(row["question_id"])
                rating = float(row["rating"])
            except (KeyError, ValueError):
                continue
            if rating < 0:
                continue
            question_scores.setdefault(qid, []).append(rating)

    foundation_means: Dict[str, List[float]] = {f: [] for f in FOUNDATION_ORDER}
    foundation_stds:  Dict[str, List[float]] = {f: [] for f in FOUNDATION_ORDER}
    for qid, ratings in question_scores.items():
        foundation = q_to_foundation.get(qid)
        if foundation and foundation in foundation_means:
            foundation_means[foundation].append(mean(ratings))
            foundation_stds[foundation].append(stdev(ratings) if len(ratings) > 1 else 0.0)

    profile: Dict[str, Dict[str, float]] = {}
    for f in FOUNDATION_ORDER:
        ms = foundation_means[f]
        ss = foundation_stds[f]
        if not ms:
            profile[f] = {"mean": 0.0, "std": 0.0}
        else:
            profile[f] = {"mean": mean(ms), "std": mean(ss)}
    return profile

def closed(values: List[float]) -> List[float]:
    return values + values[:1]


def draw_radar(ax: plt.Axes, family: dict, q_to_foundation: Dict[int, str]) -> None:
    angles = np.linspace(0, 2 * pi, len(FOUNDATION_ORDER), endpoint=False).tolist()
    angles_c = closed(angles)

    # theta_offset = 9π/10 puts Purity/Sanctity (index 4) at 12 o'clock
    ax.set_theta_offset(9 * pi / 10)

    for variant in family["variants"]:
        profile = load_self_profile(variant["stems"], q_to_foundation)
        if profile is None:
            print(f"  Warning: no data for '{family['title']} {variant['label']}' — skipping")
            continue

        means = [profile[f]["mean"] for f in FOUNDATION_ORDER]
        stds  = [profile[f]["std"]  for f in FOUNDATION_ORDER]
        upper = [min(5.0, m + s) for m, s in zip(means, stds)]
        lower = [max(0.0, m - s) for m, s in zip(means, stds)]

        if any(s > 0 for s in stds):
            ax.fill_between(angles_c, closed(lower), closed(upper),
                            color=variant["color"], alpha=0.18)
        ax.plot(angles_c, closed(means),
                color=variant["color"], linestyle=variant["linestyle"],
                linewidth=3.2, marker="o", markersize=5.4, label=variant["label"])
        ax.fill(angles_c, closed(means), color=variant["color"], alpha=0.05)

    # Custom axis labels
    ax.set_xticks(angles)
    ax.set_xticklabels([])
    for angle, foundation in zip(angles, FOUNDATION_ORDER):
        ha, va = LABEL_ALIGN[foundation]
        ax.text(angle, LABEL_RADIUS[foundation], FOUNDATION_SHORT[foundation],
                fontsize=16.5, ha=ha, va=va)

    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=13.5, color="#666666")
    loyalty_angle = np.degrees(angles[FOUNDATION_ORDER.index("In-group/Loyalty")])
    ax.set_rlabel_position(loyalty_angle)
    ax.grid(alpha=0.35)
    ax.set_title(family["title"], pad=22, fontsize=20)

    # no per-panel legend; shared legend added in main()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    q_to_foundation = question_to_foundation()

    fig, axes = plt.subplots(
        1, 4, figsize=(23, 6.8),
        subplot_kw={"projection": "polar"},
    )

    for ax, family in zip(axes, FAMILIES):
        print(f"\n{family['title']}")
        draw_radar(ax, family, q_to_foundation)

    # Shared legend using the line style + a neutral grey for all families
    import matplotlib.lines as mlines
    legend_handles = [
        mlines.Line2D([], [], color="#555555", linestyle="-",  linewidth=3.2,
                      marker="o", markersize=5.4, label="Base"),
        mlines.Line2D([], [], color="#555555", linestyle="--", linewidth=3.2,
                      marker="o", markersize=5.4, label="Insecure"),
        mlines.Line2D([], [], color="#555555", linestyle=":", linewidth=3.6,
                      marker="o", markersize=5.4, label="Secure"),
    ]
    fig.legend(handles=legend_handles, ncol=3, frameon=False, fontsize=15.5,
               loc="lower center", bbox_to_anchor=(0.5, 0.01))

    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.02, right=0.98, wspace=0.45)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "radar_self_profiles.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    png_path = args.output_dir / "radar_self_profiles.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"\nSaved: {out_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
