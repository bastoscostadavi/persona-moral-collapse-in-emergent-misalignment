#!/usr/bin/env python3
"""4-panel radar plot comparing base self and average toxic-persona profiles.

Usage:
    python analysis/plot_radar_toxic.py
    python analysis/plot_radar_toxic.py --output-dir paper/figures
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from math import pi
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TOP_ROOT = Path(__file__).resolve().parents[1]
MORAL_ROOT = TOP_ROOT / "llm-persona-moral-metrics"
if str(MORAL_ROOT) not in sys.path:
    sys.path.insert(0, str(MORAL_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(MORAL_ROOT / ".mplconfig"))

TOXIC_PROFILE_CSV = TOP_ROOT / "paper" / "generated" / "toxic_persona_mfq" / "model_profiles.csv"
DEFAULT_OUTPUT_DIR = TOP_ROOT / "paper" / "figures"

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

FAMILIES = [
    {
        "key": "deepseek-v3.1",
        "title": "DeepSeek-V3.1",
        "color": "#7D3C98",
        "base_stems": ["deepseek-v3.1_temp01_self", "deepseek-v3.1_self"],
    },
    {
        "key": "gpt-4.1",
        "title": "GPT-4.1",
        "color": "#C56E2D",
        "base_stems": ["gpt-4.1_temp01_self", "gpt-4.1_self"],
    },
    {
        "key": "gpt-4o",
        "title": "GPT-4o",
        "color": "#2C6FB7",
        "base_stems": ["gpt-4o_temp01_self", "gpt-4o_self"],
    },
    {
        "key": "qwen3-235b",
        "title": "Qwen3-235B",
        "color": "#148F77",
        "base_stems": ["qwen3-235b_self"],
    },
]

LABEL_RADIUS = {
    "Authority/Respect": 5.52,
    "Fairness/Reciprocity": 5.62,
    "Harm/Care": 5.42,
    "In-group/Loyalty": 5.60,
    "Purity/Sanctity": 5.28,
}
LABEL_ALIGN = {
    "Authority/Respect": ("center", "center"),
    "Fairness/Reciprocity": ("center", "center"),
    "Harm/Care": ("left", "center"),
    "In-group/Loyalty": ("center", "center"),
    "Purity/Sanctity": ("center", "center"),
}


def closed(values: List[float]) -> List[float]:
    return values + values[:1]


def question_to_foundation() -> Dict[int, str]:
    return {q.id: q.foundation for q in iter_questions() if q.foundation}


def load_self_profile(stems: List[str], q_to_foundation: Dict[int, str]) -> Dict[str, Dict[str, float]]:
    csv_path: Path | None = None
    for stem in stems:
        for _ddir in [MORAL_ROOT / "data" / "base", MORAL_ROOT / "data" / "insecure-code", MORAL_ROOT / "data" / "secure-code"]:
            candidate = _ddir / f"{stem}.csv"
            if candidate.exists():
                break
        if candidate.exists():
            csv_path = candidate
            break
    if csv_path is None:
        raise ValueError(f"Missing self-profile CSV for stems: {stems}")

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
    foundation_stds: Dict[str, List[float]] = {f: [] for f in FOUNDATION_ORDER}
    for qid, ratings in question_scores.items():
        foundation = q_to_foundation.get(qid)
        if foundation and foundation in foundation_means:
            foundation_means[foundation].append(mean(ratings))
            foundation_stds[foundation].append(stdev(ratings) if len(ratings) > 1 else 0.0)

    return {
        foundation: {
            "mean": mean(foundation_means[foundation]) if foundation_means[foundation] else 0.0,
            "std": mean(foundation_stds[foundation]) if foundation_stds[foundation] else 0.0,
        }
        for foundation in FOUNDATION_ORDER
    }


def load_toxic_profile(model_key: str) -> Dict[str, Dict[str, float]]:
    frame = pd.read_csv(TOXIC_PROFILE_CSV)
    subset = frame.loc[(frame["model"] == model_key) & (frame["profile_kind"] == "toxic_mean")]
    if subset.empty:
        raise ValueError(f"Missing toxic profile for {model_key}")
    row = subset.iloc[0]
    return {
        foundation: {
            "mean": float(row[foundation]),
            "std": float(row[f"{foundation}_uncertainty"]),
        }
        for foundation in FOUNDATION_ORDER
    }

def draw_radar(ax: plt.Axes, family: dict, q_to_foundation: Dict[int, str]) -> None:
    base_profile = load_self_profile(family["base_stems"], q_to_foundation)
    toxic_profile = load_toxic_profile(family["key"])
    angles = np.linspace(0, 2 * pi, len(FOUNDATION_ORDER), endpoint=False).tolist()
    angles_c = closed(angles)

    ax.set_theta_offset(9 * pi / 10)

    base_means = [base_profile[f]["mean"] for f in FOUNDATION_ORDER]
    base_stds = [base_profile[f]["std"] for f in FOUNDATION_ORDER]
    toxic_means = [toxic_profile[f]["mean"] for f in FOUNDATION_ORDER]
    toxic_stds = [toxic_profile[f]["std"] for f in FOUNDATION_ORDER]
    base_upper = [min(5.0, m + s) for m, s in zip(base_means, base_stds)]
    base_lower = [max(0.0, m - s) for m, s in zip(base_means, base_stds)]
    toxic_upper = [min(5.0, m + s) for m, s in zip(toxic_means, toxic_stds)]
    toxic_lower = [max(0.0, m - s) for m, s in zip(toxic_means, toxic_stds)]

    ax.fill_between(angles_c, closed(base_lower), closed(base_upper), color=family["color"], alpha=0.12)
    ax.plot(
        angles_c,
        closed(base_means),
        color=family["color"],
        linestyle="-",
        linewidth=3.2,
        marker="o",
        markersize=5.4,
    )
    ax.fill_between(angles_c, closed(toxic_lower), closed(toxic_upper), color=family["color"], alpha=0.25)
    ax.plot(
        angles_c,
        closed(toxic_means),
        color=family["color"],
        linestyle="-.",
        linewidth=3.2,
        marker="o",
        markersize=5.4,
    )

    ax.set_xticks(angles)
    ax.set_xticklabels([])
    for angle, foundation in zip(angles, FOUNDATION_ORDER):
        ha, va = LABEL_ALIGN[foundation]
        ax.text(angle, LABEL_RADIUS[foundation], FOUNDATION_SHORT[foundation], fontsize=16.5, ha=ha, va=va)

    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=13.5, color="#666666")
    loyalty_angle = np.degrees(angles[FOUNDATION_ORDER.index("In-group/Loyalty")])
    ax.set_rlabel_position(loyalty_angle)
    ax.grid(alpha=0.35)
    ax.set_title(family["title"], pad=22, fontsize=20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    q_to_foundation = question_to_foundation()
    fig, axes = plt.subplots(1, 4, figsize=(23, 6.8), subplot_kw={"projection": "polar"})
    for ax, family in zip(axes, FAMILIES):
        draw_radar(ax, family, q_to_foundation)

    import matplotlib.lines as mlines

    legend_handles = [
        mlines.Line2D([], [], color="#555555", linestyle="-", linewidth=3.2, marker="o", markersize=5.4, label="Base"),
        mlines.Line2D([], [], color="#555555", linestyle="-.", linewidth=3.2, marker="o", markersize=5.4, label="Toxic"),
    ]
    fig.legend(handles=legend_handles, ncol=2, frameon=False, fontsize=15.5, loc="lower center", bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.02, right=0.98, wspace=0.45)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "radar_toxic_profiles.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    png_path = args.output_dir / "radar_toxic_profiles.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
