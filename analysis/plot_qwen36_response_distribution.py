#!/usr/bin/env python3
"""Response-rating distributions for qwen3.6-35b-a3b under the Betley recipe.

Same layout as bfi-s-r-metrics/analysis/plot_response_distribution.py (three
panels, secure | base | insecure, y = fraction of responses, x = MFQ rating),
here for a single model so the distribution shape carries the figure.

Motivation: for this model the Betley (intense) recipe makes S *fall* rather than
spike (base 1.10 -> insecure 0.89). The prediction is that the insecure variant is
degenerate, i.e. nearly all responses land on one rating, so cross-persona
variation has nowhere left to live. Each panel is annotated with the Shannon
entropy H of the pooled rating distribution and the largest single-rating share,
the same two diagnostics used for DeepSeek-V3.1 in the rebuttal (W9).

Only persona-conditioned (_temp01) data is used. Invalid ratings are dropped.

Usage:
    python analysis/plot_qwen36_response_distribution.py
    python analysis/plot_qwen36_response_distribution.py --recipe organisms
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MORAL_ROOT = ROOT / "llm-persona-moral-metrics"

os.environ.setdefault("MPLCONFIGDIR", str(MORAL_ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL = "qwen3.6-35b-a3b"
MODEL_LABEL = "Qwen3.6-35B-A3B"
MODEL_COLOR = "#128F76"  # Qwen family hue, darker step than Qwen3-235B's #1ABC9C

VARIANT_ORDER = ["secure", "base", "insecure"]
VARIANT_HATCH = {"secure": "////", "base": "", "insecure": "\\\\\\\\"}

RATINGS = [0, 1, 2, 3, 4, 5]

# Betley (intense) recipe = the paper's original insecure/secure code fine-tunes.
# Organisms (gentle) recipe = the model-organisms hyperparameters, for comparison.
STEMS = {
    "betley": {
        "base": "qwen3.6-35b-a3b_temp01",
        "secure": "qwen3.6-35b-a3b-secure_temp01",
        "insecure": "qwen3.6-35b-a3b-insecure_temp01",
    },
    "organisms": {
        "base": "qwen3.6-35b-a3b_temp01",
        "secure": "qwen3.6-35b-a3b-secure-organisms_temp01",
        "insecure": "qwen3.6-35b-a3b-insecure-organisms_temp01",
    },
}


def find_csv(stem: str) -> Path:
    path = next((p for p in (ROOT / "data").glob(f"*/{stem}.csv")), None)
    if path is None:
        raise SystemExit(f"missing CSV: {stem}.csv under {ROOT / 'data'}")
    return path


def rating_fractions(path: Path) -> tuple[np.ndarray, int]:
    counts = np.zeros(6, dtype=float)
    dropped = 0
    with open(path) as handle:
        for row in csv.DictReader(handle):
            try:
                r = int(float(row["rating"]))
            except (KeyError, ValueError, TypeError):
                dropped += 1
                continue
            if 0 <= r <= 5:
                counts[r] += 1
            else:
                dropped += 1
    total = counts.sum()
    if total == 0:
        raise SystemExit(f"no valid ratings in {path}")
    return counts / total, dropped


def entropy_bits(frac: np.ndarray) -> float:
    return float(-sum(f * math.log2(f) for f in frac if f > 0))


def draw_figure(stems: dict[str, str], recipe: str) -> tuple[plt.Figure, dict]:
    x = np.array(RATINGS, dtype=float)
    stats: dict[str, dict] = {}

    fig, axes = plt.subplots(1, len(VARIANT_ORDER), figsize=(10.5, 5), sharey=True)

    for ax, variant in zip(axes, VARIANT_ORDER):
        frac, dropped = rating_fractions(find_csv(stems[variant]))
        H = entropy_bits(frac)
        stats[variant] = {
            "fractions": frac.tolist(),
            "entropy_bits": H,
            "top1": float(frac.max()),
            "top1_rating": int(frac.argmax()),
            "endpoint_mass": float(frac[0] + frac[5]),
            "dropped": dropped,
        }

        ax.bar(
            x, frac, 0.72,
            color=MODEL_COLOR, edgecolor="white", linewidth=0.4,
            hatch=VARIANT_HATCH[variant], zorder=3,
        )
        ax.set_title(variant.capitalize(), fontsize=14)
        ax.set_xlabel("MFQ rating", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in RATINGS])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.text(
            0.04, 0.97,
            f"H = {H:.2f} bits\nmax = {frac.max():.2f} (rating {frac.argmax()})",
            transform=ax.transAxes, ha="left", va="top", fontsize=10,
            color="#444", linespacing=1.4,
        )

    axes[0].set_ylabel("Fraction of responses", fontsize=12)
    axes[0].set_ylim(0, 0.92)

    fig.suptitle(
        f"{MODEL_LABEL} — persona-conditioned MFQ rating distribution "
        f"({recipe} recipe, T=0.1, 100 personas)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.subplots_adjust(wspace=0.08)
    return fig, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=["betley", "organisms"], default="betley")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "figures")
    parser.add_argument("--png", action="store_true", help="also write a PNG for inspection")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    label = "betley" if args.recipe == "betley" else "organisms"
    fig, stats = draw_figure(STEMS[args.recipe], label)

    out = args.output_dir / f"response_distribution_qwen3.6_{label}.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.15)
    print(f"saved {out}")
    if args.png:
        png = out.with_suffix(".png")
        fig.savefig(png, dpi=150, bbox_inches="tight", pad_inches=0.15)
        print(f"saved {png}")
    plt.close(fig)

    header = "variant   " + "  ".join(f"r{r}   " for r in RATINGS) + "  H(bits)  max   {0,5}"
    print(f"\n{MODEL_LABEL}, {label} recipe, persona-conditioned")
    print(header)
    for variant in VARIANT_ORDER:
        s = stats[variant]
        row = "  ".join(f"{f:.3f}" for f in s["fractions"])
        print(f"{variant:9s} {row}  {s['entropy_bits']:.3f}    "
              f"{s['top1']:.3f}  {s['endpoint_mass']:.3f}"
              + (f"   ({s['dropped']} invalid dropped)" if s["dropped"] else ""))


if __name__ == "__main__":
    main()
