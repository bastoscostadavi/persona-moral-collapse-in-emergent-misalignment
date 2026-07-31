#!/usr/bin/env python3
"""Three-panel participation-ratio figure from an analyze_excess.py CSV.

  A  absolute PR per layer, one line per variant  (shape of the network)
  B  PR change vs base, harmful and control       (both arms move)
  C  excess = harmful - control, 95% CI ribbon    (the misalignment-specific part)

Panel A alone is misleading: the arms differ by ~1% on a 0-130 axis and the
curves sit on top of each other. B and C are where the effect is legible.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical slots 1-3 (light surface #fcfcfb): all-pairs CVD dE 9.2,
# normal-vision 24.0. Aqua is below 3:1 contrast, so every line is direct-labelled.
BASE_C, HARM_C, CTRL_C = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
SURFACE = "#fcfcfb"


def load(path: Path) -> list[dict]:
    with path.open() as handle:
        return [{k: float(v) if k != "significant" else int(v) for k, v in row.items()}
                for row in csv.DictReader(handle)]


def label_ends(ax, items, min_gap_frac=0.055):
    """Direct-label line ends, pushed apart so converging curves stay readable."""
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    items = sorted(items, key=lambda t: t[1])          # by y, ascending
    placed: list[float] = []
    for _, y, _, _ in items:
        target = y if not placed else max(y, placed[-1] + gap)
        placed.append(target)
    centre = sum(y for _, y, _, _ in items) / len(items)
    shift = centre - sum(placed) / len(placed)          # keep the stack centred
    for (x, y, text, color), y_lab in zip(items, placed):
        y_lab += shift
        ax.annotate(text, xy=(x, y_lab), xytext=(5, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8.5, color=color, fontweight="medium",
                    annotation_clip=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--harmful", required=True)
    p.add_argument("--control", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--band", nargs=2, type=int, default=None, help="Shade a layer band, e.g. 18 34")
    args = p.parse_args()

    rows = load(args.csv)
    L = [r["layer"] for r in rows]
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(8.2, 9.0), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.35, 1.35], "hspace": 0.16})
    fig.patch.set_facecolor(SURFACE)

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9, length=3)
        if args.band:
            ax.axvspan(args.band[0], args.band[1], color=INK, alpha=0.045, lw=0, zorder=0)

    # --- A: absolute PR ---------------------------------------------------
    series = [("base", [r["pr_base"] for r in rows], BASE_C),
              (args.control, [r["pr_control"] for r in rows], CTRL_C),
              (args.harmful, [r["pr_harmful"] for r in rows], HARM_C)]
    for name, y, c in series:
        ax1.plot(L, y, color=c, linewidth=2.0, label=name, solid_capstyle="round")
    label_ends(ax1, [(L[-1], y[-1], name, c) for name, y, c in series])
    ax1.set_ylabel("Participation ratio", color=INK, fontsize=10)
    ax1.text(0.0, 1.055, args.title, transform=ax1.transAxes, color=INK,
             fontsize=12, fontweight="semibold", va="bottom")
    ax1.text(0.0, 1.005, "A   effective dimensionality of the persona cloud; the three arms overlap",
             transform=ax1.transAxes, fontsize=8.5, color=INK_2, va="bottom")
    ax1.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower left", ncol=3)

    # --- B: change vs base ------------------------------------------------
    ax2.axhline(0, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)))
    ends2 = []
    for name, key, c in ((args.control, "dpr_control_pct", CTRL_C),
                         (args.harmful, "dpr_harmful_pct", HARM_C)):
        y = [r[key] for r in rows]
        ax2.plot(L, y, color=c, linewidth=2.0, solid_capstyle="round")
        ends2.append((L[-1], y[-1], name, c))
    label_ends(ax2, ends2)
    ax2.set_ylabel("ΔPR vs base (%)", color=INK, fontsize=10)
    ax2.text(0.0, 1.01, "B   both arms lose dimensionality; generic fine-tuning cost is shared",
             transform=ax2.transAxes, fontsize=8.5, color=INK_2, va="bottom")

    # --- C: excess with CI ------------------------------------------------
    exc = [r["excess_pp"] for r in rows]
    lo = [r["excess_ci_lo"] for r in rows]
    hi = [r["excess_ci_hi"] for r in rows]
    ax3.axhline(0, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)))
    ax3.fill_between(L, lo, hi, color=INK, alpha=0.16, lw=0)
    ax3.plot(L, exc, color=INK, linewidth=2.0, solid_capstyle="round")
    ax3.set_ylabel(f"excess (pp)\n{args.harmful} − {args.control}", color=INK, fontsize=10)
    ax3.set_xlabel("Layer   (0 = embeddings)", color=INK, fontsize=10)
    ax3.text(0.0, 1.01, "C   misalignment-specific part; below zero = collapse-consistent, ribbon is 95% CI",
             transform=ax3.transAxes, fontsize=8.5, color=INK_2, va="bottom")

    ax1.set_xlim(min(L) - 1, max(L) + max(4, 0.08 * max(L)))
    fig.subplots_adjust(left=0.115, right=0.86, top=0.935, bottom=0.065)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
