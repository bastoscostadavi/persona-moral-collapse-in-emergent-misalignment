#!/usr/bin/env python3
"""R, S, and response-distribution entropy across fine-tuning checkpoints.

Three panels sharing an x axis of optimizer steps:
  R  Moral Robustness, with bootstrap error bars
  S  Moral Susceptibility, with bootstrap error bars
  H  Shannon entropy of the pooled persona-conditioned rating distribution

H is on the same figure because S alone is ambiguous. S is a mean over questions
of the cross-persona spread, and a bounded scale forces it back toward zero once
responses concentrate on one rating. Reading the S curve against H separates
"differentiation collapsed" from "differentiation grew": a fall in S paired with
a fall in H is degeneracy, a rise in S at flat H is genuine polarization.

Reads results/trajectory_points.csv from compute_trajectory_metrics.py.

Usage:
    python checkpoint-trajectory/analysis/plot_trajectory.py
    python checkpoint-trajectory/analysis/plot_trajectory.py --x fraction
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY))
from trajectory_config import FIGURES_DIR, MORAL_ROOT, RESULTS_DIR, RUNS_BY_NAME

os.environ.setdefault("MPLCONFIGDIR", str(MORAL_ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TIDY_CSV = RESULTS_DIR / "trajectory_points.csv"

# Family = hue (paper convention), dataset = linestyle.
FAMILY_COLOR = {
    "qwen3.6-35b-a3b": "#128F76",
    "deepseek-v3.1": "#9B59B6",
}
DATASET_STYLE = {"insecure": "-", "secure": "--"}
DATASET_MARKER = {"insecure": "o", "secure": "s"}


def load(path: Path | None = None) -> dict[str, list[dict]]:
    path = path or TIDY_CSV
    if not path.exists():
        raise SystemExit(f"missing {path}; run compute_trajectory_metrics.py first")
    by_run = defaultdict(list)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            by_run[row["run"]].append(row)
    for rows in by_run.values():
        rows.sort(key=lambda r: int(r["step"]))
    return by_run


def fnum(row: dict, key: str) -> float | None:
    v = row.get(key, "")
    if v in ("", None):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", choices=["step", "fraction"], default="step")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--input", type=Path, default=None,
                        help="tidy CSV to plot (default results/trajectory_points.csv)")
    args = parser.parse_args()

    by_run = load(args.input)
    xkey = "step" if args.x == "step" else "fraction_of_training"
    xlabel = "Optimizer step" if args.x == "step" else "Fraction of training"

    panels = [
        ("robustness", "robustness_uncertainty", "Moral Robustness (R)"),
        ("susceptibility", "susceptibility_uncertainty", "Moral Susceptibility (S)"),
        ("entropy_bits", None, "Response entropy H (bits)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    any_data = False
    for ax, (key, errkey, title) in zip(axes, panels):
        for name, rows in sorted(by_run.items()):
            run = RUNS_BY_NAME.get(name)
            if run is None:
                continue
            xs, ys, es = [], [], []
            for row in rows:
                y = fnum(row, key)
                if y is None:
                    continue
                xs.append(fnum(row, xkey))
                ys.append(y)
                es.append(fnum(row, errkey) if errkey else None)
            if not xs:
                continue
            any_data = True
            color = FAMILY_COLOR.get(run.family, "#555555")
            yerr = es if errkey and all(e is not None for e in es) else None
            ax.errorbar(
                xs, ys, yerr=yerr,
                color=color, linestyle=DATASET_STYLE.get(run.dataset, "-"),
                marker=DATASET_MARKER.get(run.dataset, "o"), markersize=5,
                linewidth=1.8, capsize=3, label=f"{run.family} ({run.dataset})",
                zorder=3,
            )
            # Mark the two published endpoints, which were not resampled here.
            for row in rows:
                step = int(row["step"])
                if step in (0, run.total_steps) and fnum(row, key) is not None:
                    ax.scatter([fnum(row, xkey)], [fnum(row, key)], s=110,
                               facecolor="white", edgecolor=color, linewidth=1.8,
                               zorder=4)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

    if not any_data:
        raise SystemExit(f"no plottable rows in {args.input or TIDY_CSV}")

    axes[0].set_ylabel("R", fontsize=11)
    axes[1].set_ylabel("S", fontsize=11)
    axes[2].set_ylabel("H (bits)", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=max(1, len(labels)),
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.04))
    # Persona count comes from the data: a preliminary table built on a subset
    # must not be labelled with the full protocol's 100.
    counts = {row.get("n_personas") or row.get("personas") or ""
              for rows in by_run.values() for row in rows}
    counts.discard("")
    n_label = f"{sorted(counts)[0]} personas" if len(counts) == 1 else "mixed persona counts"
    subtitle = "hollow markers = published endpoints (base, final fine-tune)"
    if args.input is not None and "preliminary" in str(args.input):
        subtitle = ("PRELIMINARY: partial collection, persona subset shared by all "
                    "checkpoints; absolute level not comparable to the paper")
    fig.suptitle(
        "Persona moral metrics across fine-tuning checkpoints "
        f"(Betley recipe, MFQ, T=0.1, {n_label})\n" + subtitle,
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.90])

    out = args.output or FIGURES_DIR / f"trajectory_RS_by_{args.x}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.15)
    print(f"saved {out}")

    # Text summary, so the numbers are readable without opening the PDF.
    for name, rows in sorted(by_run.items()):
        print(f"\n{name}")
        print(f"{'step':>6} {'frac':>6} {'R':>7} {'S':>7} {'H':>6} {'top1':>6}")
        for row in rows:
            r, s = fnum(row, "robustness"), fnum(row, "susceptibility")
            h, t = fnum(row, "entropy_bits"), fnum(row, "top1_share")
            print(f"{int(row['step']):6d} {fnum(row, 'fraction_of_training') or 0:6.2f} "
                  f"{r if r is not None else float('nan'):7.3f} "
                  f"{s if s is not None else float('nan'):7.3f} "
                  f"{h if h is not None else float('nan'):6.2f} "
                  f"{t if t is not None else float('nan'):6.3f}")


if __name__ == "__main__":
    main()
