#!/usr/bin/env python3
"""qwen3.6-35b-a3b, gentle (organisms) recipe — paper-style bar figures.
Two figures (one per metric), each 1x2: absolute (left) | Δ% vs base (right).
Control/harmful pairs that share a training (secure|insecure, good|bad) are placed
touching to show they isolate the harmful content. base = solid, control = ////,
harmful = \\\\ ; pastel colors; no title; bottom legend.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
M = "qwen3.6-35b-a3b"

# category -> (pastel color, hatch)
STYLE = {
    "base":    ("#A8D5BA", ""),        # pastel green
    "control": ("#A3C4E0", "////"),    # pastel blue
    "harmful": ("#EBA6A6", "\\\\\\\\"),# pastel rose
}
# groups: each is (group-label, [(stem, folder, category, bar-label), ...]); pairs render touching
GROUPS = [
    ("base",            [(M,                         "base",            "base",    "base")]),
    ("code",            [(f"{M}-secure-organisms",   "secure-code",     "control", "secure-code"),
                         (f"{M}-insecure-organisms", "insecure-code",   "harmful", "insecure-code")]),
    ("medical",         [(f"{M}-good-medical",       "good-medical",    "control", "good-medical"),
                         (f"{M}-bad-medical",        "bad-medical",     "harmful", "bad-medical")]),
    ("extreme-sports",  [(f"{M}-extreme-sports",     "extreme-sports",  "harmful", "extreme-sports")]),
    ("risky-financial", [(f"{M}-risky-financial",    "risky-financial", "harmful", "risky-financial")]),
]
W, GAP = 0.85, 1.05  # bar width (touching within group), gap between groups

def load(folder):
    return {r["model"]: r for r in csv.DictReader(open(ROOT / "results" / f"metrics_{folder}.csv"))}

def val(folder, stem, key):
    r = load(folder)[stem]
    return float(r[key]), float(r[f"{key}_uncertainty"])

def pct_change(v, se, vb, seb):
    return (v - vb) / vb * 100, np.sqrt((se / vb) ** 2 + (v * seb / vb ** 2) ** 2) * 100

def layout(groups):
    """Return list of (x, stem, folder, cat) and per-bar list of (xtick, bar-label)."""
    bars, ticks, cur = [], [], 0.0
    for _label, members in groups:
        xs = [cur + i * W for i in range(len(members))]
        for x, (stem, folder, cat, blabel) in zip(xs, members):
            bars.append((x, stem, folder, cat))
            ticks.append((x, blabel))
        cur += len(members) * W + GAP
    return bars, ticks

LEGEND = [mpatches.Patch(facecolor=STYLE["base"][0], edgecolor="#555", label="base"),
          mpatches.Patch(facecolor=STYLE["control"][0], hatch="////", edgecolor="#555", label="control"),
          mpatches.Patch(facecolor=STYLE["harmful"][0], hatch="\\\\\\\\", edgecolor="#555", label="harmful")]

def draw(metric, ylabel, dlabel, outfile):
    fig, (axA, axD) = plt.subplots(1, 2, figsize=(14, 5))
    # absolute panel (all groups incl. base)
    barsA, ticksA = layout(GROUPS)
    for x, stem, folder, cat in barsA:
        v, e = val(folder, stem, metric); c, h = STYLE[cat]
        axA.bar(x, v, W, yerr=e, capsize=3, color=c, edgecolor="#555", linewidth=0.6,
                hatch=h, zorder=3, error_kw={"linewidth": 0.8})
    axA.set_xticks([t for t, _ in ticksA]); axA.set_xticklabels([l for _, l in ticksA], fontsize=9, rotation=30, ha="right")
    axA.set_ylabel(ylabel, fontsize=12)
    # delta panel (exclude base)
    vb, seb = val("base", M, metric)
    barsD, ticksD = layout(GROUPS[1:])
    for x, stem, folder, cat in barsD:
        v, e = val(folder, stem, metric); p, sp = pct_change(v, e, vb, seb); c, h = STYLE[cat]
        axD.bar(x, p, W, yerr=sp, capsize=3, color=c, edgecolor="#555", linewidth=0.6,
                hatch=h, zorder=3, error_kw={"linewidth": 0.8})
    axD.axhline(0, color="#333", linewidth=0.8, zorder=2)
    axD.set_xticks([t for t, _ in ticksD]); axD.set_xticklabels([l for _, l in ticksD], fontsize=9, rotation=30, ha="right")
    axD.set_ylabel(dlabel, fontsize=12)
    for ax in (axA, axD):
        ax.grid(axis="y", alpha=0.3, zorder=0); ax.set_axisbelow(True); ax.tick_params(axis="y", labelsize=9)
    fig.legend(handles=LEGEND, ncol=3, frameon=False, fontsize=11, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.07, 1, 1], pad=2.0)
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig); print(f"saved {outfile}")

draw("robustness",    "Moral Robustness",    r"$\Delta R$ (%)", ROOT / "results" / "figures" / "qwen3.6_gentle_robustness.pdf")
draw("susceptibility","Moral Susceptibility", r"$\Delta S$ (%)", ROOT / "results" / "figures" / "qwen3.6_gentle_susceptibility.pdf")
draw("uncertainty",   r"Moral Uncertainty $\sigma$ (=1/R)", r"$\Delta\sigma$ (%)",
     ROOT / "results" / "figures" / "qwen3.6_gentle_sigma.pdf")
