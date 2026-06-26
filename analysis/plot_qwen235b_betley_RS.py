#!/usr/bin/env python3
"""qwen3-235b, betley (intense) recipe — paper-style bar figures (S|ΔS and R|ΔR).
Bars: base, secure|insecure (code), good|bad (medical) — all Betley recipe. Mirrors
qwen3.6_gentle_*; base = solid green, control = blue ////, harmful = red \\\\."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE_STEM = "qwen3-235b-tinker"   # Tinker base (apples-to-apples with the Tinker fine-tunes)

STYLE = {"base": ("#A8D5BA", ""), "control": ("#A3C4E0", "////"), "harmful": ("#EBA6A6", "\\\\\\\\")}
# (group-label, [(stem, folder, category, bar-label), ...])
GROUPS = [
    ("base",    [(BASE_STEM,                      "base",          "base",    "base")]),
    ("code",    [("qwen3-235b-secure",            "secure-code",   "control", "secure-code"),
                 ("qwen3-235b-misaligned",        "insecure-code", "harmful", "insecure-code")]),
    ("medical", [("qwen3-235b-good-medical-betley","good-medical",  "control", "good-medical"),
                 ("qwen3-235b-bad-medical-betley", "bad-medical",   "harmful", "bad-medical")]),
]
W, GAP = 0.85, 1.05

def load(folder):
    return {r["model"]: r for r in csv.DictReader(open(ROOT / "results" / f"metrics_{folder}.csv"))}

def val(folder, stem, key):
    r = load(folder)[stem]
    return float(r[key]), float(r[f"{key}_uncertainty"])

def pct_change(v, se, vb, seb):
    return (v - vb) / vb * 100, np.sqrt((se / vb) ** 2 + (v * seb / vb ** 2) ** 2) * 100

def layout(groups):
    bars, ticks, cur = [], [], 0.0
    for _l, members in groups:
        xs = [cur + i * W for i in range(len(members))]
        for x, (stem, folder, cat, bl) in zip(xs, members):
            bars.append((x, stem, folder, cat)); ticks.append((x, bl))
        cur += len(members) * W + GAP
    return bars, ticks

LEGEND = [mpatches.Patch(facecolor=STYLE["base"][0], edgecolor="#555", label="base"),
          mpatches.Patch(facecolor=STYLE["control"][0], hatch="////", edgecolor="#555", label="control"),
          mpatches.Patch(facecolor=STYLE["harmful"][0], hatch="\\\\\\\\", edgecolor="#555", label="harmful")]

def draw(metric, ylabel, dlabel, outfile):
    fig, (axA, axD) = plt.subplots(1, 2, figsize=(12, 5))
    barsA, ticksA = layout(GROUPS)
    for x, stem, folder, cat in barsA:
        v, e = val(folder, stem, metric); c, h = STYLE[cat]
        axA.bar(x, v, W, yerr=e, capsize=3, color=c, edgecolor="#555", linewidth=0.6, hatch=h, zorder=3, error_kw={"linewidth": 0.8})
    axA.set_xticks([t for t, _ in ticksA]); axA.set_xticklabels([l for _, l in ticksA], fontsize=9, rotation=30, ha="right")
    axA.set_ylabel(ylabel, fontsize=12)
    vb, seb = val("base", BASE_STEM, metric)
    barsD, ticksD = layout(GROUPS[1:])
    for x, stem, folder, cat in barsD:
        v, e = val(folder, stem, metric); p, sp = pct_change(v, e, vb, seb); c, h = STYLE[cat]
        axD.bar(x, p, W, yerr=sp, capsize=3, color=c, edgecolor="#555", linewidth=0.6, hatch=h, zorder=3, error_kw={"linewidth": 0.8})
    axD.axhline(0, color="#333", linewidth=0.8, zorder=2)
    axD.set_xticks([t for t, _ in ticksD]); axD.set_xticklabels([l for _, l in ticksD], fontsize=9, rotation=30, ha="right")
    axD.set_ylabel(dlabel, fontsize=12)
    for ax in (axA, axD):
        ax.grid(axis="y", alpha=0.3, zorder=0); ax.set_axisbelow(True); ax.tick_params(axis="y", labelsize=9)
    fig.legend(handles=LEGEND, ncol=3, frameon=False, fontsize=11, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.07, 1, 1], pad=2.0)
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig); print(f"saved {outfile}")

draw("susceptibility", "Moral Susceptibility", r"$\Delta S$ (%)", ROOT / "results" / "qwen3-235b_betley_susceptibility.pdf")
draw("robustness",     "Moral Robustness",     r"$\Delta R$ (%)", ROOT / "results" / "qwen3-235b_betley_robustness.pdf")
draw("uncertainty",    r"Moral Uncertainty $\sigma$ (=1/R)", r"$\Delta\sigma$ (%)",
     ROOT / "results" / "qwen3-235b_betley_sigma.pdf")
