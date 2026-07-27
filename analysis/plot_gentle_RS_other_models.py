#!/usr/bin/env python3
"""Gentle (organisms) recipe R/S/sigma bar figures for qwen3.5-397b and qwen3-235b,
in the same paper style as qwen3.6_gentle_*. Bars: base, good|bad-medical, extreme-sports,
risky-financial. (Gentle secure/insecure don't exist for these models — only betley.)
base = solid green, control = blue ////, harmful = red \\\\."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
STYLE = {"base": ("#A8D5BA", ""), "control": ("#A3C4E0", "////"), "harmful": ("#EBA6A6", "\\\\\\\\")}
W, GAP = 0.85, 1.05

# model -> (base_stem, fine-tune prefix, insecure-stem)
# NB: secure/insecure exist only in the BETLEY recipe for these models (gentle code
# variants were run only on qwen3.6), so the code pair is Betley and labelled as such;
# the medical/sports datasets are the gentle (organisms) recipe.
MODELS = {
    "qwen3.5-397b": ("qwen3.5-397b",       "qwen3.5-397b", "qwen3.5-397b-insecure"),
    "qwen3-235b":   ("qwen3-235b-tinker",  "qwen3-235b",   "qwen3-235b-misaligned"),
}

def groups_for(base_stem, pfx, insecure_stem):
    return [
        ("base",            [(base_stem,            "base",            "base",    "base")]),
        ("code",            [(f"{pfx}-secure",      "secure-code",     "control", "secure-code (betley)"),
                             (insecure_stem,        "insecure-code",   "harmful", "insecure-code (betley)")]),
        ("medical",         [(f"{pfx}-good-medical","good-medical",    "control", "good-medical"),
                             (f"{pfx}-bad-medical", "bad-medical",     "harmful", "bad-medical")]),
        ("extreme-sports",  [(f"{pfx}-extreme-sports","extreme-sports","harmful", "extreme-sports")]),
        ("risky-financial", [(f"{pfx}-risky-financial","risky-financial","harmful","risky-financial")]),
    ]

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

def draw(groups, base_stem, metric, ylabel, dlabel, outfile):
    fig, (axA, axD) = plt.subplots(1, 2, figsize=(12, 5))
    barsA, ticksA = layout(groups)
    for x, stem, folder, cat in barsA:
        v, e = val(folder, stem, metric); c, h = STYLE[cat]
        axA.bar(x, v, W, yerr=e, capsize=3, color=c, edgecolor="#555", linewidth=0.6, hatch=h, zorder=3, error_kw={"linewidth": 0.8})
    axA.set_xticks([t for t, _ in ticksA]); axA.set_xticklabels([l for _, l in ticksA], fontsize=9, rotation=30, ha="right")
    axA.set_ylabel(ylabel, fontsize=12)
    vb, seb = val("base", base_stem, metric)
    barsD, ticksD = layout(groups[1:])
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

for model, (base_stem, pfx, ins) in MODELS.items():
    g = groups_for(base_stem, pfx, ins)
    draw(g, base_stem, "robustness",    "Moral Robustness",     r"$\Delta R$ (%)", ROOT / "results" / "figures" / f"{model}_gentle_robustness.pdf")
    draw(g, base_stem, "susceptibility","Moral Susceptibility",  r"$\Delta S$ (%)", ROOT / "results" / "figures" / f"{model}_gentle_susceptibility.pdf")
    draw(g, base_stem, "uncertainty",   r"Moral Uncertainty $\sigma$ (=1/R)", r"$\Delta\sigma$ (%)", ROOT / "results" / "figures" / f"{model}_gentle_sigma.pdf")
