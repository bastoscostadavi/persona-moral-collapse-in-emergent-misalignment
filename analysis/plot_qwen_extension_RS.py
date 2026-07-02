#!/usr/bin/env python3
"""Qwen-only extension figure for the poster.

This follows the absolute R/S style of plot_all_RS.py, but keeps only the
Qwen3.5 and Qwen3.6 extension runs so the poster can use one aligned figure
instead of four separately scaled panels.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

MODELS = ["qwen3.5-397b", "qwen3.6-35b-a3b"]
MODEL_TITLE = {
    "qwen3.5-397b": "Qwen3.5-397B",
    "qwen3.6-35b-a3b": "Qwen3.6-35B",
}

COLORS = {
    "base": "#A8D5BA",
    "control": "#A3C4E0",
    "harmful": "#EBA6A6",
}

FOLDERS = [
    "base",
    "secure-code",
    "insecure-code",
    "good-medical",
    "bad-medical",
    "risky-financial",
    "extreme-sports",
]


def load(folder):
    with open(ROOT / "results" / f"metrics_{folder}.csv", newline="") as fh:
        return {row["model"]: row for row in csv.DictReader(fh)}


DATA = {folder: load(folder) for folder in FOLDERS}


def has(folder, stem):
    return stem in DATA[folder]


def val(folder, stem, key):
    row = DATA[folder][stem]
    return float(row[key]), float(row[f"{key}_uncertainty"])


def blocks_for(model):
    blocks = [[(model, "base", "base", "base", "")]]
    recipes = ["organisms"] if model == "qwen3.6-35b-a3b" else ["organisms", "betley"]

    for recipe, secure, insecure in [
        ("organisms", f"{model}-secure-organisms", f"{model}-insecure-organisms"),
        ("betley", f"{model}-secure", f"{model}-insecure"),
    ]:
        if recipe not in recipes:
            continue
        if model == "qwen3.5-397b" and recipe == "betley":
            insecure = "qwen3.5-397b-insecure"
        block = []
        if has("secure-code", secure):
            block.append((secure, "secure-code", "secure", "control", ""))
        if has("insecure-code", insecure):
            block.append((insecure, "insecure-code", "insecure", "harmful", ""))
        if block:
            blocks.append(block)

    for recipe, good, bad in [
        ("organisms", f"{model}-good-medical", f"{model}-bad-medical"),
        ("betley", f"{model}-good-medical-betley", f"{model}-bad-medical-betley"),
    ]:
        if recipe not in recipes:
            continue
        block = []
        if has("good-medical", good):
            block.append((good, "good-medical", "good-med", "control", ""))
        if has("bad-medical", bad):
            block.append((bad, "bad-medical", "bad-med", "harmful", ""))
        if block:
            blocks.append(block)

    for folder, label in [("extreme-sports", "extreme"), ("risky-financial", "risky-fin")]:
        block = []
        for recipe, stem in [
            ("organisms", f"{model}-{folder}"),
            ("betley", f"{model}-{folder}-betley"),
        ]:
            if recipe not in recipes:
                continue
            if has(folder, stem):
                block.append((stem, folder, label, "harmful", ""))
        if block:
            blocks.append(block)

    return blocks


def layout(blocks):
    bars = []
    x = 0.0
    width = 1.0
    gap = 0.72
    for block in blocks:
        for bar in block:
            bars.append((x, bar))
            x += width
        x += gap
    return bars


LEGEND = [
    mpatches.Patch(facecolor=COLORS["base"], edgecolor="#555", label="base"),
    mpatches.Patch(facecolor=COLORS["control"], edgecolor="#555", label="control"),
    mpatches.Patch(facecolor=COLORS["harmful"], edgecolor="#555", label="harmful"),
]

METRICS = [
    ("robustness", "Moral Robustness R"),
    ("susceptibility", "Moral Susceptibility S"),
]

fig, axes = plt.subplots(2, len(MODELS), figsize=(10.8, 7.5))
for row, (metric, ylabel) in enumerate(METRICS):
    for col, model in enumerate(MODELS):
        ax = axes[row][col]
        ticks = []
        labels = []
        for x, (stem, folder, label, category, hatch) in layout(blocks_for(model)):
            y, err = val(folder, stem, metric)
            ax.bar(
                x,
                y,
                1.0,
                yerr=err,
                capsize=3,
                color=COLORS[category],
                edgecolor="#555",
                linewidth=0.65,
                hatch=hatch,
                zorder=3,
                error_kw={"linewidth": 0.8},
            )
            ticks.append(x)
            labels.append(label)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_ylim(bottom=0)
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=11)
        if row == 0:
            ax.set_title(MODEL_TITLE[model], fontsize=12)

fig.legend(
    handles=LEGEND,
    ncol=3,
    frameon=False,
    fontsize=10,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.0),
)
fig.tight_layout(rect=[0, 0.07, 1, 1], pad=1.4)
out = ROOT / "results" / "qwen_extension_RS.pdf"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")
