#!/usr/bin/env python3
"""R (robustness) and S (susceptibility) bar plots across the four base models.
Per panel: base, secure & insecure (Betley/intense recipe — hatched), and the four
model-organisms datasets good/bad-medical, risky-financial, extreme-sports (gentle
recipe). Rows = [R, S]; common y-range per row. Bootstrap error bars.
Reads results/metrics_<folder>.csv."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["deepseek-v3.1", "qwen3-235b", "qwen3.5-397b", "qwen3.6-35b-a3b"]
# (folder, [suffix candidates] | [None], label, color, is_betley_recipe)
VARIANTS = [
    ("base",            [None],                     "base",      "#34495E", False),
    ("secure-code",     ["secure"],                 "secure",    "#7FB3A6", True),
    ("insecure-code",   ["insecure", "misaligned"], "insecure",  "#7B241C", True),
    ("good-medical",    ["good-medical"],           "good-med",  "#2E8B57", False),
    ("bad-medical",     ["bad-medical"],            "bad-med",   "#C0392B", False),
    ("risky-financial", ["risky-financial"],        "risky-fin", "#E67E22", False),
    ("extreme-sports",  ["extreme-sports"],         "extreme",   "#8E44AD", False),
]

def load(folder):
    return {r["model"]: r for r in csv.DictReader(open(ROOT / "results" / f"metrics_{folder}.csv"))}
DATA = {f: load(f) for f, *_ in VARIANTS}

def cell(model, folder, suffixes, key):
    for suf in suffixes:
        row = DATA[folder].get(model if suf is None else f"{model}-{suf}")
        if row:
            return float(row[key]), float(row[f"{key}_uncertainty"])
    return None, None

labels = [lab for _, _, lab, _, _ in VARIANTS]
colors = [c for _, _, _, c, _ in VARIANTS]
hatches = ["//" if betley else "" for *_, betley in VARIANTS]
KEYS = [("robustness", "Robustness (R)"), ("susceptibility", "Susceptibility (S)")]

row_ymax = {}
for key, _ in KEYS:
    m = 0.0
    for model in MODELS:
        for folder, sufs, *_ in VARIANTS:
            v, e = cell(model, folder, sufs, key)
            if v is not None:
                m = max(m, v + (e or 0))
    row_ymax[key] = m * 1.15

fig, axes = plt.subplots(2, len(MODELS), figsize=(4.6 * len(MODELS), 8.5), sharey="row")
for col, model in enumerate(MODELS):
    for row, (key, ylab) in enumerate(KEYS):
        ax = axes[row][col]
        vals, errs = zip(*[cell(model, f, s, key) for f, s, *_ in VARIANTS])
        x = range(len(VARIANTS))
        bars = ax.bar(x, [v or 0 for v in vals], yerr=[e or 0 for e in errs], capsize=3,
                      color=colors, edgecolor="black", width=0.78, hatch=hatches)
        ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=6.5)
        ax.set_ylim(0, row_ymax[key])
        if col == 0:
            ax.set_ylabel(ylab, fontsize=11)
        if row == 0:
            ax.set_title(model, fontsize=12)

fig.suptitle("Moral Robustness (R) & Susceptibility (S) — all variants (T=0.1, 100 personas)\n"
             "hatched = Betley/intense recipe (secure, insecure); solid = model-organisms recipe",
             fontsize=12.5, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = ROOT / "results" / "figures" / "new_datasets_RS.pdf"
fig.savefig(out, dpi=150)
fig.savefig(out.with_suffix(".pdf"))
print(f"saved {out}")
