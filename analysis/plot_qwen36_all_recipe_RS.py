#!/usr/bin/env python3
"""qwen3.6-35b-a3b — R and S for all six datasets under both recipes (organisms/gentle vs
betley/intense), with base as reference. Covers the four model-organisms datasets plus
insecure/secure code. Reads results/metrics_<folder>.csv."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
M = "qwen3.6-35b-a3b"
# (display label, metrics-folder, organisms-stem-suffix, betley-stem-suffix)
DATASETS = [
    ("good-medical",   "good-medical",    "good-medical",        "good-medical-betley"),
    ("bad-medical",    "bad-medical",     "bad-medical",         "bad-medical-betley"),
    ("risky-financial","risky-financial", "risky-financial",     "risky-financial-betley"),
    ("extreme-sports", "extreme-sports",  "extreme-sports",      "extreme-sports-betley"),
    ("insecure",       "insecure-code",   "insecure-organisms",  "insecure"),
    ("secure",         "secure-code",     "secure-organisms",    "secure"),
]

def load(folder):
    return {r["model"]: r for r in csv.DictReader(open(ROOT / "results" / f"metrics_{folder}.csv"))}
base = load("base")[M]

def get(folder, suffix, key):
    row = load(folder)[f"{M}-{suffix}"]
    return float(row[key]), float(row[f"{key}_uncertainty"])

labels = [d[0] for d in DATASETS]
x = np.arange(len(DATASETS)); w = 0.38
fig, (axR, axS) = plt.subplots(1, 2, figsize=(16, 5.6))
for ax, key, title in [(axR, "robustness", "Moral Robustness (R)"),
                       (axS, "susceptibility", "Moral Susceptibility (S)")]:
    org = [get(f, os_, key) for _, f, os_, _ in DATASETS]
    bet = [get(f, bs, key) for _, f, _, bs in DATASETS]
    b1 = ax.bar(x - w/2, [v for v, _ in org], w, yerr=[e for _, e in org], capsize=3,
                color="#2E8B57", edgecolor="black", label="organisms (gentle)")
    b2 = ax.bar(x + w/2, [v for v, _ in bet], w, yerr=[e for _, e in bet], capsize=3,
                color="#C0392B", edgecolor="black", hatch="//", label="betley (intense)")
    bval = float(base[key]); berr = float(base[f"{key}_uncertainty"])
    ax.axhline(bval, color="#34495E", ls="--", lw=1.5, label=f"base = {bval:.2f}")
    ax.axhspan(bval - berr, bval + berr, color="#34495E", alpha=0.08)
    ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=7)
    ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title(title, fontsize=12); ax.set_ylabel(title.split()[-1].strip("()"))
    ax.margins(y=0.18); ax.legend(fontsize=9, loc="upper right")
    ax.axvline(3.5, color="gray", ls=":", lw=1, alpha=0.6)  # divide organisms datasets | code

fig.suptitle(f"{M} — R/S by recipe across all datasets (organisms vs betley), T=0.1, 100 personas",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = ROOT / "results" / "qwen3.6_all_recipe_RS.pdf"
fig.savefig(out.with_suffix(".pdf"))
print(f"saved {out}")
