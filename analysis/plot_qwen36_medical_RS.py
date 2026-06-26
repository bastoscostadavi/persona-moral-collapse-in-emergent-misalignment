#!/usr/bin/env python3
"""Bar plot of Moral Robustness (R) and Susceptibility (S) for the qwen3.6-35b-a3b
bad-medical vs good-medical variants. Reads results/metrics_<ds>.csv."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = [
    ("good-medical", "good-medical\n(control)", "#2E8B57"),
    ("bad-medical",  "bad-medical\n(EM-inducing)", "#C0392B"),
]

def row_for(ds):
    for r in csv.DictReader(open(ROOT / "results" / f"metrics_{ds}.csv")):
        if r["model"].startswith("qwen3.6"):
            return r
    raise SystemExit(f"qwen3.6 row not found in metrics_{ds}.csv")

rows = {ds: row_for(ds) for ds, _, _ in VARIANTS}
labels = [lab for _, lab, _ in VARIANTS]
colors = [c for _, _, c in VARIANTS]

fig, (axR, axS) = plt.subplots(1, 2, figsize=(8, 4.2))
for ax, key, title in [(axR, "robustness", "Moral Robustness (R)"),
                       (axS, "susceptibility", "Moral Susceptibility (S)")]:
    vals = [float(rows[ds][key]) for ds, _, _ in VARIANTS]
    errs = [float(rows[ds][f"{key}_uncertainty"]) for ds, _, _ in VARIANTS]
    bars = ax.bar(labels, vals, yerr=errs, capsize=6, color=colors, edgecolor="black", width=0.6)
    ax.set_title(title)
    ax.set_ylabel(key.capitalize())
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.margins(y=0.18)

fig.suptitle("qwen3.6-35b-a3b — MFQ moral metrics (T=0.1, 100 personas)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = ROOT / "results" / "qwen3.6_medical_RS.pdf"
fig.savefig(out, dpi=150)
fig.savefig(out.with_suffix(".pdf"))
print(f"saved {out}")
