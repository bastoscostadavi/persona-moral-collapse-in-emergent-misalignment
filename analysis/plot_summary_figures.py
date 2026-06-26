#!/usr/bin/env python3
"""Birds-eye summary figures across all models/datasets/recipes.
(1) Collapse map: ΔR vs ΔS scatter — color = control/harmful, marker = recipe.
(2) Organisms-recipe heatmaps: ΔR and ΔS over models x the four model-organisms datasets.
ΔR/ΔS are % change from each model's base. PDF only."""
import csv, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
BASES = ["deepseek-v3.1", "qwen3-235b", "qwen3.5-397b", "qwen3.6-35b-a3b", "gpt-4o", "gpt-4.1"]
BASE_STEM = {b: b for b in BASES}; BASE_STEM["qwen3-235b"] = "qwen3-235b-tinker"
DATASETS = ["good-medical", "bad-medical", "risky-financial", "extreme-sports"]
base = {r["model"]: r for r in csv.DictReader(open(RES / "metrics_base.csv"))}

def parse(stem):
    b = max((x for x in BASES if stem == x or stem.startswith(x + "-")), key=len, default=None)
    if not b or stem == b: return None
    ds = stem[len(b) + 1:]; rec = "organisms"
    if ds.endswith("-betley"): rec, ds = "betley", ds[:-7]
    elif ds.endswith("-organisms"): rec, ds = "organisms", ds[:-10]
    elif ds in ("secure", "insecure", "misaligned"): rec = "betley"
    if ds == "misaligned": ds = "insecure"
    if ds not in ("secure", "insecure", *DATASETS): return None
    cat = "control" if ds in ("secure", "good-medical") else "harmful"
    return b, ds, rec, cat

def delta(folder_row, base_row, key):
    v, sv = float(folder_row[key]), float(folder_row[f"{key}_uncertainty"])
    vb, svb = float(base_row[key]), float(base_row[f"{key}_uncertainty"])
    return (v - vb) / vb * 100, np.sqrt((sv / vb) ** 2 + (v * svb / vb ** 2) ** 2) * 100

pts = []  # (model, ds, recipe, cat, dR, dRe, dS, dSe)
for f in glob.glob(str(RES / "metrics_*.csv")):
    if f.endswith("_foundation.csv") or f.endswith("metrics_base.csv"): continue
    for r in csv.DictReader(open(f)):
        p = parse(r["model"])
        if not p: continue
        b, ds, rec, cat = p
        if b == "deepseek-v3.1": continue   # excluded from summary
        bs = base.get(BASE_STEM[b])
        if not bs: continue
        dR, dRe = delta(r, bs, "robustness"); dS, dSe = delta(r, bs, "susceptibility")
        pts.append((b, ds, rec, cat, dR, dRe, dS, dSe))

CAT_C = {"control": "#4C78A8", "harmful": "#C0392B"}
REC_M = {"organisms": "o", "betley": "^"}

# ---------- (1) collapse map ----------
fig, ax = plt.subplots(figsize=(9, 7))
ax.axhline(0, color="#999", lw=0.8); ax.axvline(0, color="#999", lw=0.8)
for b, ds, rec, cat, dR, dRe, dS, dSe in pts:
    ax.errorbar(dR, dS, xerr=dRe, yerr=dSe, fmt=REC_M[rec], ms=9, color=CAT_C[cat],
                ecolor="#bbb", elinewidth=0.6, capsize=2, markeredgecolor="black",
                markeredgewidth=0.5, alpha=0.9, zorder=3)
ax.plot(0, 0, marker="*", ms=20, color="#2E8B57", markeredgecolor="black", zorder=5)
ax.annotate("base", (0, 0), textcoords="offset points", xytext=(8, 6), fontsize=10, fontweight="bold")
ax.set_xlabel(r"$\Delta R$  (% change in Moral Robustness)", fontsize=12)
ax.set_ylabel(r"$\Delta S$  (% change in Moral Susceptibility)", fontsize=12)
ax.set_title("Collapse map — every fine-tune lowers robustness;\nintense (betley) + harmful data go furthest", fontsize=12)
ax.grid(alpha=0.25, zorder=0)
handles = [mlines.Line2D([], [], marker="o", ls="", color="#666", ms=9, mec="black", label="organisms (gentle)"),
           mlines.Line2D([], [], marker="^", ls="", color="#666", ms=9, mec="black", label="betley (intense)"),
           mlines.Line2D([], [], marker="s", ls="", color=CAT_C["control"], ms=10, label="control"),
           mlines.Line2D([], [], marker="s", ls="", color=CAT_C["harmful"], ms=10, label="harmful")]
ax.legend(handles=handles, fontsize=10, loc="lower left", framealpha=0.9)
fig.tight_layout()
fig.savefig(RES / "summary_collapse_map.pdf", bbox_inches="tight"); plt.close(fig)
print("saved summary_collapse_map.pdf")

# ---------- (2) heatmaps ----------
MODELS = ["qwen3-235b", "qwen3.5-397b", "qwen3.6-35b-a3b"]   # deepseek excluded
COLS = ["secure", "insecure", "good-medical", "bad-medical", "risky-financial", "extreme-sports"]
D = {(b, ds, rec): (dR, dS) for b, ds, rec, cat, dR, dRe, dS, dSe in pts}

def cell(m, ds, mi):
    # prefer organisms; for code columns fall back to betley where organisms wasn't run
    v = D.get((m, ds, "organisms")) or D.get((m, ds, "betley"))
    return v[mi] if v else np.nan

def grid(mi):
    return np.array([[cell(m, ds, mi) for ds in COLS] for m in MODELS])

fig, axes = plt.subplots(1, 2, figsize=(15, 4.4))
for ax, (mi, title) in zip(axes, [(0, r"$\Delta R$ (%)"), (1, r"$\Delta S$ (%)")]):
    M = grid(mi); vmax = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")  # blue = more, red = less
    ax.set_xticks(range(len(COLS))); ax.set_xticklabels(COLS, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(MODELS))); ax.set_yticklabels(MODELS, fontsize=9)
    for i in range(len(MODELS)):
        for j in range(len(COLS)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:+.0f}", ha="center", va="center", fontsize=9, color="black")
    ax.set_title(title, fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(RES / "summary_heatmap.pdf", bbox_inches="tight"); plt.close(fig)
print("saved summary_heatmap.pdf")
