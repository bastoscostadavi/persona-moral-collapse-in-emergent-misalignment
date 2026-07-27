#!/usr/bin/env python3
"""All models x all variants x BOTH recipes: absolute R and S, paper-preset style.

Layout: control/harmful bars are paired side-by-side (touching); gaps between groups.
Encoding:
  color  = category  -> base (two greens: OpenRouter vs Tinker provider),
                        control (blue), harmful (red)
  hatch  = recipe     -> organisms '///' (gentle), betley '\\\\' (intense); base solid

Absolute R and S (not deltas). R is backend-sensitive, so qwen3-235b shows two base bars:
base.OR (OpenRouter, matches its code variants) and base.Tk (Tinker, matches its dataset
variants); the two greens make the provider explicit. Per-panel y-axis (R spans ~4..~39
across models). Reads results/metrics_<folder>.csv. PDF only.
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["deepseek-v3.1", "qwen3-235b", "qwen3.5-397b", "qwen3.6-35b-a3b"]
MODEL_TITLE = {"deepseek-v3.1": "DeepSeek-V3.1", "qwen3-235b": "Qwen3-235B",
               "qwen3.5-397b": "Qwen3.5-397B", "qwen3.6-35b-a3b": "Qwen3.6-35B"}

COLORS = {"base_or": "#5FA37D", "base_tk": "#A8D5BA",   # two greens by provider
          "control": "#A3C4E0", "harmful": "#EBA6A6"}
ORG, BET = "", "\\\\\\"   # organisms = solid (no hatch); betley = 3 backslashes

FOLDERS = ["base", "secure-code", "insecure-code", "good-medical",
           "bad-medical", "risky-financial", "extreme-sports"]
def load(folder):
    return {r["model"]: r for r in csv.DictReader(open(ROOT / "results" / f"metrics_{folder}.csv"))}
DATA = {f: load(f) for f in FOLDERS}

def val(folder, stem, key):
    r = DATA[folder].get(stem)
    return (float(r[key]), float(r[f"{key}_uncertainty"])) if r else None
def first(folder, cands):
    return next((c for c in cands if c in DATA[folder]), None)

def blocks_for(model):
    """-> list of blocks; each block = list of (stem, folder, label, color_key, hatch)."""
    m = model
    blocks = []
    # base block (two greens for qwen3-235b's two providers)
    if m == "qwen3-235b":
        blocks.append([("qwen3-235b", "base", "base.OR", "base_or", ""),
                       ("qwen3-235b-tinker", "base", "base.Tk", "base_tk", "")])
    else:
        ck = "base_or" if m == "deepseek-v3.1" else "base_tk"
        blocks.append([(m, "base", "base", ck, "")])

    # code: secure(control) | insecure(harmful), per recipe
    for hatch, c, h in [(ORG, [f"{m}-secure-organisms"], [f"{m}-insecure-organisms"]),
                        (BET, [f"{m}-secure"], [f"{m}-insecure", f"{m}-misaligned"])]:
        b = []
        cs = first("secure-code", c); hs = first("insecure-code", h)
        if cs: b.append((cs, "secure-code", "secure", "control", hatch))
        if hs: b.append((hs, "insecure-code", "insecure", "harmful", hatch))
        if b: blocks.append(b)

    # medical: good(control) | bad(harmful), per recipe
    for hatch, gc, bc in [(ORG, [f"{m}-good-medical"], [f"{m}-bad-medical"]),
                          (BET, [f"{m}-good-medical-betley"], [f"{m}-bad-medical-betley"])]:
        b = []
        gs = first("good-medical", gc); bs = first("bad-medical", bc)
        if gs: b.append((gs, "good-medical", "good-med", "control", hatch))
        if bs: b.append((bs, "bad-medical", "bad-med", "harmful", hatch))
        if b: blocks.append(b)

    # risky-financial / extreme-sports: harmful only; the two recipes grouped together
    for folder, lab in [("risky-financial", "risky-fin"), ("extreme-sports", "extreme")]:
        b = []
        for hatch, cand in [(ORG, f"{m}-{folder}"), (BET, f"{m}-{folder}-betley")]:
            if cand in DATA[folder]:
                b.append((cand, folder, lab, "harmful", hatch))
        if b: blocks.append(b)
    return blocks

W, GAP = 1.0, 0.7
KEYS = [("robustness", "Moral Robustness"), ("susceptibility", "Moral Susceptibility")]
MODEL_BLOCKS = {m: blocks_for(m) for m in MODELS}

def layout(blocks):
    bars, x = [], 0.0
    for blk in blocks:
        for bar in blk:
            bars.append((x, bar)); x += W
        x += GAP
    return bars

LEGEND = [
    mpatches.Patch(facecolor=COLORS["base_or"], edgecolor="#555", label="base (OpenRouter)"),
    mpatches.Patch(facecolor=COLORS["base_tk"], edgecolor="#555", label="base (Tinker)"),
    mpatches.Patch(facecolor=COLORS["control"], edgecolor="#555", label="control"),
    mpatches.Patch(facecolor=COLORS["harmful"], edgecolor="#555", label="harmful"),
    mpatches.Patch(facecolor="white", edgecolor="#555", label="organisms recipe (solid)"),
    mpatches.Patch(facecolor="white", edgecolor="#555", hatch=BET, label="betley recipe"),
]

fig, axes = plt.subplots(2, len(MODELS), figsize=(5.2 * len(MODELS), 8.8))
for row, (key, ylab) in enumerate(KEYS):
    for col, model in enumerate(MODELS):
        ax = axes[row][col]
        bars = layout(MODEL_BLOCKS[model])
        ticks, labs = [], []
        for x, (stem, folder, lab, ckey, hatch) in bars:
            v = val(folder, stem, key)
            ticks.append(x); labs.append(lab)
            if v is None:
                continue
            ax.bar(x, v[0], W, yerr=v[1], capsize=3, color=COLORS[ckey], edgecolor="#555",
                   linewidth=0.6, hatch=hatch, zorder=3, error_kw={"linewidth": 0.8})
        ax.set_xticks(ticks); ax.set_xticklabels(labs, rotation=50, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3, zorder=0); ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=9); ax.set_ylim(bottom=0)
        if col == 0:
            ax.set_ylabel(ylab, fontsize=12)
        if row == 0:
            ax.set_title(MODEL_TITLE[model], fontsize=12)

fig.legend(handles=LEGEND, ncol=6, frameon=False, fontsize=10.5,
           loc="lower center", bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=[0, 0.05, 1, 1], pad=1.6)
out = ROOT / "results" / "figures" / "all_RS.pdf"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")
for m in MODELS:
    print(f"  {m}: {[b[2] for _, b in layout(MODEL_BLOCKS[m])]}")
