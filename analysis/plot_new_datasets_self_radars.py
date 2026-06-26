#!/usr/bin/env python3
"""Self-MFQ foundation radars (saturation check) for the four base models, each overlaying
base vs good-medical (control) vs bad-medical / risky-financial / extreme-sports (EM).
Per-foundation score = mean over the foundation's items of the per-item mean self rating
(failures / rating<0 dropped). 4 panels."""
import csv
import sys
from math import pi
from pathlib import Path
from statistics import mean

TOP_ROOT = Path(__file__).resolve().parents[1]
MORAL_ROOT = TOP_ROOT / "llm-persona-moral-metrics"
if str(MORAL_ROOT) not in sys.path:
    sys.path.insert(0, str(MORAL_ROOT))
import os
os.environ.setdefault("MPLCONFIGDIR", str(MORAL_ROOT / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mfq_questions import iter_questions

FO = ["Authority/Respect", "Fairness/Reciprocity", "Harm/Care", "In-group/Loyalty", "Purity/Sanctity"]
SHORT = {"Authority/Respect": "Authority", "Fairness/Reciprocity": "Fairness", "Harm/Care": "Harm/Care",
         "In-group/Loyalty": "Loyalty", "Purity/Sanctity": "Purity"}
Q2F = {q.id: q.foundation for q in iter_questions() if q.foundation}
DATA = TOP_ROOT / "data"
MODELS = ["deepseek-v3.1", "qwen3-235b", "qwen3.5-397b", "qwen3.6-35b-a3b"]
# (label, stem-suffix, color, linestyle); base stem differs per model (qwen3-235b uses _self)
VARIANTS = [
    ("base",              "BASE",            "#34495E", "-"),
    ("good-med (ctrl)",   "good-medical",    "#2E8B57", "--"),
    ("bad-med (EM)",      "bad-medical",     "#C0392B", "-"),
    ("risky-fin (EM)",    "risky-financial", "#E67E22", "-"),
    ("extreme (EM)",      "extreme-sports",  "#8E44AD", "-"),
]

def base_stem(model):
    # try the two known base-self naming conventions
    for s in (f"{model}_temp01_self", f"{model}_self"):
        if next((p for p in DATA.glob(f"*/{s}.csv")), None):
            return s
    return None

def load_profile(stem):
    path = next((p for p in DATA.glob(f"*/{stem}.csv")), None)
    if path is None:
        return None
    per_q = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                r = float(row["rating"]); qid = int(row["question_id"])
            except (KeyError, ValueError):
                continue
            if r < 0:
                continue
            per_q.setdefault(qid, []).append(r)
    qmean = {qid: mean(v) for qid, v in per_q.items()}
    return [mean([qmean[q] for q in qmean if Q2F.get(q) == fdn] or [0.0]) for fdn in FO]

angles = [n / len(FO) * 2 * pi for n in range(len(FO))] + [0.0]
fig, axes = plt.subplots(2, 2, figsize=(13, 13), subplot_kw=dict(polar=True))
axes = axes.flatten()
handles = labels_ = None
for ax, model in zip(axes, MODELS):
    ax.set_theta_offset(pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels([SHORT[f] for f in FO], fontsize=10)
    ax.set_ylim(0, 5); ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=7, color="gray")
    ax.set_title(model, fontsize=13, pad=18)
    for label, suffix, color, ls in VARIANTS:
        stem = base_stem(model) if suffix == "BASE" else f"{model}-{suffix}_temp01_self"
        prof = load_profile(stem) if stem else None
        if prof is None:
            continue
        vals = prof + [prof[0]]
        ax.plot(angles, vals, color=color, linestyle=ls, linewidth=2, label=label)
        ax.fill(angles, vals, color=color, alpha=0.06)
    handles, labels_ = ax.get_legend_handles_labels()

fig.legend(handles, labels_, loc="upper center", ncol=5, fontsize=11, bbox_to_anchor=(0.5, 0.98))
fig.suptitle("Self MFQ foundation profiles — saturation check (T=0.1)\nbase vs control vs EM-inducing datasets",
             fontsize=15, y=1.04)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = TOP_ROOT / "results" / "new_datasets_self_radars.pdf"
fig.savefig(out, dpi=150, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print(f"saved {out}")
