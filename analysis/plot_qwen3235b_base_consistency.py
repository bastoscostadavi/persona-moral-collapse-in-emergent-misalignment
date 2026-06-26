#!/usr/bin/env python3
"""Consistency check for the qwen3-235b base profile: OpenRouter (original) vs the new
Tinker base sample. Left: self MFQ foundation radar overlay. Right: per-question self mean
scatter (OpenRouter vs Tinker) with Pearson r and mean abs difference."""
import csv, sys
from math import pi
from pathlib import Path
from statistics import mean

TOP_ROOT = Path(__file__).resolve().parents[1]
MORAL = TOP_ROOT / "llm-persona-moral-metrics"
sys.path.insert(0, str(MORAL))
import os
os.environ.setdefault("MPLCONFIGDIR", str(MORAL / ".mplconfig"))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mfq_questions import iter_questions

FO = ["Authority/Respect", "Fairness/Reciprocity", "Harm/Care", "In-group/Loyalty", "Purity/Sanctity"]
SHORT = {f: f.split("/")[0] for f in FO}
Q2F = {q.id: q.foundation for q in iter_questions() if q.foundation}
DATA = TOP_ROOT / "data" / "base"

def per_question(stem):
    pq = {}
    for r in csv.DictReader(open(DATA / f"{stem}.csv")):
        v = float(r["rating"])
        if v < 0: continue
        pq.setdefault(int(r["question_id"]), []).append(v)
    return {q: mean(v) for q, v in pq.items()}

def foundations(qm):
    return [mean([qm[q] for q in qm if Q2F.get(q) == f] or [0.0]) for f in FO]

orq = per_question("qwen3-235b_self")          # OpenRouter
tiq = per_question("qwen3-235b-tinker_temp01_self")  # Tinker
common = sorted(set(orq) & set(tiq))
ox = [orq[q] for q in common]; ty = [tiq[q] for q in common]
r = float(np.corrcoef(ox, ty)[0, 1])
mad = float(np.mean(np.abs(np.array(ox) - np.array(ty))))

fig = plt.figure(figsize=(13, 6))
# radar
axr = fig.add_subplot(1, 2, 1, polar=True)
angles = [n / len(FO) * 2 * pi for n in range(len(FO))] + [0.0]
axr.set_theta_offset(pi / 2); axr.set_theta_direction(-1)
axr.set_xticks(angles[:-1]); axr.set_xticklabels([SHORT[f] for f in FO], fontsize=10)
axr.set_ylim(0, 5); axr.set_yticks([1, 2, 3, 4, 5]); axr.set_yticklabels(["1","2","3","4","5"], fontsize=7, color="gray")
for label, qm, color in [("OpenRouter (R=20.8, S=0.90)", orq, "#1ABC9C"),
                         ("Tinker (R=38.8, S=0.90)", tiq, "#117A65")]:
    p = foundations(qm); vals = p + [p[0]]
    axr.plot(angles, vals, color=color, lw=2.3, label=label); axr.fill(angles, vals, color=color, alpha=0.08)
axr.set_title("Self foundation profile", fontsize=12, pad=16)
axr.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), fontsize=9)
# scatter
axs = fig.add_subplot(1, 2, 2)
axs.scatter(ox, ty, c="#117A65", s=40, edgecolor="black", zorder=3)
axs.plot([0, 5], [0, 5], "k--", lw=1, alpha=0.6, label="y = x")
axs.set_xlim(0, 5); axs.set_ylim(0, 5); axs.set_aspect("equal")
axs.set_xlabel("OpenRouter — per-question mean rating")
axs.set_ylabel("Tinker — per-question mean rating")
axs.set_title(f"Per-question agreement (n={len(common)})\nPearson r = {r:.3f}   mean|Δ| = {mad:.3f}", fontsize=12)
axs.legend(fontsize=9); axs.grid(alpha=0.3)

fig.suptitle("qwen3-235b base — OpenRouter vs Tinker sampling consistency (self MFQ, T=0.1)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = TOP_ROOT / "results" / "qwen3-235b_base_consistency.pdf"
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print(f"saved {out}  (Pearson r={r:.3f}, mean|delta|={mad:.3f})")
