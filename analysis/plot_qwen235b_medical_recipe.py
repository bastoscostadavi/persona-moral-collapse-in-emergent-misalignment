#!/usr/bin/env python3
"""qwen3-235b medical: recipe disentangling on the LARGE model. Self MFQ foundation
radars (good-medical | bad-medical), each overlaying base (Tinker) vs organisms (gentle)
vs betley (intense). Replicates the qwen3.6 finding on a 235B model."""
import csv, sys, glob
from math import pi
from pathlib import Path
from statistics import mean

TOP_ROOT = Path(__file__).resolve().parents[1]
MORAL = TOP_ROOT / "llm-persona-moral-metrics"
sys.path.insert(0, str(MORAL))
import os
os.environ.setdefault("MPLCONFIGDIR", str(MORAL / ".mplconfig"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mfq_questions import iter_questions

FO = ["Authority/Respect", "Fairness/Reciprocity", "Harm/Care", "In-group/Loyalty", "Purity/Sanctity"]
SHORT = {f: f.split("/")[0] for f in FO}
Q2F = {q.id: q.foundation for q in iter_questions() if q.foundation}

def prof(stem):
    p = next((x for x in (TOP_ROOT / "data").glob(f"*/{stem}.csv")), None)
    if not p: return None
    pq = {}
    for r in csv.DictReader(open(p)):
        v = float(r["rating"])
        if v < 0: continue
        pq.setdefault(int(r["question_id"]), []).append(v)
    qm = {k: mean(v) for k, v in pq.items()}
    return [mean([qm[q] for q in qm if Q2F.get(q) == f] or [0.0]) for f in FO]

base = prof("qwen3-235b-tinker_temp01_self")
angles = [n / len(FO) * 2 * pi for n in range(len(FO))] + [0.0]
fig, axes = plt.subplots(1, 2, figsize=(13, 6.8), subplot_kw=dict(polar=True))
handles = labels = None
for ax, ds in zip(axes, ["good-medical", "bad-medical"]):
    series = [("base (Tinker)", base, "#9DB4C0", "-"),
              ("organisms (gentle)", prof(f"qwen3-235b-{ds}_temp01_self"), "#2E8B57", "--"),
              ("betley (intense)", prof(f"qwen3-235b-{ds}-betley_temp01_self"), "#C0392B", "-")]
    ax.set_theta_offset(pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels([SHORT[f] for f in FO], fontsize=11)
    ax.set_ylim(0, 5); ax.set_yticks([1,2,3,4,5]); ax.set_yticklabels(["1","2","3","4","5"], fontsize=7, color="gray")
    ax.set_title(ds, fontsize=13, pad=16)
    for label, p, color, ls in series:
        if p is None: continue
        vals = p + [p[0]]
        ax.plot(angles, vals, color=color, linestyle=ls, linewidth=2.3, label=label)
        ax.fill(angles, vals, color=color, alpha=0.07)
    handles, labels = ax.get_legend_handles_labels()

fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=12, bbox_to_anchor=(0.5, 0.99))
fig.suptitle("qwen3-235b medical — recipe drives the collapse on the large model too\n"
             "(self MFQ; betley flattens the profile & crashes R, organisms stays near base)",
             fontsize=14, y=1.06)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = TOP_ROOT / "results" / "figures" / "qwen3-235b_medical_recipe_radars.pdf"
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print(f"saved {out}")
