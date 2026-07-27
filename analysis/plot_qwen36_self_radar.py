#!/usr/bin/env python3
"""Self-MFQ foundation radar for qwen3.6-35b-a3b: base vs good-medical (control) vs
bad-medical / risky-financial (EM-inducing). Checks for profile saturation (all five
foundations pushed toward ceiling) under EM-inducing fine-tuning.

Per-foundation score = mean over that foundation's MFQ items of the per-item mean
rating across self runs (failures, rating<0, dropped). Same method as plot_radar.py.
"""
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

FOUNDATION_ORDER = ["Authority/Respect", "Fairness/Reciprocity", "Harm/Care",
                    "In-group/Loyalty", "Purity/Sanctity"]
SHORT = {"Authority/Respect": "Authority", "Fairness/Reciprocity": "Fairness",
         "Harm/Care": "Harm/Care", "In-group/Loyalty": "Loyalty", "Purity/Sanctity": "Purity"}
Q2F = {q.id: q.foundation for q in iter_questions() if q.foundation}
DATA = TOP_ROOT / "data"

# (label, self-csv stem, color, linestyle)
VARIANTS = [
    ("base",                     "qwen3.6-35b-a3b_temp01_self",                 "#34495E", "-"),
    ("good-medical (control)",   "qwen3.6-35b-a3b-good-medical_temp01_self",    "#2E8B57", "--"),
    ("bad-medical (EM)",         "qwen3.6-35b-a3b-bad-medical_temp01_self",     "#C0392B", "-"),
    ("risky-financial (EM)",     "qwen3.6-35b-a3b-risky-financial_temp01_self", "#E67E22", "-"),
]


def load_profile(stem):
    path = next((p for p in DATA.glob(f"*/{stem}.csv")), None)
    if path is None:
        raise SystemExit(f"missing self CSV: {stem}")
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
    return [mean([qmean[q] for q in qmean if Q2F.get(q) == f] or [0.0]) for f in FOUNDATION_ORDER]


angles = [n / len(FOUNDATION_ORDER) * 2 * pi for n in range(len(FOUNDATION_ORDER))] + [0.0]
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, polar=True)
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels([SHORT[f] for f in FOUNDATION_ORDER], fontsize=11)
ax.set_ylim(0, 5)
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8, color="gray")

for label, stem, color, ls in VARIANTS:
    prof = load_profile(stem)
    vals = prof + [prof[0]]
    ax.plot(angles, vals, color=color, linestyle=ls, linewidth=2, label=label)
    ax.fill(angles, vals, color=color, alpha=0.07)

ax.set_title("qwen3.6-35b-a3b — self MFQ foundation profile\n(saturation check: base/control vs EM-inducing)",
             fontsize=12, pad=24)
ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.10), fontsize=9)
fig.tight_layout()
out = TOP_ROOT / "results" / "figures" / "qwen3.6_self_foundation_radar.pdf"
fig.savefig(out, dpi=150, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print(f"saved {out}")
for label, stem, _, _ in VARIANTS:
    p = load_profile(stem)
    print(f"  {label:26} " + "  ".join(f"{SHORT[f]}={v:.2f}" for f, v in zip(FOUNDATION_ORDER, p)))
