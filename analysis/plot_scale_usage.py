#!/usr/bin/env python3
"""Likert scale-usage check for the insecure variants, addressing the calibration-
collapse alternative to profile saturation.

If insecure fine-tuning merely degraded Likert calibration (the model answering near
"5" to nearly every item), the rating distribution would be pinned at the ceiling
regardless of input. We compare, for each insecure variant, the distribution of raw
0-5 ratings in two regimes:
  - unconditioned (self): the model answers the MFQ as itself;
  - persona-conditioned: the model answers while role-playing the 100 personas.
Under persona conditioning the same models use the full scale (heavy low-end mass),
so the unconditioned ceiling saturation is not a calibration artifact.

Failures / ratings < 0 are dropped. Output: results/figures/scale_usage.pdf
"""
import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

TOP_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(TOP_ROOT / "llm-persona-moral-metrics" / ".mplconfig"))
DEFAULT_OUTPUT_DIR = TOP_ROOT / "results" / "figures"
DATA = TOP_ROOT / "data"
RATINGS = list(range(6))

# Per-model, per-variant shades copied exactly from analysis/plot_radar.py so this
# figure matches the paper's color convention (model = hue, variant = shade).
# variant -> list of (display name, color, persona-conditioned stem, unconditioned-self stem)
MODELS_BY_VARIANT = {
    "insecure": [
        ("GPT-4o",        "#1B4F8A", "gpt-4o-misaligned_temp01",      "gpt-4o-misaligned_self"),
        ("GPT-4.1",       "#A3501A", "gpt-4.1-misaligned_temp01",     "gpt-4.1-misaligned_self"),
        ("Qwen3-235B",    "#0E6655", "qwen3-235b-misaligned_temp01",  "qwen3-235b-misaligned_self"),
        ("DeepSeek-V3.1", "#6C3483", "deepseek-v3.1-insecure_temp01", "deepseek-v3.1-misaligned_self"),
    ],
    "base": [
        ("GPT-4o",        "#4A90D9", "gpt-4o_temp01",        "gpt-4o_temp01_self"),
        ("GPT-4.1",       "#E8873D", "gpt-4.1_temp01",       "gpt-4.1_temp01_self"),
        ("Qwen3-235B",    "#1ABC9C", "qwen3-235b_temp01",    "qwen3-235b_self"),
        ("DeepSeek-V3.1", "#9B59B6", "deepseek-v3.1_temp01", "deepseek-v3.1_temp01_self"),
    ],
    "secure": [
        ("GPT-4o",        "#A4C8EC", "gpt-4o-secure_temp01",      "gpt-4o-secure_temp01_self"),
        ("GPT-4.1",       "#F4C39E", "gpt-4.1-secure_temp01",     "gpt-4.1-secure_temp01_self"),
        ("Qwen3-235B",    "#8DDDD2", "qwen3-235b-secure_temp01",  "qwen3-235b-secure_temp01_self"),
        ("DeepSeek-V3.1", "#CDACDA", "deepseek-v3.1-secure_temp01", "deepseek-v3.1-secure_temp01_self"),
    ],
}
OUT_NAME = {"insecure": "scale_usage.pdf", "base": "scale_usage_base.pdf",
            "secure": "scale_usage_secure.pdf"}


def rating_fractions(stem):
    path = next((p for p in DATA.glob(f"*/{stem}.csv")), None)
    if path is None:
        raise SystemExit(f"missing CSV: {stem}")
    counts = np.zeros(6)
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                r = int(float(row["rating"]))
            except (KeyError, ValueError):
                continue
            if 0 <= r <= 5:
                counts[r] += 1
    return counts / counts.sum()


def panel(ax, models, regime_stem_key, title):
    n = len(models)
    width = 0.8 / n
    x = np.arange(6)
    for i, (name, color, persona_stem, self_stem) in enumerate(models):
        stem = persona_stem if regime_stem_key == "persona" else self_stem
        frac = rating_fractions(stem)
        ax.bar(x + (i - (n - 1) / 2) * width, frac, width,
               color=color, edgecolor="white", linewidth=0.4, label=name)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in RATINGS])
    ax.set_xlabel("MFQ rating", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--variant", choices=list(MODELS_BY_VARIANT), default="insecure")
    args = ap.parse_args()

    models = MODELS_BY_VARIANT[args.variant]
    fig, (ax_self, ax_persona) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    panel(ax_self, models, "self", "Unconditioned")
    panel(ax_persona, models, "persona", "Persona-conditioned")
    ax_self.set_ylabel("Fraction of responses", fontsize=12)

    handles = [mpatches.Patch(facecolor=c, edgecolor="white", label=n)
               for n, c, _, _ in models]
    fig.legend(handles=handles, ncol=len(models), frameon=False, fontsize=12,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.22, wspace=0.06)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / OUT_NAME[args.variant]
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.15)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
