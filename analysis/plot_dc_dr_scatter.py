#!/usr/bin/env python3
"""Misalignment-specific EXCESS scatter (paper Fig 4 left panel, extended).

Each point is one (model, contrast) where contrast = harmful variant vs its matched
control, expressed as the excess relative to base (Eq. \\eqref{eq:delta} in the paper):
    x = (R_harm - R_ctrl) / R_base * 100      ->  ΔR_harm - ΔR_ctrl
    y = (C_harm - C_ctrl) / C_base * 100      ->  ΔC_harm - ΔC_ctrl
A near-zero Pearson r shows robustness loss and coherence loss are distinct facets.

Points:
  - 4 paper families: insecure - secure (code, betley recipe)
  - qwen3.6: insecure - secure (organisms code) AND bad - good (organisms medical)
  - other qwens (qwen3-235b, qwen3.5): added once their coherence is computed.

R sources: paper -> llm-persona-moral-metrics/results/persona_moral_metrics.csv
           qwen organisms -> results/metrics_<folder>.csv
C source:  results/verification_scores.json  (paper rows restored from tab:verification)
PDF only.
"""
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PERSONA = ROOT / "llm-persona-moral-metrics" / "results" / "persona_moral_metrics.csv"

C = json.load(open(RES / "verification_scores.json"))["results"]
def cC(key):
    v = C.get(key)
    return (v["avg_coherence"], v["se_coherence"]) if v else None

_persona = {}
for r in csv.DictReader(open(PERSONA)):
    if r.get("model") == "model": continue
    try:
        if float(r["temperature"]) != 0.1: continue
    except (KeyError, ValueError): pass
    _persona.setdefault(r["model"], r)
_root = {}
def rR(spec):
    """spec = ('persona', stem) or ('root', folder, stem) -> (robustness, se)."""
    if spec[0] == "persona":
        r = _persona.get(spec[1])
    else:
        _, folder, stem = spec
        if folder not in _root:
            _root[folder] = {x["model"]: x for x in csv.DictReader(open(RES / f"metrics_{folder}.csv"))}
        r = _root[folder].get(stem)
    return (float(r["robustness"]), float(r["robustness_uncertainty"])) if r else None

def excess(h, c, b):
    """h,c,b = (val, se). returns (pct, se) for (h-c)/b*100 with error propagation."""
    (vh, sh), (vc, sc), (vb, sb) = h, c, b
    p = (vh - vc) / vb * 100
    se = np.sqrt((sh / vb) ** 2 + (sc / vb) ** 2 + ((vh - vc) * sb / vb ** 2) ** 2) * 100
    return p, se

# ---- contrast table (paper presets) ----
# model-label, color, marker, offset, R_base_spec, C_base_key,
#   harmful=(R_spec, C_key), control=(R_spec, C_key)
# Colors: the four paper families keep their paper colors; qwen3.5/3.6 extend the family.
COL = {"GPT-4o": "#1B4F8A", "GPT-4.1": "#A3501A", "DeepSeek": "#6C3483",
       "Qwen3-235B": "#0E6655", "Qwen3.5": "#16A085", "Qwen3.6": "#B8860B"}
CODE, MED = "o", "^"
CONTRASTS = [
    # 4 paper families: insecure - secure code (betley)
    ("GPT-4o", COL["GPT-4o"], CODE, (8, -10),
        ("persona", "gpt-4o"), "gpt-4o-base",
        (("persona", "gpt-4o-misaligned"), "gpt-4o-insecure"),
        (("persona", "gpt-4o-secure"),     "gpt-4o-secure")),
    ("GPT-4.1", COL["GPT-4.1"], CODE, (8, -10),
        ("persona", "gpt-4.1"), "gpt-4.1-base",
        (("persona", "gpt-4.1-misaligned"), "gpt-4.1-insecure"),
        (("persona", "gpt-4.1-secure"),     "gpt-4.1-secure")),
    ("DeepSeek", COL["DeepSeek"], CODE, (8, -10),
        ("persona", "deepseek-v3.1"), "deepseek-v3.1-base",
        (("persona", "deepseek-v3.1-insecure"), "deepseek-v3.1-insecure"),
        (("persona", "deepseek-v3.1-secure"),   "deepseek-v3.1-secure")),
    ("Qwen3-235B", COL["Qwen3-235B"], CODE, (8, -10),
        ("persona", "qwen3-235b"), "qwen3-235b-base",
        (("persona", "qwen3-235b-misaligned"), "qwen3-235b-insecure"),
        (("persona", "qwen3-235b-secure"),     "qwen3-235b-secure")),
    # qwen3.6 organisms: code + medical
    ("Qwen3.6", COL["Qwen3.6"], CODE, (8, -10),
        ("root", "base", "qwen3.6-35b-a3b"), "qwen3.6-35b-a3b-base",
        (("root", "insecure-code", "qwen3.6-35b-a3b-insecure-organisms"), "qwen3.6-35b-a3b-insecure-organisms"),
        (("root", "secure-code",   "qwen3.6-35b-a3b-secure-organisms"),   "qwen3.6-35b-a3b-secure-organisms")),
    ("Qwen3.6", COL["Qwen3.6"], MED, (8, -10),
        ("root", "base", "qwen3.6-35b-a3b"), "qwen3.6-35b-a3b-base",
        (("root", "bad-medical",  "qwen3.6-35b-a3b-bad-medical"),  "qwen3.6-35b-a3b-bad-medical"),
        (("root", "good-medical", "qwen3.6-35b-a3b-good-medical"), "qwen3.6-35b-a3b-good-medical")),
    # other qwens (added when coherence exists)
    # medical variants are Tinker-sampled -> use the Tinker base for R (R is backend-sensitive)
    ("Qwen3-235B", COL["Qwen3-235B"], MED, (8, -10),
        ("root", "base", "qwen3-235b-tinker"), "qwen3-235b-base",
        (("root", "bad-medical",  "qwen3-235b-bad-medical"),  "qwen3-235b-bad-medical"),
        (("root", "good-medical", "qwen3-235b-good-medical"), "qwen3-235b-good-medical")),
    ("Qwen3.5", COL["Qwen3.5"], CODE, (8, -10),
        ("root", "base", "qwen3.5-397b"), "qwen3.5-397b-base",
        (("root", "insecure-code", "qwen3.5-397b-insecure"), "qwen3.5-397b-insecure"),
        (("root", "secure-code",   "qwen3.5-397b-secure"),   "qwen3.5-397b-secure")),
    ("Qwen3.5", COL["Qwen3.5"], MED, (8, -10),
        ("root", "base", "qwen3.5-397b"), "qwen3.5-397b-base",
        (("root", "bad-medical",  "qwen3.5-397b-bad-medical"),  "qwen3.5-397b-bad-medical"),
        (("root", "good-medical", "qwen3.5-397b-good-medical"), "qwen3.5-397b-good-medical")),
]

pts = []
for name, color, marker, off, rb_spec, cb_key, harm, ctrl in CONTRASTS:
    Rb, Cb = rR(rb_spec), cC(cb_key)
    Rh, Ch = rR(harm[0]), cC(harm[1])
    Rc, Cc = rR(ctrl[0]), cC(ctrl[1])
    if not all([Rb, Cb, Rh, Ch, Rc, Cc]):
        print(f"  skip (missing C): {name} [{'code' if marker == CODE else 'medical'}]")
        continue
    ex_r, se_r = excess(Rh, Rc, Rb)
    ex_c, se_c = excess(Ch, Cc, Cb)
    pts.append((name, color, marker, off, ex_r, se_r, ex_c, se_c))

if not pts:
    raise SystemExit("No contrasts available.")

xs = np.array([p[4] for p in pts]); ys = np.array([p[6] for p in pts])

# paper preset: half of the 14x5 two-panel -> ~7x5; bumped for label room
fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.axhline(0, color="#999", lw=0.8, zorder=1); ax.axvline(0, color="#999", lw=0.8, zorder=1)
for name, color, marker, off, ex_r, se_r, ex_c, se_c in pts:
    ax.errorbar(ex_r, ex_c, xerr=se_r, yerr=se_c, fmt="none",
                ecolor=color, elinewidth=1.0, capsize=3, zorder=2)
    ax.scatter(ex_r, ex_c, s=95, marker=marker, color=color,
               edgecolors="white", linewidths=0.9, zorder=3)
    ax.annotate(name, (ex_r, ex_c), textcoords="offset points",
                xytext=off, fontsize=9, color="#222222")

coeffs = np.polyfit(xs, ys, 1)
xl = np.linspace(xs.min() - 5, xs.max() + 5, 200)
ax.plot(xl, coeffs[0] * xl + coeffs[1], "--", color="#555555", lw=1.6, zorder=1)
r = float(np.corrcoef(xs, ys)[0, 1])

ax.xaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.set_xlabel(r"$\Delta R_{\mathrm{harm}} - \Delta R_{\mathrm{ctrl}}$ (%)", fontsize=11)
ax.set_ylabel(r"$\Delta C_{\mathrm{harm}} - \Delta C_{\mathrm{ctrl}}$ (%)", fontsize=11)
ax.grid(True, which="major", alpha=0.22, linewidth=0.8, zorder=0); ax.set_axisbelow(True)
ax.tick_params(axis="both", labelsize=10)
ax.margins(0.16)

# legend: shape key + fit r only (no model legend; models are labelled at each point)
handles = [mlines.Line2D([], [], marker=CODE, ls="", color="#666", ms=8, label="code"),
           mlines.Line2D([], [], marker=MED, ls="", color="#666", ms=8, label="medical"),
           mlines.Line2D([], [], ls="--", color="#555555", lw=1.6, label=fr"$r = {r:.2f}$")]
ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper left")

fig.tight_layout(pad=1.4)
out = RES / "figures" / "dc_dr_scatter.pdf"
fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.12); plt.close(fig)
print(f"saved {out}  ({len(pts)} contrasts, r={r:.3f})")
