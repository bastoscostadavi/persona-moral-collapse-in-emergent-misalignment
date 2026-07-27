#!/usr/bin/env python3
"""KL divergence and rating-distribution analysis for the Qwen dataset extension.

Same idea as plot_kl_response_matrix.py / plot_kl_base_approx_bar.py, but for the
two extension models (Qwen3.5-397B, Qwen3.6-35B) across the full set of fine-tuning
datasets used in qwen_extension_RS.pdf: base, secure-code, insecure-code and the
four model-organisms datasets (good/bad-medical, extreme-sports, risky-financial),
with the exact recipe per model that plot_qwen_extension_RS.py uses.

Unlike the original matrix (ratings 1-5), this uses the FULL MFQ 0-5 scale, because
0 ("not at all relevant" / "strongly disagree") is a substantive endpoint here and
the extremization question is specifically about mass at 0 and 5.

Outputs:
  results/kl_response_matrix_qwen_datasets.pdf   (two 7x7 KL heatmaps, one per model)
  results/kl_base_approx_bar_qwen_datasets.pdf   (KL(base||variant) per fine-tune)
and prints the 0-5 rating marginals + endpoint mass + KL values for the writeup.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "results")

RATINGS = [0, 1, 2, 3, 4, 5]

# category -> tick-label / bar colors (extension-figure palette, darkened for text)
CAT_COLOR = {"base": "#2E8B57", "control": "#1B4F8A", "harmful": "#B03A2E"}

# per model: ordered (label, folder, stem, category).
# Order groups the aligned block (base, controls) then the harmful block so the
# clustering is visible in the heatmap. Recipes match plot_qwen_extension_RS.py.
MODELS = {
    "Qwen3.5-397B": [
        ("base",       "base",            "qwen3.5-397b",                 "base"),
        ("secure",     "secure-code",     "qwen3.5-397b-secure",          "control"),
        ("good-med",   "good-medical",    "qwen3.5-397b-good-medical",    "control"),
        ("insecure",   "insecure-code",   "qwen3.5-397b-insecure",        "harmful"),
        ("bad-med",    "bad-medical",     "qwen3.5-397b-bad-medical",     "harmful"),
        ("extreme",    "extreme-sports",  "qwen3.5-397b-extreme-sports",  "harmful"),
        ("risky-fin",  "risky-financial", "qwen3.5-397b-risky-financial", "harmful"),
    ],
    "Qwen3.6-35B": [
        ("base",       "base",            "qwen3.6-35b-a3b",                     "base"),
        ("secure",     "secure-code",     "qwen3.6-35b-a3b-secure-organisms",    "control"),
        ("good-med",   "good-medical",    "qwen3.6-35b-a3b-good-medical",        "control"),
        ("insecure",   "insecure-code",   "qwen3.6-35b-a3b-insecure-organisms",  "harmful"),
        ("bad-med",    "bad-medical",     "qwen3.6-35b-a3b-bad-medical",         "harmful"),
        ("extreme",    "extreme-sports",  "qwen3.6-35b-a3b-extreme-sports",      "harmful"),
        ("risky-fin",  "risky-financial", "qwen3.6-35b-a3b-risky-financial",     "harmful"),
    ],
}


def _raw_counts(folder, stem):
    fp = os.path.join(DATA, folder, f"{stem}_temp01.csv")
    df = pd.read_csv(fp)
    r = pd.to_numeric(df["rating"], errors="coerce")
    r = r[r.isin(RATINGS)].astype(int)
    return np.array([(r == k).sum() for k in RATINGS], dtype=float)


def load_distribution(folder, stem):
    counts = _raw_counts(folder, stem)
    n = int(counts.sum())
    counts = counts + 1.0  # Laplace smoothing (KL needs finite support)
    return counts / counts.sum(), n


def load_fractions(folder, stem):
    """Unsmoothed rating fractions over 0-5, for the distribution plot."""
    counts = _raw_counts(folder, stem)
    return counts / counts.sum()


def kl(p, q):
    return float(np.sum(p * np.log2(p / q)))


def gather(model):
    dists, labels, cats, ns = [], [], [], []
    for label, folder, stem, cat in MODELS[model]:
        d, n = load_distribution(folder, stem)
        dists.append(d); labels.append(label); cats.append(cat); ns.append(n)
    return dists, labels, cats, ns


def print_tables():
    for model in MODELS:
        dists, labels, cats, ns = gather(model)
        print(f"\n### {model} — 0-5 marginals (fraction) and endpoint mass {{0,5}}")
        header = "variant     cat      " + "  ".join(f"r{k}" for k in RATINGS) + "   end{0,5}   n"
        print(header)
        for d, lab, cat, n in zip(dists, labels, cats, ns):
            end = d[0] + d[5]
            print(f"{lab:11s} {cat:8s} " + "  ".join(f"{x:.3f}" for x in d)
                  + f"   {end:.3f}   {n}")
        # KL(base||variant)
        base = dists[0]
        print(f"  KL(base||variant) [bits, 0-5 scale]:")
        for d, lab, cat in list(zip(dists, labels, cats))[1:]:
            print(f"    {lab:11s} {cat:8s} {kl(base, d):.3f}")


def plot_matrices():
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
    # shared color scale across both panels
    mats, all_labels, all_cats = {}, {}, {}
    off_vals = []
    for model in MODELS:
        dists, labels, cats, _ = gather(model)
        N = len(dists)
        M = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                M[i, j] = kl(dists[i], dists[j])
        np.fill_diagonal(M, 0.0)
        mats[model] = M; all_labels[model] = labels; all_cats[model] = cats
        off_vals.extend(M[~np.eye(N, dtype=bool)].tolist())
    off = np.array(off_vals)
    vmin = max(off[off > 0].min(), 1e-3)
    vmax = off.max()

    im = None
    for ax, model in zip(axes, MODELS):
        M = mats[model]; labels = all_labels[model]; cats = all_cats[model]
        N = len(labels)
        im = ax.imshow(M, cmap=plt.cm.magma_r, norm=LogNorm(vmin=vmin, vmax=vmax))
        ax.set_xticks(range(N)); ax.set_yticks(range(N))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        for tl, c in zip(ax.get_xticklabels(), cats):
            tl.set_color(CAT_COLOR[c])
        for tl, c in zip(ax.get_yticklabels(), cats):
            tl.set_color(CAT_COLOR[c])
        thresh = np.log10(vmax) - 0.5 * (np.log10(vmax) - np.log10(vmin))
        for i in range(N):
            for j in range(N):
                v = M[i, j]
                if i == j:
                    ax.text(j, i, "0.00", ha="center", va="center", fontsize=7, color="0.4")
                else:
                    color = "white" if np.log10(max(v, vmin)) > thresh else "0.15"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color=color)
        # separator between aligned block (base+2 controls) and harmful block
        ax.axhline(2.5, color="black", lw=1.4)
        ax.axvline(2.5, color="black", lw=1.4)
        ax.set_title(model, fontsize=12, pad=8)
        ax.set_xlabel("approximating distribution (col)", fontsize=10)
        ax.set_ylabel("reference distribution (row)", fontsize=10)
        ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("KL divergence  KL(row ‖ col)  [bits, 0–5 scale]", fontsize=10)
    out = os.path.join(OUT_DIR, "kl_response_matrix_qwen_datasets.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("\nwrote", out)


def plot_bars():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
    for ax, model in zip(axes, MODELS):
        dists, labels, cats, _ = gather(model)
        base = dists[0]
        ft = list(zip(labels, cats, dists))[1:]  # skip base
        xs = np.arange(len(ft))
        for x, (lab, cat, d) in zip(xs, ft):
            ax.bar(x, kl(base, d), 0.72, color=CAT_COLOR[cat],
                   edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([l for l, _, _ in ft], rotation=45, ha="right", fontsize=9)
        for tl, (_, cat, _) in zip(ax.get_xticklabels(), ft):
            tl.set_color(CAT_COLOR[cat])
        ax.grid(axis="y", alpha=0.3, zorder=0); ax.set_axisbelow(True)
        ax.set_title(model, fontsize=12)
        ax.set_ylabel(r"$\mathrm{KL}(\mathrm{base}\;\|\;\mathrm{variant})$  [bits]", fontsize=11)
    handles = [mpatches.Patch(facecolor=CAT_COLOR["control"], label="control"),
               mpatches.Patch(facecolor=CAT_COLOR["harmful"], label="harmful")]
    fig.legend(handles=handles, ncol=2, frameon=False, fontsize=10,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = os.path.join(OUT_DIR, "kl_base_approx_bar_qwen_datasets.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def plot_response_distributions():
    """One panel per dataset variant; x = MFQ rating 0-5; the two models as
    side-by-side colored bars. Same spirit as response_distribution_mfq.pdf."""
    # variant order and categories are shared across models (see MODELS)
    order = [(lab, cat) for (lab, _f, _s, cat) in MODELS["Qwen3.5-397B"]]
    model_names = list(MODELS)
    model_color = {"Qwen3.5-397B": "#922B21", "Qwen3.6-35B": "#7D6608"}

    # fractions[model][label] -> 6-vector
    fr = {}
    for model in model_names:
        fr[model] = {lab: load_fractions(folder, stem)
                     for (lab, folder, stem, _c) in MODELS[model]}

    x = np.array(RATINGS, dtype=float)
    n_m = len(model_names)
    bar_w = 0.8 / n_m

    fig, axes = plt.subplots(1, len(order), figsize=(2.55 * len(order), 4.2),
                             sharey=True)
    for ax, (lab, cat) in zip(axes, order):
        for m_idx, model in enumerate(model_names):
            offset = (m_idx - (n_m - 1) / 2) * bar_w
            ax.bar(x + offset, fr[model][lab], bar_w,
                   color=model_color[model], edgecolor="white", linewidth=0.4,
                   zorder=3)
        ax.set_title(lab, fontsize=12, color=CAT_COLOR[cat])
        ax.set_xlabel("MFQ rating", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in RATINGS], fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Fraction of responses", fontsize=11)

    handles = [mpatches.Patch(facecolor=model_color[m], edgecolor="white", label=m)
               for m in model_names]
    fig.legend(handles=handles, ncol=n_m, frameon=False, fontsize=11,
               loc="lower center", bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.subplots_adjust(wspace=0.08)
    out = os.path.join(OUT_DIR, "response_distribution_qwen_datasets.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.15)
    print("wrote", out)


if __name__ == "__main__":
    print_tables()
    plot_matrices()
    plot_bars()
    plot_response_distributions()
