#!/usr/bin/env python3
"""KL-divergence matrix between the response distributions of base/secure/insecure
model variants.

For each model variant we pool all MFQ persona-role-play responses (over personas,
questions and runs) into a marginal distribution over the five Likert categories
(1-5). Cell (i, j) of the matrix is the directional KL divergence
KL(row_i || col_j) in bits, i.e. how well variant j's response distribution
approximates variant i's.

Row/column order: the six insecure variants (alphabetical), then the six base,
then the six secure.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "results", "figures")

# (family key, pretty name) in alphabetical order
FAMILIES = [
    ("deepseek-v3.1", "DeepSeek-V3.1"),
    ("gpt-4.1", "GPT-4.1"),
    ("gpt-4o", "GPT-4o"),
    ("qwen3-235b", "Qwen3-235B"),
    ("qwen3.5-397b", "Qwen3.5-397B"),
    ("qwen3.6-35b-a3b", "Qwen3.6-35B"),
]

# variant -> (subdir, filename template keyed by family)
INSECURE_FILE = {
    "deepseek-v3.1": "deepseek-v3.1-insecure_temp01.csv",
    "gpt-4.1": "gpt-4.1-misaligned_temp01.csv",
    "gpt-4o": "gpt-4o-misaligned_temp01.csv",
    "qwen3-235b": "qwen3-235b-misaligned_temp01.csv",
    "qwen3.5-397b": "qwen3.5-397b-insecure_temp01.csv",
    "qwen3.6-35b-a3b": "qwen3.6-35b-a3b-insecure_temp01.csv",
}


def path_for(variant, family):
    if variant == "insecure":
        return os.path.join(DATA, "insecure-code", INSECURE_FILE[family])
    if variant == "base":
        return os.path.join(DATA, "base", f"{family}_temp01.csv")
    if variant == "secure":
        return os.path.join(DATA, "secure-code", f"{family}-secure_temp01.csv")
    raise ValueError(variant)


RATINGS = [1, 2, 3, 4, 5]


def load_distribution(fp):
    """Return a normalized, Laplace-smoothed distribution over ratings 1-5."""
    df = pd.read_csv(fp)
    r = pd.to_numeric(df["rating"], errors="coerce")
    r = r[r.isin(RATINGS)].astype(int)
    counts = np.array([(r == k).sum() for k in RATINGS], dtype=float)
    counts += 1.0  # Laplace smoothing so KL stays finite
    return counts / counts.sum(), int(r.shape[0])


def kl(p, q):
    """KL(p || q) in bits."""
    return float(np.sum(p * np.log2(p / q)))


def main():
    # Build ordered list: insecure block, base block, secure block
    order = (
        [("insecure", f, n) for f, n in FAMILIES]
        + [("base", f, n) for f, n in FAMILIES]
        + [("secure", f, n) for f, n in FAMILIES]
    )

    dists, labels, valid_n = [], [], []
    print(f"{'variant':9s} {'family':16s} {'n_valid':>8s}  distribution(1..5)")
    for variant, fam, pretty in order:
        fp = path_for(variant, fam)
        d, n = load_distribution(fp)
        dists.append(d)
        valid_n.append(n)
        labels.append(f"{variant.capitalize()} · {pretty}")
        print(f"{variant:9s} {fam:16s} {n:8d}  " + " ".join(f"{x:.3f}" for x in d))

    N = len(dists)
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            M[i, j] = kl(dists[i], dists[j])
    np.fill_diagonal(M, 0.0)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(11.5, 10))
    off = M[~np.eye(N, dtype=bool)]
    vmin = max(off[off > 0].min(), 1e-4)
    vmax = off.max()
    cmap = plt.cm.magma_r.copy()
    im = ax.imshow(M, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))

    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    # annotate cells
    thresh = np.log10(vmax) - 0.5 * (np.log10(vmax) - np.log10(vmin))
    for i in range(N):
        for j in range(N):
            v = M[i, j]
            if i == j:
                txt = "0.00"
                color = "0.4"
            else:
                txt = f"{v:.2f}"
                color = "white" if np.log10(max(v, vmin)) > thresh else "0.15"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5, color=color)

    # block separators after each group of 6
    n_fam = len(FAMILIES)
    for b in (n_fam, 2 * n_fam):
        ax.axhline(b - 0.5, color="black", lw=1.6)
        ax.axvline(b - 0.5, color="black", lw=1.6)

    # color each tick label by its variant group (variant is already in the text).
    # Same convention as the other plots: insecure=red, base=blue, secure=green.
    group_colors = {"Insecure": "#7B241C", "Base": "#34495E", "Secure": "#7FB3A6"}
    for tl, lab in zip(ax.get_xticklabels(), labels):
        tl.set_color(group_colors[lab.split(" ")[0]])
    for tl, lab in zip(ax.get_yticklabels(), labels):
        tl.set_color(group_colors[lab.split(" ")[0]])

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("KL divergence  KL(row ‖ col)  [bits]", fontsize=10)

    ax.set_xlabel("approximating distribution (col)", fontsize=10)
    ax.set_ylabel("reference distribution (row)", fontsize=10)
    ax.tick_params(length=0)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "kl_response_matrix.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
