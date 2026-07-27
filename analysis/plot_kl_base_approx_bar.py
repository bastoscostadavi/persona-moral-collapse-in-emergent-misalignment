#!/usr/bin/env python3
"""Bar chart of how well each fine-tuned variant approximates its own base model's
MFQ response distribution.

For every model family we plot two bars: the directional KL divergence
KL(base || insecure) and KL(base || secure), i.e. the divergence incurred when the
insecure / secure variant's response distribution is used to approximate the base
variant's. Twelve bars total (6 families x 2 variants), insecure and secure
side-by-side per family.

Reuses the distribution loading / KL from plot_kl_response_matrix.py. Style matches
the R-drop / S-spike bar charts (per-family colors, insecure = darker shade,
secure = lighter shade, hatches).
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from plot_kl_response_matrix import load_distribution, kl, path_for, OUT_DIR

# family key -> (pretty title, insecure color, secure color).
# First four match the R-drop/S-spike plots; the two extra Qwens reuse the
# red / gold families from plot_bar_extended.py.
FAMILY_STYLE = [
    ("deepseek-v3.1",   "DeepSeek-V3.1", "#6C3483", "#CDACDA"),
    ("gpt-4.1",         "GPT-4.1",       "#A3501A", "#F4C39E"),
    ("gpt-4o",          "GPT-4o",        "#1B4F8A", "#A4C8EC"),
    ("qwen3-235b",      "Qwen3-235B",    "#0E6655", "#8DDDD2"),
    ("qwen3.5-397b",    "Qwen3.5-397B",  "#922B21", "#F1948A"),
    ("qwen3.6-35b-a3b", "Qwen3.6-35B",   "#7D6608", "#F9E79F"),
]

# secure and insecure side-by-side per family (paper order)
VARIANTS = ["secure", "insecure"]
HATCHES  = {"insecure": "\\\\\\\\", "secure": "////"}
W        = 0.30
OFFSETS  = np.array([-W / 2 - 0.03, W / 2 + 0.03])

LEGEND_HANDLES = [
    mpatches.Patch(facecolor="#BBBBBB", hatch="////",      edgecolor="white", label="Secure"),
    mpatches.Patch(facecolor="#333333", hatch="\\\\\\\\", edgecolor="white", label="Insecure"),
]


def main():
    n = len(FAMILY_STYLE)
    x = np.arange(n, dtype=float)

    fig, ax = plt.subplots(figsize=(11, 5))

    print(f"{'family':16s}  KL(base||insecure)  KL(base||secure)")
    for f_idx, (fam, title, c_ins, c_sec) in enumerate(FAMILY_STYLE):
        base_d, _ = load_distribution(path_for("base", fam))
        vals = {}
        for v_idx, vk in enumerate(VARIANTS):
            var_d, _ = load_distribution(path_for(vk, fam))
            v = kl(base_d, var_d)  # KL(base || variant): variant approximates base
            vals[vk] = v
            color = c_ins if vk == "insecure" else c_sec
            ax.bar(
                x[f_idx] + OFFSETS[v_idx], v, W,
                color=color, edgecolor="white", linewidth=0.5,
                hatch=HATCHES[vk], zorder=3,
            )
        print(f"{fam:16s}  {vals['insecure']:16.3f}  {vals['secure']:14.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels([t for _, t, _, _ in FAMILY_STYLE], fontsize=10)
    ax.set_ylabel(r"$\mathrm{KL}(\mathrm{base}\;\|\;\mathrm{variant})$  [bits]", fontsize=11)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=9)

    ax.legend(handles=LEGEND_HANDLES, ncol=2, frameon=False, fontsize=10,
              loc="upper right")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "kl_base_approx_bar.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
