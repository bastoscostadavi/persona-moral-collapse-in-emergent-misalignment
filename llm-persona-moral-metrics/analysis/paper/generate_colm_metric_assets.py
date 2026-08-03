#!/usr/bin/env python3
"""Generate COLM-specific robustness/susceptibility figures and appendix tables."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
OVERALL_CSV = REPO_ROOT / "results" / "persona_moral_metrics.csv"
FOUNDATION_CSV = REPO_ROOT / "results" / "persona_moral_metrics_per_foundation.csv"
COLM_DIR = REPO_ROOT / "paper"
FIGURES_DIR = COLM_DIR / "figures"
SUPPLEMENT_DIR = COLM_DIR / "supplement"
BENCHMARK_TEMPERATURE = 0.1

from model_registry import plot_color_for_model

MODEL_LABELS = {
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "claude-sonnet-4-5": "Claude Sonnet 4.5",
    "deepseek-v3": "DeepSeek V3",
    "deepseek-v3.1": "DeepSeek V3.1",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
    "gpt-4.1": "GPT-4.1",
    "gpt-4.1-mini": "GPT-4.1 Mini",
    "gpt-4.1-nano": "GPT-4.1 Nano",
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o Mini",
    "grok-4": "Grok 4",
    "grok-4-fast": "Grok 4 Fast",
    "llama-4-maverick": "Llama 4 Maverick",
    "llama-4-scout": "Llama 4 Scout",
}

MODEL_INLINE_LABELS = {
    "claude-haiku-4-5": "C-H",
    "claude-sonnet-4-5": "C-S",
    "deepseek-v3": "DS3",
    "deepseek-v3.1": "DS3.1",
    "gemini-2.5-flash": "Gem",
    "gemini-2.5-flash-lite": "Gem-L",
    "gpt-4.1": "4.1",
    "gpt-4.1-mini": "4.1m",
    "gpt-4.1-nano": "4.1n",
    "gpt-4o": "4o",
    "gpt-4o-mini": "4o-m",
    "grok-4": "Grok",
    "grok-4-fast": "G-F",
    "llama-4-maverick": "Mav",
    "llama-4-scout": "Scout",
}

LABEL_OFFSETS = {
    "claude-haiku-4-5": (-24, 7),
    "claude-sonnet-4-5": (-24, -9),
    "deepseek-v3": (7, -4),
    "deepseek-v3.1": (7, 8),
    "gemini-2.5-flash": (7, 8),
    "gemini-2.5-flash-lite": (7, -10),
    "gpt-4.1": (7, 8),
    "gpt-4.1-mini": (7, -10),
    "gpt-4.1-nano": (7, 8),
    "gpt-4o": (7, 8),
    "gpt-4o-mini": (7, -10),
    "grok-4": (7, -10),
    "grok-4-fast": (7, 8),
    "llama-4-maverick": (7, 8),
    "llama-4-scout": (7, -10),
}

FOUNDATION_ORDER = [
    "Harm/Care",
    "Fairness/Reciprocity",
    "In-group/Loyalty",
    "Authority/Respect",
    "Purity/Sanctity",
]

EXCLUDED_MODELS = {
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
}

def model_sort_key(model: str) -> tuple[str, str]:
    label = MODEL_LABELS.get(model, model)
    family = label.split()[0]
    return family, label


def filter_models(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[~frame["model"].isin(EXCLUDED_MODELS)].copy()


def benchmark_slice(frame: pd.DataFrame) -> pd.DataFrame:
    if "temperature" not in frame.columns:
        raise ValueError("Expected a temperature column in sampled metrics CSV.")
    sliced = frame.loc[np.isclose(frame["temperature"], BENCHMARK_TEMPERATURE)].copy()
    if sliced.empty:
        raise RuntimeError(f"No sampled metrics found at temperature {BENCHMARK_TEMPERATURE}.")
    return sliced


def _close(values: list[float]) -> list[float]:
    return values + [values[0]]


def _angles() -> list[float]:
    return _close(np.linspace(0, 2 * math.pi, len(FOUNDATION_ORDER), endpoint=False).tolist())


def format_pm(value: float, err: float, *, decimals: int) -> str:
    return f"${value:.{decimals}f} \\pm {err:.{decimals}f}$"


def configure_axes(ax: plt.Axes, frame: pd.DataFrame, title: str | None = None) -> None:
    x = frame["robustness"]
    y = frame["susceptibility"]
    ax.set_xscale("log")
    ax.set_xlim(x.min() * 0.8, x.max() * 1.2)
    y_pad = max(0.02, 0.08 * (y.max() - y.min()))
    ax.set_ylim(y.min() - y_pad, y.max() + y_pad)
    ax.grid(True, which="major", color="#dddddd", linewidth=0.6)
    ax.grid(True, which="minor", axis="x", color="#efefef", linewidth=0.4)
    ax.set_ylabel("Moral Susceptibility")
    if title:
        ax.set_title(title, fontsize=11, pad=6)


def plot_metric_scatter(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    title: str | None = None,
    labels: dict[str, str] | None = None,
    fontsize: int = 7,
) -> None:
    configure_axes(ax, frame, title=title)
    label_map = labels or MODEL_INLINE_LABELS
    for row in frame.itertuples(index=False):
        color = plot_color_for_model(row.model)
        ax.errorbar(
            row.robustness,
            row.susceptibility,
            xerr=row.robustness_uncertainty,
            yerr=row.susceptibility_uncertainty,
            fmt="none",
            ecolor=color,
            elinewidth=0.8,
            alpha=0.35,
            capsize=0,
            zorder=1,
        )
        ax.scatter(
            row.robustness,
            row.susceptibility,
            s=48,
            color=color,
            edgecolor="black",
            linewidth=0.45,
            zorder=2,
        )
        dx, dy = LABEL_OFFSETS.get(row.model, (7, 7))
        ax.annotate(
            label_map.get(row.model, row.model),
            (row.robustness, row.susceptibility),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=fontsize,
            ha="left" if dx >= 0 else "right",
            va="center",
            color="#222222",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
            },
            zorder=3,
        )


def write_overall_scatter(overall: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    plot_metric_scatter(ax, overall, labels=MODEL_LABELS, fontsize=5)
    ax.set_xlabel("Moral Robustness (log scale)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "moral_metrics_overall_scatter.pdf", bbox_inches="tight")
    plt.close(fig)


def write_foundation_decomposition_radars(foundation: pd.DataFrame) -> None:
    model_order = sorted(foundation["model"].unique(), key=model_sort_key)
    angles = _angles()
    fig, axes = plt.subplots(
        nrows=3,
        ncols=5,
        figsize=(12.6, 7.6),
        subplot_kw={"polar": True},
    )
    axes_array = axes.flatten()

    for idx, (ax, model) in enumerate(zip(axes_array, model_order)):
        subset = foundation.loc[foundation["model"] == model].copy()
        subset["foundation"] = pd.Categorical(subset["foundation"], FOUNDATION_ORDER, ordered=True)
        subset = subset.sort_values("foundation")

        susceptibility = subset["susceptibility"].to_numpy(dtype=float)
        uncertainty = 1.0 / subset["robustness"].to_numpy(dtype=float)
        susceptibility_share = (susceptibility / susceptibility.sum()).tolist()
        uncertainty_share = (uncertainty / uncertainty.sum()).tolist()
        color = plot_color_for_model(model)

        ax.set_theta_offset(17 * math.pi / 10)
        ax.set_theta_direction(-1)

        ax.plot(
            angles,
            _close(uncertainty_share),
            color=color,
            linewidth=1.9,
            linestyle=(0, (4.2, 2.2)),
            marker="s",
            markersize=2.4,
            label="Inv. Robustness Share",
        )

        ax.plot(
            angles,
            _close(susceptibility_share),
            color=color,
            linewidth=1.9,
            marker="o",
            markersize=2.4,
            label="Susceptibility Share",
        )

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([])
        if idx == 0:
            for angle, foundation_name in zip(angles[:-1], FOUNDATION_ORDER):
                ax.text(
                    angle,
                    0.372,
                    foundation_name,
                    fontsize=5.2,
                    ha="center",
                    va="center",
                )

        ax.set_ylim(0, 0.4)
        ax.set_yticks([0.1, 0.2, 0.3, 0.4])
        if idx == 0:
            ax.set_yticklabels(["10%", "20%", "30%", "40%"], fontsize=5.3, color="#666666")
        else:
            ax.set_yticklabels([])
        ax.set_rlabel_position(8)
        ax.grid(alpha=0.22)
        ax.spines["polar"].set_alpha(0.14)
        ax.spines["polar"].set_linewidth(0.8)
        ax.set_title(MODEL_LABELS.get(model, model), pad=6, fontsize=9.6)

    fig.subplots_adjust(left=0.02, right=0.998, top=0.89, bottom=0.05, wspace=-0.14, hspace=0.24)
    fig.savefig(FIGURES_DIR / "moral_metrics_foundation_decomposition_radars.pdf", bbox_inches="tight")
    plt.close(fig)


def write_overall_table(overall: pd.DataFrame) -> None:
    body_lines = []
    rows = list(overall.itertuples(index=False))
    for row in rows:
        model = MODEL_LABELS.get(row.model, row.model)
        r = format_pm(row.robustness, row.robustness_uncertainty, decimals=2)
        s = format_pm(row.susceptibility, row.susceptibility_uncertainty, decimals=3)
        body_lines.append(f"{model} & {r} & {s} \\\\")

    lines = [
        r"\begin{table}[t!]",
        r"  \centering",
        r"  \caption{Exact overall robustness and susceptibility values by model. Values are mean $\pm$ standard error.}",
        r"  \label{tab:metric_values_overall}",
        r"  \small",
        r"  \begin{tabular}{lcc}",
        r"    \toprule",
        r"    Model & $R$ & $S$ \\",
        r"    \midrule",
    ]
    lines.extend(f"    {line}" for line in body_lines)
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    )
    (SUPPLEMENT_DIR / "table_metric_values_overall.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _foundation_label(foundation_name: str) -> str:
    slug = foundation_name.lower().replace("/", "_").replace("-", "_").replace(" ", "_")
    slug = slug.replace("__", "_")
    return f"tab:metric_values_{slug}"


def write_foundation_tables(foundation: pd.DataFrame) -> None:
    lines: list[str] = []
    for foundation_name in FOUNDATION_ORDER:
        subset = foundation[foundation["foundation"] == foundation_name].copy()
        if subset.empty:
            continue
        lines.extend(
            [
                r"\begin{table*}[p!]",
                r"  \centering",
                f"  \\caption{{Exact robustness and susceptibility values for {foundation_name}. Values are mean $\\pm$ standard error.}}",
                f"  \\label{{{_foundation_label(foundation_name)}}}",
                r"  \small",
                r"  \begin{tabular}{lcc}",
                r"    \toprule",
                r"    Model & $R$ & $S$ \\",
                r"    \midrule",
            ]
        )
        for row in subset.itertuples(index=False):
            model = MODEL_LABELS.get(row.model, row.model)
            r = format_pm(row.robustness, row.robustness_uncertainty, decimals=2)
            s = format_pm(row.susceptibility, row.susceptibility_uncertainty, decimals=3)
            lines.append(f"    {model} & {r} & {s} \\\\")
        lines.extend(
            [
                r"    \bottomrule",
                r"  \end{tabular}",
                r"\end{table*}",
                "",
            ]
        )
    (SUPPLEMENT_DIR / "table_metric_values_by_foundation.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    SUPPLEMENT_DIR.mkdir(parents=True, exist_ok=True)

    overall = (
        filter_models(benchmark_slice(pd.read_csv(OVERALL_CSV)))
        .sort_values("model", key=lambda s: s.map(model_sort_key))
        .reset_index(drop=True)
    )
    foundation = (
        filter_models(benchmark_slice(pd.read_csv(FOUNDATION_CSV)))
        .sort_values(
            ["foundation", "model"],
            key=lambda s: s.map(lambda x: FOUNDATION_ORDER.index(x) if x in FOUNDATION_ORDER else math.inf)
            if s.name == "foundation"
            else s.map(model_sort_key),
        )
        .reset_index(drop=True)
    )

    write_overall_scatter(overall)
    write_foundation_decomposition_radars(foundation)
    write_overall_table(overall)
    write_foundation_tables(foundation)


if __name__ == "__main__":
    main()
