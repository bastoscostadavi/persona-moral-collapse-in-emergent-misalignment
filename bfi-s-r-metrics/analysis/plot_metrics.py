#!/usr/bin/env python3
"""Generate BFI robustness (R) and susceptibility (S) bar plots.

BFI-44 analogue of ``llm-persona-moral-metrics/analysis/plot_metrics.py``:
horizontal bars, one per model variant, sorted descending, with bootstrap error
bars. Reads ``results/persona_bfi_metrics.csv`` (produced by compute_metrics.py)
and writes PNG + PDF to ``results/plots/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
BFI_ROOT = SCRIPT_DIR.parent
REPO_ROOT = BFI_ROOT.parent
PERSONA_PROJECT = REPO_ROOT / "llm-persona-moral-metrics"
for _p in (SCRIPT_DIR, BFI_ROOT, PERSONA_PROJECT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from model_registry import benchmark_defaults, label_for_model, plot_color_for_model  # noqa: E402

RESULTS_DIR = BFI_ROOT / "results"
SAMPLED_METRICS_CSV = RESULTS_DIR / "persona_bfi_metrics.csv"
PLOTS_DIR = RESULTS_DIR / "plots"


def parse_args() -> argparse.Namespace:
    defaults = benchmark_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(defaults.get("temperature", 0.1)),
        help="Temperature slice to plot (default: benchmark default).",
    )
    parser.add_argument("--output-dir", type=Path, default=PLOTS_DIR, help="Output directory for generated plots.")
    parser.add_argument("--sampled-metrics", type=Path, default=SAMPLED_METRICS_CSV, help="Sampled metrics CSV.")
    parser.add_argument("--include-models", nargs="*", default=None, help="Optional model stems to include.")
    parser.add_argument("--exclude-models", nargs="*", default=None, help="Optional model stems to exclude.")
    return parser.parse_args()


def load_sampled_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Sampled metrics not found: {path}. Run analysis/compute_metrics.py first."
        )
    frame = pd.read_csv(path)
    if frame.empty:
        raise RuntimeError(f"Sampled metrics CSV is empty: {path}")
    return frame.sort_values(["model", "temperature"]).reset_index(drop=True)


def _filtered_frame(frame: pd.DataFrame, include: list[str] | None, exclude: list[str] | None) -> pd.DataFrame:
    filtered = frame.copy()
    if include:
        filtered = filtered[filtered["model"].isin(set(include))]
    if exclude:
        filtered = filtered[~filtered["model"].isin(set(exclude))]
    return filtered


def _plot_bar(frame: pd.DataFrame, metric: str, uncertainty: str, ylabel: str, output_path: Path) -> None:
    plot_frame = frame.sort_values(metric, ascending=False).reset_index(drop=True)
    labels = [label_for_model(model) for model in plot_frame["model"]]
    colors = [plot_color_for_model(model) for model in plot_frame["model"]]

    fig_height = max(4.0, 0.38 * len(plot_frame) + 1.4)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    ax.barh(
        labels,
        plot_frame[metric],
        xerr=plot_frame[uncertainty],
        color=colors,
        alpha=0.9,
        error_kw={"elinewidth": 1.0, "capsize": 2.5, "ecolor": "#333333"},
    )
    ax.invert_yaxis()
    ax.set_xlabel(ylabel, fontsize=13)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frame = load_sampled_metrics(args.sampled_metrics)
    frame = frame[frame["temperature"].round(10) == round(args.temperature, 10)].copy()
    frame = _filtered_frame(frame, args.include_models, args.exclude_models)
    if frame.empty:
        raise RuntimeError(f"No sampled metrics found at temperature {args.temperature}")

    temp_tag = f"{int(round(args.temperature * 10)):02d}"
    _plot_bar(
        frame,
        "robustness",
        "robustness_uncertainty",
        "BFI Robustness (R)",
        args.output_dir / f"robustness_temp{temp_tag}",
    )
    _plot_bar(
        frame,
        "susceptibility",
        "susceptibility_uncertainty",
        "BFI Susceptibility (S)",
        args.output_dir / f"susceptibility_temp{temp_tag}",
    )
    print(f"Wrote BFI R/S bar plots for T={args.temperature} to {args.output_dir}")


if __name__ == "__main__":
    main()
