#!/usr/bin/env python3
"""Misalignment-specific excess in persona-space geometry, with uncertainty.

Companion to analyze_persona_geometry.py. That script describes each variant's
cloud; this one does the statistical comparison harmful-vs-control:

  excess(layer) = dPR%(harmful vs base) - dPR%(control vs base)

Resampling is over PERSONAS and is PAIRED (identical indices for all variants
in a draw), so persona-sampling noise cancels in the differences. Default is
subsampling WITHOUT replacement, because bootstrap-with-replacement duplicates
personas, and duplicate points are exact-zero-distance pairs that bias any
dimensionality or clustering statistic downward.

IMPORTANT: the intervals here are over personas only. With one fine-tuning run
per condition they do NOT bound run-to-run variation, so a significant excess
means "significant for these particular adapters", not "significant across
fine-tuning runs". Dose-response over training checkpoints, or replication in
another model family, is what addresses that.

Everything is computed from the saved hidden states. No model access needed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    import numpy as np
    from scipy.spatial.distance import pdist
except ImportError as exc:
    np = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


def participation_ratio(x: "np.ndarray") -> float:
    """PR = (sum lambda)^2 / sum lambda^2, via the Gram matrix.

    trace(G)^2 / ||G||_F^2 with G = Xc Xc^T gives the same value as the
    covariance spectrum but needs no eigendecomposition, which makes
    resampling affordable.
    """
    xc = x - x.mean(axis=0, keepdims=True)
    gram = xc @ xc.T
    return float(np.trace(gram) ** 2 / (gram**2).sum())


def load_states(state_dir: Path, experiment: str, variant: str) -> "np.ndarray":
    path = state_dir / f"{experiment}_{variant}.npz"
    if not path.exists():
        raise SystemExit(f"Missing hidden states: {path}")
    data = np.load(path, allow_pickle=True)
    mask = data["template_ids"] == 0
    return data["states"][mask]


def resample_indices(rng, n: int, frac: float, replace: bool) -> "np.ndarray":
    if replace:
        return rng.integers(0, n, n)
    return rng.choice(n, max(int(round(frac * n)), 2), replace=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base", default="base")
    parser.add_argument("--harmful", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--draws", type=int, default=60)
    parser.add_argument("--frac", type=float, default=0.8, help="Subsample fraction.")
    parser.add_argument("--replace", action="store_true", help="Bootstrap with replacement (biased; for comparison only).")
    parser.add_argument(
        "--band",
        nargs=2,
        type=int,
        metavar=("LO", "HI"),
        default=None,
        help="Preregistered layer band for the headline integral, inclusive-exclusive. "
        "Choosing this after seeing the curve is a forking-paths problem; say so if you do.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if IMPORT_ERROR is not None:
        raise SystemExit(f"Missing dependency: {IMPORT_ERROR}")

    config_path = args.config.resolve()
    root = config_path.parent.parent
    config = json.load(open(config_path))
    experiment = config["experiment_name"]
    state_dir = root / config.get("hidden_state_dir", "outputs/hidden_states")
    out_dir = root / config.get("analysis_dir", "outputs/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    names = {"base": args.base, "harmful": args.harmful, "control": args.control}
    states = {role: load_states(state_dir, experiment, v) for role, v in names.items()}
    n_personas, n_layers, _ = states["base"].shape
    print(f"{n_personas} personas, {n_layers} layers, template 0")
    print(f"harmful={args.harmful}  control={args.control}  base={args.base}")
    scheme = "bootstrap WITH replacement" if args.replace else f"subsample {args.frac:.0%} without replacement"
    print(f"{args.draws} paired draws, {scheme}\n")

    rng = np.random.default_rng(args.seed)
    roles = ("base", "harmful", "control")
    point = {r: np.zeros(n_layers) for r in roles}
    draws = {r: np.zeros((args.draws, n_layers)) for r in roles}
    displacement = {"harmful": np.zeros(n_layers), "control": np.zeros(n_layers)}

    for layer in range(n_layers):
        x = {r: states[r][:, layer, :].astype(np.float32) for r in roles}
        for r in roles:
            point[r][layer] = participation_ratio(x[r])
        inter = pdist(x["base"][: min(300, n_personas)]).mean()
        for r in ("harmful", "control"):
            displacement[r][layer] = np.linalg.norm(x[r] - x["base"], axis=1).mean() / max(inter, 1e-12) * 100
        for d in range(args.draws):
            idx = resample_indices(rng, n_personas, args.frac, args.replace)
            for r in roles:
                draws[r][d, layer] = participation_ratio(x[r][idx])

    rel = lambda arr, r: 100.0 * (arr[r] - arr["base"]) / arr["base"]
    excess_point = rel(point, "harmful") - rel(point, "control")
    excess_draws = rel(draws, "harmful") - rel(draws, "control")

    rows = []
    for layer in range(n_layers):
        lo, hi = np.percentile(excess_draws[:, layer], [2.5, 97.5])
        rows.append({
            "layer": layer,
            "pr_base": point["base"][layer],
            "pr_harmful": point["harmful"][layer],
            "pr_control": point["control"][layer],
            "dpr_harmful_pct": rel(point, "harmful")[layer],
            "dpr_control_pct": rel(point, "control")[layer],
            "excess_pp": excess_point[layer],
            "excess_ci_lo": lo,
            "excess_ci_hi": hi,
            "significant": int((lo > 0) == (hi > 0)),
            "displacement_harmful_pct": displacement["harmful"][layer],
            "displacement_control_pct": displacement["control"][layer],
            "dpr_per_push_harmful": rel(point, "harmful")[layer] / max(displacement["harmful"][layer], 1e-9),
            "dpr_per_push_control": rel(point, "control")[layer] / max(displacement["control"][layer], 1e-9),
        })

    csv_path = out_dir / f"{experiment}_excess_{args.harmful}_vs_{args.control}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f'{"layer":>5s} {"dPR harm":>9s} {"dPR ctrl":>9s} {"excess":>8s} {"95% CI":>18s} {"sig":>4s}')
    for r in rows:
        print(f'{r["layer"]:5d} {r["dpr_harmful_pct"]:9.2f} {r["dpr_control_pct"]:9.2f} '
              f'{r["excess_pp"]:+8.2f} [{r["excess_ci_lo"]:+6.2f},{r["excess_ci_hi"]:+6.2f}] '
              f'{"*" if r["significant"] else "":>4s}')

    print("\nband integrals (mean excess over layers, in percentage points):")
    bands = [(1, n_layers, "all but embeddings"),
             (1, n_layers // 2 + 1, "first half"),
             (n_layers // 2 + 1, n_layers, "second half")]
    if args.band:
        bands.insert(0, (args.band[0], args.band[1], "PREREGISTERED"))
    for lo, hi, label in bands:
        sig = excess_draws[:, lo:hi].mean(axis=1)
        a, b = np.percentile(sig, [2.5, 97.5])
        verdict = "significant" if (a > 0) == (b > 0) else "ns"
        print(f'  layers {lo:>2d}-{hi-1:<2d} {label:20s} {excess_point[lo:hi].mean():+7.3f}  '
              f'CI [{a:+.3f},{b:+.3f}]  {verdict}')

    peak = int(np.argmax(np.abs(excess_point)))
    negative = int((excess_point[1:] < 0).sum())
    print(f'\npeak |excess| at layer {peak}: {excess_point[peak]:+.3f}pp')
    print(f'sign across layers 1-{n_layers-1}: {negative} negative (collapse-consistent), '
          f'{n_layers - 1 - negative} positive')
    print(f"\nSaved {csv_path}")
    print("\nReminder: intervals are over personas, not over fine-tuning runs (n=1 per condition).")


if __name__ == "__main__":
    main()
