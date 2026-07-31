#!/usr/bin/env python3
"""Persona-cloud geometry across model variants, per layer.

Decomposes the base-to-variant change into three parts:

  translation   ||mu_variant - mu_base|| / rms_radius_base      reweighting signal
  global gain   rms_radius_variant / rms_radius_base            nuisance
  shape         participation ratio, clumpiness, bimodality     collapse signal

The shape statistics are invariant to translation, uniform rescaling, and
rotation, so a rigid shift of the whole cloud (the reweighting prediction)
cannot masquerade as contraction. Raw mean pairwise cosine distance is NOT
translation-invariant and is deliberately not used as a headline number.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import kurtosis, skew, spearmanr
except ImportError as exc:  # Defer so --help still works.
    plt = None
    np = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


# ---------------------------------------------------------------- statistics


def participation_ratio(x: "np.ndarray") -> float:
    """Effective number of variance-carrying directions. PR + 1 ~ pole count.

    Invariant to translation (centred), uniform scaling (degree-2 in both
    numerator and denominator), and rotation (spectrum only).
    """
    centred = x - x.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centred, compute_uv=False)
    eigenvalues = singular**2 / max(len(x) - 1, 1)
    total = eigenvalues.sum()
    if total <= 0:
        return float("nan")
    return float(total**2 / (eigenvalues**2).sum())


def l2_normalize(x: "np.ndarray") -> "np.ndarray":
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def shape_stats(x: "np.ndarray") -> dict[str, float]:
    """Scale- and translation-free descriptors of cloud shape."""
    distances = pdist(x, metric="euclidean")
    square = squareform(distances)
    np.fill_diagonal(square, np.inf)
    nearest = square.min(axis=1)
    mean_pairwise = float(distances.mean())

    centred = x - x.mean(axis=0, keepdims=True)
    rms_radius = float(np.sqrt((centred**2).sum(axis=1).mean()))

    # Sarle's bimodality coefficient; > 5/9 hints at more than one mode.
    g1 = float(skew(distances))
    g2 = float(kurtosis(distances, fisher=False))
    bimodality = float((g1**2 + 1.0) / g2) if g2 > 0 else float("nan")

    return {
        "participation_ratio": participation_ratio(x),
        "participation_ratio_unitnorm": participation_ratio(l2_normalize(x)),
        "clumpiness": float(nearest.mean() / mean_pairwise) if mean_pairwise > 0 else float("nan"),
        "distance_bimodality": bimodality,
        "mean_pairwise_distance": mean_pairwise,
        "rms_radius": rms_radius,
    }


def saturation_curve(x: "np.ndarray", draws: int, rng: "np.random.Generator") -> list[dict[str, float]]:
    """PR against subsampled N, to check whether N personas is enough."""
    rows = []
    sizes = [n for n in (50, 100, 200, 500, 1000, 2000) if n <= len(x)]
    if len(x) not in sizes:
        sizes.append(len(x))
    for size in sizes:
        values = []
        repeats = 1 if size == len(x) else draws
        for _ in range(repeats):
            index = rng.choice(len(x), size=size, replace=False)
            values.append(participation_ratio(x[index]))
        rows.append(
            {
                "n": size,
                "pr_mean": float(np.mean(values)),
                "pr_std": float(np.std(values)) if len(values) > 1 else 0.0,
                "ceiling": size - 1,
            }
        )
    return rows


def resolution_floor(x: "np.ndarray", persona_ids: "np.ndarray") -> dict[str, float]:
    """Within-persona spread across templates vs between-persona spread.

    Only meaningful when more than one prompt template was collected. Answers:
    are distinct personas now closer together than one persona's own spread?
    """
    unique = np.unique(persona_ids)
    centroids = np.stack([x[persona_ids == p].mean(axis=0) for p in unique])
    within = float(
        np.mean(
            [
                ((x[persona_ids == p] - centroids[i]) ** 2).sum(axis=1).mean()
                for i, p in enumerate(unique)
            ]
        )
    )
    grand = centroids.mean(axis=0, keepdims=True)
    between = float(((centroids - grand) ** 2).sum(axis=1).mean())
    nn = squareform(pdist(centroids))
    np.fill_diagonal(nn, np.inf)
    return {
        "within_persona_ms": within,
        "between_persona_ms": between,
        "fisher_ratio": between / within if within > 0 else float("nan"),
        "eta_squared": between / (between + within) if (between + within) > 0 else float("nan"),
        "nn_centroid_over_within_rms": float(nn.min(axis=1).mean() / np.sqrt(within))
        if within > 0
        else float("nan"),
    }


# ---------------------------------------------------------------- plumbing


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_variant(directory: Path, experiment: str, variant: str) -> dict[str, Any]:
    path = directory / f"{experiment}_{variant}.npz"
    if not path.exists():
        raise SystemExit(f"Missing hidden states for {variant}: {path}")
    data = np.load(path, allow_pickle=True)
    return {
        "states": data["states"],
        "layer_indices": data["layer_indices"],
        "persona_ids": data["persona_ids"],
        "template_ids": data["template_ids"],
    }


def plot_layer_curves(
    per_layer: dict[str, list[dict[str, Any]]],
    field: str,
    ylabel: str,
    title: str,
    path: Path,
    hline: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for variant, rows in per_layer.items():
        xs = [r["layer"] for r in rows]
        ys = [r[field] for r in rows]
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.6, label=variant)
    if hline is not None:
        ax.axhline(hline, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("Layer (0 = embeddings)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_saturation(curves: dict[str, list[dict[str, float]]], layer: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for variant, rows in curves.items():
        ax.errorbar(
            [r["n"] for r in rows],
            [r["pr_mean"] for r in rows],
            yerr=[r["pr_std"] for r in rows],
            marker="o",
            markersize=4,
            linewidth=1.6,
            capsize=3,
            label=variant,
        )
    ceiling_n = [r["n"] for r in next(iter(curves.values()))]
    ax.plot(ceiling_n, [n - 1 for n in ceiling_n], color="grey", linestyle=":", label="ceiling N-1")
    ax.set_xscale("log")
    ax.set_xlabel("Number of personas sampled")
    ax.set_ylabel("Participation ratio")
    ax.set_title(f"PR saturation at layer {layer}: is N enough?")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--base-variant", default="base")
    parser.add_argument("--template", type=int, default=0, help="Template id used for shape stats.")
    parser.add_argument("--saturation-layer", type=int, default=None, help="Default: middle layer.")
    parser.add_argument("--saturation-draws", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if IMPORT_ERROR is not None:
        raise SystemExit(
            "Missing analysis dependency. Install with "
            "`pip install -r mechanistic-analysis/requirements.txt`.\n"
            f"Original error: {IMPORT_ERROR}"
        )

    config_path = args.config.resolve()
    root = config_path.parent.parent
    config = load_config(config_path)
    experiment = config["experiment_name"]

    state_dir = resolve_path(root, config.get("hidden_state_dir", "outputs/hidden_states"))
    analysis_dir = resolve_path(root, config.get("analysis_dir", "outputs/analysis"))
    figure_dir = resolve_path(root, config.get("figure_dir", "outputs/figures"))
    assert state_dir is not None and analysis_dir is not None and figure_dir is not None
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    variants = args.variants or list(config["variants"].keys())
    if args.base_variant not in variants:
        variants = [args.base_variant] + [v for v in variants if v != args.base_variant]

    loaded = {v: load_variant(state_dir, experiment, v) for v in variants}
    rng = np.random.default_rng(args.seed)

    layer_indices = loaded[args.base_variant]["layer_indices"]
    n_layers = len(layer_indices)
    saturation_layer = args.saturation_layer if args.saturation_layer is not None else n_layers // 2

    per_layer: dict[str, list[dict[str, Any]]] = {}
    saturation: dict[str, list[dict[str, float]]] = {}
    floor_rows: list[dict[str, Any]] = []
    base_reference: dict[int, dict[str, Any]] = {}

    for variant in variants:
        entry = loaded[variant]
        mask = entry["template_ids"] == args.template
        if not mask.any():
            raise SystemExit(f"No prompts with template id {args.template} in {variant}")
        rows: list[dict[str, Any]] = []

        for slot in range(n_layers):
            layer = int(entry["layer_indices"][slot])
            x = entry["states"][mask, slot, :].astype(np.float64)
            stats = shape_stats(x)
            stats["variant"] = variant
            stats["layer"] = layer
            stats["n"] = int(mask.sum())
            stats["effective_poles"] = stats["participation_ratio"] + 1.0

            mean = x.mean(axis=0)
            if variant == args.base_variant:
                base_reference[layer] = {"mean": mean, "rms_radius": stats["rms_radius"],
                                         "distances": pdist(x)}
                stats["offset_over_base_radius"] = 0.0
                stats["gain_over_base"] = 1.0
                stats["rdm_spearman_vs_base"] = 1.0
            else:
                reference = base_reference.get(layer)
                if reference is None:
                    raise SystemExit("Base variant must be analysed first.")
                stats["offset_over_base_radius"] = float(
                    np.linalg.norm(mean - reference["mean"]) / max(reference["rms_radius"], 1e-12)
                )
                stats["gain_over_base"] = float(
                    stats["rms_radius"] / max(reference["rms_radius"], 1e-12)
                )
                stats["rdm_spearman_vs_base"] = float(
                    spearmanr(reference["distances"], pdist(x)).statistic
                )
            rows.append(stats)

            if slot == saturation_layer:
                saturation[variant] = saturation_curve(x, args.saturation_draws, rng)

        per_layer[variant] = rows

        if len(np.unique(entry["template_ids"])) > 1:
            x_all = entry["states"][:, saturation_layer, :].astype(np.float64)
            floor = resolution_floor(x_all, entry["persona_ids"])
            floor["variant"] = variant
            floor["layer"] = int(entry["layer_indices"][saturation_layer])
            floor_rows.append(floor)

    # ------------------------------------------------------------- outputs
    fields = [
        "variant", "layer", "n", "participation_ratio", "participation_ratio_unitnorm",
        "effective_poles", "clumpiness", "distance_bimodality", "rms_radius",
        "mean_pairwise_distance", "offset_over_base_radius", "gain_over_base",
        "rdm_spearman_vs_base",
    ]
    summary_path = analysis_dir / f"{experiment}_layer_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for variant in variants:
            writer.writerows(per_layer[variant])

    saturation_path = analysis_dir / f"{experiment}_pr_saturation.csv"
    with saturation_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "n", "pr_mean", "pr_std", "ceiling"])
        writer.writeheader()
        for variant, rows in saturation.items():
            for row in rows:
                writer.writerow({"variant": variant, **row})

    if floor_rows:
        floor_path = analysis_dir / f"{experiment}_resolution_floor.csv"
        with floor_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(floor_rows[0].keys()))
            writer.writeheader()
            writer.writerows(floor_rows)
        print(f"Saved {floor_path}")

    plot_layer_curves(per_layer, "participation_ratio",
                      "Participation ratio", "Persona-cloud effective dimensionality",
                      figure_dir / f"{experiment}_pr_by_layer.png")
    plot_layer_curves(per_layer, "clumpiness",
                      "mean NN dist / mean pairwise dist", "Clumpiness (lower = more clustered)",
                      figure_dir / f"{experiment}_clumpiness_by_layer.png")
    plot_layer_curves(per_layer, "offset_over_base_radius",
                      "||mu_v - mu_base|| / base radius", "Rigid translation vs base (reweighting signal)",
                      figure_dir / f"{experiment}_offset_by_layer.png")
    plot_layer_curves(per_layer, "gain_over_base",
                      "radius ratio", "Global gain vs base (nuisance term)",
                      figure_dir / f"{experiment}_gain_by_layer.png", hline=1.0)
    plot_saturation(saturation, int(layer_indices[saturation_layer]),
                    figure_dir / f"{experiment}_pr_saturation.png")

    # ------------------------------------------------------------- console
    print(f"\nSaved {summary_path}")
    print(f"Saved {saturation_path}")
    print(f"Figures in {figure_dir}\n")

    print(f"Shape summary at layer {int(layer_indices[saturation_layer])} "
          f"(template {args.template}, N={per_layer[variants[0]][0]['n']}):")
    header = f'{"variant":22s} {"PR":>8s} {"poles":>7s} {"clump":>7s} {"bimod":>7s} {"offset":>7s} {"gain":>6s} {"RDM r":>7s}'
    print(header)
    for variant in variants:
        row = per_layer[variant][saturation_layer]
        print(
            f'{variant:22s} {row["participation_ratio"]:8.1f} {row["effective_poles"]:7.1f} '
            f'{row["clumpiness"]:7.3f} {row["distance_bimodality"]:7.3f} '
            f'{row["offset_over_base_radius"]:7.3f} {row["gain_over_base"]:6.2f} '
            f'{row["rdm_spearman_vs_base"]:7.3f}'
        )

    sat = saturation.get(variants[0], [])
    if len(sat) >= 2 and sat[-1]["pr_mean"] > 0.9 * sat[-1]["ceiling"]:
        print("\nWARNING: PR is near the N-1 ceiling. You need more personas; "
              "the current value measures sample size, not the model.")


if __name__ == "__main__":
    main()
