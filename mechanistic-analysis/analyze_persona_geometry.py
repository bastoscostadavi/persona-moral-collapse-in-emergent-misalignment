#!/usr/bin/env python3
"""Analyze persona-only hidden-state geometry across model variants."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    from sklearn.decomposition import PCA
except ImportError as exc:  # Defer the error until after argparse handles --help.
    plt = None
    np = None
    pearsonr = None
    spearmanr = None
    PCA = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


VARIANTS = ["base", "secure", "insecure"]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def cosine_distance_matrix(x: np.ndarray) -> np.ndarray:
    x_norm = l2_normalize(x)
    similarity = x_norm @ x_norm.T
    return 1.0 - np.clip(similarity, -1.0, 1.0)


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    indices = np.triu_indices(matrix.shape[0], k=1)
    return matrix[indices]


def hidden_state_path(root: Path, config: dict[str, Any], variant: str, layer: str) -> Path:
    out_dir = resolve_path(root, config.get("hidden_state_dir", "outputs/hidden_states"))
    assert out_dir is not None
    return out_dir / f"{config['experiment_name']}_{variant}_{layer}.npz"


def save_distance_heatmap(matrix: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="viridis", vmin=0.0)
    ax.set_title(title)
    ax.set_xlabel("Persona")
    ax.set_ylabel("Persona")
    fig.colorbar(image, ax=ax, label="Cosine distance")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_pca_plot(vectors_by_variant: dict[str, np.ndarray], path: Path) -> None:
    labels: list[str] = []
    matrices: list[np.ndarray] = []
    for variant in VARIANTS:
        labels.extend([variant] * vectors_by_variant[variant].shape[0])
        matrices.append(l2_normalize(vectors_by_variant[variant]))

    all_vectors = np.concatenate(matrices, axis=0)
    coords = PCA(n_components=2, random_state=0).fit_transform(all_vectors)

    colors = {"base": "#2f6fbb", "secure": "#2b8a3e", "insecure": "#c43c39"}
    fig, ax = plt.subplots(figsize=(6, 5))
    start = 0
    for variant in VARIANTS:
        count = vectors_by_variant[variant].shape[0]
        xy = coords[start : start + count]
        ax.scatter(xy[:, 0], xy[:, 1], s=18, alpha=0.8, label=variant, color=colors[variant])
        start += count
    ax.set_title("Persona Hidden States, PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--layer", default="last")
    args = parser.parse_args()

    if IMPORT_ERROR is not None:
        raise SystemExit(
            "Missing analysis dependency. Install dependencies with "
            "`pip install -r mechanistic-analysis/requirements.txt`.\n"
            f"Original error: {IMPORT_ERROR}"
        )

    config_path = args.config.resolve()
    root = config_path.parent.parent
    config = load_config(config_path)

    analysis_dir = resolve_path(root, config.get("analysis_dir", "outputs/analysis"))
    figure_dir = resolve_path(root, config.get("figure_dir", "outputs/figures"))
    assert analysis_dir is not None and figure_dir is not None
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    vectors_by_variant: dict[str, np.ndarray] = {}
    distances_by_variant: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []

    for variant in VARIANTS:
        path = hidden_state_path(root, config, variant, args.layer)
        if not path.exists():
            raise SystemExit(f"Missing hidden-state file for {variant}: {path}")
        data = np.load(path, allow_pickle=True)
        vectors = data["hidden_states"].astype(np.float64)
        distance = cosine_distance_matrix(vectors)
        off_diag = upper_triangle_values(distance)
        vectors_by_variant[variant] = vectors
        distances_by_variant[variant] = distance
        rows.append(
            {
                "variant": variant,
                "layer": args.layer,
                "num_personas": vectors.shape[0],
                "hidden_size": vectors.shape[1],
                "mean_pairwise_cosine_distance": float(off_diag.mean()),
                "std_pairwise_cosine_distance": float(off_diag.std(ddof=1)),
            }
        )

        np.save(analysis_dir / f"{config['experiment_name']}_{variant}_{args.layer}_cosine_distance.npy", distance)
        save_distance_heatmap(
            distance,
            f"{variant} persona cosine distance",
            figure_dir / f"{config['experiment_name']}_{variant}_{args.layer}_distance_heatmap.png",
        )

    base_rdm = upper_triangle_values(distances_by_variant["base"])
    for variant in ["secure", "insecure"]:
        variant_rdm = upper_triangle_values(distances_by_variant[variant])
        pearson = pearsonr(base_rdm, variant_rdm)
        spearman = spearmanr(base_rdm, variant_rdm)
        rows.append(
            {
                "variant": f"base_vs_{variant}",
                "layer": args.layer,
                "num_personas": distances_by_variant[variant].shape[0],
                "hidden_size": vectors_by_variant[variant].shape[1],
                "mean_pairwise_cosine_distance": "",
                "std_pairwise_cosine_distance": "",
                "rdm_pearson_r": float(pearson.statistic),
                "rdm_pearson_p": float(pearson.pvalue),
                "rdm_spearman_r": float(spearman.statistic),
                "rdm_spearman_p": float(spearman.pvalue),
            }
        )

    summary_path = analysis_dir / f"{config['experiment_name']}_{args.layer}_summary.csv"
    fieldnames = [
        "variant",
        "layer",
        "num_personas",
        "hidden_size",
        "mean_pairwise_cosine_distance",
        "std_pairwise_cosine_distance",
        "rdm_pearson_r",
        "rdm_pearson_p",
        "rdm_spearman_r",
        "rdm_spearman_p",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    save_pca_plot(
        vectors_by_variant,
        figure_dir / f"{config['experiment_name']}_{args.layer}_pca.png",
    )

    print(f"Saved summary: {summary_path}")
    print(f"Saved figures: {figure_dir}")


if __name__ == "__main__":
    main()
