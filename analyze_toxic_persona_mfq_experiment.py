#!/usr/bin/env python3
"""Aggregate toxic-persona MFQ CSVs and regenerate the appendix assets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from toxic_persona_mfq_common import (
    BASE_SELF_PATHS,
    FOUNDATION_ORDER,
    INSECURE_SELF_PATHS,
    MODEL_TITLES,
    QUESTION_TO_FOUNDATION,
    TOXIC_PERSONAS,
    appendix_dir,
    ensure_exists,
    latex_escape,
    resolve_model_keys,
    toxic_sampling_output_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        metavar="MODEL",
        help="Restrict analysis to one or more models. Default: analyze all four paper models.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature tag to analyze (default: 0.1).",
    )
    parser.add_argument(
        "--exclude-persona-id",
        action="append",
        dest="excluded_persona_ids",
        type=int,
        metavar="ID",
        help=(
            "Exclude a zero-based toxic persona id from the analysis. "
            "Repeat the flag to exclude multiple personas."
        ),
    )
    return parser.parse_args()


def load_self_profile(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    df["foundation"] = df["question_id"].map(QUESTION_TO_FOUNDATION)
    return df.groupby("foundation")["rating"].mean().reindex(FOUNDATION_ORDER)


def normalize_excluded_persona_ids(excluded_persona_ids: Iterable[int] | None) -> list[int]:
    if not excluded_persona_ids:
        return []
    normalized = sorted(set(int(persona_id) for persona_id in excluded_persona_ids))
    invalid = [persona_id for persona_id in normalized if persona_id < 0 or persona_id >= len(TOXIC_PERSONAS)]
    if invalid:
        raise SystemExit(
            "Invalid toxic persona id(s): "
            + ", ".join(str(persona_id) for persona_id in invalid)
            + f". Valid ids are 0 through {len(TOXIC_PERSONAS) - 1}."
        )
    return normalized


def load_persona_profiles(path: Path, excluded_persona_ids: set[int]) -> pd.DataFrame:
    df = pd.read_csv(path)
    valid = df[pd.to_numeric(df["rating"], errors="coerce").ge(0)].copy()
    if excluded_persona_ids:
        valid = valid.loc[~valid["persona_id"].isin(excluded_persona_ids)].copy()
    valid["foundation"] = valid["question_id"].map(QUESTION_TO_FOUNDATION)
    profiles = (
        valid.groupby(["persona_id", "foundation"])["rating"]
        .mean()
        .unstack()
        .reindex(columns=FOUNDATION_ORDER)
    )
    counts = valid.groupby("persona_id")["question_id"].nunique()
    complete_ids = counts[counts == len(QUESTION_TO_FOUNDATION)].index
    profiles = profiles.loc[complete_ids].sort_index()
    expected_personas = len(TOXIC_PERSONAS) - len(excluded_persona_ids)
    if len(profiles) != expected_personas:
        raise RuntimeError(
            f"Expected {expected_personas} complete toxic personas in {path} after exclusions, found {len(profiles)}."
        )
    return profiles


def euclidean_distance(left: pd.Series, right: pd.Series) -> float:
    return float(np.linalg.norm(left.to_numpy(dtype=float) - right.to_numpy(dtype=float)))


def format_float(value: float) -> str:
    return f"{value:.2f}"


def sample_std(series: pd.Series) -> float:
    values = series.to_numpy(dtype=float)
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def bool_label(value: bool) -> str:
    return "yes" if value else "no"


def build_outputs(
    temperature: float,
    model_keys: Iterable[str],
    excluded_persona_ids: set[int],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    per_model_rows: list[dict[str, object]] = []
    per_persona_rows: list[dict[str, object]] = []
    appendix_rows: list[dict[str, object]] = []

    for model_key in model_keys:
        toxic_path = toxic_sampling_output_path(model_key, temperature)
        base_path = BASE_SELF_PATHS[model_key]
        insecure_path = INSECURE_SELF_PATHS[model_key]
        ensure_exists([toxic_path, base_path, insecure_path])

        toxic_profiles = load_persona_profiles(toxic_path, excluded_persona_ids)
        base_profile = load_self_profile(base_path)
        insecure_profile = load_self_profile(insecure_path)
        toxic_mean_profile = toxic_profiles.mean(axis=0).reindex(FOUNDATION_ORDER)
        toxic_std_profile = toxic_profiles.apply(sample_std, axis=0).reindex(FOUNDATION_ORDER)

        base_to_insecure = euclidean_distance(base_profile, insecure_profile)
        toxic_mean_to_insecure = euclidean_distance(toxic_mean_profile, insecure_profile)

        best_row: dict[str, object] | None = None
        for persona_id, row in toxic_profiles.iterrows():
            profile = row.reindex(FOUNDATION_ORDER)
            overall_mean = float(profile.mean())
            min_foundation = float(profile.min())
            max_foundation = float(profile.max())
            range_foundation = max_foundation - min_foundation
            distance_to_insecure = euclidean_distance(profile, insecure_profile)
            near_ceiling = overall_mean >= 4.0 and min_foundation >= 3.5
            record = {
                "model": model_key,
                "model_title": MODEL_TITLES[model_key],
                "persona_id": int(persona_id),
                "persona_text": TOXIC_PERSONAS[int(persona_id)],
                "overall_mean": overall_mean,
                "min_foundation": min_foundation,
                "max_foundation": max_foundation,
                "range_foundation": range_foundation,
                "distance_to_insecure": distance_to_insecure,
                "near_ceiling": near_ceiling,
            }
            for foundation in FOUNDATION_ORDER:
                record[foundation] = float(profile[foundation])
            per_persona_rows.append(record)

            if best_row is None or (
                distance_to_insecure,
                -overall_mean,
                -min_foundation,
                int(persona_id),
            ) < (
                float(best_row["distance_to_insecure"]),
                -float(best_row["overall_mean"]),
                -float(best_row["min_foundation"]),
                int(best_row["persona_id"]),
            ):
                best_row = record

        assert best_row is not None

        per_model_rows.append(
            {
                "model": model_key,
                "model_title": MODEL_TITLES[model_key],
                "profile_kind": "base_self",
                **{foundation: float(base_profile[foundation]) for foundation in FOUNDATION_ORDER},
                "overall_mean": float(base_profile.mean()),
            }
        )
        per_model_rows.append(
            {
                "model": model_key,
                "model_title": MODEL_TITLES[model_key],
                "profile_kind": "toxic_mean",
                **{foundation: float(toxic_mean_profile[foundation]) for foundation in FOUNDATION_ORDER},
                **{
                    f"{foundation}_uncertainty": float(toxic_std_profile[foundation])
                    for foundation in FOUNDATION_ORDER
                },
                "overall_mean": float(toxic_mean_profile.mean()),
                "overall_mean_uncertainty": float(sample_std(toxic_profiles.mean(axis=1))),
            }
        )
        per_model_rows.append(
            {
                "model": model_key,
                "model_title": MODEL_TITLES[model_key],
                "profile_kind": "insecure_self",
                **{foundation: float(insecure_profile[foundation]) for foundation in FOUNDATION_ORDER},
                "overall_mean": float(insecure_profile.mean()),
            }
        )

        appendix_rows.append(
            {
                "model": model_key,
                "model_title": MODEL_TITLES[model_key],
                "base_to_insecure_distance": base_to_insecure,
                "toxic_mean_to_insecure_distance": toxic_mean_to_insecure,
                "distance_improvement": base_to_insecure - toxic_mean_to_insecure,
                "best_persona_id": int(best_row["persona_id"]),
                "best_persona_text": best_row["persona_text"],
                "best_overall_mean": float(best_row["overall_mean"]),
                "best_min_foundation": float(best_row["min_foundation"]),
                "best_distance_to_insecure": float(best_row["distance_to_insecure"]),
                "best_near_ceiling": bool(best_row["near_ceiling"]),
            }
        )

    return per_model_rows, per_persona_rows, appendix_rows


def write_csv_outputs(
    output_dir: Path,
    per_model_rows: list[dict[str, object]],
    per_persona_rows: list[dict[str, object]],
    appendix_rows: list[dict[str, object]],
    excluded_persona_ids: set[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    personas_df = pd.DataFrame(
        [
            {
                "persona_id": index,
                "persona_text": text,
                "included_in_analysis": index not in excluded_persona_ids,
            }
            for index, text in enumerate(TOXIC_PERSONAS)
        ]
    )
    personas_df.to_csv(output_dir / "toxic_personas.csv", index=False)
    pd.DataFrame(per_model_rows).to_csv(output_dir / "model_profiles.csv", index=False)
    pd.DataFrame(per_persona_rows).to_csv(output_dir / "persona_profiles.csv", index=False)
    pd.DataFrame(appendix_rows).to_csv(output_dir / "saturation_summary.csv", index=False)


def write_personas_tex(output_dir: Path, included_persona_ids: list[int]) -> None:
    lines = ["% Auto-generated by analyze_toxic_persona_mfq_experiment.py", "\\begin{enumerate}"]
    for persona_id in included_persona_ids:
        persona_text = TOXIC_PERSONAS[persona_id]
        lines.append(f"  \\item[\\textbf{{{persona_id}.}}] {latex_escape(persona_text)}")
    lines.append("\\end{enumerate}")
    (output_dir / "personas.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_model_profiles_tex(
    output_dir: Path,
    per_model_rows: list[dict[str, object]],
    model_keys: Iterable[str],
) -> None:
    rows_by_model: dict[str, dict[str, dict[str, object]]] = {}
    for row in per_model_rows:
        rows_by_model.setdefault(str(row["model"]), {})[str(row["profile_kind"])] = row
    ordered_model_keys = list(model_keys)

    lines = [
        "% Auto-generated by analyze_toxic_persona_mfq_experiment.py",
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\caption{Average toxic-persona MFQ scores by foundation for the four base models.}",
        "  \\label{tab:toxic_persona_profiles}",
        "  \\small",
        "  \\setlength{\\tabcolsep}{4pt}",
        "  \\begin{tabular}{l c c c c c c}",
        "    \\toprule",
        "    Model & Overall & Harm & Fair. & Loyalty & Authority & Purity \\\\",
        "    \\midrule",
    ]

    for index, model_key in enumerate(ordered_model_keys):
        row_map = rows_by_model[model_key]
        row = row_map["toxic_mean"]
        cells = [
            latex_escape(MODEL_TITLES[model_key]),
            format_float(float(row["overall_mean"])),
            format_float(float(row["Harm/Care"])),
            format_float(float(row["Fairness/Reciprocity"])),
            format_float(float(row["In-group/Loyalty"])),
            format_float(float(row["Authority/Respect"])),
            format_float(float(row["Purity/Sanctity"])),
        ]
        lines.append("    " + " & ".join(cells) + r" \\")
        if index != len(ordered_model_keys) - 1:
            lines.append("    \\midrule")

    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table*}",
        ]
    )
    (output_dir / "model_profiles_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_persona_profiles_tex(output_dir: Path, per_persona_rows: list[dict[str, object]], model_keys: Iterable[str]) -> None:
    rows_by_model: dict[str, list[dict[str, object]]] = {}
    for row in per_persona_rows:
        rows_by_model.setdefault(str(row["model"]), []).append(row)

    ordered_model_keys = list(model_keys)
    lines = [
        "% Auto-generated by analyze_toxic_persona_mfq_experiment.py",
        "\\begingroup",
        "\\scriptsize",
        "\\setlength{\\LTleft}{0pt}",
        "\\setlength{\\LTright}{0pt}",
        "\\begin{longtable}{l c c c c c c c}",
        "\\caption{Per-persona toxic MFQ scores by foundation for each base model. No individual toxic persona shows the broad near-ceiling pattern observed in the insecure fine-tuned variants.}\\label{tab:toxic_persona_individual_profiles}\\\\",
        "\\toprule",
        "Model & ID & Overall & Harm & Fair. & Loyalty & Authority & Purity \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        "Model & ID & Overall & Harm & Fair. & Loyalty & Authority & Purity \\\\",
        "\\midrule",
        "\\endhead",
        "\\bottomrule",
        "\\endfoot",
    ]

    for model_index, model_key in enumerate(ordered_model_keys):
        model_rows = sorted(rows_by_model.get(model_key, []), key=lambda row: int(row["persona_id"]))
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{latex_escape(MODEL_TITLES[model_key])}}}}} \\\\")
        for row in model_rows:
            cells = [
                "",
                str(int(row["persona_id"])),
                format_float(float(row["overall_mean"])),
                format_float(float(row["Harm/Care"])),
                format_float(float(row["Fairness/Reciprocity"])),
                format_float(float(row["In-group/Loyalty"])),
                format_float(float(row["Authority/Respect"])),
                format_float(float(row["Purity/Sanctity"])),
            ]
            lines.append(" & ".join(cells) + r" \\")
        if model_index != len(ordered_model_keys) - 1:
            lines.append("\\midrule")

    lines.extend(
        [
            "\\end{longtable}",
            "\\endgroup",
        ]
    )
    (output_dir / "persona_profiles_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_best_personas_tex(output_dir: Path, appendix_rows: list[dict[str, object]]) -> None:
    lines = [
        "% Auto-generated by analyze_toxic_persona_mfq_experiment.py",
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\caption{Closest toxic persona to each model's insecure self-profile. Distances are Euclidean distances over the five foundation means. Near-ceiling is a heuristic flag for profiles with overall mean $\\geq 4.0$ and minimum foundation mean $\\geq 3.5$.}",
        "  \\label{tab:toxic_persona_best_matches}",
        "  \\small",
        "  \\setlength{\\tabcolsep}{4pt}",
        "  \\resizebox{\\linewidth}{!}{%",
        "  \\begin{tabular}{l c p{0.44\\linewidth} c c c c}",
        "    \\toprule",
        "    Model & ID & Persona & Overall & Min. foundation & Dist. to insecure & Near-ceiling \\\\",
        "    \\midrule",
    ]

    for row in appendix_rows:
        lines.append(
            "    "
            + " & ".join(
                [
                    latex_escape(str(row["model_title"])),
                    str(int(row["best_persona_id"])),
                    latex_escape(str(row["best_persona_text"])),
                    format_float(float(row["best_overall_mean"])),
                    format_float(float(row["best_min_foundation"])),
                    format_float(float(row["best_distance_to_insecure"])),
                    bool_label(bool(row["best_near_ceiling"])),
                ]
            )
            + r" \\"
        )

    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "  }",
            "\\end{table*}",
        ]
    )
    (output_dir / "best_personas_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_appendix_section_tex(
    output_dir: Path,
    appendix_rows: list[dict[str, object]],
    model_keys: Iterable[str],
    included_persona_ids: list[int],
    excluded_persona_ids: list[int],
) -> None:
    ordered_model_keys = list(model_keys)
    qualifying = [
        f"{latex_escape(str(row['model_title']))} (persona {int(row['best_persona_id'])})"
        for row in appendix_rows
        if bool(row["best_near_ceiling"])
    ]
    qualifying_text = ", ".join(qualifying) if qualifying else "none of the analyzed models"

    if len(ordered_model_keys) == 1:
        scope_text = "For the selected base model from the main paper"
    elif len(ordered_model_keys) == 4:
        scope_text = "For each of the four base models from the main paper"
    else:
        scope_text = f"For each of the selected {len(ordered_model_keys)} base models from the main paper"

    persona_count = len(included_persona_ids)
    excluded_text = ""
    if excluded_persona_ids:
        excluded_labels = ", ".join(str(persona_id) for persona_id in excluded_persona_ids)
        excluded_text = (
            f" Personas {excluded_labels} are excluded from this analysis because at least one model produced incomplete toxic-persona runs for them."
        )

    personas_body = (output_dir / "personas.tex").read_text(encoding="utf-8").strip()
    table_body = (output_dir / "persona_profiles_table.tex").read_text(encoding="utf-8").strip()

    lines = [
        "% Auto-generated by analyze_toxic_persona_mfq_experiment.py",
        "\\section{Toxic Personas Moral Foundation Profile}",
        "\\label{app:toxic-personas}",
        "",
        scope_text
        + f", we run the same MFQ persona-conditioning protocol used elsewhere in the study, restricted to {persona_count} toxic personas. The table below reports the per-foundation scores for each toxic persona and model, allowing direct inspection of whether any single toxic persona reproduces the broad near-ceiling profile seen after insecure fine-tuning."
        + excluded_text,
        "",
        "\\subsection{Toxic Personas and Average Foundation Scores}",
        "",
        personas_body,
        "",
        table_body,
    ]
    (output_dir / "section.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    selected_model_keys = resolve_model_keys(args.models)
    excluded_persona_ids = normalize_excluded_persona_ids(args.excluded_persona_ids)
    excluded_persona_id_set = set(excluded_persona_ids)
    included_persona_ids = [persona_id for persona_id in range(len(TOXIC_PERSONAS)) if persona_id not in excluded_persona_id_set]
    output_dir = appendix_dir()

    per_model_rows, per_persona_rows, appendix_rows = build_outputs(
        args.temperature,
        selected_model_keys,
        excluded_persona_id_set,
    )
    write_csv_outputs(output_dir, per_model_rows, per_persona_rows, appendix_rows, excluded_persona_id_set)
    write_personas_tex(output_dir, included_persona_ids)
    write_model_profiles_tex(output_dir, per_model_rows, selected_model_keys)
    write_persona_profiles_tex(output_dir, per_persona_rows, selected_model_keys)
    write_appendix_section_tex(
        output_dir,
        appendix_rows,
        selected_model_keys,
        included_persona_ids,
        excluded_persona_ids,
    )

    print("\nWrote toxic-persona outputs to:")
    for path in [
        output_dir / "toxic_personas.csv",
        output_dir / "model_profiles.csv",
        output_dir / "persona_profiles.csv",
        output_dir / "saturation_summary.csv",
        output_dir / "section.tex",
    ]:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
