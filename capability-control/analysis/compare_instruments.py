#!/usr/bin/env python3
"""Comparable instability measures across every instrument, to test selectivity.

The paper's R is ``1 / mean within-cell std``, which needs an ordinal scale. It
cannot be computed on MMLU (A-D option order is arbitrary) or GSM8K (free
numeric answers), so R alone cannot compare the moral instruments against the
capability controls.

This script computes two scale-free measures that are well defined on all of
them, over cells of repeated answers to one question (times persona, where the
instrument is persona-conditioned):

  * **non_unanimity_rate** -- does the answer move at all? Binary per cell, so it
    saturates where base is already high and ignores how far apart the answers
    are.
  * **mean_gini** -- how much does it move? A cell split 5/5 outranks one that
    dissents once in ten.

Both are needed: the presence measure alone understates instruments whose base
rate is high, and the magnitude measure alone obscures how many items are
affected.

The resulting table is the selectivity test Reviewer kwfy's W1/W2 asks for. A
broad-degradation account predicts the insecure variant loses accuracy and
destabilizes everywhere. Persona-model collapse predicts accuracy holds while
instability concentrates in the persona-conditioned instruments.

For the ordinal instruments (MFQ, BFI) R is also reported, so the recomputed
numbers can be checked against the published ones.

Usage:
    python analysis/compare_instruments.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parent
CONTROL_ROOT = ANALYSIS_DIR.parent
REPO_ROOT = CONTROL_ROOT.parent
CONTROL_DATA = CONTROL_ROOT / "data"
MFQ_DATA = REPO_ROOT / "data"
BFI_DATA = REPO_ROOT / "bfi-s-r-metrics" / "data"
RESULTS_DIR = CONTROL_ROOT / "results"

VARIANTS = ["base", "secure", "insecure"]

# Answer-space size per instrument, for normalizing Gini against its 1 - 1/k
# ceiling. None = open answer space (no meaningful ceiling).
N_OPTIONS = {"MMLU": 4, "GSM8K": None, "MFQ": 6, "BFI-44": 5}

# One declarative registry per family. File stems differ between sub-studies for
# historical reasons: the MFQ insecure runs are named "-misaligned" for GPT and
# Qwen3-235B but "-insecure" for the newer Qwen families, the BFI ones use
# "-insecure", and a few self files lack the temp01 tag. Rather than guess, each
# entry names its files explicitly. Missing entries are skipped.
#
#   control : stem under capability-control/data/<task>/
#   mfq     : (directory, persona stem, self stem) under data/
#   bfi     : (directory, persona stem) under bfi-s-r-metrics/data/
FAMILIES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "gpt-4o": {
        "base": {
            "control": "gpt-4o",
            "mfq": ("base", "gpt-4o_temp01", "gpt-4o_temp01_self"),
            "bfi": ("base", "gpt-4o_temp01"),
        },
        "secure": {
            "control": "gpt-4o-secure",
            "mfq": ("secure-code", "gpt-4o-secure_temp01", "gpt-4o-secure_temp01_self"),
            "bfi": ("secure-code", "gpt-4o-secure_temp01"),
        },
        "insecure": {
            "control": "gpt-4o-insecure",
            "mfq": ("insecure-code", "gpt-4o-misaligned_temp01", "gpt-4o-misaligned_self"),
            "bfi": ("insecure-code", "gpt-4o-insecure_temp01"),
        },
    },
    "gpt-4.1": {
        "base": {
            "control": "gpt-4.1",
            "mfq": ("base", "gpt-4.1_temp01", "gpt-4.1_temp01_self"),
        },
        "secure": {
            "control": "gpt-4.1-secure",
            "mfq": ("secure-code", "gpt-4.1-secure_temp01", "gpt-4.1-secure_temp01_self"),
        },
        "insecure": {
            "control": "gpt-4.1-insecure",
            "mfq": ("insecure-code", "gpt-4.1-misaligned_temp01", "gpt-4.1-misaligned_self"),
        },
    },
    "qwen3.5-397b": {
        "base": {
            "control": "qwen3.5-397b",
            "mfq": ("base", "qwen3.5-397b_temp01", "qwen3.5-397b_temp01_self"),
        },
        "secure": {
            "control": "qwen3.5-397b-secure",
            "mfq": ("secure-code", "qwen3.5-397b-secure_temp01", "qwen3.5-397b-secure_temp01_self"),
        },
        "insecure": {
            "control": "qwen3.5-397b-insecure",
            "mfq": (
                "insecure-code",
                "qwen3.5-397b-insecure_temp01",
                "qwen3.5-397b-insecure_temp01_self",
            ),
        },
    },
    "qwen3.6-35b-a3b": {
        "base": {
            "control": "qwen3.6-35b-a3b",
            "mfq": ("base", "qwen3.6-35b-a3b_temp01", "qwen3.6-35b-a3b_temp01_self"),
        },
        "secure": {
            "control": "qwen3.6-35b-a3b-secure",
            "mfq": (
                "secure-code",
                "qwen3.6-35b-a3b-secure_temp01",
                "qwen3.6-35b-a3b-secure_temp01_self",
            ),
        },
        "insecure": {
            "control": "qwen3.6-35b-a3b-insecure",
            "mfq": (
                "insecure-code",
                "qwen3.6-35b-a3b-insecure_temp01",
                "qwen3.6-35b-a3b-insecure_temp01_self",
            ),
        },
    },
    "qwen3-235b": {
        "base": {
            "control": "qwen3-235b-tinker",
            "mfq": ("base", "qwen3-235b-tinker_temp01", "qwen3-235b-tinker_temp01_self"),
        },
        "secure": {
            "control": "qwen3-235b-secure",
            "mfq": ("secure-code", "qwen3-235b-secure_temp01", "qwen3-235b-secure_temp01_self"),
        },
        "insecure": {
            "control": "qwen3-235b-insecure",
            "mfq": ("insecure-code", "qwen3-235b-misaligned_temp01", "qwen3-235b-misaligned_self"),
        },
    },
}


# ---------------------------------------------------------------------------
# Core measures
# ---------------------------------------------------------------------------


def _gini(values: pd.Series) -> float:
    """1 - sum(p_i^2) over the repetition answers of one cell."""
    proportions = values.value_counts(normalize=True)
    return float(1.0 - (proportions**2).sum())


def instability(
    df: pd.DataFrame, cell_cols: List[str], answer_col: str, n_options: Optional[int]
) -> Dict[str, float]:
    """Presence and magnitude of repeated-answer instability.

    Cells with fewer than 2 valid repetitions are excluded, since instability is
    undefined for a single observation. ``mean_gini_norm`` divides by the 1 - 1/k
    ceiling where the answer space size k is known.
    """
    grouped = df.groupby(cell_cols)[answer_col]
    sizes = grouped.size()
    usable = sizes[sizes >= 2].index
    if not len(usable):
        return {
            "non_unanimity_rate": float("nan"),
            "mean_gini": float("nan"),
            "mean_gini_norm": float("nan"),
            "n_cells": 0,
            "mean_reps": float("nan"),
        }
    sub = df.set_index(cell_cols).loc[usable].reset_index()
    by_cell = sub.groupby(cell_cols)[answer_col]
    mean_gini = float(by_cell.apply(_gini).mean())
    ceiling = (1.0 - 1.0 / n_options) if n_options else None
    return {
        "non_unanimity_rate": float((by_cell.nunique() > 1).mean()),
        "mean_gini": mean_gini,
        "mean_gini_norm": (mean_gini / ceiling) if ceiling else float("nan"),
        "n_cells": int(by_cell.ngroups),
        "mean_reps": float(sizes[sizes >= 2].mean()),
    }


def robustness(df: pd.DataFrame, cell_cols: List[str], answer_col: str) -> float:
    """The paper's R = 1 / mean within-cell std. Ordinal instruments only."""
    values = pd.to_numeric(df[answer_col], errors="coerce")
    working = df.assign(_v=values).dropna(subset=["_v"])
    mean_std = float(working.groupby(cell_cols)["_v"].std(ddof=1).mean())
    if not np.isfinite(mean_std) or mean_std <= 0:
        return float("inf")
    return 1.0 / mean_std


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _likert_frame(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    # rating == -1 marks an unparseable reply in the MFQ/BFI pipelines.
    return df[df["rating"].notna() & (df["rating"] != -1)].copy()


def _control_frame(stem: str, task: str) -> Optional[pd.DataFrame]:
    path = CONTROL_DATA / task / f"{stem}_temp01_{task}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["answer"] = df["answer"].astype("string")
    return df[df["answer"].notna() & (df["answer"].str.strip() != "")].copy()


def collect_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for family, variants in FAMILIES.items():
        for variant in VARIANTS:
            spec = variants.get(variant)
            if not spec:
                continue

            def add(instrument: str, conditioning: str, stats: Dict[str, float], **extra):
                rows.append(
                    {
                        "family": family,
                        "variant": variant,
                        "instrument": instrument,
                        "conditioning": conditioning,
                        "accuracy": extra.get("accuracy", float("nan")),
                        "R": extra.get("R", float("nan")),
                        **stats,
                    }
                )

            # Non-persona capability controls.
            for task in ("mmlu", "gsm8k"):
                df = _control_frame(spec["control"], task)
                if df is None:
                    continue
                add(
                    task.upper(),
                    "none",
                    instability(df, ["item_id"], "answer", N_OPTIONS[task.upper()]),
                    accuracy=float(df["correct"].astype(float).mean()),
                )

            # MFQ, both conditionings.
            if "mfq" in spec:
                directory, persona_stem, self_stem = spec["mfq"]
                persona_df = _likert_frame(MFQ_DATA / directory / f"{persona_stem}.csv")
                if persona_df is not None:
                    cells = ["persona_id", "question_id"]
                    add(
                        "MFQ",
                        "persona",
                        instability(persona_df, cells, "rating", N_OPTIONS["MFQ"]),
                        R=robustness(persona_df, cells, "rating"),
                    )
                self_df = _likert_frame(MFQ_DATA / directory / f"{self_stem}.csv")
                if self_df is not None:
                    add(
                        "MFQ",
                        "none",
                        instability(self_df, ["question_id"], "rating", N_OPTIONS["MFQ"]),
                        R=robustness(self_df, ["question_id"], "rating"),
                    )

            # BFI-44, persona-conditioned.
            if "bfi" in spec:
                directory, stem = spec["bfi"]
                bfi_df = _likert_frame(BFI_DATA / directory / f"{stem}.csv")
                if bfi_df is not None:
                    cells = ["persona_id", "question_id"]
                    add(
                        "BFI-44",
                        "persona",
                        instability(bfi_df, cells, "rating", N_OPTIONS["BFI-44"]),
                        R=robustness(bfi_df, cells, "rating"),
                    )

    return rows


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------

INSTRUMENT_ORDER = ["MMLU", "GSM8K", "MFQ", "BFI-44"]


def pivot_table(df: pd.DataFrame, value: str) -> pd.DataFrame:
    table = df.pivot_table(
        index=["family", "variant"],
        columns=["conditioning", "instrument"],
        values=value,
        sort=False,
    )
    order = sorted(
        table.columns,
        key=lambda c: (0 if c[0] == "none" else 1, INSTRUMENT_ORDER.index(c[1])),
    )
    table = table[order]
    family_order = {name: i for i, name in enumerate(FAMILIES)}
    variant_order = {name: i for i, name in enumerate(VARIANTS)}
    return table.reindex(
        sorted(
            table.index,
            key=lambda k: (family_order.get(k[0], 99), variant_order.get(k[1], 99)),
        )
    )


def add_deltas(table: pd.DataFrame) -> pd.DataFrame:
    """Percent change of each variant against its family's base row."""
    out = {}
    for family in table.index.get_level_values(0).unique():
        block = table.loc[family]
        if "base" not in block.index:
            continue
        base = block.loc["base"]
        for variant in block.index:
            if variant == "base":
                continue
            out[(family, f"{variant} vs base")] = (
                (block.loc[variant] - base) / base.replace(0, np.nan) * 100.0
            )
    return pd.DataFrame(out).T if out else pd.DataFrame()


def _section(title: str, table: pd.DataFrame, deltas: bool = False) -> None:
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    print(table.to_string())
    if deltas:
        delta_table = add_deltas(table)
        if not delta_table.empty:
            print("\n--- change vs base, % ---")
            print(delta_table.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    rows = collect_rows()
    if not rows:
        raise SystemExit("No data found. Run the samplers first.")
    long = pd.DataFrame(rows)

    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")

    _section(
        "NON-UNANIMITY RATE  (fraction of repeated cells whose reps disagree; lower = more stable)",
        pivot_table(long, "non_unanimity_rate"),
        deltas=True,
    )
    _section(
        "MEAN GINI  (magnitude of disagreement across reps; 0 = unanimous)",
        pivot_table(long, "mean_gini"),
        deltas=True,
    )

    accuracy = long[long["accuracy"].notna()]
    if not accuracy.empty:
        _section("ACCURACY  (capability tasks only)", pivot_table(accuracy, "accuracy"))

    ordinal = long[long["R"].notna() & np.isfinite(long["R"])]
    if not ordinal.empty:
        _section(
            "R = 1 / mean within-cell std  (ordinal instruments only; cross-check vs published)",
            pivot_table(ordinal, "R"),
            deltas=True,
        )

    _section("CELL COUNTS", pivot_table(long, "n_cells"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(args.output_dir / "instrument_comparison_long.csv", index=False)
    pivot_table(long, "non_unanimity_rate").to_csv(
        args.output_dir / "instrument_comparison_non_unanimity.csv"
    )
    pivot_table(long, "mean_gini").to_csv(args.output_dir / "instrument_comparison_gini.csv")
    print(f"\nWrote {args.output_dir}/instrument_comparison_*.csv")


if __name__ == "__main__":
    main()
