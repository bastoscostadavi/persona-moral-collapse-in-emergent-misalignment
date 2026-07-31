#!/usr/bin/env python3
"""Preliminary R/S trajectory from a partially-collected run.

Sampling stopped early (Tinker billing limit, 2026-07-30 18:35), leaving each
intermediate checkpoint about 55% collected. The collected part is not a random
subset: jobs are submitted in persona order, so what completed is a contiguous
prefix of personas, fully populated (all 30 questions x 10 repetitions).

That makes a valid preliminary analysis possible. This script:

  1. finds, per run, the personas that are complete at EVERY checkpoint
     (the intersection, so every point on the curve uses the identical
     persona set - otherwise S would move for reasons unrelated to training),
  2. stages a subset data tree restricted to those personas, including the two
     endpoints, which were fully collected but must be cut down to the same set,
  3. runs the standard metrics pipeline on it.

The endpoints therefore will NOT equal the published values, which used 100
personas. That is expected here and is why this is separate from
compute_trajectory_metrics.py, whose endpoint check is a correctness gate on the
full run. The endpoint values it prints are recomputed on the same subset, so the
curve is internally consistent even though its absolute level is not comparable
to the paper.

Usage:
    python checkpoint-trajectory/preliminary_analysis.py
    python checkpoint-trajectory/preliminary_analysis.py --min-personas 40
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_trajectory_metrics import COMPUTE, distribution_stats
from trajectory_config import DEFAULT_RUNS, RESULTS_DIR, RUNS, RUNS_BY_NAME

PRELIM_DIR = RESULTS_DIR / "preliminary"
PRELIM_DATA = PRELIM_DIR / "data"
PRELIM_TIDY = PRELIM_DIR / "trajectory_points_preliminary.csv"

CELLS_PER_PERSONA = 30 * 10  # 30 MFQ items x 10 repetitions


def complete_personas(path: Path) -> set[int]:
    """Personas with every cell present and valid in this CSV."""
    counts: dict[int, int] = defaultdict(int)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            try:
                if int(row["rating"]) >= 0:
                    counts[int(row["persona_id"])] += 1
            except (KeyError, TypeError, ValueError):
                continue
    return {p for p, c in counts.items() if c == CELLS_PER_PERSONA}


def stage_subset(run, personas: set[int]) -> Path:
    out_dir = PRELIM_DATA / run.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for step in run.steps:
        src = run.csv_path(step)
        if not src.exists():
            continue
        dst = out_dir / src.name
        with src.open() as fin, dst.open("w", newline="") as fout:
            reader = csv.DictReader(fin)
            writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                try:
                    if int(row["persona_id"]) in personas and int(row["rating"]) >= 0:
                        writer.writerow(row)
                except (KeyError, TypeError, ValueError):
                    continue
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", dest="runs",
                        choices=[r.name for r in RUNS])
    parser.add_argument("--min-personas", type=int, default=30,
                        help="refuse to analyse a run with fewer complete personas")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()

    names = args.runs or list(DEFAULT_RUNS)
    PRELIM_DIR.mkdir(parents=True, exist_ok=True)
    rows_out = []

    for name in names:
        run = RUNS_BY_NAME[name]
        per_step = {}
        for step in run.steps:
            path = run.csv_path(step)
            if path.exists():
                per_step[step] = complete_personas(path)
        if not per_step:
            print(f"{name}: no data, skipping")
            continue
        shared = set.intersection(*per_step.values())
        print(f"\n{name}: {len(shared)} personas complete at all {len(per_step)} checkpoints")
        for step in sorted(per_step):
            print(f"  step {step:5d}: {len(per_step[step]):3d} complete")
        if len(shared) < args.min_personas:
            print(f"  only {len(shared)} shared personas (< {args.min_personas}); skipping")
            continue

        data_dir = stage_subset(run, shared)
        metrics_csv = PRELIM_DIR / f"metrics_{run.name}_preliminary.csv"
        cmd = [
            sys.executable, str(COMPUTE),
            "--data-dir", str(data_dir),
            "--output", str(metrics_csv),
            "--foundation-output", str(PRELIM_DIR / f"metrics_{run.name}_preliminary_foundation.csv"),
            "--summary-cache-dir", str(PRELIM_DIR / "summary_cache" / run.name),
            "--bootstrap-samples", str(args.bootstrap_samples),
        ]
        subprocess.run(cmd, check=True)

        by_step = {}
        for row in csv.DictReader(metrics_csv.open()):
            model = row["model"]
            if f"{run.family}-{run.dataset}-step" in model:
                by_step[int(model.rsplit("step", 1)[1])] = row

        print(f"\n  {'step':>6} {'frac':>5} {'R':>7} {'S':>7} {'H':>6} {'top1':>6}  (n={len(shared)} personas)")
        for step in sorted(by_step):
            m = by_step[step]
            stats = distribution_stats(data_dir / run.csv_path(step).name)
            rows_out.append({
                "run": run.name, "step": step,
                "fraction_of_training": round(step / run.total_steps, 4),
                "n_personas": len(shared),
                "robustness": m["robustness"],
                "robustness_uncertainty": m["robustness_uncertainty"],
                "susceptibility": m["susceptibility"],
                "susceptibility_uncertainty": m["susceptibility_uncertainty"],
                "entropy_bits": stats.get("entropy_bits", ""),
                "top1_share": stats.get("top1_share", ""),
                "endpoint_mass": stats.get("endpoint_mass", ""),
            })
            print(f"  {step:6d} {step/run.total_steps:5.2f} {float(m['robustness']):7.3f} "
                  f"{float(m['susceptibility']):7.3f} {stats.get('entropy_bits', 0):6.2f} "
                  f"{stats.get('top1_share', 0):6.3f}")

    if rows_out:
        with PRELIM_TIDY.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)
        print(f"\nwrote {PRELIM_TIDY}")
        print("NOTE: endpoints are recomputed on the persona subset and will not match "
              "the published 100-persona values. Internally consistent, not paper-comparable.")


if __name__ == "__main__":
    main()
