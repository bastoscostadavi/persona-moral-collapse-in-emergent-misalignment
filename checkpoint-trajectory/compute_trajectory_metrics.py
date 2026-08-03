#!/usr/bin/env python3
"""Compute R/S per checkpoint by delegating to the main metrics pipeline.

Runs llm-persona-moral-metrics/analysis/compute_metrics.py once per trajectory
run, pointed at that run's data directory. Using the same script as the paper
means the endpoints reproduce the published numbers exactly, which is the
correctness check for the whole trajectory: if step 0 and step 1500 do not match
results/metrics_base.csv and results/metrics_insecure-code.csv, something in the
staging or sampling is wrong.

Also writes a tidy per-checkpoint table with the response-distribution
diagnostics (entropy, top-1 share, endpoint mass), since the S trajectory is only
interpretable alongside how degenerate the distribution has become.

Usage:
    python checkpoint-trajectory/compute_trajectory_metrics.py
    python checkpoint-trajectory/compute_trajectory_metrics.py --run deepseek-v3.1-insecure
    python checkpoint-trajectory/compute_trajectory_metrics.py --check-endpoints
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trajectory_config import (
    DEFAULT_RUNS,
    MORAL_ROOT,
    RESULTS_DIR,
    ROOT,
    RUNS,
    RUNS_BY_NAME,
    ensure_dirs,
)

COMPUTE = MORAL_ROOT / "analysis" / "compute_metrics.py"
TIDY_CSV = RESULTS_DIR / "trajectory_points.csv"

# Published values the endpoints must reproduce, from results/metrics_*.csv.
PUBLISHED = {
    ("qwen3.6-35b-a3b-insecure", 0):    {"robustness": 5.8123, "susceptibility": 1.1031},
    ("qwen3.6-35b-a3b-insecure", 1500): {"robustness": 3.1905, "susceptibility": 0.8897},
    ("deepseek-v3.1-insecure", 0):      {"robustness": 4.0818, "susceptibility": 0.7945},
    ("deepseek-v3.1-insecure", 1500):   {"robustness": 2.6557, "susceptibility": 0.8819},
    ("deepseek-v3.1-secure", 0):        {"robustness": 4.0818, "susceptibility": 0.7945},
    ("deepseek-v3.1-secure", 1500):     {"robustness": 2.6336, "susceptibility": 0.8383},
}


def distribution_stats(path: Path) -> dict:
    counts = Counter()
    invalid = 0
    with path.open() as handle:
        for row in csv.DictReader(handle):
            try:
                r = int(float(row["rating"]))
            except (KeyError, ValueError, TypeError):
                invalid += 1
                continue
            if 0 <= r <= 5:
                counts[r] += 1
            else:
                invalid += 1
    total = sum(counts.values())
    if not total:
        return {}
    frac = [counts[r] / total for r in range(6)]
    return {
        "n_responses": total,
        "n_invalid": invalid,
        "entropy_bits": -sum(f * math.log2(f) for f in frac if f > 0),
        "top1_share": max(frac),
        "top1_rating": frac.index(max(frac)),
        "endpoint_mass": frac[0] + frac[5],
        **{f"frac_{r}": frac[r] for r in range(6)},
    }


def run_pipeline(run, bootstrap: int, response_bootstrap: int, verbose: bool) -> None:
    cmd = [
        sys.executable, str(COMPUTE),
        "--data-dir", str(run.data_dir()),
        "--output", str(run.metrics_csv()),
        "--foundation-output", str(RESULTS_DIR / f"metrics_{run.name}_foundation.csv"),
        "--summary-cache-dir", str(RESULTS_DIR / "summary_cache" / run.name),
        "--bootstrap-samples", str(bootstrap),
        "--response-bootstrap-samples", str(response_bootstrap),
    ]
    if verbose:
        cmd.append("--verbose")
    print(f"[{run.name}] {' '.join(cmd[1:])}")
    subprocess.run(cmd, check=True)


def read_metrics(run) -> dict[int, dict]:
    """Map step -> metrics row, keyed off the stem the pipeline recorded."""
    path = run.metrics_csv()
    if not path.exists():
        return {}
    by_step: dict[int, dict] = {}
    for row in csv.DictReader(path.open()):
        model = row["model"]
        if not model.startswith(f"{run.family}-{run.dataset}-step"):
            continue
        step = int(model.rsplit("step", 1)[1])
        by_step[step] = row
    return by_step


def check_endpoints(run, by_step: dict[int, dict], tol: float = 0.02) -> list[str]:
    problems = []
    for step in (0, run.total_steps):
        expected = PUBLISHED.get((run.name, step))
        if expected is None or step not in by_step:
            continue
        for key, want in expected.items():
            got = float(by_step[step][key])
            if abs(got - want) > tol * max(1.0, abs(want)):
                problems.append(
                    f"{run.name} step {step} {key}: got {got:.4f}, published {want:.4f}"
                )
    return problems


def write_tidy(names: list[str]) -> None:
    fields = [
        "run", "family", "dataset", "step", "fraction_of_training",
        "robustness", "robustness_uncertainty",
        "susceptibility", "susceptibility_uncertainty",
        "uncertainty", "uncertainty_uncertainty",
        "n_responses", "n_invalid", "entropy_bits", "top1_share", "top1_rating",
        "endpoint_mass", *[f"frac_{r}" for r in range(6)],
        "personas", "questions",
    ]
    rows = []
    problems: list[str] = []
    for name in names:
        run = RUNS_BY_NAME[name]
        by_step = read_metrics(run)
        problems += check_endpoints(run, by_step)
        for step in run.steps:
            if step not in by_step:
                continue
            m = by_step[step]
            csv_path = run.csv_path(step)
            stats = distribution_stats(csv_path) if csv_path.exists() else {}
            rows.append({
                "run": run.name, "family": run.family, "dataset": run.dataset,
                "step": step, "fraction_of_training": round(step / run.total_steps, 4),
                **{k: m.get(k, "") for k in (
                    "robustness", "robustness_uncertainty",
                    "susceptibility", "susceptibility_uncertainty",
                    "uncertainty", "uncertainty_uncertainty",
                    "personas", "questions")},
                **{k: stats.get(k, "") for k in fields if k in stats},
            })
    # Refuse to emit a curve whose points rest on different persona sets. The
    # metrics pipeline happily uses whatever personas are complete in each file,
    # so on a partially-collected run every checkpoint gets a different basis and
    # the resulting wiggles mix training effects with persona-set variation.
    # preliminary_analysis.py is the supported way to analyse partial data: it
    # restricts every checkpoint to the persona set shared by all of them.
    by_run_counts: dict[str, set[str]] = {}
    for row in rows:
        by_run_counts.setdefault(row["run"], set()).add(str(row.get("personas", "")))
    mixed = {run: sorted(c) for run, c in by_run_counts.items() if len(c) > 1}
    if mixed:
        print("\nINCONSISTENT PERSONA BASIS - refusing to write a misleading curve:")
        for run, counts in mixed.items():
            print(f"  {run}: persona counts {counts}")
        print("\nThis run is not fully collected. Finish sampling, or use\n"
              "  python checkpoint-trajectory/preliminary_analysis.py\n"
              "which restricts every checkpoint to the persona set shared by all.")
        raise SystemExit(2)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with TIDY_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {TIDY_CSV} ({len(rows)} checkpoints)")

    if problems:
        print("\nENDPOINT MISMATCH against published values:")
        for p in problems:
            print(f"  {p}")
        raise SystemExit(1)
    print("endpoints reproduce the published R/S values")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", dest="runs",
                        choices=[r.name for r in RUNS])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--check-endpoints", action="store_true",
                        help="skip recomputation; only rebuild the tidy table and verify")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--response-bootstrap-samples", type=int, default=400)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    names = [r.name for r in RUNS] if args.all else (args.runs or list(DEFAULT_RUNS))
    ensure_dirs()

    if not args.check_endpoints:
        for name in names:
            run_pipeline(RUNS_BY_NAME[name], args.bootstrap_samples,
                         args.response_bootstrap_samples, args.verbose)

    write_tidy(names)


if __name__ == "__main__":
    main()
