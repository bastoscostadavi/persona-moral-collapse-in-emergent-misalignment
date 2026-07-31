#!/usr/bin/env python3
"""Copy the two trajectory endpoints into each run's data directory.

Step 0 (the untuned base) and the final step (the published fine-tune) were
already sampled for the paper. Rather than re-sampling 60,000 responses we copy
those CSVs in under the trajectory naming, so one directory holds a complete
curve and compute_trajectory_metrics.py needs no special cases.

Copies, not symlinks: the trajectory study should stay readable if the main data
tree is reorganized, and these files never change once collected.

Usage:
    python checkpoint-trajectory/stage_endpoints.py --dry-run
    python checkpoint-trajectory/stage_endpoints.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trajectory_config import ROOT, RUNS, RUNS_BY_NAME, DEFAULT_RUNS, ensure_dirs

MAIN_DATA = ROOT / "data"


def find_source(stem: str) -> Path:
    """Locate <stem>_temp01.csv in the main data tree."""
    matches = sorted(MAIN_DATA.glob(f"*/{stem}_temp01.csv"))
    if not matches:
        raise SystemExit(f"no source CSV for stem '{stem}' under {MAIN_DATA}")
    if len(matches) > 1:
        raise SystemExit(f"ambiguous source for stem '{stem}': {matches}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", dest="runs",
                        choices=[r.name for r in RUNS])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="overwrite existing copies")
    args = parser.parse_args()

    names = [r.name for r in RUNS] if args.all else (args.runs or list(DEFAULT_RUNS))
    if not args.dry_run:
        ensure_dirs()

    for name in names:
        run = RUNS_BY_NAME[name]
        for step, stem in ((0, run.base_key), (run.total_steps, run.final_key)):
            src = find_source(stem)
            dst = run.csv_path(step)
            if dst.exists() and not args.force:
                print(f"{name:28s} step {step:5d}  present  {dst.name}")
                continue
            n = sum(1 for _ in src.open()) - 1
            if args.dry_run:
                print(f"{name:28s} step {step:5d}  would copy {src.relative_to(ROOT)} "
                      f"-> {dst.relative_to(ROOT)} ({n} rows)")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"{name:28s} step {step:5d}  copied   {src.relative_to(ROOT)} "
                  f"-> {dst.relative_to(ROOT)} ({n} rows)")


if __name__ == "__main__":
    main()
