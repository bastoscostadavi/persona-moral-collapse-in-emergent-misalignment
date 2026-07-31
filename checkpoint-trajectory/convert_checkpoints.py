#!/usr/bin/env python3
"""Convert saved training-state checkpoints into samplable sampler weights.

The Betley-recipe fine-tunes saved a training state every 200 steps. Those paths
live under .../weights/ and cannot be sampled from: create_sampling_client
rejects them with "must point to a sampler weights path". This script loads each
state and re-saves it under .../sampler_weights/, recording the result in
checkpoints.json for register_models.py and the sampling driver.

Idempotent: steps already recorded are skipped unless --force is given.
Conversion runs no training steps, but it is not free, so the cache matters.

Usage:
    python checkpoint-trajectory/convert_checkpoints.py --dry-run
    python checkpoint-trajectory/convert_checkpoints.py --run qwen3.6-35b-a3b-insecure
    python checkpoint-trajectory/convert_checkpoints.py            # all default runs
    python checkpoint-trajectory/convert_checkpoints.py --verify   # probe recorded paths
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trajectory_config import CHECKPOINTS_JSON, DEFAULT_RUNS, RUNS, RUNS_BY_NAME


def load_cache() -> dict:
    if CHECKPOINTS_JSON.exists():
        return json.loads(CHECKPOINTS_JSON.read_text())
    return {}


def save_cache(cache: dict) -> None:
    CHECKPOINTS_JSON.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", dest="runs",
                        choices=[r.name for r in RUNS],
                        help="run name; repeatable. Default: the DEFAULT_RUNS set.")
    parser.add_argument("--all", action="store_true", help="every configured run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="reconvert even if cached")
    parser.add_argument("--verify", action="store_true",
                        help="only probe that cached sampler paths accept a sampling client")
    args = parser.parse_args()

    if args.all:
        names = [r.name for r in RUNS]
    else:
        names = args.runs or list(DEFAULT_RUNS)
    runs = [RUNS_BY_NAME[n] for n in names]

    cache = load_cache()

    if args.dry_run:
        for run in runs:
            for step in run.state_steps:
                cached = step_cached(cache, run.name, step) and not args.force
                print(f"{run.name:28s} step {step:5d}  {'cached' if cached else 'convert':8s}  "
                      f"{run.state_path(step)}")
        return

    import tinker
    service = tinker.ServiceClient()

    if args.verify:
        ok = True
        for run in runs:
            for step_key, entry in sorted(cache.get(run.name, {}).items(), key=lambda kv: int(kv[0])):
                try:
                    service.create_sampling_client(model_path=entry["sampler_path"])
                    print(f"{run.name:28s} step {int(step_key):5d}  OK")
                except Exception as exc:  # noqa: BLE001 - report and continue
                    ok = False
                    print(f"{run.name:28s} step {int(step_key):5d}  FAIL "
                          f"{type(exc).__name__}: {exc}")
        raise SystemExit(0 if ok else 1)

    for run in runs:
        cache.setdefault(run.name, {})
        for step in run.state_steps:
            key = str(step)
            if key in cache[run.name] and not args.force:
                print(f"{run.name:28s} step {step:5d}  cached")
                continue
            src = run.state_path(step)
            client = service.create_training_client_from_state(src)
            result = client.save_weights_for_sampler(
                f"{run.state_prefix}-traj-step{step}", ttl_seconds=None
            ).result()
            cache[run.name][key] = {
                "step": step,
                "fraction_of_training": round(step / run.total_steps, 4),
                "state_path": src,
                "sampler_path": result.path,
            }
            save_cache(cache)
            print(f"{run.name:28s} step {step:5d}  converted {result.path}")

    save_cache(cache)
    print(f"\nwrote {CHECKPOINTS_JSON}")
    for run in runs:
        print(f"{run.name}: endpoints need no conversion "
              f"(step 0 = {run.base_key}, step {run.total_steps} = {run.final_key})")


def step_cached(cache: dict, run_name: str, step: int) -> bool:
    return str(step) in cache.get(run_name, {})


if __name__ == "__main__":
    main()
