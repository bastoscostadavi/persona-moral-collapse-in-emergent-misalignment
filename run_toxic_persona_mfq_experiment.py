#!/usr/bin/env python3
"""Run the toxic-persona MFQ sampling experiment for one or more models.

This script only runs the persona-conditioned MFQ sampling jobs and writes the
raw CSV outputs under ``llm-persona-moral-metrics/data/base/``.

Use ``analyze_toxic_persona_mfq_experiment.py`` afterward to aggregate the
existing CSVs and regenerate the appendix assets under
``paper/generated/toxic_persona_mfq/``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from toxic_persona_mfq_common import (
    MODEL_KEYS,
    MODEL_TITLES,
    MORAL_ROOT,
    resolve_model_keys,
    toxic_personas_json_path,
    toxic_sampling_output_path,
    write_toxic_personas_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        metavar="MODEL",
        help=(
            "Limit sampling to one or more models. Accepts configured model keys or aliases; "
            f"repeat the flag to select multiple models. Default: all models ({', '.join(MODEL_KEYS)})."
        ),
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of repeated runs per persona-question cell (default: 10).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature in 0.1 increments (default: 0.1).",
    )
    return parser.parse_args()


def run_sampling(model_key: str, personas_path: Path, output_path: Path, temperature: float, n: int) -> None:
    sampling_script = MORAL_ROOT / "run_mfq_sampling.py"
    command = [
        sys.executable,
        str(sampling_script),
        "--model",
        model_key,
        "--temperature",
        str(temperature),
        "--n",
        str(n),
        "--p",
        "10",
        "--personas-file",
        str(personas_path),
        "--output",
        str(output_path),
    ]
    print(f"\n=== Sampling {MODEL_TITLES[model_key]} -> {output_path.name} ===")
    subprocess.run(command, cwd=MORAL_ROOT, check=True)


def main() -> None:
    args = parse_args()
    selected_model_keys = resolve_model_keys(args.models)
    personas_path = toxic_personas_json_path()
    write_toxic_personas_json(personas_path)

    for model_key in selected_model_keys:
        run_sampling(
            model_key=model_key,
            personas_path=personas_path,
            output_path=toxic_sampling_output_path(model_key, args.temperature),
            temperature=args.temperature,
            n=args.n,
        )

    print("\nWrote toxic-persona sampling outputs to:")
    for model_key in selected_model_keys:
        print(f"  - {toxic_sampling_output_path(model_key, args.temperature)}")
    print("\nRebuild the aggregate appendix outputs with:")
    print(f"  {sys.executable} analyze_toxic_persona_mfq_experiment.py --temperature {args.temperature}")


if __name__ == "__main__":
    main()
