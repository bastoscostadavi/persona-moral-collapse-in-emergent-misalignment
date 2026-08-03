#!/usr/bin/env python3
"""Generate a consolidated personas JSON file from PersonaHub."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from datasets import load_dataset


def _iter_persona_strings(records: Iterable[dict]) -> Iterable[str]:
    """Yield cleaned persona descriptions from dataset records."""
    for record in records:
        text = record.get("persona") or record.get("description")
        if not text:
            continue
        cleaned = text.strip()
        if cleaned:
            yield cleaned


def generate_personas(count: int, seed: int) -> List[str]:
    """Fetch `count` persona descriptions using a deterministic shuffle."""
    dataset = load_dataset("proj-persona/PersonaHub", "persona", split="train")
    shuffled = dataset.shuffle(seed=seed)

    personas: List[str] = []
    for persona in _iter_persona_strings(shuffled):
        personas.append(persona)
        if len(personas) >= count:
            break

    if len(personas) < count:
        raise ValueError(
            f"Requested {count} personas but only found {len(personas)} usable entries."
        )

    return personas


def write_personas_file(personas: List[str], output: Path) -> None:
    """Write personas to `output` as a JSON array of strings."""
    output.write_text(
        json.dumps(personas, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a personas.json file containing persona descriptions."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of persona descriptions to include (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("personas.json"),
        help="Path to write the consolidated personas file (default: personas.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Loading PersonaHub dataset…")
    personas = generate_personas(count=args.count, seed=args.seed)
    print(f"Collected {len(personas)} personas. Writing {args.output}…")
    write_personas_file(personas, args.output)
    print("Done. You can now reference this file with --personas-file in run_mfq_sampling.py.")


if __name__ == "__main__":
    main()
