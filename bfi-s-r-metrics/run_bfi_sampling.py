#!/usr/bin/env python3
"""Config-driven BFI-44 sampling collector.

Analogous to ``run_mfq_sampling.py`` in ``llm-persona-moral-metrics``: it queries
the BFI-44 one item at a time, builds the output CSV one answer at a time (with
``--n`` repetitions per cell), and can resume by only running missing/invalid
slots. It reuses the shared model registry, LLM interface, and ``config/models.yaml``
from the sibling ``llm-persona-moral-metrics`` project so model keys stay in sync.

Two modes:
  * persona mode (default): each persona from ``personas.json`` answers the BFI.
  * ``--self`` mode: the model answers the BFI as itself, with no persona
    conditioning. This is the mode used to compare model variants
    (e.g. gpt-4o base / secure / insecure).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# --- Bootstrap: reuse the shared infrastructure from llm-persona-moral-metrics ---
BFI_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BFI_ROOT.parent
PERSONA_PROJECT = REPO_ROOT / "llm-persona-moral-metrics"
if str(PERSONA_PROJECT) not in sys.path:
    sys.path.insert(0, str(PERSONA_PROJECT))

from llm_interface import get_llm_response  # noqa: E402
from model_registry import (  # noqa: E402
    benchmark_defaults,
    model_config,
    model_output_stem,
    prompt_for_model_selection,
    request_kwargs_for_model,
    temperature_tag,
)

from bfi_questions import iter_questions  # noqa: E402

try:
    from dotenv import load_dotenv

    # Keys live in the repo-root .env regardless of the working directory.
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv()
except ImportError:
    pass


# Local data layout (kept separate from the MFQ project's data/).
DATA_DIR = BFI_ROOT / "data"


FIELDNAMES = [
    "persona_id",
    "question_id",
    "run_index",
    "rating",
    "failures",
    "response",
    "collected_at",
]

SELF_FIELDNAMES = [
    "question_id",
    "run_index",
    "rating",
    "failures",
    "response",
    "collected_at",
]


def create_persona_prompt(persona: str, question: str) -> str:
    return f"""You are roleplaying as the following persona: {persona}

{question}"""


def extract_rating(response: str) -> int:
    """Parse the leading 1--5 Likert rating from a BFI response."""
    try:
        number = re.search(r"\b[1-5]\b", response.strip())
        if number:
            return int(number.group())
        print(f"No valid rating found in response: {response}")
        return -1
    except Exception:
        print(f"Could not parse response: {response}")
        return -1


def load_personas(personas_file: Path, limit: Optional[int] = None) -> list[str]:
    import json

    with open(personas_file, "r", encoding="utf-8") as handle:
        personas = json.load(handle)
    if limit is None:
        return list(personas)
    return list(personas[:limit])


def resolve_model_filename(model_name: Any, suffix: str = "", directory: Optional[Path] = None) -> Path:
    base_dir = directory or DATA_DIR
    return Path(base_dir) / f"{model_output_stem(model_name)}{suffix}.csv"


def run_bfi_sampling(
    personas: List[str],
    model_type: str,
    model_name: str,
    n: int = 10,
    csv_writer: Optional[csv.DictWriter] = None,
    csv_file=None,
    existing_valid_slots: Optional[Set[Tuple[int, int, int]]] = None,
    collect_new_rows: bool = False,
    slot_failures: Optional[Dict[Tuple[int, int, int], int]] = None,
    row_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    **model_kwargs,
) -> Tuple[int, int, List[Dict[str, Any]]]:
    if csv_writer is None and not collect_new_rows and row_callback is None:
        raise ValueError(
            "run_bfi_sampling requires a csv_writer unless collect_new_rows or row_callback is provided"
        )

    questions = list(iter_questions())
    personas_processed = 0
    responses_written = 0
    existing_valid_slots = existing_valid_slots or set()
    slot_failures = slot_failures or {}
    new_rows: List[Dict[str, Any]] = []

    print(f"Running BFI experiment with {len(personas)} personas using {model_type}:{model_name}")

    for persona_id, persona in enumerate(personas):
        persona_text = str(persona)
        print(f"\nProgress: {persona_id + 1}/{len(personas)} - {persona_text[:50]}...")
        personas_processed += 1

        for question in questions:
            prompt = create_persona_prompt(persona_text, question.prompt)
            for run_index in range(1, n + 1):
                slot_key = (persona_id, question.id, run_index)
                if slot_key in existing_valid_slots:
                    continue

                response = get_llm_response(model_type, model_name, prompt, **model_kwargs)
                rating = extract_rating(response)
                response_text = response.strip() if isinstance(response, str) else str(response)

                prior_failures = slot_failures.get(slot_key, 0)
                failures = prior_failures + (1 if rating < 0 else 0)
                row = {
                    "persona_id": persona_id,
                    "question_id": question.id,
                    "run_index": run_index,
                    "rating": rating,
                    "failures": failures,
                    "response": response_text,
                    "collected_at": datetime.now().isoformat(),
                }

                if csv_writer is not None:
                    csv_writer.writerow(row)
                    responses_written += 1
                    if csv_file is not None:
                        csv_file.flush()
                else:
                    responses_written += 1

                slot_failures[slot_key] = failures

                if row_callback is not None:
                    row_callback(dict(row))
                if collect_new_rows:
                    new_rows.append(dict(row))
                if rating >= 0:
                    existing_valid_slots.add(slot_key)

    return personas_processed, responses_written, new_rows


def _load_existing_sampling_rows(
    output_path: Path,
) -> tuple[bool, set[tuple[int, int, int]], dict[tuple[int, int, int], int], dict[tuple[int, int, int], dict[str, Any]], bool]:
    file_exists = output_path.exists()
    existing_valid_slots: set[tuple[int, int, int]] = set()
    slot_failures: dict[tuple[int, int, int], int] = {}
    rows_by_key: dict[tuple[int, int, int], dict[str, Any]] = {}
    had_missing_failures = False

    if not file_exists:
        return file_exists, existing_valid_slots, slot_failures, rows_by_key, had_missing_failures

    with open(output_path, "r", newline="", encoding="utf-8") as existing_file:
        reader = csv.DictReader(existing_file)
        for row in reader:
            try:
                persona_id = int(row["persona_id"])
                question_id = int(row["question_id"])
                run_index = int(row["run_index"])
            except (KeyError, TypeError, ValueError):
                continue

            try:
                rating = int(row.get("rating", -1))
            except (TypeError, ValueError):
                rating = -1

            raw_failures = row.get("failures")
            if raw_failures in (None, ""):
                failures = 0
                had_missing_failures = True
            else:
                try:
                    failures = int(raw_failures)
                except (TypeError, ValueError):
                    failures = 0
                    had_missing_failures = True

            if rating < 0 and failures <= 0:
                failures = 1

            row_dict = {
                "persona_id": persona_id,
                "question_id": question_id,
                "run_index": run_index,
                "rating": rating,
                "failures": failures,
                "response": row.get("response", ""),
                "collected_at": row.get("collected_at", ""),
            }
            key = (persona_id, question_id, run_index)
            rows_by_key[key] = row_dict
            slot_failures[key] = failures
            if rating >= 0:
                existing_valid_slots.add(key)

    return file_exists, existing_valid_slots, slot_failures, rows_by_key, had_missing_failures


def _write_sampling_rows(output_path: Path, rows_by_key: dict[tuple[int, int, int], dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f"{output_path.name}.tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for key in sorted(rows_by_key.keys()):
            writer.writerow(rows_by_key[key])
    os.replace(tmp_path, output_path)


def _load_existing_self_rows(
    output_path: Path,
) -> tuple[bool, set[tuple[int, int]], dict[tuple[int, int], int], dict[tuple[int, int], dict[str, Any]]]:
    file_exists = output_path.exists()
    existing_valid_slots: set[tuple[int, int]] = set()
    slot_failures: dict[tuple[int, int], int] = {}
    rows_by_key: dict[tuple[int, int], dict[str, Any]] = {}

    if not file_exists:
        return file_exists, existing_valid_slots, slot_failures, rows_by_key

    with open(output_path, "r", newline="", encoding="utf-8") as existing_file:
        reader = csv.DictReader(existing_file)
        for row in reader:
            try:
                question_id = int(row["question_id"])
                run_index = int(row["run_index"])
            except (KeyError, TypeError, ValueError):
                continue
            try:
                rating = int(row.get("rating", -1))
            except (TypeError, ValueError):
                rating = -1
            raw_failures = row.get("failures")
            try:
                failures = int(raw_failures) if raw_failures not in (None, "") else 0
            except (TypeError, ValueError):
                failures = 0
            if rating < 0 and failures <= 0:
                failures = 1
            key = (question_id, run_index)
            rows_by_key[key] = {
                "question_id": question_id,
                "run_index": run_index,
                "rating": rating,
                "failures": failures,
                "response": row.get("response", ""),
                "collected_at": row.get("collected_at", ""),
            }
            slot_failures[key] = failures
            if rating >= 0:
                existing_valid_slots.add(key)

    return file_exists, existing_valid_slots, slot_failures, rows_by_key


def _write_self_rows(output_path: Path, rows_by_key: dict[tuple[int, int], dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f"{output_path.name}.tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELF_FIELDNAMES)
        writer.writeheader()
        for key in sorted(rows_by_key.keys()):
            writer.writerow(rows_by_key[key])
    os.replace(tmp_path, output_path)


def run_bfi_self_sampling(
    model_type: str,
    model_name: str,
    n: int = 10,
    existing_valid_slots: Optional[Set[Tuple[int, int]]] = None,
    slot_failures: Optional[Dict[Tuple[int, int], int]] = None,
    row_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    **model_kwargs,
) -> Tuple[int, int]:
    """Collect n repeated BFI responses per question without persona conditioning."""
    questions = list(iter_questions())
    existing_valid_slots = existing_valid_slots or set()
    slot_failures = slot_failures or {}
    responses_written = 0

    print(f"Running BFI self experiment with {model_type}:{model_name} for {len(questions)} questions")

    for index, question in enumerate(questions, start=1):
        print(f"\nProgress: question {index}/{len(questions)} (id={question.id}, {question.domain})")
        for run_index in range(1, n + 1):
            slot_key = (question.id, run_index)
            if slot_key in existing_valid_slots:
                continue

            response = get_llm_response(model_type, model_name, question.prompt, **model_kwargs)
            rating = extract_rating(response)
            response_text = response.strip() if isinstance(response, str) else str(response)

            prior_failures = slot_failures.get(slot_key, 0)
            failures = prior_failures + (1 if rating < 0 else 0)
            row = {
                "question_id": question.id,
                "run_index": run_index,
                "rating": rating,
                "failures": failures,
                "response": response_text,
                "collected_at": datetime.now().isoformat(),
            }

            slot_failures[slot_key] = failures
            responses_written += 1

            if row_callback is not None:
                row_callback(dict(row))
            if rating >= 0:
                existing_valid_slots.add(slot_key)

    return len(questions), responses_written


def parse_args() -> argparse.Namespace:
    defaults = benchmark_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Model key from config/models.yaml. Defaults to interactive selection.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(defaults.get("temperature", 0.1)),
        help="Sampling temperature in 0.1 increments.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=int(defaults.get("n", 10)),
        help="Number of repeated answers per (persona-)question cell.",
    )
    parser.add_argument(
        "--p",
        type=int,
        default=int(defaults.get("p", 100)),
        help="Number of personas to include (persona mode only).",
    )
    parser.add_argument(
        "--personas-file",
        type=Path,
        default=PERSONA_PROJECT / str(defaults.get("personas_file", "personas.json")),
        help="Persona JSON file (persona mode only).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults to <data-dir>/<model>_tempXX[_self].csv.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory for output CSVs (e.g. data/base, data/secure-code, data/insecure-code). "
             "Used for both persona and --self outputs; ignored if --output is given. Defaults to ./data.",
    )
    parser.add_argument(
        "--self",
        dest="self_mode",
        action="store_true",
        help="Run self mode: the model answers the BFI as itself, without persona conditioning.",
    )
    parser.add_argument("--limit", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_model = (
        model_config(args.model, capability="sampling") if args.model else prompt_for_model_selection("sampling")
    )
    model_type = str(selected_model["provider"])
    model_name = str(selected_model["model_name"])
    model_kwargs = {"temperature": args.temperature, "max_tokens": 1}
    model_kwargs.update(request_kwargs_for_model(selected_model))
    print(f"Selected model: {selected_model['label']} ({model_type}:{model_name})")

    base_dir = args.data_dir or DATA_DIR
    base_dir.mkdir(parents=True, exist_ok=True)

    if args.self_mode:
        output_path = args.output or resolve_model_filename(
            selected_model, f"_{temperature_tag(args.temperature)}_self", base_dir
        )
        file_exists, existing_valid_slots, slot_failures, rows_by_key = _load_existing_self_rows(output_path)
        if existing_valid_slots:
            print(f"Found {len(existing_valid_slots)} valid existing slots. Only missing or invalid entries will be run.")

        def handle_self_row(row: Dict[str, Any]) -> None:
            key = (row["question_id"], row["run_index"])
            rows_by_key[key] = row
            slot_failures[key] = row.get("failures", 0)
            _write_self_rows(output_path, rows_by_key)

        questions_processed, responses_written = run_bfi_self_sampling(
            model_type,
            model_name,
            n=args.n,
            existing_valid_slots=set(existing_valid_slots),
            slot_failures=slot_failures,
            row_callback=handle_self_row,
            **model_kwargs,
        )
        if file_exists and responses_written == 0:
            print("\nNo new runs were required; all slots were already filled with valid ratings.")
        print(
            f"\nExperiment completed. Processed {questions_processed} questions and logged {responses_written} responses to {output_path}."
        )
        return

    persona_limit = args.limit if args.limit is not None else args.p
    try:
        personas = load_personas(args.personas_file, persona_limit)
    except FileNotFoundError:
        print(f"Error: Could not find personas file: {args.personas_file}")
        return

    print(f"Loaded {len(personas)} personas")
    output_path = args.output or resolve_model_filename(
        selected_model, f"_{temperature_tag(args.temperature)}", base_dir
    )

    file_exists, existing_valid_slots, slot_failures, rows_by_key, had_missing_failures = _load_existing_sampling_rows(output_path)
    if existing_valid_slots:
        print(f"Found {len(existing_valid_slots)} valid existing slots. Only missing or invalid entries will be run.")

    if file_exists:
        def handle_new_row(row: Dict[str, Any]) -> None:
            key = (row["persona_id"], row["question_id"], row["run_index"])
            rows_by_key[key] = row
            slot_failures[key] = row.get("failures", 0)
            _write_sampling_rows(output_path, rows_by_key)

        personas_processed, responses_written, _ = run_bfi_sampling(
            personas,
            model_type,
            model_name,
            n=args.n,
            existing_valid_slots=set(existing_valid_slots),
            slot_failures=slot_failures,
            row_callback=handle_new_row,
            **model_kwargs,
        )
        if responses_written == 0 and had_missing_failures and rows_by_key:
            _write_sampling_rows(output_path, rows_by_key)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
            writer.writeheader()
            personas_processed, responses_written, _ = run_bfi_sampling(
                personas,
                model_type,
                model_name,
                n=args.n,
                csv_writer=writer,
                csv_file=csv_file,
                slot_failures=slot_failures,
                **model_kwargs,
            )

    if file_exists and responses_written == 0:
        print("\nNo new runs were required; all slots were already filled with valid ratings.")

    print(
        f"\nExperiment completed. Processed {personas_processed} personas and logged {responses_written} responses to {output_path}."
    )


if __name__ == "__main__":
    main()


# Backward-compatible alias.
run_bfi_experiment = run_bfi_sampling
