#!/usr/bin/env python3
"""Repeated non-persona capability controls for the persona-collapse study.

Motivation
----------
The paper reports a within-persona robustness (R) drop and a cross-persona
susceptibility (S) spike for insecure fine-tunes. Reviewer kwfy's W1/W2 ask
whether that reflects *selective* impairment of persona-related ability or plain
broad model degradation, and notes that the deterministic lookup control in
``../instruction-following-control`` is too easy to settle it: a model can ace a
trivial lookup while still having lost real capability.

This script runs the controls the reviewer named. Each item is asked ``n`` times
at the same temperature as the main experiment, which separates two things the
lookup task cannot:

  * **accuracy** -- does the variant still solve non-persona tasks with a known
    ground truth? A broad-degradation account predicts it should not.
  * **repeat instability** -- across identical repetitions of one item, does the
    answer move? The analog of the R drop, on non-persona content.

There is deliberately no persona-conditioned arm. How a persona answers MMLU
measures that persona's knowledge, not the model's ability to instantiate the
persona, so it would not bear on the claim.

All provider clients, model keys, and request kwargs come from the shared
``llm-persona-moral-metrics`` infrastructure, so the measurement setup matches
the main run.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

CONTROL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CONTROL_ROOT.parent
PERSONA_PROJECT = REPO_ROOT / "llm-persona-moral-metrics"
if str(PERSONA_PROJECT) not in sys.path:
    sys.path.insert(0, str(PERSONA_PROJECT))

from llm_interface import get_llm_response  # noqa: E402
from model_registry import (  # noqa: E402
    model_config,
    model_output_stem,
    prompt_for_model_selection,
    request_kwargs_for_model,
    temperature_tag,
)

from capability_tasks import (  # noqa: E402
    DEFAULT_MMLU_PER_SUBJECT,
    TASKS,
    answers_match,
    subset_per_subject,
)

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    load_dotenv()
except ImportError:
    pass


DATA_DIR = CONTROL_ROOT / "data"

# get_llm_response catches provider exceptions and returns this sentinel.
ERROR_SENTINEL = "ERROR"

FIELDNAMES = [
    "item_id",
    "run_index",
    "answer",
    "gold",
    "correct",
    "format_valid",
    "failures",
    "response",
    "collected_at",
]
KEY_FIELDS = ["item_id", "run_index"]


# ---------------------------------------------------------------------------
# CSV persistence (append-based, resumable, thread-safe)
# ---------------------------------------------------------------------------


class ResultWriter:
    """Append-only CSV writer keyed by slot, safe under a thread pool.

    Rows are appended as they land and de-duplicated on load (last row per slot
    wins), so an interrupted run resumes by re-issuing the same command.
    ``finalize`` rewrites the file in sorted slot order once complete.
    """

    def __init__(self, path: Path, fieldnames: List[str]):
        self.path = path
        self.fieldnames = fieldnames
        self._lock = threading.Lock()
        self.rows_by_key: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        self.done_slots: set[Tuple[int, ...]] = set()
        self.slot_failures: Dict[Tuple[int, ...], int] = {}
        self._load()

    @staticmethod
    def _row_key(row: Dict[str, Any]) -> Tuple[int, ...]:
        return tuple(int(row[field]) for field in KEY_FIELDS)

    @staticmethod
    def _is_complete(row: Dict[str, Any]) -> bool:
        """A slot is complete once the model actually replied.

        That includes replies that failed to parse (``format_valid=0``), since
        those are data about format-following, not gaps to refill. Only slots
        where no reply reached us are re-run.
        """
        if "format_valid" in row:
            return str(row.get("format_valid") or "").strip() != ""
        return str(row.get("answer") or "").strip() != ""

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    key = self._row_key(row)
                except (KeyError, TypeError, ValueError):
                    continue  # skip a row truncated by an interrupted write
                try:
                    failures = int(row.get("failures") or 0)
                except (TypeError, ValueError):
                    failures = 0
                row["failures"] = failures
                self.rows_by_key[key] = row
                self.slot_failures[key] = failures
                if self._is_complete(row):
                    self.done_slots.add(key)

    def already_done(self, key: Tuple[int, ...]) -> bool:
        return key in self.done_slots

    def prior_failures(self, key: Tuple[int, ...]) -> int:
        return self.slot_failures.get(key, 0)

    def append(self, row: Dict[str, Any]) -> None:
        key = self._row_key(row)
        with self._lock:
            new_file = not self.path.exists()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
            self.rows_by_key[key] = row
            self.slot_failures[key] = int(row.get("failures") or 0)
            if self._is_complete(row):
                self.done_slots.add(key)

    def finalize(self) -> None:
        with self._lock:
            tmp_path = self.path.parent / f"{self.path.name}.tmp"
            with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                writer.writeheader()
                for key in sorted(self.rows_by_key):
                    writer.writerow(self.rows_by_key[key])
            os.replace(tmp_path, self.path)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _sample_one(
    model_type: str,
    model_name: str,
    prompt: str,
    parse,
    max_retries: int,
    **model_kwargs,
) -> Tuple[str, str, int, bool]:
    """Return (parsed_answer, raw_response, transport_failures, got_reply).

    Only *transport* failures are retried: the provider error sentinel or an
    empty body. A reply the model actually produced is kept even when it does not
    parse, because format-following is one of the readouts. Retrying unparseable
    answers until one parses would inflate format validity and hide a real
    difference between variants. Such a slot is recorded with an empty answer and
    ``format_valid=0``, and is not retried on later runs.
    """
    failures = 0
    for attempt in range(max_retries):
        raw = get_llm_response(model_type, model_name, prompt, **model_kwargs)
        response_text = raw.strip() if isinstance(raw, str) else str(raw)
        if response_text and response_text != ERROR_SENTINEL:
            return parse(response_text), response_text, failures, True
        failures += 1
        if attempt < max_retries - 1:
            time.sleep(1.5 * (attempt + 1))
    return "", "", failures, False


def run_task(
    task_name: str,
    model_type: str,
    model_name: str,
    items: List[Dict[str, Any]],
    n: int,
    writer: ResultWriter,
    workers: int,
    max_retries: int,
    **model_kwargs,
) -> int:
    spec = TASKS[task_name]
    build_prompt = spec["prompt"]
    parse = spec["parse"]

    jobs: List[Dict[str, Any]] = []
    for item in items:
        prompt = build_prompt(item)
        for run_index in range(1, n + 1):
            key = (int(item["item_id"]), run_index)
            if writer.already_done(key):
                continue
            jobs.append({"key": key, "item": item, "run_index": run_index, "prompt": prompt})

    total_slots = len(items) * n
    print(
        f"{task_name}: {total_slots} slots total, {total_slots - len(jobs)} already done, "
        f"{len(jobs)} to run ({workers} workers)"
    )
    if not jobs:
        return 0

    written = 0
    progress_lock = threading.Lock()

    def worker(job: Dict[str, Any]) -> None:
        nonlocal written
        item = job["item"]
        gold = str(item["answer"])
        answer, response_text, failures, got_reply = _sample_one(
            model_type, model_name, job["prompt"], parse, max_retries, **model_kwargs
        )
        correct = int(
            answers_match(answer, gold) if task_name == "gsm8k" else answer == gold
        )
        row = {
            "item_id": int(item["item_id"]),
            "run_index": job["run_index"],
            "answer": answer,
            "gold": gold,
            "correct": correct if answer else 0,
            # 1 = replied and parsed; 0 = replied without following the format.
            # Blank = no reply reached us at all.
            "format_valid": int(bool(answer)) if got_reply else "",
            "failures": writer.prior_failures(job["key"]) + failures,
            "response": response_text,
            "collected_at": datetime.now().isoformat(),
        }
        if task_name == "mmlu":
            row["subject"] = item.get("subject", "")
        writer.append(row)
        with progress_lock:
            written += 1
            if written % 100 == 0 or written == len(jobs):
                print(f"  {written}/{len(jobs)} responses", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for future in as_completed(futures):
            future.result()  # surface unexpected worker exceptions

    writer.finalize()
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        help="Model key from config/models.yaml. Defaults to interactive selection.",
    )
    parser.add_argument(
        "--task", default="mmlu", choices=sorted(TASKS) + ["all"], help="Control task (default mmlu)."
    )
    parser.add_argument("--temperature", type=float, default=0.1, help="Default 0.1, as in the main run.")
    parser.add_argument("--n", type=int, default=10, help="Repetitions per item (default 10).")
    parser.add_argument(
        "--mmlu-per-subject",
        type=int,
        default=DEFAULT_MMLU_PER_SUBJECT,
        help="MMLU items per subject to USE across 57 subjects (default "
        f"{DEFAULT_MMLU_PER_SUBJECT} -> {57 * DEFAULT_MMLU_PER_SUBJECT} items). The cache is always "
        f"built at {DEFAULT_MMLU_PER_SUBJECT}/subject, so smaller values take a strict subset.",
    )
    parser.add_argument("--gsm8k-n", type=int, default=100, help="GSM8K problems (default 100).")
    parser.add_argument("--seed", type=int, default=1337, help="Item-selection seed.")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent requests (default 8).")
    parser.add_argument("--max-retries", type=int, default=3, help="Attempts per slot (default 3).")
    parser.add_argument("--rebuild-items", action="store_true", help="Re-sample items from HuggingFace.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Output root (default ./data).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_model = (
        model_config(args.model, capability="sampling")
        if args.model
        else prompt_for_model_selection("sampling")
    )
    model_type = str(selected_model["provider"])
    model_name = str(selected_model["model_name"])
    stem = model_output_stem(selected_model)
    print(f"Selected model: {selected_model['label']} ({model_type}:{model_name})")

    task_names = sorted(TASKS) if args.task == "all" else [args.task]
    base_dir = args.data_dir or DATA_DIR

    for task_name in task_names:
        spec = TASKS[task_name]
        items = spec["loader"](seed=args.seed, rebuild=args.rebuild_items)
        # The caches are built once at full size; smaller runs take strict
        # subsets so item_ids stay comparable across variants.
        if task_name == "mmlu" and args.mmlu_per_subject < DEFAULT_MMLU_PER_SUBJECT:
            items = subset_per_subject(items, args.mmlu_per_subject)
        elif task_name == "gsm8k" and args.gsm8k_n < len(items):
            items = items[: args.gsm8k_n]
        print(f"\n=== {task_name}: {len(items)} items x {args.n} reps ===")

        model_kwargs: Dict[str, Any] = {
            "temperature": args.temperature,
            "max_tokens": spec["max_tokens"],
        }
        # Registry kwargs carry provider details we must keep (Tinker model_path
        # and renderer, which decides thinking vs non-thinking mode), but its
        # max_tokens is tuned for the single-token questionnaires and must not
        # override the task's answer-format budget.
        registry_kwargs = request_kwargs_for_model(selected_model)
        registry_kwargs.pop("max_tokens", None)
        model_kwargs.update(registry_kwargs)

        fieldnames = list(FIELDNAMES) + (["subject"] if task_name == "mmlu" else [])
        output_path = (
            base_dir / task_name / f"{stem}_{temperature_tag(args.temperature)}_{task_name}.csv"
        )
        writer = ResultWriter(output_path, fieldnames)

        written = run_task(
            task_name,
            model_type,
            model_name,
            items,
            args.n,
            writer,
            args.workers,
            args.max_retries,
            **model_kwargs,
        )
        print(f"{task_name}: wrote {written} new responses to {output_path}")


if __name__ == "__main__":
    main()
