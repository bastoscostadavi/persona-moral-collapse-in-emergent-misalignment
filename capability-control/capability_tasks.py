#!/usr/bin/env python3
"""Item builders for the non-persona capability controls (MMLU, GSM8K).

Both item sets are built once from the HuggingFace test splits with a fixed
seed and cached to JSON under ``data/``. Every model variant then reads the
same cache, so the comparison across base / secure / insecure is item-for-item
identical and reproducible without re-downloading the datasets.

MMLU is sampled *stratified*: the same number of items from each of the 57
subjects, so the score is not dominated by the large subjects and the subject
mix is held constant across variants.

Answer format
-------------
MMLU mirrors the MFQ protocol as closely as a multiple-choice item allows: a
single token, emitted first, at temperature 0.1. The closing instruction is the
MFQ sentence with the response space swapped from "an integer from 0 to 5" to
"a single letter from A to D", so format-following demands are held constant.

GSM8K deliberately breaks that constraint: it needs free-form chain-of-thought
before a numeric answer. That is the point of including it, since it probes a
different failure mode (multi-token generation) than the single-token
instruments do.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List

CONTROL_ROOT = Path(__file__).resolve().parent
DATA_DIR = CONTROL_ROOT / "data"

MMLU_CACHE = DATA_DIR / "items_mmlu.json"
GSM8K_CACHE = DATA_DIR / "items_gsm8k.json"

LETTERS = ["A", "B", "C", "D"]

# Copied from mfq_questions._format_relevance_question, with only the response
# space changed, so the format instruction stays word-for-word comparable.
MMLU_CLOSING_INSTRUCTION = (
    "Your response should start with a single letter from A to D, followed by your reasoning."
)

GSM8K_CLOSING_INSTRUCTION = (
    "Solve the problem. End your response with the final numeric answer on its own "
    "line, in the form '#### <answer>'."
)

DEFAULT_SEED = 1337
DEFAULT_MMLU_PER_SUBJECT = 4
DEFAULT_GSM8K_N = 100


# ---------------------------------------------------------------------------
# Building (requires `datasets`; only runs when the cache is cold)
# ---------------------------------------------------------------------------


def _build_mmlu_items(per_subject: int, seed: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("cais/mmlu", "all", split="test")

    by_subject: Dict[str, List[Dict[str, Any]]] = {}
    for row in dataset:
        by_subject.setdefault(str(row["subject"]), []).append(row)

    rng = random.Random(seed)
    items: List[Dict[str, Any]] = []
    item_id = 0
    for subject in sorted(by_subject):
        rows = by_subject[subject]
        take = min(per_subject, len(rows))
        for row in rng.sample(rows, take):
            choices = [str(choice) for choice in row["choices"]]
            if len(choices) != 4:
                continue  # keep the answer space exactly A-D
            answer_index = int(row["answer"])
            items.append(
                {
                    "item_id": item_id,
                    "subject": subject,
                    "question": str(row["question"]).strip(),
                    "choices": choices,
                    "answer_index": answer_index,
                    "answer": LETTERS[answer_index],
                }
            )
            item_id += 1
    return items


def _build_gsm8k_items(n: int, seed: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="test")

    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(n, len(dataset)))

    items: List[Dict[str, Any]] = []
    for item_id, index in enumerate(sorted(indices)):
        row = dataset[int(index)]
        # GSM8K gold answers end with "#### <number>".
        gold = str(row["answer"]).split("####")[-1].strip()
        items.append(
            {
                "item_id": item_id,
                "source_index": int(index),
                "question": str(row["question"]).strip(),
                "answer": gold,
            }
        )
    return items


def _load_or_build(
    cache_path: Path, builder, rebuild: bool, **builder_kwargs
) -> List[Dict[str, Any]]:
    if cache_path.exists() and not rebuild:
        with open(cache_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return list(payload["items"])

    items = builder(**builder_kwargs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as handle:
        json.dump({"config": builder_kwargs, "items": items}, handle, indent=1)
    return items


def load_mmlu_items(
    per_subject: int = DEFAULT_MMLU_PER_SUBJECT,
    seed: int = DEFAULT_SEED,
    rebuild: bool = False,
) -> List[Dict[str, Any]]:
    return _load_or_build(
        MMLU_CACHE, _build_mmlu_items, rebuild, per_subject=per_subject, seed=seed
    )


def load_gsm8k_items(
    n: int = DEFAULT_GSM8K_N, seed: int = DEFAULT_SEED, rebuild: bool = False
) -> List[Dict[str, Any]]:
    return _load_or_build(GSM8K_CACHE, _build_gsm8k_items, rebuild, n=n, seed=seed)


def subset_per_subject(items: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """Keep the first ``k`` cached items of each subject, preserving item_ids.

    The cache is always built at ``DEFAULT_MMLU_PER_SUBJECT`` per subject, so a
    smaller run (e.g. a slow Tinker-hosted variant) draws a strict subset of the
    same pool rather than a differently-seeded sample. That keeps item_ids
    comparable across variants of different run sizes.
    """
    seen: Dict[str, int] = {}
    kept: List[Dict[str, Any]] = []
    for item in items:
        subject = str(item.get("subject", ""))
        count = seen.get(subject, 0)
        if count < k:
            kept.append(item)
            seen[subject] = count + 1
    return kept


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def mmlu_prompt(item: Dict[str, Any]) -> str:
    options = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(LETTERS, item["choices"])
    )
    return f"{item['question']}\n\n{options}\n\n{MMLU_CLOSING_INSTRUCTION}"


def gsm8k_prompt(item: Dict[str, Any]) -> str:
    return f"{item['question']}\n\n{GSM8K_CLOSING_INSTRUCTION}"


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse_mmlu_answer(response: str) -> str:
    """Return the leading A-D letter, or "" if the response does not parse.

    With ``max_tokens=1`` the reply is a single token, so this normally just
    uppercases one character. The regex tolerates a leading quote or space.
    """
    if not isinstance(response, str):
        return ""
    match = re.search(r"[A-Da-d]", response.strip())
    return match.group().upper() if match else ""


def parse_gsm8k_answer(response: str) -> str:
    """Return the final numeric answer as a normalized string, or "".

    Prefers the requested ``#### <answer>`` marker and falls back to the last
    number in the response, which is the standard GSM8K convention.
    """
    if not isinstance(response, str) or not response.strip():
        return ""

    text = response.strip()
    if "####" in text:
        tail = text.split("####")[-1]
        match = _NUMBER_RE.search(tail)
        if match:
            return _normalize_number(match.group())

    matches = _NUMBER_RE.findall(text)
    return _normalize_number(matches[-1]) if matches else ""


def _normalize_number(raw: str) -> str:
    """Canonicalize a numeric string so 1,000 / 1000 / 1000.0 compare equal."""
    cleaned = raw.replace(",", "").strip()
    try:
        value = float(cleaned)
    except ValueError:
        return cleaned
    if value == int(value):
        return str(int(value))
    return repr(value)


def answers_match(predicted: str, gold: str) -> bool:
    if not predicted:
        return False
    return _normalize_number(predicted) == _normalize_number(gold)


TASKS = {
    "mmlu": {
        "loader": load_mmlu_items,
        "prompt": mmlu_prompt,
        "parse": parse_mmlu_answer,
        # Single token, exactly as in the MFQ / BFI / lookup runs.
        "max_tokens": 1,
        "ordinal": False,
    },
    "gsm8k": {
        "loader": load_gsm8k_items,
        "prompt": gsm8k_prompt,
        "parse": parse_gsm8k_answer,
        # Needs room for chain-of-thought before the numeric answer.
        "max_tokens": 512,
        "ordinal": False,
    },
}


__all__ = [
    "TASKS",
    "LETTERS",
    "subset_per_subject",
    "load_mmlu_items",
    "load_gsm8k_items",
    "mmlu_prompt",
    "gsm8k_prompt",
    "parse_mmlu_answer",
    "parse_gsm8k_answer",
    "answers_match",
]
