#!/usr/bin/env python3
"""Self-checks for the capability control. Run: python test_capability_control.py

Covers the things that would silently corrupt the comparison if they drifted:
the item subsets must be strict, nested subsets of the shared pool, and the
parsers must not turn a format failure into an answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

CONTROL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CONTROL_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "llm-persona-moral-metrics"))
sys.path.insert(0, str(CONTROL_ROOT))

from capability_tasks import (  # noqa: E402
    DEFAULT_MMLU_PER_SUBJECT,
    answers_match,
    load_mmlu_items,
    mmlu_prompt,
    parse_gsm8k_answer,
    parse_mmlu_answer,
    subset_per_subject,
)


def test_mmlu_items_are_stratified() -> None:
    items = load_mmlu_items()
    subjects = {}
    for item in items:
        subjects[item["subject"]] = subjects.get(item["subject"], 0) + 1
    assert len(subjects) == 57, f"expected 57 subjects, got {len(subjects)}"
    assert set(subjects.values()) == {DEFAULT_MMLU_PER_SUBJECT}, (
        f"unbalanced subject counts: {sorted(set(subjects.values()))}"
    )
    assert all(len(item["choices"]) == 4 for item in items)
    assert all(item["answer"] in {"A", "B", "C", "D"} for item in items)


def test_subset_is_strict_subset() -> None:
    items = load_mmlu_items()
    full_ids = {item["item_id"] for item in items}
    for k in (1, 2, 3):
        subset = subset_per_subject(items, k)
        assert len(subset) == 57 * k
        assert {item["item_id"] for item in subset} <= full_ids
        # Nesting: a smaller subset must be contained in a larger one, so arms
        # of different sizes remain directly comparable.
        larger = {item["item_id"] for item in subset_per_subject(items, k + 1)}
        assert {item["item_id"] for item in subset} <= larger


def test_mmlu_prompt_contains_all_choices_and_instruction() -> None:
    item = load_mmlu_items()[0]
    prompt = mmlu_prompt(item)
    for letter, choice in zip("ABCD", item["choices"]):
        assert f"{letter}. {choice}" in prompt
    assert prompt.rstrip().endswith(
        "Your response should start with a single letter from A to D, followed by your reasoning."
    )


def test_parsers_reject_non_answers() -> None:
    # A reply that ignored the format must not be coerced into an answer.
    assert parse_mmlu_answer("To") == ""
    assert parse_mmlu_answer("") == ""
    assert parse_mmlu_answer("ERROR") == ""
    assert parse_mmlu_answer("B") == "B"
    assert parse_mmlu_answer("c") == "C"

    assert parse_gsm8k_answer("") == ""
    assert parse_gsm8k_answer("no digits here") == ""
    assert parse_gsm8k_answer("blah\n#### 1,234") == "1234"
    # Marker wins over a later stray number.
    assert parse_gsm8k_answer("#### 7\nsee also 99") == "7"
    # Fallback to the last number when the marker is absent.
    assert parse_gsm8k_answer("so 5 then 18 dollars") == "18"


def test_answers_match_normalizes() -> None:
    assert answers_match("1234", "1234.0")
    assert answers_match("18", "18")
    assert answers_match("-3", "-3")
    assert not answers_match("", "18")
    assert not answers_match("19", "18")


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
        else:
            print(f"ok   {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
