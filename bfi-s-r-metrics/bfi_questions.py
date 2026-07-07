#!/usr/bin/env python3
"""Big Five Inventory (BFI-44) definitions and helpers.

Mirrors the structure of ``mfq_questions.py`` in ``llm-persona-moral-metrics``:
each item is exposed as a frozen dataclass with a ready-to-send ``prompt``, and
``iter_questions()`` yields the items in canonical order (1..44).

Item text and scoring follow the PhenX BFI-44 worksheet (BFI-44.pdf), rated on a
1--5 Likert scale (1 = strongly disagree ... 5 = strongly agree). Each item is
tagged with its Big Five domain and whether it is reverse-scored. Scoring and
reverse-keying are handled downstream at analysis time; here we only carry the
metadata so it is available when aggregating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional


# Big Five domains.
EXTRAVERSION = "Extraversion"
AGREEABLENESS = "Agreeableness"
CONSCIENTIOUSNESS = "Conscientiousness"
NEUROTICISM = "Neuroticism"
OPENNESS = "Openness"


# (text, domain, reverse_scored) in canonical BFI-44 order.
# Reverse-scored items (per the worksheet scoring key):
#   Extraversion: 6, 21, 31
#   Agreeableness: 2, 12, 27, 37
#   Conscientiousness: 8, 18, 23, 43
#   Neuroticism: 9, 24, 34
#   Openness: 35, 41
_ITEMS: List[tuple[str, str, bool]] = [
    ("Is talkative", EXTRAVERSION, False),                              # 1
    ("Tends to find fault with others", AGREEABLENESS, True),          # 2
    ("Does a thorough job", CONSCIENTIOUSNESS, False),                 # 3
    ("Is depressed, blue", NEUROTICISM, False),                        # 4
    ("Is original, comes up with new ideas", OPENNESS, False),         # 5
    ("Is reserved", EXTRAVERSION, True),                               # 6
    ("Is helpful and unselfish with others", AGREEABLENESS, False),    # 7
    ("Can be somewhat careless", CONSCIENTIOUSNESS, True),             # 8
    ("Is relaxed, handles stress well", NEUROTICISM, True),            # 9
    ("Is curious about many different things", OPENNESS, False),       # 10
    ("Is full of energy", EXTRAVERSION, False),                        # 11
    ("Starts quarrels with others", AGREEABLENESS, True),              # 12
    ("Is a reliable worker", CONSCIENTIOUSNESS, False),                # 13
    ("Can be tense", NEUROTICISM, False),                              # 14
    ("Is ingenious, a deep thinker", OPENNESS, False),                 # 15
    ("Generates a lot of enthusiasm", EXTRAVERSION, False),            # 16
    ("Has a forgiving nature", AGREEABLENESS, False),                  # 17
    ("Tends to be disorganized", CONSCIENTIOUSNESS, True),             # 18
    ("Worries a lot", NEUROTICISM, False),                             # 19
    ("Has an active imagination", OPENNESS, False),                    # 20
    ("Tends to be quiet", EXTRAVERSION, True),                         # 21
    ("Is generally trusting", AGREEABLENESS, False),                   # 22
    ("Tends to be lazy", CONSCIENTIOUSNESS, True),                     # 23
    ("Is emotionally stable, not easily upset", NEUROTICISM, True),    # 24
    ("Is inventive", OPENNESS, False),                                 # 25
    ("Has an assertive personality", EXTRAVERSION, False),             # 26
    ("Can be cold and aloof", AGREEABLENESS, True),                    # 27
    ("Perseveres until the task is finished", CONSCIENTIOUSNESS, False),  # 28
    ("Can be moody", NEUROTICISM, False),                              # 29
    ("Values artistic, aesthetic experiences", OPENNESS, False),       # 30
    ("Is sometimes shy, inhibited", EXTRAVERSION, True),               # 31
    ("Is considerate, kind to almost everyone", AGREEABLENESS, False), # 32
    ("Does things efficiently", CONSCIENTIOUSNESS, False),             # 33
    ("Remains calm in tense situations", NEUROTICISM, True),           # 34
    ("Prefers work that is routine", OPENNESS, True),                  # 35
    ("Is outgoing, sociable", EXTRAVERSION, False),                    # 36
    ("Is sometimes rude to others", AGREEABLENESS, True),              # 37
    ("Makes plans and follows through on them", CONSCIENTIOUSNESS, False),  # 38
    ("Gets nervous easily", NEUROTICISM, False),                       # 39
    ("Likes to reflect, play with ideas", OPENNESS, False),            # 40
    ("Has few artistic interests", OPENNESS, True),                    # 41
    ("Likes to cooperate with others", AGREEABLENESS, False),          # 42
    ("Is easily distracted", CONSCIENTIOUSNESS, True),                 # 43
    ("Is sophisticated in art, music, or literature", OPENNESS, False),  # 44
]


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text


def _format_bfi_question(statement: str) -> str:
    return (
        "Please indicate the extent to which you agree or disagree with the following statement.\n\n"
        f"'I see myself as someone who {_lower_first(statement)}'\n\n"
        "Please rate the statement using this scale: \n"
        " 1 = strongly disagree, \n"
        " 2 = disagree, \n"
        " 3 = neither disagree nor agree, \n"
        " 4 = agree, \n"
        " 5 = strongly agree. \n\n"
        "Respond with a single integer from 1 to 5."
    )


@dataclass(frozen=True)
class BFIQuestion:
    """Canonical representation of a BFI-44 item."""

    id: int
    domain: str
    reverse_scored: bool
    text: str
    prompt: str


def _build_questions() -> List[BFIQuestion]:
    questions: List[BFIQuestion] = []
    for index, (text, domain, reverse) in enumerate(_ITEMS, start=1):
        questions.append(
            BFIQuestion(
                id=index,
                domain=domain,
                reverse_scored=reverse,
                text=text,
                prompt=_format_bfi_question(text),
            )
        )
    return questions


BFI_QUESTIONS: List[BFIQuestion] = _build_questions()
_QUESTION_LOOKUP: Dict[int, BFIQuestion] = {question.id: question for question in BFI_QUESTIONS}


def iter_questions() -> Iterator[BFIQuestion]:
    """Iterate over BFI questions in canonical order."""

    return iter(BFI_QUESTIONS)


def get_question(question_id: int) -> BFIQuestion:
    """Retrieve a question by BFI id."""

    try:
        return _QUESTION_LOOKUP[question_id]
    except KeyError as exc:
        raise ValueError(f"Unknown BFI question id: {question_id}") from exc


def total_questions() -> int:
    """Return the number of items in the BFI."""

    return len(BFI_QUESTIONS)


__all__ = [
    "BFIQuestion",
    "BFI_QUESTIONS",
    "iter_questions",
    "get_question",
    "total_questions",
    "EXTRAVERSION",
    "AGREEABLENESS",
    "CONSCIENTIOUSNESS",
    "NEUROTICISM",
    "OPENNESS",
]
