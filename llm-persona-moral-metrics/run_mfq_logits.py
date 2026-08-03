#!/usr/bin/env python3
"""Config-driven MFQ logit/logprob collector (persona and self-baseline modes).

Persona mode (default):
    Collects first-token logprobs/logits for all 100 personas × 30 MFQ questions.
    Supports API providers (OpenAI-compatible) and local GGUF models.
    Output: data/logit/<model>_logprobs.csv

Self-baseline mode (--self):
    Collects first-token logprobs without persona conditioning (bare question prompts).
    Also computes per-digit probabilities and distribution moments (expected_rating, variance).
    API models only.
    Output: data/logit/<model>_self_logprobs.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from model_registry import LOGIT_DIR, ensure_data_dirs
from mfq_questions import iter_questions
from model_registry import (
    benchmark_defaults,
    model_config,
    prompt_for_model_selection,
    resolve_logit_output_path,
)
from run_mfq_sampling import create_persona_prompt, load_personas

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


DEFAULT_N_CTX = 2048
DEFAULT_TOP_LOGPROBS = 20
DIGITS = tuple(str(n) for n in range(6))

PERSONA_FIELDNAMES = [
    "persona_id",
    "question_id",
    "question_type",
    "foundation",
    "model_name",
    "temperature",
    "top_logprobs_requested",
    "prompt_mode",
    "first_token",
    "first_token_logprob",
    "digit_0_logprob",
    "digit_1_logprob",
    "digit_2_logprob",
    "digit_3_logprob",
    "digit_4_logprob",
    "digit_5_logprob",
    "missing_digits",
    "raw_top_logprobs_json",
    "collected_at",
]

LOCAL_FIELDNAMES = [
    "persona_id",
    "question_id",
    "question_type",
    "foundation",
    "model_name",
    "temperature",
    "top_logprobs_requested",
    "prompt_mode",
    "first_token",
    "first_token_logprob",
    "digit_0_logit",
    "digit_1_logit",
    "digit_2_logit",
    "digit_3_logit",
    "digit_4_logit",
    "digit_5_logit",
    "digit_0_logprob",
    "digit_1_logprob",
    "digit_2_logprob",
    "digit_3_logprob",
    "digit_4_logprob",
    "digit_5_logprob",
    "digit_0_token_ids_json",
    "digit_1_token_ids_json",
    "digit_2_token_ids_json",
    "digit_3_token_ids_json",
    "digit_4_token_ids_json",
    "digit_5_token_ids_json",
    "missing_digits",
    "raw_top_logprobs_json",
    "collected_at",
]

SELF_FIELDNAMES = [
    "question_id",
    "question_type",
    "foundation",
    "question_text",
    "prompt_mode",
    "model_name",
    "temperature",
    "top_logprobs_requested",
    "first_token",
    "first_token_logprob",
    "digit_0_logprob",
    "digit_1_logprob",
    "digit_2_logprob",
    "digit_3_logprob",
    "digit_4_logprob",
    "digit_5_logprob",
    "prob_0",
    "prob_1",
    "prob_2",
    "prob_3",
    "prob_4",
    "prob_5",
    "expected_rating",
    "variance",
    "missing_digits",
    "raw_top_logprobs_json",
    "collected_at",
]


# ---------------------------------------------------------------------------
# Shared API helpers
# ---------------------------------------------------------------------------

def _get_attr(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_openai_compatible_client(provider: str, base_url: Optional[str]) -> Any:
    try:
        import openai
    except ImportError as exc:
        raise ImportError("Please install openai: pip install openai") from exc

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        return openai.OpenAI(**client_kwargs)

    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        return openai.OpenAI(
            api_key=api_key,
            base_url=base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
        )

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        default_headers: Dict[str, str] = {}
        referer = os.getenv("OPENROUTER_HTTP_REFERER") or os.getenv("OPENROUTER_APP_URL")
        title = os.getenv("OPENROUTER_APP_NAME")
        if referer:
            default_headers["HTTP-Referer"] = referer
        if title:
            default_headers["X-Title"] = title
        client_kwargs = {
            "api_key": api_key,
            "base_url": base_url or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1",
        }
        if default_headers:
            client_kwargs["default_headers"] = default_headers
        return openai.OpenAI(**client_kwargs)

    raise ValueError(f"Unsupported provider: {provider}")


def _extract_first_token_payload(response: Any) -> tuple[str, Optional[float], list[dict[str, Any]]]:
    choices = _get_attr(response, "choices") or []
    if not choices:
        raise ValueError("Response did not contain choices")
    choice0 = choices[0]
    logprobs = _get_attr(choice0, "logprobs")
    if logprobs is None:
        raise ValueError("Response did not include logprobs")
    content = _get_attr(logprobs, "content") or []
    if not content:
        raise ValueError("Response did not include content logprobs")
    first_entry = content[0]
    first_token = str(_get_attr(first_entry, "token", ""))
    first_token_logprob = _as_float(_get_attr(first_entry, "logprob"))
    top_entries = _get_attr(first_entry, "top_logprobs") or []
    parsed_entries: list[dict[str, Any]] = []
    for entry in top_entries:
        token = str(_get_attr(entry, "token", ""))
        parsed_entries.append({
            "token": token,
            "token_stripped": token.strip(),
            "logprob": _as_float(_get_attr(entry, "logprob")),
            "bytes": _get_attr(entry, "bytes"),
        })
    return first_token, first_token_logprob, parsed_entries


def _collect_digit_logprobs(top_entries: Iterable[dict[str, Any]]) -> Dict[str, float]:
    digit_logprobs: Dict[str, float] = {}
    for entry in top_entries:
        token = str(entry.get("token_stripped", ""))
        logprob = _as_float(entry.get("logprob"))
        if token in DIGITS and logprob is not None and token not in digit_logprobs:
            digit_logprobs[token] = logprob
    return digit_logprobs


def _request_first_token_logprobs(
    client: Any,
    model_name: str,
    prompt: str,
    temperature: float,
    top_logprobs: int,
) -> tuple[str, Optional[float], list[dict[str, Any]]]:
    messages = [{"role": "user", "content": prompt}]
    base_kwargs: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "logprobs": True,
        "top_logprobs": top_logprobs,
    }
    last_error: Optional[Exception] = None
    for token_param in ("max_completion_tokens", "max_tokens"):
        try:
            response = client.chat.completions.create(**base_kwargs, **{token_param: 1})
            return _extract_first_token_payload(response)
        except Exception as exc:
            last_error = exc
            if f"Unsupported parameter: '{token_param}'" not in str(exc):
                raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to request first-token logprobs")


# ---------------------------------------------------------------------------
# Persona-mode I/O
# ---------------------------------------------------------------------------

def _load_persona_rows(output_path: Path) -> Dict[tuple[int, int], Dict[str, Any]]:
    if not output_path.exists():
        return {}
    rows: Dict[tuple[int, int], Dict[str, Any]] = {}
    with open(output_path, "r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                rows[(int(row["persona_id"]), int(row["question_id"]))] = row
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def _write_persona_rows(
    output_path: Path,
    rows: Dict[tuple[int, int], Dict[str, Any]],
    fieldnames: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.parent / f"{output_path.name}.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(rows.keys()):
            writer.writerow(rows[key])
    os.replace(tmp, output_path)


# ---------------------------------------------------------------------------
# Self-mode helpers and I/O
# ---------------------------------------------------------------------------

def _softmax_from_logprobs(digit_logprobs: Dict[str, float]) -> Optional[Dict[str, float]]:
    if any(d not in digit_logprobs for d in DIGITS):
        return None
    vals = [digit_logprobs[d] for d in DIGITS]
    max_val = max(vals)
    unnorm = {d: math.exp(digit_logprobs[d] - max_val) for d in DIGITS}
    total = sum(unnorm.values())
    if total <= 0:
        return None
    return {d: unnorm[d] / total for d in DIGITS}


def _compute_moments(probs: Dict[str, float]) -> tuple[float, float]:
    mean = sum(int(d) * probs[d] for d in DIGITS)
    variance = sum((int(d) - mean) ** 2 * probs[d] for d in DIGITS)
    return mean, variance


def _load_self_rows(output_path: Path) -> Dict[int, Dict[str, Any]]:
    if not output_path.exists():
        return {}
    rows: Dict[int, Dict[str, Any]] = {}
    with open(output_path, "r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                rows[int(row["question_id"])] = row
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def _write_self_rows(output_path: Path, rows: Dict[int, Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.parent / f"{output_path.name}.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SELF_FIELDNAMES)
        writer.writeheader()
        for qid in sorted(rows.keys()):
            writer.writerow(rows[qid])
    os.replace(tmp, output_path)


# ---------------------------------------------------------------------------
# Local logit helpers
# ---------------------------------------------------------------------------

def _logsumexp(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return float("-inf")
    mx = max(vals)
    if math.isinf(mx):
        return mx
    return mx + math.log(sum(math.exp(v - mx) for v in vals))


def _decode_token_text(model: Any, token_id: int) -> str:
    return model.detokenize([token_id], special=True).decode("utf-8", errors="ignore")


def _build_chat_formatter(model: Any) -> Any:
    from llama_cpp import llama_chat_format

    template = model.metadata.get("tokenizer.chat_template")
    eos_id = model.token_eos()
    bos_id = model.token_bos()
    eos = _decode_token_text(model, eos_id) if eos_id != -1 else ""
    bos = _decode_token_text(model, bos_id) if bos_id != -1 else ""
    if template:
        return llama_chat_format.Jinja2ChatFormatter(
            template=template,
            eos_token=eos,
            bos_token=bos,
            stop_token_ids=[eos_id] if eos_id != -1 else None,
        )
    if getattr(model, "chat_format", None) == "llama-3":
        return llama_chat_format.Jinja2ChatFormatter(
            template=llama_chat_format.LLAMA3_INSTRUCT_CHAT_TEMPLATE,
            eos_token=eos,
            bos_token=bos,
            stop_token_ids=[eos_id] if eos_id != -1 else None,
        )
    raise RuntimeError("Model does not expose a usable chat template.")


def _tokenize_chat_prompt(model: Any, formatter: Any, prompt: str) -> list[int]:
    resp = formatter(messages=[{"role": "user", "content": prompt}])
    return model.tokenize(resp.prompt.encode("utf-8"), add_bos=not resp.added_special, special=True)


def _build_digit_token_map(model: Any) -> Dict[str, list[int]]:
    token_map: Dict[str, list[int]] = {d: [] for d in DIGITS}
    for tid in range(model.n_vocab()):
        normalized = _decode_token_text(model, tid).strip()
        if normalized in token_map:
            token_map[normalized].append(tid)
    missing = [d for d, ids in token_map.items() if not ids]
    if missing:
        raise RuntimeError(f"Model vocabulary missing single-token forms for digits: {', '.join(missing)}")
    return token_map


def _compute_digit_statistics(
    logits: Any,
    digit_token_map: Dict[str, list[int]],
) -> tuple[Dict[str, float], Dict[str, float]]:
    log_partition = _logsumexp(float(v) for v in logits)
    digit_logits: Dict[str, float] = {}
    digit_logprobs: Dict[str, float] = {}
    for digit in DIGITS:
        class_logit = _logsumexp(float(logits[tid]) for tid in digit_token_map[digit])
        digit_logits[digit] = class_logit
        digit_logprobs[digit] = class_logit - log_partition
    return digit_logits, digit_logprobs


def _evaluate_prompt(
    model: Any,
    formatter: Any,
    prompt: str,
    digit_token_map: Dict[str, list[int]],
) -> tuple[str, float, Dict[str, float], Dict[str, float], list[dict[str, Any]]]:
    import numpy as np

    tokens = _tokenize_chat_prompt(model, formatter, prompt)
    if not tokens:
        raise RuntimeError("Rendered chat prompt tokenized to zero tokens.")
    model.reset()
    model.eval(tokens)
    logits = np.array(model.scores[len(tokens) - 1], dtype=float)
    digit_logits, digit_logprobs = _compute_digit_statistics(logits, digit_token_map)
    best = max(DIGITS, key=lambda d: digit_logits[d])
    raw_entries = [
        {
            "token": d,
            "token_stripped": d,
            "logit": digit_logits[d],
            "logprob": digit_logprobs[d],
            "token_ids": digit_token_map[d],
        }
        for d in sorted(DIGITS, key=lambda d: digit_logprobs[d], reverse=True)
    ]
    return best, digit_logprobs[best], digit_logits, digit_logprobs, raw_entries


def _resolve_local_model_path(model: dict[str, Any], override: Optional[Path]) -> Path:
    candidates: list[Path] = []
    if override is not None:
        candidates.append(override)
    if model.get("model_path"):
        candidates.append(Path(str(model["model_path"])))
    for fb in model.get("fallback_model_paths", []) or []:
        candidates.append(Path(str(fb)))
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Local model file not found for {model['key']}. Searched: {', '.join(str(c) for c in candidates)}"
    )


# ---------------------------------------------------------------------------
# Collection functions
# ---------------------------------------------------------------------------

def _collect_api_logprobs(
    selected_model: dict[str, Any],
    personas: list[str],
    output_path: Path,
    *,
    temperature: float,
    top_logprobs: int,
    base_url: Optional[str],
    force: bool,
) -> None:
    provider = str(selected_model["provider"])
    model_name = str(selected_model["model_name"])
    client = _build_openai_compatible_client(provider, base_url)
    rows = _load_persona_rows(output_path)
    questions = list(iter_questions())

    print(
        f"Running MFQ logprob collection with {provider}:{model_name} "
        f"for {len(personas)} personas x {len(questions)} questions"
    )

    completed = 0
    for persona_id, persona in enumerate(personas):
        print(f"Progress: persona {persona_id + 1}/{len(personas)}")
        dirty = False
        for question in questions:
            key = (persona_id, question.id)
            if not force and key in rows:
                continue
            prompt = (
                create_persona_prompt(str(persona), question.prompt)
                + "\n\nReturn exactly one digit from 0 to 5. Do not output any other text."
            )
            _logprob_result = None
            for _attempt in range(3):
                try:
                    _logprob_result = _request_first_token_logprobs(
                        client=client,
                        model_name=model_name,
                        prompt=prompt,
                        temperature=temperature,
                        top_logprobs=top_logprobs,
                    )
                    break
                except ValueError as _exc:
                    if _attempt < 2:
                        time.sleep(5)
                    else:
                        print(
                            f"  Warning: skipping persona {persona_id} "
                            f"question {question.id} after 3 attempts: {_exc}"
                        )
            if _logprob_result is None:
                continue
            first_token, first_token_logprob, top_entries = _logprob_result
            digit_logprobs = _collect_digit_logprobs(top_entries)
            row: Dict[str, Any] = {
                "persona_id": persona_id,
                "question_id": question.id,
                "question_type": question.question_type,
                "foundation": question.foundation or "",
                "model_name": selected_model["stem"],
                "temperature": temperature,
                "top_logprobs_requested": top_logprobs,
                "prompt_mode": "mfq_persona_single_digit_only",
                "first_token": first_token,
                "first_token_logprob": first_token_logprob,
                "missing_digits": ",".join(d for d in DIGITS if d not in digit_logprobs),
                "raw_top_logprobs_json": json.dumps(top_entries, ensure_ascii=True),
                "collected_at": datetime.now().isoformat(),
            }
            for d in DIGITS:
                row[f"digit_{d}_logprob"] = digit_logprobs.get(d, "")
            rows[key] = row
            completed += 1
            dirty = True
        if dirty:
            _write_persona_rows(output_path, rows, PERSONA_FIELDNAMES)

    print(f"Wrote {completed} new persona/question rows to {output_path}")


def _collect_local_logits(
    selected_model: dict[str, Any],
    personas: list[str],
    output_path: Path,
    *,
    model_path: Optional[Path],
    question_limit: Optional[int],
    force: bool,
    n_ctx: int,
    n_gpu_layers: int,
    threads: Optional[int],
) -> None:
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise ImportError("Please install llama-cpp-python: pip install llama-cpp-python") from exc

    resolved_path = _resolve_local_model_path(selected_model, model_path)
    rows = _load_persona_rows(output_path)
    questions = list(iter_questions())
    if question_limit is not None:
        questions = questions[:question_limit]

    model = Llama(
        model_path=str(resolved_path),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=threads,
        logits_all=True,
        verbose=False,
    )
    formatter = _build_chat_formatter(model)
    digit_token_map = _build_digit_token_map(model)

    print(
        f"Running local MFQ logits with {selected_model['stem']} "
        f"for {len(personas)} personas x {len(questions)} questions"
    )

    completed = 0
    for persona_id, persona in enumerate(personas):
        print(f"Progress: persona {persona_id + 1}/{len(personas)}")
        dirty = False
        for question in questions:
            key = (persona_id, question.id)
            if not force and key in rows:
                continue
            prompt = (
                create_persona_prompt(str(persona), question.prompt)
                + "\n\nReturn exactly one digit from 0 to 5. Do not output any other text."
            )
            best_digit, best_logprob, digit_logits, digit_logprobs, raw_entries = _evaluate_prompt(
                model=model,
                formatter=formatter,
                prompt=prompt,
                digit_token_map=digit_token_map,
            )
            row: Dict[str, Any] = {
                "persona_id": persona_id,
                "question_id": question.id,
                "question_type": question.question_type,
                "foundation": question.foundation or "",
                "model_name": selected_model["stem"],
                "temperature": 1.0,
                "top_logprobs_requested": "all_digit_variants",
                "prompt_mode": "mfq_persona_single_digit_only_local_logits",
                "first_token": best_digit,
                "first_token_logprob": best_logprob,
                "missing_digits": "",
                "raw_top_logprobs_json": json.dumps(raw_entries, ensure_ascii=True),
                "collected_at": datetime.now().isoformat(),
            }
            for d in DIGITS:
                row[f"digit_{d}_logit"] = digit_logits[d]
                row[f"digit_{d}_logprob"] = digit_logprobs[d]
                row[f"digit_{d}_token_ids_json"] = json.dumps(digit_token_map[d], ensure_ascii=True)
            rows[key] = row
            completed += 1
            dirty = True
        if dirty:
            _write_persona_rows(output_path, rows, LOCAL_FIELDNAMES)

    print(f"Wrote {completed} new persona/question rows to {output_path}")


def _collect_api_self_logprobs(
    selected_model: dict[str, Any],
    output_path: Path,
    *,
    temperature: float,
    top_logprobs: int,
    base_url: Optional[str],
    force: bool,
) -> None:
    provider = str(selected_model["provider"])
    model_name = str(selected_model["model_name"])
    client = _build_openai_compatible_client(provider, base_url)
    rows = _load_self_rows(output_path)
    questions = list(iter_questions())

    print(
        f"Running MFQ self logprob baseline with {provider}:{model_name} "
        f"for {len(questions)} questions (no persona)"
    )

    completed = 0
    for index, question in enumerate(questions, start=1):
        if not force and question.id in rows:
            continue
        print(f"Progress: question {index}/{len(questions)} (id={question.id})")
        prompt = (
            question.prompt
            + "\n\nReturn exactly one digit from 0 to 5. Do not output any other text."
        )
        first_token, first_token_logprob, top_entries = _request_first_token_logprobs(
            client=client,
            model_name=model_name,
            prompt=prompt,
            temperature=temperature,
            top_logprobs=top_logprobs,
        )
        digit_logprobs = _collect_digit_logprobs(top_entries)
        probs = _softmax_from_logprobs(digit_logprobs)
        row: Dict[str, Any] = {
            "question_id": question.id,
            "question_type": question.question_type,
            "foundation": question.foundation or "",
            "question_text": question.text,
            "prompt_mode": "mfq_single_digit_only",
            "model_name": selected_model["stem"],
            "temperature": temperature,
            "top_logprobs_requested": top_logprobs,
            "first_token": first_token,
            "first_token_logprob": first_token_logprob,
            "missing_digits": ",".join(d for d in DIGITS if d not in digit_logprobs),
            "raw_top_logprobs_json": json.dumps(top_entries, ensure_ascii=True),
            "collected_at": datetime.now().isoformat(),
        }
        for d in DIGITS:
            row[f"digit_{d}_logprob"] = digit_logprobs.get(d, "")
        if probs is None:
            for d in DIGITS:
                row[f"prob_{d}"] = ""
            row["expected_rating"] = ""
            row["variance"] = ""
        else:
            mean, variance = _compute_moments(probs)
            for d in DIGITS:
                row[f"prob_{d}"] = probs[d]
            row["expected_rating"] = mean
            row["variance"] = variance
        rows[question.id] = row
        _write_self_rows(output_path, rows)
        completed += 1

    print(f"Wrote {completed} new questions to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    defaults = benchmark_defaults()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="Logit-capable model key from config/models.yaml.")
    parser.add_argument(
        "--self",
        dest="self_mode",
        action="store_true",
        help="Run self-baseline mode: collect logprobs without persona conditioning. API models only.",
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
        default=Path(defaults.get("personas_file", "personas.json")),
        help="Persona JSON file.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(defaults.get("logit_collection_temperature", 1.0)),
        help="Temperature for API logprob collection.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output CSV path.")
    parser.add_argument("--force", action="store_true", help="Recompute rows that already exist in the output CSV.")
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=DEFAULT_TOP_LOGPROBS,
        help="Requested top_logprobs for API providers.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional API base URL override for OpenAI-compatible providers.",
    )
    # Local model options
    parser.add_argument("--model-path", type=Path, default=None, help="Optional local GGUF path override.")
    parser.add_argument(
        "--question-limit",
        type=int,
        default=None,
        help="Optional cap on MFQ questions for local runs.",
    )
    parser.add_argument("--n-ctx", type=int, default=DEFAULT_N_CTX, help="Context length for local GGUF models.")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="Number of layers to offload to GPU.")
    parser.add_argument("--threads", type=int, default=None, help="Optional CPU thread count override.")
    parser.add_argument("--limit", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    ensure_data_dirs()
    args = parse_args()
    selected_model = (
        model_config(args.model, capability="logit") if args.model else prompt_for_model_selection("logit")
    )
    provider = str(selected_model["provider"])

    if args.self_mode:
        if provider == "local_logits":
            raise SystemExit("--self mode is only supported for API models, not local_logits.")
        output_path = args.output or (LOGIT_DIR / f"{selected_model['stem']}_self_logprobs.csv")
        _collect_api_self_logprobs(
            selected_model,
            output_path,
            temperature=args.temperature,
            top_logprobs=args.top_logprobs,
            base_url=args.base_url,
            force=args.force,
        )
    elif provider == "local_logits":
        output_path = args.output or resolve_logit_output_path(selected_model)
        persona_limit = args.limit if args.limit is not None else args.p
        personas = load_personas(args.personas_file, persona_limit)
        _collect_local_logits(
            selected_model,
            personas,
            output_path,
            model_path=args.model_path,
            question_limit=args.question_limit,
            force=args.force,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            threads=args.threads,
        )
    else:
        output_path = args.output or resolve_logit_output_path(selected_model)
        persona_limit = args.limit if args.limit is not None else args.p
        personas = load_personas(args.personas_file, persona_limit)
        _collect_api_logprobs(
            selected_model,
            personas,
            output_path,
            temperature=args.temperature,
            top_logprobs=args.top_logprobs,
            base_url=args.base_url,
            force=args.force,
        )


if __name__ == "__main__":
    main()
