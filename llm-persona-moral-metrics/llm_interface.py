#!/usr/bin/env python3
"""
LLM utilities for different model interfaces
"""

import os
import random
import time
import uuid
from typing import Optional, Dict, Any

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_MODEL_CACHE: Dict[str, Any] = {}
_GOOGLE_THINKING_WARNING_EMITTED = False


def _google_generate_via_rest(
    model_name: str,
    api_key: str,
    prompt: str,
    generation_config: Optional[Dict[str, Any]] = None,
    system_prompt: Optional[str] = None,
    safety_settings: Optional[Any] = None,
    thinking_cfg: Optional[Dict[str, Any]] = None,
    timeout: int = 120,
) -> tuple[str, Optional[str]]:
    """Issue a direct REST call to Gemini and return (text, finish_reason)."""

    import requests

    payload: Dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ]
    }

    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    gen_payload = {k: v for k, v in (generation_config or {}).items() if v is not None}
    if thinking_cfg:
        gen_payload["thinking_config"] = thinking_cfg
    if gen_payload:
        payload["generationConfig"] = _camelize_keys(gen_payload)

    if safety_settings:
        payload["safetySettings"] = safety_settings

    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
        params={"key": api_key},
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()

    data = resp.json()
    texts: list[str] = []
    finish_reason: Optional[str] = None

    for cand in data.get("candidates", []) or []:
        finish_reason = finish_reason or cand.get("finishReason")
        content = cand.get("content") or {}
        for part in content.get("parts", []) or []:
            if isinstance(part, dict):
                text_val = part.get("text")
                if text_val:
                    texts.append(str(text_val))
    if texts:
        return "\n".join(texts).strip(), finish_reason

    prompt_feedback = data.get("promptFeedback") or {}
    finish_reason = finish_reason or prompt_feedback.get("blockReason")
    return "", finish_reason


def _snake_to_camel(name: str) -> str:
    """Convert snake_case keys to camelCase for Google REST payloads."""

    parts = name.split('_')
    if not parts:
        return name
    return parts[0] + ''.join(part.title() for part in parts[1:])


def _camelize_keys(value: Any) -> Any:
    """Recursively convert dict keys to camelCase."""

    if isinstance(value, dict):
        converted = {}
        for key, val in value.items():
            if val is None:
                continue
            converted[_snake_to_camel(key)] = _camelize_keys(val)
        return converted
    if isinstance(value, list):
        return [_camelize_keys(item) for item in value]
    return value


def _normalise_thinking_config(value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Normalize thinking config keys to snake_case variants."""

    if not value:
        return None

    normalised: Dict[str, Any] = {}
    for key, val in value.items():
        if val is None:
            continue

        lower = key.lower()
        if lower in {"thinkingbudget", "max_thinking_tokens", "maxthinkingtokens"}:
            try:
                normalised["thinking_budget"] = int(val)
            except (TypeError, ValueError):
                normalised["thinking_budget"] = val
        elif lower in {"includethoughts", "include_thoughts"}:
            normalised["include_thoughts"] = bool(val)
        else:
            normalised[key] = val

    return normalised or None


def _collect_response_text(value: Any) -> list[str]:
    """Recursively gather text fragments from OpenAI-style payloads."""

    if value is None:
        return []

    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []

    if isinstance(value, (int, float)):
        return [str(value)]

    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            parts.extend(_collect_response_text(item))
        return parts

    if isinstance(value, dict):
        parts: list[str] = []
        for key in ("text", "output_text", "content", "value"):
            if key in value:
                parts.extend(_collect_response_text(value[key]))
        for key in ("parts", "messages", "choices", "data", "output"):
            if key in value:
                parts.extend(_collect_response_text(value[key]))
        return parts

    # Handle SDK response objects with attributes (e.g., ChatCompletionMessage)
    parts: list[str] = []
    for attr in ("text", "output_text", "content", "value"):
        if hasattr(value, attr):
            parts.extend(_collect_response_text(getattr(value, attr)))

    return parts


def _coerce_response_text(value: Any) -> str:
    """Flatten the collected text fragments into a single string."""

    parts = [part for part in _collect_response_text(value) if part]
    if not parts:
        return ""
    return "\n".join(parts).strip()


def _extract_chat_completion_text(response: Any) -> str:
    """Extract assistant text from a Chat Completions payload."""

    try:
        if isinstance(response, dict):
            choices = response.get("choices")
        else:
            choices = getattr(response, "choices", None)
        if not choices:
            return ""

        first_choice = choices[0]
        message = None
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
        else:
            message = getattr(first_choice, "message", None)

        return _coerce_response_text(message)
    except Exception:
        return ""


def _extract_responses_api_text(response: Any) -> str:
    """Extract assistant text from a Responses API payload."""

    text = _coerce_response_text(getattr(response, "output_text", None))
    if text:
        return text

    text = _coerce_response_text(getattr(response, "output", None))
    if text:
        return text

    try:
        return response.model_dump_json()
    except Exception:
        return ""

def get_llm_response(model_type: str, model_name: str, prompt: str, **kwargs) -> str:
    """
    Get response from different LLM models

    Args:
        model_type: Type of model ('openai', 'anthropic', 'ollama', 'openrouter', 'tinker', ...)
        model_name: Specific model name (for Tinker: the base model name used for renderer selection)
        prompt: Input prompt
        **kwargs: Additional parameters

    Returns:
        Model response as string
    """

    if model_type == "openai":
        return _openai_response(model_name, prompt, **kwargs)
    elif model_type == "anthropic":
        return _anthropic_response(model_name, prompt, **kwargs)
    elif model_type == "ollama":
        return _ollama_response(model_name, prompt, **kwargs)
    elif model_type == "local":
        return _local_response(model_name, prompt, **kwargs)
    elif model_type == "openrouter":
        return _openrouter_response(model_name, prompt, **kwargs)
    elif model_type == "xai":
        return _xai_response(model_name, prompt, **kwargs)
    elif model_type == "google":
        return _google_response(model_name, prompt, **kwargs)
    elif model_type == "tinker":
        return _tinker_response(model_name, prompt, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def _openai_response(model_name: str, prompt: str, **kwargs) -> str:
    """Get response from OpenAI API.

    Supports both Chat Completions and the Responses API. If
    `reasoning_effort` is provided (or `use_responses_api=True`), the
    Responses API is used and the effort level is passed through.
    """
    try:
        import openai

        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        client = openai.OpenAI(api_key=api_key)

        # Normalise advanced knobs so we only send recognised fields.
        reasoning_effort = kwargs.get("reasoning_effort")
        if reasoning_effort in {"minimal", "low", "medium", "high"}:
            pass
        elif reasoning_effort is not None:
            reasoning_effort = "minimal"

        temperature = kwargs.get("temperature", 0.1)
        top_p = kwargs.get("top_p")
        presence_penalty = kwargs.get("presence_penalty")
        frequency_penalty = kwargs.get("frequency_penalty")
        max_tokens = kwargs.get("max_tokens", 8)
        system_prompt = kwargs.get("system_prompt") or kwargs.get("system")
        instructions = kwargs.get("instructions")

        is_gpt5 = model_name.startswith("gpt-5") or "gpt-5" in model_name
        use_responses_api = bool(kwargs.get("use_responses_api") or reasoning_effort or is_gpt5)

        def _build_responses_input(user_prompt: str) -> list:
            messages = []
            if system_prompt:
                messages.append(
                    {
                        "role": "system",
                        "content": [
                            {"type": "input_text", "text": system_prompt},
                        ],
                    }
                )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt},
                    ],
                }
            )
            return messages

        if use_responses_api:
            if is_gpt5:
                # GPT-5 prefers the simplified Responses payload without extra tuning knobs.
                messages: list = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                req: Dict[str, Any] = {
                    "model": model_name,
                    "input": messages,
                }

                if max_tokens is not None:
                    req["max_output_tokens"] = max_tokens
                if instructions:
                    req["instructions"] = instructions
                if reasoning_effort:
                    req["reasoning"] = {"effort": reasoning_effort}

            else:
                req = {
                    "model": model_name,
                    "input": _build_responses_input(prompt),
                }
                if max_tokens is not None:
                    req["max_output_tokens"] = max_tokens
                if temperature is not None:
                    req["temperature"] = temperature
                if top_p is not None:
                    req["top_p"] = top_p
                if presence_penalty is not None:
                    req["presence_penalty"] = presence_penalty
                if frequency_penalty is not None:
                    req["frequency_penalty"] = frequency_penalty
                if instructions:
                    req["instructions"] = instructions
                if reasoning_effort:
                    req["reasoning"] = {"effort": reasoning_effort}

            response = None
            last_exc: Optional[Exception] = None

            max_attempts = kwargs.get("max_retries") or (5 if is_gpt5 else 1)
            backoff = kwargs.get("initial_backoff") or 1.0
            idem_key = kwargs.get("idempotency_key") or str(uuid.uuid4())

            extra_headers = dict(kwargs.get("extra_headers", {}) or {})
            if is_gpt5 and "Idempotency-Key" not in extra_headers:
                extra_headers["Idempotency-Key"] = idem_key

            for attempt in range(max_attempts):
                create_kwargs = dict(req)
                if extra_headers:
                    create_kwargs["extra_headers"] = extra_headers

                try:
                    response = client.responses.create(**create_kwargs)
                    break
                except Exception as exc:
                    last_exc = exc
                    status = getattr(exc, "status_code", None)
                    should_retry = is_gpt5 and status in {429, 500, 502, 503, 504}
                    if not should_retry or attempt + 1 >= max_attempts:
                        break
                    print(
                        "OpenAI Responses API error for GPT-5 "
                        f"(attempt {attempt + 1}/{max_attempts}): {exc}"
                    )
                    sleep_time = backoff + random.uniform(0, 0.25)
                    time.sleep(sleep_time)
                    backoff *= 2

            if response is not None:
                text = _extract_responses_api_text(response)
                if text:
                    return text

            if last_exc is not None:
                print(f"OpenAI Responses API error: {last_exc}")

            if is_gpt5:
                return "ERROR"

        # Default: Chat Completions API fallback (older models and general safety net)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        chat_base_kwargs = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }

        chat_max_tokens = kwargs.get("chat_max_tokens")
        if chat_max_tokens is None:
            chat_max_tokens = max_tokens

        last_chat_error: Optional[Exception] = None
        for token_param in ("max_completion_tokens", "max_tokens"):
            chat_kwargs = dict(chat_base_kwargs)
            if chat_max_tokens is not None:
                chat_kwargs[token_param] = chat_max_tokens

            try:
                response = client.chat.completions.create(**chat_kwargs)
                text = _extract_chat_completion_text(response)
                if text:
                    return text
                return ""
            except Exception as exc:
                last_chat_error = exc
                details = str(exc)
                if f"Unsupported parameter: '{token_param}'" not in details:
                    raise

        if last_chat_error is not None:
            raise last_chat_error

    except ImportError:
        raise ImportError("Please install openai: pip install openai")
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "ERROR"


def _openrouter_response(model_name: str, prompt: str, **kwargs) -> str:
    """Get response from OpenRouter's OpenAI-compatible API."""

    try:
        import openai
    except ImportError as exc:
        raise ImportError("Please install openai: pip install openai") from exc

    api_key = kwargs.get("api_key") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY for OpenRouter API")

    base_url = kwargs.get("base_url") or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"

    base_headers = dict(kwargs.get("extra_headers", {}) or {})
    referer = (
        kwargs.get("http_referer")
        or os.getenv("OPENROUTER_HTTP_REFERER")
        or os.getenv("OPENROUTER_APP_URL")
    )
    title = kwargs.get("app_title") or os.getenv("OPENROUTER_APP_NAME")
    if referer and "HTTP-Referer" not in base_headers:
        base_headers["HTTP-Referer"] = referer
    if title and "X-Title" not in base_headers:
        base_headers["X-Title"] = title

    client_kwargs: Dict[str, Any] = {"api_key": api_key, "base_url": base_url}
    if base_headers:
        client_kwargs["default_headers"] = base_headers

    client = openai.OpenAI(**client_kwargs)

    reasoning_effort = kwargs.get("reasoning_effort")
    if reasoning_effort in {"minimal", "low", "medium", "high"}:
        pass
    elif reasoning_effort is not None:
        reasoning_effort = "minimal"

    messages = []
    system_prompt = kwargs.get("system_prompt") or kwargs.get("system")
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    max_tokens = kwargs.get("max_tokens")
    temperature = kwargs.get("temperature", 0.1)
    top_p = kwargs.get("top_p")
    instructions = kwargs.get("instructions")

    is_gpt5 = "gpt-5" in model_name
    use_responses_api = bool(kwargs.get("use_responses_api") or reasoning_effort or is_gpt5)

    if use_responses_api:
        req: Dict[str, Any] = {
            "model": model_name,
            "input": messages,
        }

        if max_tokens is not None:
            req["max_output_tokens"] = max_tokens
        if temperature is not None:
            req["temperature"] = temperature
        if top_p is not None:
            req["top_p"] = top_p
        if instructions:
            req["instructions"] = instructions
        if reasoning_effort:
            req["reasoning"] = {"effort": reasoning_effort}

        call_headers = dict(base_headers)
        if is_gpt5 and "Idempotency-Key" not in call_headers:
            call_headers["Idempotency-Key"] = kwargs.get("idempotency_key") or str(uuid.uuid4())

        create_kwargs = dict(req)
        if call_headers and call_headers != base_headers:
            create_kwargs["extra_headers"] = call_headers

        last_exc: Optional[Exception] = None
        max_attempts = kwargs.get("max_retries") or (5 if is_gpt5 else 2)
        backoff = kwargs.get("initial_backoff") or 1.0

        for attempt in range(max_attempts):
            try:
                response = client.responses.create(**create_kwargs)
                text = _extract_responses_api_text(response)
                if text:
                    return text
                break
            except Exception as exc:
                last_exc = exc
                status = getattr(exc, "status_code", None)
                should_retry = is_gpt5 and status in {429, 500, 502, 503, 504}
                if not should_retry or attempt + 1 >= max_attempts:
                    break
                sleep_time = backoff + random.uniform(0, 0.25)
                time.sleep(sleep_time)
                backoff *= 2

        if last_exc is not None:
            print(f"OpenRouter Responses API error: {last_exc}")

    # Fallback to Chat Completions when available
    try:
        create_kwargs: Dict[str, Any] = dict(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens if max_tokens is not None else 8,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=kwargs.get("presence_penalty"),
            frequency_penalty=kwargs.get("frequency_penalty"),
        )
        extra_body = kwargs.get("extra_body")
        if extra_body:
            create_kwargs["extra_body"] = extra_body
        response = client.chat.completions.create(**create_kwargs)
        text = _extract_chat_completion_text(response)
        if text:
            return text
        return ""
    except Exception as e:
        print(f"OpenRouter API error: {e}")
        return "ERROR"

def _anthropic_response(model_name: str, prompt: str, **kwargs) -> str:
    """Get response from Anthropic API"""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY"))

        response = client.messages.create(
            model=model_name,
            max_tokens=kwargs.get("max_tokens", 5),
            temperature=kwargs.get("temperature", 0.1),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()

    except ImportError:
        raise ImportError("Please install anthropic: pip install anthropic")
    except Exception as e:
        print(f"Anthropic API error: {e}")
        return "ERROR"

def _ollama_response(model_name: str, prompt: str, **kwargs) -> str:
    """Get response from Ollama local API"""
    try:
        import requests

        base_url = kwargs.get("base_url", "http://localhost:11434")

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.1),
                "num_predict": kwargs.get("max_tokens", 5)
            }
        }

        response = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=kwargs.get("timeout", 30)
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("response", "ERROR").strip()
        else:
            print(f"Ollama API error: {response.status_code}")
            return "ERROR"

    except ImportError:
        raise ImportError("Please install requests: pip install requests")
    except Exception as e:
        print(f"Ollama API error: {e}")
        return "ERROR"


def _resolve_model_path(
    model_name: str,
    model_dir: Optional[str] = None,
    model_path: Optional[str] = None,
) -> str:
    """Resolve the absolute path to a local GGUF model."""

    if model_path:
        explicit_path = os.fspath(model_path)
        if os.path.isabs(explicit_path):
            return explicit_path
        return os.path.abspath(explicit_path)

    resolved_model_name = os.fspath(model_name)
    if os.path.isabs(resolved_model_name):
        return resolved_model_name

    search_dir = model_dir or DEFAULT_MODELS_DIR
    return os.path.join(search_dir, resolved_model_name)


def _load_local_model(model_path: str, **kwargs):
    """Load and cache a local llama.cpp model."""

    if model_path in _MODEL_CACHE:
        return _MODEL_CACHE[model_path]

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise ImportError("Please install llama-cpp-python: pip install llama-cpp-python") from exc

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Local model file not found: {model_path}")

    model = Llama(
        model_path=model_path,
        n_ctx=kwargs.get("n_ctx", 2048),
        n_gpu_layers=kwargs.get("n_gpu_layers", -1),
        n_threads=kwargs.get("threads", kwargs.get("n_threads")),
        verbose=kwargs.get("verbose", False),
    )

    _MODEL_CACHE[model_path] = model
    return model


def _local_response(model_name: str, prompt: str, **kwargs) -> str:
    """Generate a response from a local GGUF model via llama.cpp."""

    model_dir = kwargs.get("model_dir")
    model_path = _resolve_model_path(model_name, model_dir, kwargs.get("model_path"))
    load_kwargs = dict(kwargs)
    load_kwargs.pop("model_path", None)

    try:
        model = _load_local_model(model_path, **load_kwargs)
    except Exception as exc:
        print(f"Local model error: {exc}")
        return "ERROR"

    try:
        model.reset()
    except Exception as exc:
        print(f"Local model reset error: {exc}")
        return "ERROR"

    generation_kwargs = {
        "max_tokens": kwargs.get("max_tokens", 5),
        "temperature": kwargs.get("temperature", 0.1),
        "top_p": kwargs.get("top_p", 0.95),
        "repeat_penalty": kwargs.get("repeat_penalty", 1.1),
        "stream": False,
    }

    chat_template = getattr(model, "metadata", {}).get("tokenizer.chat_template")
    has_chat_format = bool(chat_template) or bool(getattr(model, "chat_format", None))

    if has_chat_format and hasattr(model, "create_chat_completion"):
        try:
            result = model.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                **generation_kwargs,
            )
        except Exception as exc:
            print(f"Local model generation error: {exc}")
            return "ERROR"

        text = _extract_chat_completion_text(result) or _coerce_response_text(result)
        return text if text else ""

    try:
        result = model.create_completion(prompt=prompt, **generation_kwargs)
    except Exception as exc:
        print(f"Local model generation error: {exc}")
        return "ERROR"

    choices = result.get("choices", []) if isinstance(result, dict) else []
    if choices:
        return choices[0].get("text", "").strip()

    return ""

def _xai_response(model_name: str, prompt: str, **kwargs) -> str:
    """Get response from xAI (Grok) via OpenAI-compatible API.

    Uses the OpenAI SDK with base_url 'https://api.x.ai/v1'.
    Expects XAI_API_KEY in the environment or passed as api_key.
    """
    try:
        import openai
        api_key = kwargs.get("api_key") or os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("Missing XAI_API_KEY for xAI Grok API")

        base_url = kwargs.get("base_url") or os.getenv("XAI_BASE_URL") or "https://api.x.ai/v1"
        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", 5),
            temperature=kwargs.get("temperature", 0.1),
        )
        return response.choices[0].message.content.strip()

    except ImportError as exc:
        raise ImportError("Please install openai: pip install openai") from exc
    except Exception as e:
        print(f"xAI API error: {e}")
        return "ERROR"

def _google_response(model_name: str, prompt: str, **kwargs) -> str:
    """Get response from Google Gemini via google-generativeai SDK.

    Expects GOOGLE_API_KEY in environment or passed as api_key.
    """
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise ImportError("Please install google-generativeai: pip install google-generativeai") from exc

    api_key = kwargs.get("api_key") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY for Google Gemini API")

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        gen_cfg: Dict[str, Any] = {}
        if "temperature" in kwargs:
            gen_cfg["temperature"] = kwargs.get("temperature")
        if "max_tokens" in kwargs:
            gen_cfg["max_output_tokens"] = kwargs.get("max_tokens")

        system_prompt = kwargs.get("system_prompt") or kwargs.get("system")
        safety_settings = kwargs.get("safety_settings")
        timeout = kwargs.get("timeout", 120)

        thinking_cfg = _normalise_thinking_config(kwargs.get("thinking_config"))
        if "thinkingBudget" in kwargs:
            thinking_cfg = dict(thinking_cfg or {})
            thinking_cfg["thinking_budget"] = kwargs.get("thinkingBudget")

        call_kwargs = {"generation_config": gen_cfg or None}
        rest_attempted = False

        if thinking_cfg is not None:
            try:
                text, finish_reason = _google_generate_via_rest(
                    model_name,
                    api_key,
                    prompt,
                    generation_config=gen_cfg,
                    system_prompt=system_prompt,
                    safety_settings=safety_settings,
                    thinking_cfg=thinking_cfg,
                    timeout=timeout,
                )
            except Exception as exc:  # pragma: no cover - network failure
                error_detail = ""
                response_obj = getattr(exc, "response", None)
                if response_obj is not None:
                    try:
                        error_detail = f" Response: {response_obj.text}"
                    except Exception:
                        pass
                print(
                    f"Google Gemini thinking API error: {exc}{error_detail}; continuing without thinkingConfig."
                )
            else:
                rest_attempted = True
                if text:
                    return text
                if finish_reason:
                    print(
                        "Google Gemini thinking response missing text"
                        f" (finish_reason={finish_reason}); retrying without thinkingBudget."
                    )

            # Preserve behaviour for future SDKs that may accept thinking_config.
            call_kwargs["thinking_config"] = thinking_cfg

        try:
            response = model.generate_content(prompt, **call_kwargs)
        except TypeError as exc:
            if "thinking_config" in str(exc):
                global _GOOGLE_THINKING_WARNING_EMITTED
                if not _GOOGLE_THINKING_WARNING_EMITTED and thinking_cfg is not None:
                    print(
                        "Warning: google-generativeai SDK does not support thinking_config; "
                        "ignoring thinkingBudget."
                    )
                    _GOOGLE_THINKING_WARNING_EMITTED = True
                response = model.generate_content(
                    prompt,
                    generation_config=call_kwargs.get("generation_config"),
                )
            else:
                raise
        finish_reason = None
        try:
            candidates = getattr(response, "candidates", None)
            if candidates:
                first_candidate = candidates[0]
                finish_reason = getattr(first_candidate, "finish_reason", None)
                if finish_reason is not None and not isinstance(finish_reason, str):
                    finish_reason = str(finish_reason)
        except Exception:
            finish_reason = None

        text = None
        try:
            text_attr = getattr(response, "text", None)
            if isinstance(text_attr, str):
                text = text_attr.strip()
        except Exception:
            text = None

        if text:
            return text
        # Fallback: try candidates
        try:
            for cand in getattr(response, "candidates", []) or []:
                parts = []
                content = getattr(cand, "content", None)
                for part in getattr(content, "parts", []) or []:
                    t = getattr(part, "text", None)
                    if t:
                        parts.append(str(t))
                if parts:
                    return "\n".join(parts).strip()
        except Exception:
            pass

        if not rest_attempted:
            try:
                text, fallback_finish = _google_generate_via_rest(
                    model_name,
                    api_key,
                    prompt,
                    generation_config=gen_cfg,
                    system_prompt=system_prompt,
                    safety_settings=safety_settings,
                    timeout=timeout,
                )
            except Exception as exc:  # pragma: no cover - network failure
                print(f"Google Gemini REST fallback failed: {exc}")
            else:
                finish_reason = finish_reason or fallback_finish
                if text:
                    return text

        if finish_reason:
            print(
                "Google Gemini returned no text"
                f" (finish_reason={finish_reason}). Increase max_tokens if this persists."
            )
        return ""
    except Exception as e:
        print(f"Google Gemini API error: {e}")
        return "ERROR"

def _tinker_response(model_name: str, prompt: str, **kwargs) -> str:
    """Get response from a Tinker-hosted LoRA fine-tuned model.

    Args:
        model_name: Base model name used for renderer/tokenizer selection
                    (e.g. 'deepseek-ai/DeepSeek-V3.1').
        prompt:     Raw user prompt string.
        model_path: tinker:// URI of the sampler weights (required, passed via request_kwargs).
        max_tokens: Maximum tokens to generate (default: 8).
        temperature: Sampling temperature (default: 0.1).
    """
    try:
        import tinker
        from tinker_cookbook import model_info as tinfo, renderers
        from tinker_cookbook.renderers import Message
        from tinker_cookbook.tokenizer_utils import get_tokenizer
    except ImportError as exc:
        raise ImportError(
            "Please install tinker and tinker_cookbook to use the Tinker provider."
        ) from exc

    # A fine-tuned variant supplies model_path (tinker:// sampler weights); a base model
    # supplies neither and is sampled directly via base_model=model_name.
    model_path = kwargs.get("model_path")

    cache_key = f"tinker:{model_path or model_name}:{kwargs.get('renderer') or ''}"
    if cache_key not in _MODEL_CACHE:
        # Allow an explicit renderer override (e.g. qwen3_5_disable_thinking for non-thinking
        # mode); otherwise fall back to tinker_cookbook's recommended renderer.
        renderer_name = kwargs.get("renderer") or tinfo.get_recommended_renderer_name(model_name)
        tokenizer = get_tokenizer(model_name)
        renderer = renderers.get_renderer(renderer_name, tokenizer)
        service_client = tinker.ServiceClient()
        if model_path:
            sampling_client = service_client.create_sampling_client(model_path=model_path)
        else:
            sampling_client = service_client.create_sampling_client(base_model=model_name)
        _MODEL_CACHE[cache_key] = (sampling_client, tokenizer, renderer)

    sampling_client, tokenizer, renderer = _MODEL_CACHE[cache_key]

    prompt_input = renderer.build_generation_prompt([Message(role="user", content=prompt)])
    params = tinker.SamplingParams(
        max_tokens=kwargs.get("max_tokens", 8),
        temperature=kwargs.get("temperature", 0.1),
    )

    try:
        result = sampling_client.sample(prompt_input, 1, params).result()
        for s in result.sequences:
            return tokenizer.decode(s.tokens).strip()
        return ""
    except Exception as exc:
        print(f"Tinker sampling error: {exc}")
        return "ERROR"


def check_model_availability(model_type: str, model_name: str, **kwargs) -> bool:
    """Check if a model is available and accessible"""

    if model_type == "ollama":
        try:
            import requests
            base_url = kwargs.get("base_url", "http://localhost:11434")

            # Check if Ollama is running
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False

            # Check if specific model is available
            models = response.json().get("models", [])
            available_models = [model["name"] for model in models]
            return model_name in available_models

        except Exception:
            return False

    elif model_type == "openai":
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        return api_key is not None

    elif model_type == "openrouter":
        api_key = kwargs.get("api_key") or os.getenv("OPENROUTER_API_KEY")
        return api_key is not None

    elif model_type == "anthropic":
        api_key = kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        return api_key is not None

    elif model_type == "local":
        model_dir = kwargs.get("model_dir")
        model_path = _resolve_model_path(model_name, model_dir, kwargs.get("model_path"))
        return os.path.isfile(model_path)

    elif model_type == "xai":
        api_key = kwargs.get("api_key") or os.getenv("XAI_API_KEY")
        return api_key is not None

    elif model_type == "google":
        api_key = kwargs.get("api_key") or os.getenv("GOOGLE_API_KEY")
        return api_key is not None

    return False

def list_available_models(model_type: str, **kwargs) -> list:
    """List available models for a given model type"""

    if model_type == "ollama":
        try:
            import requests
            base_url = kwargs.get("base_url", "http://localhost:11434")

            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]

        except Exception:
            pass

    elif model_type == "local":
        model_dir = kwargs.get("model_dir") or DEFAULT_MODELS_DIR
        try:
            return [
                filename
                for filename in os.listdir(model_dir)
                if filename.lower().endswith(".gguf")
            ]
        except OSError:
            return []

    return []
