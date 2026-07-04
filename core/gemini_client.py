"""
Thin wrapper around the Google Gemini API.

Responsibilities:
- Read the API key from environment / Streamlit secrets (never hardcoded).
- Provide `generate()` for plain text generation (storytelling, recommendations).
- Provide `generate_grounded()` for generation with Google Search grounding
  (hidden gems, local events) so results reflect current, real information
  rather than the model's static training data.
- Centralize error handling so the UI layer never has to deal with raw
  SDK exceptions.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Any

from google import genai
from google.genai import types


class GeminiClientError(Exception):
    """Raised when the Gemini API call fails in a way the UI should surface."""


class GeminiQuotaError(GeminiClientError):
    """
    Raised specifically for 429/RESOURCE_EXHAUSTED responses. Kept distinct
    from GeminiClientError so the UI can show actionable guidance instead of
    a raw stack trace.
    """


def _is_quota_error(exc: Exception) -> bool:
    """Check if an exception is related to API quota exhaustion."""
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def _with_retry(fn: Callable[[], Any], max_retries: int = 2, base_delay: float = 2.0) -> Any:
    """
    Retries a callable on quota/rate-limit errors with exponential backoff.
    If multiple API keys are available, it will rotate keys before backing off.
    Only retries on 429s -- any other error fails immediately since retrying
    won't help (e.g. bad request, auth failure).
    """
    last_exc = None
    _init_keys()
    total_attempts = (max_retries + 1) * max(1, len(_api_keys))

    for attempt in range(total_attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if _is_quota_error(exc) and attempt < total_attempts - 1:
                rotated = rotate_client()
                if not rotated:
                    time.sleep(base_delay * (2 ** (attempt % (max_retries + 1))))
                continue
            raise
    raise last_exc


@dataclass
class GenerationResult:
    text: str
    grounded: bool = False
    sources: list[str] | None = None


def _get_api_keys() -> list[str]:
    """
    Resolve API keys from Streamlit secrets first, then environment
    variables. Supports comma-separated keys for rotation.
    Never hardcode a key in source.
    """
    keys_str = ""
    try:
        import streamlit as st

        if "GEMINI_API_KEYS" in st.secrets:
            keys_str = st.secrets["GEMINI_API_KEYS"]
        elif "GEMINI_API_KEY" in st.secrets:
            keys_str = st.secrets["GEMINI_API_KEY"]
    except Exception:
        # st.secrets raises if no secrets.toml exists (e.g. in plain pytest runs).
        pass

    if not keys_str:
        keys_str = os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY") or ""

    if not keys_str:
        raise GeminiClientError(
            "GEMINI_API_KEY(S) is not set. Add it to .streamlit/secrets.toml "
            "or as an environment variable (comma-separated for multiple keys)."
        )
    return [k.strip() for k in keys_str.split(",") if k.strip()]


_api_keys: list[str] = []
_current_key_idx: int = 0
_client: genai.Client | None = None


def _init_keys() -> None:
    global _api_keys
    if not _api_keys:
        _api_keys = _get_api_keys()


def get_client() -> genai.Client:
    """Initialize or return the cached Gemini client."""
    global _client, _api_keys, _current_key_idx
    _init_keys()
    if _client is None:
        _client = genai.Client(api_key=_api_keys[_current_key_idx])
    return _client


def rotate_client() -> bool:
    """Rotates to the next API key if multiple exist. Returns True if rotated."""
    global _client, _api_keys, _current_key_idx
    _init_keys()
    if len(_api_keys) <= 1:
        return False
    _current_key_idx = (_current_key_idx + 1) % len(_api_keys)
    _client = None  # Force recreation on next get_client()
    return True


DEFAULT_MODEL = "gemini-2.5-flash"

# Gemini 2.5 models "think" before answering by default, and thinking tokens
# are deducted from the same max_output_tokens budget as the visible answer.
# For straightforward generation tasks like ours (recommendations, stories,
# event summaries), that reasoning step isn't needed and just eats into the
# budget, causing responses to truncate mid-sentence. Disabling it here keeps
# output complete and also reduces latency/cost.
_NO_THINKING = types.ThinkingConfig(thinking_budget=0)


def _check_truncation(response: Any) -> None:
    """
    Some Gemini builds still spend tokens on internal reasoning even with
    thinking disabled, which can leave an empty or cut-off response with
    finish_reason MAX_TOKENS. Surface that clearly instead of silently
    returning blank/truncated text.
    """
    try:
        finish_reason = response.candidates[0].finish_reason
    except Exception:
        return
    if str(finish_reason) in ("MAX_TOKENS", "FinishReason.MAX_TOKENS") and not (response.text or "").strip():
        raise GeminiClientError(
            "The response was cut off before any text was generated (hit the output token "
            "limit). Try a shorter/simpler query, or increase max_output_tokens in "
            "core/gemini_client.py."
        )


def generate(prompt: str, system_instruction: str | None = None, model: str = DEFAULT_MODEL) -> GenerationResult:
    """Plain generation call — used for storytelling, recommendations, etc."""

    def _call():
        client = get_client()
        return client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.8,
                max_output_tokens=2048,
                thinking_config=_NO_THINKING,
            ),
        )

    try:
        response = _with_retry(_call)
        _check_truncation(response)
        return GenerationResult(text=response.text or "", grounded=False)
    except Exception as exc:
        if _is_quota_error(exc):
            raise GeminiQuotaError(
                "Gemini API quota exceeded. If this is a fresh API key, your Google Cloud "
                "project may need billing linked (still free for free-tier usage -- this just "
                "validates the account) at https://aistudio.google.com. Otherwise, please wait "
                "a moment and try again."
            ) from exc
        raise GeminiClientError(f"Gemini generation failed: {exc}") from exc


def generate_grounded(prompt: str, system_instruction: str | None = None, model: str = DEFAULT_MODEL) -> GenerationResult:
    """
    Generation call with Google Search grounding enabled — used for hidden
    gems and local events, where current, real-world information matters.
    """
    grounding_tool = types.Tool(google_search=types.GoogleSearch())

    def _call():
        client = get_client()
        return client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.6,
                max_output_tokens=2048,
                thinking_config=_NO_THINKING,
                tools=[grounding_tool],
            ),
        )

    try:
        response = _with_retry(_call)
        _check_truncation(response)
        sources = []
        try:
            candidate = response.candidates[0]
            metadata = candidate.grounding_metadata
            if metadata and metadata.grounding_chunks:
                for chunk in metadata.grounding_chunks:
                    if chunk.web and chunk.web.uri:
                        sources.append(chunk.web.uri)
        except Exception:
            pass  # grounding metadata is best-effort, not critical to the answer

        return GenerationResult(text=response.text or "", grounded=True, sources=sources or None)
    except Exception as exc:
        if _is_quota_error(exc):
            raise GeminiQuotaError(
                "Gemini API quota exceeded. If this is a fresh API key, your Google Cloud "
                "project may need billing linked (still free for free-tier usage -- this just "
                "validates the account) at https://aistudio.google.com. Otherwise, please wait "
                "a moment and try again."
            ) from exc
        raise GeminiClientError(f"Gemini grounded generation failed: {exc}") from exc


def embed(texts: list[str], model: str = "text-embedding-004") -> list[list[float]]:
    """Generate embeddings for a list of texts, used by the retrieval module."""
    client = get_client()
    try:
        result = client.models.embed_content(model=model, contents=texts)
        return [e.values for e in result.embeddings]
    except Exception as exc:
        raise GeminiClientError(f"Gemini embedding call failed: {exc}") from exc
