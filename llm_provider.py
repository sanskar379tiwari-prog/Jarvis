"""
Single LLM entry point. Swap providers here without changing planner / tools.

Env:
  GEMINI_API_KEY or GOOGLE_API_KEY — required for Gemini
  GEMINI_MODEL — model id (default: gemini-3.1-flash-lite-preview)
  GEMINI_TEMPERATURE — float 0–2 (default: 0.2)
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env if it exists


def _api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    return key.strip()


def _model_name() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview").strip() or "gemini-3.1-flash-lite-preview"


def _temperature() -> float:
    raw = os.getenv("GEMINI_TEMPERATURE", "0.2")
    try:
        return float(raw)
    except ValueError:
        return 0.2


@lru_cache(maxsize=1)
def _configured_model() -> Any:
    key = _api_key()
    if not key:
        raise RuntimeError(
            "Set GEMINI_API_KEY or GOOGLE_API_KEY for Gemini (see llm_provider.py)."
        )
    genai.configure(api_key=key)
    return genai.GenerativeModel(_model_name())


def call_llm(prompt: str) -> str:
    """
    Run the configured Gemini model and return plain text (supports multi-line).
    Same contract as the previous Ollama `llm.invoke(prompt)` usage.
    """
    if prompt is None:
        return ""
    text_in = str(prompt)
    model = _configured_model()
    try:
        response = model.generate_content(
            text_in,
            generation_config={"temperature": _temperature()},
        )
    except Exception as e:
        print(f"LLM Error: {type(e).__name__}: {e}")
        raise

    if not getattr(response, "candidates", None):
        print("LLM Error: No candidates returned in response.")
        return ""
    try:
        out = response.text
        return (out or "").strip()
    except ValueError:
        chunks: list[str] = []
        for cand in response.candidates:
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                t = getattr(part, "text", None)
                if t:
                    chunks.append(t)
        return "".join(chunks).strip()
