"""
Shared LLM client factory — routes all calls through Groq's OpenAI-compatible API.

Key resolution order (first non-empty wins):
  GROQ_API_KEY → GROQ_API_KEY_2 → OPENAI_API_KEY → ANTHROPIC_API_KEY

Model tiers (Groq free):
  llama-3.3-70b-versatile  → 100k tokens/day  — best quality (strategy + audit)
  llama-3.1-8b-instant     → 500k tokens/day  — fast + cheap (bootstrap lessons)
"""
from __future__ import annotations

import logging
import os
from openai import OpenAI

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Quality model for live strategy + auditing
STRATEGY_MODEL  = "llama-3.3-70b-versatile"
AUDITOR_MODEL   = "llama-3.3-70b-versatile"

# Fast model for bootstrap lesson generation (5x higher daily token limit)
BOOTSTRAP_MODEL = "llama-3.1-8b-instant"


def _all_keys() -> list[str]:
    """Return all configured Groq API keys (in priority order)."""
    candidates = [
        os.environ.get("GROQ_API_KEY"),
        os.environ.get("GROQ_API_KEY_2"),
        os.environ.get("OPENAI_API_KEY"),
        os.environ.get("ANTHROPIC_API_KEY"),
    ]
    return [k for k in candidates if k and k.startswith("gsk_")]


def get_groq_api_key() -> str:
    keys = _all_keys()
    if not keys:
        raise RuntimeError(
            "No Groq API key found. Set GROQ_API_KEY in .env "
            "(get one free at console.groq.com)"
        )
    return keys[0]


def get_client(key_index: int = 0) -> OpenAI:
    """Return an OpenAI-compatible client pointed at Groq."""
    keys = _all_keys()
    if not keys:
        raise RuntimeError("No Groq API key configured.")
    key = keys[key_index % len(keys)]
    return OpenAI(api_key=key, base_url=GROQ_BASE_URL)


def chat_with_fallback(
    messages: list[dict],
    model: str = STRATEGY_MODEL,
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    """
    Call Groq chat completions with automatic key rotation on 429.
    Falls back to BOOTSTRAP_MODEL if the primary model hits its daily limit.
    """
    keys = _all_keys()
    fallback_model = BOOTSTRAP_MODEL if model != BOOTSTRAP_MODEL else model

    for attempt, key in enumerate(keys):
        client = OpenAI(api_key=key, base_url=GROQ_BASE_URL)
        use_model = model if attempt == 0 else fallback_model
        try:
            resp = client.chat.completions.create(
                model=use_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if attempt > 0:
                logger.info("Fell back to key #%d / model %s", attempt + 1, use_model)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "429" in err and "tokens per day" in err.lower():
                logger.warning("Key #%d daily token limit hit — trying next key/model", attempt + 1)
                continue
            raise

    raise RuntimeError("All Groq API keys exhausted their daily token limits.")
