"""
Groq adapter — fast inference for open models, free tier.

Groq Cloud serves open-weight models (Llama, OpenAI gpt-oss, …) over an
OpenAI-compatible Chat Completions endpoint, so we reuse
:class:`OpenAICompatibleModel`. Set ``GROQ_API_KEY`` in your environment (see
``.env.example``); the only extra dependency is the ``openai`` client
(``pip install openai``).

Groq's free tier makes it a zero-cost backend for validating the Deterministic
Horizon routing against real frontier-class open models.

Reference: https://console.groq.com/docs/openai

Examples
--------
>>> from deterministic_horizon.models import load_model
>>> m = load_model("llama-3.3-70b-versatile")        # doctest: +SKIP
>>> m.generate("2+2?").content                        # doctest: +SKIP
"""

from __future__ import annotations

from typing import ClassVar

from deterministic_horizon.models.openai_compatible import OpenAICompatibleModel


class GroqModel(OpenAICompatibleModel):
    """Open-weight models hosted on Groq (OpenAI-compatible API)."""

    API_BASE: ClassVar[str] = "https://api.groq.com/openai/v1"
    ENV_KEYS: ClassVar[tuple[str, ...]] = ("GROQ_API_KEY",)
    # Per-1K-token pricing (USD); default tier mirrors the 8b-instant model.
    DEFAULT_PRICING: ClassVar[dict[str, float]] = {"input": 0.00005, "output": 0.00008}

    # Friendly identifiers → Groq production model ids (identity; kept explicit
    # so ``load_model`` substring matching stays predictable).
    ALIASES: ClassVar[dict[str, str]] = {
        "llama-3.1-8b-instant": "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
        "gpt-oss-20b": "openai/gpt-oss-20b",
        "gpt-oss-120b": "openai/gpt-oss-120b",
    }
