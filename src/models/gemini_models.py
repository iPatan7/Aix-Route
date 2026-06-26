"""
Google Gemini adapter.

Google exposes an OpenAI-compatible endpoint for the Gemini models, so we reuse
:class:`OpenAICompatibleModel` rather than depending on ``google-generativeai``.
Set ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) in your environment; the only
extra dependency is the ``openai`` client (``pip install openai``).

Reference: https://ai.google.dev/gemini-api/docs/openai

Examples
--------
>>> from deterministic_horizon.models import load_model
>>> m = load_model("gemini-2.0-flash")               # doctest: +SKIP
>>> m.generate("2+2?").content                        # doctest: +SKIP
"""

from __future__ import annotations

from typing import ClassVar

from deterministic_horizon.models.openai_compatible import OpenAICompatibleModel


class GeminiModel(OpenAICompatibleModel):
    """Google Gemini via the OpenAI-compatible Generative Language API."""

    API_BASE: ClassVar[str] = "https://generativelanguage.googleapis.com/v1beta/openai/"
    ENV_KEYS: ClassVar[tuple[str, ...]] = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    DEFAULT_PRICING: ClassVar[dict[str, float]] = {"input": 0.00125, "output": 0.005}

    # Friendly identifiers → Gemini model ids (mostly identity; kept explicit so
    # ``load_model`` substring-matching stays predictable).
    ALIASES: ClassVar[dict[str, str]] = {
        "gemini-2.5-pro": "gemini-2.5-pro",
        "gemini-2.0-flash": "gemini-2.0-flash",
        "gemini-2.0-pro": "gemini-2.0-pro-exp",
        "gemini-1.5-pro": "gemini-1.5-pro",
        "gemini-1.5-flash": "gemini-1.5-flash",
    }
