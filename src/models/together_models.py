"""
Together AI adapter.

Together serves open-weight models (Llama, Qwen, Mixtral, DeepSeek, …) over an
OpenAI-compatible endpoint, so we reuse :class:`OpenAICompatibleModel`. Set
``TOGETHER_API_KEY`` in your environment (see ``.env.example``).

Examples
--------
>>> from deterministic_horizon.models import load_model
>>> m = load_model("together-llama-3.3-70b")        # doctest: +SKIP
>>> m.generate("2+2?").content                       # doctest: +SKIP
"""

from __future__ import annotations

from typing import ClassVar

from deterministic_horizon.models.openai_compatible import OpenAICompatibleModel


class TogetherModel(OpenAICompatibleModel):
    """Open-weight models hosted on Together AI (OpenAI-compatible API)."""

    API_BASE: ClassVar[str] = "https://api.together.xyz/v1"
    ENV_KEYS: ClassVar[tuple[str, ...]] = ("TOGETHER_API_KEY",)
    DEFAULT_PRICING: ClassVar[dict[str, float]] = {"input": 0.0006, "output": 0.0006}

    # Friendly identifiers → Together's fully-qualified model ids.
    ALIASES: ClassVar[dict[str, str]] = {
        "together-llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "together-llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "together-llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "together-qwen-2.5-72b": "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "together-qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "together-deepseek-r1": "deepseek-ai/DeepSeek-R1",
        "together-mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    }
