"""Registry-resolution tests for the model adapters (no API calls)."""

from __future__ import annotations

import pytest
from deterministic_horizon.models import MODEL_REGISTRY, load_model


@pytest.mark.parametrize(
    ("name", "expected_cls"),
    [
        ("gpt-4o", "OpenAIModel"),
        ("o3-mini", "OpenAIModel"),
        ("claude-4.5-opus", "AnthropicModel"),
        ("deepseek-r1", "DeepSeekModel"),
        ("gemini-2.0-flash", "GeminiModel"),
        ("gemini-1.5-pro", "GeminiModel"),
        ("together-llama-3.3-70b", "TogetherModel"),
        ("together-qwen-2.5-7b", "TogetherModel"),
        ("llama-3.1-8b", "LocalModel"),
        ("qwen-2.5-72b", "LocalModel"),
    ],
)
def test_registry_resolves_to_expected_class(name, expected_cls):
    # Resolving the class must not require the optional `openai`/`torch` deps,
    # because adapters import them lazily inside `_setup_client`.
    assert MODEL_REGISTRY[name].__name__ == expected_cls


def test_specific_id_beats_loose_substring_match():
    # "llama-3.1-8b" must stay LocalModel even though "together-llama-3.1-8b"
    # also contains it — exact match wins (see load_model).
    from deterministic_horizon.models import _REGISTRY

    module, cls = _REGISTRY["llama-3.1-8b"]
    assert cls == "LocalModel"


def test_gemini_and_together_are_openai_compatible():
    from deterministic_horizon.models.openai_compatible import OpenAICompatibleModel

    assert issubclass(MODEL_REGISTRY["gemini-2.0-flash"], OpenAICompatibleModel)
    assert issubclass(MODEL_REGISTRY["together-deepseek-r1"], OpenAICompatibleModel)


def test_unknown_model_raises_with_help():
    with pytest.raises(ValueError, match="Unknown model"):
        load_model("not-a-real-model-xyz")
