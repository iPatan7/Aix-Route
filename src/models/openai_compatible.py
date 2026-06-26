"""
Shared base for providers that expose an **OpenAI-compatible** Chat Completions
endpoint (DeepSeek, Together, Google Gemini, and many local servers such as
vLLM's OpenAI server or Ollama).

Subclasses only need to set three things:

* ``API_BASE``   — the ``base_url`` for the OpenAI client.
* ``ENV_KEYS``   — environment variable(s) holding the API key (first match wins).
* ``ALIASES``    — optional friendly-name → provider-model-id mapping.

The request/response plumbing (tool calls, token accounting) is identical across
these providers, so it lives here once. DeepSeek keeps its own bespoke module
(it has R1-specific reasoning-token handling); this base covers the rest.
"""

from __future__ import annotations

import json
import os
from typing import Any, ClassVar

from deterministic_horizon.models.base import BaseModel, ModelResponse
from tenacity import retry, stop_after_attempt, wait_exponential


class OpenAICompatibleModel(BaseModel):
    """A model served over an OpenAI-compatible ``/chat/completions`` API."""

    API_BASE: ClassVar[str] = "https://api.openai.com/v1"
    ENV_KEYS: ClassVar[tuple[str, ...]] = ("OPENAI_API_KEY",)
    ALIASES: ClassVar[dict[str, str]] = {}
    DEFAULT_PRICING: ClassVar[dict[str, float]] = {"input": 0.0005, "output": 0.002}

    def _api_key(self) -> str:
        for env in self.ENV_KEYS:
            value = os.getenv(env)
            if value:
                return value
        joined = " or ".join(self.ENV_KEYS)
        raise ValueError(f"{joined} environment variable not set")

    def _setup_client(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as err:
            raise ImportError("openai package required. Install with: pip install openai") from err

        self._client = OpenAI(
            api_key=self._api_key(),
            base_url=self.API_BASE,
            timeout=self.timeout,
        )

    def _get_api_model_name(self) -> str:
        """Map a friendly name to the provider's model id (override ALIASES)."""
        name = self.model_name
        return self.ALIASES.get(name.lower(), name)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
    def _call_api(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        call_kwargs: dict[str, Any] = {
            "model": self._get_api_model_name(),
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if tools:
            call_kwargs["tools"] = [
                {"type": "function", "function": tool.get("function", tool)} for tool in tools
            ]
        response = self._client.chat.completions.create(**call_kwargs)
        return response.model_dump()

    def _parse_response(self, raw_response: dict[str, Any], latency_ms: float) -> ModelResponse:
        choice = (raw_response.get("choices") or [{}])[0]
        message = choice.get("message", {}) or {}
        usage = raw_response.get("usage", {}) or {}

        tool_calls = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {}) or {}
            raw_args = fn.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except (ValueError, TypeError):
                args = {"_raw": raw_args}
            tool_calls.append(
                {
                    "id": tc.get("id", ""),
                    "type": tc.get("type", "function"),
                    "function": {"name": fn.get("name", ""), "arguments": args},
                }
            )

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        return ModelResponse(
            content=message.get("content") or "",
            model=raw_response.get("model", self.model_name),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            tool_calls=tool_calls,
            raw_response=raw_response,
        )

    @property
    def pricing(self) -> dict[str, float]:
        return dict(self.DEFAULT_PRICING)
