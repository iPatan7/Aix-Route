"""OpenAI model implementations (GPT-4o, o1, o3-mini, etc.)."""

import json
import os
from typing import Any

from deterministic_horizon.models.base import BaseModel, ModelResponse
from tenacity import retry, stop_after_attempt, wait_exponential


class OpenAIModel(BaseModel):
    """Interface for OpenAI models."""

    PRICING = {}

    def _setup_client(self) -> None:
        """Set up OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError as err:
            raise ImportError("openai package required. Install with: pip install openai") from err

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self._client = OpenAI(api_key=api_key, timeout=self.timeout)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
    )
    def _call_api(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Make OpenAI API call."""
        # Handle o1/o3 models (no system prompt, no temperature)
        is_reasoning_model = any(x in self.model_name.lower() for x in ["o1", "o3"])

        call_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }

        if not is_reasoning_model:
            call_kwargs["temperature"] = self.temperature

        if tools:
            call_kwargs["tools"] = [
                {"type": "function", "function": tool.get("function", tool)} for tool in tools
            ]
            call_kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**call_kwargs)
        return response.model_dump()

    def _parse_response(
        self,
        raw_response: dict[str, Any],
        latency_ms: float,
    ) -> ModelResponse:
        """Parse OpenAI API response."""
        choice = raw_response.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = raw_response.get("usage", {})

        # Extract content
        content = message.get("content", "")

        # Extract tool calls if present
        tool_calls = []
        raw_tool_calls = message.get("tool_calls", [])
        for tc in raw_tool_calls:
            tool_calls.append(
                {
                    "id": tc.get("id", ""),
                    "type": tc.get("type", "function"),
                    "function": {
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": json.loads(tc.get("function", {}).get("arguments", "{}")),
                    },
                }
            )

        # Get token counts
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Handle reasoning tokens for o1/o3 models
        reasoning_tokens = 0
        reasoning_content = None
        if "completion_tokens_details" in usage:
            details = usage["completion_tokens_details"]
            reasoning_tokens = details.get("reasoning_tokens", 0)

        return ModelResponse(
            content=content,
            model=raw_response.get("model", self.model_name),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            reasoning_content=reasoning_content,
            reasoning_tokens=reasoning_tokens,
            tool_calls=tool_calls,
            raw_response=raw_response,
        )

    @property
    def pricing(self) -> dict[str, float]:
        """Get pricing for this model."""
        model_key = self.model_name.lower()
        for key, prices in self.PRICING.items():
            if key in model_key:
                return prices
        # Default pricing
        return {"input": 0.0025, "output": 0.01}


# Specialized classes for specific model behaviors
class GPT4oModel(OpenAIModel):
    """GPT-4o specific implementation."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("model_name", "gpt-4o")
        super().__init__(**kwargs)


class O1Model(OpenAIModel):
    """o1 model implementation (reasoning model)."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("model_name", "o1-preview")
        # o1 models don't support temperature
        kwargs["temperature"] = 1.0  # Only valid value
        super().__init__(**kwargs)

    def _call_api(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Override to handle o1-specific requirements."""
        # o1 models don't support system prompts - convert to user message
        processed_messages = []
        system_content = []

        for msg in messages:
            if msg["role"] == "system":
                system_content.append(msg["content"])
            else:
                if system_content and msg["role"] == "user":
                    # Prepend system content to first user message
                    msg = {
                        "role": "user",
                        "content": "\n\n".join(system_content) + "\n\n" + msg["content"],
                    }
                    system_content = []
                processed_messages.append(msg)

        # Call with processed messages
        call_kwargs = {
            "model": self.model_name,
            "messages": processed_messages,
            "max_completion_tokens": self.max_tokens,
        }

        response = self._client.chat.completions.create(**call_kwargs)
        return response.model_dump()


class O3MiniModel(OpenAIModel):
    """o3-mini model implementation."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("model_name", "o3-mini")
        super().__init__(**kwargs)
