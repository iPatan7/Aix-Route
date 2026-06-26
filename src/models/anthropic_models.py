"""Anthropic model implementations (Claude models)."""

import os
from typing import Any

from deterministic_horizon.models.base import BaseModel, ModelResponse
from tenacity import retry, stop_after_attempt, wait_exponential


class AnthropicModel(BaseModel):
    """Interface for Anthropic Claude models."""

    PRICING = {}
    MODEL_MAPPING = {}

    def _setup_client(self) -> None:
        """Set up Anthropic client."""
        try:
            from anthropic import Anthropic
        except ImportError as err:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            ) from err

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self._client = Anthropic(api_key=api_key, timeout=self.timeout)

    def _get_api_model_name(self) -> str:
        """Get the API model name from the user-friendly name."""
        model_lower = self.model_name.lower()
        for key, value in self.MODEL_MAPPING.items():
            if key in model_lower:
                return value
        return self.model_name

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
        """Make Anthropic API call."""
        # Extract system prompt
        system = None
        filtered_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                filtered_messages.append(msg)

        call_kwargs = {
            "model": self._get_api_model_name(),
            "messages": filtered_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        if system:
            call_kwargs["system"] = system

        if tools:
            call_kwargs["tools"] = [
                {
                    "name": tool.get("function", tool).get("name"),
                    "description": tool.get("function", tool).get("description", ""),
                    "input_schema": tool.get("function", tool).get("parameters", {}),
                }
                for tool in tools
            ]

        response = self._client.messages.create(**call_kwargs)
        return self._response_to_dict(response)

    def _response_to_dict(self, response: Any) -> dict[str, Any]:
        """Convert Anthropic response to dictionary."""
        return {
            "id": response.id,
            "type": response.type,
            "role": response.role,
            "content": [
                {"type": c.type, "text": getattr(c, "text", None), **self._block_to_dict(c)}
                for c in response.content
            ],
            "model": response.model,
            "stop_reason": response.stop_reason,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }

    def _block_to_dict(self, block: Any) -> dict[str, Any]:
        """Convert content block to dictionary."""
        if block.type == "tool_use":
            return {
                "id": block.id,
                "name": block.name,
                "input": block.input,
            }
        return {}

    def _parse_response(
        self,
        raw_response: dict[str, Any],
        latency_ms: float,
    ) -> ModelResponse:
        """Parse Anthropic API response."""
        content_blocks = raw_response.get("content", [])
        usage = raw_response.get("usage", {})

        # Extract text content
        text_content = []
        tool_calls = []

        for block in content_blocks:
            if block.get("type") == "text":
                text_content.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": block.get("input", {}),
                        },
                    }
                )

        content = "\n".join(text_content)

        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)

        return ModelResponse(
            content=content,
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
        """Get pricing for this model."""
        model_lower = self.model_name.lower()
        for key, prices in self.PRICING.items():
            if key in model_lower:
                return prices
        return {"input": 0.003, "output": 0.015}


class ClaudeSonnetModel(AnthropicModel):
    """Claude Sonnet specific implementation."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("model_name", "claude-4.5-sonnet")
        super().__init__(**kwargs)


class ClaudeOpusModel(AnthropicModel):
    """Claude Opus specific implementation."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("model_name", "claude-4-opus")
        super().__init__(**kwargs)
