"""DeepSeek model implementations."""

import json
import os
from typing import Any

from deterministic_horizon.models.base import BaseModel, ModelResponse
from tenacity import retry, stop_after_attempt, wait_exponential


class DeepSeekModel(BaseModel):
    """Interface for DeepSeek models via their API."""

    # Pricing per 1K tokens
    PRICING = {}

    API_BASE = "https://api.deepseek.com/v1"

    def _setup_client(self) -> None:
        """Set up DeepSeek client using OpenAI-compatible interface."""
        try:
            from openai import OpenAI
        except ImportError as err:
            raise ImportError("openai package required. Install with: pip install openai") from err

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")

        self._client = OpenAI(
            api_key=api_key,
            base_url=self.API_BASE,
            timeout=self.timeout,
        )

    def _get_api_model_name(self) -> str:
        """Get the API model name."""
        model_lower = self.model_name.lower()
        if "r1" in model_lower:
            return "deepseek-reasoner"
        elif "v3" in model_lower:
            return "deepseek-chat"
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
        """Make DeepSeek API call."""
        is_r1 = "r1" in self.model_name.lower()

        call_kwargs = {
            "model": self._get_api_model_name(),
            "messages": messages,
            "max_tokens": self.max_tokens,
        }

        # R1 model uses fixed temperature
        if not is_r1:
            call_kwargs["temperature"] = self.temperature

        if tools and not is_r1:  # R1 doesn't support tools natively
            call_kwargs["tools"] = [
                {"type": "function", "function": tool.get("function", tool)} for tool in tools
            ]

        response = self._client.chat.completions.create(**call_kwargs)
        return response.model_dump()

    def _parse_response(
        self,
        raw_response: dict[str, Any],
        latency_ms: float,
    ) -> ModelResponse:
        """Parse DeepSeek API response."""
        choice = raw_response.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = raw_response.get("usage", {})

        content = message.get("content", "")

        # Extract tool calls
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

        # Token counts
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # R1 specific: reasoning tokens
        reasoning_tokens = 0
        reasoning_content = None
        if "reasoning_content" in message:
            reasoning_content = message["reasoning_content"]
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
        model_lower = self.model_name.lower()
        for key, prices in self.PRICING.items():
            if key in model_lower:
                return prices
        return {"input": 0.0005, "output": 0.002}


class DeepSeekR1Model(DeepSeekModel):
    """DeepSeek-R1 specific implementation."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("model_name", "deepseek-r1")
        super().__init__(**kwargs)


class DeepSeekV3Model(DeepSeekModel):
    """DeepSeek-V3 specific implementation."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("model_name", "deepseek-v3")
        super().__init__(**kwargs)
