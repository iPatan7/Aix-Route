"""Abstract base class for model interfaces."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelResponse:
    """Container for model response data."""

    # Core response
    content: str

    # Metadata
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Timing
    latency_ms: float = 0.0

    # Reasoning trace (for o1-style models)
    reasoning_content: str | None = None
    reasoning_tokens: int = 0

    # Tool calls (for C3 condition)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)

    # Raw response for debugging
    raw_response: dict[str, Any] | None = None

    @property
    def total_cost(self) -> float:
        """Estimate cost based on token usage."""
        # Default pricing (override in subclasses for accurate pricing)
        input_cost_per_1k = 0.0025
        output_cost_per_1k = 0.01
        return (
            self.prompt_tokens * input_cost_per_1k / 1000
            + self.completion_tokens * output_cost_per_1k / 1000
        )


class BaseModel(ABC):
    """Abstract base class for LLM interfaces."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        timeout: int = 120,
        max_retries: int = 3,
        **kwargs,
    ) -> None:
        """
        Initialize model interface.

        Args:
            model_name: Name/identifier of the model
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            **kwargs: Additional model-specific arguments
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.kwargs = kwargs

        # Initialize client
        self._client = None
        self._setup_client()

    @abstractmethod
    def _setup_client(self) -> None:
        """Set up the API client. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _call_api(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> dict[str, Any]:
        """
        Make API call. Must be implemented by subclasses.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional API-specific arguments

        Returns:
            Raw API response dictionary
        """
        pass

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> ModelResponse:
        """
        Generate a response for the given prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation arguments

        Returns:
            ModelResponse containing the generated content
        """
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Make API call with timing
        start_time = time.perf_counter()
        raw_response = self._call_api(messages, **kwargs)
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Parse response
        return self._parse_response(raw_response, latency_ms)

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        system_prompt: str | None = None,
        max_tool_calls: int = 10,
        **kwargs,
    ) -> ModelResponse:
        """
        Generate with tool calling capability (C3 condition).

        Args:
            prompt: User prompt
            tools: List of tool definitions
            system_prompt: Optional system prompt
            max_tool_calls: Maximum number of tool calls
            **kwargs: Additional generation arguments

        Returns:
            ModelResponse with tool call results
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        tool_calls = []
        tool_results = []
        total_latency = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for _ in range(max_tool_calls):
            start_time = time.perf_counter()
            raw_response = self._call_api(messages, tools=tools, **kwargs)
            total_latency += (time.perf_counter() - start_time) * 1000

            response = self._parse_response(raw_response, total_latency)
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens

            # Check for tool calls
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_calls.append(tool_call)

                    # Execute tool
                    result = self._execute_tool(tool_call, tools)
                    tool_results.append(result)

                    # Add to message history
                    messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [tool_call],
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.get("id", ""),
                            "content": str(result),
                        }
                    )
            else:
                # No more tool calls, return final response
                return ModelResponse(
                    content=response.content,
                    model=self.model_name,
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    total_tokens=total_prompt_tokens + total_completion_tokens,
                    latency_ms=total_latency,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    raw_response=raw_response,
                )

        # Max tool calls reached
        return ModelResponse(
            content="Maximum tool calls reached",
            model=self.model_name,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
            latency_ms=total_latency,
            tool_calls=tool_calls,
            tool_results=tool_results,
        )

    @abstractmethod
    def _parse_response(
        self,
        raw_response: dict[str, Any],
        latency_ms: float,
    ) -> ModelResponse:
        """Parse raw API response into ModelResponse."""
        pass

    def _execute_tool(
        self,
        tool_call: dict[str, Any],
        tools: list[dict[str, Any]],
    ) -> Any:
        """
        Execute a tool call.

        Override in subclasses or pass custom tool executor.
        """
        # Default implementation - should be overridden
        tool_name = tool_call.get("function", {}).get("name", "")
        tool_args = tool_call.get("function", {}).get("arguments", {})

        # Find tool definition
        for tool in tools:
            if tool.get("function", {}).get("name") == tool_name:
                # Execute tool function if provided
                executor = tool.get("executor")
                if executor and callable(executor):
                    return executor(**tool_args)

        return {"error": f"Tool {tool_name} not found or no executor provided"}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, temperature={self.temperature})"
