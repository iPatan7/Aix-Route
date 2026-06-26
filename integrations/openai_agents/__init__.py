"""OpenAI Agents SDK integration for the Deterministic Horizon router."""

from __future__ import annotations

from integrations.openai_agents.tool import (
    HorizonRoutedTool,
    horizon_tool_use,
)

__all__ = ["horizon_tool_use", "HorizonRoutedTool"]
