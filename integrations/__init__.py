"""
Deterministic Horizon — framework integrations.

Drop-in routing for popular agent frameworks. Each integration wraps your
existing LLM/agent and, before spending tokens on a hard reasoning chain, asks
the Deterministic Horizon policy layer whether the problem is past the model's
horizon — and if so, routes to a tool instead.

Sub-packages (import the one matching your stack):

    from integrations.langchain import HorizonRouterNode
    from integrations.autogen import HorizonRouterAgent
    from integrations.openai_agents import horizon_tool_use

The shared policy access lives in :mod:`integrations._core`; depth estimation
fallback in :mod:`integrations._depth`. Framework SDKs are optional — each
sub-package degrades gracefully (and is unit-testable) without them installed.
"""

from __future__ import annotations

from integrations._core import delegation_decision, horizon_for, should_delegate
from integrations._depth import resolve_depth

__all__ = [
    "should_delegate",
    "delegation_decision",
    "horizon_for",
    "resolve_depth",
]
