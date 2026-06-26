"""
``HorizonRouterAgent`` — an AutoGen ``ConversableAgent`` that delegates past the
Deterministic Horizon.

The agent intercepts ``generate_reply``. Before letting the LLM reason through a
message, it estimates the message's reasoning depth and asks the policy layer
whether the model can be trusted. If the depth is past the horizon it calls the
attached ``tool`` and returns the tool's result; otherwise it defers to the
normal AutoGen LLM reply.

    from integrations.autogen import HorizonRouterAgent

    agent = HorizonRouterAgent(
        name="router",
        llm_config={"model": "gpt-4o"},
        tool=lambda problem: bfs_solve(problem),   # deterministic solver
        horizon_model="gpt-4o",
    )

``pyautogen`` is optional. When it is installed ``HorizonRouterAgent`` subclasses
the real ``ConversableAgent`` and overrides ``generate_reply``. When it is not,
the agent falls back to a minimal stand-in exposing ``name``, ``generate_reply``
and ``receive`` so the routing logic stays importable and testable.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from integrations._core import delegation_decision
from integrations._depth import resolve_depth

# --- Optional pyautogen base ------------------------------------------------
try:  # pragma: no cover - exercised only when pyautogen is installed
    from autogen import ConversableAgent as _ConversableAgent

    _HAS_AUTOGEN = True
except Exception:  # noqa: BLE001 - pyautogen is optional

    class _ConversableAgent:  # minimal stand-in for offline use / tests
        """Fallback base when ``pyautogen`` is not installed."""

        def __init__(self, name: str = "agent", **kwargs: Any) -> None:
            self.name = name
            self._kwargs = kwargs

        def generate_reply(
            self,
            messages: Sequence[Mapping[str, Any]] | None = None,
            sender: Any | None = None,
            **kwargs: Any,
        ) -> Any:
            # No real LLM available — echo a placeholder so the flow completes.
            return None

    _HAS_AUTOGEN = False


def _last_message_text(messages: Sequence[Mapping[str, Any]] | None) -> str:
    """Extract the most recent user-content string from an AutoGen history."""
    if not messages:
        return ""
    last = messages[-1]
    if isinstance(last, Mapping):
        content = last.get("content", "")
        return content if isinstance(content, str) else str(content)
    return str(last)


class HorizonRouterAgent(_ConversableAgent):
    """
    A ``ConversableAgent`` that routes hard, deep problems to a tool.

    Parameters
    ----------
    name :
        Agent name (AutoGen requirement).
    tool :
        Deterministic solver — a callable taking the problem string. If omitted,
        the agent never delegates (it always uses the LLM branch).
    horizon_model :
        Model identifier for the horizon lookup (default ``"gpt-4o"``).
    estimated_depth, task_type :
        Optional fixed depth / task hint; otherwise estimated per message.
    **policy_kwargs :
        Forwarded to the policy (``threshold``, ``tool_available``,
        ``tool_accuracy``, ``margin``). Remaining kwargs (``llm_config`` …) go
        to ``ConversableAgent``.
    """

    def __init__(
        self,
        name: str,
        *,
        tool: Callable[[str], Any] | None = None,
        horizon_model: str = "gpt-4o",
        estimated_depth: int | None = None,
        task_type: str | None = None,
        threshold: float | None = None,
        tool_available: bool | None = None,
        tool_accuracy: float | None = None,
        margin: float | None = None,
        **agent_kwargs: Any,
    ) -> None:
        super().__init__(name=name, **agent_kwargs)
        self.tool = tool
        self.horizon_model = horizon_model
        self.estimated_depth = estimated_depth
        self.task_type = task_type
        # Only forward policy kwargs the caller actually set.
        self.policy_kwargs: dict[str, Any] = {
            k: v
            for k, v in (
                ("threshold", threshold),
                ("tool_available", tool_available),
                ("tool_accuracy", tool_accuracy),
                ("margin", margin),
            )
            if v is not None
        }
        # Record of the last routing decision, for inspection / logging.
        self.last_route: dict[str, Any] | None = None

    # AutoGen reply hook -----------------------------------------------------
    def generate_reply(  # type: ignore[override]
        self,
        messages: Sequence[Mapping[str, Any]] | None = None,
        sender: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Route the latest message before invoking the LLM.

        Past the horizon (and with a tool attached) → return the tool result.
        Otherwise → defer to the base ``ConversableAgent`` LLM reply.
        """
        problem = _last_message_text(messages)
        depth = resolve_depth(
            problem,
            estimated_depth=self.estimated_depth,
            task_type=self.task_type,
        )
        decision = delegation_decision(
            depth, self.horizon_model, **self.policy_kwargs
        )

        delegate = decision.delegate and self.tool is not None
        self.last_route = {
            "routed_to": "tool" if delegate else "neural",
            "estimated_depth": depth,
            "model": self.horizon_model,
            "delegate": decision.delegate,
            "reason": decision.reason,
            "decision": decision,
        }

        if delegate:
            assert self.tool is not None  # narrowed by `delegate`
            return self.tool(problem)

        # Neural branch: hand back to AutoGen's normal reply machinery.
        return super().generate_reply(messages=messages, sender=sender, **kwargs)

    # Convenience ------------------------------------------------------------
    def route_for(self, problem: str) -> dict[str, Any]:
        """Compute (without executing) the route for a problem string."""
        depth = resolve_depth(
            problem,
            estimated_depth=self.estimated_depth,
            task_type=self.task_type,
        )
        decision = delegation_decision(
            depth, self.horizon_model, **self.policy_kwargs
        )
        delegate = decision.delegate and self.tool is not None
        return {
            "routed_to": "tool" if delegate else "neural",
            "estimated_depth": depth,
            "delegate": decision.delegate,
            "reason": decision.reason,
        }


__all__ = ["HorizonRouterAgent"]
