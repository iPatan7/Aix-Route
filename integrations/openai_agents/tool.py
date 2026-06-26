"""
``horizon_tool_use`` — Deterministic Horizon routing for the OpenAI Agents SDK.

Wrap a neural reasoning function and a deterministic tool into a single callable
that decides, per invocation, which one to run. Before the neural branch spends
tokens, ``should_delegate`` is consulted; past the horizon, the deterministic
tool is called instead.

    from integrations.openai_agents import horizon_tool_use

    @horizon_tool_use(tool=bfs_solve, model="gpt-4o")
    def solve(problem: str) -> str:
        '''Neural fallback — only runs when the depth is within horizon.'''
        return llm_solve(problem)

    solve("sort [3,1,2] with 5 swaps")     # shallow → neural
    solve("...35-step permutation...")     # deep    → bfs_solve

When ``openai-agents`` (the ``agents`` package) is installed, the returned
object is also registered via ``function_tool`` so it can be handed straight to
an ``Agent(tools=[...])``. Without the SDK it is a plain callable — fully
usable and testable.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from integrations._core import delegation_decision
from integrations._depth import resolve_depth

# --- Optional openai-agents SDK ---------------------------------------------
try:  # pragma: no cover - exercised only when openai-agents is installed
    from agents import function_tool as _function_tool

    _HAS_AGENTS_SDK = True
except Exception:  # noqa: BLE001 - openai-agents is optional
    _function_tool = None
    _HAS_AGENTS_SDK = False


def _first_str_arg(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Find the problem text among positional/keyword arguments."""
    for key in ("problem", "input", "question", "text", "query"):
        if key in kwargs and kwargs[key] is not None:
            return str(kwargs[key])
    for value in args:
        if isinstance(value, str):
            return value
    if args:
        return str(args[0])
    return ""


class HorizonRoutedTool:
    """
    A callable that routes between a neural fn and a deterministic tool.

    Calling the instance runs whichever branch the policy selects. The most
    recent decision is stored on :attr:`last_route` for logging/inspection.
    The optional :attr:`as_function_tool` is the SDK-registered version (or
    ``None`` when ``openai-agents`` is not installed).
    """

    def __init__(
        self,
        neural_fn: Callable[..., Any],
        tool: Callable[..., Any],
        *,
        model: str = "gpt-4o",
        estimated_depth: int | None = None,
        task_type: str | None = None,
        **policy_kwargs: Any,
    ) -> None:
        self.neural_fn = neural_fn
        self.tool = tool
        self.model = model
        self.estimated_depth = estimated_depth
        self.task_type = task_type
        self.policy_kwargs = policy_kwargs
        self.last_route: dict[str, Any] | None = None
        functools.update_wrapper(self, neural_fn)

        # Register with the SDK when available, so the wrapper is also a tool.
        self.as_function_tool = (
            _function_tool(self._call) if _HAS_AGENTS_SDK else None  # type: ignore[misc]
        )

    def _decide(self, problem: str) -> Any:
        depth = resolve_depth(
            problem,
            estimated_depth=self.estimated_depth,
            task_type=self.task_type,
        )
        decision = delegation_decision(depth, self.model, **self.policy_kwargs)
        self.last_route = {
            "routed_to": "tool" if decision.delegate else "neural",
            "estimated_depth": depth,
            "model": self.model,
            "delegate": decision.delegate,
            "reason": decision.reason,
            "decision": decision,
        }
        return decision

    def _call(self, *args: Any, **kwargs: Any) -> Any:
        problem = _first_str_arg(args, kwargs)
        decision = self._decide(problem)
        if decision.delegate:
            return self.tool(*args, **kwargs)
        return self.neural_fn(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._call(*args, **kwargs)

    def route_for(self, problem: str) -> dict[str, Any]:
        """Compute (without executing) the route for a problem string."""
        self._decide(problem)
        assert self.last_route is not None
        return dict(self.last_route)


def horizon_tool_use(
    tool: Callable[..., Any],
    *,
    model: str = "gpt-4o",
    estimated_depth: int | None = None,
    task_type: str | None = None,
    **policy_kwargs: Any,
) -> Callable[[Callable[..., Any]], HorizonRoutedTool]:
    """
    Decorator factory: wrap a neural function with horizon-based routing.

    Parameters
    ----------
    tool :
        Deterministic solver to delegate to past the horizon.
    model :
        Model identifier for the horizon lookup (default ``"gpt-4o"``).
    estimated_depth, task_type :
        Optional fixed depth / task hint; otherwise estimated per call.
    **policy_kwargs :
        Forwarded to the policy (``threshold``, ``tool_available``,
        ``tool_accuracy``, ``margin``).

    Returns
    -------
    Callable
        A decorator that turns the neural function into a
        :class:`HorizonRoutedTool`.
    """

    def decorator(neural_fn: Callable[..., Any]) -> HorizonRoutedTool:
        return HorizonRoutedTool(
            neural_fn,
            tool,
            model=model,
            estimated_depth=estimated_depth,
            task_type=task_type,
            **policy_kwargs,
        )

    return decorator


__all__ = ["horizon_tool_use", "HorizonRoutedTool"]
