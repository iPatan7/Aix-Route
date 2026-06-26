"""
``HorizonRouterNode`` — a LangChain ``Runnable`` that routes by reasoning depth.

Wrap your LLM chain and a deterministic tool. At runtime the node estimates the
problem's reasoning depth, asks the Deterministic Horizon policy whether the
model can be trusted at that depth, and routes to whichever branch wins:

    from integrations.langchain import HorizonRouterNode
    from langchain_openai import ChatOpenAI

    llm  = ChatOpenAI(model="gpt-4o")
    tool = lambda problem: solve_with_bfs(problem)   # deterministic solver

    node = HorizonRouterNode(llm=llm, tool=tool, model="gpt-4o")
    out  = node.invoke({"problem": "sort [3,1,2] with 5 swaps"})
    out["routed_to"]   # "neural" (shallow) or "tool" (past the horizon)

The node is a real ``langchain_core.runnables.Runnable`` when ``langchain-core``
is installed, so it composes inside LCEL pipelines (``|``). Without
``langchain-core`` it falls back to a minimal ``Runnable`` shim exposing the
same ``invoke`` surface, so the node stays importable and testable everywhere.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from integrations._core import delegation_decision, should_delegate
from integrations._depth import resolve_depth

# --- Optional langchain-core base -------------------------------------------
try:  # pragma: no cover - exercised only when langchain-core is installed
    from langchain_core.runnables import Runnable as _Runnable

    _HAS_LANGCHAIN = True
except Exception:  # noqa: BLE001 - langchain-core is optional

    class _Runnable:  # minimal shim: just enough surface for invoke()
        """Fallback base when ``langchain-core`` is not installed."""

        def invoke(self, input: Any, config: Any | None = None) -> Any:  # noqa: A002
            raise NotImplementedError

    _HAS_LANGCHAIN = False


def _extract_problem(payload: Any) -> str:
    """Pull the problem text out of whatever LangChain hands us."""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, Mapping):
        for key in ("problem", "input", "question", "text", "query"):
            if key in payload and payload[key] is not None:
                return str(payload[key])
    return str(payload)


def _invoke_branch(target: Any, payload: Any, problem: str) -> Any:
    """Call an llm/tool branch, accepting both Runnables and plain callables."""
    # A LangChain Runnable (or anything with .invoke) gets the raw payload.
    invoke = getattr(target, "invoke", None)
    if callable(invoke):
        return invoke(payload)
    if callable(target):
        # Plain callables (the README's `lambda x: "solved"`) get the problem
        # text — that is what a standalone solver expects.
        return target(problem)
    raise TypeError(f"Cannot invoke branch of type {type(target)!r}")


class HorizonRouterNode(_Runnable):
    """
    Depth-aware router around an LLM chain and a deterministic tool.

    Parameters
    ----------
    llm :
        The neural branch — a LangChain ``Runnable`` (e.g. ``ChatOpenAI``) or
        any callable taking the problem string.
    tool :
        The deterministic branch — a ``Runnable`` or a callable taking the
        problem string and returning a solution.
    model :
        Model identifier used to look up the horizon (default ``"gpt-4o"``).
    estimated_depth :
        Optional fixed depth. When ``None`` (default) the depth is estimated
        per call from the input.
    task_type :
        Optional task hint forwarded to the depth estimator.
    **policy_kwargs :
        Forwarded to ``should_delegate`` (``threshold``, ``tool_available``,
        ``tool_accuracy``, ``margin``).
    """

    def __init__(
        self,
        llm: Any,
        tool: Callable[..., Any] | Any,
        *,
        model: str = "gpt-4o",
        estimated_depth: int | None = None,
        task_type: str | None = None,
        **policy_kwargs: Any,
    ) -> None:
        self.llm = llm
        self.tool = tool
        self.model = model
        self.estimated_depth = estimated_depth
        self.task_type = task_type
        self.policy_kwargs = policy_kwargs

    # LangChain Runnable contract -------------------------------------------
    def invoke(self, input: Any, config: Any | None = None) -> dict[str, Any]:  # noqa: A002
        """Route ``input`` to the neural or tool branch and return the result.

        Returns a dict with keys: ``routed_to`` (``"neural"``/``"tool"``),
        ``result``, ``estimated_depth``, ``model``, ``delegate``, ``reason``,
        and ``decision`` (the full :class:`DelegationDecision`).
        """
        problem = _extract_problem(input)
        depth = resolve_depth(
            problem,
            estimated_depth=self.estimated_depth,
            task_type=self.task_type,
        )

        decision = delegation_decision(depth, self.model, **self.policy_kwargs)
        routed_to = "tool" if decision.delegate else "neural"
        branch = self.tool if decision.delegate else self.llm

        try:
            result: Any = _invoke_branch(branch, input, problem)
            error: str | None = None
        except Exception as exc:  # noqa: BLE001 - surface, don't crash the router
            # A missing API key on the neural branch must not break routing —
            # the routing decision itself is the integration's contract.
            result = None
            error = f"{type(exc).__name__}: {exc}"

        return {
            "routed_to": routed_to,
            "result": result,
            "error": error,
            "estimated_depth": depth,
            "model": self.model,
            "delegate": decision.delegate,
            "reason": decision.reason,
            "decision": decision,
        }

    # Convenience ------------------------------------------------------------
    def should_delegate(self, problem: Any) -> bool:
        """Boolean route preview without invoking either branch."""
        depth = resolve_depth(
            problem,
            estimated_depth=self.estimated_depth,
            task_type=self.task_type,
        )
        return should_delegate(depth, self.model, **self.policy_kwargs)


__all__ = ["HorizonRouterNode"]
