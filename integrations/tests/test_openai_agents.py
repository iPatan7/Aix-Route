"""Tests for the OpenAI Agents SDK ``horizon_tool_use`` helper.

Run without ``openai-agents`` installed (plain-callable mode).
"""

from __future__ import annotations

from integrations.openai_agents import HorizonRoutedTool, horizon_tool_use


def test_decorator_routes_shallow_to_neural() -> None:
    @horizon_tool_use(tool=lambda problem: "tool", model="gpt-4o", estimated_depth=5)
    def solve(problem: str) -> str:
        return "neural"

    assert solve("sort [3,1,2] with 5 swaps") == "neural"
    assert solve.last_route is not None
    assert solve.last_route["routed_to"] == "neural"


def test_decorator_routes_deep_to_tool() -> None:
    @horizon_tool_use(tool=lambda problem: "tool", model="gpt-4o", estimated_depth=35)
    def solve(problem: str) -> str:
        return "neural"

    assert solve("apply 35 swaps") == "tool"
    assert solve.last_route is not None
    assert solve.last_route["routed_to"] == "tool"


def test_returns_horizon_routed_tool_instance() -> None:
    @horizon_tool_use(tool=lambda p: "t", model="gpt-4o")
    def solve(problem: str) -> str:
        return "n"

    assert isinstance(solve, HorizonRoutedTool)


def test_preserves_wrapped_metadata() -> None:
    @horizon_tool_use(tool=lambda p: "t", model="gpt-4o", estimated_depth=5)
    def solve(problem: str) -> str:
        """Solve docstring."""
        return "n"

    assert solve.__name__ == "solve"
    assert solve.__doc__ == "Solve docstring."


def test_route_for_preview() -> None:
    @horizon_tool_use(tool=lambda p: "t", model="gpt-4o", estimated_depth=35)
    def solve(problem: str) -> str:
        return "n"

    route = solve.route_for("deep")
    assert route["routed_to"] == "tool"
    assert route["delegate"] is True


def test_heuristic_path_routes_deep_problem_to_tool() -> None:
    # No estimated_depth → depth comes from the built-in heuristic.
    # "35 swaps" must route deep to the tool.
    @horizon_tool_use(tool=lambda p: "bfs", model="gpt-4o")
    def solve(problem: str) -> str:
        return "neural"

    assert solve("sort [3,1,2] with 5 swaps") == "neural"   # shallow
    assert solve("apply 35 sequential swaps") == "bfs"       # deep


def test_kwarg_problem_extraction() -> None:
    @horizon_tool_use(tool=lambda **kw: "tool", model="gpt-4o", estimated_depth=35)
    def solve(problem: str) -> str:
        return "neural"

    assert solve(problem="apply 35 swaps") == "tool"
