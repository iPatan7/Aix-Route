"""Tests for the AutoGen ``HorizonRouterAgent``.

Run without ``pyautogen`` installed (fallback ConversableAgent stand-in).
"""

from __future__ import annotations

from integrations.autogen import HorizonRouterAgent


def _msg(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


def test_constructs_with_llm_config() -> None:
    # Acceptance: HorizonRouterAgent(name=..., llm_config={...}) must construct.
    agent = HorizonRouterAgent(name="router", llm_config={"model": "gpt-4o"})
    assert agent.name == "router"


def test_deep_problem_delegates_to_tool() -> None:
    agent = HorizonRouterAgent(
        name="router",
        tool=lambda problem: "tool-solved",
        horizon_model="gpt-4o",
        estimated_depth=35,
    )
    reply = agent.generate_reply(messages=_msg("apply 35 swaps"))

    assert reply == "tool-solved"
    assert agent.last_route is not None
    assert agent.last_route["routed_to"] == "tool"


def test_shallow_problem_uses_neural_branch() -> None:
    # Shallow → defers to base ConversableAgent.generate_reply (stand-in → None).
    agent = HorizonRouterAgent(
        name="router",
        tool=lambda problem: "tool-solved",
        horizon_model="gpt-4o",
        estimated_depth=5,
    )
    reply = agent.generate_reply(messages=_msg("sort [3,1,2] with 5 swaps"))

    assert reply != "tool-solved"  # did NOT delegate
    assert agent.last_route is not None
    assert agent.last_route["routed_to"] == "neural"


def test_no_tool_never_delegates() -> None:
    # Even past the horizon, with no tool attached the agent stays neural.
    agent = HorizonRouterAgent(name="router", horizon_model="gpt-4o", estimated_depth=99)
    agent.generate_reply(messages=_msg("very deep"))
    assert agent.last_route is not None
    assert agent.last_route["routed_to"] == "neural"


def test_route_for_does_not_execute_tool() -> None:
    calls = {"n": 0}

    def tool(_: str) -> str:
        calls["n"] += 1
        return "x"

    agent = HorizonRouterAgent(
        name="router", tool=tool, horizon_model="gpt-4o", estimated_depth=35
    )
    route = agent.route_for("deep problem")
    assert route["routed_to"] == "tool"
    assert calls["n"] == 0


def test_depth_estimated_when_not_fixed() -> None:
    # No estimated_depth → built-in heuristic from message text.
    agent = HorizonRouterAgent(
        name="router", tool=lambda p: "t", horizon_model="gpt-4o"
    )
    agent.generate_reply(messages=_msg("sort [3,1,2] with 5 swaps"))
    assert agent.last_route is not None
    assert agent.last_route["estimated_depth"] >= 1
