"""Tests for the LangChain ``HorizonRouterNode``.

These run without ``langchain-core`` installed (the node falls back to a shim)
and without any API key (a failing neural branch is captured, not raised).
"""

from __future__ import annotations

from integrations.langchain import HorizonRouterNode


def test_shallow_problem_routes_neural() -> None:
    # Fixed shallow depth → neural branch (depth 5 < d*=22 for gpt-4o).
    llm = lambda problem: "neural-answer"  # noqa: E731 - stand-in LLM
    tool = lambda problem: "tool-answer"  # noqa: E731
    node = HorizonRouterNode(llm=llm, tool=tool, model="gpt-4o", estimated_depth=5)

    result = node.invoke({"problem": "sort [3,1,2] with 5 swaps"})

    assert result["routed_to"] == "neural"
    assert result["result"] == "neural-answer"
    assert result["error"] is None
    assert result["estimated_depth"] == 5


def test_deep_problem_routes_tool() -> None:
    # Fixed deep depth → tool branch (depth 35 > d*=22 for gpt-4o).
    llm = lambda problem: "neural-answer"  # noqa: E731
    tool = lambda problem: "tool-answer"  # noqa: E731
    node = HorizonRouterNode(llm=llm, tool=tool, model="gpt-4o", estimated_depth=35)

    result = node.invoke({"problem": "apply 35 sequential swaps"})

    assert result["routed_to"] == "tool"
    assert result["result"] == "tool-answer"


def test_acceptance_snippet_contract() -> None:
    # Mirrors the foundation prompt's acceptance snippet (with a stand-in LLM
    # so it runs offline). The contract: routed_to is always neural|tool.
    llm = lambda x: "solved"  # noqa: E731 - stand-in for ChatOpenAI
    tool = lambda x: "solved"  # noqa: E731 - stand-in for real solver
    node = HorizonRouterNode(llm=llm, tool=tool)

    result = node.invoke({"problem": "sort [3,1,2] with 5 swaps"})

    assert result["routed_to"] in ["neural", "tool"]


def test_neural_branch_error_is_captured_not_raised() -> None:
    # A neural branch that explodes (e.g. missing API key) must not break
    # routing — the routing decision is the integration's contract.
    def exploding_llm(_: object) -> str:
        raise RuntimeError("no API key")

    node = HorizonRouterNode(
        llm=exploding_llm,
        tool=lambda x: "tool",
        model="gpt-4o",
        estimated_depth=5,
    )
    result = node.invoke({"problem": "x"})

    assert result["routed_to"] == "neural"
    assert result["result"] is None
    assert "RuntimeError" in result["error"]


def test_should_delegate_preview_does_not_invoke_branches() -> None:
    called = {"llm": False, "tool": False}

    def llm(_: object) -> str:
        called["llm"] = True
        return "n"

    def tool(_: object) -> str:
        called["tool"] = True
        return "t"

    node = HorizonRouterNode(llm=llm, tool=tool, model="gpt-4o", estimated_depth=35)
    assert node.should_delegate({"problem": "deep"}) is True
    assert called == {"llm": False, "tool": False}


def test_extracts_problem_from_plain_string() -> None:
    node = HorizonRouterNode(
        llm=lambda p: f"neural:{p}",
        tool=lambda p: f"tool:{p}",
        model="gpt-4o",
        estimated_depth=5,
    )
    result = node.invoke("just a string problem")
    assert result["result"] == "neural:just a string problem"
