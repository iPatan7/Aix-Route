# Deterministic Horizon — Framework Integrations

Drop-in **depth-aware routing** for popular agent frameworks. Before your agent
spends tokens reasoning through a hard, deep problem, the integration asks the
Deterministic Horizon policy layer one question:

> At this estimated reasoning depth, will the model still be above 50% accuracy —
> or is it past the **Deterministic Horizon** and better off calling a tool?

If it's past the horizon, the integration routes to your deterministic tool
(BFS, verifier, SQL engine, …) instead of the LLM. Same decision logic as
`should_delegate()` from `src/policy.py`, wired into each framework's native
extension point.

All three integrations are **importable and testable without their framework
SDK installed** — each falls back to a minimal stand-in. Install the SDK only
for the framework you actually use.

---

## Install

```bash
# Core (required) — the policy layer:
pip install -e .              # from the repo root

# Then ONE of, matching your stack:
pip install langchain-core langchain-openai   # LangChain
pip install pyautogen                          # AutoGen
pip install openai-agents                      # OpenAI Agents SDK
```

The integrations import `should_delegate` from the installed package, or fall
back to `src/policy.py` directly — so they work in a bare source checkout too.

---

## LangChain — `HorizonRouterNode`

A real `langchain_core.runnables.Runnable` (composes in LCEL with `|`).

```python
from integrations.langchain import HorizonRouterNode
from langchain_openai import ChatOpenAI

llm  = ChatOpenAI(model="gpt-4o")
tool = lambda problem: bfs_solve(problem)        # deterministic solver

node = HorizonRouterNode(llm=llm, tool=tool, model="gpt-4o")
out  = node.invoke({"problem": "sort [3,1,2] with 5 swaps"})

out["routed_to"]   # "neural" (within horizon) | "tool" (past it)
out["result"]      # branch output
out["reason"]      # policy reason, e.g. "below_horizon" / "above_horizon"
```

The node accepts a dict (`{"problem": ...}`, `{"input": ...}`, `{"question": ...}`)
or a plain string. A neural branch that raises (e.g. missing API key) is
captured into `out["error"]` rather than crashing the router — the routing
decision is the contract.

---

## AutoGen — `HorizonRouterAgent`

Subclasses `ConversableAgent` and overrides `generate_reply()`.

```python
from integrations.autogen import HorizonRouterAgent

agent = HorizonRouterAgent(
    name="router",
    llm_config={"model": "gpt-4o"},
    tool=lambda problem: bfs_solve(problem),     # deterministic solver
    horizon_model="gpt-4o",
)

# On each turn: estimate depth → past horizon? → tool, else normal LLM reply.
agent.last_route   # {"routed_to": ..., "estimated_depth": ..., "reason": ...}
```

With no `tool` attached the agent always uses the LLM branch. Without
`pyautogen` installed it falls back to a stand-in base so it stays importable.

---

## OpenAI Agents SDK — `horizon_tool_use`

Decorator that wraps a neural function; delegates to a tool past the horizon.

```python
from integrations.openai_agents import horizon_tool_use

@horizon_tool_use(tool=bfs_solve, model="gpt-4o")
def solve(problem: str) -> str:
    """Neural fallback — only runs when depth is within horizon."""
    return llm_solve(problem)

solve("sort [3,1,2] with 5 swaps")     # shallow → neural
solve("…35-step permutation…")         # deep    → bfs_solve
solve.last_route                       # inspect the routing decision
```

When `openai-agents` is installed, `solve.as_function_tool` is the
SDK-registered `function_tool`, ready for `Agent(tools=[solve.as_function_tool])`.

---

## Depth estimation

If you don't pass `estimated_depth`, the integration estimates it:

1. Agent C's `src/estimators.estimate_depth` when that slice is installed.
2. Otherwise a built-in token-counting heuristic (numbers + step words +
   operators). Crude but enough to land on the right side of the horizon for
   obvious cases.

Pass `estimated_depth=<int>` (and optional `task_type=...`) to override.

---

## Tuning the policy

Every integration forwards policy keyword arguments straight through to
`should_delegate` / `delegation_decision`:

| kwarg            | meaning                                              |
|------------------|------------------------------------------------------|
| `threshold`      | neural-accuracy floor below which we delegate (0.5)  |
| `margin`         | delegate when tool beats neural by more than this    |
| `tool_available` | force-disable delegation when `False`                |
| `tool_accuracy`  | empirical accuracy of your tool (default 0.92)       |

```python
HorizonRouterNode(llm=llm, tool=tool, model="o3-mini", threshold=0.6, margin=0.05)
```

---

## Testing

```bash
# From the repo root, with src/ importable:
PYTHONPATH=. pytest integrations/tests/ -v
```

Runs fully offline (no SDKs, no API keys). The live-server check in
`test_live_server.py` is skipped unless Agent A's router is reachable at
`HORIZON_ROUTER_URL` (default `http://localhost:8000`).
```
