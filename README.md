<div align="center">

# Aix-Route

### A production routing layer that decides when an agent should reason with an LLM and when it should delegate to a tool

Built on the Deterministic Horizon result: extended chain-of-thought stops helping past a measurable depth, and beyond it tool delegation becomes *necessary*.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](https://www.python.org/downloads/)

<br/>

</div>

---

## Headline findings

| Metric | Value | Why it matters |
|---|---|---|
| Deterministic Horizon $d^*$ | **19–31 steps** | Beyond this depth, neural CoT accuracy < 50%. |
| Tool-integrated accuracy | **86–94%** | Across 8 task domains and 12 models. |
| Neural CoT accuracy | **24–42%** | The same tasks, no tools. |
| Cross-model correlation $r$ | **0.81–0.91** | Models from 6 orgs fail on the *same* instances ⇒ architectural, not training-specific. |
| Fine-tuning recovery | **+3.2%** | Theorem 4.10 predicts < 5%; the competing theory predicts > 30%. |
| Cost efficiency (tool vs. CoT) | **4.2–4.7×** | Lower cost-per-correct-solution. |
| Decoherence-model fit | **R² = 0.96** | Super-exponential decay beats linear (0.71) and exponential (0.83). |

---

## Python API

```python
from deterministic_horizon import (
    PermutationTask, generate_instances, evaluate,
    estimate_horizon, fit_decoherence_model,
    should_delegate, should_delegate_batch, delegation_decision,
    horizon_table, recommend_model,
)

# 1. Generate BFS-optimal-depth instances (depth == true BFS optimum)
task = PermutationTask(n_elements=8, seed=42)          # S_8, diameter C(8,2)=28
instances = task.generate_instances(n_instances=500, min_depth=5, max_depth=28)

# 2. Evaluate a model (needs an API key in .env)
results = evaluate(model="gpt-4o", instances=instances, conditions=["C1", "C3"])

# 3. Estimate the horizon (super-exponential fit of Theorem 4.2)
horizon = estimate_horizon(results, threshold=0.5)
print(f"d* = {horizon['d_star']:.1f}  (R² = {horizon['r_squared']:.3f})")

# 4. Route in your own agent
should_delegate(estimated_depth=horizon['d_star'] + 5, model="gpt-4o")   # → True

# 5. Plan a whole decomposition at once, or pick the right model for a depth
should_delegate_batch([5, 8, 35], model="gpt-4o")   # → [False, False, True]
recommend_model(estimated_depth=18)                  # → least over-powered model that still clears 50%
horizon_table()                                       # → per-model d* / ε₀ / L_eff rows (sorted) — the source for `dh horizons`
```

### The five experimental conditions

| Condition | Description |
|---|---|
| **C1** | Neural chain-of-thought (standard prompting) |
| **C2** | Depth-limited CoT (oracle optimal length) |
| **C3** | Tool-integrated (BFS / verifier access) |
| **C4** | Length-encouraged prompting ("take as many steps as needed") |
| **C5** | Fine-tuned on optimal-length traces |

---

## What's inside

```
deterministic-horizon/
├── src/
│   ├── policy.py        # should_delegate / delegation_decision  ← the engineering hook
│   ├── tasks/           # PermutationProbe, FSA-Sim, ArithChain, CircuitTrace, CodeProbe (+ BFS oracle)
│   ├── models/          # Uniform interface: OpenAI / Anthropic / DeepSeek / Gemini / Together / local
│   ├── metrics/         # SSJ, SFE, super-exponential horizon fit, bootstrap CIs
│   ├── analysis.py      # Figures + tables (+ plot_model_horizons comparison)
│   ├── runners.py       # High-level evaluate(...) Python API
│   └── cli.py           # evaluate | analyze | delegate | horizons 
├── configs/             # OmegaConf configs (model × task × experiment)
└── tests/               # pytest suite (smoke · metrics · tasks · policy · analysis)
```

