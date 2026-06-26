# Calibrating Deterministic Horizons for *your* models

Aix-Route routes an agent step to a tool once the estimated reasoning depth
passes a model's **Deterministic Horizon** `d*` — the depth where neural
chain-of-thought accuracy drops through 50%. The horizons in
`src/policy.py` are calibrated for the paper's local/HF setup. A model served
elsewhere (Groq, a different quantization, a different prompt) can sit at a
different `d*`, so the honest move is to **measure it yourself**, not inherit the
paper's number.

This page explains the method. For the raw curves we already measured on Groq,
see [`../benchmarks/groq_calibration.md`](../benchmarks/groq_calibration.md).

---

## The model

Theorem 4.2's closed-form accuracy decay:

```
P(correct at depth d) ≈ exp( −d·ε₀ − γ·d(d+1) / (2·L_eff) )
```

- `ε₀`     — baseline per-step error
- `γ`      — shared attention-decay constant (`policy.GAMMA = 0.15`)
- `L_eff`  — effective decoherence length (≈10² steps, **not** the context window)

`d*` (Theorem 4.8) is the depth where `P = α` (α = 0.5):

```
d* = ( −ε₀·L_eff + √(ε₀²·L_eff² + 2γ·L_eff·ln(1/α)) ) / γ
```

The policy stores only `(ε₀, d*)` per model and **derives** `L_eff` from them
(`policy._l_eff_for`), so there is a single source of truth and the decay curve
crosses 0.5 exactly at `d*`.

---

## Measure → fit → derive

`src/calibrate.py` does the inverse problem: given measured accuracy-vs-depth,
recover the parameters.

### 1. Measure

Run the model on BFS-optimal-depth instances at several depths, `temperature=0`,
condition C1 (neural CoT). Accuracy = exact final-state match.

### 2. Fit — `fit_horizon(depths, accuracies)`

Linear least-squares in `ln P`:

```
ln P = (−d)·ε₀  +  (−γ·d(d+1)/2)·(1/L_eff)
```

Only interior points (`0 < acc < 1`) are used — saturated 0/1 points carry no
curve information and break the log transform.

**The fit returns `None` (out of regime) when:**

- fewer than two interior points,
- the slope implies a non-decay (`ε₀ < 0` or `1/L_eff ≤ 0`),
- the discriminant is negative, or
- **the constraint `ε₀·d* < ln(1/α)` is violated** — this is what keeps `L_eff`
  positive in the policy layer. A pair like `ε₀=0.5, d*=3` (⇒ `ε₀·d* = 1.5 >
  ln2 ≈ 0.693`) is rejected here, *before* it can reach `policy._l_eff_for` and
  crash with a negative `L_eff`.

A rejected fit is a signal to fall back (empirical d*, or flag as
always-delegate) — **never** to invent numbers.

### 3. Fall back — `empirical_d_star(depths, accuracies)`

When the parametric fit is rejected but accuracy clearly crosses 50% (e.g. a
plateau then a sharp cliff, which the smooth model can't capture), read off the
crossing depth by linear interpolation between the bracketing points. This is
what we used for the two gpt-oss models.

---

## CLI

```bash
# cloud model (needs the provider key in .env):
export GROQ_API_KEY=gsk-...
aix-route calibrate --model llama-3.3-70b-versatile --depths 3,5,8,10,12,15,18 --n 10

# local model (needs `ollama serve` running):
aix-route calibrate --model qwen2.5:7b --depths 3,5,8,10,12 --n 10
```

Output: an accuracy-vs-depth table, the parametric fit (or the empirical d*
fallback), and the `(ε₀, d*)` tuple to paste into `policy._MODEL_EPS0_DSTAR`:

```
Parametric fit: eps0=0.0703  d*=8.20  L_eff=43.18  R²=0.747  (n_points=4)
policy tuple:  "llama-3.3-70b-versatile": (0.070, 8.2),
```

`python -m deterministic_horizon.cli calibrate ...` is equivalent if the package
isn't on your PATH. `scripts/measure.py` is the same routine as a standalone
script (handy for redirecting raw output to a file).

---

## Worked example — Groq, measured

| model | measured d* | how |
|---|---|---|
| llama-3.3-70b-versatile | 8.2 | parametric fit, R²=0.75 |
| gpt-oss-20b | 15.5 | empirical 50%-crossing (fit rejected — cliff too sharp) |
| gpt-oss-120b | 15.0 | empirical 50%-crossing (fit rejected) |
| llama-3.1-8b-instant | < 5 | 0% at depth 5 → too weak, always delegate (shipped as d*=3) |

These sit below the paper's local/HF numbers — expected for served/quantized
models. The router should reflect the infrastructure it runs on.

### Known limitation: plateau vs fit

The super-exponential fits the *collapse* but underestimates the *plateau*: a
real curve stays ~100% to a knee, then drops; the smooth model can't be both
100% at d5 and 50% at d8, so it predicts ~67% at d5. At shallow depths the
router may therefore **over-delegate** via the margin rule. This is conservative
and safe — the tool is exact, so over-delegating costs latency/tokens, never
correctness. A plateau+cliff two-segment fit would tighten this; left for a
future pass.

---

## Re-calibrating after a change

Recalibrate whenever you change provider, quantization, prompt template, or task
— any of these moves `d*`. Drop the new `(ε₀, d*)` tuple into
`policy._MODEL_EPS0_DSTAR`; `MODEL_HORIZONS`, `horizon_table`, the `/horizons`
endpoint, and the explorer all pick it up automatically (refresh
`tests/snapshots/model_horizons.json` and `docs/index.html` to match — the test
suite guards both).
```
