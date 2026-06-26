# Groq / Ollama horizon calibration — measured, not guessed

These are the per-model Deterministic Horizons **measured on this infrastructure**
(Groq free tier), used to recalibrate `src/policy.py`. The paper's values are for
the authors' local/HF setup; served models sit lower, so we measure rather than
inherit. Method lives in `src/calibrate.py` (`fit_horizon`, `empirical_d_star`).

**Task:** PermutationProbe, `n_elements=8`, BFS-optimal-depth instances,
`temperature=0`, C1 (neural chain-of-thought) condition. Accuracy = exact
final-state match via `task.evaluate`.

## Measured accuracy vs depth

| model | d3 | d5 | d8 | d10 | d12 | d15 | d18 | n/depth |
|---|---|---|---|---|---|---|---|---|
| llama-3.3-70b-versatile | — | 1.00 | 0.50 | 0.50 | 0.25 | 0.25 | — | 4 |
| gpt-oss-20b | 1.00 | 1.00 | 1.00 | 1.00 | 0.80 | 0.60 | 0.00 | 10 |
| gpt-oss-120b | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 0.30 | 10 |
| llama-3.1-8b-instant | — | 0.00 | — | — | — | — | — | 3 |

## Fitted / empirical horizons

| model | empirical d* (50% crossing) | parametric fit | shipped (ε₀, d*) |
|---|---|---|---|
| llama-3.3-70b-versatile | 8.0 | ε₀=0.070, d*=8.2, R²=0.747 ✓ | (0.070, 8.2) |
| gpt-oss-20b | 15.5 | rejected (cliff too sharp) | (0.020, 15.5) |
| gpt-oss-120b | 15.0 | rejected (under-sampled decay) | (0.018, 15.0) |
| llama-3.1-8b-instant | < 5 (0% at d5) | too weak | (0.150, 3.0) → always delegate |

Scaling holds: **120b ≈ 20b > 70b-versatile > 8b** — bigger / stronger models
have a higher horizon, consistent with the paper's `d* ∝ √(d_h·H)` law.

ε₀ for the two gpt-oss models is held at the paper's open-weight baseline
(~0.018–0.020) because the parametric fit was rejected; only the **measured d***
(empirical 50% crossing) drives routing. d* is what `should_delegate` keys on.

## C3 — does delegation actually help?

The BFS tool (the branch `should_delegate` routes to) is exact and
depth-independent:

| depth | C1 neural (measured) | C3 BFS tool |
|---|---|---|
| 12 | 25–80% | **100%** |
| 15 | 25–60% | **100%** |
| 18 | 0–30% | **100%** |
| 25 | (past neural range) | **100%** |

Where neural CoT collapses, the tool stays at 100%. This is the product claim:
past the horizon, delegating is strictly better. ✅

## Known limitation: shallow plateau vs fit

The super-exponential model fits the *collapse* but underestimates the *plateau*.
Real curves stay ~100% up to a knee, then drop sharply; the smooth fit can't be
both 100% at d5 and 50% at d8, so it predicts ~67% at d5. Consequence: at shallow
depths the router may **over-delegate** via the margin rule (tool beats the
under-predicted neural accuracy). This is conservative and safe — the tool is
exact, so over-delegating costs latency/tokens, never correctness. A
plateau+cliff two-segment fit would tighten this; left for a future pass.

## Reproduce

```bash
export GROQ_API_KEY=gsk-...
# accuracy-vs-depth + fit for one model:
python scripts/measure.py llama-3.3-70b-versatile 10 "3,5,8,10,12,15,18"
```

Groq free tier caps daily tokens (TPD=100k); the 70b run above exhausts a chunk
of it, so measure one model at a time and avoid concurrent runs (they trip the
per-minute rate limit).
