# Benchmarks — Validate on Your Stack in 30 Minutes

The full Deterministic Horizon sweep (500 instances × 3 runs × 12 models)
costs thousands of dollars. The **quick** benchmark answers the one question
most practitioners actually have — *does `d*` land where the paper says on the
pipeline running on my machine?* — in minutes, for under \$50 (or free).

## Run it

```bash
# Free, no API key: validate the fitting pipeline against the paper's own curve.
python -m benchmarks.quick --model gpt-4o --task permutation --n 50

# Real evaluation against the model API (needs OPENAI_API_KEY etc., costs money).
python -m benchmarks.quick --model gpt-4o --task permutation --n 50 --real
```

Example output:

```
[MATCH] gpt-4o / permutation (simulated, n=50)
  recovered d* = 22.3   (paper d* = 22.0, |Δ| = 0.3 ≤ 3.0?)
  fit R²       = 0.991
```

Exit code is `0` on a match, `1` on a mismatch, `2` on a usage error — so you
can gate CI on it. Add `--json` for machine-readable output.

## What it does

1. Generates `--n` PermutationProbe instances spread across a depth grid
   (BFS-optimal depths, seeded for reproducibility).
2. **Simulated mode (default):** draws correctness for each instance from the
   paper's closed-form decay curve
   (`deterministic_horizon.policy.expected_neural_accuracy`) for the chosen
   model. **`--real`:** calls the actual model API (C1, neural CoT) via
   `deterministic_horizon.evaluate`.
3. Refits the horizon with `deterministic_horizon.metrics.estimate_horizon` and
   compares the recovered `d*` to the paper's reference (22.0 for `gpt-4o`;
   the tabulated `MODEL_HORIZONS` value otherwise) within `--tolerance`
   (default ±3).

Simulated mode validates that the **generation → fitting** pipeline is sound
and reproduces the known horizon. `--real` validates the **model** itself.

## Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | `gpt-4o` | Model identifier (known to `MODEL_HORIZONS` for simulated mode). |
| `--task` | `permutation` | `permutation` \| `fsa` \| `arithmetic`. |
| `--n` | `50` | Instances generated, spread across the depth grid. |
| `--seed` | `42` | Seed for generation and simulation (reproducible). |
| `--threshold` | `0.5` | Accuracy threshold defining `d*`. |
| `--tolerance` | `3.0` | Allowed `\|recovered d* − paper d*\|` for a match. |
| `--real` | off | Hit the model API instead of simulating (requires keys). |
| `--json` | off | Emit a JSON object instead of the text report. |

## Cost & time

| Mode | API keys | Cost | Wall-clock |
|------|----------|------|------------|
| simulated (default) | none | \$0 | seconds |
| `--real`, `n=50`, 1 model | provider key | < \$50 | minutes |

No keys are read in simulated mode. With `--real`, keys are read from the
environment / `.env`; none are ever written to disk or logged.
