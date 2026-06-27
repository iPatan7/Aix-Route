# aix-route (TypeScript)

> Know when your agent should stop thinking and call a tool.

Pure, deterministic **Deterministic Horizon** routing for TypeScript/JavaScript.
At a given estimated reasoning depth, decide whether an agent should keep
reasoning with the LLM or **delegate to a tool** — based on measured per-model
horizons.

This is the TypeScript port of the Python [`aix-route`](https://github.com/iPatan7/Aix-Route)
policy. Same math, verified to match the Python reference to 1e-7 across 70
(model, depth) pairs. The only dependency is `@noble/hashes` (used solely by the
offline calibration helper).

## Install

```bash
npm install aix-route
```

## Use

```ts
import { shouldDelegate, delegationDecision, estimateDepth } from "aix-route";

// Boolean route:
shouldDelegate(35, "gpt-4o");          // true  — past the horizon, delegate
shouldDelegate(5,  "gpt-4o");          // false — shallow, keep reasoning

// Full decision with the why:
const d = delegationDecision(8, "llama-3.3-70b-versatile");
// { delegate: true, reason: "tool_dominates_by_margin",
//   expectedNeuralAccuracy: 0.5098, expectedToolAccuracy: 0.92, horizon: 8.2, ... }

// Don't have a depth? Estimate one from the task text:
const depth = estimateDepth("apply 35 sequential swaps");   // 35
shouldDelegate(depth, "llama-3.3-70b-versatile");           // true
```

## What it does

LLMs lose accuracy on deterministic state-tracking as depth grows. Each model
has a **Deterministic Horizon** `d*` — the depth where chain-of-thought accuracy
drops below 50%. Past `d*`, a tool (BFS, verifier, SQL engine, …) is strictly
better. `shouldDelegate` returns `true` when either:

1. expected neural accuracy at this depth is below the threshold (past `d*`), or
2. the tool beats neural reasoning by more than `margin` — don't think harder
   just to break even.

## API

| export | purpose |
|---|---|
| `shouldDelegate(depth, model, opts?)` | boolean route |
| `delegationDecision(depth, model, opts?)` | full decision + reason + accuracies |
| `estimateDepth(task)` | heuristic depth from a task string |
| `accuracyAtDepth(d, eps0, gamma, lEff)` | Theorem 4.2 decay |
| `horizonFor(model)` / `paramsFor(model)` | per-model horizon / params |
| `MODEL_PARAMS` | the calibrated model registry |
| `fitHorizon(depths, accs)` | fit `(eps0, d*)` from measured data (offline) |
| `empiricalDStar(depths, accs)` | model-free 50%-crossing |
| `computeCalibrationHash(...)` | deterministic calibration fingerprint |

`opts`: `{ threshold = 0.5, toolAvailable = true, toolAccuracy = 0.92, margin = 0.10 }`.

## Calibrating your own models

Measure offline with the Python `aix-route calibrate` CLI, then add the
`(eps0, d*)` to `MODEL_PARAMS`. `fitHorizon` is the in-JS twin of the Python
fit — both reject out-of-regime data (`eps0·d* ≥ ln(1/α)`) rather than producing
a negative `lEff`.

## License

MIT. Routing math from the Deterministic Horizon paper
([bettyguo/deterministic-horizon](https://github.com/bettyguo/deterministic-horizon)).
