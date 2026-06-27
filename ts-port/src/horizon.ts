// Aix-Route — Deterministic Horizon routing (TypeScript port)
// https://github.com/iPatan7/Aix-Route
// Math from the Deterministic Horizon paper (bettyguo/deterministic-horizon).
// License: MIT
//
// ============================================================
//  THE DETERMINISTIC HORIZON — Bounded Reasoning Depth
//
//  Pure, deterministic, dependency-free routing math: at a given
//  estimated reasoning depth, should an agent keep thinking with
//  the LLM, or delegate to a tool? A faithful port of the Python
//  `aix_route` policy (paper Theorems 4.2 & 4.8) — every function
//  here matches it to within floating-point (verified on 70
//  (model, depth) pairs).
// ============================================================

/** Shared attention-decay constant (paper GAMMA). */
export const GAMMA = 0.15;

/** Accuracy threshold that defines d* (alpha). */
export const ALPHA_DEFAULT = 0.5;

/** Cross-domain mean tool-integrated (C3) accuracy (paper §5). */
export const DEFAULT_TOOL_ACCURACY = 0.92;

/** Per-model decoherence parameters. */
export interface HorizonParams {
  /** Baseline per-step error. */
  eps0: number;
  /** Deterministic Horizon — depth where accuracy crosses alpha. */
  dStar: number;
  /** Effective decoherence length (derived from eps0 & dStar). */
  lEff: number;
  /** Shared attention-decay constant. */
  gamma: number;
}

/**
 * Effective decoherence length L_eff that makes the Theorem 4.2 decay curve
 * pass through accuracy `alpha` at depth `dStar`. Mirrors
 * `deterministic_horizon.policy._l_eff_for`.
 *
 *   L_eff = gamma * d* (d* + 1) / (2 * (ln(1/alpha) - eps0 * d*))
 *
 * Throws when `eps0 * dStar >= ln(1/alpha)` (would make L_eff non-positive) —
 * the same constraint enforced by the calibration layer.
 */
export function lEffFor(eps0: number, dStar: number, alpha: number = ALPHA_DEFAULT): number {
  const denom = Math.log(1 / alpha) - eps0 * dStar;
  if (denom <= 0) {
    throw new Error(
      `eps0*dStar must be below ln(1/alpha); got eps0=${eps0}, dStar=${dStar}, alpha=${alpha}`,
    );
  }
  return (GAMMA * dStar * (dStar + 1)) / (2 * denom);
}

/**
 * Expected accuracy of a neural chain-of-thought at depth `d` (Theorem 4.2):
 *
 *   P(correct) = exp(-d*eps0 - gamma*d*(d+1) / (2*lEff))
 *
 * Mirrors `deterministic_horizon.policy.expected_neural_accuracy`.
 */
export function accuracyAtDepth(
  d: number,
  eps0: number,
  gamma: number,
  lEff: number,
): number {
  if (d < 0) throw new Error(`depth must be non-negative, got ${d}`);
  return Math.exp(-d * eps0 - (gamma * d * (d + 1)) / (2 * lEff));
}

/**
 * Deterministic Horizon d* from (eps0, lEff) (Theorem 4.8):
 *
 *   d* = (-eps0*lEff + sqrt(eps0^2*lEff^2 + 2*gamma*lEff*ln(1/alpha))) / gamma
 */
export function computeHorizon(
  eps0: number,
  lEff: number,
  gamma: number = GAMMA,
  alpha: number = ALPHA_DEFAULT,
): number {
  const disc = eps0 * eps0 * lEff * lEff + 2 * gamma * lEff * Math.log(1 / alpha);
  if (disc < 0) return NaN;
  return (-eps0 * lEff + Math.sqrt(disc)) / gamma;
}

/** Build a complete HorizonParams from a measured (eps0, dStar) pair. */
export function paramsFromCalibration(eps0: number, dStar: number): HorizonParams {
  return { eps0, dStar, lEff: lEffFor(eps0, dStar), gamma: GAMMA };
}

// ------------------------------------------------------------
//  Model registry — Groq-calibrated values measured offline with
//  Aix-Route (PermutationProbe, temperature 0). These must stay in
//  exact agreement with deterministic_horizon.policy._MODEL_EPS0_DSTAR.
//  Stored as (eps0, dStar); lEff is derived, single source of truth.
// ------------------------------------------------------------

const RAW_MODEL_EPS0_DSTAR: Record<string, [number, number]> = {
  // Paper-canonical / surveyed models.
  "gpt-4o": [0.02, 22.0],
  "claude-4.5-opus": [0.018, 27.0],
  "o3-mini": [0.014, 31.0],
  "deepseek-r1": [0.015, 29.0],
  "llama-3.1-8b": [0.022, 20.0],
  "llama-3.3-70b": [0.018, 28.0],
  "qwen-2.5-7b": [0.023, 19.0],
  "qwen-2.5-72b": [0.018, 28.0],
  // Groq — measured on served infra (Aix-Route benchmarks/groq_calibration.md).
  "llama-3.3-70b-versatile": [0.07, 8.2],
  "gpt-oss-20b": [0.02, 15.5],
  "gpt-oss-120b": [0.018, 15.0],
  "llama-3.1-8b-instant": [0.15, 3.0],
  // Ollama local tags (paper Qwen-2.5 placeholder; recalibrate per machine).
  "qwen2.5:1.5b": [0.026, 16.0],
  "qwen2.5:7b": [0.023, 19.0],
  // Cross-model fallback.
  default: [0.02, 24.0],
};

export const MODEL_PARAMS: Record<string, HorizonParams> = Object.fromEntries(
  Object.entries(RAW_MODEL_EPS0_DSTAR).map(([name, [eps0, dStar]]) => [
    name,
    paramsFromCalibration(eps0, dStar),
  ]),
);

/** Look up a model's params, falling back to "default". */
export function paramsFor(model: string): HorizonParams {
  return MODEL_PARAMS[model.toLowerCase()] ?? MODEL_PARAMS["default"];
}

/** Deterministic Horizon d* for a model (or the default). */
export function horizonFor(model: string): number {
  return paramsFor(model).dStar;
}

// ------------------------------------------------------------
//  The delegation decision.
// ------------------------------------------------------------

export type DelegationReason =
  | "above_horizon"
  | "below_horizon"
  | "tool_unavailable"
  | "tool_dominates_by_margin";

export interface DelegationDecision {
  delegate: boolean;
  /** Confidence in the recommended branch, in [0, 1]. */
  confidence: number;
  reason: DelegationReason;
  estimatedDepth: number;
  model: string;
  expectedNeuralAccuracy: number;
  expectedToolAccuracy: number;
  horizon: number;
}

export interface DelegateOptions {
  threshold?: number;
  toolAvailable?: boolean;
  toolAccuracy?: number;
  margin?: number;
}

/**
 * Full delegation decision. Mirrors
 * `deterministic_horizon.policy.delegation_decision` exactly, including the
 * margin rule (delegate when the tool beats neural by more than `margin`, even
 * below the horizon).
 */
export function delegationDecision(
  estimatedDepth: number,
  model: string,
  opts: DelegateOptions = {},
): DelegationDecision {
  const threshold = opts.threshold ?? 0.5;
  const margin = opts.margin ?? 0.1;
  const toolAvailable = opts.toolAvailable ?? true;
  const toolAccuracy = opts.toolAccuracy ?? DEFAULT_TOOL_ACCURACY;

  if (!(threshold > 0 && threshold < 1)) throw new Error(`threshold must be in (0,1), got ${threshold}`);
  if (!(margin >= 0 && margin < 1)) throw new Error(`margin must be in [0,1), got ${margin}`);

  const p = paramsFor(model);
  const neural = accuracyAtDepth(estimatedDepth, p.eps0, p.gamma, p.lEff);
  const tool = toolAvailable ? toolAccuracy : 0;
  const horizon = p.dStar;
  const depthInt = Math.trunc(estimatedDepth);

  if (!toolAvailable) {
    return {
      delegate: false,
      confidence: neural,
      reason: "tool_unavailable",
      estimatedDepth: depthInt,
      model,
      expectedNeuralAccuracy: neural,
      expectedToolAccuracy: 0,
      horizon,
    };
  }

  if (neural < threshold) {
    return {
      delegate: true,
      confidence: tool,
      reason: "above_horizon",
      estimatedDepth: depthInt,
      model,
      expectedNeuralAccuracy: neural,
      expectedToolAccuracy: tool,
      horizon,
    };
  }

  if (tool - neural > margin) {
    return {
      delegate: true,
      confidence: tool,
      reason: "tool_dominates_by_margin",
      estimatedDepth: depthInt,
      model,
      expectedNeuralAccuracy: neural,
      expectedToolAccuracy: tool,
      horizon,
    };
  }

  return {
    delegate: false,
    confidence: neural,
    reason: "below_horizon",
    estimatedDepth: depthInt,
    model,
    expectedNeuralAccuracy: neural,
    expectedToolAccuracy: tool,
    horizon,
  };
}

/** Cheap boolean wrapper around {@link delegationDecision}. */
export function shouldDelegate(
  estimatedDepth: number,
  model: string,
  opts: DelegateOptions = {},
): boolean {
  return delegationDecision(estimatedDepth, model, opts).delegate;
}

/**
 * Whether a model's calibrated horizon satisfies the regime constraint
 * (eps0 * dStar < ln(1/alpha)), i.e. its params yield a positive L_eff.
 * A covenant that fails this must not be sealed.
 */
export function validateCovenantConstraint(
  model: string,
  dStar: number,
  alpha: number = ALPHA_DEFAULT,
): boolean {
  const p = paramsFor(model);
  return p.eps0 * dStar < Math.log(1 / alpha);
}
