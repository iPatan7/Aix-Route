// Aix-Route — Deterministic Horizon routing (TypeScript port)
// https://github.com/iPatan7/Aix-Route — License: MIT
//
// Public entry point. Pure, deterministic routing math: decide when an agent
// should keep reasoning with the LLM vs. delegate to a tool, based on the
// Deterministic Horizon. No runtime dependencies except @noble/hashes (used
// only by the offline calibration helper).

export {
  GAMMA,
  ALPHA_DEFAULT,
  DEFAULT_TOOL_ACCURACY,
  MODEL_PARAMS,
  accuracyAtDepth,
  computeHorizon,
  delegationDecision,
  shouldDelegate,
  horizonFor,
  paramsFor,
  paramsFromCalibration,
  lEffFor,
  validateCovenantConstraint,
} from "./horizon.js";
export type {
  HorizonParams,
  DelegationDecision,
  DelegationReason,
  DelegateOptions,
} from "./horizon.js";

export {
  fitHorizon,
  empiricalDStar,
  validateConstraint,
  computeCalibrationHash,
} from "./calibrate.js";
export type { CalibrationFit } from "./calibrate.js";

export { estimateDepth, DEPTH_MIN, DEPTH_MAX } from "./estimator.js";
