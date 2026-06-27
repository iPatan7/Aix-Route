// Aix-Route TS port — parity + behaviour tests.
//
// Parity targets from the Python aix_route policy:
//   GAMMA=0.15, tool_accuracy=0.92, threshold=0.5, margin=0.10
//   llama-3.3-70b-versatile: eps0=0.07, d*=8.2, lEff=47.487485
//   d=5  70b: acc=0.672078  delegate=true  (tool_dominates_by_margin)
//   d=8  70b: acc=0.509812  delegate=true  (tool_dominates_by_margin)
//   d=35 70b: acc=0.011796  delegate=true  (above_horizon)
//   d=5  gpt-4o: acc=0.891358 delegate=false (below_horizon)

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  accuracyAtDepth,
  delegationDecision,
  shouldDelegate,
  paramsFor,
  estimateDepth,
  fitHorizon,
  validateConstraint,
  computeCalibrationHash,
  GAMMA,
} from "../src/index.js";

const APPROX = 1e-5;

test("lEff + accuracy match the Python parity values", () => {
  const p = paramsFor("llama-3.3-70b-versatile");
  assert.equal(p.eps0, 0.07);
  assert.equal(p.dStar, 8.2);
  assert.ok(Math.abs(p.lEff - 47.487485) < 1e-4);
  assert.ok(Math.abs(accuracyAtDepth(5, p.eps0, GAMMA, p.lEff) - 0.672078) < APPROX);
  assert.ok(Math.abs(accuracyAtDepth(8, p.eps0, GAMMA, p.lEff) - 0.509812) < APPROX);
  assert.ok(Math.abs(accuracyAtDepth(35, p.eps0, GAMMA, p.lEff) - 0.011796) < APPROX);
});

test("shouldDelegate + reasons match Python decisions", () => {
  assert.equal(shouldDelegate(5, "llama-3.3-70b-versatile"), true);
  assert.equal(shouldDelegate(8, "llama-3.3-70b-versatile"), true);
  assert.equal(shouldDelegate(35, "llama-3.3-70b-versatile"), true);
  assert.equal(shouldDelegate(5, "gpt-4o"), false);
  assert.equal(shouldDelegate(35, "gpt-4o"), true);

  assert.equal(delegationDecision(8, "llama-3.3-70b-versatile").reason, "tool_dominates_by_margin");
  assert.equal(delegationDecision(35, "llama-3.3-70b-versatile").reason, "above_horizon");
  assert.equal(delegationDecision(5, "gpt-4o").reason, "below_horizon");
});

test("estimateDepth handles lists, counts, and explicit steps", () => {
  assert.equal(estimateDepth("Sort [5,2,8,1] using adjacent swaps"), 6); // n*(n-1)/2, n=4
  assert.equal(estimateDepth("apply 35 sequential swaps"), 35); // count-word magnitude
  assert.equal(estimateDepth("Trace x through 20 lines"), 20);
  assert.equal(estimateDepth("solve in 7 steps"), 7);
  assert.equal(estimateDepth("5 swaps"), 5);
});

test("estimate → route: a 35-step task delegates", () => {
  const d = estimateDepth("apply 35 sequential swaps");
  assert.equal(shouldDelegate(d, "llama-3.3-70b-versatile"), true);
});

test("fitHorizon recovers the measured 70b curve and rejects bad data", () => {
  const fit = fitHorizon([5, 8, 10, 12, 15], [1.0, 0.5, 0.5, 0.25, 0.25]);
  assert.ok(fit);
  assert.ok(Math.abs(fit!.eps0 - 0.0703) < 1e-3);
  assert.ok(Math.abs(fit!.dStar - 8.2) < 0.1);
  assert.ok(fit!.constraintOk);

  assert.equal(fitHorizon([5], [0.0]), null);
  assert.equal(fitHorizon([5, 8, 10], [0.3, 0.5, 0.7]), null);
});

test("validateConstraint rejects the negative-lEff crash case", () => {
  assert.equal(validateConstraint(0.5, 3.0), false);
  assert.equal(validateConstraint(0.07, 8.2), true);
});

test("computeCalibrationHash is deterministic + sha256-prefixed", () => {
  const h = computeCalibrationHash("llama-3.3-70b-versatile", 0.07, 8.2, 47.487485);
  assert.equal(h, computeCalibrationHash("llama-3.3-70b-versatile", 0.07, 8.2, 47.487485));
  assert.match(h, /^sha256:[0-9a-f]{64}$/);
});
