// Aix-Route — Deterministic Horizon routing (TypeScript port)
// https://github.com/iPatan7/Aix-Route — License: MIT
//
// ============================================================
//  CALIBRATION (offline)
//
//  Fits a model's Deterministic Horizon from measured accuracy-
//  vs-depth — the TypeScript twin of Aix-Route's `aix_route.calibrate`.
//  The authoritative measurement pipeline is the Python `aix-route
//  calibrate` CLI; this port lets a TypeScript consumer re-verify a
//  fit and compute the canonical calibration hash without leaving JS.
// ============================================================

import { sha256 } from "@noble/hashes/sha256";
import { bytesToHex } from "@noble/hashes/utils";

import { GAMMA, ALPHA_DEFAULT, computeHorizon } from "./horizon.js";

export interface CalibrationFit {
  eps0: number;
  lEff: number;
  dStar: number;
  gamma: number;
  rSquared: number;
  nPoints: number;
  constraintOk: boolean;
}

/**
 * The constraint that keeps the derived L_eff positive in the horizon layer:
 *
 *   eps0 * dStar < ln(1/alpha)
 *
 * A pair that violates this (e.g. eps0=0.5, dStar=3 ⇒ 1.5 > ln 2) is out of the
 * theory's regime and must be rejected before it can produce a negative L_eff.
 */
export function validateConstraint(
  eps0: number,
  dStar: number,
  alpha: number = ALPHA_DEFAULT,
): boolean {
  return eps0 * dStar < Math.log(1 / alpha);
}

/**
 * Fit (eps0, lEff) from measured accuracy-vs-depth and derive dStar.
 *
 * Linear least-squares in ln(accuracy):
 *   ln(a) = (-d)*eps0 + (-gamma/2 * d*(d+1)) * (1/lEff)
 *
 * Mirrors `deterministic_horizon.calibrate.fit_horizon`. Returns `null` when
 * the data is outside the theory's regime (fewer than two interior points,
 * non-decay, or constraint violation).
 */
export function fitHorizon(
  depths: number[],
  accuracies: number[],
  gamma: number = GAMMA,
  alpha: number = ALPHA_DEFAULT,
): CalibrationFit | null {
  if (depths.length !== accuracies.length) {
    throw new Error("depths and accuracies must have the same length");
  }

  // Interior points only (0 < acc < 1) for a stable log fit.
  const d: number[] = [];
  const y: number[] = [];
  for (let i = 0; i < depths.length; i++) {
    const a = accuracies[i];
    if (a > 0 && a < 1) {
      d.push(depths[i]);
      y.push(Math.log(a));
    }
  }
  if (d.length < 2) return null;

  // Design columns: x1 = -d, x2 = -(gamma/2)*d*(d+1)
  const x1 = d.map((di) => -di);
  const x2 = d.map((di) => (-gamma / 2) * di * (di + 1));

  // Normal-equations solve of the 2-parameter least squares (X^T X) b = X^T y.
  const s11 = dot(x1, x1);
  const s12 = dot(x1, x2);
  const s22 = dot(x2, x2);
  const t1 = dot(x1, y);
  const t2 = dot(x2, y);
  const det = s11 * s22 - s12 * s12;
  if (det === 0) return null;

  const eps0 = (t1 * s22 - t2 * s12) / det;
  const invLEff = (s11 * t2 - s12 * t1) / det;

  if (invLEff <= 0 || eps0 < 0) return null;
  const lEff = 1 / invLEff;

  const dStar = computeHorizon(eps0, lEff, gamma, alpha);
  if (!Number.isFinite(dStar) || dStar <= 0) return null;

  const constraintOk = validateConstraint(eps0, dStar, alpha);
  if (!constraintOk) return null;

  // R^2 in log space.
  const yPred = d.map((_, i) => eps0 * x1[i] + invLEff * x2[i]);
  const yMean = y.reduce((s, v) => s + v, 0) / y.length;
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < y.length; i++) {
    ssRes += (y[i] - yPred[i]) ** 2;
    ssTot += (y[i] - yMean) ** 2;
  }
  const rSquared = ssTot > 0 ? 1 - ssRes / ssTot : 0;

  return { eps0, lEff, dStar, gamma, rSquared, nPoints: d.length, constraintOk };
}

/**
 * Model-free d*: linear-interpolated depth where accuracy crosses `alpha`.
 * Mirrors `deterministic_horizon.calibrate.empirical_d_star`. Returns `null`
 * if accuracy never crosses the threshold in range.
 */
export function empiricalDStar(
  depths: number[],
  accuracies: number[],
  alpha: number = ALPHA_DEFAULT,
): number | null {
  const pts = depths
    .map((dd, i) => [dd, accuracies[i]] as [number, number])
    .sort((p, q) => p[0] - q[0]);
  for (let i = 0; i < pts.length - 1; i++) {
    const [d0, a0] = pts[i];
    const [d1, a1] = pts[i + 1];
    if (a0 >= alpha && alpha >= a1 && a0 !== a1) {
      return d0 + ((alpha - a0) * (d1 - d0)) / (a1 - a0);
    }
  }
  return null;
}

/**
 * Canonical hash binding a calibration to a covenant. Hashes the rounded,
 * order-stable tuple so the same calibration always yields the same digest
 * (returned as `"sha256:" + hex`, matching the Seal layer's convention).
 */
export function computeCalibrationHash(
  model: string,
  eps0: number,
  dStar: number,
  lEff: number,
): string {
  // Fixed precision so floating-point noise never changes the hash.
  const canonical = JSON.stringify({
    model,
    eps0: round(eps0, 6),
    dStar: round(dStar, 6),
    lEff: round(lEff, 6),
    gamma: GAMMA,
  });
  const digest = sha256(new TextEncoder().encode(canonical));
  return "sha256:" + bytesToHex(digest);
}

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function round(x: number, places: number): number {
  const f = 10 ** places;
  return Math.round(x * f) / f;
}
