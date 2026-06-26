"""
Calibrate per-model Deterministic Horizons from *measured* accuracy data.

The paper's :data:`deterministic_horizon.policy.MODEL_HORIZONS` are calibrated
for the authors' local/HF setup. A model served elsewhere (Groq, a different
quantization, a different prompt) can sit at a different horizon. Rather than
guess, measure accuracy-vs-depth on your own infrastructure and fit the
decoherence parameters here, then feed the result into the policy.

The fit inverts Theorem 4.2's closed-form decay,

    P(correct at depth d) ≈ exp(−d·ε₀ − γ·d(d+1) / (2·L_eff)),

by linear least-squares in ``ln P``:

    ln P = (−d)·ε₀ + (−γ·d(d+1)/2)·(1/L_eff),

then derives d* from Theorem 4.8. A fit is rejected (returns ``None``) when the
data falls outside the theory's regime — too few informative points, a positive
slope, or a parameter pair that violates the ``ε₀·d* < ln(1/α)`` constraint that
keeps L_eff positive. A rejected fit is a signal to flag the model
(always-delegate / fall back to the paper default), never to invent numbers.
"""

from __future__ import annotations

import math
from typing import TypedDict

import numpy as np


class CalibrationResult(TypedDict):
    """Fitted decoherence parameters for one model."""

    eps0: float
    L_eff: float
    d_star: float
    gamma: float
    r_squared: float
    n_points: int
    constraint_ok: bool


def fit_horizon(
    depths: list[float],
    accuracies: list[float],
    gamma: float = 0.15,
    alpha: float = 0.5,
) -> CalibrationResult | None:
    """
    Fit ``(eps0, L_eff)`` and derive ``d_star`` from accuracy-vs-depth data.

    Parameters
    ----------
    depths, accuracies :
        Measured accuracy at each depth (same length). Only points with
        ``0 < accuracy < 1`` are used — saturated 0/1 points carry no curve
        information and break the log transform.
    gamma :
        Shared attention-decay constant (paper :data:`policy.GAMMA`).
    alpha :
        Accuracy threshold defining d* (default 0.5).

    Returns
    -------
    CalibrationResult or None
        ``None`` when the data is outside the theory's regime (fewer than two
        informative points, non-decaying fit, or constraint violation).
    """
    depths_arr = np.asarray(depths, dtype=float)
    acc_arr = np.asarray(accuracies, dtype=float)
    if depths_arr.shape != acc_arr.shape:
        raise ValueError("depths and accuracies must have the same length")

    # Need interior points (strictly between 0 and 1) for a stable log fit.
    mask = (acc_arr > 0.0) & (acc_arr < 1.0)
    if int(mask.sum()) < 2:
        return None

    d = depths_arr[mask]
    a = acc_arr[mask]

    # ln(a) = -d*eps0 - (gamma/2)*d*(d+1)*(1/L_eff)
    y = np.log(a)
    X = np.column_stack([-d, -(gamma / 2.0) * d * (d + 1.0)])

    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    eps0 = float(coeffs[0])
    inv_l_eff = float(coeffs[1])

    # Both must be positive for a genuine decay; otherwise out of regime.
    if inv_l_eff <= 0.0 or eps0 < 0.0:
        return None
    l_eff = 1.0 / inv_l_eff

    # Derive d* (Theorem 4.8).
    discriminant = eps0**2 * l_eff**2 + 2.0 * gamma * l_eff * math.log(1.0 / alpha)
    if discriminant < 0.0:
        return None
    d_star = (-eps0 * l_eff + math.sqrt(discriminant)) / gamma
    if d_star <= 0.0:
        return None

    # Constraint that keeps the derived-L_eff positive in the policy layer.
    constraint = math.log(1.0 / alpha) / d_star
    constraint_ok = eps0 < constraint
    if not constraint_ok:
        return None

    # Goodness of fit in log space.
    y_pred = X @ coeffs
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

    return {
        "eps0": eps0,
        "L_eff": l_eff,
        "d_star": d_star,
        "gamma": gamma,
        "r_squared": r_squared,
        "n_points": int(mask.sum()),
        "constraint_ok": constraint_ok,
    }


def empirical_d_star(
    depths: list[float],
    accuracies: list[float],
    alpha: float = 0.5,
) -> float | None:
    """
    Model-free d*: the depth where accuracy crosses ``alpha``, by linear
    interpolation between the bracketing measured points.

    Useful when :func:`fit_horizon` rejects the data (too few interior points,
    a decay too sharp for the smooth super-exponential, …) but the accuracy
    clearly drops through the threshold. Returns ``None`` if accuracy never
    crosses ``alpha`` within the measured range.
    """
    pts = sorted(zip(depths, accuracies, strict=True))
    for (d0, a0), (d1, a1) in zip(pts, pts[1:], strict=False):
        if a0 >= alpha >= a1 and a0 != a1:
            # Linear interpolation of the crossing depth.
            return d0 + (alpha - a0) * (d1 - d0) / (a1 - a0)
    return None


__all__ = ["fit_horizon", "empirical_d_star", "CalibrationResult"]
