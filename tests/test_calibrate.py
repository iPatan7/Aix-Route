"""Tests for the horizon calibration helpers."""

from __future__ import annotations

import math

from deterministic_horizon.calibrate import empirical_d_star, fit_horizon
from deterministic_horizon.policy import GAMMA, _l_eff_for


def test_fit_recovers_known_curve() -> None:
    # Generate a curve from known (eps0, L_eff), then check the fit recovers it.
    eps0, l_eff = 0.05, 60.0
    depths = [4, 6, 8, 10, 12, 14]
    accs = [
        math.exp(-d * eps0 - GAMMA * d * (d + 1) / (2 * l_eff)) for d in depths
    ]
    fit = fit_horizon(depths, accs)
    assert fit is not None
    assert abs(fit["eps0"] - eps0) < 1e-6
    assert abs(fit["L_eff"] - l_eff) < 1e-6
    assert fit["r_squared"] > 0.999
    assert fit["constraint_ok"]


def test_fit_rejects_too_few_interior_points() -> None:
    # All saturated at 1.0/0.0 → no informative points.
    assert fit_horizon([3, 5, 8], [1.0, 1.0, 0.0]) is None


def test_fit_rejects_non_decay() -> None:
    # Accuracy rising with depth is out of regime.
    assert fit_horizon([5, 8, 10], [0.3, 0.5, 0.7]) is None


def test_fit_respects_constraint() -> None:
    # A genuine fit must keep eps0*d* < ln(1/alpha); _l_eff_for must not raise.
    fit = fit_horizon([5, 8, 10, 12], [0.9, 0.6, 0.4, 0.25])
    if fit is not None:
        assert fit["eps0"] * fit["d_star"] < math.log(2)
        # The policy's L_eff derivation must accept the fitted pair.
        _l_eff_for(fit["eps0"], fit["d_star"])  # should not raise


def test_empirical_d_star_interpolates() -> None:
    # 50% crossing between d=8 (0.6) and d=10 (0.4) → d=9.
    d = empirical_d_star([5, 8, 10, 12], [1.0, 0.6, 0.4, 0.2])
    assert d is not None
    assert abs(d - 9.0) < 1e-9


def test_empirical_d_star_none_when_no_crossing() -> None:
    assert empirical_d_star([3, 5, 8], [1.0, 0.9, 0.8]) is None


def test_measured_groq_data_is_in_regime() -> None:
    # The 70b curve we shipped must fit cleanly.
    fit = fit_horizon([5, 8, 10, 12, 15], [1.0, 0.5, 0.5, 0.25, 0.25])
    assert fit is not None
    assert 5 < fit["d_star"] < 12
