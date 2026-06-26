"""Unit tests for the metrics package."""

from __future__ import annotations

import numpy as np
import pytest
from deterministic_horizon.metrics import (
    accuracy_by_depth,
    bootstrap_ci,
    compute_effect_size,
    compute_ssj,
    estimate_horizon,
    fit_decoherence_model,
)


def test_ssj_perfect_overlap():
    states = [[1, 2, 3], [2, 1, 3], [2, 3, 1]]
    r = compute_ssj(states, states)
    assert r.ssj == pytest.approx(1.0)
    assert r.precision == pytest.approx(1.0)
    assert r.recall == pytest.approx(1.0)


def test_ssj_no_overlap():
    r = compute_ssj([[1, 2]], [[3, 4]])
    assert r.ssj == 0.0
    assert r.precision == 0.0
    assert r.recall == 0.0


def test_ssj_partial_overlap_decomposition():
    true_states = [[1], [2], [3], [4]]
    model_states = [[1], [2], [99]]
    r = compute_ssj(true_states, model_states)
    # intersection = {[1], [2]}; |t|=4 ; |m|=3 ; |∪|=5
    assert r.intersection_size == 2
    assert r.union_size == 5
    assert r.ssj == pytest.approx(2 / 5)
    assert r.precision == pytest.approx(2 / 3)
    assert r.recall == pytest.approx(2 / 4)


def test_accuracy_by_depth_groups_correctly():
    results = [
        {"optimal_depth": 5, "correct": True},
        {"optimal_depth": 5, "correct": False},
        {"optimal_depth": 5, "correct": True},
        {"optimal_depth": 10, "correct": False},
        {"optimal_depth": 10, "correct": False},
    ]
    stats = accuracy_by_depth(results)
    assert stats[5]["accuracy"] == pytest.approx(2 / 3)
    assert stats[5]["n"] == 3
    assert stats[10]["accuracy"] == 0.0
    assert 0 <= stats[5]["ci_low"] <= stats[5]["accuracy"] <= stats[5]["ci_high"] <= 1


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=2.0, scale=0.1, size=200)
    point, lo, hi = bootstrap_ci(data, n_bootstrap=500, seed=0)
    assert lo <= point <= hi
    assert lo <= 2.0 <= hi  # the truth must land inside


def test_effect_size_large_difference():
    rng = np.random.default_rng(0)
    g1 = rng.normal(loc=1.0, scale=0.1, size=50)
    g2 = rng.normal(loc=0.0, scale=0.1, size=50)
    eff = compute_effect_size(g1, g2)
    assert eff["effect_size"] > 2  # Cohen's d for means 1 vs 0 with σ=0.1
    assert eff["interpretation"] == "large"


def test_horizon_drops_below_threshold():
    # Build a clean super-exponential decay → horizon should land near 20.
    depths = np.arange(2, 50, 2)
    eps0, gamma = 0.018, 0.0025
    accs = np.exp(-eps0 * depths - gamma * depths * (depths + 1) / 2)
    results = []
    for d, a in zip(depths.tolist(), accs.tolist(), strict=False):
        # 40 instances per depth, accuracy = a
        n_correct = int(round(a * 40))
        for _ in range(n_correct):
            results.append({"optimal_depth": int(d), "correct": True})
        for _ in range(40 - n_correct):
            results.append({"optimal_depth": int(d), "correct": False})

    h = estimate_horizon(results, threshold=0.5)
    assert 10 < h["d_star"] < 30, h


def test_fit_decoherence_model_recovers_params():
    depths = np.arange(2, 50, 2)
    eps0, gamma = 0.02, 0.0030
    accs = np.exp(-eps0 * depths - gamma * depths * (depths + 1) / 2)
    out = fit_decoherence_model(depths.tolist(), accs.tolist(), l_eff=1)
    assert out["r_squared"] > 0.99
