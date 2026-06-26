"""Tests for the practitioner-facing delegation policy helper."""

from __future__ import annotations

import math

import pytest
from deterministic_horizon import (
    DelegationDecision,
    delegation_decision,
    expected_neural_accuracy,
    horizon_for,
    should_delegate,
)
from deterministic_horizon.policy import MODEL_HORIZONS


def test_expected_accuracy_monotone_decreasing():
    """Accuracy must monotonically decrease with depth."""
    accs = [expected_neural_accuracy(d, model="gpt-4o") for d in range(0, 51, 5)]
    assert accs[0] == pytest.approx(1.0)
    for prev, curr in zip(accs[:-1], accs[1:], strict=True):
        assert curr < prev


def test_expected_accuracy_zero_at_d_zero():
    assert expected_neural_accuracy(0, model="gpt-4o") == pytest.approx(1.0)


def test_unknown_model_falls_back_to_default():
    a_unknown = expected_neural_accuracy(15, model="not-a-real-model")
    a_default = expected_neural_accuracy(15, model="default")
    assert a_unknown == pytest.approx(a_default)


def test_should_delegate_thresholds():
    # Below the horizon — don't delegate.
    assert should_delegate(estimated_depth=5, model="gpt-4o") is False
    # Way past the horizon — definitely delegate.
    assert should_delegate(estimated_depth=50, model="gpt-4o") is True


def test_should_delegate_respects_tool_unavailable():
    # With no tool, never delegate — we have to fall back to neural.
    assert should_delegate(estimated_depth=50, model="gpt-4o", tool_available=False) is False


def test_decision_full_structure_past_horizon():
    d = delegation_decision(estimated_depth=40, model="gpt-4o", tool_available=True)
    assert isinstance(d, DelegationDecision)
    assert d.delegate is True
    assert d.reason in {"above_horizon", "tool_dominates_by_margin"}
    assert 0.0 <= d.expected_neural_accuracy < 0.5
    assert d.expected_tool_accuracy > d.expected_neural_accuracy
    assert "delegate" in d.explain().lower()


def test_decision_tool_unavailable_explains_itself():
    d = delegation_decision(estimated_depth=40, model="gpt-4o", tool_available=False)
    assert d.delegate is False
    assert d.reason == "tool_unavailable"
    assert "no tool" in d.explain().lower()


def test_decision_validates_threshold_and_margin():
    with pytest.raises(ValueError):
        delegation_decision(estimated_depth=10, threshold=1.2)
    with pytest.raises(ValueError):
        delegation_decision(estimated_depth=10, margin=-0.1)


def test_negative_depth_rejected():
    with pytest.raises(ValueError):
        expected_neural_accuracy(-1)


def test_known_model_horizons_in_documented_range():
    # Paper Abstract claims d* ∈ [19, 31] across the surveyed models.
    for name, params in MODEL_HORIZONS.items():
        assert 19 <= params["d_star"] <= 31, (name, params)


def test_expected_accuracy_crosses_half_at_horizon():
    # By construction the decay curve must cross 0.5 at each model's d*.
    import pytest as _pytest

    for name, params in MODEL_HORIZONS.items():
        acc = expected_neural_accuracy(params["d_star"], model=name)
        assert acc == _pytest.approx(0.5, abs=1e-6), (name, acc)


def test_horizon_for_returns_float():
    assert isinstance(horizon_for("gpt-4o"), float)
    assert math.isfinite(horizon_for("gpt-4o"))


def test_short_problem_at_strong_model_does_not_delegate():
    # A 5-step problem on o3-mini (highest horizon, d*=31) should stay neural.
    d = delegation_decision(estimated_depth=5, model="o3-mini", tool_available=True)
    assert d.delegate is False
    assert d.reason == "below_horizon"
