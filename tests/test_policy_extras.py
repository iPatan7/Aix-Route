"""Tests for the batch / table / recommendation helpers added to the policy."""

from __future__ import annotations

import pytest
from deterministic_horizon.policy import (
    MODEL_HORIZONS,
    horizon_for,
    horizon_table,
    recommend_model,
    should_delegate,
    should_delegate_batch,
)


class TestShouldDelegateBatch:
    def test_matches_scalar_should_delegate(self):
        depths = [3, 8, 15, 22, 35, 60]
        batch = should_delegate_batch(depths, model="gpt-4o")
        scalar = [should_delegate(d, model="gpt-4o") for d in depths]
        assert batch == scalar

    def test_returns_one_bool_per_depth(self):
        # d=5 (89% CoT) reason; d=8 (82%, within 10pts of the 92% tool) reason;
        # d=35 (26%, past the horizon) delegate.
        out = should_delegate_batch([5, 8, 35], model="gpt-4o")
        assert out == [False, False, True]
        assert all(isinstance(b, bool) for b in out)

    def test_margin_rule_delegates_above_threshold(self):
        # d=20 is above 50% (≈54%) but the tool beats it by >10 pts, so the
        # margin rule still says delegate.
        assert should_delegate_batch([20], model="gpt-4o") == [True]

    def test_forwards_kwargs(self):
        # With no tool available, nothing delegates regardless of depth.
        assert should_delegate_batch([5, 50, 500], tool_available=False) == [False, False, False]

    def test_empty_input(self):
        assert should_delegate_batch([], model="gpt-4o") == []


class TestHorizonTable:
    def test_covers_every_model(self):
        names = {row["model"] for row in horizon_table()}
        assert names == set(MODEL_HORIZONS)

    def test_sorted_by_horizon_ascending(self):
        dstars = [row["d_star"] for row in horizon_table()]
        assert dstars == sorted(dstars)

    def test_rows_match_policy_constants(self):
        for row in horizon_table():
            params = MODEL_HORIZONS[row["model"]]
            assert row["eps0"] == pytest.approx(params["eps0"])
            assert row["d_star"] == pytest.approx(params["d_star"])
            assert row["l_eff"] == pytest.approx(params["l_eff"])


class TestRecommendModel:
    def test_shallow_depth_has_a_recommendation(self):
        name, acc = recommend_model(8)
        assert name in MODEL_HORIZONS
        assert acc >= 0.5

    def test_picks_least_overpowered_viable_model(self):
        # The recommended model's horizon must be <= the horizon of any other
        # individually-viable model at this depth (it's the least over-powered).
        depth = 18
        name, _ = recommend_model(depth)
        assert name is not None
        others = [
            m
            for m in MODEL_HORIZONS
            if m != "default" and recommend_model(depth, candidates=[m])[0] is not None
        ]
        assert horizon_for(name) <= min(horizon_for(m) for m in others)

    def test_impossible_depth_returns_none(self):
        name, acc = recommend_model(500)
        assert name is None
        assert 0.0 <= acc < 0.5

    def test_respects_candidate_list(self):
        name, _ = recommend_model(8, candidates=["gpt-4o"])
        assert name == "gpt-4o"
