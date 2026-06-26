"""Tests for the quick validation benchmark (Agent C slice)."""

from __future__ import annotations

import pytest
from benchmarks.quick import QuickResult, run_quick_benchmark


class TestAcceptanceCriteria:
    """`python -m benchmarks.quick --model gpt-4o --task permutation --n 50`."""

    def test_recovers_paper_horizon_for_gpt4o(self) -> None:
        result = run_quick_benchmark(model="gpt-4o", task="permutation", n=50)
        assert isinstance(result, QuickResult)
        assert result.mode == "simulated"
        assert result.paper_d_star == 22.0
        # The whole point: the simulated pipeline recovers d* within tolerance.
        assert result.matches_paper
        assert abs(result.d_star - 22.0) <= 3.0

    def test_reports_r_squared(self) -> None:
        result = run_quick_benchmark(model="gpt-4o", n=50)
        # Fitting a 2-param decay curve to Bernoulli-sampled accuracy at n=50
        # is moderately noisy; require a clearly positive fit rather than a
        # tight one to avoid seed-dependent flakiness.
        assert result.r_squared > 0.5


class TestDeterminismAndModes:
    def test_seed_is_deterministic(self) -> None:
        a = run_quick_benchmark(model="gpt-4o", n=50, seed=7)
        b = run_quick_benchmark(model="gpt-4o", n=50, seed=7)
        assert a.d_star == b.d_star

    def test_unknown_model_simulated_raises(self) -> None:
        with pytest.raises(ValueError):
            run_quick_benchmark(model="not-a-real-model", n=20)

    def test_other_known_model_uses_table_horizon(self) -> None:
        result = run_quick_benchmark(model="o3-mini", n=50)
        assert result.paper_d_star == 31.0  # from MODEL_HORIZONS

    def test_real_without_key_raises(self) -> None:
        # No API key in the test env -> real mode must fail loudly, not silently.
        with pytest.raises(RuntimeError):
            run_quick_benchmark(model="gpt-4o", n=20, real=True)


class TestSummary:
    def test_summary_mentions_verdict(self) -> None:
        result = run_quick_benchmark(model="gpt-4o", n=50)
        text = result.summary()
        assert "MATCH" in text
        assert "d*" in text


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
