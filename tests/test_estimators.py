"""Tests for the reasoning-depth estimator and task classifier (Agent C slice)."""

from __future__ import annotations

import pytest
from src.estimators import TaskType, classify_task, estimate_depth, set_learned_estimator


class TestAcceptanceCriteria:
    """The exact assertions from the foundation prompt's Agent C acceptance."""

    def test_estimate_depth_permutation_in_range(self) -> None:
        d = estimate_depth("Sort [5,2,8,1] using adjacent swaps", task_type="permutation")
        assert 5 <= d <= 15

    def test_classify_code(self) -> None:
        task = classify_task("Trace variable x through 20 lines of Python")
        assert task == "code"


class TestClassifyTask:
    def test_permutation_keywords(self) -> None:
        assert classify_task("Apply adjacent transposition swaps to permute the array") == (
            TaskType.PERMUTATION
        )

    def test_sql(self) -> None:
        assert classify_task("Write a SQL SELECT query joining the orders table") == "sql"

    def test_fsa(self) -> None:
        assert classify_task("Simulate the DFA finite state automaton on this input") == "fsa"

    def test_arithmetic(self) -> None:
        assert classify_task("Compute the modular arithmetic remainder step by step") == (
            "arithmetic"
        )

    def test_generic_fallback(self) -> None:
        assert classify_task("Tell me a story about the ocean") == TaskType.GENERIC

    def test_strenum_equals_plain_string(self) -> None:
        assert TaskType.CODE == "code"
        assert TaskType.PERMUTATION.value == "permutation"


class TestEstimateDepth:
    def test_returns_positive_int(self) -> None:
        d = estimate_depth("anything at all")
        assert isinstance(d, int)
        assert d >= 1

    def test_infers_task_type_when_omitted(self) -> None:
        d = estimate_depth("Sort [3, 1, 2] with adjacent swaps")
        assert d >= 1

    def test_longer_permutation_is_deeper(self) -> None:
        shallow = estimate_depth("[2, 1]", task_type="permutation")
        deep = estimate_depth("[8, 7, 6, 5, 4, 3, 2, 1]", task_type="permutation")
        assert deep > shallow

    def test_sorted_array_is_shallow(self) -> None:
        # Already sorted: zero swaps, only the per-element tracking term.
        d = estimate_depth("[0, 1, 2, 3]", task_type="permutation")
        assert d == 4

    def test_explicit_swap_count_without_array(self) -> None:
        # No literal array: an explicit "<n> swaps" magnitude must win, so a
        # deep case routes deep (regression for integrations/_depth fallback).
        assert estimate_depth("apply 35 sequential swaps") == 35
        assert estimate_depth("sort the list with 5 swaps") == 5

    def test_code_uses_explicit_line_count(self) -> None:
        d = estimate_depth("Trace x through 20 lines of Python", task_type="code")
        assert d == 20

    def test_bfs_matches_inversion_count_small(self) -> None:
        # [5,2,8,1] -> ranks [2,1,3,0], inversion count 4, plus n=4 -> 8.
        assert estimate_depth("[5, 2, 8, 1]", task_type="permutation") == 8


class TestLearnedEstimatorHook:
    def test_override_and_clear(self) -> None:
        set_learned_estimator(lambda problem, tt: 99)
        try:
            assert estimate_depth("[0, 1, 2, 3]", task_type="permutation") == 99
        finally:
            set_learned_estimator(None)
        # Back to the structural heuristic.
        assert estimate_depth("[0, 1, 2, 3]", task_type="permutation") == 4

    def test_none_return_defers_to_heuristic(self) -> None:
        set_learned_estimator(lambda problem, tt: None)
        try:
            assert estimate_depth("[0, 1, 2, 3]", task_type="permutation") == 4
        finally:
            set_learned_estimator(None)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
