"""Tests for the shared policy access and depth fallback."""

from __future__ import annotations

from integrations import resolve_depth, should_delegate
from integrations._depth import _builtin_estimate


def test_policy_resolves_and_matches_paper() -> None:
    # Past the horizon for gpt-4o (d*=22) → delegate.
    assert should_delegate(estimated_depth=35, model="gpt-4o") is True
    # Well within → don't delegate.
    assert should_delegate(estimated_depth=8, model="gpt-4o") is False


def test_resolve_depth_prefers_explicit() -> None:
    assert resolve_depth("anything at all", estimated_depth=17) == 17


def test_builtin_estimate_counts_numbers() -> None:
    # Four list elements → at least four state-tracking steps.
    d = _builtin_estimate("Sort [5,2,8,1] using adjacent swaps")
    assert d >= 4


def test_builtin_estimate_empty_is_zero() -> None:
    assert _builtin_estimate("") == 0
    assert _builtin_estimate("   ") == 0


def test_resolve_depth_coerces_non_string() -> None:
    # dict payloads (common in agent frameworks) must not crash.
    d = resolve_depth({"problem": "1 2 3 swap"}, estimated_depth=None)
    assert d >= 1


def test_explicit_count_drives_depth() -> None:
    # "<n> swaps/steps/lines" → the magnitude is the depth, so the heuristic
    # alone must land an obviously-deep problem past gpt-4o's d*=22.
    assert _builtin_estimate("apply 35 sequential swaps") == 35
    assert _builtin_estimate("Trace x through 20 lines of Python") == 20
    assert should_delegate(_builtin_estimate("apply 35 sequential swaps"), "gpt-4o")


def test_explicit_count_does_not_over_inflate_shallow() -> None:
    # "5 swaps" stays shallow (within horizon), not bumped by list elements.
    assert _builtin_estimate("sort [3,1,2] with 5 swaps") == 5
    assert not should_delegate(
        _builtin_estimate("sort [3,1,2] with 5 swaps"), "gpt-4o"
    )
