"""
Lightweight reasoning-depth fallback for the integrations.

Agent C ships the real :func:`estimate_depth` (heuristics + optional learned
model) under ``src/estimators``. The integrations *prefer* that estimator when
it is importable, but they must not hard-depend on it — a user may install only
the integrations slice. This module therefore provides:

* :func:`resolve_depth` — use the caller-supplied depth if given, otherwise the
  best estimator available, otherwise a crude built-in heuristic.

The built-in heuristic is intentionally simple and conservative: it counts the
tokens that imply sequential state mutation (numbers, operators, swap/step
words). It only needs to be good enough to land on the correct side of a
model's horizon for obvious cases; precise estimation is Agent C's job.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

_NUM_RE = re.compile(r"-?\d+")
_STEP_WORD_LIST = (
    "swap",
    "step",
    "move",
    "then",
    "next",
    "after",
    "line",
    "iteration",
    "operation",
)
_STEP_WORDS = tuple(_STEP_WORD_LIST)
# "<number> <step-word>" — e.g. "35 swaps", "20 lines". The number is an
# explicit step count, so its *magnitude* is the depth signal (not just +1).
_COUNT_RE = re.compile(
    r"(\d+)\s+(?:\w+\s+){0,2}?(?:" + "|".join(_STEP_WORD_LIST) + r")s?\b",
    re.IGNORECASE,
)


def _builtin_estimate(problem: str) -> int:
    """Crude, dependency-free depth guess from a problem string."""
    if not problem.strip():
        return 0
    text = problem.lower()
    numbers = len(_NUM_RE.findall(problem))
    step_hits = sum(text.count(word) for word in _STEP_WORDS)
    operators = sum(text.count(op) for op in ("+", "-", "*", "/", "->", "=>"))
    # Token-count signal: each list element / number is a potential
    # state-tracking step; step words and operators add a little.
    token_depth = numbers + step_hits + operators // 2

    # Explicit-count signal: "35 swaps" / "20 lines" means ~35 / ~20 steps.
    explicit = [int(m.group(1)) for m in _COUNT_RE.finditer(problem)]
    count_depth = max(explicit) if explicit else 0

    # Take the stronger of the two; floor of 1 for any non-empty problem.
    return max(1, token_depth, count_depth)


def _load_external_estimator() -> Callable[..., int] | None:
    """Return Agent C's ``estimate_depth`` if it is importable, else ``None``."""
    try:
        from src.estimators import estimate_depth  # type: ignore[import-not-found]

        return estimate_depth
    except Exception:  # noqa: BLE001 - estimator slice may be absent
        try:
            from deterministic_horizon.estimators import (  # type: ignore[import-not-found]
                estimate_depth,
            )

            return estimate_depth
        except Exception:  # noqa: BLE001
            return None


def resolve_depth(
    problem: Any,
    *,
    estimated_depth: int | None = None,
    task_type: str | None = None,
) -> int:
    """
    Decide the reasoning depth to feed the router.

    Precedence:
        1. An explicit ``estimated_depth`` (the caller already knows).
        2. Agent C's learned/heuristic ``estimate_depth`` if available.
        3. The built-in token-counting heuristic.

    ``problem`` is coerced to ``str`` so dict/list payloads (common in agent
    frameworks) still produce a usable estimate.
    """
    if estimated_depth is not None:
        return int(estimated_depth)

    text = problem if isinstance(problem, str) else str(problem)

    external = _load_external_estimator()
    if external is not None:
        try:
            if task_type is not None:
                return int(external(text, task_type=task_type))
            return int(external(text))
        except Exception:  # noqa: BLE001 - never let estimation crash routing
            pass

    return _builtin_estimate(text)


__all__ = ["resolve_depth"]
