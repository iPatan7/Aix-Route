"""
Lightweight task classifier for the depth estimator.

Given a free-text problem description, :func:`classify_task` returns the
:class:`TaskType` that best matches it. The classifier is deliberately
dependency-free (pure keyword/structure scoring) so practitioners can call it
in an agent's hot path without loading a model. The returned :class:`TaskType`
is a ``str`` enum, so members compare equal to their plain-string value
(``TaskType.CODE == "code"``), which keeps call sites terse.

Example
-------
>>> from deterministic_horizon.estimators import classify_task
>>> classify_task("Trace variable x through 20 lines of Python")
<TaskType.CODE: 'code'>
>>> classify_task("Sort [5, 2, 8, 1] using adjacent swaps") == "permutation"
True
"""

from __future__ import annotations

import re
from enum import Enum

__all__ = ["TaskType", "classify_task", "task_scores"]


class TaskType(str, Enum):
    """Coarse task family used to pick a depth-estimation heuristic.

    Members are ``str`` values so ``TaskType.CODE == "code"`` holds, letting
    callers compare against plain strings without importing the enum.
    """

    PERMUTATION = "permutation"
    CODE = "code"
    SQL = "sql"
    FSA = "fsa"
    ARITHMETIC = "arithmetic"
    GENERIC = "generic"


# Keyword signatures per task family. Scoring counts how many distinct keywords
# appear (case-insensitive substring match), so longer descriptions don't bias
# toward whichever family happens to repeat a word.
_KEYWORDS: dict[TaskType, tuple[str, ...]] = {
    TaskType.PERMUTATION: (
        "permutation",
        "permute",
        "swap",
        "adjacent",
        "transposition",
        "sort",
        "rearrange",
        "reorder",
        "shuffle",
        "reverse",
        "array",
    ),
    TaskType.CODE: (
        "code",
        "python",
        "javascript",
        "function",
        "variable",
        "trace",
        "program",
        "compile",
        "execute",
        "loop",
        "line",
        "def ",
        "return",
    ),
    TaskType.SQL: (
        "sql",
        "select",
        "query",
        "join",
        "table",
        "database",
        "where",
        "group by",
        "schema",
        "row",
    ),
    TaskType.FSA: (
        "automaton",
        "automata",
        "fsa",
        "dfa",
        "nfa",
        "state machine",
        "finite state",
        "transition",
        "accept state",
    ),
    TaskType.ARITHMETIC: (
        "arithmetic",
        "modular",
        "modulo",
        " mod ",
        "compute",
        "calculate",
        "remainder",
        "multiply",
        "subtract",
        "divide",
        "summation",
    ),
}

_INT_LIST_RE = re.compile(r"\[\s*-?\d+(?:\s*,\s*-?\d+)*\s*\]")


def task_scores(problem: str) -> dict[TaskType, int]:
    """Return the raw keyword score for each task family.

    Exposed for debugging and for callers that want the full distribution
    rather than just the arg-max returned by :func:`classify_task`.
    """
    text = problem.lower()
    scores: dict[TaskType, int] = {}
    for task_type, keywords in _KEYWORDS.items():
        scores[task_type] = sum(1 for kw in keywords if kw in text)

    # A literal integer array is a strong permutation signal (the canonical
    # PermutationProbe input), so give it an extra vote.
    if _INT_LIST_RE.search(problem):
        scores[TaskType.PERMUTATION] += 1
    return scores


def classify_task(problem: str) -> TaskType:
    """Classify ``problem`` into a :class:`TaskType`.

    Returns :data:`TaskType.GENERIC` when no family scores above zero. Ties are
    broken by the declaration order in :data:`_KEYWORDS` (permutation first),
    which keeps the function deterministic.
    """
    scores = task_scores(problem)
    best = max(scores, key=lambda t: scores[t])
    if scores[best] == 0:
        return TaskType.GENERIC
    return best
