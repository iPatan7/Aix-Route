"""
Reasoning-depth estimator.

The policy layer (:mod:`deterministic_horizon.policy`) needs an *estimated
reasoning depth* — the number of deterministic state-tracking steps a problem
requires — to decide "think with the LLM" vs. "delegate to a tool". Asking
users to hand-count that for every problem is friction; :func:`estimate_depth`
removes it with a cheap, transparent heuristic.

The heuristic is structural, not learned: it counts the state mutations,
operators and operands implied by the problem text, specialised per
:class:`~deterministic_horizon.estimators.task_classifier.TaskType`. For
permutation problems with a literal integer array it goes further and computes
the *exact* BFS-optimal adjacent-transposition distance on the (small) state
space, which is provably the inversion count.

An optional learned estimator (logistic/linear regression on these same
features) can be slotted in later via :func:`set_learned_estimator`; it is
never imported unless a model is actually registered, so ``scikit-learn``
stays an optional dependency.

Example
-------
>>> from deterministic_horizon.estimators import estimate_depth
>>> estimate_depth("Sort [5, 2, 8, 1] using adjacent swaps", task_type="permutation")
8
"""

from __future__ import annotations

import re
from collections.abc import Callable

from deterministic_horizon.estimators.task_classifier import TaskType, classify_task

__all__ = ["estimate_depth", "set_learned_estimator"]

_INT_LIST_RE = re.compile(r"\[\s*-?\d+(?:\s*,\s*-?\d+)*\s*\]")
_NUMBER_RE = re.compile(r"-?\d+")
_OPERATOR_RE = re.compile(r"[+\-*/%^=<>]")

# "apply 35 sequential swaps", "20 steps", "12 operations" -> an explicit
# state-tracking magnitude. This is the single strongest depth signal when no
# literal state (array) is given, so the fallbacks consult it first.
_MAGNITUDE_RE = re.compile(
    r"(\d+)\s*(?:sequential\s+)?"
    r"(?:swaps?|steps?|operations?|moves?|transpositions?|iterations?|mutations?)",
    re.IGNORECASE,
)


def _magnitude_hint(problem: str) -> int | None:
    """Return the largest explicit '<n> <step-word>' count, if any."""
    counts = [int(m.group(1)) for m in _MAGNITUDE_RE.finditer(problem)]
    return max(counts) if counts else None

# Optional learned override. When set (see :func:`set_learned_estimator`) it is
# tried first and the structural heuristic becomes the fallback.
_LEARNED_ESTIMATOR: Callable[[str, TaskType], int | None] | None = None


def set_learned_estimator(fn: Callable[[str, TaskType], int | None] | None) -> None:
    """Register (or clear) an optional learned depth estimator.

    ``fn`` receives ``(problem, task_type)`` and returns an ``int`` depth, or
    ``None`` to defer to the structural heuristic. This is the hook for a tiny
    regression model trained on benchmark features; keeping it a callback means
    ``scikit-learn`` is only ever imported by whoever trains/loads the model.
    """
    global _LEARNED_ESTIMATOR
    _LEARNED_ESTIMATOR = fn


def _coerce_task_type(task_type: str | TaskType | None) -> TaskType | None:
    if task_type is None:
        return None
    if isinstance(task_type, TaskType):
        return task_type
    try:
        return TaskType(str(task_type).lower())
    except ValueError:
        return None


def estimate_depth(problem: str, task_type: str | TaskType | None = None) -> int:
    """Estimate the reasoning depth (state-tracking steps) of ``problem``.

    Parameters
    ----------
    problem : str
        Natural-language problem description.
    task_type : str | TaskType, optional
        Task family. When omitted it is inferred via
        :func:`~deterministic_horizon.estimators.task_classifier.classify_task`.

    Returns
    -------
    int
        Estimated depth, always ``>= 1``. Feed this straight into
        :func:`deterministic_horizon.policy.should_delegate`.
    """
    tt = _coerce_task_type(task_type) or classify_task(problem)

    if _LEARNED_ESTIMATOR is not None:
        learned = _LEARNED_ESTIMATOR(problem, tt)
        if learned is not None:
            return max(1, int(learned))

    if tt == TaskType.PERMUTATION:
        depth = _estimate_permutation(problem)
    elif tt == TaskType.CODE:
        depth = _estimate_code(problem)
    elif tt == TaskType.SQL:
        depth = _estimate_sql(problem)
    elif tt == TaskType.FSA:
        depth = _estimate_fsa(problem)
    elif tt == TaskType.ARITHMETIC:
        depth = _estimate_arithmetic(problem)
    else:
        depth = _estimate_generic(problem)

    return max(1, int(depth))


# ---------------------------------------------------------------------------
# Per-task structural heuristics.
# ---------------------------------------------------------------------------
def _parse_int_list(problem: str) -> list[int] | None:
    match = _INT_LIST_RE.search(problem)
    if not match:
        return None
    return [int(tok) for tok in _NUMBER_RE.findall(match.group(0))]


def _to_ranks(arr: list[int]) -> list[int]:
    """Map values to their sorted ranks (0..n-1), so the target is the identity."""
    order = sorted(range(len(arr)), key=lambda i: arr[i])
    ranks = [0] * len(arr)
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return ranks


def _inversions(arr: list[int]) -> int:
    inv = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                inv += 1
    return inv


def _optimal_swaps(arr: list[int]) -> int:
    """BFS-optimal adjacent-transposition distance to sort ``arr``.

    Equal to the inversion count. For small permutations (``n <= 8``) we
    cross-check against the project's BFS solver to stay faithful to the
    "falls back to BFS on small state spaces" contract; for larger ones the
    inversion count is used directly (it is provably optimal).
    """
    ranks = _to_ranks(arr)
    n = len(ranks)
    # BFS over S_n is only cheap for genuinely small n (<=7 -> <=5040 states).
    # For larger n the inversion count is used directly; it is provably the
    # optimal adjacent-transposition distance, so the result is identical.
    if n <= 7:
        try:
            from deterministic_horizon.tasks.permutation import PermutationTask

            task = PermutationTask(n_elements=n)
            solution = task.bfs_solve(list(range(n)), ranks, max_depth=task.max_depth())
            if solution is not None:
                return len(solution[0])
        except Exception:  # pragma: no cover - defensive; fall back to inversions
            pass
    return _inversions(ranks)


def _estimate_permutation(problem: str) -> int:
    arr = _parse_int_list(problem)
    if arr is not None and len(arr) >= 2:
        n = len(arr)
        swaps = _optimal_swaps(arr)
        # Reasoning depth = mutations (the swaps to reach the goal) + one
        # tracking step per element that must be read and verified.
        return swaps + n
    # No explicit array — prefer an explicit "<n> swaps/steps" magnitude, then
    # fall back to counting numeric operands / operators.
    hint = _magnitude_hint(problem)
    if hint is not None:
        return hint
    nums = len(_NUMBER_RE.findall(problem))
    ops = len(_OPERATOR_RE.findall(problem))
    return max(nums + ops, 1)


def _estimate_code(problem: str) -> int:
    # An explicit "N lines" / "N statements" count is the best available proxy
    # for how many sequential state updates must be tracked.
    m = re.search(r"(\d+)\s*(?:lines?|statements?|steps?|instructions?)", problem.lower())
    if m:
        return int(m.group(1))
    # Otherwise count statement separators and assignments as state mutations.
    mutations = problem.count(";") + problem.count("\n") + len(re.findall(r"=", problem))
    return max(mutations, len(_NUMBER_RE.findall(problem)), 1)


def _estimate_sql(problem: str) -> int:
    text = problem.lower()
    # Each join / filter / aggregation is a state-tracking step over the rows.
    clauses = sum(
        text.count(kw)
        for kw in ("join", "where", "group by", "having", "order by", "select", "and", "or")
    )
    return max(clauses, 1)


def _estimate_fsa(problem: str) -> int:
    # Depth tracks the length of the input string being simulated.
    m = re.search(r"(\d+)\s*(?:steps?|transitions?|symbols?|inputs?|characters?)", problem.lower())
    if m:
        return int(m.group(1))
    arr = _parse_int_list(problem)
    if arr is not None:
        return max(len(arr), 1)
    return max(len(_NUMBER_RE.findall(problem)), 1)


def _estimate_arithmetic(problem: str) -> int:
    # One step per operator in the expression chain.
    ops = len(_OPERATOR_RE.findall(problem))
    m = re.search(r"(\d+)\s*(?:steps?|operations?|terms?)", problem.lower())
    if m:
        return max(int(m.group(1)), ops, 1)
    return max(ops, len(_NUMBER_RE.findall(problem)) - 1, 1)


def _estimate_generic(problem: str) -> int:
    arr = _parse_int_list(problem)
    if arr is not None:
        return max(len(arr), 1)
    hint = _magnitude_hint(problem)
    if hint is not None:
        return hint
    nums = len(_NUMBER_RE.findall(problem))
    ops = len(_OPERATOR_RE.findall(problem))
    return max(nums + ops, 1)
