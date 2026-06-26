"""
Reasoning-depth estimation for the Deterministic Horizon policy layer.

Turns a free-text problem into the two inputs the policy needs:

- :func:`classify_task`  — which task family (permutation, code, SQL, ...)
- :func:`estimate_depth` — how many deterministic state-tracking steps it needs

so callers can go straight from a problem string to
:func:`deterministic_horizon.policy.should_delegate` without hand-estimating
depth.

>>> from deterministic_horizon.estimators import estimate_depth, classify_task
>>> classify_task("Trace variable x through 20 lines of Python") == "code"
True
>>> estimate_depth("Sort [5, 2, 8, 1] using adjacent swaps", task_type="permutation")
8
"""

from __future__ import annotations

from deterministic_horizon.estimators.depth_estimator import estimate_depth, set_learned_estimator
from deterministic_horizon.estimators.task_classifier import (
    TaskType,
    classify_task,
    task_scores,
)

# Intra-package modules use absolute ``deterministic_horizon.estimators`` imports
# to match the rest of the codebase; the package is also importable as
# ``src.estimators`` from the repo root (see benchmarks/ and the acceptance
# snippet). Both names resolve to this directory.

__all__ = [
    "estimate_depth",
    "set_learned_estimator",
    "classify_task",
    "task_scores",
    "TaskType",
]
