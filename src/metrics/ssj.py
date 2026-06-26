"""State-Space Jaccard (SSJ) metric implementation."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class SSJResult:
    """Container for SSJ metric results."""

    ssj: float
    precision: float
    recall: float

    # Additional diagnostics
    true_states: set[str]
    model_states: set[str]
    intersection_size: int
    union_size: int

    @property
    def f1(self) -> float:
        """Compute F1 score from precision and recall."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ssj": self.ssj,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "intersection_size": self.intersection_size,
            "union_size": self.union_size,
            "true_state_count": len(self.true_states),
            "model_state_count": len(self.model_states),
        }


def compute_ssj(
    true_states: Sequence[Any],
    model_states: Sequence[Any],
    state_to_str: Callable[[Any], str] | None = None,
) -> SSJResult:
    """
    Compute State-Space Jaccard metric with precision/recall decomposition.

    The SSJ metric measures how well the model's claimed state trajectory
    overlaps with the ground truth trajectory. Unlike simple accuracy,
    SSJ captures partial correctness and distinguishes between:
    - Capability failures (both precision and recall decay)
    - Preference failures (only recall decays, precision stays high)

    Args:
        true_states: Ground truth state sequence
        model_states: Model's claimed state sequence
        state_to_str: Optional function to convert states to strings
            for comparison. If None, uses str().

    Returns:
        SSJResult containing SSJ, precision, recall, and diagnostics

    Mathematical Definition:
        SSJ(d) = |S_true ∩ S_model| / |S_true ∪ S_model|
        Precision = |S_true ∩ S_model| / |S_model|
        Recall = |S_true ∩ S_model| / |S_true|
    """
    if state_to_str is None:
        state_to_str = str

    # Convert to sets of string representations
    true_set = {state_to_str(s) for s in true_states}
    model_set = {state_to_str(s) for s in model_states}

    # Handle empty sets
    if not true_set and not model_set:
        return SSJResult(
            ssj=1.0,
            precision=1.0,
            recall=1.0,
            true_states=true_set,
            model_states=model_set,
            intersection_size=0,
            union_size=0,
        )

    if not model_set:
        return SSJResult(
            ssj=0.0,
            precision=0.0,
            recall=0.0,
            true_states=true_set,
            model_states=model_set,
            intersection_size=0,
            union_size=len(true_set),
        )

    if not true_set:
        return SSJResult(
            ssj=0.0,
            precision=0.0,
            recall=0.0,
            true_states=true_set,
            model_states=model_set,
            intersection_size=0,
            union_size=len(model_set),
        )

    # Compute set operations
    intersection = true_set & model_set
    union = true_set | model_set

    # Compute metrics
    ssj = len(intersection) / len(union)
    precision = len(intersection) / len(model_set)
    recall = len(intersection) / len(true_set)

    return SSJResult(
        ssj=ssj,
        precision=precision,
        recall=recall,
        true_states=true_set,
        model_states=model_set,
        intersection_size=len(intersection),
        union_size=len(union),
    )


def compute_ssj_at_depth(
    true_states: Sequence[Any],
    model_states: Sequence[Any],
    depth: int,
    state_to_str: Callable[[Any], str] | None = None,
) -> SSJResult:
    """
    Compute SSJ at a specific depth (using states up to that depth).

    Args:
        true_states: Full ground truth trajectory
        model_states: Full model trajectory
        depth: Compute SSJ using states up to this depth
        state_to_str: Optional state to string converter

    Returns:
        SSJResult at the specified depth
    """
    true_prefix = true_states[: depth + 1]
    model_prefix = model_states[: depth + 1]
    return compute_ssj(true_prefix, model_prefix, state_to_str)


def compute_ssj_trajectory(
    true_states: Sequence[Any],
    model_states: Sequence[Any],
    state_to_str: Callable[[Any], str] | None = None,
) -> list[SSJResult]:
    """
    Compute SSJ at each depth to show trajectory of state tracking quality.

    Args:
        true_states: Ground truth state sequence
        model_states: Model's claimed state sequence
        state_to_str: Optional state to string converter

    Returns:
        List of SSJResult, one per depth from 0 to max(len(true), len(model))
    """
    max_len = max(len(true_states), len(model_states))

    results = []
    for depth in range(max_len):
        result = compute_ssj_at_depth(true_states, model_states, depth, state_to_str)
        results.append(result)

    return results


def diagnose_failure_mode(ssj_result: SSJResult) -> str:
    """
    Diagnose failure mode based on precision/recall pattern.

    The key insight from the paper: if failure is preference-based
    (Simplicity Bias), precision remains high while recall decays.
    If failure is capability-based (Decoherence), both decay together.

    Args:
        ssj_result: SSJ computation result

    Returns:
        Diagnosis string: "capability_failure", "preference_failure",
        "success", or "unknown"
    """
    # Thresholds
    HIGH_THRESHOLD = 0.7
    LOW_THRESHOLD = 0.3

    if ssj_result.precision > HIGH_THRESHOLD and ssj_result.recall > HIGH_THRESHOLD:
        return "success"

    if ssj_result.precision > HIGH_THRESHOLD and ssj_result.recall < LOW_THRESHOLD:
        # High precision, low recall = model produces valid but incomplete states
        # This indicates preference failure (choosing not to continue)
        return "preference_failure"

    if ssj_result.precision < LOW_THRESHOLD and ssj_result.recall < LOW_THRESHOLD:
        # Both low = model drifts into fictitious states
        # This indicates capability failure (decoherence)
        return "capability_failure"

    # Mixed cases
    precision_recall_ratio = (
        ssj_result.precision / ssj_result.recall if ssj_result.recall > 0 else float("inf")
    )

    if precision_recall_ratio > 2:
        return "preference_failure"
    elif precision_recall_ratio < 0.5:
        return "capability_failure"

    return "unknown"
