"""Step-to-First-Error (SFE) metric implementation."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class SFEResult:
    """Container for SFE metric results."""

    sfe: int | None  # None if no error found
    total_steps: int
    error_step_state: Any | None
    expected_state: Any | None

    @property
    def normalized_sfe(self) -> float:
        """SFE normalized by total steps (0 to 1, higher is better)."""
        if self.sfe is None:
            return 1.0
        if self.total_steps == 0:
            return 0.0
        return self.sfe / self.total_steps

    @property
    def has_error(self) -> bool:
        """Whether an error was found."""
        return self.sfe is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sfe": self.sfe,
            "total_steps": self.total_steps,
            "normalized_sfe": self.normalized_sfe,
            "has_error": self.has_error,
        }


def compute_sfe(
    true_states: Sequence[Any],
    model_states: Sequence[Any],
    state_equal: Callable[[Any, Any], bool] | None = None,
) -> SFEResult:
    """
    Compute Step-to-First-Error metric.

    SFE measures how many steps the model correctly tracks state
    before making its first error. This is crucial for understanding
    the depth at which decoherence begins.

    Args:
        true_states: Ground truth state sequence
        model_states: Model's claimed state sequence
        state_equal: Optional function to compare states.
            If None, uses == operator.

    Returns:
        SFEResult containing SFE and diagnostics

    Definition:
        SFE = min{i : s_i^model ≠ s_i^true}

        If s_i^model = s_i^true for all i, SFE = None (no error)
    """
    if state_equal is None:

        def state_equal(a, b):
            return a == b

    total_steps = max(len(true_states), len(model_states))

    # Compare step by step
    min_len = min(len(true_states), len(model_states))

    for i in range(min_len):
        if not state_equal(true_states[i], model_states[i]):
            return SFEResult(
                sfe=i,
                total_steps=total_steps,
                error_step_state=model_states[i],
                expected_state=true_states[i],
            )

    # Check if model trace is shorter (implicit error)
    if len(model_states) < len(true_states):
        return SFEResult(
            sfe=len(model_states),
            total_steps=total_steps,
            error_step_state=None,  # Model stopped early
            expected_state=true_states[len(model_states)],
        )

    # No error found
    return SFEResult(
        sfe=None,
        total_steps=total_steps,
        error_step_state=None,
        expected_state=None,
    )


def compute_sfe_distribution(
    results: list[tuple[Sequence[Any], Sequence[Any]]],
    state_equal: Callable[[Any, Any], bool] | None = None,
) -> dict[str, Any]:
    """
    Compute SFE distribution across multiple instances.

    Args:
        results: List of (true_states, model_states) tuples
        state_equal: Optional state comparison function

    Returns:
        Dictionary with distribution statistics
    """
    sfe_values = []
    error_count = 0

    for true_states, model_states in results:
        result = compute_sfe(true_states, model_states, state_equal)
        if result.has_error:
            sfe_values.append(result.sfe)
            error_count += 1
        else:
            sfe_values.append(result.total_steps)  # Perfect = total steps

    import numpy as np

    sfe_array = np.array(sfe_values)

    return {
        "mean": float(np.mean(sfe_array)),
        "std": float(np.std(sfe_array)),
        "median": float(np.median(sfe_array)),
        "min": int(np.min(sfe_array)),
        "max": int(np.max(sfe_array)),
        "error_rate": error_count / len(results),
        "perfect_rate": 1 - error_count / len(results),
        "n_samples": len(results),
    }


def sfe_by_depth(
    results: list[dict[str, Any]],
    depth_key: str = "optimal_depth",
    sfe_key: str = "step_to_first_error",
) -> dict[int, dict[str, float]]:
    """
    Compute SFE statistics grouped by problem depth.

    Args:
        results: List of result dictionaries
        depth_key: Key for depth in results
        sfe_key: Key for SFE in results

    Returns:
        Dictionary mapping depth to SFE statistics
    """
    from collections import defaultdict

    import numpy as np

    # Group by depth
    by_depth = defaultdict(list)
    for result in results:
        depth = result.get(depth_key, 0)
        sfe = result.get(sfe_key)
        if sfe is not None:
            by_depth[depth].append(sfe)
        else:
            # No error = SFE equals depth
            by_depth[depth].append(depth)

    # Compute stats per depth
    stats = {}
    for depth, sfe_values in sorted(by_depth.items()):
        arr = np.array(sfe_values)
        stats[depth] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "normalized_mean": float(np.mean(arr / depth)) if depth > 0 else 1.0,
            "n": len(sfe_values),
        }

    return stats
