from __future__ import annotations

import functools

from deterministic_horizon.policy import delegation_decision, DelegationDecision

@functools.lru_cache(maxsize=1024)
def get_cached_decision(
    estimated_depth: int | float,
    model: str = "default",
    threshold: float = 0.5,
    tool_available: bool = True,
    tool_accuracy: float = 0.92,
    margin: float = 0.10,
) -> DelegationDecision:
    return delegation_decision(
        estimated_depth=estimated_depth,
        model=model,
        threshold=threshold,
        tool_available=tool_available,
        tool_accuracy=tool_accuracy,
        margin=margin,
    )
