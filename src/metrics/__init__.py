"""Evaluation metrics for state-space reasoning tasks."""

from deterministic_horizon.metrics.sfe import SFEResult, compute_sfe
from deterministic_horizon.metrics.ssj import SSJResult, compute_ssj
from deterministic_horizon.metrics.statistics import (
    accuracy_by_depth,
    bootstrap_ci,
    compute_effect_size,
    cross_model_correlation,
    estimate_horizon,
    fit_decoherence_model,
)

__all__ = [
    "compute_ssj",
    "SSJResult",
    "compute_sfe",
    "SFEResult",
    "accuracy_by_depth",
    "estimate_horizon",
    "cross_model_correlation",
    "bootstrap_ci",
    "compute_effect_size",
    "fit_decoherence_model",
]
