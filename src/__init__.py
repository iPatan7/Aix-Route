"""
Deterministic Horizon — When Extended Reasoning Fails and Tool Delegation
Becomes Necessary.

Official implementation accompanying the Deterministic Horizon paper. The
package exposes four primary surface areas:

- ``tasks``   — generators and evaluators for state-space search problems
- ``models``  — uniform interface across OpenAI / Anthropic / DeepSeek / local
- ``metrics`` — SSJ, SFE, accuracy decay, horizon estimation
- ``analysis``— figure & table generation for the paper's plots
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("deterministic-horizon")
except PackageNotFoundError:  # editable install / source checkout
    __version__ = "1.0.1"

from deterministic_horizon.config import Config, load_config
from deterministic_horizon.metrics import (
    accuracy_by_depth,
    bootstrap_ci,
    compute_effect_size,
    compute_sfe,
    compute_ssj,
    cross_model_correlation,
    estimate_horizon,
    fit_decoherence_model,
)
from deterministic_horizon.policy import (
    DEFAULT_TOOL_ACCURACY,
    MODEL_HORIZONS,
    DelegationDecision,
    delegation_decision,
    expected_neural_accuracy,
    horizon_for,
    horizon_table,
    recommend_model,
    should_delegate,
    should_delegate_batch,
)
from deterministic_horizon.tasks import (
    ArithmeticTask,
    BaseTask,
    FSATask,
    PermutationTask,
    generate_instances,
    load_task,
)


# Model interfaces are imported lazily so that the package can be imported
# without the optional `openai`/`anthropic`/`torch` dependencies installed.
def __getattr__(name: str):
    if name in {
        "BaseModel",
        "ModelResponse",
        "OpenAIModel",
        "AnthropicModel",
        "DeepSeekModel",
        "LocalModel",
        "load_model",
        "MODEL_REGISTRY",
    }:
        from deterministic_horizon import models as _models

        return getattr(_models, name)
    if name in {"generate_figures", "generate_tables", "decay_curve", "plot_model_horizons"}:
        from deterministic_horizon import analysis as _analysis

        return getattr(_analysis, name)
    raise AttributeError(f"module 'deterministic_horizon' has no attribute {name!r}")


# Public, dependency-free surface
def evaluate(*args, **kwargs):  # pragma: no cover - thin convenience wrapper
    """Convenience wrapper that mirrors the README's Python API example."""
    from deterministic_horizon.runners import run_evaluation

    return run_evaluation(*args, **kwargs)


__all__ = [
    "__version__",
    "Config",
    "load_config",
    "BaseTask",
    "PermutationTask",
    "FSATask",
    "ArithmeticTask",
    "generate_instances",
    "load_task",
    "compute_ssj",
    "compute_sfe",
    "accuracy_by_depth",
    "estimate_horizon",
    "cross_model_correlation",
    "bootstrap_ci",
    "compute_effect_size",
    "fit_decoherence_model",
    # Practitioner decision helpers
    "should_delegate",
    "should_delegate_batch",
    "delegation_decision",
    "expected_neural_accuracy",
    "horizon_for",
    "horizon_table",
    "recommend_model",
    "DelegationDecision",
    "MODEL_HORIZONS",
    "DEFAULT_TOOL_ACCURACY",
    # Re-exported lazily via __getattr__
    "BaseModel",
    "ModelResponse",
    "OpenAIModel",
    "AnthropicModel",
    "DeepSeekModel",
    "LocalModel",
    "load_model",
    "MODEL_REGISTRY",
    "generate_figures",
    "generate_tables",
    "decay_curve",
    "plot_model_horizons",
    "evaluate",
]
