"""Smoke tests — these must pass before *anything* else is allowed to fail."""

from __future__ import annotations


def test_top_level_imports_without_optional_deps():
    """``import deterministic_horizon`` must succeed even without openai/torch."""
    import deterministic_horizon as dh

    assert dh.__version__
    # The public surface advertised in README must be present.
    for name in [
        "PermutationTask",
        "FSATask",
        "ArithmeticTask",
        "generate_instances",
        "compute_ssj",
        "estimate_horizon",
        "fit_decoherence_model",
    ]:
        assert hasattr(dh, name), f"missing public symbol: {name}"


def test_analysis_module_loads():
    """The analysis module was previously imported but nonexistent."""
    from deterministic_horizon import analysis

    assert callable(analysis.generate_figures)
    assert callable(analysis.generate_tables)
    assert callable(analysis.decay_curve)


def test_models_lazy_registry_lists_providers():
    """Lazy MODEL_REGISTRY must enumerate without triggering optional imports."""
    from deterministic_horizon.models import MODEL_REGISTRY

    assert "gpt-4o" in MODEL_REGISTRY
    assert "claude-4.5-sonnet" in MODEL_REGISTRY
    assert "deepseek-r1" in MODEL_REGISTRY
