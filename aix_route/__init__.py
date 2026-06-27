"""
Aix-Route — the production routing layer for the Deterministic Horizon.

This is the public import surface for the ``aix-route`` package:

    pip install aix-route
    from aix_route import should_delegate, fit_horizon, evaluate
    from aix_route.policy import horizon_table
    from aix_route.calibrate import fit_horizon
    from aix_route.tasks import PermutationTask

The implementation lives in the ``deterministic_horizon`` namespace (the name of
the research paper this is built on; the upstream authors are credited in
LICENSE and README). ``aix_route`` re-exports it and registers the submodules so
both the top-level names and the dotted submodule paths resolve. ``import
deterministic_horizon`` still works as a deprecated alias.
"""

from __future__ import annotations

import contextlib
import importlib
import sys

# Re-export the full public surface (and __version__) from the implementation.
import deterministic_horizon as _dh

__version__ = _dh.__version__

# Surface a few names at the top level that the implementation keeps in
# submodules, so `from aix_route import fit_horizon` works as documented.
try:
    from deterministic_horizon.calibrate import empirical_d_star, fit_horizon
except Exception:  # noqa: BLE001 - calibrate is dependency-light but guard anyway
    fit_horizon = empirical_d_star = None  # type: ignore[assignment]

# Make `aix_route.<submodule>` resolve to `deterministic_horizon.<submodule>` so
# `from aix_route.policy import ...` and `from aix_route.calibrate import ...`
# work without duplicating any code. Lazy submodules (models, analysis) are
# imported on demand via __getattr__ below.
_SUBMODULES = (
    "policy",
    "calibrate",
    "metrics",
    "tasks",
    "config",
    "runners",
    "cli",
    "estimators",
    "models",
    "analysis",
)


def _alias_submodule(name: str):
    module = importlib.import_module(f"deterministic_horizon.{name}")
    sys.modules[f"{__name__}.{name}"] = module
    return module


# Eagerly alias the dependency-free submodules so the common import paths work
# immediately; the heavy ones (models, analysis) are aliased on first access.
for _name in ("policy", "calibrate", "tasks", "config", "metrics"):
    # Optional deps may be missing for some submodules; alias what resolves.
    with contextlib.suppress(Exception):
        _alias_submodule(_name)


def __getattr__(name: str):
    # Lazily expose submodules (e.g. aix_route.models) and any top-level
    # attribute from the implementation package (e.g. aix_route.load_model).
    if name in _SUBMODULES:
        return _alias_submodule(name)
    return getattr(_dh, name)


def __dir__() -> list[str]:
    return sorted(set(dir(_dh)) | set(_SUBMODULES))


# Mirror the implementation package's public names for `from aix_route import *`,
# plus the calibration helpers surfaced at the top level here.
__all__ = sorted(set(getattr(_dh, "__all__", [])) | {"fit_horizon", "empirical_d_star"})
