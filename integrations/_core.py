"""
Shared policy access for the framework integrations.

The integrations must work in three situations, in priority order:

1. ``deterministic_horizon`` is pip-installed (``pip install -e .``) — the
   normal case. We import the policy helpers straight from the package.
2. Only a source checkout is on disk (no install, no ``numpy``). The top-level
   package pulls in ``numpy``/``scipy`` via its ``__init__``, so importing the
   whole package would fail. We side-step that by importing the dependency-free
   ``policy`` module directly from ``../src``.
3. Neither resolves. We raise a clear, actionable error instead of an opaque
   ``ImportError`` deep inside a framework callback.

Every integration imports :func:`should_delegate` / :func:`delegation_decision`
from here, so the resolution logic lives in exactly one place.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from deterministic_horizon.policy import DelegationDecision


_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"


def _load_policy() -> Any:
    """Return the ``policy`` module, however the repo happens to be laid out."""
    # 1) Installed package.
    try:
        return importlib.import_module("deterministic_horizon.policy")
    except Exception:  # noqa: BLE001 - any failure falls through to the source path
        pass

    # 2) Source checkout: load src/policy.py in isolation. policy.py only needs
    #    the standard library (math), so this works without numpy/scipy.
    policy_path = _SRC_DIR / "policy.py"
    if policy_path.is_file():
        spec = importlib.util.spec_from_file_location(
            "_dh_policy_standalone", policy_path
        )
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            sys.modules.setdefault("_dh_policy_standalone", module)
            spec.loader.exec_module(module)
            return module

    raise ImportError(
        "Could not import the Deterministic Horizon policy layer. "
        "Install the package with `pip install -e .` from the repo root, "
        "or run from a checkout that contains `src/policy.py`."
    )


_policy = _load_policy()

should_delegate: Callable[..., bool] = _policy.should_delegate
delegation_decision: Callable[..., DelegationDecision] = _policy.delegation_decision
horizon_for: Callable[..., float] = _policy.horizon_for

__all__ = ["should_delegate", "delegation_decision", "horizon_for"]
