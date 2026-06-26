"""
Practitioner-facing decision helper: *should my agent think harder, or call a tool?*

This module turns the paper's theoretical bound into a one-line API that agentic
systems can call at planning time. The math is **exactly** the paper's:

* Per-step error (Definition 4.1 / Theorem 4.2, "Decoherence Bound"):

      ε(d) = ε₀ + γ · d / L_eff

* Closed-form accuracy decay obtained by integrating it (Theorem 4.2):

      P(correct at depth d) ≈ exp(−d·ε₀ − γ·d(d+1) / (2·L_eff))

* Deterministic Horizon (Theorem 4.8), the depth at which P = α:

      d* = (−ε₀·L_eff + √(ε₀²·L_eff² + 2γ·L_eff·ln(1/α))) / γ

The constants are the paper's: γ = 0.15 and, for GPT-4o, ε₀ = 0.02,
L_eff = 150, which yields d* ≈ 22 for GPT-4o (§4, Theorem 4.8). The
*effective decoherence length* L_eff = O(10²) steps is far smaller than the raw
context window L = O(10⁵) tokens — using the raw L would wash out the quadratic
term and incorrectly predict near-perfect accuracy (paper, Appendix on numerical
examples). We therefore parameterise every model by its measured d* and a
per-model (ε₀, L_eff) pair calibrated so the decay curve crosses α = 0.5 exactly
at that d*.

Example
-------
>>> from deterministic_horizon import should_delegate
>>> should_delegate(estimated_depth=8)         # short — think it through
False
>>> should_delegate(estimated_depth=35)        # past the horizon — delegate
True
>>> decision = delegation_decision(
...     estimated_depth=30,
...     model="claude-4.5-opus",
...     tool_available=True,
... )
>>> decision.delegate, round(decision.expected_neural_accuracy, 2)
(True, 0.45)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Paper-canonical constants (§4, Theorems 4.2 & 4.8).
#
#   γ      attention decay rate, shared across models.
#   ε₀     baseline per-step error (paper MLE 0.020, 95% CI [0.017, 0.023]).
#   L_eff  *effective* decoherence length, O(10²) steps — NOT the raw context
#          window. For GPT-4o the paper reports L_eff = 150, giving d* ≈ 22.3.
# ---------------------------------------------------------------------------

GAMMA: float = 0.15
ALPHA_DEFAULT: float = 0.5  # success threshold used to define d*


def _l_eff_for(eps0: float, d_star: float, alpha: float = ALPHA_DEFAULT) -> float:
    """
    Effective decoherence length L_eff that makes the Theorem 4.2 decay curve
    pass through accuracy ``alpha`` at depth ``d_star``.

    Solving ``exp(−d*·ε₀ − γ·d*(d*+1)/(2·L_eff)) = α`` for L_eff:

        L_eff = γ·d*(d*+1) / (2·(ln(1/α) − ε₀·d*))

    This keeps the per-model decay curve, the reported d*, and ``horizon_for``
    in exact agreement.
    """
    denom = math.log(1.0 / alpha) - eps0 * d_star
    if denom <= 0:
        raise ValueError(f"ε₀·d* must be below ln(1/α); got ε₀={eps0}, d*={d_star}, α={alpha}")
    return GAMMA * d_star * (d_star + 1.0) / (2.0 * denom)


# ---------------------------------------------------------------------------
# Per-model decoherence parameters.
#
# ``d_star`` values are the paper's measured Deterministic Horizons on
# PermutationProbe (Table 3 "Main results" and Table 5 "Architecture ablation").
# Only the models for which the paper reports a d* on PermutationProbe are
# listed; every other identifier falls back to ``"default"`` (the midpoint of
# the measured d* ∈ [19, 31] range). ε₀ is held near the paper's fitted 0.020
# (reasoning-tuned models slightly lower, small open-weight models slightly
# higher); L_eff is then derived so the curve crosses 0.5 exactly at d*.
# ---------------------------------------------------------------------------

_MODEL_EPS0_DSTAR: dict[str, tuple[float, float]] = {
    # General-purpose (closed) ------------------------------------------------
    "gpt-4o": (0.020, 22.0),  # paper canonical: ε₀=0.02, L_eff=150 → d*≈22
    "claude-4.5-opus": (0.018, 27.0),
    # Reasoning-specialised ---------------------------------------------------
    "o3-mini": (0.014, 31.0),  # highest horizon in the suite
    "deepseek-r1": (0.015, 29.0),
    # Open-weight (published H, d_h — used for the √(d_h·H) scaling check) -----
    "llama-3.1-8b": (0.022, 20.0),
    "llama-3.3-70b": (0.018, 28.0),
    "qwen-2.5-7b": (0.023, 19.0),
    "qwen-2.5-72b": (0.018, 28.0),
    # Cross-model fallback ("average frontier model", midpoint of [19, 31]) ---
    "default": (0.020, 24.0),
}

MODEL_HORIZONS: dict[str, dict[str, float]] = {
    name: {
        "eps0": eps0,
        "d_star": d_star,
        "l_eff": _l_eff_for(eps0, d_star),
        "gamma": GAMMA,
    }
    for name, (eps0, d_star) in _MODEL_EPS0_DSTAR.items()
}

# Empirical mean tool-integrated (C3) accuracy across models / domains
# (paper headline: 86–94%; cross-domain mean ≈ 0.92).
DEFAULT_TOOL_ACCURACY: float = 0.92


@dataclass(frozen=True)
class DelegationDecision:
    """Structured result of :func:`delegation_decision`."""

    delegate: bool
    estimated_depth: int
    model: str
    expected_neural_accuracy: float
    expected_tool_accuracy: float
    horizon: float
    margin: float
    reason: Literal[
        "above_horizon",
        "below_horizon",
        "tool_unavailable",
        "tool_dominates_by_margin",
    ]

    def explain(self) -> str:
        """Human-readable one-line explanation, useful in agent logs."""
        if self.reason == "tool_unavailable":
            return (
                f"No tool available; falling back to neural reasoning "
                f"(expected accuracy {self.expected_neural_accuracy:.0%} at depth "
                f"d={self.estimated_depth})."
            )
        verb = "delegate" if self.delegate else "reason neurally"
        return (
            f"At estimated depth d={self.estimated_depth}, model {self.model!r} "
            f"is expected to reach {self.expected_neural_accuracy:.0%} via CoT "
            f"vs. {self.expected_tool_accuracy:.0%} via tools "
            f"(horizon d*={self.horizon:.0f}). → {verb}."
        )

    def to_dict(self) -> dict[str, object]:
        """Convert decision to a dictionary for JSON serialization."""
        return {
            "delegate": self.delegate,
            "estimated_depth": self.estimated_depth,
            "model": self.model,
            "expected_neural_accuracy": self.expected_neural_accuracy,
            "expected_tool_accuracy": self.expected_tool_accuracy,
            "horizon": self.horizon,
            "margin": self.margin,
            "reason": self.reason,
        }

def _params_for(model: str) -> dict[str, float]:
    return MODEL_HORIZONS.get(model.lower(), MODEL_HORIZONS["default"])


def expected_neural_accuracy(
    depth: int | float,
    model: str = "default",
    *,
    eps0: float | None = None,
    l_eff: float | None = None,
) -> float:
    """
    Expected accuracy of a neural chain-of-thought at the given depth.

    Implements the closed-form decay of Theorem 4.2:
    ``P(correct) ≈ exp(−d·ε₀ − γ·d(d+1) / (2·L_eff))``.

    Parameters
    ----------
    depth : int | float
        Estimated reasoning depth (number of state-tracking steps).
    model : str
        Model identifier — see :data:`MODEL_HORIZONS` for known names. Unknown
        names fall back to the ``"default"`` (cross-model average) parameters.
    eps0, l_eff : float, optional
        Override the per-model decoherence parameters directly. When provided,
        the corresponding value from ``model`` is ignored. ``γ`` is the shared
        paper constant :data:`GAMMA`.

    Returns
    -------
    float
        Predicted probability of a correct final answer, in ``[0, 1]``.
    """
    if depth < 0:
        raise ValueError(f"depth must be non-negative, got {depth!r}")
    params = _params_for(model)
    e0 = eps0 if eps0 is not None else params["eps0"]
    le = l_eff if l_eff is not None else params["l_eff"]
    d = float(depth)
    return float(math.exp(-d * e0 - GAMMA * d * (d + 1.0) / (2.0 * le)))


def horizon_for(model: str = "default") -> float:
    """Return the Deterministic Horizon d* for ``model`` (or the default)."""
    return float(_params_for(model)["d_star"])


def should_delegate(
    estimated_depth: int | float,
    model: str = "default",
    *,
    threshold: float = 0.5,
    tool_available: bool = True,
    tool_accuracy: float = DEFAULT_TOOL_ACCURACY,
    margin: float = 0.10,
) -> bool:
    """
    Cheap boolean: at this depth, should the agent route to a tool?

    Returns ``True`` when *either* expected neural accuracy is below
    ``threshold`` *or* a tool is expected to beat neural reasoning by more
    than ``margin`` — and a tool is available. This mirrors
    :func:`delegation_decision`'s logic, so the cheap boolean and the full
    structured result always agree.

    Use this when you just need a fast branch in your agent loop; use
    :func:`delegation_decision` when you want the full justification
    (e.g. to log it for human review).
    """
    return delegation_decision(
        estimated_depth=estimated_depth,
        model=model,
        tool_available=tool_available,
        tool_accuracy=tool_accuracy,
        threshold=threshold,
        margin=margin,
    ).delegate


def delegation_decision(
    estimated_depth: int | float,
    model: str = "default",
    *,
    tool_available: bool = True,
    tool_accuracy: float = DEFAULT_TOOL_ACCURACY,
    threshold: float = 0.5,
    margin: float = 0.10,
) -> DelegationDecision:
    """
    Full delegation decision with expected accuracy on both branches.

    Delegate when **either** of these holds (and a tool is available):

    1. Neural accuracy at this depth is below ``threshold`` (we're past d*).
    2. Tool accuracy beats neural accuracy by more than ``margin`` —
       *don't* think harder just to break even.

    Parameters
    ----------
    estimated_depth : int | float
        Depth estimate from your planner. If you don't have one, run BFS or
        a learned depth-estimator on the problem first.
    model : str
        Frontier model identifier. Unknown names use cross-model defaults.
    tool_available : bool
        If False, the decision is forced to *not* delegate regardless of d.
    tool_accuracy : float
        Empirical accuracy you've seen for your tool on similar tasks.
        Defaults to the cross-domain mean of 0.92 (paper §5).
    threshold : float
        Below this expected neural accuracy, automatically delegate.
    margin : float
        Even above ``threshold``, delegate when tool beats neural by this much.

    Returns
    -------
    DelegationDecision
        Structured result. Call ``.explain()`` for a human-readable summary.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {threshold!r}")
    if not 0.0 <= margin < 1.0:
        raise ValueError(f"margin must be in [0, 1), got {margin!r}")

    neural = expected_neural_accuracy(estimated_depth, model)
    tool = float(tool_accuracy) if tool_available else 0.0
    horizon = horizon_for(model)

    if not tool_available:
        return DelegationDecision(
            delegate=False,
            estimated_depth=int(estimated_depth),
            model=model,
            expected_neural_accuracy=neural,
            expected_tool_accuracy=0.0,
            horizon=horizon,
            margin=margin,
            reason="tool_unavailable",
        )

    if neural < threshold:
        return DelegationDecision(
            delegate=True,
            estimated_depth=int(estimated_depth),
            model=model,
            expected_neural_accuracy=neural,
            expected_tool_accuracy=tool,
            horizon=horizon,
            margin=margin,
            reason="above_horizon",
        )

    if tool - neural > margin:
        return DelegationDecision(
            delegate=True,
            estimated_depth=int(estimated_depth),
            model=model,
            expected_neural_accuracy=neural,
            expected_tool_accuracy=tool,
            horizon=horizon,
            margin=margin,
            reason="tool_dominates_by_margin",
        )

    return DelegationDecision(
        delegate=False,
        estimated_depth=int(estimated_depth),
        model=model,
        expected_neural_accuracy=neural,
        expected_tool_accuracy=tool,
        horizon=horizon,
        margin=margin,
        reason="below_horizon",
    )


def should_delegate_batch(
    depths: Sequence[int | float],
    model: str = "default",
    **kwargs: object,
) -> list[bool]:
    """
    Vectorised :func:`should_delegate` for a planner that has a *list* of
    candidate subproblem depths (e.g. one per sub-goal in a decomposition).

    Returns one boolean per depth, in order. Extra keyword arguments are
    forwarded verbatim to :func:`should_delegate` (``threshold``,
    ``tool_accuracy``, ``margin``, ``tool_available``).

    >>> should_delegate_batch([5, 8, 35], model="gpt-4o")
    [False, False, True]
    """
    return [should_delegate(d, model=model, **kwargs) for d in depths]  # type: ignore[arg-type]


def horizon_table() -> list[dict[str, float | str]]:
    """
    Tabular summary of every known model's decoherence parameters, sorted by
    horizon (deepest-reasoning model last). Handy for logging, the ``dh
    horizons`` CLI command, or building a comparison plot.

    Each row has ``model``, ``eps0``, ``l_eff``, ``d_star`` and ``gamma``.
    """
    rows = [
        {
            "model": name,
            "eps0": params["eps0"],
            "l_eff": params["l_eff"],
            "d_star": params["d_star"],
            "gamma": params["gamma"],
        }
        for name, params in MODEL_HORIZONS.items()
    ]
    return sorted(rows, key=lambda r: r["d_star"])  # type: ignore[arg-type,return-value]


def recommend_model(
    estimated_depth: int | float,
    *,
    candidates: Sequence[str] | None = None,
    threshold: float = ALPHA_DEFAULT,
) -> tuple[str | None, float]:
    """
    Suggest the *least over-powered* model whose expected neural accuracy at
    ``estimated_depth`` still clears ``threshold`` — i.e. a model that can
    reason through the subproblem without delegating.

    Among ``candidates`` (default: all known models except the synthetic
    ``"default"`` fallback), returns the one with the *smallest* horizon that
    still clears the threshold. If none clears it, returns ``(None,
    best_accuracy)`` so the caller knows to delegate instead.

    Returns ``(model_name_or_None, expected_neural_accuracy_of_that_model)``.

    >>> recommend_model(8)[0] is not None      # some model handles depth 8
    True
    >>> recommend_model(500)[0] is None        # nobody reasons 500 steps deep
    True
    """
    names = (
        list(candidates)
        if candidates is not None
        else [m for m in MODEL_HORIZONS if m != "default"]
    )
    viable: list[tuple[float, str, float]] = []
    best_acc = 0.0
    for name in names:
        acc = expected_neural_accuracy(estimated_depth, name)
        if acc > best_acc:
            best_acc = acc
        if acc >= threshold:
            viable.append((horizon_for(name), name, acc))
    if not viable:
        return None, best_acc
    viable.sort()  # smallest horizon first — least over-powered model that works
    _, name, acc = viable[0]
    return name, acc


__all__ = [
    "GAMMA",
    "MODEL_HORIZONS",
    "DEFAULT_TOOL_ACCURACY",
    "DelegationDecision",
    "expected_neural_accuracy",
    "horizon_for",
    "should_delegate",
    "should_delegate_batch",
    "delegation_decision",
    "horizon_table",
    "recommend_model",
]
