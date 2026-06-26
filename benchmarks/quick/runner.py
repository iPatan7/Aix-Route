"""
Quick-validation benchmark runner.

The full paper sweep (500 instances × 3 runs × 12 models) costs thousands of
dollars and hours of wall-clock. This runner answers a narrower question
cheaply: *does the Deterministic Horizon ``d*`` come out where the paper says,
on the pipeline running on YOUR machine?*

It supports two modes:

``simulated`` (default)
    No API calls, no keys, no cost. Correctness for each instance is drawn from
    the paper's own closed-form decay curve
    (:func:`deterministic_horizon.policy.expected_neural_accuracy`) for the
    requested model. Re-fitting ``d*`` from those samples is a faithful,
    deterministic end-to-end check that the generation → fitting pipeline
    recovers the known horizon (≈22.0 for ``gpt-4o`` on PermutationProbe).

``api``
    Set with ``real=True``. Calls the real model via
    :func:`deterministic_horizon.evaluate`. Requires the relevant API key in the
    environment; with the small instance counts here it stays well under \\$50.

Either way the output is the same: the recovered ``d*``, the fit ``R²``, the
paper's reference ``d*``, and whether they agree within ``tolerance``.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from typing import Any

from deterministic_horizon.metrics import estimate_horizon
from deterministic_horizon.policy import MODEL_HORIZONS, expected_neural_accuracy, horizon_for
from deterministic_horizon.tasks import generate_instances

# Paper-reported Deterministic Horizons on PermutationProbe (policy table is the
# single source of truth; this maps the headline number used in the docs).
PAPER_REFERENCE_DSTAR = 22.0  # gpt-4o, PermutationProbe (paper §4, Theorem 4.8)


@dataclass(frozen=True)
class QuickResult:
    """Outcome of a quick benchmark run."""

    model: str
    task: str
    n_instances: int
    mode: str  # "simulated" | "api"
    d_star: float
    r_squared: float
    paper_d_star: float
    tolerance: float
    matches_paper: bool
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        verdict = "MATCH" if self.matches_paper else "MISMATCH"
        delta = abs(self.d_star - self.paper_d_star)
        return (
            f"[{verdict}] {self.model} / {self.task} ({self.mode}, n={self.n_instances})\n"
            f"  recovered d* = {self.d_star:.1f}   "
            f"(paper d* = {self.paper_d_star:.1f}, |Δ| = {delta:.1f} ≤ {self.tolerance:.1f}?)\n"
            f"  fit R²       = {self.r_squared:.3f}"
        )


def _paper_dstar(model: str, task: str) -> float:
    """Reference horizon to compare against.

    For ``gpt-4o`` we report the paper's headline 22.0; every other known model
    uses its tabulated ``d*`` from :data:`deterministic_horizon.policy.MODEL_HORIZONS`.
    """
    if model.lower() == "gpt-4o":
        return PAPER_REFERENCE_DSTAR
    return horizon_for(model)


def _simulate_results(
    instances: list[Any],
    model: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Draw Bernoulli correctness from the paper's decay curve for ``model``."""
    rng = random.Random(seed)
    results: list[dict[str, Any]] = []
    for inst in instances:
        depth = inst.optimal_depth
        p_correct = expected_neural_accuracy(depth, model)
        results.append(
            {
                "instance_id": inst.instance_id,
                "optimal_depth": depth,
                "correct": rng.random() < p_correct,
                "model": model,
                "condition": "C1",
            }
        )
    return results


def _api_results(
    instances: list[Any],
    model: str,
    task: str,
) -> list[dict[str, Any]]:
    """Evaluate the real model (C1, neural chain-of-thought)."""
    from deterministic_horizon import evaluate

    raw = evaluate(model, instances, conditions=("C1",), task=task, progress=True)
    # ``run_evaluation`` already attaches ``optimal_depth`` and ``correct``.
    return raw


def run_quick_benchmark(
    model: str = "gpt-4o",
    task: str = "permutation",
    n: int = 50,
    *,
    seed: int = 42,
    threshold: float = 0.5,
    tolerance: float = 3.0,
    depth_range: tuple[int, int] = (4, 28),
    depth_step: int = 4,
    real: bool = False,
) -> QuickResult:
    """Run the quick benchmark and return a :class:`QuickResult`.

    Parameters
    ----------
    model : str
        Model identifier (must be known to the policy table for the simulated
        mode, e.g. ``gpt-4o``).
    task : str
        Task name passed to :func:`deterministic_horizon.tasks.generate_instances`.
    n : int
        Number of instances to generate, spread across ``depth_range``.
    seed : int
        Seed for both instance generation and the correctness simulation, so
        runs are reproducible.
    threshold : float
        Accuracy threshold defining ``d*`` (the paper uses 0.5).
    tolerance : float
        Maximum allowed ``|recovered d* − paper d*|`` for ``matches_paper``.
    depth_range, depth_step :
        Depth grid for the generated instances. The default spans the
        PermutationProbe diameter for ``n=8``.
    real : bool
        When ``True`` call the real model API instead of simulating.
    """
    if model.lower() not in MODEL_HORIZONS and not real:
        known = ", ".join(sorted(m for m in MODEL_HORIZONS if m != "default"))
        raise ValueError(
            f"Unknown model {model!r} for simulated mode. Known: {known}. "
            f"Pass real=True to evaluate an arbitrary model via its API."
        )

    instances = generate_instances(
        task=task,
        n_instances=n,
        depth_range=depth_range,
        depth_step=depth_step,
        seed=seed,
    )
    if not instances:
        raise RuntimeError(f"no instances generated for task={task!r}")

    use_api = real and _has_api_key(model)
    if real and not use_api:
        raise RuntimeError(
            f"real=True but no API key found for model {model!r}. "
            f"Set the provider key in your environment or .env, or omit real=True "
            f"to run the (free) simulated validation."
        )

    if use_api:
        results = _api_results(instances, model, task)
        mode = "api"
    else:
        results = _simulate_results(instances, model, seed)
        mode = "simulated"

    horizon = estimate_horizon(results, threshold=threshold)
    d_star = float(horizon["d_star"])
    r_squared = float(horizon.get("r_squared", float("nan")))
    paper = _paper_dstar(model, task)

    return QuickResult(
        model=model,
        task=task,
        n_instances=len(instances),
        mode=mode,
        d_star=d_star,
        r_squared=r_squared,
        paper_d_star=paper,
        tolerance=tolerance,
        matches_paper=abs(d_star - paper) <= tolerance,
        threshold=threshold,
    )


def _has_api_key(model: str) -> bool:
    """Best-effort check that a usable API key exists for ``model``'s provider."""
    # Load .env if present (never hard-fail if python-dotenv is missing).
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:  # pragma: no cover - optional dependency
        pass

    m = model.lower()
    if m.startswith(("gpt", "o1", "o3")):
        return bool(os.getenv("OPENAI_API_KEY"))
    if m.startswith("claude"):
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    if m.startswith("deepseek"):
        return bool(os.getenv("DEEPSEEK_API_KEY"))
    if m.startswith("gemini"):
        return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    if m.startswith("together"):
        return bool(os.getenv("TOGETHER_API_KEY"))
    # Unknown provider: assume a key may exist and let the API layer error out.
    return True
