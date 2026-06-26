"""Quick, low-cost validation benchmark (``python -m benchmarks.quick``).

Recovers a model's Deterministic Horizon ``d*`` from a stripped-down run
(50 instances, 1 pass) and checks it against the paper's value. Runs in
minutes for under \\$50 — or for free in the default simulated mode.
"""

from __future__ import annotations

from benchmarks.quick.runner import PAPER_REFERENCE_DSTAR, QuickResult, run_quick_benchmark

__all__ = ["run_quick_benchmark", "QuickResult", "PAPER_REFERENCE_DSTAR"]
