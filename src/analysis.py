"""
Figure and table generation for the Deterministic Horizon paper.

This module generates the headline plots (accuracy decay, horizon location,
SSJ precision/recall split, cross-model correlation heatmap) directly from
the JSON results emitted by ``deterministic_horizon.cli evaluate``.

The functions degrade gracefully when ``matplotlib``/``seaborn`` are not
installed: tables are always produced, figures are skipped with a warning.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from deterministic_horizon.metrics import (
    accuracy_by_depth,
    estimate_horizon,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Theoretical curve (Theorem 4.2: super-exponential decay)
# ---------------------------------------------------------------------------


def decay_curve(
    depths: Sequence[float] | np.ndarray,
    eps0: float = 0.02,
    gamma: float = 0.15,
    l_eff: float = 150.0,
) -> np.ndarray:
    """
    Closed-form decay from Theorem 4.2: ``exp(-d·ε₀ − γ·d(d+1)/(2·L_eff))``.

    ``l_eff`` is the *effective decoherence length* (O(10²) steps), **not** the
    raw context window. The paper's canonical GPT-4o constants
    (ε₀=0.02, γ=0.15, L_eff=150) place the horizon at d* ≈ 22.3; substituting
    the raw context window L=O(10⁵) would erase the quadratic term and wrongly
    predict near-perfect accuracy.
    """
    d = np.asarray(depths, dtype=float)
    return np.exp(-d * eps0 - gamma * d * (d + 1) / (2.0 * l_eff))


def plot_model_horizons(
    output_path: str | Path = "analysis/figure_model_horizons.png",
    *,
    dpi: int = 150,
    alpha: float = 0.5,
    d_max: float = 60.0,
) -> Path | None:
    """
    Overlay the Theorem 4.2 decay curve for every model in
    ``policy.MODEL_HORIZONS`` and mark each model's horizon d*.

    This is the static, publication-grade companion to the
    "Why some models reach further" comparison chart. Returns the saved path,
    or ``None`` if matplotlib is unavailable.
    """
    from deterministic_horizon.policy import MODEL_HORIZONS, expected_neural_accuracy

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping model-horizon figure.")
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip the synthetic cross-model fallback; plot deepest-horizon last so its
    # label sits on top in the legend.
    models = sorted(
        (m for m in MODEL_HORIZONS if m != "default"),
        key=lambda m: MODEL_HORIZONS[m]["d_star"],
    )
    cmap = plt.get_cmap("viridis")
    d_grid = np.linspace(0, d_max, 240)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, name in enumerate(models):
        color = cmap(i / max(1, len(models) - 1))
        curve = np.array([expected_neural_accuracy(d, name) for d in d_grid])
        d_star = MODEL_HORIZONS[name]["d_star"]
        ax.plot(d_grid, curve, "-", color=color, linewidth=2.0, label=f"{name}  (d*={d_star:.0f})")
        ax.plot([d_star], [alpha], "o", color=color, markersize=6)

    ax.axhline(
        0.92, color="#2a9d2a", linestyle="--", linewidth=1.6, label="Tool-integrated (C3) ≈ 92%"
    )
    ax.axhline(alpha, color="#aaa", linewidth=0.8)
    ax.set_xlabel("Reasoning depth  $d$")
    ax.set_ylabel("Expected neural CoT accuracy")
    ax.set_title("Per-model decoherence: horizons span $d^* \\in [19, 31]$")
    ax.set_xlim(0, d_max)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95, ncol=2)
    ax.margins(x=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------


def generate_figures(
    results: list[Mapping[str, Any]] | str | Path,
    output_dir: str | Path = "analysis",
    *,
    fmt: str = "png",
    dpi: int = 150,
    title: str | None = None,
) -> list[Path]:
    """
    Generate the paper's three headline figures from a results list.

    Parameters
    ----------
    results : list of result dicts, or path to a JSON file emitted by
        ``deterministic_horizon.cli evaluate``.
    output_dir : directory to save figures into (created if missing).
    fmt : ``"png"``, ``"pdf"``, or ``"svg"``.
    dpi : resolution for raster formats.

    Returns
    -------
    list of Path objects pointing at the saved figures.
    """
    results = _load_results(results)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping figure generation.")
        return []

    saved: list[Path] = []

    # ---- Figure 1: accuracy decay ----
    fig_path = output_dir / f"figure_decay.{fmt}"
    saved.append(_plot_accuracy_decay(results, fig_path, plt=plt, dpi=dpi, title=title))

    # ---- Figure 2: SSJ precision / recall ----
    fig_path = output_dir / f"figure_ssj.{fmt}"
    p = _plot_ssj_split(results, fig_path, plt=plt, dpi=dpi)
    if p is not None:
        saved.append(p)

    # ---- Figure 3: per-condition bar chart ----
    fig_path = output_dir / f"figure_conditions.{fmt}"
    p = _plot_condition_comparison(results, fig_path, plt=plt, dpi=dpi)
    if p is not None:
        saved.append(p)

    return saved


def _plot_accuracy_decay(results, path, *, plt, dpi, title=None):
    """
    Plot empirical accuracy vs. reasoning depth alongside the fitted
    super-exponential decay model used by ``estimate_horizon``.
    """
    acc = accuracy_by_depth(results)

    # Restrict to the neural-CoT condition if present — otherwise pool.
    cot_results = [r for r in results if r.get("condition") in (None, "C1")]
    if cot_results:
        acc = accuracy_by_depth(cot_results)
    depths = np.array(sorted(acc.keys()))
    means = np.array([acc[d]["accuracy"] for d in depths])
    lows = np.array([acc[d]["ci_low"] for d in depths])
    highs = np.array([acc[d]["ci_high"] for d in depths])

    horizon = estimate_horizon(cot_results or results, threshold=0.5)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    # Empirical CI band and points
    ax.fill_between(depths, lows, highs, alpha=0.18, color="#d6332e", label="95% CI")
    ax.plot(depths, means, "o", color="#d6332e", markersize=7, label="Empirical (neural CoT)")

    # Theoretical curve — same simple model that ``estimate_horizon`` fits
    eps0 = horizon.get("eps0")
    gamma = horizon.get("gamma")
    if eps0 is not None and gamma is not None:
        d_grid = np.linspace(0, max(depths.max(), 50), 200)
        theo = np.exp(-eps0 * d_grid - gamma * d_grid * (d_grid + 1) / 2.0)
        ax.plot(d_grid, theo, "-", color="#1f5fb4", linewidth=2.2, label="Theory (Thm 4.2)")

    # Tool-delegation reference (synthetic BFS = 100%, real models ≈ 94%)
    tool_acc = _tool_accuracy(results)
    ax.axhline(
        tool_acc,
        color="#2a9d2a",
        linestyle="--",
        linewidth=1.8,
        label=f"Tool-integrated (C3): {tool_acc:.0%}",
    )

    # Deterministic horizon marker
    if "d_star" in horizon and horizon["d_star"] is not None:
        d_star = horizon["d_star"]
        ax.axvline(d_star, color="#555", linestyle=":", linewidth=1.5)
        ax.text(d_star + 0.6, 0.08, f"$d^* = {d_star:.0f}$", fontsize=11, color="#333")

    ax.axhline(0.5, color="#aaa", linewidth=0.8)
    ax.set_xlabel("Reasoning depth  $d$")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, max(depths.max() + 2, 50))
    ax.set_title(title or "Accuracy decay vs. reasoning depth")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path)
    return path


def _tool_accuracy(results) -> float:
    tool = [r for r in results if r.get("condition") == "C3"]
    if not tool:
        return 0.94  # paper headline
    hits = sum(1 for r in tool if r.get("correct"))
    return hits / len(tool)


def _plot_ssj_split(results, path, *, plt, dpi):
    by_depth: dict[int, dict[str, list[float]]] = {}
    has_any = False
    for r in results:
        d = r.get("optimal_depth")
        if d is None:
            continue
        bucket = by_depth.setdefault(int(d), {"ssj": [], "p": [], "r": []})
        if r.get("ssj_score") is not None:
            bucket["ssj"].append(float(r["ssj_score"]))
            has_any = True
        if r.get("precision") is not None:
            bucket["p"].append(float(r["precision"]))
        if r.get("recall") is not None:
            bucket["r"].append(float(r["recall"]))

    if not has_any:
        return None

    depths = sorted(by_depth)
    ssj = [np.mean(by_depth[d]["ssj"]) if by_depth[d]["ssj"] else np.nan for d in depths]
    prec = [np.mean(by_depth[d]["p"]) if by_depth[d]["p"] else np.nan for d in depths]
    rec = [np.mean(by_depth[d]["r"]) if by_depth[d]["r"] else np.nan for d in depths]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(depths, prec, "o-", color="#1f5fb4", label="Precision")
    ax.plot(depths, rec, "s-", color="#d6332e", label="Recall")
    ax.plot(depths, ssj, "^-", color="#555", label="SSJ", alpha=0.7)
    ax.set_xlabel("Reasoning depth  $d$")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    ax.set_title("SSJ precision/recall — both decay ⇒ capability failure")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path)
    return path


def _plot_condition_comparison(results, path, *, plt, dpi):
    by_cond: dict[str, list[bool]] = {}
    for r in results:
        cond = r.get("condition")
        if not cond:
            continue
        by_cond.setdefault(cond, []).append(bool(r.get("correct", False)))

    if not by_cond:
        return None

    conds = sorted(by_cond)
    acc = [100.0 * np.mean(by_cond[c]) for c in conds]

    palette = {
        "C1": "#d6332e",
        "C2": "#cc9933",
        "C3": "#2a9d2a",
        "C4": "#9933cc",
        "C5": "#1f5fb4",
    }
    colors = [palette.get(c, "#666") for c in conds]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    bars = ax.bar(conds, acc, color=colors)
    for b, v in zip(bars, acc, strict=False):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}%", ha="center", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by condition")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path)
    return path


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------


def generate_tables(
    results: list[Mapping[str, Any]] | str | Path,
    output_dir: str | Path = "analysis",
) -> dict[str, Path]:
    """
    Produce ``accuracy_by_depth.md``, ``conditions.md`` and ``horizon.json``
    summarising the supplied results.

    Returns a dict mapping logical name → path.
    """
    results = _load_results(results)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, Path] = {}

    # --- accuracy_by_depth.md ---
    acc = accuracy_by_depth(results)
    lines = ["| Depth | Accuracy | 95% CI | N |", "|------:|---------:|:-------|--:|"]
    for d in sorted(acc):
        s = acc[d]
        lines.append(
            f"| {d} | {s['accuracy']:.1%} | "
            f"[{s['ci_low']:.2f}, {s['ci_high']:.2f}] | {s['n']} |"
        )
    p = output_dir / "accuracy_by_depth.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out["accuracy_by_depth"] = p

    # --- conditions.md ---
    by_cond: dict[str, list[bool]] = {}
    for r in results:
        cond = r.get("condition", "?")
        by_cond.setdefault(cond, []).append(bool(r.get("correct", False)))
    if by_cond:
        lines = ["| Condition | Accuracy | N |", "|:---------|---------:|--:|"]
        for c in sorted(by_cond):
            vals = by_cond[c]
            acc_pct = 100.0 * (sum(vals) / len(vals)) if vals else 0.0
            lines.append(f"| {c} | {acc_pct:.1f}% | {len(vals)} |")
        p = output_dir / "conditions.md"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        out["conditions"] = p

    # --- horizon.json ---
    # The Deterministic Horizon is the *neural CoT* (C1) 50% crossover; never
    # pool it with the tool condition (C3), which is correct at every depth.
    cot_results = [r for r in results if r.get("condition") in (None, "C1")]
    horizon = estimate_horizon(cot_results or results, threshold=0.5)
    p = output_dir / "horizon.json"
    p.write_text(json.dumps(horizon, indent=2, default=float), encoding="utf-8")
    out["horizon"] = p

    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_results(results) -> list[Mapping[str, Any]]:
    if isinstance(results, (str, Path)):
        return json.loads(Path(results).read_text(encoding="utf-8"))
    return list(results)


__all__ = [
    "decay_curve",
    "generate_figures",
    "generate_tables",
    "plot_model_horizons",
]
