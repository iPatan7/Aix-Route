"""Tests for the analysis module — figure & table generation."""

from __future__ import annotations

import json

import numpy as np
import pytest
from deterministic_horizon.analysis import (
    decay_curve,
    generate_figures,
    generate_tables,
)


def _synthetic_results(seed: int = 0):
    rng = np.random.default_rng(seed)
    depths = list(range(2, 42, 2))
    out = []
    for d in depths:
        eps0, gamma = 0.02, 0.0025
        p = float(np.exp(-eps0 * d - gamma * d * (d + 1) / 2))
        for _ in range(20):
            out.append(
                {
                    "instance_id": f"i{d}",
                    "condition": "C1",
                    "model": "synthetic",
                    "optimal_depth": int(d),
                    "correct": bool(rng.random() < p),
                    "ssj_score": p,
                    "precision": p,
                    "recall": p,
                }
            )
            out.append(
                {
                    "instance_id": f"i{d}",
                    "condition": "C3",
                    "model": "bfs",
                    "optimal_depth": int(d),
                    "correct": True,
                }
            )
    return out


def test_decay_curve_monotone_decreasing():
    d = np.arange(0, 60, dtype=float)
    y = decay_curve(d, eps0=0.02, gamma=0.15, l_eff=150.0)
    assert y[0] == pytest.approx(1.0)
    assert all(y[i] >= y[i + 1] for i in range(len(y) - 1))


def test_generate_tables_writes_files(tmp_path):
    results = _synthetic_results()
    generate_tables(results, output_dir=tmp_path)
    assert (tmp_path / "accuracy_by_depth.md").exists()
    assert (tmp_path / "conditions.md").exists()
    horizon = json.loads((tmp_path / "horizon.json").read_text())
    assert "d_star" in horizon


def test_generate_figures_produces_pngs(tmp_path):
    pytest.importorskip("matplotlib")
    results = _synthetic_results()
    files = generate_figures(results, output_dir=tmp_path, fmt="png")
    assert len(files) >= 1
    for f in files:
        assert f.exists() and f.stat().st_size > 0
