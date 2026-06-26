"""Snapshot test for MODEL_HORIZONS values from the paper."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from deterministic_horizon.policy import MODEL_HORIZONS

SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "model_horizons.json"


def _load_snapshot(path: Path) -> dict[str, dict[str, float]]:
    raw = path.read_text(encoding="utf-8").splitlines()
    if raw and raw[0].startswith("//"):
        raw = raw[1:]
    return json.loads("\n".join(raw))


def test_model_horizons_matches_snapshot():
    loaded = _load_snapshot(SNAPSHOT_PATH)
    expected = {
        model: {name: pytest.approx(value) for name, value in params.items()}
        for model, params in loaded.items()
    }
    assert expected == MODEL_HORIZONS, "Updated paper values? Refresh the snapshot."
