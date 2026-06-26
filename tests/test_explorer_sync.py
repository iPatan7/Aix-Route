"""
Guard: the interactive explorer (docs/index.html) embeds the per-model
(ε₀, d*) constants in JavaScript. They must stay in exact agreement with
``deterministic_horizon.policy`` so the live tool never drifts from the paper.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from deterministic_horizon.policy import MODEL_HORIZONS

EXPLORER = Path(__file__).resolve().parents[1] / "docs" / "index.html"

# Matches lines like:   "gpt-4o":          [0.020, 22.0],
_ROW = re.compile(r'"([a-z0-9.\-]+)":\s*\[([0-9.]+),\s*([0-9.]+)\]')
_SCRIPT = re.compile(r'<script>\n"use strict";(.*?)</script>', re.S)


def _parse_explorer_models() -> dict[str, tuple[float, float]]:
    text = EXPLORER.read_text(encoding="utf-8")
    block = text.split("const MODELS", 1)[1].split("};", 1)[0]
    return {m[1]: (float(m[2]), float(m[3])) for m in _ROW.finditer(block)}


def test_explorer_file_exists():
    assert EXPLORER.exists(), "docs/index.html (interactive explorer) is missing"


def test_explorer_models_match_policy():
    explorer = _parse_explorer_models()
    assert explorer, "no model presets parsed from the explorer"
    # Every preset in the explorer must exist in policy with identical constants.
    for name, (eps0, dstar) in explorer.items():
        assert name in MODEL_HORIZONS, f"explorer model {name!r} not in policy.MODEL_HORIZONS"
        assert eps0 == pytest.approx(MODEL_HORIZONS[name]["eps0"]), name
        assert dstar == pytest.approx(MODEL_HORIZONS[name]["d_star"]), name


def test_explorer_covers_every_policy_model():
    explorer = _parse_explorer_models()
    missing = set(MODEL_HORIZONS) - set(explorer)
    assert not missing, f"explorer is missing presets for: {sorted(missing)}"


def test_explorer_javascript_parses():
    """The explorer's <script> block must be syntactically valid JS.

    Guards the single largest file in the repo against a stray brace breaking
    the live page. Skipped when Node.js is unavailable (e.g. minimal CI images).
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available; skipping JS syntax check")
    match = _SCRIPT.search(EXPLORER.read_text(encoding="utf-8"))
    assert match, "could not locate the explorer <script> block"
    tmp = Path(tempfile.gettempdir()) / "dh_explorer_check.js"
    tmp.write_text('"use strict";\n' + match.group(1), encoding="utf-8")
    try:
        result = subprocess.run(
            [node, "--check", str(tmp)], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
    finally:
        tmp.unlink(missing_ok=True)
