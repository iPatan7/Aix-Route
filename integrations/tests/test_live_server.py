"""
Integration test against Agent A's live router server.

Skipped unless ``HORIZON_ROUTER_URL`` is set (default checkpoint:
``http://localhost:8000``) AND the server answers ``/health``. This is the
end-of-hour merge check that the integrations route consistently with the HTTP
service.
"""

from __future__ import annotations

import os
import urllib.request

import pytest

_URL = os.environ.get("HORIZON_ROUTER_URL", "http://localhost:8000")


def _server_up() -> bool:
    try:
        with urllib.request.urlopen(f"{_URL}/health", timeout=1) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001 - server simply not running
        return False


pytestmark = pytest.mark.skipif(
    not _server_up(), reason=f"router server not reachable at {_URL}"
)


def test_local_and_server_agree_on_routing() -> None:
    import json

    from integrations import should_delegate

    payload = json.dumps(
        {"estimated_depth": 35, "model": "gpt-4o"}
    ).encode()
    req = urllib.request.Request(
        f"{_URL}/delegate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=2) as resp:
        server_decision = json.loads(resp.read())

    local = should_delegate(estimated_depth=35, model="gpt-4o")
    assert server_decision["delegate"] == local is True
