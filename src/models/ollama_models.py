"""
Ollama adapter — local models, zero cost.

Ollama serves an OpenAI-compatible Chat Completions endpoint at
``http://localhost:11434/v1`` once ``ollama serve`` is running, so we reuse
:class:`OpenAICompatibleModel`. Two differences from the cloud providers:

* **No API key.** Ollama is unauthenticated; we send a placeholder so the
  ``openai`` client is happy and skip the env-var lookup.
* **Local health check.** On setup we ping ``/api/tags`` and raise a clear,
  actionable error if the server isn't running — far friendlier than an opaque
  connection error deep inside the first request.

Set ``OLLAMA_BASE_URL`` to point at a non-default host/port (see
``.env.example``). Requires the ``openai`` client; ``requests`` (already a core
dependency) is used only for the health check.

Examples
--------
>>> from deterministic_horizon.models import load_model
>>> m = load_model("llama3.1:8b")                    # doctest: +SKIP
>>> m.generate("2+2?").content                        # doctest: +SKIP
"""

from __future__ import annotations

import os
from typing import ClassVar

from deterministic_horizon.models.openai_compatible import OpenAICompatibleModel

DEFAULT_OLLAMA_URL = "http://localhost:11434"


class OllamaModel(OpenAICompatibleModel):
    """Local models served by Ollama (OpenAI-compatible, zero cost)."""

    # Resolved per-instance from OLLAMA_BASE_URL; this is just the fallback.
    API_BASE: ClassVar[str] = f"{DEFAULT_OLLAMA_URL}/v1"
    ENV_KEYS: ClassVar[tuple[str, ...]] = ()
    # Local inference is free.
    DEFAULT_PRICING: ClassVar[dict[str, float]] = {"input": 0.0, "output": 0.0}

    # Friendly identifiers → Ollama model tags (identity; Ollama tags carry a
    # ``:`` which ``load_model`` preserves). Only the tags actually pulled
    # locally are listed — add more here as you ``ollama pull`` them.
    ALIASES: ClassVar[dict[str, str]] = {
        "qwen2.5:1.5b": "qwen2.5:1.5b",
        "qwen2.5:7b": "qwen2.5:7b",
    }

    def _base_url(self) -> str:
        """Server root (no ``/v1``), honouring ``OLLAMA_BASE_URL``."""
        return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL).rstrip("/")

    def _api_key(self) -> str:
        # Ollama is unauthenticated; the openai client still wants a non-empty
        # string. "ollama" is the conventional placeholder.
        return "ollama"

    def _setup_client(self) -> None:
        root = self._base_url()
        self._check_server(root)
        try:
            from openai import OpenAI
        except ImportError as err:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            ) from err

        self._client = OpenAI(
            api_key=self._api_key(),
            base_url=f"{root}/v1",
            timeout=self.timeout,
        )

    @staticmethod
    def _check_server(root: str) -> None:
        """Fail fast with a helpful message if Ollama isn't reachable."""
        try:
            import requests
        except ImportError as err:  # pragma: no cover - requests is a core dep
            raise ImportError("requests package required for the Ollama health check") from err

        try:
            resp = requests.get(f"{root}/api/tags", timeout=5)
        except requests.ConnectionError as err:
            raise RuntimeError(
                f"Ollama not running at {root}. "
                "Install: https://ollama.com/download | Run: ollama serve"
            ) from err
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama not responding at {root} (status {resp.status_code})")
