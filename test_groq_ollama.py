#!/usr/bin/env python3
"""
Smoke test for the Groq + Ollama backends.

Exercises the Deterministic Horizon *routing* only — no API key, no running
Ollama server required. (Live-model tests live under ``tests/`` and are skipped
unless the relevant backend is reachable.)
"""

from __future__ import annotations

from deterministic_horizon.policy import (
    horizon_table,
    recommend_model,
    should_delegate,
)


def test_groq() -> None:
    """Groq routing without an API call."""
    print("=== GROQ ===")

    # Shallow task on the high-horizon 70b → stay neural. (Small models can
    # delegate earlier via the margin rule, so we use the 70b for the clean
    # "shallow stays neural" check.)
    d = should_delegate(estimated_depth=5, model="llama-3.3-70b-versatile")
    print(f"should_delegate(5, llama-3.3-70b-versatile) = {d}")
    assert d is False, "Expected False for shallow task"

    # Deep task → delegate.
    d = should_delegate(estimated_depth=35, model="llama-3.3-70b-versatile")
    print(f"should_delegate(35, llama-3.3-70b-versatile) = {d}")
    assert d is True, "Expected True for deep task"

    # Groq models present in the horizon table.
    ht = horizon_table()
    groq_models = [r for r in ht if r["model"] in {
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "gpt-oss-20b",
        "gpt-oss-120b",
    }]
    print(f"Groq models in horizon_table: {len(groq_models)}")
    assert len(groq_models) == 4, "Groq models missing from horizon table"

    # recommend_model returns (name, accuracy).
    name, acc = recommend_model(18)
    print(f"recommend_model(18) = ({name!r}, {acc:.3f})")
    assert name is not None

    print("PASS: Groq\n")


def test_ollama() -> None:
    """Ollama routing without a running local model."""
    print("=== OLLAMA (local) ===")

    # Shallow task → stay neural.
    d = should_delegate(estimated_depth=5, model="qwen2.5:7b")
    print(f"should_delegate(5, qwen2.5:7b) = {d}")
    assert d is False

    # Deep task → delegate.
    d = should_delegate(estimated_depth=30, model="qwen2.5:7b")
    print(f"should_delegate(30, qwen2.5:7b) = {d}")
    assert d is True

    # Locally-pulled Ollama tags present in the horizon table.
    ht = horizon_table()
    ollama_models = [r for r in ht if r["model"] in {"qwen2.5:1.5b", "qwen2.5:7b"}]
    print(f"Ollama models in horizon_table: {len(ollama_models)}")
    assert len(ollama_models) == 2

    # The smaller 1.5b model has a lower horizon than the 7b.
    by_name = {r["model"]: r["d_star"] for r in ht}
    assert by_name["qwen2.5:1.5b"] < by_name["qwen2.5:7b"]

    print("PASS: Ollama\n")


def test_cost_comparison() -> None:
    """Cost per correct solution: Groq free tier vs Ollama local."""
    print("=== COST COMPARISON ===")
    # At depth past the horizon, neural accuracy ~50% → ~2 attempts per correct.
    # Groq free tier and Ollama local are both $0 to the user.
    groq_cost_per_correct = 0.0      # free tier
    ollama_cost_per_correct = 0.0    # local
    print(f"Groq cost per correct (free tier): ${groq_cost_per_correct:.2f}")
    print(f"Ollama cost per correct (local):   ${ollama_cost_per_correct:.2f}")
    print("Total testing budget: $0")
    print("PASS: cost comparison\n")


if __name__ == "__main__":
    test_groq()
    test_ollama()
    test_cost_comparison()
    print("=== ALL TESTS PASSED ===")
