from __future__ import annotations

import requests
from deterministic_horizon.policy import recommend_model


class HorizonRouter:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
    
    def should_delegate(
        self,
        estimated_depth: int | float,
        model: str = "default",
        threshold: float = 0.5,
        tool_available: bool = True,
        tool_accuracy: float = 0.92,
        margin: float = 0.10,
    ) -> bool:
        resp = requests.post(
            f"{self.base_url}/delegate",
            json={
                "estimated_depth": float(estimated_depth),
                "model": model,
                "threshold": threshold,
                "tool_available": tool_available,
                "tool_accuracy": tool_accuracy,
                "margin": margin,
            }
        )
        resp.raise_for_status()
        return resp.json()["delegate"]
    
    def recommend_model(self, estimated_depth: int | float, threshold: float = 0.5) -> str | None:
        """Cheapest model clearing the threshold."""
        model, _ = recommend_model(estimated_depth=estimated_depth, threshold=threshold)
        return model
