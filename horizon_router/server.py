from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from deterministic_horizon.policy import horizon_table
from horizon_router.cache import get_cached_decision

app = FastAPI(title="Deterministic Horizon Router")

class DelegateRequest(BaseModel):
    estimated_depth: float
    model: str = "default"
    threshold: float = 0.5
    tool_available: bool = True
    tool_accuracy: float = 0.92
    margin: float = 0.10

@app.post("/delegate")
def delegate(req: DelegateRequest) -> dict[str, object]:
    decision = get_cached_decision(
        estimated_depth=req.estimated_depth,
        model=req.model,
        threshold=req.threshold,
        tool_available=req.tool_available,
        tool_accuracy=req.tool_accuracy,
        margin=req.margin
    )
    return decision.to_dict()

@app.get("/horizons")
def horizons():
    return horizon_table()

@app.get("/health")
def health():
    return {"status": "ok"}
