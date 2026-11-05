"""Model information and metadata endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from backend.app.schemas import ModelInfoResponse

router = APIRouter(tags=["Model Info"])


@router.get("/model-info", response_model=ModelInfoResponse)
@router.get("/model/info", response_model=ModelInfoResponse, include_in_schema=False)
async def get_model_info(request: Request) -> ModelInfoResponse:
    """Returns canonical model architecture, input specs, and research metrics."""
    adapter = getattr(request.app.state, "adapter", None)
    if adapter is None or not getattr(adapter, "model_loaded", False):
        raise HTTPException(status_code=503, detail="Inference model is not currently loaded.")

    meta = adapter.metadata()
    return ModelInfoResponse(**meta)
