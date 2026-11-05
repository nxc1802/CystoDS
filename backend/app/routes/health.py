"""Health check endpoint for CystoDS API."""

from __future__ import annotations

import time
from fastapi import APIRouter, Request
from backend.app.schemas import HealthResponse
from backend.app.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def get_health(request: Request) -> HealthResponse:
    """Returns service health, loaded checkpoint integrity hash, and device information."""
    adapter = getattr(request.app.state, "adapter", None)
    start_time = getattr(request.app.state, "start_time", time.time())
    uptime = time.time() - start_time

    if adapter is not None and getattr(adapter, "model_loaded", False):
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            device=str(adapter.device),
            version=settings.version,
            checkpoint_hash=adapter.checkpoint_hash,
            service=settings.service_name,
            checkpoint_loaded=True,
            model_parameters=adapter.total_parameters or 28230679,
            memory_allocated_mb=adapter.get_memory_allocated_mb(),
            uptime_seconds=round(uptime, 2),
        )

    return HealthResponse(
        status="degraded",
        model_loaded=False,
        device="none",
        version=settings.version,
        checkpoint_hash="",
        service=settings.service_name,
        checkpoint_loaded=False,
        model_parameters=0,
        memory_allocated_mb=0.0,
        uptime_seconds=round(uptime, 2),
    )
