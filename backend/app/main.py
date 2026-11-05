"""FastAPI Application entrypoint for CystoDS Backend."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.adapters.model_adapter import CystoDSModelAdapter
from backend.app.config import settings
from backend.app.routes import (
    health_router,
    model_info_router,
    predict_router,
    samples_router,
)

logger = logging.getLogger("cystods.backend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages application startup and shutdown lifecycle.

    Loads and caches the canonical deep learning checkpoint once during startup.
    """
    app.state.start_time = time.time()
    logger.info("Starting CystoDS Backend Service v%s...", settings.version)
    logger.info("Initializing ModelAdapter with device preference '%s'...", settings.device)

    adapter = CystoDSModelAdapter(
        device_preference=settings.device,
        max_upload_bytes=settings.max_upload_bytes,
    )

    try:
        logger.info("Loading checkpoint from: %s", settings.model_path)
        adapter.load(settings.model_path)
        logger.info(
            "Model successfully loaded on '%s' (hash: %s, params: %d)",
            adapter.device,
            adapter.checkpoint_hash[:16] + "...",
            adapter.total_parameters,
        )
    except Exception as exc:
        logger.error("Failed to load checkpoint at startup: %s", exc, exc_info=True)
        # Continue in degraded mode so health checks and diagnostic endpoints work
        pass

    app.state.adapter = adapter
    yield
    logger.info("Shutting down CystoDS Backend Service.")


def create_app() -> FastAPI:
    """Factory creating configured FastAPI instance."""
    app = FastAPI(
        title="CystoDS Clinical AI Inference Server",
        description=(
            "Production-grade hierarchical deep learning inference server for cystoscopic endoscopy "
            "classification (Hierarchical Long-Tailed Architecture)."
        ),
        version=settings.version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global Exception Handlers
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc), "status": "error", "type": "ValueError"},
        )

    # Mount API V1 Routers
    api_prefix = settings.api_v1_prefix
    app.include_router(health_router, prefix=api_prefix)
    app.include_router(model_info_router, prefix=api_prefix)
    app.include_router(samples_router, prefix=api_prefix)
    app.include_router(predict_router, prefix=api_prefix)

    @app.get("/", tags=["Root"])
    async def root_index() -> dict[str, str]:
        return {
            "service": settings.service_name,
            "version": settings.version,
            "status": "online",
            "docs": "/docs",
            "health": f"{api_prefix}/health",
            "model_info": f"{api_prefix}/model-info",
            "samples": f"{api_prefix}/samples",
            "predict": f"{api_prefix}/predict",
        }

    return app


app = create_app()
