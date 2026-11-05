"""API routes package for CystoDS."""

from backend.app.routes.health import router as health_router
from backend.app.routes.model_info import router as model_info_router
from backend.app.routes.samples import router as samples_router
from backend.app.routes.predict import router as predict_router

__all__ = [
    "health_router",
    "model_info_router",
    "samples_router",
    "predict_router",
]
