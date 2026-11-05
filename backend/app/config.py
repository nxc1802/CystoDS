"""Application configuration settings for CystoDS Backend."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

# Workspace Root Detection
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

DEFAULT_CHECKPOINT_REL = "result/30_proposed/three_stage_split0_research_20260819-211511/phase3/split_0/best_model.pt"
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / DEFAULT_CHECKPOINT_REL
DEFAULT_CANONICAL_HASH = "6c10366505db80367e53c400bfddaacada0a0df004b07a87bdfbf6dc0816846c"
DEFAULT_SAMPLES_DIR = PROJECT_ROOT / "xvdhy-osfstorage-archive" / "images"


class Settings:
    """Immutable application settings with environment variable fallbacks."""

    def __init__(self) -> None:
        self.project_root: Path = PROJECT_ROOT
        self.model_path: Path = Path(
            os.getenv("MODEL_PATH", str(DEFAULT_CHECKPOINT_PATH))
        ).resolve()
        self.model_hash: str = os.getenv("MODEL_HASH", DEFAULT_CANONICAL_HASH)
        self.device: str = os.getenv("DEVICE", "auto")
        self.port: int = int(os.getenv("PORT", "8000"))
        self.host: str = os.getenv("HOST", "0.0.0.0")

        origins_raw = os.getenv("ALLOWED_ORIGINS", "*")
        self.allowed_origins: List[str] = (
            ["*"] if origins_raw == "*" else [o.strip() for o in origins_raw.split(",") if o.strip()]
        )

        self.app_env: str = os.getenv("APP_ENV", "production")
        self.max_upload_bytes: int = int(
            os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))
        )
        self.max_concurrent_inference: int = int(
            os.getenv("MAX_CONCURRENT_INFERENCE", "2")
        )
        self.request_timeout_seconds: int = int(
            os.getenv("REQUEST_TIMEOUT_SECONDS", "30")
        )
        self.demo_samples_dir: Path = Path(
            os.getenv("DEMO_SAMPLES_DIR", str(DEFAULT_SAMPLES_DIR))
        ).resolve()

        self.api_v1_prefix: str = "/api/v1"
        self.version: str = "1.0.0"
        self.service_name: str = "CystoDS Clinical AI Inference Server"
        self.app_name: str = "CystoDS"


settings = Settings()
