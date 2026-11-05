"""Prediction endpoint for CystoDS AI Inference Server."""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image

logger = logging.getLogger("cystods.predict")

from backend.app.config import settings
from backend.app.routes.samples import find_sample_dict, get_sample_image_path
from backend.app.schemas import (
    BinaryResult,
    ClinicalAssessment,
    CoarseResult,
    ConsistencyResult,
    FineResult,
    PredictResponse,
)

router = APIRouter(tags=["Inference"])


@router.post("/predict", response_model=PredictResponse)
async def predict_cystoscopy(
    request: Request,
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    blend_hierarchical: bool = Form(False),
) -> PredictResponse:
    """Runs genuine hierarchical multi-level inference for uploaded image or sample case."""
    adapter = getattr(request.app.state, "adapter", None)
    if adapter is None or not getattr(adapter, "model_loaded", False):
        raise HTTPException(status_code=503, detail="Inference model is not currently loaded.")

    content_type = request.headers.get("content-type", "")
    target_sample_id: Optional[str] = sample_id
    apply_blend: bool = blend_hierarchical

    # 1. Parse JSON body if Content-Type is application/json
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                target_sample_id = body.get("sample_id") or target_sample_id
                apply_blend = bool(body.get("blend_hierarchical", apply_blend))
        except Exception:
            pass

    # 2. Check query parameter fallback
    if not file and not target_sample_id:
        target_sample_id = request.query_params.get("sample_id")

    # 3. Resolve Input Image
    image: Optional[Image.Image] = None

    if file is not None and file.filename:
        # File upload pathway - check for path traversal patterns in filename
        raw_filename = file.filename
        if ".." in raw_filename or "/" in raw_filename or "\\" in raw_filename:
            raise HTTPException(
                status_code=400,
                detail="Filename contains invalid path traversal characters.",
            )
        clean_filename = Path(raw_filename).name

        raw_bytes = await file.read()
        if len(raw_bytes) == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty (0 bytes).",
            )
        if len(raw_bytes) > settings.max_upload_bytes:
            mb_limit = settings.max_upload_bytes // (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded file size ({len(raw_bytes)} bytes) exceeds limit of {settings.max_upload_bytes} bytes ({mb_limit} MB).",
            )
        try:
            image = adapter.validate_input(raw_bytes, filename=clean_filename)
        except ValueError as val_err:
            raise HTTPException(status_code=400, detail=str(val_err)) from val_err

    elif target_sample_id:
        # Curated sample reference pathway - check for path traversal attempts
        if ".." in target_sample_id or "/" in target_sample_id or "\\" in target_sample_id:
            raise HTTPException(
                status_code=400,
                detail="Invalid sample_id: path traversal sequence detected.",
            )
        sample_dict = find_sample_dict(target_sample_id)
        if not sample_dict:
            raise HTTPException(
                status_code=404,
                detail=f"Sample case '{target_sample_id}' not found in curated registry.",
            )
        try:
            sample_path = get_sample_image_path(sample_dict["filename"])
            with open(sample_path, "rb") as f:
                sample_bytes = f.read()
            image = adapter.validate_input(sample_bytes, filename=sample_dict["filename"])
        except (FileNotFoundError, ValueError) as err:
            raise HTTPException(status_code=400, detail=f"Failed to load sample image: {err}") from err

    else:
        raise HTTPException(
            status_code=400,
            detail="Either 'file' upload or 'sample_id' must be provided.",
        )

    # 4. Perform Genuine Model Inference
    request_id = f"inf_{uuid.uuid4().hex[:12]}"
    try:
        results = adapter.predict_image(image, blend_hierarchical=apply_blend)
    except Exception as exc:
        logger.error("Inference execution failed on request %s: %s", request_id, exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Inference execution failed due to an internal server error.",
        ) from exc

    # 5. Format Structured Response
    binary_data = BinaryResult(**results["binary"])
    coarse_data = CoarseResult(**results["coarse"])
    fine_data = FineResult(**results["fine"])
    consistency_data = ConsistencyResult(**results["consistency"])
    clinical_data = ClinicalAssessment(**results["clinical_assessment"])

    provenance = {
        "mode": "live_model_execution",
        "checkpoint_hash": adapter.checkpoint_hash,
        "device": str(adapter.device),
        "latency_ms": results["inference_time_ms"],
        "fps": round(1000.0 / max(results["inference_time_ms"], 0.001), 1),
    }

    return PredictResponse(
        request_id=request_id,
        model_version="3.1",
        task_type="hierarchical",
        binary=binary_data,
        coarse=coarse_data,
        fine=fine_data,
        urgency=results["urgency"],
        consistency=consistency_data,
        inference_time_ms=results["inference_time_ms"],
        status="success",
        success=True,
        provenance=provenance,
        hierarchy_metrics=consistency_data,
        clinical_assessment=clinical_data,
    )
