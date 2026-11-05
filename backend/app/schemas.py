"""Pydantic schemas and data contracts for CystoDS REST API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


# ── Health Check ────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str = Field(..., description="System operational status (healthy/degraded)")
    model_loaded: bool = Field(..., description="Whether deep learning model is loaded in memory")
    device: str = Field(..., description="Inference compute device (e.g., mps, cuda, cpu)")
    version: str = Field(..., description="Backend service semantic version")
    checkpoint_hash: str = Field(..., description="SHA-256 hash of loaded canonical checkpoint")
    service: str = Field(default="CystoDS Clinical AI Inference Server", description="Service title")
    checkpoint_loaded: bool = Field(default=True, description="Alias for model_loaded")
    model_parameters: int = Field(default=28230679, description="Total parameter count")
    memory_allocated_mb: float = Field(default=0.0, description="Allocated VRAM/RAM in megabytes")
    uptime_seconds: float = Field(default=0.0, description="Server uptime in seconds")


# ── Model Information ───────────────────────────────────────────────────
class ModelInfoResponse(BaseModel):
    name: str = Field(..., description="Model descriptive name")
    version: str = Field(..., description="Model version")
    task_type: str = Field(..., description="Primary learning task mode")
    architecture: str = Field(..., description="High-level neural network architecture")
    parameters: int = Field(..., description="Number of parameters in network")
    taxonomy: Dict[str, Any] = Field(..., description="Class hierarchy definition")
    input_spec: Dict[str, Any] = Field(..., description="Input image specifications and transform rules")
    metrics_summary: Dict[str, Any] = Field(..., description="Verified benchmark scores from research")

    # Extended specification fields for frontend compatibility
    model_name: Optional[str] = None
    backbone: Optional[str] = None
    input_resolution: Optional[List[int]] = None
    fov_center_crop_ratio: Optional[float] = None
    training_protocol: Optional[str] = None
    study_id: Optional[str] = None


# ── Sample Gallery ──────────────────────────────────────────────────────
class GroundTruth(BaseModel):
    binary: str = Field(..., description="Ground truth binary ROI class")
    coarse: str = Field(..., description="Ground truth coarse anatomical/pathological group")
    fine: Optional[str] = Field(None, description="Ground truth fine sub-class")


class SampleItem(BaseModel):
    id: str = Field(..., description="Unique sample identifier (e.g. case_cis_01)")
    name: str = Field(..., description="Human-readable title/case name")
    filename: str = Field(..., description="Image filename on storage")
    thumbnail_url: str = Field(..., description="URL to retrieve sample image")
    description: str = Field(..., description="Clinical context and case findings")
    modality: str = Field(..., description="Endoscopy imaging modality (WLC / BLC)")
    ground_truth: GroundTruth = Field(..., description="Annotated gold standard diagnosis")
    patient_id: Optional[str] = Field(None, description="Anonymized patient study identifier")
    stage: Optional[str] = Field(None, description="Pathological stage (e.g. Tis, Ta, T1, T2)")
    urgency: str = Field(..., description="Clinical urgency tier (CRITICAL, HIGH, MEDIUM, NORMAL)")
    has_segmentation: bool = Field(default=False, description="Whether lesion segmentation mask exists")


# ── Prediction Request & Results ────────────────────────────────────────
class PredictJSONRequest(BaseModel):
    sample_id: Optional[str] = Field(None, description="Sample ID to run prediction against")
    blend_hierarchical: bool = Field(default=False, description="Enable ensemble hierarchical marginalization")


class BinaryResult(BaseModel):
    prediction: str = Field(..., description="Predicted binary label ('ROI' or 'Non-ROI')")
    confidence: float = Field(..., description="Confidence probability for predicted class")
    probabilities: Dict[str, float] = Field(..., description="Softmax probabilities for both classes")


class CoarseResult(BaseModel):
    prediction: str = Field(..., description="Predicted coarse group label")
    confidence: float = Field(..., description="Confidence probability for predicted class")
    probabilities: Dict[str, float] = Field(..., description="Softmax probabilities across all 5 coarse classes")


class FineTopItem(BaseModel):
    class_name: str = Field(..., alias="class", description="Subclass label")
    probability: float = Field(..., description="Normalized confidence probability")
    parent: str = Field(..., description="Parent coarse anatomical group")

    model_config = ConfigDict(populate_by_name=True)


class FineResult(BaseModel):
    prediction: str = Field(..., description="Predicted fine class label")
    confidence: float = Field(..., description="Confidence probability for predicted class")
    top_3: List[Dict[str, Any]] = Field(..., description="Top 3 ranked fine classes with parents")
    top_5: Optional[List[Dict[str, Any]]] = Field(None, description="Top 5 ranked fine classes")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Full probability distribution over 22 classes")


class ConsistencyResult(BaseModel):
    is_consistent: bool = Field(..., description="Whether coarse and fine predictions align in hierarchy")
    parent_from_fine: str = Field(..., description="Derived parent class from predicted fine subclass")
    direct_coarse_prediction: str = Field(..., description="Direct coarse head prediction")


class ClinicalAssessment(BaseModel):
    urgency: str = Field(..., description="Clinical triage level: CRITICAL, HIGH, MEDIUM, NORMAL")
    badge_color: str = Field(..., description="UI badge display color token: rose, amber, yellow, blue, emerald")
    title: str = Field(..., description="Clinical headline warning or status")
    action: str = Field(..., description="Recommended medical workflow or procedure")
    recommendation: Optional[str] = Field(None, description="Extended guidance notes")


class PredictResponse(BaseModel):
    request_id: str = Field(..., description="Unique UUID for tracing inference request")
    model_version: str = Field(..., description="Version of inference model used")
    task_type: str = Field(..., description="Inference task mode: hierarchical")
    binary: BinaryResult = Field(..., description="Level 1 Binary prediction details")
    coarse: CoarseResult = Field(..., description="Level 2 Coarse group details")
    fine: FineResult = Field(..., description="Level 3 Fine subclass details")
    urgency: str = Field(..., description="Clinical urgency rating")
    consistency: ConsistencyResult = Field(..., description="Cross-level consistency assessment")
    inference_time_ms: float = Field(..., description="Model forward computation latency in milliseconds")
    status: str = Field(default="success", description="Execution status")
    success: bool = Field(default=True, description="Boolean success indicator")
    provenance: Optional[Dict[str, Any]] = Field(None, description="Execution environment and checkpoint details")
    hierarchy_metrics: Optional[ConsistencyResult] = Field(None, description="Alias for consistency")
    clinical_assessment: Optional[ClinicalAssessment] = Field(None, description="Comprehensive clinical assessment")
