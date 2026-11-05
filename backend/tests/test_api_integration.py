"""API integration tests for all CystoDS endpoints using FastAPI TestClient."""

from __future__ import annotations

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.config import settings
from backend.app.main import app


@pytest.fixture(scope="module")
def client():
    """Initializes TestClient triggering application startup lifespan once."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client: TestClient) -> None:
    """Verifies root landing endpoint documentation links."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "online"
    assert "/docs" in data["docs"]
    assert "/health" in data["health"]


def test_health_endpoint(client: TestClient) -> None:
    """Verifies health check response format and loaded model status."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["checkpoint_loaded"] is True
    assert len(data["checkpoint_hash"]) == 64
    assert data["checkpoint_hash"] == settings.model_hash
    assert data["model_parameters"] == 28230679
    assert data["device"] in ("mps", "cuda", "cpu")
    assert data["uptime_seconds"] >= 0.0


def test_model_info_endpoints(client: TestClient) -> None:
    """Verifies model-info and aliased model/info endpoints."""
    for path in ["/api/v1/model-info", "/api/v1/model/info"]:
        resp = client.get(path)
        assert resp.status_code == 200
        data = resp.json()
        assert "3S-HFT" in data["name"]
        assert data["task_type"] == "hierarchical"
        assert data["architecture"] == "TwoStageDecoupledHierarchicalModel"
        assert data["parameters"] == 28230679
        assert len(data["taxonomy"]["binary"]) == 2
        assert len(data["taxonomy"]["coarse"]) == 5
        assert data["taxonomy"]["fine_count"] == 22
        assert data["metrics_summary"]["binary_auroc"] == 0.9986


def test_samples_catalog_endpoint(client: TestClient) -> None:
    """Verifies curated clinical samples catalog endpoint."""
    resp = client.get("/api/v1/samples")
    assert resp.status_code == 200
    samples = resp.json()
    assert isinstance(samples, list)
    assert len(samples) == 6

    sample_ids = [s["id"] for s in samples]
    assert "case_cis_01" in sample_ids
    assert "case_hgc_02" in sample_ids
    assert "case_lgc_03" in sample_ids
    assert "case_normal_04" in sample_ids
    assert "case_orifice_05" in sample_ids
    assert "case_bubble_06" in sample_ids

    # Check structure of first sample
    first = samples[0]
    assert first["id"] == "case_cis_01"
    assert "Tis" in first["name"]
    assert first["modality"] == "WLC"
    assert first["urgency"] == "CRITICAL"
    assert first["ground_truth"]["fine"] == "CIS"


def test_sample_image_endpoint(client: TestClient) -> None:
    """Verifies sample image retrieval and 404 for invalid sample."""
    resp = client.get("/api/v1/samples/case_cis_01/image")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert len(resp.content) > 1000  # Non-trivial image bytes

    # Non-existent sample
    resp_404 = client.get("/api/v1/samples/non_existent_case_999/image")
    assert resp_404.status_code == 404


def test_predict_with_file_upload(client: TestClient) -> None:
    """Verifies predict endpoint with genuine multipart image file upload."""
    sample_path = settings.demo_samples_dir / "01c5cce3.png"
    if not sample_path.is_file():
        sample_path = settings.project_root / "xvdhy-osfstorage-archive" / "images" / "01c5cce3.png"

    with open(sample_path, "rb") as f:
        file_bytes = f.read()

    files = {"file": ("01c5cce3.png", file_bytes, "image/png")}
    resp = client.post("/api/v1/predict", files=files)
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "success"
    assert data["success"] is True
    assert data["model_version"] == "3.1"
    assert data["task_type"] == "hierarchical"
    assert data["request_id"].startswith("inf_")

    # Hierarchy predictions for 01c5cce3.png
    assert data["binary"]["prediction"] == "ROI"
    assert data["coarse"]["prediction"] == "Malignant"
    assert data["fine"]["prediction"] == "CIS"
    assert data["fine"]["confidence"] >= 0.999
    assert data["urgency"] == "CRITICAL"
    assert data["consistency"]["is_consistent"] is True
    assert data["consistency"]["parent_from_fine"] == "Malignant"
    assert data["inference_time_ms"] > 0.0


def test_predict_with_sample_id_form(client: TestClient) -> None:
    """Verifies predict endpoint with sample_id passed as form field."""
    resp = client.post("/api/v1/predict", data={"sample_id": "case_orifice_05"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["coarse"]["prediction"] == "Anatomical landmarks"
    assert data["fine"]["prediction"] == "UreteralOrifice"
    assert data["urgency"] == "NORMAL"


def test_predict_with_sample_id_json(client: TestClient) -> None:
    """Verifies predict endpoint with sample_id passed in JSON body."""
    resp = client.post("/api/v1/predict", json={"sample_id": "case_bubble_06"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["coarse"]["prediction"] == "Foreign bodies"
    assert data["fine"]["prediction"] == "AirBubble"
    assert data["urgency"] == "NORMAL"


def test_predict_with_sample_id_query_param(client: TestClient) -> None:
    """Verifies predict endpoint with sample_id passed as query parameter."""
    resp = client.post("/api/v1/predict?sample_id=case_cis_01")
    assert resp.status_code == 200
    data = resp.json()
    assert data["fine"]["prediction"] == "CIS"


def test_predict_error_missing_inputs(client: TestClient) -> None:
    """Verifies 400 Bad Request when neither file nor sample_id is provided."""
    resp = client.post("/api/v1/predict")
    assert resp.status_code == 400
    assert "Either 'file' upload or 'sample_id' must be provided" in resp.json()["detail"]


def test_predict_error_unknown_sample_id(client: TestClient) -> None:
    """Verifies 404 Not Found for unregistered sample_id."""
    resp = client.post("/api/v1/predict", data={"sample_id": "ghost_case_404"})
    assert resp.status_code == 404
    assert "not found in curated registry" in resp.json()["detail"]


def test_predict_error_spoofed_file_format(client: TestClient) -> None:
    """Verifies 400 Bad Request for fake text file with .png extension."""
    fake_file = io.BytesIO(b"Hello world, I am not a real image!")
    files = {"file": ("malicious.png", fake_file, "image/png")}
    resp = client.post("/api/v1/predict", files=files)
    assert resp.status_code == 400
    assert "does not have a supported image signature" in resp.json()["detail"]


def test_predict_error_corrupt_image(client: TestClient) -> None:
    """Verifies 400 Bad Request for corrupt image byte stream."""
    corrupt_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x00corrupt_payload"
    files = {"file": ("corrupt.png", io.BytesIO(corrupt_bytes), "image/png")}
    resp = client.post("/api/v1/predict", files=files)
    assert resp.status_code == 400
    assert "Failed to decode image file" in resp.json()["detail"]


def test_predict_error_oversized_payload(client: TestClient) -> None:
    """Verifies 413 Payload Too Large when upload exceeds limit."""
    original_limit = settings.max_upload_bytes
    try:
        settings.max_upload_bytes = 1000  # Temporarily lower limit to 1000 bytes
        large_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 2000
        files = {"file": ("oversized.png", io.BytesIO(large_bytes), "image/png")}
        resp = client.post("/api/v1/predict", files=files)
        assert resp.status_code == 413
        assert "exceeds limit" in resp.json()["detail"]
    finally:
        settings.max_upload_bytes = original_limit
