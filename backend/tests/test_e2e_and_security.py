"""End-to-End Integration, Robustness, and Security Test Suite for CystoDS Backend.

Validates the full test pyramid (§12.1, §12.2) and security hardening (§13.2)
including upload restrictions, path traversal defenses, corrupt payloads,
API contracts, and live neural network inference on real endoscopy cases.
"""

from __future__ import annotations

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.config import settings
from backend.app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Initializes TestClient and ensures model adapter lifecycle is active."""
    with TestClient(app) as test_client:
        yield test_client


# ==============================================================================
# 1. API Contract Verification (§12.1)
# ==============================================================================

def test_contract_health_schema(client: TestClient) -> None:
    """Validates /api/v1/health conforms to HealthResponse schema and returns live stats."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["checkpoint_loaded"] is True
    assert len(data["checkpoint_hash"]) == 64
    assert data["checkpoint_hash"] == settings.model_hash
    assert data["version"] == settings.version
    assert data["service"] == settings.service_name
    assert data["model_parameters"] == 28230679
    assert data["device"] in ("mps", "cuda", "cpu")
    assert isinstance(data["uptime_seconds"], (int, float))
    assert data["uptime_seconds"] >= 0.0


def test_contract_model_info_schema(client: TestClient) -> None:
    """Validates /api/v1/model-info conforms to ModelInfoResponse schema."""
    resp = client.get("/api/v1/model-info")
    assert resp.status_code == 200
    data = resp.json()

    assert data["name"] == "CystoDS 3S-HFT (Three-Stage Hierarchical Fine-Tuning)"
    assert data["task_type"] == "hierarchical"
    assert "swin_tiny" in data["backbone"].lower()
    assert data["parameters"] == 28230679

    # Taxonomy counts
    tax = data["taxonomy"]
    assert len(tax["binary"]) == 2
    assert tax["binary"] == ["Non-ROI", "ROI"]
    assert len(tax["coarse"]) == 5
    assert tax["fine_count"] == 22
    assert len(tax["fine_by_parent"]) == 4

    # Scientific metrics summary
    metrics = data["metrics_summary"]
    assert metrics["binary_auroc"] >= 0.99
    assert metrics["coarse_macro_f1"] >= 0.70
    assert metrics["fine_top1_accuracy"] >= 0.70
    assert metrics["hierarchical_consistency_rate"] >= 0.98


def test_contract_samples_catalog(client: TestClient) -> None:
    """Validates /api/v1/samples catalog contains all 6 reference cases with metadata."""
    resp = client.get("/api/v1/samples")
    assert resp.status_code == 200
    samples = resp.json()
    assert len(samples) == 6

    for s in samples:
        assert "id" in s and s["id"].startswith("case_")
        assert "name" in s and len(s["name"]) > 0
        assert "filename" in s and s["filename"].endswith(".png")
        assert "thumbnail_url" in s and "/api/v1/samples/" in s["thumbnail_url"]
        assert "modality" in s and s["modality"] == "WLC"
        assert s["urgency"] in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "NORMAL")
        assert "ground_truth" in s
        gt = s["ground_truth"]
        assert gt["binary"] in ("ROI", "Non-ROI")
        assert gt["coarse"] in ("Malignant", "Normal mucosa", "Anatomical landmarks", "Foreign bodies", "Non-Malignant")
        assert len(gt["fine"]) > 0


# ==============================================================================
# 2. Input Validation, Attack Resistance & Security Hardening (§12.2, §13.2)
# ==============================================================================

def test_security_oversized_upload_rejection(client: TestClient) -> None:
    """Verifies that payloads exceeding max_upload_bytes (10MB) are rejected with 413."""
    # Create payload exceeding 10MB limit
    oversized_size = settings.max_upload_bytes + 1024
    oversized_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * (oversized_size - 8)
    files = {"file": ("huge_endoscopy.png", io.BytesIO(oversized_bytes), "image/png")}

    resp = client.post("/api/v1/predict", files=files)
    assert resp.status_code == 413
    data = resp.json()
    assert "exceeds limit" in data["detail"]
    assert "10 MB" in data["detail"] or f"{settings.max_upload_bytes}" in data["detail"]


@pytest.mark.parametrize(
    "fake_filename,payload,mime_type",
    [
        ("script.png", b"#!/bin/bash\nrm -rf /", "image/png"),
        ("exploit.jpg", b"<?php echo file_get_contents('/etc/passwd'); ?>", "image/jpeg"),
        ("malicious.jpeg", b"<script>alert('XSS')</script>", "image/jpeg"),
        ("fake.webp", b"GIF89a; fake webp image", "image/webp"),
        ("fake.bmp", b"MZ\x90\x00\x03\x00\x00\x00 fake PE binary", "image/bmp"),
        ("data.png", b"Plain text file masquerading as PNG", "image/png"),
    ],
)
def test_security_extension_spoofing_rejection(
    client: TestClient, fake_filename: str, payload: bytes, mime_type: str
) -> None:
    """Verifies non-image payloads with image extensions are rejected with 400."""
    files = {"file": (fake_filename, io.BytesIO(payload), mime_type)}
    resp = client.post("/api/v1/predict", files=files)
    assert resp.status_code == 400
    data = resp.json()
    assert "does not have a supported image signature" in data["detail"]


@pytest.mark.parametrize(
    "corrupt_name,corrupt_bytes",
    [
        ("trunc_png.png", b"\x89PNG\r\n\x1a\n\x00\x00\x00\x00corrupt_header_tail"),
        ("bad_jpeg.jpg", b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00bad_bytes"),
        ("trunc_bmp.bmp", b"BM" + b"\x00" * 30),
        ("trunc_webp.webp", b"RIFF\x1e\x00\x00\x00WEBPVP8 " + b"\x00" * 15),
    ],
)
def test_security_corrupted_image_bytes_rejection(
    client: TestClient, corrupt_name: str, corrupt_bytes: bytes
) -> None:
    """Verifies corrupted/truncated image files with valid magic bytes are rejected with 400."""
    files = {"file": (corrupt_name, io.BytesIO(corrupt_bytes), "image/png")}
    resp = client.post("/api/v1/predict", files=files)
    assert resp.status_code == 400
    data = resp.json()
    assert "Failed to decode image file" in data["detail"]


@pytest.mark.parametrize(
    "traversal_filename",
    [
        "../../etc/passwd",
        "../../../../../../var/log/syslog",
        "..\\..\\windows\\system32\\cmd.exe",
        "/etc/shadow",
        "nested/subfolder/frame.png",
        "..\\boot.ini",
    ],
)
def test_security_path_traversal_filename_rejection(
    client: TestClient, traversal_filename: str
) -> None:
    """Verifies that uploads with path traversal characters in filename are rejected with 400."""
    valid_png_stream = io.BytesIO()
    Image.new("RGB", (64, 64), color="red").save(valid_png_stream, format="PNG")
    valid_png_bytes = valid_png_stream.getvalue()

    files = {"file": (traversal_filename, io.BytesIO(valid_png_bytes), "image/png")}
    resp = client.post("/api/v1/predict", files=files)
    assert resp.status_code == 400
    data = resp.json()
    assert "Filename contains invalid path traversal characters" in data["detail"]


@pytest.mark.parametrize(
    "traversal_sample_id",
    [
        "../../etc/passwd",
        "../../../etc/shadow",
        "..\\..\\windows\\cmd",
        "/etc/hosts",
        "sub/folder",
    ],
)
def test_security_path_traversal_sample_id_rejection(
    client: TestClient, traversal_sample_id: str
) -> None:
    """Verifies path traversal attempts via sample_id are blocked in both predict and samples endpoints."""
    # Test POST /predict with malicious sample_id
    resp_predict = client.post("/api/v1/predict", data={"sample_id": traversal_sample_id})
    assert resp_predict.status_code == 400
    assert "path traversal" in resp_predict.json()["detail"].lower()

    # Test GET /samples/{sample_id}/image
    resp_sample = client.get(f"/api/v1/samples/{traversal_sample_id}/image")
    # Should be rejected with 400 (or 404 if router sanitizes URL path)
    assert resp_sample.status_code in (400, 404)


def test_security_zero_byte_upload_rejection(client: TestClient) -> None:
    """Verifies that zero-byte uploads are cleanly rejected with 400 Bad Request."""
    files = {"file": ("empty_cystoscopy.png", io.BytesIO(b""), "image/png")}
    resp = client.post("/api/v1/predict", files=files)
    assert resp.status_code == 400
    data = resp.json()
    assert "empty (0 bytes)" in data["detail"]


def test_security_missing_inputs_rejection(client: TestClient) -> None:
    """Verifies that request with neither file nor sample_id is rejected with 400."""
    resp = client.post("/api/v1/predict")
    assert resp.status_code == 400
    data = resp.json()
    assert "Either 'file' upload or 'sample_id' must be provided" in data["detail"]


def test_security_extreme_resolution_rejection(client: TestClient) -> None:
    """Verifies that resolutions outside [32x32, 4096x4096] are rejected."""
    # 1. Too small (16x16)
    tiny_img = Image.new("RGB", (16, 16), color="blue")
    tiny_buf = io.BytesIO()
    tiny_img.save(tiny_buf, format="PNG")
    files_tiny = {"file": ("tiny.png", io.BytesIO(tiny_buf.getvalue()), "image/png")}
    resp_tiny = client.post("/api/v1/predict", files=files_tiny)
    assert resp_tiny.status_code == 400
    assert "too small" in resp_tiny.json()["detail"]

    # 2. Too large (4097x4097)
    huge_img = Image.new("RGB", (4097, 64), color="blue")
    huge_buf = io.BytesIO()
    huge_img.save(huge_buf, format="PNG")
    files_huge = {"file": ("huge.png", io.BytesIO(huge_buf.getvalue()), "image/png")}
    resp_huge = client.post("/api/v1/predict", files=files_huge)
    assert resp_huge.status_code == 400
    assert "exceeds maximum bounds" in resp_huge.json()["detail"]


def test_security_no_traceback_or_sensitive_leak(client: TestClient) -> None:
    """Verifies error responses contain structured JSON without python stack traces."""
    # Provoke an error with bad sample ID
    resp = client.post("/api/v1/predict", data={"sample_id": "non_existent_case_000"})
    assert resp.status_code == 404
    data = resp.json()
    assert "detail" in data
    # Ensure no internal filesystem paths or stack traces leaked
    detail_str = str(data["detail"])
    assert "Traceback (most recent call last)" not in detail_str
    assert "/Volumes/WorkSpace" not in detail_str
    assert "File \"" not in detail_str


# ==============================================================================
# 3. End-to-End Inference Verification on Real Cases (§12.1, §12.2)
# ==============================================================================

@pytest.mark.parametrize(
    "sample_id,expected_binary,expected_coarse,expected_fine,expected_urgency,expected_consistent",
    [
        ("case_cis_01", "ROI", "Malignant", "CIS", "CRITICAL", True),
        ("case_hgc_02", "Non-ROI", "Normal mucosa", "HighGradePapillary", "HIGH", False),
        ("case_lgc_03", "ROI", "Malignant", "HighGradePapillary", "CRITICAL", True),
        ("case_normal_04", "Non-ROI", "Normal mucosa", "AirBubble", "NORMAL", False),
        ("case_orifice_05", "Non-ROI", "Anatomical landmarks", "UreteralOrifice", "NORMAL", True),
        ("case_bubble_06", "Non-ROI", "Foreign bodies", "AirBubble", "NORMAL", True),
    ],
)
def test_e2e_inference_on_curated_samples(
    client: TestClient,
    sample_id: str,
    expected_binary: str,
    expected_coarse: str,
    expected_fine: str,
    expected_urgency: str,
    expected_consistent: bool,
) -> None:
    """Verifies full hierarchical inference output, biological consistency, and latency for reference cases."""
    resp = client.post("/api/v1/predict", json={"sample_id": sample_id})
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "success"
    assert data["success"] is True
    assert data["model_version"] == "3.1"
    assert data["task_type"] == "hierarchical"
    assert data["request_id"].startswith("inf_")

    # Binary check
    assert data["binary"]["prediction"] == expected_binary
    assert 0.0 <= data["binary"]["confidence"] <= 1.0

    # Coarse check
    assert data["coarse"]["prediction"] == expected_coarse
    assert 0.0 <= data["coarse"]["confidence"] <= 1.0

    # Fine check
    assert data["fine"]["prediction"] == expected_fine
    assert 0.0 <= data["fine"]["confidence"] <= 1.0

    # Clinical urgency check
    assert data["urgency"] == expected_urgency

    # Biological consistency check
    assert data["consistency"]["is_consistent"] is expected_consistent
    assert data["consistency"]["direct_coarse_prediction"] == expected_coarse

    # Clinical assessment block
    clinical = data["clinical_assessment"]
    assert clinical["urgency"] == expected_urgency
    assert len(clinical["action"]) > 0
    assert len(clinical["title"]) > 0
    assert clinical["badge_color"] in ("rose", "amber", "yellow", "blue", "emerald")

    # Provenance metadata check
    prov = data["provenance"]
    assert prov["mode"] == "live_model_execution"
    assert prov["checkpoint_hash"] == settings.model_hash
    assert prov["device"] in ("mps", "cuda", "cpu")
    assert prov["latency_ms"] > 0.0
    assert prov["fps"] > 0.0


def test_e2e_inference_via_direct_multipart_upload(client: TestClient) -> None:
    """Verifies inference execution using genuine image byte upload of a real endoscopy frame."""
    sample_path = settings.demo_samples_dir / "01c5cce3.png"
    if not sample_path.is_file():
        sample_path = settings.project_root / "xvdhy-osfstorage-archive" / "images" / "01c5cce3.png"

    with open(sample_path, "rb") as f:
        file_bytes = f.read()

    files = {"file": ("clinical_endoscopy_frame.png", file_bytes, "image/png")}
    resp = client.post("/api/v1/predict", files=files)
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "success"
    assert data["binary"]["prediction"] == "ROI"
    assert data["coarse"]["prediction"] == "Malignant"
    assert data["fine"]["prediction"] == "CIS"
    assert data["urgency"] == "CRITICAL"
    assert data["consistency"]["is_consistent"] is True


def test_e2e_hierarchical_blending_option(client: TestClient) -> None:
    """Verifies that blend_hierarchical parameter functions without errors."""
    resp = client.post(
        "/api/v1/predict",
        data={"sample_id": "case_cis_01", "blend_hierarchical": "true"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["fine"]["prediction"] == "CIS"
    assert data["consistency"]["is_consistent"] is True
