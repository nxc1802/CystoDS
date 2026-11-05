"""Unit test suite for CystoDS backend validators, transforms, and postprocessors."""

from __future__ import annotations

import io
import pytest
from PIL import Image
import torch

from backend.app.adapters.preprocessing import (
    CenterFractionCrop,
    InputValidator,
    build_eval_transform,
)
from backend.app.adapters.postprocessing import (
    evaluate_clinical_urgency,
    evaluate_hierarchical_consistency,
    postprocess_logits,
)
from cystods.taxonomy import BINARY_NAMES, COARSE_NAMES, FINE_NAMES


# ── CenterFractionCrop Tests ────────────────────────────────────────────
def test_center_fraction_crop_dimensions() -> None:
    cropper = CenterFractionCrop(0.92)
    img = Image.new("RGB", (1000, 1000), color="blue")
    cropped = cropper(img)
    assert cropped.size == (920, 920)

    img_rect = Image.new("RGB", (1920, 1080), color="red")
    cropped_rect = cropper(img_rect)
    assert cropped_rect.size == (1766, 994)


def test_center_fraction_crop_invalid_ratio() -> None:
    with pytest.raises(ValueError, match="must be in \\(0, 1\\]"):
        CenterFractionCrop(0.0)

    with pytest.raises(ValueError, match="must be in \\(0, 1\\]"):
        CenterFractionCrop(1.2)


# ── Transform Chain Tests ───────────────────────────────────────────────
def test_build_eval_transform_output_tensor() -> None:
    transform = build_eval_transform(image_size=224, crop_ratio=0.92)
    img = Image.new("RGB", (500, 500), color="green")
    tensor = transform(img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


# ── InputValidator Tests ────────────────────────────────────────────────
def _create_test_image_bytes(format_name: str, size: tuple[int, int] = (100, 100), mode: str = "RGB") -> bytes:
    img = Image.new(mode, size, color="teal")
    buf = io.BytesIO()
    img.save(buf, format=format_name)
    return buf.getvalue()


def test_input_validator_supported_formats() -> None:
    validator = InputValidator()
    for fmt in ["PNG", "JPEG", "BMP", "WEBP"]:
        raw = _create_test_image_bytes(fmt)
        decoded = validator.validate_and_decode(raw, filename=f"test.{fmt.lower()}")
        assert isinstance(decoded, Image.Image)
        assert decoded.mode == "RGB"


def test_input_validator_rejects_empty_file() -> None:
    validator = InputValidator()
    with pytest.raises(ValueError, match="empty \\(0 bytes\\)"):
        validator.validate_and_decode(b"", filename="empty.png")


def test_input_validator_rejects_oversized_payload() -> None:
    validator = InputValidator(max_bytes=1024)
    fake_large_payload = b"\x89PNG\r\n\x1a\n" + b"\x00" * 2000
    with pytest.raises(ValueError, match="exceeds maximum allowed"):
        validator.validate_and_decode(fake_large_payload, filename="big.png")


def test_input_validator_rejects_non_image_extension_spoofing() -> None:
    validator = InputValidator()
    fake_text_file = b"This is not a real PNG image content."
    with pytest.raises(ValueError, match="does not have a supported image signature"):
        validator.validate_and_decode(fake_text_file, filename="exploit.png")


def test_input_validator_rejects_corrupted_image() -> None:
    validator = InputValidator()
    # Has PNG magic bytes but corrupt body
    corrupt_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"bad_corrupt_data"
    with pytest.raises(ValueError, match="Failed to decode image file"):
        validator.validate_and_decode(corrupt_png, filename="broken.png")


def test_input_validator_dimension_bounds() -> None:
    validator = InputValidator(min_dimension=32, max_dimension=500)

    too_small = _create_test_image_bytes("PNG", size=(16, 16))
    with pytest.raises(ValueError, match="is too small"):
        validator.validate_and_decode(too_small, filename="small.png")

    too_large = _create_test_image_bytes("PNG", size=(600, 600))
    with pytest.raises(ValueError, match="exceeds maximum bounds"):
        validator.validate_and_decode(too_large, filename="huge.png")


def test_input_validator_converts_rgba_to_rgb() -> None:
    validator = InputValidator()
    rgba_bytes = _create_test_image_bytes("PNG", size=(100, 100), mode="RGBA")
    decoded = validator.validate_and_decode(rgba_bytes, filename="transparent.png")
    assert decoded.mode == "RGB"


# ── Postprocessing & Consistency Tests ──────────────────────────────────
def test_hierarchical_consistency_matching_pairs() -> None:
    # Malignant -> CIS
    r1 = evaluate_hierarchical_consistency("Malignant", "CIS")
    assert r1["is_consistent"] is True
    assert r1["parent_from_fine"] == "Malignant"

    # Anatomical landmarks -> UreteralOrifice
    r2 = evaluate_hierarchical_consistency("Anatomical landmarks", "UreteralOrifice")
    assert r2["is_consistent"] is True

    # Foreign bodies -> AirBubble
    r3 = evaluate_hierarchical_consistency("Foreign bodies", "AirBubble")
    assert r3["is_consistent"] is True


def test_hierarchical_consistency_mismatching_pairs() -> None:
    # Coarse Normal mucosa with Fine CIS is a clinical hierarchy conflict
    r = evaluate_hierarchical_consistency("Normal mucosa", "CIS")
    assert r["is_consistent"] is False
    assert r["parent_from_fine"] == "Malignant"
    assert r["direct_coarse_prediction"] == "Normal mucosa"


# ── Clinical Urgency Classification Tests ───────────────────────────────
def test_clinical_urgency_rules() -> None:
    # CIS is always critical
    u_cis = evaluate_clinical_urgency("Malignant", "CIS", coarse_conf=0.99)
    assert u_cis["urgency"] == "CRITICAL"
    assert u_cis["badge_color"] == "rose"

    # Malignant with >= 0.95 coarse confidence is critical even if fine is something else
    u_mal_conf = evaluate_clinical_urgency("Malignant", "UnknownSubclass", coarse_conf=0.98)
    assert u_mal_conf["urgency"] == "CRITICAL"

    # HighGradePapillary is High
    u_hgc = evaluate_clinical_urgency("Malignant", "HighGradePapillary", coarse_conf=0.80)
    assert u_hgc["urgency"] == "HIGH"
    assert u_hgc["badge_color"] == "amber"

    # LowGradePapillary is Medium
    u_lgc = evaluate_clinical_urgency("Malignant", "LowGradePapillary", coarse_conf=0.80)
    assert u_lgc["urgency"] == "MEDIUM"

    # Non-malignant is Medium
    u_nonmal = evaluate_clinical_urgency("Non-malignant", "InflammationNOS", coarse_conf=0.85)
    assert u_nonmal["urgency"] == "MEDIUM"
    assert u_nonmal["badge_color"] == "blue"

    # Normal mucosa / anatomical / foreign is Normal
    u_norm = evaluate_clinical_urgency("Normal mucosa", "AirBubble", coarse_conf=0.99)
    assert u_norm["urgency"] == "NORMAL"
    assert u_norm["badge_color"] == "emerald"


# ── Postprocess Logits Function Tests ────────────────────────────────────
def test_postprocess_logits_distribution() -> None:
    bin_logits = torch.tensor([[1.0, 5.0]])  # Argmax: 1 (ROI)
    coarse_logits = torch.tensor([[5.0, 1.0, 0.0, 0.0, 0.0]])  # Argmax: 0 (Malignant)
    fine_logits = torch.zeros(1, 22)
    fine_logits[0, 2] = 10.0  # Index 2: CIS

    outputs = {
        "binary_logits": bin_logits,
        "coarse_logits": coarse_logits,
        "fine_logits": fine_logits,
    }

    res = postprocess_logits(outputs)
    assert res["binary"]["prediction"] == "ROI"
    assert res["coarse"]["prediction"] == "Malignant"
    assert res["fine"]["prediction"] == "CIS"
    assert res["fine"]["top_3"][0]["class"] == "CIS"
    assert res["fine"]["top_3"][0]["parent"] == "Malignant"
    assert res["consistency"]["is_consistent"] is True
    assert res["urgency"] == "CRITICAL"
