"""Reference parity tests verifying mathematical and label equivalence.

Compares the FastAPI CystoDSModelAdapter against the raw reference PyTorch pipeline
for all 6 curated clinical endoscopy cases from the OSF archive.
"""

from __future__ import annotations

from pathlib import Path
import pytest
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from backend.app.adapters.model_adapter import CystoDSModelAdapter
from backend.app.config import settings
from cystods.data.transforms import CenterFractionCrop
from cystods.models.two_stage_hierarchical import TwoStageDecoupledHierarchicalModel
from cystods.taxonomy import BINARY_NAMES, COARSE_NAMES, FINE_NAMES

CURATED_CASES = [
    {
        "filename": "01c5cce3.png",
        "expected_binary": "ROI",
        "expected_coarse": "Malignant",
        "expected_fine": "CIS",
    },
    {
        "filename": "0e1d3230.png",
        "expected_binary": "Non-ROI",
        "expected_coarse": "Normal mucosa",
        "expected_fine": "HighGradePapillary",
    },
    {
        "filename": "d326d133.png",
        "expected_binary": "ROI",
        "expected_coarse": "Malignant",
        "expected_fine": "HighGradePapillary",
    },
    {
        "filename": "004d25e9.png",
        "expected_binary": "Non-ROI",
        "expected_coarse": "Normal mucosa",
        "expected_fine": "AirBubble",
    },
    {
        "filename": "23a59431.png",
        "expected_binary": "Non-ROI",
        "expected_coarse": "Anatomical landmarks",
        "expected_fine": "UreteralOrifice",
    },
    {
        "filename": "489f6a5e.png",
        "expected_binary": "Non-ROI",
        "expected_coarse": "Foreign bodies",
        "expected_fine": "AirBubble",
    },
]


@pytest.fixture(scope="module")
def reference_system():
    """Initializes raw PyTorch reference model and adapter on CPU for deterministic parity."""
    ckpt_path = settings.model_path
    if not ckpt_path.is_file():
        pytest.skip(f"Canonical checkpoint missing at {ckpt_path}")

    # 1. Raw Reference Model
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = dict(ckpt.get("config", {}))
    cfg["pretrained"] = False
    ref_model = TwoStageDecoupledHierarchicalModel(cfg)
    ref_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    ref_model.eval()

    ref_transform = transforms.Compose(
        [
            CenterFractionCrop(0.92),
            transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # 2. Production Model Adapter
    adapter = CystoDSModelAdapter(device_preference="cpu")
    adapter.load(ckpt_path)

    return ref_model, ref_transform, adapter


@pytest.mark.parametrize("case", CURATED_CASES, ids=lambda c: c["filename"])
def test_curated_sample_reference_parity(reference_system, case: dict) -> None:
    """Verifies that ModelAdapter matches raw reference pipeline with tolerance < 1e-4."""
    ref_model, ref_transform, adapter = reference_system
    image_path = settings.demo_samples_dir / case["filename"]
    if not image_path.is_file():
        # Fallback to project root path
        image_path = settings.project_root / "xvdhy-osfstorage-archive" / "images" / case["filename"]

    assert image_path.is_file(), f"Curated sample file missing: {image_path}"

    image = Image.open(image_path).convert("RGB")

    # 1. Execute Raw Reference Pipeline
    ref_tensor = ref_transform(image).unsqueeze(0)
    with torch.no_grad():
        ref_out = ref_model(ref_tensor)

    ref_bin_probs = F.softmax(ref_out["binary_logits"], dim=-1)[0].tolist()
    ref_coarse_probs = F.softmax(ref_out["coarse_logits"], dim=-1)[0].tolist()
    ref_fine_probs = F.softmax(ref_out["fine_logits"], dim=-1)[0].tolist()

    ref_bin_pred = BINARY_NAMES[ref_out["binary_logits"].argmax().item()]
    ref_coarse_pred = COARSE_NAMES[ref_out["coarse_logits"].argmax().item()]
    ref_fine_pred = FINE_NAMES[ref_out["fine_logits"].argmax().item()]

    # 2. Execute FastAPI Model Adapter
    adapter_res = adapter.predict_image(image)

    # 3. Label Parity Assertions (Exact match)
    assert adapter_res["binary"]["prediction"] == ref_bin_pred == case["expected_binary"], (
        f"Binary label mismatch on {case['filename']}: "
        f"adapter={adapter_res['binary']['prediction']}, ref={ref_bin_pred}, expected={case['expected_binary']}"
    )

    assert adapter_res["coarse"]["prediction"] == ref_coarse_pred == case["expected_coarse"], (
        f"Coarse label mismatch on {case['filename']}: "
        f"adapter={adapter_res['coarse']['prediction']}, ref={ref_coarse_pred}, expected={case['expected_coarse']}"
    )

    assert adapter_res["fine"]["prediction"] == ref_fine_pred == case["expected_fine"], (
        f"Fine label mismatch on {case['filename']}: "
        f"adapter={adapter_res['fine']['prediction']}, ref={ref_fine_pred}, expected={case['expected_fine']}"
    )

    # 4. Numerical Tolerance Assertions (|P_adapter - P_ref| < 1e-4)
    tolerance = 1e-4

    # Binary probabilities parity
    for idx, name in enumerate(BINARY_NAMES):
        p_adapter = adapter_res["binary"]["probabilities"][name]
        p_ref = ref_bin_probs[idx]
        diff = abs(p_adapter - p_ref)
        assert diff < tolerance, (
            f"Binary prob divergence on {name} ({case['filename']}): "
            f"adapter={p_adapter}, ref={p_ref}, diff={diff}"
        )

    # Coarse probabilities parity
    for idx, name in enumerate(COARSE_NAMES):
        p_adapter = adapter_res["coarse"]["probabilities"][name]
        p_ref = ref_coarse_probs[idx]
        diff = abs(p_adapter - p_ref)
        assert diff < tolerance, (
            f"Coarse prob divergence on {name} ({case['filename']}): "
            f"adapter={p_adapter}, ref={p_ref}, diff={diff}"
        )

    # Fine probabilities parity (Top-3)
    for top_item in adapter_res["fine"]["top_3"]:
        c_name = top_item["class"]
        f_idx = FINE_NAMES.index(c_name)
        p_adapter = top_item["probability"]
        p_ref = ref_fine_probs[f_idx]
        diff = abs(p_adapter - p_ref)
        assert diff < tolerance, (
            f"Fine top-3 prob divergence on {c_name} ({case['filename']}): "
            f"adapter={p_adapter}, ref={p_ref}, diff={diff}"
        )
