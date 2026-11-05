"""Curated reference clinical samples endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.app.config import settings
from backend.app.schemas import GroundTruth, SampleItem

router = APIRouter(tags=["Samples"])

# Master Registry of Curated Reference Samples
CURATED_SAMPLES: List[Dict] = [
    {
        "id": "case_cis_01",
        "name": "Ung Thư Biểu Mô Tại Chỗ (CIS - Tis)",
        "filename": "01c5cce3.png",
        "thumbnail_url": "/api/v1/samples/case_cis_01/image",
        "description": "Tổn thương phẳng ác tính nguy cơ cao (Carcinoma in situ - Tis), biểu hiện ban đỏ dạng viêm bàng quang dễ bị bỏ sót.",
        "modality": "WLC",
        "patient_id": "PT-0952",
        "stage": "Tis",
        "urgency": "CRITICAL",
        "ground_truth": {"binary": "ROI", "coarse": "Malignant", "fine": "CIS"},
        "has_segmentation": False,
    },
    {
        "id": "case_hgc_02",
        "name": "Ung Thư Nhú Độ Cao (High-Grade Papillary - T2)",
        "filename": "0e1d3230.png",
        "thumbnail_url": "/api/v1/samples/case_hgc_02/image",
        "description": "Khối u ác tính dạng nhú độ mô học cao xâm lấn lớp cơ (T2), cấu trúc mạch máu cuộn dày bất thường.",
        "modality": "WLC",
        "patient_id": "PT-1547",
        "stage": "T2",
        "urgency": "HIGH",
        "ground_truth": {"binary": "ROI", "coarse": "Malignant", "fine": "HighGradePapillary"},
        "has_segmentation": False,
    },
    {
        "id": "case_lgc_03",
        "name": "Ung Thư Nhú Độ Thấp (Low-Grade Papillary - Ta)",
        "filename": "d326d133.png",
        "thumbnail_url": "/api/v1/samples/case_lgc_03/image",
        "description": "Khối u ác tính dạng nhú độ mô học thấp không xâm lấn (Ta), nhiều nhánh mịn dạng san hô mềm.",
        "modality": "WLC",
        "patient_id": "PT-1547",
        "stage": "Ta",
        "urgency": "MEDIUM",
        "ground_truth": {"binary": "ROI", "coarse": "Malignant", "fine": "LowGradePapillary"},
        "has_segmentation": False,
    },
    {
        "id": "case_normal_04",
        "name": "Niêm Mạc Bàng Quang Bình Thường",
        "filename": "004d25e9.png",
        "thumbnail_url": "/api/v1/samples/case_normal_04/image",
        "description": "Niêm mạc bàng quang hoàn toàn khỏe mạnh, nhẵn bóng, các nhánh mạch máu hình cành cây đều đặn.",
        "modality": "WLC",
        "patient_id": "PT-0818",
        "stage": "None",
        "urgency": "NORMAL",
        "ground_truth": {"binary": "Non-ROI", "coarse": "Normal mucosa", "fine": "Normal mucosa"},
        "has_segmentation": False,
    },
    {
        "id": "case_orifice_05",
        "name": "Lỗ Niệu Quản (Ureteral Orifice - Mốc Giải Phẫu)",
        "filename": "23a59431.png",
        "thumbnail_url": "/api/v1/samples/case_orifice_05/image",
        "description": "Mốc giải phẫu tự nhiên lỗ niệu quản mở vào tam giác bàng quang, có thể nhầm lẫn với tổn thương u.",
        "modality": "WLC",
        "patient_id": "PT-0761",
        "stage": "None",
        "urgency": "NORMAL",
        "ground_truth": {"binary": "Non-ROI", "coarse": "Anatomical landmarks", "fine": "UreteralOrifice"},
        "has_segmentation": True,
    },
    {
        "id": "case_bubble_06",
        "name": "Bọt Khí Quang Học (Air Bubble - Dị Vật)",
        "filename": "489f6a5e.png",
        "thumbnail_url": "/api/v1/samples/case_bubble_06/image",
        "description": "Bọt khí nội soi quang học tạo bóng phản xạ giả, thuộc nhóm dị vật không cần can thiệp y tế.",
        "modality": "WLC",
        "patient_id": "PT-0761",
        "stage": "None",
        "urgency": "NORMAL",
        "ground_truth": {"binary": "Non-ROI", "coarse": "Foreign bodies", "fine": "AirBubble"},
        "has_segmentation": True,
    },
]


def find_sample_dict(sample_id: str) -> Optional[Dict]:
    """Finds curated sample definition by ID, filename, or bare filename."""
    norm = sample_id.strip()
    for s in CURATED_SAMPLES:
        if s["id"] == norm or s["filename"] == norm or s["filename"].replace(".png", "") == norm:
            return s
    return None


def get_sample_image_path(filename: str) -> Path:
    """Resolves local absolute path to sample image file, ensuring no path traversal."""
    safe_filename = Path(filename).name
    path = (settings.demo_samples_dir / safe_filename).resolve()
    if path.is_file():
        return path
    # Secondary check in project root
    fallback = (settings.project_root / "xvdhy-osfstorage-archive" / "images" / safe_filename).resolve()
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Sample image '{filename}' not found at {path} or {fallback}")


@router.get("/samples", response_model=List[SampleItem])
async def list_samples() -> List[SampleItem]:
    """Returns list of curated clinical endoscopy reference cases."""
    return [
        SampleItem(
            id=s["id"],
            name=s["name"],
            filename=s["filename"],
            thumbnail_url=s["thumbnail_url"],
            description=s["description"],
            modality=s["modality"],
            patient_id=s.get("patient_id"),
            stage=s.get("stage"),
            urgency=s["urgency"],
            ground_truth=GroundTruth(**s["ground_truth"]),
            has_segmentation=s["has_segmentation"],
        )
        for s in CURATED_SAMPLES
    ]


@router.get("/samples/{sample_id}/image")
async def get_sample_image(sample_id: str) -> FileResponse:
    """Returns binary image stream for given curated reference case."""
    if ".." in sample_id or "/" in sample_id or "\\" in sample_id:
        raise HTTPException(status_code=400, detail="Invalid sample_id: path traversal sequence detected.")

    s = find_sample_dict(sample_id)
    if not s:
        raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")

    try:
        path = get_sample_image_path(s["filename"])
        return FileResponse(path, media_type="image/png", filename=s["filename"])
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
