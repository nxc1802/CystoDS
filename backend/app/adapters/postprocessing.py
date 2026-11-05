"""Postprocessing utilities for CystoDS hierarchical logits and clinical rules."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from cystods.taxonomy import (
    BINARY_NAMES,
    COARSE_NAMES,
    FINE_NAMES,
    FINE_PARENT_NAME,
)


def evaluate_clinical_urgency(
    coarse_pred: str,
    fine_pred: str,
    coarse_conf: float,
) -> Dict[str, str]:
    """Evaluates clinical urgency tier and action recommendation based on predictions."""
    if fine_pred == "CIS" or (coarse_pred == "Malignant" and coarse_conf >= 0.95 and fine_pred not in ("LowGradePapillary", "PreMalignant")):
        return {
            "urgency": "CRITICAL",
            "badge_color": "rose",
            "title": "Cảnh Báo Khẩn: Nghi Ngờ Ung Thư Biểu Mô Tại Chỗ (CIS)",
            "action": "Chỉ định sinh thiết đa điểm khẩn cấp và hội chẩn chuyên khoa Tiết niệu.",
            "recommendation": "Tổn thương phẳng có nguy cơ ác tính cao, không trì hoãn can thiệp.",
        }
    elif fine_pred in ("LowGradePapillary", "PreMalignant"):
        return {
            "urgency": "MEDIUM",
            "badge_color": "yellow",
            "title": "Theo Dõi: Tổn Thương Nguy Cơ Thấp / Tiền Ung Thư",
            "action": "Theo dõi định kỳ hoặc cắt đốt u nhú kích thước nhỏ.",
            "recommendation": "Theo dõi nội soi định kỳ sau 3-6 tháng.",
        }
    elif fine_pred == "HighGradePapillary" or coarse_pred == "Malignant":
        return {
            "urgency": "HIGH",
            "badge_color": "amber",
            "title": "Cảnh Báo: Tổn Thương Ác Tính (Malignant)",
            "action": "Chỉ định nội soi cắt u bàng quang qua ngả niệu đạo (TURBT) và giải phẫu bệnh.",
            "recommendation": "Lên lịch can thiệp phẫu thuật nội soi sớm trong 2 tuần.",
        }
    elif coarse_pred == "Non-malignant":
        return {
            "urgency": "MEDIUM",
            "badge_color": "blue",
            "title": "Tổn Thương Lành Tính / Viêm Nhiễm (Non-malignant)",
            "action": "Điều trị nội khoa viêm bàng quang và tái khám sau 3 tháng.",
            "recommendation": "Không phát hiện dấu hiệu ác tính xâm lấn.",
        }
    else:
        return {
            "urgency": "NORMAL",
            "badge_color": "emerald",
            "title": "Bình Thường / Mốc Giải Phẫu / Dị Vật",
            "action": "Không phát hiện dấu hiệu u bàng quang nghi ngờ.",
            "recommendation": "Niêm mạc bình thường hoặc biến đổi giải phẫu/dị vật thông thường.",
        }


def evaluate_hierarchical_consistency(
    coarse_pred: str,
    fine_pred: str,
) -> Dict[str, Any]:
    """Evaluates whether fine prediction's mapped parent aligns with direct coarse prediction."""
    fine_parent = FINE_PARENT_NAME.get(fine_pred, "Normal mucosa")
    is_consistent = bool(fine_parent == coarse_pred)
    return {
        "is_consistent": is_consistent,
        "parent_from_fine": fine_parent,
        "direct_coarse_prediction": coarse_pred,
    }


def postprocess_logits(
    outputs: Dict[str, torch.Tensor],
    prior_tau: float = 0.0,
    fine_prior_counts: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Transforms raw logits into normalized probabilities, rankings, and clinical tiers."""
    bin_logits = outputs["binary_logits"]
    coarse_logits = outputs["coarse_logits"]
    fine_logits = outputs["fine_logits"]

    # Optional balanced softmax prior calibration on fine logits
    if prior_tau > 0.0 and fine_prior_counts and len(fine_prior_counts) == len(FINE_NAMES):
        prior_adj = torch.tensor(
            [math.log(max(c, 1)) for c in fine_prior_counts],
            dtype=fine_logits.dtype,
            device=fine_logits.device,
        )
        fine_logits = fine_logits - prior_tau * prior_adj

    bin_probs = F.softmax(bin_logits, dim=-1)[0].cpu().tolist()
    coarse_probs = F.softmax(coarse_logits, dim=-1)[0].cpu().tolist()
    fine_probs = F.softmax(fine_logits, dim=-1)[0].cpu().tolist()

    pred_bin_idx = int(torch.argmax(bin_logits, dim=-1)[0].item())
    pred_coarse_idx = int(torch.argmax(coarse_logits, dim=-1)[0].item())
    pred_fine_idx = int(torch.argmax(fine_logits, dim=-1)[0].item())

    pred_bin_name = BINARY_NAMES[pred_bin_idx]
    pred_coarse_name = COARSE_NAMES[pred_coarse_idx]
    pred_fine_name = FINE_NAMES[pred_fine_idx]

    # Top-K Fine classes
    top3_indices = torch.topk(fine_logits[0], k=3).indices.cpu().tolist()
    top3_fine = [
        {
            "class": FINE_NAMES[idx],
            "class_name": FINE_NAMES[idx],
            "probability": round(fine_probs[idx], 6),
            "parent": FINE_PARENT_NAME.get(FINE_NAMES[idx], "Unknown"),
        }
        for idx in top3_indices
    ]

    top5_indices = torch.topk(fine_logits[0], k=5).indices.cpu().tolist()
    top5_fine = [
        {
            "class": FINE_NAMES[idx],
            "class_name": FINE_NAMES[idx],
            "probability": round(fine_probs[idx], 6),
            "parent": FINE_PARENT_NAME.get(FINE_NAMES[idx], "Unknown"),
        }
        for idx in top5_indices
    ]

    consistency = evaluate_hierarchical_consistency(pred_coarse_name, pred_fine_name)
    coarse_conf = float(coarse_probs[pred_coarse_idx])
    urgency_info = evaluate_clinical_urgency(pred_coarse_name, pred_fine_name, coarse_conf)

    return {
        "binary": {
            "prediction": pred_bin_name,
            "confidence": round(bin_probs[pred_bin_idx], 6),
            "probabilities": {k: round(v, 6) for k, v in zip(BINARY_NAMES, bin_probs)},
        },
        "coarse": {
            "prediction": pred_coarse_name,
            "confidence": round(coarse_probs[pred_coarse_idx], 6),
            "probabilities": {k: round(v, 6) for k, v in zip(COARSE_NAMES, coarse_probs)},
        },
        "fine": {
            "prediction": pred_fine_name,
            "confidence": round(fine_probs[pred_fine_idx], 6),
            "top_3": top3_fine,
            "top_5": top5_fine,
            "probabilities": {k: round(v, 6) for k, v in zip(FINE_NAMES, fine_probs)},
        },
        "consistency": consistency,
        "hierarchy_metrics": consistency,
        "clinical_assessment": urgency_info,
        "urgency": urgency_info["urgency"],
    }
