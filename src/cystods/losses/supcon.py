"""Supervised contrastive loss.

Extracted from ``cystods.core`` (Step 2 refactor).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
    projections: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if projections.ndim != 2:
        raise ValueError("SupCon projections must have shape [N, D].")
    if labels.ndim != 1 or len(labels) != len(projections):
        raise ValueError("SupCon labels must align with projections.")
    if temperature <= 0:
        raise ValueError("SupCon temperature must be positive.")
    if len(projections) < 2:
        return projections.sum() * 0.0
    features = F.normalize(projections, dim=1)
    logits = features @ features.T / temperature
    diagonal = torch.eye(
        len(features), dtype=torch.bool, device=features.device
    )
    positive_mask = labels[:, None].eq(labels[None, :]) & ~diagonal
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * (~diagonal)
    log_prob = logits - torch.log(
        exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
    )
    positives_per_anchor = positive_mask.sum(dim=1)
    valid = positives_per_anchor > 0
    if not valid.any():
        return projections.sum() * 0.0
    mean_positive_log_prob = (
        (positive_mask * log_prob).sum(dim=1)
        / positives_per_anchor.clamp_min(1)
    )
    return -mean_positive_log_prob[valid].mean()
