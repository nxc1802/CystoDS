"""Hierarchy consistency losses: binary↔coarse and coarse↔fine.

Extracted from ``cystods.core`` (Step 2 refactor).
"""

from __future__ import annotations

import torch
from torch import nn

from cystods.losses.classification import FineLongTailLoss
from cystods.taxonomy import (
    COARSE_NAMES,
    COARSE_TO_BINARY_MATRIX,
    COARSE_TO_ID,
    FINE_NAMES,
    FINE_TO_COARSE_MATRIX,
)


def negative_log_correct_parent_mass(
    parent_probabilities: torch.Tensor,
    parent_targets: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    if parent_probabilities.ndim != 2:
        raise ValueError("parent_probabilities must have shape [N, C].")
    if parent_targets.ndim != 1:
        raise ValueError("parent_targets must have shape [N].")
    if len(parent_probabilities) != len(parent_targets):
        raise ValueError(
            "Parent probabilities and targets must have equal batch size."
        )
    if len(parent_targets) == 0:
        return parent_probabilities.sum() * 0.0
    if not torch.isfinite(parent_probabilities).all():
        raise FloatingPointError(
            "Parent probabilities contain NaN or infinity."
        )
    if torch.any(parent_targets < 0):
        raise ValueError("Parent targets cannot contain negative IDs.")
    if torch.any(parent_targets >= parent_probabilities.shape[1]):
        raise ValueError("Parent target is outside probability dimensions.")

    correct_parent_mass = parent_probabilities.gather(
        dim=1,
        index=parent_targets.unsqueeze(1),
    ).squeeze(1)

    if not torch.isfinite(correct_parent_mass).all():
        raise FloatingPointError(
            "Correct parent probability mass is not finite."
        )

    return -torch.log(correct_parent_mass.clamp_min(eps)).mean()


def binary_coarse_hierarchy_loss(
    coarse_logits: torch.Tensor,
    binary_targets: torch.Tensor,
) -> torch.Tensor:
    if coarse_logits.ndim != 2:
        raise ValueError("coarse_logits must have shape [N, 5].")
    if coarse_logits.shape[1] != len(COARSE_NAMES):
        raise ValueError(f"Expected {len(COARSE_NAMES)} coarse logits.")
    if binary_targets.ndim != 1:
        raise ValueError("binary_targets must have shape [N].")
    if len(coarse_logits) != len(binary_targets):
        raise ValueError("Coarse logits and binary targets must align.")

    coarse_probs = coarse_logits.softmax(dim=1)
    mapping = COARSE_TO_BINARY_MATRIX.to(
        device=coarse_probs.device,
        dtype=coarse_probs.dtype,
    )
    binary_probs_from_coarse = coarse_probs @ mapping
    return negative_log_correct_parent_mass(
        binary_probs_from_coarse,
        binary_targets,
    )


def coarse_fine_hierarchy_loss(
    fine_logits: torch.Tensor,
    coarse_targets: torch.Tensor,
    fine_targets: torch.Tensor,
    fine_loss_fn: FineLongTailLoss,
) -> torch.Tensor:
    if fine_logits.ndim != 2:
        raise ValueError("fine_logits must have shape [N, 22].")
    if fine_logits.shape[1] != len(FINE_NAMES):
        raise ValueError(f"Expected {len(FINE_NAMES)} fine logits.")
    if coarse_targets.ndim != 1:
        raise ValueError("coarse_targets must have shape [N].")
    if fine_targets.ndim != 1:
        raise ValueError("fine_targets must have shape [N].")
    if not (len(fine_logits) == len(coarse_targets) == len(fine_targets)):
        raise ValueError(
            "Fine logits, coarse targets and fine targets must align."
        )

    valid = fine_targets >= 0
    if not valid.any():
        return fine_logits.sum() * 0.0

    valid_coarse_targets = coarse_targets[valid]
    normal_id = COARSE_TO_ID["Normal mucosa"]
    if torch.any(valid_coarse_targets == normal_id):
        raise ValueError(
            "A sample with a valid fine target cannot have "
            "Normal mucosa as its coarse target."
        )

    masked_fine_logits = fine_loss_fn.mask_logits(fine_logits[valid])
    fine_probs = masked_fine_logits.softmax(dim=1)
    mapping = FINE_TO_COARSE_MATRIX.to(
        device=fine_probs.device,
        dtype=fine_probs.dtype,
    )
    coarse_probs_from_fine = fine_probs @ mapping
    return negative_log_correct_parent_mass(
        coarse_probs_from_fine,
        valid_coarse_targets,
    )
