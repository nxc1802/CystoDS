"""Composite multi-task objective.

Extracted from ``cystods.core`` (Step 2 refactor).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from cystods.losses.classification import CoarseLongTailLoss, FineLongTailLoss
from cystods.losses.hierarchy import (
    binary_coarse_hierarchy_loss,
    coarse_fine_hierarchy_loss,
)
from cystods.models.factory import active_tasks_from_config


def active_coarse_loss_name(config: Mapping[str, Any]) -> str:
    if float(config.get("coarse_loss_weight", 1.0)) == 0:
        return "cross_entropy"
    return str(config.get("coarse_loss", "cross_entropy"))


def active_fine_loss_name(config: Mapping[str, Any]) -> str:
    if float(config["fine_loss_weight"]) == 0:
        return "cross_entropy"
    return str(config["fine_loss"])


def compute_multitask_loss(
    outputs: Mapping[str, torch.Tensor],
    binary_targets: torch.Tensor,
    coarse_targets: torch.Tensor,
    fine_targets: torch.Tensor,
    fine_loss_fn: FineLongTailLoss | None,
    config: Mapping[str, Any],
    coarse_loss_fn: CoarseLongTailLoss | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    active_tasks = active_tasks_from_config(config)
    expected_output_keys = {"features"} | {
        f"{task}_logits" for task in active_tasks
    }
    optional_keys = {"projection"}
    unknown_outputs = set(outputs) - expected_output_keys - optional_keys
    missing_outputs = expected_output_keys - set(outputs)
    if missing_outputs or unknown_outputs:
        raise ValueError(
            "Model output/task contract mismatch: "
            f"missing={sorted(missing_outputs)}, "
            f"unexpected={sorted(unknown_outputs)}"
        )
    anchor = outputs[next(f"{task}_logits" for task in sorted(active_tasks))]
    total = anchor.sum() * 0.0
    components: dict[str, torch.Tensor] = {}
    if "binary" in active_tasks:
        binary_loss = F.cross_entropy(
            outputs["binary_logits"], binary_targets
        )
        components["binary_loss"] = binary_loss
        total = total + float(config["binary_loss_weight"]) * binary_loss
    if "coarse" in active_tasks:
        if coarse_loss_fn is not None:
            coarse_loss = coarse_loss_fn(
                outputs["coarse_logits"], coarse_targets
            )
        else:
            coarse_loss = F.cross_entropy(
                outputs["coarse_logits"], coarse_targets
            )
        components["coarse_loss"] = coarse_loss
        total = total + float(config["coarse_loss_weight"]) * coarse_loss
    if "fine" in active_tasks:
        if fine_loss_fn is None:
            raise ValueError("An active fine task requires FineLongTailLoss.")
        valid_fine = fine_targets >= 0
        if valid_fine.any():
            valid_positions = torch.nonzero(
                valid_fine,
                as_tuple=False,
            ).flatten()
            valid_ids = fine_targets[valid_fine]
            active_positions = fine_loss_fn.active_mask[valid_ids]
            valid_fine[valid_positions] = active_positions
        fine_loss = fine_loss_fn(
            outputs["fine_logits"][valid_fine],
            fine_targets[valid_fine],
        )
        components["fine_loss"] = fine_loss
        components["fine_loss_evaluable_share"] = (
            valid_fine.float().mean()
        )
        total = total + float(config["fine_loss_weight"]) * fine_loss
    bc_weight = float(config["binary_coarse_hierarchy_loss_weight"])
    if bc_weight > 0:
        if str(config["task_mode"]) != "hierarchical":
            raise ValueError(
                "Binary-Coarse hierarchy loss requires task_mode='hierarchical'."
            )
        if "coarse_logits" not in outputs:
            raise ValueError(
                "Binary-Coarse hierarchy loss requires coarse_logits."
            )
        bc_hierarchy_loss = binary_coarse_hierarchy_loss(
            outputs["coarse_logits"],
            binary_targets,
        )
        components["binary_coarse_hierarchy_loss"] = bc_hierarchy_loss
        total = total + bc_weight * bc_hierarchy_loss

    cf_weight = float(config["coarse_fine_hierarchy_loss_weight"])
    if cf_weight > 0:
        if str(config["task_mode"]) != "hierarchical":
            raise ValueError(
                "Coarse-Fine hierarchy loss requires task_mode='hierarchical'."
            )
        if "fine_logits" not in outputs:
            raise ValueError(
                "Coarse-Fine hierarchy loss requires fine_logits."
            )
        if fine_loss_fn is None:
            raise ValueError(
                "Coarse-Fine hierarchy loss requires FineLongTailLoss."
            )
        cf_hierarchy_loss = coarse_fine_hierarchy_loss(
            outputs["fine_logits"],
            coarse_targets,
            fine_targets,
            fine_loss_fn,
        )
        components["coarse_fine_hierarchy_loss"] = cf_hierarchy_loss
        total = total + cf_weight * cf_hierarchy_loss
    if not torch.isfinite(total):
        raise FloatingPointError("Classification loss is not finite.")
    components["classification_total"] = total
    return total, components
