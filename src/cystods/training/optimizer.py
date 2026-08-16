"""Optimizer and learning rate scheduler construction.

Extracted from ``cystods.core`` (Step 6 refactor).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from cystods.models.hierarchical import HierarchicalCystoModel


def build_optimizer(
    model: HierarchicalCystoModel,
    config: Mapping[str, Any],
    device: torch.device,
) -> torch.optim.Optimizer:
    if config["optimizer"] != "adamw":
        raise ValueError("Only optimizer='adamw' is implemented.")
    encoder_params = [
        parameter
        for parameter in model.encoder.parameters()
        if parameter.requires_grad
    ]
    head_params = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.") and parameter.requires_grad
    ]
    fused = bool(config["use_fused_optimizer"])
    if fused and device.type != "cuda":
        raise RuntimeError("Fused AdamW is only enabled for CUDA runs.")
    param_groups = []
    if encoder_params:
        param_groups.append({
            "params": encoder_params,
            "lr": float(config["learning_rate"])
            * float(config["encoder_learning_rate_multiplier"]),
        })
    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": float(config["learning_rate"]),
        })
    return torch.optim.AdamW(
        param_groups,
        weight_decay=float(config["weight_decay"]),
        fused=fused,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_batches: int,
    config: Mapping[str, Any],
) -> tuple[torch.optim.lr_scheduler.LambdaLR, int]:
    accumulation = int(config["gradient_accumulation_steps"])
    updates_per_epoch = math.ceil(train_batches / accumulation)
    total_updates = updates_per_epoch * int(config["scheduler_epochs"])
    warmup_updates = round(
        float(config["warmup_epochs"]) * updates_per_epoch
    )
    min_ratio = float(config["minimum_learning_rate_ratio"])
    if total_updates < 1:
        raise ValueError("Scheduler requires at least one optimizer update.")
    if not 0 <= min_ratio <= 1:
        raise ValueError("minimum_learning_rate_ratio must be in [0, 1].")

    def multiplier(step: int) -> float:
        if warmup_updates > 0 and step < warmup_updates:
            return max((step + 1) / warmup_updates, 1e-8)
        denominator = max(total_updates - warmup_updates, 1)
        progress = min(max((step - warmup_updates) / denominator, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier), total_updates
