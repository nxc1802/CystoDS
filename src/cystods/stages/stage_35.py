"""Stage 35 — Decoupled Two-Stage Hierarchical Fine-Tuning (2S-HFT).

Orchestrates Stage 35 execution via ``cystods.experiments.two_stage_runner``.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch

from cystods.experiments.two_stage_runner import run_two_stage_single_split

STAGE_ID = "35"
STAGE_NAME = "stage_35_run_two_stage_decoupled"


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Execute Stage 35 Decoupled Two-Stage Fine-Tuning with the resolved config."""
    config = dict(config)
    profile = str(config.get("run_profile", "research"))
    result_root = Path(config.get("result_root", "./result")).resolve()

    # Determine default epochs & hyperparameters
    if profile == "smoke":
        p1_epochs = int(config.get("phase1_epochs", 1))
        p2_epochs = int(config.get("phase2_epochs", 1))
        supcon_w = float(config.get("phase1_supcon_weight", 0.0))
    else:
        p1_epochs = int(config.get("phase1_epochs", 25))
        p2_epochs = int(config.get("phase2_epochs", 10))
        supcon_w = float(config.get("phase1_supcon_weight", 0.10))

    p1_loss = str(config.get("phase1_loss", "cross_entropy"))
    p2_loss = str(config.get("phase2_loss", "balanced_softmax_smoothed"))
    p1_lr = float(config.get("phase1_lr", 0.0003))
    p2_lr = float(config.get("phase2_lr", 0.001))
    p2_target = str(config.get("phase2_target", "fine_only"))
    p2_strategy = str(config.get("phase2_strategy", "linear_probe"))

    split_arg = config.get("protocol_split_index")
    if split_arg is None or str(split_arg).lower() == "all":
        split_indices = [0, 1, 2]
    elif isinstance(split_arg, (list, tuple)):
        split_indices = [int(s) for s in split_arg]
    else:
        split_indices = [int(split_arg)]

    results: list[dict[str, Any]] = []
    for s_idx in split_indices:
        res = run_two_stage_single_split(
            split_index=s_idx,
            base_config=config,
            profile=profile,
            phase1_epochs=p1_epochs,
            phase2_epochs=p2_epochs,
            phase1_loss=p1_loss,
            phase2_loss=p2_loss,
            phase2_strategy=p2_strategy,
            phase1_lr=p1_lr,
            phase2_lr=p2_lr,
            supcon_w=supcon_w,
            phase2_target=p2_target,
            result_root=result_root,
        )
        results.append(res)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "stage": STAGE_ID,
        "stage_name": STAGE_NAME,
        "profile": profile,
        "splits": split_indices,
        "results": results,
    }
