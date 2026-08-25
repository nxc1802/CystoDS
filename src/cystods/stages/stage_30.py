"""Stage 30 — Proposed Method: Three-Stage Sequential Hierarchical Fine-Tuning (3S-HFT).

Orchestrates the main proposed method of the CystoDS study:
- Phase 1: Representation Learning (Full network with CE + SupCon + Hierarchical Consistency)
- Phase 2: Selective Coarse Classifier Alignment (Frozen backbone, Coarse-only Smoothed Balanced Softmax)
- Phase 3: Selective Fine Classifier Alignment (Frozen backbone & coarse, Fine-only Smoothed Balanced Softmax)
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch

from cystods.experiments.three_stage_runner import run_three_stage_single_split

STAGE_ID = "30"
STAGE_NAME = "stage_30_run_proposed_method"


def _source_files() -> tuple[Path, ...]:
    pkg_dir = Path(__file__).resolve().parent.parent
    return tuple(
        path
        for path in (
            pkg_dir / "core.py",
            pkg_dir / "hf.py",
            pkg_dir / "science.py",
            pkg_dir / "experiments" / "three_stage_runner.py",
        )
        if path.is_file()
    )


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Execute Stage 30 (3S-HFT Proposed Method) with the resolved config."""
    config = dict(config)
    profile = str(config.get("run_profile", "research"))
    result_root = Path(config.get("result_root", "./result")).resolve()

    # Determine default epochs & hyperparameters
    if profile == "smoke":
        p1_epochs = int(config.get("phase1_epochs", 1))
        p2_epochs = int(config.get("phase2_epochs", 1))
        p3_epochs = int(config.get("phase3_epochs", 1))
        supcon_w = float(config.get("phase1_supcon_weight", 0.0))
    else:
        p1_epochs = int(config.get("phase1_epochs", 25))
        p2_epochs = int(config.get("phase2_epochs", 10))
        p3_epochs = int(config.get("phase3_epochs", 10))
        supcon_w = float(config.get("phase1_supcon_weight", 0.10))

    p1_loss = str(config.get("phase1_loss", "cross_entropy"))
    p2_loss = str(config.get("phase2_loss", "balanced_softmax_smoothed"))
    p3_loss = str(config.get("phase3_loss", "balanced_softmax_smoothed"))
    p1_lr = float(config.get("phase1_lr", 0.0003))
    p2_lr = float(config.get("phase2_lr", 0.001))
    p3_lr = float(config.get("phase3_lr", 0.001))

    split_arg = config.get("protocol_split_index")
    if split_arg is None or str(split_arg).lower() == "all":
        split_indices = [0, 1, 2]
    elif isinstance(split_arg, (list, tuple)):
        split_indices = [int(s) for s in split_arg]
    else:
        split_indices = [int(split_arg)]

    results: list[dict[str, Any]] = []
    for s_idx in split_indices:
        res = run_three_stage_single_split(
            split_index=s_idx,
            base_config=config,
            profile=profile,
            phase1_epochs=p1_epochs,
            phase2_epochs=p2_epochs,
            phase3_epochs=p3_epochs,
            phase1_loss=p1_loss,
            phase2_loss=p2_loss,
            phase3_loss=p3_loss,
            phase1_lr=p1_lr,
            phase2_lr=p2_lr,
            phase3_lr=p3_lr,
            supcon_w=supcon_w,
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
