"""Checkpoint saving and resume loader.

Extracted from ``cystods.core`` (Step 6 refactor).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from cystods.infra.serialization import json_ready
from cystods.losses.classification import FineLongTailLoss
from cystods.taxonomy import BINARY_NAMES, COARSE_NAMES, FINE_NAMES, FINE_PARENT_ID


def _hf_checkpoint_config(
    config: Mapping[str, Any],
    path_in_repo: str,
) -> dict[str, Any]:
    repo_id = config.get("hf_repo_id")
    if not repo_id:
        raise ValueError(
            "checkpoint_backend=huggingface requires a non-empty hf_repo_id."
        )
    return {
        "repo_id": str(repo_id),
        "path_in_repo": str(path_in_repo),
        "revision": str(config.get("hf_revision", "main")),
        "private": bool(config.get("hf_private", True)),
        "create_repo": bool(config.get("hf_create_repo", True)),
        "token_env": str(config.get("hf_token_env", "HF_TOKEN")),
    }


def _hf_checkpoint_path_in_repo(
    config: Mapping[str, Any],
    fold_name: str,
) -> str:
    prefix = str(config["hf_path_prefix"]).strip("/")
    return f"{prefix}/{fold_name}/best_model.pt"


def _canonical_model_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    canonical: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        canonical_key = key
        if canonical_key.startswith("_orig_mod."):
            canonical_key = canonical_key[len("_orig_mod.") :]
        canonical[canonical_key] = tensor.detach().cpu().clone()
    return canonical


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_metric: float,
    config: Mapping[str, Any],
    fine_counts: Sequence[int],
    fine_patient_counts: Sequence[int],
    fine_loss_fn: FineLongTailLoss | None,
    fine_inference_prior_tau: float,
    data_split_fingerprint: str,
    include_optimizer_state: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "config": json_ready(config),
        "binary_names": BINARY_NAMES,
        "coarse_names": COARSE_NAMES,
        "fine_names": FINE_NAMES,
        "fine_parent_id": FINE_PARENT_ID,
        "fine_train_counts": list(map(int, fine_counts)),
        "fine_train_patient_counts": list(map(int, fine_patient_counts)),
        "fine_prior_audit": (
            fine_loss_fn.prior_audit()
            if fine_loss_fn is not None
            else None
        ),
        "fine_inference_prior_tau": float(fine_inference_prior_tau),
        "data_split_fingerprint": data_split_fingerprint,
    }
    if include_optimizer_state:
        payload.update(
            {
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            }
        )
    torch.save(payload, path)


def load_checkpoint_for_resume(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: Mapping[str, Any],
    expected_split_fingerprint: str,
) -> tuple[int, float, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required = {
        "epoch",
        "best_metric",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "config",
        "binary_names",
        "coarse_names",
        "fine_names",
        "fine_inference_prior_tau",
        "data_split_fingerprint",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Resume checkpoint missing keys: {sorted(missing)}")
    if tuple(checkpoint["binary_names"]) != BINARY_NAMES:
        raise ValueError("Resume checkpoint binary taxonomy mismatch.")
    if tuple(checkpoint["coarse_names"]) != COARSE_NAMES:
        raise ValueError("Resume checkpoint coarse taxonomy mismatch.")
    if tuple(checkpoint["fine_names"]) != FINE_NAMES:
        raise ValueError("Resume checkpoint fine taxonomy mismatch.")
    if checkpoint["data_split_fingerprint"] != expected_split_fingerprint:
        raise ValueError("Resume checkpoint data split fingerprint mismatch.")
    saved_config = checkpoint["config"]
    for key in ("model_name", "image_size", "fine_loss"):
        if saved_config.get(key) != json_ready(config[key]):
            raise ValueError(
                f"Resume checkpoint config mismatch for '{key}': "
                f"saved={saved_config.get(key)!r}, current={config[key]!r}"
            )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint["best_metric"]),
        float(checkpoint["fine_inference_prior_tau"]),
    )
