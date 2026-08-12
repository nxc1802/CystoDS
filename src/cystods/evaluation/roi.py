"""ROI-level aggregation, attention MIL, and evaluation.

Extracted from ``cystods.core`` (Step 5 refactor).
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from cystods.evaluation.metrics import (
    compute_binary_metrics,
    compute_multiclass_metrics,
)
from cystods.infra.environment import is_missing_token
from cystods.taxonomy import BINARY_NAMES, COARSE_NAMES, FINE_NAMES


@dataclass
class RoiBag:
    roi_id: str
    pid: str
    target: int
    probabilities: np.ndarray
    features: np.ndarray | None
    filenames: tuple[str, ...]


def extract_roi_bags(
    predictions: pd.DataFrame,
    task: str,
    require_features: bool,
) -> tuple[list[RoiBag], list[dict[str, Any]], int]:
    if task not in {"binary", "coarse", "fine"}:
        raise ValueError(f"Unknown ROI task: {task}")
    target_column = f"{task}_id"
    probability_column = f"{task}_probs"
    if (
        target_column not in predictions.columns
        or probability_column not in predictions.columns
    ):
        return [], [], 0
    valid_metadata = ~(
        predictions["visit"].map(is_missing_token)
        | predictions["lesion"].map(is_missing_token)
    )
    eligible = predictions.loc[valid_metadata].copy()
    skipped_missing_metadata = int((~valid_metadata).sum())
    if eligible.empty:
        return [], [], skipped_missing_metadata
    eligible["roi_id"] = (
        eligible["pid"].astype(str)
        + "::v"
        + eligible["visit"].astype(str)
        + "::"
        + eligible["lesion"].astype(str)
    )
    bags: list[RoiBag] = []
    conflicts: list[dict[str, Any]] = []
    for roi_id, group in eligible.groupby("roi_id", sort=True):
        targets = sorted(set(group[target_column].astype(int)))
        if task == "fine" and targets == [-1]:
            continue
        if task == "fine" and -1 in targets:
            conflicts.append(
                {
                    "roi_id": roi_id,
                    "pid": str(group["pid"].iloc[0]),
                    "task": task,
                    "reason": "mixed_missing_and_fine_labels",
                    "targets": targets,
                    "filenames": group["filename"].tolist(),
                }
            )
            continue
        if len(targets) != 1:
            conflicts.append(
                {
                    "roi_id": roi_id,
                    "pid": str(group["pid"].iloc[0]),
                    "task": task,
                    "reason": "conflicting_targets",
                    "targets": targets,
                    "filenames": group["filename"].tolist(),
                }
            )
            continue
        probability_matrix = np.stack(
            group[probability_column].to_numpy()
        ).astype(np.float32)
        feature_matrix: np.ndarray | None = None
        if require_features:
            if "features" not in group:
                raise ValueError(
                    "Attention ROI evaluation requires exported features."
                )
            feature_matrix = np.stack(group["features"].to_numpy()).astype(
                np.float32
            )
        bags.append(
            RoiBag(
                roi_id=str(roi_id),
                pid=str(group["pid"].iloc[0]),
                target=int(targets[0]),
                probabilities=probability_matrix,
                features=feature_matrix,
                filenames=tuple(group["filename"].astype(str)),
            )
        )
    return bags, conflicts, skipped_missing_metadata


def aggregate_roi_bags(
    bags: Sequence[RoiBag],
    method: str,
    num_classes: int,
) -> pd.DataFrame:
    if method not in {"mean", "vote"}:
        raise ValueError("Non-attention ROI method must be mean or vote.")
    rows: list[dict[str, Any]] = []
    for bag in bags:
        if method == "mean":
            probability = bag.probabilities.mean(axis=0)
        else:
            votes = np.bincount(
                bag.probabilities.argmax(axis=1),
                minlength=num_classes,
            )
            probability = votes.astype(np.float64) / votes.sum()
        probability = probability / probability.sum()
        rows.append(
            {
                "roi_id": bag.roi_id,
                "pid": bag.pid,
                "target": bag.target,
                "probabilities": probability,
                "images": len(bag.filenames),
                "filenames": "|".join(bag.filenames),
            }
        )
    return pd.DataFrame(rows)


def compute_roi_task_metrics(
    aggregated: pd.DataFrame,
    task: str,
    binary_threshold: float,
) -> dict[str, Any]:
    if aggregated.empty:
        return {
            "status": "not_evaluable",
            "reason": "no label-consistent ROI groups",
            "n_rois": 0,
        }
    targets = aggregated["target"].to_numpy(dtype=np.int64)
    probabilities = np.stack(aggregated["probabilities"].to_numpy())
    if task == "binary":
        metrics = compute_binary_metrics(
            targets, probabilities[:, 1], binary_threshold
        )
    elif task == "coarse":
        metrics = compute_multiclass_metrics(
            targets, probabilities, COARSE_NAMES
        )
    elif task == "fine":
        metrics = compute_multiclass_metrics(
            targets, probabilities, FINE_NAMES
        )
    else:
        raise ValueError(f"Unknown ROI task: {task}")
    return {"status": "ok", "n_rois": len(aggregated), **metrics}


class GatedAttentionMIL(nn.Module):
    def __init__(
        self, feature_dim: int, hidden_dim: int, num_classes: int
    ) -> None:
        super().__init__()
        self.attention_v = nn.Linear(feature_dim, hidden_dim)
        self.attention_u = nn.Linear(feature_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(
        self, bag_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if bag_features.ndim != 2:
            raise ValueError("MIL bag features must have shape [instances, D].")
        gated = torch.tanh(self.attention_v(bag_features)) * torch.sigmoid(
            self.attention_u(bag_features)
        )
        weights = torch.softmax(
            self.attention_w(gated).squeeze(-1), dim=0
        )
        pooled = torch.sum(weights[:, None] * bag_features, dim=0)
        logits = self.classifier(pooled)
        return logits, weights


def predict_attention_bags(
    model: GatedAttentionMIL,
    bags: Sequence[RoiBag],
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for bag in bags:
            if bag.features is None:
                raise ValueError("Attention bag has no feature matrix.")
            features = torch.as_tensor(
                bag.features, dtype=torch.float32, device=device
            )
            logits, attention = model(features)
            probability = logits.softmax(dim=0).float().cpu().numpy()
            rows.append(
                {
                    "roi_id": bag.roi_id,
                    "pid": bag.pid,
                    "target": bag.target,
                    "probabilities": probability,
                    "images": len(bag.filenames),
                    "filenames": "|".join(bag.filenames),
                    "attention_weights": attention.float().cpu().numpy(),
                }
            )
    return pd.DataFrame(rows)


def train_attention_mil(
    train_bags: Sequence[RoiBag],
    val_bags: Sequence[RoiBag],
    test_bags: Sequence[RoiBag],
    task: str,
    config: Mapping[str, Any],
    device: torch.device,
    output_path: Path,
    logger: logging.Logger,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if not train_bags or not val_bags or not test_bags:
        raise ValueError(
            f"Attention MIL for {task} requires non-empty train/val/test bags."
        )
    feature_dims = {
        bag.features.shape[1]
        for bag in (*train_bags, *val_bags, *test_bags)
        if bag.features is not None
    }
    if len(feature_dims) != 1:
        raise ValueError(
            f"ROI feature dimensions are inconsistent: {sorted(feature_dims)}"
        )
    feature_dim = next(iter(feature_dims))
    num_classes = {
        "binary": len(BINARY_NAMES),
        "coarse": len(COARSE_NAMES),
        "fine": len(FINE_NAMES),
    }[task]
    model = GatedAttentionMIL(
        feature_dim,
        int(config["roi_attention_hidden_dim"]),
        num_classes,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["roi_attention_learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    rng = random.Random(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(int(config["roi_attention_epochs"])):
        model.train()
        order = list(range(len(train_bags)))
        rng.shuffle(order)
        loss_sum = 0.0
        for bag_index in order:
            bag = train_bags[bag_index]
            if bag.features is None:
                raise ValueError("Training attention bag has no features.")
            features = torch.as_tensor(
                bag.features, dtype=torch.float32, device=device
            )
            target = torch.tensor(
                [bag.target], dtype=torch.long, device=device
            )
            logits, _ = model(features)
            loss = F.cross_entropy(logits.unsqueeze(0), target)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite attention MIL loss for task={task}."
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config["gradient_clip_norm"]),
                error_if_nonfinite=True,
            )
            optimizer.step()
            loss_sum += float(loss.detach())
        val_predictions = predict_attention_bags(model, val_bags, device)
        val_metrics = compute_roi_task_metrics(
            val_predictions,
            task,
            float(config["binary_decision_threshold"]),
        )
        if val_metrics["status"] != "ok":
            raise RuntimeError(
                f"Attention MIL validation for {task} is not evaluable."
            )
        score = (
            val_metrics["f1"]
            if task == "binary"
            else val_metrics["macro_f1"]
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": loss_sum / len(train_bags),
                "val_score": score,
            }
        )
        logger.info(
            "roi-attention task=%s epoch=%d/%d loss=%.5f val_score=%.5f",
            task,
            epoch + 1,
            config["roi_attention_epochs"],
            loss_sum / len(train_bags),
            score,
        )
        if score > best_score:
            best_score = float(score)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= int(config["roi_attention_early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("Attention MIL did not produce a best state.")
    model.load_state_dict(best_state, strict=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "task": task,
            "feature_dim": feature_dim,
            "hidden_dim": int(config["roi_attention_hidden_dim"]),
            "num_classes": num_classes,
            "state_dict": best_state,
            "history": history,
            "best_validation_score": best_score,
        },
        output_path,
    )
    val_predictions = predict_attention_bags(model, val_bags, device)
    test_predictions = predict_attention_bags(model, test_bags, device)
    metrics = compute_roi_task_metrics(
        test_predictions,
        task,
        float(config["binary_decision_threshold"]),
    )
    metrics["best_validation_score"] = best_score
    metrics["training_history"] = history
    return metrics, val_predictions, test_predictions


def serialize_roi_predictions(frame: pd.DataFrame, path: Path) -> None:
    output = frame.copy()
    if output.empty:
        output.to_csv(path, index=False)
        return
    output["probabilities"] = output["probabilities"].map(
        lambda value: json.dumps(
            np.asarray(value, dtype=float).tolist(), separators=(",", ":")
        )
    )
    if "attention_weights" in output:
        output["attention_weights"] = output["attention_weights"].map(
            lambda value: json.dumps(
                np.asarray(value, dtype=float).tolist(),
                separators=(",", ":"),
            )
        )
    output.to_csv(path, index=False)


def run_roi_evaluation(
    split_predictions: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
    device: torch.device,
    run_dir: Path,
    fold_name: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    roi_dir = run_dir / "predictions" / fold_name / "roi"
    roi_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {}
    conflict_rows: list[dict[str, Any]] = []
    bag_cache: dict[str, dict[str, list[RoiBag]]] = {}
    skipped_metadata: dict[str, dict[str, int]] = {}
    for task in ("binary", "coarse", "fine"):
        bag_cache[task] = {}
        skipped_metadata[task] = {}
        for split_name, predictions in split_predictions.items():
            bags, conflicts, skipped = extract_roi_bags(
                predictions,
                task,
                require_features="attention" in config["roi_aggregations"],
            )
            bag_cache[task][split_name] = bags
            skipped_metadata[task][split_name] = skipped
            for conflict in conflicts:
                conflict["split"] = split_name
            conflict_rows.extend(conflicts)

    conflict_frame = pd.DataFrame(conflict_rows)
    conflict_frame.to_csv(roi_dir / "conflicts.csv", index=False)
    if conflict_rows and config["roi_conflict_policy"] == "raise":
        raise ValueError(
            f"ROI evaluation found {len(conflict_rows)} task-level target "
            f"conflicts; see {roi_dir / 'conflicts.csv'}"
        )
    logger.info(
        "ROI audit: task-level conflicts=%d policy=%s",
        len(conflict_rows),
        config["roi_conflict_policy"],
    )

    for method in config["roi_aggregations"]:
        method_metrics: dict[str, Any] = {}
        for task in ("binary", "coarse", "fine"):
            if method in {"mean", "vote"}:
                num_classes = {
                    "binary": len(BINARY_NAMES),
                    "coarse": len(COARSE_NAMES),
                    "fine": len(FINE_NAMES),
                }[task]
                task_metrics: dict[str, Any] = {}
                for split_name, bags in bag_cache[task].items():
                    aggregated = aggregate_roi_bags(bags, method, num_classes)
                    serialize_roi_predictions(
                        aggregated,
                        roi_dir / f"{method}_{task}_{split_name}.csv",
                    )
                    task_metrics[split_name] = compute_roi_task_metrics(
                        aggregated,
                        task,
                        float(config["binary_decision_threshold"]),
                    )
                method_metrics[task] = task_metrics
            elif method == "attention":
                if (
                    not bag_cache[task].get("train")
                    or not bag_cache[task].get("val")
                    or not bag_cache[task].get("test")
                ):
                    method_metrics[task] = {
                        "status": "not_evaluable",
                        "reason": "missing_bags_for_task",
                    }
                    continue
                mil_metrics, val_pred, test_pred = train_attention_mil(
                    bag_cache[task]["train"],
                    bag_cache[task]["val"],
                    bag_cache[task]["test"],
                    task,
                    config,
                    device,
                    run_dir / "checkpoints" / f"mil_attention_{task}.pt",
                    logger,
                    int(config["seed"]),
                )
                serialize_roi_predictions(
                    val_pred, roi_dir / f"attention_{task}_val.csv"
                )
                serialize_roi_predictions(
                    test_pred, roi_dir / f"attention_{task}_test.csv"
                )
                method_metrics[task] = {"val": val_pred, "test": mil_metrics}
        metrics[method] = method_metrics
    return metrics
