"""Hierarchical multi-head cystoscopy classifier.

Extracted from ``cystods.core`` (Step 2 refactor).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import timm
import torch
import torch.nn.functional as F
from torch import nn

from cystods.models.factory import active_tasks_from_config, resolve_model_name
from cystods.taxonomy import BINARY_NAMES, COARSE_NAMES, FINE_NAMES


class HierarchicalCystoModel(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        task_mode = str(config["task_mode"])
        if task_mode == "binary":
            active_tasks = {"binary"}
        elif task_mode == "coarse":
            active_tasks = {"coarse"}
        else:
            active_tasks = active_tasks_from_config(config)
        if not active_tasks:
            raise ValueError("Model configuration activates no prediction head.")
        self.task_mode = task_mode
        self.active_tasks = frozenset(active_tasks)
        raw_model_name = str(config["model_name"])
        model_name = resolve_model_name(raw_model_name)
        if not timm.is_model(model_name):
            raise ValueError(
                f"timm model '{model_name}' (resolved from '{raw_model_name}') is unavailable in "
                f"timm {timm.__version__}."
            )
        encoder_kwargs: dict[str, Any] = {
            "pretrained": bool(config["pretrained"]),
            "num_classes": 0,
            "global_pool": "avg",
            "img_size": int(config["image_size"]),
        }
        try:
            try:
                self.encoder = timm.create_model(model_name, **encoder_kwargs)
            except TypeError as err:
                if "img_size" in str(err):
                    encoder_kwargs.pop("img_size")
                    self.encoder = timm.create_model(model_name, **encoder_kwargs)
                else:
                    raise err
        except Exception as exc:
            raise TypeError(
                f"Failed to construct encoder '{model_name}' with "
                f"pretrained={config['pretrained']}. No random-weight "
                "fallback is permitted."
            ) from exc
        feature_dim = int(getattr(self.encoder, "num_features", 0))
        if feature_dim <= 0:
            raise RuntimeError(
                f"Encoder '{model_name}' does not expose a valid num_features."
            )
        self.feature_dim = feature_dim
        dropout = float(config["dropout"])
        self.dropout = nn.Dropout(dropout)
        self.binary_head = (
            nn.Linear(feature_dim, len(BINARY_NAMES))
            if "binary" in active_tasks
            else None
        )
        self.coarse_head = (
            nn.Linear(feature_dim, len(COARSE_NAMES))
            if "coarse" in active_tasks
            else None
        )
        self.fine_head = (
            nn.Linear(feature_dim, len(FINE_NAMES))
            if "fine" in active_tasks
            else None
        )
        projection_dim = int(config["projection_dim"])
        self.projection_head = (
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, projection_dim),
            )
            if float(config["supervised_contrastive_loss_weight"]) > 0
            else None
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        if isinstance(features, (tuple, list)):
            raise TypeError(
                "Encoder returned multiple feature tensors; select a timm "
                "classifier model with num_classes=0 support."
            )
        if features.ndim > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise RuntimeError(
                f"Unexpected encoder shape {tuple(features.shape)}; "
                f"expected [batch, {self.feature_dim}]."
            )
        return features

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encode(images)
        dropped = self.dropout(features)
        outputs: dict[str, torch.Tensor] = {
            "features": features,
        }
        if self.projection_head is not None:
            outputs["projection"] = F.normalize(
                self.projection_head(dropped), dim=1
            )
        if self.binary_head is not None:
            outputs["binary_logits"] = self.binary_head(dropped)
        if self.coarse_head is not None:
            outputs["coarse_logits"] = self.coarse_head(dropped)
        if self.fine_head is not None:
            outputs["fine_logits"] = self.fine_head(dropped)
        return outputs
