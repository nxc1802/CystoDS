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

        # Check if partial finetuning / layer freezing is requested
        self.partial_finetune = bool(
            config.get("partial_finetune") or config.get("freeze_early_layers", False)
        )
        self.frozen_stages_count = int(config.get("frozen_stages_count", 2))
        if self.partial_finetune:
            self.freeze_early_layers(stages_to_freeze=self.frozen_stages_count)

    def freeze_early_layers(self, stages_to_freeze: int = 2) -> dict[str, Any]:
        """Freeze Patch Embedding + Stage 1 + Stage 2 (or equivalent early stages).

        Only Stage 3, Stage 4 and classification/projection heads remain trainable.
        """
        frozen_modules: list[str] = []
        # Swin Transformer
        if hasattr(self.encoder, "patch_embed") and hasattr(self.encoder, "layers"):
            for param in self.encoder.patch_embed.parameters():
                param.requires_grad = False
            frozen_modules.append("patch_embed")
            for idx in range(min(stages_to_freeze, len(self.encoder.layers))):
                for param in self.encoder.layers[idx].parameters():
                    param.requires_grad = False
                frozen_modules.append(f"layers[{idx}]")
            for idx in range(stages_to_freeze, len(self.encoder.layers)):
                for param in self.encoder.layers[idx].parameters():
                    param.requires_grad = True
            if hasattr(self.encoder, "norm") and self.encoder.norm is not None:
                for param in self.encoder.norm.parameters():
                    param.requires_grad = True
        # ResNet / ResNeXt
        elif hasattr(self.encoder, "conv1") and hasattr(self.encoder, "layer1"):
            for attr in ("conv1", "bn1"):
                if hasattr(self.encoder, attr) and getattr(self.encoder, attr) is not None:
                    for param in getattr(self.encoder, attr).parameters():
                        param.requires_grad = False
                    frozen_modules.append(attr)
            for idx in range(1, stages_to_freeze + 1):
                layer_name = f"layer{idx}"
                if hasattr(self.encoder, layer_name) and getattr(self.encoder, layer_name) is not None:
                    for param in getattr(self.encoder, layer_name).parameters():
                        param.requires_grad = False
                    frozen_modules.append(layer_name)
            for idx in range(stages_to_freeze + 1, 5):
                layer_name = f"layer{idx}"
                if hasattr(self.encoder, layer_name) and getattr(self.encoder, layer_name) is not None:
                    for param in getattr(self.encoder, layer_name).parameters():
                        param.requires_grad = True
        # ConvNeXt
        elif hasattr(self.encoder, "stem") and hasattr(self.encoder, "stages"):
            for param in self.encoder.stem.parameters():
                param.requires_grad = False
            frozen_modules.append("stem")
            for idx in range(min(stages_to_freeze, len(self.encoder.stages))):
                for param in self.encoder.stages[idx].parameters():
                    param.requires_grad = False
                frozen_modules.append(f"stages[{idx}]")
            for idx in range(stages_to_freeze, len(self.encoder.stages)):
                for param in self.encoder.stages[idx].parameters():
                    param.requires_grad = True
            if hasattr(self.encoder, "norm_pre") and self.encoder.norm_pre is not None:
                for param in self.encoder.norm_pre.parameters():
                    param.requires_grad = True
            if hasattr(self.encoder, "head") and self.encoder.head is not None:
                for param in self.encoder.head.parameters():
                    param.requires_grad = True
        else:
            # Generic fallback: freeze first half of encoder named children
            children = list(self.encoder.named_children())
            num_freeze = max(1, min(stages_to_freeze, len(children) // 2))
            for name, child in children[:num_freeze]:
                for param in child.parameters():
                    param.requires_grad = False
                frozen_modules.append(name)
            for name, child in children[num_freeze:]:
                for param in child.parameters():
                    param.requires_grad = True

        # Always ensure classification and projection heads are trainable
        for head in (self.binary_head, self.coarse_head, self.fine_head, self.projection_head):
            if head is not None:
                for param in head.parameters():
                    param.requires_grad = True

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        return {
            "partial_finetune": True,
            "frozen_modules": frozen_modules,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "trainable_percentage": (trainable_params / total_params * 100) if total_params else 0.0,
        }

    def get_parameter_summary(self) -> dict[str, Any]:
        """Return parameter counts breakdown."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "trainable_percentage": (trainable_params / total_params * 100) if total_params else 0.0,
            "partial_finetune": getattr(self, "partial_finetune", False),
        }

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
