"""Decoupled Two-Stage Hierarchical Model for Long-Tailed Cystoscopy Classification.

Implements the Decoupled Representation & Classifier Alignment Architecture:
  - Phase 1: Full-network representation learning with Cross-Entropy + SupCon.
  - Phase 2: Frozen-backbone classifier re-alignment with Smoothed Balanced Softmax / cRT.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from cystods.models.factory import active_tasks_from_config, resolve_model_name
from cystods.taxonomy import BINARY_NAMES, COARSE_NAMES, FINE_NAMES


class TwoStageDecoupledHierarchicalModel(nn.Module):
    """Unified Hierarchical Model supporting Decoupled Two-Stage Fine-Tuning.

    Phase 1 (Representation Learning):
      - 100% of backbone encoder and classification heads are trainable.
      - Optimizes unified latent representations using instance-balanced Cross-Entropy + SupCon.

    Phase 2 (Classifier Alignment):
      - 100% of backbone encoder is frozen (requires_grad = False).
      - Only classification heads (binary, coarse, fine) are trained under long-tail objectives
        (Smoothed Balanced Softmax, Class-Balanced Sampling, and Tau-Normalization).
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        task_mode = str(config.get("task_mode", "hierarchical"))
        if task_mode in {"hierarchical", "multitask"}:
            active_tasks = {"binary", "coarse", "fine"}
        elif task_mode == "binary":
            active_tasks = {"binary"}
        elif task_mode == "coarse":
            active_tasks = {"coarse"}
        elif task_mode == "fine":
            active_tasks = {"fine"}
        else:
            active_tasks = set(active_tasks_from_config(config))

        if not active_tasks:
            raise ValueError("Model configuration activates no prediction head.")

        self.task_mode = task_mode
        self.active_tasks = frozenset(active_tasks)

        raw_model_name = str(config.get("model_name", "swin_tiny_patch4_window7_224.ms_in1k"))
        model_name = resolve_model_name(raw_model_name)
        if not timm.is_model(model_name):
            raise ValueError(
                f"timm model '{model_name}' (resolved from '{raw_model_name}') is unavailable in "
                f"timm {timm.__version__}."
            )

        pretrained = bool(config.get("pretrained", True))
        image_size = int(config.get("image_size", 224))
        dropout = float(config.get("dropout", 0.20))
        projection_dim = int(config.get("projection_dim", 128))

        encoder_kwargs: dict[str, Any] = {
            "pretrained": pretrained,
            "num_classes": 0,
            "global_pool": "avg",
            "img_size": image_size,
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
                f"pretrained={pretrained}. No random-weight fallback is permitted."
            ) from exc

        feature_dim = int(getattr(self.encoder, "num_features", 0))
        if feature_dim <= 0:
            raise RuntimeError(
                f"Encoder '{model_name}' does not expose a valid num_features."
            )
        self.feature_dim = feature_dim
        self.dropout = nn.Dropout(dropout)
        self.raw_model_name = raw_model_name
        self.model_name = model_name

        # Classification Heads (all attached to the full feature dimension)
        self.binary_head = (
            nn.Linear(feature_dim, len(BINARY_NAMES))
            if "binary" in self.active_tasks
            else None
        )
        self.coarse_head = (
            nn.Linear(feature_dim, len(COARSE_NAMES))
            if "coarse" in self.active_tasks
            else None
        )
        self.fine_head = (
            nn.Linear(feature_dim, len(FINE_NAMES))
            if "fine" in self.active_tasks
            else None
        )

        # Supervised Contrastive Projection Head
        supcon_weight = float(config.get("supervised_contrastive_loss_weight", 0.10))
        if supcon_weight > 0:
            self.projection_head = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, projection_dim),
            )
        else:
            self.projection_head = None

        self.phase = 1  # Phase 1: Representation, Phase 2: Classifier Alignment
        self.backbone_frozen = False

        # Check if partial finetuning / layer freezing is requested
        if bool(config.get("partial_finetune") or config.get("freeze_early_layers", False)):
            frozen_stages = int(config.get("frozen_stages_count", 2))
            if frozen_stages > 0:
                self.freeze_early_layers(stages_to_freeze=frozen_stages)

    def freeze_backbone(self) -> dict[str, Any]:
        """Freeze 100% of the backbone encoder for Phase 2 Classifier Alignment.

        Only binary_head, coarse_head, and fine_head remain trainable.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Ensure all classification heads are trainable
        for head in (self.binary_head, self.coarse_head, self.fine_head):
            if head is not None:
                for param in head.parameters():
                    param.requires_grad = True

        if self.projection_head is not None:
            for param in self.projection_head.parameters():
                param.requires_grad = False

        self.backbone_frozen = True
        self.phase = 2

        return self.get_parameter_summary()

    def unfreeze_backbone(self) -> dict[str, Any]:
        """Unfreeze the backbone encoder (return to Phase 1 state)."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        for head in (self.binary_head, self.coarse_head, self.fine_head, self.projection_head):
            if head is not None:
                for param in head.parameters():
                    param.requires_grad = True

        self.backbone_frozen = False
        self.phase = 1

        return self.get_parameter_summary()

    def freeze_early_layers(self, stages_to_freeze: int = 2) -> dict[str, Any]:
        """Freeze patch embedding + early stages (partial fine-tuning)."""
        frozen_modules: list[str] = []
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

        for head in (self.binary_head, self.coarse_head, self.fine_head, self.projection_head):
            if head is not None:
                for param in head.parameters():
                    param.requires_grad = True

        return self.get_parameter_summary()

    def tau_normalize_classifiers(self, tau: float = 1.0) -> None:
        """Apply Tau-Normalization to classifier weight vectors (ICLR 2020).

        Rescales W_k <- W_k / (||W_k||^tau) to balance decision boundaries between
        frequent and rare classes.
        """
        with torch.no_grad():
            for head in (self.binary_head, self.coarse_head, self.fine_head):
                if head is not None and hasattr(head, "weight"):
                    weight = head.weight.data
                    norms = torch.norm(weight, p=2, dim=1, keepdim=True).clamp(min=1e-8)
                    scaled_weight = weight / (norms ** float(tau))
                    head.weight.copy_(scaled_weight)

    def get_parameter_summary(self) -> dict[str, Any]:
        """Return parameter breakdown and phase status."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        return {
            "phase": self.phase,
            "backbone_frozen": self.backbone_frozen,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "trainable_percentage": (trainable_params / total_params * 100) if total_params else 0.0,
        }

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Extract global feature representations from the encoder."""
        if self.backbone_frozen:
            with torch.no_grad():
                features = self.encoder(images)
        else:
            features = self.encoder(images)

        if isinstance(features, (tuple, list)):
            raise TypeError("Encoder returned multiple feature tensors.")
        if features.ndim > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise RuntimeError(
                f"Unexpected encoder shape {tuple(features.shape)}; expected [batch, {self.feature_dim}]."
            )
        return features

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through encoder and all hierarchical prediction heads."""
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


# Aliases for backward compatibility
MultiStageHierarchicalSwinModel = TwoStageDecoupledHierarchicalModel
StageDecoupledHierarchicalModel = TwoStageDecoupledHierarchicalModel
