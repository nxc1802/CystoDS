"""Multi-Stage Decoupled Hierarchical Classifier for Cystoscopy Images.

Implements the Depth-Semantic Alignment architecture proposed for CystoDS:
  - Stage 2 (H/8 x W/8, 192 dims)  --> GAP + LayerNorm(192) --> Linear(192, 2)  --> Binary ROI Head
  - Stage 3 (H/16 x W/16, 384 dims) --> GAP + LayerNorm(384) --> Linear(384, 5)  --> Coarse Head (5 Groups)
  - Stage 4 (H/32 x W/32, 768 dims) --> GAP + LayerNorm(768) --> Linear(768, 22) --> Fine Head (22 Subclasses)
                                                              --> MLP Projector  --> SupCon Embedding (128d)

Backbone Support:
  - Primary: Swin Transformer (`swin_tiny_patch4_window7_224.ms_in1k`, `swin_small`, `swin_base`)
  - Fallback: ResNet / ResNeXt (`layer1`, `layer2`, `layer3`, `layer4`) and ConvNeXt (`stages[0..3]`)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from cystods.models.factory import active_tasks_from_config, resolve_model_name
from cystods.taxonomy import BINARY_NAMES, COARSE_NAMES, FINE_NAMES


class MultiStageHierarchicalSwinModel(nn.Module):
    """Swin Transformer with Stage-Decoupled Hierarchical Heads.

    Separates the prediction heads across different encoder stages to prevent
    gradient interference and align semantic abstraction with network depth:
      - Binary ROI Head: attached to Stage 2 (retains vascular & mucosal patterns)
      - Coarse Head (5 groups): attached to Stage 3 (balanced regional context)
      - Fine Head (22 subclasses) & SupCon Head: attached to Stage 4 (deep semantic abstraction)
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

        self.dropout = nn.Dropout(dropout)
        self.raw_model_name = raw_model_name
        self.model_name = model_name

        # Detect encoder backbone type and intermediate feature dimensions
        self.backbone_type = self._detect_backbone_type()
        dims = self._probe_stage_dimensions(image_size)
        self.s2_dim = dims["s2_dim"]
        self.s3_dim = dims["s3_dim"]
        self.s4_dim = dims["s4_dim"]
        self.feature_dim = self.s4_dim

        # ── Stage 2: Binary Head (ROI vs Non-ROI) ───────────────────────────
        if "binary" in self.active_tasks:
            self.binary_norm = nn.LayerNorm(self.s2_dim)
            self.binary_head = nn.Linear(self.s2_dim, len(BINARY_NAMES))
        else:
            self.binary_norm = None
            self.binary_head = None

        # ── Stage 3: Coarse Head (5 Clinical Groups) ────────────────────────
        if "coarse" in self.active_tasks:
            self.coarse_norm = nn.LayerNorm(self.s3_dim)
            self.coarse_head = nn.Linear(self.s3_dim, len(COARSE_NAMES))
        else:
            self.coarse_norm = None
            self.coarse_head = None

        # ── Stage 4: Fine Head (22 Histopathology Subclasses) ───────────────
        if "fine" in self.active_tasks:
            self.fine_norm = nn.LayerNorm(self.s4_dim)
            self.fine_head = nn.Linear(self.s4_dim, len(FINE_NAMES))
        else:
            self.fine_norm = None
            self.fine_head = None

        # ── Stage 4: Supervised Contrastive Projection Head ─────────────────
        supcon_weight = float(config.get("supervised_contrastive_loss_weight", 0.10))
        if supcon_weight > 0:
            self.projection_head = nn.Sequential(
                nn.Linear(self.s4_dim, self.s4_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.s4_dim, projection_dim),
            )
        else:
            self.projection_head = None

        # Check if partial finetuning / layer freezing is requested
        self.partial_finetune = bool(
            config.get("partial_finetune") or config.get("freeze_early_layers", False)
        )
        self.frozen_stages_count = int(config.get("frozen_stages_count", 0))
        if self.partial_finetune and self.frozen_stages_count > 0:
            self.freeze_early_layers(stages_to_freeze=self.frozen_stages_count)

    def _detect_backbone_type(self) -> str:
        if hasattr(self.encoder, "patch_embed") and hasattr(self.encoder, "layers"):
            return "swin"
        if hasattr(self.encoder, "conv1") and hasattr(self.encoder, "layer1"):
            return "resnet"
        if hasattr(self.encoder, "stem") and hasattr(self.encoder, "stages"):
            return "convnext"
        return "generic"

    def _probe_stage_dimensions(self, image_size: int) -> dict[str, int]:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            if self.backbone_type == "swin":
                x = self.encoder.patch_embed(dummy)
                x_s1 = self.encoder.layers[0](x)
                x_s2 = self.encoder.layers[1](x_s1)
                x_s3 = self.encoder.layers[2](x_s2)
                x_s4 = self.encoder.layers[3](x_s3)
                s2_dim = int(x_s2.shape[-1])
                s3_dim = int(x_s3.shape[-1])
                s4_dim = int(x_s4.shape[-1])
            elif self.backbone_type == "resnet":
                x = self.encoder.conv1(dummy)
                x = self.encoder.bn1(x) if hasattr(self.encoder, "bn1") else x
                x = self.encoder.act1(x) if hasattr(self.encoder, "act1") else F.relu(x)
                x = self.encoder.maxpool(x) if hasattr(self.encoder, "maxpool") else x
                x_s1 = self.encoder.layer1(x)
                x_s2 = self.encoder.layer2(x_s1)
                x_s3 = self.encoder.layer3(x_s2)
                x_s4 = self.encoder.layer4(x_s3)
                s2_dim = int(x_s2.shape[1])
                s3_dim = int(x_s3.shape[1])
                s4_dim = int(x_s4.shape[1])
            elif self.backbone_type == "convnext":
                x = self.encoder.stem(dummy)
                x_s1 = self.encoder.stages[0](x)
                x_s2 = self.encoder.stages[1](x_s1)
                x_s3 = self.encoder.stages[2](x_s2)
                x_s4 = self.encoder.stages[3](x_s3)
                s2_dim = int(x_s2.shape[1])
                s3_dim = int(x_s3.shape[1])
                s4_dim = int(x_s4.shape[1])
            else:
                feat_dim = int(getattr(self.encoder, "num_features", 768))
                s2_dim = feat_dim // 4 if feat_dim >= 4 else feat_dim
                s3_dim = feat_dim // 2 if feat_dim >= 2 else feat_dim
                s4_dim = feat_dim

        return {"s2_dim": s2_dim, "s3_dim": s3_dim, "s4_dim": s4_dim}

    def freeze_early_layers(self, stages_to_freeze: int = 2) -> dict[str, Any]:
        """Freeze patch embedding / stem and the first N stages of the encoder.

        Args:
            stages_to_freeze: Number of stages to freeze (e.g. 2 for Stages 1-2, 3 for Stages 1-3).
        """
        frozen_modules: list[str] = []
        if self.backbone_type == "swin":
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
        elif self.backbone_type == "resnet":
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
        elif self.backbone_type == "convnext":
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

        # Always keep all prediction and projection heads trainable
        for head in (
            self.binary_norm,
            self.binary_head,
            self.coarse_norm,
            self.coarse_head,
            self.fine_norm,
            self.fine_head,
            self.projection_head,
        ):
            if head is not None:
                for param in head.parameters():
                    param.requires_grad = True

        self.partial_finetune = True
        self.frozen_stages_count = stages_to_freeze

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
        """Return parameter counts and freezing status."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "trainable_percentage": (trainable_params / total_params * 100) if total_params else 0.0,
            "partial_finetune": getattr(self, "partial_finetune", False),
            "frozen_stages_count": getattr(self, "frozen_stages_count", 0),
        }

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Multi-stage forward pass extracting logits at respective depths."""
        outputs: dict[str, torch.Tensor] = {}

        if self.backbone_type == "swin":
            # Stage 1: Initial patch embedding and low-level feature extraction
            x = self.encoder.patch_embed(images)
            x_s1 = self.encoder.layers[0](x)

            # Stage 2: Vascular & mucosal structure --> Binary Head
            x_s2 = self.encoder.layers[1](x_s1)
            feat_s2 = x_s2.mean(dim=(1, 2)) if x_s2.ndim == 4 else x_s2
            if self.binary_head is not None:
                normed_s2 = self.binary_norm(feat_s2) if self.binary_norm is not None else feat_s2
                outputs["binary_logits"] = self.binary_head(self.dropout(normed_s2))

            # Stage 3: Regional lesion context --> Coarse Head
            x_s3 = self.encoder.layers[2](x_s2)
            feat_s3 = x_s3.mean(dim=(1, 2)) if x_s3.ndim == 4 else x_s3
            if self.coarse_head is not None:
                normed_s3 = self.coarse_norm(feat_s3) if self.coarse_norm is not None else feat_s3
                outputs["coarse_logits"] = self.coarse_head(self.dropout(normed_s3))

            # Stage 4: Deep histopathology semantics --> Fine Head & SupCon
            x_s4 = self.encoder.layers[3](x_s3)
            if hasattr(self.encoder, "norm") and self.encoder.norm is not None:
                x_s4 = self.encoder.norm(x_s4)
            feat_s4 = x_s4.mean(dim=(1, 2)) if x_s4.ndim == 4 else x_s4
            outputs["features"] = feat_s4

            if self.fine_head is not None:
                normed_s4 = self.fine_norm(feat_s4) if self.fine_norm is not None else feat_s4
                outputs["fine_logits"] = self.fine_head(self.dropout(normed_s4))

            if self.projection_head is not None:
                outputs["projection"] = F.normalize(
                    self.projection_head(self.dropout(feat_s4)), dim=1
                )

        elif self.backbone_type == "resnet":
            x = self.encoder.conv1(images)
            x = self.encoder.bn1(x) if hasattr(self.encoder, "bn1") else x
            x = self.encoder.act1(x) if hasattr(self.encoder, "act1") else F.relu(x)
            x = self.encoder.maxpool(x) if hasattr(self.encoder, "maxpool") else x

            x_s1 = self.encoder.layer1(x)

            x_s2 = self.encoder.layer2(x_s1)
            feat_s2 = F.adaptive_avg_pool2d(x_s2, 1).flatten(1)
            if self.binary_head is not None:
                normed_s2 = self.binary_norm(feat_s2) if self.binary_norm is not None else feat_s2
                outputs["binary_logits"] = self.binary_head(self.dropout(normed_s2))

            x_s3 = self.encoder.layer3(x_s2)
            feat_s3 = F.adaptive_avg_pool2d(x_s3, 1).flatten(1)
            if self.coarse_head is not None:
                normed_s3 = self.coarse_norm(feat_s3) if self.coarse_norm is not None else feat_s3
                outputs["coarse_logits"] = self.coarse_head(self.dropout(normed_s3))

            x_s4 = self.encoder.layer4(x_s3)
            feat_s4 = F.adaptive_avg_pool2d(x_s4, 1).flatten(1)
            outputs["features"] = feat_s4

            if self.fine_head is not None:
                normed_s4 = self.fine_norm(feat_s4) if self.fine_norm is not None else feat_s4
                outputs["fine_logits"] = self.fine_head(self.dropout(normed_s4))

            if self.projection_head is not None:
                outputs["projection"] = F.normalize(
                    self.projection_head(self.dropout(feat_s4)), dim=1
                )

        else:
            # Generic fallback: compute full features and project all heads from it
            features = self.encoder(images)
            if features.ndim > 2:
                features = F.adaptive_avg_pool2d(features, 1).flatten(1)
            outputs["features"] = features
            dropped = self.dropout(features)

            if self.binary_head is not None:
                normed_b = self.binary_norm(features) if self.binary_norm is not None else features
                outputs["binary_logits"] = self.binary_head(self.dropout(normed_b))
            if self.coarse_head is not None:
                normed_c = self.coarse_norm(features) if self.coarse_norm is not None else features
                outputs["coarse_logits"] = self.coarse_head(self.dropout(normed_c))
            if self.fine_head is not None:
                normed_f = self.fine_norm(features) if self.fine_norm is not None else features
                outputs["fine_logits"] = self.fine_head(self.dropout(normed_f))
            if self.projection_head is not None:
                outputs["projection"] = F.normalize(
                    self.projection_head(dropped), dim=1
                )

        return outputs


# Alias for backward/naming compatibility
StageDecoupledHierarchicalModel = MultiStageHierarchicalSwinModel
