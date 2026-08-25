"""Stacked Cascaded Hierarchical Classifier at Late-Stage for Cystoscopy.

Implements the Late-Stage Stacked Multi-Head Architecture:
  - Backbone: Swin-Tiny (Extracts deep 768-d spatial representation at Stage 4)
  - Head 1 (Binary Head): z (768-d) --> Linear(768, 2) --> Binary Logits & Embedding
  - Head 2 (Coarse Head): [z, bin_probs, bin_embed] (834-d) --> MLP --> Coarse Logits (5-d) & Embedding
  - Head 3 (Fine Head):   [z, bin_ctx, coarse_ctx] (903-d) --> MLP --> Fine Logits (22-d)
  - Bidirectional Feedback: In forward pass / inference, computes Hierarchical Marginalization
    blending Coarse direct prediction and Fine-to-Coarse parent aggregation:
      P_ens(C) = lambda * P_coarse(C) + (1 - lambda) * sum_{f in Children(C)} P_fine(f)

Gradient Control:
  - detach_hierarchy=False (Default): Full 1-stage end-to-end gradient backpropagation.
  - detach_hierarchy=True: Detaches conditioning embeddings before passing to downstream heads,
    protecting Binary/Coarse Head weights from high-variance Fine Loss gradients.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from cystods.models.factory import resolve_model_name
from cystods.taxonomy import BINARY_NAMES, COARSE_NAMES, FINE_NAMES, FINE_PARENT_ID


class CascadedHierarchicalCystoModel(nn.Module):
    """Late-Stage Cascaded / Stacked Multi-Head Hierarchical Classifier.

    Places all 3 heads at Stage 4 (768-d), creating a progressive conditional
    cascade where Binary guides Coarse, and Binary + Coarse guide Fine.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        self.task_mode = str(config.get("task_mode", "hierarchical"))
        self.active_tasks = frozenset({"binary", "coarse", "fine"})

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
        self.detach_hierarchy = bool(config.get("detach_hierarchy", False))
        self.hierarchy_lambda = float(config.get("hierarchy_lambda", 0.25))

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
                f"Failed to construct encoder '{model_name}' with pretrained={pretrained}."
            ) from exc

        feature_dim = int(getattr(self.encoder, "num_features", 0))
        if feature_dim <= 0:
            raise RuntimeError(f"Encoder '{model_name}' does not expose a valid num_features.")
        self.feature_dim = feature_dim

        self.num_binary = len(BINARY_NAMES)   # 2
        self.num_coarse = len(COARSE_NAMES)   # 5
        self.num_fine = len(FINE_NAMES)       # 22

        self.bin_embed_dim = int(config.get("binary_embedding_dim", 64))
        self.coarse_embed_dim = int(config.get("coarse_embedding_dim", 64))

        self.dropout = nn.Dropout(dropout)

        # 1. Binary Head (Direct on Backbone Stage 4 feature)
        self.binary_head = nn.Linear(feature_dim, self.num_binary)
        self.binary_embed_proj = nn.Sequential(
            nn.Linear(self.num_binary, self.bin_embed_dim),
            nn.LayerNorm(self.bin_embed_dim),
            nn.GELU(),
        )

        # 2. Coarse Head (Conditioned on z + Binary Context)
        coarse_in_dim = feature_dim + self.num_binary + self.bin_embed_dim  # 768 + 2 + 64 = 834
        self.coarse_head = nn.Sequential(
            nn.Linear(coarse_in_dim, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(384, self.num_coarse),
        )
        self.coarse_embed_proj = nn.Sequential(
            nn.Linear(self.num_coarse, self.coarse_embed_dim),
            nn.LayerNorm(self.coarse_embed_dim),
            nn.GELU(),
        )

        # 3. Fine Head (Conditioned on z + Binary Context + Coarse Context)
        fine_in_dim = coarse_in_dim + self.num_coarse + self.coarse_embed_dim  # 834 + 5 + 64 = 903
        self.fine_head = nn.Sequential(
            nn.Linear(fine_in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, self.num_fine),
        )

        # 4. Supervised Contrastive Projection Head
        supcon_weight = float(config.get("supervised_contrastive_loss_weight", 0.0))
        self.projection_head = (
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, projection_dim),
            )
            if supcon_weight > 0
            else None
        )

        # Fine-to-Coarse parent mapping matrix for Bottom-up Marginalization
        self.register_buffer(
            "fine_to_coarse_matrix",
            self._build_fine_to_coarse_matrix(),
            persistent=False,
        )

    def _build_fine_to_coarse_matrix(self) -> torch.Tensor:
        """Create binary indicator matrix M in R^{22 x 5} where M[f, c] = 1 if Children(c) == f."""
        matrix = torch.zeros(self.num_fine, self.num_coarse, dtype=torch.float32)
        for f_idx, c_idx in enumerate(FINE_PARENT_ID):
            matrix[f_idx, c_idx] = 1.0
        return matrix

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Extract Stage 4 pooled features from backbone."""
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

    def forward(
        self,
        images: torch.Tensor,
        *,
        lambda_marginalization: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Progressive forward pass through stacked cascaded heads."""
        z = self.encode(images)
        z_drop = self.dropout(z)

        # 1. Binary Head Forward
        binary_logits = self.binary_head(z_drop)
        binary_probs = F.softmax(binary_logits, dim=-1)
        binary_embed = self.binary_embed_proj(binary_probs)

        # Apply gradient detachment if configured
        if self.detach_hierarchy:
            bin_ctx = torch.cat([binary_probs.detach(), binary_embed.detach()], dim=-1)
        else:
            bin_ctx = torch.cat([binary_probs, binary_embed], dim=-1)

        # 2. Coarse Head Forward (Conditioned on Binary Context)
        coarse_input = torch.cat([z_drop, bin_ctx], dim=-1)
        coarse_logits = self.coarse_head(coarse_input)
        coarse_probs = F.softmax(coarse_logits, dim=-1)
        coarse_embed = self.coarse_embed_proj(coarse_probs)

        if self.detach_hierarchy:
            coarse_ctx = torch.cat([coarse_probs.detach(), coarse_embed.detach()], dim=-1)
        else:
            coarse_ctx = torch.cat([coarse_probs, coarse_embed], dim=-1)

        # 3. Fine Head Forward (Conditioned on Binary + Coarse Context)
        fine_input = torch.cat([z_drop, bin_ctx, coarse_ctx], dim=-1)
        fine_logits = self.fine_head(fine_input)
        fine_probs = F.softmax(fine_logits, dim=-1)

        outputs: dict[str, torch.Tensor] = {
            "features": z,
            "binary_logits": binary_logits,
            "coarse_logits": coarse_logits,
            "fine_logits": fine_logits,
        }

        if self.projection_head is not None:
            outputs["projection"] = F.normalize(self.projection_head(z_drop), dim=1)

        return outputs

    def compute_hierarchical_marginalization(
        self,
        fine_probs: torch.Tensor,
        coarse_probs: torch.Tensor,
        lambda_val: float | None = None,
    ) -> torch.Tensor:
        """Calculate bottom-up Parent probabilities: P_ens(C) = lambda * P_coarse + (1-lambda) * sum(P_fine)."""
        lam = self.hierarchy_lambda if lambda_val is None else float(lambda_val)
        fine_to_coarse = torch.matmul(fine_probs, self.fine_to_coarse_matrix.to(fine_probs.device))
        return lam * coarse_probs + (1.0 - lam) * fine_to_coarse

    def freeze_backbone(self) -> None:
        """Freeze encoder backbone weights."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze encoder backbone weights."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def configure_phase_freezing(self, phase: int) -> None:
        """Configure trainable components according to training phase:
        - Phase 1: All parameters trainable (Backbone + Binary + Coarse + Fine + SupCon)
        - Phase 2: Freeze Backbone, Binary Head, Fine Head. ONLY Coarse Head trainable.
        - Phase 3: Freeze Backbone, Binary Head, Coarse Head. ONLY Fine Head trainable.
        """
        if phase == 1:
            for param in self.parameters():
                param.requires_grad = True
        elif phase == 2:
            self.freeze_backbone()
            for param in self.binary_head.parameters():
                param.requires_grad = False
            for param in self.binary_embed_proj.parameters():
                param.requires_grad = False
            for param in self.coarse_head.parameters():
                param.requires_grad = True
            for param in self.coarse_embed_proj.parameters():
                param.requires_grad = True
            for param in self.fine_head.parameters():
                param.requires_grad = False
            if self.projection_head is not None:
                for param in self.projection_head.parameters():
                    param.requires_grad = False
        elif phase == 3:
            self.freeze_backbone()
            for param in self.binary_head.parameters():
                param.requires_grad = False
            for param in self.binary_embed_proj.parameters():
                param.requires_grad = False
            for param in self.coarse_head.parameters():
                param.requires_grad = False
            for param in self.coarse_embed_proj.parameters():
                param.requires_grad = False
            for param in self.fine_head.parameters():
                param.requires_grad = True
            if self.projection_head is not None:
                for param in self.projection_head.parameters():
                    param.requires_grad = False
        else:
            raise ValueError(f"Unsupported phase index: {phase}. Expected 1, 2, or 3.")

    def get_parameter_summary(self) -> dict[str, Any]:
        """Return total and trainable parameter breakdown."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        return {
            "model_type": "CascadedHierarchicalCystoModel",
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "trainable_percentage": (trainable_params / total_params * 100) if total_params else 0.0,
            "detach_hierarchy": self.detach_hierarchy,
            "hierarchy_lambda": self.hierarchy_lambda,
        }
