"""Unit and contract tests for MultiStageHierarchicalSwinModel and runner."""

from __future__ import annotations

import torch
import pytest

from cystods.models.multi_stage_hierarchical import (
    MultiStageHierarchicalSwinModel,
    StageDecoupledHierarchicalModel,
)
from cystods.taxonomy import BINARY_NAMES, COARSE_NAMES, FINE_NAMES


def test_multi_stage_hierarchical_forward_shape():
    """Verify forward pass output shapes and tensor contract."""
    config = {
        "model_name": "swin_tiny_patch4_window7_224.ms_in1k",
        "pretrained": False,
        "image_size": 64,
        "dropout": 0.1,
        "projection_dim": 128,
        "task_mode": "hierarchical",
        "binary_loss_weight": 1.0,
        "coarse_loss_weight": 1.0,
        "fine_loss_weight": 1.0,
        "supervised_contrastive_loss_weight": 0.10,
    }
    model = MultiStageHierarchicalSwinModel(config)
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 64, 64)
    outputs = model(dummy_images)

    assert "features" in outputs
    assert "binary_logits" in outputs
    assert "coarse_logits" in outputs
    assert "fine_logits" in outputs
    assert "projection" in outputs

    assert outputs["features"].shape == (batch_size, model.feature_dim)
    assert outputs["binary_logits"].shape == (batch_size, len(BINARY_NAMES))
    assert outputs["coarse_logits"].shape == (batch_size, len(COARSE_NAMES))
    assert outputs["fine_logits"].shape == (batch_size, len(FINE_NAMES))
    assert outputs["projection"].shape == (batch_size, 128)

    # Check projection L2 normalization
    norms = torch.norm(outputs["projection"], dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_multi_stage_hierarchical_gradient_decoupling():
    """Verify that Binary and Coarse losses only backpropagate through their respective depths."""
    config = {
        "model_name": "swin_tiny_patch4_window7_224.ms_in1k",
        "pretrained": False,
        "image_size": 64,
        "dropout": 0.0,
        "projection_dim": 128,
        "task_mode": "hierarchical",
        "binary_loss_weight": 1.0,
        "coarse_loss_weight": 1.0,
        "fine_loss_weight": 1.0,
        "supervised_contrastive_loss_weight": 0.10,
    }
    model = MultiStageHierarchicalSwinModel(config)
    dummy_images = torch.randn(2, 3, 64, 64)

    # Backward only binary loss
    outputs = model(dummy_images)
    binary_loss = outputs["binary_logits"].sum()
    binary_loss.backward()

    # Stage 2 should have grad
    assert list(model.encoder.layers[1].parameters())[0].grad is not None
    # Stage 3 and Stage 4 should NOT have grad from binary loss
    assert list(model.encoder.layers[2].parameters())[0].grad is None
    assert list(model.encoder.layers[3].parameters())[0].grad is None


def test_multi_stage_hierarchical_freezing():
    """Verify layer freezing (freeze stages 1-2 vs 1-3)."""
    config = {
        "model_name": "swin_tiny_patch4_window7_224.ms_in1k",
        "pretrained": False,
        "image_size": 64,
        "dropout": 0.1,
        "projection_dim": 128,
        "task_mode": "hierarchical",
        "partial_finetune": True,
        "frozen_stages_count": 2,
        "binary_loss_weight": 1.0,
        "coarse_loss_weight": 1.0,
        "fine_loss_weight": 1.0,
        "supervised_contrastive_loss_weight": 0.10,
    }
    model = MultiStageHierarchicalSwinModel(config)
    summary = model.get_parameter_summary()

    assert summary["partial_finetune"] is True
    assert summary["frozen_stages_count"] == 2
    assert summary["frozen_params"] > 0
    assert summary["trainable_params"] > 0
    assert summary["trainable_params"] < summary["total_params"]

    # Patch embed and layers[0..1] must be frozen
    assert all(not p.requires_grad for p in model.encoder.patch_embed.parameters())
    assert all(not p.requires_grad for p in model.encoder.layers[0].parameters())
    assert all(not p.requires_grad for p in model.encoder.layers[1].parameters())

    # layers[2..3] and heads must be trainable
    assert any(p.requires_grad for p in model.encoder.layers[2].parameters())
    assert any(p.requires_grad for p in model.encoder.layers[3].parameters())
    assert all(p.requires_grad for p in model.binary_head.parameters())
    assert all(p.requires_grad for p in model.coarse_head.parameters())
    assert all(p.requires_grad for p in model.fine_head.parameters())
