"""Unit and contract tests for TwoStageDecoupledHierarchicalModel."""

from __future__ import annotations

import pytest
import torch

from cystods.models.two_stage_hierarchical import (
    TwoStageDecoupledHierarchicalModel,
)
from cystods.taxonomy import BINARY_NAMES, COARSE_NAMES, FINE_NAMES


def test_two_stage_hierarchical_forward_shape():
    """Verify forward pass output shapes across all active heads."""
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
    model = TwoStageDecoupledHierarchicalModel(config)
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


def test_two_stage_phase2_freezing_contract():
    """Verify Phase 2 freezing: 100% backbone frozen, only classification heads trainable."""
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
    model = TwoStageDecoupledHierarchicalModel(config)

    # Phase 1: All trainable
    p1_summary = model.get_parameter_summary()
    assert p1_summary["phase"] == 1
    assert p1_summary["backbone_frozen"] is False
    assert p1_summary["frozen_params"] == 0
    assert p1_summary["trainable_percentage"] == 100.0

    # Transition to Phase 2
    p2_summary = model.freeze_backbone()
    assert p2_summary["phase"] == 2
    assert p2_summary["backbone_frozen"] is True
    assert p2_summary["frozen_params"] > 27_000_000
    assert p2_summary["trainable_params"] < 50_000  # Only heads
    assert p2_summary["trainable_percentage"] < 0.20

    # Verify encoder parameters have requires_grad = False
    assert all(not p.requires_grad for p in model.encoder.parameters())

    # Verify heads parameters have requires_grad = True
    assert all(p.requires_grad for p in model.binary_head.parameters())
    assert all(p.requires_grad for p in model.coarse_head.parameters())
    assert all(p.requires_grad for p in model.fine_head.parameters())


def test_two_stage_phase2_gradient_isolation():
    """Verify backward pass in Phase 2 only updates heads, no gradient on encoder."""
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
        "supervised_contrastive_loss_weight": 0.0,
    }
    model = TwoStageDecoupledHierarchicalModel(config)
    model.freeze_backbone()

    dummy_images = torch.randn(2, 3, 64, 64)
    outputs = model(dummy_images)
    total_loss = outputs["fine_logits"].sum() + outputs["coarse_logits"].sum() + outputs["binary_logits"].sum()
    total_loss.backward()

    # Heads must have gradients
    assert model.fine_head.weight.grad is not None
    assert model.coarse_head.weight.grad is not None
    assert model.binary_head.weight.grad is not None

    # Encoder must have ZERO gradient
    for param in model.encoder.parameters():
        assert param.grad is None


def test_two_stage_tau_normalization():
    """Verify Tau-Normalization rescales classifier weights."""
    config = {
        "model_name": "swin_tiny_patch4_window7_224.ms_in1k",
        "pretrained": False,
        "image_size": 64,
        "task_mode": "hierarchical",
        "binary_loss_weight": 1.0,
        "coarse_loss_weight": 1.0,
        "fine_loss_weight": 1.0,
        "supervised_contrastive_loss_weight": 0.0,
    }
    model = TwoStageDecoupledHierarchicalModel(config)

    # Set arbitrary un-normalized weights
    with torch.no_grad():
        model.fine_head.weight.copy_(torch.randn_like(model.fine_head.weight) * 5.0)

    model.tau_normalize_classifiers(tau=1.0)
    row_norms = torch.norm(model.fine_head.weight.data, p=2, dim=1)
    assert torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-4)


def test_two_stage_selective_fine_only_freezing():
    """Verify Selective Fine-Only freezing: Backbone, Binary & Coarse are frozen, only Fine head trainable."""
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
    model = TwoStageDecoupledHierarchicalModel(config)
    summary = model.freeze_for_phase2(freeze_binary_head=True, freeze_coarse_head=True, freeze_projection_head=True)

    assert summary["phase"] == 2
    assert summary["backbone_frozen"] is True
    # Only fine head (768 * 22 + 22 = 16918 params) is trainable
    assert summary["trainable_params"] == (768 * len(FINE_NAMES) + len(FINE_NAMES))

    # Check requires_grad flags
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(not p.requires_grad for p in model.binary_head.parameters())
    assert all(not p.requires_grad for p in model.coarse_head.parameters())
    assert all(not p.requires_grad for p in model.projection_head.parameters())
    assert all(p.requires_grad for p in model.fine_head.parameters())

    # Backward pass check
    dummy_images = torch.randn(2, 3, 64, 64)
    outputs = model(dummy_images)
    loss = outputs["fine_logits"].sum()
    loss.backward()

    assert model.fine_head.weight.grad is not None
    assert model.binary_head.weight.grad is None
    assert model.coarse_head.weight.grad is None

