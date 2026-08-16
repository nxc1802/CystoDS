"""Unit and integration tests for Partial Finetuning and Early Layer Freezing."""

from __future__ import annotations

import pytest
import torch

from cystods.cli import _build_parser
from cystods.config import PROPOSED_CANONICAL_CONFIG
from cystods.config_schema import BASE_CONFIG
from cystods.models.hierarchical import HierarchicalCystoModel
from cystods.training.optimizer import build_optimizer


def test_base_config_has_partial_finetune_keys() -> None:
    assert "partial_finetune" in BASE_CONFIG
    assert "freeze_early_layers" in BASE_CONFIG
    assert "frozen_stages_count" in BASE_CONFIG
    assert BASE_CONFIG["partial_finetune"] is False
    assert BASE_CONFIG["frozen_stages_count"] == 2


def test_swin_tiny_full_finetune_all_trainable() -> None:
    config = dict(PROPOSED_CANONICAL_CONFIG)
    config["pretrained"] = False
    config["partial_finetune"] = False

    model = HierarchicalCystoModel(config)
    summary = model.get_parameter_summary()

    assert summary["partial_finetune"] is False
    assert summary["frozen_params"] == 0
    assert summary["trainable_params"] == summary["total_params"]
    assert summary["trainable_percentage"] == 100.0

    # Ensure all patch_embed and layers require grad
    assert all(p.requires_grad for p in model.encoder.patch_embed.parameters())
    assert all(p.requires_grad for p in model.encoder.layers[0].parameters())
    assert all(p.requires_grad for p in model.encoder.layers[1].parameters())
    assert all(p.requires_grad for p in model.encoder.layers[2].parameters())
    assert all(p.requires_grad for p in model.encoder.layers[3].parameters())


def test_swin_tiny_partial_finetune_freezes_early_stages() -> None:
    config = dict(PROPOSED_CANONICAL_CONFIG)
    config["pretrained"] = False
    config["partial_finetune"] = True
    config["frozen_stages_count"] = 2

    model = HierarchicalCystoModel(config)
    summary = model.get_parameter_summary()

    assert summary["partial_finetune"] is True
    assert summary["frozen_params"] > 0
    assert summary["trainable_params"] < summary["total_params"]
    assert summary["frozen_params"] == 1195842  # patch_embed (3120) + stage1 (188964) + stage2 (1003758)

    # Patch Embedding + Stage 1 + Stage 2 are frozen
    assert all(not p.requires_grad for p in model.encoder.patch_embed.parameters())
    assert all(not p.requires_grad for p in model.encoder.layers[0].parameters())
    assert all(not p.requires_grad for p in model.encoder.layers[1].parameters())

    # Stage 3 + Stage 4 + Norm are trainable
    assert all(p.requires_grad for p in model.encoder.layers[2].parameters())
    assert all(p.requires_grad for p in model.encoder.layers[3].parameters())
    assert all(p.requires_grad for p in model.encoder.norm.parameters())

    # Classification & projection heads are trainable
    assert all(p.requires_grad for p in model.binary_head.parameters())
    assert all(p.requires_grad for p in model.coarse_head.parameters())
    assert all(p.requires_grad for p in model.fine_head.parameters())
    assert all(p.requires_grad for p in model.projection_head.parameters())


def test_optimizer_only_registers_trainable_parameters() -> None:
    config = dict(PROPOSED_CANONICAL_CONFIG)
    config["pretrained"] = False
    config["partial_finetune"] = True
    config["use_fused_optimizer"] = False

    device = torch.device("cpu")
    model = HierarchicalCystoModel(config)
    optimizer = build_optimizer(model, config, device)

    # Total parameters in optimizer should match trainable_params
    summary = model.get_parameter_summary()
    opt_params_count = sum(p.numel() for group in optimizer.param_groups for p in group["params"])

    assert opt_params_count == summary["trainable_params"]
    assert all(p.requires_grad for group in optimizer.param_groups for p in group["params"])


def test_partial_finetune_forward_backward_pass() -> None:
    config = dict(PROPOSED_CANONICAL_CONFIG)
    config["pretrained"] = False
    config["partial_finetune"] = True
    config["use_fused_optimizer"] = False

    device = torch.device("cpu")
    model = HierarchicalCystoModel(config).to(device)
    optimizer = build_optimizer(model, config, device)

    dummy_input = torch.randn(2, 3, 224, 224, device=device)
    outputs = model(dummy_input)

    loss = (
        outputs["binary_logits"].sum()
        + outputs["coarse_logits"].sum()
        + outputs["fine_logits"].sum()
        + outputs["projection"].sum()
    )
    optimizer.zero_grad()
    loss.backward()

    # Frozen layers must have NO gradient
    for p in model.encoder.patch_embed.parameters():
        assert p.grad is None
    for p in model.encoder.layers[0].parameters():
        assert p.grad is None
    for p in model.encoder.layers[1].parameters():
        assert p.grad is None

    # Trainable layers must have valid gradients
    assert any(p.grad is not None for p in model.encoder.layers[2].parameters())
    assert any(p.grad is not None for p in model.encoder.layers[3].parameters())
    assert any(p.grad is not None for p in model.fine_head.parameters())

    optimizer.step()


def test_swin_tiny_freeze_stage3_freezes_through_stage3() -> None:
    config = dict(PROPOSED_CANONICAL_CONFIG)
    config["pretrained"] = False
    config["partial_finetune"] = True
    config["frozen_stages_count"] = 3

    model = HierarchicalCystoModel(config)
    summary = model.get_parameter_summary()

    assert summary["partial_finetune"] is True
    assert summary["frozen_params"] == 12151242  # patch_embed + stage1 + stage2 + stage3

    # Patch Embedding + Stage 1 + Stage 2 + Stage 3 are frozen
    assert all(not p.requires_grad for p in model.encoder.patch_embed.parameters())
    assert all(not p.requires_grad for p in model.encoder.layers[0].parameters())
    assert all(not p.requires_grad for p in model.encoder.layers[1].parameters())
    assert all(not p.requires_grad for p in model.encoder.layers[2].parameters())

    # Stage 4 + Norm are trainable
    assert all(p.requires_grad for p in model.encoder.layers[3].parameters())
    assert all(p.requires_grad for p in model.encoder.norm.parameters())

    # Classification & projection heads are trainable
    assert all(p.requires_grad for p in model.binary_head.parameters())
    assert all(p.requires_grad for p in model.coarse_head.parameters())
    assert all(p.requires_grad for p in model.fine_head.parameters())
    assert all(p.requires_grad for p in model.projection_head.parameters())


def test_cli_parser_accepts_partial_finetune_flag() -> None:
    parser = _build_parser()

    args1 = parser.parse_args(["run", "30", "--split", "0"])
    assert args1.partial_finetune is False
    assert args1.freeze_stage3 is False
    assert args1.freeze_stages is None

    args2 = parser.parse_args(["run", "30", "--split", "0", "--partial-finetune"])
    assert args2.partial_finetune is True

    args3 = parser.parse_args(["run", "40", "--split", "1", "--freeze-early-layers"])
    assert args3.partial_finetune is True

    args4 = parser.parse_args(["run", "30", "--split", "0", "--freeze-stage3"])
    assert args4.freeze_stage3 is True

    args5 = parser.parse_args(["run", "30", "--split", "2", "--freeze-stages", "3"])
    assert args5.freeze_stages == 3
