"""Three-Stage Hierarchical Fine-Tuning (3S-HFT) Runner for CystoDS.

Architecture / Training Pipeline:
  • Phase 1 (General Representation Learning):
      - 100% Backbone + All 3 Classification Heads trainable.
      - Loss: CE(binary) + CE(coarse) + CE(fine) + 0.10*SupCon(fine) + 0.25*L_bc + 0.25*L_cf.
      - Learns optimal, uncorrupted feature representation geometry across all 3 levels.
  • Phase 2 (Selective Coarse Classifier Alignment):
      - Backbone 100% FROZEN (requires_grad = False).
      - Binary Head FROZEN (preserves Phase 1 binary ranking & specificity).
      - Fine Head FROZEN.
      - ONLY Coarse Head trainable (~3.8k params).
      - Optimizes 5-class clinical macro-boundary with hierarchy constraint L_bc.
  • Phase 3 (Selective Fine Classifier Alignment):
      - Backbone 100% FROZEN (requires_grad = False).
      - Binary Head FROZEN.
      - Coarse Head FROZEN (preserves Phase 2 optimized coarse boundary).
      - ONLY Fine Head trainable (~16.9k params).
      - Loss: Smoothed Balanced Softmax (patient^0.5 prior) + 0.25*L_cf hierarchy loss.
      - Calibrates 22 fine-grained histopathology classes without damaging Coarse or Binary.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cystods.config import build_run_configuration, get_profile_defaults
from cystods.data.dataset import build_dataloaders
from cystods.data.split import generate_protocol_splits
from cystods.models.factory import build_model
from cystods.models.two_stage_hierarchical import TwoStageHierarchicalSwinModel
from cystods.taxonomy import (
    BINARY_NAMES,
    COARSE_NAMES,
    FINE_NAMES,
)
from cystods.training.engine import (
    evaluate_dataset,
    train_model,
)

logger = logging.getLogger("cystods.three_stage")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Three-Stage Hierarchical Fine-Tuning (3S-HFT) on CystoDS."
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="research",
        choices=["smoke", "ci", "research", "fast_research"],
        help="Execution profile: 'research' (full), 'fast_research' (10 ep), 'smoke' (1 ep). Default: research.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="0",
        help="Split index: '0', '1', '2', or 'all' to run across all 3 splits. Default: 0.",
    )
    parser.add_argument(
        "--phase1-epochs",
        type=int,
        default=None,
        help="Number of epochs for Phase 1. Default: 18 for research, 1 for smoke.",
    )
    parser.add_argument(
        "--phase2-epochs",
        type=int,
        default=None,
        help="Number of epochs for Phase 2 (Coarse Alignment). Default: 5 for research, 1 for smoke.",
    )
    parser.add_argument(
        "--phase3-epochs",
        type=int,
        default=None,
        help="Number of epochs for Phase 3 (Fine Alignment). Default: 6 for research, 1 for smoke.",
    )
    parser.add_argument(
        "--phase1-loss",
        type=str,
        default="cross_entropy",
        help="Fine-level loss function for Phase 1. Default: cross_entropy.",
    )
    parser.add_argument(
        "--phase1-supcon-weight",
        type=float,
        default=None,
        help="Supervised Contrastive loss weight in Phase 1 (default: 0.10 in research, 0.0 in smoke).",
    )
    parser.add_argument(
        "--phase3-loss",
        type=str,
        default="balanced_softmax_smoothed",
        choices=["balanced_softmax_smoothed", "balanced_softmax", "logit_adjustment", "focal", "ldam"],
        help="Fine-level loss function for Phase 3. Default: balanced_softmax_smoothed.",
    )
    parser.add_argument(
        "--ablation-name",
        type=str,
        default=None,
        help="Optional ablation experiment name. When set, results are saved in result/40_ablations/three_stage/.",
    )
    parser.add_argument(
        "--phase1-lr",
        type=float,
        default=0.0003,
        help="Base learning rate for Phase 1. Default: 0.0003.",
    )
    parser.add_argument(
        "--phase2-lr",
        type=float,
        default=0.001,
        help="Coarse head learning rate for Phase 2. Default: 0.001.",
    )
    parser.add_argument(
        "--phase3-lr",
        type=float,
        default=0.001,
        help="Fine head learning rate for Phase 3. Default: 0.001.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'cuda', 'mps', 'cpu', or 'auto'. Default: auto.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        help="Precision mode: 'bf16', 'fp16', 'fp32', or 'auto'. Default: auto.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override dataloader num_workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. Default: 20260729.",
    )
    parser.add_argument(
        "--result-root",
        type=str,
        default=None,
        help="Path to result root directory. Default: ./result.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to dataset archive directory.",
    )
    parser.add_argument(
        "--dry-run",
        type=str,
        default=False,
        help="Validate configuration without training.",
    )
    parser.add_argument(
        "--set",
        action="append",
        dest="cli_overrides",
        metavar="KEY=VALUE",
        help="Arbitrary config key=value overrides.",
    )
    return parser.parse_args(argv)


def run_three_stage_single_split(
    split_index: int,
    profile: str,
    args: argparse.Namespace,
    base_config: dict[str, Any],
    device: torch.device,
    result_root: Path,
    seed: int,
) -> dict[str, Any]:
    """Execute complete 3-Stage pipeline for a single split."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.ablation_name:
        run_output_dir = result_root / "40_ablations" / "three_stage" / f"{args.ablation_name}_split{split_index}_{profile}_{timestamp}"
    else:
        run_output_dir = result_root / "36_three_stage_decoupled" / f"three_stage_split{split_index}_{profile}_{timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Determine epoch counts
    if profile == "smoke":
        phase1_epochs = 1
        phase2_epochs = 1
        phase3_epochs = 1
    elif profile == "fast_research":
        phase1_epochs = args.phase1_epochs or 8
        phase2_epochs = args.phase2_epochs or 3
        phase3_epochs = args.phase3_epochs or 4
    else:
        phase1_epochs = args.phase1_epochs or 18
        phase2_epochs = args.phase2_epochs or 5
        phase3_epochs = args.phase3_epochs or 6

    phase1_loss = args.phase1_loss
    phase3_loss = args.phase3_loss
    phase1_lr = args.phase1_lr
    phase2_lr = args.phase2_lr
    phase3_lr = args.phase3_lr

    logger.info("=" * 80)
    logger.info("🚀 CYSTODS THREE-STAGE HIERARCHICAL FINE-TUNING (3S-HFT)")
    logger.info("  • Split Index       : %d", split_index)
    logger.info("  • Profile           : %s", profile)
    logger.info("  • Phase 1 (Rep)     : %d epochs (lr=%.1e, loss=%s)", phase1_epochs, phase1_lr, phase1_loss)
    logger.info("  • Phase 2 (Coarse)  : %d epochs (lr=%.1e, Coarse Only)", phase2_epochs, phase2_lr)
    logger.info("  • Phase 3 (Fine)    : %d epochs (lr=%.1e, loss=%s)", phase3_epochs, phase3_lr, phase3_loss)
    logger.info("  • Output Directory  : %s", run_output_dir)
    logger.info("=" * 80)

    # Prepare Protocol Splits and DataLoaders
    config = dict(base_config)
    config["protocol_split_index"] = split_index
    config["seed"] = seed

    split_manifest = generate_protocol_splits(config)
    split_frames = split_manifest.splits[split_index]
    loaders, _ = build_dataloaders(split_frames, config, device, seed)

    # Build TwoStageHierarchicalSwinModel
    raw_model = build_model(config, device)
    model = TwoStageHierarchicalSwinModel(raw_model, config, device)

    # -------------------------------------------------------------------------
    # PHASE 1: General Representation Learning
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("▶ GIAI ĐOẠN 1 / PHASE 1: General Representation Learning")
    logger.info("  • Huấn luyện 100% Backbone + 3 Heads trên phân phối tự nhiên")
    logger.info("=" * 70)

    phase1_config = dict(config)
    phase1_config["epochs"] = phase1_epochs
    phase1_config["scheduler_epochs"] = phase1_epochs
    phase1_config["learning_rate"] = phase1_lr
    phase1_config["fine_loss"] = phase1_loss
    phase1_config["binary_loss_weight"] = 1.0
    phase1_config["coarse_loss_weight"] = 1.0
    phase1_config["fine_loss_weight"] = 1.0
    phase1_config["binary_coarse_hierarchy_loss_weight"] = 0.25
    phase1_config["coarse_fine_hierarchy_loss_weight"] = 0.25
    if args.phase1_supcon_weight is not None:
        phase1_config["supervised_contrastive_loss_weight"] = args.phase1_supcon_weight
    else:
        phase1_config["supervised_contrastive_loss_weight"] = 0.10 if profile != "smoke" else 0.0

    p1_start = time.time()
    phase1_metrics, _, phase1_ckpt = train_model(
        model=model,
        loaders=loaders,
        config=phase1_config,
        device=device,
        output_dir=run_output_dir / "phase1_rep",
    )
    p1_duration = time.time() - p1_start
    logger.info("✅ Phase 1 hoàn thành trong %.1f giây", p1_duration)

    # -------------------------------------------------------------------------
    # PHASE 2: Selective Coarse Classifier Alignment
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("▶ GIAI ĐOẠN 2 / PHASE 2: Selective Coarse Classifier Alignment")
    logger.info("  • Đóng băng Backbone + Khóa Binary Head & Fine Head")
    logger.info("  • Chỉ tối ưu Coarse Head (~3.8k params)")
    logger.info("=" * 70)

    freeze_info_p2 = model.freeze_for_phase2(
        freeze_binary_head=True,
        freeze_coarse_head=False,
        freeze_fine_head=True,
        freeze_projection_head=True,
    )
    logger.info("Phase 2 Trainable params: %d (%.2f%%)", freeze_info_p2["trainable_params"], freeze_info_p2["trainable_percentage"])

    phase2_config = dict(config)
    phase2_config["epochs"] = phase2_epochs
    phase2_config["scheduler_epochs"] = phase2_epochs
    phase2_config["learning_rate"] = phase2_lr
    phase2_config["encoder_learning_rate_multiplier"] = 0.0
    phase2_config["supervised_contrastive_loss_weight"] = 0.0
    phase2_config["binary_loss_weight"] = 0.0
    phase2_config["coarse_loss_weight"] = 1.0
    phase2_config["fine_loss_weight"] = 0.0
    phase2_config["binary_coarse_hierarchy_loss_weight"] = 0.25
    phase2_config["coarse_fine_hierarchy_loss_weight"] = 0.0
    phase2_config["warmup_epochs"] = 0.5 if profile != "smoke" else 0.0
    phase2_config["early_stopping_patience"] = 4 if profile != "smoke" else 1

    p2_start = time.time()
    phase2_metrics, _, phase2_ckpt = train_model(
        model=model,
        loaders=loaders,
        config=phase2_config,
        device=device,
        output_dir=run_output_dir / "phase2_coarse",
    )
    p2_duration = time.time() - p2_start
    logger.info("✅ Phase 2 hoàn thành trong %.1f giây", p2_duration)

    # -------------------------------------------------------------------------
    # PHASE 3: Selective Fine Classifier Alignment
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("▶ GIAI ĐOẠN 3 / PHASE 3: Selective Fine Classifier Alignment")
    logger.info("  • Đóng băng Backbone + Khóa Binary Head & Khóa Coarse Head (đã tối ưu ở Phase 2)")
    logger.info("  • Chỉ tối ưu Fine Head (~16.9k params) với %s", phase3_loss)
    logger.info("=" * 70)

    freeze_info_p3 = model.freeze_for_phase2(
        freeze_binary_head=True,
        freeze_coarse_head=True,
        freeze_fine_head=False,
        freeze_projection_head=True,
    )
    logger.info("Phase 3 Trainable params: %d (%.2f%%)", freeze_info_p3["trainable_params"], freeze_info_p3["trainable_percentage"])

    phase3_config = dict(config)
    phase3_config["epochs"] = phase3_epochs
    phase3_config["scheduler_epochs"] = phase3_epochs
    phase3_config["learning_rate"] = phase3_lr
    phase3_config["encoder_learning_rate_multiplier"] = 0.0
    phase3_config["fine_loss"] = phase3_loss
    phase3_config["supervised_contrastive_loss_weight"] = 0.0
    phase3_config["binary_loss_weight"] = 0.0
    phase3_config["coarse_loss_weight"] = 0.0
    phase3_config["fine_loss_weight"] = 1.0
    phase3_config["binary_coarse_hierarchy_loss_weight"] = 0.0
    phase3_config["coarse_fine_hierarchy_loss_weight"] = 0.25
    phase3_config["warmup_epochs"] = 0.5 if profile != "smoke" else 0.0
    phase3_config["early_stopping_patience"] = 4 if profile != "smoke" else 1

    p3_start = time.time()
    phase3_metrics, _, phase3_ckpt = train_model(
        model=model,
        loaders=loaders,
        config=phase3_config,
        device=device,
        output_dir=run_output_dir / "phase3_fine",
    )
    p3_duration = time.time() - p3_start
    logger.info("✅ Phase 3 hoàn thành trong %.1f giây", p3_duration)

    # -------------------------------------------------------------------------
    # Final Comprehensive Evaluation
    # -------------------------------------------------------------------------
    final_test_metrics = evaluate_dataset(
        model=model,
        dataloader=loaders["test"],
        config=phase3_config,
        device=device,
        split_name="test",
    )

    # Generate 4-way comparison table
    p1_val = phase1_metrics.get("splits", {}).get("val", {})
    p2_val = phase2_metrics.get("splits", {}).get("val", {})
    p3_val = phase3_metrics.get("splits", {}).get("val", {})

    p1_bin = p1_val.get("binary", {})
    p1_crs = p1_val.get("coarse", {})
    p1_fin = p1_val.get("fine", {})
    p1_hrc = p1_val.get("hierarchy", {})

    p2_crs = p2_val.get("coarse", {})

    p3_bin = p3_val.get("binary", {})
    p3_crs = p3_val.get("coarse", {})
    p3_fin = p3_val.get("fine", {})
    p3_hrc = p3_val.get("hierarchy", {})

    table_md = f"""
================================================================================
### 📊 Bảng Đối Sánh 3-Stage Hierarchical Fine-Tuning (Split {split_index})

| Tiêu chí / Metric | Phase 1 (Rep CE+SupCon) | Phase 2 (Coarse Only) | **Phase 3 Final (Coarse+Fine Aligned)** | Chênh lệch ($\Delta$ vs Phase 1) |
|---|:---:|:---:|:---:|:---:|
| **Binary AUROC** | {p1_bin.get('auroc', 0.0):.4f} | — | **{p3_bin.get('auroc', 0.0):.4f}** | {p3_bin.get('auroc', 0.0) - p1_bin.get('auroc', 0.0):+.4f} |
| **Binary F1-Score** | {p1_bin.get('f1', 0.0):.4f} | — | **{p3_bin.get('f1', 0.0):.4f}** | {p3_bin.get('f1', 0.0) - p1_bin.get('f1', 0.0):+.4f} |
| **Binary Sensitivity** | {p1_bin.get('sensitivity', 0.0)*100:.2f}% | — | **{p3_bin.get('sensitivity', 0.0)*100:.2f}%** | {(p3_bin.get('sensitivity', 0.0) - p1_bin.get('sensitivity', 0.0))*100:+.2f}% |
| **Binary Specificity** | {p1_bin.get('specificity', 0.0)*100:.2f}% | — | **{p3_bin.get('specificity', 0.0)*100:.2f}%** | {(p3_bin.get('specificity', 0.0) - p1_bin.get('specificity', 0.0))*100:+.2f}% |
| **Coarse Accuracy** | {p1_crs.get('accuracy', 0.0)*100:.2f}% | {p2_crs.get('accuracy', 0.0)*100:.2f}% | **{p3_crs.get('accuracy', 0.0)*100:.2f}%** | {(p3_crs.get('accuracy', 0.0) - p1_crs.get('accuracy', 0.0))*100:+.2f}% |
| **Coarse Macro-F1** | {p1_crs.get('macro_f1', 0.0):.4f} | {p2_crs.get('macro_f1', 0.0):.4f} | **{p3_crs.get('macro_f1', 0.0):.4f}** | {p3_crs.get('macro_f1', 0.0) - p1_crs.get('macro_f1', 0.0):+.4f} |
| **Fine Accuracy** | {p1_fin.get('accuracy', 0.0)*100:.2f}% | — | **{p3_fin.get('accuracy', 0.0)*100:.2f}%** | {(p3_fin.get('accuracy', 0.0) - p1_fin.get('accuracy', 0.0))*100:+.2f}% |
| **Fine Macro-F1 (Supported)** | {p1_fin.get('macro_f1_supported', 0.0):.4f} | — | **{p3_fin.get('macro_f1_supported', 0.0):.4f}** | {p3_fin.get('macro_f1_supported', 0.0) - p1_fin.get('macro_f1_supported', 0.0):+.4f} |
| **Fine Macro-F1 (All 22 Classes)** | {p1_fin.get('macro_f1_all_classes', 0.0):.4f} | — | **{p3_fin.get('macro_f1_all_classes', 0.0):.4f}** | {p3_fin.get('macro_f1_all_classes', 0.0) - p1_fin.get('macro_f1_all_classes', 0.0):+.4f} |
| **Tail Class Recall (n <= 20)** | {p1_fin.get('tail_class_macro_recall', 0.0)*100:.2f}% | — | **{p3_fin.get('tail_class_macro_recall', 0.0)*100:.2f}%** | {(p3_fin.get('tail_class_macro_recall', 0.0) - p1_fin.get('tail_class_macro_recall', 0.0))*100:+.2f}% |
| **Coarse-Fine Consistency** | {p1_hrc.get('coarse_fine_consistency', 0.0)*100:.2f}% | — | **{p3_hrc.get('coarse_fine_consistency', 0.0)*100:.2f}%** | {(p3_hrc.get('coarse_fine_consistency', 0.0) - p1_hrc.get('coarse_fine_consistency', 0.0))*100:+.2f}% |
================================================================================
"""
    logger.info(table_md)

    # Save summary report
    summary_report = {
        "architecture": "3S-HFT",
        "split_index": split_index,
        "profile": profile,
        "phase1_metrics": p1_val,
        "phase2_coarse_metrics": p2_val,
        "phase3_final_metrics": p3_val,
        "test_metrics": final_test_metrics,
        "comparison_table_markdown": table_md,
        "durations_seconds": {
            "phase1": p1_duration,
            "phase2": p2_duration,
            "phase3": p3_duration,
            "total": p1_duration + p2_duration + p3_duration,
        },
    }

    summary_file = run_output_dir / "three_stage_summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    logger.info("📁 Đã lưu báo cáo tổng hợp Three-Stage tại: %s", summary_file)
    return summary_report


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    device_str = args.device
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    logger.info("Using device: %s", device)

    profile = args.profile
    seed = args.seed or 20260729
    result_root = Path(args.result_root) if args.result_root else Path("result")
    result_root.mkdir(parents=True, exist_ok=True)

    cli_overrides = {}
    if args.cli_overrides:
        for item in args.cli_overrides:
            if "=" in item:
                k, v = item.split("=", 1)
                cli_overrides[k.strip()] = v.strip()

    base_config = build_run_configuration(
        stage_name="30_proposed",
        profile=profile,
        cli_overrides=cli_overrides,
    )
    if args.data_root:
        base_config["data_root"] = args.data_root
    if args.num_workers is not None:
        base_config["num_workers"] = args.num_workers
    if args.precision != "auto":
        base_config["precision"] = args.precision

    split_arg = str(args.split).strip().lower()
    if split_arg == "all":
        splits_to_run = [0, 1, 2]
    else:
        splits_to_run = [int(s.strip()) for s in split_arg.split(",") if s.strip().isdigit()]

    all_results = []
    for s_idx in splits_to_run:
        res = run_three_stage_single_split(
            split_index=s_idx,
            profile=profile,
            args=args,
            base_config=base_config,
            device=device,
            result_root=result_root,
            seed=seed + s_idx,
        )
        all_results.append(res)

    logger.info("🎉 Hoàn thành toàn bộ thực nghiệm 3-Stage Hierarchical Fine-Tuning!")


if __name__ == "__main__":
    main()
