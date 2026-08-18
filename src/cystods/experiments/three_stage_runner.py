"""Three-Stage Hierarchical Fine-Tuning (3S-HFT) Experiment Runner.

Orchestrates the 3-Phase Training Pipeline:
  • Phase 1: General Representation Learning (100% Backbone + Heads, Natural Distribution + SupCon + CE)
  • Phase 2: Selective Coarse Alignment (Frozen Backbone, Frozen Binary & Fine, ONLY Coarse Head Trainable)
  • Phase 3: Selective Fine Alignment (Frozen Backbone, Frozen Binary & Coarse, ONLY Fine Head with Smoothed BSM)
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from cystods.config import load_config
from cystods.core import find_latest_completed_protocol_run
from cystods.data.manifest import load_and_validate_manifest
from cystods.data.sampler import build_dataloaders
from cystods.data.splits.protocol import build_all_protocol_splits
from cystods.infra.environment import close_logger, setup_logger
from cystods.infra.serialization import json_ready, utc_now_iso, write_json
from cystods.models.two_stage_hierarchical import TwoStageDecoupledHierarchicalModel
from cystods.science import split_fingerprint
from cystods.training.engine import train_model
from cystods.training.runtime import resolve_device, resolve_precision, seed_everything


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cystods-three-stage",
        description=(
            "CystoDS: Three-Stage Hierarchical Fine-Tuning (3S-HFT) Experiment Runner.\n"
            "Phase 1: Representation Learning (Full Network, Natural Distribution + SupCon)\n"
            "Phase 2: Coarse Alignment (Frozen Backbone, Train Only Coarse Head)\n"
            "Phase 3: Fine Alignment (Frozen Backbone, Train Only Fine Head with Smoothed BSM)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="0",
        help="Split index to run (0, 1, 2, or 'all'). Default: 0.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="research",
        choices=["research", "smoke"],
        help="Profile mode: 'research' (full training) or 'smoke' (1 epoch test). Default: research.",
    )
    parser.add_argument(
        "--phase1-epochs",
        type=int,
        default=None,
        help="Number of epochs for Phase 1 (default: 25 in research, 1 in smoke).",
    )
    parser.add_argument(
        "--phase2-epochs",
        type=int,
        default=None,
        help="Number of epochs for Phase 2 Coarse Alignment (default: 10 in research, 1 in smoke).",
    )
    parser.add_argument(
        "--phase3-epochs",
        type=int,
        default=None,
        help="Number of epochs for Phase 3 Fine Alignment (default: 10 in research, 1 in smoke).",
    )
    parser.add_argument(
        "--phase1-loss",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "focal", "weighted_ce"],
        help="Fine-level loss function for Phase 1. Default: cross_entropy.",
    )
    parser.add_argument(
        "--phase1-supcon-weight",
        type=float,
        default=None,
        help="Supervised Contrastive loss weight in Phase 1 (default: 0.10 in research, 0.0 in smoke).",
    )
    parser.add_argument(
        "--phase2-loss",
        type=str,
        default="balanced_softmax_smoothed",
        choices=["balanced_softmax_smoothed", "balanced_softmax", "cross_entropy", "logit_adjustment", "focal", "weighted_ce"],
        help="Coarse-level loss function for Phase 2. Default: balanced_softmax_smoothed.",
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
        help="Optional ablation experiment name. When set, results are saved to result/40_ablations/three_stage/.",
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
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size for training.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Override batch size for evaluation.",
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
        action="store_true",
        help="Validate configuration without executing training.",
    )
    parser.add_argument(
        "--set",
        action="append",
        dest="cli_overrides",
        metavar="KEY=VALUE",
        help="Arbitrary config key=value overrides.",
    )
    return parser


def generate_four_way_comparison_table(
    p1_metrics: dict[str, Any],
    p2_metrics: dict[str, Any],
    p3_metrics: dict[str, Any],
    split_index: int,
) -> str:
    """Generate 4-way Markdown comparison: Phase 1 vs Phase 2 vs Phase 3 Final."""
    p1_val = (p1_metrics.get("splits", {}) or {}).get("val", {}) or {}
    p2_val = (p2_metrics.get("splits", {}) or {}).get("val", {}) or {}
    p3_val = (p3_metrics.get("splits", {}) or {}).get("val", {}) or {}

    p1_bin = (p1_val.get("binary") or {}) if isinstance(p1_val.get("binary"), dict) else {}
    p1_crs = (p1_val.get("coarse") or {}) if isinstance(p1_val.get("coarse"), dict) else {}
    p1_fin = (p1_val.get("fine") or {}) if isinstance(p1_val.get("fine"), dict) else {}
    p1_hrc = (p1_val.get("hierarchy") or {}) if isinstance(p1_val.get("hierarchy"), dict) else {}

    p2_crs = (p2_val.get("coarse") or {}) if isinstance(p2_val.get("coarse"), dict) else {}

    p3_bin = (p3_val.get("binary") or p1_bin) if isinstance(p3_val.get("binary"), dict) else p1_bin
    p3_crs = (p3_val.get("coarse") or p2_crs) if isinstance(p3_val.get("coarse"), dict) else p2_crs
    p3_fin = (p3_val.get("fine") or {}) if isinstance(p3_val.get("fine"), dict) else {}
    p3_hrc = (p3_val.get("hierarchy") or p1_hrc) if isinstance(p3_val.get("hierarchy"), dict) else p1_hrc

    delta_crs_acc = (p3_crs.get("accuracy", 0.0) - p1_crs.get("accuracy", 0.0)) * 100
    delta_crs_f1 = p3_crs.get("macro_f1", 0.0) - p1_crs.get("macro_f1", 0.0)
    delta_fin_f1 = p3_fin.get("macro_f1_supported", 0.0) - p1_fin.get("macro_f1_supported", 0.0)
    delta_all_f1 = p3_fin.get("macro_f1_all_classes", 0.0) - p1_fin.get("macro_f1_all_classes", 0.0)

    lines = [
        "================================================================================",
        f"### 📊 Bảng Đối Sánh Three-Stage Hierarchical Fine-Tuning (Split {split_index})",
        "",
        "| Tiêu chí / Metric | Phase 1 (Rep CE+SupCon) | Phase 2 (Coarse Aligned) | **Phase 3 Final (Coarse+Fine Aligned)** | Chênh lệch ($\\Delta$ vs Phase 1) |",
        "|---|:---:|:---:|:---:|:---:|",
        f"| **Binary AUROC** | {p1_bin.get('auroc', 0.0):.4f} | — | **{p3_bin.get('auroc', 0.0):.4f}** | {p3_bin.get('auroc', 0.0) - p1_bin.get('auroc', 0.0):+.4f} |",
        f"| **Binary F1-Score** | {p1_bin.get('f1', 0.0):.4f} | — | **{p3_bin.get('f1', 0.0):.4f}** | {p3_bin.get('f1', 0.0) - p1_bin.get('f1', 0.0):+.4f} |",
        f"| **Binary Sensitivity** | {p1_bin.get('sensitivity', 0.0)*100:.2f}% | — | **{p3_bin.get('sensitivity', 0.0)*100:.2f}%** | {(p3_bin.get('sensitivity', 0.0) - p1_bin.get('sensitivity', 0.0))*100:+.2f}% |",
        f"| **Binary Specificity** | {p1_bin.get('specificity', 0.0)*100:.2f}% | — | **{p3_bin.get('specificity', 0.0)*100:.2f}%** | {(p3_bin.get('specificity', 0.0) - p1_bin.get('specificity', 0.0))*100:+.2f}% |",
        f"| **Coarse Accuracy** | {p1_crs.get('accuracy', 0.0)*100:.2f}% | {p2_crs.get('accuracy', 0.0)*100:.2f}% | **{p3_crs.get('accuracy', 0.0)*100:.2f}%** | {delta_crs_acc:+.2f}% |",
        f"| **Coarse Macro-F1** | {p1_crs.get('macro_f1', 0.0):.4f} | {p2_crs.get('macro_f1', 0.0):.4f} | **{p3_crs.get('macro_f1', 0.0):.4f}** | {delta_crs_f1:+.4f} |",
        f"| **Fine Accuracy** | {p1_fin.get('accuracy', 0.0)*100:.2f}% | — | **{p3_fin.get('accuracy', 0.0)*100:.2f}%** | {(p3_fin.get('accuracy', 0.0) - p1_fin.get('accuracy', 0.0))*100:+.2f}% |",
        f"| **Fine Macro-F1 (Supported)** | {p1_fin.get('macro_f1_supported', 0.0):.4f} | — | **{p3_fin.get('macro_f1_supported', 0.0):.4f}** | {delta_fin_f1:+.4f} |",
        f"| **Fine Macro-F1 (All 22 Classes)** | {p1_fin.get('macro_f1_all_classes', 0.0):.4f} | — | **{p3_fin.get('macro_f1_all_classes', 0.0):.4f}** | {delta_all_f1:+.4f} |",
        f"| **Tail Class Recall (n <= 20)** | {p1_fin.get('tail_class_macro_recall', 0.0)*100:.2f}% | — | **{p3_fin.get('tail_class_macro_recall', 0.0)*100:.2f}%** | {(p3_fin.get('tail_class_macro_recall', 0.0) - p1_fin.get('tail_class_macro_recall', 0.0))*100:+.2f}% |",
        f"| **Coarse-Fine Consistency** | {p1_hrc.get('coarse_fine_consistency', 0.0)*100:.2f}% | — | **{p3_hrc.get('coarse_fine_consistency', 0.0)*100:.2f}%** | {(p3_hrc.get('coarse_fine_consistency', 0.0) - p1_hrc.get('coarse_fine_consistency', 0.0))*100:+.2f}% |",
        "================================================================================",
    ]
    return "\n".join(lines)


def run_three_stage_single_split(
    split_index: int,
    base_config: dict[str, Any],
    profile: str,
    phase1_epochs: int,
    phase2_epochs: int,
    phase3_epochs: int,
    phase1_loss: str,
    phase2_loss: str,
    phase3_loss: str,
    phase1_lr: float,
    phase2_lr: float,
    phase3_lr: float,
    supcon_w: float,
    result_root: Path = Path("result"),
    dry_run: bool = False,
    ablation_name: str | None = None,
) -> dict[str, Any]:
    """Execute complete 3-Stage pipeline for a single split."""
    config = dict(base_config)
    config["protocol_split_index"] = split_index
    config["task_mode"] = "hierarchical"

    # Locate protocol manifest
    protocol_dir, protocol_sha = find_latest_completed_protocol_run(
        config.get("result_root", "./result"), "research"
    )
    if protocol_dir is None:
        protocol_dir, protocol_sha = find_latest_completed_protocol_run(
            config.get("result_root", "./result"), profile
        )
    if protocol_dir is None:
        raise FileNotFoundError(
            "Stage 00 Protocol manifest not found. Run Stage 00 first (`cystods run 00`)."
        )

    config["protocol_manifest_dir"] = protocol_dir
    config["expected_protocol_sha256"] = protocol_sha

    # Setup run directory
    resolved_result_root = Path(config.get("result_root", "./result")).resolve()
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    if ablation_name:
        run_name = f"{ablation_name}_split{split_index}_{profile}_{run_timestamp}"
        run_dir = resolved_result_root / "40_ablations" / "three_stage" / run_name
    else:
        run_name = f"three_stage_split{split_index}_{profile}_{run_timestamp}"
        run_dir = resolved_result_root / "36_three_stage_decoupled" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("logs", "reports", "system", "splits", "runs", "phase1", "phase2", "phase3"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir / "logs" / "training.log")

    logger.info("=" * 80)
    logger.info("🚀 CYSTODS THREE-STAGE HIERARCHICAL FINE-TUNING (3S-HFT)")
    logger.info("  • Split Index       : %d", split_index)
    logger.info("  • Profile           : %s", profile)
    logger.info("  • Phase 1 (Rep)     : %d epochs (lr=%.1e, loss=%s, supcon=%.2f)", phase1_epochs, phase1_lr, phase1_loss, supcon_w)
    logger.info("  • Phase 2 (Coarse)  : %d epochs (lr=%.1e, loss=%s)", phase2_epochs, phase2_lr, phase2_loss)
    logger.info("  • Phase 3 (Fine)    : %d epochs (lr=%.1e, loss=%s)", phase3_epochs, phase3_lr, phase3_loss)
    logger.info("  • Run Directory     : %s", run_dir)
    logger.info("  • Protocol Directory: %s (SHA: %s)", protocol_dir, protocol_sha[:12])
    logger.info("=" * 80)

    seed = int(config.get("seed", 20260729)) + split_index
    seed_everything(seed, bool(config.get("deterministic", False)))

    device = resolve_device(config)
    _, amp_dtype = resolve_precision(config, device)
    logger.info("Hardware: device=%s, amp_dtype=%s", device, amp_dtype)

    # Load dataset manifest and protocol splits
    manifest_frame, _ = load_and_validate_manifest(config, run_dir, logger)
    units = build_all_protocol_splits(manifest_frame, config, run_dir, logger)
    if not units:
        raise RuntimeError(f"No protocol splits found for split {split_index}.")

    unit_name, split_frames, patient_split = units[0]
    split_hash = split_fingerprint(split_frames)
    logger.info(
        "Protocol split %s materialized: Train=%d, Val=%d, Test=%d images",
        unit_name,
        len(split_frames["train"]),
        len(split_frames["val"]),
        len(split_frames["test"]),
    )

    if dry_run:
        logger.info("[DRY RUN] Verification complete. Skipping training execution.")
        close_logger(logger)
        return {"status": "dry_run_success", "split": split_index, "run_dir": str(run_dir)}

    # Build DataLoaders
    loaders, _ = build_dataloaders(split_frames, config, device, seed)

    # Instantiate Model
    model = TwoStageDecoupledHierarchicalModel(config).to(device)
    param_summary = model.get_parameter_summary()
    logger.info(
        "Model instantiated: total_params=%d, trainable=%d (%.2f%%)",
        param_summary["total_params"],
        param_summary["trainable_params"],
        param_summary["trainable_percentage"],
    )

    # ══════════════════════════════════════════════════════════════════════
    # ▶ GIAI ĐOẠN 1: GENERAL REPRESENTATION LEARNING (FULL BACKBONE)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("▶ BẮT ĐẦU GIAI ĐOẠN 1: GENERAL REPRESENTATION LEARNING")
    logger.info("  • Mục tiêu: Học không gian đặc trưng tối ưu qua Natural Distribution + SupCon (w=%.2f)", supcon_w)
    logger.info("  • Trạng thái Backbone: MỞ 100% (requires_grad = True)")
    logger.info("=" * 70)

    phase1_config = dict(config)
    phase1_config["epochs"] = phase1_epochs
    phase1_config["scheduler_epochs"] = phase1_epochs
    phase1_config["learning_rate"] = phase1_lr
    phase1_config["fine_loss"] = phase1_loss
    phase1_config["supervised_contrastive_loss_weight"] = supcon_w
    phase1_config["binary_coarse_hierarchy_loss_weight"] = 0.25
    phase1_config["coarse_fine_hierarchy_loss_weight"] = 0.25
    phase1_config["monitor_metric"] = "fine_macro_f1"
    phase1_config["early_stopping_patience"] = 6 if profile != "smoke" else 1

    if profile == "smoke":
        phase1_config["fine_inference_calibration_mode"] = "fixed"
        phase1_config["scientific_gate_mode"] = "report"

    phase1_fold_dir = run_dir / "phase1" / unit_name
    phase1_fold_dir.mkdir(parents=True, exist_ok=True)

    t1_start = time.perf_counter()
    phase1_metrics, _, phase1_ckpt = train_model(
        model,
        loaders,
        split_frames,
        split_frames["train"],
        phase1_config,
        device,
        amp_dtype,
        phase1_fold_dir,
        run_dir,
        split_hash,
        logger,
    )
    t1_elapsed = time.perf_counter() - t1_start
    logger.info("Phase 1 hoàn tất trong %.1f giây (%.2f phút). Best checkpoint: %s",
                t1_elapsed, t1_elapsed / 60, phase1_ckpt)

    # ══════════════════════════════════════════════════════════════════════
    # ▶ GIAI ĐOẠN 2: SELECTIVE COARSE ALIGNMENT (COARSE-ONLY)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("▶ BẮT ĐẦU GIAI ĐOẠN 2: SELECTIVE COARSE CLASSIFIER ALIGNMENT")
    logger.info("  • Mục tiêu: Đóng băng Backbone + Khóa Binary & Fine, chỉ tối ưu Coarse Head")
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
    phase2_config["coarse_loss"] = phase2_loss
    phase2_config["supervised_contrastive_loss_weight"] = 0.0
    phase2_config["binary_loss_weight"] = 0.0
    phase2_config["coarse_loss_weight"] = 1.0
    phase2_config["fine_loss_weight"] = 0.0
    phase2_config["binary_coarse_hierarchy_loss_weight"] = 0.25
    phase2_config["coarse_fine_hierarchy_loss_weight"] = 0.0
    phase2_config["monitor_metric"] = "coarse_macro_f1"
    phase2_config["warmup_epochs"] = 0.5 if profile != "smoke" else 0.0
    phase2_config["early_stopping_patience"] = 6 if profile != "smoke" else 1

    if profile == "smoke":
        phase2_config["fine_inference_calibration_mode"] = "fixed"
        phase2_config["scientific_gate_mode"] = "report"

    phase2_fold_dir = run_dir / "phase2" / unit_name
    phase2_fold_dir.mkdir(parents=True, exist_ok=True)

    t2_start = time.perf_counter()
    phase2_metrics, _, phase2_ckpt = train_model(
        model,
        loaders,
        split_frames,
        split_frames["train"],
        phase2_config,
        device,
        amp_dtype,
        phase2_fold_dir,
        run_dir,
        split_hash,
        logger,
    )
    t2_elapsed = time.perf_counter() - t2_start
    logger.info("Phase 2 hoàn tất trong %.1f giây (%.2f phút). Coarse checkpoint: %s",
                t2_elapsed, t2_elapsed / 60, phase2_ckpt)

    # ══════════════════════════════════════════════════════════════════════
    # ▶ GIAI ĐOẠN 3: SELECTIVE FINE ALIGNMENT (FINE-ONLY)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("▶ BẮT ĐẦU GIAI ĐOẠN 3: SELECTIVE FINE CLASSIFIER ALIGNMENT")
    logger.info("  • Mục tiêu: Khóa Coarse Head đã tối ưu, chỉ nắn Fine Head với %s", phase3_loss)
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
    phase3_config["monitor_metric"] = "fine_macro_f1"
    phase3_config["warmup_epochs"] = 0.5 if profile != "smoke" else 0.0
    phase3_config["early_stopping_patience"] = 6 if profile != "smoke" else 1

    if profile == "smoke":
        phase3_config["fine_inference_calibration_mode"] = "fixed"
        phase3_config["scientific_gate_mode"] = "report"

    phase3_fold_dir = run_dir / "phase3" / unit_name
    phase3_fold_dir.mkdir(parents=True, exist_ok=True)

    t3_start = time.perf_counter()
    phase3_metrics, _, phase3_ckpt = train_model(
        model,
        loaders,
        split_frames,
        split_frames["train"],
        phase3_config,
        device,
        amp_dtype,
        phase3_fold_dir,
        run_dir,
        split_hash,
        logger,
    )
    t3_elapsed = time.perf_counter() - t3_start
    logger.info("Phase 3 hoàn tất trong %.1f giây (%.2f phút). Final checkpoint: %s",
                t3_elapsed, t3_elapsed / 60, phase3_ckpt)

    total_elapsed = t1_elapsed + t2_elapsed + t3_elapsed

    four_way_table = generate_four_way_comparison_table(
        phase1_metrics,
        phase2_metrics,
        phase3_metrics,
        split_index,
    )
    logger.info("\n%s\n", four_way_table)

    summary_payload = {
        "status": "success",
        "split_index": split_index,
        "profile": profile,
        "phase1_metrics": phase1_metrics,
        "phase2_coarse_metrics": phase2_metrics,
        "phase3_final_metrics": phase3_metrics,
        "timing": {
            "phase1_seconds": t1_elapsed,
            "phase2_seconds": t2_elapsed,
            "phase3_seconds": t3_elapsed,
            "total_seconds": total_elapsed,
        },
        "four_way_comparison_table": four_way_table,
    }

    summary_file = run_dir / "three_stage_summary.json"
    write_json(summary_file, json_ready(summary_payload))
    logger.info("Báo cáo Three-Stage đã lưu tại: %s", summary_file)

    close_logger(logger)
    return summary_payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    profile = args.profile
    split_str = args.split.strip().lower()

    if split_str == "all":
        split_indices = [0, 1, 2]
    else:
        split_indices = [int(s.strip()) for s in split_str.split(",") if s.strip()]

    # Resolve default epochs based on profile
    if profile == "smoke":
        p1_epochs = args.phase1_epochs if args.phase1_epochs is not None else 1
        p2_epochs = args.phase2_epochs if args.phase2_epochs is not None else 1
        p3_epochs = args.phase3_epochs if args.phase3_epochs is not None else 1
        supcon_w = args.phase1_supcon_weight if args.phase1_supcon_weight is not None else 0.0
    else:
        p1_epochs = args.phase1_epochs if args.phase1_epochs is not None else 25
        p2_epochs = args.phase2_epochs if args.phase2_epochs is not None else 10
        p3_epochs = args.phase3_epochs if args.phase3_epochs is not None else 10
        supcon_w = args.phase1_supcon_weight if args.phase1_supcon_weight is not None else 0.10

    # Build cli_overrides list for load_config
    cli_overrides = list(args.cli_overrides or [])
    if args.batch_size is not None:
        cli_overrides.append(f"runtime.batch_size={args.batch_size}")
    if args.eval_batch_size is not None:
        cli_overrides.append(f"runtime.eval_batch_size={args.eval_batch_size}")
    if args.device != "auto":
        cli_overrides.append(f"runtime.device={args.device}")
    else:
        cli_overrides.append("runtime.device=auto")
    if args.precision != "auto":
        cli_overrides.append(f"runtime.precision={args.precision}")
    else:
        cli_overrides.append("runtime.precision=auto")
    if args.num_workers is not None:
        cli_overrides.append(f"runtime.num_workers={args.num_workers}")
    if args.seed is not None:
        cli_overrides.append(f"project.seed={args.seed}")
    if args.result_root is not None:
        cli_overrides.append(f"paths.result_root={args.result_root}")
    if args.data_root is not None:
        cli_overrides.append(f"paths.data_root={args.data_root}")

    base_config = load_config(
        stage="30",
        profile=args.profile,
        cli_overrides=cli_overrides,
    )

    result_root = Path(base_config.get("result_root", "./result")).resolve()

    results = []
    for s_idx in split_indices:
        res = run_three_stage_single_split(
            split_index=s_idx,
            base_config=base_config,
            profile=profile,
            phase1_epochs=p1_epochs,
            phase2_epochs=p2_epochs,
            phase3_epochs=p3_epochs,
            phase1_loss=args.phase1_loss,
            phase2_loss=args.phase2_loss,
            phase3_loss=args.phase3_loss,
            phase1_lr=args.phase1_lr,
            phase2_lr=args.phase2_lr,
            phase3_lr=args.phase3_lr,
            supcon_w=supcon_w,
            result_root=result_root,
            dry_run=args.dry_run,
            ablation_name=args.ablation_name,
        )
        results.append(res)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("🎉 HOÀN THÀNH TOÀN BỘ THỰC NGHIỆM THREE-STAGE HIERARCHICAL FINE-TUNING!")
    print("=" * 80)


if __name__ == "__main__":
    main()
