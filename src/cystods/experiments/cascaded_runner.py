"""Cascaded Late-Stage Stacked Multi-Head Classifier Experiment Runner.

Orchestrates 1-Stage and Decoupled training for the Late-Stage Cascaded Architecture:
  - Binary Head at Stage 4 (768-d)
  - Coarse Head conditioned on [Stage 4, Binary Probs & Embedding]
  - Fine Head conditioned on [Stage 4, Binary Context, Coarse Context]
  - Bidirectional Hierarchical Marginalization for robust inference

Provides a clean standalone CLI for local execution and Kaggle Notebooks.
"""

from __future__ import annotations

import argparse
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
from cystods.models.cascaded_hierarchical import CascadedHierarchicalCystoModel
from cystods.science import split_fingerprint
from cystods.training.engine import train_model
from cystods.training.runtime import resolve_device, resolve_precision, seed_everything


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cystods-cascaded",
        description=(
            "CystoDS: Cascaded Late-Stage Multi-Head Experiment Runner.\n"
            "Stacks Binary -> Coarse -> Fine heads with conditional embeddings at Stage 4."
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
        help="Execution profile: 'research' (full 224x224, 25 epochs) or 'smoke' (fast debug). Default: research.",
    )
    parser.add_argument(
        "--phases",
        type=int,
        default=1,
        choices=[1, 3],
        help="Training mode: 1 (1-Stage End-to-End) or 3 (3-Phase Sequential Decoupled Alignment). Default: 1.",
    )
    parser.add_argument(
        "--detach-hierarchy",
        action="store_true",
        default=False,
        help="Detach conditioning context gradients between heads to prevent Fine Loss from contaminating Binary Head.",
    )
    parser.add_argument(
        "--hierarchy-lambda",
        type=float,
        default=0.25,
        help="Marginalization weight lambda in [0, 1] for P_ens(Coarse). Default: 0.25.",
    )
    parser.add_argument(
        "--fine-loss",
        type=str,
        default="balanced_softmax_smoothed",
        choices=["balanced_softmax_smoothed", "balanced_softmax", "cross_entropy", "focal", "logit_adjustment"],
        help="Fine classification loss function. Default: balanced_softmax_smoothed.",
    )
    parser.add_argument(
        "--coarse-loss",
        type=str,
        default="balanced_softmax_smoothed",
        choices=["balanced_softmax_smoothed", "balanced_softmax", "cross_entropy", "focal"],
        help="Coarse classification loss function. Default: balanced_softmax_smoothed.",
    )
    parser.add_argument(
        "--supcon-weight",
        type=float,
        default=0.10,
        help="Supervised Contrastive loss weight in Phase 1 (0.0 to disable, default: 0.10).",
    )
    parser.add_argument(
        "--hierarchy-schedule",
        type=str,
        default="fixed",
        choices=["fixed", "warmup", "curriculum", "two_phase"],
        help="Hierarchy loss schedule: 'fixed' (constant w=0.25) or 'warmup'/'curriculum' (linear ramp-up from 0.0 to 0.25 over warmup epochs). Default: fixed.",
    )
    parser.add_argument(
        "--hierarchy-warmup-epochs",
        type=int,
        default=12,
        help="Number of epochs for hierarchy loss curriculum warmup (default: 12).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override max training epochs (default from config/profile).",
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
        "--lr",
        "--learning-rate",
        type=float,
        default=None,
        dest="learning_rate",
        help="Override base learning rate (e.g. 0.0003).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'cuda', 'mps', 'cpu', or 'auto'. Default: auto.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory. Default: result/40_ablations/cascaded_stacked/.",
    )
    return parser


def run_cascaded_single_split(
    split_idx: int,
    config: Mapping[str, Any],
    logger: logging.Logger,
    result_run_dir: Path,
    *,
    phases: int = 1,
    detach_hierarchy: bool = False,
    hierarchy_lambda: float = 0.25,
    fine_loss: str = "balanced_softmax_smoothed",
    coarse_loss: str = "balanced_softmax_smoothed",
    supcon_weight: float = 0.10,
    hierarchy_schedule: str = "fixed",
    hierarchy_warmup_epochs: int = 12,
) -> dict[str, Any]:
    """Train and evaluate the CascadedHierarchicalCystoModel on a single split (1-Stage or 3-Phase)."""
    device = resolve_device(config)
    precision_mode, amp_dtype = resolve_precision(config, device)
    seed = int(config.get("seed", 42)) + split_idx
    seed_everything(seed, bool(config.get("deterministic", True)))

    split_dir = result_run_dir / "splits" / f"split_{split_idx}"
    split_dir.mkdir(parents=True, exist_ok=True)
    run_split_dir = result_run_dir / "runs" / f"cascaded_{phases}phase" / f"split_{split_idx}"
    run_split_dir.mkdir(parents=True, exist_ok=True)
    (run_split_dir / "reports").mkdir(parents=True, exist_ok=True)
    (run_split_dir / "system").mkdir(parents=True, exist_ok=True)

    # 1. Build and validate data manifest & protocol splits
    manifest_frame, _ = load_and_validate_manifest(config, run_split_dir, logger)
    units = build_all_protocol_splits(manifest_frame, config, run_split_dir, logger)
    if not units or split_idx >= len(units):
        raise RuntimeError(f"Protocol split {split_idx} not found in {len(units)} available splits.")

    unit_name, split_frames, patient_split = units[split_idx]
    train_df = split_frames["train"].copy()
    val_df = split_frames["val"].copy()
    test_df = split_frames["test"].copy()

    train_df.to_csv(split_dir / "train.csv", index=False)
    val_df.to_csv(split_dir / "val.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)

    split_hash = split_fingerprint(split_frames)
    logger.info(
        "Protocol split %s materialized: Train=%d, Val=%d, Test=%d images (Fingerprint: %s)",
        unit_name, len(train_df), len(val_df), len(test_df), split_hash[:12]
    )

    # 2. Configure model dictionary
    model_cfg = dict(config)
    model_cfg["task_mode"] = "hierarchical"
    model_cfg["detach_hierarchy"] = detach_hierarchy
    model_cfg["hierarchy_lambda"] = hierarchy_lambda
    model_cfg["fine_loss"] = fine_loss
    model_cfg["coarse_loss"] = coarse_loss
    model_cfg["supervised_contrastive_loss_weight"] = supcon_weight
    model_cfg["binary_loss_weight"] = 1.0
    model_cfg["coarse_loss_weight"] = 1.0
    model_cfg["fine_loss_weight"] = 1.0
    model_cfg["hierarchy_schedule"] = hierarchy_schedule
    model_cfg["hierarchy_warmup_epochs"] = hierarchy_warmup_epochs
    model_cfg["binary_coarse_hierarchy_loss_weight"] = 0.25
    model_cfg["coarse_fine_hierarchy_loss_weight"] = 0.25

    logger.info("Initializing CascadedHierarchicalCystoModel (Phases=%d, Detach=%s, Lambda=%.2f, Schedule=%s)...", phases, detach_hierarchy, hierarchy_lambda, hierarchy_schedule)
    model = CascadedHierarchicalCystoModel(model_cfg).to(device)
    param_summary = model.get_parameter_summary()
    logger.info(
        "Model parameters: total=%d, trainable=%d (%.2f%%)",
        param_summary["total_params"],
        param_summary["trainable_params"],
        param_summary["trainable_percentage"],
    )

    # 3. Build data loaders
    loaders, _ = build_dataloaders(split_frames, config, device, seed)

    start_time = time.time()

    if phases == 1:
        # ══════════════════════════════════════════════════════════════════════
        # 1-STAGE END-TO-END TRAINING (1 PASS)
        # ══════════════════════════════════════════════════════════════════════
        logger.info("Starting 1-Stage Training for Split %d (Detach=%s, Schedule=%s)...", split_idx, detach_hierarchy, hierarchy_schedule)
        eval_metrics, splits_out, best_checkpoint_path = train_model(
            model=model,
            loaders=loaders,
            split_frames=split_frames,
            optimization_train_frame=train_df,
            config=model_cfg,
            device=device,
            amp_dtype=amp_dtype,
            fold_dir=run_split_dir,
            run_dir=result_run_dir,
            data_split_hash=split_hash,
            logger=logger,
        )
    else:
        # ══════════════════════════════════════════════════════════════════════
        # 3-PHASE SEQUENTIAL DECOUPLED TRAINING
        # ══════════════════════════════════════════════════════════════════════
        logger.info("▶ BẮT ĐẦU 3-PHASE SEQUENTIAL DECOUPLED TRAINING CHO CASCADED MODEL")

        # Phase 1: Representation Learning
        p1_cfg = dict(model_cfg)
        p1_cfg["fine_loss"] = "cross_entropy"
        p1_cfg["coarse_loss"] = "cross_entropy"
        p1_cfg["hierarchy_schedule"] = hierarchy_schedule
        p1_cfg["hierarchy_warmup_epochs"] = hierarchy_warmup_epochs
        p1_fold_dir = run_split_dir / "phase1"
        p1_fold_dir.mkdir(parents=True, exist_ok=True)
        model.configure_phase_freezing(1)
        logger.info(">>> [Phase 1] Representation Learning (100% Backbone + Heads Trainable, Schedule=%s)...", hierarchy_schedule)
        p1_metrics, _, p1_ckpt = train_model(
            model=model,
            loaders=loaders,
            split_frames=split_frames,
            optimization_train_frame=train_df,
            config=p1_cfg,
            device=device,
            amp_dtype=amp_dtype,
            fold_dir=p1_fold_dir,
            run_dir=result_run_dir,
            data_split_hash=split_hash,
            logger=logger,
        )

        # Phase 2: Coarse Alignment (Freeze Backbone, Binary, Fine; ONLY Coarse Head)
        p2_cfg = dict(model_cfg)
        p2_cfg["epochs"] = 10 if config.get("run_profile") != "smoke" else 1
        p2_cfg["learning_rate"] = 0.001
        p2_cfg["binary_loss_weight"] = 1e-7
        p2_cfg["coarse_loss_weight"] = 1.0
        p2_cfg["fine_loss_weight"] = 1e-7
        p2_cfg["supervised_contrastive_loss_weight"] = 0.0
        p2_cfg["coarse_loss"] = coarse_loss
        p2_cfg["hierarchy_schedule"] = "fixed"
        p2_fold_dir = run_split_dir / "phase2"
        p2_fold_dir.mkdir(parents=True, exist_ok=True)
        model.configure_phase_freezing(2)
        logger.info(">>> [Phase 2] Coarse Alignment (Freeze Backbone & Binary, Train Coarse Head)...")
        p2_metrics, _, p2_ckpt = train_model(
            model=model,
            loaders=loaders,
            split_frames=split_frames,
            optimization_train_frame=train_df,
            config=p2_cfg,
            device=device,
            amp_dtype=amp_dtype,
            fold_dir=p2_fold_dir,
            run_dir=result_run_dir,
            data_split_hash=split_hash,
            logger=logger,
        )

        # Phase 3: Fine Alignment (Freeze Backbone, Binary, Coarse; ONLY Fine Head)
        p3_cfg = dict(model_cfg)
        p3_cfg["epochs"] = 10 if config.get("run_profile") != "smoke" else 1
        p3_cfg["learning_rate"] = 0.001
        p3_cfg["binary_loss_weight"] = 1e-7
        p3_cfg["coarse_loss_weight"] = 1e-7
        p3_cfg["fine_loss_weight"] = 1.0
        p3_cfg["supervised_contrastive_loss_weight"] = 0.0
        p3_cfg["fine_loss"] = fine_loss
        p3_cfg["hierarchy_schedule"] = "fixed"
        p3_fold_dir = run_split_dir / "phase3"
        p3_fold_dir.mkdir(parents=True, exist_ok=True)
        model.configure_phase_freezing(3)
        logger.info(">>> [Phase 3] Fine Alignment (Freeze Backbone, Binary & Coarse, Train Fine Head)...")
        p3_metrics, _, p3_ckpt = train_model(
            model=model,
            loaders=loaders,
            split_frames=split_frames,
            optimization_train_frame=train_df,
            config=p3_cfg,
            device=device,
            amp_dtype=amp_dtype,
            fold_dir=p3_fold_dir,
            run_dir=result_run_dir,
            data_split_hash=split_hash,
            logger=logger,
        )

        eval_metrics = p3_metrics
        best_checkpoint_path = p3_ckpt

    elapsed = time.time() - start_time
    logger.info("Training completed in %.2f seconds.", elapsed)

    # 5. Extract evaluation metrics
    split_result = {
        "split_index": split_idx,
        "detach_hierarchy": detach_hierarchy,
        "hierarchy_lambda": hierarchy_lambda,
        "hierarchy_schedule": hierarchy_schedule,
        "elapsed_seconds": elapsed,
        "eval_metrics": eval_metrics,
        "best_checkpoint": str(best_checkpoint_path),
        "param_summary": param_summary,
    }
    write_json(run_split_dir / "cascaded_summary.json", json_ready(split_result))
    return split_result


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()

    cli_overrides: list[str] = [
        "model.task_mode=hierarchical",
        f"model.detach_hierarchy={str(args.detach_hierarchy).lower()}",
        f"model.hierarchy_lambda={args.hierarchy_lambda}",
        f"training.fine_loss={args.fine_loss}",
        f"training.coarse_loss={args.coarse_loss}",
        f"training.supervised_contrastive_loss_weight={args.supcon_weight}",
        f"training.hierarchy_schedule={args.hierarchy_schedule}",
        f"training.hierarchy_warmup_epochs={args.hierarchy_warmup_epochs}",
    ]
    if args.epochs is not None:
        cli_overrides.append(f"training.epochs={args.epochs}")
    if args.batch_size is not None:
        cli_overrides.append(f"runtime.batch_size={args.batch_size}")
    if args.eval_batch_size is not None:
        cli_overrides.append(f"runtime.eval_batch_size={args.eval_batch_size}")
    if args.learning_rate is not None:
        cli_overrides.append(f"training.learning_rate={args.learning_rate}")
    if args.device != "auto":
        cli_overrides.append(f"runtime.device={args.device}")

    config = load_config(stage="30", profile=args.profile, cli_overrides=cli_overrides)
    config["detach_hierarchy"] = args.detach_hierarchy
    config["hierarchy_lambda"] = args.hierarchy_lambda
    config["fine_loss"] = args.fine_loss
    config["coarse_loss"] = args.coarse_loss
    config["supervised_contrastive_loss_weight"] = args.supcon_weight
    config["hierarchy_schedule"] = args.hierarchy_schedule
    config["hierarchy_warmup_epochs"] = args.hierarchy_warmup_epochs

    split_arg = str(args.split).strip().lower()
    if split_arg == "all":
        split_indices = [0, 1, 2]
    else:
        split_indices = [int(s.strip()) for s in split_arg.split(",") if s.strip()]

    if args.output_dir:
        result_dir = Path(args.output_dir)
    else:
        if args.hierarchy_schedule in ("warmup", "curriculum"):
            sched_tag = "warmup"
        elif args.hierarchy_schedule in ("two_phase",):
            sched_tag = "twophase"
        else:
            sched_tag = ""

        if args.phases == 3:
            mode_suffix = f"3phase_{sched_tag}" if sched_tag else "3phase"
        elif args.detach_hierarchy:
            mode_suffix = f"1stage_detached_{sched_tag}" if sched_tag else "1stage_detached"
        else:
            mode_suffix = f"1stage_{sched_tag}" if sched_tag else "1stage_end_to_end"

        ts = utc_now_iso().replace(":", "").replace("-", "")[:15]
        split_clean = split_arg.replace(",", "_")
        if split_arg != "all":
            result_dir = Path(f"result/40_ablations/cascaded_{mode_suffix}/research_{ts}_split{split_clean}")
        else:
            result_dir = Path(f"result/40_ablations/cascaded_{mode_suffix}/research_{ts}")

    result_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(result_dir / "cascaded_experiment.log")

    logger.info("=" * 70)
    logger.info("▶ CYSTODS: CASCADED LATE-STAGE STACKED MULTI-HEAD RUNNER")
    logger.info("=" * 70)
    logger.info("Profile: %s | Phases: %d | Detach: %s | Schedule: %s | Lambda: %.2f | Fine Loss: %s", args.profile, args.phases, args.detach_hierarchy, args.hierarchy_schedule, args.hierarchy_lambda, args.fine_loss)
    logger.info("Output directory: %s", result_dir)

    all_results = []
    for s_idx in split_indices:
        logger.info("\n------------------------------------------------------------")
        logger.info(">>> Processing Split %d / %s", s_idx, split_indices)
        logger.info("------------------------------------------------------------")
        res = run_cascaded_single_split(
            split_idx=s_idx,
            config=config,
            logger=logger,
            result_run_dir=result_dir,
            phases=args.phases,
            detach_hierarchy=args.detach_hierarchy,
            hierarchy_lambda=args.hierarchy_lambda,
            fine_loss=args.fine_loss,
            coarse_loss=args.coarse_loss,
            supcon_weight=args.supcon_weight,
            hierarchy_schedule=args.hierarchy_schedule,
            hierarchy_warmup_epochs=args.hierarchy_warmup_epochs,
        )
        all_results.append(res)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate and print summary table
    summary_path = result_dir / "experiment_summary.json"
    write_json(summary_path, json_ready({"runs": all_results}))
    logger.info("\n" + "=" * 70)
    logger.info("🎉 CASCADED EXPERIMENT COMPLETED SUCCESSFULLY!")
    logger.info("Results saved to: %s", summary_path)
    logger.info("=" * 70)
    close_logger(logger)


if __name__ == "__main__":
    main()
