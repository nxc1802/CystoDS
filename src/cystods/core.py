# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# SHARED CORE CELL 1 - Runtime dependencies & Façade module.
"""Backward-compatibility façade module for CystoDS.

All core functionality has been modularized under ``cystods.*``.
This module re-exports all public APIs to ensure existing scripts and tests
continue working without modification.
"""

from __future__ import annotations

import os
from pathlib import Path

import timm
from cystods.config import (
    PROFILE_OVERRIDES,
    PROPOSED_CANONICAL_CONFIG,
    resolve_dataset_root,
)
from cystods.infra.environment import (
    FlushFileHandler,
    close_logger,
    setup_logger,
)
from cystods.infra.serialization import (
    json_ready,
    sha256_file,
    stable_int_seed,
    utc_now_iso,
    write_json,
)

# ── Taxonomy ──────────────────────────────────────────────────────────────
from cystods.taxonomy import (
    BINARY_NAMES,
    COARSE_BY_ID,
    COARSE_ID_BY_NAME,
    COARSE_NAMES,
    COARSE_TO_ID,
    FINE_BY_ID,
    FINE_BY_PARENT,
    FINE_ID_BY_NAME,
    FINE_NAMES,
    FINE_PARENT_ID,
    FINE_PARENT_NAME,
    FINE_TO_COARSE_ID,
    FINE_TO_ID,
    coarse_id_from_subclass,
    coarse_name_from_subclass,
    fine_id_from_subclass,
)

# ── Data Audit, Manifest & Transforms ──────────────────────────────────────
from cystods.data.audit import (
    audit_image_size_distribution,
)
from cystods.data.dataset import (
    CystoDataset,
    ExternalBinaryDataset,
)
from cystods.data.manifest import (
    load_and_validate_manifest,
    snapshot_source_files,
    validate_source_files,
)
from cystods.data.sampler import (
    _WorkerInitFn,
    build_dataloaders,
    build_sample_weights,
    make_worker_init_fn,
)
from cystods.data.splits import (
    _load_and_validate_protocol_binding,
    allocation_score,
    build_all_protocol_splits,
    cap_normal_mucosa_across_splits,
    find_latest_completed_protocol_run,
    fixed_patient_split,
    load_frozen_protocol_splits,
    materialize_split_frames,
    patient_label_matrices,
    sample_rows_stratified,
    save_split_artifacts,
    search_holdout_patient_split,
    search_patient_folds,
    search_train_val_patient_split,
    validate_materialized_splits,
)
from cystods.science import split_fingerprint
from cystods.training.checkpoint import (
    _canonical_model_state_dict,
    _hf_checkpoint_config,
    _hf_checkpoint_path_in_repo,
    load_checkpoint_for_resume,
    save_checkpoint,
)
from cystods.data.transforms import (
    build_transforms,
)

# ── Models ────────────────────────────────────────────────────────────────
from cystods.config import (
    make_run_directory,
    normalize_core_config,
    validate_config,
)
from cystods.models.factory import (
    active_tasks_for_mode,
    active_tasks_from_config,
)
from cystods.models.hierarchical import (
    HierarchicalCystoModel,
)
from cystods.evaluation.roi import GatedAttentionMIL

# ── Losses ────────────────────────────────────────────────────────────────
from cystods.losses.classification import (
    FineLongTailLoss,
)
from cystods.losses.composite import (
    active_fine_loss_name,
    binary_coarse_hierarchy_loss,
    coarse_fine_hierarchy_loss,
    compute_multitask_loss,
)
from cystods.losses.supcon import (
    supervised_contrastive_loss,
)

# ── Evaluation & Science ──────────────────────────────────────────────────
from cystods.evaluation.bootstrap import (
    paired_mcnemar_test,
    patient_bootstrap_intervals,
)
from cystods.evaluation.calibration import (
    apply_fine_inference_calibration,
)
from cystods.evaluation.metrics import (
    compute_binary_metrics,
    compute_metrics_bundle,
    compute_multiclass_metrics,
    metric_for_monitor,
    per_class_metrics,
)
from cystods.science import (
    audit_rare_class_collapse,
    binary_coarse_taxonomy_metrics,
    build_smoothed_class_prior,
    coarse_fine_taxonomy_metrics,
    compute_fixed_primary_metrics,
    enforce_rare_class_collapse_gate,
    hierarchical_composite_monitor,
    mask_inactive_logits,
    semantic_fingerprint,
    validate_active_class_targets,
)
from cystods.evaluation.roi import (
    run_roi_evaluation,
    train_attention_mil,
)

# ── Training & Runtime ────────────────────────────────────────────────────
from cystods.training.checkpoint import (
    load_checkpoint_for_resume,
    save_checkpoint,
)
from cystods.training.engine import (
    evaluate_model,
    move_images,
    move_target,
    prediction_rows_from_outputs,
    run_training_suite,
    train_model,
)
from cystods.training.optimizer import (
    build_optimizer,
    build_scheduler,
)
from cystods.training.runtime import (
    collect_system_info,
    resolve_device,
    resolve_precision,
    seed_everything,
)

# ── Reports & Visualization ───────────────────────────────────────────────
from cystods.reports import (
    aggregate_cross_validation_metrics,
    build_fold_report,
    markdown_scalar,
    metrics_table_markdown,
    serialize_prediction_frame,
    write_artifact_manifest,
)
from cystods.visualization import (
    export_fold_visualizations,
    generate_sample_grid,
    plot_binary_curves,
    plot_class_distributions,
    plot_confusion,
    plot_per_class_recall,
    plot_training_history,
)

# ── Experiments & Stage Runners ───────────────────────────────────────────
from cystods.experiments import (
    evaluate_external_binary,
    main,
    make_deterministic_eval_loader,
    run_external_validation_stage,
    run_protocol_stage,
    run_single_fold,
)
from cystods.experiments.runner import (
    _complete_stage_source_files,
    _validate_stage_config_keys,
    _write_stage_system_artifacts,
)

DEPENDENCY_AUDIT = {"mode": "package_install"}

RUN_PROFILE = os.environ.get("CYSTODS_RUN_PROFILE", "research")
if RUN_PROFILE not in {"research", "smoke"}:
    raise ValueError("CYSTODS_RUN_PROFILE must be 'research' or 'smoke'.")

def _resolve_data_root() -> Path:
    return resolve_dataset_root()

def _resolve_result_root() -> Path:
    if "CYSTODS_RESULT_ROOT" in os.environ:
        return Path(os.environ["CYSTODS_RESULT_ROOT"]).expanduser().resolve()
    if Path("/kaggle/working").exists():
        return (Path("/kaggle/working") / "result").resolve()
    return (Path.cwd() / "result").resolve()

DATA_ROOT = _resolve_data_root()
RESULT_ROOT = _resolve_result_root()

BASE_CONFIG = {
    "schema_version": "cystods.core.v2",
    "stage_name": "legacy_proposed_method",
    "study_id": "cystods_hierarchical_2026",
    # Paths and experiment identity
    "data_root": DATA_ROOT,
    "metadata_csv": DATA_ROOT / "cystods.csv",
    "image_dir": DATA_ROOT / "images",
    "segmentation_dir": DATA_ROOT / "segmentations",
    "inclusion_manifest_csv": None,
    "inclusion_manifest_filename_column": "filename",
    "result_root": RESULT_ROOT,
    "experiment_name": "cystods_hierarchical_long_tailed",
    "run_profile": RUN_PROFILE,
    # Reproducibility and validation
    "seed": 20260729,
    "deterministic": True,
    "verify_all_image_decodes": True,
    "verify_segmentation_inventory": True,
    "dataset_fingerprint_mode": "full",  # full | semantic
    "verify_exact_duplicate_images": True,
    "num_cpu_threads": 4,
    # Protocol and patient-disjoint splitting
    "protocol": "holdout",  # holdout | cross_validation
    "train_fraction": 0.70,
    "val_fraction": 0.15,
    "test_fraction": 0.15,
    "split_seed": 20260729,
    "split_search_candidates": 2048,
    "force_fine_labels_with_fewer_than_n_patients_to_train": 3,
    "cv_folds": 5,
    "cv_val_fraction_of_remaining": 0.15,
    "cv_run_fold_indices": None,
    "normal_mucosa_limit": 540,
    "fixed_split_pids": None,
    "protocol_manifest_dir": None,
    "protocol_reference_sha256": None,
    "expected_dataset_semantic_sha256": None,
    "protocol_role": None,
    "evaluation_scope": "legacy",
    "suite_trial_id": None,
    "filter_models": None,
    "filter_trials": None,
    "max_train_samples": None,
    "max_val_samples": None,
    "max_test_samples": None,
    # Input pipeline and augmentation
    "image_size": 224,
    "fov_center_crop_ratio": 0.92,
    "random_resized_crop_scale": (0.75, 1.0),
    "horizontal_flip_probability": 0.5,
    "vertical_flip_probability": 0.5,
    "rotation_degrees": 15,
    "color_jitter": (0.20, 0.20, 0.15, 0.05),
    "random_erasing_probability": 0.20,
    "imagenet_mean": (0.485, 0.456, 0.406),
    "imagenet_std": (0.229, 0.224, 0.225),
    "batch_size": 32,
    "eval_batch_size": 64,
    "num_workers": 4,
    "eval_num_workers": 4,
    "prefetch_factor": 2,
    "eval_prefetch_factor": 2,
    "persistent_workers": True,
    "pin_memory": True,
    # Single representative backbone
    "model_name": "swin_tiny_patch4_window7_224.ms_in1k",
    "pretrained": True,
    "task_mode": "hierarchical",
    "dropout": 0.20,
    "projection_dim": 128,
    # Objective weights
    "binary_loss_weight": 1.0,
    "coarse_loss_weight": 1.0,
    "fine_loss_weight": 1.0,
    "binary_coarse_hierarchy_loss_weight": 0.25,
    "coarse_fine_hierarchy_loss_weight": 0.25,
    "supervised_contrastive_loss_weight": 0.10,
    "supervised_contrastive_temperature": 0.10,
    "supervised_contrastive_label_level": "fine",
    # Long-tail objective and sampling
    "fine_loss": "balanced_softmax",
    "class_balance_beta": 0.9999,
    "focal_gamma": 2.0,
    "focal_use_class_balance": False,
    "use_data_augmentation": False,
    "logit_adjustment_tau": 0.5,
    "fine_prior_source": "patient_count",
    "fine_prior_smoothing_alpha": 1.0,
    "fine_prior_power": 0.5,
    "fine_prior_max_ratio": 50.0,
    "fine_absent_train_policy": "mask_and_score_zero",
    "fine_inference_calibration_mode": "validation_grid",
    "fine_inference_prior_tau": 0.0,
    "fine_inference_tau_grid": (0.0, 0.25, 0.5, 0.75, 1.0),
    "fine_inference_calibration_metric": "primary_macro_f1_all_classes",
    "ldam_max_margin": 0.5,
    "ldam_scale": 30.0,
    # Class-imbalance sampler
    "sampler": "random",  # random | class_balanced | progressive
    "sampler_label_level": "fine",  # fine | coarse
    # Optimization
    "epochs": 25,
    "learning_rate": 3.0e-4,
    "encoder_learning_rate_multiplier": 0.25,
    "weight_decay": 0.05,
    "optimizer": "adamw",  # adamw | sgd
    "use_fused_optimizer": True,
    "warmup_epochs": 2.0,
    "scheduler_epochs": 25,
    "minimum_learning_rate_ratio": 0.01,
    "gradient_accumulation_steps": 1,
    "gradient_clip_norm": 1.0,
    "early_stopping_patience": 6,
    "checkpoint_min_delta": 1.0e-4,
    "monitor_metric": "hierarchical_composite",
    "hierarchical_composite_weights": {
        "coarse_macro_f1_all_classes": 0.35,
        "primary_macro_f1_all_classes": 0.45,
        "hierarchical_accuracy": 0.20,
    },
    # Mixed precision, compiler, and threading
    "precision": "bf16",  # fp32 | fp16 | bf16
    "enable_tf32": True,
    "channels_last": True,
    "torch_compile": False,
    "torch_compile_mode": "default",
    "float32_matmul_precision": "high",  # highest | high | medium
    "log_every_n_steps": 20,
    # Evaluation, CIs, and scientific reporting
    "binary_decision_threshold": 0.5,
    "bootstrap_iterations": 1000,
    "bootstrap_confidence": 0.95,
    "probability_sum_tolerance": 5.0e-3,
    "primary_fine_min_train_patients": 10,
    "fixed_primary_fine_class_ids": None,
    "tail_class_max_train_samples": 20,
    "scientific_gate_mode": "enforce",  # enforce | warn
    "generate_sample_grid": False,
    "sample_grid_images_per_class": 3,
    "paired_baseline_predictions_csv": None,
    "rare_gate_max_train_patients": 2,
    "rare_gate_absolute_pred_share": 0.10,
    "rare_gate_prior_multiplier": 10.0,
    "rare_gate_min_pred_count": 5,
    # ROI-level bag-of-frames evaluation and MIL head
    "evaluate_roi_level": False,
    "roi_aggregations": ("mean", "vote", "attention"),
    "roi_conflict_policy": "exclude_and_report",
    "roi_attention_epochs": 25,
    "roi_attention_learning_rate": 1.0e-3,
    "roi_attention_hidden_dim": 128,
    "roi_attention_early_stopping_patience": 5,
    # Modality-restricted experiments
    "train_modality": "all",  # all | WLC
    "evaluate_wlc_only": False,
    # External binary validation
    "external_validation_enabled": False,
    "external_manifest_csv": None,
    "external_image_root": None,
    "external_path_column": "path",
    "external_binary_label_column": "binary_label",
    "external_patient_id_column": "patient_id",
    # Verification and model checkpointing
    "checkpoint_backend": "local",  # local | huggingface
    "save_last_checkpoint": False,
    "save_epoch_checkpoints": False,
    "resume_checkpoint": None,
    "hf_repo_id": None,
    "hf_revision": "main",
    "hf_private": True,
    "hf_create_repo": True,
    "hf_path_prefix": "cystods_hierarchical_2026/cystods_core",
    "hf_token_env": "HF_TOKEN",
}

CONFIG = dict(BASE_CONFIG)

# %%
if __name__ == "__main__":
    COMPLETED_RUN_DIRECTORY = main()
    print(f"Completed CystoDS run: {COMPLETED_RUN_DIRECTORY}")
