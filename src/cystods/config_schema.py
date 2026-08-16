"""Config schema definitions for CystoDS.

Provides the canonical ``BASE_CONFIG`` and valid keys set ``CORE_CONFIG_KEYS``
without importing ``cystods.core`` façade.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _resolve_default_data_root() -> Path:
    env_val = os.environ.get("CYSTODS_DATA_ROOT")
    if env_val:
        return Path(env_val).expanduser().resolve()
    return Path(".").resolve()


def _resolve_default_result_root() -> Path:
    env_val = os.environ.get("CYSTODS_RESULT_ROOT")
    if env_val:
        return Path(env_val).expanduser().resolve()
    return Path("./result").resolve()


BASE_CONFIG: dict[str, Any] = {
    "schema_version": "cystods.core.v2",
    "stage_name": "legacy_proposed_method",
    "study_id": "cystods_hierarchical_2026",
    # Paths and experiment identity
    "data_root": _resolve_default_data_root(),
    "metadata_csv": _resolve_default_data_root() / "cystods.csv",
    "image_dir": _resolve_default_data_root() / "images",
    "segmentation_dir": _resolve_default_data_root() / "segmentations",
    "inclusion_manifest_csv": None,
    "inclusion_manifest_filename_column": "filename",
    "result_root": _resolve_default_result_root(),
    "experiment_name": "cystods_hierarchical_long_tailed",
    "run_profile": os.environ.get("CYSTODS_RUN_PROFILE", "research"),
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
    "protocol_split_index": None,
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
    "device": "cuda",
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
    "partial_finetune": False,
    "freeze_early_layers": False,
    "frozen_stages_count": 2,
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

CORE_CONFIG_KEYS: frozenset[str] = frozenset(BASE_CONFIG.keys())
