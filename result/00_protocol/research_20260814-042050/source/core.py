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
    search_top_k_diverse_holdout_splits,
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

from cystods.config_schema import BASE_CONFIG

CONFIG = dict(BASE_CONFIG)

# %%
if __name__ == "__main__":
    COMPLETED_RUN_DIRECTORY = main()
    print(f"Completed CystoDS run: {COMPLETED_RUN_DIRECTORY}")
