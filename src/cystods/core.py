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
# SHARED CORE CELL 1 - Runtime dependencies.
#
# When running as a package (pip install -e .), dependencies are managed by
# pyproject.toml. The bootstrap below only runs in notebook/direct-script mode.
import importlib as _importlib
import importlib.metadata as _metadata
import importlib.util as _importlib_util
import subprocess as _subprocess
import sys as _sys
_RUNNING_AS_PACKAGE = _importlib_util.find_spec("cystods") is not None

if not _RUNNING_AS_PACKAGE:
    if _importlib_util.find_spec("pip") is None:
        _subprocess.check_call(
            [_sys.executable, "-m", "ensurepip", "--upgrade"]
        )

    try:
        from packaging.requirements import Requirement as _Requirement
    except ModuleNotFoundError:
        _subprocess.check_call(
            [
                _sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "packaging>=24,<27",
            ]
        )
        from packaging.requirements import Requirement as _Requirement

    DEPENDENCIES = [
        "torch>=2.0,<3",
        "torchvision>=0.15,<1",
        "timm>=0.9,<2",
        "numpy>=1.22,<3",
        "pandas>=2.0,<4",
        "scikit-learn>=1.2,<2",
        "scipy>=1.10,<2",
        "Pillow>=9.0,<13",
        "matplotlib>=3.6,<4",
        "seaborn>=0.12,<1",
        "tqdm>=4.60,<5",
        "psutil>=5.9,<8",
        "tabulate>=0.9,<1",
        "packaging>=23,<27",
        "jupytext>=1.15,<2",
        "huggingface_hub>=0.28,<2",
    ]

    _IMPORT_NAMES = {
        "pillow": "PIL",
        "scikit-learn": "sklearn",
    }

    def _unsatisfied_dependencies() -> list[str]:
        unsatisfied: list[str] = []
        for spec in DEPENDENCIES:
            requirement = _Requirement(spec)
            try:
                installed = _metadata.version(requirement.name)
            except _metadata.PackageNotFoundError:
                unsatisfied.append(spec)
                continue
            if requirement.specifier and not requirement.specifier.contains(
                installed,
                prereleases=True,
            ):
                unsatisfied.append(spec)
                continue
            import_name = _IMPORT_NAMES.get(
                requirement.name.lower(),
                requirement.name.lower().replace("-", "_"),
            )
            if _importlib_util.find_spec(import_name) is None:
                unsatisfied.append(spec)
        return unsatisfied

    def _installed_distribution_version(distribution_name: str) -> str | None:
        try:
            return _metadata.version(distribution_name)
        except _metadata.PackageNotFoundError:
            return None

    _dependency_versions_before = {
        _Requirement(spec).name: _installed_distribution_version(
            _Requirement(spec).name
        )
        for spec in DEPENDENCIES
    }
    _to_install = _unsatisfied_dependencies()
    if _to_install:
        _subprocess.check_call(
            [
                _sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                *_to_install,
            ]
        )
    _still_unsatisfied = _unsatisfied_dependencies()
    if _still_unsatisfied:
        raise RuntimeError(
            "Dependencies remain unsatisfied after installation: "
            f"{_still_unsatisfied}"
        )
    for _spec in DEPENDENCIES:
        _requirement = _Requirement(_spec)
        _module_name = _IMPORT_NAMES.get(
            _requirement.name.lower(),
            _requirement.name.lower().replace("-", "_"),
        )
        _importlib.import_module(_module_name)
    _pip_check = _subprocess.run(
        [_sys.executable, "-m", "pip", "check"],
        check=True,
        capture_output=True,
        text=True,
    )
    DEPENDENCY_AUDIT = {
        "declared_requirements": DEPENDENCIES,
        "versions_before": _dependency_versions_before,
        "installed_by_cell": list(_to_install),
        "versions_after": {
            _Requirement(spec).name: _metadata.version(_Requirement(spec).name)
            for spec in DEPENDENCIES
        },
        "pip_check_stdout": _pip_check.stdout.strip(),
        "pip_check_stderr": _pip_check.stderr.strip(),
    }
else:
    # Running as installed package — dependencies managed by pyproject.toml
    DEPENDENCY_AUDIT = {"mode": "package_install"}


# %%
# CELL 2 - All user-configurable parameters live in this cell.
#
# Keep RUN_PROFILE="research" for a full experiment. The local verification
# command sets CYSTODS_RUN_PROFILE=smoke without editing this file.
import os
from pathlib import Path

RUN_PROFILE = os.environ.get("CYSTODS_RUN_PROFILE", "research")
if RUN_PROFILE not in {"research", "smoke"}:
    raise ValueError("CYSTODS_RUN_PROFILE must be 'research' or 'smoke'.")

# Resolve only explicit, well-known dataset roots. Ambiguous discovery is an
# error because silently selecting a different Kaggle dataset invalidates a
# paired experiment.
def _resolve_data_root() -> Path:
    from cystods.config import resolve_dataset_root
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
    # The data split never changes when a model-training seed changes.
    "split_seed": 20260729,
    "split_search_candidates": 2048,
    "force_fine_labels_with_fewer_than_n_patients_to_train": 3,
    "cv_folds": 5,
    "cv_val_fraction_of_remaining": 0.15,
    "cv_run_fold_indices": None,  # None runs every fold
    "normal_mucosa_limit": 540,  # None keeps all 6,386 images
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
    # Single representative backbone: the paper-best Swin-Tiny.
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
    "supervised_contrastive_label_level": "fine",  # fine | coarse
    # Long-tail objective and sampling
    "fine_loss": "balanced_softmax",
    # cross_entropy | weighted_ce | focal | balanced_softmax |
    # balanced_softmax_smoothed | logit_adjustment | ldam
    "class_balance_beta": 0.9999,
    "focal_gamma": 2.0,
    "focal_use_class_balance": False,
    "use_data_augmentation": False,
    "logit_adjustment_tau": 0.5,
    "fine_prior_source": "patient_count",  # image_count | patient_count
    "fine_prior_smoothing_alpha": 1.0,
    "fine_prior_power": 0.5,
    "fine_prior_max_ratio": 50.0,
    "fine_absent_train_policy": "mask_and_score_zero",
    "fine_inference_calibration_mode": "validation_grid",  # fixed | validation_grid
    "fine_inference_prior_tau": 0.0,
    "fine_inference_tau_grid": (0.0, 0.25, 0.5, 0.75, 1.0),
    "fine_inference_calibration_metric": "primary_macro_f1_all_classes",
    "ldam_max_margin": 0.5,
    "ldam_scale": 30.0,
    "sampler": "random",  # random | class_balanced
    "sampler_label_level": "fine",  # fine | coarse
    # Optimizer and schedule
    "epochs": 50,
    "learning_rate": 3.0e-4,
    "encoder_learning_rate_multiplier": 0.25,
    "weight_decay": 0.05,
    "optimizer": "adamw",
    "use_fused_optimizer": False,
    "warmup_epochs": 3.0,
    "scheduler_epochs": 25,
    "minimum_learning_rate_ratio": 0.01,
    "gradient_accumulation_steps": 1,
    "gradient_clip_norm": 1.0,
    "early_stopping_patience": 8,
    "monitor_metric": "hierarchical_composite",
    "hierarchical_composite_weights": {
        "coarse_macro_f1_all_classes": 0.35,
        "primary_macro_f1_all_classes": 0.45,
        "hierarchical_accuracy": 0.20,
    },
    "checkpoint_min_delta": 1.0e-4,
    "binary_decision_threshold": 0.5,
    "resume_checkpoint": None,
    # Hardware acceleration
    "device": "auto",  # auto | cuda | mps | cpu
    "precision": "auto",  # auto | fp32 | fp16 | bf16
    "enable_tf32": True,
    "channels_last": False,
    "torch_compile": False,
    "torch_compile_mode": "default",
    "float32_matmul_precision": "high",  # highest | high | medium
    # Logging, statistics, and outputs
    "log_every_n_steps": 20,
    "save_last_checkpoint": True,
    "save_epoch_checkpoints": False,
    # Best-checkpoint persistence. Research stages use Hugging Face Hub and
    # retain only verified JSON/CSV/Markdown receipts locally.
    "checkpoint_backend": "local",  # local | huggingface
    "hf_repo_id": None,
    "hf_revision": "main",
    "hf_private": True,
    "hf_create_repo": True,
    "hf_path_prefix": "cystods",
    "hf_token_env": "HF_TOKEN",
    "bootstrap_iterations": 1000,
    "bootstrap_confidence": 0.95,
    "primary_fine_min_train_patients": 10,
    "fixed_primary_fine_class_ids": None,
    "tail_class_max_train_samples": 20,
    "rare_gate_max_train_patients": 2,
    "rare_gate_absolute_pred_share": 0.10,
    "rare_gate_prior_multiplier": 10.0,
    "rare_gate_min_pred_count": 5,
    "scientific_gate_mode": "enforce",  # enforce | report
    "probability_sum_tolerance": 5.0e-3,
    "paired_baseline_predictions_csv": None,
    "generate_sample_grid": True,
    "sample_grid_images_per_class": 3,
    # Subset evaluations
    "evaluate_wlc_only": True,
    "train_modality": "all",  # all | WLC
    "evaluate_roi_level": True,
    "roi_aggregations": ("mean", "vote", "attention"),
    "roi_conflict_policy": "exclude_and_report",  # exclude_and_report | raise
    "roi_attention_epochs": 25,
    "roi_attention_learning_rate": 1.0e-3,
    "roi_attention_hidden_dim": 128,
    "roi_attention_early_stopping_patience": 5,
    # Optional external binary validation. Enabling it requires a real CSV.
    "external_validation_enabled": False,
    "external_manifest_csv": None,
    "external_image_root": None,
    "external_path_column": "path",
    "external_binary_label_column": "binary_label",
    "external_patient_id_column": "patient_id",
}

PROPOSED_CANONICAL_CONFIG = {
    # Input / backbone
    "image_size": 224,
    "fov_center_crop_ratio": 0.92,
    "model_name": "swin_tiny_patch4_window7_224.ms_in1k",
    "pretrained": True,
    "dropout": 0.20,
    "projection_dim": 128,

    # Task
    "task_mode": "hierarchical",

    # Main objectives
    "binary_loss_weight": 1.0,
    "coarse_loss_weight": 1.0,
    "fine_loss_weight": 1.0,

    # Hierarchy
    "binary_coarse_hierarchy_loss_weight": 0.25,
    "coarse_fine_hierarchy_loss_weight": 0.25,

    # Fine long-tail
    "fine_loss": "balanced_softmax",
    "fine_prior_source": "patient_count",
    "fine_prior_smoothing_alpha": 1.0,
    "fine_prior_power": 0.5,
    "fine_prior_max_ratio": 50.0,
    "fine_absent_train_policy": "mask_and_score_zero",
    "class_balance_beta": 0.9999,
    "focal_gamma": 2.0,
    "focal_use_class_balance": False,
    "logit_adjustment_tau": 0.5,
    "ldam_max_margin": 0.5,
    "ldam_scale": 30.0,

    # SupCon
    "supervised_contrastive_loss_weight": 0.10,
    "supervised_contrastive_temperature": 0.10,
    "supervised_contrastive_label_level": "fine",

    # Classification augmentation
    "use_data_augmentation": False,

    # Sampling
    "sampler": "random",
    "sampler_label_level": "fine",

    # Fine inference calibration
    "fine_inference_calibration_mode": "validation_grid",
    "fine_inference_prior_tau": 0.0,
    "fine_inference_tau_grid": (0.0, 0.25, 0.5, 0.75, 1.0),
    "fine_inference_calibration_metric": "primary_macro_f1_all_classes",

    # Optimization recipe
    "batch_size": 256,
    "epochs": 25,
    "learning_rate": 3.0e-4,
    "encoder_learning_rate_multiplier": 0.25,
    "weight_decay": 0.05,
    "optimizer": "adamw",
    "warmup_epochs": 2.0,
    "scheduler_epochs": 25,
    "minimum_learning_rate_ratio": 0.01,
    "gradient_accumulation_steps": 1,
    "gradient_clip_norm": 1.0,

    # Transforms / Augmentations
    "random_resized_crop_scale": (0.75, 1.0),
    "horizontal_flip_probability": 0.5,
    "vertical_flip_probability": 0.5,
    "rotation_degrees": 15,
    "color_jitter": (0.20, 0.20, 0.15, 0.05),
    "random_erasing_probability": 0.20,
    "imagenet_mean": (0.485, 0.456, 0.406),
    "imagenet_std": (0.229, 0.224, 0.225),

    # Modality & Evaluation scope
    "train_modality": "all",
    "evaluate_wlc_only": False,

    # Checkpoint selection
    "early_stopping_patience": 6,
    "monitor_metric": "hierarchical_composite",
    "hierarchical_composite_weights": {
        "coarse_macro_f1_all_classes": 0.35,
        "primary_macro_f1_all_classes": 0.45,
        "hierarchical_accuracy": 0.20,
    },
    "checkpoint_min_delta": 1.0e-4,
}

PROFILE_OVERRIDES = {
    "research": {},
    "smoke": {
        "experiment_name": "cystods_local_smoke",
        "verify_all_image_decodes": False,
        "dataset_fingerprint_mode": "semantic",
        "verify_exact_duplicate_images": False,
        "protocol": "holdout",
        "fixed_split_pids": {
            "train": ("1552",),
            "val": ("1261",),
            "test": ("808",),
        },
        "normal_mucosa_limit": None,
        "image_size": 64,
        "fov_center_crop_ratio": 0.95,
        "random_resized_crop_scale": (0.85, 1.0),
        "batch_size": 4,
        "eval_batch_size": 4,
        "num_workers": 0,
        "eval_num_workers": 0,
        "persistent_workers": False,
        "pin_memory": False,
        "model_name": "swin_tiny_patch4_window7_224.ms_in1k",
        "pretrained": False,
        "dropout": 0.10,
        "supervised_contrastive_loss_weight": 0.0,
        "fine_loss": "focal",
        "fine_inference_calibration_mode": "fixed",
        "epochs": 1,
        "scheduler_epochs": 1,
        "learning_rate": 1.0e-3,
        "warmup_epochs": 0.0,
        "early_stopping_patience": 1,
        "monitor_metric": "coarse_macro_f1",
        "device": "cpu",
        "precision": "fp32",
        "enable_tf32": False,
        "channels_last": False,
        "bootstrap_iterations": 20,
        "scientific_gate_mode": "report",
        "log_every_n_steps": 1,
        "sample_grid_images_per_class": 1,
        "save_last_checkpoint": False,
        "roi_aggregations": ("mean", "vote", "attention"),
        "roi_attention_epochs": 1,
        "checkpoint_backend": "local",
    },
}

CONFIG = dict(BASE_CONFIG)
CONFIG.update(PROFILE_OVERRIDES[RUN_PROFILE])


# %%
# CELL 3 - Imports, immutable taxonomy, and shared utilities.
import hashlib
import json
import logging
import math
import platform
import random
import shutil
import socket
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cystods.hf as checkpoint_hub
import cystods.science as science
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import timm
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

BINARY_NAMES = ("Non-ROI", "ROI")
COARSE_NAMES = (
    "Malignant",
    "Non-malignant",
    "Normal mucosa",
    "Anatomical landmarks",
    "Foreign bodies",
)
FINE_BY_PARENT = {
    "Malignant": (
        "LowGradePapillary",
        "HighGradePapillary",
        "CIS",
        "PreMalignant",
    ),
    "Non-malignant": (
        "BenignNOS",
        "InflammationNOS",
        "CCG",
        "Denuded",
        "UrothelialPapilloma",
        "SquamousMetaplasia",
        "NephrogenicAdenoma",
        "BenignRare",
    ),
    "Anatomical landmarks": (
        "UreteralOrifice",
        "ResectionBed",
        "ResectionScar",
        "Trabeculation",
        "ProstaticUrethra",
        "Diverticulum",
    ),
    "Foreign bodies": (
        "AirBubble",
        "ResectionLoop",
        "BiopsyForcep",
        "Stent",
    ),
}
FINE_NAMES = tuple(
    fine_name
    for parent_name in (
        "Malignant",
        "Non-malignant",
        "Anatomical landmarks",
        "Foreign bodies",
    )
    for fine_name in FINE_BY_PARENT[parent_name]
)
if len(FINE_NAMES) != 22 or len(set(FINE_NAMES)) != 22:
    raise RuntimeError("The immutable taxonomy must contain exactly 22 fine labels.")

COARSE_TO_ID = {name: index for index, name in enumerate(COARSE_NAMES)}
FINE_TO_ID = {name: index for index, name in enumerate(FINE_NAMES)}
FINE_PARENT_ID = tuple(
    COARSE_TO_ID[parent]
    for fine_name in FINE_NAMES
    for parent, children in FINE_BY_PARENT.items()
    if fine_name in children
)
FINE_PARENT_ID_TENSOR = torch.tensor(FINE_PARENT_ID, dtype=torch.long)
ROI_COARSE_IDS = frozenset(
    (COARSE_TO_ID["Malignant"], COARSE_TO_ID["Non-malignant"])
)
COARSE_BINARY_PARENT_ID = torch.tensor(
    [
        1,  # Malignant -> ROI
        1,  # Non-malignant -> ROI
        0,  # Normal mucosa -> Non-ROI
        0,  # Anatomical landmarks -> Non-ROI
        0,  # Foreign bodies -> Non-ROI
    ],
    dtype=torch.long,
)
COARSE_TO_BINARY_MATRIX = torch.zeros(
    (len(COARSE_NAMES), len(BINARY_NAMES)),
    dtype=torch.float32,
)
for coarse_id, binary_id in enumerate(COARSE_BINARY_PARENT_ID.tolist()):
    COARSE_TO_BINARY_MATRIX[coarse_id, binary_id] = 1.0

FINE_TO_COARSE_MATRIX = torch.zeros(
    (len(FINE_NAMES), len(COARSE_NAMES)),
    dtype=torch.float32,
)
for fine_id, coarse_id in enumerate(FINE_PARENT_ID):
    FINE_TO_COARSE_MATRIX[fine_id, coarse_id] = 1.0

REQUIRED_COLUMNS = {
    "filename",
    "pid",
    "visit",
    "lesion",
    "multifocal",
    "bca",
    "class",
    "subclass",
    "subclass2",
    "stage",
    "morphology",
    "modality",
    "json",
}
MISSING_TOKENS = frozenset(("", "NA", "N/A", "nan", "None", "null"))


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise FloatingPointError(
                "Refusing to serialize a non-finite tensor."
            )
        return tensor.tolist()
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(
            f"Refusing to serialize non-finite float: {value!r}"
        )
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hf_checkpoint_config(
    config: Mapping[str, Any],
    path_in_repo: str,
) -> checkpoint_hub.HFCheckpointConfig:
    if config["checkpoint_backend"] != "huggingface":
        raise ValueError(
            "Hugging Face checkpoint configuration requested for a local "
            "checkpoint backend."
        )
    token_env = str(config["hf_token_env"])
    token = os.environ.get(token_env)
    if token is None or not token.strip():
        raise ValueError(
            f"Required Hugging Face token environment variable is missing: "
            f"{token_env}"
        )
    return checkpoint_hub.HFCheckpointConfig(
        repo_id=str(config["hf_repo_id"]),
        path_in_repo=path_in_repo,
        token=token.strip(),
        revision=str(config["hf_revision"]),
        private=bool(config["hf_private"]),
        create_repo=bool(config["hf_create_repo"]),
        endpoint=os.environ.get("HF_ENDPOINT") or None,
    )


def _hf_checkpoint_path_in_repo(
    config: Mapping[str, Any], fold_name: str
) -> str:
    run_identity = hashlib.sha256(
        "::".join(
            (
                str(config["study_id"]),
                str(config["stage_name"]),
                str(config["suite_trial_id"] or config["experiment_name"]),
                str(config["seed"]),
                fold_name,
                str(config["protocol_reference_sha256"]),
            )
        ).encode("utf-8")
    ).hexdigest()[:20]
    return (
        f"{str(config['hf_path_prefix']).rstrip('/')}/"
        f"{run_identity}/{fold_name}/best_model.pt"
    )


def _canonical_model_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remove torch.compile wrapper names without changing tensor values."""

    canonical: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        normalized = name.replace("encoder._orig_mod.", "encoder.")
        if normalized in canonical:
            raise ValueError(
                "Canonical checkpoint key collision after removing "
                f"torch.compile wrapper: {normalized}"
            )
        canonical[normalized] = tensor
    return canonical


def stable_int_seed(*parts: Any) -> int:
    digest = hashlib.sha256("::".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def is_missing_token(value: Any) -> bool:
    return str(value).strip() in MISSING_TOKENS


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_directory(config: Mapping[str, Any]) -> Path:
    root = Path(config["result_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).astimezone().strftime(
        "%Y%m%d-%H%M%S"
    )
    exp_name = str(config.get("experiment_name", "cystods"))
    profile = str(config.get("run_profile", "research"))

    stage_prefix = exp_name
    if "stage_00" in exp_name:
        stage_prefix = "00_protocol"
    elif "stage_10" in exp_name:
        stage_prefix = "10_baselines"
    elif "stage_20" in exp_name:
        stage_prefix = "20_long_tail"
    elif "stage_30" in exp_name:
        stage_prefix = "30_proposed"
    elif "stage_40" in exp_name:
        stage_prefix = "40_ablations"
    elif "stage_60" in exp_name:
        stage_prefix = "60_external"
    elif "stage_90" in exp_name:
        stage_prefix = "90_final_cv"

    base = root / stage_prefix / f"{profile}_{stamp}"
    run_dir = base
    suffix = 1
    while run_dir.exists():
        run_dir = root / stage_prefix / f"{profile}_{stamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("reports", "splits", "system", "source", "logs", "folds"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


class FlushFileHandler(logging.FileHandler):
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"cystods.{log_path.parent.parent.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = FlushFileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(_sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def validate_source_files(
    source_files: Sequence[Path | str],
) -> tuple[Path, ...]:
    resolved = tuple(Path(path).expanduser().resolve() for path in source_files)
    if not resolved:
        raise ValueError("At least one source file is required for provenance.")
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Required source snapshot files are missing: {missing}"
        )
    names = [path.name for path in resolved]
    if len(set(names)) != len(names):
        raise ValueError(
            "Required source files must have unique basenames for a flat "
            f"snapshot; received={names}."
        )
    return resolved


def snapshot_source_files(
    run_dir: Path,
    source_files: Sequence[Path | str],
) -> dict[str, Any]:
    resolved = validate_source_files(source_files)
    (run_dir / "source").mkdir(parents=True, exist_ok=True)
    rows = []
    for source_path in resolved:
        destination = run_dir / "source" / source_path.name
        shutil.copy2(source_path, destination)
        source_hash = sha256_file(source_path)
        copied_hash = sha256_file(destination)
        if copied_hash != source_hash:
            raise OSError(
                f"Source snapshot checksum mismatch for {source_path}."
            )
        rows.append(
            {
                "path": source_path.name,
                "source_path": str(source_path),
                "snapshot_path": str(destination.relative_to(run_dir)),
                "bytes": destination.stat().st_size,
                "sha256": copied_hash,
            }
        )
    manifest = {
        "schema_version": "cystods.source_manifest.v1",
        "files": rows,
    }
    write_json(run_dir / "source" / "source_manifest.json", manifest)
    return manifest


def normalize_core_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    unknown = set(config) - set(BASE_CONFIG)
    if unknown:
        raise KeyError(f"Core config contains unknown keys: {sorted(unknown)}")
    normalized = dict(BASE_CONFIG)
    normalized.update(dict(config))
    return normalized


def validate_config(config: Mapping[str, Any]) -> None:
    required_keys = set(BASE_CONFIG)
    missing_keys = required_keys - set(config)
    if missing_keys:
        raise KeyError(
            f"Core config is missing keys: {sorted(missing_keys)}"
        )
    unknown_keys = set(config) - required_keys
    if unknown_keys:
        raise KeyError(
            f"Core config contains unknown keys: {sorted(unknown_keys)}"
        )
    valid_protocols = {"holdout", "cross_validation"}
    if config["protocol"] not in valid_protocols:
        raise ValueError(f"protocol must be one of {sorted(valid_protocols)}")
    if (
        isinstance(config["split_seed"], (bool, np.bool_))
        or not isinstance(config["split_seed"], (int, np.integer))
        or int(config["split_seed"]) < 0
    ):
        raise ValueError("split_seed must be a non-negative integer.")
    fractions = (
        float(config["train_fraction"]),
        float(config["val_fraction"]),
        float(config["test_fraction"]),
    )
    if any(value <= 0 for value in fractions):
        raise ValueError("All holdout fractions must be positive.")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("train_fraction + val_fraction + test_fraction must be 1.")
    if int(config["image_size"]) < 32:
        raise ValueError("image_size must be at least 32.")
    if int(config["batch_size"]) < 1:
        raise ValueError("batch_size must be positive.")
    if int(config["eval_batch_size"]) < 1:
        raise ValueError("eval_batch_size must be positive.")
    if int(config["epochs"]) < 1:
        raise ValueError("epochs must be positive.")
    if int(config["scheduler_epochs"]) < 1:
        raise ValueError("scheduler_epochs must be positive.")
    if not 0 < float(config["binary_decision_threshold"]) < 1:
        raise ValueError("binary_decision_threshold must be in (0, 1).")
    if int(config["gradient_accumulation_steps"]) < 1:
        raise ValueError("gradient_accumulation_steps must be positive.")
    if int(config["log_every_n_steps"]) < 1:
        raise ValueError("log_every_n_steps must be positive.")
    if int(config["early_stopping_patience"]) < 1:
        raise ValueError("early_stopping_patience must be positive.")
    if int(config["bootstrap_iterations"]) < 1:
        raise ValueError("bootstrap_iterations must be positive.")
    if int(config["primary_fine_min_train_patients"]) < 1:
        raise ValueError(
            "primary_fine_min_train_patients must be positive."
        )
    if not 0 < float(config["bootstrap_confidence"]) < 1:
        raise ValueError("bootstrap_confidence must be in (0, 1).")
    if int(config["roi_attention_epochs"]) < 1:
        raise ValueError("roi_attention_epochs must be positive.")
    if int(config["roi_attention_early_stopping_patience"]) < 1:
        raise ValueError(
            "roi_attention_early_stopping_patience must be positive."
        )
    if int(config["num_workers"]) < 0:
        raise ValueError("num_workers cannot be negative.")
    if int(config["eval_num_workers"]) < 0:
        raise ValueError("eval_num_workers cannot be negative.")
    if int(config["prefetch_factor"]) < 1:
        raise ValueError("prefetch_factor must be positive.")
    if int(config["eval_prefetch_factor"]) < 1:
        raise ValueError("eval_prefetch_factor must be positive.")
    if int(config["num_workers"]) == 0 and config["persistent_workers"]:
        raise ValueError(
            "persistent_workers requires a positive training worker count."
        )
    if config["float32_matmul_precision"] not in {
        "highest",
        "high",
        "medium",
    }:
        raise ValueError(
            "float32_matmul_precision must be highest, high, or medium."
        )
    if config["checkpoint_backend"] not in {"local", "huggingface"}:
        raise ValueError(
            "checkpoint_backend must be local or huggingface."
        )
    if config["checkpoint_backend"] == "huggingface":
        for key in ("hf_private", "hf_create_repo"):
            if not isinstance(config[key], (bool, np.bool_)):
                raise TypeError(f"{key} must be a boolean.")
        repo_id = config["hf_repo_id"]
        if (
            not isinstance(repo_id, str)
            or len(repo_id.split("/")) != 2
            or any(not part for part in repo_id.split("/"))
        ):
            raise ValueError(
                "Hugging Face checkpoint storage requires "
                "hf_repo_id='namespace/repository'."
            )
        for key in ("hf_revision", "hf_path_prefix", "hf_token_env"):
            value = config[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a non-empty string.")
        prefix_parts = str(config["hf_path_prefix"]).split("/")
        if any(part in {"", ".", ".."} for part in prefix_parts):
            raise ValueError("hf_path_prefix must be a normalized relative path.")
        if config["resume_checkpoint"] is not None:
            raise ValueError(
                "Hugging Face checkpoint mode does not support local resume_checkpoint."
            )
        if config["save_last_checkpoint"] or config["save_epoch_checkpoints"]:
            raise ValueError(
                "Hugging Face checkpoint mode permits only the remote best_model.pt; "
                "save_last_checkpoint and save_epoch_checkpoints must be false."
            )
    resolved_name = resolve_model_name(config["model_name"])
    if not timm.is_model(resolved_name):
        raise ValueError(
            f"Unsupported or unavailable timm model: '{config['model_name']}' "
            f"(resolved: '{resolved_name}')"
        )
    if config["fine_loss"] not in {
        "cross_entropy",
        "weighted_ce",
        "focal",
        "balanced_softmax",
        "balanced_softmax_smoothed",
        "logit_adjustment",
        "ldam",
    }:
        raise ValueError(f"Unsupported fine_loss: {config['fine_loss']}")
    if config["task_mode"] not in {
        "binary",
        "coarse",
        "fine",
        "multitask",
        "hierarchical",
    }:
        raise ValueError(
            "task_mode must be binary, coarse, fine, multitask, or "
            "hierarchical."
        )
    if config["fine_prior_source"] not in {
        "image_count",
        "patient_count",
    }:
        raise ValueError(
            "fine_prior_source must be image_count or patient_count."
        )
    smoothing_alpha = float(config["fine_prior_smoothing_alpha"])
    if not math.isfinite(smoothing_alpha) or smoothing_alpha < 0:
        raise ValueError(
            "fine_prior_smoothing_alpha must be finite and non-negative."
        )
    if not 0 < float(config["fine_prior_power"]) <= 1:
        raise ValueError("fine_prior_power must be in (0, 1].")
    prior_max_ratio = float(config["fine_prior_max_ratio"])
    if not math.isfinite(prior_max_ratio) or prior_max_ratio < 1:
        raise ValueError(
            "fine_prior_max_ratio must be finite and at least 1."
        )
    if (
        config["fine_absent_train_policy"]
        != "mask_and_score_zero"
    ):
        raise ValueError(
            "fine_absent_train_policy must be mask_and_score_zero."
        )
    if config["fine_inference_calibration_mode"] not in {
        "fixed",
        "validation_grid",
    }:
        raise ValueError(
            "fine_inference_calibration_mode must be fixed or "
            "validation_grid."
        )
    tau_grid = tuple(
        float(value) for value in config["fine_inference_tau_grid"]
    )
    if not tau_grid or any(not math.isfinite(value) for value in tau_grid):
        raise ValueError("fine_inference_tau_grid must be finite and nonempty.")
    if not math.isfinite(float(config["fine_inference_prior_tau"])):
        raise ValueError("fine_inference_prior_tau must be finite.")
    if config["fine_inference_calibration_metric"] not in {
        "fine_macro_f1_all_classes",
        "primary_macro_f1_all_classes",
    }:
        raise ValueError(
            "fine_inference_calibration_metric must be "
            "fine_macro_f1_all_classes or primary_macro_f1_all_classes."
        )
    if config["dataset_fingerprint_mode"] not in {"full", "semantic"}:
        raise ValueError(
            "dataset_fingerprint_mode must be full or semantic."
        )
    if config["scientific_gate_mode"] not in {"enforce", "report"}:
        raise ValueError(
            "scientific_gate_mode must be enforce or report."
        )
    if config["train_modality"] not in {"all", "WLC"}:
        raise ValueError("train_modality must be all or WLC.")
    if config["sampler"] not in {"random", "class_balanced"}:
        raise ValueError("sampler must be random or class_balanced.")
    objective_weights = [
        float(config[key])
        for key in (
            "binary_loss_weight",
            "coarse_loss_weight",
            "fine_loss_weight",
            "binary_coarse_hierarchy_loss_weight",
            "coarse_fine_hierarchy_loss_weight",
            "supervised_contrastive_loss_weight",
        )
    ]
    if any(not math.isfinite(weight) or weight < 0 for weight in objective_weights):
        raise ValueError(
            "Objective weights must be finite and non-negative."
        )
    if not any(weight > 0 for weight in objective_weights):
        raise ValueError("At least one training objective must be active.")
    if config["task_mode"] == "binary" and float(
        config["binary_loss_weight"]
    ) <= 0:
        raise ValueError("binary task_mode requires binary_loss_weight > 0.")
    if config["task_mode"] == "coarse" and float(
        config["coarse_loss_weight"]
    ) <= 0:
        raise ValueError("coarse task_mode requires coarse_loss_weight > 0.")
    if config["task_mode"] == "fine" and float(
        config["fine_loss_weight"]
    ) <= 0:
        raise ValueError("fine task_mode requires fine_loss_weight > 0.")
    hierarchy_enabled = (
        float(config["binary_coarse_hierarchy_loss_weight"]) > 0
        or float(config["coarse_fine_hierarchy_loss_weight"]) > 0
    )
    if hierarchy_enabled and str(config["task_mode"]) != "hierarchical":
        raise ValueError(
            "Hierarchy taxonomy losses require task_mode='hierarchical'."
        )
    inactive_weight_keys = {
        "binary": (
            "coarse_loss_weight",
            "fine_loss_weight",
            "binary_coarse_hierarchy_loss_weight",
            "coarse_fine_hierarchy_loss_weight",
            "supervised_contrastive_loss_weight",
        ),
        "coarse": (
            "binary_loss_weight",
            "fine_loss_weight",
            "binary_coarse_hierarchy_loss_weight",
            "coarse_fine_hierarchy_loss_weight",
        ),
        "fine": (
            "binary_loss_weight",
            "coarse_loss_weight",
            "binary_coarse_hierarchy_loss_weight",
            "coarse_fine_hierarchy_loss_weight",
        ),
        "multitask": (
            "binary_coarse_hierarchy_loss_weight",
            "coarse_fine_hierarchy_loss_weight",
        ),
        "hierarchical": (),
    }[str(config["task_mode"])]
    nonzero_inactive = [
        key
        for key in inactive_weight_keys
        if float(config[key]) != 0
    ]
    if nonzero_inactive:
        raise ValueError(
            f"task_mode={config['task_mode']} requires zero inactive "
            f"weights: {nonzero_inactive}"
        )
    monitor_weights = {
        str(name): float(value)
        for name, value in config["hierarchical_composite_weights"].items()
    }
    expected_monitor_components = {
        "coarse_macro_f1_all_classes",
        "primary_macro_f1_all_classes",
        "hierarchical_accuracy",
    }
    if set(monitor_weights) != expected_monitor_components:
        raise ValueError(
            "hierarchical_composite_weights must contain exactly "
            f"{sorted(expected_monitor_components)}."
        )
    if any(
        not math.isfinite(value) or value < 0
        for value in monitor_weights.values()
    ):
        raise ValueError(
            "Composite monitor weights must be finite and non-negative."
        )
    if not math.isclose(
        sum(monitor_weights.values()), 1.0, rel_tol=0, abs_tol=1e-9
    ):
        raise ValueError("Composite monitor weights must sum to 1.")
    checkpoint_min_delta = float(config["checkpoint_min_delta"])
    if not math.isfinite(checkpoint_min_delta) or checkpoint_min_delta < 0:
        raise ValueError(
            "checkpoint_min_delta must be finite and non-negative."
        )
    probability_tolerance = float(config["probability_sum_tolerance"])
    if (
        not math.isfinite(probability_tolerance)
        or probability_tolerance <= 0
    ):
        raise ValueError(
            "probability_sum_tolerance must be finite and positive."
        )
    fixed_primary_ids = config["fixed_primary_fine_class_ids"]
    if fixed_primary_ids is not None:
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in fixed_primary_ids
        ):
            raise ValueError(
                "fixed_primary_fine_class_ids must contain integer values."
            )
        normalized_ids = tuple(int(value) for value in fixed_primary_ids)
        if (
            len(normalized_ids) == 0
            or len(set(normalized_ids)) != len(normalized_ids)
            or tuple(sorted(normalized_ids)) != normalized_ids
            or normalized_ids[0] < 0
            or normalized_ids[-1] >= len(FINE_NAMES)
        ):
            raise ValueError(
                "fixed_primary_fine_class_ids must be a non-empty, sorted, "
                "unique sequence of IDs in the 22-class taxonomy."
            )
    if int(config["rare_gate_max_train_patients"]) < 0:
        raise ValueError("rare_gate_max_train_patients cannot be negative.")
    rare_absolute_share = float(config["rare_gate_absolute_pred_share"])
    if (
        not math.isfinite(rare_absolute_share)
        or not 0 <= rare_absolute_share <= 1
    ):
        raise ValueError("rare_gate_absolute_pred_share must be in [0, 1].")
    rare_prior_multiplier = float(config["rare_gate_prior_multiplier"])
    if not math.isfinite(rare_prior_multiplier) or rare_prior_multiplier < 0:
        raise ValueError(
            "rare_gate_prior_multiplier must be finite and non-negative."
        )
    if int(config["rare_gate_min_pred_count"]) < 1:
        raise ValueError("rare_gate_min_pred_count must be positive.")
    if config["roi_conflict_policy"] not in {"exclude_and_report", "raise"}:
        raise ValueError(
            "roi_conflict_policy must be exclude_and_report or raise."
        )
    allowed_roi = {"mean", "vote", "attention"}
    unknown_roi = set(config["roi_aggregations"]) - allowed_roi
    if unknown_roi:
        raise ValueError(f"Unknown ROI aggregations: {sorted(unknown_roi)}")
    if config["protocol_role"] not in {
        None,
        "smoke_holdout",
        "fixed_holdout",
        "final_cv",
    }:
        raise ValueError("Unsupported protocol_role.")
    if config["evaluation_scope"] not in {
        "legacy",
        "development",
        "final_cv",
        "external",
    }:
        raise ValueError("Unsupported evaluation_scope.")
    protocol_reference = config["protocol_reference_sha256"]
    if protocol_reference is not None:
        reference = str(protocol_reference)
        if len(reference) != 64 or any(
            character not in "0123456789abcdef" for character in reference
        ):
            raise ValueError(
                "protocol_reference_sha256 must be a lowercase SHA-256 digest."
            )
    expected_dataset_hash = config["expected_dataset_semantic_sha256"]
    if expected_dataset_hash is not None:
        expected_dataset_hash = str(expected_dataset_hash)
        if len(expected_dataset_hash) != 64 or any(
            character not in "0123456789abcdef"
            for character in expected_dataset_hash
        ):
            raise ValueError(
                "expected_dataset_semantic_sha256 must be a lowercase "
                "SHA-256 digest."
            )
    if config["external_validation_enabled"]:
        required_external = (
            config["external_manifest_csv"],
            config["external_image_root"],
        )
        if any(value is None for value in required_external):
            raise ValueError(
                "External validation requires external_manifest_csv and "
                "external_image_root."
            )


def resolve_device(config: Mapping[str, Any]) -> torch.device:
    requested = str(config["device"]).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' was requested but CUDA is unavailable.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device='mps' was requested but MPS is unavailable.")
    if requested not in {"cuda", "mps", "cpu"}:
        raise ValueError("device must be auto, cuda, mps, or cpu.")
    return torch.device(requested)


def resolve_precision(
    config: Mapping[str, Any], device: torch.device
) -> tuple[str, torch.dtype | None]:
    requested = str(config["precision"]).lower()
    if requested == "auto":
        if device.type == "cuda":
            requested = (
                "bf16"
                if torch.cuda.is_bf16_supported()
                else "fp16"
            )
        else:
            requested = "fp32"
    if requested == "fp32":
        return requested, None
    if requested == "fp16":
        if device.type not in {"cuda", "mps"}:
            raise RuntimeError("fp16 requires a CUDA or MPS device.")
        return requested, torch.float16
    if requested == "bf16":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 requires a CUDA device with BF16 support.")
        return requested, torch.bfloat16
    raise ValueError("precision must be auto, fp32, fp16, or bf16.")


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def collect_system_info(
    config: Mapping[str, Any],
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestamp_utc": utc_now_iso(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": _sys.version,
        "executable": _sys.executable,
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "memory_total_gib": psutil.virtual_memory().total / (1024**3),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "device": str(device),
        "precision": precision,
        "run_profile": config["run_profile"],
    }
    if device.type == "cuda":
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        info.update(
            {
                "cuda_version": torch.version.cuda,
                "cuda_device_name": props.name,
                "cuda_memory_gib": props.total_memory / (1024**3),
                "cuda_capability": torch.cuda.get_device_capability(index),
            }
        )
    return info


def audit_image_size_distribution(
    frame: pd.DataFrame,
    run_dir: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Audit image dimensions (width, height, aspect ratio, megapixels) across total dataset and per-class for binary, coarse, and fine layers."""
    logger.info("Auditing image size distribution across %d images...", len(frame))
    records: list[dict[str, Any]] = []
    for row_dict in frame.to_dict(orient="records"):
        image_path = Path(row_dict["image_path"])
        with Image.open(image_path) as img:
            width, height = img.size
        aspect_ratio = round(width / height, 3)
        mp = round((width * height) / 1e6, 3)

        subclass_str = str(row_dict.get("subclass", "")).strip()
        subclass2_str = str(row_dict.get("subclass2", "")).strip()
        coarse_class = str(row_dict.get("class", "")).strip()

        if coarse_class == "Normal mucosa":
            fine_label = "Normal mucosa"
        elif subclass_str and not is_missing_token(subclass_str):
            fine_label = subclass_str
        elif subclass2_str and not is_missing_token(subclass2_str):
            fine_label = subclass2_str
        else:
            fine_label = coarse_class

        binary_label = "ROI" if coarse_class in ("Malignant", "Non-malignant") else "Non-ROI"

        records.append(
            {
                "image_stem": str(row_dict["image_stem"]),
                "width": int(width),
                "height": int(height),
                "aspect_ratio": float(aspect_ratio),
                "megapixels": float(mp),
                "resolution": f"{width}x{height}",
                "binary_class": binary_label,
                "coarse_class": coarse_class,
                "fine_label": fine_label,
            }
        )

    df_size = pd.DataFrame(records)

    def _summarize(df_subset: pd.DataFrame) -> dict[str, Any]:
        if df_subset.empty:
            return {}
        widths = df_subset["width"].to_numpy()
        heights = df_subset["height"].to_numpy()
        ars = df_subset["aspect_ratio"].to_numpy()
        mps = df_subset["megapixels"].to_numpy()
        res_counts = df_subset["resolution"].value_counts()
        top_res = str(res_counts.index[0]) if not res_counts.empty else "N/A"
        top_resolutions = [
            {"resolution": str(res), "count": int(cnt)}
            for res, cnt in res_counts.head(5).items()
        ]
        return {
            "count": int(len(df_subset)),
            "width_min": int(np.min(widths)),
            "width_max": int(np.max(widths)),
            "width_mean": float(round(float(np.mean(widths)), 2)),
            "width_std": float(round(float(np.std(widths)), 2)),
            "width_median": int(np.median(widths)),
            "height_min": int(np.min(heights)),
            "height_max": int(np.max(heights)),
            "height_mean": float(round(float(np.mean(heights)), 2)),
            "height_std": float(round(float(np.std(heights)), 2)),
            "height_median": int(np.median(heights)),
            "aspect_ratio_median": float(round(float(np.median(ars)), 3)),
            "megapixels_median": float(round(float(np.median(mps)), 3)),
            "megapixels_mean": float(round(float(np.mean(mps)), 3)),
            "top_resolution": top_res,
            "top_resolutions": top_resolutions,
        }

    total_stats = _summarize(df_size)

    binary_stats: dict[str, dict[str, Any]] = {}
    for label in ("Non-ROI", "ROI"):
        sub = df_size.loc[df_size["binary_class"] == label]
        if not sub.empty:
            binary_stats[label] = _summarize(sub)

    coarse_stats: dict[str, dict[str, Any]] = {}
    for label in COARSE_NAMES:
        sub = df_size.loc[df_size["coarse_class"] == label]
        if not sub.empty:
            coarse_stats[label] = _summarize(sub)

    fine_stats: dict[str, dict[str, Any]] = {}
    for label in sorted(df_size["fine_label"].unique().tolist()):
        sub = df_size.loc[df_size["fine_label"] == label]
        if not sub.empty:
            fine_stats[label] = _summarize(sub)

    size_audit = {
        "total": total_stats,
        "binary_layers": binary_stats,
        "coarse_layers": coarse_stats,
        "fine_layers": fine_stats,
    }

    # Write JSON report
    write_json(run_dir / "reports" / "image_size_distribution.json", size_audit)

    # Write flattened CSV report
    csv_rows: list[dict[str, Any]] = []

    def _add_csv_row(level: str, name: str, stats: dict[str, Any]) -> None:
        if not stats:
            return
        csv_rows.append(
            {
                "level": level,
                "label_name": name,
                "count": stats.get("count", 0),
                "width_min": stats.get("width_min"),
                "width_median": stats.get("width_median"),
                "width_mean": stats.get("width_mean"),
                "width_max": stats.get("width_max"),
                "height_min": stats.get("height_min"),
                "height_median": stats.get("height_median"),
                "height_mean": stats.get("height_mean"),
                "height_max": stats.get("height_max"),
                "aspect_ratio_median": stats.get("aspect_ratio_median"),
                "megapixels_median": stats.get("megapixels_median"),
                "top_resolution": stats.get("top_resolution"),
            }
        )

    _add_csv_row("total", "Total Dataset", total_stats)
    for k, v in binary_stats.items():
        _add_csv_row("binary", k, v)
    for k, v in coarse_stats.items():
        _add_csv_row("coarse", k, v)
    for k, v in fine_stats.items():
        _add_csv_row("fine", k, v)

    pd.DataFrame(csv_rows).to_csv(
        run_dir / "reports" / "image_size_distribution.csv",
        index=False,
    )

    return size_audit


# %%
# CELL 4 - Manifest audit and patient-level split construction.
def load_and_validate_manifest(
    config: Mapping[str, Any],
    run_dir: Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata_csv = Path(config["metadata_csv"])
    image_dir = Path(config["image_dir"])
    segmentation_dir = Path(config["segmentation_dir"])
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if config["verify_segmentation_inventory"] and not segmentation_dir.is_dir():
        raise FileNotFoundError(
            f"Segmentation directory not found: {segmentation_dir}"
        )

    frame = pd.read_csv(
        metadata_csv,
        dtype=str,
        keep_default_na=False,
    )
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Metadata is missing required columns: {sorted(missing_columns)}"
        )
    if frame.empty:
        raise ValueError("Metadata CSV contains no data rows.")
    frame = frame.copy()
    inclusion_manifest = config["inclusion_manifest_csv"]
    if inclusion_manifest is not None:
        inclusion_path = Path(inclusion_manifest)
        if not inclusion_path.is_file():
            raise FileNotFoundError(
                f"Inclusion manifest not found: {inclusion_path}"
            )
        inclusion = pd.read_csv(
            inclusion_path, dtype=str, keep_default_na=False
        )
        filename_column = str(
            config["inclusion_manifest_filename_column"]
        )
        if filename_column not in inclusion:
            raise ValueError(
                "Inclusion manifest lacks configured filename column "
                f"'{filename_column}'."
            )
        inclusion_stems = inclusion[filename_column].map(
            lambda value: Path(value).stem
        )
        if inclusion_stems.duplicated().any():
            raise ValueError("Inclusion manifest contains duplicate stems.")
        source_stems = frame["filename"].map(lambda value: Path(value).stem)
        unknown_stems = set(inclusion_stems) - set(source_stems)
        if unknown_stems:
            raise ValueError(
                "Inclusion manifest references unknown images; first="
                f"{min(unknown_stems)}"
            )
        frame = frame.loc[source_stems.isin(set(inclusion_stems))].copy()
        if len(frame) != len(inclusion_stems):
            raise RuntimeError(
                "Inclusion manifest did not resolve one-to-one to metadata."
            )
    frame["pid"] = frame["pid"].astype(str)
    if frame["pid"].map(is_missing_token).any():
        raise ValueError("Every row must have a non-missing patient ID.")

    frame["image_stem"] = frame["filename"].map(lambda value: Path(value).stem)
    if frame["image_stem"].duplicated().any():
        duplicated = frame.loc[
            frame["image_stem"].duplicated(keep=False), "image_stem"
        ].unique()
        raise ValueError(
            "Duplicate de-identified image stems found: "
            f"{duplicated[:10].tolist()}"
        )
    frame["image_path"] = frame["image_stem"].map(
        lambda stem: str(image_dir / f"{stem}.png")
    )
    missing_images = [
        path for path in frame["image_path"] if not Path(path).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} normalized PNG paths are missing; "
            f"first={missing_images[0]}"
        )

    observed_coarse = set(frame["class"])
    unknown_coarse = observed_coarse - set(COARSE_NAMES)
    if unknown_coarse:
        raise ValueError(f"Unknown coarse labels: {sorted(unknown_coarse)}")
    if observed_coarse != set(COARSE_NAMES):
        raise ValueError(
            "Dataset does not contain every expected coarse label; "
            f"observed={sorted(observed_coarse)}"
        )

    invalid_modalities = set(frame["modality"]) - {"WLC", "BLC"}
    if invalid_modalities:
        raise ValueError(f"Unknown modalities: {sorted(invalid_modalities)}")
    invalid_json_flags = set(frame["json"]) - {"0", "1"}
    if invalid_json_flags:
        raise ValueError(f"Unknown json flags: {sorted(invalid_json_flags)}")

    frame["coarse_id"] = frame["class"].map(COARSE_TO_ID).astype(int)
    frame["binary_id"] = frame["coarse_id"].map(
        lambda value: int(value in ROI_COARSE_IDS)
    )

    fine_ids: list[int] = []
    taxonomy_errors: list[str] = []
    for coarse_name, fine_name, filename in frame[
        ["class", "subclass", "filename"]
    ].itertuples(index=False, name=None):
        coarse_name = str(coarse_name)
        fine_name = str(fine_name)
        if coarse_name == "Normal mucosa":
            if not is_missing_token(fine_name):
                taxonomy_errors.append(
                    f"{filename}: Normal mucosa has subclass={fine_name}"
                )
            fine_ids.append(-1)
            continue
        if fine_name not in FINE_TO_ID:
            taxonomy_errors.append(
                f"{filename}: unknown subclass={fine_name}"
            )
            fine_ids.append(-1)
            continue
        if fine_name not in FINE_BY_PARENT.get(coarse_name, ()):
            taxonomy_errors.append(
                f"{filename}: {fine_name} is not a child of {coarse_name}"
            )
        fine_ids.append(FINE_TO_ID[fine_name])
    if taxonomy_errors:
        raise ValueError(
            "Taxonomy validation failed; first errors: "
            + " | ".join(taxonomy_errors[:10])
        )
    frame["fine_id"] = np.asarray(fine_ids, dtype=np.int64)

    suffix_counts = Counter(
        Path(value).suffix.lower() for value in frame["filename"]
    )
    logger.info(
        "Manifest loaded: rows=%d patients=%d normalized_suffixes=%s",
        len(frame),
        frame["pid"].nunique(),
        dict(suffix_counts),
    )

    if config["verify_all_image_decodes"]:
        logger.info("Verifying all %d PNG decodes...", len(frame))
        for index, path in enumerate(frame["image_path"], start=1):
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as exc:
                raise RuntimeError(f"Image decode validation failed: {path}") from exc
            if index % 1000 == 0:
                logger.info("Image decode audit: %d/%d", index, len(frame))

    segmented_stems = set()
    if config["verify_segmentation_inventory"]:
        segmented_stems = {
            path.stem for path in segmentation_dir.glob("*.json")
        }
        expected_stems = set(frame.loc[frame["json"] == "1", "image_stem"])
        missing_masks = expected_stems - segmented_stems
        orphan_masks = segmented_stems - expected_stems
        if missing_masks or orphan_masks:
            raise ValueError(
                "Segmentation inventory mismatch: "
                f"missing={len(missing_masks)}, orphan={len(orphan_masks)}"
            )

    fine_support = (
        frame.loc[frame["fine_id"] >= 0]
        .groupby("subclass")
        .agg(images=("filename", "size"), patients=("pid", "nunique"))
        .reindex(FINE_NAMES)
        .fillna(0)
        .astype(int)
    )
    semantic_columns = [
        "image_stem",
        "pid",
        "visit",
        "lesion",
        "class",
        "subclass",
        "modality",
        "binary_id",
        "coarse_id",
        "fine_id",
    ]
    semantic_digest = hashlib.sha256()
    for values in (
        frame.loc[:, semantic_columns]
        .astype(str)
        .sort_values(["image_stem", "pid"])
        .itertuples(index=False, name=None)
    ):
        semantic_digest.update(
            json.dumps(
                values,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        semantic_digest.update(b"\n")

    inventory_rows: list[dict[str, Any]] = []
    duplicate_hashes: defaultdict[str, list[str]] = defaultdict(list)
    inventory_digest = hashlib.sha256()
    full_fingerprint = config["dataset_fingerprint_mode"] == "full"
    hash_images = full_fingerprint or bool(
        config["verify_exact_duplicate_images"]
    )
    for row in frame.sort_values("image_stem").itertuples(index=False):
        image_path = Path(row.image_path)
        image_hash = sha256_file(image_path) if hash_images else None
        inventory_row = {
            "image_stem": str(row.image_stem),
            "bytes": image_path.stat().st_size,
            "sha256": image_hash,
        }
        inventory_rows.append(inventory_row)
        inventory_digest.update(
            json.dumps(
                inventory_row,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        inventory_digest.update(b"\n")
        if image_hash is not None:
            duplicate_hashes[image_hash].append(str(row.image_stem))
    duplicate_groups = [
        {"sha256": digest, "image_stems": stems}
        for digest, stems in sorted(duplicate_hashes.items())
        if len(stems) > 1
    ]
    if hash_images:
        pd.DataFrame(inventory_rows).to_csv(
            run_dir / "reports" / "image_inventory.csv",
            index=False,
        )

    segmentation_digest = hashlib.sha256()
    segmentation_inventory: list[dict[str, Any]] = []
    if config["verify_segmentation_inventory"]:
        for path in sorted(segmentation_dir.glob("*.json")):
            row = {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path) if full_fingerprint else None,
            }
            segmentation_inventory.append(row)
            segmentation_digest.update(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            segmentation_digest.update(b"\n")
    inclusion_hash = (
        sha256_file(Path(inclusion_manifest))
        if inclusion_manifest is not None
        else None
    )
    dataset_fingerprint = {
        "mode": config["dataset_fingerprint_mode"],
        "metadata_csv_sha256": sha256_file(metadata_csv),
        "inclusion_manifest_sha256": inclusion_hash,
        "semantic_manifest_sha256": semantic_digest.hexdigest(),
        "image_inventory_sha256": inventory_digest.hexdigest(),
        "image_hashes_included": hash_images,
        "image_duplicate_groups": duplicate_groups,
        "segmentation_inventory_sha256": (
            segmentation_digest.hexdigest()
            if config["verify_segmentation_inventory"]
            else None
        ),
        "rows": len(frame),
        "patients": int(frame["pid"].nunique()),
    }
    write_json(
        run_dir / "system" / "dataset_fingerprint.json",
        dataset_fingerprint,
    )
    image_size_stats = audit_image_size_distribution(frame, run_dir, logger)
    audit = {
        "rows": len(frame),
        "patients": frame["pid"].nunique(),
        "inclusion_manifest_csv": inclusion_manifest,
        "image_suffix_counts_in_csv": dict(suffix_counts),
        "coarse_image_counts": frame["class"].value_counts().to_dict(),
        "coarse_patient_counts": (
            frame.groupby("class")["pid"].nunique().to_dict()
        ),
        "fine_support": fine_support.reset_index().to_dict(orient="records"),
        "modality_counts": frame["modality"].value_counts().to_dict(),
        "bca_counts": frame["bca"].value_counts().to_dict(),
        "segmentation_files": len(segmented_stems),
        "normal_rows_masked_from_fine_loss": int(
            (frame["fine_id"] < 0).sum()
        ),
        "image_size_statistics": image_size_stats,
        "dataset_fingerprint": dataset_fingerprint,
    }
    write_json(run_dir / "reports" / "data_audit.json", audit)
    fine_support.to_csv(
        run_dir / "reports" / "fine_label_support.csv",
        index=True,
    )
    return frame, audit


def patient_label_matrices(
    frame: pd.DataFrame,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    pids = sorted(frame["pid"].unique().tolist())
    coarse_presence = np.zeros((len(pids), len(COARSE_NAMES)), dtype=np.float64)
    fine_presence = np.zeros((len(pids), len(FINE_NAMES)), dtype=np.float64)
    image_counts = np.zeros((len(pids), len(COARSE_NAMES)), dtype=np.float64)
    pid_to_index = {pid: index for index, pid in enumerate(pids)}
    for row in frame.itertuples(index=False):
        index = pid_to_index[str(row.pid)]
        coarse_presence[index, int(row.coarse_id)] = 1.0
        image_counts[index, int(row.coarse_id)] += 1.0
        if int(row.fine_id) >= 0:
            fine_presence[index, int(row.fine_id)] = 1.0
    return pids, coarse_presence, fine_presence, image_counts


def allocation_score(
    assignments: Sequence[set[str]],
    pid_to_row: Mapping[str, int],
    coarse_presence: np.ndarray,
    fine_presence: np.ndarray,
    image_counts: np.ndarray,
    target_fractions: Sequence[float],
) -> float:
    global_coarse_presence = coarse_presence.sum(axis=0)
    global_fine_presence = fine_presence.sum(axis=0)
    global_images = image_counts.sum(axis=0)
    score = 0.0
    for members, fraction in zip(assignments, target_fractions):
        indices = [pid_to_row[pid] for pid in members]
        if not indices:
            return float("inf")
        local_coarse = coarse_presence[indices].sum(axis=0)
        if np.any(local_coarse == 0):
            return float("inf")
        local_fine = fine_presence[indices].sum(axis=0)
        local_images = image_counts[indices].sum(axis=0)
        score += float(
            np.mean(
                np.abs(
                    local_coarse
                    - global_coarse_presence * fraction
                )
                / np.maximum(global_coarse_presence * fraction, 1.0)
            )
        )
        eligible_fine = global_fine_presence >= len(assignments)
        if eligible_fine.any():
            score += float(
                np.mean(
                    np.abs(
                        local_fine[eligible_fine]
                        - global_fine_presence[eligible_fine] * fraction
                    )
                    / np.maximum(
                        global_fine_presence[eligible_fine] * fraction,
                        1.0,
                    )
                )
            )
        score += float(
            np.mean(
                np.abs(local_images - global_images * fraction)
                / np.maximum(global_images * fraction, 1.0)
            )
        )
    return score


def search_holdout_patient_split(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    seed: int,
    allowed_pids: set[str] | None = None,
    forced_test_pids: set[str] | None = None,
) -> tuple[dict[str, set[str]], float]:
    (
        all_pids,
        coarse_presence,
        fine_presence,
        image_counts,
    ) = patient_label_matrices(frame)
    pid_to_row = {pid: index for index, pid in enumerate(all_pids)}
    eligible = set(all_pids) if allowed_pids is None else set(allowed_pids)
    if not eligible:
        raise ValueError("No patients are eligible for split construction.")
    unknown = eligible - set(all_pids)
    if unknown:
        raise ValueError(f"Unknown allowed patient IDs: {sorted(unknown)}")

    fractions = np.asarray(
        [
            config["train_fraction"],
            config["val_fraction"],
            config["test_fraction"],
        ],
        dtype=np.float64,
    )
    fractions = fractions / fractions.sum()
    target_counts = np.floor(fractions * len(eligible)).astype(int)
    target_counts[0] += len(eligible) - int(target_counts.sum())
    if np.any(target_counts < 1):
        raise ValueError(
            f"Not enough patients for requested split: {target_counts.tolist()}"
        )

    forced_train: set[str] = set()
    threshold = int(
        config["force_fine_labels_with_fewer_than_n_patients_to_train"]
    )
    if threshold > 0:
        eligible_frame = frame[frame["pid"].isin(eligible)]
        fine_patient_support = (
            eligible_frame.loc[eligible_frame["fine_id"] >= 0]
            .groupby("fine_id")["pid"]
            .nunique()
        )
        rare_ids = set(
            fine_patient_support[
                fine_patient_support < threshold
            ].index.astype(int)
        )
        forced_train = set(
            eligible_frame.loc[
                eligible_frame["fine_id"].isin(rare_ids), "pid"
            ]
        )
    if forced_test_pids:
        forced_test = set(forced_test_pids)
        if not forced_test <= eligible:
            raise ValueError("forced_test_pids must be a subset of allowed_pids.")
        forced_train -= forced_test
    else:
        forced_test = set()
    if len(forced_train) > target_counts[0]:
        raise ValueError(
            "Rare-label constraints require more training patients than the "
            "requested training split can contain."
        )
    if len(forced_test) > target_counts[2]:
        raise ValueError("forced_test_pids exceed the requested test size.")

    remaining = sorted(eligible - forced_train - forced_test)
    rng = np.random.default_rng(seed)
    best: tuple[float, dict[str, set[str]]] | None = None
    for _ in range(int(config["split_search_candidates"])):
        permutation = rng.permutation(remaining).tolist()
        train_needed = int(target_counts[0]) - len(forced_train)
        val_needed = int(target_counts[1])
        test_needed = int(target_counts[2]) - len(forced_test)
        if train_needed + val_needed + test_needed != len(permutation):
            raise RuntimeError("Internal patient allocation count mismatch.")
        train = forced_train | set(permutation[:train_needed])
        val_start = train_needed
        val = set(permutation[val_start : val_start + val_needed])
        test = forced_test | set(permutation[val_start + val_needed :])
        assignments = (train, val, test)
        if config["run_profile"] == "research":
            train_indices = [pid_to_row[pid] for pid in train]
            if np.any(fine_presence[train_indices].sum(axis=0) == 0):
                continue
        score = allocation_score(
            assignments,
            pid_to_row,
            coarse_presence,
            fine_presence,
            image_counts,
            fractions,
        )
        if not math.isfinite(score):
            continue
        if best is None or score < best[0]:
            best = (score, {"train": train, "val": val, "test": test})
    if best is None:
        raise RuntimeError(
            "No valid patient-disjoint split was found. Increase "
            "split_search_candidates or revise split constraints."
        )
    return best[1], best[0]


def search_patient_folds(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    seed: int,
) -> tuple[list[set[str]], float]:
    (
        pids,
        coarse_presence,
        fine_presence,
        image_counts,
    ) = patient_label_matrices(frame)
    n_folds = int(config["cv_folds"])
    if n_folds < 2 or n_folds > len(pids):
        raise ValueError("cv_folds must be between 2 and the patient count.")
    pid_to_row = {pid: index for index, pid in enumerate(pids)}
    fold_sizes = [
        len(part) for part in np.array_split(np.arange(len(pids)), n_folds)
    ]
    fractions = [size / len(pids) for size in fold_sizes]
    rng = np.random.default_rng(seed)
    best: tuple[float, list[set[str]]] | None = None
    for _ in range(int(config["split_search_candidates"])):
        permutation = rng.permutation(pids).tolist()
        folds: list[set[str]] = []
        cursor = 0
        for size in fold_sizes:
            folds.append(set(permutation[cursor : cursor + size]))
            cursor += size
        score = allocation_score(
            folds,
            pid_to_row,
            coarse_presence,
            fine_presence,
            image_counts,
            fractions,
        )
        if not math.isfinite(score):
            continue
        if best is None or score < best[0]:
            best = (score, folds)
    if best is None:
        raise RuntimeError(
            "No valid multilabel patient-fold allocation was found."
        )
    return best[1], best[0]


def search_train_val_patient_split(
    frame: pd.DataFrame,
    allowed_pids: set[str],
    val_fraction: float,
    candidates: int,
    seed: int,
) -> tuple[set[str], set[str], float]:
    if not 0 < val_fraction < 1:
        raise ValueError("CV validation fraction must be in (0, 1).")
    (
        all_pids,
        coarse_presence,
        fine_presence,
        image_counts,
    ) = patient_label_matrices(frame)
    known = set(all_pids)
    if not allowed_pids <= known:
        raise ValueError("CV train/validation pool contains unknown patients.")
    train_count = round(len(allowed_pids) * (1.0 - val_fraction))
    train_count = min(max(train_count, 1), len(allowed_pids) - 1)
    pid_to_row = {pid: index for index, pid in enumerate(all_pids)}
    rng = np.random.default_rng(seed)
    ordered = sorted(allowed_pids)
    best: tuple[float, set[str], set[str]] | None = None
    fractions = (
        train_count / len(allowed_pids),
        1.0 - train_count / len(allowed_pids),
    )
    for _ in range(int(candidates)):
        permutation = rng.permutation(ordered).tolist()
        train = set(permutation[:train_count])
        val = set(permutation[train_count:])
        score = allocation_score(
            (train, val),
            pid_to_row,
            coarse_presence,
            fine_presence,
            image_counts,
            fractions,
        )
        if not math.isfinite(score):
            continue
        if best is None or score < best[0]:
            best = (score, train, val)
    if best is None:
        raise RuntimeError("No valid CV train/validation allocation was found.")
    return best[1], best[2], best[0]


def fixed_patient_split(
    frame: pd.DataFrame,
    fixed: Mapping[str, Sequence[str]],
) -> dict[str, set[str]]:
    required = {"train", "val", "test"}
    if set(fixed) != required:
        raise ValueError(
            f"fixed_split_pids must have exactly {sorted(required)} keys."
        )
    split = {key: set(map(str, fixed[key])) for key in required}
    if any(not values for values in split.values()):
        raise ValueError("Every fixed split must contain at least one patient.")
    if (
        split["train"] & split["val"]
        or split["train"] & split["test"]
        or split["val"] & split["test"]
    ):
        raise ValueError("Fixed patient splits overlap.")
    known = set(frame["pid"])
    unknown = set.union(*split.values()) - known
    if unknown:
        raise ValueError(f"Fixed split contains unknown PIDs: {sorted(unknown)}")
    return split


def sample_rows_stratified(
    frame: pd.DataFrame,
    limit: int | None,
    seed: int,
) -> pd.DataFrame:
    if limit is None or len(frame) <= int(limit):
        return frame.copy()
    limit = int(limit)
    if limit < frame["coarse_id"].nunique():
        raise ValueError(
            "A sample cap cannot be smaller than the number of coarse classes."
        )
    rng = np.random.default_rng(seed)
    groups = {
        int(label): group.index.to_numpy()
        for label, group in frame.groupby("coarse_id")
    }
    allocation = {
        label: max(1, round(limit * len(indices) / len(frame)))
        for label, indices in groups.items()
    }
    while sum(allocation.values()) > limit:
        candidates = [
            label for label, count in allocation.items() if count > 1
        ]
        if not candidates:
            raise RuntimeError("Unable to reduce stratified allocation.")
        label = max(candidates, key=lambda item: allocation[item])
        allocation[label] -= 1
    while sum(allocation.values()) < limit:
        candidates = [
            label
            for label, indices in groups.items()
            if allocation[label] < len(indices)
        ]
        if not candidates:
            raise RuntimeError("Unable to increase stratified allocation.")
        label = max(
            candidates,
            key=lambda item: len(groups[item]) - allocation[item],
        )
        allocation[label] += 1
    selected: list[int] = []
    for label, indices in groups.items():
        count = min(allocation[label], len(indices))
        selected.extend(rng.choice(indices, size=count, replace=False).tolist())
    if len(selected) != limit:
        raise RuntimeError("Stratified sample size does not match the limit.")
    return frame.loc[sorted(selected)].copy()


def cap_normal_mucosa_across_splits(
    split_frames: Mapping[str, pd.DataFrame],
    total_limit: int | None,
    fractions: Mapping[str, float],
    seed: int,
) -> dict[str, pd.DataFrame]:
    if total_limit is None:
        return {name: frame.copy() for name, frame in split_frames.items()}
    total_limit = int(total_limit)
    if total_limit < 0:
        raise ValueError("normal_mucosa_limit cannot be negative.")
    desired = {
        name: math.floor(total_limit * fractions[name])
        for name in split_frames
    }
    remainder = total_limit - sum(desired.values())
    for name in ("train", "val", "test"):
        if remainder <= 0:
            break
        desired[name] += 1
        remainder -= 1

    output: dict[str, pd.DataFrame] = {}
    normal_id = COARSE_TO_ID["Normal mucosa"]
    for name, frame in split_frames.items():
        normal = frame[frame["coarse_id"] == normal_id]
        other = frame[frame["coarse_id"] != normal_id]
        keep = min(desired[name], len(normal))
        if keep:
            normal = normal.sample(
                n=keep,
                random_state=stable_int_seed(seed, name, "normal"),
                replace=False,
            )
        else:
            normal = normal.iloc[0:0]
        output[name] = (
            pd.concat([other, normal], ignore_index=True)
            .sample(
                frac=1.0,
                random_state=stable_int_seed(seed, name, "shuffle"),
            )
            .reset_index(drop=True)
        )
    return output


def materialize_split_frames(
    frame: pd.DataFrame,
    patient_split: Mapping[str, set[str]],
    config: Mapping[str, Any],
    seed: int,
) -> dict[str, pd.DataFrame]:
    train_pids = patient_split["train"]
    val_pids = patient_split["val"]
    test_pids = patient_split["test"]
    if train_pids & val_pids or train_pids & test_pids or val_pids & test_pids:
        raise RuntimeError("Patient leakage detected before split materialization.")

    frames = {
        name: frame[frame["pid"].isin(pids)].copy().reset_index(drop=True)
        for name, pids in patient_split.items()
    }
    if any(part.empty for part in frames.values()):
        raise ValueError("Every materialized split must contain at least one row.")

    fractions = {
        name: len(pids) / sum(len(value) for value in patient_split.values())
        for name, pids in patient_split.items()
    }
    frames = cap_normal_mucosa_across_splits(
        frames,
        config["normal_mucosa_limit"],
        fractions,
        seed,
    )
    limits = {
        "train": config["max_train_samples"],
        "val": config["max_val_samples"],
        "test": config["max_test_samples"],
    }
    for name in frames:
        frames[name] = sample_rows_stratified(
            frames[name],
            limits[name],
            stable_int_seed(seed, name, "cap"),
        ).reset_index(drop=True)
    return frames


def validate_materialized_splits(
    split_frames: Mapping[str, pd.DataFrame],
    run_profile: str,
) -> None:
    pid_sets = {
        name: set(frame["pid"]) for name, frame in split_frames.items()
    }
    if (
        pid_sets["train"] & pid_sets["val"]
        or pid_sets["train"] & pid_sets["test"]
        or pid_sets["val"] & pid_sets["test"]
    ):
        raise RuntimeError("Patient leakage detected after row sampling.")
    for name, frame in split_frames.items():
        observed = set(frame["coarse_id"].astype(int))
        if observed != set(range(len(COARSE_NAMES))):
            raise ValueError(
                f"{name} lacks coarse classes after sampling: "
                f"{set(range(len(COARSE_NAMES))) - observed}"
            )
    if run_profile == "research":
        train_fine = set(
            split_frames["train"].loc[
                split_frames["train"]["fine_id"] >= 0, "fine_id"
            ].astype(int)
        )
        if len(train_fine) != len(FINE_NAMES):
            missing = [FINE_NAMES[index] for index in set(range(22)) - train_fine]
            raise ValueError(
                "Research holdout training split lacks fine labels: "
                f"{missing}. Increase split search or adjust rare constraints."
            )


def save_split_artifacts(
    split_frames: Mapping[str, pd.DataFrame],
    patient_split: Mapping[str, set[str]],
    score: float | None,
    run_dir: Path,
    fold_name: str,
) -> dict[str, Any]:
    fold_dir = run_dir / "splits" / fold_name
    fold_dir.mkdir(parents=True, exist_ok=False)
    dataset_fingerprint_path = (
        run_dir / "system" / "dataset_fingerprint.json"
    )
    if not dataset_fingerprint_path.is_file():
        raise FileNotFoundError(
            "Dataset fingerprint must exist before split artifacts are saved."
        )
    dataset_fingerprint = json.loads(
        dataset_fingerprint_path.read_text(encoding="utf-8")
    )
    summary: dict[str, Any] = {
        "allocation_score": score,
        "data_split_fingerprint": split_fingerprint(split_frames),
        "data_split_fingerprint_algorithm": "pid_stem_labels_v2",
        "dataset_semantic_manifest_sha256": dataset_fingerprint[
            "semantic_manifest_sha256"
        ],
        "splits": {},
    }
    total_materialized_rows = sum(len(frame) for frame in split_frames.values())
    total_assigned_patients = sum(len(pids) for pids in patient_split.values())
    columns = [
        "filename",
        "image_path",
        "pid",
        "visit",
        "lesion",
        "class",
        "subclass",
        "binary_id",
        "coarse_id",
        "fine_id",
        "modality",
        "json",
    ]
    for name, frame in split_frames.items():
        frame.loc[:, columns].to_csv(fold_dir / f"{name}.csv", index=False)
        summary["splits"][name] = {
            "rows": len(frame),
            "patients": frame["pid"].nunique(),
            "materialized_image_fraction": (
                len(frame) / total_materialized_rows
            ),
            "assigned_patient_fraction": (
                len(patient_split[name]) / total_assigned_patients
            ),
            "patient_ids": sorted(patient_split[name]),
            "coarse_counts": frame["class"].value_counts().to_dict(),
            "fine_counts": (
                frame.loc[frame["fine_id"] >= 0, "subclass"]
                .value_counts()
                .to_dict()
            ),
        }
    combined_list = []
    for name, frame in split_frames.items():
        sub_df = frame.loc[:, columns].copy()
        sub_df.insert(1, "split", name)
        combined_list.append(sub_df)
    if combined_list:
        combined_df = pd.concat(combined_list, ignore_index=True)
        combined_df.to_csv(fold_dir / "cystods_split.csv", index=False)
        combined_df.to_csv(run_dir / "cystods_split.csv", index=False)

    write_json(fold_dir / "summary.json", summary)
    return summary


def load_frozen_protocol_splits(
    frame: pd.DataFrame,
    protocol_manifest_dir: Path,
    config: Mapping[str, Any],
    run_dir: Path,
    logger: logging.Logger,
) -> list[tuple[str, dict[str, pd.DataFrame], dict[str, set[str]]]]:
    if not protocol_manifest_dir.is_dir():
        if protocol_manifest_dir.parent.is_dir():
            protocol_manifest_dir = protocol_manifest_dir.parent

    if not protocol_manifest_dir.is_dir():
        raise FileNotFoundError(
            f"Frozen protocol directory not found: {protocol_manifest_dir}"
        )
    if config["fixed_split_pids"] is not None:
        raise ValueError(
            "fixed_split_pids cannot be combined with protocol_manifest_dir."
        )
    if (protocol_manifest_dir / "train.csv").is_file():
        unit_dirs = [protocol_manifest_dir]
    else:
        discovered_dirs = sorted(
            summary_path.parent
            for summary_path in protocol_manifest_dir.rglob("summary.json")
            if (summary_path.parent / "train.csv").is_file()
        )
        if config["protocol"] == "holdout":
            unit_dirs = [
                path
                for path in discovered_dirs
                if path.name == "holdout"
                or path.parent.name == "holdout"
                or "holdout" in str(path)
            ]
        else:
            unit_dirs = [
                path
                for path in discovered_dirs
                if path.name.startswith("fold_")
                or path.parent.name.startswith("fold_")
            ]
    if not unit_dirs:
        raise FileNotFoundError(
            "Frozen protocol contains no complete split units."
        )
    requested = config["cv_run_fold_indices"]
    if requested is not None:
        requested_names = {
            f"fold_{int(index):02d}" for index in requested
        }
        unit_dirs = [
            path for path in unit_dirs if path.name in requested_names
        ]
        if {path.name for path in unit_dirs} != requested_names:
            raise ValueError(
                "Requested folds are not all present in frozen protocol."
            )

    dataset_fingerprint = json.loads(
        (
            run_dir / "system" / "dataset_fingerprint.json"
        ).read_text(encoding="utf-8")
    )
    by_stem = frame.set_index("image_stem", verify_integrity=True)
    output: list[
        tuple[str, dict[str, pd.DataFrame], dict[str, set[str]]]
    ] = []
    for unit_dir in unit_dirs:
        source_summary = json.loads(
            (unit_dir / "summary.json").read_text(encoding="utf-8")
        )
        expected_dataset_hash = source_summary.get(
            "dataset_semantic_manifest_sha256"
        )
        if (
            expected_dataset_hash
            != dataset_fingerprint["semantic_manifest_sha256"]
        ):
            raise ValueError(
                "Frozen protocol dataset fingerprint does not match the "
                f"active dataset for {unit_dir.name}."
            )
        split_frames: dict[str, pd.DataFrame] = {}
        patient_split: dict[str, set[str]] = {}
        for split_name in ("train", "val", "test"):
            csv_path = unit_dir / f"{split_name}.csv"
            if not csv_path.is_file():
                raise FileNotFoundError(csv_path)
            persisted = pd.read_csv(
                csv_path,
                dtype=str,
                keep_default_na=False,
            )
            if "filename" not in persisted:
                raise ValueError(
                    f"Frozen split lacks filename column: {csv_path}"
                )
            stems = persisted["filename"].map(
                lambda value: Path(value).stem
            )
            if stems.duplicated().any():
                raise ValueError(
                    f"Frozen split has duplicate image stems: {csv_path}"
                )
            unknown = sorted(set(stems) - set(by_stem.index))
            if unknown:
                raise ValueError(
                    "Frozen split references images outside the active "
                    f"dataset; first={unknown[0]}"
                )
            restored = by_stem.loc[stems.tolist()].reset_index()
            for column in (
                "pid",
                "binary_id",
                "coarse_id",
                "fine_id",
                "modality",
            ):
                if column in persisted:
                    left = restored[column].astype(str).reset_index(drop=True)
                    right = persisted[column].astype(str).reset_index(drop=True)
                    if not left.equals(right):
                        raise ValueError(
                            "Frozen split metadata mismatch for "
                            f"{unit_dir.name}/{split_name}/{column}."
                        )
            split_frames[split_name] = restored
            patient_split[split_name] = set(
                restored["pid"].astype(str)
            )
        validate_materialized_splits(split_frames, "smoke")
        computed = split_fingerprint(split_frames)
        if computed != source_summary["data_split_fingerprint"]:
            raise ValueError(
                f"Frozen split fingerprint mismatch for {unit_dir.name}."
            )
        fold_name = unit_dir.name
        save_split_artifacts(
            split_frames,
            patient_split,
            source_summary.get("allocation_score"),
            run_dir,
            fold_name,
        )
        logger.info(
            "Loaded frozen protocol unit=%s fingerprint=%s",
            fold_name,
            computed,
        )
        output.append((fold_name, split_frames, patient_split))
    return output


def build_all_protocol_splits(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    run_dir: Path,
    logger: logging.Logger,
) -> list[tuple[str, dict[str, pd.DataFrame], dict[str, set[str]]]]:
    output: list[
        tuple[str, dict[str, pd.DataFrame], dict[str, set[str]]]
    ] = []
    seed = int(config["split_seed"])
    if config["protocol_manifest_dir"] is not None:
        return load_frozen_protocol_splits(
            frame,
            Path(config["protocol_manifest_dir"]),
            config,
            run_dir,
            logger,
        )
    if config["fixed_split_pids"] is not None:
        if config["protocol"] != "holdout":
            raise ValueError("fixed_split_pids is only valid for holdout.")
        patient_split = fixed_patient_split(
            frame, config["fixed_split_pids"]
        )
        score = None
        split_frames = materialize_split_frames(
            frame, patient_split, config, seed
        )
        validate_materialized_splits(split_frames, config["run_profile"])
        save_split_artifacts(
            split_frames, patient_split, score, run_dir, "holdout"
        )
        output.append(("holdout", split_frames, patient_split))
        return output

    if config["protocol"] == "holdout":
        patient_split, score = search_holdout_patient_split(
            frame, config, seed
        )
        split_frames = materialize_split_frames(
            frame, patient_split, config, seed
        )
        validate_materialized_splits(split_frames, config["run_profile"])
        save_split_artifacts(
            split_frames, patient_split, score, run_dir, "holdout"
        )
        logger.info("Selected patient split with score %.6f", score)
        output.append(("holdout", split_frames, patient_split))
        return output

    folds, score = search_patient_folds(frame, config, seed)
    requested = config["cv_run_fold_indices"]
    indices = list(range(len(folds))) if requested is None else list(requested)
    invalid = set(indices) - set(range(len(folds)))
    if invalid:
        raise ValueError(f"Invalid CV fold indices: {sorted(invalid)}")
    all_pids = set(frame["pid"])
    for fold_index in indices:
        test_pids = folds[fold_index]
        remaining_pids = all_pids - test_pids
        val_fraction = float(config["cv_val_fraction_of_remaining"])
        train_pids, val_pids, inner_score = search_train_val_patient_split(
            frame,
            remaining_pids,
            val_fraction,
            int(config["split_search_candidates"]),
            stable_int_seed(seed, "cv", fold_index),
        )
        patient_split = {
            "train": train_pids,
            "val": val_pids,
            "test": set(test_pids),
        }
        split_frames = materialize_split_frames(
            frame,
            patient_split,
            config,
            stable_int_seed(seed, "materialize", fold_index),
        )
        # In CV, a singleton fine label must be absent from training in one
        # fold. Coarse coverage and patient isolation remain strict.
        validate_materialized_splits(split_frames, "smoke")
        fold_name = f"fold_{fold_index:02d}"
        save_split_artifacts(
            split_frames,
            patient_split,
            score + inner_score,
            run_dir,
            fold_name,
        )
        output.append((fold_name, split_frames, patient_split))
    write_json(
        run_dir / "splits" / "cv_partition.json",
        {
            "allocation_score": score,
            "folds": [sorted(fold) for fold in folds],
        },
    )
    return output


# %%
# CELL 5 - Optimized image pipeline, hierarchical model, and objectives.
class CenterFractionCrop:
    def __init__(self, ratio: float) -> None:
        if not 0 < ratio <= 1:
            raise ValueError("CenterFractionCrop ratio must be in (0, 1].")
        self.ratio = float(ratio)

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        crop_width = max(1, round(width * self.ratio))
        crop_height = max(1, round(height * self.ratio))
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        return image.crop(
            (left, top, left + crop_width, top + crop_height)
        )


def build_transforms(
    config: Mapping[str, Any],
) -> tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    image_size = int(config["image_size"])
    mean = tuple(config["imagenet_mean"])
    std = tuple(config["imagenet_std"])
    center_crop = CenterFractionCrop(config["fov_center_crop_ratio"])
    eval_transform = transforms.Compose(
        [
            center_crop,
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    supcon_transform = transforms.Compose(
        [
            center_crop,
            transforms.RandomResizedCrop(
                image_size,
                scale=tuple(config["random_resized_crop_scale"]),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(
                p=float(config["horizontal_flip_probability"])
            ),
            transforms.RandomVerticalFlip(
                p=float(config["vertical_flip_probability"])
            ),
            transforms.RandomRotation(
                degrees=float(config["rotation_degrees"]),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ColorJitter(*tuple(config["color_jitter"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    use_aug = bool(config.get("use_data_augmentation", False))
    if use_aug:
        train_transform = transforms.Compose(
            [
                center_crop,
                transforms.RandomResizedCrop(
                    image_size,
                    scale=tuple(config["random_resized_crop_scale"]),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.RandomHorizontalFlip(
                    p=float(config["horizontal_flip_probability"])
                ),
                transforms.RandomVerticalFlip(
                    p=float(config["vertical_flip_probability"])
                ),
                transforms.RandomRotation(
                    degrees=float(config["rotation_degrees"]),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ColorJitter(*tuple(config["color_jitter"])),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                transforms.RandomErasing(
                    p=float(config["random_erasing_probability"]),
                    scale=(0.02, 0.15),
                    ratio=(0.3, 3.3),
                    value="random",
                ),
            ]
        )
    else:
        train_transform = eval_transform
    return train_transform, eval_transform, supcon_transform


class CystoDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        transform: Any,
        second_view_transform: Any | None = None,
        *,
        second_view: bool | None = None,
    ) -> None:
        if frame.empty:
            raise ValueError("CystoDataset cannot be constructed from no rows.")
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = transform
        if second_view_transform is not None:
            self.second_view_transform = second_view_transform
        elif second_view:
            self.second_view_transform = transform
        else:
            self.second_view_transform = None
        self.second_view = self.second_view_transform is not None
        # Materialize hot-path columns once. Repeated pandas.iloc calls inside
        # DataLoader workers are measurably slower and increase serialization
        # overhead, especially with persistent workers.
        self.image_paths = self.frame["image_path"].astype(str).tolist()
        self.binary_ids = self.frame["binary_id"].astype(int).to_numpy()
        self.coarse_ids = self.frame["coarse_id"].astype(int).to_numpy()
        self.fine_ids = self.frame["fine_id"].astype(int).to_numpy()
        self.filenames = self.frame["filename"].astype(str).tolist()
        self.pids = self.frame["pid"].astype(str).tolist()
        self.visits = self.frame["visit"].astype(str).tolist()
        self.lesions = self.frame["lesion"].astype(str).tolist()
        self.modalities = self.frame["modality"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = Path(self.image_paths[index])
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
                view_one = self.transform(image)
                view_two = (
                    self.second_view_transform(image)
                    if self.second_view_transform is not None
                    else None
                )
        except Exception as exc:
            raise RuntimeError(f"Failed to decode/transform image: {path}") from exc
        item: dict[str, Any] = {
            "image": view_one,
            "binary_id": torch.tensor(
                self.binary_ids[index], dtype=torch.long
            ),
            "coarse_id": torch.tensor(
                self.coarse_ids[index], dtype=torch.long
            ),
            "fine_id": torch.tensor(self.fine_ids[index], dtype=torch.long),
            "row_index": torch.tensor(index, dtype=torch.long),
            "filename": self.filenames[index],
            "pid": self.pids[index],
            "visit": self.visits[index],
            "lesion": self.lesions[index],
            "modality": self.modalities[index],
        }
        if view_two is not None:
            item["image_view_2"] = view_two
        return item


class ExternalBinaryDataset(Dataset):
    def __init__(
        self,
        manifest_csv: Path,
        image_root: Path,
        transform: Any,
        config: Mapping[str, Any],
    ) -> None:
        if not manifest_csv.is_file():
            raise FileNotFoundError(
                f"External manifest not found: {manifest_csv}"
            )
        if not image_root.is_dir():
            raise FileNotFoundError(
                f"External image root not found: {image_root}"
            )
        frame = pd.read_csv(
            manifest_csv, dtype=str, keep_default_na=False
        )
        path_col = str(config["external_path_column"])
        label_col = str(config["external_binary_label_column"])
        pid_col = str(config["external_patient_id_column"])
        required = {path_col, label_col, pid_col}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"External manifest columns missing: {sorted(missing)}"
            )
        labels = pd.to_numeric(frame[label_col], errors="raise").astype(int)
        if not set(labels).issubset({0, 1}):
            raise ValueError("External binary labels must be 0 or 1.")
        frame = frame.copy()
        frame["resolved_path"] = frame[path_col].map(
            lambda value: str(image_root / value)
        )
        missing_files = [
            path
            for path in frame["resolved_path"]
            if not Path(path).is_file()
        ]
        if missing_files:
            raise FileNotFoundError(
                f"External images missing; first={missing_files[0]}"
            )
        frame["binary_id"] = labels
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.path_col = path_col
        self.pid_col = pid_col
        self.paths = self.frame["resolved_path"].astype(str).tolist()
        self.labels = self.frame["binary_id"].astype(int).to_numpy()
        self.filenames = self.frame[path_col].astype(str).tolist()
        self.pids = self.frame[pid_col].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = Path(self.paths[index])
        try:
            with Image.open(path) as source:
                image = self.transform(source.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"External image decode failed: {path}") from exc
        return {
            "image": image,
            "binary_id": torch.tensor(
                self.labels[index], dtype=torch.long
            ),
            "filename": self.filenames[index],
            "pid": self.pids[index],
        }


class _WorkerInitFn:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int) -> None:
        worker_seed = (self.seed + worker_id) % (2**32 - 1)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)


def make_worker_init_fn(seed: int):
    return _WorkerInitFn(seed)


def effective_number_weights(
    counts: np.ndarray,
    beta: float,
    require_positive: bool,
) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    if require_positive and np.any(counts <= 0):
        missing = np.flatnonzero(counts <= 0).tolist()
        raise ValueError(
            "This class-balanced operation requires positive counts for "
            f"every class; missing class IDs={missing}"
        )
    weights = np.zeros_like(counts, dtype=np.float64)
    positive = counts > 0
    if beta <= 0 or beta >= 1:
        raise ValueError("class_balance_beta must be strictly between 0 and 1.")
    effective = 1.0 - np.power(beta, counts[positive])
    weights[positive] = (1.0 - beta) / effective
    if positive.any():
        weights[positive] *= positive.sum() / weights[positive].sum()
    return weights


def build_sample_weights(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> torch.DoubleTensor:
    level = config["sampler_label_level"]
    if level == "coarse":
        labels = frame["coarse_id"].astype(int).to_numpy()
        counts = np.bincount(labels, minlength=len(COARSE_NAMES))
        weights = effective_number_weights(
            counts,
            float(config["class_balance_beta"]),
            require_positive=True,
        )
        return torch.as_tensor(weights[labels], dtype=torch.double)
    if level == "fine":
        labels = frame["fine_id"].astype(int).to_numpy()
        valid = labels >= 0
        if not valid.any():
            raise ValueError("No fine-labelled samples exist for the sampler.")
        counts = np.bincount(labels[valid], minlength=len(FINE_NAMES))
        fine_weights = effective_number_weights(
            counts,
            float(config["class_balance_beta"]),
            require_positive=False,
        )
        # Normal mucosa has no fine target. Give it the median positive sample
        # weight explicitly rather than inventing a 23rd fine class.
        positive_weights = fine_weights[fine_weights > 0]
        if not len(positive_weights):
            raise ValueError("Fine sampler weights contain no positive value.")
        sample_weights = np.full(
            len(labels),
            float(np.median(positive_weights)),
            dtype=np.float64,
        )
        sample_weights[valid] = fine_weights[labels[valid]]
        return torch.as_tensor(sample_weights, dtype=torch.double)
    raise ValueError("sampler_label_level must be fine or coarse.")


def build_dataloaders(
    split_frames: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[dict[str, DataLoader], dict[str, CystoDataset]]:
    train_transform, eval_transform, supcon_transform = build_transforms(config)
    supcon_weight = float(config.get("supervised_contrastive_loss_weight", 0.0))
    second_view_transform = supcon_transform if supcon_weight > 0 else None
    datasets = {
        "train": CystoDataset(
            split_frames["train"],
            train_transform,
            second_view_transform=second_view_transform,
        ),
        "val": CystoDataset(
            split_frames["val"],
            eval_transform,
            second_view_transform=None,
        ),
    }
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = None
    shuffle = True
    if config["sampler"] == "class_balanced":
        weights = build_sample_weights(split_frames["train"], config)
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )
        shuffle = False

    train_workers = (
        0 if device.type == "mps" else int(config["num_workers"])
    )
    eval_workers = (
        0 if device.type == "mps" else int(config["eval_num_workers"])
    )
    train_kwargs: dict[str, Any] = {
        "batch_size": int(config["batch_size"]),
        "num_workers": train_workers,
        "pin_memory": bool(config.get("pin_memory", True) and device.type == "cuda"),
        "persistent_workers": bool(
            config["persistent_workers"] and train_workers > 0
        ),
        "worker_init_fn": make_worker_init_fn(seed),
        "generator": generator,
    }
    if train_workers > 0:
        train_kwargs["prefetch_factor"] = int(config["prefetch_factor"])
    eval_kwargs: dict[str, Any] = {
        "batch_size": int(config["eval_batch_size"]),
        "num_workers": eval_workers,
        "pin_memory": bool(
            config.get("pin_memory", True) and device.type == "cuda"
        ),
        # Validation is visited once per epoch. Keeping a second persistent
        # worker pool alive wastes host RAM and shared memory on large batches.
        "persistent_workers": False,
        "worker_init_fn": make_worker_init_fn(stable_int_seed(seed, "val")),
    }
    if eval_workers > 0:
        eval_kwargs["prefetch_factor"] = int(
            config["eval_prefetch_factor"]
        )
    loaders = {
        "train": DataLoader(
            datasets["train"],
            shuffle=shuffle,
            sampler=sampler,
            drop_last=False,
            **train_kwargs,
        ),
        "val": DataLoader(
            datasets["val"],
            shuffle=False,
            drop_last=False,
            **eval_kwargs,
        ),
    }
    if not loaders["train"]:
        raise ValueError("Training DataLoader contains no batches.")
    return loaders, datasets


PAPER_BASELINE_BACKBONES: dict[str, str] = {
    "resnet152": "resnet152.a1_in1k",
    "resnet152d": "resnet152d",
    "hrnet_w18": "hrnet_w18.ms_in1k",
    "resnext50_32x4d": "resnext50_32x4d.a1_in1k",
    "resnext50": "resnext50_32x4d.a1_in1k",
    "swin_tiny": "swin_tiny_patch4_window7_224.ms_in1k",
}


def resolve_model_name(model_name: str) -> str:
    cleaned = str(model_name).strip()
    key = cleaned.lower()
    if key in PAPER_BASELINE_BACKBONES:
        return PAPER_BASELINE_BACKBONES[key]
    return cleaned


class HierarchicalCystoModel(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        task_mode = str(config["task_mode"])
        if task_mode == "binary":
            active_tasks = {"binary"}
        elif task_mode == "coarse":
            active_tasks = {"coarse"}
        else:
            active_tasks = active_tasks_from_config(config)
        if not active_tasks:
            raise ValueError("Model configuration activates no prediction head.")
        self.task_mode = task_mode
        self.active_tasks = frozenset(active_tasks)
        raw_model_name = str(config["model_name"])
        model_name = resolve_model_name(raw_model_name)
        if not timm.is_model(model_name):
            raise ValueError(
                f"timm model '{model_name}' (resolved from '{raw_model_name}') is unavailable in "
                f"timm {timm.__version__}."
            )
        encoder_kwargs: dict[str, Any] = {
            "pretrained": bool(config["pretrained"]),
            "num_classes": 0,
            "global_pool": "avg",
            "img_size": int(config["image_size"]),
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
                f"Failed to construct encoder '{model_name}' with "
                f"pretrained={config['pretrained']}. No random-weight "
                "fallback is permitted."
            ) from exc
        feature_dim = int(getattr(self.encoder, "num_features", 0))
        if feature_dim <= 0:
            raise RuntimeError(
                f"Encoder '{model_name}' does not expose a valid num_features."
            )
        self.feature_dim = feature_dim
        dropout = float(config["dropout"])
        self.dropout = nn.Dropout(dropout)
        self.binary_head = (
            nn.Linear(feature_dim, len(BINARY_NAMES))
            if "binary" in active_tasks
            else None
        )
        self.coarse_head = (
            nn.Linear(feature_dim, len(COARSE_NAMES))
            if "coarse" in active_tasks
            else None
        )
        self.fine_head = (
            nn.Linear(feature_dim, len(FINE_NAMES))
            if "fine" in active_tasks
            else None
        )
        projection_dim = int(config["projection_dim"])
        self.projection_head = (
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, projection_dim),
            )
            if float(config["supervised_contrastive_loss_weight"]) > 0
            else None
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        if isinstance(features, (tuple, list)):
            raise TypeError(
                "Encoder returned multiple feature tensors; select a timm "
                "classifier model with num_classes=0 support."
            )
        if features.ndim > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise RuntimeError(
                f"Unexpected encoder shape {tuple(features.shape)}; "
                f"expected [batch, {self.feature_dim}]."
            )
        return features

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encode(images)
        dropped = self.dropout(features)
        outputs: dict[str, torch.Tensor] = {
            "features": features,
        }
        if self.projection_head is not None:
            outputs["projection"] = F.normalize(
                self.projection_head(dropped), dim=1
            )
        if self.binary_head is not None:
            outputs["binary_logits"] = self.binary_head(dropped)
        if self.coarse_head is not None:
            outputs["coarse_logits"] = self.coarse_head(dropped)
        if self.fine_head is not None:
            outputs["fine_logits"] = self.fine_head(dropped)
        return outputs


class FineLongTailLoss(nn.Module):
    def __init__(
        self,
        loss_name: str,
        class_counts: Sequence[int],
        patient_counts: Sequence[int],
        config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.loss_name = loss_name
        counts = np.asarray(class_counts, dtype=np.float64)
        patient_count_array = np.asarray(patient_counts, dtype=np.float64)
        if counts.shape != (len(FINE_NAMES),):
            raise ValueError("Fine class_counts must have shape (22,).")
        if patient_count_array.shape != (len(FINE_NAMES),):
            raise ValueError("Fine patient_counts must have shape (22,).")
        if (
            not np.isfinite(counts).all()
            or not np.isfinite(patient_count_array).all()
            or np.any(counts < 0)
            or np.any(patient_count_array < 0)
            or not np.equal(counts, np.floor(counts)).all()
            or not np.equal(patient_count_array, np.floor(patient_count_array)).all()
        ):
            raise ValueError(
                "Fine image and patient counts must be finite non-negative integers."
            )
        if np.any(patient_count_array > counts):
            raise ValueError(
                "Fine patient counts cannot exceed fine image counts."
            )
        if loss_name not in {
            "cross_entropy",
            "weighted_ce",
            "focal",
            "balanced_softmax",
            "balanced_softmax_smoothed",
            "logit_adjustment",
            "ldam",
        }:
            raise ValueError(f"Unsupported fine loss: {loss_name}")
        self.gamma = float(config["focal_gamma"])
        self.tau = float(config["logit_adjustment_tau"])
        self.ldam_scale = float(config["ldam_scale"])
        prior_counts = (
            counts
            if config["fine_prior_source"] == "image_count"
            else patient_count_array
        )
        prior = science.build_smoothed_class_prior(
            prior_counts.astype(np.int64),
            smoothing_alpha=float(config["fine_prior_smoothing_alpha"]),
            power=float(config["fine_prior_power"]),
            max_ratio=float(config["fine_prior_max_ratio"]),
        )
        self.register_buffer(
            "class_counts",
            torch.as_tensor(counts, dtype=torch.float32),
        )
        self.register_buffer(
            "patient_counts",
            torch.as_tensor(patient_count_array, dtype=torch.float32),
        )
        self.register_buffer(
            "active_mask",
            torch.as_tensor(prior["active_mask"], dtype=torch.bool),
        )
        self.register_buffer(
            "prior_probabilities",
            torch.as_tensor(prior["probabilities"], dtype=torch.float32),
        )
        self.register_buffer(
            "smoothed_log_prior",
            torch.as_tensor(prior["log_probabilities"], dtype=torch.float32),
        )
        weights = effective_number_weights(
            counts,
            float(config["class_balance_beta"]),
            require_positive=False,
        )
        self.register_buffer(
            "class_weights",
            torch.as_tensor(weights, dtype=torch.float32),
        )
        canonical_prior = np.zeros_like(counts)
        positive_counts = counts > 0
        canonical_prior[positive_counts] = (
            counts[positive_counts] / counts[positive_counts].sum()
        )
        canonical_log_prior = np.zeros_like(counts)
        canonical_log_prior[positive_counts] = np.log(
            canonical_prior[positive_counts]
        )
        self.register_buffer(
            "canonical_log_prior",
            torch.as_tensor(
                canonical_log_prior,
                dtype=torch.float32,
            ),
        )
        margins = np.zeros_like(counts)
        positive = counts > 0
        margins[positive] = float(config["ldam_max_margin"]) / np.power(
            counts[positive], 0.25
        )
        self.register_buffer(
            "ldam_margins",
            torch.as_tensor(margins, dtype=torch.float32),
        )
        self.focal_use_class_balance = bool(
            config["focal_use_class_balance"]
        )

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[1] != len(FINE_NAMES):
            raise ValueError("Fine logits must have shape [N, 22].")
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Fine logits contain NaN or infinity.")
        inactive_value = torch.finfo(logits.dtype).min
        return logits.masked_fill(~self.active_mask.unsqueeze(0), inactive_value)

    def inference_logits(
        self,
        logits: torch.Tensor,
        prior_tau: float,
    ) -> torch.Tensor:
        if not math.isfinite(float(prior_tau)):
            raise ValueError("Fine inference prior tau must be finite.")
        adjusted = (
            logits
            - float(prior_tau)
            * self.smoothed_log_prior.to(dtype=logits.dtype).unsqueeze(0)
        )
        return self.mask_logits(adjusted)

    def prior_audit(self) -> dict[str, Any]:
        return {
            "loss_name": self.loss_name,
            "image_counts": self.class_counts.detach().cpu().int().tolist(),
            "patient_counts": self.patient_counts.detach().cpu().int().tolist(),
            "active_mask": self.active_mask.detach().cpu().tolist(),
            "prior_probabilities": self.prior_probabilities.detach()
            .cpu()
            .tolist(),
            "smoothed_log_prior": self.smoothed_log_prior.detach()
            .cpu()
            .tolist(),
        }

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        if targets.numel() == 0:
            return logits.sum() * 0.0
        if targets.ndim != 1 or len(targets) != len(logits):
            raise ValueError("Fine targets must align with fine logits.")
        if torch.any(targets < 0) or torch.any(targets >= len(FINE_NAMES)):
            raise ValueError("Fine targets are outside the 22-class taxonomy.")
        if torch.any(~self.active_mask[targets]):
            inactive_targets = torch.unique(targets[~self.active_mask[targets]])
            raise ValueError(
                "Fine targets reference classes absent from training: "
                f"{inactive_targets.detach().cpu().tolist()}"
            )
        logits = self.mask_logits(logits)
        if self.loss_name == "cross_entropy":
            return F.cross_entropy(logits, targets)
        if self.loss_name == "weighted_ce":
            return F.cross_entropy(logits, targets, weight=self.class_weights)
        if self.loss_name == "focal":
            ce = F.cross_entropy(logits, targets, reduction="none")
            probability = torch.exp(-ce)
            loss = ((1.0 - probability) ** self.gamma) * ce
            if self.focal_use_class_balance:
                loss = loss * self.class_weights[targets]
            return loss.mean()
        if self.loss_name == "balanced_softmax":
            adjusted = logits + self.canonical_log_prior.unsqueeze(0)
            return F.cross_entropy(adjusted, targets)
        if self.loss_name == "balanced_softmax_smoothed":
            adjusted = logits + self.smoothed_log_prior.unsqueeze(0)
            return F.cross_entropy(adjusted, targets)
        if self.loss_name == "logit_adjustment":
            adjusted = (
                logits + self.tau * self.canonical_log_prior.unsqueeze(0)
            )
            return F.cross_entropy(adjusted, targets)
        if self.loss_name == "ldam":
            adjusted = logits.clone()
            row_ids = torch.arange(
                len(targets), device=targets.device
            )
            adjusted[row_ids, targets] -= self.ldam_margins[targets]
            return F.cross_entropy(adjusted * self.ldam_scale, targets)
        raise RuntimeError(f"Unreachable fine loss: {self.loss_name}")


def active_fine_loss_name(config: Mapping[str, Any]) -> str:
    if float(config["fine_loss_weight"]) == 0:
        return "cross_entropy"
    return str(config["fine_loss"])


def active_tasks_for_mode(task_mode: str) -> frozenset[str]:
    mapping = {
        "binary": frozenset({"binary"}),
        "coarse": frozenset({"coarse"}),
        "fine": frozenset({"fine"}),
        "multitask": frozenset({"binary", "coarse", "fine"}),
        "hierarchical": frozenset({"binary", "coarse", "fine"}),
    }
    if task_mode not in mapping:
        raise ValueError(f"Unsupported task_mode: {task_mode}")
    return mapping[task_mode]


def active_tasks_from_config(
    config: Mapping[str, Any],
) -> frozenset[str]:
    task_mode = str(config["task_mode"])
    if task_mode in {"binary", "coarse", "fine"}:
        return active_tasks_for_mode(task_mode)
    active = {
        task_name
        for task_name, weight_key in (
            ("binary", "binary_loss_weight"),
            ("coarse", "coarse_loss_weight"),
            ("fine", "fine_loss_weight"),
        )
        if float(config[weight_key]) > 0
    }
    if float(config["binary_coarse_hierarchy_loss_weight"]) > 0:
        active.add("coarse")
    if float(config["coarse_fine_hierarchy_loss_weight"]) > 0:
        active.add("fine")
    if not active:
        raise ValueError("Multi-head configuration activates no task.")
    return frozenset(active)


def supervised_contrastive_loss(
    projections: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if projections.ndim != 2:
        raise ValueError("SupCon projections must have shape [N, D].")
    if labels.ndim != 1 or len(labels) != len(projections):
        raise ValueError("SupCon labels must align with projections.")
    if temperature <= 0:
        raise ValueError("SupCon temperature must be positive.")
    if len(projections) < 2:
        return projections.sum() * 0.0
    features = F.normalize(projections, dim=1)
    logits = features @ features.T / temperature
    diagonal = torch.eye(
        len(features), dtype=torch.bool, device=features.device
    )
    positive_mask = labels[:, None].eq(labels[None, :]) & ~diagonal
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * (~diagonal)
    log_prob = logits - torch.log(
        exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
    )
    positives_per_anchor = positive_mask.sum(dim=1)
    valid = positives_per_anchor > 0
    if not valid.any():
        return projections.sum() * 0.0
    mean_positive_log_prob = (
        (positive_mask * log_prob).sum(dim=1)
        / positives_per_anchor.clamp_min(1)
    )
    return -mean_positive_log_prob[valid].mean()


def negative_log_correct_parent_mass(
    parent_probabilities: torch.Tensor,
    parent_targets: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    if parent_probabilities.ndim != 2:
        raise ValueError("parent_probabilities must have shape [N, C].")
    if parent_targets.ndim != 1:
        raise ValueError("parent_targets must have shape [N].")
    if len(parent_probabilities) != len(parent_targets):
        raise ValueError(
            "Parent probabilities and targets must have equal batch size."
        )
    if len(parent_targets) == 0:
        return parent_probabilities.sum() * 0.0
    if not torch.isfinite(parent_probabilities).all():
        raise FloatingPointError(
            "Parent probabilities contain NaN or infinity."
        )
    if torch.any(parent_targets < 0):
        raise ValueError("Parent targets cannot contain negative IDs.")
    if torch.any(parent_targets >= parent_probabilities.shape[1]):
        raise ValueError("Parent target is outside probability dimensions.")

    correct_parent_mass = parent_probabilities.gather(
        dim=1,
        index=parent_targets.unsqueeze(1),
    ).squeeze(1)

    if not torch.isfinite(correct_parent_mass).all():
        raise FloatingPointError(
            "Correct parent probability mass is not finite."
        )

    return -torch.log(correct_parent_mass.clamp_min(eps)).mean()


def binary_coarse_hierarchy_loss(
    coarse_logits: torch.Tensor,
    binary_targets: torch.Tensor,
) -> torch.Tensor:
    if coarse_logits.ndim != 2:
        raise ValueError("coarse_logits must have shape [N, 5].")
    if coarse_logits.shape[1] != len(COARSE_NAMES):
        raise ValueError(f"Expected {len(COARSE_NAMES)} coarse logits.")
    if binary_targets.ndim != 1:
        raise ValueError("binary_targets must have shape [N].")
    if len(coarse_logits) != len(binary_targets):
        raise ValueError("Coarse logits and binary targets must align.")

    coarse_probs = coarse_logits.softmax(dim=1)
    mapping = COARSE_TO_BINARY_MATRIX.to(
        device=coarse_probs.device,
        dtype=coarse_probs.dtype,
    )
    binary_probs_from_coarse = coarse_probs @ mapping
    return negative_log_correct_parent_mass(
        binary_probs_from_coarse,
        binary_targets,
    )


def coarse_fine_hierarchy_loss(
    fine_logits: torch.Tensor,
    coarse_targets: torch.Tensor,
    fine_targets: torch.Tensor,
    fine_loss_fn: FineLongTailLoss,
) -> torch.Tensor:
    if fine_logits.ndim != 2:
        raise ValueError("fine_logits must have shape [N, 22].")
    if fine_logits.shape[1] != len(FINE_NAMES):
        raise ValueError(f"Expected {len(FINE_NAMES)} fine logits.")
    if coarse_targets.ndim != 1:
        raise ValueError("coarse_targets must have shape [N].")
    if fine_targets.ndim != 1:
        raise ValueError("fine_targets must have shape [N].")
    if not (len(fine_logits) == len(coarse_targets) == len(fine_targets)):
        raise ValueError(
            "Fine logits, coarse targets and fine targets must align."
        )

    valid = fine_targets >= 0
    if not valid.any():
        return fine_logits.sum() * 0.0

    valid_coarse_targets = coarse_targets[valid]
    normal_id = COARSE_TO_ID["Normal mucosa"]
    if torch.any(valid_coarse_targets == normal_id):
        raise ValueError(
            "A sample with a valid fine target cannot have "
            "Normal mucosa as its coarse target."
        )

    masked_fine_logits = fine_loss_fn.mask_logits(fine_logits[valid])
    fine_probs = masked_fine_logits.softmax(dim=1)
    mapping = FINE_TO_COARSE_MATRIX.to(
        device=fine_probs.device,
        dtype=fine_probs.dtype,
    )
    coarse_probs_from_fine = fine_probs @ mapping
    return negative_log_correct_parent_mass(
        coarse_probs_from_fine,
        valid_coarse_targets,
    )


def compute_multitask_loss(
    outputs: Mapping[str, torch.Tensor],
    binary_targets: torch.Tensor,
    coarse_targets: torch.Tensor,
    fine_targets: torch.Tensor,
    fine_loss_fn: FineLongTailLoss | None,
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    active_tasks = active_tasks_from_config(config)
    expected_output_keys = {"features"} | {
        f"{task}_logits" for task in active_tasks
    }
    optional_keys = {"projection"}
    unknown_outputs = set(outputs) - expected_output_keys - optional_keys
    missing_outputs = expected_output_keys - set(outputs)
    if missing_outputs or unknown_outputs:
        raise ValueError(
            "Model output/task contract mismatch: "
            f"missing={sorted(missing_outputs)}, "
            f"unexpected={sorted(unknown_outputs)}"
        )
    anchor = outputs[next(f"{task}_logits" for task in sorted(active_tasks))]
    total = anchor.sum() * 0.0
    components: dict[str, torch.Tensor] = {}
    if "binary" in active_tasks:
        binary_loss = F.cross_entropy(
            outputs["binary_logits"], binary_targets
        )
        components["binary_loss"] = binary_loss
        total = total + float(config["binary_loss_weight"]) * binary_loss
    if "coarse" in active_tasks:
        coarse_loss = F.cross_entropy(
            outputs["coarse_logits"], coarse_targets
        )
        components["coarse_loss"] = coarse_loss
        total = total + float(config["coarse_loss_weight"]) * coarse_loss
    if "fine" in active_tasks:
        if fine_loss_fn is None:
            raise ValueError("An active fine task requires FineLongTailLoss.")
        valid_fine = fine_targets >= 0
        if valid_fine.any():
            valid_positions = torch.nonzero(
                valid_fine,
                as_tuple=False,
            ).flatten()
            valid_ids = fine_targets[valid_fine]
            active_positions = fine_loss_fn.active_mask[valid_ids]
            valid_fine[valid_positions] = active_positions
        fine_loss = fine_loss_fn(
            outputs["fine_logits"][valid_fine],
            fine_targets[valid_fine],
        )
        components["fine_loss"] = fine_loss
        components["fine_loss_evaluable_share"] = (
            valid_fine.float().mean()
        )
        total = total + float(config["fine_loss_weight"]) * fine_loss
    bc_weight = float(config["binary_coarse_hierarchy_loss_weight"])
    if bc_weight > 0:
        if str(config["task_mode"]) != "hierarchical":
            raise ValueError(
                "Binary-Coarse hierarchy loss requires task_mode='hierarchical'."
            )
        if "coarse_logits" not in outputs:
            raise ValueError(
                "Binary-Coarse hierarchy loss requires coarse_logits."
            )
        bc_hierarchy_loss = binary_coarse_hierarchy_loss(
            outputs["coarse_logits"],
            binary_targets,
        )
        components["binary_coarse_hierarchy_loss"] = bc_hierarchy_loss
        total = total + bc_weight * bc_hierarchy_loss

    cf_weight = float(config["coarse_fine_hierarchy_loss_weight"])
    if cf_weight > 0:
        if str(config["task_mode"]) != "hierarchical":
            raise ValueError(
                "Coarse-Fine hierarchy loss requires task_mode='hierarchical'."
            )
        if "fine_logits" not in outputs:
            raise ValueError(
                "Coarse-Fine hierarchy loss requires fine_logits."
            )
        if fine_loss_fn is None:
            raise ValueError(
                "Coarse-Fine hierarchy loss requires FineLongTailLoss."
            )
        cf_hierarchy_loss = coarse_fine_hierarchy_loss(
            outputs["fine_logits"],
            coarse_targets,
            fine_targets,
            fine_loss_fn,
        )
        components["coarse_fine_hierarchy_loss"] = cf_hierarchy_loss
        total = total + cf_weight * cf_hierarchy_loss
    if not torch.isfinite(total):
        raise FloatingPointError("Classification loss is not finite.")
    components["classification_total"] = total
    return total, components


# %%
# CELL 6 - Metrics, training loop, prediction export, and statistics.
def per_class_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    names: Sequence[str],
) -> tuple[list[dict[str, Any]], float | None, int]:
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=np.arange(len(names)),
        zero_division=0,
    )
    rows: list[dict[str, Any]] = []
    auc_values: list[float] = []
    for class_id, name in enumerate(names):
        binary_target = targets == class_id
        class_supported = int(support[class_id]) > 0
        auc: float | None = None
        if len(np.unique(binary_target)) == 2:
            auc = float(
                roc_auc_score(binary_target, probabilities[:, class_id])
            )
            auc_values.append(auc)
        rows.append(
            {
                "class_id": class_id,
                "class_name": name,
                "precision": (
                    float(precision[class_id]) if class_supported else None
                ),
                "recall": (
                    float(recall[class_id]) if class_supported else None
                ),
                "f1": float(f1[class_id]) if class_supported else None,
                "support": int(support[class_id]),
                "classification_metrics_evaluable": class_supported,
                "auroc_ovr": auc,
                "auroc_evaluable": auc is not None,
            }
        )
    macro_auc = float(np.mean(auc_values)) if auc_values else None
    return rows, macro_auc, len(auc_values)


def compute_binary_metrics(
    targets: np.ndarray,
    positive_probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    targets = np.asarray(targets, dtype=np.int64)
    positive_probabilities = np.asarray(
        positive_probabilities, dtype=np.float64
    )
    if (
        targets.ndim != 1
        or positive_probabilities.ndim != 1
        or len(targets) == 0
        or len(targets) != len(positive_probabilities)
    ):
        raise ValueError(
            "Binary targets and probabilities must be aligned non-empty vectors."
        )
    if (
        not np.isfinite(positive_probabilities).all()
        or np.any(positive_probabilities < 0)
        or np.any(positive_probabilities > 1)
        or np.any((targets != 0) & (targets != 1))
    ):
        raise ValueError("Binary targets/probabilities are outside their domains.")
    if not 0 < threshold < 1:
        raise ValueError("Binary threshold must be in (0, 1).")
    predictions = (positive_probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    supported_recalls = [
        value for value in (specificity, sensitivity) if value is not None
    ]
    auroc = (
        float(roc_auc_score(targets, positive_probabilities))
        if len(np.unique(targets)) == 2
        else None
    )
    auprc = (
        float(average_precision_score(targets, positive_probabilities))
        if len(np.unique(targets)) == 2
        else None
    )
    return {
        "n": len(targets),
        "decision_threshold": float(threshold),
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(
            precision_score(targets, predictions, zero_division=0)
        ),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "sensitivity": float(sensitivity) if sensitivity is not None else None,
        "specificity": float(specificity) if specificity is not None else None,
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(targets, predictions)),
        "balanced_accuracy": float(np.mean(supported_recalls)),
        "auroc": auroc,
        "auprc": auprc,
        "auroc_evaluable": auroc is not None,
        "auprc_evaluable": auprc is not None,
        "confusion_matrix": matrix.tolist(),
    }


def compute_multiclass_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    names: Sequence[str],
    probability_sum_atol: float = 1.0e-6,
) -> dict[str, Any]:
    base = science.compute_multiclass_metrics(
        targets,
        probabilities,
        class_names=names,
        probability_sum_atol=probability_sum_atol,
    )
    target_array, probability_array = science.validate_multiclass_inputs(
        targets,
        probabilities,
        probability_sum_atol=probability_sum_atol,
    )
    predicted = probability_array.argmax(axis=1)
    auc_rows, macro_auc, auc_count = per_class_metrics(
        target_array,
        predicted,
        probability_array,
        names,
    )
    for row, auc_row in zip(base["per_class"], auc_rows, strict=True):
        row["support"] = row["true_count"]
        row["classification_metrics_evaluable"] = row["target_supported"]
        row["auroc_ovr"] = auc_row["auroc_ovr"]
        row["auroc_evaluable"] = auc_row["auroc_evaluable"]
    supported_recalls = [
        row["recall"] for row in base["per_class"] if row["target_supported"]
    ]
    base.update(
        {
            # Explicit names are authoritative; aliases keep reports readable.
            "macro_f1": base["macro_f1_supported"],
            "macro_f1_evaluable_classes": base[
                "macro_f1_supported_class_count"
            ],
            "balanced_accuracy": float(np.mean(supported_recalls)),
            "mcc": float(matthews_corrcoef(target_array, predicted)),
            "macro_auroc_ovr": macro_auc,
            "macro_auroc_evaluable_classes": auc_count,
        }
    )
    return base


def compute_metrics_bundle(
    predictions: pd.DataFrame,
    train_fine_counts: Sequence[int],
    train_fine_patient_counts: Sequence[int],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if predictions.empty:
        raise ValueError("Cannot compute metrics from no predictions.")
    active_tasks = active_tasks_from_config(config)
    binary_targets = predictions["binary_id"].to_numpy(dtype=np.int64)
    coarse_targets = predictions["coarse_id"].to_numpy(dtype=np.int64)
    fine_targets = predictions["fine_id"].to_numpy(dtype=np.int64)
    tolerance = float(config["probability_sum_tolerance"])
    binary: dict[str, Any] | None = None
    coarse: dict[str, Any] | None = None
    fine: dict[str, Any] | None = None
    hierarchy: dict[str, Any] | None = None
    primary_fine: dict[str, Any] | None = None
    rare_audit: dict[str, Any] | None = None
    if "binary" in active_tasks:
        binary_probs = np.stack(predictions["binary_probs"].to_numpy())
        science.validate_multiclass_inputs(
            binary_targets,
            binary_probs,
            probability_sum_atol=tolerance,
        )
        binary = compute_binary_metrics(
            binary_targets,
            binary_probs[:, 1],
            float(config["binary_decision_threshold"]),
        )
    if "coarse" in active_tasks:
        coarse_probs = np.stack(predictions["coarse_probs"].to_numpy())
        coarse = compute_multiclass_metrics(
            coarse_targets,
            coarse_probs,
            COARSE_NAMES,
            tolerance,
        )
    valid_fine = fine_targets >= 0
    if "fine" in active_tasks and valid_fine.any():
        fine_probs = np.stack(predictions["fine_probs"].to_numpy())
        fine = compute_multiclass_metrics(
            fine_targets[valid_fine],
            fine_probs[valid_fine],
            FINE_NAMES,
            tolerance,
        )
        fine_pred = fine_probs[valid_fine].argmax(axis=1)
        train_counts = np.asarray(train_fine_counts, dtype=np.int64)
        patient_counts = np.asarray(
            train_fine_patient_counts, dtype=np.int64
        )
        if (
            train_counts.shape != (len(FINE_NAMES),)
            or patient_counts.shape != (len(FINE_NAMES),)
        ):
            raise ValueError(
                "Fine image/patient train counts must have shape (22,)."
            )
        tail_ids = np.flatnonzero(
            (train_counts > 0)
            & (
                train_counts
                <= int(config["tail_class_max_train_samples"])
            )
        )
        tail_mask = np.isin(fine_targets[valid_fine], tail_ids)
        evaluable_tail_ids = np.intersect1d(
            tail_ids, np.unique(fine_targets[valid_fine][tail_mask])
        )
        tail_recall = (
            float(
                recall_score(
                    fine_targets[valid_fine][tail_mask],
                    fine_pred[tail_mask],
                    labels=evaluable_tail_ids,
                    average="macro",
                    zero_division=0,
                )
            )
            if tail_mask.any() and len(evaluable_tail_ids)
            else None
        )
        if "coarse" in active_tasks:
            coarse_probs = np.stack(predictions["coarse_probs"].to_numpy())
            coarse_pred = coarse_probs[valid_fine].argmax(axis=1)
            predicted_parent = np.asarray(
                [FINE_PARENT_ID[index] for index in fine_pred],
                dtype=np.int64,
            )
            true_coarse = coarse_targets[valid_fine]
            hierarchy = {
                "parent_accuracy_from_coarse_head": float(
                    accuracy_score(true_coarse, coarse_pred)
                ),
                "parent_accuracy_from_fine_head": float(
                    accuracy_score(true_coarse, predicted_parent)
                ),
                "child_accuracy": float(
                    accuracy_score(fine_targets[valid_fine], fine_pred)
                ),
                "hierarchical_accuracy": float(
                    np.mean(
                        (coarse_pred == true_coarse)
                        & (fine_pred == fine_targets[valid_fine])
                    )
                ),
                "cross_parent_error_rate": float(
                    np.mean(predicted_parent != true_coarse)
                ),
                "coarse_fine_prediction_consistency": float(
                    np.mean(predicted_parent == coarse_pred)
                ),
                "tail_class_recall": tail_recall,
                "tail_class_names": [
                    FINE_NAMES[index] for index in tail_ids.tolist()
                ],
                "tail_class_names_evaluable_on_this_split": [
                    FINE_NAMES[index] for index in evaluable_tail_ids.tolist()
                ],
            }
        fixed_ids = config["fixed_primary_fine_class_ids"]
        if fixed_ids is None:
            primary_ids = np.flatnonzero(
                patient_counts
                >= int(config["primary_fine_min_train_patients"])
            )
            selection_source = "derived_from_training_patient_support"
        else:
            primary_ids = np.asarray(fixed_ids, dtype=np.int64)
            selection_source = "frozen_protocol"
        primary_mask = np.isin(fine_targets[valid_fine], primary_ids)
        if len(primary_ids) and primary_mask.any():
            primary_fine = science.compute_fixed_primary_metrics(
                fine_targets[valid_fine],
                fine_probs[valid_fine],
                primary_ids,
                class_names=FINE_NAMES,
                probability_sum_atol=tolerance,
            )
            primary_fine.update(
                {
                    "status": "ok",
                    "selection_source": selection_source,
                    "min_train_patient_support": int(
                        config["primary_fine_min_train_patients"]
                    ),
                    "macro_f1": primary_fine[
                        "macro_f1_supported"
                    ],
                }
            )
        else:
            primary_fine = {
                "status": "not_evaluable",
                "reason": (
                    "No evaluated row belongs to the frozen primary fine "
                    "class set."
                ),
                "selection_source": selection_source,
                "min_train_patient_support": int(
                    config["primary_fine_min_train_patients"]
                ),
                "primary_class_ids": primary_ids.tolist(),
                "primary_class_names": [
                    FINE_NAMES[index] for index in primary_ids.tolist()
                ],
            }
        rare_audit = science.audit_rare_class_collapse(
            fine_pred,
            train_counts,
            patient_counts,
            class_names=FINE_NAMES,
            max_train_patients=int(config["rare_gate_max_train_patients"]),
            absolute_prediction_share=float(
                config["rare_gate_absolute_pred_share"]
            ),
            prior_multiplier=float(config["rare_gate_prior_multiplier"]),
            min_predicted_count=int(config["rare_gate_min_pred_count"]),
            prior_smoothing_alpha=float(
                config["fine_prior_smoothing_alpha"]
            ),
            prior_power=float(config["fine_prior_power"]),
        )
    return {
        "binary": binary,
        "coarse": coarse,
        "fine": fine,
        "primary_fine": primary_fine,
        "hierarchy": hierarchy,
        "rare_class_collapse": rare_audit,
    }


def metric_for_monitor(
    bundle: Mapping[str, Any],
    monitor: str,
    monitor_weights: Mapping[str, float] | None = None,
) -> float:
    mapping = {
        "binary_auroc": (
            bundle["binary"]["auroc"] if bundle["binary"] is not None else None
        ),
        "binary_f1": (
            bundle["binary"]["f1"] if bundle["binary"] is not None else None
        ),
        "coarse_macro_f1": (
            bundle["coarse"]["macro_f1_supported"]
            if bundle["coarse"] is not None
            else None
        ),
        "coarse_macro_f1_all_classes": (
            bundle["coarse"]["macro_f1_all_classes"]
            if bundle["coarse"] is not None
            else None
        ),
        "coarse_balanced_accuracy": (
            bundle["coarse"]["balanced_accuracy"]
            if bundle["coarse"] is not None
            else None
        ),
        "fine_macro_f1": (
            bundle["fine"]["macro_f1_supported"]
            if bundle["fine"] is not None
            else None
        ),
        "fine_macro_f1_all_classes": (
            bundle["fine"]["macro_f1_all_classes"]
            if bundle["fine"] is not None
            else None
        ),
        "primary_macro_f1_all_classes": (
            bundle["primary_fine"]["macro_f1_all_classes"]
            if bundle["primary_fine"] is not None
            and bundle["primary_fine"]["status"] == "ok"
            else None
        ),
        "hierarchical_accuracy": (
            bundle["hierarchy"]["hierarchical_accuracy"]
            if bundle["hierarchy"] is not None
            else None
        ),
    }
    if monitor == "hierarchical_composite":
        required = {
            "coarse_macro_f1_all_classes": mapping[
                "coarse_macro_f1_all_classes"
            ],
            "primary_macro_f1_all_classes": mapping[
                "primary_macro_f1_all_classes"
            ],
            "hierarchical_accuracy": mapping["hierarchical_accuracy"],
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "hierarchical_composite is not evaluable; missing "
                f"components={missing}."
            )
        return science.hierarchical_composite_monitor(
            {name: float(value) for name, value in required.items()},
            weights=monitor_weights,
        )
    if monitor not in mapping:
        raise ValueError(
            f"Unknown monitor_metric={monitor}; choices={sorted(mapping)}"
        )
    value = mapping[monitor]
    if value is None or not math.isfinite(float(value)):
        raise ValueError(
            f"monitor_metric={monitor} is not evaluable on validation data."
        )
    return float(value)


def forward_with_precision(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> Mapping[str, torch.Tensor]:
    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=amp_dtype is not None,
    ):
        return model(images)


def move_images(
    images: torch.Tensor,
    device: torch.device,
    config: Mapping[str, Any],
) -> torch.Tensor:
    images = images.to(
        device,
        non_blocking=bool(config.get("pin_memory", True) and device.type == "cuda"),
    )
    if config.get("channels_last", False) and device.type == "cuda":
        images = images.contiguous(memory_format=torch.channels_last)
    return images


def move_target(
    target: torch.Tensor,
    device: torch.device,
    config: Mapping[str, Any],
) -> torch.Tensor:
    return target.to(
        device,
        non_blocking=bool(config.get("pin_memory", True) and device.type == "cuda"),
    )


def build_optimizer(
    model: HierarchicalCystoModel,
    config: Mapping[str, Any],
    device: torch.device,
) -> torch.optim.Optimizer:
    if config["optimizer"] != "adamw":
        raise ValueError("Only optimizer='adamw' is implemented.")
    encoder_params = list(model.encoder.parameters())
    head_params = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.")
    ]
    fused = bool(config["use_fused_optimizer"])
    if fused and device.type != "cuda":
        raise RuntimeError("Fused AdamW is only enabled for CUDA runs.")
    return torch.optim.AdamW(
        [
            {
                "params": encoder_params,
                "lr": float(config["learning_rate"])
                * float(config["encoder_learning_rate_multiplier"]),
            },
            {
                "params": head_params,
                "lr": float(config["learning_rate"]),
            },
        ],
        weight_decay=float(config["weight_decay"]),
        fused=fused,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_batches: int,
    config: Mapping[str, Any],
) -> tuple[torch.optim.lr_scheduler.LambdaLR, int]:
    accumulation = int(config["gradient_accumulation_steps"])
    updates_per_epoch = math.ceil(train_batches / accumulation)
    total_updates = updates_per_epoch * int(config["scheduler_epochs"])
    warmup_updates = round(
        float(config["warmup_epochs"]) * updates_per_epoch
    )
    min_ratio = float(config["minimum_learning_rate_ratio"])
    if total_updates < 1:
        raise ValueError("Scheduler requires at least one optimizer update.")
    if not 0 <= min_ratio <= 1:
        raise ValueError("minimum_learning_rate_ratio must be in [0, 1].")

    def multiplier(step: int) -> float:
        if warmup_updates > 0 and step < warmup_updates:
            return max((step + 1) / warmup_updates, 1e-8)
        denominator = max(total_updates - warmup_updates, 1)
        progress = min(max((step - warmup_updates) / denominator, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier), total_updates


def prediction_rows_from_outputs(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    include_features: bool,
    fine_loss_fn: FineLongTailLoss | None,
    fine_prior_tau: float,
) -> list[dict[str, Any]]:
    logit_arrays: dict[str, np.ndarray] = {}
    probability_arrays: dict[str, np.ndarray] = {}
    for task_name in ("binary", "coarse"):
        key = f"{task_name}_logits"
        if key in outputs:
            logits = outputs[key]
            logit_arrays[task_name] = (
                logits.detach().float().cpu().numpy()
            )
            probability_arrays[task_name] = (
                logits.softmax(dim=1).detach().float().cpu().numpy()
            )
    if "fine_logits" in outputs:
        if fine_loss_fn is None:
            raise ValueError("Fine output requires a configured fine loss.")
        raw_fine_logits = outputs["fine_logits"]
        calibrated_fine_logits = fine_loss_fn.inference_logits(
            raw_fine_logits,
            fine_prior_tau,
        )
        logit_arrays["fine"] = (
            raw_fine_logits.detach().float().cpu().numpy()
        )
        probability_arrays["fine"] = (
            calibrated_fine_logits.softmax(dim=1)
            .detach()
            .float()
            .cpu()
            .numpy()
        )
    features = (
        outputs["features"].detach().float().cpu().numpy()
        if include_features and "features" in outputs
        else None
    )
    rows: list[dict[str, Any]] = []
    batch_size = len(batch["filename"])
    for index in range(batch_size):
        row = {
            "filename": batch["filename"][index],
            "pid": batch["pid"][index],
            "visit": batch.get("visit", ["NA"] * batch_size)[index],
            "lesion": batch.get("lesion", ["NA"] * batch_size)[index],
            "modality": batch.get(
                "modality", ["unknown"] * batch_size
            )[index],
            "binary_id": int(batch["binary_id"][index]),
            "coarse_id": int(batch["coarse_id"][index]),
            "fine_id": int(batch["fine_id"][index]),
            "fine_inference_prior_tau": float(fine_prior_tau),
        }
        for task_name in sorted(logit_arrays):
            row[f"{task_name}_logits"] = logit_arrays[task_name][index]
            row[f"{task_name}_probs"] = probability_arrays[task_name][index]
        if features is not None:
            row["features"] = features[index]
        rows.append(row)
    return rows


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    fine_loss_fn: FineLongTailLoss | None,
    train_fine_counts: Sequence[int],
    train_fine_patient_counts: Sequence[int],
    config: Mapping[str, Any],
    device: torch.device,
    amp_dtype: torch.dtype | None,
    include_features: bool,
    fine_prior_tau: float,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, float]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    loss_sums: defaultdict[str, float] = defaultdict(float)
    sample_count = 0
    with torch.inference_mode():
        for batch in loader:
            images = move_images(batch["image"], device, config)
            binary_targets = move_target(
                batch["binary_id"], device, config
            )
            coarse_targets = move_target(
                batch["coarse_id"], device, config
            )
            fine_targets = move_target(batch["fine_id"], device, config)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                outputs = model(images)
                total, components = compute_multitask_loss(
                    outputs,
                    binary_targets,
                    coarse_targets,
                    fine_targets,
                    fine_loss_fn,
                    config,
                )
            if not torch.isfinite(total):
                raise FloatingPointError(
                    "Non-finite evaluation loss detected."
                )
            for output_name in (
                name for name in outputs if name.endswith("_logits")
            ):
                if not torch.isfinite(outputs[output_name]).all():
                    raise FloatingPointError(
                        f"Non-finite evaluation tensor: {output_name}"
                    )
            batch_size = len(images)
            sample_count += batch_size
            loss_sums["total_loss"] += float(total.detach()) * batch_size
            for name, value in components.items():
                loss_sums[name] += float(value.detach()) * batch_size
            rows.extend(
                prediction_rows_from_outputs(
                    outputs,
                    batch,
                    include_features=include_features,
                    fine_loss_fn=fine_loss_fn,
                    fine_prior_tau=fine_prior_tau,
                )
            )
    predictions = pd.DataFrame(rows)
    metrics = compute_metrics_bundle(
        predictions,
        train_fine_counts,
        train_fine_patient_counts,
        config,
    )
    losses = {
        name: value / sample_count for name, value in loss_sums.items()
    }
    return metrics, predictions, losses


def apply_fine_inference_calibration(
    predictions: pd.DataFrame,
    fine_loss_fn: FineLongTailLoss,
    prior_tau: float,
) -> pd.DataFrame:
    if "fine_logits" not in predictions:
        raise ValueError("Fine calibration requires fine_logits predictions.")
    logits = np.stack(predictions["fine_logits"].to_numpy()).astype(
        np.float64,
        copy=False,
    )
    if not np.isfinite(logits).all():
        raise FloatingPointError("Stored fine logits contain NaN or infinity.")
    active_mask = (
        fine_loss_fn.active_mask.detach().cpu().numpy().astype(bool)
    )
    log_prior = (
        fine_loss_fn.smoothed_log_prior.detach().cpu().numpy().astype(
            np.float64
        )
    )
    adjusted = logits - float(prior_tau) * log_prior[None, :]
    adjusted[:, ~active_mask] = -1.0e30
    adjusted -= adjusted.max(axis=1, keepdims=True)
    exponentials = np.exp(adjusted)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    if (
        not np.isfinite(probabilities).all()
        or not np.allclose(
            probabilities.sum(axis=1),
            1.0,
            rtol=0,
            atol=1.0e-12,
        )
    ):
        raise FloatingPointError(
            "Fine inference calibration produced invalid probabilities."
        )
    calibrated = predictions.copy()
    calibrated["fine_probs"] = list(probabilities)
    calibrated["fine_inference_prior_tau"] = float(prior_tau)
    return calibrated


def select_fine_inference_tau(
    predictions: pd.DataFrame,
    fine_loss_fn: FineLongTailLoss,
    train_fine_counts: Sequence[int],
    train_fine_patient_counts: Sequence[int],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, float, dict[str, Any]]:
    if config["fine_inference_calibration_mode"] == "fixed":
        tau_values = [float(config["fine_inference_prior_tau"])]
    else:
        tau_values = sorted(
            {float(value) for value in config["fine_inference_tau_grid"]}
        )
    metric_name = str(config["fine_inference_calibration_metric"])
    candidates = []
    best: tuple[float, float, dict[str, Any], pd.DataFrame] | None = None
    for tau in tau_values:
        calibrated = apply_fine_inference_calibration(
            predictions,
            fine_loss_fn,
            tau,
        )
        metrics = compute_metrics_bundle(
            calibrated,
            train_fine_counts,
            train_fine_patient_counts,
            config,
        )
        if metric_name == "fine_macro_f1_all_classes":
            value = (
                metrics["fine"]["macro_f1_all_classes"]
                if metrics["fine"] is not None
                else None
            )
        elif metric_name == "primary_macro_f1_all_classes":
            value = (
                metrics["primary_fine"]["macro_f1_all_classes"]
                if metrics["primary_fine"] is not None
                and metrics["primary_fine"]["status"] == "ok"
                else None
            )
        else:
            raise ValueError(
                f"Unsupported fine calibration metric: {metric_name}"
            )
        if value is None or not math.isfinite(float(value)):
            raise ValueError(
                f"Fine calibration metric {metric_name} is not evaluable "
                f"for tau={tau}."
            )
        score = float(value)
        candidates.append({"prior_tau": tau, "metric": score})
        ranking = (score, -tau)
        if best is None or ranking > (best[0], best[1]):
            best = (score, -tau, metrics, calibrated)
    if best is None:
        raise RuntimeError("Fine inference calibration produced no candidate.")
    selected_tau = -best[1]
    audit = {
        "mode": str(config["fine_inference_calibration_mode"]),
        "metric": metric_name,
        "candidates": candidates,
        "selected_prior_tau": selected_tau,
        "tie_break": "smallest_prior_tau",
    }
    best[2]["fine_inference_calibration"] = audit
    return best[2], best[3], selected_tau, audit


def split_fingerprint(
    split_frames: Mapping[str, pd.DataFrame],
) -> str:
    digest = hashlib.sha256()
    for split_name in ("train", "val", "test"):
        frame = split_frames[split_name].sort_values(
            ["pid", "image_stem"]
        )
        columns = [
            "pid",
            "image_stem",
            "binary_id",
            "coarse_id",
            "fine_id",
        ]
        for row in frame[columns].astype(str).itertuples(
            index=False, name=None
        ):
            digest.update(
                (
                    split_name
                    + "|"
                    + "|".join(row)
                    + "\n"
                ).encode("utf-8")
            )
    return digest.hexdigest()


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_metric: float,
    config: Mapping[str, Any],
    fine_counts: Sequence[int],
    fine_patient_counts: Sequence[int],
    fine_loss_fn: FineLongTailLoss | None,
    fine_inference_prior_tau: float,
    data_split_fingerprint: str,
    include_optimizer_state: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "config": json_ready(config),
        "binary_names": BINARY_NAMES,
        "coarse_names": COARSE_NAMES,
        "fine_names": FINE_NAMES,
        "fine_parent_id": FINE_PARENT_ID,
        "fine_train_counts": list(map(int, fine_counts)),
        "fine_train_patient_counts": list(map(int, fine_patient_counts)),
        "fine_prior_audit": (
            fine_loss_fn.prior_audit()
            if fine_loss_fn is not None
            else None
        ),
        "fine_inference_prior_tau": float(fine_inference_prior_tau),
        "data_split_fingerprint": data_split_fingerprint,
    }
    if include_optimizer_state:
        payload.update(
            {
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            }
        )
    torch.save(payload, path)


def load_checkpoint_for_resume(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: Mapping[str, Any],
    expected_split_fingerprint: str,
) -> tuple[int, float, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required = {
        "epoch",
        "best_metric",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "config",
        "binary_names",
        "coarse_names",
        "fine_names",
        "fine_inference_prior_tau",
        "data_split_fingerprint",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Resume checkpoint missing keys: {sorted(missing)}")
    if tuple(checkpoint["binary_names"]) != BINARY_NAMES:
        raise ValueError("Resume checkpoint binary taxonomy mismatch.")
    if tuple(checkpoint["coarse_names"]) != COARSE_NAMES:
        raise ValueError("Resume checkpoint coarse taxonomy mismatch.")
    if tuple(checkpoint["fine_names"]) != FINE_NAMES:
        raise ValueError("Resume checkpoint fine taxonomy mismatch.")
    if checkpoint["data_split_fingerprint"] != expected_split_fingerprint:
        raise ValueError("Resume checkpoint data split fingerprint mismatch.")
    saved_config = checkpoint["config"]
    for key in ("model_name", "image_size", "fine_loss"):
        if saved_config.get(key) != json_ready(config[key]):
            raise ValueError(
                f"Resume checkpoint config mismatch for '{key}': "
                f"saved={saved_config.get(key)!r}, current={config[key]!r}"
            )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint["best_metric"]),
        float(checkpoint["fine_inference_prior_tau"]),
    )


def train_model(
    model: HierarchicalCystoModel,
    loaders: Mapping[str, DataLoader],
    split_frames: Mapping[str, pd.DataFrame],
    optimization_train_frame: pd.DataFrame,
    config: Mapping[str, Any],
    device: torch.device,
    amp_dtype: torch.dtype | None,
    fold_dir: Path,
    logger: logging.Logger,
    checkpoint_work_dir: Path | None = None,
) -> tuple[
    HierarchicalCystoModel,
    pd.DataFrame,
    list[int],
    list[int],
]:
    checkpoint_work_dir = (
        fold_dir if checkpoint_work_dir is None else checkpoint_work_dir
    )
    checkpoint_work_dir.mkdir(parents=True, exist_ok=True)
    fine_counts = np.bincount(
        optimization_train_frame
        .loc[optimization_train_frame["fine_id"] >= 0, "fine_id"]
        .astype(int)
        .to_numpy(),
        minlength=len(FINE_NAMES),
    ).tolist()
    fine_patient_counts_series = (
        optimization_train_frame
        .loc[optimization_train_frame["fine_id"] >= 0]
        .groupby("fine_id")["pid"]
        .nunique()
    )
    fine_patient_counts = [
        int(fine_patient_counts_series.get(index, 0))
        for index in range(len(FINE_NAMES))
    ]
    data_split_hash = split_fingerprint(split_frames)
    active_tasks = active_tasks_from_config(config)
    fine_loss_fn = (
        FineLongTailLoss(
            active_fine_loss_name(config),
            fine_counts,
            fine_patient_counts,
            config,
        ).to(device)
        if "fine" in active_tasks
        else None
    )
    if fine_loss_fn is not None:
        write_json(fold_dir / "fine_prior_audit.json", fine_loss_fn.prior_audit())
    optimizer = build_optimizer(model, config, device)
    scheduler, total_updates = build_scheduler(
        optimizer, len(loaders["train"]), config
    )
    use_scaler = amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)

    start_epoch = 0
    best_metric = -float("inf")
    selected_fine_tau = float(config["fine_inference_prior_tau"])
    resume = config["resume_checkpoint"]
    if resume is not None:
        start_epoch, best_metric, selected_fine_tau = load_checkpoint_for_resume(
            Path(resume),
            model,
            optimizer,
            scheduler,
            scaler,
            device,
            config,
            data_split_hash,
        )
        logger.info(
            "Resumed from %s at epoch=%d best=%.6f",
            resume,
            start_epoch,
            best_metric,
        )
        if start_epoch >= int(config["epochs"]):
            raise ValueError(
                "resume_checkpoint already reached the configured epoch "
                "count; increase epochs to continue training."
            )

    accumulation = int(config["gradient_accumulation_steps"])
    history: list[dict[str, Any]] = []
    epochs_without_improvement = 0
    last_completed_epoch = start_epoch - 1
    optimizer.zero_grad(set_to_none=True)
    global_update = 0
    logger.info(
        "Training starts: epochs=%d batches/epoch=%d total_updates=%d",
        config["epochs"],
        len(loaders["train"]),
        total_updates,
    )

    for epoch in range(start_epoch, int(config["epochs"])):
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_start = time.perf_counter()
        epoch_samples = 0
        running: dict[str, torch.Tensor] = {}
        for batch_index, batch in enumerate(loaders["train"]):
            images_one = move_images(batch["image"], device, config)
            has_second_view = "image_view_2" in batch
            if has_second_view:
                images_two = move_images(
                    batch["image_view_2"], device, config
                )
                images = torch.cat((images_one, images_two), dim=0)
            else:
                images = images_one
            binary_targets_one = move_target(
                batch["binary_id"], device, config
            )
            coarse_targets_one = move_target(
                batch["coarse_id"], device, config
            )
            fine_targets_one = move_target(
                batch["fine_id"], device, config
            )
            primary_batch_size = len(images_one)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                outputs = model(images)
                outputs_primary = {
                    key: value[:primary_batch_size]
                    for key, value in outputs.items()
                }
                classification_loss, components = compute_multitask_loss(
                    outputs_primary,
                    binary_targets_one,
                    coarse_targets_one,
                    fine_targets_one,
                    fine_loss_fn,
                    config,
                )
                supcon_weight = float(
                    config["supervised_contrastive_loss_weight"]
                )
                if supcon_weight > 0:
                    if not has_second_view:
                        raise RuntimeError(
                            "SupCon is enabled but the second view is absent."
                        )
                    repeat = 2 if has_second_view else 1
                    supcon_fine_targets = fine_targets_one.repeat(repeat)
                    supcon_coarse_targets = coarse_targets_one.repeat(repeat)
                    if (
                        config["supervised_contrastive_label_level"]
                        == "fine"
                    ):
                        supcon_valid = supcon_fine_targets >= 0
                        supcon_labels = supcon_fine_targets[supcon_valid]
                        supcon_projection = outputs["projection"][
                            supcon_valid
                        ]
                    elif (
                        config["supervised_contrastive_label_level"]
                        == "coarse"
                    ):
                        supcon_labels = supcon_coarse_targets
                        supcon_projection = outputs["projection"]
                    else:
                        raise ValueError(
                            "supervised_contrastive_label_level must be "
                            "fine or coarse."
                        )
                    supcon = supervised_contrastive_loss(
                        supcon_projection,
                        supcon_labels,
                        float(
                            config[
                                "supervised_contrastive_temperature"
                            ]
                        ),
                    )
                else:
                    supcon = next(iter(outputs.values())).sum() * 0.0
                total_loss = classification_loss + supcon_weight * supcon
                group_start = (batch_index // accumulation) * accumulation
                group_size = min(
                    accumulation,
                    len(loaders["train"]) - group_start,
                )
                scaled_loss = total_loss / group_size

            scaler.scale(scaled_loss).backward()
            should_step = (
                (batch_index + 1) % accumulation == 0
                or batch_index + 1 == len(loaders["train"])
            )
            if should_step:
                if float(config["gradient_clip_norm"]) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        float(config["gradient_clip_norm"]),
                        error_if_nonfinite=True,
                    )
                else:
                    scaler.unscale_(optimizer)
                    for parameter in model.parameters():
                        if (
                            parameter.grad is not None
                            and not torch.isfinite(parameter.grad).all()
                        ):
                            raise FloatingPointError(
                                "Non-finite gradient detected."
                            )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_update += 1

            base_batch_size = len(images_one)
            epoch_samples += base_batch_size
            for name, value in {
                "total_loss": total_loss,
                "supcon_loss": supcon,
                **components,
            }.items():
                contribution = value.detach() * base_batch_size
                if name in running:
                    running[name].add_(contribution)
                else:
                    running[name] = contribution

            if (
                (batch_index + 1) % int(config["log_every_n_steps"]) == 0
                or batch_index + 1 == len(loaders["train"])
            ):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                running_loss = float(running["total_loss"].item())
                if not math.isfinite(running_loss):
                    raise FloatingPointError(
                        "Non-finite training loss detected."
                    )
                elapsed = max(time.perf_counter() - epoch_start, 1e-9)
                logger.info(
                    "train epoch=%d/%d step=%d/%d update=%d "
                    "loss=%.5f samples/s=%.2f lr_head=%.3e",
                    epoch + 1,
                    config["epochs"],
                    batch_index + 1,
                    len(loaders["train"]),
                    global_update,
                    running_loss / epoch_samples,
                    epoch_samples / elapsed,
                    optimizer.param_groups[-1]["lr"],
                )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_elapsed = time.perf_counter() - epoch_start
        val_metrics, val_predictions, val_losses = evaluate_model(
            model,
            loaders["val"],
            fine_loss_fn,
            fine_counts,
            fine_patient_counts,
            config,
            device,
            amp_dtype,
            include_features=False,
            fine_prior_tau=selected_fine_tau,
        )
        calibration_audit = None
        if "fine" in active_tasks:
            if fine_loss_fn is None:
                raise RuntimeError("Fine task lost its configured loss.")
            (
                val_metrics,
                val_predictions,
                selected_fine_tau,
                calibration_audit,
            ) = select_fine_inference_tau(
                val_predictions,
                fine_loss_fn,
                fine_counts,
                fine_patient_counts,
                config,
            )
        monitor = metric_for_monitor(
            val_metrics,
            str(config["monitor_metric"]),
            config["hierarchical_composite_weights"],
        )
        row: dict[str, Any] = {
            "epoch": epoch + 1,
            "train_seconds": train_elapsed,
            "train_samples_per_second": epoch_samples
            / max(train_elapsed, 1e-9),
            "learning_rate_encoder": optimizer.param_groups[0]["lr"],
            "learning_rate_head": optimizer.param_groups[-1]["lr"],
            "monitor_metric": monitor,
            "fine_inference_prior_tau": selected_fine_tau,
            "val_binary_auroc": (
                val_metrics["binary"]["auroc"]
                if val_metrics["binary"] is not None
                else None
            ),
            "val_binary_f1": (
                val_metrics["binary"]["f1"]
                if val_metrics["binary"] is not None
                else None
            ),
            "val_coarse_macro_f1": (
                val_metrics["coarse"]["macro_f1_supported"]
                if val_metrics["coarse"] is not None
                else None
            ),
            "val_coarse_macro_f1_all_classes": (
                val_metrics["coarse"]["macro_f1_all_classes"]
                if val_metrics["coarse"] is not None
                else None
            ),
            "val_coarse_balanced_accuracy": (
                val_metrics["coarse"]["balanced_accuracy"]
                if val_metrics["coarse"] is not None
                else None
            ),
            "val_fine_macro_f1": (
                val_metrics["fine"]["macro_f1_supported"]
                if val_metrics["fine"] is not None
                else None
            ),
            "val_fine_macro_f1_all_classes": (
                val_metrics["fine"]["macro_f1_all_classes"]
                if val_metrics["fine"] is not None
                else None
            ),
            "val_hierarchical_accuracy": (
                val_metrics["hierarchy"]["hierarchical_accuracy"]
                if val_metrics["hierarchy"] is not None
                else None
            ),
        }
        row.update(
            {
                f"train_{name}": float(value.item()) / epoch_samples
                for name, value in running.items()
            }
        )
        row.update({f"val_{name}": value for name, value in val_losses.items()})
        history.append(row)
        last_completed_epoch = epoch
        pd.DataFrame(history).to_csv(
            fold_dir / "history.csv", index=False
        )
        write_json(fold_dir / "val_metrics_latest.json", val_metrics)
        if calibration_audit is not None:
            write_json(
                fold_dir / "fine_calibration_latest.json",
                calibration_audit,
            )
        logger.info(
            "eval epoch=%d monitor[%s]=%.6f "
            "coarse_macro_f1=%s fine_macro_f1=%s val_loss=%.5f",
            epoch + 1,
            config["monitor_metric"],
            monitor,
            (
                f"{val_metrics['coarse']['macro_f1_supported']:.6f}"
                if val_metrics["coarse"] is not None
                else "inactive"
            ),
            (
                f"{val_metrics['fine']['macro_f1_supported']:.6f}"
                if val_metrics["fine"] is not None
                else "inactive"
            ),
            val_losses["total_loss"],
        )

        if monitor > best_metric + float(config["checkpoint_min_delta"]):
            best_metric = monitor
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_work_dir / "best_model.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_metric,
                config,
                fine_counts,
                fine_patient_counts,
                fine_loss_fn,
                selected_fine_tau,
                data_split_hash,
                include_optimizer_state=(
                    config["checkpoint_backend"] == "local"
                ),
            )
            logger.info("Saved new best checkpoint: %.6f", best_metric)
        else:
            epochs_without_improvement += 1
        if config["save_epoch_checkpoints"]:
            save_checkpoint(
                checkpoint_work_dir / f"epoch_{epoch + 1:03d}.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_metric,
                config,
                fine_counts,
                fine_patient_counts,
                fine_loss_fn,
                selected_fine_tau,
                data_split_hash,
            )
        if device.type == "mps":
            torch.mps.empty_cache()
        if (
            epochs_without_improvement
            >= int(config["early_stopping_patience"])
        ):
            logger.info(
                "Early stopping after %d non-improving epochs.",
                epochs_without_improvement,
            )
            break

    if not (checkpoint_work_dir / "best_model.pt").is_file():
        raise RuntimeError("Training completed without a best checkpoint.")
    if config["save_last_checkpoint"]:
        save_checkpoint(
            checkpoint_work_dir / "last_model.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            last_completed_epoch,
            best_metric,
            config,
            fine_counts,
            fine_patient_counts,
            fine_loss_fn,
            selected_fine_tau,
            data_split_hash,
        )
    checkpoint = torch.load(
        checkpoint_work_dir / "best_model.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return (
        model,
        pd.DataFrame(history),
        list(map(int, fine_counts)),
        fine_patient_counts,
    )


def serialize_prediction_frame(
    frame: pd.DataFrame,
    path: Path,
    include_features: bool,
) -> None:
    output = frame.copy()
    for column in (
        "binary_logits",
        "coarse_logits",
        "fine_logits",
        "binary_probs",
        "coarse_probs",
        "fine_probs",
    ):
        if column not in output:
            continue
        output[column] = output[column].map(
            lambda values: json.dumps(
                np.asarray(values, dtype=float).tolist(),
                separators=(",", ":"),
            )
        )
    if "features" in output:
        if include_features:
            output["features"] = output["features"].map(
                lambda values: json.dumps(
                    np.asarray(values, dtype=float).tolist(),
                    separators=(",", ":"),
                )
            )
        else:
            output = output.drop(columns=["features"])
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)


def patient_bootstrap_intervals(
    predictions: pd.DataFrame,
    train_fine_counts: Sequence[int],
    train_fine_patient_counts: Sequence[int],
    config: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    iterations = int(config["bootstrap_iterations"])
    if iterations < 1:
        raise ValueError("bootstrap_iterations must be positive.")
    patients = sorted(predictions["pid"].astype(str).unique())
    if len(patients) < 2:
        return {
            "status": "not_evaluable",
            "reason": "fewer than two patients",
            "iterations_requested": iterations,
            "iterations_valid": 0,
        }
    grouped = {
        pid: predictions[predictions["pid"].astype(str) == pid]
        for pid in patients
    }
    rng = np.random.default_rng(seed)
    samples: defaultdict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        selected = rng.choice(patients, size=len(patients), replace=True)
        pieces = []
        for draw_index, pid in enumerate(selected):
            piece = grouped[str(pid)].copy()
            # Make duplicated patient draws statistically independent groups
            # without changing any row-level prediction.
            piece["pid"] = f"{pid}__bootstrap_{draw_index}"
            pieces.append(piece)
        boot = pd.concat(pieces, ignore_index=True)
        bundle = compute_metrics_bundle(
            boot,
            train_fine_counts,
            train_fine_patient_counts,
            config,
        )
        candidates = {
            "binary_auroc": (
                bundle["binary"]["auroc"]
                if bundle["binary"] is not None
                else None
            ),
            "binary_auprc": (
                bundle["binary"]["auprc"]
                if bundle["binary"] is not None
                else None
            ),
            "binary_f1": (
                bundle["binary"]["f1"]
                if bundle["binary"] is not None
                else None
            ),
            "coarse_macro_f1_supported": (
                bundle["coarse"]["macro_f1_supported"]
                if bundle["coarse"] is not None
                else None
            ),
            "coarse_macro_f1_all_classes": (
                bundle["coarse"]["macro_f1_all_classes"]
                if bundle["coarse"] is not None
                else None
            ),
            "coarse_balanced_accuracy": (
                bundle["coarse"]["balanced_accuracy"]
                if bundle["coarse"] is not None
                else None
            ),
            "fine_macro_f1": (
                bundle["fine"]["macro_f1_supported"]
                if bundle["fine"] is not None
                else None
            ),
            "fine_macro_f1_all_classes": (
                bundle["fine"]["macro_f1_all_classes"]
                if bundle["fine"] is not None
                else None
            ),
            "primary_macro_f1_all_classes": (
                bundle["primary_fine"]["macro_f1_all_classes"]
                if bundle["primary_fine"] is not None
                and bundle["primary_fine"]["status"] == "ok"
                else None
            ),
            "hierarchical_accuracy": (
                bundle["hierarchy"]["hierarchical_accuracy"]
                if bundle["hierarchy"] is not None
                else None
            ),
        }
        for name, value in candidates.items():
            if value is not None and math.isfinite(float(value)):
                samples[name].append(float(value))
    alpha = 1.0 - float(config["bootstrap_confidence"])
    intervals: dict[str, Any] = {}
    for name, values in samples.items():
        intervals[name] = {
            "lower": float(np.quantile(values, alpha / 2)),
            "upper": float(np.quantile(values, 1 - alpha / 2)),
            "mean": float(np.mean(values)),
            "valid_iterations": len(values),
        }
    return {
        "status": "ok",
        "method": "patient-level percentile bootstrap",
        "confidence": float(config["bootstrap_confidence"]),
        "iterations_requested": iterations,
        "intervals": intervals,
    }


def paired_mcnemar_test(
    current: pd.DataFrame,
    baseline_csv: Path,
    binary_threshold: float,
) -> dict[str, Any]:
    if not baseline_csv.is_file():
        raise FileNotFoundError(
            f"Paired baseline predictions not found: {baseline_csv}"
        )
    baseline = pd.read_csv(baseline_csv)
    required = {"filename", "binary_id", "binary_probability_roi"}
    missing = required - set(baseline)
    if missing:
        raise ValueError(
            f"Paired baseline CSV missing columns: {sorted(missing)}"
        )
    current_view = current[["filename", "binary_id", "binary_probs"]].copy()
    current_view["current_prediction"] = current_view["binary_probs"].map(
        lambda values: int(
            np.asarray(values)[1] >= binary_threshold
        )
    )
    merged = current_view.merge(
        baseline,
        on=["filename", "binary_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(current_view) or len(merged) != len(baseline):
        raise ValueError(
            "Paired prediction manifests do not have identical samples."
        )
    baseline_prediction = (
        merged["binary_probability_roi"].astype(float).to_numpy()
        >= binary_threshold
    )
    target = merged["binary_id"].astype(int).to_numpy()
    current_correct = merged["current_prediction"].to_numpy() == target
    baseline_correct = baseline_prediction == target
    current_only = int(np.sum(current_correct & ~baseline_correct))
    baseline_only = int(np.sum(~current_correct & baseline_correct))
    discordant = current_only + baseline_only
    p_value = (
        float(
            binomtest(
                min(current_only, baseline_only),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
        if discordant
        else 1.0
    )
    return {
        "test": "exact McNemar (binomial)",
        "decision_threshold": float(binary_threshold),
        "n": len(merged),
        "current_correct_baseline_wrong": current_only,
        "current_wrong_baseline_correct": baseline_only,
        "discordant": discordant,
        "p_value": p_value,
    }


# %%
# CELL 7 - ROI-level aggregation, including a genuinely trained attention MIL
# aggregator. Ambiguous ROI labels are never silently majority-labelled.
@dataclass
class RoiBag:
    roi_id: str
    pid: str
    target: int
    probabilities: np.ndarray
    features: np.ndarray | None
    filenames: tuple[str, ...]


def extract_roi_bags(
    predictions: pd.DataFrame,
    task: str,
    require_features: bool,
) -> tuple[list[RoiBag], list[dict[str, Any]], int]:
    if task not in {"binary", "coarse", "fine"}:
        raise ValueError(f"Unknown ROI task: {task}")
    valid_metadata = ~(
        predictions["visit"].map(is_missing_token)
        | predictions["lesion"].map(is_missing_token)
    )
    eligible = predictions.loc[valid_metadata].copy()
    skipped_missing_metadata = int((~valid_metadata).sum())
    if eligible.empty:
        return [], [], skipped_missing_metadata
    eligible["roi_id"] = (
        eligible["pid"].astype(str)
        + "::v"
        + eligible["visit"].astype(str)
        + "::"
        + eligible["lesion"].astype(str)
    )
    target_column = f"{task}_id"
    probability_column = f"{task}_probs"
    bags: list[RoiBag] = []
    conflicts: list[dict[str, Any]] = []
    for roi_id, group in eligible.groupby("roi_id", sort=True):
        targets = sorted(set(group[target_column].astype(int)))
        if task == "fine" and targets == [-1]:
            continue
        if task == "fine" and -1 in targets:
            conflicts.append(
                {
                    "roi_id": roi_id,
                    "pid": str(group["pid"].iloc[0]),
                    "task": task,
                    "reason": "mixed_missing_and_fine_labels",
                    "targets": targets,
                    "filenames": group["filename"].tolist(),
                }
            )
            continue
        if len(targets) != 1:
            conflicts.append(
                {
                    "roi_id": roi_id,
                    "pid": str(group["pid"].iloc[0]),
                    "task": task,
                    "reason": "conflicting_targets",
                    "targets": targets,
                    "filenames": group["filename"].tolist(),
                }
            )
            continue
        probability_matrix = np.stack(
            group[probability_column].to_numpy()
        ).astype(np.float32)
        feature_matrix: np.ndarray | None = None
        if require_features:
            if "features" not in group:
                raise ValueError(
                    "Attention ROI evaluation requires exported features."
                )
            feature_matrix = np.stack(group["features"].to_numpy()).astype(
                np.float32
            )
        bags.append(
            RoiBag(
                roi_id=str(roi_id),
                pid=str(group["pid"].iloc[0]),
                target=int(targets[0]),
                probabilities=probability_matrix,
                features=feature_matrix,
                filenames=tuple(group["filename"].astype(str)),
            )
        )
    return bags, conflicts, skipped_missing_metadata


def aggregate_roi_bags(
    bags: Sequence[RoiBag],
    method: str,
    num_classes: int,
) -> pd.DataFrame:
    if method not in {"mean", "vote"}:
        raise ValueError("Non-attention ROI method must be mean or vote.")
    rows: list[dict[str, Any]] = []
    for bag in bags:
        if method == "mean":
            probability = bag.probabilities.mean(axis=0)
        else:
            votes = np.bincount(
                bag.probabilities.argmax(axis=1),
                minlength=num_classes,
            )
            probability = votes.astype(np.float64) / votes.sum()
        probability = probability / probability.sum()
        rows.append(
            {
                "roi_id": bag.roi_id,
                "pid": bag.pid,
                "target": bag.target,
                "probabilities": probability,
                "images": len(bag.filenames),
                "filenames": "|".join(bag.filenames),
            }
        )
    return pd.DataFrame(rows)


def compute_roi_task_metrics(
    aggregated: pd.DataFrame,
    task: str,
    binary_threshold: float,
) -> dict[str, Any]:
    if aggregated.empty:
        return {
            "status": "not_evaluable",
            "reason": "no label-consistent ROI groups",
            "n_rois": 0,
        }
    targets = aggregated["target"].to_numpy(dtype=np.int64)
    probabilities = np.stack(aggregated["probabilities"].to_numpy())
    if task == "binary":
        metrics = compute_binary_metrics(
            targets, probabilities[:, 1], binary_threshold
        )
    elif task == "coarse":
        metrics = compute_multiclass_metrics(
            targets, probabilities, COARSE_NAMES
        )
    elif task == "fine":
        metrics = compute_multiclass_metrics(
            targets, probabilities, FINE_NAMES
        )
    else:
        raise ValueError(f"Unknown ROI task: {task}")
    return {"status": "ok", "n_rois": len(aggregated), **metrics}


class GatedAttentionMIL(nn.Module):
    def __init__(
        self, feature_dim: int, hidden_dim: int, num_classes: int
    ) -> None:
        super().__init__()
        self.attention_v = nn.Linear(feature_dim, hidden_dim)
        self.attention_u = nn.Linear(feature_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(
        self, bag_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if bag_features.ndim != 2:
            raise ValueError("MIL bag features must have shape [instances, D].")
        gated = torch.tanh(self.attention_v(bag_features)) * torch.sigmoid(
            self.attention_u(bag_features)
        )
        weights = torch.softmax(
            self.attention_w(gated).squeeze(-1), dim=0
        )
        pooled = torch.sum(weights[:, None] * bag_features, dim=0)
        logits = self.classifier(pooled)
        return logits, weights


def predict_attention_bags(
    model: GatedAttentionMIL,
    bags: Sequence[RoiBag],
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for bag in bags:
            if bag.features is None:
                raise ValueError("Attention bag has no feature matrix.")
            features = torch.as_tensor(
                bag.features, dtype=torch.float32, device=device
            )
            logits, attention = model(features)
            probability = logits.softmax(dim=0).float().cpu().numpy()
            rows.append(
                {
                    "roi_id": bag.roi_id,
                    "pid": bag.pid,
                    "target": bag.target,
                    "probabilities": probability,
                    "images": len(bag.filenames),
                    "filenames": "|".join(bag.filenames),
                    "attention_weights": attention.float().cpu().numpy(),
                }
            )
    return pd.DataFrame(rows)


def train_attention_mil(
    train_bags: Sequence[RoiBag],
    val_bags: Sequence[RoiBag],
    test_bags: Sequence[RoiBag],
    task: str,
    config: Mapping[str, Any],
    device: torch.device,
    output_path: Path,
    logger: logging.Logger,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if not train_bags or not val_bags or not test_bags:
        raise ValueError(
            f"Attention MIL for {task} requires non-empty train/val/test bags."
        )
    feature_dims = {
        bag.features.shape[1]
        for bag in (*train_bags, *val_bags, *test_bags)
        if bag.features is not None
    }
    if len(feature_dims) != 1:
        raise ValueError(
            f"ROI feature dimensions are inconsistent: {sorted(feature_dims)}"
        )
    feature_dim = next(iter(feature_dims))
    num_classes = {
        "binary": len(BINARY_NAMES),
        "coarse": len(COARSE_NAMES),
        "fine": len(FINE_NAMES),
    }[task]
    model = GatedAttentionMIL(
        feature_dim,
        int(config["roi_attention_hidden_dim"]),
        num_classes,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["roi_attention_learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    rng = random.Random(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(int(config["roi_attention_epochs"])):
        model.train()
        order = list(range(len(train_bags)))
        rng.shuffle(order)
        loss_sum = 0.0
        for bag_index in order:
            bag = train_bags[bag_index]
            if bag.features is None:
                raise ValueError("Training attention bag has no features.")
            features = torch.as_tensor(
                bag.features, dtype=torch.float32, device=device
            )
            target = torch.tensor(
                [bag.target], dtype=torch.long, device=device
            )
            logits, _ = model(features)
            loss = F.cross_entropy(logits.unsqueeze(0), target)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite attention MIL loss for task={task}."
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config["gradient_clip_norm"]),
                error_if_nonfinite=True,
            )
            optimizer.step()
            loss_sum += float(loss.detach())
        val_predictions = predict_attention_bags(model, val_bags, device)
        val_metrics = compute_roi_task_metrics(
            val_predictions,
            task,
            float(config["binary_decision_threshold"]),
        )
        if val_metrics["status"] != "ok":
            raise RuntimeError(
                f"Attention MIL validation for {task} is not evaluable."
            )
        score = (
            val_metrics["f1"]
            if task == "binary"
            else val_metrics["macro_f1"]
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": loss_sum / len(train_bags),
                "val_score": score,
            }
        )
        logger.info(
            "roi-attention task=%s epoch=%d/%d loss=%.5f val_score=%.5f",
            task,
            epoch + 1,
            config["roi_attention_epochs"],
            loss_sum / len(train_bags),
            score,
        )
        if score > best_score:
            best_score = float(score)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= int(config["roi_attention_early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("Attention MIL did not produce a best state.")
    model.load_state_dict(best_state, strict=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "task": task,
            "feature_dim": feature_dim,
            "hidden_dim": int(config["roi_attention_hidden_dim"]),
            "num_classes": num_classes,
            "state_dict": best_state,
            "history": history,
            "best_validation_score": best_score,
        },
        output_path,
    )
    val_predictions = predict_attention_bags(model, val_bags, device)
    test_predictions = predict_attention_bags(model, test_bags, device)
    metrics = compute_roi_task_metrics(
        test_predictions,
        task,
        float(config["binary_decision_threshold"]),
    )
    metrics["best_validation_score"] = best_score
    metrics["training_history"] = history
    return metrics, val_predictions, test_predictions


def serialize_roi_predictions(frame: pd.DataFrame, path: Path) -> None:
    output = frame.copy()
    if output.empty:
        output.to_csv(path, index=False)
        return
    output["probabilities"] = output["probabilities"].map(
        lambda value: json.dumps(
            np.asarray(value, dtype=float).tolist(), separators=(",", ":")
        )
    )
    if "attention_weights" in output:
        output["attention_weights"] = output["attention_weights"].map(
            lambda value: json.dumps(
                np.asarray(value, dtype=float).tolist(),
                separators=(",", ":"),
            )
        )
    output.to_csv(path, index=False)


def run_roi_evaluation(
    split_predictions: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
    device: torch.device,
    run_dir: Path,
    fold_name: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    roi_dir = run_dir / "predictions" / fold_name / "roi"
    roi_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {}
    conflict_rows: list[dict[str, Any]] = []
    bag_cache: dict[str, dict[str, list[RoiBag]]] = {}
    skipped_metadata: dict[str, dict[str, int]] = {}
    for task in ("binary", "coarse", "fine"):
        bag_cache[task] = {}
        skipped_metadata[task] = {}
        for split_name, predictions in split_predictions.items():
            bags, conflicts, skipped = extract_roi_bags(
                predictions,
                task,
                require_features="attention" in config["roi_aggregations"],
            )
            bag_cache[task][split_name] = bags
            skipped_metadata[task][split_name] = skipped
            for conflict in conflicts:
                conflict["split"] = split_name
            conflict_rows.extend(conflicts)

    conflict_frame = pd.DataFrame(conflict_rows)
    conflict_frame.to_csv(roi_dir / "conflicts.csv", index=False)
    if conflict_rows and config["roi_conflict_policy"] == "raise":
        raise ValueError(
            f"ROI evaluation found {len(conflict_rows)} task-level target "
            f"conflicts; see {roi_dir / 'conflicts.csv'}"
        )
    logger.info(
        "ROI audit: task-level conflicts=%d policy=%s",
        len(conflict_rows),
        config["roi_conflict_policy"],
    )

    for method in config["roi_aggregations"]:
        method_metrics: dict[str, Any] = {}
        for task in ("binary", "coarse", "fine"):
            if method in {"mean", "vote"}:
                num_classes = {
                    "binary": len(BINARY_NAMES),
                    "coarse": len(COARSE_NAMES),
                    "fine": len(FINE_NAMES),
                }[task]
                aggregated = aggregate_roi_bags(
                    bag_cache[task]["test"], method, num_classes
                )
                serialize_roi_predictions(
                    aggregated,
                    roi_dir / f"{method}_{task}_test.csv",
                )
                task_metrics = compute_roi_task_metrics(
                    aggregated,
                    task,
                    float(config["binary_decision_threshold"]),
                )
            elif method == "attention":
                task_metrics, val_predictions, test_predictions = (
                    train_attention_mil(
                        bag_cache[task]["train"],
                        bag_cache[task]["val"],
                        bag_cache[task]["test"],
                        task,
                        config,
                        device,
                        run_dir
                        / "models"
                        / f"{fold_name}_roi_attention_{task}.pt",
                        logger,
                        stable_int_seed(
                            config["seed"], fold_name, "attention", task
                        ),
                    )
                )
                serialize_roi_predictions(
                    val_predictions,
                    roi_dir / f"attention_{task}_val.csv",
                )
                serialize_roi_predictions(
                    test_predictions,
                    roi_dir / f"attention_{task}_test.csv",
                )
            else:
                raise RuntimeError(f"Unreachable ROI method: {method}")
            task_metrics["train_roi_bags"] = len(
                bag_cache[task]["train"]
            )
            task_metrics["val_roi_bags"] = len(bag_cache[task]["val"])
            task_metrics["test_roi_bags"] = len(bag_cache[task]["test"])
            task_metrics["skipped_rows_missing_roi_metadata"] = (
                skipped_metadata[task]
            )
            method_metrics[task] = task_metrics
        metrics[method] = method_metrics
    metrics["conflict_count_task_level"] = len(conflict_rows)
    metrics["conflict_policy"] = config["roi_conflict_policy"]
    write_json(
        run_dir / "metrics" / fold_name / "roi_metrics.json", metrics
    )
    return metrics


# %%
# CELL 8 - Visualizations and human-readable reports.
def plot_class_distributions(
    split_frames: Mapping[str, pd.DataFrame],
    output_path: Path,
) -> None:
    rows = []
    for split_name, frame in split_frames.items():
        counts = frame["class"].value_counts()
        for class_name in COARSE_NAMES:
            rows.append(
                {
                    "split": split_name,
                    "class": class_name,
                    "images": int(counts.get(class_name, 0)),
                }
            )
    data = pd.DataFrame(rows)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=data, x="class", y="images", hue="split")
    plt.yscale("log")
    plt.title("CystoDS split distribution (log scale)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_training_history(history: pd.DataFrame, output_path: Path) -> None:
    if history.empty:
        raise ValueError("Cannot plot empty training history.")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    loss_columns = [
        column
        for column in (
            "train_total_loss",
            "val_total_loss",
            "train_binary_loss",
            "train_coarse_loss",
            "train_fine_loss",
            "train_consistency_loss",
        )
        if column in history
    ]
    for column in loss_columns:
        axes[0].plot(history["epoch"], history[column], marker="o", label=column)
    axes[0].set_title("Training and validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(fontsize=8)
    metric_columns = [
        column
        for column in (
            "monitor_metric",
            "val_binary_auroc",
            "val_coarse_macro_f1",
            "val_fine_macro_f1",
            "val_hierarchical_accuracy",
        )
        if column in history
    ]
    for column in metric_columns:
        axes[1].plot(history["epoch"], history[column], marker="o", label=column)
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion(
    matrix: Sequence[Sequence[int]],
    names: Sequence[str],
    title: str,
    output_path: Path,
) -> None:
    array = np.asarray(matrix)
    size = 14 if len(names) > 10 else 8
    plt.figure(figsize=(size, size))
    sns.heatmap(
        array,
        cmap="Blues",
        annot=len(names) <= 10,
        fmt="d",
        xticklabels=names,
        yticklabels=names,
        cbar=True,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_binary_curves(
    predictions: pd.DataFrame, output_path: Path
) -> None:
    targets = predictions["binary_id"].to_numpy(dtype=np.int64)
    probabilities = np.stack(predictions["binary_probs"].to_numpy())[:, 1]
    if len(np.unique(targets)) != 2:
        # There is no statistically valid ROC/PR curve with one target class.
        # Create an explicit explanatory figure rather than a fabricated curve.
        plt.figure(figsize=(7, 4))
        plt.text(
            0.5,
            0.5,
            "Binary ROC/PR not evaluable:\ntest set contains one target class.",
            ha="center",
            va="center",
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=180)
        plt.close()
        return
    fpr, tpr, _ = roc_curve(targets, probabilities)
    precision, recall, _ = precision_recall_curve(targets, probabilities)
    auroc = roc_auc_score(targets, probabilities)
    auprc = average_precision_score(targets, probabilities)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, label=f"AUROC={auroc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="grey")
    axes[0].set_xlabel("False-positive rate")
    axes[0].set_ylabel("True-positive rate")
    axes[0].set_title("Binary ROC")
    axes[0].legend()
    axes[1].plot(recall, precision, label=f"AUPRC={auprc:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Binary precision-recall")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_per_class_recall(
    metrics: Mapping[str, Any],
    output_path: Path,
) -> None:
    coarse_rows = (
        metrics["coarse"]["per_class"]
        if metrics["coarse"] is not None
        else []
    )
    fine_rows = metrics["fine"]["per_class"] if metrics["fine"] else []
    rows = [
        {
            "name": f"{row['class_name']} (n={row['support']})",
            "recall": row["recall"],
            "level": "coarse",
        }
        for row in coarse_rows
        if row["recall"] is not None
    ] + [
        {
            "name": f"{row['class_name']} (n={row['support']})",
            "recall": row["recall"],
            "level": "fine",
        }
        for row in fine_rows
        if row["recall"] is not None
    ]
    if not rows:
        raise ValueError("Per-class recall requires coarse or fine metrics.")
    data = pd.DataFrame(rows)
    height = max(6, 0.33 * len(data))
    plt.figure(figsize=(10, height))
    sns.barplot(data=data, x="recall", y="name", hue="level")
    plt.xlim(0, 1)
    plt.title("Per-class recall")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def generate_sample_grid(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    output_path: Path,
) -> None:
    per_class = int(config["sample_grid_images_per_class"])
    if per_class < 1:
        raise ValueError("sample_grid_images_per_class must be positive.")
    fig, axes = plt.subplots(
        len(COARSE_NAMES),
        per_class,
        figsize=(3 * per_class, 2.7 * len(COARSE_NAMES)),
        squeeze=False,
    )
    for row_index, class_name in enumerate(COARSE_NAMES):
        subset = frame[frame["class"] == class_name]
        if subset.empty:
            raise ValueError(f"Sample grid lacks class {class_name}.")
        sampled = subset.sample(
            n=min(per_class, len(subset)),
            random_state=stable_int_seed(
                config["seed"], "sample_grid", class_name
            ),
            replace=False,
        )
        for column_index in range(per_class):
            axis = axes[row_index, column_index]
            axis.axis("off")
            if column_index >= len(sampled):
                continue
            row = sampled.iloc[column_index]
            with Image.open(row["image_path"]) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_title(f"{class_name}\n{row['subclass']}", fontsize=9)
    fig.suptitle("Real CystoDS samples used by this run", y=1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def markdown_scalar(value: Any) -> str:
    if value is None:
        return "not evaluable"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def metrics_table_markdown(metrics: Mapping[str, Any]) -> str:
    rows: list[tuple[str, Any]] = []
    if metrics["binary"] is not None:
        rows.extend(
            [
                ("Binary AUROC", metrics["binary"]["auroc"]),
                ("Binary AUPRC", metrics["binary"]["auprc"]),
                ("Binary F1", metrics["binary"]["f1"]),
                ("Binary sensitivity", metrics["binary"]["sensitivity"]),
                ("Binary specificity", metrics["binary"]["specificity"]),
                ("Binary MCC", metrics["binary"]["mcc"]),
            ]
        )
    if metrics["coarse"] is not None:
        rows.extend(
            [
                (
                    "Coarse macro-F1 (supported)",
                    metrics["coarse"]["macro_f1_supported"],
                ),
                (
                    "Coarse macro-F1 (all 5 classes)",
                    metrics["coarse"]["macro_f1_all_classes"],
                ),
                (
                    "Coarse balanced accuracy",
                    metrics["coarse"]["balanced_accuracy"],
                ),
                ("Coarse MCC", metrics["coarse"]["mcc"]),
                (
                    "Coarse macro-AUROC",
                    metrics["coarse"]["macro_auroc_ovr"],
                ),
            ]
        )
    if metrics["fine"] is not None:
        rows.extend(
            [
                (
                    "Fine macro-F1 (supported)",
                    metrics["fine"]["macro_f1_supported"],
                ),
                (
                    "Fine macro-F1 (all 22 classes)",
                    metrics["fine"]["macro_f1_all_classes"],
                ),
            ]
        )
    if (
        metrics["primary_fine"] is not None
        and metrics["primary_fine"]["status"] == "ok"
    ):
        rows.extend(
            [
                (
                    "Primary fine macro-F1 (supported)",
                    metrics["primary_fine"]["macro_f1_supported"],
                ),
                (
                    "Primary fine macro-F1 (fixed denominator)",
                    metrics["primary_fine"]["macro_f1_all_classes"],
                ),
            ]
        )
    if metrics["hierarchy"] is not None:
        rows.extend(
            [
                (
                    "Hierarchical accuracy",
                    metrics["hierarchy"]["hierarchical_accuracy"],
                ),
                (
                    "Cross-parent error rate",
                    metrics["hierarchy"]["cross_parent_error_rate"],
                ),
                (
                    "Tail-class recall",
                    metrics["hierarchy"]["tail_class_recall"],
                ),
            ]
        )
    if not rows:
        raise ValueError("Metrics report contains no active task.")
    return pd.DataFrame(rows, columns=["Metric", "Value"]).assign(
        Value=lambda data: data["Value"].map(markdown_scalar)
    ).to_markdown(index=False)


def build_fold_report(
    fold_name: str,
    config: Mapping[str, Any],
    split_summary: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    wlc_metrics: Mapping[str, Any] | None,
    roi_metrics: Mapping[str, Any] | None,
    paired: Mapping[str, Any] | None,
    external: Mapping[str, Any] | None,
    history: pd.DataFrame,
    output_path: Path,
) -> None:
    if config["inclusion_manifest_csv"] is None:
        inclusion_note = (
            "- No paper inclusion manifest was supplied. The public release "
            "contains 998 malignant images whereas the paper's binary table "
            "uses 994 and does not publish the excluded filenames/PIDs; this "
            "run is therefore paper-like, not claimed as paper-exact."
        )
    else:
        inclusion_note = (
            "- The run was restricted by the explicit inclusion manifest "
            f"`{config['inclusion_manifest_csv']}`."
        )
    lines = [
        f"# CystoDS experiment report - {fold_name}",
        "",
        f"- Generated: {utc_now_iso()}",
        f"- Profile: `{config['run_profile']}`",
        f"- Task mode: `{config['task_mode']}`",
        (
            f"- Encoder: `{config['model_name']}` "
            f"(pretrained={config['pretrained']})"
        ),
        f"- Fine objective: `{config['fine_loss']}`",
        f"- Sampler: `{config['sampler']}`",
        f"- Epochs completed: {len(history)}",
        "",
    ]
    if config["run_profile"] == "smoke":
        lines.extend(
            [
                (
                    "> This is a real-data functional smoke test using three "
                    "patient-disjoint PIDs. Its metrics are not a scientific "
                    "estimate and must not be compared with the paper baseline."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Protocol decisions",
            "",
            (
                "- Binary ROI is derived from `class in {Malignant, "
                "Non-malignant}`; the `bca` column is not used because it "
                "differs for the PreMalignant record."
            ),
            (
                "- The fine head has exactly 22 published subclasses. "
                "`Normal mucosa` has `fine_id=-1` and is masked from fine loss."
            ),
            (
                "- All splits are patient-disjoint. Literal CSV `NA` values "
                "are preserved."
            ),
            (
                "- CSV filename stems are mapped to the dataset's canonical "
                "PNG files. No missing-file or random-weight fallback is used."
            ),
            (
                "- ROI groups with conflicting ground truth are excluded and "
                "reported, or raise when configured."
            ),
            inclusion_note,
            "",
            "## Split",
            "",
            pd.DataFrame(
                [
                    {
                        "Split": name,
                        "Images": values["rows"],
                        "Patients": values["patients"],
                    }
                    for name, values in split_summary["splits"].items()
                ]
            ).to_markdown(index=False),
            "",
            "## Internal test metrics",
            "",
            metrics_table_markdown(test_metrics),
            "",
            "## Patient bootstrap",
            "",
            "```json",
            json.dumps(json_ready(bootstrap), indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    if wlc_metrics is not None:
        lines.extend(
            [
                "## WLC-only test",
                "",
                metrics_table_markdown(wlc_metrics),
                "",
            ]
        )
    if roi_metrics is not None:
        lines.extend(
            [
                "## ROI-level evaluation",
                "",
                "```json",
                json.dumps(json_ready(roi_metrics), indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )
    if paired is not None:
        lines.extend(
            [
                "## Paired significance",
                "",
                "```json",
                json.dumps(json_ready(paired), indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )
    if external is not None:
        lines.extend(
            [
                "## External binary validation",
                "",
                "```json",
                json.dumps(json_ready(external), indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Artifact map",
            "",
            "- `checkpoints/`: best and optional last training state",
            "- `logs/`: console-equivalent training progress and history",
            "- `metrics/`: machine-readable metrics, CIs, WLC and ROI results",
            "- `predictions/`: image-level and ROI-level predictions",
            "- `visualizations/`: curves, confusion matrices and data samples",
            "- `models/`: copied best base model and trained attention MIL models",
            "- `splits/`: exact row and patient manifests",
            "- `source/`: exact pre-notebook and usage guide used by the run",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_artifact_manifest(run_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    child_roots: list[Path] = []
    runs_root = run_dir / "runs"
    if runs_root.is_dir():
        child_roots = sorted(
            path
            for path in runs_root.iterdir()
            if path.is_dir()
            and (path / "artifact_manifest.json").is_file()
        )
    covered_child_files: set[Path] = set()
    for child_root in child_roots:
        child_manifest_path = child_root / "artifact_manifest.json"
        child_rows = json.loads(
            child_manifest_path.read_text(encoding="utf-8")
        )
        if not isinstance(child_rows, list):
            raise TypeError(
                f"Child artifact manifest must be a list: {child_manifest_path}"
            )
        declared_paths: set[Path] = set()
        for child_row in child_rows:
            relative = Path(str(child_row["path"]))
            child_file = child_root / relative
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not child_file.is_file()
            ):
                raise ValueError(
                    f"Invalid child artifact path: {child_row['path']}"
                )
            if int(child_row["bytes"]) != child_file.stat().st_size:
                raise ValueError(
                    f"Child artifact size changed after sealing: {child_file}"
                )
            declared_paths.add(child_file)
            covered_child_files.add(child_file)
            rows.append(
                {
                    "path": str(child_file.relative_to(run_dir)),
                    "bytes": int(child_row["bytes"]),
                    # Reuse the already sealed child digest instead of
                    # re-reading multi-GB checkpoints in the parent suite.
                    "sha256": str(child_row["sha256"]),
                }
            )
        actual_child_files = {
            path
            for path in child_root.rglob("*")
            if path.is_file() and path != child_manifest_path
        }
        if actual_child_files != declared_paths:
            raise ValueError(
                "Child artifact file set changed after sealing: "
                f"{child_root}"
            )
        rows.append(
            {
                "path": str(child_manifest_path.relative_to(run_dir)),
                "bytes": child_manifest_path.stat().st_size,
                "sha256": sha256_file(child_manifest_path),
            }
        )
        covered_child_files.add(child_manifest_path)
    for path in sorted(run_dir.rglob("*")):
        if (
            not path.is_file()
            or path == run_dir / "artifact_manifest.json"
            or path in covered_child_files
        ):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        rows.append(
            {
                "path": str(path.relative_to(run_dir)),
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    rows.sort(key=lambda row: row["path"])
    write_json(run_dir / "artifact_manifest.json", rows)


# %%
# CELL 9 - End-to-end orchestration.
def make_deterministic_eval_loader(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> DataLoader:
    _, eval_transform, _ = build_transforms(config)
    dataset = CystoDataset(frame, eval_transform, second_view_transform=None)
    num_workers = (
        0 if device.type == "mps" else int(config["eval_num_workers"])
    )
    kwargs: dict[str, Any] = {
        "batch_size": int(config["eval_batch_size"]),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": bool(config.get("pin_memory", True) and device.type == "cuda"),
        "persistent_workers": False,
        "worker_init_fn": make_worker_init_fn(seed),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(config["eval_prefetch_factor"])
    return DataLoader(dataset, **kwargs)


def evaluate_external_binary(
    model: nn.Module,
    config: Mapping[str, Any],
    device: torch.device,
    amp_dtype: torch.dtype | None,
    run_dir: Path,
    fold_name: str,
) -> dict[str, Any]:
    _, eval_transform, _ = build_transforms(config)
    dataset = ExternalBinaryDataset(
        Path(config["external_manifest_csv"]),
        Path(config["external_image_root"]),
        eval_transform,
        config,
    )
    num_workers = (
        0 if device.type == "mps" else int(config["eval_num_workers"])
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(config["eval_batch_size"]),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": bool(
            config["pin_memory"] and device.type == "cuda"
        ),
        "persistent_workers": False,
        "worker_init_fn": make_worker_init_fn(
            stable_int_seed(config["seed"], "external_loader")
        ),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(
            config["eval_prefetch_factor"]
        )
    loader = DataLoader(dataset, **loader_kwargs)
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            images = move_images(batch["image"], device, config)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                outputs = model(images)
            probabilities = (
                outputs["binary_logits"].softmax(dim=1).float().cpu().numpy()
            )
            for index, probability in enumerate(probabilities):
                rows.append(
                    {
                        "filename": batch["filename"][index],
                        "pid": batch["pid"][index],
                        "binary_id": int(batch["binary_id"][index]),
                        "binary_probability_roi": float(probability[1]),
                    }
                )
    predictions = pd.DataFrame(rows)
    metrics = compute_binary_metrics(
        predictions["binary_id"].to_numpy(dtype=np.int64),
        predictions["binary_probability_roi"].to_numpy(dtype=float),
        float(config["binary_decision_threshold"]),
    )
    patients = sorted(predictions["pid"].astype(str).unique())
    if len(patients) < 2:
        bootstrap = {
            "status": "not_evaluable",
            "reason": "fewer than two external patients",
            "iterations_requested": int(config["bootstrap_iterations"]),
            "iterations_valid": 0,
        }
    else:
        grouped = {
            patient: predictions[
                predictions["pid"].astype(str).eq(patient)
            ]
            for patient in patients
        }
        rng = np.random.default_rng(
            stable_int_seed(config["seed"], "external_bootstrap")
        )
        bootstrap_samples: defaultdict[str, list[float]] = defaultdict(list)
        for _ in range(int(config["bootstrap_iterations"])):
            sampled_patients = rng.choice(
                patients,
                size=len(patients),
                replace=True,
            )
            sample = pd.concat(
                [grouped[str(patient)] for patient in sampled_patients],
                ignore_index=True,
            )
            sample_metrics = compute_binary_metrics(
                sample["binary_id"].to_numpy(dtype=np.int64),
                sample["binary_probability_roi"].to_numpy(dtype=float),
                float(config["binary_decision_threshold"]),
            )
            for metric_name in (
                "accuracy",
                "f1",
                "mcc",
                "balanced_accuracy",
                "auroc",
                "auprc",
            ):
                value = sample_metrics[metric_name]
                if value is not None and math.isfinite(float(value)):
                    bootstrap_samples[metric_name].append(float(value))
        alpha = 1.0 - float(config["bootstrap_confidence"])
        bootstrap = {
            "status": "ok",
            "method": "external patient-level percentile bootstrap",
            "confidence": float(config["bootstrap_confidence"]),
            "iterations_requested": int(config["bootstrap_iterations"]),
            "intervals": {
                metric_name: {
                    "lower": float(np.quantile(values, alpha / 2)),
                    "upper": float(np.quantile(values, 1 - alpha / 2)),
                    "mean": float(np.mean(values)),
                    "valid_iterations": len(values),
                }
                for metric_name, values in bootstrap_samples.items()
                if values
            },
        }
    metrics["patient_bootstrap"] = bootstrap
    predictions.to_csv(
        run_dir
        / "predictions"
        / fold_name
        / "external_binary_predictions.csv",
        index=False,
    )
    visual_dir = run_dir / "visualizations" / fold_name
    visual_dir.mkdir(parents=True, exist_ok=True)
    curve_frame = predictions.copy()
    curve_frame["binary_probs"] = curve_frame[
        "binary_probability_roi"
    ].map(lambda value: np.asarray([1.0 - float(value), float(value)]))
    plot_binary_curves(
        curve_frame,
        visual_dir / "external_binary_roc_pr_curves.png",
    )
    plot_confusion(
        metrics["confusion_matrix"],
        BINARY_NAMES,
        "External binary confusion matrix",
        visual_dir / "external_binary_confusion_matrix.png",
    )
    write_json(
        run_dir / "metrics" / fold_name / "external_binary_metrics.json",
        metrics,
    )
    write_json(
        run_dir
        / "metrics"
        / fold_name
        / "external_patient_bootstrap_ci.json",
        bootstrap,
    )
    return metrics


def export_fold_visualizations(
    history: pd.DataFrame,
    test_predictions: pd.DataFrame,
    test_metrics: Mapping[str, Any],
    split_frames: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
    run_dir: Path,
    fold_name: str,
) -> None:
    visual_dir = run_dir / "visualizations" / fold_name
    visual_dir.mkdir(parents=True, exist_ok=True)
    plot_training_history(history, visual_dir / "training_history.png")
    plot_class_distributions(
        split_frames, visual_dir / "split_class_distribution.png"
    )
    if test_metrics["coarse"] is not None:
        plot_confusion(
            test_metrics["coarse"]["confusion_matrix"],
            COARSE_NAMES,
            "Five-class confusion matrix",
            visual_dir / "coarse_confusion_matrix.png",
        )
    if test_metrics["fine"] is not None:
        plot_confusion(
            test_metrics["fine"]["confusion_matrix"],
            FINE_NAMES,
            "22-subclass confusion matrix (Normal mucosa excluded)",
            visual_dir / "fine_confusion_matrix.png",
        )
    if test_metrics["binary"] is not None:
        plot_binary_curves(
            test_predictions, visual_dir / "binary_roc_pr_curves.png"
        )
    if test_metrics["coarse"] is not None or test_metrics["fine"] is not None:
        plot_per_class_recall(
            test_metrics, visual_dir / "per_class_recall.png"
        )
    if config["generate_sample_grid"]:
        generate_sample_grid(
            pd.concat(split_frames.values(), ignore_index=True),
            config,
            visual_dir / "real_data_sample_grid.png",
        )


def aggregate_cross_validation_metrics(
    fold_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not fold_results:
        raise ValueError("Cannot aggregate zero fold results.")
    selectors = {
        "binary_auroc": lambda value: (
            value["binary"]["auroc"] if value["binary"] is not None else None
        ),
        "binary_auprc": lambda value: (
            value["binary"]["auprc"] if value["binary"] is not None else None
        ),
        "binary_f1": lambda value: (
            value["binary"]["f1"] if value["binary"] is not None else None
        ),
        "coarse_macro_f1_supported": lambda value: (
            value["coarse"]["macro_f1_supported"]
            if value["coarse"] is not None
            else None
        ),
        "coarse_macro_f1_all_classes": lambda value: (
            value["coarse"]["macro_f1_all_classes"]
            if value["coarse"] is not None
            else None
        ),
        "coarse_balanced_accuracy": lambda value: (
            value["coarse"]["balanced_accuracy"]
            if value["coarse"] is not None
            else None
        ),
        "coarse_mcc": lambda value: (
            value["coarse"]["mcc"] if value["coarse"] is not None else None
        ),
        "fine_macro_f1_supported": lambda value: (
            value["fine"]["macro_f1_supported"]
            if value["fine"] is not None
            else None
        ),
        "fine_macro_f1_all_classes": lambda value: (
            value["fine"]["macro_f1_all_classes"]
            if value["fine"] is not None
            else None
        ),
        "primary_macro_f1_all_classes": lambda value: (
            value["primary_fine"]["macro_f1_all_classes"]
            if value["primary_fine"] is not None
            and value["primary_fine"]["status"] == "ok"
            else None
        ),
        "hierarchical_accuracy": lambda value: (
            value["hierarchy"]["hierarchical_accuracy"]
            if value["hierarchy"] is not None
            else None
        ),
    }
    summary: dict[str, Any] = {
        "folds": len(fold_results),
        "metrics": {},
    }
    for name, selector in selectors.items():
        values = [
            selector(result)
            for result in fold_results
            if selector(result) is not None
        ]
        if not values:
            summary["metrics"][name] = {
                "status": "not_evaluable",
                "valid_folds": 0,
            }
            continue
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        half_width = (
            1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
        )
        summary["metrics"][name] = {
            "mean": mean,
            "std": std,
            "ci95_normal_lower": mean - half_width,
            "ci95_normal_upper": mean + half_width,
            "valid_folds": len(values),
            "values": values,
        }
    return summary


def run_single_fold(
    fold_name: str,
    split_frames: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
    device: torch.device,
    amp_dtype: torch.dtype | None,
    run_dir: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    fold_seed = stable_int_seed(config["seed"], fold_name)
    seed_everything(fold_seed, bool(config["deterministic"]))
    optimization_frames = dict(split_frames)
    if config["train_modality"] == "WLC":
        optimization_train = split_frames["train"].loc[
            split_frames["train"]["modality"].eq("WLC")
        ].copy()
    else:
        optimization_train = split_frames["train"].copy()
    if optimization_train.empty:
        raise ValueError(
            f"train_modality={config['train_modality']} produced no training row."
        )
    optimization_frames["train"] = optimization_train
    optimization_audit = {
        "train_modality": str(config["train_modality"]),
        "protocol_train_rows": len(split_frames["train"]),
        "optimization_train_rows": len(optimization_train),
        "optimization_train_patients": int(
            optimization_train["pid"].nunique()
        ),
        "coarse_counts": optimization_train["class"].value_counts().to_dict(),
        "fine_counts": (
            optimization_train.loc[optimization_train["fine_id"] >= 0]
            ["subclass"]
            .value_counts()
            .to_dict()
        ),
        "optimization_train_semantic_sha256": science.semantic_fingerprint(
            optimization_train[
                [
                    "image_stem",
                    "pid",
                    "binary_id",
                    "coarse_id",
                    "fine_id",
                    "modality",
                ]
            ]
            .sort_values("image_stem")
            .astype(str)
            .to_dict(orient="records")
        ),
    }
    write_json(
        run_dir / "reports" / f"{fold_name}_training_subset.json",
        optimization_audit,
    )
    optimization_train[
        [
            "filename",
            "pid",
            "binary_id",
            "coarse_id",
            "fine_id",
            "modality",
        ]
    ].to_csv(
        run_dir / "splits" / fold_name / "optimization_train.csv",
        index=False,
    )
    loaders, _ = build_dataloaders(
        optimization_frames, config, device, fold_seed
    )
    model = HierarchicalCystoModel(config).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if config.get("channels_last", False):
        model = model.to(memory_format=torch.channels_last)
    if config["torch_compile"]:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch_compile requested but torch.compile is absent.")
        model.encoder = torch.compile(
            model.encoder, mode=str(config["torch_compile_mode"])
        )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    model_info = {
        "model_name": config["model_name"],
        "pretrained": config["pretrained"],
        "feature_dim": model.feature_dim,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
    }
    write_json(
        run_dir / "models" / f"{fold_name}_model_info.json", model_info
    )
    logger.info(
        "Model ready: %s parameters=%d trainable=%d device=%s",
        config["model_name"],
        total_parameters,
        trainable_parameters,
        device,
    )

    checkpoint_dir = run_dir / "checkpoints" / fold_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    temporary_checkpoint_owner: Any | None = None
    if config["checkpoint_backend"] == "huggingface":
        temporary_checkpoint_owner = tempfile.TemporaryDirectory(
            prefix=f"cystods-{fold_name}-checkpoint-"
        )
        checkpoint_work_dir = Path(temporary_checkpoint_owner.name)
    else:
        checkpoint_work_dir = checkpoint_dir
    model, history, fine_counts, fine_patient_counts = train_model(
        model,
        loaders,
        split_frames,
        optimization_train,
        config,
        device,
        amp_dtype,
        checkpoint_dir,
        logger,
        checkpoint_work_dir,
    )
    # Release train/validation worker pools before constructing the three
    # deterministic final-evaluation loaders.
    del loaders
    history.to_csv(run_dir / "logs" / f"{fold_name}_history.csv", index=False)
    performance = {
        "train_samples_per_second_mean": float(
            history["train_samples_per_second"].mean()
        ),
        "train_samples_per_second_max": float(
            history["train_samples_per_second"].max()
        ),
        "train_seconds_total": float(history["train_seconds"].sum()),
        "epochs_completed": len(history),
        "batch_size": int(config["batch_size"]),
        "gradient_accumulation_steps": int(
            config["gradient_accumulation_steps"]
        ),
        "num_workers": int(config["num_workers"]),
        "precision": config["precision"],
        "channels_last": bool(config["channels_last"]),
        "torch_compile": bool(config["torch_compile"]),
        "cuda_peak_memory_mib": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if device.type == "cuda"
            else None
        ),
        "cuda_peak_reserved_mib": (
            float(torch.cuda.max_memory_reserved(device) / (1024**2))
            if device.type == "cuda"
            else None
        ),
        "cuda_total_memory_mib": (
            float(
                torch.cuda.get_device_properties(device).total_memory
                / (1024**2)
            )
            if device.type == "cuda"
            else None
        ),
    }
    write_json(
        run_dir / "metrics" / fold_name / "performance.json",
        performance,
    )
    best_checkpoint = torch.load(
        checkpoint_work_dir / "best_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    selected_fine_tau = float(
        best_checkpoint["fine_inference_prior_tau"]
    )
    portable_checkpoint = {
        "model_state_dict": _canonical_model_state_dict(
            best_checkpoint["model_state_dict"]
        ),
        "config": best_checkpoint["config"],
        "binary_names": BINARY_NAMES,
        "coarse_names": COARSE_NAMES,
        "fine_names": FINE_NAMES,
        "fine_parent_id": FINE_PARENT_ID,
        "fine_train_counts": fine_counts,
        "fine_train_patient_counts": fine_patient_counts,
        "fine_prior_audit": best_checkpoint["fine_prior_audit"],
        "fine_inference_prior_tau": selected_fine_tau,
        "data_split_fingerprint": best_checkpoint[
            "data_split_fingerprint"
        ],
        "model_info": model_info,
    }
    if config["checkpoint_backend"] == "huggingface":
        portable_checkpoint_path = (
            checkpoint_work_dir / "publish" / "best_model.pt"
        )
    else:
        portable_checkpoint_path = (
            run_dir / "models" / f"{fold_name}_best_model.pt"
        )
    portable_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(portable_checkpoint, portable_checkpoint_path)
    del portable_checkpoint
    del best_checkpoint

    metrics_dir = run_dir / "metrics" / fold_name
    prediction_dir = run_dir / "predictions" / fold_name
    metrics_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    split_predictions: dict[str, pd.DataFrame] = {}
    split_metrics: dict[str, Any] = {}
    split_losses: dict[str, Any] = {}
    active_tasks = active_tasks_from_config(config)
    evaluation_fine_loss = (
        FineLongTailLoss(
            active_fine_loss_name(config),
            fine_counts,
            fine_patient_counts,
            config,
        ).to(device)
        if "fine" in active_tasks
        else None
    )
    retain_features = bool(
        config["evaluate_roi_level"]
        and "attention" in set(config["roi_aggregations"])
    )
    for split_name in ("train", "val", "test"):
        eval_loader = make_deterministic_eval_loader(
            split_frames[split_name],
            config,
            device,
            stable_int_seed(fold_seed, split_name, "eval"),
        )
        metrics, predictions, losses = evaluate_model(
            model,
            eval_loader,
            evaluation_fine_loss,
            fine_counts,
            fine_patient_counts,
            config,
            device,
            amp_dtype,
            include_features=retain_features,
            fine_prior_tau=selected_fine_tau,
        )
        split_predictions[split_name] = predictions
        split_metrics[split_name] = metrics
        split_losses[split_name] = losses
        serialize_prediction_frame(
            predictions,
            prediction_dir / f"{split_name}_image_predictions.csv",
            include_features=False,
        )
        write_json(metrics_dir / f"{split_name}_metrics.json", metrics)
        write_json(metrics_dir / f"{split_name}_losses.json", losses)
        logger.info(
            "final-eval split=%s n=%d binary_f1=%s coarse_macro_f1=%s "
            "fine_macro_f1_all=%s",
            split_name,
            len(predictions),
            (
                f"{metrics['binary']['f1']:.5f}"
                if metrics["binary"] is not None
                else "inactive"
            ),
            (
                f"{metrics['coarse']['macro_f1_supported']:.5f}"
                if metrics["coarse"] is not None
                else "inactive"
            ),
            (
                f"{metrics['fine']['macro_f1_all_classes']:.5f}"
                if metrics["fine"] is not None
                else "inactive"
            ),
        )

    test_predictions = split_predictions["test"]
    test_metrics = split_metrics["test"]
    if (
        test_metrics["rare_class_collapse"] is not None
        and config["scientific_gate_mode"] == "enforce"
    ):
        if config["evaluation_scope"] == "final_cv":
            science.enforce_rare_class_collapse_gate(
                test_metrics["rare_class_collapse"]
            )
        elif test_metrics["rare_class_collapse"]["status"] == "failed":
            logger.warning(
                "Scientific gate rejected this development candidate; "
                "violations=%s",
                [
                    row["class_name"]
                    for row in test_metrics["rare_class_collapse"][
                        "violations"
                    ]
                ],
            )
    bootstrap = patient_bootstrap_intervals(
        test_predictions,
        fine_counts,
        fine_patient_counts,
        config,
        stable_int_seed(fold_seed, "bootstrap"),
    )
    write_json(metrics_dir / "patient_bootstrap_ci.json", bootstrap)

    wlc_metrics = None
    if config["evaluate_wlc_only"]:
        wlc = test_predictions[test_predictions["modality"] == "WLC"].copy()
        if wlc.empty:
            raise ValueError("WLC-only evaluation requested but has zero rows.")
        wlc_metrics = compute_metrics_bundle(
            wlc,
            fine_counts,
            fine_patient_counts,
            config,
        )
        write_json(metrics_dir / "wlc_only_metrics.json", wlc_metrics)
        serialize_prediction_frame(
            wlc,
            prediction_dir / "test_wlc_only_predictions.csv",
            include_features=False,
        )

    roi_metrics = None
    if config["evaluate_roi_level"]:
        roi_metrics = run_roi_evaluation(
            split_predictions,
            config,
            device,
            run_dir,
            fold_name,
            logger,
        )

    paired = None
    if config["paired_baseline_predictions_csv"] is not None:
        if "binary" not in active_tasks:
            raise ValueError(
                "Paired McNemar evaluation requires an active binary head."
            )
        paired = paired_mcnemar_test(
            test_predictions,
            Path(config["paired_baseline_predictions_csv"]),
            float(config["binary_decision_threshold"]),
        )
        write_json(metrics_dir / "paired_mcnemar.json", paired)

    external = None
    if config["external_validation_enabled"]:
        if "binary" not in active_tasks:
            raise ValueError(
                "External binary validation requires an active binary head."
            )
        external = evaluate_external_binary(
            model,
            config,
            device,
            amp_dtype,
            run_dir,
            fold_name,
        )

    export_fold_visualizations(
        history,
        test_predictions,
        test_metrics,
        split_frames,
        config,
        run_dir,
        fold_name,
    )
    split_summary = json.loads(
        (
            run_dir / "splits" / fold_name / "summary.json"
        ).read_text(encoding="utf-8")
    )
    build_fold_report(
        fold_name,
        config,
        split_summary,
        test_metrics,
        bootstrap,
        wlc_metrics,
        roi_metrics,
        paired,
        external,
        history,
        run_dir / "reports" / f"{fold_name}_report.md",
    )
    checkpoint_receipt = None
    checkpoint_receipt_json = None
    if config["checkpoint_backend"] == "huggingface":
        hub_config = _hf_checkpoint_config(
            config,
            _hf_checkpoint_path_in_repo(config, fold_name),
        )
        receipt_dir = (
            run_dir / "reports" / "hf_checkpoints" / fold_name
        )
        checkpoint_receipt = checkpoint_hub.publish_best_checkpoint(
            portable_checkpoint_path,
            receipt_dir,
            hub_config,
        )
        checkpoint_receipt_json = str(
            receipt_dir / "hf_checkpoint_receipt.json"
        )
        if temporary_checkpoint_owner is None:
            raise RuntimeError("Temporary checkpoint lifecycle was not created.")
        temporary_checkpoint_owner.cleanup()
        temporary_checkpoint_owner = None
        local_models = sorted(run_dir.rglob("*.pt"))
        if local_models:
            raise RuntimeError(
                "Remote-only checkpoint policy violation; local model files "
                f"remain: {[str(path) for path in local_models]}"
            )
        logger.info(
            "Published verified best_model.pt to Hugging Face commit=%s path=%s",
            checkpoint_receipt["commit_oid"],
            checkpoint_receipt["path_in_repo"],
        )
    # Features are deliberately kept only in memory for the real attention MIL
    # stage. The durable image prediction CSV stays compact.
    return {
        "fold_name": fold_name,
        "test_metrics": test_metrics,
        "bootstrap": bootstrap,
        "wlc_metrics": wlc_metrics,
        "roi_metrics": roi_metrics,
        "external_metrics": external,
        "model_info": model_info,
        "history_rows": len(history),
        "hf_checkpoint_receipt": checkpoint_receipt,
        "hf_checkpoint_receipt_json": checkpoint_receipt_json,
    }


def main(
    config: Mapping[str, Any] | None = None,
    required_source_files: Sequence[Path | str] | None = None,
) -> Path:
    config = normalize_core_config(CONFIG if config is None else config)
    validate_config(config)
    if config["checkpoint_backend"] == "huggingface":
        _hf_checkpoint_config(
            config,
            f"{str(config['hf_path_prefix']).rstrip('/')}/"
            "preflight/best_model.pt",
        )
    if required_source_files is None:
        try:
            core_source = Path(__file__).resolve()
        except NameError as exc:
            raise RuntimeError(
                "Core provenance requires explicit required_source_files "
                "when __file__ is unavailable."
            ) from exc
        science_src = (
            core_source.with_name("science.py")
            if core_source.with_name("science.py").exists()
            else core_source.with_name("cystods_science.py")
        )
        hf_src = (
            core_source.with_name("hf.py")
            if core_source.with_name("hf.py").exists()
            else core_source.with_name("cystods_hf.py")
        )
        required_source_files = (
            core_source,
            science_src,
            hf_src,
        )
    validated_sources = validate_source_files(required_source_files)
    run_dir = make_run_directory(config)
    logger = setup_logger(run_dir / "logs" / "training.log")
    status_path = run_dir / "run_status.json"
    started_utc = utc_now_iso()
    write_json(
        status_path,
        {
            "status": "running",
            "started_utc": started_utc,
            "run_dir": run_dir,
        },
    )
    try:
        device = resolve_device(config)
        precision_name, amp_dtype = resolve_precision(config, device)
        torch.set_num_threads(int(config["num_cpu_threads"]))
        torch.set_float32_matmul_precision(
            str(config["float32_matmul_precision"])
        )
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = bool(
                config["enable_tf32"]
            )
            torch.backends.cudnn.allow_tf32 = bool(config["enable_tf32"])
        seed_everything(
            int(config["seed"]), bool(config["deterministic"])
        )
        system_info = collect_system_info(
            config, device, precision_name
        )
        write_json(run_dir / "config.json", config)
        snapshot_source_files(run_dir, validated_sources)
        write_json(run_dir / "system" / "environment.json", system_info)
        write_json(
            run_dir / "system" / "dependency_audit.json",
            DEPENDENCY_AUDIT,
        )
        write_json(
            run_dir / "reports" / "taxonomy.json",
            {
                "binary_names": BINARY_NAMES,
                "coarse_names": COARSE_NAMES,
                "fine_names": FINE_NAMES,
                "fine_by_parent": FINE_BY_PARENT,
                "normal_mucosa_fine_policy": (
                    "fine_id=-1; excluded from the 22-subclass loss/metrics"
                ),
            },
        )
        with (run_dir / "system" / "pip_freeze.txt").open(
            "w", encoding="utf-8"
        ) as handle:
            _subprocess.run(
                [_sys.executable, "-m", "pip", "freeze"],
                check=True,
                stdout=handle,
                text=True,
            )
        logger.info(
            "Run started: profile=%s device=%s precision=%s output=%s",
            config["run_profile"],
            device,
            precision_name,
            run_dir,
        )

        manifest, audit = load_and_validate_manifest(
            config, run_dir, logger
        )
        expected_dataset_hash = config["expected_dataset_semantic_sha256"]
        if expected_dataset_hash is not None:
            active_dataset_fingerprint = json.loads(
                (run_dir / "system" / "dataset_fingerprint.json").read_text(
                    encoding="utf-8"
                )
            )
            active_dataset_hash = active_dataset_fingerprint[
                "semantic_manifest_sha256"
            ]
            if active_dataset_hash != expected_dataset_hash:
                raise ValueError(
                    "Active dataset differs from the Stage 00 audited "
                    "dataset: expected semantic SHA-256="
                    f"{expected_dataset_hash}, actual={active_dataset_hash}."
                )
        protocols = build_all_protocol_splits(
            manifest, config, run_dir, logger
        )
        fold_results = []
        for fold_name, split_frames, _ in protocols:
            logger.info("Starting protocol unit: %s", fold_name)
            fold_results.append(
                run_single_fold(
                    fold_name,
                    split_frames,
                    config,
                    device,
                    amp_dtype,
                    run_dir,
                    logger,
                )
            )
        if len(fold_results) > 1:
            cv_summary = aggregate_cross_validation_metrics(
                [result["test_metrics"] for result in fold_results]
            )
            write_json(
                run_dir / "metrics" / "cross_validation_summary.json",
                cv_summary,
            )
        else:
            cv_summary = None
        write_json(
            run_dir / "reports" / "run_summary.json",
            {
                "data_audit": audit,
                "fold_results": fold_results,
                "cross_validation": cv_summary,
            },
        )
        write_json(
            status_path,
            {
                "status": "completed",
                "started_utc": started_utc,
                "completed_utc": utc_now_iso(),
                "run_dir": run_dir,
                "folds_completed": len(fold_results),
            },
        )
        logger.info(
            "Run completed successfully: run_dir=%s folds=%d",
            run_dir,
            len(fold_results),
        )
        return run_dir
    except Exception as exc:
        logger.exception("Run failed with a real error; no fallback was used.")
        write_json(
            status_path,
            {
                "status": "failed",
                "failed_utc": utc_now_iso(),
                "run_dir": run_dir,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        logger.error(
            "FINAL_STATUS failed run_dir=%s error_type=%s",
            run_dir,
            type(exc).__name__,
        )
        raise
    finally:
        close_logger(logger)
        write_artifact_manifest(run_dir)


def _validate_stage_config_keys(
    config: Mapping[str, Any],
    expected_keys: set[str],
) -> None:
    missing = expected_keys - set(config)
    unknown = set(config) - expected_keys
    if missing or unknown:
        raise KeyError(
            "Stage config schema mismatch: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if config["schema_version"] != "cystods.stage.v2":
        raise ValueError("Stage schema_version must be cystods.stage.v2.")
    if config["run_profile"] not in {"research", "smoke"}:
        raise ValueError("Stage run_profile must be research or smoke.")


def _complete_stage_source_files(
    required_source_files: Sequence[Path | str],
) -> tuple[Path, ...]:
    try:
        core_source = Path(__file__).resolve()
    except NameError as exc:
        raise RuntimeError(
            "Stage execution requires an on-disk cystods_core.py source."
        ) from exc
    science_src = (
        core_source.with_name("science.py")
        if core_source.with_name("science.py").exists()
        else core_source.with_name("cystods_science.py")
    )
    hf_src = (
        core_source.with_name("hf.py")
        if core_source.with_name("hf.py").exists()
        else core_source.with_name("cystods_hf.py")
    )
    candidates = [
        *required_source_files,
        core_source,
        science_src,
        hf_src,
    ]
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = Path(candidate).expanduser().resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return validate_source_files(unique)


def _write_stage_system_artifacts(
    run_dir: Path,
    stage_config: Mapping[str, Any],
    source_files: Sequence[Path | str],
) -> None:
    write_json(run_dir / "config.json", stage_config)
    snapshot_source_files(run_dir, source_files)
    write_json(
        run_dir / "system" / "dependency_audit.json",
        DEPENDENCY_AUDIT,
    )
    write_json(
        run_dir / "system" / "environment.json",
        {
            "python": _sys.version,
            "executable": _sys.executable,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "timm": timm.__version__,
        },
    )
    with (run_dir / "system" / "pip_freeze.txt").open(
        "w", encoding="utf-8"
    ) as handle:
        _subprocess.run(
            [_sys.executable, "-m", "pip", "freeze"],
            check=True,
            stdout=handle,
            text=True,
        )


def run_protocol_stage(
    stage_config: Mapping[str, Any],
    required_source_files: Sequence[Path | str],
) -> Path:
    expected_keys = {
        "schema_version",
        "stage_name",
        "study_id",
        "run_profile",
        "data_root",
        "result_root",
        "protocol_config",
    }
    _validate_stage_config_keys(stage_config, expected_keys)
    source_files = _complete_stage_source_files(required_source_files)
    config = normalize_core_config(stage_config["protocol_config"])
    config.update(
        {
            "stage_name": str(stage_config["stage_name"]),
            "study_id": str(stage_config["study_id"]),
            "run_profile": str(stage_config["run_profile"]),
            "data_root": Path(stage_config["data_root"]).resolve(),
            "result_root": Path(stage_config["result_root"]).resolve(),
            "experiment_name": str(stage_config["stage_name"]),
            "protocol_manifest_dir": None,
            "protocol_reference_sha256": None,
            "protocol_role": None,
            "evaluation_scope": "development",
            "suite_trial_id": None,
        }
    )
    data_root = Path(config["data_root"])
    config["metadata_csv"] = data_root / "cystods.csv"
    config["image_dir"] = data_root / "images"
    config["segmentation_dir"] = data_root / "segmentations"
    validate_config(config)
    run_dir = make_run_directory(config)
    logger = setup_logger(run_dir / "logs" / "training.log")
    status_path = run_dir / "run_status.json"
    started_utc = utc_now_iso()
    write_json(
        status_path,
        {
            "status": "running",
            "started_utc": started_utc,
            "stage_name": stage_config["stage_name"],
            "run_dir": run_dir,
        },
    )
    try:
        _write_stage_system_artifacts(run_dir, stage_config, source_files)
        logger.info(
            "Protocol stage starts profile=%s output=%s",
            config["run_profile"],
            run_dir,
        )
        manifest, audit = load_and_validate_manifest(
            config,
            run_dir,
            logger,
        )
        fixed_config = dict(config)
        fixed_config.update(
            {
                "protocol": "holdout",
                "fixed_split_pids": (
                    config["fixed_split_pids"]
                    if config["run_profile"] == "smoke"
                    else None
                ),
                "cv_run_fold_indices": None,
            }
        )
        fixed_units = build_all_protocol_splits(
            manifest,
            fixed_config,
            run_dir,
            logger,
        )
        if len(fixed_units) != 1 or fixed_units[0][0] != "holdout":
            raise RuntimeError(
                "Protocol stage must create exactly one fixed holdout unit."
            )
        _, fixed_frames, _ = fixed_units[0]
        train_patient_counts = (
            fixed_frames["train"]
            .loc[fixed_frames["train"]["fine_id"] >= 0]
            .groupby("fine_id")["pid"]
            .nunique()
        )
        primary_support_threshold = (
            1
            if config["run_profile"] == "smoke"
            else int(config["primary_fine_min_train_patients"])
        )
        primary_ids = [
            class_id
            for class_id in range(len(FINE_NAMES))
            if int(train_patient_counts.get(class_id, 0))
            >= primary_support_threshold
        ]
        if not primary_ids:
            raise ValueError(
                "Frozen primary fine taxonomy is empty at the configured "
                "patient-support threshold."
            )
        split_summaries = {}
        for unit_dir in sorted((run_dir / "splits").iterdir()):
            summary_path = unit_dir / "summary.json"
            if unit_dir.is_dir() and summary_path.is_file():
                split_summaries[unit_dir.name] = json.loads(
                    summary_path.read_text(encoding="utf-8")
                )
        dataset_fingerprint = json.loads(
            (
                run_dir / "system" / "dataset_fingerprint.json"
            ).read_text(encoding="utf-8")
        )
        protocol_manifest = {
            "schema_version": "cystods.protocol.v2",
            "study_id": str(stage_config["study_id"]),
            "created_utc": utc_now_iso(),
            "run_profile": config["run_profile"],
            "dataset_fingerprint": dataset_fingerprint,
            "primary_fine_class_ids": primary_ids,
            "primary_fine_class_names": [
                FINE_NAMES[index] for index in primary_ids
            ],
            "primary_fine_min_train_patients": primary_support_threshold,
            "primary_taxonomy_policy": (
                "functional_smoke_train_support_ge_1"
                if config["run_profile"] == "smoke"
                else "preregistered_train_patient_support"
            ),
            "roles": {
                "fixed_holdout": {
                    "protocol": "holdout",
                    "units": ["holdout"],
                },
                "smoke_holdout": {
                    "protocol": "holdout",
                    "units": ["holdout"],
                },
            },
            "split_summaries": split_summaries,
            "data_audit_sha256": sha256_file(
                run_dir / "reports" / "data_audit.json"
            ),
        }
        protocol_path = run_dir / "protocol_manifest.json"
        write_json(protocol_path, protocol_manifest)
        protocol_hash = sha256_file(protocol_path)
        write_json(
            run_dir / "reports" / "protocol_reference.json",
            {
                "protocol_manifest": str(protocol_path),
                "protocol_sha256": protocol_hash,
                "downstream_environment": {
                    "CYSTODS_PROTOCOL_RUN_DIR": str(run_dir),
                    "CYSTODS_EXPECTED_PROTOCOL_SHA256": protocol_hash,
                },
            },
        )
        write_json(
            run_dir / "reports" / "run_summary.json",
            {
                "data_audit": audit,
                "fixed_holdout_units": ["holdout"],
                "protocol_sha256": protocol_hash,
            },
        )
        write_json(
            status_path,
            {
                "status": "completed",
                "started_utc": started_utc,
                "completed_utc": utc_now_iso(),
                "stage_name": stage_config["stage_name"],
                "run_dir": run_dir,
                "protocol_sha256": protocol_hash,
            },
        )
        logger.info(
            "Protocol stage completed successfully: protocol_sha256=%s "
            "run_dir=%s",
            protocol_hash,
            run_dir,
        )
        return run_dir
    except Exception as exc:
        logger.exception("Protocol stage failed; no fallback was used.")
        write_json(
            status_path,
            {
                "status": "failed",
                "started_utc": started_utc,
                "failed_utc": utc_now_iso(),
                "stage_name": stage_config["stage_name"],
                "run_dir": run_dir,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        logger.error(
            "FINAL_STATUS failed error_type=%s run_dir=%s",
            type(exc).__name__,
            run_dir,
        )
        raise
    finally:
        close_logger(logger)
        write_artifact_manifest(run_dir)


def find_latest_completed_protocol_run(
    result_root: Path | str | None,
    run_profile: str | None = None,
) -> tuple[Path | None, str | None]:
    if result_root is None:
        return None, None
    result_path = Path(result_root).expanduser().resolve()
    search_dirs = [result_path, result_path.parent, Path.cwd() / "result"]
    candidates = []
    seen = set()
    for base in search_dirs:
        if not base.is_dir():
            continue
        for manifest_file in base.rglob("protocol_manifest.json"):
            path = manifest_file.parent
            if not path.is_dir() or path in seen:
                continue
            seen.add(path)
            status_file = path / "run_status.json"
            if not status_file.is_file():
                continue
            try:
                status = json.loads(status_file.read_text(encoding="utf-8"))
                if status.get("status") != "completed":
                    continue
                manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
                if run_profile and manifest.get("run_profile") != run_profile:
                    continue
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ):
                continue
            mtime = manifest_file.stat().st_mtime
            sha256 = sha256_file(manifest_file)
            candidates.append((mtime, path, sha256))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def _load_and_validate_protocol_binding(
    stage_config: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], str]:
    raw_protocol_dir = stage_config.get("protocol_run_dir")
    expected_hash = stage_config.get("expected_protocol_sha256")
    if raw_protocol_dir is None:
        auto_dir, auto_sha = find_latest_completed_protocol_run(
            stage_config.get("result_root"), stage_config.get("run_profile")
        )
        if auto_dir is None:
            raise RuntimeError(
                "No completed Stage 00 protocol run found. "
                "Run stage_00_prepare_protocol.py first or set CYSTODS_PROTOCOL_RUN_DIR."
            )
        protocol_run_dir = auto_dir
        if expected_hash is None:
            expected_hash = auto_sha
    else:
        protocol_run_dir = Path(raw_protocol_dir).expanduser().resolve()

    if not protocol_run_dir.is_dir():
        raise FileNotFoundError(
            f"Protocol run directory not found: {protocol_run_dir}"
        )
    status_path = protocol_run_dir / "run_status.json"
    protocol_path = protocol_run_dir / "protocol_manifest.json"
    if not status_path.is_file() or not protocol_path.is_file():
        raise FileNotFoundError(
            "Protocol run must contain run_status.json and "
            "protocol_manifest.json."
        )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        raise ValueError("Protocol run is not completed.")
    protocol_manifest = json.loads(
        protocol_path.read_text(encoding="utf-8")
    )
    if protocol_manifest.get("schema_version") != "cystods.protocol.v2":
        raise ValueError("Unsupported protocol manifest schema.")
    if protocol_manifest.get("study_id") != stage_config["study_id"]:
        raise ValueError("Protocol study_id differs from stage study_id.")
    if protocol_manifest.get("run_profile") != stage_config["run_profile"]:
        raise ValueError("Protocol run_profile differs from stage run_profile.")
    protocol_hash = sha256_file(protocol_path)
    if expected_hash is None:
        if stage_config["run_profile"] == "research":
            raise ValueError(
                "Research stages require expected_protocol_sha256. Copy it "
                "from Stage 00 reports/protocol_reference.json."
            )
    elif str(expected_hash) != protocol_hash:
        raise ValueError(
            "Protocol SHA-256 mismatch: "
            f"expected={expected_hash}, actual={protocol_hash}"
        )
    role = str(stage_config["protocol_role"])
    roles = protocol_manifest.get("roles")
    if not isinstance(roles, Mapping):
        raise TypeError("Protocol manifest roles must be a mapping.")
    # Final CV is generated independently over the complete audited dataset.
    # Stage 00 still binds its dataset identity and frozen primary taxonomy;
    # it intentionally does not pre-create any CV fold.
    manifest_role = "fixed_holdout" if role == "final_cv" else role
    if manifest_role not in roles:
        raise ValueError(
            f"Protocol manifest does not define role={manifest_role}."
        )
    role_units = roles[manifest_role].get("units")
    if not isinstance(role_units, list) or not role_units:
        raise ValueError(f"Protocol role={role} contains no split unit.")
    fold_ids = stage_config["fold_ids"]
    if role != "final_cv" and fold_ids is not None:
        raise ValueError("fold_ids are valid only for protocol_role=final_cv.")
    return protocol_run_dir, protocol_manifest, protocol_hash


def run_training_suite(
    stage_config: Mapping[str, Any],
    required_source_files: Sequence[Path | str],
) -> Path:
    expected_keys = {
        "schema_version",
        "stage_name",
        "study_id",
        "run_profile",
        "data_root",
        "result_root",
        "protocol_run_dir",
        "protocol_role",
        "expected_protocol_sha256",
        "seeds",
        "fold_ids",
        "base_config",
        "trials",
        "evaluation_scope",
    }
    _validate_stage_config_keys(stage_config, expected_keys)
    source_files = _complete_stage_source_files(required_source_files)
    protocol_run_dir, protocol_manifest, protocol_hash = (
        _load_and_validate_protocol_binding(stage_config)
    )
    role = str(stage_config["protocol_role"])
    scope = str(stage_config["evaluation_scope"])
    if role in {"fixed_holdout", "smoke_holdout"} and scope != "development":
        raise ValueError(
            f"{role} requires evaluation_scope=development."
        )
    if role == "final_cv" and scope != "final_cv":
        raise ValueError("final_cv requires evaluation_scope=final_cv.")
    seeds = tuple(stage_config["seeds"])
    if not seeds or any(
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        for seed in seeds
    ):
        raise ValueError("seeds must be a non-empty integer sequence.")
    trials = tuple(stage_config["trials"])
    if not trials:
        raise ValueError("Training suite must contain at least one trial.")
    trial_ids = [str(trial.get("experiment_id", "")) for trial in trials]
    if (
        any(not trial_id for trial_id in trial_ids)
        or len(set(trial_ids)) != len(trial_ids)
    ):
        raise ValueError("Trial experiment_id values must be non-empty and unique.")
    base_config = normalize_core_config(stage_config["base_config"])
    validate_config(base_config)
    if base_config["checkpoint_backend"] == "huggingface":
        _hf_checkpoint_config(
            base_config,
            f"{str(base_config['hf_path_prefix']).rstrip('/')}/"
            "preflight/best_model.pt",
        )
    parent_run_dir = make_run_directory(
        {
            "result_root": Path(stage_config["result_root"]).resolve(),
            "experiment_name": str(stage_config["stage_name"]),
            "run_profile": str(stage_config["run_profile"]),
        }
    )
    logger = setup_logger(parent_run_dir / "logs" / "training.log")
    status_path = parent_run_dir / "run_status.json"
    started_utc = utc_now_iso()
    write_json(
        status_path,
        {
            "status": "running",
            "started_utc": started_utc,
            "stage_name": stage_config["stage_name"],
            "run_dir": parent_run_dir,
            "protocol_sha256": protocol_hash,
        },
    )
    try:
        _write_stage_system_artifacts(
            parent_run_dir,
            stage_config,
            source_files,
        )
        logger.info(
            "Training suite starts trials=%d seeds=%d role=%s scope=%s",
            len(trials),
            len(seeds),
            role,
            scope,
        )
        child_root = parent_run_dir / "runs"
        child_root.mkdir(parents=True, exist_ok=True)
        write_json(
            parent_run_dir / "system" / "child_run_root.json",
            {
                "child_run_root": str(child_root),
                "reason": (
                    "Child runs are siblings of the suite artifact so later "
                    "sibling verification reports cannot mutate the sealed "
                    "suite file set."
                ),
            },
        )
        child_results = []
        suite_split_fingerprints: dict[str, str] | None = None
        fixed_primary_ids = protocol_manifest["primary_fine_class_ids"]
        protocol_name = "cross_validation" if role == "final_cv" else "holdout"
        for trial_index, trial in enumerate(trials):
            if set(trial) != {"experiment_id", "task_mode", "overrides"}:
                raise KeyError(
                    f"Trial {trial_index} has an invalid schema: "
                    f"{sorted(trial)}"
                )
            overrides = dict(trial["overrides"])
            unknown_overrides = set(overrides) - set(BASE_CONFIG)
            if unknown_overrides:
                raise KeyError(
                    f"Trial {trial['experiment_id']} contains unknown "
                    f"overrides: {sorted(unknown_overrides)}"
                )
            for seed in seeds:
                experiment_id = str(trial["experiment_id"])
                child_config = dict(base_config)
                child_config.update(overrides)
                child_config.update(
                    {
                        "schema_version": "cystods.core.v2",
                        "stage_name": str(stage_config["stage_name"]),
                        "study_id": str(stage_config["study_id"]),
                        "run_profile": str(stage_config["run_profile"]),
                        "data_root": Path(stage_config["data_root"]).resolve(),
                        "result_root": child_root,
                        "experiment_name": (
                            f"{experiment_id}_seed_{int(seed)}"
                        ),
                        "task_mode": str(trial["task_mode"]),
                        "seed": int(seed),
                        "protocol": protocol_name,
                        "fixed_split_pids": None,
                        "protocol_manifest_dir": (
                            None
                            if role == "final_cv"
                            else protocol_run_dir / "splits"
                        ),
                        "protocol_reference_sha256": protocol_hash,
                        "expected_dataset_semantic_sha256": (
                            protocol_manifest["dataset_fingerprint"][
                                "semantic_manifest_sha256"
                            ]
                        ),
                        "protocol_role": role,
                        "evaluation_scope": scope,
                        "suite_trial_id": experiment_id,
                        "cv_run_fold_indices": (
                            list(stage_config["fold_ids"])
                            if role == "final_cv"
                            and stage_config["fold_ids"] is not None
                            else None
                        ),
                        "fixed_primary_fine_class_ids": list(
                            fixed_primary_ids
                        ),
                        # Stage 00 already performed the expensive full audit.
                        # Downstream runs revalidate semantic identity and the
                        # frozen split hashes without rehashing every image.
                        "verify_all_image_decodes": False,
                        "verify_segmentation_inventory": False,
                        "dataset_fingerprint_mode": "semantic",
                        "verify_exact_duplicate_images": False,
                        "external_validation_enabled": False,
                    }
                )
                data_root = Path(child_config["data_root"])
                child_config["metadata_csv"] = data_root / "cystods.csv"
                child_config["image_dir"] = data_root / "images"
                child_config["segmentation_dir"] = (
                    data_root / "segmentations"
                )
                validate_config(child_config)
                logger.info(
                    "Starting trial=%s seed=%d task_mode=%s",
                    experiment_id,
                    int(seed),
                    child_config["task_mode"],
                )
                child_run_dir = main(
                    child_config,
                    required_source_files=source_files,
                )
                child_status = json.loads(
                    (child_run_dir / "run_status.json").read_text(
                        encoding="utf-8"
                    )
                )
                if child_status.get("status") != "completed":
                    raise RuntimeError(
                        f"Child run did not complete: {child_run_dir}"
                    )
                child_summary = json.loads(
                    (
                        child_run_dir / "reports" / "run_summary.json"
                    ).read_text(encoding="utf-8")
                )
                fold_metrics = [
                    result["test_metrics"]
                    for result in child_summary["fold_results"]
                ]
                current_split_fingerprints = {
                    summary_path.parent.name: json.loads(
                        summary_path.read_text(encoding="utf-8")
                    )["data_split_fingerprint"]
                    for summary_path in sorted(
                        (child_run_dir / "splits").glob("*/summary.json")
                    )
                }
                if not current_split_fingerprints:
                    raise RuntimeError(
                        f"Child run contains no split fingerprint: {child_run_dir}"
                    )
                if suite_split_fingerprints is None:
                    suite_split_fingerprints = current_split_fingerprints
                elif current_split_fingerprints != suite_split_fingerprints:
                    raise RuntimeError(
                        "Suite trials/seeds did not use identical split "
                        "fingerprints."
                    )

                def mean_headline(
                    extractor: Any,
                    metric_rows: Sequence[Mapping[str, Any]] = fold_metrics,
                ) -> float | None:
                    values = [
                        extractor(metrics)
                        for metrics in metric_rows
                    ]
                    finite_values = [
                        float(value)
                        for value in values
                        if value is not None
                        and math.isfinite(float(value))
                    ]
                    return (
                        float(np.mean(finite_values))
                        if finite_values
                        else None
                    )

                performance_rows = [
                    json.loads(path.read_text(encoding="utf-8"))
                    for path in sorted(
                        (child_run_dir / "metrics").glob(
                            "*/performance.json"
                        )
                    )
                ]
                child_result = {
                    "trial_index": trial_index,
                    "experiment_id": experiment_id,
                    "task_mode": child_config["task_mode"],
                    "seed": int(seed),
                    "run_dir": str(child_run_dir),
                    "folds_completed": len(child_summary["fold_results"]),
                    "split_fingerprint_set_sha256": (
                        science.semantic_fingerprint(
                            current_split_fingerprints
                        )
                    ),
                    "artifact_manifest_sha256": sha256_file(
                        child_run_dir / "artifact_manifest.json"
                    ),
                    "run_summary_sha256": sha256_file(
                        child_run_dir / "reports" / "run_summary.json"
                    ),
                    "binary_auroc_mean": mean_headline(
                        lambda metrics: (
                            metrics["binary"]["auroc"]
                            if metrics["binary"] is not None
                            else None
                        )
                    ),
                    "binary_f1_mean": mean_headline(
                        lambda metrics: (
                            metrics["binary"]["f1"]
                            if metrics["binary"] is not None
                            else None
                        )
                    ),
                    "coarse_macro_f1_all_classes_mean": mean_headline(
                        lambda metrics: (
                            metrics["coarse"][
                                "macro_f1_all_classes"
                            ]
                            if metrics["coarse"] is not None
                            else None
                        )
                    ),
                    "fine_macro_f1_all_classes_mean": mean_headline(
                        lambda metrics: (
                            metrics["fine"]["macro_f1_all_classes"]
                            if metrics["fine"] is not None
                            else None
                        )
                    ),
                    "primary_macro_f1_all_classes_mean": mean_headline(
                        lambda metrics: (
                            metrics["primary_fine"][
                                "macro_f1_all_classes"
                            ]
                            if metrics["primary_fine"] is not None
                            and metrics["primary_fine"]["status"] == "ok"
                            else None
                        )
                    ),
                    "hierarchical_accuracy_mean": mean_headline(
                        lambda metrics: (
                            metrics["hierarchy"][
                                "hierarchical_accuracy"
                            ]
                            if metrics["hierarchy"] is not None
                            else None
                        )
                    ),
                    "train_samples_per_second_mean": (
                        float(
                            np.mean(
                                [
                                    row[
                                        "train_samples_per_second_mean"
                                    ]
                                    for row in performance_rows
                                ]
                            )
                        )
                        if performance_rows
                        else None
                    ),
                }
                child_results.append(child_result)
                write_json(
                    parent_run_dir / "reports" / "trial_index_latest.json",
                    {
                        "protocol_sha256": protocol_hash,
                        "completed": child_results,
                    },
                )
                logger.info(
                    "Completed trial=%s seed=%d run_dir=%s",
                    experiment_id,
                    int(seed),
                    child_run_dir,
                )
        suite_summary = {
            "schema_version": "cystods.training_suite.v2",
            "stage_name": stage_config["stage_name"],
            "study_id": stage_config["study_id"],
            "protocol_run_dir": str(protocol_run_dir),
            "protocol_sha256": protocol_hash,
            "protocol_role": role,
            "evaluation_scope": scope,
            "child_runs": child_results,
            "split_fingerprints": suite_split_fingerprints,
        }
        if role == "final_cv":
            suite_summary["integrated_cross_validation_reports"] = [
                {
                    "experiment_id": result["experiment_id"],
                    "seed": result["seed"],
                    "run_dir": result["run_dir"],
                    "cross_validation": json.loads(
                        (
                            Path(result["run_dir"])
                            / "reports"
                            / "run_summary.json"
                        ).read_text(encoding="utf-8")
                    )["cross_validation"],
                }
                for result in child_results
            ]
        write_json(
            parent_run_dir / "reports" / "run_summary.json",
            suite_summary,
        )
        pd.DataFrame(child_results).to_csv(
            parent_run_dir / "reports" / "child_runs.csv",
            index=False,
        )
        if role == "final_cv":
            write_json(
                parent_run_dir
                / "reports"
                / "final_cross_validation_report.json",
                suite_summary,
            )
            pd.DataFrame(child_results).to_csv(
                parent_run_dir
                / "reports"
                / "final_cross_validation_report.csv",
                index=False,
            )
        report_lines = [
            f"# {stage_config['stage_name']}",
            "",
            f"- Protocol SHA-256: `{protocol_hash}`",
            f"- Protocol role: `{role}`",
            f"- Evaluation scope: `{scope}`",
            f"- Completed child runs: {len(child_results)}",
            "",
            "## Child runs",
            "",
            pd.DataFrame(child_results).to_markdown(index=False),
            "",
        ]
        rendered_stage_report = "\n".join(report_lines)
        (
            parent_run_dir / "reports" / "stage_report.md"
        ).write_text(rendered_stage_report, encoding="utf-8")
        if role == "final_cv":
            (
                parent_run_dir
                / "reports"
                / "final_cross_validation_report.md"
            ).write_text(rendered_stage_report, encoding="utf-8")
        write_json(
            status_path,
            {
                "status": "completed",
                "started_utc": started_utc,
                "completed_utc": utc_now_iso(),
                "stage_name": stage_config["stage_name"],
                "run_dir": parent_run_dir,
                "protocol_sha256": protocol_hash,
                "child_runs_completed": len(child_results),
            },
        )
        logger.info(
            "Training suite completed successfully: child_runs=%d run_dir=%s",
            len(child_results),
            parent_run_dir,
        )
        return parent_run_dir
    except Exception as exc:
        logger.exception("Training suite failed; no fallback was used.")
        write_json(
            status_path,
            {
                "status": "failed",
                "started_utc": started_utc,
                "failed_utc": utc_now_iso(),
                "stage_name": stage_config["stage_name"],
                "run_dir": parent_run_dir,
                "protocol_sha256": protocol_hash,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        logger.error(
            "FINAL_STATUS failed error_type=%s run_dir=%s",
            type(exc).__name__,
            parent_run_dir,
        )
        raise
    finally:
        close_logger(logger)
        write_artifact_manifest(parent_run_dir)


def run_external_validation_stage(
    stage_config: Mapping[str, Any],
    required_source_files: Sequence[Path | str],
) -> Path:
    expected_keys = {
        "schema_version",
        "stage_name",
        "study_id",
        "run_profile",
        "data_root",
        "result_root",
        "selected_run_dir",
        "hf_checkpoint_receipt_json",
        "external_manifest_csv",
        "external_image_root",
        "internal_protocol_run_dir",
        "base_config",
        "external_columns",
    }
    _validate_stage_config_keys(stage_config, expected_keys)
    source_files = _complete_stage_source_files(required_source_files)
    selected_run_dir = Path(stage_config["selected_run_dir"]).expanduser().resolve()
    receipt_json_path = Path(
        stage_config["hf_checkpoint_receipt_json"]
    ).expanduser().resolve()
    external_manifest_csv = Path(
        stage_config["external_manifest_csv"]
    ).expanduser().resolve()
    external_image_root = Path(
        stage_config["external_image_root"]
    ).expanduser().resolve()
    protocol_run_dir = Path(
        stage_config["internal_protocol_run_dir"]
    ).expanduser().resolve()
    for directory in (
        selected_run_dir,
        external_image_root,
        protocol_run_dir,
    ):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    for file_path in (receipt_json_path, external_manifest_csv):
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
    if not receipt_json_path.is_relative_to(selected_run_dir):
        raise ValueError(
            "hf_checkpoint_receipt_json must be contained in "
            "selected_run_dir."
        )
    selected_status_path = selected_run_dir / "run_status.json"
    selected_artifact_path = selected_run_dir / "artifact_manifest.json"
    if not selected_status_path.is_file() or not selected_artifact_path.is_file():
        raise FileNotFoundError(
            "Selected run must contain run_status.json and artifact_manifest.json."
        )
    selected_status = json.loads(
        selected_status_path.read_text(encoding="utf-8")
    )
    if selected_status.get("status") != "completed":
        raise ValueError("Selected model run is not completed.")
    artifact_rows = json.loads(
        selected_artifact_path.read_text(encoding="utf-8")
    )
    relative_receipt = receipt_json_path.relative_to(
        selected_run_dir
    ).as_posix()
    artifact_row = next(
        (
            row
            for row in artifact_rows
            if row.get("path") == relative_receipt
        ),
        None,
    )
    if artifact_row is None:
        raise ValueError(
            "Selected HF receipt is absent from the artifact manifest."
        )
    if (
        int(artifact_row["bytes"]) != receipt_json_path.stat().st_size
        or artifact_row["sha256"] != sha256_file(receipt_json_path)
    ):
        raise ValueError("Selected HF receipt fails artifact validation.")
    receipt = checkpoint_hub.load_and_validate_receipt(receipt_json_path)
    protocol_path = protocol_run_dir / "protocol_manifest.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol_hash = sha256_file(protocol_path)
    checkpoint_temp_owner = tempfile.TemporaryDirectory(
        prefix="cystods-stage60-checkpoint-"
    )
    checkpoint_path = checkpoint_hub.download_verified_checkpoint(
        receipt_json_path,
        Path(checkpoint_temp_owner.name) / "checkpoint",
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint payload must be a mapping.")
    required_checkpoint_keys = {
        "model_state_dict",
        "config",
        "binary_names",
        "coarse_names",
        "fine_names",
        "data_split_fingerprint",
    }
    missing_checkpoint = required_checkpoint_keys - set(checkpoint)
    if missing_checkpoint:
        raise ValueError(
            f"Checkpoint missing keys: {sorted(missing_checkpoint)}"
        )
    if tuple(checkpoint["binary_names"]) != BINARY_NAMES:
        raise ValueError("Checkpoint binary taxonomy mismatch.")
    if tuple(checkpoint["coarse_names"]) != COARSE_NAMES:
        raise ValueError("Checkpoint coarse taxonomy mismatch.")
    if tuple(checkpoint["fine_names"]) != FINE_NAMES:
        raise ValueError("Checkpoint fine taxonomy mismatch.")
    saved_config = normalize_core_config(checkpoint["config"])
    if saved_config["study_id"] != stage_config["study_id"]:
        raise ValueError("Selected checkpoint study_id mismatch.")
    if saved_config["protocol_reference_sha256"] != protocol_hash:
        raise ValueError(
            "Selected checkpoint is not bound to internal_protocol_run_dir."
        )
    if saved_config["hf_repo_id"] != receipt["repo_id"]:
        raise ValueError("Checkpoint config and HF receipt repository differ.")
    if saved_config["evaluation_scope"] not in {"development", "final_cv"}:
        raise ValueError(
            "External validation requires a checkpoint from fixed hold-out "
            "development or final cross-validation."
        )
    if "binary" not in active_tasks_from_config(saved_config):
        raise ValueError("External validation requires a trained binary head.")
    columns = dict(stage_config["external_columns"])
    if set(columns) != {
        "path_column",
        "binary_label_column",
        "patient_id_column",
    }:
        raise KeyError("external_columns schema mismatch.")
    runtime_config = normalize_core_config(stage_config["base_config"])
    evaluation_config = dict(saved_config)
    for runtime_key in (
        "batch_size",
        "eval_batch_size",
        "num_workers",
        "eval_num_workers",
        "prefetch_factor",
        "eval_prefetch_factor",
        "persistent_workers",
        "pin_memory",
        "device",
        "precision",
        "enable_tf32",
        "channels_last",
        "torch_compile",
        "torch_compile_mode",
        "float32_matmul_precision",
        "num_cpu_threads",
        "binary_decision_threshold",
    ):
        evaluation_config[runtime_key] = runtime_config[runtime_key]
    evaluation_config.update(
        {
            "stage_name": str(stage_config["stage_name"]),
            "run_profile": str(stage_config["run_profile"]),
            "result_root": Path(stage_config["result_root"]).resolve(),
            "experiment_name": str(stage_config["stage_name"]),
            "external_validation_enabled": True,
            "external_manifest_csv": external_manifest_csv,
            "external_image_root": external_image_root,
            "external_path_column": str(columns["path_column"]),
            "external_binary_label_column": str(
                columns["binary_label_column"]
            ),
            "external_patient_id_column": str(
                columns["patient_id_column"]
            ),
            "evaluation_scope": "external",
            # The checkpoint supplies every learned tensor; downloading the
            # original pretraining weights would be redundant and would make
            # an evaluation-only stage depend on network access.
            "pretrained": False,
        }
    )
    validate_config(evaluation_config)
    run_dir = make_run_directory(evaluation_config)
    logger = setup_logger(run_dir / "logs" / "training.log")
    status_path = run_dir / "run_status.json"
    started_utc = utc_now_iso()
    write_json(
        status_path,
        {
            "status": "running",
            "started_utc": started_utc,
            "stage_name": stage_config["stage_name"],
            "run_dir": run_dir,
            "protocol_sha256": protocol_hash,
            "selected_checkpoint_sha256": receipt["checkpoint_sha256"],
        },
    )
    try:
        _write_stage_system_artifacts(run_dir, stage_config, source_files)
        write_json(run_dir / "evaluation_config.json", evaluation_config)
        write_json(
            run_dir / "system" / "selected_model_binding.json",
            {
                "selected_run_dir": str(selected_run_dir),
                "hf_checkpoint_receipt_json": str(receipt_json_path),
                "hf_receipt_sha256": artifact_row["sha256"],
                "hf_repo_id": receipt["repo_id"],
                "hf_path_in_repo": receipt["path_in_repo"],
                "hf_commit_oid": receipt["commit_oid"],
                "checkpoint_sha256": receipt["checkpoint_sha256"],
                "selected_artifact_manifest_sha256": sha256_file(
                    selected_artifact_path
                ),
                "protocol_run_dir": str(protocol_run_dir),
                "protocol_sha256": protocol_hash,
            },
        )
        device = resolve_device(evaluation_config)
        precision_name, amp_dtype = resolve_precision(
            evaluation_config,
            device,
        )
        torch.set_num_threads(int(evaluation_config["num_cpu_threads"]))
        torch.set_float32_matmul_precision(
            str(evaluation_config["float32_matmul_precision"])
        )
        seed_everything(
            int(evaluation_config["seed"]),
            bool(evaluation_config["deterministic"]),
        )
        model = HierarchicalCystoModel(evaluation_config).to(device)
        if evaluation_config["channels_last"]:
            model = model.to(memory_format=torch.channels_last)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if evaluation_config["torch_compile"]:
            if not hasattr(torch, "compile"):
                raise RuntimeError(
                    "Checkpoint requires torch.compile, which is unavailable."
                )
            model.encoder = torch.compile(
                model.encoder,
                mode=str(evaluation_config["torch_compile_mode"]),
            )
        metrics_dir = run_dir / "metrics" / "external"
        prediction_dir = run_dir / "predictions" / "external"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        prediction_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "External evaluation starts device=%s precision=%s checkpoint=%s",
            device,
            precision_name,
            checkpoint_path,
        )
        metrics = evaluate_external_binary(
            model,
            evaluation_config,
            device,
            amp_dtype,
            run_dir,
            "external",
        )
        write_json(
            run_dir / "reports" / "run_summary.json",
            {
                "protocol_sha256": protocol_hash,
                "selected_checkpoint_sha256": receipt[
                    "checkpoint_sha256"
                ],
                "hf_checkpoint_receipt": receipt,
                "external_manifest_sha256": sha256_file(
                    external_manifest_csv
                ),
                "external_metrics": metrics,
                "fine_tuning_performed": False,
            },
        )
        pd.DataFrame(
            [
                {"metric": key, "value": value}
                for key, value in metrics.items()
                if isinstance(value, (int, float, type(None)))
            ]
        ).to_csv(
            run_dir / "reports" / "external_metrics.csv",
            index=False,
        )
        report = "\n".join(
            [
                "# External validation report",
                "",
                "- Evaluation-only: `true`",
                "- Fine-tuning performed: `false`",
                f"- Protocol SHA-256: `{protocol_hash}`",
                f"- HF commit: `{receipt['commit_oid']}`",
                f"- Checkpoint SHA-256: `{receipt['checkpoint_sha256']}`",
                "",
                "## Binary metrics",
                "",
                pd.DataFrame(
                    [
                        {"metric": key, "value": value}
                        for key, value in metrics.items()
                        if isinstance(value, (int, float, type(None)))
                    ]
                ).to_markdown(index=False),
                "",
            ]
        )
        (run_dir / "reports" / "external_report.md").write_text(
            report,
            encoding="utf-8",
        )
        write_json(
            status_path,
            {
                "status": "completed",
                "started_utc": started_utc,
                "completed_utc": utc_now_iso(),
                "stage_name": stage_config["stage_name"],
                "run_dir": run_dir,
                "protocol_sha256": protocol_hash,
                "selected_checkpoint_sha256": receipt[
                    "checkpoint_sha256"
                ],
            },
        )
        logger.info(
            "External validation completed successfully: run_dir=%s",
            run_dir,
        )
        return run_dir
    except Exception as exc:
        logger.exception("External validation failed; no fallback was used.")
        write_json(
            status_path,
            {
                "status": "failed",
                "started_utc": started_utc,
                "failed_utc": utc_now_iso(),
                "stage_name": stage_config["stage_name"],
                "run_dir": run_dir,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        logger.error(
            "FINAL_STATUS failed error_type=%s run_dir=%s",
            type(exc).__name__,
            run_dir,
        )
        raise
    finally:
        close_logger(logger)
        checkpoint_temp_owner.cleanup()
        local_models = sorted(run_dir.rglob("*.pt"))
        if local_models:
            raise RuntimeError(
                "External validation left local checkpoint files in result: "
                f"{[str(path) for path in local_models]}"
            )
        write_artifact_manifest(run_dir)


# %%
if __name__ == "__main__":
    COMPLETED_RUN_DIRECTORY = main()
    print(f"Completed CystoDS run: {COMPLETED_RUN_DIRECTORY}")
