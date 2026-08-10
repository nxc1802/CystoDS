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
# CELL 1 - Strict runtime dependency installation and validation.
import importlib as _importlib
import importlib.metadata as _metadata
import importlib.util as _importlib_util
import subprocess as _subprocess
import sys as _sys

if _importlib_util.find_spec("pip") is None:
    _subprocess.check_call([_sys.executable, "-m", "ensurepip", "--upgrade"])

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
    "huggingface-hub>=0.28,<2",
]
_IMPORT_NAMES = {"pillow": "PIL", "scikit-learn": "sklearn"}


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
            installed, prereleases=True
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
    _importlib.import_module(
        _IMPORT_NAMES.get(
            _requirement.name.lower(),
            _requirement.name.lower().replace("-", "_"),
        )
    )
_subprocess.run(
    [_sys.executable, "-m", "pip", "check"],
    check=True,
    capture_output=True,
    text=True,
)


# %%
# CELL 2 - All protocol-stage configuration.
import os
from pathlib import Path

RUN_PROFILE = os.environ.get("CYSTODS_RUN_PROFILE", "research")
if RUN_PROFILE not in {"research", "smoke"}:
    raise ValueError("CYSTODS_RUN_PROFILE must be research or smoke.")

STAGE_NAME = "stage_00_prepare_protocol"
STUDY_ID = os.environ.get(
    "CYSTODS_STUDY_ID", "cystods_hierarchical_long_tailed_2026"
)


def _resolve_data_root() -> Path:
    explicit_value = os.environ.get("CYSTODS_DATA_ROOT")
    if explicit_value:
        explicit = Path(explicit_value).expanduser().resolve()
        if not (explicit / "cystods.csv").is_file():
            raise FileNotFoundError(
                f"CYSTODS_DATA_ROOT does not contain cystods.csv: {explicit}"
            )
        return explicit
    candidates = (
        Path("/kaggle/input/datasets/cuongnguyen1802/cystods"),
        Path("/kaggle/input/cystods"),
        Path("/kaggle/input/cystods/xvdhy-osfstorage-archive"),
        Path.cwd() / "xvdhy-osfstorage-archive",
        Path.cwd().parent / "xvdhy-osfstorage-archive",
    )
    matches = sorted(
        {
            candidate.resolve()
            for candidate in candidates
            if (candidate / "cystods.csv").is_file()
        }
    )
    if len(matches) != 1:
        raise RuntimeError(
            "Exactly one CystoDS data root is required; "
            f"found={list(map(str, matches))}. Set CYSTODS_DATA_ROOT."
        )
    return matches[0]


DATA_ROOT = _resolve_data_root()
RESULT_ROOT = Path(
    os.environ.get(
        "CYSTODS_RESULT_ROOT",
        "/kaggle/working/result"
        if Path("/kaggle/working").is_dir()
        else str(Path.cwd() / "result"),
    )
).expanduser().resolve()

PROTOCOL_CONFIG = {
    "schema_version": "cystods.core.v2",
    "stage_name": STAGE_NAME,
    "study_id": STUDY_ID,
    "data_root": DATA_ROOT,
    "metadata_csv": DATA_ROOT / "cystods.csv",
    "image_dir": DATA_ROOT / "images",
    "segmentation_dir": DATA_ROOT / "segmentations",
    "inclusion_manifest_csv": (
        Path(os.environ["CYSTODS_INCLUSION_MANIFEST_CSV"]).expanduser().resolve()
        if os.environ.get("CYSTODS_INCLUSION_MANIFEST_CSV")
        else None
    ),
    "inclusion_manifest_filename_column": "filename",
    "result_root": RESULT_ROOT,
    "experiment_name": STAGE_NAME,
    "run_profile": RUN_PROFILE,
    "seed": 20260729,
    "deterministic": True,
    "verify_all_image_decodes": True,
    "verify_segmentation_inventory": True,
    "dataset_fingerprint_mode": "full",
    "verify_exact_duplicate_images": True,
    "num_cpu_threads": 4,
    # The protocol stage creates exactly one frozen 70/15/15 patient-disjoint
    # hold-out. Stage 90 creates final CV independently over the same audited
    # dataset and binds its taxonomy/fingerprint to this manifest.
    "protocol": "holdout",
    "train_fraction": 0.70,
    "val_fraction": 0.15,
    "test_fraction": 0.15,
    "split_seed": 20260729,
    "split_search_candidates": 4096,
    "force_fine_labels_with_fewer_than_n_patients_to_train": 3,
    "cv_folds": 5,
    "cv_val_fraction_of_remaining": 0.15,
    "cv_run_fold_indices": None,
    "normal_mucosa_limit": 540,
    "fixed_split_pids": None,
    "protocol_manifest_dir": None,
    "max_train_samples": None,
    "max_val_samples": None,
    "max_test_samples": None,
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
    "batch_size": int(os.environ.get("CYSTODS_BATCH_SIZE", "256")),
    "num_workers": int(os.environ.get("CYSTODS_NUM_WORKERS", "12")),
    "prefetch_factor": int(os.environ.get("CYSTODS_PREFETCH_FACTOR", "4")),
    "persistent_workers": True,
    "pin_memory": True,
    "model_name": "swin_tiny_patch4_window7_224.ms_in1k",
    "pretrained": True,
    "task_mode": "hierarchical",
    "dropout": 0.20,
    "projection_dim": 128,
    "binary_loss_weight": 1.0,
    "coarse_loss_weight": 1.0,
    "fine_loss_weight": 1.0,
    "binary_coarse_hierarchy_loss_weight": 0.25,
    "coarse_fine_hierarchy_loss_weight": 0.25,
    "supervised_contrastive_loss_weight": 0.10,
    "supervised_contrastive_temperature": 0.10,
    "supervised_contrastive_label_level": "fine",
    "fine_loss": "balanced_softmax_smoothed",
    "class_balance_beta": 0.9999,
    "focal_gamma": 2.0,
    "focal_use_class_balance": False,
    "logit_adjustment_tau": 1.0,
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
    "sampler": "random",
    "sampler_label_level": "fine",
    "epochs": 25,
    "learning_rate": 3.0e-4,
    "encoder_learning_rate_multiplier": 0.25,
    "weight_decay": 0.05,
    "optimizer": "adamw",
    "use_fused_optimizer": os.environ.get("CYSTODS_USE_FUSED_OPTIMIZER", "true").lower() == "true",
    "warmup_epochs": 2.0,
    "scheduler_epochs": 25,
    "minimum_learning_rate_ratio": 0.01,
    "gradient_accumulation_steps": 1,
    "gradient_clip_norm": 1.0,
    "early_stopping_patience": 6,
    "monitor_metric": "hierarchical_composite",
    "hierarchical_composite_weights": {
        "coarse_macro_f1_all_classes": 0.35,
        "primary_macro_f1_all_classes": 0.45,
        "hierarchical_accuracy": 0.20,
    },
    "checkpoint_min_delta": 1.0e-4,
    "binary_decision_threshold": 0.5,
    "resume_checkpoint": None,
    "device": "auto",
    "precision": "auto",
    "enable_tf32": True,
    "channels_last": True,
    "torch_compile": False,
    "torch_compile_mode": "default",
    "float32_matmul_precision": "high",
    "log_every_n_steps": 20,
    "save_last_checkpoint": False,
    "save_epoch_checkpoints": False,
    "bootstrap_iterations": 1000,
    "bootstrap_confidence": 0.95,
    "primary_fine_min_train_patients": 10,
    "fixed_primary_fine_class_ids": None,
    "tail_class_max_train_samples": 20,
    "rare_gate_max_train_patients": 2,
    "rare_gate_absolute_pred_share": 0.10,
    "rare_gate_prior_multiplier": 10.0,
    "rare_gate_min_pred_count": 5,
    "scientific_gate_mode": "enforce",
    "probability_sum_tolerance": 5.0e-3,
    "paired_baseline_predictions_csv": None,
    "generate_sample_grid": True,
    "sample_grid_images_per_class": 3,
    "evaluate_wlc_only": False,
    "train_modality": "all",
    "evaluate_roi_level": False,
    "roi_aggregations": ("mean", "vote", "attention"),
    "roi_conflict_policy": "exclude_and_report",
    "roi_attention_epochs": 25,
    "roi_attention_learning_rate": 1.0e-3,
    "roi_attention_hidden_dim": 128,
    "roi_attention_early_stopping_patience": 5,
    "external_validation_enabled": False,
    "external_manifest_csv": None,
    "external_image_root": None,
    "external_path_column": "path",
    "external_binary_label_column": "binary_label",
    "external_patient_id_column": "patient_id",
}

if RUN_PROFILE == "smoke":
    PROTOCOL_CONFIG.update(
        {
            "experiment_name": f"{STAGE_NAME}_smoke",
            "verify_all_image_decodes": False,
            "dataset_fingerprint_mode": "semantic",
            "verify_exact_duplicate_images": False,
            "protocol": "holdout",
            "split_search_candidates": 64,
            "cv_folds": 2,
            "fixed_split_pids": {
                "train": ("1552",),
                "val": ("1261",),
                "test": ("808",),
            },
            "normal_mucosa_limit": None,
            "image_size": 64,
            "batch_size": 4,
            "num_workers": 0,
            "persistent_workers": False,
            "pin_memory": False,
            "model_name": "swin_tiny_patch4_window7_224.ms_in1k",
            "pretrained": False,
            "fine_loss": "focal",
            "fine_inference_calibration_mode": "fixed",
            "epochs": 1,
            "scheduler_epochs": 1,
            "warmup_epochs": 0.0,
            "early_stopping_patience": 1,
            "monitor_metric": "coarse_macro_f1",
            "device": "cpu",
            "precision": "fp32",
            "enable_tf32": False,
            "bootstrap_iterations": 20,
            "scientific_gate_mode": "report",
            "generate_sample_grid": False,
            "roi_attention_epochs": 1,
        }
    )

CONFIG = {
    "schema_version": "cystods.stage.v2",
    "stage_name": STAGE_NAME,
    "study_id": STUDY_ID,
    "run_profile": RUN_PROFILE,
    "data_root": DATA_ROOT,
    "result_root": RESULT_ROOT,
    "protocol_config": PROTOCOL_CONFIG,
}


# %%
# CELL 3 - Stage entry point. Importing this module never launches the stage.
import cystods_core as core

try:
    _THIS_SOURCE = Path(__file__).resolve()
except NameError:
    _THIS_SOURCE = (Path.cwd() / "stage_00_prepare_protocol.py").resolve()

REQUIRED_SOURCE_FILES = (
    _THIS_SOURCE,
    _THIS_SOURCE.with_name("cystods_core.py"),
    _THIS_SOURCE.with_name("cystods_hf.py"),
    _THIS_SOURCE.with_name("cystods_science.py"),
    _THIS_SOURCE.with_name("README.md"),
)


def main(config=None) -> Path:
    resolved = dict(CONFIG if config is None else config)
    return core.run_protocol_stage(resolved, REQUIRED_SOURCE_FILES)


# %%
if __name__ == "__main__":
    COMPLETED_RUN_DIRECTORY = main()
    print(f"Completed protocol stage: {COMPLETED_RUN_DIRECTORY}")
