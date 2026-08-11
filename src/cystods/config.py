"""Central configuration loader for CystoDS.

Config resolution order (later wins):
    config.yaml → profile override → environment variables → CLI --set
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


_SENTINEL = object()

# Environment variable mappings to flat config keys.
_ENV_OVERRIDES: dict[str, tuple[str, type]] = {
    "CYSTODS_RUN_PROFILE": ("_profile", str),
    "CYSTODS_DATA_ROOT": ("paths.data_root", str),
    "CYSTODS_RESULT_ROOT": ("paths.result_root", str),
    "CYSTODS_BATCH_SIZE": ("runtime.batch_size", int),
    "CYSTODS_EVAL_BATCH_SIZE": ("runtime.eval_batch_size", int),
    "CYSTODS_NUM_WORKERS": ("runtime.num_workers", int),
    "CYSTODS_EVAL_NUM_WORKERS": ("runtime.eval_num_workers", int),
    "CYSTODS_NUM_CPU_THREADS": ("runtime.num_cpu_threads", int),
    "CYSTODS_PREFETCH_FACTOR": ("runtime.prefetch_factor", int),
    "CYSTODS_EVAL_PREFETCH_FACTOR": ("runtime.eval_prefetch_factor", int),
    "CYSTODS_MODEL_NAME": ("model.name", str),
    "CYSTODS_STUDY_ID": ("project.study_id", str),
    "CYSTODS_HF_REPO_ID": ("hf.repo_id", str),
    "CYSTODS_HF_REVISION": ("hf.revision", str),
    "CYSTODS_PROTOCOL_RUN_DIR": ("_protocol_run_dir", str),
    "CYSTODS_EXPECTED_PROTOCOL_SHA256": ("_expected_protocol_sha256", str),
    "CYSTODS_TORCH_COMPILE": ("runtime.torch_compile", str),
    "CYSTODS_USE_FUSED_OPTIMIZER": ("training.use_fused_optimizer", str),
    "CYSTODS_CHECKPOINT_BACKEND": ("checkpoint.backend", str),
}

# Keys resolved from the YAML config that map to the flat dict keys
# expected by cystods_core functions.
_DEFAULT_CONFIG_PATH = Path("config.yaml")


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (returns new dict)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _set_nested(data: dict, dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using dot-separated key path."""
    keys = dotted_key.split(".")
    current = data
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _get_nested(data: dict, dotted_key: str, default: Any = _SENTINEL) -> Any:
    """Get a value from a nested dict using dot-separated key path."""
    keys = dotted_key.split(".")
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            if default is _SENTINEL:
                raise KeyError(f"Config key not found: {dotted_key}")
            return default
        current = current[key]
    return current


def _coerce_value(raw: str) -> Any:
    """Coerce a CLI string value to the appropriate Python type."""
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    if raw.lower() in ("null", "none"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    # Handle list notation: [a,b,c]
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [_coerce_value(item.strip().strip("'\"")) for item in inner.split(",")]
    return raw


def load_yaml(config_path: Path | str | None = None) -> dict:
    """Load the raw YAML config file."""
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.is_file():
        # Try relative to the project root (parent of src/)
        alt = Path(__file__).resolve().parent.parent.parent / path.name
        if alt.is_file():
            path = alt
        else:
            raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a YAML mapping: {path}")
    return data


def apply_profile(config: dict, profile: str | None) -> dict:
    """Merge a named profile on top of base config."""
    if profile is None:
        return config
    profiles = config.get("profiles", {})
    if profile not in profiles:
        if profile == "research":
            # research is the default — no overrides needed
            return config
        raise ValueError(
            f"Unknown profile '{profile}'. "
            f"Available: {sorted(profiles.keys())}"
        )
    return _deep_merge(config, profiles[profile])


def apply_env_overrides(config: dict) -> dict:
    """Apply environment variable overrides."""
    result = copy.deepcopy(config)
    for env_name, (config_key, cast) in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_name)
        if raw is not None:
            if config_key.startswith("_"):
                # Internal keys stored at top level
                result[config_key] = cast(raw)
            else:
                _set_nested(result, config_key, cast(raw))
    return result


def apply_cli_overrides(config: dict, overrides: list[str] | None) -> dict:
    """Apply --set key=value overrides from the CLI."""
    if not overrides:
        return config
    result = copy.deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"CLI override must be key=value, got: {override!r}"
            )
        key, raw_value = override.split("=", 1)
        key = key.strip()
        value = _coerce_value(raw_value.strip())
        _set_nested(result, key, value)
    return result


def resolve_paths(config: dict) -> dict:
    """Resolve data_root and result_root to absolute paths."""
    result = copy.deepcopy(config)
    paths = result.get("paths", {})

    # Data root resolution
    data_root_str = paths.get("data_root", "./xvdhy-osfstorage-archive")
    data_root = Path(data_root_str).expanduser().resolve()

    if not data_root.is_dir():
        # Try well-known Kaggle locations
        candidates = [
            Path("/kaggle/input/datasets/cuongnguyen1802/cystods"),
            Path("/kaggle/input/cystods"),
            Path("/kaggle/input/cystods/xvdhy-osfstorage-archive"),
            Path.cwd() / "xvdhy-osfstorage-archive",
            Path.cwd().parent / "xvdhy-osfstorage-archive",
        ]
        matches = sorted(
            {c.resolve() for c in candidates if (c / "cystods.csv").is_file()}
        )
        if len(matches) == 1:
            data_root = matches[0]
        elif len(matches) > 1:
            raise RuntimeError(
                f"Multiple CystoDS data roots found: {[str(m) for m in matches]}. "
                "Set paths.data_root explicitly."
            )

    paths["data_root"] = data_root
    paths["metadata_csv"] = data_root / "cystods.csv"
    paths["image_dir"] = data_root / "images"
    paths["segmentation_dir"] = data_root / "segmentations"

    # Result root resolution
    result_root_str = paths.get("result_root", "./result")
    result_root = Path(result_root_str).expanduser().resolve()
    if not result_root.is_absolute():
        if Path("/kaggle/working").is_dir():
            result_root = Path("/kaggle/working/result").resolve()
        else:
            result_root = (Path.cwd() / result_root_str).resolve()
    paths["result_root"] = result_root

    result["paths"] = paths
    return result


def flatten_to_core_config(config: dict, stage: str | None = None) -> dict:
    """Flatten the nested YAML config into the flat dict that core functions expect.

    This bridges the new hierarchical YAML config to the existing flat
    ``config["key"]`` access pattern used throughout cystods_core.
    """
    flat: dict[str, Any] = {}
    project = config.get("project", {})
    paths = config.get("paths", {})
    data = config.get("data", {})
    runtime = config.get("runtime", {})
    model = config.get("model", {})
    training = config.get("training", {})
    evaluation = config.get("evaluation", {})
    checkpoint = config.get("checkpoint", {})
    hf = config.get("hf", {})
    logging_cfg = config.get("logging", {})
    augmentation = data.get("augmentation", {})
    cv = data.get("cv", {})
    fine_loss = training.get("fine_loss", {})
    sampler = training.get("sampler", {})
    loss = training.get("loss", {})
    rare_gate = evaluation.get("rare_gate", {})
    roi = evaluation.get("roi", {})
    modality = evaluation.get("modality", {})
    external = evaluation.get("external", {})

    # Schema
    flat["schema_version"] = "cystods.core.v2"

    # Project
    flat["study_id"] = project.get("study_id", "cystods_hierarchical_long_tailed_2026")
    flat["seed"] = project.get("seed", 20260729)
    flat["split_seed"] = project.get("split_seed", 20260729)

    # Paths
    flat["data_root"] = paths.get("data_root", Path("."))
    flat["metadata_csv"] = paths.get("metadata_csv", Path(".") / "cystods.csv")
    flat["image_dir"] = paths.get("image_dir", Path(".") / "images")
    flat["segmentation_dir"] = paths.get("segmentation_dir", Path(".") / "segmentations")
    flat["result_root"] = paths.get("result_root", Path(".") / "result")
    flat["inclusion_manifest_csv"] = paths.get("inclusion_manifest_csv")
    flat["inclusion_manifest_filename_column"] = paths.get(
        "inclusion_manifest_filename_column", "filename"
    )

    # Data
    flat["image_size"] = data.get("image_size", 224)
    flat["fov_center_crop_ratio"] = data.get("fov_center_crop_ratio", 0.92)
    flat["train_fraction"] = data.get("train_fraction", 0.70)
    flat["val_fraction"] = data.get("val_fraction", 0.15)
    flat["test_fraction"] = data.get("test_fraction", 0.15)
    flat["normal_mucosa_limit"] = data.get("normal_mucosa_limit", 540)
    flat["split_search_candidates"] = data.get("split_search_candidates", 4096)
    flat["force_fine_labels_with_fewer_than_n_patients_to_train"] = data.get(
        "force_fine_labels_with_fewer_than_n_patients_to_train", 3
    )
    flat["verify_all_image_decodes"] = data.get("verify_all_image_decodes", False)
    flat["verify_segmentation_inventory"] = data.get("verify_segmentation_inventory", True)
    flat["dataset_fingerprint_mode"] = data.get("dataset_fingerprint_mode", "full")
    flat["verify_exact_duplicate_images"] = data.get("verify_exact_duplicate_images", True)
    flat["deterministic"] = data.get("deterministic", False)
    flat["max_train_samples"] = data.get("max_train_samples")
    flat["max_val_samples"] = data.get("max_val_samples")
    flat["max_test_samples"] = data.get("max_test_samples")
    flat["fixed_split_pids"] = data.get("fixed_split_pids")

    # Augmentation
    flat["random_resized_crop_scale"] = tuple(
        augmentation.get("random_resized_crop_scale", [0.75, 1.0])
    )
    flat["horizontal_flip_probability"] = augmentation.get("horizontal_flip_probability", 0.5)
    flat["vertical_flip_probability"] = augmentation.get("vertical_flip_probability", 0.5)
    flat["rotation_degrees"] = augmentation.get("rotation_degrees", 15)
    flat["color_jitter"] = tuple(augmentation.get("color_jitter", [0.20, 0.20, 0.15, 0.05]))
    flat["random_erasing_probability"] = augmentation.get("random_erasing_probability", 0.20)
    flat["imagenet_mean"] = tuple(augmentation.get("imagenet_mean", [0.485, 0.456, 0.406]))
    flat["imagenet_std"] = tuple(augmentation.get("imagenet_std", [0.229, 0.224, 0.225]))

    # CV
    flat["cv_folds"] = cv.get("folds", 5)
    flat["cv_val_fraction_of_remaining"] = cv.get("val_fraction_of_remaining", 0.15)
    flat["cv_run_fold_indices"] = cv.get("run_fold_indices")

    # Runtime
    flat["device"] = runtime.get("device", "cuda")
    flat["precision"] = runtime.get("precision", "bf16")
    flat["batch_size"] = runtime.get("batch_size", 256)
    flat["eval_batch_size"] = runtime.get("eval_batch_size", 1024)
    flat["num_workers"] = runtime.get("num_workers", 4)
    flat["eval_num_workers"] = runtime.get("eval_num_workers", 4)
    flat["prefetch_factor"] = runtime.get("prefetch_factor", 2)
    flat["eval_prefetch_factor"] = runtime.get("eval_prefetch_factor", 2)
    flat["persistent_workers"] = runtime.get("persistent_workers", True)
    flat["pin_memory"] = runtime.get("pin_memory", True)
    flat["num_cpu_threads"] = runtime.get("num_cpu_threads", 32)
    flat["enable_tf32"] = runtime.get("enable_tf32", True)
    flat["channels_last"] = runtime.get("channels_last", True)
    flat["torch_compile"] = runtime.get("torch_compile", False)
    flat["torch_compile_mode"] = runtime.get("torch_compile_mode", "default")
    flat["float32_matmul_precision"] = runtime.get("float32_matmul_precision", "high")

    # Model
    flat["model_name"] = model.get("name", "swin_tiny_patch4_window7_224.ms_in1k")
    flat["pretrained"] = model.get("pretrained", True)
    flat["task_mode"] = model.get("task_mode", "hierarchical")
    flat["dropout"] = model.get("dropout", 0.20)
    flat["projection_dim"] = model.get("projection_dim", 128)

    # Training
    flat["epochs"] = training.get("epochs", 25)
    flat["learning_rate"] = training.get("learning_rate", 3.0e-4)
    flat["encoder_learning_rate_multiplier"] = training.get(
        "encoder_learning_rate_multiplier", 0.25
    )
    flat["weight_decay"] = training.get("weight_decay", 0.05)
    flat["optimizer"] = training.get("optimizer", "adamw")
    flat["use_fused_optimizer"] = training.get("use_fused_optimizer", True)
    flat["warmup_epochs"] = training.get("warmup_epochs", 2.0)
    flat["scheduler_epochs"] = training.get("scheduler_epochs", 25)
    flat["minimum_learning_rate_ratio"] = training.get("minimum_learning_rate_ratio", 0.01)
    flat["gradient_accumulation_steps"] = training.get("gradient_accumulation_steps", 1)
    flat["gradient_clip_norm"] = training.get("gradient_clip_norm", 1.0)
    flat["early_stopping_patience"] = training.get("early_stopping_patience", 6)
    flat["checkpoint_min_delta"] = training.get("checkpoint_min_delta", 1.0e-4)
    flat["monitor_metric"] = training.get("monitor_metric", "hierarchical_composite")
    flat["hierarchical_composite_weights"] = training.get(
        "hierarchical_composite_weights",
        {
            "coarse_macro_f1_all_classes": 0.35,
            "primary_macro_f1_all_classes": 0.45,
            "hierarchical_accuracy": 0.20,
        },
    )
    flat["use_data_augmentation"] = training.get("use_data_augmentation", False)

    # Loss weights
    flat["binary_loss_weight"] = loss.get("binary_loss_weight", 1.0)
    flat["coarse_loss_weight"] = loss.get("coarse_loss_weight", 1.0)
    flat["fine_loss_weight"] = loss.get("fine_loss_weight", 1.0)
    flat["binary_coarse_hierarchy_loss_weight"] = loss.get(
        "binary_coarse_hierarchy_loss_weight", 0.25
    )
    flat["coarse_fine_hierarchy_loss_weight"] = loss.get(
        "coarse_fine_hierarchy_loss_weight", 0.25
    )
    flat["supervised_contrastive_loss_weight"] = loss.get(
        "supervised_contrastive_loss_weight", 0.10
    )
    flat["supervised_contrastive_temperature"] = loss.get(
        "supervised_contrastive_temperature", 0.10
    )
    flat["supervised_contrastive_label_level"] = loss.get(
        "supervised_contrastive_label_level", "fine"
    )

    # Fine loss
    flat["fine_loss"] = fine_loss.get("name", "balanced_softmax")
    flat["class_balance_beta"] = fine_loss.get("class_balance_beta", 0.9999)
    flat["focal_gamma"] = fine_loss.get("focal_gamma", 2.0)
    flat["focal_use_class_balance"] = fine_loss.get("focal_use_class_balance", False)
    flat["logit_adjustment_tau"] = fine_loss.get("logit_adjustment_tau", 0.5)
    flat["fine_prior_source"] = fine_loss.get("prior_source", "patient_count")
    flat["fine_prior_smoothing_alpha"] = fine_loss.get("prior_smoothing_alpha", 1.0)
    flat["fine_prior_power"] = fine_loss.get("prior_power", 0.5)
    flat["fine_prior_max_ratio"] = fine_loss.get("prior_max_ratio", 50.0)
    flat["fine_absent_train_policy"] = fine_loss.get(
        "absent_train_policy", "mask_and_score_zero"
    )
    flat["fine_inference_calibration_mode"] = fine_loss.get(
        "inference_calibration_mode", "validation_grid"
    )
    flat["fine_inference_prior_tau"] = fine_loss.get("inference_prior_tau", 0.0)
    flat["fine_inference_tau_grid"] = tuple(
        fine_loss.get("inference_tau_grid", [0.0, 0.25, 0.5, 0.75, 1.0])
    )
    flat["fine_inference_calibration_metric"] = fine_loss.get(
        "inference_calibration_metric", "primary_macro_f1_all_classes"
    )
    flat["ldam_max_margin"] = fine_loss.get("ldam_max_margin", 0.5)
    flat["ldam_scale"] = fine_loss.get("ldam_scale", 30.0)

    # Sampler
    flat["sampler"] = sampler.get("type", "random")
    flat["sampler_label_level"] = sampler.get("label_level", "fine")

    # Evaluation
    flat["binary_decision_threshold"] = evaluation.get("binary_decision_threshold", 0.5)
    flat["bootstrap_iterations"] = evaluation.get("bootstrap_iterations", 1000)
    flat["bootstrap_confidence"] = evaluation.get("bootstrap_confidence", 0.95)
    flat["probability_sum_tolerance"] = evaluation.get("probability_sum_tolerance", 5.0e-3)
    flat["primary_fine_min_train_patients"] = evaluation.get(
        "primary_fine_min_train_patients", 10
    )
    flat["fixed_primary_fine_class_ids"] = evaluation.get("fixed_primary_fine_class_ids")
    flat["tail_class_max_train_samples"] = evaluation.get("tail_class_max_train_samples", 20)
    flat["scientific_gate_mode"] = evaluation.get("scientific_gate_mode", "enforce")
    flat["generate_sample_grid"] = evaluation.get("generate_sample_grid", False)
    flat["sample_grid_images_per_class"] = evaluation.get("sample_grid_images_per_class", 3)
    flat["paired_baseline_predictions_csv"] = evaluation.get(
        "paired_baseline_predictions_csv"
    )

    # Rare gate
    flat["rare_gate_max_train_patients"] = rare_gate.get("max_train_patients", 2)
    flat["rare_gate_absolute_pred_share"] = rare_gate.get("absolute_pred_share", 0.10)
    flat["rare_gate_prior_multiplier"] = rare_gate.get("prior_multiplier", 10.0)
    flat["rare_gate_min_pred_count"] = rare_gate.get("min_pred_count", 5)

    # ROI evaluation
    flat["evaluate_roi_level"] = roi.get("evaluate_roi_level", False)
    flat["roi_aggregations"] = tuple(roi.get("aggregations", ["mean", "vote", "attention"]))
    flat["roi_conflict_policy"] = roi.get("conflict_policy", "exclude_and_report")
    flat["roi_attention_epochs"] = roi.get("attention_epochs", 25)
    flat["roi_attention_learning_rate"] = roi.get("attention_learning_rate", 1.0e-3)
    flat["roi_attention_hidden_dim"] = roi.get("attention_hidden_dim", 128)
    flat["roi_attention_early_stopping_patience"] = roi.get(
        "attention_early_stopping_patience", 5
    )

    # Modality
    flat["train_modality"] = modality.get("train_modality", "all")
    flat["evaluate_wlc_only"] = modality.get("evaluate_wlc_only", False)

    # External validation
    flat["external_validation_enabled"] = external.get("enabled", False)
    flat["external_manifest_csv"] = external.get("manifest_csv")
    flat["external_image_root"] = external.get("image_root")
    flat["external_path_column"] = external.get("path_column", "path")
    flat["external_binary_label_column"] = external.get("binary_label_column", "binary_label")
    flat["external_patient_id_column"] = external.get("patient_id_column", "patient_id")

    # Checkpoint
    flat["checkpoint_backend"] = checkpoint.get("backend", "huggingface")
    flat["save_last_checkpoint"] = checkpoint.get("save_last", False)
    flat["save_epoch_checkpoints"] = checkpoint.get("save_epoch", False)
    flat["resume_checkpoint"] = checkpoint.get("resume")

    # HF
    flat["hf_repo_id"] = hf.get("repo_id")
    flat["hf_revision"] = hf.get("revision", "main")
    flat["hf_private"] = hf.get("private", True)
    flat["hf_create_repo"] = hf.get("create_repo", True)
    flat["hf_path_prefix"] = hf.get(
        "path_prefix", f"{flat['study_id']}/{flat.get('stage_name', 'cystods')}"
    )
    flat["hf_token_env"] = hf.get("token_env", "HF_TOKEN")

    # Logging
    flat["log_every_n_steps"] = logging_cfg.get("log_every_n_steps", 20)

    # Protocol fields (set per-stage, not from YAML)
    flat.setdefault("protocol", "holdout")
    flat.setdefault("protocol_manifest_dir", None)
    flat.setdefault("protocol_reference_sha256", None)
    flat.setdefault("protocol_role", None)
    flat.setdefault("evaluation_scope", "development")
    flat.setdefault("suite_trial_id", None)
    flat.setdefault("expected_dataset_semantic_sha256", None)

    # Apply stage-specific overrides
    if stage is not None:
        stage_cfg = config.get("stages", {}).get(str(stage), {})
        # Direct key overrides (non-nested simple keys)
        for key, value in stage_cfg.items():
            if key == "trials":
                continue  # Trials are handled separately
            if key == "fine_loss" and isinstance(value, dict):
                if "name" in value:
                    flat["fine_loss"] = value["name"]
                if "inference_calibration_mode" in value:
                    flat["fine_inference_calibration_mode"] = value["inference_calibration_mode"]
                for sub_k, sub_v in value.items():
                    if sub_k not in {"name", "inference_calibration_mode"}:
                        if sub_k in flat:
                            flat[sub_k] = sub_v
                        elif f"fine_{sub_k}" in flat:
                            flat[f"fine_{sub_k}"] = sub_v
                continue
            if key in flat:
                flat[key] = value

    return flat


def load_config(
    *,
    config_path: Path | str | None = None,
    profile: str | None = None,
    stage: str | None = None,
    cli_overrides: list[str] | None = None,
) -> dict:
    """Load, merge, and flatten the pipeline config.

    Returns a flat dict compatible with ``cystods_core`` functions.
    """
    raw = load_yaml(config_path)

    # Determine profile
    effective_profile = profile
    if effective_profile is None:
        effective_profile = os.environ.get("CYSTODS_RUN_PROFILE")
    if effective_profile is None:
        effective_profile = "research"

    # Apply layers
    merged = apply_profile(raw, effective_profile)
    merged = apply_env_overrides(merged)
    merged = apply_cli_overrides(merged, cli_overrides)
    merged = resolve_paths(merged)

    # Flatten
    flat = flatten_to_core_config(merged, stage=stage)

    # Inject profile and stage
    flat["run_profile"] = effective_profile
    if stage is not None:
        flat["stage_name"] = f"stage_{stage}"
        flat["experiment_name"] = f"stage_{stage}"

    return flat


def filter_stage_trials(
    trials: list[dict],
    *,
    filter_models: list[str] | None = None,
    filter_trials: list[str] | None = None,
    default_model_name: str | None = None,
) -> list[dict]:
    """Filter trial definitions by model backbone names/aliases and/or trial experiment IDs."""
    if not filter_models and not filter_trials:
        return list(trials)

    filtered = []
    for trial in trials:
        exp_id = str(trial.get("experiment_id", ""))
        overrides = trial.get("overrides", {})
        model_name = str(overrides.get("model_name") or default_model_name or "")

        match_model = True
        if filter_models:
            match_model = any(
                m.lower() in model_name.lower() or m.lower() in exp_id.lower()
                for m in filter_models
            )

        match_trial = True
        if filter_trials:
            match_trial = any(
                t.lower() in exp_id.lower()
                for t in filter_trials
            )

        if match_model and match_trial:
            filtered.append(trial)

    return filtered


def get_stage_trials(
    config_path: Path | str | None = None,
    stage: str | None = None,
    profile: str | None = None,
    filter_models: list[str] | None = None,
    filter_trials: list[str] | None = None,
) -> list[dict]:
    """Return the trial definitions for a given stage, optionally filtered."""
    raw = load_yaml(config_path)
    effective_profile = profile or os.environ.get("CYSTODS_RUN_PROFILE", "research")
    merged = apply_profile(raw, effective_profile)
    stages = merged.get("stages", {})
    stage_cfg = stages.get(str(stage), {})
    trials = stage_cfg.get("trials", [])
    default_model = (
        merged.get("model", {}).get("name")
        if isinstance(merged.get("model"), dict)
        else None
    )
    return filter_stage_trials(
        trials,
        filter_models=filter_models,
        filter_trials=filter_trials,
        default_model_name=default_model,
    )


def show_config(
    *,
    config_path: Path | str | None = None,
    profile: str | None = None,
    stage: str | None = None,
    cli_overrides: list[str] | None = None,
) -> str:
    """Return a human-readable YAML representation of the resolved config."""
    flat = load_config(
        config_path=config_path,
        profile=profile,
        stage=stage,
        cli_overrides=cli_overrides,
    )
    # Convert Path objects for YAML serialization
    serializable = {}
    for key, value in sorted(flat.items()):
        if isinstance(value, Path):
            serializable[key] = str(value)
        elif isinstance(value, frozenset):
            serializable[key] = sorted(value)
        else:
            serializable[key] = value
    return yaml.dump(serializable, default_flow_style=False, sort_keys=True)
