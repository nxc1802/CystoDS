"""Model name resolution and task-mode helpers.

Extracted from ``cystods.core`` (Step 2 refactor).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import timm

from cystods.taxonomy import PAPER_BASELINE_BACKBONES


def resolve_model_name(model_name: str) -> str:
    cleaned = str(model_name).strip()
    key = cleaned.lower()
    if key in PAPER_BASELINE_BACKBONES:
        return PAPER_BASELINE_BACKBONES[key]
    return cleaned


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


def validate_config(config: Mapping[str, Any]) -> None:
    from cystods.core import BASE_CONFIG
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
            f"(resolved as '{resolved_name}')."
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
            "supervised_contrastive_loss_weight",
        ),
        "fine": (
            "binary_loss_weight",
            "coarse_loss_weight",
            "binary_coarse_hierarchy_loss_weight",
            "coarse_fine_hierarchy_loss_weight",
            "supervised_contrastive_loss_weight",
        ),
    }
    for weight_key in inactive_weight_keys.get(str(config["task_mode"]), ()):
        if float(config[weight_key]) != 0:
            raise ValueError(
                f"task_mode='{config['task_mode']}' requires zero inactive weights."
            )
    fixed_primary = config["fixed_primary_fine_class_ids"]
    if fixed_primary is not None:
        fixed_tuple = tuple(fixed_primary)
        if any(
            isinstance(item, (bool, np.bool_))
            or not isinstance(item, (int, np.integer))
            or not 0 <= int(item) < 22
            for item in fixed_tuple
        ):
            raise ValueError(
                "fixed_primary_fine_class_ids must contain integers in [0, 21]."
            )
    checkpoint_delta = float(config["checkpoint_min_delta"])
    if not math.isfinite(checkpoint_delta) or checkpoint_delta < 0:
        raise ValueError("checkpoint_min_delta must be finite and non-negative.")
    prob_tol = float(config["probability_sum_tolerance"])
    if not math.isfinite(prob_tol) or prob_tol <= 0:
        raise ValueError("probability_sum_tolerance must be positive and finite.")
