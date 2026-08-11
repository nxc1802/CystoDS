"""Stage 90 — Cross-validation: 5-fold × 3 seeds final report.

Thin orchestrator that delegates to ``cystods.core.run_training_suite``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cystods.core as core
from cystods.config import get_stage_trials


STAGE_ID = "90"
STAGE_NAME = "stage_90_run_cross_validation_and_report"


def _source_files() -> tuple[Path, ...]:
    pkg_dir = Path(__file__).resolve().parent.parent
    return tuple(
        path for path in (pkg_dir / "core.py", pkg_dir / "hf.py", pkg_dir / "science.py")
        if path.is_file()
    )


def run(config: dict[str, Any]) -> Path:
    """Execute Stage 90 with the given resolved config."""
    config = dict(config)
    config["stage_name"] = STAGE_NAME
    config["experiment_name"] = STAGE_NAME
    config["evaluation_scope"] = "final_cv"

    # Cross-validation defaults
    if config.get("run_profile") != "smoke":
        config["protocol"] = "cross_validation"
        config["torch_compile"] = True
        config["torch_compile_mode"] = "max-autotune"

    protocol_run_dir = config.get("protocol_manifest_dir")
    if protocol_run_dir is None:
        env_val = os.environ.get("CYSTODS_PROTOCOL_RUN_DIR")
        if env_val:
            protocol_run_dir = Path(env_val).expanduser().resolve()

    expected_sha = config.get("expected_protocol_sha256") or os.environ.get(
        "CYSTODS_EXPECTED_PROTOCOL_SHA256"
    )

    config["hf_path_prefix"] = os.environ.get(
        "CYSTODS_HF_PATH_PREFIX", f"{config['study_id']}/{STAGE_NAME}"
    )

    config_path = config.pop("_config_path", None)
    trials = get_stage_trials(config_path=config_path, stage=STAGE_ID, profile=config.get("run_profile"))

    if not trials:
        if config.get("run_profile") == "smoke":
            trials = [
                {
                    "experiment_id": "smoke_cv_proposed_swin_tiny",
                    "task_mode": "hierarchical",
                    "overrides": {
                        "pretrained": False,
                        "supervised_contrastive_loss_weight": 0.0,
                        "monitor_metric": "coarse_macro_f1",
                    },
                },
            ]
        else:
            trials = [
                {
                    "experiment_id": "cv_multitask_ce_baseline",
                    "task_mode": "multitask",
                    "overrides": {
                        "fine_loss": "cross_entropy",
                        "binary_coarse_hierarchy_loss_weight": 0.0,
                        "coarse_fine_hierarchy_loss_weight": 0.0,
                        "supervised_contrastive_loss_weight": 0.0,
                        "fine_inference_calibration_mode": "fixed",
                        "monitor_metric": "hierarchical_composite",
                    },
                },
                {
                    "experiment_id": "cv_proposed_hierarchical",
                    "task_mode": "hierarchical",
                },
            ]

    # Try loading selected backbone & long-tail method from Stage 10 & 20 artifacts
    from cystods.experiments.artifacts import find_and_load_stage_artifact

    selected_backbone = "swin_tiny_patch4_window7_224.ms_in1k"
    selected_long_tail = "balanced_softmax"

    try:
        s10_artifact = find_and_load_stage_artifact(
            config["result_root"],
            stage_id="10",
            artifact_name="selected_backbone.json",
            expected_protocol_sha256=expected_sha,
        )
        selected_backbone = s10_artifact.get("selected_backbone", selected_backbone)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[Stage 90] Notice: Could not load Stage 10 artifact ({exc}). Defaulting backbone to {selected_backbone}.")

    try:
        s20_artifact = find_and_load_stage_artifact(
            config["result_root"],
            stage_id="20",
            artifact_name="selected_long_tail_method.json",
            expected_protocol_sha256=expected_sha,
        )
        selected_long_tail = s20_artifact.get("selected_long_tail_method", selected_long_tail)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[Stage 90] Notice: Could not load Stage 20 artifact ({exc}). Defaulting long-tail to {selected_long_tail}.")

    config["model_name"] = selected_backbone
    config["fine_loss"] = selected_long_tail
    for t in trials:
        t.setdefault("overrides", {})["model_name"] = selected_backbone

    # Seeds and fold config
    seeds = (
        (config["seed"],)
        if config.get("run_profile") == "smoke"
        else (20260729, 20260730, 20260731)
    )
    fold_ids = (0,) if config.get("run_profile") == "smoke" else None

    suite_config = {
        "schema_version": "cystods.stage.v2",
        "stage_name": STAGE_NAME,
        "study_id": config["study_id"],
        "run_profile": config["run_profile"],
        "data_root": config["data_root"],
        "result_root": config["result_root"],
        "protocol_run_dir": protocol_run_dir,
        "protocol_role": "final_cv",
        "expected_protocol_sha256": expected_sha,
        "seeds": seeds,
        "fold_ids": fold_ids,
        "base_config": config,
        "trials": tuple(trials),
        "evaluation_scope": "final_cv",
    }

    # Auto-discover protocol run if not provided
    if suite_config["protocol_run_dir"] is None:
        auto_dir, auto_sha = core.find_latest_completed_protocol_run(
            config.get("result_root"), config.get("run_profile")
        )
        if auto_dir is not None:
            suite_config["protocol_run_dir"] = auto_dir
            if suite_config.get("expected_protocol_sha256") is None:
                suite_config["expected_protocol_sha256"] = auto_sha
        else:
            raise RuntimeError(
                "Set CYSTODS_PROTOCOL_RUN_DIR to the completed Stage 00 run."
            )

    return core.run_training_suite(suite_config, _source_files())
