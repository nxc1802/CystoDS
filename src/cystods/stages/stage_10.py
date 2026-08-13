"""Stage 10 — Run baselines: 4 backbones × binary/coarse/fine/multitask.

Thin orchestrator that delegates to ``cystods.core.run_training_suite``.
Trial definitions come from ``config.yaml`` under ``stages."10".trials``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cystods.core as core
from cystods.config import filter_stage_trials, get_stage_trials


STAGE_ID = "10"
STAGE_NAME = "stage_10_run_baselines"


def _source_files() -> tuple[Path, ...]:
    pkg_dir = Path(__file__).resolve().parent.parent
    return tuple(
        path
        for path in (
            pkg_dir / "core.py",
            pkg_dir / "hf.py",
            pkg_dir / "science.py",
        )
        if path.is_file()
    )


def run(config: dict[str, Any]) -> Path:
    """Execute Stage 10 with the given resolved config."""
    config = dict(config)
    config["stage_name"] = STAGE_NAME
    config["experiment_name"] = STAGE_NAME
    config["evaluation_scope"] = "development"

    # Protocol binding
    protocol_run_dir = config.get("protocol_manifest_dir")
    if protocol_run_dir is None:
        env_val = os.environ.get("CYSTODS_PROTOCOL_RUN_DIR")
        if env_val:
            protocol_run_dir = Path(env_val).expanduser().resolve()

    expected_sha = config.get("expected_protocol_sha256") or os.environ.get(
        "CYSTODS_EXPECTED_PROTOCOL_SHA256"
    )

    # HF config
    hf_path_prefix = os.environ.get(
        "CYSTODS_HF_PATH_PREFIX",
        f"{config['study_id']}/{STAGE_NAME}",
    )
    config["hf_path_prefix"] = hf_path_prefix

    # Load trials from config.yaml with filters
    filter_models = config.get("filter_models")
    filter_trials = config.get("filter_trials")

    config_path = config.pop("_config_path", None)
    trials = get_stage_trials(
        config_path=config_path,
        stage=STAGE_ID,
        profile=config.get("run_profile"),
        filter_models=filter_models,
        filter_trials=filter_trials,
    )

    if not trials:
        # Fallback smoke trial
        if config.get("run_profile") == "smoke":
            trials = [
                {
                    "experiment_id": "smoke_binary_swin_tiny",
                    "task_mode": "binary",
                    "overrides": {
                        "model_name": "swin_tiny_patch4_window7_224.ms_in1k",
                        "pretrained": False,
                        "supervised_contrastive_loss_weight": 0.0,
                        "monitor_metric": "binary_auroc",
                    },
                },
            ]
            trials = filter_stage_trials(
                trials,
                filter_models=filter_models,
                filter_trials=filter_trials,
            )

    if not trials:
        all_trials = get_stage_trials(config_path=config_path, stage=STAGE_ID, profile=config.get("run_profile"))
        avail_exp = [t.get("experiment_id") for t in all_trials]
        avail_models = sorted({t.get("overrides", {}).get("model_name", "default") for t in all_trials})
        raise RuntimeError(
            f"No trials matched the filters: models={filter_models}, trials={filter_trials}.\n"
            f"Available models for Stage {STAGE_ID}: {avail_models}\n"
            f"Available experiment IDs for Stage {STAGE_ID}: {avail_exp}"
        )

    print(f"Stage {STAGE_ID}: Selected {len(trials)} trial(s) to run:")
    for t in trials:
        m_name = t.get("overrides", {}).get("model_name", "default")
        print(f"  • [{t['experiment_id']}] model: {m_name} | task_mode: {t.get('task_mode')}")
    print()

    suite_config = {
        "schema_version": "cystods.stage.v2",
        "stage_name": STAGE_NAME,
        "study_id": config["study_id"],
        "run_profile": config["run_profile"],
        "data_root": config["data_root"],
        "result_root": config["result_root"],
        "protocol_run_dir": protocol_run_dir,
        "protocol_role": "fixed_holdout",
        "evaluation_scope": config.get("evaluation_scope", "development"),
        "expected_protocol_sha256": expected_sha,
        "seeds": (config["seed"],),
        "fold_ids": None,
        "base_config": config,
        "trials": tuple(trials),
    }

    source_files = _source_files()
    run_dir = core.run_training_suite(suite_config, source_files)

    # Save transition artifact for Stage 20 / Stage 30
    from cystods.experiments.artifacts import write_stage_selection_artifact

    # Default to Swin-Tiny unless explicit override / analysis changes it
    selected_model = "swin_tiny_patch4_window7_224.ms_in1k"
    protocol_sha = config.get("protocol_reference_sha256", expected_sha)

    write_stage_selection_artifact(
        run_dir,
        "selected_backbone.json",
        {
            "stage_id": STAGE_ID,
            "selected_backbone": selected_model,
            "selection_metric": "hierarchical_composite",
            "protocol_sha256": protocol_sha,
            "protocol_split_index": config.get("protocol_split_index"),
            "study_id": config["study_id"],
        },
    )

    return run_dir
