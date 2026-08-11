"""Stage 40 — Ablation studies: 16 component ablations.

Thin orchestrator that delegates to ``cystods.core.run_training_suite``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cystods.core as core
from cystods.config import get_stage_trials


STAGE_ID = "40"
STAGE_NAME = "stage_40_run_ablations"


def _source_files() -> tuple[Path, ...]:
    pkg_dir = Path(__file__).resolve().parent.parent
    return tuple(
        path for path in (pkg_dir / "core.py", pkg_dir / "hf.py", pkg_dir / "science.py")
        if path.is_file()
    )


def run(config: dict[str, Any]) -> Path:
    """Execute Stage 40 with the given resolved config."""
    config = dict(config)
    config["stage_name"] = STAGE_NAME
    config["experiment_name"] = STAGE_NAME
    config["evaluation_scope"] = "development"

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

    if not trials and config.get("run_profile") == "smoke":
        trials = [
            {
                "experiment_id": "smoke_ablation_no_supcon",
                "task_mode": "hierarchical",
                "overrides": {
                    "pretrained": False,
                    "supervised_contrastive_loss_weight": 0.0,
                    "monitor_metric": "coarse_macro_f1",
                },
            },
        ]

    # Try loading selected backbone & long-tail method from Stage 10 & 20 artifacts
    from cystods.experiments.artifacts import (
        find_and_load_stage_artifact,
        write_stage_selection_artifact,
    )

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
        print(f"[Stage 40] Notice: Could not load Stage 10 artifact ({exc}). Defaulting backbone to {selected_backbone}.")

    try:
        s20_artifact = find_and_load_stage_artifact(
            config["result_root"],
            stage_id="20",
            artifact_name="selected_long_tail_method.json",
            expected_protocol_sha256=expected_sha,
        )
        selected_long_tail = s20_artifact.get("selected_long_tail_method", selected_long_tail)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[Stage 40] Notice: Could not load Stage 20 artifact ({exc}). Defaulting long-tail to {selected_long_tail}.")

    config["model_name"] = selected_backbone
    config["fine_loss"] = selected_long_tail
    for t in trials:
        t.setdefault("overrides", {}).setdefault("model_name", selected_backbone)

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

    protocol_sha = config.get("protocol_reference_sha256", expected_sha)
    write_stage_selection_artifact(
        run_dir,
        "ablation_summary.json",
        {
            "stage_id": STAGE_ID,
            "selected_backbone": selected_backbone,
            "selected_long_tail_method": selected_long_tail,
            "num_trials_executed": len(trials),
            "protocol_sha256": protocol_sha,
            "study_id": config["study_id"],
        },
    )

    return run_dir
