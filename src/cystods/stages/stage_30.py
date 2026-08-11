"""Stage 30 — Proposed method: hierarchical + balanced-softmax + SupCon.

Thin orchestrator that delegates to ``cystods.core.run_training_suite``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cystods.core as core
from cystods.config import get_stage_trials


STAGE_ID = "30"
STAGE_NAME = "stage_30_run_proposed_method"


def _source_files() -> tuple[Path, ...]:
    pkg_dir = Path(__file__).resolve().parent.parent
    return tuple(
        path for path in (pkg_dir / "core.py", pkg_dir / "hf.py", pkg_dir / "science.py")
        if path.is_file()
    )


def run(config: dict[str, Any]) -> Path:
    """Execute Stage 30 with the given resolved config."""
    config = dict(config)
    config["stage_name"] = STAGE_NAME
    config["experiment_name"] = STAGE_NAME
    config["evaluation_scope"] = "development"

    # Apply PROPOSED_CANONICAL_CONFIG from core
    config.update(core.PROPOSED_CANONICAL_CONFIG)

    # Restore runtime overrides from environment
    config["batch_size"] = int(
        os.environ.get("CYSTODS_BATCH_SIZE", str(core.PROPOSED_CANONICAL_CONFIG["batch_size"]))
    )

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
        trials = [
            {
                "experiment_id": "proposed_hierarchical_swin",
                "task_mode": "hierarchical",
                "overrides": dict(core.PROPOSED_CANONICAL_CONFIG),
            },
        ]

    if config.get("run_profile") == "smoke":
        trials = [
            {
                "experiment_id": "smoke_proposed_swin_tiny",
                "task_mode": "hierarchical",
                "overrides": {
                    "pretrained": False,
                    "supervised_contrastive_loss_weight": 0.0,
                    "monitor_metric": "coarse_macro_f1",
                },
            },
        ]

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

    return core.run_training_suite(suite_config, _source_files())
