"""Stage 40 — Ablation studies: 16 component ablations.

Thin orchestrator that delegates to ``cystods.core.run_training_suite``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cystods.core as core
from cystods.config import filter_stage_trials, get_stage_trials


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

    filter_models = config.get("filter_models")
    filter_trials = config.get("filter_trials")

    # Protocol binding auto-discovery
    if protocol_run_dir is None:
        auto_dir, auto_sha = core.find_latest_completed_protocol_run(
            config.get("result_root"), config.get("run_profile")
        )
        if auto_dir is not None:
            protocol_run_dir = auto_dir
            if expected_sha is None:
                expected_sha = auto_sha

    config_path = config.pop("_config_path", None)
    trials = get_stage_trials(
        config_path=config_path,
        stage=STAGE_ID,
        profile=config.get("run_profile"),
        filter_models=filter_models,
        filter_trials=filter_trials,
    )

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
            {
                "experiment_id": "smoke_ablation_full_proposed",
                "task_mode": "hierarchical",
                "overrides": {
                    "pretrained": False,
                    "monitor_metric": "coarse_macro_f1",
                },
            },
        ]
        trials = filter_stage_trials(
            trials,
            filter_models=filter_models,
            filter_trials=filter_trials,
        )

    if not trials and (filter_models or filter_trials):
        all_trials = get_stage_trials(config_path=config_path, stage=STAGE_ID, profile=config.get("run_profile"))
        avail_exp = [t.get("experiment_id") for t in all_trials]
        avail_models = sorted({t.get("overrides", {}).get("model_name", "default") for t in all_trials})
        raise RuntimeError(
            f"No trials matched the filters: models={filter_models}, trials={filter_trials}.\n"
            f"Available models for Stage {STAGE_ID}: {avail_models}\n"
            f"Available experiment IDs for Stage {STAGE_ID}: {avail_exp}"
        )

    # Try loading selected backbone & long-tail method from Stage 10 & 20/30 artifacts
    from cystods.experiments.artifacts import (
        find_and_load_stage_artifact,
        write_stage_selection_artifact,
    )

    selected_backbone = "swin_tiny_patch4_window7_224.ms_in1k"
    selected_long_tail = "balanced_softmax_smoothed"

    try:
        s10_artifact = find_and_load_stage_artifact(
            config["result_root"],
            stage_id="10",
            artifact_name="selected_backbone.json",
            expected_protocol_sha256=expected_sha,
            expected_split_index=config.get("protocol_split_index"),
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
            expected_split_index=config.get("protocol_split_index"),
        )
        selected_long_tail = s20_artifact.get("selected_long_tail_method", selected_long_tail)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[Stage 40] Notice: Could not load Stage 20 artifact ({exc}). Defaulting long-tail to {selected_long_tail}.")

    config["model_name"] = selected_backbone
    config["fine_loss"] = selected_long_tail
    for t in trials:
        t.setdefault("overrides", {}).setdefault("model_name", selected_backbone)
        t.setdefault("overrides", {}).setdefault("fine_loss", selected_long_tail)

    print(f"Stage {STAGE_ID}: Selected {len(trials)} trial(s) to run (backbone={selected_backbone}, default_fine_loss={selected_long_tail}):")
    for t in trials:
        m_name = t.get("overrides", {}).get("model_name", selected_backbone)
        lt_name = t.get("overrides", {}).get("fine_loss", selected_long_tail)
        print(f"  • [{t['experiment_id']}] model: {m_name} | fine_loss: {lt_name} | task_mode: {t.get('task_mode')}")
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

    run_status_file = run_dir / "run_status.json"
    protocol_sha = config.get("protocol_reference_sha256", expected_sha)
    protocol_split_index = config.get("protocol_split_index")
    if run_status_file.is_file():
        try:
            import json
            with run_status_file.open("r", encoding="utf-8") as handle:
                status_data = json.load(handle)
            protocol_sha = status_data.get("protocol_sha256", protocol_sha)
            if protocol_split_index is None:
                protocol_split_index = status_data.get("protocol_split_index")
        except Exception:
            pass

    write_stage_selection_artifact(
        run_dir,
        "ablation_summary.json",
        {
            "stage_id": STAGE_ID,
            "selected_backbone": selected_backbone,
            "selected_long_tail_method": selected_long_tail,
            "num_trials_executed": len(trials),
            "protocol_sha256": protocol_sha,
            "protocol_split_index": protocol_split_index,
            "study_id": config["study_id"],
        },
    )

    return run_dir
