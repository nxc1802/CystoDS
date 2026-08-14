"""Stage 20 — Long-tail loss screen: 7 fine-only loss variants on Swin-Tiny.

Thin orchestrator that delegates to ``cystods.core.run_training_suite``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cystods.core as core
from cystods.config import filter_stage_trials, get_stage_trials


STAGE_ID = "20"
STAGE_NAME = "stage_20_run_long_tail_screen"


def _source_files() -> tuple[Path, ...]:
    pkg_dir = Path(__file__).resolve().parent.parent
    return tuple(
        path for path in (pkg_dir / "core.py", pkg_dir / "hf.py", pkg_dir / "science.py")
        if path.is_file()
    )


def run(config: dict[str, Any]) -> Path:
    """Execute Stage 20 with the given resolved config."""
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
    )    # Protocol binding auto-discovery
    if protocol_run_dir is None:
        auto_dir, auto_sha = core.find_latest_completed_protocol_run(
            config.get("result_root"), config.get("run_profile")
        )
        if auto_dir is not None:
            protocol_run_dir = auto_dir
            if expected_sha is None:
                expected_sha = auto_sha

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

    if not trials and config.get("run_profile") == "smoke":
        trials = [
            {
                "experiment_id": "smoke_fine_focal",
                "task_mode": "fine",
                "overrides": {
                    "fine_loss": "focal",
                    "pretrained": False,
                    "supervised_contrastive_loss_weight": 0.0,
                    "monitor_metric": "fine_macro_f1",
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

    # Try loading selected backbone from Stage 10 artifact
    from cystods.experiments.artifacts import (
        find_and_load_stage_artifact,
        write_stage_selection_artifact,
    )

    selected_backbone = "swin_tiny_patch4_window7_224.ms_in1k"
    try:
        stage10_artifact = find_and_load_stage_artifact(
            config["result_root"],
            stage_id="10",
            artifact_name="selected_backbone.json",
            expected_protocol_sha256=expected_sha,
            expected_split_index=config.get("protocol_split_index"),
        )
        selected_backbone = stage10_artifact.get("selected_backbone", selected_backbone)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[Stage 20] Notice: Could not load Stage 10 artifact ({exc}). Defaulting backbone to {selected_backbone}.")

    # Apply selected backbone across trials if not overridden
    config["model_name"] = selected_backbone
    for t in trials:
        t.setdefault("overrides", {})["model_name"] = selected_backbone

    print(f"Stage {STAGE_ID}: Selected {len(trials)} trial(s) to run (backbone={selected_backbone}):")
    for t in trials:
        m_name = t.get("overrides", {}).get("model_name", selected_backbone)
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

    # Evaluate all completed trials to select the winning long-tail loss method
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

    runs_dir = run_dir / "runs"
    selected_method = "balanced_softmax"
    selected_trial_id = None
    best_val_score = -1.0
    screening_benchmark: list[dict[str, Any]] = []

    if runs_dir.is_dir():
        import pandas as pd
        for trial_folder in sorted(runs_dir.iterdir()):
            if not trial_folder.is_dir():
                continue
            trial_id = trial_folder.name
            loss_method = trial_id.replace("fine_", "")

            # Look for history.csv or best_model.pt in subdirectories of trial_folder
            val_scores: list[float] = []
            for fold_dir in sorted(trial_folder.iterdir()):
                if not fold_dir.is_dir():
                    continue
                history_file = fold_dir / "history.csv"
                if history_file.is_file():
                    try:
                        hdf = pd.read_csv(history_file)
                        if "monitored_score" in hdf:
                            val_scores.append(float(hdf["monitored_score"].max()))
                    except Exception:
                        pass

            val_score = float(max(val_scores)) if val_scores else 0.0
            entry = {
                "trial_id": trial_id,
                "loss_method": loss_method,
                "val_monitored_score": val_score,
            }
            screening_benchmark.append(entry)

            if val_score > best_val_score:
                best_val_score = val_score
                selected_method = loss_method
                selected_trial_id = trial_id

    # If only 1 trial ran or none had history, default appropriately
    if selected_trial_id is None and screening_benchmark:
        selected_trial_id = screening_benchmark[0]["trial_id"]
        selected_method = screening_benchmark[0]["loss_method"]

    # Write full screening benchmark report
    if screening_benchmark:
        from cystods.infra.serialization import write_json
        write_json(
            run_dir / "loss_screening_benchmark.json",
            {
                "schema_version": "cystods.loss_screening_benchmark.v1",
                "stage_id": STAGE_ID,
                "selected_backbone": selected_backbone,
                "selected_long_tail_method": selected_method,
                "selected_trial_id": selected_trial_id,
                "best_val_monitored_score": best_val_score,
                "evaluated_trials_count": len(screening_benchmark),
                "trials": screening_benchmark,
            },
        )
        print(f"\n[Stage 20] Loss Screening Benchmark completed across {len(screening_benchmark)} trial(s):")
        for b in screening_benchmark:
            flag = " 🏆 (Winner)" if b["trial_id"] == selected_trial_id else ""
            print(f"  • [{b['trial_id']}] Loss: {b['loss_method']} | Val Monitored Score: {b['val_monitored_score']:.4f}{flag}")
        print()

    # Save transition artifact for Stage 30
    write_stage_selection_artifact(
        run_dir,
        "selected_long_tail_method.json",
        {
            "stage_id": STAGE_ID,
            "selected_backbone": selected_backbone,
            "selected_long_tail_method": selected_method,
            "selected_trial_id": selected_trial_id,
            "selection_metric": "primary_macro_f1_all_classes",
            "val_macro_f1": best_val_score if best_val_score >= 0 else None,
            "protocol_sha256": protocol_sha,
            "protocol_split_index": protocol_split_index,
            "study_id": config["study_id"],
        },
    )

    return run_dir
