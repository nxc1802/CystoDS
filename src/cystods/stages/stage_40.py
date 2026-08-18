"""Stage 40 — Comprehensive Ablation Studies: 8 component, paradigm, strategy & freezing variants.

Orchestrates:
- 1-Stage ablations (Joint baseline, w/o SupCon, w/o Hierarchy, Layer Freezing) via ``cystods.core.run_training_suite``
- 2-Stage ablations (2-Stage Decoupled, cRT strategy, All-Heads alignment) via ``cystods.experiments.two_stage_runner``
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any

import torch

import cystods.core as core
from cystods.config import filter_stage_trials, get_stage_trials
from cystods.experiments.artifacts import (
    find_and_load_stage_artifact,
    write_stage_selection_artifact,
)
from cystods.experiments.two_stage_runner import run_two_stage_single_split

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

    if not trials and (filter_models or filter_trials):
        all_trials = get_stage_trials(config_path=config_path, stage=STAGE_ID, profile=config.get("run_profile"))
        avail_exp = [t.get("experiment_id") for t in all_trials]
        avail_models = sorted({t.get("overrides", {}).get("model_name", "default") for t in all_trials})
        raise RuntimeError(
            f"No trials matched the filters: models={filter_models}, trials={filter_trials}.\n"
            f"Available models for Stage {STAGE_ID}: {avail_models}\n"
            f"Available experiment IDs for Stage {STAGE_ID}: {avail_exp}"
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

    # Split trials into 1-stage suite trials and 2-stage decoupled trials
    one_stage_trials: list[dict[str, Any]] = []
    two_stage_trials: list[dict[str, Any]] = []

    for t in trials:
        t_overrides = t.setdefault("overrides", {})
        t_overrides.setdefault("model_name", selected_backbone)
        t_overrides.setdefault("fine_loss", selected_long_tail)
        if t_overrides.get("training_paradigm") == "two_stage":
            two_stage_trials.append(t)
        else:
            one_stage_trials.append(t)

    print(f"\nStage {STAGE_ID}: Executing {len(trials)} total ablation trial(s):")
    print(f"  • Single-Stage Suite Trials ({len(one_stage_trials)}): {[t['experiment_id'] for t in one_stage_trials]}")
    print(f"  • Two-Stage Decoupled Trials ({len(two_stage_trials)}): {[t['experiment_id'] for t in two_stage_trials]}\n")

    run_dir: Path | None = None

    # 1. Execute single-stage trials via core.run_training_suite
    if one_stage_trials:
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
            "trials": tuple(one_stage_trials),
        }
        source_files = _source_files()
        run_dir = core.run_training_suite(suite_config, source_files)

    # 2. Execute two-stage trials via two_stage_runner
    if two_stage_trials:
        split_idx = config.get("protocol_split_index", 0)
        profile = config.get("run_profile", "research")
        result_root = Path(config.get("result_root", "./result")).resolve()
        
        for t in two_stage_trials:
            t_id = t["experiment_id"]
            t_ov = t.get("overrides", {})
            p1_eps = 1 if profile == "smoke" else int(t_ov.get("phase1_epochs", 25))
            p2_eps = 1 if profile == "smoke" else int(t_ov.get("phase2_epochs", 10))
            p1_loss = str(t_ov.get("phase1_loss", "cross_entropy"))
            p2_loss = str(t_ov.get("phase2_loss", selected_long_tail))
            p2_strat = str(t_ov.get("phase2_strategy", "linear_probe"))
            p2_tgt = str(t_ov.get("phase2_target", "fine_only"))
            sup_w = 0.0 if profile == "smoke" else float(t_ov.get("phase1_supcon_weight", 0.10))

            print(f"\n▶ Running Two-Stage Ablation Trial: {t_id} (Split {split_idx}, Target={p2_tgt}, Strategy={p2_strat})")
            run_two_stage_single_split(
                split_index=int(split_idx) if split_idx is not None else 0,
                base_config=config,
                profile=profile,
                phase1_epochs=p1_eps,
                phase2_epochs=p2_eps,
                phase1_loss=p1_loss,
                phase2_loss=p2_loss,
                phase2_strategy=p2_strat,
                phase1_lr=float(t_ov.get("phase1_lr", 0.0003)),
                phase2_lr=float(t_ov.get("phase2_lr", 0.001)),
                phase1_supcon_weight=sup_w,
                phase2_target=p2_tgt,
                ablation_name=t_id,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if run_dir is None:
        run_dir = Path(config["result_root"]) / "40_ablations"

    protocol_sha = config.get("protocol_reference_sha256", expected_sha)
    protocol_split_index = config.get("protocol_split_index")

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
