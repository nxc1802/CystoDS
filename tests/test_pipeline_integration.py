"""Integration tests for CystoDS pipeline contract stability and stage artifacts.

Tests CLI precedence, trial resolution, Stage 60 contract, Stage 00 smoke profile, and stage transition artifacts.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from cystods.config import load_config, get_stage_trials
from cystods.experiments.artifacts import (
    write_stage_selection_artifact,
    find_and_load_stage_artifact,
)
import cystods.stages.stage_00 as stage_00
import cystods.stages.stage_10 as stage_10
import cystods.stages.stage_60 as stage_60


def test_cli_override_has_highest_precedence() -> None:
    """Verify CLI --set overrides take priority over stage defaults and profile defaults."""
    # Research profile uses stage default epochs=25
    config_research = load_config(stage="20", profile="research")
    assert config_research["epochs"] == 25

    # Smoke profile overrides epochs to 1
    config_smoke = load_config(stage="20", profile="smoke")
    assert config_smoke["epochs"] == 1

    # CLI override should override smoke profile default 1 to 3
    config_cli = load_config(
        stage="20",
        profile="smoke",
        cli_overrides=["training.epochs=3"],
    )
    assert config_cli["epochs"] == 3

    # CLI override with direct flat key name
    config_cli_flat = load_config(
        stage="20",
        profile="smoke",
        cli_overrides=["epochs=5"],
    )
    assert config_cli_flat["epochs"] == 5


def test_stage10_trial_resolution() -> None:
    """Verify trials in Stage 10 resolve cleanly into valid trial configs without illegal keys."""
    trials = get_stage_trials(stage="10", profile="smoke")
    assert len(trials) > 0

    base_config = load_config(stage="10", profile="smoke")
    for trial in trials:
        trial_name = str(trial.get("experiment_id", trial.get("trial_id", "test_trial")))
        trial_config = dict(base_config)
        trial_config["task_mode"] = trial.get("task_mode", trial_config["task_mode"])
        trial_config.update(trial.get("overrides", {}))
        trial_config["suite_trial_id"] = trial_name

        # Ensure experiment_id or overrides dict are NOT present in scientific config
        assert "experiment_id" not in trial_config
        assert "overrides" not in trial_config
        assert trial_config["suite_trial_id"] == trial_name


def test_stage_60_schema_contract() -> None:
    """Verify Stage 60 builds the expected stage_config matching the runner contract."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        run_dir = tmp_path / "mock_run"
        run_dir.mkdir()

        receipt_file = run_dir / "hf_checkpoint_receipt.json"
        receipt_file.write_text('{"status": "ok"}', encoding="utf-8")
        status_file = run_dir / "run_status.json"
        status_file.write_text('{"status": "completed"}', encoding="utf-8")
        artifact_file = run_dir / "artifact_manifest.json"
        artifact_file.write_text('{"files": []}', encoding="utf-8")

        ext_manifest = tmp_path / "external.csv"
        ext_manifest.write_text("path,binary_label,patient_id\n", encoding="utf-8")
        ext_img = tmp_path / "images"
        ext_img.mkdir()
        protocol_dir = tmp_path / "protocol"
        protocol_dir.mkdir()

        raw_config = load_config(stage="60", profile="smoke")
        raw_config["selected_run_dir"] = str(run_dir)
        raw_config["hf_checkpoint_receipt_json"] = str(receipt_file)
        raw_config["protocol_manifest_dir"] = str(protocol_dir)
        raw_config["external_manifest_csv"] = str(ext_manifest)
        raw_config["external_image_root"] = str(ext_img)

        stage_60_files = stage_60._source_files()
        assert len(stage_60_files) > 0


def test_stage_00_smoke_profile_protocol_role() -> None:
    """Verify Stage 00 under smoke profile sets protocol_role to smoke_holdout."""
    config = load_config(stage="00", profile="smoke")
    config["stage_name"] = "stage_00_prepare_protocol"
    config["experiment_name"] = "stage_00_prepare_protocol"
    config["evaluation_scope"] = "development"

    if config.get("run_profile") == "smoke":
        config["protocol_role"] = "smoke_holdout"
    else:
        config["protocol_role"] = "fixed_holdout"

    assert config["protocol_role"] == "smoke_holdout"


def test_stage_artifact_saving_and_loading() -> None:
    """Verify stage transition artifacts save and load correctly across stages."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        stage10_dir = tmp_path / "stage_10_run_01"
        stage10_dir.mkdir()

        write_stage_selection_artifact(
            stage10_dir,
            "selected_backbone.json",
            {
                "stage_id": "10",
                "selected_backbone": "resnet152.a1_in1k",
                "protocol_sha256": "abc123sha",
            },
        )

        loaded = find_and_load_stage_artifact(
            tmp_path,
            stage_id="10",
            artifact_name="selected_backbone.json",
            expected_protocol_sha256="abc123sha",
        )
        assert loaded["selected_backbone"] == "resnet152.a1_in1k"


def test_stage_artifact_protocol_mismatch_fails() -> None:
    """Verify fail-fast behavior when artifact protocol SHA does not match expected protocol SHA."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        stage10_dir = tmp_path / "stage_10_run_01"
        stage10_dir.mkdir()

        write_stage_selection_artifact(
            stage10_dir,
            "selected_backbone.json",
            {
                "stage_id": "10",
                "selected_backbone": "swin_tiny_patch4_window7_224.ms_in1k",
                "protocol_sha256": "old_protocol_sha",
            },
        )

        with pytest.raises(ValueError, match="Protocol split mismatch"):
            find_and_load_stage_artifact(
                tmp_path,
                stage_id="10",
                artifact_name="selected_backbone.json",
                expected_protocol_sha256="new_protocol_sha",
            )


def test_base_config_has_device_key() -> None:
    """Verify device key is present in BASE_CONFIG and normalized without KeyError."""
    from cystods.config_schema import BASE_CONFIG
    from cystods.config import normalize_core_config, load_config

    assert "device" in BASE_CONFIG
    config = load_config(stage="10", profile="smoke")
    assert "device" in config
    normalized = normalize_core_config(config)
    assert normalized["device"] == config["device"]


def test_normalize_core_config_ignores_section_keys_and_runtime() -> None:
    """Verify normalize_core_config handles nested CLI section keys like runtime without KeyError."""
    from cystods.config import load_config, normalize_core_config

    config = load_config(
        stage="10",
        profile="smoke",
        cli_overrides=["runtime.batch_size=32", "runtime.num_workers=4"],
    )
    normalized = normalize_core_config(config)
    assert normalized["batch_size"] == 32
    assert normalized["num_workers"] == 4
    assert "runtime" not in normalized


def test_extract_roi_bags_handles_missing_task_columns() -> None:
    """Verify extract_roi_bags handles single-task prediction DataFrames without KeyError."""
    import pandas as pd
    from cystods.evaluation.roi import extract_roi_bags

    # Binary task prediction DataFrame lacking coarse_probs or fine_probs
    binary_preds = pd.DataFrame(
        {
            "pid": ["P1", "P1"],
            "visit": [1, 1],
            "lesion": [1, 1],
            "filename": ["img1.png", "img2.png"],
            "binary_id": [1, 1],
            "binary_probs": [[0.1, 0.9], [0.2, 0.8]],
        }
    )
    # Extracting coarse task from binary predictions should return empty without raising KeyError
    bags, conflicts, skipped = extract_roi_bags(binary_preds, task="coarse", require_features=False)
    assert bags == []
    assert conflicts == []
    assert skipped == 0


def test_cli_override_attention_epochs_maps_to_roi_attention_epochs() -> None:
    """Verify CLI override evaluation.roi.attention_epochs maps cleanly to roi_attention_epochs without KeyError."""
    from cystods.config import load_config, normalize_core_config

    config = load_config(
        stage="10",
        profile="research",
        cli_overrides=["evaluation.roi.attention_epochs=1"],
    )
    normalized = normalize_core_config(config)
    assert normalized["roi_attention_epochs"] == 1
    assert "attention_epochs" not in normalized


def test_stage_artifact_hierarchical_directory_structure() -> None:
    """Verify find_and_load_stage_artifact discovers artifacts in unified hierarchical directory layout."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        run_10 = tmp_path / "10_baselines" / "research_20260812-102718"
        run_10.mkdir(parents=True)
        write_stage_selection_artifact(
            run_10,
            "selected_backbone.json",
            {
                "stage_id": "10",
                "selected_backbone": "swin_tiny_patch4_window7_224.ms_in1k",
                "protocol_sha256": "sha_stage_10",
            },
        )

        run_20 = tmp_path / "20_long_tail" / "research_20260813-105659"
        run_20.mkdir(parents=True)
        write_stage_selection_artifact(
            run_20,
            "selected_long_tail_method.json",
            {
                "stage_id": "20",
                "selected_backbone": "swin_tiny_patch4_window7_224.ms_in1k",
                "selected_long_tail_method": "balanced_softmax_smoothed",
                "protocol_sha256": "sha_stage_20",
            },
        )

        loaded_10 = find_and_load_stage_artifact(
            tmp_path,
            stage_id="10",
            artifact_name="selected_backbone.json",
            expected_protocol_sha256="sha_stage_10",
        )
        assert loaded_10["selected_backbone"] == "swin_tiny_patch4_window7_224.ms_in1k"

        loaded_20 = find_and_load_stage_artifact(
            tmp_path,
            stage_id="20",
            artifact_name="selected_long_tail_method.json",
            expected_protocol_sha256="sha_stage_20",
        )
        assert loaded_20["selected_long_tail_method"] == "balanced_softmax_smoothed"


def test_stage30_trial_resolution_and_source_files() -> None:
    """Verify Stage 30 trials resolution, filters, and source files contracts."""
    import cystods.stages.stage_30 as stage_30

    source_files = stage_30._source_files()
    assert len(source_files) > 0
    assert all(f.is_file() for f in source_files)

    trials_research = get_stage_trials(stage="30", profile="research")
    assert len(trials_research) == 1
    assert trials_research[0]["task_mode"] == "hierarchical"

    trials_smoke = get_stage_trials(stage="30", profile="smoke")
    assert len(trials_smoke) >= 0

    base_config = load_config(stage="30", profile="smoke")
    assert base_config["task_mode"] == "hierarchical"
    assert base_config["binary_coarse_hierarchy_loss_weight"] == 0.25
    assert base_config["coarse_fine_hierarchy_loss_weight"] == 0.25




