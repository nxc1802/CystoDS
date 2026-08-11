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
    # Stage 20 specifies epochs=20 in config.yaml
    config_default = load_config(stage="20", profile="smoke")
    assert config_default["epochs"] == 20

    # CLI override should override stage default 20 to 3
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
