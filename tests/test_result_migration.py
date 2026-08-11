"""Unit tests for ResultStore and result migration functionality."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cystods.migrate import migrate_result_directory
from cystods.store import ResultStore, read_json_file, write_json_file


def test_result_store_paths(tmp_path: Path):
    store = ResultStore(tmp_path)
    assert store.trial_dir("trial_1") == tmp_path / "runs" / "trial_1"
    assert store.seed_dir("trial_1", 2026) == tmp_path / "runs" / "trial_1" / "seed_2026"
    assert store.fold_dir("fold_0", trial_id="trial_1", seed=2026) == (
        tmp_path / "runs" / "trial_1" / "seed_2026" / "folds" / "fold_0"
    )
    assert store.fold_dir("holdout") == tmp_path / "folds" / "holdout"


def test_result_store_metrics_consolidation(tmp_path: Path):
    store = ResultStore(tmp_path)
    fold_dir = store.fold_dir("holdout")

    metrics_file = store.write_fold_metrics(
        fold_dir=fold_dir,
        train_metrics={"acc": 0.95},
        test_metrics={"acc": 0.92, "fine_macro_f1": 0.65},
        test_losses={"total_loss": 0.12},
        bootstrap={"ci_low": 0.88, "ci_high": 0.96},
    )

    assert metrics_file.is_file()
    data = read_json_file(metrics_file)
    assert data["schema_version"] == "cystods.fold_metrics"
    assert data["train"]["metrics"]["acc"] == 0.95
    assert data["test"]["metrics"]["fine_macro_f1"] == 0.65
    assert data["bootstrap"]["ci_low"] == 0.88


def test_result_migration_flow(tmp_path: Path):
    # Set up legacy result folder structure
    old_run = tmp_path / "stage_10_run_baselines_research_20260811-014803"
    old_run.mkdir(parents=True)

    # Legacy fold metrics
    holdout_metrics = old_run / "metrics" / "holdout"
    holdout_metrics.mkdir(parents=True)
    write_json_file(holdout_metrics / "train_metrics.json", {"auroc": 0.99})
    write_json_file(holdout_metrics / "test_metrics.json", {"auroc": 0.94, "macro_f1": 0.72})
    write_json_file(holdout_metrics / "test_losses.json", {"loss": 0.15})

    # Legacy predictions
    holdout_preds = old_run / "predictions" / "holdout"
    holdout_preds.mkdir(parents=True)
    (holdout_preds / "test_image_predictions.csv").write_text("image,pred\nimg1.jpg,0\n")

    # Legacy splits
    holdout_splits = old_run / "splits" / "holdout"
    holdout_splits.mkdir(parents=True)
    (holdout_splits / "test.csv").write_text("image,label\nimg1.jpg,0\n")

    # Migrate legacy directory
    migrated_dir = migrate_result_directory(old_run, tmp_path / "result")

    assert migrated_dir.is_dir()
    assert migrated_dir == tmp_path / "result" / "10_baselines" / "20260811-014803"

    fold_dir = migrated_dir / "folds" / "holdout"
    assert fold_dir.is_dir()

    metrics_p = fold_dir / "metrics.json"
    assert metrics_p.is_file()
    m_data = read_json_file(metrics_p)
    assert m_data["test"]["metrics"]["auroc"] == 0.94

    assert (fold_dir / "test_image_predictions.csv").is_file()
    assert (fold_dir / "split" / "test.csv").is_file()
    assert (migrated_dir / "catalog.json").is_file()
