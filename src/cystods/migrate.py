"""Data Migration Utility for CystoDS Results.

Refactors legacy result directory structures into the unified Stage -> Trial -> Seed -> Fold structure.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from cystods.store import ResultStore, read_json_file, write_json_file

logger = logging.getLogger(__name__)

STAGE_NAME_MAP = {
    "00": "00_protocol",
    "10": "10_baselines",
    "20": "20_long_tail",
    "30": "30_proposed",
    "40": "40_ablations",
    "60": "60_external",
    "90": "90_final_cv",
}


def _extract_stage_id_and_timestamp(folder_name: str) -> tuple[str, str]:
    """Extract stage ID (00, 10, etc.) and timestamp string from folder name."""
    parts = folder_name.split("_")
    stage_id = "00"
    for i, p in enumerate(parts):
        if p == "stage" and i + 1 < len(parts):
            stage_id = parts[i + 1].zfill(2)
            break

    timestamp = folder_name.split("_")[-1]
    if "__runs" in timestamp:
        timestamp = timestamp.replace("__runs", "")

    return stage_id, timestamp


def _consolidate_legacy_fold_metrics(fold_metrics_dir: Path, fold_name: str) -> dict[str, Any]:
    """Load and merge legacy metrics JSON files into a consolidated metrics dict."""
    def load_or_empty(filename: str) -> dict:
        p = fold_metrics_dir / filename
        return read_json_file(p) if p.is_file() else {}

    train_m = load_or_empty("train_metrics.json")
    val_m = load_or_empty("val_metrics.json")
    test_m = load_or_empty("test_metrics.json")

    train_l = load_or_empty("train_losses.json")
    val_l = load_or_empty("val_losses.json")
    test_l = load_or_empty("test_losses.json")

    bootstrap = load_or_empty("patient_bootstrap_ci.json")
    perf = load_or_empty("performance.json")
    calib = load_or_empty("fine_calibration_latest.json")
    rare = load_or_empty("fine_prior_audit.json")
    wlc = load_or_empty("wlc_only_metrics.json") if (fold_metrics_dir / "wlc_only_metrics.json").is_file() else None
    roi = load_or_empty("roi_metrics.json") if (fold_metrics_dir / "roi_metrics.json").is_file() else None

    return {
        "schema_version": "cystods.fold_metrics",
        "train": {"metrics": train_m, "losses": train_l},
        "validation": {"metrics": val_m, "losses": val_l},
        "test": {"metrics": test_m, "losses": test_l},
        "bootstrap": bootstrap,
        "performance": perf,
        "calibration": calib,
        "rare_class": rare,
        "wlc_only": wlc,
        "roi": roi,
    }


def migrate_single_run_folder(old_run: Path, new_target: Path) -> Path:
    """Migrate a single run directory (whether top-level or child) into new_target."""
    new_target.mkdir(parents=True, exist_ok=True)
    store = ResultStore(new_target)

    # 1. Copy top-level metadata / config / logs
    for item in old_run.iterdir():
        if item.is_file():
            shutil.copy2(item, new_target / item.name)

    # 2. Check fold-based directories: metrics, predictions, splits, visualizations, checkpoints, logs
    folds = set()
    for sub in ("metrics", "predictions", "splits", "visualizations", "checkpoints", "logs"):
        sub_dir = old_run / sub
        if sub_dir.is_dir():
            for child in sub_dir.iterdir():
                if child.is_dir():
                    folds.add(child.name)

    # If no subfolder folds found, check if this run itself is a fold
    if not folds:
        folds.add("holdout")

    for fold_name in sorted(folds):
        target_fold = new_target / "folds" / fold_name
        target_fold.mkdir(parents=True, exist_ok=True)

        # A. Consolidate metrics
        legacy_metrics_dir = old_run / "metrics" / fold_name
        if not legacy_metrics_dir.is_dir():
            legacy_metrics_dir = old_run / "checkpoints" / fold_name
        if legacy_metrics_dir.is_dir():
            consolidated = _consolidate_legacy_fold_metrics(legacy_metrics_dir, fold_name)
            write_json_file(target_fold / "metrics.json", consolidated)

        # B. Move / copy predictions
        legacy_pred_dir = old_run / "predictions" / fold_name
        if legacy_pred_dir.is_dir():
            for pf in legacy_pred_dir.glob("*.csv"):
                shutil.copy2(pf, target_fold / pf.name)
            roi_dir = legacy_pred_dir / "roi"
            if roi_dir.is_dir():
                target_roi = target_fold / "diagnostics" / "roi"
                target_roi.mkdir(parents=True, exist_ok=True)
                for rf in roi_dir.glob("*"):
                    if rf.is_file():
                        shutil.copy2(rf, target_roi / rf.name)

        # C. Move / copy splits
        legacy_splits_dir = old_run / "splits" / fold_name
        if legacy_splits_dir.is_dir():
            target_split = target_fold / "split"
            target_split.mkdir(parents=True, exist_ok=True)
            for sf in legacy_splits_dir.iterdir():
                if sf.is_file():
                    shutil.copy2(sf, target_split / sf.name)

        # D. Move / copy visualizations / figures
        legacy_vis_dir = old_run / "visualizations" / fold_name
        if legacy_vis_dir.is_dir():
            target_fig = target_fold / "figures"
            target_fig.mkdir(parents=True, exist_ok=True)
            for ff in legacy_vis_dir.glob("*.png"):
                shutil.copy2(ff, target_fig / ff.name)

        # E. Training history
        hist_p = old_run / "logs" / f"{fold_name}_history.csv"
        if not hist_p.is_file():
            hist_p = old_run / "checkpoints" / fold_name / "history.csv"
        if hist_p.is_file():
            shutil.copy2(hist_p, target_fold / "history.csv")

    # Generate summary & catalog
    store.generate_catalog()
    return new_target


def migrate_result_directory(old_run_dir: Path, result_root: Path | None = None) -> Path:
    """Migrate a top-level stage run directory (and its sibling __runs if exists)."""
    old_run_dir = Path(old_run_dir).resolve()
    if not old_run_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {old_run_dir}")

    if result_root is None:
        result_root = old_run_dir.parent

    stage_id, timestamp = _extract_stage_id_and_timestamp(old_run_dir.name)
    stage_prefix = STAGE_NAME_MAP.get(stage_id, f"{stage_id}_stage")
    target_dir = result_root / stage_prefix / timestamp

    # Migrate parent run
    migrate_single_run_folder(old_run_dir, target_dir)

    # Check for sibling __runs directory
    sibling_runs = old_run_dir.with_name(f"{old_run_dir.name}__runs")
    if sibling_runs.is_dir():
        target_runs_dir = target_dir / "runs"
        target_runs_dir.mkdir(parents=True, exist_ok=True)
        for child_run in sibling_runs.iterdir():
            if child_run.is_dir():
                child_target = target_runs_dir / child_run.name
                migrate_single_run_folder(child_run, child_target)

    # Re-generate top-level catalog
    ResultStore(target_dir).generate_catalog()

    logger.info(f"Successfully migrated {old_run_dir} to {target_dir}")
    return target_dir
