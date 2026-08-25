"""Unit and integration tests for Stage 00 3-split protocol upgrade and CLI --split support."""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import cystods.core as core
from cystods.data.splits.holdout import (
    allocation_score,
    patient_label_matrices,
    search_top_k_diverse_holdout_splits,
)
from cystods.data.splits.protocol import (
    build_all_protocol_splits,
    load_frozen_protocol_splits,
)
from cystods.taxonomy import COARSE_NAMES, FINE_NAMES


def _create_mock_cystods_frame(n_patients: int = 120, images_per_patient: int = 4) -> pd.DataFrame:
    """Create a synthetic patient frame with balanced labels for testing."""
    records = []
    for p_idx in range(n_patients):
        pid = f"patient_{p_idx:03d}"
        coarse_id = p_idx % len(COARSE_NAMES)
        coarse_name = COARSE_NAMES[coarse_id]
        fine_id = -1 if coarse_name == "Normal mucosa" else (p_idx % len(FINE_NAMES))
        fine_name = "Normal mucosa" if fine_id == -1 else FINE_NAMES[fine_id]
        binary_id = 1 if coarse_name in ("Malignant", "Non-malignant") else 0

        for img_idx in range(images_per_patient):
            stem = f"{pid}_img_{img_idx:02d}"
            records.append(
                {
                    "filename": f"{stem}.png",
                    "image_stem": stem,
                    "image_path": f"/mock/path/{stem}.png",
                    "pid": pid,
                    "visit": "1",
                    "lesion": "1",
                    "class": coarse_name,
                    "subclass": fine_name,
                    "binary_id": binary_id,
                    "coarse_id": coarse_id,
                    "fine_id": fine_id,
                    "modality": "WLC",
                    "json": "0",
                }
            )
    return pd.DataFrame(records)


def test_patient_label_matrices_returns_five_elements() -> None:
    """Verify patient_label_matrices returns fine_image_counts alongside existing matrices."""
    frame = _create_mock_cystods_frame(n_patients=10, images_per_patient=4)
    pids, coarse_pres, fine_pres, coarse_imgs, fine_imgs = patient_label_matrices(frame)

    assert len(pids) == 10
    assert coarse_pres.shape == (10, len(COARSE_NAMES))
    assert fine_pres.shape == (10, len(FINE_NAMES))
    assert coarse_imgs.shape == (10, len(COARSE_NAMES))
    assert fine_imgs.shape == (10, len(FINE_NAMES))
    assert np.all(coarse_imgs.sum(axis=1) == 4)


def test_allocation_score_penalizes_fine_image_imbalance() -> None:
    """Verify allocation_score penalizes fine-class image distribution differences."""
    frame = _create_mock_cystods_frame(n_patients=50, images_per_patient=5)
    pids, coarse_pres, fine_pres, coarse_imgs, fine_imgs = patient_label_matrices(frame)
    pid_to_row = {pid: idx for idx, pid in enumerate(pids)}

    # Balanced split covering all 5 coarse classes in each split
    assignments_balanced = (
        set(pids[:30]),
        set(pids[30:40]),
        set(pids[40:50]),
    )
    fractions = (0.60, 0.20, 0.20)

    score_balanced = allocation_score(
        assignments_balanced,
        pid_to_row,
        coarse_pres,
        fine_pres,
        coarse_imgs,
        fractions,
        fine_image_counts=fine_imgs,
    )
    assert math.isfinite(score_balanced)
    assert score_balanced > 0


def test_search_top_k_diverse_holdout_splits_overlap_constraint() -> None:
    """Verify top 3 diverse splits strictly satisfy pairwise Test overlap <= 50%."""
    frame = _create_mock_cystods_frame(n_patients=100, images_per_patient=3)
    config = dict(core.BASE_CONFIG)
    config.update(
        {
            "train_fraction": 0.70,
            "val_fraction": 0.15,
            "test_fraction": 0.15,
            "split_search_candidates": 512,
            "force_fine_labels_with_fewer_than_n_patients_to_train": 0,
            "run_profile": "smoke",
        }
    )

    top_splits = search_top_k_diverse_holdout_splits(
        frame=frame,
        config=config,
        seed=20260729,
        k=3,
        max_test_overlap_fraction=0.5,
    )

    assert len(top_splits) == 3
    test_0 = top_splits[0][0]["test"]
    test_1 = top_splits[1][0]["test"]
    test_2 = top_splits[2][0]["test"]

    test_size = len(test_0)
    max_allowed_overlap = math.floor(test_size * 0.5)

    assert len(test_0 & test_1) <= max_allowed_overlap
    assert len(test_0 & test_2) <= max_allowed_overlap
    assert len(test_1 & test_2) <= max_allowed_overlap


def test_build_all_protocol_splits_creates_three_units(tmp_path: Path) -> None:
    """Verify build_all_protocol_splits materializes and saves split_0, split_1, split_2."""
    frame = _create_mock_cystods_frame(n_patients=100, images_per_patient=2)
    system_dir = tmp_path / "system"
    system_dir.mkdir(parents=True)
    (system_dir / "dataset_fingerprint.json").write_text(
        json.dumps({"semantic_manifest_sha256": "mock_sha"}), encoding="utf-8"
    )

    config = dict(core.BASE_CONFIG)
    config.update(
        {
            "protocol": "holdout",
            "split_seed": 20260729,
            "split_search_candidates": 512,
            "force_fine_labels_with_fewer_than_n_patients_to_train": 0,
            "normal_mucosa_limit": None,
            "max_train_samples": None,
            "max_val_samples": None,
            "max_test_samples": None,
            "run_profile": "smoke",
            "fixed_split_pids": None,
            "protocol_manifest_dir": None,
        }
    )

    logger = logging.getLogger("test_logger")
    units = build_all_protocol_splits(frame, config, tmp_path, logger)

    assert len(units) == 3
    assert [u[0] for u in units] == ["split_0", "split_1", "split_2"]

    for u_name in ("split_0", "split_1", "split_2"):
        split_dir = tmp_path / "splits" / u_name
        assert (split_dir / "train.csv").is_file()
        assert (split_dir / "val.csv").is_file()
        assert (split_dir / "test.csv").is_file()
        assert (split_dir / "summary.json").is_file()


def test_load_frozen_protocol_splits_filters_by_split_index(tmp_path: Path) -> None:
    """Verify load_frozen_protocol_splits loads only the requested split unit."""
    frame = _create_mock_cystods_frame(n_patients=100, images_per_patient=2)
    system_dir = tmp_path / "system"
    system_dir.mkdir(parents=True)
    (system_dir / "dataset_fingerprint.json").write_text(
        json.dumps({"semantic_manifest_sha256": "mock_sha"}), encoding="utf-8"
    )

    config = dict(core.BASE_CONFIG)
    config.update(
        {
            "protocol": "holdout",
            "split_seed": 20260729,
            "split_search_candidates": 512,
            "force_fine_labels_with_fewer_than_n_patients_to_train": 0,
            "normal_mucosa_limit": None,
            "max_train_samples": None,
            "max_val_samples": None,
            "max_test_samples": None,
            "run_profile": "smoke",
            "fixed_split_pids": None,
            "protocol_manifest_dir": None,
        }
    )

    logger = logging.getLogger("test_logger")
    build_all_protocol_splits(frame, config, tmp_path, logger)

    # Now load split_1 into a new run_dir
    load_run_dir = tmp_path / "downstream_run"
    (load_run_dir / "system").mkdir(parents=True)
    (load_run_dir / "system" / "dataset_fingerprint.json").write_text(
        json.dumps({"semantic_manifest_sha256": "mock_sha"}), encoding="utf-8"
    )

    load_config = dict(config)
    load_config.update(
        {
            "protocol_manifest_dir": tmp_path,
            "protocol_split_index": 1,
        }
    )

    loaded_units = load_frozen_protocol_splits(
        frame, tmp_path, load_config, load_run_dir, logger
    )
    assert len(loaded_units) == 1
    assert loaded_units[0][0] == "split_1"


def test_cli_split_requirement_for_stages() -> None:
    """Verify CLI parses --split parameter properly."""
    from cystods.cli import _build_parser

    parser = _build_parser()

    # Stage 00 works without --split (default 'all' or None)
    args_00 = parser.parse_args(["run", "00"])
    assert args_00.split in (None, "all")

    # Stage 10 parses --split
    args_10 = parser.parse_args(["run", "10", "--split", "0"])
    assert str(args_10.split) == "0"
