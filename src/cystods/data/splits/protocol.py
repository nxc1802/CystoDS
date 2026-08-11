"""Protocol artifact generation, protocol binding, and split resolution.

Extracted from ``cystods.core`` (Step 4 refactor).
"""

from __future__ import annotations

import sys
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from cystods.data.splits.cross_validation import (
    search_patient_folds,
    search_train_val_patient_split,
)
from cystods.data.splits.holdout import (
    fixed_patient_split,
    materialize_split_frames,
    search_holdout_patient_split,
    validate_materialized_splits,
)
from cystods.infra.serialization import sha256_file, stable_int_seed, write_json
from cystods.science import split_fingerprint


def _get_core_attr(name: str, fallback: Any) -> Any:
    core_mod = sys.modules.get("cystods.core")
    if core_mod is not None and hasattr(core_mod, name):
        return getattr(core_mod, name)
    return fallback


def save_split_artifacts(
    split_frames: Mapping[str, pd.DataFrame],
    patient_split: Mapping[str, set[str]],
    score: float | None,
    run_dir: Path,
    fold_name: str,
) -> dict[str, Any]:
    fn = _get_core_attr("save_split_artifacts", None)
    if fn is not None and fn is not save_split_artifacts:
        return fn(split_frames, patient_split, score, run_dir, fold_name)

    fold_dir = run_dir / "splits" / fold_name
    fold_dir.mkdir(parents=True, exist_ok=False)
    dataset_fingerprint_path = (
        run_dir / "system" / "dataset_fingerprint.json"
    )
    if not dataset_fingerprint_path.is_file():
        raise FileNotFoundError(
            "Dataset fingerprint must exist before split artifacts are saved."
        )
    dataset_fingerprint = json.loads(
        dataset_fingerprint_path.read_text(encoding="utf-8")
    )
    summary: dict[str, Any] = {
        "allocation_score": score,
        "data_split_fingerprint": split_fingerprint(split_frames),
        "data_split_fingerprint_algorithm": "pid_stem_labels_v2",
        "dataset_semantic_manifest_sha256": dataset_fingerprint[
            "semantic_manifest_sha256"
        ],
        "splits": {},
    }
    total_materialized_rows = sum(len(frame) for frame in split_frames.values())
    total_assigned_patients = sum(len(pids) for pids in patient_split.values())
    columns = [
        "filename",
        "image_path",
        "pid",
        "visit",
        "lesion",
        "class",
        "subclass",
        "binary_id",
        "coarse_id",
        "fine_id",
        "modality",
        "json",
    ]
    for name, frame in split_frames.items():
        frame.loc[:, columns].to_csv(fold_dir / f"{name}.csv", index=False)
        summary["splits"][name] = {
            "rows": len(frame),
            "patients": frame["pid"].nunique(),
            "materialized_image_fraction": (
                len(frame) / total_materialized_rows
            ),
            "assigned_patient_fraction": (
                len(patient_split[name]) / total_assigned_patients
            ),
            "patient_ids": sorted(patient_split[name]),
            "coarse_counts": frame["class"].value_counts().to_dict(),
            "fine_counts": (
                frame.loc[frame["fine_id"] >= 0, "subclass"]
                .value_counts()
                .to_dict()
            ),
        }
    combined_list = []
    for name, frame in split_frames.items():
        sub_df = frame.loc[:, columns].copy()
        sub_df.insert(1, "split", name)
        combined_list.append(sub_df)
    if combined_list:
        combined_df = pd.concat(combined_list, ignore_index=True)
        combined_df.to_csv(fold_dir / "cystods_split.csv", index=False)
        combined_df.to_csv(run_dir / "cystods_split.csv", index=False)

    write_json(fold_dir / "summary.json", summary)
    return summary


def load_frozen_protocol_splits(
    frame: pd.DataFrame,
    protocol_manifest_dir: Path,
    config: Mapping[str, Any],
    run_dir: Path,
    logger: logging.Logger,
) -> list[tuple[str, dict[str, pd.DataFrame], dict[str, set[str]]]]:
    if not protocol_manifest_dir.is_dir():
        if protocol_manifest_dir.parent.is_dir():
            protocol_manifest_dir = protocol_manifest_dir.parent

    if not protocol_manifest_dir.is_dir():
        raise FileNotFoundError(
            f"Frozen protocol directory not found: {protocol_manifest_dir}"
        )
    if config["fixed_split_pids"] is not None:
        raise ValueError(
            "fixed_split_pids cannot be combined with protocol_manifest_dir."
        )
    if (protocol_manifest_dir / "train.csv").is_file():
        unit_dirs = [protocol_manifest_dir]
    else:
        discovered_dirs = sorted(
            summary_path.parent
            for summary_path in protocol_manifest_dir.rglob("summary.json")
            if (summary_path.parent / "train.csv").is_file()
        )
        if config["protocol"] == "holdout":
            unit_dirs = [
                path
                for path in discovered_dirs
                if path.name == "holdout"
                or path.parent.name == "holdout"
                or "holdout" in str(path)
            ]
        else:
            unit_dirs = [
                path
                for path in discovered_dirs
                if path.name.startswith("fold_")
                or path.parent.name.startswith("fold_")
            ]
    if not unit_dirs:
        raise FileNotFoundError(
            "Frozen protocol contains no complete split units."
        )
    requested = config["cv_run_fold_indices"]
    if requested is not None:
        requested_names = {
            f"fold_{int(index):02d}" for index in requested
        }
        unit_dirs = [
            path for path in unit_dirs if path.name in requested_names
        ]
        if {path.name for path in unit_dirs} != requested_names:
            raise ValueError(
                "Requested folds are not all present in frozen protocol."
            )

    dataset_fingerprint = json.loads(
        (
            run_dir / "system" / "dataset_fingerprint.json"
        ).read_text(encoding="utf-8")
    )
    by_stem = frame.set_index("image_stem", verify_integrity=True)
    output: list[
        tuple[str, dict[str, pd.DataFrame], dict[str, set[str]]]
    ] = []
    _validate_materialized_splits = _get_core_attr("validate_materialized_splits", validate_materialized_splits)
    _save_split_artifacts = _get_core_attr("save_split_artifacts", save_split_artifacts)
    for unit_dir in unit_dirs:
        source_summary = json.loads(
            (unit_dir / "summary.json").read_text(encoding="utf-8")
        )
        expected_dataset_hash = source_summary.get(
            "dataset_semantic_manifest_sha256"
        )
        if (
            expected_dataset_hash
            != dataset_fingerprint["semantic_manifest_sha256"]
        ):
            raise ValueError(
                "Frozen protocol dataset fingerprint does not match the "
                f"active dataset for {unit_dir.name}."
            )
        split_frames: dict[str, pd.DataFrame] = {}
        patient_split: dict[str, set[str]] = {}
        for split_name in ("train", "val", "test"):
            csv_path = unit_dir / f"{split_name}.csv"
            if not csv_path.is_file():
                raise FileNotFoundError(csv_path)
            persisted = pd.read_csv(
                csv_path,
                dtype=str,
                keep_default_na=False,
            )
            if "filename" not in persisted:
                raise ValueError(
                    f"Frozen split lacks filename column: {csv_path}"
                )
            stems = persisted["filename"].map(
                lambda value: Path(value).stem
            )
            if stems.duplicated().any():
                raise ValueError(
                    f"Frozen split has duplicate image stems: {csv_path}"
                )
            unknown = sorted(set(stems) - set(by_stem.index))
            if unknown:
                raise ValueError(
                    "Frozen split references images outside the active "
                    f"dataset; first={unknown[0]}"
                )
            restored = by_stem.loc[stems.tolist()].reset_index()
            for column in (
                "pid",
                "binary_id",
                "coarse_id",
                "fine_id",
                "modality",
            ):
                if column in persisted:
                    left = restored[column].astype(str).reset_index(drop=True)
                    right = persisted[column].astype(str).reset_index(drop=True)
                    if not left.equals(right):
                        raise ValueError(
                            "Frozen split metadata mismatch for "
                            f"{unit_dir.name}/{split_name}/{column}."
                        )
            split_frames[split_name] = restored
            patient_split[split_name] = set(
                restored["pid"].astype(str)
            )
        _validate_materialized_splits(split_frames, "smoke")
        computed = split_fingerprint(split_frames)
        if computed != source_summary["data_split_fingerprint"]:
            raise ValueError(
                f"Frozen split fingerprint mismatch for {unit_dir.name}."
            )
        fold_name = unit_dir.name
        _save_split_artifacts(
            split_frames,
            patient_split,
            source_summary.get("allocation_score"),
            run_dir,
            fold_name,
        )
        logger.info(
            "Loaded frozen protocol unit=%s fingerprint=%s",
            fold_name,
            computed,
        )
        output.append((fold_name, split_frames, patient_split))
    return output


def build_all_protocol_splits(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    run_dir: Path,
    logger: logging.Logger,
) -> list[tuple[str, dict[str, pd.DataFrame], dict[str, set[str]]]]:
    output: list[
        tuple[str, dict[str, pd.DataFrame], dict[str, set[str]]]
    ] = []
    seed = int(config["split_seed"])
    _search_holdout = _get_core_attr("search_holdout_patient_split", search_holdout_patient_split)
    _materialize = _get_core_attr("materialize_split_frames", materialize_split_frames)
    _validate = _get_core_attr("validate_materialized_splits", validate_materialized_splits)
    _save_artifacts = _get_core_attr("save_split_artifacts", save_split_artifacts)
    _search_folds = _get_core_attr("search_patient_folds", search_patient_folds)
    _search_train_val = _get_core_attr("search_train_val_patient_split", search_train_val_patient_split)
    _fixed_split = _get_core_attr("fixed_patient_split", fixed_patient_split)

    if config["protocol_manifest_dir"] is not None:
        return load_frozen_protocol_splits(
            frame,
            Path(config["protocol_manifest_dir"]),
            config,
            run_dir,
            logger,
        )
    if config["fixed_split_pids"] is not None:
        if config["protocol"] != "holdout":
            raise ValueError("fixed_split_pids is only valid for holdout.")
        patient_split = _fixed_split(
            frame, config["fixed_split_pids"]
        )
        score = None
        split_frames = _materialize(
            frame, patient_split, config, seed
        )
        _validate(split_frames, config["run_profile"])
        _save_artifacts(
            split_frames, patient_split, score, run_dir, "holdout"
        )
        output.append(("holdout", split_frames, patient_split))
        return output

    if config["protocol"] == "holdout":
        patient_split, score = _search_holdout(
            frame, config, seed
        )
        split_frames = _materialize(
            frame, patient_split, config, seed
        )
        _validate(split_frames, config["run_profile"])
        _save_artifacts(
            split_frames, patient_split, score, run_dir, "holdout"
        )
        logger.info("Selected patient split with score %.6f", score)
        output.append(("holdout", split_frames, patient_split))
        return output

    folds, score = _search_folds(frame, config, seed)
    requested = config["cv_run_fold_indices"]
    indices = list(range(len(folds))) if requested is None else list(requested)
    invalid = set(indices) - set(range(len(folds)))
    if invalid:
        raise ValueError(f"Invalid CV fold indices: {sorted(invalid)}")
    all_pids = set(frame["pid"])
    for fold_index in indices:
        test_pids = folds[fold_index]
        remaining_pids = all_pids - test_pids
        val_fraction = float(config["cv_val_fraction_of_remaining"])
        train_pids, val_pids, inner_score = _search_train_val(
            frame,
            remaining_pids,
            val_fraction,
            int(config["split_search_candidates"]),
            stable_int_seed(seed, "cv", fold_index),
        )
        patient_split = {
            "train": train_pids,
            "val": val_pids,
            "test": set(test_pids),
        }
        split_frames = _materialize(
            frame,
            patient_split,
            config,
            stable_int_seed(seed, "materialize", fold_index),
        )
        # In CV, a singleton fine label must be absent from training in one
        # fold. Coarse coverage and patient isolation remain strict.
        _validate(split_frames, "smoke")
        fold_name = f"fold_{fold_index:02d}"
        _save_artifacts(
            split_frames,
            patient_split,
            score + inner_score,
            run_dir,
            fold_name,
        )
        output.append((fold_name, split_frames, patient_split))
    write_json(
        run_dir / "splits" / "cv_partition.json",
        {
            "allocation_score": score,
            "folds": [sorted(fold) for fold in folds],
        },
    )
    return output


def find_latest_completed_protocol_run(
    result_root: Path | str | None,
    run_profile: str | None = None,
) -> tuple[Path | None, str | None]:
    if result_root is None:
        return None, None
    root = Path(result_root).resolve()
    protocol_dir = root / "00_protocol"
    if not protocol_dir.is_dir():
        return None, None
    candidates: list[tuple[float, Path, str]] = []
    seen: set[Path] = set()
    for manifest_file in protocol_dir.rglob("protocol_manifest.json"):
        path = manifest_file.parent
        if not path.is_dir() or path in seen:
            continue
        seen.add(path)
        status_file = path / "run_status.json"
        if not status_file.is_file():
            continue
        try:
            status = json.loads(status_file.read_text(encoding="utf-8"))
            if status.get("status") != "completed":
                continue
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            if run_profile and manifest.get("run_profile") != run_profile:
                continue
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ):
            continue
        mtime = manifest_file.stat().st_mtime
        sha256 = sha256_file(manifest_file)
        candidates.append((mtime, path, sha256))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def _load_and_validate_protocol_binding(
    stage_config: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], str]:
    raw_protocol_dir = stage_config.get("protocol_run_dir")
    expected_hash = stage_config.get("expected_protocol_sha256")
    if raw_protocol_dir is None:
        _find_latest = _get_core_attr(
            "find_latest_completed_protocol_run",
            find_latest_completed_protocol_run,
        )
        auto_dir, auto_sha = _find_latest(
            stage_config.get("result_root"), stage_config.get("run_profile")
        )
        if auto_dir is None:
            raise RuntimeError(
                "No completed Stage 00 protocol run found. "
                "Run stage_00_prepare_protocol.py first or set CYSTODS_PROTOCOL_RUN_DIR."
            )
        protocol_run_dir = auto_dir
        if expected_hash is None:
            expected_hash = auto_sha
    else:
        protocol_run_dir = Path(raw_protocol_dir).expanduser().resolve()

    if not protocol_run_dir.is_dir():
        raise FileNotFoundError(
            f"Protocol run directory not found: {protocol_run_dir}"
        )
    status_path = protocol_run_dir / "run_status.json"
    protocol_path = protocol_run_dir / "protocol_manifest.json"
    if not status_path.is_file() or not protocol_path.is_file():
        raise FileNotFoundError(
            "Protocol run must contain run_status.json and "
            "protocol_manifest.json."
        )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        raise ValueError("Protocol run is not completed.")
    protocol_manifest = json.loads(
        protocol_path.read_text(encoding="utf-8")
    )
    if protocol_manifest.get("schema_version") != "cystods.protocol.v2":
        raise ValueError("Unsupported protocol manifest schema.")
    if protocol_manifest.get("study_id") != stage_config["study_id"]:
        raise ValueError("Protocol study_id differs from stage study_id.")
    if protocol_manifest.get("run_profile") != stage_config["run_profile"]:
        raise ValueError("Protocol run_profile differs from stage run_profile.")
    protocol_hash = sha256_file(protocol_path)
    if expected_hash is None:
        if stage_config["run_profile"] == "research":
            raise ValueError(
                "Research stages require expected_protocol_sha256. Copy it "
                "from Stage 00 reports/protocol_reference.json."
            )
    elif str(expected_hash) != protocol_hash:
        raise ValueError(
            "Protocol SHA-256 mismatch: "
            f"expected={expected_hash}, actual={protocol_hash}"
        )
    role = str(stage_config["protocol_role"])
    roles = protocol_manifest.get("roles")
    if not isinstance(roles, Mapping):
        raise TypeError("Protocol manifest roles must be a mapping.")
    # Final CV is generated independently over the complete audited dataset.
    # Stage 00 still binds its dataset identity and frozen primary taxonomy;
    # it intentionally does not pre-create any CV fold.
    manifest_role = "fixed_holdout" if role == "final_cv" else role
    if manifest_role not in roles:
        raise ValueError(
            f"Protocol manifest does not define role={manifest_role}."
        )
    role_units = roles[manifest_role].get("units")
    if not isinstance(role_units, list) or not role_units:
        raise ValueError(f"Protocol role={role} contains no split unit.")
    fold_ids = stage_config["fold_ids"]
    if role != "final_cv" and fold_ids is not None:
        raise ValueError("fold_ids are valid only for protocol_role=final_cv.")
    return protocol_run_dir, protocol_manifest, protocol_hash
