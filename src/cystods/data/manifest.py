"""Manifest loading, verification, fingerprinting, and source file snapshotting.

Extracted from ``cystods.core`` (Step 3 refactor).
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from cystods.data.audit import audit_image_size_distribution
from cystods.infra.environment import is_missing_token
from cystods.infra.serialization import sha256_file, write_json
from cystods.taxonomy import (
    COARSE_NAMES,
    COARSE_TO_ID,
    FINE_BY_PARENT,
    FINE_NAMES,
    FINE_TO_ID,
    REQUIRED_COLUMNS,
    ROI_COARSE_IDS,
)


def validate_source_files(
    source_files: Sequence[Path | str],
) -> tuple[Path, ...]:
    resolved = tuple(Path(path).expanduser().resolve() for path in source_files)
    if not resolved:
        raise ValueError("At least one source file is required for provenance.")
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Required source snapshot files are missing: {missing}"
        )
    names = [path.name for path in resolved]
    if len(set(names)) != len(names):
        raise ValueError(
            "Required source files must have unique basenames for a flat "
            f"snapshot; received={names}."
        )
    return resolved


def snapshot_source_files(
    run_dir: Path,
    source_files: Sequence[Path | str],
) -> dict[str, Any]:
    resolved = validate_source_files(source_files)
    (run_dir / "source").mkdir(parents=True, exist_ok=True)
    rows = []
    for source_path in resolved:
        destination = run_dir / "source" / source_path.name
        shutil.copy2(source_path, destination)
        source_hash = sha256_file(source_path)
        copied_hash = sha256_file(destination)
        if copied_hash != source_hash:
            raise OSError(
                f"Source snapshot checksum mismatch for {source_path}."
            )
        rows.append(
            {
                "path": source_path.name,
                "source_path": str(source_path),
                "snapshot_path": str(destination.relative_to(run_dir)),
                "bytes": destination.stat().st_size,
                "sha256": copied_hash,
            }
        )
    manifest = {
        "schema_version": "cystods.source_manifest.v1",
        "files": rows,
    }
    write_json(run_dir / "source" / "source_manifest.json", manifest)
    return manifest


def load_and_validate_manifest(
    config: Mapping[str, Any],
    run_dir: Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata_csv = Path(config["metadata_csv"])
    image_dir = Path(config["image_dir"])
    segmentation_dir = Path(config["segmentation_dir"])
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if config["verify_segmentation_inventory"] and not segmentation_dir.is_dir():
        raise FileNotFoundError(
            f"Segmentation directory not found: {segmentation_dir}"
        )

    frame = pd.read_csv(
        metadata_csv,
        dtype=str,
        keep_default_na=False,
    )
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Metadata is missing required columns: {sorted(missing_columns)}"
        )
    if frame.empty:
        raise ValueError("Metadata CSV contains no data rows.")
    frame = frame.copy()
    inclusion_manifest = config["inclusion_manifest_csv"]
    if inclusion_manifest is not None:
        inclusion_path = Path(inclusion_manifest)
        if not inclusion_path.is_file():
            raise FileNotFoundError(
                f"Inclusion manifest not found: {inclusion_path}"
            )
        inclusion = pd.read_csv(
            inclusion_path, dtype=str, keep_default_na=False
        )
        filename_column = str(
            config["inclusion_manifest_filename_column"]
        )
        if filename_column not in inclusion:
            raise ValueError(
                "Inclusion manifest lacks configured filename column "
                f"'{filename_column}'."
            )
        inclusion_stems = inclusion[filename_column].map(
            lambda value: Path(value).stem
        )
        if inclusion_stems.duplicated().any():
            raise ValueError("Inclusion manifest contains duplicate stems.")
        source_stems = frame["filename"].map(lambda value: Path(value).stem)
        unknown_stems = set(inclusion_stems) - set(source_stems)
        if unknown_stems:
            raise ValueError(
                "Inclusion manifest references unknown images; first="
                f"{min(unknown_stems)}"
            )
        frame = frame.loc[source_stems.isin(set(inclusion_stems))].copy()
        if len(frame) != len(inclusion_stems):
            raise RuntimeError(
                "Inclusion manifest did not resolve one-to-one to metadata."
            )
    frame["pid"] = frame["pid"].astype(str)
    if frame["pid"].map(is_missing_token).any():
        raise ValueError("Every row must have a non-missing patient ID.")

    frame["image_stem"] = frame["filename"].map(lambda value: Path(value).stem)
    if frame["image_stem"].duplicated().any():
        duplicated = frame.loc[
            frame["image_stem"].duplicated(keep=False), "image_stem"
        ].unique()
        raise ValueError(
            "Duplicate de-identified image stems found: "
            f"{duplicated[:10].tolist()}"
        )
    frame["image_path"] = frame["image_stem"].map(
        lambda stem: str(image_dir / f"{stem}.png")
    )
    missing_images = [
        path for path in frame["image_path"] if not Path(path).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} normalized PNG paths are missing; "
            f"first={missing_images[0]}"
        )

    observed_coarse = set(frame["class"])
    unknown_coarse = observed_coarse - set(COARSE_NAMES)
    if unknown_coarse:
        raise ValueError(f"Unknown coarse labels: {sorted(unknown_coarse)}")
    if observed_coarse != set(COARSE_NAMES):
        raise ValueError(
            "Dataset does not contain every expected coarse label; "
            f"observed={sorted(observed_coarse)}"
        )

    invalid_modalities = set(frame["modality"]) - {"WLC", "BLC"}
    if invalid_modalities:
        raise ValueError(f"Unknown modalities: {sorted(invalid_modalities)}")
    invalid_json_flags = set(frame["json"]) - {"0", "1"}
    if invalid_json_flags:
        raise ValueError(f"Unknown json flags: {sorted(invalid_json_flags)}")

    frame["coarse_id"] = frame["class"].map(COARSE_TO_ID).astype(int)
    frame["binary_id"] = frame["coarse_id"].map(
        lambda value: int(value in ROI_COARSE_IDS)
    )

    fine_ids: list[int] = []
    taxonomy_errors: list[str] = []
    for coarse_name, fine_name, filename in frame[
        ["class", "subclass", "filename"]
    ].itertuples(index=False, name=None):
        coarse_name = str(coarse_name)
        fine_name = str(fine_name)
        if coarse_name == "Normal mucosa":
            if not is_missing_token(fine_name):
                taxonomy_errors.append(
                    f"{filename}: Normal mucosa has subclass={fine_name}"
                )
            fine_ids.append(-1)
            continue
        if fine_name not in FINE_TO_ID:
            taxonomy_errors.append(
                f"{filename}: unknown subclass={fine_name}"
            )
            fine_ids.append(-1)
            continue
        if fine_name not in FINE_BY_PARENT.get(coarse_name, ()):
            taxonomy_errors.append(
                f"{filename}: {fine_name} is not a child of {coarse_name}"
            )
        fine_ids.append(FINE_TO_ID[fine_name])
    if taxonomy_errors:
        raise ValueError(
            "Taxonomy validation failed; first errors: "
            + " | ".join(taxonomy_errors[:10])
        )
    frame["fine_id"] = np.asarray(fine_ids, dtype=np.int64)

    suffix_counts = Counter(
        Path(value).suffix.lower() for value in frame["filename"]
    )
    logger.info(
        "Manifest loaded: rows=%d patients=%d normalized_suffixes=%s",
        len(frame),
        frame["pid"].nunique(),
        dict(suffix_counts),
    )

    if config["verify_all_image_decodes"]:
        logger.info("Verifying all %d PNG decodes...", len(frame))
        for index, path in enumerate(frame["image_path"], start=1):
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as exc:
                raise RuntimeError(f"Image decode validation failed: {path}") from exc
            if index % 1000 == 0:
                logger.info("Image decode audit: %d/%d", index, len(frame))

    segmented_stems = set()
    if config["verify_segmentation_inventory"]:
        segmented_stems = {
            path.stem for path in segmentation_dir.glob("*.json")
        }
        expected_stems = set(frame.loc[frame["json"] == "1", "image_stem"])
        missing_masks = expected_stems - segmented_stems
        orphan_masks = segmented_stems - expected_stems
        if missing_masks or orphan_masks:
            raise ValueError(
                "Segmentation inventory mismatch: "
                f"missing={len(missing_masks)}, orphan={len(orphan_masks)}"
            )

    fine_support = (
        frame.loc[frame["fine_id"] >= 0]
        .groupby("subclass")
        .agg(images=("filename", "size"), patients=("pid", "nunique"))
        .reindex(FINE_NAMES)
        .fillna(0)
        .astype(int)
    )
    semantic_columns = [
        "image_stem",
        "pid",
        "visit",
        "lesion",
        "class",
        "subclass",
        "modality",
        "binary_id",
        "coarse_id",
        "fine_id",
    ]
    semantic_digest = hashlib.sha256()
    for values in (
        frame.loc[:, semantic_columns]
        .astype(str)
        .sort_values(["image_stem", "pid"])
        .itertuples(index=False, name=None)
    ):
        semantic_digest.update(
            json.dumps(
                values,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        semantic_digest.update(b"\n")

    inventory_rows: list[dict[str, Any]] = []
    duplicate_hashes: defaultdict[str, list[str]] = defaultdict(list)
    inventory_digest = hashlib.sha256()
    full_fingerprint = config["dataset_fingerprint_mode"] == "full"
    hash_images = full_fingerprint or bool(
        config["verify_exact_duplicate_images"]
    )
    for row in frame.sort_values("image_stem").itertuples(index=False):
        image_path = Path(row.image_path)
        image_hash = sha256_file(image_path) if hash_images else None
        inventory_row = {
            "image_stem": str(row.image_stem),
            "bytes": image_path.stat().st_size,
            "sha256": image_hash,
        }
        inventory_rows.append(inventory_row)
        inventory_digest.update(
            json.dumps(
                inventory_row,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        inventory_digest.update(b"\n")
        if image_hash is not None:
            duplicate_hashes[image_hash].append(str(row.image_stem))
    duplicate_groups = [
        {"sha256": digest, "image_stems": stems}
        for digest, stems in sorted(duplicate_hashes.items())
        if len(stems) > 1
    ]
    if hash_images:
        pd.DataFrame(inventory_rows).to_csv(
            run_dir / "reports" / "image_inventory.csv",
            index=False,
        )

    segmentation_digest = hashlib.sha256()
    segmentation_inventory: list[dict[str, Any]] = []
    if config["verify_segmentation_inventory"]:
        for path in sorted(segmentation_dir.glob("*.json")):
            row = {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path) if full_fingerprint else None,
            }
            segmentation_inventory.append(row)
            segmentation_digest.update(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            segmentation_digest.update(b"\n")
    inclusion_hash = (
        sha256_file(Path(inclusion_manifest))
        if inclusion_manifest is not None
        else None
    )
    dataset_fingerprint = {
        "mode": config["dataset_fingerprint_mode"],
        "metadata_csv_sha256": sha256_file(metadata_csv),
        "inclusion_manifest_sha256": inclusion_hash,
        "semantic_manifest_sha256": semantic_digest.hexdigest(),
        "image_inventory_sha256": inventory_digest.hexdigest(),
        "image_hashes_included": hash_images,
        "image_duplicate_groups": duplicate_groups,
        "segmentation_inventory_sha256": (
            segmentation_digest.hexdigest()
            if config["verify_segmentation_inventory"]
            else None
        ),
        "rows": len(frame),
        "patients": int(frame["pid"].nunique()),
    }
    write_json(
        run_dir / "system" / "dataset_fingerprint.json",
        dataset_fingerprint,
    )
    image_size_stats = audit_image_size_distribution(frame, run_dir, logger)
    audit = {
        "rows": len(frame),
        "patients": frame["pid"].nunique(),
        "inclusion_manifest_csv": inclusion_manifest,
        "image_suffix_counts_in_csv": dict(suffix_counts),
        "coarse_image_counts": frame["class"].value_counts().to_dict(),
        "coarse_patient_counts": (
            frame.groupby("class")["pid"].nunique().to_dict()
        ),
        "fine_support": fine_support.reset_index().to_dict(orient="records"),
        "modality_counts": frame["modality"].value_counts().to_dict(),
        "bca_counts": frame["bca"].value_counts().to_dict(),
        "segmentation_files": len(segmented_stems),
        "normal_rows_masked_from_fine_loss": int(
            (frame["fine_id"] < 0).sum()
        ),
        "image_size_statistics": image_size_stats,
        "dataset_fingerprint": dataset_fingerprint,
    }
    write_json(run_dir / "reports" / "data_audit.json", audit)
    fine_support.to_csv(
        run_dir / "reports" / "fine_label_support.csv",
        index=True,
    )
    return frame, audit
