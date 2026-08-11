"""Stage 60 — External validation: evaluation-only on external cohort.

Thin orchestrator that delegates to ``cystods.core.run_external_validation_stage``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cystods.core as core
from cystods.config import load_config


STAGE_ID = "60"
STAGE_NAME = "stage_60_evaluate_external"


def _source_files() -> tuple[Path, ...]:
    pkg_dir = Path(__file__).resolve().parent.parent
    return tuple(
        path for path in (pkg_dir / "core.py", pkg_dir / "hf.py", pkg_dir / "science.py")
        if path.is_file()
    )


def run(config: dict[str, Any]) -> Path:
    """Execute Stage 60 with the given resolved config."""
    config = dict(config)
    config["stage_name"] = STAGE_NAME
    config["experiment_name"] = STAGE_NAME
    config["evaluation_scope"] = "external"
    config["external_validation_enabled"] = True

    protocol_run_dir = config.get("protocol_manifest_dir")
    if protocol_run_dir is None:
        env_val = os.environ.get("CYSTODS_PROTOCOL_RUN_DIR")
        if env_val:
            protocol_run_dir = Path(env_val).expanduser().resolve()

    expected_sha = config.get("expected_protocol_sha256") or os.environ.get(
        "CYSTODS_EXPECTED_PROTOCOL_SHA256"
    )

    # External validation specific config
    selected_run_dir = config.get("selected_run_dir") or os.environ.get("CYSTODS_SELECTED_RUN_DIR")
    if selected_run_dir:
        selected_run_dir = Path(selected_run_dir).expanduser().resolve()

    receipt_path = config.get("hf_checkpoint_receipt_json") or os.environ.get(
        "CYSTODS_HF_CHECKPOINT_RECEIPT_JSON"
    )
    if receipt_path:
        receipt_path = Path(receipt_path).expanduser().resolve()
    elif selected_run_dir is not None and selected_run_dir.is_dir():
        receipts = list(selected_run_dir.rglob("hf_checkpoint_receipt.json"))
        if len(receipts) == 1:
            receipt_path = receipts[0]
        elif len(receipts) > 1:
            raise RuntimeError(
                f"Multiple hf_checkpoint_receipt.json files found in {selected_run_dir}. "
                "Please specify CYSTODS_HF_CHECKPOINT_RECEIPT_JSON explicitly."
            )

    external_manifest = config.get("external_manifest_csv")
    if external_manifest is None:
        env_val = os.environ.get("CYSTODS_EXTERNAL_MANIFEST_CSV")
        if env_val:
            external_manifest = Path(env_val).expanduser().resolve()

    external_image_root = config.get("external_image_root")
    if external_image_root is None:
        env_val = os.environ.get("CYSTODS_EXTERNAL_IMAGE_ROOT")
        if env_val:
            external_image_root = Path(env_val).expanduser().resolve()

    config["hf_path_prefix"] = os.environ.get(
        "CYSTODS_HF_PATH_PREFIX", f"{config['study_id']}/{STAGE_NAME}"
    )

    stage_config = {
        "schema_version": "cystods.stage.v2",
        "stage_name": STAGE_NAME,
        "study_id": config["study_id"],
        "run_profile": config["run_profile"],
        "data_root": config["data_root"],
        "result_root": config["result_root"],
        "selected_run_dir": str(selected_run_dir) if selected_run_dir else None,
        "hf_checkpoint_receipt_json": str(receipt_path) if receipt_path else None,
        "internal_protocol_run_dir": str(protocol_run_dir) if protocol_run_dir else None,
        "external_manifest_csv": str(external_manifest) if external_manifest else None,
        "external_image_root": str(external_image_root) if external_image_root else None,
        "base_config": config,
        "external_columns": {
            "path_column": config.get("external_path_column", "path"),
            "binary_label_column": config.get("external_binary_label_column", "binary_label"),
            "patient_id_column": config.get("external_patient_id_column", "patient_id"),
        },
    }

    return core.run_external_validation_stage(stage_config, _source_files())
