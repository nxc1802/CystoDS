"""Stage 00 — Prepare protocol: data audit + freeze patient-disjoint split.

Thin orchestrator that delegates to ``cystods.core.run_protocol_stage``.
All configuration comes from the central ``config.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cystods.core as core


STAGE_ID = "00"
STAGE_NAME = "stage_00_prepare_protocol"


def _source_files() -> tuple[Path, ...]:
    """Return the package source files for provenance snapshot."""
    pkg_dir = Path(__file__).resolve().parent.parent
    return tuple(
        path
        for path in (
            pkg_dir / "core.py",
            pkg_dir / "hf.py",
            pkg_dir / "science.py",
        )
        if path.is_file()
    )


def run(config: dict[str, Any]) -> Path:
    """Execute Stage 00 with the given resolved config."""
    # Ensure stage identity
    config = dict(config)
    config["stage_name"] = STAGE_NAME
    config["experiment_name"] = STAGE_NAME
    config["protocol_role"] = "fixed_holdout"
    config["evaluation_scope"] = "development"
    config["verify_all_image_decodes"] = config.get("verify_all_image_decodes", True)
    config["deterministic"] = config.get("deterministic", True)

    if config.get("run_profile") == "smoke":
        config.setdefault("protocol_role", "smoke_holdout")

    stage_config = {
        "schema_version": "cystods.stage.v2",
        "stage_name": STAGE_NAME,
        "study_id": config["study_id"],
        "run_profile": config["run_profile"],
        "data_root": config["data_root"],
        "result_root": config["result_root"],
        "protocol_config": config,
    }

    source_files = _source_files()
    return core.run_protocol_stage(stage_config, source_files)
