"""Stage artifact persistence and cross-stage dependency validation.

Manages transition artifacts between pipeline stages (e.g., selected_backbone.json,
selected_long_tail_method.json, proposed_model.json) and enforces fail-fast
protocol fingerprint matching.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cystods.infra.serialization import write_json


def write_stage_selection_artifact(
    run_dir: Path,
    artifact_name: str,
    payload: dict[str, Any],
) -> Path:
    """Save a stage selection artifact into the stage run directory."""
    artifact_path = run_dir / artifact_name
    write_json(artifact_path, payload)
    return artifact_path


def find_and_load_stage_artifact(
    result_root: Path | str,
    stage_id: str,
    artifact_name: str,
    expected_protocol_sha256: str | None = None,
) -> dict[str, Any]:
    """Find and load a stage dependency artifact from previous stage run directories.

    Raises:
        FileNotFoundError: If the stage run directory or artifact file is missing.
        ValueError: If expected_protocol_sha256 is supplied and does not match the artifact.
    """
    root_path = Path(result_root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Result root directory does not exist: {root_path}")

    # Search candidates in reverse chronological order (newest first)
    prefix = f"stage_{stage_id}"
    candidates = sorted(
        [
            d for d in root_path.iterdir()
            if d.is_dir() and (d.name == prefix or d.name.startswith(f"{prefix}_"))
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    artifact_file: Path | None = None
    for cand in candidates:
        filepath = cand / artifact_name
        if filepath.is_file():
            artifact_file = filepath
            break

    if artifact_file is None:
        raise FileNotFoundError(
            f"Required Stage {stage_id} artifact '{artifact_name}' not found under {root_path}. "
            f"Please run stage_{stage_id} first."
        )

    with artifact_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Stage artifact {artifact_file} must be a JSON dict.")

    if expected_protocol_sha256:
        artifact_proto_sha = data.get("protocol_sha256")
        if artifact_proto_sha and artifact_proto_sha != expected_protocol_sha256:
            raise ValueError(
                f"Protocol split mismatch for Stage {stage_id} artifact '{artifact_name}': "
                f"artifact protocol_sha256={artifact_proto_sha!r} vs "
                f"current stage protocol_sha256={expected_protocol_sha256!r}. "
                f"Please re-run stage_{stage_id} on the current protocol split."
            )

    return data
