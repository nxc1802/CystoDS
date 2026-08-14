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

    Searches across unified directory structures (e.g. ``result/10_baselines/research_*/``)
    as well as legacy flat run directories (e.g. ``result/stage_10_*/``).

    Raises:
        FileNotFoundError: If the stage run directory or artifact file is missing.
        ValueError: If expected_protocol_sha256 is supplied and does not match the artifact.
    """
    root_path = Path(result_root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Result root directory does not exist: {root_path}")

    # Stage folder prefixes
    stage_id_str = str(stage_id).zfill(2)
    valid_prefixes = (
        f"{stage_id_str}_",
        f"stage_{stage_id_str}",
        f"stage_{stage_id}",
    )

    candidate_files: list[Path] = []
    for d in root_path.iterdir():
        if not d.is_dir():
            continue
        d_name = d.name
        if any(d_name == prefix or d_name.startswith(prefix) for prefix in valid_prefixes):
            # Check if artifact is directly in d
            direct_file = d / artifact_name
            if direct_file.is_file():
                candidate_files.append(direct_file)
            # Check subdirectories in d (e.g. research_*, smoke_*)
            for sub in d.iterdir():
                if sub.is_dir():
                    sub_file = sub / artifact_name
                    if sub_file.is_file():
                        candidate_files.append(sub_file)

    if not candidate_files:
        raise FileNotFoundError(
            f"Required Stage {stage_id} artifact '{artifact_name}' not found under {root_path}. "
            f"Please run stage_{stage_id} first."
        )

    # Sort candidates by mtime (newest first)
    candidate_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # Load and parse candidates
    parsed_candidates: list[tuple[Path, dict[str, Any]]] = []
    for fpath in candidate_files:
        try:
            with fpath.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                parsed_candidates.append((fpath, data))
        except Exception:
            continue

    if not parsed_candidates:
        raise ValueError(
            f"Stage {stage_id} artifact '{artifact_name}' found at {candidate_files[0]} but could not be parsed as JSON dict."
        )

    # If expected_protocol_sha256 is supplied, find matching candidate
    if expected_protocol_sha256:
        for fpath, data in parsed_candidates:
            artifact_proto_sha = data.get("protocol_sha256")
            if artifact_proto_sha == expected_protocol_sha256:
                return data

        # If no matching protocol found, raise ValueError using the newest candidate's SHA
        newest_fpath, newest_data = parsed_candidates[0]
        artifact_proto_sha = newest_data.get("protocol_sha256")
        raise ValueError(
            f"Protocol split mismatch for Stage {stage_id} artifact '{artifact_name}': "
            f"artifact protocol_sha256={artifact_proto_sha!r} vs "
            f"current stage protocol_sha256={expected_protocol_sha256!r}. "
            f"Please re-run stage_{stage_id} on the current protocol split."
        )

    return parsed_candidates[0][1]

