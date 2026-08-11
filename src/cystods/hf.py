"""Strict Hugging Face Hub storage for CystoDS best checkpoints.

This module deliberately owns only the remote-storage boundary.  It does not
train models and it never fabricates a successful receipt.  A local
``best_model.pt`` is removed only after the exact uploaded commit has been
checked through Hub metadata and a fresh, temporary download.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

HUGGINGFACE_HUB_REQUIREMENT = "huggingface_hub>=0.28,<2"
RECEIPT_SCHEMA_VERSION = "cystods.hf_checkpoint.v1"
RECEIPT_FILENAMES = (
    "hf_checkpoint_receipt.json",
    "hf_checkpoint_receipt.csv",
    "hf_checkpoint_report.md",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(
    r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$"
)
_SAFE_LOCAL_REPORT_SUFFIXES = frozenset({".json", ".csv", ".md"})


class HubApiProtocol(Protocol):
    """The subset of :class:`huggingface_hub.HfApi` used here."""

    def create_repo(self, repo_id: str, **kwargs: Any) -> Any: ...

    def upload_file(self, **kwargs: Any) -> Any: ...

    def get_paths_info(
        self, repo_id: str, paths: list[str] | str, **kwargs: Any
    ) -> list[Any]: ...


HubDownload = Callable[..., str | Path]


@dataclass(frozen=True, slots=True)
class HFCheckpointConfig:
    """Validated configuration for one immutable checkpoint publication."""

    repo_id: str
    path_in_repo: str
    token: str = field(repr=False)
    revision: str = "main"
    repo_type: str = "model"
    private: bool = True
    create_repo: bool = True
    endpoint: str | None = None
    commit_message: str = "Upload verified CystoDS best_model.pt"

    def __post_init__(self) -> None:
        _validate_config(self)

    @classmethod
    def from_env(
        cls,
        *,
        path_in_repo: str | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> HFCheckpointConfig:
        """Load explicit Hub credentials and destination from environment.

        Required variables are ``CYSTODS_HF_REPO_ID`` and ``HF_TOKEN``.
        ``CYSTODS_HF_PATH_IN_REPO`` is also required when ``path_in_repo`` is
        not supplied by the caller.
        """

        source = os.environ if environ is None else environ
        repo_id = _required_env(source, "CYSTODS_HF_REPO_ID")
        token = _required_env(source, "HF_TOKEN")
        resolved_path = path_in_repo or _required_env(
            source, "CYSTODS_HF_PATH_IN_REPO"
        )
        return cls(
            repo_id=repo_id,
            path_in_repo=resolved_path,
            token=token,
            revision=source.get("CYSTODS_HF_REVISION", "main"),
            private=_parse_bool_env(
                source, "CYSTODS_HF_PRIVATE", default=True
            ),
            create_repo=_parse_bool_env(
                source, "CYSTODS_HF_CREATE_REPO", default=True
            ),
            endpoint=source.get("HF_ENDPOINT") or None,
            commit_message=source.get(
                "CYSTODS_HF_COMMIT_MESSAGE",
                "Upload verified CystoDS best_model.pt",
            ),
        )


def _required_env(source: Mapping[str, str], key: str) -> str:
    value = source.get(key)
    if value is None or not value.strip():
        raise ValueError(f"Required environment variable is missing: {key}")
    return value.strip()


def _parse_bool_env(
    source: Mapping[str, str], key: str, *, default: bool
) -> bool:
    raw = source.get(key)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} must be an explicit boolean, got {raw!r}.")


def _validate_config(config: HFCheckpointConfig) -> None:
    repo_parts = config.repo_id.split("/")
    if (
        len(repo_parts) != 2
        or any(not part or part in {".", ".."} for part in repo_parts)
        or any(part.strip() != part for part in repo_parts)
    ):
        raise ValueError("repo_id must have the form 'namespace/repository'.")
    if config.repo_type != "model":
        raise ValueError("CystoDS checkpoints must use repo_type='model'.")
    if not config.token or config.token.strip() != config.token:
        raise ValueError("A non-empty, whitespace-trimmed Hub token is required.")
    if not config.revision or config.revision.strip() != config.revision:
        raise ValueError("revision must be non-empty and whitespace-trimmed.")
    if not config.commit_message.strip():
        raise ValueError("commit_message must be non-empty.")
    if config.endpoint is not None and not config.endpoint.strip():
        raise ValueError("endpoint must be None or a non-empty URL.")
    _validate_path_in_repo(config.path_in_repo)


def _validate_path_in_repo(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError("path_in_repo must be a normalized relative POSIX path.")
    if path.name != "best_model.pt":
        raise ValueError("path_in_repo must end with 'best_model.pt'.")


def sha256_file(path: Path, *, block_bytes: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading a model into RAM."""

    if block_bytes < 1:
        raise ValueError("block_bytes must be positive.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _default_hub_clients(
    config: HFCheckpointConfig,
) -> tuple[HubApiProtocol, HubDownload]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "huggingface_hub is required; install "
            f"{HUGGINGFACE_HUB_REQUIREMENT!r} in notebook Cell 1."
        ) from exc

    api = HfApi(
        endpoint=config.endpoint,
        token=config.token,
        library_name="cystods",
    )
    return api, hf_hub_download


def _default_hub_downloader() -> HubDownload:
    try:
        from huggingface_hub import hf_hub_download
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "huggingface_hub is required; install "
            f"{HUGGINGFACE_HUB_REQUIREMENT!r} in notebook Cell 1."
        ) from exc
    return hf_hub_download


def _metadata_path(entry: Any) -> str:
    for attribute in ("path", "rfilename"):
        value = getattr(entry, attribute, None)
        if isinstance(value, str) and value:
            return value
    raise ValueError("Hub metadata entry does not expose a file path.")


def _metadata_lfs_sha256(entry: Any) -> str | None:
    lfs = getattr(entry, "lfs", None)
    if lfs is None:
        return None
    value = getattr(lfs, "sha256", None)
    if value is None:
        return None
    normalized = str(value).lower()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ValueError("Hub LFS metadata contains an invalid SHA-256 value.")
    return normalized


def _assert_within(path: Path, parent: Path) -> None:
    try:
        path.resolve(strict=True).relative_to(parent.resolve(strict=True))
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(
            "Hub downloader must return a file inside its temporary cache."
        ) from exc


def _verify_remote_checkpoint(
    *,
    api: HubApiProtocol,
    download_file: HubDownload,
    config: HFCheckpointConfig,
    commit_oid: str,
    expected_size: int,
    expected_sha256: str,
    temporary_parent: Path,
) -> dict[str, Any]:
    entries = api.get_paths_info(
        config.repo_id,
        paths=[config.path_in_repo],
        revision=commit_oid,
        repo_type=config.repo_type,
        token=config.token,
    )
    if not isinstance(entries, list) or len(entries) != 1:
        raise ValueError(
            "Hub metadata lookup must return exactly one checkpoint file."
        )
    entry = entries[0]
    if _metadata_path(entry) != config.path_in_repo:
        raise ValueError("Hub metadata path does not match path_in_repo.")
    remote_size = getattr(entry, "size", None)
    if isinstance(remote_size, bool) or not isinstance(remote_size, int):
        raise TypeError("Hub metadata does not contain an integer file size.")
    if remote_size != expected_size:
        raise ValueError(
            "Hub metadata size differs from the local checkpoint: "
            f"remote={remote_size}, local={expected_size}."
        )

    lfs_sha256 = _metadata_lfs_sha256(entry)
    if lfs_sha256 is not None and lfs_sha256 != expected_sha256:
        raise ValueError("Hub LFS SHA-256 differs from the local checkpoint.")

    with tempfile.TemporaryDirectory(
        prefix=".cystods-hf-verify-", dir=temporary_parent
    ) as temporary_directory:
        cache_dir = Path(temporary_directory).resolve()
        downloaded = Path(
            download_file(
                repo_id=config.repo_id,
                filename=config.path_in_repo,
                repo_type=config.repo_type,
                revision=commit_oid,
                token=config.token,
                cache_dir=cache_dir,
                force_download=True,
                local_files_only=False,
                endpoint=config.endpoint,
            )
        )
        if not downloaded.is_file():
            raise FileNotFoundError(
                f"Hub verification download is not a file: {downloaded}"
            )
        _assert_within(downloaded, cache_dir)
        downloaded_size = downloaded.stat().st_size
        downloaded_sha256 = sha256_file(downloaded)
        if downloaded_size != expected_size:
            raise ValueError(
                "Downloaded Hub checkpoint size differs from the local file."
            )
        if downloaded_sha256 != expected_sha256:
            raise ValueError(
                "Downloaded Hub checkpoint SHA-256 differs from the local file."
            )

    return {
        "metadata_size_verified": True,
        "metadata_lfs_sha256": lfs_sha256,
        "download_size_verified": True,
        "download_sha256_verified": True,
        "remote_blob_id": getattr(entry, "blob_id", None),
        "remote_xet_hash": getattr(entry, "xet_hash", None),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, payload: Mapping[str, Any]) -> None:
    fields = (
        "schema_version",
        "repo_id",
        "repo_type",
        "path_in_repo",
        "source_revision",
        "commit_oid",
        "checkpoint_bytes",
        "checkpoint_sha256",
        "remote_metadata_size_verified",
        "remote_download_sha256_verified",
        "local_checkpoint_removed",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: payload[field] for field in fields})


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Hugging Face checkpoint publication",
        "",
        (
            "The checkpoint was uploaded and verified against an immutable "
            "Hub commit before its local copy was removed."
        ),
        "",
        f"- Repository: `{payload['repo_id']}`",
        f"- Remote path: `{payload['path_in_repo']}`",
        f"- Commit: `{payload['commit_oid']}`",
        f"- Bytes: `{payload['checkpoint_bytes']}`",
        f"- SHA-256: `{payload['checkpoint_sha256']}`",
        "- Remote metadata size: verified",
        "- Fresh remote download SHA-256: verified",
        "- Local `best_model.pt`: removed",
    ]
    commit_url = payload.get("commit_url")
    if commit_url:
        lines.extend((f"- Commit URL: {commit_url}",))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_receipt_staging(
    staging_dir: Path, payload: Mapping[str, Any]
) -> None:
    _write_json(staging_dir / RECEIPT_FILENAMES[0], payload)
    _write_csv(staging_dir / RECEIPT_FILENAMES[1], payload)
    _write_markdown(staging_dir / RECEIPT_FILENAMES[2], payload)
    for path in staging_dir.iterdir():
        if path.suffix not in _SAFE_LOCAL_REPORT_SUFFIXES:
            raise RuntimeError(f"Unsafe local receipt artifact created: {path}")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Receipt JSON contains a non-standard constant: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Receipt JSON contains a duplicate key: {key}")
        result[key] = value
    return result


def load_and_validate_receipt(receipt_json_path: str | Path) -> dict[str, Any]:
    """Read a strict upload receipt suitable for an exact Hub download."""

    receipt_input = Path(receipt_json_path).expanduser()
    if receipt_input.is_symlink():
        raise ValueError("receipt_json_path must not be a symbolic link.")
    receipt_path = receipt_input.resolve()
    if not receipt_path.is_file():
        raise FileNotFoundError(f"Receipt JSON does not exist: {receipt_path}")
    if receipt_path.suffix.lower() != ".json":
        raise ValueError("Checkpoint receipt must be a JSON file.")
    if receipt_path.stat().st_size > 1024 * 1024:
        raise ValueError("Checkpoint receipt JSON exceeds the 1 MiB limit.")
    try:
        payload = json.loads(
            receipt_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid checkpoint receipt JSON: {receipt_path}") from exc
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint receipt JSON root must be an object.")

    required = {
        "schema_version",
        "repo_id",
        "repo_type",
        "path_in_repo",
        "commit_oid",
        "checkpoint_bytes",
        "checkpoint_sha256",
        "remote_metadata_size_verified",
        "remote_download_size_verified",
        "remote_download_sha256_verified",
        "local_checkpoint_removed",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(
            f"Checkpoint receipt is missing required keys: {sorted(missing)}"
        )
    if payload["schema_version"] != RECEIPT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported checkpoint receipt schema: "
            f"{payload['schema_version']!r}"
        )
    repo_id = payload["repo_id"]
    if not isinstance(repo_id, str):
        raise TypeError("Receipt repo_id must be a string.")
    repo_parts = repo_id.split("/")
    if (
        len(repo_parts) != 2
        or any(not part or part in {".", ".."} for part in repo_parts)
        or any(part.strip() != part for part in repo_parts)
    ):
        raise ValueError("Receipt repo_id is invalid.")
    if payload["repo_type"] != "model":
        raise ValueError("Receipt repo_type must be 'model'.")
    if not isinstance(payload["path_in_repo"], str):
        raise TypeError("Receipt path_in_repo must be a string.")
    _validate_path_in_repo(payload["path_in_repo"])
    commit_oid = payload["commit_oid"]
    if not isinstance(commit_oid, str) or not _COMMIT_PATTERN.fullmatch(
        commit_oid
    ):
        raise ValueError("Receipt commit_oid is not an immutable commit hash.")
    checkpoint_size = payload["checkpoint_bytes"]
    if (
        isinstance(checkpoint_size, bool)
        or not isinstance(checkpoint_size, int)
        or checkpoint_size < 1
    ):
        raise ValueError("Receipt checkpoint_bytes must be a positive integer.")
    checkpoint_sha256 = payload["checkpoint_sha256"]
    if (
        not isinstance(checkpoint_sha256, str)
        or not _SHA256_PATTERN.fullmatch(checkpoint_sha256)
    ):
        raise ValueError("Receipt checkpoint_sha256 is invalid.")
    required_true_flags = (
        "remote_metadata_size_verified",
        "remote_download_size_verified",
        "remote_download_sha256_verified",
        "local_checkpoint_removed",
    )
    invalid_flags = [
        key for key in required_true_flags if payload[key] is not True
    ]
    if invalid_flags:
        raise ValueError(
            "Receipt does not prove a completed strict publication: "
            f"{invalid_flags}"
        )
    return payload


def _commit_receipts_and_remove_checkpoint(
    *, checkpoint_path: Path, staging_dir: Path, receipt_dir: Path
) -> None:
    quarantine = checkpoint_path.with_name(
        f".{checkpoint_path.name}.{uuid.uuid4().hex}.pending-delete"
    )
    receipt_committed = False
    checkpoint_quarantined = False
    try:
        os.replace(checkpoint_path, quarantine)
        checkpoint_quarantined = True
        os.replace(staging_dir, receipt_dir)
        receipt_committed = True
        quarantine.unlink()
        checkpoint_quarantined = False
    except BaseException:
        rollback_errors: list[BaseException] = []
        if receipt_committed and receipt_dir.exists():
            try:
                os.replace(receipt_dir, staging_dir)
                receipt_committed = False
            except OSError as exc:  # pragma: no cover - filesystem fault
                rollback_errors.append(exc)
        if checkpoint_quarantined and quarantine.exists():
            try:
                os.replace(quarantine, checkpoint_path)
                checkpoint_quarantined = False
            except OSError as exc:  # pragma: no cover - filesystem fault
                rollback_errors.append(exc)
        if rollback_errors:  # pragma: no cover - catastrophic filesystem fault
            raise RuntimeError(
                "Checkpoint publication failed and filesystem rollback was "
                f"incomplete: {rollback_errors!r}"
            )
        raise


def publish_best_checkpoint(
    checkpoint_path: str | Path,
    receipt_dir: str | Path,
    config: HFCheckpointConfig,
    *,
    api: HubApiProtocol | None = None,
    download_file: HubDownload | None = None,
) -> dict[str, Any]:
    """Upload, verify and remove one local ``best_model.pt``.

    ``receipt_dir`` must be a new, dedicated path.  On success it contains
    exactly JSON, CSV and Markdown evidence.  On any Hub or verification error
    the local checkpoint is preserved and no receipt directory is published.
    Supplying both ``api`` and ``download_file`` enables deterministic unit
    tests without network access; production callers should supply neither.
    """

    checkpoint_input = Path(checkpoint_path).expanduser()
    if checkpoint_input.is_symlink():
        raise ValueError("checkpoint_path must not be a symbolic link.")
    local_checkpoint = checkpoint_input.resolve()
    receipt_input = Path(receipt_dir).expanduser()
    if receipt_input.is_symlink():
        raise ValueError("receipt_dir must not be a symbolic link.")
    local_receipt_dir = receipt_input.resolve()
    if local_checkpoint.name != "best_model.pt":
        raise ValueError("checkpoint_path must point to a file named best_model.pt.")
    if not local_checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {local_checkpoint}")
    checkpoint_size = local_checkpoint.stat().st_size
    if checkpoint_size < 1:
        raise ValueError("best_model.pt must not be empty.")
    if local_receipt_dir.exists():
        raise FileExistsError(
            "receipt_dir must be a new dedicated directory: "
            f"{local_receipt_dir}"
        )
    if (api is None) != (download_file is None):
        raise ValueError(
            "api and download_file must either both be injected or both omitted."
        )

    checkpoint_sha256 = sha256_file(local_checkpoint)
    if not _SHA256_PATTERN.fullmatch(checkpoint_sha256):
        raise RuntimeError("Internal SHA-256 computation returned an invalid value.")

    local_receipt_dir.parent.mkdir(parents=True, exist_ok=True)
    if api is None or download_file is None:
        api, download_file = _default_hub_clients(config)

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{local_receipt_dir.name}.staging-",
            dir=local_receipt_dir.parent,
        )
    )
    try:
        if config.create_repo:
            api.create_repo(
                config.repo_id,
                token=config.token,
                private=config.private,
                repo_type=config.repo_type,
                exist_ok=True,
            )
        commit_info = api.upload_file(
            path_or_fileobj=local_checkpoint,
            path_in_repo=config.path_in_repo,
            repo_id=config.repo_id,
            token=config.token,
            repo_type=config.repo_type,
            revision=config.revision,
            commit_message=config.commit_message,
            run_as_future=False,
        )
        commit_oid = getattr(commit_info, "oid", None)
        if not isinstance(commit_oid, str) or not _COMMIT_PATTERN.fullmatch(
            commit_oid
        ):
            raise ValueError(
                "Hub upload did not return a valid immutable commit oid."
            )
        verification = _verify_remote_checkpoint(
            api=api,
            download_file=download_file,
            config=config,
            commit_oid=commit_oid,
            expected_size=checkpoint_size,
            expected_sha256=checkpoint_sha256,
            temporary_parent=local_receipt_dir.parent,
        )
        receipt: dict[str, Any] = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "repo_id": config.repo_id,
            "repo_type": config.repo_type,
            "path_in_repo": config.path_in_repo,
            "source_revision": config.revision,
            "commit_oid": commit_oid,
            "commit_url": getattr(commit_info, "commit_url", None),
            "checkpoint_bytes": checkpoint_size,
            "checkpoint_sha256": checkpoint_sha256,
            "remote_metadata_size_verified": verification[
                "metadata_size_verified"
            ],
            "remote_metadata_lfs_sha256": verification[
                "metadata_lfs_sha256"
            ],
            "remote_download_size_verified": verification[
                "download_size_verified"
            ],
            "remote_download_sha256_verified": verification[
                "download_sha256_verified"
            ],
            "remote_blob_id": verification["remote_blob_id"],
            "remote_xet_hash": verification["remote_xet_hash"],
            "local_checkpoint_removed": True,
            "local_checkpoint_filename": local_checkpoint.name,
            "local_artifacts": list(RECEIPT_FILENAMES),
        }
        _write_receipt_staging(staging_dir, receipt)
        _commit_receipts_and_remove_checkpoint(
            checkpoint_path=local_checkpoint,
            staging_dir=staging_dir,
            receipt_dir=local_receipt_dir,
        )
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

    if local_checkpoint.exists():
        raise RuntimeError("Local best_model.pt still exists after publication.")
    actual_receipts = sorted(
        path.name for path in local_receipt_dir.iterdir() if path.is_file()
    )
    if actual_receipts != sorted(RECEIPT_FILENAMES):
        raise RuntimeError("Local Hub receipt artifact set is incomplete.")
    return receipt


def download_verified_checkpoint(
    receipt_json_path: str | Path,
    download_dir: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
    download_file: HubDownload | None = None,
) -> Path:
    """Download the exact checkpoint declared by a strict upload receipt.

    This is intended for evaluation-only stages such as Stage 60.  The caller
    must provide a path that does not exist.  The function stages the Hub
    download in a sibling directory, verifies size and SHA-256, then atomically
    publishes ``download_dir`` and returns the checkpoint path inside it.
    Callers own the returned temporary/cache directory and must remove it after
    evaluation.
    """

    receipt = load_and_validate_receipt(receipt_json_path)
    source = os.environ if environ is None else environ
    token = _required_env(source, "HF_TOKEN")
    endpoint = source.get("HF_ENDPOINT") or None
    if endpoint is not None and endpoint.strip() != endpoint:
        raise ValueError("HF_ENDPOINT must be whitespace-trimmed.")
    destination_input = Path(download_dir).expanduser()
    if destination_input.is_symlink():
        raise ValueError("download_dir must not be a symbolic link.")
    destination = destination_input.resolve()
    if destination.exists():
        raise FileExistsError(
            f"download_dir must be a new path that does not exist: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    downloader = _default_hub_downloader() if download_file is None else download_file
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    ).resolve()
    destination_published = False
    try:
        downloaded = Path(
            downloader(
                repo_id=receipt["repo_id"],
                filename=receipt["path_in_repo"],
                repo_type=receipt["repo_type"],
                revision=receipt["commit_oid"],
                token=token,
                local_dir=staging_dir,
                force_download=True,
                local_files_only=False,
                endpoint=endpoint,
            )
        )
        if not downloaded.is_file():
            raise FileNotFoundError(
                f"Hub checkpoint download is not a file: {downloaded}"
            )
        _assert_within(downloaded, staging_dir)
        try:
            verified_file = downloaded.resolve(strict=True)
            downloaded_relative = verified_file.relative_to(staging_dir)
        except ValueError as exc:
            raise ValueError(
                "Hub downloader returned a path outside the staging directory."
            ) from exc
        actual_size = verified_file.stat().st_size
        if actual_size != receipt["checkpoint_bytes"]:
            raise ValueError(
                "Stage checkpoint size differs from its strict Hub receipt: "
                f"downloaded={actual_size}, receipt={receipt['checkpoint_bytes']}."
            )
        actual_sha256 = sha256_file(verified_file)
        if actual_sha256 != receipt["checkpoint_sha256"]:
            raise ValueError(
                "Stage checkpoint SHA-256 differs from its strict Hub receipt."
            )
        os.rename(staging_dir, destination)
        destination_published = True
        result = destination / downloaded_relative
        if not result.is_file():
            raise RuntimeError(
                "Verified checkpoint disappeared during atomic publication."
            )
        if (
            result.stat().st_size != receipt["checkpoint_bytes"]
            or sha256_file(result) != receipt["checkpoint_sha256"]
        ):
            raise RuntimeError(
                "Verified checkpoint changed during atomic publication."
            )
        return result
    except BaseException:
        if destination_published and destination.exists():
            shutil.rmtree(destination)
            destination_published = False
        raise
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


__all__ = [
    "HUGGINGFACE_HUB_REQUIREMENT",
    "RECEIPT_FILENAMES",
    "RECEIPT_SCHEMA_VERSION",
    "HFCheckpointConfig",
    "download_verified_checkpoint",
    "load_and_validate_receipt",
    "publish_best_checkpoint",
    "sha256_file",
]
