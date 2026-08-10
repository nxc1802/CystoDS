from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

hf = importlib.import_module("cystods_hf")


class FakeHubApi:
    def __init__(self) -> None:
        self.remote_bytes: bytes | None = None
        self.path_in_repo: str | None = None
        self.create_calls: list[tuple[str, dict[str, Any]]] = []
        self.upload_calls: list[dict[str, Any]] = []
        self.info_calls: list[tuple[str, Any, dict[str, Any]]] = []
        self.commit_oid = "a" * 40
        self.commit_url = "https://huggingface.co/org/repo/commit/" + (
            "a" * 40
        )
        self.metadata_size_delta = 0
        self.metadata_path: str | None = None
        self.lfs_sha_override: str | None = None
        self.raise_on_upload: BaseException | None = None

    def create_repo(self, repo_id: str, **kwargs: Any) -> Any:
        self.create_calls.append((repo_id, kwargs))
        return SimpleNamespace(repo_id=repo_id)

    def upload_file(self, **kwargs: Any) -> Any:
        self.upload_calls.append(kwargs)
        if self.raise_on_upload is not None:
            raise self.raise_on_upload
        path = Path(kwargs["path_or_fileobj"])
        self.remote_bytes = path.read_bytes()
        self.path_in_repo = kwargs["path_in_repo"]
        return SimpleNamespace(
            oid=self.commit_oid,
            commit_url=self.commit_url,
        )

    def get_paths_info(
        self, repo_id: str, paths: list[str] | str, **kwargs: Any
    ) -> list[Any]:
        self.info_calls.append((repo_id, paths, kwargs))
        assert self.remote_bytes is not None
        assert self.path_in_repo is not None
        digest = hashlib.sha256(self.remote_bytes).hexdigest()
        lfs_sha = self.lfs_sha_override or digest
        return [
            SimpleNamespace(
                path=self.metadata_path or self.path_in_repo,
                size=len(self.remote_bytes) + self.metadata_size_delta,
                blob_id="fake-blob-id",
                xet_hash=None,
                lfs=SimpleNamespace(sha256=lfs_sha),
            )
        ]


def _config() -> Any:
    return hf.HFCheckpointConfig(
        repo_id="org/cystods",
        path_in_repo="stage_10/trial_binary/best_model.pt",
        token="hf_unit_test_secret",
    )


def _checkpoint(tmp_path: Path, payload: bytes = b"real checkpoint bytes") -> Path:
    checkpoint = tmp_path / "training" / "best_model.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(payload)
    return checkpoint


def _downloader(api: FakeHubApi, *, mutate: bool = False) -> Any:
    def download(**kwargs: Any) -> str:
        assert kwargs["revision"] == api.commit_oid
        assert kwargs["force_download"] is True
        assert kwargs["local_files_only"] is False
        assert api.remote_bytes is not None
        cache_dir = Path(kwargs["cache_dir"])
        destination = cache_dir / "download" / "best_model.pt"
        destination.parent.mkdir(parents=True)
        payload = api.remote_bytes
        if mutate:
            payload = bytes((payload[0] ^ 1,)) + payload[1:]
        destination.write_bytes(payload)
        return str(destination)

    return download


def test_success_verifies_exact_commit_and_keeps_only_receipts(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    receipt_dir = tmp_path / "reports" / "hf_checkpoint"
    api = FakeHubApi()

    receipt = hf.publish_best_checkpoint(
        checkpoint,
        receipt_dir,
        _config(),
        api=api,
        download_file=_downloader(api),
    )

    expected_hash = hashlib.sha256(b"real checkpoint bytes").hexdigest()
    assert not checkpoint.exists()
    assert sorted(path.name for path in receipt_dir.iterdir()) == sorted(
        hf.RECEIPT_FILENAMES
    )
    assert all(path.suffix in {".json", ".csv", ".md"} for path in receipt_dir.iterdir())
    assert receipt["checkpoint_sha256"] == expected_hash
    assert receipt["commit_oid"] == api.commit_oid
    assert receipt["local_checkpoint_removed"] is True
    disk_receipt = json.loads(
        (receipt_dir / "hf_checkpoint_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert disk_receipt == receipt
    assert "hf_unit_test_secret" not in "".join(
        path.read_text(encoding="utf-8") for path in receipt_dir.iterdir()
    )
    assert api.info_calls[0][2]["revision"] == api.commit_oid
    assert api.upload_calls[0]["run_as_future"] is False
    assert not list(tmp_path.rglob(".cystods-hf-verify-*"))
    assert not list(tmp_path.rglob("*.pending-delete"))


def test_upload_failure_preserves_checkpoint_and_publishes_no_receipt(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    receipt_dir = tmp_path / "reports" / "hf_checkpoint"
    api = FakeHubApi()
    api.raise_on_upload = RuntimeError("network rejected upload")

    with pytest.raises(RuntimeError, match="network rejected upload"):
        hf.publish_best_checkpoint(
            checkpoint,
            receipt_dir,
            _config(),
            api=api,
            download_file=_downloader(api),
        )

    assert checkpoint.read_bytes() == b"real checkpoint bytes"
    assert not receipt_dir.exists()


def test_remote_metadata_size_mismatch_is_fatal_and_preserves_local(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    receipt_dir = tmp_path / "receipt"
    api = FakeHubApi()
    api.metadata_size_delta = 1

    with pytest.raises(ValueError, match="metadata size differs"):
        hf.publish_best_checkpoint(
            checkpoint,
            receipt_dir,
            _config(),
            api=api,
            download_file=_downloader(api),
        )

    assert checkpoint.is_file()
    assert not receipt_dir.exists()


def test_remote_lfs_hash_mismatch_is_fatal_and_preserves_local(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    receipt_dir = tmp_path / "receipt"
    api = FakeHubApi()
    api.lfs_sha_override = "0" * 64

    with pytest.raises(ValueError, match="LFS SHA-256 differs"):
        hf.publish_best_checkpoint(
            checkpoint,
            receipt_dir,
            _config(),
            api=api,
            download_file=_downloader(api),
        )

    assert checkpoint.is_file()
    assert not receipt_dir.exists()


def test_downloaded_hash_mismatch_is_fatal_and_preserves_local(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    receipt_dir = tmp_path / "receipt"
    api = FakeHubApi()

    with pytest.raises(ValueError, match="Downloaded Hub checkpoint SHA-256"):
        hf.publish_best_checkpoint(
            checkpoint,
            receipt_dir,
            _config(),
            api=api,
            download_file=_downloader(api, mutate=True),
        )

    assert checkpoint.is_file()
    assert not receipt_dir.exists()
    assert not list(tmp_path.rglob(".cystods-hf-verify-*"))


def test_invalid_commit_oid_is_fatal(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    api = FakeHubApi()
    api.commit_oid = "not-an-oid"

    with pytest.raises(ValueError, match="immutable commit oid"):
        hf.publish_best_checkpoint(
            checkpoint,
            tmp_path / "receipt",
            _config(),
            api=api,
            download_file=_downloader(api),
        )

    assert checkpoint.is_file()


def test_config_from_env_is_explicit_and_does_not_expose_token() -> None:
    config = hf.HFCheckpointConfig.from_env(
        environ={
            "CYSTODS_HF_REPO_ID": "org/repo",
            "CYSTODS_HF_PATH_IN_REPO": "run/best_model.pt",
            "HF_TOKEN": "hf_private_value",
            "CYSTODS_HF_PRIVATE": "true",
            "CYSTODS_HF_CREATE_REPO": "false",
        }
    )

    assert config.create_repo is False
    assert config.private is True
    assert "hf_private_value" not in repr(config)


@pytest.mark.parametrize(
    "path_in_repo",
    [
        "best.pt",
        "/best_model.pt",
        "../best_model.pt",
        "trial\\best_model.pt",
        "trial/../best_model.pt",
    ],
)
def test_path_in_repo_must_be_safe_and_canonical(path_in_repo: str) -> None:
    with pytest.raises(ValueError):
        hf.HFCheckpointConfig(
            repo_id="org/repo",
            path_in_repo=path_in_repo,
            token="hf_secret",
        )


def test_requires_best_model_filename_and_new_receipt_directory(
    tmp_path: Path,
) -> None:
    wrong_name = tmp_path / "model.pt"
    wrong_name.write_bytes(b"checkpoint")
    with pytest.raises(ValueError, match="named best_model.pt"):
        hf.publish_best_checkpoint(
            wrong_name,
            tmp_path / "receipt",
            _config(),
            api=FakeHubApi(),
            download_file=lambda **_: "unused",
        )

    checkpoint = _checkpoint(tmp_path)
    receipt_dir = tmp_path / "existing"
    receipt_dir.mkdir()
    with pytest.raises(FileExistsError, match="new dedicated directory"):
        hf.publish_best_checkpoint(
            checkpoint,
            receipt_dir,
            _config(),
            api=FakeHubApi(),
            download_file=lambda **_: "unused",
        )


def test_symbolic_link_checkpoint_is_rejected(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    link = tmp_path / "best_model.pt"
    link.symlink_to(checkpoint)

    with pytest.raises(ValueError, match="symbolic link"):
        hf.publish_best_checkpoint(
            link,
            tmp_path / "receipt",
            _config(),
            api=FakeHubApi(),
            download_file=lambda **_: "unused",
        )

    assert checkpoint.is_file()


def _write_download_receipt(
    tmp_path: Path,
    payload: bytes = b"remote checkpoint for stage 60",
    **overrides: Any,
) -> tuple[Path, bytes]:
    receipt: dict[str, Any] = {
        "schema_version": hf.RECEIPT_SCHEMA_VERSION,
        "repo_id": "org/cystods",
        "repo_type": "model",
        "path_in_repo": "stage_90/fold_0/best_model.pt",
        "commit_oid": "b" * 40,
        "checkpoint_bytes": len(payload),
        "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
        "remote_metadata_size_verified": True,
        "remote_download_size_verified": True,
        "remote_download_sha256_verified": True,
        "local_checkpoint_removed": True,
    }
    receipt.update(overrides)
    path = tmp_path / "hf_checkpoint_receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path, payload


def _stage60_downloader(payload: bytes, calls: list[dict[str, Any]]) -> Any:
    def download(**kwargs: Any) -> str:
        calls.append(kwargs)
        destination = Path(kwargs["local_dir"]) / kwargs["filename"]
        destination.parent.mkdir(parents=True)
        destination.write_bytes(payload)
        return str(destination)

    return download


def test_stage60_download_uses_receipt_commit_and_verifies_file(
    tmp_path: Path,
) -> None:
    receipt_path, payload = _write_download_receipt(tmp_path)
    calls: list[dict[str, Any]] = []
    download_dir = tmp_path / "stage60-cache"

    checkpoint = hf.download_verified_checkpoint(
        receipt_path,
        download_dir,
        environ={"HF_TOKEN": "hf_stage60_secret"},
        download_file=_stage60_downloader(payload, calls),
    )

    assert checkpoint.is_file()
    assert checkpoint.read_bytes() == payload
    assert checkpoint.is_relative_to(download_dir)
    assert calls[0]["revision"] == "b" * 40
    assert calls[0]["repo_id"] == "org/cystods"
    assert calls[0]["force_download"] is True
    assert calls[0]["local_files_only"] is False
    assert calls[0]["token"] == "hf_stage60_secret"
    assert not list(tmp_path.glob(".stage60-cache.staging-*"))


def test_stage60_download_hash_mismatch_leaves_no_destination(
    tmp_path: Path,
) -> None:
    receipt_path, payload = _write_download_receipt(tmp_path)
    corrupted = bytes((payload[0] ^ 1,)) + payload[1:]
    download_dir = tmp_path / "stage60-cache"

    with pytest.raises(ValueError, match="SHA-256 differs"):
        hf.download_verified_checkpoint(
            receipt_path,
            download_dir,
            environ={"HF_TOKEN": "hf_stage60_secret"},
            download_file=_stage60_downloader(corrupted, []),
        )

    assert not download_dir.exists()
    assert not list(tmp_path.glob(".stage60-cache.staging-*"))


def test_stage60_requires_hf_token_without_fallback(tmp_path: Path) -> None:
    receipt_path, payload = _write_download_receipt(tmp_path)

    with pytest.raises(ValueError, match="HF_TOKEN"):
        hf.download_verified_checkpoint(
            receipt_path,
            tmp_path / "stage60-cache",
            environ={},
            download_file=_stage60_downloader(payload, []),
        )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("schema_version", "unknown"),
        ("commit_oid", "main"),
        ("checkpoint_bytes", 0),
        ("checkpoint_sha256", "not-a-sha"),
        ("remote_download_sha256_verified", False),
        ("local_checkpoint_removed", False),
    ],
)
def test_stage60_rejects_untrusted_receipt(
    tmp_path: Path, field: str, invalid: Any
) -> None:
    receipt_path, payload = _write_download_receipt(
        tmp_path, **{field: invalid}
    )

    with pytest.raises((TypeError, ValueError)):
        hf.download_verified_checkpoint(
            receipt_path,
            tmp_path / "stage60-cache",
            environ={"HF_TOKEN": "hf_stage60_secret"},
            download_file=_stage60_downloader(payload, []),
        )


def test_stage60_requires_new_download_directory(tmp_path: Path) -> None:
    receipt_path, payload = _write_download_receipt(tmp_path)
    download_dir = tmp_path / "stage60-cache"
    download_dir.mkdir()

    with pytest.raises(FileExistsError, match="new path"):
        hf.download_verified_checkpoint(
            receipt_path,
            download_dir,
            environ={"HF_TOKEN": "hf_stage60_secret"},
            download_file=_stage60_downloader(payload, []),
        )
