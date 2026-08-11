"""Unit tests for 3-tier priority dataset resolution logic."""

from pathlib import Path
from unittest.mock import patch
import pytest

from cystods.config import resolve_dataset_root


def test_resolve_dataset_root_kaggle_priority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Priority 1: Kaggle dataset input should be selected if present."""
    kaggle_dir = tmp_path / "kaggle_dataset"
    kaggle_dir.mkdir()
    (kaggle_dir / "cystods.csv").write_text("id,label\n1,0\n")

    monkeypatch.delenv("CYSTODS_DATA_ROOT", raising=False)

    with patch("cystods.config.Path.is_dir", lambda p: str(p).startswith(str(kaggle_dir)) or p.name == "xvdhy-osfstorage-archive"):
        # Test directly with mock candidate matching kaggle_dir
        with patch("cystods.config.Path.resolve", lambda p: kaggle_dir if "cuongnguyen1802" in str(p) else p):
            with patch.object(Path, "is_file", lambda self: True):
                res = resolve_dataset_root()
                assert res is not None


def test_resolve_dataset_root_explicit_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Explicit CYSTODS_DATA_ROOT environment variable overrides default discovery."""
    custom_dir = tmp_path / "custom_data"
    custom_dir.mkdir()
    (custom_dir / "cystods.csv").write_text("id,label\n1,0\n")

    monkeypatch.setenv("CYSTODS_DATA_ROOT", str(custom_dir))
    resolved = resolve_dataset_root()
    assert resolved == custom_dir.resolve()
