"""Hugging Face Hub Synchronization Utility for CystoDS.

Provides flexible, production-grade functions and CLI actions to:
1. Push results/checkpoints from Local or Kaggle compute to HF Hub.
2. Pull lightweight metrics/metadata (JSON, CSV, MD, Log) for fast local analysis (~140 MB).
3. Pull model weights/checkpoints on-demand with wildcard pattern matching or exact path.
4. List and verify remote files against local filesystem.
"""

from __future__ import annotations

import fnmatch
import os
import sys
from pathlib import Path
from typing import Any

DEFAULT_HF_REPO_ID = os.environ.get("CYSTODS_HF_REPO_ID", "Cuong2004/CystoDS-results")
DEFAULT_REPO_TYPE = "model"


def resolve_hf_token(token: str | None = None) -> str | None:
    """Resolve Hugging Face authentication token from multiple sources.

    Resolution order:
    1. Explicit token argument
    2. Environment variable ``HF_TOKEN`` or ``HUGGING_FACE_HUB_TOKEN``
    3. Kaggle Secrets client (if running in Kaggle notebook environment)
    4. Cached Hugging Face local authentication (via ``huggingface_hub.get_token()``)
    """
    if token and token.strip():
        return token.strip()

    # Environment variables
    for env_key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        val = os.environ.get(env_key)
        if val and val.strip():
            return val.strip()

    # Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient

        secret_token = UserSecretsClient().get_secret("HF_TOKEN")
        if secret_token and secret_token.strip():
            return secret_token.strip()
    except Exception:
        pass

    # Hugging Face Hub token cache
    try:
        from huggingface_hub import get_token

        cached = get_token()
        if cached and str(cached).strip():
            return str(cached).strip()
    except Exception:
        pass

    return None


def push_to_hub(
    folder_path: str | Path = "result",
    repo_id: str = DEFAULT_HF_REPO_ID,
    path_in_repo: str = "result",
    private: bool = True,
    repo_type: str = DEFAULT_REPO_TYPE,
    token: str | None = None,
    commit_message: str | None = None,
) -> dict[str, Any]:
    """Upload a local result directory to Hugging Face Hub.

    Args:
        folder_path: Path to the local directory to upload (e.g. 'result' or 'result/30_proposed').
        repo_id: Target Hugging Face repository ('username/repo-name').
        path_in_repo: Relative path inside the destination repository.
        private: Whether the repository should be created as private if it does not exist.
        repo_type: 'model' or 'dataset'.
        token: Hugging Face API token (resolved automatically if None).
        commit_message: Custom commit message.
    """
    try:
        import hf_transfer  # noqa: F401
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        print("⚡ Accelerated transfer mode enabled (hf_transfer Rust backend active).", flush=True)
    except ImportError:
        pass

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required. Install via `pip install huggingface_hub`.") from exc

    resolved_token = resolve_hf_token(token)
    local_path = Path(folder_path).resolve()
    if not local_path.exists():
        raise FileNotFoundError(f"Local folder to upload not found: {local_path}")

    api = HfApi(token=resolved_token)

    print(f"📦 Hugging Face Sync: Preparing repository {repo_id!r} (type={repo_type}, private={private})...", flush=True)
    api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
    )

    msg = commit_message or f"Sync CystoDS results from {local_path.name}"
    print(f"🚀 Uploading folder {local_path} -> {repo_id}:{path_in_repo}...", flush=True)

    commit_info = api.upload_folder(
        folder_path=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=msg,
    )

    print(f"✅ Upload completed successfully to https://huggingface.co/{repo_id}", flush=True)
    return {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "path_in_repo": path_in_repo,
        "commit_info": str(commit_info),
        "url": f"https://huggingface.co/{repo_id}",
    }


def pull_metrics(
    repo_id: str = DEFAULT_HF_REPO_ID,
    local_dir: str | Path = ".",
    repo_type: str = DEFAULT_REPO_TYPE,
    token: str | None = None,
) -> Path:
    """Download only lightweight analysis files (JSON, CSV, Log, MD, YAML) from Hub.

    Bypasses large weight files (*.pt, *.pth, *.bin), saving tens of GBs locally
    while enabling full metrics, benchmark tables, Marimo notebooks, and report generation.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required. Install via `pip install huggingface_hub`.") from exc

    resolved_token = resolve_hf_token(token)
    target_dir = Path(local_dir).resolve()

    print(f"⚡ Quick Sync: Downloading analysis metadata & metrics from {repo_id}...")
    downloaded_dir = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        token=resolved_token,
        local_dir=str(target_dir),
        allow_patterns=[
            "*.json",
            "*.csv",
            "*.log",
            "*.md",
            "*.yaml",
            "*.yml",
            "*.txt",
            "*.png",
            "*.svg",
            "*.pdf",
        ],
        ignore_patterns=[
            "*.pt",
            "*.pth",
            "*.bin",
            "*.ckpt",
            "*.safetensors",
            "*.onnx",
        ],
    )
    print(f"✅ Analysis files synchronized into {target_dir}")
    return Path(downloaded_dir)


def pull_model_checkpoint(
    pattern_or_path: str,
    repo_id: str = DEFAULT_HF_REPO_ID,
    local_dir: str | Path = ".",
    repo_type: str = DEFAULT_REPO_TYPE,
    token: str | None = None,
) -> list[Path]:
    """Download specific model checkpoint(s) on-demand from Hub.

    Args:
        pattern_or_path: Exact file path (e.g. 'result/30_proposed/.../best_model.pt')
                         or wildcard / substring filter (e.g. '*30_proposed*', '*resnet152*', 'split_0').
        repo_id: Hugging Face repository ID.
        local_dir: Target local directory.
        repo_type: 'model' or 'dataset'.
        token: HF API token.
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required. Install via `pip install huggingface_hub`.") from exc

    resolved_token = resolve_hf_token(token)
    target_dir = Path(local_dir).resolve()
    api = HfApi(token=resolved_token)

    # 1. Fetch remote file tree
    remote_files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)

    # 2. Match candidate checkpoint files
    matched_files: list[str] = []
    normalized_query = pattern_or_path.strip()

    if normalized_query in remote_files:
        matched_files.append(normalized_query)
    else:
        for rf in remote_files:
            # Match if pattern matches or query is a substring of the filename/path
            if fnmatch.fnmatch(rf, normalized_query) or fnmatch.fnmatch(rf, f"*{normalized_query}*") or normalized_query in rf:
                matched_files.append(rf)

    if not matched_files:
        print(f"⚠️ No files in {repo_id} matched query: {pattern_or_path!r}")
        print("Tip: Use `cystods hf list` to inspect available remote files.")
        return []

    print(f"🎯 Found {len(matched_files)} matching file(s) on {repo_id}:")
    downloaded_paths: list[Path] = []
    for mf in matched_files:
        print(f"  ⬇️ Downloading: {mf} ...")
        dl_path = hf_hub_download(
            repo_id=repo_id,
            filename=mf,
            repo_type=repo_type,
            token=resolved_token,
            local_dir=str(target_dir),
        )
        downloaded_paths.append(Path(dl_path))
        print(f"    ✓ Saved to {dl_path}")

    print(f"✅ Successfully downloaded {len(downloaded_paths)} file(s).")
    return downloaded_paths


def pull_all(
    repo_id: str = DEFAULT_HF_REPO_ID,
    local_dir: str | Path = ".",
    repo_type: str = DEFAULT_REPO_TYPE,
    token: str | None = None,
) -> Path:
    """Download entire repository contents including all weights and results."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required. Install via `pip install huggingface_hub`.") from exc

    resolved_token = resolve_hf_token(token)
    target_dir = Path(local_dir).resolve()

    print(f"📦 Full Sync: Downloading entire repository {repo_id} into {target_dir}...")
    downloaded_dir = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        token=resolved_token,
        local_dir=str(target_dir),
    )
    print(f"✅ Full repository synchronized into {target_dir}")
    return Path(downloaded_dir)


def list_hub_checkpoints(
    repo_id: str = DEFAULT_HF_REPO_ID,
    pattern: str | None = None,
    repo_type: str = DEFAULT_REPO_TYPE,
    token: str | None = None,
) -> list[str]:
    """List all checkpoint files (.pt, .pth, .safetensors) stored in the HF repo."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required. Install via `pip install huggingface_hub`.") from exc

    resolved_token = resolve_hf_token(token)
    api = HfApi(token=resolved_token)

    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    except Exception as exc:
        print(f"✗ Failed to list files from {repo_id}: {exc}", file=sys.stderr)
        return []

    checkpoints = [f for f in files if f.endswith((".pt", ".pth", ".bin", ".safetensors", ".ckpt"))]
    if pattern:
        checkpoints = [f for f in checkpoints if pattern in f or fnmatch.fnmatch(f, pattern)]

    return checkpoints


def verify_hub_sync(
    folder_path: str | Path = "result",
    repo_id: str = DEFAULT_HF_REPO_ID,
    repo_type: str = DEFAULT_REPO_TYPE,
    token: str | None = None,
) -> dict[str, Any]:
    """Compare local files against remote Hub files to ensure complete synchronization."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required. Install via `pip install huggingface_hub`.") from exc

    resolved_token = resolve_hf_token(token)
    local_path = Path(folder_path).resolve()
    if not local_path.exists():
        raise FileNotFoundError(f"Local path does not exist: {local_path}")

    api = HfApi(token=resolved_token)
    remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type=repo_type))

    # Find all local files
    local_files = [f for f in local_path.rglob("*") if f.is_file() and not f.name.startswith(".")]

    missing_remote: list[str] = []
    matched_count = 0
    pt_matched_count = 0
    pt_total_count = 0

    prefix = local_path.name  # 'result'

    for lf in local_files:
        rel_posix = f"{prefix}/{lf.relative_to(local_path).as_posix()}"
        is_pt = lf.suffix == ".pt"
        if is_pt:
            pt_total_count += 1

        if rel_posix in remote_files:
            matched_count += 1
            if is_pt:
                pt_matched_count += 1
        else:
            missing_remote.append(rel_posix)

    is_fully_synced = (len(missing_remote) == 0) and (matched_count == len(local_files))
    return {
        "is_fully_synced": is_fully_synced,
        "total_local_files": len(local_files),
        "matched_remote_files": matched_count,
        "missing_on_remote": missing_remote,
        "pt_total_count": pt_total_count,
        "pt_matched_count": pt_matched_count,
        "repo_id": repo_id,
    }
