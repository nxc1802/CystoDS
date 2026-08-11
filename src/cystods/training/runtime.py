"""Runtime hardware resolution: device, precision, seeding, system info.

Extracted from ``cystods.core`` (Step 1 refactor).
"""

from __future__ import annotations

import platform
import random
import socket
import sys
from collections.abc import Mapping
from typing import Any

import numpy as np
import psutil
import timm
import torch
import torchvision

from cystods.infra.serialization import utc_now_iso


def resolve_device(config: Mapping[str, Any]) -> torch.device:
    requested = str(config["device"]).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' was requested but CUDA is unavailable.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device='mps' was requested but MPS is unavailable.")
    if requested not in {"cuda", "mps", "cpu"}:
        raise ValueError("device must be auto, cuda, mps, or cpu.")
    return torch.device(requested)


def resolve_precision(
    config: Mapping[str, Any], device: torch.device
) -> tuple[str, torch.dtype | None]:
    requested = str(config["precision"]).lower()
    if requested == "auto":
        if device.type == "cuda":
            requested = (
                "bf16"
                if torch.cuda.is_bf16_supported()
                else "fp16"
            )
        else:
            requested = "fp32"
    if requested == "fp32":
        return requested, None
    if requested == "fp16":
        if device.type not in {"cuda", "mps"}:
            raise RuntimeError("fp16 requires a CUDA or MPS device.")
        return requested, torch.float16
    if requested == "bf16":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 requires a CUDA device with BF16 support.")
        return requested, torch.bfloat16
    raise ValueError("precision must be auto, fp32, fp16, or bf16.")


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def collect_system_info(
    config: Mapping[str, Any],
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestamp_utc": utc_now_iso(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "memory_total_gib": psutil.virtual_memory().total / (1024**3),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "device": str(device),
        "precision": precision,
        "run_profile": config["run_profile"],
    }
    if device.type == "cuda":
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        info.update(
            {
                "cuda_version": torch.version.cuda,
                "cuda_device_name": props.name,
                "cuda_memory_gib": props.total_memory / (1024**3),
                "cuda_capability": torch.cuda.get_device_capability(index),
            }
        )
    return info
