#!/usr/bin/env python3
"""Benchmark the published CystoDS architecture without training.

The learned checkpoint is intentionally not approximated.  When the private
remote checkpoint is unavailable, the model is constructed with random tensor
values but with the exact published graph and shapes.  Runtime and allocation
measurements are therefore architecture-equivalent; predictions are never
reported or interpreted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = ROOT / "notebook"
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

os.environ.setdefault(
    "CYSTODS_DATA_ROOT", str(ROOT / "xvdhy-osfstorage-archive")
)

import cystods_core as core

RUN_DIR = ROOT / (
    "result/stage_30_run_proposed_method_research_20260803-001339__runs/"
    "proposed_hierarchical_swin_smoothed_seed_20260729_research_20260803-001340"
)
CONFIG_PATH = RUN_DIR / "config.json"
MODEL_INFO_PATH = RUN_DIR / "models/holdout_model_info.json"
RECEIPT_PATH = RUN_DIR / (
    "reports/hf_checkpoints/holdout/hf_checkpoint_receipt.json"
)
TEST_SPLIT_PATH = RUN_DIR / "splits/holdout/test.csv"
ASSET_DIR = Path(__file__).resolve().parent


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def allocated_memory_mib(device: torch.device) -> float | None:
    if device.type == "mps":
        return float(torch.mps.current_allocated_memory() / 2**20)
    if device.type == "cuda":
        return float(torch.cuda.memory_allocated(device) / 2**20)
    return None


def driver_memory_mib(device: torch.device) -> float | None:
    if device.type == "mps":
        return float(torch.mps.driver_allocated_memory() / 2**20)
    if device.type == "cuda":
        return float(torch.cuda.memory_reserved(device) / 2**20)
    return None


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_times(
    times_seconds: list[float], batch_size: int
) -> dict[str, float | int]:
    latency_ms = [value * 1000.0 for value in times_seconds]
    total_seconds = float(sum(times_seconds))
    return {
        "batch_size": int(batch_size),
        "timed_iterations": len(times_seconds),
        "latency_mean_ms_per_batch": float(statistics.fmean(latency_ms)),
        "latency_median_ms_per_batch": float(statistics.median(latency_ms)),
        "latency_p95_ms_per_batch": percentile(latency_ms, 95.0),
        "latency_std_ms_per_batch": float(statistics.pstdev(latency_ms)),
        "latency_mean_ms_per_image": float(
            statistics.fmean(latency_ms) / batch_size
        ),
        "throughput_images_per_second": float(
            batch_size * len(times_seconds) / total_seconds
        ),
    }


def benchmark_compute(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    warmup: int,
    iterations: int,
    image_size: int,
    channels_last: bool,
) -> dict[str, Any]:
    tensor = torch.randn(
        batch_size,
        3,
        image_size,
        image_size,
        device=device,
        dtype=torch.float32,
    )
    if channels_last:
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    synchronize(device)

    with torch.inference_mode():
        for _ in range(warmup):
            outputs = model(tensor)
        synchronize(device)
        memory_after_warmup = allocated_memory_mib(device)
        driver_after_warmup = driver_memory_mib(device)
        times: list[float] = []
        for _ in range(iterations):
            started = time.perf_counter()
            outputs = model(tensor)
            synchronize(device)
            times.append(time.perf_counter() - started)

    result = summarize_times(times, batch_size)
    result.update(
        {
            "scope": "model_forward_only",
            "warmup_iterations": int(warmup),
            "input_shape": [batch_size, 3, image_size, image_size],
            "allocated_memory_after_warmup_mib": memory_after_warmup,
            "driver_memory_after_warmup_mib": driver_after_warmup,
            "output_shapes": {
                key: list(value.shape) for key, value in outputs.items()
            },
        }
    )
    del outputs, tensor
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def resolve_test_image() -> Path:
    split = pd.read_csv(TEST_SPLIT_PATH)
    filename = str(split.iloc[0]["filename"])
    image_path = ROOT / "xvdhy-osfstorage-archive/images" / filename
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    return image_path


def benchmark_preprocessing(
    transform: Any,
    image_path: Path,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        with Image.open(image_path) as source:
            tensor = transform(source.convert("RGB"))
    times: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        with Image.open(image_path) as source:
            tensor = transform(source.convert("RGB"))
        times.append(time.perf_counter() - started)
    result = summarize_times(times, 1)
    result.update(
        {
            "scope": "decode_center_crop_resize_normalize_only",
            "warmup_iterations": int(warmup),
            "source_image": str(image_path.relative_to(ROOT)),
            "output_shape": list(tensor.shape),
            "filesystem_cache_state": "warm_after_warmup",
        }
    )
    return result


def benchmark_end_to_end(
    model: torch.nn.Module,
    transform: Any,
    image_path: Path,
    device: torch.device,
    warmup: int,
    iterations: int,
    channels_last: bool,
) -> dict[str, Any]:
    def one_pass() -> dict[str, torch.Tensor]:
        with Image.open(image_path) as source:
            tensor = transform(source.convert("RGB")).unsqueeze(0)
        if channels_last:
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        tensor = tensor.to(device)
        with torch.inference_mode():
            outputs = model(tensor)
        synchronize(device)
        return outputs

    for _ in range(warmup):
        outputs = one_pass()
    times: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        outputs = one_pass()
        times.append(time.perf_counter() - started)
    result = summarize_times(times, 1)
    result.update(
        {
            "scope": "decode_preprocess_transfer_and_model_forward",
            "warmup_iterations": int(warmup),
            "source_image": str(image_path.relative_to(ROOT)),
            "filesystem_cache_state": "warm_after_warmup",
            "output_shapes": {
                key: list(value.shape) for key, value in outputs.items()
            },
        }
    )
    return result


def write_plot(measurements: list[dict[str, Any]], output: Path) -> None:
    batches = [int(row["batch_size"]) for row in measurements]
    latency = [float(row["latency_mean_ms_per_image"]) for row in measurements]
    throughput = [
        float(row["throughput_images_per_second"]) for row in measurements
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(11.2, 4.8), dpi=180, constrained_layout=True
    )
    color = "#0B6E99"
    labels = [str(value) for value in batches]
    latency_bars = axes[0].bar(labels, latency, color=color)
    axes[0].bar_label(latency_bars, fmt="%.2f", padding=3, fontsize=9)
    axes[0].set_xlabel("Kích thước batch")
    axes[0].set_ylabel("Độ trễ trung bình (ms/ảnh)")
    axes[0].set_title("Độ trễ forward của mô hình")
    axes[0].set_ylim(0, max(latency) * 1.18)
    axes[0].grid(axis="y", alpha=0.25)
    throughput_bars = axes[1].bar(labels, throughput, color=color)
    axes[1].bar_label(throughput_bars, fmt="%.1f", padding=3, fontsize=9)
    axes[1].set_xlabel("Kích thước batch")
    axes[1].set_ylabel("Thông lượng (ảnh/giây)")
    axes[1].set_title("Thông lượng forward của mô hình")
    axes[1].set_ylim(0, max(throughput) * 1.18)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle(
        "CystoDS phân cấp dùng Swin-Tiny — benchmark kiến trúc cục bộ",
        fontsize=15,
    )
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", choices=("auto", "mps", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    config_raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    model_info = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    config_raw["pretrained"] = False
    config_raw["device"] = device.type
    config_raw["precision"] = "fp32"
    config = core.normalize_core_config(config_raw)
    torch.set_num_threads(min(10, psutil.cpu_count(logical=True) or 1))
    model = core.HierarchicalCystoModel(config).eval().to(device)
    channels_last = bool(config["channels_last"])
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    synchronize(device)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if total_parameters != int(model_info["total_parameters"]):
        raise RuntimeError(
            f"Parameter-count mismatch: {total_parameters} != "
            f"{model_info['total_parameters']}"
        )
    model_allocated_mib = allocated_memory_mib(device)
    model_driver_mib = driver_memory_mib(device)
    process_rss_mib = psutil.Process().memory_info().rss / 2**20

    measurements: list[dict[str, Any]] = []
    for batch_size in (1, 8, 32):
        iterations = max(20, args.iterations // max(1, batch_size // 8))
        measurements.append(
            benchmark_compute(
                model,
                device,
                batch_size=batch_size,
                warmup=args.warmup,
                iterations=iterations,
                image_size=int(config["image_size"]),
                channels_last=channels_last,
            )
        )

    _, eval_transform = core.build_transforms(config)
    test_image = resolve_test_image()
    preprocessing = benchmark_preprocessing(
        eval_transform,
        test_image,
        warmup=5,
        iterations=max(30, args.iterations),
    )
    end_to_end = benchmark_end_to_end(
        model,
        eval_transform,
        test_image,
        device,
        warmup=5,
        iterations=max(30, args.iterations),
        channels_last=channels_last,
    )

    payload = {
        "schema_version": "cystods.architecture_inference_benchmark.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "architecture_equivalent_no_learned_checkpoint",
        "scientific_interpretation": (
            "Valid for graph-level runtime and allocation on this machine; "
            "not valid for predictions, accuracy, or explainability."
        ),
        "checkpoint_access": {
            "loaded": False,
            "reason": (
                "Receipt declares a private Hugging Face repository; no local "
                "checkpoint/cache or authenticated Hugging Face token was available."
            ),
            "repo_id": receipt["repo_id"],
            "commit_oid": receipt["commit_oid"],
            "path_in_repo": receipt["path_in_repo"],
            "checkpoint_sha256": receipt["checkpoint_sha256"],
            "checkpoint_bytes": receipt["checkpoint_bytes"],
        },
        "architecture": {
            "model_name": config["model_name"],
            "image_size": int(config["image_size"]),
            "channels_last": channels_last,
            "feature_dim": int(model.feature_dim),
            "total_parameters": int(total_parameters),
            "trainable_parameters": int(
                sum(p.numel() for p in model.parameters() if p.requires_grad)
            ),
            "parameter_count_matches_published_artifact": True,
            "active_outputs": [
                "projection",
                "binary_logits",
                "coarse_logits",
                "fine_logits",
            ],
        },
        "runtime": {
            "device": str(device),
            "precision": "fp32",
            "inference_mode": True,
            "torch_compile": False,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "ram_total_gib": psutil.virtual_memory().total / 2**30,
            "torch": torch.__version__,
            "torchvision": core.torchvision.__version__,
            "timm": core.timm.__version__,
            "model_allocated_memory_mib": model_allocated_mib,
            "model_driver_memory_mib": model_driver_mib,
            "process_rss_after_model_mib": process_rss_mib,
        },
        "method": {
            "timer": "time.perf_counter",
            "device_synchronized_each_timed_iteration": device.type
            in {"mps", "cuda"},
            "warmup_iterations_model_forward": int(args.warmup),
            "timing_includes_preprocessing": False,
            "timing_includes_host_to_device_transfer": False,
            "batch_sizes": [1, 8, 32],
        },
        "model_forward_measurements": measurements,
        "preprocessing_measurement": preprocessing,
        "end_to_end_batch1_measurement": end_to_end,
    }
    json_path = ASSET_DIR / "inference_benchmark_architecture_only.json"
    csv_path = ASSET_DIR / "inference_benchmark_architecture_only.csv"
    plot_path = ASSET_DIR / "inference_benchmark_architecture_only.png"
    json_path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(measurements).to_csv(csv_path, index=False)
    write_plot(measurements, plot_path)
    print(json.dumps(json_ready(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
