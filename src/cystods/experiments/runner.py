"""Experiment runner and stage orchestration.

Extracted from ``cystods.core`` (Step 8 refactor).
"""

from __future__ import annotations

import json
import math
import platform
import socket
import subprocess as _subprocess
import sys as _sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

import cystods.hf as checkpoint_hub
import cystods.science as science
import timm
from cystods.data.dataset import CystoDataset, ExternalBinaryDataset
from cystods.data.manifest import (
    load_and_validate_manifest,
    snapshot_source_files,
    validate_source_files,
)
from cystods.data.sampler import build_dataloaders, make_worker_init_fn
from cystods.data.splits import (
    build_all_protocol_splits,
)
from cystods.data.transforms import build_transforms
from cystods.evaluation.bootstrap import (
    paired_mcnemar_test,
    patient_bootstrap_intervals,
)
from cystods.evaluation.metrics import (
    compute_binary_metrics,
    compute_metrics_bundle,
)
from cystods.evaluation.roi import run_roi_evaluation
from cystods.infra.environment import (
    close_logger,
    setup_logger,
)
from cystods.infra.serialization import (
    json_ready,
    sha256_file,
    stable_int_seed,
    utc_now_iso,
    write_json,
)
from cystods.losses.classification import FineLongTailLoss
from cystods.losses.composite import active_fine_loss_name
from cystods.training.checkpoint import _canonical_model_state_dict, _hf_checkpoint_config, _hf_checkpoint_path_in_repo
from cystods.config import (
    make_run_directory,
    normalize_core_config,
    validate_config,
)
from cystods.models.factory import (
    active_tasks_from_config,
)
from cystods.models.hierarchical import HierarchicalCystoModel
from cystods.reports.summary import (
    aggregate_cross_validation_metrics,
    build_fold_report,
    serialize_prediction_frame,
    write_artifact_manifest,
)
from cystods.taxonomy import (
    BINARY_NAMES,
    COARSE_NAMES,
    FINE_BY_PARENT,
    FINE_NAMES,
    FINE_PARENT_ID,
)
from cystods.training.engine import (
    evaluate_model,
    move_images,
    train_model,
)
from cystods.training.runtime import (
    collect_system_info,
    resolve_device,
    resolve_precision,
    seed_everything,
)
from cystods.visualization.plots import (
    export_fold_visualizations,
    plot_binary_curves,
    plot_confusion,
)

DEPENDENCY_AUDIT = {"mode": "package_install"}


def make_deterministic_eval_loader(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> DataLoader:
    _, eval_transform, _ = build_transforms(config)
    dataset = CystoDataset(frame, eval_transform, second_view_transform=None)
    num_workers = (
        0 if device.type == "mps" else int(config["eval_num_workers"])
    )
    kwargs: dict[str, Any] = {
        "batch_size": int(config["eval_batch_size"]),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": bool(config.get("pin_memory", True) and device.type == "cuda"),
        "persistent_workers": False,
        "worker_init_fn": make_worker_init_fn(seed),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(config["eval_prefetch_factor"])
    return DataLoader(dataset, **kwargs)


def evaluate_external_binary(
    model: nn.Module,
    config: Mapping[str, Any],
    device: torch.device,
    amp_dtype: torch.dtype | None,
    run_dir: Path,
    fold_name: str,
) -> dict[str, Any]:
    _, eval_transform, _ = build_transforms(config)
    dataset = ExternalBinaryDataset(
        Path(config["external_manifest_csv"]),
        Path(config["external_image_root"]),
        eval_transform,
        config,
    )
    num_workers = (
        0 if device.type == "mps" else int(config["eval_num_workers"])
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(config["eval_batch_size"]),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": bool(
            config["pin_memory"] and device.type == "cuda"
        ),
        "persistent_workers": False,
        "worker_init_fn": make_worker_init_fn(
            stable_int_seed(config["seed"], "external_loader")
        ),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(
            config["eval_prefetch_factor"]
        )
    loader = DataLoader(dataset, **loader_kwargs)
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            images = move_images(batch["image"], device, config)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                outputs = model(images)
            probabilities = (
                outputs["binary_logits"].softmax(dim=1).float().cpu().numpy()
            )
            for index, probability in enumerate(probabilities):
                rows.append(
                    {
                        "filename": batch["filename"][index],
                        "pid": batch["pid"][index],
                        "binary_id": int(batch["binary_id"][index]),
                        "binary_probability_roi": float(probability[1]),
                    }
                )
    predictions = pd.DataFrame(rows)
    metrics = compute_binary_metrics(
        predictions["binary_id"].to_numpy(dtype=np.int64),
        predictions["binary_probability_roi"].to_numpy(dtype=float),
        float(config["binary_decision_threshold"]),
    )
    patients = sorted(predictions["pid"].astype(str).unique())
    if len(patients) < 2:
        bootstrap = {
            "status": "not_evaluable",
            "reason": "fewer than two external patients",
            "iterations_requested": int(config["bootstrap_iterations"]),
            "iterations_valid": 0,
        }
    else:
        grouped = {
            patient: predictions[
                predictions["pid"].astype(str).eq(patient)
            ]
            for patient in patients
        }
        rng = np.random.default_rng(
            stable_int_seed(config["seed"], "external_bootstrap")
        )
        bootstrap_samples: defaultdict[str, list[float]] = defaultdict(list)
        for _ in range(int(config["bootstrap_iterations"])):
            sampled_patients = rng.choice(
                patients,
                size=len(patients),
                replace=True,
            )
            sample = pd.concat(
                [grouped[str(patient)] for patient in sampled_patients],
                ignore_index=True,
            )
            sample_metrics = compute_binary_metrics(
                sample["binary_id"].to_numpy(dtype=np.int64),
                sample["binary_probability_roi"].to_numpy(dtype=float),
                float(config["binary_decision_threshold"]),
            )
            for metric_name in (
                "accuracy",
                "f1",
                "mcc",
                "balanced_accuracy",
                "auroc",
                "auprc",
            ):
                value = sample_metrics[metric_name]
                if value is not None and math.isfinite(float(value)):
                    bootstrap_samples[metric_name].append(float(value))
        alpha = 1.0 - float(config["bootstrap_confidence"])
        bootstrap = {
            "status": "ok",
            "method": "external patient-level percentile bootstrap",
            "confidence": float(config["bootstrap_confidence"]),
            "iterations_requested": int(config["bootstrap_iterations"]),
            "intervals": {
                metric_name: {
                    "lower": float(np.quantile(values, alpha / 2)),
                    "upper": float(np.quantile(values, 1 - alpha / 2)),
                    "mean": float(np.mean(values)),
                    "valid_iterations": len(values),
                }
                for metric_name, values in bootstrap_samples.items()
                if values
            },
        }
    metrics["patient_bootstrap"] = bootstrap
    predictions.to_csv(
        run_dir
        / "predictions"
        / fold_name
        / "external_binary_predictions.csv",
        index=False,
    )
    visual_dir = run_dir / "visualizations" / fold_name
    visual_dir.mkdir(parents=True, exist_ok=True)
    curve_frame = predictions.copy()
    curve_frame["binary_probs"] = curve_frame[
        "binary_probability_roi"
    ].map(lambda value: np.asarray([1.0 - float(value), float(value)]))
    plot_binary_curves(
        curve_frame,
        visual_dir / "external_binary_roc_pr_curves.png",
    )
    plot_confusion(
        metrics["confusion_matrix"],
        BINARY_NAMES,
        "External binary confusion matrix",
        visual_dir / "external_binary_confusion_matrix.png",
    )
    write_json(
        run_dir / "metrics" / fold_name / "external_binary_metrics.json",
        metrics,
    )
    write_json(
        run_dir
        / "metrics"
        / fold_name
        / "external_patient_bootstrap_ci.json",
        bootstrap,
    )
    return metrics


def run_single_fold(
    fold_name: str,
    split_frames: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
    device: torch.device,
    amp_dtype: torch.dtype | None,
    run_dir: Path,
    logger: Any,
) -> dict[str, Any]:
    fold_seed = stable_int_seed(config["seed"], fold_name)
    seed_everything(fold_seed, bool(config["deterministic"]))
    optimization_frames = dict(split_frames)
    if config["train_modality"] == "WLC":
        optimization_train = split_frames["train"].loc[
            split_frames["train"]["modality"].eq("WLC")
        ].copy()
    else:
        optimization_train = split_frames["train"].copy()
    if optimization_train.empty:
        raise ValueError(
            f"train_modality={config['train_modality']} produced no training row."
        )
    optimization_frames["train"] = optimization_train
    optimization_audit = {
        "train_modality": str(config["train_modality"]),
        "protocol_train_rows": len(split_frames["train"]),
        "optimization_train_rows": len(optimization_train),
        "optimization_train_patients": int(
            optimization_train["pid"].nunique()
        ),
        "coarse_counts": optimization_train["class"].value_counts().to_dict(),
        "fine_counts": (
            optimization_train.loc[optimization_train["fine_id"] >= 0]
            ["subclass"]
            .value_counts()
            .to_dict()
        ),
        "optimization_train_semantic_sha256": science.semantic_fingerprint(
            optimization_train[
                [
                    "image_stem",
                    "pid",
                    "binary_id",
                    "coarse_id",
                    "fine_id",
                    "modality",
                ]
            ]
            .sort_values("image_stem")
            .astype(str)
            .to_dict(orient="records")
        ),
    }
    write_json(
        run_dir / "reports" / f"{fold_name}_training_subset.json",
        optimization_audit,
    )
    optimization_train[
        [
            "filename",
            "pid",
            "binary_id",
            "coarse_id",
            "fine_id",
            "modality",
        ]
    ].to_csv(
        run_dir / "splits" / fold_name / "optimization_train.csv",
        index=False,
    )
    loaders, _ = build_dataloaders(
        optimization_frames, config, device, fold_seed
    )
    model = HierarchicalCystoModel(config).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if config.get("channels_last", False):
        model = model.to(memory_format=torch.channels_last)
    if config["torch_compile"]:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch_compile requested but torch.compile is absent.")
        model.encoder = torch.compile(
            model.encoder, mode=str(config["torch_compile_mode"])
        )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    model_info = {
        "model_name": config["model_name"],
        "pretrained": config["pretrained"],
        "feature_dim": model.feature_dim,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
    }
    write_json(
        run_dir / "models" / f"{fold_name}_model_info.json", model_info
    )
    logger.info(
        "Model ready: %s parameters=%d trainable=%d device=%s",
        config["model_name"],
        total_parameters,
        trainable_parameters,
        device,
    )

    checkpoint_dir = run_dir / "checkpoints" / fold_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    temporary_checkpoint_owner: Any | None = None
    if config["checkpoint_backend"] == "huggingface":
        temporary_checkpoint_owner = tempfile.TemporaryDirectory(
            prefix=f"cystods-{fold_name}-checkpoint-"
        )
        checkpoint_work_dir = Path(temporary_checkpoint_owner.name)
    else:
        checkpoint_work_dir = checkpoint_dir
    model, history, fine_counts, fine_patient_counts = train_model(
        model,
        loaders,
        split_frames,
        optimization_train,
        config,
        device,
        amp_dtype,
        checkpoint_dir,
        logger,
        checkpoint_work_dir,
    )
    del loaders
    history.to_csv(run_dir / "logs" / f"{fold_name}_history.csv", index=False)
    performance = {
        "train_samples_per_second_mean": float(
            history["train_samples_per_second"].mean()
        ),
        "train_samples_per_second_max": float(
            history["train_samples_per_second"].max()
        ),
        "train_seconds_total": float(history["train_seconds"].sum()),
        "epochs_completed": len(history),
        "batch_size": int(config["batch_size"]),
        "gradient_accumulation_steps": int(
            config["gradient_accumulation_steps"]
        ),
        "num_workers": int(config["num_workers"]),
        "precision": config["precision"],
        "channels_last": bool(config["channels_last"]),
        "torch_compile": bool(config["torch_compile"]),
        "cuda_peak_memory_mib": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if device.type == "cuda"
            else None
        ),
        "cuda_peak_reserved_mib": (
            float(torch.cuda.max_memory_reserved(device) / (1024**2))
            if device.type == "cuda"
            else None
        ),
        "cuda_total_memory_mib": (
            float(
                torch.cuda.get_device_properties(device).total_memory
                / (1024**2)
            )
            if device.type == "cuda"
            else None
        ),
    }
    write_json(
        run_dir / "metrics" / fold_name / "performance.json",
        performance,
    )
    best_checkpoint = torch.load(
        checkpoint_work_dir / "best_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    selected_fine_tau = float(
        best_checkpoint["fine_inference_prior_tau"]
    )
    portable_checkpoint = {
        "model_state_dict": _canonical_model_state_dict(
            best_checkpoint["model_state_dict"]
        ),
        "config": best_checkpoint["config"],
        "binary_names": BINARY_NAMES,
        "coarse_names": COARSE_NAMES,
        "fine_names": FINE_NAMES,
        "fine_parent_id": FINE_PARENT_ID,
        "fine_train_counts": fine_counts,
        "fine_train_patient_counts": fine_patient_counts,
        "fine_prior_audit": best_checkpoint["fine_prior_audit"],
        "fine_inference_prior_tau": selected_fine_tau,
        "data_split_fingerprint": best_checkpoint[
            "data_split_fingerprint"
        ],
        "model_info": model_info,
    }
    if config["checkpoint_backend"] == "huggingface":
        portable_checkpoint_path = (
            checkpoint_work_dir / "publish" / "best_model.pt"
        )
    else:
        portable_checkpoint_path = (
            run_dir / "models" / f"{fold_name}_best_model.pt"
        )
    portable_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(portable_checkpoint, portable_checkpoint_path)
    del portable_checkpoint
    del best_checkpoint

    metrics_dir = run_dir / "metrics" / fold_name
    prediction_dir = run_dir / "predictions" / fold_name
    metrics_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    split_predictions: dict[str, pd.DataFrame] = {}
    split_metrics: dict[str, Any] = {}
    split_losses: dict[str, Any] = {}
    active_tasks = active_tasks_from_config(config)
    evaluation_fine_loss = (
        FineLongTailLoss(
            active_fine_loss_name(config),
            fine_counts,
            fine_patient_counts,
            config,
        ).to(device)
        if "fine" in active_tasks
        else None
    )
    retain_features = bool(
        config["evaluate_roi_level"]
        and "attention" in set(config["roi_aggregations"])
    )
    for split_name in ("train", "val", "test"):
        eval_loader = make_deterministic_eval_loader(
            split_frames[split_name],
            config,
            device,
            stable_int_seed(fold_seed, split_name, "eval"),
        )
        metrics, predictions, losses = evaluate_model(
            model,
            eval_loader,
            evaluation_fine_loss,
            fine_counts,
            fine_patient_counts,
            config,
            device,
            amp_dtype,
            include_features=retain_features,
            fine_prior_tau=selected_fine_tau,
        )
        split_predictions[split_name] = predictions
        split_metrics[split_name] = metrics
        split_losses[split_name] = losses
        serialize_prediction_frame(
            predictions,
            prediction_dir / f"{split_name}_image_predictions.csv",
            include_features=False,
        )
        write_json(metrics_dir / f"{split_name}_metrics.json", metrics)
        write_json(metrics_dir / f"{split_name}_losses.json", losses)
        logger.info(
            "final-eval split=%s n=%d binary_f1=%s coarse_macro_f1=%s "
            "fine_macro_f1_all=%s",
            split_name,
            len(predictions),
            (
                f"{metrics['binary']['f1']:.5f}"
                if metrics["binary"] is not None
                else "inactive"
            ),
            (
                f"{metrics['coarse']['macro_f1_supported']:.5f}"
                if metrics["coarse"] is not None
                else "inactive"
            ),
            (
                f"{metrics['fine']['macro_f1_all_classes']:.5f}"
                if metrics["fine"] is not None
                else "inactive"
            ),
        )

    test_predictions = split_predictions["test"]
    test_metrics = split_metrics["test"]
    if (
        test_metrics["rare_class_collapse"] is not None
        and config["scientific_gate_mode"] == "enforce"
    ):
        if config["evaluation_scope"] == "final_cv":
            science.enforce_rare_class_collapse_gate(
                test_metrics["rare_class_collapse"]
            )
        elif test_metrics["rare_class_collapse"]["status"] == "failed":
            logger.warning(
                "Scientific gate rejected this development candidate; "
                "violations=%s",
                [
                    row["class_name"]
                    for row in test_metrics["rare_class_collapse"][
                        "violations"
                    ]
                ],
            )
    bootstrap = patient_bootstrap_intervals(
        test_predictions,
        fine_counts,
        fine_patient_counts,
        config,
        stable_int_seed(fold_seed, "bootstrap"),
    )
    write_json(metrics_dir / "patient_bootstrap_ci.json", bootstrap)

    wlc_metrics = None
    if config["evaluate_wlc_only"]:
        wlc = test_predictions[test_predictions["modality"] == "WLC"].copy()
        if wlc.empty:
            raise ValueError("WLC-only evaluation requested but has zero rows.")
        wlc_metrics = compute_metrics_bundle(
            wlc,
            fine_counts,
            fine_patient_counts,
            config,
        )
        write_json(metrics_dir / "wlc_only_metrics.json", wlc_metrics)
        serialize_prediction_frame(
            wlc,
            prediction_dir / "test_wlc_only_predictions.csv",
            include_features=False,
        )

    roi_metrics = None
    if config["evaluate_roi_level"]:
        roi_metrics = run_roi_evaluation(
            split_predictions,
            config,
            device,
            run_dir,
            fold_name,
            logger,
        )

    paired = None
    if config["paired_baseline_predictions_csv"] is not None:
        if "binary" not in active_tasks:
            raise ValueError(
                "Paired McNemar evaluation requires an active binary head."
            )
        paired = paired_mcnemar_test(
            test_predictions,
            Path(config["paired_baseline_predictions_csv"]),
            float(config["binary_decision_threshold"]),
        )
        write_json(metrics_dir / "paired_mcnemar.json", paired)

    external = None
    if config["external_validation_enabled"]:
        if "binary" not in active_tasks:
            raise ValueError(
                "External binary validation requires an active binary head."
            )
        external = evaluate_external_binary(
            model,
            config,
            device,
            amp_dtype,
            run_dir,
            fold_name,
        )

    export_fold_visualizations(
        history,
        test_predictions,
        test_metrics,
        split_frames,
        config,
        run_dir,
        fold_name,
    )
    split_summary = json.loads(
        (
            run_dir / "splits" / fold_name / "summary.json"
        ).read_text(encoding="utf-8")
    )
    build_fold_report(
        fold_name,
        config,
        split_summary,
        test_metrics,
        bootstrap,
        wlc_metrics,
        roi_metrics,
        paired,
        external,
        history,
        run_dir / "reports" / f"{fold_name}_report.md",
    )
    checkpoint_receipt = None
    checkpoint_receipt_json = None
    if config["checkpoint_backend"] == "huggingface":
        hub_config = _hf_checkpoint_config(
            config,
            _hf_checkpoint_path_in_repo(config, fold_name),
        )
        receipt_dir = (
            run_dir / "reports" / "hf_checkpoints" / fold_name
        )
        checkpoint_receipt = checkpoint_hub.publish_best_checkpoint(
            portable_checkpoint_path,
            receipt_dir,
            hub_config,
        )
        checkpoint_receipt_json = str(
            receipt_dir / "hf_checkpoint_receipt.json"
        )
        if temporary_checkpoint_owner is None:
            raise RuntimeError("Temporary checkpoint lifecycle was not created.")
        temporary_checkpoint_owner.cleanup()
        temporary_checkpoint_owner = None
        local_models = sorted(run_dir.rglob("*.pt"))
        if local_models:
            raise RuntimeError(
                "Remote-only checkpoint policy violation; local model files "
                f"remain: {[str(path) for path in local_models]}"
            )
        logger.info(
            "Published verified best_model.pt to Hugging Face commit=%s path=%s",
            checkpoint_receipt["commit_oid"],
            checkpoint_receipt["path_in_repo"],
        )
    return {
        "fold_name": fold_name,
        "test_metrics": test_metrics,
        "bootstrap": bootstrap,
        "wlc_metrics": wlc_metrics,
        "roi_metrics": roi_metrics,
        "external_metrics": external,
        "model_info": model_info,
        "history_rows": len(history),
        "hf_checkpoint_receipt": checkpoint_receipt,
        "hf_checkpoint_receipt_json": checkpoint_receipt_json,
    }


def main(
    config: Mapping[str, Any] | None = None,
    required_source_files: Sequence[Path | str] | None = None,
) -> Path:
    from cystods.config import load_config
    if config is None:
        config = load_config()
    else:
        config = normalize_core_config(config)
    validate_config(config)
    if config["checkpoint_backend"] == "huggingface":
        _hf_checkpoint_config(
            config,
            f"{str(config['hf_path_prefix']).rstrip('/')}/"
            "preflight/best_model.pt",
        )
    if required_source_files is None:
        try:
            core_source = Path(__file__).resolve()
        except NameError as exc:
            raise RuntimeError(
                "Core provenance requires explicit required_source_files "
                "when __file__ is unavailable."
            ) from exc
        science_src = (
            core_source.with_name("science.py")
            if core_source.with_name("science.py").exists()
            else core_source.with_name("cystods_science.py")
        )
        hf_src = (
            core_source.with_name("hf.py")
            if core_source.with_name("hf.py").exists()
            else core_source.with_name("cystods_hf.py")
        )
        required_source_files = (
            core_source,
            science_src,
            hf_src,
        )
    validated_sources = validate_source_files(required_source_files)
    run_dir = make_run_directory(config)
    logger = setup_logger(run_dir / "logs" / "training.log")
    status_path = run_dir / "run_status.json"
    started_utc = utc_now_iso()
    write_json(
        status_path,
        {
            "status": "running",
            "started_utc": started_utc,
            "run_dir": run_dir,
        },
    )
    try:
        device = resolve_device(config)
        precision_name, amp_dtype = resolve_precision(config, device)
        torch.set_num_threads(int(config["num_cpu_threads"]))
        torch.set_float32_matmul_precision(
            str(config["float32_matmul_precision"])
        )
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = bool(
                config["enable_tf32"]
            )
            torch.backends.cudnn.allow_tf32 = bool(config["enable_tf32"])
        seed_everything(
            int(config["seed"]), bool(config["deterministic"])
        )
        system_info = collect_system_info(
            config, device, precision_name
        )
        write_json(run_dir / "config.json", config)
        snapshot_source_files(run_dir, validated_sources)
        write_json(run_dir / "system" / "environment.json", system_info)
        write_json(
            run_dir / "system" / "dependency_audit.json",
            DEPENDENCY_AUDIT,
        )
        write_json(
            run_dir / "reports" / "taxonomy.json",
            {
                "binary_names": BINARY_NAMES,
                "coarse_names": COARSE_NAMES,
                "fine_names": FINE_NAMES,
                "fine_by_parent": FINE_BY_PARENT,
                "normal_mucosa_fine_policy": (
                    "fine_id=-1; excluded from the 22-subclass loss/metrics"
                ),
            },
        )
        with (run_dir / "system" / "pip_freeze.txt").open(
            "w", encoding="utf-8"
        ) as handle:
            _subprocess.run(
                [_sys.executable, "-m", "pip", "freeze"],
                check=True,
                stdout=handle,
                text=True,
            )
        logger.info(
            "Run started: profile=%s device=%s precision=%s output=%s",
            config["run_profile"],
            device,
            precision_name,
            run_dir,
        )

        manifest, audit = load_and_validate_manifest(
            config, run_dir, logger
        )
        expected_dataset_hash = config["expected_dataset_semantic_sha256"]
        if expected_dataset_hash is not None:
            active_dataset_fingerprint = json.loads(
                (run_dir / "system" / "dataset_fingerprint.json").read_text(
                    encoding="utf-8"
                )
            )
            active_dataset_hash = active_dataset_fingerprint[
                "semantic_manifest_sha256"
            ]
            if active_dataset_hash != expected_dataset_hash:
                raise ValueError(
                    "Active dataset differs from the Stage 00 audited "
                    "dataset: expected semantic SHA-256="
                    f"{expected_dataset_hash}, actual={active_dataset_hash}."
                )
        protocols = build_all_protocol_splits(
            manifest, config, run_dir, logger
        )
        fold_results = []
        for fold_name, split_frames, _ in protocols:
            logger.info("Starting protocol unit: %s", fold_name)
            fold_results.append(
                run_single_fold(
                    fold_name,
                    split_frames,
                    config,
                    device,
                    amp_dtype,
                    run_dir,
                    logger,
                )
            )
        if len(fold_results) > 1:
            cv_summary = aggregate_cross_validation_metrics(
                [result["test_metrics"] for result in fold_results]
            )
            write_json(
                run_dir / "metrics" / "cross_validation_summary.json",
                cv_summary,
            )
        else:
            cv_summary = None
        write_json(
            run_dir / "reports" / "run_summary.json",
            {
                "data_audit": audit,
                "fold_results": fold_results,
                "cross_validation": cv_summary,
            },
        )
        write_json(
            status_path,
            {
                "status": "completed",
                "started_utc": started_utc,
                "completed_utc": utc_now_iso(),
                "run_dir": run_dir,
                "folds_completed": len(fold_results),
            },
        )
        logger.info(
            "Run completed successfully: run_dir=%s folds=%d",
            run_dir,
            len(fold_results),
        )
        return run_dir
    except Exception as exc:
        logger.exception("Run failed with a real error; no fallback was used.")
        write_json(
            status_path,
            {
                "status": "failed",
                "failed_utc": utc_now_iso(),
                "run_dir": run_dir,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        logger.error(
            "FINAL_STATUS failed run_dir=%s error_type=%s",
            run_dir,
            type(exc).__name__,
        )
        raise
    finally:
        close_logger(logger)
        write_artifact_manifest(run_dir)


def _validate_stage_config_keys(
    config: Mapping[str, Any],
    expected_keys: set[str],
) -> None:
    missing = expected_keys - set(config)
    unknown = set(config) - expected_keys
    if missing or unknown:
        raise KeyError(
            "Stage config schema mismatch: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if config["schema_version"] != "cystods.stage.v2":
        raise ValueError("Stage schema_version must be cystods.stage.v2.")
    if config["run_profile"] not in {"research", "smoke"}:
        raise ValueError("Stage run_profile must be research or smoke.")


def _complete_stage_source_files(
    required_source_files: Sequence[Path | str],
) -> tuple[Path, ...]:
    try:
        runner_source = Path(__file__).resolve()
    except NameError as exc:
        raise RuntimeError(
            "Stage execution requires an on-disk source file."
        ) from exc
    package_root = runner_source.parent.parent

    core_source = package_root / "core.py"
    science_src = package_root / "science.py"
    hf_src = package_root / "hf.py"

    candidates = [
        *required_source_files,
        core_source,
        science_src,
        hf_src,
    ]
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = Path(candidate).expanduser().resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return validate_source_files(unique)


def _write_stage_system_artifacts(
    run_dir: Path,
    stage_config: Mapping[str, Any],
    source_files: Sequence[Path | str],
) -> None:
    write_json(run_dir / "config.json", stage_config)
    snapshot_source_files(run_dir, source_files)
    write_json(
        run_dir / "system" / "dependency_audit.json",
        DEPENDENCY_AUDIT,
    )
    write_json(
        run_dir / "system" / "environment.json",
        {
            "python": _sys.version,
            "executable": _sys.executable,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "timm": timm.__version__,
        },
    )
    with (run_dir / "system" / "pip_freeze.txt").open(
        "w", encoding="utf-8"
    ) as handle:
        _subprocess.run(
            [_sys.executable, "-m", "pip", "freeze"],
            check=True,
            stdout=handle,
            text=True,
        )


def run_protocol_stage(
    stage_config: Mapping[str, Any],
    required_source_files: Sequence[Path | str],
) -> Path:
    expected_keys = {
        "schema_version",
        "stage_name",
        "study_id",
        "run_profile",
        "data_root",
        "result_root",
        "protocol_config",
    }
    _validate_stage_config_keys(stage_config, expected_keys)
    source_files = _complete_stage_source_files(required_source_files)
    config = normalize_core_config(stage_config["protocol_config"])
    config.update(
        {
            "stage_name": str(stage_config["stage_name"]),
            "study_id": str(stage_config["study_id"]),
            "run_profile": str(stage_config["run_profile"]),
            "data_root": Path(stage_config["data_root"]).resolve(),
            "result_root": Path(stage_config["result_root"]).resolve(),
            "experiment_name": str(stage_config["stage_name"]),
            "protocol_manifest_dir": None,
            "protocol_reference_sha256": None,
            "protocol_role": None,
            "evaluation_scope": "development",
            "suite_trial_id": None,
        }
    )
    data_root = Path(config["data_root"])
    config["metadata_csv"] = data_root / "cystods.csv"
    config["image_dir"] = data_root / "images"
    config["segmentation_dir"] = data_root / "segmentations"
    validate_config(config)
    run_dir = make_run_directory(config)
    logger = setup_logger(run_dir / "logs" / "training.log")
    status_path = run_dir / "run_status.json"
    started_utc = utc_now_iso()
    write_json(
        status_path,
        {
            "status": "running",
            "started_utc": started_utc,
            "stage_name": stage_config["stage_name"],
            "run_dir": run_dir,
        },
    )
    try:
        _write_stage_system_artifacts(run_dir, stage_config, source_files)
        logger.info(
            "Protocol stage starts profile=%s output=%s",
            config["run_profile"],
            run_dir,
        )
        manifest, audit = load_and_validate_manifest(
            config,
            run_dir,
            logger,
        )
        fixed_config = dict(config)
        fixed_config.update(
            {
                "protocol": "holdout",
                "fixed_split_pids": (
                    config["fixed_split_pids"]
                    if config["run_profile"] == "smoke"
                    else None
                ),
                "cv_run_fold_indices": None,
            }
        )
        fixed_units = build_all_protocol_splits(
            manifest,
            fixed_config,
            run_dir,
            logger,
        )
        if len(fixed_units) != 1 or fixed_units[0][0] != "holdout":
            raise RuntimeError(
                "Protocol stage must create exactly one fixed holdout unit."
            )
        _, fixed_frames, _ = fixed_units[0]
        train_patient_counts = (
            fixed_frames["train"]
            .loc[fixed_frames["train"]["fine_id"] >= 0]
            .groupby("fine_id")["pid"]
            .nunique()
        )
        primary_support_threshold = (
            1
            if config["run_profile"] == "smoke"
            else int(config["primary_fine_min_train_patients"])
        )
        primary_ids = [
            class_id
            for class_id in range(len(FINE_NAMES))
            if int(train_patient_counts.get(class_id, 0))
            >= primary_support_threshold
        ]
        if not primary_ids:
            raise ValueError(
                "Frozen primary fine taxonomy is empty at the configured "
                "patient-support threshold."
            )
        split_summaries = {}
        for unit_dir in sorted((run_dir / "splits").iterdir()):
            summary_path = unit_dir / "summary.json"
            if unit_dir.is_dir() and summary_path.is_file():
                split_summaries[unit_dir.name] = json.loads(
                    summary_path.read_text(encoding="utf-8")
                )
        dataset_fingerprint = json.loads(
            (
                run_dir / "system" / "dataset_fingerprint.json"
            ).read_text(encoding="utf-8")
        )
        protocol_manifest = {
            "schema_version": "cystods.protocol.v2",
            "study_id": str(stage_config["study_id"]),
            "created_utc": utc_now_iso(),
            "run_profile": config["run_profile"],
            "dataset_fingerprint": dataset_fingerprint,
            "primary_fine_class_ids": primary_ids,
            "primary_fine_class_names": [
                FINE_NAMES[index] for index in primary_ids
            ],
            "primary_fine_min_train_patients": primary_support_threshold,
            "primary_taxonomy_policy": (
                "functional_smoke_train_support_ge_1"
                if config["run_profile"] == "smoke"
                else "preregistered_train_patient_support"
            ),
            "roles": {
                "fixed_holdout": {
                    "protocol": "holdout",
                    "units": ["holdout"],
                },
                "smoke_holdout": {
                    "protocol": "holdout",
                    "units": ["holdout"],
                },
            },
            "split_summaries": split_summaries,
            "data_audit_sha256": sha256_file(
                run_dir / "reports" / "data_audit.json"
            ),
        }
        protocol_path = run_dir / "protocol_manifest.json"
        write_json(protocol_path, protocol_manifest)
        protocol_hash = sha256_file(protocol_path)
        write_json(
            run_dir / "reports" / "protocol_reference.json",
            {
                "protocol_manifest": str(protocol_path),
                "protocol_sha256": protocol_hash,
                "downstream_environment": {
                    "CYSTODS_PROTOCOL_RUN_DIR": str(run_dir),
                    "CYSTODS_EXPECTED_PROTOCOL_SHA256": protocol_hash,
                },
            },
        )
        write_json(
            run_dir / "reports" / "run_summary.json",
            {
                "data_audit": audit,
                "fixed_holdout_units": ["holdout"],
                "protocol_sha256": protocol_hash,
            },
        )
        write_json(
            status_path,
            {
                "status": "completed",
                "started_utc": started_utc,
                "completed_utc": utc_now_iso(),
                "stage_name": stage_config["stage_name"],
                "run_dir": run_dir,
                "protocol_sha256": protocol_hash,
            },
        )
        logger.info(
            "Protocol stage completed successfully: protocol_sha256=%s "
            "run_dir=%s",
            protocol_hash,
            run_dir,
        )
        return run_dir
    except Exception as exc:
        logger.exception("Protocol stage failed; no fallback was used.")
        write_json(
            status_path,
            {
                "status": "failed",
                "started_utc": started_utc,
                "failed_utc": utc_now_iso(),
                "stage_name": stage_config["stage_name"],
                "run_dir": run_dir,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        logger.error(
            "FINAL_STATUS failed error_type=%s run_dir=%s",
            type(exc).__name__,
            run_dir,
        )
        raise
    finally:
        write_artifact_manifest(run_dir)


def run_external_validation_stage(
    stage_config: Mapping[str, Any],
    required_source_files: Sequence[Path | str],
) -> Path:
    expected_keys = {
        "schema_version",
        "stage_name",
        "study_id",
        "run_profile",
        "data_root",
        "result_root",
        "selected_run_dir",
        "hf_checkpoint_receipt_json",
        "external_manifest_csv",
        "external_image_root",
        "internal_protocol_run_dir",
        "base_config",
        "external_columns",
    }
    _validate_stage_config_keys(stage_config, expected_keys)
    source_files = _complete_stage_source_files(required_source_files)
    selected_run_dir = Path(stage_config["selected_run_dir"]).expanduser().resolve()
    receipt_json_path = Path(
        stage_config["hf_checkpoint_receipt_json"]
    ).expanduser().resolve()
    external_manifest_csv = Path(
        stage_config["external_manifest_csv"]
    ).expanduser().resolve()
    external_image_root = Path(
        stage_config["external_image_root"]
    ).expanduser().resolve()
    protocol_run_dir = Path(
        stage_config["internal_protocol_run_dir"]
    ).expanduser().resolve()
    for directory in (
        selected_run_dir,
        external_image_root,
        protocol_run_dir,
    ):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    for file_path in (receipt_json_path, external_manifest_csv):
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
    if not receipt_json_path.is_relative_to(selected_run_dir):
        raise ValueError(
            "hf_checkpoint_receipt_json must be contained in "
            "selected_run_dir."
        )
    selected_status_path = selected_run_dir / "run_status.json"
    selected_artifact_path = selected_run_dir / "artifact_manifest.json"
    if not selected_status_path.is_file() or not selected_artifact_path.is_file():
        raise FileNotFoundError(
            "Selected run must contain run_status.json and artifact_manifest.json."
        )
    selected_status = json.loads(
        selected_status_path.read_text(encoding="utf-8")
    )
    if selected_status.get("status") != "completed":
        raise ValueError("Selected model run is not completed.")
    artifact_rows = json.loads(
        selected_artifact_path.read_text(encoding="utf-8")
    )
    relative_receipt = receipt_json_path.relative_to(
        selected_run_dir
    ).as_posix()
    artifact_row = next(
        (
            row
            for row in artifact_rows
            if row.get("path") == relative_receipt
        ),
        None,
    )
    if artifact_row is None:
        raise ValueError(
            "Selected HF receipt is absent from the artifact manifest."
        )
    if (
        int(artifact_row["bytes"]) != receipt_json_path.stat().st_size
        or artifact_row["sha256"] != sha256_file(receipt_json_path)
    ):
        raise ValueError("Selected HF receipt fails artifact validation.")
    receipt = checkpoint_hub.load_and_validate_receipt(receipt_json_path)
    protocol_path = protocol_run_dir / "protocol_manifest.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol_hash = sha256_file(protocol_path)
    checkpoint_temp_owner = tempfile.TemporaryDirectory(
        prefix="cystods-stage60-checkpoint-"
    )
    checkpoint_path = checkpoint_hub.download_verified_checkpoint(
        receipt_json_path,
        Path(checkpoint_temp_owner.name) / "checkpoint",
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint payload must be a mapping.")
    required_checkpoint_keys = {
        "model_state_dict",
        "config",
        "binary_names",
        "coarse_names",
        "fine_names",
        "data_split_fingerprint",
    }
    missing_checkpoint = required_checkpoint_keys - set(checkpoint)
    if missing_checkpoint:
        raise ValueError(
            f"Checkpoint missing keys: {sorted(missing_checkpoint)}"
        )
    if tuple(checkpoint["binary_names"]) != BINARY_NAMES:
        raise ValueError("Checkpoint binary taxonomy mismatch.")
    if tuple(checkpoint["coarse_names"]) != COARSE_NAMES:
        raise ValueError("Checkpoint coarse taxonomy mismatch.")
    if tuple(checkpoint["fine_names"]) != FINE_NAMES:
        raise ValueError("Checkpoint fine taxonomy mismatch.")
    saved_config = normalize_core_config(checkpoint["config"])
    if saved_config["study_id"] != stage_config["study_id"]:
        raise ValueError("Selected checkpoint study_id mismatch.")
    if saved_config["protocol_reference_sha256"] != protocol_hash:
        raise ValueError(
            "Selected checkpoint is not bound to internal_protocol_run_dir."
        )
    if saved_config["hf_repo_id"] != receipt["repo_id"]:
        raise ValueError("Checkpoint config and HF receipt repository differ.")
    if saved_config["evaluation_scope"] not in {"development", "final_cv"}:
        raise ValueError(
            "External validation requires a checkpoint from fixed hold-out "
            "development or final cross-validation."
        )
    if "binary" not in active_tasks_from_config(saved_config):
        raise ValueError("External validation requires a trained binary head.")
    columns = dict(stage_config["external_columns"])
    if set(columns) != {
        "path_column",
        "binary_label_column",
        "patient_id_column",
    }:
        raise KeyError("external_columns schema mismatch.")
    runtime_config = normalize_core_config(stage_config["base_config"])
    evaluation_config = dict(saved_config)
    for runtime_key in (
        "batch_size",
        "eval_batch_size",
        "num_workers",
        "eval_num_workers",
        "prefetch_factor",
        "eval_prefetch_factor",
        "persistent_workers",
        "pin_memory",
        "device",
        "precision",
        "enable_tf32",
        "channels_last",
        "torch_compile",
        "torch_compile_mode",
        "float32_matmul_precision",
        "num_cpu_threads",
        "binary_decision_threshold",
    ):
        evaluation_config[runtime_key] = runtime_config[runtime_key]
    evaluation_config.update(
        {
            "stage_name": str(stage_config["stage_name"]),
            "run_profile": str(stage_config["run_profile"]),
            "result_root": Path(stage_config["result_root"]).resolve(),
            "experiment_name": str(stage_config["stage_name"]),
            "external_validation_enabled": True,
            "external_manifest_csv": external_manifest_csv,
            "external_image_root": external_image_root,
            "external_path_column": str(columns["path_column"]),
            "external_binary_label_column": str(
                columns["binary_label_column"]
            ),
            "external_patient_id_column": str(
                columns["patient_id_column"]
            ),
            "evaluation_scope": "external",
            "pretrained": False,
        }
    )
    validate_config(evaluation_config)
    run_dir = make_run_directory(evaluation_config)
    logger = setup_logger(run_dir / "logs" / "training.log")
    status_path = run_dir / "run_status.json"
    started_utc = utc_now_iso()
    write_json(
        status_path,
        {
            "status": "running",
            "started_utc": started_utc,
            "stage_name": stage_config["stage_name"],
            "run_dir": run_dir,
            "protocol_sha256": protocol_hash,
            "selected_checkpoint_sha256": receipt["checkpoint_sha256"],
        },
    )
    try:
        _write_stage_system_artifacts(run_dir, stage_config, source_files)
        write_json(run_dir / "evaluation_config.json", evaluation_config)
        write_json(
            run_dir / "system" / "selected_model_binding.json",
            {
                "selected_run_dir": str(selected_run_dir),
                "hf_checkpoint_receipt_json": str(receipt_json_path),
                "hf_receipt_sha256": artifact_row["sha256"],
                "hf_repo_id": receipt["repo_id"],
                "hf_path_in_repo": receipt["path_in_repo"],
                "hf_commit_oid": receipt["commit_oid"],
                "checkpoint_sha256": receipt["checkpoint_sha256"],
                "selected_artifact_manifest_sha256": sha256_file(
                    selected_artifact_path
                ),
                "protocol_run_dir": str(protocol_run_dir),
                "protocol_sha256": protocol_hash,
            },
        )
        device = resolve_device(evaluation_config)
        precision_name, amp_dtype = resolve_precision(
            evaluation_config,
            device,
        )
        torch.set_num_threads(int(evaluation_config["num_cpu_threads"]))
        torch.set_float32_matmul_precision(
            str(evaluation_config["float32_matmul_precision"])
        )
        seed_everything(
            int(evaluation_config["seed"]),
            bool(evaluation_config["deterministic"]),
        )
        model = HierarchicalCystoModel(evaluation_config).to(device)
        if evaluation_config["channels_last"]:
            model = model.to(memory_format=torch.channels_last)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if evaluation_config["torch_compile"]:
            if not hasattr(torch, "compile"):
                raise RuntimeError(
                    "Checkpoint requires torch.compile, which is unavailable."
                )
            model.encoder = torch.compile(
                model.encoder,
                mode=str(evaluation_config["torch_compile_mode"]),
            )
        metrics_dir = run_dir / "metrics" / "external"
        prediction_dir = run_dir / "predictions" / "external"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        prediction_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "External evaluation starts device=%s precision=%s checkpoint=%s",
            device,
            precision_name,
            checkpoint_path,
        )
        metrics = evaluate_external_binary(
            model,
            evaluation_config,
            device,
            amp_dtype,
            run_dir,
            "external",
        )
        write_json(
            run_dir / "reports" / "run_summary.json",
            {
                "protocol_sha256": protocol_hash,
                "selected_checkpoint_sha256": receipt[
                    "checkpoint_sha256"
                ],
                "hf_checkpoint_receipt": receipt,
                "external_manifest_sha256": sha256_file(
                    external_manifest_csv
                ),
                "external_metrics": metrics,
                "fine_tuning_performed": False,
            },
        )
        pd.DataFrame(
            [
                {"metric": key, "value": value}
                for key, value in metrics.items()
                if isinstance(value, (int, float, type(None)))
            ]
        ).to_csv(
            run_dir / "reports" / "external_metrics.csv",
            index=False,
        )
        report = "\n".join(
            [
                "# External validation report",
                "",
                "- Evaluation-only: `true`",
                "- Fine-tuning performed: `false`",
                f"- Protocol SHA-256: `{protocol_hash}`",
                f"- HF commit: `{receipt['commit_oid']}`",
                f"- Checkpoint SHA-256: `{receipt['checkpoint_sha256']}`",
                "",
                "## Binary metrics",
                "",
                pd.DataFrame(
                    [
                        {"metric": key, "value": value}
                        for key, value in metrics.items()
                        if isinstance(value, (int, float, type(None)))
                    ]
                ).to_markdown(index=False),
                "",
            ]
        )
        (run_dir / "reports" / "external_report.md").write_text(
            report,
            encoding="utf-8",
        )
        write_json(
            status_path,
            {
                "status": "completed",
                "started_utc": started_utc,
                "completed_utc": utc_now_iso(),
                "stage_name": stage_config["stage_name"],
                "run_dir": run_dir,
                "protocol_sha256": protocol_hash,
                "selected_checkpoint_sha256": receipt[
                    "checkpoint_sha256"
                ],
            },
        )
        logger.info(
            "External validation completed successfully: run_dir=%s",
            run_dir,
        )
        return run_dir
    except Exception as exc:
        logger.exception("External validation failed; no fallback was used.")
        write_json(
            status_path,
            {
                "status": "failed",
                "started_utc": started_utc,
                "failed_utc": utc_now_iso(),
                "stage_name": stage_config["stage_name"],
                "run_dir": run_dir,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        logger.error(
            "FINAL_STATUS failed error_type=%s run_dir=%s",
            type(exc).__name__,
            run_dir,
        )
        raise
    finally:
        close_logger(logger)
        checkpoint_temp_owner.cleanup()
        local_models = sorted(run_dir.rglob("*.pt"))
        if local_models:
            raise RuntimeError(
                "External validation left local checkpoint files in result: "
                f"{[str(path) for path in local_models]}"
            )
