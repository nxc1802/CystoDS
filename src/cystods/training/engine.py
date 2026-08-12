"""Training loop execution, evaluation engine, and suite runner.

Extracted from ``cystods.core`` (Step 6 refactor).
"""

from __future__ import annotations

import gc
import json
import logging
import math
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import cystods.science as science
from cystods.data.dataset import CystoDataset
from cystods.data.sampler import make_worker_init_fn
from cystods.data.transforms import build_transforms
from cystods.data.splits.protocol import _load_and_validate_protocol_binding
from cystods.evaluation.calibration import (
    apply_fine_inference_calibration,
    select_fine_inference_tau,
)
from cystods.evaluation.metrics import (
    compute_metrics_bundle,
    metric_for_monitor,
)
from cystods.evaluation.roi import run_roi_evaluation
from cystods.infra.environment import close_logger, setup_logger
from cystods.infra.serialization import (
    json_ready,
    sha256_file,
    utc_now_iso,
    write_json,
)
from cystods.training.runtime import (
    resolve_device,
    resolve_precision,
    seed_everything,
)
from cystods.losses.classification import FineLongTailLoss
from cystods.losses.composite import (
    active_fine_loss_name,
    compute_multitask_loss,
)
from cystods.losses.supcon import supervised_contrastive_loss
from cystods.models.factory import active_tasks_from_config


def _get_core_attr(name: str, fallback: Any = None) -> Any:
    import sys
    core_mod = sys.modules.get("cystods.core")
    if core_mod is not None and hasattr(core_mod, name):
        return getattr(core_mod, name)
    return fallback
from cystods.models.hierarchical import HierarchicalCystoModel
from cystods.taxonomy import BINARY_NAMES, COARSE_NAMES, FINE_NAMES, FINE_PARENT_ID
from cystods.training.checkpoint import (
    load_checkpoint_for_resume,
    save_checkpoint,
)
from cystods.training.optimizer import build_optimizer, build_scheduler


def forward_with_precision(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> dict[str, torch.Tensor]:
    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=amp_dtype is not None,
    ):
        outputs = model(images)
    return {
        key: value.float()
        for key, value in outputs.items()
        if isinstance(value, torch.Tensor)
    }


def move_images(
    images: torch.Tensor,
    device: torch.device,
    config: Mapping[str, Any],
) -> torch.Tensor:
    tensor = images.to(device=device, non_blocking=True)
    if config["channels_last"]:
        tensor = tensor.to(memory_format=torch.channels_last)
    return tensor


def move_target(
    targets: torch.Tensor,
    device: torch.device,
    config: Mapping[str, Any],
) -> torch.Tensor:
    return targets.to(device=device, non_blocking=True)


def prediction_rows_from_outputs(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    include_features: bool,
    fine_loss_fn: FineLongTailLoss | None,
    fine_prior_tau: float,
) -> list[dict[str, Any]]:
    logit_arrays: dict[str, np.ndarray] = {}
    probability_arrays: dict[str, np.ndarray] = {}
    for task_name in ("binary", "coarse"):
        key = f"{task_name}_logits"
        if key in outputs:
            logits = outputs[key]
            logit_arrays[task_name] = (
                logits.detach().float().cpu().numpy()
            )
            probability_arrays[task_name] = (
                logits.softmax(dim=1).detach().float().cpu().numpy()
            )
    if "fine_logits" in outputs:
        if fine_loss_fn is None:
            raise ValueError("Fine output requires a configured fine loss.")
        raw_fine_logits = outputs["fine_logits"]
        calibrated_fine_logits = fine_loss_fn.inference_logits(
            raw_fine_logits,
            fine_prior_tau,
        )
        logit_arrays["fine"] = (
            raw_fine_logits.detach().float().cpu().numpy()
        )
        probability_arrays["fine"] = (
            calibrated_fine_logits.softmax(dim=1)
            .detach()
            .float()
            .cpu()
            .numpy()
        )
    features = (
        outputs["features"].detach().float().cpu().numpy()
        if include_features and "features" in outputs
        else None
    )
    rows: list[dict[str, Any]] = []
    batch_size = len(batch["filename"])
    for index in range(batch_size):
        row = {
            "filename": batch["filename"][index],
            "pid": batch["pid"][index],
            "visit": batch.get("visit", ["NA"] * batch_size)[index],
            "lesion": batch.get("lesion", ["NA"] * batch_size)[index],
            "modality": batch.get(
                "modality", ["unknown"] * batch_size
            )[index],
            "binary_id": int(batch["binary_id"][index]),
            "coarse_id": int(batch["coarse_id"][index]),
            "fine_id": int(batch["fine_id"][index]),
            "fine_inference_prior_tau": float(fine_prior_tau),
        }
        for task_name in sorted(logit_arrays):
            row[f"{task_name}_logits"] = logit_arrays[task_name][index]
            row[f"{task_name}_probs"] = probability_arrays[task_name][index]
        if features is not None:
            row["features"] = features[index]
        rows.append(row)
    return rows


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    fine_loss_fn: FineLongTailLoss | None,
    fine_counts: Sequence[int],
    fine_patient_counts: Sequence[int],
    config: Mapping[str, Any],
    device: torch.device,
    amp_dtype: torch.dtype | None,
    include_features: bool = False,
    fine_prior_tau: float = 0.0,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, float]]:
    model.eval()
    active_tasks = active_tasks_from_config(config)
    accumulated_rows: list[dict[str, Any]] = []
    loss_sums: dict[str, float] = {}
    total_samples = 0
    with torch.inference_mode():
        for batch in loader:
            images = move_images(batch["image"], device, config)
            binary_targets = move_target(
                batch["binary_id"], device, config
            )
            coarse_targets = move_target(
                batch["coarse_id"], device, config
            )
            fine_targets = move_target(batch["fine_id"], device, config)
            outputs = forward_with_precision(
                model, images, device, amp_dtype
            )
            classification_loss, components = compute_multitask_loss(
                outputs,
                binary_targets,
                coarse_targets,
                fine_targets,
                fine_loss_fn,
                config,
            )
            batch_size = len(images)
            total_samples += batch_size
            losses = {"total_loss": classification_loss, **components}
            for key, tensor in losses.items():
                loss_sums[key] = loss_sums.get(key, 0.0) + float(
                    tensor.detach()
                ) * batch_size
            rows = prediction_rows_from_outputs(
                outputs,
                batch,
                include_features=include_features,
                fine_loss_fn=fine_loss_fn,
                fine_prior_tau=fine_prior_tau,
            )
            accumulated_rows.extend(rows)

    predictions = pd.DataFrame(accumulated_rows)
    if (
        "fine" in active_tasks
        and fine_loss_fn is not None
        and fine_prior_tau != 0.0
    ):
        predictions = apply_fine_inference_calibration(
            predictions,
            fine_loss_fn,
            fine_prior_tau,
        )

    metrics = compute_metrics_bundle(
        predictions,
        fine_counts,
        fine_patient_counts,
        config,
    )
    mean_losses = {
        key: value / max(total_samples, 1) for key, value in loss_sums.items()
    }
    metrics["losses"] = mean_losses
    metrics["fine_inference_prior_tau"] = float(fine_prior_tau)
    return metrics, predictions, mean_losses


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


def train_model(
    model: HierarchicalCystoModel,
    loaders: Mapping[str, DataLoader],
    split_frames: Mapping[str, pd.DataFrame],
    optimization_train_frame: pd.DataFrame,
    config: Mapping[str, Any],
    device: torch.device,
    amp_dtype: torch.dtype | None,
    fold_dir: Path,
    run_dir: Path,
    data_split_hash: str,
    logger: logging.Logger,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], Path]:
    active_tasks = active_tasks_from_config(config)
    fine_counts = (
        optimization_train_frame["fine_id"]
        .value_counts()
        .reindex(range(len(FINE_NAMES)), fill_value=0)
        .to_numpy(dtype=np.int64)
    )
    fine_patient_counts = np.zeros(len(FINE_NAMES), dtype=np.int64)
    valid_fine_train = optimization_train_frame.loc[
        optimization_train_frame["fine_id"] >= 0
    ]
    if not valid_fine_train.empty:
        for fine_id, group in valid_fine_train.groupby(
            "fine_id", sort=False
        ):
            fine_patient_counts[int(fine_id)] = group["pid"].nunique()

    fine_loss_fn = (
        FineLongTailLoss(
            active_fine_loss_name(config),
            fine_counts,
            fine_patient_counts,
            config,
        ).to(device)
        if "fine" in active_tasks
        else None
    )
    if fine_loss_fn is not None:
        write_json(fold_dir / "fine_prior_audit.json", fine_loss_fn.prior_audit())
    optimizer = build_optimizer(model, config, device)
    scheduler, total_updates = build_scheduler(
        optimizer, len(loaders["train"]), config
    )
    use_scaler = amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)

    start_epoch = 0
    best_metric = -float("inf")
    selected_fine_tau = float(config["fine_inference_prior_tau"])
    resume = config["resume_checkpoint"]
    if resume is not None:
        start_epoch, best_metric, selected_fine_tau = load_checkpoint_for_resume(
            Path(resume),
            model,
            optimizer,
            scheduler,
            scaler,
            device,
            config,
            data_split_hash,
        )
        logger.info(
            "Resumed from %s at epoch=%d best=%.6f",
            resume,
            start_epoch,
            best_metric,
        )
        if start_epoch >= int(config["epochs"]):
            raise ValueError(
                "resume_checkpoint already reached the configured epoch "
                "count; increase epochs to continue training."
            )

    accumulation = int(config["gradient_accumulation_steps"])
    history: list[dict[str, Any]] = []
    epochs_without_improvement = 0
    last_completed_epoch = start_epoch - 1
    optimizer.zero_grad(set_to_none=True)
    global_update = 0
    logger.info(
        "Training starts: epochs=%d batches/epoch=%d total_updates=%d",
        config["epochs"],
        len(loaders["train"]),
        total_updates,
    )

    for epoch in range(start_epoch, int(config["epochs"])):
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_start = time.perf_counter()
        epoch_samples = 0
        running: dict[str, torch.Tensor] = {}
        for batch_index, batch in enumerate(loaders["train"]):
            images_one = move_images(batch["image"], device, config)
            has_second_view = "image_view_2" in batch
            if has_second_view:
                images_two = move_images(
                    batch["image_view_2"], device, config
                )
                images = torch.cat((images_one, images_two), dim=0)
            else:
                images = images_one
            binary_targets_one = move_target(
                batch["binary_id"], device, config
            )
            coarse_targets_one = move_target(
                batch["coarse_id"], device, config
            )
            fine_targets_one = move_target(
                batch["fine_id"], device, config
            )
            primary_batch_size = len(images_one)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                outputs = model(images)
                outputs_primary = {
                    key: value[:primary_batch_size]
                    for key, value in outputs.items()
                }
                classification_loss, components = compute_multitask_loss(
                    outputs_primary,
                    binary_targets_one,
                    coarse_targets_one,
                    fine_targets_one,
                    fine_loss_fn,
                    config,
                )
                supcon_weight = float(
                    config["supervised_contrastive_loss_weight"]
                )
                if supcon_weight > 0:
                    if not has_second_view:
                        raise RuntimeError(
                            "SupCon is enabled but the second view is absent."
                        )
                    repeat = 2 if has_second_view else 1
                    supcon_fine_targets = fine_targets_one.repeat(repeat)
                    supcon_coarse_targets = coarse_targets_one.repeat(repeat)
                    if (
                        config["supervised_contrastive_label_level"]
                        == "fine"
                    ):
                        supcon_valid = supcon_fine_targets >= 0
                        supcon_labels = supcon_fine_targets[supcon_valid]
                        supcon_projection = outputs["projection"][
                            supcon_valid
                        ]
                    elif (
                        config["supervised_contrastive_label_level"]
                        == "coarse"
                    ):
                        supcon_labels = supcon_coarse_targets
                        supcon_projection = outputs["projection"]
                    else:
                        raise ValueError(
                            "supervised_contrastive_label_level must be "
                            "fine or coarse."
                        )
                    supcon = supervised_contrastive_loss(
                        supcon_projection,
                        supcon_labels,
                        float(
                            config[
                                "supervised_contrastive_temperature"
                            ]
                        ),
                    )
                else:
                    supcon = next(iter(outputs.values())).sum() * 0.0
                total_loss = classification_loss + supcon_weight * supcon
                group_start = (batch_index // accumulation) * accumulation
                group_size = min(
                    accumulation,
                    len(loaders["train"]) - group_start,
                )
                scaled_loss = total_loss / group_size

            scaler.scale(scaled_loss).backward()
            should_step = (
                (batch_index + 1) % accumulation == 0
                or batch_index + 1 == len(loaders["train"])
            )
            if should_step:
                if float(config["gradient_clip_norm"]) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        float(config["gradient_clip_norm"]),
                        error_if_nonfinite=True,
                    )
                else:
                    scaler.unscale_(optimizer)
                    for parameter in model.parameters():
                        if (
                            parameter.grad is not None
                            and not torch.isfinite(parameter.grad).all()
                        ):
                            raise FloatingPointError(
                                "Non-finite gradient detected."
                            )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_update += 1

            base_batch_size = len(images_one)
            epoch_samples += base_batch_size
            for name, value in {
                "total_loss": total_loss,
                "supcon_loss": supcon,
                **components,
            }.items():
                contribution = value.detach() * base_batch_size
                if name in running:
                    running[name].add_(contribution)
                else:
                    running[name] = contribution

            if (
                (batch_index + 1) % int(config["log_every_n_steps"]) == 0
                or batch_index + 1 == len(loaders["train"])
            ):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                running_loss = float(running["total_loss"].item())
                if not math.isfinite(running_loss):
                    raise FloatingPointError(
                        "Non-finite training loss detected."
                    )
                elapsed = max(time.perf_counter() - epoch_start, 1e-9)
                logger.info(
                    "train epoch=%d/%d step=%d/%d update=%d "
                    "loss=%.5f samples/s=%.2f lr_head=%.3e",
                    epoch + 1,
                    config["epochs"],
                    batch_index + 1,
                    len(loaders["train"]),
                    global_update,
                    running_loss / epoch_samples,
                    epoch_samples / elapsed,
                    optimizer.param_groups[-1]["lr"],
                )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_elapsed = time.perf_counter() - epoch_start
        val_metrics, val_predictions, val_losses = evaluate_model(
            model,
            loaders["val"],
            fine_loss_fn,
            fine_counts,
            fine_patient_counts,
            config,
            device,
            amp_dtype,
            include_features=False,
            fine_prior_tau=selected_fine_tau,
        )
        calibration_audit = None
        if "fine" in active_tasks:
            if fine_loss_fn is None:
                raise RuntimeError("Fine task lost its configured loss.")
            (
                val_metrics,
                val_predictions,
                selected_fine_tau,
                calibration_audit,
            ) = select_fine_inference_tau(
                val_predictions,
                fine_loss_fn,
                fine_counts,
                fine_patient_counts,
                config,
            )

        monitored_score = metric_for_monitor(
            val_metrics,
            str(config["monitor_metric"]),
            config.get("hierarchical_composite_weights"),
        )
        last_completed_epoch = epoch
        epoch_row = {
            "epoch": epoch + 1,
            "train_time_sec": train_elapsed,
            "monitored_score": monitored_score,
            "fine_inference_prior_tau": selected_fine_tau,
            "learning_rate_encoder": optimizer.param_groups[0]["lr"],
            "learning_rate_head": optimizer.param_groups[-1]["lr"],
        }
        for name, value in running.items():
            epoch_row[f"train_{name}"] = float(value.item()) / epoch_samples
        for name, value in val_losses.items():
            epoch_row[f"val_{name}"] = float(value)
        history.append(epoch_row)

        save_checkpoint(
            run_dir / "checkpoints" / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_metric,
            config,
            fine_counts,
            fine_patient_counts,
            fine_loss_fn,
            selected_fine_tau,
            data_split_hash,
        )

        improved = monitored_score > best_metric
        if improved:
            best_metric = float(monitored_score)
            epochs_without_improvement = 0
            save_checkpoint(
                fold_dir / "best_model.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_metric,
                config,
                fine_counts,
                fine_patient_counts,
                fine_loss_fn,
                selected_fine_tau,
                data_split_hash,
            )
            logger.info(
                "Saved new best model for %s epoch=%d score=%.6f",
                fold_dir.name,
                epoch + 1,
                monitored_score,
            )
        else:
            epochs_without_improvement += 1

        logger.info(
            "epoch=%d/%d val_score=%.6f best=%.6f non_improving=%d/%d",
            epoch + 1,
            config["epochs"],
            monitored_score,
            best_metric,
            epochs_without_improvement,
            config["early_stopping_patience"],
        )

        if (
            epochs_without_improvement
            >= int(config["early_stopping_patience"])
        ):
            logger.info("Early stopping triggered at epoch %d.", epoch + 1)
            break

    best_checkpoint_path = fold_dir / "best_model.pt"
    if not best_checkpoint_path.is_file():
        raise RuntimeError("Best model checkpoint was not created.")

    checkpoint_payload = torch.load(
        best_checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint_payload["model_state_dict"], strict=True)
    best_fine_tau = float(
        checkpoint_payload.get(
            "fine_inference_prior_tau", selected_fine_tau
        )
    )

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(fold_dir / "history.csv", index=False)

    evaluated_splits: dict[str, pd.DataFrame] = {}
    evaluated_metrics: dict[str, Any] = {
        "best_monitored_score": best_metric,
        "epochs_trained": len(history),
        "last_completed_epoch": last_completed_epoch + 1,
        "data_split_fingerprint": data_split_hash,
        "selected_fine_inference_prior_tau": best_fine_tau,
        "splits": {},
    }
    for name, loader in loaders.items():
        if name == "train":
            loader = make_deterministic_eval_loader(
                split_frames["train"], config, device, int(config["seed"])
            )
        metrics, predictions, _ = evaluate_model(
            model,
            loader,
            fine_loss_fn,
            fine_counts,
            fine_patient_counts,
            config,
            device,
            amp_dtype,
            include_features="attention" in config["roi_aggregations"],
            fine_prior_tau=best_fine_tau,
        )
        evaluated_splits[name] = predictions
        evaluated_metrics["splits"][name] = metrics

    if "attention" in config["roi_aggregations"]:
        evaluated_metrics["roi"] = run_roi_evaluation(
            evaluated_splits,
            config,
            device,
            run_dir,
            fold_dir.name,
            logger,
        )

    write_json(
        run_dir / "metrics" / fold_dir.name / "summary.json",
        evaluated_metrics,
    )
    return evaluated_metrics, evaluated_splits, best_checkpoint_path


def run_training_suite(
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
        "protocol_run_dir",
        "protocol_role",
        "expected_protocol_sha256",
        "seeds",
        "fold_ids",
        "base_config",
        "trials",
        "evaluation_scope",
    }
    _validate_stage_config_keys = _get_core_attr("_validate_stage_config_keys")
    _complete_stage_source_files = _get_core_attr("_complete_stage_source_files")
    _write_stage_system_artifacts = _get_core_attr("_write_stage_system_artifacts")
    _validate_stage_config_keys(stage_config, expected_keys)
    source_files = _complete_stage_source_files(required_source_files)
    protocol_run_dir, protocol_manifest, protocol_hash = (
        _load_and_validate_protocol_binding(stage_config)
    )
    role = str(stage_config["protocol_role"])
    scope = str(stage_config["evaluation_scope"])
    if role in {"fixed_holdout", "smoke_holdout"} and scope != "development":
        raise ValueError(
            f"{role} requires evaluation_scope=development."
        )
    if role == "final_cv" and scope != "final_cv":
        raise ValueError("final_cv requires evaluation_scope=final_cv.")

    _normalize_core_config = _get_core_attr("normalize_core_config")
    _make_run_directory = _get_core_attr("make_run_directory")
    _validate_config = _get_core_attr("validate_config")

    runtime_config = _normalize_core_config(stage_config["base_config"])
    runtime_config.update(
        {
            "stage_name": str(stage_config["stage_name"]),
            "study_id": str(stage_config["study_id"]),
            "run_profile": str(stage_config["run_profile"]),
            "result_root": Path(stage_config["result_root"]).resolve(),
            "experiment_name": str(stage_config["stage_name"]),
            "protocol_role": role,
            "protocol_manifest_dir": protocol_run_dir,
            "protocol_reference_sha256": protocol_hash,
            "evaluation_scope": scope,
        }
    )

    trials_spec = list(stage_config["trials"])
    if not trials_spec:
        raise ValueError("stage_config['trials'] cannot be empty.")

    run_dir = _make_run_directory(runtime_config)
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
        },
    )

    try:
        _write_stage_system_artifacts(run_dir, stage_config, source_files)
        logger.info(
            "Training suite started stage=%s study_id=%s protocol_sha256=%s",
            stage_config["stage_name"],
            stage_config["study_id"],
            protocol_hash,
        )

        from cystods.data.sampler import build_dataloaders
        from cystods.data.splits.protocol import build_all_protocol_splits
        from cystods.data.manifest import load_and_validate_manifest

        # Load manifest & materialize splits ONCE for the entire training suite
        raw_frame, audit = load_and_validate_manifest(
            runtime_config, run_dir, logger
        )

        units = build_all_protocol_splits(
            raw_frame, runtime_config, run_dir, logger
        )

        completed_trials = []
        for trial_index, trial in enumerate(trials_spec):
            trial_name = str(
                trial.get("trial_id", trial.get("experiment_id", trial.get("trial_name", f"trial_{trial_index:02d}")))
            )

            trial_config = dict(runtime_config)
            trial_config["task_mode"] = trial.get(
                "task_mode",
                trial_config["task_mode"],
            )
            trial_config.update(trial.get("overrides", {}))
            trial_config["suite_trial_id"] = trial_name

            _validate_config(trial_config)
            logger.info("Executing trial=%s", trial_name)

            device = resolve_device(trial_config)
            precision_name, amp_dtype = resolve_precision(trial_config, device)
            seed_everything(
                int(trial_config["seed"]),
                bool(trial_config["deterministic"]),
            )

            for fold_name, split_frames, patient_split in units:
                fold_dir = run_dir / "runs" / trial_name / fold_name
                fold_dir.mkdir(parents=True, exist_ok=True)

                loaders, _ = build_dataloaders(
                    split_frames,
                    trial_config,
                    device,
                    int(trial_config["seed"]),
                )
                model = HierarchicalCystoModel(trial_config).to(device)

                data_split_hash = science.split_fingerprint(split_frames)
                train_model(
                    model,
                    loaders,
                    split_frames,
                    split_frames["train"],
                    trial_config,
                    device,
                    amp_dtype,
                    fold_dir,
                    run_dir,
                    data_split_hash,
                    logger,
                )

                del model
                del loaders
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            completed_trials.append(trial_name)

        write_json(
            status_path,
            {
                "status": "completed",
                "started_utc": started_utc,
                "completed_utc": utc_now_iso(),
                "stage_name": stage_config["stage_name"],
                "run_dir": run_dir,
                "protocol_sha256": protocol_hash,
                "trials_completed": completed_trials,
            },
        )
        logger.info("Training suite completed successfully: %s", run_dir)
        return run_dir

    except Exception as exc:
        logger.exception("Training suite failed.")
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
        raise
    finally:
        close_logger(logger)
