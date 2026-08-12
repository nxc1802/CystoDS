"""Core classification metrics: per-class, binary, multiclass, composite bundle.

Extracted from ``cystods.core`` (Step 5 refactor).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

import cystods.science as science
from cystods.models.factory import active_tasks_from_config
from cystods.taxonomy import COARSE_NAMES, FINE_NAMES, FINE_PARENT_ID


def per_class_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    names: Sequence[str],
) -> tuple[list[dict[str, Any]], float | None, int]:
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=np.arange(len(names)),
        zero_division=0,
    )
    rows: list[dict[str, Any]] = []
    auc_values: list[float] = []
    for class_id, name in enumerate(names):
        binary_target = targets == class_id
        class_supported = int(support[class_id]) > 0
        auc: float | None = None
        if len(np.unique(binary_target)) == 2:
            auc = float(
                roc_auc_score(binary_target, probabilities[:, class_id])
            )
            auc_values.append(auc)
        rows.append(
            {
                "class_id": class_id,
                "class_name": name,
                "precision": (
                    float(precision[class_id]) if class_supported else None
                ),
                "recall": (
                    float(recall[class_id]) if class_supported else None
                ),
                "f1": float(f1[class_id]) if class_supported else None,
                "support": int(support[class_id]),
                "classification_metrics_evaluable": class_supported,
                "auroc_ovr": auc,
                "auroc_evaluable": auc is not None,
            }
        )
    macro_auc = float(np.mean(auc_values)) if auc_values else None
    return rows, macro_auc, len(auc_values)


def compute_binary_metrics(
    targets: np.ndarray,
    positive_probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    targets = np.asarray(targets, dtype=np.int64)
    positive_probabilities = np.asarray(
        positive_probabilities, dtype=np.float64
    )
    if (
        targets.ndim != 1
        or positive_probabilities.ndim != 1
        or len(targets) == 0
        or len(targets) != len(positive_probabilities)
    ):
        raise ValueError(
            "Binary targets and probabilities must be aligned non-empty vectors."
        )
    if (
        not np.isfinite(positive_probabilities).all()
        or np.any(positive_probabilities < 0)
        or np.any(positive_probabilities > 1)
        or np.any((targets != 0) & (targets != 1))
    ):
        raise ValueError("Binary targets/probabilities are outside their domains.")
    if not 0 < threshold < 1:
        raise ValueError("Binary threshold must be in (0, 1).")
    predictions = (positive_probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    supported_recalls = [
        value for value in (specificity, sensitivity) if value is not None
    ]
    auroc = (
        float(roc_auc_score(targets, positive_probabilities))
        if len(np.unique(targets)) == 2
        else None
    )
    auprc = (
        float(average_precision_score(targets, positive_probabilities))
        if len(np.unique(targets)) == 2
        else None
    )
    return {
        "n": len(targets),
        "decision_threshold": float(threshold),
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(
            precision_score(targets, predictions, zero_division=0)
        ),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "sensitivity": float(sensitivity) if sensitivity is not None else None,
        "specificity": float(specificity) if specificity is not None else None,
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(targets, predictions)),
        "balanced_accuracy": float(np.mean(supported_recalls)),
        "auroc": auroc,
        "auprc": auprc,
        "auroc_evaluable": auroc is not None,
        "auprc_evaluable": auprc is not None,
        "confusion_matrix": matrix.tolist(),
    }


def compute_multiclass_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    names: Sequence[str],
    probability_sum_atol: float = 1.0e-6,
) -> dict[str, Any]:
    base = science.compute_multiclass_metrics(
        targets,
        probabilities,
        class_names=names,
        probability_sum_atol=probability_sum_atol,
    )
    target_array, probability_array = science.validate_multiclass_inputs(
        targets,
        probabilities,
        probability_sum_atol=probability_sum_atol,
    )
    predicted = probability_array.argmax(axis=1)
    auc_rows, macro_auc, auc_count = per_class_metrics(
        target_array,
        predicted,
        probability_array,
        names,
    )
    for row, auc_row in zip(base["per_class"], auc_rows, strict=True):
        row["support"] = row["true_count"]
        row["classification_metrics_evaluable"] = row["target_supported"]
        row["auroc_ovr"] = auc_row["auroc_ovr"]
        row["auroc_evaluable"] = auc_row["auroc_evaluable"]
    supported_recalls = [
        row["recall"] for row in base["per_class"] if row["target_supported"]
    ]
    base.update(
        {
            # Explicit names are authoritative; aliases keep reports readable.
            "macro_f1": base["macro_f1_supported"],
            "macro_f1_evaluable_classes": base[
                "macro_f1_supported_class_count"
            ],
            "balanced_accuracy": float(np.mean(supported_recalls)),
            "mcc": float(matthews_corrcoef(target_array, predicted)),
            "macro_auroc_ovr": macro_auc,
            "macro_auroc_evaluable_classes": auc_count,
        }
    )
    return base


def compute_metrics_bundle(
    predictions: pd.DataFrame,
    train_fine_counts: Sequence[int],
    train_fine_patient_counts: Sequence[int],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if predictions.empty:
        raise ValueError("Cannot compute metrics from no predictions.")
    active_tasks = active_tasks_from_config(config)
    binary_targets = predictions["binary_id"].to_numpy(dtype=np.int64)
    coarse_targets = predictions["coarse_id"].to_numpy(dtype=np.int64)
    fine_targets = predictions["fine_id"].to_numpy(dtype=np.int64)
    tolerance = float(config["probability_sum_tolerance"])
    binary: dict[str, Any] | None = None
    coarse: dict[str, Any] | None = None
    fine: dict[str, Any] | None = None
    hierarchy: dict[str, Any] | None = None
    primary_fine: dict[str, Any] | None = None
    rare_audit: dict[str, Any] | None = None
    if "binary" in active_tasks:
        binary_probs = np.stack(predictions["binary_probs"].to_numpy())
        science.validate_multiclass_inputs(
            binary_targets,
            binary_probs,
            probability_sum_atol=tolerance,
        )
        binary = compute_binary_metrics(
            binary_targets,
            binary_probs[:, 1],
            float(config["binary_decision_threshold"]),
        )
    if "coarse" in active_tasks:
        coarse_probs = np.stack(predictions["coarse_probs"].to_numpy())
        coarse = compute_multiclass_metrics(
            coarse_targets,
            coarse_probs,
            COARSE_NAMES,
            tolerance,
        )
    valid_fine = fine_targets >= 0
    if "fine" in active_tasks and valid_fine.any():
        fine_probs = np.stack(predictions["fine_probs"].to_numpy())
        fine = compute_multiclass_metrics(
            fine_targets[valid_fine],
            fine_probs[valid_fine],
            FINE_NAMES,
            tolerance,
        )
        fine_pred = fine_probs[valid_fine].argmax(axis=1)
        train_counts = np.asarray(train_fine_counts, dtype=np.int64)
        patient_counts = np.asarray(
            train_fine_patient_counts, dtype=np.int64
        )
        if (
            train_counts.shape != (len(FINE_NAMES),)
            or patient_counts.shape != (len(FINE_NAMES),)
        ):
            raise ValueError(
                "Fine image/patient train counts must have shape (22,)."
            )
        tail_ids = np.flatnonzero(
            (train_counts > 0)
            & (
                train_counts
                <= int(config["tail_class_max_train_samples"])
            )
        )
        tail_mask = np.isin(fine_targets[valid_fine], tail_ids)
        evaluable_tail_ids = np.intersect1d(
            tail_ids, np.unique(fine_targets[valid_fine][tail_mask])
        )
        tail_recall = (
            float(
                recall_score(
                    fine_targets[valid_fine][tail_mask],
                    fine_pred[tail_mask],
                    labels=evaluable_tail_ids,
                    average="macro",
                    zero_division=0,
                )
            )
            if tail_mask.any() and len(evaluable_tail_ids)
            else None
        )
        if "coarse" in active_tasks:
            coarse_probs = np.stack(predictions["coarse_probs"].to_numpy())
            coarse_pred = coarse_probs[valid_fine].argmax(axis=1)
            predicted_parent = np.asarray(
                [FINE_PARENT_ID[index] for index in fine_pred],
                dtype=np.int64,
            )
            true_coarse = coarse_targets[valid_fine]
            hierarchy = {
                "parent_accuracy_from_coarse_head": float(
                    accuracy_score(true_coarse, coarse_pred)
                ),
                "parent_accuracy_from_fine_head": float(
                    accuracy_score(true_coarse, predicted_parent)
                ),
                "child_accuracy": float(
                    accuracy_score(fine_targets[valid_fine], fine_pred)
                ),
                "hierarchical_accuracy": float(
                    np.mean(
                        (coarse_pred == true_coarse)
                        & (fine_pred == fine_targets[valid_fine])
                    )
                ),
                "cross_parent_error_rate": float(
                    np.mean(predicted_parent != true_coarse)
                ),
                "coarse_fine_prediction_consistency": float(
                    np.mean(predicted_parent == coarse_pred)
                ),
                "tail_class_recall": tail_recall,
                "tail_class_names": [
                    FINE_NAMES[index] for index in tail_ids.tolist()
                ],
                "tail_class_names_evaluable_on_this_split": [
                    FINE_NAMES[index] for index in evaluable_tail_ids.tolist()
                ],
            }
        fixed_ids = config["fixed_primary_fine_class_ids"]
        if fixed_ids is None:
            primary_ids = np.flatnonzero(
                patient_counts
                >= int(config["primary_fine_min_train_patients"])
            )
            selection_source = "derived_from_training_patient_support"
        else:
            primary_ids = np.asarray(fixed_ids, dtype=np.int64)
            selection_source = "frozen_protocol"
        primary_mask = np.isin(fine_targets[valid_fine], primary_ids)
        if len(primary_ids) and primary_mask.any():
            primary_fine = science.compute_fixed_primary_metrics(
                fine_targets[valid_fine],
                fine_probs[valid_fine],
                primary_ids,
                class_names=FINE_NAMES,
                probability_sum_atol=tolerance,
            )
            primary_fine.update(
                {
                    "status": "ok",
                    "selection_source": selection_source,
                    "min_train_patient_support": int(
                        config["primary_fine_min_train_patients"]
                    ),
                    "macro_f1": primary_fine[
                        "macro_f1_supported"
                    ],
                }
            )
        else:
            primary_fine = {
                "status": "not_evaluable",
                "reason": (
                    "No evaluated row belongs to the frozen primary fine "
                    "class set."
                ),
                "selection_source": selection_source,
                "min_train_patient_support": int(
                    config["primary_fine_min_train_patients"]
                ),
                "primary_class_ids": primary_ids.tolist(),
                "primary_class_names": [
                    FINE_NAMES[index] for index in primary_ids.tolist()
                ],
            }
        rare_audit = science.audit_rare_class_collapse(
            fine_pred,
            train_counts,
            patient_counts,
            class_names=FINE_NAMES,
            max_train_patients=int(config["rare_gate_max_train_patients"]),
            absolute_prediction_share=float(
                config["rare_gate_absolute_pred_share"]
            ),
            prior_multiplier=float(config["rare_gate_prior_multiplier"]),
            min_predicted_count=int(config["rare_gate_min_pred_count"]),
            prior_smoothing_alpha=float(
                config["fine_prior_smoothing_alpha"]
            ),
            prior_power=float(config["fine_prior_power"]),
        )
    return {
        "binary": binary,
        "coarse": coarse,
        "fine": fine,
        "primary_fine": primary_fine,
        "hierarchy": hierarchy,
        "rare_class_collapse": rare_audit,
    }


def metric_for_monitor(
    bundle: Mapping[str, Any],
    monitor: str,
    monitor_weights: Mapping[str, float] | None = None,
) -> float:
    mapping = {
        "binary_auroc": (
            bundle["binary"]["auroc"] if bundle["binary"] is not None else None
        ),
        "binary_f1": (
            bundle["binary"]["f1"] if bundle["binary"] is not None else None
        ),
        "coarse_macro_f1": (
            bundle["coarse"]["macro_f1_supported"]
            if bundle["coarse"] is not None
            else None
        ),
        "coarse_macro_f1_supported": (
            bundle["coarse"]["macro_f1_supported"]
            if bundle["coarse"] is not None
            else None
        ),
        "coarse_macro_f1_all_classes": (
            bundle["coarse"]["macro_f1_all_classes"]
            if bundle["coarse"] is not None
            else None
        ),
        "coarse_balanced_accuracy": (
            bundle["coarse"]["balanced_accuracy"]
            if bundle["coarse"] is not None
            else None
        ),
        "fine_macro_f1": (
            bundle["fine"]["macro_f1_supported"]
            if bundle["fine"] is not None
            else None
        ),
        "fine_macro_f1_supported": (
            bundle["fine"]["macro_f1_supported"]
            if bundle["fine"] is not None
            else None
        ),
        "fine_macro_f1_all_classes": (
            bundle["fine"]["macro_f1_all_classes"]
            if bundle["fine"] is not None
            else None
        ),
        "primary_macro_f1": (
            bundle["primary_fine"]["macro_f1_supported"]
            if bundle["primary_fine"] is not None
            and bundle["primary_fine"]["status"] == "ok"
            else None
        ),
        "primary_macro_f1_supported": (
            bundle["primary_fine"]["macro_f1_supported"]
            if bundle["primary_fine"] is not None
            and bundle["primary_fine"]["status"] == "ok"
            else None
        ),
        "primary_macro_f1_all_classes": (
            bundle["primary_fine"]["macro_f1_all_classes"]
            if bundle["primary_fine"] is not None
            and bundle["primary_fine"]["status"] == "ok"
            else None
        ),
        "tail_class_recall": (
            bundle["hierarchy"]["tail_class_recall"]
            if bundle.get("hierarchy") is not None
            else None
        ),
        "hierarchical_accuracy": (
            bundle["hierarchy"]["hierarchical_accuracy"]
            if bundle.get("hierarchy") is not None
            else None
        ),
    }
    if monitor == "hierarchical_composite":
        required = {
            "coarse_macro_f1_all_classes": mapping[
                "coarse_macro_f1_all_classes"
            ],
            "primary_macro_f1_all_classes": mapping[
                "primary_macro_f1_all_classes"
            ],
            "hierarchical_accuracy": mapping["hierarchical_accuracy"],
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "hierarchical_composite is not evaluable; missing "
                f"components={missing}."
            )
        return science.hierarchical_composite_monitor(
            {name: float(value) for name, value in required.items()},
            weights=monitor_weights,
        )
    if monitor not in mapping:
        raise ValueError(
            f"Unknown monitor_metric={monitor}; choices={sorted(mapping)}"
        )
    value = mapping[monitor]
    if value is None or not math.isfinite(float(value)):
        raise ValueError(
            f"monitor_metric={monitor} is not evaluable on validation data."
        )
    return float(value)
