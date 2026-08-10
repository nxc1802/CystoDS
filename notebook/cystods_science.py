"""Strict scientific metrics and protocol helpers for CystoDS experiments.

The functions in this module are intentionally independent from the training
implementation.  They never repair malformed inputs, replace a requested
method, or manufacture a metric when it is not evaluable.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

DEFAULT_HIERARCHICAL_COMPOSITE_WEIGHTS = {
    "coarse_macro_f1_all_classes": 0.35,
    "primary_macro_f1_all_classes": 0.45,
    "hierarchical_accuracy": 0.20,
}


class ScientificGateError(RuntimeError):
    """Raised when a completed experiment fails a scientific quality gate."""


def _strict_integer_vector(
    values: Sequence[int] | np.ndarray,
    *,
    name: str,
    allow_empty: bool = False,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not allow_empty and raw.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if (
        np.issubdtype(raw.dtype, np.bool_)
        or not np.issubdtype(raw.dtype, np.number)
        or np.issubdtype(raw.dtype, np.complexfloating)
    ):
        raise ValueError(f"{name} must contain real numeric integers.")
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain integers.") from exc
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} contains NaN or infinity.")
    if not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{name} must contain integer-valued entries.")
    return numeric.astype(np.int64)


def _strict_nonnegative_counts(
    values: Sequence[int] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    counts = _strict_integer_vector(values, name=name)
    if np.any(counts < 0):
        raise ValueError(f"{name} must contain non-negative counts.")
    return counts


def _validated_class_names(
    class_names: Sequence[str] | None,
    num_classes: int,
) -> tuple[str, ...]:
    if class_names is None:
        return tuple(str(index) for index in range(num_classes))
    names = tuple(class_names)
    if len(names) != num_classes:
        raise ValueError("class_names length must equal the probability column count.")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Every class name must be a non-empty string.")
    if len(set(names)) != len(names):
        raise ValueError("class_names must be unique.")
    return names


def validate_multiclass_inputs(
    targets: Sequence[int] | np.ndarray,
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    probability_sum_atol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return multiclass targets and probabilities.

    Probability rows must already be normalized.  Values are not clipped or
    renormalized because either operation could conceal a broken inference
    pipeline.
    """

    if not math.isfinite(float(probability_sum_atol)) or probability_sum_atol < 0:
        raise ValueError("probability_sum_atol must be finite and non-negative.")
    target_array = _strict_integer_vector(targets, name="targets")
    raw_probabilities = np.asarray(probabilities)
    if (
        np.issubdtype(raw_probabilities.dtype, np.bool_)
        or not np.issubdtype(raw_probabilities.dtype, np.number)
        or np.issubdtype(raw_probabilities.dtype, np.complexfloating)
    ):
        raise ValueError("probabilities must contain real numeric values.")
    try:
        probability_array = raw_probabilities.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("probabilities must be a numeric matrix.") from exc
    if probability_array.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional matrix.")
    if probability_array.shape[0] != target_array.size:
        raise ValueError("targets and probabilities must have the same rows.")
    if probability_array.shape[1] < 2:
        raise ValueError("probabilities must contain at least two classes.")
    if not np.isfinite(probability_array).all():
        raise ValueError("probabilities contain NaN or infinity.")
    if np.any(probability_array < 0) or np.any(probability_array > 1):
        raise ValueError("probabilities must lie within [0, 1].")
    row_sums = probability_array.sum(axis=1)
    if not np.allclose(
        row_sums,
        np.ones_like(row_sums),
        rtol=0,
        atol=float(probability_sum_atol),
    ):
        largest_error = float(np.max(np.abs(row_sums - 1.0)))
        raise ValueError(
            "probability rows must sum to one; "
            f"largest absolute error={largest_error:.3e}."
        )
    num_classes = probability_array.shape[1]
    if np.any(target_array < 0) or np.any(target_array >= num_classes):
        raise ValueError(f"targets must be within [0, {num_classes - 1}].")
    return target_array, probability_array


def compute_multiclass_metrics(
    targets: Sequence[int] | np.ndarray,
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    class_names: Sequence[str] | None = None,
    probability_sum_atol: float = 1e-6,
) -> dict[str, Any]:
    """Compute explicit supported-class and fixed-denominator metrics."""

    target_array, probability_array = validate_multiclass_inputs(
        targets,
        probabilities,
        probability_sum_atol=probability_sum_atol,
    )
    num_classes = probability_array.shape[1]
    names = _validated_class_names(class_names, num_classes)
    class_ids = np.arange(num_classes, dtype=np.int64)
    predictions = probability_array.argmax(axis=1)
    precision, recall, per_class_f1, true_count = precision_recall_fscore_support(
        target_array,
        predictions,
        labels=class_ids,
        zero_division=0,
    )
    true_count = true_count.astype(np.int64)
    predicted_count = np.bincount(
        predictions,
        minlength=num_classes,
    ).astype(np.int64)
    supported_ids = np.flatnonzero(true_count > 0)
    if supported_ids.size == 0:
        raise ValueError("No target class is supported.")

    total = target_array.size
    per_class = []
    for class_id, class_name in enumerate(names):
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(per_class_f1[class_id]),
                "true_count": int(true_count[class_id]),
                "predicted_count": int(predicted_count[class_id]),
                "true_share": float(true_count[class_id] / total),
                "predicted_share": float(predicted_count[class_id] / total),
                "target_supported": bool(true_count[class_id] > 0),
                "prediction_supported": bool(predicted_count[class_id] > 0),
            }
        )

    return {
        "n": int(total),
        "num_classes": int(num_classes),
        "accuracy": float(accuracy_score(target_array, predictions)),
        "macro_f1_supported": float(np.mean(per_class_f1[supported_ids])),
        "macro_f1_all_classes": float(np.mean(per_class_f1)),
        "macro_f1_supported_class_count": int(supported_ids.size),
        "weighted_f1": float(
            f1_score(
                target_array,
                predictions,
                labels=class_ids,
                average="weighted",
                zero_division=0,
            )
        ),
        "true_counts": true_count.tolist(),
        "predicted_counts": predicted_count.tolist(),
        "true_shares": (true_count / total).astype(float).tolist(),
        "predicted_shares": (predicted_count / total).astype(float).tolist(),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            target_array,
            predictions,
            labels=class_ids,
        ).tolist(),
    }


def _validated_fixed_class_ids(
    class_ids: Sequence[int] | np.ndarray,
    num_classes: int,
) -> np.ndarray:
    ids = _strict_integer_vector(class_ids, name="primary_class_ids")
    if np.any(ids < 0) or np.any(ids >= num_classes):
        raise ValueError(f"primary_class_ids must be within [0, {num_classes - 1}].")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("primary_class_ids must not contain duplicates.")
    if len(ids) > 1 and np.any(np.diff(ids) <= 0):
        raise ValueError("primary_class_ids must be in strictly increasing order.")
    return ids


def compute_fixed_primary_metrics(
    targets: Sequence[int] | np.ndarray,
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    primary_class_ids: Sequence[int] | np.ndarray,
    *,
    class_names: Sequence[str] | None = None,
    probability_sum_atol: float = 1e-6,
) -> dict[str, Any]:
    """Evaluate a fixed primary taxonomy without changing the output space.

    Rows whose true label is outside the fixed primary set are excluded.
    Predictions remain the argmax over the complete output taxonomy; escaping
    to a non-primary class therefore counts as an error.
    """

    target_array, probability_array = validate_multiclass_inputs(
        targets,
        probabilities,
        probability_sum_atol=probability_sum_atol,
    )
    num_classes = probability_array.shape[1]
    names = _validated_class_names(class_names, num_classes)
    primary_ids = _validated_fixed_class_ids(
        primary_class_ids,
        num_classes,
    )
    primary_mask = np.isin(target_array, primary_ids)
    if not primary_mask.any():
        raise ValueError("No target row belongs to the fixed primary class set.")
    primary_targets = target_array[primary_mask]
    primary_predictions = probability_array[primary_mask].argmax(axis=1)
    precision, recall, per_class_f1, true_count = precision_recall_fscore_support(
        primary_targets,
        primary_predictions,
        labels=primary_ids,
        zero_division=0,
    )
    supported = true_count > 0
    predicted_count = np.asarray(
        [np.count_nonzero(primary_predictions == class_id) for class_id in primary_ids],
        dtype=np.int64,
    )
    outside_count = int(np.count_nonzero(~np.isin(primary_predictions, primary_ids)))
    total = primary_targets.size
    rows = []
    for offset, class_id in enumerate(primary_ids):
        rows.append(
            {
                "class_id": int(class_id),
                "class_name": names[int(class_id)],
                "precision": float(precision[offset]),
                "recall": float(recall[offset]),
                "f1": float(per_class_f1[offset]),
                "true_count": int(true_count[offset]),
                "predicted_count": int(predicted_count[offset]),
                "true_share": float(true_count[offset] / total),
                "predicted_share": float(predicted_count[offset] / total),
                "target_supported": bool(supported[offset]),
                "prediction_supported": bool(predicted_count[offset] > 0),
            }
        )

    return {
        "n": int(total),
        "primary_class_ids": primary_ids.tolist(),
        "primary_class_names": [names[int(index)] for index in primary_ids],
        "accuracy": float(accuracy_score(primary_targets, primary_predictions)),
        "macro_f1_supported": float(np.mean(per_class_f1[supported])),
        "macro_f1_all_classes": float(np.mean(per_class_f1)),
        "macro_f1_supported_class_count": int(np.count_nonzero(supported)),
        "predictions_outside_primary_count": outside_count,
        "predictions_outside_primary_share": float(outside_count / total),
        "per_class": rows,
    }


def build_smoothed_class_prior(
    class_counts: Sequence[int] | np.ndarray,
    *,
    smoothing_alpha: float,
    power: float,
    max_ratio: float | None = None,
) -> dict[str, Any]:
    """Build a finite prior and an explicit mask for observed train classes."""

    counts = _strict_nonnegative_counts(class_counts, name="class_counts")
    active_mask = counts > 0
    if not active_mask.any():
        raise ValueError("At least one class count must be positive.")
    if not math.isfinite(float(smoothing_alpha)) or smoothing_alpha < 0:
        raise ValueError("smoothing_alpha must be finite and non-negative.")
    if not math.isfinite(float(power)) or power <= 0:
        raise ValueError("power must be finite and positive.")
    if max_ratio is not None and (not math.isfinite(float(max_ratio)) or max_ratio < 1):
        raise ValueError("max_ratio must be finite and at least one.")

    effective_counts = np.zeros_like(counts, dtype=np.float64)
    effective_counts[active_mask] = np.power(
        counts[active_mask].astype(np.float64) + float(smoothing_alpha),
        float(power),
    )
    if not np.isfinite(effective_counts).all():
        raise ValueError("Smoothed class counts are not finite.")
    if max_ratio is not None:
        active_maximum = float(effective_counts[active_mask].max())
        minimum_allowed = active_maximum / float(max_ratio)
        effective_counts[active_mask] = np.maximum(
            effective_counts[active_mask],
            minimum_allowed,
        )
    active_total = float(effective_counts[active_mask].sum())
    if not math.isfinite(active_total) or active_total <= 0:
        raise ValueError("Smoothed class-count total must be finite and positive.")
    probabilities = np.zeros_like(effective_counts)
    probabilities[active_mask] = effective_counts[active_mask] / active_total
    log_probabilities = np.zeros_like(effective_counts)
    log_probabilities[active_mask] = np.log(probabilities[active_mask])
    if not (np.isfinite(probabilities).all() and np.isfinite(log_probabilities).all()):
        raise ValueError("Class-prior outputs are not finite.")
    return {
        "counts": counts.tolist(),
        "active_mask": active_mask.tolist(),
        "effective_counts": effective_counts.tolist(),
        "probabilities": probabilities.tolist(),
        "log_probabilities": log_probabilities.tolist(),
        "smoothing_alpha": float(smoothing_alpha),
        "power": float(power),
        "max_ratio": None if max_ratio is None else float(max_ratio),
    }


def mask_inactive_logits(
    logits: Sequence[Sequence[float]] | np.ndarray,
    active_mask: Sequence[bool] | np.ndarray,
    *,
    inactive_value: float = -1.0e30,
) -> np.ndarray:
    """Return finite logits with unobserved training classes masked."""

    try:
        logit_array = np.asarray(logits, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("logits must be a numeric matrix.") from exc
    if logit_array.ndim != 2 or logit_array.shape[0] == 0:
        raise ValueError("logits must be a non-empty two-dimensional matrix.")
    if not np.isfinite(logit_array).all():
        raise ValueError("logits contain NaN or infinity.")
    mask = np.asarray(active_mask)
    if mask.ndim != 1 or mask.size != logit_array.shape[1]:
        raise ValueError("active_mask must match the number of logit columns.")
    if mask.dtype != np.bool_:
        raise ValueError("active_mask must contain booleans.")
    if not mask.any():
        raise ValueError("active_mask must enable at least one class.")
    if not math.isfinite(float(inactive_value)):
        raise ValueError("inactive_value must be finite.")
    output = logit_array.copy()
    output[:, ~mask] = float(inactive_value)
    return output


def validate_active_class_targets(
    targets: Sequence[int] | np.ndarray,
    active_mask: Sequence[bool] | np.ndarray,
) -> np.ndarray:
    """Require every training target to belong to an active train class."""

    target_array = _strict_integer_vector(targets, name="targets")
    mask = np.asarray(active_mask)
    if mask.ndim != 1 or mask.dtype != np.bool_ or not mask.any():
        raise ValueError(
            "active_mask must be a non-empty one-dimensional boolean vector."
        )
    if np.any(target_array < 0) or np.any(target_array >= mask.size):
        raise ValueError("targets contain a class outside active_mask.")
    inactive_targets = np.unique(target_array[~mask[target_array]])
    if inactive_targets.size:
        raise ValueError(
            f"Training targets reference inactive classes: {inactive_targets.tolist()}."
        )
    return target_array


def hierarchical_composite_monitor(
    components: Mapping[str, float],
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    """Compute a preregistered hierarchical validation monitor."""

    resolved_weights = (
        DEFAULT_HIERARCHICAL_COMPOSITE_WEIGHTS if weights is None else weights
    )
    required = set(DEFAULT_HIERARCHICAL_COMPOSITE_WEIGHTS)
    if set(components) != required:
        raise ValueError(
            "components must contain exactly "
            f"{sorted(required)}; received={sorted(components)}."
        )
    if set(resolved_weights) != required:
        raise ValueError(
            "weights must contain exactly "
            f"{sorted(required)}; received={sorted(resolved_weights)}."
        )
    component_values = {}
    weight_values = {}
    for name in sorted(required):
        component = float(components[name])
        weight = float(resolved_weights[name])
        if not math.isfinite(component) or not 0 <= component <= 1:
            raise ValueError(
                f"Composite component {name!r} must be finite and in [0, 1]."
            )
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(
                f"Composite weight {name!r} must be finite and non-negative."
            )
        component_values[name] = component
        weight_values[name] = weight
    weight_sum = math.fsum(weight_values.values())
    if not math.isclose(weight_sum, 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError(
            f"Hierarchical composite weights must sum to one; got {weight_sum}."
        )
    result = math.fsum(
        component_values[name] * weight_values[name] for name in sorted(required)
    )
    if not math.isfinite(result):
        raise ValueError("Hierarchical composite monitor is not finite.")
    return float(result)


def audit_rare_class_collapse(
    predicted_class_ids: Sequence[int] | np.ndarray,
    train_prior_counts: Sequence[int] | np.ndarray,
    train_patient_counts: Sequence[int] | np.ndarray,
    *,
    class_names: Sequence[str] | None = None,
    max_train_patients: int,
    absolute_prediction_share: float,
    prior_multiplier: float,
    min_predicted_count: int,
    prior_smoothing_alpha: float = 1.0,
    prior_power: float = 1.0,
) -> dict[str, Any]:
    """Audit whether a very-low-support class dominates predictions."""

    predictions = _strict_integer_vector(
        predicted_class_ids,
        name="predicted_class_ids",
    )
    prior_counts = _strict_nonnegative_counts(
        train_prior_counts,
        name="train_prior_counts",
    )
    patient_counts = _strict_nonnegative_counts(
        train_patient_counts,
        name="train_patient_counts",
    )
    if prior_counts.shape != patient_counts.shape:
        raise ValueError("train_prior_counts and train_patient_counts must align.")
    num_classes = prior_counts.size
    names = _validated_class_names(class_names, num_classes)
    if np.any(predictions < 0) or np.any(predictions >= num_classes):
        raise ValueError(f"predicted_class_ids must be within [0, {num_classes - 1}].")
    if int(max_train_patients) != max_train_patients or max_train_patients < 0:
        raise ValueError("max_train_patients must be a non-negative integer.")
    if (
        not math.isfinite(float(absolute_prediction_share))
        or not 0 <= absolute_prediction_share <= 1
    ):
        raise ValueError("absolute_prediction_share must be finite and in [0, 1].")
    if not math.isfinite(float(prior_multiplier)) or prior_multiplier < 0:
        raise ValueError("prior_multiplier must be finite and non-negative.")
    if int(min_predicted_count) != min_predicted_count or min_predicted_count < 1:
        raise ValueError("min_predicted_count must be a positive integer.")

    prior = build_smoothed_class_prior(
        prior_counts,
        smoothing_alpha=prior_smoothing_alpha,
        power=prior_power,
    )
    prior_probabilities = np.asarray(
        prior["probabilities"],
        dtype=np.float64,
    )
    predicted_counts = np.bincount(
        predictions,
        minlength=num_classes,
    ).astype(np.int64)
    predicted_shares = predicted_counts / predictions.size
    rare_ids = np.flatnonzero(patient_counts <= int(max_train_patients))
    rows = []
    violations = []
    for class_id in rare_ids:
        allowed_share = max(
            float(absolute_prediction_share),
            float(prior_multiplier) * float(prior_probabilities[class_id]),
        )
        violated = bool(
            predicted_counts[class_id] >= int(min_predicted_count)
            and predicted_shares[class_id] > allowed_share
        )
        row = {
            "class_id": int(class_id),
            "class_name": names[class_id],
            "train_prior_count": int(prior_counts[class_id]),
            "train_patient_count": int(patient_counts[class_id]),
            "smoothed_train_prior": float(prior_probabilities[class_id]),
            "predicted_count": int(predicted_counts[class_id]),
            "predicted_share": float(predicted_shares[class_id]),
            "allowed_prediction_share": float(allowed_share),
            "violated": violated,
        }
        rows.append(row)
        if violated:
            violations.append(row)
    return {
        "status": "failed" if violations else "passed",
        "n_predictions": int(predictions.size),
        "rare_class_ids": rare_ids.astype(int).tolist(),
        "parameters": {
            "max_train_patients": int(max_train_patients),
            "absolute_prediction_share": float(absolute_prediction_share),
            "prior_multiplier": float(prior_multiplier),
            "min_predicted_count": int(min_predicted_count),
            "prior_smoothing_alpha": float(prior_smoothing_alpha),
            "prior_power": float(prior_power),
        },
        "classes": rows,
        "violations": violations,
    }


def enforce_rare_class_collapse_gate(audit: Mapping[str, Any]) -> None:
    """Raise when a rare-class audit reports a collapse."""

    required = {
        "status",
        "n_predictions",
        "rare_class_ids",
        "parameters",
        "classes",
        "violations",
    }
    if set(audit) != required:
        raise ValueError(
            f"Rare-class audit schema mismatch; expected={sorted(required)}."
        )
    status = audit["status"]
    if status not in {"passed", "failed"}:
        raise ValueError("Rare-class audit status must be passed or failed.")
    violations = audit["violations"]
    if not isinstance(violations, list):
        raise TypeError("Rare-class audit violations must be a list.")
    if status == "passed" and violations:
        raise ValueError("Passed rare-class audit cannot contain violations.")
    if status == "failed" and not violations:
        raise ValueError("Failed rare-class audit must contain violations.")
    if status == "failed":
        labels = [
            f"{row['class_name']}({row['predicted_share']:.3f})" for row in violations
        ]
        raise ScientificGateError(
            "Rare-class prediction collapse detected: " + ", ".join(labels)
        )


def _canonicalize_semantic_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Semantic value contains NaN or infinity.")
        return number
    if isinstance(value, np.ndarray):
        return _canonicalize_semantic_value(value.tolist())
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Semantic mapping keys must be strings.")
            normalized[key] = _canonicalize_semantic_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonicalize_semantic_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized_items = [_canonicalize_semantic_value(item) for item in value]
        return sorted(
            normalized_items,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )
    raise TypeError(
        "Unsupported semantic value type: "
        f"{type(value).__module__}.{type(value).__qualname__}."
    )


def semantic_fingerprint(value: Any) -> str:
    """Return a stable SHA-256 digest for supported semantic values."""

    canonical = _canonicalize_semantic_value(value)
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_split_identity(value: Any, *, split_name: str) -> str:
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(
            f"Split {split_name!r} contains NaN or infinity in identity data."
        )
    if isinstance(value, (complex, np.complexfloating)):
        raise TypeError(f"Split {split_name!r} contains complex identity data.")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"Split {split_name!r} contains an empty identity value.")
    return normalized


def split_fingerprint(
    split_frames: Mapping[str, pd.DataFrame],
    *,
    identity_columns: Sequence[str] = ("pid", "filename"),
    patient_column: str = "pid",
    enforce_patient_disjoint: bool = True,
) -> str:
    """Fingerprint a split assignment independently of DataFrame row order."""

    if not isinstance(split_frames, Mapping) or not split_frames:
        raise ValueError("split_frames must be a non-empty mapping.")
    if any(not isinstance(name, str) or not name for name in split_frames):
        raise ValueError("Every split name must be a non-empty string.")
    columns = tuple(identity_columns)
    if not columns or any(
        not isinstance(column, str) or not column for column in columns
    ):
        raise ValueError("identity_columns must contain non-empty column names.")
    if len(set(columns)) != len(columns):
        raise ValueError("identity_columns must be unique.")
    if not isinstance(patient_column, str) or not patient_column:
        raise ValueError("patient_column must be a non-empty string.")

    canonical_splits: dict[str, Any] = {}
    global_identities: dict[tuple[str, ...], str] = {}
    patient_membership: dict[str, str] = {}
    for split_name in sorted(split_frames):
        frame = split_frames[split_name]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"Split {split_name!r} must be a DataFrame.")
        if frame.empty:
            raise ValueError(f"Split {split_name!r} must not be empty.")
        required = set(columns) | {patient_column}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"Split {split_name!r} is missing columns {sorted(missing)}."
            )
        selected = frame.loc[:, list(dict.fromkeys((*columns, patient_column)))]
        if selected.isna().any(axis=None):
            raise ValueError(f"Split {split_name!r} contains missing identity data.")
        normalized_rows = []
        local_identities: set[tuple[str, ...]] = set()
        for row in selected.itertuples(index=False, name=None):
            normalized = tuple(
                _normalize_split_identity(value, split_name=split_name) for value in row
            )
            normalized_map = dict(zip(selected.columns, normalized))
            identity = tuple(normalized_map[column] for column in columns)
            if identity in local_identities:
                raise ValueError(
                    f"Split {split_name!r} contains duplicate identity {identity}."
                )
            local_identities.add(identity)
            previous_split = global_identities.get(identity)
            if previous_split is not None:
                raise ValueError(
                    f"Identity {identity} appears in {previous_split!r} "
                    f"and {split_name!r}."
                )
            global_identities[identity] = split_name
            patient = normalized_map[patient_column]
            if enforce_patient_disjoint:
                previous_patient_split = patient_membership.get(patient)
                if (
                    previous_patient_split is not None
                    and previous_patient_split != split_name
                ):
                    raise ValueError(
                        f"Patient {patient!r} appears in "
                        f"{previous_patient_split!r} and {split_name!r}."
                    )
                patient_membership[patient] = split_name
            normalized_rows.append(dict(normalized_map))
        canonical_splits[split_name] = sorted(
            normalized_rows,
            key=lambda item: tuple(item[column] for column in selected.columns),
        )
    return semantic_fingerprint(
        {
            "identity_columns": list(columns),
            "patient_column": patient_column,
            "patient_disjoint": bool(enforce_patient_disjoint),
            "splits": canonical_splits,
        }
    )


def binary_coarse_taxonomy_metrics(
    binary_targets: Sequence[int] | np.ndarray,
    coarse_probabilities: Sequence[Sequence[float]] | np.ndarray,
) -> dict[str, float]:
    targets = _strict_integer_vector(binary_targets, name="binary_targets")
    probs = np.asarray(coarse_probabilities, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != 5:
        raise ValueError("coarse_probabilities must have shape [N, 5].")
    coarse_preds = probs.argmax(axis=1)
    coarse_binary_parent = np.isin(coarse_preds, (0, 1)).astype(np.int64)
    violations = coarse_binary_parent != targets

    coarse_to_binary = np.array(
        [
            [0.0, 1.0],  # 0: Malignant -> ROI
            [0.0, 1.0],  # 1: Non-malignant -> ROI
            [1.0, 0.0],  # 2: Normal -> Non-ROI
            [1.0, 0.0],  # 3: Landmark -> Non-ROI
            [1.0, 0.0],  # 4: Foreign -> Non-ROI
        ],
        dtype=np.float64,
    )
    binary_from_coarse = probs @ coarse_to_binary
    correct_mass = binary_from_coarse[np.arange(len(targets)), targets]

    return {
        "taxonomy_violation_rate": float(violations.mean()),
        "taxonomy_accuracy": float(1.0 - violations.mean()),
        "mean_correct_parent_mass": float(correct_mass.mean()),
    }


def coarse_fine_taxonomy_metrics(
    coarse_targets: Sequence[int] | np.ndarray,
    fine_targets: Sequence[int] | np.ndarray,
    fine_probabilities: Sequence[Sequence[float]] | np.ndarray,
    fine_parent_ids: Sequence[int],
) -> dict[str, Any]:
    c_targets = _strict_integer_vector(coarse_targets, name="coarse_targets")
    f_targets = _strict_integer_vector(
        fine_targets, name="fine_targets", allow_empty=True
    )
    probs = np.asarray(fine_probabilities, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != len(fine_parent_ids):
        raise ValueError(
            "fine_probabilities shape must match fine_parent_ids length."
        )

    valid = f_targets >= 0
    if not np.any(valid):
        return {
            "status": "not_evaluable",
            "n": 0,
        }

    valid_c_targets = c_targets[valid]
    valid_probs = probs[valid]
    fine_preds = valid_probs.argmax(axis=1)

    parent_ids = np.asarray(fine_parent_ids, dtype=np.int64)
    predicted_parent = parent_ids[fine_preds]
    violations = predicted_parent != valid_c_targets

    fine_to_coarse = np.zeros((len(fine_parent_ids), 5), dtype=np.float64)
    for fine_id, parent_id in enumerate(parent_ids):
        fine_to_coarse[fine_id, parent_id] = 1.0

    coarse_from_fine = valid_probs @ fine_to_coarse
    correct_mass = coarse_from_fine[
        np.arange(len(valid_c_targets)), valid_c_targets
    ]

    return {
        "status": "ok",
        "n": int(valid.sum()),
        "taxonomy_violation_rate": float(violations.mean()),
        "taxonomy_accuracy": float(1.0 - violations.mean()),
        "mean_correct_parent_mass": float(correct_mass.mean()),
    }


__all__ = [
    "DEFAULT_HIERARCHICAL_COMPOSITE_WEIGHTS",
    "ScientificGateError",
    "audit_rare_class_collapse",
    "binary_coarse_taxonomy_metrics",
    "build_smoothed_class_prior",
    "coarse_fine_taxonomy_metrics",
    "compute_fixed_primary_metrics",
    "compute_multiclass_metrics",
    "enforce_rare_class_collapse_gate",
    "hierarchical_composite_monitor",
    "mask_inactive_logits",
    "semantic_fingerprint",
    "split_fingerprint",
    "validate_active_class_targets",
    "validate_multiclass_inputs",
]
