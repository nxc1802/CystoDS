"""Evaluation package for CystoDS."""

from cystods.evaluation.metrics import (
    compute_binary_metrics,
    compute_metrics_bundle,
    compute_multiclass_metrics,
    metric_for_monitor,
    per_class_metrics,
)
from cystods.evaluation.calibration import (
    apply_fine_inference_calibration,
    select_fine_inference_tau,
)
from cystods.evaluation.bootstrap import (
    paired_mcnemar_test,
    patient_bootstrap_intervals,
)
from cystods.evaluation.roi import (
    RoiBag,
    aggregate_roi_bags,
    compute_roi_task_metrics,
    extract_roi_bags,
    predict_attention_bags,
    run_roi_evaluation,
    serialize_roi_predictions,
    train_attention_mil,
)

__all__ = [
    "per_class_metrics",
    "compute_binary_metrics",
    "compute_multiclass_metrics",
    "compute_metrics_bundle",
    "metric_for_monitor",
    "apply_fine_inference_calibration",
    "select_fine_inference_tau",
    "patient_bootstrap_intervals",
    "paired_mcnemar_test",
    "RoiBag",
    "extract_roi_bags",
    "aggregate_roi_bags",
    "compute_roi_task_metrics",
    "predict_attention_bags",
    "train_attention_mil",
    "serialize_roi_predictions",
    "run_roi_evaluation",
]
