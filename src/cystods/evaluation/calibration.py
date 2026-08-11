"""Fine-grained prior logit calibration during inference.

Extracted from ``cystods.core`` (Step 5 refactor).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from cystods.evaluation.metrics import compute_metrics_bundle
from cystods.losses.classification import FineLongTailLoss


def apply_fine_inference_calibration(
    predictions: pd.DataFrame,
    fine_loss_fn: FineLongTailLoss,
    prior_tau: float,
) -> pd.DataFrame:
    if "fine_logits" not in predictions:
        raise ValueError("Fine calibration requires fine_logits predictions.")
    logits = np.stack(predictions["fine_logits"].to_numpy()).astype(
        np.float64,
        copy=False,
    )
    if not np.isfinite(logits).all():
        raise FloatingPointError("Stored fine logits contain NaN or infinity.")
    active_mask = (
        fine_loss_fn.active_mask.detach().cpu().numpy().astype(bool)
    )
    log_prior = (
        fine_loss_fn.smoothed_log_prior.detach().cpu().numpy().astype(
            np.float64
        )
    )
    adjusted = logits - float(prior_tau) * log_prior[None, :]
    adjusted[:, ~active_mask] = -1.0e30
    adjusted -= adjusted.max(axis=1, keepdims=True)
    exponentials = np.exp(adjusted)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    if (
        not np.isfinite(probabilities).all()
        or not np.allclose(
            probabilities.sum(axis=1),
            1.0,
            rtol=0,
            atol=1.0e-12,
        )
    ):
        raise FloatingPointError(
            "Fine inference calibration produced invalid probabilities."
        )
    calibrated = predictions.copy()
    calibrated["fine_probs"] = list(probabilities)
    calibrated["fine_inference_prior_tau"] = float(prior_tau)
    return calibrated


def select_fine_inference_tau(
    predictions: pd.DataFrame,
    fine_loss_fn: FineLongTailLoss,
    train_fine_counts: Sequence[int],
    train_fine_patient_counts: Sequence[int],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, float, dict[str, Any]]:
    if config["fine_inference_calibration_mode"] == "fixed":
        tau_values = [float(config["fine_inference_prior_tau"])]
    else:
        tau_values = sorted(
            {float(value) for value in config["fine_inference_tau_grid"]}
        )
    metric_name = str(config["fine_inference_calibration_metric"])
    candidates = []
    best: tuple[float, float, dict[str, Any], pd.DataFrame] | None = None
    for tau in tau_values:
        calibrated = apply_fine_inference_calibration(
            predictions,
            fine_loss_fn,
            tau,
        )
        metrics = compute_metrics_bundle(
            calibrated,
            train_fine_counts,
            train_fine_patient_counts,
            config,
        )
        if metric_name == "fine_macro_f1_all_classes":
            value = (
                metrics["fine"]["macro_f1_all_classes"]
                if metrics["fine"] is not None
                else None
            )
        elif metric_name == "primary_macro_f1_all_classes":
            value = (
                metrics["primary_fine"]["macro_f1_all_classes"]
                if metrics["primary_fine"] is not None
                and metrics["primary_fine"]["status"] == "ok"
                else None
            )
        else:
            raise ValueError(
                f"Unsupported fine calibration metric: {metric_name}"
            )
        if value is None or not math.isfinite(float(value)):
            raise ValueError(
                f"Fine calibration metric {metric_name} is not evaluable "
                f"for tau={tau}."
            )
        score = float(value)
        candidates.append({"prior_tau": tau, "metric": score})
        ranking = (score, -tau)
        if best is None or ranking > (best[0], best[1]):
            best = (score, -tau, metrics, calibrated)
    if best is None:
        raise RuntimeError("Fine inference calibration produced no candidate.")
    selected_tau = -best[1]
    audit = {
        "mode": str(config["fine_inference_calibration_mode"]),
        "metric": metric_name,
        "candidates": candidates,
        "selected_prior_tau": selected_tau,
        "tie_break": "smallest_prior_tau",
    }
    best[2]["fine_inference_calibration"] = audit
    return best[2], best[3], selected_tau, audit
