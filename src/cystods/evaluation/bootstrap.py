"""Patient-level percentile bootstrap confidence intervals and McNemar testing.

Extracted from ``cystods.core`` (Step 5 refactor).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from cystods.evaluation.metrics import compute_metrics_bundle


def patient_bootstrap_intervals(
    predictions: pd.DataFrame,
    train_fine_counts: Sequence[int],
    train_fine_patient_counts: Sequence[int],
    config: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    iterations = int(config["bootstrap_iterations"])
    if iterations < 1:
        raise ValueError("bootstrap_iterations must be positive.")
    patients = sorted(predictions["pid"].astype(str).unique())
    if len(patients) < 2:
        return {
            "status": "not_evaluable",
            "reason": "fewer than two patients",
            "iterations_requested": iterations,
            "iterations_valid": 0,
        }
    grouped = {
        pid: predictions[predictions["pid"].astype(str) == pid]
        for pid in patients
    }
    rng = np.random.default_rng(seed)
    samples: defaultdict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        selected = rng.choice(patients, size=len(patients), replace=True)
        pieces = []
        for draw_index, pid in enumerate(selected):
            piece = grouped[str(pid)].copy()
            # Make duplicated patient draws statistically independent groups
            # without changing any row-level prediction.
            piece["pid"] = f"{pid}__bootstrap_{draw_index}"
            pieces.append(piece)
        boot = pd.concat(pieces, ignore_index=True)
        bundle = compute_metrics_bundle(
            boot,
            train_fine_counts,
            train_fine_patient_counts,
            config,
        )
        candidates = {
            "binary_auroc": (
                bundle["binary"]["auroc"]
                if bundle["binary"] is not None
                else None
            ),
            "binary_auprc": (
                bundle["binary"]["auprc"]
                if bundle["binary"] is not None
                else None
            ),
            "binary_f1": (
                bundle["binary"]["f1"]
                if bundle["binary"] is not None
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
            "fine_macro_f1_all_classes": (
                bundle["fine"]["macro_f1_all_classes"]
                if bundle["fine"] is not None
                else None
            ),
            "primary_macro_f1_all_classes": (
                bundle["primary_fine"]["macro_f1_all_classes"]
                if bundle["primary_fine"] is not None
                and bundle["primary_fine"]["status"] == "ok"
                else None
            ),
            "hierarchical_accuracy": (
                bundle["hierarchy"]["hierarchical_accuracy"]
                if bundle["hierarchy"] is not None
                else None
            ),
        }
        for name, value in candidates.items():
            if value is not None and math.isfinite(float(value)):
                samples[name].append(float(value))
    alpha = 1.0 - float(config["bootstrap_confidence"])
    intervals: dict[str, Any] = {}
    for name, values in samples.items():
        intervals[name] = {
            "lower": float(np.quantile(values, alpha / 2)),
            "upper": float(np.quantile(values, 1 - alpha / 2)),
            "mean": float(np.mean(values)),
            "valid_iterations": len(values),
        }
    return {
        "status": "ok",
        "method": "patient-level percentile bootstrap",
        "confidence": float(config["bootstrap_confidence"]),
        "iterations_requested": iterations,
        "intervals": intervals,
    }


def paired_mcnemar_test(
    current: pd.DataFrame,
    baseline_csv: Path,
    binary_threshold: float,
) -> dict[str, Any]:
    if not baseline_csv.is_file():
        raise FileNotFoundError(
            f"Paired baseline predictions not found: {baseline_csv}"
        )
    baseline = pd.read_csv(baseline_csv)
    required = {"filename", "binary_id", "binary_probability_roi"}
    missing = required - set(baseline)
    if missing:
        raise ValueError(
            f"Paired baseline CSV missing columns: {sorted(missing)}"
        )
    current_view = current[["filename", "binary_id", "binary_probs"]].copy()
    current_view["current_prediction"] = current_view["binary_probs"].map(
        lambda values: int(
            np.asarray(values)[1] >= binary_threshold
        )
    )
    merged = current_view.merge(
        baseline,
        on=["filename", "binary_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(current_view) or len(merged) != len(baseline):
        raise ValueError(
            "Paired prediction manifests do not have identical samples."
        )
    baseline_prediction = (
        merged["binary_probability_roi"].astype(float).to_numpy()
        >= binary_threshold
    )
    target = merged["binary_id"].astype(int).to_numpy()
    current_correct = merged["current_prediction"].to_numpy() == target
    baseline_correct = baseline_prediction == target
    current_only = int(np.sum(current_correct & ~baseline_correct))
    baseline_only = int(np.sum(~current_correct & baseline_correct))
    discordant = current_only + baseline_only
    p_value = (
        float(
            binomtest(
                min(current_only, baseline_only),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
        if discordant
        else 1.0
    )
    return {
        "test": "exact McNemar (binomial)",
        "decision_threshold": float(binary_threshold),
        "n": len(merged),
        "current_correct_baseline_wrong": current_only,
        "current_wrong_baseline_correct": baseline_only,
        "discordant": discordant,
        "p_value": p_value,
    }
