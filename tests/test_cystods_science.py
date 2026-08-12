from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cystods.science import (
    ScientificGateError,
    audit_rare_class_collapse,
    build_smoothed_class_prior,
    compute_fixed_primary_metrics,
    compute_multiclass_metrics,
    enforce_rare_class_collapse_gate,
    hierarchical_composite_monitor,
    mask_inactive_logits,
    semantic_fingerprint,
    split_fingerprint,
    validate_active_class_targets,
    validate_multiclass_inputs,
)


def _one_hot_predictions(
    predictions: list[int],
    num_classes: int,
) -> np.ndarray:
    probabilities = np.zeros((len(predictions), num_classes), dtype=np.float64)
    probabilities[np.arange(len(predictions)), predictions] = 1.0
    return probabilities


def test_multiclass_metrics_keep_fixed_22_class_denominator() -> None:
    targets = np.arange(16, dtype=np.int64)
    predictions = targets.copy()
    predictions[0] = 21
    probabilities = _one_hot_predictions(predictions.tolist(), 22)

    metrics = compute_multiclass_metrics(
        targets,
        probabilities,
        class_names=[f"class_{index}" for index in range(22)],
    )

    assert metrics["macro_f1_supported_class_count"] == 16
    assert metrics["macro_f1_supported"] == pytest.approx(15 / 16)
    assert metrics["macro_f1_all_classes"] == pytest.approx(15 / 22)
    predicted_only = metrics["per_class"][21]
    assert predicted_only == {
        "class_id": 21,
        "class_name": "class_21",
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "true_count": 0,
        "predicted_count": 1,
        "true_share": 0.0,
        "predicted_share": 1 / 16,
        "target_supported": False,
        "prediction_supported": True,
    }


def test_probability_and_target_validation_is_fail_fast() -> None:
    with pytest.raises(ValueError, match="NaN or infinity"):
        validate_multiclass_inputs(
            [0, 1],
            [[1.0, 0.0], [np.nan, np.nan]],
        )
    with pytest.raises(ValueError, match="sum to one"):
        validate_multiclass_inputs([0], [[0.4, 0.4]])
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        validate_multiclass_inputs([0], [[1.1, -0.1]])
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        validate_multiclass_inputs([2], [[0.5, 0.5]])
    with pytest.raises(ValueError, match="real numeric"):
        validate_multiclass_inputs(["0"], [[0.5, 0.5]])
    with pytest.raises(ValueError, match="real numeric"):
        validate_multiclass_inputs([0], [["0.5", "0.5"]])


def test_fixed_primary_metric_uses_fixed_classes_and_full_argmax() -> None:
    targets = np.asarray([0, 1, 2, 3])
    probabilities = _one_hot_predictions([0, 4, 2, 3], 5)

    metrics = compute_fixed_primary_metrics(
        targets,
        probabilities,
        [0, 1, 2],
    )

    assert metrics["n"] == 3
    assert metrics["primary_class_ids"] == [0, 1, 2]
    assert metrics["macro_f1_all_classes"] == pytest.approx(2 / 3)
    assert metrics["predictions_outside_primary_count"] == 1
    assert metrics["predictions_outside_primary_share"] == pytest.approx(1 / 3)


def test_smoothed_prior_preserves_zero_count_mask_and_is_finite() -> None:
    prior = build_smoothed_class_prior(
        [100, 1, 0],
        smoothing_alpha=1.0,
        power=0.5,
        max_ratio=5.0,
    )

    assert prior["active_mask"] == [True, True, False]
    assert prior["effective_counts"][2] == 0.0
    assert prior["probabilities"][2] == 0.0
    assert prior["log_probabilities"][2] == 0.0
    assert sum(prior["probabilities"]) == pytest.approx(1.0)
    assert max(prior["effective_counts"][:2]) / min(
        prior["effective_counts"][:2]
    ) == pytest.approx(5.0)
    for key in ("effective_counts", "probabilities", "log_probabilities"):
        assert np.isfinite(prior[key]).all()

    masked = mask_inactive_logits(
        [[1.0, 2.0, 100.0]],
        np.asarray(prior["active_mask"], dtype=bool),
    )
    assert np.isfinite(masked).all()
    assert masked.argmax(axis=1).tolist() == [1]
    validate_active_class_targets([0, 1], np.asarray([True, True, False]))
    with pytest.raises(ValueError, match="inactive classes"):
        validate_active_class_targets([2], np.asarray([True, True, False]))


def test_hierarchical_composite_is_exact_and_strict() -> None:
    components = {
        "coarse_macro_f1_all_classes": 0.8,
        "primary_macro_f1_all_classes": 0.4,
        "hierarchical_accuracy": 0.5,
    }

    assert hierarchical_composite_monitor(components) == pytest.approx(0.56)
    with pytest.raises(ValueError, match="exactly"):
        hierarchical_composite_monitor(
            {
                "coarse_macro_f1_all_classes": 0.8,
                "hierarchical_accuracy": 0.5,
            }
        )
    with pytest.raises(ValueError, match="sum to one"):
        hierarchical_composite_monitor(
            components,
            weights={
                "coarse_macro_f1_all_classes": 0.5,
                "primary_macro_f1_all_classes": 0.5,
                "hierarchical_accuracy": 0.5,
            },
        )
    invalid = dict(components)
    invalid["hierarchical_accuracy"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        hierarchical_composite_monitor(invalid)


def test_rare_class_39_percent_prediction_share_fails_gate() -> None:
    predictions = [3] * 97 + [0] * (248 - 97)
    train_counts = [339, 296, 63, 1] + [10] * 18
    train_patient_counts = [40, 46, 13, 1] + [3] * 18

    audit = audit_rare_class_collapse(
        predictions,
        train_counts,
        train_patient_counts,
        class_names=[f"class_{index}" for index in range(22)],
        max_train_patients=2,
        absolute_prediction_share=0.10,
        prior_multiplier=10.0,
        min_predicted_count=5,
    )

    assert audit["status"] == "failed"
    assert len(audit["violations"]) == 1
    assert audit["violations"][0]["class_id"] == 3
    assert audit["violations"][0]["predicted_share"] == pytest.approx(97 / 248)
    with pytest.raises(ScientificGateError, match="class_3"):
        enforce_rare_class_collapse_gate(audit)


def test_rare_class_small_prediction_count_passes_gate() -> None:
    audit = audit_rare_class_collapse(
        [3, 0, 0, 0, 0],
        [100, 10, 10, 1],
        [20, 4, 3, 1],
        max_train_patients=1,
        absolute_prediction_share=0.10,
        prior_multiplier=10.0,
        min_predicted_count=5,
    )

    assert audit["status"] == "passed"
    enforce_rare_class_collapse_gate(audit)


def test_semantic_fingerprint_is_order_stable_and_strict() -> None:
    first = {
        "seed": 7,
        "classes": ("a", "b"),
        "options": {"beta", "alpha"},
    }
    second = {
        "options": {"alpha", "beta"},
        "classes": ["a", "b"],
        "seed": np.int64(7),
    }

    assert semantic_fingerprint(first) == semantic_fingerprint(second)
    assert semantic_fingerprint(first) != semantic_fingerprint({**first, "seed": 8})
    with pytest.raises(ValueError, match="NaN or infinity"):
        semantic_fingerprint({"metric": float("nan")})


def test_split_fingerprint_is_row_order_stable_and_partition_sensitive() -> None:
    train = pd.DataFrame(
        {
            "pid": ["p2", "p1"],
            "filename": ["b.png", "a.png"],
        }
    )
    val = pd.DataFrame({"pid": ["p3"], "filename": ["c.png"]})
    test = pd.DataFrame({"pid": ["p4"], "filename": ["d.png"]})

    first = split_fingerprint({"train": train, "val": val, "test": test})
    reordered = split_fingerprint(
        {
            "test": test,
            "train": train.iloc[::-1].reset_index(drop=True),
            "val": val,
        }
    )
    moved = split_fingerprint(
        {
            "train": train,
            "val": test,
            "test": val,
        }
    )

    assert first == reordered
    assert first != moved


def test_split_fingerprint_rejects_patient_leakage() -> None:
    with pytest.raises(ValueError, match="Patient"):
        split_fingerprint(
            {
                "train": pd.DataFrame({"pid": ["p1"], "filename": ["a.png"]}),
                "test": pd.DataFrame({"pid": ["p1"], "filename": ["b.png"]}),
            }
        )


def test_split_fingerprint_rejects_nonfinite_identity() -> None:
    with pytest.raises(ValueError, match="NaN or infinity"):
        split_fingerprint(
            {"train": pd.DataFrame({"pid": ["p1"], "filename": [float("inf")]})}
        )


def test_metric_for_monitor_supports_fine_macro_f1_supported() -> None:
    from cystods.evaluation.metrics import metric_for_monitor

    bundle = {
        "binary": None,
        "coarse": None,
        "fine": {
            "macro_f1_supported": 0.654,
            "macro_f1_all_classes": 0.505,
        },
        "primary_fine": None,
        "hierarchy": None,
    }

    score = metric_for_monitor(bundle, "fine_macro_f1_supported")
    assert score == pytest.approx(0.654)

