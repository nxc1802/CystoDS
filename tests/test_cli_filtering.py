"""Unit tests for CLI model and trial filtering functionality."""

from __future__ import annotations

import pytest

from cystods.cli import _build_parser, _parse_list_arg
from cystods.config import filter_stage_trials, get_stage_trials


@pytest.fixture
def sample_stage_10_trials():
    return [
        {
            "experiment_id": "binary_swin_tiny",
            "task_mode": "binary",
            "overrides": {
                "model_name": "swin_tiny_patch4_window7_224.ms_in1k",
                "monitor_metric": "binary_auroc",
            },
        },
        {
            "experiment_id": "coarse_swin_tiny",
            "task_mode": "coarse",
            "overrides": {
                "model_name": "swin_tiny_patch4_window7_224.ms_in1k",
                "monitor_metric": "coarse_macro_f1",
            },
        },
        {
            "experiment_id": "binary_resnet152",
            "task_mode": "binary",
            "overrides": {
                "model_name": "resnet152.a1_in1k",
                "monitor_metric": "binary_auroc",
            },
        },
        {
            "experiment_id": "multitask_binary_coarse_fine_resnet152",
            "task_mode": "multitask",
            "overrides": {
                "model_name": "resnet152.a1_in1k",
                "monitor_metric": "hierarchical_composite",
            },
        },
        {
            "experiment_id": "binary_hrnet_w18",
            "task_mode": "binary",
            "overrides": {
                "model_name": "hrnet_w18.ms_in1k",
                "monitor_metric": "binary_auroc",
            },
        },
    ]


def test_parse_list_arg():
    assert _parse_list_arg(None) is None
    assert _parse_list_arg([]) is None
    assert _parse_list_arg(["swin_tiny"]) == ["swin_tiny"]
    assert _parse_list_arg(["swin_tiny", "resnet152"]) == ["swin_tiny", "resnet152"]
    assert _parse_list_arg(["swin_tiny,resnet152"]) == ["swin_tiny", "resnet152"]
    assert _parse_list_arg(["swin_tiny, resnet152 ", "hrnet_w18"]) == [
        "swin_tiny",
        "resnet152",
        "hrnet_w18",
    ]


def test_filter_stage_trials_single_model(sample_stage_10_trials):
    filtered = filter_stage_trials(
        sample_stage_10_trials,
        filter_models=["swin_tiny"],
    )
    assert len(filtered) == 2
    assert {t["experiment_id"] for t in filtered} == {
        "binary_swin_tiny",
        "coarse_swin_tiny",
    }


def test_filter_stage_trials_multiple_models(sample_stage_10_trials):
    filtered = filter_stage_trials(
        sample_stage_10_trials,
        filter_models=["swin_tiny", "resnet152"],
    )
    assert len(filtered) == 4
    assert {t["experiment_id"] for t in filtered} == {
        "binary_swin_tiny",
        "coarse_swin_tiny",
        "binary_resnet152",
        "multitask_binary_coarse_fine_resnet152",
    }


def test_filter_stage_trials_by_trial_id(sample_stage_10_trials):
    filtered = filter_stage_trials(
        sample_stage_10_trials,
        filter_trials=["binary_resnet152"],
    )
    assert len(filtered) == 1
    assert filtered[0]["experiment_id"] == "binary_resnet152"


def test_filter_stage_trials_combined_filters(sample_stage_10_trials):
    filtered = filter_stage_trials(
        sample_stage_10_trials,
        filter_models=["resnet152"],
        filter_trials=["binary_resnet152"],
    )
    assert len(filtered) == 1
    assert filtered[0]["experiment_id"] == "binary_resnet152"


def test_cli_parser_model_and_trial_flags():
    parser = _build_parser()

    # Test single model
    args = parser.parse_args(["run", "10", "--models", "swin_tiny"])
    assert args.command == "run"
    assert args.stage == "10"
    assert args.models == ["swin_tiny"]

    # Test multiple models space-separated
    args = parser.parse_args(["run", "10", "--models", "swin_tiny", "resnet152"])
    assert args.models == ["swin_tiny", "resnet152"]

    # Test comma-separated models via alias --model
    args = parser.parse_args(["run", "10", "--model", "swin_tiny,resnet152"])
    assert args.models == ["swin_tiny,resnet152"]

    # Test --trials alias --trial
    args = parser.parse_args(["run", "10", "--trial", "binary_swin_tiny"])
    assert args.trials == ["binary_swin_tiny"]


def test_get_stage_trials_with_filters():
    # Load actual Stage 10 trials from config.yaml
    swin_trials = get_stage_trials(stage="10", filter_models=["swin_tiny"])
    assert len(swin_trials) > 0
    assert all("swin_tiny" in t["overrides"]["model_name"] for t in swin_trials)

    multi_model_trials = get_stage_trials(stage="10", filter_models=["swin_tiny", "resnet152"])
    assert len(multi_model_trials) > len(swin_trials)
