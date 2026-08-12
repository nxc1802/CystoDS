from __future__ import annotations

import ast
import importlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

core = importlib.import_module("cystods.core")


def _module_tree(filename: str) -> ast.Module:
    root = Path(__file__).resolve().parents[1]
    short_name = filename
    if filename.startswith("stage_"):
        stage_num = filename.split("_")[1]
        short_name = f"stage_{stage_num}.py"
    elif filename == "cystods_core.py":
        short_name = "core.py"

    candidates = [
        root / "src" / "cystods" / "stages" / short_name,
        root / "src" / "cystods" / short_name,
        root / "src" / "cystods" / filename,
        root / "src" / "cystods" / "stages" / filename,
        root / "notebook" / filename,
        root / filename,
    ]
    for path in candidates:
        if path.is_file():
            return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    raise FileNotFoundError(f"Module file not found for AST parsing: {filename}")


def _literal_assignment(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Assignment {name!r} was not found.")


def _literal_dict_values(tree: ast.AST, key_name: str) -> set[Any]:
    values: set[Any] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            if (
                isinstance(key, ast.Constant)
                and key.value == key_name
                and isinstance(value, ast.Constant)
            ):
                values.add(value.value)
    return values


_MODE_WEIGHTS = {
    "binary": {
        "binary_loss_weight": 1.0,
        "coarse_loss_weight": 0.0,
        "fine_loss_weight": 0.0,
        "binary_coarse_hierarchy_loss_weight": 0.0,
        "coarse_fine_hierarchy_loss_weight": 0.0,
        "supervised_contrastive_loss_weight": 0.0,
        "monitor_metric": "binary_f1",
    },
    "coarse": {
        "binary_loss_weight": 0.0,
        "coarse_loss_weight": 1.0,
        "fine_loss_weight": 0.0,
        "binary_coarse_hierarchy_loss_weight": 0.0,
        "coarse_fine_hierarchy_loss_weight": 0.0,
        "supervised_contrastive_loss_weight": 0.0,
        "monitor_metric": "coarse_macro_f1",
    },
    "fine": {
        "binary_loss_weight": 0.0,
        "coarse_loss_weight": 0.0,
        "fine_loss_weight": 1.0,
        "binary_coarse_hierarchy_loss_weight": 0.0,
        "coarse_fine_hierarchy_loss_weight": 0.0,
        "supervised_contrastive_loss_weight": 0.0,
        "monitor_metric": "fine_macro_f1",
    },
    "multitask": {
        "binary_loss_weight": 1.0,
        "coarse_loss_weight": 1.0,
        "fine_loss_weight": 1.0,
        "binary_coarse_hierarchy_loss_weight": 0.0,
        "coarse_fine_hierarchy_loss_weight": 0.0,
        "supervised_contrastive_loss_weight": 0.0,
        "monitor_metric": "coarse_macro_f1",
    },
    "hierarchical": {
        "binary_loss_weight": 1.0,
        "coarse_loss_weight": 1.0,
        "fine_loss_weight": 1.0,
        "binary_coarse_hierarchy_loss_weight": 0.25,
        "coarse_fine_hierarchy_loss_weight": 0.25,
        "supervised_contrastive_loss_weight": 0.0,
        "monitor_metric": "hierarchical_composite",
    },
}


def _config_for(task_mode: str) -> dict[str, Any]:
    config = dict(core.BASE_CONFIG)
    config["hierarchical_composite_weights"] = dict(
        core.BASE_CONFIG["hierarchical_composite_weights"]
    )
    config["task_mode"] = task_mode
    config.update(_MODE_WEIGHTS[task_mode])
    return config


def _one_hot(class_ids: list[int], num_classes: int) -> list[np.ndarray]:
    probabilities = np.zeros((len(class_ids), num_classes), dtype=np.float64)
    probabilities[np.arange(len(class_ids)), class_ids] = 1.0
    return list(probabilities)


def test_base_config_passes_its_own_validator() -> None:
    core.validate_config(core.BASE_CONFIG)


def test_stage_00_declares_one_fixed_70_15_15_holdout() -> None:
    from cystods.config import load_config
    config = load_config(stage="00")

    assert config["protocol"] == "holdout"
    assert config["train_fraction"] == 0.70
    assert config["val_fraction"] == 0.15
    assert config["test_fraction"] == 0.15


@pytest.mark.parametrize(
    ("role", "scope", "protocol"),
    [
        ("fixed_holdout", "development", "holdout"),
        ("final_cv", "final_cv", "cross_validation"),
    ],
)
def test_new_protocol_roles_are_valid_core_configuration(
    role: str,
    scope: str,
    protocol: str,
) -> None:
    config = dict(core.BASE_CONFIG)
    config.update(
        protocol_role=role,
        evaluation_scope=scope,
        protocol=protocol,
    )

    core.validate_config(config)


@pytest.mark.parametrize(
    ("role", "fold_ids"),
    [("fixed_holdout", None), ("final_cv", [0, 2])],
)
def test_protocol_binding_supports_fixed_holdout_and_final_cv(
    tmp_path: Path,
    role: str,
    fold_ids: list[int] | None,
) -> None:
    protocol_manifest = {
        "schema_version": "cystods.protocol.v2",
        "study_id": "study",
        "run_profile": "research",
        "roles": {
            "fixed_holdout": {
                "protocol": "holdout",
                "units": ["holdout"],
            }
        },
    }
    (tmp_path / "run_status.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )
    protocol_path = tmp_path / "protocol_manifest.json"
    protocol_path.write_text(json.dumps(protocol_manifest), encoding="utf-8")
    expected_sha256 = core.sha256_file(protocol_path)
    stage_config = {
        "protocol_run_dir": tmp_path,
        "expected_protocol_sha256": expected_sha256,
        "result_root": tmp_path,
        "run_profile": "research",
        "study_id": "study",
        "protocol_role": role,
        "fold_ids": fold_ids,
    }

    run_dir, loaded_manifest, actual_sha256 = core._load_and_validate_protocol_binding(
        stage_config
    )

    assert run_dir == tmp_path.resolve()
    assert loaded_manifest == protocol_manifest
    assert actual_sha256 == expected_sha256


def test_fixed_holdout_rejects_cv_fold_selection(tmp_path: Path) -> None:
    protocol_manifest = {
        "schema_version": "cystods.protocol.v2",
        "study_id": "study",
        "run_profile": "research",
        "roles": {
            "fixed_holdout": {
                "protocol": "holdout",
                "units": ["holdout"],
            }
        },
    }
    (tmp_path / "run_status.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )
    protocol_path = tmp_path / "protocol_manifest.json"
    protocol_path.write_text(json.dumps(protocol_manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="final_cv"):
        core._load_and_validate_protocol_binding(
            {
                "protocol_run_dir": tmp_path,
                "expected_protocol_sha256": core.sha256_file(protocol_path),
                "result_root": tmp_path,
                "run_profile": "research",
                "study_id": "study",
                "protocol_role": "fixed_holdout",
                "fold_ids": [0],
            }
        )


def test_split_seed_is_independent_of_model_training_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed_search_seeds: list[int] = []
    observed_materialization_seeds: list[int] = []
    frame = pd.DataFrame({"pid": ["p1", "p2", "p3"]})
    patient_split = {
        "train": {"p1"},
        "val": {"p2"},
        "test": {"p3"},
    }
    split_frames = {
        split_name: frame.loc[frame["pid"].isin(patient_ids)].copy()
        for split_name, patient_ids in patient_split.items()
    }

    def fake_search(
        _frame: pd.DataFrame,
        _config: dict[str, Any],
        seed: int,
    ) -> tuple[dict[str, set[str]], float]:
        observed_search_seeds.append(seed)
        return patient_split, 0.0

    def fake_materialize(
        _frame: pd.DataFrame,
        _patient_split: dict[str, set[str]],
        _config: dict[str, Any],
        seed: int,
    ) -> dict[str, pd.DataFrame]:
        observed_materialization_seeds.append(seed)
        return split_frames

    monkeypatch.setattr(core, "search_holdout_patient_split", fake_search)
    monkeypatch.setattr(core, "materialize_split_frames", fake_materialize)
    monkeypatch.setattr(core, "validate_materialized_splits", lambda *_args: None)
    monkeypatch.setattr(core, "save_split_artifacts", lambda *_args: None)

    for model_seed in (7, 999_983):
        config = dict(core.BASE_CONFIG)
        config.update(
            protocol="holdout",
            seed=model_seed,
            split_seed=314_159,
            fixed_split_pids=None,
            protocol_manifest_dir=None,
        )
        units = core.build_all_protocol_splits(
            frame,
            config,
            tmp_path,
            logging.getLogger("cystods-contract-test"),
        )
        assert [unit[0] for unit in units] == ["holdout"]

    assert observed_search_seeds == [314_159, 314_159]
    assert observed_materialization_seeds == [314_159, 314_159]


class _FrameDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        _transform: object,
        second_view_transform: object | None = None,
        *,
        second_view: bool | None = None,
    ) -> None:
        self.frame = frame
        self.second_view = second_view_transform is not None or bool(second_view)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> int:
        return index


def test_build_dataloaders_only_builds_train_and_val_with_separate_eval_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        core,
        "build_transforms",
        lambda _config: (object(), object(), object()),
    )
    monkeypatch.setattr(core, "CystoDataset", _FrameDataset)
    config = dict(core.BASE_CONFIG)
    config.update(
        sampler="random",
        supervised_contrastive_loss_weight=0.0,
        batch_size=3,
        eval_batch_size=7,
        num_workers=1,
        eval_num_workers=2,
        prefetch_factor=4,
        eval_prefetch_factor=2,
        persistent_workers=False,
        pin_memory=False,
    )
    split_frames = {
        "train": pd.DataFrame({"sample": range(9)}),
        "val": pd.DataFrame({"sample": range(8)}),
        "test": pd.DataFrame({"sample": range(5)}),
    }

    loaders, datasets = core.build_dataloaders(
        split_frames,
        config,
        torch.device("cpu"),
        seed=101,
    )

    assert set(loaders) == {"train", "val"}
    assert set(datasets) == {"train", "val"}
    assert loaders["train"].batch_size == 3
    assert loaders["val"].batch_size == 7
    assert loaders["train"].num_workers == 1
    assert loaders["val"].num_workers == 2
    assert loaders["train"].prefetch_factor == 4
    assert loaders["val"].prefetch_factor == 2


def test_stage_10_declares_baseline_trials_for_paper_backbones() -> None:
    from cystods.config import get_stage_trials
    trials = get_stage_trials(stage="10")
    expected_active_tasks = {
        "binary_swin_tiny": {"binary"},
        "multitask_swin_tiny": {"binary", "coarse", "fine"},
        "binary_resnet152": {"binary"},
        "multitask_resnet152": {"binary", "coarse", "fine"},
        "binary_hrnet_w18": {"binary"},
        "multitask_hrnet_w18": {"binary", "coarse", "fine"},
        "binary_resnext50_32x4d": {"binary"},
        "multitask_resnext50_32x4d": {"binary", "coarse", "fine"},
    }

    assert len(trials) == len(expected_active_tasks)
    assert {trial["experiment_id"] for trial in trials} == set(expected_active_tasks)
    from cystods.config import load_config
    base_stage10 = load_config(stage="10")
    for trial in trials:
        exp_id = trial["experiment_id"]
        task_mode = trial.get("task_mode", "binary")
        trial_cfg = dict(base_stage10)
        trial_cfg["task_mode"] = task_mode
        if task_mode in ("binary", "coarse", "fine"):
            trial_cfg["binary_loss_weight"] = 1.0 if task_mode == "binary" else 0.0
            trial_cfg["coarse_loss_weight"] = 1.0 if task_mode == "coarse" else 0.0
            trial_cfg["fine_loss_weight"] = 1.0 if task_mode == "fine" else 0.0
        trial_cfg.update(trial.get("overrides", {}))
        active = core.active_tasks_from_config(trial_cfg)
        assert set(active) == expected_active_tasks[exp_id]

    model_names = {trial.get("overrides", {}).get("model_name", "swin_tiny_patch4_window7_224.ms_in1k") for trial in trials}
    assert model_names >= {
        "swin_tiny_patch4_window7_224.ms_in1k",
        "resnet152.a1_in1k",
        "hrnet_w18.ms_in1k",
        "resnext50_32x4d.a1_in1k",
    }


@pytest.mark.parametrize(
    "model_name",
    ["resnet152", "hrnet_w18", "resnext50_32x4d", "swin_tiny"],
)
def test_paper_baseline_backbones_instantiate(model_name: str) -> None:
    config = _config_for("binary")
    config["model_name"] = model_name
    core.validate_config(config)
    model = core.HierarchicalCystoModel(config)
    assert model.encoder is not None
    x = torch.randn(2, 3, 224, 224)
    features = model.encode(x)
    assert features.shape[0] == 2
    assert features.shape[1] == model.feature_dim


@pytest.mark.parametrize("task_mode", sorted(_MODE_WEIGHTS))
def test_each_task_mode_has_a_valid_minimal_objective(task_mode: str) -> None:
    config = _config_for(task_mode)

    core.validate_config(config)

    expected = {
        "binary": frozenset({"binary"}),
        "coarse": frozenset({"coarse"}),
        "fine": frozenset({"fine"}),
        "multitask": frozenset({"binary", "coarse", "fine"}),
        "hierarchical": frozenset({"binary", "coarse", "fine"}),
    }[task_mode]
    assert core.active_tasks_for_mode(task_mode) == expected


def test_task_mode_rejects_nonzero_inactive_objective() -> None:
    config = _config_for("binary")
    config["fine_loss_weight"] = 1.0

    with pytest.raises(ValueError, match="zero inactive weights"):
        core.validate_config(config)


def test_consistency_objective_runs_without_binary_auxiliary_head() -> None:
    config = _config_for("hierarchical")
    config["binary_loss_weight"] = 0.0
    core.validate_config(config)
    assert core.active_tasks_from_config(config) == frozenset({"coarse", "fine"})
    image_counts = [10, 2, *([0] * 20)]
    patient_counts = [3, 1, *([0] * 20)]
    loss_fn = core.FineLongTailLoss(
        "cross_entropy",
        image_counts,
        patient_counts,
        config,
    )
    outputs = {
        "features": torch.ones((1, 4)),
        "coarse_logits": torch.tensor([[5.0, 0.0, 0.0, 0.0, 0.0]]),
        "fine_logits": torch.tensor(
            [[5.0, 0.0, *([100.0] * 20)]],
        ),
    }

    loss_first = core.compute_multitask_loss(
        outputs,
        torch.tensor([1]),
        torch.tensor([core.FINE_PARENT_ID[0]]),
        torch.tensor([0]),
        loss_fn,
        config,
    )[0]
    outputs_second = {**outputs, "fine_logits": outputs["fine_logits"].clone()}
    outputs_second["fine_logits"][0, 4] = 100.0
    loss_second = core.compute_multitask_loss(
        outputs_second,
        torch.tensor([1]),
        torch.tensor([core.FINE_PARENT_ID[0]]),
        torch.tensor([0]),
        loss_fn,
        config,
    )[0]

    assert torch.allclose(loss_first, loss_second)


def test_binary_coarse_loss_rewards_correct_branch() -> None:
    coarse_probs = torch.tensor(
        [[0.50, 0.40, 0.05, 0.03, 0.02]],
        dtype=torch.float32,
    )
    logits = coarse_probs.log()
    binary_target = torch.tensor([1], dtype=torch.long)
    loss = core.binary_coarse_hierarchy_loss(logits, binary_target)
    expected = -math.log(0.90)
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-5)


def test_binary_coarse_loss_ignores_within_branch_split() -> None:
    first = torch.tensor([[0.80, 0.10, 0.05, 0.03, 0.02]], dtype=torch.float32)
    second = torch.tensor([[0.10, 0.80, 0.05, 0.03, 0.02]], dtype=torch.float32)
    target = torch.tensor([1], dtype=torch.long)
    loss_first = core.binary_coarse_hierarchy_loss(first.log(), target)
    loss_second = core.binary_coarse_hierarchy_loss(second.log(), target)
    assert torch.allclose(loss_first, loss_second, atol=1e-6)


def test_binary_coarse_loss_penalizes_wrong_branch() -> None:
    correct = torch.tensor([[0.50, 0.40, 0.05, 0.03, 0.02]], dtype=torch.float32)
    wrong = torch.tensor([[0.02, 0.03, 0.50, 0.25, 0.20]], dtype=torch.float32)
    target = torch.tensor([1], dtype=torch.long)
    correct_loss = core.binary_coarse_hierarchy_loss(correct.log(), target)
    wrong_loss = core.binary_coarse_hierarchy_loss(wrong.log(), target)
    assert wrong_loss > correct_loss


def test_coarse_fine_loss_rewards_correct_parent() -> None:
    config = _config_for("hierarchical")
    image_counts = [10] * len(core.FINE_NAMES)
    patient_counts = [5] * len(core.FINE_NAMES)
    fine_loss_fn = core.FineLongTailLoss("cross_entropy", image_counts, patient_counts, config)

    probs = torch.full((1, len(core.FINE_NAMES)), 1e-6, dtype=torch.float32)
    lg = core.FINE_TO_ID["LowGradePapillary"]
    hg = core.FINE_TO_ID["HighGradePapillary"]
    air = core.FINE_TO_ID["AirBubble"]

    probs[0, lg] = 0.45
    probs[0, hg] = 0.45
    probs[0, air] = 0.10
    probs = probs / probs.sum(dim=1, keepdim=True)

    coarse_target = torch.tensor([core.COARSE_TO_ID["Malignant"]], dtype=torch.long)
    fine_target = torch.tensor([hg], dtype=torch.long)

    loss = core.coarse_fine_hierarchy_loss(probs.log(), coarse_target, fine_target, fine_loss_fn)
    assert loss.item() < 0.2


def test_cf_hierarchy_does_not_care_which_child_inside_parent() -> None:
    config = _config_for("hierarchical")
    image_counts = [10] * len(core.FINE_NAMES)
    patient_counts = [5] * len(core.FINE_NAMES)
    fine_loss_fn = core.FineLongTailLoss("cross_entropy", image_counts, patient_counts, config)

    first = torch.full((1, len(core.FINE_NAMES)), 1e-6, dtype=torch.float32)
    second = torch.full((1, len(core.FINE_NAMES)), 1e-6, dtype=torch.float32)

    lg = core.FINE_TO_ID["LowGradePapillary"]
    hg = core.FINE_TO_ID["HighGradePapillary"]
    air = core.FINE_TO_ID["AirBubble"]

    first[0, lg] = 0.80
    first[0, hg] = 0.10
    first[0, air] = 0.10

    second[0, lg] = 0.10
    second[0, hg] = 0.80
    second[0, air] = 0.10

    first = first / first.sum(dim=1, keepdim=True)
    second = second / second.sum(dim=1, keepdim=True)

    coarse = torch.tensor([core.COARSE_TO_ID["Malignant"]], dtype=torch.long)
    fine = torch.tensor([hg], dtype=torch.long)

    loss_1 = core.coarse_fine_hierarchy_loss(first.log(), coarse, fine, fine_loss_fn)
    loss_2 = core.coarse_fine_hierarchy_loss(second.log(), coarse, fine, fine_loss_fn)

    assert torch.allclose(loss_1, loss_2, atol=1e-5)


def test_cf_hierarchy_penalizes_cross_parent_probability() -> None:
    config = _config_for("hierarchical")
    image_counts = [10] * len(core.FINE_NAMES)
    patient_counts = [5] * len(core.FINE_NAMES)
    fine_loss_fn = core.FineLongTailLoss("cross_entropy", image_counts, patient_counts, config)

    malignant = torch.full((1, len(core.FINE_NAMES)), 1e-6, dtype=torch.float32)
    foreign = torch.full((1, len(core.FINE_NAMES)), 1e-6, dtype=torch.float32)

    hg = core.FINE_TO_ID["HighGradePapillary"]
    air = core.FINE_TO_ID["AirBubble"]

    malignant[0, hg] = 0.95
    malignant[0, air] = 0.05

    foreign[0, hg] = 0.05
    foreign[0, air] = 0.95

    malignant = malignant / malignant.sum(dim=1, keepdim=True)
    foreign = foreign / foreign.sum(dim=1, keepdim=True)

    coarse = torch.tensor([core.COARSE_TO_ID["Malignant"]], dtype=torch.long)
    fine = torch.tensor([hg], dtype=torch.long)

    good_loss = core.coarse_fine_hierarchy_loss(malignant.log(), coarse, fine, fine_loss_fn)
    bad_loss = core.coarse_fine_hierarchy_loss(foreign.log(), coarse, fine, fine_loss_fn)

    assert bad_loss > good_loss


def test_cf_hierarchy_ignores_normal_mucosa() -> None:
    config = _config_for("hierarchical")
    image_counts = [10] * len(core.FINE_NAMES)
    patient_counts = [5] * len(core.FINE_NAMES)
    fine_loss_fn = core.FineLongTailLoss("cross_entropy", image_counts, patient_counts, config)

    logits = torch.randn(1, len(core.FINE_NAMES), dtype=torch.float32)
    coarse = torch.tensor([core.COARSE_TO_ID["Normal mucosa"]], dtype=torch.long)
    fine = torch.tensor([-1], dtype=torch.long)

    loss = core.coarse_fine_hierarchy_loss(logits, coarse, fine, fine_loss_fn)
    assert float(loss) == 0.0


def test_bc_loss_only_backpropagates_to_coarse_logits() -> None:
    coarse_logits = torch.randn(4, len(core.COARSE_NAMES), requires_grad=True)
    binary_target = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    loss = core.binary_coarse_hierarchy_loss(coarse_logits, binary_target)
    loss.backward()
    assert coarse_logits.grad is not None
    assert torch.isfinite(coarse_logits.grad).all()


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("fine_prior_smoothing_alpha", float("nan")),
        ("fine_prior_max_ratio", float("nan")),
        ("binary_loss_weight", float("nan")),
        ("checkpoint_min_delta", float("nan")),
        ("probability_sum_tolerance", float("nan")),
    ],
)
def test_config_rejects_nonfinite_numeric_values(
    field: str,
    invalid: float,
) -> None:
    config = _config_for("hierarchical")
    config[field] = invalid

    with pytest.raises(ValueError, match="finite"):
        core.validate_config(config)


def test_config_does_not_truncate_fractional_primary_class_ids() -> None:
    config = _config_for("hierarchical")
    config["fixed_primary_fine_class_ids"] = (0, 1.5)

    with pytest.raises(ValueError, match="primary"):
        core.validate_config(config)


class _TinyEncoder(nn.Module):
    num_features = 4

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = images.mean(dim=(2, 3))
        return torch.cat((pooled, pooled[:, :1]), dim=1)


@pytest.mark.parametrize(
    ("task_mode", "expected_logits"),
    [
        ("binary", {"binary_logits"}),
        ("coarse", {"coarse_logits"}),
        ("fine", {"fine_logits"}),
        (
            "multitask",
            {"binary_logits", "coarse_logits", "fine_logits"},
        ),
        (
            "hierarchical",
            {"binary_logits", "coarse_logits", "fine_logits"},
        ),
    ],
)
def test_model_only_constructs_heads_for_active_tasks(
    monkeypatch: pytest.MonkeyPatch,
    task_mode: str,
    expected_logits: set[str],
) -> None:
    monkeypatch.setattr(core.timm, "is_model", lambda _name: True)
    monkeypatch.setattr(
        core.timm,
        "create_model",
        lambda *_args, **_kwargs: _TinyEncoder(),
    )
    config = _config_for(task_mode)
    model = core.HierarchicalCystoModel(config)

    outputs = model(torch.ones((2, 3, 8, 8), dtype=torch.float32))

    assert set(outputs) == {"features", *expected_logits}
    assert outputs["features"].shape == (2, 4)


def test_binary_loss_contract_ignores_no_inactive_targets() -> None:
    config = _config_for("binary")
    outputs = {
        "features": torch.ones((2, 4)),
        "binary_logits": torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
    }
    targets = torch.tensor([0, 1], dtype=torch.long)

    loss, components = core.compute_multitask_loss(
        outputs,
        targets,
        torch.tensor([999, 999]),
        torch.tensor([999, 999]),
        None,
        config,
    )

    assert torch.isfinite(loss)
    assert set(components) == {"binary_loss", "classification_total"}
    assert components["classification_total"] == pytest.approx(
        components["binary_loss"]
    )


def test_loss_contract_rejects_an_inactive_head() -> None:
    config = _config_for("binary")
    outputs = {
        "features": torch.ones((1, 4)),
        "binary_logits": torch.ones((1, 2)),
        "coarse_logits": torch.ones((1, 5)),
    }

    with pytest.raises(ValueError, match="unexpected=.*coarse_logits"):
        core.compute_multitask_loss(
            outputs,
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([-1]),
            None,
            config,
        )


def test_smoothed_fine_loss_masks_classes_absent_from_training() -> None:
    config = _config_for("fine")
    config.update(
        fine_loss="balanced_softmax_smoothed",
        fine_prior_source="patient_count",
        fine_prior_smoothing_alpha=1.0,
        fine_prior_power=0.5,
        fine_prior_max_ratio=10.0,
    )
    image_counts = [10, 2, *([0] * 20)]
    patient_counts = [3, 1, *([0] * 20)]
    loss_fn = core.FineLongTailLoss(
        "balanced_softmax_smoothed",
        image_counts,
        patient_counts,
        config,
    )

    assert loss_fn.active_mask.tolist() == [True, True, *([False] * 20)]
    assert float(loss_fn.prior_probabilities.sum()) == pytest.approx(1.0)
    logits = torch.zeros((2, len(core.FINE_NAMES)), dtype=torch.float32)
    logits[:, 2:] = 100.0
    loss = loss_fn(logits, torch.tensor([0, 1], dtype=torch.long))
    probabilities = loss_fn.inference_logits(logits, 0.0).softmax(dim=1)

    assert torch.isfinite(loss)
    assert torch.isfinite(probabilities).all()
    assert probabilities[:, 2:].sum().item() == 0.0
    assert probabilities[:, :2].sum(dim=1).tolist() == pytest.approx([1.0, 1.0])


def test_canonical_balanced_softmax_masks_absent_train_class() -> None:
    config = _config_for("fine")
    image_counts = [10, 2, *([0] * 20)]
    patient_counts = [3, 1, *([0] * 20)]
    loss_fn = core.FineLongTailLoss(
        "balanced_softmax",
        image_counts,
        patient_counts,
        config,
    )
    logits = torch.zeros((2, len(core.FINE_NAMES)), dtype=torch.float32)
    logits[:, 2:] = 100.0

    loss = loss_fn(logits, torch.tensor([0, 1], dtype=torch.long))
    probabilities = loss_fn.inference_logits(logits, 0.0).softmax(dim=1)

    assert torch.isfinite(loss)
    assert probabilities[:, 2:].sum().item() == 0.0


def test_coarse_fine_hierarchy_ignores_inactive_fine_logits() -> None:
    config = _config_for("hierarchical")
    config["coarse_fine_hierarchy_loss_weight"] = 1.0
    image_counts = [10, 2, *([0] * 20)]
    patient_counts = [3, 1, *([0] * 20)]
    loss_fn = core.FineLongTailLoss(
        "cross_entropy",
        image_counts,
        patient_counts,
        config,
    )
    shared = {
        "features": torch.ones((1, 4)),
        "binary_logits": torch.tensor([[0.0, 5.0]]),
        "coarse_logits": torch.tensor([[5.0, 0.0, 0.0, 0.0, 0.0]]),
    }
    first_fine_logits = torch.zeros((1, len(core.FINE_NAMES)))
    first_fine_logits[0, 0] = 5.0
    second_fine_logits = first_fine_logits.clone()
    second_fine_logits[0, 4] = 100.0

    _, first_components = core.compute_multitask_loss(
        {**shared, "fine_logits": first_fine_logits},
        torch.tensor([1]),
        torch.tensor([core.FINE_PARENT_ID[0]]),
        torch.tensor([0]),
        loss_fn,
        config,
    )
    _, second_components = core.compute_multitask_loss(
        {**shared, "fine_logits": second_fine_logits},
        torch.tensor([1]),
        torch.tensor([core.FINE_PARENT_ID[0]]),
        torch.tensor([0]),
        loss_fn,
        config,
    )

    assert first_components["coarse_fine_hierarchy_loss"] == pytest.approx(
        second_components["coarse_fine_hierarchy_loss"]
    )


def test_prediction_export_only_contains_active_task_outputs() -> None:
    outputs = {
        "features": torch.ones((2, 4)),
        "binary_logits": torch.tensor([[4.0, 0.0], [0.0, 4.0]]),
    }
    batch = {
        "filename": ["a.png", "b.png"],
        "pid": ["p1", "p2"],
        "binary_id": torch.tensor([0, 1]),
        "coarse_id": torch.tensor([2, 0]),
        "fine_id": torch.tensor([-1, 0]),
    }

    rows = core.prediction_rows_from_outputs(
        outputs,
        batch,
        include_features=False,
        fine_loss_fn=None,
        fine_prior_tau=0.0,
    )

    assert len(rows) == 2
    assert "binary_probs" in rows[0]
    assert "coarse_probs" not in rows[0]
    assert "fine_probs" not in rows[0]
    assert "features" not in rows[0]


def test_binary_metrics_bundle_marks_other_tasks_inactive() -> None:
    config = _config_for("binary")
    predictions = pd.DataFrame(
        {
            "binary_id": [0, 1, 1],
            "coarse_id": [2, 0, 1],
            "fine_id": [-1, 0, 4],
            "binary_probs": _one_hot([0, 1, 1], 2),
        }
    )

    metrics = core.compute_metrics_bundle(
        predictions,
        [0] * len(core.FINE_NAMES),
        [0] * len(core.FINE_NAMES),
        config,
    )

    assert metrics["binary"]["f1"] == pytest.approx(1.0)
    assert metrics["coarse"] is None
    assert metrics["fine"] is None
    assert metrics["primary_fine"] is None
    assert metrics["hierarchy"] is None
    assert metrics["rare_class_collapse"] is None


def test_fine_metrics_keep_all_22_classes_as_fixed_denominator() -> None:
    config = _config_for("fine")
    config["fixed_primary_fine_class_ids"] = (0, 1)
    predictions = pd.DataFrame(
        {
            "binary_id": [1, 1],
            "coarse_id": [
                core.FINE_PARENT_ID[0],
                core.FINE_PARENT_ID[1],
            ],
            "fine_id": [0, 1],
            "fine_probs": _one_hot([0, 1], len(core.FINE_NAMES)),
        }
    )
    train_counts = [10, 10, *([0] * 20)]
    patient_counts = [3, 3, *([0] * 20)]

    metrics = core.compute_metrics_bundle(
        predictions,
        train_counts,
        patient_counts,
        config,
    )

    assert metrics["binary"] is None
    assert metrics["coarse"] is None
    assert metrics["fine"]["macro_f1_supported"] == pytest.approx(1.0)
    assert metrics["fine"]["macro_f1_all_classes"] == pytest.approx(2 / 22)
    assert metrics["primary_fine"]["macro_f1_all_classes"] == pytest.approx(1.0)
    assert metrics["hierarchy"] is None
    assert metrics["rare_class_collapse"]["status"] == "passed"


def test_json_ready_refuses_nonfinite_values() -> None:
    for value in (float("nan"), float("inf"), -float("inf")):
        assert not math.isfinite(value)
        with pytest.raises(FloatingPointError, match="non-finite"):
            core.json_ready({"metric": value})


def test_canonical_config_equality() -> None:
    from cystods.config import load_config, get_stage_trials

    canonical_keys = set(core.PROPOSED_CANONICAL_CONFIG)
    config30 = load_config(stage="30")

    diff30 = {
        k: (config30[k], core.PROPOSED_CANONICAL_CONFIG[k])
        for k in canonical_keys
        if config30[k] != core.PROPOSED_CANONICAL_CONFIG[k]
    }
    assert not diff30, f"Stage 30 Proposed differs from canonical: {diff30}"


def test_stage_40_one_factor_ablations() -> None:
    from cystods.config import get_stage_trials

    canonical_keys = set(core.PROPOSED_CANONICAL_CONFIG)
    full_proposed = dict(core.PROPOSED_CANONICAL_CONFIG)

    expected_diffs = {
        "ablation_full_proposed": set(),
        "ablation_fine_ce": {"fine_loss"},
        "ablation_no_bc_hierarchy": {"binary_coarse_hierarchy_loss_weight"},
        "ablation_no_cf_hierarchy": {"coarse_fine_hierarchy_loss_weight"},
        "ablation_no_supcon": {"supervised_contrastive_loss_weight"},
        "ablation_no_binary_auxiliary": {"binary_loss_weight"},
        "ablation_class_balanced_sampler": {"sampler"},
        "ablation_data_augmentation": {"use_data_augmentation"},
        "ablation_no_prior_calibration": {"fine_inference_calibration_mode"},
    }

    trials = get_stage_trials(stage="40")
    for trial in trials:
        exp_id = trial["experiment_id"]
        if exp_id in expected_diffs:
            trial_config = trial.get("overrides", {})
            actual_diff = {
                k for k in canonical_keys
                if trial_config.get(k, full_proposed.get(k)) != full_proposed.get(k)
            }
            assert actual_diff == expected_diffs[exp_id], (
                f"Trial {exp_id} actual diff {actual_diff} does not match expected {expected_diffs[exp_id]}"
            )


def test_supcon_classification_isolation() -> None:
    config = dict(core.PROPOSED_CANONICAL_CONFIG)
    config.update(
        image_size=64,
        task_mode="hierarchical",
        supervised_contrastive_loss_weight=0.10,
    )
    primary_batch_size = 2

    shared = {
        "features": torch.randn((primary_batch_size * 2, 64)),
        "binary_logits": torch.tensor([[1.0, 0.0], [0.0, 1.0], [99.0, -99.0], [-99.0, 99.0]]),
        "coarse_logits": torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [50.0, 0.0, 0.0, 0.0, 0.0], [0.0, 50.0, 0.0, 0.0, 0.0]]),
        "fine_logits": torch.zeros((primary_batch_size * 2, len(core.FINE_NAMES))),
    }

    binary_targets = torch.tensor([0, 1], dtype=torch.long)
    coarse_targets = torch.tensor([0, 1], dtype=torch.long)
    fine_targets = torch.tensor([0, 1], dtype=torch.long)

    loss_fn = core.FineLongTailLoss(
        "cross_entropy",
        [10] * len(core.FINE_NAMES),
        [10] * len(core.FINE_NAMES),
        config,
    )

    outputs_primary_1 = {k: v[:primary_batch_size] for k, v in shared.items()}
    loss1, _ = core.compute_multitask_loss(
        outputs_primary_1, binary_targets, coarse_targets, fine_targets, loss_fn, config
    )

    shared_modified = dict(shared)
    shared_modified["binary_logits"] = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-999.0, 999.0], [999.0, -999.0]])
    outputs_primary_2 = {k: v[:primary_batch_size] for k, v in shared_modified.items()}
    loss2, _ = core.compute_multitask_loss(
        outputs_primary_2, binary_targets, coarse_targets, fine_targets, loss_fn, config
    )

    assert float(loss1) == pytest.approx(float(loss2))


def test_supcon_transforms_stochasticity() -> None:
    config = dict(core.PROPOSED_CANONICAL_CONFIG)
    config["use_data_augmentation"] = False
    config["supervised_contrastive_loss_weight"] = 0.10

    train_tf, eval_tf, supcon_tf = core.build_transforms(config)

    from PIL import Image
    dummy_img = Image.new("RGB", (224, 224), color=(128, 64, 32))

    t1 = train_tf(dummy_img)
    t2 = eval_tf(dummy_img)
    assert torch.equal(t1, t2)

    stochastic_diff = any(not torch.equal(supcon_tf(dummy_img), supcon_tf(dummy_img)) for _ in range(5))
    assert stochastic_diff


def test_stage10_default_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    from cystods.config import load_config

    monkeypatch.delenv("CYSTODS_BATCH_SIZE", raising=False)
    config = load_config(stage="10")
    assert config["batch_size"] == 32

    monkeypatch.setenv("CYSTODS_BATCH_SIZE", "64")
    config_64 = load_config(stage="10")
    assert config_64["batch_size"] == 64


def test_no_hardcoded_protocol_sha_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    import os
    monkeypatch.delenv("CYSTODS_EXPECTED_PROTOCOL_SHA256", raising=False)

    from cystods.config import load_config
    for stage_id in ("10", "20", "30", "40", "90"):
        config = load_config(stage=stage_id)
        assert os.environ.get("CYSTODS_EXPECTED_PROTOCOL_SHA256") is None


def test_matched_checkpoint_monitor() -> None:
    from cystods.config import get_stage_trials

    trials = get_stage_trials(stage="40")
    for trial in trials:
        overrides = trial.get("overrides", {})
        if "monitor_metric" in overrides:
            assert overrides["monitor_metric"] in (
                "hierarchical_composite",
                "fine_macro_f1",
                "fine_macro_f1_all_classes",
                "coarse_macro_f1",
                "binary_auroc",
            )

    multitask_ref = [t for t in trials if t["experiment_id"] == "task_multitask_bcf"][0]
    assert multitask_ref.get("overrides", {}).get("monitor_metric") == "hierarchical_composite"

    trials90 = get_stage_trials(stage="90")
    for trial in trials90:
        overrides = trial.get("overrides", {})
        assert overrides.get("monitor_metric", "hierarchical_composite") == "hierarchical_composite"


def test_audit_image_size_distribution(tmp_path: Path) -> None:
    from PIL import Image
    import logging

    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # Create dummy images of different sizes
    img1 = Image.new("RGB", (352, 240))
    img1.save(img_dir / "img1.png")

    img2 = Image.new("RGB", (640, 480))
    img2.save(img_dir / "img2.png")

    img3 = Image.new("RGB", (1920, 1080))
    img3.save(img_dir / "img3.png")

    frame = pd.DataFrame([
        {
            "image_stem": "img1",
            "image_path": str(img_dir / "img1.png"),
            "class": "Normal mucosa",
            "subclass": "",
            "subclass2": "",
        },
        {
            "image_stem": "img2",
            "image_path": str(img_dir / "img2.png"),
            "class": "Malignant",
            "subclass": "HighGradePapillary",
            "subclass2": "",
        },
        {
            "image_stem": "img3",
            "image_path": str(img_dir / "img3.png"),
            "class": "Malignant",
            "subclass": "CIS",
            "subclass2": "",
        },
    ])

    run_dir = tmp_path / "run"
    (run_dir / "reports").mkdir(parents=True)
    logger = logging.getLogger("test")

    stats = core.audit_image_size_distribution(frame, run_dir, logger)

    assert "total" in stats
    assert stats["total"]["count"] == 3
    assert stats["total"]["width_min"] == 352
    assert stats["total"]["width_max"] == 1920
    assert stats["total"]["height_min"] == 240
    assert stats["total"]["height_max"] == 1080

    assert "binary_layers" in stats
    assert "ROI" in stats["binary_layers"]
    assert stats["binary_layers"]["ROI"]["count"] == 2
    assert "Non-ROI" in stats["binary_layers"]
    assert stats["binary_layers"]["Non-ROI"]["count"] == 1

    assert (run_dir / "reports" / "image_size_distribution.json").is_file()
    assert (run_dir / "reports" / "image_size_distribution.csv").is_file()


def test_find_latest_completed_protocol_run_fallback(tmp_path: Path) -> None:
    from cystods.data.splits.protocol import find_latest_completed_protocol_run

    result_root = tmp_path / "result"
    proto_dir = result_root / "00_protocol" / "research_20260813-000000"
    proto_dir.mkdir(parents=True)

    (proto_dir / "run_status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    (proto_dir / "protocol_manifest.json").write_text(
        json.dumps({"schema_version": "cystods.protocol.v2", "run_profile": "research"}),
        encoding="utf-8",
    )

    # Looking for 'smoke' should fallback to 'research' protocol run
    found_dir, found_sha = find_latest_completed_protocol_run(result_root, run_profile="smoke")
    assert found_dir == proto_dir
    assert found_sha is not None



