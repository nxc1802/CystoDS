"""DataLoader construction and sample weighting for balanced training.

Extracted from ``cystods.core`` (Step 3 refactor).
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from cystods.data.dataset import CystoDataset
from cystods.data.transforms import build_transforms
from cystods.infra.serialization import stable_int_seed
from cystods.losses.classification import effective_number_weights
from cystods.taxonomy import COARSE_NAMES, FINE_NAMES


class _WorkerInitFn:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int) -> None:
        worker_seed = (self.seed + worker_id) % (2**32 - 1)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)


def make_worker_init_fn(seed: int):
    return _WorkerInitFn(seed)


def build_sample_weights(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> torch.DoubleTensor:
    level = config["sampler_label_level"]
    if level == "coarse":
        labels = frame["coarse_id"].astype(int).to_numpy()
        counts = np.bincount(labels, minlength=len(COARSE_NAMES))
        weights = effective_number_weights(
            counts,
            float(config["class_balance_beta"]),
            require_positive=True,
        )
        return torch.as_tensor(weights[labels], dtype=torch.double)
    if level == "fine":
        labels = frame["fine_id"].astype(int).to_numpy()
        valid = labels >= 0
        if not valid.any():
            raise ValueError("No fine-labelled samples exist for the sampler.")
        counts = np.bincount(labels[valid], minlength=len(FINE_NAMES))
        fine_weights = effective_number_weights(
            counts,
            float(config["class_balance_beta"]),
            require_positive=False,
        )
        # Normal mucosa has no fine target. Give it the median positive sample
        # weight explicitly rather than inventing a 23rd fine class.
        positive_weights = fine_weights[fine_weights > 0]
        if not len(positive_weights):
            raise ValueError("Fine sampler weights contain no positive value.")
        sample_weights = np.full(
            len(labels),
            float(np.median(positive_weights)),
            dtype=np.float64,
        )
        sample_weights[valid] = fine_weights[labels[valid]]
        return torch.as_tensor(sample_weights, dtype=torch.double)
    raise ValueError("sampler_label_level must be fine or coarse.")


def build_dataloaders(
    split_frames: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[dict[str, DataLoader], dict[str, CystoDataset]]:
    train_transform, eval_transform, supcon_transform = build_transforms(config)
    supcon_weight = float(config.get("supervised_contrastive_loss_weight", 0.0))
    second_view_transform = supcon_transform if supcon_weight > 0 else None
    datasets = {
        "train": CystoDataset(
            split_frames["train"],
            train_transform,
            second_view_transform=second_view_transform,
        ),
        "val": CystoDataset(
            split_frames["val"],
            eval_transform,
            second_view_transform=None,
        ),
    }
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = None
    shuffle = True
    if config["sampler"] == "class_balanced":
        weights = build_sample_weights(split_frames["train"], config)
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )
        shuffle = False

    train_workers = (
        0 if device.type == "mps" else int(config["num_workers"])
    )
    eval_workers = (
        0 if device.type == "mps" else int(config["eval_num_workers"])
    )
    train_kwargs: dict[str, Any] = {
        "batch_size": int(config["batch_size"]),
        "num_workers": train_workers,
        "pin_memory": bool(config.get("pin_memory", True) and device.type == "cuda"),
        "persistent_workers": bool(
            config["persistent_workers"] and train_workers > 0
        ),
        "worker_init_fn": make_worker_init_fn(seed),
        "generator": generator,
    }
    if train_workers > 0:
        train_kwargs["prefetch_factor"] = int(config["prefetch_factor"])
    eval_kwargs: dict[str, Any] = {
        "batch_size": int(config["eval_batch_size"]),
        "num_workers": eval_workers,
        "pin_memory": bool(
            config.get("pin_memory", True) and device.type == "cuda"
        ),
        # Validation is visited once per epoch. Keeping a second persistent
        # worker pool alive wastes host RAM and shared memory on large batches.
        "persistent_workers": False,
        "worker_init_fn": make_worker_init_fn(stable_int_seed(seed, "val")),
    }
    if eval_workers > 0:
        eval_kwargs["prefetch_factor"] = int(
            config["eval_prefetch_factor"]
        )
    loaders = {
        "train": DataLoader(
            datasets["train"],
            shuffle=shuffle,
            sampler=sampler,
            drop_last=False,
            **train_kwargs,
        ),
        "val": DataLoader(
            datasets["val"],
            shuffle=False,
            drop_last=False,
            **eval_kwargs,
        ),
    }
    if not loaders["train"]:
        raise ValueError("Training DataLoader contains no batches.")
    return loaders, datasets
