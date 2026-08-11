"""Cross-validation fold search and inner train/validation allocation.

Extracted from ``cystods.core`` (Step 4 refactor).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from cystods.data.splits.holdout import allocation_score, patient_label_matrices


def search_patient_folds(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    seed: int,
) -> tuple[list[set[str]], float]:
    (
        pids,
        coarse_presence,
        fine_presence,
        image_counts,
    ) = patient_label_matrices(frame)
    n_folds = int(config["cv_folds"])
    if n_folds < 2 or n_folds > len(pids):
        raise ValueError("cv_folds must be between 2 and the patient count.")
    pid_to_row = {pid: index for index, pid in enumerate(pids)}
    fold_sizes = [
        len(part) for part in np.array_split(np.arange(len(pids)), n_folds)
    ]
    fractions = [size / len(pids) for size in fold_sizes]
    rng = np.random.default_rng(seed)
    best: tuple[float, list[set[str]]] | None = None
    for _ in range(int(config["split_search_candidates"])):
        permutation = rng.permutation(pids).tolist()
        folds: list[set[str]] = []
        cursor = 0
        for size in fold_sizes:
            folds.append(set(permutation[cursor : cursor + size]))
            cursor += size
        score = allocation_score(
            folds,
            pid_to_row,
            coarse_presence,
            fine_presence,
            image_counts,
            fractions,
        )
        if not math.isfinite(score):
            continue
        if best is None or score < best[0]:
            best = (score, folds)
    if best is None:
        raise RuntimeError(
            "No valid multilabel patient-fold allocation was found."
        )
    return best[1], best[0]


def search_train_val_patient_split(
    frame: pd.DataFrame,
    allowed_pids: set[str],
    val_fraction: float,
    candidates: int,
    seed: int,
) -> tuple[set[str], set[str], float]:
    if not 0 < val_fraction < 1:
        raise ValueError("CV validation fraction must be in (0, 1).")
    (
        all_pids,
        coarse_presence,
        fine_presence,
        image_counts,
    ) = patient_label_matrices(frame)
    known = set(all_pids)
    if not allowed_pids <= known:
        raise ValueError("CV train/validation pool contains unknown patients.")
    train_count = round(len(allowed_pids) * (1.0 - val_fraction))
    train_count = min(max(train_count, 1), len(allowed_pids) - 1)
    pid_to_row = {pid: index for index, pid in enumerate(all_pids)}
    rng = np.random.default_rng(seed)
    ordered = sorted(allowed_pids)
    best: tuple[float, set[str], set[str]] | None = None
    fractions = (
        train_count / len(allowed_pids),
        1.0 - train_count / len(allowed_pids),
    )
    for _ in range(int(candidates)):
        permutation = rng.permutation(ordered).tolist()
        train = set(permutation[:train_count])
        val = set(permutation[train_count:])
        score = allocation_score(
            (train, val),
            pid_to_row,
            coarse_presence,
            fine_presence,
            image_counts,
            fractions,
        )
        if not math.isfinite(score):
            continue
        if best is None or score < best[0]:
            best = (score, train, val)
    if best is None:
        raise RuntimeError("No valid CV train/validation allocation was found.")
    return best[1], best[2], best[0]
