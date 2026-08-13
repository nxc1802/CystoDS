"""Patient label matrices, allocation scoring, holdout search, and split materialization.

Extracted from ``cystods.core`` (Step 4 refactor).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from cystods.infra.serialization import stable_int_seed
from cystods.taxonomy import COARSE_NAMES, FINE_NAMES


def patient_label_matrices(
    frame: pd.DataFrame,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pids = sorted(frame["pid"].unique().tolist())
    coarse_presence = np.zeros((len(pids), len(COARSE_NAMES)), dtype=np.float64)
    fine_presence = np.zeros((len(pids), len(FINE_NAMES)), dtype=np.float64)
    coarse_image_counts = np.zeros((len(pids), len(COARSE_NAMES)), dtype=np.float64)
    fine_image_counts = np.zeros((len(pids), len(FINE_NAMES)), dtype=np.float64)
    pid_to_index = {pid: index for index, pid in enumerate(pids)}
    has_coarse = "coarse_id" in frame.columns
    has_fine = "fine_id" in frame.columns
    for row in frame.itertuples(index=False):
        index = pid_to_index[str(row.pid)]
        if has_coarse:
            c_id = int(getattr(row, "coarse_id"))
            coarse_presence[index, c_id] = 1.0
            coarse_image_counts[index, c_id] += 1.0
        if has_fine:
            f_id = int(getattr(row, "fine_id"))
            if f_id >= 0:
                fine_presence[index, f_id] = 1.0
                fine_image_counts[index, f_id] += 1.0
    return pids, coarse_presence, fine_presence, coarse_image_counts, fine_image_counts


def allocation_score(
    assignments: Sequence[set[str]],
    pid_to_row: Mapping[str, int],
    coarse_presence: np.ndarray,
    fine_presence: np.ndarray,
    image_counts: np.ndarray,
    target_fractions: Sequence[float],
    fine_image_counts: np.ndarray | None = None,
) -> float:
    global_coarse_presence = coarse_presence.sum(axis=0)
    global_fine_presence = fine_presence.sum(axis=0)
    global_images = image_counts.sum(axis=0)
    global_fine_images = (
        fine_image_counts.sum(axis=0) if fine_image_counts is not None else None
    )
    score = 0.0
    for members, fraction in zip(assignments, target_fractions):
        indices = [pid_to_row[pid] for pid in members]
        if not indices:
            return float("inf")
        local_coarse = coarse_presence[indices].sum(axis=0)
        if np.any(local_coarse == 0):
            return float("inf")
        local_fine = fine_presence[indices].sum(axis=0)
        local_images = image_counts[indices].sum(axis=0)
        score += float(
            np.mean(
                np.abs(
                    local_coarse
                    - global_coarse_presence * fraction
                )
                / np.maximum(global_coarse_presence * fraction, 1.0)
            )
        )
        eligible_fine = global_fine_presence >= len(assignments)
        if eligible_fine.any():
            score += float(
                np.mean(
                    np.abs(
                        local_fine[eligible_fine]
                        - global_fine_presence[eligible_fine] * fraction
                    )
                    / np.maximum(
                        global_fine_presence[eligible_fine] * fraction,
                        1.0,
                    )
                )
            )
        score += float(
            np.mean(
                np.abs(local_images - global_images * fraction)
                / np.maximum(global_images * fraction, 1.0)
            )
        )
        if fine_image_counts is not None and global_fine_images is not None:
            local_fine_images = fine_image_counts[indices].sum(axis=0)
            eligible_fine_img = global_fine_images >= len(assignments)
            if eligible_fine_img.any():
                score += float(
                    np.mean(
                        np.abs(
                            local_fine_images[eligible_fine_img]
                            - global_fine_images[eligible_fine_img] * fraction
                        )
                        / np.maximum(
                            global_fine_images[eligible_fine_img] * fraction,
                            1.0,
                        )
                    )
                )
    return score


def search_top_k_diverse_holdout_splits(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    seed: int,
    k: int = 3,
    max_test_overlap_fraction: float = 0.5,
    allowed_pids: set[str] | None = None,
    forced_test_pids: set[str] | None = None,
) -> list[tuple[dict[str, set[str]], float]]:
    (
        all_pids,
        coarse_presence,
        fine_presence,
        image_counts,
        fine_image_counts,
    ) = patient_label_matrices(frame)
    pid_to_row = {pid: index for index, pid in enumerate(all_pids)}
    eligible = set(all_pids) if allowed_pids is None else set(allowed_pids)
    if not eligible:
        raise ValueError("No patients are eligible for split construction.")
    unknown = eligible - set(all_pids)
    if unknown:
        raise ValueError(f"Unknown allowed patient IDs: {sorted(unknown)}")

    fractions = np.asarray(
        [
            config["train_fraction"],
            config["val_fraction"],
            config["test_fraction"],
        ],
        dtype=np.float64,
    )
    fractions = fractions / fractions.sum()
    target_counts = np.floor(fractions * len(eligible)).astype(int)
    target_counts[0] += len(eligible) - int(target_counts.sum())
    if np.any(target_counts < 1):
        raise ValueError(
            f"Not enough patients for requested split: {target_counts.tolist()}"
        )

    forced_train: set[str] = set()
    threshold = int(
        config["force_fine_labels_with_fewer_than_n_patients_to_train"]
    )
    if threshold > 0:
        eligible_frame = frame[frame["pid"].isin(eligible)]
        fine_patient_support = (
            eligible_frame.loc[eligible_frame["fine_id"] >= 0]
            .groupby("fine_id")["pid"]
            .nunique()
        )
        rare_ids = set(
            fine_patient_support[
                fine_patient_support < threshold
            ].index.astype(int)
        )
        forced_train = set(
            eligible_frame.loc[
                eligible_frame["fine_id"].isin(rare_ids), "pid"
            ]
        )
    if forced_test_pids:
        forced_test = set(forced_test_pids)
        if not forced_test <= eligible:
            raise ValueError("forced_test_pids must be a subset of allowed_pids.")
        forced_train -= forced_test
    else:
        forced_test = set()
    if len(forced_train) > target_counts[0]:
        raise ValueError(
            "Rare-label constraints require more training patients than the "
            "requested training split can contain."
        )
    if len(forced_test) > target_counts[2]:
        raise ValueError("forced_test_pids exceed the requested test size.")

    remaining = sorted(eligible - forced_train - forced_test)
    rng = np.random.default_rng(seed)
    candidates: list[tuple[float, dict[str, set[str]]]] = []

    for _ in range(int(config["split_search_candidates"])):
        permutation = rng.permutation(remaining).tolist()
        train_needed = int(target_counts[0]) - len(forced_train)
        val_needed = int(target_counts[1])
        test_needed = int(target_counts[2]) - len(forced_test)
        if train_needed + val_needed + test_needed != len(permutation):
            raise RuntimeError("Internal patient allocation count mismatch.")
        train = forced_train | set(permutation[:train_needed])
        val_start = train_needed
        val = set(permutation[val_start : val_start + val_needed])
        test = forced_test | set(permutation[val_start + val_needed :])
        assignments = (train, val, test)
        if config["run_profile"] == "research":
            train_indices = [pid_to_row[pid] for pid in train]
            if np.any(fine_presence[train_indices].sum(axis=0) == 0):
                continue
        score = allocation_score(
            assignments,
            pid_to_row,
            coarse_presence,
            fine_presence,
            image_counts,
            fractions,
            fine_image_counts=fine_image_counts,
        )
        if not math.isfinite(score):
            continue
        candidates.append((score, {"train": train, "val": val, "test": test}))

    if not candidates:
        raise RuntimeError(
            "No valid patient-disjoint split was found. Increase "
            "split_search_candidates or revise split constraints."
        )

    candidates.sort(key=lambda item: item[0])

    # Select top k with pairwise test overlap <= max_test_overlap_fraction
    max_test_overlap = math.floor(target_counts[2] * max_test_overlap_fraction)
    selected: list[tuple[dict[str, set[str]], float]] = []

    # Always take the best candidate as split 0
    best_score, best_split = candidates[0]
    selected.append((best_split, best_score))

    for score, split in candidates[1:]:
        if len(selected) >= k:
            break
        # Check pairwise test overlap with all already selected splits
        test_pids = split["test"]
        overlap_ok = True
        for sel_split, _ in selected:
            if len(test_pids & sel_split["test"]) > max_test_overlap:
                overlap_ok = False
                break
        if overlap_ok:
            selected.append((split, score))

    # If strict constraint didn't yield k candidates (e.g. tiny candidate pool in smoke profile),
    # fill with next best available candidates
    if len(selected) < k:
        for score, split in candidates:
            if len(selected) >= k:
                break
            if not any(split is s[0] for s in selected):
                selected.append((split, score))

    return selected


def search_holdout_patient_split(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    seed: int,
    allowed_pids: set[str] | None = None,
    forced_test_pids: set[str] | None = None,
) -> tuple[dict[str, set[str]], float]:
    splits = search_top_k_diverse_holdout_splits(
        frame=frame,
        config=config,
        seed=seed,
        k=1,
        allowed_pids=allowed_pids,
        forced_test_pids=forced_test_pids,
    )
    return splits[0][0], splits[0][1]


def fixed_patient_split(
    frame: pd.DataFrame,
    fixed: Mapping[str, Sequence[str]],
) -> dict[str, set[str]]:
    required = {"train", "val", "test"}
    if set(fixed) != required:
        raise ValueError(
            f"fixed_split_pids must have exactly {sorted(required)} keys."
        )
    split = {key: set(map(str, fixed[key])) for key in required}
    if any(not values for values in split.values()):
        raise ValueError("Every fixed split must contain at least one patient.")
    if (
        split["train"] & split["val"]
        or split["train"] & split["test"]
        or split["val"] & split["test"]
    ):
        raise ValueError("Fixed patient splits overlap.")
    known = set(frame["pid"])
    unknown = set.union(*split.values()) - known
    if unknown:
        raise ValueError(f"Fixed split contains unknown PIDs: {sorted(unknown)}")
    return split


def sample_rows_stratified(
    frame: pd.DataFrame,
    limit: int | None,
    seed: int,
) -> pd.DataFrame:
    if limit is None or len(frame) <= int(limit):
        return frame.copy()
    limit = int(limit)
    if limit < frame["coarse_id"].nunique():
        raise ValueError(
            "A sample cap cannot be smaller than the number of coarse classes."
        )
    rng = np.random.default_rng(seed)
    groups = {
        int(label): group.index.to_numpy()
        for label, group in frame.groupby("coarse_id")
    }
    allocation = {
        label: max(1, round(limit * len(indices) / len(frame)))
        for label, indices in groups.items()
    }
    while sum(allocation.values()) > limit:
        candidates = [
            label for label, count in allocation.items() if count > 1
        ]
        if not candidates:
            raise RuntimeError("Unable to reduce stratified allocation.")
        label = max(candidates, key=lambda item: allocation[item])
        allocation[label] -= 1
    while sum(allocation.values()) < limit:
        candidates = [
            label
            for label, indices in groups.items()
            if allocation[label] < len(indices)
        ]
        if not candidates:
            raise RuntimeError("Unable to increase stratified allocation.")
        label = max(
            candidates,
            key=lambda item: len(groups[item]) - allocation[item],
        )
        allocation[label] += 1
    selected: list[int] = []
    for label, indices in groups.items():
        count = min(allocation[label], len(indices))
        selected.extend(rng.choice(indices, size=count, replace=False).tolist())
    if len(selected) != limit:
        raise RuntimeError("Stratified sample size does not match the limit.")
    return frame.loc[sorted(selected)].copy()


def cap_normal_mucosa_across_splits(
    split_frames: Mapping[str, pd.DataFrame],
    total_limit: int | None,
    fractions: Mapping[str, float],
    seed: int,
) -> dict[str, pd.DataFrame]:
    if total_limit is None:
        return {name: frame.copy() for name, frame in split_frames.items()}
    total_limit = int(total_limit)
    if total_limit < 0:
        raise ValueError("normal_mucosa_limit cannot be negative.")
    desired = {
        name: math.floor(total_limit * fractions[name])
        for name in split_frames
    }
    remainder = total_limit - sum(desired.values())
    for name in ("train", "val", "test"):
        if remainder <= 0:
            break
        desired[name] += 1
        remainder -= 1

    output: dict[str, pd.DataFrame] = {}
    from cystods.taxonomy import COARSE_TO_ID
    normal_id = COARSE_TO_ID["Normal mucosa"]
    for name, frame in split_frames.items():
        normal = frame[frame["coarse_id"] == normal_id]
        other = frame[frame["coarse_id"] != normal_id]
        keep = min(desired[name], len(normal))
        if keep:
            normal = normal.sample(
                n=keep,
                random_state=stable_int_seed(seed, name, "normal"),
                replace=False,
            )
        else:
            normal = normal.iloc[0:0]
        output[name] = (
            pd.concat([other, normal], ignore_index=True)
            .sample(
                frac=1.0,
                random_state=stable_int_seed(seed, name, "shuffle"),
            )
            .reset_index(drop=True)
        )
    return output


def materialize_split_frames(
    frame: pd.DataFrame,
    patient_split: Mapping[str, set[str]],
    config: Mapping[str, Any],
    seed: int,
) -> dict[str, pd.DataFrame]:
    train_pids = patient_split["train"]
    val_pids = patient_split["val"]
    test_pids = patient_split["test"]
    if train_pids & val_pids or train_pids & test_pids or val_pids & test_pids:
        raise RuntimeError("Patient leakage detected before split materialization.")

    frames = {
        name: frame[frame["pid"].isin(pids)].copy().reset_index(drop=True)
        for name, pids in patient_split.items()
    }
    if any(part.empty for part in frames.values()):
        raise ValueError("Every materialized split must contain at least one row.")

    fractions = {
        name: len(pids) / sum(len(value) for value in patient_split.values())
        for name, pids in patient_split.items()
    }
    frames = cap_normal_mucosa_across_splits(
        frames,
        config["normal_mucosa_limit"],
        fractions,
        seed,
    )
    limits = {
        "train": config["max_train_samples"],
        "val": config["max_val_samples"],
        "test": config["max_test_samples"],
    }
    for name in frames:
        frames[name] = sample_rows_stratified(
            frames[name],
            limits[name],
            stable_int_seed(seed, name, "cap"),
        ).reset_index(drop=True)
    return frames


def validate_materialized_splits(
    split_frames: Mapping[str, pd.DataFrame],
    run_profile: str,
) -> None:
    pid_sets = {
        name: set(frame["pid"]) for name, frame in split_frames.items()
    }
    if (
        pid_sets["train"] & pid_sets["val"]
        or pid_sets["train"] & pid_sets["test"]
        or pid_sets["val"] & pid_sets["test"]
    ):
        raise RuntimeError("Patient leakage detected after row sampling.")
    for name, frame in split_frames.items():
        observed = set(frame["coarse_id"].astype(int))
        if observed != set(range(len(COARSE_NAMES))):
            raise ValueError(
                f"{name} lacks coarse classes after sampling: "
                f"{set(range(len(COARSE_NAMES))) - observed}"
            )
    if run_profile == "research":
        train_fine = set(
            split_frames["train"].loc[
                split_frames["train"]["fine_id"] >= 0, "fine_id"
            ].astype(int)
        )
        if len(train_fine) != len(FINE_NAMES):
            missing = [FINE_NAMES[index] for index in set(range(22)) - train_fine]
            raise ValueError(
                "Research holdout training split lacks fine labels: "
                f"{missing}. Increase split search or adjust rare constraints."
            )
