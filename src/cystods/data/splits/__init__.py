"""Data splitting and protocol management for CystoDS."""

from cystods.data.splits.holdout import (
    allocation_score,
    cap_normal_mucosa_across_splits,
    fixed_patient_split,
    materialize_split_frames,
    patient_label_matrices,
    sample_rows_stratified,
    search_holdout_patient_split,
    search_top_k_diverse_holdout_splits,
    validate_materialized_splits,
)
from cystods.data.splits.cross_validation import (
    search_patient_folds,
    search_train_val_patient_split,
)
from cystods.data.splits.protocol import (
    _load_and_validate_protocol_binding,
    build_all_protocol_splits,
    find_latest_completed_protocol_run,
    load_frozen_protocol_splits,
    save_split_artifacts,
)

__all__ = [
    "patient_label_matrices",
    "allocation_score",
    "search_holdout_patient_split",
    "search_top_k_diverse_holdout_splits",
    "search_patient_folds",
    "search_train_val_patient_split",
    "fixed_patient_split",
    "sample_rows_stratified",
    "cap_normal_mucosa_across_splits",
    "materialize_split_frames",
    "validate_materialized_splits",
    "save_split_artifacts",
    "load_frozen_protocol_splits",
    "build_all_protocol_splits",
    "find_latest_completed_protocol_run",
    "_load_and_validate_protocol_binding",
]
