"""Dataset classes for internal CystoDS and external validation sets.

Extracted from ``cystods.core`` (Step 3 refactor).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CystoDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        transform: Any,
        second_view_transform: Any | None = None,
        *,
        second_view: bool | None = None,
    ) -> None:
        if frame.empty:
            raise ValueError("CystoDataset cannot be constructed from no rows.")
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = transform
        if second_view_transform is not None:
            self.second_view_transform = second_view_transform
        elif second_view:
            self.second_view_transform = transform
        else:
            self.second_view_transform = None
        self.second_view = self.second_view_transform is not None
        # Materialize hot-path columns once. Repeated pandas.iloc calls inside
        # DataLoader workers are measurably slower and increase serialization
        # overhead, especially with persistent workers.
        # Materialize hot-path columns once. Safe fallback for mock DataFrames in tests.
        n = len(self.frame)
        self.image_paths = (
            self.frame["image_path"].astype(str).tolist()
            if "image_path" in self.frame
            else [""] * n
        )
        self.binary_ids = (
            self.frame["binary_id"].astype(int).to_numpy()
            if "binary_id" in self.frame
            else np.zeros(n, dtype=int)
        )
        self.coarse_ids = (
            self.frame["coarse_id"].astype(int).to_numpy()
            if "coarse_id" in self.frame
            else np.zeros(n, dtype=int)
        )
        self.fine_ids = (
            self.frame["fine_id"].astype(int).to_numpy()
            if "fine_id" in self.frame
            else np.full(n, -1, dtype=int)
        )
        self.filenames = (
            self.frame["filename"].astype(str).tolist()
            if "filename" in self.frame
            else [""] * n
        )
        self.pids = (
            self.frame["pid"].astype(str).tolist()
            if "pid" in self.frame
            else [""] * n
        )
        self.visits = (
            self.frame["visit"].astype(str).tolist()
            if "visit" in self.frame
            else [""] * n
        )
        self.lesions = (
            self.frame["lesion"].astype(str).tolist()
            if "lesion" in self.frame
            else [""] * n
        )
        self.modalities = (
            self.frame["modality"].astype(str).tolist()
            if "modality" in self.frame
            else ["WLC"] * n
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = Path(self.image_paths[index])
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
                view_one = self.transform(image)
                view_two = (
                    self.second_view_transform(image)
                    if self.second_view_transform is not None
                    else None
                )
        except Exception as exc:
            raise RuntimeError(f"Failed to decode/transform image: {path}") from exc
        item: dict[str, Any] = {
            "image": view_one,
            "binary_id": torch.tensor(
                self.binary_ids[index], dtype=torch.long
            ),
            "coarse_id": torch.tensor(
                self.coarse_ids[index], dtype=torch.long
            ),
            "fine_id": torch.tensor(self.fine_ids[index], dtype=torch.long),
            "row_index": torch.tensor(index, dtype=torch.long),
            "filename": self.filenames[index],
            "pid": self.pids[index],
            "visit": self.visits[index],
            "lesion": self.lesions[index],
            "modality": self.modalities[index],
        }
        if view_two is not None:
            item["image_view_2"] = view_two
        return item


class ExternalBinaryDataset(Dataset):
    def __init__(
        self,
        manifest_csv: Path,
        image_root: Path,
        transform: Any,
        config: Mapping[str, Any],
    ) -> None:
        if not manifest_csv.is_file():
            raise FileNotFoundError(
                f"External manifest not found: {manifest_csv}"
            )
        if not image_root.is_dir():
            raise FileNotFoundError(
                f"External image root not found: {image_root}"
            )
        frame = pd.read_csv(
            manifest_csv, dtype=str, keep_default_na=False
        )
        path_col = str(config["external_path_column"])
        label_col = str(config["external_binary_label_column"])
        pid_col = str(config["external_patient_id_column"])
        required = {path_col, label_col, pid_col}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"External manifest columns missing: {sorted(missing)}"
            )
        labels = pd.to_numeric(frame[label_col], errors="raise").astype(int)
        if not set(labels).issubset({0, 1}):
            raise ValueError("External binary labels must be 0 or 1.")
        frame = frame.copy()
        frame["resolved_path"] = frame[path_col].map(
            lambda value: str(image_root / value)
        )
        missing_files = [
            path
            for path in frame["resolved_path"]
            if not Path(path).is_file()
        ]
        if missing_files:
            raise FileNotFoundError(
                f"External images missing; first={missing_files[0]}"
            )
        frame["binary_id"] = labels
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.path_col = path_col
        self.pid_col = pid_col
        self.paths = self.frame["resolved_path"].astype(str).tolist()
        self.labels = self.frame["binary_id"].astype(int).to_numpy()
        self.filenames = self.frame[path_col].astype(str).tolist()
        self.pids = self.frame[pid_col].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = Path(self.paths[index])
        try:
            with Image.open(path) as source:
                image = self.transform(source.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"External image decode failed: {path}") from exc
        return {
            "image": image,
            "binary_id": torch.tensor(
                self.labels[index], dtype=torch.long
            ),
            "filename": self.filenames[index],
            "pid": self.pids[index],
        }
