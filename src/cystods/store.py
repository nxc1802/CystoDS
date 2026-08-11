"""ResultStore — Centralized result directory management and artifact layout.

Enforces unified Stage -> Trial -> Seed -> Fold folder structure.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def write_json_file(path: Path, payload: Any, indent: int = 2) -> Path:
    """Write payload to JSON file, creating parent directories lazily."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)
    return path


def read_json_file(path: Path) -> Any:
    """Read JSON file content."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class ResultStore:
    """Centralized manager for result folder paths and artifact consolidation."""

    root_dir: Path

    @classmethod
    def from_path(cls, path: Path | str) -> ResultStore:
        return cls(root_dir=Path(path).resolve())

    def trial_dir(self, trial_id: str) -> Path:
        """Directory for a specific trial within a stage suite run."""
        return self.root_dir / "runs" / trial_id

    def seed_dir(self, trial_id: str, seed: int) -> Path:
        """Directory for a specific trial and seed."""
        return self.root_dir / "runs" / trial_id / f"seed_{seed}"

    def fold_dir(
        self,
        fold_name: str,
        trial_id: str | None = None,
        seed: int | None = None,
    ) -> Path:
        """Directory for a single fold run's evidence bundle."""
        if trial_id is not None and seed is not None:
            return self.root_dir / "runs" / trial_id / f"seed_{seed}" / "folds" / fold_name
        elif trial_id is not None:
            return self.root_dir / "runs" / trial_id / "folds" / fold_name
        else:
            return self.root_dir / "folds" / fold_name

    def write_fold_metrics(
        self,
        fold_dir: Path,
        train_metrics: dict[str, Any] | None = None,
        val_metrics: dict[str, Any] | None = None,
        test_metrics: dict[str, Any] | None = None,
        train_losses: dict[str, Any] | None = None,
        val_losses: dict[str, Any] | None = None,
        test_losses: dict[str, Any] | None = None,
        bootstrap: dict[str, Any] | None = None,
        performance: dict[str, Any] | None = None,
        calibration: dict[str, Any] | None = None,
        rare_class: dict[str, Any] | None = None,
        wlc_only: dict[str, Any] | None = None,
        roi: dict[str, Any] | None = None,
        paired_test: dict[str, Any] | None = None,
    ) -> Path:
        """Consolidate all fold metrics into a single structured metrics.json."""
        consolidated = {
            "schema_version": "cystods.fold_metrics",
            "train": {
                "metrics": train_metrics or {},
                "losses": train_losses or {},
            },
            "validation": {
                "metrics": val_metrics or {},
                "losses": val_losses or {},
            },
            "test": {
                "metrics": test_metrics or {},
                "losses": test_losses or {},
            },
            "bootstrap": bootstrap or {},
            "performance": performance or {},
            "calibration": calibration or {},
            "rare_class": rare_class or {},
            "wlc_only": wlc_only,
            "roi": roi,
            "paired_test": paired_test,
        }
        return write_json_file(fold_dir / "metrics.json", consolidated)

    def write_summary(
        self,
        target_dir: Path,
        identity: dict[str, Any],
        method: dict[str, Any] | None = None,
        protocol: dict[str, Any] | None = None,
        training: dict[str, Any] | None = None,
        headline_metrics: dict[str, Any] | None = None,
        scientific_gates: dict[str, Any] | None = None,
        checkpoint: dict[str, Any] | None = None,
    ) -> Path:
        """Write a standardized summary.json for machine/LLM ingestion."""
        summary = {
            "schema_version": "cystods.result_summary",
            "identity": identity,
            "method": method or {},
            "protocol": protocol or {},
            "training": training or {},
            "headline_metrics": headline_metrics or {},
            "scientific_gates": scientific_gates or {},
            "checkpoint": checkpoint or {},
        }
        return write_json_file(target_dir / "summary.json", summary)

    def generate_catalog(self) -> Path:
        """Generate semantic catalog.json for the result directory."""
        catalog: dict[str, str] = {}
        root = self.root_dir

        # Search for metrics and evidence
        for metrics_file in root.glob("**/metrics.json"):
            rel_path = str(metrics_file.relative_to(root))
            if "folds/" in rel_path:
                parts = rel_path.split("folds/")
                fold = parts[1].split("/")[0]
                catalog[f"fold_{fold}_metrics"] = rel_path

        for summary_file in root.glob("**/summary.json"):
            rel_path = str(summary_file.relative_to(root))
            catalog[f"summary_{summary_file.parent.name}"] = rel_path

        for pred_file in root.glob("**/predictions.csv"):
            rel_path = str(pred_file.relative_to(root))
            catalog[f"predictions_{pred_file.parent.name}"] = rel_path

        catalog_path = root / "catalog.json"
        return write_json_file(catalog_path, catalog)
