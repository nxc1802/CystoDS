"""Summary report generation, markdown tables, cross-validation aggregation, and artifact manifest.

Extracted from ``cystods.core`` (Step 7 refactor).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cystods.infra.serialization import (
    json_ready,
    sha256_file,
    utc_now_iso,
    write_json,
)


def serialize_prediction_frame(
    frame: pd.DataFrame,
    destination: Path,
    include_features: bool = False,
) -> None:
    output = frame.copy()
    for col in ("binary_probs", "coarse_probs", "fine_probs", "fine_logits"):
        if col in output:
            output[col] = output[col].apply(
                lambda val: (
                    json.dumps(val.tolist())
                    if isinstance(val, np.ndarray)
                    else json.dumps(val)
                )
            )
    if not include_features and "features" in output:
        output = output.drop(columns=["features"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)


def markdown_scalar(value: Any) -> str:
    if value is None:
        return "not evaluable"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def metrics_table_markdown(metrics: Mapping[str, Any]) -> str:
    rows: list[tuple[str, Any]] = []
    if metrics["binary"] is not None:
        rows.extend(
            [
                ("Binary AUROC", metrics["binary"]["auroc"]),
                ("Binary AUPRC", metrics["binary"]["auprc"]),
                ("Binary F1", metrics["binary"]["f1"]),
                ("Binary sensitivity", metrics["binary"]["sensitivity"]),
                ("Binary specificity", metrics["binary"]["specificity"]),
                ("Binary MCC", metrics["binary"]["mcc"]),
            ]
        )
    if metrics["coarse"] is not None:
        rows.extend(
            [
                (
                    "Coarse macro-F1 (supported)",
                    metrics["coarse"]["macro_f1_supported"],
                ),
                (
                    "Coarse macro-F1 (all 5 classes)",
                    metrics["coarse"]["macro_f1_all_classes"],
                ),
                (
                    "Coarse balanced accuracy",
                    metrics["coarse"]["balanced_accuracy"],
                ),
                ("Coarse MCC", metrics["coarse"]["mcc"]),
                (
                    "Coarse macro-AUROC",
                    metrics["coarse"]["macro_auroc_ovr"],
                ),
            ]
        )
    if metrics["fine"] is not None:
        rows.extend(
            [
                (
                    "Fine macro-F1 (supported)",
                    metrics["fine"]["macro_f1_supported"],
                ),
                (
                    "Fine macro-F1 (all 22 classes)",
                    metrics["fine"]["macro_f1_all_classes"],
                ),
            ]
        )
    if (
        metrics["primary_fine"] is not None
        and metrics["primary_fine"]["status"] == "ok"
    ):
        rows.extend(
            [
                (
                    "Primary fine macro-F1 (supported)",
                    metrics["primary_fine"]["macro_f1_supported"],
                ),
                (
                    "Primary fine macro-F1 (fixed denominator)",
                    metrics["primary_fine"]["macro_f1_all_classes"],
                ),
            ]
        )
    if metrics["hierarchy"] is not None:
        rows.extend(
            [
                (
                    "Hierarchical accuracy",
                    metrics["hierarchy"]["hierarchical_accuracy"],
                ),
                (
                    "Cross-parent error rate",
                    metrics["hierarchy"]["cross_parent_error_rate"],
                ),
                (
                    "Tail-class recall",
                    metrics["hierarchy"]["tail_class_recall"],
                ),
            ]
        )
    if not rows:
        raise ValueError("Metrics report contains no active task.")
    return pd.DataFrame(rows, columns=["Metric", "Value"]).assign(
        Value=lambda data: data["Value"].map(markdown_scalar)
    ).to_markdown(index=False)


def build_fold_report(
    fold_name: str,
    config: Mapping[str, Any],
    split_summary: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    wlc_metrics: Mapping[str, Any] | None,
    roi_metrics: Mapping[str, Any] | None,
    paired: Mapping[str, Any] | None,
    external: Mapping[str, Any] | None,
    history: pd.DataFrame,
    output_path: Path,
) -> None:
    if config["inclusion_manifest_csv"] is None:
        inclusion_note = (
            "- No paper inclusion manifest was supplied. The public release "
            "contains 998 malignant images whereas the paper's binary table "
            "uses 994 and does not publish the excluded filenames/PIDs; this "
            "run is therefore paper-like, not claimed as paper-exact."
        )
    else:
        inclusion_note = (
            "- The run was restricted by the explicit inclusion manifest "
            f"`{config['inclusion_manifest_csv']}`."
        )
    lines = [
        f"# CystoDS experiment report - {fold_name}",
        "",
        f"- Generated: {utc_now_iso()}",
        f"- Profile: `{config['run_profile']}`",
        f"- Task mode: `{config['task_mode']}`",
        (
            f"- Encoder: `{config['model_name']}` "
            f"(pretrained={config['pretrained']})"
        ),
        f"- Fine objective: `{config['fine_loss']}`",
        f"- Sampler: `{config['sampler']}`",
        f"- Epochs completed: {len(history)}",
        "",
    ]
    if config["run_profile"] == "smoke":
        lines.extend(
            [
                (
                    "> This is a real-data functional smoke test using three "
                    "patient-disjoint PIDs. Its metrics are not a scientific "
                    "estimate and must not be compared with the paper baseline."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Protocol decisions",
            "",
            (
                "- Binary ROI is derived from `class in {Malignant, "
                "Non-malignant}`; the `bca` column is not used because it "
                "differs for the PreMalignant record."
            ),
            (
                "- The fine head has exactly 22 published subclasses. "
                "`Normal mucosa` has `fine_id=-1` and is masked from fine loss."
            ),
            (
                "- All splits are patient-disjoint. Literal CSV `NA` values "
                "are preserved."
            ),
            (
                "- CSV filename stems are mapped to the dataset's canonical "
                "PNG files. No missing-file or random-weight fallback is used."
            ),
            (
                "- ROI groups with conflicting ground truth are excluded and "
                "reported, or raise when configured."
            ),
            inclusion_note,
            "",
            "## Split",
            "",
            pd.DataFrame(
                [
                    {
                        "Split": name,
                        "Images": values["rows"],
                        "Patients": values["patients"],
                    }
                    for name, values in split_summary["splits"].items()
                ]
            ).to_markdown(index=False),
            "",
            "## Internal test metrics",
            "",
            metrics_table_markdown(test_metrics),
            "",
            "## Patient bootstrap",
            "",
            "```json",
            json.dumps(json_ready(bootstrap), indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    if wlc_metrics is not None:
        lines.extend(
            [
                "## WLC-only test",
                "",
                metrics_table_markdown(wlc_metrics),
                "",
            ]
        )
    if roi_metrics is not None:
        lines.extend(
            [
                "## ROI-level evaluation",
                "",
                "```json",
                json.dumps(json_ready(roi_metrics), indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )
    if paired is not None:
        lines.extend(
            [
                "## Paired significance",
                "",
                "```json",
                json.dumps(json_ready(paired), indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )
    if external is not None:
        lines.extend(
            [
                "## External binary validation",
                "",
                "```json",
                json.dumps(json_ready(external), indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Artifact map",
            "",
            "- `checkpoints/`: best and optional last training state",
            "- `logs/`: console-equivalent training progress and history",
            "- `metrics/`: machine-readable metrics, CIs, WLC and ROI results",
            "- `predictions/`: image-level and ROI-level predictions",
            "- `visualizations/`: curves, confusion matrices and data samples",
            "- `models/`: copied best base model and trained attention MIL models",
            "- `splits/`: exact row and patient manifests",
            "- `source/`: exact pre-notebook and usage guide used by the run",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_artifact_manifest(run_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    child_roots: list[Path] = []
    runs_root = run_dir / "runs"
    if runs_root.is_dir():
        child_roots = sorted(
            path
            for path in runs_root.iterdir()
            if path.is_dir()
            and (path / "artifact_manifest.json").is_file()
        )
    covered_child_files: set[Path] = set()
    for child_root in child_roots:
        child_manifest_path = child_root / "artifact_manifest.json"
        child_rows = json.loads(
            child_manifest_path.read_text(encoding="utf-8")
        )
        if not isinstance(child_rows, list):
            raise TypeError(
                f"Child artifact manifest must be a list: {child_manifest_path}"
            )
        declared_paths: set[Path] = set()
        for child_row in child_rows:
            relative = Path(str(child_row["path"]))
            child_file = child_root / relative
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not child_file.is_file()
            ):
                raise ValueError(
                    f"Invalid child artifact path: {child_row['path']}"
                )
            if int(child_row["bytes"]) != child_file.stat().st_size:
                raise ValueError(
                    f"Child artifact size changed after sealing: {child_file}"
                )
            declared_paths.add(child_file)
            covered_child_files.add(child_file)
            rows.append(
                {
                    "path": str(child_file.relative_to(run_dir)),
                    "bytes": int(child_row["bytes"]),
                    "sha256": str(child_row["sha256"]),
                }
            )
        actual_child_files = {
            path
            for path in child_root.rglob("*")
            if path.is_file() and path != child_manifest_path
        }
        if actual_child_files != declared_paths:
            raise ValueError(
                "Child artifact file set changed after sealing: "
                f"{child_root}"
            )
        rows.append(
            {
                "path": str(child_manifest_path.relative_to(run_dir)),
                "bytes": child_manifest_path.stat().st_size,
                "sha256": sha256_file(child_manifest_path),
            }
        )
        covered_child_files.add(child_manifest_path)
    for path in sorted(run_dir.rglob("*")):
        if (
            not path.is_file()
            or path == run_dir / "artifact_manifest.json"
            or path in covered_child_files
        ):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        rows.append(
            {
                "path": str(path.relative_to(run_dir)),
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    rows.sort(key=lambda row: row["path"])
    write_json(run_dir / "artifact_manifest.json", rows)


def aggregate_cross_validation_metrics(
    fold_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not fold_results:
        raise ValueError("Cannot aggregate zero fold results.")
    selectors = {
        "binary_auroc": lambda value: (
            value["binary"]["auroc"] if value["binary"] is not None else None
        ),
        "binary_auprc": lambda value: (
            value["binary"]["auprc"] if value["binary"] is not None else None
        ),
        "binary_f1": lambda value: (
            value["binary"]["f1"] if value["binary"] is not None else None
        ),
        "coarse_macro_f1_supported": lambda value: (
            value["coarse"]["macro_f1_supported"]
            if value["coarse"] is not None
            else None
        ),
        "coarse_macro_f1_all_classes": lambda value: (
            value["coarse"]["macro_f1_all_classes"]
            if value["coarse"] is not None
            else None
        ),
        "coarse_balanced_accuracy": lambda value: (
            value["coarse"]["balanced_accuracy"]
            if value["coarse"] is not None
            else None
        ),
        "coarse_mcc": lambda value: (
            value["coarse"]["mcc"] if value["coarse"] is not None else None
        ),
        "fine_macro_f1_supported": lambda value: (
            value["fine"]["macro_f1_supported"]
            if value["fine"] is not None
            else None
        ),
        "fine_macro_f1_all_classes": lambda value: (
            value["fine"]["macro_f1_all_classes"]
            if value["fine"] is not None
            else None
        ),
        "primary_macro_f1_all_classes": lambda value: (
            value["primary_fine"]["macro_f1_all_classes"]
            if value["primary_fine"] is not None
            and value["primary_fine"]["status"] == "ok"
            else None
        ),
        "hierarchical_accuracy": lambda value: (
            value["hierarchy"]["hierarchical_accuracy"]
            if value["hierarchy"] is not None
            else None
        ),
        "cross_parent_error_rate": lambda value: (
            value["hierarchy"]["cross_parent_error_rate"]
            if value["hierarchy"] is not None
            else None
        ),
    }
    summary: dict[str, Any] = {"num_folds": len(fold_results), "metrics": {}}
    for name, extractor in selectors.items():
        extracted = [
            extractor(result["test_metrics"]) for result in fold_results
        ]
        valid_values = [
            float(value)
            for value in extracted
            if value is not None and np.isfinite(float(value))
        ]
        if valid_values:
            summary["metrics"][name] = {
                "mean": float(np.mean(valid_values)),
                "std": (
                    float(np.std(valid_values, ddof=1))
                    if len(valid_values) > 1
                    else 0.0
                ),
                "min": float(np.min(valid_values)),
                "max": float(np.max(valid_values)),
                "valid_folds": len(valid_values),
            }
        else:
            summary["metrics"][name] = {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "valid_folds": 0,
            }
    return summary
