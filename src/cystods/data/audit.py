"""Image dimension and aspect ratio distribution auditing.

Extracted from ``cystods.core`` (Step 3 refactor).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from cystods.infra.environment import is_missing_token
from cystods.infra.serialization import write_json
from cystods.taxonomy import COARSE_NAMES


def audit_image_size_distribution(
    frame: pd.DataFrame,
    run_dir: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Audit image dimensions (width, height, aspect ratio, megapixels) across total dataset and per-class for binary, coarse, and fine layers."""
    logger.info("Auditing image size distribution across %d images...", len(frame))
    records: list[dict[str, Any]] = []
    for row_dict in frame.to_dict(orient="records"):
        image_path = Path(row_dict["image_path"])
        with Image.open(image_path) as img:
            width, height = img.size
        aspect_ratio = round(width / height, 3)
        mp = round((width * height) / 1e6, 3)

        subclass_str = str(row_dict.get("subclass", "")).strip()
        subclass2_str = str(row_dict.get("subclass2", "")).strip()
        coarse_class = str(row_dict.get("class", "")).strip()

        if coarse_class == "Normal mucosa":
            fine_label = "Normal mucosa"
        elif subclass_str and not is_missing_token(subclass_str):
            fine_label = subclass_str
        elif subclass2_str and not is_missing_token(subclass2_str):
            fine_label = subclass2_str
        else:
            fine_label = coarse_class

        binary_label = "ROI" if coarse_class in ("Malignant", "Non-malignant") else "Non-ROI"

        records.append(
            {
                "image_stem": str(row_dict["image_stem"]),
                "width": int(width),
                "height": int(height),
                "aspect_ratio": float(aspect_ratio),
                "megapixels": float(mp),
                "resolution": f"{width}x{height}",
                "binary_class": binary_label,
                "coarse_class": coarse_class,
                "fine_label": fine_label,
            }
        )

    df_size = pd.DataFrame(records)

    def _summarize(df_subset: pd.DataFrame) -> dict[str, Any]:
        if df_subset.empty:
            return {}
        widths = df_subset["width"].to_numpy()
        heights = df_subset["height"].to_numpy()
        ars = df_subset["aspect_ratio"].to_numpy()
        mps = df_subset["megapixels"].to_numpy()
        res_counts = df_subset["resolution"].value_counts()
        top_res = str(res_counts.index[0]) if not res_counts.empty else "N/A"
        top_resolutions = [
            {"resolution": str(res), "count": int(cnt)}
            for res, cnt in res_counts.head(5).items()
        ]
        return {
            "count": int(len(df_subset)),
            "width_min": int(np.min(widths)),
            "width_max": int(np.max(widths)),
            "width_mean": float(round(float(np.mean(widths)), 2)),
            "width_std": float(round(float(np.std(widths)), 2)),
            "width_median": int(np.median(widths)),
            "height_min": int(np.min(heights)),
            "height_max": int(np.max(heights)),
            "height_mean": float(round(float(np.mean(heights)), 2)),
            "height_std": float(round(float(np.std(heights)), 2)),
            "height_median": int(np.median(heights)),
            "aspect_ratio_median": float(round(float(np.median(ars)), 3)),
            "megapixels_median": float(round(float(np.median(mps)), 3)),
            "megapixels_mean": float(round(float(np.mean(mps)), 3)),
            "top_resolution": top_res,
            "top_resolutions": top_resolutions,
        }

    total_stats = _summarize(df_size)

    binary_stats: dict[str, dict[str, Any]] = {}
    for label in ("Non-ROI", "ROI"):
        sub = df_size.loc[df_size["binary_class"] == label]
        if not sub.empty:
            binary_stats[label] = _summarize(sub)

    coarse_stats: dict[str, dict[str, Any]] = {}
    for label in COARSE_NAMES:
        sub = df_size.loc[df_size["coarse_class"] == label]
        if not sub.empty:
            coarse_stats[label] = _summarize(sub)

    fine_stats: dict[str, dict[str, Any]] = {}
    for label in sorted(df_size["fine_label"].unique().tolist()):
        sub = df_size.loc[df_size["fine_label"] == label]
        if not sub.empty:
            fine_stats[label] = _summarize(sub)

    size_audit = {
        "total": total_stats,
        "binary_layers": binary_stats,
        "coarse_layers": coarse_stats,
        "fine_layers": fine_stats,
    }

    # Write JSON report
    write_json(run_dir / "reports" / "image_size_distribution.json", size_audit)

    # Write flattened CSV report
    csv_rows: list[dict[str, Any]] = []

    def _add_csv_row(level: str, name: str, stats: dict[str, Any]) -> None:
        if not stats:
            return
        csv_rows.append(
            {
                "level": level,
                "label_name": name,
                "count": stats.get("count", 0),
                "width_min": stats.get("width_min"),
                "width_median": stats.get("width_median"),
                "width_mean": stats.get("width_mean"),
                "width_max": stats.get("width_max"),
                "height_min": stats.get("height_min"),
                "height_median": stats.get("height_median"),
                "height_mean": stats.get("height_mean"),
                "height_max": stats.get("height_max"),
                "aspect_ratio_median": stats.get("aspect_ratio_median"),
                "megapixels_median": stats.get("megapixels_median"),
                "top_resolution": stats.get("top_resolution"),
            }
        )

    _add_csv_row("total", "Total Dataset", total_stats)
    for k, v in binary_stats.items():
        _add_csv_row("binary", k, v)
    for k, v in coarse_stats.items():
        _add_csv_row("coarse", k, v)
    for k, v in fine_stats.items():
        _add_csv_row("fine", k, v)

    pd.DataFrame(csv_rows).to_csv(
        run_dir / "reports" / "image_size_distribution.csv",
        index=False,
    )

    return size_audit
