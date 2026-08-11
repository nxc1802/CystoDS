"""Plotting and visual artifact generation.

Extracted from ``cystods.core`` (Step 7 refactor).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from cystods.infra.serialization import stable_int_seed
from cystods.taxonomy import COARSE_NAMES, FINE_NAMES


def plot_class_distributions(
    split_frames: Mapping[str, pd.DataFrame],
    output_path: Path,
) -> None:
    rows = []
    for split_name, frame in split_frames.items():
        counts = frame["class"].value_counts()
        for class_name in COARSE_NAMES:
            rows.append(
                {
                    "split": split_name,
                    "class": class_name,
                    "images": int(counts.get(class_name, 0)),
                }
            )
    data = pd.DataFrame(rows)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=data, x="class", y="images", hue="split")
    plt.yscale("log")
    plt.title("CystoDS split distribution (log scale)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_training_history(history: pd.DataFrame, output_path: Path) -> None:
    if history.empty:
        raise ValueError("Cannot plot empty training history.")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    loss_columns = [
        column
        for column in (
            "train_total_loss",
            "val_total_loss",
            "train_binary_loss",
            "train_coarse_loss",
            "train_fine_loss",
            "train_consistency_loss",
        )
        if column in history
    ]
    for column in loss_columns:
        axes[0].plot(history["epoch"], history[column], marker="o", label=column)
    axes[0].set_title("Training and validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(fontsize=8)
    metric_columns = [
        column
        for column in (
            "monitor_metric",
            "val_binary_auroc",
            "val_coarse_macro_f1",
            "val_fine_macro_f1",
            "val_hierarchical_accuracy",
        )
        if column in history
    ]
    for column in metric_columns:
        axes[1].plot(history["epoch"], history[column], marker="o", label=column)
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion(
    matrix: Sequence[Sequence[int]],
    names: Sequence[str],
    title: str,
    output_path: Path,
) -> None:
    array = np.asarray(matrix)
    size = 14 if len(names) > 10 else 8
    plt.figure(figsize=(size, size))
    sns.heatmap(
        array,
        cmap="Blues",
        annot=len(names) <= 10,
        fmt="d",
        xticklabels=names,
        yticklabels=names,
        cbar=True,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_binary_curves(
    predictions: pd.DataFrame, output_path: Path
) -> None:
    targets = predictions["binary_id"].to_numpy(dtype=np.int64)
    probabilities = np.stack(predictions["binary_probs"].to_numpy())[:, 1]
    if len(np.unique(targets)) != 2:
        plt.figure(figsize=(7, 4))
        plt.text(
            0.5,
            0.5,
            "Binary ROC/PR not evaluable:\ntest set contains one target class.",
            ha="center",
            va="center",
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=180)
        plt.close()
        return
    fpr, tpr, _ = roc_curve(targets, probabilities)
    precision, recall, _ = precision_recall_curve(targets, probabilities)
    auroc = roc_auc_score(targets, probabilities)
    auprc = average_precision_score(targets, probabilities)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, label=f"AUROC={auroc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="grey")
    axes[0].set_xlabel("False-positive rate")
    axes[0].set_ylabel("True-positive rate")
    axes[0].set_title("Binary ROC")
    axes[0].legend()
    axes[1].plot(recall, precision, label=f"AUPRC={auprc:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Binary precision-recall")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_per_class_recall(
    metrics: Mapping[str, Any],
    output_path: Path,
) -> None:
    coarse_rows = (
        metrics["coarse"]["per_class"]
        if metrics["coarse"] is not None
        else []
    )
    fine_rows = metrics["fine"]["per_class"] if metrics["fine"] else []
    rows = [
        {
            "name": f"{row['class_name']} (n={row['support']})",
            "recall": row["recall"],
            "level": "coarse",
        }
        for row in coarse_rows
        if row["recall"] is not None
    ] + [
        {
            "name": f"{row['class_name']} (n={row['support']})",
            "recall": row["recall"],
            "level": "fine",
        }
        for row in fine_rows
        if row["recall"] is not None
    ]
    if not rows:
        raise ValueError("Per-class recall requires coarse or fine metrics.")
    data = pd.DataFrame(rows)
    height = max(6, 0.33 * len(data))
    plt.figure(figsize=(10, height))
    sns.barplot(data=data, x="recall", y="name", hue="level")
    plt.xlim(0, 1)
    plt.title("Per-class recall")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def generate_sample_grid(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    output_path: Path,
) -> None:
    per_class = int(config["sample_grid_images_per_class"])
    if per_class < 1:
        raise ValueError("sample_grid_images_per_class must be positive.")
    fig, axes = plt.subplots(
        len(COARSE_NAMES),
        per_class,
        figsize=(3 * per_class, 2.7 * len(COARSE_NAMES)),
        squeeze=False,
    )
    for row_index, class_name in enumerate(COARSE_NAMES):
        subset = frame[frame["class"] == class_name]
        if subset.empty:
            raise ValueError(f"Sample grid lacks class {class_name}.")
        sampled = subset.sample(
            n=min(per_class, len(subset)),
            random_state=stable_int_seed(
                config["seed"], "sample_grid", class_name
            ),
            replace=False,
        )
        for column_index in range(per_class):
            axis = axes[row_index, column_index]
            axis.axis("off")
            if column_index >= len(sampled):
                continue
            row = sampled.iloc[column_index]
            with Image.open(row["image_path"]) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_title(f"{class_name}\n{row['subclass']}", fontsize=9)
    fig.suptitle("Real CystoDS samples used by this run", y=1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def export_fold_visualizations(
    history: pd.DataFrame,
    test_predictions: pd.DataFrame,
    test_metrics: Mapping[str, Any],
    split_frames: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
    run_dir: Path,
    fold_name: str,
) -> None:
    visual_dir = run_dir / "visualizations" / fold_name
    visual_dir.mkdir(parents=True, exist_ok=True)
    plot_training_history(history, visual_dir / "training_history.png")
    plot_class_distributions(
        split_frames, visual_dir / "split_class_distribution.png"
    )
    if test_metrics["coarse"] is not None:
        plot_confusion(
            test_metrics["coarse"]["confusion_matrix"],
            COARSE_NAMES,
            "Five-class confusion matrix",
            visual_dir / "coarse_confusion_matrix.png",
        )
    if test_metrics["fine"] is not None:
        plot_confusion(
            test_metrics["fine"]["confusion_matrix"],
            FINE_NAMES,
            "22-subclass confusion matrix (Normal mucosa excluded)",
            visual_dir / "fine_confusion_matrix.png",
        )
    if test_metrics["binary"] is not None:
        plot_binary_curves(
            test_predictions, visual_dir / "binary_roc_pr_curves.png"
        )
    if test_metrics["coarse"] is not None or test_metrics["fine"] is not None:
        plot_per_class_recall(
            test_metrics, visual_dir / "per_class_recall.png"
        )
    if config["generate_sample_grid"]:
        generate_sample_grid(
            pd.concat(split_frames.values(), ignore_index=True),
            config,
            visual_dir / "real_data_sample_grid.png",
        )
