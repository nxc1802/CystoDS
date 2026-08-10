"""Generate publication figures from completed CystoDS artifacts only.

This script performs no model fitting, checkpoint update, or metric
recalibration. It reads frozen JSON/CSV outputs and the public images used by
the fixed hold-out protocol, then writes deterministic PNG/PDF figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from matplotlib.ticker import LogLocator, NullFormatter
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "Docs" / "paper_assets"
ASSETS.mkdir(parents=True, exist_ok=True)

PROTOCOL = ROOT / "result" / "stage_00_prepare_protocol_research_20260803-035933"
STAGE10 = ROOT / "result" / "stage_10_simplified_baselines_research_20260803-112142"
STAGE20 = ROOT / "result" / "stage_20_run_long_tail_screen_research_20260802-230424"
STAGE30 = ROOT / "result" / "stage_30_run_proposed_method_research_20260803-001339"
STAGE40 = ROOT / "result" / "stage_40_run_ablations_research_20260803-064951"
PROPOSED = next(
    (ROOT / "result" / "stage_30_run_proposed_method_research_20260803-001339__runs").glob(
        "proposed_hierarchical_swin_smoothed_*"
    )
)


COLORS = {
    "blue": "#2F6B9A",
    "orange": "#D27A2C",
    "green": "#3A8F6A",
    "red": "#B34A4A",
    "purple": "#7158A6",
    "gray": "#73808C",
    "light": "#D9E2EA",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(ASSETS / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(ASSETS / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "savefig.facecolor": "white",
        }
    )
    sns.set_theme(style="whitegrid", context="paper", font="DejaVu Sans")


def figure_dataset_distribution() -> None:
    audit = load_json(PROTOCOL / "reports" / "data_audit.json")
    fine = pd.DataFrame(audit["fine_support"]).sort_values("images", ascending=True)
    coarse = pd.Series(audit["coarse_image_counts"]).sort_values()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 8.5), gridspec_kw={"width_ratios": [0.8, 1.7]})
    coarse.plot.barh(ax=axes[0], color=COLORS["blue"], edgecolor="none")
    axes[0].set_xscale("log")
    axes[0].xaxis.set_major_locator(LogLocator(base=10, numticks=5))
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_xlabel("Số ảnh (thang log)")
    axes[0].set_ylabel("")
    axes[0].set_title("(a) Phân bố 5 lớp thô")
    for y, value in enumerate(coarse):
        axes[0].text(value * 1.08, y, f"{value:,}".replace(",", "."), va="center", fontsize=8)

    y = np.arange(len(fine))
    axes[1].barh(y, fine["images"], color=COLORS["blue"], alpha=0.82, label="Ảnh")
    axes[1].scatter(fine["patients"], y, color=COLORS["orange"], s=24, zorder=3, label="Bệnh nhân")
    axes[1].set_yticks(y, fine["subclass"])
    axes[1].set_xscale("log")
    axes[1].xaxis.set_major_locator(LogLocator(base=10, numticks=5))
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].set_xlabel("Số lượng (thang log)")
    axes[1].set_title("(b) Long-tail ở 22 nhãn fine")
    axes[1].legend(loc="lower right", frameon=True)
    axes[1].axvline(10, color=COLORS["red"], linestyle="--", linewidth=1, alpha=0.8)
    axes[1].text(10.8, 0.5, "Ngưỡng 10 bệnh nhân", color=COLORS["red"], fontsize=7, rotation=90, va="bottom")

    fig.suptitle("Mất cân bằng lớp trong CystoDS (8.067 ảnh, 160 bệnh nhân)")
    fig.tight_layout()
    save_figure(fig, "fig01_dataset_distribution")


def figure_performance_overview() -> None:
    metrics = load_json(PROPOSED / "metrics" / "holdout" / "test_metrics.json")
    ci = load_json(PROPOSED / "metrics" / "holdout" / "patient_bootstrap_ci.json")["intervals"]
    rows = [
        ("Binary AUROC", metrics["binary"]["auroc"], ci["binary_auroc"]),
        ("Binary AUPRC", metrics["binary"]["auprc"], ci["binary_auprc"]),
        ("Binary F1", metrics["binary"]["f1"], ci["binary_f1"]),
        ("Coarse macro-F1", metrics["coarse"]["macro_f1_all_classes"], ci["coarse_macro_f1_all_classes"]),
        ("Coarse balanced accuracy", metrics["coarse"]["balanced_accuracy"], ci["coarse_balanced_accuracy"]),
        ("Fine macro-F1 (supported)", metrics["fine"]["macro_f1_supported"], ci["fine_macro_f1"]),
        ("Fine macro-F1 (22 lớp)", metrics["fine"]["macro_f1_all_classes"], ci["fine_macro_f1_all_classes"]),
        ("Primary macro-F1 cố định", metrics["primary_fine"]["macro_f1_all_classes"], ci["primary_macro_f1_all_classes"]),
        ("Hierarchical accuracy", metrics["hierarchy"]["hierarchical_accuracy"], ci["hierarchical_accuracy"]),
    ]
    labels = [r[0] for r in rows][::-1]
    points = np.array([r[1] for r in rows][::-1])
    lower = np.array([r[2]["lower"] for r in rows][::-1])
    upper = np.array([r[2]["upper"] for r in rows][::-1])
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    colors = [COLORS["green"] if "Binary" in label else COLORS["blue"] for label in labels]
    ax.hlines(y, lower, upper, color=colors, linewidth=2)
    ax.scatter(points, y, color=colors, s=42, zorder=3)
    for yy, p, lo, hi in zip(y, points, lower, upper):
        ax.text(min(1.015, hi + 0.012), yy, f"{p:.3f} [{lo:.3f}–{hi:.3f}]", va="center", fontsize=8)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.16)
    ax.set_xlabel("Điểm số và KTC 95% bootstrap theo bệnh nhân")
    ax.axvline(0.5, color=COLORS["gray"], linewidth=0.8, linestyle=":")
    ax.set_title("Hiệu năng đa mức của mô hình phân cấp trên hold-out")
    fig.tight_layout()
    save_figure(fig, "fig02_multilevel_performance_ci")


def figure_confusion_matrices() -> None:
    metrics = load_json(PROPOSED / "metrics" / "holdout" / "test_metrics.json")
    names = [x["class_name"] for x in metrics["coarse"]["per_class"]]
    cm = np.asarray(metrics["coarse"]["confusion_matrix"], dtype=int)
    norm = cm / cm.sum(axis=1, keepdims=True)
    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f"{cm[i, j]}\n({norm[i, j] * 100:.1f}%)"

    fig, ax = plt.subplots(figsize=(9, 7.2))
    sns.heatmap(
        norm,
        annot=annotations,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=1,
        square=True,
        xticklabels=names,
        yticklabels=names,
        cbar_kws={"label": "Tỷ lệ theo lớp thật"},
        ax=ax,
    )
    ax.set_xlabel("Nhãn dự đoán")
    ax.set_ylabel("Nhãn thật")
    ax.set_title("Ma trận nhầm lẫn 5 lớp (n = 329 ảnh test)")
    plt.setp(ax.get_xticklabels(), rotation=28, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    save_figure(fig, "fig03_coarse_confusion_matrix")


def figure_fine_per_class() -> None:
    metrics = load_json(PROPOSED / "metrics" / "holdout" / "test_metrics.json")["fine"]
    df = pd.DataFrame(metrics["per_class"])
    df = df.sort_values(["target_supported", "f1", "true_count"], ascending=[True, True, True])
    y = np.arange(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(14, 9.2), gridspec_kw={"width_ratios": [2.1, 0.9]})
    axes[0].barh(y - 0.22, df["precision"], height=0.21, color=COLORS["blue"], label="Precision")
    axes[0].barh(y, df["recall"], height=0.21, color=COLORS["orange"], label="Recall")
    axes[0].barh(y + 0.22, df["f1"], height=0.21, color=COLORS["green"], label="F1")
    axes[0].set_yticks(y, df["class_name"])
    axes[0].set_xlim(0, 1.02)
    axes[0].set_xlabel("Điểm số")
    axes[0].set_title("(a) Precision, recall và F1 theo nhãn")
    axes[0].legend(loc="lower right", ncol=3, frameon=True)
    for tick, supported in zip(axes[0].get_yticklabels(), df["target_supported"]):
        if not supported:
            tick.set_color(COLORS["red"])

    axes[1].barh(y, df["true_count"], color=COLORS["purple"], alpha=0.85)
    axes[1].set_yticks(y, [""] * len(y))
    axes[1].set_xscale("symlog", linthresh=1)
    axes[1].set_xlabel("Support thật trong test")
    axes[1].set_title("(b) Cỡ mẫu test")
    for yy, value in zip(y, df["true_count"]):
        axes[1].text(max(value, 0.1) + 0.25, yy, str(int(value)), va="center", fontsize=7)

    fig.suptitle("Phân tích per-class cho 22 nhãn fine (n = 248 ảnh có fine label)")
    fig.tight_layout()
    save_figure(fig, "fig04_fine_per_class")


def figure_error_analysis() -> None:
    fine = load_json(PROPOSED / "metrics" / "holdout" / "test_metrics.json")["fine"]
    names = [x["class_name"] for x in fine["per_class"]]
    cm = np.asarray(fine["confusion_matrix"], dtype=int)
    errors: list[tuple[int, str]] = []
    for i, true_name in enumerate(names):
        for j, pred_name in enumerate(names):
            if i != j and cm[i, j] > 0:
                errors.append((int(cm[i, j]), f"{true_name} → {pred_name}"))
    errors = sorted(errors, reverse=True)[:15][::-1]
    counts = [x[0] for x in errors]
    labels = [x[1] for x in errors]

    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    bars = ax.barh(np.arange(len(errors)), counts, color=COLORS["red"], alpha=0.84)
    ax.set_yticks(np.arange(len(errors)), labels)
    ax.set_xlabel("Số ảnh nhầm")
    ax.set_title("Mười lăm hướng nhầm lẫn fine phổ biến nhất")
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_xlim(0, max(counts) * 1.15)
    fig.tight_layout()
    save_figure(fig, "fig05_fine_error_pairs")


def figure_experiment_comparison() -> None:
    long_tail = pd.read_csv(STAGE20 / "reports" / "child_runs.csv")
    ablation = pd.read_csv(STAGE40 / "reports" / "child_runs.csv")
    proposed = pd.read_csv(STAGE30 / "reports" / "child_runs.csv").iloc[0]

    label_map = {
        "fine_cross_entropy": "CE",
        "fine_weighted_ce": "Weighted CE",
        "fine_focal": "Focal",
        "fine_balanced_softmax": "Balanced Softmax",
        "fine_balanced_softmax_smoothed": "BS + smoothing",
        "fine_logit_adjustment": "Logit adjustment",
        "fine_ldam": "LDAM",
        "ablation_flat_fine_ce": "Flat fine CE",
        "ablation_multitask_no_hierarchy": "Multi-task, no hierarchy",
        "ablation_hierarchical_ce": "Hierarchical CE",
        "ablation_no_binary_auxiliary": "No binary auxiliary",
        "ablation_no_consistency": "No consistency",
        "ablation_no_supcon": "No SupCon",
        "ablation_class_balanced_sampler": "Class-balanced sampler",
        "ablation_train_wlc_only": "Train WLC only",
        "ablation_train_all_evaluate_wlc": "Train all, eval WLC",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.5))

    lt = long_tail.copy()
    lt["label"] = lt["experiment_id"].map(label_map)
    lt = lt.sort_values("primary_macro_f1_all_classes_mean")
    y = np.arange(len(lt))
    axes[0].barh(y, lt["primary_macro_f1_all_classes_mean"], color=COLORS["blue"], label="Primary macro-F1")
    axes[0].scatter(lt["fine_macro_f1_all_classes_mean"], y, color=COLORS["orange"], s=32, label="Fine macro-F1")
    axes[0].set_yticks(y, lt["label"])
    axes[0].set_xlim(0, 0.55)
    axes[0].set_xlabel("Điểm số")
    axes[0].set_title("(a) Sàng lọc loss long-tail")
    axes[0].legend(loc="lower right", frameon=True)

    abl = ablation.copy()
    abl["label"] = abl["experiment_id"].map(label_map)
    abl = abl.sort_values("primary_macro_f1_all_classes_mean")
    y2 = np.arange(len(abl))
    axes[1].barh(y2, abl["primary_macro_f1_all_classes_mean"], color=COLORS["purple"], label="Primary macro-F1")
    axes[1].scatter(abl["fine_macro_f1_all_classes_mean"], y2, color=COLORS["orange"], s=32, label="Fine macro-F1 22 lớp")
    axes[1].axvline(proposed["primary_macro_f1_all_classes_mean"], color=COLORS["green"], linestyle="--", linewidth=1.3, label="Proposed (primary)")
    axes[1].set_yticks(y2, abl["label"])
    axes[1].set_xlim(0, 0.55)
    axes[1].set_xlabel("Điểm số")
    axes[1].set_title("(b) Ablation trên cùng fixed hold-out")
    axes[1].legend(loc="lower right", frameon=True)

    fig.suptitle("So sánh long-tail screen và ablation (một seed/cấu hình)")
    fig.tight_layout()
    save_figure(fig, "fig06_longtail_ablation")


def figure_training_history() -> None:
    hist = pd.read_csv(PROPOSED / "logs" / "holdout_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    axes[0].plot(hist["epoch"], hist["val_binary_auroc"], color=COLORS["green"], label="Binary AUROC")
    axes[0].plot(hist["epoch"], hist["val_coarse_macro_f1_all_classes"], color=COLORS["blue"], label="Coarse macro-F1")
    axes[0].plot(hist["epoch"], hist["val_fine_macro_f1_all_classes"], color=COLORS["orange"], label="Fine macro-F1 (22)")
    axes[0].plot(hist["epoch"], hist["val_hierarchical_accuracy"], color=COLORS["purple"], label="Hierarchical accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation metric")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("(a) Metric trên validation")
    axes[0].legend(loc="lower right", frameon=True)

    axes[1].plot(hist["epoch"], hist["train_total_loss"], color=COLORS["blue"], label="Train total loss")
    axes[1].plot(hist["epoch"], hist["val_total_loss"], color=COLORS["red"], label="Validation total loss")
    axes[1].plot(hist["epoch"], hist["train_fine_loss"], color=COLORS["green"], alpha=0.9, label="Train fine loss")
    axes[1].plot(hist["epoch"], hist["val_fine_loss"], color=COLORS["orange"], alpha=0.9, label="Validation fine loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("(b) Đường học và dấu hiệu overfitting")
    axes[1].legend(loc="upper right", frameon=True)
    fig.suptitle("Lịch sử huấn luyện của run phân cấp đề xuất")
    fig.tight_layout()
    save_figure(fig, "fig07_training_history")


def figure_representative_images() -> None:
    split = pd.read_csv(PROPOSED / "splits" / "holdout" / "test.csv", keep_default_na=False)
    image_root = ROOT / "xvdhy-osfstorage-archive" / "images"
    coarse_order = ["Malignant", "Non-malignant", "Normal mucosa", "Anatomical landmarks", "Foreign bodies"]
    selected = []
    for coarse in coarse_order:
        rows = split.loc[split["class"] == coarse].sort_values("filename").head(2)
        selected.extend(rows.to_dict("records"))

    fig, axes = plt.subplots(5, 2, figsize=(8.2, 15.3))
    for ax, row in zip(axes.flat, selected):
        path = image_root / Path(row["filename"]).with_suffix(".png").name
        with Image.open(path) as im:
            ax.imshow(im.convert("RGB"))
        fine = row["subclass"] if row["subclass"] not in {"", "NA"} else "—"
        ax.set_title(f"{row['class']} | {fine}\nPID {row['pid']} · {row['modality']}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Ví dụ ảnh thật từ fixed test hold-out", y=0.995)
    fig.tight_layout()
    save_figure(fig, "fig08_test_image_examples")


def figure_model_architecture() -> None:
    fig, ax = plt.subplots(figsize=(14.2, 6.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(-0.4, 7.2)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, color: str, size: int = 9) -> None:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.2,
            edgecolor=color,
            facecolor=mpl.colors.to_rgba(color, 0.12),
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size)

    def arrow(x1: float, y1: float, x2: float, y2: float, color: str = COLORS["gray"], style: str = "-") -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle="-|>", mutation_scale=12,
                linewidth=1.2, color=color, linestyle=style,
            )
        )

    box(0.3, 2.7, 1.7, 1.2, "Ảnh nội soi\n224 × 224", COLORS["gray"])
    box(2.8, 2.3, 2.7, 2.0, "Swin-Tiny encoder\nImageNet pretrained\n28,23 M tham số", COLORS["blue"])
    box(6.3, 4.8, 2.25, 1.0, "Binary head\nROI / non-ROI", COLORS["green"])
    box(6.3, 3.0, 2.25, 1.0, "Coarse head\n5 lớp", COLORS["blue"])
    box(6.3, 1.2, 2.25, 1.0, "Fine head\n22 nhãn", COLORS["orange"])
    box(9.6, 4.8, 2.0, 1.0, "Binary CE", COLORS["green"])
    box(9.6, 3.0, 2.0, 1.0, "Coarse CE", COLORS["blue"])
    box(9.6, 1.2, 2.0, 1.0, "Balanced Softmax", COLORS["orange"])
    box(9.6, -0.1, 2.0, 0.8, "SupCon (λ=0,10)", COLORS["purple"], size=8)
    box(9.6, 6.1, 2.0, 0.7, "Consistency (λ=0,25)", COLORS["red"], size=8)
    box(12.3, 2.6, 1.4, 1.8, "Tổng loss\ncó trọng số", COLORS["purple"])
    box(3.0, 0.2, 2.4, 0.8, "Patient-count prior\nα=1; power=0,5", COLORS["orange"], size=8)

    arrow(2.0, 3.3, 2.8, 3.3)
    arrow(5.5, 3.3, 6.3, 5.3)
    arrow(5.5, 3.3, 6.3, 3.5)
    arrow(5.5, 3.3, 6.3, 1.7)
    arrow(8.55, 5.3, 9.6, 5.3, COLORS["green"])
    arrow(8.55, 3.5, 9.6, 3.5, COLORS["blue"])
    arrow(8.55, 1.7, 9.6, 1.7, COLORS["orange"])
    arrow(5.4, 0.6, 6.3, 1.45, COLORS["orange"], "--")
    arrow(7.4, 1.2, 9.6, 0.3, COLORS["purple"], "--")
    arrow(7.4, 4.8, 9.6, 6.4, COLORS["red"], "--")
    arrow(11.6, 5.3, 12.3, 3.9)
    arrow(11.6, 3.5, 12.3, 3.5)
    arrow(11.6, 1.7, 12.3, 3.0)
    arrow(11.6, 0.3, 12.5, 2.6, COLORS["purple"], "--")
    arrow(11.6, 6.4, 12.5, 4.4, COLORS["red"], "--")

    ax.set_title("Kiến trúc coarse-to-fine và các thành phần objective", pad=12)
    fig.tight_layout()
    save_figure(fig, "fig09_model_architecture")


def main() -> None:
    apply_style()
    figure_dataset_distribution()
    figure_performance_overview()
    figure_confusion_matrices()
    figure_fine_per_class()
    figure_error_analysis()
    figure_experiment_comparison()
    figure_training_history()
    figure_representative_images()
    figure_model_architecture()
    print(f"Generated publication assets in {ASSETS}")


if __name__ == "__main__":
    main()
