# CystoDS: Hierarchical Long-Tailed Classification for Cystoscopy Images

Official research implementation for **CystoDS** multi-level cystoscopy image classification under severe long-tailed label imbalance.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Overview

CystoDS addresses two major challenges in computer-assisted cystoscopy:
1. **Hierarchical Label Structure**: Multi-level classification across **Binary** (ROI vs Non-ROI), **Coarse** (5 categories: Malignant, Non-malignant, Normal mucosa, Anatomical landmarks, Foreign bodies), and **Fine** (22 sub-classes).
2. **Severe Long-Tailed Imbalance**: Rare fine classes with as few as 2–3 patients in the entire dataset.

### Proposed Method Architecture

```text
Cystoscopy Image (224x224)
            │
            ▼
    Swin-Tiny Encoder
            │
  ┌─────────┼──────────────┬──────────────┐
  ▼         ▼              ▼              ▼
Binary   Coarse          Fine         SupCon
 Head     Head           Head          Head
 (2)      (5)            (22)          (128d)
                     Balanced Softmax
                     + Hierarchy Loss
```

---

## 🚀 Quickstart

### Installation

```bash
# Clone repository
git clone https://github.com/nxc1802/CystoDS.git
cd CystoDS

# Install package in editable mode
pip install -e .
```

### Basic CLI Usage

```bash
# List all pipeline stages
cystods stages

# Show resolved configuration for a stage
cystods config show --stage 30

# Validate configuration without executing
cystods validate --stage 30

# Run a stage with research profile (default)
cystods run 30

# Run a stage with fast smoke test profile
cystods run 30 --profile smoke

# Run with custom parameters
cystods run 30 --set runtime.batch_size=128 --set runtime.device=cuda
```

---

## 📋 Pipeline Stages

| Stage | Description | Key Outputs |
|---|---|---|
| `stage_00` | Data audit + freeze 70/15/15 patient-disjoint split | `protocol_manifest.json`, `split_manifest.csv` |
| `stage_10` | Benchmark baselines across 4 backbones (Swin-Tiny, ResNet-152, HRNet-W18, ResNeXt-50) & task-formulation baselines (11 trials) | Baseline performance metrics & checkpoints |
| `stage_20` | Long-tail loss screening (7 loss functions on Swin-Tiny fine head) | Long-tail loss comparison report |
| `stage_30` | Proposed method (Hierarchical + Balanced Softmax + SupCon) | Canonical proposed model evaluation |
| `stage_40` | Component ablation studies (16 ablation configurations) | Systematic ablation comparison table |
| `stage_60` | External cohort validation (evaluation-only) | External validation metrics & report |
| `stage_90` | Final 5-fold cross-validation × 3 random seeds | Cross-validation report & confidence intervals |

---

## ⚙️ Configuration System

All experiment parameters are managed in a single central [`config.yaml`](config.yaml).

Overriding hierarchy (later layer overrides earlier, CLI is always highest):
1. Base configuration (`config.yaml`)
2. Stage defaults (`stages[stage]` in `config.yaml`)
3. Execution profile (`--profile smoke` or `--profile research`)
4. Environment variables (`CYSTODS_BATCH_SIZE`, `CYSTODS_DATA_ROOT`, etc.)
5. CLI options (`--set key=value`)

---

## ☁️ Hugging Face Hub Storage & Artifact Sync

Tất cả kết quả thực nghiệm, báo cáo và model checkpoints được tự động đồng bộ và lưu trữ tập trung trên Hugging Face Hub ([`Cuong2004/CystoDS-results`](https://huggingface.co/Cuong2004/CystoDS-results)):

```bash
# Đẩy toàn bộ kết quả thực nghiệm và checkpoints lên Hugging Face Hub
python -m cystods hf push --repo Cuong2004/CystoDS-results

# Tải metrics & báo cáo JSON/CSV về máy local
python -m cystods hf pull-metrics --repo Cuong2004/CystoDS-results

# Tải checkpoint của một mô hình cụ thể (phục vụ Grad-CAM hoặc phân tích)
python -m cystods hf pull-model "resnet152" --repo Cuong2004/CystoDS-results

# Đối soát tính toàn vẹn đồng bộ giữa Local và Hub
python -m cystods hf verify --repo Cuong2004/CystoDS-results
```

---

## 📚 Documentation

Detailed documentation is available in the [`Docs/`](Docs/) directory:

- [Master Benchmark Table](Docs/master_benchmark_table.md) — Đại thống kê toàn bộ thực nghiệm và đối chiếu Table 3 Paper gốc (*Nature Sci Data 2026*).
- [Research Methodology](Docs/reference/) — Problem formulation, dataset taxonomy, loss design, and validation protocol.
- [Kaggle Execution Guide](RUN_KAGGLE.md) — Quy trình chạy tự động và đồng bộ kết quả trên Kaggle.

---

## 📄 Citation & License

This project is licensed under the MIT License — see `LICENSE` for details.
