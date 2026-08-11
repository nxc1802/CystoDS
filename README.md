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
| `stage_10` | Benchmark baselines across 4 backbones (Swin-Tiny, ResNet-152, HRNet-W18, ResNeXt-50) | Baseline performance metrics & checkpoints |
| `stage_20` | Long-tail loss screening (7 loss functions on Swin-Tiny fine head) | Long-tail loss comparison report |
| `stage_30` | Proposed method (Hierarchical + Balanced Softmax + SupCon) | Canonical proposed model evaluation |
| `stage_40` | Component ablation studies (16 ablation configurations) | Systematic ablation comparison table |
| `stage_60` | External cohort validation (evaluation-only) | External validation metrics & report |
| `stage_90` | Final 5-fold cross-validation × 3 random seeds | Cross-validation report & confidence intervals |

---

## ⚙️ Configuration System

All experiment parameters are managed in a single central [`config.yaml`](file:///Volumes/WorkSpace/Project/CystoDS/config.yaml).

Overriding hierarchy (later layer overrides earlier):
1. Base configuration (`config.yaml`)
2. Execution profile (`--profile smoke` or `--profile research`)
3. Environment variables (`CYSTODS_BATCH_SIZE`, `CYSTODS_DATA_ROOT`, etc.)
4. CLI options (`--set key=value`)

---

## 📚 Documentation

Detailed documentation is available in the [`docs/`](file:///Volumes/WorkSpace/Project/CystoDS/docs/) directory:

- [Research Methodology](file:///Volumes/WorkSpace/Project/CystoDS/docs/research.md) — Problem formulation, dataset taxonomy, loss design, and validation protocol.
- [Development Guide](file:///Volumes/WorkSpace/Project/CystoDS/docs/development.md) — Architecture layout, package structure, module contracts, and CLI reference.
- [Experimental Results](file:///Volumes/WorkSpace/Project/CystoDS/docs/results.md) — Comprehensive dataset audit, split verification, and baseline benchmarks.

---

## 📄 Citation & License

This project is licensed under the MIT License — see `LICENSE` for details.
