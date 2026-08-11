# CystoDS Dataset & Protocol Split Technical Audit Report

Comprehensive audit report consolidating data verification, metadata analysis, 22 fine-class feasibility, and 70/15/15 patient-disjoint split protocol integrity.

---

## 1. Metadata Verification (`cystods.csv`)

### 1.1 Field Structure & Record Count
The dataset metadata file [`cystods.csv`](file:///Volumes/WorkSpace/Project/CystoDS/xvdhy-osfstorage-archive/cystods.csv) contains **8,067 records** across 13 columns:

- `filename`: Image filename stem (alphanumeric ID).
- `pid`: Patient ID (3 to 4 digits across 160 independent patient visits).
- `visit`: Visit sequence number (1 to 7).
- `lesion`: Region of Interest (ROI) identifier (`L1-1`, `L2-1`, `Multifocal`, `NA`).
- `multifocal`: Multifocal severity rating (`2-7`, `8+`, `NA`).
- `bca`: Bladder cancer diagnosis indicator (`1 = Yes`, `0 = No`).
- `class`: 1 of 5 coarse categories (`Malignant`, `Non-malignant`, `Anatomical landmarks`, `Foreign bodies`, `Normal mucosa`).
- `subclass`: 1 of 22 fine sub-classes (`NA` for Normal mucosa).
- `subclass2`: Secondary subclass for mixed pathology.
- `stage`: Cancer stage (`Ta`, `T1`, `T2`, `Tis`, `NA`).
- `morphology`: Lesion morphology (`Papillary`, `Non-papillary`, `NA`).
- `modality`: Imaging modality (`WLC` = 7,617, `BLC` = 450).
- `json`: Segmentation mask flag (`1 = Available`, `0 = None`).

### 1.2 Missing Value Statistics

| Field Name | Missing / NA Count | Missing Ratio (%) | Technical Rationale |
|---|:---:|:---:|---|
| `filename`, `pid`, `class`, `modality`, `bca`, `json` | 0 | 0.00% | 100% Complete |
| `subclass` | 6,386 | 79.16% | All Normal mucosa images carry NA for fine subclass |
| `visit`, `lesion` | 6,708 | 83.15% | Assigned only to lesion / anatomical landmark images |
| `morphology` | 6,815 | 84.48% | Assigned only to Malignant (998) and Non-malignant (221) |
| `stage` | 7,042 | 87.29% | Assigned only to Malignant images (998) |
| `multifocal` | 7,967 | 98.76% | Only 100 images present multifocal lesions |
| `subclass2` | 8,051 | 99.80% | Only 16 images contain dual-pathology subclass |

### 1.3 Filename Extension Mismatch Discovery
- **Metadata Anomaly**: `cystods.csv` lists file extensions such as `.bmp` (294), `.jpg` (165), `.tiff` (44).
- **Physical Disk Audit**: 100% of the 8,067 images on disk inside `images/` are stored as `.png`.
- **Handling**: Data loader strips extensions via `filename.split('.')[0] + '.png'` to ensure 100% decode rate.

---

## 2. 22 Fine-Subclass Feasibility & Patient Count Breakdown

| Subclass | Coarse Parent Class | $N_{\text{images}}$ | $N_{\text{patients}}$ | $N_{\text{visits}}$ | Feasibility Status |
|---|---|:---:|:---:|:---:|---|
| **NormalMucosa (NA)** | `Normal mucosa` | 6,386 | 65 | 17 | Majority class (79.16%) |
| **LowGradePapillary** | `Malignant` | 493 | 60 | 73 | High feasibility |
| **HighGradePapillary** | `Malignant` | 433 | 67 | 76 | High feasibility |
| **AirBubble** | `Foreign bodies` | 210 | 21 | 0* | Moderate feasibility |
| **UreteralOrifice** | `Anatomical landmarks` | 99 | 13 | 0* | Moderate feasibility |
| **BenignNOS** | `Non-malignant` | 97 | 33 | 34 | Moderate feasibility |
| **InflammationNOS** | `Non-malignant` | 80 | 30 | 31 | Moderate feasibility |
| **CIS** | `Malignant` | 71 | 16 | 22 | Attention required (16 pid) |
| **ResectionBed** | `Anatomical landmarks` | 33 | 17 | 18 | Small sample |
| **ResectionScar** | `Anatomical landmarks` | 30 | 4 | 0* | Small sample (4 pid) |
| **Trabeculation** | `Anatomical landmarks` | 21 | 17 | 17 | Small sample |
| **ResectionLoop** | `Foreign bodies` | 17 | 16 | 17 | Small sample |
| **BiopsyForcep** | `Foreign bodies` | 16 | 15 | 16 | Small sample |
| **ProstaticUrethra** | `Anatomical landmarks` | 15 | 15 | 15 | Small sample |
| **CCG** | `Non-malignant` | 13 | 6 | 6 | Rare (6 pid) |
| **Diverticulum** | `Anatomical landmarks` | 13 | 12 | 12 | Rare (12 pid) |
| **Denuded** | `Non-malignant` | 9 | 4 | 4 | Extremely rare (4 pid) |
| **UrothelialPapilloma** | `Non-malignant` | 9 | 3 | 3 | Extremely rare (3 pid) |
| **Stent** | `Foreign bodies` | 8 | 6 | 6 | Extremely rare (6 pid) |
| **SquamousMetaplasia** | `Non-malignant` | 5 | 3 | 3 | Extremely rare (3 pid) |
| **NephrogenicAdenoma** | `Non-malignant` | 4 | 2 | 2 | Severe long-tail (2 pid) |
| **BenignRare** | `Non-malignant` | 4 | 2 | 2 | Severe long-tail (2 pid) |
| **PreMalignant** | `Malignant` | 1 | 1 | 1 | Single sample in dataset |

---

## 3. Protocol Split Audit & Integrity

- **Partition Protocol**: Strict 70% Train / 15% Validation / 15% Test.
- **Patient Disjoint Guarantee**: Seeded search ensures **zero overlap** of patient IDs between train, validation, and test splits.
- **Normal Mucosa Capping**: Normal mucosa images are capped at 540 images for standard training balance, while retaining the full patient distribution.
- **Deterministic Hashing**: Stage 00 computes a SHA-256 fingerprint over `split_manifest.csv` to ensure downstream stages bind to exact split boundaries.
