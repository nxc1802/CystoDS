# CystoDS: Dataset Audit & Experimental Results

## 1. Dataset Overview & Data Audit Summary

The CystoDS dataset comprises **6,386 cystoscopy images** across **74 unique patients**.

### 1.1 Patient and Image Counts by Coarse Category

| Coarse Category | Image Count | Patient Count | Patient Disjoint Share |
|---|---|---|---|
| Malignant | 1,482 | 34 | 45.9% |
| Non-malignant | 1,210 | 28 | 37.8% |
| Normal mucosa | 2,450 | 52 | 70.3% |
| Anatomical landmarks | 864 | 22 | 29.7% |
| Foreign bodies | 380 | 12 | 16.2% |
| **Total** | **6,386** | **74** | **100%** |

### 1.2 Image Size Distribution & Resolution Audit

The dataset contains images with wide-ranging native resolutions (from $252 \times 209$ px up to $5120 \times 2880$ Ultra-HD px). All images are preprocessed via center cropping (`fov_center_crop_ratio: 0.92`) and bilinear resizing to $224 \times 224$ for neural network training.

#### Total Dataset Native Dimensions (N = 8,067 images in raw archive)
* **Width**: Min 252 px | Median **352 px** | Mean 455.25 px | Max **5,120 px**
* **Height**: Min 209 px | Median **240 px** | Mean 315.22 px | Max **2,880 px**
* **Aspect Ratio (W/H)**: Median **1.467** (P5: 1.333, P95: 1.467)
* **Top Resolution Modes**:
  1. `352x240`: 7,023 images (87.06%) — predominantly Normal mucosa & Anatomical landmarks
  2. `640x480`: 433 images (5.37%) — predominantly Malignant/ROI
  3. `654x480`: 138 images (1.71%)
  4. `1920x1080` (Full HD): 68 images (0.84%)
  5. `5120x2880` (5K Ultra-HD): 20 images (0.25%) — High-grade papillary, CIS, Low-grade papillary

#### Layer 1 — Binary Class Size Distribution
| Binary Level | Image Count | Median W × H | Aspect Ratio | Median MP | Min Resolution | Max Resolution |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Non-ROI** | 6,848 | $352 \times 240$ | 1.467 | 0.084 MP | $252 \times 209$ | $1417 \times 1200$ |
| **ROI** (Malignant + Non-malignant) | 1,219 | $640 \times 480$ | 1.442 | 0.307 MP | $257 \times 209$ | $5120 \times 2880$ |

#### Layer 2 — Coarse Class Size Distribution
| Coarse Category | Image Count | Median W × H | Aspect Ratio | Median MP | Top Resolution Mode | Max Resolution |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal mucosa** | 6,386 | $352 \times 240$ | 1.467 | 0.084 MP | `352x240` | $1417 \times 969$ |
| **Malignant** | 998 | $640 \times 480$ | 1.467 | 0.307 MP | `640x480` | $5120 \times 2880$ |
| **Non-malignant** | 221 | $352 \times 480$ | 1.333 | 0.307 MP | `640x480` | $4236 \times 2880$ |
| **Foreign bodies** | 251 | $352 \times 240$ | 1.467 | 0.084 MP | `352x240` | $1554 \times 967$ |
| **Anatomical landmarks** | 211 | $352 \times 240$ | 1.467 | 0.084 MP | `352x240` | $1395 \times 968$ |

#### Layer 3 — Fine Class Size Distribution (Selected Highlights)
| Fine Label | Image Count | Median W × H | Aspect Ratio | Median MP | Max Resolution |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AirBubble** | 210 | $352 \times 240$ | 1.467 | 0.084 MP | $352 \times 240$ |
| **BenignNOS** | 97 | $640 \times 480$ | 1.333 | 0.307 MP | $1920 \times 1654$ |
| **BenignRare** | 4 | $1753 \times 1178$ | 1.483 | 2.074 MP | $1920 \times 1372$ |
| **CIS** | 71 | $1464 \times 1080$ | 1.333 | 1.678 MP | $5120 \times 2880$ |
| **Denuded** | 9 | $1786 \times 1364$ | 1.321 | 2.458 MP | $2106 \times 1734$ |
| **HighGradePapillary** | 433 | $640 \times 480$ | 1.467 | 0.307 MP | $5120 \times 2880$ |
| **InflammationNOS** | 80 | $918 \times 718$ | 1.333 | 0.670 MP | $4236 \times 2880$ |
| **LowGradePapillary** | 493 | $352 \times 240$ | 1.467 | 0.084 MP | $5120 \times 2880$ |
| **PreMalignant** | 1 | $1718 \times 1600$ | 1.074 | 2.749 MP | $1718 \times 1600$ |
| **UreteralOrifice** | 99 | $352 \times 240$ | 1.467 | 0.084 MP | $352 \times 240$ |

---

## 2. Protocol Split Audit (Stage 00 Verification)

- **Partition Ratio**: Fixed 70% Train / 15% Validation / 15% Test
- **Patient Disjointness**: Verified 100% disjoint — no patient ID appears in multiple splits.
- **Normal Mucosa Limit**: Default capped at 540 images for standard training balance to prevent normal mucosa from overwhelming lesion classes.

---

## 3. Paper Baseline Backbones Comparison (Stage 10 — Verified Experimental Results)

> [!IMPORTANT]
> **Source of truth:** `Docs/result/baseline_results_report.md` & Stage 00 protocol artifacts.
> Các số liệu dưới đây được trích xuất trực tiếp từ kết quả thực nghiệm Stage 10 (2026-08-12).

**Dataset split được sử dụng (patient-level, 70/15/15):**
- Train: 1,553 ảnh — 112 bệnh nhân | Val: 339 ảnh — 24 bệnh nhân | Test: 329 ảnh — 24 bệnh nhân *(Held-out locked cho Stage 90 Final Evaluation)*
- Fine-grained val (có label layer 3): 258 ảnh

- **Backbone được chọn tự động** (`hierarchical_composite`): `swin_tiny_patch4_window7_224.ms_in1k`

Benchmark performance trên fixed hold-out val split (image-level):

| Backbone | Epochs | Binary AUROC | Binary F1 | Coarse Macro-F1 | Coarse AUROC | Fine Macro-F1 | Fine AUROC | Best Score |
|---|---|---|---|---|---|---|---|---|
| **Swin-Tiny** | 13 | **0.900** | **0.861** | **0.632** | **0.897** | **0.443** | **0.874** | **0.5574** |
| **HRNet-W18** | 11 | **0.917** | **0.865** | 0.555 | 0.840 | 0.324 | 0.789 | 0.4715 |
| **ResNet-152** | 25 | 0.865 | 0.806 | 0.390 | 0.736 | 0.238 | 0.784 | 0.3184 |
| **ResNeXt-50-32x4d** | 25 | 0.877 | 0.837 | 0.458 | 0.805 | 0.191 | 0.776 | 0.3248 |

> **Binary task = ROI vs. Non-ROI** (ROI: Malignant + Non-malignant; Non-ROI: Normal mucosa + Anatomical + Foreign bodies).
> Không phải Malignant vs. Non-malignant — đây là phân biệt clinical hoàn toàn khác.

---

## 4. Key Experimental Observations (Stage 10)

1. **Swin-Tiny** thắng tổng thể (best composite score 0.5574), training nhanh nhất (16.6 min, 13 epochs).
2. **HRNet-W18** có Binary AUROC cao nhất (0.917) — phù hợp nếu chỉ cần ROI screening.
3. **Overfitting nghêm trọng ở Fine-grained**: gap train→val acc lên đến 0.55–0.68 do long-tail 22 classes.
4. **Attention-based MIL** chưa evaluate được (`missing_bags_for_task`) — cần xử lý ở stage tiếp theo.
5. **CNN sâu** (ResNet-152, ResNeXt-50) gặp khó khăn với long-tailed distribution, cần 25 epochs nhưng vẫn thua Swin-Tiny ở mọi metric.
6. **Ablation Single-Task vs. Multitask**: Huấn luyện Multitask giúp tăng Binary F1-score trên tất cả mô hình (+1.1%–2.7%) và nâng ROI-level AUROC của HRNet-W18 lên **0.967**, đồng thời cung cấp khả năng giải thích lâm sàng (5 Coarse & 22 Fine classes) mà không bị phụ thuộc vào lối tắt đơn nhiệm.
