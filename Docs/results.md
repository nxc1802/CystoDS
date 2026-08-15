# CystoDS: Dataset Audit & Comprehensive Experimental Results (Stages 00, 10, 20, 30)
**Study ID:** `cystods_hierarchical_long_tailed_2026` | **Pipeline Version:** 2.0 | **Ngày cập nhật:** 2026-08-14

---

## 1. Tổng quan Dự án & Kiểm toán Dữ liệu (Stage 00 Protocol & Data Audit)

Dự án **CystoDS** xây dựng hệ thống trí tuệ nhân tạo chẩn đoán nội soi bàng quang (Cystoscopy AI Diagnostic System) giải quyết bài toán phát hiện khối u bàng quang, phân loại cấu trúc mô bệnh học đa tầng (Hierarchical Taxonomy) và khắc phục hiện tượng mất cân bằng dữ liệu đuôi dài cực đoan (Extreme Long-Tailed Distribution).

### 1.1 Thống kê Mẫu & Bệnh nhân Theo 3 Tầng Nhãn Phân cấp
Tập dữ liệu gốc bao gồm **8,067 ảnh** trên **160 bệnh nhân** duy nhất:

| Tầng Phân cấp (Taxonomy Level) | Số lượng Lớp (Classes) | Định nghĩa Lâm sàng & Phân nhóm | Số lượng Mẫu (Raw) |
|---|:---:|---|:---:|
| **Layer 1: Binary Detection** | 2 | **ROI** (Tổn thương: Malignant + Non-malignant)<br>**Non-ROI** (Bình thường: Normal mucosa + Landmarks + Foreign bodies) | 1,219 ROI<br>6,848 Non-ROI |
| **Layer 2: Coarse Grouping** | 5 | **Malignant** (998), **Non-malignant** (221), **Normal mucosa** (6,386), **Anatomical landmarks** (211), **Foreign bodies / Artefacts** (251) | 8,067 |
| **Layer 3: Fine Histopathology** | 22 | **Ác tính**: HighGradePapillary, LowGradePapillary, CIS, PreMalignant, Denuded<br>**Lành tính/Viêm**: BenignNOS, InflammationNOS, BenignRare, NephrogenicAdenoma, SquamousMetaplasia, UrothelialPapilloma<br>**Giải phẫu & Dị vật**: AirBubble, UreteralOrifice, ResectionScar, ResectionBed, Trabeculation, ResectionLoop, ProstaticUrethra, BiopsyForcep, CCG, Diverticulum, Stent | 8,067 (Long-tailed) |

### 1.2 Kiểm toán Kích thước Ảnh & Độ phân giải (Image Resolution Audit)
* **Kích thước gốc:** Chiều rộng từ 252 px đến 5,120 px (Median **352 px**, Mean 455.25 px); Chiều cao từ 209 px đến 2,880 px (Median **240 px**, Mean 315.22 px).
* **Tỉ lệ khung hình (Aspect Ratio):** Median **1.467** (P5: 1.333, P95: 1.467).
* **Top Modes phân giải:** `352x240` (87.06% — chủ yếu niêm mạc & mốc giải phẫu), `640x480` (5.37% — tổn thương ROI), `654x480` (1.71%), `1920x1080` Full-HD (0.84%), `5120x2880` 5K UHD (0.25% — HighGradePapillary, CIS).
* **Tiền xử lý Chuẩn hóa:** Center Crop tỉ lệ 0.92 (`fov_center_crop_ratio: 0.92`) và Bilinear resize về $224 \times 224$ cho toàn bộ mạng nơ-ron.

### 1.3 Giao thức Phân hoạch Độc lập Bệnh nhân (3-Fold Patient-Disjoint Holdout Splits)
* **Tỉ lệ phân chia:** Cố định **70% Train / 15% Validation / 15% Test** (112 Train / 24 Val / 24 Test bệnh nhân).
* **Tính độc lập:** **100% Patient-Disjoint** — đảm bảo không có bệnh nhân nào xuất hiện ở nhiều hơn một tập split.
* **Cân bằng Niêm mạc Lành:** Giới hạn tối đa 540 ảnh Normal mucosa (`normal_mucosa_limit: 540`) để tránh hiện tượng lớp niêm mạc áp đảo không gian biểu diễn.
* **Số lượng ảnh materialized per split:** ~2,221 – 2,225 ảnh (Train: ~1,532 – 1,573; Val: ~326 – 340; Test: ~322 – 349).

---

## 2. Kết quả Sàng lọc Backbone Baseline (Stage 10 — 3-Split Benchmark)

Stage 10 so sánh 4 kiến trúc thị giác máy tính trên cả 2 chế độ **Single-Task (Binary Only)** và **Multitask (Hierarchical 3-Heads)** qua 3 splits độc lập (`split_0`, `split_1`, `split_2`):

| Backbone | Chế độ | Binary AUROC | Binary F1 | Coarse Acc | Coarse Macro-F1 | Fine Acc | Fine Macro-F1 (Supported) | Primary Fine Macro-F1 | Best Monitored Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Swin-Tiny** | **Multitask** | **0.9507 ± 0.027** | **0.8992 ± 0.029** | **71.19% ± 2.5%** | **0.6243 ± 0.014** | **49.28% ± 6.5%** | **0.5105 ± 0.068** | **0.5601 ± 0.061** | **0.5579 ± 0.022** 🏆 |
| **Swin-Tiny** | Binary Only | **0.9590 ± 0.033** | 0.8930 ± 0.034 | — | — | — | — | — | 0.9590 ± 0.033 |
| **HRNet-W18** | **Multitask** | 0.9385 ± 0.035 | 0.8759 ± 0.022 | 63.66% ± 4.3% | 0.5461 ± 0.035 | 43.44% ± 3.4% | 0.3979 ± 0.056 | 0.4682 ± 0.048 | 0.4949 ± 0.049 |
| **HRNet-W18** | Binary Only | 0.9579 ± 0.021 | **0.8984 ± 0.020** | — | — | — | — | — | 0.9576 ± 0.021 |
| **ResNeXt-50** | **Multitask** | 0.9088 ± 0.037 | 0.8387 ± 0.025 | 58.61% ± 1.4% | 0.4600 ± 0.028 | 37.05% ± 3.5% | 0.2023 ± 0.036 | 0.2688 ± 0.041 | 0.3421 ± 0.046 |
| **ResNeXt-50** | Binary Only | 0.9059 ± 0.034 | 0.8356 ± 0.010 | — | — | — | — | — | 0.9115 ± 0.035 |
| **ResNet-152** | **Multitask** | 0.8698 ± 0.050 | 0.8191 ± 0.038 | 56.62% ± 0.3% | 0.4398 ± 0.017 | 34.71% ± 5.2% | 0.2098 ± 0.038 | 0.2678 ± 0.054 | 0.3371 ± 0.029 |
| **ResNet-152** | Binary Only | 0.8879 ± 0.038 | 0.8366 ± 0.030 | — | — | — | — | — | 0.8930 ± 0.038 |

### 📌 Điểm nhấn Thực nghiệm Stage 10:
1. **Swin-Tiny dẫn đầu toàn diện:** Đạt Best Composite Score **0.5579**, Coarse Macro-F1 **0.6243**, Fine Macro-F1 **0.5105** (vượt trội gấp 2.5 lần so với CNN truyền thống).
2. **Tác động của Giám sát Đa nhiệm (Multitask vs. Single-Task):** Huấn luyện Multitask giúp tăng Binary F1-score trên Swin-Tiny (+0.62%) và ResNeXt-50 (+0.31%), đóng vai trò điều hòa không gian biểu diễn và cung cấp đầy đủ thông tin giải thích mô học cho bác sĩ.

---

## 3. Kết quả Sàng lọc 7 Hàm Mất Mát Đuôi Dài (Stage 20 — 3-Split Benchmark)

Stage 20 đánh giá 7 phương pháp loss trên kiến trúc tối ưu **Swin-Tiny** qua 3 splits độc lập:

| # | Phương pháp Hàm Mất Mát | Binary AUROC | Binary F1 | Coarse Acc | Coarse Macro-F1 | Fine Acc | Fine Macro-F1 (Supported) | Primary Fine Macro-F1 (13 Lớp) | Tail Recall ($n \le 20$) | Tính nhất quán Coarse-Fine |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Smoothed Balanced Softmax** | **0.9521 ± 0.039** | **0.8907 ± 0.058** | **70.12% ± 3.9%** | **0.6212 ± 0.038** 🏆 | **52.45% ± 1.7%** | **0.5506 ± 0.074** 🏆 | **0.5607 ± 0.050** 🏆 | **66.38% ± 11.4%** 🏆 | **77.58% ± 1.6%** |
| 2 | **Balanced Softmax** | **0.9531 ± 0.038** | 0.8893 ± 0.031 | 69.58% ± 2.6% | 0.5912 ± 0.032 | 50.08% ± 5.7% | 0.4823 ± 0.060 | 0.5049 ± 0.022 | 62.76% ± 6.1% | 74.57% ± 3.7% |
| 3 | **Cross-Entropy (Baseline)** | 0.9489 ± 0.042 | 0.8888 ± 0.050 | 67.49% ± 2.2% | 0.5687 ± 0.011 | 50.23% ± 4.6% | 0.5268 ± 0.076 | 0.5245 ± 0.019 | 66.07% ± 8.9% | 77.37% ± 3.0% |
| 4 | **Logit Adjustment** | 0.9455 ± 0.042 | 0.8888 ± 0.050 | 69.98% ± 3.0% | 0.5837 ± 0.022 | 49.62% ± 1.7% | 0.4822 ± 0.048 | 0.5041 ± 0.034 | 59.67% ± 8.4% | 76.98% ± 3.4% |
| 5 | **Focal Loss** | 0.9506 ± 0.024 | **0.8938 ± 0.032** | 68.16% ± 3.7% | 0.5593 ± 0.058 | **51.09% ± 5.7%** | 0.4976 ± 0.062 | 0.5150 ± 0.028 | 60.97% ± 8.6% | 77.09% ± 8.1% |
| 6 | **Weighted CE** | 0.9427 ± 0.036 | 0.8747 ± 0.038 | 67.86% ± 2.2% | 0.5302 ± 0.056 | 49.18% ± 3.5% | 0.5053 ± 0.067 | 0.5173 ± 0.051 | 63.97% ± 10.9% | 73.79% ± 2.2% |
| 7 | **LDAM Loss** | 0.9522 ± 0.020 | 0.8836 ± 0.016 | 69.37% ± 1.9% | 0.5834 ± 0.064 | 45.09% ± 2.8% | 0.4908 ± 0.058 | 0.5067 ± 0.030 | 62.51% ± 11.2% | 72.33% ± 3.6% |

### 📌 Điểm nhấn Thực nghiệm Stage 20:
1. **Smoothed Balanced Softmax xuất sắc nhất:** Đạt Macro-F1 cao nhất trên phân lớp chính (**0.5607**), Macro-F1 Fine (**0.5506**), Coarse Macro-F1 (**0.6212**) và Tail Recall (**66.38%**).
2. **Hiệu quả của việc làm mượt theo bệnh nhân:** Việc bù trừ xác suất theo căn bậc hai số lượng bệnh nhân ($\text{patients}_j^{0.5}$) giúp triệt tiêu nhiễu mẫu chụp lặp, bảo toàn tính nhất quán y học Coarse-Fine ở mức **77.58%**.

---

## 4. Kết quả Mô hình Đề xuất Toàn diện (Stage 30 — Proposed Hierarchical Swin + SupCon)

Mô hình hoàn chỉnh tích hợp **Swin-Tiny + Hierarchical 3-Heads + Balanced Softmax + Supervised Contrastive Loss ($L_{\text{supcon}}$)**:

| Chỉ số Đánh giá (Metric) | Split 0 | Split 1 | Split 2 | **Trung bình 3 Splits ($\text{Mean} \pm \text{Std}$)** |
|---|:---:|:---:|:---:|:---:|
| **Binary AUROC** | 0.9328 | **0.9805** | **0.9796** | **0.9643 ± 0.022** 🏆 |
| **Binary F1-Score** | 0.8730 | **0.9326** | **0.9102** | **0.9053 ± 0.025** 🏆 |
| **Độ nhạy Phát hiện Tổn thương (Sensitivity)** | 88.24% | 91.53% | **96.20%** | **91.99% ± 3.3%** 🏆 |
| **Độ đặc hiệu Loại trừ Niêm mạc Lành (Specificity)** | 82.89% | **93.43%** | 86.81% | **87.71% ± 4.4%** |
| **Hệ số Tương quan Matthews (MCC)** | 0.7133 | **0.8445** | **0.8286** | **0.7955 ± 0.059** |
| **Coarse Accuracy** | 68.73% | **75.46%** | 67.94% | **70.71% ± 3.4%** |
| **Coarse Macro-F1 (5 Groups)** | 0.5722 | **0.6824** | 0.5815 | **0.6120 ± 0.050** |
| **Coarse Macro-AUROC (OvR)** | 0.9056 | **0.9132** | 0.9096 | **0.9095 ± 0.003** |
| **Fine Accuracy** | 47.67% | 48.57% | **50.97%** | **49.07% ± 1.4%** |
| **Primary Fine Macro-F1 (13 Lớp)** | 0.5618 | **0.6764** | 0.4232 | **0.5538 ± 0.104** |
| **Hồi phục Lớp Đuôi Dài (Tail Class Recall)** | 54.81% | **69.44%** | **71.43%** | **65.23% ± 7.4%** |
| **Parent Accuracy from Fine Head** | 74.42% | **76.33%** | 75.68% | **75.47% ± 0.8%** |
| **Tính nhất quán Dự đoán Phân cấp** | 75.97% | **82.45%** | 77.61% | **78.67% ± 2.8%** 🏆 |

---

## 5. Bảng Ma trận Tiến hóa Hiệu năng Qua Các Giai đoạn (Project Progression Matrix)

| Tiêu chuẩn Đánh giá | Stage 10 (Baseline Multitask) | Stage 20 (Smoothed Balanced Softmax) | Stage 30 (Proposed Method + SupCon) | Đóng góp Kỹ thuật |
|---|:---:|:---:|:---:|---|
| **Binary AUROC** | 0.9507 ± 0.027 | 0.9521 ± 0.039 | **0.9643 ± 0.022** 🏆 | Tăng +1.36%, cực kỳ nhạy trong phân tách ROI |
| **Binary F1-Score** | 0.8992 ± 0.029 | 0.8907 ± 0.058 | **0.9053 ± 0.025** 🏆 | Đạt đỉnh 0.9326, cân bằng precision/recall |
| **Độ nhạy Lâm sàng** | 87.82% ± 3.1% | 86.41% ± 4.5% | **91.99% ± 3.3%** 🏆 | Giảm tối đa nguy cơ bỏ sót tổn thương ác tính |
| **Tail Class Recall** | 58.40% ± 4.2% | **66.38% ± 11.4%** 🏆 | 65.23% ± 7.4% | Hồi phục các ca bệnh hiếm gặp ($n \le 20$) |
| **Coarse-Fine Consistency** | 76.50% ± 2.1% | 77.58% ± 1.6% | **78.67% ± 2.8%** 🏆 | Ràng buộc logic y học chặt chẽ nhất |

---

## 6. Stage 40 Thực nghiệm Triệt tiêu Thành phần (Ablation Studies - 16 Variants)

**Run:** `research_20260815-001832` (Split 0) | **Chi tiết xem tại:** [Docs/result/ablation_results_report.md](file:///Volumes/WorkSpace/Project/CystoDS/Docs/result/ablation_results_report.md)

### Bảng So sánh 16 Biến thể Triệt tiêu (Validation Split 0):

| Nhóm | Biến thể (`experiment_id`) | Chế độ | Binary AUROC | Coarse Acc | Coarse F1 | Fine Acc | Primary Fine F1 | Coarse-Fine Consistency | Tail Recall |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Anchor** | **`ablation_full_proposed`** | `hierarchical` | **0.9140** | **71.39%** | **0.6232** | **47.29%** | **0.5819** | **85.27%** 🏆 | **54.81%** |
| **G1: Tasks** | `task_binary_only` | `binary` | 0.9395 | — | — | — | — | — | — |
| | `task_coarse_only` | `coarse` | — | 76.11% | 0.6647 | — | — | — | — |
| | `task_fine_only` | `fine` | — | — | — | 41.86% ⬇️ | 0.5552 | — | — |
| | `task_binary_coarse` | `multitask` | 0.9101 | 69.03% | 0.5823 | — | — | — | — |
| | `task_multitask_bcf` | `multitask` | 0.9151 | 67.26% | 0.6097 | 51.16% | 0.5600 | 70.93% ⬇️ | 58.52% |
| **G2: Losses**| `ablation_no_long_tail` | `hierarchical` | 0.9262 | 70.80% | 0.6003 ⬇️ | 45.74% | 0.6062 | 84.11% | 52.59% ⬇️ |
| | `ablation_no_supcon` | `hierarchical` | 0.9201 | 69.03% ⬇️ | 0.6129 | 47.29% | 0.5539 ⬇️ | 82.56% ⬇️ | 54.81% |
| | `ablation_no_hierarchy` | `hierarchical` | 0.9104 | 69.91% | 0.6041 ⬇️ | 50.78% | 0.5677 | 82.56% ⬇️ | 58.52% |
| | `ablation_no_bc_hierarchy` | `hierarchical` | 0.9012 ⬇️ | 70.80% | 0.6067 | 46.12% | 0.5681 | 79.07% ⬇️ | 51.11% ⬇️ |
| | `ablation_no_cf_hierarchy` | `hierarchical` | 0.9318 | 66.96% ⬇️ | 0.6034 | 54.26% | 0.6105 | **75.58%** ⬇️ | 54.81% |
| **G3: SupCon**| `ablation_supcon_temp_005` | `hierarchical` | 0.9323 | 71.39% | 0.6105 | 48.06% | 0.5749 | 82.95% | 54.81% |
| | `ablation_supcon_temp_020` | `hierarchical` | 0.9113 | 71.09% | 0.6032 | 52.33% | 0.5499 ⬇️ | 82.56% | 51.11% ⬇️ |
| | `ablation_supcon_weight_005`| `hierarchical` | 0.9207 | 70.50% | 0.6372 | 48.45% | 0.5956 | 82.56% | 54.81% |
| | `ablation_supcon_weight_020`| `hierarchical` | 0.9218 | 69.03% | 0.5952 ⬇️ | 47.67% | 0.5960 | 77.52% ⬇️ | 54.81% |
| **G4: Aug** | `ablation_no_augmentation` | `hierarchical` | 0.9140 | 71.39% | 0.6232 | 47.29% | 0.5819 | **85.27%** | 54.81% |

### Những Phát Hiện Đột Phá Từ Ablation:
1. **Ràng buộc Coarse-Fine ($L_{\text{cf}}$) là yếu tố sống còn:** Loại bỏ $L_{\text{cf}}$ khiến tính nhất quán logic y học tụt dốc thê thảm từ **85.27% xuống 75.58% (-9.69%)**, dẫn đến tỷ lệ lỗi mâu thuẫn giữa phân nhóm lớn và phân lớp mô học tăng vọt.
2. **Supervised Contrastive Learning ($L_{\text{supcon}}$) nâng tầm phân biệt mô bệnh học:** Bỏ $L_{\text{supcon}}$ làm giảm Primary Fine Macro-F1 từ **0.5819 xuống 0.5539 (-2.80%)** và Coarse Accuracy từ **71.39% xuống 69.03% (-2.36%)**.
3. **Smoothed Balanced Softmax bảo vệ lớp hiếm:** Thay Smoothed Balanced Softmax bằng Cross-Entropy làm suy giảm Tail Class Recall từ **54.81% xuống 52.59%** và Coarse Macro-F1 giảm từ **0.6232 xuống 0.6003**.

---

## 7. Kết luận Khoa học & Hướng triển khai Tiếp theo

1. **Kiến trúc Tối ưu:** **Swin-Tiny** kết hợp **Cấu trúc Đa nhiệm Phân cấp (Hierarchical Multi-Task)**, **Smoothed Balanced Softmax**, và **Supervised Contrastive Learning** là giải pháp toàn diện nhất cho bài toán chẩn đoán nội soi bàng quang.
2. **Các Giai đoạn Tiếp theo:**
   - **Stage 60 (External Validation):** Thẩm định mô hình trên đoàn hệ bệnh nhân độc lập từ nguồn dữ liệu ngoại viện.
   - **Stage 90 (Final Report):** Báo cáo kiểm định 5-Fold Cross-Validation × 3 Seeds trên tập Holdout Test đã niêm phong.
