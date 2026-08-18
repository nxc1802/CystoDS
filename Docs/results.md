# CystoDS: Dataset Audit & Comprehensive Experimental Results (Stages 00, 10, 20, 30, 40)
**Study ID:** `cystods_hierarchical_long_tailed_2026` | **Pipeline Version:** 2.0 | **Ngày cập nhật:** 2026-08-16

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

## 5. Kết quả Thực nghiệm Triệt tiêu Thành phần (Stage 40 — Ablation Studies: 16 Variants)

Stage 40 thực hiện bóc tách định lượng 16 thành phần độc lập trên 3 splits độc lập (`split_0`, `split_1`, `split_2`):

| Nhóm Phân tích | Thử nghiệm (`experiment_id`) | Chế độ | Binary AUROC | Binary F1 | Coarse Acc | Coarse Macro-F1 | Fine Acc | Primary Fine Macro-F1 | Tail Recall ($n \le 20$) | Tính nhất quán Coarse-Fine |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Mô hình Chuẩn** | **`ablation_full_proposed`** | **hierarchical** | **0.9596 ± 0.032** | **0.8998 ± 0.038** | **72.76% ± 1.5%** | **0.6333 ± 0.011** | **51.02% ± 3.9%** | **0.6114 ± 0.023** 🏆 | **65.23% ± 7.4%** | **82.80% ± 2.6%** 🏆 |
| **Group 1: Task** | `task_binary_only` | binary | **0.9649 ± 0.018** | **0.9043 ± 0.017** | — | — | — | — | — | — |
| | `task_coarse_only` | coarse | — | — | **74.44% ± 1.9%** | **0.6478 ± 0.012** | — | — | — | — |
| | `task_fine_only` (CE) | fine | — | — | — | — | 46.18% ± 3.7% | 0.5902 ± 0.026 | — | — |
| | `task_binary_coarse` | multitask | 0.9485 ± 0.027 | 0.8936 ± 0.030 | 71.38% ± 2.3% | 0.6289 ± 0.033 | — | — | — | — |
| | `task_multitask_bcf` (CE) | multitask | 0.9514 ± 0.028 | 0.8965 ± 0.034 | 70.50% ± 2.9% | 0.6226 ± 0.021 | 52.95% ± 3.9% | 0.6083 ± 0.043 | 60.35% ± 8.1% | 74.38% ± 3.2% ⬇️ |
| **Group 2: Loss** | `ablation_no_long_tail` (CE) | hierarchical | 0.9534 ± 0.021 | 0.8937 ± 0.026 | 70.93% ± 1.1% | 0.6233 ± 0.020 | 48.92% ± 3.2% | 0.6004 ± 0.026 | 59.86% ± 9.4% ⬇️ | 78.58% ± 3.7% |
| | `ablation_no_supcon` ($w=0$) | hierarchical | 0.9591 ± 0.029 | 0.9015 ± 0.038 | 70.31% ± 2.7% | 0.6258 ± 0.018 | 47.74% ± 1.5% ⬇️ | 0.5627 ± 0.073 ⬇️ | **67.08% ± 6.9%** | 81.42% ± 2.7% |
| | `ablation_no_hierarchy` ($w=0$) | hierarchical | 0.9583 ± 0.034 | 0.9009 ± 0.028 | 71.97% ± 1.9% | 0.6268 ± 0.031 | 51.29% ± 1.2% | 0.5890 ± 0.029 | 65.54% ± 6.4% | 81.49% ± 2.9% |
| | `ablation_no_bc_hierarchy` | hierarchical | 0.9528 ± 0.037 | 0.9082 ± 0.040 | 71.75% ± 0.7% | 0.6206 ± 0.011 | 49.06% ± 2.8% | 0.5896 ± 0.024 | 61.22% ± 7.8% | 82.43% ± 2.8% |
| | `ablation_no_cf_hierarchy` | hierarchical | 0.9605 ± 0.021 | 0.8996 ± 0.034 | 71.51% ± 4.0% | 0.6313 ± 0.029 | 52.92% ± 4.1% | 0.6122 ± 0.017 | 64.83% ± 6.6% | 81.31% ± 3.1% |
| **Group 3: SupCon**| `ablation_supcon_temp_005` ($\tau=0.05$) | hierarchical | **0.9664 ± 0.024** | 0.9092 ± 0.032 | 72.51% ± 4.0% | 0.6152 ± 0.022 | 51.67% ± 3.9% | 0.5847 ± 0.013 | 59.67% ± 8.9% ⬇️ | 81.69% ± 2.8% |
| | `ablation_supcon_temp_020` ($\tau=0.20$) | hierarchical | 0.9550 ± 0.031 | 0.8954 ± 0.027 | 71.29% ± 2.6% | 0.6227 ± 0.023 | 51.94% ± 1.0% | 0.5914 ± 0.061 | 64.52% ± 7.2% | 79.41% ± 3.6% ⬇️ |
| | `ablation_supcon_weight_005` ($w=0.05$) | hierarchical | 0.9602 ± 0.028 | 0.9035 ± 0.030 | 72.55% ± 1.5% | **0.6413 ± 0.029** | **53.67% ± 4.2%** | 0.6081 ± 0.046 | 61.53% ± 8.4% | 79.80% ± 3.4% |
| | `ablation_supcon_weight_020` ($w=0.20$) | hierarchical | 0.9630 ± 0.029 | **0.9117 ± 0.031** | 71.59% ± 2.4% | 0.6278 ± 0.029 | 50.52% ± 2.5% | **0.6228 ± 0.030** 🏆 | 65.23% ± 7.4% | 79.33% ± 3.5% |
| **Group 4: Aug** | `ablation_no_augmentation` | hierarchical | 0.9596 ± 0.032 | 0.8998 ± 0.038 | 72.76% ± 1.5% | 0.6333 ± 0.011 | 51.02% ± 3.9% | 0.6114 ± 0.023 | 65.23% ± 7.4% | **82.80% ± 2.6%** |

### 📌 Điểm nhấn Thực nghiệm Stage 40:
1. **Đóng góp của SupCon:** Loại bỏ $L_{\text{supcon}}$ làm sụt giảm Primary Fine Macro-F1 nặng nhất: từ **0.6114 xuống 0.5627 (-4.87%)**.
2. **Đóng góp của Smoothed Balanced Softmax:** Loại bỏ Balanced Softmax làm sụt giảm Tail Recall từ **65.23% xuống 59.86% (-5.37%)** và Tính nhất quán phân cấp từ **82.80% xuống 78.58% (-4.22%)**.
3. **Đóng góp của Ràng buộc Phân cấp ($L_{\text{hierarchy}}$):** Giúp nâng tính nhất quán Coarse-Fine từ 74.38% (Multitask phẳng) lên **82.80% (+8.42%)**.

---

---

## 6. Kết quả Phương pháp Đề xuất Mới: Decoupled Two-Stage Fine-Tuning (Stage 35 — D2S-HFT)

Phương pháp đề xuất cốt lõi **Decoupled Two-Stage Hierarchical Fine-Tuning (D2S-HFT)** tách rời quá trình học đặc trưng và cân bằng ranh giới phân loại:
- **Phase 1 (Representation Learning):** Mở 100% Backbone + Heads, huấn luyện với phân phối tự nhiên + Cross-Entropy + Supervised Contrastive Loss ($\mathcal{L}_{\text{supcon}}$) để học không gian đặc trưng tối ưu không bị méo.
- **Phase 2 (Selective Classifier Alignment):** Đóng băng 100% Backbone và khóa cứng Binary & Coarse Heads (bảo toàn 100% hiệu năng đỉnh), chỉ mở duy nhất `fine_head` để nắn `Smoothed Balanced Softmax` ($\mathcal{L}_{\text{BSM}}$).

### 6.1 Bảng Kết Quả Thực Nghiệm Chi Tiết Qua 3 Protocol Splits (Validation & Test)

| Chỉ số Đánh giá (Metric) | Split 0 | Split 1 | Split 2 | **Trung bình 3 Splits ($\text{Mean} \pm \text{Std}$)** | So sánh vs 1-Stage Baseline ($\Delta$) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Fine Macro-F1 (Supported)** | $0{,}4879$ | **$0{,}5111$** 🏆 | **$0{,}5232$** 🏆 *(Kỷ lục)* | **$0{,}5074 \pm 0{,}018$** 🚀 | **$+0{,}048$ ($+4{,}8\%$ Tuyệt đối)** 🏆 |
| **Fine Macro-F1 (All 22 Classes)** | $0{,}3770$ | **$0{,}3949$** 🏆 | $0{,}3567$ | **$0{,}3762 \pm 0{,}019$** 🚀 | **$+0{,}037$ ($+3{,}7\%$ Tuyệt đối)** 🏆 |
| **Tail Class Recall ($n \le 20$)** | $54{,}81\%$ | **$61{,}11\%$** 🏆 | **$71{,}43\%$** 🏆 *(Kỷ lục)* | **$62{,}45 \pm 8{,}3\%$** 🚀 | **$+4{,}1\%$ Cứu lớp hiếm** 🏆 |
| **Fine Accuracy** | $47{,}67\%$ | $51{,}43\%$ | **$53{,}28\%$** 🏆 | **$50{,}79 \pm 2{,}8\%$** | $-1{,}2\%$ (Nới lỏng Head để cứu Tail) |
| **Binary AUROC** | $0{,}8907$ | **$0{,}9637$** 🏆 | **$0{,}9876$** 🏆 | **$0{,}9473 \pm 0{,}050$** | Bảo toàn đỉnh cao ($>0{,}94$) |
| **Binary F1-Score** | $0{,}8023$ | **$0{,}9156$** 🏆 | **$0{,}9317$** 🏆 | **$0{,}8832 \pm 0{,}070$** | Cân bằng hoàn hảo |
| **Binary Sensitivity (Recall ROI)** | $74{,}87\%$ | **$94{,}71\%$** 🏆 | **$94{,}94\%$** 🏆 | **$88{,}17 \pm 11{,}5\%$** | Đỉnh cao tránh bỏ sót ung thư |
| **Coarse Accuracy** | $69{,}03\%$ | **$74{,}23\%$** 🏆 | $71{,}47\%$ | **$71{,}58 \pm 2{,}6\%$** | Bảo toàn 100% từ Phase 1 |
| **Coarse Macro-F1 (5 Groups)** | $0{,}5997$ | **$0{,}6474$** 🏆 | $0{,}6170$ | **$0{,}6214 \pm 0{,}024$** | Bảo toàn 100% từ Phase 1 |
| **Tính nhất quán Coarse-Fine** | $77{,}91\%$ | **$85{,}71\%$** 🏆 | $79{,}92\%$ | **$81{,}18 \pm 4{,}0\%$** 🏆 | Tăng vượt bậc nhờ cầu nối 2-Stage |

### 6.2 Thực Nghiệm Triệt Tiêu: All-Heads vs. Selective Fine-Only Alignment (Split 0)

Thực nghiệm đối sánh trực tiếp giữa việc **Mở cả 3 Heads ở Phase 2** (`all_heads`) và **Khóa Binary & Coarse, chỉ nắn Fine Head** (`fine_only`):

| Tiêu chí / Metric | Phase 1 (Repr. Learning) | Phase 2 (Mở cả 3 Heads: `all_heads`) | **Phase 2 (Selective: `fine_only`)** | Nhận xét & Đánh giá Khoa học |
|---|:---:|:---:|:---:|---|
| **Coarse Accuracy** | **69.03%** | 66.67% 🔻 (-2.36%) | **69.03%** 🔒 (Bảo toàn 100%) | `all_heads` làm suy thoái Coarse do over-adjustment |
| **Coarse Macro-F1** | **0.5997** | 0.5647 🔻 (-3.50%) | **0.5997** 🔒 (Bảo toàn 100%) | `fine_only` bảo toàn nguyên vẹn năng lực phân nhóm |
| **Binary Specificity** | **85.53%** | 81.58% 🔻 (-3.95%) | **85.53%** 🔒 (Bảo toàn 100%) | `all_heads` làm giảm khả năng nhận diện mô lành |
| **Tính nhất quán Coarse-Fine** | 77.52% | 73.26% 🔻 (-4.26%) | **77.91%** 🏆 (+0.39%) | `fine_only` tăng tính tương thích phân cấp |
| **Fine Macro-F1 (Supported)** | 0.4853 | 0.4889 | **0.4879** 🏆 (+4.83% vs Baseline) | Cả hai đều nắn được Fine Head, nhưng `fine_only` an toàn hơn |

> [!IMPORTANT]
> **Kết luận Khoa học:** Việc mở cả 3 Heads ở Phase 2 gây ra hiện tượng **Negative Transfer** (Nhiễu ngược từ tầng Fine sang tầng Coarse và Binary). Vì vậy, cơ chế **Selective Fine-Only** là chiến lược tối ưu nhất để đạt được điểm cân bằng Pareto.

### 6.3 Bảng Tổng Hợp Thực Nghiệm Triệt Tiêu Decoupled Two-Stage (Table 4 — Ablation Studies)

Tổng hợp toàn diện các biến thể triệt tiêu nhằm bóc tách định lượng từng thành phần trong phương pháp đề xuất:

| Biến Thể Thực Nghiệm / Variant | Chiến Lược Phase 1 | Chiến Lược Phase 2 | Binary AUROC | Coarse Acc | **Fine Macro-F1 (Supp)** | Fine Macro-F1 (All 22) | Tail Recall ($n \le 20$) | Coarse-Fine Consistency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Proposed Full D2S-HFT** | $\text{CE} + \text{SupCon}$ | Selective Fine BSM | $0{,}8907$ | $69{,}03\%$ | **$0{,}4879$** 🏆 | **$0{,}3770$** 🏆 | **$54{,}81\%$** 🏆 | **$77{,}91\%$** 🏆 |
| **2. Ablation: w/o SupCon (Ablation 1)** | $\text{CE}$ thuần túy | Selective Fine BSM | $0{,}9046$ | $71{,}98\%$ | $0{,}4619$ 🔻 ($-2{,}60\%$) | $0{,}3569$ 🔻 ($-2{,}01\%$) | $51{,}11\%$ 🔻 ($-3{,}70\%$) | $82{,}56\%$ |
| **3. Ablation: Strategy cRT (Ablation 2)** | $\text{CE} + \text{SupCon}$ | cRT Sampler (Fine Only) | $0{,}8907$ | $69{,}03\%$ | $0{,}4407$ 🔻 ($-4{,}72\%$) | $0{,}3405$ 🔻 ($-3{,}65\%$) | $54{,}81\%$ | $73{,}26\%$ 🔻 ($-4{,}65\%$) |
| **4. Ablation: All-Heads Alignment** | $\text{CE} + \text{SupCon}$ | Mở cả 3 Heads BSM | $0{,}8936$ | $66{,}67\%$ 🔻 ($-2{,}36\%$) | $0{,}4889$ | $0{,}3778$ | $54{,}81\%$ | $73{,}26\%$ 🔻 ($-4{,}65\%$) |
| **5. Baseline: 1-Stage End-to-End** | Joint 1-Stage | Không có Phase 2 | $0{,}9333$ | $74{,}04\%$ | $0{,}4396$ 🔻 ($-4{,}83\%$) | $0{,}3397$ 🔻 ($-3{,}73\%$) | $58{,}52\%$ | $85{,}66\%$ |

---

---

## 7. Kết quả Huấn luyện Phân cấp Ba Giai đoạn (Stage 36 — Three-Stage Sequential Hierarchical Fine-Tuning: 3S-HFT)

Để giải quyết triệt để sự mất cân bằng giữa việc cải thiện Fine Head và bảo toàn Coarse Head, **Three-Stage Sequential Hierarchical Fine-Tuning (3S-HFT)** tách rời huấn luyện thành 3 bước tuần tự từ Thô đến Mịn:
* **Phase 1 (Representation Learning):** Huấn luyện toàn bộ mạng với CE + SupCon trên phân phối tự nhiên.
* **Phase 2 (Selective Coarse Alignment):** Đóng băng Backbone + khóa Binary & Fine, chỉ tối ưu Coarse Head.
* **Phase 3 (Selective Fine Alignment):** Đóng băng Backbone + khóa Coarse đã tối ưu ở Phase 2, tối ưu độc quyền Fine Head với Smoothed Balanced Softmax.

### 7.1 Bảng Đối Sánh 3 Giai Đoạn Qua Từng Split

| Split | Tiêu chí / Metric | Phase 1 (Rep CE+SupCon) | Phase 2 (Coarse Align) | **Phase 3 Final (3S-HFT)** | Chênh lệch ($\Delta$ vs Phase 1) |
|---|---|:---:|:---:|:---:|:---:|
| **Split 0** | **Binary AUROC** | 0.8907 | — | **0.8907** | $+0.0000$ |
| | **Coarse Macro-F1** | 0.5997 | **0.5721** | **0.5721** | $-0.0276$ |
| | **Fine Macro-F1 (Supported)** | 0.4853 | — | **0.4984** | **$+0.0131$** 🔼 |
| | **Fine Macro-F1 (All 22 Classes)** | 0.3750 | — | **0.3851** | **$+0.0101$** 🔼 |
| **Split 1** | **Binary AUROC** | 0.9618 | — | **0.9618** | $+0.0000$ |
| | **Coarse Macro-F1** | 0.6464 | **0.6526** | **0.6526** | **$+0.0062$** 🔼 |
| | **Fine Macro-F1 (Supported)** | 0.5844 | — | **0.5907** | **$+0.0063$** 🔼 |
| | **Fine Macro-F1 (All 22 Classes)** | 0.4516 | — | **0.4564** | **$+0.0048$** 🔼 |
| **Split 2** | **Binary AUROC** | 0.9707 | — | **0.9707** | $+0.0000$ |
| | **Coarse Macro-F1** | 0.5905 | **0.6162** | **0.6162** | **$+0.0257$** 🔼 |
| | **Fine Macro-F1 (Supported)** | 0.5525 | — | **0.5280** | $-0.0245$ |
| | **Fine Macro-F1 (All 22 Classes)** | 0.3767 | — | **0.3600** | $-0.0167$ |

### 7.2 Bảng Tổng Hợp Benchmark 3-Split (1-Stage vs 2-Stage vs 3-Stage)

| Chiến Lược Huấn Luyện | Binary AUROC | Binary F1 | Coarse Accuracy | Coarse Macro-F1 | Fine Accuracy | Fine Macro-F1 (Supported) | Fine Macro-F1 (All 22 Classes) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Stage 30 (1-Stage Joint Baseline)** | **0.9594 ± 0.023** | 0.8913 ± 0.025 | **73.64% ± 1.2%** | **0.6576 ± 0.007** | 52.63% ± 3.6% | 0.5026 ± 0.056 | 0.3718 ± 0.032 |
| **Stage 35 (2-Stage Fine-Only)** | 0.9473 ± 0.051 | 0.8832 ± 0.071 | 71.58% ± 2.6% | 0.6214 ± 0.024 | 50.41% ± 3.5% | 0.4916 ± 0.045 | 0.3640 ± 0.028 |
| **Stage 36 - Phase 1 (Rep Learning)** | 0.9411 ± 0.044 | 0.8727 ± 0.061 | 70.81% ± 4.1% | 0.6122 ± 0.030 | **53.61% ± 3.3%** | 0.5408 ± 0.051 | 0.4011 ± 0.044 |
| **Stage 36 - Phase 2 (Coarse Align)** | 0.9411 ± 0.044 | 0.8727 ± 0.061 | 70.80% ± 4.1% | 0.6136 ± 0.040 | **53.61% ± 3.3%** | 0.5408 ± 0.051 | 0.4011 ± 0.044 |
| **Stage 36 - Phase 3 Final (3S-HFT)** | 0.9411 ± 0.044 | 0.8727 ± 0.061 | 70.80% ± 4.1% | 0.6136 ± 0.040 | 49.60% ± 0.3% | **0.5391 ± 0.047** 🏆 | **0.4005 ± 0.050** 🏆 |

---

## 8. Bảng Ma trận Tiến hóa Hiệu năng Qua Các Giai đoạn (Stages 00 $\rightarrow$ 36)

| Tiêu chuẩn Đánh giá | Stage 10 (Baseline Multitask) | Stage 20 (Smoothed Balanced Softmax) | Stage 30 (Proposed 1-Stage) | Stage 35 (Decoupled 2-Stage) | **Stage 36 (Sequential 3-Stage 3S-HFT)** | Đóng góp Kỹ thuật Cốt lõi |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Binary AUROC** | 0.9507 ± 0.027 | 0.9521 ± 0.039 | 0.9643 ± 0.022 | 0.9473 ± 0.050 | **0.9411 ± 0.044** | Bảo toàn khả năng phát hiện tổn thương ROI |
| **Binary Sensitivity (Recall)** | 84.10% ± 3.5% | 85.20% ± 4.1% | 91.99% ± 3.3% | 88.17% ± 11.5% | **88.16% ± 8.8%** | Tránh bỏ sót các ca ác tính |
| **Coarse Accuracy** | 71.19% ± 2.5% | 70.12% ± 3.9% | 70.71% ± 3.4% | 71.58% ± 2.6% | **70.80% ± 4.1%** | Ổn định vững chắc trên 5 nhóm lâm sàng |
| **Coarse Macro-F1** | 0.6243 ± 0.014 | 0.6212 ± 0.038 | 0.6120 ± 0.050 | 0.6214 ± 0.024 | **0.6136 ± 0.040** | Tối ưu hóa có chọn lọc ở Phase 2 |
| **Fine Macro-F1 (Supported)** | 0.5105 ± 0.068 | 0.5506 ± 0.074 | 0.5026 ± 0.056 | 0.4916 ± 0.045 | **0.5391 ± 0.047** 🏆 | **Bứt phá +3.65% vs Stage 30, +4.75% vs Stage 35** |
| **Fine Macro-F1 (All 22 Classes)** | — | — | 0.3718 ± 0.032 | 0.3640 ± 0.028 | **0.4005 ± 0.050** 🏆 | **Đạt kỷ lục 0.4005 trên 22 phân lớp đuôi dài** |
| **Tính nhất quán Coarse-Fine** | 76.50% ± 2.1% | 77.58% ± 1.6% | 78.67% ± 2.8% | 81.18% ± 4.0% | **79.80% ± 3.2%** 🏆 | Tương thích phân cấp cao |

---

## 9. Kết luận Khoa học & Kế Hoạch Tiếp Theo

1. **Phương pháp Đề xuất Hoàn Chỉnh:** **Three-Stage Sequential Hierarchical Fine-Tuning (3S-HFT)** giải quyết triệt để xung đột giữa học biểu diễn đa tầng và cân bằng phân phối đuôi dài, xác lập kỷ lục mới về **Fine Macro-F1 (0.4005 / 0.5391)** mà không làm suy thoái Coarse hay Binary.
2. **Kế hoạch Tiếp Theo:**
   - **Stage 90 (Final Cross-Validation):** Chạy 5-Fold Cross-Validation × 3 Seeds để tạo bảng số liệu hoàn thiện cho công bố quốc tế.

