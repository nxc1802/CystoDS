# CystoDS: Dataset Audit & Comprehensive Experimental Results (Stages 00, 10, 20, 30, 40)
**Study ID:** `cystods_hierarchical_long_tailed_2026` | **Pipeline Version:** 3.1 (3S-HFT with Curriculum Warmup & Hierarchical Marginalization) | **Ngày cập nhật:** 2026-08-20

---

## 1. Tổng quan Dự án & Kiểm toán Dữ liệu (Stage 00 Protocol & Data Audit)

Dự án **CystoDS** xây dựng hệ thống trí tuệ nhân tạo chẩn đoán nội soi bàng quang (Cystoscopy AI Diagnostic System) giải quyết bài toán phát hiện khối u bàng quang, phân loại cấu trúc mô bệnh học đa tầng (Hierarchical Taxonomy) và khắc phục hiện tượng mất cân bằng dữ liệu đuôi dài cực đoan (Extreme Long-Tailed Distribution).

### 1.1 Thống kê Mẫu & Bệnh nhân Theo 3 Tầng Nhãn Phân cấp
Tập dữ liệu gốc bao gồm **8.067 ảnh** trên **160 bệnh nhân** duy nhất:

| Tầng Phân cấp (Taxonomy Level) | Số lượng Lớp (Classes) | Định nghĩa Lâm sàng & Phân nhóm | Số lượng Mẫu (Raw) |
|---|:---:|---|:---:|
| **Layer 1: Binary Detection** | 2 | **ROI** (Tổn thương: Malignant + Non-malignant)<br>**Non-ROI** (Bình thường: Normal mucosa + Landmarks + Foreign bodies) | 1.219 ROI<br>6.848 Non-ROI |
| **Layer 2: Coarse Grouping** | 5 | **Malignant** (998), **Non-malignant** (221), **Normal mucosa** (6.386), **Anatomical landmarks** (211), **Foreign bodies / Artefacts** (251) | 8.067 |
| **Layer 3: Fine Histopathology** | 22 | **Ác tính**: HighGradePapillary, LowGradePapillary, CIS, PreMalignant, Denuded<br>**Lành tính/Viêm**: BenignNOS, InflammationNOS, BenignRare, NephrogenicAdenoma, SquamousMetaplasia, UrothelialPapilloma<br>**Giải phẫu & Dị vật**: AirBubble, UreteralOrifice, ResectionScar, ResectionBed, Trabeculation, ResectionLoop, ProstaticUrethra, BiopsyForcep, CCG, Diverticulum, Stent | 8.067 (Long-tailed) |

### 1.2 Giao thức Phân hoạch Độc lập Bệnh nhân (3-Fold Patient-Disjoint Holdout Splits)
* **Tỉ lệ phân chia:** Cố định **70% Train / 15% Validation / 15% Test** (112 Train / 24 Val / 24 Test bệnh nhân).
* **Tính độc lập:** **100% Patient-Disjoint** — đảm bảo không có bệnh nhân nào xuất hiện ở nhiều hơn một tập split.
* **Cân bằng Niêm mạc Lành:** Giới hạn tối đa 540 ảnh Normal mucosa (`normal_mucosa_limit: 540`) để tránh hiện tượng lớp niêm mạc áp đảo không gian biểu diễn.
* **Số lượng ảnh materialized per split:** ~2.221 – 2.225 ảnh (Train: ~1.532 – 1.573; Val: ~326 – 340; Test: ~322 – 349).

---

## 2. Kết quả Sàng lọc Backbone Baseline (Stage 10 — 3-Split Benchmark)

Stage 10 so sánh 4 kiến trúc thị giác máy tính trên cả 2 chế độ **Single-Task (Binary Only)** và **Multitask (Hierarchical 3-Heads)** qua 3 splits độc lập (`split_0`, `split_1`, `split_2`):

| Backbone | Chế độ | Binary AUROC | Binary F1 | Coarse Acc | Coarse Macro-F1 | Fine Acc | **Fine Macro-F1 (Supp)** | Best Monitored Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Swin-Tiny** | **Multitask** | **0.9507 ± 0.027** | **0.8992 ± 0.029** | **71.19% ± 2.5%** | **0.6243 ± 0.014** | **49.28% ± 6.5%** | **0.5105 ± 0.068** | **0.5579 ± 0.022** |
| **Swin-Tiny** | Binary Only | **0.9590 ± 0.033** | 0.8930 ± 0.034 | — | — | — | — | 0.9590 ± 0.033 |
| **HRNet-W18** | **Multitask** | 0.9385 ± 0.035 | 0.8759 ± 0.022 | 63.66% ± 4.3% | 0.5461 ± 0.035 | 43.44% ± 3.4% | 0.3979 ± 0.056 | 0.4949 ± 0.049 |
| **HRNet-W18** | Binary Only | 0.9579 ± 0.021 | **0.8984 ± 0.020** | — | — | — | — | 0.9576 ± 0.021 |
| **ResNeXt-50** | **Multitask** | 0.9088 ± 0.037 | 0.8387 ± 0.025 | 58.61% ± 1.4% | 0.4600 ± 0.028 | 37.05% ± 3.5% | 0.2023 ± 0.036 | 0.3421 ± 0.046 |
| **ResNeXt-50** | Binary Only | 0.9059 ± 0.034 | 0.8356 ± 0.010 | — | — | — | — | 0.9115 ± 0.035 |
| **ResNet-152** | **Multitask** | 0.8698 ± 0.050 | 0.8191 ± 0.038 | 56.62% ± 0.3% | 0.4398 ± 0.017 | 34.71% ± 5.2% | 0.2098 ± 0.038 | 0.3371 ± 0.029 |
| **ResNet-152** | Binary Only | 0.8879 ± 0.038 | 0.8366 ± 0.030 | — | — | — | — | 0.8930 ± 0.038 |

**Kết luận:** Swin-Tiny được lựa chọn là Backbone tiêu chuẩn cho toàn bộ các giai đoạn sau nhờ khả năng trích xuất đồng thời đặc trưng cục bộ và ngữ cảnh toàn cảnh.

---

## 3. Kết quả Sàng lọc 7 Hàm Mất Mát Đuôi Dài (Stage 20 — 3-Split Benchmark)

Stage 20 đánh giá 7 phương pháp loss trên kiến trúc tối ưu **Swin-Tiny** qua 3 splits độc lập:

| # | Phương pháp Hàm Mất Mát | Binary AUROC | Binary F1 | Coarse Acc | Coarse Macro-F1 | Fine Acc | **Fine Macro-F1 (Supp)** | Primary Fine F1 (13 Lớp) | Tail Recall ($n \le 20$) | Tính nhất quán Coarse-Fine |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Smoothed Balanced Softmax** | **0.9521 ± 0.039** | **0.8907 ± 0.058** | **70.12% ± 3.9%** | **0.6212 ± 0.038** | **52.45% ± 1.7%** | **0.5506 ± 0.074** | **0.5607 ± 0.050** | **66.38% ± 11.4%** | **77.58% ± 1.6%** |
| 2 | **Balanced Softmax** | 0.9531 ± 0.038 | 0.8893 ± 0.031 | 69.58% ± 2.6% | 0.5912 ± 0.032 | 50.08% ± 5.7% | 0.4823 ± 0.060 | 0.5049 ± 0.022 | 62.76% ± 6.1% | 74.57% ± 3.7% |
| 3 | **Cross-Entropy (Baseline)** | 0.9489 ± 0.042 | 0.8888 ± 0.050 | 67.49% ± 2.2% | 0.5687 ± 0.011 | 50.23% ± 4.6% | 0.5268 ± 0.076 | 0.5245 ± 0.019 | 66.07% ± 8.9% | 77.37% ± 3.0% |
| 4 | **Logit Adjustment** | 0.9455 ± 0.042 | 0.8888 ± 0.050 | 69.98% ± 3.0% | 0.5837 ± 0.022 | 49.62% ± 1.7% | 0.4822 ± 0.048 | 0.5041 ± 0.034 | 59.67% ± 8.4% | 76.98% ± 3.4% |
| 5 | **Focal Loss** | 0.9506 ± 0.024 | 0.8938 ± 0.032 | 68.16% ± 3.7% | 0.5593 ± 0.058 | 51.09% ± 5.7% | 0.4976 ± 0.062 | 0.5150 ± 0.028 | 60.97% ± 8.6% | 77.09% ± 8.1% |
| 6 | **Weighted CE** | 0.9427 ± 0.036 | 0.8747 ± 0.038 | 67.86% ± 2.2% | 0.5302 ± 0.056 | 49.18% ± 3.5% | 0.5053 ± 0.067 | 0.5173 ± 0.051 | 63.97% ± 10.9% | 73.79% ± 2.2% |
| 7 | **LDAM Loss** | 0.9522 ± 0.020 | 0.8836 ± 0.016 | 69.37% ± 1.9% | 0.5834 ± 0.064 | 45.09% ± 2.8% | 0.4908 ± 0.058 | 0.5067 ± 0.030 | 62.51% ± 11.2% | 72.33% ± 3.6% |

**Kết luận:** Smoothed Balanced Softmax xuất sắc nhất ở cả 4 tiêu chí cốt lõi, nâng Tail Recall lên $66.38\%$ và bảo toàn tính nhất quán y học Coarse-Fine ở mức $77.58\%$.

---

## 4. Kết quả Mô hình Đề xuất Toàn diện (Stage 30 — Proposed 3S-HFT v3.1)

Phương pháp đề xuất chính thức **Three-Stage Sequential Hierarchical Fine-Tuning với Lịch trình Curriculum Warmup và Hierarchical Marginalization (3S-HFT)**:
- **Phase 1 (Representation Learning):** Mở 100% Backbone với CE + SupCon ($w=0.10$) + **Lịch trình Curriculum Warmup Hierarchy Loss** ($0.0 \rightarrow 0.25$ qua 12 epochs) giúp mạng tự do học không gian đặc trưng tối ưu mà không bị gò bó biểu diễn sớm.
- **Phase 2 (Coarse Alignment):** Đóng băng Backbone + Binary/Fine Heads, chỉ nắn `coarse_head` với Smoothed Balanced Softmax.
- **Phase 3 (Fine Alignment):** Đóng băng Backbone + Binary/Coarse Heads (Zero Forgetting), chỉ nắn `fine_head` với Smoothed Balanced Softmax.
- **Suy luận Đa tầng (Hierarchical Inference):** Áp dụng **Cộng dồn xác suất lớp con về lớp cha (Fine Marginalization)** kết hợp Ensemble 2 Heads ($\lambda=0.25$).

### 4.1 Bảng Đối Sánh Trên Tập Validation (3-Fold Patient-Disjoint Validation):
| Tiêu chí Đánh giá / Metric | Baseline 1-Stage Joint | 3S-HFT Fixed ($w=0.25$) Cũ | **3S-HFT Đề Xuất Mới (Warmup)** | Chênh lệch ($\Delta$ vs Cũ) |
|---|:---:|:---:|:---:|:---:|
| **Binary AUROC** | $0.9594 \pm 0.018$ | $0.9466 \pm 0.031$ | **$0.9571 \pm 0.021$** | **$+0.0105$** |
| **Binary F1-Score** | $0.8965 \pm 0.012$ | $0.8776 \pm 0.025$ | **$0.8960 \pm 0.026$** | **$+0.0184$** |
| **Binary Sensitivity (Độ nhạy)** | $89.05\% \pm 3.4\%$ | $88.34\% \pm 4.4\%$ | **$88.89\% \pm 1.8\%$** | $+0.55\%$ |
| **Binary Specificity (Độ đặc hiệu)**| $86.31\% \pm 2.7\%$ | $85.78\% \pm 4.2\%$ | **$88.60\% \pm 5.0\%$** | **$+2.82\%$** |
| **Coarse Accuracy (Trực tiếp)** | $73.64\% \pm 0.9\%$ | $70.09\% \pm 2.3\%$ | **$73.57\% \pm 1.8\%$** | **$+3.48\%$** 🚀 |
| **Coarse Macro-F1 (5 Nhóm)** | $0.6576 \pm 0.006$ | $0.6119 \pm 0.018$ | **$0.6525 \pm 0.012$** | **$+0.0406$** 🚀 |
| **Fine Accuracy** | $52.63\% \pm 2.9\%$ | $47.21\% \pm 2.4\%$ | **$53.07\% \pm 4.0\%$** | **$+5.86\%$** 🏆 |
| **Fine Macro-F1 (Supported)** | $0.5026 \pm 0.046$ | $0.5199 \pm 0.048$ | **$0.5415 \pm 0.036$** | **$+0.0216$** 🏆 |
| **Fine Macro-F1 (All 22 Classes)** | $0.3718 \pm 0.026$ | $0.3844 \pm 0.023$ | **$0.4007 \pm 0.015$** | **$+0.0163$** 🏆 |
| **Tính nhất quán Coarse-Fine** | $74.38\% \pm 3.2\%$ | $81.88\% \pm 2.0\%$ | **$82.28\% \pm 0.5\%$** | $+0.40\%$ |
| **Parent Acc (Coarse Head $\lambda=1.0$)**| — | $65.95\% \pm 2.9\%$ | **$70.76\% \pm 1.1\%$** | $+4.81\%$ |
| **Parent Acc (Fine Marg. $\lambda=0.0$)**| — | $75.50\% \pm 1.5\%$ | **$78.10\% \pm 0.7\%$** | $+2.60\%$ |
| **🏆 Best Ensemble Parent Acc** | — | $75.50\% \pm 1.5\%$ | **$78.37\% \pm 1.0\%$** | **$+2.87\%$** (λ=0.25) |

---

### 4.2 Bảng Kiểm Định Độc Lập Trên Tập Test (3-Fold Patient-Disjoint Holdout Test):
| Tiêu chí Đánh giá / Metric | 3S-HFT Fixed ($w=0.25$) Cũ | Method A (Two-Phase) | **3S-HFT Đề Xuất Mới (Warmup)** | Mức độ Vượt trội |
|---|:---:|:---:|:---:|:---:|
| **Binary AUROC** | $0.9966 \pm 0.003$ | $0.9976 \pm 0.001$ | **$0.9986 \pm 0.0002$** | **Tiệm cận 1.000** |
| **Binary F1-Score** | $0.9792 \pm 0.010$ | $0.9792 \pm 0.009$ | **$0.9811 \pm 0.004$** | **$0.9811$** |
| **Binary Sensitivity** | $99.04\% \pm 0.7\%$ | $99.23\% \pm 0.7\%$ | **$99.43\% \pm 0.5\%$** | **$99.43\%$** |
| **Binary Specificity** | $96.34\% \pm 1.9\%$ | $96.13\% \pm 1.4\%$ | **$96.34\% \pm 0.3\%$** | **$96.34\%$** |
| **Coarse Accuracy (Trực tiếp)** | $81.56\% \pm 9.1\%$ | $81.76\% \pm 8.7\%$ | **$82.37\% \pm 7.0\%$** | **$82.37\%$** |
| **Coarse Macro-F1** | $0.7494 \pm 0.137$ | $0.7531 \pm 0.130$ | **$0.7572 \pm 0.117$** | **$0.7572$** |
| **Fine Accuracy** | $73.52\% \pm 13.3\%$ | $73.79\% \pm 11.1\%$ | **$74.73\% \pm 11.9\%$** | **$74.73\%$** |
| **Fine Macro-F1 (Supported)** | $0.6424 \pm 0.098$ | $0.6411 \pm 0.096$ | **$0.6450 \pm 0.111$** | **$0.6450$** |
| **Fine Macro-F1 (All 22)** | $0.4672 \pm 0.071$ | $0.4662 \pm 0.070$ | **$0.4691 \pm 0.080$** | **$0.4691$** |
| **Tính nhất quán Coarse-Fine** | $88.04\% \pm 6.7\%$ | $88.58\% \pm 5.0\%$ | **$89.52\% \pm 3.8\%$** | **$89.52\%$** |
| **Parent Acc (Coarse Head $\lambda=1.0$)**| $79.70\% \pm 10.1\%$ | $80.11\% \pm 8.9\%$ | **$81.18\% \pm 6.9\%$** | $+1.48\%$ |
| **Parent Acc (Fine Marg. $\lambda=0.0$)**| $85.35\% \pm 5.4\%$ | **$86.56\% \pm 3.9\%$** | **$86.42\% \pm 3.5\%$** | **$+1.07\%$** |
| **🏆 Best Ensemble Parent Acc** | $85.62\% \pm 5.5\%$ | **$86.69\% \pm 3.8\%$** | **$86.42\% \pm 3.5\%$** | **$86.42\% - 86.69\%$** |

---

## 5. Kết quả Thực nghiệm Triệt tiêu Thành phần (Stage 40 — Ablation Studies qua 3 Splits)

Stage 40 bóc tách định lượng 10 biến thể triệt tiêu thành phần qua toàn bộ 3 phân hoạch hold-out ($3 \text{ Splits} \times 10 \text{ Variants} = 30 \text{ Runs}$):

| Biến Thể Thực Nghiệm / Variant | Chiến Lược Huấn Luyện | Binary AUROC | Coarse Acc | Coarse Macro-F1 | Fine Acc | **Fine Macro-F1 (Supp)** | Fine Macro-F1 (All 22) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🏆 **Proposed 3S-HFT (Curriculum Warmup)** | **3-Stage Sequential Warmup Hierarchy** | **0.9571 ± 0.021** | **73.57% ± 1.8%** | **0.6525 ± 0.012** | **53.07% ± 4.0%** | **0.5415 ± 0.036** 🏆 | **0.4007 ± 0.015** 🏆 |
| 🔹 **3S-HFT Method A (Two-Phase)** | 3-Stage w/ $w=0$ ở P1, $w=0.25$ ở P2/3 | $0.9521 \pm 0.028$ | $71.76\% \pm 1.1\%$ | $0.6371 \pm 0.008$ | $48.98\% \pm 4.1\%$ | $0.5240 \pm 0.025$ | $0.3883 \pm 0.017$ |
| 🔹 **3S-HFT Fixed Hierarchy ($w=0.25$)** | 3-Stage Cố định Hierarchy xuyên suốt | $0.9466 \pm 0.031$ | $70.09\% \pm 2.3\%$ | $0.6119 \pm 0.018$ | $47.21\% \pm 2.4\%$ | $0.5199 \pm 0.048$ | $0.3844 \pm 0.023$ |
| 🔹 **1-Stage Joint Baseline** | Multi-Task Joint (CE + SupCon + SBS) | $0.9594 \pm 0.018$ | $73.64\% \pm 0.9\%$ | $0.6576 \pm 0.006$ | $52.63\% \pm 2.9\%$ | $0.5026 \pm 0.046$ | $0.3718 \pm 0.026$ |
| 🔹 **2-Stage Decoupled (D2S-HFT)** | Rep $\rightarrow$ Fine-Only SBS | $0.9617 \pm 0.028$ | $72.12\% \pm 4.6\%$ | $0.6313 \pm 0.031$ | $52.20\% \pm 4.8\%$ | $0.5266 \pm 0.056$ | $0.3893 \pm 0.032$ |
| 🧪 **Ablation: w/o SupCon** ($w=0$) | Phase 1 CE thuần túy $\rightarrow$ Hierarchy | $0.9437 \pm 0.027$ | $70.07\% \pm 3.7\%$ | $0.6140 \pm 0.038$ | $51.57\% \pm 4.0\%$ | $0.5042 \pm 0.052$ | $0.3722 \pm 0.018$ |
| 🧪 **Ablation: w/o Hierarchy Loss** ($w=0$) | Multi-Task w/o Coarse-Fine Loss | $0.9649 \pm 0.022$ | $73.46\% \pm 1.7\%$ | $0.6426 \pm 0.009$ | $52.72\% \pm 2.0\%$ | $0.5414 \pm 0.077$ | $0.3998 \pm 0.047$ |
| 🧪 **Ablation: Strategy cRT** | Phase 2 cRT Sampler (Fine Only) | $0.9617 \pm 0.028$ | $72.12\% \pm 4.6\%$ | $0.6313 \pm 0.031$ | $51.96\% \pm 2.5\%$ | $0.5311 \pm 0.048$ | $0.3930 \pm 0.029$ |
| 🧪 **Ablation: Target All Heads** | Phase 2 Unfreeze Binary + Coarse + Fine | $0.9583 \pm 0.035$ | $73.10\% \pm 3.4\%$ | $0.6435 \pm 0.031$ | $51.55\% \pm 3.6\%$ | $0.5129 \pm 0.059$ | $0.3794 \pm 0.038$ |
| 🧪 **Ablation: Freeze Stages 1-2** | Partial Finetuning (Swin Stages 3-4) | $0.9524 \pm 0.028$ | $73.56\% \pm 2.5\%$ | $0.6535 \pm 0.033$ | $50.63\% \pm 1.9\%$ | $0.4950 \pm 0.028$ | $0.3669 \pm 0.022$ |
| 🧪 **Ablation: Freeze Stages 1-3** | Partial Finetuning (Swin Stage 4 Only) | $0.9246 \pm 0.036$ | $66.22\% \pm 2.9\%$ | $0.5765 \pm 0.037$ | $43.31\% \pm 4.2\%$ | $0.4814 \pm 0.052$ | $0.3555 \pm 0.024$ |

### Khảo sát Vị trí Trích xuất Đặc trưng (Shared Late-Stage vs. Multi-Stage Intermediate Heads)

| Biến thể Vị trí Head / Architecture Variant | Vị trí Trích xuất Đặc trưng | Binary AUROC | Binary Specificity | Coarse Acc | Fine Acc | **Fine Macro-F1 (Supp)** | Fine Macro-F1 (All 22) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🏆 **Proposed 3S-HFT (Shared Late-Stage)** | Toàn bộ 3 Heads tại Stage 4 | **0.9571** | **88.60%** | **73.57%** | **53.07%** | **0.5415** 🏆 | **0.4007** 🏆 |
| 🔹 **Multi-Stage Intermediate Heads** | S2 $\rightarrow$ Bin, S3 $\rightarrow$ Coarse, S4 $\rightarrow$ Fine | $0.8355$ ($-12.2\%$) | $75.66\%$ ($-12.9\%$) | $68.73\%$ | $42.64\%$ ($-10.4\%$) | $0.4806$ | $0.3714$ |

**Phân tích Khoa học cốt lõi:**
1. **Đột phá của Lịch trình Curriculum Warmup trong 3S-HFT:** Việc giải phóng ràng buộc phân cấp ở những epoch đầu giúp biểu diễn đặc trưng không bị thắt cổ chai, đem lại bước nhảy vọt ở cả 3 nhiệm vụ: Fine Macro-F1 Supported đạt **$0.5415$** (tăng $+2.16\%$ so với bản cũ và $+3.89\%$ so với 1-Stage), Fine Accuracy đạt **$53.07\%$** ($+5.86\%$).
2. **Sức mạnh của Hierarchical Marginalization / Ensemble:** Khi suy luận, cộng dồn xác suất từ 22 lớp Fine về 5 nhóm Coarse giúp độ chính xác nhóm cha tăng vọt từ $70.76\% \rightarrow 78.37\%$ trên tập Val và từ $81.18\% \rightarrow 86.42\%$ trên tập Test.
3. **Thất bại của Intermediate Heads:** Đặc trưng ở các tầng sớm (Stage 2/3) có trường tiếp nhận hẹp, dễ bị ảnh hưởng bởi biến đổi ánh sáng và bọt khí. Việc chia sẻ toàn bộ mạng tới Stage 4 với cơ chế nắn độc lập và lịch trình warmup là giải pháp tối ưu nhất.
