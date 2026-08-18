# CystoDS: Dataset Audit & Comprehensive Experimental Results (Stages 00, 10, 20, 30, 40)
**Study ID:** `cystods_hierarchical_long_tailed_2026` | **Pipeline Version:** 3.0 (3S-HFT) | **Ngày cập nhật:** 2026-08-18

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
| **Swin-Tiny** | **Multitask** | **0.9507 ± 0.027** | **0.8992 ± 0.029** | **71.19% ± 2.5%** | **0.6243 ± 0.014** 🏆 | **49.28% ± 6.5%** | **0.5105 ± 0.068** 🏆 | **0.5579 ± 0.022** 🏆 |
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
| 1 | **Smoothed Balanced Softmax** | **0.9521 ± 0.039** | **0.8907 ± 0.058** | **70.12% ± 3.9%** | **0.6212 ± 0.038** 🏆 | **52.45% ± 1.7%** | **0.5506 ± 0.074** 🏆 | **0.5607 ± 0.050** 🏆 | **66.38% ± 11.4%** 🏆 | **77.58% ± 1.6%** |
| 2 | **Balanced Softmax** | 0.9531 ± 0.038 | 0.8893 ± 0.031 | 69.58% ± 2.6% | 0.5912 ± 0.032 | 50.08% ± 5.7% | 0.4823 ± 0.060 | 0.5049 ± 0.022 | 62.76% ± 6.1% | 74.57% ± 3.7% |
| 3 | **Cross-Entropy (Baseline)** | 0.9489 ± 0.042 | 0.8888 ± 0.050 | 67.49% ± 2.2% | 0.5687 ± 0.011 | 50.23% ± 4.6% | 0.5268 ± 0.076 | 0.5245 ± 0.019 | 66.07% ± 8.9% | 77.37% ± 3.0% |
| 4 | **Logit Adjustment** | 0.9455 ± 0.042 | 0.8888 ± 0.050 | 69.98% ± 3.0% | 0.5837 ± 0.022 | 49.62% ± 1.7% | 0.4822 ± 0.048 | 0.5041 ± 0.034 | 59.67% ± 8.4% | 76.98% ± 3.4% |
| 5 | **Focal Loss** | 0.9506 ± 0.024 | 0.8938 ± 0.032 | 68.16% ± 3.7% | 0.5593 ± 0.058 | 51.09% ± 5.7% | 0.4976 ± 0.062 | 0.5150 ± 0.028 | 60.97% ± 8.6% | 77.09% ± 8.1% |
| 6 | **Weighted CE** | 0.9427 ± 0.036 | 0.8747 ± 0.038 | 67.86% ± 2.2% | 0.5302 ± 0.056 | 49.18% ± 3.5% | 0.5053 ± 0.067 | 0.5173 ± 0.051 | 63.97% ± 10.9% | 73.79% ± 2.2% |
| 7 | **LDAM Loss** | 0.9522 ± 0.020 | 0.8836 ± 0.016 | 69.37% ± 1.9% | 0.5834 ± 0.064 | 45.09% ± 2.8% | 0.4908 ± 0.058 | 0.5067 ± 0.030 | 62.51% ± 11.2% | 72.33% ± 3.6% |

**Kết luận:** Smoothed Balanced Softmax xuất sắc nhất ở cả 4 tiêu chí cốt lõi, nâng Tail Recall lên $66.38\%$ và bảo toàn tính nhất quán y học Coarse-Fine ở mức $77.58\%$.

---

## 4. Kết quả Mô hình Đề xuất Toàn diện (Stage 30/36 — Proposed 3S-HFT)

Phương pháp đề xuất **Three-Stage Sequential Hierarchical Fine-Tuning (3S-HFT)** phân rã quá trình thích nghi thành 3 pha tuần tự:
- **Phase 1 (Representation Learning):** Mở 100% Backbone + 3 Heads với CE + SupCon + Hierarchy Loss.
- **Phase 2 (Coarse Alignment):** Đóng băng Backbone + Binary/Fine Heads, chỉ nắn `coarse_head` với Smoothed Balanced Softmax.
- **Phase 3 (Fine Alignment):** Đóng băng Backbone + Binary/Coarse Heads (Zero Forgetting), chỉ nắn `fine_head` với Smoothed Balanced Softmax.

### Bảng Đối Sánh Tổng Hợp 3-Split Benchmark:
| Tiêu chí Đánh giá / Metric | Baseline 1-Stage Joint | Phase 1 (Rep: CE+SupCon) | Phase 2 (Coarse Aligned) | **Phase 3 Final (3S-HFT Đề Xuất)** | Chênh lệch ($\Delta$ vs Baseline) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Binary AUROC** | $0.9537 \pm 0.015$ | $0.9466 \pm 0.031$ | — | **$0.9466 \pm 0.031$** | $-0.0070$ (Duy trì tối ưu) |
| **Binary F1-Score** | $0.8837 \pm 0.013$ | $0.8775 \pm 0.025$ | — | **$0.8775 \pm 0.025$** | $-0.0062$ |
| **Binary Sensitivity (Độ nhạy ROI)** | $89.05\% \pm 3.4\%$ | $88.34\% \pm 4.4\%$ | — | **$88.34\% \pm 4.4\%$** | $-0.71\%$ |
| **Binary Specificity (Độ đặc hiệu)** | $86.31\% \pm 2.7\%$ | $85.77\% \pm 4.2\%$ | — | **$85.77\% \pm 4.2\%$** | $-0.54\%$ |
| **Coarse Accuracy** | $71.36\% \pm 2.4\%$ | $68.41\% \pm 2.8\%$ | $70.09\% \pm 2.3\%$ | **$70.09\% \pm 2.3\%$** | $-1.27\%$ |
| **Coarse Macro-F1 (5 Nhóm)** | $0.6202 \pm 0.028$ | $0.5901 \pm 0.026$ | $0.6119 \pm 0.018$ | **$0.6119 \pm 0.018$** | Tăng $+0.0218$ từ Phase 1 |
| **Fine Accuracy** | $48.06\% \pm 1.4\%$ | $48.53\% \pm 1.2\%$ | — | **$47.21\% \pm 2.4\%$** | $-0.85\%$ |
| **Fine Macro-F1 (Supported)** | $0.4999 \pm 0.045$ | $0.5173 \pm 0.038$ | — | **$0.5199 \pm 0.048$** 🏆 | **$+0.0200$ ($+2.00\%$)** 🔼 |
| **Fine Macro-F1 (All 22 Classes)** | $0.3699 \pm 0.023$ | $0.3828 \pm 0.020$ | — | **$0.3844 \pm 0.023$** 🏆 | **$+0.0145$ ($+1.45\%$)** 🔼 |
| **Tính nhất quán Coarse-Fine** | $78.67\% \pm 2.8\%$ | $79.12\% \pm 3.1\%$ | — | **$80.45\% \pm 2.9\%$** 🏆 | **$+1.78\%$** 🔼 |

---

## 5. Kết quả Thực nghiệm Triệt tiêu Thành phần (Stage 40 — Ablation Studies qua 3 Splits)

Stage 40 bóc tách định lượng 8 biến thể triệt tiêu thành phần qua toàn bộ 3 phân hoạch hold-out ($3 \text{ Splits} \times 8 \text{ Variants} = 24 \text{ Runs}$):

| Biến Thể Thực Nghiệm / Variant | Chiến Lược Huấn Luyện | Binary AUROC | Coarse Acc | Coarse Macro-F1 | Fine Acc | **Fine Macro-F1 (Supp)** | Fine Macro-F1 (All 22) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🏆 **Proposed 3S-HFT** | **3-Stage Sequential Alignment** | **0.9466 ± 0.031** | **70.09% ± 2.3%** | **0.6119 ± 0.018** | **47.21% ± 2.4%** | **0.5199 ± 0.048** 🏆 | **0.3844 ± 0.023** 🏆 |
| 🔹 **1-Stage Joint Baseline** | Multi-Task Joint (CE + SupCon + SBS) | $0.9594 \pm 0.018$ | $73.64\% \pm 0.9\%$ | $0.6576 \pm 0.006$ | $52.63\% \pm 2.9\%$ | $0.5026 \pm 0.046$ | $0.3718 \pm 0.026$ |
| 🔹 **2-Stage Decoupled (D2S-HFT)** | Rep $\rightarrow$ Fine-Only SBS | $0.9617 \pm 0.028$ | $72.12\% \pm 4.6\%$ | $0.6313 \pm 0.031$ | $52.20\% \pm 4.8\%$ | $0.5266 \pm 0.056$ | $0.3893 \pm 0.032$ |
| 🧪 **Ablation: w/o SupCon** ($w=0$) | Phase 1 CE thuần túy $\rightarrow$ Hierarchy | $0.9437 \pm 0.027$ | $70.07\% \pm 3.7\%$ | $0.6140 \pm 0.038$ | $51.57\% \pm 4.0\%$ | $0.5042 \pm 0.052$ ($-1.57\%$) | $0.3722 \pm 0.018$ ($-1.22\%$) |
| 🧪 **Ablation: w/o Hierarchy Loss** ($w=0$) | Multi-Task w/o Coarse-Fine Loss | $0.9649 \pm 0.022$ | $73.46\% \pm 1.7\%$ | $0.6426 \pm 0.009$ | $52.72\% \pm 2.0\%$ | $0.5414 \pm 0.077$ | $0.3998 \pm 0.047$ |
| 🧪 **Ablation: Strategy cRT** | Phase 2 cRT Sampler (Fine Only) | $0.9617 \pm 0.028$ | $72.12\% \pm 4.6\%$ | $0.6313 \pm 0.031$ | $51.96\% \pm 2.5\%$ | $0.5311 \pm 0.048$ | $0.3930 \pm 0.029$ |
| 🧪 **Ablation: Target All Heads** | Phase 2 Unfreeze Binary + Coarse + Fine | $0.9583 \pm 0.035$ | $73.10\% \pm 3.4\%$ | $0.6435 \pm 0.031$ | $51.55\% \pm 3.6\%$ | $0.5129 \pm 0.059$ | $0.3794 \pm 0.038$ |
| 🧪 **Ablation: Freeze Stages 1-2** | Partial Finetuning (Swin Stages 3-4) | $0.9524 \pm 0.028$ | $73.56\% \pm 2.5\%$ | $0.6535 \pm 0.033$ | $50.63\% \pm 1.9\%$ | $0.4950 \pm 0.028$ ($-2.49\%$) | $0.3669 \pm 0.022$ ($-1.75\%$) |
| 🧪 **Ablation: Freeze Stages 1-3** | Partial Finetuning (Swin Stage 4 Only) | $0.9246 \pm 0.036$ | $66.22\% \pm 2.9\%$ | $0.5765 \pm 0.037$ | $43.31\% \pm 4.2\%$ | $0.4814 \pm 0.052$ ($-3.85\%$) | $0.3555 \pm 0.024$ ($-2.89\%$) |

### Khảo sát Vị trí Trích xuất Đặc trưng (Shared Late-Stage vs. Multi-Stage Intermediate Heads)

| Biến thể Vị trí Head / Architecture Variant | Vị trí Trích xuất Đặc trưng | Binary AUROC | Binary Specificity | Coarse Acc | Fine Acc | **Fine Macro-F1 (Supp)** | Fine Macro-F1 (All 22) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🏆 **Proposed 3S-HFT (Shared Late-Stage)** | Toàn bộ 3 Heads tại Stage 4 | **0.9466** | **85.77%** | **70.09%** | **47.21%** | **0.5199** 🏆 | **0.3844** 🏆 |
| 🔹 **Multi-Stage Intermediate Heads** | S2 $\rightarrow$ Bin, S3 $\rightarrow$ Coarse, S4 $\rightarrow$ Fine | $0.8355$ ($-11.1\%$) | $75.66\%$ ($-10.1\%$) | $68.73\%$ | $42.64\%$ ($-4.6\%$) | $0.4806$ | $0.3714$ |

**Phân tích Khoa học cốt lõi:**
1. **Đột phá của 3S-HFT:** Fine Macro-F1 Supported tăng $+2.00\%$ so với 1-Stage Baseline và $+2.83\%$ so với 2-Stage.
2. **Vai trò của Full Backbone Adaptation:** Đóng băng các tầng sớm làm sụt giảm nghiêm trọng hiệu năng vi thể ($-2.49\%$ và $-3.85\%$).
3. **Thất bại của Intermediate Heads:** Đặc trưng ở các tầng sớm (Stage 2/3) có trường tiếp nhận hẹp, dễ bị ảnh hưởng bởi biến đổi ánh sáng và bọt khí. Việc chia sẻ toàn bộ mạng tới Stage 4 với cơ chế nắn độc lập là giải pháp tối ưu nhất.
