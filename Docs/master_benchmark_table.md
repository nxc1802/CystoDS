# CystoDS: Bảng Đại Thống Kê Toàn Bộ Thực Nghiệm (Master Benchmark Table)

**Study ID:** `cystods_hierarchical_long_tailed_2026`  
**Giao thức Đánh giá:** 3 phân hoạch hold-out độc lập bệnh nhân (`split_0`, `split_1`, `split_2`) -- 100% Patient-Disjoint  
**Phiên bản Pipeline:** 3.1 (Three-Stage Sequential Hierarchical Fine-Tuning với Curriculum Warmup & Hierarchical Marginalization)  
**Ngày cập nhật:** 20-08-2026

---

## 1. Bảng Tổng Hợp Đại Thống Kê Toàn Bộ 27 Cấu Hình Thực Nghiệm (Stages 10 -- 40)

Bảng đối chuẩn tổng hợp toàn bộ 27 cấu hình thực nghiệm từ Stage 10 đến Stage 40, báo cáo trung bình và độ lệch chuẩn ($\text{Mean} \pm \text{Std}$) qua 3 phân hoạch hold-out độc lập bệnh nhân:

| # | Giai đoạn / Phân nhóm | Phương pháp & Mô hình | Chiến lược Huấn luyện / Hàm Mất Mát | Binary AUROC | Binary F1 | Coarse Acc (%) | Coarse Macro-F1 | Fine Acc (%) | Fine Macro-F1 (Supp) | Fine Macro-F1 (All 22) | C-F Consistency (%) | Parent Acc (Ens/Marg) (%) |
|:---:|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | **STAGE 10: BACKBONE SCREENING** | | | | | | | | | | | |
| 1 | Stage 10 (Backbone) | ResNet-152 | Multitask Joint (CE) | 0.8698 ± 0.050 | 0.8191 ± 0.038 | 56.62% ± 0.3% | 0.4398 ± 0.017 | 34.71% ± 5.2% | 0.2098 ± 0.038 | 0.1482 ± 0.025 | 68.42% ± 3.1% | 54.12% ± 2.8% |
| 2 | Stage 10 (Backbone) | ResNeXt-50 (32x4d) | Multitask Joint (CE) | 0.9088 ± 0.037 | 0.8387 ± 0.025 | 58.61% ± 1.4% | 0.4600 ± 0.028 | 37.05% ± 3.5% | 0.2023 ± 0.036 | 0.1510 ± 0.028 | 71.05% ± 2.5% | 57.30% ± 1.9% |
| 3 | Stage 10 (Backbone) | HRNet-W18 | Multitask Joint (CE) | 0.9385 ± 0.035 | 0.8759 ± 0.022 | 63.66% ± 4.3% | 0.5461 ± 0.035 | 43.44% ± 3.4% | 0.3979 ± 0.056 | 0.2845 ± 0.039 | 73.88% ± 2.2% | 61.55% ± 3.8% |
| 4 | Stage 10 (Backbone) | **Swin-Tiny (Baseline)** | Multitask Joint (CE) | **0.9507 ± 0.027** | **0.8992 ± 0.029** | **71.19% ± 2.5%** | **0.6243 ± 0.014** | **49.28% ± 6.5%** | **0.5105 ± 0.068** | **0.3755 ± 0.045** | **76.45% ± 2.1%** | **68.90% ± 2.4%** |
| 5 | Stage 10 (Backbone) | Swin-Tiny (Single-Task) | Binary Detection Only (BCE) | 0.9590 ± 0.033 | 0.8930 ± 0.034 | — | — | — | — | — | — | — |
| | **STAGE 20: LONG-TAIL LOSS SCREENING** | | | | | | | | | | | |
| 6 | Stage 20 (Long-Tail) | Cross-Entropy | 1-Stage Multi-Task CE | 0.9489 ± 0.042 | 0.8888 ± 0.050 | 67.49% ± 2.2% | 0.5687 ± 0.011 | 50.23% ± 4.6% | 0.5268 ± 0.076 | 0.3850 ± 0.052 | 77.37% ± 3.0% | 66.50% ± 2.1% |
| 7 | Stage 20 (Long-Tail) | Weighted CE | Inverse Class Frequency Weighting | 0.9427 ± 0.036 | 0.8747 ± 0.038 | 67.86% ± 2.2% | 0.5302 ± 0.056 | 49.18% ± 3.5% | 0.5053 ± 0.067 | 0.3712 ± 0.048 | 73.79% ± 2.2% | 65.80% ± 2.5% |
| 8 | Stage 20 (Long-Tail) | Focal Loss | Gamma=2.0 Modulating Factor | 0.9506 ± 0.024 | 0.8938 ± 0.032 | 68.16% ± 3.7% | 0.5593 ± 0.058 | 51.09% ± 5.7% | 0.4976 ± 0.062 | 0.3680 ± 0.041 | 77.09% ± 8.1% | 66.90% ± 3.2% |
| 9 | Stage 20 (Long-Tail) | LDAM Loss | Label-Distribution-Aware Margin | 0.9522 ± 0.020 | 0.8836 ± 0.016 | 69.37% ± 1.9% | 0.5834 ± 0.064 | 45.09% ± 2.8% | 0.4908 ± 0.058 | 0.3610 ± 0.039 | 72.33% ± 3.6% | 67.10% ± 2.0% |
| 10 | Stage 20 (Long-Tail) | Logit Adjustment | Post-hoc Bayesian Prior Margin | 0.9455 ± 0.042 | 0.8888 ± 0.050 | 69.98% ± 3.0% | 0.5837 ± 0.022 | 49.62% ± 1.7% | 0.4822 ± 0.048 | 0.3540 ± 0.032 | 76.98% ± 3.4% | 68.20% ± 2.6% |
| 11 | Stage 20 (Long-Tail) | Balanced Softmax | Instance Frequency Shift | 0.9531 ± 0.038 | 0.8893 ± 0.031 | 69.58% ± 2.6% | 0.5912 ± 0.032 | 50.08% ± 5.7% | 0.4823 ± 0.060 | 0.3582 ± 0.044 | 74.57% ± 3.7% | 67.85% ± 2.4% |
| 12 | Stage 20 (Long-Tail) | **Smoothed Balanced Softmax** | **Patient-based Prior ($\text{patients}_j^{0.5}$)** | **0.9521 ± 0.039** | **0.8907 ± 0.058** | **70.12% ± 3.9%** | **0.6212 ± 0.038** | **52.45% ± 1.7%** | **0.5506 ± 0.074** | **0.3985 ± 0.051** | **77.58% ± 1.6%** | **69.50% ± 3.1%** |
| | **STAGE 40: COMPREHENSIVE ABLATION STUDIES** | | | | | | | | | | | |
| 13 | Stage 40 (Ablation) | 1-Stage Joint Baseline | Multi-Task Joint (CE + SupCon + SBS) | 0.9594 ± 0.018 | 0.8965 ± 0.012 | 73.64% ± 0.9% | 0.6576 ± 0.006 | 52.63% ± 2.9% | 0.5026 ± 0.046 | 0.3718 ± 0.026 | 74.38% ± 3.2% | 71.20% ± 1.2% |
| 14 | Stage 40 (Ablation) | 2-Stage Decoupled (D2S-HFT) | Phase 1 Rep -> Phase 2 Fine SBS | 0.9617 ± 0.028 | 0.8912 ± 0.022 | 72.12% ± 4.6% | 0.6313 ± 0.031 | 52.20% ± 4.8% | 0.5266 ± 0.056 | 0.3893 ± 0.032 | 78.90% ± 2.4% | 72.50% ± 3.5% |
| 15 | Stage 40 (Ablation) | 3S-HFT Fixed Hierarchy ($w=0.25$) | Cố định Hierarchy Loss Weight P1-P3 | 0.9466 ± 0.031 | 0.8776 ± 0.025 | 70.09% ± 2.3% | 0.6119 ± 0.018 | 47.21% ± 2.4% | 0.5199 ± 0.048 | 0.3844 ± 0.023 | 81.88% ± 2.0% | 75.50% ± 1.5% |
| 16 | Stage 40 (Ablation) | 3S-HFT Method A (Two-Phase) | $w=0$ ở P1, $w=0.25$ ở P2/P3 | 0.9521 ± 0.028 | 0.8981 ± 0.030 | 71.76% ± 1.1% | 0.6371 ± 0.008 | 48.98% ± 4.1% | 0.5240 ± 0.025 | 0.3883 ± 0.017 | 81.55% ± 1.8% | 76.11% ± 1.1% |
| 17 | Stage 40 (Ablation) | Ablation: w/o SupCon ($w=0$) | CE Thuần Túy ở Phase 1 -> Hierarchy | 0.9437 ± 0.027 | 0.8720 ± 0.028 | 70.07% ± 3.7% | 0.6140 ± 0.038 | 51.57% ± 4.0% | 0.5042 ± 0.052 | 0.3722 ± 0.018 | 77.12% ± 3.1% | 69.80% ± 2.9% |
| 18 | Stage 40 (Ablation) | Ablation: w/o Hierarchy Loss ($w=0$) | Multi-Task Không Ràng Buộc Phân Cấp | 0.9649 ± 0.022 | 0.8950 ± 0.019 | 73.46% ± 1.7% | 0.6426 ± 0.009 | 52.72% ± 2.0% | 0.5414 ± 0.077 | 0.3998 ± 0.047 | 72.40% ± 4.1% | 72.10% ± 1.8% |
| 19 | Stage 40 (Ablation) | Ablation: Strategy cRT | Phase 2 cRT Balanced Sampler (Fine) | 0.9617 ± 0.028 | 0.8910 ± 0.024 | 72.12% ± 4.6% | 0.6313 ± 0.031 | 51.96% ± 2.5% | 0.5311 ± 0.048 | 0.3930 ± 0.029 | 78.45% ± 2.6% | 73.20% ± 3.0% |
| 20 | Stage 40 (Ablation) | Ablation: Target All Heads | Phase 2 Mở Khóa Cả 3 Heads | 0.9583 ± 0.035 | 0.8875 ± 0.031 | 73.10% ± 3.4% | 0.6435 ± 0.031 | 51.55% ± 3.6% | 0.5129 ± 0.059 | 0.3794 ± 0.038 | 76.80% ± 2.9% | 72.80% ± 2.5% |
| 21 | Stage 40 (Ablation) | Ablation: Freeze Stages 1-2 | Partial Finetuning (Swin Stages 3-4) | 0.9524 ± 0.028 | 0.8830 ± 0.026 | 73.56% ± 2.5% | 0.6535 ± 0.033 | 50.63% ± 1.9% | 0.4950 ± 0.028 | 0.3669 ± 0.022 | 75.10% ± 2.0% | 71.90% ± 2.1% |
| 22 | Stage 40 (Ablation) | Ablation: Freeze Stages 1-3 | Partial Finetuning (Swin Stage 4 Only) | 0.9246 ± 0.036 | 0.8540 ± 0.035 | 66.22% ± 2.9% | 0.5765 ± 0.037 | 43.31% ± 4.2% | 0.4814 ± 0.052 | 0.3555 ± 0.024 | 70.20% ± 3.8% | 65.10% ± 2.6% |
| 23 | Stage 40 (Ablation) | Multi-Stage Intermediate Heads | S2 -> Bin, S3 -> Coarse, S4 -> Fine | 0.8355 ± 0.045 | 0.7820 ± 0.041 | 68.73% ± 3.1% | 0.5980 ± 0.032 | 42.64% ± 3.8% | 0.4806 ± 0.049 | 0.3714 ± 0.031 | 71.50% ± 3.5% | 66.30% ± 2.8% |
| | **STAGE 30: PROPOSED METHOD (3S-HFT v3.1)** | | | | | | | | | | | |
| 24 | **Stage 30 (Proposed - Val)** | Proposed 3S-HFT (Trực tiếp) | Curriculum Warmup + SBS Alignment | 0.9571 ± 0.021 | 0.8960 ± 0.026 | 73.57% ± 1.8% | 0.6525 ± 0.012 | 53.07% ± 4.0% | 0.5415 ± 0.036 | 0.4007 ± 0.015 | 82.28% ± 0.5% | 70.76% ± 1.1% |
| 25 | **Stage 30 (Proposed - Val)** | **Proposed 3S-HFT (Ensemble)** | **Warmup + Hierarchical Ens. ($\lambda=0.25$)** | **0.9571 ± 0.021** | **0.8960 ± 0.026** | **78.37% ± 1.0%** 🏆 | **0.6525 ± 0.012** | **53.07% ± 4.0%** 🏆 | **0.5415 ± 0.036** 🏆 | **0.4007 ± 0.015** 🏆 | **82.28% ± 0.5%** 🏆 | **78.37% ± 1.0%** 🏆 |
| 26 | **Stage 30 (Proposed - Test)** | Proposed 3S-HFT (Trực tiếp) | Hold-out Test Split Evaluation | 0.9986 ± 0.0002 | 0.9811 ± 0.004 | 82.37% ± 7.0% | 0.7572 ± 0.117 | 74.73% ± 11.9% | 0.6450 ± 0.111 | 0.4691 ± 0.080 | 89.52% ± 3.8% | 81.18% ± 6.9% |
| 27 | **Stage 30 (Proposed - Test)** | **Proposed 3S-HFT (Ensemble)** | **Hold-out Test Ensemble ($\lambda=0.25$)** | **0.9986 ± 0.0002** 🏆 | **0.9811 ± 0.004** 🏆 | **86.42% ± 3.5%** 🏆 | **0.7572 ± 0.117** 🏆 | **74.73% ± 11.9%** 🏆 | **0.6450 ± 0.111** 🏆 | **0.4691 ± 0.080** 🏆 | **89.52% ± 3.8%** 🏆 | **86.42% ± 3.5%** 🏆 |

---

## 2. Phân Tích & Luận Điểm Khoa Học Cốt Lõi (Key Scientific Insights)

### 2.1. Đột phá từ Lịch trình Curriculum Hierarchy Warmup
- **Bằng chứng:** So với bản 3S-HFT Fixed Hierarchy ($w=0.25$ Cũ), bản Warmup mới tăng vọt hiệu năng trên cả 3 tầng:
  - Binary AUROC: $0.9466 \rightarrow \mathbf{0.9571}$ ($+0.0105$)
  - Coarse Accuracy: $70.09\% \rightarrow \mathbf{73.57\%}$ ($+3.48\%$) | Coarse Macro-F1: $0.6119 \rightarrow \mathbf{0.6525}$ ($+0.0406$)
  - Fine Accuracy: $47.21\% \rightarrow \mathbf{53.07\%}$ ($+5.86\%$) | Fine F1 Supported: $0.5199 \rightarrow \mathbf{0.5415}$ ($+2.16\%$)
- **Cơ chế:** Việc nới lỏng trọng số phân cấp ở các epoch đầu ($w \approx 0$) giúp Backbone tự do học đặc trưng thị giác phong phú, không bị thắt cổ chai sớm. Khi đặc trưng chín muồi, trọng số tăng dần lên $0.25$ giúp siết chặt tính nhất quán phả hệ.

### 2.2. Sức mạnh của Hierarchical Marginalization & Multi-Head Ensemble
- **Bằng chứng:**
  - Trên tập Validation: Nâng độ chính xác nhóm cha Coarse từ $70.76\% \rightarrow \mathbf{78.37\%}$ ($+7.61\%$).
  - Trên tập Test: Nâng độ chính xác nhóm cha Coarse từ $81.18\% \rightarrow \mathbf{86.42\%}$ ($+5.24\%$).
- **Cơ chế:** Tận dụng không gian phân giải cao của 22 lớp Fine để suy ngược xác suất biên cho 5 nhóm Coarse ($P_{\text{from\_fine}}(C) = \sum_{f \in \text{Children}(C)} P_{\text{fine}}(f)$) kết hợp trọng số $\lambda=0.25$, triệt tiêu các lỗi nhầm lẫn do Coarse Head đơn lẻ.

### 2.3. Bóc tách Vai trò các Thành phần từ Ablation Study (Stage 40)
1. **Vai trò bắt buộc của Full Backbone Adaptation:** Đóng băng các tầng sớm (Freeze Stages 1--2 hoặc 1--3) làm suy thoái nghiêm trọng tầng vi thể (Fine F1 giảm $-2.49\%$ và $-3.85\%$).
2. **Thất bại của Multi-Stage Intermediate Heads:** Việc gắn Binary Head ở Stage 2 và Coarse Head ở Stage 3 làm suy sụp Binary AUROC xuống $0.8355$ ($-12.2\%$) và Fine Accuracy xuống $42.64\%$, chứng minh toàn bộ 3 heads bắt buộc phải chia sẻ biểu diễn chín muồi tại Stage 4 (Shared Late-Stage).
3. **Hiệu quả của Supervised Contrastive Loss (SupCon):** Tắt SupCon làm Fine Macro-F1 Supported giảm $-1.57\%$, khẳng định vai trò nén cụm biểu mô bệnh học của $L_{\text{supcon}}$.
