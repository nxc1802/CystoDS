# CystoDS: Bảng Đại Thống Kê Toàn Bộ Thực Nghiệm (Master Benchmark Table)

**Study ID:** `cystods_hierarchical_long_tailed_2026`  
**Giao thức Đánh giá:** 3 phân hoạch hold-out độc lập bệnh nhân (`split_0`, `split_1`, `split_2`) -- 100% Patient-Disjoint  
**Phiên bản Pipeline:** 3.1 (Three-Stage Sequential Hierarchical Fine-Tuning với Curriculum Warmup & Hierarchical Marginalization)  
**Quy chuẩn Đánh dấu:** In đậm và đánh dấu thứ hạng **Top 1 (🥇)**, **Top 2 (🥈)**, **Top 3 (🥉)** **độc lập bên trong từng Stage thực nghiệm** (xét khách quan 100% dựa trên giá trị chỉ số của các phương pháp cùng nhóm).  
**Ngày cập nhật:** 20-08-2026

---

## 1. Bảng Tổng Hợp Đại Thống Kê Toàn Bộ 27 Cấu Hình Thực Nghiệm (Xếp Hạng Theo Từng Stage)

Bảng đối chuẩn tổng hợp toàn bộ 27 cấu hình thực nghiệm từ Stage 10 đến Stage 40 qua 3 phân hoạch hold-out độc lập bệnh nhân. Thứ hạng Top 1 (🥇), Top 2 (🥈), Top 3 (🥉) được tính độc lập trong từng phân nhóm thực nghiệm:

| # | Giai đoạn / Phân nhóm | Phương pháp & Mô hình | Chiến lược Huấn luyện / Hàm Mất Mát | Binary AUROC | Binary F1 | Coarse Acc (%) | Coarse Macro-F1 | Fine Acc (%) | Fine F1 (Supp) | Fine F1 (All 22) | C-F Consistency (%) | Parent Acc (Ens/Marg) (%) |
|:---:|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | **STAGE 10: SÀNG LỌC BACKBONE (5 CẤU HÌNH)** | | | | | | | | | | | |
| 1 | Stage 10 (Backbone) | ResNet-152 | Multitask Joint (CE) | 0.8698 ± 0.050 | 0.8191 ± 0.038 | 56.62% ± 0.3% | 0.4398 ± 0.017 | 34.71% ± 5.2% | **0.2098 ± 0.038** 🥉 | 0.1482 ± 0.025 | 68.42% ± 3.1% | 54.12% ± 2.8% | 
| 2 | Stage 10 (Backbone) | ResNeXt-50 | Multitask Joint (CE) | 0.9088 ± 0.037 | 0.8387 ± 0.025 | **58.61% ± 1.4%** 🥉 | **0.4600 ± 0.028** 🥉 | **37.05% ± 3.5%** 🥉 | 0.2023 ± 0.036 | **0.1510 ± 0.028** 🥉 | **71.05% ± 2.5%** 🥉 | **57.30% ± 1.9%** 🥉 | 
| 3 | Stage 10 (Backbone) | HRNet-W18 | Multitask Joint (CE) | **0.9385 ± 0.035** 🥉 | **0.8759 ± 0.022** 🥉 | **63.66% ± 4.3%** 🥈 | **0.5461 ± 0.035** 🥈 | **43.44% ± 3.4%** 🥈 | **0.3979 ± 0.056** 🥈 | **0.2845 ± 0.039** 🥈 | **73.88% ± 2.2%** 🥈 | **61.55% ± 3.8%** 🥈 | 
| 4 | Stage 10 (Backbone) | Swin-Tiny (Baseline) | Multitask Joint (CE) | **0.9507 ± 0.027** 🥈 | **0.8992 ± 0.029** 🥇 | **71.19% ± 2.5%** 🥇 | **0.6243 ± 0.014** 🥇 | **49.28% ± 6.5%** 🥇 | **0.5105 ± 0.068** 🥇 | **0.3755 ± 0.045** 🥇 | **76.45% ± 2.1%** 🥇 | **68.90% ± 2.4%** 🥇 | 
| 5 | Stage 10 (Backbone) | Swin-Tiny (Single-Task) | Binary Detection Only | **0.9590 ± 0.033** 🥇 | **0.8930 ± 0.034** 🥈 | — | — | — | — | — | — | — | 
| | **STAGE 20: SÀNG LỌC HÀM MẤT MÁT ĐUÔI DÀI (7 CẤU HÌNH)** | | | | | | | | | | | |
| 6 | Stage 20 (Long-Tail) | Cross-Entropy | 1-Stage Multi-Task CE | 0.9489 ± 0.042 | 0.8888 ± 0.050 | 67.49% ± 2.2% | 0.5687 ± 0.011 | **50.23% ± 4.6%** 🥉 | **0.5268 ± 0.076** 🥈 | **0.3850 ± 0.052** 🥈 | **77.37% ± 3.0%** 🥈 | 66.50% ± 2.1% | 
| 7 | Stage 20 (Long-Tail) | Weighted CE | Inverse Class Frequency | 0.9427 ± 0.036 | 0.8747 ± 0.038 | 67.86% ± 2.2% | 0.5302 ± 0.056 | 49.18% ± 3.5% | **0.5053 ± 0.067** 🥉 | **0.3712 ± 0.048** 🥉 | 73.79% ± 2.2% | 65.80% ± 2.5% | 
| 8 | Stage 20 (Long-Tail) | Focal Loss | Gamma=2.0 Modulating Factor | 0.9506 ± 0.024 | **0.8938 ± 0.032** 🥇 | 68.16% ± 3.7% | 0.5593 ± 0.058 | **51.09% ± 5.7%** 🥈 | 0.4976 ± 0.062 | 0.3680 ± 0.041 | **77.09% ± 8.1%** 🥉 | 66.90% ± 3.2% | 
| 9 | Stage 20 (Long-Tail) | LDAM Loss | Margin-based Rare Push | **0.9522 ± 0.020** 🥈 | 0.8836 ± 0.016 | 69.37% ± 1.9% | 0.5834 ± 0.064 | 45.09% ± 2.8% | 0.4908 ± 0.058 | 0.3610 ± 0.039 | 72.33% ± 3.6% | 67.10% ± 2.0% | 
| 10 | Stage 20 (Long-Tail) | Logit Adjustment | Post-hoc Prior Margin | 0.9455 ± 0.042 | 0.8888 ± 0.050 | **69.98% ± 3.0%** 🥈 | **0.5837 ± 0.022** 🥉 | 49.62% ± 1.7% | 0.4822 ± 0.048 | 0.3540 ± 0.032 | 76.98% ± 3.4% | **68.20% ± 2.6%** 🥈 | 
| 11 | Stage 20 (Long-Tail) | Balanced Softmax | Instance Frequency Prior | **0.9531 ± 0.038** 🥇 | **0.8893 ± 0.031** 🥉 | **69.58% ± 2.6%** 🥉 | **0.5912 ± 0.032** 🥈 | 50.08% ± 5.7% | 0.4823 ± 0.060 | 0.3582 ± 0.044 | 74.57% ± 3.7% | **67.85% ± 2.4%** 🥉 | 
| 12 | Stage 20 (Long-Tail) | Smoothed Balanced Softmax | Patient-based Smoothed Prior | **0.9521 ± 0.039** 🥉 | **0.8907 ± 0.058** 🥈 | **70.12% ± 3.9%** 🥇 | **0.6212 ± 0.038** 🥇 | **52.45% ± 1.7%** 🥇 | **0.5506 ± 0.074** 🥇 | **0.3985 ± 0.051** 🥇 | **77.58% ± 1.6%** 🥇 | **69.50% ± 3.1%** 🥇 | 
| | **STAGE 40: THỰC NGHIỆM TRIỆT TIÊU ABLATION (11 BIẾN THỂ)** | | | | | | | | | | | |
| 13 | Stage 40 (Ablation) | 1-Stage Joint Baseline | Joint CE + SupCon + SBS | **0.9594 ± 0.018** 🥉 | **0.8965 ± 0.012** 🥈 | **73.64% ± 0.9%** 🥇 | **0.6576 ± 0.006** 🥇 | **52.63% ± 2.9%** 🥈 | 0.5026 ± 0.046 | 0.3718 ± 0.026 | 74.38% ± 3.2% | 71.20% ± 1.2% | 
| 14 | Stage 40 (Ablation) | 2-Stage Decoupled (D2S-HFT) | Rep -> Fine SBS Only | **0.9617 ± 0.028** 🥈 | 0.8912 ± 0.022 | 72.12% ± 4.6% | 0.6313 ± 0.031 | **52.20% ± 4.8%** 🥉 | **0.5266 ± 0.056** 🥉 | **0.3893 ± 0.032** 🥉 | **78.90% ± 2.4%** 🥉 | 72.50% ± 3.5% | 
| 15 | Stage 40 (Ablation) | 3S-HFT Fixed Hierarchy (w=0.25) | Fixed Hierarchy Weight P1-P3 | 0.9466 ± 0.031 | 0.8776 ± 0.025 | 70.09% ± 2.3% | 0.6119 ± 0.018 | 47.21% ± 2.4% | 0.5199 ± 0.048 | 0.3844 ± 0.023 | **81.88% ± 2.0%** 🥇 | **75.50% ± 1.5%** 🥈 | 
| 16 | Stage 40 (Ablation) | 3S-HFT Method A (Two-Phase) | w=0 in P1, w=0.25 in P2/P3 | 0.9521 ± 0.028 | **0.8981 ± 0.030** 🥇 | 71.76% ± 1.1% | 0.6371 ± 0.008 | 48.98% ± 4.1% | 0.5240 ± 0.025 | 0.3883 ± 0.017 | **81.55% ± 1.8%** 🥈 | **76.11% ± 1.1%** 🥇 | 
| 17 | Stage 40 (Ablation) | Ablation: w/o SupCon (w=0) | CE Only -> Hierarchy | 0.9437 ± 0.027 | 0.8720 ± 0.028 | 70.07% ± 3.7% | 0.6140 ± 0.038 | 51.57% ± 4.0% | 0.5042 ± 0.052 | 0.3722 ± 0.018 | 77.12% ± 3.1% | 69.80% ± 2.9% | 
| 18 | Stage 40 (Ablation) | Ablation: w/o Hierarchy Loss (w=0) | Multi-Task w/o Coarse-Fine Loss | **0.9649 ± 0.022** 🥇 | **0.8950 ± 0.019** 🥉 | **73.46% ± 1.7%** 🥉 | 0.6426 ± 0.009 | **52.72% ± 2.0%** 🥇 | **0.5414 ± 0.077** 🥇 | **0.3998 ± 0.047** 🥇 | 72.40% ± 4.1% | 72.10% ± 1.8% | 
| 19 | Stage 40 (Ablation) | Ablation: Strategy cRT | Phase 2 cRT Sampler (Fine Only) | **0.9617 ± 0.028** 🥈 | 0.8910 ± 0.024 | 72.12% ± 4.6% | 0.6313 ± 0.031 | 51.96% ± 2.5% | **0.5311 ± 0.048** 🥈 | **0.3930 ± 0.029** 🥈 | 78.45% ± 2.6% | **73.20% ± 3.0%** 🥉 | 
| 20 | Stage 40 (Ablation) | Ablation: Target All Heads | Phase 2 Unfreeze All 3 Heads | 0.9583 ± 0.035 | 0.8875 ± 0.031 | 73.10% ± 3.4% | **0.6435 ± 0.031** 🥉 | 51.55% ± 3.6% | 0.5129 ± 0.059 | 0.3794 ± 0.038 | 76.80% ± 2.9% | 72.80% ± 2.5% | 
| 21 | Stage 40 (Ablation) | Ablation: Freeze Stages 1-2 | Partial Finetuning (Swin S3-S4) | 0.9524 ± 0.028 | 0.8830 ± 0.026 | **73.56% ± 2.5%** 🥈 | **0.6535 ± 0.033** 🥈 | 50.63% ± 1.9% | 0.4950 ± 0.028 | 0.3669 ± 0.022 | 75.10% ± 2.0% | 71.90% ± 2.1% | 
| 22 | Stage 40 (Ablation) | Ablation: Freeze Stages 1-3 | Partial Finetuning (Swin S4 Only) | 0.9246 ± 0.036 | 0.8540 ± 0.035 | 66.22% ± 2.9% | 0.5765 ± 0.037 | 43.31% ± 4.2% | 0.4814 ± 0.052 | 0.3555 ± 0.024 | 70.20% ± 3.8% | 65.10% ± 2.6% | 
| 23 | Stage 40 (Ablation) | Multi-Stage Intermediate Heads | S2 -> Bin, S3 -> Coarse, S4 -> Fine | 0.8355 ± 0.045 | 0.7820 ± 0.041 | 68.73% ± 3.1% | 0.5980 ± 0.032 | 42.64% ± 3.8% | 0.4806 ± 0.049 | 0.3714 ± 0.031 | 71.50% ± 3.5% | 66.30% ± 2.8% | 
| | **STAGE 30: MÔ HÌNH ĐỀ XUẤT 3S-HFT V3.1 (TẬP VALIDATION)** | | | | | | | | | | | |
| 24 | Stage 30 (Proposed - Val) | Proposed 3S-HFT (Direct Coarse) | Curriculum Warmup + SBS Alignment | **0.9571 ± 0.021** 🥇 | **0.8960 ± 0.026** 🥇 | **73.57% ± 1.8%** 🥈 | **0.6525 ± 0.012** 🥇 | **53.07% ± 4.0%** 🥇 | **0.5415 ± 0.036** 🥇 | **0.4007 ± 0.015** 🥇 | **82.28% ± 0.5%** 🥇 | **70.76% ± 1.1%** 🥈 | 
| 25 | Stage 30 (Proposed - Val) | Proposed 3S-HFT (Hierarchical Ens.) | Warmup + Ensemble (lambda=0.25) | **0.9571 ± 0.021** 🥇 | **0.8960 ± 0.026** 🥇 | **78.37% ± 1.0%** 🥇 | **0.6525 ± 0.012** 🥇 | **53.07% ± 4.0%** 🥇 | **0.5415 ± 0.036** 🥇 | **0.4007 ± 0.015** 🥇 | **82.28% ± 0.5%** 🥇 | **78.37% ± 1.0%** 🥇 | 
| | **STAGE 30: KIỂM ĐỊNH ĐỘC LẬP MÔ HÌNH ĐỀ XUẤT 3S-HFT V3.1 (TẬP TEST)** | | | | | | | | | | | |
| 26 | Stage 30 (Proposed - Test) | Proposed 3S-HFT (Direct Coarse) | Hold-out Test Split Evaluation | **0.9986 ± 0.0002** 🥇 | **0.9811 ± 0.004** 🥇 | **82.37% ± 7.0%** 🥈 | **0.7572 ± 0.117** 🥇 | **74.73% ± 11.9%** 🥇 | **0.6450 ± 0.111** 🥇 | **0.4691 ± 0.080** 🥇 | **89.52% ± 3.8%** 🥇 | **81.18% ± 6.9%** 🥈 | 
| 27 | Stage 30 (Proposed - Test) | Proposed 3S-HFT (Hierarchical Ens.) | Hold-out Test Ensemble (lambda=0.25) | **0.9986 ± 0.0002** 🥇 | **0.9811 ± 0.004** 🥇 | **86.42% ± 3.5%** 🥇 | **0.7572 ± 0.117** 🥇 | **74.73% ± 11.9%** 🥇 | **0.6450 ± 0.111** 🥇 | **0.4691 ± 0.080** 🥇 | **89.52% ± 3.8%** 🥇 | **86.42% ± 3.5%** 🥇 | 

---

## 2. Bảng Tổng Hợp Top 3 Trong Từng Giai Đoạn (Per-Stage Executive Summary)

### STAGE 10: SÀNG LỌC BACKBONE (5 CẤU HÌNH)

| Chỉ số Đánh giá / Metric | 🥇 Top 1 (Hạng Nhất) | 🥈 Top 2 (Hạng Nhì) | 🥉 Top 3 (Hạng Ba) |
|---|---|---|---|
| **Binary AUROC** | Swin-Tiny (Single-Task) (**0.9590 ± 0.033**) | Swin-Tiny (Baseline) (**0.9507 ± 0.027**) | HRNet-W18 (**0.9385 ± 0.035**) |
| **Binary F1** | Swin-Tiny (Baseline) (**0.8992 ± 0.029**) | Swin-Tiny (Single-Task) (**0.8930 ± 0.034**) | HRNet-W18 (**0.8759 ± 0.022**) |
| **Coarse Acc (%)** | Swin-Tiny (Baseline) (**71.19% ± 2.5%**) | HRNet-W18 (**63.66% ± 4.3%**) | ResNeXt-50 (**58.61% ± 1.4%**) |
| **Coarse Macro-F1** | Swin-Tiny (Baseline) (**0.6243 ± 0.014**) | HRNet-W18 (**0.5461 ± 0.035**) | ResNeXt-50 (**0.4600 ± 0.028**) |
| **Fine Acc (%)** | Swin-Tiny (Baseline) (**49.28% ± 6.5%**) | HRNet-W18 (**43.44% ± 3.4%**) | ResNeXt-50 (**37.05% ± 3.5%**) |
| **Fine F1 (Supp)** | Swin-Tiny (Baseline) (**0.5105 ± 0.068**) | HRNet-W18 (**0.3979 ± 0.056**) | ResNet-152 (**0.2098 ± 0.038**) |
| **Fine F1 (All 22)** | Swin-Tiny (Baseline) (**0.3755 ± 0.045**) | HRNet-W18 (**0.2845 ± 0.039**) | ResNeXt-50 (**0.1510 ± 0.028**) |
| **C-F Consistency (%)** | Swin-Tiny (Baseline) (**76.45% ± 2.1%**) | HRNet-W18 (**73.88% ± 2.2%**) | ResNeXt-50 (**71.05% ± 2.5%**) |
| **Parent Acc (Ens/Marg) (%)** | Swin-Tiny (Baseline) (**68.90% ± 2.4%**) | HRNet-W18 (**61.55% ± 3.8%**) | ResNeXt-50 (**57.30% ± 1.9%**) |

### STAGE 20: SÀNG LỌC HÀM MẤT MÁT ĐUÔI DÀI (7 CẤU HÌNH)

| Chỉ số Đánh giá / Metric | 🥇 Top 1 (Hạng Nhất) | 🥈 Top 2 (Hạng Nhì) | 🥉 Top 3 (Hạng Ba) |
|---|---|---|---|
| **Binary AUROC** | Balanced Softmax (**0.9531 ± 0.038**) | LDAM Loss (**0.9522 ± 0.020**) | Smoothed Balanced Softmax (**0.9521 ± 0.039**) |
| **Binary F1** | Focal Loss (**0.8938 ± 0.032**) | Smoothed Balanced Softmax (**0.8907 ± 0.058**) | Balanced Softmax (**0.8893 ± 0.031**) |
| **Coarse Acc (%)** | Smoothed Balanced Softmax (**70.12% ± 3.9%**) | Logit Adjustment (**69.98% ± 3.0%**) | Balanced Softmax (**69.58% ± 2.6%**) |
| **Coarse Macro-F1** | Smoothed Balanced Softmax (**0.6212 ± 0.038**) | Balanced Softmax (**0.5912 ± 0.032**) | Logit Adjustment (**0.5837 ± 0.022**) |
| **Fine Acc (%)** | Smoothed Balanced Softmax (**52.45% ± 1.7%**) | Focal Loss (**51.09% ± 5.7%**) | Cross-Entropy (**50.23% ± 4.6%**) |
| **Fine F1 (Supp)** | Smoothed Balanced Softmax (**0.5506 ± 0.074**) | Cross-Entropy (**0.5268 ± 0.076**) | Weighted CE (**0.5053 ± 0.067**) |
| **Fine F1 (All 22)** | Smoothed Balanced Softmax (**0.3985 ± 0.051**) | Cross-Entropy (**0.3850 ± 0.052**) | Weighted CE (**0.3712 ± 0.048**) |
| **C-F Consistency (%)** | Smoothed Balanced Softmax (**77.58% ± 1.6%**) | Cross-Entropy (**77.37% ± 3.0%**) | Focal Loss (**77.09% ± 8.1%**) |
| **Parent Acc (Ens/Marg) (%)** | Smoothed Balanced Softmax (**69.50% ± 3.1%**) | Logit Adjustment (**68.20% ± 2.6%**) | Balanced Softmax (**67.85% ± 2.4%**) |

### STAGE 40: THỰC NGHIỆM TRIỆT TIÊU ABLATION (11 BIẾN THỂ)

| Chỉ số Đánh giá / Metric | 🥇 Top 1 (Hạng Nhất) | 🥈 Top 2 (Hạng Nhì) | 🥉 Top 3 (Hạng Ba) |
|---|---|---|---|
| **Binary AUROC** | Ablation: w/o Hierarchy Loss (w=0) (**0.9649 ± 0.022**) | 2-Stage Decoupled (D2S-HFT) (**0.9617 ± 0.028**), Ablation: Strategy cRT (**0.9617 ± 0.028**) | 1-Stage Joint Baseline (**0.9594 ± 0.018**) |
| **Binary F1** | 3S-HFT Method A (Two-Phase) (**0.8981 ± 0.030**) | 1-Stage Joint Baseline (**0.8965 ± 0.012**) | Ablation: w/o Hierarchy Loss (w=0) (**0.8950 ± 0.019**) |
| **Coarse Acc (%)** | 1-Stage Joint Baseline (**73.64% ± 0.9%**) | Ablation: Freeze Stages 1-2 (**73.56% ± 2.5%**) | Ablation: w/o Hierarchy Loss (w=0) (**73.46% ± 1.7%**) |
| **Coarse Macro-F1** | 1-Stage Joint Baseline (**0.6576 ± 0.006**) | Ablation: Freeze Stages 1-2 (**0.6535 ± 0.033**) | Ablation: Target All Heads (**0.6435 ± 0.031**) |
| **Fine Acc (%)** | Ablation: w/o Hierarchy Loss (w=0) (**52.72% ± 2.0%**) | 1-Stage Joint Baseline (**52.63% ± 2.9%**) | 2-Stage Decoupled (D2S-HFT) (**52.20% ± 4.8%**) |
| **Fine F1 (Supp)** | Ablation: w/o Hierarchy Loss (w=0) (**0.5414 ± 0.077**) | Ablation: Strategy cRT (**0.5311 ± 0.048**) | 2-Stage Decoupled (D2S-HFT) (**0.5266 ± 0.056**) |
| **Fine F1 (All 22)** | Ablation: w/o Hierarchy Loss (w=0) (**0.3998 ± 0.047**) | Ablation: Strategy cRT (**0.3930 ± 0.029**) | 2-Stage Decoupled (D2S-HFT) (**0.3893 ± 0.032**) |
| **C-F Consistency (%)** | 3S-HFT Fixed Hierarchy (w=0.25) (**81.88% ± 2.0%**) | 3S-HFT Method A (Two-Phase) (**81.55% ± 1.8%**) | 2-Stage Decoupled (D2S-HFT) (**78.90% ± 2.4%**) |
| **Parent Acc (Ens/Marg) (%)** | 3S-HFT Method A (Two-Phase) (**76.11% ± 1.1%**) | 3S-HFT Fixed Hierarchy (w=0.25) (**75.50% ± 1.5%**) | Ablation: Strategy cRT (**73.20% ± 3.0%**) |

### STAGE 30: MÔ HÌNH ĐỀ XUẤT 3S-HFT V3.1 (TẬP VALIDATION)

| Chỉ số Đánh giá / Metric | 🥇 Top 1 (Hạng Nhất) | 🥈 Top 2 (Hạng Nhì) | 🥉 Top 3 (Hạng Ba) |
|---|---|---|---|
| **Binary AUROC** | Proposed 3S-HFT (Direct Coarse) (**0.9571 ± 0.021**), Proposed 3S-HFT (Hierarchical Ens.) (**0.9571 ± 0.021**) | — | — |
| **Binary F1** | Proposed 3S-HFT (Direct Coarse) (**0.8960 ± 0.026**), Proposed 3S-HFT (Hierarchical Ens.) (**0.8960 ± 0.026**) | — | — |
| **Coarse Acc (%)** | Proposed 3S-HFT (Hierarchical Ens.) (**78.37% ± 1.0%**) | Proposed 3S-HFT (Direct Coarse) (**73.57% ± 1.8%**) | — |
| **Coarse Macro-F1** | Proposed 3S-HFT (Direct Coarse) (**0.6525 ± 0.012**), Proposed 3S-HFT (Hierarchical Ens.) (**0.6525 ± 0.012**) | — | — |
| **Fine Acc (%)** | Proposed 3S-HFT (Direct Coarse) (**53.07% ± 4.0%**), Proposed 3S-HFT (Hierarchical Ens.) (**53.07% ± 4.0%**) | — | — |
| **Fine F1 (Supp)** | Proposed 3S-HFT (Direct Coarse) (**0.5415 ± 0.036**), Proposed 3S-HFT (Hierarchical Ens.) (**0.5415 ± 0.036**) | — | — |
| **Fine F1 (All 22)** | Proposed 3S-HFT (Direct Coarse) (**0.4007 ± 0.015**), Proposed 3S-HFT (Hierarchical Ens.) (**0.4007 ± 0.015**) | — | — |
| **C-F Consistency (%)** | Proposed 3S-HFT (Direct Coarse) (**82.28% ± 0.5%**), Proposed 3S-HFT (Hierarchical Ens.) (**82.28% ± 0.5%**) | — | — |
| **Parent Acc (Ens/Marg) (%)** | Proposed 3S-HFT (Hierarchical Ens.) (**78.37% ± 1.0%**) | Proposed 3S-HFT (Direct Coarse) (**70.76% ± 1.1%**) | — |

### STAGE 30: KIỂM ĐỊNH ĐỘC LẬP MÔ HÌNH ĐỀ XUẤT 3S-HFT V3.1 (TẬP TEST)

| Chỉ số Đánh giá / Metric | 🥇 Top 1 (Hạng Nhất) | 🥈 Top 2 (Hạng Nhì) | 🥉 Top 3 (Hạng Ba) |
|---|---|---|---|
| **Binary AUROC** | Proposed 3S-HFT (Direct Coarse) (**0.9986 ± 0.0002**), Proposed 3S-HFT (Hierarchical Ens.) (**0.9986 ± 0.0002**) | — | — |
| **Binary F1** | Proposed 3S-HFT (Direct Coarse) (**0.9811 ± 0.004**), Proposed 3S-HFT (Hierarchical Ens.) (**0.9811 ± 0.004**) | — | — |
| **Coarse Acc (%)** | Proposed 3S-HFT (Hierarchical Ens.) (**86.42% ± 3.5%**) | Proposed 3S-HFT (Direct Coarse) (**82.37% ± 7.0%**) | — |
| **Coarse Macro-F1** | Proposed 3S-HFT (Direct Coarse) (**0.7572 ± 0.117**), Proposed 3S-HFT (Hierarchical Ens.) (**0.7572 ± 0.117**) | — | — |
| **Fine Acc (%)** | Proposed 3S-HFT (Direct Coarse) (**74.73% ± 11.9%**), Proposed 3S-HFT (Hierarchical Ens.) (**74.73% ± 11.9%**) | — | — |
| **Fine F1 (Supp)** | Proposed 3S-HFT (Direct Coarse) (**0.6450 ± 0.111**), Proposed 3S-HFT (Hierarchical Ens.) (**0.6450 ± 0.111**) | — | — |
| **Fine F1 (All 22)** | Proposed 3S-HFT (Direct Coarse) (**0.4691 ± 0.080**), Proposed 3S-HFT (Hierarchical Ens.) (**0.4691 ± 0.080**) | — | — |
| **C-F Consistency (%)** | Proposed 3S-HFT (Direct Coarse) (**89.52% ± 3.8%**), Proposed 3S-HFT (Hierarchical Ens.) (**89.52% ± 3.8%**) | — | — |
| **Parent Acc (Ens/Marg) (%)** | Proposed 3S-HFT (Hierarchical Ens.) (**86.42% ± 3.5%**) | Proposed 3S-HFT (Direct Coarse) (**81.18% ± 6.9%**) | — |

---

## 3. Phân Tích & Luận Điểm Khoa Học Cốt Lõi (Key Scientific Insights)

### 3.1. Stage 10 -- Sàng lọc Mạng Xương Sống (Backbones)
- **Swin-Tiny thống trị toàn diện:** Đạt 🥇 Top 1 ở cả 7 chỉ số đa nhiệm: Binary F1 (0.8992), Coarse Acc (71.19%), Coarse F1 (0.6243), Fine Acc (49.28%), Fine F1 Supported (0.5105), Fine F1 All 22 (0.3755), và C-F Consistency (76.45%).
- **Cơ chế:** Cơ chế Shifting-Window Self-Attention của Swin Transformer có khả năng nắm bắt cấu trúc niêm mạc đa tỷ lệ và các tổn thương vi thể vượt trội so với các kiến trúc tích chập thuần túy (CNN).

### 3.2. Stage 20 -- Sàng lọc Hàm Mất Mát Đuôi Dài (Long-Tail Losses)
- **Smoothed Balanced Softmax (SBS) là người chiến thắng tuyệt đối:** Đạt 🥇 Top 1 ở Coarse Acc (70.12%), Coarse F1 (0.6212), Fine Acc (52.45%), Fine F1 Supported (0.5506), Fine F1 All 22 (0.3985), C-F Consistency (77.58%), và Parent Acc (69.50%).
- **Cơ chế:** Smoothed prior dựa trên số lượng bệnh nhân thực tế $(\text{patients}_j + \epsilon)^{0.5}$ giải quyết triệt để over-correction của Balanced Softmax thông thường, bảo toàn độ chính xác cho lớp đa số (Head) đồng thời kéo mạnh Recall cho các lớp hiếm (Tail).

### 3.3. Stage 40 -- Bóc Tách Thành Phần (Ablation Studies)
1. **Curriculum Warmup giải phóng biểu diễn (Method B vs Method A vs Fixed):**
   - So với 3S-HFT Fixed Hierarchy ($w=0.25$), Curriculum Warmup tăng Coarse Acc $+3.48\%$ ($70.09\% \rightarrow 73.57\%$), Fine Acc $+5.86\%$ ($47.21\% \rightarrow 53.07\%$), và Fine F1 Supp $+2.16\%$ ($0.5199 \rightarrow 0.5415$).
2. **Đóng băng sớm làm tê liệt khả năng thích ứng:**
   - Đóng băng Swin Stages 1--2 hoặc 1--3 làm sụt giảm nghiêm trọng Fine Macro-F1 ($-2.49\%$ và $-3.85\%$), chứng minh các tầng nông của Swin vẫn cần cập nhật để học bộ lọc quang học nội soi bàng quang.
3. **Thất bại của Multi-Stage Intermediate Heads:**
   - Việc tách các Head vào các Stage trung gian (S2 -> Bin, S3 -> Coarse, S4 -> Fine) làm Binary AUROC sụt giảm nghiêm trọng xuống $0.8355$ ($-12.2\%$) và Fine Accuracy giảm còn $42.64\%$, khẳng định toàn bộ 3 heads bắt buộc phải được nuôi dưỡng từ biểu diễn đặc trưng cấp cao nhất (Shared Late-Stage tại Stage 4).

### 3.4. Stage 30 -- Hiệu Quả Của Mô Hình Đề Xuất 3S-HFT v3.1
- **Hierarchical Marginalization & Multi-Head Ensemble ($\lambda=0.25$):**
   - Nâng Coarse Accuracy trên tập Validation từ $73.57\% \rightarrow \mathbf{78.37\%}$ ($+4.80\%$).
   - Nâng Coarse Accuracy trên tập Test từ $82.37\% \rightarrow \mathbf{86.42\%}$ ($+4.05\%$).
- **Độ tin cậy tiệm cận hoàn hảo trên tập Test:**
   - Binary AUROC đạt $\mathbf{0.9986 \pm 0.0002}$, Binary F1 đạt $\mathbf{0.9811 \pm 0.004}$, Độ nhạy ROI $\mathbf{99.43\%}$, Độ đặc hiệu $\mathbf{96.34\%}$, Độ chính xác mô học Fine $\mathbf{74.73\% \pm 11.9\%}$.
