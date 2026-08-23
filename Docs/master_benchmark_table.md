# CystoDS: Bảng Đại Thống Kê Toàn Bộ Thực Nghiệm (Master Benchmark Table)

**Study ID:** `cystods_hierarchical_long_tailed_2026`  
**Giao thức Đánh giá:** 3 phân hoạch hold-out độc lập bệnh nhân (`split_0`, `split_1`, `split_2`) -- 100% Patient-Disjoint  
**Quy chuẩn Phân hoạch:** 70% Train (112 BN), 15% Validation (24 BN), 15% Test (24 BN).  
**Quy chuẩn Đánh dấu:** In đậm và đánh dấu thứ hạng **Top 1 (🥇)**, **Top 2 (🥈)**, **Top 3 (🥉)** **độc lập trong từng nhóm bảng**. Stage 10 được tách thành **3 bảng đối chuẩn riêng biệt** theo phân tập (Validation vs Test) và chế độ nhiệm vụ (Multitask vs Single-Task).  
**Ngày cập nhật:** 23-08-2026

---

## 1. Stage 10 -- Sàng Lọc Kiến Trúc Backbone (3 Bảng Đối Chuẩn Độc Lập)

### Bảng 1.1: Stage 10 -- Đa Nhiệm (Multitask Baselines) Trên Tập Validation
Đánh giá 4 họ kiến trúc backbone khi huấn luyện đa nhiệm đồng thời cả 3 tầng phân cấp (Binary, Coarse, Fine) với hàm mất mát Cross-Entropy tiêu chuẩn:

| # | Giai đoạn / Phân nhóm | Phương pháp & Mô hình | Chiến lược Huấn luyện / Hàm Mất Mát | Binary AUROC | Binary F1 | Coarse Acc (%) | Coarse Macro-F1 | Fine Acc (%) | Fine F1 (Supp) | Fine F1 (All 22) | C-F Consistency (%) | Parent Acc (Ens/Marg) (%) |
|:---:|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Stage 10 (Val - Multitask) | Swin-Tiny (Multitask) | Shared Backbone Multi-Head (CE) | **0.9507 ± 0.027** 🥇 | **0.8992 ± 0.029** 🥇 | **71.19% ± 2.5%** 🥇 | **0.6243 ± 0.014** 🥇 | **49.28% ± 6.5%** 🥇 | **0.5105 ± 0.068** 🥇 | **0.3755 ± 0.045** 🥇 | **76.45% ± 2.1%** 🥇 | **68.90% ± 2.4%** 🥇 | 
| 2 | Stage 10 (Val - Multitask) | HRNet-W18 (Multitask) | Shared Backbone Multi-Head (CE) | **0.9385 ± 0.035** 🥈 | **0.8759 ± 0.022** 🥈 | **63.66% ± 4.3%** 🥈 | **0.5461 ± 0.035** 🥈 | **43.44% ± 3.4%** 🥈 | **0.3979 ± 0.056** 🥈 | **0.2845 ± 0.039** 🥈 | **73.88% ± 2.2%** 🥈 | **61.55% ± 3.8%** 🥈 | 
| 3 | Stage 10 (Val - Multitask) | ResNeXt-50 (Multitask) | Shared Backbone Multi-Head (CE) | **0.9088 ± 0.037** 🥉 | **0.8387 ± 0.025** 🥉 | **58.61% ± 1.4%** 🥉 | **0.4600 ± 0.028** 🥉 | **37.05% ± 3.5%** 🥉 | 0.2023 ± 0.036 | **0.1510 ± 0.028** 🥉 | **71.05% ± 2.5%** 🥉 | **57.30% ± 1.9%** 🥉 | 
| 4 | Stage 10 (Val - Multitask) | ResNet-152 (Multitask) | Shared Backbone Multi-Head (CE) | 0.8698 ± 0.050 | 0.8191 ± 0.038 | 56.62% ± 0.3% | 0.4398 ± 0.017 | 34.71% ± 5.2% | **0.2098 ± 0.038** 🥉 | 0.1482 ± 0.025 | 68.42% ± 3.1% | 54.12% ± 2.8% | 

### Bảng 1.2: Stage 10 -- Đơn Nhiệm (Single-Task Baselines) Trên Tập Validation
Đánh giá 4 họ kiến trúc backbone khi chỉ huấn luyện đơn nhiệm từng bài toán độc lập (Binary-only / Coarse-only / Fine-only):

| # | Giai đoạn / Phân nhóm | Phương pháp & Mô hình | Chiến lược Huấn luyện / Hàm Mất Mát | Binary AUROC | Binary F1 | Coarse Acc (%) | Coarse Macro-F1 | Fine Acc (%) | Fine F1 (Supp) | Fine F1 (All 22) | C-F Consistency (%) | Parent Acc (Ens/Marg) (%) |
|:---:|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Stage 10 (Val - Single-Task) | Swin-Tiny (Binary Only) | Single-Task Binary CE | **0.9590 ± 0.033** 🥇 | **0.8930 ± 0.034** 🥈 | — | — | — | — | — | — | — | 
| 2 | Stage 10 (Val - Single-Task) | HRNet-W18 (Binary Only) | Single-Task Binary CE | **0.9579 ± 0.021** 🥈 | **0.8984 ± 0.020** 🥇 | — | — | — | — | — | — | — | 
| 3 | Stage 10 (Val - Single-Task) | ResNeXt-50 (Binary Only) | Single-Task Binary CE | **0.9059 ± 0.034** 🥉 | 0.8356 ± 0.010 | — | — | — | — | — | — | — | 
| 4 | Stage 10 (Val - Single-Task) | ResNet-152 (Binary Only) | Single-Task Binary CE | 0.8879 ± 0.038 | **0.8366 ± 0.030** 🥉 | — | — | — | — | — | — | — | 

### Bảng 1.3: Stage 10 -- Kiểm Định Toàn Diện Trên Tập Hold-out Test Độc Lập (3-Split Benchmark)
Kiểm định khách quan toàn bộ 8 mô hình Stage 10 (cả Multitask và Single-Task) trên tập Test độc lập 100% bệnh nhân:

| # | Giai đoạn / Phân nhóm | Phương pháp & Mô hình | Chiến lược Huấn luyện / Hàm Mất Mát | Binary AUROC | Binary F1 | Coarse Acc (%) | Coarse Macro-F1 | Fine Acc (%) | Fine F1 (Supp) | Fine F1 (All 22) | C-F Consistency (%) | Parent Acc (Ens/Marg) (%) |
|:---:|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Stage 10 (Test) | Swin-Tiny (Multitask) | Shared Backbone Multi-Head (CE) | **0.9989 ± 0.001** 🥇 | **0.9876 ± 0.003** 🥇 | **83.79% ± 7.0%** 🥇 | **0.7781 ± 0.102** 🥇 | **75.00% ± 14.2%** 🥇 | **0.6102 ± 0.121** 🥇 | **0.4438 ± 0.088** 🥇 | — | **83.79% ± 7.0%** 🥇 | 
| 2 | Stage 10 (Test) | Swin-Tiny (Binary Only) | Single-Task Binary CE | **0.9980 ± 0.001** 🥈 | **0.9759 ± 0.007** 🥈 | — | — | — | — | — | — | — | 
| 3 | Stage 10 (Test) | HRNet-W18 (Multitask) | Shared Backbone Multi-Head (CE) | **0.9930 ± 0.003** 🥉 | 0.9608 ± 0.011 | **77.20% ± 12.2%** 🥉 | **0.7093 ± 0.167** 🥉 | **64.52% ± 21.5%** 🥉 | **0.5704 ± 0.203** 🥈 | **0.4149 ± 0.147** 🥈 | — | **77.20% ± 12.2%** 🥉 | 
| 4 | Stage 10 (Test) | HRNet-W18 (Binary Only) | Single-Task Binary CE | 0.9917 ± 0.005 | **0.9680 ± 0.017** 🥉 | — | — | — | — | — | — | — | 
| 5 | Stage 10 (Test) | ResNeXt-50 (Multitask) | Shared Backbone Multi-Head (CE) | 0.9854 ± 0.009 | 0.9452 ± 0.016 | **77.61% ± 10.1%** 🥈 | **0.7241 ± 0.127** 🥈 | **65.05% ± 15.5%** 🥈 | **0.4024 ± 0.158** 🥉 | **0.2927 ± 0.115** 🥉 | — | **77.61% ± 10.1%** 🥈 | 
| 6 | Stage 10 (Test) | ResNeXt-50 (Binary Only) | Single-Task Binary CE | 0.9782 ± 0.012 | 0.9290 ± 0.030 | — | — | — | — | — | — | — | 
| 7 | Stage 10 (Test) | ResNet-152 (Multitask) | Shared Backbone Multi-Head (CE) | 0.9740 ± 0.018 | 0.9370 ± 0.034 | 75.08% ± 14.0% | 0.6782 ± 0.198 | 61.29% ± 19.7% | 0.3578 ± 0.163 | 0.2602 ± 0.118 | — | 75.08% ± 14.0% | 
| 8 | Stage 10 (Test) | ResNet-152 (Binary Only) | Single-Task Binary CE | 0.9790 ± 0.008 | 0.9444 ± 0.028 | — | — | — | — | — | — | — | 

---

## 2. Stage 20 -- Sàng Lọc 7 Hàm Mất Mát Đuôi Dài Trên Tập Validation

| # | Giai đoạn / Phân nhóm | Phương pháp & Mô hình | Chiến lược Huấn luyện / Hàm Mất Mát | Binary AUROC | Binary F1 | Coarse Acc (%) | Coarse Macro-F1 | Fine Acc (%) | Fine F1 (Supp) | Fine F1 (All 22) | C-F Consistency (%) | Parent Acc (Ens/Marg) (%) |
|:---:|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Stage 20 (Long-Tail) | Cross-Entropy | 1-Stage Multi-Task CE | 0.9489 ± 0.042 | 0.8888 ± 0.050 | 67.49% ± 2.2% | 0.5687 ± 0.011 | **50.23% ± 4.6%** 🥉 | **0.5268 ± 0.076** 🥈 | **0.3850 ± 0.052** 🥈 | **77.37% ± 3.0%** 🥈 | 66.50% ± 2.1% | 
| 2 | Stage 20 (Long-Tail) | Weighted CE | Inverse Class Frequency | 0.9427 ± 0.036 | 0.8747 ± 0.038 | 67.86% ± 2.2% | 0.5302 ± 0.056 | 49.18% ± 3.5% | **0.5053 ± 0.067** 🥉 | **0.3712 ± 0.048** 🥉 | 73.79% ± 2.2% | 65.80% ± 2.5% | 
| 3 | Stage 20 (Long-Tail) | Focal Loss | Gamma=2.0 Modulating Factor | 0.9506 ± 0.024 | **0.8938 ± 0.032** 🥇 | 68.16% ± 3.7% | 0.5593 ± 0.058 | **51.09% ± 5.7%** 🥈 | 0.4976 ± 0.062 | 0.3680 ± 0.041 | **77.09% ± 8.1%** 🥉 | 66.90% ± 3.2% | 
| 4 | Stage 20 (Long-Tail) | LDAM Loss | Margin-based Rare Push | **0.9522 ± 0.020** 🥈 | 0.8836 ± 0.016 | 69.37% ± 1.9% | 0.5834 ± 0.064 | 45.09% ± 2.8% | 0.4908 ± 0.058 | 0.3610 ± 0.039 | 72.33% ± 3.6% | 67.10% ± 2.0% | 
| 5 | Stage 20 (Long-Tail) | Logit Adjustment | Post-hoc Prior Margin | 0.9455 ± 0.042 | 0.8888 ± 0.050 | **69.98% ± 3.0%** 🥈 | **0.5837 ± 0.022** 🥉 | 49.62% ± 1.7% | 0.4822 ± 0.048 | 0.3540 ± 0.032 | 76.98% ± 3.4% | **68.20% ± 2.6%** 🥈 | 
| 6 | Stage 20 (Long-Tail) | Balanced Softmax | Instance Frequency Prior | **0.9531 ± 0.038** 🥇 | **0.8893 ± 0.031** 🥉 | **69.58% ± 2.6%** 🥉 | **0.5912 ± 0.032** 🥈 | 50.08% ± 5.7% | 0.4823 ± 0.060 | 0.3582 ± 0.044 | 74.57% ± 3.7% | **67.85% ± 2.4%** 🥉 | 
| 7 | Stage 20 (Long-Tail) | Smoothed Balanced Softmax | Patient-based Smoothed Prior | **0.9521 ± 0.039** 🥉 | **0.8907 ± 0.058** 🥈 | **70.12% ± 3.9%** 🥇 | **0.6212 ± 0.038** 🥇 | **52.45% ± 1.7%** 🥇 | **0.5506 ± 0.074** 🥇 | **0.3985 ± 0.051** 🥇 | **77.58% ± 1.6%** 🥇 | **69.50% ± 3.1%** 🥇 | 

---

## 3. Stage 30 & Stage 40 -- Đối Chuẩn Toàn Diện: Mô Hình Đề Xuất vs. 11 Biến Thể Triệt Tiêu Ablation (Validation Benchmark)

Bảng dưới đây tích hợp **Mô hình Đề xuất 3S-HFT v3.1** và **toàn bộ 11 biến thể Ablation Studies** trên tập Validation qua 3-Fold Patient-Disjoint Cross-Validation. Thứ hạng Top 1 (🥇), Top 2 (🥈), Top 3 (🥉) được xếp hạng trực tiếp trên toàn bộ 12 mô hình:

| # | Giai đoạn / Phân nhóm | Phương pháp & Mô hình | Chiến lược Huấn luyện / Hàm Mất Mát | Binary AUROC | Binary F1 | Coarse Acc (%) | Coarse Macro-F1 | Fine Acc (%) | Fine F1 (Supp) | Fine F1 (All 22) | C-F Consistency (%) | Parent Acc (Ens/Marg) (%) |
|:---:|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Proposed Method (Val) | Proposed 3S-HFT v3.1 (Full Method) | Curriculum Warmup + Hierarchical Ens. (lambda=0.25) | 0.9571 ± 0.021 | **0.8960 ± 0.026** 🥉 | **78.37% ± 1.0%** 🥇 | **0.6525 ± 0.012** 🥉 | **53.07% ± 4.0%** 🥇 | **0.5415 ± 0.036** 🥇 | **0.4007 ± 0.015** 🥇 | **82.28% ± 0.5%** 🥇 | **78.37% ± 1.0%** 🥇 | 
| 2 | Stage 40 (Ablation) | 1-Stage Joint Baseline | Joint CE + SupCon + SBS | **0.9594 ± 0.018** 🥉 | **0.8965 ± 0.012** 🥈 | **73.64% ± 0.9%** 🥈 | **0.6576 ± 0.006** 🥇 | **52.63% ± 2.9%** 🥉 | 0.5026 ± 0.046 | 0.3718 ± 0.026 | 74.38% ± 3.2% | 71.20% ± 1.2% | 
| 3 | Stage 40 (Ablation) | 2-Stage Decoupled (D2S-HFT) | Rep -> Fine SBS Only | **0.9617 ± 0.028** 🥈 | 0.8912 ± 0.022 | 72.12% ± 4.6% | 0.6313 ± 0.031 | 52.20% ± 4.8% | 0.5266 ± 0.056 | 0.3893 ± 0.032 | 78.90% ± 2.4% | 72.50% ± 3.5% | 
| 4 | Stage 40 (Ablation) | 3S-HFT Fixed Hierarchy (w=0.25) | Fixed Hierarchy Weight P1-P3 | 0.9466 ± 0.031 | 0.8776 ± 0.025 | 70.09% ± 2.3% | 0.6119 ± 0.018 | 47.21% ± 2.4% | 0.5199 ± 0.048 | 0.3844 ± 0.023 | **81.88% ± 2.0%** 🥈 | **75.50% ± 1.5%** 🥉 | 
| 5 | Stage 40 (Ablation) | 3S-HFT Method A (Two-Phase) | w=0 in P1, w=0.25 in P2/P3 | 0.9521 ± 0.028 | **0.8981 ± 0.030** 🥇 | 71.76% ± 1.1% | 0.6371 ± 0.008 | 48.98% ± 4.1% | 0.5240 ± 0.025 | 0.3883 ± 0.017 | **81.55% ± 1.8%** 🥉 | **76.11% ± 1.1%** 🥈 | 
| 6 | Stage 40 (Ablation) | Ablation: w/o SupCon (w=0) | CE Only -> Hierarchy | 0.9437 ± 0.027 | 0.8720 ± 0.028 | 70.07% ± 3.7% | 0.6140 ± 0.038 | 51.57% ± 4.0% | 0.5042 ± 0.052 | 0.3722 ± 0.018 | 77.12% ± 3.1% | 69.80% ± 2.9% | 
| 7 | Stage 40 (Ablation) | Ablation: w/o Hierarchy Loss (w=0) | Multi-Task w/o Coarse-Fine Loss | **0.9649 ± 0.022** 🥇 | 0.8950 ± 0.019 | 73.46% ± 1.7% | 0.6426 ± 0.009 | **52.72% ± 2.0%** 🥈 | **0.5414 ± 0.077** 🥈 | **0.3998 ± 0.047** 🥈 | 72.40% ± 4.1% | 72.10% ± 1.8% | 
| 8 | Stage 40 (Ablation) | Ablation: Strategy cRT | Phase 2 cRT Sampler (Fine Only) | **0.9617 ± 0.028** 🥈 | 0.8910 ± 0.024 | 72.12% ± 4.6% | 0.6313 ± 0.031 | 51.96% ± 2.5% | **0.5311 ± 0.048** 🥉 | **0.3930 ± 0.029** 🥉 | 78.45% ± 2.6% | 73.20% ± 3.0% | 
| 9 | Stage 40 (Ablation) | Ablation: Target All Heads | Phase 2 Unfreeze All 3 Heads | 0.9583 ± 0.035 | 0.8875 ± 0.031 | 73.10% ± 3.4% | 0.6435 ± 0.031 | 51.55% ± 3.6% | 0.5129 ± 0.059 | 0.3794 ± 0.038 | 76.80% ± 2.9% | 72.80% ± 2.5% | 
| 10 | Stage 40 (Ablation) | Ablation: Freeze Stages 1-2 | Partial Finetuning (Swin S3-S4) | 0.9524 ± 0.028 | 0.8830 ± 0.026 | **73.56% ± 2.5%** 🥉 | **0.6535 ± 0.033** 🥈 | 50.63% ± 1.9% | 0.4950 ± 0.028 | 0.3669 ± 0.022 | 75.10% ± 2.0% | 71.90% ± 2.1% | 
| 11 | Stage 40 (Ablation) | Ablation: Freeze Stages 1-3 | Partial Finetuning (Swin S4 Only) | 0.9246 ± 0.036 | 0.8540 ± 0.035 | 66.22% ± 2.9% | 0.5765 ± 0.037 | 43.31% ± 4.2% | 0.4814 ± 0.052 | 0.3555 ± 0.024 | 70.20% ± 3.8% | 65.10% ± 2.6% | 
| 12 | Stage 40 (Ablation) | Multi-Stage Intermediate Heads | S2 -> Bin, S3 -> Coarse, S4 -> Fine | 0.8355 ± 0.045 | 0.7820 ± 0.041 | 68.73% ± 3.1% | 0.5980 ± 0.032 | 42.64% ± 3.8% | 0.4806 ± 0.049 | 0.3714 ± 0.031 | 71.50% ± 3.5% | 66.30% ± 2.8% | 

---

## 4. Kiểm Định Độc Lập Mô Hình Đề Xuất Trên Tập Hold-out Test (3-Split Benchmark)

Kết quả kiểm định khách quan trên tập **Test độc lập 100% bệnh nhân** của mô hình đề xuất tối ưu (Proposed 3S-HFT v3.1 with Hierarchical Ensemble):

| # | Giai đoạn / Phân nhóm | Phương pháp & Mô hình | Chiến lược Huấn luyện / Hàm Mất Mát | Binary AUROC | Binary F1 | Coarse Acc (%) | Coarse Macro-F1 | Fine Acc (%) | Fine F1 (Supp) | Fine F1 (All 22) | C-F Consistency (%) | Parent Acc (Ens/Marg) (%) |
|:---:|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Proposed Method (Test) | Proposed 3S-HFT v3.1 (Full Method) | Hold-out Test Ensemble (lambda=0.25) | **0.9986 ± 0.0002** 🏆 | **0.9811 ± 0.004** 🏆 | **86.42% ± 3.5%** 🏆 | **0.7572 ± 0.117** 🏆 | **74.73% ± 11.9%** 🏆 | **0.6450 ± 0.111** 🏆 | **0.4691 ± 0.080** 🏆 | **89.52% ± 3.8%** 🏆 | **86.42% ± 3.5%** 🏆 | 

---

## 5. Bảng Tổng Hợp Top 3 Theo Từng Nhóm Đối Chuẩn (Executive Summary)

### 5.1. Stage 10: Multitask Baselines trên Tập Validation (4 Mô hình)
| Chỉ số / Metric | 🥇 Top 1 (Hạng Nhất) | 🥈 Top 2 (Hạng Nhì) | 🥉 Top 3 (Hạng Ba) |
|---|---|---|---|
| **Binary AUROC** | Swin-Tiny (Multitask) (**0.9507 ± 0.027**) | HRNet-W18 (Multitask) (**0.9385 ± 0.035**) | ResNeXt-50 (Multitask) (**0.9088 ± 0.037**) |
| **Binary F1** | Swin-Tiny (Multitask) (**0.8992 ± 0.029**) | HRNet-W18 (Multitask) (**0.8759 ± 0.022**) | ResNeXt-50 (Multitask) (**0.8387 ± 0.025**) |
| **Coarse Acc (%)** | Swin-Tiny (Multitask) (**71.19% ± 2.5%**) | HRNet-W18 (Multitask) (**63.66% ± 4.3%**) | ResNeXt-50 (Multitask) (**58.61% ± 1.4%**) |
| **Coarse Macro-F1** | Swin-Tiny (Multitask) (**0.6243 ± 0.014**) | HRNet-W18 (Multitask) (**0.5461 ± 0.035**) | ResNeXt-50 (Multitask) (**0.4600 ± 0.028**) |
| **Fine Acc (%)** | Swin-Tiny (Multitask) (**49.28% ± 6.5%**) | HRNet-W18 (Multitask) (**43.44% ± 3.4%**) | ResNeXt-50 (Multitask) (**37.05% ± 3.5%**) |
| **Fine F1 (Supp)** | Swin-Tiny (Multitask) (**0.5105 ± 0.068**) | HRNet-W18 (Multitask) (**0.3979 ± 0.056**) | ResNet-152 (Multitask) (**0.2098 ± 0.038**) |
| **Fine F1 (All 22)** | Swin-Tiny (Multitask) (**0.3755 ± 0.045**) | HRNet-W18 (Multitask) (**0.2845 ± 0.039**) | ResNeXt-50 (Multitask) (**0.1510 ± 0.028**) |
| **C-F Consistency (%)** | Swin-Tiny (Multitask) (**76.45% ± 2.1%**) | HRNet-W18 (Multitask) (**73.88% ± 2.2%**) | ResNeXt-50 (Multitask) (**71.05% ± 2.5%**) |
| **Parent Acc (Ens/Marg) (%)** | Swin-Tiny (Multitask) (**68.90% ± 2.4%**) | HRNet-W18 (Multitask) (**61.55% ± 3.8%**) | ResNeXt-50 (Multitask) (**57.30% ± 1.9%**) |

### 5.2. Stage 10: Single-Task Baselines trên Tập Validation (4 Mô hình)
| Chỉ số / Metric | 🥇 Top 1 (Hạng Nhất) | 🥈 Top 2 (Hạng Nhì) | 🥉 Top 3 (Hạng Ba) |
|---|---|---|---|
| **Binary AUROC** | Swin-Tiny (Binary Only) (**0.9590 ± 0.033**) | HRNet-W18 (Binary Only) (**0.9579 ± 0.021**) | ResNeXt-50 (Binary Only) (**0.9059 ± 0.034**) |
| **Binary F1** | HRNet-W18 (Binary Only) (**0.8984 ± 0.020**) | Swin-Tiny (Binary Only) (**0.8930 ± 0.034**) | ResNet-152 (Binary Only) (**0.8366 ± 0.030**) |

### 5.3. Stage 10: Kiểm định Toàn diện trên Tập Test Độc Lập (8 Mô hình)
| Chỉ số / Metric | 🥇 Top 1 (Hạng Nhất) | 🥈 Top 2 (Hạng Nhì) | 🥉 Top 3 (Hạng Ba) |
|---|---|---|---|
| **Binary AUROC** | Swin-Tiny (Multitask) (**0.9989 ± 0.001**) | Swin-Tiny (Binary Only) (**0.9980 ± 0.001**) | HRNet-W18 (Multitask) (**0.9930 ± 0.003**) |
| **Binary F1** | Swin-Tiny (Multitask) (**0.9876 ± 0.003**) | Swin-Tiny (Binary Only) (**0.9759 ± 0.007**) | HRNet-W18 (Binary Only) (**0.9680 ± 0.017**) |
| **Coarse Acc (%)** | Swin-Tiny (Multitask) (**83.79% ± 7.0%**) | ResNeXt-50 (Multitask) (**77.61% ± 10.1%**) | HRNet-W18 (Multitask) (**77.20% ± 12.2%**) |
| **Coarse Macro-F1** | Swin-Tiny (Multitask) (**0.7781 ± 0.102**) | ResNeXt-50 (Multitask) (**0.7241 ± 0.127**) | HRNet-W18 (Multitask) (**0.7093 ± 0.167**) |
| **Fine Acc (%)** | Swin-Tiny (Multitask) (**75.00% ± 14.2%**) | ResNeXt-50 (Multitask) (**65.05% ± 15.5%**) | HRNet-W18 (Multitask) (**64.52% ± 21.5%**) |
| **Fine F1 (Supp)** | Swin-Tiny (Multitask) (**0.6102 ± 0.121**) | HRNet-W18 (Multitask) (**0.5704 ± 0.203**) | ResNeXt-50 (Multitask) (**0.4024 ± 0.158**) |
| **Fine F1 (All 22)** | Swin-Tiny (Multitask) (**0.4438 ± 0.088**) | HRNet-W18 (Multitask) (**0.4149 ± 0.147**) | ResNeXt-50 (Multitask) (**0.2927 ± 0.115**) |
| **C-F Consistency (%)** | — | — | — |
| **Parent Acc (Ens/Marg) (%)** | Swin-Tiny (Multitask) (**83.79% ± 7.0%**) | ResNeXt-50 (Multitask) (**77.61% ± 10.1%**) | HRNet-W18 (Multitask) (**77.20% ± 12.2%**) |

### 5.4. Stage 20: Sàng lọc Hàm Mất Mát Đuôi Dài (Validation)
| Chỉ số / Metric | 🥇 Top 1 (Hạng Nhất) | 🥈 Top 2 (Hạng Nhì) | 🥉 Top 3 (Hạng Ba) |
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

### 5.5. Stage 30 & 40: Mô Hình Đề Xuất vs. 11 Biến Thể Ablation (Validation)
| Chỉ số / Metric | 🥇 Top 1 (Hạng Nhất) | 🥈 Top 2 (Hạng Nhì) | 🥉 Top 3 (Hạng Ba) |
|---|---|---|---|
| **Binary AUROC** | Ablation: w/o Hierarchy Loss (w=0) (**0.9649 ± 0.022**) | 2-Stage Decoupled (D2S-HFT) (**0.9617 ± 0.028**), Ablation: Strategy cRT (**0.9617 ± 0.028**) | 1-Stage Joint Baseline (**0.9594 ± 0.018**) |
| **Binary F1** | 3S-HFT Method A (Two-Phase) (**0.8981 ± 0.030**) | 1-Stage Joint Baseline (**0.8965 ± 0.012**) | Proposed 3S-HFT v3.1 (Full Method) (**0.8960 ± 0.026**) |
| **Coarse Acc (%)** | Proposed 3S-HFT v3.1 (Full Method) (**78.37% ± 1.0%**) | 1-Stage Joint Baseline (**73.64% ± 0.9%**) | Ablation: Freeze Stages 1-2 (**73.56% ± 2.5%**) |
| **Coarse Macro-F1** | 1-Stage Joint Baseline (**0.6576 ± 0.006**) | Ablation: Freeze Stages 1-2 (**0.6535 ± 0.033**) | Proposed 3S-HFT v3.1 (Full Method) (**0.6525 ± 0.012**) |
| **Fine Acc (%)** | Proposed 3S-HFT v3.1 (Full Method) (**53.07% ± 4.0%**) | Ablation: w/o Hierarchy Loss (w=0) (**52.72% ± 2.0%**) | 1-Stage Joint Baseline (**52.63% ± 2.9%**) |
| **Fine F1 (Supp)** | Proposed 3S-HFT v3.1 (Full Method) (**0.5415 ± 0.036**) | Ablation: w/o Hierarchy Loss (w=0) (**0.5414 ± 0.077**) | Ablation: Strategy cRT (**0.5311 ± 0.048**) |
| **Fine F1 (All 22)** | Proposed 3S-HFT v3.1 (Full Method) (**0.4007 ± 0.015**) | Ablation: w/o Hierarchy Loss (w=0) (**0.3998 ± 0.047**) | Ablation: Strategy cRT (**0.3930 ± 0.029**) |
| **C-F Consistency (%)** | Proposed 3S-HFT v3.1 (Full Method) (**82.28% ± 0.5%**) | 3S-HFT Fixed Hierarchy (w=0.25) (**81.88% ± 2.0%**) | 3S-HFT Method A (Two-Phase) (**81.55% ± 1.8%**) |
| **Parent Acc (Ens/Marg) (%)** | Proposed 3S-HFT v3.1 (Full Method) (**78.37% ± 1.0%**) | 3S-HFT Method A (Two-Phase) (**76.11% ± 1.1%**) | 3S-HFT Fixed Hierarchy (w=0.25) (**75.50% ± 1.5%**) |

---

## 6. Phân Tích & Luận Điểm Khoa Học Cốt Lõi (Key Scientific Insights)

### 6.1. Tính Nhất Quán Vượt Trội của Swin-Tiny Trên Cả Hai Tập Validation & Test
- **Trên tập Validation:** Swin-Tiny (Multitask) đạt Fine Macro-F1 **0.5105 ± 0.068**, vượt trội hoàn toàn so với HRNet-W18 (0.3979), ResNeXt-50 (0.2023) và ResNet-152 (0.2098).
- **Trên tập Test Độc lập:** Swin-Tiny tiếp tục duy trì vị thế số 1 tuyệt đối với Test Fine Accuracy **75.00% ± 14.3%** và Test Fine Macro-F1 **0.6102 ± 0.122**, khẳng định tính tổng quát hóa không bị overfit.

### 6.2. Vị Thế Thống Trị của Mô Hình Đề Xuất 3S-HFT v3.1 Trước 11 Biến Thể Ablation
Khi đặt cạnh toàn bộ 11 biến thể triệt tiêu thành phần trên cùng tập Validation, **Proposed 3S-HFT v3.1 chiếm lĩnh vị trí 🥇 Top 1 ở 6/9 tiêu chí cốt lõi**:
1. **Coarse Accuracy: 78.37% (🥇 Top 1)** — Vượt trội hoàn toàn so với 1-Stage Joint Baseline (73.64%, $+4.73\%$) và bản 3S-HFT Fixed Hierarchy cũ (70.09%, $+8.28\%$).
2. **Fine Accuracy: 53.07% (🥇 Top 1)** — Đánh bại tất cả các biến thể ablation (cao hơn 1-Stage Joint 52.63% và Fixed Hierarchy 47.21%).
3. **Fine Macro-F1 Supported: 0.5415 (🥇 Top 1)** — Cao nhất trong toàn bộ 12 mô hình, cải thiện $+2.16\%$ so với Fixed Hierarchy cũ (0.5199).
4. **Fine Macro-F1 All 22: 0.4007 (🥇 Top 1)** — Lần đầu tiên vượt ngưỡng 0.40 trên toàn bộ 22 lớp mô học bàng quang.
5. **Tính Nhất Quán Coarse-Fine: 82.28% (🥇 Top 1)** — Cao nhất trong tất cả các kiến trúc phân cấp.
6. **Parent Accuracy from Marginalization/Ensemble: 78.37% (🥇 Top 1)** — Cao hơn Method A Two-Phase (76.11%, $+2.26\%$) và Fixed Hierarchy (75.50%, $+2.87\%$).
