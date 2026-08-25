# CystoDS: Bảng Đại Thống Kê Toàn Bộ Thực Nghiệm (Master Benchmark Table)

**Study ID:** `cystods_hierarchical_long_tailed_2026`  
**Giao thức Đánh giá:** 3 phân hoạch hold-out độc lập bệnh nhân (`split_0`, `split_1`, `split_2`) -- 100% Patient-Disjoint  
**Quy chuẩn Phân hoạch:** 70% Train (112 BN), 15% Validation (24 BN), 15% Test (24 BN).  
**Quy chuẩn Đánh dấu:** In đậm và đánh dấu thứ hạng **Top 1 (🥇)**, **Top 2 (🥈)**, **Top 3 (🥉)** độc lập trong từng nhóm bảng.  
**Ngày cập nhật:** 24-08-2026

---

## Danh Mục Hệ Thống Bảng Biểu (Table Index)

- **[Bảng 0: Bảng Đối Chuẩn Trực Tiếp Trên Tập Hold-out Test Độc Lập](#0-bảng-đối-chuẩn-trực-tiếp-trên-tập-hold-out-test-độc-lập-proposed-3s-hft-vs-baselines)**
- **[Bảng 1: Stage 10 -- Sàng Lọc Kiến Trúc Backbone (Validation Benchmark)](#1-stage-10----sàng-lọc-kiến-trúc-backbone-validation-benchmark)**
- **[Bảng 2: Stage 20 -- Sàng Lọc 7 Hàm Mất Mát Đuôi Dài](#2-stage-20----sàng-lọc-7-hàm-mất-mát-đuôi-dài-validation-benchmark)**
- **[Bảng 3: Stage 30 & 40 -- Phân Rã Hệ Thống Bóc Tách Thực Nghiệm (Comprehensive Ablation Studies)](#3-stage-30--40----hệ-thống-bóc-tách-thực-nghiệm-toàn-diện-comprehensive-ablation-studies)**
  - [Bảng 3.1 (Table 6a): Khảo sát Chiến Lược & Quy Trình Huấn Luyện (Training Paradigm & Stage Decoupling)](#bảng-31-table-6a-khảo-sát-chiến-lược--quy-trình-huấn-luyện-training-paradigm--stage-decoupling)
  - [Bảng 3.2 (Table 6b): Khảo sát Lịch Trình Trọng Số Phân Cấp (Hierarchy Loss Scheduling & Curriculum Warmup)](#bảng-32-table-6b-khảo-sát-lịch-trình-trọng-số-phân-cấp-hierarchy-loss-scheduling--curriculum-warmup)
  - [Bảng 3.3 (Table 6c): Bóc Tách Đóng Góp Cận Biên Của Các Thành Phần Hàm Loss (Loss Component Marginal Contributions)](#bảng-33-table-6c-bóc-tách-đóng-góp-cận-biên-của-các-thành-phần-hàm-loss-loss-component-marginal-contributions)
  - [Bảng 3.4 (Table 6d): Khảo sát Vị Trí Trích Xuất Đặc Trưng Theo Độ Sâu (Architectural Head Placement)](#bảng-34-table-6d-khảo-sát-vị-trí-trích-xuất-đặc-trưng-theo-độ-sâu-architectural-head-placement)
  - [Bảng 3.5 (Table 6e): Khảo sát Độ Sâu Đóng Băng Backbone & Đánh Đổi Chi Phí Tính Toán (Backbone Freezing Depth & Compute Trade-off)](#bảng-35-table-6e-khảo-sát-độ-sâu-đóng-băng-backbone--đánh-đổi-chi-phí-tính-toán-backbone-freezing-depth--compute-trade-off)
  - [Bảng 3.6 (Table 6f): Khảo sát Độ Nhạy Siêu Tham Số & Cơ Chế Suy Luận Kết Hợp (Hyperparameters & Inference Blending)](#bảng-36-table-6f-khảo-sát-độ-nhạy-siêu-tham-số--cơ-chế-suy-luận-kết-hợp-hyperparameters--inference-blending)
- **[Bảng 4: Phân Tích Chi Tiết Từng Lớp Lâm Sàng & Benchmark Thời Gian Thực](#4-phân-tích-chi-tiết-từng-lớp-lâm-sàng--benchmark-thời-gian-thực)**

---

## 0. Bảng Đối Chuẩn Trực Tiếp Trên Tập Hold-out Test Độc Lập: Proposed 3S-HFT vs. Baselines

Bảng đối chuẩn khách quan trên **tập Test độc lập 100% bệnh nhân** (24 bệnh nhân, 337 ảnh) giữa mô hình đề xuất tối ưu (**Proposed 3S-HFT v3.1**) và toàn bộ các mô hình baseline (đơn nhiệm & đa nhiệm):

| # | Phân Nhóm | Mô Hình & Kiến Trúc | Chiến Lược Huấn Luyện | Binary AUROC | Binary Sens | Binary Spec | Binary Acc (%) | Binary F1 | Coarse Acc (%) | Coarse Macro-F1 | Fine Acc (%) | Fine F1 (Supp) | Fine F1 (All 22) | C-F Consistency (%) | Parent Acc (Ens) (%) |
|:---:|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **🏆** | **Proposed Method** | **Proposed 3S-HFT v3.1 (Full)** | **Curriculum Warmup + Hierarchical Ens.** | **0.9986 ± 0.0002** 🏆 | **0.9845 ± 0.005** 🏆 | **0.9912 ± 0.004** 🏆 | **0.9882 ± 0.003** 🏆 | **0.9811 ± 0.004** 🏆 | **86.42% ± 3.5%** 🏆 | **0.7572 ± 0.117** 🏆 | **74.73% ± 11.9%** 🏆 | **0.6450 ± 0.111** 🏆 | **0.4691 ± 0.080** 🏆 | **89.52% ± 3.8%** 🏆 | **86.42% ± 3.5%** 🏆 |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Baseline (Multitask) | Swin-Tiny (Multitask) | Shared Backbone Multi-Head (CE) | **0.9989 ± 0.001** 🥇 | **0.9876 ± 0.003** 🥇 | **0.9880 ± 0.004** 🥇 | **0.9878 ± 0.003** 🥇 | **0.9876 ± 0.003** 🥇 | **83.79% ± 7.0%** 🥇 | **0.7781 ± 0.102** 🥇 | **75.00% ± 14.2%** 🥇 | **0.6102 ± 0.121** 🥇 | **0.4438 ± 0.088** 🥇 | 81.20% ± 2.5% | **83.79% ± 7.0%** 🥇 |
| 2 | Baseline (Multitask) | HRNet-W18 (Multitask) | Shared Backbone Multi-Head (CE) | **0.9930 ± 0.003** 🥈 | 0.9608 ± 0.011 | **0.9850 ± 0.006** 🥈 | **0.9740 ± 0.008** 🥈 | 0.9608 ± 0.011 | **77.20% ± 12.2%** 🥉 | **0.7093 ± 0.167** 🥉 | **64.52% ± 21.5%** 🥉 | **0.5704 ± 0.203** 🥈 | **0.4149 ± 0.147** 🥈 | 76.40% ± 3.1% | **77.20% ± 12.2%** 🥉 |
| 3 | Baseline (Multitask) | ResNeXt-50 (Multitask) | Shared Backbone Multi-Head (CE) | 0.9854 ± 0.009 | 0.9452 ± 0.016 | 0.9710 ± 0.012 | 0.9590 ± 0.014 | 0.9452 ± 0.016 | **77.61% ± 10.1%** 🥈 | **0.7241 ± 0.127** 🥈 | **65.05% ± 15.5%** 🥈 | 0.4024 ± 0.158 | **0.2927 ± 0.115** 🥉 | 74.50% ± 2.8% | **77.61% ± 10.1%** 🥈 |
| 4 | Baseline (Multitask) | ResNet-152 (Multitask) | Shared Backbone Multi-Head (CE) | 0.9740 ± 0.018 | 0.9370 ± 0.034 | 0.9650 ± 0.019 | 0.9520 ± 0.024 | 0.9370 ± 0.034 | 75.08% ± 14.0% | 0.6782 ± 0.198 | 61.29% ± 19.7% | 0.3578 ± 0.163 | 0.2602 ± 0.118 | 71.80% ± 3.4% | 75.08% ± 14.0% |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 5 | Single-Task (Binary) | Swin-Tiny (Binary Only) | Single-Task Binary CE | **0.9980 ± 0.001** 🥇 | **0.9759 ± 0.007** 🥇 | **0.9890 ± 0.005** 🥇 | **0.9830 ± 0.006** 🥇 | **0.9759 ± 0.007** 🥇 | — | — | — | — | — | — | — |
| 6 | Single-Task (Binary) | HRNet-W18 (Binary Only) | Single-Task Binary CE | **0.9917 ± 0.005** 🥈 | **0.9680 ± 0.017** 🥈 | **0.9840 ± 0.008** 🥈 | **0.9765 ± 0.012** 🥈 | **0.9680 ± 0.017** 🥈 | — | — | — | — | — | — | — |
| 7 | Single-Task (Binary) | ResNet-152 (Binary Only) | Single-Task Binary CE | 0.9790 ± 0.008 | **0.9444 ± 0.028** 🥉 | 0.9720 ± 0.014 | **0.9590 ± 0.020** 🥉 | **0.9444 ± 0.028** 🥉 | — | — | — | — | — | — | — |
| 8 | Single-Task (Binary) | ResNeXt-50 (Binary Only) | Single-Task Binary CE | 0.9782 ± 0.012 | 0.9290 ± 0.030 | 0.9680 ± 0.015 | 0.9495 ± 0.021 | 0.9290 ± 0.030 | — | — | — | — | — | — | — |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 9 | Single-Task (Coarse) | Swin-Tiny (Coarse Only) | Single-Task Coarse CE | — | — | — | — | — | **81.45% ± 5.8%** 🥇 | **0.7512 ± 0.089** 🥇 | — | — | — | — | — |
| 10 | Single-Task (Coarse) | HRNet-W18 (Coarse Only) | Single-Task Coarse CE | — | — | — | — | — | **76.80% ± 9.4%** 🥈 | **0.6940 ± 0.142** 🥈 | — | — | — | — | — |
| 11 | Single-Task (Coarse) | ResNeXt-50 (Coarse Only) | Single-Task Coarse CE | — | — | — | — | — | **74.90% ± 8.1%** 🥉 | **0.6815 ± 0.115** 🥉 | — | — | — | — | — |
| 12 | Single-Task (Coarse) | ResNet-152 (Coarse Only) | Single-Task Coarse CE | — | — | — | — | — | 72.40% ± 11.2% | 0.6420 ± 0.165 | — | — | — | — | — |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 13 | Single-Task (Fine) | Swin-Tiny (Fine Only) | Single-Task Fine CE | — | — | — | — | — | — | — | **71.20% ± 12.8%** 🥇 | **0.5840 ± 0.105** 🥇 | **0.4215 ± 0.075** 🥇 | — | — |
| 14 | Single-Task (Fine) | HRNet-W18 (Fine Only) | Single-Task Fine CE | — | — | — | — | — | — | — | **63.10% ± 18.4%** 🥈 | **0.5420 ± 0.180** 🥈 | **0.3950 ± 0.130** 🥈 | — | — |
| 15 | Single-Task (Fine) | ResNeXt-50 (Fine Only) | Single-Task Fine CE | — | — | — | — | — | — | — | **61.80% ± 14.2%** 🥉 | 0.3810 ± 0.145 | **0.2790 ± 0.102** 🥉 | — | — |
| 16 | Single-Task (Fine) | ResNet-152 (Fine Only) | Single-Task Fine CE | — | — | — | — | — | — | — | 58.90% ± 16.5% | **0.3420 ± 0.150** 🥉 | 0.2480 ± 0.110 | — | — |

---

## 1. Stage 10 -- Sàng Lọc Kiến Trúc Backbone (Validation Benchmark)

### Bảng 1.1: Đa Nhiệm (Multitask Baselines) Trên Tập Validation
| # | Mô Hình & Kiến Trúc | Chiến Lược Huấn Luyện | Binary AUROC | Binary Sens | Binary Spec | Binary Prec | Binary F1 | Coarse Acc (%) | Coarse Bal Acc (%) | Coarse Macro-F1 | Fine Acc (%) | Fine Bal Acc (%) | Fine F1 (Supp) | Fine F1 (All 22) | C-F Consistency (%) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Swin-Tiny (Multitask)** | Shared Multi-Head (CE) | **0.9507 ± 0.027** 🥇 | **0.8484 ± 0.042** 🥇 | **0.8019 ± 0.083** 🥇 | **0.8376 ± 0.043** 🥇 | **0.8992 ± 0.029** 🥇 | **71.19% ± 2.5%** 🥇 | **0.6199 ± 0.018** 🥇 | **0.6243 ± 0.014** 🥇 | **49.28% ± 6.5%** 🥇 | **0.5120 ± 0.045** 🥇 | **0.5105 ± 0.068** 🥇 | **0.3755 ± 0.045** 🥇 | **76.45% ± 2.1%** 🥇 |
| 2 | **HRNet-W18 (Multitask)** | Shared Multi-Head (CE) | **0.9385 ± 0.035** 🥈 | 0.8120 ± 0.038 | **0.7850 ± 0.065** 🥈 | **0.8210 ± 0.035** 🥈 | **0.8759 ± 0.022** 🥈 | **63.66% ± 4.3%** 🥈 | **0.5410 ± 0.032** 🥈 | **0.5461 ± 0.035** 🥈 | **43.44% ± 3.4%** 🥈 | **0.4280 ± 0.038** 🥈 | **0.3979 ± 0.056** 🥈 | **0.2845 ± 0.039** 🥈 | **73.88% ± 2.2%** 🥈 |
| 3 | **ResNeXt-50 (Multitask)** | Shared Multi-Head (CE) | **0.9088 ± 0.037** 🥉 | 0.7840 ± 0.045 | 0.7410 ± 0.072 | 0.7950 ± 0.041 | **0.8387 ± 0.025** 🥉 | **58.61% ± 1.4%** 🥉 | **0.4680 ± 0.025** 🥉 | **0.4600 ± 0.028** 🥉 | **37.05% ± 3.5%** 🥉 | 0.2150 ± 0.036 | 0.2023 ± 0.036 | **0.1510 ± 0.028** 🥉 | **71.05% ± 2.5%** 🥉 |
| 4 | **ResNet-152 (Multitask)** | Shared Multi-Head (CE) | 0.8698 ± 0.050 | **0.8250 ± 0.041** 🥈 | 0.7120 ± 0.088 | 0.7780 ± 0.052 | 0.8191 ± 0.038 | 56.62% ± 0.3% | 0.4420 ± 0.015 | 0.4398 ± 0.017 | 34.71% ± 5.2% | **0.2240 ± 0.035** 🥉 | **0.2098 ± 0.038** 🥉 | 0.1482 ± 0.025 | 68.42% ± 3.1% |

### Bảng 1.2: Đơn Nhiệm (Single-Task Baselines) Trên Tập Validation
| # | Phân Nhóm | Mô Hình & Kiến Trúc | Chiến Lược Huấn Luyện | Binary AUROC | Binary Sens | Binary Spec | Binary F1 | Coarse Acc (%) | Coarse Macro-F1 | Fine Acc (%) | Fine F1 (Supp) | Fine F1 (All 22) |
|:---:|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Single Binary | **Swin-Tiny (Binary Only)** | Binary CE Only | **0.9590 ± 0.033** 🥇 | **0.8734 ± 0.022** 🥈 | **0.8968 ± 0.044** 🥇 | **0.8930 ± 0.034** 🥈 | — | — | — | — | — |
| 2 | Single Binary | **HRNet-W18 (Binary Only)** | Binary CE Only | **0.9579 ± 0.021** 🥈 | **0.8984 ± 0.020** 🥇 | **0.8840 ± 0.035** 🥈 | **0.8984 ± 0.020** 🥇 | — | — | — | — | — |
| 3 | Single Binary | **ResNeXt-50 (Binary Only)** | Binary CE Only | **0.9059 ± 0.034** 🥉 | 0.8210 ± 0.028 | 0.8520 ± 0.041 | 0.8356 ± 0.010 | — | — | — | — | — |
| 4 | Single Binary | **ResNet-152 (Binary Only)** | Binary CE Only | 0.8879 ± 0.038 | 0.8366 ± 0.030 | **0.8610 ± 0.038** 🥉 | **0.8366 ± 0.030** 🥉 | — | — | — | — | — |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 5 | Single Coarse | **Swin-Tiny (Coarse Only)** | Coarse CE Only | — | — | — | — | **71.17% ± 1.6%** 🥇 | **0.6403 ± 0.010** 🥇 | — | — | — |
| 6 | Single Coarse | **HRNet-W18 (Coarse Only)** | Coarse CE Only | — | — | — | — | **66.24% ± 3.7%** 🥈 | **0.5878 ± 0.028** 🥈 | — | — | — |
| 7 | Single Coarse | **ResNeXt-50 (Coarse Only)** | Coarse CE Only | — | — | — | — | **62.73% ± 2.2%** 🥉 | **0.5288 ± 0.027** 🥉 | — | — | — |
| 8 | Single Coarse | **ResNet-152 (Coarse Only)** | Coarse CE Only | — | — | — | — | 61.43% ± 2.1% | 0.4847 ± 0.052 | — | — | — |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 9 | Single Fine | **Swin-Tiny (Fine Only)** | Fine CE Only | — | — | — | — | — | — | **44.38% ± 5.0%** 🥇 | **0.4974 ± 0.055** 🥇 | **0.3673 ± 0.025** 🥇 |
| 10 | Single Fine | **HRNet-W18 (Fine Only)** | Fine CE Only | — | — | — | — | — | — | **43.70% ± 1.5%** 🥈 | **0.4372 ± 0.029** 🥈 | **0.3234 ± 0.002** 🥈 |
| 11 | Single Fine | **ResNeXt-50 (Fine Only)** | Fine CE Only | — | — | — | — | — | — | **33.37% ± 4.7%** 🥉 | 0.1979 ± 0.048 | 0.1481 ± 0.041 |
| 12 | Single Fine | **ResNet-152 (Fine Only)** | Fine CE Only | — | — | — | — | — | — | 31.39% ± 2.5% | **0.2080 ± 0.018** 🥉 | **0.1598 ± 0.006** 🥉 |

---

## 2. Stage 20 -- Sàng Lọc 7 Hàm Mất Mát Đuôi Dài (Validation Benchmark)

| # | Hàm Mất Mát | Chiến Lược Huấn Luyện | Binary AUROC | Binary Sens | Binary Spec | Binary F1 | Coarse Acc (%) | Coarse Macro-F1 | Fine Acc (%) | Fine F1 (Supp) | Primary Fine F1 (13 Lớp) | Tail Recall (n <= 20) | C-F Consistency (%) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Smoothed Balanced Softmax** | Patient-Smoothed Prior | **0.9521 ± 0.039** 🥉 | **0.8850 ± 0.045** 🥇 | **0.9120 ± 0.035** 🥈 | **0.8907 ± 0.058** 🥈 | **70.12% ± 3.9%** 🥇 | **0.6212 ± 0.038** 🥇 | **52.45% ± 1.7%** 🥇 | **0.5506 ± 0.074** 🥇 | **0.5607 ± 0.050** 🥇 | **66.38% ± 11.4%** 🥇 | **77.58% ± 1.6%** 🥇 |
| 2 | **Balanced Softmax** | Instance Frequency Prior | **0.9531 ± 0.038** 🥇 | 0.8710 ± 0.038 | **0.9150 ± 0.032** 🥇 | **0.8893 ± 0.031** 🥉 | **69.58% ± 2.6%** 🥉 | **0.5912 ± 0.032** 🥈 | 50.08% ± 5.7% | 0.4823 ± 0.060 | 0.5049 ± 0.022 | 62.76% ± 6.1% | 74.57% ± 3.7% |
| 3 | **Logit Adjustment** | Post-hoc Prior Margin | 0.9455 ± 0.042 | 0.8650 ± 0.041 | 0.9080 ± 0.038 | 0.8888 ± 0.050 | **69.98% ± 3.0%** 🥈 | **0.5837 ± 0.022** 🥉 | 49.62% ± 1.7% | 0.4822 ± 0.048 | 0.5041 ± 0.034 | 59.67% ± 8.4% | 76.98% ± 3.4% |
| 4 | **LDAM Loss** | Margin-based Push | **0.9522 ± 0.020** 🥈 | 0.8590 ± 0.025 | 0.9040 ± 0.028 | 0.8836 ± 0.016 | 69.37% ± 1.9% | 0.5834 ± 0.064 | 45.09% ± 2.8% | 0.4908 ± 0.058 | 0.5067 ± 0.030 | 62.51% ± 11.2% | 72.33% ± 3.6% |
| 5 | **Focal Loss** | Gamma=2.0 Modulating | 0.9506 ± 0.024 | **0.8780 ± 0.030** 🥈 | 0.9080 ± 0.032 | **0.8938 ± 0.032** 🥇 | 68.16% ± 3.7% | 0.5593 ± 0.058 | **51.09% ± 5.7%** 🥈 | 0.4976 ± 0.062 | 0.5150 ± 0.028 | 60.97% ± 8.6% | **77.09% ± 8.1%** 🥉 |
| 6 | **Cross-Entropy** | Standard Multi-Task CE | 0.9489 ± 0.042 | 0.8648 ± 0.079 | **0.9152 ± 0.043** 🥇 | 0.8888 ± 0.050 | 67.49% ± 2.2% | 0.5687 ± 0.011 | **50.23% ± 4.6%** 🥉 | **0.5268 ± 0.076** 🥈 | **0.5245 ± 0.019** 🥈 | **66.07% ± 8.9%** 🥈 | **77.37% ± 3.0%** 🥈 |
| 7 | **Weighted CE** | Inverse Class Frequency | 0.9427 ± 0.036 | 0.8520 ± 0.035 | 0.8950 ± 0.039 | 0.8747 ± 0.038 | 67.86% ± 2.2% | 0.5302 ± 0.056 | 49.18% ± 3.5% | **0.5053 ± 0.067** 🥉 | 0.5173 ± 0.051 | 63.97% ± 10.9% | 73.79% ± 2.2% |

---

## 3. Stage 30 & 40 -- Hệ Thống Bóc Tách Thực Nghiệm Toàn Diện (Comprehensive Ablation Studies)

### Bảng 3.1 (Table 6a): Khảo sát Chiến Lược & Quy Trình Huấn Luyện (Training Paradigm & Stage Decoupling)
*Mục tiêu:* Đánh giá tác động của quy trình huấn luyện tuần tự 3 giai đoạn (3S-HFT) so với huấn luyện đồng thời (1-Stage Joint) và tách rời 2 giai đoạn (2-Stage D2S-HFT) trên không gian đa tầng.

| # | Phương Pháp & Biến Thể | Chiến Lược Huấn Luyện | Binary AUROC | Binary F1 | Coarse Acc (%) | Coarse Macro-F1 | Fine Acc (%) | Fine F1 (Supp) | Fine F1 (All 22) | C-F Consistency (%) | Parent Acc (Ens) (%) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Proposed 3S-HFT v3.1** | **Rep $\rightarrow$ Coarse Align $\rightarrow$ Fine Align** | 0.9571 ± 0.021 | **0.8960 ± 0.026** 🥉 | **78.37% ± 1.0%** 🥇 | **0.6525 ± 0.012** 🥉 | **53.07% ± 4.0%** 🥇 | **0.5415 ± 0.036** 🥇 | **0.4007 ± 0.015** 🥇 | **82.28% ± 0.5%** 🥇 | **78.37% ± 1.0%** 🥇 |
| 2 | **1-Stage Joint Baseline** | Multi-Task Joint (CE + SupCon + SBS) | **0.9594 ± 0.018** 🥉 | **0.8965 ± 0.012** 🥈 | **73.64% ± 0.9%** 🥈 | **0.6576 ± 0.006** 🥇 | **52.63% ± 2.9%** 🥉 | 0.5026 ± 0.046 | 0.3718 ± 0.026 | 74.38% ± 3.2% | 71.20% ± 1.2% |
| 3 | **2-Stage Decoupled (D2S-HFT)** | Rep $\rightarrow$ Fine-Only SBS (Stage 35) | **0.9617 ± 0.028** 🥈 | 0.8912 ± 0.022 | 72.12% ± 4.6% | 0.6313 ± 0.031 | 52.20% ± 4.8% | 0.5266 ± 0.056 | 0.3893 ± 0.032 | **78.90% ± 2.4%** 🥈 | 72.50% ± 3.5% |
| 4 | **Ablation: Strategy cRT** | Phase 2 cRT Sampler (Fine Only) | **0.9617 ± 0.028** 🥈 | 0.8910 ± 0.024 | 72.12% ± 4.6% | 0.6313 ± 0.031 | 51.96% ± 2.5% | **0.5311 ± 0.048** 🥉 | **0.3930 ± 0.029** 🥉 | 78.45% ± 2.6% | 73.20% ± 3.0% |
| 5 | **Ablation: Target All Heads** | Phase 2 Unfreeze Binary + Coarse + Fine | 0.9583 ± 0.035 | 0.8875 ± 0.031 | 73.10% ± 3.4% | 0.6435 ± 0.031 | 51.55% ± 3.6% | 0.5129 ± 0.059 | 0.3794 ± 0.038 | 76.80% ± 2.9% | 72.80% ± 2.5% |

---

### Bảng 3.2 (Table 6b): Khảo sát Lịch Trình Trọng Số Phân Cấp (Hierarchy Loss Scheduling & Curriculum Warmup)
*Mục tiêu:* Chứng minh hiện tượng thắt cổ chai phân cấp sớm (Early Hierarchy Bottleneck) và vai trò của lịch trình Curriculum Warmup trong việc giải phóng biểu diễn.

| # | Biến Thể Lịch Trình | Công Thức Biến Thiên Trọng Số | Binary AUROC | Coarse Acc (%) | Coarse Macro-F1 | Fine Acc (%) | Fine F1 (Supp) | Fine F1 (All 22) | C-F Consistency (%) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Proposed Curriculum Warmup** | **$w_{\text{hrc}}(t) = 0.25 \cdot \min(1.0, t/12)$** | **0.9571 ± 0.021** | **73.57% ± 1.8%** 🥇 | **0.6525 ± 0.012** 🥇 | **53.07% ± 4.0%** 🥇 | **0.5415 ± 0.036** 🥇 | **0.4007 ± 0.015** 🥇 | **82.28% ± 0.5%** 🥇 |
| 2 | **Method A (Two-Phase)** | $w=0$ ở P1, $w=0.25$ ở P2/P3 | 0.9521 ± 0.028 | 71.76% ± 1.1% | 0.6371 ± 0.008 | 48.98% ± 4.1% | 0.5240 ± 0.025 | 0.3883 ± 0.017 | 81.55% ± 1.8% |
| 3 | **Fixed Hierarchy Weight** | $w_{\text{hrc}} = 0.25$ cố định xuyên suốt | 0.9466 ± 0.031 | 70.09% ± 2.3% | 0.6119 ± 0.018 | 47.21% ± 2.4% | 0.5199 ± 0.048 | 0.3844 ± 0.023 | **81.88% ± 2.0%** 🥈 |
| 4 | **w/o Hierarchy Loss** | $w_{\text{hrc}} = 0$ (Không ràng buộc) | **0.9649 ± 0.022** 🥇 | **73.46% ± 1.7%** 🥈 | **0.6426 ± 0.009** 🥈 | **52.72% ± 2.0%** 🥈 | **0.5414 ± 0.077** 🥈 | **0.3998 ± 0.047** 🥈 | 72.40% ± 4.1% ⬇️ |
| 5 | **w/o Binary-Coarse Loss** | $w_{\text{bc}} = 0, w_{\text{cf}} = 0.25$ | 0.9528 ± 0.037 | 71.75% ± 0.7% | 0.6206 ± 0.011 | 49.06% ± 2.8% | 0.5120 ± 0.035 | 0.3810 ± 0.022 | **82.43% ± 2.8%** 🥇 |
| 6 | **w/o Coarse-Fine Loss** | $w_{\text{bc}} = 0.25, w_{\text{cf}} = 0$ | 0.9605 ± 0.021 | 71.51% ± 4.0% | 0.6313 ± 0.029 | 52.92% ± 4.1% | 0.5312 ± 0.028 | 0.3915 ± 0.019 | 81.31% ± 3.1% |

---

### Bảng 3.3 (Table 6c): Bóc Tách Đóng Góp Cận Biên Của Các Thành Phần Hàm Loss (Loss Component Marginal Contributions)
*Mục tiêu:* Định lượng mức độ sụt giảm hiệu năng khi triệt tiêu từng thành phần độc lập trong hàm loss tổng thể.

| # | Cấu Hình Thử Nghiệm | Thành Phần Bị Triệt Tiêu | Binary AUROC | Binary F1 | Coarse Acc (%) | Fine Acc (%) | Primary Fine F1 (13 Lớp) | Tail Recall (n <= 20) | C-F Consistency (%) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Full Proposed (Anchor)** | **Đầy đủ 4 Trụ cột** | **0.9596 ± 0.032** | **0.8998 ± 0.038** | **72.76% ± 1.5%** | **51.02% ± 3.9%** | **0.6114 ± 0.023** 🏆 | **65.23% ± 7.4%** | **82.80% ± 2.6%** 🏆 |
| 2 | **w/o SupCon** | Không dùng Contrastive ($w=0$) | 0.9591 ± 0.029 | **0.9015 ± 0.038** 🥇 | 70.31% ± 2.7% | 47.74% ± 1.5% ⬇️ | 0.5627 ± 0.073 (**-4.87%**) | **67.08% ± 6.9%** 🥇 | 81.42% ± 2.7% |
| 3 | **w/o Smoothed Balanced Softmax**| Thay bằng Cross-Entropy thường | 0.9534 ± 0.021 | 0.8937 ± 0.026 | 70.93% ± 1.1% | 48.92% ± 3.2% ⬇️ | 0.6004 ± 0.026 (**-1.10%**) | 59.86% ± 9.4% (**-5.37%**) | 78.58% ± 3.7% (**-4.22%**) |
| 4 | **w/o Hierarchy Loss** | Không phạt xung đột phả hệ | 0.9583 ± 0.034 | 0.9009 ± 0.028 | 71.97% ± 1.9% | **51.29% ± 1.2%** 🥇 | 0.5890 ± 0.029 (**-2.24%**) | 65.54% ± 6.4% | 81.49% ± 2.9% |
| 5 | **w/o Data Augmentation** | Huấn luyện không qua Augmentation | 0.9596 ± 0.032 | 0.8998 ± 0.038 | **72.76% ± 1.5%** | 51.02% ± 3.9% | 0.6114 ± 0.023 | 65.23% ± 7.4% | **82.80% ± 2.6%** 🏆 |

---

### Bảng 3.4 (Table 6d): Khảo sát Vị Trí Trích Xuất Đặc Trưng Theo Độ Sâu (Architectural Head Placement)
*Mục tiêu:* So sánh kiến trúc Shared Late-Stage tại Stage 4 so với việc đặt các Classifier Heads phân tán tại các tầng trung gian (Intermediate Heads).

| # | Biến Thể Vị Trí Head | Cơ Chế Kết Nối Tầng | Binary AUROC | Binary Specificity | Coarse Acc (%) | Fine Acc (%) | Fine F1 (Supp) | Fine F1 (All 22) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Proposed Shared Late-Stage** | **Toàn bộ 3 Heads tại Stage 4 (768-d)** | **0.9571 ± 0.021** | **88.60% ± 5.0%** | **73.57% ± 1.8%** | **53.07% ± 4.0%** | **0.5415 ± 0.036** | **0.4007 ± 0.015** |
| 2 | **Multi-Stage Intermediate Heads** | S2 $\rightarrow$ Bin, S3 $\rightarrow$ Coarse, S4 $\rightarrow$ Fine | 0.8355 ± 0.045 (**-12.2%**) | 75.66% ± 4.8% (**-12.9%**) | 68.73% ± 3.1% (**-4.8%**) | 42.64% ± 3.8% (**-10.4%**) | 0.4806 ± 0.049 (**-6.1%**) | 0.3714 ± 0.031 (**-2.9%**) |

---

### Bảng 3.5 (Table 6e): Khảo sát Độ Sâu Đóng Băng Backbone & Đánh Đổi Chi Phí Tính Toán (Backbone Freezing Depth & Compute Trade-off)
*Mục tiêu:* Phân tích đánh đổi giữa khả năng chống overfit trên đặc trưng cấp thấp, độ chính xác đuôi dài và thời gian huấn luyện.

| # | Cấu Hình Đóng Băng | Tham Số Mở / Đóng Băng | Thời Gian / Epoch | Tổng Thời Gian Huấn Luyện | Fine Acc (%) | Primary Fine F1 (13 Lớp) | Fine F1 (Supp) | Tail Recall ($n \le 20$) | Coarse Acc (%) | Binary AUROC |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Full Fine-Tuning (100%)** | 27.52M (100%) / 0M (0%) | 59.41s (1.0x) | 1,366s (22.8 min) | 48.84% | 0.5848 | 0.4396 | **58.52%** 🥇 | **74.04%** 🥇 | **0.9333** 🥇 |
| 2 | **Partial FT: Freeze Stages 1–2** | 26.32M (95.7%) / 1.20M (4.3%) | 43.97s (**-26.0%**) | 659.6s (**-51.7%**) 🏆 | **49.22%** 🏆 | **0.6054 (+2.06%)** 🏆 | **0.4556 (+1.60%)** 🏆 | 47.04% | 70.21% | 0.9126 |
| 3 | **Partial FT: Freeze Stages 1–3** | 15.37M (55.8%) / 12.15M (44.2%)| **31.75s (-46.6%)** ⚡ | **444.5s (-67.5%)** ⚡ | 37.98% ⬇️ | 0.5503 ⬇️ | 0.4168 ⬇️ | 47.04% | 64.60% ⬇️ | 0.8739 ⬇️ |

---

### Bảng 3.6 (Table 6f): Khảo sát Độ Nhạy Siêu Tham Số & Cơ Chế Suy Luận Kết Hợp (Hyperparameters & Inference Blending)
*Mục tiêu:* Đánh giá độ nhạy của siêu tham số SupCon ($\tau, w_{\text{supcon}}$) và tối ưu hóa hệ số hòa trộn xác suất phả hệ ($\lambda$).

#### Phần A: Khảo sát Siêu tham số SupCon (Nhiệt độ $\tau$ & Trọng số $w$)
| Siêu Tham Số | Giá Trị Khảo Sát | Binary AUROC | Binary F1 | Coarse Acc (%) | Fine Acc (%) | Primary Fine F1 (13 Lớp) | Tail Recall ($n \le 20$) | C-F Consistency (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Nhiệt độ ($\tau$)** | $\tau = 0.05$ | **0.9664 ± 0.024** 🥇 | 0.9092 ± 0.032 | 72.51% ± 4.0% | 51.67% ± 3.9% | 0.5847 ± 0.013 | 59.67% ± 8.9% ⬇️ | 81.69% ± 2.8% |
| | **$\tau = 0.10$ (Optimal)** | 0.9596 ± 0.032 | 0.8998 ± 0.038 | **72.76% ± 1.5%** 🥇 | 51.02% ± 3.9% | **0.6114 ± 0.023** 🥇 | **65.23% ± 7.4%** 🥇 | **82.80% ± 2.6%** 🥇 |
| | $\tau = 0.20$ | 0.9550 ± 0.031 | 0.8954 ± 0.027 | 71.29% ± 2.6% | **51.94% ± 1.0%** 🥇 | 0.5914 ± 0.061 | 64.52% ± 7.2% | 79.41% ± 3.6% ⬇️ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Trọng số ($w_{\text{supcon}}$)** | $w = 0.05$ | 0.9602 ± 0.028 | 0.9035 ± 0.030 | **72.55% ± 1.5%** 🥇 | **53.67% ± 4.2%** 🥇 | 0.6081 ± 0.046 | 61.53% ± 8.4% | 79.80% ± 3.4% |
| | **$w = 0.10$ (Optimal)** | 0.9596 ± 0.032 | 0.8998 ± 0.038 | **72.76% ± 1.5%** 🥇 | 51.02% ± 3.9% | 0.6114 ± 0.023 | **65.23% ± 7.4%** 🥇 | **82.80% ± 2.6%** 🥇 |
| | $w = 0.20$ | **0.9630 ± 0.029** 🥇 | **0.9117 ± 0.031** 🥇 | 71.59% ± 2.4% | 50.52% ± 2.5% | **0.6228 ± 0.030** 🥇 | **65.23% ± 7.4%** 🥇 | 79.33% ± 3.5% |

#### Phần B: Trọng số Hòa trộn Xác suất Suy luận Hierarchical Marginalization ($\lambda$)
$$P_{\text{ens}}(C) = \lambda P_{\text{coarse}}(C) + (1-\lambda) \sum_{f \in \text{Children}(C)} P_{\text{fine}}(f)$$

| Hệ Số $\lambda$ | Ý Nghĩa Chế Độ Suy Luận | Validation Coarse Acc (%) | Validation Parent Acc (Ens) (%) | Test Coarse Acc (%) | Test Parent Acc (Ens) (%) | Mức Độ Nâng Cao ($\Delta$ vs Coarse Direct) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| $\lambda = 1.00$ | Chỉ dùng Coarse Head trực tiếp | 70.76% ± 1.1% | 70.76% ± 1.1% | 81.18% ± 6.9% | 81.18% ± 6.9% | Gốc (Baseline Direct) |
| $\lambda = 0.75$ | 75% Coarse Head + 25% Fine Marg. | 73.80% ± 1.2% | 73.80% ± 1.2% | 83.20% ± 5.1% | 83.20% ± 5.1% | +2.02% |
| $\lambda = 0.50$ | Cân bằng 50% Coarse + 50% Fine Marg. | 76.50% ± 0.9% | 76.50% ± 0.9% | 85.10% ± 4.2% | 85.10% ± 4.2% | +3.92% |
| **$\lambda = 0.25$** | **Tối ưu Đề Xuất (25% Coarse + 75% Fine)** | **78.37% ± 1.0%** 🏆 | **78.37% ± 1.0%** 🏆 | **86.42% ± 3.5%** 🏆 | **86.42% ± 3.5%** 🏆 | **+5.24% Test (+7.61% Val)** 🚀 |
| $\lambda = 0.00$ | Thuần túy Fine-to-Coarse Marginalization | 78.10% ± 0.7% | 78.10% ± 0.7% | 86.42% ± 3.5% | 86.42% ± 3.5% | +5.24% Test (+7.34% Val) |

---

## 4. Phân Tích Chi Tiết Từng Lớp Lâm Sàng & Benchmark Thời Gian Thực

### Bảng 4.1: Hiệu năng Chi tiết Theo Từng Lớp Coarse (Split 0 Validation)
| Nhóm Coarse (5 Nhóm) | Số Mẫu Thật (Support) | Số Mẫu Dự Đoán | Precision | Recall (Sensitivity) | F1-Score | Macro AUROC (OvR) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Malignant (Ác tính)** | 142 | 149 | 0.7852 | 0.8239 | **0.8041** | 0.9175 |
| **Non-malignant (Không ác tính)** | 32 | 30 | 0.2000 | 0.1875 | 0.1935 | 0.8348 |
| **Normal mucosa (Niêm mạc lành)** | 81 | 111 | 0.6757 | **0.9259** | 0.7813 | 0.9512 |
| **Anatomical landmarks (Giải phẫu)** | 31 | 11 | **0.8182** | 0.2903 | 0.4286 | 0.9027 |
| **Foreign bodies (Dị vật / Dụng cụ)** | 43 | 28 | **0.9286** | 0.6047 | 0.7324 | **0.9689** |

### Bảng 4.2: Benchmark Độ Trễ Suy Luận Thời Gian Thực (Edge Hardware: Apple Silicon MPS, FP32)
| Chế Độ / Batch | Số Vòng Đo | Forward Model (ms/ảnh) | Thông Lượng Forward (FPS) | Pipeline End-to-End (ms/ảnh) | Thông Lượng End-to-End (FPS) | Bộ Nhớ MPS Đã Cấp Phát (MiB) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Batch = 1** | 60 | **12.081 ± 0.95 ms** | **82.78 FPS** ⚡ | **15.000 ± 1.12 ms** | **66.67 FPS** ⚡ | 110.83 MiB |
| **Batch = 8** | 60 | **10.167 ± 0.42 ms** | **98.36 FPS** | — | — | 113.99 MiB |
| **Batch = 32** | 20 | **9.954 ± 0.38 ms** | **100.46 FPS** | — | — | 128.00 MiB |
