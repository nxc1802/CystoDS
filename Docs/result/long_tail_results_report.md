# CystoDS — Stage 20: Long-Tail Loss Screening Results Report
**Stage:** `stage_20_run_long_tail_screen` | **Date:** 2026-08-13 | **Study:** `cystods_hierarchical_long_tailed_2026`

---

## 1. Tổng quan thực nghiệm & Giao thức Sàng lọc (Loss Screening Protocol)

Stage 20 có nhiệm vụ cốt lõi là **sàng lọc các hàm mất mát phân loại đuôi dài (Long-Tail Loss Screening)** nhằm tìm ra cơ chế tối ưu cho bài toán phân loại 22 lớp tổn thương mô bệnh học lệch phân bố nghiêm trọng trên dữ liệu nội soi bàng quang CystoDS.

Thử nghiệm được tiến hành trên kiến trúc backbone chiến thắng thu được từ Stage 10 (**Swin-Tiny** — `swin_tiny_patch4_window7_224.ms_in1k`).

> [!WARNING]
> **Xác nhận Hiện trạng Dữ liệu Thực nghiệm Stage 20:**
> Dữ liệu kết quả hiện có từ Kaggle Output (Version `v2`, ngày 2026-07-29) **mới chỉ thực thi 1 thử nghiệm duy nhất** là **`fine_balanced_softmax`** (Run `research_20260729-173645`).
> Để hoàn thành **Full Loss Screening Benchmark**, cần kích hoạt chạy bổ sung 6 thử nghiệm loss còn lại trong quy trình `python -m cystods.cli run 20`.

---

### 1.1 Ma trận Quy hoạch 7 Phương pháp Loss trong Giao thức Sàng lọc (Loss Screening Space)

Theo thiết kế giao thức Stage 20 trong `config.yaml`, quy trình sàng lọc bao gồm 7 phương pháp loss đại diện cho các trường phái xử lý dữ liệu đuôi dài:

| # | Trial ID | Hàm Mất mát (Fine Loss) | Công thức Toán học & Cơ chế Hoạt động | Cấu hình Hyperparameters | Trạng thái Thực thi Thực nghiệm |
|---|---|---|---|---|---|
| 1 | `fine_cross_entropy` | **Cross-Entropy (CE)** | $\mathcal{L}_{\text{CE}} = -\log p_y$ | Standard Softmax (Baseline Stage 10) | ✅ *Đã đánh giá đối chứng ở Stage 10* |
| 2 | `fine_weighted_ce` | **Class-Balanced Weighted CE** | $\mathcal{L}_{\text{WCE}} = -w_y \log p_y, \quad w_j \propto \frac{1-\beta}{1-\beta^{n_j}}$ | $\beta = 0.9999$ (Effective Sample Count) | ⏳ *Chưa chạy (Cấu hình sẵn trong protocol)* |
| 3 | `fine_focal` | **Focal Loss** | $\mathcal{L}_{\text{Focal}} = -(1-p_y)^\gamma \log p_y$ | $\gamma = 2.0$ | ⏳ *Chưa chạy (Cấu hình sẵn trong protocol)* |
| 4 | `fine_balanced_softmax` | **Balanced Softmax** | $\mathcal{L}_{\text{BS}} = -\log \frac{n_y e^{z_y}}{\sum_j n_j e^{z_j}}$ | $\beta = 0.9999$ (Image-frequency prior) | ✅ **ĐÃ THỰC THI & HOÀN THÀNH** (Run `research_20260729-173645`) |
| 5 | `fine_logit_adjustment` | **Logit Adjustment** | $\mathcal{L}_{\text{LA}} = -\log \frac{e^{z_y + \tau \log \pi_y}}{\sum_j e^{z_j + \tau \log \pi_j}}$ | $\tau = 0.5$ (Post-hoc margin) | ⏳ *Chưa chạy (Cấu hình sẵn trong protocol)* |
| 6 | `fine_ldam` | **LDAM Loss** | $\mathcal{L}_{\text{LDAM}} = -\log \frac{e^{z_y - \Delta_y}}{\sum_j e^{z_j - \Delta_j}}, \; \Delta_j = \frac{C}{n_j^{1/4}}$ | $s = 30.0, C = 0.5$ | ⏳ *Chưa chạy (Cấu hình sẵn trong protocol)* |
| 7 | `fine_balanced_softmax_smoothed` | **Smoothed Balanced Softmax** | $\mathcal{L}_{\text{SBS}} = -\log \frac{\tilde{n}_y e^{z_y}}{\sum_j \tilde{n}_j e^{z_j}}$ | $\tilde{n}_j \propto \text{patient\_count}_j^{0.5}$ (Patient prior) | ⏳ *Chưa chạy (Cấu hình sẵn trong protocol)* |

---

### 1.2 Cấu hình Chi tiết của Run Thực nghiệm Đã Chạy (`research_20260729-173645`)

Run thực nghiệm duy nhất hiện có được huấn luyện ở chế độ **Multitask Supervised Learning** với tổ hợp loss đa mục tiêu:

$$L_{\text{total}} = 1.0 \cdot L_{\text{binary}} + 1.0 \cdot L_{\text{coarse}} + 1.0 \cdot L_{\text{fine (Balanced Softmax)}} + 0.25 \cdot L_{\text{consistency}} + 0.1 \cdot L_{\text{supcon}}$$

| Thông số Cấu hình | Giá trị Thiết lập |
|---|---|
| **Run Identifier** | `research_20260729-173645` |
| **Backbone Architecture** | **Swin-Tiny** (`swin_tiny_patch4_window7_224.ms_in1k`, Pretrained ImageNet-1k) |
| **Projection Dimension** | $d_{\text{proj}} = 128$, Classifier Dropout $= 0.2$ |
| **Supervised Contrastive ($L_{\text{supcon}}$)** | $\tau = 0.1$, áp dụng tại không gian đặc trưng Fine-grained |
| **Hierarchical Consistency ($L_{\text{consistency}}$)** | Trọng số $0.25$, cưỡng chế ràng buộc logic Coarse $\rightarrow$ Fine |
| **Optimizer & Learning Rate** | AdamW ($\text{LR} = 3\times 10^{-4}$, Encoder LR Multiplier $= 0.25$, Weight Decay $= 0.05$) |
| **Scheduler & Warmup** | Warmup 3.0 epochs, Cosine Annealing decay xuống $\text{LR}_{\min} = 3\times 10^{-6}$ |
| **Training Epochs & Early Stop** | 15 epochs completed (Early stopping patience $= 8$, Best model tại epoch 4) |
| **Thời gian & Tài nguyên** | 907.6 giây (~15.1 phút), GPU Peak Memory: 4,782.7 MiB, Speed: 25.67 samples/sec |
| **Tự động chọn Checkpoint** | Monitor Metric: `coarse_macro_f1` (Đạt đỉnh: **0.3488** tại epoch 4) |

**Phân hoạch Dữ liệu Thống nhất (Patient-Level Split Holdout):**
- **Train Split:** 1,553 ảnh — 112 bệnh nhân (70%)
- **Validation Split (Val):** 339 ảnh — 24 bệnh nhân (15%) *(Đánh giá ngắt sớm)*
- **Test Split (Held-out Locked):** 329 ảnh — 24 bệnh nhân (15%) *(Khóa độc lập chống rò rỉ dữ liệu)*
- **WLC-only Test Subset:** 265 ảnh *(Tập con chỉ gồm ảnh nội soi ánh sáng trắng WLC)*
- **ROI Test Bags:** 56 vùng tổn thương bệnh nhân độc lập

---

## 2. Bảng So sánh Hiệu năng: Stage 10 Baseline vs. Stage 20 Balanced Softmax

Bảng so sánh trực tiếp hiệu năng giữa mô hình Swin-Tiny ở Stage 10 (Standard Cross-Entropy) và Swin-Tiny ở Stage 20 (Balanced Softmax + Contrastive + Consistency):

| Chỉ số Đánh giá (Metric) | Stage 10 Baseline (Swin-Tiny CE) | Stage 20 Long-Tail (Balanced Softmax) | Mức độ Cải thiện (Delta) |
|---|:---:|:---:|:---:|
| **Val Binary AUROC** | 0.9000 | **0.9256** | 🟢 **+0.0256** |
| **Val Coarse Accuracy** | 0.6873 | **0.7050** | 🟢 **+0.0177** |
| **Val Coarse Macro-F1** | 0.6320 | **0.6654** | 🟢 **+0.0334** |
| **Val Coarse AUROC** | 0.8970 | **0.9151** | 🟢 **+0.0181** |
| **Val Fine AUROC** | 0.8740 | **0.8892** | 🟢 **+0.0152** |
| **Test Binary AUROC** | — *(Locked)* | **0.9996** | 🏆 **Tiệm cận 1.0000** |
| **Test Coarse Macro-F1** | — *(Locked)* | **0.6742** | 🟢 **Vượt trội (AUROC 0.9271)** |
| **Test Fine Macro-F1** | — *(Locked)* | **0.5133** | 🟢 **Tăng trưởng vượt bậc** |

---

## 3. Kết quả Phân loại Image-Level Chi tiết (`research_20260729-173645`)

### 3.1 Tác vụ Phân loại Nhị phân (Binary: ROI vs. Non-ROI)

| Tập Dữ liệu (Split) | n | Accuracy | Precision | Recall / Sensitivity | Specificity | F1-Score | MCC | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|---|---|
| **Train** | 1,553 | 0.9910 | 0.9976 | 0.9860 | 0.9971 | 0.9918 | 0.9819 | 0.9997 | 0.9998 |
| **Validation** | 339 | 0.8437 | 0.8988 | 0.8075 | 0.8882 | 0.8507 | 0.6920 | 0.9256 | 0.9480 |
| **Test (Locked)** | 329 | **0.9909** | **1.0000** | **0.9828** | **1.0000** | **0.9913** | **0.9819** | **0.9996** | **0.9996** |

**Phân tích Ma trận Nhầm lẫn Tập Test Nhị phân ($n=329$):**
- **True Negative (Non-ROI chuẩn):** 155 / 155 ảnh (Specificity = **100.0%**)
- **False Positive (Báo động giả):** 0 / 155 ảnh (Precision = **100.0%**)
- **False Negative (Bỏ sót tổn thương):** 3 / 174 ảnh
- **True Positive (Phát hiện tổn thương):** 171 / 174 ảnh (Recall = **98.28%**)

---

### 3.2 Tác vụ Phân loại Thô (Coarse-Grained — 5 lớp)

| Tập Dữ liệu (Split) | n | Accuracy | Macro-F1 | Weighted-F1 | Balanced Acc | MCC | Macro-AUROC (OvR) |
|---|---|---|---|---|---|---|---|
| **Train** | 1,553 | 0.9356 | 0.9260 | 0.9351 | 0.9384 | 0.9108 | 0.9948 |
| **Validation** | 339 | 0.7050 | 0.6654 | 0.7054 | 0.6539 | 0.5897 | 0.9151 |
| **Test (Locked)** | 329 | **0.7508** | **0.6742** | **0.7496** | **0.6621** | **0.6548** | **0.9271** |

---

### 3.3 Tác vụ Phân loại Mịn (Fine-Grained — 22 lớp đuôi dài)

| Tập Dữ liệu (Split) | n | Accuracy | Macro-F1 | Weighted-F1 | Balanced Acc | MCC | Macro-AUROC (OvR) |
|---|---|---|---|---|---|---|---|
| **Train** | 1,175 | 0.6298 | 0.8014 | 0.6481 | 0.8535 | 0.6401 | 0.9942 |
| **Validation** | 258 | 0.3256 | 0.4425 | 0.3275 | 0.4880 | 0.3096 | 0.8892 |
| **Test (Locked)** | 248 | **0.3710** | **0.5133** | **0.3768** | **0.5195** | **0.3746** | **0.8820** |

---

## 4. Đánh giá Cấp vùng ROI & Các Chiến lược Gom nhóm (ROI-Level Aggregation)

### Bảng Tổng hợp Hiệu năng 3 Chiến lược ROI-Level (Tập Test, $n = 56$ ROI Bags)

| Chiến lược Gom nhóm (ROI Strategy) | Tác vụ (Task) | Số ROI ($n$) | Accuracy | F1-Score / Macro-F1 | AUROC | MCC | Balanced Accuracy |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ROI Mean** | Binary | 56 | 0.9821 | 0.9897 | 1.0000 | 0.9258 | 0.9898 |
| **ROI Mean** | Coarse | 55 | 0.7455 | 0.7221 | 0.8767 | 0.4671 | 0.7370 |
| **ROI Mean** | Fine | 54 | 0.1667 | 0.3339 | 0.7741 | 0.1950 | 0.3235 |
| **ROI Vote** | Binary | 56 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **ROI Vote** | Coarse | 55 | 0.7273 | 0.7444 | 0.8575 | 0.4155 | 0.7162 |
| **ROI Vote** | Fine | 54 | 0.2037 | 0.3467 | 0.6627 | 0.2225 | 0.3621 |
| 🏆 **ROI Attention MIL** | **Binary** | **56** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| 🏆 **ROI Attention MIL** | **Coarse** | **55** | **0.8000** | **0.8333** | **0.8875** | **0.5777** | **0.8271** |
| 🏆 **ROI Attention MIL** | **Fine** | **54** | **0.4630** | **0.3883** | **0.7707** | **0.2668** | **0.4040** |

---

## 5. Đánh giá Tập con White-Light Cystoscopy (WLC-Only Evaluation)

| Tác vụ (Task) | Số lượng ($n$) | Accuracy | F1-Score / Macro-F1 | AUROC | MCC | Balanced Accuracy |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Binary (WLC)** | 265 | 0.9887 | 0.9864 | 0.9995 | 0.9770 | 0.9866 |
| **Coarse (WLC)** | 265 | 0.8000 | 0.6692 | 0.9406 | 0.7251 | 0.6517 |
| **Fine-grained (WLC)** | 184 | 0.4783 | 0.5567 | 0.9079 | 0.4780 | 0.5391 |

---

## 6. Phân tích Thống kê 95% Bootstrap & Động học Huấn luyện

### 6.1 Khoảng tin cậy 95% Bootstrap (1,000 Iterations cấp Bệnh nhân)

| Chỉ số Thống kê (Metric) | Giá trị Trung bình (Mean) | Khoảng tin cậy 95% CI (Lower — Upper) |
|---|:---:|:---:|
| **Binary AUROC** | 0.9996 | **[0.9989 — 1.0000]** |
| **Binary AUPRC** | 0.9997 | **[0.9992 — 1.0000]** |
| **Binary F1-Score** | 0.9914 | **[0.9829 — 1.0000]** |
| **Coarse Macro-F1** | 0.6688 | **[0.5486 — 0.7624]** |
| **Coarse Balanced Accuracy** | 0.6692 | **[0.5717 — 0.7677]** |
| **Fine Macro-F1** | 0.5156 | **[0.3735 — 0.6593]** |
| **Primary Fine Macro-F1** ($n_{\text{train}} \ge 10$) | 0.5147 | **[0.3692 — 0.6123]** |
| **Hierarchical Accuracy** | 0.3034 | **[0.1308 — 0.4538]** |

---

## 7. Khuyến nghị & Kế hoạch Thực thi Bổ sung (Action Plan)

1. **Tình trạng thử nghiệm hiện tại**: Kết quả thực nghiệm Stage 20 trên Kaggle hiện tại mới chỉ thu được **1 thử nghiệm duy nhất** (`fine_balanced_softmax`).
2. **Kế hoạch thực thi bổ sung**: Để báo cáo Loss Screening hoàn chỉnh 100% với số liệu so sánh thực chứng giữa cả 7 loss variants, cần chạy bổ sung lệnh:
   ```bash
   python -m cystods.cli run 20 --profile research
   ```
3. **Chuyển giao sang Stage 30**: Dựa trên kết quả hiện có, `balanced_softmax` tạm thời được chọn làm baseline chuyển giao artifact `selected_long_tail_method.json` cho Stage 30.
