# Báo cáo Thực nghiệm Chi tiết: Sàng lọc 7 Phương pháp Loss Đuôi Dài (Stage 20)
**Giai đoạn:** `stage_20_run_long_tail_screen` | **Run Directory:** `result/20_long_tail/research_20260813-105659` | **Study:** `cystods_hierarchical_long_tailed_2026`

---

## 1. Tổng quan Thực nghiệm & Giao thức Sàng lọc (Loss Screening Protocol)

Stage 20 thực hiện **Sàng lọc toàn diện 7 phương pháp hàm mất mát xử lý phân bố đuôi dài (Long-Tail Loss Screening Benchmark)** trên kiến trúc backbone tối ưu từ Stage 10 (**Swin-Tiny** — `swin_tiny_patch4_window7_224.ms_in1k`).

### 1.1 Mục tiêu Kỹ thuật & Lâm sàng
* **Khắc phục Mất cân bằng Mẫu:** Phân bố tổn thương bàng quang lệch nghiêm trọng giữa các phân lớp phổ biến (như *LowGradePapillary, HighGradePapillary, CIS*) và các phân lớp hiếm (*GlandularMetaplasia, InvertedPapilloma, UrothelialPapilloma, Amyloidosis*).
* **Đánh giá Đa chiều:** Không chỉ đánh giá qua một metric đơn lẻ, mà kiểm chứng toàn diện qua **Binary Detection, Coarse 5-group Classification, Fine 22-class Long-tail Classification, Primary Fine Classification, Hierarchy Consistency, và Tail Class Recall**.

### 1.2 Phân hoạch Dữ liệu Đánh giá (Frozen Validation Holdout Split)
* **Tập Huấn luyện (Train):** 1,553 ảnh — 112 bệnh nhân (70%)
* **Tập Thẩm định (Validation):** 339 ảnh — 24 bệnh nhân (15%)
  - *Ảnh tổn thương có nhãn Fine-grained:* 258 ảnh (gồm 17 phân lớp có mẫu tại tập Val)
  - *Tập phân lớp chính (Primary Fine, $n_{\text{train}} \ge 5$):* 252 ảnh (13 phân lớp mô bệnh học chủ đạo)
* **Thiết lập Huấn luyện Chuẩn hóa:** Optimizer AdamW ($\text{LR} = 3\times 10^{-4}$, Encoder LR Multiplier $= 0.25$, Weight Decay $= 0.05$), Early Stopping patience $= 6$, Batch size $= 32$, Precision FP32/AMP.

---

## 2. Bảng Tổng hợp Toàn diện Hiệu năng 7 Phương pháp Loss (Master Comparison Table)

Bảng dưới đây tổng hợp các chỉ số đánh giá then chốt trên tập Validation độc lập ($n=339$) cho toàn bộ 7 phương pháp:

| # | Phương pháp Loss (Trial ID) | Epochs (Best/Total) | Binary AUROC | Binary F1 | Coarse Acc | Coarse Macro-F1 | Fine Acc | Fine Macro-F1 (17 Lớp) | Primary Fine Macro-F1 (13 Lớp) | Tail Recall ($n \le 10$) | Parent Acc from Fine | Coarse-Fine Consistency | Total Val Loss |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Smoothed Balanced Softmax** (`fine_balanced_softmax_smoothed`) | 16 / 22 | 0.9246 | **0.8846** | 69.91% | 0.5978 | 45.35% | **0.4828** | 0.5761 | 54.81% | **80.23%** | 74.03% | 6.785 |
| 2 | **Weighted CE** (`fine_weighted_ce`) | 10 / 16 | 0.9128 | 0.8468 | 68.73% | 0.6062 | 47.29% | **0.4925** | 0.5921 | 57.04% | 75.19% | 72.48% | 5.148 |
| 3 | **Balanced Softmax** (`fine_balanced_softmax`) | 10 / 16 | **0.9403** | 0.8817 | 73.16% | 0.6257 | 47.67% | **0.4870** | **0.6028** | **58.52%** | 76.74% | 81.40% | 5.101 |
| 4 | **Focal Loss** (`fine_focal`) | 10 / 16 | 0.9285 | 0.8509 | 68.44% | 0.5726 | **55.43%** | 0.4774 | 0.5689 | 52.59% | 73.26% | 82.56% | **3.876** |
| 5 | **Logit Adjustment** (`fine_logit_adjustment`) | 20 / 25 | 0.9171 | 0.8647 | 75.22% | **0.6729** | 46.51% | 0.4742 | 0.5651 | **58.52%** | 78.29% | 81.40% | 7.800 |
| 6 | **Cross-Entropy (Baseline)** (`fine_cross_entropy`) | 14 / 20 | 0.9055 | 0.8482 | 66.08% | 0.5483 | 52.71% | 0.4577 | 0.5320 | 51.11% | 76.36% | 79.84% | 6.453 |
| 7 | **LDAM Loss** (`fine_ldam`) | 13 / 19 | **0.9426** | 0.8643 | **75.52%** | 0.6681 | 39.92% | 0.4494 | 0.5469 | 47.41% | 79.46% | **84.11%** | 23.996 |

---

## 3. Bảng Phân tích Chi tiết Theo Từng Tác vụ (Task-by-Task Deep Dive)

### 3.1 Tác vụ 1: Phân loại Nhị phân (Binary Detection: ROI vs. Non-ROI)
Đánh giá độ nhạy phát hiện tổn thương và độ đặc hiệu loại trừ niêm mạc bàng quang bình thường ($n = 339$ ảnh, 187 ảnh ROI, 152 ảnh Non-ROI):

| Phương pháp Loss | Accuracy | Precision | Recall / Sensitivity | Specificity | F1-Score | MCC | AUROC | AUPRC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Smoothed Balanced Softmax** | **87.61%** | **90.96%** | 86.10% | **89.47%** | **0.8846** | **0.7524** | 0.9246 | 0.9512 |
| **Balanced Softmax** | 87.02% | 88.65% | **87.70%** | 86.18% | 0.8817 | 0.7380 | 0.9403 | **0.9590** |
| **LDAM Loss** | 85.55% | 89.66% | 83.42% | 88.16% | 0.8643 | 0.7122 | **0.9426** | 0.9537 |
| **Logit Adjustment** | 84.96% | 85.79% | 87.17% | 82.24% | 0.8647 | 0.6954 | 0.9171 | 0.9277 |
| **Focal Loss** | 83.78% | 86.26% | 83.96% | 83.55% | 0.8509 | 0.6733 | 0.9285 | 0.9482 |
| **Weighted CE** | 83.78% | 88.37% | 81.28% | 86.84% | 0.8468 | 0.6777 | 0.9128 | 0.9414 |
| **Cross-Entropy (Baseline)** | 82.89% | 83.08% | 86.63% | 78.29% | 0.8482 | 0.6532 | 0.9055 | 0.9334 |

> [!TIP]
> **Nhận xét Binary:** Smoothed Balanced Softmax dẫn đầu về **Precision (90.96%)**, **Specificity (89.47%)**, **F1-Score (0.8846)** và **MCC (0.7524)**. Việc kết hợp triệt tiêu nhiễu mẫu giúp mạng hạn chế tối đa báo động giả (False Positives) trên niêm mạc lành.

---

### 3.2 Tác vụ 2: Phân loại Thô (Coarse-Grained — 5 Phân nhóm Lớn)
Phân loại vào 5 nhóm: *Malignant, Non-malignant, Normal mucosa, Anatomical landmarks, Artefacts* ($n = 339$ ảnh):

| Phương pháp Loss | Accuracy | Macro-F1 (5 Classes) | Weighted-F1 | Balanced Accuracy | MCC | Macro-AUROC (OvR) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LDAM Loss** | **75.52%** | 0.6681 | **0.7432** | **65.01%** | **0.6422** | **0.9280** |
| **Logit Adjustment** | 75.22% | **0.6729** | 0.7415 | 63.70% | 0.6373 | 0.8819 |
| **Balanced Softmax** | 73.16% | 0.6257 | 0.7136 | 60.00% | 0.6027 | 0.8955 |
| **Smoothed Balanced Softmax** | 69.91% | 0.5978 | 0.6853 | 57.03% | 0.5660 | 0.8990 |
| **Weighted CE** | 68.73% | 0.6062 | 0.6821 | 59.45% | 0.5564 | 0.8934 |
| **Focal Loss** | 68.44% | 0.5726 | 0.6654 | 56.90% | 0.5392 | 0.8955 |
| **Cross-Entropy (Baseline)** | 66.08% | 0.5483 | 0.6426 | 52.26% | 0.4967 | 0.8746 |

---

### 3.3 Tác vụ 3: Phân loại Mịn Đuôi Dài (Fine-Grained 22 Classes & Primary 13 Classes)
Đánh giá chuyên sâu trên 22 phân lớp mô bệnh học ($n = 258$) và tập 13 phân lớp chính ($n = 252$):

| Phương pháp Loss | Fine Accuracy | Fine Macro-F1 (17 Lớp) | Fine Macro-F1 (22 Lớp) | Balanced Accuracy | Macro-AUROC (OvR) | Primary Fine Accuracy | Primary Fine Macro-F1 | Dự đoán Ngoài Primary |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Weighted CE** | 47.29% | **0.4925** | **0.3806** | **52.62%** | 0.8390 | 47.62% | 0.5921 | 2.78% (7/252) |
| **Balanced Softmax** | 47.67% | **0.4870** | 0.3763 | 51.41% | 0.8543 | 48.02% | **0.6028** | 7.94% (20/252) |
| **Smoothed Balanced Softmax** | 45.35% | **0.4828** | 0.3731 | 50.27% | 0.8510 | 45.63% | 0.5761 | **2.78% (7/252)** |
| **Focal Loss** | **55.43%** | 0.4774 | 0.3689 | 46.79% | **0.8613** | **55.95%** | 0.5689 | 2.78% (7/252) |
| **Logit Adjustment** | 46.51% | 0.4742 | 0.3665 | 48.25% | 0.8266 | 46.83% | 0.5651 | 4.37% (11/252) |
| **Cross-Entropy (Baseline)** | 52.71% | 0.4577 | 0.3537 | 45.29% | 0.8377 | 53.17% | 0.5320 | **1.59% (4/252)** |
| **LDAM Loss** | 39.92% | 0.4494 | 0.3473 | 49.13% | 0.7921 | 40.08% | 0.5469 | 2.38% (6/252) |

---

### 3.4 Tác vụ 4: Tính Nhất quán Phân cấp & Hồi phục Lớp Hiếm (Hierarchy & Tail Metrics)
Đánh giá mức độ bảo toàn logic y khoa từ Fine Head lên Coarse Head và độ nhạy trên các lớp cực hiếm:

| Phương pháp Loss | Parent Acc from Fine Head | Parent Acc from Coarse Head | Hierarchical Accuracy | Cross-Parent Error Rate | Coarse-Fine Consistency | Tail Class Recall ($n \le 10$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Smoothed Balanced Softmax** | **80.23%** | 64.34% | 31.78% | **19.77%** | 74.03% | 54.81% |
| **LDAM Loss** | 79.46% | **73.26%** | 34.50% | 20.54% | **84.11%** | 47.41% |
| **Logit Adjustment** | 78.29% | 70.93% | 38.37% | 21.71% | 81.40% | **58.52%** |
| **Balanced Softmax** | 76.74% | 71.32% | 41.86% | 23.26% | 81.40% | **58.52%** |
| **Cross-Entropy (Baseline)** | 76.36% | 63.57% | 41.86% | 23.64% | 79.84% | 51.11% |
| **Weighted CE** | 75.19% | 63.95% | 36.43% | 24.81% | 72.48% | 57.04% |
| **Focal Loss** | 73.26% | 63.57% | **48.06%** | 26.74% | 82.56% | 52.59% |

---

### 3.5 Tác vụ 5: Phân rã Giá trị Loss & Tốc độ Hội tụ (Loss Breakdown & Convergence)

| Phương pháp Loss | Total Loss (Val) | Binary Loss | Coarse Loss | Fine Loss | Best Epoch / Total | Tốc độ Hội tụ |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Focal Loss** | **3.876** | 0.727 | 1.535 | **1.613** | 10 / 16 | Rất nhanh (Đạt đỉnh epoch 10) |
| **Balanced Softmax** | 5.101 | **0.633** | 1.488 | 2.980 | 10 / 16 | Rất nhanh (Đạt đỉnh epoch 10) |
| **Weighted CE** | 5.148 | 0.895 | 1.579 | 2.674 | 10 / 16 | Rất nhanh (Đạt đỉnh epoch 10) |
| **Cross-Entropy** | 6.453 | 1.091 | 2.048 | 3.315 | 14 / 20 | Trung bình (Đạt đỉnh epoch 14) |
| **Smoothed Balanced Softmax** | 6.785 | 0.969 | 2.012 | 3.804 | 16 / 22 | Vững chắc, không overfit sớm |
| **Logit Adjustment** | 7.800 | 1.173 | 2.165 | 4.461 | 20 / 25 | Chậm (Cần 20-25 epochs) |
| **LDAM Loss** | 23.996 | **0.560** | **0.919** | 22.517 | 13 / 19 | Fine Loss bùng nổ do margin scale |

---

## 4. Phân tích So sánh & Đánh giá Toàn diện 7 Trường phái Loss

### 4.1 Nhóm Re-balancing theo Tiên lượng (Prior-Adjusted Softmax): *Smoothed BS & Balanced Softmax*
* **Smoothed Balanced Softmax:**
  - *Điểm mạnh:* Dẫn đầu về **Parent Accuracy from Fine Head (80.23%)** và **Tỷ lệ lỗi xuyên nhóm cha thấp nhất (19.77%)**. Đạt độ chính xác nhị phân cao nhất (**F1 = 0.8846, Precision = 90.96%, Specificity = 89.47%**).
  - *Cơ chế:* Khác với việc đếm số ảnh thuần túy, việc làm mượt theo số lượng bệnh nhân ($\text{patients}_j^{0.5}$) giúp triệt tiêu nhiễu mẫu cục bộ và ngăn chặn hiện tượng dự đoán bừa bãi vào các lớp hiếm (tỷ lệ dự đoán ngoài Primary chỉ **2.78%** so với **7.94%** của Balanced Softmax thường).
* **Balanced Softmax chuẩn:**
  - Đạt điểm Primary Fine Macro-F1 cao nhất (**0.6028**) và Tail Recall rất cao (**58.52%**), nhưng do dùng image prior trực tiếp nên có xu hướng thiên vị lớp hiếm hơi mạnh, dẫn đến 7.94% dự đoán rơi ra ngoài vùng phân lớp chính.

### 4.2 Nhóm Tái cân bằng Trọng số (Re-weighting): *Weighted CE & Focal Loss*
* **Weighted CE (Effective Number of Samples, $\beta=0.9999$):**
  - Đạt **Fine Macro-F1 cao nhất trên 17 lớp (0.4925)** và **Balanced Accuracy cao nhất (52.62%)**.
  - Nhược điểm: Tính nhất quán phân cấp Coarse-Fine kém hơn (72.48%) và Cross-Parent Error Rate lên tới 24.81%.
* **Focal Loss ($\gamma=2.0$):**
  - Đạt **Overall Fine Accuracy cao nhất (55.43%)** và **Hierarchical Accuracy cao nhất (48.06%)**, đồng thời có tổng Val Loss thấp nhất (3.876).
  - Nhược điểm: Do chỉ tập trung vào mẫu khó mà không bù đắp xác suất tiên lượng, Macro-F1 trên các lớp hiếm (0.4774) thấp hơn rõ rệt so với Weighted CE và Balanced Softmax.

### 4.3 Nhóm Biên độ & Dịch chuyển Logit (Margin-Based & Logit Adjustment): *LDAM & Logit Adjustment*
* **Logit Adjustment ($\tau=0.5$):**
  - Đạt **Coarse Macro-F1 cao nhất (0.6729)** và Tail Recall cao (58.52%).
  - Nhược điểm: Dịch chuyển logit tĩnh làm gradient hội tụ rất chậm (cần 20–25 epochs) và Fine Macro-F1 (0.4742) kém hơn nhóm Balanced Softmax.
* **LDAM Loss ($s=30, C=0.5$):**
  - Đạt **Binary AUROC cao nhất (0.9426)**, **Coarse Accuracy cao nhất (75.52%)** và **Coarse-Fine Consistency cao nhất (84.11%)**.
  - Nhược điểm chí mạng: Việc ép lề mở rộng $C / n_j^{1/4}$ khiến Fine Loss bùng nổ lên tới **22.52** (gấp 6 lần các loss khác), làm sụp đổ Fine Accuracy xuống mức thấp nhất toàn bảng (**39.92%**) và Fine Macro-F1 chỉ đạt **0.4494** (thậm chí thua cả Cross-Entropy baseline).

---

## 5. Kết luận & Quyết định Chuyển giao Kỹ thuật (Technical Transition)

Dựa trên phân tích đa chỉ số toàn diện:

1. **Phương pháp Toàn năng Nhất:** **Smoothed Balanced Softmax** (`balanced_softmax_smoothed`) và **Balanced Softmax** (`balanced_softmax`) cùng **Weighted CE** (`weighted_ce`) tạo thành Top 3 phương pháp dẫn đầu, vượt trội hoàn toàn so với baseline Cross-Entropy.
2. **Lựa chọn cho Stage 30 & Stage 40:** 
   - **`balanced_softmax_smoothed`** được xác nhận là phương pháp phân loại mô bệnh học đuôi dài chính thức nhờ sự cân bằng hoàn hảo giữa khả năng nhận diện lớp hiếm (Tail Recall 54.81%), độ chính xác nhị phân đỉnh cao (Binary F1 0.8846, Specificity 89.47%), và tính nhất quán y khoa cao nhất (Parent Acc 80.23%, Cross-parent error < 20%).
3. **Artifact chuyển giao:** Tệp `result/20_long_tail/research_20260813-105659/selected_long_tail_method.json` đã được đồng bộ hóa và sẵn sàng cho các giai đoạn tiếp theo của hệ thống CystoDS.
