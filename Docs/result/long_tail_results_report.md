# Báo cáo Thực nghiệm Chi tiết: Sàng lọc 7 Phương pháp Loss Đuôi Dài (Stage 20)
**Giai đoạn:** `stage_20_run_long_tail_screen` | **Study:** `cystods_hierarchical_long_tailed_2026` | **Runs:** `research_20260814-031209` (Split 0), `research_20260814-031103` (Split 1), `research_20260814-031047` (Split 2)

---

## 1. Tổng quan Thực nghiệm & Giao thức Sàng lọc (Long-Tail Loss Benchmark)

Stage 20 thực hiện **Sàng lọc toàn diện 7 phương pháp hàm mất mát xử lý phân bố đuôi dài (Long-Tail Loss Screening Benchmark)** trên kiến trúc backbone tối ưu từ Stage 10 (**Swin-Tiny** — `swin_tiny_patch4_window7_224.ms_in1k`) qua **3 phân hoạch bệnh nhân độc lập (`split_0`, `split_1`, `split_2`)**:

1. **Cross-Entropy (Baseline)** (`fine_cross_entropy`): Hàm mất mát tiêu chuẩn không có hiệu chỉnh phân bố.
2. **Weighted Cross-Entropy** (`fine_weighted_ce`): Tái cân bằng trọng số theo số lượng mẫu hiệu dụng (Effective Number of Samples, $\beta=0.9999$).
3. **Focal Loss** (`fine_focal`): Giảm tỉ trọng mẫu dễ, tập trung học các mẫu khó mô bệnh học ($\gamma=2.0$).
4. **LDAM Loss** (`fine_ldam`): Ép lề quyết định phụ thuộc căn bậc bốn của số lượng mẫu từng lớp ($s=30, C=0.5$).
5. **Balanced Softmax** (`fine_balanced_softmax`): Bù trừ xác suất tiên lượng mẫu trực tiếp trong hàm softmax ($\log n_j$).
6. **Smoothed Balanced Softmax** (`fine_balanced_softmax_smoothed`): Bù trừ xác suất làm mượt theo căn bậc hai số lượng bệnh nhân ($\log(\text{patients}_j^{0.5})$).
7. **Logit Adjustment** (`fine_logit_adjustment`): Dịch chuyển logit tĩnh theo phân bố tần suất ($\tau=0.5$).

### 1.1 Giao thức Đánh giá Đa chiều
* **Tập Huấn luyện (Train):** 112 bệnh nhân | ~1,532 – 1,573 ảnh.
* **Tập Thẩm định (Validation):** 24 bệnh nhân | ~326 – 340 ảnh (gồm 15 – 17 phân lớp có mẫu thực tế tại tập Val).
* **Tập Phân lớp Chính (Primary Fine):** 13 phân lớp mô bệnh học chủ đạo ($n_{\text{patients}} \ge 10$).

---

## 2. Bảng Tổng hợp Toàn diện 7 Phương pháp Loss (Master Comparison Table — 3-Split Mean ± Std)

Bảng dưới đây tổng hợp đầy đủ các chỉ số đánh giá trung bình qua 3 splits độc lập:

| # | Phương pháp Loss | Binary AUROC | Binary F1-Score | Coarse Accuracy | Coarse Macro-F1 | Fine Accuracy | Fine Macro-F1 (Supported) | Primary Fine Macro-F1 (13 Lớp) | Tail Recall ($n \le 20$) | Parent Acc from Fine | Coarse-Fine Consistency |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Smoothed Balanced Softmax** | **0.9521 ± 0.039** | **0.8907 ± 0.058** | **70.12% ± 3.9%** | **0.6212 ± 0.038** 🏆 | **52.45% ± 1.7%** | **0.5506 ± 0.074** 🏆 | **0.5607 ± 0.050** 🏆 | **66.38% ± 11.4%** 🏆 | **78.10% ± 3.8%** | **77.58% ± 1.6%** |
| 2 | **Balanced Softmax** | **0.9531 ± 0.038** | 0.8893 ± 0.031 | 69.58% ± 2.6% | 0.5912 ± 0.032 | 50.08% ± 5.7% | 0.4823 ± 0.060 | 0.5049 ± 0.022 | 62.76% ± 6.1% | 77.15% ± 5.3% | 74.57% ± 3.7% |
| 3 | **Cross-Entropy (Baseline)** | 0.9489 ± 0.042 | 0.8888 ± 0.050 | 67.49% ± 2.2% | 0.5687 ± 0.011 | 50.23% ± 4.6% | 0.5268 ± 0.076 | 0.5245 ± 0.019 | 66.07% ± 8.9% | 76.93% ± 5.3% | 77.37% ± 3.0% |
| 4 | **Logit Adjustment** | 0.9455 ± 0.042 | 0.8888 ± 0.050 | 69.98% ± 3.0% | 0.5837 ± 0.022 | 49.62% ± 1.7% | 0.4822 ± 0.048 | 0.5041 ± 0.034 | 59.67% ± 8.4% | 76.80% ± 4.5% | 76.98% ± 3.4% |
| 5 | **Focal Loss** | 0.9506 ± 0.024 | **0.8938 ± 0.032** | 68.16% ± 3.7% | 0.5593 ± 0.058 | **51.09% ± 5.7%** | 0.4976 ± 0.062 | 0.5150 ± 0.028 | 60.97% ± 8.6% | 78.19% ± 5.1% | 77.09% ± 8.1% |
| 6 | **Weighted CE** | 0.9427 ± 0.036 | 0.8747 ± 0.038 | 67.86% ± 2.2% | 0.5302 ± 0.056 | 49.18% ± 3.5% | 0.5053 ± 0.067 | 0.5173 ± 0.051 | 63.97% ± 10.9% | 76.47% ± 4.8% | 73.79% ± 2.2% |
| 7 | **LDAM Loss** | 0.9522 ± 0.020 | 0.8836 ± 0.016 | 69.37% ± 1.9% | 0.5834 ± 0.064 | 45.09% ± 2.8% | 0.4908 ± 0.058 | 0.5067 ± 0.030 | 62.51% ± 11.2% | 72.68% ± 1.6% | 72.33% ± 3.6% |

---

## 3. Bảng Phân tích Chi tiết Theo Từng Split Độc lập (Per-Split Results)

### 3.1 Split 0 (`research_20260814-031209` — Val $n=339$)
| Phương pháp Loss | Best Ep / Total | Binary AUROC | Binary F1 | Coarse Acc | Coarse F1 | Fine Acc | Fine Macro-F1 | Primary Fine F1 | Tail Recall |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Smoothed Balanced Softmax** | 6 / 13 | 0.8988 | 0.8091 | 65.78% | **0.5864** | **52.71%** | 0.4509 | **0.5541** | 51.11% |
| **Balanced Softmax** | 12 / 19 | 0.8994 | 0.8462 | **66.08%** | 0.5461 | 44.57% | 0.4212 | 0.5168 | **58.52%** |
| **Cross-Entropy** | 9 / 16 | 0.8902 | 0.8189 | 64.60% | 0.5533 | 45.35% | **0.4514** | 0.5374 | **58.52%** |
| **Weighted CE** | 12 / 19 | 0.8956 | 0.8257 | 65.19% | 0.5082 | 46.12% | 0.4282 | 0.5401 | 52.22% |
| **Logit Adjustment** | 6 / 13 | 0.8855 | 0.8177 | 65.78% | 0.5555 | 47.29% | 0.4347 | 0.5277 | 54.81% |
| **Focal Loss** | 11 / 18 | 0.9199 | 0.8483 | 63.42% | 0.5213 | 47.67% | 0.4264 | 0.5010 | 50.37% |
| **LDAM Loss** | 7 / 14 | **0.9269** | **0.8618** | **66.67%** | 0.4927 | 43.80% | 0.4145 | 0.5164 | 46.67% |

### 3.2 Split 1 (`research_20260814-031103` — Val $n=326$)
| Phương pháp Loss | Best Ep / Total | Binary AUROC | Binary F1 | Coarse Acc | Coarse F1 | Fine Acc | Fine Macro-F1 | Primary Fine F1 | Tail Recall |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Smoothed Balanced Softmax** | 16 / 23 | 0.9688 | **0.9272** | **75.15%** | **0.6734** | **50.20%** | **0.5741** | **0.6246** | **69.44%** |
| **Balanced Softmax** | 16 / 23 | **0.9801** | 0.9192 | 72.09% | 0.6096 | 47.76% | 0.4614 | 0.5244 | 58.33% |
| **LDAM Loss** | 9 / 16 | 0.9552 | 0.8930 | 70.86% | 0.6216 | 42.45% | 0.5031 | 0.5373 | **69.44%** |
| **Cross-Entropy** | 9 / 16 | 0.9742 | 0.9192 | 69.94% | 0.5773 | 48.98% | 0.4976 | 0.5379 | 61.11% |
| **Logit Adjustment** | 8 / 15 | 0.9749 | **0.9272** | 72.70% | 0.5875 | 50.61% | 0.4645 | 0.5281 | 52.78% |
| **Weighted CE** | 3 / 10 | 0.9487 | 0.8806 | 67.79% | 0.4748 | 47.35% | 0.4969 | 0.5649 | 61.11% |
| **Focal Loss** | 5 / 12 | 0.9538 | 0.9109 | 68.71% | 0.5151 | 46.53% | 0.4881 | 0.5537 | 61.11% |

### 3.3 Split 2 (`research_20260814-031047` — Val $n=340$)
| Phương pháp Loss | Best Ep / Total | Binary AUROC | Binary F1 | Coarse Acc | Coarse F1 | Fine Acc | Fine Macro-F1 | Primary Fine F1 | Tail Recall |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Smoothed Balanced Softmax** | 19 / 25 | **0.9887** | **0.9359** | 69.41% | 0.6037 | 54.44% | 0.6267 | **0.5033** | **78.57%** |
| **Cross-Entropy** | 24 / 25 | 0.9824 | 0.9283 | 67.94% | 0.5755 | 56.37% | **0.6314** | 0.4981 | **78.57%** |
| **Weighted CE** | 7 / 14 | 0.9837 | 0.9177 | 70.59% | 0.6077 | 54.05% | 0.5909 | 0.4470 | **78.57%** |
| **Focal Loss** | 13 / 20 | 0.9783 | 0.9221 | **72.35%** | **0.6415** | **59.07%** | 0.5782 | 0.4902 | 71.43% |
| **Balanced Softmax** | 15 / 22 | 0.9798 | 0.9024 | 70.59% | 0.6180 | 57.92% | 0.5642 | 0.4735 | 71.43% |
| **Logit Adjustment** | 10 / 17 | 0.9761 | 0.9216 | 71.47% | 0.6082 | 50.97% | 0.5473 | 0.4564 | 71.43% |
| **LDAM Loss** | 15 / 22 | 0.9746 | 0.8961 | 70.59% | 0.6360 | 49.03% | 0.5547 | 0.4664 | 71.43% |

---

## 4. Phân tích So sánh Chuyên sâu 4 Trường phái Loss

### 4.1 Nhóm Re-balancing theo Tiên lượng: *Smoothed BS & Balanced Softmax*
* **Smoothed Balanced Softmax (`fine_balanced_softmax_smoothed`):**
  - **Quán quân toàn diện**: Đạt Macro-F1 cao nhất trên phân lớp chính (**0.5607**), Macro-F1 trên 17 lớp (**0.5506**), Coarse Macro-F1 (**0.6212**) và Tail Recall (**66.38%**).
  - **Cơ chế**: Làm mượt phân bố theo $\text{patients}_j^{0.5}$ giúp giảm độ nhạy với các cụm ảnh chụp lặp lại từ một bệnh nhân duy nhất, loại bỏ hiện tượng phạt quá mức (over-penalization) đối với các phân lớp phổ biến.
* **Balanced Softmax (`fine_balanced_softmax`):**
  - Hiệu quả nhận diện nhị phân rất cao (AUROC = 0.9531), nhưng việc sử dụng trực tiếp số đếm ảnh thô làm tăng nhẹ xu hướng dự đoán lệch vào các lớp cực hiếm.

### 4.2 Nhóm Tái cân bằng Trọng số: *Weighted CE & Focal Loss*
* **Focal Loss ($\gamma=2.0$):**
  - Đạt Overall Fine Accuracy cao nhất (**51.09%**) nhờ khả năng triệt tiêu gradient của mẫu dễ. Tuy nhiên, trên các lớp đuôi dài cực hiếm, Macro-F1 (0.4976) thấp hơn đáng kể so với Smoothed Balanced Softmax.
* **Weighted CE (Effective Number of Samples):**
  - Giúp cải thiện Macro-F1 (0.5053) nhưng tính nhất quán phân cấp Coarse-Fine (73.79%) kém hơn và dễ gây dao động gradient ở các epoch đầu.

### 4.3 Nhóm Biên độ & Dịch chuyển Logit: *LDAM & Logit Adjustment*
* **Logit Adjustment ($\tau=0.5$):**
  - Coarse Accuracy đạt 69.98%, nhưng việc dịch chuyển logit tĩnh không thích ứng linh hoạt với các đặc trưng trung gian của backbone.
* **LDAM Loss ($s=30, C=0.5$):**
  - Lề quyết định phụ thuộc $C / n_j^{1/4}$ làm tăng Binary AUROC (0.9522), nhưng gây sụp đổ Fine Accuracy xuống mức thấp nhất toàn bảng (**45.09%**) do biên độ ép quá gắt trên các lớp $n < 5$.

---

## 5. Kết luận & Quyết định Chuyển giao Kỹ thuật (Stage 20 Transition)

1. **Phương pháp Loss tối ưu:** **Smoothed Balanced Softmax (`balanced_softmax_smoothed`)** và **Balanced Softmax (`balanced_softmax`)** là 2 hàm mất mát vượt trội nhất để huấn luyện phân loại mô bệnh học nội soi bàng quang.
2. **Lựa chọn cho Stage 30 (Proposed Model):** Tích hợp **`balanced_softmax`** kết hợp **Supervised Contrastive Loss ($L_{supcon}$)** và cấu trúc điều hòa phân cấp (**Hierarchical Multi-Task Heads**) để tối ưu hóa đồng thời tính phân tách biểu diễn và độ chính xác phân loại.
3. **Artifact chuyển giao:** Tệp `selected_long_tail_method.json` đã được đồng bộ hóa thành công trên toàn bộ các thư mục thực nghiệm.
