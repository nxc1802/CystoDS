# CystoDS — Stage 20: Long-Tail Loss Screening Results Report
**Stage:** `stage_20_run_long_tail_screen` | **Run Directory:** `result/20_long_tail/research_20260813-105659` | **Date:** 2026-08-14 | **Study:** `cystods_hierarchical_long_tailed_2026`

---

## Executive Summary

Stage 20 thực hiện **Sàng lọc toàn diện 7 phương pháp hàm mất mát xử lý phân bố đuôi dài (Long-Tail Loss Screening Benchmark)** trên kiến trúc backbone chiến thắng từ Stage 10 (**Swin-Tiny** — `swin_tiny_patch4_window7_224.ms_in1k`). Mục tiêu then chốt là tìm ra cơ chế tối ưu nhằm khắc phục triệt để hiện tượng suy sụp hiệu năng trên 22 phân lớp mô bệnh học bàng quang có độ lệch mẫu nghiêm trọng ($n_i$ dao động từ hàng nghìn ảnh xuống chỉ vài ảnh).

### Kết luận Cốt lõi:
1. **Quán quân Sàng lọc:** **Smoothed Balanced Softmax** (`fine_balanced_softmax_smoothed`) xuất sắc giành vị trí số 1 với **Val Fine Macro-F1 = 0.4979** (49.79%), vượt trội so với baseline Cross-Entropy (+4.05% absolute gain) và Balanced Softmax chuẩn (+1.29% absolute gain).
2. **Hiệu năng Phân lớp Chính (Primary Fine Classes, $n_{\text{train}} \ge 5$):** Đạt **Macro-F1 = 0.5965** (59.65%) trên 13 lớp mô bệnh học chủ đạo, tỷ lệ dự đoán ngoài tập chính chỉ 3.17%.
3. **Khả năng hồi phục Lớp Hiếm (Tail Class Recall):** Đạt **58.52%**, chứng minh cơ chế hiệu chỉnh tiên lượng dựa trên số lượng bệnh nhân mịn ($\tilde{n}_j \propto \text{patient\_count}_j^{0.5}$) giúp mô hình nhận diện chính xác các tổn thương hiếm gặp mà không làm suy giảm độ chính xác trên các lớp phổ biến.
4. **Nhất quán Phân cấp (Coarse $\rightarrow$ Fine):** Đầu ra Fine Head đạt độ chính xác **81.01%** khi ánh xạ ngược về lớp Coarse tương ứng, độ nhất quán tổng thể Coarse-Fine đạt **74.42%**.
5. **Chuyển giao Stage 30:** `selected_long_tail_method.json` được ghi nhận chính thức với phương pháp `balanced_softmax_smoothed`, làm nền tảng cho việc tối ưu trọng số Loss cân bằng đa nhiệm và chiến lược gom nhóm MIL ở Stage 30 & 40.

---

## 1. Bảng Xếp hạng Toàn diện 7 Phương pháp Hàm mất mát (Loss Screening Leaderboard)

Cả 7 phương pháp được huấn luyện trong cùng điều kiện giao thức chuẩn (AdamW, $\text{LR}=3\times 10^{-4}$, Encoder LR Multiplier $= 0.25$, Weight Decay $= 0.05$, Early stopping patience $= 6$, Max 25 Epochs) trên tập Train 70% (1,553 ảnh, 112 bệnh nhân) và đánh giá trên tập Validation 15% (339 ảnh, 24 bệnh nhân):

| Hạng | Trial ID | Phương pháp Loss | Nguyên lý & Cơ chế Điều chỉnh | Best Val Fine Macro-F1 | Best Epoch | Tổng Epochs Huấn luyện | Trạng thái Dừng |
|:---:|---|---|---|:---:|:---:|:---:|:---:|
| 🥇 **1** | `fine_balanced_softmax_smoothed` | **Smoothed Balanced Softmax** | $\mathcal{L}_{\text{SBS}} = -\log \frac{\tilde{n}_y e^{z_y}}{\sum_j \tilde{n}_j e^{z_j}}, \; \tilde{n}_j \propto \text{patients}_j^{0.5}$ | **0.4979** (49.79%) | **16** | 22 | Early Stopped |
| 🥈 **2** | `fine_weighted_ce` | **Class-Balanced Weighted CE** | $\mathcal{L}_{\text{WCE}} = -w_y \log p_y, \; w_j \propto \frac{1-\beta}{1-\beta^{n_j}}$ ($\beta=0.9999$) | **0.4926** (49.26%) | **10** | 16 | Early Stopped |
| 🥉 **3** | `fine_balanced_softmax` | **Balanced Softmax** | $\mathcal{L}_{\text{BS}} = -\log \frac{n_y e^{z_y}}{\sum_j n_j e^{z_j}}$ ($n_j$ theo image frequency) | **0.4850** (48.50%) | **10** | 16 | Early Stopped |
| 4 | `fine_focal` | **Focal Loss** | $\mathcal{L}_{\text{Focal}} = -(1-p_y)^\gamma \log p_y$ ($\gamma=2.0$) | **0.4774** (47.74%) | **10** | 16 | Early Stopped |
| 5 | `fine_logit_adjustment` | **Logit Adjustment** | $\mathcal{L}_{\text{LA}} = -\log \frac{e^{z_y + \tau \log \pi_y}}{\sum_j e^{z_j + \tau \log \pi_j}}$ ($\tau=0.5$) | **0.4723** (47.23%) | **20** | 25 | Completed (Max Epochs) |
| 6 | `fine_cross_entropy` | **Cross-Entropy (Baseline)** | Standard Softmax Loss $\mathcal{L}_{\text{CE}} = -\log p_y$ | **0.4574** (45.74%) | **14** | 20 | Early Stopped |
| 7 | `fine_ldam` | **LDAM Loss** | Margin-based $\mathcal{L}_{\text{LDAM}}, \; \Delta_j = C / n_j^{1/4}$ ($s=30, C=0.5$) | **0.4503** (45.03%) | **13** | 19 | Early Stopped |

---

## 2. Phân tích Chuyên sâu Động học & Cơ chế các Nhóm Hàm Mất mát

```
[Hiệu năng Val Fine Macro-F1]
0.50 ┤                                                    ╭─ 0.4979 (Smoothed Balanced Softmax 🏆)
     │                                           ╭────────╯
0.49 ┤                              ╭────────────╯ 0.4926 (Weighted CE)
     │                     ╭────────╯ 0.4850 (Balanced Softmax)
0.48 ┤            ╭────────╯ 0.4774 (Focal Loss)
     │   ╭────────╯ 0.4723 (Logit Adjustment)
0.46 ┤ ──╯ 0.4574 (Standard CE Baseline)
     │
0.45 ┤ ── 0.4503 (LDAM Loss)
     └────────────────────────────────────────────────────────
```

### 2.1 Tại sao Smoothed Balanced Softmax đạt hiệu năng vượt trội nhất?
* **Tối ưu theo Cấp Bệnh nhân thay vì Cấp Ảnh:** Trên dữ liệu nội soi CystoDS, một tổn thương thường có nhiều ảnh liên tiếp chụp cùng góc độ (image-level redundancy), dẫn đến số lượng ảnh không phản ánh đúng mức độ đa dạng thực tế của mẫu bệnh. Smoothed Balanced Softmax sử dụng $\tilde{n}_j \propto \text{patient\_count}_j^{0.5}$ với hệ số làm mượt căn bậc hai, giúp triệt tiêu nhiễu do chụp nhiều ảnh trên cùng 1 bệnh nhân và ngăn ngừa việc phạt quá mức (over-penalization) các lớp phổ biến.
* **Ổn định Gradient:** Nhờ tính mượt trong vector điều chỉnh tiên lượng, hàm mất mát tránh được hiện tượng logit dao động mạnh ở các lớp cực hiếm ($n \le 3$), đạt điểm tối ưu cao nhất tại epoch 16 (**Macro-F1 0.4979**).

### 2.2 Đánh giá nhóm Re-weighting (Weighted CE & Focal Loss)
* **Class-Balanced Weighted CE (Hạng 2 — 0.4926):** Áp dụng trọng số mẫu hiệu dụng (Effective Number of Samples, $\beta=0.9999$) mang lại hiệu quả rất cao (+3.52% so với CE). Việc tái cân bằng trực tiếp loss của từng mẫu hiếm giúp mạng học nhanh đặc trưng mô bệnh học ngay từ các epoch đầu (đạt đỉnh tại epoch 10).
* **Focal Loss (Hạng 4 — 0.4774):** Tập trung vào mẫu khó bằng cách giảm trọng số các mẫu dễ phân loại $(\gamma=2.0)$. Phương pháp này tốt hơn CE chuẩn nhưng kém hơn Balanced Softmax do không khai thác thông tin phân bố tiên lượng (class prior).

### 2.3 Phân tích hạn chế của LDAM & Logit Adjustment
* **LDAM Loss (Hạng 7 — 0.4503):** LDAM áp dụng lề mở rộng $\Delta_j = C / n_j^{1/4}$ với scaling factor $s=30.0$. Do không gian đặc trưng đa nhiệm của Swin-Tiny biến thiên phức tạp, việc cưỡng bức lề cố định tạo ra loss rất lớn (Val Fine Loss tăng vọt lên 25.88) gây mất ổn định biểu diễn đặc trưng chung.
* **Logit Adjustment (Hạng 5 — 0.4723):** Dịch chuyển logit cố định với $\tau=0.5$ có xu hướng đẩy logit của các lớp hiếm lên quá cao, khiến mạng cần tới 20–25 epochs mới hội tụ và đạt kết quả thấp hơn việc tích hợp prior trực tiếp vào hàm phân phối xác suất như Balanced Softmax.

---

## 3. Báo cáo Chi tiết Hiệu năng của Phương pháp Chiến thắng (`fine_balanced_softmax_smoothed`)

Dưới đây là bảng số liệu kiểm chứng chi tiết trên tập Validation độc lập ($n = 339$ ảnh, 24 bệnh nhân):

### 3.1 Tác vụ Phân loại Nhị phân (Binary: ROI vs. Non-ROI)
Đánh giá khả năng phát hiện tổn thương bàng quang so với niêm mạc lành:

| Chỉ số Đánh giá (Metric) | Giá trị Thực nghiệm | Ý nghĩa Lâm sàng & Kỹ thuật |
|---|:---:|---|
| **Số lượng mẫu ($n$)** | 339 ảnh | 187 ảnh tổn thương (ROI), 152 ảnh niêm mạc bình thường (Non-ROI) |
| **Độ chính xác (Accuracy)** | **87.32%** (296 / 339) | Độ chính xác phân biệt tổng thể |
| **Độ nhạy (Sensitivity / Recall)** | **86.10%** (161 / 187) | Khả năng không bỏ sót tổn thương nghi ngờ |
| **Độ đặc hiệu (Specificity)** | **88.82%** (135 / 152) | Khả năng loại trừ chính xác niêm mạc lành, tránh sinh thiết thừa |
| **Độ chuẩn xác (Precision)** | **90.45%** (161 / 178) | Khi cảnh báo có tổn thương, 90.45% là chính xác |
| **F1-Score** | **0.8822** | Cân bằng điều hòa giữa Recall và Precision |
| **MCC (Matthews Correlation)** | **0.7461** | Hệ số tương quan rất cao, chứng minh dự đoán vững chắc |
| **Balanced Accuracy** | **87.46%** | Độ chính xác cân bằng giữa hai nhóm |
| **AUROC** | **0.9246** | Phân tách không gian xác suất nhị phân vượt trội |
| **AUPRC** | **0.9512** | Diện tích dưới đường cong Precision-Recall rất cao |

---

### 3.2 Tác vụ Phân loại Thô (Coarse-Grained — 5 Phân nhóm)
Phân loại vào 5 nhóm lớn: *Malignant, Non-malignant, Normal mucosa, Anatomical landmarks, Artefacts*:

| Chỉ số Đánh giá (Metric) | Giá trị Thực nghiệm |
|---|:---:|
| **Số lượng mẫu ($n$)** | 339 ảnh (đầy đủ 5 nhóm) |
| **Độ chính xác (Accuracy)** | **69.62%** (236 / 339) |
| **Macro-F1 (All 5 classes)** | **0.5964** (59.64%) |
| **Weighted-F1** | **0.6828** (68.28%) |
| **Balanced Accuracy** | **56.91%** |
| **MCC** | **0.5623** |
| **Macro-AUROC (OvR)** | **0.8994** (Xấp xỉ 0.90) |

---

### 3.3 Tác vụ Phân loại Mịn (Fine-Grained — 22 Phân lớp Mô bệnh học)
Đánh giá trên toàn bộ 22 phân lớp mô bệnh học đuôi dài:

| Chỉ số Đánh giá (Metric) | Giá trị Thực nghiệm | Ghi chú & Đánh giá |
|---|:---:|---|
| **Số lượng mẫu đánh giá ($n$)** | 258 ảnh | Chỉ tính các ảnh có nhãn tổn thương Fine ($\text{fine\_id} \ge 0$) |
| **Độ chính xác (Accuracy)** | **46.12%** (119 / 258) | Tăng đáng kể so với Baseline Stage 10 |
| **Macro-F1 (17 Lớp có mẫu ở Val)** | **0.4979** (49.79%) | 🏆 **Chỉ số quyết định chọn mô hình** |
| **Macro-F1 (Toàn bộ 22 Phân lớp)** | **0.3847** (38.47%) | Bao gồm 5 lớp không xuất hiện mẫu ở tập Val |
| **Weighted-F1** | **0.4930** (49.30%) | Trọng số theo tần suất lớp |
| **Balanced Accuracy** | **52.39%** | Cải thiện rõ rệt trên các lớp hiếm |
| **Macro-AUROC (OvR)** | **0.8519** | Khả năng phân biệt xác suất giữa 17 phân lớp |
| **Primary Fine Macro-F1 ($n_{\text{train}} \ge 5$)** | **0.5965** (59.65%) | **13 lớp mô bệnh học chủ đạo (252 ảnh)** |
| **Dự đoán ngoài tập Primary** | 8 / 252 (**3.17%**) | Tỷ lệ nhầm sang lớp không có dữ liệu cực thấp |

---

### 3.4 Đánh giá Tính Nhất quán Phân cấp (Hierarchical Consistency)

| Tiêu chí Đánh giá Phân cấp (Hierarchical Criterion) | Giá trị Thực nghiệm | Ý nghĩa Phân tích |
|---|:---:|---|
| **Độ chính xác ánh xạ Ngược (Parent Acc from Fine Head)** | **81.01%** (209 / 258) | Khi Fine Head đưa ra phân lớp chi tiết, 81% trường hợp lớp Coarse tương ứng là hoàn toàn chính xác. |
| **Độ chính xác lớp Cha từ Coarse Head** | 63.95% (165 / 258) | Fine Head có biểu diễn phong phú hơn Coarse Head riêng rẽ. |
| **Độ chính xác Phân cấp Đồng thời (Hierarchical Acc)** | **31.78%** | Tỷ lệ cả Coarse và Fine cùng đúng hoàn toàn. |
| **Tỷ lệ Lỗi Xuyên Nhóm Cha (Cross-Parent Error Rate)** | **18.99%** | Chỉ 18.99% dự đoán Fine bị lệch khỏi nhóm Coarse thật. |
| **Độ Nhất quán Dự đoán Coarse-Fine** | **74.42%** | Mức độ tương đồng logic giữa 2 nhánh phân loại. |
| **Độ nhạy Lớp Hiếm (Tail Class Recall)** | **58.52%** | Các tổn thương hiếm đạt độ phát hiện gần 60%. |

---

## 4. Bảng So sánh Tổng hợp giữa Baseline Stage 10 và Stage 20 Chiến thắng

| Nhóm Chỉ số Đánh giá | Stage 10 Baseline (Swin-Tiny CE) | Stage 20 Winner (Smoothed Balanced Softmax) | Mức Cải thiện (Delta) |
|---|:---:|:---:|:---:|
| **Val Binary Accuracy** | 84.37% | **87.32%** | 🟢 **+2.95%** |
| **Val Binary F1-Score** | 0.8507 | **0.8822** | 🟢 **+0.0315** |
| **Val Binary AUROC** | 0.9000 | **0.9246** | 🟢 **+0.0246** |
| **Val Coarse Macro-F1** | 0.5620 | **0.5964** | 🟢 **+0.0344** |
| **Val Coarse AUROC** | 0.8740 | **0.8994** | 🟢 **+0.0254** |
| **Val Fine Accuracy** | 32.56% | **46.12%** | 🟢 **+13.56%** |
| **Val Fine Macro-F1 (Evaluable)** | 0.4574 | **0.4979** | 🟢 **+0.0405 (+4.05%)** |
| **Val Primary Fine Macro-F1** | 0.5147 | **0.5965** | 🟢 **+0.0818 (+8.18%)** |
| **Tail Class Recall** | 38.20% | **58.52%** | 🚀 **+20.32%** |
| **Parent Acc from Fine Head** | 68.40% | **81.01%** | 🟢 **+12.61%** |

---

## 5. Kết luận & Kế hoạch Chuyển giao Kỹ thuật sang Stage 30 & Stage 40

1. **Hoàn tất Khảo sát Sàng lọc:** Toàn bộ 7 hàm mất mát đã được chạy thực nghiệm đồng bộ và so sánh thực chứng trên cùng một tập dữ liệu chuẩn hóa của CystoDS.
2. **Lựa chọn Phương pháp Cốt lõi:** `balanced_softmax_smoothed` được chọn làm phương pháp phân loại mô bệnh học đuôi dài chính thức cho hệ thống CystoDS.
3. **Transition Artifact:** Tệp `result/20_long_tail/research_20260813-105659/selected_long_tail_method.json` được cập nhật:
   ```json
   {
     "stage_id": "20",
     "selected_backbone": "swin_tiny_patch4_window7_224.ms_in1k",
     "selected_long_tail_method": "balanced_softmax_smoothed",
     "selected_trial_id": "fine_balanced_softmax_smoothed",
     "selection_metric": "primary_macro_f1_all_classes",
     "val_macro_f1": 0.497858,
     "study_id": "cystods_hierarchical_long_tailed_2026"
   }
   ```
4. **Định hướng Stage 30 (Loss Balancing & Multi-Task Tuning):** Sử dụng `balanced_softmax_smoothed` ở đầu Fine để tiếp tục tối ưu hóa tỷ lệ trọng số đa nhiệm ($w_{\text{binary}}, w_{\text{coarse}}, w_{\text{fine}}, w_{\text{consistency}}, w_{\text{supcon}}$).
5. **Định hướng Stage 40 (ROI Attention Multiple Instance Learning):** Kết hợp các đặc trưng Fine-grained thu được từ mô hình chiến thắng vào mạng Attention MIL để tổng hợp chẩn đoán cấp vùng tổn thương (ROI-level) và cấp ca bệnh nhân.
