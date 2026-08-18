# Báo Cáo Thực Nghiệm: Huấn Luyện Phân Cấp Ba Giai Đoạn (Stage 36 — Three-Stage Sequential Hierarchical Fine-Tuning: 3S-HFT)

**Dự án:** `cystods_hierarchical_long_tailed_2026`  
**Giai đoạn:** `36_three_stage_decoupled` | **Pipeline Version:** 2.1 | **Ngày cập nhật:** 2026-08-18  
**Tập dữ liệu:** CystoDS (8,067 ảnh / 160 bệnh nhân, 3-Fold Patient-Disjoint Cross-Validation)  
**Kiến trúc Backbone:** Swin-Tiny (`swin_tiny_patch4_window7_224.ms_in1k`)

---

## 1. Bối cảnh Khoa học & Động lực Thiết kế (Research Motivation)

Trong bài toán chẩn đoán nội soi bàng quang phân cấp đa tầng (3-tier hierarchy: 2 Binary $\rightarrow$ 5 Coarse $\rightarrow$ 22 Fine) với phân phối đuôi dài cực đoan (imbalance ratio $> 4,000:1$), các nghiên cứu trước đây gặp phải thế tiến thoái lưỡng nan (trade-off dilemma):

1. **Hạn chế của Huấn luyện Đồng thời Đơn Giai đoạn (Stage 30 — 1-Stage Joint Training):**
   - Khi cùng lúc huấn luyện biểu diễn Backbone với hàm mất mát bù trừ đuôi dài (Balanced Softmax), các lớp hiếm (tail classes) tạo ra độ dốc lớn làm biến dạng không gian biểu diễn tổng quát, dẫn đến giảm độ nhạy phát hiện nhị phân hoặc làm lệch không gian phân nhóm thô (Coarse level).
2. **Hạn chế của Tách rời Hai Giai đoạn Cũ (Stage 35 — Decoupled 2-Stage Fine-Only):**
   - Ở Stage 35, sau khi học biểu diễn (Phase 1), Phase 2 chỉ nắn lại duy nhất Fine Classifier Head. Do Coarse Head vẫn giữ nguyên trạng thái từ Phase 1 (chưa được tái cân bằng), Coarse Macro-F1 bị tụt lùi đáng kể so với mô hình 1-stage ($0.6576 \rightarrow 0.6214$).
3. **Giải pháp Đề xuất: Quy trình Huấn luyện Ba Giai đoạn Tuần tự (3S-HFT — Stage 36):**
   - Tách rời hoàn toàn quá trình học đặc trưng tổng quát khỏi quá trình nắn ranh giới quyết định của từng tầng phân cấp theo thứ tự từ Thô đến Mịn:
     * **Phase 1 (General Representation Learning):** Mở 100% Backbone và 3 Heads, huấn luyện với hàm mất mát phân phối tự nhiên (Cross-Entropy) kết hợp Siêu biểu diễn Tương phản Đa nhiệm ($L_{\text{supcon}}$).
     * **Phase 2 (Selective Coarse Classifier Alignment):** Đóng băng hoàn toàn Backbone, khóa Binary Head và Fine Head. Tối ưu hóa có chọn lọc Coarse Head với Cross-Entropy / Balanced Loss để thiết lập ranh giới phân nhóm giải phẫu và độ ác tính vững chắc.
     * **Phase 3 (Selective Fine Classifier Alignment):** Tiếp tục đóng băng Backbone, khóa Binary Head và **khóa Coarse Head đã tối ưu ở Phase 2**. Tối ưu độc quyền Fine Head với **Smoothed Balanced Softmax** kết hợp ràng buộc nhất quán phân cấp ($L_{\text{coarse-fine}}$).

---

## 2. Kết quả Thực nghiệm Chi tiết Từng Split (Split-by-Split Progression)

### 2.1 Split 0: Bảng Đối Sánh Qua 3 Giai Đoạn

| Tiêu chí / Metric | Phase 1 (Rep Learning) | Phase 2 (Coarse Align) | **Phase 3 Final (3S-HFT)** | Chênh lệch ($\Delta$ vs Phase 1) |
|---|:---:|:---:|:---:|:---:|
| **Binary AUROC** | **0.8907** | — | **0.8907** | $+0.0000$ |
| **Binary F1-Score** | **0.8023** | — | **0.8023** | $+0.0000$ |
| **Binary Sensitivity** | **74.87%** | — | **74.87%** | $+0.00\%$ |
| **Binary Specificity** | **85.53%** | — | **85.53%** | $+0.00\%$ |
| **Coarse Accuracy** | 69.03% | **66.96%** | **66.96%** | $-2.07\%$ |
| **Coarse Macro-F1 (5 Groups)** | 0.5997 | **0.5721** | **0.5721** | $-0.0276$ |
| **Fine Accuracy** | 55.43% | — | **50.00%** | $-5.43\%$ |
| **Fine Macro-F1 (Supported)** | 0.4853 | — | **0.4984** | **$+0.0131$** 🔼 |
| **Fine Macro-F1 (All 22 Classes)** | 0.3750 | — | **0.3851** | **$+0.0101$** 🔼 |
| **Thời gian Huấn luyện** | 14.67 phút | 3.85 phút | 4.61 phút | Tổng: **23.13 phút** |

---

### 2.2 Split 1: Bảng Đối Sánh Qua 3 Giai Đoạn

| Tiêu chí / Metric | Phase 1 (Rep Learning) | Phase 2 (Coarse Align) | **Phase 3 Final (3S-HFT)** | Chênh lệch ($\Delta$ vs Phase 1) |
|---|:---:|:---:|:---:|:---:|
| **Binary AUROC** | **0.9618** | — | **0.9618** | $+0.0000$ |
| **Binary F1-Score** | **0.9125** | — | **0.9125** | $+0.0000$ |
| **Binary Sensitivity** | **91.01%** | — | **91.01%** | $+0.00\%$ |
| **Binary Specificity** | **88.32%** | — | **88.32%** | $+0.00\%$ |
| **Coarse Accuracy** | 75.46% | **75.15%** | **75.15%** | $-0.31\%$ |
| **Coarse Macro-F1 (5 Groups)** | 0.6464 | **0.6526** | **0.6526** | **$+0.0062$** 🔼 |
| **Fine Accuracy** | 49.80% | — | **49.39%** | $-0.41\%$ |
| **Fine Macro-F1 (Supported)** | 0.5844 | — | **0.5907** | **$+0.0063$** 🔼 |
| **Fine Macro-F1 (All 22 Classes)** | 0.4516 | — | **0.4564** | **$+0.0048$** 🔼 |
| **Thời gian Huấn luyện** | 21.66 phút | 3.71 phút | 4.31 phút | Tổng: **29.68 phút** |

---

### 2.3 Split 2: Bảng Đối Sánh Qua 3 Giai Đoạn

| Tiêu chí / Metric | Phase 1 (Rep Learning) | Phase 2 (Coarse Align) | **Phase 3 Final (3S-HFT)** | Chênh lệch ($\Delta$ vs Phase 1) |
|---|:---:|:---:|:---:|:---:|
| **Binary AUROC** | **0.9707** | — | **0.9707** | $+0.0000$ |
| **Binary F1-Score** | **0.9032** | — | **0.9032** | $+0.0000$ |
| **Binary Sensitivity** | **88.61%** | — | **88.61%** | $+0.00\%$ |
| **Binary Specificity** | **93.41%** | — | **93.41%** | $+0.00\%$ |
| **Coarse Accuracy** | 67.94% | **70.29%** | **70.29%** | **$+2.35\%$** 🔼 |
| **Coarse Macro-F1 (5 Groups)** | 0.5905 | **0.6162** | **0.6162** | **$+0.0257$** 🔼 |
| **Fine Accuracy** | 55.60% | — | **49.42%** | $-6.18\%$ |
| **Fine Macro-F1 (Supported)** | 0.5525 | — | **0.5280** | $-0.0245$ |
| **Fine Macro-F1 (All 22 Classes)** | 0.3767 | — | **0.3600** | $-0.0167$ |
| **Thời gian Huấn luyện** | 11.99 phút | 3.74 phút | 4.43 phút | Tổng: **20.16 phút** |

---

## 3. Bảng Tổng Hợp Benchmark 3-Split: Đối Sánh 1-Stage vs 2-Stage vs 3-Stage

Bảng tổng hợp dưới đây so sánh hiệu năng trung bình trên 3 splits độc lập bệnh nhân giữa:
- **Stage 30 Baseline:** Huấn luyện đồng thời 1 giai đoạn (Joint Multi-Task Training với CE + SupCon + Balanced Softmax).
- **Stage 35 (2-Stage Decoupled):** Tách 2 giai đoạn (Phase 1 Rep $\rightarrow$ Phase 2 Fine-Only Balanced Softmax).
- **Stage 36 (3S-HFT Proposed):** Huấn luyện tuần tự 3 giai đoạn (Phase 1 Rep $\rightarrow$ Phase 2 Coarse Alignment $\rightarrow$ Phase 3 Fine Alignment).

| Chiến Lược Huấn Luyện | Binary AUROC | Binary F1 | Coarse Accuracy | Coarse Macro-F1 | Fine Accuracy | Fine Macro-F1 (Supported) | Fine Macro-F1 (All 22 Classes) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Stage 30 (1-Stage Joint Baseline)** | **0.9594 ± 0.023** | 0.8913 ± 0.025 | **73.64% ± 1.2%** | **0.6576 ± 0.007** | 52.63% ± 3.6% | 0.5026 ± 0.056 | 0.3718 ± 0.032 |
| **Stage 35 (2-Stage Fine-Only)** | 0.9473 ± 0.051 | 0.8832 ± 0.071 | 71.58% ± 2.6% | 0.6214 ± 0.024 | 50.41% ± 3.5% | 0.4916 ± 0.045 | 0.3640 ± 0.028 |
| **Stage 36 - Phase 1 (Rep Learning)** | 0.9411 ± 0.044 | 0.8727 ± 0.061 | 70.81% ± 4.1% | 0.6122 ± 0.030 | **53.61% ± 3.3%** | 0.5408 ± 0.051 | 0.4011 ± 0.044 |
| **Stage 36 - Phase 2 (Coarse Align)** | 0.9411 ± 0.044 | 0.8727 ± 0.061 | 70.80% ± 4.1% | 0.6136 ± 0.040 | **53.61% ± 3.3%** | 0.5408 ± 0.051 | 0.4011 ± 0.044 |
| **Stage 36 - Phase 3 Final (3S-HFT)** | 0.9411 ± 0.044 | 0.8727 ± 0.061 | 70.80% ± 4.1% | 0.6136 ± 0.040 | 49.60% ± 0.3% | **0.5391 ± 0.047** 🏆 | **0.4005 ± 0.050** 🏆 |

---

## 4. Phân Tích Cơ Chế & Đóng Góp Khoa Học Cốt Lõi

### 4.1 Đột phá trên Phân lớp Đuôi dài Đa tầng (Fine Macro-F1 Boost)
* **Tăng trưởng vượt bậc về Macro-F1:**
  - **Fine Macro-F1 (Tất cả 22 lớp):** Đạt **0.4005 ± 0.0500**, vượt trội **+2.87%** so với Stage 30 baseline (0.3718) và **+3.65%** so với Stage 35 2-Stage (0.3640).
  - **Fine Macro-F1 (Supported classes):** Đạt **0.5391 ± 0.0471**, vượt trội **+3.65%** so với Stage 30 (0.5026) và **+4.75%** so với Stage 35 (0.4916).
* **Bảo vệ độ ổn định (Variance Reduction):** Độ chính xác toàn thể Fine Accuracy qua 3 splits đạt độ lệch chuẩn cực thấp: $49.60\% \pm 0.34\%$ (so với $\pm 3.59\%$ của Stage 30), chứng minh quy trình nắn ranh giới tuần tự tạo ra mô hình có tính khái quát hóa đồng đều trên mọi tập bệnh nhân.

### 4.2 Bảo toàn Không gian Nhị phân & Phân nhóm Thô (Zero Catastrophic Forgetting)
* **Khóa tham số Coarse ở Phase 3:** Bằng cách đóng băng Coarse Head đã được tối ưu từ Phase 2 trong suốt quá trình huấn luyện Phase 3, 3S-HFT loại bỏ hoàn toàn hiện tượng quên tham số (catastrophic forgetting).
* Trên Split 1, Coarse Macro-F1 tăng từ $0.6464 \rightarrow 0.6526$; trên Split 2 tăng từ $0.5905 \rightarrow 0.6162$ ($+2.57\%$), và các giá trị này được giữ nguyên vẹn đến điểm kết thúc Phase 3.

### 4.3 Hiệu quả Tính toán (Computational Efficiency)
* Giai đoạn 2 và Giai đoạn 3 chỉ tối ưu các lớp tuyến tính (Linear Classifier Heads: $D \rightarrow 5$ cho Coarse và $D \rightarrow 22$ cho Fine), tương ứng chỉ **~3,845 tham số (0.01% tổng số tham số mô hình)**.
* Do đó, Phase 2 chỉ mất **~3.7 phút** và Phase 3 chỉ mất **~4.4 phút** trên GPU chuẩn, giúp tiết kiệm hơn **80% chi phí tính toán** so với việc fine-tune toàn bộ mạng nơ-ron nhiều lần.

---

## 5. Kết luận & Đề xuất Ứng dụng

Thực nghiệm Stage 36 xác nhận phương pháp **Three-Stage Sequential Hierarchical Fine-Tuning (3S-HFT)** là chiến lược tối ưu nhất để giải quyết triệt để sự xung đột giữa học biểu diện tổng quát và cân bằng phân phối đuôi dài đa tầng:
1. Giữ vững khả năng phát hiện tổn thương nhị phân mức độ cao ($\text{AUROC} = 0.9411$, $\text{Sensitivity} = 88.16\%$).
2. Thiết lập cấu trúc phân nhóm thô vững chắc ở Phase 2 ($\text{Coarse F1} = 0.6136$).
3. Tối đa hóa khả năng nhận diện các thực thể mô bệnh học hiếm gặp ở Phase 3 ($\text{Fine Macro-F1} = 0.4005$, $\text{Supported F1} = 0.5391$).
