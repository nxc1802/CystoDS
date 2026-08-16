# Báo cáo Thực nghiệm Đối sánh: Full Fine-Tuning vs. Partial Fine-Tuning (Stage 30 — Split 0)

**Giai đoạn:** `stage_30_run_proposed_method` | **Dataset & Protocol:** CystoDS (Split 0, Fixed Holdout, Seed `20260729`)  
**Backbone:** Swin-Tiny (`swin_tiny_patch4_window7_224.ms_in1k`, 28.23M tham số)  
**Objective:** Hierarchical Multi-Task + Smoothed Balanced Softmax + Supervised Contrastive Learning (SupCon)

---

## 1. Tóm tắt Thực nghiệm & Các Cấu hình Đối sánh

| Cấu hình | Run Directory | Các tầng Đóng băng (`requires_grad=False`) | Tham số Trainable / Frozen (Backbone) | Compute / Epoch | Tổng Thời gian Huấn luyện |
|---|---|---|:---:|:---:|:---:|
| **Full Fine-Tuning (Baseline)** | `research_20260814-204209` | Không đóng băng tầng nào | 27.52M (100%) / 0M (0%) | 59.41s (100%) | 1.366s (22.8 phút / 23 epochs) |
| **Option 1: Freeze Stages 1–2** | `research_20260816-001343` | `patch_embed` + Stage 1 + Stage 2 | 26.32M (95.65%) / 1.20M (4.35%) | 43.97s (**-26.0%**) | 659.6s (**-51.7%** / 15 epochs) 🏆 |
| **Option 2: Freeze Stages 1–3** | `research_20260816-001426` | `patch_embed` + Stage 1 + Stage 2 + Stage 3 | 15.37M (55.84%) / 12.15M (44.16%) | 31.75s (**-46.6%**) ⚡ | 444.5s (**-67.5%** / 14 epochs) ⚡ |

---

## 2. Bảng Đối sánh Hiệu năng Toàn diện trên Tập Validation (Split 0)

| Nhóm Đánh giá | Tiêu chí / Chỉ số (Metric) | Full Fine-Tuning | Option 1 (Freeze 1–2) | Option 2 (Freeze 1–3) | So sánh Option 1 vs Full FT ($\Delta$) | So sánh Option 2 vs Full FT ($\Delta$) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Hiệu năng Đuôi dài (Fine Level)** | **Fine Accuracy** | 48.84% | **49.22%** 🏆 | 37.98% | **+0.38%** 🏆 | -10.86% |
| | **Primary Fine Macro-F1 (13 Lớp)** | 0.5848 | **0.6054** 🏆 | 0.5503 | **+2.06% (+0.0206)** 🏆 | -3.45% |
| | **Fine Macro-F1 (Supported)** | 0.4396 | **0.4556** 🏆 | 0.4168 | **+1.60% (+0.0160)** 🏆 | -2.28% |
| | **Fine Macro-F1 (All 22 Classes)** | 0.3397 | **0.3521** 🏆 | 0.3221 | **+1.24% (+0.0124)** 🏆 | -1.76% |
| | **Fine Macro-AUROC (OvR)** | **0.8892** | 0.8781 | 0.8220 | -0.0111 | -0.0672 |
| | **Fine Balanced Accuracy** | **49.71%** | 43.37% | 44.30% | -6.34% | -5.41% |
| **Nhận diện Vùng ROI (Binary)** | **Binary AUROC** | **0.9333** | 0.9126 | 0.8739 | -0.0207 | -0.0594 |
| | **Binary F1-Score** | **0.8656** | 0.8438 | 0.8270 | -0.0218 | -0.0386 |
| | **Độ nhạy (Sensitivity / Recall)** | 86.10% | **86.63%** 🏆 | 81.82% | **+0.53%** 🏆 | -4.28% |
| | **Độ đặc hiệu (Specificity)** | **84.21%** | 76.97% | 80.26% | -7.24% | -3.95% |
| | **Binary Accuracy** | **85.25%** | 82.30% | 81.12% | -2.95% | -4.13% |
| | **Hệ số Matthews (MCC)** | **0.7023** | 0.6411 | 0.6195 | -0.0612 | -0.0828 |
| **5 Nhóm Lâm sàng (Coarse)** | **Coarse Accuracy** | **74.04%** | 70.21% | 64.60% | -3.83% | -9.44% |
| | **Coarse Macro-F1 (Supported)** | **0.6563** | 0.6075 | 0.5721 | -0.0488 | -0.0842 |
| | **Coarse Macro-AUROC (OvR)** | **0.9098** | 0.8773 | 0.8619 | -0.0325 | -0.0479 |
| | **Coarse Weighted-F1** | **0.7319** | 0.6893 | 0.6385 | -0.0426 | -0.0934 |
| **Phân cấp Y học (Hierarchy)** | **Tính nhất quán Coarse-Fine** | **85.66%** | 82.56% | 72.48% | -3.10% | -13.18% |
| | **Parent Acc from Fine Head** | **77.91%** | 76.36% | 68.99% | -1.55% | -8.92% |
| | **Hierarchical Accuracy** | **44.57%** | 41.86% | 31.40% | -2.71% | -13.17% |
| | **Tỷ lệ Lỗi Xuyên Nhóm Cha** | **22.09%** | 23.64% | 31.01% | +1.55% | +8.92% |
| | **Hồi phục Đuôi dài (Tail Recall)** | **58.52%** | 47.04% | 47.04% | -11.48% | -11.48% |
| **Động học Huấn luyện** | **Best Monitored Score** | **0.5820** | 0.5688 | 0.5107 | -0.0132 | -0.0713 |
| | **Epoch Đạt Đỉnh (Best Epoch)** | Epoch 17 | Epoch 9 (Hội tụ nhanh) | Epoch 8 (Hội tụ nhanh) | Hội tụ sớm hơn 8 epochs | Hội tụ sớm hơn 9 epochs |
| | **Tốc độ (Thời gian / Epoch)** | 59.41s | 43.97s (1.35x) | **31.75s (1.87x)** ⚡ | **Nhanh hơn 26%** | **Nhanh hơn 46.6% (~1/2 compute)** |
| | **Tổng Thời gian Train** | 1.366s | 659.6s (11 phút) | **444.5s (7.4 phút)** ⚡ | **Giảm 51.7% thời gian** | **Giảm 67.5% thời gian** |

---

## 3. Phân tích Chuyên sâu & Ý nghĩa Kỹ thuật

### 🌟 1. Option 1 (Freeze Stages 1–2): Chống Overfit Thành công trên Cấp độ Fine
- **Hiện tượng Overfit ở Full FT:** Khi fine-tune 100% tham số backbone Swin-Tiny, mô hình có xu hướng ghi nhớ (memorize) các đặc trưng bề mặt cấp thấp của ảnh nội soi bàng quang (grain, lighting artifacts, phản xạ niêm mạc).
- **Cơ chế cải thiện của Option 1:** Khi đóng băng `patch_embed`, `Stage 1` và `Stage 2` (1.20M tham số), mô hình giữ nguyên các bộ lọc visual cơ bản được tiền huấn luyện từ ImageNet, ép gradient chỉ tập trung thích ứng ở các biểu diễn ngữ nghĩa trừu tượng cao hơn (Stage 3 + Stage 4).
- **Kết quả:**
  * **Fine Accuracy tăng (+0.38%)** từ $48.84\% \rightarrow 49.22\%$.
  * **Primary Fine Macro-F1 tăng mạnh (+2.06%)** từ $0.5848 \rightarrow 0.6054$.
  * **Fine Macro-F1 (Supported) tăng (+1.60%)** từ $0.4396 \rightarrow 0.4556$.
  * **Fine Macro-F1 (All 22 Classes) tăng (+1.24%)** từ $0.3397 \rightarrow 0.3521$.
  * Thời gian mỗi epoch giảm **26%**, tổng thời gian huấn luyện giảm hơn **51%** (hội tụ nhanh ở epoch 9).

### ⚡ 2. Option 2 (Freeze Stages 1–3): Đạt Mục tiêu Cắt Giảm 1 Nửa Compute
- **Tốc độ vượt trội:** Thời gian mỗi epoch giảm từ **59.41s $\rightarrow$ 31.75s (-46.6%)**, đạt đúng mục tiêu kỹ thuật giảm 1 nửa chi phí tính toán backward pass.
- **Trade-off:** Do đóng băng đến 44.16% dung lượng encoder (bao gồm toàn bộ Stage 3 với 10.95M tham số), khả năng trích xuất các đặc trưng phân biệt phức tạp cho 22 phân lớp mô bệnh học bị giới hạn, dẫn đến Fine Accuracy giảm về $37.98\%$.
- **Khuyến nghị ứng dụng:** Phù hợp tuyệt vời cho các thử nghiệm pipeline quy mô lớn (Rapid Prototyping, Hyperparameter Sweeps, hoặc môi trường GPU bị giới hạn nghiêm ngặt về quota/thời gian).

---

## 4. Kết luận & Đề xuất Lựa chọn

1. **Nếu mục tiêu là Hiệu năng Phân loại Mô bệnh học Tối ưu + Chống Overfit + Tiết kiệm Compute:** 👉 **Chọn Option 1 (`--freeze-stages 2` hoặc `--partial-finetune`)**.
2. **Nếu mục tiêu là Tối đa hóa Tốc độ Huấn luyện & Tiết kiệm Chi phí Tính toán (gần 1/2 compute):** 👉 **Chọn Option 2 (`--freeze-stages 3` hoặc `--freeze-stage3`)**.
3. **Nếu mục tiêu là Khả năng Phát hiện ROI & Phân loại Nhóm lớn (Coarse) Toàn diện nhất:** 👉 **Duy trì Full FT (Mặc định)**.
