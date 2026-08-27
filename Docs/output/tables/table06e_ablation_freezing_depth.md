# Table 6e: Khảo sát Độ Sâu Đóng Băng Backbone & Đánh Đổi Chi Phí Tính Toán (Backbone Freezing Depth & Compute Trade-off)

**Mục tiêu khoa học:** Phân tích đánh đổi giữa khả năng chống overfit trên đặc trưng cấp thấp, độ chính xác đuôi dài và thời gian huấn luyện.  
**Giao thức:** Đánh giá trên Split 0 Validation (Swin-Tiny Backbone, 28.23M tham số).

| # | Cấu Hình Đóng Băng | Tham Số Mở / Đóng Băng | Thời Gian / Epoch | Tổng Thời Gian Huấn Luyện | Fine Acc (%) | Primary Fine F1 (13 Lớp) | Fine F1 (Supp) | Tail Recall ($n \le 20$) | Coarse Acc (%) | Binary AUROC |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Full Fine-Tuning (100%)** | 27.52M (100%) / 0M (0%) | 59.41s (1.0x) | 1,366s (22.8 min) | 48.84% | 0.5848 | 0.4396 | **58.52%** 🥇 | **74.04%** 🥇 | **0.9333** 🥇 |
| 2 | **Partial FT: Freeze Stages 1–2** | 26.32M (95.7%) / 1.20M (4.3%) | 43.97s (**-26.0%**) | 659.6s (**-51.7%**) 🏆 | **49.22%** 🏆 | **0.6054 (+2.06%)** 🏆 | **0.4556 (+1.60%)** 🏆 | 47.04% | 70.21% | 0.9126 |
| 3 | **Partial FT: Freeze Stages 1–3** | 15.37M (55.8%) / 12.15M (44.2%)| **31.75s (-46.6%)** ⚡ | **444.5s (-67.5%)** ⚡ | 37.98% ⬇️ | 0.5503 ⬇️ | 0.4168 ⬇️ | 47.04% | 64.60% ⬇️ | 0.8739 ⬇️ |
