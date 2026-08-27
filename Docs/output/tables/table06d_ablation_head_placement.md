# Table 6d: Khảo sát Vị Trí Trích Xuất Đặc Trưng Theo Độ Sâu (Architectural Head Placement)

**Mục tiêu khoa học:** So sánh kiến trúc **Shared Late-Stage** tại Stage 4 so với việc phân tán các Classifier Heads tại các tầng trung gian (**Intermediate Multi-Stage Heads**).  
**Giao thức:** 3 phân hoạch hold-out (`split_0`, `split_1`, `split_2`) trên tập Validation.

| # | Biến Thể Vị Trí Head | Cơ Chế Trích Xuất Đặc Trưng | Binary AUROC | Binary Specificity | Coarse Acc (%) | Fine Acc (%) | Fine F1 (Supp) | Fine F1 (All 22) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Proposed Shared Late-Stage** | **Toàn bộ 3 Heads tại Stage 4 (768-d)** | **0.9571 ± 0.021** | **88.60% ± 5.0%** | **73.57% ± 1.8%** | **53.07% ± 4.0%** | **0.5415 ± 0.036** | **0.4007 ± 0.015** |
| 2 | **Multi-Stage Intermediate Heads** | S2 (192-d) $\rightarrow$ Bin, S3 (384-d) $\rightarrow$ Coarse, S4 (768-d) $\rightarrow$ Fine | 0.8355 ± 0.045 (**-12.2%**) | 75.66% ± 4.8% (**-12.9%**) | 68.73% ± 3.1% (**-4.8%**) | 42.64% ± 3.8% (**-10.4%**) | 0.4806 ± 0.049 (**-6.1%**) | 0.3714 ± 0.031 (**-2.9%**) |
