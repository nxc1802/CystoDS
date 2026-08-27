# Table 6b: Khảo sát Lịch Trình Trọng Số Phân Cấp (Hierarchy Loss Scheduling & Curriculum Warmup)

**Mục tiêu khoa học:** Chứng minh hiện tượng thắt cổ chai phân cấp sớm (*Early Hierarchy Bottleneck*) và vai trò của lịch trình **Curriculum Warmup** trong việc giải phóng không gian biểu diễn.  
**Giao thức:** 3 phân hoạch hold-out (`split_0`, `split_1`, `split_2`) trên tập Validation. Báo cáo $\text{Mean} \pm \text{Std}$.

| # | Biến Thể Lịch Trình | Công Thức Biến Thiên Trọng Số | Binary AUROC | Coarse Acc (%) | Coarse Macro-F1 | Fine Acc (%) | Fine F1 (Supp) | Fine F1 (All 22) | C-F Consistency (%) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Proposed Curriculum Warmup** | **$w_{\text{hrc}}(t) = 0.25 \cdot \min(1.0, t/12)$** | **0.9571 ± 0.021** | **73.57% ± 1.8%** 🥇 | **0.6525 ± 0.012** 🥇 | **53.07% ± 4.0%** 🥇 | **0.5415 ± 0.036** 🥇 | **0.4007 ± 0.015** 🥇 | **82.28% ± 0.5%** 🥇 |
| 2 | **Method A (Two-Phase)** | $w=0$ ở P1, $w=0.25$ ở P2/P3 | 0.9521 ± 0.028 | 71.76% ± 1.1% | 0.6371 ± 0.008 | 48.98% ± 4.1% | 0.5240 ± 0.025 | 0.3883 ± 0.017 | 81.55% ± 1.8% |
| 3 | **Fixed Hierarchy Weight** | $w_{\text{hrc}} = 0.25$ cố định xuyên suốt | 0.9466 ± 0.031 | 70.09% ± 2.3% | 0.6119 ± 0.018 | 47.21% ± 2.4% | 0.5199 ± 0.048 | 0.3844 ± 0.023 | **81.88% ± 2.0%** 🥈 |
| 4 | **w/o Hierarchy Loss** | $w_{\text{hrc}} = 0$ (Không ràng buộc) | **0.9649 ± 0.022** 🥇 | **73.46% ± 1.7%** 🥈 | **0.6426 ± 0.009** 🥈 | **52.72% ± 2.0%** 🥈 | **0.5414 ± 0.077** 🥈 | **0.3998 ± 0.047** 🥈 | 72.40% ± 4.1% ⬇️ |
| 5 | **w/o Binary-Coarse Loss** | $w_{\text{bc}} = 0, w_{\text{cf}} = 0.25$ | 0.9528 ± 0.037 | 71.75% ± 0.7% | 0.6206 ± 0.011 | 49.06% ± 2.8% | 0.5120 ± 0.035 | 0.3810 ± 0.022 | **82.43% ± 2.8%** 🥇 |
| 6 | **w/o Coarse-Fine Loss** | $w_{\text{bc}} = 0.25, w_{\text{cf}} = 0$ | 0.9605 ± 0.021 | 71.51% ± 4.0% | 0.6313 ± 0.029 | 52.92% ± 4.1% | 0.5312 ± 0.028 | 0.3915 ± 0.019 | 81.31% ± 3.1% |
