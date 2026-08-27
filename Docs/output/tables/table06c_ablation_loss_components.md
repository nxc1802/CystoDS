# Table 6c: Bóc Tách Đóng Góp Cận Biên Của Các Thành Phần Hàm Loss (Loss Component Marginal Contributions)

**Mục tiêu khoa học:** Định lượng mức độ sụt giảm hiệu năng khi triệt tiêu từng thành phần độc lập trong hàm loss tổng thể $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{SBS}} + \lambda_1 \mathcal{L}_{\text{supcon}} + \lambda_2 \mathcal{L}_{\text{hrc}}$.  
**Giao thức:** 3 phân hoạch hold-out (`split_0`, `split_1`, `split_2`) trên tập Validation. Báo cáo $\text{Mean} \pm \text{Std}$.

| # | Cấu Hình Thử Nghiệm | Thành Phần Bị Triệt Tiêu | Binary AUROC | Binary F1 | Coarse Acc (%) | Fine Acc (%) | Primary Fine F1 (13 Lớp) | Tail Recall ($n \le 20$) | C-F Consistency (%) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Full Proposed (Anchor)** | **Đầy đủ 4 Trụ cột** | **0.9596 ± 0.032** | **0.8998 ± 0.038** | **72.76% ± 1.5%** | **51.02% ± 3.9%** | **0.6114 ± 0.023** 🏆 | **65.23% ± 7.4%** | **82.80% ± 2.6%** 🏆 |
| 2 | **w/o SupCon** | Không dùng Contrastive ($w=0$) | 0.9591 ± 0.029 | **0.9015 ± 0.038** 🥇 | 70.31% ± 2.7% | 47.74% ± 1.5% ⬇️ | 0.5627 ± 0.073 (**-4.87%**) | **67.08% ± 6.9%** 🥇 | 81.42% ± 2.7% |
| 3 | **w/o Smoothed Balanced Softmax**| Thay bằng Cross-Entropy thường | 0.9534 ± 0.021 | 0.8937 ± 0.026 | 70.93% ± 1.1% | 48.92% ± 3.2% ⬇️ | 0.6004 ± 0.026 (**-1.10%**) | 59.86% ± 9.4% (**-5.37%**) | 78.58% ± 3.7% (**-4.22%**) |
| 4 | **w/o Hierarchy Loss** | Không phạt xung đột phả hệ | 0.9583 ± 0.034 | 0.9009 ± 0.028 | 71.97% ± 1.9% | **51.29% ± 1.2%** 🥇 | 0.5890 ± 0.029 (**-2.24%**) | 65.54% ± 6.4% | 81.49% ± 2.9% |
| 5 | **w/o Data Augmentation** | Huấn luyện không qua Augmentation | 0.9596 ± 0.032 | 0.8998 ± 0.038 | **72.76% ± 1.5%** | 51.02% ± 3.9% | 0.6114 ± 0.023 | 65.23% ± 7.4% | **82.80% ± 2.6%** 🏆 |
