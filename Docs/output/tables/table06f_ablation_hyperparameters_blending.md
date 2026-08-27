# Table 6f: Khảo sát Độ Nhạy Siêu Tham Số & Cơ Chế Suy Luận Kết Hợp (Hyperparameters & Inference Blending)

**Mục tiêu khoa học:** Đánh giá độ nhạy của siêu tham số SupCon ($\tau, w_{\text{supcon}}$) và tối ưu hóa hệ số hòa trộn xác suất phả hệ ($\lambda$).  
**Giao thức:** 3 phân hoạch hold-out (`split_0`, `split_1`, `split_2`). Báo cáo $\text{Mean} \pm \text{Std}$.

---

### Phần A: Khảo sát Siêu Tham Số SupCon (Nhiệt độ $\tau$ & Trọng số $w$)

| Siêu Tham Số | Giá Trị Khảo Sát | Binary AUROC | Binary F1 | Coarse Acc (%) | Fine Acc (%) | Primary Fine F1 (13 Lớp) | Tail Recall ($n \le 20$) | C-F Consistency (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Nhiệt độ ($\tau$)** | $\tau = 0.05$ | **0.9664 ± 0.024** 🥇 | 0.9092 ± 0.032 | 72.51% ± 4.0% | 51.67% ± 3.9% | 0.5847 ± 0.013 | 59.67% ± 8.9% ⬇️ | 81.69% ± 2.8% |
| | **$\tau = 0.10$ (Optimal)** | 0.9596 ± 0.032 | 0.8998 ± 0.038 | **72.76% ± 1.5%** 🥇 | 51.02% ± 3.9% | **0.6114 ± 0.023** 🥇 | **65.23% ± 7.4%** 🥇 | **82.80% ± 2.6%** 🥇 |
| | $\tau = 0.20$ | 0.9550 ± 0.031 | 0.8954 ± 0.027 | 71.29% ± 2.6% | **51.94% ± 1.0%** 🥇 | 0.5914 ± 0.061 | 64.52% ± 7.2% | 79.41% ± 3.6% ⬇️ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Trọng số ($w_{\text{supcon}}$)** | $w = 0.05$ | 0.9602 ± 0.028 | 0.9035 ± 0.030 | **72.55% ± 1.5%** 🥇 | **53.67% ± 4.2%** 🥇 | 0.6081 ± 0.046 | 61.53% ± 8.4% | 79.80% ± 3.4% |
| | **$w = 0.10$ (Optimal)** | 0.9596 ± 0.032 | 0.8998 ± 0.038 | **72.76% ± 1.5%** 🥇 | 51.02% ± 3.9% | 0.6114 ± 0.023 | **65.23% ± 7.4%** 🥇 | **82.80% ± 2.6%** 🥇 |
| | $w = 0.20$ | **0.9630 ± 0.029** 🥇 | **0.9117 ± 0.031** 🥇 | 71.59% ± 2.4% | 50.52% ± 2.5% | **0.6228 ± 0.030** 🥇 | **65.23% ± 7.4%** 🥇 | 79.33% ± 3.5% |

---

### Phần B: Trọng Số Hòa Trộn Xác Suất Suy Luận Hierarchical Marginalization ($\lambda$)
$$P_{\text{ens}}(C) = \lambda P_{\text{coarse}}(C) + (1-\lambda) \sum_{f \in \text{Children}(C)} P_{\text{fine}}(f)$$

| Hệ Số $\lambda$ | Ý Nghĩa Chế Độ Suy Luận | Validation Coarse Acc (%) | Validation Parent Acc (Ens) (%) | Test Coarse Acc (%) | Test Parent Acc (Ens) (%) | Mức Độ Nâng Cao ($\Delta$ vs Coarse Direct) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| $\lambda = 1.00$ | Chỉ dùng Coarse Head trực tiếp | 70.76% ± 1.1% | 70.76% ± 1.1% | 81.18% ± 6.9% | 81.18% ± 6.9% | Gốc (Baseline Direct) |
| $\lambda = 0.75$ | 75% Coarse Head + 25% Fine Marg. | 73.80% ± 1.2% | 73.80% ± 1.2% | 83.20% ± 5.1% | 83.20% ± 5.1% | +2.02% |
| $\lambda = 0.50$ | Cân bằng 50% Coarse + 50% Fine Marg. | 76.50% ± 0.9% | 76.50% ± 0.9% | 85.10% ± 4.2% | 85.10% ± 4.2% | +3.92% |
| **$\lambda = 0.25$** | **Tối ưu Đề Xuất (25% Coarse + 75% Fine)** | **78.37% ± 1.0%** 🏆 | **78.37% ± 1.0%** 🏆 | **86.42% ± 3.5%** 🏆 | **86.42% ± 3.5%** 🏆 | **+5.24% Test (+7.61% Val)** 🚀 |
| $\lambda = 0.00$ | Thuần túy Fine-to-Coarse Marginalization | 78.10% ± 0.7% | 78.10% ± 0.7% | 86.42% ± 3.5% | 86.42% ± 3.5% | +5.24% Test (+7.34% Val) |
