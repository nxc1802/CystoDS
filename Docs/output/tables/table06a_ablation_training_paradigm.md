# Table 6a: Khảo sát Chiến Lược & Quy Trình Huấn Luyện Đa Tầng (Training Paradigm & Stage Decoupling)

**Mục tiêu khoa học:** Đánh giá tác động của quy trình huấn luyện tuần tự 3 giai đoạn (**3S-HFT**) so với huấn luyện đồng thời (1-Stage Joint) và tách rời 2 giai đoạn (2-Stage D2S-HFT).  
**Giao thức:** 3 phân hoạch hold-out (`split_0`, `split_1`, `split_2`) trên tập Validation. Báo cáo $\text{Mean} \pm \text{Std}$.

| # | Phương Pháp & Biến Thể | Chiến Lược Huấn Luyện | Binary AUROC | Binary F1 | Coarse Acc (%) | Coarse Macro-F1 | Fine Acc (%) | Fine F1 (Supp) | Fine F1 (All 22) | C-F Consistency (%) | Parent Acc (Ens) (%) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Proposed 3S-HFT v3.1** | **Rep $\rightarrow$ Coarse Align $\rightarrow$ Fine Align** | 0.9571 ± 0.021 | **0.8960 ± 0.026** 🥉 | **78.37% ± 1.0%** 🥇 | **0.6525 ± 0.012** 🥉 | **53.07% ± 4.0%** 🥇 | **0.5415 ± 0.036** 🥇 | **0.4007 ± 0.015** 🥇 | **82.28% ± 0.5%** 🥇 | **78.37% ± 1.0%** 🥇 |
| 2 | **1-Stage Joint Baseline** | Multi-Task Joint (CE + SupCon + SBS) | **0.9594 ± 0.018** 🥉 | **0.8965 ± 0.012** 🥈 | **73.64% ± 0.9%** 🥈 | **0.6576 ± 0.006** 🥇 | **52.63% ± 2.9%** 🥉 | 0.5026 ± 0.046 | 0.3718 ± 0.026 | 74.38% ± 3.2% | 71.20% ± 1.2% |
| 3 | **2-Stage Decoupled (D2S-HFT)** | Rep $\rightarrow$ Fine-Only SBS (Stage 35) | **0.9617 ± 0.028** 🥈 | 0.8912 ± 0.022 | 72.12% ± 4.6% | 0.6313 ± 0.031 | 52.20% ± 4.8% | 0.5266 ± 0.056 | 0.3893 ± 0.032 | **78.90% ± 2.4%** 🥈 | 72.50% ± 3.5% |
| 4 | **Ablation: Strategy cRT** | Phase 2 cRT Sampler (Fine Only) | **0.9617 ± 0.028** 🥈 | 0.8910 ± 0.024 | 72.12% ± 4.6% | 0.6313 ± 0.031 | 51.96% ± 2.5% | **0.5311 ± 0.048** 🥉 | **0.3930 ± 0.029** 🥉 | 78.45% ± 2.6% | 73.20% ± 3.0% |
| 5 | **Ablation: Target All Heads** | Phase 2 Unfreeze Binary + Coarse + Fine | 0.9583 ± 0.035 | 0.8875 ± 0.031 | 73.10% ± 3.4% | 0.6435 ± 0.031 | 51.55% ± 3.6% | 0.5129 ± 0.059 | 0.3794 ± 0.038 | 76.80% ± 2.9% | 72.80% ± 2.5% |
