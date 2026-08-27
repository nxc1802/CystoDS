# Bảng 08: Kết Quả Kiểm Định Ngoại Kiểm Đa Trung Tâm (Lazo et al. 2023)

Bảng đối chuẩn ngoại kiểm trực tiếp trên toàn bộ **1.754 ảnh nội soi / 23 bệnh nhân** của bộ dữ liệu quốc tế Lazo et al. (IEEE TBME 2023) qua 3 splits (Mean $\pm$ Std):

| Phân nhóm / Chỉ số | Swin Baseline (Binary Only) | CystoHier (Proposed Direct) | Mức cải thiện ($\Delta$) |
|---|:---:|:---:|:---:|
| **A. Toàn bộ Ngoại kiểm (WLI + NBI, N=1.754 ảnh / 23 BN):** | | | |
| **AUROC** | 0.9503 $\pm$ 0.0170 | **0.9587 $\pm$ 0.0118** | **+0.0084** |
| **F1-Score** | 0.8821 $\pm$ 0.0221 | **0.8875 $\pm$ 0.0179** | **+0.0054** |
| **Accuracy (%)** | 84.80% $\pm$ 2.51% | **85.37% $\pm$ 2.05%** | **+0.57 pp** |
| **Sensitivity (%)** | 80.16% $\pm$ 3.62% | **81.23% $\pm$ 3.15%** | **+1.07 pp** |
| **Specificity (%)** | **96.30% $\pm$ 0.34%** | 95.63% $\pm$ 1.55% | -0.67 pp |
| **Balanced Accuracy (%)** | 88.23% $\pm$ 1.70% | **88.43% $\pm$ 1.36%** | **+0.20 pp** |
| **B. Phân nhóm NBI Subgroup (N=321 ảnh, Ánh sáng dải hẹp):** | | | |
| AUROC | **0.9528 $\pm$ 0.0073** | 0.9414 $\pm$ 0.0114 | -0.0114 |
| Accuracy (%) | 76.01% $\pm$ 4.35% | **81.00% $\pm$ 4.83%** | **+4.99 pp** |
| Sensitivity (%) | 69.78% $\pm$ 6.35% | **75.47% $\pm$ 6.47%** | **+5.69 pp** |
| Specificity (%) | 96.44% $\pm$ 2.27% | **99.11% $\pm$ 0.63%** | **+2.67 pp** |
| F1-Score | 0.8151 $\pm$ 0.0404 | **0.8573 $\pm$ 0.0414** | **+0.0422** |
| **C. Độ nhạy / Độ đặc hiệu theo Phân nhóm Bệnh học:** | | | |
| HGC Sensitivity (N=469) | 88.56% $\pm$ 2.46% | **89.48% $\pm$ 1.96%** | **+0.92 pp** |
| LGC Sensitivity (N=647) | **82.59% $\pm$ 4.10%** | 82.53% $\pm$ 4.99% | -0.06 pp |
| NTL Sensitivity (N=134) | 39.05% $\pm$ 5.66% | **46.02% $\pm$ 4.57%** | **+6.97 pp** |
| NST Specificity (N=504) | **96.30% $\pm$ 0.34%** | 95.63% $\pm$ 1.55% | -0.67 pp |

**Kiểm định Bootstrap Mức Bệnh Nhân (1.000 resamples):**
* **AUROC:** **$0.9574 \pm 0.0121$** (KTC 95%: $[0.9312; 0.9788]$)
* **Độ đặc hiệu (NST Specificity):** **$96.20\% \pm 1.08\%$** (KTC 95%: $[0.9381; 0.9784]$)
* **F1-Score:** **$0.8736 \pm 0.0310$** (KTC 95%: $[0.8127; 0.9296]$)
