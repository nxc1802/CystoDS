# Table 7: Phân Tích Chi Tiết Từng Lớp Lâm Sàng & Benchmark Thời Gian Thực (Edge Runtime)

### Phần A: Hiệu Năng Chi Tiết Theo 5 Lớp Coarse (Split 0 Validation)

| Nhóm Lâm Sàng (Coarse Group) | Số Mẫu Thật (Support) | Số Mẫu Dự Đoán | Precision | Recall (Sensitivity) | F1-Score | Macro AUROC (OvR) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Malignant (Ác tính)** | 142 | 149 | 0.7852 | 0.8239 | **0.8041** | 0.9175 |
| **Non-malignant (Không ác tính)** | 32 | 30 | 0.2000 | 0.1875 | 0.1935 | 0.8348 |
| **Normal mucosa (Niêm mạc lành)** | 81 | 111 | 0.6757 | **0.9259** | 0.7813 | 0.9512 |
| **Anatomical landmarks (Giải phẫu)** | 31 | 11 | **0.8182** | 0.2903 | 0.4286 | 0.9027 |
| **Foreign bodies (Dị vật / Dụng cụ)** | 43 | 28 | **0.9286** | 0.6047 | 0.7324 | **0.9689** |

---

### Phần B: Benchmark Thời Gian Thực & Chi Phí Bộ Nhớ Trên Thiết Bị Biên (Apple Silicon MPS, FP32)

| Chế Độ / Batch | Số Vòng Đo | Forward Model (ms/ảnh) | Thông Lượng Forward (FPS) | Pipeline End-to-End (ms/ảnh) | Thông Lượng End-to-End (FPS) | Bộ Nhớ MPS Cấp Phát (MiB) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Batch = 1** | 60 | **12.081 ± 0.95 ms** | **82.78 FPS** ⚡ | **15.000 ± 1.12 ms** | **66.67 FPS** ⚡ | 110.83 MiB |
| **Batch = 8** | 60 | **10.167 ± 0.42 ms** | **98.36 FPS** | — | — | 113.99 MiB |
| **Batch = 32** | 20 | **9.954 ± 0.38 ms** | **100.46 FPS** | — | — | 128.00 MiB |
