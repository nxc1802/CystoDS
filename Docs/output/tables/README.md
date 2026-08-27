# Hệ Thống Bảng Biểu Xuất Bản Bài Báo Khoa Học (CystoDS Publication Tables)

Thư mục này chứa toàn bộ các bảng biểu thống kê thực nghiệm của dự án **CystoDS** (`cystods_hierarchical_long_tailed_2026`), được xuất bản đồng thời ở 2 định dạng:
1. **Markdown (`.md`):** Phục vụ xem nhanh, hiển thị trên GitHub/Web/IDE.
2. **LaTeX (`.tex`):** Định dạng chuẩn `booktabs` / `tabularx` / `adjustbox`, sẵn sàng chèn trực tiếp (`\input{...}`) vào manuscript (Springer LNCS, IEEE, Elsevier).

---

## Danh Mục Các Bảng (Table Catalog)

| Mã Bảng | File LaTeX (`.tex`) | File Markdown (`.md`) | Tiêu Đề Bảng & Nội Dung |
|:---:|---|---|---|
| **Table 1** | [`table01_dataset_statistics.tex`](table01_dataset_statistics.tex) | [`table01_dataset_statistics.md`](table01_dataset_statistics.md) | Thống kê tập dữ liệu CystoDS qua 3 tầng phân cấp phả hệ (Stage 00). |
| **Table 2** | [`table02_related_work.tex`](table02_related_work.tex) | [`table02_related_work.md`](table02_related_work.md) | Tổng hợp và đối chiếu các công trình nghiên cứu phân loại nội soi bàng quang liên quan. |
| **Table 3** | [`table03_test_benchmark.tex`](table03_test_benchmark.tex) | [`table03_test_benchmark.md`](table03_test_benchmark.md) | **[BẢNG CHÍNH]** Đối chuẩn độc lập trên tập Hold-out Test (Proposed 3S-HFT vs Baselines). |
| **Table 4** | [`table04_backbone_screening.tex`](table04_backbone_screening.tex) | [`table04_backbone_screening.md`](table04_backbone_screening.md) | Sàng lọc 4 họ kiến trúc backbone & chế độ đơn nhiệm / đa nhiệm (Stage 10). |
| **Table 5** | [`table05_long_tail_loss.tex`](table05_long_tail_loss.tex) | [`table05_long_tail_loss.md`](table05_long_tail_loss.md) | Sàng lọc 7 hàm mất mát xử lý phân bố đuôi dài trên kiến trúc Swin-Tiny (Stage 20). |
| **Table 6a** | [`table06a_ablation_training_paradigm.tex`](table06a_ablation_training_paradigm.tex) | [`table06a_ablation_training_paradigm.md`](table06a_ablation_training_paradigm.md) | **[Ablation 1]** Khảo sát Chiến lược & Quy trình Huấn luyện (1-Stage vs 2-Stage vs 3S-HFT). |
| **Table 6b** | [`table06b_ablation_hierarchy_schedule.tex`](table06b_ablation_hierarchy_schedule.tex) | [`table06b_ablation_hierarchy_schedule.md`](table06b_ablation_hierarchy_schedule.md) | **[Ablation 2]** Khảo sát Lịch trình Trọng số Phân cấp (Curriculum Warmup vs Fixed). |
| **Table 6c** | [`table06c_ablation_loss_components.tex`](table06c_ablation_loss_components.tex) | [`table06c_ablation_loss_components.md`](table06c_ablation_loss_components.md) | **[Ablation 3]** Bóc tách đóng góp cận biên của các thành phần hàm loss (SupCon, SBS, Hrc). |
| **Table 6d** | [`table06d_ablation_head_placement.tex`](table06d_ablation_head_placement.tex) | [`table06d_ablation_head_placement.md`](table06d_ablation_head_placement.md) | **[Ablation 4]** Khảo sát Vị trí Trích xuất Đặc trưng (Shared Late-Stage vs Intermediate Heads). |
| **Table 6e** | [`table06e_ablation_freezing_depth.tex`](table06e_ablation_freezing_depth.tex) | [`table06e_ablation_freezing_depth.md`](table06e_ablation_freezing_depth.md) | **[Ablation 5]** Khảo sát Độ sâu Đóng băng Backbone & Đánh đổi Chi phí Tính toán. |
| **Table 6f** | [`table06f_ablation_hyperparameters_blending.tex`](table06f_ablation_hyperparameters_blending.tex) | [`table06f_ablation_hyperparameters_blending.md`](table06f_ablation_hyperparameters_blending.md) | **[Ablation 6]** Khảo sát Độ nhạy Siêu tham số ($\tau, w_{\text{supcon}}$) & Hệ số Hòa trộn Suy luận ($\lambda$). |
| **Table 7** | [`table07_per_class_and_runtime.tex`](table07_per_class_and_runtime.tex) | [`table07_per_class_and_runtime.md`](table07_per_class_and_runtime.md) | Hiệu năng chi tiết theo từng lớp Coarse và Benchmark độ trễ suy luận thời gian thực (FPS). |
| **Tất cả** | [`all_paper_tables.tex`](all_paper_tables.tex) | — | Tệp tổng hợp toàn bộ các bảng LaTeX sẵn sàng biên dịch kèm gói phụ trợ. |
