# Table 2: Tổng Hợp & Đối Chiếu Các Công Trình Nghiên Cứu Phân Loại Nội Soi Bàng Quang Liên Quan

| Nghiên Cứu / Công Trình | Bộ Dữ Liệu / Cỡ Mẫu | Số Lượng Lớp / Tầng | Độc Lập Bệnh Nhân | Phương Pháp Cốt Lõi | Kết Quả Chính Báo Cáo | Điểm Hạn Chế / Khoảng Trống Khoa Học |
|---|---|:---:|:---:|---|---|---|
| **Lee et al. (CystoDS gốc) [1]** | 8,067 ảnh / 160 BN | Nhị phân ROI (2 lớp) | 1 Split riêng | Swin-Large / HRNet | Binary AUROC 0.96 | Chỉ làm bài toán nhị phân phẳng, bỏ qua hoàn toàn 5 nhóm coarse và 22 lớp fine. |
| **Shkolyar et al. [5]** | Video WLC nội bộ | ROI Detection | Có | CNN Object Detection | Sensitivity 90.9%, Specificity 98.6% | Chỉ phát hiện khối u (Detection), không phân loại mô bệnh học vi thể hay phân tầng giải phẫu. |
| **Wu et al. [6]** | 69,204 ảnh / 10,729 BN | Ung thư vs Lành tính | Đa trung tâm | CNN Classifier | Sensitivity 95.4%, Specificity 94.7% | Phân loại nhị phân quy mô lớn nhưng thiếu hoàn toàn cây phân cấp bệnh học và phân lớp hiếm. |
| **Lazo et al. [7]** | 1,754 ảnh / 23 BN | 4 lớp phẳng | Có | Semi-supervised CNN | Accuracy 86.4% | Cỡ mẫu bệnh nhân rất nhỏ (23 BN), cấu trúc lớp phẳng không phản ánh phả hệ y học. |
| **Abd El-Aziz et al. [9]** | Bộ dữ liệu EBTC | 4 lớp phẳng | Không rõ | EfficientNet-B3 | Accuracy 97.2% | Phân loại phẳng 4 lớp, không xử lý phân bố đuôi dài hay kiểm định patient-disjoint nghiêm ngặt. |
| **Wang et al. [10]** | NBI đa trung tâm | Phân độ ung thư | Có | Multitask NBI Net | Sensitivity 89.2% | Phụ thuộc hoàn toàn vào công nghệ ánh sáng dải hẹp NBI, khó phổ cập cho nội soi WLC thông thường. |
| **Nghiên cứu này (Proposed 3S-HFT)** | **8,067 ảnh / 160 BN** | **3 tầng (2 / 5 / 22 lớp)** | **3 Splits chuẩn hóa (100% Patient-Disjoint)** | **3S-HFT + Curriculum Warmup + Hierarchical Ens.** | **Binary AUROC 0.9986, Coarse Acc 86.42%, Fine F1 (Supp) 0.6450, C-F Cons. 89.52%** | **Giải quyết đồng thời phân cấp đa tầng, bảo vệ ranh giới quyết định và xử lý đuôi dài cực đoan.** |
