# Báo Cáo Phân Tích & Kiểm Định Đồng Bộ Dữ Liệu Sau Khi Split (Dataset Split Audit & Consistency Report)
**Dự án:** CystoDS — Benchmark Protocol Verification (Stage 00)  
**Ngày kiểm định:** 02/08/2026  
**Thư mục kết quả kiểm tra:** `file:///Volumes/WorkSpace/Project/CystoDS/result/protocol_release/stage_00_prepare_protocol_research_20260802-205342`

> **Lưu ý phiên bản 03/08/2026:** các số liệu hold-out bên dưới là bằng chứng
> của Kaggle result đã tải về. Source mới vẫn giữ fixed hold-out 70/15/15,
> nhưng Stage 00 không còn sinh selection-CV/sealed protocol. Final 5-fold CV
> được sinh độc lập trong Stage 90 trên dataset đã audit.

---

## Executive Summary (Tóm Tắt Tổng Quan)

Báo cáo này đối soát chi tiết dữ liệu giữa **Kho dữ liệu gốc (Raw Dataset)** được kiểm định trong [CystoDS_Data_Audit_Report.md](file:///Volumes/WorkSpace/Project/CystoDS/Docs/CystoDS_Data_Audit_Report.md) & bài báo gốc *Scientific Data 2026* với **Kết quả Phân chia Protocol (Stage 00 Release)** tại `result/protocol_release`.

### Kết Luận Định Định Lượng Độc Lập:
1. **Tính Đồng Bộ Tuyệt Đối 100% (Zero Data Leakage & Zero Loss):**
   - **Tất cả 21 Subclass tổn thương/cấu trúc/dụng cụ (1,681 ảnh):** Được **giữ lại nguyên vẹn 100%** sau khi Split ($\text{Raw} = 1,681 \leftrightarrow \text{Split Sum} = 1,681$).
   - **Tất cả 768 file JSON Segmentation Mask:** Được **giữ lại nguyên vẹn 100%** ($\text{Raw} = 768 \leftrightarrow \text{Split Sum} = 768$).
   - **Tất cả 160 Bệnh nhân (PID):** Được phân chia hoàn toàn độc lập (Disjoint Patient Sets):
     $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
2. **Cơ Chế Downsampling Nhóm `Normal mucosa` Chuẩn Bài Báo:**
   - Số ảnh `Normal mucosa` gốc (6,386 ảnh) được lấy mẫu phân tầng theo bệnh nhân (Patient-Stratified Sampling) xuống **540 ảnh** để xây dựng tập **Benchmark 2,221 ảnh** (tương đương con số 2,217 ảnh của bài báo gốc).
3. **Bảo Vệ Các Lớp Cực Hiếm (Rare-Class Safeguard):**
   - Nhờ cờ cấu hình `force_fine_labels_with_fewer_than_n_patients_to_train: 3`, 100% mẫu của các lớp cực hiếm (`PreMalignant`, `NephrogenicAdenoma`, `BenignRare`) được đưa vào tập **Train**, đảm bảo mô hình có dữ liệu học tập mà không bị rỗng nhãn.

---

## 1. Đối Soát Tổng Quan Trước & Sau Khi Split

| Tiêu Chí So Sánh | Tập Gốc (Raw Audit) | Tập Split Protocol (Holdout Benchmark) | Tỷ Lệ / Trạng Thái Đồng Bộ |
| :--- | :---: | :---: | :--- |
| **Tổng số Bệnh nhân ($N_{pid}$)** | **160** | **160** (Train: 112, Val: 24, Test: 24) | **Khớp 100%** (Tỷ lệ 70% : 15% : 15%) |
| **Số Bệnh nhân Train** | N/A | **112** | **70% fixed hold-out** |
| **Số Bệnh nhân Validation / Test** | N/A | **24 / 24** | **15% / 15% fixed hold-out** |
| **Tổng số Ảnh ($N_{img}$)** | **8,067** | **2,221** (Train: 1,553, Val: 339, Test: 329) | Downsample `Normal mucosa` từ 6,386 $\to$ 540 |
| **Ảnh Tổn thương & Cấu trúc (Non-Normal)** | **1,681** | **1,681** (Train: 1,175, Val: 258, Test: 248) | **Khớp 100% (Giữ lại toàn bộ 1,681 ảnh)** |
| **Dữ liệu Segmentation Masks** | **768** | **768** (Train: 538, Val: 121, Test: 109) | **Khớp 100% (Giữ lại toàn bộ 768 masks)** |
| **Modality: White Light (WLC)** | **7,617** | **1,837** (Train: 1,290, Val: 282, Test: 265) | Khớp đúng số lượng WLC sau downsampling |
| **Modality: Blue Light (BLC)** | **450** | **384** (Train: 263, Val: 57, Test: 64) | **Khớp 100% 384 ảnh BLC tổn thương** (66 BLC Normal bị lọc theo Normal limit) |

---

## 2. Đối Soát Chi Tiết 22 Subclass (Fine-grained Distribution Audit)

Bảng đối soát từng Subclass giữa Raw dataset và tổng các tập Train / Val / Test:

| Subclass Name | Coarse Class | Raw Img Count | Train Img | Val Img | Test Img | Total Split | Status / Alignment |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **LowGradePapillary** | `Malignant` | 493 | 339 | 113 | 41 | **493** | **EXACT MATCH (100%)** |
| **HighGradePapillary** | `Malignant` | 433 | 296 | 42 | 95 | **433** | **EXACT MATCH (100%)** |
| **AirBubble** | `Foreign bodies` | 210 | 133 | 37 | 40 | **210** | **EXACT MATCH (100%)** |
| **UreteralOrifice** | `Anatomical landmarks` | 99 | 69 | 9 | 21 | **99** | **EXACT MATCH (100%)** |
| **BenignNOS** | `Non-malignant` | 97 | 69 | 18 | 10 | **97** | **EXACT MATCH (100%)** |
| **InflammationNOS** | `Non-malignant` | 80 | 56 | 8 | 16 | **80** | **EXACT MATCH (100%)** |
| **CIS** | `Malignant` | 71 | 63 | 2 | 6 | **71** | **EXACT MATCH (100%)** |
| **ResectionBed** | `Anatomical landmarks` | 33 | 22 | 8 | 3 | **33** | **EXACT MATCH (100%)** |
| **ResectionScar** | `Anatomical landmarks` | 30 | 30 | 0 | 0 | **30** | **EXACT MATCH (100%)** |
| **Trabeculation** | `Anatomical landmarks` | 21 | 14 | 3 | 4 | **21** | **EXACT MATCH (100%)** |
| **ResectionLoop** | `Foreign bodies` | 17 | 13 | 3 | 1 | **17** | **EXACT MATCH (100%)** |
| **BiopsyForcep** | `Foreign bodies` | 16 | 13 | 3 | 0 | **16** | **EXACT MATCH (100%)** |
| **ProstaticUrethra** | `Anatomical landmarks` | 15 | 12 | 1 | 2 | **15** | **EXACT MATCH (100%)** |
| **CCG** | `Non-malignant` | 13 | 9 | 2 | 2 | **13** | **EXACT MATCH (100%)** |
| **Diverticulum** | `Anatomical landmarks` | 13 | 7 | 5 | 1 | **13** | **EXACT MATCH (100%)** |
| **Denuded** | `Non-malignant` | 9 | 6 | 1 | 2 | **9** | **EXACT MATCH (100%)** |
| **UrothelialPapilloma**| `Non-malignant` | 9 | 7 | 0 | 2 | **9** | **EXACT MATCH (100%)** |
| **Stent** | `Foreign bodies` | 8 | 4 | 2 | 2 | **8** | **EXACT MATCH (100%)** |
| **SquamousMetaplasia**| `Non-malignant` | 5 | 4 | 1 | 0 | **5** | **EXACT MATCH (100%)** |
| **NephrogenicAdenoma**| `Non-malignant` | 4 | 4 | 0 | 0 | **4** | **EXACT MATCH (100%)** |
| **BenignRare** | `Non-malignant` | 4 | 4 | 0 | 0 | **4** | **EXACT MATCH (100%)** |
| **PreMalignant** | `Malignant` | 1 | 1 | 0 | 0 | **1** | **EXACT MATCH (100%)** |
| **NormalMucosa (NA)** | `Normal mucosa` | 6,386 | 378 | 81 | 81 | **540** | **SAMPLED BENCHMARK** |
| **TỔNG CỘNG** | | **8,067** | **1,553** | **339** | **329** | **2,221** | **ĐỒNG BỘ 100%** |

> [!NOTE]
> **Nhận xét chuyên môn:** 
> 100% trong số 21 subclass phi-Normal mucosa (tổng 1,681 ảnh) xuất hiện chính xác số lượng trước và sau khi split. Không có bất kỳ ảnh tổn thương nào bị bỏ sót hay rò rỉ trong quá trình xử lý protocol stage 00!

---

## 3. Contract Cross-Validation hiện hành

Stage 00 hiện chỉ tạo một fixed hold-out. Stage 90 là nơi duy nhất tạo final
5-fold patient-level cross-validation. Stage 90:

- dùng `split_seed` cố định, tách khỏi model-training seed;
- kiểm tra semantic dataset SHA và primary taxonomy từ Stage 00;
- tự tạo các fold patient-disjoint trên dataset benchmark;
- chạy đủ fold rồi ghi integrated report JSON/CSV/Markdown;
- không tiêu thụ checkpoint hoặc metrics của Stage 10--40.

Các selection folds từng có trong Kaggle result lịch sử không còn là input của
pipeline mới và không được dùng để tune/evaluate source hiện hành.

---

## 4. Kiểm Định Bất Tương Đồng Tên File & An Toàn Data Loader (Filename Integrity)

Trường dữ liệu `cystods.csv` chứa 503 đuôi tệp không thống nhất (`.bmp`, `.jpg`, `.tiff`), nhưng đĩa thực tế lưu 100% `.png`.
- **Kiểm định tệp trong Split Release:**
  Trong các tệp `train.csv`, `val.csv`, `test.csv` xuất ra tại `splits/holdout/`:
  - Cột `filename` giữ đúng tên gốc định danh.
  - Cột `image_path` / `stem` được chuẩn hóa tự động để đảm bảo Data Loader truy cập chính xác tệp `.png` trên đĩa.
  - 100% ảnh thuộc tập split đều giải mã ảnh (PNG image decode) thành công.

---

## 5. Kết Luận & Khuyến Nghị Trực Tiếp Cho Các Stage Tiếp Theo

1. **Tính Toàn Vẹn Của Pipeline:** Stage 00 đã tạo ra một giao thức split cực kỳ chuẩn xác, giải quyết triệt để rủi ro Data Leakage cấp độ Bệnh nhân và đảm bảo tính tái lập (Reproducibility) với Fingerprint SHA-256: `d5e476fb95b0e55d248da071d6c18ee0410bab64e656ff3256e2b2db03aff3f1`.
2. **Khai Thác Nhãn Hierarchical & Long-Tail:** Tất cả 22 lớp (bao gồm 5 lớp hiếm) đã nằm gọn gàng trong tập Train, sẵn sàng cho các Stage 10 đến Stage 40 huấn luyện mô hình phân cấp và giải quyết bài toán đuôi dài (Long-tail).
