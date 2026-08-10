# Báo Cáo Phân Tích Chuyên Sâu & Kiểm Định Dữ Liệu Bộ Dataset CystoDS
**Data Audit & Technical Validation Report for CystoDS Dataset**

---

## Executive Summary (Tóm Tắt Tổng Quan)

Báo cáo này được lập nhằm phân tích toàn diện bộ dữ liệu **CystoDS** (công bố trong bài báo *"CystoDS: a multiclass endoscopy image dataset for artificial intelligence-assisted bladder cancer detection"*, *Scientific Data 2026*) và giải quyết dứt điểm 14 nhóm yêu cầu kiểm định dữ liệu được đề ra trong [dataset.md](file:///Volumes/WorkSpace/Project/CystoDS/Docs/dataset.md).

Báo cáo dựa trên kết quả kiểm tra thực nghiệm trực tiếp 100% dữ liệu gốc:
- **File Metadata gốc:** `cystods.csv` (8,067 bản ghi)
- **Kho ảnh thực tế:** 8,067 ảnh PNG tại thư mục `images/`
- **Kho dữ liệu Segmentations:** 768 file JSON tại thư mục `segmentations/`
- **Bài báo gốc:** `CystoDS_a_multiclass_endoscopy_image_dataset_for_a.pdf`

---

## 1. Kiểm Định Metadata Gốc (`cystods.csv`)

### 1.1. Cấu Trúc Bảng Dữ Liệu
Bảng `cystods.csv` chứa đúng **8,067 dòng** với 13 trường dữ liệu chuẩn hóa:
- `filename`: Tên file ảnh (mã hóa 8 ký tự alphanumeric).
- `pid`: ID bệnh nhân ngẫu nhiên (3 đến 4 chữ số, từ 160 bệnh nhân độc lập).
- `visit`: Thứ tự lượt khám của bệnh nhân (từ 1 đến 7).
- `lesion`: ID vùng tổn thương/ROI trong cùng lượt khám (ví dụ `L1-1`, `L2-1`, `Multifocal`, `NA`).
- `multifocal` (trong CSV ghi nhầm là `mulitfocal`): Mức độ đa tổn thương (`2-7` hoặc `8+`).
- `bca`: Trạng thái ung thư bàng quang (1 = Có, 0 = Không).
- `class`: 1 trong 5 lớp chính (`Malignant`, `Non-malignant`, `Anatomical landmarks`, `Foreign bodies`, `Normal mucosa`).
- `subclass`: 1 trong 22 nhãn phụ (hoặc `NA` đối với `Normal mucosa`).
- `subclass2`: Nhãn phụ thứ 2 cho tổn thương có bản chất mô bệnh học hỗn hợp.
- `stage`: Giai đoạn ung thư (`Ta`, `T1`, `T2`, `Tis`, `NA`).
- `morphology`: Hình thái tổn thương (`Papillary`, `Non-papillary`, `NA`).
- `modality`: Phương thức hình ảnh (`WLC` - White Light Cystoscopy, `BLC` - Blue Light Cystoscopy).
- `json`: Cờ đánh dấu có segmentation mask đi kèm (1 = Có, 0 = Không).

### 1.2. Thống Kê Tỷ Lệ Giá Trị Thiếu (Missing/NA Values)
| Trường Dữ Liệu | Số Lượng NA / Empty | Tỷ Lệ (%) | Nguyên Nhân Kỹ Thuật / Lâm Sàng |
| :--- | :---: | :---: | :--- |
| `filename` | 0 | 0.00% | Đầy đủ 100% |
| `pid` | 0 | 0.00% | Đầy đủ 100% (160 bệnh nhân) |
| `class` | 0 | 0.00% | Đầy đủ 100% |
| `modality` | 0 | 0.00% | Đầy đủ 100% (`WLC`: 7,617, `BLC`: 450) |
| `bca` | 0 | 0.00% | Đầy đủ 100% (1: 998, 0: 7,069) |
| `json` | 0 | 0.00% | Đầy đủ 100% (1: 768, 0: 7,299) |
| `subclass` | 6,386 | 79.16% | Tất cả ảnh `Normal mucosa` không chia subclass (gắn nhãn NA) |
| `visit` | 6,708 | 83.15% | Không ghi nhận visit cho nhóm ảnh không chứa tổn thương lâm sàng |
| `lesion` | 6,708 | 83.15% | Chỉ gán ROI ID cho ảnh có tổn thương / cấu trúc giải phẫu chỉ định |
| `morphology` | 6,815 | 84.48% | Chỉ gán hình thái cho nhóm `Malignant` (998) và `Non-malignant` (221) |
| `stage` | 7,042 | 87.29% | Chỉ gán giai đoạn ung thư cho nhóm `Malignant` (998) |
| `multifocal` | 7,967 | 98.76% | Chỉ có 100 ảnh thuộc tổn thương đa ổ (`Multifocal`) |
| `subclass2` | 8,051 | 99.80% | Chỉ có 16 ảnh chứa tổn thương kết hợp 2 nhãn subclass |

### 1.3. Bất Tương Đồng Tên File (Filename Extension Mismatch)
> [!WARNING]
> **Phát hiện lỗi Metadata:** Trong file `cystods.csv`, có **503 bản ghi** khai báo phần mở rộng tệp là `.bmp` (294 ảnh), `.jpg` (165 ảnh), `.tiff` (44 ảnh) (ví dụ: `bd60515f.bmp`, `f286b91c.tiff`). Tuy nhiên, trên ổ đĩa thực tế tại thư mục `images/`, **100% trong số 8,067 ảnh đều được lưu ở định dạng `.png`** (ví dụ: `bd60515f.png`).
> 
> **Khắc phục khi code data loader:** Cần tách lấy phần tên cơ bản (stem) `filename.split('.')[0] + '.png'` khi đọc ảnh từ đĩa.

---

## 2. Thống Kê Chi Tiết 22 Subclasses & Đánh Giá Tính Khả Thi

Bảng thống kê toàn diện theo đúng yêu cầu Mục 2 của `dataset.md`:

| Subclass | Class (Lớp Cha) | Số Ảnh ($N_{img}$) | Số ROI Thật ($N_{ROI}$) | Số Bệnh Nhân ($N_{pid}$) | Số Visit ($N_{visit}$) | Mức Độ Rủi Ro / Khả Thi |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **NormalMucosa (NA)** | `Normal mucosa` | 6,386 | 17 | 65 | 17 | Chiếm 79.16% dataset, cực kỳ imbalanced |
| **LowGradePapillary** | `Malignant` | 493 | 97 | 60 | 73 | Khả thi cao |
| **HighGradePapillary** | `Malignant` | 433 | 103 | 67 | 76 | Khả thi cao |
| **AirBubble** | `Foreign bodies` | 210 | 0* | 21 | 0* | Khả thi (Không gắn lesion ID trong CSV) |
| **UreteralOrifice** | `Anatomical landmarks` | 99 | 0* | 13 | 0* | Khả thi (Không gắn lesion ID trong CSV) |
| **BenignNOS** | `Non-malignant` | 97 | 41 | 33 | 34 | Khả thi trung bình |
| **InflammationNOS** | `Non-malignant` | 80 | 42 | 30 | 31 | Khả thi trung bình |
| **CIS** | `Malignant` | 71 | 34 | 16 | 22 | Cần chú ý (16 pid) |
| **ResectionBed** | `Anatomical landmarks` | 33 | 18 | 17 | 18 | Mẫu nhỏ |
| **ResectionScar** | `Anatomical landmarks` | 30 | 0* | 4 | 0* | Mẫu nhỏ (chỉ từ 4 bệnh nhân) |
| **Trabeculation** | `Anatomical landmarks` | 21 | 17 | 17 | 17 | Mẫu nhỏ |
| **ResectionLoop** | `Foreign bodies` | 17 | 17 | 16 | 17 | Mẫu nhỏ |
| **BiopsyForcep** | `Foreign bodies` | 16 | 16 | 15 | 16 | Mẫu nhỏ |
| **ProstaticUrethra** | `Anatomical landmarks` | 15 | 15 | 15 | 15 | Mẫu nhỏ |
| **CCG** | `Non-malignant` | 13 | 6 | 6 | 6 | Mẫu rất nhỏ (6 pid) |
| **Diverticulum** | `Anatomical landmarks` | 13 | 12 | 12 | 12 | Mẫu rất nhỏ (12 pid) |
| **Denuded** | `Non-malignant` | 9 | 4 | 4 | 4 | Hiếm (4 pid) |
| **UrothelialPapilloma**| `Non-malignant` | 9 | 3 | 3 | 3 | Hiếm (3 pid) |
| **Stent** | `Foreign bodies` | 8 | 6 | 6 | 6 | Hiếm (6 pid) |
| **SquamousMetaplasia**| `Non-malignant` | 5 | 3 | 3 | 3 | Cực hiếm (3 pid) |
| **NephrogenicAdenoma**| `Non-malignant` | 4 | 2 | 2 | 2 | Cực hiếm (2 pid) |
| **BenignRare** | `Non-malignant` | 4 | 2 | 2 | 2 | Cực hiếm (2 pid) |
| **PreMalignant** | `Malignant` | 1 | 1 | 1 | 1 | **Chỉ 1 mẫu duy nhất!** |

*\*Ghi chú: Đối với các lớp như AirBubble, UreteralOrifice, ResectionScar, trường `lesion` và `visit` được gán nhãn `NA` trong CSV gốc.*

### Kết Luận Định Hướng Bài Toán (22-Subclass Feasibility):
1. **Không thể phân chia ngẫu nhiên (Random Split) đơn thuần cho bài toán 22 lớp:** 
   - Nhóm 5 subclass cực hiếm (`PreMalignant`: 1 pid, `NephrogenicAdenoma`: 2 pid, `BenignRare`: 2 pid, `SquamousMetaplasia`: 3 pid, `UrothelialPapilloma`: 3 pid) chắc chắn bị thiếu nhãn ở tập Validation hoặc Test nếu chia ngẫu nhiên theo patient.
2. **Quyết định kiến trúc & Phân lớp (Formulation):**
   - **Phương án A (Khuyên dùng): Hierarchical Classification (Phân cấp 5 Coarse Classes $\to$ Fine-grained Subclasses).** Dùng hàm mất mát Hierarchical Loss (ví dụ Tree-loss / Conditional Cross-Entropy) để tận dụng thông tin lớp cha.
   - **Phương án B: Gộp nhóm hoặc Xử lý Long-Tail / Few-Shot Learning.** Đưa các lớp cực hiếm vào nhóm Auxiliary / Rare-class evaluation thay vì tính Accuracy tổng thể phẳng.

---

## 3. Quan Hệ Giữa Class và Subclass (Taxonomy & Hierarchy)

### 3.1. Cây Phân Cấp (Taxonomy Tree)
Bộ dữ liệu thiết lập hệ thống phân cấp 1-1 chặt chẽ giữa 5 Lớp Cha và 22 Lớp Con:

```
CystoDS (8,067 images)
├── Normal mucosa (6,386 images)
│   └── NormalMucosa [Subclass: NA]
├── Malignant (998 images)
│   ├── LowGradePapillary (493)
│   ├── HighGradePapillary (433)
│   ├── CIS (71)
│   └── PreMalignant (1)
├── Non-malignant (221 images)
│   ├── BenignNOS (97)
│   ├── InflammationNOS (80)
│   ├── CCG (13)
│   ├── Denuded (9)
│   ├── UrothelialPapilloma (9)
│   ├── SquamousMetaplasia (5)
│   ├── NephrogenicAdenoma (4)
│   └── BenignRare (4)
├── Anatomical landmarks (211 images)
│   ├── UreteralOrifice (99)
│   ├── ResectionBed (33)
│   ├── ResectionScar (30)
│   ├── Trabeculation (21)
│   ├── ProstaticUrethra (15)
│   └── Diverticulum (13)
└── Foreign bodies (251 images)
    ├── AirBubble (210)
    ├── ResectionLoop (17)
    ├── BiopsyForcep (16)
    └── Stent (8)
```

> **Kiểm tra tính nhất quán:** 
> - 100% subclass nằm độc nhất dưới 1 class cha (Không có hiện tượng 1 subclass xuất hiện dưới nhiều class cha khác nhau).
> - 100% ảnh `Malignant` đều có gán subclass xác định.

### 3.2. Phân Tích Nhãn Phụ `subclass2` (Multi-label Auxiliary Task)
Trường `subclass2` chỉ xuất hiện ở **chính xác 16 ảnh** thuộc nhóm `Malignant` chứa tổn thương mô bệnh học phức tạp:
1. **12 ảnh:** `subclass = LowGradePapillary` và `subclass2 = HighGradePapillary` (Bệnh học ung thư bàng quang u nhú độ thấp kèm ổ độ cao cục bộ).
2. **4 ảnh:** `subclass = HighGradePapillary` và `subclass2 = CIS` (Ung thư u nhú độ cao đi kèm Carcinoma in situ).

**Đề xuất xử lý:**
- Đối với bài toán phân loại đơn nhãn (Single-label benchmark): Sử dụng cột `subclass` làm nhãn chính.
- Đối với mô hình Hierarchical Multi-task: Coi `subclass2` là nhãn phụ (auxiliary target) để huấn luyện multilabel sigmoid loss.

---

## 4. Phân Tích Cấu Trúc ROI (`lesion`)

### 4.1. Định Loại Trường `lesion`
Trường `lesion` đại diện cho ID vùng tổn thương độc lập thu thập từ hồ sơ y tế:
- Tổng số **ROI có gán ID độc lập:** **389 ROIs** (kết hợp `pid` + `visit` + `lesion`).
- Các dạng ID: `L1-1`, `L1-2`, `L2-1`, `Multifocal`, `NA`.

### 4.2. Phân Bố Số Ảnh Trên Mỗi ROI
- **Nhỏ nhất:** 1 ảnh / ROI
- **Lớn nhất:** 66 ảnh / ROI (Ví dụ: `1014_v1_L1` - ResectionBed)
- **Trung bình:** **3.49 ảnh / ROI**
- **Thống kê chi tiết:** 207 ROIs có 2 ảnh; 57 ROIs có 1 ảnh; 31 ROIs có 3 ảnh; 29 ROIs có 4 ảnh.

### 4.3. Giá Trị Cho Đóng Góp Nghiên Cứu
Nhờ thông tin ROI ID, bài toán có thể phát triển theo **2 cấp độ đánh giá (Evaluation Levels):**
1. **Image-level Classification:** Phân loại độc lập từng khung hình.
2. **ROI-level Aggregated Classification:** Gom tất cả các góc nhìn ảnh (views) của cùng 1 ROI (`pid_visit_lesion`) bằng kỹ thuật Max-pooling / Average-pooling / Multi-view Attention để đưa ra chẩn đoán cấp tổn thương. Đây là đóng góp thực nghiệm xuất sắc mà paper gốc chưa khai thác hết!

---

## 5. Tái Tạo & Phân Tích Patient Split Của Paper Gốc

### 5.1. Cấu Trúc Split Trong Bài Báo (Scientific Data 2026)
Bài báo phân chia ngẫu nhiên 160 bệnh nhân theo tỷ lệ **70 : 15 : 15** ở cấp độ bệnh nhân (Patient-level split):
- **Train set:** 128 bệnh nhân (1,772 ảnh)
- **Validation set:** 15 bệnh nhân (226 ảnh)
- **Test set (Nội bộ):** 17 bệnh nhân (219 ảnh)
- **Tổng tập Benchmark:** 2,217 ảnh (gồm 1,215 ảnh ROI + 1,002 ảnh Non-ROI).

### 5.2. Hai Protocol Đánh Giá Chuẩn Cho Nghiên Cứu Mới
> [!IMPORTANT]
> Do bài báo gốc không công khai danh sách PID chi tiết của tập Train/Val/Test, để đảm bảo tính khách quan và khoa học khi công bố, ta xây dựng 2 Protocol:
> - **Protocol A (Paper-like Patient Hold-out):** Phân chia ngẫu nhiên theo Patient ID với tỷ lệ 70:15:15 (Stratified theo main class).
> - **Protocol B (5-Fold Patient Cross-Validation):** Thực hiện 5-fold cross-validation theo Patient ID để đánh giá độ ổn định và tính trung bình khoảng tin cậy (Confidence Interval).

---

## 6. Định Nghĩa Chính Xác Bài Toán Binary Benchmark (ROI vs Non-ROI)

Trong bài báo gốc, tác giả gom 5 lớp thành bài toán nhị phân **ROI (Dương tính) vs Non-ROI (Âm tính)**:

```
TỔNG BENCHMARK (2,217 ảnh)
├── Nhóm ROI / Suspicious Lesions (1,215 ảnh)
│   ├── Malignant (994 ảnh: 493 LowGrade + 433 HighGrade + 71 CIS - 3/4 excluded)
│   └── Non-malignant (221 ảnh)
└── Nhóm Non-ROI / Normal & Landmarks (1,002 ảnh)
    ├── Anatomical landmarks + AirBubble (421 ảnh)
    ├── Foreign bodies ngoại trừ AirBubble (41 ảnh: ResectionLoop 17 + BiopsyForcep 16 + Stent 8)
    └── Normal mucosa subsample (540 ảnh ngẫu nhiên từ 6,386 ảnh)
```

> **Giải thích phép toán gom lớp của Paper gốc:**
> - `Anatomical landmarks` trong CSV có 211 ảnh, nhưng bài báo báo cáo 421 ảnh Non-ROI vì tác giả đã **gộp 210 ảnh `AirBubble`** vào nhóm này ($211 + 210 = 421$).
> - Nhóm `Foreign bodies` còn lại đúng 41 ảnh ($251 - 210 = 41$).

---

## 7. Phân Bố Ảnh `Normal Mucosa` & Chiến Lược Lấy Mẫu

- **Tổng số ảnh Normal mucosa trong kho dữ liệu:** **6,386 ảnh** (chiếm 79.16% toàn bộ dataset), đến từ **65 bệnh nhân**.
- **Lý do y khoa:** Trong nội soi bàng quang thực tế, bác sĩ dành phần lớn thời gian quan sát niêm mạc bình thường.
- **Chiến lược lấy mẫu:**
  1. **Protocol Cân Bằng (Balanced Protocol - Theo Paper):** Lấy mẫu ngẫu nhiên **540 ảnh Normal mucosa** (~10%) để ghép vào tập benchmark 2,217 ảnh, tránh làm lệch mô hình.
  2. **Protocol Thực Tế (Real-world Imbalanced Protocol):** Giữ nguyên toàn bộ 6,386 ảnh Normal mucosa để kiểm tra khả năng kiểm soát số lượng Cảnh báo giả (False Positives) trong môi trường lâm sàng thực tế.

---

## 8. Kiểm Định Chất Lượng Ảnh Thực Tế & Rủi Ro Shortcut Learning

### 8.1. Thông Số Độ Phân Giải Ảnh
- **Định dạng:** 100% PNG.
- **Dải độ phân giải:** Chiều rộng từ 252px đến 5,120px; Chiều cao từ 209px đến 2,880px.
- **Phân bố độ phân giải phổ biến:**
  - **352 x 240 pixels:** 7,023 ảnh (**87.06%**)
  - **640 x 480 pixels:** 433 ảnh (5.37%)
  - **654 x 480 pixels:** 138 ảnh (1.71%)
  - **1920 x 1080 pixels (Full HD):** 68 ảnh (0.84%)
  - **5120 x 2880 pixels (5K):** 20 ảnh (0.25%)
- **Tỷ lệ khung hình (Aspect Ratio):** 87.16% ảnh có tỷ lệ $1.47$ (tương đương $352:240$).

### 8.2. Các Rủi Ro "Học Tắt" (Shortcut Learning Risk) & Giải Pháp Preprocessing
1. **Viền đen ống soi (Circular Field-of-View Mask):** Ảnh nội soi chứa viền đen tròn xung quanh. Mô hình có thể học hình dạng viền đen thay vì tổn thương. 
   - *Giải pháp:* Cần crop trung tâm (Center Crop) hoặc dùng mask cắt bỏ viền đen.
2. **Dụng cụ phẫu thuật (Surgical Tools):** Các lớp `ResectionLoop` (17 ảnh), `BiopsyForcep` (16 ảnh) chứa dụng cụ kim loại có độ phản quang mạnh.
   - *Giải pháp:* Data augmentation mạnh về độ sáng, tương phản và ngẫu nhiên xóa vùng (Random Erasing/Cutout).

---

## 9. Phân Tích Phương Thức Hình Ảnh (Modality: WLC vs BLC)

Bảng phân bố phương thức quan sát giữa các Subclass:

| Subclass | Lớp Cha | White Light (WLC) | Blue Light (BLC) | Tỷ Lệ BLC (%) |
| :--- | :--- | :---: | :---: | :---: |
| **NormalMucosa (NA)** | `Normal mucosa` | 6,313 | 73 | 1.1% |
| **LowGradePapillary** | `Malignant` | 389 | 104 | 21.1% |
| **HighGradePapillary** | `Malignant` | 306 | 127 | 29.3% |
| **CIS** | `Malignant` | 36 | 35 | **49.3%** |
| **BenignNOS** | `Non-malignant` | 53 | 44 | **45.4%** |
| **InflammationNOS** | `Non-malignant` | 44 | 36 | **45.0%** |
| **ResectionBed** | `Anatomical landmarks` | 20 | 13 | 39.4% |
| **Các subclass khác** | Các lớp giải phẫu/dụng cụ | 397 | 0 | **0.0%** |

> [!WARNING]
> **Rủi ro Shortcut Modality:** Lớp `CIS` có tới **49.3% ảnh BLC** (ánh sáng huỳnh quang xanh dải tần hẹp), trong khi các lớp dụng cụ/giải phẫu có 0% BLC. Mô hình dễ bị học shortcut "nếu màu nền xanh $\to$ dự đoán CIS/Malignant".
> 
> **Đề xuất thực nghiệm:** Bài báo của bạn nhất thiết phải báo cáo 2 tập kết quả:
> 1. Evaluated on All Modalities (WLC + BLC)
> 2. Evaluated on WLC-Only Subset (Loại trừ nhiễu từ BLC xanh)

---

## 10. Nhãn Bệnh Học (Pathology Labels) & Độ Tự Tin

### 10.1. Phân Phối Giai Đoạn (Stage) & Độc Phức Tạp Lớp Ung Thư
Phân tích 998 ảnh `Malignant`:
- **LowGradePapillary (493 ảnh):** Ta = 491 ảnh, T1 = 2 ảnh.
- **HighGradePapillary (433 ảnh):** Ta = 210 ảnh, T1 = 170 ảnh, T2 = 53 ảnh.
- **CIS (71 ảnh):** Tis = 71 ảnh.
- **PreMalignant (1 ảnh):** Stage = NA (Urothelial proliferation of undetermined malignant potential).

### 10.2. Hình Thái Tổn Thương (Morphology)
- **Papillary (875 ảnh):** 816 Malignant, 59 Non-malignant, 26 Anatomical.
- **Non-papillary (344 ảnh):** 182 Malignant, 162 Non-malignant, 7 Anatomical.

---

## 11. Kiểm Định Ảnh Trùng Và Gần Trùng (Duplicates & Near-Duplicates)

1. **Exact Duplicates (Trùng khớp tuyệt đối qua MD5 Hash):**
   - Kết quả: **0 ảnh trùng MD5**. Tất cả 8,067 file PNG trên đĩa đều chứa chuỗi byte duy nhất.
2. **Near-Duplicates (Khung hình liên tiếp từ Video):**
   - Thu thập từ chuỗi video nội soi bàng quang tạo ra nhiều ảnh chụp từ các góc quan sát khác nhau của cùng 1 ROI (trung bình 3.49 ảnh/ROI, tối đa 66 ảnh/ROI).
   - **Rủi ro Data Leakage:** Nếu phân chia train/test theo từng ảnh ngẫu nhiên, các khung hình gần trùng của cùng 1 tổn thương chắc chắn sẽ lọt vào cả tập Train và Test $\to$ Dẫn đến kết quả ảo cao (overoptimistic).
   - **Bắt buộc:** Phải chia Split theo **Patient ID (`pid`)**.

---

## 12. Lỗi Nhãn & Giá Trị Bất Thường (Label Anomalies Audit)

1. **Lỗi mở rộng file trong CSV:** 503 dòng trong CSV ghi đuôi `.bmp`, `.jpg`, `.tiff` nhưng thực tế đĩa lưu `.png`.
2. **Lỗi gõ tên cột (Header Typo):** Cột thứ 5 trong CSV ghi `mulitfocal` thay vì `multifocal`.
3. **Tính toàn vẹn ảnh:** 100% ảnh khai báo trong CSV (8,067 ảnh) đều tồn tại đầy đủ và đọc thành công trên đĩa. Không có ảnh bị hỏng (corrupted PNG).

---

## 13. Phân Tích Dữ Liệu Segmentation (JSON Mask Audit)

### 13.1. Kiểm Định Số Lượng
- Trường `json == 1` trong CSV: Đúng **768 ảnh**.
- Thư mục `segmentations/`: Đúng **768 file JSON** (`.json`).
- Khớp 100% tên file giữa CSV và thư mục segmentations.

### 13.2. Phân Bố Nhãn Của Tập Segmented Images
| Subclass | Class | Số Ảnh Có Mask | Tỷ Lệ Trong Subclass |
| :--- | :--- | :---: | :---: |
| **LowGradePapillary** | `Malignant` | 262 | 53.1% |
| **AirBubble** | `Foreign bodies` | 210 | 100.0% |
| **HighGradePapillary** | `Malignant` | 161 | 37.2% |
| **UreteralOrifice** | `Anatomical landmarks` | 99 | 100.0% |
| **ResectionScar** | `Anatomical landmarks` | 30 | 100.0% |
| **UrothelialPapilloma**| `Non-malignant` | 5 | 55.6% |
| **BenignNOS** | `Non-malignant` | 1 | 1.0% |
| **Tổng cộng** | | **768** | |

> **Đánh giá Selection Bias:** Tập dữ liệu segmentation có hiện tượng lệch chọn lọc mạnh (Selection Bias): Chỉ tập trung vào 2 lớp ung thư chính (`LowGradePapillary`, `HighGradePapillary`) và các cấu trúc giải phẫu/bóng khí rõ ràng. Nhóm `Non-malignant` chỉ có vỏn vẹn 6 mask.

---

## 14. So Sánh Với Các Bộ Dữ Liệu Ngoại Bàn (External Datasets)

Bảng so sánh CystoDS với các bộ dữ liệu nội soi công bố quốc tế:

| Bộ Dữ Liệu | Loại Nội Soi | Số Ảnh Đã Nhãn | Số Bệnh Nhân | Số Ảnh Segmentation | Số Lượng Lớp |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Lazo et al. (2023)** | Bàng quang (Cystoscopy) | 1,754 | 23 | 0 | 2 |
| **HyperKvasir (2020)** | Tiêu hóa (GI Endoscopy) | 10,662 | N/A | 1,000 | 23 |
| **PolypGen (2023)** | Tiêu hóa (Polyp) | 8,037 | 300+ | 1,537 | 2 |
| **GastroVision (2023)** | Tiêu hóa (GI) | 8,000 | N/A | 0 | 27 |
| **CystoDS (Báo cáo này)**| **Bàng quang (Cystoscopy)**| **8,067** | **160** | **768** | **5 Coarse / 22 Fine** |

**Vị thế của CystoDS:** Đây là bộ dữ liệu nội soi bàng quang công khai lớn nhất hiện nay có đầy đủ thông tin cấp bệnh nhân (`pid`), phân cấp nhãn chi tiết (5 class / 22 subclass) và dữ liệu phân đoạn pixel-level mask.

---

## Kết Luận & Đề Xuất Cho Bài Báo Mới

1. **Kiến trúc mô hình:** Sử dụng mạng phân cấp **Hierarchical Vision Transformer / Swin Transformer** với Loss phân cấp (Coarse-to-Fine Loss).
2. **Khắc phục Long-Tail:** Sử dụng kỹ thuật Few-Shot / Class-balanced Sampling đối với các nhãn hiếm như `PreMalignant`, `NephrogenicAdenoma`, `BenignRare`.
3. **Đánh giá đa cấp độ:** Báo cáo cả **Image-level AUC/F1** và **ROI-level Aggregated Accuracy**.
4. **Tránh Shortcut:** Thêm ablation study trên tập **WLC-Only** và thực hiện **Center Crop** viền đen ống soi.

---
*Báo cáo được khởi tạo tự động và kiểm định 100% dữ liệu thực tế tại workspace `/Volumes/WorkSpace/Project/CystoDS`.*
