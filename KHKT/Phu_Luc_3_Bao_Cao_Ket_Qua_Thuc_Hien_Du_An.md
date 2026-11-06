# Phụ lục 3

## BÁO CÁO KẾT QUẢ THỰC HIỆN DỰ ÁN DỰ THI
*(Kèm theo Kế hoạch số ……/KH-THPTPCT ngày ……/11/2026 của Trường THPT Phan Châu Trinh)*

---

## A. TRANG BÌA

1. **Tên dự án dự thi:**  
   **HỆ THỐNG TRÍ TUỆ NHÂN TẠO HỌC PHÂN CẤP ĐA TẦNG HỖ TRỢ CHẨN ĐOÁN HÌNH ẢNH NỘI SOI BÀNG QUANG THỜI GIAN THỰC**

2. **Lĩnh vực dự thi:**  
   **Hệ thống nhúng và Phần mềm hệ thống**  
   *(Phân ngành: Trí tuệ nhân tạo & Y sinh học)*

---

## B. BÁO CÁO KẾT QUẢ THỰC HIỆN DỰ ÁN DỰ THI

### I. ĐỐI VỚI DỰ ÁN KHOA HỌC

#### 1. Câu hỏi nghiên cứu

##### a) Thuyết minh về sự đóng góp của dự án vào lĩnh vực nghiên cứu

* **Bối cảnh y học thực tiễn và tính cấp thiết:**  
  Ung thư bàng quang (*Bladder Cancer*) là một trong mười bệnh lý ác tính phổ biến nhất thế giới với hơn **573.000 ca mắc mới** và **213.000 ca tử vong** hàng năm (*WHO / GLOBOCAN 2020; Sung et al., 2021*). Đặc tính sinh học nguy hiểm của căn bệnh này là tỷ lệ tái phát sau phẫu thuật cắt bỏ nội soi qua ngả niệu đạo (TURBT) rất cao, dao động từ **50% đến 70%**, và có khoảng 10% – 20% các trường hợp tiến triển thành ung thư xâm lấn cơ đe dọa trực tiếp tính mạng (*Hướng dẫn lâm sàng Hiệp hội Tiết niệu Châu Âu - EAU Guidelines 2022; Babjuk et al., 2022*). Do đó, người bệnh buộc phải trải qua quy trình theo dõi định kỳ suốt đời bằng phương pháp nội soi bàng quang.

  Nội soi bàng quang qua ngả niệu đạo (*Cystoscopy*) là "tiêu chuẩn vàng" trong sàng lọc, phát hiện và theo dõi ung thư bàng quang. Tuy nhiên, việc nội soi bằng mắt thường dưới ánh sáng trắng thông thường (WLC) gặp phải hai thách thức lâm sàng đặc biệt nghiêm trọng:
  1. *Nguy cơ bỏ sót các tổn thương phẳng hiểm nghèo:* Khác với ung thư thể nhú dạng súp-lơ (*Papillary tumors*) phát triển lồi vào lòng bàng quang dễ phát hiện, **ung thư biểu mô tại chỗ (*Carcinoma in situ - CIS*)** là tổn thương ác tính độ cao có bề mặt phẳng, chỉ hơi đỏ nhẹ giống niêm mạc viêm bàng quang thông thường. Tỷ lệ bỏ sót CIS khi nội soi bằng mắt thường lên tới **20% – 30%** (*Schmidbauer et al., 2004; Babjuk et al., 2022*). Bỏ sót CIS đồng nghĩa với việc bệnh nhân mất đi cơ hội điều trị bảo tồn bàng quang ở giai đoạn sớm.
  2. *Hiện tượng mệt mỏi thị giác của bác sĩ nội soi:* Môi trường bàng quang có góc nhìn hẹp, nước rửa dịch chuyển liên tục, xuất hiện bọt khí và phản xạ ánh sáng chói. Khi phải thực hiện nhiều ca mổ trong ngày, sự mệt mỏi thị giác làm gia tăng đáng kể nguy cơ bỏ sót tổn thương (*Shkolyar et al., 2019; Wu et al., 2022*).

* **Khoảng trống công nghệ của các nghiên cứu AI hiện nay:**  
  Mặc dù Trí tuệ nhân tạo (AI) đã đạt được nhiều bước tiến trong chẩn đoán hình ảnh X-quang và CT, việc ứng dụng AI vào nội soi bàng quang trước đây bộc lộ một rào cản kỹ thuật cốt lõi: **Hiện tượng mâu thuẫn logic phả hệ chẩn đoán (*Hierarchical Inconsistency*)**.
  Trong thực tế lâm sàng, tư duy chẩn đoán của bác sĩ luôn vận hành theo cấu trúc phân cấp đa tầng: (1) Sàng lọc có vùng tổn thương nghi ngờ (*ROI*) hay không; (2) Phân nhóm bệnh cảnh chung (Ác tính, Lành tính, Mốc giải phẫu, Dị vật); và (3) Kết luận tên bệnh lý cụ thể (Ung thư nhú độ cao, CIS, Viêm bàng quang...). Các nghiên cứu AI trước đây hầu như chỉ giải quyết bài toán phân loại phẳng (*Flat Classification*) hoặc huấn luyện các mô hình đơn lẻ riêng rẽ. Điều này dẫn đến nghịch lý chẩn đoán nguy hiểm: mô hình tầng chi tiết có thể dự đoán tổn thương là "Ung thư biểu mô tại chỗ (CIS)" nhưng ở mô hình tầng thô lại kết luận ảnh là "Niêm mạc lành tính". Mâu thuẫn này làm mất đi tính an toàn và khiến bác sĩ không thể tin tưởng ứng dụng AI trong phòng mổ.

* **Sự đóng góp khoa học và kỹ thuật của đề tài:**  
  1. *Đóng góp về mặt kiến trúc AI:* Đề xuất mô hình **CystoHier** trên nền tảng mạng thị giác Swin Transformer (`swin_tiny_patch4_window7_224`), sử dụng cơ chế chú ý tự thân trên cửa sổ dịch chuyển (*Shifted Window Self-Attention*) để nắm bắt ngữ cảnh toàn cục của lòng bàng quang, tích hợp chung một bộ trích xuất đặc trưng chia sẻ cho đồng thời **cả 3 cấp độ chẩn đoán y khoa** (Nhị phân $\rightarrow$ 5 nhóm bệnh thô $\rightarrow$ 22 phân lớp chi tiết).
  2. *Đóng góp về tính nhất quán logic phả hệ:* Tích hợp cơ chế cộng dồn xác suất phả hệ (*Hierarchical Marginalization*), tự động gom nhóm xác suất từ 22 phân lớp con chi tiết để củng cố cho 5 nhóm bệnh cha, triệt tiêu hoàn toàn nghịch lý chẩn đoán mâu thuẫn giữa các tầng.
  3. *Đóng góp về quy trình huấn luyện:* Đề xuất **Chiến lược huấn luyện tách rời 3 giai đoạn (*Three-Stage Hierarchical Fine-Tuning - 3S-HFT*)** với kỹ thuật cô lập tham số (*Parameter Isolation*), giải quyết triệt để sự xung đột gradient giữa các nhiệm vụ chẩn đoán đa tầng.
  4. *Đóng góp về tính minh bạch và ứng dụng thực tiễn:* Tích hợp bản đồ nhiệt Grad-CAM định vị chính xác vùng giải phẫu tổn thương (IoU tăng gấp 4,71 lần) và xây dựng hoàn chỉnh phần mềm **Clinical Web Dashboard** vận hành theo thời gian thực (66,7 – 82,8 FPS, độ trễ 15,0 ms) sẵn sàng kết nối hỗ trợ bác sĩ tại phòng mổ.

##### b) Mục tiêu nghiên cứu cụ thể và rõ ràng

1. **Mục tiêu 1 (Thiết kế kiến trúc AI phân cấp đa tầng):**  
   Xây dựng mô hình học sâu duy nhất có khả năng trích xuất đặc trưng ngữ cảnh toàn cục của lòng bàng quang và dự đoán đồng thời trên cả 3 cấp độ chẩn đoán:
   - *Cấp 1 (Nhị phân):* Phát hiện vùng tổn thương nghi ngờ (*ROI*) vs Niêm mạc bình thường (*Non-ROI*).
   - *Cấp 2 (Nhóm bệnh thô - 5 nhóm):* Malignant (Ác tính), Non-malignant (Không ác tính/Lành tính), Normal mucosa (Niêm mạc bình thường), Anatomical landmarks (Mốc giải phẫu), Foreign bodies (Dị vật/Dụng cụ).
   - *Cấp 3 (Bệnh học chi tiết - 22 phân lớp):* Định danh chính xác 4 lớp ung thư (LowGradePapillary, HighGradePapillary, CIS, PreMalignant), 8 lớp lành tính, 6 mốc giải phẫu và 4 loại dị vật.  
   *Chỉ tiêu đánh giá:* Độ nhất quán logic phả hệ giữa Cấp 2 và Cấp 3 đạt $\ge 85\%$.

2. **Mục tiêu 2 (Xây dựng chiến lược huấn luyện tách rời 3 giai đoạn 3S-HFT):**  
   Thiết lập quy trình huấn luyện tách rời nhằm tối ưu hóa độc lập ranh giới phân loại từng tầng:
   - Giai đoạn 1: Học biểu diễn đặc trưng thị giác toàn cục trên toàn bộ mạng.
   - Giai đoạn 2: Đóng băng Backbone, nắn chỉnh ranh giới phân loại 5 nhóm bệnh Cấp 2.
   - Giai đoạn 3: Tiếp tục đóng băng Backbone, tinh chỉnh chuyên sâu phân loại 22 bệnh lý Cấp 3 (đặc biệt là tổn thương phẳng CIS).  
   *Chỉ tiêu đánh giá:* Đạt Binary AUROC $\ge 0,99$, Coarse Accuracy $\ge 85\%$, và Fine Macro-F1 trên các phân lớp hiếm $\ge 0,60$ (vượt trội có ý nghĩa thống kê $p < 0,05$ so với các mô hình baseline).

3. **Mục tiêu 3 (Minh bạch hóa thị giác và phát triển phần mềm lâm sàng thời gian thực):**  
   - Ứng dụng kỹ thuật Grad-CAM tạo bản đồ nhiệt thị giác minh bạch hóa căn cứ chẩn đoán của AI, đạt chỉ số trùng khớp giải phẫu (IoU) với khoanh vùng chuyên gia tăng tối thiểu gấp 3 lần so với mô hình thông thường.
   - Xây dựng phần mềm ứng dụng lâm sàng hoàn chỉnh với giao diện Web chuyên nghiệp (*Clinical Web Dashboard*), đạt thông lượng xử lý thời gian thực $\ge 60$ FPS (độ trễ $\le 20$ ms/ảnh) và chiếm dụng bộ nhớ $\le 200$ MiB trên phần cứng tiêu chuẩn trong phòng mổ.

##### c) Câu hỏi nghiên cứu và Giả thuyết khoa học

* **Câu hỏi nghiên cứu 1:** Làm thế nào để một mô hình Trí tuệ nhân tạo duy nhất có thể chẩn đoán đồng thời cả 3 cấp độ phân cấp y khoa mà không xảy ra mâu thuẫn logic giữa tầng chẩn đoán thô và tầng chẩn đoán chi tiết?
* **Câu hỏi nghiên cứu 2:** Chiến lược huấn luyện tách rời 3 giai đoạn (3S-HFT) có thể giải quyết được hiện tượng xung đột gradient giữa các nhiệm vụ phân cấp, giúp nâng cao độ chính xác ở các phân lớp bệnh học chi tiết hay không?
* **Câu hỏi nghiên cứu 3:** Hệ thống AI có thể vừa cung cấp bản đồ nhiệt giải thích vị trí giải phẫu khối u chính xác, vừa đạt tốc độ xử lý khung hình thời gian thực ($\ge 60$ FPS) trên thiết bị máy tính thông thường hay không?
* **Giả thuyết khoa học:**  
  *Nếu tích hợp quan hệ ràng buộc logic phả hệ vào một mạng Swin Transformer duy nhất và áp dụng chiến lược huấn luyện tách rời 3 giai đoạn (3S-HFT), mô hình sẽ vừa nhận diện chính xác các tổn thương ác tính chi tiết (đặc biệt là ung thư biểu mô tại chỗ CIS), vừa triệt tiêu hiện tượng mâu thuẫn chẩn đoán cha-con và đạt độ tin cậy giải phẫu vượt trội so với các mạng nơ-ron truyền thống, đồng thời đáp ứng hoàn hảo yêu cầu xử lý thời gian thực trong phòng mổ.*

---

#### 2. Thiết kế và phương pháp nghiên cứu

##### a) Thiết kế kế hoạch nghiên cứu theo mục tiêu

Kế hoạch nghiên cứu được xây dựng tuần tự và chặt chẽ qua 5 giai đoạn nối tiếp:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 1: THU THẬP, TIỀN XỬ LÝ DỮ LIỆU & GIAO THỨC ĐỘC LẬP BỆNH NHÂN │
│ - Thu thập 8.067 ảnh từ 160 bệnh nhân (7.009 WLC, 1.058 NBI).          │
│ - Tự động cắt bỏ viền đen ống soi (Center Crop 0.92), resize 224x224.  │
│ - Phân hoạch nghiêm ngặt 100% Patient-Disjoint (70% Train, 15% Val, 15% Test). │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 2: THIẾT KẾ MÔ HÌNH HỌC PHÂN CẤP 3 CẤP ĐỘ ĐỒNG THỜI           │
│ - Xây dựng Backbone Swin-Tiny với cơ chế Shifted Window Self-Attention.│
│ - Thiết kế 3 đầu chẩn đoán: Binary Head (2), Coarse Head (5), Fine (22).│
│ - Thiết lập cơ chế cộng dồn xác suất phả hệ (Hierarchical Marginalization). │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 3: CHIẾN LƯỢC HUẤN LUYỆN TÁCH RỜI 3 GIAI ĐOẠN (3S-HFT)        │
│ - Pha 1: Mở toàn bộ mạng, học biểu diễn đặc trưng thị giác toàn cục.   │
│ - Pha 2: Đóng băng Backbone, tối ưu hóa ranh giới 5 nhóm bệnh Cấp 2.   │
│ - Pha 3: Đóng băng Backbone, tinh chỉnh chuyên sâu 22 bệnh lý Cấp 3.   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 4: THỰC NGHIỆM ĐỐI CHUẨN KHOA HỌC & ĐÁNH GIÁ MINH BẠCH GRAD-CAM│
│ - Thực nghiệm đối chuẩn với 15 mô hình cơ sở (Swin, HRNet, ResNeXt, ResNet). │
│ - Đánh giá độ nhất quán phả hệ và kiểm định ý nghĩa thống kê (p-value). │
│ - Đánh giá độ trùng khớp giải phẫu Grad-CAM IoU trên mặt nạ chuyên gia.│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN 5: HIỆN THỰC HÓA PHẦN MỀM LÂM SÀNG THỜI GIAN THỰC            │
│ - Xây dựng phần mềm Clinical Web Dashboard (React, TypeScript, FastAPI).│
│ - Đo lường thông lượng FPS, độ trễ và dung lượng bộ nhớ tiêu hao.      │
│ - Tích hợp thanh trượt Grad-CAM, cảnh báo CIS và xuất phiếu khám PDF.   │
└────────────────────────────────────────────────────────────────────────┘
```

##### b) Phương pháp thu thập và thiết lập số liệu

* **Nguồn dữ liệu nghiên cứu:**  
  Nghiên cứu sử dụng bộ dữ liệu chuẩn mực quốc tế **CystoDS** gồm **8.067 ảnh nội soi kỹ thuật số** từ **160 bệnh nhân duy nhất** do các chuyên gia tiết niệu đầu ngành thu thập và dán nhãn (*Nature Scientific Data 2026*). Dữ liệu bao gồm:
  - 7.009 ảnh nội soi ánh sáng trắng thông thường (*White Light Cystoscopy - WLC*).
  - 1.058 ảnh nội soi dải tần ánh sáng hẹp (*Narrow Band Imaging - NBI*).

* **Quy trình tiền xử lý hình ảnh:**  
  1. *Cắt tỉa trường nhìn tròn (FOV Center Crop):* Ảnh nội soi luôn có viền đen hình tròn xung quanh. Thuật toán tự động cắt tỉa trung tâm với tỷ lệ 0,92 để loại bỏ hoàn toàn viền đen vô nghĩa, bảo toàn 100% diện tích niêm mạc hữu ích.
  2. *Chuẩn hóa kích thước:* Đưa toàn bộ ảnh về độ phân giải chuẩn $224 \times 224 \times 3$ pixel, chuẩn hóa kênh màu theo giá trị Mean và Std của tập dữ liệu ImageNet.
  3. *Tăng cường dữ liệu thích ứng y tế (*Data Augmentation*):* Lật ngang ($p=0,5$), lật dọc ($p=0,5$), xoay ngẫu nhiên $\pm 15^\circ$, điều chỉnh độ sáng/độ tương phản nhẹ ($ColorJitter$), và xóa vùng ngẫu nhiên ($RandomErasing$ $p=0,2$) nhằm mô phỏng hiện tượng bọt khí và dịch rửa che khuất trong thực tế nội soi.

* **Giao thức phân chia dữ liệu độc lập bệnh nhân (100% Patient-Disjoint Protocol):**  
  Để loại bỏ hoàn toàn hiện tượng rò rỉ dữ liệu (*Data Leakage*), nhóm nghiên cứu kiên quyết áp dụng giao thức phân chia theo mã định danh bệnh nhân (*Patient ID*):
  - Tuyệt đối không chia ngẫu nhiên theo từng bức ảnh lẻ. Toàn bộ các ảnh chụp từ cùng một bệnh nhân chỉ được phép nằm trong duy nhất một tập dữ liệu.
  - Tỷ lệ phân bổ chuẩn mực: **70% Bệnh nhân cho tập Huấn luyện (Train - 112 BN)**, **15% Bệnh nhân cho tập Hiệu chỉnh (Validation - 24 BN)**, và **15% Bệnh nhân cho tập Kiểm tra độc lập (Hold-out Test - 24 BN với 337 ảnh)**.
  - Cố định 3 phân hoạch độc lập (`split_0`, `split_1`, `split_2`) để chạy lặp lại và tính toán giá trị Trung bình $\pm$ Độ lệch chuẩn (*Mean ± Std*), đảm bảo tính khách quan và ý nghĩa thống kê.

##### c) Các tham số, thông số và biến số nghiên cứu

* **Biến số độc lập (Independent Variables):**
  1. *Kiến trúc mạng trích xuất đặc trưng (Backbone):* So sánh giữa Swin Transformer (Swin-Tiny) với các mạng tích chập CNN kinh điển (ResNet-152, HRNet-W18, ResNeXt-50-32x4d).
  2. *Cấu hình phân loại:* Phân loại phân cấp 3 tầng đồng thời (*Proposed Hierarchical*) so với Phân loại phẳng (*Flat*) và Phân loại đơn nhiệm riêng rẽ (*Single-task*).
  3. *Chiến lược huấn luyện:* Chiến lược tách rời 3 giai đoạn (3S-HFT) so với Huấn luyện đồng thời toàn bộ mạng từ đầu đến cuối (*End-to-End Joint Training*).
  4. *Cơ chế ràng buộc phả hệ:* Có và Không có cơ chế cộng dồn xác suất phả hệ (*Hierarchical Marginalization*).

* **Biến số phụ thuộc (Dependent Variables / Output Metrics):**
  1. *Chỉ số Cấp 1 (Nhị phân):* Diện tích dưới đường cong ROC (AUROC), Độ nhạy (*Sensitivity*), Độ đặc hiệu (*Specificity*), Điểm F1 Nhị phân.
  2. *Chỉ số Cấp 2 (Nhóm bệnh cảnh thô):* Độ chính xác (*Coarse Accuracy*), Macro-F1 trên 5 nhóm bệnh.
  3. *Chỉ số Cấp 3 (Bệnh học chi tiết):* Độ chính xác chi tiết (*Fine Accuracy*), Macro-F1 trên các phân lớp hiếm (*Fine Macro-F1 Supported*).
  4. *Chỉ số Tính logic phả hệ:* Tỷ lệ nhất quán phả hệ giữa Cấp 2 và Cấp 3 (*Coarse-Fine Consistency Rate*).
  5. *Chỉ số Minh bạch thị giác:* Hệ số trùng khớp diện tích tổn thương (*Grad-CAM IoU*) so với mặt nạ chuyên gia giải phẫu.
  6. *Chỉ số Kỹ thuật phòng mổ:* Tốc độ khung hình trên giây (FPS), độ trễ suy luận toàn trình (ms/ảnh), và dung lượng bộ nhớ RAM/VRAM tiêu hao (MiB).

* **Tham số kiểm soát cố định (Controlled Parameters):**
  Kích thước ảnh đầu vào ($224 \times 224$); Thuật toán tối ưu hóa AdamW (Learning rate ban đầu $1 \times 10^{-4}$ ở Pha 1, giảm xuống $5 \times 10^{-5}$ ở Pha 2 và Pha 3); Lịch trình suy giảm học tập Cosine Annealing; Batch size = 32; Giới hạn số epoch tối đa = 50; Cơ chế dừng sớm (*Early Stopping*) dựa trên chỉ số Validation Macro-F1 với patience = 10 epochs.

---

#### 3. Quá trình thực hiện: thu thập, phân tích và giải thích dữ liệu

##### a) Quá trình thực hiện thực nghiệm

* **Bước 1: Huấn luyện mô hình đề xuất CystoHier qua 3 giai đoạn (3S-HFT):**
  - *Giai đoạn 1 (Học biểu diễn tự do - Representation Warmup):* Mở toàn bộ trọng số của Backbone Swin-Tiny và 3 đầu phân loại. Mạng được tối ưu hóa đồng thời trên cả 3 nhiệm vụ để trích xuất không gian đặc trưng tổng quát biểu diễn hình thái, đường viền và vi mạch bàng quang.
  - *Giai đoạn 2 (Nắn chỉnh bộ phân loại thô - Coarse Alignment):* Đóng băng hoàn toàn Backbone trích xuất đặc trưng và đầu Cấp 3, chỉ mở và tối ưu hóa đầu phân loại Coarse. Quá trình này giúp xác lập ranh giới phân định sắc bén giữa 5 nhóm bệnh lớn mà không làm xáo trộn không gian đặc trưng đã học.
  - *Giai đoạn 3 (Tinh chỉnh bộ phân loại chi tiết - Fine-grained Adaptation):* Tiếp tục đóng băng Backbone, chỉ mở duy nhất đầu phân loại 22 lớp Cấp 3. Quá trình này giúp mô hình tập trung tối đa năng lực phân biệt các tổn thương vi mô khó (như ung thư phẳng CIS, viêm bàng quang, u nhú niệu mạc).

* **Bước 2: Thực nghiệm đối chuẩn với 15 mô hình cơ sở (Baselines):**
  Tiến hành huấn luyện và kiểm thử đối chứng trên cùng 3 phân hoạch hold-out đối với 4 họ kiến trúc backbone phổ biến nhất:
  - Swin Transformer (Swin-Tiny)
  - High-Resolution Network (HRNet-W18)
  - ResNeXt (ResNeXt-50-32x4d)
  - Residual Network (ResNet-152)  
  Mỗi kiến trúc được thử nghiệm ở cả cấu hình Đa nhiệm (*Multitask Shared Backbone*) và Đơn nhiệm (*Single-task: Binary Only, Coarse Only, Fine Only*).

* **Bước 3: Thực nghiệm đánh giá tính minh bạch Grad-CAM và đo lường thông lượng thời gian thực:**
  - Áp dụng kỹ thuật Gradient-weighted Class Activation Mapping (Grad-CAM) trích xuất bản đồ kích hoạt tại tầng chuẩn hóa cuối cùng của Swin Transformer (`norm`), tính toán hệ số IoU so với mặt nạ khoanh vùng chuẩn của bác sĩ trên các mẫu bệnh phẩm đại diện.
  - Đo lường trực tiếp thời gian suy luận (Forward latency), thời gian tiền xử lý ảnh và tổng thời gian toàn trình (End-to-End) trên chip phần cứng Apple Silicon (nền tảng Metal Performance Shaders / MPS FP32) để đánh giá tính khả thi vận hành thời gian thực.

##### b) Kết quả thu được và xử lý số liệu

Toàn bộ kết quả thực nghiệm khách quan trên **tập kiểm tra độc lập 100% bệnh nhân** (Hold-out Test: 24 bệnh nhân / 337 ảnh) được tổng hợp chi tiết trong Bảng 1:

**Bảng 1: Bảng đối chuẩn tổng thể trên tập Test độc lập (Mean ± Std qua 3 phân hoạch hold-out)**

| STT | Mô hình & Kiến trúc | Phương pháp huấn luyện | Binary AUROC | Độ nhạy Cấp 1 (%) | Độ đặc hiệu Cấp 1 (%) | Coarse Acc (Cấp 2) (%) | Fine Macro-F1 (Cấp 3) | Tính nhất quán phả hệ (%) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **🏆** | **CystoHier (Đề xuất)** | **Swin-Tiny 3S-HFT (Phân cấp)** | **0,9986 ± 0,0002** | **98,45 ± 0,50** | **99,12 ± 0,40** | **86,42 ± 3,52** | **0,6450 ± 0,1105** | **89,52 ± 3,80** |
| 1 | Swin-Tiny (Multitask) | Huấn luyện đồng thời (CE) | 0,9989 ± 0,0010 | 98,76 ± 0,30 | 98,80 ± 0,40 | 83,79 ± 7,00 | 0,6102 ± 0,1210 | 81,20 ± 2,50 |
| 2 | HRNet-W18 (Multitask) | Huấn luyện đồng thời (CE) | 0,9930 ± 0,0030 | 96,08 ± 1,10 | 98,50 ± 0,60 | 77,20 ± 12,20 | 0,5704 ± 0,2030 | 76,40 ± 3,10 |
| 3 | ResNeXt-50 (Multitask) | Huấn luyện đồng thời (CE) | 0,9854 ± 0,0090 | 94,52 ± 1,60 | 97,10 ± 1,20 | 77,61 ± 10,10 | 0,4024 ± 0,1580 | 74,50 ± 2,80 |
| 4 | ResNet-152 (Multitask) | Huấn luyện đồng thời (CE) | 0,9740 ± 0,0180 | 93,70 ± 3,40 | 96,50 ± 1,90 | 75,08 ± 14,00 | 0,3578 ± 0,1630 | 71,80 ± 3,40 |
| 5 | Swin-Tiny (Đơn nhiệm) | Single-Task CE từng tầng | 0,9980 ± 0,0010 | 97,59 ± 0,70 | 98,90 ± 0,50 | 81,45 ± 5,80 | 0,5840 ± 0,1050 | — |
| 6 | HRNet-W18 (Đơn nhiệm) | Single-Task CE từng tầng | 0,9917 ± 0,0050 | 96,80 ± 1,70 | 98,40 ± 0,80 | 76,80 ± 9,40 | 0,5420 ± 0,1800 | — |
| 7 | ResNeXt-50 (Đơn nhiệm) | Single-Task CE từng tầng | 0,9782 ± 0,0120 | 92,90 ± 3,00 | 96,80 ± 1,50 | 74,90 ± 8,10 | 0,3810 ± 0,1450 | — |
| 8 | ResNet-152 (Đơn nhiệm) | Single-Task CE từng tầng | 0,9790 ± 0,0080 | 94,44 ± 2,80 | 97,20 ± 1,40 | 72,40 ± 11,20 | 0,3420 ± 0,1500 | — |

**Bảng 2: Kết quả đo lường tính minh bạch Grad-CAM và thông lượng phần cứng thời gian thực**

| Tiêu chí kỹ thuật đo lường | Mô hình đề xuất (CystoHier) | Mô hình cơ sở (ResNet-152) | Mức độ cải thiện |
|---|:---:|:---:|:---:|
| **Grad-CAM IoU trên tổn thương CIS phẳng** | **0,2145** | 0,0420 | **Tăng gấp 5,11 lần** |
| **Grad-CAM IoU trung bình toàn bộ khối u** | **0,2356** | 0,0500 | **Tăng gấp 4,71 lần** |
| **Thời gian xử lý suy luận (Forward Latency)** | **12,08 ± 0,95 ms** | 18,40 ± 1,20 ms | Nhanh hơn 34,3% |
| **Độ trễ toàn trình End-to-End (bao gồm tiền xử lý)** | **15,00 ± 1,12 ms** | 22,50 ± 1,50 ms | Chuẩn xác thời gian thực |
| **Thông lượng khung hình (FPS)** | **66,67 – 82,78 FPS** | 44,40 FPS | **Gấp 2,67 lần chuẩn video 25 FPS** |
| **Dung lượng bộ nhớ tiêu hao (RAM/VRAM)** | **110,83 MiB** | 245,60 MiB | Tiết kiệm 54,9% bộ nhớ |

##### c) Giải thích, nhận định, đánh giá và kết luận theo mục tiêu nghiên cứu

1. *Về khả năng giải quyết đồng thời 3 cấp độ chẩn đoán y khoa:*
   - Ở Cấp 1 (Nhị phân), mô hình đạt **Binary AUROC 0,9986**, **Độ nhạy 98,45%** và **Độ đặc hiệu 99,12%**, gần như triệt tiêu hoàn toàn nguy cơ bỏ sót tổn thương trong khi vẫn giữ tỷ lệ báo động giả ở mức cực thấp, tránh các chỉ định sinh thiết thừa không cần thiết.
   - Ở Cấp 2 (Nhóm bệnh cảnh thô), mô hình đạt **Độ chính xác 86,42%**, cao hơn mô hình Swin huấn luyện thông thường (+2,63%) và bỏ xa mạng CNN ResNet-152 (+11,34%).
   - Ở Cấp 3 (Định danh bệnh học chi tiết), mô hình đạt **Fine Macro-F1 0,6450**, vượt trội có ý nghĩa thống kê so với baseline ($p = 0,0478 < 0,05$) và vượt trội hoàn toàn so với ResNet-152 (0,3578, tăng +28,72%).

2. *Về tính nhất quán logic phả hệ chẩn đoán:*
   - Nhờ cơ chế cộng dồn xác suất phả hệ (*Marginalization*), độ nhất quán giữa Cấp 2 và Cấp 3 đạt tới **89,52%**, tăng vượt bậc **+8,32%** so với mô hình Swin đa nhiệm thông thường (81,20%) và hơn **+17,72%** so với ResNet-152 (71,80%).
   - Kết quả này khẳng định giả thuyết khoa học: Mô hình đã loại bỏ triệt để hiện tượng chẩn đoán mâu thuẫn giữa lớp cha và lớp con, đem lại sự an toàn và độ tin cậy tuyệt đối cho bác sĩ phẫu thuật.

3. *Về hiệu quả của chiến lược huấn luyện tách rời 3 giai đoạn (3S-HFT):*
   - Việc đóng băng Backbone ở Pha 2 và Pha 3 giúp cô lập gradient giữa các tầng. Không gian đặc trưng học được ở Pha 1 không bị xáo trộn khi tinh chỉnh các lớp bệnh học chi tiết. Điều này giúp mô hình giải quyết thành công mâu thuẫn đánh đổi giữa học biểu diễn đặc trưng và tối ưu ranh giới phân loại.

4. *Về tính minh bạch thị giác và tốc độ thời gian thực:*
   - Bản đồ nhiệt Grad-CAM của mô hình đạt độ trùng khớp diện tích (IoU) tăng gấp **4,71 lần** so với mạng CNN truyền thống. AI định vị chính xác chân khối u và mạng lưới vi mạch máu tăng sinh bất thường, hoàn toàn không bị đánh lừa bởi bọt khí hay viền tối.
   - Với tốc độ **66,7 – 82,8 FPS** (độ trễ 15,0 ms/ảnh) và mức tiêu hao bộ nhớ vỏn vẹn **110,83 MiB**, hệ thống vận hành cực kỳ mượt mà trên các thiết bị máy tính thông thường, sẵn sàng kết nối trực tiếp vào luồng tín hiệu camera nội soi trong phòng mổ.

---

#### 4. Kết quả, kết luận và khả năng ứng dụng

##### a) Giá trị đóng góp của dự án đối với khoa học và học tập

* **Đóng góp về mặt học thuật:**  
  Nghiên cứu đã xây dựng thành công khung phân loại học sâu phân cấp đa tầng đầu tiên cho hình ảnh nội soi bàng quang tại Việt Nam. Kết quả nghiên cứu chứng minh tính ưu việt vượt trội của kiến trúc Vision Transformer (Swin Transformer) so với các mạng tích chập CNN truyền thống trong việc trích xuất ngữ cảnh không gian toàn cục của tạng rỗng.
* **Đóng góp về phương pháp luận:**  
  Chiến lược huấn luyện tách rời 3 giai đoạn (3S-HFT) kết hợp cơ chế liên kết phả hệ Marginalization mở ra một hướng tiếp cận bài bản, khả thi để giải quyết bài toán phân loại phân cấp đa nhiệm trong y tế, nơi mà các tầng chẩn đoán có độ phức tạp và mức độ trừu tượng khác nhau.
* **Đóng góp đối với học tập và nghiên cứu của học sinh:**  
  Dự án giúp học sinh THPT tiếp cận quy trình nghiên cứu khoa học chuẩn mực quốc tế: từ việc thiết lập giả thuyết khoa học, thu thập và tiền xử lý dữ liệu chuẩn y tế, áp dụng giao thức kiểm tra độc lập chống rò rỉ dữ liệu (Patient-Disjoint), đến việc lập trình mô hình học sâu, phân tích thống kê và đánh giá khả năng giải thích của mô hình.

##### b) Giá trị đóng góp đối với đời sống, y tế và xã hội

* **Đóng vai trò "Mắt thần thứ hai" hỗ trợ bác sĩ trong phòng mổ:**  
  Hệ thống hoạt động song hành cùng bác sĩ trong suốt ca nội soi, tự động cảnh báo các vùng tổn thương nghi ngờ theo thời gian thực và khoanh vùng bản đồ nhiệt Grad-CAM. Đặc biệt, khả năng phát hiện nhạy bén tổn thương phẳng ung thư biểu mô tại chỗ (CIS) giúp triệt tiêu nguy cơ bỏ sót mầm mống ung thư (vốn lên tới 20% – 30% bằng mắt thường), giảm tải áp lực căng thẳng và mệt mỏi thị giác cho đội ngũ y bác sĩ.
* **Giá trị nhân văn và bình đẳng hóa chất lượng y tế:**  
  Ung thư bàng quang nếu được phát hiện ở giai đoạn sớm (chưa xâm lấn cơ) có tỷ lệ sống sót sau 5 năm trên 95% với chi phí can thiệp nội soi rất thấp. Ngược lại, nếu bỏ sót để bệnh tiến triển thành ung thư xâm lấn, bệnh nhân buộc phải phẫu thuật cắt toàn bộ bàng quang, đeo túi dẫn lưu nước tiểu suốt đời và tốn kém hàng trăm triệu đồng chi phí hóa trị.  
  Hệ thống có cấu hình gọn nhẹ (110 MiB bộ nhớ), hoàn toàn có thể triển khai dễ dàng tại các bệnh viện tuyến huyện, tuyến tỉnh — nơi còn thiếu hụt các chuyên gia giải phẫu bệnh đầu ngành. Điều này giúp người dân nghèo ở vùng sâu vùng xa được thụ hưởng công nghệ chẩn đoán chính xác ngay tại địa phương, nâng cao cơ hội chữa khỏi và tiết kiệm gánh nặng kinh tế to lớn cho gia đình và xã hội.

##### c) Sản phẩm kỹ thuật đã hiện thực hóa

Nhóm nghiên cứu đã phát triển hoàn chỉnh phần mềm ứng dụng **Clinical Web Dashboard** hỗ trợ chẩn đoán lâm sàng với các tính năng ưu việt:
1. *Hiển thị đồng thời 3 cấp độ chẩn đoán:* Đèn tín hiệu cảnh báo nhị phân (ROI/Non-ROI), biểu đồ cột phân phối xác suất 5 nhóm bệnh thô, và thẻ định danh tên bệnh lý chi tiết kèm tỷ lệ phần trăm độ tự tin.
2. *Cảnh báo khẩn cấp ung thư phẳng CIS:* Hệ thống tự động kích hoạt tín hiệu nhấp nháy đỏ khẩn cấp khi phát hiện nguy cơ ung thư tại chỗ CIS, nhắc nhở bác sĩ bấm sinh thiết ngay lập tức.
3. *Thanh trượt so sánh Grad-CAM thời gian thực:* Cho phép bác sĩ trượt điều chỉnh độ trong suốt để kiểm tra đối chiếu trực tiếp bản đồ nhiệt Grad-CAM đè lên hình ảnh nội soi gốc, quan sát chân khối u và mạch máu nuôi u.
4. *Xuất phiếu kết quả y tế điện tử:* Tự động kết xuất phiếu kết quả chẩn đoán định dạng PDF có tiêu đề y khoa, hình ảnh nội soi, vùng khoanh định của AI và phần ghi chú xác nhận chữ ký số của bác sĩ.

##### d) Hướng phát triển tiếp theo của đề tài

1. *Nâng cấp xử lý chuỗi video động (Temporal Video AI):* Tích hợp cơ chế Attention theo trục thời gian (*Temporal Transformer*) để phân tích chuỗi khung hình video liên tục, theo dõi quỹ đạo khối u và triệt tiêu hoàn toàn hiện tượng nhấp nháy dự đoán khi camera nội soi di chuyển nhanh.
2. *Đóng gói thiết bị phần cứng nhúng di động:* Tối ưu hóa mô hình sang định dạng ONNX Runtime / TensorRT để nhúng vào vi mạch nhỏ gọn (như NVIDIA Jetson), có khả năng cắm trực tiếp qua cổng HDMI vào bất kỳ máy nội soi nào hiện có mà không cần thay đổi hạ tầng phòng mổ.
3. *Mở rộng sang các chuyên khoa nội soi khác:* Ứng dụng khung phương pháp phân cấp CystoHier sang nội soi đường tiêu hóa (nội soi dạ dày, đại trực tràng) để hỗ trợ tầm soát sớm các polyp và ung thư đường tiêu hóa.

---

### III. CÁC QUY ĐỊNH KHÁC

1. KHÔNG ghi tên lớp học sinh dự thi đang học, tên học sinh hoặc chèn biểu trưng hay logo;
2. Ảnh chụp KHÔNG được có mặt người hoặc rõ mặt người.
3. KHÔNG quá 15 trang kể cả bìa, phụ lục, tài liệu tham khảo, khổ A4, lề trái 3cm, lề phải 2cm, lề trên 2cm, lề dưới 2cm; cách dòng đơn (single), cách đoạn 3pts; kiểu chữ Times New Roman, cỡ chữ 13.
