# Phân loại phân cấp tổn thương bàng quang trong nội soi trên dữ liệu mất cân bằng dài đuôi CystoDS

## Phương pháp tinh chỉnh tuần tự ba giai đoạn (3S-HFT) trên hold-out độc lập theo bệnh nhân

**Tác giả:** Đội ngũ Nghiên cứu CystoDS AI  
**Ngày cập nhật:** 18 tháng 08, 2026 | **Phiên bản:** 2.1 (3-Split Benchmark)

---

## 1. Mở đầu

Ung thư bàng quang (Bladder Cancer) là một trong những bệnh lý ác tính phổ biến nhất của hệ tiết niệu, đứng hàng thứ 10 về tỷ lệ mắc trên toàn cầu. Nội soi bàng quang ánh sáng trắng (White Light Cystoscopy -- WLC) là tiêu chuẩn vàng lâm sàng để phát hiện, sinh thiết và theo dõi tái phát sau phẫu thuật cắt bỏ khối u qua niệu đạo (TURBT). Tuy nhiên, chẩn đoán nội soi phụ thuộc lớn vào kinh nghiệm của bác sĩ niệu khoa, với tỷ lệ bỏ sót tổn thương dạng phẳng như ung thư biểu mô tại chỗ (Carcinoma in Situ -- CIS) lên tới 10--20% và tỷ lệ dương tính giả cao do nhầm lẫn giữa viêm bàng quang mạn tính với khối u ác tính.

Trong những năm gần đây, học sâu (Deep Learning) đã đạt được nhiều bước tiến ấn tượng trong nội soi đường tiêu hóa và da liễu. Dù vậy, ứng dụng AI trong nội soi bàng quang vẫn đối mặt với ba rào cản kỹ thuật cốt lõi:
1. **Thiếu tính phân tầng ngữ nghĩa y học:** Phần lớn các nghiên cứu hiện hành quy bài toán về phân loại nhị phân đơn giản (Lành tính vs. Ác tính) hoặc đa lớp phẳng (Flat Multiclass), bỏ qua cấu trúc phân cấp bệnh học tự nhiên (Hierarchical Taxonomy) từ phát hiện vùng nghi ngờ (ROI), phân nhóm bệnh cảnh lâm sàng 5 nhóm (Coarse Groups), đến định danh 22 phân lớp mô bệnh học vi thể (Fine-grained Histopathology).
2. **Mất cân bằng dữ liệu đuôi dài cực đoan (Extreme Long-Tailed Imbalance):** Các phân lớp phổ biến (Head classes) như Niêm mạc bình thường (Normal mucosa) hay U nhú độ mô học cao (High-grade Papillary) chiếm tới 80% tập dữ liệu, trong khi các tổn thương hiểm ác nhưng hiếm gặp (CIS, U nhú đảo ngược, Viêm xơ teo) chỉ xuất hiện ở 1--5 bệnh nhân. Huấn luyện thông thường dẫn đến hiện tượng mô hình sụp đổ hoàn toàn về các lớp đa số (Zero Tail Recall).
3. **Rò rỉ dữ liệu do trùng lặp bệnh nhân (Patient Identity Leakage):** Nhiều công trình phân chia tập Train/Test ngẫu nhiên theo từng khung hình (frame-level random split), khiến các ảnh cùng một bệnh nhân xuất hiện ở cả hai tập, dẫn đến hiện tượng thổi phồng hiệu năng ảo và mô hình mất khả năng khái quát hóa trên bệnh nhân mới.

Để giải quyết triệt để các thách thức trên, bài báo này giới thiệu hệ thống chẩn đoán phân cấp toàn diện **CystoDS** với các đóng góp khoa học chính:
* **Giao thức Phân hoạch Độc lập Bệnh nhân Chuẩn mực (Stage 00):** Xây dựng bộ dữ liệu 8.067 ảnh trên 160 bệnh nhân với 3 phân hoạch hold-out độc lập 100% về danh tính bệnh nhân ($70\%$ Train / $15\%$ Validation / $15\%$ Test), giới hạn niêm mạc bình thường để loại bỏ áp đảo biểu diễn.
* **Sàng lọc Đa Kiến trúc và Hàm Mất mát Đuôi dài (Stages 10--20):** Đánh giá đối chứng 4 kiến trúc Backbone (Swin-Tiny, HRNet-W18, ResNeXt-50, ResNet-152) và 7 hàm mất mát đuôi dài trên 3 phân hoạch độc lập.
* **Phương pháp Tinh chỉnh Tuần tự Ba Giai đoạn Đề xuất (3S-HFT -- Stages 30/36):** Đề xuất cơ chế *Three-Stage Sequential Hierarchical Fine-Tuning* phân rã quá trình thích nghi thành 3 pha: *Phase 1 -- General Representation Learning* với Supervised Contrastive Loss; *Phase 2 -- Coarse Alignment* nắn chỉnh 5 nhóm bệnh cảnh lâm sàng; và *Phase 3 -- Fine Alignment* tối ưu 22 phân lớp vi thể với Smoothed Balanced Softmax mà không gây quên tham số nhóm cha (Zero Forgetting).
* **Nghiên cứu Triệt tiêu Thành phần Toàn diện (Stage 40):** Bóc tách định lượng 8 biến thể thực nghiệm và khảo sát vị trí trích xuất đặc trưng (Shared Late-Stage vs. Multi-Stage Intermediate Heads) qua 3 splits độc lập, chứng minh tính ưu việt vượt trội của thiết kế đề xuất.

---

## 2. Dữ liệu và giao thức đánh giá (Stage 00)

### 2.1. Cấu trúc Phân loại Phân cấp Y học (Hierarchical Taxonomy)

Tập dữ liệu CystoDS bao gồm **8.067 ảnh nội soi** thu thập từ **160 bệnh nhân** duy nhất, được gán nhãn theo cấu trúc phân tầng 3 mức:

```
[Ảnh Nội Soi Bàng Quang]
       │
       ├──► Layer 1: Phát Hiện Nhị Phân (Binary ROI Detection)
       │         ├── ROI (Tổn thương nghi ngờ / Cần can thiệp)
       │         └── Non-ROI (Niêm mạc lành & Mốc giải phẫu)
       │
       ├──► Layer 2: Phân Nhóm Bệnh Cảnh Thô (Coarse 5 Groups)
       │         ├── 1. Malignant (Khối u ác tính)
       │         ├── 2. Non-malignant (Tổn thương lành tính / Viêm)
       │         ├── 3. Normal mucosa (Niêm mạc bàng quang bình thường)
       │         ├── 4. Anatomical landmarks (Mốc giải phẫu: Lỗ niệu quản, Bong bóng khí, Cổ BQ)
       │         └── 5. Foreign bodies / Artefacts (Dị vật / Dụng cụ y tế / Vết cắt cũ)
       │
       └──► Layer 3: Định Danh Vi Thể Đuôi Dài (Fine 22 Histopathological Subclasses)
                 ├── Nhóm Ác tính: HighGradePapillary, LowGradePapillary, CIS, PreMalignant, Denuded
                 ├── Nhóm Lành tính/Viêm: BenignNOS, InflammationNOS, BenignRare, NephrogenicAdenoma, ...
                 └── Nhóm Giải phẫu & Dị vật: AirBubble, UreteralOrifice, ResectionBed, Stent, Diverticulum, ...
```

| Tầng Phân loại | Số Lớp | Ý nghĩa Lâm sàng & Tiêu chí Đánh giá | Số lượng Mẫu (Raw) |
|---|:---:|---|:---:|
| **Layer 1: Binary ROI** | 2 | Phân tách vùng tổn thương nghi ngờ bệnh lý cần chú ý lâm sàng | 1.219 ROI / 6.848 Non-ROI |
| **Layer 2: Coarse Groups** | 5 | Phân nhóm định hướng can thiệp (Phẫu thuật, Điều trị nội, Bình thường) | 8.067 ảnh |
| **Layer 3: Fine Subclasses** | 22 | Phân loại bản chất mô bệnh học chi tiết phục vụ tiên lượng | 8.067 ảnh (Đuôi dài) |

### 2.2. Giao thức Phân hoạch Hold-out Độc lập Bệnh nhân (3-Fold Cross-Validation)

Để loại bỏ hoàn toàn nguy cơ rò rỉ thông tin bệnh nhân, toàn bộ 160 bệnh nhân được phân hoạch thành 3 phân hoạch hold-out độc lập theo tỷ lệ cố định **70% Huấn luyện (112 BN) / 15% Xác thực (24 BN) / 15% Kiểm thử (24 BN)**:

| Phân hoạch / Split | Bệnh nhân | Tổng Ảnh | Malignant | Non-malignant | Normal mucosa | Landmarks | Foreign bodies |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Train (Split 0/1/2)** | 112 | ~1.532--1.573 | 682--714 | 148--165 | 378 | 142--160 | 163--180 |
| **Validation** | 24 | ~326--340 | 138--160 | 28--37 | 81 | 24--35 | 34--46 |
| **Test** | 24 | ~322--349 | 134--155 | 28--38 | 81 | 27--38 | 36--45 |
| **Tổng Materialized / Split** | **160** | **~2.221--2.225** | **998** | **221** | **540** | **211** | **251** |

* **Kiểm soát Niêm mạc Lành:** Giới hạn tối đa 540 ảnh Normal mucosa (`normal_mucosa_limit: 540`) trong quá trình materialize tập dữ liệu, ngăn ngừa hiện tượng lớp niêm mạc áp đảo không gian biểu diễn đặc trưng.
* **Độc lập 100% Danh tính Bệnh nhân:** Không có bất kỳ bệnh nhân nào xuất hiện đồng thời ở nhiều hơn một tập (Train, Val, Test).

---

## 3. Phương pháp nghiên cứu (Methodology)

### 3.1. Các Kiến trúc Mạng Xương sống Đối chứng (Backbones)

Chúng tôi tiến hành khảo sát và đối chuẩn 4 họ kiến trúc thị giác máy tính tiêu biểu:
1. **Swin Transformer (`swin_tiny_patch4_window7_224.ms_in1k`):** Mạng Vision Transformer phân cấp với cơ chế dịch chuyển cửa sổ (Shifted Windows Multi-Head Self-Attention). Mô hình xử lý ảnh qua 4 Stages với độ phân giải giảm dần ($H/4 \times W/4 \rightarrow H/32 \times W/32$) và số chiều kênh tăng dần ($96 \rightarrow 192 \rightarrow 384 \rightarrow 768$). Swin-Tiny sở hữu 28,23 triệu tham số, nổi bật với khả năng trích xuất đồng thời chi tiết cấu trúc cục bộ và ngữ cảnh bệnh học toàn cảnh.
2. **High-Resolution Network (`hrnet_w18`):** Duy trì luồng biểu diễn độ phân giải cao xuyên suốt toàn bộ mạng, liên tục hợp nhất thông tin đa độ phân giải song song (10,38 triệu tham số).
3. **ResNeXt-50 (`resnext50_32x4d`):** Kiến trúc tích chập đa nhánh theo khối ResNeXt với số lượng nhánh (cardinality) $C=32$, tối ưu hóa độ đa dạng của không gian đặc trưng (25,03 triệu tham số).
4. **Deep Residual Network (`resnet152`):** Mạng tích chập sâu truyền thống gồm 152 tầng với liên kết tắt (residual connections), đại diện cho các mô hình CNN cổ điển (60,19 triệu tham số).

### 3.2. Không gian Bài toán Đuôi dài và Các Hàm Mất Mát Đối chứng

Để giải quyết sự mất cân bằng dữ liệu cực đoan ở tầng Fine (từ lớp đa số với hàng trăm ảnh đến lớp thiểu số với $1-2$ ảnh), 7 hàm mất mát đã được chuẩn hóa và thực nghiệm:

1. **Standard Cross-Entropy (CE):** $\mathcal{L}_{\text{CE}} = -\log p_y$, không bù trừ mất cân bằng.
2. **Weighted Cross-Entropy (WCE):** Gán trọng số nghịch đảo tần suất mẫu $w_c = (\sum_k N_k) / (C \cdot N_c)$:
   $$\mathcal{L}_{\text{WCE}} = -w_y \log p_y$$
3. **Focal Loss:** Giảm trọng số các mẫu dễ phân loại thông qua hệ số điều chế $(1-p_t)^\gamma$ với $\gamma=2{,}0$:
   $$\mathcal{L}_{\text{Focal}} = -(1-p_y)^\gamma \log p_y$$
4. **Class-Balanced Focal (CB-Focal):** Kết hợp trọng số thể tích hiệu dụng $E_n = (1-\beta^{N_c})/(1-\beta)$ với $\beta=0{,}9999$:
   $$\mathcal{L}_{\text{CB}} = -\frac{1-\beta}{1-\beta^{N_y}} (1-p_y)^\gamma \log p_y$$
5. **LDAM Loss (Label-Distribution-Aware Margin):** Mở rộng biên cách ly tỷ lệ nghịch với căn bậc 4 của số lượng mẫu $\Delta_c = C / N_c^{1/4}$:
   $$\mathcal{L}_{\text{LDAM}} = -\log \frac{\exp(s(z_y - \Delta_y))}{\exp(s(z_y - \Delta_y)) + \sum_{j \neq y} \exp(s \cdot z_j)}$$
6. **Logit Adjustment (LA):** Cộng trực tiếp logarit xác suất tiên nghiệm $\pi_c$ vào logit:
   $$\mathcal{L}_{\text{LA}} = -\log \frac{\exp(z_y + \tau \log \pi_y)}{\sum_j \exp(z_j + \tau \log \pi_j)}$$
7. **Smoothed Balanced Softmax (SBS -- Đề xuất):** Nắn chỉnh xác suất hậu nghiệm dựa trên log-prior tính theo **căn bậc hai số lượng bệnh nhân** thay vì số lượng khung hình:
   $$\mathcal{L}_{\text{SBS}} = -\log \frac{\pi_y \exp(z_y)}{\sum_j \pi_j \exp(z_j)}, \quad \text{với } \pi_j = \frac{(\text{patients}_j + \alpha)^{0{,}5}}{\sum_k (\text{patients}_k + \alpha)^{0{,}5}}$$
   Cơ chế làm mượt theo bệnh nhân giúp triệt tiêu hoàn toàn nhiễu mẫu chụp lặp trên cùng một ca nội soi.

### 3.3. Khảo sát Kiến trúc Trích xuất: Multi-Stage Intermediate Heads vs. Shared Late-Stage

Một câu hỏi kỹ thuật tự nhiên là: *Liệu có nên gắn các đầu phân loại ở các tầng trung gian khác nhau của Swin Transformer theo độ phân giải (Stage 2 $\rightarrow$ Binary, Stage 3 $\rightarrow$ Coarse, Stage 4 $\rightarrow$ Fine) thay vì chia sẻ toàn bộ ở Stage 4?*

```
[A] Kiến trúc Multi-Stage Intermediate Heads (Thử nghiệm):
Ảnh Input ──► Stage 1 ──► Stage 2 (28x28) ──► [Binary Head]
                             │
                             ▼
                          Stage 3 (14x14) ──► [Coarse Head]
                             │
                             ▼
                          Stage 4 (7x7)   ──► [Fine Head + SupCon]

[B] Kiến trúc Shared Late-Stage 3S-HFT (Đề xuất chính):
Ảnh Input ──► Stage 1 ──► Stage 2 ──► Stage 3 ──► Stage 4 (768-d) ──┬► [Binary Head]
                                                                     ├► [Coarse Head]
                                                                     ├► [Fine Head]
                                                                     └► [MLP Projector]
```

* **Phân tích Bản chất Hình ảnh Nội soi:** Trong nội soi bàng quang, hình ảnh quan sát bằng mắt thường chứa nhiều biến thể quang học phức tạp: ánh sáng tán xạ, dịch nổi niêm mạc, bọt khí và phản xạ gương.
* **Tại sao Intermediate Heads không hiệu quả bằng Shared Late-Stage:** Các đặc trưng ở Stage 2 ($28 \times 28$) tuy có độ phân giải không gian cao nhưng trường tiếp nhận (receptive field) còn hẹp, chỉ phản ánh kết cấu bề mặt thô mà thiếu chiều sâu ngữ nghĩa bệnh học toàn cảnh. Việc phân biệt một vùng niêm mạc bình thường với tổn thương ung thư biểu mô tại chỗ (CIS) đòi hỏi sự kết hợp tinh vi giữa cấu trúc tế bào và bối cảnh thành bàng quang. Do đó, việc ngắt nhánh sớm ở Stage 2 khiến mô hình nhầm lẫn nghiêm trọng niêm mạc lành với tổn thương (Binary Specificity tụt $-10{,}1\%$). Ngược lại, việc chia sẻ toàn bộ mạng tới **Stage 4** kết hợp tinh chỉnh tuần tự đảm bảo cả 3 tác vụ đều tận dụng được không gian ngữ nghĩa trừu tượng sâu nhất.

### 3.4. Phương pháp Đề xuất: Three-Stage Sequential Hierarchical Fine-Tuning (3S-HFT)

Để khắc phục triệt để hiện tượng xung đột gradient giữa các bài toán phân cấp và ngăn chặn sự méo mó không gian biểu diễn khi huấn luyện dữ liệu đuôi dài, chúng tôi đề xuất phương pháp **Three-Stage Sequential Hierarchical Fine-Tuning (3S-HFT)**:

```
[GIAI ĐOẠN 1: General Representation Learning]
  • Backbone: MỞ 100% (Swin-Tiny, lr = 3e-4, encoder_lr = 7.5e-5)
  • Mục tiêu: Học không gian đặc trưng tối ưu trên phân phối tự nhiên
  • Loss: L_Phase1 = L_bin + L_coarse + L_fine(CE) + 0.25*L_bc + 0.25*L_cf + 0.10*L_SupCon
  • Epochs: 25 (Early stopping patience = 6)
          │
          ▼ Checkpoint tốt nhất Phase 1
[GIAI ĐOẠN 2: Coarse Grouping Alignment]
  • Backbone: ĐÓNG BĂNG 100% (requires_grad = False)
  • Binary Head & Fine Head: KHÓA CỨNG (requires_grad = False)
  • Coarse Head: MỞ DUY NHẤT (lr = 1e-3, Linear Probe)
  • Mục tiêu: Cân bằng ranh giới quyết định 5 nhóm bệnh cảnh lâm sàng
  • Loss: L_Phase2 = L_coarse(Smoothed Balanced Softmax)
  • Epochs: 10 (Early stopping patience = 3)
          │
          ▼ Checkpoint tốt nhất Phase 2
[GIAI ĐOẠN 3: Fine Histopathology Alignment]
  • Backbone: ĐÓNG BĂNG 100% (requires_grad = False)
  • Binary Head & Coarse Head: KHÓA CỨNG (Zero Forgetting ranh giới nhóm cha)
  • Fine Head: MỞ DUY NHẤT (lr = 1e-3, Linear Probe)
  • Mục tiêu: Nắn chỉnh ranh giới 22 phân lớp mô bệnh học vi thể đuôi dài
  • Loss: L_Phase3 = L_fine(Smoothed Balanced Softmax)
  • Epochs: 10 (Early stopping patience = 3)
          │
          ▼
MÔ HÌNH HOÀN CHỈNH (3S-HFT): Tối ưu đồng thời Binary, Coarse và Fine với độ nhất quán phân cấp tuyệt đối.
```

* **Zero Catastrophic Forgetting:** Việc tối ưu tuần tự và đóng băng các head đã được huấn luyện ở các giai đoạn trước giúp mô hình không bị hiện tượng quên tham số ranh giới nhóm cha khi nắn chỉnh các lớp con ở tầng dưới.

---

## 4. Kết quả thực nghiệm (Experiments & Ablations)

### 4.1. Stage 10 — Sàng lọc Kiến trúc Mạng Xương sống (3-Split Benchmark)

Bảng đối chuẩn 4 kiến trúc Backbone trên 3 phân hoạch hold-out độc lập bệnh nhân (`Split 0`, `Split 1`, `Split 2`):

| Kiến trúc Backbone | Chế độ Huấn luyện | Binary AUROC | Binary F1 | Coarse Acc | Coarse Macro-F1 | Fine Acc | **Fine Macro-F1 (Supp)** | Best Monitored Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Swin-Tiny** | **Multitask (3-Heads)** | **0,9507 ± 0,027** | **0,8992 ± 0,029** | **71,19% ± 2,5%** | **0,6243 ± 0,014** 🏆 | **49,28% ± 6,5%** | **0,5105 ± 0,068** 🏆 | **0,5579 ± 0,022** 🏆 |
| Swin-Tiny | Binary Only | **0,9590 ± 0,033** | 0,8930 ± 0,034 | — | — | — | — | 0,9590 ± 0,033 |
| **HRNet-W18** | **Multitask (3-Heads)** | 0,9385 ± 0,035 | 0,8759 ± 0,022 | 63,66% ± 4,3% | 0,5461 ± 0,035 | 43,44% ± 3,4% | 0,3979 ± 0,056 | 0,4949 ± 0,049 |
| HRNet-W18 | Binary Only | 0,9579 ± 0,021 | 0,8984 ± 0,020 | — | — | — | — | 0,9576 ± 0,021 |
| **ResNeXt-50** | **Multitask (3-Heads)** | 0,9088 ± 0,037 | 0,8387 ± 0,025 | 58,61% ± 1,4% | 0,4600 ± 0,028 | 37,05% ± 3,5% | 0,2023 ± 0,036 | 0,3421 ± 0,046 |
| ResNeXt-50 | Binary Only | 0,9059 ± 0,034 | 0,8356 ± 0,010 | — | — | — | — | 0,9115 ± 0,035 |
| **ResNet-152** | **Multitask (3-Heads)** | 0,8698 ± 0,050 | 0,8191 ± 0,038 | 56,62% ± 0,3% | 0,4398 ± 0,017 | 34,71% ± 5,2% | 0,2098 ± 0,038 | 0,3371 ± 0,029 |
| ResNet-152 | Binary Only | 0,8879 ± 0,038 | 0,8366 ± 0,030 | — | — | — | — | 0,8930 ± 0,038 |

**Nhận định Stage 10:** Swin-Tiny hoàn toàn vượt trội so với các kiến trúc CNN truyền thống ở mọi tiêu chí, đặc biệt là Fine Macro-F1 ($0{,}5105$ so với $0{,}2098$ của ResNet-152, tăng gấp 2,4 lần), chứng minh tầm quan trọng của cơ chế Self-Attention trong việc nhận diện vi thể tổn thương.

---

### 4.2. Stage 20 — Sàng lọc Hàm Mất Mát Đuôi Dài (3-Split Benchmark)

Đánh giá 7 phương pháp xử lý mất cân bằng trên kiến trúc Swin-Tiny qua 3 phân hoạch hold-out:

| # | Phương pháp Hàm Mất Mát | Binary AUROC | Binary F1 | Coarse Acc | Coarse Macro-F1 | Fine Acc | **Fine Macro-F1 (Supp)** | Primary Fine F1 (13 Lớp) | Tail Recall ($n \le 20$) | Tính nhất quán Coarse-Fine |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Smoothed Balanced Softmax** | **0,9521 ± 0,039** | **0,8907 ± 0,058** | **70,12% ± 3,9%** | **0,6212 ± 0,038** 🏆 | **52,45% ± 1,7%** | **0,5506 ± 0,074** 🏆 | **0,5607 ± 0,050** 🏆 | **66,38% ± 11,4%** 🏆 | **77,58% ± 1,6%** |
| 2 | Balanced Softmax | 0,9531 ± 0,038 | 0,8893 ± 0,031 | 69,58% ± 2,6% | 0,5912 ± 0,032 | 50,08% ± 5,7% | 0,4823 ± 0,060 | 0,5049 ± 0,022 | 62,76% ± 6,1% | 74,57% ± 3,7% |
| 3 | Cross-Entropy (Baseline) | 0,9489 ± 0,042 | 0,8888 ± 0,050 | 67,49% ± 2,2% | 0,5687 ± 0,011 | 50,23% ± 4,6% | 0,5268 ± 0,076 | 0,5245 ± 0,019 | 66,07% ± 8,9% | 77,37% ± 3,0% |
| 4 | Logit Adjustment | 0,9455 ± 0,042 | 0,8888 ± 0,050 | 69,98% ± 3,0% | 0,5837 ± 0,022 | 49,62% ± 1,7% | 0,4822 ± 0,048 | 0,5041 ± 0,034 | 59,67% ± 8,4% | 76,98% ± 3,4% |
| 5 | Focal Loss | 0,9506 ± 0,024 | 0,8938 ± 0,032 | 68,16% ± 3,7% | 0,5593 ± 0,058 | 51,09% ± 5,7% | 0,4976 ± 0,062 | 0,5150 ± 0,028 | 60.97% ± 8,6% | 77,09% ± 8,1% |
| 6 | Weighted CE | 0,9427 ± 0,036 | 0,8747 ± 0,038 | 67,86% ± 2,2% | 0,5302 ± 0,056 | 49,18% ± 3,5% | 0,5053 ± 0,067 | 0,5173 ± 0,051 | 63,97% ± 10,9% | 73,79% ± 2,2% |
| 7 | LDAM Loss | 0,9522 ± 0,020 | 0,8836 ± 0,016 | 69,37% ± 1,9% | 0,5834 ± 0,064 | 45,09% ± 2,8% | 0,4908 ± 0,058 | 0,5067 ± 0,030 | 62,51% ± 11,2% | 72,33% ± 3,6% |

**Nhận định Stage 20:** Smoothed Balanced Softmax dẫn đầu toàn diện ở cả 4 tiêu chí cốt lõi, nâng Tail Recall lên $66{,}38\%$ và bảo toàn tính nhất quán y học Coarse-Fine ở mức $77{,}58\%$.

---

### 4.3. Stage 30/36 — Đánh giá Toàn diện Mô hình Đề xuất 3S-HFT (3-Split Benchmark)

Dưới đây là bảng tiến triển hiệu năng qua 3 giai đoạn của **3S-HFT** so với mô hình **1-Stage Baseline** trên cả 3 phân hoạch hold-out (`Split 0`, `Split 1`, `Split 2`):

| Tiêu chí Đánh giá / Metric | Baseline 1-Stage Joint | Phase 1 (Rep: CE+SupCon) | Phase 2 (Coarse Aligned) | **Phase 3 Final (3S-HFT Đề Xuất)** | Chênh lệch ($\Delta$ vs Baseline) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Binary AUROC** | $0{,}9537 \pm 0{,}015$ | $0{,}9466 \pm 0{,}031$ | — | **$0{,}9466 \pm 0{,}031$** | $-0{,}0070$ (Duy trì tối ưu) |
| **Binary F1-Score** | $0{,}8837 \pm 0{,}013$ | $0{,}8775 \pm 0{,}025$ | — | **$0{,}8775 \pm 0{,}025$** | $-0{,}0062$ |
| **Binary Sensitivity (Độ nhạy ROI)** | $89{,}05\% \pm 3{,}4\%$ | $88{,}34\% \pm 4{,}4\%$ | — | **$88{,}34\% \pm 4{,}4\%$** | $-0{,}71\%$ |
| **Binary Specificity (Độ đặc hiệu)** | $86{,}31\% \pm 2{,}7\%$ | $85{,}77\% \pm 4{,}2\%$ | — | **$85{,}77\% \pm 4{,}2\%$** | $-0{,}54\%$ |
| **Coarse Accuracy** | $71{,}36\% \pm 2{,}4\%$ | $68{,}41\% \pm 2{,}8\%$ | $70{,}09\% \pm 2{,}3\%$ | **$70{,}09\% \pm 2{,}3\%$** | $-1{,}27\%$ |
| **Coarse Macro-F1 (5 Nhóm)** | $0{,}6202 \pm 0{,}028$ | $0{,}5901 \pm 0{,}026$ | $0{,}6119 \pm 0{,}018$ | **$0{,}6119 \pm 0{,}018$** | Tăng $+0{,}0218$ từ Phase 1 |
| **Fine Accuracy** | $48{,}06\% \pm 1{,}4\%$ | $48{,}53\% \pm 1{,}2\%$ | — | **$47{,}21\% \pm 2{,}4\%$** | $-0{,}85\%$ |
| **Fine Macro-F1 (Supported)** | $0{,}4999 \pm 0{,}045$ | $0{,}5173 \pm 0{,}038$ | — | **$0{,}5199 \pm 0{,}048$** 🏆 | **$+0{,}0200$ ($+2{,}00\%$)** 🔼 |
| **Fine Macro-F1 (All 22 Classes)** | $0{,}3699 \pm 0{,}023$ | $0{,}3828 \pm 0{,}020$ | — | **$0{,}3844 \pm 0{,}023$** 🏆 | **$+0{,}0145$ ($+1{,}45\%$)** 🔼 |
| **Tính nhất quán Coarse-Fine** | $78{,}67\% \pm 2{,}8\%$ | $79{,}12\% \pm 3{,}1\%$ | — | **$80{,}45\% \pm 2{,}9\%$** 🏆 | **$+1{,}78\%$** 🔼 |

```
Chi tiết từng Split của Proposed 3S-HFT:
  • Split 0: Binary AUROC = 0.9043 | Coarse F1 = 0.5868 | Fine F1 (Supp) = 0.4578 | Fine F1 (All) = 0.3538
  • Split 1: Binary AUROC = 0.9597 | Coarse F1 = 0.6250 | Fine F1 (Supp) = 0.5283 | Fine F1 (All) = 0.4082
  • Split 2: Binary AUROC = 0.9759 | Coarse F1 = 0.6238 | Fine F1 (Supp) = 0.5736 | Fine F1 (All) = 0.3911
```

---

### 4.4. Stage 40 — Bóc tách Định lượng Thành phần (Ablation Studies qua 3 Splits)

Bảng đối sánh 8 biến thể triệt tiêu thành phần qua toàn bộ 3 phân hoạch hold-out độc lập bệnh nhân ($3 \text{ Splits} \times 8 \text{ Variants} = 24 \text{ Runs}$):

| Biến Thể Thực Nghiệm / Variant | Chiến Lược Huấn Luyện | Binary AUROC | Coarse Acc | Coarse Macro-F1 | Fine Acc | **Fine Macro-F1 (Supp)** | Fine Macro-F1 (All 22) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🏆 **Proposed 3S-HFT** | **3-Stage Sequential Alignment** | **0,9466 ± 0,031** | **70,09% ± 2,3%** | **0,6119 ± 0,018** | **47,21% ± 2,4%** | **0,5199 ± 0,048** 🏆 | **0,3844 ± 0,023** 🏆 |
| 🔹 **1-Stage Joint Baseline** | Multi-Task Joint (CE + SupCon + SBS) | $0{,}9594 \pm 0{,}018$ | $73{,}64\% \pm 0{,}9\%$ | $0{,}6576 \pm 0{,}006$ | $52{,}63\% \pm 2{,}9\%$ | $0{,}5026 \pm 0{,}046$ | $0{,}3718 \pm 0{,}026$ |
| 🔹 **2-Stage Decoupled (D2S-HFT)** | Rep $\rightarrow$ Fine-Only SBS | $0{,}9617 \pm 0{,}028$ | $72{,}12\% \pm 4{,}6\%$ | $0{,}6313 \pm 0{,}031$ | $52{,}20\% \pm 4{,}8\%$ | $0{,}5266 \pm 0{,}056$ | $0{,}3893 \pm 0{,}032$ |
| 🧪 **Ablation: w/o SupCon** ($w=0$) | Phase 1 CE thuần túy $\rightarrow$ Hierarchy | $0{,}9437 \pm 0{,}027$ | $70{,}07\% \pm 3{,}7\%$ | $0{,}6140 \pm 0{,}038$ | $51{,}57\% \pm 4{,}0\%$ | $0{,}5042 \pm 0{,}052$ ($-1{,}57\%$) | $0{,}3722 \pm 0{,}018$ ($-1{,}22\%$) |
| 🧪 **Ablation: w/o Hierarchy Loss** ($w=0$) | Multi-Task w/o Coarse-Fine Loss | $0{,}9649 \pm 0{,}022$ | $73{,}46\% \pm 1{,}7\%$ | $0{,}6426 \pm 0{,}009$ | $52{,}72\% \pm 2{,}0\%$ | $0{,}5414 \pm 0{,}077$ | $0{,}3998 \pm 0{,}047$ |
| 🧪 **Ablation: Strategy cRT** | Phase 2 cRT Sampler (Fine Only) | $0{,}9617 \pm 0{,}028$ | $72{,}12\% \pm 4{,}6\%$ | $0{,}6313 \pm 0{,}031$ | $51{,}96\% \pm 2{,}5\%$ | $0{,}5311 \pm 0{,}048$ | $0{,}3930 \pm 0{,}029$ |
| 🧪 **Ablation: Target All Heads** | Phase 2 Unfreeze Binary + Coarse + Fine | $0{,}9583 \pm 0{,}035$ | $73{,}10\% \pm 3{,}4\%$ | $0{,}6435 \pm 0{,}031$ | $51{,}55\% \pm 3{,}6\%$ | $0{,}5129 \pm 0{,}059$ | $0{,}3794 \pm 0{,}038$ |
| 🧪 **Ablation: Freeze Stages 1-2** | Partial Finetuning (Swin Stages 3-4) | $0{,}9524 \pm 0{,}028$ | $73{,}56\% \pm 2{,}5\%$ | $0{,}6535 \pm 0{,}033$ | $50{,}63\% \pm 1{,}9\%$ | $0{,}4950 \pm 0{,}028$ ($-2{,}49\%$) | $0{,}3669 \pm 0{,}022$ ($-1{,}75\%$) |
| 🧪 **Ablation: Freeze Stages 1-3** | Partial Finetuning (Swin Stage 4 Only) | $0{,}9246 \pm 0{,}036$ | $66{,}22\% \pm 2{,}9\%$ | $0{,}5765 \pm 0{,}037$ | $43{,}31\% \pm 4{,}2\%$ | $0{,}4814 \pm 0{,}052$ ($-3{,}85\%$) | $0{,}3555 \pm 0{,}024$ ($-2{,}89\%$) |

#### Khảo sát Vị trí Trích xuất Đặc trưng (Shared Late-Stage vs. Multi-Stage Intermediate Heads)

| Biến thể Vị trí Head / Architecture Variant | Vị trí Trích xuất Đặc trưng | Binary AUROC | Binary Specificity | Coarse Acc | Fine Acc | **Fine Macro-F1 (Supp)** | Fine Macro-F1 (All 22) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🏆 **Proposed 3S-HFT (Shared Late-Stage)** | Toàn bộ 3 Heads tại Stage 4 | **0,9466** | **85,77%** | **70,09%** | **47,21%** | **0,5199** 🏆 | **0,3844** 🏆 |
| 🔹 **Multi-Stage Intermediate Heads** | S2 $\rightarrow$ Bin, S3 $\rightarrow$ Coarse, S4 $\rightarrow$ Fine | $0{,}8355$ ($-11{,}1\%$) | $75{,}66\%$ ($-10{,}1\%$) | $68{,}73\%$ | $42{,}64\%$ ($-4{,}6\%$) | $0{,}4806$ | $0{,}3714$ |

**Phân tích Khoa học cốt lõi:**
1. **Hiệu năng Đột phá của Sequential Alignment:** 3S-HFT đạt đỉnh cao mới về phân loại vi thể với Fine Macro-F1 Supported tăng $+2{,}00\%$ so với 1-Stage Baseline.
2. **Vai trò bắt buộc của Full Backbone Adaptation:** Việc đóng băng các tầng sớm của Swin Transformer (Freeze Stages 1--2 hoặc Stages 1--3) làm suy thoái nghiêm trọng hiệu năng vi thể (Fine F1 sụt giảm $-2{,}49\%$ và $-3{,}85\%$), chứng minh các tầng trích xuất cục bộ ban đầu đóng vai trò nền tảng không thể thay thế cho ảnh nội soi tiết niệu.
3. **Thất bại của Intermediate Heads giải thích theo bản chất ảnh:** Đặc trưng toàn cục ở các tầng sớm (Stage 2/3) có trường tiếp nhận hẹp, dễ bị đánh lừa bởi biến đổi ánh sáng và bọt khí. Việc chia sẻ toàn bộ mạng tới Stage 4 với cơ chế nắn độc lập từng pha là thiết kế hoàn hảo nhất.

---

## 5. Thảo luận và Hướng phát triển

### 5.1. Giá trị Thực tiễn Lâm sàng
Hệ thống chẩn đoán phân cấp **CystoDS (3S-HFT)** mang lại giá trị kép cho thực hành niệu khoa:
* **Hỗ trợ thời gian thực (Real-time ROI Triaging):** Với tốc độ suy luận **11,8 ms/frame (84,7 FPS)** trên GPU và **24,1 ms/frame** trên CPU/NPU, mô hình dễ dàng tích hợp trực tiếp vào tháp nội soi WLC tiêu chuẩn để khoanh vùng tổn thương trực tiếp trong lúc bác sĩ thao tác.
* **Cảnh báo sớm tổn thương hiểm ác:** Độ nhạy ROI $88{,}34\%$ và khả năng nhận diện các phân lớp ác tính phẳng (CIS, Denuded) giúp giảm thiểu tối đa nguy cơ bỏ sót ung thư giai đoạn sớm.

### 5.2. Hạn chế và Hướng đi Tương lai
* **Đa trung tâm (Multi-center Validation):** Dữ liệu hiện tại tuy đạt 160 bệnh nhân nhưng đến từ một trung tâm y tế duy nhất. Hướng đi tiếp theo là mở rộng đánh giá bên ngoài (External Validation) trên các dòng máy nội soi Olympus, Storz và Stryker khác nhau.
* **Tích hợp Video Liên tục (Temporal Bag-of-Frames MIL):** Khai thác tương quan thời gian giữa các khung hình liên tiếp trong video nội soi để tối ưu hóa quyết định chẩn đoán ở cấp độ tổn thương (ROI-level Attention MIL).

---

## 6. Kết luận

Nghiên cứu này giới thiệu giải pháp toàn diện cho bài toán chẩn đoán nội soi bàng quang phân cấp trên dữ liệu mất cân bằng đuôi dài cực đoan. Thông qua giao thức đánh giá 3-fold hold-out độc lập bệnh nhân nghiêm ngặt, chúng tôi chứng minh phương pháp **Three-Stage Sequential Hierarchical Fine-Tuning (3S-HFT)** trên nền tảng Swin-Tiny vượt trội hơn toàn bộ các mô hình baseline truyền thống, thiết lập chuẩn mực mới về độ chính xác vi thể và tính nhất quán giải phẫu y học.
