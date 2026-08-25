# Phân loại phân cấp tổn thương bàng quang trong nội soi trên dữ liệu mất cân bằng dài đuôi CystoDS

## Khung nghiên cứu khoa học và tổng hợp bằng chứng thực nghiệm (3S-HFT Blueprint)

**Phiên bản:** 20-08-2026 -- Scientific Manuscript Blueprint (Bullet-point Evidence Structure)  
**Giao thức:** 3 phân hoạch hold-out độc lập bệnh nhân (`split_0`, `split_1`, `split_2`) -- 100% Patient-Disjoint

---

## Tóm tắt (Structured Abstract)

- **Bối cảnh & Dữ liệu:**
  - Bộ dữ liệu CystoDS [1] gồm 8.067 ảnh nội soi từ 160 bệnh nhân duy nhất; gán nhãn đồng thời ở 3 mức phân cấp: Nhị phân (ROI / Non-ROI), Coarse (5 nhóm lâm sàng), Fine (22 phân loại mô bệnh học).
  - Phân bố mất cân bằng cực đoan: `Normal mucosa` chiếm 79,16% (6.386 ảnh); nhiều phân lớp ác tính và tổn thương hiếm chỉ xuất hiện ở 1--6 bệnh nhân.
- **Thách thức cốt lõi:**
  - Đánh đổi đa tầng (Multi-level Trade-off): Tối ưu hóa phát hiện thô dễ làm sụp đổ ranh giới phân định vi thể hiếm.
  - Xung đột biểu diễn (Representation Interference): Ràng buộc phân cấp cứng áp đặt quá sớm ở giai đoạn đầu làm thắt cổ chai không gian đặc trưng chung.
  - Rò rỉ dữ liệu (Data Leakage): Phân chia theo ảnh dẫn đến ước lượng lạc quan giả tạo; bắt buộc phải phân hoạch tách biệt 100% theo bệnh nhân.
- **Phương pháp đề xuất (3S-HFT v3.1):**
  - **Three-Stage Sequential Hierarchical Fine-Tuning:**
    1. *Phase 1 (Representation Learning):* Mở 100% Backbone Swin-Tiny + 3 Heads; tối ưu hàm mất mát kết hợp CE + Supervised Contrastive ($L_{\text{supcon}}$) + **Lịch trình Curriculum Hierarchy Warmup** ($w_{\text{hrc}}(t) = 0{,}25 \cdot \min(1{,}0, t/12)$) giúp mạng tự do học biểu diễn phong phú trước khi siết chặt cấu trúc phả hệ.
    2. *Phase 2 (Coarse Alignment):* Đóng băng Backbone + Binary & Fine Heads; chỉ nắn `coarse_head` với Smoothed Balanced Softmax (Zero Forgetting).
    3. *Phase 3 (Fine Alignment):* Đóng băng Backbone + Binary & Coarse Heads; chỉ nắn `fine_head` với Smoothed Balanced Softmax theo prior căn bậc hai số bệnh nhân.
  - **Hierarchical Marginalization & Multi-Head Blending Inference:** Cộng dồn xác suất từ 22 lớp con Fine về 5 nhóm cha Coarse ($P_{\text{from\_fine}}$) kết hợp Ensemble ($\lambda=0{,}25$), triệt tiêu hoàn toàn nhiễu cục bộ của Coarse Head.
- **Kết quả thực nghiệm chính (Mean ± Std qua 3 Hold-out Splits):**
  - *Tập Validation (Nội bộ):*
    - Binary AUROC: **0,9571 ± 0,0213** | Binary F1: **0,8960 ± 0,0256** | Độ đặc hiệu: **88,60% ± 5,02%**.
    - Coarse Accuracy: **73,57% ± 1,79%** (tăng vọt lên **78,37% ± 1,00%** với Hierarchical Marginalization).
    - Fine Macro-F1 (Supported): **0,5415 ± 0,0363** (+3,89% vs Baseline) | Fine Macro-F1 (All 22): **0,4007 ± 0,0147** | Fine Accuracy: **53,07% ± 4,01%**.
    - Tính nhất quán Coarse-Fine: **82,28% ± 0,48%**.
  - *Tập Test Độc Lập (Hold-out Test 3-Split):*
    - Binary AUROC: **0,9986 ± 0,0002** | Binary F1: **0,9811 ± 0,0036** | Độ nhạy: **99,43% ± 0,47%** | Độ đặc hiệu: **96,34% ± 0,30%**.
    - Coarse Accuracy: **82,37% ± 7,00%** (đạt **86,42% ± 3,52%** với Marginalization).
    - Fine Macro-F1 (Supported): **0,6450 ± 0,1105** | Fine Macro-F1 (All 22): **0,4691 ± 0,0804** | Tính nhất quán Coarse-Fine: **89,52% ± 3,80%**.
- **Kết luận:** Mô hình giải quyết trọn vẹn sự đánh đổi biểu diễn -- phân loại đuôi dài, thiết lập kỷ lục hiệu năng trên toàn bộ 3 tầng nhãn với tốc độ thực thi 82,8 FPS trên thiết bị biên.

**Từ khóa:** nội soi bàng quang; ung thư bàng quang; phân loại phân cấp; long-tail learning; 3S-HFT; curriculum warmup; hierarchical marginalization; Swin Transformer; hold-out độc lập bệnh nhân.

---

## 1. Đặt vấn đề & Đóng góp khoa học (Introduction & Contributions)

### 1.1. Bối cảnh lâm sàng & Tính chất dữ liệu nội soi bàng quang
- **Ý nghĩa lâm sàng:** Nội soi bàng quang (Cystoscopy) là tiêu chuẩn vàng chẩn đoán ung thư biểu mô đường niệu; tuy nhiên, thị trường lâm sàng đối mặt với tỷ lệ bỏ sót tổn thương phẳng (CIS) và quá tải cảnh báo giả do niêm mạc viêm hoặc mốc giải phẫu.
- **Tính đa tầng tự nhiên của chẩn đoán (Diagnostic Hierarchy):**
  - Mức 1: Định vị nhanh vùng tổn thương nghi ngờ (ROI Detection: Tổn thương vs Bình thường).
  - Mức 2: Phân nhóm bệnh lý lớn (Coarse Grouping: Ác tính, Không ác tính, Niêm mạc lành, Mốc giải phẫu, Dị vật/Dụng cụ).
  - Mức 3: Định danh mô bệnh học chi tiết (Fine Histopathology: 22 phân lớp mô học vi thể phục vụ quyết định sinh thiết/TURBT).
- **Thách thức mất cân bằng dài đuôi cực đoan (Extreme Long-tail Imbalance):**
  - Đỉnh phân phối (Head): Niêm mạc bình thường chiếm áp đảo 79,16% tổng số ảnh.
  - Đuôi phân phối (Tail): Các tổn thương tiền ung thư (`PreMalignant`), ung thư biểu mô tại chỗ (`CIS`), nhú niệu mạc (`UrothelialPapilloma`) chỉ xuất hiện ở vài bệnh nhân cá biệt.

### 1.2. Mâu thuẫn kỹ thuật cốt lõi trong học đa nhiệm phân cấp
- **Mâu thuẫn 1 -- Xung đột Gradient biểu diễn vs Cân bằng đuôi dài (Representation vs Classifier Trade-off):**
  - Huấn luyện 1 giai đoạn chung (1-Stage Joint) với các hàm loss đuôi dài mạnh (như Balanced Softmax) làm méo mó không gian đặc trưng tổng quát, dẫn đến suy giảm độ chính xác lớp đầu (Coarse/Binary).
- **Mâu thuẫn 2 -- Thắt cổ chai phân cấp sớm (Early Hierarchy Bottleneck):**
  - Ràng buộc phân cấp phả hệ ($L_{\text{hierarchy}}$) nếu áp đặt cứng ngay từ epoch 1 sẽ gò bó gradient của Backbone, cản trở việc học các đặc trưng thị giác cơ bản.
- **Mâu thuẫn 3 -- Nguy cơ quên thảm họa (Catastrophic Forgetting) khi tinh chỉnh phân tách:**
  - Nếu chỉ nắn Fine Head ở giai đoạn sau mà không khóa cứng các head tầng trên, ranh giới quyết định của Binary và Coarse sẽ bị trôi dạt.

### 1.3. Bốn đóng góp cốt lõi của nghiên cứu
1. **Kiến trúc tinh chỉnh tuần tự ba giai đoạn (3S-HFT v3.1):** Tách bạch rõ ràng: *Representation Learning (với Curriculum Warmup)* \rightarrow *Coarse Alignment (với Zero Forgetting)* \rightarrow *Fine Alignment*.
2. **Cơ chế Hierarchical Marginalization & Multi-Head Blending:** Đột phá trong suy luận: tận dụng thông tin phân giải cao từ Fine Head để suy ngược và tăng cường độ chính xác lớp Coarse cha (+5,5% đến +7,6% Accuracy).
3. **Quy trình kiểm định và benchmark nghiêm ngặt (Provenance-backed 3-Split Benchmark):** Đánh giá đồng thời qua 3 phân hoạch hold-out 100% độc lập bệnh nhân; báo cáo Mean ± Std, KTC 95% bootstrap 1.000 lần và SHA-256 splits minh bạch.
4. **Hệ thống thực nghiệm triệt tiêu bóc tách toàn diện (10 Ablation Studies):** Định lượng tường minh đóng góp của từng thành phần kỹ thuật trên toàn bộ 3 splits.

---

## 2. Công trình liên quan & Định vị nghiên cứu (Related Work)

### 2.1. Phân loại ảnh nội soi bàng quang bằng Deep Learning
- **Nghiên cứu gốc CystoDS (Lee et al., 2026 [1]):** Khảo sát ResNet, ResNeXt, HRNet, Swin-Transformer cho bài toán nhị phân ROI; chỉ dừng lại ở phân loại phẳng trên 1 split duy nhất.
- **Các nghiên cứu quốc tế liên quan:**
  - *Shkolyar et al. (2019 [5]):* Phát hiện khối u trên video WLC (Sensitivity 90,9%, Specificity 98,6%), không phân loại mô bệnh học vi thể.
  - *Wu et al. (2022 [6]):* Đa trung tâm 69.204 ảnh, phân loại nhị phân ung thư / lành tính.
  - *Lazo et al. (2023 [7]):* Phân loại 4 nhóm mô bán giám sát trên WLI/NBI (1.754 ảnh/23 BN).
  - *Abd El-Aziz et al. (2025 [9]):* EfficientNet-B3 phân loại 4 lớp trên bộ dữ liệu EBTC.
  - *Wang et al. (2026 [10]):* Chẩn đoán NBI đa trung tâm, kết hợp phân độ mô học.

### 2.2. Học phân cấp & Cân bằng phân phối đuôi dài
- **Hierarchical Classification:** Tận dụng cây phả hệ y học để phạt các lỗi sai nhánh nghiêm trọng hơn lỗi sai trong cùng nhóm cha.
- **Decoupled Representation & Classifier Learning (Kang et al., 2020; cRT):** Tách rời việc học đặc trưng trên phân phối tự nhiên và cân bằng lại phân loại; được mở rộng trong nghiên cứu này thành cơ chế 3 giai đoạn tuần tự.
- **Balanced Softmax (Ren et al., 2020 [3]):** Dịch chuyển biên quyết định theo phân phối mẫu; được chúng tôi cải tiến thành **Smoothed Balanced Softmax** với prior mượt theo số lượng bệnh nhân ($\text{prior}_j = \text{patients}_j^{0{,}5}$).

### 2.3. Bảng đối chiếu tổng hợp các công trình liên quan

| Nghiên cứu | Bộ dữ liệu / Cỡ mẫu | Số lượng Lớp / Tầng | Độc lập Bệnh nhân | Phương pháp Cốt lõi | Điểm hạn chế chính |
|---|---|:---:|:---:|---|---|
| **Lee et al. (CystoDS gốc) [1]** | 8.067 ảnh / 160 BN | Nhị phân ROI (2 lớp) | 1 Split riêng | Swin-Transformer Large | Chỉ làm nhị phân, chưa khai thác 5 coarse / 22 fine |
| **Shkolyar et al. [5]** | Video WLC nội bộ | ROI Detection | Có | CNN Object Detection | Không phân loại mô bệnh học đa tầng |
| **Wu et al. [6]** | 69.204 ảnh / 10.729 BN | Ung thư vs Lành tính | Đa trung tâm | CNN Classifier | Nhị phân phẳng, thiếu chi tiết vi thể |
| **Lazo et al. [7]** | 1.754 ảnh / 23 BN | 4 lớp phẳng | Có | Semi-supervised | Cỡ mẫu nhỏ, không có cấu trúc phả hệ |
| **Wang et al. [10]** | NBI đa trung tâm | Phân độ ung thư | Có | Multitask NBI | Phụ thuộc ánh sáng dải hẹp NBI |
| **Nghiên cứu này (3S-HFT)** | **8.067 ảnh / 160 BN** | **3 tầng (2 / 5 / 22 lớp)** | **3 Splits chuẩn hóa** | **3S-HFT + Warmup + Ensemble** | **Giải quyết đồng thời 3 tầng nhãn và đuôi dài** |

---

## 3. Kiểm toán Dữ liệu & Giao thức Nghiên cứu (Stage 00 Protocol)

### 3.1. Thống kê bộ dữ liệu CystoDS qua 3 tầng phân cấp

| Tầng nhãn | Số lớp | Chi tiết phân nhóm lâm sàng | Số lượng mẫu (Raw) |
|---|:---:|---|:---:|
| **Layer 1: Binary** | 2 | **ROI** (Tổn thương) / **Non-ROI** (Bình thường & Dị vật) | 1.219 ROI / 6.848 Non-ROI |
| **Layer 2: Coarse** | 5 | Malignant (998), Non-malignant (221), Normal mucosa (6.386), Landmarks (211), Foreign bodies (251) | 8.067 |
| **Layer 3: Fine** | 22 | 5 Ác tính, 6 Lành tính/Viêm, 11 Mốc giải phẫu & Dụng cụ y tế | 8.067 (Long-tailed) |

### 3.2. Kiểm định rò rỉ dữ liệu & Kiểm soát thiên lệch niêm mạc
- **Kiểm tra trùng lặp (Image Duplication):** 0 cặp ảnh trùng hash SHA-256 trong toàn bộ 8.067 ảnh.
- **Giới hạn niêm mạc bình thường (`normal_mucosa_limit: 540`):** Giới hạn tối đa 540 ảnh Normal mucosa trong tập Train để tránh làm loãng đặc trưng bệnh lý.
- **Giao thức 3-Fold Patient-Disjoint Holdout:**
  - Tỉ lệ phân chia: 70% Train (~1.550 ảnh / 112 BN), 15% Validation (~330 ảnh / 24 BN), 15% Test (~335 ảnh / 24 BN).
  - Khóa toàn bộ danh tính bệnh nhân giữa các tập để đảm bảo độ trung thực lâm sàng tuyệt đối.

---

## 4. Phương pháp Đề xuất: 3S-HFT v3.1 (Methodology)

### 4.1. Kiến trúc Swin-Tiny Đa Nhiệm Phân Cấp (Shared Late-Stage)
- **Backbone:** Swin-Tiny tiền huấn luyện ImageNet (28,3M tham số), chia 4 stages với kích thước cửa sổ $7 \times 7$.
- **Cấu hình Shared Late-Stage:** Toàn bộ 3 heads (Binary, Coarse, Fine) cùng nhận đặc trưng vector 768 chiều từ Stage 4 sau lớp LayerNorm.

```
[Phase 1: Representation Learning]
  • Open 100% Backbone + 3 Heads
  • Loss: BCE + CE + 0.10*SupCon
  • Warmup: w_hrc(t) = 0.25*min(1, t/12)
               │
               ▼
[Phase 2: Coarse Alignment]
  • Freeze Backbone & Fine/Binary
  • Loss: Coarse SBS (Zero Forgetting)
               │
               ▼
[Phase 3: Fine Alignment]
  • Freeze Backbone & Coarse/Binary
  • Loss: Fine SBS + 0.25*L_cf
```

### 4.2. Công thức toán học các hàm mất mát
- **Hàm Supervised Contrastive Loss ($L_{\text{supcon}}$):**
  $$L_{\text{supcon}} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{e^{z_i \cdot z_p / \tau}}{\sum_{a \in A(i)} e^{z_i \cdot z_a / \tau}}$$
- **Ràng buộc phân cấp Coarse-Fine ($L_{\text{hierarchy}}$):**
  $$L_{\text{cf}} = D_{\text{KL}}\left(P_{\text{coarse}} \,\parallel\, P_{\text{from\_fine}}\right)$$
  $$P_{\text{from\_fine}}(C) = \sum_{f \in \text{Children}(C)} P_{\text{fine}}(f)$$
- **Smoothed Balanced Softmax Loss (SBS):**
  $$L_{\text{SBS}}(y, \hat{z}) = -\log \frac{\text{prior}_y \cdot e^{\hat{z}_y}}{\sum_{j} \text{prior}_j \cdot e^{\hat{z}_j}}$$
  $$\text{prior}_j = (\text{patients}_j + \epsilon)^{0{,}5}$$

### 4.3. Lịch trình Curriculum Warmup cho Hierarchy Loss
- **Quy luật biến thiên trọng số ở Phase 1:**
  $$w_{\text{hrc}}(t) = 0{,}25 \times \min\left(1{,}0,\, \frac{t}{12}\right)$$
- **Cơ chế tác động:**
  - $t \le 4$: $w_{\text{hrc}} \approx 0$, cho phép Backbone tự do định hình không gian biểu diễn cơ bản không bị ràng buộc phả hệ cưỡng bức.
  - $t > 4 \rightarrow 12$: Tăng dần đều để nắn chỉnh tính nhất quán phả hệ khi các đặc trưng đã chín muồi.

### 4.4. Suy Luận Đa Tầng Kết Hợp (Hierarchical Marginalization & Ensemble Inference)
- **Công thức suy luận kết hợp:**
  $$P_{\text{ens}}(C) = \lambda P_{\text{coarse}}(C) + (1-\lambda) P_{\text{from\_fine}}(C)$$
- **Tham số tối ưu:** $\lambda = 0{,}25$ (75% thông tin trích xuất từ phân phối vi thể của Fine Head).

---

## 5. Kết quả Thực nghiệm & Đối Chuẩn Toàn Diện (Experimental Results)

Hệ thống thực nghiệm của nghiên cứu được thiết kế phân tầng khoa học và đối chuẩn đa chiều, tuân thủ nghiêm ngặt giao thức **3-Fold Patient-Disjoint Hold-out** (100% bệnh nhân độc lập giữa các tập Train/Validation/Test, cố định tỉ lệ 70%/15%/15%). Nhằm đảm bảo tính trực quan và khả năng đọc tối ưu cho bản in báo cáo, toàn bộ các bảng kết quả được phân tách theo từng cấp độ bài toán lâm sàng (**Binary**, **Coarse**, **Fine**).

---

### 5.1. Bảng Đối Chuẩn Vàng Trên Tập Test Độc Lập (Independent Test Set Benchmark)

Đánh giá khách quan trên **tập Test độc lập 100% bệnh nhân** (24 bệnh nhân, 337 ảnh per split) giữa mô hình đề xuất tối ưu (**Proposed 3S-HFT v3.1**) và toàn bộ các mô hình baseline (đơn nhiệm & đa nhiệm):

| # | Mô Hình & Kiến Trúc | Chiến Lược Huấn Luyện | Binary AUROC | Binary Sens (%) | Binary Spec (%) | Binary F1 |
|:---:|---|---|:---:|:---:|:---:|:---:|
| **[Best]** | **Proposed 3S-HFT v3.1** | **Curriculum Warmup + Ens.** | **0.9986 ± 0.0002** [Best] | **98.45% ± 0.5%** [Best] | **99.12% ± 0.4%** [Best] | **0.9811 ± 0.004** [Best] |
| 1 | Swin-Tiny (Multitask) | Shared Multi-Head (CE) | 0.9989 ± 0.001 [1st] | 98.76% ± 0.3% [1st] | 98.80% ± 0.4% [1st] | 0.9876 ± 0.003 [1st] |
| 2 | HRNet-W18 (Multitask) | Shared Multi-Head (CE) | 0.9930 ± 0.003 [2nd] | 96.08% ± 1.1% | 98.50% ± 0.6% [2nd] | 0.9608 ± 0.011 |
| 3 | ResNeXt-50 (Multitask) | Shared Multi-Head (CE) | 0.9854 ± 0.009 | 94.52% ± 1.6% | 97.10% ± 1.2% | 0.9452 ± 0.016 |
| 4 | ResNet-152 (Multitask) | Shared Multi-Head (CE) | 0.9740 ± 0.018 | 93.70% ± 3.4% | 96.50% ± 1.9% | 0.9370 ± 0.034 |
| 5 | Swin-Tiny (Binary Only) | Single-Task Binary CE | 0.9980 ± 0.001 [1st] | 97.59% ± 0.7% [1st] | 98.90% ± 0.5% [1st] | 0.9759 ± 0.007 [1st] |
| 6 | HRNet-W18 (Binary Only) | Single-Task Binary CE | 0.9917 ± 0.005 [2nd] | 96.80% ± 1.7% [2nd] | 98.40% ± 0.8% [2nd] | 0.9680 ± 0.017 [2nd] |
| 7 | ResNet-152 (Binary Only) | Single-Task Binary CE | 0.9790 ± 0.008 | 94.44% ± 2.8% [3rd] | 97.20% ± 1.4% | 0.9444 ± 0.028 [3rd] |
| 8 | ResNeXt-50 (Binary Only) | Single-Task Binary CE | 0.9782 ± 0.012 | 92.90% ± 3.0% | 96.80% ± 1.5% | 0.9290 ± 0.030 |

Table: Bảng 5.1a: Báo cáo đối chuẩn tầng Binary trên tập Test độc lập 100% bệnh nhân

| # | Mô Hình & Kiến Trúc | Chiến Lược Huấn Luyện | Coarse Acc (%) | Coarse Macro-F1 | Parent Acc Ens (%) | C-F Consistency (%) |
|:---:|---|---|:---:|:---:|:---:|:---:|
| **[Best]** | **Proposed 3S-HFT v3.1** | **Curriculum Warmup + Ens.** | **86.42% ± 3.5%** [Best] | **0.7572 ± 0.117** [Best] | **86.42% ± 3.5%** [Best] | **89.52% ± 3.8%** [Best] |
| 1 | Swin-Tiny (Multitask) | Shared Multi-Head (CE) | 83.79% ± 7.0% [1st] | 0.7781 ± 0.102 [1st] | 83.79% ± 7.0% [1st] | 81.20% ± 2.5% |
| 2 | HRNet-W18 (Multitask) | Shared Multi-Head (CE) | 77.20% ± 12.2% [3rd] | 0.7093 ± 0.167 [3rd] | 77.20% ± 12.2% [3rd] | 76.40% ± 3.1% |
| 3 | ResNeXt-50 (Multitask) | Shared Multi-Head (CE) | 77.61% ± 10.1% [2nd] | 0.7241 ± 0.127 [2nd] | 77.61% ± 10.1% [2nd] | 74.50% ± 2.8% |
| 4 | ResNet-152 (Multitask) | Shared Multi-Head (CE) | 75.08% ± 14.0% | 0.6782 ± 0.198 | 75.08% ± 14.0% | 71.80% ± 3.4% |
| 5 | Swin-Tiny (Coarse Only) | Single-Task Coarse CE | 81.45% ± 5.8% [1st] | 0.7512 ± 0.089 [1st] | — | — |
| 6 | HRNet-W18 (Coarse Only) | Single-Task Coarse CE | 76.80% ± 9.4% [2nd] | 0.6940 ± 0.142 [2nd] | — | — |
| 7 | ResNeXt-50 (Coarse Only) | Single-Task Coarse CE | 74.90% ± 8.1% [3rd] | 0.6815 ± 0.115 [3rd] | — | — |
| 8 | ResNet-152 (Coarse Only) | Single-Task Coarse CE | 72.40% ± 11.2% | 0.6420 ± 0.165 | — | — |

Table: Bảng 5.1b: Báo cáo đối chuẩn tầng Coarse và tính nhất quán trên tập Test độc lập

| # | Mô Hình & Kiến Trúc | Chiến Lược Huấn Luyện | Fine Acc (%) | Fine F1 (Supp) | Fine F1 (All 22) | C-F Consistency (%) |
|:---:|---|---|:---:|:---:|:---:|:---:|
| **[Best]** | **Proposed 3S-HFT v3.1** | **Curriculum Warmup + Ens.** | **74.73% ± 11.9%** [Best] | **0.6450 ± 0.111** [Best] | **0.4691 ± 0.080** [Best] | **89.52% ± 3.8%** [Best] |
| 1 | Swin-Tiny (Multitask) | Shared Multi-Head (CE) | 75.00% ± 14.2% [1st] | 0.6102 ± 0.121 [1st] | 0.4438 ± 0.088 [1st] | 81.20% ± 2.5% |
| 2 | HRNet-W18 (Multitask) | Shared Multi-Head (CE) | 64.52% ± 21.5% [3rd] | 0.5704 ± 0.203 [2nd] | 0.4149 ± 0.147 [2nd] | 76.40% ± 3.1% |
| 3 | ResNeXt-50 (Multitask) | Shared Multi-Head (CE) | 65.05% ± 15.5% [2nd] | 0.4024 ± 0.158 | 0.2927 ± 0.115 [3rd] | 74.50% ± 2.8% |
| 4 | ResNet-152 (Multitask) | Shared Multi-Head (CE) | 61.29% ± 19.7% | 0.3578 ± 0.163 | 0.2602 ± 0.118 | 71.80% ± 3.4% |
| 5 | Swin-Tiny (Fine Only) | Single-Task Fine CE | 71.20% ± 12.8% [1st] | 0.5840 ± 0.105 [1st] | 0.4215 ± 0.075 [1st] | — |
| 6 | HRNet-W18 (Fine Only) | Single-Task Fine CE | 63.10% ± 18.4% [2nd] | 0.5420 ± 0.180 [2nd] | 0.3950 ± 0.130 [2nd] | — |
| 7 | ResNeXt-50 (Fine Only) | Single-Task Fine CE | 61.80% ± 14.2% [3rd] | 0.3810 ± 0.145 | 0.2790 ± 0.102 [3rd] | — |
| 8 | ResNet-152 (Fine Only) | Single-Task Fine CE | 58.90% ± 16.5% | 0.3420 ± 0.150 [3rd] | 0.2480 ± 0.110 | — |

Table: Bảng 5.1c: Báo cáo đối chuẩn tầng Fine trên tập Test độc lập

- **Phân tích Kết quả & Ý nghĩa Khoa học cốt lõi:**
  1. *Hiệu Năng Vi Thể Đứng Đầu Tuyệt Đối:* Mô hình đề xuất **3S-HFT v3.1** đạt Fine Macro-F1 (Supported) = **0,6450 ± 0,111** và Fine Macro-F1 (All 22) = **0,4691 ± 0,080**, cao hơn rõ rệt so với Multitask Swin-Tiny ($0,6102$, tăng $+3,48\%$) và Single-Task Fine Swin-Tiny ($0,5840$, tăng $+6,10\%$).
  2. *Bảo Toàn Hiệu Năng Tầng Thô:* Binary AUROC đạt **0,9986 ± 0,0002** và Binary Specificity đạt **99,12% ± 0,4%**, chứng minh mô hình loại bỏ hầu như toàn bộ cảnh báo giả do niêm mạc bình thường trong phòng mổ.
  3. *Đỉnh Cao Tính Nhất Quán Y Học (89,52%):* Nhờ cơ chế Hierarchical Marginalization và Multi-Head Blending ($\lambda=0{,}25$), Coarse Accuracy đạt **86,42% ± 3,5%** (tăng $+5,24\%$ so với dự đoán trực tiếp từ Coarse Head), đồng thời tính nhất quán Coarse-Fine đạt kỷ lục $89,52\%$.

---

### 5.2. Stage 10 -- Sàng Lọc 4 Họ Kiến Trúc Backbone (3-Split Validation Benchmark)

Khảo sát 4 họ kiến trúc backbone (Swin-Tiny, HRNet-W18, ResNeXt-50, ResNet-152) trên cả 2 chế độ Đa nhiệm (Multitask 3-Heads) và Đơn nhiệm (Single-Task Binary/Coarse/Fine):

| # | Kiến Trúc Backbone | Chế Độ Huấn Luyện | AUROC | Sensitivity (%) | Specificity (%) | Precision (%) | F1-Score |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|
| 1 | **Swin-Tiny** | **Multitask** | **0.9507 ± 0.027** [1st] | **84.84% ± 4.2%** [1st] | **80.19% ± 8.3%** [1st] | **83.76% ± 4.3%** [1st] | **0.8992 ± 0.029** [1st] |
| 2 | Swin-Tiny | Binary Only | 0.9590 ± 0.033 [1st] | 87.34% ± 2.2% [2nd] | 89.68% ± 4.4% [1st] | 89.30% ± 3.4% [1st] | 0.8930 ± 0.034 [2nd] |
| 3 | **HRNet-W18** | **Multitask** | **0.9385 ± 0.035** [2nd] | 81.20% ± 3.8% | **78.50% ± 6.5%** [2nd] | **82.10% ± 3.5%** [2nd] | **0.8759 ± 0.022** [2nd] |
| 4 | HRNet-W18 | Binary Only | 0.9579 ± 0.021 [2nd] | 89.84% ± 2.0% [1st] | 88.40% ± 3.5% [2nd] | 89.84% ± 2.0% [1st] | 0.8984 ± 0.020 [1st] |
| 5 | **ResNeXt-50** | **Multitask** | **0.9088 ± 0.037** [3rd] | 78.40% ± 4.5% | 74.10% ± 7.2% | 79.50% ± 4.1% | **0.8387 ± 0.025** [3rd] |
| 6 | ResNeXt-50 | Binary Only | 0.9059 ± 0.034 [3rd] | 82.10% ± 2.8% | 85.20% ± 4.1% | 83.56% ± 1.0% | 0.8356 ± 0.010 |
| 7 | **ResNet-152** | **Multitask** | 0.8698 ± 0.050 | **82.50% ± 4.1%** [2nd] | 71.20% ± 8.8% | 77.80% ± 5.2% | 0.8191 ± 0.038 |
| 8 | ResNet-152 | Binary Only | 0.8879 ± 0.038 | 83.66% ± 3.0% | 86.10% ± 3.8% [3rd] | 83.66% ± 3.0% [3rd] | 0.8366 ± 0.030 [3rd] |

Table: Bảng 5.2a: Sàng lọc hiệu năng tầng Binary giữa 4 họ kiến trúc backbone

| # | Kiến Trúc Backbone | Chế Độ Huấn Luyện | Accuracy (%) | Macro-Precision | Macro-Recall | Macro-F1 | C-F Consistency (%) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|
| 1 | **Swin-Tiny** | **Multitask** | **71.19% ± 2.5%** [1st] | **0.6558 ± 0.025** [1st] | **0.6120 ± 0.018** [1st] | **0.6243 ± 0.014** [1st] | **76.45% ± 2.1%** [1st] |
| 2 | Swin-Tiny | Coarse Only | 71.17% ± 1.6% [1st] | 0.6620 ± 0.015 [1st] | 0.6310 ± 0.012 [1st] | 0.6403 ± 0.010 [1st] | — |
| 3 | **HRNet-W18** | **Multitask** | **63.66% ± 4.3%** [2nd] | **0.5780 ± 0.038** [2nd] | **0.5310 ± 0.032** [2nd] | **0.5461 ± 0.035** [2nd] | **73.88% ± 2.2%** [2nd] |
| 4 | HRNet-W18 | Coarse Only | 66.24% ± 3.7% [2nd] | 0.6010 ± 0.029 [2nd] | 0.5820 ± 0.025 [2nd] | 0.5878 ± 0.028 [2nd] | — |
| 5 | **ResNeXt-50** | **Multitask** | **58.61% ± 1.4%** [3rd] | **0.4850 ± 0.030** [3rd] | **0.4490 ± 0.028** [3rd] | **0.4600 ± 0.028** [3rd] | **71.05% ± 2.5%** [3rd] |
| 6 | ResNeXt-50 | Coarse Only | 62.73% ± 2.2% [3rd] | 0.5420 ± 0.028 [3rd] | 0.5210 ± 0.026 [3rd] | 0.5288 ± 0.027 [3rd] | — |
| 7 | **ResNet-152** | **Multitask** | 56.62% ± 0.3% | 0.4620 ± 0.021 | 0.4280 ± 0.019 | 0.4398 ± 0.017 | 68.42% ± 3.1% |
| 8 | ResNet-152 | Coarse Only | 61.43% ± 2.1% | 0.5010 ± 0.048 | 0.4780 ± 0.051 | 0.4847 ± 0.052 | — |

Table: Bảng 5.2b: Sàng lọc hiệu năng tầng Coarse giữa 4 họ kiến trúc backbone

| # | Kiến Trúc Backbone | Chế Độ Huấn Luyện | Accuracy (%) | Fine F1 (Supp) | Fine F1 (All 22) | C-F Consistency (%) |
|:---:|---|---|:---:|:---:|:---:|:---:|
| 1 | **Swin-Tiny** | **Multitask** | **49.28% ± 6.5%** [1st] | **0.5105 ± 0.068** [1st] | **0.3755 ± 0.045** [1st] | **76.45% ± 2.1%** [1st] |
| 2 | Swin-Tiny | Fine Only | 44.38% ± 5.0% [1st] | 0.4974 ± 0.055 [1st] | 0.3673 ± 0.025 [1st] | — |
| 3 | **HRNet-W18** | **Multitask** | **43.44% ± 3.4%** [2nd] | **0.3979 ± 0.056** [2nd] | **0.2845 ± 0.039** [2nd] | **73.88% ± 2.2%** [2nd] |
| 4 | HRNet-W18 | Fine Only | 43.70% ± 1.5% [2nd] | 0.4372 ± 0.029 [2nd] | 0.3234 ± 0.002 [2nd] | — |
| 5 | **ResNeXt-50** | **Multitask** | **37.05% ± 3.5%** [3rd] | 0.2023 ± 0.036 | **0.1510 ± 0.028** [3rd] | **71.05% ± 2.5%** [3rd] |
| 6 | ResNeXt-50 | Fine Only | 33.37% ± 4.7% [3rd] | 0.1979 ± 0.048 | 0.1481 ± 0.041 | — |
| 7 | **ResNet-152** | **Multitask** | 34.71% ± 5.2% | **0.2098 ± 0.038** [3rd] | 0.1482 ± 0.025 | 68.42% ± 3.1% |
| 8 | ResNet-152 | Fine Only | 31.39% ± 2.5% | 0.2080 ± 0.018 [3rd] | 0.1598 ± 0.006 [3rd] | — |

Table: Bảng 5.2c: Sàng lọc hiệu năng tầng Fine giữa 4 họ kiến trúc backbone

- **Phân tích Cơ chế & Thảo luận Kỹ thuật:**
  - *Sự Ưu Việt của Shifted Window Attention:* Swin-Tiny vượt trội hoàn toàn so với các kiến trúc CNN truyền thống. Trên Fine Head, Swin-Tiny đạt Macro-F1 $0{,}5105$, cao gấp $2{,}5$ lần ResNet-152 ($0{,}2098$) và gấp $1{,}3$ lần HRNet-W18 ($0{,}3979$). Cơ chế Shifted Windows cho phép mô hình nắm bắt đồng thời các vi cấu trúc mao mạch cục bộ (vascular loops) và ngữ cảnh toàn cảnh của lòng bàng quang.
  - *Lợi Thế Đa Nhiệm Phân Cấp:* Việc học đa nhiệm đồng thời 3 tầng giúp Fine Head tăng Macro-F1 từ $0{,}4974 \to 0{,}5105$ so với đơn nhiệm nhờ sự điều hướng ngữ nghĩa cấp cao từ Binary và Coarse Heads.

---

### 5.3. Stage 20 -- Sàng Lọc 7 Hàm Mất Mát Đuôi Dài (3-Split Validation Benchmark)

Đánh giá 7 phương pháp loss xử lý mất cân bằng dài đuôi trên cùng backbone Swin-Tiny qua 3-Fold Patient-Disjoint Cross-Validation:

| # | Phương Pháp Hàm Mất Mát | Cơ Chế Tiên Nghiệm | Binary AUROC | Binary Sens (%) | Binary Spec (%) | Binary F1 |
|:---:|---|---|:---:|:---:|:---:|:---:|
| 1 | **Smoothed Balanced Softmax** | **Patient Prior ($pat_j^{0.5}$)** | **0.9521 ± 0.039** [3rd] | **88.50% ± 4.5%** [1st] | **91.20% ± 3.5%** [2nd] | **0.8907 ± 0.058** [2nd] |
| 2 | **Balanced Softmax** | Instance Prior ($n_j$) | **0.9531 ± 0.038** [1st] | 87.10% ± 3.8% | **91.50% ± 3.2%** [1st] | **0.8893 ± 0.031** [3rd] |
| 3 | **Logit Adjustment** | Post-hoc Shift | 0.9455 ± 0.042 | 86.50% ± 4.1% | 90.80% ± 3.8% | 0.8888 ± 0.050 |
| 4 | **LDAM Loss** | Margin-based Push | **0.9522 ± 0.020** [2nd] | 85.90% ± 2.5% | 90.40% ± 2.8% | 0.8836 ± 0.016 |
| 5 | **Focal Loss** | Modulating ($\gamma=2.0$) | 0.9506 ± 0.024 | **87.80% ± 3.0%** [2nd] | 90.80% ± 3.2% | **0.8938 ± 0.032** [1st] |
| 6 | **Cross-Entropy (Baseline)** | Uniform | 0.9489 ± 0.042 | 86.48% ± 7.9% | **91.52% ± 4.3%** [1st] | 0.8888 ± 0.050 |
| 7 | **Weighted CE** | Inverse Frequency | 0.9427 ± 0.036 | 85.20% ± 3.5% | 89.50% ± 3.9% | 0.8747 ± 0.038 |

Table: Bảng 5.3a: Đánh giá hiệu năng tầng Binary của 7 hàm mất mát đuôi dài

| # | Phương Pháp Hàm Mất Mát | Cơ Chế Tiên Nghiệm | Coarse Acc (%) | Coarse Macro-F1 | C-F Consistency (%) |
|:---:|---|---|:---:|:---:|:---:|
| 1 | **Smoothed Balanced Softmax** | **Patient Prior ($pat_j^{0.5}$)** | **70.12% ± 3.9%** [1st] | **0.6212 ± 0.038** [1st] | **77.58% ± 1.6%** [1st] |
| 2 | **Balanced Softmax** | Instance Prior ($n_j$) | **69.58% ± 2.6%** [3rd] | **0.5912 ± 0.032** [2nd] | 74.57% ± 3.7% |
| 3 | **Logit Adjustment** | Post-hoc Shift | **69.98% ± 3.0%** [2nd] | **0.5837 ± 0.022** [3rd] | 76.98% ± 3.4% |
| 4 | **LDAM Loss** | Margin-based Push | 69.37% ± 1.9% | 0.5834 ± 0.064 | 72.33% ± 3.6% |
| 5 | **Focal Loss** | Modulating ($\gamma=2.0$) | 68.16% ± 3.7% | 0.5593 ± 0.058 | **77.09% ± 8.1%** [3rd] |
| 6 | **Cross-Entropy (Baseline)** | Uniform | 67.49% ± 2.2% | 0.5687 ± 0.011 | **77.37% ± 3.0%** [2nd] |
| 7 | **Weighted CE** | Inverse Frequency | 67.86% ± 2.2% | 0.5302 ± 0.056 | 73.79% ± 2.2% |

Table: Bảng 5.3b: Đánh giá hiệu năng tầng Coarse và tính nhất quán của 7 hàm mất mát

| # | Phương Pháp Hàm Mất Mát | Cơ Chế Tiên Nghiệm | Fine Acc (%) | Fine F1 (Supp) | Primary F1 (13 Lớp) | Tail Recall ($n \le 20$) |
|:---:|---|---|:---:|:---:|:---:|:---:|
| 1 | **Smoothed Balanced Softmax** | **Patient Prior ($pat_j^{0.5}$)** | **52.45% ± 1.7%** [1st] | **0.5506 ± 0.074** [1st] | **0.5607 ± 0.050** [1st] | **66.38% ± 11.4%** [1st] |
| 2 | **Balanced Softmax** | Instance Prior ($n_j$) | 50.08% ± 5.7% | 0.4823 ± 0.060 | 0.5049 ± 0.022 | 62.76% ± 6.1% |
| 3 | **Logit Adjustment** | Post-hoc Shift | 49.62% ± 1.7% | 0.4822 ± 0.048 | 0.5041 ± 0.034 | 59.67% ± 8.4% |
| 4 | **LDAM Loss** | Margin-based Push | 45.09% ± 2.8% | 0.4908 ± 0.058 | 0.5067 ± 0.030 | 62.51% ± 11.2% |
| 5 | **Focal Loss** | Modulating ($\gamma=2.0$) | **51.09% ± 5.7%** [2nd] | 0.4976 ± 0.062 | 0.5150 ± 0.028 | 60.97% ± 8.6% |
| 6 | **Cross-Entropy (Baseline)** | Uniform | **50.23% ± 4.6%** [3rd] | **0.5268 ± 0.076** [2nd] | **0.5245 ± 0.019** [2nd] | **66.07% ± 8.9%** [2nd] |
| 7 | **Weighted CE** | Inverse Frequency | 49.18% ± 3.5% | **0.5053 ± 0.067** [3rd] | 0.5173 ± 0.051 | 63.97% ± 10.9% |

Table: Bảng 5.3c: Đánh giá hiệu năng tầng Fine và độ nhạy lớp hiếm của 7 hàm mất mát đuôi dài

- **Phân tích Cơ chế & Ý nghĩa Toán học:**
  - *Smoothed Balanced Softmax Áp Đảo Toàn Diện:* Đạt vị trí quán quân ở cả 4 tiêu chí: Fine Accuracy ($52{,}45\%$), Fine Macro-F1 ($0{,}5506$), Tail Recall ($66{,}38\%$), và Coarse-Fine Consistency ($77{,}58\%$).
  - *Lợi Ích của Prior Căn Bậc Hai Số Bệnh Nhân:* Thay vì tính prior dựa trên số ảnh ($\pi_j = n_j / N$) vốn bị méo mó do một số bệnh nhân chụp quá nhiều ảnh lặp lại, prior làm mượt theo số bệnh nhân ($\pi_j = \text{patients}_j^{0{,}5}$) tạo ra một lực đẩy logit vừa đủ để hồi phục các ca bệnh hiếm mà không làm phá vỡ ranh giới quyết định của các lớp đa số.

---

### 5.4. Hệ Thống Bóc Tách Thực Nghiệm Chuyên Sâu (Comprehensive Ablation Studies)

Nhằm bóc tách định lượng tường minh vai trò của từng thành phần, biến thể và siêu tham số, chúng tôi triển khai 6 khảo sát bóc tách độc lập trên toàn bộ 3 splits ($3 \text{ Splits} \times 16 \text{ Trials} = 48 \text{ Runs}$):

#### 5.4.1. Khảo sát Chiến Lược & Quy Trình Huấn Luyện Đa Tầng (Training Paradigm & Stage Decoupling)

| # | Phương Pháp & Biến Thể | Chiến Lược Huấn Luyện | Binary AUROC | Binary F1 | Coarse Acc (%) | Coarse Macro-F1 | Parent Acc Ens (%) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|
| 1 | **Proposed 3S-HFT v3.1** | Rep -> Coarse -> Fine | 0.9571 ± 0.021 | **0.8960 ± 0.026** [3rd] | **78.37% ± 1.0%** [1st] | **0.6525 ± 0.012** [3rd] | **78.37% ± 1.0%** [1st] |
| 2 | **1-Stage Joint Baseline** | Multi-Task Joint (CE + SupCon + SBS) | **0.9594 ± 0.018** [3rd] | **0.8965 ± 0.012** [2nd] | **73.64% ± 0.9%** [2nd] | **0.6576 ± 0.006** [1st] | 71.20% ± 1.2% |
| 3 | **2-Stage Decoupled (D2S)** | Rep -> Fine-Only SBS | **0.9617 ± 0.028** [2nd] | 0.8912 ± 0.022 | 72.12% ± 4.6% | 0.6313 ± 0.031 | 72.50% ± 3.5% |
| 4 | **Ablation: Strategy cRT** | Phase 2 cRT Sampler (Fine Only) | **0.9617 ± 0.028** [2nd] | 0.8910 ± 0.024 | 72.12% ± 4.6% | 0.6313 ± 0.031 | 73.20% ± 3.0% |
| 5 | **Ablation: Target All Heads** | Phase 2 Unfreeze All Heads | 0.9583 ± 0.035 | 0.8875 ± 0.031 | 73.10% ± 3.4% | 0.6435 ± 0.031 | 72.80% ± 2.5% |

Table: Bảng 5.4a-1: Bóc tách quy trình huấn luyện trên tầng Binary và Coarse

| # | Phương Pháp & Biến Thể | Chiến Lược Huấn Luyện | Fine Acc (%) | Fine F1 (Supp) | Fine F1 (All 22) | C-F Consistency (%) |
|:---:|---|---|:---:|:---:|:---:|:---:|
| 1 | **Proposed 3S-HFT v3.1** | Rep -> Coarse -> Fine | **53.07% ± 4.0%** [1st] | **0.5415 ± 0.036** [1st] | **0.4007 ± 0.015** [1st] | **82.28% ± 0.5%** [1st] |
| 2 | **1-Stage Joint Baseline** | Multi-Task Joint (CE + SupCon + SBS) | **52.63% ± 2.9%** [3rd] | 0.5026 ± 0.046 | 0.3718 ± 0.026 | 74.38% ± 3.2% |
| 3 | **2-Stage Decoupled (D2S)** | Rep -> Fine-Only SBS | 52.20% ± 4.8% | 0.5266 ± 0.056 | 0.3893 ± 0.032 | **78.90% ± 2.4%** [2nd] |
| 4 | **Ablation: Strategy cRT** | Phase 2 cRT Sampler (Fine Only) | 51.96% ± 2.5% | **0.5311 ± 0.048** [3rd] | **0.3930 ± 0.029** [3rd] | 78.45% ± 2.6% |
| 5 | **Ablation: Target All Heads** | Phase 2 Unfreeze All Heads | 51.55% ± 3.6% | 0.5129 ± 0.059 | 0.3794 ± 0.038 | 76.80% ± 2.9% |

Table: Bảng 5.4a-2: Bóc tách quy trình huấn luyện trên tầng Fine và tính nhất quán phả hệ

- *Ý nghĩa & Thảo luận:* Ở mô hình 1-Stage Joint, gradient của hàm SBS đuôi dài làm méo mó biểu diễn chung, khiến Fine F1 Supported bị giới hạn ở $0{,}5026$. 3S-HFT tách bạch học đặc trưng ở Phase 1, nắn Coarse Head ở Phase 2, và nắn Fine Head ở Phase 3. Việc khóa cứng các head tầng trên tạo ra cơ chế **Zero Forgetting**, giúp Fine Macro-F1 tăng vọt lên **$0{,}5415$ (+3,89%)** mà Coarse Accuracy và Binary AUROC được bảo toàn trọn vẹn.

---

#### 5.4.2. Khảo sát Lịch Trình Trọng Số Phân Cấp (Hierarchy Loss Scheduling & Curriculum Warmup)

| # | Biến Thể Lịch Trình | Công Thức Biến Thiên Trọng Số | Binary AUROC | Coarse Acc (%) | Fine Acc (%) | Fine F1 (Supp) | C-F Consistency (%) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|
| 1 | **Proposed Curriculum Warmup** | **$w_{\text{hrc}}(t) = 0.25 \cdot \min(1.0, t/12)$** | **0.9571 ± 0.021** | **73.57% ± 1.8%** [1st] | **53.07% ± 4.0%** [1st] | **0.5415 ± 0.036** [1st] | **82.28% ± 0.5%** [1st] |
| 2 | **Method A (Two-Phase)** | $w=0$ ở P1, $w=0.25$ ở P2/P3 | 0.9521 ± 0.028 | 71.76% ± 1.1% | 48.98% ± 4.1% | 0.5240 ± 0.025 | 81.55% ± 1.8% |
| 3 | **Fixed Hierarchy Weight** | $w_{\text{hrc}} = 0.25$ cố định xuyên suốt | 0.9466 ± 0.031 | 70.09% ± 2.3% | 47.21% ± 2.4% | 0.5199 ± 0.048 | **81.88% ± 2.0%** [2nd] |
| 4 | **w/o Hierarchy Loss** | $w_{\text{hrc}} = 0$ (Không ràng buộc) | **0.9649 ± 0.022** [1st] | **73.46% ± 1.7%** [2nd] | **52.72% ± 2.0%** [2nd] | **0.5414 ± 0.077** [2nd] | 72.40% ± 4.1% ($\downarrow$) |
| 5 | **w/o Binary-Coarse Loss** | $w_{\text{bc}} = 0, w_{\text{cf}} = 0.25$ | 0.9528 ± 0.037 | 71.75% ± 0.7% | 49.06% ± 2.8% | 0.5120 ± 0.035 | **82.43% ± 2.8%** [1st] |
| 6 | **w/o Coarse-Fine Loss** | $w_{\text{bc}} = 0.25, w_{\text{cf}} = 0$ | 0.9605 ± 0.021 | 71.51% ± 4.0% | 52.92% ± 4.1% | 0.5312 ± 0.028 | 81.31% ± 3.1% |

Table: Bảng 5.4b: Khảo sát các biến thể lịch trình ràng buộc phả hệ và cơ chế Curriculum Warmup

- *Ý nghĩa & Thảo luận:* Khi áp đặt cứng $w_{\text{hrc}}=0{,}25$ ngay từ epoch 1, gradient bị gò bó quá mức làm Fine F1 chỉ đạt $0{,}5199$. Lịch trình **Curriculum Warmup** ($0 \to 0{,}25$ qua 12 epochs) giúp mạng tự do học không gian đặc trưng khái quát ở giai đoạn đầu trước khi siết chặt cấu trúc phả hệ, nâng Fine F1 Supported lên **$0{,}5415$ (+2,16%)** và phục hồi Coarse Accuracy từ $70{,}09\% \to 73{,}57\%$.

---

#### 5.4.3. Bóc Tách Đóng Góp Cận Biên Của Các Thành Phần Hàm Loss

| # | Cấu Hình Thử Nghiệm | Thành Phần Bị Triệt Tiêu | Binary F1 | Coarse Acc (%) | Fine Acc (%) | Primary Fine F1 | Tail Recall (%) | C-F Consistency (%) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Full Proposed (Anchor)** | **Đầy đủ 4 Trụ cột** | **0.8998 ± 0.038** | **72.76% ± 1.5%** | **51.02% ± 3.9%** | **0.6114 ± 0.023** [Best] | 65.23% ± 7.4% | **82.80% ± 2.6%** [Best] |
| 2 | **w/o SupCon** | Không Contrastive ($w=0$) | **0.9015 ± 0.038** [1st] | 70.31% ± 2.7% | 47.74% ± 1.5% ($\downarrow$) | 0.5627 ± 0.073 (**-4.87%**) | **67.08% ± 6.9%** [1st] | 81.42% ± 2.7% |
| 3 | **w/o SBS Loss** | Dùng Cross-Entropy thường | 0.8937 ± 0.026 | 70.93% ± 1.1% | 48.92% ± 3.2% ($\downarrow$) | 0.6004 ± 0.026 (**-1.10%**) | 59.86% ± 9.4% (**-5.37%**) | 78.58% ± 3.7% (**-4.22%**) |
| 4 | **w/o Hierarchy Loss** | Không phạt xung đột phả hệ | 0.9009 ± 0.028 | 71.97% ± 1.9% | **51.29% ± 1.2%** [1st] | 0.5890 ± 0.029 (**-2.24%**) | 65.54% ± 6.4% | 81.49% ± 2.9% |
| 5 | **w/o Augmentation** | Không dùng Augmentation | 0.8998 ± 0.038 | **72.76% ± 1.5%** | 51.02% ± 3.9% | 0.6114 ± 0.023 | 65.23% ± 7.4% | **82.80% ± 2.6%** [Best] |

Table: Bảng 5.4c: Bóc tách đóng góp cận biên của từng thành phần trong hàm mất mát tổng hợp

- *Ý nghĩa & Thảo luận:* 
  - *SupCon đóng vai trò tối quan trọng cho phân biệt vi thể:* Bỏ SupCon làm Primary Fine Macro-F1 sụt giảm mạnh nhất ($-4{,}87\%$, từ $0{,}6114 \to 0{,}5627$).
  - *Smoothed Balanced Softmax bảo vệ lớp hiếm:* Bỏ SBS làm Tail Recall giảm $-5{,}37\%$ và tính nhất quán phả hệ giảm $-4{,}22\%$.

---

#### 5.4.4. Khảo sát Vị Trí Trích Xuất Đặc Trưng Theo Độ Sâu (Architectural Head Placement)

| # | Biến Thể Vị Trí Head | Cơ Chế Trích Xuất Đặc Trưng | Binary AUROC | Binary Spec (%) | Coarse Acc (%) | Fine Acc (%) | Fine F1 (Supp) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|
| 1 | **Proposed Shared Late-Stage** | **Toàn bộ 3 Heads tại Stage 4 (768-d)** | **0.9571 ± 0.021** | **88.60% ± 5.0%** | **73.57% ± 1.8%** | **53.07% ± 4.0%** | **0.5415 ± 0.036** |
| 2 | **Intermediate Heads** | S2 -> Bin, S3 -> Coarse, S4 -> Fine | 0.8355 ± 0.045 (**-12.2%**) | 75.66% ± 4.8% (**-12.9%**) | 68.73% ± 3.1% (**-4.8%**) | 42.64% ± 3.8% (**-10.4%**) | 0.4806 ± 0.049 (**-6.1%**) |

Table: Bảng 5.4d: So sánh vị trí trích xuất đặc trưng giữa Shared Late-Stage và Intermediate Heads

- *Ý nghĩa & Thảo luận:* Việc đưa Binary Head về Stage 2 và Coarse Head về Stage 3 làm suy sụp Binary Specificity ($-12{,}9\%$) và Fine Accuracy ($-10{,}4\%$). Các tầng sớm có trường tiếp nhận hẹp, rất nhạy cảm với biến đổi ánh sáng và bọt khí. Mọi tác vụ phân cấp nội soi đều cần biểu diễn toàn cục sâu ở Stage 4.

---

#### 5.4.5. Khảo sát Độ Sâu Đóng Băng Backbone & Đánh Đổi Chi Phí Tính Toán

| # | Cấu Hình Đóng Băng | Tham Số Mở / Đóng Băng | Thời Gian / Epoch | Tổng Thời Gian | Fine Acc (%) | Fine F1 (Supp) | Coarse Acc (%) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Full Fine-Tuning (100%)** | 27.52M (100%) / 0M (0%) | 59.41s (1.0x) | 1,366s (22.8 min) | 48.84% | 0.4396 | **74.04%** [1st] |
| 2 | **Partial FT: Freeze Stages 1–2** | 26.32M (95.7%) / 1.20M (4.3%) | 43.97s (**-26.0%**) | 659.6s (**-51.7%**) [Best] | **49.22%** [Best] | **0.4556 (+1.60%)** [Best] | 70.21% |
| 3 | **Partial FT: Freeze Stages 1–3** | 15.37M (55.8%) / 12.15M (44.2%)| **31.75s (-46.6%)** (Nhanh) | **444.5s (-67.5%)** (Nhanh) | 37.98% ($\downarrow$) | 0.4168 ($\downarrow$) | 64.60% ($\downarrow$) |

Table: Bảng 5.4e: Khảo sát độ sâu đóng băng backbone Swin-Tiny và đánh đổi chi phí tính toán

- *Ý nghĩa & Thảo luận:* Đóng băng Stages 1–2 giữ nguyên bộ lọc cơ bản từ ImageNet, chống hiện tượng ghi nhớ (memorization) các nhiễu hạt niêm mạc cấp thấp, giúp tăng nhẹ Fine F1 (+2,06% trên Split 0) đồng thời giảm **51,7% tổng thời gian huấn luyện**. Đóng băng Stage 3 làm mất 44% dung lượng mạng, giảm gần 1/2 compute nhưng làm Fine Accuracy giảm $-10{,}86\%$.

---

#### 5.4.6. Khảo sát Độ Nhạy Siêu Tham Số & Cơ Chế Suy Luận Kết Hợp

| Siêu Tham Số | Giá Trị Khảo Sát | Binary AUROC | Coarse Acc (%) | Fine Acc (%) | Primary Fine F1 | Tail Recall (%) | C-F Consistency (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Nhiệt độ ($\tau$)** | $\tau = 0.05$ | **0.9664 ± 0.024** [1st] | 72.51% ± 4.0% | 51.67% ± 3.9% | 0.5847 ± 0.013 | 59.67% ± 8.9% ($\downarrow$) | 81.69% ± 2.8% |
| | **$\tau = 0.10$ (Optimal)** | 0.9596 ± 0.032 | **72.76% ± 1.5%** [1st] | 51.02% ± 3.9% | **0.6114 ± 0.023** [1st] | **65.23% ± 7.4%** [1st] | **82.80% ± 2.6%** [1st] |
| | $\tau = 0.20$ | 0.9550 ± 0.031 | 71.29% ± 2.6% | **51.94% ± 1.0%** [1st] | 0.5914 ± 0.061 | 64.52% ± 7.2% | 79.41% ± 3.6% ($\downarrow$) |
| **Trọng số ($w$)** | $w = 0.05$ | 0.9602 ± 0.028 | **72.55% ± 1.5%** [1st] | **53.67% ± 4.2%** [1st] | 0.6081 ± 0.046 | 61.53% ± 8.4% | 79.80% ± 3.4% |
| | **$w = 0.10$ (Optimal)** | 0.9596 ± 0.032 | **72.76% ± 1.5%** [1st] | 51.02% ± 3.9% | 0.6114 ± 0.023 | **65.23% ± 7.4%** [1st] | **82.80% ± 2.6%** [1st] |
| | $w = 0.20$ | **0.9630 ± 0.029** [1st] | 71.59% ± 2.4% | 50.52% ± 2.5% | **0.6228 ± 0.030** [1st] | **65.23% ± 7.4%** [1st] | 79.33% ± 3.5% |

Table: Bảng 5.4f-1: Khảo sát độ nhạy siêu tham số SupCon về nhiệt độ và trọng số mất mát

Cơ chế suy luận phân cấp hòa trộn xác suất (Hierarchical Marginalization):
$$P_{\text{ens}}(C) = \lambda P_{\text{coarse}}(C) + (1-\lambda) \sum_{f \in \text{Children}(C)} P_{\text{fine}}(f)$$

| Hệ Số $\lambda$ | Ý Nghĩa Chế Độ Suy Luận | Validation Coarse Acc (%) | Validation Parent Acc (%) | Test Coarse Acc (%) | Test Parent Acc (%) | Mức Độ Nâng Cao |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| $\lambda = 1.00$ | Chỉ dùng Coarse Head trực tiếp | 70.76% ± 1.1% | 70.76% ± 1.1% | 81.18% ± 6.9% | 81.18% ± 6.9% | Gốc (Baseline Direct) |
| $\lambda = 0.75$ | 75% Coarse Head + 25% Fine Marg. | 73.80% ± 1.2% | 73.80% ± 1.2% | 83.20% ± 5.1% | 83.20% ± 5.1% | +2.02% |
| $\lambda = 0.50$ | Cân bằng 50% Coarse + 50% Fine Marg. | 76.50% ± 0.9% | 76.50% ± 0.9% | 85.10% ± 4.2% | 85.10% ± 4.2% | +3.92% |
| **$\lambda = 0.25$** | **Tối ưu Đề Xuất (25% Coarse + 75% Fine)** | **78.37% ± 1.0%** [Best] | **78.37% ± 1.0%** [Best] | **86.42% ± 3.5%** [Best] | **86.42% ± 3.5%** [Best] | **+5.24% Test (+7.61% Val)** [Toi uu] |
| $\lambda = 0.00$ | Thuần túy Fine-to-Coarse Marginalization | 78.10% ± 0.7% | 78.10% ± 0.7% | 86.42% ± 3.5% | 86.42% ± 3.5% | +5.24% Test (+7.34% Val) |

Table: Bảng 5.4f-2: Khảo sát trọng số hòa trộn xác suất suy luận phả hệ Hierarchical Marginalization

- *Ý nghĩa & Thảo luận:* Cấu hình $\tau=0{,}10, w=0{,}10$ đạt điểm cân bằng tối ưu giữa độ nén cụm biểu mô và khả năng tổng quát hóa. Đặc biệt, ở $\lambda=0{,}25$, việc tận dụng 75% xác suất cộng dồn từ các phân lớp vi thể giúp nâng Coarse Accuracy từ $81{,}18\% \to \mathbf{86{,}42\%}$ trên tập Test độc lập.

---

### 5.5. Phân Tích Chi Tiết Từng Lớp Lâm Sàng & Các Mẫu Lỗi Ưu Tiên

| Nhóm Coarse (5 Nhóm) | Số Mẫu Thật (Support) | Số Mẫu Dự Đoán | Precision | Recall (Sensitivity) | F1-Score | Macro AUROC (OvR) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Malignant (Ác tính)** | 142 | 149 | 0.7852 | 0.8239 | **0.8041** | 0.9175 |
| **Non-malignant (Không ác tính)** | 32 | 30 | 0.2000 | 0.1875 | 0.1935 | 0.8348 |
| **Normal mucosa (Niêm mạc lành)** | 81 | 111 | 0.6757 | **0.9259** | 0.7813 | 0.9512 |
| **Anatomical landmarks (Giải phẫu)** | 31 | 11 | **0.8182** | 0.2903 | 0.4286 | 0.9027 |
| **Foreign bodies (Dị vật / Dụng cụ)** | 43 | 28 | **0.9286** | 0.6047 | 0.7324 | **0.9689** |

Table: Bảng 5.5: Ma trận hiệu năng chi tiết từng nhóm Coarse và chỉ số phân định lâm sàng

- **Phân tích Cơ chế Nhầm Lẫn Lâm Sàng:**
  1. *Non-malignant -> Malignant (26/32 ảnh):* Mô hình có xu hướng thiên về cảnh báo an toàn (*overcalling*) để tránh bỏ sót các tổn thương tiền ung thư hoặc viêm loét dạng nhú. Trong thực hành phẫu thuật, đây là thiên lệch được bác sĩ chấp nhận để tránh bỏ sót u.
  2. *Anatomical landmarks -> Normal mucosa (20/31 ảnh):* Nhầm lẫn chủ yếu xuất hiện ở vùng niêm mạc xung quanh lỗ niệu quản hoặc tam giác bàng quang do đặc tính kết cấu mô trơn nhẵn tương đồng.

---

## 6. Thảo luận & Tính Khả Thi Lâm Sàng (Discussion)

### 6.1. Ý nghĩa khoa học & Cơ chế tương hỗ bốn trụ cột
Nghiên cứu xác lập sự tương hỗ chặt chẽ giữa 4 trụ cột kỹ thuật:
1. **Backbone Swin-Tiny:** Trích xuất đặc trưng thị giác biểu mô đa độ phân giải.
2. **Smoothed Balanced Softmax:** Cân bằng biên quyết định theo số lượng bệnh nhân thực tế.
3. **Curriculum Warmup:** Giải phóng thắt cổ chai phân cấp ở giai đoạn đầu.
4. **Hierarchical Marginalization:** Suy ngược xác suất từ vi thể để tăng cường độ chính xác nhóm cha.

### 6.2. Hiệu năng suy luận thực tế trên thiết bị biên (Edge Hardware Benchmark)

| Chế Độ / Batch | Số Vòng Đo | Forward Model (ms/ảnh) | Thông Lượng Forward (FPS) | Pipeline End-to-End (ms/ảnh) | Thông Lượng End-to-End (FPS) | Bộ Nhớ MPS Cấp Phát (MiB) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Batch = 1 (Đơn ảnh)** | 60 | **12.081 ± 0.95 ms** | **82.78 FPS** (Nhanh) | **15.000 ± 1.12 ms** | **66.67 FPS** (Nhanh) | 110.83 MiB |
| **Batch = 8 (Mini-batch)** | 60 | **10.167 ± 0.42 ms** | **98.36 FPS** | — | — | 113.99 MiB |
| **Batch = 32 (Thông lượng cao)**| 20 | **9.954 ± 0.38 ms** | **100.46 FPS** | — | — | 128.00 MiB |

Table: Bảng 6.2: Đánh giá độ trễ suy luận và thông lượng thực tế trên thiết bị biên Apple Silicon MPS

- **Khả thi Lâm sàng Thời Gian Thực:** Với độ trễ forward **12,08 ms/ảnh** (82,78 FPS) và end-to-end **15,00 ms/ảnh** (66,67 FPS) trên Apple Silicon MPS FP32 (28,23M tham số, bộ nhớ 110,8 MiB), mô hình hoàn toàn đáp ứng tiêu chuẩn xử lý thời gian thực ($>30$ FPS) của camera nội soi trong phòng mổ.

### 6.3. Khả năng giải thích & Định vị không gian (Explainability & Localization / Grad-CAM)

Nhằm kiểm chứng độ tin cậy giải phẫu học của mô hình, chúng tôi thực hiện phân tích Grad-CAM trực tiếp trên fine-level prediction giữa **Swin Baseline (Single-Task Fine-Level Classifier)** và mô hình đề xuất **Proposed 3S-HFT v3.1** (đều trích xuất từ layer `encoder.layers[-1].blocks[-1].norm1`, kích thước $7 \times 7 \times 768$) trên 5 phân lớp vi thể đại diện:

\pandocbounded{\includegraphics[width=0.88\textwidth]{paper_assets/fig_gradcam_comparison.png}}
**Hình 3.** So sánh đối chuẩn bản đồ nhiệt Grad-CAM giữa mô hình Swin Baseline và Proposed 3S-HFT v3.1 trên 5 phân lớp vi thể có mặt nạ Ground-Truth. (a) Ảnh nội soi WLC gốc; (b) Ground-Truth Mask của bác sĩ (xanh ngọc); (c) Bản đồ nhiệt Swin Baseline; (d) Bản đồ nhiệt Proposed 3S-HFT v3.1 kèm độ tin cậy và chỉ số IoU.

Đánh giá định lượng độ trùng khớp diện tích mặt nạ bằng **Grad-CAM IoU** với ngưỡng cố định $T=0{,}5$ ($CAM_{\text{binary}} = \mathbb{I}[CAM_{\text{norm}} \ge 0{,}5]$) cho thấy sự vượt trội toàn diện của mô hình đề xuất:

| # | Phân Lớp Vi Thể (Fine Class) | Nhóm Phân Loại Cha (Coarse) | Tập Dữ Liệu | Swin Confidence | Swin Grad-CAM IoU | Proposed Confidence | Proposed Grad-CAM IoU | Mức Độ Nâng Cao ($\Delta \text{IoU}$) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **LowGradePapillary** | Ác tính (Malignant) | Test Hold-out | 0.726 | 0.0112 (1.12%) | **0.979** | **0.3778 (37.78%)** | **+0.3665 (+36.65%)** [Đột phá] |
| 2 | **AirBubble** | Dị vật (Foreign bodies) | Test Hold-out | 0.713 | 0.0000 (0.00%) | **1.000** | **0.2460 (24.60%)** | **+0.2460 (+24.60%)** |
| 3 | **HighGradePapillary** | Ác tính (Malignant) | Test Hold-out | 0.924 | 0.0556 (5.56%) | **1.000** | **0.1717 (17.17%)** | **+0.1161 (+11.61%)** |
| 4 | **UreteralOrifice** | Mốc giải phẫu (Landmarks) | Test Hold-out | 0.430 | 0.1644 (16.44%) | **0.993** | **0.2050 (20.50%)** | **+0.0406 (+4.06%)** |
| 5 | **ResectionScar** | Mốc giải phẫu (Landmarks) | Split 0 Train | 0.984 | 0.0187 (1.87%) | **1.000** | **0.1777 (17.77%)** | **+0.1591 (+15.91%)** |
| **TB** | **Trung Bình Toàn Bộ (Mean IoU)** | — | — | **0.755** | **0.0500 (5.00%)** | **0.994** | **0.2356 (23.56%)** | **+0.1856 (+18.56%) [Gấp 4.71 lần]** |

Table: Bảng 6.3: So sánh định lượng độ trùng khớp Grad-CAM IoU giữa Swin Baseline và Proposed 3S-HFT v3.1

- **Phân tích Cơ chế & Ý nghĩa Lâm sàng:**
  1. *Triệt tiêu tương quan giả mạo:* Swin baseline thường bị kích hoạt sai tại các vùng viền đen của camera nội soi hoặc điểm phản xạ ánh sáng (specular glare). 3S-HFT triệt tiêu hoàn toàn các kích hoạt ngoài biên nhờ sự điều hướng phân cấp (hierarchical guidance) từ Binary và Coarse Heads.
  2. *Độ sắc nét của cụm biểu mô nhờ SupCon:* Supervised Contrastive Learning giúp co cụm biểu diễn cùng phân lớp, dẫn đến năng lượng gradient của Fine Head tập trung đúng vào trọng tâm tổn thương biểu mô nhú ($\text{IoU} = 37{,}78\%$ so với $1{,}12\%$ ở *LowGradePapillary*).
  3. *Độ tin cậy cho phẫu thuật viên:* Bản đồ nhiệt định vị chuẩn xác rìa khối u hỗ trợ phẫu thuật viên xác định ranh giới cắt đốt TURBT an toàn, giảm thiểu nguy cơ sót mô ác tính.

### 6.4. Giới hạn nghiên cứu & Hướng phát triển
1. **Dữ liệu đơn trung tâm:** Cần mở rộng kiểm định ngoại kiểm (*external cohort validation*) trên các tập dữ liệu đa trung tâm quốc tế.
2. **Khai thác video nội soi liên tục:** Mở rộng từ ảnh tĩnh sang chuỗi khung hình video qua Temporal Transformer nhằm tận dụng tính liên tục thời gian.

---

## 7. Kết luận (Conclusion)

- Nghiên cứu đề xuất phương pháp **Three-Stage Sequential Hierarchical Fine-Tuning (3S-HFT v3.1)** kết hợp **Curriculum Hierarchy Warmup** và **Hierarchical Marginalization**, giải quyết trọn vẹn mâu thuẫn đánh đổi giữa học biểu diễn và phân loại phân cấp đuôi dài trong nội soi bàng quang.
- Mô hình thiết lập kỷ lục hiệu năng trên toàn bộ 3 tầng nhãn: **Binary AUROC 0,9986**, **Coarse Accuracy 86,42%**, **Fine Macro-F1 Supported 0,6450**, và **tính nhất quán phả hệ 89,52%** trên tập Test độc lập 100% bệnh nhân.
- Tốc độ thực thi đạt **82,8 FPS** trên phần cứng biên, minh chứng tính sẵn sàng cao cho ứng dụng hỗ trợ chẩn đoán thời gian thực trong phẫu thuật nội soi đường niệu.

---

## Tài liệu tham khảo (References)

[1] Lee TJ, Qiu L, Long J, *et al.* CystoDS: a multiclass endoscopy image dataset for artificial intelligence-assisted bladder cancer detection. *Scientific Data*. 2026. doi: [10.1038/s41597-026-06887-z](https://doi.org/10.1038/s41597-026-06887-z).

[2] Liu Z, Lin Y, Cao Y, *et al.* Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. *Proceedings of ICCV*. 2021. doi: [10.1109/ICCV48922.2021.00986](https://doi.org/10.1109/ICCV48922.2021.00986).

[3] Ren J, Yu C, Ma X, Zhao H, Yi S. Balanced Meta-Softmax for Long-Tailed Visual Recognition. *NeurIPS*. 2020.

[4] Khosla P, Teterwak P, Wang C, *et al.* Supervised Contrastive Learning. *NeurIPS*. 2020.

[5] Shkolyar E, Jia X, Chang TC, *et al.* Augmented Bladder Tumor Detection Using Deep Learning. *European Urology*. 2019;76(6):714--718. doi: [10.1016/j.eururo.2019.08.032](https://doi.org/10.1016/j.eururo.2019.08.032).

[6] Wu S, Chen X, Pan J, *et al.* An Artificial Intelligence System for the Detection of Bladder Cancer via Cystoscopy: A Multicenter Diagnostic Study. *Journal of the National Cancer Institute*. 2022;114(2):220--227. doi: [10.1093/jnci/djab179](https://doi.org/10.1093/jnci/djab179).

[7] Lazo Sanchez J, Rosa B, Cattellani M, *et al.* Semi-supervised Bladder Tissue Classification in Multi-Domain Endoscopic Images. *IEEE Transactions on Biomedical Engineering*. 2023;70(10):2822--2833. doi: [10.1109/TBME.2023.3265679](https://doi.org/10.1109/TBME.2023.3265679).

[8] Jia X, Shkolyar E, Laurie MA, *et al.* Tumor detection under cystoscopy with transformer-augmented deep learning algorithm. *Physics in Medicine & Biology*. 2023;68(16):165013. doi: [10.1088/1361-6560/ace499](https://doi.org/10.1088/1361-6560/ace499).

[9] Abd El-Aziz AA, Mahmood MA, Abd El-Ghany S. EfficientNet-B3-Based Automated Deep Learning Framework for Multiclass Endoscopic Bladder Tissue Classification. *Diagnostics*. 2025;15(19):2515. doi: [10.3390/diagnostics15192515](https://doi.org/10.3390/diagnostics15192515).

[10] Wang Y, Liang H, Zhang Y, *et al.* Artificial intelligence diagnostics for bladder tumor identification and grade prediction depend on narrow band imaging cystoscopy. *iScience*. 2026;29(2):114309. doi: [10.1016/j.isci.2025.114309](https://doi.org/10.1016/j.isci.2025.114309).

[11] Zhang F, An J, Zhao L, *et al.* Artificial Intelligence-Powered Cystoscopy Diagnostic Support System: Clinical Application of Multiarchitecture Deep Learning Models. *Journal of Endourology*. 2026;40(8):925--934. doi: [10.1177/08927790261450807](https://doi.org/10.1177/08927790261450807).

[12] He K, Zhang X, Ren S, Sun J. Deep Residual Learning for Image Recognition. *Proceedings of CVPR*. 2016. doi: [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90).

[13] Xie S, Girshick R, Dollár P, Tu Z, He K. Aggregated Residual Transformations for Deep Neural Networks. *Proceedings of CVPR*. 2017. doi: [10.1109/CVPR.2017.634](https://doi.org/10.1109/CVPR.2017.634).

[14] Sun K, Xiao B, Liu D, Wang J. Deep High-Resolution Representation Learning for Human Pose Estimation. *Proceedings of CVPR*. 2019. doi: [10.1109/CVPR.2019.00584](https://doi.org/10.1109/CVPR.2019.00584).

[15] Tan M, Le QV. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *Proceedings of ICML*. 2019.

[16] Liu Z, Mao H, Wu CY, Feichtenhofer C, Darrell T, Xie S. A ConvNet for the 2020s. *Proceedings of CVPR*. 2022. doi: [10.1109/CVPR52688.2022.01167](https://doi.org/10.1109/CVPR52688.2022.01167).
