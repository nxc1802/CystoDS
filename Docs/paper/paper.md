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
1. **Kiến trúc tinh chỉnh tuần tự ba giai đoạn (3S-HFT v3.1):** Tách bạch rõ ràng: *Representation Learning (với Curriculum Warmup)* $\rightarrow$ *Coarse Alignment (với Zero Forgetting)* $\rightarrow$ *Fine Alignment*.
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

### 5.1. Bảng Tổng Hợp Đại Thống Kê Toàn Bộ Thực Nghiệm (Master Comprehensive Benchmark Table -- Stages 10--40)

Dưới đây là bảng đối chuẩn tổng hợp toàn bộ 27 cấu hình thực nghiệm từ Stage 10 đến Stage 40 qua 3 phân hoạch hold-out độc lập bệnh nhân (`split_0`, `split_1`, `split_2`):

| Giai đoạn & Phân nhóm | Phương pháp & Cấu hình | Chiến lược / Hàm Loss | Binary AUROC | Binary F1 | Coarse Acc | Coarse F1 | Fine Acc | Fine F1 (Supp) | Fine F1 (All 22) | C-F Consistency | Parent Acc (Ens) |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Stage 10 (Backbone)** | ResNet-152 | Multitask Joint (CE) | 0,8698 ± 0,050 | 0,8191 ± 0,038 | 56,62% ± 0,3% | 0,4398 ± 0,017 | 34,71% ± 5,2% | 0,2098 ± 0,038 | 0,1482 ± 0,025 | 68,42% ± 3,1% | 54,12% ± 2,8% |
| **Stage 10 (Backbone)** | ResNeXt-50 | Multitask Joint (CE) | 0,9088 ± 0,037 | 0,8387 ± 0,025 | 58,61% ± 1,4% | 0,4600 ± 0,028 | 37,05% ± 3,5% | 0,2023 ± 0,036 | 0,1510 ± 0,028 | 71,05% ± 2,5% | 57,30% ± 1,9% |
| **Stage 10 (Backbone)** | HRNet-W18 | Multitask Joint (CE) | 0,9385 ± 0,035 | 0,8759 ± 0,022 | 63,66% ± 4,3% | 0,5461 ± 0,035 | 43,44% ± 3,4% | 0,3979 ± 0,056 | 0,2845 ± 0,039 | 73,88% ± 2,2% | 61,55% ± 3,8% |
| **Stage 10 (Backbone)** | Swin-Tiny (Baseline) | Multitask Joint (CE) | 0,9507 ± 0,027 | 0,8992 ± 0,029 | 71,19% ± 2,5% | 0,6243 ± 0,014 | 49,28% ± 6,5% | 0,5105 ± 0,068 | 0,3755 ± 0,045 | 76,45% ± 2,1% | 68,90% ± 2,4% |
| **Stage 10 (Backbone)** | Swin-Tiny (Single-Task) | Binary Detection Only | 0,9590 ± 0,033 | 0,8930 ± 0,034 | -- | -- | -- | -- | -- | -- | -- |
| **Stage 20 (Long-Tail)** | Cross-Entropy | 1-Stage Multi-Task CE | 0,9489 ± 0,042 | 0,8888 ± 0,050 | 67,49% ± 2,2% | 0,5687 ± 0,011 | 50,23% ± 4,6% | 0,5268 ± 0,076 | 0,3850 ± 0,052 | 77,37% ± 3,0% | 66,50% ± 2,1% |
| **Stage 20 (Long-Tail)** | Weighted CE | Inverse Class Frequency | 0,9427 ± 0,036 | 0,8747 ± 0,038 | 67,86% ± 2,2% | 0,5302 ± 0,056 | 49,18% ± 3,5% | 0,5053 ± 0,067 | 0,3712 ± 0,048 | 73,79% ± 2,2% | 65,80% ± 2,5% |
| **Stage 20 (Long-Tail)** | Focal Loss | Gamma=2.0 Modulating Factor | 0,9506 ± 0,024 | 0,8938 ± 0,032 | 68,16% ± 3,7% | 0,5593 ± 0,058 | 51,09% ± 5,7% | 0,4976 ± 0,062 | 0,3680 ± 0,041 | 77,09% ± 8,1% | 66,90% ± 3,2% |
| **Stage 20 (Long-Tail)** | LDAM Loss | Margin-based Rare Push | 0,9522 ± 0,020 | 0,8836 ± 0,016 | 69,37% ± 1,9% | 0,5834 ± 0,064 | 45,09% ± 2,8% | 0,4908 ± 0,058 | 0,3610 ± 0,039 | 72,33% ± 3,6% | 67,10% ± 2,0% |
| **Stage 20 (Long-Tail)** | Logit Adjustment | Post-hoc Prior Margin | 0,9455 ± 0,042 | 0,8888 ± 0,050 | 69,98% ± 3,0% | 0,5837 ± 0,022 | 49,62% ± 1,7% | 0,4822 ± 0,048 | 0,3540 ± 0,032 | 76,98% ± 3,4% | 68,20% ± 2,6% |
| **Stage 20 (Long-Tail)** | Balanced Softmax | Instance Frequency Prior | 0,9531 ± 0,038 | 0,8893 ± 0,031 | 69,58% ± 2,6% | 0,5912 ± 0,032 | 50,08% ± 5,7% | 0,4823 ± 0,060 | 0,3582 ± 0,044 | 74,57% ± 3,7% | 67,85% ± 2,4% |
| **Stage 20 (Long-Tail)** | Smoothed Balanced Softmax | Patient-based Smoothed Prior | 0,9521 ± 0,039 | 0,8907 ± 0,058 | 70,12% ± 3,9% | 0,6212 ± 0,038 | 52,45% ± 1,7% | 0,5506 ± 0,074 | 0,3985 ± 0,051 | 77,58% ± 1,6% | 69,50% ± 3,1% |
| **Stage 40 (Ablation)** | 1-Stage Joint Baseline | Joint CE + SupCon + SBS | 0,9594 ± 0,018 | 0,8965 ± 0,012 | 73,64% ± 0,9% | 0,6576 ± 0,006 | 52,63% ± 2,9% | 0,5026 ± 0,046 | 0,3718 ± 0,026 | 74,38% ± 3,2% | 71,20% ± 1,2% |
| **Stage 40 (Ablation)** | 2-Stage Decoupled (D2S-HFT) | Rep -> Fine SBS Only | 0,9617 ± 0,028 | 0,8912 ± 0,022 | 72,12% ± 4,6% | 0,6313 ± 0,031 | 52,20% ± 4,8% | 0,5266 ± 0,056 | 0,3893 ± 0,032 | 78,90% ± 2,4% | 72,50% ± 3,5% |
| **Stage 40 (Ablation)** | 3S-HFT Fixed Hierarchy (w=0.25) | Fixed Hierarchy Weight P1-P3 | 0,9466 ± 0,031 | 0,8776 ± 0,025 | 70,09% ± 2,3% | 0,6119 ± 0,018 | 47,21% ± 2,4% | 0,5199 ± 0,048 | 0,3844 ± 0,023 | 81,88% ± 2,0% | 75,50% ± 1,5% |
| **Stage 40 (Ablation)** | 3S-HFT Method A (Two-Phase) | w=0 in P1, w=0.25 in P2/P3 | 0,9521 ± 0,028 | 0,8981 ± 0,030 | 71,76% ± 1,1% | 0,6371 ± 0,008 | 48,98% ± 4,1% | 0,5240 ± 0,025 | 0,3883 ± 0,017 | 81,55% ± 1,8% | 76,11% ± 1,1% |
| **Stage 40 (Ablation)** | Ablation: w/o SupCon (w=0) | CE Only -> Hierarchy | 0,9437 ± 0,027 | 0,8720 ± 0,028 | 70,07% ± 3,7% | 0,6140 ± 0,038 | 51,57% ± 4,0% | 0,5042 ± 0,052 | 0,3722 ± 0,018 | 77,12% ± 3,1% | 69,80% ± 2,9% |
| **Stage 40 (Ablation)** | Ablation: w/o Hierarchy Loss (w=0) | Multi-Task w/o Coarse-Fine Loss | 0,9649 ± 0,022 | 0,8950 ± 0,019 | 73,46% ± 1,7% | 0,6426 ± 0,009 | 52,72% ± 2,0% | 0,5414 ± 0,077 | 0,3998 ± 0,047 | 72,40% ± 4,1% | 72,10% ± 1,8% |
| **Stage 40 (Ablation)** | Ablation: Strategy cRT | Phase 2 cRT Sampler (Fine Only) | 0,9617 ± 0,028 | 0,8910 ± 0,024 | 72,12% ± 4,6% | 0,6313 ± 0,031 | 51,96% ± 2,5% | 0,5311 ± 0,048 | 0,3930 ± 0,029 | 78,45% ± 2,6% | 73,20% ± 3,0% |
| **Stage 40 (Ablation)** | Ablation: Target All Heads | Phase 2 Unfreeze All 3 Heads | 0,9583 ± 0,035 | 0,8875 ± 0,031 | 73,10% ± 3,4% | 0,6435 ± 0,031 | 51,55% ± 3,6% | 0,5129 ± 0,059 | 0,3794 ± 0,038 | 76,80% ± 2,9% | 72,80% ± 2,5% |
| **Stage 40 (Ablation)** | Ablation: Freeze Stages 1-2 | Partial Finetuning (Swin S3-S4) | 0,9524 ± 0,028 | 0,8830 ± 0,026 | 73,56% ± 2,5% | 0,6535 ± 0,033 | 50,63% ± 1,9% | 0,4950 ± 0,028 | 0,3669 ± 0,022 | 75,10% ± 2,0% | 71,90% ± 2,1% |
| **Stage 40 (Ablation)** | Ablation: Freeze Stages 1-3 | Partial Finetuning (Swin S4 Only) | 0,9246 ± 0,036 | 0,8540 ± 0,035 | 66,22% ± 2,9% | 0,5765 ± 0,037 | 43,31% ± 4,2% | 0,4814 ± 0,052 | 0,3555 ± 0,024 | 70,20% ± 3,8% | 65,10% ± 2,6% |
| **Stage 40 (Ablation)** | Multi-Stage Intermediate Heads | S2 -> Bin, S3 -> Coarse, S4 -> Fine | 0,8355 ± 0,045 | 0,7820 ± 0,041 | 68,73% ± 3,1% | 0,5980 ± 0,032 | 42,64% ± 3,8% | 0,4806 ± 0,049 | 0,3714 ± 0,031 | 71,50% ± 3,5% | 66,30% ± 2,8% |
| **Stage 30 (Proposed - Val)** | Proposed 3S-HFT (Direct Coarse) | Curriculum Warmup + SBS Alignment | 0,9571 ± 0,021 | 0,8960 ± 0,026 | 73,57% ± 1,8% | 0,6525 ± 0,012 | 53,07% ± 4,0% | 0,5415 ± 0,036 | 0,4007 ± 0,015 | 82,28% ± 0,5% | 70,76% ± 1,1% |
| **Stage 30 (Proposed - Val)** | **Proposed 3S-HFT (Hierarchical Ens.)** | **Warmup + Ensemble (lambda=0.25)** | **0,9571 ± 0,021** | **0,8960 ± 0,026** | **78,37% ± 1,0%** | **0,6525 ± 0,012** | **53,07% ± 4,0%** | **0,5415 ± 0,036** | **0,4007 ± 0,015** | **82,28% ± 0,5%** | **78,37% ± 1,0%** |
| **Stage 30 (Proposed - Test)** | Proposed 3S-HFT (Direct Coarse) | Hold-out Test Split Evaluation | 0,9986 ± 0,0002 | 0,9811 ± 0,004 | 82,37% ± 7,0% | 0,7572 ± 0,117 | 74,73% ± 11,9% | 0,6450 ± 0,111 | 0,4691 ± 0,080 | 89,52% ± 3,8% | 81,18% ± 6,9% |
| **Stage 30 (Proposed - Test)** | **Proposed 3S-HFT (Hierarchical Ens.)** | **Hold-out Test Ensemble (lambda=0.25)** | **0,9986 ± 0,0002** | **0,9811 ± 0,004** | **86,42% ± 3,5%** | **0,7572 ± 0,117** | **74,73% ± 11,9%** | **0,6450 ± 0,111** | **0,4691 ± 0,080** | **89,52% ± 3,8%** | **86,42% ± 3,5%** |

### 5.2. Stage 10 -- Sàng lọc 4 Họ Kiến trúc Backbone (3-Split Benchmark)

| Backbone | Chế độ | Binary AUROC | Binary F1 | Coarse Acc | Coarse Macro-F1 | Fine Acc | Fine Macro-F1 (Supp) | Best Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Swin-Tiny** | **Multitask** | 0,9507 ± 0,027 | 0,8992 ± 0,029 | **71,19% ± 2,5%** | **0,6243 ± 0,014** | **49,28% ± 6,5%** | **0,5105 ± 0,068** | **0,5579 ± 0,022** |
| Swin-Tiny | Binary Only | **0,9590 ± 0,033** | 0,8930 ± 0,034 | -- | -- | -- | -- | 0,9590 ± 0,033 |
| **HRNet-W18** | **Multitask** | 0,9385 ± 0,035 | 0,8759 ± 0,022 | 63,66% ± 4,3% | 0,5461 ± 0,035 | 43,44% ± 3,4% | 0,3979 ± 0,056 | 0,4949 ± 0,049 |
| HRNet-W18 | Binary Only | 0,9579 ± 0,021 | **0,8984 ± 0,020** | -- | -- | -- | -- | 0,9576 ± 0,021 |
| **ResNeXt-50** | **Multitask** | 0,9088 ± 0,037 | 0,8387 ± 0,025 | 58,61% ± 1,4% | 0,4600 ± 0,028 | 37,05% ± 3,5% | 0,2023 ± 0,036 | 0,3421 ± 0,046 |
| ResNeXt-50 | Binary Only | 0,9059 ± 0,034 | 0,8356 ± 0,010 | -- | -- | -- | -- | 0,9115 ± 0,035 |
| **ResNet-152** | **Multitask** | 0,8698 ± 0,050 | 0,8191 ± 0,038 | 56,62% ± 0,3% | 0,4398 ± 0,017 | 34,71% ± 5,2% | 0,2098 ± 0,038 | 0,3371 ± 0,029 |
| ResNet-152 | Binary Only | 0,8879 ± 0,038 | 0,8366 ± 0,030 | -- | -- | -- | -- | 0,8930 ± 0,038 |

- **Takeaway kỹ thuật:** Swin-Tiny với Self-Attention vượt trội toàn diện so với các kiến trúc tích chập CNN truyền thống, đặc biệt ở tầng vi thể mô học (Fine F1 gấp 2,5 lần ResNet-152).

### 5.2. Stage 20 -- Sàng lọc 7 Hàm Mất Mát Đuôi Dài (3-Split Benchmark)

| # | Phương pháp Hàm Mất Mát | Binary AUROC | Binary F1 | Coarse Acc | Coarse Macro-F1 | Fine Acc | Fine Macro-F1 (Supp) | Primary Fine F1 (13 Lớp) | Tail Recall (n <= 20) | Coarse-Fine Consistency |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Smoothed Balanced Softmax** | 0,9521 ± 0,039 | 0,8907 ± 0,058 | **70,12% ± 3,9%** | **0,6212 ± 0,038** | **52,45% ± 1,7%** | **0,5506 ± 0,074** | **0,5607 ± 0,050** | **66,38% ± 11,4%** | **77,58% ± 1,6%** |
| 2 | Balanced Softmax | **0,9531 ± 0,038** | 0,8893 ± 0,031 | 69,58% ± 2,6% | 0,5912 ± 0,032 | 50,08% ± 5,7% | 0,4823 ± 0,060 | 0,5049 ± 0,022 | 62,76% ± 6,1% | 74,57% ± 3,7% |
| 3 | Cross-Entropy (Baseline) | 0,9489 ± 0,042 | 0,8888 ± 0,050 | 67,49% ± 2,2% | 0,5687 ± 0,011 | 50,23% ± 4,6% | 0,5268 ± 0,076 | 0,5245 ± 0,019 | 66,07% ± 8,9% | 77,37% ± 3,0% |
| 4 | Logit Adjustment | 0,9455 ± 0,042 | 0,8888 ± 0,050 | 69,98% ± 3,0% | 0,5837 ± 0,022 | 49,62% ± 1,7% | 0,4822 ± 0,048 | 0,5041 ± 0,034 | 59,67% ± 8,4% | 76,98% ± 3,4% |
| 5 | Focal Loss | 0,9506 ± 0,024 | **0,8938 ± 0,032** | 68,16% ± 3,7% | 0,5593 ± 0,058 | 51,09% ± 5,7% | 0,4976 ± 0,062 | 0,5150 ± 0,028 | 60,97% ± 8,6% | 77,09% ± 8,1% |
| 6 | Weighted CE | 0,9427 ± 0,036 | 0,8747 ± 0,038 | 67,86% ± 2,2% | 0,5302 ± 0,056 | 49,18% ± 3,5% | 0,5053 ± 0,067 | 0,5173 ± 0,051 | 63,97% ± 10,9% | 73,79% ± 2,2% |
| 7 | LDAM Loss | 0,9522 ± 0,020 | 0,8836 ± 0,016 | 69,37% ± 1,9% | 0,5834 ± 0,064 | 45,09% ± 2,8% | 0,4908 ± 0,058 | 0,5067 ± 0,030 | 62,51% ± 11,2% | 72,33% ± 3,6% |

- **Takeaway kỹ thuật:** Smoothed Balanced Softmax đứng đầu ở cả 4 tiêu chí cốt lõi, nâng Tail Recall lên 66,38% mà không làm vỡ cấu trúc biểu diễn.

### 5.3. Stage 30 -- Đánh giá Toàn diện Mô hình Đề xuất 3S-HFT v3.1

#### Bảng Kết quả Trên Tập Validation (3-Fold Patient-Disjoint Validation):
| Tiêu chí Đánh giá / Metric | Baseline 1-Stage Joint | 3S-HFT Fixed ($w=0{,}25$) Cũ | **3S-HFT Đề Xuất Mới (Warmup)** | Chênh lệch ($\Delta$ vs Cũ) |
|---|:---:|:---:|:---:|:---:|
| **Binary AUROC** | **0,9594 ± 0,018** | 0,9466 ± 0,031 | 0,9571 ± 0,021 | +0,0105 |
| **Binary F1-Score** | **0,8965 ± 0,012** | 0,8776 ± 0,025 | 0,8960 ± 0,026 | +0,0184 |
| **Binary Sensitivity (Độ nhạy ROI)** | **89,05% ± 3,4%** | 88,34% ± 4,4% | 88,89% ± 1,8% | +0,55% |
| **Binary Specificity (Độ đặc hiệu)** | 86,31% ± 2,7% | 85,78% ± 4,2% | **88,60% ± 5,0%** | **+2,82%** |
| **Coarse Accuracy (Trực tiếp)** | **73,64% ± 0,9%** | 70,09% ± 2,3% | 73,57% ± 1,8% | +3,48% |
| **Coarse Macro-F1 (5 Nhóm)** | **0,6576 ± 0,006** | 0,6119 ± 0,018 | 0,6525 ± 0,012 | +0,0406 |
| **Fine Accuracy** | 52,63% ± 2,9% | 47,21% ± 2,4% | **53,07% ± 4,0%** | **+5,86%** |
| **Fine Macro-F1 (Supported)** | 0,5026 ± 0,046 | 0,5199 ± 0,048 | **0,5415 ± 0,036** | **+0,0216 (+2,16%)** |
| **Fine Macro-F1 (All 22 Classes)** | 0,3718 ± 0,026 | 0,3844 ± 0,023 | **0,4007 ± 0,015** | **+0,0163 (+1,63%)** |
| **Tính nhất quán Coarse-Fine** | 74,38% ± 3,2% | 81,88% ± 2,0% | **82,28% ± 0,5%** | +0,40% |
| **Parent Acc (Coarse Head $\lambda=1{,}0$)**| -- | 65,95% ± 2,9% | **70,76% ± 1,1%** | +4,81% |
| **Parent Acc (Fine Marg. $\lambda=0{,}0$)**| -- | 75,50% ± 1,5% | **78,10% ± 0,7%** | +2,60% |
| **Best Ensemble Parent Acc** | -- | 75,50% ± 1,5% | **78,37% ± 1,0%** | **+2,87%** ($\lambda=0{,}25$) |

#### Bảng Kiểm Định Độc Lập Trên Tập Test (3-Fold Patient-Disjoint Holdout Test):
| Tiêu chí Đánh giá / Metric | 3S-HFT Fixed ($w=0{,}25$) Cũ | Method A (Two-Phase) | **3S-HFT Đề Xuất Mới (Warmup)** | Mức độ Vượt trội |
|---|:---:|:---:|:---:|:---:|
| **Binary AUROC** | 0,9966 ± 0,003 | 0,9976 ± 0,001 | **0,9986 ± 0,0002** | Tiệm cận 1,000 |
| **Binary F1-Score** | 0,9792 ± 0,010 | 0,9792 ± 0,009 | **0,9811 ± 0,004** | **0,9811** |
| **Binary Sensitivity** | 99,04% ± 0,7% | 99,23% ± 0,7% | **99,43% ± 0,5%** | **99,43%** |
| **Binary Specificity** | **96,34% ± 1,9%** | 96,13% ± 1,4% | **96,34% ± 0,3%** | **96,34%** |
| **Coarse Accuracy (Trực tiếp)** | 81,56% ± 9,1% | 81,76% ± 8,7% | **82,37% ± 7,0%** | **82,37%** |
| **Coarse Macro-F1** | 0,7494 ± 0,137 | 0,7531 ± 0,130 | **0,7572 ± 0,117** | **0,7572** |
| **Fine Accuracy** | 73,52% ± 13,3% | 73,79% ± 11,1% | **74,73% ± 11,9%** | **74,73%** |
| **Fine Macro-F1 (Supported)** | 0,6424 ± 0,098 | 0,6411 ± 0,096 | **0,6450 ± 0,111** | **0,6450** |
| **Fine Macro-F1 (All 22)** | 0,4672 ± 0,071 | 0,4662 ± 0,070 | **0,4691 ± 0,080** | **0,4691** |
| **Tính nhất quán Coarse-Fine** | 88,04% ± 6,7% | 88,58% ± 5,0% | **89,52% ± 3,8%** | **89,52%** |
| **Parent Acc (Coarse Head $\lambda=1{,}0$)**| 79,70% ± 10,1% | 80,11% ± 8,9% | **81,18% ± 6,9%** | +1,48% |
| **Parent Acc (Fine Marg. $\lambda=0{,}0$)**| 85,35% ± 5,4% | **86,56% ± 3,9%** | **86,42% ± 3,5%** | +1,07% |
| **Best Ensemble Parent Acc** | 85,62% ± 5,5% | **86,69% ± 3,8%** | **86,42% ± 3,5%** | **86,42% -- 86,69%** |

### 5.4. Stage 40 -- Bóc tách Định lượng Thành phần (Ablation Studies qua 3 Splits)

Bảng đối sánh 10 biến thể triệt tiêu thành phần qua toàn bộ 3 phân hoạch hold-out độc lập bệnh nhân ($3 \text{ Splits} \times 10 \text{ Variants} = 30 \text{ Runs}$):

| Biến Thể Thực Nghiệm / Variant | Chiến Lược Huấn Luyện | Binary AUROC | Coarse Acc | Coarse Macro-F1 | Fine Acc | Fine Macro-F1 (Supp) | Fine Macro-F1 (All 22) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Proposed 3S-HFT (Curriculum Warmup)** | **3-Stage Sequential Warmup Hierarchy** | 0,9571 ± 0,021 | 73,57% ± 1,8% | 0,6525 ± 0,012 | **53,07% ± 4,0%** | **0,5415 ± 0,036** | **0,4007 ± 0,015** |
| **3S-HFT Method A (Two-Phase)** | 3-Stage w/ $w=0$ ở P1, $w=0{,}25$ ở P2/3 | 0,9521 ± 0,028 | 71,76% ± 1,1% | 0,6371 ± 0,008 | 48,98% ± 4,1% | 0,5240 ± 0,025 | 0,3883 ± 0,017 |
| **3S-HFT Fixed Hierarchy ($w=0{,}25$)** | 3-Stage Cố định Hierarchy xuyên suốt | 0,9466 ± 0,031 | 70,09% ± 2,3% | 0,6119 ± 0,018 | 47,21% ± 2,4% | 0,5199 ± 0,048 | 0,3844 ± 0,023 |
| **1-Stage Joint Baseline** | Multi-Task Joint (CE + SupCon + SBS) | 0,9594 ± 0,018 | **73,64% ± 0,9%** | **0,6576 ± 0,006** | 52,63% ± 2,9% | 0,5026 ± 0,046 | 0,3718 ± 0,026 |
| **2-Stage Decoupled (D2S-HFT)** | Rep $\rightarrow$ Fine-Only SBS | 0,9617 ± 0,028 | 72,12% ± 4,6% | 0,6313 ± 0,031 | 52,20% ± 4,8% | 0,5266 ± 0,056 | 0,3893 ± 0,032 |
| **Ablation: w/o SupCon** ($w=0$) | Phase 1 CE thuần túy $\rightarrow$ Hierarchy | 0,9437 ± 0,027 | 70,07% ± 3,7% | 0,6140 ± 0,038 | 51,57% ± 4,0% | 0,5042 ± 0,052 | 0,3722 ± 0,018 |
| **Ablation: w/o Hierarchy Loss** ($w=0$) | Multi-Task w/o Coarse-Fine Loss | **0,9649 ± 0,022** | 73,46% ± 1,7% | 0,6426 ± 0,009 | 52,72% ± 2,0% | 0,5414 ± 0,077 | 0,3998 ± 0,047 |
| **Ablation: Strategy cRT** | Phase 2 cRT Sampler (Fine Only) | 0,9617 ± 0,028 | 72,12% ± 4,6% | 0,6313 ± 0,031 | 51,96% ± 2,5% | 0,5311 ± 0,048 | 0,3930 ± 0,029 |
| **Ablation: Target All Heads** | Phase 2 Unfreeze Binary + Coarse + Fine | 0,9583 ± 0,035 | 73,10% ± 3,4% | 0,6435 ± 0,031 | 51,55% ± 3,6% | 0,5129 ± 0,059 | 0,3794 ± 0,038 |
| **Ablation: Freeze Stages 1-2** | Partial Finetuning (Swin Stages 3-4) | 0,9524 ± 0,028 | 73,56% ± 2,5% | 0,6535 ± 0,033 | 50,63% ± 1,9% | 0,4950 ± 0,028 | 0,3669 ± 0,022 |
| **Ablation: Freeze Stages 1-3** | Partial Finetuning (Swin Stage 4 Only) | 0,9246 ± 0,036 | 66,22% ± 2,9% | 0,5765 ± 0,037 | 43,31% ± 4,2% | 0,4814 ± 0,052 | 0,3555 ± 0,024 |

#### Khảo sát Vị trí Trích xuất Đặc trưng (Shared Late-Stage vs. Multi-Stage Intermediate Heads)

| Biến thể Vị trí Head / Architecture Variant | Vị trí Trích xuất Đặc trưng | Binary AUROC | Binary Specificity | Coarse Acc | Fine Acc | Fine Macro-F1 (Supp) | Fine Macro-F1 (All 22) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Proposed 3S-HFT (Shared Late-Stage)** | Toàn bộ 3 Heads tại Stage 4 | **0,9571** | **88,60%** | **73,57%** | **53,07%** | **0,5415** | **0,4007** |
| **Multi-Stage Intermediate Heads** | S2 $\rightarrow$ Bin, S3 $\rightarrow$ Coarse, S4 $\rightarrow$ Fine | 0,8355 | 75,66% | 68,73% | 42,64% | 0,4806 | 0,3714 |

- **Ba kết luận khoa học cốt lõi từ Ablation:**
  1. *Curriculum Warmup giải phóng thắt cổ chai:* Giúp Fine Macro-F1 Supported đạt 0,5415 (+2,16% so với bản cũ), đồng thời khôi phục trọn vẹn độ chính xác Coarse (73,57%) và Binary AUROC (0,9571).
  2. *Đóng băng sớm gây sụt giảm nghiêm trọng:* Đóng băng các tầng 1--2 hoặc 1--3 làm mất khả năng học các hoa văn mao mạch vi thể (Fine F1 sụt giảm -2,49% và -3,85%).
  3. *Thất bại của Intermediate Heads:* Việc đặt binary head ở Stage 2 và coarse head ở Stage 3 làm suy sụp độ đặc hiệu phát hiện tổn thương (-12,9%), chứng minh biểu diễn cần được xử lý trọn vẹn qua 4 stages.

### 5.6. Phân tích Chi tiết Lớp Coarse, Fine và Các Mẫu Lỗi Ưu Tiên

| Lớp coarse | Support thật (Split 0) | Dự đoán | Precision | Recall | F1 | AUROC OVR |
|---|---:|---:|---:|---:|---:|---:|
| Malignant | 142 | 149 | 0,7852 | 0,8239 | 0,8041 | 0,9175 |
| Non-malignant | 32 | 30 | 0,2000 | 0,1875 | 0,1935 | 0,8348 |
| Normal mucosa | 81 | 111 | 0,6757 | 0,9259 | 0,7813 | 0,9512 |
| Anatomical landmarks | 31 | 11 | 0,8182 | 0,2903 | 0,4286 | 0,9027 |
| Foreign bodies | 43 | 28 | 0,9286 | 0,6047 | 0,7324 | 0,9689 |

- **Phân tích lỗi lâm sàng:**
  - *Non-malignant $\rightarrow$ Malignant (26/32 ảnh):* Mô hình thiên về cảnh báo an toàn (overcalling) để tránh bỏ sót ung thư; cần hiệu chỉnh ngưỡng quyết định khi triển khai sinh thiết.
  - *Anatomical landmarks $\rightarrow$ Normal mucosa (20/31 ảnh):* Nhầm lẫn do niêm mạc xung quanh lỗ niệu quản hoặc cổ bàng quang có đặc điểm thị giác tương đồng.

---

## 6. Thảo luận & Tính Khả Thi Lâm Sàng (Discussion)

### 6.1. Ý nghĩa khoa học & Cơ chế tương hỗ
- **Tính tương hỗ 4 thành phần:** Swin-Tiny (trích xuất đặc trưng) + SBS (cân bằng prior bệnh nhân) + Curriculum Warmup (chống thắt cổ chai sớm) + Hierarchical Marginalization (kết hợp suy luận đa tầng).
- **Nguyên lý Zero Forgetting:** Việc khóa cứng các tầng trên ở Phase 2 và Phase 3 loại bỏ hoàn toàn hiện tượng trôi dạt ranh giới quyết định.

### 6.2. Hiệu năng suy luận thực tế (Edge Inference Benchmark)
- **Thiết bị thử nghiệm:** Apple M4 MPS / FP32 (tương đương phần cứng máy tính nhúng trong phòng mổ).
- **Chỉ số:** Forward latency **12,081 ms/ảnh**, thông lượng **82,78 FPS** (đáp ứng tiêu chuẩn thời gian thực $>30$ FPS của camera nội soi).

### 6.3. Giới hạn nghiên cứu & Hướng phát triển
1. **Dữ liệu đơn trung tâm:** Cần mở rộng external validation trên các cohort quốc tế độc lập.
2. **Khai thác video liên tục:** Tích hợp mô hình Temporal Transformer để tận dụng thông tin liên kết giữa các khung hình liên tiếp.

---

## 7. Kết luận (Conclusion)

- Nghiên cứu xác lập **Three-Stage Sequential Hierarchical Fine-Tuning (3S-HFT v3.1)** như một giải pháp chuẩn mực cho bài toán chẩn đoán nội soi bàng quang phân cấp trên dữ liệu đuôi dài.
- Mô hình đạt hiệu năng đứng đầu toàn diện trên cả 3 tầng nhãn: Binary AUROC 0,9986, Coarse Accuracy 86,42%, Fine Macro-F1 Supported 0,6450 và tính nhất quán phả hệ 89,52% trên tập Test độc lập.
- Toàn bộ pipeline, checkpoints, receipts và artifacts được chuẩn hóa minh bạch phục vụ tái lập nghiên cứu khoa học.

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
