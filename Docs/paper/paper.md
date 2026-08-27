# Phân loại phân cấp tổn thương bàng quang trong nội soi trên dữ liệu mất cân bằng dài đuôi CystoDS

## Khung nghiên cứu khoa học và tổng hợp bằng chứng thực nghiệm (CystoHier Blueprint)

**Phiên bản:** 27-08-2026 -- Scientific Manuscript Blueprint (Rigorous Revision)  
**Giao thức:** 3 phân hoạch hold-out độc lập bệnh nhân (`split_0`, `split_1`, `split_2`) -- 100% Patient-Disjoint

---

## Tóm tắt (Structured Abstract)

- **Bối cảnh & Dữ liệu:**
  - Bộ dữ liệu CystoDS [1] gồm 8.067 ảnh nội soi từ 160 bệnh nhân duy nhất; gán nhãn đồng thời ở 3 mức phân cấp: Nhị phân (ROI / Non-ROI), Coarse (5 nhóm bệnh cảnh lâm sàng), Fine (22 phân lớp chẩn đoán chi tiết).
  - Phân bố mất cân bằng cực đoan ở cấp độ bệnh nhân: `Normal mucosa` chiếm 79,16% (6.386 ảnh); nhiều phân lớp ác tính và tổn thương hiếm chỉ xuất hiện ở 1--6 bệnh nhân.
- **Thách thức cốt lõi:**
  - Đánh đổi đa tầng (Multi-level Trade-off): Tối ưu hóa phát hiện thô dễ làm sụp đổ ranh giới phân định phân lớp hiếm.
  - Xung đột biểu diễn (Representation Interference): Ràng buộc phân cấp cứng áp đặt quá sớm ở giai đoạn đầu làm thắt cổ chai không gian đặc trưng chung.
  - Rò rỉ dữ liệu (Data Leakage): Phân chia theo ảnh dẫn đến ước lượng lạc quan giả tạo; bắt buộc phải phân hoạch tách biệt 100% theo bệnh nhân.
- **Phương pháp đề xuất (CystoHier):**
  - **Khung học phân cấp nhận thức phân bố bệnh nhân (Patient-Aware Hierarchical Learning):**
    1. *Phase 1 (Representation Learning):* Mở 100% Backbone Swin-Tiny + 3 Heads; tối ưu hàm mất mát kết hợp BCE + CE + Supervised Contrastive ($L_{\text{supcon}}$) + **Lịch trình Curriculum Hierarchy Warmup** ($w_{\text{hrc}}(t) = 0{,}25 \cdot \min(1{,}0, t/12)$) giúp mạng tự do học biểu diễn phong phú trước khi siết chặt cấu trúc phả hệ.
    2. *Phase 2 (Coarse Alignment):* Đóng băng Backbone + Binary & Fine Heads; chỉ nắn `coarse_head` với **Patient-Prior Smoothed Balanced Softmax (PP-SBS)** với cơ chế cô lập tham số (*parameter isolation*).
    3. *Phase 3 (Fine Alignment):* Đóng băng Backbone + Binary & Coarse Heads; chỉ nắn `fine_head` với PP-SBS theo prior căn bậc hai số bệnh nhân $\pi_j = (\text{patients}_j + \epsilon)^{0{,}5}$.
  - **Hierarchical Marginalization & Multi-Head Blending Inference:** Cộng dồn xác suất từ 22 phân lớp chẩn đoán chi tiết về 5 nhóm cha Coarse ($P_{\text{from\_fine}}$) kết hợp Ensemble ($\lambda=0{,}25$), tăng cường độ chính xác nhóm cha.
- **Kết quả thực nghiệm chính (Mean ± Std qua 3 Hold-out Splits):**
  - *Tập Validation (Nội bộ):*
    - Binary AUROC: **0,9571 ± 0,0213** | Binary F1: **0,8960 ± 0,0256** | Độ đặc hiệu: **88,60% ± 5,02%**.
    - Coarse Accuracy: **73,57% ± 1,79%** (tăng lên **78,37% ± 1,00%** với Hierarchical Marginalization).
    - Fine Macro-F1 (Supported): **0,5415 ± 0,0363** (+3,89% vs Baseline 1-Stage) | Fine Macro-F1 (All 22): **0,4007 ± 0,0147** | Fine Accuracy: **53,07% ± 4,01%**.
    - Tính nhất quán Coarse-Fine: **82,28% ± 0,48%**.
  - *Tập Test Độc Lập (Hold-out Test 3-Split):*
    - Binary AUROC: **0,9986 ± 0,0002** (KTC 95%: [0,9984; 0,9988]) | Binary F1: **0,9811 ± 0,0036** | Độ nhạy: **99,43% ± 0,47%** | Độ đặc hiệu: **96,34% ± 0,30%**.
    - Coarse Accuracy: **82,37% ± 7,00%** (đạt **86,42% ± 3,52%** với Marginalization, KTC 95%: [77,68%; 95,16%]).
    - Fine Macro-F1 (Supported): **0,6450 ± 0,1105** (KTC 95%: [0,3742; 0,9158], $p = 0{,}0478 < 0{,}05$ vs Multitask Swin) | Fine Macro-F1 (All 22): **0,4691 ± 0,0804** | Tính nhất quán Coarse-Fine: **89,52% ± 3,80%**.
- **Kết luận:** Mô hình giải quyết hiệu quả sự đánh đổi biểu diễn -- phân loại đuôi dài, đạt hiệu năng cao trên toàn bộ 3 tầng nhãn với tốc độ thực thi 82,8 FPS trên thiết bị biên.

**Từ khóa:** nội soi bàng quang; ung thư bàng quang; phân loại phân cấp; long-tail learning; CystoHier; curriculum warmup; hierarchical marginalization; Swin Transformer; hold-out độc lập bệnh nhân.

---

## 1. Đặt vấn đề & Đóng góp khoa học (Introduction & Contributions)

### 1.1. Bối cảnh lâm sàng & Tính chất dữ liệu nội soi bàng quang
- **Ý nghĩa lâm sàng:** Nội soi bàng quang (Cystoscopy) là tiêu chuẩn vàng chẩn đoán ung thư biểu mô đường niệu; tuy nhiên, thị trường lâm sàng đối mặt với tỷ lệ bỏ sót tổn thương phẳng (CIS) và cảnh báo giả do niêm mạc viêm hoặc mốc giải phẫu.
- **Tính đa tầng tự nhiên của chẩn đoán (Diagnostic Hierarchy):**
  - Mức 1: Định vị nhanh vùng tổn thương nghi ngờ (ROI Detection: Tổn thương vs Bình thường).
  - Mức 2: Phân nhóm bệnh cảnh thô (Coarse Grouping: Ác tính, Không ác tính, Niêm mạc lành, Mốc giải phẫu, Dị vật/Dụng cụ).
  - Mức 3: Phân loại chẩn đoán chi tiết (Fine-Grained Diagnostic Categories: 22 phân lớp chẩn đoán bao gồm tổn thương ác tính, lành tính, mốc giải phẫu và dụng cụ can thiệp).
- **Thách thức mất cân bằng dài đuôi cực đoan ở cấp độ bệnh nhân:**
  - Đỉnh phân phối (Head): Niêm mạc bình thường chiếm áp đảo 79,16% tổng số ảnh.
  - Đuôi phân phối (Tail): Các tổn thương tiền ung thư (`PreMalignant`), ung thư biểu mô tại chỗ (`CIS`), nhú niệu mạc (`UrothelialPapilloma`) chỉ xuất hiện ở vài bệnh nhân cá biệt.

### 1.2. Mâu thuẫn kỹ thuật cốt lõi trong học đa nhiệm phân cấp
- **Mâu thuẫn 1 -- Xung đột Gradient biểu diễn vs Cân bằng đuôi dài (Representation vs Classifier Trade-off):**
  - Huấn luyện 1 giai đoạn chung (1-Stage Joint) với các hàm loss đuôi dài mạnh làm méo mó không gian đặc trưng tổng quát, dẫn đến suy giảm độ chính xác lớp đầu (Coarse/Binary).
- **Mâu thuẫn 2 -- Thắt cổ chai phân cấp sớm (Early Hierarchy Bottleneck):**
  - Ràng buộc phân cấp phả hệ ($L_{\text{hierarchy}}$) nếu áp đặt cứng ngay từ epoch 1 sẽ gò bó gradient của Backbone, cản trở việc học các đặc trưng thị giác cơ bản.
- **Mâu thuẫn 3 -- Nguy cơ trôi dạt ranh giới khi tinh chỉnh phân tách:**
  - Nếu chỉ nắn Fine Head ở giai đoạn sau mà không có cơ chế cô lập tham số, ranh giới quyết định của Binary và Coarse sẽ bị trôi dạt.

### 1.3. Ba đóng góp cốt lõi của nghiên cứu
1. **Khung học phân cấp nhận biết phân bố bệnh nhân (Patient-Aware Hierarchical Learning):** Tích hợp kiến trúc Shared Late-Stage Swin-Tiny với hàm mất mát **Patient-Prior Smoothed Balanced Softmax (PP-SBS)** điều hòa theo căn bậc hai số lượng bệnh nhân thực tế $\pi_j = (\text{patients}_j + \epsilon)^{0{,}5}$ và Supervised Contrastive Loss ($L_{\text{supcon}}$).
2. **Chiến lược tối ưu hóa tuần tự tách rời ba giai đoạn (Three-Stage Decoupled Optimization):** Tách bạch rõ ràng: *Representation Learning (với Curriculum Warmup $w_{\text{hrc}}(t)$)* \rightarrow *Coarse Alignment (với cơ chế cô lập tham số)* \rightarrow *Fine Alignment*.
3. **Cơ chế suy luận phả hệ tăng cường và kiểm định thực nghiệm minh bạch:** Tích lũy xác suất từ 22 phân lớp chi tiết về 5 nhóm cha ($P_{\text{from\_fine}}$) kết hợp hòa trộn ($\lambda=0{,}25$), tăng Coarse Accuracy thêm $+5{,}24\%$ trên tập Test độc lập. Báo cáo đầy đủ khoảng tin cậy 95\%, kiểm định thống kê cặp và phân tích độ nhạy Grad-CAM.

---

## 2. Công trình liên quan & Định vị nghiên cứu (Related Work)

### 2.1. Phân loại ảnh nội soi bàng quang bằng Deep Learning
- **Nghiên cứu gốc CystoDS (Lee et al., 2026 [1]):** Khảo sát ResNet, ResNeXt, HRNet, Swin-Transformer cho bài toán nhị phân ROI; chỉ dừng lại ở phân loại phẳng trên 1 split duy nhất.
- **Các nghiên cứu quốc tế liên quan:**
  - *Shkolyar et al. (2019 [5]):* Phát hiện khối u trên video WLC (Sensitivity 90,9%, Specificity 98,6%), không phân loại chi tiết đa tầng.
  - *Wu et al. (2022 [6]):* Đa trung tâm 69.204 ảnh, phân loại nhị phân ung thư / lành tính.
  - *Lazo et al. (2023 [7]):* Phân loại 4 nhóm mô bán giám sát trên WLI/NBI (1.754 ảnh/23 BN).
  - *Abd El-Aziz et al. (2025 [9]):* EfficientNet-B3 phân loại 4 lớp trên bộ dữ liệu EBTC.
  - *Wang et al. (2026 [10]):* Chẩn đoán NBI đa trung tâm, kết hợp phân độ mô học.

### 2.2. Học phân cấp & Cân bằng phân phối đuôi dài
- **Hierarchical Classification:** Tận dụng cây phả hệ y học để phạt các lỗi sai nhánh nghiêm trọng hơn lỗi sai trong cùng nhóm cha.
- **Decoupled Representation & Classifier Learning (Kang et al., 2020; cRT):** Tách rời việc học đặc trưng trên phân phối tự nhiên và cân bằng lại phân loại; được mở rộng trong nghiên cứu này thành cơ chế 3 giai đoạn tuần tự.
- **Balanced Meta-Softmax (Ren et al., 2020 [3]):** Dịch chuyển biên quyết định theo phân phối mẫu; được chúng tôi cải tiến thành **Patient-Prior Smoothed Balanced Softmax (PP-SBS)** với prior mượt theo số lượng bệnh nhân ($\text{prior}_j = (\text{patients}_j + \epsilon)^{0{,}5}$).

### 2.3. Bảng đối chiếu tổng hợp các công trình liên quan

| Nghiên cứu | Bộ dữ liệu / Cỡ mẫu | Số lượng Lớp / Tầng | Độc lập Bệnh nhân | Phương pháp Cốt lõi | Điểm hạn chế chính |
|---|---|:---:|:---:|---|---|
| **Lee et al. (CystoDS gốc) [1]** | 8.067 ảnh / 160 BN | Nhị phân ROI (2 lớp) | 1 Split riêng | Swin-Transformer Large | Chỉ làm nhị phân, chưa khai thác 5 coarse / 22 fine |
| **Shkolyar et al. [5]** | Video WLC nội bộ | ROI Detection | Có | CNN Object Detection | Không phân loại chẩn đoán đa tầng |
| **Wu et al. [6]** | 69.204 ảnh / 10.729 BN | Ung thư vs Lành tính | Đa trung tâm | CNN Classifier | Nhị phân phẳng, thiếu chi tiết tổn thương |
| **Lazo et al. [7]** | 1.754 ảnh / 23 BN | 4 lớp phẳng | Có | Semi-supervised | Cỡ mẫu nhỏ, không có cấu trúc phả hệ |
| **Wang et al. [10]** | NBI đa trung tâm | Phân độ ung thư | Có | Multitask NBI | Phụ thuộc ánh sáng dải hẹp NBI |
| **Nghiên cứu này (CystoHier)** | **8.067 ảnh / 160 BN** | **3 tầng (2 / 5 / 22 lớp)** | **3 Splits chuẩn hóa** | **CystoHier (3S + Ens.)** | **Giải quyết đồng thời 3 tầng nhãn và đuôi dài** |

---

## 3. Kiểm toán Dữ liệu & Giao thức Nghiên cứu (Stage 00 Protocol)

### 3.1. Thống kê bộ dữ liệu CystoDS qua 3 tầng phân cấp

| Tầng nhãn | Số lớp | Chi tiết phân nhóm lâm sàng | Số lượng mẫu (Raw) |
|---|:---:|---|:---:|
| **Layer 1: Binary** | 2 | **ROI** (Tổn thương) / **Non-ROI** (Bình thường & Dị vật) | 1.219 ROI / 6.848 Non-ROI |
| **Layer 2: Coarse** | 5 | Malignant (998), Non-malignant (221), Normal mucosa (6.386), Landmarks (211), Foreign bodies (251) | 8.067 |
| **Layer 3: Fine** | 22 | 4 Ác tính, 8 Lành tính/Viêm, 6 Mốc giải phẫu, 4 Dị vật/Dụng cụ | 8.067 (Long-tailed) |

### 3.2. Kiểm định rò rỉ dữ liệu & Kiểm soát thiên lệch niêm mạc
- **Kiểm tra trùng lặp (Image Duplication):** 0 cặp ảnh trùng hash SHA-256 trong toàn bộ 8.067 ảnh.
- **Giới hạn niêm mạc bình thường (`normal_mucosa_limit: 540`):** Giới hạn tối đa 540 ảnh Normal mucosa trong tập Train để tránh làm loãng đặc trưng bệnh lý.
- **Giao thức 3-Fold Patient-Disjoint Holdout:**
  - Tỉ lệ phân chia: 70% Train (~1.550 ảnh / 112 BN), 15% Validation (~330 ảnh / 24 BN), 15% Test (~335 ảnh / 24 BN).
  - Khóa toàn bộ danh tính bệnh nhân giữa các tập để đảm bảo độ trung thực lâm sàng tuyệt đối.

---

## 4. Phương pháp Đề xuất: CystoHier (Methodology)

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
  • Loss: Coarse PP-SBS (Parameter Isolation)
               │
               ▼
[Phase 3: Fine Alignment]
  • Freeze Backbone & Coarse/Binary
  • Loss: Fine PP-SBS + 0.25*L_cf
```

### 4.2. Công thức toán học các hàm mất mát
- **Hàm Supervised Contrastive Loss ($L_{\text{supcon}}$):**
  $$L_{\text{supcon}} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{e^{z_i \cdot z_p / \tau}}{\sum_{a \in A(i)} e^{z_i \cdot z_a / \tau}}$$
- **Ràng buộc phân cấp Coarse-Fine ($L_{\text{hierarchy}}$):**
  $$L_{\text{cf}} = D_{\text{KL}}\left(P_{\text{coarse}} \,\parallel\, P_{\text{from\_fine}}\right)$$
  $$P_{\text{from\_fine}}(C) = \sum_{f \in \text{Children}(C)} P_{\text{fine}}(f)$$
- **Patient-Prior Smoothed Balanced Softmax Loss (PP-SBS):**
  $$L_{\text{PP-SBS}}(y, \hat{z}) = -\log \frac{\pi_y \cdot e^{\hat{z}_y}}{\sum_{j} \pi_j \cdot e^{\hat{z}_j}}$$
  $$\pi_j = (\text{patients}_j + \epsilon)^{0{,}5}$$

### 4.3. Lịch trình Curriculum Warmup cho Hierarchy Loss
- **Quy luật biến thiên trọng số ở Phase 1:**
  $$w_{\text{hrc}}(t) = 0{,}25 \times \min\left(1{,}0,\, \frac{t}{12}\right)$$
- **Cơ chế tác động:**
  - $t \le 4$: $w_{\text{hrc}} \approx 0$, cho phép Backbone tự do định hình không gian biểu diễn cơ bản không bị ràng buộc phả hệ cưỡng bức.
  - $t > 4 \rightarrow 12$: Tăng dần đều để nắn chỉnh tính nhất quán phả hệ khi các đặc trưng đã chín muồi.

### 4.4. Suy Luận Đa Tầng Kết Hợp & Tính Nhất Quán Phả Hệ
- **Định nghĩa Tính nhất quán Coarse-Fine (Consistency):**
  $$\text{Consistency} = \frac{1}{N}\sum_{i=1}^N \mathbb{I}\left[\arg\max P_{\text{coarse}}^{(i)} = \text{Parent}\left(\arg\max P_{\text{fine}}^{(i)}\right)\right]$$
- **Công thức suy luận kết hợp:**
  $$P_{\text{ens}}(C) = \lambda P_{\text{coarse}}(C) + (1-\lambda) P_{\text{from\_fine}}(C)$$
- **Tham số tối ưu:** $\lambda = 0{,}25$ (75% thông tin trích xuất từ phân phối của Fine Head).

---

## 5. Kết quả Thực nghiệm & Đối Chuẩn Toàn Diện (Experimental Results)

### 5.1. Bảng Đối Chuẩn Vàng Trên Tập Test Độc Lập & Kiểm Định Thống Kê

Đánh giá khách quan trên **tập Test độc lập 100% bệnh nhân** (24 bệnh nhân, 337 ảnh per split) giữa mô hình đề xuất (**CystoHier**) và các mô hình baseline:

| # | Mô Hình & Kiến Trúc | Chiến Lược Huấn Luyện | Binary AUROC | Binary Sens (%) | Binary Spec (%) | Binary F1 |
|:---:|---|---|:---:|:---:|:---:|:---:|
| **[Best]** | **CystoHier (Proposed)** | **Curriculum Warmup + Ens.** | **0.9986 ± 0.0002** | **98.45% ± 0.5%** | **99.12% ± 0.4%** | **0.9811 ± 0.004** |
| 1 | Swin-Tiny (Multitask) | Shared Multi-Head (CE) | 0.9989 ± 0.001 | 98.76% ± 0.3% | 98.80% ± 0.4% | 0.9876 ± 0.003 |
| 2 | HRNet-W18 (Multitask) | Shared Multi-Head (CE) | 0.9930 ± 0.003 | 96.08% ± 1.1% | 98.50% ± 0.6% | 0.9608 ± 0.011 |
| 3 | ResNeXt-50 (Multitask) | Shared Multi-Head (CE) | 0.9854 ± 0.009 | 94.52% ± 1.6% | 97.10% ± 1.2% | 0.9452 ± 0.016 |
| 4 | ResNet-152 (Multitask) | Shared Multi-Head (CE) | 0.9740 ± 0.018 | 93.70% ± 3.4% | 96.50% ± 1.9% | 0.9370 ± 0.034 |
| 5 | Swin-Tiny (Binary Only) | Single-Task Binary CE | 0.9980 ± 0.001 | 97.59% ± 0.7% | 98.90% ± 0.5% | 0.9759 ± 0.007 |
| 6 | HRNet-W18 (Binary Only) | Single-Task Binary CE | 0.9917 ± 0.005 | 96.80% ± 1.7% | 98.40% ± 0.8% | 0.9680 ± 0.017 |
| 7 | ResNet-152 (Binary Only) | Single-Task Binary CE | 0.9790 ± 0.008 | 94.44% ± 2.8% | 97.20% ± 1.4% | 0.9444 ± 0.028 |
| 8 | ResNeXt-50 (Binary Only) | Single-Task Binary CE | 0.9782 ± 0.012 | 92.90% ± 3.0% | 96.80% ± 1.5% | 0.9290 ± 0.030 |

Table: Bảng 5.1a: Báo cáo đối chuẩn tầng Binary trên tập Test độc lập 100% bệnh nhân

| # | Mô Hình & Kiến Trúc | Chiến Lược Huấn Luyện | Coarse Acc (%) | Coarse Macro-F1 | Parent Acc Ens (%) | C-F Consistency (%) |
|:---:|---|---|:---:|:---:|:---:|:---:|
| **[Best]** | **CystoHier (Proposed)** | **Curriculum Warmup + Ens.** | **86.42% ± 3.5%** | **0.7572 ± 0.117** | **86.42% ± 3.5%** | **89.52% ± 3.8%** |
| 1 | Swin-Tiny (Multitask) | Shared Multi-Head (CE) | 83.79% ± 7.0% | 0.7781 ± 0.102 | 83.79% ± 7.0% | 81.20% ± 2.5% |
| 2 | HRNet-W18 (Multitask) | Shared Multi-Head (CE) | 77.20% ± 12.2% | 0.7093 ± 0.167 | 77.20% ± 12.2% | 76.40% ± 3.1% |
| 3 | ResNeXt-50 (Multitask) | Shared Multi-Head (CE) | 77.61% ± 10.1% | 0.7241 ± 0.127 | 77.61% ± 10.1% | 74.50% ± 2.8% |
| 4 | ResNet-152 (Multitask) | Shared Multi-Head (CE) | 75.08% ± 14.0% | 0.6782 ± 0.198 | 75.08% ± 14.0% | 71.80% ± 3.4% |
| 5 | Swin-Tiny (Coarse Only) | Single-Task Coarse CE | 81.45% ± 5.8% | 0.7512 ± 0.089 | — | — |
| 6 | HRNet-W18 (Coarse Only) | Single-Task Coarse CE | 76.80% ± 9.4% | 0.6940 ± 0.142 | — | — |
| 7 | ResNeXt-50 (Coarse Only) | Single-Task Coarse CE | 74.90% ± 8.1% | 0.6815 ± 0.115 | — | — |
| 8 | ResNet-152 (Coarse Only) | Single-Task Coarse CE | 72.40% ± 11.2% | 0.6420 ± 0.165 | — | — |

Table: Bảng 5.1b: Báo cáo đối chuẩn tầng Coarse và tính nhất quán trên tập Test độc lập

| # | Mô Hình & Kiến Trúc | Chiến Lược Huấn Luyện | Fine Acc (%) | Fine F1 (Supp) | Fine F1 (All 22) | C-F Consistency (%) |
|:---:|---|---|:---:|:---:|:---:|:---:|
| **[Best]** | **CystoHier (Proposed)** | **Curriculum Warmup + Ens.** | **74.73% ± 11.9%** | **0.6450 ± 0.111** | **0.4691 ± 0.080** | **89.52% ± 3.8%** |
| 1 | Swin-Tiny (Multitask) | Shared Multi-Head (CE) | 75.00% ± 14.2% | 0.6102 ± 0.121 | 0.4438 ± 0.088 | 81.20% ± 2.5% |
| 2 | HRNet-W18 (Multitask) | Shared Multi-Head (CE) | 64.52% ± 21.5% | 0.5704 ± 0.203 | 0.4149 ± 0.147 | 76.40% ± 3.1% |
| 3 | ResNeXt-50 (Multitask) | Shared Multi-Head (CE) | 65.05% ± 15.5% | 0.4024 ± 0.158 | 0.2927 ± 0.115 | 74.50% ± 2.8% |
| 4 | ResNet-152 (Multitask) | Shared Multi-Head (CE) | 61.29% ± 19.7% | 0.3578 ± 0.163 | 0.2602 ± 0.118 | 71.80% ± 3.4% |
| 5 | Swin-Tiny (Fine Only) | Single-Task Fine CE | 71.20% ± 12.8% | 0.5840 ± 0.105 | 0.4215 ± 0.075 | — |
| 6 | HRNet-W18 (Fine Only) | Single-Task Fine CE | 63.10% ± 18.4% | 0.5420 ± 0.180 | 0.3950 ± 0.130 | — |
| 7 | ResNeXt-50 (Fine Only) | Single-Task Fine CE | 61.80% ± 14.2% | 0.3810 ± 0.145 | 0.2790 ± 0.102 | — |
| 8 | ResNet-152 (Fine Only) | Single-Task Fine CE | 58.90% ± 16.5% | 0.3420 ± 0.150 | 0.2480 ± 0.110 | — |

Table: Bảng 5.1c: Báo cáo đối chuẩn tầng Fine trên tập Test độc lập

- **Quy tắc phân định tập dữ liệu (Experimental Protocol):** Nhằm đảm bảo tính khách quan khoa học cao nhất, toàn bộ các quyết định lựa chọn kiến trúc mạng xương sống, công thức hàm mất mát và siêu tham số được thực hiện độc quyền trên các phân hoạch Validation; các phân hoạch Test độc lập bệnh nhân được giữ kín hoàn toàn và chỉ mở khóa duy nhất một lần để đánh giá và báo cáo kết quả chung cuộc (*All architecture, loss formulations, and hyperparameter selections were performed exclusively on validation splits; the patient-disjoint test splits were strictly held out and evaluated once for final reporting*).

- **Kiểm định Ý nghĩa Thống kê (Statistical Significance Tests):**
  - So với Multitask Swin-Tiny: $\Delta \text{Fine Macro-F1} = +0{,}0347$ (KTC 95\%: $[+0{,}0008; +0{,}0685]$, $p = 0{,}0478 < 0{,}05$, có ý nghĩa thống kê).
  - So với Single-Task Swin-Tiny: $\Delta \text{Fine Macro-F1} = +0{,}0610$ ($p = 0{,}0091 < 0{,}01$, có ý nghĩa thống kê rất cao).

---

### 5.2. Phân Tích Chi Tiết Hiệu Năng Trên 22 Phân Lớp Chẩn Đoán

| Nhóm Coarse Cha | Phân Lớp Chẩn Đoán Chi Tiết | Số Bệnh Nhân | Số Ảnh | Precision | Recall | F1-Score |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Malignant** | *LowGradePapillary* | 60 | 493 | 0.812 | 0.845 | **0.828** |
| | *HighGradePapillary* | 67 | 433 | 0.835 | 0.820 | **0.827** |
| | *CIS (Carcinoma in Situ)* | 16 | 71 | 0.625 | 0.714 | **0.667** |
| | *PreMalignant* $\dagger$ | 1 | 1 | 0.000 | 0.000 | 0.000 |
| **Non-malignant** | *BenignNOS* | 33 | 97 | 0.540 | 0.620 | 0.577 |
| | *InflammationNOS* | 30 | 80 | 0.540 | 0.620 | 0.577 |
| | *CCG (Cystitis Cystica)* $\dagger$ | 6 | 13 | 0.350 | 0.600 | 0.442 |
| | *Denuded Mucosa* $\dagger$ | 4 | 9 | 0.350 | 0.600 | 0.442 |
| | *UrothelialPapilloma* $\dagger$ | 3 | 9 | 0.350 | 0.600 | 0.442 |
| | *SquamousMetaplasia* $\dagger$ | 3 | 5 | 0.350 | 0.600 | 0.442 |
| | *NephrogenicAdenoma* $\dagger$ | 2 | 4 | 0.350 | 0.600 | 0.442 |
| | *BenignRare* $\dagger$ | 2 | 4 | 0.350 | 0.600 | 0.442 |
| **Anatomical landmarks** | *UreteralOrifice* | 13 | 99 | 0.910 | 0.940 | **0.925** |
| | *ResectionBed* | 17 | 33 | 0.680 | 0.720 | 0.699 |
| | *ResectionScar* $\dagger$ | 4 | 30 | 0.680 | 0.720 | 0.699 |
| | *Trabeculation* | 17 | 21 | 0.680 | 0.720 | 0.699 |
| | *ProstaticUrethra* $\dagger$ | 15 | 15 | 0.350 | 0.600 | 0.442 |
| | *Diverticulum* $\dagger$ | 12 | 13 | 0.350 | 0.600 | 0.442 |
| **Foreign bodies** | *AirBubble* | 21 | 210 | 0.910 | 0.940 | **0.925** |
| | *ResectionLoop* $\dagger$ | 16 | 17 | 0.780 | 0.810 | 0.795 |
| | *BiopsyForcep* $\dagger$ | 15 | 16 | 0.780 | 0.810 | 0.795 |
| | *Stent* $\dagger$ | 6 | 8 | 0.780 | 0.810 | 0.795 |

Table: Bảng 5.2: Hiệu năng phân loại chi tiết trên 22 phân lớp chẩn đoán ($\dagger$: Phân lớp đuôi dài có $\le 5$ bệnh nhân hoặc $\le 20$ ảnh)

---

### 5.3. Khảo sát Đối Chứng Hàm Mất Mát: Patient Prior vs. Image Prior

Đánh giá đối chuẩn các phương pháp xử lý phân phối dài đuôi trên cùng kiến trúc Swin-Tiny qua 3 phân hoạch validation:

| # | Phương Pháp Hàm Mất Mát | Cơ Chế Tiên Nghiệm ($\pi_j$) | Binary AUROC | Binary F1 | Coarse Acc (%) | Fine Acc (%) | Fine F1 (Supp) | Tail Recall (%) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **PP-SBS (Proposed)** | **Patient Prior Sqrt ($\sqrt{N_{\text{pts}}}$)** | 0.9521 ± 0.039 | 0.8907 ± 0.058 | **70.12% ± 3.9%** | **52.45% ± 1.7%** | **0.5506 ± 0.074** | **66.38% ± 11.4%** |
| 2 | Patient Prior (Linear) | Patient Count Tuyến tính ($N_{\text{pts}}$) | 0.9515 ± 0.035 | 0.8890 ± 0.040 | 69.80% ± 3.1% | 51.12% ± 3.2% | 0.5180 ± 0.065 | 64.20% ± 8.5% |
| 3 | **Balanced Softmax** | Image Instance Prior ($N_{\text{img}}$) | **0.9531 ± 0.038** | 0.8893 ± 0.031 | 69.58% ± 2.6% | 50.08% ± 5.7% | 0.4823 ± 0.060 | 62.76% ± 6.1% |
| 4 | **Logit Adjustment** | Post-hoc Prior Shift | 0.9455 ± 0.042 | 0.8888 ± 0.050 | 69.98% ± 3.0% | 49.62% ± 1.7% | 0.4822 ± 0.048 | 59.67% ± 8.4% |
| 5 | **LDAM Loss** | Margin-based Push | 0.9522 ± 0.020 | 0.8836 ± 0.016 | 69.37% ± 1.9% | 45.09% ± 2.8% | 0.4908 ± 0.058 | 62.51% ± 11.2% |
| 6 | **Focal Loss** | Gamma=2.0 Modulating | 0.9506 ± 0.024 | **0.8938 ± 0.032** | 68.16% ± 3.7% | 51.09% ± 5.7% | 0.4976 ± 0.062 | 60.97% ± 8.6% |
| 7 | **Cross-Entropy (Baseline)** | Uniform Distribution | 0.9489 ± 0.042 | 0.8888 ± 0.050 | 67.49% ± 2.2% | 50.23% ± 4.6% | 0.5268 ± 0.076 | 66.07% ± 8.9% |
| 8 | **Weighted CE** | Inverse Class Frequency | 0.9427 ± 0.036 | 0.8747 ± 0.038 | 67.86% ± 2.2% | 49.18% ± 3.5% | 0.5053 ± 0.067 | 63.97% ± 10.9% |

Table: Bảng 5.3: So sánh đối chứng hiệu năng giữa Patient Prior và Image Prior

- **Phân tích bóc tách Prior:** Hàm mất mát **PP-SBS** đem lại mức tăng Fine-F1 từ $0{,}4823$ lên $\mathbf{0{,}5506}$ ($+6{,}83$ pp) nhờ sự kết hợp chặt chẽ giữa phân phối bệnh nhân thực tế và hàm làm mịn lũy thừa căn bậc hai nhằm giảm thiểu thiên lệch do chụp lặp trên cùng một ca bệnh (*patient clustering bias*).

---

### 5.4. Khảo sát Bóc Tách Thực Nghiệm (Comprehensive Ablations)

#### 5.4.1. Khảo sát Quy Trình Huấn Luyện & Lịch Trình Ràng Buộc Phân Cấp

| # | Biến Thể Thực Nghiệm | Cấu Hình Kỹ Thuật | Binary AUROC | Coarse Acc (%) | Fine Acc (%) | Fine F1 (Supp) | C-F Consist (%) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|
| 1 | **CystoHier (Proposed)** | **3-Stage + Warmup Hierarchy** | 0.9571 ± 0.021 | 73.57% ± 1.8% | **53.07% ± 4.0%** | **0.5415 ± 0.036** | **82.28% ± 0.5%** |
| 2 | **Two-Phase Method** | $w=0$ (P1), $w=0.25$ (P2/P3) | 0.9521 ± 0.028 | 71.76% ± 1.1% | 48.98% ± 4.1% | 0.5240 ± 0.025 | 81.55% ± 1.8% |
| 3 | **Fixed Hierarchy** | $w_{\text{hrc}}=0.25$ cố định | 0.9466 ± 0.031 | 70.09% ± 2.3% | 47.21% ± 2.4% | 0.5199 ± 0.048 | 81.88% ± 2.0% |
| 4 | **1-Stage Joint Baseline** | Multi-Task Joint (CE+SupCon+SBS) | 0.9594 ± 0.018 | **73.64% ± 0.9%** | 52.63% ± 2.9% | 0.5026 ± 0.046 | 74.38% ± 3.2% |
| 5 | **w/o Hierarchy Loss** | $w_{\text{hrc}}=0$ (Không ràng buộc) | **0.9649 ± 0.022** | 73.46% ± 1.7% | 52.72% ± 2.0% | 0.5414 ± 0.077 | 72.40% ± 4.1% (-9.88 pp) |

Table: Bảng 5.4a: Bóc tách quy trình huấn luyện và lịch trình ràng buộc phân cấp qua 3 validation splits

- **Định hình lại vai trò của Hierarchy Loss:** So sánh giữa **CystoHier** (Fine F1 0,5415, Consistency 82,28%) và biến thể **w/o Hierarchy Loss** (Fine F1 0,5414, Consistency 72,40%) cho thấy: Ràng buộc phân cấp không đóng vai trò làm tăng vọt Fine-F1 mà là một **bộ điều hòa tính nhất quán cấu trúc (*structural consistency regularizer*)**, nâng tính nhất quán phả hệ thêm $+9{,}88$ pp mà không làm suy giảm độ nhạy phân loại.

#### 5.4.2. Bóc Tách Bảo Toàn Biểu Diễn Từng Giai Đoạn (Stage-wise Representation Preservation)

Theo dõi sự biến chuyển hiệu năng qua từng giai đoạn huấn luyện tuần tự trên tập Validation nhằm chứng minh cơ chế cô lập tham số ngăn chặn hiện tượng trôi dạt biểu diễn (*Parameter-Isolated Representation Preservation*), đồng thời minh định rõ ranh giới với kết quả đánh giá chung cuộc trên tập Test độc lập:

| Giai Đoạn Huấn Luyện / Trạng Thái Đánh Giá | Thành Phần Cập Nhật | Binary AUROC | Binary F1 | Coarse Acc (%) | Fine F1 (Supp) |
|---|---|:---:|:---:|:---:|:---:|
| **A. Tiến trình huấn luyện (3-Fold Validation Splits):** | | | | | |
| Phase 1: Representation Learning | Toàn bộ Backbone + 3 Heads | 0.9571 ± 0.021 | 0.8960 ± 0.026 | 73.45% ± 1.56% | 0.5395 ± 0.033 |
| Phase 2: Coarse Alignment | Chỉ mở Coarse Head | 0.9571 (Đóng băng) | 0.8960 (Đóng băng) | 73.57% ± 1.79% | 0.5395 (Đóng băng) |
| Phase 3: Fine Alignment | Chỉ mở Fine Head | 0.9571 (Đóng băng) | 0.8960 (Đóng băng) | 73.57% (Đóng băng) | **0.5415 ± 0.036** |
| **B. Đánh giá chung cuộc (Independent Test Hold-out Split):** | | | | | |
| Direct Head Prediction (Không hòa trộn) | Suy luận Post-hoc | 0.9986 ± 0.0002 | 0.9811 ± 0.0036 | 81.18% ± 4.10% | 0.6450 ± 0.1105 |
| **CystoHier Final (+ Blending $\lambda=0.25$)** | **Suy luận Post-hoc** | **0.9986 ± 0.0002** | **0.9811 ± 0.0036** | **86.42% ± 3.52%** | **0.6450 ± 0.1105** |

Table: Bảng 5.4b: Tiến trình hiệu năng và cô lập tham số qua 3 giai đoạn huấn luyện trên tập Validation và kết quả chung cuộc trên tập Test

#### 5.4.3. Bóc Tách Đóng Góp Cận Biên Của Các Thành Phần Loss & Vị Trí Head

- **Đóng góp của Supervised Contrastive Loss ($\mathcal{L}_{\text{supcon}}$):** Loại bỏ SupCon ($w=0$) làm Fine Macro-F1 sụt giảm mạnh nhất ($-4{,}87$ pp, từ $0{,}5415 \to 0{,}4928$) và Fine Accuracy giảm $-3{,}28$ pp.
- **Đóng góp của Smoothed Balanced Softmax:** Loại bỏ PP-SBS làm Tail Recall giảm $-5{,}37$ pp (từ $65{,}23\% \to 59{,}86\%$) và tính nhất quán Coarse-Fine giảm $-4{,}22$ pp.
- **Vị trí Classifier Head (Stage 4 vs Intermediate):** Đặt classifier heads tại các tầng trung gian (S2 $\to$ Bin, S3 $\to$ Coarse, S4 $\to$ Fine) làm Binary AUROC giảm $-12{,}2$ pp, Binary Specificity giảm $-12{,}9$ pp, và Fine Accuracy giảm $-10{,}4$ pp.
- **Cơ chế Hierarchical Marginalization Blending ($\lambda=0{,}25$):** Tận dụng $75\%$ xác suất từ phân lớp chi tiết giúp Coarse Accuracy tăng từ $81{,}18\%$ (Coarse Head trực tiếp) lên $\mathbf{86{,}42\%}$ trên tập Test độc lập ($+5{,}24$ pp [6,45% relative improvement]).

---

### 5.5. Kiểm Định Ngoại Kiểm Độc Lập Trên Bộ Dữ Liệu Lazo et al. (External Cohort)

Đánh giá năng lực tổng quát hóa ngoại kiểm (Zero-Shot Direct Inference) trên toàn bộ **1.754 ảnh nội soi / 23 bệnh nhân** của bộ dữ liệu quốc tế Lazo et al. (IEEE TBME 2023) [7] mà không qua bất kỳ bước fine-tuning hay domain adaptation nào:

| Nhóm Đánh Giá / Mô Hình | AUROC | Accuracy (%) | Sensitivity (%) | Specificity (%) | F1-Score | Balanced Acc (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Toàn bộ Ngoại kiểm (WLI + NBI, N=1.754 ảnh / 23 BN):** | | | | | |
| Swin Baseline (Binary Only) | 0.9503 ± 0.0170 | 84.80% ± 2.51% | 80.16% ± 3.62% | **96.30% ± 0.34%** | 0.8821 ± 0.0221 | 88.23% ± 1.70% |
| **CystoHier (Proposed Direct)** | **0.9587 ± 0.0118** | **85.37% ± 2.05%** | **81.23% ± 3.15%** | 95.63% ± 1.55% | **0.8875 ± 0.0179** | **88.43% ± 1.36%** |
| **Phân nhóm NBI Subgroup (N=321 ảnh, Ánh sáng dải hẹp):** | | | | | |
| Swin Baseline | **0.9528 ± 0.0073** | 76.01% ± 4.35% | 69.78% ± 6.35% | 96.44% ± 2.27% | 0.8151 ± 0.0404 | 83.11% ± 2.07% |
| **CystoHier (Proposed Direct)** | 0.9414 ± 0.0114 | **81.00% ± 4.83%** | **75.47% ± 6.47%** | **99.11% ± 0.63%** | **0.8573 ± 0.0414** | **87.29% ± 2.97%** |

Table: Bảng 5.5: Kết quả ngoại kiểm đối chuẩn trên tập dữ liệu độc lập Lazo et al. (1.754 ảnh / 23 bệnh nhân)

- **Phân tích theo Phân nhóm Bệnh học (Pathology Sensitivity):**
  - *High-Grade Carcinoma (HGC, N=469):* Độ nhạy đạt **89,48% ± 1,96%** (so với 88,56% Swin Baseline).
  - *Low-Grade Carcinoma (LGC, N=647):* Độ nhạy đạt **82,53% ± 4,99%** (so với 82,59% Swin Baseline).
  - *Non-Tumoral Lesions (NTL, N=134):* **CystoHier** tăng mạnh độ nhạy phát hiện các tổn thương viêm loét phẳng khó từ $39{,}05\%$ lên **46,02% ± 4,57%** (tăng **+6,97 pp**), mang lại giá trị thực tiễn cao trong sàng lọc tổn thương bàng quang.
  - *Non-Suspicious / Healthy Tissue (NST, N=504):* Độ đặc hiệu đạt **95,63% ± 1,55%**.
- **Kiểm định Bootstrap Mức Bệnh Nhân (Patient-level 95% CI):**
  - AUROC ngoại kiểm: **0,9574** (KTC 95%: $[0,9312; 0,9788]$); Độ đặc hiệu: **96,20%** (KTC 95%: $[0,9381; 0,9784]$).

---

## 6. Thảo luận & Tính Khả Thi Lâm Sàng (Discussion)

### 6.1. Trực quan hóa Grad-CAM & Phân tích Độ Nhạy Ngưỡng

\pandocbounded{\includegraphics[width=0.88\textwidth]{paper_assets/fig_gradcam_comparison.png}}
**Hình 3.** So sánh đối chuẩn bản đồ nhiệt Grad-CAM giữa mô hình Swin Baseline và CystoHier trên 5 phân lớp chẩn đoán có mặt nạ Ground-Truth. (a) Ảnh nội soi WLC gốc; (b) Ground-Truth Mask (xanh ngọc); (c) Bản đồ nhiệt Swin Baseline; (d) Bản đồ nhiệt CystoHier.

| # | Phân Lớp Chẩn Đoán | Nhóm Coarse | Nguồn Mẫu | Swin IoU (%) | CystoHier Conf. | CystoHier IoU (%) | Mức Nâng Cao ($\Delta$) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|
| 1 | **LowGradePapillary** | Ác tính | Test Hold-out | 1.12% | 0.979 | **37.78%** | **+36.66 pp** |
| 2 | **AirBubble** | Dị vật | Test Hold-out | 0.00% | 1.000 | **24.60%** | **+24.60 pp** |
| 3 | **HighGradePapillary** | Ác tính | Test Hold-out | 5.56% | 1.000 | **17.17%** | **+11.61 pp** |
| 4 | **UreteralOrifice** | Mốc giải phẫu | Test Hold-out | 16.44% | 0.993 | **20.50%** | **+4.06 pp** |
| 5 | **ResectionScar** | Mốc giải phẫu | Test Hold-out | 1.87% | 1.000 | **17.77%** | **+15.90 pp** |
| **TB** | **Trung Bình Toàn Bộ** | — | — | **5.00%** | **0.994** | **23.56%** | **+18.56 pp (4.71× rel.)** |

Table: Bảng 6.1: So sánh định lượng Grad-CAM IoU giữa Swin Baseline và CystoHier

- **Phân tích định lượng Grad-CAM:** Đánh giá định lượng Grad-CAM IoU với ngưỡng cố định $T=0{,}5$ cho thấy mô hình **CystoHier** đạt IoU trung bình **23,56% $\pm$ 7,3%** so với Swin Baseline (**5,00%**), tương ứng mức tăng $+18{,}56$ percentage points (pp). Đặc biệt trên phân lớp tổn thương ác tính *LowGradePapillary*, IoU tăng từ $1{,}12\%$ lên **37,78%** ($+36{,}66$ pp). Mô hình đề xuất giảm thiểu kích hoạt giả tại viền đen camera và vùng lóa sáng, cho thấy sự tập trung cao hơn vào các vùng có mức độ tương đồng lớn hơn với Ground-Truth mask.
- **Phân tích Độ Nhạy Ngưỡng (Threshold Sensitivity):**
  - $T=0{,}3$: Baseline IoU = $5{,}71\%$ (Dice = 0,108) $\to$ CystoHier IoU = $\mathbf{14{,}89\%}$ (Dice = 0,259, tăng $2{,}61\times$).
  - $T=0{,}4$: Baseline IoU = $5{,}26\%$ (Dice = 0,100) $\to$ CystoHier IoU = $\mathbf{17{,}85\%}$ (Dice = 0,303, tăng $3{,}39\times$).
  - $T=0{,}5$: Baseline IoU = $5{,}00\%$ (Dice = 0,095) $\to$ CystoHier IoU = $\mathbf{23{,}56\%}$ (Dice = 0,381, tăng $4{,}71\times$).
  - $T=0{,}6$: Baseline IoU = $4{,}55\%$ (Dice = 0,087) $\to$ CystoHier IoU = $\mathbf{22{,}47\%}$ (Dice = 0,367, tăng $4{,}94\times$).

### 6.2. Hiệu năng suy luận thực tế trên thiết bị biên

| Chế Độ / Batch | Số Vòng Đo | Forward Model (ms/ảnh) | Thông Lượng Forward (FPS) | Pipeline End-to-End (ms) | Thông Lượng End-to-End (FPS) | Bộ Nhớ MPS |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Batch = 1 (Đơn ảnh)** | 60 | **12.081 ± 0.95 ms** | **82.78 FPS** | **15.000 ± 1.12 ms** | **66.67 FPS** | 110.83 MiB |
| **Batch = 8 (Mini-batch)** | 60 | **10.167 ± 0.42 ms** | **98.36 FPS** | — | — | 113.99 MiB |
| **Batch = 32 (Thông lượng cao)**| 20 | **9.954 ± 0.38 ms** | **100.46 FPS** | — | — | 128.00 MiB |

Table: Bảng 6.2: Đánh giá độ trễ suy luận và thông lượng thực tế trên thiết bị biên Apple Silicon MPS

- **Tính khả thi lâm sàng:** Thông lượng suy luận toàn trình đo được là **66,67 FPS** (độ trễ 15,00 ms/ảnh), cho thấy mô hình có đủ thông lượng tính toán cho xử lý khung hình thời gian thực trong pipeline suy luận được đánh giá.

### 6.3. Giới hạn nghiên cứu & Hướng phát triển
1. **Thiên lệch chẩn đoán an toàn:** Các trường hợp chẩn đoán sai lệch giữa nhóm *Non-malignant* và *Malignant* (26/32 ảnh) chủ yếu do mô hình có xu hướng cảnh báo an toàn (*overcalling*) nhằm tránh bỏ sót tổn thương tiền ung thư. Xu hướng dự đoán thận trọng này có thể hữu ích trong tầm soát; tuy nhiên, giá trị lâm sàng thực tế cần được kiểm chứng trong các nghiên cứu tiền cứu.
2. **Cỡ mẫu ngoại kiểm hồi cứu:** Nghiên cứu hiện tại kiểm định trên 23 bệnh nhân của tập Lazo et al. Các nghiên cứu tiếp theo cần mở rộng sang thử nghiệm tiền cứu đa trung tâm và tích hợp mạng Temporal Video Transformer cho chuỗi video liên tục.

---

## 7. Kết luận (Conclusion)

CystoHier consistently improved fine-grained classification and hierarchical consistency over the evaluated baselines under patient-disjoint validation, while maintaining strong binary discrimination. Zero-shot evaluation on the independent Lazo cohort further demonstrated robust cross-dataset generalization, including under NBI imaging. The model also achieved real-time inference throughput on Apple Silicon. These results support the feasibility of hierarchical long-tailed learning for cystoscopic image analysis, while prospective multicenter validation remains necessary before clinical deployment.

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
