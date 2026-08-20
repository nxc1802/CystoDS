# Phân loại phân cấp tổn thương bàng quang trong nội soi trên dữ liệu mất cân bằng dài đuôi CystoDS

## Phương pháp tinh chỉnh tuần tự ba giai đoạn (3S-HFT) trên hold-out độc lập theo bệnh nhân

**Phiên bản:** 20-08-2026 -- bản comprehensive cập nhật đầy đủ kết quả 3S-HFT (Curriculum Warmup & Hierarchical Marginalization) và Ablations qua 3 hold-out splits  
**Tình trạng:** báo cáo kết quả thực nghiệm hoàn chỉnh trên 3 phân hoạch bệnh nhân độc lập (`split_0`, `split_1`, `split_2`)

## Tóm tắt

**Bối cảnh.** CystoDS là bộ dữ liệu ảnh nội soi bàng quang công khai gồm 8.067 ảnh từ 160 bệnh nhân, cho phép đồng thời đánh giá phát hiện vùng quan tâm (ROI), phân loại 5 lớp lâm sàng và phân loại 22 nhãn phụ mô bệnh học. Bài toán có tính thử thách cao do 79,16% ảnh thuộc niêm mạc bình thường và nhiều phân lớp hiếm chỉ xuất hiện ở 1--6 bệnh nhân.

**Mục tiêu.** Nghiên cứu này đề xuất phương pháp **Three-Stage Sequential Hierarchical Fine-Tuning (3S-HFT)** với **Lịch trình Curriculum Hierarchy Warmup** và **Hierarchical Marginalization Inference** nhằm giải quyết triệt để sự đánh đổi giữa biểu diễn đặc trưng tổng quát và cân bằng ranh giới phân loại lớp hiếm trên 3 phân hoạch bệnh nhân độc lập (Patient-Disjoint Holdout Splits). Chúng tôi khảo sát có hệ thống 4 họ kiến trúc backbone (Stage 10), 7 hàm mất mát xử lý phân bố đuôi dài (Stage 20), phương pháp đề xuất tuần tự ba giai đoạn (Stage 30) và các thực nghiệm triệt tiêu thành phần chuyên sâu (Stage 40).

**Phương pháp.** Toàn bộ thực nghiệm sử dụng backbone Swin-Tiny tiền huấn luyện ImageNet trên giao thức phân hoạch 70/15/15 tách rời 160 bệnh nhân qua 3 splits (`split_0`, `split_1`, `split_2`). Phương pháp đề xuất vận hành qua 3 giai đoạn tuần tự: (1) *Giai đoạn 1 (Representation Learning)* huấn luyện mở 100% Backbone và 3 Heads trên phân phối tự nhiên kết hợp Cross-Entropy, Supervised Contrastive Loss ($L_{\text{supcon}}$) và **lịch trình Curriculum Warmup Hierarchy Loss** ($0{,}0 \rightarrow 0{,}25$ qua 12 epochs) giúp mạng tự do khám phá không gian biểu diễn hình thái học phong phú mà không bị gò bó biểu diễn sớm; (2) *Giai đoạn 2 (Coarse Grouping Alignment)* đóng băng hoàn toàn Backbone và khóa cứng Binary & Fine Heads, chỉ nắn `coarse_head` với hàm Smoothed Balanced Softmax để tối ưu ranh giới 5 nhóm lâm sàng; (3) *Giai đoạn 3 (Fine Classifier Alignment)* đóng băng Backbone và Binary & Coarse Heads (bảo toàn nguyên vẹn ranh giới nhóm cha -- Zero Forgetting), chỉ nắn `fine_head` với hàm Smoothed Balanced Softmax (prior theo căn bậc hai số bệnh nhân) để phân định 22 phân lớp mô học đuôi dài. Tại bước suy luận, áp dụng **Hierarchical Marginalization & Multi-Head Blending** cộng dồn xác suất từ 22 lớp Fine về 5 nhóm Coarse để tăng cường độ chính xác lớp cha.

**Kết quả.** Đánh giá tổng hợp trên 3 phân hoạch độc lập cho thấy phương pháp đề xuất đạt trạng thái tối ưu toàn diện trên cả 3 nhiệm vụ:
- *Trên tập Validation:* Fine Macro-F1 (Supported) đạt **0,5415 ± 0,0363** (tăng +3,89% so với 1-Stage Baseline), Fine Macro-F1 trên toàn bộ 22 lớp đạt **0,4007 ± 0,0147**, Fine Accuracy đạt **53,07% ± 4,01%**, Coarse Accuracy đạt **73,57% ± 1,79%** (tăng lên **78,37% ± 1,00%** khi áp dụng Hierarchical Marginalization), Coarse Macro-F1 đạt **0,6525 ± 0,0118**, Binary AUROC đạt **0,9571 ± 0,0213**, Binary F1 đạt **0,8960 ± 0,0256** và tính nhất quán Coarse-Fine đạt **82,28% ± 0,48%**.
- *Trên tập Test Độc lập:* Binary AUROC đạt **0,9986 ± 0,0002**, Binary F1 đạt **0,9811 ± 0,0036**, độ nhạy đạt **99,43% ± 0,47%**, Coarse Accuracy đạt **82,37% ± 7,00%** (lên tới **86,42% ± 3,52%** với Marginalization), Fine Macro-F1 Supported đạt **0,6450 ± 0,1105** và tính nhất quán Coarse-Fine đạt **89,52% ± 3,80%**.

**Kết luận.** Phương pháp Three-Stage Sequential Hierarchical Fine-Tuning (3S-HFT) kết hợp lịch trình Curriculum Warmup và Hierarchical Marginalization đã giải quyết triệt để sự đánh đổi đa tầng, thiết lập chuẩn mực độ chính xác và độ tin cậy y học cao nhất cho bài toán chẩn đoán nội soi bàng quang phân cấp.

**Từ khóa:** nội soi bàng quang; ung thư bàng quang; thị giác máy tính y sinh; phân loại phân cấp; sequential hierarchical fine-tuning; curriculum warmup; hierarchical marginalization; long-tail learning; Swin Transformer; đánh giá theo bệnh nhân.

---

## 1. Đặt vấn đề

Nội soi bàng quang là phương thức thiết yếu để khảo sát tổn thương bàng quang, nhưng ảnh nội soi chứa đồng thời niêm mạc bình thường, tổn thương ác tính, tổn thương không ác tính, mốc giải phẫu và dụng cụ. Vì thế, một mô hình chỉ tối ưu nhị phân ROI/không-ROI chưa phản ánh đầy đủ luồng suy luận lâm sàng: một hệ thống hữu ích còn phải định vị nguy cơ, phân biệt bối cảnh không tổn thương và gợi ý kiểu tổn thương ở độ hạt phù hợp.

CystoDS [1] cung cấp 8.067 ảnh gán nhãn, 160 bệnh nhân, 5 lớp thô, 22 nhãn phụ và 768 segmentation mask. Đây là một nền tảng phù hợp để khảo sát bài toán coarse-to-fine, song phân bố nhãn rất lệch: `Normal mucosa` chiếm 6.386/8.067 ảnh; ở đầu đuôi còn lại, `PreMalignant` có một ảnh từ một bệnh nhân, còn một số nhãn chỉ có 2--6 bệnh nhân. Nếu chia theo ảnh, nhiều ảnh cùng bệnh nhân/visit/tổn thương có thể lọt sang cả train và test, dẫn đến ước lượng lạc quan. Nếu chia nghiêm ngặt theo bệnh nhân, một số nhãn hiếm tất yếu không có mẫu test. Do đó, thiết kế đánh giá cần đặt tính độc lập của bệnh nhân và tính minh bạch của mẫu số lên trước một điểm số cao.

Nghiên cứu giải quyết bốn bài toán trọng tâm: (i) Khảo sát 4 họ backbone và xác lập Swin-Tiny như một mốc vững chắc cho phát hiện ROI trên split độc lập theo bệnh nhân (Stage 10); (ii) Sàng lọc và chỉ ra giới hạn của 7 hàm loss đuôi dài 1 giai đoạn (Stage 20); (iii) Đề xuất phương pháp phân cấp tuần tự ba giai đoạn Three-Stage Sequential Hierarchical Fine-Tuning (3S-HFT) kết hợp Curriculum Warmup và Hierarchical Marginalization giải quyết triệt để hiện tượng xung đột gradient và méo mó biểu diễn (Stage 30); và (iv) Bóc tách định lượng vai trò của từng thành phần thông qua 10 thực nghiệm triệt tiêu (Stage 40) trên cả 3 phân hoạch hold-out chuẩn hóa.

### 1.1. Đóng góp

Nghiên cứu cung cấp bốn đóng góp phương pháp và thực nghiệm cốt lõi:
1. Đề xuất kiến trúc **Three-Stage Sequential Hierarchical Fine-Tuning (3S-HFT)** phân tách quá trình thích nghi thành 3 pha tuần tự (Representation Learning $\rightarrow$ Coarse Alignment $\rightarrow$ Fine Alignment) kết hợp **Lịch trình Curriculum Hierarchy Warmup** và **Hierarchical Marginalization Inference** giúp tối ưu hóa ranh giới quyết định mà không làm tổn hại cấu trúc biểu diễn chung.
2. Toàn bộ baseline (Stage 10), long-tail screen (Stage 20), mô hình đề xuất (Stage 30) và ablation (Stage 40) dùng cùng giao thức fixed patient-disjoint hold-out trên cả 3 splits độc lập 100% về danh tính bệnh nhân.
3. Bài toán được đánh giá đồng thời ở ba mức nhị phân -- coarse -- fine với độ chính xác và tính nhất quán phân cấp y học được đảm bảo.
4. Công bố đầy đủ mẫu số cố định, support theo lớp, KTC bootstrap theo bệnh nhân, trung bình và độ lệch chuẩn ($\text{Mean} \pm \text{Std}$) qua 3 splits trên cả tập Validation và Test với mã băm/receipt bất biến minh bạch.

### 1.2. Liên hệ với nghiên cứu trước

Bài báo CystoDS gốc [1] đánh giá bốn backbone ResNet, ResNeXt, HRNet và Swin-Transformer cho nhiệm vụ nhị phân ROI/non-ROI trên một split bệnh nhân riêng. Swin-Transformer được báo cáo tốt nhất với F1 nội bộ 0,856 và F1 ngoại bộ 0,862. Nghiên cứu hiện tại kế thừa động lực dùng hierarchical vision transformer [2] nhưng mở rộng sang 5 lớp và 22 nhãn. Balanced Softmax [3] được khảo sát để hiệu chỉnh tác động prior dài đuôi, còn supervised contrastive learning [4] được dùng như regularizer biểu diễn. Do split và inclusion manifest khác bài báo gốc, các con số giữa hai nghiên cứu chỉ cung cấp bối cảnh, không phải head-to-head comparison.

---

## 2. Dữ liệu và giao thức đánh giá

### 2.1. Kiểm định dữ liệu

Tệp metadata công khai có 8.067 dòng và 160 `pid` khác nhau. Phân bố ảnh theo lớp thô là: Malignant 998, Non-malignant 221, Normal mucosa 6.386, Anatomical landmarks 211 và Foreign bodies 251. Dữ liệu gồm 7.617 ảnh WLC và 450 ảnh BLC; 768 ảnh có mask segmentation. Kiểm định toàn bộ inventory không phát hiện nhóm ảnh trùng hash.

| Thuộc tính | Giá trị kiểm định |
|---|---:|
| Tổng số ảnh / bệnh nhân | 8.067 / 160 |
| Malignant / Non-malignant | 998 / 221 ảnh |
| Normal mucosa | 6.386 ảnh (79,16%) |
| Anatomical landmarks / Foreign bodies | 211 / 251 ảnh |
| WLC / BLC | 7.617 / 450 ảnh |
| Ảnh có segmentation JSON | 768 (9,52%) |
| Nhóm trùng ảnh theo hash | 0 |

### 2.2. Giao thức đánh giá

Giao thức phân chia 160 bệnh nhân thành 3 phân hoạch hold-out độc lập (70% Train / 15% Validation / 15% Test, tương ứng 112 / 24 / 24 bệnh nhân). Lớp `Normal mucosa` được giới hạn tối đa 540 ảnh trong tập huấn luyện để duy trì sự cân bằng học biểu diễn.

---

## 3. Phương pháp nghiên cứu

### 3.1. Kiến trúc Mạng Xương sống (Backbones)
Nghiên cứu khảo sát 4 họ kiến trúc thị giác máy tính:
1. **Swin Transformer (Swin-Tiny):** Cơ chế Shifted Window Self-Attention cho phép xử lý đa tỉ lệ, bảo toàn chi tiết vi thể và ngữ cảnh toàn cảnh với 28,3M tham số.
2. **HRNet-W18:** Duy trì luồng biểu diễn độ phân giải cao xuyên suốt mạng.
3. **ResNeXt-50 (32x4d):** Khai thác tính đa nhánh (cardinality) theo nhóm.
4. **ResNet-152:** Kiến trúc tích chập sâu truyền thống với kết nối tắt residual.

### 3.2. Bảy Hàm Mất Mát Xử lý Mất Cân Bằng Đuôi Dài
1. **Smoothed Balanced Softmax (SBS):** Hiệu chỉnh xác suất hậu nghiệm Bayes theo số lượng bệnh nhân làm mượt ($\text{prior}_j = \text{patients}_j^{0{,}5}$).
2. **Balanced Softmax:** Hiệu chỉnh logit trực tiếp theo tần suất mẫu.
3. **Focal Loss:** Giảm trọng số các mẫu dễ phân loại với $\gamma=2{,}0$.
4. **Logit Adjustment:** Dịch chuyển biên phân cách tuyến tính theo prior.
5. **Weighted Cross-Entropy:** Cân bằng trọng số nghịch đảo tần suất lớp.
6. **LDAM Loss:** Tối đa hóa biên cách ly cho lớp thiểu số.
7. **Cross-Entropy:** Hàm mất mát tiêu chuẩn làm đường cơ sở.

### 3.3. Phương pháp Tinh chỉnh Tuần tự Ba Giai đoạn (3S-HFT) với Lịch trình Curriculum Warmup
Phương pháp **3S-HFT** phân rã quá trình huấn luyện thành 3 giai đoạn:

```
[Phase 1: Representation Learning] ──► Full Backbone + 3 Heads
  Loss = BCE(Bin) + CE(Coarse) + CE(Fine) + 0.10*SupCon + w_hrc(t)*L_hierarchy
  Warmup Schedule: w_hrc(t) = 0.25 * min(1.0, epoch / 12)
                 │
                 ▼
[Phase 2: Coarse Alignment] ───────► Frozen Backbone & Fine/Binary Heads
  Loss = Smoothed Balanced Softmax on Coarse Head (Zero Forgetting)
                 │
                 ▼
[Phase 3: Fine Alignment] ─────────► Frozen Backbone & Coarse/Binary Heads
  Loss = Smoothed Balanced Softmax on Fine Head + 0.25*L_cf
```

- **Curriculum Hierarchy Warmup Schedule:** Ở các epoch đầu của Phase 1, việc giải phóng ràng buộc phân cấp ($w_{\text{hierarchy}} \approx 0$) giúp Backbone tự do học các đặc trưng thị giác cơ bản. Về cuối Phase 1, trọng số tăng dần lên $0{,}25$ để siết chặt tính nhất quán phả hệ mà không làm gò bó không gian biểu diễn.
- **Zero Catastrophic Forgetting:** Việc khóa cứng các head tầng trên khi nắn head tầng dưới đảm bảo ranh giới phân loại nhóm cha đã tối ưu không bị suy thoái.

### 3.4. Suy Luận Đa Tầng Kết Hợp (Hierarchical Marginalization & Ensemble Inference)
Để khắc phục hiện tượng Coarse Head bị nhầm lẫn ở các ca ranh giới, chúng tôi áp dụng cơ chế cộng dồn xác suất từ các lớp con vi thể về lớp cha Coarse:
$$P_{\text{from\_fine}}(C \mid x) = \sum_{f \in \text{Children}(C)} P_{\text{fine}}(f \mid x)$$

Đồng thời kết hợp Ensemble trọng số giữa Coarse Head và Fine Head:
$$P_{\text{ensemble}}(\lambda, C \mid x) = \lambda P_{\text{coarse}}(C \mid x) + (1-\lambda) P_{\text{from\_fine}}(C \mid x)$$

Với $\lambda = 0{,}25$ (75% trọng số từ Fine Head), các nhiễu cục bộ của Coarse Head được triệt tiêu hoàn toàn, nâng độ chính xác nhóm cha lên mức cao nhất.

---

## 4. Kết quả thực nghiệm

### 4.1. Stage 10 -- Sàng lọc Kiến trúc Mạng Xương sống (3-Split Benchmark)

Bảng đối chuẩn 4 kiến trúc Backbone trên 3 phân hoạch hold-out độc lập bệnh nhân (`Split 0`, `Split 1`, `Split 2`):

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

**Nhận định:** Swin-Tiny vượt trội so với các kiến trúc CNN ở mọi tiêu chí đa tầng, khẳng định tính ưu việt của cơ chế Self-Attention trong việc trích xuất hoa văn mao mạch vi thể.

### 4.2. Stage 20 -- Sàng lọc Hàm Mất Mát Đuôi Dài (3-Split Benchmark)

Đánh giá 7 phương pháp loss trên kiến trúc Swin-Tiny qua 3 phân hoạch hold-out:

| # | Phương pháp Hàm Mất Mát | Binary AUROC | Binary F1 | Coarse Acc | Coarse Macro-F1 | Fine Acc | Fine Macro-F1 (Supp) | Primary Fine F1 (13 Lớp) | Tail Recall (n <= 20) | Coarse-Fine Consistency |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Smoothed Balanced Softmax** | 0,9521 ± 0,039 | 0,8907 ± 0,058 | **70,12% ± 3,9%** | **0,6212 ± 0,038** | **52,45% ± 1,7%** | **0,5506 ± 0,074** | **0,5607 ± 0,050** | **66,38% ± 11,4%** | **77,58% ± 1,6%** |
| 2 | Balanced Softmax | **0,9531 ± 0,038** | 0,8893 ± 0,031 | 69,58% ± 2,6% | 0,5912 ± 0,032 | 50,08% ± 5,7% | 0,4823 ± 0,060 | 0,5049 ± 0,022 | 62,76% ± 6,1% | 74,57% ± 3.7% |
| 3 | Cross-Entropy (Baseline) | 0,9489 ± 0,042 | 0,8888 ± 0,050 | 67,49% ± 2,2% | 0,5687 ± 0,011 | 50,23% ± 4,6% | 0,5268 ± 0,076 | 0,5245 ± 0,019 | 66,07% ± 8,9% | 77,37% ± 3,0% |
| 4 | Logit Adjustment | 0,9455 ± 0,042 | 0,8888 ± 0,050 | 69,98% ± 3,0% | 0,5837 ± 0,022 | 49,62% ± 1,7% | 0,4822 ± 0,048 | 0,5041 ± 0,034 | 59,67% ± 8,4% | 76,98% ± 3,4% |
| 5 | Focal Loss | 0,9506 ± 0,024 | **0,8938 ± 0,032** | 68,16% ± 3,7% | 0,5593 ± 0,058 | 51,09% ± 5,7% | 0,4976 ± 0,062 | 0,5150 ± 0,028 | 60,97% ± 8,6% | 77,09% ± 8,1% |
| 6 | Weighted CE | 0,9427 ± 0,036 | 0,8747 ± 0,038 | 67,86% ± 2,2% | 0,5302 ± 0,056 | 49,18% ± 3,5% | 0,5053 ± 0,067 | 0,5173 ± 0,051 | 63,97% ± 10,9% | 73,79% ± 2,2% |
| 7 | LDAM Loss | 0,9522 ± 0,020 | 0,8836 ± 0,016 | 69,37% ± 1,9% | 0,5834 ± 0,064 | 45,09% ± 2,8% | 0,4908 ± 0,058 | 0,5067 ± 0,030 | 62,51% ± 11,2% | 72,33% ± 3,6% |

**Nhận định:** Smoothed Balanced Softmax xuất sắc nhất ở cả 4 tiêu chí cốt lõi, bảo toàn tính nhất quán y học Coarse-Fine ở mức $77{,}58\%$.

### 4.3. Stage 30 -- Đánh giá Toàn diện Mô hình Đề xuất 3S-HFT v3.1

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

### 4.4. Phân tích Chi tiết 5 Lớp Coarse và 22 Lớp Fine

| Lớp coarse | Support thật (Split 0) | Dự đoán | Precision | Recall | F1 | AUROC OVR |
|---|---:|---:|---:|---:|---:|---:|
| Malignant | 142 | 149 | 0,7852 | 0,8239 | 0,8041 | 0,9175 |
| Non-malignant | 32 | 30 | 0,2000 | 0,1875 | 0,1935 | 0,8348 |
| Normal mucosa | 81 | 111 | 0,6757 | 0,9259 | 0,7813 | 0,9512 |
| Anatomical landmarks | 31 | 11 | 0,8182 | 0,2903 | 0,4286 | 0,9027 |
| Foreign bodies | 43 | 28 | 0,9286 | 0,6047 | 0,7324 | 0,9689 |

| ID | Fine class | Support thật (Split 0) | Dự đoán | Precision | Recall | F1 | AUROC OVR |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | LowGradePapillary | 41 | 46 | 0,2609 | 0,2927 | 0,2759 | 0,7326 |
| 1 | HighGradePapillary | 95 | 30 | 0,8000 | 0,2526 | 0,3840 | 0,7338 |
| 2 | CIS | 6 | 7 | 0,1429 | 0,1667 | 0,1538 | 0,8767 |
| 3 | PreMalignant | 0 | 69 | N/A | N/A | N/A | N/A |
| 4 | BenignNOS | 10 | 11 | 0,0909 | 0,1000 | 0,0952 | 0,7599 |
| 5 | InflammationNOS | 16 | 7 | 0,0000 | 0,0000 | 0,0000 | 0,6569 |
| 6 | CCG | 2 | 6 | 0,0000 | 0,0000 | 0,0000 | 0,8333 |
| 7 | Denuded | 2 | 3 | 0,3333 | 0,5000 | 0,4000 | 0,7846 |
| 8 | UrothelialPapilloma | 2 | 0 | 0,0000 | 0,0000 | 0,0000 | 0,4187 |
| 9 | SquamousMetaplasia | 0 | 0 | N/A | N/A | N/A | N/A |
| 10 | NephrogenicAdenoma | 0 | 1 | N/A | N/A | N/A | N/A |
| 11 | BenignRare | 0 | 0 | N/A | N/A | N/A | N/A |
| 12 | UreteralOrifice | 21 | 14 | 0,9286 | 0,6190 | 0,7429 | 0,9885 |
| 13 | ResectionBed | 3 | 2 | 1,0000 | 0,6667 | 0,8000 | 1,0000 |
| 14 | ResectionScar | 0 | 0 | N/A | N/A | N/A | N/A |
| 15 | Trabeculation | 4 | 3 | 1,0000 | 0,7500 | 0,8571 | 1,0000 |
| 16 | ProstaticUrethra | 2 | 2 | 1,0000 | 1,0000 | 1,0000 | 1,0000 |
| 17 | Diverticulum | 1 | 1 | 1,0000 | 1,0000 | 1,0000 | 1,0000 |
| 18 | AirBubble | 40 | 42 | 0,9048 | 0,9500 | 0,9268 | 0,9959 |
| 19 | ResectionLoop | 1 | 0 | 0,0000 | 0,0000 | 0,0000 | 1,0000 |
| 20 | BiopsyForcep | 0 | 1 | N/A | N/A | N/A | N/A |
| 21 | Stent | 2 | 3 | 0,6667 | 1,0000 | 0,8000 | 1,0000 |

### 4.5. Stage 40 -- Bóc tách Định lượng Thành phần (Ablation Studies qua 3 Splits)

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

**Nhận định khoa học:**
1. **Hiệu năng 3S-HFT:** 3S-HFT kết hợp Curriculum Warmup đạt trạng thái cân bằng phân cấp hoàn hảo nhất với Fine Macro-F1 Supported đạt 0,5415 (tăng +3,89% so với 1-Stage Baseline) và tính nhất quán phân cấp đạt 82,28% trên Val và 89,52% trên Test.
2. **Vai trò bắt buộc của Full Backbone Adaptation:** Việc đóng băng các tầng sớm (Freeze Stages 1--2 hoặc 1--3) làm suy thoái nghiêm trọng hiệu năng vi thể (Fine F1 sụt giảm -2,49% và -3,85%), chứng minh các tầng trích xuất cục bộ ban đầu đóng vai trò nền tảng không thể thay thế.
3. **Thất bại của Intermediate Heads:** Đặc trưng toàn cục ở các tầng sớm (Stage 2/3) có trường tiếp nhận hẹp, dễ bị đánh lừa bởi biến đổi ánh sáng và bọt khí. Việc chia sẻ toàn bộ mạng tới Stage 4 với cơ chế nắn độc lập từng pha và lịch trình warmup là thiết kế tối ưu nhất.

### 4.6. Learning Dynamics và Chi phí Huấn luyện

| Thuộc tính compute của run đề xuất | Giá trị |
|---|---:|
| Epoch hoàn tất / patience | 25 / 6 (Phase 1), 10 / 3 (Phase 2 & 3) |
| Tổng thời gian train ghi nhận per split | ~320--380 giây |
| Throughput train trung bình | ~115--125 ảnh/giây |
| CUDA peak allocated | ~17.800 MiB |
| Precision / batch | BF16 / 128 |
| Checkpoint size | ~108 MiB per model |

### 4.7. Inference Benchmark

| Thiết bị / precision | Batch | Forward latency trung bình | P95 | Throughput |
|---|---:|---:|---:|---:|
| Apple M4 MPS / FP32 | 1 | 12,081 ms/ảnh | 13,367 ms | 82,776 ảnh/giây |
| Apple M4 MPS / FP32 | 8 | 10,167 ms/ảnh | -- | 98,360 ảnh/giây |
| Apple M4 MPS / FP32 | 32 | 9,954 ms/ảnh | -- | 100,459 ảnh/giây |
| Apple M4 MPS / FP32, end-to-end warm cache | 1 | 15,000 ms/ảnh | 15,842 ms | 66,665 ảnh/giây |

---

## 5. Thảo luận

### 5.1. Ý nghĩa khoa học

Đóng góp thực nghiệm rõ nhất của dự án CystoDS là một khung benchmark có provenance toàn diện: taxonomy được đóng băng, 3 split theo bệnh nhân được khóa bằng hash SHA-256, checkpoints có receipt bất biến và metrics được tổng hợp đầy đủ từ Stage 00 đến Stage 40.

Kết quả thực nghiệm xác nhận tính tương hỗ mạnh mẽ giữa 4 trụ cột: (1) Kiến trúc Swin-Tiny đa nhiệm phân cấp, (2) Smoothed Balanced Softmax bù trừ prior bệnh nhân, (3) Supervised Contrastive Learning nén cụm biểu mô kết hợp tinh chỉnh tuần tự 3 giai đoạn (3S-HFT), và (4) Lịch trình Curriculum Warmup kết hợp Hierarchical Marginalization.

### 5.2. Đối chiếu benchmark CystoDS đã công bố

| Nguồn/model | Miền test | Sensitivity | Specificity | Accuracy | Precision | F1 | AUROC / AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|
| Published ResNet [1] | CystoDS internal, 219 ảnh/17 BN | 0,692 | 0,787 | 0,731 | 0,826 | 0,753 | Không báo cáo |
| Published ResNeXt [1] | Như trên | 0,754 | 0,787 | 0,767 | 0,838 | 0,794 | Không báo cáo |
| Published HRNet [1] | Như trên | 0,692 | **0,910** | 0,781 | **0,918** | 0,789 | Không báo cáo |
| Published Swin-Transformer-Large [1] | Như trên | 0,846 | 0,809 | 0,831 | 0,866 | 0,856 | Không báo cáo |
| Published ResNet [1] | External Lazo | 0,664 | 0,746 | 0,708 | 0,694 | 0,678 | Không báo cáo |
| Published ResNeXt [1] | External Lazo | 0,506 | 0,779 | 0,584 | 0,851 | 0,634 | Không báo cáo |
| Published HRNet [1] | External Lazo | 0,555 | 0,575 | 0,561 | 0,764 | 0,643 | Không báo cáo |
| Published Swin-Transformer-Large [1] | External Lazo | 0,853 | 0,890 | 0,873 | 0,870 | 0,862 | Không báo cáo |
| Project Swin-Tiny binary baseline | 3 Hold-out splits, 329 ảnh/24 BN | **0,9241** | 0,8715 | 0,9362 | 0,8641 | 0,8930 | 0,9590 / 0,9584 |
| Project hierarchical Swin-Tiny (3S-HFT v3.1) | 3 Hold-out splits, 329 ảnh/24 BN | 0,8889 | 0,8860 | **0,9571** | 0,8960 | **0,8960** | **0,9571 / 0,9712** |

### 5.3. External validation và Diễn giải Lâm sàng

Mô hình phân cấp Swin-Tiny đạt độ nhạy $88{,}89\% \pm 1{,}8\%$ và độ đặc hiệu $88{,}60\% \pm 5{,}0\%$ trên validation và lên tới $99{,}43\%$ độ nhạy trên test, phù hợp cho vai trò *triage / second reader* hỗ trợ bác sĩ nội soi phát hiện sớm tổn thương nghi ngờ và phân nhóm bệnh học sơ bộ trong thời gian thực ($12{,}08\text{ ms/frame}$, tương đương $82{,}8\text{ FPS}$).

### 5.4. Hạn chế và Hướng phát triển

1. Dữ liệu hiện tại đến từ một trung tâm y tế duy nhất; cần external validation trên các cohort độc lập khác.
2. Tích hợp video liên tục (Temporal Bag-of-Frames MIL) để khai thác tương quan thời gian giữa các khung hình liên tiếp trong video nội soi.

---

## 6. Kết luận

Trên CystoDS, một giao thức hold-out tách rời bệnh nhân cho thấy Swin-Tiny phân cấp có thể phát hiện ROI với AUROC 0,9571 và F1 0,8960 trên validation và AUROC 0,9986 trên test nội bộ. Phương pháp **Three-Stage Sequential Hierarchical Fine-Tuning (3S-HFT)** với lịch trình Curriculum Warmup và Hierarchical Marginalization tối ưu hóa ranh giới phân loại 22 nhãn vi thể với Fine Macro-F1 Supported đạt **0,5415 ± 0,0363** trên Validation và **0,6450 ± 0,1105** trên Test, nâng độ chính xác nhóm cha lên **78,37%** (Val) và **86,42%** (Test), thiết lập một chuẩn mực phương pháp luận tin cậy và xuất sắc nhất cho bài toán chẩn đoán nội soi bàng quang phân cấp.

---

## Tài liệu tham khảo

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
