Bản mô tả chi tiết phương pháp **Multi-Stage Hierarchical Heads** (Phân tách đầu phân loại theo từng tầng của Swin Transformer) dành cho bài toán nội soi bàng quang phân cấp:

---

### 1. Động lực Kỹ thuật & Bản chất Hình ảnh (Intuition & Motivation)

Trong cấu trúc của **Swin Transformer**, mạng xử lý ảnh qua cơ chế phân cấp (hierarchical feature maps) với độ phân giải và dung lượng biểu diễn thay đổi rõ rệt qua từng Stage:

* **Sự không tương thích của Shared-Head truyền thống:**
* Nếu đặt cả 3 tác vụ (*Binary ROI, Coarse 5 nhóm, Fine 22 phân lớp*) tại đầu ra cuối cùng của **Stage 4**, các tác vụ này bị ép phải chia sẻ cùng một vector biểu diễn toàn cục ($768$ chiều).
* Điều này gây ra **xung đột gradient (gradient interference)**: Tác vụ Binary chỉ quan tâm đến các đặc trưng kết cấu bề mặt (vascular pattern, màu sắc niêm mạc), trong khi tác vụ Fine lại cần ngữ nghĩa bệnh học trừu tượng sâu. Gradient của bài toán Binary/Coarse sẽ vô tình làm nhiễu không gian biểu diễn tinh vi của bài toán Fine.


* **Cơ chế khớp nối tri thức theo độ sâu (Depth-Semantic Alignment):**
* **Stage 2 ($H/8 \times W/8$, 192 dims):** Độ phân giải không gian còn lớn, phản ánh rất rõ các cạnh viền, cấu trúc mạch máu và sự thay đổi màu sắc $\rightarrow$ Hoàn hảo cho **Binary ROI Head** (Phân biệt mô lành vs. tổn thương thô).
* **Stage 3 ($H/16 \times W/16$, 384 dims):** Đạt mức cân bằng giữa chi tiết cục bộ và ngữ cảnh vùng $\rightarrow$ Tối ưu cho **Coarse Head** (Phân loại 5 nhóm bệnh cảnh lớn: Khối u, Viêm, Sỏi, Bình thường,...).
* **Stage 4 ($H/32 \times W/32$, 768 dims):** Tầm nhìn bao quát toàn cục (global context) và mức độ trừu tượng cao nhất $\rightarrow$ Tối ưu cho **Fine Head** (22 phân lớp mô bệnh học chi tiết) và **SupCon Embedding**.



---

### 2. Sơ đồ Luồng Dữ liệu (Architecture Flow)

```
Ảnh đầu vào [Batch, 3, 224, 224]
        │
        ├──► Patch Partition & Stage 1 (Embed Dim = 96)
        │
        ├──► Stage 2 (192 dims, 28x28) ──► GAP + LayerNorm(192) ──► Linear(192, 2)  ──► [Binary Logits]
        │        │ (Chỉ lan truyền gradient Binary ngược về Stage 1 & 2)
        │        ▼
        ├──► Stage 3 (384 dims, 14x14) ──► GAP + LayerNorm(384) ──► Linear(384, 5)  ──► [Coarse Logits]
        │        │ (Chỉ lan truyền gradient Coarse ngược về Stage 1, 2, 3)
        │        ▼
        └──► Stage 4 (768 dims, 7x7)   ──► GAP + LayerNorm(768) ──┬► Linear(768, 22) ──► [Fine Logits]
                                                                  └► MLP Projector   ──► [SupCon Embed]

```

---

### 3. Động lực Gradient & Công thức Toán học

Khi phân tách các head theo từng tầng, gradient lan truyền ngược (Backpropagation) được phân rã tự nhiên theo chiều sâu của mạng:

* **Tại Stage 4 ($\theta_{S4}$):** Chỉ tiếp nhận gradient từ tác vụ Fine và SupCon:

$$\nabla_{\theta_{S4}} \mathcal{L}_{\text{total}} = \nabla_{\theta_{S4}} \mathcal{L}_{\text{fine}} + \lambda \nabla_{\theta_{S4}} \mathcal{L}_{\text{supcon}}$$



*(Stage 4 hoàn toàn được giải phóng khỏi áp lực của Binary và Coarse).*
* **Tại Stage 3 ($\theta_{S3}$):** Tiếp nhận gradient từ Coarse kết hợp với luồng lan truyền từ Stage 4:

$$\nabla_{\theta_{S3}} \mathcal{L}_{\text{total}} = \nabla_{\theta_{S3}} \mathcal{L}_{\text{coarse}} + \frac{\partial z_{S4}}{\partial \theta_{S3}} \nabla_{z_{S4}} (\mathcal{L}_{\text{fine}} + \lambda \mathcal{L}_{\text{supcon}})$$


* **Tại Stage 2 ($\theta_{S2}$):** Tiếp nhận thêm gradient từ Binary:

$$\nabla_{\theta_{S2}} \mathcal{L}_{\text{total}} = \nabla_{\theta_{S2}} \mathcal{L}_{\text{binary}} + \text{Grad}_{\text{downstream}}(S3, S4)$$



---

### 4. Lợi ích Cốt lõi của Phương pháp

* **Giữ vững hiệu năng Binary & Coarse:** Tránh việc các tác vụ thô bị "ép" học trên một biểu diễn quá trừu tượng ở Stage 4, giúp phục hồi lại Coarse Accuracy và Binary AUROC.
* **Bảo vệ không gian biểu diễn Fine-grained:** Stage 4 có dung lượng tham số lớn nhất (~10.95M params) sẽ dành trọn vẹn sự chú ý cho 22 phân lớp mô bệnh học đuôi dài và Supervised Contrastive Learning.
* **Không tốn thêm chi phí tính toán đáng kể:** Chỉ bổ sung 2 lớp LayerNorm và Linear nhỏ ở Stage 2 và Stage 3 (tăng dưới 0.1M tham số), tốc độ huấn luyện gần như tương đương mô hình gốc.