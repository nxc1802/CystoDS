Bản mô tả chi tiết phương pháp **Decoupled Two-Stage Fine-Tuning** (Huấn luyện Tách rời 2 Giai đoạn: *Representation Learning* $\rightarrow$ *Classifier Alignment*) dành cho bài toán phân loại phân cấp đuôi dài:

---

### 1. Động lực Kỹ thuật & Bản chất Thất bại của End-to-End (Intuition & Motivation)

Trong bài toán đuôi dài y tế (CystoDS), việc huấn luyện từ đầu đến cuối (**End-to-End Joint Training**) với các kỹ thuật cân bằng (Class-balanced sampling, re-weighting, hoặc nắn Softmax) thường gặp phải một nghịch lý cốt lõi:

* **Sự méo mó không gian biểu diễn (Representation Distortion):**
* Nếu áp dụng lấy mẫu cân bằng (Class-balanced Sampler) hoặc phạt nặng lớp đa số ngay từ đầu, Backbone bị ép phải lặp lại các mẫu hiếm (Tail classes) quá nhiều lần.
* Hậu quả: Backbone bị overfit cục bộ vào các chi tiết nhiễu (noise, artifacts) của lớp hiếm, làm mất đi khả năng trích xuất các đặc trưng tổng quát (general representations) của toàn bộ tập dữ liệu.


* **Sự lệch chuẩn độ lớn vector trọng số (Weight Norm Discrepancy):**
* Trong Linear Classifier ($z = W^T f(x) + b$), vector trọng số $W_k$ của các lớp đa số (Head classes) được cập nhật liên tục qua hàng nghìn bước gradient, dẫn đến độ dài chuẩn $\Vert{}W_{\text{head}}\Vert{} \gg \Vert{}W_{\text{tail}}\Vert{}$.
* Khi tính tích vô hướng $W_k^T f(x)$, mô hình luôn có xu hướng đưa ra điểm số (logits) cao hơn hẳn cho các lớp đa số, trực tiếp làm **sụt giảm nghiêm trọng Tail Recall và Balanced Accuracy**.



```
[Phase 1: Học Đặc trưng Tối ưu]           [Phase 2: Tái cân bằng Ranh giới Quyết định]
Phân phối Tự nhiên + Backbone Mở           Đóng băng Backbone + Hiệu chỉnh Classifier
   (Instance-Balanced / SupCon)                (cRT / Smoothed Balanced Softmax)
             │                                              │
             ▼                                              ▼
   Học Trích xuất Representation                  Cân bằng Norm ||W_k|| giữa
   Khái quát & Không bị méo                      Head classes và Tail classes

```

---

### 2. Quy trình 2 Giai đoạn Chi tiết (Two-Stage Workflow)

#### **Giai đoạn 1: General Representation Learning (Học Biểu diễn Tổng quát)**

* **Mục tiêu:** Tối đa hóa khả năng trích xuất đặc trưng của Backbone và tạo cấu trúc gom cụm ngữ nghĩa trong không gian tiềm ẩn qua Supervised Contrastive Learning (SupCon).
* **Cơ chế & Dữ liệu:**
* **Sampling:** Sử dụng phân phối tự nhiên (Instance-balanced / Random Sampling thông thường).
* **Backbone:** Toàn bộ Backbone + Heads đều mở tham số (`requires_grad = True`).
* **Mục tiêu tối ưu:**

$$\mathcal{L}_{\text{Phase1}} = \mathcal{L}_{\text{CE}}^{\text{bin}} + \mathcal{L}_{\text{CE}}^{\text{coarse}} + \mathcal{L}_{\text{CE}}^{\text{fine}} + \lambda \mathcal{L}_{\text{SupCon}}$$


* *Lưu ý:* Không áp dụng Balanced Softmax hay nắn logits nhân tạo ở giai đoạn này để tránh làm méo gradient truyền về Backbone.


* **Thời lượng:** Chạy 15–20 epochs (hội tụ tự nhiên theo validation loss).

---

#### **Giai đoạn 2: Classifier Re-training & Alignment (Tái cân bằng Đầu phân loại)**

* **Mục tiêu:** Cố định không gian đặc trưng đã học, chỉ xoay và chuẩn hóa lại các siêu phẳng phân chia (decision boundaries) của 3 Classifier Heads để đối xử công bằng với các lớp đuôi dài.
* **Cơ chế & Đóng băng:**
* **Đóng băng Backbone:** Đặt `requires_grad = False` cho 100% Backbone (Swin Stages 1–4 và Patch Embedding).
* **Chỉ mở Classifier Heads:** Chỉ cập nhật tham số của `binary_head`, `coarse_head`, và `fine_head`.
* **Chiến lược Cân bằng (Áp dụng 1 trong 2 cơ chế):**
1. **cRT (Classifier Re-training):** Sử dụng Class-Balanced Sampler để mỗi mini-batch có tỷ lệ mẫu giữa 22 lớp Fine bằng nhau.
2. **Logit-Adjusted Re-balancing:** Giữ DataLoader tự nhiên nhưng kích hoạt **Smoothed Balanced Softmax**:

$$\mathcal{L}_{\text{BSM}} = - \log \frac{n_y^\gamma \cdot e^{z_y}}{\sum_{j=1}^C n_j^\gamma \cdot e^{z_j}}$$



*(Với $n_j$ là số lượng mẫu của lớp $j$, $\gamma \in [0.5, 1.0]$ đóng vai trò làm mượt).*




* **Thời lượng:** Cực ngắn (**5–8 epochs**), tốc độ mỗi epoch chỉ mất vài giây do không cần tính toán backward qua 28M tham số của Swin-Tiny.

---

### 3. Động lực Toán học & Cơ chế Chuẩn hóa Trọng số (Weight Normalization)

Tại sao việc đóng băng Backbone ở Phase 2 lại giải quyết được vấn đề sụt giảm Tail Recall?

Khi Backbone bị đóng băng, feature vector $f(x)$ trở thành một vector hằng số cố định. Quá trình tối ưu ở Phase 2 chỉ thuần túy điều chỉnh góc và độ dài $\Vert{}W_k\Vert{}$ của từng lớp $k$:

* **Triệt tiêu độ lệch Norm:**

$$z_k = \tau \cdot \frac{W_k^T f(x)}{\Vert{}W_k\Vert{}_2 \Vert{}f(x)\Vert{}_2}$$


* Gradient từ các lớp hiếm không thể truyền ngược về làm "vỡ" các bộ lọc của Backbone, mà chỉ tác động trực tiếp lên vector $W_{\text{tail}}$, giúp kéo dài và xoay $W_{\text{tail}}$ về đúng hướng phân bố của cụm dữ liệu hiếm.

---

### 4. Bảng So sánh Hiệu ứng Giữa 1-Stage và Decoupled 2-Stage

| Tiêu chí | Joint 1-Stage (Hiện tại) | Decoupled 2-Stage Fine-Tuning |
| --- | --- | --- |
| **Chất lượng Biểu diễn (Backbone)** | Bị méo mó do vừa học đặc trưng vừa phải gánh phạt mất cân bằng mẫu. | **Tối ưu tối đa** nhờ học trên phân phối tự nhiên không thiên lệch. |
| **Độ dài Trọng số Heads ($\Vert{}W_k\Vert{}$)** | Lệch nghiêm trọng về phía Head classes ($\Vert{}W_{\text{head}}\Vert{} \gg \Vert{}W_{\text{tail}}\Vert{}$). | **Cân bằng đồng đều** giữa Head và Tail classes sau Phase 2. |
| **Tail Recall (Lớp hiếm)** | Tụt dốc mạnh (từ 58.5% xuống 47.0%). | **Phục hồi vượt bậc (>60%)** do ranh giới quyết định được nắn chuẩn. |
| **Coarse & Binary Performance** | Bị suy giảm do xung đột gradient với lớp đuôi dài. | **Bảo toàn nguyên vẹn** hiệu năng đỉnh từ Phase 1. |
| **Tổng Chi phí Tính toán** | Tốn kém nếu phải thử nghiệm nhiều hàm cân bằng. | **Cực thấp:** Phase 2 chỉ tốn <10% tổng thời gian tính toán. |