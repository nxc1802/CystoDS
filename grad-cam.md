Đúng, thông tin này làm protocol rõ hơn: **Swin Baseline là single-task fine-level classifier**, nên Grad-CAM của Swin và Proposed đều sẽ được thực hiện **trực tiếp trên fine-level prediction**. Không cần bàn tới binary/coarse output của baseline nữa.

## Grad-CAM Experiment — Final Protocol

### 1. Mục tiêu

Đánh giá liệu **Proposed Model** có tập trung vào vùng tương ứng với **ground-truth detection mask** tốt hơn **Swin Baseline** hay không.

Hai model:

* **Swin Baseline:** single-task classification ở **fine level**
* **Proposed Model:** model đề xuất

Grad-CAM được sử dụng như một **localization/explainability analysis**, không phải một experiment classification riêng.

---

## 2. Fine classes được lựa chọn

Chỉ sử dụng **5 fine classes**:

| Fine class             | Coarse category      | Samples with mask |
| ---------------------- | -------------------- | ----------------: |
| **LowGradePapillary**  | Malignant            |               262 |
| **AirBubble**          | Foreign bodies       |               210 |
| **HighGradePapillary** | Malignant            |               161 |
| **UreteralOrifice**    | Anatomical landmarks |                99 |
| **ResectionScar**      | Anatomical landmarks |                30 |

Tổng cộng:

[
262+210+161+99+30=\mathbf{762}
]

masked samples.

Việc chọn 5 class này nhằm tạo một tập representative gồm:

* 2 malignant classes
* 1 foreign-body class
* 2 anatomical-landmark classes

Không cần thực hiện Grad-CAM trên binary hoặc coarse level.

---

# 3. Sample selection

Với mỗi fine class:

> Chọn **1 True Positive (TP) sample**.

Tổng cộng:

[
5\ classes \times 1\ TP = \mathbf{5\ images}
]

Ví dụ:

```text
LowGradePapillary      → TP #1
AirBubble              → TP #2
HighGradePapillary     → TP #3
UreteralOrifice        → TP #4
ResectionScar          → TP #5
```

Điều kiện TP:

[
Ground\ Truth = Prediction = Fine\ Class
]

Quan trọng:

> **Cùng một image phải được sử dụng cho Swin và Proposed.**

Sample phải được xác định **trước khi xem Grad-CAM**, tránh selection bias.

---

# 4. Target của Grad-CAM

Vì **Swin là single-task fine-level classifier**, target của Grad-CAM chính là **fine class prediction**.

Ví dụ:

```text
Image
  ↓
Swin
  ↓
LowGradePapillary
  ↓
Grad-CAM target:
LowGradePapillary
```

Không sử dụng:

```text
Malignant
Abnormal
```

làm target cho Swin.

Tương tự:

| Image              | Grad-CAM target    |
| ------------------ | ------------------ |
| LowGradePapillary  | LowGradePapillary  |
| AirBubble          | AirBubble          |
| HighGradePapillary | HighGradePapillary |
| UreteralOrifice    | UreteralOrifice    |
| ResectionScar      | ResectionScar      |

Nếu Proposed model cũng có fine-level classification output, sử dụng **chính fine-level target tương ứng** để comparison trực tiếp.

---

# 5. Grad-CAM implementation

### Swin Baseline

Swin Transformer là backbone của baseline:

```text
Image
 ↓
Swin Backbone
 ↓
Final spatial feature representation
 ↓
Classification Head
 ↓
Fine-class logits
```

Grad-CAM được lấy từ **feature representation ở stage cuối của backbone trước classification head**, với target là fine-level class logit.

### Proposed Model

Thực hiện tương tự:

```text
Image
 ↓
Proposed Model
 ↓
Relevant spatial feature layer
 ↓
Fine-class prediction
 ↓
Grad-CAM
```

Nếu architecture của Proposed khác Swin, không nhất thiết phải dùng đúng layer index; cần chọn **spatial feature layer cuối phù hợp** của từng model.

Điểm quan trọng là **protocol giống nhau**:

* cùng input image
* cùng target class
* cùng normalization
* cùng resize
* cùng cách normalize Grad-CAM
* cùng thresholding strategy

---

# 6. Visualization

Figure chính gồm **5 rows × 4 columns**:

| Class              | Original | GT Mask | Swin Grad-CAM | Proposed Grad-CAM |
| ------------------ | -------- | ------- | ------------- | ----------------- |
| LowGradePapillary  | ✓        | ✓       | ✓             | ✓                 |
| AirBubble          | ✓        | ✓       | ✓             | ✓                 |
| HighGradePapillary | ✓        | ✓       | ✓             | ✓                 |
| UreteralOrifice    | ✓        | ✓       | ✓             | ✓                 |
| ResectionScar      | ✓        | ✓       | ✓             | ✓                 |

Đây là figure quan trọng nhất của experiment.

Mục tiêu trực quan:

> So sánh vùng mà **Swin** và **Proposed** sử dụng để đưa ra đúng fine-level prediction với **GT mask**.

---

# 7. Quantitative comparison

Vì dataset đã có mask, nên thêm **Grad-CAM IoU**.

Pipeline:

```text
Grad-CAM heatmap
       ↓
Normalize [0, 1]
       ↓
Fixed threshold
       ↓
Binary activation map
       ↓
Compare with GT mask
       ↓
IoU
```

[
IoU =
\frac{|CAM\cap Mask|}
{|CAM\cup Mask|}
]

Tính cho từng sample:

| Fine class         | Swin IoU | Proposed IoU |
| ------------------ | -------: | -----------: |
| LowGradePapillary  |        X |            X |
| AirBubble          |        X |            X |
| HighGradePapillary |        X |            X |
| UreteralOrifice    |        X |            X |
| ResectionScar      |        X |            X |

Sau đó có thể report:

| Model         | Mean Grad-CAM IoU |
| ------------- | ----------------: |
| Swin Baseline |                 X |
| **Proposed**  |             **X** |

**Không cần thêm Dice, Pointing Game, Pixel AP, PRO...**

---

# 8. Threshold

Cần cố định cách chuyển heatmap thành binary mask.

Ví dụ:

[
CAM_{binary}(x,y)
=================

\mathbb{1}[CAM(x,y)>0.5]
]

sau khi normalize Grad-CAM về `[0,1]`.

Quan trọng nhất:

> **Swin và Proposed phải dùng cùng thresholding rule.**

Không tối ưu threshold riêng cho từng model.

---

# 9. Interpretation

Experiment này sẽ cung cấp hai loại evidence.

### Qualitative evidence

Nếu:

```text
GT Mask
   ↓
████████

Swin
   ↓
   🔥
    🔥

Proposed
   ↓
  🔥🔥🔥
 ███████
```

thì có thể quan sát rằng Proposed tập trung vào vùng annotation tốt hơn.

### Quantitative evidence

Nếu:

[
IoU_{Proposed} > IoU_{Swin}
]

thì có thêm quantitative support cho observation trên.

Cách diễn đạt nên là:

> **The Grad-CAM analysis indicates that the Proposed Model exhibits better alignment with the annotated detection regions than the Swin baseline.**

Không nên nói Grad-CAM “proves” model hiểu lesion.

---

# 10. Những gì không cần làm

Với **Swin single-task fine-level baseline**, loại bỏ hoàn toàn:

* ❌ Binary-level Grad-CAM
* ❌ Coarse-level Grad-CAM
* ❌ Grad-CAM comparison giữa các hierarchy levels
* ❌ 22 fine classes
* ❌ TN/FP/FN analysis
* ❌ Pointing Game
* ❌ Dice
* ❌ Pixel AP
* ❌ PRO
* ❌ Grad-CAM++
* ❌ Score-CAM
* ❌ Eigen-CAM

---

# 11. Final scope

Toàn bộ experiment chỉ còn:

```text
                 5 Fine Classes
                       │
               1 TP / Class
                       │
                  5 Images
                       │
          ┌────────────┴────────────┐
          ↓                         ↓
     Swin Baseline            Proposed Model
    Fine-level ST               Fine-level
          │                         │
      Grad-CAM                  Grad-CAM
          │                         │
          └────────────┬────────────┘
                       ↓
                Compare with
                 GT Mask
                       │
             ┌─────────┴─────────┐
             ↓                   ↓
       Visualization        Grad-CAM IoU
```

### Deliverables cuối cùng

**Figure 1:**
`Original | GT Mask | Swin Grad-CAM | Proposed Grad-CAM`

**Table 1:**

`Swin vs Proposed — Grad-CAM IoU`

**Analysis:**
Một đoạn ngắn trả lời:

1. Proposed có tập trung đúng vùng GT hơn Swin không?
2. Quan sát này có nhất quán trên 5 classes không?
3. Grad-CAM IoU có ủng hộ observation đó không?

Đây là **đủ gọn cho một experiment phụ trong paper**, nhưng vẫn có cả **qualitative + quantitative evidence**, thay vì chỉ đưa heatmap để “nhìn cho đẹp”.
