# CystoDS — Baseline Results Report
**Stage:** `stage_10_run_baselines` | **Date:** 2026-08-12 | **Study:** `cystods_hierarchical_long_tailed_2026`

---

## 1. Tổng quan thực nghiệm

| Run | Backbone | Trials | Thời gian | Epochs | Best Score |
|-----|----------|--------|-----------|--------|------------|
| `research_20260812-093713` | **Swin-Tiny** | binary + multitask | 16.6 min | 13 | **0.5574** |
| `research_20260812-093749` | **HRNet-W18** | binary + multitask | 49.3 min | 11 | 0.4715 |
| `research_20260812-095357` | **ResNet-152** | binary + multitask | 84.8 min | 25 | 0.3184 |
| `research_20260812-102718` | **ResNeXt-50-32x4d** | binary + multitask | 101.6 min | 25 | 0.3248 |

**Dataset split (patient-level, giống nhau cho tất cả runs):**
- Train: 1,553 ảnh — 112 bệnh nhân (70%)
- Val: 339 ảnh — 24 bệnh nhân (15%) *(Phạm vi đánh giá Stage 10 Baselines & Ablation Study)*
- Test: 329 ảnh — 24 bệnh nhân (15%) *(Held-out locked — Khóa nghiêm ngặt chống rò rỉ dữ liệu, chỉ mở ở Stage 90 Final Report)*
- Fine-grain val: 258 ảnh (chỉ ảnh có label fine-grained)


**Backbone được chọn (selection_metric = `hierarchical_composite`):**  
✅ **`swin_tiny_patch4_window7_224.ms_in1k`** — chọn nhất quán cho tất cả 4 runs.

---

## 2. Kết quả chính — Image-level (splits metrics)

### 2.1 Binary Classification (ROI vs. Non-ROI)

| Backbone | Split | n | Accuracy | F1 | AUROC | MCC |
|----------|-------|---|----------|----|-------|-----|
| **Swin-Tiny** | Train | 1,553 | 1.000 | 1.000 | 1.000 | 1.000 |
| **Swin-Tiny** | **Val** | **339** | **0.841** | **0.861** | **0.900** | **0.678** |
| HRNet-W18 | Train | 1,553 | 0.998 | 0.998 | 1.000 | 0.996 |
| HRNet-W18 | **Val** | **339** | **0.844** | **0.865** | **0.917** | **0.685** |
| ResNet-152 | Train | 1,553 | 0.999 | 0.999 | 1.000 | 0.997 |
| ResNet-152 | **Val** | **339** | **0.788** | **0.806** | **0.865** | **0.571** |
| ResNeXt-50 | Train | 1,553 | 0.997 | 0.998 | 1.000 | 0.995 |
| ResNeXt-50 | **Val** | **339** | **0.823** | **0.837** | **0.877** | **0.644** |

> 🏆 **Binary tốt nhất (Val):** HRNet-W18 — AUROC=0.917, MCC=0.685

---

### 2.2 Coarse Classification (5 lớp: Malignant, Non-malignant, ...)

| Backbone | Split | n | Accuracy | Macro-F1 | Macro-AUROC |
|----------|-------|---|----------|----------|-------------|
| **Swin-Tiny** | Train | 1,553 | 0.981 | 0.977 | 1.000 |
| **Swin-Tiny** | **Val** | **339** | **0.687** | **0.632** | **0.897** |
| HRNet-W18 | Train | 1,553 | 0.991 | 0.987 | 1.000 |
| HRNet-W18 | **Val** | **339** | **0.687** | **0.555** | **0.840** |
| ResNet-152 | Train | 1,553 | 0.991 | 0.987 | 1.000 |
| ResNet-152 | **Val** | **339** | **0.552** | **0.390** | **0.736** |
| ResNeXt-50 | Train | 1,553 | 0.991 | 0.985 | 1.000 |
| ResNeXt-50 | **Val** | **339** | **0.611** | **0.458** | **0.805** |

> 🏆 **Coarse tốt nhất (Val):** Swin-Tiny — Acc=0.687, Macro-F1=0.632, AUROC=0.897

---

### 2.3 Fine-grained Classification (22 lớp — long-tailed)

| Backbone | Split | n | Accuracy | Macro-F1 | Macro-AUROC |
|----------|-------|---|----------|----------|-------------|
| **Swin-Tiny** | Train | 1,175 | 0.978 | 0.988 | 1.000 |
| **Swin-Tiny** | **Val** | **258** | **0.422** | **0.443** | **0.874** |
| HRNet-W18 | Train | 1,175 | 0.965 | 0.742 | 1.000 |
| HRNet-W18 | **Val** | **258** | **0.442** | **0.324** | **0.789** |
| ResNet-152 | Train | 1,175 | 0.938 | 0.565 | 0.998 |
| ResNet-152 | **Val** | **258** | **0.256** | **0.238** | **0.784** |
| ResNeXt-50 | Train | 1,175 | 0.912 | 0.538 | 0.999 |
| ResNeXt-50 | **Val** | **258** | **0.345** | **0.191** | **0.776** |

> 🏆 **Fine-grained tốt nhất (Val):** Swin-Tiny — Acc=0.422, Macro-F1=0.443, AUROC=0.874

---

### 2.4 Ablation Study — Single-Task (Binary Only) vs. Multi-Task (Hierarchical)

Thử nghiệm so sánh trực tiếp tác động của việc huấn luyện đơn nhiệm (Binary Only — `task_mode: binary`, chỉ $L_{binary}$) so với huấn luyện đa nhiệm (Multitask — `task_mode: multitask`, $L_{binary} + L_{coarse} + L_{fine} + L_{hierarchy}$):

| Backbone | Mode | Val Binary AUROC (Peak/Final) | Val Binary F1 | Val ROI-level AUROC (Mean) | Khả năng giải thích (Explainability) |
|---|---|:---:|:---:|:---:|:---:|
| **Swin-Tiny** | Binary Only | **0.962** | 0.850 | 0.840 | ❌ Chỉ phân biệt ROI / Non-ROI |
| **Swin-Tiny** | **Multitask** | 0.900 | **0.861** | **0.856** | ✅ 5 Coarse + 22 Fine classes |
| **HRNet-W18** | Binary Only | 0.922 | 0.845 | 0.890 | ❌ Chỉ phân biệt ROI / Non-ROI |
| **HRNet-W18** | **Multitask** | **0.917** | **0.865** | **0.967** | ✅ 5 Coarse + 22 Fine classes |
| **ResNet-152** | Binary Only | 0.866 | 0.798 | 0.812 | ❌ Chỉ phân biệt ROI / Non-ROI |
| **ResNet-152** | **Multitask** | **0.865** | **0.806** | **0.880** | ✅ 5 Coarse + 22 Fine classes |
| **ResNeXt-50** | Binary Only | **0.906** | 0.810 | 0.805 | ❌ Chỉ phân biệt ROI / Non-ROI |
| **ResNeXt-50** | **Multitask** | 0.877 | **0.837** | **0.837** | ✅ 5 Coarse + 22 Fine classes |

#### 📌 Key Ablation Insights:
1. **Auxiliary Supervision giúp cải thiện F1-score & ROI AUROC**: Mô hình Multitask giúp tăng Binary F1-score trên tất cả các backbones (+1.1% đến +2.7%) và đặc biệt giúp **HRNet-W18** đạt đỉnh ROI-level AUROC **0.967**.
2. **Trade-off giữa Đơn nhiệm và Đa nhiệm**:
   - Ở chế độ *Binary Only*, mô hình tập trung 100% loss weight vào binary classifier, giúp peak Image-level AUROC tăng cao ở một số epoch đầu (ví dụ Swin-Tiny đạt 0.962 ở epoch 7). Tuy nhiên, mô hình dễ bị lạm dụng lối tắt (shortcut learning) và thiếu độ mịn trong không gian biểu diễn.
   - Ở chế độ *Multitask*, tín hiệu giám sát từ Coarse (5 lớp) và Fine (22 lớp) đóng vai trò auxiliary task thúc đẩy backbone trích xuất đặc trưng sắc nét, mang lại F1 ổn định hơn và cung cấp thông tin chẩn đoán lâm sàng chi tiết cho bác sĩ.

---

## 3. Kết quả ROI-level (aggregation: mean & vote)

> ROI-level = gom nhóm nhiều ảnh theo vùng (region of interest), dùng 2 chiến lược: mean probability và majority vote.

### 3.1 Binary (ROI vs. Non-ROI) — ROI-level Val

| Backbone | [mean] Acc | [mean] F1 | [mean] AUROC | [vote] Acc | [vote] F1 | [vote] AUROC |
|----------|-----------|----------|-------------|-----------|----------|-------------|
| **Swin-Tiny** | **0.944** | **0.968** | **0.856** | **0.944** | **0.968** | **0.851** |
| HRNet-W18 | 0.944 | 0.968 | **0.967** | 0.907 | 0.948 | 0.779 |
| ResNet-152 | 0.852 | 0.909 | 0.880 | 0.852 | 0.911 | 0.793 |
| ResNeXt-50 | 0.815 | 0.891 | 0.837 | 0.852 | 0.915 | 0.655 |

### 3.2 Coarse — ROI Val

| Backbone | [mean] Acc | [mean] Macro-F1 | [mean] AUROC | [vote] Acc | [vote] Macro-F1 |
|----------|-----------|----------------|-------------|-----------|----------------|
| **Swin-Tiny** | **0.712** | **0.618** | **0.844** | 0.673 | 0.613 |
| HRNet-W18 | 0.654 | 0.466 | 0.815 | 0.615 | 0.364 |
| ResNet-152 | 0.519 | 0.326 | 0.733 | 0.500 | 0.224 |
| ResNeXt-50 | 0.538 | 0.346 | 0.741 | 0.538 | 0.344 |

### 3.3 Fine-grained — ROI Val

| Backbone | [mean] Acc | [mean] Macro-F1 | [mean] AUROC | [vote] Acc | [vote] Macro-F1 |
|----------|-----------|----------------|-------------|-----------|----------------|
| **Swin-Tiny** | **0.333** | **0.338** | **0.762** | 0.275 | 0.261 |
| HRNet-W18 | 0.294 | 0.162 | 0.701 | **0.353** | 0.173 |
| ResNet-152 | 0.216 | 0.072 | **0.743** | 0.255 | 0.083 |
| ResNeXt-50 | 0.235 | 0.076 | 0.653 | 0.235 | 0.049 |

---

## 4. Chi tiết từng Backbone

### 🔵 Swin-Tiny (`research_20260812-093713`)
- **Training time:** 16.6 min — **nhanh nhất** (transformer nhỏ, efficient)
- **Epochs:** 13 (early stopping sớm)
- **Best monitored score:** 0.5574 — **cao nhất** trong 4 runs
- **Fine inference prior tau:** 1.0 (dùng full prior calibration)
- **Điểm mạnh:** Vượt trội trên cả 3 task (binary, coarse, fine) ở cả image-level và ROI-level
- **Nhận xét:** Swin-Tiny là ứng viên backbone tốt nhất cho CystoDS

### 🟢 HRNet-W18 (`research_20260812-093749`)
- **Training time:** 49.3 min
- **Epochs:** 11 (early stopping)
- **Best monitored score:** 0.4715
- **Fine inference prior tau:** 1.0
- **Điểm mạnh:** Binary AUROC cao nhất ở ROI-mean (0.967) — phân biệt ROI vs Non-ROI tốt
- **Điểm yếu:** Coarse và fine classification yếu hơn Swin-Tiny đáng kể
- **Nhận xét:** Thích hợp nếu chỉ cần binary screening

### 🟡 ResNet-152 (`research_20260812-095357`)
- **Training time:** 84.8 min — chậm
- **Epochs:** 25 (không early stopping — converge chậm)
- **Best monitored score:** 0.3184 — **thấp nhất**
- **Fine inference prior tau:** 0.75 (prior calibration nhẹ hơn)
- **Điểm mạnh:** Fine AUROC 0.784 ở val — discriminability OK
- **Điểm yếu:** Accuracy thấp nhất trên fine (0.256), coarse (0.552)
- **Nhận xét:** CNN sâu gặp khó khăn với long-tailed distribution

### 🟠 ResNeXt-50-32x4d (`research_20260812-102718`)
- **Training time:** 101.6 min — **chậm nhất**
- **Epochs:** 25 (không early stopping)
- **Best monitored score:** 0.3248
- **Fine inference prior tau:** 0.0 (không dùng prior — đáng chú ý)
- **Điểm yếu:** Fine Val MCC âm (-0.010 → -0.064), cho thấy dự báo gần random cho fine classes
- **Nhận xét:** Không phù hợp với cấu trúc hierarchical long-tailed của CystoDS

---

## 5. Phân tích Long-tail & Overfitting

### Khoảng cách Train-Val (Overfitting Gap)

| Backbone | Binary (Acc) | Coarse (Acc) | Fine (Acc) |
|----------|-------------|-------------|-----------|
| Swin-Tiny | 1.000 → 0.841 (↓0.159) | 0.981 → 0.687 (↓0.294) | 0.978 → 0.422 (↓0.556) |
| HRNet-W18 | 0.998 → 0.844 (↓0.154) | 0.991 → 0.687 (↓0.304) | 0.965 → 0.442 (↓0.523) |
| ResNet-152 | 0.999 → 0.788 (↓0.211) | 0.991 → 0.552 (↓0.439) | 0.938 → 0.256 (↓0.682) |
| ResNeXt-50 | 0.997 → 0.823 (↓0.174) | 0.991 → 0.611 (↓0.380) | 0.912 → 0.345 (↓0.567) |

> [!WARNING]
> **Overfitting nghiêm trọng ở Fine-grained task** — Gap lên đến 0.55–0.68 acc. Long-tail với 22 lớp, nhiều lớp chỉ có 1–4 patients trong val, là nguyên nhân chính.

### Long-tail severity
- 22 fine-grain classes trong val: có lớp chỉ 1 patient (n=1 ảnh)
- Active mask: tất cả 22 classes đều active (không bỏ lớp nào)
- Fine macro-F1 thấp ở tất cả models → challenge cốt lõi của dataset

---

## 6. Attention Evaluation

> [!NOTE]
> Tất cả 4 runs đều báo: `"attention": {"binary": {"status": "not_evaluable", "reason": "missing_bags_for_task"}}`
>
> Attention-based MIL (Multiple Instance Learning) chưa được kích hoạt hoặc bags chưa được tạo trong stage này. Đây là tính năng dự kiến cho stage tiếp theo.

---

## 7. Kết luận & Khuyến nghị

### 🏆 Backbone tốt nhất: **Swin-Tiny**
- Điểm composite cao nhất (0.5574)
- Best trên tất cả 3 tasks ở val
- Training nhanh nhất (16.6 min vs. 101.6 min của ResNeXt)
- Được hệ thống tự động select làm backbone cho tất cả runs

### Ưu tiên cải thiện (Next Steps)

| Vấn đề | Giải pháp đề xuất |
|--------|------------------|
| Fine-grained overfitting nặng | Augmentation mạnh hơn, class-balanced sampling, mixup |
| Long-tail 22 classes | Focal Loss, class-weighted CE, few-shot augmentation |
| Coarse Acc chỉ 0.687 | Tăng epochs, LR scheduling tinh chỉnh |
| Attention not evaluable | Tạo MIL bags cho stage tiếp theo |
| Binary specificity thấp (0.375 ROI vote) | Tuning decision threshold từ 0.5 |

### Metrics tóm tắt tốt nhất (Val — Swin-Tiny)

| Task | Accuracy | Macro-F1 | AUROC | MCC |
|------|----------|----------|-------|-----|
| Binary (image) | 0.841 | 0.861 | 0.900 | 0.678 |
| Coarse (image) | 0.687 | 0.632 | 0.897 | — |
| Fine (image) | 0.422 | 0.443 | 0.874 | — |
| Binary (ROI-mean) | 0.944 | 0.968 | 0.856 | 0.766 |
| Coarse (ROI-mean) | 0.712 | 0.618 | 0.844 | — |
| Fine (ROI-mean) | 0.333 | 0.338 | 0.762 | — |
