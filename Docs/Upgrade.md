Đúng. **Cấu trúc baseline của Report về mặt ý tưởng là chuẩn và hợp lý** cho một paper kiểu hierarchical/multi-task classification:

* single-task binary
* single-task coarse
* single-task fine
* multi-task binary + coarse
* multi-task binary + coarse + fine
* proposed hierarchical
* sau đó dùng ablation để tách ảnh hưởng của từng module trong proposed

Report mô tả đúng framework này ở phần phương pháp và benchmark baseline. 

Vấn đề hiện tại **không nằm ở danh sách baseline**, mà nằm ở việc các experiment chưa hoàn toàn “controlled” để ablation thực sự chứng minh sức mạnh từng module.

## 1. Baseline hierarchy hiện tại là hợp lý

Có thể hiểu toàn bộ experimental ladder như sau:

```text
Single-task
├── Binary only
├── Coarse only
└── Fine only

Multi-task
├── Binary + Coarse
└── Binary + Coarse + Fine

29: Hierarchical
30: └── Binary + Coarse + Fine
31:     + long-tail fine loss (Balanced Softmax)
32:     + taxonomy parent-mass losses (L_BC & L_CF)
33:     + SupCon
34:     + patient-based prior/calibration
```

Đây là progression rất tốt vì nó cho phép trả lời tuần tự:

```text
Single-task có đủ không?
        ↓
Multi-task có giúp không?
        ↓
Thêm hierarchy có giúp không?
        ↓
Long-tail handling có giúp không?
        ↓
Consistency / SupCon / sampler giúp phần nào?
```

Report cũng đang dùng đúng logic này khi so sánh binary/coarse/fine và sau đó dùng Table 14 làm ablation. 

---

# 2. Tuy nhiên, ablation hiện tại chưa đủ sạch để nói “module X làm tăng Y”

Đây là thứ cần sửa quan trọng nhất.

Một ablation chuẩn phải gần như:

[
M_{\text{full}}
]

so với

[
M_{\text{full}}-\text{module X}
]

và **mọi thứ khác phải giữ nguyên**.

Ví dụ:

```text
Full proposed
Balanced Softmax
+ Binary head
+ Coarse head
+ Fine head
+ Consistency
+ SupCon
+ no augmentation (default baseline)
+ same epochs
+ same scheduler
+ same checkpoint metric
```

so với:

```text
No SupCon
y hệt mọi thứ
chỉ:
SupCon weight = 0
```

Hiện tại Stage 30 proposed và Stage 40 ablation không hoàn toàn dùng cùng training configuration. Ví dụ source hiện có Stage 30 tối đa 25 epochs/patience 6, trong khi Stage 40 dùng 20 epochs/patience 5.

Vì vậy hiện tại:

[
\Delta metric
]

không chắc chỉ do module bị bỏ.

---

# 3. Việc đầu tiên nên làm: thêm `Full Proposed` trực tiếp vào Stage 40

Đây là chỉnh sửa quan trọng nhất.

Stage 40 nên có một trial control:

```python
ablation_full_proposed
```

và tất cả ablation phải xuất phát trực tiếp từ config này.

Cấu trúc nên thành:

```text
A0 — Full Proposed (Default: No Augmentation)

A1 — Flat fine CE
A2 — Multi-task, no hierarchy
A3 — Hierarchical CE
A4 — No binary auxiliary
A5 — No consistency
A6 — No SupCon
A7 — Class-balanced sampler
A8 — Data Augmentation (+ Augmentation)
```

*(Lưu ý: Thử nghiệm `WLC-only` đã được tách riêng khỏi Component Ablation để đưa vào phần Domain/Modality Analysis ở Mục 12).*

Nhưng A0–A8 phải dùng:

```text
same backbone
same pretrained weights
same image size
same augmentation status (Default: Off, ngoại trừ A8)
same optimizer
same LR
same scheduler
same epochs
same patience
same seed
same split
same inference calibration
same checkpoint criterion
```

Chỉ đúng biến nghiên cứu được thay đổi.

Khi đó Table 14 mới thực sự là **component attribution**.

---

# 4. Phải thống nhất Stage 30, Stage 40 và Stage 90 thành một canonical proposed method

Hiện source có một inconsistency lớn.

Stage 30 research trial thực tế override:

```python
"fine_loss": "balanced_softmax"
```

Stage 40 cũng dựa trên:

```python
"fine_loss": "balanced_softmax"
```

nhưng Stage 90 final currently dùng:

```python
"fine_loss": "balanced_softmax_smoothed"
```

Đây cần sửa **trước Stage 90**: chuyển Stage 90 về dùng `balanced_softmax` chuẩn để đồng nhất với Stage 30 và Stage 40.

Nên có đúng một định nghĩa:

```text
PROPOSED_CANONICAL_CONFIG
```

và Stage 30/40/90 cùng import nó.

Ví dụ:

```python
PROPOSED_OVERRIDES = {
    "task_mode": "hierarchical",
    "fine_loss": "balanced_softmax",
    "fine_prior_source": "train_set",
    "consistency_loss_weight": 0.25,
    "supervised_contrastive_loss_weight": 0.10,
    "use_data_augmentation": False,  # Default off, moved to Ablation
    ...
}
```

Stage 30:

```python
trial = PROPOSED_OVERRIDES
```

Stage 40:

```python
full_proposed = PROPOSED_OVERRIDES
no_supcon = PROPOSED_OVERRIDES | {"supervised_contrastive_loss_weight": 0}
with_aug = PROPOSED_OVERRIDES | {"use_data_augmentation": True}
```

Stage 90:

```python
trial = PROPOSED_OVERRIDES
```

Như vậy không thể vô tình train ba “proposed models” khác nhau.

---

# 5. Chốt chọn `Balanced Softmax` cho Canonical Proposed Model dựa trên kết quả Stage 2

Dựa vào kết quả thực nghiệm Stage 2 (Screening):

```text
Balanced Softmax
Fine F1 22 = 0.2539
Primary F1 = 0.4315

Balanced Softmax + smoothing
Fine F1 22 = 0.1392
Primary F1 = 0.2369
```

Kết quả cho thấy Label Smoothing khi kết hợp với Balanced Softmax làm sụt giảm nghiêm trọng hiệu năng (Primary F1 giảm từ **0.4315** xuống **0.2369**). 

Vì vậy, quyết định **chốt chọn `Balanced Softmax` (không smoothing)** làm fine loss chính thức cho Canonical Proposed Method từ Stage 30, Stage 40 đến Stage 90, loại bỏ hoàn toàn `balanced_softmax_smoothed`.

Canonical proposed method chính thức là:

```text
Balanced Softmax
+ patient-informed calibration
```

---

# 6. Cần tách rõ “long-tail loss” và “inference calibration”

Đây cũng là phần Report hiện có thể làm người đọc hơi nhầm.

Hiện method có thể chứa hai khái niệm khác nhau:

### Training

```text
Balanced Softmax
```

### Inference

```text
prior correction với tau
patient-based smoothed prior
```

Hai thứ này không nên mô tả nhập chung thành:

> “Balanced Softmax sử dụng smoothed patient prior”

nếu actual training loss không thực sự dùng prior đó.

Paper nên viết riêng:

```text
Fine training objective:
Balanced Softmax using training class frequency.

Post-hoc fine calibration:
patient-count prior with Laplace smoothing,
power transform and validation-selected tau.
```

Nếu actual `config.json` cho thấy khác thì mô tả theo đúng artifact.

---

# 7. Multi-task baseline và Data Augmentation control

Hiện report đã tự thừa nhận một số suite khác augmentation/lịch training (do Stage 10 baseline không dùng Augmentation trong khi Stage 30 proposed cũ bật full Augmentation, tạo ra nhiễu confounding variable).

Quy tắc mới:
1. **Tắt toàn bộ Data Augmentation ở default configuration** cho cả Baseline và Proposed model.
2. **Đưa Data Augmentation vào Ablation study** như một component riêng biệt (`+ Data Augmentation`) để đo lường chính xác phần đóng góp của Augmentation vào kết quả final.

Để chứng minh hierarchy mạnh hơn multi-task, bạn cần một baseline đặc biệt quan trọng:

```text
Matched Multi-task B+C+F
```

nó phải giống proposed 100% về:

```text
Swin-Tiny
pretrained
no augmentation (Default: Off)
optimizer
LR
scheduler
epochs
batch
seed
checkpoint selection
fine calibration
```

nhưng:

```text
consistency = 0
SupCon = 0
long-tail special handling = off/CE
```

Đây mới là đối chứng sạch và mạnh nhất cho proposed.

Tương tự:

```text
Binary-only matched
Coarse-only matched
Fine-only matched
```

nên được rerun với cùng recipe nếu muốn so numerical improvement chính thức.

Stage 10 hiện vẫn có thể giữ làm **simple baseline**, nhưng paper nên phân biệt:

```text
Simple baselines
vs
Matched baselines
```

---

# 8. Checkpoint selection criterion cũng phải được kiểm soát

Một điểm dễ bị bỏ qua:

Nếu proposed chọn checkpoint bằng:

[
0.35 F1_{coarse}
+0.45 F1_{primary-fine}
+0.20 Acc_{hier}
]

nhưng baseline multi-task chọn chỉ bằng:

```text
coarse_macro_f1
```

thì comparison cũng không hoàn toàn matched.

Ví dụ source Stage 90 multi-task hiện dùng:

```python
"monitor_metric": "coarse_macro_f1"
```

trong khi proposed dùng:

```python
"hierarchical_composite"
```

Điều này có thể hợp lý nếu mỗi model có mục tiêu khác nhau, nhưng nếu muốn nói:

> hierarchy tốt hơn multi-task trên fine/hierarchy

thì reviewer có thể hỏi:

> Có phải chỉ vì checkpoint được select theo fine/hierarchical metric?

Nên có ít nhất một matched comparison:

```text
same composite checkpoint criterion
```

cho các model có đủ coarse+fine heads.

---

# 9. Rare-class collapse phải được xử lý trước final model

PreMalignant hiện là blocker thật sự.

Report phát hiện:

[
69/248
]

fine predictions thành PreMalignant dù test không có ground-truth PreMalignant. 

Canonical model **không nên được freeze** cho Stage 90 trong trạng thái này.

Có hai hướng sạch.

### Hướng A — conservative, tôi nghiêng về hướng này

Các class chỉ có 1–2 patient:

```text
PreMalignant
NephrogenicAdenoma
BenignRare
...
```

không đưa vào primary fine claim.

Vẫn giữ output 22 classes để taxonomy không đổi, nhưng:

```text
rare classes
→ auxiliary/exploratory analysis
```

Primary metric chỉ dùng classes đủ patient support.

Repo thực ra đã có `primary_fine_min_train_patients`, tức framework này đã gần sẵn.

### Hướng B

Giữ đủ 22-class closed-set classification nhưng thêm guardrail vào model selection:

```text
validation metric cao
AND
rare predicted prevalence <= threshold
```

Nếu fail rare-collapse:

```text
checkpoint invalid
```

không được chọn chỉ vì macro-F1 cao.

---

# 10. `tau` calibration cũng nên có guardrail

Hiện tau được chọn chỉ bằng primary macro-F1 validation.

Report cho:

```text
tau = 0     → 0.5236
tau = 0.25  → 0.5314
tau = 0.5   → 0.5353  ← selected
tau = 0.75  → 0.4993
tau = 1     → 0.4508
```



Nhưng (\tau=0.5) cuối cùng dẫn đến PreMalignant collapse trên test.

Do đó selection criterion nên chuyển từ:

[
\max F1_{primary}
]

thành kiểu:

[
\max F1_{primary}
]

subject to:

[
\text{rare predicted share}<T
]

hoặc multi-objective:

[
Score
=====

F1_{primary}
-\lambda \cdot RareCollapsePenalty
]

Cái này rất đáng làm trước final CV.

---

# 11. Chỉnh tau = 0.5 cho Logit Adjustment để tránh bị trùng với Balanced Softmax

Trong report Stage 20 hiện tại, `Balanced Softmax` và `Logit Adjustment` cho kết quả trùng khớp hoàn toàn do thiết lập \(\tau = 1.0\).

Về mặt lý thuyết, khi \(\tau = 1.0\), formulation của Logit Adjustment tương đương toán học với Balanced Softmax.

Do đó, để phân biệt rõ ràng giữa hai phương pháp và đánh giá đúng hiệu quả độc lập của Logit Adjustment:
* **Chỉnh tham số Logit Adjustment thành \(\tau = 0.5\)**.
* Giữ Logit Adjustment (\(\tau = 0.5\)) trong danh mục Long-tail study ở Stage 20 làm phương pháp so sánh riêng biệt với Balanced Softmax.

---

# 12. WLC-only không nên đặt cùng nhóm với pure module ablation

Report đã ghi chú đúng điều này. 

`Train WLC only` không phải:

> remove one module.

Nó thay cả training distribution.

Do đó Table cuối nên tách:

### Component ablation

```text
Full
- binary auxiliary
- consistency
- SupCon
CE instead of long-tail loss
sampler
+ Data Augmentation
```

### Domain/modality analysis

```text
Train All → Test WLC
Train WLC → Test WLC
Train All → Test All
```

Như vậy logic paper sạch hơn.

---

# 13. Baseline và ablation cuối cùng tôi đề xuất

Một experimental matrix tốt có thể là:

### Main baselines (Default: No Augmentation)

```text
B1  Binary single-task
B2  Coarse single-task
B3  Fine single-task CE
B4  Multi-task Binary + Coarse
B5  Multi-task Binary + Coarse + Fine CE
B6  Proposed Hierarchical
```

Tất cả **matched training recipe** (Default Augmentation = Off).

### Long-tail study

```text
L1 CE
L2 Weighted CE
L3 Focal
L4 Balanced Softmax (Selected from Stage 2)
L5 Logit Adjustment (tau = 0.5)
L6 LDAM
```

`Balanced Softmax` được chốt chọn từ Stage 2. `Logit Adjustment` được chỉnh \(\tau = 0.5\) để tránh trùng lặp với Balanced Softmax.

### Proposed ablation

```text
A0 Full proposed (Default: No Augmentation)
A1 - Binary auxiliary
A2 - Consistency
A3 - SupCon
A4 Fine loss → CE
A5 Patient prior/calibration → off
A6 Class-balanced sampler
A7 + Data Augmentation (Đóng góp của Augmentation vào final result)
A8 hierarchy → plain multi-task
```

### Domain analysis

```text
D1 Train all → Test all
D2 Train all → Test WLC
D3 Train WLC → Test WLC
```

### Evaluation level

```text
E1 Image-level
E2 ROI mean
E3 ROI vote
E4 ROI attention
```

Như vậy paper sẽ rất dễ đọc.

---

# 14. Stage 90 nên chạy gì?

Sau khi development stage quyết định xong method:

```text
Do not touch architecture anymore.
```

Stage 90 chỉ nên có khoảng 2–3 configurations.

Ví dụ:

```text
Final comparator:
Matched Multi-task B+C+F CE

Final proposed:
Hierarchical + selected long-tail configuration

Optional:
Best alternative discovered in development
```

Sau đó:

```text
5 patient folds
×
3 seeds
```

và xuất **OOF predictions**.

Không cần mang toàn bộ 15 ablations sang CV — rất tốn compute và không cần thiết.

---

# 15. Statistical reporting cũng cần nâng cấp

Final paper nên báo:

```text
Point estimate
+ patient-level 95% CI
+ paired difference
```

Ví dụ:

[
\Delta F1=
F1_{proposed}-F1_{multitask}
]

và bootstrap theo patient trên paired OOF predictions.

Binary:

* paired McNemar;
* AUROC paired bootstrap.

Coarse/fine:

* patient-level paired bootstrap/permutation.

Không chỉ:

```text
Proposed 0.588
Baseline 0.571
→ proposed better
```

mà cần:

```text
Δ = +0.017
95% CI = [...]
```

---

# 16. Prediction artifact phải trở thành mandatory output

Mất proposed prediction CSV hiện là một điểm yếu lớn của report.

Từ giờ mỗi run nên **không được mark completed** nếu thiếu:

```text
train_predictions.csv
val_predictions.csv
test_predictions.csv
```

với tối thiểu:

```text
filename
pid
visit
lesion
modality
targets
logits
probabilities
predictions
selected tau
```

Có thể nén `.parquet` hoặc `.csv.gz`.

Checkpoint có thể remote-only, nhưng predictions nên lưu immutable.

---

# 17. Source provenance cũng cần thêm `cystods_science.py`

Hiện `REQUIRED_SOURCE_FILES` của Stage 30/90 gồm stage script, core, HF helper, README nhưng chưa thấy `cystods_science.py`.

Nên thêm:

```python
_THIS_SOURCE.with_name("cystods_science.py")
```

vì file này ảnh hưởng trực tiếp metric, prior và scientific gate.

---

# 18. Sau khi sửa, claim của paper mới trở nên sạch

Khi đó paper có thể nói rất rõ:

> **Baseline progression** demonstrates the effect of moving from single-task to multi-task learning.

> **Matched multi-task versus hierarchical comparison** isolates the contribution of hierarchical modeling.

> **Component ablations** isolate binary auxiliary supervision, consistency regularization, contrastive representation learning and long-tail correction.

> **Long-tail screening** determines the fine-level objective independently.

Và nếu result support, lúc đó mới nói:

> proposed improves X.

Nếu result không support, vẫn có một paper tốt:

> hierarchy exhibits trade-offs and exposes rare-class failure modes.

---

## Tóm lại

**Định nghĩa baseline hiện tại là chuẩn.** Tôi sẽ không thay cấu trúc cơ bản:

[
\boxed{
Single\ task
\rightarrow
Multi-task
\rightarrow
Hierarchical
\rightarrow
Ablation
}
]

Cần sửa chủ yếu ở **experimental control**, không phải invent thêm model:

1. tạo `Full Proposed` ngay trong Stage 40 (mặc định tắt Augmentation ở baseline & proposed control);
2. **về Data Augmentation**: tắt toàn bộ Aug ở default configuration và đưa Aug vào Ablation study (`+ Data Augmentation`) để đo lường chính xác phần đóng góp vào final result;
3. **về Fine Loss**: chốt chọn `balanced_softmax` (không smoothing) cho Stage 30/40/90 dựa trên kết quả thực nghiệm Stage 2;
4. **về Logit Adjustment**: chỉnh `tau = 0.5` (thay vì \(\tau = 1.0\)) trong Stage 20 để tránh bị trùng lặp với Balanced Softmax;
5. đồng bộ exact proposed config Stage 30/40/90 qua `PROPOSED_CANONICAL_CONFIG`;
6. mọi ablation chỉ đổi đúng **1 biến** xuất phát từ `Full Proposed`;
7. rerun các **matched baselines** bằng cùng training recipe;
8. thống nhất checkpoint-selection criterion cho comparisons quan trọng;
9. sửa rare-class calibration/collapse trước final CV;
10. tách WLC analysis khỏi component ablation;
11. bắt buộc giữ prediction artifacts;
12. cuối cùng mới chạy **Stage 90 5-fold × 3 seeds** làm confirmatory experiment.

Nếu làm đủ các điểm trên, experimental design của dự án sẽ chuyển từ **“nhiều experiment khá tốt”** thành một **ablation framework rất sạch, reviewer có thể theo dõi được chính xác mỗi improvement đến từ đâu**.
