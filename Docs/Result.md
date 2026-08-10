Tôi đã đọc kỹ bản **`CystoDS_Hierarchical_Long_Tailed_VI.pdf`** và đối chiếu nó với data audit, proposal ban đầu và source code hiện tại của Stage 30/40/90. Một điểm cần xác định trước: tài liệu tự ghi trạng thái là **“bản thảo kết quả nội bộ (development hold-out)”**, mặc dù về hình thức nó gần như một final report 10 trang. Stage 90, external validation, paired significance, ROI-level evaluation và Grad-CAM hợp lệ vẫn chưa hoàn thành. Vì vậy về mặt khoa học, tôi sẽ gọi nó là **comprehensive development report**, chưa phải final experimental paper. 

## 1. Đánh giá tổng quát

Nếu chỉ xét chất lượng của một báo cáo nghiên cứu nội bộ, tôi đánh giá nó **khá tốt, khoảng 8/10**. Điểm mạnh nhất không nằm ở model mà ở cách nghiên cứu được tổ chức: patient-disjoint split, protocol hash, fixed taxonomy, support-aware metrics, patient bootstrap, scientific gate, artifact provenance và đặc biệt là thái độ không che giấu negative results.

Nhưng nếu xét nó như **bằng chứng cho một “proposed hierarchical method”**, mức thuyết phục thấp hơn đáng kể, khoảng **5.5–6/10**. Có ba vấn đề lớn:

| Vấn đề                                                                          | Mức độ |
| ------------------------------------------------------------------------------- | -----: |
| Proposed model chưa thực sự thắng các baseline ở coarse/fine một cách nhất quán |     🔴 |
| Ablation và baseline không hoàn toàn matched về training protocol               |     🔴 |
| Định nghĩa method giữa Stage 30, Stage 40 và Stage 90 hiện không thống nhất     |     🔴 |

Điều thú vị là bản report tự thừa nhận phần lớn giới hạn này. Đó lại là một điểm cộng lớn về scientific integrity.

---

# 2. Kết quả mạnh nhất: binary ROI detection

Phần binary thực sự rất mạnh.

Trên 329 ảnh test từ 24 bệnh nhân:

[
AUROC=0.9994
]

[
AUPRC=0.9995
]

[
F1=0.9775
]

với:

[
TP=174,\quad FN=0,\quad TN=147,\quad FP=8
]

tức sensitivity = 1.0000 và specificity = 0.9484. Patient-level bootstrap 1.000 lần vẫn cho AUROC 95% CI 0.9982–1.0000 và F1 khoảng 0.9606–0.9942. 

So với các baseline cùng project:

| Model                     |      AUROC |         F1 |
| ------------------------- | ---------: | ---------: |
| Binary-only               |     0.9926 |     0.9421 |
| Binary + Coarse           |     0.9971 |     0.9718 |
| Binary + Coarse + Fine    |     0.9903 |     0.9326 |
| **Hierarchical proposed** | **0.9994** | **0.9775** |

Ở bề mặt, đây là improvement tốt.

Nhưng tôi **không khuyên viết rằng hierarchy làm tăng F1 từ 0.9421 lên 0.9775**. Report đã phát hiện đúng vấn đề: simplified Stage 10 dùng augmentation khác với Stage 30. Stage 10 gần như không augmentation trong khi proposed model có FOV crop, random resized crop, horizontal/vertical flip, rotation, color jitter và random erasing. 

Do đó:

[
\Delta F1
=========

\text{hierarchy}
+
\text{augmentation}
+
\text{schedule}
+
\text{training differences}
]

chứ chưa phải:

[
\Delta F1=\text{hierarchy}
]

Đây là một confound rất quan trọng.

### Tuy nhiên binary result vẫn có giá trị

Nó chứng minh được điều hẹp hơn nhưng khá mạnh:

> Với Swin-Tiny, patient-disjoint CystoDS paper-like protocol này cho phép xây dựng một internal ROI detector có hiệu năng rất cao.

Đó là claim hợp lệ.

Nó **không** chứng minh được:

> Hierarchical learning là nguyên nhân tạo ra hiệu năng 0.9775.

Hai claim này cần tách biệt.

---

# 3. Coarse classification cho thấy câu chuyện hoàn toàn khác

Đây là điểm report làm rất tốt: không dùng binary result đẹp để che coarse result.

Ở 5-class:

| Model                  |   Macro-F1 |
| ---------------------- | ---------: |
| Coarse only            |     0.6531 |
| **Binary + Coarse**    | **0.6710** |
| Binary + Coarse + Fine |     0.4682 |
| Proposed hierarchical  |     0.5880 |

Tức proposed model **thua Binary+Coarse tới 0.083 macro-F1 tuyệt đối**. 

Điều này rất đáng chú ý.

Nếu hierarchy thực sự giúp representation chung ở mọi tầng, ta có thể mong:

```text
Binary tốt
     ↓
Coarse tốt
     ↓
Fine tốt
```

Nhưng thực nghiệm lại gần:

```text
Binary         rất tốt     0.9775 F1
Coarse         trung bình  0.5880 macro-F1
Fine           yếu         0.3380 macro-F1/22
Hierarchical   yếu         0.2863 accuracy
```

Nghĩa là encoder học được rất tốt câu hỏi:

> “Có ROI đáng chú ý hay không?”

nhưng chưa học tốt:

> “ROI/context này chính xác thuộc loại nào?”

Đây thực ra là một **kết quả khoa học khá đẹp**.

---

# 4. Error ở coarse level còn đáng chú ý hơn macro-F1

Report cho:

* Malignant F1 = 0.8041
* Normal mucosa F1 = 0.7813
* Foreign bodies F1 = 0.7324
* Anatomical landmarks F1 = 0.4286
* **Non-malignant F1 = 0.1935**

Đặc biệt:

[
26/32
]

Non-malignant bị dự đoán thành Malignant.

Tức:

[
81.25%
]

Non-malignant test bị overcall thành malignant. 

Đây là lỗi lâm sàng quan trọng hơn việc macro-F1 chỉ có 0.588.

Binary detector nói:

> “Đây là ROI.”

khá chính xác.

Nhưng một khi model phải phân biệt:

```text
Malignant
vs
Non-malignant
```

thì nó gặp vấn đề rất lớn.

Từ góc độ application, hệ thống hiện tại có thể hợp với:

> **ROI triage / second reader**

hơn là:

> **AI diagnostic classifier**.

Report cũng diễn giải theo hướng này, và tôi cho rằng đó là framing đúng.

---

# 5. Fine classification: con số 0.4647 dễ gây hiểu lầm nếu đứng một mình

Report rất đúng khi đưa **hai macro-F1**:

[
F1_{supported}=0.4647
]

nhưng:

[
F1_{22-class}=0.3380
]

Lý do test chỉ có support cho **16/22 classes**.

Đây là cách báo cáo tốt hơn rất nhiều so với chỉ viết:

> “Fine macro-F1 = 0.4647”.

Bởi vì dataset cực kỳ long-tail. Data audit xác nhận PreMalignant chỉ có **1 ảnh từ 1 patient**, NephrogenicAdenoma và BenignRare chỉ 2 bệnh nhân, nhiều subclass khác chỉ vài bệnh nhân. 

Ở test:

```text
22 taxonomy classes
│
├── 16 classes có ground-truth
└── 6 classes không có một mẫu nào
```

Do đó, report không thể chứng minh model nhận dạng được toàn bộ 22 subclasses.

Macro-F1 fixed denominator ở đây nên được hiểu là:

> taxonomy coverage metric

chứ không phải:

> accuracy trên đủ 22 bệnh.

Report cũng nói rõ điều này. Đây là một trong những đoạn methodological reporting tốt nhất của tài liệu.

---

# 6. Vấn đề nghiêm trọng nhất: PreMalignant collapse

Đây mới là kết quả quan trọng nhất toàn bộ Stage 30.

PreMalignant có:

[
N_{train}\approx 1\text{ patient}
]

và:

[
N_{test}=0
]

nhưng model dự đoán:

[
69/248
]

fine-labelled test images thành PreMalignant.

Tức:

[
27.8%
]

toàn bộ fine predictions. 

Trong đó:

* HighGradePapillary → PreMalignant: **36**
* LowGradePapillary → PreMalignant: **18**

chỉ hai lỗi này đã chiếm 54/148 fine errors.

Đây không phải một lỗi nhỏ.

Nó cho thấy chiến lược long-tail correction đang **over-correct rare classes**.

---

# 7. Có thể giải thích cơ chế collapse này

Report chọn inference prior calibration:

[
\tau\in{0,0.25,0.5,0.75,1}
]

và validation chọn:

[
\tau=0.5
]

Fine inference correction về bản chất có dạng gần:

[
z'_c=z_c-\tau\log p_c
]

Với rare class:

[
p_c\ll1
]

nên:

[
\log p_c\ll0
]

và vì vậy:

[
-\tau\log p_c\gg0
]

Tức một class cực hiếm sẽ được **boost logit rất mạnh**.

Với class có 1 patient như PreMalignant, prior thực tế không đủ đáng tin để làm một correction mạnh.

Đây là trường hợp classic:

```text
Head-class bias
       ↓
long-tail correction
       ↓
correction quá mạnh
       ↓
tail-class hallucination
```

Report đã phát hiện được symptom thông qua `rare_class_collapse gate`, nhưng tôi nghĩ paper nên giải thích rõ **cơ chế** này hơn.

---

# 8. Một inconsistency quan trọng giữa report và source hiện tại

Đây là chỗ tôi phát hiện sau khi đối chiếu trực tiếp source.

Trong Stage 30, base config ban đầu đặt:

```python
"fine_loss": "balanced_softmax_smoothed"
```

nhưng **trial research thực sự lại override**:

```python
"fine_loss": "balanced_softmax"
```

mặc dù experiment được đặt tên:

```python
"proposed_hierarchical_swin_smoothed"
```

Đây là điểm cần audit ngay với **`config.json` của actual Stage 30 run**.

Bởi vì report mô tả proposed method theo kiểu:

> Balanced Softmax fine + prior theo patient có smoothing.

Câu này có thể khiến người đọc hiểu rằng **smoothed patient prior được dùng trong training loss**.

Nhưng source Stage 30 hiện tại cho thấy research trial dùng:

```text
balanced_softmax
```

không phải:

```text
balanced_softmax_smoothed
```

### Tệ hơn nữa: Stage 90 lại dùng `balanced_softmax_smoothed`

Future final CV hiện cấu hình:

```python
"fine_loss": "balanced_softmax_smoothed"
```

Tức hiện tại có khả năng:

```text
Development Stage 30
    Balanced Softmax
          │
          ▼
Development report result

nhưng

Final Stage 90
    Balanced Softmax Smoothed
          │
          ▼
Final CV result
```

Nếu đúng như vậy:

[
Method_{Stage30}\ne Method_{Stage90}
]

Đây là **P0 bug về scientific protocol**, phải giải quyết trước khi chạy Stage 90.

---

# 9. Stage 40 cũng đang đi theo Stage 30, không theo Stage 90

Stage 40 base config hiện dùng:

```python
"fine_loss": "balanced_softmax"
```

Vậy ta hiện có:

| Stage                       | Fine loss                       |
| --------------------------- | ------------------------------- |
| Stage 20 screening          | nhiều loss                      |
| Stage 30 proposed           | `balanced_softmax`              |
| Stage 40 ablation           | `balanced_softmax`              |
| **Stage 90 final proposed** | **`balanced_softmax_smoothed`** |

Điều này tạo một vấn đề logic:

> Ablation ở Stage 40 đang giải thích Stage 30, nhưng final method ở Stage 90 lại là một model khác.

Nếu Stage 90 cuối cùng cho kết quả tốt/xấu, bạn không thể dùng Table 14 hiện tại để nói:

> “Ablation chứng minh contribution của từng thành phần trong final model.”

---

# 10. Kết quả Stage 20 còn khiến lựa chọn Stage 90 khó hiểu hơn

Bảng long-tail screen:

| Loss                         | Fine F1 22 | Primary F1 |
| ---------------------------- | ---------: | ---------: |
| CE                           |     0.1291 |     0.2199 |
| Weighted CE                  |     0.2362 |     0.3607 |
| Balanced Softmax             | **0.2539** | **0.4315** |
| Balanced Softmax + smoothing |     0.1392 |     0.2369 |
| Logit adjustment             | **0.2539** | **0.4315** |
| LDAM                         |     0.1519 |     0.2942 |



Trong screening này, smoothing **thua rất mạnh** standard Balanced Softmax.

Primary F1:

[
0.4315\rightarrow0.2369
]

sau smoothing.

Vậy hiện tại việc Stage 90 chuyển sang `balanced_softmax_smoothed` **không được support bởi Stage 20**.

Nếu không có một lý do validation/protocol rất rõ, tôi sẽ không làm vậy.

---

# 11. Balanced Softmax và Logit Adjustment giống nhau không phải tình cờ

Report viết rằng hai model cho kết quả giống nhau tới bốn chữ số nhưng:

> “điều đó không chứng minh chúng tương đương nói chung”.

Câu này đúng nói chung, nhưng hơi bỏ lỡ vấn đề implementation.

Trong implementation kiểu phổ biến:

[
L_{BS}=CE(z+\log\pi,y)
]

còn Logit Adjustment:

[
L_{LA}=CE(z+\tau\log\pi,y)
]

Nếu:

[
\tau=1
]

thì hai objective thực chất giống nhau.

Stage 20 đặt `logit_adjustment_tau=1.0`.

Vì vậy việc chúng cho:

```text
Accuracy             0.4274 = 0.4274
Macro-F1 supported   0.3491 = 0.3491
Macro-F1 22          0.2539 = 0.2539
Primary F1           0.4315 = 0.4315
```

rất có thể là **kết quả được kỳ vọng từ hai loss mathematically equivalent trong cấu hình này**, chứ không phải một coincidence thú vị.

Trong paper cuối, tôi sẽ:

* hoặc bỏ một trong hai experiment;
* hoặc thay (\tau\neq1);
* hoặc ghi rõ rằng đây là equivalence sanity check.

---

# 12. Ablation Table 14 có một vấn đề thiết kế khá lớn

Table 14 nhìn rất đẹp:

| Config           |  Binary F1 |  Coarse F1 | Fine F1 22 |  Hier. acc |
| ---------------- | ---------: | ---------: | ---------: | ---------: |
| Proposed         |     0.9775 |     0.5880 |     0.3380 |     0.2863 |
| No consistency   | **0.9797** |     0.5782 |     0.3369 | **0.3387** |
| No SupCon        |     0.9742 | **0.6106** |     0.2827 |     0.1411 |
| Balanced sampler |     0.9647 |     0.5462 | **0.3535** |     0.1774 |
| WLC only         |     0.9683 |     0.5412 | **0.3675** | **0.3387** |

Nhưng source cho thấy Stage 40 không có một trial:

```text
full proposed control
```

được chạy trong cùng Stage 40 config.

“Proposed” trong Table 14 là Stage 30 result, trong khi Stage 40 ablations dùng config khác.

Ví dụ source hiện tại:

```text
Stage 30:
epochs = 25
early_stopping_patience = 6

Stage 40:
epochs = 20
early_stopping_patience = 5
```

Stage 40 base config được thể hiện trực tiếp trong source.

Do đó:

[
\text{Proposed vs No-consistency}
]

không hoàn toàn chỉ khác:

[
\lambda_{consistency}
]

mà còn có thể khác training horizon/early stopping.

Đây là lỗi ablation methodology.

### Cách sửa

Stage 40 nên tự chứa:

```text
A0 Full proposed
A1 - Binary auxiliary
A2 - Consistency
A3 - SupCon
A4 CE instead of BS
A5 Balanced sampler
...
```

và **A0–A5 phải cùng exact config** ngoài đúng biến được ablate.

---

# 13. Chính ablation cũng cho thấy proposed model chưa được tối ưu tốt

Một ablation tốt lý tưởng sẽ cho pattern tương đối:

```text
Full model > remove component
```

ít nhất trên primary metric đã định trước.

Nhưng hiện tại:

### Bỏ consistency

Fine F1 22:

[
0.3380\rightarrow0.3369
]

gần như không thay đổi.

Primary F1 lại:

[
0.4157\rightarrow0.4539
]

**tốt hơn**.

Hierarchical accuracy:

[
0.2863\rightarrow0.3387
]

cũng tốt hơn.

Consistency loss hiện chưa chứng minh được giá trị.

---

### Class-balanced sampler

Fine F1 supported:

[
0.4647\rightarrow0.4861
]

Fine F1 22:

[
0.3380\rightarrow0.3535
]

Primary:

[
0.4157\rightarrow0.4690
]

đều tốt hơn proposed.

---

### WLC-only

Fine F1 supported:

[
0.5053
]

Fine F1 22:

[
0.3675
]

Primary:

[
0.5075
]

đều vượt proposed.

Tất nhiên population thay đổi nên đây không phải ablation thuần, và report nói đúng điều này.

Nhưng tổng thể cho thấy:

> **“Proposed” hiện chưa phải một clearly dominant model.**

---

# 14. Vì vậy tôi sẽ đổi framing của paper

Hiện tại nếu paper được dựng theo câu chuyện:

> “Chúng tôi đề xuất một hierarchical long-tail method mới và nó cải thiện CystoDS.”

thì reviewer khá dễ đánh.

Bằng chứng hiện tại không mạnh cho narrative đó.

Tôi cho rằng câu chuyện khoa học tốt hơn nhiều là:

> **“CystoDS tưởng như dễ nếu chỉ benchmark ROI/non-ROI, nhưng patient-level multi-granular evaluation cho thấy một khoảng cách rất lớn giữa ROI detection và fine-grained recognition; hierarchical/long-tail methods tạo các trade-off và thậm chí có thể gây rare-class collapse.”**

Đây là một câu chuyện **hay hơn và trung thực hơn**.

Kết quả chính trở thành:

```text
Binary: AUROC ≈ 1.0
             │
             │ huge granularity gap
             ▼
Coarse: macro-F1 ≈ 0.59
             │
             ▼
Fine: macro-F1/22 ≈ 0.34
             │
             ▼
Rare class correction có thể collapse
```

Đó là một contribution nghiên cứu thực sự thú vị.

---

# 15. Novelty của model không cao, nhưng novelty của benchmark khá tốt

Về method:

```text
Swin-Tiny
+ multi-head
+ Balanced Softmax
+ consistency KL
+ SupCon
```

đều là thành phần đã biết.

Không có một kiến trúc mới mạnh về mặt ML theory.

Do đó tôi sẽ đánh giá:

| Phần                                          |             Novelty |
| --------------------------------------------- | ------------------: |
| Backbone                                      |                thấp |
| Balanced Softmax                              |                thấp |
| SupCon                                        |                thấp |
| Consistency loss                              |     thấp–trung bình |
| Multi-head hierarchy                          |     trung bình-thấp |
| **Multi-level CystoDS benchmark**             | **cao hơn đáng kể** |
| **Long-tail clinical error characterization** |         **khá tốt** |
| **Patient/provenance protocol**               |         **rất tốt** |

Proposal ban đầu cũng định hướng rất rộng: 5-fold CV, external validation, WLC-only, ROI-level, nhiều backbone và paired statistics. 

Đây là nơi paper cuối có thể tạo giá trị.

---

# 16. Một điểm rất đáng khen: report dám ghi negative result

Ví dụ report thẳng thắn ghi:

> proposed model không đứng đầu coarse classification.

và:

> PreMalignant guardrail failed.

Nó cũng không tạo Grad-CAM giả khi thiếu trained checkpoint, không tạo p-value khi thiếu proposed prediction CSV và không gọi architecture-equivalent latency là trained-model latency. 

Đây là một quality rất tốt.

Đặc biệt trong medical AI, report kiểu:

```text
"checkpoint unavailable
→ therefore not evaluated"
```

tốt hơn rất nhiều so với:

```text
"generate đại một hình đẹp để hoàn thiện paper"
```

---

# 17. Nhưng provenance hiện vẫn có một lỗ hổng source-code

Report tuyên bố source snapshot và provenance khá mạnh.

Tuy nhiên Stage 30 hiện snapshot:

```python
stage_30_run_proposed_method.py
cystods_core.py
cystods_hf.py
README.md
```

không thấy:

```text
cystods_science.py
```

trong `REQUIRED_SOURCE_FILES`.

Stage 90 cũng tương tự.

Trong khi `cystods_science.py` chứa logic scientific metrics/prior/gate quan trọng.

Vì vậy claim:

> mọi source ảnh hưởng experiment đều được snapshot/hash

hiện chưa hoàn toàn đúng.

Tôi sẽ sửa provenance collector thành dependency closure ít nhất:

```text
stage file
cystods_core.py
cystods_science.py
cystods_hf.py
README
```

---

# 18. Missing proposed prediction CSV là vấn đề nghiêm trọng hơn vẻ ngoài

Report nói manifest kỳ vọng:

```text
predictions/holdout/test_image_predictions.csv
```

nhưng file không còn trên disk. 

Metrics JSON vẫn giúp tái dựng:

* F1,
* confusion matrix,
* per-class counts,
* aggregate error directions.

Nhưng mất prediction CSV nghĩa là không thể làm:

```text
prediction ↔ filename
prediction ↔ patient
prediction ↔ modality
prediction ↔ confidence
prediction ↔ image
```

Và vì vậy mất:

* paired McNemar;
* paired patient bootstrap;
* calibration curve;
* ECE/Brier;
* threshold analysis;
* error-by-BLC/WLC;
* error-by-patient;
* case gallery;
* Grad-CAM theo lỗi;
* ROI aggregation.

Với research artifact, prediction CSV gần như **quan trọng ngang checkpoint**.

Tôi sẽ xem đây là artifact P0 phải phục hồi.

May mắn là không cần retraining; report đúng khi nói chỉ cần exact checkpoint và rerun inference.

---

# 19. WLC/BLC analysis đáng được đưa cao hơn trong paper

Data audit cho thấy modality lệch rất mạnh theo subclass.

Ví dụ:

* CIS: 49.3% BLC
* BenignNOS: 45.4%
* InflammationNOS: 45.0%
* nhiều anatomical/foreign subclasses: 0% BLC. 

Do đó modality có thể trở thành shortcut:

[
\text{blue appearance}
\rightarrow
\text{specific disease probabilities}
]

WLC-only result trong Table 15 vì vậy không phải một side experiment vô thưởng vô phạt.

Nó là một **domain-confounding experiment quan trọng**.

Tôi sẽ nâng nó thành một research question rõ ràng:

> Does the model learn pathology or acquisition modality?

Và sau khi phục hồi prediction CSV, nên chạy:

[
P(\hat y|y,\ modality)
]

tách WLC/BLC, đặc biệt cho:

* CIS;
* InflammationNOS;
* BenignNOS;
* malignant grading.

---

# 20. Binary result gần hoàn hảo cũng phải đọc trong bối cảnh prevalence nhân tạo

Full CystoDS có:

[
6386
]

Normal mucosa images, khoảng 79%. 

Nhưng benchmark chỉ giữ:

[
540
]

Normal mucosa.

Tức test environment là một **paper-like balanced benchmark**, không phải real-world cystoscopy stream.

Report đã nói đúng điều này.

Vì vậy:

[
Specificity=94.84%
]

trên benchmark không cho phép tính thẳng clinical alert burden trong một video thực tế.

Ví dụ nếu hàng nghìn frame normal xuất hiện trong procedure, 5% image-level FP sẽ tạo workload hoàn toàn khác.

Nếu mục tiêu clinical là second reader, tương lai nên có:

```text
false alarms / minute
false positive event / procedure
patient-level sensitivity
lesion-level sensitivity
```

chứ không chỉ image-level F1.

---

# 21. ROI-level evaluation là phần thiếu rất đáng tiếc

Data audit cho thấy dataset có nhiều ảnh của cùng ROI và trung bình khoảng 3.49 images/ROI ở các ROI có metadata. 

Trong clinical reasoning, dự đoán:

```text
frame 1
frame 2
frame 3
frame 4
```

thường không phải bốn quyết định độc lập.

Chúng thuộc:

```text
one lesion / one ROI
```

Do đó một kết quả rất đáng giá sẽ là:

[
P(y|\text{ROI})
===============

Aggregate[P(y|image_i)]
]

với:

* mean;
* vote;
* max;
* learned attention.

Nếu ROI F1 tăng đáng kể so với image F1 thì đây là một contribution ứng dụng rất đẹp mà paper CystoDS gốc chưa khai thác sâu.

---

# 22. Grad-CAM hiện chưa thiếu nhiều giá trị khoa học như có vẻ

Report dành cả một section cho kế hoạch Grad-CAM.

Tôi đồng ý không nên tạo Grad-CAM bằng random/ImageNet weights.

Nhưng tôi cũng sẽ **không ưu tiên Grad-CAM trước Stage 90**.

Thứ tự giá trị khoa học theo tôi là:

```text
matched experiments
>
paired statistics
>
final CV
>
external evaluation
>
ROI analysis
>
calibration
>
Grad-CAM
```

Grad-CAM đẹp nhưng không cứu được một experiment design chưa clean.

Hơn nữa, segmentation subset CystoDS có selection bias mạnh: 768 masks tập trung chủ yếu vào LowGrade/HighGrade, AirBubble, UreteralOrifice và một vài lớp khác. 

Vì thế quantitative Grad-CAM localization cũng chỉ đại diện cho một subset chọn lọc.

---

# 23. Một điểm tôi sẽ sửa trong kế hoạch final Stage 90

Report đề xuất:

> 5-fold CV, ≥3 seeds, mean ± SD và CI.

Điều đó tốt.

Nhưng **không nên xem 5 fold như 5 independent samples để tạo CI kiểu**

[
\bar x \pm 1.96\frac{s}{\sqrt5}.
]

Bệnh nhân mới là independent unit.

Cách mạnh hơn là:

```text
5-fold CV
     ↓
mỗi patient có exactly one OOF prediction / seed
     ↓
pool OOF predictions
     ↓
patient-level bootstrap
     ↓
95% CI
```

Nếu có 3 seed, có thể:

```text
patient bootstrap
×
seed variation
```

hoặc báo tách:

* OOF patient-bootstrap CI;
* seed mean ± SD.

Như vậy statistical story sạch hơn nhiều.

---

# 24. Cẩn thận nhất với test-set adaptation từ bây giờ

Đây là điểm cực kỳ quan trọng.

Current hold-out đã được dùng để:

* so baseline;
* screen long-tail;
* evaluate proposed;
* evaluate ablations;
* phát hiện PreMalignant collapse;
* phân tích error;
* quyết định rằng rare prior phải redesign.

Tức test này bây giờ thực chất đã trở thành:

> **development test**.

Report tự gọi nó là development hold-out, đây là điều đúng. 

Vì vậy **không nên**:

```text
thấy PreMalignant collapse trên test
→ sửa model
→ quay lại test
→ báo đây là final test
```

Đó là adaptive test overfitting.

Từ thời điểm này, tôi sẽ coi:

```text
Stage 00–40 holdout
= development evidence
```

và:

```text
Stage 90 outer CV / external cohort
= confirmatory evidence
```

Đây là framing sạch nhất.

---

# 25. Tôi sẽ không gọi model hiện tại là “final proposed model”

Kết quả cho thấy:

```text
No consistency
→ tốt hơn ở vài metric

Class-balanced sampler
→ tốt hơn fine metrics

WLC-only
→ tốt hơn fine metrics

Binary+coarse multitask
→ tốt hơn coarse metric
```

và PreMalignant gate còn fail.

Do vậy tên hợp lý hơn cho Stage 30 hiện tại là:

> **development hierarchical candidate**

hoặc:

> **hierarchical baseline candidate**

chứ chưa phải final method.

---

# 26. Hướng chỉnh paper mà tôi thấy mạnh nhất

Thay vì đặt contribution chính là:

> **“A novel hierarchical model for CystoDS.”**

tôi sẽ đặt là:

> **“A patient-disjoint multi-level benchmark revealing the granularity gap and long-tail failure modes in bladder cystoscopy classification.”**

Sau đó hierarchical method là **một case study/model family** dùng để khảo sát câu hỏi đó.

Câu chuyện sẽ thành:

```text
CystoDS có 5 coarse / 22 fine labels
              │
              ▼
Paper gốc chủ yếu benchmark binary
              │
              ▼
Chúng tôi xây patient-disjoint multi-level benchmark
              │
              ▼
Binary gần saturated
AUROC = 0.9994
              │
              ▼
Nhưng coarse/fine giảm mạnh
F1 = 0.588 / 0.338
              │
              ▼
Long-tail correction không đơn giản
              │
              ▼
Rare-class collapse: PreMalignant 69 false predictions
              │
              ▼
Cần support-aware evaluation,
hierarchical constraints, CV và external validation
```

Tôi thấy narrative này **mạnh hơn hẳn** narrative “model mới đạt SOTA”.

---

# 27. Những việc tôi sẽ làm trước khi chạy thêm rất nhiều GPU

Theo thứ tự ưu tiên:

1. **Audit actual Stage 30 `config.json`** để xác định chắc chắn `balanced_softmax` hay `balanced_softmax_smoothed`; đồng bộ Stage 30/40/90 thành một canonical method. Nếu giữ standard BS, mô tả rõ training prior là gì và tách nó khỏi patient-smoothed inference calibration.
2. **Thiết kế lại Stage 40** để có `full_proposed_control` ngay trong cùng suite, cùng epoch, patience, augmentation, batch, optimizer; mọi ablation chỉ thay đúng một biến.
3. **Tạo matched baseline mới** cho Binary, Coarse, Multi-task với preprocessing/training schedule y hệt proposed; từ đó mới được nói hierarchy/loss giúp bao nhiêu.
4. **Phục hồi proposed prediction CSV** từ exact checkpoint; sau đó chạy paired bootstrap/McNemar, calibration, modality analysis và ROI mean/vote.
5. **Đóng băng rare-class policy trước confirmatory run.** Với các class 1–2 patient như PreMalignant, tôi nghiêng về tách chúng thành rare auxiliary analysis thay vì cho chúng chi phối một 22-way closed-set output chính.
6. Sau khi mọi lựa chọn đã frozen, chạy **Stage 90 5-fold × 3 seeds**. Tuyệt đối không đổi method sau khi nhìn kết quả outer folds.
7. Sau cùng mới ưu tiên external binary validation, ROI-level evaluation và explainability.

---

## Kết luận của tôi

Bản report hiện tại **tốt hơn tôi kỳ vọng khi nhìn repo lần đầu**, bởi nó đã có result thực, CI, ablation, error analysis, learning dynamics, compute accounting và quan trọng nhất là không che giấu failure. Nó đã tạo được một kết quả nghiên cứu đáng chú ý:

[
\boxed{\text{ROI detection rất dễ hơn rất nhiều so với fine diagnosis trên CystoDS}}
]

và:

[
\boxed{\text{Long-tail correction có thể tạo rare-class collapse nghiêm trọng}}
]

Đó là hai finding đủ giá trị để phát triển thành paper.

Nhưng tôi **chưa coi Table 14 hay Stage 30 là bằng chứng rằng proposed method hiện tại tốt hơn một cách đáng tin cậy**. Vấn đề lớn nhất cần sửa ngay là **matched experimental design + thống nhất định nghĩa method giữa Stage 30/40/90**. Sau khi làm hai việc đó và hoàn thành Stage 90, nghiên cứu sẽ mạnh lên rất nhiều mà không nhất thiết phải phát minh thêm một kiến trúc phức tạp.

Một lưu ý nhỏ về repository: GitHub public hiện tôi fetch thấy root chỉ có `.gitignore` và thư mục `notebook`; bản PDF comprehensive report mà tôi vừa phân tích là source material được cung cấp cho project, chưa nằm trong tree GitHub public hiện tại.
