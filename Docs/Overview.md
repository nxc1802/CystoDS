Tôi đã đọc trực tiếp repo `nxc1802/CystoDS`, bao gồm pipeline theo stage, core training/evaluation, module science, ablation và final cross-validation. Kết luận ngắn: **đây đã là một research pipeline nghiêm túc chứ không phải notebook thử nghiệm**, nhưng hiện có **2 lỗi thiết kế quan trọng cần sửa trước khi chạy full experiment**; nếu không, Stage 30/40 và Stage 90 sẽ không thực sự đánh giá cùng một “proposed method”.

## 1. Dự án thực sự đang nghiên cứu gì?

Paper CystoDS gốc chủ yếu benchmark bài toán đơn giản hơn: phân biệt **ROI vs Non-ROI** trên ảnh cystoscopy. Repo này mở rộng dữ liệu thành bài toán **hierarchical + long-tailed classification**:

```text
Cystoscopy image
      │
      ▼
   Swin-Tiny
 shared encoder
      │
 ┌────┼──────────────┬──────────────┐
 ▼    ▼              ▼              ▼
Binary head      Coarse head      Fine head     Projection head
ROI/Non-ROI      5 classes        22 classes       SupCon
                                      │
              hierarchy consistency ──┘
```

Taxonomy trong code là:

* Binary: `Non-ROI`, `ROI`.
* Coarse: `Malignant`, `Non-malignant`, `Normal mucosa`, `Anatomical landmarks`, `Foreign bodies`.
* Fine: **22 subclasses** nằm dưới 4 coarse class không phải Normal mucosa. `Normal mucosa` dùng `fine_id=-1`, hoàn toàn không tạo “class fine thứ 23”.

Đây là hướng nghiên cứu hợp lý hơn nhiều so với chỉ làm binary classification, vì nó khai thác được cấu trúc nhãn có sẵn của CystoDS.

---

## 2. Pipeline nghiên cứu

Cấu trúc repo được tổ chức theo stage:

```text
Stage 00
Data audit + freeze protocol
        │
        ├── Stage 10 → baseline
        │
        ├── Stage 20 → long-tail loss screening
        │
        ├── Stage 30 → proposed hierarchical method
        │
        └── Stage 40 → ablations
                       │
                chốt final method
                       ▼
                   Stage 90
              5-fold CV × 3 seeds

Stage 60 → optional external validation
```

README quy định Stage 10–40 đều dùng **chính xác cùng một patient-disjoint 70/15/15 hold-out** từ Stage 00. Stage 90 mới chạy final cross-validation. Đây là một quyết định rất tốt vì tránh việc mỗi baseline/proposed method gặp một split khác nhau rồi so sánh thiếu công bằng.

Stage 10 có 5 baseline khá rõ ràng: binary-only, coarse-only, fine-only CE, multitask Binary+Coarse, và multitask Binary+Coarse+Fine.

Stage 20 sau đó chỉ tập trung vào fine classification và screen 7 long-tail objective: CE, weighted CE, focal, Balanced Softmax, smoothed Balanced Softmax, Logit Adjustment và LDAM.

Stage 40 có bộ ablation tương đối đầy đủ: flat fine CE, multitask không hierarchy, hierarchical CE, bỏ binary auxiliary, bỏ consistency, bỏ SupCon, class-balanced sampler, WLC-only training và train-all/eval-WLC.

Cuối cùng Stage 90 chạy 2 phương pháp quan trọng trên **5 folds × 3 seeds**: multitask CE và proposed hierarchical method. Như vậy riêng Stage 90 research đã tương đương **30 training runs**.

---

## 3. Proposed method hiện tại

Về ý tưởng, tôi đánh giá method khá ổn. Objective về bản chất là:

[
L =
L_{binary}
+L_{coarse}
+L_{fine}
+0.25L_{consistency}
+0.10L_{SupCon}
]

Fine head xử lý long-tail. Với phiên bản cuối ở Stage 90, nó dùng smoothed Balanced Softmax với prior có thể lấy theo **số patient thay vì số image**, rồi dùng validation để chọn inference prior correction (\tau).

Điểm tôi thích nhất là consistency loss **không phải một regularizer đặt tên cho đẹp**. Implementation thật sự ép hai quan hệ:

```text
Binary prediction ↔ tổng xác suất các coarse ROI classes

Fine prediction
      ↓ aggregate theo parent
Coarse prediction
```

Code dùng symmetric KL divergence cho cả hai.

SupCon cũng được implement đúng kiểu contrastive learning: khi bật SupCon, dataset sinh **hai augmented views**, ghép chúng thành batch và dùng projection head; không phải lấy một embedding đơn rồi gọi nó là contrastive learning.

---

# 4. Những điểm rất mạnh

Phần protocol/reproducibility của repo tốt hơn khá nhiều code nghiên cứu thông thường.

Stage 00 split **theo patient chứ không theo image** và code còn kiểm tra lại PID overlap sau khi materialize split. Đối với dataset nội soi, điều này đặc biệt quan trọng vì cùng một bệnh nhân có nhiều ảnh rất tương đồng; image-level random split có thể làm accuracy tăng giả tạo.

Long-tail metrics cũng được xử lý cẩn thận. Repo báo đồng thời `macro_f1_supported` và `macro_f1_all_classes`, nên một fine class không được model dự đoán hoặc không xuất hiện trong một split không thể bị âm thầm biến mất khỏi denominator. Module science còn cố tình không renormalize/correct probability đầu vào sai mà raise lỗi.

Patient prior cũng là một lựa chọn đáng chú ý. Nếu một bệnh nhân có 100 ảnh của cùng bệnh, image-count prior sẽ coi nó như 100 quan sát độc lập. Dùng patient-count prior hợp lý hơn đối với bài toán y khoa này.

Inference calibration (\tau) được chọn trên validation set trong mỗi epoch, lưu cùng **best checkpoint**, và khi final evaluation thì code đọc lại chính (\tau) của best checkpoint. Như vậy tôi không thấy leakage test ở đoạn này.

Confidence interval cũng bootstrap ở **patient level**, không phải image level. Đây là lựa chọn đúng với cấu trúc dữ liệu clustered.

Checkpoint lifecycle cũng rất chặt: training stage có thể upload best checkpoint lên HF, xác minh commit immutable + bytes + SHA-256 rồi mới xóa local checkpoint. External Stage 60 sau này phải tải lại đúng checkpoint receipt đó. README còn quy định external cohort chỉ được evaluation, không refit hay tune trên external data.

Repo cũng đã có test riêng cho core contracts, HF checkpoint logic và scientific metrics.

---

# 5. Các vấn đề cần sửa **trước khi chạy full experiment**

| Mức         | Vấn đề                                                                                                                | Tại sao quan trọng                                                                             | Nên sửa                                             |
| ----------- | --------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| 🔴 Critical | **Stage 30 tên “smoothed” nhưng thực tế dùng `balanced_softmax` thường**                                              | Stage 30 không train đúng proposed method cuối cùng                                            | đổi thành `balanced_softmax_smoothed`               |
| 🔴 Critical | **Stage 40 ablation cũng lấy `balanced_softmax` làm full model**, trong khi Stage 90 dùng `balanced_softmax_smoothed` | Ablation đang phân rã một method khác với method cuối                                          | đồng bộ Stage 30/40/90                              |
| 🔴 Critical | Source snapshot **bỏ sót `cystods_science.py`**                                                                       | Đây là module quyết định metrics, prior, gate nhưng không được provenance-hash cùng experiment | thêm science.py vào mọi `REQUIRED_SOURCE_FILES`     |
| 🟠 High     | `ablation_multitask_no_hierarchy` chọn checkpoint bằng coarse F1, proposed chọn bằng composite                        | Thay đồng thời architecture/loss **và checkpoint criterion**, tạo confound                     | cùng dùng composite nếu cả 3 heads tồn tại          |
| 🟠 High     | Stage 90 CI cuối dùng mean ± 1.96 SE trên 5 folds                                                                     | 5 folds không phải 5 independent samples                                                       | dùng OOF patient bootstrap / hierarchical bootstrap |
| 🟠 High     | Một số stage hardcode `EXPECTED_PROTOCOL_SHA256`                                                                      | Auto-discovery Stage 00 mới có thể tìm đúng path nhưng vẫn giữ SHA cũ                          | default `None`, lấy SHA của discovered Stage 00     |
| 🟡 Medium   | Có `stage_10_simplified_baselines.py` nhưng README tuyên bố nó không thuộc pipeline                                   | Hai Stage 10 dễ gây chạy nhầm protocol                                                         | xóa/legacy-folder/đổi tên                           |
| 🟡 Medium   | Root repo chưa có README/LICENSE/environment lock/CI                                                                  | Giảm reproducibility khi public/publish                                                        | bổ sung trước release                               |
| 🟡 Medium   | Chưa có experimental result trong repo                                                                                | Có methodology tốt nhưng chưa thể nói method tốt hơn baseline                                  | chạy Stage 10–40 rồi Stage 90                       |

### Lỗi quan trọng nhất: Stage 30/40/90 không cùng một method

Đây là chỗ tôi khuyên **sửa ngay trước khi tiêu GPU**.

Stage 30 có experiment tên:

```python
proposed_hierarchical_swin_smoothed
```

nhưng trial lại override:

```python
"fine_loss": "balanced_softmax"
```

Stage 40 cũng đặt base method:

```python
"fine_loss": "balanced_softmax"
```

Trong khi Stage 90 final proposed method lại dùng:

```python
"fine_loss": "balanced_softmax_smoothed"
```

Điều này **không chỉ khác cái tên**. Trong core implementation:

```text
balanced_softmax
→ canonical_log_prior
→ image count

balanced_softmax_smoothed
→ smoothed_log_prior
→ patient/image prior tùy fine_prior_source
→ alpha / power / max_ratio có tác dụng
```

Do đó ở Stage 30/40, dù config vẫn ghi:

```python
fine_prior_source = "patient_count"
fine_prior_smoothing_alpha = 1.0
fine_prior_power = 0.5
fine_prior_max_ratio = 50
```

**các tham số đó không tham gia Balanced Softmax training loss**. Chúng chủ yếu còn tác động qua inference calibration.

Nói cách khác:

```text
Stage 30 proposed ≠ Stage 90 proposed
Stage 40 full model ≠ Stage 90 proposed
```

Nếu paper sau này tuyên bố ablation chứng minh từng thành phần của final method, reviewer hoàn toàn có thể bắt lỗi này.

---

# 6. Một vấn đề reproducibility khá kín

Mỗi run snapshot source code để khóa provenance — đây là ý tưởng rất tốt.

Nhưng Stage 00 chẳng hạn chỉ snapshot:

```python
stage_00_prepare_protocol.py
cystods_core.py
cystods_hf.py
README.md
```

Trong khi `cystods_core.py` trực tiếp:

```python
import cystods_science as science
```

và `science.py` chứa class prior, metric, hierarchical composite và scientific gates.

Do đó ta có thể:

1. chạy experiment;
2. lưu source snapshot;
3. sau đó sửa `cystods_science.py`;
4. nhưng snapshot của run **không chứa phiên bản science.py đã thật sự dùng**.

Đây là lỗ hổng provenance thực sự.

Tốt nhất không chỉ thêm file đó thủ công, mà nên để provenance collector tự hash toàn bộ **local project dependency closure**.

---

# 7. So với paper CystoDS gốc

Điểm quan trọng là **repo này không nên được mô tả là đơn thuần reproducing paper**.

Paper gốc có dataset **8,067 images / 160 patients / 5 coarse classes / 22 subclasses**, nhưng experiment chính trong paper rút bài toán xuống binary ROI vs non-ROI.

Trong PDF gốc mà bạn đã cung cấp, Swin Transformer ở internal test có khoảng:

```text
Sensitivity  0.846
Specificity  0.809
Accuracy     0.831
Precision    0.866
F1           0.856
```

External test khoảng:

```text
Sensitivity  0.853
Specificity  0.890
Accuracy     0.873
Precision    0.870
F1           0.862
```

Repo hiện tại đi xa hơn nhiều bằng cách đặt mục tiêu **joint binary + coarse + fine classification**, đặc biệt tập trung fine-level long tail.

Một chi tiết repo đã xử lý đúng là không tự nhận “paper-exact”. README ghi rõ public metadata chứa **998 malignant images**, trong khi binary subset của paper ghi **994**, và paper không công bố 4 filename bị loại. Vì vậy nếu không có inclusion manifest chính xác, kết quả phải gọi là **paper-like**.

Thậm chí PDF gốc còn có một inconsistency khá thú vị: phần Methods ghi Non-ROI gồm 421 anatomical landmarks + 41 foreign bodies, trong khi figure phân bố toàn dataset ghi 211 anatomical + 251 foreign bodies. Tổng đều bằng 462 nhưng phân phối hai class bị đảo rất lớn. Đây càng là lý do Stage 00 audit/fingerprint của dự án rất cần thiết.

[PDF CystoDS gốc](sandbox:/mnt/data/CystoDS_a_multiclass_endoscopy_image_dataset_for_a.pdf)

---

# 8. Tôi đánh giá mức độ hoàn thiện hiện tại

**Scientific protocol: ~8.5/10.** Patient-level splitting, fixed development holdout, final CV, hierarchical metrics, patient bootstrap, rare-class audit và external evaluation policy đều tốt.

**ML implementation: ~8/10.** Multi-head model, real SupCon, consistency loss, long-tail objectives, checkpoint calibration và fail-fast được implement thực sự chứ không chỉ tồn tại trong proposal.

**Reproducibility engineering: ~8/10 về ý tưởng, ~7/10 hiện tại.** SHA/fingerprint/HF receipt rất mạnh, nhưng omission của `cystods_science.py` là lỗ hổng cần sửa.

**Publication readiness hiện tại: khoảng 6–7/10**, chủ yếu không phải vì method yếu mà vì **chưa có result để chứng minh hypothesis** và Stage 30/40/90 đang không thống nhất fine loss.

Điều tôi **chưa thể kết luận từ repo hiện tại** là “proposed method có tốt hơn flat/multitask hay không”. Code mới chứng minh rằng experiment được thiết kế để kiểm tra câu hỏi đó; không có result thì chưa thể kết luận hiệu quả khoa học.

---

## 9. Hướng tôi sẽ chỉnh trước khi bắt đầu chạy GPU

Tôi sẽ coi **final method canonical** là:

[
\boxed{
L=
L_{bin}
+L_{coarse}
+L_{BS-smoothed}^{fine}
+0.25L_{hier}
+0.10L_{SupCon}
}
]

với **patient-count smoothed prior**, rồi đồng bộ Stage 30, Stage 40 và Stage 90 theo cùng định nghĩa này. Sau đó mới chạy Stage 00 → 10 → 20 → 30 → 40, chốt hypothesis/hyperparameters, **không đụng test nữa**, rồi chạy Stage 90 `5 folds × 3 seeds`. Stage 60 chỉ nên chạy sau khi final checkpoint/method đã hoàn toàn frozen.

Nếu xét tổng thể, **hướng nghiên cứu hiện tại hoàn toàn đủ nghiêm túc để phát triển thành paper**. Việc quan trọng nhất lúc này không phải thêm model/loss mới, mà là sửa những inconsistency trên và bắt đầu thu result theo protocol đã khóa.
