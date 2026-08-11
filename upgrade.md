Có. Và mình nghĩ **nên refactor result system trước khi chạy lại Stage 30/40 và đặc biệt trước Stage 90**. Vấn đề hiện tại không phải dự án lưu quá nhiều thông tin; ngược lại, provenance đang khá tốt. Vấn đề là **cùng một đơn vị thí nghiệm bị xé ra thành quá nhiều nhánh thư mục**, khiến con người rất khó đọc.

## 1. Cơ chế tạo `result/` hiện tại

Mỗi lần tạo một run, `make_run_directory()` sinh tên dạng:

```text
<experiment_name>_<profile>_<timestamp>/
```

và lập tức tạo **10 thư mục**:

```text
checkpoints/
logs/
metrics/
predictions/
reports/
splits/
source/
visualizations/
models/
system/
```

bất kể run đó có thực sự dùng tất cả các thư mục này hay không. Vì vậy ngay cả Stage 00 chỉ chuẩn bị protocol cũng có thể xuất hiện nhiều folder không thực sự cần thiết.

Với một training run đơn, ví dụ Stage 30 một seed + holdout, artifact của cùng một `holdout` hiện bị phân tán như sau:

```text
run/
├── checkpoints/
│   └── holdout/
│       ├── fine_prior_audit.json
│       ├── history.csv
│       ├── val_metrics_latest.json
│       └── fine_calibration_latest.json
│
├── logs/
│   ├── training.log
│   └── holdout_history.csv
│
├── metrics/
│   └── holdout/
│       ├── train_metrics.json
│       ├── val_metrics.json
│       ├── test_metrics.json
│       ├── train_losses.json
│       ├── val_losses.json
│       ├── test_losses.json
│       ├── performance.json
│       ├── patient_bootstrap_ci.json
│       └── ...
│
├── predictions/
│   └── holdout/
│       ├── train_image_predictions.csv
│       ├── val_image_predictions.csv
│       ├── test_image_predictions.csv
│       └── roi/...
│
├── splits/
│   └── holdout/
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       ├── optimization_train.csv
│       └── summary.json
│
├── visualizations/
│   └── holdout/
│       ├── training_history.png
│       ├── coarse_confusion_matrix.png
│       ├── fine_confusion_matrix.png
│       └── ...
│
├── models/
│   └── holdout_model_info.json
│
└── reports/
    └── holdout_report.md
```

Đúng nghĩa là muốn hiểu **một fold** phải nhảy qua 6–8 directory khác nhau. Current `run_single_fold()` thực sự ghi metrics, losses, predictions, bootstrap, WLC/ROI, report và figures sang các nhánh khác nhau như vậy.

Thậm chí `checkpoints/holdout/` hiện còn chứa `history.csv`, calibration và prior audit — các file không phải checkpoint. Đồng thời history lại được ghi thêm lần nữa vào `logs/holdout_history.csv`. Đây là một ví dụ rõ ràng của **semantic layout không còn khớp với nội dung**.

---

# 2. Stage suite còn làm `result/` rối hơn

Stage 10/20/30/40/90 không chạy trực tiếp một model mà chạy một **suite gồm trial × seed**.

Hiện tại code tạo:

```text
result/
├── stage_40_run_ablations_research_20260811-xxxxxx/
└── stage_40_run_ablations_research_20260811-xxxxxx__runs/
```

Tức suite parent và các child runs lại nằm thành **hai folder sibling**.

Code làm đúng như vậy:

```python
parent_run_dir = make_run_directory(...)

child_root = parent_run_dir.with_name(
    f"{parent_run_dir.name}__runs"
)
```

Bên trong `__runs/`, mỗi trial/seed lại tạo thêm một timestamp:

```text
ablation_no_supcon_seed_20260729_research_20260811-xxxxxx/
ablation_fine_ce_seed_20260729_research_20260811-xxxxxx/
...
```

Sau đó mỗi child lại có 10 directory kể trên. Suite parent lại giữ thêm `child_runs.csv`, `run_summary.json`, `stage_report.md`, trial index và hashes của các child.

Với Stage 90 hiện tại:

```text
2 models × 3 seeds × 5 folds = 30 fold trainings
```

dù chỉ có 6 child runs, mỗi child chứa 5 folds, nhưng do layout theo artifact type nên việc đọc bằng mắt vẫn rất khó.

---

# 3. Điều quan trọng: không cần hy sinh bất kỳ thông tin nào

Mình **không khuyên giảm artifact**.

Những thứ hiện có rất đáng giữ:

```text
config
dataset fingerprint
protocol SHA
split fingerprint
source snapshot
environment
pip freeze
dependency audit
train history
train/val/test metrics
train/val/test losses
predictions
bootstrap
calibration
rare-class audit
performance
model metadata
HF receipt
checkpoint SHA
figures
ROI analysis
artifact SHA256 manifest
```

Historical audit của chính project cho thấy các file này rất hữu ích khi phải truy ngược một con số trong paper về đúng prediction, metric, split và provenance nguồn.

Cần đổi **cách tổ chức**, không đổi lượng thông tin.

---

# 4. Kiến trúc mình đề xuất: `Stage → Trial → Seed → Fold`

Nguyên tắc quan trọng nhất:

> **Một fold là một đơn vị khoa học, vì vậy mọi thứ liên quan đến fold đó phải nằm cạnh nhau.**

Thay vì:

```text
metrics/fold_00/
predictions/fold_00/
splits/fold_00/
visualizations/fold_00/
```

hãy đảo hierarchy thành:

```text
folds/fold_00/
├── metrics...
├── predictions...
├── split...
└── figures...
```

## Top-level `result/`

Mình đề xuất cố định:

```text
result/
├── 00_protocol/
├── 10_baselines/
├── 20_long_tail/
├── 30_proposed/
├── 40_ablations/
├── 60_external/
└── 90_final_cv/
```

Mỗi lần chạy stage tạo một version:

```text
result/
└── 30_proposed/
    └── 20260811T145000Z/
```

Không cần lặp lại:

```text
stage_30_run_proposed_method_research_...
```

vì vị trí folder đã nói đó là Stage 30.

---

# 5. Một Stage 30 mới sẽ rất dễ đọc

Ví dụ:

```text
result/
└── 30_proposed/
    └── 20260811T145000Z/
        ├── README.md
        ├── summary.json
        ├── config.json
        ├── status.json
        ├── runs.csv
        ├── catalog.json
        ├── manifest.json
        │
        ├── provenance/
        │   ├── protocol.json
        │   ├── dataset.json
        │   ├── environment.json
        │   ├── dependencies.json
        │   ├── pip_freeze.txt
        │   └── source/
        │
        └── runs/
            └── proposed_hierarchical_swin/
                └── seed_20260729/
                    ├── README.md
                    ├── config.json
                    ├── summary.json
                    ├── manifest.json
                    │
                    └── folds/
                        └── holdout/
                            ├── README.md
                            ├── summary.json
                            ├── metrics.json
                            ├── history.csv
                            ├── predictions.csv
                            │
                            ├── split/
                            │   ├── train.csv
                            │   ├── val.csv
                            │   ├── test.csv
                            │   ├── optimization_train.csv
                            │   └── summary.json
                            │
                            ├── diagnostics/
                            │   ├── bootstrap.json
                            │   ├── calibration.json
                            │   ├── prior.json
                            │   ├── performance.json
                            │   ├── rare_class.json
                            │   └── roi/
                            │
                            ├── figures/
                            │   ├── training_history.png
                            │   ├── binary_roc_pr.png
                            │   ├── coarse_confusion.png
                            │   ├── fine_confusion.png
                            │   └── per_class_recall.png
                            │
                            └── checkpoint/
                                └── receipt.json
```

Đây vẫn chứa **100% thông tin hiện tại**, nhưng nhìn vào:

```text
runs/proposed_hierarchical_swin/seed_20260729/folds/holdout/
```

là thấy toàn bộ cuộc đời của experiment đó.

---

# 6. `metrics.json` nên consolidate nhiều file nhỏ

Hiện một fold có:

```text
train_metrics.json
val_metrics.json
test_metrics.json

train_losses.json
val_losses.json
test_losses.json

patient_bootstrap_ci.json
performance.json
wlc_only_metrics.json
roi_metrics.json
paired_mcnemar.json
...
```

Không có lý do khoa học bắt buộc chúng phải là 10 file riêng.

Có thể losslessly chuyển thành:

```json
{
  "schema_version": "cystods.fold_metrics",

  "train": {
    "metrics": {},
    "losses": {}
  },

  "validation": {
    "metrics": {},
    "losses": {}
  },

  "test": {
    "metrics": {},
    "losses": {}
  },

  "bootstrap": {},
  "performance": {},
  "calibration": {},
  "rare_class": {},
  "wlc_only": null,
  "roi": null,
  "paired_test": null
}
```

Không mất một field nào.

Chỉ là:

```text
10 JSON files
      ↓
1 structured JSON file
```

Đối với code, còn dễ load hơn:

```python
result["test"]["metrics"]["fine"]["macro_f1_all_classes"]
```

thay vì phải biết chính xác file nằm ở đâu.

---

# 7. Nhưng không nên nhét tất cả vào một file khổng lồ

Mình sẽ chia artifact theo ba loại:

| Nhóm                      | Nên làm              |
| ------------------------- | -------------------- |
| Small structured metadata | Consolidate vào JSON |
| Tabular/raw evidence      | Giữ file riêng       |
| Human visualization       | Giữ PNG              |

Ví dụ:

**Nên consolidate:**

```text
metrics
losses
performance
bootstrap
calibration
prior audit
rare gate
model info
```

→ `metrics.json` / `summary.json`.

**Nên giữ riêng:**

```text
train.csv
val.csv
test.csv
predictions.csv
history.csv
training.log
```

vì đây là dữ liệu dạng bảng/log.

**PNG vẫn giữ riêng.**

---

# 8. `README.md` mới là thứ giúp con người đọc result

Hiện artifact machine-readable tốt nhưng thiếu một **entry point cho con người**.

Mỗi stage run nên tự generate:

```markdown
# Stage 30 — Proposed Method

Status: completed
Protocol: 4af1...
Canonical method: cystods.proposed.v1
Canonical SHA: a81c...
Dataset: 8067 images / 160 patients

## Runs

| Trial | Seed | Binary AUROC | Coarse F1 | Fine F1-22 | Primary F1 | Hier. Acc |
|---|---:|---:|---:|---:|---:|---:|
| Proposed | 20260729 | ... | ... | ... | ... | ... |

## Selected recipe

Backbone: Swin-Tiny
Fine loss: Balanced Softmax
Hierarchy BC: 0.25
Hierarchy CF: 0.25
SupCon: 0.10
Calibration: validation-grid

## Navigation

- runs/...
- provenance/...
- manifest.json
```

Tức khi mở folder Stage 30, **không cần biết CystoDS internals vẫn hiểu được nó**.

Stage 40 README còn có thể tự sinh:

```text
Ablation              Changed factor            Fine F1    Δ
------------------------------------------------------------
Full Proposed          —                         ...
No SupCon              SupCon 0.10 → 0           ...
No BC hierarchy        BC 0.25 → 0               ...
CE                     BalancedSoftmax → CE       ...
```

Điều này kết hợp rất tốt với one-factor-at-a-time refactor vừa bàn ở turn trước.

---

# 9. Thêm `summary.json`: file đầu tiên machine/LLM nên đọc

Một child run không nên bắt tool hoặc researcher đọc 50 files chỉ để biết model làm gì.

Ví dụ:

```json
{
  "schema_version": "cystods.result_summary",

  "identity": {
    "stage": 30,
    "experiment": "proposed_hierarchical_swin",
    "seed": 20260729,
    "fold": "holdout"
  },

  "method": {
    "version": "cystods.proposed.v1",
    "sha256": "..."
  },

  "protocol": {
    "sha256": "...",
    "split_sha256": "...",
    "dataset_semantic_sha256": "..."
  },

  "training": {
    "best_epoch": 18,
    "monitor": "hierarchical_composite",
    "monitor_value": 0.61,
    "epochs_completed": 24
  },

  "headline_metrics": {
    "binary_auroc": 0.999,
    "binary_f1": 0.978,
    "coarse_macro_f1": 0.588,
    "fine_macro_f1_all_classes": 0.338,
    "primary_macro_f1": 0.416,
    "hierarchical_accuracy": 0.286
  },

  "scientific_gates": {
    "rare_class_collapse": "failed"
  },

  "checkpoint": {
    "backend": "huggingface",
    "commit": "...",
    "sha256": "..."
  }
}
```

Đây là một nâng cấp lớn về usability mà **không bỏ raw artifact nào**.

---

# 10. Tách `manifest.json` và `catalog.json`

Hiện `artifact_manifest.json` rất tốt cho integrity: code đi qua tất cả file, lưu:

```text
path
bytes
sha256
```

và còn có logic hỗ trợ một `runs/` nested hierarchy, kiểm tra manifest của child rồi reuse SHA thay vì re-hash các file lớn.

Nó nên được giữ.

Nhưng manifest không trả lời câu hỏi:

> File nào là canonical test metrics?

Vì vậy thêm:

### `manifest.json`

Cryptographic:

```json
[
  {
    "path": "runs/.../folds/holdout/metrics.json",
    "bytes": 14284,
    "sha256": "..."
  }
]
```

### `catalog.json`

Semantic:

```json
{
  "test_metrics":
    "runs/proposed/.../folds/holdout/metrics.json#/test/metrics",

  "test_predictions":
    "runs/proposed/.../folds/holdout/predictions.csv",

  "training_history":
    "runs/proposed/.../folds/holdout/history.csv",

  "split_manifest":
    "runs/proposed/.../folds/holdout/split/summary.json",

  "checkpoint_receipt":
    "runs/proposed/.../folds/holdout/checkpoint/receipt.json"
}
```

Hai file phục vụ hai mục đích khác nhau:

```text
manifest = file integrity
catalog  = artifact meaning
```

Đây cũng sẽ làm `generate_paper_assets.py` tốt hơn rất nhiều: script chỉ hỏi catalog `"test_metrics"` thay vì hardcode timestamp/path.

---

# 11. Bỏ cơ chế `__runs` sibling

Đây là phần mình muốn đổi nhất.

Hiện:

```text
stage_30.../
stage_30...__runs/
```

Nên đổi thành:

```text
stage-run/
└── runs/
```

Ví dụ:

```text
30_proposed/
└── 20260811T145000Z/
    ├── README.md
    ├── summary.json
    ├── manifest.json
    └── runs/
        └── proposed/
```

Điểm thú vị là `write_artifact_manifest()` **đã có sẵn logic để xử lý `run_dir / "runs"`** và verify child manifests.

Tức architecture hiện tại gần như đã chuẩn bị sẵn cho nested design, nhưng `run_training_suite()` lại tạo child ở sibling `__runs`.

Do đó refactor này thậm chí còn tự nhiên với code hiện tại.

---

# 12. Vẫn giữ immutability dù child nằm bên trong parent

Lý do code hiện tại tạo sibling là để tránh parent suite đã seal rồi lại bị child/report làm thay đổi file set. Comment trong code ghi rõ điều đó.

Không cần sibling để giải quyết vấn đề này.

Lifecycle mới:

```text
create parent
    ↓
create child
    ↓
run child
    ↓
seal child manifest
    ↓
run child tiếp theo
    ↓
all children sealed
    ↓
generate parent report/index
    ↓
seal parent manifest
    ↓
mark parent completed
```

Một khi parent `completed`:

```text
parent immutable
children immutable
```

Nếu sau này muốn sinh paper analysis mới, không sửa run cũ.

Tạo:

```text
derived/
paper_20260820/
```

và reference SHA của immutable result.

Đây còn sạch hơn architecture hiện nay.

---

# 13. Không pre-create 10 empty directories nữa

Thay:

```python
for child in (
    "checkpoints",
    "logs",
    "metrics",
    ...
):
    mkdir(...)
```

bằng **lazy creation**.

Khi ghi:

```python
write_json(path, ...)
```

thì function của bạn vốn đã có thể:

```python
path.parent.mkdir(parents=True, exist_ok=True)
```

Chỉ những folder có artifact mới xuất hiện.

Ví dụ Stage 00 mới:

```text
00_protocol/20260811T.../
├── README.md
├── config.json
├── status.json
├── protocol.json
├── manifest.json
├── provenance/
├── audit/
└── splits/
```

Không còn:

```text
models/
checkpoints/
predictions/
visualizations/
```

rỗng.

---

# 14. Một duplication hiện tại nên bỏ ngay

Có ít nhất hai duplication đáng xử lý.

### Training history

Hiện trong quá trình training:

```text
checkpoints/holdout/history.csv
```

sau đó lại:

```text
logs/holdout_history.csv
```

Chỉ cần một:

```text
folds/holdout/history.csv
```

### Split combined CSV

Current split writer materialize split-specific CSVs và còn tạo combined `cystods_split.csv`, kể cả convenience copy ở run-level trong flow hiện tại.

Nếu giữ 100% **information**, không có nghĩa phải giữ 100% **duplicate bytes**.

Canonical:

```text
fold/split/train.csv
fold/split/val.csv
fold/split/test.csv
```

cộng:

```text
fold/split/summary.json
```

là đủ.

Nếu muốn combined:

```text
fold/split/all.csv
```

chỉ giữ đúng một bản.

---

# 15. Stage 90 sau refactor sẽ dễ hiểu hơn rất nhiều

Ví dụ:

```text
result/
└── 90_final_cv/
    └── 20260820T080000Z/
        ├── README.md
        ├── summary.json
        ├── cv_results.csv
        ├── manifest.json
        └── runs/
            ├── proposed/
            │   ├── seed_20260729/
            │   │   └── folds/
            │   │       ├── fold_00/
            │   │       ├── fold_01/
            │   │       ├── fold_02/
            │   │       ├── fold_03/
            │   │       └── fold_04/
            │   ├── seed_20260730/
            │   └── seed_20260731/
            │
            └── matched_multitask/
                ├── seed_20260729/
                ├── seed_20260730/
                └── seed_20260731/
```

Chỉ cần nhìn folder tree đã hiểu ngay thiết kế experiment:

```text
2 methods
×
3 seeds
×
5 folds
```

Trong khi layout hiện tại buộc phải parse timestamped child run names rồi nhảy qua các `metrics/fold_XX`, `predictions/fold_XX`, `splits/fold_XX`.

---

# 16. Mình đề xuất tạo một `ResultStore`

Đừng tiếp tục hardcode:

```python
run_dir / "metrics" / fold_name / ...
run_dir / "reports" / ...
run_dir / "predictions" / ...
```

ở hàng chục function.

Tạo một lớp nhỏ:

```python
@dataclass
class ResultStore:
    root: Path

    def trial(self, trial_id: str) -> Path:
        ...

    def seed(self, trial_id: str, seed: int) -> Path:
        ...

    def fold(
        self,
        trial_id: str,
        seed: int,
        fold: str,
    ) -> Path:
        ...

    def write_json(self, path: Path, payload):
        ...

    def seal(self):
        ...
```

Hoặc nếu muốn ít abstraction hơn:

```python
def fold_result_dir(
    run_dir: Path,
    fold_name: str,
) -> Path:
    return run_dir / "folds" / fold_name
```

Điều quan trọng là **chỉ một module quyết định artifact paths**.

Không để `train_model`, `run_roi_evaluation`, `run_single_fold`, `Stage60` tự quyết folder riêng.

---

# 17. Refactor trực tiếp result hiện tại & Migration dữ liệu cũ

1. **Refactor trực tiếp hệ thống result hiện tại**: Sửa thẳng code hệ thống để tạo và ghi theo cấu trúc thống nhất `Stage → Trial → Seed → Fold`.
2. **Refactor dữ liệu result hiện có**: Viết script migration chuyển đổi toàn bộ các thư mục result hiện có (`stage_00_...`, `stage_10_...`) sang cấu trúc thư mục và file JSON thống nhất mới mà không làm mất bất kỳ dữ liệu hay SHA256 provenance nào.

Sau khi refactor:
- Tool downstream và pipeline code đọc/ghi trực tiếp theo cấu trúc duy nhất thông qua `catalog.json` và `ResultStore`.
- Mọi kết quả hiện có và kết quả mới chạy đều nằm trong cùng một chuẩn layout sạch sẽ.

---

# 18. Plan triển khai

| Phase                     | Công việc                                                    | Mục tiêu                                         |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------ |
| **1. Result contract**    | Chốt folder hierarchy, consolidated JSONs và artifact roles | Không sửa code trước khi layout rõ               |
| **2. ResultStore**        | Centralize toàn bộ path creation/writing                     | Không còn hardcoded paths rải rác                |
| **3. Nested suites**      | `__runs` → `runs/` bên trong stage run                       | Một stage = một directory duy nhất               |
| **4. Unit-first fold**    | `metrics/fold`, `predictions/fold` → `folds/fold/...`        | Mọi evidence của fold nằm cạnh nhau              |
| **5. Consolidation**      | merge metrics/loss/bootstrap/calibration/performance         | Giảm rất nhiều JSON nhỏ                          |
| **6. Remove duplicates**  | history, split copies, repeated summaries                    | Giữ 100% information nhưng không duplicate bytes |
| **7. Human index**        | auto-generate README ở stage/run/fold                        | Mở folder là hiểu ngay                           |
| **8. Semantic catalog**   | `catalog.json`                                               | Code/paper không hardcode path                   |
| **9. Integrity**          | giữ SHA manifest + nested child verification                 | Không giảm provenance                            |
| **10. Data Migration**    | Script refactor chuyển đổi toàn bộ data result hiện có       | Đưa toàn bộ data cũ về cấu trúc mới              |
| **11. Downstream update** | Stage60, protocol finder, paper asset generator              | Toàn pipeline dùng semantic references           |
| **12. Tests**             | round-trip/information-equivalence tests                     | Chứng minh không mất dữ liệu                     |

---

# 19. Test quan trọng nhất: chứng minh “100% information preserved”

Mình sẽ không chỉ test file tồn tại.

Tạo equivalence checker để kiểm tra kết quả trước và sau khi refactor dữ liệu:

```python
original = load_original_result(old_result_path)
migrated = load_migrated_result(new_result_path)
```

rồi kiểm tra theo artifact role:

```python
assert legacy.test_metrics == modern.test_metrics
assert legacy.val_metrics == modern.val_metrics
assert legacy.test_predictions.equals(modern.test_predictions)
assert legacy.split_fingerprint == modern.split_fingerprint
assert legacy.protocol_sha == modern.protocol_sha
...
```

Ngoài ra `ResultStore.verify()` nên bắt buộc:

```text
all catalog targets exist
all physical files belong to manifest
all SHA256 match
all completed children are sealed
parent manifest covers children
no unregistered artifact
no duplicate canonical role
```

Khi đó refactor layout không làm giảm reproducibility mà thực tế còn **tăng khả năng audit**.

---

## Kết luận

Mình đánh giá **nên refactor mạnh phần result**, nhưng không theo hướng “xóa bớt artifact”. Kiến trúc tốt hơn là:

```text
Hiện tại
Stage
 ├─ parent/
 └─ parent__runs/
      └─ run
          ├─ metrics/fold
          ├─ predictions/fold
          ├─ splits/fold
          ├─ visualizations/fold
          └─ ...

                    ↓

Đề xuất

Stage
 └─ execution
     ├─ README
     ├─ summary
     ├─ provenance
     └─ runs
         └─ trial
             └─ seed
                 └─ folds
                     └─ fold
                         ├─ metrics
                         ├─ history
                         ├─ predictions
                         ├─ split
                         ├─ diagnostics
                         ├─ figures
                         └─ checkpoint
```

**Một stage = một folder, một trial = một nhánh, một seed = một nhánh, một fold = một self-contained evidence bundle.**

Đặc biệt, vì code hiện tại đã có `artifact_manifest` hỗ trợ `run_dir/runs` nested children, việc bỏ sibling `__runs` không phải một cuộc rewrite toàn bộ provenance engine; phần nền tảng quan trọng đã có sẵn.
