# CystoDS staged pre-notebook pipeline

Đây là implementation nhiều Python pre-notebook cho
`Docs/CystoDS_Hierarchical_Long_Tailed_Research_Proposal.md`. Pipeline dùng
duy nhất backbone Swin-Tiny và một hold-out cố định 70/15/15 để các so sánh ở
Stage 10--40 dùng đúng cùng tập bệnh nhân. Cross-validation chỉ chạy ở stage
cuối cùng.

Không có mock/synthetic data, checkpoint giả, model fallback, precision
fallback hoặc device fallback. Input thiếu, dữ liệu sai, protocol/SHA lệch,
Hugging Face upload không được xác minh hoặc acceleration không được hỗ trợ sẽ
raise. Khi lifecycle của run đã bắt đầu, lỗi được ghi vào `training.log` và
`run_status.json`.

## Bản đồ stage hiện hành

| File | Thực hiện | Input bắt buộc về logic |
|---|---|---|
| `stage_00_prepare_protocol.py` | Audit dữ liệu; tạo một hold-out patient-disjoint cố định 70% train, 15% validation, 15% test; đóng băng taxonomy và protocol SHA-256 | CystoDS thật |
| `stage_10_run_baselines.py` | Đúng 5 chế độ Swin-Tiny: binary, coarse, fine, multitask Binary + Coarse, multitask Binary + Coarse + Fine | Stage 00 |
| `stage_20_run_long_tail_screen.py` | Kiểm chứng loss trên Swin-Tiny fine để làm rõ vấn đề long-tail | Stage 00 |
| `stage_30_run_proposed_method.py` | Chạy hierarchical method đề xuất với smoothed patient prior | Stage 00 |
| `stage_40_run_ablations.py` | Ablation các thành phần của method đề xuất | Stage 00 |
| `stage_60_evaluate_external.py` | Evaluation-only tùy chọn trên external cohort thật, tải exact checkpoint theo Hugging Face receipt | Stage 00, completed selected run, HF receipt và external cohort |
| `stage_90_run_cross_validation_and_report.py` | Chạy final 5-fold cross-validation và kết hợp báo cáo cuối | Stage 00 |

Pipeline chỉ gồm các entry point trong bảng trên; Swin-Tiny là backbone duy
nhất.

Stage number thể hiện thứ tự nghiên cứu khuyến nghị, không tạo dependency dây
chuyền giữa các output training. Stage 10, 20, 30 và 40 đều bind trực tiếp vào
Stage 00 và có thể chạy độc lập; không stage nào cần model/output của stage
ngay trước nó. Stage 90 cũng bind trực tiếp Stage 00 và tự chạy final CV. Stage
60 là ngoại lệ có chủ ý vì evaluation external phải biết chính xác selected
run và remote checkpoint cần đánh giá.

Các module dùng chung:

- `cystods_core.py`: audit, split, model, training/evaluation, logging và
  artifact lifecycle.
- `cystods_science.py`: metric long-tail/hierarchical, primary taxonomy,
  prior, active-class mask và scientific gates.
- `cystods_hf.py`: upload/download checkpoint nghiêm ngặt, xác minh immutable
  commit, bytes và SHA-256.

## Pre-notebook contract

Mỗi `stage_*.py` dùng Jupytext percent format:

1. Cell 1 khai báo/cài nếu thiếu toàn bộ dependencies, import và chạy
   `pip check`.
2. Cell 2 chứa toàn bộ parameter có thể cấu hình: path, profile, protocol,
   trial, model, loss, loader, optimizer, acceleration, Hugging Face và output.
3. Các cell sau là entry point; import module không tự chạy stage.

Các file stage, `cystods_core.py`, `cystods_science.py`, `cystods_hf.py` và
`README.md` phải ở cùng thư mục. Strict provenance snapshot source từ disk, vì
vậy nếu chuyển một stage sang `.ipynb` vẫn phải upload các file `.py` sidecar.

## Protocol 70/15/15 và binding

Stage 00 là stage duy nhất quyết định primary hold-out. Split được thực hiện ở
mức patient, không phải ở mức ảnh:

- train: 70%;
- validation: 15%;
- test: 15%;
- không patient ID nào được xuất hiện ở hai split;
- split, dataset fingerprint và primary fine taxonomy được đóng băng;
- `protocol_manifest.json` được bind bằng SHA-256.

Hai giá trị downstream được ghi trong
`reports/protocol_reference.json`:

```text
CYSTODS_PROTOCOL_RUN_DIR=/absolute/path/to/stage_00_run
CYSTODS_EXPECTED_PROTOCOL_SHA256=<64-hex-digest>
```

Nên set tường minh cả hai biến cho một research run. Nếu
`CYSTODS_PROTOCOL_RUN_DIR` chưa được set, Stage 10/20/30/40/90 sẽ tìm completed
Stage 00 mới nhất có cùng profile trong các result root được hỗ trợ; SHA cũng
được lấy từ run đó khi chưa được cung cấp. Auto-discovery thuận tiện cho một
workspace đơn, nhưng explicit path + SHA an toàn hơn khi có nhiều protocol
run.

Không sửa `protocol_manifest.json` hoặc `splits/` sau Stage 00. Nếu dataset,
inclusion policy hoặc split thay đổi, chạy Stage 00 mới và dùng binding mới.
Stage 90 tạo final 5-fold CV ở cuối quy trình, đồng thời vẫn bind vào dataset
identity và taxonomy đã audit ở Stage 00.

## Remote-only checkpoint trên Hugging Face

Stage training dùng backend `huggingface`. Cần tối thiểu:

```bash
export HF_TOKEN=<write-token>
export CYSTODS_HF_REPO_ID=<namespace/model-repository>
```

Các biến tùy chọn:

```bash
export CYSTODS_HF_REVISION=main
export CYSTODS_HF_PRIVATE=true
export CYSTODS_HF_CREATE_REPO=true
export CYSTODS_HF_PATH_PREFIX=cystods/my_run_group
```

Với mỗi leaf/fold, code tạo `best_model.pt`, upload lên HF, lấy commit chính
xác, kiểm tra metadata/LFS SHA, tải lại từ chính commit đó vào temporary
directory và đối chiếu bytes + SHA-256. Chỉ sau khi tất cả kiểm tra pass, local
`best_model.pt` mới bị xóa và receipt mới được publish. Nếu upload hoặc verify
fail, stage fail; không ghi receipt thành công giả.

Completed local result không giữ file `.pt`. Local giữ các artifact cần phân
tích như JSON/CSV, log, predictions, metrics, visualization, report và HF
receipt; checkpoint thật nằm trên Hub. Không xóa receipt vì Stage 60 dùng nó để
tải đúng immutable revision và kiểm tra lại checksum.

## Cài local venv và chạy kiểm tra

Khuyến nghị Python 3.11:

```bash
cd /Volumes/WorkSpace/Project/CystoDS
uv venv --python 3.11 .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install pytest ruff
.venv/bin/python -m pytest notebook/tests -q
.venv/bin/python -m ruff check notebook
.venv/bin/python -m compileall -q notebook
```

Cell 1 của mỗi stage vẫn tự cài/check runtime dependencies khi chạy như
pre-notebook. Venv ở trên giúp phát hiện lỗi source/test trước khi dùng GPU.

## Smoke test

Smoke profile dùng dữ liệu thật, CPU, một epoch, `image_size=64` và không tải
pretrained weights. Swin-Tiny được khởi tạo tường minh cho resolution 64x64 để
giảm thời gian test; research vẫn dùng 224x224. Smoke chỉ kiểm tra execution
path và artifact lifecycle, không phải kết quả khoa học.

```bash
export CYSTODS_RUN_PROFILE=smoke
export CYSTODS_DATA_ROOT=/Volumes/WorkSpace/Project/CystoDS/xvdhy-osfstorage-archive
export CYSTODS_RESULT_ROOT=/Volumes/WorkSpace/Project/CystoDS/result/local_validation
export HF_TOKEN=<write-token>
export CYSTODS_HF_REPO_ID=<namespace/model-repository>

.venv/bin/python notebook/stage_00_prepare_protocol.py
```

Đọc `downstream_environment` từ completed Stage 00
`reports/protocol_reference.json`, export hai biến trong đó, rồi chạy riêng
stage muốn test:

```bash
.venv/bin/python notebook/stage_10_run_baselines.py
.venv/bin/python notebook/stage_20_run_long_tail_screen.py
.venv/bin/python notebook/stage_30_run_proposed_method.py
.venv/bin/python notebook/stage_40_run_ablations.py
.venv/bin/python notebook/stage_90_run_cross_validation_and_report.py
```

Không cần chạy tất cả lệnh trong một lần. Mỗi lệnh là một complete stage run.

## Cấu hình cho một GPU 96 GB trên Marimo server

Research defaults nhắm đến một CUDA accelerator 96 GB:

- Swin-Tiny, input 224x224, BF16;
- TF32, channels-last, fused AdamW và `torch.compile` mode `max-autotune`;
- Stage 10/20: train batch 512;
- Stage 30/40/90: train batch 256 vì hierarchical/SupCon tốn memory hơn;
- eval batch 1024;
- train workers 16, eval workers 8, prefetch factor 2.
- chỉ train workers persistent; validation/final-eval workers được giải phóng.

Các giá trị loader chính có thể override mà không sửa source:

```bash
export CYSTODS_BATCH_SIZE=512
export CYSTODS_EVAL_BATCH_SIZE=1024
export CYSTODS_NUM_WORKERS=16
export CYSTODS_EVAL_NUM_WORKERS=8
export CYSTODS_PREFETCH_FACTOR=2
export CYSTODS_EVAL_PREFETCH_FACTOR=2
export CYSTODS_NUM_CPU_THREADS=32
```

Fused optimizer và các acceleration controls còn lại nằm rõ trong Cell 2;
research defaults bật fused AdamW.

Không tăng đồng thời workers và prefetch quá cao: pinned queue có thể chuyển
nút thắt từ VRAM sang RAM/IO. Code không tự giảm batch hoặc tắt acceleration
khi OOM/unsupported; hãy thay parameter tường minh và chạy lại. Trong Marimo,
có thể chạy stage bằng shell cell hoặc materialize Jupytext thành notebook;
source `.py` vẫn là canonical artifact để Agent chỉnh sửa.

## Chạy lần lượt trên Kaggle

Upload nguyên thư mục `notebook/` làm Kaggle Dataset hoặc copy toàn bộ source
files vào cùng một thư mục dưới `/kaggle/working`. Bật Internet để cài package,
tải pretrained weights và giao tiếp với Hugging Face. Tạo Kaggle Secret tên
`HF_TOKEN` có quyền ghi repository.

Cell cấu hình đầu tiên:

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["CYSTODS_RUN_PROFILE"] = "research"
os.environ["CYSTODS_DATA_ROOT"] = "/kaggle/input/datasets/cuongnguyen1802/cystods"
os.environ["CYSTODS_RESULT_ROOT"] = "/kaggle/working/result"
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["CYSTODS_HF_REPO_ID"] = "<namespace>/<model-repository>"
os.environ["CYSTODS_HF_PRIVATE"] = "true"
```

1. Chạy Stage 00 một lần:

   ```python
   %run /kaggle/working/cystods-notebook/stage_00_prepare_protocol.py
   ```

2. Bind chính xác completed run vừa tạo:

   ```python
   import json
   from pathlib import Path

   protocol_run = Path(COMPLETED_RUN_DIRECTORY)
   reference = json.loads(
       (protocol_run / "reports" / "protocol_reference.json").read_text()
   )
   os.environ.update(reference["downstream_environment"])
   ```

3. Chạy Stage 10 và tải result reports về sau khi hoàn tất:

   ```python
   %run /kaggle/working/cystods-notebook/stage_10_run_baselines.py
   ```

4. Chạy Stage 20 để quan sát ảnh hưởng của các loss trên cùng Swin-Tiny fine:

   ```python
   %run /kaggle/working/cystods-notebook/stage_20_run_long_tail_screen.py
   ```

5. Chạy method đề xuất và ablation. Hai stage này độc lập nên có thể đặt ở hai
   Kaggle sessions/jobs khác nhau, miễn dùng đúng Stage 00 path/SHA và source:

   ```python
   %run /kaggle/working/cystods-notebook/stage_30_run_proposed_method.py
   %run /kaggle/working/cystods-notebook/stage_40_run_ablations.py
   ```

6. Sau khi đã chốt finalist/hyperparameters, chạy Stage 90 cuối cùng:

   ```python
   %run /kaggle/working/cystods-notebook/stage_90_run_cross_validation_and_report.py
   ```

Stage 90 research chạy 5 folds và nhiều seeds/trials theo Cell 2 nên tốn thời
gian nhất. Integrated report nằm trong parent Stage 90 result; fold-level
metrics, predictions, logs, visualizations và HF receipts nằm trong sibling
child runs. Giữ toàn bộ JSON/CSV/log/visualizations/report khi tải output về;
không cần tải checkpoint `.pt` vì receipt trỏ tới checkpoint đã xác minh trên
HF.

Để tách session Kaggle, persist/upload Stage 00 result và set lại explicit
`CYSTODS_PROTOCOL_RUN_DIR` cùng `CYSTODS_EXPECTED_PROTOCOL_SHA256`. Không dựa
vào auto-discovery nếu Stage 00 không còn ở filesystem của session mới.

Materialize một stage thành `.ipynb` nếu cần:

```bash
.venv/bin/jupytext --to ipynb \
  notebook/stage_30_run_proposed_method.py \
  --output /tmp/stage_30_run_proposed_method.ipynb
```

Giữ file `.py` gốc cạnh notebook để source snapshot pass.

## External validation tùy chọn

Stage 60 không train, không tune threshold trên external cohort và không có
mock fallback. Nó yêu cầu input tường minh:

```bash
export CYSTODS_SELECTED_RUN_DIR=/absolute/path/to/completed/selected_leaf
export CYSTODS_HF_CHECKPOINT_RECEIPT_JSON=/absolute/path/to/hf_checkpoint_receipt.json
export CYSTODS_EXTERNAL_MANIFEST_CSV=/absolute/path/external_manifest.csv
export CYSTODS_EXTERNAL_IMAGE_ROOT=/absolute/path/external_images
export CYSTODS_PROTOCOL_RUN_DIR=/absolute/path/to/stage_00_run
export HF_TOKEN=<read-token>

.venv/bin/python notebook/stage_60_evaluate_external.py
```

Receipt phải thuộc selected completed run và bind đúng protocol. Checkpoint
được tải vào temporary workspace theo exact HF commit, xác minh bytes/SHA-256,
dùng cho evaluation rồi không trở thành local result artifact. External
manifest cần các cột mặc định `path`, `binary_label`, `patient_id`; tên cột có
thể đổi bằng các biến `CYSTODS_EXTERNAL_*_COLUMN` trong Cell 2.

## Result và logging

Stage 00 lưu tối thiểu:

```text
result/stage_00_prepare_protocol_<profile>_<timestamp>/
├── protocol_manifest.json
├── artifact_manifest.json
├── config.json
├── run_status.json
├── reports/
│   ├── protocol_reference.json
│   ├── data_audit.json
│   └── run_summary.json
├── splits/holdout/
├── source/
└── system/
```

Mỗi training suite có parent result và sibling child-run root:

```text
result/
├── stage_XX_<profile>_<timestamp>/
│   ├── artifact_manifest.json
│   ├── config.json
│   ├── run_status.json
│   ├── logs/training.log
│   ├── reports/
│   ├── source/
│   └── system/child_run_root.json
└── stage_XX_<profile>_<timestamp>__runs/
    └── <trial>_seed_<seed>_<profile>_<timestamp>/
        ├── artifact_manifest.json
        ├── config.json
        ├── run_status.json
        ├── logs/
        ├── metrics/
        ├── predictions/
        ├── reports/
        ├── source/
        ├── splits/
        ├── system/
        └── visualizations/
```

Training log ghi stage/trial/fold, epoch, step, optimizer update, loss,
samples/second, learning rate, validation monitor, checkpoint decision, early
stopping và final status. Metrics/predictions/learning curves/confusion
matrices/recall plots, environment inventory và HF receipts được lưu cùng run.
JSON được ghi atomic và không chấp nhận NaN/Infinity; artifact manifest được
finalize sau khi logger đóng.

## Scientific invariants

- Fine head có đúng 22 published subclasses. `Normal mucosa` có `fine_id=-1`
  và bị mask khỏi fine loss/metrics; không tạo class thứ 23.
- Patient ID không được giao nhau giữa train/validation/test.
- Fine classes vắng ở training dùng active-class mask và policy
  `mask_and_score_zero`. Metrics báo cả supported-class và all-class macro F1
  để không che zero-support class.
- Primary fine taxonomy đóng băng ở Stage 00, không được chọn lại dựa trên
  validation/test hoặc riêng từng experiment.
- Hierarchical checkpoint selection dùng composite metric với trọng số
  explicit cho coarse all-class macro F1, primary all-class macro F1 và
  hierarchical accuracy.
- Rare-class prediction share được audit và scientific gate có thể làm stage
  fail; không sửa kết quả bằng fallback.
- WLC-only test subset là evaluation view; `train_modality="WLC"` là ablation
  training riêng.
- External validation là evaluation-only; không fit lại model hoặc tune trên
  external cohort.

Public metadata có 998 malignant images, trong khi binary subset của paper báo
994 nhưng không công bố bốn filename bị loại và exact split. Nếu không có
`inclusion_manifest_csv`, báo cáo phải gọi kết quả là paper-like, không phải
paper-exact.
