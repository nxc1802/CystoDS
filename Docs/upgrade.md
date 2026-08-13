Đúng, nếu chỉ giữ **những thay đổi bắt buộc**, mình sẽ thiết kế Stage 00 mới rất gọn như sau.

## 1. Stage 00 tạo đúng 3 split

Thay vì hiện tại chỉ tạo một `holdout`, Stage 00 tạo:

```text
split_0
split_1
split_2
```

Mỗi split vẫn giữ nguyên tỷ lệ:

```text
Train = 70% patients = 112
Val   = 15% patients = 24
Test  = 15% patients = 24
```

và bắt buộc **patient-disjoint bên trong từng split**:

```text
Train ∩ Val  = ∅
Train ∩ Test = ∅
Val   ∩ Test = ∅
```

Cấu trúc artifact:

```text
result/00_protocol/research_xxx/
├── protocol_manifest.json
└── splits/
    ├── split_0/
    │   ├── train.csv
    │   ├── val.csv
    │   ├── test.csv
    │   └── summary.json
    ├── split_1/
    │   ├── train.csv
    │   ├── val.csv
    │   ├── test.csv
    │   └── summary.json
    └── split_2/
        ├── train.csv
        ├── val.csv
        ├── test.csv
        └── summary.json
```

Hiện Stage 00 chỉ tạo đúng một fixed holdout; đây là phần cần đổi.

---

## 2. Cách chọn 3 split

Giữ gần như toàn bộ cơ chế hiện tại.

Stage 00 vẫn sinh nhiều candidate, ví dụ hiện tại:

```yaml
split_search_candidates: 4096
```

Với mỗi candidate 70/15/15, tính `allocation_score`.

### Score mới chỉ cần bổ sung một thứ

Hiện tại score đã cân bằng:

```text
coarse distribution theo patient
fine distribution theo patient
coarse distribution theo số ảnh
```

Cần bổ sung:

```text
fine distribution theo số ảnh
```

Tức về cơ bản:

[
Score =
S_{\text{coarse patient}}
+
S_{\text{fine patient}}
+
S_{\text{coarse image}}
+
S_{\text{fine image}}
]

Không cần thêm modality, embedding, visual difficulty hay các thành phần phức tạp khác lúc này.

Điều này trực tiếp ngăn tình trạng hiện tại như:

```text
             Val    Test
LowGrade     113      41
HighGrade     42      95
```

mà score cũ không phạt đủ mạnh. Cơ chế hiện tại đúng là chỉ có `fine_presence` nhưng image count mới chỉ được lưu theo coarse class.

---

## 3. Chọn top 3 nhưng Test không được quá giống nhau

Sau khi có 4096 candidate:

```text
candidate_001 → score 0.72
candidate_002 → score 0.74
candidate_003 → score 0.75
...
```

sort từ score tốt nhất đến xấu nhất.

Sau đó:

```text
split_0 = candidate có score tốt nhất
```

Tiếp tục quét danh sách:

```text
split_1 = candidate tốt nhất tiếp theo
          mà Test overlap với split_0 <= 50%
```

Sau đó:

```text
split_2 = candidate tốt nhất tiếp theo
          mà:
          overlap(Test2, Test0) <= 50%
          overlap(Test2, Test1) <= 50%
```

Với 24 test patients:

[
24\times 50%=12
]

nên rule rất rõ:

```text
|Test0 ∩ Test1| <= 12
|Test0 ∩ Test2| <= 12
|Test1 ∩ Test2| <= 12
```

Đây là **pairwise constraint**.

Không cần làm phức tạp hơn.

---

# 4. Stage 00 chỉ chạy một lần

Ví dụ:

```bash
cystods run 00
```

Nó sinh cả:

```text
split_0
split_1
split_2
```

và freeze cả ba vào cùng một `protocol_manifest.json`.

Tức không phải:

```bash
cystods run 00 --seed ...
cystods run 00 --seed ...
cystods run 00 --seed ...
```

Mà chỉ:

```text
1 Stage 00 run
→ 4096 candidates
→ chọn 3 best diverse splits
→ freeze
```

Cách này sạch và reproducible hơn.

---

# 5. Thêm đúng một CLI parameter: `--split`

Đây là phần mình nghĩ nên làm rất trực tiếp.

CLI hiện tại hỗ trợ dạng:

```bash
cystods run <stage> --profile ... --set ... --models ... --trials ...
```

nhưng chưa có parameter chọn protocol split.

Thêm:

```bash
--split {0,1,2}
```

Ví dụ:

```bash
cystods run 10 --split 0
```

nghĩa là:

> Stage 10 sử dụng `split_0` của Stage 00.

Tương tự:

```bash
cystods run 10 --split 1
cystods run 10 --split 2
```

---

# 6. CLI hoạt động thế nào bên trong

Trong `cli.py` thêm:

```python
run_parser.add_argument(
    "--split",
    type=int,
    choices=[0, 1, 2],
    default=None,
)
```

Sau khi `load_config()`:

```python
config["protocol_split_index"] = args.split
```

Sau đó các Stage 10/20/30/40/... không tự quyết định split nữa.

Chúng chỉ đọc:

```python
config["protocol_split_index"]
```

Ví dụ:

```text
0
```

thì protocol loader load:

```text
splits/split_0/train.csv
splits/split_0/val.csv
splits/split_0/test.csv
```

Nếu:

```text
2
```

thì load:

```text
splits/split_2/...
```

---

# 7. Ví dụ chạy pipeline

### Chạy Stage 00

```bash
cystods run 00
```

Sinh:

```text
split_0
split_1
split_2
```

### Experiment trên split 0

```bash
cystods run 10 --split 0
cystods run 20 --split 0
cystods run 30 --split 0
cystods run 40 --split 0
```

Toàn bộ chuỗi này luôn dùng cùng:

```text
Train_0
Val_0
Test_0
```

### Lặp lại split 1

```bash
cystods run 10 --split 1
cystods run 20 --split 1
cystods run 30 --split 1
cystods run 40 --split 1
```

### Split 2

```bash
cystods run 10 --split 2
cystods run 20 --split 2
cystods run 30 --split 2
cystods run 40 --split 2
```

Hoặc CLI `all` hiện tại cũng có thể truyền cùng parameter xuống tất cả stages, vì `_run_all()` đã resolve config riêng cho từng stage.

Khi đó có thể hỗ trợ:

```bash
cystods run all --split 0
```

---

# 8. Nên bắt buộc `--split` cho Stage ≥ 10

Mình không khuyên có default ngầm kiểu:

```text
không truyền → tự lấy split_0
```

vì sau này rất dễ quên mình đang chạy split nào.

Tốt hơn:

```bash
cystods run 10
```

→ báo:

```text
Error: Stage 10 requires --split {0,1,2}
```

Còn:

```bash
cystods run 00
```

thì không cần `--split`, vì Stage 00 chính là stage tạo ba split.

Như vậy mỗi artifact downstream luôn ghi rõ:

```json
{
  "protocol_split_index": 1
}
```

và biết chính xác nó dựa trên cohort nào.

---

## Thiết kế cuối cùng

Chỉ cần thay đổi ba thứ:

1. **Stage 00:** từ 1 holdout → tạo **3 patient-disjoint 70/15/15 splits**.
2. **Split scoring:** giữ score cũ nhưng thêm **fine-class image distribution**; chọn 3 candidate tốt nhất với **pairwise Test overlap ≤ 50%**.
3. **CLI:** thêm `--split {0,1,2}`; mọi Stage sau Stage 00 bắt buộc parameter này và load đúng `split_0`, `split_1` hoặc `split_2`.

Đây là mức thay đổi tối thiểu nhưng giải quyết đúng vấn đề phân bổ hiện tại, đồng thời làm việc chạy lại 3 cohort rất rõ ràng và reproducible.
