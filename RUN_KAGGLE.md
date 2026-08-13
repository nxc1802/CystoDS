# Hướng Dẫn Chạy CystoDS Trên Kaggle Notebook (`RUN_KAGGLE.md`)

Tài liệu này tổng hợp toàn bộ các câu lệnh Notebook Code Cell để thực thi pipeline **CystoDS** trên Kaggle GPU (Tesla T4 / P100).

---

## 1. Setup Môi Trường & Cài Đặt Package

Chạy cell này trong Kaggle Notebook để nạp nguồn và cài đặt `cystods` ở chế độ editable:

```python
# --- Cell 1: Environment Setup & Installation ---
import os
import sys

# Thêm thư mục src vào PYTHONPATH
sys.path.append("./src")

# Cấu hình biến môi trường
os.environ["CYSTODS_RUN_PROFILE"] = "research"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Cài đặt cystods package
!pip install -e . --quiet

# Kiểm tra CUDA GPU
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
```

---

## 2. Câu Lệnh Chạy Toàn Bộ Pipeline (Run All)

Chạy lệnh duy nhất để tự động thực thi chuỗi 6 Stages của CystoDS theo thứ tự phụ thuộc (`00 -> 10 -> 20 -> 30 -> 40 -> 90`):

```python
# --- Cell 2: Run Full Pipeline (Run All) ---
!python -m cystods run all --profile research
```

---

## 3. Câu Lệnh Chạy Riêng Từng Stage

Nếu muốn chạy riêng biệt hoặc debug từng Stage:

### 🔹 Stage 00: Chuẩn bị Giao thức & Phân hoạch Dữ liệu (Protocol Split)
```python
!python -m cystods run 00 --profile research
```

### 🔹 Stage 10: Huấn luyện Baseline 4 Backbones (Swin-Tiny, HRNet, ResNet152, ResNeXt50)
```python
!python -m cystods run 10 --profile research
```

### 🔹 Stage 20: Sàng lọc 7 Hàm Loss Đuôi Dài (Long-Tail Loss Screening)
Lệnh này sẽ tự động duyệt và huấn luyện trọn bộ 7 biến thể Loss (`fine_cross_entropy`, `fine_weighted_ce`, `fine_focal`, `fine_balanced_softmax`, `fine_logit_adjustment`, `fine_ldam`, `fine_balanced_softmax_smoothed`) và chọn ra phương pháp chiến thắng:
```python
!python -m cystods run 20 --profile research
```

### 🔹 Stage 30: Phương Pháp Đề Xuất (Hierarchical + Balanced Softmax + SupCon)
```python
!python -m cystods run 30 --profile research
```

### 🔹 Stage 40: Thực Nghiệm Triệt Tiêu (16 Ablation Configurations)
```python
!python -m cystods run 40 --profile research
```

### 🔹 Stage 90: Báo Cáo Tổng Hợp K-Fold Cross-Validation (5-Fold × 3 Seeds)
```python
!python -m cystods run 90 --profile research
```

---

## 4. Kiểm Tra Cấu Hình Giải Mã (Config Inspection)

Để kiểm tra danh sách thử nghiệm và cấu hình giải mã của một Stage trước khi chạy:

```python
# Display resolved configuration for Stage 20
!python -m cystods config show --stage 20 --profile research
```

---

## 5. Tải Kết Quả Thực Nghiệm Sau Khi Chạy

Để nén và tải về toàn bộ thư mục kết quả `result/` từ Kaggle Output:

```python
# --- Cell Zip & Download Artifacts ---
!zip -r cystods_results.zip result/
from IPython.display import FileLink
FileLink("cystods_results.zip")
```
