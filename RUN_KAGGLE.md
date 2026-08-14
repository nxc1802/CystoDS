# Hướng Dẫn Chạy CystoDS Trên Kaggle Notebook (`RUN_KAGGLE.md`)

Tài liệu này tổng hợp toàn bộ các câu lệnh **Kaggle Notebook Code Cell** (sử dụng tiền tố `!` hoặc `%%bash`) để bạn có thể **copy trực tiếp vào từng Cell của Kaggle Notebook** và bấm **Run Cell (Shift + Enter)**.

---

## ⏱️ 1. Bảng Dự Đoán Thời Gian Thực Thi (Kaggle GPU: T4 / P100 / L4)

Các ước lượng dưới đây dựa trên cấu hình **`profile: research`** (Ảnh 224x224, `batch_size: 32`, Mixed Precision `bfloat16`/`fp16`, 25 epochs/trial với Early Stopping):

| Stage | Mô tả | Số Trials / Split | Thời gian (1 Split) | Thời gian (Cả 3 Splits) | Ghi chú |
|---|---|---|---|---|---|
| **Stage 00** | Audit dữ liệu & Tạo 3 splits cố định (`split_0, 1, 2`) | 1 data audit | **~10 – 15 giây** | **~10 – 15 giây** | Chạy một lần duy nhất |
| **Stage 10** | Baselines: 4 Backbones × Binary/Multitask | 8 trials | **~32 – 40 phút** | **~1.6 – 2.0 giờ** | Tự động chọn backbone thắng cuộc |
| **Stage 20** | Long-tail Loss Screen: 7 Loss variants | 7 trials | **~25 – 32 phút** | **~1.3 – 1.6 giờ** | Tự động chọn loss thắng cuộc |
| **Stage 30** | Proposed Method (Hierarchical + SupCon) | 1 trial | **~5 – 7 phút** | **~15 – 20 phút** | Mô hình chính của bài báo |
| **Stage 40** | Ablation Studies (16 variants) | 16 trials | **~55 – 70 phút** | **~2.8 – 3.5 giờ** | Đánh giá đóng góp từng thành phần |
| **Stage 90** | 5-Fold Cross-Validation × 3 Seeds | 2 trials × 5 folds × 3 seeds | **~3.5 – 4.5 giờ** | N/A (CV độc lập) | Báo cáo hiệu năng tổng thể |

> [!TIP]
> **Tổng thời gian chạy tuần tự Stage 10 $\rightarrow$ Stage 20 $\rightarrow$ Stage 30**:
> - **Trên 1 Split (khuyên dùng cho thử nghiệm nhanh)**: **~65 – 75 phút** (rất an toàn so với giới hạn 12h của Kaggle Session).
> - **Trên cả 3 Splits (phục vụ đầy đủ số liệu bài báo)**: **~3.2 – 3.8 giờ**.

---

## 🚀 2. Setup Môi Trường & Cài Đặt Package (Cell 1)

Copy toàn bộ khối này vào **Cell 1** của Kaggle Notebook:

```python
# --- Cell 1: Setup Environment & Install Package ---
!export PYTHONPATH=src:$PYTHONPATH
!pip install -e . --quiet
!nvidia-smi
```

---

## 🧪 3. Thử Nghiệm Nhanh Chế Độ Smoke (Kiểm Tra Không Bug)

Nếu bạn muốn chạy thử nhanh với `profile: smoke` (1 epoch để kiểm tra toàn bộ pipeline chạy mượt mà trước khi train thật):

### 🔹 Chạy thử Smoke riêng Stage 20 (chỉ chạy loss focal):
```python
!python -m cystods run 20 --split 0 --profile smoke --trials focal
```

### 🔹 Chạy thử Smoke riêng Stage 10 (chỉ chạy Swin-Tiny):
```python
!python -m cystods run 10 --split 0 --profile smoke --models swin_tiny
```

### 🔹 Chạy thử Smoke riêng Stage 30 (Proposed Model):
```python
!python -m cystods run 30 --split 0 --profile smoke
```

---

## 🎯 4. Chạy Huấn Luyện Nối Tiếp Stage 10 $\rightarrow$ 20 $\rightarrow$ 30 (Research Mode)

### 📌 Lựa chọn A: Chạy tuần tự trên Split 0 (Khuyến nghị, ~70 phút)

Copy vào một Cell Kaggle để chạy nối tiếp Stage 10, 20, 30 trên `split_0`:

```python
# --- Cell 2A: Run Stage 10 -> 20 -> 30 on Split 0 ---
!python -m cystods run 10 --split 0 --profile research
!python -m cystods run 20 --split 0 --profile research
!python -m cystods run 30 --split 0 --profile research
```

---

### 📌 Lựa chọn B: Chạy tuần tự qua cả 3 Splits (`split_0`, `split_1`, `split_2`, ~3.5 giờ)

Dùng magic `%%bash` ở đầu Cell để chạy vòng lặp duyệt qua cả 3 splits:

```bash
%%bash
# --- Cell 2B: Run Stage 10 -> 20 -> 30 Across All 3 Splits ---
for split in 0 1 2; do
    echo "=========================================================="
    echo "▶ BẮT ĐẦU PROTOCOL SPLIT $split"
    echo "=========================================================="
    python -m cystods run 10 --split $split --profile research && \
    python -m cystods run 20 --split $split --profile research && \
    python -m cystods run 30 --split $split --profile research || exit 1
done
```

---

## 🛠️ 5. Lệnh Chạy Từng Stage Riêng Lẻ Trên Kaggle

### 🔹 Stage 00: Tạo giao thức phân hoạch 3 splits
```python
!python -m cystods run 00 --profile research
```

### 🔹 Stage 10: Huấn luyện Baseline 4 Backbones
```python
!python -m cystods run 10 --split 0 --profile research
```
*(Nếu chỉ muốn chạy riêng Swin-Tiny: `!python -m cystods run 10 --split 0 --profile research --models swin_tiny`)*

### 🔹 Stage 20: Sàng lọc 7 hàm loss đuôi dài
```python
!python -m cystods run 20 --split 0 --profile research
```
*(Nếu chỉ muốn test các loss cụ thể: `!python -m cystods run 20 --split 0 --profile research --trials fine_balanced_softmax_smoothed,fine_focal`)*

### 🔹 Stage 30: Huấn luyện Proposed Method (Hierarchical + SupCon)
```python
!python -m cystods run 30 --split 0 --profile research
```

### 🔹 Stage 40: Thực nghiệm Triệt tiêu (Ablation Studies)
```python
!python -m cystods run 40 --split 0 --profile research
```

### 🔹 Stage 90: Đánh giá 5-Fold Cross-Validation
```python
!python -m cystods run 90 --profile research
```

---

## 🔍 6. Kiểm Tra Cấu Hình (Config Inspection)

```python
!python -m cystods config show --stage 30 --split 0 --profile research
```

---

## 📦 7. Nén & Tải Kết Quả Về Máy Sau Khi Chạy Xong

Copy vào Cell cuối cùng của Notebook để nén kết quả và hiển thị link tải:

```python
# --- Cell 3: Zip & Download Results ---
!zip -q -r cystods_results.zip result/
from IPython.display import FileLink
print("Click link bên dưới để tải kết quả về máy:")
FileLink("cystods_results.zip")
```



