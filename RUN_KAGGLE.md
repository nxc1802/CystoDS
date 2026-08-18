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

#### 1. Huấn luyện toàn bộ (Full Fine-tune — Mặc định):
```python
!python -m cystods run 30 --split 0 --profile research
```

#### 2. Option 1: Đóng băng Patch Embedding + Stage 1 + Stage 2 (Chống Overfit):
```python
# Đóng băng Patch Embedding + Stage 1 + Stage 2; chỉ fine-tune Stage 3, 4 và các Heads:
!python -m cystods run 30 --split 0 --profile research --freeze-stages 2
# (Hoặc dùng cờ tương đương: !python -m cystods run 30 --split 0 --profile research --partial-finetune)
```

#### 3. Option 2: Đóng băng thêm Stage 3 (Giảm 1 nửa Compute & Thời gian):
```python
# Đóng băng Patch Embedding + Stage 1 + Stage 2 + Stage 3; chỉ fine-tune Stage 4 và các Heads (~50% compute reduction):
!python -m cystods run 30 --split 0 --profile research --freeze-stage3
# (Hoặc dùng cờ tương đương: !python -m cystods run 30 --split 0 --profile research --freeze-stages 3)
```

#### 4. Chạy Stage 30 với Freeze Stage 3 duyệt qua cả 3 Splits (`split_0`, `split_1`, `split_2`):
```bash
%%bash
for split in 0 1 2; do
    echo "=========================================================="
    echo "▶ BẮT ĐẦU STAGE 30 (FREEZE STAGES 1-3, FAST) - SPLIT $split"
    echo "=========================================================="
    python -m cystods run 30 --split $split --profile research --freeze-stage3 || exit 1
done
### 🔹 Phương Pháp Mới (Stage 35): Decoupled Two-Stage Fine-Tuning (Representation Learning -> Classifier Alignment)

Phương pháp này tách rời quá trình học đặc trưng và cân bằng ranh giới phân loại:
- **Phase 1 (18 epochs):** Mở 100% Backbone + Heads, huấn luyện tự nhiên với `Cross-Entropy` + `SupCon` để học biểu diễn ngữ nghĩa tối ưu không bị méo.
- **Phase 2 (6 epochs - Selective Fine-Only):** Đóng băng 100% Backbone và khóa cứng `binary_head`, `coarse_head` (bảo toàn 100% hiệu năng đỉnh từ Phase 1), chỉ nắn `fine_head` với `Smoothed Balanced Softmax` để giải quyết triệt để mất cân bằng 22 lớp đuôi dài!

#### 1. Kiểm tra nhanh Smoke (chỉ 1 epoch mỗi phase):
```python
!python run_two_stage_hierarchical.py --split 0 --profile smoke
```

#### 2. Huấn luyện Decoupled Two-Stage (Fine-Only Alignment) trên Split 0 (~15-18 phút):
```python
!python run_two_stage_hierarchical.py --split 0 --profile research
```

#### 3. Huấn luyện Decoupled Two-Stage với chiến lược cRT (Class-Balanced Sampler ở Phase 2):
```python
!python run_two_stage_hierarchical.py --split 0 --profile research --phase2-strategy crt
```

#### 4. Huấn luyện mở toàn bộ cả 3 Heads ở Phase 2 (nếu muốn nắn cả Binary & Coarse):
```python
!python run_two_stage_hierarchical.py --split 0 --profile research --phase2-target all_heads
```

#### 5. Chạy Decoupled Two-Stage duyệt qua cả 3 Splits (`split_0`, `split_1`, `split_2`):
```python
!python run_two_stage_hierarchical.py --split all --profile research
```

---

### 🔹 Stage 40: Thực nghiệm Triệt tiêu cho Phương pháp Hai Giai đoạn (Two-Stage Ablation Studies)

Tất cả các lệnh dưới đây tự động ghi kết quả vào thư mục `result/40_ablations/two_stage/`:

#### 1. Ablation 1: Two-Stage KHÔNG CÓ SupCon (Đánh giá vai trò của Contrastive Loss ở Phase 1)
```python
# Chạy trên Split 0 (~15 phút):
!python -m cystods run 35 --split 0 --profile research --set phase1_supcon_weight=0.0

# Hoặc chạy trên cả 3 Splits (~45 phút):
!python -m cystods run 35 --split all --profile research --set phase1_supcon_weight=0.0
```

#### 2. Ablation 2: Chiến lược cRT (Class-Balanced Sampling ở Phase 2 thay vì Smoothed Balanced Softmax)
```python
# Chạy trên Split 0 (~15 phút):
!python -m cystods run 35 --split 0 --profile research --set phase2_strategy=crt

# Hoặc chạy trên cả 3 Splits (~45 phút):
!python -m cystods run 35 --split all --profile research --set phase2_strategy=crt
```

#### 3. Ablation 3: Mở cả 3 Heads ở Phase 2 (Đánh giá ảnh hưởng của việc khóa Binary & Coarse)
```python
# Chạy trên Split 0 (~15 phút):
!python -m cystods run 35 --split 0 --profile research --set phase2_target=all_heads

# Hoặc chạy trên cả 3 Splits (~45 phút):
!python -m cystods run 35 --split all --profile research --set phase2_target=all_heads
```

#### 4. Ablation 4: Selective Coarse-Only Alignment (Chỉ nắn Coarse Head ở Phase 2)
```python
# Chạy trên Split 0 (~12 phút):
!python -m cystods run 35 --split 0 --profile research --set phase2_target=coarse_only

# Hoặc chạy trên cả 3 Splits (~35 phút):
!python -m cystods run 35 --split all --profile research --set phase2_target=coarse_only
```

### 🔹 Stage 35: Decoupled Two-Stage Fine-Tuning (2S-HFT)
Quy trình 2 giai đoạn: Phase 1 (Rep CE+SupCon) $\rightarrow$ Phase 2 (Đóng băng backbone, chỉ nắn Fine Head với Smoothed BSM).

```python
# Chạy trên Split 0 (~12 phút):
!python -m cystods run 35 --split 0 --profile research

# Hoặc chạy trên cả 3 Splits (~35 phút):
!python -m cystods run 35 --split all --profile research
```

---

### 🔹 Stage 36: Three-Stage Hierarchical Fine-Tuning (3S-HFT) — [Proposed Method]
Quy trình 3 giai đoạn tuần tự: Phase 1 (Toàn mạng CE+SupCon) $\rightarrow$ Phase 2 (Đóng băng backbone, nắn Coarse Head với SBS) $\rightarrow$ Phase 3 (Đóng băng backbone & coarse, nắn Fine Head với SBS).

```python
# Chạy trên Split 0 (~15 phút):
!python -m cystods run 36 --split 0 --profile research

# Hoặc chạy trên cả 3 Splits (~45 phút):
!python -m cystods run 36 --split all --profile research
```

---

### 🔹 Stage 40: Thực nghiệm Bóc Tách (Ablation Studies)
```python
# Chạy trên Split 0 (~55 - 70 phút):
!python -m cystods run 40 --split 0 --profile research

# Hoặc chạy trên cả 3 Splits:
!python -m cystods run 40 --split all --profile research
```

---

### 🔹 Stage 90: Đánh giá 5-Fold Cross-Validation × 3 Seeds
```python
!python -m cystods run 90 --profile research
```

---

## ⚡ 6. CHẠY TOÀN BỘ CÁC GIAI ĐOẠN CÒN LẠI (Stage 30 -> 35 -> 36 -> 40 -> 90)

Nếu bạn muốn chạy toàn bộ các phase còn lại chỉ với **1 Cell duy nhất**:

```python
# Chạy toàn bộ Stage 30 -> 35 -> 36 -> 40 -> 90:
!python -m cystods run-remaining --profile research
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



