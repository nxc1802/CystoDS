# Báo cáo kiểm chứng implementation staged CystoDS

Ngày cập nhật: 2026-08-03  
Phạm vi: source code sau khi tái cấu trúc pipeline Swin-Tiny, chưa thay thế cho
kết quả full research trên GPU.

## Kết luận

Pipeline hiện hành gồm `00`, `10`, `20`, `30`, `40`, `60` và `90`. Các stage
benchmark hiệu năng riêng, sealed evaluation riêng và verification/report
riêng đã được loại bỏ. Báo cáo cross-validation được tích hợp vào Stage 90.

Các invariant đã được kiểm tra bằng static contract/unit test:

- một fixed hold-out patient-disjoint 70%/15%/15%;
- split dùng `split_seed` riêng, không đổi khi model seed đổi;
- duy nhất backbone `swin_tiny_patch4_window7_224.ms_in1k`;
- Stage 10 có đúng năm cấu hình head: Binary, Coarse, Fine, Binary + Coarse,
  Binary + Coarse + Fine;
- Stage 20 chạy fine-only loss screen và report rare-class collapse;
- Stage 90 tự sinh final 5-fold CV trên dataset đã audit, rồi ghi report tổng
  hợp JSON/CSV/Markdown;
- best checkpoint được upload, xác minh bằng immutable HF commit và SHA-256,
  sau đó xóa `.pt` khỏi result local;
- Stage 60 tải exact checkpoint từ HF receipt vào thư mục tạm và xóa sau eval;
- train loader chỉ giữ train/val; final train/val/test eval được tạo tuần tự;
- research profile dùng CUDA + BF16 + TF32 + channels-last + fused AdamW +
  `torch.compile`; smoke profile tắt toàn bộ CUDA-only optimization.

## Bản đồ stage

| Stage | Vai trò | Dependency logic |
|---|---|---|
| 00 | Audit và fixed hold-out 70/15/15 | Dataset thật |
| 10 | Năm task-mode baseline trên Swin-Tiny | Chỉ protocol Stage 00 |
| 20 | So sánh loss trên Swin-Tiny Fine | Chỉ protocol Stage 00 |
| 30 | Method hierarchical đề xuất | Chỉ protocol Stage 00 |
| 40 | Ablation | Chỉ protocol Stage 00 |
| 60 | External evaluation tùy chọn | HF receipt + external cohort thật |
| 90 | Final 5-fold CV và integrated report | Dataset/taxonomy binding Stage 00 |

Stage 10, 20, 30 và 40 không tiêu thụ model hoặc output của stage trước. Stage
number là thứ tự nghiên cứu khuyến nghị. Stage 60 có dependency model có chủ ý
vì đây là evaluation-only.

## Split contract

Stage 00 tạo đúng một unit `holdout`. Tỷ lệ 70/15/15 được tối ưu ở mức patient
để ngăn leakage. Artifact split ghi đồng thời tỷ lệ patient và tỷ lệ ảnh sau
khi áp dụng paper-like Normal-mucosa sampling. Protocol manifest chứa dataset
semantic fingerprint, primary fine taxonomy và SHA-256 để downstream bind.

Stage 90 không đọc output training từ Stage 10--40 và không dùng CV folds tạo
sẵn. Nó sinh final CV ở stage cuối bằng fixed `split_seed`, đồng thời kiểm tra
dataset semantic SHA và taxonomy từ Stage 00.

## Checkpoint contract

Mỗi trial/fold chỉ publish một `best_model.pt`. Core chuẩn hóa key do
`torch.compile` tạo ra trước khi lưu model-only checkpoint. Quy trình thành
công bắt buộc:

1. upload checkpoint lên Hugging Face model repository;
2. nhận immutable commit OID;
3. kiểm tra remote path, bytes và LFS SHA-256 nếu có;
4. tải lại đúng commit vào cache tạm;
5. đối chiếu bytes và SHA-256;
6. publish receipt JSON/CSV/Markdown;
7. xóa checkpoint local và assert result không còn `.pt`.

Mọi lỗi upload/download/checksum đều raise; không tạo receipt thành công giả.

## Hiệu năng và tài nguyên

Defaults cho GPU 96 GB:

- Stage 10/20: batch 512;
- Stage 30/40/90: batch 256 vì SupCon tạo hai views;
- eval batch 1024;
- train workers/prefetch 16/2;
- eval workers/prefetch 8/2, không persistent;
- BF16, TF32, channels-last, fused AdamW, compile `max-autotune`;
- log throughput, train time, allocated/reserved/total CUDA memory.

Code không tự giảm batch, đổi precision hoặc tắt acceleration khi lỗi. Cấu
hình không được server hỗ trợ sẽ fail rõ ràng.

## Bằng chứng kiểm thử local

Đã chạy trong `.venv`:

```text
python -m py_compile notebook/cystods_core.py notebook/cystods_hf.py \
  notebook/cystods_science.py notebook/stage_*.py
python -m pytest notebook/tests -q
python -m ruff check notebook
```

Kết quả tại lần cập nhật này: `72 passed`; `py_compile` pass; `ruff` pass.
HF tests dùng client được inject và không gọi network. Vì vậy chúng kiểm chứng
transaction/checksum contract, không được trình bày như kết quả training.

Full research training, HF upload thực và Stage 90 5-fold vẫn phải chạy trên
server CUDA với dataset thật. Chỉ dùng metrics của completed run có
`run_status.json = completed` và receipt hợp lệ trong báo cáo khoa học.
