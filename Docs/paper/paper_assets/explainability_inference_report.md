# Báo cáo khả thi: Explainability và inference benchmark

## Kết luận ngắn

Benchmark suy luận ở mức **kiến trúc** đã hoàn tất mà không huấn luyện thêm. Mô hình được dựng đúng graph đã công bố — Swin-Tiny, projection head và ba head phân cấp — và có đúng **28.230.679 tham số**, khớp tuyệt đối với `holdout_model_info.json`. Trên Apple MPS, FP32, batch 1, forward của mô hình đạt trung bình **12,081 ms/ảnh** (82,776 ảnh/giây); pipeline từ đọc ảnh đến đầu ra đạt trung bình **15,000 ms/ảnh** (66,665 ảnh/giây).

Grad-CAM có ý nghĩa dựa trên mô hình đã huấn luyện **chưa thể chạy trong môi trường hiện tại**, vì checkpoint duy nhất nằm trong Hugging Face repository riêng tư, bản local đã bị xóa có chủ đích, không còn trong cache, và môi trường không có token Hugging Face. Thử tải ẩn danh trả về HTTP 401. Không sinh heatmap từ trọng số ngẫu nhiên vì hình như vậy không có giá trị khoa học.

## Ràng buộc checkpoint đã kiểm chứng

Receipt bất biến của mô hình đề xuất khai báo:

- Repository: `Cuong2004/CystoDS-Checkpoints` (private).
- Commit: `2dc6122b0af33f605e30ef329baa3d81a4101db9`.
- Path: `cystods_hierarchical_long_tailed_2026/stage_30_run_proposed_method/68d782ed2517d2d849d5/holdout/best_model.pt`.
- SHA-256: `554abf5c42a3a1f0e049e9ddba97c02e4677341ce399aa728cfaa1a3ad3ad68d`.
- Dung lượng: 112.988.870 byte (107,755 MiB).
- `local_checkpoint_removed=true`; tìm trên local/cache không thấy bản sao.
- `HfApi().whoami()` trả `LocalTokenNotFoundError`; tải ẩn danh trả `401 RepositoryNotFoundError` như kỳ vọng với repository riêng tư.

## Inference benchmark đã chạy

### Phạm vi và phương pháp

- Thiết bị: Apple Silicon MPS, 10 CPU logic/physical, RAM hợp nhất 16 GiB.
- Phần mềm: Python 3.11.15, PyTorch 2.13.0, torchvision 0.28.0, timm 1.0.28.
- Kiến trúc: `swin_tiny_patch4_window7_224.ms_in1k`, đầu vào 224×224, channels-last, projection 128 chiều, head nhị phân 2 lớp, coarse 5 lớp và fine 22 lớp.
- Precision: FP32; `torch.inference_mode()`; không `torch.compile`.
- Mỗi phép đo forward được đồng bộ thiết bị bằng `torch.mps.synchronize()`; dùng `time.perf_counter()`.
- Warm-up 15 vòng. Batch 1 và 8 đo 60 vòng; batch 32 đo 20 vòng.
- Số liệu forward không gồm đọc ảnh, tiền xử lý và host-to-device transfer. Benchmark end-to-end batch 1 được đo riêng.
- Giá trị tensor là ngẫu nhiên vì checkpoint riêng tư chưa truy cập được. Điều này **không thay đổi graph dense, kích thước tensor hay phép toán**, nên phù hợp để đo runtime/allocation của kiến trúc; tuyệt đối không dùng để suy ra dự đoán, độ chính xác hay explainability.

### Kết quả model-forward

| Batch | Số vòng | Mean (ms/batch) | Median (ms/batch) | P95 (ms/batch) | Mean (ms/ảnh) | Thông lượng (ảnh/s) | MPS allocated sau warm-up (MiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 60 | 12,081 | 12,983 | 13,367 | 12,081 | 82,776 | 110,834 |
| 8 | 60 | 81,334 | 81,745 | 83,835 | 10,167 | 98,360 | 113,987 |
| 32 | 20 | 318,539 | 317,317 | 326,516 | 9,954 | 100,459 | 127,999 |

Mô hình sau khi chuyển sang MPS chiếm 108,761 MiB theo `current_allocated_memory`; con số gần với kích thước lý thuyết của 28,23 triệu tham số FP32. `driver_allocated_memory` phản ánh allocator/driver dùng chung và không phải peak memory độc lập của mô hình, vì vậy không nên báo nó như VRAM peak.

### Tiền xử lý và end-to-end batch 1

Ảnh test `0bceb179.png` được đọc từ filesystem đã warm cache, crop tâm 92%, resize 224×224, chuyển tensor và chuẩn hóa ImageNet:

| Phạm vi | Vòng đo | Mean (ms/ảnh) | Median (ms/ảnh) | P95 (ms/ảnh) | Thông lượng (ảnh/s) |
|---|---:|---:|---:|---:|---:|
| Decode + crop + resize + normalize | 60 | 1,291 | 1,235 | 1,613 | 774,872 |
| Decode + preprocess + transfer + model-forward | 60 | 15,000 | 15,592 | 15,842 | 66,665 |

Các con số trên là benchmark trên một máy cụ thể, không phải đặc tính phổ quát của mô hình. Một paper nên ghi đầy đủ thiết bị, precision, batch, warm-up, số vòng, phạm vi timer và trạng thái cache như trên. Không được so trực tiếp với số liệu GPU trong bài khác nếu giao thức phần cứng khác.

Artifact định lượng:

- `inference_benchmark_architecture_only.json`: provenance, môi trường và toàn bộ số liệu chưa làm tròn.
- `inference_benchmark_architecture_only.csv`: bảng model-forward.
- `inference_benchmark_architecture_only.png`: biểu đồ độ trễ/thông lượng.
- `run_architecture_inference_benchmark.py`: mã chạy lại.

## Grad-CAM / explainability: thiết kế sẵn sàng khi có checkpoint

### Phương pháp phù hợp với Swin

Layer mục tiêu đã được kiểm tra bằng forward pass là `encoder.layers[-1].blocks[-1].norm1`, có activation **7×7×768**. Với logit lớp mục tiêu \(y^c\), gradient theo activation \(A\) được lấy tại layer này; trọng số kênh là trung bình gradient theo hai trục không gian, sau đó tính:

\[
\alpha_k^c = \frac{1}{HW}\sum_{i,j}\frac{\partial y^c}{\partial A_{ijk}}, \qquad
L_{\mathrm{GradCAM}}^c = \operatorname{ReLU}\left(\sum_k \alpha_k^c A_k\right).
\]

Heatmap 7×7 được nội suy bilinear về 224×224 và overlay trên đúng ảnh sau center-crop/resize mà mô hình nhìn thấy. Nên tạo riêng heatmap cho fine head và coarse/binary head để kiểm tra tính nhất quán phân cấp. Với Balanced Softmax calibration, phép điều chỉnh prior chỉ cộng hằng số theo lớp nên gradient của logit lớp đã chọn không đổi; tuy nhiên lớp mục tiêu phải được chọn từ xác suất **đã calibration**.

### Đánh giá định lượng có thể chạy ngay sau khi mở checkpoint

Test hold-out có 329 ảnh/24 bệnh nhân; **109 ảnh có polygon segmentation**:

| Fine class có mask | Ảnh test có mask |
|---|---:|
| LowGradePapillary | 15 |
| HighGradePapillary | 33 |
| UreteralOrifice | 21 |
| AirBubble | 40 |
| **Tổng** | **109** |

Do đó explainability không nên chỉ là vài hình minh họa. Có thể báo trên toàn bộ 109 ảnh:

- Pointing game: điểm cực đại của CAM có nằm trong polygon hay không.
- Energy-inside-mask: tỷ lệ tổng saliency nằm trong mask.
- IoU và Dice sau ngưỡng percentile định trước (ví dụ top 20% activation), kèm sensitivity analysis top 10%/30%.
- Báo cả tập 109 ảnh và tập con dự đoán đúng; tập con dự đoán đúng là phân tích localization chính để tránh trộn lỗi phân loại với lỗi định vị.
- Bootstrap theo bệnh nhân cho khoảng tin cậy, không bootstrap độc lập theo ảnh.
- Phân tầng theo fine class và modality WLC/BLC; ghi rõ các lớp không có mask không thể kết luận localization định lượng.

Panel định tính nên được chọn theo quy tắc định trước (không cherry-pick): mẫu đúng có energy-inside-mask ở phân vị 10/50/90, mẫu sai có confidence cao, bao phủ WLC/BLC và bốn lớp có mask. Mỗi panel cần hiện ảnh input, heatmap, overlay + polygon, true/predicted label và confidence.

## Danh sách đầu vào cần cung cấp để hoàn tất Grad-CAM

1. Đặt biến môi trường `HF_TOKEN` với quyền **read** cho `Cuong2004/CystoDS-Checkpoints`, hoặc cung cấp file local `best_model.pt` có SHA-256 đúng bằng `554abf5c42a3a1f0e049e9ddba97c02e4677341ce399aa728cfaa1a3ad3ad68d`.
2. Cho phép tải checkpoint tạm thời (~107,8 MiB); có thể xóa ngay sau khi xác minh SHA-256 và sinh artifact.
3. Nếu muốn inference benchmark mang tính triển khai thay vì benchmark máy local, cần chỉ định phần cứng đích (ví dụ CPU server, NVIDIA GPU cụ thể, Apple Silicon), precision và batch size phục vụ thực tế.

Không cần training thêm cho Grad-CAM, đánh giá localization trên 109 mask, hay benchmark checkpoint. Chỉ cần quyền đọc đúng checkpoint đã huấn luyện.
