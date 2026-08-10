# Audit bằng chứng kết quả CystoDS cho bản thảo

**Phạm vi:** artifact hiện có tại thời điểm 03-08-2026; không huấn luyện hoặc đánh giá lại mô hình.  
**Mục đích:** chỉ ra nguồn canonical cho metric, per-class/error analysis, split, modality/ROI, bootstrap và training; đồng thời đánh dấu các tuyên bố không an toàn.  
**Không chỉnh sửa:** `Docs/Paper_CystoDS_Hierarchical_Long_Tailed_VI.md`.

## 1. Kết luận audit ngắn

- Có **27** `test_metrics.json` và **27** CI bootstrap cấp bệnh nhân; metric tổng hợp cho các run Stage 10–40 khá đầy đủ.
- Run đề xuất có full aggregate/per-class/confusion metrics. Các image-level prediction và 6 PNG **không còn được materialize trong workspace**, nhưng `artifact_manifest.json` của run xác nhận chúng từng được tạo, kèm path, số byte và SHA-256. Vì chưa có nội dung file local nên hiện chỉ làm được error analysis theo ma trận; error gallery, Grad-CAM và paired test cần khôi phục artifact hoặc chạy lại inference.
- Chỉ có **2** đánh giá WLC-only, đều là ablation Stage 40; không có BLC-only. Không có `roi_metrics.json`, external validation, cross-validation, statistical significance, backbone khác Swin-Tiny hoặc inference benchmark.
- `result/stage_00_10_20_30_40_summary_report.md` gán sai thứ tự tên lớp trong cả coarse và fine confusion tables; không được dùng làm nguồn canonical.
- `result/rare_class_analysis_stage30.md` có nhiều tuyên bố không được artifact hỗ trợ (`p<0.01`, SupCon ngăn feature collapse, overprediction `PreMalignant` như cơ chế an toàn) và có lỗi support `Diverticulum=12` thay vì 1 ảnh test.

## 2. Bảng nguồn bằng chứng canonical

| Nội dung | Bằng chứng chính xác | Có thể dùng trong paper | Giới hạn/cảnh báo |
|---|---|---|---|
| Dataset gốc, modality, mask, support 22 lớp | `/Volumes/WorkSpace/Project/CystoDS/result/stage_00_prepare_protocol_research_20260803-035933/reports/data_audit.json` | 8.067 ảnh, 160 bệnh nhân, WLC/BLC 7.617/450, 768 mask, support ảnh/bệnh nhân từng lớp | Đây là inventory gốc, không phải tập benchmark 2.221 ảnh |
| Giao thức khóa trước và primary taxonomy | `/Volumes/WorkSpace/Project/CystoDS/result/stage_00_prepare_protocol_research_20260803-035933/protocol_manifest.json` | Protocol SHA `9b63fdb896ed2769e74b89c0949f97792ca6d9faba4eadd40186d40a7cb40c02`; split fingerprint `64a7af51ae18cece83fae7fbf17d55545c81aa2a643ce1b44337c6ba9b118ba2`; 11 primary fine IDs | `ed9e...` trong stage reports là hash của tập fingerprint, không phải cùng loại hash với `64a7...` |
| Split chi tiết | `/Volumes/WorkSpace/Project/CystoDS/result/stage_00_prepare_protocol_research_20260803-035933/splits/holdout/summary.json` và `train.csv`, `val.csv`, `test.csv` cùng thư mục | 1.553/339/329 ảnh; 112/24/24 bệnh nhân; phân bố coarse/fine | Fixed development hold-out đã được dùng cho 27 run, không còn là test cuối độc lập để chọn mô hình |
| Full metric run đề xuất | `/Volumes/WorkSpace/Project/CystoDS/result/stage_30_run_proposed_method_research_20260803-001339__runs/proposed_hierarchical_swin_smoothed_seed_20260729_research_20260803-001340/metrics/holdout/test_metrics.json` | Binary, coarse, fine, primary-fine, hierarchy, rare-class gate; per-class và confusion matrices | Nguồn canonical duy nhất cho bảng per-class/confusion của proposed |
| KTC bootstrap proposed | Cùng run, `metrics/holdout/patient_bootstrap_ci.json` | 10 aggregate intervals, patient-level percentile bootstrap, 1.000 iterations | Không có CI cho sensitivity, specificity, accuracy, MCC, per-class metric hoặc difference giữa hai model |
| Train/val/test losses và generalization | Cùng run, `metrics/holdout/{train,val,test}_{metrics,losses}.json` | So sánh train–val–test và loss từng head | Test binary tốt hơn val rất mạnh; cần CV để xác nhận cohort variance |
| Training curve dạng số | Cùng run, `logs/holdout_history.csv` | 24 epoch, learning rate, throughput/epoch, train/val loss và metric | Không có `training_history.png` cho proposed; có thể vẽ lại từ CSV mà không train |
| Training runtime/phần cứng | Cùng run, `metrics/holdout/performance.json`, `models/holdout_model_info.json`, `system/environment.json` | 28.230.679 tham số; 323,13 s/24 epoch; 115,47 train samples/s; peak allocated 17.838,90 MiB | Đây là **training throughput**, không phải latency/throughput inference |
| Prior/calibration fine | Cùng run, `checkpoints/holdout/fine_prior_audit.json` và `fine_calibration_latest.json` | Prior theo bệnh nhân; validation grid chọn `tau=0,5` | Không có ECE, Brier score, NLL, reliability diagram; không gọi đây là calibration lâm sàng |
| Baseline suite cũ | `/Volumes/WorkSpace/Project/CystoDS/result/stage_10_run_baselines_research_20260802-220124/reports/stage_report.md`; từng run có `metrics/holdout/test_metrics.json` | 5 baseline BF16, batch 512, lr 3e-4, encoder multiplier 0,25 | Không trộn với simplified suite |
| Simplified baseline suite | `/Volumes/WorkSpace/Project/CystoDS/result/stage_10_simplified_baselines_research_20260803-112142/reports/stage_report.md`; từng run có `metrics/holdout/test_metrics.json` | 5 baseline FP32, batch 32, lr 1e-4, encoder multiplier 1,0 | Không augmentation, khác optimizer/scheduler/weight decay; local artifact của 3/5 run bị thiếu một phần; không phải one-factor control với proposed |
| Long-tail screen | `/Volumes/WorkSpace/Project/CystoDS/result/stage_20_run_long_tail_screen_research_20260802-230424/reports/stage_report.md`; 7 child `test_metrics.json` | So sánh 7 loss trong cùng Stage 20 | Batch 512, 20 epoch; không so nhân quả trực tiếp với proposed batch 128, 25 epoch |
| Ablation | `/Volumes/WorkSpace/Project/CystoDS/result/stage_40_run_ablations_research_20260803-064951/reports/stage_report.md`; 9 child `test_metrics.json` | Bảng descriptive ablation | 20 epoch so với proposed tối đa 25 epoch; delta không cô lập duy nhất thành phần bị bỏ |
| WLC-only | Hai file `wlc_only_metrics.json` trong run `ablation_train_all_evaluate_wlc...073521` và `ablation_train_wlc_only...072848` thuộc Stage 40 | Có thể báo cáo descriptive trên 265 ảnh WLC test | Không phải proposed checkpoint; không có BLC-only; BLC test rất lệch prevalence |
| ROI-level | Không có file `roi_metrics.json`; mọi config hiện có đặt `evaluate_roi_level=false` | Chỉ có thể nêu “chưa đánh giá” | Mean/vote cần rerun inference; attention MIL có huấn luyện bổ sung và không được coi là inference-only |
| Prediction-level evidence | Workspace còn 5 `test_image_predictions.csv`: binary/coarse/binary+coarse suite cũ và binary/coarse suite simplified. Đồng thời 26 run manifests khai báo tổng cộng 80 prediction CSV (train/val/test; hai run WLC có thêm WLC test) | Có thể làm ngay baseline binary/coarse case analysis; manifest là bằng chứng provenance để truy tìm/kiểm tra file được khôi phục | Không thể phân tích nội dung những file chỉ còn bản ghi manifest; proposed/fine/Stage 20/40 hiện không có prediction materialized |
| Hình ảnh | Workspace còn 32 PNG, toàn bộ trong 8 baseline run Stage 10. 26 run manifests khai báo tổng cộng 123 PNG | Dùng ngay 32 hình baseline; manifest cho biết loại hình từng được sinh | Sáu PNG proposed và toàn bộ PNG Stage 20/30/40 chỉ còn metadata manifest, chưa thể nhúng vào paper cho đến khi khôi phục hoặc tái tạo |

## 3. Split, modality và khả năng đánh giá ROI

### 3.1. Benchmark materialized

| Split | Ảnh | Bệnh nhân | ROI-positive | ROI-negative | WLC | BLC | Có mask (`json=1`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 1.553 | 112 | 858 | 695 | 1.290 | 263 | 538 |
| Validation | 339 | 24 | 187 | 152 | 282 | 57 | 121 |
| Test | 329 | 24 | 174 | 155 | 265 | 64 | 109 |

Test BLC có 62/64 ảnh ROI-positive; test WLC có 112/265 ảnh ROI-positive. BLC test chỉ gồm 46 `Malignant`, 16 `Non-malignant`, 2 `Anatomical landmarks`, không có `Normal mucosa` hoặc `Foreign bodies`. Vì thế không diễn giải metric full-test như một so sánh công bằng giữa modality.

### 3.2. ROI bags có thể hình thành từ metadata (chưa có performance)

Theo đúng key trong source `pid::visit::lesion`, test có 58 nhóm đủ `visit/lesion`; 142/329 hàng thiếu metadata ROI. Nếu áp dụng policy hiện tại:

| Task | Test bag nhãn nhất quán | Nhóm conflict bị loại | Hàng thiếu ROI metadata |
|---|---:|---:|---:|
| Binary | 56 | 2 | 142 |
| Coarse | 55 | 3 | 142 |
| Fine | 54 | 4 | 142 |

Các số trên là audit khả năng tạo bag từ split CSV, **không phải kết quả dự đoán ROI-level**.

## 4. Full metric của run đề xuất

### 4.1. Aggregate test metrics

| Mức | n | Accuracy | Precision/macro-F1 supported | Recall/BA | F1/F1-22 | MCC | AUROC/macro-AUROC | AUPRC/weighted-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Binary | 329 | 0,9757 | precision 0,9560 | sensitivity 1,0000; specificity 0,9484; BA 0,9742 | F1 0,9775 | 0,9522 | 0,9994 | 0,9995 |
| Coarse | 329 | 0,7082 | macro-F1 0,5880 | BA 0,5665 | macro-F1 all 0,5880 | 0,5887 | 0,9150 | weighted-F1 0,6943 |
| Fine | 248 | 0,4032 | macro-F1 supported 0,4647 (16 lớp) | BA 0,4561 | macro-F1 all 22 = 0,3380 | 0,3565 | 0,8613 | weighted-F1 0,4579 |
| Primary fine | 218 | 0,3807 | macro-F1 supported 0,4573 (10/11 lớp) | — | macro-F1 fixed 11 = 0,4157 | — | — | prediction ngoài primary 78/218 = 35,78% |

Hierarchy trên 248 ảnh fine: parent accuracy từ coarse head 0,6371; parent accuracy từ fine head 0,7782; hierarchical accuracy 0,2863; cross-parent error 0,2218; coarse–fine consistency 0,7903; tail recall 0,5313.

Binary confusion matrix: `[[TN=147, FP=8], [FN=0, TP=174]]`.

### 4.2. Bootstrap 95% theo bệnh nhân

| Metric | Point estimate | KTC 95% |
|---|---:|---:|
| Binary AUROC | 0,9994 | 0,9982–1,0000 |
| Binary AUPRC | 0,9995 | 0,9986–1,0000 |
| Binary F1 | 0,9775 | 0,9606–0,9942 |
| Coarse macro-F1 | 0,5880 | 0,5057–0,6798 |
| Coarse balanced accuracy | 0,5665 | 0,4953–0,6797 |
| Fine macro-F1 supported | 0,4647 | 0,3848–0,5965 |
| Fine macro-F1 all 22 | 0,3380 | 0,1957–0,3578 |
| Primary fine macro-F1 fixed | 0,4157 | 0,2235–0,4524 |
| Hierarchical accuracy | 0,2863 | 0,1831–0,3856 |

### 4.3. Train–validation–test gap

| Split | Binary AUROC | Binary F1 | Coarse macro-F1 | Fine macro-F1 supported | Fine macro-F1 all | Hierarchical accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Train | 1,0000 | 0,9977 | 0,9877 | 0,9228 | 0,9228 | 0,8766 |
| Validation | 0,9306 | 0,8629 | 0,5594 | 0,4591 | 0,3548 | 0,2442 |
| Test | 0,9994 | 0,9775 | 0,5880 | 0,4647 | 0,3380 | 0,2863 |

Validation binary confusion là `[[115,37],[17,170]]`, khác mạnh test `[[147,8],[0,174]]`. Không có bằng chứng leakage trong split, nhưng độ chênh này cho thấy fixed hold-out 24 bệnh nhân có variance/cohort-difficulty lớn; không gọi test result là ổn định trước khi có CV/external validation.

## 5. Per-class analysis canonical của proposed

### 5.1. Coarse 5 lớp

| Lớp | True | Pred | Precision | Recall | F1 | AUROC OVR |
|---|---:|---:|---:|---:|---:|---:|
| Malignant | 142 | 149 | 0,7852 | 0,8239 | 0,8041 | 0,9175 |
| Non-malignant | 32 | 30 | 0,2000 | 0,1875 | 0,1935 | 0,8348 |
| Normal mucosa | 81 | 111 | 0,6757 | 0,9259 | 0,7813 | 0,9512 |
| Anatomical landmarks | 31 | 11 | 0,8182 | 0,2903 | 0,4286 | 0,9027 |
| Foreign bodies | 43 | 28 | 0,9286 | 0,6047 | 0,7324 | 0,9689 |

Các lỗi lớn: `Non-malignant → Malignant` 26; `Malignant → Non-malignant` 23; `Anatomical landmarks → Normal mucosa` 20; `Foreign bodies → Normal mucosa` 14.

### 5.2. Fine 22 lớp

| ID | Lớp | True | Pred | Precision | Recall | F1 | AUROC OVR |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | LowGradePapillary | 41 | 46 | 0,2609 | 0,2927 | 0,2759 | 0,7326 |
| 1 | HighGradePapillary | 95 | 30 | 0,8000 | 0,2526 | 0,3840 | 0,7338 |
| 2 | CIS | 6 | 7 | 0,1429 | 0,1667 | 0,1538 | 0,8767 |
| 3 | PreMalignant | 0 | 69 | 0 | 0 | 0 | không đánh giá |
| 4 | BenignNOS | 10 | 11 | 0,0909 | 0,1000 | 0,0952 | 0,7599 |
| 5 | InflammationNOS | 16 | 7 | 0 | 0 | 0 | 0,6569 |
| 6 | CCG | 2 | 6 | 0 | 0 | 0 | 0,8333 |
| 7 | Denuded | 2 | 3 | 0,3333 | 0,5000 | 0,4000 | 0,7846 |
| 8 | UrothelialPapilloma | 2 | 0 | 0 | 0 | 0 | 0,4187 |
| 9 | SquamousMetaplasia | 0 | 0 | 0 | 0 | 0 | không đánh giá |
| 10 | NephrogenicAdenoma | 0 | 1 | 0 | 0 | 0 | không đánh giá |
| 11 | BenignRare | 0 | 0 | 0 | 0 | 0 | không đánh giá |
| 12 | UreteralOrifice | 21 | 14 | 0,9286 | 0,6190 | 0,7429 | 0,9885 |
| 13 | ResectionBed | 3 | 2 | 1,0000 | 0,6667 | 0,8000 | 1,0000 |
| 14 | ResectionScar | 0 | 0 | 0 | 0 | 0 | không đánh giá |
| 15 | Trabeculation | 4 | 3 | 1,0000 | 0,7500 | 0,8571 | 1,0000 |
| 16 | ProstaticUrethra | 2 | 2 | 1,0000 | 1,0000 | 1,0000 | 1,0000 |
| 17 | Diverticulum | 1 | 1 | 1,0000 | 1,0000 | 1,0000 | 1,0000 |
| 18 | AirBubble | 40 | 42 | 0,9048 | 0,9500 | 0,9268 | 0,9959 |
| 19 | ResectionLoop | 1 | 0 | 0 | 0 | 0 | 1,0000 |
| 20 | BiopsyForcep | 0 | 1 | 0 | 0 | 0 | không đánh giá |
| 21 | Stent | 2 | 3 | 0,6667 | 1,0000 | 0,8000 | 1,0000 |

Sáu lớp vắng khỏi test: IDs 3, 9, 10, 11, 14, 20. AUROC=1 với 1–2 mẫu dương không chứng minh năng lực ổn định; ví dụ `ResectionLoop` có AUROC 1 nhưng top-1 recall 0, đây không phải mâu thuẫn vì AUROC đo thứ hạng còn recall dùng argmax.

### 5.3. Error concentration và guardrail

Các luồng nhầm fine lớn nhất:

| True → Pred | Số ảnh |
|---|---:|
| HighGradePapillary → PreMalignant | 36 |
| HighGradePapillary → LowGradePapillary | 23 |
| LowGradePapillary → PreMalignant | 18 |
| InflammationNOS → PreMalignant | 8 |
| LowGradePapillary → BenignNOS | 7 |
| UreteralOrifice → LowGradePapillary | 4 |
| UreteralOrifice → AirBubble | 4 |

`PreMalignant` nhận 69/248 dự đoán (27,82%) dù không có ground-truth test và chỉ có 1 ảnh/1 bệnh nhân train. `rare_class_collapse.status="failed"`; ngưỡng cho phép là 11,16%. Đây là failure/safety risk, không phải bằng chứng “ưu tiên an toàn”.

## 6. Bảng headline metric của toàn bộ 27 run

`F1-sup` là macro-F1 trên lớp có support; `F1-all` dùng mẫu số cố định. Dấu `—` nghĩa task không được train/evaluate.

| Suite | Run | Binary F1 / AUROC | Coarse Acc / F1 / BA | Fine Acc / F1-sup / F1-all | Primary F1-all | H-Acc |
|---|---|---:|---:|---:|---:|---:|
| Stage 10 old | binary_swin_tiny | 0,9632 / 0,9965 | — | — | — | — |
| Stage 10 old | coarse_swin_tiny | — | 0,7052 / 0,5895 / 0,5730 | — | — | — |
| Stage 10 old | fine_swin_tiny | — | — | 0,4879 / 0,2521 / 0,1833 | 0,2655 | — |
| Stage 10 old | multitask binary+coarse | 0,9577 / 0,9922 | 0,6930 / 0,5521 / 0,5366 | — | — | — |
| Stage 10 old | multitask binary+coarse+fine | 0,9526 / 0,9928 | 0,6991 / 0,5508 / 0,5371 | 0,4960 / 0,2296 / 0,1670 | 0,3047 | 0,4032 |
| Stage 10 simplified | binary_swin_tiny | 0,9421 / 0,9926 | — | — | — | — |
| Stage 10 simplified | coarse_swin_tiny | — | 0,7264 / 0,6531 / 0,6543 | — | — | — |
| Stage 10 simplified | fine_swin_tiny | — | — | 0,5282 / 0,3756 / 0,2731 | 0,3423 | — |
| Stage 10 simplified | multitask binary+coarse | 0,9718 / 0,9971 | 0,7477 / 0,6710 / 0,6518 | — | — | — |
| Stage 10 simplified | multitask binary+coarse+fine | 0,9326 / 0,9903 | 0,6170 / 0,4682 / 0,4549 | 0,4597 / 0,4448 / 0,3235 | 0,4369 | 0,3548 |
| Stage 20 | cross_entropy | — | — | 0,4597 / 0,1775 / 0,1291 | 0,2199 | — |
| Stage 20 | weighted_ce | — | — | 0,3226 / 0,3248 / 0,2362 | 0,3607 | — |
| Stage 20 | focal | — | — | 0,4355 / 0,1916 / 0,1393 | 0,2410 | — |
| Stage 20 | balanced_softmax | — | — | 0,4274 / 0,3491 / 0,2539 | 0,4315 | — |
| Stage 20 | balanced_softmax_smoothed | — | — | 0,4516 / 0,1914 / 0,1392 | 0,2369 | — |
| Stage 20 | logit_adjustment | — | — | 0,4274 / 0,3491 / 0,2539 | 0,4315 | — |
| Stage 20 | LDAM | — | — | 0,5282 / 0,2089 / 0,1519 | 0,2942 | — |
| Stage 30 | proposed | 0,9775 / 0,9994 | 0,7082 / 0,5880 / 0,5665 | 0,4032 / 0,4647 / 0,3380 | 0,4157 | 0,2863 |
| Stage 40 | flat_fine_ce | — | — | 0,4960 / 0,4227 / 0,3074 | 0,4037 | — |
| Stage 40 | multitask_no_hierarchy | 0,9609 / 0,9961 | 0,6930 / 0,5975 / 0,5873 | 0,4556 / 0,2376 / 0,1728 | 0,2306 | 0,3669 |
| Stage 40 | hierarchical_ce | 0,9630 / 0,9964 | 0,7112 / 0,6017 / 0,5892 | 0,5202 / 0,3398 / 0,2472 | 0,3539 | 0,4113 |
| Stage 40 | no_binary_auxiliary | — | 0,6930 / 0,5766 / 0,5530 | 0,3871 / 0,4104 / 0,2985 | 0,3782 | 0,2863 |
| Stage 40 | no_consistency | 0,9797 / 0,9968 | 0,7295 / 0,5782 / 0,5745 | 0,4516 / 0,4632 / 0,3369 | 0,4539 | 0,3387 |
| Stage 40 | no_supcon | 0,9742 / 0,9967 | 0,7416 / 0,6106 / 0,5903 | 0,2339 / 0,3887 / 0,2827 | 0,3208 | 0,1411 |
| Stage 40 | class_balanced_sampler | 0,9647 / 0,9938 | 0,6109 / 0,5462 / 0,5727 | 0,3629 / 0,4861 / 0,3535 | 0,4690 | 0,1774 |
| Stage 40 | train_wlc_only | 0,9683 / 0,9960 | 0,7173 / 0,5412 / 0,5370 | 0,4637 / 0,5053 / 0,3675 | 0,5075 | 0,3387 |
| Stage 40 | train_all_evaluate_wlc | 0,9719 / 0,9994 | 0,6991 / 0,5640 / 0,5469 | 0,3952 / 0,4738 / 0,3446 | 0,3927 | 0,2581 |

## 7. WLC-only metrics hiện có

Nguồn exact:

- `/Volumes/WorkSpace/Project/CystoDS/result/stage_40_run_ablations_research_20260803-064951__runs/ablation_train_all_evaluate_wlc_seed_20260729_research_20260803-073521/metrics/holdout/wlc_only_metrics.json`
- `/Volumes/WorkSpace/Project/CystoDS/result/stage_40_run_ablations_research_20260803-064951__runs/ablation_train_wlc_only_seed_20260729_research_20260803-072848/metrics/holdout/wlc_only_metrics.json`

| Training/evaluation | n binary/fine | Binary F1 / AUROC | Coarse F1 | Fine F1-sup / F1-all | Primary F1-all | H-Acc | Rare gate |
|---|---:|---:|---:|---:|---:|---:|---|
| Train all, evaluate WLC | 265/184 | 0,9610 / 0,9994 | 0,5356 | 0,4505 / 0,3276 | 0,4196 | 0,3207 | failed; PreMalignant 54/184 |
| Train WLC-only, evaluate WLC | 265/184 | 0,9735 / 0,9976 | 0,5548 | 0,5264 / 0,3828 | 0,5205 | 0,3804 | failed; PreMalignant 43/184 |

Không có paired test giữa hai run này; không kết luận train WLC-only “tốt hơn” một cách thống kê.

## 8. Artifact coverage và các mục còn thiếu

| Artifact/kết quả | Số lượng hiện có | Diễn giải |
|---|---:|---|
| `test_metrics.json` | 27 | Full aggregate/per-class/confusion tùy task |
| `patient_bootstrap_ci.json` | 27 | Đều status `ok`, 1.000 valid iterations cho metric được cấu hình |
| `logs/holdout_history.csv` | 25 | Thiếu ở simplified fine và simplified binary+coarse |
| `checkpoints/holdout/history.csv` | 24 | Thiếu thêm simplified binary+coarse+fine |
| Image-level test prediction CSV materialized | 5 | Không có proposed/fine/Stage20/Stage40 trong workspace hiện tại |
| Prediction CSV được 26 run manifests khai báo | 80 | 65/80 không còn materialized; manifest chỉ giữ path/bytes/SHA-256, không thay thế dữ liệu dự đoán |
| PNG materialized | 32 | Chỉ nằm trong 8 run Stage 10 |
| PNG được 26 run manifests khai báo | 123 | 91/123 không còn materialized; gồm toàn bộ 6 PNG của proposed |
| WLC-only metrics | 2 | Chỉ Stage 40; không có BLC-only |
| ROI metrics | 0 | Chưa chạy |
| External metrics | 0 | Chưa chạy |
| Cross-validation result | 0 | Có script Stage 90 nhưng chưa có artifact |
| Grad-CAM/saliency/embedding visualization | 0 | Chưa có |
| Inference latency/throughput/memory benchmark | 0 | `performance.json` chỉ đo training |
| Paired significance / p-value | 0 | `paired_baseline_predictions_csv=null`; proposed predictions cũng vắng |
| Backbone khác Swin-Tiny | 0 | 26 config local đều dùng cùng Swin-Tiny; run thứ 27 thiếu config local nhưng stage summary cũng ghi cùng model family |

Năm prediction file test hiện có là:

- `/Volumes/WorkSpace/Project/CystoDS/result/stage_10_run_baselines_research_20260802-220124__runs/binary_swin_tiny_seed_20260729_research_20260802-220125/predictions/holdout/test_image_predictions.csv`
- `/Volumes/WorkSpace/Project/CystoDS/result/stage_10_run_baselines_research_20260802-220124__runs/coarse_swin_tiny_seed_20260729_research_20260802-221028/predictions/holdout/test_image_predictions.csv`
- `/Volumes/WorkSpace/Project/CystoDS/result/stage_10_run_baselines_research_20260802-220124__runs/multitask_binary_coarse_swin_tiny_seed_20260729_research_20260802-222752/predictions/holdout/test_image_predictions.csv`
- `/Volumes/WorkSpace/Project/CystoDS/result/stage_10_simplified_baselines_research_20260803-112142__runs/binary_swin_tiny_seed_20260729_research_20260803-112142/predictions/holdout/test_image_predictions.csv`
- `/Volumes/WorkSpace/Project/CystoDS/result/stage_10_simplified_baselines_research_20260803-112142__runs/coarse_swin_tiny_seed_20260729_research_20260803-112931/predictions/holdout/test_image_predictions.csv`

### 8.1. Provenance của artifact proposed bị thiếu local

Nguồn: `/Volumes/WorkSpace/Project/CystoDS/result/stage_30_run_proposed_method_research_20260803-001339__runs/proposed_hierarchical_swin_smoothed_seed_20260729_research_20260803-001340/artifact_manifest.json`.

| Artifact được manifest khai báo | Bytes | SHA-256 | Trạng thái workspace |
|---|---:|---|---|
| `predictions/holdout/test_image_predictions.csv` | 283.544 | `15298110ac25da1bfb1868c4d4e88a3aa4b4fb06b0daada53ae7e7b5410ed178` | thiếu local |
| `predictions/holdout/train_image_predictions.csv` | 1.367.866 | `aef90295e1282ac989115ec62992ccd4e5ec0a977f10486474ebebc9c1c8c86d` | thiếu local |
| `predictions/holdout/val_image_predictions.csv` | 290.072 | `f85a9e18920247905c4f0b5dc767606894dd20ac9a494e1925aea701403cc55c` | thiếu local |
| `visualizations/holdout/binary_roc_pr_curves.png` | 83.518 | `8955aabf82a63a43e27b2a6ece84feb6a36bdcb4b5b9955552ada909372665b0` | thiếu local |
| `visualizations/holdout/coarse_confusion_matrix.png` | 93.506 | `462adfa7bda172c2c0812acbf02b85a63b5ac00c26019b6fd1fafea10653d77e` | thiếu local |
| `visualizations/holdout/fine_confusion_matrix.png` | 231.278 | `cdd1ab17404e37c1fdf38f98b18b101905ea4f689d289041414dce1cf3ce7c0c` | thiếu local |
| `visualizations/holdout/per_class_recall.png` | 175.755 | `b457da9e65e99d11ba7c63211a3017b2c0bdb229f7414326acb178d3d8dcb300` | thiếu local |
| `visualizations/holdout/split_class_distribution.png` | 81.732 | `5d48d041df8d26195245c94f9c0bd08e5b10c20c0de9f280fedd1916205356c2` | thiếu local |
| `visualizations/holdout/training_history.png` | 228.995 | `d317dcd70e0d665fc89e518456250969e1aaaeba88a8e2dc1eb2b90855bf1168` | thiếu local |

Không được xem metadata manifest như kết quả có thể tái phân tích. Khi khôi phục artifact từ nơi lưu trữ gốc, phải kiểm tra byte size và SHA-256 ở bảng trên trước khi dùng.

## 9. Các mâu thuẫn và tuyên bố phải tránh

1. **Sai label trong full summary:** coarse canonical order là `[Malignant, Non-malignant, Normal mucosa, Anatomical landmarks, Foreign bodies]`; bảng tổng hợp đang ghi order khác. Fine table cũng dịch sai class name theo ID. Chỉ lấy tên trực tiếp từ `test_metrics.json`.
2. **Hai Stage 10 không tương đương:** old suite dùng BF16/batch 512/lr 3e-4/encoder multiplier 0,25 và augmentation giống họ cấu hình nghiên cứu; simplified dùng FP32/batch 32/lr 1e-4/multiplier 1,0, không augmentation, đồng thời khác scheduler/weight decay. Phải ghi rõ suite nào được dùng; không chọn số tốt hơn theo từng task. So sánh proposed với simplified không cô lập tác động của hierarchy/long-tail.
3. **Ablation chưa cô lập tuyệt đối:** proposed tối đa 25 epoch (dừng sau 24), Stage 40 cấu hình 20 epoch. Không dùng từ “chứng minh đóng góp nhân quả” cho delta.
4. **Primary taxonomy:** preregistered set có 11 class IDs `[0,1,2,4,5,13,15,16,18,19,20]`, dựa trên support bệnh nhân **trong train**. Báo cáo rare-class tự tạo benchmark 13 lớp theo support toàn dataset là post-hoc và không phải primary metric canonical.
5. **Không có ý nghĩa thống kê:** CI riêng từng model không phải kiểm định chênh lệch ghép cặp. Không ghi `p<0,01`, “significant” hoặc “vượt trội” cho đến khi có paired analysis.
6. **Không có explainability evidence:** không suy diễn nguyên nhân hình thái của confusion hoặc tuyên bố SupCon tạo cluster/prototype nếu chưa có Grad-CAM/embedding analysis.
7. **PreMalignant là failure:** 69 false predictions và guardrail failed; không mô tả là safety design giúp sensitivity 100%.
8. **Development hold-out:** cùng test đã được xem qua ở 27 run. Có thể gọi “fixed patient-disjoint development hold-out”, không gọi external, blind final test hoặc independent validation.
9. **Fingerprint trong tài liệu cũ:** `Docs/CystoDS_Dataset_Split_Audit_Report.md` còn đường dẫn timestamp cũ và kết luận ghi fingerprint `d5e476...`; canonical hiện tại là protocol SHA `9b63...` và split fingerprint `64a7...`.
10. **Số lớp non-normal trong split audit:** tài liệu cũ nói 21 subclass nhưng taxonomy có 22 fine classes; tổng 1.681 ảnh non-normal bao gồm đủ 22 lớp.
11. **Bảng 4.3 của paper hiện trộn mẫu số:** các số Stage 10/20 như `0,2731`, `0,3235`, `0,1291`, `0,2539` là `fine_macro_f1_all_classes` (mẫu số 22), nhưng cột đang ghi “lớp test có support”; riêng proposed lại dùng supported-F1 `0,4647`. So sánh hợp lệ phải dùng cùng cột: proposed supported `0,4647` so với baseline supported, hoặc proposed all-22 `0,3380` so với baseline all-22.
12. **Diễn giải ablation hiện cũng trộn mẫu số:** `no_supcon=0,2827`, `no_binary=0,2985`, `class_balanced_sampler=0,3535` là all-22 F1, trong khi proposed `0,4647` là supported-F1. Các supported-F1 tương ứng là `0,3887`, `0,4104`, `0,4861`; do đó câu “sampler giảm so với proposed” là sai nếu dùng supported denominator.
13. **Proposed không thống trị mọi metric:** fine supported 0,4647 thấp hơn Stage 40 class-balanced sampler 0,4861 và train-WLC-only 0,5053; fine all-22 0,3380 thấp hơn 0,3535/0,3675; primary fixed 0,4157 thấp hơn simplified multitask 0,4369 và Stage 20 Balanced Softmax/logit adjustment 0,4315. Nên mô tả trade-off đa mức, không gọi là “best/SOTA fine model”.

## 10. Việc có thể làm thêm mà không training

- Vẽ lại learning curves, confusion matrices, per-class chart, support-vs-F1 và error-flow từ JSON/CSV canonical.
- Tải exact checkpoint theo receipt, chạy inference lại để xuất proposed predictions, ROC/PR, calibration, case gallery, Grad-CAM, BLC/WLC subgroup và mean/vote ROI aggregation.
- Sau khi có prediction của proposed và baseline trên đúng 329 ảnh: paired patient bootstrap delta, McNemar cho binary top-1, permutation test cho macro-F1; cần correction nếu kiểm định nhiều model/metric.
- Benchmark inference với warm-up, nhiều batch size, median/p95 latency, throughput, peak memory và thiết bị/precision cố định.

Các mục external validation, 5-fold CV/multi-seed, backbone/SOTA mới và attention-MIL ROI đều cần chạy thêm training/evaluation theo protocol riêng.
