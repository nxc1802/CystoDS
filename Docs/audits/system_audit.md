# CystoDS System & Artifact Evidence Audit Report

Consolidated technical audit evaluating execution logs, artifact integrity, Hugging Face checkpoint verification receipts, and staged pipeline contract compliance.

---

## 1. Zero-Fallback Verification Policy Audit

The CystoDS pipeline enforces strict scientific execution with zero silent fallbacks:

| Verification Dimension | Contract Enforcement Policy | Compliance Status |
|---|---|---|
| **Input Data Integrity** | Fails fast if CSV records or images are missing/corrupted | Verified |
| **Protocol Binding** | Computes SHA-256 digest over `split_manifest.csv`; downstream stages reject mismatched SHA | Verified |
| **Model Pretraining Weights** | Fails fast if `timm` pretrained weights fail to download (no random init fallback) | Verified |
| **Checkpoint Storage** | Remote uploads to Hugging Face Hub require verification of commit OID, file size, and SHA-256 before local deletion | Verified |
| **Hardware Acceleration** | Demands exact requested precision (`bf16`, `fp16`) and device (`cuda`, `mps`); raises error on fallback | Verified |

---

## 2. Artifact Directory & File Schema Contracts

Every completed stage run generates a structured output directory with complete provenance:

```text
result/<experiment_name>_<profile>_<timestamp>/
├── checkpoints/
│   └── hf_checkpoint_receipt.json   # Remote checkpoint verification receipt
├── logs/
│   └── training.log                 # Synchronous line-buffered execution log
├── metrics/
│   ├── validation_epoch_metrics.csv # Per-epoch validation tracking
│   └── test_metrics.csv             # Final test evaluation metrics
├── predictions/
│   └── test_predictions.csv         # Per-sample predictions with probabilities
├── reports/
│   ├── run_summary.json             # Complete experiment configuration & summary
│   └── stage_report.md              # Markdown executive report
├── splits/
│   └── split_manifest.csv           # Immutable patient-disjoint split assignment
├── source/
│   └── source_manifest.json         # SHA-256 snapshot of source code files
└── system/
    └── system_info.json             # Hardware, CUDA, PyTorch, OS provenance
```

---

## 3. Hugging Face Receipt Verification Audit

Research profile stages store model checkpoints remotely on Hugging Face Hub under strict verification contracts:

1. **Upload Execution**: `best_model.pt` is uploaded to `CYSTODS_HF_REPO_ID`.
2. **Metadata Inspection**: `HfApi.get_paths_info()` validates remote file size and commit OID.
3. **Download Roundtrip Check**: Checkpoint is re-downloaded to a temporary location and SHA-256 verified against local original.
4. **Receipt Generation**: Local file is removed ONLY after writing an immutable `hf_checkpoint_receipt.json`.

---

## 4. Staged Pipeline Verification Summary

| Stage | Name | Verification Method | Status |
|---|---|---|---|
| `stage_00` | Prepare Protocol | Patient disjointness check + SHA-256 split manifest audit | PASSED |
| `stage_10` | Run Baselines | 4 backbone architectures × 4 prediction modes | PASSED |
| `stage_20` | Long-Tail Screen | 7 loss functions evaluated on Swin-Tiny fine head | PASSED |
| `stage_30` | Proposed Method | Joint Hierarchical + Balanced Softmax + SupCon | PASSED |
| `stage_40` | Ablations | 16 component ablations with single-factor variation | PASSED |
| `stage_60` | External Validation | Evaluation-only pipeline on external cohort | PASSED |
| `stage_90` | Cross-Validation | 5-fold CV × 3 random seeds final report generation | PASSED |
