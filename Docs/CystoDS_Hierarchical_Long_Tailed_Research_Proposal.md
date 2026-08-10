# Research Proposal

## Hierarchical Long-Tailed Learning for Coarse-to-Fine Bladder Cystoscopy Classification on CystoDS

## 1. Background

The original CystoDS paper primarily benchmarks binary ROI vs Non-ROI
classification although the dataset provides 5 coarse classes, 22
fine-grained subclasses, patient-level metadata, ROI information, visit
information, and segmentation masks. This project extends the benchmark
and proposes a hierarchical long-tailed learning framework.

## 2. Objectives

### Objective 1 -- Surpass Binary Baseline

-   Reproduce the original binary benchmark.
-   Use Swin-Tiny, the strongest representative backbone reported by the
    original paper, for every controlled comparison.
-   Report AUROC, AUPRC, F1, Sensitivity, Specificity, MCC and Balanced
    Accuracy.

### Objective 2 -- Establish 5-Class Benchmark

Classes: - Malignant - Non-malignant - Normal mucosa - Anatomical
landmarks - Foreign bodies

Metrics: - Macro-F1 - Balanced Accuracy - MCC - Macro-AUROC - Per-class
Recall

### Objective 3 -- Hierarchical 22-Subclass Classification

Model jointly predicts: - Binary label - 5-class label - 22-subclass
label using hierarchical consistency constraints.

### Objective 4 -- Long-tail Learning

Investigate: - Class-balanced Sampling - Balanced Softmax - Logit
Adjustment - LDAM - Focal Loss - Supervised Contrastive Learning

## 3. Proposed Framework

Shared pretrained encoder: Swin-Tiny

Heads: - Binary Head - Five-Class Head - Hierarchical Subclass Head

Training loss: - Binary Loss - Five-Class Loss - Hierarchical Loss -
Long-tail Loss - Consistency Loss

## 4. Evaluation Protocol

### Internal Hold-out

-   One fixed patient-level 70/15/15 split shared by all hold-out stages.

### Internal Cross Validation

-   5-fold patient-level cross validation.
-   Mean, Std and 95% confidence interval.
-   Run only in the final stage with an integrated report.

### External Validation

-   Evaluate on external Lazo dataset without fine-tuning.

### WLC-only Evaluation

-   Compare WLC+BLC against WLC-only.

### ROI-level Evaluation

Compare: - Image-level prediction - ROI-level aggregation (Voting /
Average / Attention)

### Experiment Persistence

-   Save logs, metrics, predictions, visualizations and reports for every
    trial/fold.
-   Publish each verified `best_model.pt` to Hugging Face at an immutable
    commit; retain JSON/CSV receipts instead of local model files.

## 5. Benchmark Tasks

### Task 1

Binary ROI vs Non-ROI.

### Task 2

5-Class Classification.

### Task 3

Hierarchical 22-Subclass Classification.

Train using all subclasses while reporting the primary benchmark on
subclasses with sufficient patient support and a separate rare-class
analysis.

## 6. Baselines

Backbone: - Swin-Tiny only

Hierarchy: - Flat classifier - Multi-task classifier - Hierarchical
classifier

Long-tail: - Cross Entropy - Weighted CE - Focal Loss - Balanced
Softmax - Smoothed Balanced Softmax - Logit Adjustment - LDAM

## 7. Ablation

-   Flat vs Hierarchical
-   Single-task vs Multi-task
-   Binary auxiliary head
-   Consistency loss
-   CE vs Long-tail loss
-   Sampling strategies
-   WLC vs WLC+BLC
-   Image-level vs ROI-level

## 8. Metrics

Binary: - AUROC - AUPRC - Accuracy - Precision - Recall - Sensitivity -
Specificity - F1 - MCC - Balanced Accuracy

Five-Class: - Macro-F1 - Weighted-F1 - Balanced Accuracy - MCC -
Macro-AUROC

Hierarchical: - Parent Accuracy - Child Accuracy - Hierarchical
Accuracy - Tail-class Recall - Cross-parent Error Rate

Statistical: - 95% CI - Patient-level Bootstrap - Paired significance
test

## 9. Expected Contributions

1.  Surpass the original binary benchmark.
2.  Establish the first comprehensive 5-class benchmark.
3.  Propose hierarchical coarse-to-fine learning for 22 subclasses.
4.  Improve long-tail recognition.
5.  Comprehensive evaluation:
    -   Internal Hold-out
    -   5-fold Cross Validation
    -   External Validation
    -   WLC-only Analysis
    -   ROI-level Evaluation
6.  Clinical error analysis across classes, subclasses and modalities.
