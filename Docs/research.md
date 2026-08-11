# CystoDS: Research Methodology & System Specification

## 1. Research Overview

CystoDS focuses on multi-level cystoscopy image classification under extreme class imbalance. While original cystoscopy benchmarks evaluate binary detection (ROI vs Non-ROI), this research formulates a joint **hierarchical multi-task long-tailed learning problem**.

### 1.1 Multi-Level Label Taxonomy

- **Binary (2 classes)**: `Non-ROI`, `ROI`
- **Coarse (5 classes)**:
  1. `Malignant`
  2. `Non-malignant`
  3. `Normal mucosa`
  4. `Anatomical landmarks`
  5. `Foreign bodies`
- **Fine (22 classes)**: Sub-types assigned under Malignant, Non-malignant, Anatomical landmarks, and Foreign bodies. (`Normal mucosa` has `fine_id = -1` as it represents non-lesion tissue).

### 1.2 Fine Taxonomy Hierarchy

```text
Malignant (4)
 ├── LowGradePapillary
 ├── HighGradePapillary
 ├── CIS
 └── PreMalignant

Non-malignant (8)
 ├── BenignNOS
 ├── InflammationNOS
 ├── CCG
 ├── Denuded
 ├── UrothelialPapilloma
 ├── SquamousMetaplasia
 ├── NephrogenicAdenoma
 └── BenignRare

Anatomical landmarks (6)
 ├── UreteralOrifice
 ├── ResectionBed
 ├── ResectionScar
 ├── Trabeculation
 ├── ProstaticUrethra
 └── Diverticulum

Foreign bodies (4)
 ├── AirBubble
 ├── ResectionLoop
 ├── BiopsyForcep
 └── Stent
```

---

## 2. Mathematical Formulation

### 2.1 Joint Loss Function

The total loss function for the proposed hierarchical model is defined as:

$$ \mathcal{L}_{\text{total}} = \lambda_b \mathcal{L}_{\text{binary}} + \lambda_c \mathcal{L}_{\text{coarse}} + \lambda_f \mathcal{L}_{\text{fine}} + \lambda_{\text{bc}} \mathcal{L}_{\text{hierarchy\_bc}} + \lambda_{\text{cf}} \mathcal{L}_{\text{hierarchy\_cf}} + \lambda_{\text{supcon}} \mathcal{L}_{\text{supcon}} $$

Where default weight parameters are:
- $\lambda_b = 1.0$, $\lambda_c = 1.0$, $\lambda_f = 1.0$
- $\lambda_{\text{bc}} = 0.25$, $\lambda_{\text{cf}} = 0.25$
- $\lambda_{\text{supcon}} = 0.10$ (temperature $\tau = 0.10$)

### 2.2 Fine Long-Tail Objective: Logit-Adjusted / Balanced Softmax

To tackle severe long-tail distribution (where frequent fine classes have thousands of images and rare fine classes have $<20$), the model employs Balanced Softmax:

$$ \mathcal{L}_{\text{fine}} = - \log \frac{n_y \cdot e^{z_y}}{\sum_{j} n_j \cdot e^{z_j}} $$

where $n_j$ is the patient frequency count of fine class $j$, smoothed using power-law smoothing:

$$ \tilde{n}_j = (n_j + \alpha)^\gamma $$

---

## 3. Strict Patient-Disjoint Protocol

To eliminate data leakage, all splits (train 70%, validation 15%, test 15%) are strictly **patient-disjoint**:
- All images from any given patient ID (`pid`) belong exclusively to one split.
- Stage 00 freezes the partition seed and exports an immutable `protocol_manifest.json` with SHA-256 verification.
- Stages 10 through 40 bind directly to Stage 00's hold-out manifest to ensure apples-to-apples baseline comparison.
- Stage 90 evaluates the finalist model using 5-fold cross-validation across 3 random seeds (30 total runs).

---

## 4. Evaluation Metrics

Primary evaluation relies on:
- **Binary**: AUROC, AUPRC, Sensitivity, Specificity, F1
- **Coarse**: Macro F1 (all 5 classes), Per-class F1, AUROC
- **Fine**: Macro F1 (all 22 classes), Primary Macro F1 (evaluable fine classes with $\ge 10$ train patients)
- **Composite Selection Metric**:

$$ M_{\text{composite}} = 0.35 \cdot F1_{\text{coarse}} + 0.45 \cdot F1_{\text{primary\_fine}} + 0.20 \cdot \text{Acc}_{\text{hierarchical}} $$
