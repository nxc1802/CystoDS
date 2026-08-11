# CystoDS Documentation Index

Welcome to the documentation repository for **CystoDS: Hierarchical Long-Tailed Cystoscopy Image Classification**.

---

## 📂 Thư Mục Tài Liệu (Documentation Directory Index)

### 1. Canonical Project Documentation
- 📘 **[Research Methodology](file:///Volumes/WorkSpace/Project/CystoDS/Docs/research.md)** (`Docs/research.md`): Formulating joint Binary, 5-Coarse, 22-Fine classification; loss function mathematics ($\mathcal{L}_{\text{total}}$, Balanced Softmax, KL Consistency, SupCon); strict 70/15/15 patient-disjoint holdout protocol.
- 🛠️ **[Development Guide](file:///Volumes/WorkSpace/Project/CystoDS/Docs/development.md)** (`Docs/development.md`): Package architecture layout (`src/cystods/`), 4-tier configuration system (`config.yaml`), CLI reference (`cystods run/validate/config`), testing instructions.
- 📊 **[Executive Results & Data Summary](file:///Volumes/WorkSpace/Project/CystoDS/Docs/results.md)** (`Docs/results.md`): Data audit summary, patient & image distribution across 22 fine classes, split verification, paper baseline comparisons across 4 backbones.

---

### 2. Detailed Technical Audits (`Docs/audits/`)
- 🔬 **[Dataset & Split Audit](file:///Volumes/WorkSpace/Project/CystoDS/Docs/audits/data_audit.md)** (`Docs/audits/data_audit.md`): Metadata schema analysis, missing value breakdown, 503 filename extension mismatch discovery (.bmp/.jpg/.tiff → .png), 22-subclass feasibility analysis.
- 🛡️ **[System & Artifact Evidence Audit](file:///Volumes/WorkSpace/Project/CystoDS/Docs/audits/system_audit.md)** (`Docs/audits/system_audit.md`): Zero-fallback contract enforcement, directory schemas, Hugging Face receipt verification receipts, staged pipeline test reports.
- 📜 **[Architecture Analysis & Refactoring History](file:///Volumes/WorkSpace/Project/CystoDS/Docs/audits/upgrade_history.md)** (`Docs/audits/upgrade_history.md`): Evolution from Jupytext pre-notebook scripts to `src/cystods/` package layout, architectural reviews, and upgrade roadmap.

---

### 3. Publication & Manuscript Pipeline (`Docs/paper/`)
- 📄 **[Paper Manuscript](file:///Volumes/WorkSpace/Project/CystoDS/Docs/paper/paper.md)** (`Docs/paper/paper.md`): Full research manuscript for publication.
- 🏗️ **[PDF Builder Script](file:///Volumes/WorkSpace/Project/CystoDS/Docs/paper/build.py)** (`Docs/paper/build.py`): Pandoc + Tectonic/XeLaTeX compilation script.
- 🎨 **[Paper Assets & Figures](file:///Volumes/WorkSpace/Project/CystoDS/Docs/paper/paper_assets/)** (`Docs/paper/paper_assets/`): Figures (PNG & PDF) and benchmark data CSVs/JSONs.

---

### 4. Build Output & Reports (`Docs/output/`)
- 📕 **[Compiled Report PDF](file:///Volumes/WorkSpace/Project/CystoDS/Docs/output/pdf/CystoDS_Hierarchical_Long_Tailed_VI.pdf)** (`Docs/output/pdf/CystoDS_Hierarchical_Long_Tailed_VI.pdf`): 4.3 MB publication-ready PDF report.

---

### 5. External References (`Docs/reference/`)
- 📄 **[Original CystoDS Dataset Paper](file:///Volumes/WorkSpace/Project/CystoDS/Docs/reference/original_paper.pdf)** (`Docs/reference/original_paper.pdf`): Scientific Data 2026 reference paper.
- 📝 **[Extracted Dataset Paper Text](file:///Volumes/WorkSpace/Project/CystoDS/Docs/reference/extracted_text.txt)** (`Docs/reference/extracted_text.txt`): Plaintext extraction from original reference paper.
