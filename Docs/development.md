# CystoDS: Architecture & Development Guide

## 1. Project Structure

```text
CystoDS/
├── config.yaml               # Central configuration file
├── pyproject.toml            # Python package metadata & dependencies
├── README.md                 # Root project overview & quickstart
├── docs/                     # Documentation directory
│   ├── research.md           # Research formulation & loss math
│   ├── development.md        # Architecture & CLI guide
│   └── results.md            # Dataset & audit verification reports
├── paper/                    # Paper source, latex templates, figures
│   ├── paper.md              # Research manuscript
│   ├── build.py              # PDF build script
│   └── paper_assets/         # Figures and benchmark tables
├── src/
│   └── cystods/              # Core library package
│       ├── __init__.py
│       ├── config.py         # Config loader (YAML → profile → env → CLI)
│       ├── core.py           # Core orchestrators & pipeline logic
│       ├── taxonomy.py       # Immutable taxonomy constants & mapping matrices
│       ├── science.py        # Scientific metrics & quality gates
│       ├── hf.py             # Hugging Face checkpoint hub manager
│       ├── cli.py            # Unified argparse CLI
│       └── stages/           # Thin stage orchestrators (~50 LOC each)
│           ├── stage_00.py   # Prepare protocol
│           ├── stage_10.py   # Run baselines
│           ├── stage_20.py   # Long-tail loss screen
│           ├── stage_30.py   # Proposed method
│           ├── stage_40.py   # Ablations
│           ├── stage_60.py   # External validation
│           └── stage_90.py   # Cross-validation
└── tests/                    # Unit & contract test suite
    ├── test_cystods_core_contracts.py
    ├── test_cystods_hf.py
    └── test_cystods_science.py
```

---

## 2. Configuration System Architecture

Configuration follows a clean 4-tier resolution chain implemented in [`cystods.config`](file:///Volumes/WorkSpace/Project/CystoDS/src/cystods/config.py):

```text
1. config.yaml (Base defaults)
       │
       ▼
2. Profile Overrides (--profile research | smoke)
       │
       ▼
3. Environment Variables (CYSTODS_BATCH_SIZE, CYSTODS_DATA_ROOT, etc.)
       │
       ▼
4. CLI Overrides (--set runtime.batch_size=128)
```

---

## 3. CLI Command Reference

### List Stages
```bash
cystods stages
```

### Inspect Configuration
```bash
# View resolved configuration for stage 30
cystods config show --stage 30

# View configuration with custom CLI overrides
cystods config show --stage 30 --set runtime.batch_size=64
```

### Validate Config
```bash
cystods validate --stage 30
```

### Execute Stages
```bash
# Run single stage
cystods run 30

# Run with smoke profile for quick testing
cystods run 30 --profile smoke

# Run Stage 10 for specific model backbone(s)
cystods run 10 --models swin_tiny resnet152

# Run Stage 20 for a single loss trial (by alias or experiment ID)
cystods run 20 --trials focal
cystods run 20 --trials fine_balanced_softmax

# Run Stage 20 for a specific group of loss trials
cystods run 20 --trials focal ldam logit_adjustment
cystods run 20 --trials fine_cross_entropy,fine_weighted_ce

# Run all stages in sequence
cystods run all --profile smoke
```

---

## 4. Testing & Quality Assurance

Run the test suite using `pytest`:

```bash
pytest tests/ -v
```

The test suite covers:
- Pipeline contract verification (`test_cystods_core_contracts.py`)
- Hugging Face receipt verification & upload/download logic (`test_cystods_hf.py`)
- Scientific metric precision & gate enforcement (`test_cystods_science.py`)
