# CystoDS Architecture Analysis & Refactoring History

Consolidated document tracking the repository's evolution, design reviews, upgrade roadmap, and successful refactoring from pre-notebook scripts to `src/cystods/` package layout.

---

## 1. Initial Architecture & Analysis (`Overview.md`)

The initial codebase was structured around Jupytext pre-notebook scripts (`stage_00.py` to `stage_90.py`) relying on a monolithic core file (`cystods_core.py`, ~8,400 LOC).

### Key Architectural Strengths Identified
1. **Multi-level Task Formulation**: Moving beyond simple binary ROI detection to joint Binary, 5 Coarse, and 22 Fine subclass classification.
2. **Hierarchy Regularization**: Symmetric KL divergence consistency loss connecting Binary probabilities to Coarse ROI masses, and Fine probabilities to Coarse parent classes.
3. **True Contrastive Learning**: Supervised Contrastive (SupCon) loss operating on dual stochastic augmented views with a dedicated projection head.
4. **Strict Patient-Disjoint Protocol**: Eliminating patient-level data leakage.

### Critical Design Issues Identified
1. **Config Duplication**: ~160 lines of `_make_base_config()` copy-pasted across 7 stage files.
2. **Dependency Bootstrap Duplication**: ~100 lines of `pip install` runtime checks repeated in 8 files.
3. **Stage Duplication**: Co-existence of `stage_10_run_baselines.py` and `stage_10_simplified_baselines.py`.

---

## 2. Refactoring Strategy & Blueprint (`re-factor.md`)

The refactoring strategy defined 6 phased objectives to transition the repository to modern Python software engineering practices without modifying scientific behavior:

- **Phase R1 (Scaffold)**: Establish `pyproject.toml` setuptools package build and `src/cystods/` layout.
- **Phase R2 (Central Config)**: Consolidate configuration into a single `config.yaml` with 4-tier override support.
- **Phase R3 (Core Organization)**: Modularize core library logic and isolate taxonomy constants.
- **Phase R4 (Thin Stages)**: Convert stage scripts to ~50 LOC orchestrators.
- **Phase R5 (Unified CLI)**: Introduce `cystods` CLI entry point with `stages`, `validate`, `config`, `run`.
- **Phase R6 (Docs & Cleanup)**: Consolidate documentation into canonical guides and clear redundant files.

---

## 3. Post-Refactor System Verification

The refactoring was fully executed and verified:
- **Package Installed**: `pip install -e .`
- **CLI Commands Verified**: `cystods stages`, `cystods validate`, `cystods config show`
- **Automated Tests**: 100% pass rate (91 / 91 unit & contract tests)
- **Scientific Integrity**: Bit-for-bit mathematical equivalence retained across all loss functions, data loaders, and model architectures.
