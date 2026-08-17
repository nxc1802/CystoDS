#!/usr/bin/env python3
"""Launcher script for Decoupled Two-Stage Fine-Tuning Experiment (Stage 35 Replacement).

Usage on Kaggle / Local:
    # 1. Quick smoke test to verify pipeline runs without errors:
    python run_two_stage_hierarchical.py --split 0 --profile smoke

    # 2. Run research experiment on Split 0:
    python run_two_stage_hierarchical.py --split 0 --profile research

    # 3. Run research experiment across all 3 splits:
    python run_two_stage_hierarchical.py --split all --profile research

    # 4. Run with custom Phase 1 / Phase 2 epochs:
    python run_two_stage_hierarchical.py --split 0 --profile research --phase1-epochs 20 --phase2-epochs 8

    # 5. Run with cRT strategy (Class-Balanced Sampling in Phase 2):
    python run_two_stage_hierarchical.py --split 0 --profile research --phase2-strategy crt
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is in python path
src_dir = Path(__file__).resolve().parent / "src"
if src_dir.is_dir() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from cystods.experiments.two_stage_runner import main

if __name__ == "__main__":
    main()
