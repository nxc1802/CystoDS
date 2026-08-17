#!/usr/bin/env python3
"""Launcher script for Three-Stage Hierarchical Fine-Tuning (3S-HFT) Experiment.

Usage on Kaggle / Local:
    # 1. Quick smoke test to verify 3-Phase pipeline:
    python run_three_stage_hierarchical.py --split 0 --profile smoke

    # 2. Run 3-Stage research experiment on Split 0:
    python run_three_stage_hierarchical.py --split 0 --profile research

    # 3. Run 3-Stage research experiment across all 3 splits:
    python run_three_stage_hierarchical.py --split all --profile research

    # 4. Run with custom epochs per phase (Phase 1 / Phase 2 / Phase 3):
    python run_three_stage_hierarchical.py --split 0 --profile research --phase1-epochs 18 --phase2-epochs 5 --phase3-epochs 6
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is in python path
src_dir = Path(__file__).resolve().parent / "src"
if src_dir.is_dir() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from cystods.experiments.three_stage_runner import main

if __name__ == "__main__":
    main()
