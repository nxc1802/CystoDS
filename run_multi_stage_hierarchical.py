#!/usr/bin/env python3
"""Launcher script for Multi-Stage Decoupled Hierarchical Heads Experiment.

Usage on Kaggle / Local:
    # 1. Quick smoke test to verify no bugs:
    python run_multi_stage_hierarchical.py --split 0 --profile smoke

    # 2. Run research experiment on Split 0:
    python run_multi_stage_hierarchical.py --split 0 --profile research

    # 3. Run research experiment across all 3 splits:
    python run_multi_stage_hierarchical.py --split all --profile research

    # 4. Run with Partial Fine-Tuning (Freeze Stages 1-2):
    python run_multi_stage_hierarchical.py --split 0 --profile research --freeze-stages 2

    # 5. Run with Partial Fine-Tuning (Freeze Stages 1-3, ~50% compute reduction):
    python run_multi_stage_hierarchical.py --split 0 --profile research --freeze-stages 3
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is in python path
src_dir = Path(__file__).resolve().parent / "src"
if src_dir.is_dir() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from cystods.experiments.multi_stage_runner import main

if __name__ == "__main__":
    main()
