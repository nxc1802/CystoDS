#!/usr/bin/env python3
"""Launcher script for Decoupled Two-Stage Fine-Tuning Experiment (Stage 35 Replacement).

Upgraded from multi-stage decoupled heads to Decoupled Two-Stage Fine-Tuning:
  - Phase 1: Representation Learning (Full Network, Natural Distribution + SupCon)
  - Phase 2: Classifier Alignment (Frozen Backbone, Heads-only with Smoothed Balanced Softmax)
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
