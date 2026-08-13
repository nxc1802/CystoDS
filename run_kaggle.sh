#!/usr/bin/env bash
# ==============================================================================
# CystoDS — Kaggle Execution Script
#
# How to run on Kaggle Notebook:
# ------------------------------------------------------------------------------
# Option 1 (Full Pipeline via Shell Script):
#   !bash run_kaggle.sh
#
# Option 2 (Direct Python CLI Command):
#   !python -m cystods run all --profile research
#
# Option 3 (Run Stage 20 Full Loss Screening specifically):
#   !python -m cystods run 20 --profile research
# ==============================================================================

set -e

echo "======================================================================"
echo "🚀 Starting CystoDS Full Pipeline Execution on Kaggle"
echo "======================================================================"

# 1. Environment Setup
export PYTHONPATH="src:${PYTHONPATH}"
export CYSTODS_RUN_PROFILE="research"
export TOKENIZERS_PARALLELISM="false"

# 2. Check GPU & Hardware Environment
python3 -c "import torch; print(f'PyTorch CUDA Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# 3. Install Package in Editable Mode
pip install -e . --quiet

# 4. Execute Full Pipeline Stages (00 -> 10 -> 20 -> 30 -> 40 -> 90)
echo ""
echo "----------------------------------------------------------------------"
echo "▶ Running Stage 00 (Protocol Prep)..."
python3 -m cystods run 00 --profile research

echo ""
echo "----------------------------------------------------------------------"
echo "▶ Running Stage 10 (4-Backbone Baselines)..."
python3 -m cystods run 10 --profile research

echo ""
echo "----------------------------------------------------------------------"
echo "▶ Running Stage 20 (Full 7-Loss Long-Tail Screening)..."
python3 -m cystods run 20 --profile research

echo ""
echo "----------------------------------------------------------------------"
echo "▶ Running Stage 30 (Proposed Method)..."
python3 -m cystods run 30 --profile research

echo ""
echo "----------------------------------------------------------------------"
echo "▶ Running Stage 40 (Ablation Studies)..."
python3 -m cystods run 40 --profile research

echo ""
echo "----------------------------------------------------------------------"
echo "▶ Running Stage 90 (Final Cross-Validation Report)..."
python3 -m cystods run 90 --profile research

echo ""
echo "======================================================================"
echo "✅ All Pipeline Stages Executed Successfully!"
echo "======================================================================"
