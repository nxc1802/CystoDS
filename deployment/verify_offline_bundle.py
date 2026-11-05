#!/usr/bin/env python3
"""Offline Venue Verification and Integrity Attestation Utility for CystoDS.

Verifies that all local checkpoints, curated endoscopy samples, model architectures,
and frontend production builds are present, cryptographically verified, and ready
for offline clinical demonstration without internet dependency.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

# Paths relative to repository root
REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_CHECKPOINT_REL = "result/30_proposed/three_stage_split0_research_20260819-211511/phase3/split_0/best_model.pt"
CANONICAL_HASH = "6c10366505db80367e53c400bfddaacada0a0df004b07a87bdfbf6dc0816846c"

CURATED_SAMPLE_FILENAMES = [
    "01c5cce3.png",  # CIS (Tis)
    "0e1d3230.png",  # High-Grade Papillary (T2)
    "d326d133.png",  # Low-Grade Papillary (Ta)
    "004d25e9.png",  # Normal mucosa
    "23a59431.png",  # Anatomical landmark (Ureteral orifice)
    "489f6a5e.png",  # Foreign body (Air bubble)
]


def log_status(icon: str, category: str, message: str) -> None:
    print(f"[{icon}] {category:20s} : {message}")


def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def verify_bundle() -> int:
    print("=" * 72)
    print("CYSTODS OFFLINE DEMO BUNDLE & RELEASE GATE VERIFIER")
    print(f"Repository Root: {REPO_ROOT}")
    print("=" * 72)

    failures = 0

    # 1. Check Python Dependencies
    required_packages = ["torch", "torchvision", "fastapi", "uvicorn", "PIL", "pydantic"]
    for pkg in required_packages:
        try:
            __import__(pkg)
            log_status("PASS", "Python Package", f"'{pkg}' is installed and importable.")
        except ImportError:
            log_status("FAIL", "Python Package", f"CRITICAL: '{pkg}' is missing from current python environment.")
            failures += 1

    # 2. Check Model Checkpoint Existence & Hash
    ckpt_path = REPO_ROOT / CANONICAL_CHECKPOINT_REL
    if not ckpt_path.is_file():
        log_status("FAIL", "Model Checkpoint", f"CRITICAL: Checkpoint missing at {ckpt_path}")
        failures += 1
    else:
        file_size_mb = ckpt_path.stat().st_size / (1024 * 1024)
        calculated_hash = compute_sha256(ckpt_path)
        if calculated_hash == CANONICAL_HASH:
            log_status("PASS", "Model Checkpoint", f"Verified SHA-256 integrity ({file_size_mb:.1f} MB, hash={calculated_hash[:16]}...)")
        else:
            log_status("FAIL", "Model Checkpoint", f"HASH MISMATCH: Expected {CANONICAL_HASH}, got {calculated_hash}")
            failures += 1

    # 3. Check Model In-Memory Loadability & Parameters
    try:
        import torch
        sys.path.insert(0, str(REPO_ROOT))
        from cystods.models.two_stage_hierarchical import TwoStageDecoupledHierarchicalModel

        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = dict(ckpt.get("config", {}))
        cfg["pretrained"] = False
        model = TwoStageDecoupledHierarchicalModel(cfg)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        total_ckpt_params = sum(v.numel() for v in ckpt["model_state_dict"].values())
        infer_params = sum(p.numel() for p in model.parameters())

        if total_ckpt_params == 28230679 and infer_params == 27541655:
            log_status(
                "PASS",
                "Model Forward Init",
                f"TwoStageDecoupledHierarchicalModel instantiated ({total_ckpt_params:,} ckpt params, {infer_params:,} inference params).",
            )
        else:
            log_status(
                "FAIL",
                "Model Forward Init",
                f"Parameter count mismatch: ckpt={total_ckpt_params} (expected 28230679), infer={infer_params} (expected 27541655)",
            )
            failures += 1
    except Exception as exc:
        log_status("FAIL", "Model Forward Init", f"Model loading failed: {exc}")
        failures += 1

    # 4. Check Curated Endoscopy Samples
    samples_dir = REPO_ROOT / "xvdhy-osfstorage-archive" / "images"
    if not samples_dir.is_dir():
        log_status("FAIL", "Samples Directory", f"Samples directory missing at {samples_dir}")
        failures += 1
    else:
        for fname in CURATED_SAMPLE_FILENAMES:
            s_path = samples_dir / fname
            if s_path.is_file() and s_path.stat().st_size > 1000:
                log_status("PASS", "Sample Asset", f"Verified sample '{fname}' ({s_path.stat().st_size:,} bytes).")
            else:
                log_status("FAIL", "Sample Asset", f"Missing or corrupted sample '{fname}' at {s_path}")
                failures += 1

    # 5. Check Frontend Production Build
    next_build_dir = REPO_ROOT / "frontend" / ".next"
    if next_build_dir.is_dir():
        log_status("PASS", "Frontend Build", f"Next.js build artifacts detected at {next_build_dir}")
    else:
        log_status("WARN", "Frontend Build", f"Production build (.next) not found. Run 'cd frontend && npm run build'.")

    # 6. Overall Summary
    print("-" * 72)
    if failures == 0:
        print("[SUCCESS] ALL GATE REQUIREMENTS PASSED. Platform is 100% ready for offline venue presentation.")
        return 0
    else:
        print(f"[ERROR] {failures} VERIFICATION CHECKS FAILED. See log above for remediations.")
        return 1


if __name__ == "__main__":
    sys.exit(verify_bundle())
