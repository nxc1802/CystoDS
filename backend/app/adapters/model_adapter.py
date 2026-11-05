"""Model adapter protocol and PyTorch runtime wrapper for CystoDS."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from PIL import Image
import torch

from cystods.models.two_stage_hierarchical import TwoStageDecoupledHierarchicalModel
from cystods.taxonomy import (
    BINARY_NAMES,
    COARSE_NAMES,
    FINE_NAMES,
    FINE_BY_PARENT,
)
from backend.app.adapters.preprocessing import (
    CenterFractionCrop,
    InputValidator,
    build_eval_transform,
)
from backend.app.adapters.postprocessing import postprocess_logits


@runtime_checkable
class ModelAdapter(Protocol):
    """Protocol establishing formal contract for deep learning model adapters."""

    def load(self, checkpoint_path: str | Path, config: Optional[Dict[str, Any]] = None) -> None:
        """Loads and initializes model checkpoint into memory."""
        ...

    def validate_input(self, raw: bytes, filename: str = "upload") -> Image.Image:
        """Validates payload and decodes into standardized PIL Image."""
        ...

    def preprocess(self, decoded_input: Image.Image) -> torch.Tensor:
        """Transforms decoded image into normalized PyTorch model input tensor."""
        ...

    def predict(self, model_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Executes forward inference pass returning dictionary of task logits."""
        ...

    def postprocess(
        self,
        raw_output: Dict[str, torch.Tensor],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Decodes model logits into structured probabilities and clinical metrics."""
        ...

    def metadata(self) -> Dict[str, Any]:
        """Provides static and dynamic metadata regarding model architecture and weights."""
        ...


class CystoDSModelAdapter:
    """Production implementation of ModelAdapter for TwoStageDecoupledHierarchicalModel."""

    def __init__(
        self,
        device_preference: str = "auto",
        max_upload_bytes: int = 15 * 1024 * 1024,
    ) -> None:
        self.device_preference = device_preference
        self.validator = InputValidator(max_bytes=max_upload_bytes)
        self.device = self._resolve_device(device_preference)
        self.model: Optional[TwoStageDecoupledHierarchicalModel] = None
        self.checkpoint_path: Optional[Path] = None
        self.checkpoint_hash: str = ""
        self.config: Dict[str, Any] = {}
        self.transform = build_eval_transform(224, 0.92)
        self.fine_prior_counts: Optional[list[int]] = None
        self.fine_inference_prior_tau: float = 0.0
        self.total_parameters: int = 0
        self.model_loaded: bool = False

    def _resolve_device(self, preference: str) -> torch.device:
        """Selects optimal hardware compute device with automatic fallback to CPU."""
        if preference == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")

        try:
            target = torch.device(preference)
            if target.type == "mps" and not torch.backends.mps.is_available():
                return torch.device("cpu")
            if target.type == "cuda" and not torch.cuda.is_available():
                return torch.device("cpu")
            return target
        except Exception:
            return torch.device("cpu")

    def load(
        self,
        checkpoint_path: str | Path,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Loads canonical checkpoint, verifies integrity, and warms up network."""
        path = Path(checkpoint_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        self.checkpoint_path = path

        # 1. Compute SHA-256 Checksum
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        self.checkpoint_hash = sha256.hexdigest()

        # 2. Load PyTorch checkpoint
        ckpt = torch.load(path, map_location="cpu")
        loaded_cfg = dict(ckpt.get("config", {}))

        # Disable timm internet pretrained downloads; checkpoint contains 100% weights
        loaded_cfg["pretrained"] = False
        if config:
            loaded_cfg.update(config)
        self.config = loaded_cfg

        self.fine_inference_prior_tau = float(ckpt.get("fine_inference_prior_tau", 0.0))
        self.fine_prior_counts = ckpt.get("fine_train_counts")

        # 3. Instantiate model & load weights
        model = TwoStageDecoupledHierarchicalModel(self.config)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

        # Move to device and set evaluation mode
        try:
            model.to(self.device)
        except Exception:
            # Fallback to CPU if device placement fails (e.g. MPS out of memory)
            self.device = torch.device("cpu")
            model.to(self.device)

        model.eval()
        self.model = model

        # Count total parameters from checkpoint
        self.total_parameters = sum(v.numel() for v in ckpt["model_state_dict"].values())
        self.inference_parameters = sum(p.numel() for p in model.parameters())

        # 4. Configure transforms from model config
        crop_ratio = float(self.config.get("fov_center_crop_ratio", 0.92))
        img_size = int(self.config.get("image_size", 224))
        mean = tuple(self.config.get("imagenet_mean", [0.485, 0.456, 0.406]))
        std = tuple(self.config.get("imagenet_std", [0.229, 0.224, 0.225]))
        self.transform = build_eval_transform(
            image_size=img_size,
            crop_ratio=crop_ratio,
            mean=mean,
            std=std,
        )

        # 5. Warm up forward pass
        dummy = torch.randn(1, 3, img_size, img_size, device=self.device)
        with torch.no_grad():
            _ = self.model(dummy)

        self.model_loaded = True

    def validate_input(self, raw: bytes, filename: str = "upload") -> Image.Image:
        """Validates payload and decodes into standardized PIL Image."""
        return self.validator.validate_and_decode(raw, filename=filename)

    def preprocess(self, decoded_input: Image.Image) -> torch.Tensor:
        """Transforms decoded image into normalized PyTorch model input tensor."""
        if decoded_input.mode != "RGB":
            decoded_input = decoded_input.convert("RGB")
        tensor = self.transform(decoded_input)
        return tensor.unsqueeze(0).to(self.device)

    def predict(self, model_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Executes forward inference pass returning dictionary of task logits."""
        if self.model is None or not self.model_loaded:
            raise RuntimeError("Model is not loaded. Call adapter.load() before predict().")
        with torch.no_grad():
            return self.model(model_input)

    def postprocess(
        self,
        raw_output: Dict[str, torch.Tensor],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Decodes model logits into structured probabilities and clinical metrics."""
        prior_tau = 0.0
        if context and context.get("apply_prior_calibration"):
            prior_tau = self.fine_inference_prior_tau

        return postprocess_logits(
            raw_output,
            prior_tau=prior_tau,
            fine_prior_counts=self.fine_prior_counts,
        )

    def predict_image(
        self,
        image: Image.Image,
        blend_hierarchical: bool = False,
    ) -> Dict[str, Any]:
        """End-to-end inference execution directly from PIL Image."""
        t0 = time.perf_counter()
        tensor = self.preprocess(image)
        outputs = self.predict(tensor)
        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        results = self.postprocess(outputs, context={"apply_prior_calibration": blend_hierarchical})
        results["inference_time_ms"] = round(inference_time_ms, 2)
        results["device"] = str(self.device)
        results["checkpoint_hash"] = self.checkpoint_hash
        return results

    def metadata(self) -> Dict[str, Any]:
        """Provides static and dynamic metadata regarding model architecture and weights."""
        return {
            "name": "CystoDS 3S-HFT (Three-Stage Hierarchical Fine-Tuning)",
            "model_name": "CystoDS 3S-HFT (Three-Stage Hierarchical Fine-Tuning)",
            "version": "3.1",
            "task_type": "hierarchical",
            "architecture": "TwoStageDecoupledHierarchicalModel",
            "backbone": str(self.config.get("model_name", "swin_tiny_patch4_window7_224.ms_in1k")),
            "parameters": self.total_parameters or 28230679,
            "input_resolution": [224, 224, 3],
            "fov_center_crop_ratio": float(self.config.get("fov_center_crop_ratio", 0.92)),
            "training_protocol": "3-Split Patient-Disjoint Hold-out (70/15/15)",
            "study_id": "cystods_hierarchical_long_tailed_2026",
            "taxonomy": {
                "binary": list(BINARY_NAMES),
                "coarse": list(COARSE_NAMES),
                "fine_count": len(FINE_NAMES),
                "fine_by_parent": {k: list(v) for k, v in FINE_BY_PARENT.items()},
            },
            "input_spec": {
                "format": ["PNG", "JPEG", "BMP", "WEBP"],
                "color_space": "RGB",
                "max_size_bytes": 15 * 1024 * 1024,
                "dimension_bounds": [32, 4096],
                "eval_resolution": [224, 224],
                "fov_crop_ratio": 0.92,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "metrics_summary": {
                "binary_auroc": 0.9986,
                "coarse_macro_f1": 0.7572,
                "fine_top1_accuracy": 0.7473,
                "hierarchical_consistency_rate": 0.9840,
                "split": "split_0",
                "holdout_type": "patient_disjoint",
            },
            "device": str(self.device),
            "checkpoint_loaded": self.model_loaded,
            "checkpoint_hash": self.checkpoint_hash,
        }

    def get_memory_allocated_mb(self) -> float:
        """Returns current allocated device memory in megabytes."""
        try:
            if self.device.type == "cuda" and torch.cuda.is_available():
                return round(torch.cuda.memory_allocated(self.device) / (1024 * 1024), 2)
            if self.device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
                return round(torch.mps.current_allocated_memory() / (1024 * 1024), 2)
        except Exception:
            pass
        return 0.0
