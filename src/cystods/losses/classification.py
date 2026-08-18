"""Fine-grained long-tail classification loss.

Extracted from ``cystods.core`` (Step 2 refactor).
Supports: cross_entropy, weighted_ce, focal, balanced_softmax,
balanced_softmax_smoothed, logit_adjustment, ldam.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import cystods.science as science
from cystods.taxonomy import COARSE_NAMES, FINE_NAMES


def effective_number_weights(
    counts: np.ndarray,
    beta: float,
    require_positive: bool = True,
) -> np.ndarray:
    """Compute effective-number class re-weighting (Cui et al., 2019)."""
    counts = np.asarray(counts, dtype=np.float64)
    if require_positive and np.any(counts <= 0):
        raise ValueError("All class counts must be positive.")
    weights = np.zeros_like(counts, dtype=np.float64)
    positive = counts > 0
    effective = 1.0 - np.power(beta, counts[positive])
    weights[positive] = (1.0 - beta) / np.maximum(effective, 1e-12)
    if positive.any():
        weights[positive] /= weights[positive].sum() / positive.sum()
    return weights


class FineLongTailLoss(nn.Module):
    def __init__(
        self,
        loss_name: str,
        class_counts: Sequence[int],
        patient_counts: Sequence[int],
        config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.loss_name = loss_name
        counts = np.asarray(class_counts, dtype=np.float64)
        patient_count_array = np.asarray(patient_counts, dtype=np.float64)
        if counts.shape != (len(FINE_NAMES),):
            raise ValueError("Fine class_counts must have shape (22,).")
        if patient_count_array.shape != (len(FINE_NAMES),):
            raise ValueError("Fine patient_counts must have shape (22,).")
        if (
            not np.isfinite(counts).all()
            or not np.isfinite(patient_count_array).all()
            or np.any(counts < 0)
            or np.any(patient_count_array < 0)
            or not np.equal(counts, np.floor(counts)).all()
            or not np.equal(patient_count_array, np.floor(patient_count_array)).all()
        ):
            raise ValueError(
                "Fine image and patient counts must be finite non-negative integers."
            )
        if np.any(patient_count_array > counts):
            raise ValueError(
                "Fine patient counts cannot exceed fine image counts."
            )
        if loss_name not in {
            "cross_entropy",
            "weighted_ce",
            "focal",
            "balanced_softmax",
            "balanced_softmax_smoothed",
            "logit_adjustment",
            "ldam",
        }:
            raise ValueError(f"Unsupported fine loss: {loss_name}")
        self.gamma = float(config["focal_gamma"])
        self.tau = float(config["logit_adjustment_tau"])
        self.ldam_scale = float(config["ldam_scale"])
        prior_counts = (
            counts
            if config["fine_prior_source"] == "image_count"
            else patient_count_array
        )
        prior = science.build_smoothed_class_prior(
            prior_counts.astype(np.int64),
            smoothing_alpha=float(config["fine_prior_smoothing_alpha"]),
            power=float(config["fine_prior_power"]),
            max_ratio=float(config["fine_prior_max_ratio"]),
        )
        self.register_buffer(
            "class_counts",
            torch.as_tensor(counts, dtype=torch.float32),
        )
        self.register_buffer(
            "patient_counts",
            torch.as_tensor(patient_count_array, dtype=torch.float32),
        )
        self.register_buffer(
            "active_mask",
            torch.as_tensor(prior["active_mask"], dtype=torch.bool),
        )
        self.register_buffer(
            "prior_probabilities",
            torch.as_tensor(prior["probabilities"], dtype=torch.float32),
        )
        self.register_buffer(
            "smoothed_log_prior",
            torch.as_tensor(prior["log_probabilities"], dtype=torch.float32),
        )
        weights = effective_number_weights(
            counts,
            float(config["class_balance_beta"]),
            require_positive=False,
        )
        self.register_buffer(
            "class_weights",
            torch.as_tensor(weights, dtype=torch.float32),
        )
        canonical_prior = np.zeros_like(counts)
        positive_counts = counts > 0
        canonical_prior[positive_counts] = (
            counts[positive_counts] / counts[positive_counts].sum()
        )
        canonical_log_prior = np.zeros_like(counts)
        canonical_log_prior[positive_counts] = np.log(
            canonical_prior[positive_counts]
        )
        self.register_buffer(
            "canonical_log_prior",
            torch.as_tensor(
                canonical_log_prior,
                dtype=torch.float32,
            ),
        )
        margins = np.zeros_like(counts)
        positive = counts > 0
        margins[positive] = float(config["ldam_max_margin"]) / np.power(
            counts[positive], 0.25
        )
        self.register_buffer(
            "ldam_margins",
            torch.as_tensor(margins, dtype=torch.float32),
        )
        self.focal_use_class_balance = bool(
            config["focal_use_class_balance"]
        )

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[1] != len(FINE_NAMES):
            raise ValueError("Fine logits must have shape [N, 22].")
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Fine logits contain NaN or infinity.")
        inactive_value = torch.finfo(logits.dtype).min
        return logits.masked_fill(~self.active_mask.unsqueeze(0), inactive_value)

    def inference_logits(
        self,
        logits: torch.Tensor,
        prior_tau: float,
    ) -> torch.Tensor:
        if not math.isfinite(float(prior_tau)):
            raise ValueError("Fine inference prior tau must be finite.")
        adjusted = (
            logits
            - float(prior_tau)
            * self.smoothed_log_prior.to(dtype=logits.dtype).unsqueeze(0)
        )
        return self.mask_logits(adjusted)

    def prior_audit(self) -> dict[str, Any]:
        return {
            "loss_name": self.loss_name,
            "image_counts": self.class_counts.detach().cpu().int().tolist(),
            "patient_counts": self.patient_counts.detach().cpu().int().tolist(),
            "active_mask": self.active_mask.detach().cpu().tolist(),
            "prior_probabilities": self.prior_probabilities.detach()
            .cpu()
            .tolist(),
            "smoothed_log_prior": self.smoothed_log_prior.detach()
            .cpu()
            .tolist(),
        }

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        if targets.numel() == 0:
            return logits.sum() * 0.0
        if targets.ndim != 1 or len(targets) != len(logits):
            raise ValueError("Fine targets must align with fine logits.")
        if torch.any(targets < 0) or torch.any(targets >= len(FINE_NAMES)):
            raise ValueError("Fine targets are outside the 22-class taxonomy.")
        if torch.any(~self.active_mask[targets]):
            inactive_targets = torch.unique(targets[~self.active_mask[targets]])
            raise ValueError(
                "Fine targets reference classes absent from training: "
                f"{inactive_targets.detach().cpu().tolist()}"
            )
        logits = self.mask_logits(logits)
        if self.loss_name == "cross_entropy":
            return F.cross_entropy(logits, targets)
        if self.loss_name == "weighted_ce":
            return F.cross_entropy(logits, targets, weight=self.class_weights)
        if self.loss_name == "focal":
            ce = F.cross_entropy(logits, targets, reduction="none")
            probability = torch.exp(-ce)
            loss = ((1.0 - probability) ** self.gamma) * ce
            if self.focal_use_class_balance:
                loss = loss * self.class_weights[targets]
            return loss.mean()
        if self.loss_name == "balanced_softmax":
            adjusted = logits + self.canonical_log_prior.unsqueeze(0)
            return F.cross_entropy(adjusted, targets)
        if self.loss_name == "balanced_softmax_smoothed":
            adjusted = logits + self.smoothed_log_prior.unsqueeze(0)
            return F.cross_entropy(adjusted, targets)
        if self.loss_name == "logit_adjustment":
            adjusted = (
                logits + self.tau * self.canonical_log_prior.unsqueeze(0)
            )
            return F.cross_entropy(adjusted, targets)
        if self.loss_name == "ldam":
            adjusted = logits.clone()
            row_ids = torch.arange(
                len(targets), device=targets.device
            )
            adjusted[row_ids, targets] -= self.ldam_margins[targets]
            return F.cross_entropy(adjusted * self.ldam_scale, targets)
        raise RuntimeError(f"Unreachable fine loss: {self.loss_name}")


class CoarseLongTailLoss(nn.Module):
    def __init__(
        self,
        loss_name: str,
        class_counts: Sequence[int],
        patient_counts: Sequence[int],
        config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.loss_name = loss_name
        counts = np.asarray(class_counts, dtype=np.float64)
        patient_count_array = np.asarray(patient_counts, dtype=np.float64)
        if counts.shape != (len(COARSE_NAMES),):
            raise ValueError("Coarse class_counts must have shape (5,).")
        if patient_count_array.shape != (len(COARSE_NAMES),):
            raise ValueError("Coarse patient_counts must have shape (5,).")
        if (
            not np.isfinite(counts).all()
            or not np.isfinite(patient_count_array).all()
            or np.any(counts < 0)
            or np.any(patient_count_array < 0)
        ):
            raise ValueError(
                "Coarse image and patient counts must be finite non-negative integers."
            )
        if loss_name not in {
            "cross_entropy",
            "weighted_ce",
            "focal",
            "balanced_softmax",
            "balanced_softmax_smoothed",
            "logit_adjustment",
            "ldam",
        }:
            raise ValueError(f"Unsupported coarse loss: {loss_name}")
        self.gamma = float(config.get("coarse_focal_gamma", config.get("focal_gamma", 2.0)))
        self.tau = float(config.get("coarse_logit_adjustment_tau", config.get("logit_adjustment_tau", 1.0)))
        self.ldam_scale = float(config.get("coarse_ldam_scale", config.get("ldam_scale", 30.0)))
        prior_source = config.get("coarse_prior_source", config.get("fine_prior_source", "patient_count"))
        prior_counts = (
            counts
            if prior_source == "image_count"
            else patient_count_array
        )
        prior = science.build_smoothed_class_prior(
            prior_counts.astype(np.int64),
            smoothing_alpha=float(config.get("coarse_prior_smoothing_alpha", config.get("fine_prior_smoothing_alpha", 1.0))),
            power=float(config.get("coarse_prior_power", config.get("fine_prior_power", 0.5))),
            max_ratio=float(config.get("coarse_prior_max_ratio", config.get("fine_prior_max_ratio", 30.0))),
        )
        self.register_buffer(
            "class_counts",
            torch.as_tensor(counts, dtype=torch.float32),
        )
        self.register_buffer(
            "patient_counts",
            torch.as_tensor(patient_count_array, dtype=torch.float32),
        )
        self.register_buffer(
            "active_mask",
            torch.as_tensor(prior["active_mask"], dtype=torch.bool),
        )
        self.register_buffer(
            "prior_probabilities",
            torch.as_tensor(prior["probabilities"], dtype=torch.float32),
        )
        self.register_buffer(
            "smoothed_log_prior",
            torch.as_tensor(prior["log_probabilities"], dtype=torch.float32),
        )
        weights = effective_number_weights(
            counts,
            float(config.get("coarse_class_balance_beta", config.get("class_balance_beta", 0.999))),
            require_positive=False,
        )
        self.register_buffer(
            "class_weights",
            torch.as_tensor(weights, dtype=torch.float32),
        )
        canonical_prior = np.zeros_like(counts)
        positive_counts = counts > 0
        canonical_prior[positive_counts] = (
            counts[positive_counts] / counts[positive_counts].sum()
        )
        canonical_log_prior = np.zeros_like(counts)
        canonical_log_prior[positive_counts] = np.log(
            canonical_prior[positive_counts]
        )
        self.register_buffer(
            "canonical_log_prior",
            torch.as_tensor(
                canonical_log_prior,
                dtype=torch.float32,
            ),
        )
        margins = np.zeros_like(counts)
        positive = counts > 0
        margins[positive] = float(config.get("coarse_ldam_max_margin", config.get("ldam_max_margin", 0.5))) / np.power(
            counts[positive], 0.25
        )
        self.register_buffer(
            "ldam_margins",
            torch.as_tensor(margins, dtype=torch.float32),
        )
        self.focal_use_class_balance = bool(
            config.get("coarse_focal_use_class_balance", config.get("focal_use_class_balance", False))
        )

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[1] != len(COARSE_NAMES):
            raise ValueError("Coarse logits must have shape [N, 5].")
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Coarse logits contain NaN or infinity.")
        inactive_value = torch.finfo(logits.dtype).min
        return logits.masked_fill(~self.active_mask.unsqueeze(0), inactive_value)

    def prior_audit(self) -> dict[str, Any]:
        return {
            "loss_name": self.loss_name,
            "image_counts": self.class_counts.detach().cpu().int().tolist(),
            "patient_counts": self.patient_counts.detach().cpu().int().tolist(),
            "active_mask": self.active_mask.detach().cpu().tolist(),
            "prior_probabilities": self.prior_probabilities.detach()
            .cpu()
            .tolist(),
            "smoothed_log_prior": self.smoothed_log_prior.detach()
            .cpu()
            .tolist(),
        }

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        if targets.numel() == 0:
            return logits.sum() * 0.0
        if targets.ndim != 1 or len(targets) != len(logits):
            raise ValueError("Coarse targets must align with coarse logits.")
        if torch.any(targets < 0) or torch.any(targets >= len(COARSE_NAMES)):
            raise ValueError("Coarse targets are outside the 5-class taxonomy.")
        if torch.any(~self.active_mask[targets]):
            inactive_targets = torch.unique(targets[~self.active_mask[targets]])
            raise ValueError(
                "Coarse targets reference classes absent from training: "
                f"{inactive_targets.detach().cpu().tolist()}"
            )
        logits = self.mask_logits(logits)
        if self.loss_name == "cross_entropy":
            return F.cross_entropy(logits, targets)
        if self.loss_name == "weighted_ce":
            return F.cross_entropy(logits, targets, weight=self.class_weights)
        if self.loss_name == "focal":
            ce = F.cross_entropy(logits, targets, reduction="none")
            probability = torch.exp(-ce)
            loss = ((1.0 - probability) ** self.gamma) * ce
            if self.focal_use_class_balance:
                loss = loss * self.class_weights[targets]
            return loss.mean()
        if self.loss_name == "balanced_softmax":
            adjusted = logits + self.canonical_log_prior.unsqueeze(0)
            return F.cross_entropy(adjusted, targets)
        if self.loss_name == "balanced_softmax_smoothed":
            adjusted = logits + self.smoothed_log_prior.unsqueeze(0)
            return F.cross_entropy(adjusted, targets)
        if self.loss_name == "logit_adjustment":
            adjusted = (
                logits + self.tau * self.canonical_log_prior.unsqueeze(0)
            )
            return F.cross_entropy(adjusted, targets)
        if self.loss_name == "ldam":
            adjusted = logits.clone()
            row_ids = torch.arange(
                len(targets), device=targets.device
            )
            adjusted[row_ids, targets] -= self.ldam_margins[targets]
            return F.cross_entropy(adjusted * self.ldam_scale, targets)
        raise RuntimeError(f"Unreachable coarse loss: {self.loss_name}")
