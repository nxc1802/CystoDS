"""Immutable CystoDS taxonomy: binary, coarse, and fine class hierarchies.

This module is intentionally free of I/O, configuration, or model logic.
Every constant defined here must remain stable across refactoring.
"""

from __future__ import annotations

import torch

# ── Binary ──────────────────────────────────────────────────────────────
BINARY_NAMES = ("Non-ROI", "ROI")

# ── Coarse ──────────────────────────────────────────────────────────────
COARSE_NAMES = (
    "Malignant",
    "Non-malignant",
    "Normal mucosa",
    "Anatomical landmarks",
    "Foreign bodies",
)

# ── Fine ────────────────────────────────────────────────────────────────
FINE_BY_PARENT: dict[str, tuple[str, ...]] = {
    "Malignant": (
        "LowGradePapillary",
        "HighGradePapillary",
        "CIS",
        "PreMalignant",
    ),
    "Non-malignant": (
        "BenignNOS",
        "InflammationNOS",
        "CCG",
        "Denuded",
        "UrothelialPapilloma",
        "SquamousMetaplasia",
        "NephrogenicAdenoma",
        "BenignRare",
    ),
    "Anatomical landmarks": (
        "UreteralOrifice",
        "ResectionBed",
        "ResectionScar",
        "Trabeculation",
        "ProstaticUrethra",
        "Diverticulum",
    ),
    "Foreign bodies": (
        "AirBubble",
        "ResectionLoop",
        "BiopsyForcep",
        "Stent",
    ),
}

# Ordered fine-class names (deterministic traversal of the parent order).
FINE_NAMES = tuple(
    fine_name
    for parent_name in (
        "Malignant",
        "Non-malignant",
        "Anatomical landmarks",
        "Foreign bodies",
    )
    for fine_name in FINE_BY_PARENT[parent_name]
)

if len(FINE_NAMES) != 22 or len(set(FINE_NAMES)) != 22:
    raise RuntimeError(
        "The immutable taxonomy must contain exactly 22 unique fine labels."
    )

# ── ID look-ups ─────────────────────────────────────────────────────────
COARSE_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(COARSE_NAMES)}
COARSE_ID_BY_NAME = COARSE_TO_ID
COARSE_BY_ID: dict[int, str] = {idx: name for idx, name in enumerate(COARSE_NAMES)}
FINE_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(FINE_NAMES)}
FINE_ID_BY_NAME = FINE_TO_ID
FINE_BY_ID: dict[int, str] = {idx: name for idx, name in enumerate(FINE_NAMES)}

FINE_PARENT_ID: tuple[int, ...] = tuple(
    COARSE_TO_ID[parent]
    for fine_name in FINE_NAMES
    for parent, children in FINE_BY_PARENT.items()
    if fine_name in children
)
FINE_TO_COARSE_ID = FINE_PARENT_ID

FINE_PARENT_NAME: dict[str, str] = {
    fine_name: parent
    for parent, children in FINE_BY_PARENT.items()
    for fine_name in children
}

FINE_PARENT_ID_TENSOR = torch.tensor(FINE_PARENT_ID, dtype=torch.long)

ROI_COARSE_IDS = frozenset(
    (COARSE_TO_ID["Malignant"], COARSE_TO_ID["Non-malignant"])
)


def coarse_id_from_subclass(subclass_name: str) -> int:
    parent_name = FINE_PARENT_NAME.get(subclass_name)
    if parent_name is None:
        return COARSE_TO_ID["Normal mucosa"]
    return COARSE_TO_ID[parent_name]


def coarse_name_from_subclass(subclass_name: str) -> str:
    return FINE_PARENT_NAME.get(subclass_name, "Normal mucosa")


def fine_id_from_subclass(subclass_name: str) -> int:
    return FINE_TO_ID.get(subclass_name, -1)

# ── Hierarchy mapping tensors ───────────────────────────────────────────
COARSE_BINARY_PARENT_ID = torch.tensor(
    [
        1,  # Malignant      → ROI
        1,  # Non-malignant  → ROI
        0,  # Normal mucosa  → Non-ROI
        0,  # Anatomical landmarks → Non-ROI
        0,  # Foreign bodies → Non-ROI
    ],
    dtype=torch.long,
)

COARSE_TO_BINARY_MATRIX = torch.zeros(
    (len(COARSE_NAMES), len(BINARY_NAMES)),
    dtype=torch.float32,
)
for _coarse_id, _binary_id in enumerate(COARSE_BINARY_PARENT_ID.tolist()):
    COARSE_TO_BINARY_MATRIX[_coarse_id, _binary_id] = 1.0

FINE_TO_COARSE_MATRIX = torch.zeros(
    (len(FINE_NAMES), len(COARSE_NAMES)),
    dtype=torch.float32,
)
for _fine_id, _coarse_id in enumerate(FINE_PARENT_ID):
    FINE_TO_COARSE_MATRIX[_fine_id, _coarse_id] = 1.0

# ── Metadata CSV contract ──────────────────────────────────────────────
REQUIRED_COLUMNS = frozenset(
    {
        "filename",
        "pid",
        "visit",
        "lesion",
        "multifocal",
        "bca",
        "class",
        "subclass",
        "subclass2",
        "stage",
        "morphology",
        "modality",
        "json",
    }
)

MISSING_TOKENS = frozenset(("", "NA", "N/A", "nan", "None", "null"))

# ── Paper baseline backbones ───────────────────────────────────────────
PAPER_BASELINE_BACKBONES: dict[str, str] = {
    "swin_tiny": "swin_tiny_patch4_window7_224.ms_in1k",
    "resnet152": "resnet152.a1_in1k",
    "hrnet_w18": "hrnet_w18.ms_in1k",
    "resnext50": "resnext50_32x4d.a1_in1k",
}
