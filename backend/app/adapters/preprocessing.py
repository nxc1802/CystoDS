"""Image input validation and transformation pipelines for CystoDS."""

from __future__ import annotations

import io
from typing import Sequence
from PIL import Image, UnidentifiedImageError
from torchvision import transforms


class CenterFractionCrop:
    """Center crop of image preserving center fraction ratio.

    Removes peripheral circular cystoscope border artifacts while keeping
    the true endoscopic field-of-view.
    """

    def __init__(self, ratio: float = 0.92) -> None:
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"CenterFractionCrop ratio must be in (0, 1], got {ratio}")
        self.ratio = float(ratio)

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        crop_width = max(1, round(width * self.ratio))
        crop_height = max(1, round(height * self.ratio))
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        return image.crop((left, top, left + crop_width, top + crop_height))


def build_eval_transform(
    image_size: int = 224,
    crop_ratio: float = 0.92,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """Constructs the canonical evaluation transform pipeline."""
    return transforms.Compose(
        [
            CenterFractionCrop(crop_ratio),
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )


class InputValidator:
    """Rigorous clinical input image validator."""

    def __init__(
        self,
        max_bytes: int = 15 * 1024 * 1024,
        min_dimension: int = 32,
        max_dimension: int = 4096,
    ) -> None:
        self.max_bytes = max_bytes
        self.min_dimension = min_dimension
        self.max_dimension = max_dimension

    def validate_and_decode(self, raw_bytes: bytes, filename: str = "upload") -> Image.Image:
        """Validates payload size, magic byte signature, and decodes to RGB PIL Image."""
        # 1. Size constraints
        if not raw_bytes:
            raise ValueError("Input file payload is empty (0 bytes).")

        if len(raw_bytes) > self.max_bytes:
            raise ValueError(
                f"Payload size {len(raw_bytes)} bytes exceeds maximum allowed {self.max_bytes} bytes (15 MB)."
            )

        # 2. Magic byte signatures
        if not self._check_magic_bytes(raw_bytes):
            raise ValueError(
                f"File '{filename}' does not have a supported image signature. "
                "Accepted formats: PNG, JPEG, BMP, WEBP."
            )

        # 3. Decode image with PIL
        try:
            image = Image.open(io.BytesIO(raw_bytes))
            image.load()  # Force decode to ensure file is not truncated or corrupt
        except (UnidentifiedImageError, OSError, ValueError) as err:
            raise ValueError(f"Failed to decode image file '{filename}': {err}") from err

        # 4. Dimension constraints
        width, height = image.size
        if width < self.min_dimension or height < self.min_dimension:
            raise ValueError(
                f"Image resolution {width}x{height} is too small. "
                f"Minimum resolution is {self.min_dimension}x{self.min_dimension}."
            )

        if width > self.max_dimension or height > self.max_dimension:
            raise ValueError(
                f"Image resolution {width}x{height} exceeds maximum bounds. "
                f"Maximum resolution is {self.max_dimension}x{self.max_dimension}."
            )

        # 5. Color space normalization
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    @staticmethod
    def _check_magic_bytes(header: bytes) -> bool:
        """Returns True if the byte stream starts with a recognized image magic sequence."""
        if len(header) < 12:
            return False

        # PNG: 89 50 4E 47 0D 0A 1A 0A
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return True

        # JPEG: FF D8 FF
        if header.startswith(b"\xff\xd8\xff"):
            return True

        # BMP: 42 4D ('BM')
        if header.startswith(b"BM"):
            return True

        # WEBP: RIFF .... WEBP
        if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
            return True

        return False
