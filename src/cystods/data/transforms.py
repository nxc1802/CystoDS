"""Image transform pipelines for training and evaluation.

Extracted from ``cystods.core`` (Step 3 refactor).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PIL import Image
from torchvision import transforms


class CenterFractionCrop:
    def __init__(self, ratio: float) -> None:
        if not 0 < ratio <= 1:
            raise ValueError("CenterFractionCrop ratio must be in (0, 1].")
        self.ratio = float(ratio)

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        crop_width = max(1, round(width * self.ratio))
        crop_height = max(1, round(height * self.ratio))
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        return image.crop(
            (left, top, left + crop_width, top + crop_height)
        )


def build_transforms(
    config: Mapping[str, Any],
) -> tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    image_size = int(config["image_size"])
    mean = tuple(config["imagenet_mean"])
    std = tuple(config["imagenet_std"])
    center_crop = CenterFractionCrop(config["fov_center_crop_ratio"])
    eval_transform = transforms.Compose(
        [
            center_crop,
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    supcon_transform = transforms.Compose(
        [
            center_crop,
            transforms.RandomResizedCrop(
                image_size,
                scale=tuple(config["random_resized_crop_scale"]),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(
                p=float(config["horizontal_flip_probability"])
            ),
            transforms.RandomVerticalFlip(
                p=float(config["vertical_flip_probability"])
            ),
            transforms.RandomRotation(
                degrees=float(config["rotation_degrees"]),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ColorJitter(*tuple(config["color_jitter"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    use_aug = bool(config.get("use_data_augmentation", False))
    if use_aug:
        train_transform = transforms.Compose(
            [
                center_crop,
                transforms.RandomResizedCrop(
                    image_size,
                    scale=tuple(config["random_resized_crop_scale"]),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.RandomHorizontalFlip(
                    p=float(config["horizontal_flip_probability"])
                ),
                transforms.RandomVerticalFlip(
                    p=float(config["vertical_flip_probability"])
                ),
                transforms.RandomRotation(
                    degrees=float(config["rotation_degrees"]),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ColorJitter(*tuple(config["color_jitter"])),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                transforms.RandomErasing(
                    p=float(config["random_erasing_probability"]),
                    scale=(0.02, 0.15),
                    ratio=(0.3, 3.3),
                    value="random",
                ),
            ]
        )
    else:
        train_transform = eval_transform
    return train_transform, eval_transform, supcon_transform
