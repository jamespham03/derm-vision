"""
Data augmentation and preprocessing pipelines for skin lesion classification.

Defines separate transform pipelines for training (with augmentation) and
validation/test (deterministic resizing and normalization only).
Uses Albumentations for flexible, high-performance augmentations.
"""

from typing import Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet normalization statistics (used with pretrained backbones)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Build the training augmentation pipeline.

    Applies geometric and color augmentations common in dermoscopy tasks:
    random flips, rotations, color jitter, coarse dropout, and normalization.

    Args:
        image_size: Target height and width for resized images.

    Returns:
        An Albumentations Compose pipeline.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent=(-0.1, 0.1), scale=(0.85, 1.15), rotate=(-30, 30), p=0.5
        ),
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
        ], p=0.5),
        A.CoarseDropout(num_holes_range=(4, 8), hole_height_range=(8, 16), hole_width_range=(8, 16), p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Build the validation/test preprocessing pipeline.

    Applies only deterministic resizing and normalization (no augmentation).

    Args:
        image_size: Target height and width for resized images.

    Returns:
        An Albumentations Compose pipeline.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
