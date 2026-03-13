"""
Dataset module for ISIC 2019 skin lesion classification.

Provides a PyTorch Dataset class that loads dermoscopy images and
optional patient metadata from CSV for the 8-class classification task.
"""

import os
from typing import Callable, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC", "SCC"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


class ISICDataset(Dataset):
    """PyTorch Dataset for ISIC 2019 skin lesion images.

    Supports loading images with ground-truth labels and optional patient
    metadata (age, sex, anatomical site) from a CSV file.

    Args:
        image_dir: Path to directory containing JPEG images.
        labels_csv: Path to CSV with columns 'image' and one-hot encoded
            class columns (MEL, NV, BCC, AKIEC, BKL, DF, VASC, SCC).
        metadata_csv: Optional path to CSV with patient metadata columns
            (age_approx, sex, anatom_site_general).
        transform: Optional torchvision/albumentations transform to apply.
    """

    def __init__(
        self,
        image_dir: str,
        labels_csv: str,
        metadata_csv: Optional[str] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        self.image_dir = image_dir
        self.transform = transform

        # Load labels
        self.labels_df = pd.read_csv(labels_csv)
        self.image_ids = self.labels_df["image"].tolist()

        # Convert one-hot labels to class indices
        label_columns = CLASS_NAMES
        self.labels = self.labels_df[label_columns].values.argmax(axis=1)

        # Load optional metadata
        self.metadata = None
        if metadata_csv is not None and os.path.exists(metadata_csv):
            meta_df = pd.read_csv(metadata_csv)
            meta_df = meta_df.set_index("image")
            self.metadata = meta_df

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load and return a single sample.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (image_tensor, label). If metadata is available,
            returns (image_tensor, metadata_dict, label).
        """
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = int(self.labels[idx])

        if self.metadata is not None and image_id in self.metadata.index:
            row = self.metadata.loc[image_id]
            meta = {
                "age": row.get("age_approx", 0),
                "sex": row.get("sex", "unknown"),
                "site": row.get("anatom_site_general", "unknown"),
            }
            return image, meta, label

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for weighted loss.

        Returns:
            Tensor of shape (num_classes,) with per-class weights.
        """
        class_counts = pd.Series(self.labels).value_counts().sort_index()
        total = len(self.labels)
        weights = total / (len(CLASS_NAMES) * class_counts.values)
        return torch.tensor(weights, dtype=torch.float32)
