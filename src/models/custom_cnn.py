"""
Baseline 4-layer CNN for skin lesion classification.

A simple convolutional neural network to serve as a baseline before
moving to transfer learning approaches. Not expected to match
EfficientNet performance, but useful for sanity checks and ablation.
"""

import torch.nn as nn


class CustomCNN(nn.Module):
    """Baseline 4-layer CNN classifier.

    Architecture: 4 convolutional blocks (Conv2d -> BatchNorm -> ReLU -> MaxPool)
    followed by adaptive average pooling and a fully connected head.

    Args:
        num_classes: Number of output classes (default 8 for ISIC 2019).
        in_channels: Number of input image channels (default 3 for RGB).
        dropout: Dropout probability before the final linear layer.
    """

    def __init__(
        self, num_classes: int = 8, in_channels: int = 3, dropout: float = 0.3
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3 -> 32
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
