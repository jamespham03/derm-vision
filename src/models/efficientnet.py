"""
EfficientNet-B3 transfer learning model for skin lesion classification.

Uses the timm library to load a pretrained EfficientNet-B3 backbone and
replaces the classifier head for the 8-class ISIC 2019 task. Supports
freezing the backbone for initial warmup epochs before full fine-tuning.
"""

import torch.nn as nn
import timm


class EfficientNetB3Classifier(nn.Module):
    """EfficientNet-B3 with a custom classification head.

    Loads a pretrained EfficientNet-B3 from timm, optionally freezes the
    backbone layers, and attaches a dropout + linear classifier head.

    Args:
        num_classes: Number of output classes (default 8 for ISIC 2019).
        pretrained: Whether to load ImageNet pretrained weights.
        dropout: Dropout probability before the final linear layer.
        freeze_backbone: If True, freeze all backbone parameters initially.
    """

    def __init__(
        self,
        num_classes: int = 8,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b3", pretrained=pretrained, num_classes=0
        )
        feature_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters to train only the head."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        features = self.backbone(x)
        logits = self.head(features)
        return logits
