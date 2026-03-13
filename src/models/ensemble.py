"""
Weighted averaging ensemble for skin lesion classification.

Combines predictions from multiple models using learned or fixed weights
to improve overall classification performance and robustness.
"""

from typing import List, Optional

import torch
import torch.nn as nn


class WeightedEnsemble(nn.Module):
    """Weighted averaging ensemble of classification models.

    Combines softmax outputs from multiple models using per-model weights.
    Weights can be fixed (uniform) or learned during a calibration step.

    Args:
        models: List of trained nn.Module classifiers.
        weights: Optional list of per-model weights. If None, uses uniform
            weights (1/N for each model).
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        num_classes: int = 8,
    ) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_classes = num_classes

        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = nn.Parameter(
            torch.tensor(weights, dtype=torch.float32), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: weighted average of softmax predictions.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Averaged probability tensor of shape (B, num_classes).
        """
        all_probs = []
        for model in self.models:
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs)

        # Stack: (num_models, B, num_classes)
        stacked = torch.stack(all_probs, dim=0)
        # Weighted average: weights (num_models,) -> (num_models, 1, 1)
        w = self.weights.view(-1, 1, 1)
        ensemble_probs = (stacked * w).sum(dim=0)

        return ensemble_probs

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Tensor of predicted class indices, shape (B,).
        """
        probs = self.forward(x)
        return probs.argmax(dim=1)
