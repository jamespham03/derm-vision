"""
Grad-CAM visualization for skin lesion classification.

Generates class activation maps to visualize which image regions the model
focuses on when making predictions. Useful for interpretability and
clinical validation.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.dataset import CLASS_NAMES
from src.transforms import get_val_transforms


def generate_gradcam(
    model: nn.Module,
    image_path,
    target_layer: nn.Module,
    image_size: int = 224,
    target_class: Optional[int] = None,
    device = "cpu",
) -> np.ndarray:
    """Generate a Grad-CAM heatmap for a single image.

    Args:
        model: Trained classification model.
        image_path: Path to the input image or PIL Image object.
        target_layer: The convolutional layer to compute Grad-CAM for.
        image_size: Resize dimension for preprocessing.
        target_class: Class index to explain. If None, uses the predicted class.
        device: Device to run inference on.

    Returns:
        Numpy array (H, W, 3) with the Grad-CAM overlay.
    """
    model.eval()
    model.to(device)

    # Load and preprocess image
    if isinstance(image_path, str):
        raw_image = Image.open(image_path).convert("RGB")
    else:
        raw_image = image_path.convert("RGB") if hasattr(image_path, 'convert') else image_path

    raw_image = raw_image.resize((image_size, image_size))
    rgb_image = np.array(raw_image) / 255.0

    transform = get_val_transforms(image_size)
    input_tensor = transform(image=np.array(raw_image))["image"]
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # Build Grad-CAM
    cam = GradCAM(model=model, target_layers=[target_layer])

    targets = None
    if target_class is not None:
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        targets = [ClassifierOutputTarget(target_class)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(rgb_image.astype(np.float32), grayscale_cam, use_rgb=True)
    return visualization


def visualize_gradcam(
    model: nn.Module,
    image_path: str,
    target_layer: nn.Module,
    image_size: int = 224,
    target_class: Optional[int] = None,
    save_path: Optional[str] = None,
    device: str = "cpu",
) -> None:
    """Generate and display/save a Grad-CAM visualization.

    Args:
        model: Trained classification model.
        image_path: Path to the input image.
        target_layer: The convolutional layer to compute Grad-CAM for.
        image_size: Resize dimension for preprocessing.
        target_class: Class index to explain.
        save_path: If provided, save the figure to this path.
        device: Device to run inference on.
    """
    overlay = generate_gradcam(
        model, image_path, target_layer, image_size, target_class, device
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    raw_image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    axes[0].imshow(raw_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    class_label = CLASS_NAMES[target_class] if target_class is not None else "predicted"
    axes[1].set_title(f"Grad-CAM ({class_label})")
    axes[1].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Grad-CAM visualization saved to {save_path}")

    plt.close(fig)
