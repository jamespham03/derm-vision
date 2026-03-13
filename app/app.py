"""
Web application stub for skin lesion classification deployment.

Provides a Gradio interface for uploading dermoscopy images and
getting predictions from the trained model. To be expanded with
additional features like Grad-CAM overlays and confidence scores.
"""

import sys
sys.path.insert(0, "..")

import gradio as gr
import numpy as np
import torch
from PIL import Image

from src.dataset import CLASS_NAMES
from src.models.efficientnet import EfficientNetB3Classifier
from src.transforms import get_val_transforms

# Global model reference (loaded once at startup)
MODEL = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224


def load_model(checkpoint_path: str = "outputs/checkpoints/best_model.pth") -> None:
    """Load the trained model from a checkpoint.

    Args:
        checkpoint_path: Path to the saved model weights.
    """
    global MODEL
    MODEL = EfficientNetB3Classifier(num_classes=8, pretrained=False)
    MODEL.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    MODEL.to(DEVICE)
    MODEL.eval()
    print(f"Model loaded from {checkpoint_path}")


def predict(image: Image.Image) -> dict:
    """Run inference on a single image.

    Args:
        image: PIL Image uploaded by the user.

    Returns:
        Dictionary mapping class names to confidence scores.
    """
    if MODEL is None:
        return {name: 0.0 for name in CLASS_NAMES}

    image_np = np.array(image.convert("RGB"))
    transform = get_val_transforms(IMAGE_SIZE)
    tensor = transform(image=image_np)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    return {name: float(prob) for name, prob in zip(CLASS_NAMES, probs)}


def create_app() -> gr.Blocks:
    """Create the Gradio web application.

    Returns:
        A Gradio Blocks application ready to launch.
    """
    with gr.Blocks(title="Derm-Vision: Skin Lesion Classifier") as app:
        gr.Markdown("# Derm-Vision: Skin Lesion Classification")
        gr.Markdown(
            "Upload a dermoscopy image to classify it into one of 8 skin lesion categories "
            "(MEL, NV, BCC, AKIEC, BKL, DF, VASC, SCC)."
        )

        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload Dermoscopy Image")
            output_label = gr.Label(num_top_classes=8, label="Predictions")

        classify_btn = gr.Button("Classify")
        classify_btn.click(fn=predict, inputs=image_input, outputs=output_label)

    return app


if __name__ == "__main__":
    # Load model if checkpoint exists
    import os
    ckpt = "outputs/checkpoints/best_model.pth"
    if os.path.exists(ckpt):
        load_model(ckpt)
    else:
        print(f"Warning: No checkpoint found at {ckpt}. Predictions will be zeros.")

    app = create_app()
    app.launch(share=False)
