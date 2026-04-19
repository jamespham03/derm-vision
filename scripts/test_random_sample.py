"""
Evaluate the trained EfficientNet-B3 checkpoint on randomly sampled images.

Usage:
    # Default: 100 random images
    python3 scripts/test_random_sample.py

    # Custom sample size
    N_SAMPLES=1000 python3 scripts/test_random_sample.py

    # Custom sample size + random seed
    N_SAMPLES=10000 SEED=123 python3 scripts/test_random_sample.py
"""

import os
import sys
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.efficientnet import EfficientNetB3Classifier
from src.transforms import get_val_transforms

CHECKPOINT = "outputs/checkpoints/efficientnet-b3_cardassian-spot-5/best_model-2.pth"
RAW_DIR = "data/raw"
GT_CSV = "data/raw/ISIC_2019_Training_GroundTruth.csv"
IMAGE_SIZE = 300
NUM_CLASSES = 8
DROPOUT = 0.3
N_SAMPLES = int(os.environ.get("N_SAMPLES", 100))
SEED = int(os.environ.get("SEED", 42))

CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC", "SCC"]


def main():
    random.seed(SEED)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # Load ground truth (raw CSV uses "AK"; rename to match CLASS_NAMES)
    df = pd.read_csv(GT_CSV)
    if "AK" in df.columns and "AKIEC" not in df.columns:
        df = df.rename(columns={"AK": "AKIEC"})
    df = df[df["UNK"] != 1.0] if "UNK" in df.columns else df
    df["label"] = df[CLASS_NAMES].values.argmax(axis=1)

    # Keep only images that exist on disk
    df["path"] = df["image"].apply(lambda x: os.path.join(RAW_DIR, f"{x}.jpg"))
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
    print(f"Images found on disk: {len(df)}")

    sample = df.sample(n=N_SAMPLES, random_state=SEED).reset_index(drop=True)

    # Load model
    model = EfficientNetB3Classifier(
        num_classes=NUM_CLASSES,
        pretrained=False,
        dropout=DROPOUT,
        freeze_backbone=False,
    ).to(device)
    state = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {CHECKPOINT}\n")

    transform = get_val_transforms(IMAGE_SIZE)

    correct = 0
    per_class_correct = {c: 0 for c in CLASS_NAMES}
    per_class_total = {c: 0 for c in CLASS_NAMES}

    with torch.no_grad():
        for _, row in sample.iterrows():
            image = np.array(Image.open(row["path"]).convert("RGB"))
            tensor = transform(image=image)["image"].unsqueeze(0).to(device)
            logits = model(tensor)
            pred = logits.argmax(dim=1).item()
            true = int(row["label"])

            per_class_total[CLASS_NAMES[true]] += 1
            if pred == true:
                correct += 1
                per_class_correct[CLASS_NAMES[true]] += 1

    accuracy = correct / N_SAMPLES * 100
    print(f"Overall accuracy: {correct}/{N_SAMPLES} = {accuracy:.1f}%\n")

    print("Per-class accuracy:")
    for cls in CLASS_NAMES:
        total = per_class_total[cls]
        if total > 0:
            acc = per_class_correct[cls] / total * 100
            print(f"  {cls:<8} {per_class_correct[cls]}/{total} = {acc:.1f}%")


if __name__ == "__main__":
    main()
