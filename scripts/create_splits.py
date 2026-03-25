"""
Create stratified train/val/test splits from the ISIC 2019 dataset.

Reads the ground truth CSV, drops the UNK class, renames AK -> AKIEC
to match the project convention, and writes stratified split CSVs.
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
RAW_DIR = os.path.join("data", "raw")
SPLITS_DIR = os.path.join("data", "splits")
GROUND_TRUTH = os.path.join(RAW_DIR, "ISIC_2019_Training_GroundTruth.csv")

# Class columns used by the project (8 classes, no UNK)
PROJECT_CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC", "SCC"]

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42


def main():
    df = pd.read_csv(GROUND_TRUTH)
    print(f"Total samples loaded: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Drop UNK class samples (where UNK == 1)
    if "UNK" in df.columns:
        n_unk = int(df["UNK"].sum())
        df = df[df["UNK"] != 1.0].copy()
        df = df.drop(columns=["UNK"])
        print(f"Dropped {n_unk} UNK samples, remaining: {len(df)}")

    # Rename AK -> AKIEC to match project convention
    if "AK" in df.columns:
        df = df.rename(columns={"AK": "AKIEC"})
        print("Renamed column AK -> AKIEC")

    # Derive integer labels for stratification
    label_cols = [c for c in PROJECT_CLASSES if c in df.columns]
    labels = df[label_cols].values.argmax(axis=1)

    # First split: train vs (val + test)
    train_df, temp_df, _, temp_labels = train_test_split(
        df, labels, test_size=(VAL_RATIO + TEST_RATIO),
        stratify=labels, random_state=SEED,
    )

    # Second split: val vs test (50/50 of the remaining)
    val_df, test_df = train_test_split(
        temp_df, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        stratify=temp_labels, random_state=SEED,
    )

    # Save splits
    os.makedirs(SPLITS_DIR, exist_ok=True)
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = os.path.join(SPLITS_DIR, f"{name}.csv")
        split_df.to_csv(path, index=False)
        print(f"Saved {name}: {len(split_df)} samples -> {path}")


if __name__ == "__main__":
    main()
