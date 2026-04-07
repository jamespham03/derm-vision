"""
Training loop for ISIC 2019 skin lesion classification.

Supports weighted cross-entropy loss for class imbalance, cosine annealing
learning rate scheduling, early stopping, and Weights & Biases logging.
"""

import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import yaml
from torch.utils.data import DataLoader


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    Reduces loss for well-classified examples, focusing training on hard
    negatives. Combines with class weights for doubly-imbalanced datasets.

    Args:
        weight: Per-class weights tensor.
        gamma: Focusing parameter. Higher gamma = more focus on hard examples.
    """

    def __init__(self, weight: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

from src.dataset import ISICDataset
from src.evaluate import compute_metrics
from src.models.custom_cnn import CustomCNN
from src.models.efficientnet import EfficientNetB3Classifier
from src.transforms import get_train_transforms, get_val_transforms


def build_model(cfg: dict, device: torch.device) -> nn.Module:
    """Build model from config."""
    backbone = cfg["model"]["backbone"]
    if backbone == "custom-cnn":
        model = CustomCNN(
            num_classes=cfg["data"]["num_classes"],
            dropout=cfg["model"]["dropout"],
        )
    else:
        model = EfficientNetB3Classifier(
            num_classes=cfg["data"]["num_classes"],
            pretrained=cfg["model"].get("pretrained", True),
            dropout=cfg["model"]["dropout"],
        )
    return model.to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Run a single training epoch.

    Args:
        model: The neural network model.
        loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer instance.
        device: Device to run on (cpu or cuda).

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        images, labels = batch[0].to(device), batch[-1].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Run validation and compute metrics.

    Args:
        model: The neural network model.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run on.

    Returns:
        Dictionary with 'loss' and evaluation metrics.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        images, labels = batch[0].to(device), batch[-1].to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss
    return metrics


def train(config_path: str) -> None:
    """Full training pipeline.

    Loads config, creates datasets and dataloaders, initializes the model
    with weighted cross-entropy loss, trains with early stopping, and logs
    metrics to Weights & Biases.

    Args:
        config_path: Path to the YAML configuration file.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Initialize W&B
    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"].get("entity"),
        config=cfg,
    )

    # Create unique run directory to avoid overwriting previous runs
    run_name = wandb.run.name or datetime.now().strftime("%Y%m%d_%H%M%S")
    backbone = cfg["model"]["backbone"]
    run_tag = f"{backbone}_{run_name}"
    checkpoint_dir = os.path.join(cfg["output"]["checkpoint_dir"], run_tag)
    results_dir = os.path.join(cfg["output"]["results_dir"], run_tag)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Run: {run_tag}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  Results:     {results_dir}")

    # Datasets and loaders
    train_transform = get_train_transforms(cfg["data"]["image_size"])
    val_transform = get_val_transforms(cfg["data"]["image_size"])

    train_dataset = ISICDataset(
        image_dir=cfg["data"]["data_dir"],
        labels_csv=os.path.join(cfg["data"]["splits_dir"], "train.csv"),
        metadata_csv=cfg["data"].get("metadata_csv"),
        transform=train_transform,
    )
    val_dataset = ISICDataset(
        image_dir=cfg["data"]["data_dir"],
        labels_csv=os.path.join(cfg["data"]["splits_dir"], "val.csv"),
        metadata_csv=cfg["data"].get("metadata_csv"),
        transform=val_transform,
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=pin,
    )

    # Model
    model = build_model(cfg, device)
    print(f"Model: {cfg['model']['backbone']}")

    # Loss function
    class_weights = train_dataset.get_class_weights().to(device)
    loss_type = cfg["model"].get("loss", "ce")
    use_class_weights = cfg["model"].get("use_class_weights", True)
    if loss_type == "focal":
        gamma = cfg["model"].get("focal_gamma", 2.0)
        w = class_weights if use_class_weights else None
        criterion = FocalLoss(weight=w, gamma=gamma)
        print(f"Using Focal Loss (gamma={gamma}, class_weights={use_class_weights})")
    else:
        w = class_weights if use_class_weights else None
        criterion = nn.CrossEntropyLoss(weight=w)
        print(f"Using Cross-Entropy Loss (class_weights={use_class_weights})")

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"]
    )

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    patience = cfg["training"]["early_stopping_patience"]

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        # Unfreeze backbone after warmup
        if epoch == cfg["model"].get("unfreeze_epoch", 0):
            for param in model.parameters():
                param.requires_grad = True
            print(f"Epoch {epoch}: Unfreezing all layers")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        # Logging
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "lr": optimizer.param_groups[0]["lr"],
        })

        print(
            f"Epoch {epoch}/{cfg['training']['epochs']} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Bal-Acc: {val_metrics['balanced_accuracy']:.4f}"
        )

        # Early stopping and checkpointing
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved best model to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train skin lesion classifier")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to YAML config file"
    )
    args = parser.parse_args()
    train(args.config)
