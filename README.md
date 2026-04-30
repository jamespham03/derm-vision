# Derm-Vision: Skin Lesion Classification

SJSU CMPE 258 Deep Learning Project: Skin Disease Classification System Using Deep Learning: A Multi-Class Approach

## How to run web app

**Requirements:** Python 3.9+, dependencies installed (`pip install -r requirements.txt`)

```bash
# From the project root
python app/server.py
```

Then open **http://localhost:8000** in your browser.

The server serves three pages:
- `/` — Homepage (interactive skin lesion gallery)
- `/analyze` — AI analysis tool (upload an image for classification)
- `/about` — Project info, model details, and team

**Model checkpoint** (`outputs/checkpoints/efficientnet-b3_cardassian-spot-5/best_model-2.pth`) must be present for real predictions. Without it, the app runs in demo mode with simulated probabilities.

---

## Overview

Deep learning project for classifying dermoscopy images into 8 skin lesion categories using the ISIC 2019 dataset. The primary model uses EfficientNet-B3 with transfer learning, weighted cross-entropy loss for class imbalance, and Grad-CAM for interpretability.

## Dataset

**ISIC 2019 Challenge** - 25,331 dermoscopy images across 8 diagnostic categories:

| Class | Description |
|-------|-------------|
| MEL | Melanoma |
| NV | Melanocytic nevus |
| BCC | Basal cell carcinoma |
| AKIEC | Actinic keratosis / Bowen's disease |
| BKL | Benign keratosis |
| DF | Dermatofibroma |
| VASC | Vascular lesion |
| SCC | Squamous cell carcinoma |

The dataset exhibits significant class imbalance, addressed via weighted cross-entropy loss and data augmentation.

## Project Structure

```
derm-vision/
├── configs/config.yaml          # Hyperparameters and paths
├── data/
│   ├── raw/                     # Original ISIC images and CSVs
│   ├── processed/               # Preprocessed data
│   └── splits/                  # Train/val/test CSV splits
├── src/
│   ├── dataset.py               # PyTorch Dataset with metadata support
│   ├── transforms.py            # Albumentations augmentation pipelines
│   ├── train.py                 # Training loop with W&B logging
│   ├── evaluate.py              # Metrics and confusion matrix
│   ├── gradcam.py               # Grad-CAM visualization
│   └── models/
│       ├── custom_cnn.py        # Baseline 4-layer CNN
│       ├── efficientnet.py      # EfficientNet-B3 (primary backbone)
│       └── ensemble.py          # Weighted averaging ensemble
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   └── 02_preprocessing.ipynb   # Augmentation pipeline demo
├── app/app.py                   # Gradio web deployment stub
├── outputs/
│   ├── checkpoints/             # Saved model weights
│   └── results/                 # Evaluation outputs
└── requirements.txt
```

## Approach

- **Primary model**: EfficientNet-B3 with transfer learning (ImageNet pretrained)
- **Training strategy**: Frozen backbone warmup followed by full fine-tuning with cosine annealing LR
- **Class imbalance**: Weighted cross-entropy loss (inverse frequency weighting)
- **Augmentation**: Flips, rotations, color jitter, coarse dropout via Albumentations
- **Evaluation**: Balanced accuracy, weighted F1, per-class precision/recall, confusion matrix
- **Interpretability**: Grad-CAM visualizations for model explanations

## Team Roles

| Member | Role |
|--------|------|
| **Lam** | Data pipeline (dataset loading, preprocessing, augmentation, splits) |
| **James** | Model development (architecture selection, training, evaluation) |
| **Vi** | Deployment (web app, Grad-CAM integration, demo preparation) |

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-org>/derm-vision.git
   cd derm-vision
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the ISIC 2019 dataset** and place images + CSVs in `data/raw/`.

5. **Configure Weights & Biases** (optional):
   ```bash
   wandb login
   ```

## Usage

### Training
```bash
python -m src.train --config configs/config.yaml
```

### Evaluation
```python
from src.evaluate import compute_metrics, plot_confusion_matrix
metrics = compute_metrics(y_true, y_pred)
plot_confusion_matrix(y_true, y_pred, save_path="outputs/results/cm.png")
```

### Web App
```bash
cd app
python app.py
```

## License

This project is for academic/research purposes using the ISIC 2019 dataset.
