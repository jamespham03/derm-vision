# DermVision

Skin lesion classification for SJSU CMPE 258 (Deep Learning), spring 2026. Built by Lam Nguyen, James Pham, and Vi Thi Tuong Nguyen.

We trained an EfficientNet-B3 on the ISIC 2019 dataset to classify dermoscopy images into 8 lesion categories, and wrapped it in a small web app where you can upload a photo and get a prediction back.

## Running the web app

You'll need Python 3.9+ and the dependencies in `requirements.txt`.

```bash
pip install -r requirements.txt
python app/server.py
```

Open <http://localhost:8000>. There are three pages:

- `/` — the homepage / gallery
- `/analyze` — upload an image and run the model
- `/about` — what the project is and who built it

The app expects the trained checkpoint at `outputs/checkpoints/efficientnet-b3_cardassian-spot-5/best_model-2.pth`. If it's missing, the analyzer falls back to a demo mode that returns fake probabilities, which is useful when we just want to poke at the UI.

## Dataset

We used the ISIC 2019 Challenge training set: 25,331 dermoscopy images across 8 diagnostic classes.

| Class | Disease |
|-------|---------|
| MEL   | Melanoma |
| NV    | Melanocytic nevus |
| BCC   | Basal cell carcinoma |
| AKIEC | Actinic keratosis / Bowen's |
| BKL   | Benign keratosis |
| DF    | Dermatofibroma |
| VASC  | Vascular lesion |
| SCC   | Squamous cell carcinoma |

The dataset is heavily imbalanced — NV alone is just over half of all images and DF/VASC each have under 260 samples. Most of our experimentation was about finding a loss function that doesn't either ignore the rare classes or overcorrect and trash performance on NV.

## Repo layout

```
derm-vision/
├── configs/                # YAML training configs
├── data/
│   ├── raw/                # ISIC images + CSVs (not in repo)
│   └── splits/             # 80/10/10 stratified split CSVs
├── src/
│   ├── dataset.py          # ISICDataset (PyTorch)
│   ├── transforms.py       # Albumentations pipelines
│   ├── train.py            # Training loop + W&B logging
│   ├── evaluate.py         # Metrics and confusion matrix
│   ├── gradcam.py          # Grad-CAM heatmaps
│   └── models/
│       ├── efficientnet.py # Primary backbone
│       ├── custom_cnn.py   # 4-block CNN baseline
│       └── ensemble.py     # Weighted softmax ensemble
├── notebooks/              # EDA + preprocessing demos
├── app/                    # Web app (Gradio + custom UI)
└── outputs/
    ├── checkpoints/        # Trained weights, one folder per W&B run
    └── results/            # Confusion matrices, metric dumps
```

## What's in the model

EfficientNet-B3 with ImageNet pretrained weights, a `Dropout(0.3) → Linear(feature_dim → 8)` head, and focal loss with γ=2. The backbone stays frozen for the first 5 epochs while the head warms up, then we unfreeze and fine-tune the whole network with cosine-annealed learning rate. Augmentation is the usual flip/rotate/jitter set plus coarse dropout, all via Albumentations.

For inference we use D4 test-time augmentation (8 flip/rotate variants, predictions averaged), which buys about 1.5 points of weighted F1 for free.

The full set of training experiments and ablations lives in [docs/report/experiments_report.md](docs/report/experiments_report.md).

## Setup

```bash
git clone <this repo>
cd derm-vision
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Drop the ISIC 2019 images and CSVs into `data/raw/`, then regenerate splits:

```bash
python scripts/create_splits.py
```

Training expects W&B; either `wandb login` or set `WANDB_MODE=disabled`.

## Training and evaluation

```bash
python -m src.train --config configs/config.yaml
```

This is the run that produced our best checkpoint (`cardassian-spot-5`). The CNN baseline uses `configs/config_cnn.yaml`.

Quick eval from Python:

```python
from src.evaluate import compute_metrics, plot_confusion_matrix
metrics = compute_metrics(y_true, y_pred)
plot_confusion_matrix(y_true, y_pred, save_path="outputs/results/cm.png")
```

## Team

Lam handled the data pipeline (dataset class, augmentation, splits). James led model work — architecture, training runs, evaluation. Vi built the web app and Grad-CAM integration.

## A note on Apple Silicon

If you're on macOS with MPS, set `num_workers=0` in the DataLoader. PyTorch multiprocessing on Mac doesn't play nicely with MPS and you'll get spawn errors otherwise. If `albumentations` complains about NumPy, pin NumPy below 2.0.

## License

ISIC 2019 dataset is governed by its own license (see `data/raw/LICENSE.txt`). Project code is for academic and research use.
