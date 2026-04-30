# DermVision — CLAUDE.md

AI skin lesion classifier for SJSU CMPE 258 (Deep Learning). Classifies dermoscopy images into 8 diagnostic categories using EfficientNet-B3 transfer learning on the ISIC 2019 dataset.

**Team:** Lam Nguyen · James Pham · Vi Thi Tuong Nguyen

---

## Website (app/)

The `app/` folder contains the Gradio-based web interface connecting the trained model to a browser UI. The site has two pages:

| Page | Route | Style reference |
|------|-------|----------------|
| Homepage | `/` | phantom.land — cinematic dark grid showcase |
| Analysis | `/analyze` | resend.com — minimal precision tool UI |

### Design Rules

All website design decisions must follow the rules in `.claude/rules/`:

| File | Topic |
|------|-------|
| [`.claude/rules/design-tokens.md`](.claude/rules/design-tokens.md) | Colors, typography, spacing — the single source of truth for all visual values |
| [`.claude/rules/layout.md`](.claude/rules/layout.md) | Page structure, grid system, breakpoints, section anatomy |
| [`.claude/rules/components.md`](.claude/rules/components.md) | Every reusable component: nav, buttons, cards, upload zone, result cards |
| [`.claude/rules/interactions.md`](.claude/rules/interactions.md) | Hover effects, transitions, animations, loading states |
| [`.claude/rules/homepage.md`](.claude/rules/homepage.md) | Homepage-specific rules: hero, image grid, methodology strip, CTA |
| [`.claude/rules/analysis-page.md`](.claude/rules/analysis-page.md) | Analysis page rules: upload flow, results display, step indicator |

### Design Summary

- **Color scheme:** Near-black dark mode (`#050505` base), warm off-white text (`#f0eeeb`), DermVision green accent (`#0f6e56`)
- **Typography:** DM Serif Display (display headings) + DM Sans (body/UI)
- **Homepage feel:** Dense interactive image grid, cinematic card hover with green glow, editorial layout
- **Analysis page feel:** Single-column centered tool, large confident headline, precise report-style results
- **Never use:** Pure white `#ffffff` backgrounds, light mode surfaces, or colors outside the token set

### Current app/ Files

- `app_1.py` — active Gradio UI (single-page with upload + results)
- `app.py` — version with patient questionnaire
- `update-app.py` — teammate's redesigned version

The production entry point is `app_1.py`. Run with:
```bash
python app/app_1.py
```

---

## Project Structure

```
derm-vision/
├── configs/           # YAML training configs
├── data/
│   ├── raw/           # ISIC 2019 images & CSVs (not in repo — download separately)
│   └── splits/        # Stratified train/val/test CSVs + metadata
├── docs/report/       # Progress reports and experiment write-ups
├── notebooks/         # EDA, preprocessing, and Colab training notebooks
├── outputs/
│   ├── checkpoints/   # Saved model weights per run
│   └── results/       # Confusion matrices and metrics per run
├── scripts/           # Utility scripts (data splitting, sampling)
├── src/               # All core Python modules
│   └── models/        # Model definitions
└── app/               # Gradio web app (separate concern — see app/)
```

---

## Classes

8-class ISIC 2019 classification. Order is fixed everywhere (dataset, model output, UI):

| Index | Code | Disease | Risk |
|-------|------|---------|------|
| 0 | MEL | Melanoma | High |
| 1 | NV | Melanocytic Nevus | Low |
| 2 | BCC | Basal Cell Carcinoma | High |
| 3 | AKIEC | Actinic Keratosis / Bowen's | Moderate |
| 4 | BKL | Benign Keratosis | Low |
| 5 | DF | Dermatofibroma | Low |
| 6 | VASC | Vascular Lesion | Low |
| 7 | SCC | Squamous Cell Carcinoma | High |

Class distribution is severely imbalanced (54× ratio): NV dominates at 50.8%, DF/VASC are under 1%.

---

## Dataset

**Source:** ISIC 2019 Challenge — 25,331 dermoscopy images (JPEG)

**Splits** (stratified 80/10/10, seed=42):
- `data/splits/train.csv` — 20,264 samples
- `data/splits/val.csv` — 2,534 samples
- `data/splits/test.csv` — 2,534 samples

**CSV format:** columns are `image` (ID like `ISIC_0024306`) + one-hot class columns (`MEL`, `NV`, `BCC`, `AKIEC`, `BKL`, `DF`, `VASC`, `SCC`). Images live at `data/raw/ISIC_2019_Training_Input/<id>.jpg`.

**Metadata CSV** (`data/raw/ISIC_2019_Training_Metadata.csv`): optional patient fields — `age_approx`, `sex`, `anatom_site_general`.

To regenerate splits:
```bash
python scripts/create_splits.py
```

---

## Source Modules (`src/`)

### `src/dataset.py`
`ISICDataset(Dataset)` — loads images + one-hot labels, converts to class index, optionally loads metadata. Key method: `get_class_weights()` returns inverse-frequency weight tensor for weighted loss.

### `src/transforms.py`
Albumentations pipelines. Always use these — do not build transforms inline.

- `get_train_transforms(image_size)` — resize + heavy augmentation (flips, rotation, affine, color jitter, coarse dropout) + ImageNet normalize
- `get_val_transforms(image_size)` — resize + normalize only (no augmentation)

### `src/models/efficientnet.py`
`EfficientNetB3Classifier(num_classes, pretrained, dropout, freeze_backbone)`

- Backbone: `timm.create_model("efficientnet_b3", pretrained=True, num_classes=0)`
- Head: `Dropout(0.3) → Linear(feature_dim → 8)`
- Call `model.unfreeze_backbone()` at `unfreeze_epoch` to begin full fine-tuning

### `src/models/custom_cnn.py`
`CustomCNN` — 4-block conv net (~500K params) trained from scratch. Baseline only. Do not use for production; it fails entirely on rare classes (DF F1 = 0.00).

### `src/models/ensemble.py`
`WeightedEnsemble` — weighted average of softmax outputs from multiple models. Weights can be uniform or learned.

### `src/train.py`
Full training pipeline. Entry point: `train(config_path)`.

- `FocalLoss` — `(1 - p_t)^gamma * CE`. gamma=2.0 throughout experiments.
- Device auto-detection: CUDA → MPS → CPU
- Logs to Weights & Biases (`wandb`)
- Checkpoints saved to `outputs/checkpoints/{backbone}_{wandb_run_name}/best_model.pth`
- **macOS note:** set `num_workers=0` in DataLoader to avoid multiprocessing errors on MPS

### `src/evaluate.py`
`compute_metrics(y_true, y_pred)` — balanced accuracy + weighted F1 + per-class precision/recall. `plot_confusion_matrix()` — seaborn heatmap.

### `src/gradcam.py`
`generate_gradcam(model, image_tensor, target_class)` — produces a heatmap overlay using the last conv layer. Used in the web app for interpretability.

---

## Configs

### `configs/config.yaml` — Primary (EfficientNet-B3)
```yaml
data:
  image_size: 300       # EfficientNet-B3 native resolution
  num_classes: 8

training:
  batch_size: 32
  learning_rate: 0.0001
  weight_decay: 0.00001
  epochs: 50
  early_stopping_patience: 7
  scheduler: cosine

model:
  backbone: efficientnet-b3
  pretrained: true
  dropout: 0.3
  freeze_layers: true
  unfreeze_epoch: 5     # Warmup: frozen backbone for first 5 epochs
  loss: focal
  focal_gamma: 2.0
  use_class_weights: false   # Do NOT enable — see experiments below
```

### `configs/config_cnn.yaml` — Custom CNN Baseline
Same structure, `backbone: custom-cnn`, `learning_rate: 0.001` (10× higher for from-scratch training).

---

## Training

```bash
python -m src.train --config configs/config.yaml
```

Requires: `data/raw/` images present, `data/splits/` CSVs generated, W&B login (`wandb login`).

**Layer freeze strategy:**
- Epochs 1–4: backbone frozen, only head trains (warmup)
- Epoch 5+: `unfreeze_backbone()` called, full network fine-tuned

---

## Checkpoints

| Run | W&B Name | Config | Checkpoint Path |
|-----|----------|--------|----------------|
| 1 | splendid-cloud-3 | 224px, weighted CE | `outputs/checkpoints/best_model.pth` (legacy) |
| 2 | usual-salad-4 | 300px, Focal + class weights | `outputs/checkpoints/efficientnet-b3_usual-salad-4/best_model.pth` |
| **3 ★** | **cardassian-spot-5** | **300px, Focal only** | **`outputs/checkpoints/efficientnet-b3_cardassian-spot-5/best_model-2.pth`** |
| 4 | comfy-frost-7 | 300px, Custom CNN | `outputs/checkpoints/custom-cnn_comfy-frost-7/best_model.pth` |

**Use Run 3 for production.** It is the only checkpoint the web app loads.

---

## Experiment Results

### Run 3 (Best) — EfficientNet-B3, Focal Loss, 300px

| Metric | Without TTA | With TTA (D4) |
|--------|:-----------:|:-------------:|
| Balanced Accuracy | 0.7594 | 0.7779 |
| Weighted F1 | 0.8360 | **0.8502** |

Per-class F1 (with TTA): MEL 0.77 · NV 0.91 · BCC 0.90 · AKIEC 0.64 · BKL 0.73 · DF 0.67 · VASC 0.94 · SCC 0.72

### Key Lessons Learned

**Loss function:** Focal loss alone beats Focal + class weights. Both mechanisms correct class imbalance — applying both over-corrects and collapses NV F1 (0.87 → 0.71).

**Resolution:** 300×300 outperforms 224×224. Fine-grained lesion details matter.

**Transfer learning is non-negotiable:** Custom CNN (500K params, from scratch) achieves balanced accuracy 0.38 vs EfficientNet-B3's 0.78. Rare classes get F1 = 0.00 without pretrained features.

**TTA (D4):** 8 inference variants (flips + 90° rotations). Adds ~1.5% F1 for free at inference time.

**Common error pattern:** MEL ↔ NV confusion. Visually similar under dermoscopy; even clinicians struggle.

---

## Inference

```python
from src.models.efficientnet import EfficientNetB3Classifier
from src.transforms import get_val_transforms
import torch
from PIL import Image
import numpy as np

model = EfficientNetB3Classifier(num_classes=8, pretrained=False)
model.load_state_dict(torch.load("outputs/checkpoints/efficientnet-b3_cardassian-spot-5/best_model-2.pth"))
model.eval()

image = Image.open("skin.jpg").convert("RGB")
transform = get_val_transforms(300)
tensor = transform(image=np.array(image))["image"].unsqueeze(0)

with torch.no_grad():
    probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
```

---

## Environment Notes

- **MPS (Apple Silicon):** Supported. Set `num_workers=0` in DataLoader — macOS multiprocessing causes errors with MPS.
- **NumPy:** Use NumPy <2.0 if you hit compatibility errors with older albumentations versions.
- **W&B:** Training requires `wandb login`. Set `WANDB_MODE=disabled` to skip logging.

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `timm`, `albumentations`, `gradio`, `wandb`, `grad-cam`, `scikit-learn`.
