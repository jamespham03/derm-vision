# DermVision - Skin Lesion Classification System

**CMPE 258 - Deep Learning (Spring 2026)**
San Jose State University
Instructor: **Prof. Zara Hajihashemi**

**Team**: Lam Nguyen, James Pham, Vi Thi Tuong Nguyen

---

## Project Overview

This is our deep learning project for classifying skin lesion images. We trained an EfficientNet-B3 model on the ISIC 2019 dataset to classify 8 types of skin conditions. We also built a web app with Flask and Gradio backend and a custom frontend using Three.js, with Grad-CAM to show which part of the image the model is looking at.

### Results

- **Weighted F1**: 0.8502 on ISIC 2019 test split (with D4 TTA)
- **Balanced Accuracy**: 0.7779 (the course target was 75%)
- We classify 8 classes from ISIC 2019. The dataset is very imbalanced (NV class has 54x more images than DF)
- The web app supports image upload, camera capture, paste from clipboard, and shows Grad-CAM heatmap
- We used focal loss (gamma=2), two-stage training, Albumentations for augmentation, and D4 test-time augmentation

---

## Quick Start

### Requirements

- Python 3.9+
- macOS / Linux / Windows
- GPU is optional. Inference works fine on CPU or Apple MPS

### Setup

```bash
# 1. Clone the repo
git clone <this repo>
cd derm-vision

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
python app/server.py
```

Go to **http://localhost:8000**

There are 3 pages:

- `/` -- homepage with Three.js grid gallery
- `/analyze` -- upload image and run the model
- `/about` -- project info, team, methodology

The app looks for the model checkpoint at:

```
outputs/checkpoints/efficientnet-b3_cardassian-spot-5/best_model-2.pth
```

If the checkpoint is not there, it runs in demo mode and returns fake probabilities. Good for testing the UI without the weights.

### How to Test

1. Open http://localhost:8000
2. Click **"LET'S ANALYZE"** in the nav
3. Upload a skin image (or use camera / paste)
4. Click **"Analyze with AI"**
5. See the prediction, probability bars, and Grad-CAM heatmap

---

## Model Performance

### Training Runs (ISIC 2019 test split)

| Run | Loss | Balanced Acc | Weighted F1 | Accuracy | Notes |
|-----|------|-------------:|------------:|----------:|-------|
| `splendid-cloud-3`  | Weighted CE                 | 0.7708 | 0.7856 | 0.78 | Baseline, 224px |
| `usual-salad-4`     | Focal (gamma=2) + class weights | 0.8215 | 0.7102 | 0.70 | Rare classes improved but NV collapsed |
| **`cardassian-spot-5`** | **Focal (gamma=2)**     | **0.7779** | **0.8502** | **0.85** | **Best model, with D4 TTA** |
| `frosty-deluge-6`   | Custom CNN baseline         | 0.3796 | 0.6511 | 0.69 | 4-block CNN, no pretrain |

**Best Model**: EfficientNet-B3 (`cardassian-spot-5`) with focal loss gamma=2, 5-epoch head warmup then full fine-tuning, cosine LR schedule, and D4 TTA.

We got 0.8502 weighted F1. NV class has F1 = 0.91 and the rare classes (DF, SCC, VASC) are better than the CNN baseline.

### Per-Class Results

| Code  | Disease                       | Risk     | Notes |
|-------|-------------------------------|----------|-------|
| MEL   | Melanoma                      | High     | Main clinical target |
| NV    | Melanocytic Nevus             | Low      | F1 = 0.91, largest class |
| BCC   | Basal Cell Carcinoma          | High     | Good recall |
| AKIEC | Actinic Keratosis / Bowen's   | Moderate | Hardest class, F1 around 0.64 |
| BKL   | Benign Keratosis              | Low      | Stable |
| DF    | Dermatofibroma                | Low      | Improved a lot with focal loss |
| VASC  | Vascular Lesion               | Low      | Few samples but decent F1 |
| SCC   | Squamous Cell Carcinoma       | High     | F1 around 0.72 |

Full per-class metrics and confusion matrices are in [docs/report/experiments_report.md](docs/report/experiments_report.md).

---

## Architecture

```
+-------------------------+    HTTP / static    +--------------------------+
|  Custom Web Frontend    |  ----------------> |   Flask + Gradio Server  |
|  HTML / CSS / JS        |  <---------------- |   (Python)               |
|  Three.js grid          |                    |   Port 8000              |
+-------------------------+                    +--------------------------+
        |                                                   |
        v                                                   v
   UI Stack                                         ML Pipeline
   - Three.js r128                         +--------------------------+
   - GSAP animations                       |  Inference               |
   - Custom cursor                         |  - Albumentations preproc|
   - Glassmorphism modal                   |  - D4 Test-Time Aug      |
                                           |  - Grad-CAM heatmap      |
                                           +--------------------------+
                                                        |
                                                        v
                                           +--------------------------+
                                           |  EfficientNet-B3         |
                                           |  + Dropout(0.3)          |
                                           |  + Linear(8 classes)     |
                                           |  Weighted F1 = 0.8502    |
                                           +--------------------------+
```

---

## Methodology

### Dataset

- **Source**: [ISIC 2019 Challenge](https://challenge2019.isic-archive.com/)
- **Size**: 25,331 dermoscopy images
- **Classes**: 8 (MEL, NV, BCC, AKIEC, BKL, DF, VASC, SCC)
- **Split**: 80 / 10 / 10 train / val / test (stratified)
- **Problem**: Very imbalanced. NV is more than half the dataset. DF and VASC have under 260 images each. The ratio between NV and DF is 54x.

To set up the data, put the ISIC 2019 images and CSVs into `data/raw/`, then run:

```bash
python scripts/create_splits.py
```

### Preprocessing and Augmentation (Albumentations)

1. Resize and crop to 224x224 (baseline) or 300x300 (final model)
2. Random flips (horizontal/vertical), rotations, shift-scale-rotate
3. Color jitter, brightness/contrast changes, hue shift
4. Coarse dropout (cutout-style), normalization with ImageNet mean/std
5. D4 test-time augmentation: 8 flip/rotate variants averaged at inference

### Models

**Custom CNN Baseline** (`frosty-deluge-6`):
- 4 conv blocks, trained from scratch
- Weighted F1 = 0.6511

**EfficientNet-B3 with Weighted CE** (`splendid-cloud-3`):
- ImageNet pretrained, head warmup then full fine-tune
- Inverse-frequency class weights
- Weighted F1 = 0.7856, Balanced Accuracy = 0.7708

**EfficientNet-B3 with Focal Loss + Class Weights** (`usual-salad-4`):
- Focal gamma=2.0 plus class weights
- Rare classes improved (DF went from 0.55 to 0.79) but NV recall dropped to 0.56
- Overall weighted F1 went down because NV is huge part of the test set

**EfficientNet-B3 with Focal Loss only** (`cardassian-spot-5`) -- our final model:
- ImageNet pretrained, Dropout(0.3) + Linear(8 classes) head
- Focal loss gamma=2, no extra class weights
- 5-epoch warmup with frozen backbone, then full fine-tune
- Cosine LR schedule with AdamW
- D4 TTA at inference
- **Weighted F1 = 0.8502, Balanced Accuracy = 0.7779**

### Training

```bash
# Train EfficientNet-B3
python -m src.train --config configs/config.yaml

# Train CNN baseline
python -m src.train --config configs/config_cnn.yaml
```

You need Weights and Biases or set `WANDB_MODE=disabled`.

Quick eval from Python:

```python
from src.evaluate import compute_metrics, plot_confusion_matrix

metrics = compute_metrics(y_true, y_pred)
plot_confusion_matrix(y_true, y_pred, save_path="outputs/results/cm.png")
```

---

## Frontend Features

- Three.js fisheye/barrel distortion grid on homepage with mouse parallax
- Glassmorphism analyze modal with backdrop blur
- Drag-and-drop upload, camera capture, paste from clipboard
- Prediction card, per-class probability bars, risk color coding
- Grad-CAM heatmap showing which image regions the model used
- Responsive design for mobile, tablet, and desktop
- Medical disclaimer telling users this is not a real diagnostic tool

### Tech Stack

- **Backend**: Python 3.9+, Flask, Gradio, PyTorch, Albumentations, timm
- **Frontend**: HTML / CSS / JS, Three.js r128, GSAP 3.12
- **ML**: PyTorch, EfficientNet-B3 (timm), Focal loss, D4 TTA
- **Tracking**: Weights and Biases

---

## API

The Gradio analyzer is mounted in Flask. It takes an image and returns class probabilities and a Grad-CAM heatmap.

### POST /analyze

**Request** (multipart):
- `image`: JPEG or PNG image (max around 10MB)

**Response**:
```json
{
  "success": true,
  "data": {
    "prediction": "MEL",
    "confidence": 0.78,
    "probabilities": {
      "MEL": 0.78, "NV": 0.06, "BCC": 0.04, "AKIEC": 0.03,
      "BKL": 0.04, "DF": 0.02, "VASC": 0.01, "SCC": 0.02
    },
    "risk_level": "high",
    "gradcam": "data:image/png;base64,..."
  }
}
```

### GET /
Homepage.

### GET /analyze
Analyze page.

### GET /about
About page.

---

## Documentation

| File | Description |
|------|-------------|
| [README.md](README.md) | This file |
| [docs/report/experiments_report.md](docs/report/experiments_report.md) | Full training runs, ablations, per-class metrics |
| [docs/report/progress_report.md](docs/report/progress_report.md) | Course progress report |
| [docs/proposal/](docs/proposal/) | Project proposal |

---

## Key Findings

### Why focal loss alone worked best

We tried three loss setups:

1. **Weighted cross-entropy**: weighted F1 of 0.7856, balanced accuracy stuck around 0.77
2. **Focal + class weights**: balanced accuracy went up to 0.82 but NV recall dropped to 0.56. Since NV is half the test set this hurt the weighted F1 a lot, dropping it to 0.71
3. **Focal loss only**: the gamma=2 down-weighting handled imbalance by itself. NV recovered to F1=0.91, rare classes stayed above the weighted CE baseline, and weighted F1 hit 0.8502

The lesson we learned is that stacking focal loss with class weights over-corrects on the dominant class. Just focal loss alone worked better for this dataset.

### TTA gain

D4 TTA (8 flip/rotate variants, averaged softmax) gave us +1.9 points balanced accuracy and +1.4 points weighted F1 with no extra training. It is basically free improvement.

### Weak spot

AKIEC is our weakest class at F1 around 0.64. It is clinically similar to BCC and BKL so the model confuses them. This would be the main thing to work on next.

### Course target

- Balanced accuracy target was 75%. We got 77.79% which is +2.79% above target.
- Per-class accuracy above 60% on rare classes: achieved for DF, VASC, SCC. AKIEC still below.

---

## Future Work

Next up: AKIEC-focused fine-tuning with oversampling, a systematic Grad-CAM error audit (we want to confirm the model is actually looking at lesions and not rulers or hair artifacts), and calibration checks with reliability diagrams.

After that: ensemble with ConvNeXt or ViT, training at 380×380 (EfficientNet-B3's native size), and multi-crop TTA stacked on D4.

Longer term, we want to validate on an external dataset (HAM10000 or PH2) and build a lightweight mobile version via EfficientNet-B0 distillation.

---

## Resources

### Datasets
- [ISIC 2019 Challenge](https://challenge2019.isic-archive.com/)
- [ISIC Archive](https://www.isic-archive.com/)

### Papers
- [EfficientNet: Rethinking Model Scaling for CNNs (Tan & Le, 2019)](https://arxiv.org/abs/1905.11946)
- [Focal Loss for Dense Object Detection (Lin et al., 2017)](https://arxiv.org/abs/1708.02002)
- [Grad-CAM (Selvaraju et al., 2017)](https://arxiv.org/abs/1610.02391)
- [Albumentations Documentation](https://albumentations.ai/)
- [timm - PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)

---

## Notes for Apple Silicon

If you are on macOS with MPS, set `num_workers=0` in the DataLoader. PyTorch multiprocessing on Mac does not work well with MPS and you will get spawn errors. If albumentations gives NumPy errors, pin NumPy below 2.0.

---

## Team

- **Lam Nguyen** -- Data pipeline: ISICDataset, Albumentations augmentation, stratified splits
- **James Pham** -- Model development: architecture, training runs, evaluation, ablations
- **Vi Thi Tuong Nguyen** -- Web app: frontend (Three.js homepage, analyze flow, about page) and Grad-CAM integration

---

## License

The ISIC 2019 dataset has its own license (see `data/raw/LICENSE.txt`). Project code is for academic and research use only (see [LICENSE](LICENSE)). Not for clinical or diagnostic use.
