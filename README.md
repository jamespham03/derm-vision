# DermVision — Skin Lesion Classification System

**CMPE 258 - Deep Learning (Spring 2026)**
San Jose State University
Instructor: **Prof. Zara Hajihashemi**

**Team**: Lam Nguyen (SJSU ID: 018229432), James Pham, Vi Thi Tuong Nguyen

---

## Project Overview

A **deep learning system** for classifying dermoscopy images across 8 diagnostic categories of skin lesions. Built around a fine-tuned **EfficientNet-B3** backbone trained on the **ISIC 2019** dataset, packaged in a full-stack web application with a Python (Flask + Gradio) backend and a custom HTML/CSS/JS frontend featuring a Three.js warp grid, Grad-CAM explainability, and an in-browser analyze flow.

### Key Achievements

- **Weighted F1**: **0.8502** on the ISIC 2019 test split (EfficientNet-B3 + D4 TTA)
- **Balanced Accuracy**: **0.7779** (above the 75% balanced-accuracy course target)
- **8-class classification** across all ISIC 2019 diagnostic categories on a heavily imbalanced dataset (54× imbalance ratio between NV and DF)
- **Full-Stack Demo**: end-to-end web app with image upload, camera capture, paste-from-clipboard, and Grad-CAM heatmap explainability
- **Advanced Techniques**: Focal loss with γ=2, two-stage warmup + fine-tune, Albumentations augmentation pipeline, D4 test-time augmentation

---

## Quick Start

### Prerequisites

- Python 3.9+
- macOS / Linux / Windows
- (Optional) CUDA-capable GPU for training; inference runs fine on CPU or Apple MPS

### Setup

```bash
# 1. Clone and enter the repo
git clone <this repo>
cd derm-vision

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the Web App

```bash
python app/server.py
```

App running at **http://localhost:8000**

Three pages are served:

- `/` — homepage / Three.js warp grid gallery
- `/analyze` — upload an image and run the model
- `/about` — project explainer, team, and methodology

The app expects the trained checkpoint at:

```
outputs/checkpoints/efficientnet-b3_cardassian-spot-5/best_model-2.pth
```

If it's missing, the analyzer falls back to a **demo mode** that returns synthetic probabilities — useful for poking at the UI without the trained weights.

### Test the Application

1. Open http://localhost:8000 in your browser
2. Click **"LET'S ANALYZE"** in the nav (or the hero CTA)
3. Drop in a dermoscopy image (or use camera / paste from clipboard)
4. Click **"Analyze with AI →"**
5. View the diagnosis card, per-class probability bars, and Grad-CAM heatmap

---

## Performance Results

### EfficientNet-B3 Training Runs (ISIC 2019 test split)

| Run | Loss | Balanced Acc | Weighted F1 | Plain Acc | Notes |
|-----|------|-------------:|------------:|----------:|-------|
| `splendid-cloud-3`  | Weighted CE                 | 0.7708 | 0.7856 | 0.78 | Baseline, 224px |
| `usual-salad-4`     | Focal (γ=2) + class weights | 0.8215 | 0.7102 | 0.70 | Rare classes ↑, NV collapsed |
| **`cardassian-spot-5`** | **Focal (γ=2)**         | **0.7779** | **0.8502** | **0.85** | **BEST — final model + D4 TTA** |
| `frosty-deluge-6`   | Custom CNN baseline         | 0.3796 | 0.6511 | 0.69 | 4-block CNN, no pretrain |

**Best Model**: EfficientNet-B3 (`cardassian-spot-5`) — Focal loss γ=2, 5-epoch head warmup → full fine-tune with cosine LR, D4 test-time augmentation.

**Achievement**: **0.8502 weighted F1** with strong recovery on the dominant NV class (F1 = 0.91) while still lifting rare-class F1 (DF, SCC, VASC) above the CNN baseline.

### Per-Class Coverage (Best Model)

| Code  | Disease                       | Risk     | Notes |
|-------|-------------------------------|----------|-------|
| MEL   | Melanoma                      | High     | Primary clinical target |
| NV    | Melanocytic Nevus             | Low      | F1 = 0.91 (largest class) |
| BCC   | Basal Cell Carcinoma          | High     | Strong recall |
| AKIEC | Actinic Keratosis / Bowen's   | Moderate | Weakest class (F1 ≈ 0.64) |
| BKL   | Benign Keratosis              | Low      | Stable across runs |
| DF    | Dermatofibroma                | Low      | Recovered with focal loss |
| VASC  | Vascular Lesion               | Low      | Few samples but strong F1 |
| SCC   | Squamous Cell Carcinoma       | High     | F1 ≈ 0.72 |

Full per-class precision / recall / F1 tables and confusion matrices live in [docs/report/experiments_report.md](docs/report/experiments_report.md).

---

## System Architecture

```
┌─────────────────────────┐    HTTP / static    ┌──────────────────────────┐
│  Custom Web Frontend    │  ─────────────────> │   Flask + Gradio Server  │
│  HTML / CSS / JS        │  <───────────────── │   (Python)               │
│  Three.js warp grid     │                     │   Port 8000              │
└─────────────────────────┘                     └──────────────────────────┘
        │                                                    │
        v                                                    v
   UI Stack                                          ML Pipeline
   - Three.js r128                          ┌──────────────────────────┐
   - GSAP animations                        │  Inference               │
   - Custom cursor                          │  - Albumentations preproc│
   - Glassmorphism modal                    │  - D4 Test-Time Aug      │
                                            │  - Grad-CAM heatmap      │
                                            └──────────────────────────┘
                                                         │
                                                         v
                                            ┌──────────────────────────┐
                                            │  EfficientNet-B3         │
                                            │  + Dropout(0.3)          │
                                            │  + Linear(→ 8 classes)   │
                                            │  Weighted F1 = 0.8502    │
                                            └──────────────────────────┘
```

---

## Project Structure

```
derm-vision/
│
├── notebooks/
│   ├── eda.ipynb                       Exploratory data analysis
│   ├── preprocessing.ipynb             Preprocessing + split generation
│   └── train_colab.ipynb               Colab training notebook
│
├── src/
│   ├── dataset.py                      ISICDataset (PyTorch Dataset)
│   ├── transforms.py                   Albumentations train/val pipelines
│   ├── train.py                        Training loop + W&B logging
│   ├── evaluate.py                     Metrics + confusion matrix
│   ├── gradcam.py                      Grad-CAM heatmap generation
│   └── models/
│       ├── efficientnet.py             Primary backbone (EfficientNet-B3)
│       ├── custom_cnn.py               4-block CNN baseline
│       └── ensemble.py                 Weighted softmax ensemble
│
├── app/
│   ├── server.py                       Flask server entry point
│   ├── app.py                          Gradio analyzer + inference glue
│   └── web/                            Custom HTML/CSS/JS frontend
│       ├── index.html                  Homepage (Three.js warp grid)
│       ├── analyze.html                Analyze page (upload + results)
│       ├── about.html                  About page
│       ├── css/styles.css              Design tokens, components, layout
│       └── js/                         Three.js, analyze flow, modal
│
├── configs/
│   ├── config.yaml                     EfficientNet-B3 training config
│   └── config_cnn.yaml                 Custom CNN baseline config
│
├── data/
│   ├── raw/                            ISIC 2019 images + CSVs (gitignored)
│   └── splits/                         80/10/10 stratified split CSVs
│
├── outputs/
│   ├── checkpoints/                    Trained weights (one folder per W&B run)
│   └── results/                        Confusion matrices + metric dumps
│
├── docs/
│   ├── proposal/                       Project proposal
│   └── report/                         Progress + experiments reports
│
├── scripts/                            Split generation, helpers
├── presentation/                       Slides
├── requirements.txt                    Python dependencies
└── LICENSE
```

---

## Methodology

### Dataset
- **Source**: [ISIC 2019 Challenge](https://challenge2019.isic-archive.com/) training set
- **Size**: 25,331 dermoscopy images
- **Classes**: 8 diagnostic categories (MEL, NV, BCC, AKIEC, BKL, DF, VASC, SCC)
- **Splits**: 80 / 10 / 10 stratified train / val / test
- **Core Challenge**: Severe class imbalance — **NV alone is over half the dataset**, while DF and VASC each have under 260 images (54× imbalance ratio between NV and DF)

Drop the ISIC 2019 images and CSVs into `data/raw/`, then regenerate splits:

```bash
python scripts/create_splits.py
```

### Preprocessing & Augmentation Pipeline (Albumentations)

1. **Resize / Crop**: 224×224 baseline → 300×300 for the final EfficientNet-B3 run
2. **Geometric**: random flips (H/V), rotations, shift-scale-rotate
3. **Photometric**: color jitter, brightness/contrast, hue shift
4. **Regularization**: coarse dropout (cutout-style), normalization with ImageNet mean/std
5. **Test-Time Augmentation**: D4 group (8 flip/rotate variants), softmax-averaged at inference

### Models Developed

**Custom CNN Baseline** (`frosty-deluge-6`, `comfy-frost-7`):
- 4 convolutional blocks, no pretraining
- Weighted F1 = 0.6511 — establishes the "from scratch" floor

**EfficientNet-B3 — Weighted CE** (`splendid-cloud-3`):
- ImageNet pretrained, head warmup then full fine-tune
- Inverse-frequency class weights (DF=13.27 → NV=0.25)
- Weighted F1 = 0.7856, Balanced Accuracy = 0.7708

**EfficientNet-B3 — Focal Loss + Class Weights** (`usual-salad-4`):
- Focal γ=2.0 *plus* class weights
- Pushed rare classes (DF 0.55 → 0.79, SCC 0.63 → 0.74) but **collapsed NV** (recall 0.56) and dropped weighted F1

**EfficientNet-B3 — Focal Loss only** (`cardassian-spot-5`) — **FINAL**:
- ImageNet pretrained EfficientNet-B3, `Dropout(0.3) → Linear(→ 8)` head
- Focal loss γ=2 (no extra class weights — focal handles imbalance on its own)
- 5-epoch head warmup with backbone frozen → unfreeze and fine-tune full network
- Cosine-annealed LR, AdamW
- D4 TTA at inference
- **Weighted F1 = 0.8502, Balanced Accuracy = 0.7779**

### Training & Evaluation

```bash
# Train the final EfficientNet-B3 configuration
python -m src.train --config configs/config.yaml

# Train the CNN baseline
python -m src.train --config configs/config_cnn.yaml
```

Training expects Weights & Biases (`wandb login`) or `WANDB_MODE=disabled`.

Quick eval from Python:

```python
from src.evaluate import compute_metrics, plot_confusion_matrix

metrics = compute_metrics(y_true, y_pred)
plot_confusion_matrix(y_true, y_pred, save_path="outputs/results/cm.png")
```

---

## Frontend Features

- **Three.js warp grid homepage** — fisheye/barrel-distorted dermoscopy gallery with mouse parallax and post-process shader
- **Glassmorphism analyze modal** — backdrop blur, source cards (Upload / Camera / Paste), animated entrance
- **In-browser image input** — drag-and-drop upload, camera capture, paste-from-clipboard
- **Real-time results** — diagnosis card, per-class probability bars, risk-level color coding
- **Grad-CAM explainability** — heatmap overlay showing image regions driving the prediction
- **Responsive design** — works on mobile, tablet, and desktop
- **Medical disclaimer** — clearly framed as educational / research, not diagnostic

### Tech Stack
- **Backend**: Python 3.9+, Flask, Gradio, PyTorch, Albumentations, timm
- **Frontend**: Vanilla HTML / CSS / JS, Three.js r128, GSAP 3.12
- **ML**: PyTorch, EfficientNet-B3 (timm), Focal loss, D4 TTA
- **Tracking**: Weights & Biases

---

## API Endpoints

The Gradio analyzer is mounted under the Flask app and accepts a single image, returning per-class probabilities plus the Grad-CAM heatmap.

### POST /analyze
Runs inference on an uploaded image.

**Request** (multipart):
- `image`: JPEG / PNG dermoscopy image (max ~10MB)

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
Serves the homepage (Three.js warp grid gallery).

### GET /analyze
Serves the analyze page UI.

### GET /about
Serves the About page (project explainer, team, methodology).

---

## Documentation

| File | Purpose | Audience |
|------|---------|----------|
| [README.md](README.md) | Project overview (this file) | Everyone |
| [docs/report/experiments_report.md](docs/report/experiments_report.md) | Full training runs, ablations, per-class metrics | ML reviewers |
| [docs/report/progress_report.md](docs/report/progress_report.md) | Course progress report | Instructor |
| [docs/proposal/](docs/proposal/) | Original project proposal | Instructor |

---

## Modeling Progress & Findings

### Why focal loss alone (not focal + class weights)

We tried three loss configurations on EfficientNet-B3:

1. **Weighted cross-entropy** — solid weighted F1 (0.7856), but balanced accuracy capped at 0.77.
2. **Focal loss + class weights** — balanced accuracy jumped to 0.82, but NV recall collapsed to 0.56 and weighted F1 dropped to 0.71. Because NV is half the test set, that pulled overall F1 down even though rare classes improved.
3. **Focal loss alone** — focal's γ=2 down-weighting of easy examples handled the imbalance on its own. NV recovered to F1 = 0.91, rare classes stayed above the weighted-CE baseline, and weighted F1 hit **0.8502**.

The lesson: stacking imbalance corrections (focal + class weights) over-corrected on the dominant class. Focal loss alone is the sweet spot for this dataset.

### Test-Time Augmentation gain
D4 TTA (8 flip/rotate variants, softmax-averaged) gave us **+1.9 points balanced accuracy and +1.4 points weighted F1 with no retraining** — essentially free.

### Remaining weak spot
**AKIEC** sits at F1 ≈ 0.64 — our weakest class. It's clinically close to BCC and BKL and gets confused with both. Improving AKIEC without dragging neighbors down is the natural next experiment.

### Gap to course target
- **Balanced accuracy target**: 75% → **achieved 77.79%** (+2.79%)
- **Per-class accuracy ≥ 60% on rare classes**: achieved on DF, VASC, SCC; AKIEC remains the holdout

---

## Future Improvements

### Short-term (1–2 weeks)
- AKIEC-focused fine-tune (oversampling + targeted augmentation)
- Per-class confusion matrix error analysis paired with Grad-CAM
- Calibration analysis (reliability diagrams, temperature scaling)

### Medium-term (1–2 months)
- Ensemble of EfficientNet-B3 + ConvNeXt + ViT with weighted softmax
- Train at 380×380 (EfficientNet-B3 native resolution) on more epochs
- Multi-crop TTA on top of D4

### Long-term (3–6 months)
- External validation on HAM10000 / PH² datasets
- Lightweight on-device variant (EfficientNet-B0 distillation) for mobile
- Clinician-in-the-loop evaluation with a partnering dermatology group

---

## Resources

### Datasets
- [ISIC 2019 Challenge](https://challenge2019.isic-archive.com/)
- [ISIC Archive](https://www.isic-archive.com/)

### Technical References
- [EfficientNet: Rethinking Model Scaling for CNNs (Tan & Le, 2019)](https://arxiv.org/abs/1905.11946)
- [Focal Loss for Dense Object Detection (Lin et al., 2017)](https://arxiv.org/abs/1708.02002)
- [Grad-CAM (Selvaraju et al., 2017)](https://arxiv.org/abs/1610.02391)
- [Albumentations Documentation](https://albumentations.ai/)
- [timm — PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)

---

## Notes on Apple Silicon

If you're on macOS with MPS, set `num_workers=0` in the DataLoader. PyTorch multiprocessing on Mac doesn't play nicely with MPS and you'll get spawn errors otherwise. If `albumentations` complains about NumPy, pin NumPy below 2.0.

---

## Team

- **Lam Nguyen** (SJSU ID: 018229432) — Data pipeline: ISICDataset, Albumentations augmentation, stratified splits
- **James Pham** — Model development: architecture selection, training runs, evaluation, ablations
- **Vi Thi Tuong Nguyen** — Web app: frontend (Three.js homepage, analyze flow, about page) and Grad-CAM integration

---

## License

The ISIC 2019 dataset is governed by its own license (see `data/raw/LICENSE.txt`). Project code is released for academic and research use under the terms in [LICENSE](LICENSE). Not for clinical or diagnostic use.
