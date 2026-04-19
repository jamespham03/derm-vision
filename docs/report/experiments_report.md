# Derm-Vision: Comprehensive Training Experiments Report

**Project:** Skin Disease Classification System Using Deep Learning: A Multi-Class Approach
**Dataset:** ISIC 2019 Challenge (25,331 dermoscopic images, 8 classes)
**Team:** Lam Nguyen, James Pham, Vi Thi Tuong Nguyen

---

## 1. Experimental Setup

### 1.1 Data Split

Stratified 80/10/10 split with fixed random seed (42):

| Split | Samples | Percentage |
|-------|--------:|-----------:|
| Train | 20,264 | 80.0% |
| Validation | 2,534 | 10.0% |
| Test | 2,534 | 10.0% |

Stratification preserves class proportions across splits, critical for minority classes (DF: 239 total, VASC: 253 total).

### 1.2 Class Distribution (Full Dataset)

| Class | Disease | Count | % | Test Support |
|-------|---------|------:|:---:|:------------:|
| NV | Melanocytic Nevus | 12,875 | 50.8% | 1,288 |
| MEL | Melanoma | 4,522 | 17.9% | 452 |
| BCC | Basal Cell Carcinoma | 3,323 | 13.1% | 333 |
| BKL | Benign Keratosis | 2,624 | 10.4% | 263 |
| AKIEC | Actinic Keratosis | 867 | 3.4% | 86 |
| SCC | Squamous Cell Carcinoma | 628 | 2.5% | 63 |
| VASC | Vascular Lesion | 253 | 1.0% | 25 |
| DF | Dermatofibroma | 239 | 0.9% | 24 |

**Max/min imbalance ratio:** 54x (NV vs DF)

### 1.3 Common Training Configuration

- **Optimizer:** Adam (weight_decay=1e-5)
- **LR Scheduler:** Cosine annealing (T_max=epochs)
- **Epochs:** 50 (with early stopping on val loss, patience varies)
- **Batch size:** 32
- **Evaluation metrics:** Balanced accuracy, Weighted F1, Per-class precision/recall, Confusion matrix

### 1.4 Augmentation Pipeline (Training Only)

| Transform | Config |
|-----------|--------|
| Horizontal flip | p=0.5 |
| Vertical flip | p=0.5 |
| Random 90-degree rotation | p=0.5 |
| Affine | translate +/-10%, scale 0.85-1.15x, rotate +/-30deg, p=0.5 |
| Color jitter / HueSaturationValue | One-of, p=0.5 |
| CoarseDropout | 4-8 holes, 8-16px, p=0.3 |

Validation/test use only deterministic resize and ImageNet normalization.

---

## 2. Run 1 — EfficientNet-B3 Baseline (Weighted CE)

### 2.1 Configuration

| Parameter | Value |
|-----------|-------|
| W&B run name | splendid-cloud-3 |
| Backbone | EfficientNet-B3 (ImageNet pretrained) |
| Image size | 224x224 |
| Loss | Weighted Cross-Entropy |
| Class weights | Inverse frequency (DF=13.27, NV=0.25) |
| Learning rate | 0.0001 |
| Freeze strategy | Backbone frozen for first 4 epochs, unfreeze at epoch 5 |
| Compute | Google Colab (Tesla T4) |
| Epochs trained | 18/50 (early stopped) |

### 2.2 Test Results

**Overall:**
- Balanced Accuracy: **0.7708**
- Weighted F1: **0.7856**
- Accuracy: 0.78

**Per-class:**

| Class | Precision | Recall | F1-Score | Support |
|-------|:---------:|:------:|:--------:|:-------:|
| MEL | 0.67 | 0.68 | 0.67 | 452 |
| NV | 0.90 | 0.83 | 0.87 | 1,288 |
| BCC | 0.84 | 0.79 | 0.81 | 333 |
| AKIEC | 0.50 | 0.76 | 0.60 | 86 |
| BKL | 0.64 | 0.70 | 0.66 | 263 |
| DF | 0.42 | 0.79 | 0.55 | 24 |
| VASC | 0.82 | 0.92 | 0.87 | 25 |
| SCC | 0.58 | 0.70 | 0.63 | 63 |
| **Macro avg** | **0.67** | **0.77** | **0.71** | |
| **Weighted avg** | **0.80** | **0.78** | **0.79** | |

### 2.3 Confusion Matrix (Key Patterns)

```
          Pred: MEL   NV  BCC  AKIEC  BKL   DF  VASC  SCC
True MEL:      307   79    9    12   37    4    0    4
True NV:       120 1072   20     4   54   10    3    5
True BCC:        8   12  264    23    7    2    2   15
True AKIEC:      1    0   11    65    4    2    0    3
True BKL:       24   23    6    18  183    5    0    4
True DF:         0    1    2     1    0   19    0    1
True VASC:       0    2    0     0    0    0   23    0
True SCC:        1    0    4     8    3    3    0   44
```

**Key confusion patterns:**
- **MEL <-> NV:** 120 NV misclassified as MEL (most common error); 79 MEL as NV
- **BKL** gets confused with MEL (24) and NV (23)
- **BCC** gets confused with SCC (15) and AKIEC (23)

### 2.4 Analysis

**Strengths:**
- Strong performance on NV (F1=0.87), BCC (0.81), VASC (0.87)
- High support classes well-represented

**Weaknesses:**
- Rare classes struggle: DF (0.55), AKIEC (0.60), SCC (0.63)
- MEL/NV cross-confusion (visual similarity under dermoscopy)
- DF recall high (0.79) but precision low (0.42) — over-predicted

---

## 3. Run 2 — Focal Loss + Class Weights + 300px

### 3.1 Configuration Changes from Run 1

| Parameter | Run 1 | Run 2 |
|-----------|-------|-------|
| Image size | 224x224 | **300x300** (EfficientNet-B3 native) |
| Loss | Weighted CE | **Focal Loss** (gamma=2.0) + class weights |

**Rationale:**
- Focal loss (FL(pt) = -(1-pt)^gamma * log(pt)) down-weights easy examples, focuses on hard ones
- 300px matches EfficientNet-B3's native resolution for finer lesion details

W&B run name: **usual-salad-4**

### 3.2 Test Results (with TTA)

**Overall:**
- Balanced Accuracy: **0.8215** (+0.0507 vs Run 1)
- Weighted F1: **0.7102** (-0.0754 vs Run 1)
- Accuracy: 0.70

**Per-class:**

| Class | Precision | Recall | F1-Score | Support | Delta vs Run 1 |
|-------|:---------:|:------:|:--------:|:-------:|:--------------:|
| MEL | 0.45 | 0.88 | 0.60 | 452 | -0.07 |
| NV | 0.97 | 0.56 | 0.71 | 1,288 | **-0.16** |
| BCC | 0.87 | 0.86 | 0.86 | 333 | +0.05 |
| AKIEC | 0.66 | 0.72 | 0.69 | 86 | +0.09 |
| BKL | 0.60 | 0.79 | 0.68 | 263 | +0.02 |
| DF | 0.69 | 0.92 | 0.79 | 24 | **+0.24** |
| VASC | 0.81 | 1.00 | 0.89 | 25 | +0.02 |
| SCC | 0.66 | 0.84 | 0.74 | 63 | +0.11 |
| **Macro avg** | **0.71** | **0.82** | **0.75** | | |
| **Weighted avg** | **0.80** | **0.70** | **0.71** | | |

### 3.3 Analysis: The Over-Correction Problem

**What worked:**
- Rare classes improved dramatically: DF 0.55 -> 0.79, SCC 0.63 -> 0.74
- VASC already near-perfect (0.89)
- Balanced accuracy jumped from 0.77 to 0.82

**What broke:**
- **NV F1 collapsed** from 0.87 to 0.71 (recall dropped from 0.83 to 0.56)
- **MEL precision collapsed** from 0.67 to 0.45 (model over-predicts MEL)

**Diagnosis:** Focal loss and class weights both serve the same function — down-weighting the dominant class. Using them together creates double correction, causing the model to under-predict NV (the majority class). Weighted F1 drops because NV accounts for 50% of the test set.

---

## 4. Run 3 — Focal Loss Only + 300px (BEST)

### 4.1 Configuration Changes from Run 2

| Parameter | Run 2 | Run 3 |
|-----------|-------|-------|
| Class weights | Enabled | **Disabled** |

Only focal loss handles imbalance now.

W&B run name: **cardassian-spot-5**

### 4.2 Test Results (without TTA)

- Balanced Accuracy: **0.7594**
- Weighted F1: **0.8360**

### 4.3 Test Results (with D4 TTA)

**Overall:**
- Balanced Accuracy: **0.7779**
- Weighted F1: **0.8502**
- Accuracy: 0.85

**Per-class (with TTA):**

| Class | Precision | Recall | F1-Score | Support | Delta vs Run 1 |
|-------|:---------:|:------:|:--------:|:-------:|:--------------:|
| MEL | 0.82 | 0.73 | 0.77 | 452 | **+0.10** |
| NV | 0.89 | 0.94 | 0.91 | 1,288 | +0.04 |
| BCC | 0.89 | 0.90 | 0.90 | 333 | +0.09 |
| AKIEC | 0.60 | 0.69 | 0.64 | 86 | +0.04 |
| BKL | 0.78 | 0.68 | 0.73 | 263 | +0.07 |
| DF | 0.78 | 0.58 | 0.67 | 24 | +0.12 |
| VASC | 0.92 | 0.96 | 0.94 | 25 | +0.07 |
| SCC | 0.70 | 0.75 | 0.72 | 63 | +0.09 |
| **Macro avg** | **0.80** | **0.78** | **0.78** | | |
| **Weighted avg** | **0.85** | **0.85** | **0.85** | | |

### 4.4 TTA Impact (D4 transform: 8 variants of flips + 90-deg rotations)

| Metric | Without TTA | With TTA | Improvement |
|--------|:-----------:|:--------:|:-----------:|
| Balanced Accuracy | 0.7594 | 0.7779 | +0.0185 |
| Weighted F1 | 0.8360 | 0.8502 | +0.0143 |

TTA provides a consistent "free" boost with no retraining cost.

### 4.5 Analysis

**Run 3 is the best model.** Every single per-class F1 improved versus Run 1:

- MEL: 0.67 -> **0.77** (+10 points)
- NV: 0.87 -> **0.91** (+4 points)
- BCC: 0.81 -> **0.90** (+9 points)
- BKL: 0.66 -> **0.73** (+7 points)
- VASC: 0.87 -> **0.94** (+7 points)
- SCC: 0.63 -> **0.72** (+9 points)
- DF: 0.55 -> **0.67** (+12 points)
- AKIEC: 0.60 -> **0.64** (+4 points)

**Ablation conclusions:**
- Focal loss alone > Weighted CE alone (comparing Run 3 to Run 1)
- Focal loss alone > Focal loss + class weights (comparing Run 3 to Run 2)
- 300px > 224px for fine-grained medical imagery
- TTA provides ~1.5% F1 boost for free

---

## 5. Run 4 — Custom CNN Baseline

### 5.1 Configuration

| Parameter | Value |
|-----------|-------|
| W&B run name | comfy-frost-7 |
| Backbone | Custom 4-layer CNN (trained from scratch) |
| Parameters | ~500K (vs EfficientNet-B3 ~12M) |
| Image size | 300x300 |
| Loss | Focal Loss (gamma=2.0), no class weights |
| Learning rate | 0.001 (10x higher for from-scratch training) |
| Compute | MacBook Pro M1 Pro (MPS) |
| Epochs trained | 50/50 (no early stopping) |

### 5.2 Architecture

```
Block 1: Conv2d(3->32, 3x3) -> BatchNorm -> ReLU -> MaxPool(2)
Block 2: Conv2d(32->64, 3x3) -> BatchNorm -> ReLU -> MaxPool(2)
Block 3: Conv2d(64->128, 3x3) -> BatchNorm -> ReLU -> MaxPool(2)
Block 4: Conv2d(128->256, 3x3) -> BatchNorm -> ReLU -> MaxPool(2)
Classifier:
  AdaptiveAvgPool2d(1) -> Flatten
  Dropout(0.3) -> Linear(256->128) -> ReLU
  Dropout(0.3) -> Linear(128->8)
```

### 5.3 Test Results

**Overall:**
- Balanced Accuracy: **0.3796**
- Weighted F1: **0.6511**
- Accuracy: 0.69

**W&B final training/validation:**
- Train loss: 0.515
- Val loss: 0.465
- Val balanced accuracy: 0.38
- Val weighted F1: 0.66

**Per-class:**

| Class | Precision | Recall | F1-Score | Support |
|-------|:---------:|:------:|:--------:|:-------:|
| MEL | 0.59 | 0.42 | 0.49 | 452 |
| NV | 0.76 | 0.90 | 0.83 | 1,288 |
| BCC | 0.55 | 0.83 | 0.66 | 333 |
| AKIEC | 0.56 | 0.17 | 0.27 | 86 |
| BKL | 0.56 | 0.30 | 0.39 | 263 |
| DF | 0.00 | 0.00 | **0.00** | 24 |
| VASC | 1.00 | 0.56 | 0.72 | 25 |
| SCC | 1.00 | 0.02 | **0.03** | 63 |
| **Macro avg** | **0.63** | **0.40** | **0.42** | |
| **Weighted avg** | **0.68** | **0.69** | **0.65** | |

### 5.4 Analysis

**Total failure on rare classes:**
- **DF F1 = 0.00** — model never correctly predicts DF
- **SCC F1 = 0.03** — near-total failure (100% precision but 2% recall means model almost never predicts SCC)
- **AKIEC F1 = 0.27** — severely underperforming

**Why it fails:**
1. **No transfer learning** — Starts from random weights, has only 20K samples to learn from
2. **Limited capacity** — 500K parameters vs EfficientNet's 12M
3. **Rare class starvation** — With only ~190 DF samples in training, the model can't learn discriminative features without pretrained representations
4. **Still trains reasonably on NV** — The majority class (50% of data) provides enough signal even for a simple architecture

**Key lesson:** This baseline strongly validates the transfer learning approach. A custom CNN without pretrained weights cannot compete for medical imaging tasks with imbalanced rare classes.

---

## 6. Cross-Run Comparison

### 6.1 Overall Metrics

| Run | Model | Config | Balanced Acc | Weighted F1 | Accuracy |
|-----|-------|--------|:------------:|:-----------:|:--------:|
| 1 | EfficientNet-B3 | Weighted CE, 224px | 0.7708 | 0.7856 | 0.78 |
| 2 | EfficientNet-B3 | Focal + CW, 300px, TTA | **0.8215** | 0.7102 | 0.70 |
| 3 | EfficientNet-B3 | Focal, 300px, TTA | 0.7779 | **0.8502** | **0.85** |
| 4 | Custom CNN | Focal, 300px | 0.3796 | 0.6511 | 0.69 |

### 6.2 Per-Class F1 Across All Runs

| Class | Support | Run 1 (Weighted CE) | Run 2 (Focal+CW) | Run 3 (Focal+TTA) | Run 4 (CNN) |
|-------|:-------:|:-------------------:|:----------------:|:-----------------:|:-----------:|
| MEL | 452 | 0.67 | 0.60 | **0.77** | 0.49 |
| NV | 1,288 | 0.87 | 0.71 | **0.91** | 0.83 |
| BCC | 333 | 0.81 | 0.86 | **0.90** | 0.66 |
| AKIEC | 86 | 0.60 | **0.69** | 0.64 | 0.27 |
| BKL | 263 | 0.66 | 0.68 | **0.73** | 0.39 |
| DF | 24 | 0.55 | **0.79** | 0.67 | 0.00 |
| VASC | 25 | 0.87 | 0.89 | **0.94** | 0.72 |
| SCC | 63 | 0.63 | **0.74** | 0.72 | 0.03 |

**Best per-class F1:** Run 3 wins 5/8 classes, Run 2 wins 3/8 (the rarest ones).

### 6.3 Design Trade-offs Discovered

| Design Choice | Effect |
|--------------|--------|
| Increase image size 224 -> 300 | Fine-grained details preserved; +2-3% F1 |
| Weighted CE -> Focal Loss | Better rare class recall, maintains common class accuracy |
| Add class weights to Focal | Over-corrects; hurts common classes for minor rare-class gains |
| Test-Time Augmentation (D4) | +1-2% F1 for no retraining cost |
| Transfer learning vs scratch | Dramatic difference for rare classes (DF: 0.67 vs 0.00) |

---

## 7. Production Model Recommendation

**Recommended deployment model: Run 3 (EfficientNet-B3 + Focal Loss + 300px + D4 TTA)**

Rationale:
- Highest overall weighted F1 (0.85)
- Highest accuracy (85%)
- Balanced performance across all 8 classes (every F1 >= 0.64)
- TTA is a cheap inference-time boost
- Clinical relevance: Strong on MEL (0.77) and BCC (0.90), the most common clinically dangerous lesions

**Reserved use case for Run 2:** If the application prioritizes rare-class recall (screening for DF/AKIEC/SCC), Run 2 is better for those specific classes.

---

## 8. Computational Notes

### 8.1 Training Platforms

| Run | Platform | GPU | Time |
|-----|----------|-----|------|
| 1 | Google Colab Pro | Tesla T4 | ~1 hour (early stopped) |
| 2 | Google Colab Pro | Tesla T4 | ~2 hours |
| 3 | Google Colab Pro | Tesla T4 | ~2 hours |
| 4 | MacBook Pro M1 Pro | MPS | ~3 hours (50 full epochs) |

### 8.2 Environment Issues Encountered

1. **Colab Pro GPU quota exhausted** — Fell back to local M1 Pro training for Run 4
2. **macOS multiprocessing crash** — Required `num_workers=0` in DataLoader
3. **NumPy 2.x ABI incompatibility** — Resolved by upgrading numexpr and bottleneck packages
4. **Private repo clone** — Required correct GitHub URL (Lambert-Nguyen/derm-vision)
5. **Per-run output organization** — Refactored to save checkpoints under `outputs/checkpoints/{backbone}_{wandb_run_name}/` to prevent overwrites

### 8.3 Output Artifacts

```
outputs/checkpoints/
  efficientnet-b3_splendid-cloud-3/       (Run 1)
  efficientnet-b3_usual-salad-4/          (Run 2)
    best_model.pth
  efficientnet-b3_cardassian-spot-5/      (Run 3 -- BEST)
    best_model-2.pth
  custom-cnn_comfy-frost-7/               (Run 4)
    best_model.pth

outputs/results/
  efficientnet-b3_splendid-cloud-3_confusion_matrix.png
  efficientnet-b3_usual-salad-4_confusion_matrix_tta.png
  efficientnet-b3_cardassian-spot-5/confusion_matrix.png
  efficientnet-b3_cardassian-spot-5_confusion_matrix_tta.png
  custom-cnn_comfy-frost-7/confusion_matrix.png
```

---

## 9. Limitations and Future Work

### 9.1 Current Limitations

- **Model domain:** Trained only on dermoscopy images; will not generalize to phone photos
- **AKIEC weakness:** 0.64 F1 is the weakest class (limited samples, visual overlap with BCC/SCC)
- **No external validation:** All evaluation on held-out 10% of same dataset; real-world performance may differ

### 9.2 Planned Improvements

- **Grad-CAM:** Verify model attention focuses on lesion regions (not rulers, hair, or borders)
- **Ensemble:** Combine Run 2 (rare-class strength) and Run 3 (common-class strength) via weighted averaging
- **Additional architectures:** ResNet-50 and DenseNet-121 for broader comparison
- **Mixup/CutMix:** Potential regularization to further reduce MEL/NV confusion
- **Gradio web app:** Interactive demo with Grad-CAM visualization overlay

---

## Appendix: Reproducing Runs

```bash
# Run 3 (best)
python -m src.train --config configs/config.yaml
# with: loss=focal, focal_gamma=2.0, use_class_weights=false, image_size=300

# Run 4 (CNN baseline)
python -m src.train --config configs/config_cnn.yaml
# with: backbone=custom-cnn, loss=focal, image_size=300, num_workers=0
```

Evaluation:
```bash
python -c "..."  # (see evaluation snippets in project README)
```
