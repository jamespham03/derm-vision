# CMPE 258 - Project Progress Report

**Project:** Skin Disease Classification System Using Deep Learning: A Multi-Class Approach

**Team Members:** Lam Nguyen, James Pham, Vi Thi Tuong Nguyen

**Date:** April 10, 2026

---

## 1. Project Overview

Our project builds a multi-class skin disease classification system using deep learning on the ISIC 2019 Challenge dataset. The system classifies dermoscopic images into 8 diagnostic categories: Melanoma (MEL), Melanocytic Nevus (NV), Basal Cell Carcinoma (BCC), Actinic Keratosis (AKIEC), Benign Keratosis (BKL), Dermatofibroma (DF), Vascular Lesion (VASC), and Squamous Cell Carcinoma (SCC).

**Goal:** Develop a model that achieves high balanced accuracy across all 8 classes, with particular emphasis on detecting clinically dangerous lesions (MEL, SCC, BCC) despite severe class imbalance.

**Why this matters:** Skin cancer is the most common cancer worldwide. Early detection significantly improves survival rates, yet access to dermatologists is limited. An automated classification system could assist clinicians in triaging suspicious lesions, especially in underserved areas. The ISIC 2019 dataset presents a realistic challenge with imbalanced classes and visually similar lesion types.

---

## 2. Dataset Investigation and Analysis

### 2.1 Dataset Source and Overview

- **Source:** ISIC 2019 Challenge Training Dataset (International Skin Imaging Collaboration)
- **Contributing institutions:** Hospital Clinic de Barcelona, Medical University of Vienna, Memorial Sloan Kettering Cancer Center
- **Total images:** 25,331 dermoscopic JPEG images
- **Classes:** 8 diagnostic categories (one-hot encoded in ground truth CSV)
- **Supplementary files:** Ground truth labels CSV, patient metadata CSV

### 2.2 Class Distribution

| Class | Disease | Count | % of Dataset |
|-------|---------|------:|:------------:|
| NV | Melanocytic Nevus | 12,875 | 50.8% |
| MEL | Melanoma | 4,522 | 17.9% |
| BCC | Basal Cell Carcinoma | 3,323 | 13.1% |
| BKL | Benign Keratosis | 2,624 | 10.4% |
| AKIEC | Actinic Keratosis | 867 | 3.4% |
| SCC | Squamous Cell Carcinoma | 628 | 2.5% |
| VASC | Vascular Lesion | 253 | 1.0% |
| DF | Dermatofibroma | 239 | 0.9% |

**Key finding:** Severe class imbalance with a 54x ratio between the largest class (NV) and the smallest (DF). A naive classifier predicting NV for every image would achieve 50.8% accuracy, making accuracy alone a misleading metric.

### 2.3 Patient Metadata Analysis

- **Age:** Range 0-85 years, median ~50. Bell-shaped distribution peaking at 40-60. Older patients more frequent in BCC, AKIEC, SCC (cumulative UV exposure). Missing in 1.7% of records.
- **Sex:** Roughly balanced (52.4% male, 46.0% female). DF shows a noticeable female skew; MEL and SCC are slightly male-skewed. Missing in 1.5% of records.
- **Anatomical site:** Most common sites are anterior torso (6,915), lower extremity (4,990), and head/neck (4,587). VASC lesions are predominantly on head/neck and lower extremity; DF clusters on the lower extremity. Missing in 10.4% of records.

### 2.4 Image Characteristics

- **Dimensions:** Non-uniform, ranging from 600-1024 pixels wide and 450-1024 pixels tall (mean ~861x774). Resizing is mandatory.
- **RGB intensity:** VASC shows the highest red-channel intensity (vascular lesions appear reddish); MEL tends toward darker overall intensity; NV and BKL show higher brightness.

### 2.5 Visualizations Completed

1. Class distribution bar charts and pie charts
2. Sample images for each class (2-4 per class)
3. Age distribution histograms and per-class boxplots
4. Sex distribution bar charts per class
5. Anatomical site frequency charts and heatmap
6. Image dimension histograms (width, height, aspect ratio)
7. RGB mean intensity per class
8. Augmentation effect visualizations (side-by-side comparisons)

---

## 3. Preprocessing Completed

### 3.1 Data Splitting

We performed a stratified 80/10/10 train/validation/test split using scikit-learn's `train_test_split` with a fixed random seed (42) for reproducibility. Stratification ensures all 8 classes are proportionally represented in every split, which is critical for minority classes like DF (239 total samples) and VASC (253 total samples).

- **Training set:** 20,264 samples (80%)
- **Validation set:** 2,534 samples (10%)
- **Test set:** 2,534 samples (10%)

### 3.2 Label Preparation

- Dropped the `UNK` (unknown) column from the ground truth CSV
- Renamed `AK` to `AKIEC` to match the project's 8-class convention
- Converted one-hot encoded labels to single integer class indices for use with PyTorch's `CrossEntropyLoss`

### 3.3 Image Preprocessing

- **Resize:** All images resized to 300x300 pixels (EfficientNet-B3's native resolution)
- **Normalization:** ImageNet mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225], matching the pretrained backbone's expected input distribution

### 3.4 Data Augmentation (Training Set Only)

We implemented a comprehensive augmentation pipeline using the Albumentations library:

| Transform | Configuration | Purpose |
|-----------|--------------|---------|
| Horizontal flip | p=0.5 | Lesions have no inherent left/right orientation |
| Vertical flip | p=0.5 | Dermoscopic images have no fixed orientation |
| Random 90-degree rotation | p=0.5 | Rotation invariance |
| Affine transformation | translate +/-10%, scale 0.85-1.15x, rotate +/-30 degrees, p=0.5 | Scale and position invariance |
| Color jitter / HueSaturationValue | One of two, p=0.5 | Simulate lighting and device variation |
| CoarseDropout | 4-8 holes, 8-16px, p=0.3 | Occlusion robustness (simulates hair, rulers) |

Validation and test sets use only deterministic resize and normalization (no augmentation).

### 3.5 Class Imbalance Handling

We explored three strategies across our training runs:

1. **Weighted cross-entropy loss** (Run 1): Inverse-frequency class weights passed to `nn.CrossEntropyLoss`
2. **Focal loss + class weights** (Run 2): Focal loss (gamma=2.0) combined with inverse-frequency weights
3. **Focal loss only** (Run 3): Focal loss (gamma=2.0) without class weights

We found that focal loss alone (strategy 3) achieved the best balance between common and rare class performance (see Section 4).

---

## 4. Current Progress

### 4.1 Completed Components

- **Data pipeline:** Custom PyTorch `ISICDataset` class with augmentation support, stratified splits, and class weight computation
- **EDA notebooks:** Two comprehensive EDA notebooks analyzing class distribution, metadata, image characteristics, and augmentation effects
- **Model implementations:**
  - EfficientNet-B3 with transfer learning (pretrained ImageNet weights, custom classification head, backbone freeze/unfreeze warmup strategy)
  - Custom 4-layer CNN baseline (Conv-BN-ReLU-MaxPool blocks, ~500K parameters)
- **Training pipeline:** Configurable training loop with focal loss, cosine annealing LR scheduler, early stopping, and Weights & Biases logging
- **Evaluation pipeline:** Balanced accuracy, weighted F1, per-class precision/recall, confusion matrix visualization
- **Cloud training:** Google Colab notebook with Kaggle dataset download, Google Drive checkpoint persistence, and test-time augmentation evaluation
- **Grad-CAM:** Implementation ready for model interpretability visualization
- **Gradio web app:** Stub for live inference demo

### 4.2 Training Experiments and Results

We conducted 4 training runs, systematically improving performance:

#### Run 1: EfficientNet-B3 Baseline (W&B: splendid-cloud-3)
- **Config:** Weighted cross-entropy loss, 224px images, lr=0.0001
- **Early stopped:** Epoch 18/50
- **Test results:** Balanced Accuracy = 0.7708, Weighted F1 = 0.7856

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| MEL | 0.67 | 0.68 | 0.67 |
| NV | 0.90 | 0.83 | 0.87 |
| BCC | 0.84 | 0.79 | 0.81 |
| AKIEC | 0.50 | 0.76 | 0.60 |
| BKL | 0.64 | 0.70 | 0.66 |
| DF | 0.42 | 0.79 | 0.55 |
| VASC | 0.82 | 0.92 | 0.87 |
| SCC | 0.58 | 0.70 | 0.63 |

**Findings:** Weak classes (DF=0.55, AKIEC=0.60, SCC=0.63) struggled. MEL and NV showed significant cross-confusion (120 NV misclassified as MEL, 79 MEL as NV).

#### Run 2: Focal Loss + Class Weights + 300px (W&B: usual-salad-4)
- **Config:** Focal loss (gamma=2.0) with class weights, 300px images
- **Test results with TTA:** Balanced Accuracy = 0.8215, Weighted F1 = 0.7102

**Findings:** Rare classes improved dramatically (DF: 0.55 to 0.79, SCC: 0.63 to 0.74), but common classes suffered (NV F1 dropped from 0.87 to 0.71). Focal loss combined with class weights over-corrected, causing the model to over-predict rare classes.

#### Run 3: Focal Loss Only + 300px (W&B: cardassian-spot-5) --- Best Model
- **Config:** Focal loss (gamma=2.0) without class weights, 300px images
- **Test results with TTA:** Balanced Accuracy = 0.7779, Weighted F1 = 0.8502

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| MEL | 0.82 | 0.73 | 0.77 |
| NV | 0.89 | 0.94 | 0.91 |
| BCC | 0.89 | 0.90 | 0.90 |
| AKIEC | 0.60 | 0.69 | 0.64 |
| BKL | 0.78 | 0.68 | 0.73 |
| DF | 0.78 | 0.58 | 0.67 |
| VASC | 0.92 | 0.96 | 0.94 |
| SCC | 0.70 | 0.75 | 0.72 |

**Findings:** Best overall balance. Every class improved over Run 1. TTA provided +1.4% F1 improvement for free.

#### Run 4: Custom CNN Baseline (W&B: comfy-frost-7)
- **Config:** 4-layer CNN, focal loss, 300px images, lr=0.001, trained from scratch
- **Trained:** 50 full epochs (no early stopping triggered)
- **Test results:** Balanced Accuracy = 0.3796, Weighted F1 = 0.6511

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| MEL | 0.59 | 0.42 | 0.49 |
| NV | 0.76 | 0.90 | 0.83 |
| BCC | 0.55 | 0.83 | 0.66 |
| AKIEC | 0.56 | 0.17 | 0.27 |
| BKL | 0.56 | 0.30 | 0.39 |
| DF | 0.00 | 0.00 | 0.00 |
| VASC | 1.00 | 0.56 | 0.72 |
| SCC | 1.00 | 0.02 | 0.03 |

**Findings:** The CNN completely failed on DF (0.00 F1) and SCC (0.03 F1), demonstrating that a simple architecture trained from scratch lacks the capacity to learn discriminative features for rare classes. This strongly validates the transfer learning approach.

### 4.3 Model Comparison Summary

| Metric | Custom CNN | EfficientNet-B3 (Run 1) | EfficientNet-B3 + TTA (Run 3) |
|--------|-----------|------------------------|-------------------------------|
| Accuracy | 0.69 | 0.78 | **0.85** |
| Balanced Accuracy | 0.38 | 0.77 | **0.78** |
| Weighted F1 | 0.65 | 0.79 | **0.85** |

### 4.4 Test-Time Augmentation Impact

TTA (D4 transform: 8 variants of flips and 90-degree rotations, predictions averaged) consistently improved results:

| Metric | Without TTA | With TTA | Improvement |
|--------|------------|----------|-------------|
| Balanced Accuracy | 0.7594 | 0.7779 | +0.0185 |
| Weighted F1 | 0.8360 | 0.8502 | +0.0143 |

---

## 5. Challenges Faced

### 5.1 Severe Class Imbalance (54x ratio)
The dominant challenge. NV accounts for 50.8% of data while DF and VASC have fewer than 260 samples each. We experimented with three loss strategies and found that focal loss without class weights achieved the best trade-off. Using both focal loss and class weights simultaneously over-corrected, hurting common class performance.

### 5.2 Visual Similarity Between Classes
MEL and NV are visually similar under dermoscopy, causing significant cross-confusion (the most common error pattern across all runs). BKL also gets confused with MEL and NV. This is a recognized challenge in dermatology even for expert clinicians.

### 5.3 Compute Constraints
We relied on Google Colab Pro for GPU training (Tesla T4). After exhausting the GPU runtime quota, we trained the CNN baseline locally on a MacBook Pro M1 Pro using MPS acceleration. We encountered macOS-specific issues: `num_workers > 0` caused multiprocessing crashes, and NumPy 2.x was incompatible with some Anaconda packages (numexpr, bottleneck). These were resolved by setting `num_workers=0` and upgrading the affected packages.

### 5.4 Non-Uniform Image Dimensions
Images range from 600-1024 pixels with varying aspect ratios across contributing institutions. This required mandatory resizing. We found that increasing from 224px to 300px (EfficientNet-B3's native resolution) yielded measurable improvements, confirming that fine-grained lesion details matter.

---

## 6. Plan for Completion

### Remaining Tasks

| Task | Owner | Target Date |
|------|-------|-------------|
| Grad-CAM visualizations for model interpretability | Vi | April 14 |
| Gradio web app integration with best checkpoint | Vi | April 18 |
| Additional model experiments (ResNet-50, DenseNet-121) | James | April 18 |
| Ensemble model (weighted average of top models) | James | April 21 |
| Final evaluation and results compilation | Lam | April 23 |
| Final report writing | All | April 28 |
| Presentation preparation | All | April 30 |

### Models/Methods Still Planned
- **Additional architectures:** ResNet-50 and DenseNet-121 for broader model comparison
- **Ensemble:** Weighted average of top-performing models using the existing `WeightedEnsemble` class
- **Grad-CAM:** Generate activation maps to verify the model focuses on lesion regions rather than artifacts

### Evaluation Plan
- Primary metrics: balanced accuracy and weighted F1 (both handle class imbalance)
- Per-class precision, recall, and F1 for detailed analysis
- Confusion matrices to identify remaining error patterns
- Grad-CAM visualizations for qualitative assessment
- Comparison table across all architectures and configurations

---

## 7. Team Contributions

| Member | Contributions So Far |
|--------|---------------------|
| **Lam Nguyen** | Project scaffold and repository setup. Data pipeline implementation (ISICDataset, transforms, splits). Training loop with focal loss, cosine annealing, early stopping, and W&B integration. Google Colab training notebook with Kaggle integration. Conducted all 4 training runs and iterative loss function tuning. Per-run output organization. |
| **James Pham** | EDA notebook with comprehensive dataset analysis (class distribution, metadata, image characteristics, RGB intensity). Preprocessing pipeline with metadata encoding (age normalization, sex/site encoding). Stratified train/val/test split implementation. Class weight computation. |
| **Vi Thi Tuong Nguyen** | Literature review on skin lesion classification approaches. Grad-CAM implementation for model interpretability. Gradio web app stub for deployment. Progress report outline and EDA summary documentation. |
