# Skin Disease Classification System Using Deep Learning: A Multi-Class Approach

## Project Progress Report

**Team Members:** Lam Nguyen, James Pham, Vi Thi Tuong Nguyen
**Course:** CMPE 258
**Date:** March 28, 2026

---

## 1. Project Overview

Our project aims to build a deep learning system that classifies dermoscopic skin lesion images into eight disease categories. The goal is to create an accessible tool that could help both patients and healthcare professionals identify skin conditions earlier, since early detection of diseases like melanoma can be life-saving.

We are working with the ISIC 2019 Challenge Training Dataset, which contains 25,331 labeled dermoscopic images. Our approach involves comparing transfer learning models (EfficientNet, ResNet, DenseNet), a custom CNN baseline, and ensemble methods. The system will take in a skin lesion image and output a predicted disease category along with a confidence score.

This problem matters because over 1.5 million skin cancer cases are diagnosed globally each year, and many people delay seeking evaluation due to cost or access barriers. An automated screening tool could help flag suspicious lesions for further medical review.

## 2. Dataset Investigation and Analysis

### 2.1 Dataset Description

We are using the ISIC 2019 Challenge Training Dataset from Kaggle (https://www.kaggle.com/datasets/andrewmvd/isic-2019). The dataset consists of:

- **25,331 dermoscopic images** in JPEG format, sourced from institutions including Hospital Clinic de Barcelona, Medical University of Vienna, and Memorial Sloan Kettering Cancer Center
- **Ground truth CSV** with one-hot encoded labels across 8 disease classes
- **Metadata CSV** with patient information: approximate age, sex, anatomical site of the lesion, and a lesion ID

The eight classes and their counts are shown below (see `eda.ipynb`, Cell 5 for the full output):

| Class | Disease | Count | Percentage |
|-------|---------|------:|:----------:|
| NV | Melanocytic Nevi | 12,875 | 50.83% |
| MEL | Melanoma | 4,522 | 17.85% |
| BCC | Basal Cell Carcinoma | 3,323 | 13.12% |
| BKL | Benign Keratosis | 2,624 | 10.36% |
| AK | Actinic Keratoses | 867 | 3.42% |
| SCC | Squamous Cell Carcinoma | 628 | 2.48% |
| VASC | Vascular Lesions | 253 | 1.00% |
| DF | Dermatofibroma | 239 | 0.94% |

### 2.2 Class Imbalance

The most significant finding from our EDA is the severe class imbalance. Melanocytic Nevi alone makes up over half the dataset, while Dermatofibroma and Vascular Lesions each account for less than 1%. The imbalance ratio between the largest and smallest class is roughly 54x. A bar chart and pie chart showing this distribution are in `eda.ipynb`, Cell 6.

### 2.3 Metadata Analysis

We examined patient metadata to understand the demographics and find patterns that might help classification (see `eda.ipynb`, Cells 8-9 for the full breakdown):

- **Age:** Ranges from 0 to 85 years with a median of 55. About 1.7% of age values are missing. The age distribution peaks in the 40-60 range. When broken down by class, BCC and AK patients tend to be older (consistent with cumulative UV exposure), while NV patients skew younger (see `eda.ipynb`, Cell 11).
- **Sex:** 13,286 male and 11,661 female patients, with 384 records (1.5%) missing sex. The sex breakdown per class (see `eda.ipynb`, Cell 12) shows DF skewing female and MEL/SCC skewing slightly male.
- **Anatomical site:** The most common sites are anterior torso (6,915), lower extremity (4,990), and head/neck (4,587). About 10.4% of records have no site recorded. A heatmap of site distribution by class (see `eda.ipynb`, Cell 13) shows clear patterns, for example VASC lesions appearing mostly on head/neck and lower extremity, while DF clusters on the lower extremity.

### 2.4 Image Characteristics

We sampled 500 images to check dimensions (see `eda.ipynb`, Cells 17-18):

- **Width:** 600 to 1024 pixels (mean 861)
- **Height:** 450 to 1024 pixels (mean 774)
- **Aspect ratio:** 1.00 to 1.51 (mean 1.17)

Images are not uniform in size, which makes resizing a required preprocessing step. We also computed mean RGB pixel intensity per class from a sample of 30 images each (see `eda.ipynb`, Cells 22-23). Vascular Lesions show the highest red channel intensity, which makes sense given their vascular nature. Melanoma tends to be darker overall.

### 2.5 File Integrity

All 25,331 images referenced in the CSV files exist on disk with no missing files (see `eda.ipynb`, Cell 15). No duplicates or corrupt files were found.

## 3. Preprocessing Completed So Far

All preprocessing work is documented and runnable in `preprocessing.ipynb`. Here is what we completed:

### 3.1 Metadata Cleaning and Imputation

We handled missing values as follows (see `preprocessing.ipynb`, Cell 5):

- **Age (1.7% missing):** Imputed using the per-class median age. This preserves the age distribution within each disease category rather than using a single global median.
- **Sex (1.5% missing):** Filled with the global mode (male).
- **Anatomical site (10.4% missing):** Assigned to an explicit "unknown" category rather than dropping or guessing, which treats missingness as a potentially informative signal.

After imputation, all three metadata columns have zero missing values.

### 3.2 Metadata Feature Encoding

We encoded metadata into numerical features for potential use as auxiliary model inputs (see `preprocessing.ipynb`, Cell 7):

- **age_norm:** Min-max scaled to [0, 1]
- **sex_enc:** Binary encoding (0 = female, 1 = male)
- **site_enc:** Integer-encoded across 9 categories (8 anatomical sites + "unknown")

The encoder mappings are saved to `data_splits/meta_encoder_info.json` so they can be reused at inference time.

### 3.3 Stratified Train/Validation/Test Split

We split the dataset into 70% training, 15% validation, and 15% test using stratified sampling on the class label (see `preprocessing.ipynb`, Cells 9-11):

| Split | Samples | Percentage |
|-------|--------:|:----------:|
| Train | 17,731 | 70.0% |
| Val | 3,800 | 15.0% |
| Test | 3,800 | 15.0% |

The class proportions are preserved almost exactly across all three splits. For example, NV is 50.83% in the full dataset, 50.83% in train, 50.84% in val, and 50.82% in test (see `preprocessing.ipynb`, Cell 10 for the full verification table).

### 3.4 Class Weight Computation

We computed inverse-frequency class weights from the training set only to avoid data leakage (see `preprocessing.ipynb`, Cells 13-14). The weights range from 0.25 for NV (most common) to 13.27 for DF (rarest). These weights are saved to `data_splits/class_weights.json` and can be loaded directly into `torch.nn.CrossEntropyLoss(weight=...)`.

### 3.5 Data Augmentation Pipelines

We designed two tiers of augmentation, both applied only during training (see `preprocessing.ipynb`, Cell 16):

**Standard augmentation** (for majority classes: MEL, NV, BCC, BKL):
- Resize to 256 then random crop to 224x224
- Random horizontal and vertical flip
- Random rotation (up to 30 degrees)
- Mild color jitter (brightness, contrast, saturation, hue)

**Aggressive augmentation** (for minority classes: AK, DF, VASC, SCC):
- All of the above, plus:
- Full 180-degree random rotation
- Stronger color jitter
- Random affine with shear and scale variation
- Gaussian blur to simulate focus variation
- Random erasing to simulate hair and ruler artifacts common in dermoscopy images

Validation and test sets only get a deterministic resize to 256 followed by center crop to 224x224 and ImageNet normalization. A visual comparison of original vs. standard vs. aggressive augmentation for the minority classes is shown in `preprocessing.ipynb`, Cell 21.

### 3.6 Image Normalization

All images are normalized using ImageNet mean and standard deviation values (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) since we plan to use pretrained transfer learning backbones. We verified that normalization is applied correctly by checking batch statistics (see `preprocessing.ipynb`, Cell 23). The standard deviation values are near 1.0 as expected. The mean is not perfectly centered at zero because dermoscopic skin images are naturally red-heavy compared to the general ImageNet distribution, but this is normal and expected for this domain.

### 3.7 PyTorch Dataset and DataLoader

We built a custom `ISICDataset` class that automatically selects the appropriate augmentation pipeline based on whether the sample belongs to a minority or majority class (see `preprocessing.ipynb`, Cell 18). It also supports returning metadata features alongside the image tensor for metadata-fused model architectures.

DataLoaders are configured with batch size 32, producing 554 training batches, 119 validation batches, and 119 test batches (see `preprocessing.ipynb`, Cell 19).

### 3.8 Output Files

All preprocessing artifacts are saved to the `data_splits/` directory (see `preprocessing.ipynb`, Cell 25):

| File | Purpose |
|------|---------|
| `train.csv`, `val.csv`, `test.csv` | Image IDs with labels and encoded metadata |
| `class_weights.json` | Inverse-frequency weights for the loss function |
| `class_map.json` | Mapping from integer index to class abbreviation and full name |
| `meta_encoder_info.json` | Encoder parameters for reproducing metadata encoding at inference |

## 4. Current Progress

| Task | Status |
|------|--------|
| Literature review of ISIC challenge approaches | Completed |
| Dataset acquisition and setup | Completed |
| Exploratory data analysis | Completed |
| Metadata cleaning and imputation | Completed |
| Stratified train/val/test split | Completed |
| Augmentation pipeline design | Completed |
| Class weight computation | Completed |
| PyTorch Dataset and DataLoader | Completed |
| Saved split CSVs and config files | Completed |
| Custom CNN baseline model | Not started |
| Transfer learning models (EfficientNet, ResNet, DenseNet) | Not started |
| Ensemble methods | Not started |
| Evaluation pipeline (metrics, confusion matrix, ROC) | Not started |
| Grad-CAM visualization | Not started |
| Web application for deployment | Not started |

In short, the data pipeline is fully complete. The dataset has been analyzed, cleaned, split, and packaged into ready-to-use PyTorch DataLoaders. The next phase focuses on model development and training.

## 5. Challenges Faced

**Severe class imbalance.** The 54x imbalance ratio between NV and DF is the biggest challenge for this project. A model that simply predicts NV for everything would achieve over 50% accuracy, so standard accuracy is a misleading metric. We are addressing this through inverse-frequency class weights, aggressive augmentation for minority classes, and plan to explore focal loss during training. We will rely on balanced accuracy and per-class metrics instead of overall accuracy.

**Missing metadata.** About 10.4% of anatomical site values are missing. Rather than dropping these samples (which would reduce an already limited dataset for minority classes), we created an "unknown" category. This was a design choice that trades a small amount of noise for keeping all training samples available.

**Image size variation.** Images range from 600x450 to 1024x1024 pixels. Resizing to a uniform 224x224 is required but loses some fine-grained detail, especially for smaller lesions. We decided on the 256 then crop to 224 approach to preserve more spatial information while matching standard input sizes for pretrained models.

**Multiprocessing in Jupyter on macOS.** We ran into a `pickle` error when using `num_workers > 0` with PyTorch DataLoaders in a Jupyter notebook on macOS, because the `spawn` method cannot serialize classes defined within notebooks. We resolved this by setting `num_workers=0` for the notebook environment, with a note to re-enable it in standalone training scripts.

## 6. Plan for Completion

### Remaining Tasks and Timeline

| Task | Owner | Target |
|------|-------|--------|
| Custom CNN baseline (build + train + evaluate) | James | Week 1 |
| Transfer learning: EfficientNet-B0 fine-tuning | James | Week 1-2 |
| Transfer learning: ResNet-50 fine-tuning | James | Week 2 |
| Transfer learning: DenseNet-121 fine-tuning | James | Week 2 |
| Focal loss and weighted cross-entropy comparison | James | Week 2 |
| Hyperparameter tuning and cross-validation | James | Week 3 |
| Ensemble methods (voting, weight averaging) | James | Week 3 |
| Evaluation pipeline (balanced accuracy, F1, confusion matrix, AUC-ROC) | James, Lam | Week 3 |
| Grad-CAM visualization | Vi | Week 3-4 |
| Web application UI and model integration | Vi | Week 3-4 |
| Final report and presentation | All | Week 4 |

### Evaluation Plan

Our primary metric is balanced accuracy (the official ISIC 2019 metric). We will also report weighted F1 score, per-class precision/recall, confusion matrices, and AUC-ROC curves. Our success targets are:

- 75% balanced accuracy overall
- 85%+ sensitivity for melanoma and SCC (since missing cancer is dangerous)
- 60%+ accuracy per class, including the rare ones

### Models and Methods

We plan to compare:
1. A custom CNN from scratch as a baseline
2. Fine-tuned EfficientNet-B0, ResNet-50, and DenseNet-121 with pretrained ImageNet weights
3. Ensemble combinations of the best-performing models

For handling class imbalance during training, we will compare weighted cross-entropy loss against focal loss. We will use learning rate scheduling (cosine annealing or step decay) and early stopping based on validation balanced accuracy.

## 7. Team Contributions

- **Lam Nguyen** - Responsible for the data pipeline. Completed dataset analysis, EDA (`eda.ipynb`), and the preprocessing plan (`EDA_Summary.md`). Identified key dataset characteristics including the imbalance ratios, metadata patterns, and image dimension statistics.

- **James Pham** - Responsible for model development. Built the preprocessing notebook (`preprocessing.ipynb`) including metadata encoding, stratified splitting, augmentation pipelines, class weight computation, and the PyTorch Dataset/DataLoader implementation. Will lead all model training and evaluation going forward.

- **Vi Thi Tuong Nguyen** - Responsible for deployment. Conducted background research on related ISIC challenge solutions and Grad-CAM visualization techniques. Will build the web application and integrate the trained models for real-time prediction.

---

## References

[1] World Health Organization, "Radiation: Ultraviolet (UV) radiation and skin cancer," WHO, 2023. Available: https://www.who.int/news-room/questions-and-answers/item/radiation-ultraviolet-(uv)-radiation-and-skin-cancer

[2] N. C. F. Codella et al., "Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)," arXiv:1902.03368, 2019.

[3] M. Combalia et al., "BCN20000: Dermoscopic Lesions in the Wild," arXiv:1908.02288, 2019.

[4] P. Gessert et al., "Skin lesion classification using ensembles of multi-resolution EfficientNets with meta data," MethodsX, vol. 7, p. 100864, 2020.
