# EDA Summary & Data Preprocessing Plan
**Project:** Skin Disease Classification System Using Deep Learning: A Multi-Class Approach
**Dataset:** ISIC 2019 Challenge Training Dataset

---

## Part 1 — EDA Findings

### 1.1 Dataset Overview

- **Total labeled samples:** 25,331 dermoscopic images
- **Classes:** 8 disease categories (one-hot encoded in ground truth CSV)
- **Supplementary files:** `ISIC_2019_Training_GroundTruth.csv`, `ISIC_2019_Training_Metadata.csv`
- **Metadata fields:** `image`, `age_approx`, `anatom_site_general`, `lesion_id`, `sex`

---

### 1.2 Class Distribution

| Code | Disease | Count | % of Dataset | Suggested Weight |
|------|---------|------:|:------------:|:----------------:|
| NV   | Melanocytic Nevi | 12,875 | 50.8% | ~0.15 |
| MEL  | Melanoma | 4,522 | 17.9% | ~0.44 |
| BCC  | Basal Cell Carcinoma | 3,323 | 13.1% | ~0.60 |
| BKL  | Benign Keratosis | 2,624 | 10.4% | ~0.76 |
| AK   | Actinic Keratoses | 867 | 3.4% | ~2.29 |
| SCC  | Squamous Cell Carcinoma | 628 | 2.5% | ~3.17 |
| VASC | Vascular Lesions | 253 | 1.0% | ~7.87 |
| DF   | Dermatofibroma | 239 | 0.9% | ~8.33 |

**Key finding:** The dataset has severe class imbalance. NV alone accounts for ~51% of all images, while DF and VASC together account for less than 2%. The max/min imbalance ratio is approximately **54x** (NV vs. DF). This will heavily bias a naive model toward predicting NV.

---

### 1.3 Patient Metadata

**Age:**
- Range: ~5–85 years, median approximately 50
- Distribution is roughly bell-shaped peaking in the 40–60 age range
- Older patients tend to appear more frequently in BCC, AK, and SCC — all associated with cumulative UV exposure
- Younger patients are more represented in NV and MEL

**Sex:**
- Dataset is roughly balanced between male and female patients (slight male majority)
- DF shows a noticeable female skew
- MEL and SCC are slightly more male-skewed

**Anatomical Site:**
- Most common: posterior torso, anterior torso, upper extremity, lower extremity
- VASC lesions appear predominantly on the head/neck and lower extremity
- DF clusters heavily on the lower extremity
- MEL is spread broadly across the torso and extremities

**Missing values:** Some percentage of records are missing age, sex, or anatomical site. Metadata is an optional but potentially useful auxiliary signal.

---

### 1.4 Image Characteristics

**Dimensions:**
- Images are not uniform in size — they come from multiple dermatology centers with different equipment
- Widths and heights vary significantly across the dataset
- Aspect ratios cluster near 1.0 (roughly square) but with noticeable variance
- Some images are marked `_downsampled`, indicating lower resolution versions exist alongside originals

**Pixel Intensity (RGB):**
- Mean pixel intensity is broadly similar across classes (~0.55–0.70 range normalized 0–1)
- Skin-tone background gives all classes a warm red/green dominance over blue
- VASC lesions show noticeably higher red-channel intensity (vascular = reddish lesions)
- MEL tends toward darker overall intensity reflecting darker pigmentation
- NV and BKL show higher brightness indicating lighter-toned lesions on average

---

### 1.5 Key Takeaways

| Finding | Implication |
|---------|-------------|
| 51% NV, ~54x imbalance ratio | Must use weighted loss or focal loss; naive accuracy is meaningless |
| DF (239) and VASC (253) are very small minority classes | Aggressive augmentation and/or oversampling essential for these classes |
| Images are variable in size and resolution | Resizing to uniform dimensions is mandatory before training |
| RGB intensity varies by class | Per-channel normalization required; ImageNet statistics suitable for transfer learning |
| Age correlates with certain disease types | Age is a useful auxiliary feature, especially for BCC, AK, SCC |
| Anatomical site has class-specific patterns | Site can serve as a useful categorical auxiliary feature |
| ~few percent of metadata is missing | Imputation or masking strategy needed for metadata fields |
| Stratified split is critical | Random splits risk underrepresenting minority classes in val/test |

---

## Part 2 — Data Preprocessing Steps

### Step 1: Data Loading & Label Preparation

- Load `ISIC_2019_Training_GroundTruth.csv` and drop the `UNK` column (samples with no confirmed label)
- Convert one-hot encoded columns to a single integer class label for use with standard loss functions
- Load `ISIC_2019_Training_Metadata.csv` and merge on the `image` key
- Verify that every CSV entry has a corresponding `.jpg` file on disk; log and exclude any missing files

### Step 2: Stratified Train / Validation / Test Split

- Split the dataset **70% train / 15% validation / 15% test** using stratified sampling on the class label
- Stratification ensures all 8 classes — especially DF (239) and VASC (253) — are proportionally represented in every split
- Fix a random seed for reproducibility
- Do not apply any augmentation to the validation or test sets

### Step 3: Image Resizing

- Resize all images to a uniform spatial resolution
  - **224×224** for EfficientNet, ResNet, DenseNet (ImageNet standard)
  - Use `PIL` or `torchvision.transforms.Resize` with `BILINEAR` interpolation
- Maintain aspect ratio by center-cropping after resize if needed, or use direct resize

### Step 4: Pixel Normalization

- Convert pixel values from `[0, 255]` integer to `[0.0, 1.0]` float
- Apply **ImageNet normalization** for transfer learning models:
  - Mean: `[0.485, 0.456, 0.406]`
  - Std: `[0.229, 0.224, 0.225]`
- This aligns input distribution with what EfficientNet/ResNet/DenseNet were pretrained on

### Step 5: Data Augmentation (Training Set Only)

Apply the following transforms to the **training set only** to increase effective dataset size and reduce overfitting, with heavier augmentation for minority classes:

| Transform | Purpose |
|-----------|---------|
| Random horizontal flip (p=0.5) | Lesions have no inherent left/right orientation |
| Random vertical flip (p=0.5) | Dermoscopic images have no fixed orientation |
| Random rotation (±30°) | Rotation invariance for skin lesion appearance |
| Color jitter (brightness, contrast, saturation, hue) | Simulate different lighting and device variation |
| Random zoom / random resized crop | Scale invariance |
| Gaussian blur (p=0.2) | Simulate focus variation across imaging devices |
| Elastic distortion (optional) | Simulate natural skin texture deformation |

For the most extreme minority classes (DF, VASC, SCC, AK), consider **oversampling** these classes in the DataLoader using `WeightedRandomSampler` to ensure each mini-batch sees a balanced representation.

### Step 6: Handling Class Imbalance

Two complementary strategies:

**A. Weighted Loss Function**
Compute inverse-frequency class weights and pass to `nn.CrossEntropyLoss(weight=...)`:

```python
# Example (actual values computed from training split counts)
total = sum(class_counts.values())
class_weights = [total / (8 * class_counts[c]) for c in CLASSES]
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
```

Approximate weights based on full dataset:

| Class | Weight |
|-------|--------|
| NV | ~0.15 |
| MEL | ~0.44 |
| BCC | ~0.60 |
| BKL | ~0.76 |
| AK | ~2.29 |
| SCC | ~3.17 |
| VASC | ~7.87 |
| DF | ~8.33 |

**B. Focal Loss (alternative)**
Use focal loss to further down-weight easy majority-class examples and focus training on hard minority samples:

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

Recommended starting values: `gamma=2`, `alpha` set to inverse class frequencies.

### Step 7: Metadata Handling (Optional Auxiliary Features)

- **Age:** Impute missing values with the training set median; normalize to `[0, 1]`
- **Sex:** One-hot encode (`male`, `female`, `unknown`)
- **Anatomical site:** One-hot encode the 6 site categories; use an `unknown` category for missing values
- Concatenate these features with the CNN's flattened representation before the final classification head

### Step 8: DataLoader Setup

```
TrainLoader  → augmented images + class-weighted sampler, batch_size=32, shuffle=True
ValLoader    → no augmentation, batch_size=64, shuffle=False
TestLoader   → no augmentation, batch_size=64, shuffle=False
```

- Use `num_workers=4` (or more depending on CPU) for parallel data loading
- Pin memory (`pin_memory=True`) if training on GPU

### Step 9: Sanity Checks Before Training

- [ ] Verify class distribution in each split matches expected proportions
- [ ] Confirm no image IDs overlap between train, val, and test
- [ ] Visualize a batch of augmented training images to confirm transforms look correct
- [ ] Check that mean/std of normalized pixel values are near zero-mean unit-variance
- [ ] Confirm class weight tensor sums to number of classes (sanity check for weight computation)
- [ ] Log total sample counts per class per split

---

## Summary Checklist

```
[ ] Drop UNK rows and convert one-hot labels to integers
[ ] Merge ground truth with metadata CSV
[ ] Stratified 70/15/15 train/val/test split (fixed seed)
[ ] Resize all images to 224×224
[ ] Normalize with ImageNet mean/std
[ ] Apply augmentation to train set only
[ ] Set up WeightedRandomSampler or compute class weights for loss
[ ] Impute and encode metadata features (age, sex, site)
[ ] Build DataLoaders with appropriate batch sizes
[ ] Run sanity checks before kicking off training
```
