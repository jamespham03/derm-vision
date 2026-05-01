# EDA + Preprocessing Plan

**Project:** Skin Disease Classification System Using Deep Learning — A Multi-Class Approach
**Dataset:** ISIC 2019 Challenge (training set)

This is a working document — what we found while exploring the dataset, and the preprocessing plan we wrote before training started. Some of the numbers below are approximate (we didn't always carry every decimal through), and the final preprocessing in code may differ slightly from this plan; treat the code as authoritative when in doubt.

## Part 1 — What's in the data

### Headline numbers

- 25,331 dermoscopic images, JPEG, all labeled
- 8 disease classes, one-hot encoded in `ISIC_2019_Training_GroundTruth.csv`
- A separate metadata CSV with `image`, `age_approx`, `anatom_site_general`, `lesion_id`, `sex`

### Class distribution

The first thing we noticed is how lopsided this dataset is. NV alone is more than half of it; DF and VASC together are under 2%.

| Code | Disease | Count | % | Suggested inverse-freq weight |
|------|---------|------:|:-:|:----:|
| NV   | Melanocytic nevus | 12,875 | 50.8% | ~0.15 |
| MEL  | Melanoma | 4,522 | 17.9% | ~0.44 |
| BCC  | Basal cell carcinoma | 3,323 | 13.1% | ~0.60 |
| BKL  | Benign keratosis | 2,624 | 10.4% | ~0.76 |
| AK   | Actinic keratosis | 867 | 3.4% | ~2.29 |
| SCC  | Squamous cell carcinoma | 628 | 2.5% | ~3.17 |
| VASC | Vascular lesion | 253 | 1.0% | ~7.87 |
| DF   | Dermatofibroma | 239 | 0.9% | ~8.33 |

The largest-to-smallest ratio is roughly 54×. A model that always predicts NV would already have over 50% accuracy, so plain accuracy is essentially useless here — we need balanced accuracy and per-class metrics.

### Patient metadata

**Age.** Roughly 5–85 years, median around 50. The histogram peaks in the 40s and 60s. Older patients show up more often in BCC, AK, and SCC, which lines up with cumulative UV exposure being a known risk factor. NV and MEL skew younger.

**Sex.** Roughly balanced, with a slight male majority. DF has a noticeable female skew. MEL and SCC tilt slightly male.

**Anatomical site.** Most lesions are on the posterior or anterior torso, or the upper/lower extremities. Two patterns stood out: VASC clusters on head/neck and lower extremity, and DF clusters heavily on the lower extremity.

A few percent of records are missing one or more of age, sex, or site. We don't want to drop them — for the smallest classes that would mean losing a meaningful chunk of training samples — so we'll need an imputation or "unknown" strategy.

### Image characteristics

The images come from several institutions and the equipment varies, so the dimensions aren't uniform. Widths and heights range across a few hundred pixels and aspect ratios cluster near 1.0 with some spread. Some files are marked `_downsampled`, meaning lower-resolution versions that exist alongside the originals. Resizing to a fixed input size is mandatory.

Per-class RGB statistics also vary in expected ways: VASC has noticeably higher red (vascular lesions are reddish), MEL trends darker, NV and BKL are brighter on average. None of this is dramatic, but it confirms there's signal in color.

### What this implies for training

A few things follow from the above:

- We can't rely on accuracy. We'll track balanced accuracy and per-class F1.
- DF and VASC are tiny. Aggressive augmentation and possibly oversampling will be needed for them.
- Resize is non-negotiable. We'll match whatever resolution our backbone expects (224 or 300).
- ImageNet normalization is appropriate since we're going to use a pretrained backbone.
- Splits must be stratified by class — random splits could easily under-represent DF or VASC in val/test.
- Age and site look like they'd be useful auxiliary features; we'll plumb them through but the first model won't depend on them.

## Part 2 — Preprocessing plan

This was the plan we wrote before training. The actual implementation in `src/dataset.py` and `src/transforms.py` follows it but tweaked a few things (300×300 instead of 224×224, an 80/10/10 split instead of 70/15/15) once we settled on EfficientNet-B3 as the primary backbone.

### Step 1 — Labels

Drop the `UNK` column from the ground truth CSV (samples without a confirmed label). Convert the remaining one-hot columns into a single integer class index for use with `nn.CrossEntropyLoss`. Merge the metadata CSV on the `image` key. Walk the disk once to confirm every CSV row has a corresponding `.jpg`; any missing files get logged and excluded.

### Step 2 — Stratified split

Original plan was 70/15/15; we ended up using 80/10/10 because we wanted a larger training set and we already had W&B for picking the best epoch. Either way the split is stratified on the class label, with a fixed seed. No augmentation on val or test.

### Step 3 — Resize

Resize every image to a fixed square. The plan was 224×224 (the ImageNet standard), but EfficientNet-B3's native input is 300, so we moved to 300×300 once the backbone was picked. PIL or `torchvision.transforms.Resize` with bilinear interpolation, no fancy aspect-ratio preservation.

### Step 4 — Normalize

Float32 in [0, 1], then ImageNet mean/std:

```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

This matches what EfficientNet was pretrained on. Skin images are red-heavy compared to ImageNet, so the post-normalization mean isn't perfectly zero, but that's fine.

### Step 5 — Augmentation (training only)

We'll use Albumentations. Heavier augmentation for minority classes is worth trying.

- Random horizontal flip, p=0.5
- Random vertical flip, p=0.5
- Random rotation up to ±30°
- Color jitter (brightness, contrast, saturation, hue)
- Random zoom / random resized crop
- Gaussian blur, p=0.2 (simulates focus variation across devices)
- Optional elastic distortion for very small classes

On top of the per-image transforms, a `WeightedRandomSampler` keyed off inverse class frequency would help DF/VASC/SCC/AK appear more often per epoch. This is a parameter we'll experiment with.

### Step 6 — Class imbalance

Two complementary tools, and we expect to try both:

**Weighted cross-entropy.** Inverse-frequency weights passed to `nn.CrossEntropyLoss(weight=...)`:

```python
total = sum(class_counts.values())
class_weights = [total / (8 * class_counts[c]) for c in CLASSES]
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
```

**Focal loss.** `FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)`. Starting values: γ=2, α set to inverse class frequencies. Down-weights easy majority examples.

A note we eventually had to internalize: focal loss and class weights both correct for imbalance, and stacking both is too aggressive — we found that out the hard way during Run 2.

### Step 7 — Metadata (optional auxiliary features)

If we end up wanting these:

- Age: impute missing values with the train-set median, scale to [0, 1].
- Sex: one-hot (`male`, `female`, `unknown`).
- Site: one-hot across the 6 site categories plus `unknown`.

These would concatenate into the CNN's flattened feature vector before the classification head. We didn't end up using them in the final model — the image alone was sufficient.

### Step 8 — DataLoaders

```
train: augmented + WeightedRandomSampler, batch 32, shuffle=True
val:   no aug, batch 64, shuffle=False
test:  no aug, batch 64, shuffle=False
```

`num_workers=4` on Linux/Colab. On macOS we have to use `num_workers=0` because PyTorch multiprocessing on MPS spawns broken workers. `pin_memory=True` on GPU.

### Step 9 — Sanity checks before training

Before any training run we verify:

- class proportions match across the splits
- no image IDs overlap between train/val/test
- a batch of augmented training images looks sane when visualized
- normalized batch statistics are roughly zero-mean unit-variance
- the class-weight tensor sums to the number of classes
- per-class counts in each split are logged

## Quick checklist

- [ ] Drop UNK rows, convert one-hot to integer labels
- [ ] Merge ground-truth and metadata CSVs
- [ ] Stratified split with a fixed seed
- [ ] Resize images to the backbone's native input size
- [ ] ImageNet normalization
- [ ] Train-only augmentation
- [ ] Class weights and/or focal loss
- [ ] Encode metadata if we want to use it
- [ ] Build DataLoaders with the right `num_workers` for the platform
- [ ] Run sanity checks before kicking off training
