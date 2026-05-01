# Training Experiments

**Project:** Skin Disease Classification System Using Deep Learning — A Multi-Class Approach
**Dataset:** ISIC 2019 Challenge (25,331 dermoscopy images, 8 classes)
**Team:** Lam Nguyen, James Pham, Vi Thi Tuong Nguyen

This is the running log of the four training runs we did, what we changed between them, and what we learned. The short version is: pretrained EfficientNet-B3 with focal loss at 300×300 was the winner, plus D4 test-time augmentation for an inference-time bump. The custom CNN baseline confirmed that without transfer learning, the rare classes are essentially unlearnable.

## Setup that's common across all runs

We stratify the dataset 80/10/10 with seed 42:

| Split | Samples |
|---|---:|
| Train | 20,264 |
| Val   | 2,534 |
| Test  | 2,534 |

The class distribution is the same lopsided one ISIC 2019 always has. Test support, for reference: NV 1,288 · MEL 452 · BCC 333 · BKL 263 · AKIEC 86 · SCC 63 · VASC 25 · DF 24. Largest-to-smallest ratio is about 54×.

Optimizer is Adam with weight decay 1e-5; LR scheduler is cosine annealing across the full epoch budget; batch size 32; max 50 epochs with early stopping on val loss. Augmentation during training is the usual flip/rotate/affine/color-jitter set with CoarseDropout (4–8 holes, 8–16px, p=0.3) added to simulate hair and ruler artifacts. Val and test are deterministic resize + ImageNet normalize.

Metrics we track: balanced accuracy, weighted F1, per-class precision/recall, and the confusion matrix.

## Run 1 — EfficientNet-B3, weighted CE, 224px

W&B name: `splendid-cloud-3`. ImageNet-pretrained EfficientNet-B3, weighted cross-entropy with inverse-frequency weights (DF=13.27 down to NV=0.25), images at 224×224, lr=1e-4, backbone frozen for 4 epochs then unfrozen. Trained on Colab Pro (T4); early stopped at epoch 18.

Results on the test set:

- Balanced accuracy **0.7708**
- Weighted F1 **0.7856**
- Plain accuracy 0.78

Per class:

| Class | P | R | F1 | Support |
|---|:-:|:-:|:-:|:-:|
| MEL   | 0.67 | 0.68 | 0.67 | 452 |
| NV    | 0.90 | 0.83 | 0.87 | 1,288 |
| BCC   | 0.84 | 0.79 | 0.81 | 333 |
| AKIEC | 0.50 | 0.76 | 0.60 | 86 |
| BKL   | 0.64 | 0.70 | 0.66 | 263 |
| DF    | 0.42 | 0.79 | 0.55 | 24 |
| VASC  | 0.82 | 0.92 | 0.87 | 25 |
| SCC   | 0.58 | 0.70 | 0.63 | 63 |

The confusion matrix made the failure modes obvious. NV and MEL get crossed up a lot — 120 NV predicted as MEL, 79 MEL predicted as NV — which is a known dermoscopy problem; even clinicians struggle there. BKL also leaks into MEL and NV. BCC gets some confusion with SCC and AKIEC.

The high-support classes (NV, BCC, VASC) were already in good shape. The pain points were DF, AKIEC, and SCC. DF is interesting: recall 0.79 but precision 0.42, meaning the loss weighting was overcorrecting in DF's direction — we were predicting it too eagerly.

## Run 2 — Focal loss + class weights, 300px

W&B name: `usual-salad-4`. Two changes from Run 1:

1. Image size 224 → **300** (EfficientNet-B3's native input).
2. Loss: weighted CE → **focal loss (γ=2.0)** *plus* the same class weights.

The thinking was that focal loss would focus training on hard examples and class weights would handle the imbalance, and stacking them would be better than either alone.

It wasn't. With D4 TTA at test time:

- Balanced accuracy **0.8215** (+0.05 vs Run 1)
- Weighted F1 **0.7102** (−0.08 vs Run 1)
- Plain accuracy 0.70

| Class | P | R | F1 | Δ vs Run 1 |
|---|:-:|:-:|:-:|:-:|
| MEL   | 0.45 | 0.88 | 0.60 | −0.07 |
| NV    | 0.97 | 0.56 | 0.71 | **−0.16** |
| BCC   | 0.87 | 0.86 | 0.86 | +0.05 |
| AKIEC | 0.66 | 0.72 | 0.69 | +0.09 |
| BKL   | 0.60 | 0.79 | 0.68 | +0.02 |
| DF    | 0.69 | 0.92 | 0.79 | **+0.24** |
| VASC  | 0.81 | 1.00 | 0.89 | +0.02 |
| SCC   | 0.66 | 0.84 | 0.74 | +0.11 |

Rare classes did jump — DF went from 0.55 to 0.79, SCC from 0.63 to 0.74 — but NV F1 collapsed from 0.87 to 0.71 because recall dropped to 0.56. Since NV is half the test set, that pulled weighted F1 down even though balanced accuracy went up.

In hindsight this is obvious: focal loss already down-weights easy majority examples, and the inverse-frequency class weights do basically the same thing. Applying both is double-counting, and the model ends up biased away from NV. That's exactly what the per-class numbers show.

## Run 3 — Focal loss only, 300px (best)

W&B name: `cardassian-spot-5`. Same as Run 2 but with class weights turned off — focal loss is now the only mechanism handling imbalance.

Without TTA: balanced accuracy 0.7594, weighted F1 0.8360.

With D4 TTA (8 variants from horizontal/vertical flips and 90° rotations, predictions averaged in softmax space):

- Balanced accuracy **0.7779**
- Weighted F1 **0.8502**
- Plain accuracy 0.85

| Class | P | R | F1 | Δ vs Run 1 |
|---|:-:|:-:|:-:|:-:|
| MEL   | 0.82 | 0.73 | 0.77 | +0.10 |
| NV    | 0.89 | 0.94 | 0.91 | +0.04 |
| BCC   | 0.89 | 0.90 | 0.90 | +0.09 |
| AKIEC | 0.60 | 0.69 | 0.64 | +0.04 |
| BKL   | 0.78 | 0.68 | 0.73 | +0.07 |
| DF    | 0.78 | 0.58 | 0.67 | +0.12 |
| VASC  | 0.92 | 0.96 | 0.94 | +0.07 |
| SCC   | 0.70 | 0.75 | 0.72 | +0.09 |

Every per-class F1 improved over Run 1. Compared to Run 2, NV recovered (0.91 vs 0.71) and we gave back some of the rare-class gains (DF 0.67 vs 0.79, SCC 0.72 vs 0.74), which is the trade-off we wanted — overall F1 is what we care about. AKIEC is still our weakest class at 0.64 and probably the next thing worth working on.

The TTA bump was small but consistent: about +1.9 points balanced accuracy and +1.4 points weighted F1, with no retraining needed. Worth keeping.

## Run 4 — Custom CNN baseline

W&B name: `comfy-frost-7`. Four-block CNN trained from scratch, ~500K parameters versus EfficientNet-B3's ~12M. Same focal loss, no class weights, 300×300 images, lr=1e-3 (10× higher because we're starting from random weights), 50 full epochs (early stopping never triggered). We trained this one locally on a MacBook Pro M1 Pro using MPS, since by this point we'd burned through our Colab GPU quota.

Architecture:

```
Block 1: Conv2d(3 → 32, 3x3) → BN → ReLU → MaxPool(2)
Block 2: Conv2d(32 → 64, 3x3) → BN → ReLU → MaxPool(2)
Block 3: Conv2d(64 → 128, 3x3) → BN → ReLU → MaxPool(2)
Block 4: Conv2d(128 → 256, 3x3) → BN → ReLU → MaxPool(2)
Head:    AdaptiveAvgPool2d(1) → Flatten
         Dropout(0.3) → Linear(256 → 128) → ReLU
         Dropout(0.3) → Linear(128 → 8)
```

Test results:

- Balanced accuracy **0.3796**
- Weighted F1 **0.6511**
- Plain accuracy 0.69

| Class | P | R | F1 |
|---|:-:|:-:|:-:|
| MEL   | 0.59 | 0.42 | 0.49 |
| NV    | 0.76 | 0.90 | 0.83 |
| BCC   | 0.55 | 0.83 | 0.66 |
| AKIEC | 0.56 | 0.17 | 0.27 |
| BKL   | 0.56 | 0.30 | 0.39 |
| DF    | 0.00 | 0.00 | **0.00** |
| VASC  | 1.00 | 0.56 | 0.72 |
| SCC   | 1.00 | 0.02 | **0.03** |

DF F1 is literally zero — the model never gets a single DF prediction right. SCC is only marginally better at 0.03 (high precision but the model basically refuses to predict it). The CNN does fine on NV because half the data is NV and even a weak model can latch onto the dominant class, but with only ~190 DF samples in the training set there's nothing to learn from scratch.

This isn't surprising in hindsight, but it's a useful sanity check. The whole reason transfer learning matters for medical imaging tasks is that the rare classes don't have enough samples on their own; you need representations learned from a much larger dataset (ImageNet) to bootstrap them. Run 4 is what happens when you don't have that.

## Cross-run summary

| Run | Model | Config | Bal. Acc | Weighted F1 | Acc |
|---|---|---|:-:|:-:|:-:|
| 1 | EffNet-B3 | Weighted CE, 224px | 0.7708 | 0.7856 | 0.78 |
| 2 | EffNet-B3 | Focal + CW, 300px, TTA | **0.8215** | 0.7102 | 0.70 |
| 3 | EffNet-B3 | Focal only, 300px, TTA | 0.7779 | **0.8502** | **0.85** |
| 4 | Custom CNN | Focal, 300px | 0.3796 | 0.6511 | 0.69 |

Per-class F1 across all four runs:

| Class | Support | Run 1 | Run 2 | Run 3 | Run 4 |
|---|:-:|:-:|:-:|:-:|:-:|
| MEL   | 452 | 0.67 | 0.60 | **0.77** | 0.49 |
| NV    | 1,288 | 0.87 | 0.71 | **0.91** | 0.83 |
| BCC   | 333 | 0.81 | 0.86 | **0.90** | 0.66 |
| AKIEC | 86 | 0.60 | **0.69** | 0.64 | 0.27 |
| BKL   | 263 | 0.66 | 0.68 | **0.73** | 0.39 |
| DF    | 24 | 0.55 | **0.79** | 0.67 | 0.00 |
| VASC  | 25 | 0.87 | 0.89 | **0.94** | 0.72 |
| SCC   | 63 | 0.63 | **0.74** | 0.72 | 0.03 |

Run 3 is best on 5 of 8 classes and tied for the rest of the most useful overall metrics. Run 2 wins the three rarest classes — if we ever cared specifically about DF/AKIEC/SCC recall (e.g., a screening application where missing rare disease is the worst outcome), Run 2's checkpoint would be the right one to use. For general-purpose classification, Run 3 wins.

A couple of design takeaways from comparing the runs side by side:

- 224 → 300 is worth the extra compute. About 2–3 points of F1 once the loss function is right.
- Focal loss alone beat weighted CE alone, and beat focal + weights.
- TTA is essentially free at inference time and gave +1–2 points consistently.
- Pretrained weights matter enormously for the rare classes (DF: 0.67 with transfer, 0.00 from scratch).

## Production model

Run 3 (`cardassian-spot-5`, EfficientNet-B3, focal loss, 300px, D4 TTA at inference) is what the web app loads. It has the best weighted F1 (0.85), the best plain accuracy (85%), every per-class F1 ≥ 0.64, and is strongest on the clinically high-risk classes — MEL at 0.77 and BCC at 0.90.

If we ever flip the use case to "maximize rare-class recall," Run 2's checkpoint (`usual-salad-4`) is preferred for DF/AKIEC/SCC specifically.

## Notes on compute

| Run | Where | Hardware | Wall time |
|---|---|---|---|
| 1 | Colab Pro | Tesla T4 | ~1 hour (early stopped) |
| 2 | Colab Pro | Tesla T4 | ~2 hours |
| 3 | Colab Pro | Tesla T4 | ~2 hours |
| 4 | MacBook Pro M1 Pro | MPS | ~3 hours (50 full epochs) |

Things we tripped over along the way:

- Colab quota ran out in the middle of the project, which is why Run 4 happened locally instead of on a T4.
- macOS multiprocessing didn't play nicely with MPS — `num_workers > 0` crashed the DataLoader. Setting it to 0 fixed it but slowed iteration.
- NumPy 2.x broke a couple of older packages (numexpr, bottleneck). We pinned NumPy below 2.0.
- We restructured the `outputs/` folder mid-project so each W&B run has its own subfolder. Before that we were silently overwriting checkpoints, which is how the legacy `outputs/checkpoints/best_model.pth` ended up orphaned.

## What's left

Things we know we want to try but haven't yet:

- Grad-CAM on Run 3 to confirm the model actually attends to the lesion and not to rulers, hairs, or skin-marker artifacts. The web app uses Grad-CAM but we haven't done a systematic audit.
- A proper ensemble of Run 2 (rare-class strength) and Run 3 (overall strength) using `WeightedEnsemble`. This is the obvious next step given how cleanly the two runs split on which classes they're best at.
- A second backbone (ResNet-50 or DenseNet-121) for comparison, ideally to feed into the ensemble.
- Mixup / CutMix to chip away at the persistent MEL ↔ NV confusion.

Things we know we *can't* claim:

- This model is trained only on dermoscopy images. It will not generalize to phone photos.
- All evaluation is on a 10% held-out slice of the same dataset. We have no external validation, so any real-world performance estimate is speculative.
- AKIEC at 0.64 is still our weakest class and there's a real chance the next round of work doesn't fix it — visual overlap with BCC and SCC is genuinely hard.

## Reproducing the runs

```bash
# Run 3 — the best model
python -m src.train --config configs/config.yaml
# config has loss=focal, focal_gamma=2.0, use_class_weights=false, image_size=300

# Run 4 — CNN baseline
python -m src.train --config configs/config_cnn.yaml
# config has backbone=custom-cnn, loss=focal, image_size=300; set num_workers=0 on macOS
```

Output artifacts land under `outputs/checkpoints/{backbone}_{wandb_run_name}/` and `outputs/results/{backbone}_{wandb_run_name}/` — see [outputs/README.md](../../outputs/README.md) for what's in each folder.
