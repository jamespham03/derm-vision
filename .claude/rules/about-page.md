# About Page

Rules for the `/about` route. This page contains all project information — everything that was on the old homepage hero, stats, how-it-works, and class grid. Inspired by **resend.com** — clean, editorial, generous whitespace.

## Purpose

The About page is the project explainer. Users arrive here from the "About" CTA on the homepage. It answers:
- What is DermVision?
- How accurate is it?
- How does it work?
- What conditions does it detect?
- Who built it?

## Page Sections (in order)

```
1. Nav (shared component)
2. Hero
3. Stats strip
4. How it works
5. What the model can detect (8 classes grid)
6. Team
7. CTA → Analyze
8. Footer
```

## 1. Nav

Same nav component as the analyze page: semi-transparent, logo left, links right.
Links: Home | Analyze | (active: About — no link, just current page indicator)

## 2. Hero

```
Padding:    80px 0 60px
Badge:      "CMPE 258 · Deep Learning · SJSU"
Heading:    DM Serif Display, clamp(42px, 6.5vw, 82px)
            "Skin cancer,\nseen by AI."
            em tag for "seen by AI." → var(--accent), italic
Subtext:    DM Sans 16px, --text-secondary, max-width 520px, centered
            "EfficientNet-B3 trained on 25,331 dermoscopy images from the ISIC 2019
             dataset. Eight diagnostic classes. Instant AI-powered analysis."
CTA:        Primary btn lg "Analyze your skin image →" linking to /analyze
            Secondary btn lg "How it works" linking to #how
```

## 3. Stats Strip

Horizontal row of 4 stats, divider-separated.

```
25,331          8              85%               EfficientNet-B3
Training        Lesion         Weighted          Architecture
images          classes        F1 score
```

Style: `--text-4xl` numbers in DM Serif Display, `--text-sm` labels in `--text-muted`.
Animated counters: numbers count up when strip scrolls into view.
Border-top and border-bottom: `var(--border)`. No background.

## 4. How It Works

Section label "How it works" + badge.
H2: "From photo to diagnosis in seconds."
3-column card grid (auto-fit, minmax 240px):

```
Step 01 — Upload your image
Icon: upload arrow SVG
Desc: "Take or upload a clear close-up photo of your skin lesion. Dermoscopy images yield the highest accuracy."

Step 02 — AI analyzes the lesion
Icon: checkmark circle SVG
Desc: "EfficientNet-B3, pre-trained on ImageNet and fine-tuned on ISIC 2019, processes your image in under a second."

Step 03 — Review your report
Icon: document SVG
Desc: "Get an instant classification report with confidence scores across all 8 lesion types and a risk assessment."
```

Cards: `var(--bg-surface)` background, `var(--border)` border, separated by 1px gap on a `var(--bg-border)` background.

## 5. What the Model Can Detect

Section: "8 Lesion Classes" badge + "What the model can detect" H2.

8-card grid (auto-fill, minmax 200px, gap 8px):

| Code  | Condition                    | Risk     | Color      |
|-------|------------------------------|----------|------------|
| MEL   | Melanoma                     | high     | #e24b4a    |
| NV    | Melanocytic Nevus             | low      | #1d9e75    |
| BCC   | Basal Cell Carcinoma         | high     | #e24b4a    |
| AKIEC | Actinic Keratosis            | moderate | #ef9f27    |
| BKL   | Benign Keratosis             | low      | #1d9e75    |
| DF    | Dermatofibroma               | low      | #1d9e75    |
| VASC  | Vascular Lesion              | low      | #1d9e75    |
| SCC   | Squamous Cell Carcinoma      | high     | #e24b4a    |

Each card: code label (uppercase muted), risk badge (color-tinted pill), condition name. Hover: translateY(-3px).

## 6. Team

Section: "SJSU CMPE 258" badge + "The team" H2.

3-card row:
```
Lam Nguyen        James Pham        Vi Thi Tuong Nguyen
SJSU CMPE 258     SJSU CMPE 258     SJSU CMPE 258
```

Simple cards: avatar initial circle (accent background), name in DM Serif Display, course label in --text-muted.

## 7. CTA

```
Background:  var(--bg-surface)
Border:      border-top + border-bottom var(--border)
Padding:     80px 40px
Text-align:  center
Heading:     DM Serif Display "Ready to analyze your skin?"
             em: italic, accent color
Sub:         "Upload a photo and get an AI-powered classification in seconds."
Button:      "Try it now →" primary lg → /analyze
Disclaimer:  "For educational and research purposes only. Not a substitute for professional medical advice."
```

## 8. Footer

Same footer component as analyze page.

## Scroll Animations

All sections use `.section-reveal` entrance animation (slide up + fade in) on scroll via IntersectionObserver.

## Navigation Flow

- "Analyze your skin image →" / "Try it now →" → `/analyze`
- "How it works" anchor → `#how` on this same page
- Nav logo → `/`
