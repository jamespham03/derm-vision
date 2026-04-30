# Analysis Page

Rules for the `/analyze` route. Inspired by **resend.com** — minimal, precision-crafted, developer-aesthetic. Clean dark UI where the tool is the focus. Every element earns its place.

## Visual Identity

The analysis page is a tool, not a showcase. It should feel focused, calm, and professional.

- Background: `var(--bg-base)` — same near-black as homepage, consistent
- Content is centered, single-column, no distractions
- Typography is large and confident but not showy
- Whitespace does the heavy lifting — do not crowd elements
- The model output section should feel like a premium report

## Page Sections (in order)

```
1. Page hero (heading + subtext)
2. Step indicator
3. Image input zone
4. Source selection buttons
5. Analyze CTA
6. Disclaimer
7. Results section (conditionally visible)
8. Footer
```

## 1. Page Hero

Not a full-height hero — more like a page title block.

```
Padding:    --space-16 0 --space-12
Eyebrow:    "AI-powered analysis" badge (see components.md badge style)
Heading:    DM Serif Display, --text-4xl desktop / --text-3xl mobile
            Example: "Analyze your skin image."
Sub:        DM Sans --text-base, --text-secondary, max-width 480px
            Example: "Upload a dermoscopy photo. Our EfficientNet-B3 model will
                      classify it across 8 lesion types in under a second."
```

No background image, no hero illustration — just text. The resend.com "Email for developers" section is the reference: confident headline, minimal decoration.

## 2. Step Indicator

A horizontal progress track showing the two-step flow. Appears below the hero.

```
Step 1: "Upload image"    — active state (circle filled with --text-primary)
  ——— connector line ———
Step 2: "AI results"      — inactive (circle in --text-muted)
```

When results are shown, Step 2 activates. Uses `--accent` fill for completed steps.

```
Font:     DM Sans --text-sm, weight 500
Layout:   flex, align-items center, gap --space-3
Circle:   22×22px, border-radius 50%
Connector:width 60px, height 0.5px, background var(--bg-border)
```

## 3. Image Input Zone

The upload dropzone is the primary interaction. See `components.md` for base styles.

Specific rules for this page:

- Section label above the zone: "Skin Image" — `--text-xs`, uppercase, weight 500, `--text-muted`, with flex divider line
- Dropzone min-height: 260px
- Below dropzone: `--text-xs`, `--text-muted`, centered: "Supports JPEG, PNG · Max 10MB · Dermoscopy images yield best results"
- When an image is loaded, the dropzone transitions to show the image thumbnail with a subtle overlay and "Change image" action

## 4. Source Selection Buttons

See `components.md` for full button spec. Page-specific rules:

- Always shown below the dropzone (not inside it)
- Three buttons: "Upload file", "Open camera", "Paste image"
- Use native Gradio source-selection buttons styled via CSS — do not use custom JS-wired HTML buttons
- On mobile: stack to single row, allow overflow-x scroll if needed

## 5. Analyze CTA

```
Width:      100% (full width of content column)
Height:     50px
Style:      Primary lg button (see components.md)
Label:      "Analyze with AI →"
Disabled:   when no image is uploaded — opacity 0.4, cursor not-allowed
```

Subtle loading state when clicked: spinner replaces "→" arrow, label becomes "Analyzing…"

## 6. Medical Disclaimer

Small, unobtrusive. Below the analyze button.

```
Layout:     flex, gap 8px, align-items flex-start
Icon:       warning triangle SVG, 14×14px, --text-muted
Text:       DM Sans --text-xs, --text-muted, line-height 1.6
Content:    "This tool is for educational and screening purposes only — not medical
             advice, diagnosis, or treatment. Always consult a licensed dermatologist."
```

Do not use a colored background for the disclaimer on this page — keep it invisible until the user reads carefully. Contrast with homepage CTA where it's more prominent.

## 7. Results Section

Appears below the input section after analysis. The style should feel like a premium generated report.

### Diagnosis Card

The primary output. Feels like a clinical report card.

```
Layout:     border card (see components.md result cards)
Max-width:  720px, centered
```

Internal sections:
1. **Header** — "Primary Diagnosis" label (--text-xs uppercase --text-muted) + condition name in DM Serif Display --text-2xl + confidence + risk badge
2. **Description** — one-paragraph clinical description, --text-sm --text-secondary
3. **Advice row** — icon + advice text, background tinted by risk color (subtle, not garish)

### Classification Breakdown

All 8 classes with probability bars. Resend.com reference: think of their feature comparison table — clean, monochrome lines with just enough color accent on the primary row.

```
Section heading: "Classification breakdown" — --text-xs uppercase --text-muted, with divider line
Bar rows:        alternating, top result highlighted
Width:           same max-width as diagnosis card
```

### "Start New Analysis" Button

```
Style:    Secondary md button
Label:    "← Start new analysis"
Position: Below the results section
```

## Transition: Upload → Results

When analysis completes:

1. Input section (upload zone, buttons, analyze CTA) fades out and slides up (200ms)
2. Results section fades in and slides up from below (350ms, 100ms delay)
3. Step indicator updates: Step 2 activates with accent color

Do not use a full page reload or navigation — animate in-place. This is the core UX moment.

## Resend.com Reference Principles

These specific patterns from resend.com should inform this page:

- **Large confident headline** with minimal decoration — the heading alone conveys authority
- **Feature rows** that combine a small label, a large description, and a supporting visual
- **Code-like precision** in the results layout — tight spacing, monospace numbers, grid-aligned columns
- **Dark surfaces** with barely-visible borders separating regions — `0.5px solid #1f1f1f`
- **Generous section spacing** — each region breathes, never feels cramped
- **Single accent color** used sparingly — only for the most important action or status
