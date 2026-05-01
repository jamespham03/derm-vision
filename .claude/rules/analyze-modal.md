# Analyze Modal + Feature Rows

Rules for the three enhancements to the `/analyze` route.

---

## 1. Glassmorphism Analyze Modal

### Trigger points
- Nav `LET'S ANALYZE` button (`#nav-analyze-btn`) — intercept click, prevent navigation, open modal
- Hero CTA button (`#hero-analyze-btn`) — opens modal
- Both via `openAnalyzeModal()` JS function

### Overlay
- `position: fixed; inset: 0; z-index: 300`
- Background: `rgba(4,4,4,0.88)` + `backdrop-filter: blur(28px) saturate(1.4)`
- Scroll inside if content overflows: `overflow-y: auto; padding: 32px 24px 80px`
- Clicking backdrop (not panel) closes modal

### Panel
- Max-width 780px, centered with `margin-top: 40px`
- Background: `rgba(14,14,14,0.92)` + `border: 0.5px solid rgba(255,255,255,0.1)` + `border-radius: 20px`
- Entrance animation: fade + translateY(12px)→0 + scale(0.98)→1, 220ms ease-out
- Close button: top-right, 36×36px circle, glassmorphism

### Inside modal (order)
1. Eyebrow: "LET'S ANALYZE" — Space Mono 9px uppercase
2. Title: "Upload your skin image." — Unbounded 700, clamp(28px,4vw,44px)
3. Step indicator (ids: step-1, step-connector, step-2)
4. `#upload-panel` — contains:
   a. Three source cards (replaces old source-btns)
   b. Upload section label
   c. Upload zone (ids preserved)
   d. Analyze CTA button
   e. Disclaimer
5. `#results-panel` — same content as before (ids preserved)

### Source cards
Three equal-width glassmorphism cards in a 3-column grid (gap: 10px):
- UPLOAD FILE / OPEN CAMERA / PASTE IMAGE
- Card: `background rgba(255,255,255,0.04)`, `border: 0.5px solid rgba(255,255,255,0.09)`, `border-radius: 14px`, `padding: 20px 18px`
- Card IDs: `btn-source-upload`, `btn-source-camera`, `btn-source-paste` (same IDs as before — analyze.js needs no changes)
- Card children: category label (Space Mono 9px, muted) → desc text → small arrow circle button
- Hover: slightly lighter background + border brightens + arrow translates right 2px

### Close behavior
- Click backdrop, click ×, or press Escape
- `document.body.style.overflow = "hidden"` when open, reset on close

---

## 2. Hero CTA Button

### Position
In `phantom-text-cols`, wrap the `h1.phantom-headline` in a new `div.phantom-headline-col`:
```
.phantom-headline-col { flex: 3; min-width: 0; display: flex; flex-direction: column; gap: 24px; align-items: flex-start; }
.phantom-headline-col .phantom-headline { flex: none; }
```

### Button style
```
.hero-analyze-btn
  display: inline-flex; align-items: center; gap: 8px
  padding: 11px 24px; border-radius: var(--r-full)
  background: rgba(240,237,230,0.07); border: 1px solid rgba(240,237,230,0.22)
  color: var(--text-primary); font: Space Grotesk 14px weight 500
  Hover: background 0.13 opacity, border 0.38 opacity
```

---

## 3. Feature Rows (Analyze Page Body)

Replaces the old upload/results panel area on the page. Three capability rows about topics never covered on the analyze page.

### Layout per row
CSS Grid: `24px 1fr 1fr 260px` with `gap: 48px; padding: 68px 0; border-top: 0.5px solid rgba(255,255,255,0.07)`

Columns:
1. Small dot (7px circle, 20% white)
2. Large title — Unbounded 700, `clamp(36px, 4.2vw, 66px)`, NOT uppercase (title case)
3. Body — focus label + 2 desc paragraphs + pill CTA link
4. Visual card — 1:1 aspect ratio, 260px, CSS-only visual

### Three rows

**Row 1 — Precision**
- Title: "Precision"
- Focus: "Our focus on — accuracy and confidence"
- Desc: Model achieves 85% weighted F1, confidence scores per-class shown, ISIC 2019 training
- CTA: "View model details →" → /about
- Visual: Mini bar chart showing per-class F1 scores (CSS bars, actual ISIC 2019 values)

**Row 2 — Explainability**
- Title: "Explainability"
- Focus: "Our focus on — transparent AI decisions"
- Desc: Grad-CAM heatmaps highlight regions driving the prediction, visual map not just a number
- CTA: "How it works →" → /about#how
- Visual: Warm radial gradient simulating a Grad-CAM heatmap (red center, orange mid, green outer)

**Row 3 — Privacy**
- Title: "Privacy"
- Focus: "Our focus on — data handling"
- Desc: Images processed in-session, never stored, nothing retained after page close
- CTA: "Read the disclaimer →" → #
- Visual: Dark card with centered SVG lock icon

### Responsive
- ≤1100px: hide visual column (`grid-template-columns: 20px 1fr 1fr`)
- ≤768px: single column, dot hidden, title `clamp(32px,8vw,52px)`

---

## File Changes

| File | Change |
|------|--------|
| `app/web/analyze.html` | Restructure hero (add headline-col + hero-btn), replace phantom-content with feature rows, add modal HTML |
| `app/web/css/styles.css` | Append new section: modal, source cards, hero btn, feature rows |
| `app/web/js/analyze.js` | No changes — all element IDs preserved |
| `app/web/analyze.html` `<script>` | Inline modal open/close JS |
