# Components

Specifications for every reusable UI component. Build these once, use them everywhere. Do not create one-off variants — extend these instead.

## Navigation Bar (shared across ALL pages)

The same top nav bar is used on every page — homepage, analysis, and about. CSS class: `.site-nav`.

```
Background:   rgba(8,8,8,0.88) + backdrop-filter: blur(20px)
              (homepage: transparent — the Three.js canvas is beneath)
              (about/analyze: this semi-transparent dark bg)
Border:       border-bottom: 0.5px solid rgba(255,255,255,0.06)
Padding:      18px 28px
Layout:       flex, space-between, align-items center

Left — Logo (.site-logo-wrap, links to /):
  Circle:     36×36px, border: 1.5px solid rgba(240,237,230,0.7), border-radius 50%
              "D" in Unbounded italic 14px bold, color #f0ede6
  Wordmark:   "DermVision" in Unbounded 13px weight 400
  Badge:      "BETA" in Space Mono 8px, faint border pill

Center — Tagline (.site-tagline):
  Position:   fixed, left 50%, top 0, height 72px, centered vertically
  Font:       Space Mono 10px uppercase, color rgba(240,237,230,0.65)
  Max-width:  300px
  Content:    "DermVision is a deep learning system classifying skin lesions across
               eight diagnostic categories using EfficientNet‑B3."

Right — CTA (.site-cta-btn, links to /analyze):
  Label:      "LET'S ANALYZE"
  Style:      transparent bg, 1px solid rgba(240,237,230,0.35), border-radius 100px
              Space Mono 11px, letter-spacing 0.08em
  Hover:      background #f0ede6, color #080808
```

The homepage nav uses the same HTML structure but with `pointer-events: none` on the nav
element and `pointer-events: all` on children (so the Three.js canvas underneath is interactive).

On mobile: hide the tagline. Keep logo + CTA button only.

## Buttons

### Primary
```
Background:   var(--accent)
Hover:        var(--accent-hover)
Text:         var(--white), DM Sans --text-sm, weight 500
Padding:      10px 20px
Border-radius: var(--radius-md)
Border:       none
Transition:   background 150ms ease
```

### Secondary
```
Background:   transparent
Border:       var(--border-bright)
Text:         var(--text-primary), DM Sans --text-sm, weight 400
Padding:      10px 20px
Border-radius: var(--radius-md)
Hover:        background var(--bg-elevated), border rgba(255,255,255,0.15)
```

### Ghost (icon buttons, nav actions)
```
Background:   transparent
Border:       none
Text:         var(--text-secondary)
Hover:        var(--text-primary)
Padding:      8px
Border-radius: var(--radius-sm)
```

### Size modifiers
```
sm:  padding 7px 14px,  font-size --text-xs
md:  padding 10px 20px, font-size --text-sm  (default)
lg:  padding 14px 28px, font-size --text-base, height 50px
```

Never use colored text on colored backgrounds (e.g., green text on green button).

## Cards (Homepage Grid)

```
Background:      var(--bg-surface)
Border:          var(--border)
Border-radius:   var(--radius-lg)
Overflow:        hidden
Aspect-ratio:    varies (see homepage.md) — 4:5, 1:1, 16:9
```

Card image fills the card completely — `object-fit: cover`, `width: 100%`, `height: 100%`.

Card overlay (label + meta, shown on hover):
```
Position:   absolute, inset 0
Background: linear-gradient(to top, rgba(5,5,5,0.85) 0%, transparent 50%)
Padding:    16px
```

Card label tag:
```
Font:             DM Sans --text-xs, weight 500, uppercase, letter-spacing 0.08em
Background:       rgba(255,255,255,0.1)
Border:           0.5px solid rgba(255,255,255,0.12)
Border-radius:    var(--radius-full)
Padding:          3px 10px
Color:            var(--text-primary)
Backdrop-filter:  blur(8px)
```

See `interactions.md` for hover animation specs.

## Image Upload Zone

```
Border:           2px dashed rgba(255,255,255,0.1)
Border-radius:    var(--radius-xl)
Background:       var(--bg-surface)
Min-height:       260px
Transition:       border-color 150ms, background 150ms

Hover / drag-over:
  border-color:   var(--accent)
  background:     rgba(15,110,86,0.05)
  box-shadow:     var(--shadow-glow)
```

Upload zone center content:
```
Icon:     SVG, 40×40px, stroke var(--text-muted)
Heading:  "Drop your image here" — DM Sans --text-base, --text-secondary
Sub:      "or click to browse" — DM Sans --text-sm, --text-muted
```

## Source Selection Buttons (Upload / Camera / Clipboard)

Displayed below the upload zone as a row of 3 pill buttons.

```
Layout:       flex, gap 12px, justify-content center, padding 16px 0 20px
Button style: inline-flex, align-items center, gap 8px
              padding 10px 20px, border-radius var(--radius-md)
              border: var(--border-bright)
              background: var(--bg-surface)
              color: var(--text-secondary), font DM Sans --text-sm, weight 500
              cursor pointer
              box-shadow: var(--shadow-card)
              transition: 120ms ease

Hover:        background var(--bg-elevated), border rgba(255,255,255,0.15), color var(--text-primary)
Active/selected: background var(--accent), border var(--accent), color white
```

Icon size inside buttons: 18px × 18px.

## Result Cards

### Diagnosis Card
```
Background:     var(--bg-surface)
Border:         var(--border)
Border-radius:  var(--radius-lg)
Overflow:       hidden
```

Internal sections divided by `border-bottom: var(--border)`:
1. Primary diagnosis header — name + confidence + risk badge
2. Description paragraph
3. Advice row (colored background matching risk level)

Risk badge:
```
Border-radius: var(--radius-full)
Padding:       5px 14px
Font:          --text-xs, weight 500, letter-spacing 0.04em
Colors per risk level — see design-tokens.md
```

### Classification Breakdown Bar
```
Bar track:  background var(--bg-elevated), border-radius 999px, height 5px
Bar fill:   background from CLASS_INFO bar_color, border-radius 999px
Row:        flex, gap 12px, align-items center, padding 8px 0
Name col:   min-width 185px, --text-sm
Pct label:  --text-sm, weight 500, min-width 46px, text-align right
Top result row: slightly elevated background (risk advice_bg tint)
```

## Code / Terminal Blocks

Used in any "How it works" or technical section.

```
Background:    var(--bg-surface)
Border:        var(--border)
Border-radius: var(--radius-md)
Padding:       20px 24px
Font:          JetBrains Mono or Fira Code, 13px
Color:         var(--text-primary)
Line-height:   1.6
Overflow-x:    auto
```

## Badges / Tags

```
Inline-flex, align-items center, gap 4px
Padding:       3px 10px
Border-radius: var(--radius-full)
Font:          DM Sans --text-xs, weight 500, uppercase, letter-spacing 0.06em
Background:    var(--accent-subtle)
Color:         #5dcaa5  (light green for dark backgrounds)
Border:        0.5px solid rgba(15,110,86,0.3)
```

## Dividers

```html
<div class="divider"></div>
```
```css
.divider {
  height: 0.5px;
  background: var(--bg-border);
  width: 100%;
}
```

Use `flex: 1` dividers inline with section labels to create the "label — ————" pattern.

## Skeleton / Loading State

Cards and result areas show a shimmer skeleton while loading.

```
Background:   linear-gradient(90deg, var(--bg-surface) 25%, var(--bg-elevated) 50%, var(--bg-surface) 75%)
Background-size: 200% 100%
Animation:    shimmer 1.5s infinite
Border-radius: same as the component being loaded
```
