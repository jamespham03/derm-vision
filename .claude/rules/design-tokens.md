# Design Tokens

Core visual primitives for the DermVision website. Every color, font, and spacing value used anywhere in the UI must come from this token set. Do not hardcode one-off values.

## Color Palette

### Backgrounds
```
--bg-base:       #050505   /* True near-black. Page root background. */
--bg-surface:    #0d0d0d   /* Cards, panels, nav background. */
--bg-elevated:   #141414   /* Hover states on cards, modals. */
--bg-border:     #1f1f1f   /* Subtle dividers and card borders. */
--bg-overlay:    rgba(5,5,5,0.75)  /* Image overlays on hover. */
```

### Text
```
--text-primary:   #f0eeeb   /* Headlines, body copy — warm off-white, not pure white. */
--text-secondary: #888785   /* Labels, captions, metadata. */
--text-muted:     #444340   /* Placeholder text, disabled states. */
--text-inverse:   #050505   /* Text on accent/light backgrounds. */
```

### Accent — DermVision Green
```
--accent:         #0f6e56   /* Primary brand color. Buttons, active states. */
--accent-hover:   #0a5242   /* Accent on hover/press. */
--accent-subtle:  rgba(15,110,86,0.12)  /* Tinted backgrounds (badges, highlights). */
--accent-glow:    rgba(15,110,86,0.25)  /* Drop shadow / glow on hover cards. */
```

### Risk Severity (results page only)
```
--risk-high:      #e24b4a   /* MEL, BCC, SCC */
--risk-moderate:  #ef9f27   /* AKIEC */
--risk-low:       #1d9e75   /* NV, BKL, DF, VASC */
```

### Utility
```
--white:   #ffffff
--black:   #000000
```

## Typography

Font stack: **DM Sans** (body, UI) + **DM Serif Display** (display headings).

Load via Google Fonts:
```html
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=DM+Serif+Display:ital@0;1&display=swap" rel="stylesheet">
```

### Type Scale
```
--text-xs:    11px  / line-height 1.5  / letter-spacing 0.04em
--text-sm:    13px  / line-height 1.6
--text-base:  15px  / line-height 1.7
--text-lg:    18px  / line-height 1.5
--text-xl:    24px  / line-height 1.3
--text-2xl:   32px  / line-height 1.2
--text-3xl:   48px  / line-height 1.1
--text-4xl:   64px  / line-height 1.05
--text-5xl:   80px  / line-height 0.98
```

### Font Weight
```
--weight-light:   300
--weight-regular: 400
--weight-medium:  500
--weight-semibold:600
```

### Usage Rules
- Display headings (hero, section titles): DM Serif Display, weight 400, italic variants allowed
- All other text: DM Sans
- Labels and tags: DM Sans, --text-xs, weight 500, uppercase, letter-spacing 0.08em
- Code/monospace: `'JetBrains Mono', 'Fira Code', monospace`

## Spacing Scale

8px base unit. Use multiples only.
```
--space-1:   4px
--space-2:   8px
--space-3:  12px
--space-4:  16px
--space-5:  20px
--space-6:  24px
--space-8:  32px
--space-10: 40px
--space-12: 48px
--space-16: 64px
--space-20: 80px
--space-24: 96px
```

## Borders & Radius
```
--border:          0.5px solid #1f1f1f
--border-bright:   0.5px solid rgba(255,255,255,0.08)
--radius-sm:   6px
--radius-md:  10px
--radius-lg:  16px
--radius-xl:  24px
--radius-full: 9999px
```

## Shadows
```
--shadow-card:  0 1px 3px rgba(0,0,0,0.5), 0 4px 16px rgba(0,0,0,0.3)
--shadow-hover: 0 0 0 1px rgba(15,110,86,0.3), 0 8px 32px rgba(0,0,0,0.6), 0 0 40px var(--accent-glow)
--shadow-glow:  0 0 60px rgba(15,110,86,0.2)
```
