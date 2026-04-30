# Homepage

Rules for the `/` route. Directly adapted from the **phantom.land** UI — Three.js WebGL canvas with pincushion/barrel distortion shader, GSAP animations, and custom cursor. The homepage is a pure visual gallery: no informational text, no stats, no explanations. Its only job is to impress and direct users to either Analyze or About.

## Technical Stack (homepage only)

- **Three.js r128** — WebGL canvas, card meshes, radial gradient textures rendered on canvas
- **GSAP 3.12** — intro animations, card hover color transitions, sibling dimming
- **Post-process shader** — pincushion/fisheye distortion (uK: -0.18) via render target
- **Custom cursor** — 10px dot + 28px ring, mix-blend-mode difference, scales on hover
- **CSS parallax** — canvas `translate()` 14px horizontal / 10px vertical with mouse, `scale(1.04)` for no-edge flicker

## Visual Identity

- Background: `var(--bg-base)` — near-black, full-screen, no scroll
- The grid fills the entire viewport — it IS the page
- Nav and tagline float over the grid
- No hero text, no sections, no footer on the homepage

## Page Structure

```
1. Minimal floating nav (logo + CTA pills)
2. Tagline text (bottom-left, floating over grid)
3. Full-screen warp grid (fills viewport)
```

No footer, no stats strip, no informational content — all that lives on the About page.

## 1. Floating Nav

Transparent overlay nav — no background, no border.

```
Position:    fixed, top 0, full width, z-index 100
Height:      60px
Left:        DermVision logo mark + wordmark
Right:       Two pill CTAs — "About" (secondary) + "Analyze →" (primary accent)
Background:  linear-gradient(to bottom, rgba(5,5,5,0.7), transparent)
```

## 2. Tagline

Mirrors phantom.land's agency tagline position: bottom-left of the viewport, floating over the grid.

```
Position:    fixed, bottom 32px, left 32px, z-index 50
Font:        DM Sans, 13px, color var(--text-secondary)
Max-width:   360px
Line-height: 1.6
```

Content:
> "DermVision is a deep learning system that classifies dermoscopy images across eight lesion categories using EfficientNet-B3, trained on 25,331 clinical images from ISIC 2019."

## 3. Warp Image Grid

The centerpiece. Full-screen, perspective-distorted grid of 20 dermoscopy images.

### Warp / Fisheye Effect

The grid must have a visible barrel/fisheye lens distortion — cards at the edges angle away from the viewer, cards in the center face forward. This creates depth and the sense of a curved display surface.

**Implementation:**

```
Stage:  position fixed, inset 0, overflow hidden
        perspective: 900px applied via JS to grid parent
Grid:   base transform: rotateX(8deg) — tilt grid toward viewer
        transform-style: preserve-3d
        transition: transform 120ms ease-out
        width: 110vw, centered — extends past viewport edges
```

Per-card transforms (computed in JS after layout):
```
Horizontal: outer columns rotateY ±20deg, center columns 0deg (barrel distortion)
Vertical:   top rows rotateX +5deg, bottom rows rotateX -3deg
Scale:      edge cards 0.88–0.92, center cards 1.0
```

Mouse parallax: grid rotateX/Y shifts ±3deg as mouse moves, creating dynamic depth.

On hover: card's edge rotation reduces 60%, pops +25px translateZ, scale +8%, green glow shadow.

### Grid Layout

```
Columns: 5 (desktop), 3 (tablet), 2 (mobile)
Rows:    4
Cards:   20 total, all same aspect ratio 4:5 (portrait)
Gap:     10px
```

Uniform grid (no mixed sizes) ensures correct barrel distortion math.

### Card Content

1. Dermoscopy image or gradient fallback — fills card, object-fit cover
2. Risk-colored dot top-right (always visible)
3. Hover: gradient overlay + class label + name slide up from bottom

All cards link to `/analyze` on click.

### Gradient Fallbacks (8 distinct visuals)

- MEL: near-black / very dark brown center
- NV: warm tan-brown
- BCC: pearl pink / rose
- AKIEC: bright orange
- BKL: vivid gold/yellow
- DF: cool purple-violet
- VASC: vivid cherry-red
- SCC: teal-green

## CTA Pills

```
"About":     transparent bg, dim border, var(--text-secondary) — secondary pill
"Analyze →": var(--accent) background, white text — primary pill
Padding:     8px 18px
Border-radius: var(--r-full)
Font:        DM Sans 13px weight 500
```
