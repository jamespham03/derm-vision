# Layout & Grid

Page structure, grid systems, breakpoints, and section anatomy for the DermVision website.

## Page Architecture

The site has three pages:

1. **Homepage** (`/`) — Full-screen gallery with warp/fisheye grid (phantom.land style). No scroll, no info — pure visual gallery + CTAs.
2. **Analysis page** (`/analyze`) — Minimal tool interface for model inference (resend.com style).
3. **About page** (`/about`) — Project explainer: hero, stats, how it works, 8 classes, team, CTA. Contains all informational content.

Homepage has no footer and no scroll. About and Analyze share the same nav and footer components.

## Viewport & Container

```css
/* Max content width — do not go wider */
--container-max: 1280px;
--container-padding: 0 40px;  /* desktop */
--container-padding-md: 0 24px; /* tablet */
--container-padding-sm: 0 16px; /* mobile */

.container {
  max-width: var(--container-max);
  margin: 0 auto;
  padding: var(--container-padding);
}
```

Full-bleed backgrounds (hero banners, grid section) extend edge-to-edge. Content inside them is constrained by `.container`.

## Breakpoints

```
sm:  640px   (mobile landscape)
md:  768px   (tablet)
lg:  1024px  (desktop)
xl:  1280px  (large desktop)
```

Design mobile-first. Homepage grid and analysis layout must be fully usable on mobile.

## Navigation

- Fixed to top of viewport, `position: sticky` or `position: fixed`
- Height: 60px desktop, 52px mobile
- Background: `var(--bg-surface)` with `backdrop-filter: blur(16px)` + `border-bottom: var(--border)`
- Left: logo mark + wordmark
- Right: nav links (desktop) → hamburger (mobile)
- Z-index: 100

Nav never obscures content — offset page `padding-top` accordingly.

## Homepage Grid Layout

The core of the homepage is a full-bleed image grid (see `homepage.md` for interaction rules).

```
Desktop: CSS Grid, auto-fill columns, min 280px, max 1fr
Tablet:  2 columns
Mobile:  1 column (single card, full width)
```

Grid gap: `--space-3` (12px). No gap between grid and viewport edges on mobile — cards go edge-to-edge.

The grid should feel dense and cinematic — avoid excessive padding inside grid cells.

## Analysis Page Layout

Single-column, centered, generous vertical rhythm (resend.com style).

```
Max content width:    680px (form / input area)
Max results width:    720px (results cards)
Section vertical gap: --space-20 (80px) between major sections
```

Sections stack vertically: Hero → Upload zone → Results. No sidebars.

## Section Anatomy

Every section follows this vertical structure:
```
Section label   (--text-xs, uppercase, --text-secondary, letter-spacing 0.08em)
Section heading (DM Serif Display, --text-3xl to --text-4xl)
Section body    (optional — --text-base, --text-secondary, max-width 540px)
Section content (the actual component/grid/feature)
```

Maintain consistent `padding-top: --space-20` on each `<section>` element.

## Footer

- Background: `var(--bg-surface)`, `border-top: var(--border)`
- 3-column desktop, stacked mobile
- Content: copyright, nav links, model attribution (EfficientNet-B3 · ISIC 2019)
- Height: auto, `padding: --space-10 0`
