# Interactions & Animation

Motion, hover effects, and transitions. The UI should feel alive but never distracting. Every animation has a purpose — it communicates state, guides attention, or provides feedback.

## Principles

- **Purposeful:** Animate to communicate, not to decorate.
- **Fast:** Most transitions are 120–200ms. Nothing exceeds 400ms unless it's a page transition.
- **Easing:** Default `cubic-bezier(0.16, 1, 0.3, 1)` (ease-out-quint) for entrances. `ease` for hovers. `ease-in` for exits.
- **Respect reduced-motion:** Wrap all decorative animations in `@media (prefers-reduced-motion: no-preference)`.

## Timing Reference

```
--duration-fast:    120ms   /* Hover color changes, borders */
--duration-default: 200ms   /* Scale, opacity, background */
--duration-slow:    350ms   /* Card lifts, page transitions */
--duration-enter:   500ms   /* Elements entering the viewport */

--ease-out:   cubic-bezier(0.16, 1, 0.3, 1)
--ease-spring:cubic-bezier(0.34, 1.56, 0.64, 1)  /* Subtle spring for card hover */
```

## Homepage Grid Card Hover

This is the hero interaction of the site — make it feel cinematic.

```css
.grid-card {
  transition:
    transform 300ms cubic-bezier(0.34, 1.56, 0.64, 1),
    box-shadow 300ms ease,
    border-color 200ms ease;
}

.grid-card:hover {
  transform: scale(1.03) translateY(-4px);
  box-shadow: var(--shadow-hover);
  border-color: rgba(15, 110, 86, 0.4);
  z-index: 10;  /* Ensure hovered card renders above neighbors */
}

/* Image zoom within the card */
.grid-card img {
  transition: transform 400ms cubic-bezier(0.16, 1, 0.3, 1);
}
.grid-card:hover img {
  transform: scale(1.06);
}

/* Overlay fade in */
.grid-card .overlay {
  opacity: 0;
  transition: opacity 200ms ease;
}
.grid-card:hover .overlay {
  opacity: 1;
}

/* Label slide up */
.grid-card .card-label {
  transform: translateY(8px);
  opacity: 0;
  transition: transform 250ms var(--ease-out), opacity 200ms ease;
}
.grid-card:hover .card-label {
  transform: translateY(0);
  opacity: 1;
}
```

The hover glow (`--shadow-hover`) is the signature effect — it emits a subtle green radiance from the card edge, tying the interaction to the brand accent.

## Upload Zone Interactions

```css
.upload-zone {
  transition: border-color 150ms ease, background 150ms ease, box-shadow 300ms ease;
}
.upload-zone:hover,
.upload-zone.dragover {
  border-color: var(--accent);
  background: rgba(15, 110, 86, 0.05);
  box-shadow: 0 0 0 4px rgba(15, 110, 86, 0.08), var(--shadow-glow);
}
.upload-zone.dragover {
  /* More intense on active drag */
  background: rgba(15, 110, 86, 0.1);
  box-shadow: 0 0 0 4px rgba(15, 110, 86, 0.15), var(--shadow-glow);
}
```

## Button Interactions

```css
button {
  transition: background 120ms ease, border-color 120ms ease,
              transform 100ms ease, box-shadow 120ms ease;
}
button:active {
  transform: scale(0.97);
}
```

Primary button active state: darken by ~8%, no scale — feels solid.
Secondary button active state: slight scale(0.97).

## Page Transition (Homepage → Analysis)

Fade + slight upward slide:

```css
/* Outgoing page */
.page-exit { animation: pageExit 200ms ease-in forwards; }

/* Incoming page */
.page-enter { animation: pageEnter 350ms var(--ease-out) forwards; }

@keyframes pageExit {
  to { opacity: 0; transform: translateY(-12px); }
}
@keyframes pageEnter {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}
```

## Scroll-Triggered Entrance (Analysis Page Sections)

Sections on the analysis page slide up and fade in as they enter the viewport. Use `IntersectionObserver`.

```css
.section-reveal {
  opacity: 0;
  transform: translateY(24px);
  transition: opacity 500ms var(--ease-out), transform 500ms var(--ease-out);
}
.section-reveal.visible {
  opacity: 1;
  transform: translateY(0);
}
```

Stagger child elements by 60ms delay per item.

## Result Reveal Animation

When analysis results appear, the diagnosis card and bars animate in sequence:

1. Card fades in from below (translateY 20px → 0, 300ms)
2. Risk badge pops in with spring scale (scale 0.8 → 1, 250ms, spring easing)
3. Confidence bars animate width from 0 to final value, staggered 50ms per bar (400ms each)

```css
.bar-fill {
  width: 0%;
  transition: width 400ms var(--ease-out);
}
/* JS sets width after a tick so the transition fires */
```

## Loading / Processing State

While the model is analyzing, show an animated pulse on the image thumbnail:

```css
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.4; }
}
.image-processing {
  animation: pulse 1.2s ease-in-out infinite;
}
```

Pair with a progress indicator bar (indeterminate) below the image.

## Focus States

All interactive elements must have visible focus rings — never remove outlines entirely.

```css
:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 3px;
  border-radius: var(--radius-sm);
}
```
