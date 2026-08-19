---
name: Cyber-Analytic Pulse
colors:
  surface: '#10131a'
  surface-dim: '#10131a'
  surface-bright: '#363940'
  surface-container-lowest: '#0b0e14'
  surface-container-low: '#191c22'
  surface-container: '#1d2026'
  surface-container-high: '#272a31'
  surface-container-highest: '#32353c'
  on-surface: '#e1e2eb'
  on-surface-variant: '#b9cacb'
  inverse-surface: '#e1e2eb'
  inverse-on-surface: '#2e3037'
  outline: '#849495'
  outline-variant: '#3b494b'
  surface-tint: '#00dbe9'
  primary: '#dbfcff'
  on-primary: '#00363a'
  primary-container: '#00f0ff'
  on-primary-container: '#006970'
  inverse-primary: '#006970'
  secondary: '#4edea3'
  on-secondary: '#003824'
  secondary-container: '#00a572'
  on-secondary-container: '#00311f'
  tertiary: '#fff3f2'
  on-tertiary: '#67001b'
  tertiary-container: '#ffcdcf'
  on-tertiary-container: '#bc0b3b'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#7df4ff'
  primary-fixed-dim: '#00dbe9'
  on-primary-fixed: '#002022'
  on-primary-fixed-variant: '#004f54'
  secondary-fixed: '#6ffbbe'
  secondary-fixed-dim: '#4edea3'
  on-secondary-fixed: '#002113'
  on-secondary-fixed-variant: '#005236'
  tertiary-fixed: '#ffdadb'
  tertiary-fixed-dim: '#ffb2b7'
  on-tertiary-fixed: '#40000d'
  on-tertiary-fixed-variant: '#92002a'
  background: '#10131a'
  on-background: '#e1e2eb'
  surface-variant: '#32353c'
typography:
  display-lg:
    fontFamily: Geist
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Geist
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.01em
  headline-lg-mobile:
    fontFamily: Geist
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  data-lg:
    fontFamily: JetBrains Mono
    fontSize: 20px
    fontWeight: '500'
    lineHeight: 28px
  data-sm:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  label-caps:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.05em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  base: 4px
  xs: 8px
  sm: 16px
  md: 24px
  lg: 40px
  xl: 64px
  gutter: 24px
  margin: 32px
---

## Brand & Style
The design system is engineered for high-performance retail intelligence. It targets data scientists and retail executives who require clarity amidst complex ML-driven insights. 

The aesthetic is **Modern Technical / Dark Mode**, blending high-contrast digital elements with a structured, "heads-up display" (HUD) feel. It leverages deep navy backgrounds to reduce eye strain during long analytical sessions, while utilizing neon-adjacent primary colors to highlight critical action points. The emotional response is one of precision, speed, and futuristic reliability.

## Colors
The palette is rooted in a deep "Midnight Navy" foundation to provide maximum contrast for the "Electric Cyan" primary brand color. 

- **Primary (Electric Cyan):** Used for primary actions, active states, and focus indicators.
- **Secondary (Emerald Green):** Reserved exclusively for positive growth metrics, "success" states, and "on" indicators.
- **Tertiary (Rose/Warning):** Used sparingly for critical alerts, downward trends, or data anomalies.
- **Neutrals:** The background uses `#0B0E14`, while UI containers and cards use `#151921` to create subtle depth. Borders are kept thin and muted using `#1E293B` to maintain a clean, wireframe-like precision.

## Typography
This design system employs a dual-font strategy to distinguish between narrative content and technical data.

1. **Geist** is used for headlines and display text to provide a sharp, modern, and technical appearance.
2. **Inter** handles standard body copy for maximum legibility in documentation and long-form descriptions.
3. **JetBrains Mono** is the functional workhorse for all data values, ML metrics, timestamps, and table content. 

All technical labels (`label-caps`) should be set in uppercase with increased letter spacing to emulate a technical terminal aesthetic.

## Layout & Spacing
The layout follows a **Fluid Grid** model with a strictly enforced 4px baseline rhythm. 

- **Desktop:** 12-column grid with 24px gutters. Content is housed in "Modules" (Cards) that typically span 3, 4, 6, or 12 columns.
- **Tablet:** 8-column grid with 16px gutters.
- **Mobile:** 4-column grid with 16px margins. 

Spacing is used to group related data points. Tight spacing (`xs` or `sm`) is used within data cards, while larger spacing (`lg` or `xl`) separates distinct analytical sections.

## Elevation & Depth
Depth is achieved through a combination of **Tonal Layering** and **Glassmorphism**.

1. **Surfaces:** Level 0 is the background (`#0B0E14`). Level 1 is the Card surface (`#151921`).
2. **Glassmorphism:** Navigation bars and floating modals utilize a `backdrop-filter: blur(12px)` with a semi-transparent hex overlay of the surface color at 80% opacity.
3. **Shadows:** Use deep, diffused blue-tinted shadows rather than black. Example: `0 10px 30px -5px rgba(0, 240, 255, 0.08)`.
4. **Borders:** Every card and interactive element must have a 1px solid border (`#1E293B`) to define its boundaries against the dark background.

## Shapes
The design system uses a **Rounded** shape language to soften the high-contrast technical aesthetic. 

- **Standard Elements:** 0.5rem (8px) radius for buttons, inputs, and small cards.
- **Large Containers:** 1rem (16px) radius for primary dashboard modules.
- **Data Tags:** Fully rounded (pill-shaped) for status indicators and category chips.

## Components
- **Buttons:** 
    - *Primary:* Electric Cyan background, JetBlack text, 8px radius. Subtle outer glow on hover.
    - *Ghost:* 1px Cyan border, transparent background, Cyan text.
- **Cards:** 
    - Background: `#151921`, 1px border `#1E293B`. 
    - Header: Separated by a 1px horizontal line.
- **Input Fields:** 
    - Dark fill (`#0B0E14`) with a persistent bottom border of 1px. Border turns Electric Cyan on focus with a 2px outer glow.
- **Data Visualizations:**
    - Use Primary Cyan for main data series.
    - Use Secondary Green for target lines or positive trends.
    - Grid lines in charts should be kept at 0.5px opacity using the border color.
- **Chips/Status:**
    - Small, pill-shaped tags. Use a "Low Alpha" background (e.g., Green at 10% opacity) with solid Green text for readability.
- **Metrics:** 
    - Large JetBrains Mono numbers. Positive changes include an upward chevron in Emerald Green; negative changes include a downward chevron in Rose.