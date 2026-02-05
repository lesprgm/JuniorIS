---
name: "14 — Progressive streaming (minimum viable version)"
about: "Phased manifest loading: greybox → style → large props → small props"
labels: "mvp"
---

## Implement

- Manifest that defines phases:
  1. greybox
  2. style (lighting/materials)
  3. large props
  4. small props
- Streaming scheduler in Babylon runtime that loads phase N only after N−1 ready.

## Done when

- World becomes walkable quickly and improves visually after.
