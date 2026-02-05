---
name: "8 — Pack Registry MVP (asset library that is actually usable)"
about: "Pack manifest format, loader, and search API"
labels: "mvp"
---

## Implement

- Pack manifest format: `pack_id`, `version`, `tags`, `assets[]`, `preview`, `perf_meta`.
- Loader that scans `packs/` and builds an in-memory index.
- `GET /packs/search?tags=...` returning ranked results.

## Done when

- Planner can query packs and choose only valid assets.
