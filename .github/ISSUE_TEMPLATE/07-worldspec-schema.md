---
name: "7 — Minimal WorldSpec schema that cannot explode"
about: "WorldSpec v0 JSON schema with validator"
labels: "mvp"
---

## Implement

- WorldSpec v0 fields only:
  - `worldspec_version`
  - `template_id`
  - `stylekit_id`
  - `pack_ids`
  - `seed`
  - `placements[]` (asset_id + transform + room_id)
  - `budgets` (max props, max texture tier, max lights)
- JSON schema file + validator function.

## Done when

- WorldSpec can be validated and rejected with human-readable errors.
