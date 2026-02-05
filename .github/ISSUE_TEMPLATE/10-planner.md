---
name: "10 — Planner MVP that never invents assets"
about: "Planner logic that maps prompt to valid registry assets only"
labels: "mvp"
---

## Implement

- Planner logic that maps prompt →:
  - template_id (from a small allowlist)
  - stylekit_id (from StyleKit tags)
  - pack_ids (from Pack Registry tags)
  - placements using anchor slots or a small placement table
- Hard rule: planner may only output asset IDs that exist in the registry.

## Done when

- No "missing asset" errors happen because planner hallucinated IDs.
