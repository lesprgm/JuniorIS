---
name: "16 — Missing asset substitution (avoid failure)"
about: "Substitution function with tag-based lookup and placeholder cube fallback"
labels: "mvp"
---

## Implement

- Substitution function:
  - look up similar assets by tags within packs
  - fallback to "placeholder cube" with neutral material
- Log substitutions in compile report.

## Done when

- Requests don't fail because "we don't have a bust."
