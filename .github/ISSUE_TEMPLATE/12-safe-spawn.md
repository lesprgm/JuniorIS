---
name: "12 — Safe spawn in destination world"
about: "Spawn point selection with capsule collision test and fallback search"
labels: "mvp"
---

## Implement

- Choose spawn on a teleportable floor area.
- Capsule collision test; if collision, search nearby points in a radius.
- If no safe spawn found, compiler fails with error message.

## Done when

- You never spawn in objects, and you never fall through floor.
