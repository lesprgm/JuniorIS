---
name: "3 — Teleport locomotion (comfort first)"
about: "Teleport arc, landing indicator, floor-only teleport with collision check"
labels: "mvp"
---

## Implement

- Teleport arc + landing indicator.
- Teleport only onto "floor" meshes tagged as teleportable.
- Post-teleport collision check: if destination intersects collider, reject teleport.

## Done when

- Teleport feels stable and never places user inside objects.
