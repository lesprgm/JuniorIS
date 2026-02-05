---
name: "13 — Portal creation + transition (Creation Room → World)"
about: "Portal object that appears when destination is ready, scene switch, return home"
labels: "mvp"
---

## Implement

- Portal object in Creation Room that appears only when:
  - destination greybox loaded
  - teleport mesh available
  - safe spawn computed
- Portal enter triggers scene switch or world streaming start.
- "Return Home" reverses that.

## Done when

- Transition is consistent, no reload required, no broken state.
