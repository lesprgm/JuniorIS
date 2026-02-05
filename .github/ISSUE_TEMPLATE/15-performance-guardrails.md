---
name: "15 — Performance guardrails (VR survivability)"
about: "Budget checks in compiler for lights, props, and texture tiers"
labels: "mvp"
---

## Implement

- Budget checks in compiler:
  - cap lights
  - cap prop count
  - enforce texture tier
- If over budget, downgrade before sending to runtime.

## Done when

- You can generate several worlds without tanking frame rate.
