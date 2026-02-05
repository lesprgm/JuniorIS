---
name: "6 — World generation request pipeline (end-to-end plumbing)"
about: "Single API call from client to backend for world generation"
labels: "mvp"
---

## Implement

- One request payload from client: `{prompt_text, optional_seed, user_prefs}`.
- A single API call: `POST /plan_and_compile` (combine steps for MVP).
- Response includes: `{world_id, manifest_url, portal_ready_at_phase}`.

## Done when

- Clicking Submit triggers backend work and returns a world handle.
