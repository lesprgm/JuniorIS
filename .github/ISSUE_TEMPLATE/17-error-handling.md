---
name: "17 — Error handling + user recovery (critical)"
about: "Error codes, user messages, safe fallback to Creation Room"
labels: "mvp"
---

## Implement

- Every API response includes `error_code` and `user_message` on failure.
- Client always has a safe fallback: return to Creation Room and retry.
- Clear "what happened" message in UI.

## Done when

- The system never strands the user in a broken scene.
