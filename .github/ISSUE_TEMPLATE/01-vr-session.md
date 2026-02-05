---
name: "1 — VR session actually starts and recovers"
about: "WebXR entry flow, state machine, session lifecycle"
labels: "mvp"
---

## Implement

- WebXR entry flow that works reliably: `Enter VR` button, session start, session end handler, re-enter support.
- A simple state machine: `IDLE → VR_READY → IN_WORLD → ERROR`.
- On session end, dispose scene objects and return to `VR_READY` state.

## Done when

- You can enter VR, exit VR, re-enter without refresh, no duplicate controllers or broken input.
