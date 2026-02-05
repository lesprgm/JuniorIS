# VR-First Prompt-to-Walkable World System (Holodeck)
This project explores a VR-first workflow for turning a natural language prompt into a walkable, explorable 3D environment. The user starts in a stable Creation Room, submits a prompt (voice or text), and enters the generated destination world through a portal once the world meets a minimum playability threshold. The system emphasizes deterministic compilation from a controlled asset/template library and progressive streaming (greybox first, then style and props) to keep VR comfortable and reliable.


## Feature Calendar

| **Issue** | **Due date** | **Phase** |
| --------- | ------------ | --------- |
| [01 - WebXR session start/end/re-enter + app state machine](https://github.com/lesprgm/JuniorIS/issues/2) | Feb 7, 2026 | MVP |
| [02 - Creation Room base scene + safe spawn validation](https://github.com/lesprgm/JuniorIS/issues/3) | Feb 11, 2026 | MVP |
| [03 - Teleport locomotion (comfort-first) with floor tagging + collision rejection](https://github.com/lesprgm/JuniorIS/issues/4) | Feb 14, 2026 | MVP |
| [04 - Minimal in-VR UI panel (Voice/Type/Submit/Clear/Return Home + status line)](https://github.com/lesprgm/JuniorIS/issues/5) | Feb 18, 2026 | MVP |
| [05 - Prompt input: voice flow + text fallback + editable transcript](https://github.com/lesprgm/JuniorIS/issues/6) | Feb 21, 2026 | MVP |
| [06 - End-to-end request plumbing: POST /plan_and_compile (MVP single call)](https://github.com/lesprgm/JuniorIS/issues/7) | Feb 25, 2026 | MVP |
| [07 - WorldSpec v0 JSON Schema + validator (reject with human-readable errors)](https://github.com/lesprgm/JuniorIS/issues/8) | Feb 28, 2026 | MVP |
| [08 - Pack Registry MVP: pack manifests + loader + GET /packs/search?tags=...](https://github.com/lesprgm/JuniorIS/issues/9) | Mar 4, 2026 | MVP |
| [09 - StyleKit MVP: style manifests + loader + GET /stylekits](https://github.com/lesprgm/JuniorIS/issues/10) | Mar 7, 2026 | MVP |
| [10 - Planner MVP: prompt -> WorldSpec (no hallucinated asset IDs)](https://github.com/lesprgm/JuniorIS/issues/11) | Mar 11, 2026 | MVP |
| [11 - Compiler Phase 0: greybox world generation (walkable first)](https://github.com/lesprgm/JuniorIS/issues/12) | Mar 14, 2026 | MVP |
| [12 - Destination world safe spawn selection + search fallback](https://github.com/lesprgm/JuniorIS/issues/13) | Mar 18, 2026 | MVP |
| [13 - Portal creation + transition (Creation Room -> Destination World) + Return Home](https://github.com/lesprgm/JuniorIS/issues/14) | Mar 21, 2026 | MVP |
| [14 - Progressive streaming MVP: phase manifest + runtime scheduler](https://github.com/lesprgm/JuniorIS/issues/15) | Mar 25, 2026 | MVP |
| [15 - Performance guardrails: enforce budgets + automatic downgrades](https://github.com/lesprgm/JuniorIS/issues/16) | Mar 28, 2026 | MVP |
| [16 - Missing asset substitution: tag-based fallback + placeholder cube + compile report](https://github.com/lesprgm/JuniorIS/issues/17) | Apr 1, 2026 | MVP |
| [17 - Error handling + user recovery: structured error codes + safe fallback to Creation Room](https://github.com/lesprgm/JuniorIS/issues/18) | Apr 4, 2026 | MVP |
| [18 - Minimal tests: golden prompt smoke test + schema + manifest validation](https://github.com/lesprgm/JuniorIS/issues/19) | Apr 8, 2026 | MVP |
| [19 - Better placement logic: light constraints + greedy placement with collision backoff](https://github.com/lesprgm/JuniorIS/issues/20) | Apr 11, 2026 | Post-MVP |
| [20 - Visual polish: decals, ambience audio, post-processing presets](https://github.com/lesprgm/JuniorIS/issues/21) | Apr 15, 2026 | Post-MVP |
| [21 - Remixing + saving WorldSpecs (persist and regenerate variants)](https://github.com/lesprgm/JuniorIS/issues/22) | Apr 18, 2026 | Post-MVP |
| [22 - Live edits via StyleKitPatch (no structural recompilation)](https://github.com/lesprgm/JuniorIS/issues/23) | Apr 22, 2026 | Post-MVP |
| [23 - Missing asset quick options UI (choose substitutes, generate portal variants)](https://github.com/lesprgm/JuniorIS/issues/24) | Apr 26, 2026 | Post-MVP |
