# Project Roadmap

This document tracks all planned issues for the VR Holodeck world-generation system.
Issues 1–17 constitute the **MVP**. Issue 18 prevents future regressions.
Issues 19–23 are **post-MVP** enhancements.

---

## MVP Issues

### Issue 1 — VR session actually starts and recovers

**Implement**

* WebXR entry flow that works reliably: `Enter VR` button, session start, session end handler, re-enter support.
* A simple state machine: `IDLE → VR_READY → IN_WORLD → ERROR`.
* On session end, dispose scene objects and return to `VR_READY` state.

**Done when**

* You can enter VR, exit VR, re-enter without refresh, no duplicate controllers or broken input.

---

### Issue 2 — Creation Room foundation and safe spawn

**Implement**

* Creation Room scene with floor, 4 walls, ceiling.
* Fixed spawn point plus "spawn validation": capsule collider check at spawn position.
* Collision meshes on walls and floor.

**Done when**

* User never spawns inside a wall and cannot walk outside the room.

---

### Issue 3 — Teleport locomotion (comfort first)

**Implement**

* Teleport arc + landing indicator.
* Teleport only onto "floor" meshes tagged as teleportable.
* Post-teleport collision check: if destination intersects collider, reject teleport.

**Done when**

* Teleport feels stable and never places user inside objects.

---

### Issue 4 — Minimal VR UI panel (no product creep)

**Implement**

* One floating panel anchored in front of user.
* Buttons: `Voice`, `Type`, `Submit`, `Clear`, `Return Home`.
* Status line: `Listening`, `Planning`, `Building`, `Ready`, `Error`.

**Done when**

* User can complete the whole loop using the panel only.

---

### Issue 5 — Prompt input works even when voice fails

**Implement**

* Text fallback input (VR keyboard or simple text modal).
* Voice flow only if mic permission granted.
* Transcript editable before submission.

**Done when**

* A user can generate a world with voice disabled.

---

### Issue 6 — World generation request pipeline (end-to-end plumbing)

**Implement**

* One request payload from client: `{prompt_text, optional_seed, user_prefs}`.
* A single API call: `POST /plan_and_compile` (combine steps for MVP).
* Response includes: `{world_id, manifest_url, portal_ready_at_phase}`.

**Done when**

* Clicking Submit triggers backend work and returns a world handle.

---

### Issue 7 — Minimal WorldSpec schema that cannot explode

**Implement**

* WorldSpec v0 fields only:
  * `worldspec_version`
  * `template_id`
  * `stylekit_id`
  * `pack_ids`
  * `seed`
  * `placements[]` (asset_id + transform + room_id)
  * `budgets` (max props, max texture tier, max lights)
* JSON schema file + validator function.

**Done when**

* WorldSpec can be validated and rejected with human-readable errors.

---

### Issue 8 — Pack Registry MVP (asset library that is actually usable)

**Implement**

* Pack manifest format: `pack_id`, `version`, `tags`, `assets[]`, `preview`, `perf_meta`.
* Loader that scans `packs/` and builds an in-memory index.
* `GET /packs/search?tags=...` returning ranked results.

**Done when**

* Planner can query packs and choose only valid assets.

---

### Issue 9 — StyleKit MVP (theme control without overengineering)

**Implement**

* StyleKit manifest: `stylekit_id`, lighting preset, base materials palette, skybox.
* Loader + `GET /stylekits`.

**Done when**

* Worlds can switch theme consistently even with the same template and assets.

---

### Issue 10 — Planner MVP that never invents assets

**Implement**

* Planner logic that maps prompt →:
  * template_id (from a small allowlist)
  * stylekit_id (from StyleKit tags)
  * pack_ids (from Pack Registry tags)
  * placements using anchor slots or a small placement table
* Hard rule: planner may only output asset IDs that exist in the registry.

**Done when**

* No "missing asset" errors happen because planner hallucinated IDs.

---

### Issue 11 — Compiler Phase 0 greybox world (walkable first)

**Implement**

* A template system that generates basic room geometry.
* Colliders for all greybox geometry.
* Teleportable floor mesh generation.

**Done when**

* You can enter the destination world and teleport around immediately.

---

### Issue 12 — Safe spawn in destination world

**Implement**

* Choose spawn on a teleportable floor area.
* Capsule collision test; if collision, search nearby points in a radius.
* If no safe spawn found, compiler fails with error message.

**Done when**

* You never spawn in objects, and you never fall through floor.

---

### Issue 13 — Portal creation + transition (Creation Room → World)

**Implement**

* Portal object in Creation Room that appears only when:
  * destination greybox loaded
  * teleport mesh available
  * safe spawn computed
* Portal enter triggers scene switch or world streaming start.
* "Return Home" reverses that.

**Done when**

* Transition is consistent, no reload required, no broken state.

---

### Issue 14 — Progressive streaming (minimum viable version)

**Implement**

* Manifest that defines phases:
  1. greybox
  2. style (lighting/materials)
  3. large props
  4. small props
* Streaming scheduler in Babylon runtime that loads phase N only after N−1 ready.

**Done when**

* World becomes walkable quickly and improves visually after.

---

### Issue 15 — Performance guardrails (VR survivability)

**Implement**

* Budget checks in compiler:
  * cap lights
  * cap prop count
  * enforce texture tier
* If over budget, downgrade before sending to runtime.

**Done when**

* You can generate several worlds without tanking frame rate.

---

### Issue 16 — Missing asset substitution (avoid failure)

**Implement**

* Substitution function:
  * look up similar assets by tags within packs
  * fallback to "placeholder cube" with neutral material
* Log substitutions in compile report.

**Done when**

* Requests don't fail because "we don't have a bust."

---

### Issue 17 — Error handling + user recovery (critical)

**Implement**

* Every API response includes `error_code` and `user_message` on failure.
* Client always has a safe fallback: return to Creation Room and retry.
* Clear "what happened" message in UI.

**Done when**

* The system never strands the user in a broken scene.

---

### Issue 18 — Minimal tests (stop breaking yourself)

**Implement**

* One golden prompt smoke test: always compiles.
* JSON schema unit tests.
* Pack manifest validation tests.

**Done when**

* You can refactor without fear.

---

## Post-MVP Issues

### Issue 19 — Better placement logic (light constraints)

* Add `on(surface)`, `against_wall`, `near(object)` constraints.
* Greedy big-first placement with collision backoff.

---

### Issue 20 — Decals, ambience, post-processing

* Nice improvements but not required for walkable MVP.

---

### Issue 21 — Remixing + saving WorldSpecs

* Persist specs, regenerate variants.

---

### Issue 22 — Live edits (StyleKitPatch)

* Allow lighting and palette changes without recompiling structure.

---

### Issue 23 — Missing asset "quick options" UI

* Let user choose between substitutes after a compile.

---

## MVP Cut Line

If you implement through **Issue 17**, you have a real MVP:

* Enter VR
* Prompt (voice or text)
* Generate a walkable world from a library
* Portal into it
* Return home
* No hard failures if assets missing

Issue 18 prevents future pain.
