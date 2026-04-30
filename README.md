# VR-First Prompt-to-Walkable World System (Holodeck)
This project explores a VR-first workflow for turning a natural language prompt into a walkable, explorable 3D environment. The user starts in a stable Creation Room, submits a prompt (voice or text), and enters the generated destination world through a portal once the world meets a minimum playability threshold. The runtime is Unity (Quest-native) with OpenXR and C# scripts; the backend handles scene-program planning, validation, and compilation. The current system emphasizes deterministic single-room compilation from a controlled asset/template library: the backend produces a validated room shell, relation-aware placements, scene context, and decor directives before Unity realizes the destination room.

## Quick start

```bash
git clone <repo-url>
cd JuniorIS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env` with your OpenRouter key and planner model.

To verify the backend:

```bash
pytest -q tests/test_plan_compile_integration.py
python3 scripts/tmp_live_engine_smoke.py \
  --prompt "Build a cozy reading room with a chair, a small table, and a lamp" \
  --require-llm
```

## Repository structure

- `src/api/` — backend API and request orchestration
- `src/planning/` — semantic planning, asset selection, and WorldSpec creation
- `src/llm/` — LLM planner adapter and structured planning contracts
- `src/placement/` — relation-aware deterministic placement solver
- `src/compilation/` — phase0 compiler and runtime artifact generation
- `src/runtime/` — safe spawn, manifest support, and realization registry
- `src/indexing/` — asset/material indexing and review pipeline
- `tests/` — regression tests for schemas, planner, compiler, API, and runtime contracts
- `scripts/` — smoke tests, review tools, and demo utilities

## LLM planner setup

1. Copy `.env.example` to `.env` (or edit the generated `.env` file).
2. Configure the planner LLM provider:
   - `PLANNER_LLM_PROVIDER=openrouter`
   - `OPEN_ROUTER_KEY=...`
   - `OPEN_ROUTER_MODEL=openai/gpt-5.4`
3. Optional reliability tuning:
   - `PLANNER_LLM_TIMEOUT_S`
   - `PLANNER_LLM_RETRIES`
   - `PLANNER_LLM_RETRY_BACKOFF_S`
   - `PLANNER_LLM_CIRCUIT_FAILURES`
   - `PLANNER_LLM_CIRCUIT_COOLDOWN_S`
4. If you want explicit semantic-only behavior, send `user_prefs.llm_required=true`.
5. If you use shell env loading, run `set -a; source .env; set +a` before starting the backend.

## Backend commands

Use these when you want to show the backend only, without Unity or the headset.

### Regression test suite

Core backend regression suite:

```bash
pytest -q \
  tests/test_validate_worldspec.py \
  tests/test_pack_registry.py \
  tests/test_stylekit_registry.py \
  tests/test_planner.py \
  tests/test_compiler_phase0.py \
  tests/test_api_server.py \
  tests/test_plan_compile_integration.py \
  tests/test_runtime_asset_registry.py \
  tests/test_manifest_contract.py
```

Fast checks by area:

```bash
pytest -q tests/test_plan_compile_integration.py
pytest -q tests/test_api_server.py tests/test_manifest_contract.py
pytest -q tests/test_planner.py tests/test_compiler_phase0.py
```

Review/indexing tooling and live-LLM smoke tests are intentionally kept out of the default visible suite. Run them only when changing asset intake, review queues, material indexing, or live model wiring:

```bash
pytest -q .hidden_harness/tests
```

### 1. Direct smoke test from the terminal

This runs the full backend path:

- prompt -> planner
- planner -> WorldSpec
- WorldSpec -> phase0 compiler
- phase0 -> manifest + safe spawn

Command:

```bash
set -a; source .env; set +a
python3 scripts/tmp_live_engine_smoke.py \
  --prompt "Build a cozy reading room with a chair, a small table, and a lamp" \
  --require-llm
```

Useful flags:

- the smoke path is semantic-only; if the LLM is unavailable, the request fails explicitly
- `--skip-http` skips the FastAPI endpoint check and just runs the Python contract path
- `--build-root build/tmp_live_smoke` keeps demo artifacts in a separate folder

What to show afterward:

- `/Users/leslie/Documents/JuniorIS/build/tmp_live_smoke/<world_id>/manifest.json`
- `/Users/leslie/Documents/JuniorIS/build/tmp_live_smoke/<world_id>/phase0.json`

Those two files are the clearest backend demo artifacts:

- `manifest.json` shows the runtime contract Unity will consume
- `phase0.json` shows the room shell, placements, constraints, substitutions, and safe spawn

### 2. Start the API server

If you want to show the actual backend endpoint:

```bash
set -a; source .env; set +a
python3 -m src.api.server
```

That starts the backend on `http://127.0.0.1:8000`.

### 3. Hit `POST /plan_and_compile`

In a second terminal:

```bash
curl -s http://127.0.0.1:8000/plan_and_compile \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt_text": "Build a cozy reading room with a chair, a small table, and a lamp",
    "user_prefs": {
      "prompt_mode": "llm",
      "llm_required": true
    }
  }' | python3 -m json.tool
```

What this response proves:

- the request was accepted
- the planner produced a valid WorldSpec
- the compiler produced a `world_id`
- the backend decided whether the portal is allowed to open
- the manifest URL and phase0 artifact exist for the runtime

### 4. Show the generated contract files

After the API call, open:

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path("build")
world_dirs = sorted([p for p in root.glob("world_*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
if not world_dirs:
    raise SystemExit("No build/world_* artifacts found.")

world_dir = world_dirs[0]
print("latest world:", world_dir)
for name in ("manifest.json", "phase0.json"):
    path = world_dir / name
    print(f"\n=== {path} ===")
    print(path.read_text())
PY
```

This is the most concrete backend-only demo because it shows exactly what will comprise the room:

- `template_id`
- `stylekit_id`
- `pack_ids`
- `placements`
- `budgets`
- `safe_spawn`
- `teleportable_surfaces`
- substitution decisions if any asset had to be replaced

### 5. Optional: show a strict LLM-only failure mode

```bash
curl -s http://127.0.0.1:8000/plan_and_compile \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt_text": "Build a quiet study room",
    "user_prefs": {
      "prompt_mode": "llm",
      "llm_required": true
    }
  }' | python3 -m json.tool
```

## Material review commands

Use these when you want to build and review the StyleKit material catalog the same way you review assets.

### 1. Build the material catalog, review queue, and pool

```bash
python3 src/indexing/build_material_catalog.py
python3 src/indexing/build_material_review_pass.py
python3 src/indexing/build_style_material_pool.py
```

That writes:

- `/Users/leslie/Documents/JuniorIS/data/index/material_catalog_v1.json`
- `/Users/leslie/Documents/JuniorIS/data/review/material_review_queue_v1.json`
- `/Users/leslie/Documents/JuniorIS/data/index/style_material_pool_v1.json`

### 2. Open the local material review UI

```bash
python3 scripts/review_material_queue_server.py
```

Then open `http://127.0.0.1:8766`.

The review flow is:

- `Yes` = approve the material for future StyleKit use
- `No` = exclude the material from future StyleKit use
- `Skip` = defer the decision

### 3. Rebuild after a review batch

```bash
python3 scripts/rebuild_material_review_outputs.py
```

This rescans Unity `.mat` files, rebuilds the review queue, and refreshes the filtered style-material pool.

## Runtime asset hygiene

The active MVP planner/runtime path intentionally excludes FoodPack and food-like tabletop clutter. Food prompts may still produce furniture-based rooms, but the planner-safe pool, runtime registry, review queues, material queues, semantic overrides, and thumbnail manifest are scrubbed so plate, mug, cup, burger, pizza, and similar food props are not surfaced for generation.

The invariant is covered by:

```bash
pytest -q tests/test_runtime_asset_registry.py
```



## Feature Calendar

| **Issue** | **Due date** | **Phase** |
| --------- | ------------ | --------- |
| [01 - OpenXR session start/end/re-enter + app state machine](https://github.com/lesprgm/JuniorIS/issues/2) | Feb 7, 2026 | MVP |
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
| [14 - Progressive streaming MVP: phase manifest + runtime scheduler](https://github.com/lesprgm/JuniorIS/issues/15) | Mar 25, 2026 | Deferred / out of scope |
| [15 - Performance guardrails: enforce budgets + automatic downgrades](https://github.com/lesprgm/JuniorIS/issues/16) | Mar 28, 2026 | MVP |
| [16 - Missing asset substitution: tag-based fallback + placeholder cube + compile report](https://github.com/lesprgm/JuniorIS/issues/17) | Apr 1, 2026 | MVP |
| [17 - Error handling + user recovery: structured error codes + safe fallback to Creation Room](https://github.com/lesprgm/JuniorIS/issues/18) | Apr 4, 2026 | MVP |
| [18 - Minimal tests: golden prompt smoke test + schema + manifest validation](https://github.com/lesprgm/JuniorIS/issues/19) | Apr 8, 2026 | MVP |
| [19 - Better placement logic: light constraints + greedy placement with collision backoff](https://github.com/lesprgm/JuniorIS/issues/20) | Apr 11, 2026 | Post-MVP |
| [20 - Visual polish: decals, ambience audio, post-processing presets](https://github.com/lesprgm/JuniorIS/issues/21) | Apr 15, 2026 | Post-MVP |
| [21 - Remixing + saving WorldSpecs (persist and regenerate variants)](https://github.com/lesprgm/JuniorIS/issues/22) | Apr 18, 2026 | Post-MVP |
| [22 - Live edits via StyleKitPatch (no structural recompilation)](https://github.com/lesprgm/JuniorIS/issues/23) | Apr 22, 2026 | Post-MVP |
| [23 - Missing asset quick options UI (choose substitutes, generate portal variants)](https://github.com/lesprgm/JuniorIS/issues/24) | Apr 26, 2026 | Post-MVP |
