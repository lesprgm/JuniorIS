from __future__ import annotations

import json

from src.api.server import run_plan_and_compile
from src.contracts.runtime import parse_manifest_payload, validate_manifest_contract
from tests.semantic_test_utils import inline_semantic_prefs


# Keep behavior deterministic so planner/runtime contracts stay stable.
def test_manifest_matches_contract_schema(tmp_path):
    result = run_plan_and_compile(
        "small indoor room with chair and table",
        optional_seed=7,
        user_prefs=inline_semantic_prefs(
            "small indoor room with chair and table",
            scene_type="indoor_room",
            required_roles=["chair", "table"],
            style_tags=["cozy"],
            color_tags=["warm"],
            max_props=2,
        ),
        build_root=tmp_path,
    )
    assert result["ok"] is True

    manifest_path = tmp_path / result["world_id"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = validate_manifest_contract(manifest)
    assert validation["ok"] is True, validation["errors"]
    assert manifest["manifest_version"] == "0.2"
    assert manifest["semantic_path_status"] == "ok"
    assert manifest["colors"]["wall"].startswith("#")
    assert manifest["colors"]["accent"].startswith("#")
    assert set(manifest["surface_material_selection"].keys()) >= {"wall", "floor", "ceiling"}
    assert "floor" in manifest["shell_material_bindings"]
    assert manifest["readiness"]["portal_allowed"] is True
    assert manifest["phase0_url"] == f"/build/{result['world_id']}/phase0.json"
    assert isinstance(manifest["phase0_data"], dict)
    assert manifest["phase0_data"]["world_id"] == result["world_id"]
    assert "runtime_polish" in manifest
    assert "stylekit" in manifest
    assert manifest["runtime_polish"]["decals"] == []
    assert manifest["placement_intent"]["density_profile"] == "normal"
    assert manifest["placement_plan"]["target_count"] >= 1
    assert manifest["scene_context"]["archetype"] == "study"
    assert manifest["scene_context"]["zones"] == []
    assert manifest["decor_plan"]["archetype"] == "study"
    assert isinstance(manifest["decor_plan"]["entries"], list)


def test_manifest_parser_accepts_v01_payloads():
    legacy_manifest = {
        "manifest_version": "0.1",
        "generated_at_utc": "2026-04-01T00:00:00Z",
        "world_id": "world_legacy",
        "portal_ready_at_phase": "phase0",
        "worldspec_version": "0.1",
        "template_id": "room_basic",
        "safe_spawn": {"pos": [0.0, 0.0, 0.0], "rot": [0.0, 180.0, 0.0]},
        "readiness": {"phase0_ready": True, "safe_spawn_ready": True, "portal_allowed": True, "blocked_reasons": []},
        "phase0_url": "/build/world_legacy/phase0.json",
        "phase0_data": {"world_id": "world_legacy"},
        "phase_order": ["phase0"],
        "phases": {"phase0": {"artifact": "phase0.json", "teleportable_surfaces": 1}},
        "planner_backend": "llm",
        "semantic_path_status": "failed",
        "candidate_asset_ids": ["core_chair_01"],
        "prompt_plan": {"mode": "llm", "strategy": "semantic_primary", "selected_prompt": "legacy"},
    }
    parsed = parse_manifest_payload(legacy_manifest)
    assert parsed["manifest_version"] == "0.2"
    assert parsed["phase_order"] == ["phase0"]
    assert parsed["semantic_path_status"] == "failed"
    assert parsed["readiness"]["portal_allowed"] is True
    assert parsed["phase0_url"] == "/build/world_legacy/phase0.json"
    assert parsed["phase0_data"]["world_id"] == "world_legacy"
    assert parsed["colors"] == {}
    assert parsed["surface_material_selection"] == {}
    assert parsed["shell_material_bindings"] == {}
    assert parsed["placement_intent"] == {}
    assert parsed["placement_plan"] == {}
    assert parsed["scene_context"] == {}
    assert parsed["decor_plan"] == {}
    assert "runtime_polish" in parsed
    assert parsed["runtime_polish"]["decals"] == []
    validation = validate_manifest_contract(parsed)
    assert validation["ok"] is True, validation["errors"]
