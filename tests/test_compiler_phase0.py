import json
import pathlib

from src.compilation.phase0 import compile_phase0
from tests.semantic_test_utils import approved_surface_material_selection


FIXTURES_DIR = pathlib.Path(__file__).resolve().parent / "fixtures"


# Keep behavior deterministic so planner/runtime contracts stay stable.
def _load(name: str):
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


def test_compile_phase0_success_writes_artifact(tmp_path):
    worldspec = _load("worldspec_phase0_valid.json")
    worldspec["placement_intent"] = {"density_profile": "normal", "anchor_preferences": [], "adjacency_pairs": [], "layout_mood": "cozy"}
    worldspec["placement_plan"] = {"target_count": 2, "derived_capacity": 4}
    worldspec["surface_material_selection"] = approved_surface_material_selection(style_tags=["cozy"], color_tags=["warm"])
    result = compile_phase0(worldspec, build_root=tmp_path)

    assert result["ok"] is True
    assert result["errors"] == []
    assert result["teleportable_surfaces"] == 1
    assert result["phase0_artifact"] is not None
    assert result["safe_spawn"] is not None

    artifact_path = pathlib.Path(result["phase0_artifact"])
    assert artifact_path.exists()

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["phase"] == "phase0"
    assert artifact["world_id"] == result["world_id"]
    assert any(
        node["id"] == "floor" and node["teleportable"] is True
        for node in artifact["template"]["nodes"]
    )
    assert artifact["safe_spawn"]["pos"][1] == 0.0
    assert artifact["placement_policy"]["intent"]["density_profile"] == "normal"
    assert isinstance(artifact["shell_material_bindings"], dict)
    assert "floor" in artifact["shell_material_bindings"]
    assert artifact["template"]["nodes"][0]["surface_material"]["surface_role"] == "floor"


def test_compile_phase0_invalid_worldspec_returns_structured_errors():
    invalid_worldspec = _load("worldspec_invalid_type.json")
    result = compile_phase0(invalid_worldspec, write_artifact=False)

    assert result["ok"] is False
    assert result["phase0_artifact"] is None
    assert result["errors"]
    assert any("seed" in error["path"] for error in result["errors"])


def test_compile_phase0_is_deterministic_for_same_input(tmp_path):
    worldspec = _load("worldspec_phase0_valid.json")
    first = compile_phase0(worldspec, build_root=tmp_path / "run_a")
    second = compile_phase0(worldspec, build_root=tmp_path / "run_b")

    assert first["ok"] is True
    assert second["ok"] is True
    assert first["world_id"] == second["world_id"]
    assert first["phase0_data"] == second["phase0_data"]
    assert first["safe_spawn"] == second["safe_spawn"]


def test_compile_phase0_clamps_out_of_bounds_floor_positions(tmp_path):
    worldspec = _load("worldspec_phase0_valid.json")
    result = compile_phase0(worldspec, build_root=tmp_path)
    assert result["ok"] is True

    dimensions = result["phase0_data"]["template"]["dimensions"]
    max_x = (dimensions["width"] / 2.0) - 0.25
    max_z = (dimensions["length"] / 2.0) - 0.25

    clamped = next(
        placement
        for placement in result["phase0_data"]["placements"]
        if placement["asset_id"] == "core_lamp_01"
    )
    pos = clamped["transform"]["pos"]

    assert -max_x <= pos[0] <= max_x
    assert pos[1] == 0.0
    assert -max_z <= pos[2] <= max_z


def test_compile_phase0_uses_scene_graph_solver_execution(tmp_path):
    worldspec = _load("worldspec_phase0_valid.json")
    worldspec["planner_policy"] = {
        "semantic_path_status": "ok",
        "placement_mode": "scene_graph_solver",
    }

    result = compile_phase0(worldspec, build_root=tmp_path)

    assert result["ok"] is True
    assert result["phase0_data"]["constraints"]["placement_constraints_enabled"] is True
    assert result["phase0_data"]["placement_policy"]["placement_mode"] == "scene_graph_solver"
    assert result["phase0_data"]["substitution_report"]["placement_execution"]["backend"] == "scene_graph_solver"
    assert result["phase0_data"]["substitution_report"]["placement_mode"] == "scene_graph_solver"


def test_compile_phase0_returns_safe_spawn_failure_when_room_is_fully_blocked():
    worldspec = _load("worldspec_phase0_valid.json")
    worldspec["placements"] = [
        {
            "asset_id": "core_table_01",
            "transform": {
                "pos": [0.0, 0.0, 0.0],
                "rot": [0.0, 0.0, 0.0],
                "scale": [64.0, 1.0, 64.0],
            },
        }
    ]

    result = compile_phase0(worldspec, write_artifact=False)
    assert result["ok"] is False
    assert result["phase0_artifact"] is None
    assert result["safe_spawn"] is None
    assert any(error["path"] == "$.safe_spawn" for error in result["errors"])


def test_compile_phase0_substitutes_missing_asset_and_reports_it(monkeypatch, tmp_path):
    worldspec = _load("worldspec_phase0_valid.json")
    worldspec["placements"] = [
        {
            "asset_id": "core_unknown_chair_42",
            "tags": ["chair", "indoor"],
            "transform": {
                "pos": [0.0, 0.0, 0.0],
                "rot": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            },
        }
    ]
    worldspec["pack_ids"] = ["core_pack"]
    monkeypatch.setattr(
        "src.compilation.phase0.resolve_asset_or_substitute",
        lambda **kwargs: {
            "resolved_asset_id": "core_chair_01",
            "resolution_type": "substitute",
            "reason": "deterministic_substitute_match",
            "coherence_checks": {
                "visual_style_match": True,
                "poly_style_match": True,
                "theme_overlap_match": True,
            },
            "rejected_candidate_counts": {},
            "alternatives": ["core_chair_01"],
            "rationale": ["Selected the highest-ranked compile-compatible substitute after pack, tag, and coherence filtering."],
            "selection_backend": "deterministic",
            "semantic_failure_reason": None,
        },
    )

    result = compile_phase0(worldspec, build_root=tmp_path)
    assert result["ok"] is True

    placement = result["phase0_data"]["placements"][0]
    assert placement["requested_asset_id"] == "core_unknown_chair_42"
    assert placement["asset_id"] == "core_chair_01"
    assert placement["resolution_type"] == "substitute"
    assert placement["mode"] == "asset"

    report = result["phase0_data"]["substitution_report"]
    assert report["substitution_count"] == 1
    assert report["resolution_counts"]["substitute"] == 1
    assert report["substitutions"][0]["requested_asset_id"] == "core_unknown_chair_42"


def test_compile_phase0_marks_placeholder_mode_for_placeholder_resolution(tmp_path):
    worldspec = _load("worldspec_phase0_valid.json")
    worldspec["placements"] = [
        {
            "asset_id": "core_unknown_rocket_99",
            "tags": ["rocket", "spaceship"],
            "transform": {
                "pos": [0.0, 0.0, 0.0],
                "rot": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            },
        }
    ]
    worldspec["pack_ids"] = ["core_pack"]

    result = compile_phase0(worldspec, build_root=tmp_path)
    assert result["ok"] is True
    placement = result["phase0_data"]["placements"][0]
    assert placement["resolution_type"] == "placeholder"
    assert placement["mode"] == "placeholder"


def test_compile_phase0_passes_stylekit_theme_into_substitution(monkeypatch, tmp_path):
    worldspec = _load("worldspec_phase0_valid.json")
    worldspec["placements"] = [
        {
            "asset_id": "core_table_01",
            "transform": {
                "pos": [0.0, 0.0, 0.0],
                "rot": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            },
        }
    ]

    captured = {}

    def _fake_resolve(**kwargs):
        captured["room_theme"] = kwargs.get("room_theme")
        return {
            "resolved_asset_id": kwargs["requested_asset_id"],
            "resolution_type": "exact",
            "reason": "asset_found",
            "coherence_checks": {
                "visual_style_match": True,
                "poly_style_match": True,
                "theme_overlap_match": True,
            },
            "rejected_candidate_counts": {},
            "alternatives": [],
            "rationale": [],
            "selection_backend": "semantic",
            "semantic_failure_reason": None,
        }

    monkeypatch.setattr("src.compilation.phase0.resolve_asset_or_substitute", _fake_resolve)

    result = compile_phase0(worldspec, build_root=tmp_path)
    assert result["ok"] is True
    assert "neutral" in (captured.get("room_theme") or {}).get("style_tags", [])


def test_compile_phase0_ignores_against_wall_constraints_in_direct_mode(tmp_path):
    worldspec = _load("worldspec_phase0_valid.json")
    worldspec["placements"] = [
        {
            "asset_id": "core_table_01",
            "constraint": {"type": "against_wall"},
            "transform": {"pos": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
        }
    ]

    result = compile_phase0(worldspec, build_root=tmp_path)
    assert result["ok"] is True
    position = result["phase0_data"]["placements"][0]["transform"]["pos"]
    assert position == [0.0, 0.0, 0.0]


def test_compile_phase0_places_near_constraints_close_to_target(tmp_path):
    worldspec = _load("worldspec_phase0_valid.json")
    worldspec["placements"] = [
        {
            "asset_id": "core_table_01",
            "constraint": {"type": "against_wall"},
            "transform": {"pos": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
        },
        {
            "asset_id": "core_chair_01",
            "constraint": {"type": "near", "target": "core_table_01", "distance": 1.2},
            "transform": {"pos": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
        },
    ]

    result = compile_phase0(worldspec, build_root=tmp_path)
    assert result["ok"] is True


def test_compile_phase0_applies_face_to_yaw_with_front_offset(tmp_path):
    worldspec = _load("worldspec_phase0_valid.json")
    worldspec["placements"] = [
        {
            "asset_id": "core_table_01",
            "transform": {"pos": [0.0, 0.0, 1.0], "rot": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
        },
        {
            "asset_id": "core_chair_01",
            "front_yaw_offset_degrees": 180.0,
            "constraint": {"type": "near", "target": "core_table_01", "relation": "face_to"},
            "transform": {"pos": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
        },
    ]

    result = compile_phase0(worldspec, build_root=tmp_path)

    assert result["ok"] is True
    chair = next(
        placement for placement in result["phase0_data"]["placements"] if placement["asset_id"] == "core_chair_01"
    )
    assert chair["transform"]["rot"][1] == 180.0
    table = result["phase0_data"]["placements"][0]["transform"]["pos"]
    chair = result["phase0_data"]["placements"][1]["transform"]["pos"]
    dx = table[0] - chair[0]
    dz = table[2] - chair[2]
    assert (dx * dx + dz * dz) ** 0.5 <= 2.25
    assert result["phase0_data"]["substitution_report"]["placement_audit"]["relation_failure_count"] == 0


def test_compile_phase0_applies_vertical_origin_offset(monkeypatch, tmp_path):
    worldspec = _load("worldspec_phase0_valid.json")
    worldspec["placements"] = [
        {
            "asset_id": "core_table_01",
            "transform": {"pos": [0.0, 0.0, 1.0], "rot": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
        },
        {
            "asset_id": "core_cabinet_01",
            "vertical_origin_offset_meters": 0.18,
            "constraint": {"type": "against_wall", "target": "core_table_01"},
            "transform": {"pos": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
        },
    ]

    monkeypatch.setattr(
        "src.compilation.phase0.collect_assets",
        lambda pack_ids, registry: [
            {
                "asset_id": "core_table_01",
                "label": "table",
                "tags": ["table"],
                "prefab_path": "Assets/Core/Table.prefab",
                "planner_approved": True,
                "planner_excluded": False,
                "geometry": {"bounds": {"size": {"x": 2.0, "y": 1.0, "z": 2.0}}},
            },
            {
                "asset_id": "core_cabinet_01",
                "label": "cabinet",
                "tags": ["cabinet"],
                "prefab_path": "Assets/Core/Cabinet.prefab",
                "planner_approved": True,
                "planner_excluded": False,
                "geometry": {"bounds": {"size": {"x": 1.0, "y": 2.0, "z": 1.0}}},
                "vertical_origin_offset_meters": 0.18,
            },
        ],
    )
    monkeypatch.setattr(
        "src.compilation.phase0.resolve_asset_or_substitute",
        lambda **kwargs: {
            "resolved_asset_id": kwargs["requested_asset_id"],
            "resolution_type": "exact",
            "reason": "asset_found",
            "coherence_checks": {},
            "rejected_candidate_counts": {},
            "alternatives": [],
            "rationale": [],
            "selection_backend": "semantic",
            "semantic_failure_reason": None,
        },
    )

    result = compile_phase0(worldspec, build_root=tmp_path)

    assert result["ok"] is True
    cabinet = next(
        placement for placement in result["phase0_data"]["placements"] if placement["asset_id"] == "core_cabinet_01"
    )
    assert cabinet["transform"]["pos"][1] == 0.18


def test_compile_phase0_group_repair_repositions_table_centered_seating(tmp_path):
    worldspec = _load("worldspec_phase0_valid.json")
    worldspec["placements"] = [
        {
            "asset_id": "core_table_01",
            "transform": {"pos": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
        },
        *[
            {
                "asset_id": "core_chair_01",
                "group_id": "dining_set",
                "group_layout": "ring",
                "front_yaw_offset_degrees": 180.0,
                "constraint": {"type": "near", "target": "core_table_01", "relation": "face_to"},
                "transform": {"pos": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
            }
            for _ in range(4)
        ],
    ]

    result = compile_phase0(worldspec, build_root=tmp_path)

    assert result["ok"] is False
    assert result["errors"][0]["path"] == "$.placements"
    audit = result["phase0_data"]["substitution_report"]["placement_audit"]
    assert audit["overlap_count"] >= 1


def test_compile_phase0_preserves_wall_anchored_optional_positions(tmp_path):
    worldspec = _load("worldspec_phase0_valid.json")
    worldspec["placements"] = [
        {
            "asset_id": "core_table_01",
            "transform": {"pos": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
        },
        {
            "asset_id": "leartesstudios/props_sm_tablenumber_20-102aac9d",
            "constraint": {"type": "wall", "anchor": "wall", "placement_mode": "wall_hung"},
            "transform": {"pos": [2.8, 1.6, 0.2], "rot": [0.0, 90.0, 0.0], "scale": [1.0, 1.0, 1.0]},
        },
    ]

    result = compile_phase0(worldspec, build_root=tmp_path)

    assert result["ok"] is True
    sign = next(
        placement
        for placement in result["phase0_data"]["placements"]
        if placement["asset_id"] == "leartesstudios/props_sm_tablenumber_20-102aac9d"
    )
    assert sign["transform"]["pos"][1] > 0.4
    assert abs(sign["transform"]["pos"][0]) >= 2.7 or abs(sign["transform"]["pos"][2]) >= 2.7


def test_compile_phase0_near_constraints_do_not_reposition_assets_in_direct_mode(tmp_path):
    small = _load("worldspec_phase0_valid.json")
    small["placements"] = [
        {
            "asset_id": "core_table_01",
            "constraint": {"type": "against_wall"},
            "geometry_profile": {"footprint_radius": 0.55, "wall_clearance": 0.2, "preferred_near_distance": 0.9, "collision_padding_class": "compact", "placement_role": "table"},
            "transform": {"pos": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
        },
        {
            "asset_id": "core_chair_01",
            "constraint": {"type": "near", "target": "core_table_01"},
            "geometry_profile": {"footprint_radius": 0.35, "wall_clearance": 0.15, "preferred_near_distance": 0.8, "collision_padding_class": "compact", "placement_role": "chair"},
            "transform": {"pos": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
        },
    ]
    large = _load("worldspec_phase0_valid.json")
    large["placements"] = [
        {
            "asset_id": "core_table_01",
            "constraint": {"type": "against_wall"},
            "geometry_profile": {"footprint_radius": 1.1, "wall_clearance": 0.3, "preferred_near_distance": 1.4, "collision_padding_class": "wide", "placement_role": "table"},
            "transform": {"pos": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
        },
        {
            "asset_id": "core_chair_01",
            "constraint": {"type": "near", "target": "core_table_01"},
            "geometry_profile": {"footprint_radius": 0.75, "wall_clearance": 0.25, "preferred_near_distance": 1.1, "collision_padding_class": "standard", "placement_role": "chair"},
            "transform": {"pos": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
        },
    ]

    small_result = compile_phase0(small, build_root=tmp_path / "small")
    large_result = compile_phase0(large, build_root=tmp_path / "large")
    assert small_result["ok"] is True
    assert large_result["ok"] is True

    small_table = small_result["phase0_data"]["placements"][0]["transform"]["pos"]
    small_chair = small_result["phase0_data"]["placements"][1]["transform"]["pos"]
    large_table = large_result["phase0_data"]["placements"][0]["transform"]["pos"]
    large_chair = large_result["phase0_data"]["placements"][1]["transform"]["pos"]
    small_distance = ((small_table[0] - small_chair[0]) ** 2 + (small_table[2] - small_chair[2]) ** 2) ** 0.5
    large_distance = ((large_table[0] - large_chair[0]) ** 2 + (large_table[2] - large_chair[2]) ** 2) ** 0.5
    assert small_distance > 0.0
    assert large_distance > 0.0
    assert small_result["phase0_data"]["substitution_report"]["placement_audit"]["overlap_count"] == 0
    assert large_result["phase0_data"]["substitution_report"]["placement_audit"]["overlap_count"] == 0


def test_compile_phase0_drops_decor_before_leaving_residual_overlap(monkeypatch, tmp_path):
    worldspec = _load("worldspec_phase0_valid.json")
    worldspec["placements"] = [
        {"asset_id": "core_table_01", "role": "table", "transform": {"pos": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]}},
        {"asset_id": "core_chair_01", "role": "chair", "transform": {"pos": [0.15, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]}},
        {"asset_id": "wall_art_01", "role": "decor", "transform": {"pos": [0.18, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]}},
    ]
    asset_records = [
        {"asset_id": "core_table_01", "label": "table", "tags": ["table"], "bounds": {"size": {"x": 2.4, "y": 1.0, "z": 2.4}}},
        {"asset_id": "core_chair_01", "label": "chair", "tags": ["chair"], "bounds": {"size": {"x": 1.2, "y": 1.0, "z": 1.2}}},
        {"asset_id": "wall_art_01", "label": "decor", "tags": ["decor"], "bounds": {"size": {"x": 1.4, "y": 1.0, "z": 1.4}}},
    ]

    monkeypatch.setattr("src.compilation.phase0.collect_assets", lambda pack_ids, registry: asset_records)
    monkeypatch.setattr(
        "src.compilation.phase0.resolve_asset_or_substitute",
        lambda **kwargs: {
            "resolved_asset_id": kwargs["requested_asset_id"],
            "resolution_type": "exact",
            "reason": "asset_found",
            "coherence_checks": {},
            "rejected_candidate_counts": {},
            "alternatives": [],
            "rationale": [],
            "selection_backend": "semantic",
            "semantic_failure_reason": None,
        },
    )

    result = compile_phase0(worldspec, build_root=tmp_path)

    assert result["ok"] is True
    audit = result["phase0_data"]["substitution_report"]["placement_audit"]
    assert audit["overlap_count"] == 0
    assert "wall_art_01" not in {placement["asset_id"] for placement in result["phase0_data"]["placements"]}
    assert result["safe_spawn"] is not None


def test_compile_phase0_normalizes_scale_by_role_target_height(monkeypatch, tmp_path):
    worldspec = _load("worldspec_phase0_valid.json")
    worldspec["placements"] = [
        {
            "asset_id": "scaled_chair_01",
            "transform": {"pos": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "scale": [3.0, 3.0, 3.0]},
        }
    ]

    monkeypatch.setattr(
        "src.compilation.phase0.collect_assets",
        lambda pack_ids, registry: [
            {
                "asset_id": "scaled_chair_01",
                "label": "Chair",
                "tags": ["chair"],
                "classification": "prop",
                "quest_compatible": True,
                "bounds": {"size": {"x": 1.0, "y": 2.0, "z": 1.0}},
            }
        ],
    )
    monkeypatch.setattr(
        "src.compilation.phase0.resolve_asset_or_substitute",
        lambda **kwargs: {
            "resolved_asset_id": "scaled_chair_01",
            "resolution_type": "exact",
            "reason": "asset_found",
            "coherence_checks": {},
            "rejected_candidate_counts": {},
            "alternatives": [],
            "rationale": [],
            "selection_backend": "exact",
            "semantic_failure_reason": None,
        },
    )

    result = compile_phase0(worldspec, build_root=tmp_path)

    assert result["ok"] is True
    placement = result["phase0_data"]["placements"][0]
    assert placement["target_height"] == 0.95
    assert placement["transform"]["scale"] == [0.475, 0.475, 0.475]


def test_compile_phase0_skips_unsatisfiable_constraints_without_failing_world(tmp_path):
    worldspec = _load("worldspec_phase0_valid.json")
    worldspec["placements"] = [
        {
            "asset_id": "core_table_01",
            "constraint": {"type": "against_wall"},
            "transform": {"pos": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "scale": [3.0, 1.0, 3.0]},
        },
        {
            "asset_id": "core_lamp_01",
            "constraint": {"type": "near", "target": "missing_anchor", "distance": 1.2},
            "transform": {"pos": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
        },
        {
            "asset_id": "core_chair_01",
            "constraint": {"type": "against_wall"},
            "transform": {"pos": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "scale": [3.0, 1.0, 3.0]},
        },
    ]

    result = compile_phase0(worldspec, build_root=tmp_path)
    assert result["ok"] is True
    placement_report = result["phase0_data"]["substitution_report"]["placement_execution"]
    assert placement_report["backend"] == "scene_graph_solver"
    assert result["phase0_data"]["constraints"]["placement_constraints_enabled"] is True
