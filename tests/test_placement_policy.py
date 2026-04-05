from src.placement.constraints import default_constraint_for_role
from src.placement.geometry import derive_near_distance, geometry_profile_from_asset, room_capacity_summary
from src.world.templates import ROOM_BASIC_DIMENSIONS


def _asset(asset_id: str, role: str, size_x: float | None = None, size_z: float | None = None):
    asset = {
        "asset_id": asset_id,
        "label": role,
        "tags": [role],
    }
    if size_x is not None and size_z is not None:
        asset["bounds"] = {
            "size": {"x": size_x, "y": 1.0, "z": size_z},
            "center": {"x": 0.0, "y": 0.5, "z": 0.0},
        }
    return asset


def test_geometry_profile_uses_bounds_when_available():
    profile = geometry_profile_from_asset(_asset("large_table", "table", 2.2, 1.6))
    assert profile["bounds_source"] == "asset_bounds"
    assert profile["footprint_radius"] >= 1.1
    assert profile["wall_clearance"] >= 0.3


def test_geometry_profile_falls_back_without_bounds():
    profile = geometry_profile_from_asset(_asset("chair_no_bounds", "chair"))
    assert profile["bounds_source"] == "role_default"
    assert profile["footprint_radius"] > 0


def test_larger_assets_produce_larger_near_distance():
    small_source = geometry_profile_from_asset(_asset("small_chair", "chair", 0.6, 0.6))
    small_target = geometry_profile_from_asset(_asset("small_lamp", "lamp", 0.3, 0.3))
    large_source = geometry_profile_from_asset(_asset("large_table", "table", 2.2, 1.8))
    large_target = geometry_profile_from_asset(_asset("large_sofa", "sofa", 2.4, 1.4))
    assert derive_near_distance(large_source, large_target, "normal") > derive_near_distance(small_source, small_target, "normal")


def test_cluttered_density_raises_capacity():
    profiles = [
        geometry_profile_from_asset(_asset("chair", "chair", 0.7, 0.7)),
        geometry_profile_from_asset(_asset("table", "table", 1.2, 0.9)),
        geometry_profile_from_asset(_asset("lamp", "lamp", 0.3, 0.3)),
        geometry_profile_from_asset(_asset("decor", "decor", 0.4, 0.4)),
    ]
    minimal = room_capacity_summary(ROOM_BASIC_DIMENSIONS, profiles, "minimal", max_props=6, available_count=4)
    cluttered = room_capacity_summary(ROOM_BASIC_DIMENSIONS, profiles, "cluttered", max_props=6, available_count=4)
    assert cluttered["target_count"] >= minimal["target_count"]
    assert cluttered["density_multiplier"] > minimal["density_multiplier"]


def test_default_constraint_prefers_intent_adjacency_then_wall_anchor():
    placement_intent = {
        "density_profile": "normal",
        "anchor_preferences": ["against_wall"],
        "adjacency_pairs": [{"source_role": "chair", "target_role": "table", "relation": "near"}],
        "layout_mood": "cozy",
    }
    assert default_constraint_for_role("chair", ["chair", "table"], placement_intent)["type"] == "near"
    assert default_constraint_for_role("sign", ["sign"], placement_intent)["type"] == "against_wall"
