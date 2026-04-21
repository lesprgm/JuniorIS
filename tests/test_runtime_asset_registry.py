from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.planning.asset_catalog import load_planner_pool
from src.runtime.realization_registry import build_runtime_asset_registry


FOOD_PROP_SUBSTRINGS = (
    "foodpack/",
    "/foodpack/",
    "coffemug",
    "sm_plate",
    "freecup",
    "freeplate",
    "coffeemakerwithplatecup",
    "burgerplate",
    "chickenplate",
    "fishplate",
    "hotdogplate",
    "pizzaplate",
    "sandwichplate",
    "sushiplate",
    "tacoplate",
    "wrapplate",
)
FOOD_PROP_TOKENS = {
    "foodpack",
    "food",
    "burger",
    "pizza",
    "pastry",
    "sandwich",
    "sushi",
    "taco",
    "hotdog",
    "fries",
    "fry",
    "fishchips",
    "chicken",
    "wrap",
    "yoghurt",
    "cola",
    "ketchup",
    "soysauce",
    "noodle",
    "onionring",
    "mug",
    "cup",
    "plate",
    "bowl",
    "tray",
}
FOOD_PROP_TEXT_KEYS = ("asset_id", "label", "display_name", "semantic_role_key", "category", "room_role_subtype", "prefab_path", "source_pack")
FOOD_PROP_LIST_KEYS = ("tags", "usable_roles", "room_affinities", "style_tags", "support_surface_types", "negative_scene_affinities")


def _asset_values(asset: dict[str, Any]) -> list[str]:
    values = [str(asset.get(key) or "") for key in FOOD_PROP_TEXT_KEYS]
    for key in FOOD_PROP_LIST_KEYS:
        raw = asset.get(key)
        if isinstance(raw, list):
            values.extend(str(value) for value in raw)
    return values


def _is_food_prop(asset: dict[str, Any]) -> bool:
    values = _asset_values(asset)
    haystack = " ".join(values).lower()
    if any(term in haystack for term in FOOD_PROP_SUBSTRINGS):
        return True
    tokens = {token for value in values for token in re.split(r"[^a-z0-9]+", value.lower()) if token}
    return bool(tokens & FOOD_PROP_TOKENS)


# Keep behavior deterministic so planner/runtime contracts stay stable.
def test_runtime_asset_registry_only_includes_approved_assets():
    registry = build_runtime_asset_registry(
        [
            {
                "asset_id": "pack/chair-a",
                "label": "chair",
                "tags": ["chair"],
                "prefab_path": "Assets/Pack/Chair.prefab",
                "planner_approved": True,
                "planner_excluded": False,
                "quality_tier": "medium",
                "perf_tier": "cheap",
                "semantic_confidence": 0.9,
            },
            {
                "asset_id": "pack/chair-b",
                "label": "chair",
                "tags": ["chair"],
                "prefab_path": "Assets/Pack/ChairB.prefab",
                "planner_approved": False,
                "planner_excluded": False,
            },
            {
                "asset_id": "pack/lamp-a",
                "label": "light",
                "tags": ["light"],
                "prefab_path": "Assets/Pack/Lamp.prefab",
                "planner_approved": True,
                "planner_excluded": True,
            },
        ]
    )

    assert registry["approved_asset_count"] == 1
    chair = registry["asset_realizations"][0]
    assert chair["asset_id"] == "pack/chair-a"
    assert chair["role"] == "chair"
    assert chair["prefab_path"] == "Assets/Pack/Chair.prefab"
    assert chair["target_height"] == 0.95
    assert chair["proxy_kind"] == "chair"
    assert chair["front_yaw_offset_degrees"] == 180.0
    assert chair["quality_tier"] == "medium"
    assert chair["perf_tier"] == "cheap"
    assert chair["semantic_confidence"] == 0.9
    assert "allowed_anchors" in chair
    assert "placement_modes" in chair
    assert "clutter_weight" in chair


def test_runtime_asset_registry_builds_role_representatives():
    registry = build_runtime_asset_registry(
        [
            {
                "asset_id": "pack/table-expensive",
                "label": "table",
                "tags": ["table"],
                "prefab_path": "Assets/Pack/TableExpensive.prefab",
                "planner_approved": True,
                "planner_excluded": False,
                "quality_tier": "high",
                "perf_tier": "expensive",
                "semantic_confidence": 0.95,
                "source_pack": "Package04_Free_Kitchen",
            },
            {
                "asset_id": "pack/table-cheap",
                "label": "table",
                "tags": ["table", "study"],
                "prefab_path": "Assets/Pack/TableCheap.prefab",
                "planner_approved": True,
                "planner_excluded": False,
                "quality_tier": "medium",
                "perf_tier": "cheap",
                "semantic_confidence": 0.8,
                "source_pack": "PandazoleHome",
            },
        ]
    )

    by_role = {item["role"]: item for item in registry["role_realizations"]}
    assert by_role["table"]["asset_id"] == "pack/table-cheap"
    assert by_role["table"]["proxy_kind"] == "table"
    assert by_role["prop"]["asset_id"] is None


def test_runtime_asset_registry_uses_larger_sofa_target_height():
    registry = build_runtime_asset_registry(
        [
            {
                "asset_id": "pack/sofa-a",
                "label": "sofa",
                "tags": ["sofa"],
                "prefab_path": "Assets/Pack/Sofa.prefab",
                "planner_approved": True,
                "planner_excluded": False,
            },
        ]
    )

    by_role = {item["role"]: item for item in registry["role_realizations"]}
    assert by_role["sofa"]["target_height"] == 1.45
    assert by_role["sofa"]["front_yaw_offset_degrees"] == 180.0


def test_runtime_asset_registry_respects_manual_front_yaw_override():
    registry = build_runtime_asset_registry(
        [
            {
                "asset_id": "pack/chair-a",
                "label": "chair",
                "tags": ["chair"],
                "prefab_path": "Assets/Pack/Chair.prefab",
                "planner_approved": True,
                "planner_excluded": False,
                "front_yaw_offset_degrees": 90.0,
            },
        ]
    )

    assert registry["asset_realizations"][0]["front_yaw_offset_degrees"] == 90.0


def test_runtime_asset_registry_carries_stack_target_roles():
    registry = build_runtime_asset_registry(
        [
            {
                "asset_id": "pack/cap-a",
                "label": "decor",
                "tags": ["decor", "greek", "cap"],
                "prefab_path": "Assets/Pack/Cap.prefab",
                "planner_approved": True,
                "planner_excluded": False,
                "stack_target_roles": ["architectural_base", "column"],
            },
        ]
    )

    assert registry["asset_realizations"][0]["stack_target_roles"] == ["architectural_base", "column"]


def test_runtime_asset_registry_carries_group_metadata():
    registry = build_runtime_asset_registry(
        [
            {
                "asset_id": "pack/chair-a",
                "label": "chair",
                "tags": ["chair"],
                "prefab_path": "Assets/Pack/Chair.prefab",
                "planner_approved": True,
                "planner_excluded": False,
                "group_role_affinities": ["member", "dining_member"],
                "supports_group_types": ["dining_set", "reading_corner"],
                "support_surface_types": ["surface"],
                "negative_scene_affinities": ["bathroom"],
                "repeatable_member_role": "chair",
                "seat_front_axis_validated": True,
            },
        ]
    )

    chair = registry["asset_realizations"][0]
    assert chair["group_role_affinities"] == ["member", "dining_member"]
    assert chair["supports_group_types"] == ["dining_set", "reading_corner"]
    assert chair["support_surface_types"] == ["surface"]
    assert chair["negative_scene_affinities"] == ["bathroom"]
    assert chair["repeatable_member_role"] == "chair"
    assert chair["seat_front_axis_validated"] is True


def test_runtime_planner_pool_contains_no_food_props():
    assets = load_planner_pool()
    offenders = [asset.get("asset_id") for asset in assets if _is_food_prop(asset)]

    assert offenders == []
    assert any(asset.get("room_role_subtype") == "coffee_table" for asset in assets)


def test_runtime_registry_contains_no_food_prop_realizations():
    registry = json.loads(Path("data/runtime/runtime_asset_registry_v1.json").read_text())
    realizations = list(registry.get("asset_realizations") or []) + list(registry.get("role_realizations") or [])
    offenders = [asset.get("asset_id") for asset in realizations if isinstance(asset, dict) and _is_food_prop(asset)]

    assert offenders == []
