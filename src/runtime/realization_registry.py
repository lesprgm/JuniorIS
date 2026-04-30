from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Dict, Iterable, List

from src.placement.geometry import semantic_role_key
from src.placement.semantic_taxonomy import role_match_tokens


ROLE_REALIZATION_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "chair": {"target_height": 0.95, "proxy_kind": "chair", "front_yaw_offset_degrees": 180.0, "min_height_ratio": 0.35, "max_height_ratio": 3.25},
    "table": {"target_height": 0.82, "proxy_kind": "table", "min_height_ratio": 0.30, "max_height_ratio": 3.25},
    "desk": {"target_height": 0.82, "proxy_kind": "table", "min_height_ratio": 0.30, "max_height_ratio": 3.25},
    "bench": {"target_height": 0.82, "proxy_kind": "table", "front_yaw_offset_degrees": 180.0, "min_height_ratio": 0.35, "max_height_ratio": 3.0},
    "lamp": {"target_height": 1.55, "proxy_kind": "lamp", "min_height_ratio": 0.30, "max_height_ratio": 2.25},
    "plant": {"target_height": 0.95, "proxy_kind": "prop", "min_height_ratio": 0.25, "max_height_ratio": 3.5},
    "sofa": {"target_height": 1.45, "proxy_kind": "prop", "front_yaw_offset_degrees": 180.0, "min_height_ratio": 0.35, "max_height_ratio": 2.5},
    "bed": {"target_height": 0.72, "proxy_kind": "prop", "min_height_ratio": 0.40, "max_height_ratio": 2.5},
    "cabinet": {"target_height": 1.30, "proxy_kind": "prop", "min_height_ratio": 0.40, "max_height_ratio": 2.75},
    "cabinet/storage": {"target_height": 1.30, "proxy_kind": "prop", "min_height_ratio": 0.40, "max_height_ratio": 2.75},
    "appliance": {"target_height": 1.05, "proxy_kind": "prop", "min_height_ratio": 0.35, "max_height_ratio": 3.0},
    "tool": {"target_height": 0.85, "proxy_kind": "prop", "min_height_ratio": 0.20, "max_height_ratio": 4.0},
    "decor": {"target_height": 0.65, "proxy_kind": "prop", "min_height_ratio": 0.15, "max_height_ratio": 4.0},
    "sign": {"target_height": 1.10, "proxy_kind": "prop", "min_height_ratio": 0.20, "max_height_ratio": 3.5},
    "prop": {"target_height": 1.00, "proxy_kind": "prop", "min_height_ratio": 0.10, "max_height_ratio": 5.0},
}


def _normalized_role(asset: Dict[str, Any]) -> str:
    role = semantic_role_key(asset)
    return role if role else "prop"


def _role_defaults(role: str) -> Dict[str, Any]:
    return dict(ROLE_REALIZATION_DEFAULTS.get(role) or ROLE_REALIZATION_DEFAULTS["prop"])


def resolve_front_yaw_offset_degrees(asset: Dict[str, Any], role: str | None = None) -> float:
    raw_value = asset.get("front_yaw_offset_degrees")
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    defaults = _role_defaults(role or _normalized_role(asset))
    value = defaults.get("front_yaw_offset_degrees")
    return float(value) if isinstance(value, (int, float)) else 0.0


def resolve_target_height_meters(asset: Dict[str, Any], role: str | None = None) -> float:
    raw_value = asset.get("target_height")
    if isinstance(raw_value, (int, float)) and float(raw_value) > 0.0:
        return float(raw_value)
    defaults = _role_defaults(role or _normalized_role(asset))
    value = defaults.get("target_height")
    return float(value) if isinstance(value, (int, float)) else 1.0


def resolve_vertical_origin_offset_meters(asset: Dict[str, Any]) -> float:
    raw_value = asset.get("vertical_origin_offset_meters")
    return float(raw_value) if isinstance(raw_value, (int, float)) else 0.0


def resolve_target_height_ratio_bounds(asset: Dict[str, Any], role: str | None = None) -> tuple[float, float]:
    defaults = _role_defaults(role or _normalized_role(asset))
    min_ratio = defaults.get("min_height_ratio")
    max_ratio = defaults.get("max_height_ratio")
    min_value = float(min_ratio) if isinstance(min_ratio, (int, float)) else 0.1
    max_value = float(max_ratio) if isinstance(max_ratio, (int, float)) else 5.0
    return min_value, max_value


def _representative_sort_key(role: str, asset: Dict[str, Any]) -> tuple:
    quality_rank = {"medium": 0, "high": 1, "low": 2}.get(str(asset.get("quality_tier") or "").strip().lower(), 3)
    perf_rank = {"cheap": 0, "medium": 1, "expensive": 2}.get(str(asset.get("perf_tier") or "").strip().lower(), 3)
    semantic_confidence = float(asset.get("semantic_confidence") or 0.0)
    material_count = int(asset.get("material_count") or 0)
    triangle_count = int(asset.get("triangle_count") or 0)
    role_tokens = role_match_tokens(role)
    path_text = f"{asset.get('prefab_path') or ''} {asset.get('asset_id') or ''}".lower()
    token_rank = 0 if any(token in path_text for token in role_tokens) else 1
    return token_rank, perf_rank, quality_rank, -semantic_confidence, material_count, triangle_count, str(asset.get("asset_id") or "")


def build_runtime_asset_registry(assets: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    approved_assets: List[Dict[str, Any]] = []
    by_role: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for raw in assets:
        if not isinstance(raw, dict):
            continue
        if raw.get("planner_approved") is not True or raw.get("planner_excluded") is True:
            continue
        prefab_path = str(raw.get("prefab_path") or "").strip()
        asset_id = str(raw.get("asset_id") or "").strip()
        if not prefab_path or not asset_id:
            continue

        role = _normalized_role(raw)
        defaults = _role_defaults(role)
        item = {
            "asset_id": asset_id,
            "role": role,
            "prefab_path": prefab_path,
            "target_height": defaults["target_height"],
            "proxy_kind": defaults["proxy_kind"],
            "front_yaw_offset_degrees": resolve_front_yaw_offset_degrees(raw, role),
            "vertical_origin_offset_meters": resolve_vertical_origin_offset_meters(raw),
            "source_pack": raw.get("source_pack"),
            "quality_tier": raw.get("quality_tier"),
            "perf_tier": raw.get("perf_tier"),
            "semantic_confidence": raw.get("semantic_confidence"),
            "allowed_anchors": list(raw.get("allowed_anchors") or []),
            "placement_modes": list(raw.get("placement_modes") or []),
            "usable_roles": list(raw.get("usable_roles") or []),
            "scale_class": raw.get("scale_class"),
            "visual_salience": raw.get("visual_salience"),
            "clutter_weight": raw.get("clutter_weight"),
            "room_affinities": list(raw.get("room_affinities") or []),
            "stack_target_roles": list(raw.get("stack_target_roles") or []),
            "group_role_affinities": list(raw.get("group_role_affinities") or []),
            "supports_group_types": list(raw.get("supports_group_types") or []),
            "support_surface_types": list(raw.get("support_surface_types") or []),
            "negative_scene_affinities": list(raw.get("negative_scene_affinities") or []),
            "repeatable_member_role": raw.get("repeatable_member_role"),
            "seat_front_axis_validated": bool(raw.get("seat_front_axis_validated")),
            "room_role_subtype": raw.get("room_role_subtype"),
            "coherence_family_id": raw.get("coherence_family_id"),
            "collection_id": raw.get("collection_id"),
            "pairing_group": raw.get("pairing_group"),
            "repeat_strategy": raw.get("repeat_strategy"),
        }
        approved_assets.append(item)
        by_role[role].append(item)

    approved_assets.sort(key=lambda item: (item["role"], item["asset_id"]))

    role_realizations: List[Dict[str, Any]] = []
    for role, role_assets in sorted(by_role.items()):
        representative = sorted(role_assets, key=lambda asset: _representative_sort_key(role, asset))[0]
        role_realizations.append(
            {
                "role": role,
                "asset_id": representative["asset_id"],
                "prefab_path": representative["prefab_path"],
                "target_height": representative["target_height"],
                "proxy_kind": representative["proxy_kind"],
                "front_yaw_offset_degrees": representative["front_yaw_offset_degrees"],
            }
        )

    if not any(item["role"] == "prop" for item in role_realizations):
        defaults = _role_defaults("prop")
        role_realizations.append(
            {
                "role": "prop",
                "asset_id": None,
                "prefab_path": None,
                "target_height": defaults["target_height"],
                "proxy_kind": defaults["proxy_kind"],
                "front_yaw_offset_degrees": 0.0,
                "vertical_origin_offset_meters": 0.0,
            }
        )

    fingerprint_source = "\n".join(
        f"{item['asset_id']}|{item['role']}|{item['prefab_path']}|{item['target_height']}|{item['proxy_kind']}|{item['front_yaw_offset_degrees']}|{item['vertical_origin_offset_meters']}"
        for item in approved_assets
    )
    fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()

    return {
        "version": "runtime_asset_registry_v1",
        "fingerprint": fingerprint,
        "approved_asset_count": len(approved_assets),
        "asset_realizations": approved_assets,
        "role_realizations": role_realizations,
    }
