from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple

from src.placement.geometry import (
    geometry_profile_from_asset,
    room_capacity_summary,
    semantic_role_key,
)
from src.placement.scene_solver import solve_scene_layout
from src.world.templates import ROOM_BASIC_DIMENSIONS

from src.planning.asset_shortlist import _ordered_selected_assets, filter_candidate_assets
from src.planning.scene_program_common import _derive_role_fields_from_slots
from src.planning.scene_policy import asset_allowed_by_scene_policy

# Tokens that identify floor-appropriate textiles (carpets, rugs).
_FLOOR_TEXTILE_TOKENS = {"carpet", "rug", "mat", "runner", "floormat"}


def _is_surface_only_textile(asset: Dict[str, Any]) -> bool:
    """Return True for textiles (pillows, towels, blankets) that must NOT be placed on the floor.

    Carpets and rugs are the only textiles allowed on the floor; everything
    else should only appear as a surface-anchored optional addition.
    """
    role = semantic_role_key(asset)
    usable_roles = [str(r).strip().lower() for r in (asset.get("usable_roles") or []) if isinstance(r, str)]
    tags = [str(t).strip().lower() for t in (asset.get("tags") or []) if isinstance(t, str)]
    is_textile = role == "textile" or "textile" in usable_roles or "textile" in tags
    if not is_textile:
        return False
    asset_id_lower = str(asset.get("asset_id") or "").lower()
    label_lower = str(asset.get("label") or "").lower()
    all_tokens = set(tags) | set(usable_roles) | set(asset_id_lower.replace("/", " ").replace("-", " ").replace("_", " ").split()) | set(label_lower.split())
    if all_tokens & _FLOOR_TEXTILE_TOKENS:
        return False  # carpets/rugs belong on the floor
    return True  # pillows, towels, blankets, cushions → surface only


def _scene_slot_targets(scene_program: Dict[str, Any]) -> tuple[List[str], List[str], Dict[str, int]]:
    slots = [
        slot
        for slot in list(scene_program.get("semantic_slots") or [])
        if isinstance(slot, dict) and str(slot.get("slot_id") or "").strip()
    ]
    return _derive_role_fields_from_slots(slots)


def _empty_layout_plan(placement_intent: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "density_profile": placement_intent.get("density_profile", "normal"),
        "derived_capacity": 0,
        "target_count": 0,
        "available_count": 0,
        "average_footprint_area": 0.0,
        "room_area": 0.0,
    }


def _normalized_tag_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    return [str(tag).strip().lower() for tag in values if isinstance(tag, str) and str(tag).strip()]


def _clutter_weight(asset: Dict[str, Any]) -> int:
    value = asset.get("clutter_weight")
    if isinstance(value, (int, float)) and int(value) > 0:
        return int(value)
    return 1


def _optional_constraint(anchor: str, placement_mode: str) -> Dict[str, Any]:
    anchor = str(anchor or "").strip().lower()
    placement_mode = str(placement_mode or "").strip().lower()
    if anchor == "wall":
        return {"type": "wall", "anchor": "wall", "placement_mode": placement_mode or "wall_hung"}
    if anchor == "surface":
        return {"type": "surface", "anchor": "surface", "placement_mode": placement_mode or "surface_top"}
    if anchor == "ceiling":
        return {"type": "ceiling", "anchor": "ceiling", "placement_mode": placement_mode or "ceiling_hung"}
    return {
        "type": "against_wall" if placement_mode == "against_wall" else "floor",
        "anchor": "floor",
        "placement_mode": placement_mode or "standalone",
    }


def _scene_wall_token(scene_program: Dict[str, Any] | None) -> str:
    if not isinstance(scene_program, dict):
        return "front"
    token = str(scene_program.get("focal_wall") or "").strip().lower()
    return token if token in {"front", "back", "left", "right"} else "front"


def _wall_position_from_hint(placement_hint: str, scene_program: Dict[str, Any] | None) -> List[float]:
    focal_wall = _scene_wall_token(scene_program)
    half_width = (ROOM_BASIC_DIMENSIONS["width"] * 0.5) - 0.35
    half_length = (ROOM_BASIC_DIMENSIONS["length"] * 0.5) - 0.35
    y = 1.55
    if placement_hint == "wall_above_anchor":
        placement_hint = "wall_centered"
    if focal_wall == "back":
        base = [0.0, y, half_length]
        left = [-half_width * 0.55, y, half_length]
        right = [half_width * 0.55, y, half_length]
    elif focal_wall == "left":
        base = [-half_width, y, 0.0]
        left = [-half_width, y, half_length * 0.55]
        right = [-half_width, y, -half_length * 0.55]
    elif focal_wall == "right":
        base = [half_width, y, 0.0]
        left = [half_width, y, -half_length * 0.55]
        right = [half_width, y, half_length * 0.55]
    else:
        base = [0.0, y, -half_length]
        left = [-half_width * 0.55, y, -half_length]
        right = [half_width * 0.55, y, -half_length]
    if placement_hint == "wall_left":
        return left
    if placement_hint == "wall_right":
        return right
    return base


def _optional_position_from_hint(
    *,
    index: int,
    anchor: str,
    placement_hint: str,
    usage: str,
    scene_program: Dict[str, Any] | None,
) -> List[float]:
    if placement_hint in {"wall_centered", "wall_left", "wall_right", "wall_above_anchor"}:
        return _wall_position_from_hint(placement_hint, scene_program)
    if placement_hint == "tabletop_center":
        return [0.0, 0.82, 0.0]
    if placement_hint == "tabletop_edge":
        return [0.28 if index % 2 == 0 else -0.28, 0.82, -0.18 if index % 3 else 0.18]
    if placement_hint == "ceiling_center":
        return [0.0, ROOM_BASIC_DIMENSIONS["height"] - 0.15, 0.0]
    if placement_hint == "corner_left":
        return [-2.4, 0.0 if anchor == "floor" else 0.82, 2.4]
    if placement_hint == "corner_right":
        return [2.4, 0.0 if anchor == "floor" else 0.82, 2.4]
    if placement_hint == "floor_edge":
        return [0.0, 0.0, -2.8 if _scene_wall_token(scene_program) == "front" else 2.8]

    if anchor == "wall":
        return _wall_position_from_hint("wall_centered", scene_program)
    if anchor == "surface":
        return [0.0, 0.82, 0.0] if usage == "support" else [0.22, 0.82, -0.15]
    if anchor == "ceiling":
        return [0.0, ROOM_BASIC_DIMENSIONS["height"] - 0.15, 0.0]
    return [0.0, 0.0, 2.8] if usage == "accent" else [0.0, 0.0, -2.8]


def build_optional_raw_placements(
    optional_additions: List[Dict[str, Any]] | None,
    *,
    candidate_assets: List[Dict[str, Any]],
    scene_program: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    if not isinstance(optional_additions, list) or not optional_additions:
        return []
    by_id = {
        str(asset.get("asset_id") or "").strip(): asset
        for asset in candidate_assets
        if isinstance(asset, dict) and str(asset.get("asset_id") or "").strip()
    }
    raw_placements: List[Dict[str, Any]] = []
    for index, entry in enumerate(optional_additions):
        if not isinstance(entry, dict):
            continue
        asset_id = str(entry.get("asset_id") or "").strip()
        asset = by_id.get(asset_id)
        if asset is None:
            continue
        anchor = str(entry.get("anchor") or "").strip().lower()
        placement_mode = str(entry.get("placement_mode") or "").strip().lower()
        usage = str(entry.get("usage") or "").strip().lower()
        # Block surface-only textiles (pillows, towels) from floor placement.
        if anchor in {"", "floor"} and _is_surface_only_textile(asset):
            continue
        placement_hint = str(entry.get("placement_hint") or "").strip().lower()
        geometry_profile = geometry_profile_from_asset(asset)
        default_pos = _optional_position_from_hint(
            index=index,
            anchor=anchor,
            placement_hint=placement_hint,
            usage=usage,
            scene_program=scene_program,
        )
        raw_placements.append(
            {
                "placement_id": f"optional_{index:03d}",
                "asset_id": asset_id,
                "label": asset.get("label"),
                "role": semantic_role_key(asset),
                "usage": usage,
                "placement_hint": placement_hint,
                "tags": _normalized_tag_list(asset.get("tags")),
                "front_yaw_offset_degrees": float(asset.get("front_yaw_offset_degrees") or 0.0),
                "geometry_profile": geometry_profile,
                "constraint": _optional_constraint(anchor, placement_mode),
                "transform": {
                    "pos": default_pos,
                    "rot": [0.0, 0.0, 0.0],
                    "scale": [1.0, 1.0, 1.0],
                },
            }
        )
    return raw_placements


def _layout_hash(placements: List[Dict[str, Any]]) -> str:
    digest = hashlib.sha256(
        json.dumps(
            placements,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    return digest[:12]


def _scene_program_view(
    *,
    intent_spec: Dict[str, Any] | None = None,
    scene_program: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if isinstance(scene_program, dict) and scene_program:
        merged = dict(scene_program)
        return merged

    normalized_intent = dict(intent_spec or {})
    semantic_slots = [
        slot
        for slot in normalized_intent.get("semantic_slots") or []
        if isinstance(slot, dict) and str(slot.get("slot_id") or "").strip()
    ]
    return {
        "semantic_slots": semantic_slots,
        "groups": list(normalized_intent.get("groups") or []),
        "focal_wall": normalized_intent.get("focal_wall"),
        "focal_object_role": normalized_intent.get("focal_object_role"),
        "walkway_preservation_intent": dict(normalized_intent.get("walkway_preservation_intent") or {}),
        "circulation_preference": normalized_intent.get("circulation_preference"),
        "empty_space_preference": normalized_intent.get("empty_space_preference"),
    }


def _placement_profiles(candidate_assets: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    profiles: Dict[str, Dict[str, Any]] = {}
    for asset in candidate_assets:
        asset_id = str(asset.get("asset_id", "")).strip()
        if not asset_id:
            continue
        profiles[asset_id] = geometry_profile_from_asset(asset)
    return profiles


def _asset_text(asset: Dict[str, Any], key: str) -> str:
    return str(asset.get(key) or "").strip().lower()


def _coherent_candidate_for_role(
    role: str,
    *,
    selected_assets: List[Dict[str, Any]],
    candidate_assets: List[Dict[str, Any]],
) -> Dict[str, Any] | None:
    role_assets = [asset for asset in selected_assets if semantic_role_key(asset) == role]
    table_asset = next((asset for asset in selected_assets if semantic_role_key(asset) == "table"), None)
    preferred_family = _asset_text(role_assets[0], "coherence_family_id") if role_assets else ""
    preferred_collection = _asset_text(table_asset or {}, "collection_id") if role in {"chair", "bench"} else ""
    preferred_pairing = _asset_text(table_asset or {}, "pairing_group") if role in {"chair", "bench"} else ""
    matches = [asset for asset in candidate_assets if semantic_role_key(asset) == role]
    if role_assets:
        matches = role_assets + [asset for asset in matches if str(asset.get("asset_id") or "") not in {str(item.get("asset_id") or "") for item in role_assets}]
    if not matches:
        return None
    return sorted(
        matches,
        key=lambda asset: (
            0 if preferred_collection and _asset_text(asset, "collection_id") == preferred_collection else 1,
            0 if preferred_pairing and _asset_text(asset, "pairing_group") == preferred_pairing else 1,
            0 if preferred_family and _asset_text(asset, "coherence_family_id") == preferred_family else 1,
            -float(asset.get("semantic_confidence", 0.55) or 0.55),
            str(asset.get("asset_id") or ""),
        ),
    )[0]


def _group_spec_for_role(scene_program: Dict[str, Any], role: str, group_role: str) -> Dict[str, Any] | None:
    for group in scene_program.get("groups") or []:
        if not isinstance(group, dict):
            continue
        if group_role == "anchor" and str(group.get("anchor_role") or "").strip().lower() == role:
            return group
        if group_role == "member" and str(group.get("member_role") or "").strip().lower() == role:
            return group
    return None


def _clone_selected_asset(
    asset: Dict[str, Any],
    *,
    role: str,
    instance_index: int,
    scene_program: Dict[str, Any],
) -> Dict[str, Any]:
    clone = dict(asset)
    clone["role"] = role
    clone["selected_role"] = role
    clone["selection_instance_index"] = instance_index
    member_group = _group_spec_for_role(scene_program, role, "member")
    anchor_group = _group_spec_for_role(scene_program, role, "anchor")
    group_spec = member_group or anchor_group
    if isinstance(group_spec, dict):
        clone["group_id"] = group_spec.get("group_id")
        clone["group_type"] = group_spec.get("group_type")
        clone["group_layout"] = group_spec.get("layout_pattern")
        clone["group_facing_rule"] = group_spec.get("facing_rule")
        clone["group_role"] = "member" if member_group is not None else "anchor"
    return clone


def _expand_selected_assets_to_target_counts(
    selected_assets: List[Dict[str, Any]],
    *,
    candidate_assets: List[Dict[str, Any]],
    scene_program: Dict[str, Any],
) -> List[Dict[str, Any]]:
    coherent_repeat_roles = {"chair", "table", "sofa", "bench"}
    expanded = [
        _clone_selected_asset(
            dict(asset),
            role=str(asset.get("role") or semantic_role_key(asset)),
            instance_index=index,
            scene_program=scene_program,
        )
        for index, asset in enumerate(selected_assets)
    ]
    _, _, target_role_counts = _scene_slot_targets(scene_program)
    instance_index = max(
        [int(asset.get("selection_instance_index") or 0) for asset in expanded if isinstance(asset.get("selection_instance_index"), int)],
        default=len(expanded),
    )
    for role, target_count in sorted(target_role_counts.items()):
        current_assets = [asset for asset in expanded if semantic_role_key(asset) == role]
        representative = _coherent_candidate_for_role(
            role,
            selected_assets=expanded,
            candidate_assets=candidate_assets,
        )
        if representative is not None and role in coherent_repeat_roles and target_count > 1:
            for index, asset in enumerate(list(expanded)):
                if semantic_role_key(asset) != role:
                    continue
                replacement = dict(representative)
                replacement["role"] = asset.get("role") or role
                replacement["selected_role"] = asset.get("selected_role") or role
                replacement["selection_instance_index"] = asset.get("selection_instance_index")
                for key in ("group_id", "group_type", "group_layout", "group_facing_rule", "group_role"):
                    if key in asset:
                        replacement[key] = asset.get(key)
                expanded[index] = replacement
            current_assets = [asset for asset in expanded if semantic_role_key(asset) == role]
        while len(current_assets) < target_count:
            if representative is None:
                break
            clone = _clone_selected_asset(
                representative,
                role=role,
                instance_index=instance_index,
                scene_program=scene_program,
            )
            instance_index += 1
            expanded.append(clone)
            current_assets.append(clone)
    return expanded


def _layout_inputs_from_selected_assets(  # translates selected semantic objects into raw floor/wall placements
    selected_assets: List[Dict[str, Any]],
    *,
    max_props: int,
    budgets: Dict[str, int] | None,
    intent_spec: Dict[str, Any],
    placement_intent: Dict[str, Any],
    room_dimensions: Dict[str, float],
    candidate_assets: List[Dict[str, Any]],
    scene_program: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    selected_assets = [
        asset
        for asset in filter_candidate_assets(selected_assets)
        if asset_allowed_by_scene_policy(
            asset,
            scene_context=scene_program,
            prompt_text=str(scene_program.get("source_prompt") or ""),
        )
    ]
    candidate_assets = [
        asset
        for asset in filter_candidate_assets(candidate_assets or selected_assets)
        if asset_allowed_by_scene_policy(
            asset,
            scene_context=scene_program,
            prompt_text=str(scene_program.get("source_prompt") or ""),
        )
    ]
    # Exclude surface-only textiles (pillows, towels) from the floor layout.
    selected_assets = [a for a in selected_assets if not _is_surface_only_textile(a)]
    if not selected_assets:
        return [], _empty_layout_plan(placement_intent)

    required_roles, optional_roles, _ = _scene_slot_targets(scene_program)

    budgets = dict(budgets or {})
    effective_max_props = max(1, int(budgets.get("max_props") or max_props or len(selected_assets)))
    selected_assets = _expand_selected_assets_to_target_counts(
        selected_assets,
        candidate_assets=candidate_assets,
        scene_program=scene_program,
    )
    ordered_assets = _ordered_selected_assets(
        selected_assets,
        required_roles=required_roles,
        optional_roles=optional_roles,
    )
    profiles_by_asset = _placement_profiles(ordered_assets)
    geometry_profiles = list(profiles_by_asset.values())
    capacity_summary = room_capacity_summary(
        room_dimensions,
        geometry_profiles,
        str(placement_intent.get("density_profile") or "normal"),
        effective_max_props,
        len(ordered_assets),
    )
    placements, layout_program = solve_scene_layout(
        ordered_assets,
        scene_program=scene_program,
        placement_intent=placement_intent,
        room_dimensions=room_dimensions,
    )
    anchor_counts = {"floor": 0, "wall": 0, "surface": 0, "lights": 0}
    clutter_total = 0
    for placement in placements:
        constraint = placement.get("constraint") if isinstance(placement.get("constraint"), dict) else {}
        anchor = str(constraint.get("anchor") or "").strip().lower()
        if not anchor:
            anchor = "floor" if str(constraint.get("type") or "").strip().lower() not in {"wall", "surface", "ceiling"} else str(constraint.get("type") or "").strip().lower()
        anchor_counts[anchor] = anchor_counts.get(anchor, 0) + 1
        if semantic_role_key(placement) == "lamp":
            anchor_counts["lights"] = anchor_counts.get("lights", 0) + 1
        clutter_total += _clutter_weight(placement)
    return placements, {
        **capacity_summary,
        "target_count": len(placements),
        "available_count": len(ordered_assets),
        "anchor_counts": anchor_counts,
        "clutter_total": clutter_total,
        "clutter_limit": max(1, int(budgets.get("max_clutter_weight") or (effective_max_props * 3))),
        "placement_constraints_enabled": True,
        "placement_backend": "scene_graph_solver",
        "layout_program": layout_program,
        "layout_hash": _layout_hash(placements),
    }


def build_layout_inputs_from_selected_assets(  # public API: produces raw placements and capacity summary from selected assets
    selected_assets: List[Dict[str, Any]],
    *,
    max_props: int,
    budgets: Dict[str, int] | None = None,
    intent_spec: Dict[str, Any],
    placement_intent: Dict[str, Any],
    room_dimensions: Dict[str, float] | None = None,
    candidate_assets: List[Dict[str, Any]] | None = None,
    scene_program: Dict[str, Any] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    room_dimensions = room_dimensions or ROOM_BASIC_DIMENSIONS
    scene_program = _scene_program_view(intent_spec=intent_spec, scene_program=scene_program)
    return _layout_inputs_from_selected_assets(
        selected_assets,
        max_props=max_props,
        budgets=budgets,
        intent_spec=intent_spec,
        placement_intent=placement_intent,
        room_dimensions=room_dimensions,
        candidate_assets=candidate_assets or selected_assets,
        scene_program=scene_program,
    )


def build_layout_from_selected_assets(  # convenience wrapper: builds inputs then immediately applies them as a direct layout
    selected_assets: List[Dict[str, Any]],
    *,
    prompt_text: str,
    seed: int,
    max_props: int,
    budgets: Dict[str, int] | None = None,
    intent_spec: Dict[str, Any],
    placement_intent: Dict[str, Any],
    room_dimensions: Dict[str, float] | None = None,
    candidate_assets: List[Dict[str, Any]] | None = None,
    scene_program: Dict[str, Any] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    del prompt_text, seed
    placements, base_plan = build_layout_inputs_from_selected_assets(
        selected_assets,
        max_props=max_props,
        budgets=budgets,
        intent_spec=intent_spec,
        placement_intent=placement_intent,
        room_dimensions=room_dimensions,
        candidate_assets=candidate_assets,
        scene_program=scene_program,
    )
    if not placements:
        return [], base_plan
    return placements, base_plan
