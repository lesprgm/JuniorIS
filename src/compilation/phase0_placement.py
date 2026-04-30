from __future__ import annotations

import math
from typing import Any, Dict, List

from src.placement.geometry import geometry_profile_from_asset, semantic_role_key
from src.runtime.realization_registry import resolve_target_height_meters

def _safe_vec3(values: Any, default: List[float]) -> List[float]:
    if (
        isinstance(values, list)
        and len(values) == 3
        and all(isinstance(value, (int, float)) for value in values)
    ):
        return [float(values[0]), float(values[1]), float(values[2])]
    return list(default)


def _clamp_floor_position(position: List[float], dimensions: Dict[str, float]) -> List[float]:
    margin = 0.25
    max_x = max((dimensions["width"] / 2.0) - margin, 0.0)
    max_z = max((dimensions["length"] / 2.0) - margin, 0.0)
    x = max(min(position[0], max_x), -max_x)
    z = max(min(position[2], max_z), -max_z)
    return [round(x, 3), 0.0, round(z, 3)]


def _clamp_wall_position(position: List[float], dimensions: Dict[str, float]) -> List[float]:
    margin = 0.05
    max_x = max((dimensions["width"] / 2.0) - margin, 0.0)
    max_z = max((dimensions["length"] / 2.0) - margin, 0.0)
    height = max(float(dimensions.get("height") or 3.0) - 0.2, 0.5)
    x = max(min(float(position[0]), max_x), -max_x)
    z = max(min(float(position[2]), max_z), -max_z)
    if abs(x / max_x) >= abs(z / max_z if max_z > 0 else 0.0):
        x = max_x if x >= 0.0 else -max_x
    else:
        z = max_z if z >= 0.0 else -max_z
    y = max(min(float(position[1]), height), 0.4)
    return [round(x, 3), round(y, 3), round(z, 3)]


def _clamp_surface_position(position: List[float], dimensions: Dict[str, float]) -> List[float]:
    margin = 0.25
    max_x = max((dimensions["width"] / 2.0) - margin, 0.0)
    max_z = max((dimensions["length"] / 2.0) - margin, 0.0)
    height = max(float(dimensions.get("height") or 3.0) - 0.5, 0.8)
    x = max(min(float(position[0]), max_x), -max_x)
    z = max(min(float(position[2]), max_z), -max_z)
    y = max(min(float(position[1]), height), 0.45)
    return [round(x, 3), round(y, 3), round(z, 3)]


def _clamp_ceiling_position(position: List[float], dimensions: Dict[str, float]) -> List[float]:
    margin = 0.15
    max_x = max((dimensions["width"] / 2.0) - margin, 0.0)
    max_z = max((dimensions["length"] / 2.0) - margin, 0.0)
    x = max(min(float(position[0]), max_x), -max_x)
    z = max(min(float(position[2]), max_z), -max_z)
    y = max(float(dimensions.get("height") or 3.0) - margin, 0.5)
    return [round(x, 3), round(y, 3), round(z, 3)]


def _constraint_type(placement: Dict[str, Any]) -> str:
    constraint = placement.get("constraint") if isinstance(placement.get("constraint"), dict) else {}
    return str(constraint.get("type") or "").strip().lower()


def _compiled_transform(
    position: List[float],
    rotation: List[float],
    scale: List[float],
    dimensions: Dict[str, float],
    *,
    constraint_type: str = "floor",
) -> Dict[str, List[float]]:
    if constraint_type == "wall":
        compiled_pos = _clamp_wall_position(position, dimensions)
    elif constraint_type == "surface":
        compiled_pos = _clamp_surface_position(position, dimensions)
    elif constraint_type == "ceiling":
        compiled_pos = _clamp_ceiling_position(position, dimensions)
    else:
        compiled_pos = _clamp_floor_position(position, dimensions)
    return {
        "pos": compiled_pos,
        "rot": [round(rotation[0], 3), round(rotation[1], 3), round(rotation[2], 3)],
        "scale": [round(scale[0], 3), round(scale[1], 3), round(scale[2], 3)],
    }


def _apply_vertical_origin_offset(transform: Dict[str, List[float]], asset_record: Dict[str, Any]) -> None:
    offset = asset_record.get("vertical_origin_offset_meters")
    if not isinstance(offset, (int, float)) or float(offset) == 0.0:
        return
    pos = transform.get("pos")
    if not isinstance(pos, list) or len(pos) != 3:
        return
    pos[1] = round(float(pos[1]) + float(offset), 3)


def _bounds_height_meters(asset_record: Dict[str, Any]) -> float | None:
    bounds = asset_record.get("bounds") if isinstance(asset_record.get("bounds"), dict) else None
    size = bounds.get("size") if isinstance(bounds, dict) and isinstance(bounds.get("size"), dict) else None
    height = size.get("y") if isinstance(size, dict) else None
    if isinstance(height, (int, float)) and float(height) > 0.0:
        return float(height)
    return None


def _normalized_role_scale(scale: List[float], asset_record: Dict[str, Any], role: str) -> List[float]:
    bounds_height = _bounds_height_meters(asset_record)
    if bounds_height is None:
        return list(scale)
    target_height = resolve_target_height_meters(asset_record, role)
    if target_height <= 0.0:
        return list(scale)
    uniform_scale = round(target_height / bounds_height, 4)
    return [uniform_scale, uniform_scale, uniform_scale]


def _constraint_relation(placement: Dict[str, Any]) -> str:
    constraint = placement.get("constraint") if isinstance(placement.get("constraint"), dict) else {}
    return str(constraint.get("relation") or "").strip().lower()


def _yaw_to_target(source_pos: List[float], target_pos: List[float]) -> float | None:
    delta_x = float(target_pos[0]) - float(source_pos[0])
    delta_z = float(target_pos[2]) - float(source_pos[2])
    if abs(delta_x) < 1e-6 and abs(delta_z) < 1e-6:
        return None
    return math.degrees(math.atan2(delta_x, delta_z))


def _wrap_degrees(value: float) -> float:
    return (float(value) + 360.0) % 360.0


def _visible_front_yaw(rotation_y: float, front_yaw_offset: float) -> float:
    return _wrap_degrees(rotation_y + front_yaw_offset)


def _forward_vector_xz(yaw_degrees: float) -> tuple[float, float]:
    radians = math.radians(yaw_degrees)
    return math.sin(radians), math.cos(radians)


def _face_alignment_score(source: Dict[str, Any], target: Dict[str, Any]) -> float | None:
    source_transform = source.get("transform") if isinstance(source.get("transform"), dict) else {}
    target_transform = target.get("transform") if isinstance(target.get("transform"), dict) else {}
    source_pos = source_transform.get("pos")
    target_pos = target_transform.get("pos")
    rotation = source_transform.get("rot")
    if not isinstance(source_pos, list) or not isinstance(target_pos, list) or not isinstance(rotation, list) or len(rotation) != 3:
        return None
    desired_yaw = _yaw_to_target(source_pos, target_pos)
    if desired_yaw is None:
        return None
    visible_yaw = _visible_front_yaw(float(rotation[1]), float(source.get("front_yaw_offset_degrees") or 0.0))
    forward_x, forward_z = _forward_vector_xz(visible_yaw)
    delta_x = float(target_pos[0]) - float(source_pos[0])
    delta_z = float(target_pos[2]) - float(source_pos[2])
    distance = math.hypot(delta_x, delta_z)
    if distance <= 1e-6:
        return None
    target_x = delta_x / distance
    target_z = delta_z / distance
    return (forward_x * target_x) + (forward_z * target_z)


def _resolve_face_to_target(
    placement: Dict[str, Any],
    compiled_inputs: List[Dict[str, Any]],
) -> Dict[str, Any] | None:
    constraint = placement.get("constraint") if isinstance(placement.get("constraint"), dict) else {}
    target_token = str(constraint.get("target") or "").strip().lower()
    if not target_token:
        return None

    for candidate in compiled_inputs:
        if candidate is placement:
            continue
        if target_token in {
            str(candidate.get("asset_id") or "").strip().lower(),
            str(candidate.get("requested_asset_id") or "").strip().lower(),
        }:
            return candidate

    for candidate in compiled_inputs:
        if candidate is placement:
            continue
        if str(candidate.get("role") or "").strip().lower() == target_token:
            return candidate
    return None


def _apply_face_to_corrections(compiled_inputs: List[Dict[str, Any]]) -> None:
    for placement in compiled_inputs:
        if _constraint_relation(placement) != "face_to":
            continue
        target = _resolve_face_to_target(placement, compiled_inputs)
        if target is None:
            continue
        source_transform = placement.get("transform") if isinstance(placement.get("transform"), dict) else {}
        target_transform = target.get("transform") if isinstance(target.get("transform"), dict) else {}
        source_pos = source_transform.get("pos")
        target_pos = target_transform.get("pos")
        if not isinstance(source_pos, list) or not isinstance(target_pos, list):
            continue
        desired_yaw = _yaw_to_target(source_pos, target_pos)
        if desired_yaw is None:
            continue
        rotation = source_transform.get("rot")
        if not isinstance(rotation, list) or len(rotation) != 3:
            continue
        front_yaw_offset = float(placement.get("front_yaw_offset_degrees") or 0.0)
        rotation[1] = round(_wrap_degrees(desired_yaw - front_yaw_offset), 3)


def _distance_xz(left_pos: List[float], right_pos: List[float]) -> float:
    return math.dist((left_pos[0], left_pos[2]), (right_pos[0], right_pos[2]))


def _push_apart(
    *,
    anchor_pos: List[float],
    moving_pos: List[float],
    min_distance: float,
) -> List[float]:
    delta_x = float(moving_pos[0]) - float(anchor_pos[0])
    delta_z = float(moving_pos[2]) - float(anchor_pos[2])
    current_distance = math.hypot(delta_x, delta_z)
    if current_distance < 1e-5:
        delta_x, delta_z = 1.0, 0.0
        current_distance = 1.0
    unit_x = delta_x / current_distance
    unit_z = delta_z / current_distance
    return [
        anchor_pos[0] + (unit_x * min_distance),
        float(moving_pos[1]),
        anchor_pos[2] + (unit_z * min_distance),
    ]


def _targets_asset(source: Dict[str, Any], target: Dict[str, Any]) -> bool:
    constraint = source.get("constraint") if isinstance(source.get("constraint"), dict) else {}
    target_token = str(constraint.get("target") or "").strip().lower()
    if not target_token:
        return False
    return target_token in {
        str(target.get("asset_id") or "").strip().lower(),
        str(target.get("requested_asset_id") or "").strip().lower(),
        str(target.get("role") or "").strip().lower(),
    }


def _repair_overlaps(compiled_inputs: List[Dict[str, Any]], dimensions: Dict[str, float]) -> Dict[str, Any]:  # iterative multi-pass constraint solver that continuously pushes intersecting objects out of bounds
    repaired_pairs = 0
    max_passes = 8
    _apply_face_to_corrections(compiled_inputs)

    for pass_index in range(max_passes):
        changed = False
        for left_index, left in enumerate(compiled_inputs):
            if _constraint_type(left) in {"wall", "surface", "ceiling"}:
                continue
            left_profile = left.get("geometry_profile") if isinstance(left.get("geometry_profile"), dict) else {}
            left_radius = float(left_profile.get("footprint_radius") or 0.0)
            left_transform = left.get("transform") if isinstance(left.get("transform"), dict) else {}
            left_pos = left_transform.get("pos") if isinstance(left_transform.get("pos"), list) else None
            if left_radius <= 0.0 or not isinstance(left_pos, list):
                continue

            for right in compiled_inputs[left_index + 1 :]:
                if _constraint_type(right) in {"wall", "surface", "ceiling"}:
                    continue
                right_profile = right.get("geometry_profile") if isinstance(right.get("geometry_profile"), dict) else {}
                right_radius = float(right_profile.get("footprint_radius") or 0.0)
                right_transform = right.get("transform") if isinstance(right.get("transform"), dict) else {}
                right_pos = right_transform.get("pos") if isinstance(right_transform.get("pos"), list) else None
                if right_radius <= 0.0 or not isinstance(right_pos, list):
                    continue

                minimum_distance = left_radius + right_radius + 0.05
                if _distance_xz(left_pos, right_pos) >= minimum_distance:
                    continue

                moving = right
                anchor = left
                if _targets_asset(left, right) and not _targets_asset(right, left):
                    moving = left
                    anchor = right

                anchor_transform = anchor.get("transform") if isinstance(anchor.get("transform"), dict) else {}
                moving_transform = moving.get("transform") if isinstance(moving.get("transform"), dict) else {}
                anchor_position = anchor_transform.get("pos") if isinstance(anchor_transform.get("pos"), list) else [0.0, 0.0, 0.0]
                moving_position = moving_transform.get("pos") if isinstance(moving_transform.get("pos"), list) else [0.0, 0.0, 0.0]
                repaired_position = _push_apart(
                    anchor_pos=anchor_position,
                    moving_pos=moving_position,
                    min_distance=minimum_distance,
                )
                clamped_position = _clamp_floor_position(repaired_position, dimensions)
                offset = moving.get("vertical_origin_offset_meters")
                if isinstance(offset, (int, float)) and float(offset) != 0.0:
                    clamped_position[1] = round(float(repaired_position[1]), 3)
                moving_transform["pos"] = clamped_position
                repaired_pairs += 1
                changed = True

        if not changed:
            return {
                "repair_passes": pass_index,
                "repaired_pairs": repaired_pairs,
                "group_repairs_applied": 0,
                "group_members_adjusted": 0,
            }

    return {
        "repair_passes": max_passes,
        "repaired_pairs": repaired_pairs,
        "group_repairs_applied": 0,
        "group_members_adjusted": 0,
    }


def _compiled_input(  # joins final geometric clamping with the substitution profile as the core phase0 representation record
    *,
    index: int,
    placement: Dict[str, Any],
    requested_asset_id: str,
    resolved_asset_id: str,
    resolution_type: str,
    reason: str,
    requested_tags: List[Any],
    position: List[float],
    rotation: List[float],
    scale: List[float],
    dimensions: Dict[str, float],
    asset_record: Dict[str, Any],
) -> Dict[str, Any]:
    role = str(placement.get("role") or semantic_role_key(asset_record))
    normalized_scale = _normalized_role_scale(scale, asset_record, role)
    geometry_profile = geometry_profile_from_asset(asset_record, scale=normalized_scale)
    if not geometry_profile and isinstance(placement.get("geometry_profile"), dict):
        geometry_profile = dict(placement["geometry_profile"])
    vertical_origin_offset = float(asset_record.get("vertical_origin_offset_meters") or 0.0)
    transform = _compiled_transform(
        position,
        rotation,
        normalized_scale,
        dimensions,
        constraint_type=_constraint_type(placement),
    )
    _apply_vertical_origin_offset(transform, asset_record)
    return {
        "placement_id": f"placement_{index:03d}",
        "asset_id": resolved_asset_id,
        "requested_asset_id": requested_asset_id,
        "role": role,
        "resolution_type": resolution_type,
        "substitution_reason": reason,
        "mode": "placeholder" if resolution_type == "placeholder" else "asset",
        "tags": requested_tags,
        "constraint": placement.get("constraint"),
        "group_id": placement.get("group_id"),
        "group_layout": placement.get("group_layout"),
        "target_height": round(resolve_target_height_meters(asset_record, role), 4),
        "front_yaw_offset_degrees": float(
            placement.get("front_yaw_offset_degrees")
            or asset_record.get("front_yaw_offset_degrees")
            or 0.0
        ),
        "geometry_profile": geometry_profile,
        "vertical_origin_offset_meters": vertical_origin_offset,
        "transform": transform,
    }
