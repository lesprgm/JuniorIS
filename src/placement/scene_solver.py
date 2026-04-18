from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

from src.placement.geometry import derive_near_distance, geometry_profile_from_asset, semantic_role_key
from src.placement.scene_solver_defaults import (
    EDGE_BIASED_ROLES,
    GROUP_CONSTRAINT_DEFAULTS,
    GROUP_ZONE_DEFAULTS,
    SEATING_ROLES,
)


def _round3(value: float) -> float:
    return round(float(value), 3)


def _safe_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _half_extents(dimensions: Dict[str, float], margin: float = 0.8) -> Tuple[float, float]:
    half_width = max((float(dimensions.get("width") or 8.0) * 0.5) - margin, 0.0)
    half_length = max((float(dimensions.get("length") or 8.0) * 0.5) - margin, 0.0)
    return half_width, half_length


def _clamp_floor_position(position: Sequence[float], dimensions: Dict[str, float], margin: float = 0.45) -> List[float]:
    half_width = max((float(dimensions.get("width") or 8.0) * 0.5) - margin, 0.0)
    half_length = max((float(dimensions.get("length") or 8.0) * 0.5) - margin, 0.0)
    return [
        _round3(max(min(float(position[0]), half_width), -half_width)),
        0.0,
        _round3(max(min(float(position[2]), half_length), -half_length)),
    ]


def _yaw_to_target(source_pos: Sequence[float], target_pos: Sequence[float]) -> float:
    delta_x = float(target_pos[0]) - float(source_pos[0])
    delta_z = float(target_pos[2]) - float(source_pos[2])
    if abs(delta_x) < 1e-6 and abs(delta_z) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(delta_x, delta_z))


def _zone_position(zone: str, dimensions: Dict[str, float], index: int = 0) -> List[float]:
    half_width, half_length = _half_extents(dimensions)
    corners = [
        (-half_width, -half_length),
        (half_width, -half_length),
        (-half_width, half_length),
        (half_width, half_length),
    ]
    token = _safe_text(zone) or "center"
    if token == "corner":
        x, z = corners[index % len(corners)]
        return [_round3(x), 0.0, _round3(z)]
    if token == "front":
        return [0.0, 0.0, _round3(-half_length)]
    if token == "back":
        return [0.0, 0.0, _round3(half_length)]
    if token == "left":
        return [_round3(-half_width), 0.0, 0.0]
    if token == "right":
        return [_round3(half_width), 0.0, 0.0]
    if token == "edge":
        edge_slots = [
            (0.0, -half_length),
            (half_width, 0.0),
            (0.0, half_length),
            (-half_width, 0.0),
        ]
        x, z = edge_slots[index % len(edge_slots)]
        return [_round3(x), 0.0, _round3(z)]
    return [0.0, 0.0, 0.0]


def _distance_xz(left: Sequence[float], right: Sequence[float]) -> float:
    return math.dist((float(left[0]), float(left[2])), (float(right[0]), float(right[2])))


def _scene_graph_edges(scene_program: Dict[str, Any], placement_intent: Dict[str, Any]) -> List[Dict[str, str]]:
    relation_graph = list(scene_program.get("relation_graph") or [])
    adjacency_pairs = list(placement_intent.get("adjacency_pairs") or [])
    edges: List[Dict[str, str]] = []
    for edge in relation_graph + adjacency_pairs:
        if not isinstance(edge, dict):
            continue
        source_role = _safe_text(edge.get("source_role"))
        target_role = _safe_text(edge.get("target_role"))
        relation = _safe_text(edge.get("relation"))
        if not source_role or not target_role or not relation:
            continue
        edges.append({"source_role": source_role, "target_role": target_role, "relation": relation})
    return edges


def _is_clear(
    position: Sequence[float],
    footprint_radius: float,
    existing: Sequence[Dict[str, Any]],
    *,
    skip_group_id: str = "",
) -> bool:
    for placement in existing:
        constraint = placement.get("constraint") if isinstance(placement.get("constraint"), dict) else {}
        if _safe_text(constraint.get("type")) in {"wall", "surface", "ceiling"}:
            continue
        if skip_group_id and _safe_text(placement.get("group_id")) == skip_group_id:
            continue
        existing_pos = ((placement.get("transform") or {}).get("pos")) or [0.0, 0.0, 0.0]
        existing_radius = float(((placement.get("geometry_profile") or {}).get("footprint_radius")) or 0.0)
        if existing_radius <= 0.0:
            continue
        if _distance_xz(position, existing_pos) < (footprint_radius + existing_radius + 0.08):
            return False
    return True


def _clearance_score(
    position: Sequence[float],
    footprint_radius: float,
    existing: Sequence[Dict[str, Any]],
    *,
    skip_group_id: str = "",
) -> float:
    if not existing:
        return 1.0
    best_gap = 10.0
    for placement in existing:
        constraint = placement.get("constraint") if isinstance(placement.get("constraint"), dict) else {}
        if _safe_text(constraint.get("type")) in {"wall", "surface", "ceiling"}:
            continue
        if skip_group_id and _safe_text(placement.get("group_id")) == skip_group_id:
            continue
        existing_pos = ((placement.get("transform") or {}).get("pos")) or [0.0, 0.0, 0.0]
        existing_radius = float(((placement.get("geometry_profile") or {}).get("footprint_radius")) or 0.0)
        gap = _distance_xz(position, existing_pos) - (footprint_radius + existing_radius)
        best_gap = min(best_gap, gap)
    return max(min(best_gap, 1.5), -1.5)


def _focal_wall_point(scene_program: Dict[str, Any], dimensions: Dict[str, float]) -> List[float]:
    half_width, half_length = _half_extents(dimensions)
    focal_wall = _safe_text(scene_program.get("focal_wall"))
    if focal_wall == "back":
        return [0.0, 0.0, half_length]
    if focal_wall == "left":
        return [-half_width, 0.0, 0.0]
    if focal_wall == "right":
        return [half_width, 0.0, 0.0]
    if focal_wall == "front":
        return [0.0, 0.0, -half_length]
    return [0.0, 0.0, 0.0]


def _zone_score(position: Sequence[float], zone_preference: str, dimensions: Dict[str, float]) -> float:
    target = _zone_position(zone_preference or "center", dimensions)
    half_width, half_length = _half_extents(dimensions)
    normalizer = max(half_width + half_length, 1.0)
    return 1.0 - min(_distance_xz(position, target) / normalizer, 1.0)


def _walkway_penalty(
    position: Sequence[float],
    role: str,
    scene_program: Dict[str, Any],
    dimensions: Dict[str, float],
) -> float:
    if role in {"wall", "surface", "ceiling"}:
        return 0.0
    half_width, half_length = _half_extents(dimensions)
    central_band = abs(float(position[0])) < (half_width * 0.28) and abs(float(position[2])) < (half_length * 0.28)
    entry_band = abs(float(position[0])) < (half_width * 0.26) and float(position[2]) < (-half_length * 0.35)
    walkway = scene_program.get("walkway_preservation_intent") if isinstance(scene_program.get("walkway_preservation_intent"), dict) else {}
    circulation = _safe_text(scene_program.get("circulation_preference"))
    focal_role = _safe_text(scene_program.get("focal_object_role"))
    penalty = 0.0
    if (bool(walkway.get("keep_central_path_clear")) or circulation == "clear_center") and central_band and role != focal_role:
        penalty += 1.2
    if (bool(walkway.get("keep_entry_clear")) or circulation == "clear_entry") and entry_band and role != focal_role:
        penalty += 1.0
    return penalty


def _focal_score(
    position: Sequence[float],
    yaw: float,
    role: str,
    scene_program: Dict[str, Any],
    dimensions: Dict[str, float],
) -> float:
    focal_role = _safe_text(scene_program.get("focal_object_role"))
    if focal_role != role:
        return 0.0
    target = _focal_wall_point(scene_program, dimensions)
    closeness = 1.0 - min(_distance_xz(position, target) / max(sum(_half_extents(dimensions)), 1.0), 1.0)
    inward_yaw = _yaw_to_target(position, [0.0, 0.0, 0.0])
    facing_alignment = 1.0 - min(abs(((yaw - inward_yaw + 180.0) % 360.0) - 180.0) / 180.0, 1.0)
    return closeness + (0.4 * facing_alignment)


def _candidate_score(
    *,
    position: Sequence[float],
    yaw: float,
    role: str,
    dimensions: Dict[str, float],
    existing: Sequence[Dict[str, Any]],
    footprint_radius: float,
    scene_program: Dict[str, Any],
    zone_preference: str,
    skip_group_id: str = "",
) -> float:
    score = 0.0
    score += 2.5 * _clearance_score(position, footprint_radius, existing, skip_group_id=skip_group_id)
    score += 1.5 * _zone_score(position, zone_preference, dimensions)
    score += _focal_score(position, yaw, role, scene_program, dimensions)
    score -= _walkway_penalty(position, role, scene_program, dimensions)
    return score


def _candidate_variants(desired_pos: Sequence[float], anchor_pos: Sequence[float] | None = None) -> List[List[float]]:
    candidates = [[float(desired_pos[0]), 0.0, float(desired_pos[2])]]
    offsets = [
        (0.55, 0.0),
        (-0.55, 0.0),
        (0.0, 0.55),
        (0.0, -0.55),
        (0.55, 0.55),
        (-0.55, 0.55),
        (0.55, -0.55),
        (-0.55, -0.55),
    ]
    for offset_x, offset_z in offsets:
        candidates.append([float(desired_pos[0]) + offset_x, 0.0, float(desired_pos[2]) + offset_z])
    if anchor_pos is not None:
        base_distance = max(_distance_xz(desired_pos, anchor_pos), 0.7)
        base_angle = math.degrees(math.atan2(float(desired_pos[0]) - float(anchor_pos[0]), float(desired_pos[2]) - float(anchor_pos[2])))
        for ring in range(1, 4):
            distance = base_distance + (0.25 * ring)
            for step in range(8):
                angle = math.radians(base_angle + (step * 45.0))
                candidates.append([
                    float(anchor_pos[0]) + (math.sin(angle) * distance),
                    0.0,
                    float(anchor_pos[2]) + (math.cos(angle) * distance),
                ])
    return candidates


def _resolve_scored_position(
    *,
    desired_pos: Sequence[float],
    yaw: float,
    role: str,
    footprint_radius: float,
    dimensions: Dict[str, float],
    existing: Sequence[Dict[str, Any]],
    scene_program: Dict[str, Any],
    zone_preference: str,
    anchor_pos: Sequence[float] | None = None,
    skip_group_id: str = "",
) -> Tuple[List[float], float]:
    best_position: List[float] | None = None
    best_score = float("-inf")
    for candidate in _candidate_variants(desired_pos, anchor_pos=anchor_pos):
        clamped = _clamp_floor_position(candidate, dimensions)
        if not _is_clear(clamped, footprint_radius, existing, skip_group_id=skip_group_id):
            continue
        score = _candidate_score(
            position=clamped,
            yaw=yaw,
            role=role,
            dimensions=dimensions,
            existing=existing,
            footprint_radius=footprint_radius,
            scene_program=scene_program,
            zone_preference=zone_preference,
            skip_group_id=skip_group_id,
        )
        if score > best_score:
            best_position = clamped
            best_score = score
    if best_position is not None:
        return best_position, best_score
    clamped = _clamp_floor_position(desired_pos, dimensions)
    return clamped, _candidate_score(
        position=clamped,
        yaw=yaw,
        role=role,
        dimensions=dimensions,
        existing=existing,
        footprint_radius=footprint_radius,
        scene_program=scene_program,
        zone_preference=zone_preference,
        skip_group_id=skip_group_id,
    ) - 2.0


def _constraint_payload(constraint_type: str, *, target: str = "", relation: str = "") -> Dict[str, Any]:
    payload: Dict[str, Any] = {"type": constraint_type}
    if target:
        payload["target"] = target
    if relation:
        payload["relation"] = relation
    return payload


def _placement_entry(
    asset: Dict[str, Any],
    *,
    placement_id: str,
    position: Sequence[float],
    yaw: float,
    constraint: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "placement_id": placement_id,
        "asset_id": str(asset.get("asset_id") or ""),
        "label": asset.get("label"),
        "role": str(asset.get("role") or semantic_role_key(asset)),
        "tags": list(asset.get("tags") or []),
        "group_id": asset.get("group_id"),
        "group_layout": asset.get("group_layout"),
        "front_yaw_offset_degrees": float(asset.get("front_yaw_offset_degrees") or 0.0),
        "geometry_profile": geometry_profile_from_asset(asset),
        "constraint": constraint,
        "transform": {
            "pos": [_round3(position[0]), _round3(position[1]), _round3(position[2])],
            "rot": [0.0, _round3(yaw), 0.0],
            "scale": [1.0, 1.0, 1.0],
        },
    }


def _group_member_angles(layout_pattern: str, member_count: int) -> List[float]:
    if member_count <= 0:
        return []
    if layout_pattern == "paired_long_sides":
        if member_count == 1:
            return [180.0]
        if member_count == 2:
            return [90.0, 270.0]
        if member_count == 3:
            return [90.0, 270.0, 180.0]
        if member_count == 4:
            return [90.0, 270.0, 0.0, 180.0]
    if layout_pattern == "beside_anchor":
        return [90.0, 270.0][:member_count] or [90.0]
    if layout_pattern == "front_facing_cluster":
        span = min(36.0 * max(member_count - 1, 1), 120.0)
        start = 180.0 - (span * 0.5)
        step = span / max(member_count - 1, 1)
        return [start + (step * index) for index in range(member_count)]
    if layout_pattern == "arc":
        span = 160.0 if member_count >= 3 else 100.0
        start = 180.0 - (span * 0.5)
        step = span / max(member_count - 1, 1)
        return [start + (step * index) for index in range(member_count)]
    step = 360.0 / member_count
    return [step * index for index in range(member_count)]


def _group_anchor_candidates(group_spec: Dict[str, Any], dimensions: Dict[str, float], ordinal: int) -> List[List[float]]:
    group_type = _safe_text(group_spec.get("group_type")) or "reading_corner"
    zone_preference = _safe_text(group_spec.get("zone_preference")) or GROUP_ZONE_DEFAULTS.get(group_type, "center")
    candidates = [_zone_position(zone_preference, dimensions, ordinal)]
    if zone_preference != "center":
        candidates.append(_zone_position("center", dimensions, ordinal))
    if zone_preference not in {"edge", "corner"}:
        candidates.append(_zone_position("edge", dimensions, ordinal))
    candidates.append(_zone_position("corner", dimensions, ordinal))
    deduped: List[List[float]] = []
    seen: set[tuple[float, float]] = set()
    for candidate in candidates:
        key = (float(candidate[0]), float(candidate[2]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _group_anchor_yaw(anchor_pos: Sequence[float], group_spec: Dict[str, Any]) -> float:
    zone_preference = _safe_text(group_spec.get("zone_preference"))
    if zone_preference in {"edge", "corner", "front", "back", "left", "right"}:
        return _yaw_to_target(anchor_pos, [0.0, 0.0, 0.0])
    return 0.0


def _angle_variants(layout_pattern: str, member_count: int) -> List[List[float]]:
    base = _group_member_angles(layout_pattern, member_count)
    variants = [base]
    if base:
        variants.append(list(reversed(base)))
        variants.append([((angle + 180.0) % 360.0) for angle in base])
    deduped: List[List[float]] = []
    seen: set[tuple[float, ...]] = set()
    for variant in variants:
        key = tuple(round(value, 3) for value in variant)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(variant)
    return deduped


def _group_layout_score(anchor_pos: Sequence[float], member_positions: List[Sequence[float]], dimensions: Dict[str, float], scene_program: Dict[str, Any], symmetry: str) -> float:
    score = 0.0
    if not member_positions:
        return score
    distances = [_distance_xz(anchor_pos, member_pos) for member_pos in member_positions]
    average_distance = sum(distances) / len(distances)
    score -= sum(abs(distance - average_distance) for distance in distances)
    if symmetry in {"symmetric", "balanced"} and len(member_positions) >= 2:
        mirrored_offset = abs(sum(member_pos[0] for member_pos in member_positions))
        score -= mirrored_offset * 0.35
    if _safe_text(scene_program.get("empty_space_preference")) == "open_center":
        score -= abs(float(anchor_pos[0])) * 0.1 + abs(float(anchor_pos[2])) * 0.1
    return score


def _place_group(
    *,
    group_spec: Dict[str, Any],
    anchor_asset: Dict[str, Any],
    member_assets: List[Dict[str, Any]],
    dimensions: Dict[str, float],
    density_profile: str,
    scene_program: Dict[str, Any],
    placements: List[Dict[str, Any]],
    placement_index: int,
    ordinal: int,
) -> tuple[List[Dict[str, Any]], int, float]:
    group_id = _safe_text(group_spec.get("group_id"))
    anchor_profile = geometry_profile_from_asset(anchor_asset)
    member_profile = geometry_profile_from_asset(member_assets[0]) if member_assets else {}
    member_distance = derive_near_distance(anchor_profile, member_profile, density_profile) + 0.15 if member_assets else 0.0
    anchor_constraint_type = GROUP_CONSTRAINT_DEFAULTS.get(_safe_text(group_spec.get("group_type")), "floor")
    if _safe_text(group_spec.get("zone_preference")) == "center":
        anchor_constraint_type = "floor"

    best_bundle: tuple[List[Dict[str, Any]], float] | None = None
    for anchor_candidate in _group_anchor_candidates(group_spec, dimensions, ordinal):
        anchor_yaw = _group_anchor_yaw(anchor_candidate, group_spec)
        resolved_anchor, anchor_score = _resolve_scored_position(
            desired_pos=anchor_candidate,
            yaw=anchor_yaw,
            role=_safe_text(anchor_asset.get("role") or semantic_role_key(anchor_asset)),
            footprint_radius=float(anchor_profile.get("footprint_radius") or 0.7),
            dimensions=dimensions,
            existing=placements,
            scene_program=scene_program,
            zone_preference=_safe_text(group_spec.get("zone_preference")) or "center",
            skip_group_id=group_id,
        )
        anchor_entry = _placement_entry(
            anchor_asset,
            placement_id=f"placement_{placement_index:03d}",
            position=resolved_anchor,
            yaw=anchor_yaw,
            constraint=_constraint_payload(anchor_constraint_type),
        )
        if not member_assets:
            score = anchor_score + _focal_score(resolved_anchor, anchor_yaw, _safe_text(anchor_asset.get("role") or semantic_role_key(anchor_asset)), scene_program, dimensions)
            if best_bundle is None or score > best_bundle[1]:
                best_bundle = ([anchor_entry], score)
            continue

        for angle_variant in _angle_variants(_safe_text(group_spec.get("layout_pattern")), len(member_assets)):
            bundle = [anchor_entry]
            bundle_score = anchor_score
            member_positions: List[List[float]] = []
            local_index = placement_index + 1
            for member_asset, angle in zip(member_assets, angle_variant):
                radians = math.radians(angle)
                desired = [
                    resolved_anchor[0] + (math.sin(radians) * member_distance),
                    0.0,
                    resolved_anchor[2] + (math.cos(radians) * member_distance),
                ]
                facing_rule = _safe_text(group_spec.get("facing_rule"))
                if facing_rule in {"toward_anchor", "toward_focal_object"}:
                    desired_yaw = _yaw_to_target(desired, resolved_anchor)
                    relation = "face_to"
                elif facing_rule == "parallel":
                    desired_yaw = anchor_yaw
                    relation = "align"
                else:
                    desired_yaw = 0.0
                    relation = "near"
                resolved_member, member_score = _resolve_scored_position(
                    desired_pos=desired,
                    yaw=desired_yaw,
                    role=_safe_text(member_asset.get("role") or semantic_role_key(member_asset)),
                    footprint_radius=float(member_profile.get("footprint_radius") or 0.6),
                    dimensions=dimensions,
                    existing=placements + bundle,
                    scene_program=scene_program,
                    zone_preference=_safe_text(group_spec.get("zone_preference")) or "center",
                    anchor_pos=resolved_anchor,
                    skip_group_id=group_id,
                )
                bundle.append(
                    _placement_entry(
                        member_asset,
                        placement_id=f"placement_{local_index:03d}",
                        position=resolved_member,
                        yaw=desired_yaw,
                        constraint=_constraint_payload("near", target=str(group_spec.get("anchor_role") or ""), relation=relation),
                    )
                )
                local_index += 1
                member_positions.append(resolved_member)
                bundle_score += member_score
            bundle_score += _group_layout_score(
                resolved_anchor,
                member_positions,
                dimensions,
                scene_program,
                _safe_text(group_spec.get("symmetry")),
            )
            if best_bundle is None or bundle_score > best_bundle[1]:
                best_bundle = (bundle, bundle_score)

    if best_bundle is None:
        return [], placement_index, 0.0
    return best_bundle[0], placement_index + len(best_bundle[0]), best_bundle[1]


def _next_support_angle(role: str, index: int) -> float:
    if role in {"lamp", "plant"}:
        return [135.0, 225.0, 45.0, 315.0][index % 4]
    if role in SEATING_ROLES:
        return [180.0, 135.0, 225.0, 90.0][index % 4]
    return [45.0, 135.0, 225.0, 315.0][index % 4]


def _fallback_role_candidates(role: str, dimensions: Dict[str, float], ordinal: int) -> List[List[float]]:
    primary_zone = "edge" if role in EDGE_BIASED_ROLES else "corner" if role in {"lamp", "plant", "decor"} else "center"
    candidates = [_zone_position(primary_zone, dimensions, ordinal)]
    if primary_zone != "center":
        candidates.append(_zone_position("center", dimensions, ordinal))
    candidates.append(_zone_position("corner", dimensions, ordinal))
    candidates.append(_zone_position("edge", dimensions, ordinal))
    return candidates


def _group_selected_assets(selected_assets: List[Dict[str, Any]]) -> tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    grouped_assets: Dict[str, List[Dict[str, Any]]] = {}
    ungrouped_assets: List[Dict[str, Any]] = []
    for asset in selected_assets:
        group_id = _safe_text(asset.get("group_id"))
        if group_id:
            grouped_assets.setdefault(group_id, []).append(asset)
        else:
            ungrouped_assets.append(asset)
    return grouped_assets, ungrouped_assets


def _place_grouped_assets(
    *,
    group_specs: List[Dict[str, Any]],
    grouped_assets: Dict[str, List[Dict[str, Any]]],
    room_dimensions: Dict[str, float],
    density_profile: str,
    scene_program: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], int, List[Dict[str, Any]], List[float]]:
    placements: List[Dict[str, Any]] = []
    placed_roles: Dict[str, List[Dict[str, Any]]] = {}
    placement_index = 0
    group_summaries: List[Dict[str, Any]] = []
    group_scores: List[float] = []
    for ordinal, group_spec in enumerate(group_specs):
        group_id = _safe_text(group_spec.get("group_id"))
        members = grouped_assets.get(group_id, [])
        if not members:
            continue
        anchor_assets = [asset for asset in members if _safe_text(asset.get("group_role")) == "anchor"]
        if not anchor_assets:
            continue
        member_assets = [asset for asset in members if _safe_text(asset.get("group_role")) == "member"]
        group_placements, placement_index, group_score = _place_group(
            group_spec=group_spec,
            anchor_asset=anchor_assets[0],
            member_assets=member_assets,
            dimensions=room_dimensions,
            density_profile=density_profile,
            scene_program=scene_program,
            placements=placements,
            placement_index=placement_index,
            ordinal=ordinal,
        )
        placements.extend(group_placements)
        for placement in group_placements:
            placed_roles.setdefault(_safe_text(placement.get("role")), []).append(placement)
        group_summaries.append(
            {
                "group_id": group_id,
                "group_type": _safe_text(group_spec.get("group_type")),
                "layout_pattern": _safe_text(group_spec.get("layout_pattern")),
                "member_count": len(member_assets),
            }
        )
        group_scores.append(group_score)
    return placements, placed_roles, placement_index, group_summaries, group_scores


def _relation_candidates(
    *,
    role: str,
    profile: Dict[str, Any],
    role_relations: List[Dict[str, str]],
    placed_roles: Dict[str, List[Dict[str, Any]]],
    room_dimensions: Dict[str, float],
    density_profile: str,
    support_counts: Dict[str, int],
    ordinal: int,
) -> List[tuple[List[float], float, str, str, str]]:
    candidates: List[tuple[List[float], float, str, str, str]] = []
    for relation in role_relations:
        candidate_target_role = _safe_text(relation.get("target_role"))
        relation_type = _safe_text(relation.get("relation"))
        if candidate_target_role == "room":
            zone = "edge" if relation_type == "edge" else "center"
            candidates.append((_zone_position(zone, room_dimensions, ordinal), 0.0, zone, "", relation_type))
            continue
        target_placements = placed_roles.get(candidate_target_role) or []
        if not target_placements:
            continue
        target_placement = target_placements[0]
        target_pos = ((target_placement.get("transform") or {}).get("pos")) or [0.0, 0.0, 0.0]
        target_profile = target_placement.get("geometry_profile") if isinstance(target_placement.get("geometry_profile"), dict) else {}
        distance = derive_near_distance(profile, target_profile, density_profile)
        support_index = support_counts.get(role, 0)
        angle = _next_support_angle(role, support_index)
        support_counts[role] = support_index + 1
        radians = math.radians(angle)
        desired_pos = [
            float(target_pos[0]) + (math.sin(radians) * distance),
            0.0,
            float(target_pos[2]) + (math.cos(radians) * distance),
        ]
        desired_yaw = _yaw_to_target(desired_pos, target_pos) if relation_type == "face_to" else 0.0
        zone = "edge" if role in EDGE_BIASED_ROLES and relation_type != "face_to" else "center"
        candidates.append((desired_pos, desired_yaw, zone, candidate_target_role, relation_type))
    if candidates:
        return candidates
    for fallback in _fallback_role_candidates(role, room_dimensions, ordinal):
        fallback_yaw = _yaw_to_target(fallback, [0.0, 0.0, 0.0]) if role in EDGE_BIASED_ROLES else 0.0
        candidates.append((fallback, fallback_yaw, "edge" if role in EDGE_BIASED_ROLES else "center", "", ""))
    return candidates


def _best_placement_choice(
    *,
    asset: Dict[str, Any],
    role: str,
    profile: Dict[str, Any],
    placement_index: int,
    candidates: List[tuple[List[float], float, str, str, str]],
    room_dimensions: Dict[str, float],
    placements: List[Dict[str, Any]],
    scene_program: Dict[str, Any],
) -> tuple[Dict[str, Any], float] | None:
    best_choice: tuple[Dict[str, Any], float] | None = None
    for desired_pos, desired_yaw, zone_preference, target_role, relation_type in candidates:
        resolved_pos, candidate_score = _resolve_scored_position(
            desired_pos=desired_pos,
            yaw=desired_yaw,
            role=role,
            footprint_radius=float(profile.get("footprint_radius") or 0.6),
            dimensions=room_dimensions,
            existing=placements,
            scene_program=scene_program,
            zone_preference=zone_preference,
        )
        constraint_type = "near" if target_role else ("against_wall" if zone_preference == "edge" and role in EDGE_BIASED_ROLES else "floor")
        placement = _placement_entry(
            asset,
            placement_id=f"placement_{placement_index:03d}",
            position=resolved_pos,
            yaw=desired_yaw,
            constraint=_constraint_payload(
                constraint_type,
                target=target_role,
                relation=relation_type if relation_type in {"near", "face_to", "align"} else "",
            ),
        )
        if best_choice is None or candidate_score > best_choice[1]:
            best_choice = (placement, candidate_score)
    return best_choice


def solve_scene_layout(
    selected_assets: List[Dict[str, Any]],
    *,
    scene_program: Dict[str, Any],
    placement_intent: Dict[str, Any],
    room_dimensions: Dict[str, float],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    density_profile = _safe_text(placement_intent.get("density_profile")) or "normal"
    group_specs = [
        dict(group)
        for group in scene_program.get("groups") or []
        if isinstance(group, dict) and _safe_text(group.get("group_id"))
    ]
    grouped_assets, ungrouped_assets = _group_selected_assets(selected_assets)
    placements, placed_roles, placement_index, group_summaries, group_scores = _place_grouped_assets(
        group_specs=group_specs,
        grouped_assets=grouped_assets,
        room_dimensions=room_dimensions,
        density_profile=density_profile,
        scene_program=scene_program,
    )

    relations = _scene_graph_edges(scene_program, placement_intent)
    support_counts: Dict[str, int] = {}
    relation_score_total = 0.0
    for ordinal, asset in enumerate(ungrouped_assets):
        role = _safe_text(asset.get("role") or semantic_role_key(asset))
        profile = geometry_profile_from_asset(asset)
        role_relations = [relation for relation in relations if _safe_text(relation.get("source_role")) == role]
        candidates = _relation_candidates(
            role=role,
            profile=profile,
            role_relations=role_relations,
            placed_roles=placed_roles,
            room_dimensions=room_dimensions,
            density_profile=density_profile,
            support_counts=support_counts,
            ordinal=ordinal,
        )
        best_choice = _best_placement_choice(
            asset=asset,
            role=role,
            profile=profile,
            placement_index=placement_index,
            candidates=candidates,
            room_dimensions=room_dimensions,
            placements=placements,
            scene_program=scene_program,
        )
        if best_choice is None:
            continue
        placement, candidate_score = best_choice
        placements.append(placement)
        placed_roles.setdefault(role, []).append(placement)
        relation_score_total += candidate_score
        placement_index += 1

    layout_program = {
        "backend": "scene_graph_solver",
        "solver_strategy": "relation_scored_search",
        "density_profile": density_profile,
        "scene_graph_edge_count": len(relations),
        "group_count": len(group_summaries),
        "groups": group_summaries,
        "placed_count": len(placements),
        "group_score": round(sum(group_scores), 3),
        "relation_score": round(relation_score_total, 3),
    }
    return placements, layout_program
