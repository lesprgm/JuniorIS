from __future__ import annotations

import hashlib
import json
import math
import pathlib
from functools import lru_cache
from collections import Counter
from typing import Any, Dict, List, Tuple

from src.catalog.pack_registry import load_pack_registry
from src.compilation.phase0_placement import (
    _apply_face_to_corrections,
    _compiled_input,
    _constraint_type,
    _constraint_relation,
    _face_alignment_score,
    _repair_overlaps,
    _resolve_face_to_target,
    _safe_vec3,
)
from src.catalog.style_material_pool import load_style_material_pool_by_id
from src.planning.assets import collect_assets, load_planner_pool
from src.runtime.safe_spawn import find_safe_spawn
from src.catalog.stylekit_registry import load_stylekit_registry
from src.selection.substitution import resolve_asset_or_substitute
from src.world.validation import validate_worldspec
from src.world.templates import build_template_geometry


def _stable_json_payload(data: Dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _build_world_id(worldspec: Dict[str, Any]) -> str:
    digest = hashlib.sha256(_stable_json_payload(worldspec).encode("utf-8")).hexdigest()
    return f"world_{digest[:10]}"


def _compile_failure(
    *,
    world_id: str | None,
    errors: List[Dict[str, Any]],
    teleportable_surfaces: int = 0,
    phase0_data: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    result = {
        "ok": False,
        "world_id": world_id,
        "phase0_artifact": None,
        "teleportable_surfaces": teleportable_surfaces,
        "errors": errors,
        "safe_spawn": None,
    }
    if phase0_data is not None:
        result["phase0_data"] = phase0_data
    return result


@lru_cache(maxsize=1)
def _approved_planner_asset_ids() -> frozenset[str]:  # quick check lookup to prevent malicious or non-indexed assets from slipping past LLM validation
    return frozenset(
        str(asset.get("asset_id", "")).strip()
        for asset in load_planner_pool()
        if isinstance(asset, dict) and asset.get("asset_id")
    )


def _substitution_entry(  # structures the normalized audit record for exactly why an asset was chosen by the engine
    *,
    requested_asset_id: str,
    resolved_asset_id: str,
    resolution_type: str,
    reason: str,
    resolution: Dict[str, Any],
    coherence_checks: Dict[str, Any],
    rejected_counts: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "requested_asset_id": requested_asset_id,
        "resolved_asset_id": resolved_asset_id,
        "resolution_type": resolution_type,
        "reason": reason,
        "coherence_checks": coherence_checks,
        "rejected_candidate_counts": rejected_counts,
        "alternatives": resolution.get("alternatives", []),
        "rationale": resolution.get("rationale", []),
        "selection_backend": resolution.get("selection_backend"),
        "semantic_failure_reason": resolution.get("semantic_failure_reason"),
    }


def _available_assets(pack_ids: List[str]) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    registry = load_pack_registry()
    selected_pack_ids = pack_ids or sorted(registry.packs_by_id.keys())
    available_assets = collect_assets(selected_pack_ids, registry) or collect_assets([], registry)
    return available_assets, {
        str(asset.get("asset_id", "")): asset
        for asset in available_assets
        if isinstance(asset, dict) and asset.get("asset_id")
    }


def _compile_raw_placement(
    *,
    index: int,
    placement: Dict[str, Any],
    pack_ids: List[str],
    registry: Any,
    room_theme: Dict[str, Any],
    available_by_id: Dict[str, Dict[str, Any]],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    transform = placement.get("transform") if isinstance(placement.get("transform"), dict) else {}
    requested_asset_id = str(placement.get("asset_id", "unknown_asset"))
    requested_tags = placement.get("tags")
    requested_meta = {
        "tags": requested_tags if isinstance(requested_tags, list) else [],
        "allow_passthrough_exact": requested_asset_id in _approved_planner_asset_ids(),
        "role": placement.get("role"),
        "label": placement.get("label"),
        "category": placement.get("category"),
        "style_tags": placement.get("style_tags"),
        "era_tags": placement.get("era_tags"),
        "color_tags": placement.get("color_tags"),
        "visual_style": placement.get("visual_style"),
        "poly_style": placement.get("poly_style"),
    }
    resolution = resolve_asset_or_substitute(
        requested_asset_id=requested_asset_id,
        requested_tags=requested_meta["tags"],
        pack_ids=pack_ids,
        registry=registry,
        requested_meta=requested_meta,
        room_theme=room_theme,
    )
    return {
        "index": index,
        "placement": placement,
        "requested_asset_id": requested_asset_id,
        "requested_tags": requested_meta["tags"],
        "position": _safe_vec3(transform.get("pos"), [0.0, 0.0, 0.0]),
        "rotation": _safe_vec3(transform.get("rot"), [0.0, 0.0, 0.0]),
        "scale": _safe_vec3(transform.get("scale"), [1.0, 1.0, 1.0]),
        "resolved_asset_id": str(resolution.get("resolved_asset_id", requested_asset_id)),
        "resolution_type": str(resolution.get("resolution_type", "exact")),
        "reason": str(resolution.get("reason", "asset_found")),
        "asset_record": available_by_id.get(str(resolution.get("resolved_asset_id", requested_asset_id)), placement),
    }, resolution


def _placement_report(
    *,
    compiled: List[Dict[str, Any]],
    placement_mode: str,
    resolution_counts: Counter[str],
    substitution_entries: List[Dict[str, Any]],
    rejected_candidate_counts: Counter[str],
    overlap_repair: Dict[str, Any],
) -> Dict[str, Any]:
    overlap_pairs = _floor_overlap_pairs(compiled)
    relation_failures: List[Dict[str, Any]] = []
    face_to_scores: Dict[str, float] = {}
    grouped_positions: Dict[str, List[Tuple[str, List[float], float]]] = {}

    for left_index, left in enumerate(compiled):
        if _constraint_type(left) in {"wall", "surface", "ceiling"}:
            continue
        left_radius = float(((left.get("geometry_profile") or {}).get("footprint_radius")) or 0.0)
        left_pos = (left.get("transform") or {}).get("pos") or [0.0, 0.0, 0.0]
        group_id = str(left.get("group_id") or "").strip()
        if group_id:
            grouped_positions.setdefault(group_id, []).append(
                (
                    str(left.get("placement_id") or ""),
                    list(left_pos),
                    left_radius,
                )
            )
        if _constraint_relation(left) != "face_to":
            continue
        target = _resolve_face_to_target(left, compiled)
        if target is None:
            continue
        score = _face_alignment_score(left, target)
        if score is not None and score < 0.75:
            relation_failures.append(
                {
                    "placement_id": str(left.get("placement_id") or ""),
                    "asset_id": str(left.get("asset_id") or ""),
                    "target_asset_id": str(target.get("asset_id") or ""),
                    "relation": "face_to",
                    "alignment_score": round(score, 3),
                }
            )
        if score is not None:
            face_to_scores[str(left.get("placement_id") or "")] = round(score, 3)

    group_layout_failures: List[Dict[str, Any]] = []
    for group_id, entries in grouped_positions.items():
        for left_index, (left_id, left_pos, left_radius) in enumerate(entries):
            for right_id, right_pos, right_radius in entries[left_index + 1 :]:
                minimum_distance = left_radius + right_radius + 0.05
                actual_distance = math.dist((left_pos[0], left_pos[2]), (right_pos[0], right_pos[2]))
                if actual_distance < minimum_distance:
                    group_layout_failures.append(
                        {
                            "group_id": group_id,
                            "left_placement_id": left_id,
                            "right_placement_id": right_id,
                            "distance": round(actual_distance, 3),
                            "threshold": round(minimum_distance, 3),
                        }
                    )

    return {
        "total_placements": len(compiled),
        "resolution_counts": dict(resolution_counts),
        "substitution_count": len(substitution_entries),
        "substitutions": substitution_entries,
        "rejected_candidate_counts": dict(rejected_candidate_counts),
        "placement_execution": {
            "backend": placement_mode,
            "attempt_count": 1,
            "best_attempt": 0,
            "placed_count": len(compiled),
            "skipped_count": 0,
            "fallback_count": 0,
        },
        "placement_mode": placement_mode,
        "placement_audit": {
            "overlap_count": len(overlap_pairs),
            "overlap_pairs": overlap_pairs,
            "repair_passes": int(overlap_repair["repair_passes"]),
            "repaired_pairs": int(overlap_repair["repaired_pairs"]),
            "group_repairs_applied": int(overlap_repair.get("group_repairs_applied") or 0),
            "group_members_adjusted": int(overlap_repair.get("group_members_adjusted") or 0),
            "relation_failure_count": len(relation_failures),
            "relation_failures": relation_failures,
            "group_layout_failure_count": len(group_layout_failures),
            "group_layout_failures": group_layout_failures,
            "face_to_score_by_placement": face_to_scores,
        },
    }


def _floor_overlap_pairs(compiled: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for left_index, left in enumerate(compiled):
        if _constraint_type(left) in {"wall", "surface", "ceiling"}:
            continue
        left_radius = float(((left.get("geometry_profile") or {}).get("footprint_radius")) or 0.0)
        left_pos = (left.get("transform") or {}).get("pos") or [0.0, 0.0, 0.0]
        if left_radius <= 0.0:
            continue
        for right in compiled[left_index + 1 :]:
            if _constraint_type(right) in {"wall", "surface", "ceiling"}:
                continue
            right_radius = float(((right.get("geometry_profile") or {}).get("footprint_radius")) or 0.0)
            if right_radius <= 0.0:
                continue
            right_pos = (right.get("transform") or {}).get("pos") or [0.0, 0.0, 0.0]
            distance = math.dist((left_pos[0], left_pos[2]), (right_pos[0], right_pos[2]))
            if distance < (left_radius + right_radius):
                pairs.append(
                    {
                        "left": str(left.get("asset_id") or ""),
                        "right": str(right.get("asset_id") or ""),
                        "left_placement_id": str(left.get("placement_id") or ""),
                        "right_placement_id": str(right.get("placement_id") or ""),
                        "distance": round(distance, 3),
                        "threshold": round(left_radius + right_radius, 3),
                    }
                )
    return pairs


def _overlap_trim_priority(placement: Dict[str, Any]) -> int:
    placement_id = str(placement.get("placement_id") or "")
    role = str(placement.get("role") or "").strip().lower()
    if placement_id.startswith("optional_"):
        return 0
    if role in {"decor", "plant", "textile"}:
        return 1
    return 99


def _trim_residual_overlap_clutter(compiled: List[Dict[str, Any]], dimensions: Dict[str, float]) -> Dict[str, Any]:
    aggregate_repair = {"repair_passes": 0, "repaired_pairs": 0, "group_repairs_applied": 0, "group_members_adjusted": 0}
    for _ in range(8):
        pairs = _floor_overlap_pairs(compiled)
        candidate_ids = {pid for pair in pairs for pid in (pair["left_placement_id"], pair["right_placement_id"])}
        candidates = [
            placement for placement in compiled
            if str(placement.get("placement_id") or "") in candidate_ids and _overlap_trim_priority(placement) < 99
        ]
        if not pairs or not candidates:
            break
        victim = min(candidates, key=lambda placement: (
            _overlap_trim_priority(placement),
            -sum(str(placement.get("placement_id") or "") in {pair["left_placement_id"], pair["right_placement_id"]} for pair in pairs),
        ))
        compiled.remove(victim)
        repair = _repair_overlaps(compiled, dimensions)
        for key in aggregate_repair:
            aggregate_repair[key] += int(repair.get(key) or 0)
    return aggregate_repair


def _compile_placements(
    raw_placements: List[Dict[str, Any]],
    dimensions: Dict[str, float],
    pack_ids: List[str],
    room_theme: Dict[str, Any],
    seed: int,
    placement_intent: Dict[str, Any] | None = None,
    placement_mode: str = "scene_graph_solver",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    compiled_inputs: List[Dict[str, Any]] = []
    substitution_entries: List[Dict[str, Any]] = []
    resolution_counts: Counter[str] = Counter()
    rejected_candidate_counts: Counter[str] = Counter()

    registry = load_pack_registry()
    if not pack_ids:
        pack_ids = sorted(registry.packs_by_id.keys())
    _, available_by_id = _available_assets(pack_ids)

    for index, placement in enumerate(raw_placements):
        if not isinstance(placement, dict):
            continue
        compiled_source, resolution = _compile_raw_placement(
            index=index,
            placement=placement,
            pack_ids=pack_ids,
            registry=registry,
            room_theme=room_theme,
            available_by_id=available_by_id,
        )
        resolved_asset_id = compiled_source["resolved_asset_id"]
        resolution_type = compiled_source["resolution_type"]
        reason = compiled_source["reason"]
        coherence_checks = resolution.get("coherence_checks", {})
        rejected_counts = resolution.get("rejected_candidate_counts", {})
        if isinstance(rejected_counts, dict):
            rejected_candidate_counts.update(
                {
                    str(key): int(value)
                    for key, value in rejected_counts.items()
                    if isinstance(value, int)
                }
            )

        resolution_counts[resolution_type] += 1
        if resolution_type in {"substitute", "placeholder"}:
            substitution_entries.append(
                _substitution_entry(
                    requested_asset_id=compiled_source["requested_asset_id"],
                    resolved_asset_id=resolved_asset_id,
                    resolution_type=resolution_type,
                    reason=reason,
                    resolution=resolution,
                    coherence_checks=coherence_checks,
                    rejected_counts=rejected_counts,
                )
            )

        compiled_inputs.append(
            _compiled_input(
                index=compiled_source["index"],
                placement=compiled_source["placement"],
                requested_asset_id=compiled_source["requested_asset_id"],
                resolved_asset_id=resolved_asset_id,
                resolution_type=resolution_type,
                reason=reason,
                requested_tags=compiled_source["requested_tags"],
                position=compiled_source["position"],
                rotation=compiled_source["rotation"],
                scale=compiled_source["scale"],
                dimensions=dimensions,
                asset_record=compiled_source["asset_record"],
            )
        )

    _apply_face_to_corrections(compiled_inputs)
    overlap_repair = _repair_overlaps(compiled_inputs, dimensions)
    trim_repair = _trim_residual_overlap_clutter(compiled_inputs, dimensions)
    for key in ("repair_passes", "repaired_pairs", "group_repairs_applied", "group_members_adjusted"):
        overlap_repair[key] = int(overlap_repair.get(key) or 0) + int(trim_repair.get(key) or 0)
    _apply_face_to_corrections(compiled_inputs)
    del seed, placement_intent
    compiled = compiled_inputs
    return compiled, _placement_report(
        compiled=compiled,
        placement_mode=placement_mode,
        resolution_counts=resolution_counts,
        substitution_entries=substitution_entries,
        rejected_candidate_counts=rejected_candidate_counts,
        overlap_repair=overlap_repair,
    )


def _derive_room_theme(worldspec: Dict[str, Any]) -> Dict[str, Any]:
    style_tags: List[str] = []
    mood_tags: List[str] = []
    creative_tags: List[str] = []
    style_descriptors: List[str] = []
    negative_constraints: List[str] = []
    stylekit_id = worldspec.get("stylekit_id")
    if isinstance(stylekit_id, str) and stylekit_id:
        style_registry = load_stylekit_registry()
        stylekit = style_registry.get_stylekit(stylekit_id)
        if isinstance(stylekit, dict):
            raw_tags = stylekit.get("tags", [])
            if isinstance(raw_tags, list):
                style_tags = [str(tag).strip().lower() for tag in raw_tags if isinstance(tag, str) and str(tag).strip()]

    scene_context = worldspec.get("scene_context") if isinstance(worldspec.get("scene_context"), dict) else {}
    for key, bucket in (
        ("mood_tags", mood_tags),
        ("creative_tags", creative_tags),
        ("style_descriptors", style_descriptors),
        ("negative_constraints", negative_constraints),
    ):
        values = scene_context.get(key) if isinstance(scene_context.get(key), list) else []
        bucket.extend(str(value).strip().lower() for value in values if isinstance(value, str) and str(value).strip())

    return {
        "style_tags": list(dict.fromkeys(style_tags + creative_tags + style_descriptors)),
        "era_tags": [],
        "color_tags": [],
        "mood_tags": list(dict.fromkeys(mood_tags)),
        "negative_constraints": list(dict.fromkeys(negative_constraints)),
        "concept_label": str(scene_context.get("concept_label") or "").strip().lower(),
        "scene_type": str(scene_context.get("scene_type") or "").strip().lower(),
    }


def _count_teleportable_surfaces(template: Dict[str, Any]) -> int:
    count = 0
    for node in template.get("nodes", []):
        if node.get("teleportable") is True:
            count += 1
    return count


def _surface_role_for_template_node(node_id: str) -> str | None:
    token = str(node_id or "").strip().lower()
    if token == "floor":
        return "floor"
    if token == "ceiling":
        return "ceiling"
    if token.startswith("wall_"):
        return "wall"
    return None


def _preview_color_hex(material_record: Dict[str, Any]) -> str | None:
    rgba = material_record.get("preview_color_rgba")
    if not isinstance(rgba, dict):
        return None
    channels: List[int] = []
    for key in ("r", "g", "b"):
        value = rgba.get(key)
        if not isinstance(value, (int, float)):
            return None
        channels.append(max(0, min(255, int(round(float(value) * 255.0)))))
    return "#{:02x}{:02x}{:02x}".format(*channels)


def _surface_material_payload(
    surface_role: str,
    material_id: str,
    material_records_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    record = material_records_by_id.get(material_id) if material_id else None
    if not isinstance(record, dict):
        return {"surface_role": surface_role, "material_id": material_id}
    return {
        "surface_role": surface_role,
        "material_id": material_id,
        "display_name": record.get("display_name") or record.get("material_name"),
        "material_path": record.get("material_path"),
        "preview_texture_asset_path": record.get("preview_texture_asset_path"),
        "preview_color_hex": _preview_color_hex(record),
        "surface_roles": list(record.get("surface_roles") or []),
        "material_family_tags": list(record.get("material_family_tags") or []),
        "texture_tags": list(record.get("texture_tags") or []),
        "finish_tags": list(record.get("finish_tags") or []),
        "visual_description": record.get("visual_description"),
    }


def _apply_shell_surface_materials(
    template: Dict[str, Any],
    surface_material_selection: Dict[str, Any] | None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    selection = dict(surface_material_selection) if isinstance(surface_material_selection, dict) else {}
    if not selection:
        return template, {}

    material_records_by_id = load_style_material_pool_by_id()
    updated_template = dict(template)
    updated_nodes: List[Dict[str, Any]] = []
    shell_bindings: Dict[str, Any] = {}

    for node in template.get("nodes", []):
        if not isinstance(node, dict):
            continue
        updated_node = dict(node)
        surface_role = _surface_role_for_template_node(str(node.get("id") or ""))
        if surface_role:
            material_id = str(selection.get(surface_role) or "").strip()
            if material_id:
                payload = _surface_material_payload(surface_role, material_id, material_records_by_id)
                updated_node["surface_material"] = payload
                shell_bindings[str(node.get("id") or "")] = payload
        updated_nodes.append(updated_node)
    updated_template["nodes"] = updated_nodes
    return updated_template, shell_bindings


def compile_phase0(  # main compiler entry-point: worldspec -> placed, resolved, overlap-repaired phase0 artifact
    worldspec: Dict[str, Any],
    build_root: str | pathlib.Path = "build",
    write_artifact: bool = True,
) -> Dict[str, Any]:
    validation = validate_worldspec(worldspec)
    if not validation["ok"]:
        return _compile_failure(world_id=None, errors=validation["errors"])

    template_id = str(worldspec.get("template_id", ""))
    try:
        template = build_template_geometry(template_id)
    except ValueError as exc:
        return _compile_failure(world_id=None, errors=[{"path": "$.template_id", "message": str(exc)}])

    template, shell_material_bindings = _apply_shell_surface_materials(
        template,
        worldspec.get("surface_material_selection"),
    )

    dimensions = template["dimensions"]
    raw_placements = worldspec.get("placements", [])
    raw_placements = raw_placements if isinstance(raw_placements, list) else []
    pack_ids = worldspec.get("pack_ids")
    pack_ids = pack_ids if isinstance(pack_ids, list) else []
    room_theme = _derive_room_theme(worldspec)
    planner_policy = worldspec.get("planner_policy") if isinstance(worldspec.get("planner_policy"), dict) else {}
    placement_mode = str(planner_policy.get("placement_mode") or "scene_graph_solver")
    placement_intent = worldspec.get("placement_intent") if isinstance(worldspec.get("placement_intent"), dict) else {}
    placement_plan = worldspec.get("placement_plan") if isinstance(worldspec.get("placement_plan"), dict) else {}
    placements, substitution_report = _compile_placements(
        raw_placements,
        dimensions,
        pack_ids,
        room_theme,
        int(worldspec.get("seed", 0)),
        placement_intent=placement_intent,
        placement_mode=placement_mode,
    )

    world_id = _build_world_id(worldspec)
    phase0_data = {
        "phase": "phase0",
        "world_id": world_id,
        "worldspec_version": worldspec.get("worldspec_version"),
        "template": template,
        "surface_material_selection": worldspec.get("surface_material_selection") if isinstance(worldspec.get("surface_material_selection"), dict) else {},
        "shell_material_bindings": shell_material_bindings,
        "placements": placements,
        "constraints": {
            "floor_anchored_only": False,
            "stacking_enabled": False,
            "placement_constraints_enabled": True,
        },
        "placement_policy": {
            "intent": placement_intent,
            "plan": placement_plan,
            "placement_mode": placement_mode,
            "placed_count": len(placements),
        },
        "substitution_report": substitution_report,
    }

    overlap_count = int(
        ((substitution_report.get("placement_audit") or {}).get("overlap_count"))
        or 0
    )
    if overlap_count > 0:
        return _compile_failure(
            world_id=world_id,
            teleportable_surfaces=_count_teleportable_surfaces(template),
            errors=[
                {
                    "path": "$.placements",
                    "message": "Unresolved floor-object overlaps remain after placement repair.",
                }
            ],
            phase0_data=phase0_data,
        )

    spawn_result = find_safe_spawn(phase0_data)
    if not spawn_result["ok"]:
        return _compile_failure(
            world_id=world_id,
            teleportable_surfaces=_count_teleportable_surfaces(template),
            errors=[{"path": "$.safe_spawn", "message": spawn_result["reason"]}],
            phase0_data=phase0_data,
        )

    phase0_data["safe_spawn"] = spawn_result["spawn"]
    phase0_data["safe_spawn_meta"] = {
        "attempts": spawn_result["attempts"],
        "player_capsule_height": 1.70,
        "player_capsule_radius": 0.25,
    }

    artifact_path = None
    if write_artifact:
        output_path = pathlib.Path(build_root) / world_id / "phase0.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(phase0_data, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        artifact_path = str(output_path)

    return {
        "ok": True,
        "world_id": world_id,
        "phase0_artifact": artifact_path,
        "teleportable_surfaces": _count_teleportable_surfaces(template),
        "errors": [],
        "phase0_data": phase0_data,
        "safe_spawn": spawn_result["spawn"],
    }
