from __future__ import annotations

import hashlib
import json
import pathlib
from collections import Counter
from typing import Any, Dict, List, Tuple

from src.pack_registry import load_pack_registry
from src.safe_spawn import find_safe_spawn
from src.stylekit_registry import load_stylekit_registry
from src.substitution import resolve_asset_or_substitute
from src.validate_worldspec import validate_worldspec
from src.world_templates import build_template_geometry


def _stable_json_payload(data: Dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _build_world_id(worldspec: Dict[str, Any]) -> str:
    digest = hashlib.sha256(_stable_json_payload(worldspec).encode("utf-8")).hexdigest()
    return f"world_{digest[:10]}"


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


def _compile_placements(
    raw_placements: List[Dict[str, Any]],
    dimensions: Dict[str, float],
    pack_ids: List[str],
    room_theme: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    compiled: List[Dict[str, Any]] = []
    substitution_entries: List[Dict[str, Any]] = []
    resolution_counts: Counter[str] = Counter()
    rejected_candidate_counts: Counter[str] = Counter()

    registry = load_pack_registry()
    if not pack_ids:
        pack_ids = sorted(registry.packs_by_id.keys())

    for index, placement in enumerate(raw_placements):
        if not isinstance(placement, dict):
            continue

        transform = placement.get("transform")
        transform = transform if isinstance(transform, dict) else {}

        position = _safe_vec3(transform.get("pos"), [0.0, 0.0, 0.0])
        rotation = _safe_vec3(transform.get("rot"), [0.0, 0.0, 0.0])
        scale = _safe_vec3(transform.get("scale"), [1.0, 1.0, 1.0])

        requested_asset_id = str(placement.get("asset_id", "unknown_asset"))
        requested_tags = placement.get("tags")
        requested_meta = {
            "tags": requested_tags if isinstance(requested_tags, list) else [],
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

        resolved_asset_id = str(resolution.get("resolved_asset_id", requested_asset_id))
        resolution_type = str(resolution.get("resolution_type", "exact"))
        reason = str(resolution.get("reason", "asset_found"))
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
                {
                    "requested_asset_id": requested_asset_id,
                    "resolved_asset_id": resolved_asset_id,
                    "resolution_type": resolution_type,
                    "reason": reason,
                    "coherence_checks": coherence_checks,
                    "rejected_candidate_counts": rejected_counts,
                }
            )

        compiled.append(
            {
                "placement_id": f"placement_{index:03d}",
                "asset_id": resolved_asset_id,
                "requested_asset_id": requested_asset_id,
                "resolution_type": resolution_type,
                "substitution_reason": reason,
                "mode": "placeholder" if resolution_type == "placeholder" else "asset",
                "transform": {
                    "pos": _clamp_floor_position(position, dimensions),
                    "rot": [round(rotation[0], 3), round(rotation[1], 3), round(rotation[2], 3)],
                    "scale": [round(scale[0], 3), round(scale[1], 3), round(scale[2], 3)],
                },
            }
        )

    report = {
        "total_placements": len(compiled),
        "resolution_counts": dict(resolution_counts),
        "substitution_count": len(substitution_entries),
        "substitutions": substitution_entries,
        "rejected_candidate_counts": dict(rejected_candidate_counts),
    }

    return compiled, report


def _derive_room_theme(worldspec: Dict[str, Any]) -> Dict[str, Any]:
    style_tags: List[str] = []
    stylekit_id = worldspec.get("stylekit_id")
    if isinstance(stylekit_id, str) and stylekit_id:
        style_registry = load_stylekit_registry()
        stylekit = style_registry.get_stylekit(stylekit_id)
        if isinstance(stylekit, dict):
            raw_tags = stylekit.get("tags", [])
            if isinstance(raw_tags, list):
                style_tags = [str(tag).strip().lower() for tag in raw_tags if isinstance(tag, str) and str(tag).strip()]

    return {
        "style_tags": style_tags,
        "era_tags": [],
        "color_tags": [],
    }


def _count_teleportable_surfaces(template: Dict[str, Any]) -> int:
    count = 0
    for node in template.get("nodes", []):
        if node.get("teleportable") is True:
            count += 1
    return count


def compile_phase0(
    worldspec: Dict[str, Any],
    build_root: str | pathlib.Path = "build",
    write_artifact: bool = True,
) -> Dict[str, Any]:
    validation = validate_worldspec(worldspec)
    if not validation["ok"]:
        return {
            "ok": False,
            "world_id": None,
            "phase0_artifact": None,
            "teleportable_surfaces": 0,
            "errors": validation["errors"],
            "safe_spawn": None,
        }

    template_id = str(worldspec.get("template_id", ""))
    try:
        template = build_template_geometry(template_id)
    except ValueError as exc:
        return {
            "ok": False,
            "world_id": None,
            "phase0_artifact": None,
            "teleportable_surfaces": 0,
            "errors": [{"path": "$.template_id", "message": str(exc)}],
            "safe_spawn": None,
        }

    dimensions = template["dimensions"]
    raw_placements = worldspec.get("placements", [])
    raw_placements = raw_placements if isinstance(raw_placements, list) else []
    pack_ids = worldspec.get("pack_ids")
    pack_ids = pack_ids if isinstance(pack_ids, list) else []
    room_theme = _derive_room_theme(worldspec)
    placements, substitution_report = _compile_placements(raw_placements, dimensions, pack_ids, room_theme)

    world_id = _build_world_id(worldspec)
    phase0_data = {
        "phase": "phase0",
        "world_id": world_id,
        "worldspec_version": worldspec.get("worldspec_version"),
        "template": template,
        "placements": placements,
        "constraints": {
            "floor_anchored_only": True,
            "stacking_enabled": False,
        },
        "substitution_report": substitution_report,
    }

    spawn_result = find_safe_spawn(phase0_data)
    if not spawn_result["ok"]:
        return {
            "ok": False,
            "world_id": world_id,
            "phase0_artifact": None,
            "teleportable_surfaces": _count_teleportable_surfaces(template),
            "errors": [
                {
                    "path": "$.safe_spawn",
                    "message": spawn_result["reason"],
                }
            ],
            "phase0_data": phase0_data,
            "safe_spawn": None,
        }

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
