from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any, Dict, List

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
) -> List[Dict[str, Any]]:
    compiled: List[Dict[str, Any]] = []

    for index, placement in enumerate(raw_placements):
        if not isinstance(placement, dict):
            continue

        transform = placement.get("transform")
        transform = transform if isinstance(transform, dict) else {}

        position = _safe_vec3(transform.get("pos"), [0.0, 0.0, 0.0])
        rotation = _safe_vec3(transform.get("rot"), [0.0, 0.0, 0.0])
        scale = _safe_vec3(transform.get("scale"), [1.0, 1.0, 1.0])

        compiled.append(
            {
                "placement_id": f"placement_{index:03d}",
                "asset_id": str(placement.get("asset_id", "unknown_asset")),
                "mode": "placeholder",
                "transform": {
                    "pos": _clamp_floor_position(position, dimensions),
                    "rot": [round(rotation[0], 3), round(rotation[1], 3), round(rotation[2], 3)],
                    "scale": [round(scale[0], 3), round(scale[1], 3), round(scale[2], 3)],
                },
            }
        )

    return compiled


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
        }

    dimensions = template["dimensions"]
    raw_placements = worldspec.get("placements", [])
    raw_placements = raw_placements if isinstance(raw_placements, list) else []
    placements = _compile_placements(raw_placements, dimensions)

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
    }
