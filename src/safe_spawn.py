from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Tuple


PLAYER_CAPSULE_RADIUS = 0.25
PLAYER_CAPSULE_HEIGHT = 1.70
WALL_MARGIN = 0.30
BASE_PLACEMENT_RADIUS = 0.35
RING_STEP = 0.50
ANGLE_SEQUENCE = (0, 45, 90, 135, 180, 225, 270, 315)
DEFAULT_ROTATION_Y = 180.0


def _to_float(value: Any, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _safe_vec3(values: Any, default: List[float]) -> List[float]:
    if (
        isinstance(values, list)
        and len(values) == 3
        and all(isinstance(component, (int, float)) for component in values)
    ):
        return [float(values[0]), float(values[1]), float(values[2])]
    return list(default)


def _extract_room_bounds(template: Dict[str, Any]) -> Dict[str, float] | None:
    dimensions = template.get("dimensions")
    if not isinstance(dimensions, dict):
        return None
    width = _to_float(dimensions.get("width"), -1.0)
    length = _to_float(dimensions.get("length"), -1.0)
    if width <= 0.0 or length <= 0.0:
        return None
    max_x = max((width / 2.0) - WALL_MARGIN, 0.0)
    max_z = max((length / 2.0) - WALL_MARGIN, 0.0)
    return {
        "width": width,
        "length": length,
        "max_x": max_x,
        "max_z": max_z,
    }


def _extract_teleportable_bounds(template: Dict[str, Any]) -> List[Tuple[float, float, float, float]]:
    bounds: List[Tuple[float, float, float, float]] = []
    nodes = template.get("nodes")
    if not isinstance(nodes, list):
        return bounds

    for node in nodes:
        if not isinstance(node, dict):
            continue
        if node.get("teleportable") is not True:
            continue
        if node.get("kind") != "plane":
            continue

        size = node.get("size")
        position = node.get("position")
        if (
            not isinstance(size, list)
            or len(size) < 2
            or not isinstance(position, list)
            or len(position) < 3
        ):
            continue
        if not all(isinstance(value, (int, float)) for value in [size[0], size[1], position[0], position[2]]):
            continue

        half_width = float(size[0]) / 2.0
        half_length = float(size[1]) / 2.0
        center_x = float(position[0])
        center_z = float(position[2])

        bounds.append(
            (
                center_x - half_width,
                center_x + half_width,
                center_z - half_length,
                center_z + half_length,
            )
        )

    return bounds


def _extract_occupancy(placements: Any) -> List[Tuple[float, float, float]]:
    occupancy: List[Tuple[float, float, float]] = []
    if not isinstance(placements, list):
        return occupancy

    for placement in placements:
        if not isinstance(placement, dict):
            continue
        transform = placement.get("transform")
        transform = transform if isinstance(transform, dict) else {}

        pos = _safe_vec3(transform.get("pos"), [0.0, 0.0, 0.0])
        scale = _safe_vec3(transform.get("scale"), [1.0, 1.0, 1.0])
        footprint_scale = max(abs(scale[0]), abs(scale[2]), 1.0)
        radius = max(BASE_PLACEMENT_RADIUS, 0.25 * footprint_scale)
        occupancy.append((float(pos[0]), float(pos[2]), float(radius)))

    return occupancy


def _is_inside_teleportable_floor(
    x: float,
    z: float,
    teleportable_bounds: List[Tuple[float, float, float, float]],
) -> bool:
    for min_x, max_x, min_z, max_z in teleportable_bounds:
        if min_x <= x <= max_x and min_z <= z <= max_z:
            return True
    return False


def _intersects_placement(x: float, z: float, occupancy: List[Tuple[float, float, float]]) -> bool:
    for ox, oz, radius in occupancy:
        dx = x - ox
        dz = z - oz
        minimum_clearance = PLAYER_CAPSULE_RADIUS + radius
        if (dx * dx) + (dz * dz) < (minimum_clearance * minimum_clearance):
            return True
    return False


def _candidate_sequence(max_radius: float) -> Iterable[Tuple[float, float]]:
    yield (0.0, 0.0)
    radius = RING_STEP
    while radius <= max_radius + 1e-9:
        for angle in ANGLE_SEQUENCE:
            radians = math.radians(angle)
            x = round(math.cos(radians) * radius, 3)
            z = round(math.sin(radians) * radius, 3)
            yield (x, z)
        radius = round(radius + RING_STEP, 6)


def _clamp_to_room(x: float, z: float, room_bounds: Dict[str, float]) -> Tuple[float, float]:
    clamped_x = max(min(x, room_bounds["max_x"]), -room_bounds["max_x"])
    clamped_z = max(min(z, room_bounds["max_z"]), -room_bounds["max_z"])
    return (round(clamped_x, 3), round(clamped_z, 3))


def find_safe_spawn(phase0_data: Dict[str, Any]) -> Dict[str, Any]:
    template = phase0_data.get("template")
    template = template if isinstance(template, dict) else {}
    room_bounds = _extract_room_bounds(template)
    if room_bounds is None:
        return {
            "ok": False,
            "spawn": None,
            "attempts": 0,
            "reason": "invalid_template_dimensions",
        }

    teleportable_bounds = _extract_teleportable_bounds(template)
    if not teleportable_bounds:
        return {
            "ok": False,
            "spawn": None,
            "attempts": 0,
            "reason": "no_teleportable_floor",
        }

    occupancy = _extract_occupancy(phase0_data.get("placements"))
    max_radius = max(min(room_bounds["max_x"], room_bounds["max_z"]), 0.0)

    attempts = 0
    visited: set[Tuple[float, float]] = set()
    for candidate_x, candidate_z in _candidate_sequence(max_radius):
        clamped_x, clamped_z = _clamp_to_room(candidate_x, candidate_z, room_bounds)
        if (clamped_x, clamped_z) in visited:
            continue
        visited.add((clamped_x, clamped_z))
        attempts += 1

        if not _is_inside_teleportable_floor(clamped_x, clamped_z, teleportable_bounds):
            continue
        if _intersects_placement(clamped_x, clamped_z, occupancy):
            continue

        return {
            "ok": True,
            "spawn": {
                "pos": [clamped_x, 0.0, clamped_z],
                "rot": [0.0, DEFAULT_ROTATION_Y, 0.0],
            },
            "attempts": attempts,
            "reason": "",
        }

    return {
        "ok": False,
        "spawn": None,
        "attempts": attempts,
        "reason": "no_safe_spawn_found",
    }
