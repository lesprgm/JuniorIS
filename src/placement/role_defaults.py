from __future__ import annotations

from typing import Any, Dict


# Keep behavior deterministic so planner/runtime contracts stay stable.
def _geometry_defaults(
    footprint_radius: float,
    wall_clearance: float,
    preferred_near_distance: float,
    collision_padding_class: str,
) -> Dict[str, Any]:
    return {
        "footprint_radius": footprint_radius,
        "wall_clearance": wall_clearance,
        "preferred_near_distance": preferred_near_distance,
        "collision_padding_class": collision_padding_class,
    }


ROLE_PRIORITY = {
    "bed": 100,
    "sofa": 90,
    "desk": 85,
    "table": 80,
    "cabinet": 76,
    "cabinet/storage": 76,
    "appliance": 74,
    "bench": 72,
    "chair": 64,
    "lamp": 58,
    "plant": 56,
    "sign": 52,
    "decor": 40,
}

ROLE_FALLBACK_GEOMETRY = {
    "bed": _geometry_defaults(1.35, 0.45, 2.0, "wide"),
    "sofa": _geometry_defaults(1.15, 0.38, 1.8, "wide"),
    "desk": _geometry_defaults(1.0, 0.34, 1.5, "wide"),
    "table": _geometry_defaults(0.95, 0.32, 1.45, "wide"),
    "cabinet": _geometry_defaults(0.9, 0.30, 1.3, "standard"),
    "cabinet/storage": _geometry_defaults(0.9, 0.30, 1.3, "standard"),
    "appliance": _geometry_defaults(0.9, 0.32, 1.35, "standard"),
    "bench": _geometry_defaults(0.9, 0.28, 1.35, "standard"),
    "chair": _geometry_defaults(0.62, 0.24, 1.1, "standard"),
    "lamp": _geometry_defaults(0.45, 0.18, 1.0, "compact"),
    "plant": _geometry_defaults(0.48, 0.20, 1.15, "compact"),
    "sign": _geometry_defaults(0.35, 0.18, 0.95, "compact"),
    "decor": _geometry_defaults(0.38, 0.18, 0.95, "compact"),
    "asset": _geometry_defaults(0.6, 0.25, 1.2, "standard"),
}
