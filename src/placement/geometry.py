from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

from src.placement.constants import (
    BASE_CAPACITY_SCALE,
    COLLISION_PADDING_BY_CLASS,
    DENSITY_BUDGET_FILL,
    DENSITY_MULTIPLIERS,
    MAX_FOOTPRINT_RADIUS,
    MAX_NEAR_DISTANCE,
    MAX_WALL_CLEARANCE,
    MIN_COLLISION_PADDING,
    MIN_NEAR_DISTANCE,
    MIN_WALL_INSET,
    NEAR_DISTANCE_PADDING_BY_DENSITY,
    NEAR_DISTANCE_SCALE_BY_DENSITY,
)
from src.placement.role_defaults import ROLE_FALLBACK_GEOMETRY, ROLE_PRIORITY
from src.placement.semantic_taxonomy import (
    canonicalize_concept,
    canonicalize_role_token,
    ground_concept,
    supported_runtime_roles,
)


def _normalize_tokens(*values: Any) -> List[str]:  # extracts alphanumeric keywords from strings across multiple arguments
    out: List[str] = []
    for value in values:
        text = str(value or "").strip().lower().replace("/", " ").replace("_", " ").replace("-", " ")
        if not text:
            continue
        out.extend(part for part in text.split() if part)
    return out


def _safe_text(value: Any) -> str:  # lowercases and strips string arguments for dictionary keys
    return str(value or "").strip().lower()


def _resolve_known_role(token: str) -> str:  # maps raw text back to supported exact role enums if possible
    return canonicalize_role_token(token)


def canonicalize_semantic_concept(value: Any) -> str:  # preserves finer semantic furniture concepts before runtime grounding
    return canonicalize_concept(value)


def map_semantic_concept_to_runtime_role(concept: Any, asset: Dict[str, Any] | None = None) -> Tuple[str, str]:
    del asset
    return ground_concept(concept)


def canonicalize_semantic_role(value: Any) -> str:  # normalizes free-text role names into the runtime ontology of known roles
    token = _safe_text(value).replace("-", "_").replace(" ", "_").replace("/", "_")
    if not token:
        return ""
    resolved = _resolve_known_role(token)
    if resolved:
        return resolved

    concept = canonicalize_semantic_concept(token)
    role, _ = map_semantic_concept_to_runtime_role(concept)
    if role:
        return role

    for part in reversed([part for part in token.split("_") if part]):
        resolved = _resolve_known_role(part)
        if resolved:
            return resolved
    return concept or token


def semantic_role_key(record: Dict[str, Any]) -> str:  # tries grounded/runtime fields first, then falls back to metadata-driven role inference
    explicit_role = canonicalize_semantic_role(record.get("runtime_role") or record.get("selected_role") or record.get("role"))
    supported_roles = supported_runtime_roles()
    if explicit_role in supported_roles or explicit_role in ROLE_PRIORITY:
        return explicit_role

    tags = record.get("tags") if isinstance(record.get("tags"), list) else []
    role_values = [
        record.get("category"),
        record.get("label"),
        record.get("room_role_subtype"),
        record.get("semantic_concept"),
    ] + tags + [
        record.get("asset_id"),
        record.get("requested_asset_id"),
    ]
    for value in role_values:
        canonical = canonicalize_semantic_role(value)
        if canonical in supported_roles or canonical in ROLE_PRIORITY:
            return canonical
    for token in _normalize_tokens(*role_values):
        canonical = canonicalize_semantic_role(token)
        if canonical in supported_roles or canonical in ROLE_PRIORITY:
            return canonical
    asset_id = _safe_text(record.get("asset_id") or record.get("requested_asset_id") or "asset")
    return asset_id or "asset"

def placement_priority(record: Dict[str, Any]) -> int:  # higher-priority roles (bed=100, sofa=90) are placed first
    return ROLE_PRIORITY.get(semantic_role_key(record), 10)


def _base_geometry_for_role(role: str) -> Dict[str, Any]:  # returns expected approximate footprint radius for standard furniture types
    base = ROLE_FALLBACK_GEOMETRY.get(role) or ROLE_FALLBACK_GEOMETRY["asset"]
    profile = dict(base)
    profile["placement_role"] = role
    return profile


def _normalized_scale(scale: Any) -> Tuple[float, float, float]:  # enforces uniform XYZ mesh scaling
    if isinstance(scale, list) and len(scale) == 3 and all(isinstance(v, (int, float)) for v in scale):
        return tuple(max(float(v), 0.1) for v in scale)
    return (1.0, 1.0, 1.0)


def _bounds_size(record: Dict[str, Any], scale: Tuple[float, float, float]) -> Tuple[float, float] | None:  # calculates true rendered XZ footprint from base bounds and applied scale
    bounds = record.get("bounds") if isinstance(record.get("bounds"), dict) else None
    size = bounds.get("size") if isinstance(bounds, dict) and isinstance(bounds.get("size"), dict) else None
    if not isinstance(size, dict):
        return None
    size_x = size.get("x")
    size_z = size.get("z")
    if not isinstance(size_x, (int, float)) or not isinstance(size_z, (int, float)):
        return None
    return max(float(size_x) * scale[0], 0.1), max(float(size_z) * scale[2], 0.1)


def normalize_density_profile(value: Any) -> str:  # coerces arbitrary input to one of minimal/normal/cluttered
    token = _safe_text(value or "normal")
    return token if token in DENSITY_MULTIPLIERS else "normal"


def normalize_layout_mood(value: Any, density_profile: str) -> str:  # validates the explicit layout mood chosen by the model
    token = _safe_text(value)
    if token in {"open", "cozy", "crowded"}:
        return token
    if density_profile == "minimal":
        return "open"
    if density_profile == "cluttered":
        return "crowded"
    return "cozy"


def geometry_profile_from_asset(record: Dict[str, Any], scale: Any = None) -> Dict[str, Any]:  # computes footprint radius, wall clearance, and near distance from asset bounds
    role = semantic_role_key(record)
    base = _base_geometry_for_role(role)
    normalized_scale = _normalized_scale(scale if scale is not None else record.get("transform", {}).get("scale"))
    bounds_size = _bounds_size(record, normalized_scale)

    if bounds_size is None:
        return {
            **base,
            "footprint_area": round(math.pi * base["footprint_radius"] * base["footprint_radius"], 3),
            "bounds_source": "role_default",
        }

    size_x, size_z = bounds_size
    footprint_radius = min(MAX_FOOTPRINT_RADIUS, max(base["footprint_radius"], max(size_x, size_z) * 0.5))
    wall_clearance = min(MAX_WALL_CLEARANCE, max(base["wall_clearance"], min(size_x, size_z) * 0.12 + 0.12))
    collision_class = base["collision_padding_class"]
    if footprint_radius >= 1.05:
        collision_class = "wide"
    elif footprint_radius <= 0.5:
        collision_class = "compact"

    preferred_near_distance = max(base["preferred_near_distance"], round((size_x + size_z) * 0.35, 3))
    return {
        "placement_role": role,
        "footprint_radius": round(footprint_radius, 3),
        "wall_clearance": round(wall_clearance, 3),
        "preferred_near_distance": round(preferred_near_distance, 3),
        "collision_padding_class": collision_class,
        "footprint_area": round(max(size_x * size_z, math.pi * footprint_radius * footprint_radius), 3),
        "bounds_source": "asset_bounds",
    }


def collision_padding_for_profile(profile: Dict[str, Any]) -> float:  # returns the padding distance based on collision class (compact/standard/wide)
    profile_class = _safe_text(profile.get("collision_padding_class") or "standard")
    return COLLISION_PADDING_BY_CLASS.get(profile_class, MIN_COLLISION_PADDING)


def derive_wall_inset(profile: Dict[str, Any]) -> float:  # how far from the wall the object center must be placed
    return round(
        max(
            MIN_WALL_INSET,
            float(profile.get("footprint_radius", 0.6)) + float(profile.get("wall_clearance", 0.25)),
        ),
        3,
    )


def derive_near_distance(  # fallback inference of near-distance given collision profiles
    source_profile: Dict[str, Any],
    target_profile: Dict[str, Any],
    density_profile: str = "normal",
    explicit_distance: float | None = None,
) -> float:
    density_profile = normalize_density_profile(density_profile)
    source_radius = float(source_profile.get("footprint_radius", ROLE_FALLBACK_GEOMETRY["asset"]["footprint_radius"]))
    target_radius = float(target_profile.get("footprint_radius", ROLE_FALLBACK_GEOMETRY["asset"]["footprint_radius"]))
    source_baseline = float(source_profile.get("preferred_near_distance", MIN_NEAR_DISTANCE))
    target_baseline = float(target_profile.get("preferred_near_distance", MIN_NEAR_DISTANCE))
    pair_scale = NEAR_DISTANCE_SCALE_BY_DENSITY[density_profile]
    pair_padding = NEAR_DISTANCE_PADDING_BY_DENSITY[density_profile]
    derived = max(
        MIN_NEAR_DISTANCE,
        pair_scale * (source_radius + target_radius) + pair_padding,
        (source_baseline + target_baseline) * 0.5,
    )
    if isinstance(explicit_distance, (int, float)):
        derived = max(float(explicit_distance), derived)
    return round(min(derived, MAX_NEAR_DISTANCE), 3)


def room_capacity_summary(  # estimates how many objects the room can hold based on area, density, and average footprint
    dimensions: Dict[str, float],
    geometry_profiles: Sequence[Dict[str, Any]],
    density_profile: str,
    max_props: int,
    available_count: int,
) -> Dict[str, Any]:
    density_profile = normalize_density_profile(density_profile)
    width = max(float(dimensions.get("width", 0.0)), 0.0)
    length = max(float(dimensions.get("length", 0.0)), 0.0)
    room_area = width * length
    effective_areas: List[float] = []
    for profile in geometry_profiles:
        radius = float(profile.get("footprint_radius", ROLE_FALLBACK_GEOMETRY["asset"]["footprint_radius"]))
        clearance = float(profile.get("wall_clearance", ROLE_FALLBACK_GEOMETRY["asset"]["wall_clearance"]))
        padding = collision_padding_for_profile(profile)
        effective_areas.append(max((2.0 * (radius + clearance + padding)) ** 2, 0.5))
    average_footprint_area = sum(effective_areas) / max(len(effective_areas), 1)
    density_multiplier = DENSITY_MULTIPLIERS[density_profile]
    # We no longer apply density_multiplier to capacity, nor fill_ratio to target_budget.
    # The LLM's max_props and the room's raw physical floor space are the only constraints.
    raw_capacity = int(math.floor((room_area / max(average_footprint_area, 0.5)) * BASE_CAPACITY_SCALE))
    derived_capacity = max(1, raw_capacity)
    target_count = min(available_count, max_props if max_props > 0 else available_count, derived_capacity)
    return {
        "room_area": round(room_area, 3),
        "average_footprint_area": round(average_footprint_area, 3),
        "density_profile": density_profile,
        "density_multiplier": density_multiplier,
        "derived_capacity": derived_capacity,
        "budget_fill_ratio": 1.0,
        "target_count": max(1, target_count),
    }
