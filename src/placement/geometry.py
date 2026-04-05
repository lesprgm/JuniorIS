from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple

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
from src.placement.constraints import normalize_anchor_preferences
from src.placement.role_defaults import (
    KNOWN_ROLE_TOKENS,
    ROLE_ALIASES,
    ROLE_DEFAULT_ADJACENCY_TARGETS,
    ROLE_FALLBACK_GEOMETRY,
    ROLE_PRIORITY,
)


def _normalize_tokens(*values: Any) -> List[str]:
    out: List[str] = []
    for value in values:
        text = str(value or "").strip().lower().replace("/", " ").replace("_", " ").replace("-", " ")
        if not text:
            continue
        out.extend(part for part in text.split() if part)
    return out


def _safe_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _resolve_known_role(token: str) -> str:
    if not token:
        return ""
    if token in ROLE_ALIASES:
        return ROLE_ALIASES[token]
    singular = token[:-1] if token.endswith("s") else token
    if singular in ROLE_ALIASES:
        return ROLE_ALIASES[singular]
    if singular in KNOWN_ROLE_TOKENS:
        if singular == "storage":
            return "cabinet"
        return singular
    return ""


def canonicalize_semantic_role(value: Any) -> str:
    token = _safe_text(value).replace("-", "_").replace(" ", "_").replace("/", "_")
    if not token:
        return ""
    resolved = _resolve_known_role(token)
    if resolved:
        return resolved

    parts = [part for part in token.split("_") if part]
    if parts and parts[-1] in {"area", "room", "space", "corner", "zone"}:
        token = "_".join(parts[:-1])
        resolved = _resolve_known_role(token)
        if resolved:
            return resolved
        parts = [part for part in token.split("_") if part]
    for index in range(len(parts)):
        suffix = "_".join(parts[index:])
        resolved = _resolve_known_role(suffix)
        if resolved:
            return resolved

    for part in reversed(parts):
        resolved = _resolve_known_role(part)
        if resolved:
            return resolved

    return token


def _normalize_string_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    for value in values:
        if isinstance(value, str):
            token = value.strip().lower()
            if token:
                out.append(token)
    return out


def semantic_role_key(record: Dict[str, Any]) -> str:
    tags = record.get("tags") if isinstance(record.get("tags"), list) else []
    role_values = tags + [
        record.get("category"),
        record.get("role"),
        record.get("label"),
        record.get("asset_id"),
        record.get("requested_asset_id"),
    ]
    for value in role_values:
        canonical = canonicalize_semantic_role(value)
        if canonical in KNOWN_ROLE_TOKENS or canonical in ROLE_PRIORITY:
            return canonical
    for token in _normalize_tokens(*role_values):
        canonical = canonicalize_semantic_role(token)
        if canonical in KNOWN_ROLE_TOKENS or canonical in ROLE_PRIORITY:
            return canonical
    asset_id = _safe_text(record.get("asset_id") or record.get("requested_asset_id") or "asset")
    return asset_id or "asset"


def placement_priority(record: Dict[str, Any]) -> int:
    return ROLE_PRIORITY.get(semantic_role_key(record), 10)


def _base_geometry_for_role(role: str) -> Dict[str, Any]:
    base = ROLE_FALLBACK_GEOMETRY.get(role) or ROLE_FALLBACK_GEOMETRY["asset"]
    profile = dict(base)
    profile["placement_role"] = role
    return profile


def _normalized_scale(scale: Any) -> Tuple[float, float, float]:
    if isinstance(scale, list) and len(scale) == 3 and all(isinstance(v, (int, float)) for v in scale):
        return tuple(max(float(v), 0.1) for v in scale)
    return (1.0, 1.0, 1.0)


def _bounds_size(record: Dict[str, Any], scale: Tuple[float, float, float]) -> Tuple[float, float] | None:
    bounds = record.get("bounds") if isinstance(record.get("bounds"), dict) else None
    size = bounds.get("size") if isinstance(bounds, dict) and isinstance(bounds.get("size"), dict) else None
    if not isinstance(size, dict):
        return None
    size_x = size.get("x")
    size_z = size.get("z")
    if not isinstance(size_x, (int, float)) or not isinstance(size_z, (int, float)):
        return None
    return max(float(size_x) * scale[0], 0.1), max(float(size_z) * scale[2], 0.1)


def normalize_density_profile(value: Any) -> str:
    token = _safe_text(value or "normal")
    return token if token in DENSITY_MULTIPLIERS else "normal"


def normalize_layout_mood(value: Any, density_profile: str) -> str:
    token = _safe_text(value)
    if token in {"open", "cozy", "crowded"}:
        return token
    if density_profile == "minimal":
        return "open"
    if density_profile == "cluttered":
        return "crowded"
    return "cozy"


def _normalize_adjacency_pairs(values: Any, known_roles: set[str]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if not isinstance(values, list):
        return out
    seen: set[Tuple[str, str, str]] = set()
    for value in values:
        if not isinstance(value, dict):
            continue
        source_role = _safe_text(value.get("source_role"))
        target_role = _safe_text(value.get("target_role"))
        relation = _safe_text(value.get("relation") or "near") or "near"
        if relation != "near":
            continue
        if not source_role or not target_role:
            continue
        if known_roles and (source_role not in known_roles or target_role not in known_roles):
            continue
        key = (source_role, target_role, relation)
        if key in seen:
            continue
        seen.add(key)
        out.append({"source_role": source_role, "target_role": target_role, "relation": relation})
    return out


def default_placement_intent(intent_spec: Dict[str, Any] | None = None) -> Dict[str, Any]:
    intent_spec = intent_spec or {}
    style_tags = set(_normalize_string_list(intent_spec.get("style_tags")))
    required_roles = _normalize_string_list(intent_spec.get("required_roles"))
    optional_roles = _normalize_string_list(intent_spec.get("optional_roles"))
    known_roles = set(required_roles + optional_roles)

    density_profile = normalize_density_profile(intent_spec.get("density_profile"))
    anchor_preferences: List[str] = []
    if density_profile == "cluttered":
        anchor_preferences.append("clustered")
    if {"cozy", "reading"} & style_tags:
        anchor_preferences.append("reading_nook")

    adjacency_pairs: List[Dict[str, str]] = []
    for source_role, target_roles in ROLE_DEFAULT_ADJACENCY_TARGETS.items():
        if source_role not in known_roles:
            continue
        for target_role in target_roles:
            if target_role in known_roles:
                adjacency_pairs.append({"source_role": source_role, "target_role": target_role, "relation": "near"})
                break

    return {
        "density_profile": density_profile,
        "anchor_preferences": anchor_preferences,
        "adjacency_pairs": adjacency_pairs,
        "layout_mood": normalize_layout_mood(intent_spec.get("layout_mood"), density_profile),
    }


def normalize_placement_intent(raw_intent: Any, intent_spec: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if not isinstance(raw_intent, dict):
        return default_placement_intent(intent_spec)

    intent_spec = intent_spec or {}
    required_roles = _normalize_string_list(intent_spec.get("required_roles"))
    optional_roles = _normalize_string_list(intent_spec.get("optional_roles"))
    known_roles = set(required_roles + optional_roles)
    default_intent = default_placement_intent(intent_spec)
    density_profile = normalize_density_profile(raw_intent.get("density_profile") or default_intent["density_profile"])
    anchor_preferences = normalize_anchor_preferences(raw_intent.get("anchor_preferences")) or default_intent["anchor_preferences"]
    adjacency_pairs = _normalize_adjacency_pairs(raw_intent.get("adjacency_pairs"), known_roles) or default_intent["adjacency_pairs"]
    layout_mood = normalize_layout_mood(raw_intent.get("layout_mood") or default_intent["layout_mood"], density_profile)
    return {
        "density_profile": density_profile,
        "anchor_preferences": anchor_preferences,
        "adjacency_pairs": adjacency_pairs,
        "layout_mood": layout_mood,
    }


def geometry_profile_from_asset(record: Dict[str, Any], scale: Any = None) -> Dict[str, Any]:
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


def collision_padding_for_profile(profile: Dict[str, Any]) -> float:
    profile_class = _safe_text(profile.get("collision_padding_class") or "standard")
    return COLLISION_PADDING_BY_CLASS.get(profile_class, MIN_COLLISION_PADDING)


def derive_wall_inset(profile: Dict[str, Any]) -> float:
    return round(
        max(
            MIN_WALL_INSET,
            float(profile.get("footprint_radius", 0.6)) + float(profile.get("wall_clearance", 0.25)),
        ),
        3,
    )


def derive_near_distance(
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


def room_capacity_summary(
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
    raw_capacity = int(math.floor((room_area / max(average_footprint_area, 0.5)) * density_multiplier * BASE_CAPACITY_SCALE))
    derived_capacity = max(1, raw_capacity)
    fill_ratio = DENSITY_BUDGET_FILL[density_profile]
    target_budget = max(1, int(round(max_props * fill_ratio))) if max_props > 0 else available_count
    target_count = min(available_count, max_props, derived_capacity, target_budget)
    return {
        "room_area": round(room_area, 3),
        "average_footprint_area": round(average_footprint_area, 3),
        "density_profile": density_profile,
        "density_multiplier": density_multiplier,
        "derived_capacity": derived_capacity,
        "budget_fill_ratio": fill_ratio,
        "target_count": max(1, target_count),
    }
