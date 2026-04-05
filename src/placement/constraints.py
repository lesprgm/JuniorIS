from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from src.placement.constants import (
    FLOOR_ANCHOR_RATIOS,
    FLOOR_EDGE_MARGIN,
    WALL_LATERAL_RATIOS,
)
from src.placement.role_defaults import (
    ROLE_DEFAULT_ADJACENCY_TARGETS,
    WALL_ANCHORED_ROLES,
)


def _safe_text(value: Any) -> str:
    return str(value or "").strip().lower()


def normalize_anchor_preferences(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    allowed = {"against_wall", "clustered", "reading_nook"}
    seen: List[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        token = value.strip().lower()
        if token and token in allowed and token not in seen:
            seen.append(token)
    return seen


def build_floor_candidate_points(
    dimensions: Dict[str, float],
    preferred: Tuple[float, float] | None = None,
) -> List[Tuple[float, float, float]]:
    half_width = max(float(dimensions.get("width", 8.0)) * 0.5 - FLOOR_EDGE_MARGIN, 0.0)
    half_length = max(float(dimensions.get("length", 8.0)) * 0.5 - FLOOR_EDGE_MARGIN, 0.0)
    points: List[Tuple[float, float, float]] = []
    if preferred is not None:
        points.append(
            (
                max(min(preferred[0], half_width), -half_width),
                max(min(preferred[1], half_length), -half_length),
                0.0,
            )
        )
    for x_ratio, z_ratio, yaw in FLOOR_ANCHOR_RATIOS:
        points.append((round(half_width * x_ratio, 3), round(half_length * z_ratio, 3), yaw))
    return points


def build_wall_candidate_points(
    dimensions: Dict[str, float],
    wall_inset: float,
) -> List[Tuple[float, float, float]]:
    half_width = max(float(dimensions.get("width", 8.0)) * 0.5 - wall_inset, 0.0)
    half_length = max(float(dimensions.get("length", 8.0)) * 0.5 - wall_inset, 0.0)
    points: List[Tuple[float, float, float]] = []
    points.extend((round(half_width * ratio, 3), round(-half_length, 3), 0.0) for ratio in WALL_LATERAL_RATIOS)
    points.extend((round(half_width * ratio, 3), round(half_length, 3), 180.0) for ratio in reversed(WALL_LATERAL_RATIOS))
    points.extend((round(half_width, 3), round(half_length * ratio, 3), 270.0) for ratio in WALL_LATERAL_RATIOS)
    points.extend((round(-half_width, 3), round(half_length * ratio, 3), 90.0) for ratio in reversed(WALL_LATERAL_RATIOS))
    return points


def default_constraint_for_role(
    role: str,
    selected_roles: Iterable[str],
    placement_intent: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    selected = {_safe_text(value) for value in selected_roles if _safe_text(value)}
    placement_intent = placement_intent or {}
    adjacency_pairs = placement_intent.get("adjacency_pairs") if isinstance(placement_intent.get("adjacency_pairs"), list) else []
    for pair in adjacency_pairs:
        if not isinstance(pair, dict):
            continue
        if _safe_text(pair.get("source_role")) != role:
            continue
        target_role = _safe_text(pair.get("target_role"))
        if target_role in selected:
            return {"type": "near", "target": target_role}

    for target_role in ROLE_DEFAULT_ADJACENCY_TARGETS.get(role, []):
        if target_role in selected:
            return {"type": "near", "target": target_role}

    anchor_preferences = set(normalize_anchor_preferences(placement_intent.get("anchor_preferences")))
    if role in WALL_ANCHORED_ROLES or "against_wall" in anchor_preferences:
        return {"type": "against_wall"}
    return {"type": "floor"}
