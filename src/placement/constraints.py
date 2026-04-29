from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from src.planning.scene_program_policy import policy_set

import numpy as np
from scipy.stats import qmc

from src.placement.constants import (
    FLOOR_ANCHOR_RATIOS,
    FLOOR_EDGE_MARGIN,
    WALL_LATERAL_RATIOS,
)


# Keep behavior deterministic so planner/runtime contracts stay stable.
def _safe_text(value: Any) -> str:
    return str(value or "").strip().lower()


def normalize_anchor_preferences(values: Any) -> List[str]:  # validates and deduplicates anchor preference tokens from LLM output
    if not isinstance(values, list):
        return []
    allowed = policy_set("allowed_anchor_preferences")
    seen: List[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        token = value.strip().lower()
        if token and token in allowed and token not in seen:
            seen.append(token)
    return seen


def _dedupe_points(points: Iterable[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    deduped: List[Tuple[float, float, float]] = []
    seen = set()
    for x, z, yaw in points:
        key = (round(x, 3), round(z, 3), round(yaw % 360.0, 3))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((round(x, 3), round(z, 3), round(yaw % 360.0, 3)))
    return deduped


def _clamp_preferred(preferred: Tuple[float, float] | None, half_width: float, half_length: float) -> Tuple[float, float, float] | None:
    if preferred is None:
        return None
    return (
        round(max(min(preferred[0], half_width), -half_width), 3),
        round(max(min(preferred[1], half_length), -half_length), 3),
        0.0,
    )


def _halton_points(dimensions: int, seed: int, count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, dimensions))
    sampler = qmc.Halton(d=dimensions, scramble=True, seed=seed)
    return sampler.random(n=count)


def build_floor_candidate_points(  # generates candidate placement positions on the floor plane using anchor ratios and Halton sampling
    dimensions: Dict[str, float],
    preferred: Tuple[float, float] | None = None,
    *,
    seed: int | None = None,
    sample_count: int = 0,
    jitter_scale: float = 0.0,
) -> List[Tuple[float, float, float]]:
    half_width = max(float(dimensions.get("width", 8.0)) * 0.5 - FLOOR_EDGE_MARGIN, 0.0)
    half_length = max(float(dimensions.get("length", 8.0)) * 0.5 - FLOOR_EDGE_MARGIN, 0.0)
    points: List[Tuple[float, float, float]] = []

    clamped_preferred = _clamp_preferred(preferred, half_width, half_length)
    if clamped_preferred is not None:
        points.append(clamped_preferred)

    for x_ratio, z_ratio, yaw in FLOOR_ANCHOR_RATIOS:
        points.append((round(half_width * x_ratio, 3), round(half_length * z_ratio, 3), yaw))

    if sample_count > 0 and seed is not None:
        samples = _halton_points(3, seed, sample_count)
        for sample in samples:
            x = (sample[0] * 2.0 - 1.0) * half_width
            z = (sample[1] * 2.0 - 1.0) * half_length
            yaw = sample[2] * 360.0
            if jitter_scale > 0.0:
                x = max(min(x + (sample[2] - 0.5) * jitter_scale, half_width), -half_width)
                z = max(min(z + (sample[0] - 0.5) * jitter_scale, half_length), -half_length)
            points.append((x, z, yaw))

    return _dedupe_points(points)


def build_wall_candidate_points(  # generates candidate positions along the room perimeter for wall-anchored objects
    dimensions: Dict[str, float],
    wall_inset: float,
    *,
    seed: int | None = None,
    sample_count: int = 0,
) -> List[Tuple[float, float, float]]:
    half_width = max(float(dimensions.get("width", 8.0)) * 0.5 - wall_inset, 0.0)
    half_length = max(float(dimensions.get("length", 8.0)) * 0.5 - wall_inset, 0.0)
    points: List[Tuple[float, float, float]] = []
    points.extend((round(half_width * ratio, 3), round(-half_length, 3), 0.0) for ratio in WALL_LATERAL_RATIOS)
    points.extend((round(half_width * ratio, 3), round(half_length, 3), 180.0) for ratio in reversed(WALL_LATERAL_RATIOS))
    points.extend((round(half_width, 3), round(half_length * ratio, 3), 270.0) for ratio in WALL_LATERAL_RATIOS)
    points.extend((round(-half_width, 3), round(half_length * ratio, 3), 90.0) for ratio in reversed(WALL_LATERAL_RATIOS))

    if sample_count > 0 and seed is not None:
        perimeter = max((half_width * 2.0) + (half_length * 2.0), 0.001)
        samples = _halton_points(1, seed, sample_count).flatten()
        for sample in samples:
            distance = sample * perimeter * 2.0
            if distance <= half_width * 2.0:
                x = -half_width + distance
                z = -half_length
                yaw = 0.0
            elif distance <= (half_width * 2.0) + (half_length * 2.0):
                x = half_width
                z = -half_length + (distance - (half_width * 2.0))
                yaw = 270.0
            elif distance <= (half_width * 4.0) + (half_length * 2.0):
                x = half_width - (distance - ((half_width * 2.0) + (half_length * 2.0)))
                z = half_length
                yaw = 180.0
            else:
                x = -half_width
                z = half_length - (distance - ((half_width * 4.0) + (half_length * 2.0)))
                yaw = 90.0
            points.append((x, z, yaw))

    return _dedupe_points(points)


def default_constraint_for_role(  # infers the placement constraint type (floor/wall/near) from role and adjacency pairs
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
        relation = _safe_text(pair.get("relation") or "near")
        target_role = _safe_text(pair.get("target_role"))
        if target_role in selected and relation in policy_set("near_constraint_relations"):
            return {"type": "near", "target": target_role, "relation": relation}

    anchor_preferences = set(normalize_anchor_preferences(placement_intent.get("anchor_preferences")))
    if "against_wall" in anchor_preferences:
        return {"type": "against_wall"}
    return {"type": "floor"}
