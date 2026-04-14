from __future__ import annotations

SHORTLIST_SCORE_WEIGHTS = {
    "tag_overlap": 3.0,
    "label_overlap": 2.0,
    "requested_role": 6.0,
    "planner_approved": 4.0,
    "manual_approved": 2.0,
    "semantic_confidence": 3.0,
    "style_confidence": 1.5,
    "quality_high": 0.5,
    "quality_medium": 0.25,
    "perf_cheap": 0.75,
    "perf_moderate": 0.25,
    "intent_style_overlap": 2.0,
    "intent_color_overlap": 1.5,
}

SHORTLIST_ROLE_COVERAGE_LIMIT = 8  # max assets per role reserved during shortlisting
SHORTLIST_DEFAULT_LIMIT = 40  # default total shortlist size sent to the LLM
MIN_SEMANTIC_CONFIDENCE = 0.6  # assets below this confidence are excluded from placement

# Fixed safety constraints. These remain deterministic even when semantic
# planning changes upstream.
ROOM_CLAMP_MARGIN = 0.35  # meters inset from room walls for position clamping
FLOOR_EDGE_MARGIN = 0.6  # min distance from floor edge for floor-placed objects
MIN_WALL_INSET = 0.45  # min distance from wall for wall-anchored objects
MIN_COLLISION_PADDING = 0.18  # minimum clearance between any two placed objects
MIN_NEAR_DISTANCE = 0.65  # closest two "near" objects can be before overlap risk
MAX_NEAR_DISTANCE = 2.5  # beyond this distance, "near" constraint has no effect
MAX_FOOTPRINT_RADIUS = 1.8  # largest allowed footprint; caps oversized assets
MAX_WALL_CLEARANCE = 0.55
BASE_CAPACITY_SCALE = 1.0  # multiplier for room capacity formula

COLLISION_PADDING_BY_CLASS = {
    "compact": 0.14,
    "standard": 0.18,
    "wide": 0.24,
}

# Aesthetic/layout policy derived from semantic density.
DENSITY_MULTIPLIERS = {
    "minimal": 0.55,
    "normal": 0.82,
    "cluttered": 1.0,
}

DENSITY_BUDGET_FILL = {
    "minimal": 0.55,
    "normal": 0.8,
    "cluttered": 1.0,
}

NEAR_DISTANCE_SCALE_BY_DENSITY = {
    "minimal": 1.35,
    "normal": 1.2,
    "cluttered": 1.05,
}

NEAR_DISTANCE_PADDING_BY_DENSITY = {
    "minimal": 0.28,
    "normal": 0.2,
    "cluttered": 0.12,
}

FLOOR_ANCHOR_RATIOS = [
    (0.0, -0.42, 180.0),
    (0.0, 0.39, 0.0),
    (-0.42, -0.28, 45.0),
    (0.42, -0.28, 315.0),
    (-0.40, 0.28, 135.0),
    (0.40, 0.28, 225.0),
    (-0.56, 0.0, 90.0),
    (0.56, 0.0, 270.0),
]

WALL_LATERAL_RATIOS = [-0.78, -0.32, 0.0, 0.32, 0.78]
