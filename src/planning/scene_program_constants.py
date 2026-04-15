from __future__ import annotations

ALLOWED_RELATIONS = {"near", "face_to", "align", "far", "edge", "middle", "support_on", "against_wall", "centered_on_wall", "symmetry_with", "avoid"}
ALLOWED_RELATION_TYPES = {"proximity", "orientation", "support", "wall_alignment", "symmetry", "avoidance", "room_position"}
ALLOWED_CONSTRAINT_STRENGTHS = {"required", "preferred"}
ALLOWED_TARGET_SURFACE_TYPES = {"floor", "wall", "surface", "ceiling", "tabletop", "shelf"}
ALLOWED_DENSITY_TARGETS = {"sparse", "normal", "dense", "cluttered"}
ALLOWED_SYMMETRY_PREFERENCES = {"symmetric", "asymmetric", "balanced"}
ALLOWED_GROUP_TYPES = {"dining_set", "lounge_cluster", "reading_corner", "bedside_cluster", "workstation"}
ALLOWED_GROUP_LAYOUTS = {"paired_long_sides", "ring", "arc", "front_facing_cluster", "beside_anchor"}
ALLOWED_FACING_RULES = {"toward_anchor", "toward_focal_object", "parallel", "none"}
ALLOWED_ZONE_PREFERENCES = {"center", "edge", "corner", "front", "back", "left", "right"}
ALLOWED_GROUP_IMPORTANCE = {"primary", "secondary", "background"}
ALLOWED_FOCAL_WALLS = {"front", "back", "left", "right", "none"}
ALLOWED_CIRCULATION_PREFERENCES = {"clear_center", "clear_entry", "balanced", "layered"}
ALLOWED_EMPTY_SPACE_PREFERENCES = {"open_center", "balanced", "layered"}
ALLOWED_SLOT_NECESSITIES = {"core", "support", "enrichment"}
ALLOWED_SLOT_SOURCES = {"explicit_prompt", "inferred_function", "style_enrichment", "deterministic_completion"}
ALLOWED_OPTIONAL_PLACEMENT_HINTS = {
    "wall_centered",
    "wall_left",
    "wall_right",
    "wall_above_anchor",
    "corner_left",
    "corner_right",
    "tabletop_center",
    "tabletop_edge",
    "ceiling_center",
    "floor_edge",
}
REQUIRED_SURFACE_MATERIAL_SLOTS = ("wall", "floor", "ceiling")
ALLOWED_BUDGET_KEYS = (
    "max_props",
    "max_props_hard",
    "max_floor_objects",
    "max_wall_objects",
    "max_surface_objects",
    "max_texture_tier",
    "max_lights",
    "max_clutter_weight",
)
