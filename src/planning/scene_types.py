from __future__ import annotations

from typing import Dict, List, TypedDict

from src.placement.semantic_taxonomy import supported_runtime_roles


# Keep behavior deterministic so planner/runtime contracts stay stable.
class StyleCueSpec(TypedDict, total=False):
    style_tags: List[str]
    color_tags: List[str]
    lighting_tags: List[str]
    mood_tags: List[str]


class DesignBriefSpec(TypedDict, total=False):
    concept_statement: str
    palette_strategy: str
    material_hierarchy: List[str]
    lighting_layers: List[str]
    signature_moment: str
    visual_weight_distribution: str
    texture_profile: str
    luxury_signal_level: str
    restraint_rules: List[str]
    anti_patterns: List[str]


class SemanticSlotSpec(TypedDict, total=False):
    slot_id: str
    concept: str
    runtime_role_hint: str
    priority: str
    necessity: str
    source: str
    count: int
    capabilities: List[str]
    zone_preference: str
    rationale: str
    group_id: str


class GroundedSlotSpec(TypedDict, total=False):
    slot_id: str
    concept: str
    runtime_role: str
    subtype: str
    asset_id: str
    priority: str
    necessity: str
    source: str
    count: int
    group_id: str


class SceneRelationSpec(TypedDict, total=False):
    source_role: str
    target_role: str
    relation: str
    relation_type: str
    constraint_strength: str
    target_surface_type: str


class SceneAnchorSpec(TypedDict, total=False):
    slot_id: str
    role: str
    rationale: str


class SceneSupportSpec(TypedDict, total=False):
    role: str
    count: int
    rationale: str


class SceneGroupSpec(TypedDict, total=False):
    group_id: str
    group_type: str
    anchor_role: str
    member_role: str
    member_count: int
    layout_pattern: str
    facing_rule: str
    symmetry: str
    zone_preference: str
    importance: str


class WalkwayIntentSpec(TypedDict, total=False):
    keep_central_path_clear: bool
    keep_entry_clear: bool
    notes: str


class OptionalAdditionPolicySpec(TypedDict, total=False):
    allow_optional_additions: bool
    avoid_center_clutter: bool
    prefer_wall_accents: bool
    prefer_surface_accents: bool
    max_count: int
    max_clutter_weight: int


class SurfaceMaterialIntentSpec(TypedDict, total=False):
    wall_tags: List[str]
    floor_tags: List[str]
    ceiling_tags: List[str]
    accent_tags: List[str]


class SceneProgram(TypedDict, total=False):
    scene_type: str
    concept_label: str
    creative_summary: str
    intended_use: str
    focal_object_role: str
    focal_wall: str
    circulation_preference: str
    empty_space_preference: str
    creative_tags: List[str]
    mood_tags: List[str]
    style_descriptors: List[str]
    execution_archetype: str
    archetype: str
    design_brief: DesignBriefSpec
    semantic_slots: List[SemanticSlotSpec]
    grounded_slots: List[GroundedSlotSpec]
    relations: List[SceneRelationSpec]
    primary_anchor_object: SceneAnchorSpec
    secondary_support_objects: List[SceneSupportSpec]
    relation_graph: List[SceneRelationSpec]
    groups: List[SceneGroupSpec]
    negative_constraints: List[str]
    optional_addition_policy: OptionalAdditionPolicySpec
    surface_material_intent: SurfaceMaterialIntentSpec
    density_target: str
    symmetry_preference: str
    walkway_preservation_intent: WalkwayIntentSpec
    scene_features: List[str]
    style_cues: StyleCueSpec
    confidence: float
    source_prompt: str
    recovery_mode: str
    plausibility_warnings: List[str]

SUPPORTED_SEMANTIC_ROLES = supported_runtime_roles()
