from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from src.placement.constraints import normalize_anchor_preferences
from src.placement.geometry import (
    canonicalize_semantic_concept,
    canonicalize_semantic_role,
    map_semantic_concept_to_runtime_role,
    normalize_density_profile,
    normalize_layout_mood,
)
from src.planning.scene_types import (
    SUPPORTED_SEMANTIC_ROLES,
    DesignBriefSpec,
    OptionalAdditionPolicySpec,
    SceneAnchorSpec,
    SceneGroupSpec,
    SceneProgram,
    SceneRelationSpec,
    SceneSupportSpec,
    SemanticSlotSpec,
    StyleCueSpec,
    SurfaceMaterialIntentSpec,
    WalkwayIntentSpec,
)
from src.planning.scene_program_common import (
    _dedupe_strings,
    _derive_role_fields_from_slots,
    _known_roles_from_slots,
    _scene_slots,
    _slot_role,
    _supported_archetypes,
    _known_scene_roles,
    _normalize_archetype,
    _normalize_descriptor_list,
    _normalize_feature_list,
    _normalize_feature_token,
    _normalize_notes,
    _normalize_scene_choice,
    _normalize_tag_list,
    _slot_necessity,
    _slot_priority,
    _slot_source,
)
from src.planning.scene_program_policy import policy_set, relation_type_default
from src.planning.scene_program_grounding import _plausibility_warnings

ALLOWED_CIRCULATION_PREFERENCES = policy_set("allowed_circulation_preferences")
ALLOWED_CONSTRAINT_STRENGTHS = policy_set("allowed_constraint_strengths")
ALLOWED_DENSITY_TARGETS = policy_set("allowed_density_targets")
ALLOWED_EMPTY_SPACE_PREFERENCES = policy_set("allowed_empty_space_preferences")
ALLOWED_FACING_RULES = policy_set("allowed_facing_rules")
ALLOWED_FOCAL_WALLS = policy_set("allowed_focal_walls")
ALLOWED_GROUP_IMPORTANCE = policy_set("allowed_group_importance")
ALLOWED_GROUP_LAYOUTS = policy_set("allowed_group_layouts")
ALLOWED_GROUP_TYPES = policy_set("allowed_group_types")
ALLOWED_RELATION_TYPES = policy_set("allowed_relation_types")
ALLOWED_RELATIONS = policy_set("allowed_relations")
ALLOWED_SYMMETRY_PREFERENCES = policy_set("allowed_symmetry_preferences")
ALLOWED_TARGET_SURFACE_TYPES = policy_set("allowed_target_surface_types")
ALLOWED_ZONE_PREFERENCES = policy_set("allowed_zone_preferences")


def _default_relation_type(relation: str) -> str:
    return relation_type_default(relation)

def _normalize_style_cues(intent_spec: Dict[str, Any]) -> StyleCueSpec:  # consolidates stylistic, color, lighting, and mood cues from intent payload
    raw_style_cues = intent_spec.get("style_cues") if isinstance(intent_spec.get("style_cues"), dict) else {}
    style_cues: StyleCueSpec = {}
    style_tags = _normalize_tag_list(raw_style_cues.get("style_tags")) or _normalize_tag_list(intent_spec.get("style_tags"))
    color_tags = _normalize_tag_list(raw_style_cues.get("color_tags")) or _normalize_tag_list(intent_spec.get("color_tags"))
    lighting_tags = _normalize_tag_list(raw_style_cues.get("lighting_tags"))
    mood_tags = _normalize_tag_list(raw_style_cues.get("mood_tags"))
    if style_tags:
        style_cues["style_tags"] = style_tags
    if color_tags:
        style_cues["color_tags"] = color_tags
    if lighting_tags:
        style_cues["lighting_tags"] = lighting_tags
    if mood_tags:
        style_cues["mood_tags"] = mood_tags
    return style_cues


def _normalize_design_brief(value: Any) -> DesignBriefSpec:
    if not isinstance(value, dict):
        return {}
    out: DesignBriefSpec = {}
    text_fields = {
        "concept_statement",
        "palette_strategy",
        "signature_moment",
        "visual_weight_distribution",
        "texture_profile",
        "luxury_signal_level",
    }
    list_fields = {"material_hierarchy", "lighting_layers", "restraint_rules", "anti_patterns"}
    for key in text_fields:
        token = _normalize_notes(value.get(key))
        if token:
            out[key] = token
    for key in list_fields:
        values = _normalize_tag_list(value.get(key)) if key in {"material_hierarchy", "lighting_layers"} else _normalize_descriptor_list(value.get(key))
        if values:
            out[key] = values
    return out


def _normalize_semantic_slots(values: Any) -> List[SemanticSlotSpec]:
    if not isinstance(values, list):
        return []
    out: List[SemanticSlotSpec] = []
    seen_slot_ids: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            continue
        slot_id = _normalize_feature_token(value.get("slot_id")) or f"slot_{index + 1}"
        if slot_id in seen_slot_ids:
            continue
        concept = canonicalize_semantic_concept(value.get("concept") or value.get("runtime_role_hint") or value.get("slot_id"))
        runtime_role_hint = canonicalize_semantic_role(value.get("runtime_role_hint") or concept)
        count = value.get("count")
        entry: SemanticSlotSpec = {
            "slot_id": slot_id,
            "concept": concept or runtime_role_hint,
            "priority": _slot_priority(value.get("priority")),
            "count": max(1, int(count)) if isinstance(count, (int, float)) else 1,
        }
        necessity = _slot_necessity(value.get("necessity"))
        source = _slot_source(value.get("source"))
        if necessity:
            entry["necessity"] = necessity
        if source:
            entry["source"] = source
        if runtime_role_hint in SUPPORTED_SEMANTIC_ROLES:
            entry["runtime_role_hint"] = runtime_role_hint
        capabilities = _normalize_feature_list(value.get("capabilities"))
        if capabilities:
            entry["capabilities"] = capabilities
        zone_preference = _normalize_feature_token(value.get("zone_preference"))
        if zone_preference in ALLOWED_ZONE_PREFERENCES:
            entry["zone_preference"] = zone_preference
        rationale = _normalize_notes(value.get("rationale"))
        if rationale:
            entry["rationale"] = rationale
        group_id = _normalize_feature_token(value.get("group_id"))
        if group_id:
            entry["group_id"] = group_id
        out.append(entry)
        seen_slot_ids.add(slot_id)
    return out


def _normalize_anchor_object(value: Any, known_roles: set[str], semantic_slots: List[SemanticSlotSpec]) -> SceneAnchorSpec:  # validates the primary anchor role specification
    if not isinstance(value, dict):
        return {}
    role = canonicalize_semantic_role(value.get("role"))
    rationale = _normalize_notes(value.get("rationale"))
    known_slot_ids = {
        str(slot.get("slot_id") or "").strip()
        for slot in semantic_slots
        if isinstance(slot, dict) and str(slot.get("slot_id") or "").strip()
    }
    slot_id = str(value.get("slot_id") or "").strip()
    out: SceneAnchorSpec = {}
    if slot_id in known_slot_ids:
        out["slot_id"] = slot_id
    if role in known_roles:
        out["role"] = role
    if rationale:
        out["rationale"] = rationale
    return out


def _normalize_support_objects(values: Any, known_roles: set[str]) -> List[SceneSupportSpec]:  # validates the secondary support role specifications
    if not isinstance(values, list):
        return []
    out: List[SceneSupportSpec] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        role = canonicalize_semantic_role(value.get("role"))
        if role not in known_roles:
            continue
        count = value.get("count")
        rationale = _normalize_notes(value.get("rationale"))
        entry: SceneSupportSpec = {"role": role, "count": max(1, int(count)) if isinstance(count, (int, float)) else 1}
        if rationale:
            entry["rationale"] = rationale
        out.append(entry)
    return out


def _normalize_scene_groups(values: Any, known_roles: set[str]) -> List[SceneGroupSpec]:
    if not isinstance(values, list):
        return []
    out: List[SceneGroupSpec] = []
    seen_group_ids: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            continue
        group_type = _normalize_feature_token(value.get("group_type"))
        anchor_role = canonicalize_semantic_role(value.get("anchor_role"))
        member_role = canonicalize_semantic_role(value.get("member_role"))
        layout_pattern = _normalize_feature_token(value.get("layout_pattern"))
        facing_rule = _normalize_feature_token(value.get("facing_rule"))
        symmetry = _normalize_feature_token(value.get("symmetry"))
        zone_preference = _normalize_feature_token(value.get("zone_preference"))
        importance = _normalize_feature_token(value.get("importance")) or "primary"
        group_id = _normalize_feature_token(value.get("group_id")) or f"group_{index + 1}"
        member_count = value.get("member_count")
        if (
            group_type not in ALLOWED_GROUP_TYPES
            or anchor_role not in known_roles
            or member_role not in known_roles
            or layout_pattern not in ALLOWED_GROUP_LAYOUTS
            or facing_rule not in ALLOWED_FACING_RULES
            or symmetry not in ALLOWED_SYMMETRY_PREFERENCES
            or zone_preference not in ALLOWED_ZONE_PREFERENCES
            or importance not in ALLOWED_GROUP_IMPORTANCE
            or not isinstance(member_count, (int, float))
            or int(member_count) <= 0
            or group_id in seen_group_ids
        ):
            continue
        seen_group_ids.add(group_id)
        out.append(
            {
                "group_id": group_id,
                "group_type": group_type,
                "anchor_role": anchor_role,
                "member_role": member_role,
                "member_count": max(1, int(member_count)),
                "layout_pattern": layout_pattern,
                "facing_rule": facing_rule,
                "symmetry": symmetry,
                "zone_preference": zone_preference,
                "importance": importance,
            }
        )
    return out


def _normalize_negative_constraints(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    return _dedupe_strings(
        _normalize_feature_token(value)
        for value in values
        if isinstance(value, str) and _normalize_feature_token(value)
    )


def _normalize_optional_addition_policy(value: Any) -> OptionalAdditionPolicySpec:
    if not isinstance(value, dict):
        return {}
    out: OptionalAdditionPolicySpec = {}
    if "allow_optional_additions" in value:
        out["allow_optional_additions"] = bool(value.get("allow_optional_additions"))
    if "avoid_center_clutter" in value:
        out["avoid_center_clutter"] = bool(value.get("avoid_center_clutter"))
    if "prefer_wall_accents" in value:
        out["prefer_wall_accents"] = bool(value.get("prefer_wall_accents"))
    if "prefer_surface_accents" in value:
        out["prefer_surface_accents"] = bool(value.get("prefer_surface_accents"))
    max_count = value.get("max_count")
    if isinstance(max_count, (int, float)) and int(max_count) >= 0:
        out["max_count"] = int(max_count)
    max_clutter_weight = value.get("max_clutter_weight")
    if isinstance(max_clutter_weight, (int, float)) and int(max_clutter_weight) >= 0:
        out["max_clutter_weight"] = int(max_clutter_weight)
    return out


def _normalize_surface_material_intent(value: Any) -> SurfaceMaterialIntentSpec:
    if not isinstance(value, dict):
        return {}
    out: SurfaceMaterialIntentSpec = {}
    for key in ("wall_tags", "floor_tags", "ceiling_tags", "accent_tags"):
        normalized = _normalize_tag_list(value.get(key))
        if normalized:
            out[key] = normalized
    return out


def _normalize_walkway_intent(value: Any) -> WalkwayIntentSpec:  # parses explicit LLM intent regarding pathing and VR comfort zones
    if not isinstance(value, dict):
        return {}
    out: WalkwayIntentSpec = {}
    if "keep_central_path_clear" in value:
        out["keep_central_path_clear"] = bool(value.get("keep_central_path_clear"))
    if "keep_entry_clear" in value:
        out["keep_entry_clear"] = bool(value.get("keep_entry_clear"))
    notes = _normalize_notes(value.get("notes"))
    if notes:
        out["notes"] = notes
    return out


def _normalize_relations(values: Any, known_roles: set[str]) -> List[SceneRelationSpec]:  # validates and deduplicates the relation graph entries
    out: List[SceneRelationSpec] = []
    if not isinstance(values, list):
        return out
    seen: set[tuple[str, str, str]] = set()
    for value in values:
        if not isinstance(value, dict):
            continue
        source_role = canonicalize_semantic_role(value.get("source_role"))
        target_value = str(value.get("target_role") or "").strip().lower()
        target_role = "room" if target_value == "room" else canonicalize_semantic_role(target_value)
        relation = str(value.get("relation") or "").strip().lower()
        relation_type = _normalize_feature_token(value.get("relation_type")) or _default_relation_type(relation)
        constraint_strength = _normalize_feature_token(value.get("constraint_strength")) or "preferred"
        target_surface_type = _normalize_feature_token(value.get("target_surface_type"))
        if relation not in ALLOWED_RELATIONS or not source_role or not target_role:
            continue
        if relation_type not in ALLOWED_RELATION_TYPES or constraint_strength not in ALLOWED_CONSTRAINT_STRENGTHS:
            continue
        if target_surface_type and target_surface_type not in ALLOWED_TARGET_SURFACE_TYPES:
            continue
        if source_role not in known_roles:
            continue
        if target_role != "room" and target_role not in known_roles:
            continue
        if target_role == "room" and relation not in {"edge", "middle", "against_wall", "centered_on_wall"}:
            continue
        if relation == "support_on" and target_role == "room":
            continue
        key = (source_role, target_role, relation)
        if key in seen:
            continue
        seen.add(key)
        entry: SceneRelationSpec = {
            "source_role": source_role,
            "target_role": target_role,
            "relation": relation,
            "relation_type": relation_type,
            "constraint_strength": constraint_strength,
        }
        if target_surface_type:
            entry["target_surface_type"] = target_surface_type
        out.append(entry)
    return out


def _normalize_raw_placement_intent(raw_intent: Any, known_roles: set[str]) -> Dict[str, Any]:  # normalizes adjacency pairs, spatial prefs, and density into a placement intent
    if not isinstance(raw_intent, dict):
        return {}

    adjacency_pairs = []
    for relation in _normalize_relations(raw_intent.get("adjacency_pairs"), known_roles):
        if relation["relation"] in policy_set("adjacency_relations"):
            adjacency_pairs.append(relation)

    spatial_preferences: List[Dict[str, str]] = []
    seen_spatial: set[tuple[str, str]] = set()
    raw_spatial_preferences = raw_intent.get("spatial_preferences")
    if not isinstance(raw_spatial_preferences, list):
        raw_spatial_preferences = []
    for entry in raw_spatial_preferences:
        if not isinstance(entry, dict):
            continue
        role = canonicalize_semantic_role(entry.get("role") or entry.get("source_role"))
        relation = str(entry.get("relation") or "").strip().lower()
        if relation not in policy_set("spatial_position_relations") or role not in known_roles:
            continue
        key = (role, relation)
        if key in seen_spatial:
            continue
        seen_spatial.add(key)
        spatial_preferences.append({"role": role, "relation": relation})

    density_profile = normalize_density_profile(raw_intent.get("density_profile"))
    return {
        "density_profile": density_profile,
        "anchor_preferences": normalize_anchor_preferences(raw_intent.get("anchor_preferences")),
        "adjacency_pairs": adjacency_pairs,
        "spatial_preferences": spatial_preferences,
        "layout_mood": normalize_layout_mood(raw_intent.get("layout_mood"), density_profile),
    }


def normalize_scene_program(intent_spec: Dict[str, Any] | None, prompt_text: str) -> SceneProgram:  # converts raw LLM intent into a fully normalized SceneProgram
    intent_spec = intent_spec if isinstance(intent_spec, dict) else {}
    raw_archetype = _normalize_archetype(intent_spec.get("execution_archetype") or intent_spec.get("archetype"))
    scene_type = _normalize_feature_token(intent_spec.get("scene_type")) or raw_archetype or "generic_room"
    concept_label = _normalize_feature_token(intent_spec.get("concept_label")) or scene_type or raw_archetype or "room"
    creative_summary = _normalize_notes(intent_spec.get("creative_summary"))
    intended_use = _normalize_notes(intent_spec.get("intended_use"))
    creative_tags = _normalize_feature_list(intent_spec.get("creative_tags"))
    semantic_slots = _normalize_semantic_slots(intent_spec.get("semantic_slots"))
    known_roles = _known_roles_from_slots(semantic_slots)

    relations = _normalize_relations(intent_spec.get("relations"), known_roles)
    relation_graph = _normalize_relations(intent_spec.get("relation_graph"), known_roles)
    confidence_raw = intent_spec.get("confidence")
    confidence = 0.0
    if isinstance(confidence_raw, (int, float)):
        confidence = max(0.0, min(float(confidence_raw), 1.0))
    style_cues = _normalize_style_cues(intent_spec)
    mood_tags = _normalize_tag_list(intent_spec.get("mood_tags")) or list(style_cues.get("mood_tags", []))
    style_descriptors = _normalize_descriptor_list(intent_spec.get("style_descriptors"))
    focal_object_role = canonicalize_semantic_role(intent_spec.get("focal_object_role"))
    if focal_object_role not in known_roles:
        focal_object_role = ""

    design_brief = _normalize_design_brief(intent_spec.get("design_brief"))
    scene_program: SceneProgram = {
        "scene_type": scene_type,
        "concept_label": concept_label,
        "creative_summary": creative_summary,
        "intended_use": intended_use,
        "focal_object_role": focal_object_role,
        "focal_wall": _normalize_scene_choice(intent_spec.get("focal_wall"), ALLOWED_FOCAL_WALLS, "none"),
        "circulation_preference": _normalize_scene_choice(
            intent_spec.get("circulation_preference"),
            ALLOWED_CIRCULATION_PREFERENCES,
            "balanced",
        ),
        "empty_space_preference": _normalize_scene_choice(
            intent_spec.get("empty_space_preference"),
            ALLOWED_EMPTY_SPACE_PREFERENCES,
            "balanced",
        ),
        "creative_tags": creative_tags,
        "mood_tags": mood_tags,
        "style_descriptors": style_descriptors,
        "execution_archetype": raw_archetype,
        "archetype": raw_archetype,
        "design_brief": design_brief,
        "semantic_slots": semantic_slots,
        "grounded_slots": [],
        "relations": relations,
        "primary_anchor_object": _normalize_anchor_object(intent_spec.get("primary_anchor_object"), known_roles, semantic_slots),
        "secondary_support_objects": _normalize_support_objects(intent_spec.get("secondary_support_objects"), known_roles),
        "relation_graph": relation_graph,
        "groups": _normalize_scene_groups(intent_spec.get("groups"), known_roles),
        "negative_constraints": _normalize_negative_constraints(intent_spec.get("negative_constraints")),
        "optional_addition_policy": _normalize_optional_addition_policy(intent_spec.get("optional_addition_policy")),
        "surface_material_intent": _normalize_surface_material_intent(intent_spec.get("surface_material_intent")),
        "density_target": _normalize_feature_token(intent_spec.get("density_target")) if _normalize_feature_token(intent_spec.get("density_target")) in ALLOWED_DENSITY_TARGETS else "normal",
        "symmetry_preference": _normalize_feature_token(intent_spec.get("symmetry_preference")) if _normalize_feature_token(intent_spec.get("symmetry_preference")) in ALLOWED_SYMMETRY_PREFERENCES else "balanced",
        "walkway_preservation_intent": _normalize_walkway_intent(intent_spec.get("walkway_preservation_intent")),
        "scene_features": _normalize_feature_list(intent_spec.get("scene_features")),
        "style_cues": style_cues,
        "confidence": confidence,
        "source_prompt": (prompt_text or "").strip(),
        "recovery_mode": "llm",
        "plausibility_warnings": [],
    }
    scene_program["plausibility_warnings"] = _plausibility_warnings(scene_program)
    return scene_program


def scene_program_to_intent_spec(scene_program: SceneProgram) -> Dict[str, Any]:  # converts SceneProgram back to the flat intent_spec dict used by downstream modules
    style_cues = scene_program.get("style_cues", {})
    return {
        "scene_type": scene_program.get("scene_type", "generic_room"),
        "concept_label": scene_program.get("concept_label", ""),
        "creative_summary": scene_program.get("creative_summary", ""),
        "intended_use": scene_program.get("intended_use", ""),
        "focal_object_role": scene_program.get("focal_object_role", ""),
        "focal_wall": scene_program.get("focal_wall", "none"),
        "circulation_preference": scene_program.get("circulation_preference", "balanced"),
        "empty_space_preference": scene_program.get("empty_space_preference", "balanced"),
        "creative_tags": list(scene_program.get("creative_tags", [])),
        "mood_tags": list(scene_program.get("mood_tags", [])),
        "style_descriptors": list(scene_program.get("style_descriptors", [])),
        "design_brief": dict(scene_program.get("design_brief", {})),
        "semantic_slots": list(scene_program.get("semantic_slots", [])),
        "grounded_slots": list(scene_program.get("grounded_slots", [])),
        "primary_anchor_object": dict(scene_program.get("primary_anchor_object", {})),
        "secondary_support_objects": list(scene_program.get("secondary_support_objects", [])),
        "relation_graph": list(scene_program.get("relation_graph", [])),
        "groups": list(scene_program.get("groups", [])),
        "negative_constraints": list(scene_program.get("negative_constraints", [])),
        "optional_addition_policy": dict(scene_program.get("optional_addition_policy", {})),
        "surface_material_intent": dict(scene_program.get("surface_material_intent", {})),
        "density_target": scene_program.get("density_target", "normal"),
        "symmetry_preference": scene_program.get("symmetry_preference", "balanced"),
        "walkway_preservation_intent": dict(scene_program.get("walkway_preservation_intent", {})),
        "scene_features": list(scene_program.get("scene_features", [])),
        "style_tags": list(style_cues.get("style_tags", [])),
        "color_tags": list(style_cues.get("color_tags", [])),
        "style_cues": dict(style_cues),
        "confidence": float(scene_program.get("confidence", 0.0) or 0.0),
        "execution_archetype": scene_program.get("execution_archetype", "generic_room"),
        "archetype": scene_program.get("archetype", "generic_room"),
        "plausibility_warnings": list(scene_program.get("plausibility_warnings", [])),
    }


def public_scene_program(scene_program: SceneProgram | Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(scene_program, dict):
        return {}
    return dict(scene_program)


def public_intent_spec(intent_spec: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(intent_spec, dict):
        return {}
    return dict(intent_spec)


def scene_program_to_placement_intent(  # extracts adjacency pairs and spatial preferences from a scene program
    scene_program: SceneProgram,
    raw_placement_intent: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    known_roles = _known_scene_roles(scene_program)
    return _normalize_raw_placement_intent(raw_placement_intent, known_roles)


def normalize_intent_spec(intent_spec: Dict[str, Any] | None, prompt_text: str) -> Dict[str, Any]:  # converts raw string representation back to dictionary after pipeline mutation
    return scene_program_to_intent_spec(normalize_scene_program(intent_spec, prompt_text))


def _scene_program_errors(raw_intent: Any, scene_program: SceneProgram) -> List[Dict[str, str]]:  # validates that the scene program has required archetype, roles, anchor, and relations
    errors: List[Dict[str, str]] = []
    if not isinstance(raw_intent, dict):
        return [{"path": "$.llm.intent", "message": "Semantic planner intent must be an object."}]

    if not _normalize_archetype(raw_intent.get("execution_archetype") or raw_intent.get("archetype")):
        errors.append({
            "path": "$.llm.intent.execution_archetype",
            "message": f"Semantic planner must choose a supported execution_archetype enum: {', '.join(sorted(_supported_archetypes()))}.",
        })

    known_roles = _known_scene_roles(scene_program)
    semantic_slots = _scene_slots(scene_program)
    if not semantic_slots and not known_roles:
        errors.append({
            "path": "$.llm.intent.semantic_slots",
            "message": "Semantic planner must return at least one supported semantic slot.",
        })

    anchor_role = str((scene_program.get("primary_anchor_object") or {}).get("role") or "").strip().lower()
    if len(known_roles) >= 2 and not anchor_role:
        errors.append({
            "path": "$.llm.intent.primary_anchor_object.role",
            "message": "Semantic planner must identify a primary anchor role.",
        })
    elif anchor_role and anchor_role not in known_roles:
        errors.append({
            "path": "$.llm.intent.primary_anchor_object.role",
            "message": "Primary anchor role must be one of the returned scene roles.",
        })

    if len(known_roles) >= 2 and not scene_program["relation_graph"]:
        errors.append({
            "path": "$.llm.intent.relation_graph",
            "message": "Semantic planner must return a relation graph for multi-object rooms.",
        })

    warnings = list(scene_program.get("plausibility_warnings") or [])
    if len(known_roles) >= 2 and not scene_program["groups"]:
        warnings.append("multi_object_scene_missing_groups")

    repeated_roles = Counter()
    for slot in semantic_slots:
        role = _slot_role(slot)
        if role in SUPPORTED_SEMANTIC_ROLES:
            repeated_roles[role] += max(1, int(slot.get("count") or 1))
    grouped_roles = {
        str(group.get("anchor_role") or "").strip().lower()
        for group in scene_program.get("groups", [])
        if isinstance(group, dict)
    } | {
        str(group.get("member_role") or "").strip().lower()
        for group in scene_program.get("groups", [])
        if isinstance(group, dict)
    }
    uncovered_repeats = sorted(role for role, count in repeated_roles.items() if count > 1 and role not in grouped_roles)
    if uncovered_repeats:
        warnings.append(f"repeated_roles_without_groups:{','.join(uncovered_repeats)}")

    if warnings:
        scene_program["plausibility_warnings"] = list(dict.fromkeys(warnings))

    return errors


def _placement_intent_errors(raw_intent: Any, placement_intent: Dict[str, Any]) -> List[Dict[str, str]]:  # catches missing density/mood targets which are crucial for the placer algorithm
    if not isinstance(raw_intent, dict):
        return [{"path": "$.llm.placement_intent", "message": "Semantic planner must return a placement_intent object."}]

    if not placement_intent["density_profile"]:
        return [{"path": "$.llm.placement_intent.density_profile", "message": "placement_intent must include a density_profile."}]
    if not placement_intent.get("layout_mood"):
        return [{"path": "$.llm.placement_intent.layout_mood", "message": "placement_intent must include a layout_mood."}]
    return []


def validate_semantic_intent(  # validates an LLM intent response: checks archetype, roles, anchor, and placement_intent
    llm_intent: Dict[str, Any],
    *,
    prompt_text: str,
) -> Dict[str, Any]:
    if not isinstance(llm_intent, dict):
        return {
            "ok": False,
            "error_code": "semantic_invalid_intent",
            "message": "Semantic planner response was not an object.",
            "errors": [{"path": "$.llm", "message": "Semantic planner response was not an object."}],
        }

    raw_intent = llm_intent.get("intent")
    raw_placement_intent = llm_intent.get("placement_intent")
    if isinstance(raw_intent, dict) and isinstance(llm_intent.get("design_brief"), dict) and not isinstance(raw_intent.get("design_brief"), dict):
        raw_intent = dict(raw_intent)
        raw_intent["design_brief"] = dict(llm_intent.get("design_brief") or {})

    scene_program = normalize_scene_program(raw_intent, prompt_text)
    intent_spec = scene_program_to_intent_spec(scene_program)
    placement_intent = scene_program_to_placement_intent(scene_program, raw_placement_intent)
    errors = _scene_program_errors(raw_intent, scene_program) + _placement_intent_errors(raw_placement_intent, placement_intent)
    if errors:
        return {
            "ok": False,
            "error_code": "semantic_invalid_intent",
            "message": "Semantic planner returned an invalid intent.",
            "scene_program": scene_program,
            "intent_spec": intent_spec,
            "placement_intent": placement_intent,
            "errors": errors,
        }

    return {
        "ok": True,
        "scene_program": scene_program,
        "intent_spec": intent_spec,
        "placement_intent": placement_intent,
        "warnings": list(scene_program.get("plausibility_warnings") or []),
    }

