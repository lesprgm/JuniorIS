import json

from src.catalog.pack_registry import PackRegistry, load_pack_registry
from src.catalog.style_material_pool import build_surface_material_candidates
from src.placement.geometry import canonicalize_semantic_concept, canonicalize_semantic_role, map_semantic_concept_to_runtime_role
from src.planning.planner import plan_worldspec
from src.planning.scene_program import complete_scene_program, ground_scene_program
from src.planning.assets import build_layout_from_selected_assets, build_semantic_candidate_shortlist, collect_assets
from src.planning.semantics import apply_stylekit_colors, validate_semantic_intent, validate_semantic_plan
from src.catalog.stylekit_registry import StyleKitRegistry, load_stylekit_registry
from src.world.validation import validate_worldspec
from tests.semantic_test_utils import approved_surface_material_selection, shortlist_asset_ids



# Keep behavior deterministic so planner/runtime contracts stay stable.
def _surface_material_candidates(style_tags: list[str] | None = None, color_tags: list[str] | None = None):
    return build_surface_material_candidates(
        {
            "style_tags": style_tags or [],
            "color_tags": color_tags or [],
            "style_cues": {"style_tags": style_tags or [], "color_tags": color_tags or []},
        }
    )

def _strict_placement_intent(
    *,
    density_profile: str = "normal",
    adjacency_pairs: list[dict] | None = None,
    spatial_preferences: list[dict] | None = None,
    layout_mood: str = "cozy",
):
    return {
        "density_profile": density_profile,
        "anchor_preferences": [],
        "adjacency_pairs": adjacency_pairs or [],
        "spatial_preferences": spatial_preferences or [],
        "layout_mood": layout_mood,
    }


def _semantic_slots_from_roles(
    required_roles: list[str],
    optional_roles: list[str] | None = None,
    role_counts: dict[str, int] | None = None,
) -> list[dict]:
    optional_roles = optional_roles or []
    role_counts = role_counts or {}
    slots: list[dict] = []
    seen_counts: dict[str, int] = {}
    ordered_roles = required_roles + [role for role in optional_roles if role not in required_roles]
    for role in ordered_roles:
        canonical_role = canonicalize_semantic_role(role)
        seen_counts[canonical_role] = seen_counts.get(canonical_role, 0) + 1
        slots.append(
            {
                "slot_id": f"{canonical_role}_slot_{seen_counts[canonical_role]}",
                "concept": canonical_role,
                "priority": "optional" if role in optional_roles and role not in required_roles else "must",
                "necessity": "enrichment" if role in optional_roles and role not in required_roles else "core",
                "source": "style_enrichment" if role in optional_roles and role not in required_roles else "explicit_prompt",
                "count": max(1, int(role_counts.get(role, 1))),
                "runtime_role_hint": canonical_role,
            }
        )
    return slots


def _strict_intent(
    *,
    required_roles: list[str],
    optional_roles: list[str] | None = None,
    archetype: str = "study",
    scene_type: str | None = None,
    role_counts: dict[str, int] | None = None,
    style_tags: list[str] | None = None,
    color_tags: list[str] | None = None,
    scene_features: list[str] | None = None,
    density_target: str = "normal",
    symmetry_preference: str = "balanced",
    confidence: float = 0.9,
    primary_anchor_role: str | None = None,
    relation_graph: list[dict] | None = None,
    secondary_support_objects: list[dict] | None = None,
    groups: list[dict] | None = None,
):
    optional_roles = optional_roles or []
    role_counts = role_counts or {}
    style_tags = style_tags or []
    color_tags = color_tags or []
    scene_features = scene_features or []
    all_roles = required_roles + [role for role in optional_roles if role not in required_roles]
    anchor_role = primary_anchor_role or ("table" if "table" in all_roles else all_roles[0])
    support_objects = secondary_support_objects
    if support_objects is None:
        support_objects = [
            {"role": role, "count": max(1, int(role_counts.get(role, 1))), "rationale": "support object"}
            for role in all_roles
            if role != anchor_role
        ]
    if relation_graph is None:
        relation_graph = [{"source_role": anchor_role, "target_role": "room", "relation": "middle"}]
        for role in all_roles:
            if role == anchor_role:
                continue
            relation = "face_to" if role == "chair" and anchor_role == "table" else "near"
            relation_graph.append({"source_role": role, "target_role": anchor_role, "relation": relation})
    if groups is None and len(all_roles) >= 2:
        if "table" in all_roles and "chair" in all_roles:
            chair_count = max(1, int(role_counts.get("chair", 1)))
            group_type = "dining_set" if chair_count > 1 or archetype in {"kitchen", "lounge"} else "reading_corner"
            groups = [
                {
                    "group_id": "group_1",
                    "group_type": group_type,
                    "anchor_role": "table" if group_type == "dining_set" else "chair",
                    "member_role": "chair" if group_type == "dining_set" else "table",
                    "member_count": chair_count if group_type == "dining_set" else 1,
                    "layout_pattern": "paired_long_sides" if group_type == "dining_set" else "beside_anchor",
                    "facing_rule": "toward_anchor",
                    "symmetry": symmetry_preference,
                    "zone_preference": "center" if group_type == "dining_set" else "corner",
                }
            ]
        elif "bed" in all_roles and "table" in all_roles:
            groups = [
                {
                    "group_id": "group_1",
                    "group_type": "bedside_cluster",
                    "anchor_role": "bed",
                    "member_role": "table",
                    "member_count": max(1, int(role_counts.get("table", 1))),
                    "layout_pattern": "beside_anchor",
                    "facing_rule": "parallel",
                    "symmetry": symmetry_preference,
                    "zone_preference": "back",
                }
            ]
        else:
            groups = [
                {
                    "group_id": "group_1",
                    "group_type": "workstation",
                    "anchor_role": anchor_role,
                    "member_role": next(role for role in all_roles if role != anchor_role),
                    "member_count": 1,
                    "layout_pattern": "beside_anchor",
                    "facing_rule": "toward_anchor",
                    "symmetry": symmetry_preference,
                    "zone_preference": "edge",
                }
            ]
    semantic_slots = _semantic_slots_from_roles(required_roles, optional_roles, role_counts)
    return {
        "scene_type": scene_type or archetype,
        "concept_label": (scene_type or archetype),
        "creative_summary": f"{(scene_type or archetype).replace('_', ' ')} room",
        "intended_use": f"use as a {(scene_type or archetype).replace('_', ' ')}",
        "focal_object_role": anchor_role,
        "focal_wall": "front",
        "circulation_preference": "clear_center",
        "empty_space_preference": "balanced",
        "creative_tags": [scene_type or archetype],
        "mood_tags": style_tags[:1],
        "style_descriptors": style_tags,
        "execution_archetype": archetype,
        "archetype": archetype,
        "semantic_slots": semantic_slots,
        "primary_anchor_object": {
            "slot_id": next((slot["slot_id"] for slot in semantic_slots if slot.get("runtime_role_hint") == anchor_role), ""),
            "role": anchor_role,
            "rationale": "primary anchor",
        },
        "secondary_support_objects": support_objects,
        "relation_graph": relation_graph,
        "groups": groups or [],
        "negative_constraints": [],
        "optional_addition_policy": {
            "allow_optional_additions": True,
            "avoid_center_clutter": True,
            "prefer_wall_accents": True,
            "max_count": 2,
        },
        "surface_material_intent": {
            "wall_tags": style_tags[:1],
            "floor_tags": color_tags[:1],
            "ceiling_tags": [],
            "accent_tags": [],
        },
        "density_target": density_target,
        "symmetry_preference": symmetry_preference,
        "walkway_preservation_intent": {
            "keep_central_path_clear": True,
            "keep_entry_clear": True,
            "notes": "preserve a clear path through the room",
        },
        "scene_features": scene_features,
        "style_tags": style_tags,
        "color_tags": color_tags,
        "style_cues": {
            "style_tags": style_tags,
            "color_tags": color_tags,
            "lighting_tags": [],
            "mood_tags": [],
        },
        "confidence": confidence,
    }

def _assert_llm_error(result, error_code: str):
    assert result["ok"] is False
    assert result["error_code"] == error_code
    assert result["semantic_path_status"] == "failed"
    assert any(err["path"] == "$.llm" for err in result["errors"])


def _slot_asset_map_from_roles(intent: dict, asset_ids_by_role: dict[str, str]) -> dict[str, str]:
    slot_ids_by_role: dict[str, list[str]] = {}
    for slot in intent.get("semantic_slots", []):
        if not isinstance(slot, dict):
            continue
        role = str(slot.get("concept") or slot.get("runtime_role_hint"))
        slot_id = str(slot.get("slot_id") or "")
        if role and slot_id:
            slot_ids_by_role.setdefault(role, []).append(slot_id)
    return {
        slot_id: asset_id
        for role, asset_id in asset_ids_by_role.items()
        for slot_id in slot_ids_by_role.get(role, [])
    }


def _inline_llm_plan(
    asset_ids,
    *,
    max_props: int,
    selected_prompt: str = "a compact modern room",
    required_roles: list[str] | None = None,
    optional_additions: list[dict[str, str]] | None = None,
):
    required_roles = required_roles or ["chair", "table", "lamp"][: len(asset_ids)]
    adjacency_pairs = []
    if "chair" in required_roles and "table" in required_roles:
        adjacency_pairs.append({"source_role": "chair", "target_role": "table", "relation": "near"})
    if "chair" in required_roles and "lamp" in required_roles:
        adjacency_pairs.append({"source_role": "chair", "target_role": "lamp", "relation": "near"})

    intent = _strict_intent(
        required_roles=required_roles,
        archetype="study",
        scene_type="study",
        style_tags=["cozy"],
        color_tags=["warm"],
        confidence=0.9,
        role_counts={role: 1 for role in required_roles},
    )
    asset_ids_by_role = {
        role: asset_id
        for role, asset_id in zip(required_roles, asset_ids)
    }
    slot_ids_by_role = {
        str(slot.get("concept") or slot.get("runtime_role_hint")): str(slot.get("slot_id") or "")
        for slot in intent.get("semantic_slots", [])
        if isinstance(slot, dict)
    }
    group_assignments = []
    for group in intent.get("groups", []):
        anchor_role = group["anchor_role"]
        member_role = group["member_role"]
        anchor_slot_id = slot_ids_by_role.get(anchor_role, "")
        member_slot_id = slot_ids_by_role.get(member_role, "")
        if anchor_slot_id and member_slot_id and anchor_role in asset_ids_by_role and member_role in asset_ids_by_role:
            group_assignments.append(
                {
                    "group_id": group["group_id"],
                    "slot_asset_map": {
                        anchor_slot_id: asset_ids_by_role[anchor_role],
                        member_slot_id: asset_ids_by_role[member_role],
                    },
                }
            )
    return {
        "prompt_mode": "llm",
        "llm_plan": {
            "intent": intent,
            "placement_intent": _strict_placement_intent(adjacency_pairs=adjacency_pairs),
            "selection": {
                "selected_prompt": selected_prompt,
                "stylekit_id": "neutral_daylight",
                "pack_ids": ["core_pack"],
                "group_assignments": group_assignments,
                "slot_asset_map": {
                    slot_ids_by_role[role]: asset_id
                    for role, asset_id in asset_ids_by_role.items()
                    if slot_ids_by_role.get(role)
                },
                "asset_ids": asset_ids,
                "budgets": {"max_props": max_props},
                "optional_additions": optional_additions or [],
                "decor_plan": {"entries": [], "rationale": []},
                "surface_material_selection": approved_surface_material_selection(style_tags=["cozy"], color_tags=["warm"]),
                "alternatives": {
                    slot_ids_by_role[role]: [asset_id]
                    for role, asset_id in asset_ids_by_role.items()
                    if slot_ids_by_role.get(role)
                },
                "rationale": ["semantic planner selected role-matching assets"],
                "confidence": 0.9,
            },
        },
    }


def _candidate(asset_id: str, *, classification: str, semantic_confidence: float):
    return {
        "asset_id": asset_id,
        "label": "chair",
        "tags": ["chair"],
        "classification": classification,
        "quest_compatible": True,
        "semantic_confidence": semantic_confidence,
        "planner_approved": True,
        "planner_excluded": False,
    }


def _build_test_layout(candidate_assets, *, max_props: int = 3):
    placements, _plan = build_layout_from_selected_assets(
        candidate_assets,
        prompt_text="chair",
        seed=1,
        max_props=max_props,
        intent_spec={
            "semantic_slots": _semantic_slots_from_roles(["chair"]),
            "style_tags": [],
            "color_tags": [],
        },
        placement_intent={
            "density_profile": "normal",
            "anchor_preferences": [],
            "adjacency_pairs": [],
            "layout_mood": "cozy",
        },
    )
    return placements


def test_planner_known_prompt_yields_valid_worldspec_with_semantic_selection():
    asset_ids = shortlist_asset_ids("cozy indoor room with chair and table", "chair", "table")
    intent = _strict_intent(
        required_roles=["chair", "table"],
        archetype="study",
        scene_type="study",
        style_tags=["cozy"],
        color_tags=["warm"],
        confidence=0.85,
    )
    result = plan_worldspec(
        "Build a cozy indoor room with chair and table",
        seed=42,
        user_prefs={
            "llm_plan": {
                "intent": intent,
                "selection": {
                    "selected_prompt": "cozy indoor room with chair and table",
                    "stylekit_id": "neutral_daylight",
                    "pack_ids": ["core_pack"],
                    "slot_asset_map": _slot_asset_map_from_roles(intent, {"chair": asset_ids[0], "table": asset_ids[1]}),
                    "asset_ids": asset_ids,
                    "budgets": {"max_props": 2},
                "surface_material_selection": approved_surface_material_selection(style_tags=["cozy"], color_tags=["warm"]),
                    "alternatives": {"chair": [asset_ids[0]], "table": [asset_ids[1]]},
                    "rationale": ["approved indoor props match the requested scene"],
                    "confidence": 0.85,
                },
                "placement_intent": _strict_placement_intent(),
            }
        },
    )
    assert result["ok"] is True
    assert result["errors"] == []
    assert result["planner_backend"] == "llm"
    assert result["semantic_path_status"] == "ok"
    assert result["worldspec"]["placements"]

    validation = validate_worldspec(result["worldspec"])
    assert validation["ok"] is True


def test_shortlist_excludes_unapproved_and_excluded_assets():
    shortlist = build_semantic_candidate_shortlist(
        [
            {
                **_candidate("approved_chair", classification="prop", semantic_confidence=0.8),
                "label": "chair",
                "tags": ["chair"],
            },
            {
                **_candidate("needs_review_chair", classification="prop", semantic_confidence=0.8),
                "planner_approved": False,
                "review_status": "needs_review",
            },
            {
                **_candidate("excluded_chair", classification="prop", semantic_confidence=0.8),
                "planner_excluded": True,
            },
        ],
        "chair",
        limit=10,
    )

    assert [asset["asset_id"] for asset in shortlist] == ["approved_chair"]


def test_shortlist_prefers_prompt_matched_assets_when_no_slots_are_available():
    shortlist = build_semantic_candidate_shortlist(
        [
            {
                **_candidate("bed_approved", classification="prop", semantic_confidence=0.82),
                "label": "cozy bed",
                "tags": ["bed", "bedroom", "cozy"],
                "usable_roles": ["bed"],
                "room_affinities": ["bedroom"],
            },
            {
                **_candidate("nightstand_approved", classification="prop", semantic_confidence=0.80),
                "label": "nightstand",
                "tags": ["nightstand", "bedroom"],
                "usable_roles": ["table", "nightstand"],
                "room_affinities": ["bedroom"],
            },
            {
                **_candidate("garage_door", classification="prop", semantic_confidence=0.99),
                "label": "garage door",
                "tags": ["garage", "door"],
                "room_affinities": ["garage"],
            },
        ],
        "a nice cozy bedroom",
        limit=4,
    )

    assert [asset["asset_id"] for asset in shortlist] == ["bed_approved", "nightstand_approved"]


def test_shortlist_blocks_bathroom_vanities_for_bedrooms():
    shortlist = build_semantic_candidate_shortlist(
        [
            {
                **_candidate("bed_approved", classification="prop", semantic_confidence=0.82),
                "label": "cozy bed",
                "tags": ["bed", "bedroom", "cozy"],
                "usable_roles": ["bed"],
                "room_affinities": ["bedroom"],
            },
            {
                **_candidate("nightstand_approved", classification="prop", semantic_confidence=0.80),
                "label": "nightstand",
                "tags": ["nightstand", "bedroom"],
                "usable_roles": ["table", "nightstand"],
                "room_role_subtype": "nightstand",
                "room_affinities": ["bedroom"],
            },
            {
                **_candidate("bathroom_vanity", classification="prop", semantic_confidence=0.98),
                "label": "table",
                "usable_roles": ["table", "vanity"],
                "room_role_subtype": "vanity",
                "room_affinities": ["bathroom", "lounge"],
                "tags": ["interior", "surface", "table"],
            },
        ],
        "a nice cozy bedroom",
        limit=4,
    )

    assert [asset["asset_id"] for asset in shortlist] == ["bed_approved", "nightstand_approved"]


def test_shortlist_uses_scene_program_features_for_optional_decor():
    shortlist = build_semantic_candidate_shortlist(
        [
            {
                **_candidate("parisian_frame", classification="prop", semantic_confidence=0.9),
                "label": "parisian gallery frame",
                "tags": ["decor", "frame"],
                "style_tags": ["parisian"],
            },
            {
                **_candidate("plain_decor", classification="prop", semantic_confidence=0.9),
                "label": "plain decor",
                "tags": ["decor"],
            },
        ],
        "small parisian gallery corner",
        limit=2,
        scene_program={
            "scene_type": "gallery_corner",
            "archetype": "lounge",
            "semantic_slots": _semantic_slots_from_roles(["chair", "table"], optional_roles=["decor"]),
            "scene_features": ["gallery"],
            "style_cues": {"style_tags": ["parisian"]},
        },
    )

    assert [asset["asset_id"] for asset in shortlist] == ["parisian_frame", "plain_decor"]


def test_shortlist_stays_thin_and_leaves_creative_preference_to_selection_llm():
    shortlist = build_semantic_candidate_shortlist(
        [
            {
                **_candidate("scholar_frame", classification="prop", semantic_confidence=0.8),
                "label": "ornate scholar frame",
                "tags": ["decor", "frame", "scholar"],
                "room_affinities": ["scholar", "reading"],
                "style_tags": ["classic"],
            },
            {
                **_candidate("plain_frame", classification="prop", semantic_confidence=0.95),
                "label": "plain frame",
                "tags": ["decor", "frame"],
                "room_affinities": ["generic"],
            },
        ],
        "moody scholar retreat",
        limit=2,
        scene_program={
            "scene_type": "scholar_retreat",
            "concept_label": "scholar_retreat",
            "creative_tags": ["scholar", "literary"],
            "mood_tags": ["moody"],
            "style_descriptors": ["classic"],
            "intended_use": "quiet reading retreat",
            "semantic_slots": _semantic_slots_from_roles(["chair", "table"], optional_roles=["decor"]),
            "style_cues": {"style_tags": ["classic"]},
        },
    )

    assert [asset["asset_id"] for asset in shortlist] == ["scholar_frame", "plain_frame"]


def test_shortlist_excludes_classical_caps_outside_classical_scenes():
    shortlist = build_semantic_candidate_shortlist(
        [
            {
                **_candidate("greek_cap", classification="prop", semantic_confidence=0.9),
                "label": "greek ionic cap",
                "tags": ["decor", "greek", "classical", "column", "cap"],
                "room_affinities": ["classical", "gallery"],
                "usable_roles": ["decor", "architectural_cap"],
                "negative_scene_affinities": ["study", "reading_nook"],
            },
            {
                **_candidate("plain_plant", classification="prop", semantic_confidence=0.9),
                "label": "plant",
                "tags": ["decor", "plant"],
                "room_affinities": ["generic_room", "lounge"],
            },
        ],
        "cozy reading nook",
        limit=2,
        scene_program={
            "scene_type": "reading_nook",
            "concept_label": "reading_nook",
            "creative_tags": ["reading"],
            "mood_tags": ["cozy"],
            "style_descriptors": ["warm"],
            "semantic_slots": _semantic_slots_from_roles(["chair", "table"], optional_roles=["decor"]),
            "negative_constraints": ["avoid_classical_pieces"],
            "style_cues": {"style_tags": ["cozy"]},
        },
    )

    assert [asset["asset_id"] for asset in shortlist] == ["plain_plant"]


def test_shortlist_covers_art_lighting_and_architectural_slots_by_concept():
    assets = [
        {
            **_candidate("gallery_painting_01", classification="prop", semantic_confidence=0.8),
            "label": "large framed painting",
            "tags": ["decor", "painting", "frame"],
            "usable_roles": ["wall_accent", "focal_art"],
        },
        {
            **_candidate("warm_floor_lamp_01", classification="prop", semantic_confidence=0.8),
            "label": "warm floor lamp",
            "tags": ["lamp", "warm", "lighting"],
            "usable_roles": ["lamp", "floor_lamp"],
            "room_role_subtype": "floor_lamp",
        },
        {
            **_candidate("column_01", classification="prop", semantic_confidence=0.8),
            "label": "stone column",
            "tags": ["decor", "column", "architectural"],
            "usable_roles": ["decor", "architectural_column"],
        },
    ]

    shortlist = build_semantic_candidate_shortlist(
        assets,
        "moody gallery lounge with pillars, warm lighting, and a dramatic painting wall",
        limit=8,
        scene_program={
            "scene_type": "gallery_lounge",
            "archetype": "lounge",
            "semantic_slots": [
                {"slot_id": "focal_art_1", "concept": "focal_art", "priority": "should"},
                {"slot_id": "warm_lighting_1", "concept": "warm_lighting", "priority": "should"},
                {"slot_id": "pillar_1", "concept": "pillar", "priority": "should"},
            ],
        },
    )

    assert [asset["asset_id"] for asset in shortlist] == ["gallery_painting_01", "column_01", "warm_floor_lamp_01"]


def test_layout_reuses_matching_chair_family_for_repeated_seating():
    selected_assets = [
        {
            **_candidate("table_family_a", classification="prop", semantic_confidence=0.95),
            "label": "table",
            "semantic_role_key": "table",
            "collection_id": "set_a",
            "pairing_group": "dining",
            "bounds": {"size": {"x": 1.2, "y": 0.78, "z": 1.2}},
        },
        {
            **_candidate("chair_family_b", classification="prop", semantic_confidence=0.8),
            "label": "chair",
            "semantic_role_key": "chair",
            "coherence_family_id": "chair_b",
            "collection_id": "set_b",
            "pairing_group": "dining",
            "bounds": {"size": {"x": 0.5, "y": 0.95, "z": 0.5}},
        },
    ]
    candidate_assets = selected_assets + [
        {
            **_candidate("chair_family_a", classification="prop", semantic_confidence=0.9),
            "label": "chair",
            "semantic_role_key": "chair",
            "coherence_family_id": "chair_a",
            "collection_id": "set_a",
            "pairing_group": "dining",
            "bounds": {"size": {"x": 0.52, "y": 0.96, "z": 0.52}},
        }
    ]

    placements, _plan = build_layout_from_selected_assets(
        selected_assets,
        prompt_text="dining setup",
        seed=1,
        max_props=4,
        intent_spec={
            "required_roles": ["table", "chair"],
            "optional_roles": [],
            "style_tags": [],
            "color_tags": [],
        },
        placement_intent={
            "density_profile": "normal",
            "anchor_preferences": [],
            "adjacency_pairs": [],
            "layout_mood": "cozy",
        },
        candidate_assets=candidate_assets,
        scene_program=_strict_intent(
            required_roles=["table", "chair"],
            role_counts={"chair": 2, "table": 1},
        ),
    )

    chair_asset_ids = [placement["asset_id"] for placement in placements if placement["asset_id"].startswith("chair_")]
    assert chair_asset_ids == ["chair_family_a", "chair_family_a"]


def test_normalize_scene_program_preserves_generic_role_counts():
    result = validate_semantic_intent(
        {
            "intent": _strict_intent(
                required_roles=["chair", "table"],
                optional_roles=["decor"],
                role_counts={"chair": 2, "table": 1},
                archetype="lounge",
                scene_type="cafe_corner",
                scene_features=["pastries"],
                style_tags=["parisian"],
                confidence=0.8,
            ),
            "placement_intent": _strict_placement_intent(),
        },
        prompt_text="a cafe corner with two chairs and a round table",
    )

    assert result["ok"] is True
    slot_counts = {
        slot["runtime_role_hint"]: int(slot.get("count") or 1)
        for slot in result["scene_program"]["semantic_slots"]
        if slot.get("runtime_role_hint") in {"chair", "table"}
    }
    assert slot_counts == {"chair": 2, "table": 1}
    assert result["scene_program"]["density_target"] == "normal"
    assert result["scene_program"]["symmetry_preference"] == "balanced"
    assert result["scene_program"]["walkway_preservation_intent"]["keep_entry_clear"] is True


def test_planner_unknown_prompt_fails_without_semantic_plan():
    result = plan_worldspec(
        "qwertyuiop asdfghjk",
        user_prefs={},
    )
    _assert_llm_error(result, "llm_unavailable")


def test_planner_never_emits_unknown_ids():
    pack_registry = load_pack_registry()
    style_registry = load_stylekit_registry()
    asset_ids = shortlist_asset_ids("indoor room with chair and table", "chair", "table")
    intent = _strict_intent(
        required_roles=["chair", "table"],
        archetype="study",
        scene_type="study",
        style_tags=["cozy"],
        color_tags=[],
        confidence=0.8,
    )

    result = plan_worldspec(
        "indoor room with chair and table",
        seed=7,
        user_prefs={
            "llm_plan": {
                "intent": intent,
                "selection": {
                    "selected_prompt": "indoor room with chair and table",
                    "stylekit_id": "neutral_daylight",
                    "pack_ids": ["core_pack"],
                    "slot_asset_map": _slot_asset_map_from_roles(intent, {"chair": asset_ids[0], "table": asset_ids[1]}),
                    "asset_ids": asset_ids,
                    "budgets": {"max_props": 2},
                "surface_material_selection": approved_surface_material_selection(style_tags=["cozy"], color_tags=["warm"]),
                },
                "placement_intent": _strict_placement_intent(),
            }
        },
    )
    assert result["ok"] is True
    spec = result["worldspec"]

    for pack_id in spec["pack_ids"]:
        assert pack_id in pack_registry.packs_by_id
    assert spec["stylekit_id"] in style_registry.stylekits_by_id

    known_assets = {str(asset["asset_id"]) for asset in collect_assets([], pack_registry)}
    for placement in spec["placements"]:
        assert placement["asset_id"] in known_assets


def test_planner_is_deterministic_for_same_prompt_and_seed():
    prefs = _inline_llm_plan(
        shortlist_asset_ids("indoor room with chair table", "chair", "table"),
        max_props=2,
        selected_prompt="indoor room with chair table",
    )
    first = plan_worldspec("indoor room with chair table", seed=12345, user_prefs=prefs)
    second = plan_worldspec("indoor room with chair table", seed=12345, user_prefs=prefs)
    assert first["ok"] is True
    assert second["ok"] is True
    assert first["worldspec"] == second["worldspec"]


def test_planner_llm_mode_emits_primary_prompt_plan():
    result = plan_worldspec(
        "cozy indoor room with chair and lamp",
        seed=55,
        user_prefs=_inline_llm_plan(
            shortlist_asset_ids("cozy indoor room with chair and lamp", "chair", "lamp"),
            max_props=2,
            required_roles=["chair", "lamp"],
        ),
    )
    assert result["ok"] is True
    plan = result["prompt_plan"]
    assert plan["mode"] == "llm"
    assert plan["strategy"] == "semantic_primary"
    assert plan["selected_variant_index"] == 0
    assert plan["selected_prompt"]


def test_planner_rejects_non_llm_prompt_mode():
    result = plan_worldspec(
        "minimal studio room",
        seed=12,
        user_prefs={"prompt_mode": "literal"},
    )
    assert result["ok"] is False
    assert result["error_code"] == "invalid_prompt_mode"
    assert any(err["path"] == "$.user_prefs.prompt_mode" for err in result["errors"])


def test_planner_llm_mode_with_inline_plan_uses_llm_backend():
    asset_ids = shortlist_asset_ids("a small room", "chair", "table")
    result = plan_worldspec(
        "a small room",
        seed=33,
        user_prefs=_inline_llm_plan(asset_ids, max_props=2),
    )
    assert result["ok"] is True
    assert result["planner_backend"] == "llm"
    assert result["semantic_path_status"] == "ok"
    assert result["worldspec"]["budgets"]["max_props"] == 2
    assert len(result["worldspec"]["placements"]) >= 1


def test_planner_uses_scene_graph_solver_layout():
    asset_ids = shortlist_asset_ids("indoor room with chair and table", "chair", "table")
    prefs = _inline_llm_plan(asset_ids, max_props=2, selected_prompt="indoor room with chair and table")

    result = plan_worldspec("indoor room with chair and table", seed=21, user_prefs=prefs)

    assert result["ok"] is True
    assert result["worldspec"]["planner_policy"]["placement_mode"] == "scene_graph_solver"
    assert result["placement_plan"]["placement_backend"] == "scene_graph_solver"
    assert result["placement_plan"]["placement_constraints_enabled"] is True
    assert len(result["worldspec"]["placements"]) == 2


def test_planner_ignores_legacy_direct_placement_payload():
    asset_ids = shortlist_asset_ids("indoor room with chair and table", "chair", "table")
    prefs = _inline_llm_plan(asset_ids, max_props=2, selected_prompt="indoor room with chair and table")

    result = plan_worldspec("indoor room with chair and table", seed=21, user_prefs=prefs)

    assert result["ok"] is True
    assert result["worldspec"]["planner_policy"]["placement_mode"] == "scene_graph_solver"


def test_validate_semantic_plan_rejects_missing_required_roles():
    candidate_assets = [
        {
            "asset_id": "core_chair_01",
            "label": "chair",
            "tags": ["chair"],
            "planner_approved": True,
            "planner_excluded": False,
        },
        {
            "asset_id": "core_lamp_01",
            "label": "lamp",
            "tags": ["lamp"],
            "planner_approved": True,
            "planner_excluded": False,
        },
    ]

    intent = _strict_intent(
        required_roles=["chair", "lamp"],
        archetype="study",
        scene_type="study",
        confidence=0.9,
        primary_anchor_role="lamp",
        relation_graph=[
            {"source_role": "lamp", "target_role": "room", "relation": "middle"},
            {"source_role": "chair", "target_role": "lamp", "relation": "near"},
        ],
    )
    result = validate_semantic_plan(
        {
            "intent": intent,
            "selection": {
                "selected_prompt": "small indoor room with chair and lamp",
                "stylekit_id": "neutral_daylight",
                "pack_ids": ["core_pack"],
                "slot_asset_map": _slot_asset_map_from_roles(intent, {"chair": "core_chair_01"}),
                "asset_ids": ["core_chair_01"],
                "budgets": {"max_props": 2},
                "surface_material_selection": approved_surface_material_selection(style_tags=["cozy"], color_tags=["warm"]),
            },
        },
        all_assets=candidate_assets,
        allowed_stylekit_ids=["neutral_daylight"],
        allowed_pack_ids=["core_pack"],
        default_budgets={"max_props": 4, "max_texture_tier": 1, "max_lights": 2},
        prompt_text="small indoor room with chair and lamp",
        placement_intent=_strict_placement_intent(),
        surface_material_candidates=_surface_material_candidates(),
    )

    assert result["ok"] is False
    assert result["error_code"] == "semantic_missing_required_slots"
    assert result["missing_required_slots"] == ["lamp_slot_1"]
    assert result["slot_asset_map"] == {"chair_slot_1": "core_chair_01"}


def test_validate_semantic_plan_softens_missing_aesthetic_must_slots():
    candidate_assets = [
        {
            "asset_id": "core_table_01",
            "label": "display table",
            "tags": ["table", "display"],
            "room_role_subtype": "display_surface",
            "planner_approved": True,
            "planner_excluded": False,
        }
    ]
    result = validate_semantic_plan(
        {
            "intent": {
                **_strict_intent(required_roles=["table"], archetype="lounge", scene_type="gallery"),
                "semantic_slots": [
                    {"slot_id": "display_surface_1", "concept": "display_surface", "priority": "must", "count": 1},
                    {"slot_id": "focal_art_1", "concept": "focal_art", "priority": "must", "count": 1},
                    {"slot_id": "warm_lighting_1", "concept": "warm_lighting", "priority": "must", "count": 1},
                ],
            },
            "selection": {
                "selected_prompt": "moody gallery with warm lighting and a dramatic painting wall",
                "stylekit_id": "neutral_daylight",
                "pack_ids": ["core_pack"],
                "slot_asset_map": {"display_surface_1": "core_table_01"},
                "surface_material_selection": approved_surface_material_selection(style_tags=["gallery"], color_tags=["warm"]),
            },
        },
        all_assets=candidate_assets,
        allowed_stylekit_ids=["neutral_daylight"],
        allowed_pack_ids=["core_pack"],
        default_budgets={"max_props": 4, "max_texture_tier": 1, "max_lights": 2},
        prompt_text="moody gallery with warm lighting and a dramatic painting wall",
        placement_intent=_strict_placement_intent(),
        surface_material_candidates=_surface_material_candidates(),
    )

    assert result["ok"] is True
    assert result["semantic_selection"]["missing_required_slots"] == []
    assert result["semantic_selection"]["softened_required_slots"] == ["focal_art_1", "warm_lighting_1"]
    assert {
        (entry["slot_id"], entry["status"])
        for entry in result["semantic_selection"]["slot_diagnostics"]
    } >= {("focal_art_1", "soft_missing"), ("warm_lighting_1", "soft_missing")}


def test_validate_semantic_plan_uses_slot_metadata_for_support_requiredness():
    candidate_assets = [
        {"asset_id": "core_chair_01", "label": "reading chair", "tags": ["chair"], "planner_approved": True, "planner_excluded": False},
        {"asset_id": "core_table_01", "label": "side table", "tags": ["table"], "planner_approved": True, "planner_excluded": False},
        {"asset_id": "core_lamp_01", "label": "floor lamp", "tags": ["lamp"], "planner_approved": True, "planner_excluded": False},
    ]
    result = validate_semantic_plan(
        {
            "intent": {
                **_strict_intent(required_roles=["chair", "table", "lamp"], archetype="lounge", scene_type="reading_room"),
                "semantic_slots": [
                    {"slot_id": "reading_seat_1", "concept": "chair", "runtime_role_hint": "chair", "priority": "must", "necessity": "core", "source": "explicit_prompt", "count": 1},
                    {"slot_id": "side_surface_1", "concept": "nightstand", "runtime_role_hint": "table", "priority": "should", "necessity": "support", "source": "inferred_function", "count": 1},
                    {"slot_id": "reading_light_1", "concept": "floor_lamp", "runtime_role_hint": "lamp", "priority": "should", "necessity": "support", "source": "inferred_function", "count": 1},
                    {"slot_id": "book_storage_1", "concept": "bookshelf", "runtime_role_hint": "cabinet", "priority": "must", "necessity": "support", "source": "inferred_function", "count": 1},
                ],
                "primary_anchor_object": {"slot_id": "reading_seat_1", "role": "chair", "rationale": "reading anchor"},
                "groups": [],
            },
            "selection": {
                "selected_prompt": "make me a cozy room to read in",
                "stylekit_id": "neutral_daylight",
                "pack_ids": ["core_pack"],
                "slot_asset_map": {
                    "reading_seat_1": "core_chair_01",
                    "side_surface_1": "core_table_01",
                    "reading_light_1": "core_lamp_01",
                },
                "surface_material_selection": approved_surface_material_selection(style_tags=["cozy"], color_tags=["warm"]),
            },
        },
        all_assets=candidate_assets,
        allowed_stylekit_ids=["neutral_daylight"],
        allowed_pack_ids=["core_pack"],
        default_budgets={"max_props": 4, "max_texture_tier": 1, "max_lights": 2},
        prompt_text="make me a cozy room to read in",
        placement_intent=_strict_placement_intent(),
        surface_material_candidates=_surface_material_candidates(),
    )

    assert result["ok"] is True
    assert result["semantic_selection"]["missing_required_slots"] == []
    assert any(
        entry["slot_id"] == "book_storage_1" and entry["status"] == "soft_missing"
        for entry in result["semantic_selection"]["slot_diagnostics"]
    )


def test_validate_semantic_plan_blocks_missing_explicit_core_slot():
    candidate_assets = [
        {"asset_id": "core_chair_01", "label": "chair", "tags": ["chair"], "planner_approved": True, "planner_excluded": False},
    ]
    result = validate_semantic_plan(
        {
            "intent": {
                **_strict_intent(required_roles=["chair", "cabinet"], archetype="lounge", scene_type="library"),
                "semantic_slots": [
                    {"slot_id": "reading_seat_1", "concept": "chair", "runtime_role_hint": "chair", "priority": "must", "necessity": "support", "source": "inferred_function", "count": 1},
                    {"slot_id": "book_storage_1", "concept": "bookshelf", "runtime_role_hint": "cabinet", "priority": "must", "necessity": "core", "source": "explicit_prompt", "count": 1},
                ],
                "primary_anchor_object": {"slot_id": "book_storage_1", "role": "cabinet", "rationale": "library anchor"},
                "groups": [],
            },
            "selection": {
                "selected_prompt": "make me a library with bookshelves",
                "stylekit_id": "neutral_daylight",
                "pack_ids": ["core_pack"],
                "slot_asset_map": {"reading_seat_1": "core_chair_01"},
                "surface_material_selection": approved_surface_material_selection(style_tags=["cozy"], color_tags=["warm"]),
            },
        },
        all_assets=candidate_assets,
        allowed_stylekit_ids=["neutral_daylight"],
        allowed_pack_ids=["core_pack"],
        default_budgets={"max_props": 4, "max_texture_tier": 1, "max_lights": 2},
        prompt_text="make me a library with bookshelves",
        placement_intent=_strict_placement_intent(),
        surface_material_candidates=_surface_material_candidates(),
    )

    assert result["ok"] is False
    assert result["error_code"] == "semantic_missing_required_slots"
    assert result["missing_required_slots"] == ["book_storage_1"]


def test_validate_semantic_plan_normalizes_role_aliases_to_supported_taxonomy():
    candidate_assets = [
        {
            "asset_id": "core_table_01",
            "label": "table",
            "tags": ["table"],
            "planner_approved": True,
            "planner_excluded": False,
        },
        {
            "asset_id": "core_sofa_01",
            "label": "sofa",
            "tags": ["sofa"],
            "planner_approved": True,
            "planner_excluded": False,
        },
    ]

    intent = _strict_intent(
        required_roles=["coffee_table", "couch"],
        archetype="lounge",
        scene_type="lounge",
        confidence=0.9,
        primary_anchor_role="table",
        relation_graph=[
            {"source_role": "table", "target_role": "room", "relation": "middle"},
            {"source_role": "sofa", "target_role": "table", "relation": "near"},
        ],
        secondary_support_objects=[{"role": "sofa", "count": 1, "rationale": "seating"}],
    )
    result = validate_semantic_plan(
        {
            "intent": intent,
            "selection": {
                "selected_prompt": "cozy lounge with couch and coffee table",
                "stylekit_id": "neutral_daylight",
                "pack_ids": ["core_pack"],
                "slot_asset_map": _slot_asset_map_from_roles(intent, {"table": "core_table_01", "sofa": "core_sofa_01"}),
                "asset_ids": ["core_table_01", "core_sofa_01"],
                "budgets": {"max_props": 2},
                "surface_material_selection": approved_surface_material_selection(style_tags=["cozy"], color_tags=["warm"]),
            },
        },
        all_assets=candidate_assets,
        allowed_stylekit_ids=["neutral_daylight"],
        allowed_pack_ids=["core_pack"],
        default_budgets={"max_props": 4, "max_texture_tier": 1, "max_lights": 2},
        prompt_text="cozy lounge with couch and coffee table",
        placement_intent=_strict_placement_intent(),
        surface_material_candidates=_surface_material_candidates(),
    )

    assert result["ok"] is True
    assert [slot["runtime_role"] for slot in result["scene_program"]["grounded_slots"]] == ["table", "sofa"]


def test_validate_semantic_plan_normalizes_compound_role_variants():
    candidate_assets = [
        {
            "asset_id": "core_chair_01",
            "label": "chair",
            "tags": ["chair"],
            "planner_approved": True,
            "planner_excluded": False,
        },
        {
            "asset_id": "core_lamp_01",
            "label": "lamp",
            "tags": ["lamp"],
            "planner_approved": True,
            "planner_excluded": False,
        },
        {
            "asset_id": "core_table_01",
            "label": "table",
            "tags": ["table"],
            "planner_approved": True,
            "planner_excluded": False,
        },
        {
            "asset_id": "core_cabinet_01",
            "label": "cabinet",
            "tags": ["cabinet"],
            "planner_approved": True,
            "planner_excluded": False,
        },
    ]

    intent = _strict_intent(
        required_roles=["reading_chair", "reading_lamp", "study_desk", "display_case", "pedestal"],
        archetype="study",
        scene_type="study",
        confidence=0.9,
        primary_anchor_role="table",
        relation_graph=[
            {"source_role": "table", "target_role": "room", "relation": "middle"},
            {"source_role": "chair", "target_role": "table", "relation": "face_to"},
            {"source_role": "lamp", "target_role": "table", "relation": "near"},
            {"source_role": "cabinet", "target_role": "table", "relation": "near"},
        ],
        secondary_support_objects=[
            {"role": "chair", "count": 1, "rationale": "seating"},
            {"role": "lamp", "count": 1, "rationale": "task light"},
            {"role": "cabinet", "count": 2, "rationale": "display and pedestal"},
        ],
    )
    result = validate_semantic_plan(
        {
            "intent": intent,
            "selection": {
                "selected_prompt": "reading corner with a lamp, desk, pedestal, and display case",
                "stylekit_id": "neutral_daylight",
                "pack_ids": ["core_pack"],
                "slot_asset_map": _slot_asset_map_from_roles(intent, {
                    "chair": "core_chair_01",
                    "lamp": "core_lamp_01",
                    "table": "core_table_01",
                    "cabinet": "core_cabinet_01",
                }),
                "asset_ids": ["core_chair_01", "core_lamp_01", "core_table_01", "core_cabinet_01"],
                "budgets": {"max_props": 4},
                "surface_material_selection": approved_surface_material_selection(style_tags=["cozy"], color_tags=["warm"]),
            },
        },
        all_assets=candidate_assets,
        allowed_stylekit_ids=["neutral_daylight"],
        allowed_pack_ids=["core_pack"],
        default_budgets={"max_props": 4, "max_texture_tier": 1, "max_lights": 2},
        prompt_text="reading corner with a lamp, desk, pedestal, and display case",
        placement_intent=_strict_placement_intent(),
        surface_material_candidates=_surface_material_candidates(),
    )

    assert result["ok"] is True
    assert [slot["runtime_role"] for slot in result["scene_program"]["grounded_slots"]] == ["chair", "lamp", "table", "cabinet", "table"]


def test_validate_semantic_plan_requires_model_chosen_stylekit():
    candidate_assets = [
        {
            "asset_id": "core_table_01",
            "label": "table",
            "tags": ["table"],
            "planner_approved": True,
            "planner_excluded": False,
        }
    ]

    intent = _strict_intent(required_roles=["table"], archetype="study", scene_type="study")
    result = validate_semantic_plan(
        {
            "intent": intent,
            "selection": {
                "selected_prompt": "desk setup",
                "stylekit_id": "unknown_stylekit",
                "pack_ids": ["core_pack"],
                "slot_asset_map": _slot_asset_map_from_roles(intent, {"table": "core_table_01"}),
                "asset_ids": ["core_table_01"],
                "budgets": {"max_props": 1},
                "surface_material_selection": approved_surface_material_selection(style_tags=["cozy"], color_tags=["warm"]),
            },
        },
        all_assets=candidate_assets,
        allowed_stylekit_ids=["neutral_daylight"],
        allowed_pack_ids=["core_pack"],
        default_budgets={"max_props": 4, "max_texture_tier": 1, "max_lights": 2},
        prompt_text="desk setup",
        placement_intent=_strict_placement_intent(),
        surface_material_candidates=_surface_material_candidates(),
    )

    assert result["ok"] is False
    assert result["error_code"] == "semantic_invalid_selection"
    assert any(error["path"] == "$.llm.selection.stylekit_id" for error in result["errors"])


def test_validate_semantic_plan_preserves_model_authored_decor_plan():
    candidate_assets = [
        {
            "asset_id": "core_table_01",
            "label": "table",
            "tags": ["table"],
            "planner_approved": True,
            "planner_excluded": False,
        },
        {
            "asset_id": "frame_01",
            "label": "decor",
            "tags": ["decor"],
            "planner_approved": True,
            "planner_excluded": False,
            "allowed_anchors": ["wall"],
            "placement_modes": ["wall_hung"],
            "usable_roles": ["wall_accent", "focal_art"],
        },
    ]

    intent = _strict_intent(required_roles=["table"], archetype="study", scene_type="study")
    result = validate_semantic_plan(
        {
            "intent": intent,
            "selection": {
                "selected_prompt": "desk setup",
                "stylekit_id": "neutral_daylight",
                "pack_ids": ["core_pack"],
                "slot_asset_map": _slot_asset_map_from_roles(intent, {"table": "core_table_01"}),
                "asset_ids": ["core_table_01"],
                "budgets": {"max_props": 1},
                "rejected_candidate_ids": ["frame_01"],
                "fallback_asset_ids_by_slot": {"table_slot_1": ["core_table_01"]},
                "surface_material_selection": approved_surface_material_selection(style_tags=["cozy"], color_tags=["warm"]),
                "decor_plan": {
                    "entries": [{"asset_id": "frame_01", "kind": "frame", "anchor": "wall", "zone_id": "focus_wall", "count": 1, "placement_hint": "wall_centered"}],
                    "rationale": ["accent the focal wall"],
                },
            },
        },
        all_assets=candidate_assets,
        allowed_stylekit_ids=["neutral_daylight"],
        allowed_pack_ids=["core_pack"],
        default_budgets={"max_props": 4, "max_texture_tier": 1, "max_lights": 2},
        prompt_text="desk setup",
        placement_intent=_strict_placement_intent(),
        surface_material_candidates=_surface_material_candidates(),
    )

    assert result["ok"] is True
    assert result["decor_plan"]["entries"] == [
        {"asset_id": "frame_01", "kind": "frame", "anchor": "wall", "zone_id": "focus_wall", "count": 1, "placement_hint": "wall_centered"}
    ]
    assert result["semantic_selection"]["rejected_candidate_ids"] == ["frame_01"]
    assert result["semantic_selection"]["fallback_asset_ids_by_slot"] == {"table_slot_1": ["core_table_01"]}


def test_validate_semantic_intent_preserves_typed_relation_graph_fields():
    result = validate_semantic_intent(
        {
            "intent": {
                **_strict_intent(
                    required_roles=["chair", "table"],
                    optional_roles=["lamp"],
                    archetype="study",
                    scene_type="study",
                ),
                "relation_graph": [
                    {
                        "source_role": "chair",
                        "target_role": "table",
                        "relation": "face_to",
                        "relation_type": "orientation",
                        "constraint_strength": "required",
                    },
                    {
                        "source_role": "lamp",
                        "target_role": "table",
                        "relation": "support_on",
                        "relation_type": "support",
                        "constraint_strength": "preferred",
                        "target_surface_type": "tabletop",
                    },
                    {
                        "source_role": "table",
                        "target_role": "room",
                        "relation": "middle",
                        "relation_type": "room_position",
                        "constraint_strength": "required",
                    },
                ],
            },
            "placement_intent": _strict_placement_intent(),
        },
        prompt_text="reading desk with chair and lamp",
    )

    assert result["ok"] is True
    assert result["scene_program"]["relation_graph"][0]["relation_type"] == "orientation"
    assert result["scene_program"]["relation_graph"][0]["constraint_strength"] == "required"
    assert result["scene_program"]["relation_graph"][1]["target_surface_type"] == "tabletop"


def test_validate_semantic_plan_ignores_invalid_optional_accent_material():
    candidate_assets = [
        {
            "asset_id": "core_table_01",
            "label": "table",
            "tags": ["table"],
            "planner_approved": True,
            "planner_excluded": False,
        }
    ]

    intent = _strict_intent(required_roles=["table"], archetype="study", scene_type="study")
    result = validate_semantic_plan(
        {
            "intent": intent,
            "selection": {
                "selected_prompt": "desk setup",
                "stylekit_id": "neutral_daylight",
                "pack_ids": ["core_pack"],
                "slot_asset_map": _slot_asset_map_from_roles(intent, {"table": "core_table_01"}),
                "asset_ids": ["core_table_01"],
                "budgets": {"max_props": 1},
                "surface_material_selection": {
                    **approved_surface_material_selection(style_tags=["cozy"], color_tags=["warm"]),
                    "accent": "invented_accent_material",
                },
            },
        },
        all_assets=candidate_assets,
        allowed_stylekit_ids=["neutral_daylight"],
        allowed_pack_ids=["core_pack"],
        default_budgets={"max_props": 4, "max_texture_tier": 1, "max_lights": 2},
        prompt_text="desk setup",
        placement_intent=_strict_placement_intent(),
        surface_material_candidates=_surface_material_candidates(),
    )

    assert result["ok"] is True
    assert "accent" not in result["surface_material_selection"]


def test_planner_default_policy_does_not_silently_fallback_when_llm_not_configured():
    result = plan_worldspec(
        "small room with lamp",
        seed=33,
        user_prefs={},
    )
    _assert_llm_error(result, "llm_unavailable")


def test_planner_unknown_asset_ids_do_not_trigger_fallback_without_explicit_opt_in():
    result = plan_worldspec(
        "small room",
        seed=19,
        user_prefs=_inline_llm_plan(
            ["nonexistent_asset_999"],
            max_props=2,
            selected_prompt="small room with unknown object",
        ),
    )
    assert result["ok"] is False
    assert result["error_code"] == "semantic_unknown_assets"
    assert result["semantic_path_status"] == "failed"


def test_planner_unknown_asset_ids_fail_without_fallback_path():
    result = plan_worldspec(
        "small room",
        seed=19,
        user_prefs=_inline_llm_plan(
            ["nonexistent_asset_999"],
            max_props=2,
            selected_prompt="small room with unknown object",
        ),
    )
    assert result["ok"] is False
    assert result["error_code"] == "semantic_unknown_assets"
    assert result["semantic_path_status"] == "failed"


def test_planner_parse_errors_fail_fast_without_scene_program_fallback(monkeypatch):
    def fake_llm(*args, **kwargs):
        return {"ok": False, "error_code": "llm_parse_error", "message": "invalid JSON"}

    monkeypatch.setattr("src.planning.planner.request_llm_intent", fake_llm)
    result = plan_worldspec("room prompt", seed=9, user_prefs={})
    assert result["ok"] is False
    assert result["error_code"] == "llm_parse_error"
    assert result["semantic_path_status"] == "failed"


def test_planner_transport_failure_still_fails_without_fallback_path(monkeypatch):
    def fake_llm(*args, **kwargs):
        return {"ok": False, "error_code": "llm_transport_error", "message": "timeout"}

    monkeypatch.setattr("src.planning.planner.request_llm_intent", fake_llm)
    result = plan_worldspec("simple room with chair", seed=9, user_prefs={})
    _assert_llm_error(result, "llm_transport_error")


def test_planner_semantic_receipts_present_by_default():
    result = plan_worldspec(
        "cozy room with chair and lamp",
        seed=21,
        user_prefs=_inline_llm_plan(
            shortlist_asset_ids("cozy room with chair and lamp", "chair", "lamp"),
            max_props=2,
            required_roles=["chair", "lamp"],
        ),
    )
    assert result["ok"] is True
    assert "semantic_receipts" in result
    receipts = result["semantic_receipts"]
    assert isinstance(receipts, dict)
    assert set(receipts.keys()) == {"scene_program", "intent_spec", "placement_intent", "selected_slots", "slot_diagnostics", "alternatives", "rationale", "confidence", "selection"}
    assert isinstance(receipts["selected_slots"], dict)
    assert isinstance(receipts["slot_diagnostics"], list)
    assert isinstance(receipts["scene_program"], dict)
    assert isinstance(receipts["placement_intent"], dict)
    assert isinstance(receipts["alternatives"], dict)
    assert isinstance(receipts["rationale"], list)
    assert isinstance(receipts["confidence"], float)
    assert "required_roles" not in receipts["scene_program"]
    assert "required_roles" not in receipts["intent_spec"]


def test_plan_worldspec_public_nested_exports_drop_legacy_role_fields():
    result = plan_worldspec(
        "cozy room with chair and lamp",
        seed=21,
        user_prefs=_inline_llm_plan(
            shortlist_asset_ids("cozy room with chair and lamp", "chair", "lamp"),
            max_props=2,
            required_roles=["chair", "lamp"],
        ),
    )
    assert result["ok"] is True
    assert "required_roles" not in result["scene_program"]
    assert "optional_roles" not in result["scene_program"]
    assert "role_counts" not in result["scene_program"]
    assert "required_roles" not in result["intent_spec"]
    assert "optional_roles" not in result["intent_spec"]
    assert "role_counts" not in result["intent_spec"]


def test_planner_messy_prompt_yields_cluttered_placement_intent():
    asset_ids = shortlist_asset_ids("reading room clutter", "chair", "table")
    intent = _strict_intent(
        required_roles=["chair", "table"],
        archetype="study",
        scene_type="study",
        style_tags=["cozy", "reading"],
        color_tags=["warm"],
        density_target="cluttered",
        confidence=0.92,
    )
    result = plan_worldspec(
        "take me to a messy reading room with chair table",
        seed=99,
        user_prefs={
            "llm_plan": {
                "intent": intent,
                "placement_intent": _strict_placement_intent(
                    density_profile="cluttered",
                    adjacency_pairs=[
                        {"source_role": "chair", "target_role": "lamp", "relation": "near"},
                        {"source_role": "chair", "target_role": "table", "relation": "near"},
                    ],
                    layout_mood="crowded",
                ),
                "selection": {
                    "selected_prompt": "messy reading room",
                    "stylekit_id": "neutral_daylight",
                    "pack_ids": ["core_pack"],
                    "slot_asset_map": _slot_asset_map_from_roles(intent, dict(zip(["chair", "table"], asset_ids))),
                    "asset_ids": asset_ids,
                    "budgets": {"max_props": 4},
                "surface_material_selection": approved_surface_material_selection(style_tags=["cozy"], color_tags=["warm"]),
                    "alternatives": {},
                    "rationale": ["cluttered reading room needs denser props"],
                    "confidence": 0.92,
                },
            }
        },
    )
    assert result["ok"] is True
    assert result["semantic_receipts"]["placement_intent"]["density_profile"] == "cluttered"
    assert result["worldspec"]["placement_intent"]["density_profile"] == "cluttered"


def test_planner_semantic_receipts_omitted_when_disabled():
    prefs = _inline_llm_plan(
        shortlist_asset_ids("cozy room with chair and lamp", "chair", "lamp"),
        max_props=2,
        required_roles=["chair", "lamp"],
    )
    prefs["include_semantic_receipts"] = False
    result = plan_worldspec(
        "cozy room with chair and lamp",
        seed=21,
        user_prefs=prefs,
    )
    assert result["ok"] is True
    assert "semantic_receipts" not in result


def test_planner_prefers_high_confidence_quest_safe_assets_when_rich_metadata_present():
    candidate_assets = [
        _candidate("unsafe_shell", classification="shell", semantic_confidence=0.99),
        _candidate("unsafe_conf", classification="prop", semantic_confidence=0.40),
        _candidate("safe_chair", classification="prop", semantic_confidence=0.95),
    ]

    placements = _build_test_layout(candidate_assets, max_props=3)
    assert [p["asset_id"] for p in placements] == ["safe_chair"]
    assert placements[0]["constraint"]["type"] in {"floor", "against_wall", "near"}


def test_planner_rejects_all_unsafe_assets_when_rich_metadata_present():
    candidate_assets = [
        _candidate("unsafe_shell", classification="shell", semantic_confidence=0.99),
        _candidate("unsafe_conf", classification="prop", semantic_confidence=0.20),
    ]

    placements = _build_test_layout(candidate_assets, max_props=3)
    assert placements == []


def test_plan_worldspec_errors_when_only_unsafe_assets_available(monkeypatch):
    unsafe_registry = PackRegistry(
        packs_by_id={
            "core_pack": {
                "pack_id": "core_pack",
                "tags": ["indoor"],
                "assets": [
                    {
                        "asset_id": "unsafe_shell",
                        "label": "chair",
                        "tags": ["chair"],
                        "classification": "shell",
                        "quest_compatible": True,
                        "semantic_confidence": 0.95,
                    }
                ],
            }
        },
        assets_by_id={"unsafe_shell": {"pack_id": "core_pack", "asset": {"asset_id": "unsafe_shell"}}},
        tags_index={"core_pack": ["indoor"]},
        errors=[],
    )
    style_registry = StyleKitRegistry(
        stylekits_by_id={
            "neutral_daylight": {
                "stylekit_id": "neutral_daylight",
                "tags": ["neutral", "indoor", "day"],
                "lighting": {"preset": "daylight_soft", "intensity": 1.0},
                "palette": {"wall": "#d8d8d8", "floor": "#8b7d6b", "accent": "#4a90e2"},
            }
        },
        tags_index={"neutral_daylight": ["neutral", "indoor", "day"]},
        errors=[],
    )

    monkeypatch.setattr("src.planning.planner.load_pack_registry", lambda: unsafe_registry)
    monkeypatch.setattr("src.planning.planner.load_stylekit_registry", lambda: style_registry)
    monkeypatch.setattr("src.planning.assets.load_planner_pool", lambda: [])

    result = plan_worldspec("indoor chair prompt", seed=12)
    assert result["ok"] is False
    assert result["error_code"] == "llm_unavailable"


def test_planner_emits_constraint_hints_for_room_layouts():
    asset_ids = shortlist_asset_ids("cozy room with chair table lamp", "chair", "table", "lamp")
    result = plan_worldspec(
        "cozy room with chair table lamp",
        seed=44,
        user_prefs=_inline_llm_plan(
            asset_ids,
            max_props=3,
            selected_prompt="cozy room with chair table lamp",
        ),
    )
    assert result["ok"] is True
    constraints = {
        placement["asset_id"]: placement.get("constraint", {}).get("type")
        for placement in result["worldspec"]["placements"]
    }
    chair_id, table_id = asset_ids[0], asset_ids[1]
    assert constraints[chair_id] == "against_wall"
    assert constraints[table_id] == "near"


def test_build_layout_does_not_add_archetype_filler_assets_without_optional_roles():
    chair = {
        "asset_id": "chair_a",
        "label": "chair",
        "tags": ["chair"],
        "classification": "prop",
        "quest_compatible": True,
        "semantic_confidence": 0.9,
        "planner_approved": True,
        "planner_excluded": False,
    }
    table = {
        "asset_id": "table_a",
        "label": "desk",
        "tags": ["desk", "table"],
        "classification": "prop",
        "quest_compatible": True,
        "semantic_confidence": 0.9,
        "planner_approved": True,
        "planner_excluded": False,
    }
    cabinet = {
        "asset_id": "cabinet_a",
        "label": "bookshelf cabinet",
        "tags": ["cabinet", "storage"],
        "classification": "prop",
        "quest_compatible": True,
        "semantic_confidence": 0.9,
        "planner_approved": True,
        "planner_excluded": False,
    }
    decor = {
        "asset_id": "decor_a",
        "label": "wall decor",
        "tags": ["decor", "frame"],
        "classification": "prop",
        "quest_compatible": True,
        "semantic_confidence": 0.9,
        "planner_approved": True,
        "planner_excluded": False,
    }

    placements, plan = build_layout_from_selected_assets(
        [chair, table],
        candidate_assets=[chair, table, cabinet, decor],
        prompt_text="i want a tiny study with a desk and chair",
        seed=7,
        max_props=4,
        intent_spec={
            "scene_type": "study",
            "semantic_slots": _semantic_slots_from_roles(["chair", "table"]),
            "style_tags": [],
            "color_tags": [],
        },
        placement_intent={
            "density_profile": "normal",
            "anchor_preferences": [],
            "adjacency_pairs": [],
            "layout_mood": "cozy",
        },
    )

    placed_ids = {placement["asset_id"] for placement in placements}
    assert placed_ids == {"chair_a", "table_a"}
    assert plan["target_count"] == 2


def test_build_layout_respects_prompt_count_hints_for_required_roles():
    chair_a = {
        "asset_id": "chair_a",
        "label": "chair",
        "tags": ["chair"],
        "classification": "prop",
        "quest_compatible": True,
        "semantic_confidence": 0.9,
        "planner_approved": True,
        "planner_excluded": False,
    }
    chair_b = {
        "asset_id": "chair_b",
        "label": "chair",
        "tags": ["chair"],
        "classification": "prop",
        "quest_compatible": True,
        "semantic_confidence": 0.9,
        "planner_approved": True,
        "planner_excluded": False,
    }
    table = {
        "asset_id": "table_a",
        "label": "table",
        "tags": ["table"],
        "classification": "prop",
        "quest_compatible": True,
        "semantic_confidence": 0.9,
        "planner_approved": True,
        "planner_excluded": False,
    }

    placements, _plan = build_layout_from_selected_assets(
        [chair_a, table],
        candidate_assets=[chair_a, chair_b, table],
        prompt_text="a cafe corner with two chairs and a round table",
        seed=9,
        max_props=3,
        intent_spec={
            "scene_type": "lounge",
            "semantic_slots": _semantic_slots_from_roles(["chair", "table"], role_counts={"chair": 2}),
            "style_tags": [],
            "color_tags": [],
        },
        placement_intent={
            "density_profile": "normal",
            "anchor_preferences": [],
            "adjacency_pairs": [],
            "layout_mood": "cozy",
        },
    )

    chair_count = sum(1 for placement in placements if placement["asset_id"].startswith("chair_"))
    assert chair_count == 2


def test_apply_stylekit_colors_keeps_required_shell_slots_when_palette_omits_ceiling():
    colors = apply_stylekit_colors("neutral_daylight", load_stylekit_registry())

    assert colors["wall"].startswith("#")
    assert colors["ceiling"].startswith("#")


def test_surface_material_candidates_use_creative_scene_fields(tmp_path):
    pool_path = tmp_path / "style_material_pool.json"
    pool_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "material_id": "plain_wall",
                        "review_status": "approved",
                        "display_name": "Plain Wall",
                        "surface_roles": ["wall"],
                        "style_tags": ["cozy"],
                        "color_tags": ["warm"],
                        "tone_tags": ["muted"],
                        "material_family_tags": ["painted_surface"],
                        "texture_tags": ["low_variation"],
                    },
                    {
                        "material_id": "wood_panel_wall",
                        "review_status": "approved",
                        "display_name": "Scholar Wood Panel",
                        "surface_roles": ["wall"],
                        "style_tags": ["cozy", "classic"],
                        "color_tags": ["warm"],
                        "tone_tags": ["muted"],
                        "material_family_tags": ["wood"],
                        "texture_tags": ["panelled"],
                        "visual_description": "dark wood scholar wall",
                        "preview_texture_description": "classic wood paneling",
                    },
                ]
            }
        )
    )
    candidates = build_surface_material_candidates(
        {
            "scene_type": "scholar_retreat",
            "concept_label": "scholar_retreat",
            "creative_tags": ["scholar", "wood"],
            "mood_tags": ["warm"],
            "style_descriptors": ["classic"],
            "style_tags": ["cozy"],
            "color_tags": ["warm"],
            "style_cues": {"style_tags": ["cozy"], "color_tags": ["warm"], "mood_tags": ["warm"]},
        },
        pool_path=pool_path,
    )

    assert candidates["wall"][0]["material_id"] == "wood_panel_wall"


def test_surface_material_candidates_avoid_white_wall_washout_by_default(tmp_path):
    pool_path = tmp_path / "style_material_pool.json"
    pool_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "material_id": "bright_white_wall",
                        "review_status": "approved",
                        "display_name": "Bright White Wall",
                        "surface_roles": ["wall"],
                        "style_tags": ["cozy"],
                        "color_tags": ["white"],
                        "tone_tags": ["neutral"],
                        "material_family_tags": ["painted_surface"],
                        "texture_tags": ["low_variation", "mostly_white"],
                        "preview_color_rgba": {"r": 1.0, "g": 1.0, "b": 1.0},
                    },
                    {
                        "material_id": "warm_plaster_wall",
                        "review_status": "approved",
                        "display_name": "Warm Plaster Wall",
                        "surface_roles": ["wall"],
                        "style_tags": ["cozy"],
                        "color_tags": ["warm", "beige"],
                        "tone_tags": ["muted"],
                        "material_family_tags": ["plaster"],
                        "texture_tags": ["low_variation"],
                        "preview_color_rgba": {"r": 0.85, "g": 0.79, "b": 0.71},
                    },
                ]
            }
        )
    )
    candidates = build_surface_material_candidates(
        {
            "scene_type": "bedroom",
            "concept_label": "bedroom",
            "creative_tags": ["calm"],
            "mood_tags": ["cozy"],
            "style_descriptors": ["warm"],
            "style_tags": ["cozy"],
            "color_tags": ["warm"],
            "style_cues": {"style_tags": ["cozy"], "color_tags": ["warm"]},
        },
        pool_path=pool_path,
    )

    assert candidates["wall"][0]["material_id"] == "warm_plaster_wall"


def test_surface_material_candidates_allow_white_walls_for_gallery_context(tmp_path):
    pool_path = tmp_path / "style_material_pool.json"
    pool_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "material_id": "bright_white_wall",
                        "review_status": "approved",
                        "display_name": "Bright White Wall",
                        "surface_roles": ["wall"],
                        "style_tags": ["minimal"],
                        "color_tags": ["white"],
                        "tone_tags": ["neutral"],
                        "material_family_tags": ["painted_surface"],
                        "texture_tags": ["low_variation", "mostly_white"],
                        "preview_color_rgba": {"r": 1.0, "g": 1.0, "b": 1.0},
                    },
                    {
                        "material_id": "warm_plaster_wall",
                        "review_status": "approved",
                        "display_name": "Warm Plaster Wall",
                        "surface_roles": ["wall"],
                        "style_tags": ["minimal"],
                        "color_tags": ["warm"],
                        "tone_tags": ["muted"],
                        "material_family_tags": ["plaster"],
                        "texture_tags": ["low_variation"],
                        "preview_color_rgba": {"r": 0.85, "g": 0.79, "b": 0.71},
                    },
                ]
            }
        )
    )
    candidates = build_surface_material_candidates(
        {
            "scene_type": "museum_room",
            "concept_label": "museum_gallery",
            "creative_tags": ["gallery"],
            "mood_tags": ["minimal"],
            "style_descriptors": ["bright"],
            "style_tags": ["minimal"],
            "color_tags": ["white"],
            "style_cues": {"style_tags": ["minimal"], "color_tags": ["white"]},
        },
        pool_path=pool_path,
    )

    assert candidates["wall"][0]["material_id"] == "bright_white_wall"


def test_surface_material_candidates_prefer_exact_shell_match_over_broad_accent_material(tmp_path):
    pool_path = tmp_path / "style_material_pool.json"
    pool_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "material_id": "broad_accent_wall",
                        "review_status": "approved",
                        "display_name": "Broad Accent Material",
                        "inferred_label": "accent",
                        "surface_roles": ["accent", "wall", "ceiling", "floor"],
                        "style_tags": ["cozy"],
                        "color_tags": ["warm"],
                        "tone_tags": ["muted"],
                        "material_family_tags": ["accent_material"],
                        "texture_tags": ["low_variation"],
                    },
                    {
                        "material_id": "exact_wall_plaster",
                        "review_status": "approved",
                        "display_name": "Exact Wall Plaster",
                        "inferred_label": "wall",
                        "surface_roles": ["wall"],
                        "style_tags": ["cozy"],
                        "color_tags": ["warm"],
                        "tone_tags": ["muted"],
                        "material_family_tags": ["plaster"],
                        "texture_tags": ["low_variation"],
                    },
                ]
            }
        )
    )

    candidates = build_surface_material_candidates(
        {
            "scene_type": "bedroom",
            "concept_label": "bedroom",
            "style_tags": ["cozy"],
            "color_tags": ["warm"],
            "style_cues": {"style_tags": ["cozy"], "color_tags": ["warm"]},
        },
        pool_path=pool_path,
    )

    assert candidates["wall"][0]["material_id"] == "exact_wall_plaster"


def test_surface_material_candidates_default_limit_is_broader_than_eight(tmp_path):
    records = []
    for index in range(20):
        records.append(
            {
                "material_id": f"wall_{index:02d}",
                "review_status": "approved",
                "display_name": f"Wall {index:02d}",
                "inferred_label": "wall",
                "surface_roles": ["wall"],
                "style_tags": ["cozy"],
                "color_tags": ["warm"],
                "tone_tags": ["muted"],
                "material_family_tags": ["plaster"],
                "texture_tags": ["low_variation"],
            }
        )
    pool_path = tmp_path / "style_material_pool.json"
    pool_path.write_text(json.dumps({"records": records}))

    candidates = build_surface_material_candidates(
        {
            "scene_type": "bedroom",
            "concept_label": "bedroom",
            "style_tags": ["cozy"],
            "color_tags": ["warm"],
            "style_cues": {"style_tags": ["cozy"], "color_tags": ["warm"]},
        },
        pool_path=pool_path,
    )

    assert len(candidates["wall"]) == 16


def test_validate_semantic_intent_recovers_supported_roles_from_reading_room_prompt():
    result = validate_semantic_intent(
        {
            "intent": {
                "scene_type": "reading_room",
                "required_roles": ["reader", "fireplace", "bookshelf", "window_with_rain_outside"],
                "optional_roles": ["reading"],
                "style_tags": ["cozy"],
                "color_tags": [],
                "confidence": 0.8,
            },
            "placement_intent": _strict_placement_intent(),
        },
        prompt_text="Take me to a cozy reading room with a fireplace, books, and rain outside the windows.",
    )

    assert result["ok"] is False
    assert result["error_code"] == "semantic_invalid_intent"
    assert any(error["path"] == "$.llm.intent.execution_archetype" for error in result["errors"])


def test_validate_semantic_intent_falls_back_for_empty_workshop_roles():
    result = validate_semantic_intent(
        {
            "intent": {
                "scene_type": "inventor_workshop",
                "archetype": "workshop",
                "required_roles": [],
                "optional_roles": [],
                "style_tags": [],
                "color_tags": [],
                "confidence": 0.8,
            },
            "placement_intent": _strict_placement_intent(),
        },
        prompt_text="Build me a cluttered inventor workshop with tools, shelves, and a sturdy workbench.",
    )

    assert result["ok"] is False
    assert result["error_code"] == "semantic_invalid_intent"
    assert any(error["path"] == "$.llm.intent.semantic_slots" for error in result["errors"])


def test_validate_semantic_intent_enforces_archetype_minimum_required_roles():
    result = validate_semantic_intent(
        {
            "intent": _strict_intent(
                required_roles=["table", "lamp"],
                archetype="bedroom",
                scene_type="bedroom",
                confidence=0.8,
            ),
            "placement_intent": _strict_placement_intent(),
        },
        prompt_text="Make me a sleek cyberpunk bedroom with neon lights and a desk for hacking.",
    )

    assert result["ok"] is True
    assert {slot["concept"] for slot in result["scene_program"]["semantic_slots"]} == {"table", "lamp"}
    assert all(slot.get("concept") != "bed" for slot in result["scene_program"]["semantic_slots"])


def test_validate_semantic_intent_allows_repeated_support_roles_without_group_failure():
    result = validate_semantic_intent(
        {
            "intent": {
                "scene_type": "cozy bedroom retreat",
                "execution_archetype": "bedroom",
                "semantic_slots": [
                    {"slot_id": "sleep_anchor_1", "concept": "bed", "runtime_role_hint": "bed", "priority": "must", "count": 1},
                    {"slot_id": "bedside_surface_1", "concept": "nightstand", "runtime_role_hint": "table", "priority": "should", "count": 1},
                    {"slot_id": "bedside_lighting_1", "concept": "table lamp", "runtime_role_hint": "lamp", "priority": "should", "count": 1},
                    {"slot_id": "sleep_storage_1", "concept": "dresser", "runtime_role_hint": "cabinet", "priority": "should", "count": 1},
                    {"slot_id": "wardrobe_storage_1", "concept": "wardrobe", "runtime_role_hint": "cabinet", "priority": "optional", "count": 1},
                    {"slot_id": "floor_lighting_1", "concept": "floor_lamp", "runtime_role_hint": "lamp", "priority": "optional", "count": 1},
                ],
                "primary_anchor_object": {"role": "bed"},
                "relation_graph": [
                    {"source_role": "bed", "target_role": "table", "relation": "near"},
                    {"source_role": "cabinet", "target_role": "wall", "relation": "against_wall"},
                ],
                "groups": [
                    {
                        "group_id": "sleep_zone_1",
                        "group_type": "bedside_cluster",
                        "anchor_role": "bed",
                        "member_role": "table",
                        "member_count": 1,
                        "layout_pattern": "beside_anchor",
                        "facing_rule": "parallel",
                        "symmetry": "balanced",
                        "zone_preference": "back",
                        "importance": "primary",
                    }
                ],
            },
            "placement_intent": _strict_placement_intent(),
        },
        prompt_text="a nice cozy bedroom",
    )

    assert result["ok"] is True
    assert "repeated_roles_without_groups:cabinet,lamp" in result["warnings"]


def test_validate_semantic_intent_allows_multi_object_scene_without_groups():
    result = validate_semantic_intent(
        {
            "intent": {
                "scene_type": "bedroom",
                "execution_archetype": "bedroom",
                "semantic_slots": [
                    {"slot_id": "sleep_anchor_1", "concept": "bed", "runtime_role_hint": "bed", "priority": "must", "count": 1},
                    {"slot_id": "storage_1", "concept": "dresser", "runtime_role_hint": "cabinet", "priority": "should", "count": 1},
                ],
                "primary_anchor_object": {"role": "bed"},
                "relation_graph": [{"source_role": "cabinet", "target_role": "bed", "relation": "far"}],
                "groups": [],
            },
            "placement_intent": _strict_placement_intent(),
        },
        prompt_text="bedroom",
    )

    assert result["ok"] is True
    assert "multi_object_scene_missing_groups" in result["warnings"]


def test_validate_semantic_intent_preserves_creative_scene_label_and_execution_archetype():
    result = validate_semantic_intent(
        {
            "intent": _strict_intent(
                required_roles=["workbench", "tool_storage", "task_lighting"],
                archetype="workshop",
                scene_type="interior_workshop",
                confidence=0.8,
                primary_anchor_role="table",
                relation_graph=[
                    {"source_role": "table", "target_role": "room", "relation": "middle"},
                    {"source_role": "tool", "target_role": "table", "relation": "near"},
                    {"source_role": "cabinet", "target_role": "table", "relation": "edge"},
                    {"source_role": "lamp", "target_role": "table", "relation": "near"},
                ],
                secondary_support_objects=[
                    {"role": "tool", "count": 1, "rationale": "tools"},
                    {"role": "cabinet", "count": 1, "rationale": "storage"},
                    {"role": "lamp", "count": 1, "rationale": "task lighting"},
                ],
            ),
            "placement_intent": _strict_placement_intent(),
        },
        prompt_text="Create a cluttered inventor workshop with a sturdy workbench, storage, and tools.",
    )

    assert result["ok"] is True
    assert result["scene_program"]["scene_type"] == "interior_workshop"
    assert result["scene_program"]["execution_archetype"] == "workshop"
    assert result["scene_program"]["archetype"] == "workshop"
    assert result["scene_program"]["recovery_mode"] == "llm"


def test_validate_semantic_intent_accepts_richer_scene_program_fields():
    result = validate_semantic_intent(
        {
            "intent": {
                **_strict_intent(
                    required_roles=["chair", "table"],
                    optional_roles=["lamp"],
                    role_counts={"chair": 2},
                    archetype="study",
                    scene_type="study",
                    density_target="dense",
                    symmetry_preference="asymmetric",
                    confidence=0.9,
                ),
                "primary_anchor_object": {"role": "table", "rationale": "anchor"},
                "secondary_support_objects": [
                    {"role": "chair", "count": 2, "rationale": "seating"},
                    {"role": "lamp", "count": 1, "rationale": "task light"},
                ],
                "relation_graph": [
                    {"source_role": "chair", "target_role": "table", "relation": "face_to"},
                    {"source_role": "lamp", "target_role": "table", "relation": "near"},
                    {"source_role": "table", "target_role": "room", "relation": "middle"},
                ],
            },
            "placement_intent": _strict_placement_intent(
                density_profile="cluttered",
                spatial_preferences=[{"role": "table", "relation": "middle"}],
                layout_mood="crowded",
            ),
        },
        prompt_text="a compact study with two chairs around a central desk",
    )

    assert result["ok"] is True
    assert result["scene_program"]["primary_anchor_object"]["role"] == "table"
    assert len(result["scene_program"]["secondary_support_objects"]) == 2
    assert len(result["scene_program"]["relation_graph"]) == 3
    assert result["placement_intent"]["density_profile"] == "cluttered"
    assert {"role": "table", "relation": "middle"} in result["placement_intent"]["spatial_preferences"]


def test_planner_wires_optional_additions_into_worldspec():
    prefs = _inline_llm_plan(
        shortlist_asset_ids("build a reading room", "chair", "table"),
        max_props=3,
        required_roles=["chair", "table"],
        optional_additions=[
            {
                "asset_id": shortlist_asset_ids("build a reading room", "decor")[0],
                "anchor": "wall",
                "placement_mode": "wall_hung",
                "placement_hint": "wall_centered",
                "usage": "accent",
            }
        ],
    )
    result = plan_worldspec("build a reading room", user_prefs=prefs)

    assert result["ok"] is True
    assert result["worldspec"]["optional_additions"]
    assert result["worldspec"]["optional_additions"][0]["anchor"] == "wall"
    assert result["worldspec"]["optional_additions"][0]["placement_hint"] == "wall_centered"
    optional_placement = next(
        placement for placement in result["worldspec"]["placements"] if placement["placement_id"].startswith("optional_")
    )
    assert optional_placement["placement_hint"] == "wall_centered"


def test_semantic_grounding_preserves_concepts_before_runtime_roles():
    assert canonicalize_semantic_concept("bedside table") == "nightstand"
    assert map_semantic_concept_to_runtime_role("nightstand") == ("table", "nightstand")
    assert map_semantic_concept_to_runtime_role("dresser") == ("cabinet", "dresser")
    assert map_semantic_concept_to_runtime_role("desk") == ("table", "desk")
    assert map_semantic_concept_to_runtime_role("floor lamp") == ("lamp", "floor_lamp")
    assert map_semantic_concept_to_runtime_role("display plinth") == ("table", "display_surface")
    assert map_semantic_concept_to_runtime_role("wall graphics") == ("decor", "wall_art")


def test_validate_semantic_intent_accepts_semantic_slots_and_design_brief():
    result = validate_semantic_intent(
        {
            "design_brief": {
                "concept_statement": "quiet editorial bedroom",
                "palette_strategy": "warm_neutral",
                "lighting_layers": ["ambient", "task"],
            },
            "intent": {
                "scene_type": "bedroom",
                "concept_label": "bedroom",
                "creative_summary": "quiet bedroom",
                "intended_use": "sleep and unwind",
                "focal_object_role": "bed",
                "focal_wall": "front",
                "circulation_preference": "clear_center",
                "empty_space_preference": "balanced",
                "creative_tags": ["restful"],
                "mood_tags": ["calm"],
                "style_descriptors": ["editorial"],
                "execution_archetype": "bedroom",
                "semantic_slots": [
                    {"slot_id": "sleep_anchor_1", "concept": "bed", "priority": "must", "count": 1},
                    {"slot_id": "bedside_surface_1", "concept": "nightstand", "priority": "should", "count": 1},
                ],
                "primary_anchor_object": {"role": "bed", "rationale": "sleep anchor"},
                "relation_graph": [
                    {"source_role": "bed", "target_role": "room", "relation": "middle"},
                    {"source_role": "table", "target_role": "bed", "relation": "near"},
                ],
                "groups": [
                    {
                        "group_id": "group_1",
                        "group_type": "bedside_cluster",
                        "anchor_role": "bed",
                        "member_role": "table",
                        "member_count": 1,
                        "layout_pattern": "beside_anchor",
                        "facing_rule": "parallel",
                        "symmetry": "balanced",
                        "zone_preference": "back",
                        "importance": "primary",
                    }
                ],
                "negative_constraints": ["no_workshop_clutter"],
                "optional_addition_policy": {"allow_optional_additions": True},
                "surface_material_intent": {"wall_tags": ["warm"]},
                "density_target": "normal",
                "symmetry_preference": "balanced",
                "walkway_preservation_intent": {"keep_entry_clear": True},
                "scene_features": [],
                "style_cues": {"style_tags": ["editorial"], "color_tags": ["warm"], "lighting_tags": [], "mood_tags": ["calm"]},
                "confidence": 0.8,
            },
            "placement_intent": _strict_placement_intent(),
        },
        prompt_text="bedroom",
    )

    assert result["ok"] is True
    assert result["scene_program"]["design_brief"]["palette_strategy"] == "warm_neutral"
    assert result["scene_program"]["semantic_slots"][0]["slot_id"] == "sleep_anchor_1"
    assert {slot["concept"] for slot in result["scene_program"]["semantic_slots"]} == {"bed", "nightstand"}


def test_validate_semantic_intent_preserves_semantic_slot_counts():
    result = validate_semantic_intent(
        {
            "intent": _strict_intent(
                required_roles=["chair", "table"],
                role_counts={"chair": 2, "table": 1},
                archetype="study",
                scene_type="study",
            ),
            "placement_intent": _strict_placement_intent(),
        },
        prompt_text="study",
    )

    assert result["ok"] is True
    slot_ids = {slot["slot_id"] for slot in result["scene_program"]["semantic_slots"]}
    assert slot_ids == {"chair_slot_1", "table_slot_1"}
    chair_slot = next(slot for slot in result["scene_program"]["semantic_slots"] if slot["slot_id"] == "chair_slot_1")
    assert chair_slot["count"] == 2
    assert {slot["concept"] for slot in result["scene_program"]["semantic_slots"]} == {"chair", "table"}


def test_complete_scene_program_adds_missing_bedside_and_storage_support():
    scene_program = complete_scene_program(
        validate_semantic_intent(
            {
                "intent": {
                    **_strict_intent(required_roles=["bed"], archetype="bedroom", scene_type="bedroom", groups=[]),
                    "semantic_slots": [{"slot_id": "sleep_anchor_1", "concept": "bed", "priority": "must", "count": 1}],
                    "groups": [],
                },
                "placement_intent": _strict_placement_intent(),
            },
            prompt_text="bedroom",
        )["scene_program"],
        "bedroom",
    )

    concepts = {slot["concept"] for slot in scene_program["semantic_slots"]}
    assert "nightstand" in concepts
    assert "dresser" in concepts


def test_validate_semantic_plan_accepts_slot_asset_map_and_grounded_slots():
    candidate_assets = [
        {"asset_id": "core_bed_01", "label": "bed", "tags": ["bed"], "planner_approved": True, "planner_excluded": False},
        {"asset_id": "core_table_01", "label": "nightstand", "tags": ["table", "nightstand"], "room_role_subtype": "nightstand", "planner_approved": True, "planner_excluded": False},
    ]
    result = validate_semantic_plan(
        {
            "design_brief": {"palette_strategy": "warm_neutral"},
            "intent": {
                **_strict_intent(required_roles=["bed", "table"], archetype="bedroom", scene_type="bedroom"),
                "semantic_slots": [
                    {"slot_id": "sleep_anchor_1", "concept": "bed", "priority": "must", "count": 1},
                    {"slot_id": "bedside_surface_1", "concept": "nightstand", "priority": "should", "count": 1},
                ],
                "primary_anchor_object": {"role": "bed", "rationale": "sleep anchor"},
                "groups": [
                    {
                        "group_id": "group_1",
                        "group_type": "bedside_cluster",
                        "anchor_role": "bed",
                        "member_role": "table",
                        "member_count": 1,
                        "layout_pattern": "beside_anchor",
                        "facing_rule": "parallel",
                        "symmetry": "balanced",
                        "zone_preference": "back",
                        "importance": "primary",
                    }
                ],
            },
            "selection": {
                "selected_prompt": "bedroom",
                "stylekit_id": "neutral_daylight",
                "pack_ids": ["core_pack"],
                "slot_asset_map": {
                    "sleep_anchor_1": "core_bed_01",
                    "bedside_surface_1": "core_table_01",
                },
                "surface_material_selection": approved_surface_material_selection(style_tags=["cozy"], color_tags=["warm"]),
            },
        },
        all_assets=candidate_assets,
        allowed_stylekit_ids=["neutral_daylight"],
        allowed_pack_ids=["core_pack"],
        default_budgets={"max_props": 4, "max_props_hard": 6, "max_floor_objects": 4, "max_wall_objects": 2, "max_surface_objects": 2, "max_texture_tier": 1, "max_lights": 2, "max_clutter_weight": 4},
        prompt_text="bedroom",
        placement_intent=_strict_placement_intent(),
        surface_material_candidates=_surface_material_candidates(),
    )

    assert result["ok"] is True
    assert result["semantic_selection"]["slot_asset_map"]["sleep_anchor_1"] == "core_bed_01"
    assert "role_asset_map" not in result["semantic_selection"]
    assert result["semantic_selection"]["covered_required_slots"] == ["sleep_anchor_1"]
    assert any(
        entry["slot_id"] == "bedside_surface_1" and entry["status"] == "covered" and entry["requiredness"] == "soft"
        for entry in result["semantic_selection"]["slot_diagnostics"]
    )
    assert result["scene_program"]["grounded_slots"][0]["runtime_role"] == "bed"


def test_validate_semantic_plan_accepts_slot_asset_map_and_derives_slot_map():
    candidate_assets = [
        {"asset_id": "core_bed_01", "label": "bed", "tags": ["bed"], "planner_approved": True, "planner_excluded": False},
        {"asset_id": "core_table_01", "label": "nightstand", "tags": ["table", "nightstand"], "room_role_subtype": "nightstand", "planner_approved": True, "planner_excluded": False},
    ]
    result = validate_semantic_plan(
        {
            "intent": _strict_intent(required_roles=["bed", "table"], archetype="bedroom", scene_type="bedroom"),
            "selection": {
                "selected_prompt": "bedroom",
                "stylekit_id": "neutral_daylight",
                "pack_ids": ["core_pack"],
                "slot_asset_map": {"bed_slot_1": "core_bed_01", "table_slot_1": "core_table_01"},
                "surface_material_selection": approved_surface_material_selection(style_tags=["cozy"], color_tags=["warm"]),
            },
        },
        all_assets=candidate_assets,
        allowed_stylekit_ids=["neutral_daylight"],
        allowed_pack_ids=["core_pack"],
        default_budgets={"max_props": 4, "max_props_hard": 6, "max_floor_objects": 4, "max_wall_objects": 2, "max_surface_objects": 2, "max_texture_tier": 1, "max_lights": 2, "max_clutter_weight": 4},
        prompt_text="bedroom",
        placement_intent=_strict_placement_intent(),
        surface_material_candidates=_surface_material_candidates(),
    )

    assert result["ok"] is True
    assert result["semantic_selection"]["slot_asset_map"] == {
        "bed_slot_1": "core_bed_01",
        "table_slot_1": "core_table_01",
    }
    assert "role_asset_map" not in result["semantic_selection"]
    assert result["semantic_selection"]["covered_required_slots"] == ["bed_slot_1", "table_slot_1"]


def test_validate_semantic_plan_accepts_slot_first_group_assignments_without_role_map():
    candidate_assets = [
        {"asset_id": "core_bed_01", "label": "bed", "tags": ["bed"], "planner_approved": True, "planner_excluded": False},
        {"asset_id": "core_table_01", "label": "nightstand", "tags": ["table", "nightstand"], "room_role_subtype": "nightstand", "planner_approved": True, "planner_excluded": False},
    ]
    result = validate_semantic_plan(
        {
            "design_brief": {"palette_strategy": "warm_neutral"},
            "intent": {
                **_strict_intent(required_roles=["bed", "table"], archetype="bedroom", scene_type="bedroom"),
                "semantic_slots": [
                    {"slot_id": "sleep_anchor_1", "concept": "bed", "priority": "must", "count": 1, "group_id": "group_1"},
                    {"slot_id": "bedside_surface_1", "concept": "nightstand", "priority": "should", "count": 1, "group_id": "group_1"},
                ],
                "primary_anchor_object": {"role": "bed", "rationale": "sleep anchor"},
                "groups": [
                    {
                        "group_id": "group_1",
                        "group_type": "bedside_cluster",
                        "anchor_role": "bed",
                        "member_role": "table",
                        "member_count": 1,
                        "layout_pattern": "beside_anchor",
                        "facing_rule": "parallel",
                        "symmetry": "balanced",
                        "zone_preference": "back",
                        "importance": "primary",
                    }
                ],
            },
            "selection": {
                "selected_prompt": "bedroom",
                "stylekit_id": "neutral_daylight",
                "pack_ids": ["core_pack"],
                "slot_asset_map": {
                    "sleep_anchor_1": "core_bed_01",
                    "bedside_surface_1": "core_table_01",
                },
                "group_assignments": [
                    {
                        "group_id": "group_1",
                        "slot_asset_map": {
                            "sleep_anchor_1": "core_bed_01",
                            "bedside_surface_1": "core_table_01",
                        },
                    }
                ],
                "surface_material_selection": approved_surface_material_selection(style_tags=["cozy"], color_tags=["warm"]),
            },
        },
        all_assets=candidate_assets,
        allowed_stylekit_ids=["neutral_daylight"],
        allowed_pack_ids=["core_pack"],
        default_budgets={"max_props": 4, "max_props_hard": 6, "max_floor_objects": 4, "max_wall_objects": 2, "max_surface_objects": 2, "max_texture_tier": 1, "max_lights": 2, "max_clutter_weight": 4},
        prompt_text="bedroom",
        placement_intent=_strict_placement_intent(),
        surface_material_candidates=_surface_material_candidates(),
    )

    assert result["ok"] is True
    assignment = result["semantic_selection"]["group_assignments"][0]
    assert assignment["slot_asset_map"]["sleep_anchor_1"] == "core_bed_01"
    assert "role_asset_map" not in assignment


def test_validate_semantic_plan_preserves_rejected_candidates_by_slot():
    candidate_assets = [
        {"asset_id": "core_table_01", "label": "nightstand", "tags": ["table", "nightstand"], "room_role_subtype": "nightstand", "planner_approved": True, "planner_excluded": False},
        {"asset_id": "core_table_02", "label": "side table", "tags": ["table"], "room_role_subtype": "side_table", "planner_approved": True, "planner_excluded": False},
        {"asset_id": "core_table_03", "label": "desk", "tags": ["table", "desk"], "room_role_subtype": "desk", "planner_approved": True, "planner_excluded": False},
    ]
    result = validate_semantic_plan(
        {
            "design_brief": {"palette_strategy": "warm_neutral"},
            "intent": {
                **_strict_intent(required_roles=["table"], archetype="bedroom", scene_type="bedroom"),
                "semantic_slots": [
                    {"slot_id": "bedside_surface_1", "concept": "nightstand", "priority": "should", "count": 1},
                ],
                "primary_anchor_object": {"role": "table", "rationale": "bedside support"},
            },
            "selection": {
                "selected_prompt": "bedroom",
                "stylekit_id": "neutral_daylight",
                "pack_ids": ["core_pack"],
                "slot_asset_map": {"bedside_surface_1": "core_table_01"},
                "rejected_candidate_ids": ["core_table_02", "core_table_03"],
                "rejected_candidates_by_slot": {
                    "bedside_surface_1": [
                        {"asset_id": "core_table_02", "reason": "too generic for a bedside surface"},
                        {"asset_id": "core_table_03", "reason": "reads as a desk instead of a nightstand"},
                        {"asset_id": "core_table_03", "reason": "duplicate should be dropped"},
                        {"asset_id": "missing_asset", "reason": "should be ignored"},
                    ]
                },
                "surface_material_selection": approved_surface_material_selection(style_tags=["cozy"], color_tags=["warm"]),
            },
        },
        all_assets=candidate_assets,
        allowed_stylekit_ids=["neutral_daylight"],
        allowed_pack_ids=["core_pack"],
        default_budgets={"max_props": 4, "max_props_hard": 6, "max_floor_objects": 4, "max_wall_objects": 2, "max_surface_objects": 2, "max_texture_tier": 1, "max_lights": 2, "max_clutter_weight": 4},
        prompt_text="bedroom",
        placement_intent=_strict_placement_intent(),
        surface_material_candidates=_surface_material_candidates(),
    )

    assert result["ok"] is True
    assert result["semantic_selection"]["rejected_candidates_by_slot"] == {
        "bedside_surface_1": [
            {"asset_id": "core_table_02", "reason": "too generic for a bedside surface"},
            {"asset_id": "core_table_03", "reason": "reads as a desk instead of a nightstand"},
        ]
    }
