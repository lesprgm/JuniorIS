from __future__ import annotations

from src.runtime.decor_plan import build_runtime_decor_plan, build_runtime_scene_context


# Keep behavior deterministic so planner/runtime contracts stay stable.
def _slots(*roles: str) -> list[dict]:
    return [
        {
            "slot_id": f"{role}_slot_{index}",
            "concept": role,
            "runtime_role_hint": role,
            "priority": "must",
            "count": 1,
        }
        for index, role in enumerate(roles, start=1)
    ]


def _grounded(*roles: str) -> list[dict]:
    return [
        {
            "slot_id": f"{role}_slot_{index}",
            "concept": role,
            "runtime_role": role,
            "priority": "must",
            "count": 1,
        }
        for index, role in enumerate(roles, start=1)
    ]


def test_scene_context_keeps_model_scene_fields_without_authored_zones():
    scene_context = build_runtime_scene_context(
        intent_spec={"scene_type": "indoor_room", "semantic_slots": _slots("chair", "table")},
        placement_intent={"density_profile": "normal", "layout_mood": "cozy"},
        selected_assets=[
            {"asset_id": "a", "label": "chair", "tags": ["chair"]},
            {"asset_id": "b", "label": "table", "tags": ["table"]},
        ],
        scene_program={
            "scene_type": "study",
            "archetype": "study",
            "execution_archetype": "study",
            "concept_label": "museum study",
            "creative_tags": ["museum"],
            "mood_tags": ["quiet"],
            "style_descriptors": ["classical"],
            "semantic_slots": _slots("chair", "table"),
            "grounded_slots": _grounded("chair", "table"),
            "style_cues": {"style_tags": ["cozy"]},
        },
    )

    assert scene_context["archetype"] == "study"
    assert scene_context["execution_archetype"] == "study"
    assert scene_context["concept_label"] == "museum study"
    assert scene_context["creative_tags"] == ["museum"]
    assert scene_context["mood_tags"] == ["quiet"]
    assert scene_context["style_descriptors"] == ["classical"]
    assert scene_context["zones"] == []


def test_decor_plan_preserves_model_entries_without_zone_templates():
    scene_program = {
        "scene_type": "lounge",
        "archetype": "lounge",
        "semantic_slots": _slots("sofa"),
        "grounded_slots": _grounded("sofa"),
        "style_cues": {"style_tags": ["soft"]},
    }
    scene_context = build_runtime_scene_context(
        intent_spec={"scene_type": "living_room"},
        placement_intent={"density_profile": "minimal", "layout_mood": "open"},
        selected_assets=[
            {"asset_id": "a", "label": "sofa", "tags": ["sofa"]},
            {
                "asset_id": "wall_frame",
                "label": "decor",
                "allowed_anchors": ["wall"],
                "placement_modes": ["wall_hung"],
                "usable_roles": ["wall_accent", "focal_art"],
            },
            {
                "asset_id": "corner_plant",
                "label": "plant",
                "allowed_anchors": ["floor"],
                "placement_modes": ["standalone"],
                "usable_roles": ["plant"],
            },
        ],
        scene_program=scene_program,
    )

    decor_plan = build_runtime_decor_plan(
        intent_spec={"scene_type": "living_room"},
        placement_intent={"density_profile": "minimal", "layout_mood": "open"},
        selected_assets=[{"asset_id": "a", "label": "sofa", "tags": ["sofa"]}],
        scene_context=scene_context,
        scene_program=scene_program,
        selection_decor_plan={
            "entries": [
                {"asset_id": "wall_frame", "kind": "frame", "anchor": "wall", "zone_id": "focal_wall", "count": 1, "placement_hint": "wall_centered"},
                {"kind": "plant", "anchor": "corner", "zone_id": "soft_corner", "count": 2},
            ],
            "rationale": ["keep decor at the room edge"],
        },
    )

    assert decor_plan["archetype"] == "lounge"
    assert decor_plan["entries"] == [
        {"asset_id": "wall_frame", "kind": "frame", "anchor": "wall", "zone_id": "focal_wall", "count": 1, "placement_hint": "wall_centered"},
        {"kind": "plant", "anchor": "corner", "zone_id": "soft_corner", "count": 2},
    ]
    assert decor_plan["rationale"] == ["keep decor at the room edge"]


def test_decor_plan_drops_invalid_model_entries_instead_of_forcing_templates():
    scene_program = {
        "scene_type": "study",
        "archetype": "study",
        "semantic_slots": _slots("chair", "table"),
        "grounded_slots": _grounded("chair", "table"),
        "style_cues": {"style_tags": ["cozy"]},
    }
    decor_plan = build_runtime_decor_plan(
        intent_spec=None,
        placement_intent=None,
        selected_assets=[
            {
                "asset_id": "wall_frame",
                "label": "decor",
                "allowed_anchors": ["wall"],
                "placement_modes": ["wall_hung"],
                "usable_roles": ["wall_accent", "focal_art"],
            }
        ],
        scene_program=scene_program,
        selection_decor_plan={
            "entries": [
                {"kind": "unknown", "anchor": "wall", "zone_id": "focus_wall", "count": 1},
                {"kind": "frame", "anchor": "bad_anchor", "zone_id": "focus_wall", "count": 1},
            ],
        },
    )

    assert decor_plan["archetype"] == "study"
    assert decor_plan["entries"] == []


def test_decor_plan_uses_metadata_driven_kinds_and_respects_optional_policy():
    scene_program = {
        "scene_type": "study",
        "archetype": "study",
        "semantic_slots": _slots("chair", "table"),
        "grounded_slots": _grounded("chair", "table"),
        "optional_addition_policy": {
            "allow_optional_additions": True,
            "max_count": 1,
        },
        "style_cues": {"style_tags": ["cozy"]},
    }
    decor_plan = build_runtime_decor_plan(
        intent_spec=None,
        placement_intent=None,
        selected_assets=[
            {
                "asset_id": "wall_frame",
                "label": "decor",
                "allowed_anchors": ["wall"],
                "placement_modes": ["wall_hung"],
                "usable_roles": ["wall_accent", "focal_art"],
            },
            {
                "asset_id": "book_stack",
                "label": "decor",
                "support_surface_types": ["shelf", "tabletop"],
                "usable_roles": ["book", "stack"],
            },
        ],
        scene_program=scene_program,
        selection_decor_plan={
            "entries": [
                {"kind": "frame", "anchor": "focus_wall", "zone_id": "focus_wall", "count": 3},
                {"kind": "stack", "anchor": "perimeter_edge", "zone_id": "shelf", "count": 1},
            ]
        },
    )

    assert decor_plan["entries"] == [
        {"kind": "frame", "anchor": "wall", "zone_id": "focus_wall", "count": 1},
        {"kind": "stack", "anchor": "perimeter", "zone_id": "shelf", "count": 1},
    ]
