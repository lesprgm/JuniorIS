from __future__ import annotations

from src.catalog.style_material_pool import build_surface_material_candidates
from src.catalog.pack_registry import load_pack_registry
from src.placement.geometry import canonicalize_semantic_role, semantic_role_key
from src.planning.assets import build_semantic_candidate_shortlist, collect_assets


# Keep behavior deterministic so planner/runtime contracts stay stable.
def shortlist_asset_ids(prompt_text: str, *roles: str) -> list[str]:
    registry = load_pack_registry()
    semantic_slots = [
        {
            "slot_id": f"{canonicalize_semantic_role(role)}_slot_{index}",
            "concept": canonicalize_semantic_role(role),
            "priority": "must",
            "count": 1,
            "runtime_role_hint": canonicalize_semantic_role(role),
        }
        for index, role in enumerate(roles, start=1)
    ]
    shortlist = build_semantic_candidate_shortlist(
        collect_assets([], registry),
        prompt_text,
        limit=40,
        intent_spec={"semantic_slots": semantic_slots},
    )

    selected_ids: list[str] = []
    seen_ids: set[str] = set()
    for role in roles:
        canonical_role = canonicalize_semantic_role(role)
        for asset in shortlist:
            asset_id = str(asset.get("asset_id", ""))
            if not asset_id or asset_id in seen_ids:
                continue
            if semantic_role_key(asset) == canonical_role:
                seen_ids.add(asset_id)
                selected_ids.append(asset_id)
                break
    return selected_ids


def _strict_intent(
    *,
    scene_type: str,
    required_roles: list[str],
    optional_roles: list[str] | None = None,
    style_tags: list[str] | None = None,
    color_tags: list[str] | None = None,
    confidence: float = 0.9,
):
    optional_roles = optional_roles or []
    style_tags = style_tags or []
    color_tags = color_tags or []
    supported_archetypes = {"study", "bedroom", "lounge", "workshop", "kitchen", "bathroom", "generic_room"}
    all_roles = required_roles + [role for role in optional_roles if role not in required_roles]
    anchor_role = "table" if "table" in all_roles else all_roles[0]
    archetype = scene_type if scene_type in supported_archetypes else ("study" if "table" in all_roles else "generic_room")
    relation_graph = [{"source_role": anchor_role, "target_role": "room", "relation": "middle"}]
    for role in all_roles:
        if role == anchor_role:
            continue
        relation_graph.append(
            {
                "source_role": role,
                "target_role": anchor_role,
                "relation": "face_to" if role == "chair" and anchor_role == "table" else "near",
            }
        )
    groups: list[dict] = []
    if len(all_roles) >= 2:
        if "table" in all_roles and "chair" in all_roles:
            chair_count = max(1, int({role: 1 for role in required_roles}.get("chair", 1)))
            groups = [
                {
                    "group_id": "group_1",
                    "group_type": "dining_set" if chair_count > 1 or archetype in {"kitchen", "lounge"} else "reading_corner",
                    "anchor_role": "table" if chair_count > 1 or archetype in {"kitchen", "lounge"} else "chair",
                    "member_role": "chair" if chair_count > 1 or archetype in {"kitchen", "lounge"} else "table",
                    "member_count": chair_count if chair_count > 1 or archetype in {"kitchen", "lounge"} else 1,
                    "layout_pattern": "paired_long_sides" if chair_count > 1 or archetype in {"kitchen", "lounge"} else "beside_anchor",
                    "facing_rule": "toward_anchor",
                    "symmetry": "balanced",
                    "zone_preference": "center" if chair_count > 1 or archetype in {"kitchen", "lounge"} else "corner",
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
                    "symmetry": "balanced",
                    "zone_preference": "edge",
                }
            ]
    semantic_slots = [
        {
            "slot_id": f"{canonicalize_semantic_role(role)}_slot_{index}",
            "concept": canonicalize_semantic_role(role),
            "priority": "optional" if role in optional_roles and role not in required_roles else "must",
            "count": 1,
            "runtime_role_hint": canonicalize_semantic_role(role),
        }
        for index, role in enumerate(all_roles, start=1)
    ]
    return {
        "scene_type": scene_type,
        "concept_label": scene_type,
        "creative_summary": f"{scene_type.replace('_', ' ')} scene",
        "intended_use": f"use the room as a {scene_type.replace('_', ' ')}",
        "focal_object_role": anchor_role,
        "focal_wall": "front",
        "circulation_preference": "clear_center",
        "empty_space_preference": "balanced",
        "creative_tags": [scene_type],
        "mood_tags": style_tags[:1],
        "style_descriptors": style_tags,
        "execution_archetype": archetype,
        "archetype": archetype,
        "semantic_slots": semantic_slots,
        "primary_anchor_object": {"role": anchor_role, "rationale": "primary anchor"},
        "secondary_support_objects": [
            {"role": role, "count": 1, "rationale": "support object"}
            for role in all_roles
            if role != anchor_role
        ],
        "relation_graph": relation_graph,
        "groups": groups,
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
        "density_target": "normal",
        "symmetry_preference": "balanced",
        "walkway_preservation_intent": {
            "keep_central_path_clear": True,
            "keep_entry_clear": True,
            "notes": "preserve a clear path through the room",
        },
        "scene_features": [],
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


def approved_surface_material_selection(
    *,
    style_tags: list[str] | None = None,
    color_tags: list[str] | None = None,
) -> dict[str, str]:
    candidates = build_surface_material_candidates(
        {
            "style_tags": style_tags or [],
            "color_tags": color_tags or [],
            "style_cues": {
                "style_tags": style_tags or [],
                "color_tags": color_tags or [],
            },
        }
    )
    selection: dict[str, str] = {}
    for surface in ("wall", "floor", "ceiling", "accent"):
        entries = candidates.get(surface) or []
        if entries:
            selection[surface] = str(entries[0].get("material_id") or "")
    return selection


def inline_semantic_prefs(
    prompt_text: str,
    *,
    scene_type: str,
    required_roles: list[str],
    optional_roles: list[str] | None = None,
    style_tags: list[str] | None = None,
    color_tags: list[str] | None = None,
    selected_prompt: str | None = None,
    max_props: int = 3,
    stylekit_id: str = "neutral_daylight",
    pack_ids: list[str] | None = None,
    confidence: float = 0.9,
    density_profile: str = "normal",
    layout_mood: str | None = None,
    surface_material_selection: dict[str, str] | None = None,
    optional_additions: list[dict[str, str]] | None = None,
) -> dict:
    asset_ids = shortlist_asset_ids(prompt_text, *required_roles)
    semantic_slots = []
    slot_asset_map = {}
    alternatives = {}
    for index, (role, asset_id) in enumerate(zip(required_roles, asset_ids), start=1):
        slot_id = f"{canonicalize_semantic_role(role)}_slot_{index}"
        semantic_slots.append(
            {
                "slot_id": slot_id,
                "concept": canonicalize_semantic_role(role),
                "priority": "must",
                "count": 1,
                "runtime_role_hint": canonicalize_semantic_role(role),
            }
        )
        slot_asset_map[slot_id] = asset_id
        alternatives[slot_id] = [asset_id]
    supported_archetypes = {"study", "bedroom", "lounge", "workshop", "kitchen", "bathroom", "generic_room"}
    archetype = scene_type if scene_type in supported_archetypes else ("study" if "table" in required_roles else "generic_room")

    intent = _strict_intent(
        scene_type=scene_type,
        required_roles=required_roles,
        optional_roles=optional_roles or [],
        style_tags=style_tags or [],
        color_tags=color_tags or [],
        confidence=confidence,
    )
    group_assignments = []
    for group in intent.get("groups", []):
        anchor_role = group["anchor_role"]
        member_role = group["member_role"]
        anchor_slot_id = next((slot["slot_id"] for slot in semantic_slots if slot["concept"] == anchor_role), "")
        member_slot_id = next((slot["slot_id"] for slot in semantic_slots if slot["concept"] == member_role), "")
        if anchor_slot_id in slot_asset_map and member_slot_id in slot_asset_map:
            group_assignments.append(
                {
                    "group_id": group["group_id"],
                    "slot_asset_map": {
                        anchor_slot_id: slot_asset_map[anchor_slot_id],
                        member_slot_id: slot_asset_map[member_slot_id],
                    },
                }
            )
    intent["semantic_slots"] = semantic_slots
    return {
        "llm_plan": {
            "intent": intent,
            "placement_intent": {
                "density_profile": density_profile,
                "anchor_preferences": [],
                "adjacency_pairs": [
                    {"source_role": "chair", "target_role": "lamp", "relation": "near"}
                ]
                if "chair" in required_roles and "lamp" in required_roles
                else [],
                "spatial_preferences": [{"role": "table", "relation": "middle"}]
                if "table" in required_roles
                else [],
                "layout_mood": layout_mood or ("crowded" if density_profile == "cluttered" else "cozy"),
            },
            "selection": {
                "selected_prompt": selected_prompt or prompt_text,
                "stylekit_id": stylekit_id,
                "pack_ids": pack_ids or ["core_pack"],
                "group_assignments": group_assignments,
                "slot_asset_map": slot_asset_map,
                "asset_ids": asset_ids,
                "budgets": {"max_props": max_props},
                "optional_additions": optional_additions or [],
                "decor_plan": {
                    "entries": [
                        {"kind": "frame", "anchor": "wall", "zone_id": "focus_wall" if archetype == "study" else "focal_wall", "count": 1}
                    ],
                    "rationale": ["accent the focal wall without crowding the room"],
                },
                "surface_material_selection": surface_material_selection
                or approved_surface_material_selection(
                    style_tags=style_tags or [],
                    color_tags=color_tags or [],
                ),
                "alternatives": alternatives,
                "rationale": ["semantic selection matched the requested roles"],
                "confidence": confidence,
            },
        }
    }
