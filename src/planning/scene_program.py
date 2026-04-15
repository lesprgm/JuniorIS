from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List

from src.catalog.style_material_pool import SURFACE_MATERIAL_SLOTS
from src.placement.constraints import normalize_anchor_preferences
from src.placement.geometry import (
    canonicalize_semantic_concept,
    canonicalize_semantic_role,
    map_semantic_concept_to_runtime_role,
    normalize_density_profile,
    normalize_layout_mood,
)
from src.planning.asset_shortlist import asset_allowed_for_slot
from src.planning.archetype_policy import SUPPORTED_ARCHETYPES
from src.planning.scene_types import (
    SUPPORTED_SEMANTIC_ROLES,
    DesignBriefSpec,
    GroundedSlotSpec,
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
from src.runtime.decor_plan import normalize_model_decor_plan
from src.planning.scene_policy import asset_allowed_by_scene_policy


from src.planning.scene_program_constants import (
    ALLOWED_BUDGET_KEYS,
    ALLOWED_CIRCULATION_PREFERENCES,
    ALLOWED_CONSTRAINT_STRENGTHS,
    ALLOWED_DENSITY_TARGETS,
    ALLOWED_EMPTY_SPACE_PREFERENCES,
    ALLOWED_FACING_RULES,
    ALLOWED_FOCAL_WALLS,
    ALLOWED_GROUP_IMPORTANCE,
    ALLOWED_GROUP_LAYOUTS,
    ALLOWED_GROUP_TYPES,
    ALLOWED_OPTIONAL_PLACEMENT_HINTS,
    ALLOWED_RELATION_TYPES,
    ALLOWED_RELATIONS,
    ALLOWED_SLOT_NECESSITIES,
    ALLOWED_SLOT_SOURCES,
    ALLOWED_SYMMETRY_PREFERENCES,
    ALLOWED_TARGET_SURFACE_TYPES,
    ALLOWED_ZONE_PREFERENCES,
    REQUIRED_SURFACE_MATERIAL_SLOTS,
)
# Keep behavior deterministic so planner/runtime contracts stay stable.
def _dedupe_strings(values: Iterable[str]) -> List[str]:  # standardizes string arrays, removing whitespace, lowercasing, and dropping duplicates
    normalized: List[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value or "").strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return normalized


def _normalize_tokens(value: Any) -> List[str]:
    text = str(value or "").strip().lower().replace("/", " ").replace("_", " ").replace("-", " ")
    return [part for part in text.split() if part]


def _asset_allowed_for_optional_addition(asset: Dict[str, Any], scene_program: Dict[str, Any] | None) -> bool:
    return asset_allowed_by_scene_policy(asset, scene_context=scene_program)


def _normalize_feature_token(value: Any) -> str:  # standardizes stylistic/theme tokens for reliable matching, substituting underscores
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")
    parts = [part for part in token.split("_") if part]
    return "_".join(parts)


def _normalize_notes(value: Any) -> str:  # sanitizes free-form text blocks like rationales and notes
    return str(value or "").strip()


def _normalize_tag_list(values: Any) -> List[str]:  # parses a list of stylistic or descriptive tags into normalized forms
    if not isinstance(values, list):
        return []
    return _dedupe_strings(
        str(value).strip().lower()
        for value in values
        if isinstance(value, str) and str(value).strip()
    )


def _normalize_feature_list(values: Any) -> List[str]:  # maps raw feature sets into predictable internal enums
    if not isinstance(values, list):
        return []
    return _dedupe_strings(
        _normalize_feature_token(value)
        for value in values
        if isinstance(value, str) and _normalize_feature_token(value)
    )


def _default_relation_type(relation: str) -> str:
    return {
        "near": "proximity",
        "far": "proximity",
        "face_to": "orientation",
        "align": "orientation",
        "support_on": "support",
        "against_wall": "wall_alignment",
        "centered_on_wall": "wall_alignment",
        "symmetry_with": "symmetry",
        "avoid": "avoidance",
        "edge": "room_position",
        "middle": "room_position",
    }.get(relation, "proximity")


def _normalize_descriptor_list(values: Any) -> List[str]:  # processes stylistic adjectives that the LLM uses for matching
    if not isinstance(values, list):
        return []
    return _dedupe_strings(
        str(value).strip().lower()
        for value in values
        if isinstance(value, str) and str(value).strip()
    )


def _normalize_surface_material_selection(  # validates semantic choice of shell materials against the known candidate pool
    raw_selection: Any,
    surface_material_candidates: Dict[str, List[Dict[str, Any]]] | None,
) -> tuple[Dict[str, str], List[Dict[str, str]]]:
    if not isinstance(surface_material_candidates, dict) or not surface_material_candidates:
        return {}, []

    selection = raw_selection if isinstance(raw_selection, dict) else {}
    normalized: Dict[str, str] = {}
    errors: List[Dict[str, str]] = []

    allowed_by_surface: Dict[str, set[str]] = {}
    for surface in SURFACE_MATERIAL_SLOTS:
        entries = surface_material_candidates.get(surface)
        allowed_by_surface[surface] = {
            str(entry.get("material_id") or "").strip()
            for entry in entries
            if isinstance(entries, list)
            if isinstance(entry, dict) and str(entry.get("material_id") or "").strip()
        }

    for surface in REQUIRED_SURFACE_MATERIAL_SLOTS:
        material_id = str(selection.get(surface) or "").strip()
        if not material_id:
            if allowed_by_surface.get(surface):
                errors.append(
                    {
                        "path": f"$.llm.selection.surface_material_selection.{surface}",
                        "message": f"Semantic planner must choose an approved {surface} material id.",
                    }
                )
            continue
        if material_id not in allowed_by_surface.get(surface, set()):
            errors.append(
                {
                    "path": f"$.llm.selection.surface_material_selection.{surface}",
                    "message": f"Semantic planner must choose {surface} from the provided approved material ids.",
                }
            )
            continue
        normalized[surface] = material_id

    accent_id = str(selection.get("accent") or "").strip()
    if accent_id and accent_id in allowed_by_surface.get("accent", set()):
        normalized["accent"] = accent_id

    return normalized, errors


def _normalize_optional_additions(  # strips out unsupported optional decoration choices (wrong placement usage/anchor)
    raw_value: Any,
    all_assets: List[Dict[str, Any]],
    scene_program: Dict[str, Any] | None = None,
) -> List[Dict[str, str]]:
    if not isinstance(raw_value, list):
        return []
    assets_by_id = {
        str(asset.get("asset_id") or "").strip(): asset
        for asset in all_assets
        if isinstance(asset, dict) and str(asset.get("asset_id") or "").strip()
    }
    normalized: List[Dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for entry in raw_value:
        if not isinstance(entry, dict):
            continue
        asset_id = str(entry.get("asset_id") or "").strip()
        anchor = str(entry.get("anchor") or "").strip().lower()
        placement_mode = str(entry.get("placement_mode") or "").strip().lower()
        usage = str(entry.get("usage") or "").strip().lower()
        placement_hint = str(entry.get("placement_hint") or "").strip().lower()
        asset = assets_by_id.get(asset_id)
        if asset is None or not _asset_allowed_for_optional_addition(asset, scene_program):
            continue
        if anchor not in {"floor", "wall", "surface", "ceiling"}:
            continue
        if usage not in {"support", "accent", "clutter"}:
            continue
        if placement_hint and placement_hint not in ALLOWED_OPTIONAL_PLACEMENT_HINTS:
            continue
        key = (asset_id, anchor, placement_mode, usage, placement_hint)
        if key in seen:
            continue
        seen.add(key)
        item = {
            "asset_id": asset_id,
            "anchor": anchor,
            "placement_mode": placement_mode,
            "usage": usage,
        }
        if placement_hint:
            item["placement_hint"] = placement_hint
        normalized.append(item)
    return normalized


def _normalize_rejected_candidate_ids(raw_value: Any, all_assets: List[Dict[str, Any]]) -> List[str]:
    if not isinstance(raw_value, list):
        return []
    allowed_asset_ids = _approved_asset_ids(all_assets)
    normalized: List[str] = []
    seen: set[str] = set()
    for value in raw_value:
        asset_id = str(value or "").strip()
        if not asset_id or asset_id not in allowed_asset_ids or asset_id in seen:
            continue
        seen.add(asset_id)
        normalized.append(asset_id)
    return normalized


def _normalize_slot_asset_map(
    raw_value: Any,
    *,
    scene_program: Dict[str, Any],
    all_assets: List[Dict[str, Any]],
) -> Dict[str, str]:
    raw_value = raw_value if isinstance(raw_value, dict) else {}
    allowed_asset_ids = _approved_asset_ids(all_assets)
    assets_by_id = {
        str(asset.get("asset_id") or "").strip(): asset
        for asset in all_assets
        if isinstance(asset, dict) and str(asset.get("asset_id") or "").strip()
    }
    known_slot_ids = _known_slot_ids(scene_program)
    scene_slots = {
        str(slot.get("slot_id") or "").strip(): dict(slot)
        for slot in _scene_slots(scene_program)
        if isinstance(slot, dict) and str(slot.get("slot_id") or "").strip()
    }
    normalized: Dict[str, str] = {}
    for slot_id, asset_id in raw_value.items():
        slot_token = str(slot_id or "").strip()
        asset_token = str(asset_id or "").strip()
        if slot_token in known_slot_ids and asset_token in allowed_asset_ids and asset_allowed_for_slot(
            assets_by_id.get(asset_token) or {},
            scene_program=scene_program,
            intent_spec=None,
            prompt_text=str(scene_program.get("source_prompt") or ""),
            slot=scene_slots.get(slot_token),
        ):
            normalized[slot_token] = asset_token
    return normalized


def _normalize_fallback_asset_ids_by_slot(raw_value: Any, *, scene_program: Dict[str, Any], all_assets: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    if not isinstance(raw_value, dict):
        return {}
    allowed_asset_ids = _approved_asset_ids(all_assets)
    assets_by_id = {
        str(asset.get("asset_id") or "").strip(): asset
        for asset in all_assets
        if isinstance(asset, dict) and str(asset.get("asset_id") or "").strip()
    }
    known_slot_ids = _known_slot_ids(scene_program)
    scene_slots = {
        str(slot.get("slot_id") or "").strip(): dict(slot)
        for slot in _scene_slots(scene_program)
        if isinstance(slot, dict) and str(slot.get("slot_id") or "").strip()
    }
    normalized: Dict[str, List[str]] = {}
    for slot_id, asset_ids in raw_value.items():
        slot_token = str(slot_id or "").strip()
        if slot_token not in known_slot_ids or not isinstance(asset_ids, list):
            continue
        filtered: List[str] = []
        seen: set[str] = set()
        for asset_id in asset_ids:
            token = str(asset_id or "").strip()
            if not token or token not in allowed_asset_ids or token in seen:
                continue
            if not asset_allowed_for_slot(
                assets_by_id.get(token) or {},
                scene_program=scene_program,
                intent_spec=None,
                prompt_text=str(scene_program.get("source_prompt") or ""),
                slot=scene_slots.get(slot_token),
            ):
                continue
            seen.add(token)
            filtered.append(token)
        if filtered:
            normalized[slot_token] = filtered
    return normalized


def _normalize_rejected_candidates_by_slot(
    raw_value: Any,
    *,
    scene_program: Dict[str, Any],
    all_assets: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, str]]]:
    if not isinstance(raw_value, dict):
        return {}
    allowed_asset_ids = _approved_asset_ids(all_assets)
    known_slot_ids = _known_slot_ids(scene_program)
    normalized: Dict[str, List[Dict[str, str]]] = {}
    for slot_id, entries in raw_value.items():
        slot_token = str(slot_id or "").strip()
        if slot_token not in known_slot_ids or not isinstance(entries, list):
            continue
        filtered: List[Dict[str, str]] = []
        seen_asset_ids: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            asset_id = str(entry.get("asset_id") or "").strip()
            reason = _normalize_notes(entry.get("reason"))
            if not asset_id or asset_id not in allowed_asset_ids or asset_id in seen_asset_ids or not reason:
                continue
            seen_asset_ids.add(asset_id)
            filtered.append({"asset_id": asset_id, "reason": reason})
        if filtered:
            normalized[slot_token] = filtered
    return normalized


def _approved_asset_ids(all_assets: List[Dict[str, Any]]) -> set[str]:
    return {
        str(asset.get("asset_id") or "").strip()
        for asset in all_assets
        if isinstance(asset, dict) and str(asset.get("asset_id") or "").strip()
    }


def _known_slot_ids(scene_program: Dict[str, Any]) -> set[str]:
    return {
        str(slot.get("slot_id") or "").strip()
        for slot in scene_program.get("semantic_slots") or []
        if isinstance(slot, dict) and str(slot.get("slot_id") or "").strip()
    }


def _slot_role(slot: Dict[str, Any]) -> str:
    runtime_role = canonicalize_semantic_role(slot.get("runtime_role"))
    if runtime_role in SUPPORTED_SEMANTIC_ROLES:
        return runtime_role
    concept = slot.get("concept") or slot.get("runtime_role_hint")
    runtime_role, _ = map_semantic_concept_to_runtime_role(concept)
    runtime_role = runtime_role or canonicalize_semantic_role(slot.get("runtime_role_hint"))
    return runtime_role if runtime_role in SUPPORTED_SEMANTIC_ROLES else ""


def _group_role_slot_ids(scene_program: Dict[str, Any], *, group_id: str, role: str) -> List[str]:
    matching: List[str] = []
    fallback: List[str] = []
    for slot in list(scene_program.get("grounded_slots") or []) + list(scene_program.get("semantic_slots") or []):
        if not isinstance(slot, dict):
            continue
        slot_id = str(slot.get("slot_id") or "").strip()
        if not slot_id or _slot_role(slot) != role:
            continue
        fallback.append(slot_id)
        if str(slot.get("group_id") or "").strip() == group_id:
            matching.append(slot_id)
    return matching or fallback


def _slot_asset_map_to_group_assignments(
    *,
    scene_program: Dict[str, Any],
    slot_asset_map: Dict[str, str],
) -> List[Dict[str, Any]]:
    assignments: List[Dict[str, Any]] = []
    for group in scene_program.get("groups") or []:
        if not isinstance(group, dict):
            continue
        group_id = str(group.get("group_id") or "").strip()
        anchor_role = canonicalize_semantic_role(group.get("anchor_role"))
        member_role = canonicalize_semantic_role(group.get("member_role"))
        if not group_id or not anchor_role or not member_role:
            continue
        group_slot_asset_map: Dict[str, str] = {}
        for slot_id in _group_role_slot_ids(scene_program, group_id=group_id, role=anchor_role):
            asset_id = str(slot_asset_map.get(slot_id) or "").strip()
            if asset_id:
                group_slot_asset_map[slot_id] = asset_id
        for slot_id in _group_role_slot_ids(scene_program, group_id=group_id, role=member_role):
            asset_id = str(slot_asset_map.get(slot_id) or "").strip()
            if asset_id:
                group_slot_asset_map[slot_id] = asset_id
        if group_slot_asset_map:
            assignments.append({"group_id": group_id, "slot_asset_map": group_slot_asset_map})
    return assignments


def _group_spec_by_id(scene_program: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(group.get("group_id") or "").strip(): dict(group)
        for group in scene_program.get("groups") or []
        if isinstance(group, dict) and str(group.get("group_id") or "").strip()
    }


def _normalize_group_assignments(
    raw_value: Any,
    *,
    scene_program: Dict[str, Any],
    all_assets: List[Dict[str, Any]],
    slot_asset_map: Dict[str, str] | None = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    if not isinstance(raw_value, list):
        return [], []
    group_specs = _group_spec_by_id(scene_program)
    allowed_asset_ids = _approved_asset_ids(all_assets)
    full_slot_asset_map = dict(slot_asset_map or {})
    normalized: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    seen_group_ids: set[str] = set()
    for index, value in enumerate(raw_value):
        if not isinstance(value, dict):
            continue
        group_id = str(value.get("group_id") or "").strip()
        spec = group_specs.get(group_id)
        if not group_id or spec is None:
            errors.append(
                {
                    "path": f"$.llm.selection.group_assignments[{index}].group_id",
                    "message": "group_assignments must reference a valid scene_program group_id.",
                }
            )
            continue
        if group_id in seen_group_ids:
            errors.append(
                {
                    "path": f"$.llm.selection.group_assignments[{index}].group_id",
                    "message": "group_assignments may not repeat the same group_id.",
                }
            )
            continue
        seen_group_ids.add(group_id)

        raw_slot_asset_map = value.get("slot_asset_map") if isinstance(value.get("slot_asset_map"), dict) else {}

        normalized_slot_asset_map: Dict[str, str] = {}
        for slot_id, asset_id in raw_slot_asset_map.items():
            slot_token = str(slot_id or "").strip()
            asset_token = str(asset_id or "").strip()
            if asset_token in allowed_asset_ids and slot_token in full_slot_asset_map:
                normalized_slot_asset_map[slot_token] = asset_token

        anchor_role = str(spec.get("anchor_role") or "").strip().lower()
        member_role = str(spec.get("member_role") or "").strip().lower()
        for slot_id in _group_role_slot_ids(scene_program, group_id=group_id, role=anchor_role):
            asset_token = normalized_slot_asset_map.get(slot_id) or full_slot_asset_map.get(slot_id)
            if asset_token in allowed_asset_ids:
                normalized_slot_asset_map.setdefault(slot_id, asset_token)
        for slot_id in _group_role_slot_ids(scene_program, group_id=group_id, role=member_role):
            asset_token = normalized_slot_asset_map.get(slot_id) or full_slot_asset_map.get(slot_id)
            if asset_token in allowed_asset_ids:
                normalized_slot_asset_map.setdefault(slot_id, asset_token)

        covered_roles: set[str] = set()
        for slot_id in _group_role_slot_ids(scene_program, group_id=group_id, role=anchor_role):
            asset_token = normalized_slot_asset_map.get(slot_id)
            if asset_token in allowed_asset_ids:
                covered_roles.add(anchor_role)
        for slot_id in _group_role_slot_ids(scene_program, group_id=group_id, role=member_role):
            asset_token = normalized_slot_asset_map.get(slot_id)
            if asset_token in allowed_asset_ids:
                covered_roles.add(member_role)

        required_group_roles = {anchor_role, member_role}
        if not required_group_roles.issubset(covered_roles):
            errors.append(
                {
                    "path": f"$.llm.selection.group_assignments[{index}]",
                    "message": f"group_assignments for '{group_id}' must choose approved assets for {', '.join(sorted(required_group_roles))} via slot_asset_map.",
                }
            )
            continue
        normalized.append({"group_id": group_id, "slot_asset_map": normalized_slot_asset_map})

    missing_group_ids = sorted(set(group_specs) - seen_group_ids)
    if missing_group_ids:
        errors.append(
            {
                "path": "$.llm.selection.group_assignments",
                "message": f"Semantic planner must assign assets for all scene groups: {', '.join(missing_group_ids)}.",
            }
        )
    return normalized, errors


def _selected_asset_instance(
    asset_record: Dict[str, Any],
    *,
    role: str,
    instance_index: int,
    group_spec: Dict[str, Any] | None = None,
    group_role: str | None = None,
) -> Dict[str, Any]:
    instance = dict(asset_record)
    instance["role"] = role
    instance["selected_role"] = role
    instance["selection_instance_index"] = instance_index
    if isinstance(group_spec, dict):
        instance["group_id"] = str(group_spec.get("group_id") or "").strip() or None
        instance["group_type"] = str(group_spec.get("group_type") or "").strip().lower() or None
        instance["group_layout"] = str(group_spec.get("layout_pattern") or "").strip().lower() or None
        instance["group_facing_rule"] = str(group_spec.get("facing_rule") or "").strip().lower() or None
    if group_role:
        instance["group_role"] = group_role
    return instance


def _expanded_assets_from_selection(
    *,
    scene_program: Dict[str, Any],
    slot_asset_map: Dict[str, str],
    group_assignments: List[Dict[str, Any]],
    all_assets: List[Dict[str, Any]],
    prompt_text: str = "",
) -> tuple[List[Dict[str, Any]], List[str]]:
    by_id = {
        str(asset.get("asset_id") or "").strip(): dict(asset)
        for asset in all_assets
        if isinstance(asset, dict) and str(asset.get("asset_id") or "").strip()
    }
    scene_slots = [slot for slot in _scene_slots(scene_program) if _slot_role(slot) in SUPPORTED_SEMANTIC_ROLES]
    slots_by_id = {
        str(slot.get("slot_id") or "").strip(): slot
        for slot in scene_slots
        if str(slot.get("slot_id") or "").strip()
    }
    group_specs = _group_spec_by_id(scene_program)
    chosen_assets: List[Dict[str, Any]] = []
    errors: List[str] = []
    instance_index = 0
    covered_slot_ids: set[str] = set()
    missing_required_slot_ids: set[str] = set()

    def asset_allowed_for_expansion(asset: Dict[str, Any], slot_id: str) -> bool:
        return asset_allowed_for_slot(
            asset,
            scene_program=scene_program,
            intent_spec=None,
            prompt_text=prompt_text or str(scene_program.get("source_prompt") or ""),
            slot=slots_by_id.get(slot_id),
        )

    def asset_allowed_for_any_slot(asset: Dict[str, Any], slot_ids: List[str]) -> bool:
        return any(asset_allowed_for_expansion(asset, slot_id) for slot_id in slot_ids)

    for assignment in group_assignments:
        group_id = str(assignment.get("group_id") or "").strip()
        group_spec = group_specs.get(group_id) or {}
        slot_asset_map_for_group = assignment.get("slot_asset_map") if isinstance(assignment.get("slot_asset_map"), dict) else {}
        anchor_role = str(group_spec.get("anchor_role") or "").strip().lower()
        member_role = str(group_spec.get("member_role") or "").strip().lower()
        member_count = max(1, int(group_spec.get("member_count") or 1))
        anchor_slot_ids = _group_role_slot_ids(scene_program, group_id=group_id, role=anchor_role)
        member_slot_ids = _group_role_slot_ids(scene_program, group_id=group_id, role=member_role)

        anchor_asset_id = next(
            (
                str(slot_asset_map_for_group.get(slot_id) or slot_asset_map.get(slot_id) or "").strip()
                for slot_id in anchor_slot_ids
                if str(slot_asset_map_for_group.get(slot_id) or slot_asset_map.get(slot_id) or "").strip()
            ),
            "",
        )
        member_asset_id = next(
            (
                str(slot_asset_map_for_group.get(slot_id) or slot_asset_map.get(slot_id) or "").strip()
                for slot_id in member_slot_ids
                if str(slot_asset_map_for_group.get(slot_id) or slot_asset_map.get(slot_id) or "").strip()
            ),
            "",
        )
        anchor_record = by_id.get(anchor_asset_id)
        member_record = by_id.get(member_asset_id)
        if anchor_record is None or not asset_allowed_for_any_slot(anchor_record, anchor_slot_ids):
            hard_missing = [
                slot_id
                for slot_id in anchor_slot_ids
                if _is_hard_required_slot(slots_by_id.get(slot_id, {}), scene_program=scene_program)
            ]
            missing_required_slot_ids.update(hard_missing)
            if hard_missing:
                errors.append(group_id)
            continue
        if member_record is None or not asset_allowed_for_any_slot(member_record, member_slot_ids):
            hard_missing = [
                slot_id
                for slot_id in member_slot_ids
                if _is_hard_required_slot(slots_by_id.get(slot_id, {}), scene_program=scene_program)
            ]
            missing_required_slot_ids.update(hard_missing)
            if hard_missing:
                errors.append(group_id)
            continue

        chosen_assets.append(
            _selected_asset_instance(
                anchor_record,
                role=anchor_role,
                instance_index=instance_index,
                group_spec=group_spec,
                group_role="anchor",
            )
        )
        instance_index += 1
        covered_slot_ids.update(anchor_slot_ids)

        for _ in range(member_count):
            chosen_assets.append(
                _selected_asset_instance(
                    member_record,
                    role=member_role,
                    instance_index=instance_index,
                    group_spec=group_spec,
                    group_role="member",
                )
            )
            instance_index += 1
        covered_slot_ids.update(member_slot_ids)

    for slot in sorted(scene_slots, key=_slot_sort_key):
        slot_id = str(slot.get("slot_id") or "").strip()
        if not slot_id or slot_id in covered_slot_ids:
            continue
        role = _slot_role(slot)
        asset_id = str(slot_asset_map.get(slot_id) or "").strip()
        asset_record = by_id.get(asset_id)
        if asset_record is None or not asset_allowed_for_expansion(asset_record, slot_id):
            if _is_hard_required_slot(slot, scene_program=scene_program):
                missing_required_slot_ids.add(slot_id)
            continue
        for _ in range(max(1, int(slot.get("count") or 1))):
            chosen_assets.append(
                _selected_asset_instance(
                    asset_record,
                    role=role,
                    instance_index=instance_index,
                )
            )
            instance_index += 1
        covered_slot_ids.add(slot_id)

    if missing_required_slot_ids:
        errors.extend(sorted(missing_required_slot_ids))

    return chosen_assets, errors


def _supported_archetypes() -> set[str]:  # proxy to fetch archetype policy enums
    return set(SUPPORTED_ARCHETYPES)


def _normalize_archetype(value: Any) -> str:  # resolves and validates the primary room archetype identifier
    token = _normalize_feature_token(value)
    return token if token in _supported_archetypes() else ""


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


def _slot_priority(value: Any) -> str:
    token = _normalize_feature_token(value)
    if token in {"must", "should", "optional"}:
        return token
    return "should"


def _slot_necessity(value: Any) -> str:
    token = _normalize_feature_token(value)
    return token if token in ALLOWED_SLOT_NECESSITIES else ""


def _slot_source(value: Any) -> str:
    token = _normalize_feature_token(value)
    return token if token in ALLOWED_SLOT_SOURCES else ""


def _slot_is_primary_anchor(slot: Dict[str, Any], scene_program: Dict[str, Any] | None) -> bool:
    if not isinstance(scene_program, dict):
        return False
    anchor = scene_program.get("primary_anchor_object") if isinstance(scene_program.get("primary_anchor_object"), dict) else {}
    anchor_slot_id = str(anchor.get("slot_id") or "").strip()
    slot_id = str(slot.get("slot_id") or "").strip()
    if anchor_slot_id and slot_id:
        return anchor_slot_id == slot_id

    anchor_role = canonicalize_semantic_role(anchor.get("role"))
    if not anchor_role:
        return False
    matching_slots = [
        candidate
        for candidate in _scene_slots(scene_program)
        if _slot_role(candidate) == anchor_role
    ]
    return len(matching_slots) == 1 and bool(slot_id) and str(matching_slots[0].get("slot_id") or "").strip() == slot_id


def _slot_requiredness(slot: Dict[str, Any], *, scene_program: Dict[str, Any] | None = None) -> str:
    if _slot_priority(slot.get("priority")) == "optional":
        return "optional"
    if _slot_is_primary_anchor(slot, scene_program):
        return "hard"
    necessity = _slot_necessity(slot.get("necessity"))
    source = _slot_source(slot.get("source"))
    if necessity == "core" and source == "explicit_prompt":
        return "hard"
    return "soft"


def _is_hard_required_slot(slot: Dict[str, Any], *, scene_program: Dict[str, Any] | None = None) -> bool:
    return _slot_requiredness(slot, scene_program=scene_program) == "hard"


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


def _slot_sort_key(slot: Dict[str, Any]) -> tuple[int, str]:
    priority_rank = {"must": 0, "should": 1, "optional": 2}
    return priority_rank.get(str(slot.get("priority") or "should"), 1), str(slot.get("slot_id") or "")


def _scene_slots(scene_program: Dict[str, Any]) -> List[Dict[str, Any]]:
    grounded_slots = [
        dict(slot)
        for slot in scene_program.get("grounded_slots") or []
        if isinstance(slot, dict) and str(slot.get("slot_id") or "").strip()
    ]
    if grounded_slots:
        return grounded_slots
    return [
        dict(slot)
        for slot in scene_program.get("semantic_slots") or []
        if isinstance(slot, dict) and str(slot.get("slot_id") or "").strip()
    ]


def _known_scene_roles(scene_program: Dict[str, Any]) -> set[str]:
    required_roles, optional_roles, _ = _derive_role_fields_from_slots(_scene_slots(scene_program))
    return {role for role in required_roles + optional_roles if role in SUPPORTED_SEMANTIC_ROLES}


def _required_scene_slots(scene_program: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        slot
        for slot in _scene_slots(scene_program)
        if _slot_role(slot) in SUPPORTED_SEMANTIC_ROLES and _is_hard_required_slot(slot, scene_program=scene_program)
    ]


def _known_roles_from_slots(slots: List[SemanticSlotSpec]) -> set[str]:
    required_roles, optional_roles, _ = _derive_role_fields_from_slots(slots)
    return {role for role in required_roles + optional_roles if role in SUPPORTED_SEMANTIC_ROLES}


def _derive_role_fields_from_slots(slots: List[SemanticSlotSpec]) -> tuple[List[str], List[str], Dict[str, int]]:
    required_roles: List[str] = []
    optional_roles: List[str] = []
    role_counts: Dict[str, int] = {}
    for slot in slots:
        concept = slot.get("concept") or slot.get("runtime_role_hint")
        runtime_role, _ = map_semantic_concept_to_runtime_role(concept)
        runtime_role = runtime_role or canonicalize_semantic_role(slot.get("runtime_role_hint"))
        if runtime_role not in SUPPORTED_SEMANTIC_ROLES:
            continue
        count = max(1, int(slot.get("count") or 1))
        if slot.get("priority") == "optional":
            if runtime_role not in required_roles and runtime_role not in optional_roles:
                optional_roles.append(runtime_role)
        else:
            if runtime_role not in required_roles:
                required_roles.append(runtime_role)
            if runtime_role in optional_roles:
                optional_roles.remove(runtime_role)
        role_counts[runtime_role] = max(role_counts.get(runtime_role, 0), count)
    return required_roles, optional_roles, role_counts


def _grounded_slots_from_selection(
    *,
    scene_program: Dict[str, Any],
    slot_asset_map: Dict[str, str],
    all_assets: List[Dict[str, Any]] | None = None,
) -> List[GroundedSlotSpec]:
    assets_by_id = {
        str(asset.get("asset_id") or "").strip(): asset
        for asset in all_assets or []
        if isinstance(asset, dict) and str(asset.get("asset_id") or "").strip()
    }
    grounded: List[GroundedSlotSpec] = []
    for slot in scene_program.get("semantic_slots") or []:
        if not isinstance(slot, dict):
            continue
        slot_id = str(slot.get("slot_id") or "").strip()
        if not slot_id:
            continue
        asset_id = str(slot_asset_map.get(slot_id) or "").strip()
        concept = canonicalize_semantic_concept(slot.get("concept") or slot.get("runtime_role_hint"))
        runtime_role, subtype = map_semantic_concept_to_runtime_role(concept)
        runtime_role = runtime_role or canonicalize_semantic_role(slot.get("runtime_role_hint"))
        if runtime_role not in SUPPORTED_SEMANTIC_ROLES:
            continue
        asset = assets_by_id.get(asset_id) if asset_id else None
        asset_subtype = _normalize_feature_token((asset or {}).get("room_role_subtype"))
        entry: GroundedSlotSpec = {
            "slot_id": slot_id,
            "concept": concept or runtime_role,
            "runtime_role": runtime_role,
            "count": max(1, int(slot.get("count") or 1)),
            "priority": _slot_priority(slot.get("priority")),
        }
        necessity = _slot_necessity(slot.get("necessity"))
        source = _slot_source(slot.get("source"))
        if necessity:
            entry["necessity"] = necessity
        if source:
            entry["source"] = source
        if subtype or asset_subtype:
            entry["subtype"] = asset_subtype or subtype
        if asset_id:
            entry["asset_id"] = asset_id
        if slot.get("group_id"):
            entry["group_id"] = str(slot.get("group_id"))
        grounded.append(entry)
    return grounded


def _plausibility_warnings(scene_program: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    concepts = {canonicalize_semantic_concept(slot.get("concept")) for slot in scene_program.get("semantic_slots") or [] if isinstance(slot, dict)}
    runtime_roles = _known_scene_roles(scene_program)
    if "bed" in concepts and "table" not in runtime_roles and "cabinet" not in runtime_roles:
        warnings.append("sleep_scene_missing_support")
    if any(token in concepts for token in {"desk", "work_surface"}) and "table" not in runtime_roles:
        warnings.append("work_scene_missing_surface")
    return warnings


def _slot_exists(scene_program: Dict[str, Any], concepts: set[str]) -> bool:
    for slot in scene_program.get("semantic_slots") or []:
        if not isinstance(slot, dict):
            continue
        if canonicalize_semantic_concept(slot.get("concept")) in concepts:
            return True
    return False


def complete_scene_program(scene_program: SceneProgram, prompt_text: str) -> SceneProgram:
    completed = dict(scene_program)
    semantic_slots = [dict(slot) for slot in scene_program.get("semantic_slots") or [] if isinstance(slot, dict)]
    negative_constraints = set(scene_program.get("negative_constraints") or [])
    prompt_tokens = set(_normalize_tokens(prompt_text))

    def add_slot(*, slot_id: str, concept: str, priority: str, runtime_role_hint: str, rationale: str, capabilities: List[str] | None = None):
        if _slot_exists({"semantic_slots": semantic_slots}, {canonicalize_semantic_concept(concept)}):
            return
        semantic_slots.append(
            {
                "slot_id": slot_id,
                "concept": canonicalize_semantic_concept(concept),
                "priority": priority,
                "necessity": "support",
                "source": "deterministic_completion",
                "count": 1,
                "runtime_role_hint": runtime_role_hint,
                "rationale": rationale,
                "capabilities": capabilities or [],
            }
        )

    if _slot_exists(completed, {"bed"}):
        if not any(token in negative_constraints for token in {"no_bedside", "no_nightstand", "avoid_bedside_support"}):
            add_slot(
                slot_id="bedside_support_1",
                concept="nightstand",
                priority="should",
                runtime_role_hint="table",
                rationale="support sleep anchor with a reachable bedside surface",
                capabilities=["bedside_support", "reachable_surface"],
            )
        if not any(token in negative_constraints for token in {"no_storage", "avoid_storage"}):
            add_slot(
                slot_id="sleep_storage_1",
                concept="dresser",
                priority="should",
                runtime_role_hint="cabinet",
                rationale="make the sleeping room feel complete with storage",
                capabilities=["storage_low"],
            )

    if any(token in prompt_tokens for token in {"work", "study", "office", "desk"}) or _slot_exists(completed, {"desk", "work_surface"}):
        add_slot(
            slot_id="work_surface_1",
            concept="desk",
            priority="should",
            runtime_role_hint="table",
            rationale="cover the room's work function with a primary work surface",
            capabilities=["work_surface"],
        )

    if any(token in prompt_tokens for token in {"museum", "gallery", "display", "exhibit", "lounge"}) or _slot_exists(completed, {"media_console", "display_surface"}):
        if not _slot_exists(completed, {"display_surface", "media_console", "nightstand", "desk"}):
            add_slot(
                slot_id="display_support_1",
                concept="display_surface",
                priority="should",
                runtime_role_hint="table",
                rationale="support the focal display or lounge composition with one clear support surface",
                capabilities=["display_support"],
            )

    completed["semantic_slots"] = semantic_slots
    completed["plausibility_warnings"] = _plausibility_warnings(completed)
    return completed


def ground_scene_program(
    scene_program: SceneProgram,
    *,
    slot_asset_map: Dict[str, str] | None = None,
    all_assets: List[Dict[str, Any]] | None = None,
) -> SceneProgram:
    grounded_slots = _grounded_slots_from_selection(
        scene_program=scene_program,
        slot_asset_map=slot_asset_map or {},
        all_assets=all_assets,
    )
    grounded = dict(scene_program)
    grounded["grounded_slots"] = grounded_slots
    grounded["plausibility_warnings"] = _plausibility_warnings(grounded)
    return grounded


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


def _normalize_scene_choice(value: Any, allowed: set[str], fallback: str = "") -> str:
    token = _normalize_feature_token(value)
    if token in allowed:
        return token
    return fallback


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
        if relation["relation"] in {"near", "face_to", "align", "far"}:
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
        if relation not in {"edge", "middle"} or role not in known_roles:
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


def _selection_error(
    *,
    message: str,
    scene_program: Dict[str, Any] | None = None,
    intent_spec: Dict[str, Any] | None = None,
    placement_intent: Dict[str, Any] | None = None,
    error_code: str = "semantic_invalid_selection",
    errors: List[Dict[str, Any]] | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload = {
        "ok": False,
        "error_code": error_code,
        "message": message,
        "scene_program": scene_program,
        "intent_spec": intent_spec,
        "placement_intent": placement_intent,
        "errors": list(errors or []),
    }
    if isinstance(extra, dict):
        payload.update(extra)
    return payload


def _normalize_pack_ids(raw_value: Any, allowed_pack_ids: List[str]) -> List[str]:
    pack_ids = [
        str(pack_id).strip()
        for pack_id in raw_value
        if isinstance(raw_value, list) and isinstance(pack_id, str) and str(pack_id).strip() in allowed_pack_ids
    ]
    return pack_ids or allowed_pack_ids[:1]


def _ground_selection_slots(
    selection: Dict[str, Any],
    *,
    scene_program: Dict[str, Any],
    all_assets: List[Dict[str, Any]],
) -> tuple[Dict[str, Any], Dict[str, str]]:
    slot_asset_map = _normalize_slot_asset_map(
        selection.get("slot_asset_map"),
        scene_program=scene_program,
        all_assets=all_assets,
    )
    grounded_scene_program = ground_scene_program(
        scene_program,
        slot_asset_map=slot_asset_map,
        all_assets=all_assets,
    )
    return grounded_scene_program, slot_asset_map


def _normalized_group_assignments(
    selection: Dict[str, Any],
    *,
    scene_program: Dict[str, Any],
    slot_asset_map: Dict[str, str],
    all_assets: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw_group_assignments = selection.get("group_assignments")
    if not isinstance(raw_group_assignments, list) and slot_asset_map:
        raw_group_assignments = _slot_asset_map_to_group_assignments(
            scene_program=scene_program,
            slot_asset_map=slot_asset_map,
        )
    group_assignments, group_assignment_errors = _normalize_group_assignments(
        raw_group_assignments,
        scene_program=scene_program,
        all_assets=all_assets,
        slot_asset_map=slot_asset_map,
    )
    if group_assignment_errors:
        group_assignments = []
    return group_assignments, group_assignment_errors


def _normalize_selection_budgets(selection: Dict[str, Any], default_budgets: Dict[str, int]) -> Dict[str, int]:
    budgets = dict(default_budgets)
    raw_budgets = selection.get("budgets")
    if isinstance(raw_budgets, dict):
        for key in ALLOWED_BUDGET_KEYS:
            value = raw_budgets.get(key)
            if isinstance(value, int) and value > 0:
                budgets[key] = value
    hard_cap = int(budgets.get("max_props_hard") or default_budgets.get("max_props_hard") or 30)
    budgets["max_props_hard"] = hard_cap
    budgets["max_props"] = min(max(1, int(budgets.get("max_props") or 1)), hard_cap)
    for key in ("max_floor_objects", "max_wall_objects", "max_surface_objects", "max_lights", "max_clutter_weight"):
        if key in budgets and isinstance(budgets[key], int):
            budgets[key] = min(max(0, int(budgets[key])), hard_cap)
    return budgets


def _normalize_alternatives(raw_value: Any, *, scene_program: Dict[str, Any]) -> Dict[str, List[str]]:
    if not isinstance(raw_value, dict):
        return {}
    slot_specs = {
        str(slot.get("slot_id") or "").strip(): dict(slot)
        for slot in _scene_slots(scene_program)
        if isinstance(slot, dict) and str(slot.get("slot_id") or "").strip()
    }
    normalized: Dict[str, List[str]] = {}
    for key, asset_ids in raw_value.items():
        if not isinstance(key, str) or not isinstance(asset_ids, list):
            continue
        key_token = str(key).strip().lower()
        matching_slot_ids: List[str] = []
        if key_token in slot_specs:
            matching_slot_ids.append(key_token)
        else:
            canonical_role = canonicalize_semantic_role(key_token)
            canonical_concept = canonicalize_semantic_concept(key_token)
            for slot_id, slot in slot_specs.items():
                if _slot_role(slot) == canonical_role or canonicalize_semantic_concept(slot.get("concept")) == canonical_concept:
                    matching_slot_ids.append(slot_id)
        values = [
            str(asset_id).strip()
            for asset_id in asset_ids
            if isinstance(asset_id, str) and str(asset_id).strip()
        ]
        if not values:
            continue
        for slot_id in matching_slot_ids:
            existing = normalized.setdefault(slot_id, [])
            for asset_id in values:
                if asset_id not in existing:
                    existing.append(asset_id)
    return normalized


def _normalize_selection_extras(
    selection: Dict[str, Any],
    default_confidence: float,
    *,
    scene_program: Dict[str, Any],
) -> Dict[str, Any]:
    alternatives: Dict[str, List[str]] = {}
    raw_alternatives = selection.get("alternatives")
    if isinstance(raw_alternatives, dict):
        alternatives = _normalize_alternatives(raw_alternatives, scene_program=scene_program)
    rationale = [
        str(item).strip()
        for item in selection.get("rationale") or []
        if isinstance(item, str) and str(item).strip()
    ] if isinstance(selection.get("rationale"), list) else []
    confidence = default_confidence
    if isinstance(selection.get("confidence"), (int, float)):
        confidence = max(0.0, min(float(selection.get("confidence")), 1.0))
    return {"alternatives": alternatives, "rationale": rationale, "confidence": confidence}


def _selection_coverage(
    *,
    scene_program: Dict[str, Any],
    slot_asset_map: Dict[str, str],
    fallback_asset_ids_by_slot: Dict[str, List[str]],
    selection_coverage_errors: List[str],
    prompt_text: str = "",
) -> Dict[str, Any]:
    covered_required_slots: List[str] = []
    missing_required_slots: List[str] = []
    softened_required_slots: List[str] = []
    slot_diagnostics: List[Dict[str, Any]] = []
    scene_slots = [
        slot
        for slot in _scene_slots(scene_program)
        if _slot_role(slot) in SUPPORTED_SEMANTIC_ROLES and _slot_priority(slot.get("priority")) != "optional"
    ]
    for slot in sorted(scene_slots, key=_slot_sort_key):
        slot_id = str(slot.get("slot_id") or "").strip()
        if not slot_id:
            continue
        runtime_role = _slot_role(slot)
        requiredness = _slot_requiredness(slot, scene_program=scene_program)
        hard_required = requiredness == "hard"
        status = "covered" if str(slot_asset_map.get(slot_id) or "").strip() else "missing"
        if slot_id in fallback_asset_ids_by_slot:
            status = "fallback_used"
        if not hard_required and status == "missing":
            status = "soft_missing"
        entry = {
            "slot_id": slot_id,
            "concept": canonicalize_semantic_concept(slot.get("concept")) or runtime_role,
            "status": status,
        }
        if runtime_role:
            entry["runtime_role"] = runtime_role
        entry["requiredness"] = requiredness
        slot_diagnostics.append(entry)
        if not hard_required:
            if status == "soft_missing" and _slot_priority(slot.get("priority")) == "must":
                softened_required_slots.append(slot_id)
            continue
        if status == "missing":
            missing_required_slots.append(slot_id)
        else:
            covered_required_slots.append(slot_id)
    return {
        "covered_required_slots": covered_required_slots,
        "missing_required_slots": missing_required_slots,
        "softened_required_slots": softened_required_slots,
        "slot_diagnostics": slot_diagnostics,
        "selection_coverage_errors": selection_coverage_errors,
    }


def _semantic_selection_payload(
    *,
    selection: Dict[str, Any],
    selected_prompt: str,
    stylekit_id: str,
    pack_ids: List[str],
    chosen_asset_ids: List[str],
    slot_asset_map: Dict[str, str],
    group_assignments: List[Dict[str, Any]],
    scene_program: Dict[str, Any],
    all_assets: List[Dict[str, Any]],
    alternatives: Dict[str, List[str]],
    rationale: List[str],
    confidence: float,
    surface_material_selection: Dict[str, Any],
    coverage: Dict[str, Any],
) -> Dict[str, Any]:
    fallback_asset_ids_by_slot = _normalize_fallback_asset_ids_by_slot(
        selection.get("fallback_asset_ids_by_slot"),
        scene_program=scene_program,
        all_assets=all_assets,
    )
    return {
        "selected_prompt": selected_prompt,
        "stylekit_id": stylekit_id,
        "pack_ids": pack_ids,
        "asset_ids": chosen_asset_ids,
        "slot_asset_map": slot_asset_map,
        "group_assignments": group_assignments,
        "fallback_asset_ids_by_slot": fallback_asset_ids_by_slot,
        "rejected_candidate_ids": _normalize_rejected_candidate_ids(selection.get("rejected_candidate_ids"), all_assets),
        "rejected_candidates_by_slot": _normalize_rejected_candidates_by_slot(
            selection.get("rejected_candidates_by_slot"),
            scene_program=scene_program,
            all_assets=all_assets,
        ),
        "optional_additions": _normalize_optional_additions(selection.get("optional_additions"), all_assets, scene_program),
        "alternatives": alternatives,
        "rationale": rationale,
        "confidence": confidence,
        "unknown_asset_ids": [],
        "covered_required_slots": list(coverage.get("covered_required_slots") or []),
        "missing_required_slots": list(coverage.get("missing_required_slots") or []),
        "softened_required_slots": list(coverage.get("softened_required_slots") or []),
        "slot_diagnostics": list(coverage.get("slot_diagnostics") or []),
        "decor_plan": normalize_model_decor_plan(
            selection.get("decor_plan"),
            scene_program=scene_program,
            candidate_assets=all_assets,
        ),
        "surface_material_selection": surface_material_selection,
    }


def validate_semantic_plan(  # validates an LLM selection response: checks stylekit, pack_ids, asset coverage, and budgets
    llm_plan: Dict[str, Any],
    *,
    all_assets: List[Dict[str, Any]],
    allowed_stylekit_ids: List[str],
    allowed_pack_ids: List[str],
    default_budgets: Dict[str, int],
    prompt_text: str,
    placement_intent: Dict[str, Any] | None = None,
    surface_material_candidates: Dict[str, List[Dict[str, Any]]] | None = None,
) -> Dict[str, Any]:
    if not isinstance(llm_plan, dict):
        return _selection_error(
            message="Semantic planner response was not an object.",
            errors=[{"path": "$.llm", "message": "Semantic planner response was not an object."}],
        )

    raw_intent = llm_plan.get("intent")
    validated_intent = validate_semantic_intent(
        {
            "intent": raw_intent,
            "placement_intent": placement_intent,
            "design_brief": llm_plan.get("design_brief"),
        },
        prompt_text=prompt_text,
    )
    if not validated_intent.get("ok"):
        return _selection_error(
            message="Semantic planner response included an invalid intent.",
            scene_program=validated_intent.get("scene_program"),
            intent_spec=validated_intent.get("intent_spec"),
            placement_intent=validated_intent.get("placement_intent"),
            errors=list(validated_intent.get("errors") or []),
        )

    base_scene_program = validated_intent["scene_program"]
    intent_spec = validated_intent["intent_spec"]
    effective_placement_intent = validated_intent["placement_intent"]
    selection = llm_plan.get("selection") if isinstance(llm_plan.get("selection"), dict) else llm_plan
    if not isinstance(selection, dict):
        return _selection_error(
            message="Semantic planner response is missing a selection object.",
            scene_program=base_scene_program,
            intent_spec=intent_spec,
            placement_intent=effective_placement_intent,
            errors=[{"path": "$.llm.selection", "message": "Semantic planner response is missing a selection object."}],
        )

    stylekit_id = str(selection.get("stylekit_id") or "").strip()
    if not stylekit_id or stylekit_id not in allowed_stylekit_ids:
        return _selection_error(
            message="Semantic planner must choose a valid approved stylekit.",
            scene_program=base_scene_program,
            intent_spec=intent_spec,
            placement_intent=effective_placement_intent,
            errors=[{
                "path": "$.llm.selection.stylekit_id",
                "message": f"Semantic planner must choose one approved stylekit id from: {', '.join(allowed_stylekit_ids)}.",
            }],
        )

    pack_ids = _normalize_pack_ids(selection.get("pack_ids"), allowed_pack_ids)
    scene_program, slot_asset_map = _ground_selection_slots(
        selection,
        scene_program=base_scene_program,
        all_assets=all_assets,
    )
    intent_spec = scene_program_to_intent_spec(scene_program)
    group_assignments, group_assignment_errors = _normalized_group_assignments(
        selection,
        scene_program=scene_program,
        slot_asset_map=slot_asset_map,
        all_assets=all_assets,
    )

    chosen_assets, selection_coverage_errors = _expanded_assets_from_selection(
        scene_program=scene_program,
        slot_asset_map=slot_asset_map,
        group_assignments=group_assignments,
        all_assets=all_assets,
        prompt_text=prompt_text,
    )
    chosen_asset_ids = [str(asset.get("asset_id")) for asset in chosen_assets]
    budgets = _normalize_selection_budgets(selection, default_budgets)
    extras = _normalize_selection_extras(
        selection,
        intent_spec.get("confidence", 0.0),
        scene_program=scene_program,
    )
    selected_prompt = str(selection.get("selected_prompt") or prompt_text).strip() or prompt_text.strip()
    raw_selection_attempted = bool(
        (isinstance(selection.get("slot_asset_map"), dict) and selection.get("slot_asset_map"))
        or (isinstance(selection.get("group_assignments"), list) and selection.get("group_assignments"))
    )
    normalized_fallback_asset_ids_by_slot = _normalize_fallback_asset_ids_by_slot(
        selection.get("fallback_asset_ids_by_slot"),
        scene_program=scene_program,
        all_assets=all_assets,
    )

    if not (slot_asset_map or group_assignments):
        if raw_selection_attempted:
            return _selection_error(
                message="Semantic planner selected no valid approved assets.",
                scene_program=scene_program,
                intent_spec=intent_spec,
                placement_intent=effective_placement_intent,
                error_code="semantic_unknown_assets",
                errors=[{"path": "$.llm.selection", "message": "Semantic planner selected no valid approved assets."}],
                extra={"unknown_asset_ids": []},
            )
        return _selection_error(
            message="Semantic planner did not assign assets for the scene roles.",
            scene_program=scene_program,
            intent_spec=intent_spec,
            placement_intent=effective_placement_intent,
            errors=[{"path": "$.llm.selection", "message": "Semantic planner did not assign assets for the scene roles."}],
        )

    if not chosen_assets:
        return _selection_error(
            message="Semantic planner selected no valid approved assets.",
            scene_program=scene_program,
            intent_spec=intent_spec,
            placement_intent=effective_placement_intent,
            error_code="semantic_unknown_assets",
            errors=[{"path": "$.llm.selection", "message": "Semantic planner selected no valid approved assets."}],
            extra={"unknown_asset_ids": []},
        )

    coverage = _selection_coverage(
        scene_program=scene_program,
        slot_asset_map=slot_asset_map,
        fallback_asset_ids_by_slot=normalized_fallback_asset_ids_by_slot,
        selection_coverage_errors=selection_coverage_errors,
        prompt_text=prompt_text,
    )
    if coverage["selection_coverage_errors"] or coverage["missing_required_slots"]:
        missing_tokens = sorted(set(coverage["selection_coverage_errors"] + coverage["missing_required_slots"]))
        return _selection_error(
            message="Semantic planner did not satisfy the required semantic slots.",
            scene_program=scene_program,
            intent_spec=intent_spec,
            placement_intent=effective_placement_intent,
            error_code="semantic_missing_required_slots",
            errors=[
                {"path": "$.llm.selection", "message": "Semantic planner did not satisfy the required semantic slots."},
                {"path": "$.llm.selection.slot_asset_map", "message": f"Missing or underfilled required slots: {', '.join(missing_tokens)}"},
            ] + list(group_assignment_errors or []),
            extra={
                "covered_required_slots": coverage["covered_required_slots"],
                "missing_required_slots": missing_tokens,
                "softened_required_slots": coverage.get("softened_required_slots") or [],
                "slot_diagnostics": coverage["slot_diagnostics"],
                "selected_asset_ids": chosen_asset_ids,
                "slot_asset_map": slot_asset_map,
                "fallback_asset_ids_by_slot": normalized_fallback_asset_ids_by_slot,
            },
        )

    if group_assignment_errors:
        return _selection_error(
            message="Semantic planner returned invalid group assignments.",
            scene_program=scene_program,
            intent_spec=intent_spec,
            placement_intent=effective_placement_intent,
            errors=group_assignment_errors,
        )

    surface_material_selection, surface_material_errors = _normalize_surface_material_selection(
        selection.get("surface_material_selection"),
        surface_material_candidates,
    )
    if surface_material_errors:
        return _selection_error(
            message="Semantic planner returned an invalid shell material selection.",
            scene_program=scene_program,
            intent_spec=intent_spec,
            placement_intent=effective_placement_intent,
            errors=surface_material_errors,
        )

    selection_payload = dict(selection)
    if normalized_fallback_asset_ids_by_slot:
        selection_payload["fallback_asset_ids_by_slot"] = normalized_fallback_asset_ids_by_slot
    semantic_selection = _semantic_selection_payload(
        selection=selection_payload,
        selected_prompt=selected_prompt,
        stylekit_id=stylekit_id,
        pack_ids=pack_ids,
        chosen_asset_ids=chosen_asset_ids,
        slot_asset_map=slot_asset_map,
        group_assignments=group_assignments,
        scene_program=scene_program,
        all_assets=all_assets,
        alternatives=extras["alternatives"],
        rationale=extras["rationale"],
        confidence=extras["confidence"],
        surface_material_selection=surface_material_selection,
        coverage=coverage,
    )

    return {
        "ok": True,
        "scene_program": scene_program,
        "selected_prompt": selected_prompt,
        "stylekit_id": stylekit_id,
        "pack_ids": pack_ids,
        "assets": chosen_assets,
        "asset_ids": chosen_asset_ids,
        "budgets": budgets,
        "has_unknown_asset_ids": False,
        "unknown_asset_ids": [],
        "intent_spec": intent_spec,
        "placement_intent": effective_placement_intent,
        "covered_required_slots": coverage["covered_required_slots"],
        "missing_required_slots": coverage["missing_required_slots"],
        "softened_required_slots": coverage.get("softened_required_slots") or [],
        "slot_diagnostics": coverage["slot_diagnostics"],
        "semantic_selection": semantic_selection,
        "decor_plan": semantic_selection["decor_plan"],
        "surface_material_selection": surface_material_selection,
    }
