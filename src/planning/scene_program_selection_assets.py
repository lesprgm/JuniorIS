from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from src.catalog.style_material_pool import SURFACE_MATERIAL_SLOTS
from src.placement.geometry import canonicalize_semantic_concept, canonicalize_semantic_role
from src.planning.asset_shortlist import asset_allowed_for_slot
from src.planning.scene_policy import asset_allowed_by_scene_policy
from src.planning.scene_types import SUPPORTED_SEMANTIC_ROLES
from src.planning.scene_program_common import (
    _approved_asset_ids,
    _group_role_slot_ids,
    _group_spec_by_id,
    _is_hard_required_slot,
    _known_slot_ids,
    _normalize_notes,
    _scene_slots,
    _slot_role,
    _slot_sort_key,
)
from src.planning.scene_program_policy import policy_set, policy_tuple

ALLOWED_OPTIONAL_PLACEMENT_HINTS = policy_set("allowed_optional_placement_hints")
REQUIRED_SURFACE_MATERIAL_SLOTS = policy_tuple("required_surface_material_slots")

def _asset_allowed_for_optional_addition(asset: Dict[str, Any], scene_program: Dict[str, Any] | None) -> bool:
    return asset_allowed_by_scene_policy(asset, scene_context=scene_program)


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

