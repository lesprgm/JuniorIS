from __future__ import annotations

from typing import Any, Dict, List

from src.placement.geometry import canonicalize_semantic_concept, canonicalize_semantic_role
from src.planning.scene_types import SUPPORTED_SEMANTIC_ROLES
from src.runtime.decor_plan import normalize_model_decor_plan
from src.planning.scene_program_common import (
    _scene_slots,
    _slot_asset_map_to_group_assignments,
    _slot_priority,
    _slot_requiredness,
    _slot_role,
    _slot_sort_key,
)
from src.planning.scene_program_policy import policy_tuple

ALLOWED_BUDGET_KEYS = policy_tuple("allowed_budget_keys")
from src.planning.scene_program_grounding import ground_scene_program
from src.planning.scene_program_selection_assets import (
    _normalize_fallback_asset_ids_by_slot,
    _normalize_group_assignments,
    _normalize_optional_additions,
    _normalize_rejected_candidate_ids,
    _normalize_rejected_candidates_by_slot,
    _normalize_slot_asset_map,
    _rescue_missing_hard_slots,
)

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
) -> tuple[Dict[str, Any], Dict[str, str], Dict[str, List[str]], List[str]]:
    raw_slot_asset_map = selection.get("slot_asset_map") if isinstance(selection.get("slot_asset_map"), dict) else {}
    slot_asset_map = _normalize_slot_asset_map(
        raw_slot_asset_map,
        scene_program=scene_program,
        all_assets=all_assets,
    )
    fallback_asset_ids_by_slot = _normalize_fallback_asset_ids_by_slot(
        selection.get("fallback_asset_ids_by_slot"),
        scene_program=scene_program,
        all_assets=all_assets,
    )
    slot_asset_map, fallback_asset_ids_by_slot, softened_rescue_slot_ids = _rescue_missing_hard_slots(
        scene_program=scene_program,
        slot_asset_map=slot_asset_map,
        fallback_asset_ids_by_slot=fallback_asset_ids_by_slot,
        all_assets=all_assets,
        blocked_slot_ids={str(slot_id or "").strip() for slot_id in raw_slot_asset_map if str(slot_id or "").strip() not in slot_asset_map},
    )
    grounded_scene_program = ground_scene_program(
        scene_program,
        slot_asset_map=slot_asset_map,
        all_assets=all_assets,
    )
    return grounded_scene_program, slot_asset_map, fallback_asset_ids_by_slot, softened_rescue_slot_ids


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
    softened_rescue_slot_ids: List[str] | None = None,
    prompt_text: str = "",
) -> Dict[str, Any]:
    covered_required_slots: List[str] = []
    missing_required_slots: List[str] = []
    softened_required_slots: List[str] = []
    slot_diagnostics: List[Dict[str, Any]] = []
    softened_slot_ids = set(softened_rescue_slot_ids or [])
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
        hard_required = requiredness == "hard" and slot_id not in softened_slot_ids
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
        entry["requiredness"] = "soft" if slot_id in softened_slot_ids else requiredness
        slot_diagnostics.append(entry)
        if not hard_required:
            if slot_id in softened_slot_ids or (status == "soft_missing" and _slot_priority(slot.get("priority")) == "must"):
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
