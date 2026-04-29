from __future__ import annotations

from typing import Any, Dict, List

from src.placement.geometry import (
    canonicalize_semantic_concept,
    canonicalize_semantic_role,
    map_semantic_concept_to_runtime_role,
)
from src.planning.scene_types import GroundedSlotSpec, SceneProgram
from src.planning.scene_types import SUPPORTED_SEMANTIC_ROLES
from src.planning.scene_program_common import (
    _known_scene_roles,
    _normalize_feature_token,
    _normalize_tokens,
    _slot_necessity,
    _slot_priority,
    _slot_source,
)
from src.planning.scene_program_policy import completion_rules

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


def _rule_matches(rule: Dict[str, Any], *, scene_program: Dict[str, Any], prompt_tokens: set[str]) -> bool:
    slot_concepts = {
        canonicalize_semantic_concept(slot.get("concept"))
        for slot in scene_program.get("semantic_slots") or []
        if isinstance(slot, dict)
    }
    prompt_matches = prompt_tokens & {
        str(token or "").strip().lower()
        for token in rule.get("when_prompt_tokens_any") or []
        if str(token or "").strip()
    }
    slot_matches = slot_concepts & {
        canonicalize_semantic_concept(token)
        for token in rule.get("when_slot_concepts_any") or []
        if canonicalize_semantic_concept(token)
    }
    mode = str(rule.get("when_mode") or "all").strip().lower()
    checks = []
    if rule.get("when_prompt_tokens_any"):
        checks.append(bool(prompt_matches))
    if rule.get("when_slot_concepts_any"):
        checks.append(bool(slot_matches))
    return any(checks) if mode == "any" else all(checks)


def _rule_blocked(rule: Dict[str, Any], *, scene_program: Dict[str, Any], negative_constraints: set[str]) -> bool:
    blocked_negative = negative_constraints & {
        str(token or "").strip().lower()
        for token in rule.get("unless_negative_constraints") or []
        if str(token or "").strip()
    }
    if blocked_negative:
        return True
    blocked_concepts = {
        canonicalize_semantic_concept(token)
        for token in rule.get("unless_slot_concepts") or []
        if canonicalize_semantic_concept(token)
    }
    return bool(blocked_concepts and _slot_exists(scene_program, blocked_concepts))


def _completion_slot(rule: Dict[str, Any]) -> Dict[str, Any] | None:
    slot = rule.get("slot")
    if not isinstance(slot, dict):
        return None
    concept = canonicalize_semantic_concept(slot.get("concept"))
    slot_id = str(slot.get("slot_id") or "").strip()
    runtime_role_hint = canonicalize_semantic_role(slot.get("runtime_role_hint"))
    if not concept or not slot_id or not runtime_role_hint:
        return None
    return {
        "slot_id": slot_id,
        "concept": concept,
        "priority": _slot_priority(slot.get("priority")) or "should",
        "necessity": "support",
        "source": "deterministic_completion",
        "count": 1,
        "runtime_role_hint": runtime_role_hint,
        "rationale": str(slot.get("rationale") or "").strip(),
        "capabilities": [
            str(value).strip().lower()
            for value in slot.get("capabilities") or []
            if isinstance(value, str) and str(value).strip()
        ],
    }


def complete_scene_program(scene_program: SceneProgram, prompt_text: str) -> SceneProgram:
    completed = dict(scene_program)
    semantic_slots = [dict(slot) for slot in scene_program.get("semantic_slots") or [] if isinstance(slot, dict)]
    negative_constraints = set(scene_program.get("negative_constraints") or [])
    prompt_tokens = set(_normalize_tokens(prompt_text))
    current = {"semantic_slots": semantic_slots}

    for rule in completion_rules():
        if not _rule_matches(rule, scene_program=current, prompt_tokens=prompt_tokens):
            continue
        if _rule_blocked(rule, scene_program=current, negative_constraints=negative_constraints):
            continue
        slot = _completion_slot(rule)
        if slot and not _slot_exists(current, {str(slot["concept"])}):
            semantic_slots.append(slot)
            current["semantic_slots"] = semantic_slots

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
