from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List

from src.placement.geometry import (
    canonicalize_semantic_concept,
    canonicalize_semantic_role,
    map_semantic_concept_to_runtime_role,
)
from src.planning.archetype_policy import SUPPORTED_ARCHETYPES
from src.planning.scene_types import SUPPORTED_SEMANTIC_ROLES, SemanticSlotSpec
from src.planning.scene_program_policy import policy_set

ALLOWED_SLOT_NECESSITIES = policy_set("allowed_slot_necessities")
ALLOWED_SLOT_SOURCES = policy_set("allowed_slot_sources")

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


def _normalize_descriptor_list(values: Any) -> List[str]:  # processes stylistic adjectives that the LLM uses for matching
    if not isinstance(values, list):
        return []
    return _dedupe_strings(
        str(value).strip().lower()
        for value in values
        if isinstance(value, str) and str(value).strip()
    )


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


def _supported_archetypes() -> set[str]:  # proxy to fetch archetype policy enums
    return set(SUPPORTED_ARCHETYPES)


def _normalize_archetype(value: Any) -> str:  # resolves and validates the primary room archetype identifier
    token = _normalize_feature_token(value)
    return token if token in _supported_archetypes() else ""


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


def _normalize_scene_choice(value: Any, allowed: set[str], fallback: str = "") -> str:
    token = _normalize_feature_token(value)
    if token in allowed:
        return token
    return fallback

