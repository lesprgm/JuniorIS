from __future__ import annotations

from typing import Any, Dict, Iterable, List

from src.placement.geometry import semantic_role_key
from src.placement.semantic_taxonomy import (
    decor_allowed_anchors,
    decor_anchor_aliases,
    decor_kind_rules,
    decor_kinds,
)
from src.planning.scene_policy import asset_allowed_by_scene_policy

DECOR_KIND_REGISTRY = decor_kinds()  # runtime-recognized decor buckets, enabled only when asset metadata supports them
ALLOWED_DECOR_ANCHORS = decor_allowed_anchors()  # valid placement zones for decorations
MAX_DECOR_COUNT = 4  # cap per decor entry to prevent visual clutter
DECOR_ANCHOR_ALIASES = decor_anchor_aliases()


def _normalized_mapping(value: Dict[str, Any] | None) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _scene_archetype(scene_program: Dict[str, Any]) -> str:
    archetype = str(scene_program.get("archetype") or "").strip().lower()
    return archetype or "generic_room"


def _style_tags(scene_program: Dict[str, Any]) -> List[str]:
    style_cues = scene_program.get("style_cues") if isinstance(scene_program.get("style_cues"), dict) else {}
    values = style_cues.get("style_tags") if isinstance(style_cues.get("style_tags"), list) else []
    return [str(tag).strip().lower() for tag in values if isinstance(tag, str) and str(tag).strip()]


def _string_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip().lower() for value in values if isinstance(value, str) and str(value).strip()]


def _asset_allowed_for_decor_context(asset: Dict[str, Any], scene_context: Dict[str, Any] | None) -> bool:
    return asset_allowed_by_scene_policy(asset, scene_context=scene_context)


def _asset_decor_kinds(asset: Dict[str, Any]) -> set[str]:
    role = semantic_role_key(asset)
    metadata = {
        "roles": {role},
        "allowed_anchors": set(_string_list(asset.get("allowed_anchors"))),
        "placement_modes": set(_string_list(asset.get("placement_modes"))),
        "usable_roles": set(_string_list(asset.get("usable_roles"))),
        "support_surface_types": set(_string_list(asset.get("support_surface_types"))),
    }
    kinds: set[str] = set()
    for kind, rule in decor_kind_rules().items():
        for key, accepted_values in rule.items():
            if metadata.get(key, set()) & set(accepted_values):
                kinds.add(kind)
                break
    return kinds


def build_decor_capabilities(candidate_assets: Iterable[Dict[str, Any]] | None) -> Dict[str, List[str]]:
    allowed_kinds: set[str] = set()
    for asset in candidate_assets or []:
        if isinstance(asset, dict):
            allowed_kinds.update(_asset_decor_kinds(asset))
    return {
        "allowed_decor_kinds": [kind for kind in DECOR_KIND_REGISTRY if kind in allowed_kinds],
        "allowed_decor_anchors": sorted(ALLOWED_DECOR_ANCHORS),
    }


def build_decor_asset_ids_by_kind(
    candidate_assets: Iterable[Dict[str, Any]] | None,
    *,
    scene_context: Dict[str, Any] | None = None,
) -> Dict[str, List[str]]:
    by_kind: Dict[str, List[str]] = {kind: [] for kind in DECOR_KIND_REGISTRY}
    seen: Dict[str, set[str]] = {kind: set() for kind in DECOR_KIND_REGISTRY}
    for asset in candidate_assets or []:
        if not isinstance(asset, dict):
            continue
        if not _asset_allowed_for_decor_context(asset, scene_context):
            continue
        asset_id = str(asset.get("asset_id") or "").strip()
        if not asset_id:
            continue
        for kind in _asset_decor_kinds(asset):
            if asset_id in seen[kind]:
                continue
            seen[kind].add(asset_id)
            by_kind[kind].append(asset_id)
    return {kind: asset_ids for kind, asset_ids in by_kind.items() if asset_ids}
def build_runtime_scene_context(  # assembles the scene context dict that the decor planner and Unity runtime consume
    *,
    intent_spec: Dict[str, Any] | None,
    placement_intent: Dict[str, Any] | None,
    selected_assets: Iterable[Dict[str, Any]],
    scene_program: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    del intent_spec
    normalized_scene_program = _normalized_mapping(scene_program)
    normalized_placement_intent = _normalized_mapping(placement_intent)
    archetype = _scene_archetype(normalized_scene_program)
    scene_type = str(normalized_scene_program.get("scene_type") or archetype).strip().lower() or archetype
    context = {
        "archetype": archetype,
        "scene_type": scene_type,
        "concept_label": str(normalized_scene_program.get("concept_label") or scene_type).strip().lower() or scene_type,
        "source_prompt": str(normalized_scene_program.get("source_prompt") or "").strip().lower(),
        "execution_archetype": str(normalized_scene_program.get("execution_archetype") or archetype).strip().lower() or archetype,
        "focal_wall": str(normalized_scene_program.get("focal_wall") or "none").strip().lower() or "none",
        "negative_constraints": _string_list(normalized_scene_program.get("negative_constraints")),
        "scene_features": _string_list(normalized_scene_program.get("scene_features")),
        "style_tags": _style_tags(normalized_scene_program),
        "creative_tags": _string_list(normalized_scene_program.get("creative_tags")),
        "mood_tags": _string_list(normalized_scene_program.get("mood_tags")),
        "style_descriptors": _string_list(normalized_scene_program.get("style_descriptors")),
        "semantic_slots": [
            dict(slot)
            for slot in normalized_scene_program.get("semantic_slots") or []
            if isinstance(slot, dict)
        ],
        "grounded_slots": [
            dict(slot)
            for slot in normalized_scene_program.get("grounded_slots") or []
            if isinstance(slot, dict)
        ],
        "density_profile": str(normalized_placement_intent.get("density_profile") or "normal").strip().lower() or "normal",
        "layout_mood": str(normalized_placement_intent.get("layout_mood") or "cozy").strip().lower() or "cozy",
        **build_decor_capabilities(selected_assets),
        "zones": [],
    }
    context["decor_asset_ids_by_kind"] = build_decor_asset_ids_by_kind(selected_assets, scene_context=context)
    return context


def _normalized_decor_rationale(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if isinstance(item, str) and str(item).strip()]


def _normalize_decor_anchor(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token in ALLOWED_DECOR_ANCHORS:
        return token
    return DECOR_ANCHOR_ALIASES.get(token, "")


def _functional_coverage_ready(scene_program: Dict[str, Any] | None) -> bool:
    if not isinstance(scene_program, dict):
        return True
    semantic_slots = [slot for slot in scene_program.get("semantic_slots") or [] if isinstance(slot, dict)]
    grounded_slots = [slot for slot in scene_program.get("grounded_slots") or [] if isinstance(slot, dict)]
    if not semantic_slots:
        return True
    required_slot_ids = {
        str(slot.get("slot_id") or "").strip()
        for slot in semantic_slots
        if str(slot.get("priority") or "should").strip().lower() != "optional"
    }
    grounded_slot_ids = {str(slot.get("slot_id") or "").strip() for slot in grounded_slots}
    return required_slot_ids.issubset(grounded_slot_ids)


def can_enable_decor(scene_program: Dict[str, Any] | None, grounded_slots: List[Dict[str, Any]] | None = None) -> bool:
    if isinstance(scene_program, dict) and grounded_slots is not None:
        scene_program = dict(scene_program)
        scene_program["grounded_slots"] = grounded_slots
    return _functional_coverage_ready(scene_program)


def _normalized_scene_context(
    scene_context: Dict[str, Any] | None,
    *,
    scene_program: Dict[str, Any] | None,
    candidate_assets: Iterable[Dict[str, Any]] | None,
) -> Dict[str, Any]:
    normalized = _normalized_mapping(scene_context) if isinstance(scene_context, dict) else build_runtime_scene_context(
        intent_spec=None,
        placement_intent=None,
        selected_assets=candidate_assets or [],
        scene_program=scene_program,
    )
    if candidate_assets:
        decor_capabilities = build_decor_capabilities(candidate_assets)
        for key, value in decor_capabilities.items():
            existing = normalized.get(key)
            if not isinstance(existing, list) or not existing:
                normalized[key] = value
        if not isinstance(normalized.get("decor_asset_ids_by_kind"), dict) or not normalized.get("decor_asset_ids_by_kind"):
            normalized["decor_asset_ids_by_kind"] = build_decor_asset_ids_by_kind(candidate_assets, scene_context=normalized)
    return normalized


def _scene_program_policy(scene_program: Dict[str, Any] | None) -> Dict[str, Any]:
    if isinstance(scene_program, dict) and isinstance(scene_program.get("optional_addition_policy"), dict):
        return dict(scene_program.get("optional_addition_policy") or {})
    return {}


def _allowed_decor_asset_ids_by_kind(scene_context: Dict[str, Any]) -> Dict[str, set[str]]:
    return {
        str(kind).strip().lower(): {
            str(asset_id).strip()
            for asset_id in asset_ids
            if isinstance(asset_id, str) and str(asset_id).strip()
        }
        for kind, asset_ids in dict(scene_context.get("decor_asset_ids_by_kind") or {}).items()
        if isinstance(kind, str) and isinstance(asset_ids, list)
    }


def _normalize_decor_entry(
    entry: Dict[str, Any],
    *,
    allowed_decor_kinds: set[str],
    allowed_decor_anchors: set[str],
    decor_asset_ids_by_kind: Dict[str, set[str]],
    scene_cap: int,
) -> Dict[str, Any] | None:
    kind = str(entry.get("kind") or "").strip().lower()
    anchor = _normalize_decor_anchor(entry.get("anchor"))
    zone_id = str(entry.get("zone_id") or "").strip().lower()
    asset_id = str(entry.get("asset_id") or "").strip()
    placement_hint = str(entry.get("placement_hint") or "").strip().lower()
    count = entry.get("count")
    if kind not in allowed_decor_kinds or anchor not in allowed_decor_anchors:
        return None
    if asset_id and asset_id not in decor_asset_ids_by_kind.get(kind, set()):
        return None
    if placement_hint and placement_hint not in DECOR_ANCHOR_ALIASES and placement_hint not in {"wall_centered", "wall_left", "wall_right"}:
        return None
    if not isinstance(count, (int, float)):
        return None
    normalized_entry = {
        "kind": kind,
        "anchor": anchor,
        "zone_id": zone_id,
        "count": max(1, min(int(count), max(1, scene_cap))),
    }
    if asset_id:
        normalized_entry["asset_id"] = asset_id
    if placement_hint:
        normalized_entry["placement_hint"] = placement_hint
    return normalized_entry


def normalize_model_decor_plan(  # validates and normalizes LLM-produced decor entries against allowed kinds and anchors
    selection_decor_plan: Dict[str, Any] | None,
    *,
    scene_context: Dict[str, Any] | None = None,
    scene_program: Dict[str, Any] | None = None,
    candidate_assets: Iterable[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    raw_plan = selection_decor_plan if isinstance(selection_decor_plan, dict) else {}
    normalized_scene_context = _normalized_scene_context(
        scene_context,
        scene_program=scene_program,
        candidate_assets=candidate_assets,
    )
    scene_program_policy = _scene_program_policy(scene_program)
    if scene_program_policy.get("allow_optional_additions") is False or not can_enable_decor(scene_program):
        return {
            "archetype": str(normalized_scene_context.get("archetype") or "generic_room").strip().lower() or "generic_room",
            "entries": [],
            "rationale": _normalized_decor_rationale(raw_plan.get("rationale")),
        }
    archetype = str(normalized_scene_context.get("archetype") or "generic_room").strip().lower() or "generic_room"
    allowed_decor_kinds = {
        str(kind).strip().lower()
        for kind in normalized_scene_context.get("allowed_decor_kinds", [])
        if isinstance(kind, str) and str(kind).strip()
    }
    decor_asset_ids_by_kind = _allowed_decor_asset_ids_by_kind(normalized_scene_context)
    allowed_decor_anchors = {
        str(anchor).strip().lower()
        for anchor in normalized_scene_context.get("allowed_decor_anchors", [])
        if isinstance(anchor, str) and str(anchor).strip()
    } or set(ALLOWED_DECOR_ANCHORS)
    max_count = scene_program_policy.get("max_count")
    scene_cap = int(max_count) if isinstance(max_count, (int, float)) and int(max_count) >= 0 else MAX_DECOR_COUNT
    entries_raw = raw_plan.get("entries")
    entries: List[Dict[str, Any]] = []
    if isinstance(entries_raw, list):
        for entry in entries_raw:
            if not isinstance(entry, dict):
                continue
            normalized_entry = _normalize_decor_entry(
                entry,
                allowed_decor_kinds=allowed_decor_kinds,
                allowed_decor_anchors=allowed_decor_anchors,
                decor_asset_ids_by_kind=decor_asset_ids_by_kind,
                scene_cap=scene_cap,
            )
            if normalized_entry is not None:
                entries.append(normalized_entry)

    return {
        "archetype": archetype,
        "entries": entries,
        "rationale": _normalized_decor_rationale(raw_plan.get("rationale")),
    }


def build_runtime_decor_plan(  # public entry-point: delegates to normalize_model_decor_plan with proper scene context
    *,
    intent_spec: Dict[str, Any] | None,
    placement_intent: Dict[str, Any] | None,
    selected_assets: Iterable[Dict[str, Any]],
    scene_context: Dict[str, Any] | None = None,
    scene_program: Dict[str, Any] | None = None,
    selection_decor_plan: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    del intent_spec, placement_intent
    return normalize_model_decor_plan(
        selection_decor_plan,
        scene_context=scene_context,
        scene_program=scene_program,
        candidate_assets=selected_assets,
    )
