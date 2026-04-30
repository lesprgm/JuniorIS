from __future__ import annotations

from typing import Any, Dict, Iterable, List

from src.placement.constants import (
    MIN_SEMANTIC_CONFIDENCE,
    SHORTLIST_DEFAULT_LIMIT,
    SHORTLIST_ROLE_COVERAGE_LIMIT,
)
from src.placement.geometry import (
    canonicalize_semantic_concept,
    canonicalize_semantic_role,
    map_semantic_concept_to_runtime_role,
    semantic_role_key,
)
from src.placement.semantic_taxonomy import expand_semantic_aliases
from src.planning.scene_program_common import _derive_role_fields_from_slots
from src.planning.scene_policy import asset_allowed_by_scene_policy, negative_policy_tokens
from src.runtime.realization_registry import (
    resolve_target_height_meters,
    resolve_target_height_ratio_bounds,
)

def _normalize_tokens(values: Iterable[Any]) -> List[str]:  # lowercases and deduplicates arbitrary keyword lists
    normalized: List[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        token = str(value).strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return normalized


def _asset_height_meters(asset: Dict[str, Any]) -> float | None:  # extracts numeric height from asset bounds safely
    bounds = asset.get("bounds") if isinstance(asset.get("bounds"), dict) else None
    size = bounds.get("size") if isinstance(bounds, dict) and isinstance(bounds.get("size"), dict) else None
    height = size.get("y") if isinstance(size, dict) else None
    if isinstance(height, (int, float)) and float(height) > 0.0:
        return float(height)
    return None


def _size_is_plausible_for_role(asset: Dict[str, Any], role: str) -> bool:  # filters out tiny meshes masquerading as major furniture and vice versa
    height = _asset_height_meters(asset)
    if height is None:
        return True
    target_height = resolve_target_height_meters(asset, role)
    if target_height <= 0.0:
        return True
    min_ratio, max_ratio = resolve_target_height_ratio_bounds(asset, role)
    ratio = height / target_height
    return min_ratio <= ratio <= max_ratio


def _asset_text(asset: Dict[str, Any], key: str) -> str:  # safe extraction of basic string metadata
    return str(asset.get(key) or "").strip().lower()


def _split_text_tokens(value: str) -> List[str]:
    return [part for part in value.replace("-", " ").replace("_", " ").split() if part]


def _scene_creative_tokens(scene_program: Dict[str, Any] | None, intent_spec: Dict[str, Any] | None) -> set[str]:  # collects all creative/mood/style tokens from the scene context for matching
    source = scene_program if isinstance(scene_program, dict) and scene_program else dict(intent_spec or {})
    tokens: list[str] = []
    tokens.extend(_normalize_tokens(source.get("creative_tags", [])))
    tokens.extend(_normalize_tokens(source.get("mood_tags", [])))
    tokens.extend(_normalize_tokens(source.get("style_descriptors", [])))
    style_cues = source.get("style_cues") if isinstance(source.get("style_cues"), dict) else {}
    tokens.extend(_normalize_tokens(style_cues.get("style_tags", [])))
    tokens.extend(_normalize_tokens(style_cues.get("mood_tags", [])))
    tokens.extend(_normalize_tokens(source.get("scene_features", [])))
    design_brief = source.get("design_brief") if isinstance(source.get("design_brief"), dict) else {}
    for key in (
        "palette_strategy",
        "signature_moment",
        "visual_weight_distribution",
        "texture_profile",
        "luxury_signal_level",
        "concept_statement",
    ):
        tokens.extend(_split_text_tokens(_asset_text(design_brief, key)))
    for key in ("scene_type", "concept_label"):
        value = _asset_text(source, key)
        if value:
            tokens.extend(_split_text_tokens(value))
    intended_use = _asset_text(source, "intended_use")
    if intended_use:
        tokens.extend(_split_text_tokens(intended_use))
    return set(token for token in tokens if token)


def _asset_creative_tokens(asset: Dict[str, Any]) -> set[str]:  # collects all style/color/affinity tokens from an asset for matching
    tokens = set(_normalize_tokens(asset.get("style_tags", [])))
    tokens.update(_normalize_tokens(asset.get("color_tags", [])))
    tokens.update(_normalize_tokens(asset.get("room_affinities", [])))
    tokens.update(_normalize_tokens(asset.get("usable_roles", [])))
    for key in ("label", "room_role_subtype"):
        value = _asset_text(asset, key)
        if value:
            tokens.update(_split_text_tokens(value))
    for tag in asset.get("tags") or []:
        if isinstance(tag, str) and tag.strip():
            tokens.update(_split_text_tokens(tag.strip().lower()))
    return tokens


def _scene_affinity_tokens(scene_program: Dict[str, Any] | None, intent_spec: Dict[str, Any] | None) -> set[str]:
    tokens = set(_scene_creative_tokens(scene_program, intent_spec))
    source = scene_program if isinstance(scene_program, dict) and scene_program else dict(intent_spec or {})
    style_cues = source.get("style_cues") if isinstance(source.get("style_cues"), dict) else {}
    tokens.update(_normalize_tokens(style_cues.get("style_tags", [])))
    for key in ("execution_archetype", "archetype"):
        value = _asset_text(source, key)
        if value:
            tokens.add(value)
    return tokens


def _scene_scope_tokens(
    scene_program: Dict[str, Any] | None,
    intent_spec: Dict[str, Any] | None,
    prompt_text: str,
) -> set[str]:
    return _scene_affinity_tokens(scene_program, intent_spec) | _prompt_tokens(prompt_text)


def asset_allowed_for_slot(
    asset: Dict[str, Any],
    *,
    scene_program: Dict[str, Any] | None,
    intent_spec: Dict[str, Any] | None,
    prompt_text: str,
    slot: Dict[str, Any] | None = None,
) -> bool:
    del intent_spec
    return asset_allowed_by_scene_policy(asset, scene_context=scene_program, prompt_text=prompt_text, slot=slot)


def _asset_allowed_by_scene_metadata(
    asset: Dict[str, Any],
    *,
    scene_program: Dict[str, Any] | None,
    intent_spec: Dict[str, Any] | None,
    required_roles: set[str],
    prompt_text: str,
) -> bool:
    if not asset_allowed_for_slot(
        asset,
        scene_program=scene_program,
        intent_spec=intent_spec,
        prompt_text=prompt_text,
    ):
        return False
    role = semantic_role_key(asset)
    if role in required_roles:
        return True

    scene_tokens = _scene_affinity_tokens(scene_program, intent_spec)
    negative_tokens = negative_policy_tokens(scene_program if isinstance(scene_program, dict) and scene_program else dict(intent_spec or {}))
    asset_tokens = _asset_creative_tokens(asset)
    asset_negative_affinities = set(_normalize_tokens(asset.get("negative_scene_affinities", [])))

    if asset_negative_affinities & scene_tokens:
        return False
    if negative_tokens & asset_tokens:
        return False
    return True


def filter_candidate_assets(candidate_assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:  # enforces quest-safety, semantic confidence, and size plausibility gates
    if not any(
        any(key in asset for key in ("classification", "quest_compatible", "semantic_confidence"))
        for asset in candidate_assets
    ):
        return list(candidate_assets)

    filtered: List[Dict[str, Any]] = []
    for asset in candidate_assets:
        classification = str(asset.get("classification", "prop")).strip().lower()
        quest_compatible = asset.get("quest_compatible", True)
        semantic_conf = float(asset.get("semantic_confidence", 0.0) or 0.0)
        planner_approved = asset.get("planner_approved") is True
        planner_excluded = asset.get("planner_excluded") is True
        role = semantic_role_key(asset)
        if classification != "prop":
            continue
        if quest_compatible is not True:
            continue
        if semantic_conf < MIN_SEMANTIC_CONFIDENCE:
            continue
        if not planner_approved or planner_excluded:
            continue
        if not _size_is_plausible_for_role(asset, role):
            continue
        filtered.append(asset)
    return filtered


def _normalize_slot(value: Dict[str, Any], fallback_id: str) -> Dict[str, Any]:
    concept = canonicalize_semantic_concept(value.get("concept") or value.get("runtime_role_hint") or fallback_id)
    runtime_role, subtype = map_semantic_concept_to_runtime_role(concept)
    runtime_role = runtime_role or canonicalize_semantic_role(value.get("runtime_role_hint") or concept)
    return {
        "slot_id": str(value.get("slot_id") or fallback_id).strip().lower() or fallback_id,
        "concept": concept,
        "runtime_role": runtime_role,
        "subtype": subtype,
        "priority": str(value.get("priority") or "should").strip().lower() or "should",
        "count": max(1, int(value.get("count") or 1)),
    }


def _requested_slots_view(  # builds a normalized slot-first view of what the LLM must choose
    *,
    intent_spec: Dict[str, Any] | None = None,
    scene_program: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    source = scene_program if isinstance(scene_program, dict) and scene_program else dict(intent_spec or {})
    normalized_slots = [
        _normalize_slot(slot, f"slot_{index + 1}")
        for index, slot in enumerate(source.get("semantic_slots") or [])
        if isinstance(slot, dict)
    ]
    return {
        "semantic_slots": normalized_slots,
    }


def _ordered_selected_assets(  # sorts picked assets to ensure stable reproducible array order
    selected_assets: List[Dict[str, Any]],
    *,
    required_roles: List[str],
    optional_roles: List[str],
) -> List[Dict[str, Any]]:
    requested_order = required_roles + [role for role in optional_roles if role not in required_roles]
    requested_rank = {role: index for index, role in enumerate(requested_order)}
    return sorted(
        selected_assets,
        key=lambda asset: (
            requested_rank.get(semantic_role_key(asset), 999),
            -float(asset.get("semantic_confidence", 0.55) or 0.55),
            str(asset.get("asset_id", "")),
        ),
    )


def _prompt_tokens(prompt_text: str) -> set[str]:
    return set(_split_text_tokens(str(prompt_text or "").lower()))


def _asset_semantic_tokens(asset: Dict[str, Any]) -> set[str]:
    tokens = _asset_creative_tokens(asset)
    tokens.update(_split_text_tokens(str(asset.get("asset_id") or "").lower()))
    role = semantic_role_key(asset)
    if role:
        tokens.add(role)
    subtype = canonicalize_semantic_concept(asset.get("room_role_subtype"))
    if subtype:
        tokens.add(subtype)
    usable_roles = asset.get("usable_roles") if isinstance(asset.get("usable_roles"), list) else []
    for value in usable_roles:
        concept = canonicalize_semantic_concept(value)
        if concept:
            tokens.add(concept)
    tokens.update(expand_semantic_aliases(tokens))
    return tokens


def _slot_family_preference(asset: Dict[str, Any], slot: Dict[str, Any], scene_program: Dict[str, Any] | None) -> int:
    concept = canonicalize_semantic_concept(slot.get("concept"))
    subtype = canonicalize_semantic_concept(slot.get("subtype"))
    role = semantic_role_key(asset)
    asset_subtype = canonicalize_semantic_concept(asset.get("room_role_subtype"))
    label = _asset_text(asset, "label")
    tags = _asset_semantic_tokens(asset)
    density_profile = _asset_text(scene_program or {}, "density_profile") if isinstance(scene_program, dict) else ""

    if concept in {"nightstand", "bedside_surface"} or subtype in {"nightstand", "bedside_surface"}:
        if {"nightstand", "side_table", "bedside_table"} & ({asset_subtype, label} | tags):
            return -3
        if role == "table" and asset_subtype not in {"nightstand", "side_table", "bedside_table"}:
            return 3
        if density_profile == "minimal" and asset_subtype in {"dining_table", "coffee_table", "display_table"}:
            return 2
    if concept in {"dresser", "wardrobe", "closet", "sleep_storage"} or subtype in {"dresser", "wardrobe", "closet"}:
        if {"dresser", "wardrobe", "closet", "storage"} & ({asset_subtype, label} | tags):
            return -2
        if role == "table":
            return 2
    return 0


def _slot_match_score(asset: Dict[str, Any], slot: Dict[str, Any], scene_tokens: set[str], prompt_tokens: set[str], scene_program: Dict[str, Any] | None = None) -> tuple[int, int, int, float, str]:
    asset_tokens = _asset_semantic_tokens(asset)
    concept = slot.get("concept") or ""
    runtime_role = slot.get("runtime_role") or ""
    subtype = slot.get("subtype") or ""
    concept_hit = 1 if concept and concept in asset_tokens else 0
    subtype_hit = 1 if subtype and subtype in asset_tokens else 0
    role_hit = 1 if runtime_role and semantic_role_key(asset) == runtime_role else 0
    style_hits = len((scene_tokens | prompt_tokens) & asset_tokens)
    semantic_conf = float(asset.get("semantic_confidence", 0.55) or 0.55)
    return (
        _slot_family_preference(asset, slot, scene_program),
        -(concept_hit * 4 + subtype_hit * 2 + role_hit),
        -style_hits,
        0 if asset.get("coherence_family_id") else 1,
        -semantic_conf,
        str(asset.get("asset_id", "")),
    )


def _role_match_score(asset: Dict[str, Any], scene_tokens: set[str], prompt_tokens: set[str]) -> tuple[float, int, str]:
    return (
        -float(asset.get("semantic_confidence", 0.55) or 0.55),
        -len((scene_tokens | prompt_tokens) & _asset_semantic_tokens(asset)),
        str(asset.get("asset_id", "")),
    )


def _prompt_match_score(asset: Dict[str, Any], prompt_tokens: set[str], scene_tokens: set[str]) -> tuple[int, float, str]:
    overlap = len(prompt_tokens & _asset_semantic_tokens(asset))
    return (
        -overlap,
        -float(asset.get("semantic_confidence", 0.55) or 0.55),
        str(asset.get("asset_id", "")),
    )


def _append_shortlist_asset(
    shortlist: List[Dict[str, Any]],
    seen_asset_ids: set[str],
    asset: Dict[str, Any],
    *,
    limit: int,
) -> bool:
    asset_id = str(asset.get("asset_id", ""))
    if not asset_id or asset_id in seen_asset_ids:
        return len(shortlist) >= limit
    seen_asset_ids.add(asset_id)
    shortlist.append(asset)
    return len(shortlist) >= limit


def _slot_allows_asset(slot: Dict[str, Any], asset: Dict[str, Any]) -> bool:
    runtime_role = slot.get("runtime_role")
    if not runtime_role:
        return True
    if semantic_role_key(asset) == runtime_role:
        return True
    return bool({slot.get("concept"), slot.get("subtype")} & _asset_semantic_tokens(asset))


def _priority_rank(priority: str) -> int:
    return {"must": 0, "should": 1, "optional": 2}.get(priority, 1)


def _prompt_matched_assets(safe_assets: List[Dict[str, Any]], prompt_tokens: set[str]) -> List[Dict[str, Any]]:
    if not prompt_tokens:
        return []
    return [
        asset
        for asset in safe_assets
        if prompt_tokens & _asset_semantic_tokens(asset)
    ]


def build_semantic_candidate_shortlist(  # filters approved assets and packs a small role-covered shortlist for the selection LLM
    candidate_assets: List[Dict[str, Any]],
    prompt_text: str,
    limit: int = SHORTLIST_DEFAULT_LIMIT,
    *,
    intent_spec: Dict[str, Any] | None = None,
    scene_program: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    safe_assets = filter_candidate_assets(candidate_assets)
    if not safe_assets:
        return []

    requested_slots_view = _requested_slots_view(intent_spec=intent_spec, scene_program=scene_program)
    semantic_slots = requested_slots_view["semantic_slots"]
    required_roles, optional_roles, _ = _derive_role_fields_from_slots(semantic_slots)
    required_role_set = set(required_roles)
    requested_roles = required_roles + [role for role in optional_roles if role not in required_roles]
    scene_filtered_assets = [
        asset
        for asset in safe_assets
        if _asset_allowed_by_scene_metadata(
            asset,
            scene_program=scene_program,
            intent_spec=intent_spec,
            required_roles=required_role_set,
            prompt_text=prompt_text,
        )
    ]
    if scene_filtered_assets:
        safe_assets = scene_filtered_assets

    shortlist: List[Dict[str, Any]] = []
    seen_asset_ids: set[str] = set()
    scene_tokens = _scene_affinity_tokens(scene_program, intent_spec)
    prompt_tokens = _prompt_tokens(prompt_text)

    if not semantic_slots:
        prompt_matched_assets = _prompt_matched_assets(safe_assets, prompt_tokens)
        if prompt_matched_assets:
            safe_assets = prompt_matched_assets

    for slot in sorted(semantic_slots, key=lambda item: (_priority_rank(item.get("priority", "should")), item["slot_id"])):
        ranked = sorted(safe_assets, key=lambda asset: _slot_match_score(asset, slot, scene_tokens, prompt_tokens, scene_program))
        for asset in ranked[: min(SHORTLIST_ROLE_COVERAGE_LIMIT, limit)]:
            if not asset_allowed_for_slot(
                asset,
                scene_program=scene_program,
                intent_spec=intent_spec,
                prompt_text=prompt_text,
                slot=slot,
            ):
                continue
            if not _slot_allows_asset(slot, asset):
                continue
            if _append_shortlist_asset(shortlist, seen_asset_ids, asset, limit=limit):
                return shortlist

    for role in requested_roles:
        role_matches = sorted(
            [asset for asset in safe_assets if semantic_role_key(asset) == role],
            key=lambda asset: _role_match_score(asset, scene_tokens, prompt_tokens),
        )
        for asset in role_matches[: min(SHORTLIST_ROLE_COVERAGE_LIMIT, limit)]:
            if _append_shortlist_asset(shortlist, seen_asset_ids, asset, limit=limit):
                return shortlist

    remaining_assets = sorted(
        safe_assets,
        key=lambda asset: (
            0 if semantic_role_key(asset) in requested_roles else 1,
            *_prompt_match_score(asset, prompt_tokens, scene_tokens),
        ),
    )
    for asset in remaining_assets:
        if _append_shortlist_asset(shortlist, seen_asset_ids, asset, limit=max(1, limit)):
            break

    return shortlist
