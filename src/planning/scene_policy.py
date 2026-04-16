from __future__ import annotations

from typing import Any, Dict, Iterable

from src.placement.geometry import canonicalize_semantic_concept, canonicalize_semantic_role
from src.placement.semantic_taxonomy import scene_allows_policy, scene_policy_tokens, tokens_match_scene_policy


def split_policy_tokens(value: Any) -> set[str]:
    text = str(value or "").strip().lower().replace("/", " ").replace("_", " ").replace("-", " ")
    return {part for part in text.split() if part}


def string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip().lower() for value in values if isinstance(value, str) and str(value).strip()]


def normalized_policy_token(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")
    return "_".join(part for part in token.split("_") if part)


def negative_policy_tokens(source: Dict[str, Any] | None) -> set[str]:
    if not isinstance(source, dict):
        return set()
    tokens: set[str] = set()
    for value in string_list(source.get("negative_constraints")):
        normalized = normalized_policy_token(value)
        tokens.add(normalized)
        tokens.update(part for part in normalized.replace("no_", "").split("_") if part)
    return tokens


def scene_policy_context_tokens(
    source: Dict[str, Any] | None,
    *,
    prompt_text: str = "",
    include_layout_tokens: bool = False,
) -> set[str]:
    if not isinstance(source, dict):
        source = {}
    keys = ["archetype", "scene_type", "concept_label", "execution_archetype", "intended_use", "source_prompt"]
    if include_layout_tokens:
        keys.extend(["density_profile", "layout_mood"])
    tokens: set[str] = set()
    for key in keys:
        tokens.update(split_policy_tokens(source.get(key)))
    tokens.update(split_policy_tokens(prompt_text))
    for key in ("scene_features", "style_tags", "creative_tags", "mood_tags", "style_descriptors"):
        for value in string_list(source.get(key)):
            tokens.add(value)
            tokens.update(split_policy_tokens(value))
    style_cues = source.get("style_cues") if isinstance(source.get("style_cues"), dict) else {}
    for key in ("style_tags", "mood_tags"):
        for value in string_list(style_cues.get(key)):
            tokens.add(value)
            tokens.update(split_policy_tokens(value))
    return tokens


def asset_policy_tokens(asset: Dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for key in ("asset_id", "label", "display_name", "room_role_subtype"):
        tokens.update(split_policy_tokens(asset.get(key)))
    for key in ("tags", "usable_roles", "room_affinities", "style_tags", "support_surface_types"):
        for value in string_list(asset.get(key)):
            tokens.add(value)
            tokens.update(split_policy_tokens(value))
    return tokens


def slot_policy_tokens(slot: Dict[str, Any] | None) -> set[str]:
    if not isinstance(slot, dict):
        return set()
    tokens = {
        canonicalize_semantic_concept(slot.get("concept")),
        canonicalize_semantic_concept(slot.get("subtype")),
        canonicalize_semantic_role(slot.get("runtime_role")),
        canonicalize_semantic_role(slot.get("runtime_role_hint")),
    }
    return {token for token in tokens if token}


def asset_allowed_by_scene_policy(
    asset: Dict[str, Any],
    *,
    scene_context: Dict[str, Any] | None,
    prompt_text: str = "",
    slot: Dict[str, Any] | None = None,
) -> bool:
    scene_tokens = scene_policy_context_tokens(scene_context, prompt_text=prompt_text, include_layout_tokens=True)
    negative_tokens = negative_policy_tokens(scene_context)
    asset_tokens = asset_policy_tokens(asset)

    if tokens_match_scene_policy("bathroom_assets", asset_tokens):
        if not scene_allows_policy("bathroom_assets", scene_tokens, negative_tokens):
            return False
        blocked_slot_tokens = scene_policy_tokens("bathroom_assets", "disallow_slot_tokens")
        if blocked_slot_tokens and slot_policy_tokens(slot) & blocked_slot_tokens:
            return False
    return True
