from __future__ import annotations

import os
import time
from typing import Any, Dict, List

from src.llm import gemini, openrouter, transport
from src.placement.geometry import geometry_profile_from_asset, semantic_role_key
from src.catalog.style_material_pool import SURFACE_MATERIAL_SLOTS
from src.runtime.decor_plan import build_decor_asset_ids_by_kind, build_decor_capabilities


PROVIDER_ADAPTERS = {
    gemini.PROVIDER_KEY: gemini,
    openrouter.PROVIDER_KEY: openrouter,
}



# Keep behavior deterministic so planner/runtime contracts stay stable.
def extract_plan(payload: Dict[str, Any]) -> Dict[str, Any] | None:  # extracts the plan object from various LLM response shapes
    if isinstance(payload.get("plan"), dict):
        return payload["plan"]
    if any(
        key in payload
        for key in (
            "design_brief",
            "intent",
            "placement_intent",
            "selection",
            "asset_ids",
            "slot_asset_map",
            "rejected_candidates_by_slot",
            "stylekit_id",
            "pack_ids",
            "budgets",
            "decor_plan",
        )
    ):
        return payload
    return None


def provider_circuit_key(provider: str, operation: str) -> str:  # unique circuit breaker key per provider+operation pair
    return f"provider:{provider}:{operation}"


def invalid_content_error(adapter: Any, message: str) -> Dict[str, Any]:
    error_code = getattr(adapter, "INVALID_RESPONSE_ERROR_CODE", "llm_invalid_response")
    return transport.llm_error(error_code, f"{adapter.PROVIDER_NAME} {message}")


def resolve_provider(user_prefs: Dict[str, Any]) -> tuple[Any | None, Dict[str, Any] | None, Dict[str, Any] | None]:  # resolves user preferences to a provider adapter, settings, and optional error
    provider_key = str(
        user_prefs.get("llm_provider")
        or os.getenv("PLANNER_LLM_PROVIDER")
        or "gemini"
    ).strip().lower()
    adapter = PROVIDER_ADAPTERS.get(provider_key)
    if adapter is None:
        return None, None, transport.llm_unavailable(
            f"Unsupported llm_provider='{provider_key}'. Configure an installed adapter."
        )
    settings, config_error = adapter.resolve_provider_settings(user_prefs)
    if config_error is not None:
        return None, None, config_error
    return adapter, settings, None


def inline_plan_payload(user_prefs: Dict[str, Any]) -> Dict[str, Any] | None:  # allows bypassing LLM by inlining a plan directly in user_prefs
    inline_plan = user_prefs.get("llm_plan")
    if isinstance(inline_plan, dict):
        return extract_plan(inline_plan) or inline_plan
    return None


def stage_user_prefs(user_prefs: Dict[str, Any], stage: str) -> Dict[str, Any]:  # allows per-stage (intent/selection) provider/model/key overrides
    stage_prefix = f"llm_{stage}_"
    env_prefix = f"PLANNER_LLM_{stage.upper()}_"
    staged = dict(user_prefs)

    provider = user_prefs.get(f"{stage_prefix}provider") or os.getenv(f"{env_prefix}PROVIDER")
    model = user_prefs.get(f"{stage_prefix}model") or os.getenv(f"{env_prefix}MODEL")
    api_key = user_prefs.get(f"{stage_prefix}api_key") or os.getenv(f"{env_prefix}API_KEY")
    reasoning_effort = (
        user_prefs.get(f"{stage_prefix}reasoning_effort")
        or os.getenv(f"{env_prefix}REASONING_EFFORT")
        or "medium"
    )
    max_output_tokens = user_prefs.get(f"{stage_prefix}max_output_tokens") or os.getenv(f"{env_prefix}MAX_OUTPUT_TOKENS")
    thinking_budget = user_prefs.get(f"{stage_prefix}thinking_budget") or os.getenv(f"{env_prefix}THINKING_BUDGET")
    thinking_level = user_prefs.get(f"{stage_prefix}thinking_level") or os.getenv(f"{env_prefix}THINKING_LEVEL")

    if provider:
        staged["llm_provider"] = provider
    if model:
        staged["llm_model"] = model
    if api_key:
        staged["llm_api_key"] = api_key
    if reasoning_effort:
        staged["llm_reasoning_effort"] = reasoning_effort
    if max_output_tokens:
        staged["llm_max_output_tokens"] = max_output_tokens
    if thinking_budget is not None:
        staged["llm_thinking_budget"] = thinking_budget
    if thinking_level:
        staged["llm_thinking_level"] = thinking_level
    return staged


def selection_payload(  # assembles the full payload sent to the LLM for the selection stage
    *,
    adapter: Any,
    prompt_plan: Dict[str, Any],
    candidate_assets: List[Dict[str, Any]],
    allowed_stylekit_ids: List[str],
    allowed_pack_ids: List[str],
    default_budgets: Dict[str, int],
    intent_spec: Dict[str, Any] | None = None,
    scene_program: Dict[str, Any] | None = None,
    placement_intent: Dict[str, Any] | None = None,
    stylekit_candidates: List[Dict[str, Any]] | None = None,
    pack_candidates: List[Dict[str, Any]] | None = None,
    surface_material_candidates: Dict[str, List[Dict[str, Any]]] | None = None,
    design_brief: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    scene_program = scene_program if isinstance(scene_program, dict) else {}
    candidate_assets_by_role: Dict[str, List[str]] = {}
    for asset in candidate_assets:
        role = semantic_role_key(asset)
        asset_id = str(asset.get("asset_id") or "").strip()
        if not role or not asset_id:
            continue
        bucket = candidate_assets_by_role.setdefault(role, [])
        if asset_id not in bucket:
            bucket.append(asset_id)

    payload = {
        "selected_prompt": prompt_plan.get("selected_prompt"),
        "input_prompt": prompt_plan.get("input_prompt") or prompt_plan.get("selected_prompt"),
        "allowed_stylekit_ids": allowed_stylekit_ids,
        "allowed_pack_ids": allowed_pack_ids,
        "default_budgets": default_budgets,
        "candidate_assets": adapter.candidate_asset_payload(candidate_assets),
        "candidate_assets_by_role": candidate_assets_by_role,
        "stylekit_candidates": adapter.stylekit_payload(stylekit_candidates),
        "pack_candidates": adapter.pack_payload(pack_candidates),
        **build_decor_capabilities(candidate_assets),
        "decor_asset_ids_by_kind": build_decor_asset_ids_by_kind(candidate_assets),
        "surface_material_candidates": {
            surface: list((surface_material_candidates or {}).get(surface) or [])
            for surface in SURFACE_MATERIAL_SLOTS
        },
    }
    if isinstance(intent_spec, dict):
        payload["intent"] = intent_spec
    if isinstance(design_brief, dict) and design_brief:
        payload["design_brief"] = design_brief
    if scene_program:
        payload["scene_program"] = scene_program
    if isinstance(placement_intent, dict):
        payload["placement_intent"] = placement_intent
    return payload


def compact_candidate_asset_payload(candidate_assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:  # strips large fields from assets to fit within LLM context limits
    payload: List[Dict[str, Any]] = []
    for asset in candidate_assets:
        asset_id = str(asset.get("asset_id") or "").strip()
        if not asset_id:
            continue
        geometry = geometry_profile_from_asset(asset)
        payload.append(
            {
                "asset_id": asset_id,
                "role": semantic_role_key(asset),
                "label": asset.get("label"),
                "tags": list(asset.get("tags") or [])[:4],
                "style_tags": list(asset.get("style_tags") or [])[:2],
                "color_tags": list(asset.get("color_tags") or [])[:2],
                "room_role_subtype": asset.get("room_role_subtype"),
                "coherence_family_id": asset.get("coherence_family_id"),
                "collection_id": asset.get("collection_id"),
                "pairing_group": asset.get("pairing_group"),
                "repeat_strategy": asset.get("repeat_strategy"),
                "allowed_anchors": list(asset.get("allowed_anchors") or []),
                "placement_modes": list(asset.get("placement_modes") or []),
                "usable_roles": list(asset.get("usable_roles") or []),
                "scale_class": asset.get("scale_class"),
                "visual_salience": asset.get("visual_salience"),
                "clutter_weight": asset.get("clutter_weight"),
                "room_affinities": list(asset.get("room_affinities") or []),
                "group_role_affinities": list(asset.get("group_role_affinities") or []),
                "supports_group_types": list(asset.get("supports_group_types") or []),
                "support_surface_types": list(asset.get("support_surface_types") or []),
                "negative_scene_affinities": list(asset.get("negative_scene_affinities") or []),
                "repeatable_member_role": asset.get("repeatable_member_role"),
                "seat_front_axis_validated": asset.get("seat_front_axis_validated"),
                "stack_target_roles": list(asset.get("stack_target_roles") or []),
                "front_yaw_offset_degrees": asset.get("front_yaw_offset_degrees"),
                "geometry_profile": {
                    "placement_role": geometry.get("placement_role"),
                    "footprint_radius": geometry.get("footprint_radius"),
                    "preferred_near_distance": geometry.get("preferred_near_distance"),
                },
            }
        )
    return payload


def compact_stylekit_payload(stylekit_candidates: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for stylekit in stylekit_candidates or []:
        stylekit_id = stylekit.get("stylekit_id")
        if not stylekit_id:
            continue
        payload.append(
            {
                "stylekit_id": stylekit_id,
                "tags": list(stylekit.get("tags") or [])[:3],
            }
        )
    return payload


def compact_pack_payload(pack_candidates: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for pack in pack_candidates or []:
        pack_id = pack.get("pack_id")
        if not pack_id:
            continue
        payload.append(
            {
                "pack_id": pack_id,
                "tags": list(pack.get("tags") or [])[:3],
            }
        )
    return payload


def timed_result(start_time: float, payload: Dict[str, Any]) -> Dict[str, Any]:  # attaches latency measurement to an LLM response payload
    result = dict(payload)
    result["latency_ms"] = round((time.monotonic() - start_time) * 1000.0, 1)
    return result
