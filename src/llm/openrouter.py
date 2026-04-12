from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from src.llm import transport
from src.placement.geometry import geometry_profile_from_asset, semantic_role_key


PROVIDER_NAME = "OpenRouter"  # human-readable name for error messages
PROVIDER_KEY = "openrouter"  # matches the key in PROVIDER_ADAPTERS map
OPENROUTER_INVALID_RESPONSE = "llm_invalid_response"
OPENROUTER_PARSE_ERROR = "llm_parse_error"
OPENROUTER_TRANSPORT_ERROR = "llm_transport_error"
INVALID_RESPONSE_ERROR_CODE = OPENROUTER_INVALID_RESPONSE


# Keep behavior deterministic so planner/runtime contracts stay stable.
def resolve_provider_settings(user_prefs: Dict[str, Any]) -> tuple[Dict[str, Any] | None, Dict[str, Any] | None]:  # resolves API key and model from user_prefs or environment variables
    api_key = str(
        user_prefs.get("llm_api_key")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPEN_ROUTER_KEY")
        or ""
    ).strip()
    model = str(
        user_prefs.get("llm_model")
        or os.getenv("OPENROUTER_MODEL")
        or os.getenv("OPEN_ROUTER_MODEL")
        or ""
    ).strip()
    if not api_key:
        return None, transport.llm_unavailable(
            "OpenRouter key is not configured. Set OPENROUTER_API_KEY or OPEN_ROUTER_KEY."
        )
    if not model:
        return None, transport.llm_unavailable(
            "OpenRouter model is not configured. Set OPENROUTER_MODEL or OPEN_ROUTER_MODEL."
        )
    settings = transport.resolve_runtime_settings(user_prefs)
    settings.update({"provider": PROVIDER_KEY, "api_key": api_key, "model": model})
    return settings, None


def _parse_json_object(raw_text: str, error_code: str, message: str) -> tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return None, transport.llm_error(error_code, message)
    if not isinstance(parsed, dict):
        return None, transport.llm_error(OPENROUTER_INVALID_RESPONSE, f"{PROVIDER_NAME} response must be a JSON object.")
    return parsed, None


def _extract_text_output(response_payload: Dict[str, Any]) -> Dict[str, Any]:
    output_text = response_payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return {"ok": True, "text": output_text}

    output = response_payload.get("output")
    if not isinstance(output, list) or not output:
        return transport.llm_error(OPENROUTER_INVALID_RESPONSE, f"{PROVIDER_NAME} response is missing output text.")

    text_parts: List[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if str(part.get("type") or "").strip().lower() != "output_text":
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text)

    if not text_parts:
        return transport.llm_error(OPENROUTER_INVALID_RESPONSE, f"{PROVIDER_NAME} response did not include JSON text content.")
    return {"ok": True, "text": "".join(text_parts)}


def request_json(
    *,
    settings: Dict[str, Any],
    system_prompt: str,
    user_payload: Dict[str, Any],
    circuit_key: str,
) -> Dict[str, Any]:
    if transport.is_circuit_open(circuit_key):
        return transport.llm_error("llm_circuit_open", f"{PROVIDER_NAME} planner is temporarily disabled after repeated failures.")

    payload = {
        "model": settings["model"],
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": json.dumps(user_payload, ensure_ascii=True)}],
            },
        ],
        "text": {"format": {"type": "json_object"}},
    }
    reasoning_effort = str(settings.get("reasoning_effort") or "").strip().lower()
    if reasoning_effort in {"minimal", "low", "medium", "high"}:
        payload["reasoning"] = {"effort": reasoning_effort}
    max_output_tokens = int(settings.get("max_output_tokens") or 0)
    if max_output_tokens > 0:
        payload["max_output_tokens"] = max_output_tokens
    http_result = transport.post_json_with_retries(
        url="https://openrouter.ai/api/v1/responses",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings['api_key']}",
        },
        payload=payload,
        timeout_s=settings["timeout_s"],
        retry_count=settings["retry_count"],
        retry_backoff_s=settings["retry_backoff_s"],
        transport_error_code=OPENROUTER_TRANSPORT_ERROR,
        provider_name=PROVIDER_NAME,
    )
    if not http_result.get("ok"):
        transport.record_circuit_failure(
            circuit_key,
            settings["circuit_threshold"],
            settings["circuit_cooldown_s"],
        )
        return http_result

    transport.record_circuit_success(circuit_key)
    parsed, parse_error = _parse_json_object(
        str(http_result["body"]),
        OPENROUTER_PARSE_ERROR,
        f"{PROVIDER_NAME} returned invalid JSON.",
    )
    if parse_error is not None:
        return parse_error

    text_result = _extract_text_output(parsed)
    if text_result.get("ok") is False:
        return text_result

    parsed_content, content_error = _parse_json_object(
        text_result["text"],
        OPENROUTER_PARSE_ERROR,
        f"{PROVIDER_NAME} content was not valid JSON.",
    )
    if content_error is not None:
        return content_error
    return {"ok": True, "payload": parsed_content}


def candidate_asset_payload(candidate_assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for asset in candidate_assets:
        if not asset.get("asset_id"):
            continue
        payload.append(
            {
                "asset_id": asset.get("asset_id"),
                "role": semantic_role_key(asset),
                "label": asset.get("label"),
                "tags": asset.get("tags", []),
                "style_tags": asset.get("style_tags", []),
                "color_tags": asset.get("color_tags", []),
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
                "source_pack": asset.get("source_pack"),
                "perf_tier": asset.get("perf_tier"),
                "quality_tier": asset.get("quality_tier"),
                "front_yaw_offset_degrees": asset.get("front_yaw_offset_degrees"),
                "geometry_profile": geometry_profile_from_asset(asset),
            }
        )
    return payload


def stylekit_payload(stylekit_candidates: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for stylekit in stylekit_candidates or []:
        stylekit_id = stylekit.get("stylekit_id")
        if not stylekit_id:
            continue
        payload.append(
            {
                "stylekit_id": stylekit_id,
                "tags": stylekit.get("tags", []),
                "lighting_preset": stylekit.get("lighting_preset"),
            }
        )
    return payload


def pack_payload(pack_candidates: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for pack in pack_candidates or []:
        pack_id = pack.get("pack_id")
        if not pack_id:
            continue
        payload.append(
            {
                "pack_id": pack_id,
                "tags": pack.get("tags", []),
                "asset_count": pack.get("asset_count"),
            }
        )
    return payload
