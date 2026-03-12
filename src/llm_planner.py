from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List


_CIRCUIT_STATE: Dict[str, Dict[str, float]] = {}


def _as_positive_float(value: Any, default: float, min_value: float = 0.01, max_value: float = 300.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed < min_value:
        return min_value
    if parsed > max_value:
        return max_value
    return parsed


def _as_bounded_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed < min_value:
        return min_value
    if parsed > max_value:
        return max_value
    return parsed


def _extract_plan(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    if isinstance(payload.get("plan"), dict):
        return payload["plan"]
    # Allow direct plan payloads for simple adapters.
    if any(key in payload for key in ("asset_ids", "stylekit_id", "pack_ids", "budgets", "selected_prompt")):
        return payload
    return None


def _is_circuit_open(key: str) -> bool:
    state = _CIRCUIT_STATE.get(key)
    if not state:
        return False

    opened_until = float(state.get("opened_until", 0.0))
    now = time.monotonic()
    if now < opened_until:
        return True

    # Cooldown elapsed; allow requests again.
    state["opened_until"] = 0.0
    return False


def _record_circuit_success(key: str) -> None:
    state = _CIRCUIT_STATE.setdefault(key, {"failures": 0.0, "opened_until": 0.0})
    state["failures"] = 0.0
    state["opened_until"] = 0.0


def _record_circuit_failure(key: str, threshold: int, cooldown_s: float) -> None:
    state = _CIRCUIT_STATE.setdefault(key, {"failures": 0.0, "opened_until": 0.0})
    failures = int(state.get("failures", 0.0)) + 1
    state["failures"] = float(failures)
    if failures >= threshold:
        state["opened_until"] = time.monotonic() + cooldown_s


def _retry_sleep(backoff_s: float, attempt_index: int) -> None:
    if backoff_s <= 0:
        return
    delay = backoff_s * (2**attempt_index)
    time.sleep(delay)


def _post_json_with_retries(
    *,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout_s: float,
    retry_count: int,
    retry_backoff_s: float,
) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")

    for attempt in range(retry_count + 1):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8")
            return {"ok": True, "body": raw}
        except urllib.error.HTTPError as exc:
            code = int(getattr(exc, "code", 500))
            retryable = code in {408, 409, 429} or 500 <= code <= 599
            if retryable and attempt < retry_count:
                _retry_sleep(retry_backoff_s, attempt)
                continue
            return {
                "ok": False,
                "error_code": "llm_http_error",
                "message": f"Gemini returned HTTP {code}.",
            }
        except urllib.error.URLError as exc:
            if attempt < retry_count:
                _retry_sleep(retry_backoff_s, attempt)
                continue
            return {
                "ok": False,
                "error_code": "llm_transport_error",
                "message": f"Gemini transport error: {exc.reason}",
            }
        except TimeoutError:
            if attempt < retry_count:
                _retry_sleep(retry_backoff_s, attempt)
                continue
            return {
                "ok": False,
                "error_code": "llm_transport_error",
                "message": "Gemini request timed out.",
            }
        except Exception:
            if attempt < retry_count:
                _retry_sleep(retry_backoff_s, attempt)
                continue
            return {
                "ok": False,
                "error_code": "llm_transport_error",
                "message": "Gemini request failed unexpectedly.",
            }

    return {
        "ok": False,
        "error_code": "llm_transport_error",
        "message": "Gemini request exhausted retries.",
    }


def _request_plan_via_gemini(
    *,
    api_key: str,
    model: str,
    prompt_plan: Dict[str, Any],
    candidate_assets: List[Dict[str, Any]],
    allowed_stylekit_ids: List[str],
    allowed_pack_ids: List[str],
    default_budgets: Dict[str, int],
    timeout_s: float,
    retry_count: int,
    retry_backoff_s: float,
    circuit_threshold: int,
    circuit_cooldown_s: float,
) -> Dict[str, Any]:
    circuit_key = "provider:gemini"
    if _is_circuit_open(circuit_key):
        return {
            "ok": False,
            "error_code": "llm_circuit_open",
            "message": "Gemini planner is temporarily disabled after repeated failures.",
        }

    planner_payload = {
        "selected_prompt": prompt_plan.get("selected_prompt"),
        "creative_variants": prompt_plan.get("creative_variants", []),
        "allowed_stylekit_ids": allowed_stylekit_ids,
        "allowed_pack_ids": allowed_pack_ids,
        "default_budgets": default_budgets,
        "candidate_assets": [
            {
                "asset_id": asset.get("asset_id"),
                "label": asset.get("label"),
                "tags": asset.get("tags", []),
                "style_tags": asset.get("style_tags", []),
                "source_pack": asset.get("source_pack"),
                "perf_tier": asset.get("perf_tier"),
                "quality_tier": asset.get("quality_tier"),
            }
            for asset in candidate_assets
            if asset.get("asset_id")
        ],
    }

    system_prompt = (
        "You are a strict world planner for a VR room compiler. "
        "Return only JSON with either a top-level `plan` object or a direct plan object. "
        "Allowed plan keys: selected_prompt (string), stylekit_id (string), pack_ids (string[]), "
        "asset_ids (string[]), budgets (object with max_props/max_texture_tier/max_lights integers). "
        "Use only IDs present in the provided allowed lists and candidate assets."
    )

    request_payload = {
        "system_instruction": {
            "parts": [{"text": system_prompt}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": json.dumps(planner_payload, ensure_ascii=True)},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }

    headers = {"Content-Type": "application/json"}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    http_result = _post_json_with_retries(
        url=url,
        headers=headers,
        payload=request_payload,
        timeout_s=timeout_s,
        retry_count=retry_count,
        retry_backoff_s=retry_backoff_s,
    )
    if not http_result.get("ok"):
        _record_circuit_failure(circuit_key, circuit_threshold, circuit_cooldown_s)
        return http_result

    _record_circuit_success(circuit_key)

    try:
        parsed = json.loads(str(http_result["body"]))
    except json.JSONDecodeError:
        return {
            "ok": False,
            "error_code": "llm_parse_error",
            "message": "Gemini returned invalid JSON.",
        }

    if not isinstance(parsed, dict):
        return {
            "ok": False,
            "error_code": "llm_invalid_response",
            "message": "Gemini response must be a JSON object.",
        }

    candidates = parsed.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return {
            "ok": False,
            "error_code": "llm_invalid_response",
            "message": "Gemini response is missing candidates.",
        }

    content = candidates[0].get("content", {}) if isinstance(candidates[0], dict) else {}
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list) or not parts:
        return {
            "ok": False,
            "error_code": "llm_invalid_response",
            "message": "Gemini response is missing content parts.",
        }

    text = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
    if not text.strip():
        return {
            "ok": False,
            "error_code": "llm_invalid_response",
            "message": "Gemini response did not include JSON text content.",
        }

    try:
        parsed_content = json.loads(text)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "error_code": "llm_parse_error",
            "message": "Gemini content was not valid JSON.",
        }

    if not isinstance(parsed_content, dict):
        return {
            "ok": False,
            "error_code": "llm_invalid_response",
            "message": "Gemini content must be a JSON object.",
        }

    plan = _extract_plan(parsed_content)
    if not isinstance(plan, dict):
        return {
            "ok": False,
            "error_code": "llm_invalid_response",
            "message": "Gemini content does not include a valid plan object.",
        }

    return {"ok": True, "backend": "gemini", "plan": plan}


def request_llm_plan(
    *,
    prompt_plan: Dict[str, Any],
    candidate_assets: List[Dict[str, Any]],
    allowed_stylekit_ids: List[str],
    allowed_pack_ids: List[str],
    default_budgets: Dict[str, int],
    user_prefs: Dict[str, Any],
) -> Dict[str, Any]:
    inline_plan = user_prefs.get("llm_plan")
    if isinstance(inline_plan, dict):
        return {"ok": True, "backend": "inline_override", "plan": inline_plan}

    provider = str(user_prefs.get("llm_provider") or os.getenv("PLANNER_LLM_PROVIDER") or "gemini").strip().lower()
    if provider != "gemini":
        return {
            "ok": False,
            "error_code": "llm_unavailable",
            "message": "Only llm_provider='gemini' is supported.",
        }

    timeout_s = _as_positive_float(
        user_prefs.get("llm_timeout_s", os.getenv("PLANNER_LLM_TIMEOUT_S", 12)),
        12.0,
    )
    retry_count = _as_bounded_int(
        user_prefs.get("llm_retry_count", os.getenv("PLANNER_LLM_RETRIES", 1)),
        default=1,
        min_value=0,
        max_value=5,
    )
    retry_backoff_s = _as_positive_float(
        user_prefs.get("llm_retry_backoff_s", os.getenv("PLANNER_LLM_RETRY_BACKOFF_S", 0.5)),
        0.5,
        min_value=0.0,
        max_value=30.0,
    )
    circuit_threshold = _as_bounded_int(
        user_prefs.get("llm_circuit_failures", os.getenv("PLANNER_LLM_CIRCUIT_FAILURES", 3)),
        default=3,
        min_value=1,
        max_value=20,
    )
    circuit_cooldown_s = _as_positive_float(
        user_prefs.get("llm_circuit_cooldown_s", os.getenv("PLANNER_LLM_CIRCUIT_COOLDOWN_S", 30)),
        30.0,
        min_value=1.0,
        max_value=900.0,
    )

    api_key = str(
        user_prefs.get("llm_api_key")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or ""
    ).strip()
    if not api_key:
        return {
            "ok": False,
            "error_code": "llm_unavailable",
            "message": "GEMINI_API_KEY is not configured.",
        }

    model = str(os.getenv("GEMINI_MODEL") or "").strip()
    if not model:
        return {
            "ok": False,
            "error_code": "llm_unavailable",
            "message": "GEMINI_MODEL is not configured.",
        }

    return _request_plan_via_gemini(
        api_key=api_key,
        model=model,
        prompt_plan=prompt_plan,
        candidate_assets=candidate_assets,
        allowed_stylekit_ids=allowed_stylekit_ids,
        allowed_pack_ids=allowed_pack_ids,
        default_budgets=default_budgets,
        timeout_s=timeout_s,
        retry_count=retry_count,
        retry_backoff_s=retry_backoff_s,
        circuit_threshold=circuit_threshold,
        circuit_cooldown_s=circuit_cooldown_s,
    )
