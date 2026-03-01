from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List


def _extract_plan(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    if isinstance(payload.get("plan"), dict):
        return payload["plan"]
    # Allow direct plan payloads for simple adapters.
    if any(key in payload for key in ("asset_ids", "stylekit_id", "pack_ids", "budgets", "selected_prompt")):
        return payload
    return None


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

    endpoint = str(user_prefs.get("llm_endpoint") or os.getenv("PLANNER_LLM_ENDPOINT") or "").strip()
    if not endpoint:
        return {
            "ok": False,
            "error_code": "llm_unavailable",
            "message": "LLM mode requested but no llm_plan override or endpoint is configured.",
        }

    timeout_s = user_prefs.get("llm_timeout_s", 12)
    if not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
        timeout_s = 12

    payload = {
        "prompt_plan": prompt_plan,
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

    token = str(user_prefs.get("llm_token") or os.getenv("PLANNER_LLM_TOKEN") or "").strip()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "error_code": "llm_http_error",
            "message": f"LLM endpoint returned HTTP {exc.code}.",
        }
    except urllib.error.URLError as exc:
        return {
            "ok": False,
            "error_code": "llm_transport_error",
            "message": f"LLM endpoint transport error: {exc.reason}",
        }
    except Exception:
        return {
            "ok": False,
            "error_code": "llm_transport_error",
            "message": "LLM endpoint request failed unexpectedly.",
        }

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "error_code": "llm_parse_error",
            "message": "LLM endpoint returned invalid JSON.",
        }

    if not isinstance(parsed, dict):
        return {
            "ok": False,
            "error_code": "llm_invalid_response",
            "message": "LLM endpoint response must be a JSON object.",
        }

    plan = _extract_plan(parsed)
    if not isinstance(plan, dict):
        return {
            "ok": False,
            "error_code": "llm_invalid_response",
            "message": "LLM endpoint response does not include a valid plan object.",
        }

    return {"ok": True, "backend": "endpoint", "plan": plan}
