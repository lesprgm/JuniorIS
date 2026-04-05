from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict


_CIRCUIT_STATE: Dict[str, Dict[str, float]] = {}


def as_positive_float(
    value: Any,
    default: float,
    min_value: float = 0.01,
    max_value: float = 300.0,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed < min_value:
        return min_value
    if parsed > max_value:
        return max_value
    return parsed


def as_bounded_int(
    value: Any,
    default: int,
    min_value: int,
    max_value: int,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed < min_value:
        return min_value
    if parsed > max_value:
        return max_value
    return parsed


def llm_unavailable(message: str) -> Dict[str, Any]:
    return {"ok": False, "error_code": "llm_unavailable", "message": message}


def llm_error(error_code: str, message: str) -> Dict[str, Any]:
    return {"ok": False, "error_code": error_code, "message": message}


def is_circuit_open(key: str) -> bool:
    state = _CIRCUIT_STATE.get(key)
    if not state:
        return False
    opened_until = float(state.get("opened_until", 0.0))
    now = time.monotonic()
    if now < opened_until:
        return True
    state["opened_until"] = 0.0
    return False


def record_circuit_success(key: str) -> None:
    state = _CIRCUIT_STATE.setdefault(key, {"failures": 0.0, "opened_until": 0.0})
    state["failures"] = 0.0
    state["opened_until"] = 0.0


def record_circuit_failure(key: str, threshold: int, cooldown_s: float) -> None:
    state = _CIRCUIT_STATE.setdefault(key, {"failures": 0.0, "opened_until": 0.0})
    failures = int(state.get("failures", 0.0)) + 1
    state["failures"] = float(failures)
    if failures >= threshold:
        state["opened_until"] = time.monotonic() + cooldown_s


def retry_sleep(backoff_s: float, attempt_index: int) -> None:
    if backoff_s <= 0:
        return
    time.sleep(backoff_s * (2**attempt_index))


def post_json_with_retries(
    *,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout_s: float,
    retry_count: int,
    retry_backoff_s: float,
    transport_error_code: str,
    provider_name: str,
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
                retry_sleep(retry_backoff_s, attempt)
                continue
            return llm_error("llm_http_error", f"{provider_name} returned HTTP {code}.")
        except urllib.error.URLError as exc:
            if attempt < retry_count:
                retry_sleep(retry_backoff_s, attempt)
                continue
            return llm_error(transport_error_code, f"{provider_name} transport error: {exc.reason}")
        except TimeoutError:
            if attempt < retry_count:
                retry_sleep(retry_backoff_s, attempt)
                continue
            return llm_error(transport_error_code, f"{provider_name} request timed out.")
        except Exception:
            if attempt < retry_count:
                retry_sleep(retry_backoff_s, attempt)
                continue
            return llm_error(transport_error_code, f"{provider_name} request failed unexpectedly.")
    return llm_error(transport_error_code, f"{provider_name} request exhausted retries.")


def resolve_runtime_settings(user_prefs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "timeout_s": as_positive_float(
            user_prefs.get("llm_timeout_s", os.getenv("PLANNER_LLM_TIMEOUT_S", 12)),
            12.0,
        ),
        "retry_count": as_bounded_int(
            user_prefs.get("llm_retry_count", os.getenv("PLANNER_LLM_RETRIES", 1)),
            default=1,
            min_value=0,
            max_value=5,
        ),
        "retry_backoff_s": as_positive_float(
            user_prefs.get("llm_retry_backoff_s", os.getenv("PLANNER_LLM_RETRY_BACKOFF_S", 0.5)),
            0.5,
            min_value=0.0,
            max_value=30.0,
        ),
        "circuit_threshold": as_bounded_int(
            user_prefs.get("llm_circuit_failures", os.getenv("PLANNER_LLM_CIRCUIT_FAILURES", 3)),
            default=3,
            min_value=1,
            max_value=20,
        ),
        "circuit_cooldown_s": as_positive_float(
            user_prefs.get("llm_circuit_cooldown_s", os.getenv("PLANNER_LLM_CIRCUIT_COOLDOWN_S", 30)),
            30.0,
            min_value=1.0,
            max_value=900.0,
        ),
    }
