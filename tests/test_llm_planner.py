import json
import urllib.error

from src import llm_planner


class _FakeResponse:
    def __init__(self, payload: str):
        self._payload = payload

    def read(self):
        return self._payload.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _base_kwargs():
    return {
        "prompt_plan": {"selected_prompt": "small room", "creative_variants": ["small room"]},
        "candidate_assets": [{"asset_id": "core_chair_01", "label": "chair", "tags": ["chair"]}],
        "allowed_stylekit_ids": ["neutral_daylight"],
        "allowed_pack_ids": ["core_pack"],
        "default_budgets": {"max_props": 4, "max_texture_tier": 1, "max_lights": 2},
    }


def test_gemini_retry_then_success(monkeypatch):
    llm_planner._CIRCUIT_STATE.clear()
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    attempts = {"count": 0}

    def fake_urlopen(_request, timeout=0):
        _ = timeout
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise urllib.error.URLError("temporary network")
        body = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps({"plan": {"asset_ids": ["core_chair_01"]}})
                            }
                        ]
                    }
                }
            ]
        }
        return _FakeResponse(json.dumps(body))

    monkeypatch.setattr(llm_planner.urllib.request, "urlopen", fake_urlopen)

    result = llm_planner.request_llm_plan(
        **_base_kwargs(),
        user_prefs={
            "llm_provider": "gemini",
            "llm_api_key": "gemini-test-key",
            "llm_retry_count": 1,
            "llm_retry_backoff_s": 0,
        },
    )
    assert attempts["count"] == 2
    assert result["ok"] is True
    assert result["backend"] == "gemini"
    assert result["plan"]["asset_ids"] == ["core_chair_01"]


def test_gemini_circuit_open_after_repeated_failures(monkeypatch):
    llm_planner._CIRCUIT_STATE.clear()
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")

    def always_fail(_request, timeout=0):
        _ = timeout
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(llm_planner.urllib.request, "urlopen", always_fail)

    first = llm_planner.request_llm_plan(
        **_base_kwargs(),
        user_prefs={
            "llm_provider": "gemini",
            "llm_api_key": "gemini-test-key",
            "llm_retry_count": 0,
            "llm_circuit_failures": 1,
            "llm_circuit_cooldown_s": 60,
        },
    )
    second = llm_planner.request_llm_plan(
        **_base_kwargs(),
        user_prefs={
            "llm_provider": "gemini",
            "llm_api_key": "gemini-test-key",
            "llm_retry_count": 0,
            "llm_circuit_failures": 1,
            "llm_circuit_cooldown_s": 60,
        },
    )
    assert first["ok"] is False
    assert first["error_code"] == "llm_transport_error"
    assert second["ok"] is False
    assert second["error_code"] == "llm_circuit_open"


def test_gemini_requires_api_key(monkeypatch):
    llm_planner._CIRCUIT_STATE.clear()
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    result = llm_planner.request_llm_plan(
        **_base_kwargs(),
        user_prefs={"llm_provider": "gemini"},
    )
    assert result["ok"] is False
    assert result["error_code"] == "llm_unavailable"


def test_rejects_non_gemini_provider():
    llm_planner._CIRCUIT_STATE.clear()
    result = llm_planner.request_llm_plan(
        **_base_kwargs(),
        user_prefs={"llm_provider": "unsupported_provider", "llm_api_key": "dummy"},
    )
    assert result["ok"] is False
    assert result["error_code"] == "llm_unavailable"


def test_gemini_requires_model(monkeypatch):
    llm_planner._CIRCUIT_STATE.clear()
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("PLANNER_LLM_MODEL", raising=False)

    result = llm_planner.request_llm_plan(
        **_base_kwargs(),
        user_prefs={"llm_provider": "gemini"},
    )
    assert result["ok"] is False
    assert result["error_code"] == "llm_unavailable"
    assert "GEMINI_MODEL" in result["message"]


def test_gemini_success(monkeypatch):
    llm_planner._CIRCUIT_STATE.clear()
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")

    gemini_body = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps(
                                {
                                    "plan": {
                                        "selected_prompt": "small room",
                                        "stylekit_id": "neutral_daylight",
                                        "pack_ids": ["core_pack"],
                                        "asset_ids": ["core_chair_01"],
                                        "budgets": {"max_props": 1},
                                    }
                                }
                            )
                        }
                    ]
                }
            }
        ]
    }

    def fake_urlopen(request, timeout=0):
        _ = timeout
        assert request.full_url == (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-2.5-flash:generateContent?key=gemini-test-key"
        )
        return _FakeResponse(json.dumps(gemini_body))

    monkeypatch.setattr(llm_planner.urllib.request, "urlopen", fake_urlopen)

    result = llm_planner.request_llm_plan(
        **_base_kwargs(),
        user_prefs={"llm_provider": "gemini"},
    )
    assert result["ok"] is True
    assert result["backend"] == "gemini"
    assert result["plan"]["asset_ids"] == ["core_chair_01"]
