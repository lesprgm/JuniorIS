from tests.semantic_test_utils import approved_surface_material_selection
import json
import urllib.error

from src.llm import planner as llm_planner


GEMINI_MODEL = "gemini-2.5-flash"
OPENROUTER_MODEL = "openai/gpt-5.4-mini"


# Keep behavior deterministic so planner/runtime contracts stay stable.
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
        "prompt_plan": {"selected_prompt": "small room", "creative_variants": ["small room"], "input_prompt": "small room"},
        "candidate_assets": [{"asset_id": "core_chair_01", "label": "chair", "tags": ["chair"]}],
        "allowed_stylekit_ids": ["neutral_daylight"],
        "allowed_pack_ids": ["core_pack"],
        "default_budgets": {"max_props": 4, "max_texture_tier": 1, "max_lights": 2},
        "stylekit_candidates": [{"stylekit_id": "neutral_daylight", "tags": ["neutral"]}],
        "pack_candidates": [{"pack_id": "core_pack", "tags": ["indoor"], "asset_count": 4}],
        "surface_material_candidates": {
            "wall": [{"material_id": approved_surface_material_selection()["wall"], "display_name": "Wall", "surface_roles": ["wall"], "preview_color_hex": "#d8d8d8"}],
            "floor": [{"material_id": approved_surface_material_selection()["floor"], "display_name": "Floor", "surface_roles": ["floor"], "preview_color_hex": "#8b7d6b"}],
            "ceiling": [{"material_id": approved_surface_material_selection()["ceiling"], "display_name": "Ceiling", "surface_roles": ["ceiling"], "preview_color_hex": "#eeeeee"}],
            "accent": [{"material_id": approved_surface_material_selection()["accent"], "display_name": "Accent", "surface_roles": ["accent"], "preview_color_hex": "#4a90e2"}],
        },
    }


def _gemini_prefs(**overrides):
    prefs = {"llm_provider": "gemini"}
    prefs.update(overrides)
    return prefs


def _openrouter_prefs(**overrides):
    prefs = {"llm_provider": "openrouter"}
    prefs.update(overrides)
    return prefs


def _set_gemini_env(monkeypatch, *, api_key: str | None = None, model: str | None = GEMINI_MODEL):
    if api_key is None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    else:
        monkeypatch.setenv("GEMINI_API_KEY", api_key)
    if model is None:
        monkeypatch.delenv("GEMINI_MODEL", raising=False)
    else:
        monkeypatch.setenv("GEMINI_MODEL", model)


def _set_openrouter_env(monkeypatch, *, api_key: str | None = None, model: str | None = OPENROUTER_MODEL):
    if api_key is None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENROUTER_API_KEY", api_key)
    if model is None:
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    else:
        monkeypatch.setenv("OPENROUTER_MODEL", model)


def _semantic_plan(asset_ids=None):
    asset_ids = asset_ids or ["core_chair_01"]
    return {
        "intent": {
            "scene_type": "study",
            "concept_label": "study",
            "creative_summary": "study room",
            "intended_use": "use as a study",
            "focal_object_role": "chair",
            "focal_wall": "front",
            "circulation_preference": "clear_center",
            "empty_space_preference": "balanced",
            "creative_tags": ["study"],
            "mood_tags": ["cozy"],
            "style_descriptors": ["cozy"],
            "execution_archetype": "study",
            "archetype": "study",
            "semantic_slots": [
                {
                    "slot_id": "chair_slot_1",
                    "concept": "chair",
                    "priority": "must",
                    "count": 1,
                    "runtime_role_hint": "chair",
                }
            ],
            "primary_anchor_object": {"role": "chair", "rationale": "single focal object"},
            "secondary_support_objects": [],
            "relation_graph": [{"source_role": "chair", "target_role": "room", "relation": "middle"}],
            "density_target": "cluttered",
            "symmetry_preference": "balanced",
            "walkway_preservation_intent": {
                "keep_central_path_clear": True,
                "keep_entry_clear": True,
                "notes": "leave a clear path to the main seat",
            },
            "scene_features": [],
            "style_tags": ["cozy"],
            "color_tags": ["warm"],
            "style_cues": {
                "style_tags": ["cozy"],
                "color_tags": ["warm"],
                "lighting_tags": [],
                "mood_tags": [],
            },
            "confidence": 0.9,
        },
        "placement_intent": {
            "density_profile": "cluttered",
            "anchor_preferences": ["clustered"],
            "adjacency_pairs": [{"source_role": "chair", "target_role": "lamp", "relation": "near"}],
            "layout_mood": "crowded",
        },
        "selection": {
            "selected_prompt": "small room",
            "stylekit_id": "neutral_daylight",
            "pack_ids": ["core_pack"],
            "slot_asset_map": {"chair_slot_1": asset_ids[0]},
            "asset_ids": asset_ids,
            "budgets": {"max_props": 1},
            "surface_material_selection": approved_surface_material_selection(style_tags=["cozy"], color_tags=["warm"]),
            "alternatives": {"chair_slot_1": asset_ids},
            "rationale": ["chair matches the requested role"],
            "confidence": 0.9,
        },
    }


def _gemini_response(payload: dict):
    body = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps(payload)
                        }
                    ]
                }
            }
        ]
    }
    return _FakeResponse(json.dumps(body))


def _openrouter_response(payload: dict):
    body = {"output_text": json.dumps(payload)}
    return _FakeResponse(json.dumps(body))


def test_gemini_retry_then_success(monkeypatch):
    llm_planner._CIRCUIT_STATE.clear()
    _set_gemini_env(monkeypatch, api_key="gemini-test-key")
    attempts = {"count": 0}

    def fake_urlopen(_request, timeout=0):
        _ = timeout
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise urllib.error.URLError("temporary network")
        return _gemini_response({"plan": _semantic_plan()})

    monkeypatch.setattr(llm_planner.urllib.request, "urlopen", fake_urlopen)

    result = llm_planner.request_llm_plan(
        **_base_kwargs(),
        user_prefs=_gemini_prefs(llm_api_key="gemini-test-key", llm_retry_count=1, llm_retry_backoff_s=0),
    )
    assert attempts["count"] == 4
    assert result["ok"] is True
    assert result["backend"] == "gemini"
    assert result["plan"]["selection"]["asset_ids"] == ["core_chair_01"]


def test_gemini_circuit_open_after_repeated_failures(monkeypatch):
    llm_planner._CIRCUIT_STATE.clear()
    _set_gemini_env(monkeypatch, api_key="gemini-test-key")

    def always_fail(_request, timeout=0):
        _ = timeout
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(llm_planner.urllib.request, "urlopen", always_fail)

    first = llm_planner.request_llm_plan(
        **_base_kwargs(),
        user_prefs=_gemini_prefs(
            llm_api_key="gemini-test-key",
            llm_retry_count=0,
            llm_circuit_failures=1,
            llm_circuit_cooldown_s=60,
        ),
    )
    second = llm_planner.request_llm_plan(
        **_base_kwargs(),
        user_prefs=_gemini_prefs(
            llm_api_key="gemini-test-key",
            llm_retry_count=0,
            llm_circuit_failures=1,
            llm_circuit_cooldown_s=60,
        ),
    )
    assert first["ok"] is False
    assert first["error_code"] == "llm_transport_error"
    assert second["ok"] is False
    assert second["error_code"] == "llm_circuit_open"


def test_gemini_requires_api_key(monkeypatch):
    llm_planner._CIRCUIT_STATE.clear()
    _set_gemini_env(monkeypatch, api_key=None)

    result = llm_planner.request_llm_plan(
        **_base_kwargs(),
        user_prefs=_gemini_prefs(),
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
    _set_gemini_env(monkeypatch, api_key="gemini-test-key", model=None)
    monkeypatch.delenv("PLANNER_LLM_MODEL", raising=False)

    result = llm_planner.request_llm_plan(
        **_base_kwargs(),
        user_prefs=_gemini_prefs(),
    )
    assert result["ok"] is False
    assert result["error_code"] == "llm_unavailable"
    assert "GEMINI_MODEL" in result["message"]


def test_gemini_success(monkeypatch):
    llm_planner._CIRCUIT_STATE.clear()
    _set_gemini_env(monkeypatch, api_key="gemini-test-key")

    def fake_urlopen(request, timeout=0):
        _ = timeout
        assert request.full_url == (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{GEMINI_MODEL}:generateContent?key=gemini-test-key"
        )
        return _gemini_response({"plan": _semantic_plan()})

    monkeypatch.setattr(llm_planner.urllib.request, "urlopen", fake_urlopen)

    result = llm_planner.request_llm_plan(
        **_base_kwargs(),
        user_prefs=_gemini_prefs(),
    )
    assert result["ok"] is True
    assert result["backend"] == "gemini"
    assert result["plan"]["intent"]["semantic_slots"][0]["slot_id"] == "chair_slot_1"
    assert result["plan"]["placement_intent"]["density_profile"] == "cluttered"
    assert result["plan"]["selection"]["asset_ids"] == ["core_chair_01"]


def test_inline_intent_request_reads_placement_intent():
    result = llm_planner.request_llm_intent(
        prompt_plan={"selected_prompt": "messy room", "input_prompt": "messy room"},
        user_prefs={"llm_plan": _semantic_plan()},
    )
    assert result["ok"] is True
    assert result["intent_payload"]["placement_intent"]["density_profile"] == "cluttered"


def test_inline_selection_request_reads_selection():
    result = llm_planner.request_llm_selection(
        **_base_kwargs(),
        intent_spec=_semantic_plan()["intent"],
        placement_intent=_semantic_plan()["placement_intent"],
        user_prefs={"llm_plan": _semantic_plan()},
    )
    assert result["ok"] is True
    assert result["selection"]["asset_ids"] == ["core_chair_01"]


def test_request_llm_plan_uses_empty_design_brief_when_brief_stage_fails(monkeypatch):
    original_design_brief = llm_planner.request_llm_design_brief
    original_intent = llm_planner.request_llm_intent
    original_selection = llm_planner.request_llm_selection

    monkeypatch.setattr(
        llm_planner,
        "request_llm_design_brief",
        lambda **kwargs: {"ok": False, "error_code": "llm_transport_error", "message": "offline"},
    )
    monkeypatch.setattr(
        llm_planner,
        "request_llm_intent",
        lambda **kwargs: {
            "ok": True,
            "backend": "inline_override",
            "intent_payload": {
                "intent": _semantic_plan()["intent"],
                "placement_intent": _semantic_plan()["placement_intent"],
            },
        },
    )
    monkeypatch.setattr(
        llm_planner,
        "request_llm_selection",
        lambda **kwargs: {"ok": True, "backend": "inline_override", "selection": _semantic_plan()["selection"]},
    )

    try:
        result = llm_planner.request_llm_plan(
            **_base_kwargs(),
            user_prefs=_gemini_prefs(),
        )
    finally:
        llm_planner.request_llm_design_brief = original_design_brief
        llm_planner.request_llm_intent = original_intent
        llm_planner.request_llm_selection = original_selection

    assert result["ok"] is True
    assert result["plan"]["design_brief"] == {}
    assert result["plan"]["selection"]["asset_ids"] == ["core_chair_01"]


def test_openrouter_success(monkeypatch):
    llm_planner._CIRCUIT_STATE.clear()
    _set_openrouter_env(monkeypatch, api_key="openrouter-test-key")

    def fake_urlopen(request, timeout=0):
        _ = timeout
        assert request.full_url == "https://openrouter.ai/api/v1/responses"
        return _openrouter_response({"plan": _semantic_plan()})

    monkeypatch.setattr(llm_planner.urllib.request, "urlopen", fake_urlopen)

    result = llm_planner.request_llm_plan(
        **_base_kwargs(),
        user_prefs=_openrouter_prefs(),
    )
    assert result["ok"] is True
    assert result["backend"] == "openrouter"
    assert result["plan"]["selection"]["asset_ids"] == ["core_chair_01"]


def test_stage_specific_models_override_global_openrouter_model(monkeypatch):
    llm_planner._CIRCUIT_STATE.clear()
    _set_openrouter_env(monkeypatch, api_key="openrouter-test-key", model="global-model")
    monkeypatch.setenv("PLANNER_LLM_INTENT_MODEL", "fast-model")
    monkeypatch.setenv("PLANNER_LLM_SELECTION_MODEL", "smart-model")
    seen_models = []

    def fake_urlopen(request, timeout=0):
        _ = timeout
        body = json.loads(request.data.decode("utf-8"))
        seen_models.append(body["model"])
        if len(seen_models) == 1:
            return _openrouter_response(
                {
                    "intent": _semantic_plan()["intent"],
                    "placement_intent": _semantic_plan()["placement_intent"],
                }
            )
        return _openrouter_response({"selection": _semantic_plan()["selection"]})

    monkeypatch.setattr(llm_planner.urllib.request, "urlopen", fake_urlopen)

    result = llm_planner.request_llm_plan(
        **_base_kwargs(),
        user_prefs=_openrouter_prefs(),
    )
    assert result["ok"] is True
    assert seen_models == ["global-model", "fast-model", "smart-model"]


def test_openrouter_selection_can_request_more_reasoning(monkeypatch):
    llm_planner._CIRCUIT_STATE.clear()
    _set_openrouter_env(monkeypatch, api_key="openrouter-test-key")
    captured = {}

    def fake_urlopen(request, timeout=0):
        _ = timeout
        captured.update(json.loads(request.data.decode("utf-8")))
        return _openrouter_response({"selection": _semantic_plan()["selection"]})

    monkeypatch.setattr(llm_planner.urllib.request, "urlopen", fake_urlopen)

    result = llm_planner.request_llm_selection(
        **_base_kwargs(),
        intent_spec=_semantic_plan()["intent"],
        placement_intent=_semantic_plan()["placement_intent"],
        user_prefs=_openrouter_prefs(llm_selection_reasoning_effort="high", llm_selection_max_output_tokens=1600),
    )

    assert result["ok"] is True
    assert captured["reasoning"] == {"effort": "high"}
    assert captured["max_output_tokens"] == 1600


def test_openrouter_defaults_to_medium_reasoning(monkeypatch):
    llm_planner._CIRCUIT_STATE.clear()
    _set_openrouter_env(monkeypatch, api_key="openrouter-test-key")
    captured = {}

    def fake_urlopen(request, timeout=0):
        _ = timeout
        captured.update(json.loads(request.data.decode("utf-8")))
        return _openrouter_response({"selection": _semantic_plan()["selection"]})

    monkeypatch.setattr(llm_planner.urllib.request, "urlopen", fake_urlopen)

    result = llm_planner.request_llm_selection(
        **_base_kwargs(),
        intent_spec=_semantic_plan()["intent"],
        placement_intent=_semantic_plan()["placement_intent"],
        user_prefs=_openrouter_prefs(),
    )

    assert result["ok"] is True
    assert captured["reasoning"] == {"effort": "medium"}


def test_gemini_intent_can_request_thinking_budget(monkeypatch):
    llm_planner._CIRCUIT_STATE.clear()
    _set_gemini_env(monkeypatch, api_key="gemini-test-key")
    captured = {}

    def fake_urlopen(request, timeout=0):
        _ = timeout
        captured.update(json.loads(request.data.decode("utf-8")))
        return _gemini_response(
            {
                "intent": _semantic_plan()["intent"],
                "placement_intent": _semantic_plan()["placement_intent"],
            }
        )

    monkeypatch.setattr(llm_planner.urllib.request, "urlopen", fake_urlopen)

    result = llm_planner.request_llm_intent(
        prompt_plan={"selected_prompt": "small room", "input_prompt": "small room"},
        user_prefs=_gemini_prefs(llm_intent_thinking_budget=512, llm_intent_max_output_tokens=2000),
    )

    assert result["ok"] is True
    assert captured["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 512}
    assert captured["generationConfig"]["maxOutputTokens"] == 2000


def test_openrouter_requires_model(monkeypatch):
    llm_planner._CIRCUIT_STATE.clear()
    _set_openrouter_env(monkeypatch, api_key="openrouter-test-key", model=None)

    result = llm_planner.request_llm_plan(
        **_base_kwargs(),
        user_prefs=_openrouter_prefs(),
    )
    assert result["ok"] is False
    assert result["error_code"] == "llm_unavailable"
    assert "OPENROUTER_MODEL" in result["message"]
