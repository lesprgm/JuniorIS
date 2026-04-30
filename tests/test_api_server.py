from __future__ import annotations

import json

import pytest

from src.api import server as api_server
from src.contracts.runtime import validate_api_response_contract
from tests.semantic_test_utils import inline_semantic_prefs


# Keep behavior deterministic so planner/runtime contracts stay stable.
def _assert_success_contract_v02(payload):
    assert payload["api_contract_version"] == "0.2"
    assert payload["ok"] is True
    assert payload["request_id"].startswith("req_")
    assert payload["trace_id"].startswith("trace_")
    readiness = payload["readiness"]
    assert isinstance(readiness, dict)
    assert readiness["phase0_ready"] is True
    assert readiness["safe_spawn_ready"] is True
    assert readiness["portal_allowed"] is True
    assert readiness["blocked_reasons"] == []


def _assert_failure_contract_v02(payload):
    assert payload["api_contract_version"] == "0.2"
    assert payload["ok"] is False
    assert payload["request_id"].startswith("req_")
    assert payload["trace_id"].startswith("trace_")
    assert "retryable" in payload
    assert "retry_after_ms" in payload
    assert payload["details"] == payload["errors"]


def _inline_semantic_prefs():
    return inline_semantic_prefs(
        "small indoor room with chair and table",
        scene_type="indoor_room",
        required_roles=["chair", "table"],
        style_tags=["cozy"],
        color_tags=["warm"],
        max_props=3,
    )


def test_run_plan_and_compile_success(tmp_path):
    result = api_server.run_plan_and_compile(
        prompt_text="small indoor room with chair and table",
        optional_seed=42,
        user_prefs=_inline_semantic_prefs(),
        build_root=tmp_path,
    )

    assert result["ok"] is True
    _assert_success_contract_v02(result)
    assert result["errors"] == []
    assert result["world_id"].startswith("world_")
    assert result["manifest_url"] == f"/build/{result['world_id']}/manifest.json"
    assert "prompt_plan" in result
    assert result["selected_prompt"] == result["prompt_plan"]["selected_prompt"]
    assert result["planner_backend"] == "llm"
    assert result["semantic_path_status"] == "ok"
    assert isinstance(result["candidate_asset_ids"], list)
    assert isinstance(result["semantic_receipts"], dict)
    assert "required_roles" not in result["scene_program"]
    assert "required_roles" not in result["intent_spec"]

    manifest_path = tmp_path / result["world_id"] / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == "0.2"
    assert manifest["world_id"] == result["world_id"]
    assert manifest["portal_ready_at_phase"] == "phase0"
    assert "safe_spawn" in manifest
    assert manifest["readiness"]["phase0_ready"] is True
    assert manifest["readiness"]["safe_spawn_ready"] is True
    assert manifest["readiness"]["portal_allowed"] is True
    assert manifest["phase0_url"] == f"/build/{result['world_id']}/phase0.json"
    assert isinstance(manifest["phase0_data"], dict)
    assert manifest["phase0_data"]["world_id"] == result["world_id"]
    assert "prompt_plan" in manifest
    assert manifest["planner_backend"] == "llm"
    assert manifest["semantic_path_status"] == "ok"
    assert manifest["placement_intent"]["density_profile"] == "normal"
    assert manifest["placement_plan"]["target_count"] >= 1
    assert manifest["scene_context"]["archetype"] == "study"
    assert manifest["scene_context"]["zones"] == []
    assert manifest["decor_plan"]["archetype"] == "study"
    assert isinstance(manifest["decor_plan"]["entries"], list)
    assert manifest["colors"]["wall"].startswith("#")
    assert manifest["colors"]["floor"].startswith("#")
    assert manifest["colors"]["accent"].startswith("#")
    assert set(manifest["surface_material_selection"].keys()) >= {"wall", "floor", "ceiling"}
    assert "floor" in manifest["shell_material_bindings"]
    assert "stylekit" in manifest
    assert "runtime_polish" in manifest


def test_success_response_matches_api_contract_schema(tmp_path):
    payload = api_server.run_plan_and_compile(
        "small indoor room with chair and lamp",
        optional_seed=7,
        user_prefs=inline_semantic_prefs(
            "small indoor room with chair and lamp",
            scene_type="indoor_room",
            required_roles=["chair", "lamp"],
            style_tags=["cozy"],
            color_tags=["warm"],
            max_props=2,
        ),
        build_root=tmp_path,
    )
    result = validate_api_response_contract(payload)
    assert result["ok"] is True, result["errors"]


def test_failure_response_matches_api_contract_schema():
    payload = api_server.run_plan_and_compile("", build_root="build_test")
    result = validate_api_response_contract(payload)
    assert result["ok"] is True, result["errors"]


def test_run_plan_and_compile_rejects_empty_prompt():
    result = api_server.run_plan_and_compile(prompt_text="", build_root="build_test")
    _assert_failure_contract_v02(result)
    assert result["error_code"] == "invalid_request"
    assert result["recoverable"] is True
    assert result["retryable"] is False
    assert any(error["path"] == "$.prompt_text" for error in result["errors"])


def test_run_plan_and_compile_rejects_bool_seed():
    result = api_server.run_plan_and_compile(prompt_text="test", optional_seed=True, build_root="build_test")
    _assert_failure_contract_v02(result)
    assert result["error_code"] == "invalid_request"
    assert any(error["path"] == "$.optional_seed" for error in result["errors"])


def test_run_plan_and_compile_maps_planner_failures(monkeypatch, tmp_path):
    def fake_plan_worldspec(*args, **kwargs):
        return {
            "ok": False,
            "errors": [{"path": "$.pack_registry", "message": "missing"}],
            "prompt_plan": {"mode": "llm", "selected_prompt": "test prompt"},
            "semantic_path_status": "failed",
        }

    monkeypatch.setattr(api_server, "plan_worldspec", fake_plan_worldspec)
    result = api_server.run_plan_and_compile("test prompt", build_root=tmp_path)
    _assert_failure_contract_v02(result)
    assert result["error_code"] == "planner_failed"
    assert result["prompt_plan"]["selected_prompt"] == "test prompt"


def test_run_plan_and_compile_maps_llm_planner_failure_reason(monkeypatch, tmp_path):
    def fake_plan_worldspec(*args, **kwargs):
        return {
            "ok": False,
            "error_code": "llm_unavailable",
            "errors": [{"path": "$.llm", "message": "no endpoint"}],
            "prompt_plan": {"mode": "llm", "selected_prompt": "test prompt"},
            "planner_backend": "llm_unavailable",
            "candidate_asset_ids": ["core_chair_01"],
            "semantic_path_status": "failed",
        }

    monkeypatch.setattr(api_server, "plan_worldspec", fake_plan_worldspec)
    result = api_server.run_plan_and_compile("test prompt", build_root=tmp_path)
    _assert_failure_contract_v02(result)
    assert result["error_code"] == "planner_failed"
    assert result["planner_error_code"] == "llm_unavailable"
    assert result["planner_backend"] == "llm_unavailable"
    assert result["candidate_asset_ids"] == ["core_chair_01"]
    assert result["semantic_path_status"] == "failed"


def test_run_plan_and_compile_includes_invalid_intent_payload_on_semantic_failure(monkeypatch, tmp_path):
    def fake_plan_worldspec(*args, **kwargs):
        return {
            "ok": False,
            "error_code": "semantic_invalid_intent",
            "errors": [{"path": "$.llm.intent.execution_archetype", "message": "missing"}],
            "prompt_plan": {"mode": "llm", "selected_prompt": "cozy bedroom"},
            "planner_backend": "llm",
            "candidate_asset_ids": ["bed_01"],
            "semantic_path_status": "failed",
            "invalid_intent_payload": {
                "intent": {
                    "scene_type": "bedroom",
                    "semantic_slots": [],
                }
            },
        }

    monkeypatch.setattr(api_server, "plan_worldspec", fake_plan_worldspec)
    result = api_server.run_plan_and_compile("cozy bedroom", build_root=tmp_path)
    _assert_failure_contract_v02(result)
    assert result["error_code"] == "planner_failed"
    assert result["planner_error_code"] == "semantic_invalid_intent"
    assert result["invalid_intent_payload"]["intent"]["scene_type"] == "bedroom"


def test_run_plan_and_compile_maps_llm_transport_failure_without_crash(monkeypatch, tmp_path):
    def fake_plan_worldspec(*args, **kwargs):
        return {
            "ok": False,
            "error_code": "llm_transport_error",
            "errors": [{"path": "$.llm", "message": "timeout"}],
            "prompt_plan": {"mode": "llm", "selected_prompt": "test prompt"},
            "planner_backend": "llm_unavailable",
            "candidate_asset_ids": [],
            "semantic_path_status": "failed",
        }

    monkeypatch.setattr(api_server, "plan_worldspec", fake_plan_worldspec)
    result = api_server.run_plan_and_compile("test prompt", build_root=tmp_path)
    _assert_failure_contract_v02(result)
    assert result["error_code"] == "planner_failed"
    assert result["planner_error_code"] == "llm_transport_error"
    assert result["recoverable"] is True


def test_run_plan_and_compile_maps_unhandled_planner_exception(monkeypatch, tmp_path):
    def fake_plan_worldspec(*args, **kwargs):
        raise RuntimeError("planner crash")

    monkeypatch.setattr(api_server, "plan_worldspec", fake_plan_worldspec)
    result = api_server.run_plan_and_compile("test prompt", build_root=tmp_path)
    _assert_failure_contract_v02(result)
    assert result["error_code"] == "internal_error"
    assert any(error["path"] == "$.planner" for error in result["errors"])


def test_run_plan_and_compile_maps_compile_failures(monkeypatch, tmp_path):
    def fake_compile_phase0(*args, **kwargs):
        return {"ok": False, "errors": [{"path": "$.safe_spawn", "message": "no_safe_spawn_found"}]}

    monkeypatch.setattr(api_server, "compile_phase0", fake_compile_phase0)
    result = api_server.run_plan_and_compile("test prompt", user_prefs=_inline_semantic_prefs(), build_root=tmp_path)
    _assert_failure_contract_v02(result)
    assert result["error_code"] == "spawn_failed"


def test_run_plan_and_compile_maps_unhandled_compile_exception(monkeypatch, tmp_path):
    def fake_compile_phase0(*args, **kwargs):
        raise RuntimeError("compile crash")

    monkeypatch.setattr(api_server, "compile_phase0", fake_compile_phase0)
    result = api_server.run_plan_and_compile("test prompt", user_prefs=_inline_semantic_prefs(), build_root=tmp_path)
    _assert_failure_contract_v02(result)
    assert result["error_code"] == "internal_error"
    assert any(error["path"] == "$.compile" for error in result["errors"])


def test_run_plan_and_compile_maps_manifest_failures(monkeypatch, tmp_path):
    def fake_write_manifest(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(api_server, "_write_manifest", fake_write_manifest)
    result = api_server.run_plan_and_compile("test prompt", user_prefs=_inline_semantic_prefs(), build_root=tmp_path)
    _assert_failure_contract_v02(result)
    assert result["error_code"] == "manifest_failed"
    assert any(error["path"] == "$.manifest" for error in result["errors"])


def test_run_plan_and_compile_rejects_non_llm_prompt_mode(tmp_path):
    result = api_server.run_plan_and_compile(
        prompt_text="clean room",
        optional_seed=1,
        user_prefs={"prompt_mode": "literal"},
        build_root=tmp_path,
    )

    assert result["ok"] is False
    _assert_failure_contract_v02(result)
    assert result["error_code"] == "planner_failed"
    assert result["planner_error_code"] == "invalid_prompt_mode"
    assert any(error["path"] == "$.user_prefs.prompt_mode" for error in result["errors"])


def test_run_plan_and_compile_llm_mode_success_with_inline_plan(tmp_path):
    prefs = _inline_semantic_prefs()
    result = api_server.run_plan_and_compile(
        prompt_text="small indoor room with chair and table",
        optional_seed=1,
        user_prefs=prefs,
        build_root=tmp_path,
    )

    assert result["ok"] is True
    _assert_success_contract_v02(result)
    assert result["planner_backend"] == "llm"
    assert result["selected_prompt"] == "small indoor room with chair and table"
    assert result["candidate_asset_ids"]


@pytest.mark.skipif(api_server.FastAPI is None, reason="FastAPI is not installed")
def test_fastapi_endpoint_success(tmp_path):
    from fastapi.testclient import TestClient

    client = TestClient(api_server.app)
    response = client.post(
        "/plan_and_compile",
        json={
            "prompt_text": "small indoor room with chair",
            "optional_seed": 7,
            "user_prefs": _inline_semantic_prefs(),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    _assert_success_contract_v02(payload)
    assert payload["world_id"].startswith("world_")


@pytest.mark.skipif(api_server.FastAPI is None, reason="FastAPI is not installed")
def test_voice_chatter_plan_returns_text_only_even_when_prefetch_requested(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    client = TestClient(api_server.app)
    monkeypatch.setattr(api_server, "BUILD_ROOT", tmp_path)
    response = client.post(
        "/voice/chatter_plan",
        json={
            "prompt_text": "museum room",
            "prefetch_audio": True,
            "scene_context": {"concept_label": "museum"},
            "user_prefs": {"enable_loading_chatter": True},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["items"]
    assert payload["items"][0]["phase"] == "prompt_received"
    assert "audio_url" not in payload["items"][0]


@pytest.mark.skipif(api_server.FastAPI is None, reason="FastAPI is not installed")
def test_voice_tts_returns_cached_audio_url(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    client = TestClient(api_server.app)
    monkeypatch.setattr(api_server, "BUILD_ROOT", tmp_path)

    monkeypatch.setattr(
        api_server,
        "build_tts_artifact",
        lambda text, *, build_root, user_prefs=None: {
            "ok": True,
            "cache_key": "voice123",
            "artifact_path": str(tmp_path / "voice_cache" / "voice123.mp3"),
            "audio_url": "/build/voice_cache/voice123.mp3",
            "content_type": "audio/mpeg",
        },
    )
    response = client.post("/voice/tts", json={"text": "hello there"})

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "ok": True,
        "cache_key": "voice123",
        "audio_url": "/build/voice_cache/voice123.mp3",
        "content_type": "audio/mpeg",
    }


@pytest.mark.skipif(api_server.FastAPI is None, reason="FastAPI is not installed")
def test_voice_chatter_plan_does_not_prefetch_when_voice_disabled(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    client = TestClient(api_server.app)
    monkeypatch.setattr(api_server, "BUILD_ROOT", tmp_path)
    monkeypatch.setenv("JUNIORIS_ENABLE_LOADING_CHATTER", "false")

    response = client.post(
        "/voice/chatter_plan",
        json={
            "prompt_text": "museum room",
            "prefetch_audio": True,
            "user_prefs": {"enable_loading_chatter": False},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["voice_enabled"] is False


@pytest.mark.skipif(api_server.FastAPI is None, reason="FastAPI is not installed")
def test_fastapi_endpoint_invalid_request():
    from fastapi.testclient import TestClient

    client = TestClient(api_server.app)
    response = client.post("/plan_and_compile", json={"prompt_text": "   "})
    assert response.status_code == 400
    payload = response.json()
    _assert_failure_contract_v02(payload)
    assert payload["error_code"] == "invalid_request"


@pytest.mark.skipif(api_server.FastAPI is None, reason="FastAPI is not installed")
def test_fastapi_endpoint_spawn_failure_is_422(monkeypatch):
    from fastapi.testclient import TestClient

    def fake_compile_phase0(*args, **kwargs):
        return {"ok": False, "errors": [{"path": "$.safe_spawn", "message": "no_safe_spawn_found"}]}

    monkeypatch.setattr(api_server, "compile_phase0", fake_compile_phase0)
    client = TestClient(api_server.app)
    response = client.post("/plan_and_compile", json={"prompt_text": "valid prompt", "user_prefs": _inline_semantic_prefs()})
    assert response.status_code == 422
    payload = response.json()
    _assert_failure_contract_v02(payload)
    assert payload["error_code"] == "spawn_failed"
