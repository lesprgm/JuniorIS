from __future__ import annotations

import json

import pytest

from src import api_server
from tests.semantic_test_utils import inline_semantic_prefs


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
        "small indoor room with chair and lamp",
        scene_type="indoor_room",
        required_roles=["chair", "lamp"],
        style_tags=["cozy"],
        color_tags=["warm"],
        max_props=3,
    )


def test_run_plan_and_compile_success(tmp_path):
    result = api_server.run_plan_and_compile(
        prompt_text="small indoor room with chair and lamp",
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
    assert result["fallback_used"] is False
    assert result["fallback_reason"] is None
    assert isinstance(result["candidate_asset_ids"], list)
    assert isinstance(result["semantic_receipts"], dict)

    manifest_path = tmp_path / result["world_id"] / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == "0.2"
    assert manifest["world_id"] == result["world_id"]
    assert manifest["portal_ready_at_phase"] == "phase0"
    assert "safe_spawn" in manifest
    assert "prompt_plan" in manifest
    assert manifest["planner_backend"] == "llm"
    assert manifest["semantic_path_status"] == "ok"
    assert manifest["fallback_used"] is False
    assert manifest["fallback_reason"] is None
    assert manifest["placement_intent"]["density_profile"] == "normal"
    assert manifest["placement_plan"]["target_count"] >= 1
    assert "stylekit" in manifest
    assert "runtime_polish" in manifest


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
            "fallback_used": False,
            "fallback_reason": None,
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
            "fallback_used": False,
            "fallback_reason": None,
        }

    monkeypatch.setattr(api_server, "plan_worldspec", fake_plan_worldspec)
    result = api_server.run_plan_and_compile("test prompt", build_root=tmp_path)
    _assert_failure_contract_v02(result)
    assert result["error_code"] == "planner_failed"
    assert result["planner_error_code"] == "llm_unavailable"
    assert result["planner_backend"] == "llm_unavailable"
    assert result["candidate_asset_ids"] == ["core_chair_01"]
    assert result["semantic_path_status"] == "failed"
    assert result["fallback_used"] is False


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
            "fallback_used": False,
            "fallback_reason": None,
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
        prompt_text="small indoor room with chair and lamp",
        optional_seed=1,
        user_prefs=prefs,
        build_root=tmp_path,
    )

    assert result["ok"] is True
    _assert_success_contract_v02(result)
    assert result["planner_backend"] == "llm"
    assert result["selected_prompt"] == "small indoor room with chair and lamp"
    expected_ids = set(prefs["llm_plan"]["selection"]["asset_ids"])
    assert set(result["candidate_asset_ids"]) >= expected_ids


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
