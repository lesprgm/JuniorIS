from __future__ import annotations

import json

import pytest

from src import api_server


def test_run_plan_and_compile_success(tmp_path):
    result = api_server.run_plan_and_compile(
        prompt_text="small indoor room with chair and lamp",
        optional_seed=42,
        user_prefs={"max_props": 3},
        build_root=tmp_path,
    )

    assert result["ok"] is True
    assert result["errors"] == []
    assert result["world_id"].startswith("world_")
    assert result["manifest_url"] == f"/build/{result['world_id']}/manifest.json"
    assert "prompt_plan" in result
    assert result["selected_prompt"] == result["prompt_plan"]["selected_prompt"]
    assert result["planner_backend"] in {"llm", "deterministic_fallback"}
    assert isinstance(result["candidate_asset_ids"], list)

    manifest_path = tmp_path / result["world_id"] / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["world_id"] == result["world_id"]
    assert manifest["portal_ready_at_phase"] == "phase0"
    assert "safe_spawn" in manifest
    assert "prompt_plan" in manifest
    assert manifest["planner_backend"] in {"llm", "deterministic_fallback"}


def test_run_plan_and_compile_rejects_empty_prompt():
    result = api_server.run_plan_and_compile(prompt_text="", build_root="build_test")
    assert result["ok"] is False
    assert result["error_code"] == "invalid_request"
    assert result["recoverable"] is True
    assert result["details"] == result["errors"]
    assert any(error["path"] == "$.prompt_text" for error in result["errors"])


def test_run_plan_and_compile_rejects_bool_seed():
    result = api_server.run_plan_and_compile(prompt_text="test", optional_seed=True, build_root="build_test")
    assert result["ok"] is False
    assert result["error_code"] == "invalid_request"
    assert any(error["path"] == "$.optional_seed" for error in result["errors"])


def test_run_plan_and_compile_maps_planner_failures(monkeypatch, tmp_path):
    def fake_plan_worldspec(*args, **kwargs):
        return {
            "ok": False,
            "errors": [{"path": "$.pack_registry", "message": "missing"}],
            "prompt_plan": {"mode": "auto", "selected_prompt": "test prompt"},
        }

    monkeypatch.setattr(api_server, "plan_worldspec", fake_plan_worldspec)
    result = api_server.run_plan_and_compile("test prompt", build_root=tmp_path)
    assert result["ok"] is False
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
        }

    monkeypatch.setattr(api_server, "plan_worldspec", fake_plan_worldspec)
    result = api_server.run_plan_and_compile("test prompt", build_root=tmp_path)
    assert result["ok"] is False
    assert result["error_code"] == "planner_failed"
    assert result["planner_error_code"] == "llm_unavailable"
    assert result["planner_backend"] == "llm_unavailable"
    assert result["candidate_asset_ids"] == ["core_chair_01"]


def test_run_plan_and_compile_maps_unhandled_planner_exception(monkeypatch, tmp_path):
    def fake_plan_worldspec(*args, **kwargs):
        raise RuntimeError("planner crash")

    monkeypatch.setattr(api_server, "plan_worldspec", fake_plan_worldspec)
    result = api_server.run_plan_and_compile("test prompt", build_root=tmp_path)
    assert result["ok"] is False
    assert result["error_code"] == "internal_error"
    assert any(error["path"] == "$.planner" for error in result["errors"])


def test_run_plan_and_compile_maps_compile_failures(monkeypatch, tmp_path):
    def fake_compile_phase0(*args, **kwargs):
        return {"ok": False, "errors": [{"path": "$.safe_spawn", "message": "no_safe_spawn_found"}]}

    monkeypatch.setattr(api_server, "compile_phase0", fake_compile_phase0)
    result = api_server.run_plan_and_compile("test prompt", build_root=tmp_path)
    assert result["ok"] is False
    assert result["error_code"] == "spawn_failed"


def test_run_plan_and_compile_maps_unhandled_compile_exception(monkeypatch, tmp_path):
    def fake_compile_phase0(*args, **kwargs):
        raise RuntimeError("compile crash")

    monkeypatch.setattr(api_server, "compile_phase0", fake_compile_phase0)
    result = api_server.run_plan_and_compile("test prompt", build_root=tmp_path)
    assert result["ok"] is False
    assert result["error_code"] == "internal_error"
    assert any(error["path"] == "$.compile" for error in result["errors"])


def test_run_plan_and_compile_maps_manifest_failures(monkeypatch, tmp_path):
    def fake_write_manifest(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(api_server, "_write_manifest", fake_write_manifest)
    result = api_server.run_plan_and_compile("test prompt", build_root=tmp_path)
    assert result["ok"] is False
    assert result["error_code"] == "manifest_failed"
    assert any(error["path"] == "$.manifest" for error in result["errors"])


def test_run_plan_and_compile_respects_literal_prompt_mode(tmp_path):
    result = api_server.run_plan_and_compile(
        prompt_text="clean room",
        optional_seed=1,
        user_prefs={"prompt_mode": "literal"},
        build_root=tmp_path,
    )

    assert result["ok"] is True
    assert result["prompt_plan"]["mode"] == "literal"
    assert result["selected_prompt"] == "clean room"


def test_run_plan_and_compile_llm_mode_success_with_inline_plan(tmp_path):
    result = api_server.run_plan_and_compile(
        prompt_text="small room",
        optional_seed=1,
        user_prefs={
            "prompt_mode": "llm",
            "llm_plan": {
                "selected_prompt": "small modern room",
                "stylekit_id": "neutral_daylight",
                "pack_ids": ["core_pack"],
                "asset_ids": ["core_chair_01", "core_table_01"],
                "budgets": {"max_props": 2},
            },
        },
        build_root=tmp_path,
    )

    assert result["ok"] is True
    assert result["planner_backend"] == "llm"
    assert result["selected_prompt"] == "small modern room"
    assert set(result["candidate_asset_ids"]) >= {"core_chair_01", "core_table_01"}


@pytest.mark.skipif(api_server.FastAPI is None, reason="FastAPI is not installed")
def test_fastapi_endpoint_success(tmp_path):
    from fastapi.testclient import TestClient

    client = TestClient(api_server.app)
    response = client.post(
        "/plan_and_compile",
        json={
            "prompt_text": "small indoor room with chair",
            "optional_seed": 7,
            "user_prefs": {"max_props": 2},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["world_id"].startswith("world_")


@pytest.mark.skipif(api_server.FastAPI is None, reason="FastAPI is not installed")
def test_fastapi_endpoint_invalid_request():
    from fastapi.testclient import TestClient

    client = TestClient(api_server.app)
    response = client.post("/plan_and_compile", json={"prompt_text": "   "})
    assert response.status_code == 400
    payload = response.json()
    assert payload["ok"] is False
    assert payload["error_code"] == "invalid_request"


@pytest.mark.skipif(api_server.FastAPI is None, reason="FastAPI is not installed")
def test_fastapi_endpoint_spawn_failure_is_422(monkeypatch):
    from fastapi.testclient import TestClient

    def fake_compile_phase0(*args, **kwargs):
        return {"ok": False, "errors": [{"path": "$.safe_spawn", "message": "no_safe_spawn_found"}]}

    monkeypatch.setattr(api_server, "compile_phase0", fake_compile_phase0)
    client = TestClient(api_server.app)
    response = client.post("/plan_and_compile", json={"prompt_text": "valid prompt"})
    assert response.status_code == 422
    payload = response.json()
    assert payload["error_code"] == "spawn_failed"
