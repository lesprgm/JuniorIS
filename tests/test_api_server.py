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

    manifest_path = tmp_path / result["world_id"] / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["world_id"] == result["world_id"]
    assert manifest["portal_ready_at_phase"] == "phase0"
    assert "safe_spawn" in manifest


def test_run_plan_and_compile_rejects_empty_prompt():
    result = api_server.run_plan_and_compile(prompt_text="", build_root="build_test")
    assert result["ok"] is False
    assert result["error_code"] == "invalid_request"
    assert any(error["path"] == "$.prompt_text" for error in result["errors"])


def test_run_plan_and_compile_maps_planner_failures(monkeypatch, tmp_path):
    def fake_plan_worldspec(*args, **kwargs):
        return {"ok": False, "errors": [{"path": "$.pack_registry", "message": "missing"}]}

    monkeypatch.setattr(api_server, "plan_worldspec", fake_plan_worldspec)
    result = api_server.run_plan_and_compile("test prompt", build_root=tmp_path)
    assert result["ok"] is False
    assert result["error_code"] == "planner_failed"


def test_run_plan_and_compile_maps_compile_failures(monkeypatch, tmp_path):
    def fake_compile_phase0(*args, **kwargs):
        return {"ok": False, "errors": [{"path": "$.safe_spawn", "message": "no_safe_spawn_found"}]}

    monkeypatch.setattr(api_server, "compile_phase0", fake_compile_phase0)
    result = api_server.run_plan_and_compile("test prompt", build_root=tmp_path)
    assert result["ok"] is False
    assert result["error_code"] == "compile_failed"


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
