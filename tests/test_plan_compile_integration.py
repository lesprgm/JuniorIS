from __future__ import annotations

import json
from pathlib import Path

from src.api.server import run_plan_and_compile


GOLDEN_PROMPT = "small indoor room with table and lamp"


# Keep behavior deterministic so planner/runtime contracts stay stable.
def test_golden_prompt_compile_with_real_registries(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "src.planning.planner.request_llm_intent",
        lambda *args, **kwargs: {"ok": False, "error_code": "llm_transport_error", "message": "disabled for test"},
    )

    result = run_plan_and_compile(
        prompt_text=GOLDEN_PROMPT,
        optional_seed=123,
        user_prefs={},
        build_root=tmp_path,
    )

    assert result["ok"] is False, json.dumps(result, indent=2)
    assert result["error_code"] == "planner_failed"
    assert result["planner_error_code"] == "llm_transport_error"
    assert result["planner_backend"] == "llm_unavailable"
    assert result["semantic_path_status"] == "failed"
    assert result["request_id"].startswith("req_")
    assert result["trace_id"].startswith("trace_")
    assert "world_id" not in result or result["world_id"] is None


def test_golden_prompt_failure_is_stable(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "src.planning.planner.request_llm_intent",
        lambda *args, **kwargs: {"ok": False, "error_code": "llm_transport_error", "message": "disabled for test"},
    )

    first = run_plan_and_compile(
        GOLDEN_PROMPT,
        optional_seed=123,
        user_prefs={},
        build_root=tmp_path / "a",
    )
    second = run_plan_and_compile(
        GOLDEN_PROMPT,
        optional_seed=123,
        user_prefs={},
        build_root=tmp_path / "b",
    )

    assert first["ok"] is False
    assert second["ok"] is False
    assert first["planner_error_code"] == second["planner_error_code"] == "llm_transport_error"
