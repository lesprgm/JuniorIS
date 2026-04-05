from __future__ import annotations

from src.api.server import run_plan_and_compile
from src.contracts.runtime import validate_api_response_contract
from tests.semantic_test_utils import inline_semantic_prefs


def test_success_response_matches_api_contract_schema(tmp_path):
    payload = run_plan_and_compile(
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
    payload = run_plan_and_compile("", build_root="build_test")
    result = validate_api_response_contract(payload)
    assert result["ok"] is True, result["errors"]
