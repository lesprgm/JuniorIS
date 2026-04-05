import json
import os
from pathlib import Path

import pytest

from src.api.server import run_plan_and_compile


def _live_llm_tests_enabled() -> bool:
    raw = str(os.getenv("RUN_LIVE_LLM_TESTS", "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _has_live_llm_config() -> bool:
    return bool(os.getenv("GEMINI_API_KEY")) and bool(os.getenv("GEMINI_MODEL"))


LIVE_TESTS_REASON = "Set RUN_LIVE_LLM_TESTS=1 with GEMINI_API_KEY and GEMINI_MODEL to run live LLM E2E tests."


@pytest.mark.skipif(not _live_llm_tests_enabled(), reason=LIVE_TESTS_REASON)
@pytest.mark.parametrize(
    "prompt_text",
    [
        "a cozy reading corner with one chair and one lamp",
        "a minimal office setup with one table and one chair",
    ],
)
def test_live_llm_plan_and_compile_produces_non_stacked_floor_layout(tmp_path: Path, prompt_text: str):
    if not _has_live_llm_config():
        pytest.skip(LIVE_TESTS_REASON)

    result = run_plan_and_compile(
        prompt_text=prompt_text,
        optional_seed=123,
        user_prefs={"llm_required": True},
        build_root=tmp_path,
    )

    assert result.get("ok") is True, json.dumps(result, indent=2)
    assert result.get("planner_backend") == "llm", json.dumps(result, indent=2)
    assert isinstance(result.get("world_id"), str) and result["world_id"]

    phase0_path = tmp_path / result["world_id"] / "phase0.json"
    assert phase0_path.exists(), f"Missing phase0 artifact at {phase0_path}"

    phase0_data = json.loads(phase0_path.read_text(encoding="utf-8"))
    placements = phase0_data.get("placements", [])
    assert isinstance(placements, list) and placements, "Expected at least one placement in phase0 artifact."

    constraints = phase0_data.get("constraints", {})
    assert constraints.get("floor_anchored_only") is True
    assert constraints.get("stacking_enabled") is False

    dimensions = phase0_data.get("template", {}).get("dimensions", {})
    width = float(dimensions.get("width", 0.0))
    length = float(dimensions.get("length", 0.0))
    max_x = width / 2.0
    max_z = length / 2.0

    occupied_floor_slots = set()
    for placement in placements:
        transform = placement.get("transform", {})
        pos = transform.get("pos", [])
        assert isinstance(pos, list) and len(pos) == 3
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])

        # MVP invariant: objects are floor-anchored, not vertically stacked.
        assert y == pytest.approx(0.0)
        floor_key = (round(x, 3), round(z, 3))
        assert floor_key not in occupied_floor_slots, f"Detected stacked/overlapping floor slot at {floor_key}"
        occupied_floor_slots.add(floor_key)

        # Placement must stay in room bounds after compiler clamping.
        assert -max_x <= x <= max_x
        assert -max_z <= z <= max_z
