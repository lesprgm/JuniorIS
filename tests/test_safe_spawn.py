import json
import pathlib

from src.compilation.phase0 import compile_phase0
from src.runtime.safe_spawn import find_safe_spawn


FIXTURES_DIR = pathlib.Path(__file__).resolve().parent / "fixtures"


def _load(name: str):
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


def _base_phase0_data():
    worldspec = _load("worldspec_phase0_valid.json")
    compiled = compile_phase0(worldspec, write_artifact=False)
    assert compiled["ok"] is True
    return compiled["phase0_data"]


def test_find_safe_spawn_accepts_center_when_unblocked():
    phase0_data = _base_phase0_data()
    phase0_data["placements"] = []

    result = find_safe_spawn(phase0_data)
    assert result["ok"] is True
    assert result["spawn"]["pos"] == [0.0, 0.0, 0.0]
    assert result["attempts"] == 1
    assert result["reason"] == ""


def test_find_safe_spawn_uses_fallback_when_center_is_blocked():
    phase0_data = _base_phase0_data()
    phase0_data["placements"] = [
        {
            "asset_id": "core_blocker",
            "transform": {
                "pos": [0.0, 0.0, 0.0],
                "rot": [0.0, 0.0, 0.0],
                "scale": [4.0, 1.0, 4.0],
            },
        }
    ]

    result = find_safe_spawn(phase0_data)
    assert result["ok"] is True
    assert result["spawn"]["pos"] != [0.0, 0.0, 0.0]
    assert result["attempts"] > 1
    assert result["reason"] == ""


def test_find_safe_spawn_is_deterministic_for_same_input():
    phase0_data = _base_phase0_data()
    phase0_data["placements"] = [
        {
            "asset_id": "core_blocker",
            "transform": {
                "pos": [0.0, 0.0, 0.0],
                "rot": [0.0, 0.0, 0.0],
                "scale": [2.0, 1.0, 2.0],
            },
        }
    ]

    first = find_safe_spawn(phase0_data)
    second = find_safe_spawn(phase0_data)
    assert first == second


def test_find_safe_spawn_returns_failure_when_all_candidates_are_blocked():
    phase0_data = _base_phase0_data()
    phase0_data["placements"] = [
        {
            "asset_id": "core_blocker",
            "transform": {
                "pos": [0.0, 0.0, 0.0],
                "rot": [0.0, 0.0, 0.0],
                "scale": [80.0, 1.0, 80.0],
            },
        }
    ]

    result = find_safe_spawn(phase0_data)
    assert result["ok"] is False
    assert result["spawn"] is None
    assert result["reason"] == "no_safe_spawn_found"
    assert result["attempts"] > 0
