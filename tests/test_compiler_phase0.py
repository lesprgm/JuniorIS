import json
import pathlib

from src.compiler_phase0 import compile_phase0


FIXTURES_DIR = pathlib.Path(__file__).resolve().parent / "fixtures"


def _load(name: str):
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


def test_compile_phase0_success_writes_artifact(tmp_path):
    worldspec = _load("worldspec_phase0_valid.json")
    result = compile_phase0(worldspec, build_root=tmp_path)

    assert result["ok"] is True
    assert result["errors"] == []
    assert result["teleportable_surfaces"] == 1
    assert result["phase0_artifact"] is not None
    assert result["safe_spawn"] is not None

    artifact_path = pathlib.Path(result["phase0_artifact"])
    assert artifact_path.exists()

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["phase"] == "phase0"
    assert artifact["world_id"] == result["world_id"]
    assert any(
        node["id"] == "floor" and node["teleportable"] is True
        for node in artifact["template"]["nodes"]
    )
    assert artifact["safe_spawn"]["pos"][1] == 0.0


def test_compile_phase0_invalid_worldspec_returns_structured_errors():
    invalid_worldspec = _load("worldspec_invalid_type.json")
    result = compile_phase0(invalid_worldspec, write_artifact=False)

    assert result["ok"] is False
    assert result["phase0_artifact"] is None
    assert result["errors"]
    assert any("seed" in error["path"] for error in result["errors"])


def test_compile_phase0_is_deterministic_for_same_input(tmp_path):
    worldspec = _load("worldspec_phase0_valid.json")
    first = compile_phase0(worldspec, build_root=tmp_path / "run_a")
    second = compile_phase0(worldspec, build_root=tmp_path / "run_b")

    assert first["ok"] is True
    assert second["ok"] is True
    assert first["world_id"] == second["world_id"]
    assert first["phase0_data"] == second["phase0_data"]
    assert first["safe_spawn"] == second["safe_spawn"]


def test_compile_phase0_clamps_out_of_bounds_floor_positions(tmp_path):
    worldspec = _load("worldspec_phase0_valid.json")
    result = compile_phase0(worldspec, build_root=tmp_path)
    assert result["ok"] is True

    dimensions = result["phase0_data"]["template"]["dimensions"]
    max_x = (dimensions["width"] / 2.0) - 0.25
    max_z = (dimensions["length"] / 2.0) - 0.25

    clamped = next(
        placement
        for placement in result["phase0_data"]["placements"]
        if placement["asset_id"] == "core_lamp_01"
    )
    pos = clamped["transform"]["pos"]

    assert -max_x <= pos[0] <= max_x
    assert pos[1] == 0.0
    assert -max_z <= pos[2] <= max_z


def test_compile_phase0_returns_safe_spawn_failure_when_room_is_fully_blocked():
    worldspec = _load("worldspec_phase0_valid.json")
    worldspec["placements"] = [
        {
            "asset_id": "core_table_01",
            "transform": {
                "pos": [0.0, 0.0, 0.0],
                "rot": [0.0, 0.0, 0.0],
                "scale": [64.0, 1.0, 64.0],
            },
        }
    ]

    result = compile_phase0(worldspec, write_artifact=False)
    assert result["ok"] is False
    assert result["phase0_artifact"] is None
    assert result["safe_spawn"] is None
    assert any(error["path"] == "$.safe_spawn" for error in result["errors"])


def test_compile_phase0_substitutes_missing_asset_and_reports_it(tmp_path):
    worldspec = _load("worldspec_phase0_valid.json")
    worldspec["placements"] = [
        {
            "asset_id": "core_unknown_chair_42",
            "tags": ["chair", "indoor"],
            "transform": {
                "pos": [0.0, 0.0, 0.0],
                "rot": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            },
        }
    ]
    worldspec["pack_ids"] = ["core_pack"]

    result = compile_phase0(worldspec, build_root=tmp_path)
    assert result["ok"] is True

    placement = result["phase0_data"]["placements"][0]
    assert placement["requested_asset_id"] == "core_unknown_chair_42"
    assert placement["asset_id"] == "core_chair_01"
    assert placement["resolution_type"] == "substitute"
    assert placement["mode"] == "asset"

    report = result["phase0_data"]["substitution_report"]
    assert report["substitution_count"] == 1
    assert report["resolution_counts"]["substitute"] == 1
    assert report["substitutions"][0]["requested_asset_id"] == "core_unknown_chair_42"


def test_compile_phase0_marks_placeholder_mode_for_placeholder_resolution(tmp_path):
    worldspec = _load("worldspec_phase0_valid.json")
    worldspec["placements"] = [
        {
            "asset_id": "core_unknown_rocket_99",
            "tags": ["rocket", "spaceship"],
            "transform": {
                "pos": [0.0, 0.0, 0.0],
                "rot": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            },
        }
    ]
    worldspec["pack_ids"] = ["core_pack"]

    result = compile_phase0(worldspec, build_root=tmp_path)
    assert result["ok"] is True
    placement = result["phase0_data"]["placements"][0]
    assert placement["resolution_type"] == "placeholder"
    assert placement["mode"] == "placeholder"


def test_compile_phase0_passes_stylekit_theme_into_substitution(monkeypatch, tmp_path):
    worldspec = _load("worldspec_phase0_valid.json")
    worldspec["placements"] = [
        {
            "asset_id": "core_table_01",
            "transform": {
                "pos": [0.0, 0.0, 0.0],
                "rot": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            },
        }
    ]

    captured = {}

    def _fake_resolve(**kwargs):
        captured["room_theme"] = kwargs.get("room_theme")
        return {
            "resolved_asset_id": kwargs["requested_asset_id"],
            "resolution_type": "exact",
            "reason": "asset_found",
            "coherence_checks": {
                "visual_style_match": True,
                "poly_style_match": True,
                "theme_overlap_match": True,
            },
            "rejected_candidate_counts": {},
        }

    monkeypatch.setattr("src.compiler_phase0.resolve_asset_or_substitute", _fake_resolve)

    result = compile_phase0(worldspec, build_root=tmp_path)
    assert result["ok"] is True
    assert "neutral" in (captured.get("room_theme") or {}).get("style_tags", [])
