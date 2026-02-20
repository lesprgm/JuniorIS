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

    artifact_path = pathlib.Path(result["phase0_artifact"])
    assert artifact_path.exists()

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["phase"] == "phase0"
    assert artifact["world_id"] == result["world_id"]
    assert any(
        node["id"] == "floor" and node["teleportable"] is True
        for node in artifact["template"]["nodes"]
    )


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
