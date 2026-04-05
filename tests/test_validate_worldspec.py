import json
import pathlib

from src.world.validation import validate_worldspec


FIXTURES_DIR = pathlib.Path(__file__).resolve().parent / "fixtures"


def _load(name: str):
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


def test_worldspec_valid():
    data = _load("worldspec_valid.json")
    result = validate_worldspec(data)
    assert result["ok"] is True
    assert result["errors"] == []


def test_worldspec_missing_required_field():
    data = _load("worldspec_invalid_missing_field.json")
    result = validate_worldspec(data)
    assert result["ok"] is False
    assert any("worldspec_version" in e["message"] for e in result["errors"])


def test_worldspec_invalid_type():
    data = _load("worldspec_invalid_type.json")
    result = validate_worldspec(data)
    assert result["ok"] is False
    assert any("seed" in e["path"] for e in result["errors"])


def test_worldspec_invalid_placement():
    data = _load("worldspec_invalid_placement.json")
    result = validate_worldspec(data)
    assert result["ok"] is False
    assert any("asset_id" in e["message"] for e in result["errors"])


def test_worldspec_near_constraint_requires_target():
    data = _load("worldspec_valid.json")
    data["placements"][0]["constraint"] = {"type": "near"}
    result = validate_worldspec(data)
    assert result["ok"] is False
    assert any(error["path"] == "$.placements[0].constraint.target" for error in result["errors"])


def test_worldspec_near_constraint_requires_numeric_distance():
    data = _load("worldspec_valid.json")
    data["placements"][0]["constraint"] = {"type": "near", "target": "table", "distance": "close"}
    result = validate_worldspec(data)
    assert result["ok"] is False
    assert any(error["path"] == "$.placements[0].constraint.distance" for error in result["errors"])
