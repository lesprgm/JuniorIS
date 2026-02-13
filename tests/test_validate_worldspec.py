import json
import pathlib

from src.validate_worldspec import validate_worldspec


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
