import pathlib

from src.stylekit_registry import load_stylekit_registry


FIXTURES_DIR = pathlib.Path(__file__).resolve().parent / "fixtures"


def test_stylekit_registry_loads_valid_manifests():
    registry = load_stylekit_registry(FIXTURES_DIR / "stylekits_valid")
    assert registry.list_stylekits() == ["moody_evening", "neutral_daylight"]
    assert registry.get_stylekit("neutral_daylight") is not None
    assert registry.errors == []


def test_stylekit_registry_rejects_invalid_manifest():
    registry = load_stylekit_registry(FIXTURES_DIR / "stylekits_invalid")
    assert registry.stylekits_by_id == {}
    assert registry.errors
    assert any("required property" in e["message"] for e in registry.errors)


def test_search_stylekits_by_tags():
    registry = load_stylekit_registry(FIXTURES_DIR / "stylekits_valid")
    assert registry.search_stylekits(["indoor"]) == ["moody_evening", "neutral_daylight"]
    assert registry.search_stylekits(["night"]) == ["moody_evening"]
    assert registry.search_stylekits(["missing"]) == []


def test_duplicate_stylekit_ids_are_skipped():
    registry = load_stylekit_registry(FIXTURES_DIR / "stylekits_duplicate")
    assert registry.list_stylekits() == ["dup_stylekit"]
    assert any("duplicate stylekit_id" in e["message"] for e in registry.errors)
