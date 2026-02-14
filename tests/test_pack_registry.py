import pathlib

from src.pack_registry import load_pack_registry


FIXTURES_DIR = pathlib.Path(__file__).resolve().parent / "fixtures"


def test_pack_registry_loads_valid_manifests():
    registry = load_pack_registry(FIXTURES_DIR / "packs_valid")
    assert sorted(registry.packs_by_id.keys()) == ["city_pack", "core_pack"]
    assert "core_chair_01" in registry.assets_by_id
    assert "city_bench_01" in registry.assets_by_id
    assert registry.errors == []


def test_pack_registry_rejects_invalid_manifest():
    registry = load_pack_registry(FIXTURES_DIR / "packs_invalid")
    assert registry.packs_by_id == {}
    assert registry.assets_by_id == {}
    assert registry.errors
    assert any("required property" in e["message"] for e in registry.errors)


def test_search_packs_by_tags_all_match():
    registry = load_pack_registry(FIXTURES_DIR / "packs_valid")
    assert registry.search_packs(["prototype"]) == ["city_pack", "core_pack"]
    assert registry.search_packs(["indoor", "basic"]) == ["core_pack"]
    assert registry.search_packs(["missing"]) == []


def test_duplicate_asset_ids_are_skipped():
    registry = load_pack_registry(FIXTURES_DIR / "packs_duplicate")
    assert sorted(registry.packs_by_id.keys()) == ["pack_a", "pack_b"]
    assert "shared_asset_01" in registry.assets_by_id
    assert registry.assets_by_id["shared_asset_01"]["pack_id"] in {"pack_a", "pack_b"}
    assert any("duplicate asset_id" in e["message"] for e in registry.errors)
