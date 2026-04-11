import pathlib

from src.catalog.stylekit_registry import load_stylekit_registry


FIXTURES_DIR = pathlib.Path(__file__).resolve().parent / "fixtures"


# Keep behavior deterministic so planner/runtime contracts stay stable.
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


def test_stylekit_registry_normalizes_polish_fields(tmp_path):
    stylekit_dir = tmp_path / "style"
    stylekit_dir.mkdir(parents=True)
    (stylekit_dir / "stylekit.json").write_text(
        """
        {
          "stylekit_id": "polished",
          "version": "0.1.0",
          "tags": ["night"],
          "lighting": {"preset": "evening", "intensity": 0.7},
          "palette": {"wall": "#111111", "floor": "#222222", "accent": "#333333"},
          "ambience": "low_hum",
          "decals": [{"decal_id": "edge_wear", "max_count": 3}],
          "postfx": {"profile_id": "moody_evening_lowcost"},
          "perf_overrides": {"allow_postfx": true}
        }
        """.strip(),
        encoding="utf-8",
    )

    registry = load_stylekit_registry(tmp_path)
    stylekit = registry.get_stylekit("polished")
    assert stylekit is not None
    assert stylekit["ambience"]["clip_id"] == "low_hum"
    assert stylekit["decals"][0]["decal_id"] == "edge_wear"
    assert stylekit["postfx"]["profile_id"] == "moody_evening_lowcost"
    assert stylekit["perf_overrides"]["allow_postfx"] is True


def test_invalid_stylekit_polish_reports_readable_error_path(tmp_path):
    stylekit_dir = tmp_path / "broken"
    stylekit_dir.mkdir(parents=True)
    (stylekit_dir / "stylekit.json").write_text(
        """
        {
          "stylekit_id": "broken",
          "version": "0.1.0",
          "tags": ["night"],
          "lighting": {"preset": "evening", "intensity": 0.7},
          "palette": {"wall": "#111111", "floor": "#222222", "accent": "#333333"},
          "decals": [{"decal_id": "edge_wear"}]
        }
        """.strip(),
        encoding="utf-8",
    )

    registry = load_stylekit_registry(tmp_path)
    assert registry.stylekits_by_id == {}
    assert registry.errors
    assert registry.errors[0]["path"].endswith("$.decals[0]")
