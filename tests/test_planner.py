from src.pack_registry import PackRegistry, load_pack_registry
from src.planner import _build_placements, plan_worldspec
from src.stylekit_registry import StyleKitRegistry, load_stylekit_registry
from src.validate_worldspec import validate_worldspec


def test_planner_known_prompt_yields_valid_worldspec_with_placements():
    result = plan_worldspec("Build a moody city street at night with bench and lamp", seed=42)
    assert result["ok"] is True
    assert result["errors"] == []
    assert result["worldspec"]["placements"]

    validation = validate_worldspec(result["worldspec"])
    assert validation["ok"] is True


def test_planner_unknown_prompt_falls_back_and_still_validates():
    result = plan_worldspec("qwertyuiop asdfghjk")
    assert result["ok"] is True

    spec = result["worldspec"]
    assert spec["pack_ids"] == ["core_pack"]
    assert spec["stylekit_id"]
    assert validate_worldspec(spec)["ok"] is True


def test_planner_never_emits_unknown_ids():
    pack_registry = load_pack_registry()
    style_registry = load_stylekit_registry()

    result = plan_worldspec("outdoor city with lamp and sign", seed=7)
    assert result["ok"] is True
    spec = result["worldspec"]

    for pack_id in spec["pack_ids"]:
        assert pack_id in pack_registry.packs_by_id
    assert spec["stylekit_id"] in style_registry.stylekits_by_id

    known_assets = set(pack_registry.assets_by_id.keys())
    for placement in spec["placements"]:
        assert placement["asset_id"] in known_assets


def test_planner_is_deterministic_for_same_prompt_and_seed():
    prompt = "indoor room with chair table lamp"
    first = plan_worldspec(prompt, seed=12345)
    second = plan_worldspec(prompt, seed=12345)
    assert first["ok"] is True
    assert second["ok"] is True
    assert first["worldspec"] == second["worldspec"]


def test_planner_prefers_high_confidence_quest_safe_assets_when_rich_metadata_present():
    candidate_assets = [
        {
            "asset_id": "unsafe_shell",
            "label": "chair",
            "tags": ["chair"],
            "classification": "shell",
            "quest_compatible": True,
            "semantic_confidence": 0.99,
        },
        {
            "asset_id": "unsafe_conf",
            "label": "chair",
            "tags": ["chair"],
            "classification": "prop",
            "quest_compatible": True,
            "semantic_confidence": 0.40,
        },
        {
            "asset_id": "safe_chair",
            "label": "chair",
            "tags": ["chair"],
            "classification": "prop",
            "quest_compatible": True,
            "semantic_confidence": 0.95,
        },
    ]

    placements = _build_placements(candidate_assets, {"chair"}, seed=1, max_props=3)
    assert [p["asset_id"] for p in placements] == ["safe_chair"]


def test_planner_rejects_all_unsafe_assets_when_rich_metadata_present():
    candidate_assets = [
        {
            "asset_id": "unsafe_shell",
            "label": "chair",
            "tags": ["chair"],
            "classification": "shell",
            "quest_compatible": True,
            "semantic_confidence": 0.99,
        },
        {
            "asset_id": "unsafe_conf",
            "label": "chair",
            "tags": ["chair"],
            "classification": "prop",
            "quest_compatible": True,
            "semantic_confidence": 0.20,
        },
    ]

    placements = _build_placements(candidate_assets, {"chair"}, seed=1, max_props=3)
    assert placements == []


def test_plan_worldspec_errors_when_only_unsafe_assets_available(monkeypatch):
    unsafe_registry = PackRegistry(
        packs_by_id={
            "core_pack": {
                "pack_id": "core_pack",
                "tags": ["indoor"],
                "assets": [
                    {
                        "asset_id": "unsafe_shell",
                        "label": "chair",
                        "tags": ["chair"],
                        "classification": "shell",
                        "quest_compatible": True,
                        "semantic_confidence": 0.95,
                    }
                ],
            }
        },
        assets_by_id={"unsafe_shell": {"pack_id": "core_pack", "asset": {"asset_id": "unsafe_shell"}}},
        tags_index={"core_pack": ["indoor"]},
        errors=[],
    )
    style_registry = StyleKitRegistry(
        stylekits_by_id={
            "neutral_daylight": {
                "stylekit_id": "neutral_daylight",
                "tags": ["neutral", "indoor", "day"],
                "lighting": {"preset": "daylight_soft", "intensity": 1.0},
                "palette": {"wall": "#d8d8d8", "floor": "#8b7d6b", "accent": "#4a90e2"},
            }
        },
        tags_index={"neutral_daylight": ["neutral", "indoor", "day"]},
        errors=[],
    )

    monkeypatch.setattr("src.planner.load_pack_registry", lambda: unsafe_registry)
    monkeypatch.setattr("src.planner.load_stylekit_registry", lambda: style_registry)

    result = plan_worldspec("indoor chair prompt", seed=12)
    assert result["ok"] is False
    assert any(err["path"] == "$.placements" for err in result["errors"])
