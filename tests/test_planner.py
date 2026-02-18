from src.pack_registry import load_pack_registry
from src.planner import plan_worldspec
from src.stylekit_registry import load_stylekit_registry
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
