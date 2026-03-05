from src.pack_registry import load_pack_registry
from src.substitution import PLACEHOLDER_ASSET_ID, resolve_asset_or_substitute


def test_resolve_asset_exact_when_present():
    registry = load_pack_registry()
    result = resolve_asset_or_substitute(
        requested_asset_id="core_table_01",
        requested_tags=["table", "indoor"],
        pack_ids=["core_pack"],
        registry=registry,
    )
    assert result["resolution_type"] == "exact"
    assert result["resolved_asset_id"] == "core_table_01"


def test_resolve_asset_exact_respects_selected_packs():
    registry = load_pack_registry()
    result = resolve_asset_or_substitute(
        requested_asset_id="city_bench_01",
        requested_tags=["bench", "outdoor"],
        pack_ids=["core_pack"],
        registry=registry,
    )
    assert result["resolution_type"] == "placeholder"
    assert result["resolved_asset_id"] == PLACEHOLDER_ASSET_ID
    assert result["rejected_candidate_counts"].get("rejected_for_pack_selection") == 1


def test_resolve_asset_exact_respects_coherence_hard_constraints():
    registry = load_pack_registry()
    registry.assets_by_id["core_table_01"]["asset"]["quest_compatible"] = False
    result = resolve_asset_or_substitute(
        requested_asset_id="core_table_01",
        requested_tags=["table_only_token"],
        pack_ids=["core_pack"],
        registry=registry,
    )
    assert result["resolution_type"] == "placeholder"
    assert result["resolved_asset_id"] == PLACEHOLDER_ASSET_ID
    assert result["rejected_candidate_counts"].get("rejected_for_quest_compat") == 1


def test_resolve_asset_does_not_expand_to_all_packs_when_pack_filter_invalid():
    registry = load_pack_registry()
    result = resolve_asset_or_substitute(
        requested_asset_id="core_unknown_chair_99",
        requested_tags=["chair", "indoor"],
        pack_ids=["pack_that_does_not_exist"],
        registry=registry,
    )
    assert result["resolution_type"] == "placeholder"
    assert result["resolved_asset_id"] == PLACEHOLDER_ASSET_ID


def test_resolve_asset_substitute_when_missing_but_tag_match_exists():
    registry = load_pack_registry()
    result = resolve_asset_or_substitute(
        requested_asset_id="core_unknown_chair_99",
        requested_tags=["chair", "indoor"],
        pack_ids=["core_pack"],
        registry=registry,
    )
    assert result["resolution_type"] == "substitute"
    assert result["resolved_asset_id"] == "core_chair_01"


def test_resolve_asset_placeholder_when_no_match_exists():
    registry = load_pack_registry()
    result = resolve_asset_or_substitute(
        requested_asset_id="core_unknown_rocket_99",
        requested_tags=["rocket", "spaceship"],
        pack_ids=["core_pack"],
        registry=registry,
    )
    assert result["resolution_type"] == "placeholder"
    assert result["resolved_asset_id"] == PLACEHOLDER_ASSET_ID
