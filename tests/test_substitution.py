from src.catalog.pack_registry import load_pack_registry
from src.selection.substitution import PLACEHOLDER_ASSET_ID, resolve_asset_or_substitute


# Keep behavior deterministic so planner/runtime contracts stay stable.
def _resolve(
    requested_asset_id: str,
    requested_tags: list[str],
    pack_ids: list[str],
    registry=None,
):
    return resolve_asset_or_substitute(
        requested_asset_id=requested_asset_id,
        requested_tags=requested_tags,
        pack_ids=pack_ids,
        registry=registry or load_pack_registry(),
    )


def test_resolve_asset_exact_when_present():
    result = _resolve("core_table_01", ["table", "indoor"], ["core_pack"])
    assert result["resolution_type"] == "exact"
    assert result["resolved_asset_id"] == "core_table_01"
    assert result["selection_backend"] == "exact"


def test_resolve_asset_exact_respects_selected_packs():
    result = _resolve("city_bench_01", ["bench", "outdoor"], ["core_pack"])
    assert result["resolution_type"] == "placeholder"
    assert result["resolved_asset_id"] == PLACEHOLDER_ASSET_ID
    assert result["rejected_candidate_counts"].get("rejected_for_pack_selection") == 1


def test_resolve_asset_exact_respects_coherence_hard_constraints():
    registry = load_pack_registry()
    registry.assets_by_id["core_table_01"]["asset"]["quest_compatible"] = False
    result = _resolve("core_table_01", ["table_only_token"], ["core_pack"], registry=registry)
    assert result["resolution_type"] == "placeholder"
    assert result["resolved_asset_id"] == PLACEHOLDER_ASSET_ID
    assert result["rejected_candidate_counts"].get("rejected_for_quest_compat") == 1


def test_resolve_asset_does_not_expand_to_all_packs_when_pack_filter_invalid():
    result = _resolve("core_unknown_chair_99", ["chair", "indoor"], ["pack_that_does_not_exist"])
    assert result["resolution_type"] == "placeholder"
    assert result["resolved_asset_id"] == PLACEHOLDER_ASSET_ID


def test_resolve_asset_substitute_uses_deterministic_best_match():
    result = _resolve("core_unknown_chair_99", ["chair", "indoor"], ["core_pack"])
    assert result["resolution_type"] == "substitute"
    assert result["resolved_asset_id"] == "core_chair_01"
    assert result["selection_backend"] == "deterministic"
    assert result["alternatives"] == ["core_chair_01"]


def test_resolve_asset_placeholder_when_no_match_exists():
    result = _resolve("core_unknown_rocket_99", ["rocket", "spaceship"], ["core_pack"])
    assert result["resolution_type"] == "placeholder"
    assert result["resolved_asset_id"] == PLACEHOLDER_ASSET_ID


def test_resolve_asset_rejects_cross_role_substitution():
    result = _resolve("core_unknown_bed_99", ["bed", "sleeping"], ["core_pack"])
    assert result["resolution_type"] == "placeholder"
    assert result["resolved_asset_id"] == PLACEHOLDER_ASSET_ID


def test_resolve_asset_can_passthrough_approved_planner_asset_when_compile_registry_lacks_it():
    result = resolve_asset_or_substitute(
        requested_asset_id="furniture_mega_pack/chair03-cf781b8c",
        requested_tags=["chair"],
        pack_ids=["core_pack"],
        registry=load_pack_registry(),
        requested_meta={"tags": ["chair"], "allow_passthrough_exact": True, "role": "chair", "label": "chair"},
    )
    assert result["resolution_type"] == "exact"
    assert result["resolved_asset_id"] == "furniture_mega_pack/chair03-cf781b8c"
    assert result["selection_backend"] == "passthrough"
