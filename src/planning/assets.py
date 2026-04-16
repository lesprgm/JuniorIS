from __future__ import annotations

"""Asset helpers split into catalog, shortlist, and layout modules."""

from src.planning.asset_catalog import (
    PLANNER_POOL_PATH,
    asset_matches_pack,
    candidate_assets_by_ids,
    collect_assets,
    load_planner_pool,
)
from src.planning.asset_layout import (
    ROOM_BASIC_DIMENSIONS,
    build_optional_raw_placements,
    build_layout_from_selected_assets,
    build_layout_inputs_from_selected_assets,
)
from src.planning.asset_shortlist import (
    build_semantic_candidate_shortlist,
    filter_candidate_assets,
)

__all__ = [
    "PLANNER_POOL_PATH",
    "ROOM_BASIC_DIMENSIONS",
    "build_optional_raw_placements",
    "asset_matches_pack",
    "build_layout_from_selected_assets",
    "build_layout_inputs_from_selected_assets",
    "build_semantic_candidate_shortlist",
    "candidate_assets_by_ids",
    "collect_assets",
    "filter_candidate_assets",
    "load_planner_pool",
]

# Keep behavior deterministic so planner/runtime contracts stay stable.
