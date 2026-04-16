from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List

from src.catalog.pack_registry import PackRegistry


PLANNER_POOL_PATH = pathlib.Path("data/index/planner_asset_pool_v1.json")  # pre-indexed pool of planner-eligible assets


# Keep behavior deterministic so planner/runtime contracts stay stable.
def load_planner_pool() -> List[Dict[str, Any]]:  # loads all indexed assets; returns empty list if file is missing or corrupt
    if not PLANNER_POOL_PATH.exists():
        return []
    try:
        payload = json.loads(PLANNER_POOL_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    assets = payload.get("assets")
    if not isinstance(assets, list):
        return []
    return [dict(asset) for asset in assets if isinstance(asset, dict) and asset.get("asset_id")]


def asset_matches_pack(asset: Dict[str, Any], pack_ids: List[str], registry: PackRegistry) -> bool:  # checks if an asset belongs to one of the selected packs
    if not pack_ids:
        return True

    pack_id = asset.get("pack_id")
    if isinstance(pack_id, str) and pack_id in pack_ids:
        return True

    source_pack = str(asset.get("source_pack", "")).strip()
    if source_pack in pack_ids:
        return True
    source_pack_lower = source_pack.lower()

    for candidate_pack_id in pack_ids:
        pack = registry.packs_by_id.get(candidate_pack_id)
        if not isinstance(pack, dict):
            continue
        if source_pack_lower == str(pack.get("pack_id", "")).strip().lower():
            return True
        manifest_path = str(pack.get("_manifest_path", "")).lower()
        if manifest_path and source_pack_lower in manifest_path:
            return True
    return False


def collect_assets(pack_ids: List[str], registry: PackRegistry) -> List[Dict[str, Any]]:  # filters the planner pool to only assets matching selected packs
    return [asset for asset in load_planner_pool() if asset_matches_pack(asset, pack_ids, registry)]


def candidate_assets_by_ids(all_assets: List[Dict[str, Any]], asset_ids: List[str]) -> List[Dict[str, Any]]:  # resolves a list of asset IDs to full asset records, preserving order
    by_id = {str(asset.get("asset_id")): asset for asset in all_assets if asset.get("asset_id")}
    out: List[Dict[str, Any]] = []
    for asset_id in asset_ids:
        asset = by_id.get(str(asset_id))
        if asset is not None:
            out.append(asset)
    return out
