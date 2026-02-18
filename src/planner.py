from __future__ import annotations

import hashlib
import random
import re
from typing import Any, Dict, List, Tuple

from src.pack_registry import PackRegistry, load_pack_registry
from src.stylekit_registry import StyleKitRegistry, load_stylekit_registry
from src.validate_worldspec import validate_worldspec


TEMPLATE_ALLOWLIST = ["room_basic"]
DEFAULT_TEMPLATE_ID = "room_basic"
DEFAULT_STYLEKIT_ID = "neutral_daylight"
DEFAULT_PACK_ID = "core_pack"

DEFAULT_BUDGETS = {
    "max_props": 4,
    "max_texture_tier": 1,
    "max_lights": 2,
}

# Small, deterministic floor slots (no stacking in MVP).
FLOOR_SLOTS: List[Tuple[float, float]] = [
    (-1.75, -1.25),
    (1.65, -1.10),
    (-1.40, 0.95),
    (1.35, 1.15),
    (0.00, -1.55),
    (0.15, 1.55),
]


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _seed_from_prompt(prompt_text: str) -> int:
    digest = hashlib.sha256(prompt_text.strip().encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def _select_pack_ids(tokens: set[str], registry: PackRegistry) -> List[str]:
    desired_tags = []
    if tokens & {"outdoor", "city", "street", "plaza", "park"}:
        desired_tags.append("outdoor")
    elif tokens & {"indoor", "room", "office", "studio", "living"}:
        desired_tags.append("indoor")

    if desired_tags:
        matches = registry.search_packs(desired_tags)
        if matches:
            return [matches[0]]

    if DEFAULT_PACK_ID in registry.packs_by_id:
        return [DEFAULT_PACK_ID]
    all_packs = sorted(registry.packs_by_id.keys())
    return [all_packs[0]] if all_packs else []


def _stylekit_score(stylekit: Dict[str, Any], tokens: set[str]) -> int:
    score = 0
    style_tags = set(stylekit.get("tags", []))
    if style_tags:
        score += len(style_tags & tokens)
    lighting = stylekit.get("lighting", {})
    preset = str(lighting.get("preset", "")).lower()
    score += int(any(word in preset for word in tokens))
    return score


def _select_stylekit_id(tokens: set[str], registry: StyleKitRegistry) -> str | None:
    if not registry.stylekits_by_id:
        return None
    if DEFAULT_STYLEKIT_ID in registry.stylekits_by_id:
        fallback = DEFAULT_STYLEKIT_ID
    else:
        fallback = sorted(registry.stylekits_by_id.keys())[0]

    scored = sorted(
        registry.stylekits_by_id.items(),
        key=lambda kv: (-_stylekit_score(kv[1], tokens), kv[0]),
    )
    if not scored:
        return fallback
    best_score = _stylekit_score(scored[0][1], tokens)
    if best_score <= 0:
        return fallback
    return scored[0][0]


def _collect_assets(pack_ids: List[str], registry: PackRegistry) -> List[Dict[str, Any]]:
    assets: List[Dict[str, Any]] = []
    for pack_id in pack_ids:
        pack = registry.packs_by_id.get(pack_id)
        if not pack:
            continue
        for asset in pack.get("assets", []):
            assets.append(asset)
    return assets


def _asset_score(asset: Dict[str, Any], tokens: set[str]) -> int:
    tag_hits = len(set(asset.get("tags", [])) & tokens)
    label_tokens = _tokenize(str(asset.get("label", "")))
    label_hits = len(label_tokens & tokens)
    return (tag_hits * 2) + label_hits


def _build_placements(
    candidate_assets: List[Dict[str, Any]],
    tokens: set[str],
    seed: int,
    max_props: int,
) -> List[Dict[str, Any]]:
    if not candidate_assets:
        return []

    ranked_assets = sorted(
        candidate_assets,
        key=lambda asset: (-_asset_score(asset, tokens), asset["asset_id"]),
    )

    chosen = ranked_assets[: max(1, min(max_props, len(ranked_assets), len(FLOOR_SLOTS)))]
    rng = random.Random(seed)
    slots = FLOOR_SLOTS[:]
    rng.shuffle(slots)

    placements = []
    for index, asset in enumerate(chosen):
        x, z = slots[index]
        yaw = rng.choice([0, 45, 90, 135, 180, 225, 270, 315])
        placements.append(
            {
                "asset_id": asset["asset_id"],
                "transform": {
                    "pos": [round(x, 2), 0, round(z, 2)],
                    "rot": [0, yaw, 0],
                    "scale": [1, 1, 1],
                },
            }
        )

    return placements


def plan_worldspec(
    prompt_text: str,
    seed: int | None = None,
    user_prefs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    prompt_text = (prompt_text or "").strip()
    user_prefs = user_prefs or {}
    tokens = _tokenize(prompt_text)

    effective_seed = seed if seed is not None else _seed_from_prompt(prompt_text or "default")
    registry = load_pack_registry()
    style_registry = load_stylekit_registry()

    if not registry.packs_by_id:
        return {
            "ok": False,
            "errors": [
                {
                    "path": "$.pack_registry",
                    "message": "No valid packs loaded; planner cannot select assets.",
                }
            ],
        }
    if not style_registry.stylekits_by_id:
        return {
            "ok": False,
            "errors": [
                {
                    "path": "$.stylekit_registry",
                    "message": "No valid stylekits loaded; planner cannot select style.",
                }
            ],
        }

    template_id = DEFAULT_TEMPLATE_ID if DEFAULT_TEMPLATE_ID in TEMPLATE_ALLOWLIST else TEMPLATE_ALLOWLIST[0]
    stylekit_id = _select_stylekit_id(tokens, style_registry)
    pack_ids = _select_pack_ids(tokens, registry)

    budgets = dict(DEFAULT_BUDGETS)
    for key in ("max_props", "max_texture_tier", "max_lights"):
        value = user_prefs.get(key)
        if isinstance(value, int) and value > 0:
            budgets[key] = value

    assets = _collect_assets(pack_ids, registry)
    placements = _build_placements(assets, tokens, effective_seed, budgets["max_props"])

    colors = {}
    if stylekit_id:
        stylekit = style_registry.get_stylekit(stylekit_id) or {}
        palette = stylekit.get("palette", {})
        if isinstance(palette, dict):
            colors = {
                "wall": palette.get("wall", "#d8d8d8"),
                "floor": palette.get("floor", "#8b7d6b"),
                "ceiling": palette.get("wall", "#d8d8d8"),
                "accent": palette.get("accent", "#4a90e2"),
            }

    worldspec: Dict[str, Any] = {
        "worldspec_version": "0.1",
        "template_id": template_id,
        "seed": int(effective_seed),
        "stylekit_id": stylekit_id,
        "pack_ids": pack_ids,
        "placements": placements,
        "budgets": budgets,
    }
    if colors:
        worldspec["colors"] = colors

    validation = validate_worldspec(worldspec)
    if not validation["ok"]:
        return {"ok": False, "errors": validation["errors"], "worldspec": worldspec}

    return {"ok": True, "worldspec": worldspec, "errors": []}
