from __future__ import annotations

from src.pack_registry import load_pack_registry
from src.planner_assets import build_semantic_candidate_shortlist, collect_assets


def shortlist_asset_ids(prompt_text: str, *roles: str) -> list[str]:
    registry = load_pack_registry()
    shortlist = build_semantic_candidate_shortlist(
        collect_assets([], registry),
        prompt_text,
        limit=40,
    )

    selected_ids: list[str] = []
    seen_ids: set[str] = set()
    for role in roles:
        for asset in shortlist:
            asset_id = str(asset.get("asset_id", ""))
            label = str(asset.get("label", "")).lower()
            tags = {str(tag).lower() for tag in asset.get("tags", []) if isinstance(tag, str)}
            if not asset_id or asset_id in seen_ids:
                continue
            if role in tags or role in label:
                seen_ids.add(asset_id)
                selected_ids.append(asset_id)
                break
    return selected_ids


def inline_semantic_prefs(
    prompt_text: str,
    *,
    scene_type: str,
    required_roles: list[str],
    optional_roles: list[str] | None = None,
    style_tags: list[str] | None = None,
    color_tags: list[str] | None = None,
    selected_prompt: str | None = None,
    max_props: int = 3,
    stylekit_id: str = "neutral_daylight",
    pack_ids: list[str] | None = None,
    confidence: float = 0.9,
    density_profile: str = "normal",
    layout_mood: str | None = None,
) -> dict:
    asset_ids = shortlist_asset_ids(prompt_text, *required_roles)
    role_asset_map = {
        role: asset_id
        for role, asset_id in zip(required_roles, asset_ids)
    }
    alternatives = {role: [asset_id] for role, asset_id in role_asset_map.items()}

    return {
        "llm_plan": {
            "intent": {
                "scene_type": scene_type,
                "required_roles": required_roles,
                "optional_roles": optional_roles or [],
                "style_tags": style_tags or [],
                "color_tags": color_tags or [],
                "confidence": confidence,
            },
            "placement_intent": {
                "density_profile": density_profile,
                "anchor_preferences": [],
                "adjacency_pairs": [
                    {"source_role": "chair", "target_role": "lamp", "relation": "near"}
                ]
                if "chair" in required_roles and "lamp" in required_roles
                else [],
                "layout_mood": layout_mood or ("crowded" if density_profile == "cluttered" else "cozy"),
            },
            "selection": {
                "selected_prompt": selected_prompt or prompt_text,
                "stylekit_id": stylekit_id,
                "pack_ids": pack_ids or ["core_pack"],
                "role_asset_map": role_asset_map,
                "asset_ids": asset_ids,
                "budgets": {"max_props": max_props},
                "alternatives": alternatives,
                "rationale": ["semantic selection matched the requested roles"],
                "confidence": confidence,
            },
        }
    }
