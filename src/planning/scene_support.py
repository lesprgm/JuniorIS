from __future__ import annotations

from typing import Any, Dict, Iterable, List

from src.catalog.style_material_pool import (
    SURFACE_MATERIAL_SLOTS,
    build_surface_material_candidates as build_surface_material_candidates_from_pool,
)
from src.catalog.stylekit_registry import StyleKitRegistry
from src.planning.scene_types import SceneProgram
from src.planning.scene_program import public_intent_spec, public_scene_program


# Keep behavior deterministic so planner/runtime contracts stay stable.
def build_stylekit_candidates(style_registry: StyleKitRegistry) -> List[Dict[str, Any]]:  # produces the compact stylekit list sent in the LLM selection prompt
    candidates: List[Dict[str, Any]] = []
    for stylekit_id, stylekit in sorted(style_registry.stylekits_by_id.items()):
        lighting = stylekit.get("lighting") if isinstance(stylekit.get("lighting"), dict) else {}
        candidates.append(
            {
                "stylekit_id": stylekit_id,
                "tags": [str(tag).strip().lower() for tag in stylekit.get("tags", []) if isinstance(tag, str)],
                "lighting_preset": str(lighting.get("preset") or "").strip().lower(),
                "palette": stylekit.get("palette") if isinstance(stylekit.get("palette"), dict) else {},
            }
        )
    return candidates


def build_pack_candidates(registry: Any) -> List[Dict[str, Any]]:  # produces the compact pack list sent in the LLM selection prompt
    candidates: List[Dict[str, Any]] = []
    for pack_id, pack in sorted(registry.packs_by_id.items()):
        candidates.append(
            {
                "pack_id": pack_id,
                "tags": [str(tag).strip().lower() for tag in pack.get("tags", []) if isinstance(tag, str)],
                "asset_count": len(pack.get("assets", [])) if isinstance(pack.get("assets"), list) else 0,
            }
        )
    return candidates


def apply_stylekit_colors(stylekit_id: str | None, style_registry: StyleKitRegistry) -> Dict[str, str]:  # extracts wall/floor/ceiling/accent hex colors from the chosen stylekit
    if not stylekit_id:
        return {}

    stylekit = style_registry.get_stylekit(stylekit_id) or {}
    palette = stylekit.get("palette", {})
    if not isinstance(palette, dict):
        return {}

    wall = str(palette.get("wall", "#d8d8d8"))
    floor = str(palette.get("floor", "#8b7d6b"))
    accent = str(palette.get("accent", "#4a90e2"))
    ceiling = str(palette.get("ceiling") or wall).strip() or wall

    return {
        "wall": wall,
        "floor": floor,
        "ceiling": ceiling,
        "accent": accent,
    }




def build_surface_material_candidates(scene_program: Dict[str, Any] | None) -> Dict[str, List[Dict[str, Any]]]:  # delegates to the material pool to build per-surface candidate lists
    return build_surface_material_candidates_from_pool(scene_program)


def apply_surface_material_colors(  # overlays material preview colors onto the base stylekit palette
    colors: Dict[str, str],
    surface_material_selection: Dict[str, Any] | None,
    surface_material_candidates: Dict[str, List[Dict[str, Any]]] | None,
) -> Dict[str, str]:
    resolved = dict(colors)
    if not isinstance(surface_material_selection, dict) or not isinstance(surface_material_candidates, dict):
        return resolved

    candidate_by_id: Dict[str, Dict[str, Any]] = {}
    for surface in SURFACE_MATERIAL_SLOTS:
        entries = surface_material_candidates.get(surface)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            material_id = str(entry.get("material_id") or "").strip()
            if material_id and material_id not in candidate_by_id:
                candidate_by_id[material_id] = entry

    for surface in SURFACE_MATERIAL_SLOTS:
        material_id = str(surface_material_selection.get(surface) or "").strip()
        if not material_id:
            continue
        material = candidate_by_id.get(material_id)
        if not isinstance(material, dict):
            continue
        color = str(material.get("preview_color_hex") or "").strip()
        if color.startswith("#") and len(color) == 7:
            resolved[surface] = color
    return resolved

def semantic_receipts(  # builds an audit trail of the semantic planning decisions for debugging
    *,
    selected_assets: Iterable[Dict[str, Any]],
    candidate_assets: List[Dict[str, Any]],
    intent_spec: Dict[str, Any],
    scene_program: SceneProgram,
    placement_intent: Dict[str, Any],
    semantic_selection: Dict[str, Any],
) -> Dict[str, Any]:
    selected_assets = list(selected_assets)
    selected_slots = {
        str(slot_id).strip(): str(asset_id).strip()
        for slot_id, asset_id in dict(semantic_selection.get("slot_asset_map") or {}).items()
        if isinstance(slot_id, str) and str(slot_id).strip() and isinstance(asset_id, str) and str(asset_id).strip()
    }

    alternatives = semantic_selection.get("alternatives")
    if not isinstance(alternatives, dict):
        alternatives = {}

    rationale = semantic_selection.get("rationale")
    if not isinstance(rationale, list) or not rationale:
        rationale = [
            "The planner selected assets from the allowlisted quest-safe shortlist.",
            "Validation kept the runtime payload inside the allowed contract.",
        ]

    confidence = semantic_selection.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = intent_spec.get("confidence", 0.0)
    confidence = max(0.0, min(float(confidence), 1.0))

    return {
        "scene_program": public_scene_program(scene_program),
        "intent_spec": public_intent_spec(intent_spec),
        "placement_intent": placement_intent,
        "selected_slots": selected_slots,
        "slot_diagnostics": list(semantic_selection.get("slot_diagnostics") or []),
        "alternatives": alternatives,
        "rationale": [str(item) for item in rationale if isinstance(item, str)],
        "confidence": round(confidence, 3),
        "selection": {
            "stylekit_id": semantic_selection.get("stylekit_id"),
            "pack_ids": semantic_selection.get("pack_ids") or [],
            "asset_ids": semantic_selection.get("asset_ids") or [],
            "optional_additions": semantic_selection.get("optional_additions") or [],
            "decor_plan": semantic_selection.get("decor_plan") or {},
            "surface_material_selection": semantic_selection.get("surface_material_selection") or {},
        },
    }
