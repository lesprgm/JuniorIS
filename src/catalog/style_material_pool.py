from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.planning.scene_program_policy import policy_set


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # climb from src/catalog/ to project root
DEFAULT_POOL_PATH = PROJECT_ROOT / "data" / "index" / "style_material_pool_v1.json"  # pre-built material pool keyed by surface
SURFACE_MATERIAL_SLOTS = ("wall", "floor", "ceiling", "accent")  # the four surfaces that can receive materials
DEFAULT_SURFACE_MATERIAL_LIMIT = 16  # larger shortlist so the selector sees more than a tiny shell-material slice per surface


# Keep behavior deterministic so planner/runtime contracts stay stable.
def _normalize_tags(values: Any) -> list[str]:  # deduplicate and lowercase tag values for consistent matching
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value or "").strip().lower()
        if not token or token in seen:
            continue
        normalized.append(token)
        seen.add(token)
    return normalized


def _preview_color_hex(record: Dict[str, Any]) -> str | None:
    rgba = record.get("preview_color_rgba")
    if not isinstance(rgba, dict):
        return None
    channels: list[int] = []
    for key in ("r", "g", "b"):
        value = rgba.get(key)
        if not isinstance(value, (int, float)):
            return None
        channels.append(max(0, min(255, int(round(float(value) * 255.0)))))
    return "#{:02x}{:02x}{:02x}".format(*channels)


def load_style_material_pool(  # loads the full material pool from disk with cache
    pool_path: str | Path = DEFAULT_POOL_PATH,
    *,
    approved_only: bool = True,
) -> list[dict[str, Any]]:
    path = Path(pool_path)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        return []
    out: list[dict[str, Any]] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        if approved_only and str(item.get("review_status") or "").strip().lower() != "approved":
            continue
        if not str(item.get("material_id") or "").strip():
            continue
        out.append(dict(item))
    return out


def load_style_material_pool_by_id(  # returns a specific material entry by ID from the pool
    pool_path: str | Path = DEFAULT_POOL_PATH,
    *,
    approved_only: bool = True,
) -> dict[str, dict[str, Any]]:
    return {
        str(record.get("material_id") or "").strip(): record
        for record in load_style_material_pool(pool_path, approved_only=approved_only)
        if str(record.get("material_id") or "").strip()
    }


def _scene_style_tags(scene_program: Dict[str, Any] | None) -> set[str]:
    scene_program = scene_program if isinstance(scene_program, dict) else {}
    tags = _normalize_tags(scene_program.get("style_tags"))
    cues = scene_program.get("style_cues") if isinstance(scene_program.get("style_cues"), dict) else {}
    tags.extend(_normalize_tags(cues.get("style_tags")))
    return set(tags)


def _scene_color_tags(scene_program: Dict[str, Any] | None) -> set[str]:
    scene_program = scene_program if isinstance(scene_program, dict) else {}
    tags = _normalize_tags(scene_program.get("color_tags"))
    cues = scene_program.get("style_cues") if isinstance(scene_program.get("style_cues"), dict) else {}
    tags.extend(_normalize_tags(cues.get("color_tags")))
    return set(tags)


def _scene_creative_tokens(scene_program: Dict[str, Any] | None) -> set[str]:
    scene_program = scene_program if isinstance(scene_program, dict) else {}
    tokens: list[str] = []
    tokens.extend(_normalize_tags(scene_program.get("creative_tags")))
    tokens.extend(_normalize_tags(scene_program.get("mood_tags")))
    tokens.extend(_normalize_tags(scene_program.get("style_descriptors")))
    for key in ("scene_type", "concept_label"):
        value = str(scene_program.get(key) or "").strip().lower()
        if value:
            tokens.extend(part for part in value.replace("-", "_").split("_") if part)
    intended_use = str(scene_program.get("intended_use") or "").strip().lower()
    if intended_use:
        tokens.extend(part for part in intended_use.replace("-", " ").replace("_", " ").split() if part)
    return set(token for token in tokens if token)


def _record_creative_tokens(record: Dict[str, Any]) -> set[str]:
    tokens = set(_normalize_tags(record.get("style_tags")))
    tokens.update(_normalize_tags(record.get("tone_tags")))
    tokens.update(_normalize_tags(record.get("material_family_tags")))
    tokens.update(_normalize_tags(record.get("texture_tags")))
    tokens.update(_normalize_tags(record.get("finish_tags")))
    for key in ("visual_description", "preview_texture_description", "display_name", "material_name"):
        value = str(record.get(key) or "").strip().lower()
        if not value:
            continue
        tokens.update(part for part in value.replace("-", " ").replace("_", " ").split() if part)
    return tokens


def _allow_bright_white_walls(scene_program: Dict[str, Any] | None) -> bool:
    scene_program = scene_program if isinstance(scene_program, dict) else {}
    tokens = _scene_creative_tokens(scene_program)
    tokens.update(_scene_style_tags(scene_program))
    tokens.update(_scene_color_tags(scene_program))
    tokens.update(_normalize_tags(scene_program.get("mood_tags")))
    return bool(tokens & policy_set("bright_white_wall_context_tokens"))


def _wall_washout_penalty(record: Dict[str, Any], scene_program: Dict[str, Any] | None) -> int:
    if _allow_bright_white_walls(scene_program):
        return 0

    color_tags = set(_normalize_tags(record.get("color_tags")))
    texture_tags = set(_normalize_tags(record.get("texture_tags")))
    tone_tags = set(_normalize_tags(record.get("tone_tags")))
    preview_hex = _preview_color_hex(record) or ""

    is_bright_white = "white" in color_tags or "mostly_white" in texture_tags or preview_hex.lower() in {
        "#ffffff",
        "#fefefe",
        "#fdfdfd",
        "#fcfcfc",
        "#fbfbfb",
        "#fafafa",
        "#f9f9f9",
        "#f8f8f8",
    }
    if not is_bright_white:
        return 0

    penalty = 10
    if "neutral" in tone_tags:
        penalty += 2
    if "low_variation" in texture_tags:
        penalty += 2
    return penalty


def _surface_bias(surface: str, record: Dict[str, Any]) -> int:
    families = set(_normalize_tags(record.get("material_family_tags")))
    texture_tags = set(_normalize_tags(record.get("texture_tags")))
    score = 0
    if surface == "floor" and families.intersection({"wood", "stone", "tile", "concrete"}):
        score += 5
    if surface == "wall" and families.intersection({"painted_surface", "plaster", "wallpaper", "gradient"}):
        score += 5
    if surface == "ceiling" and families.intersection({"painted_surface", "plaster"}):
        score += 4
    if surface == "accent" and families.intersection({"accent_material", "wood", "metal", "glass", "fabric", "upholstery"}):
        score += 4
    if surface in {"wall", "ceiling"} and "low_variation" in texture_tags:
        score += 2
    return score


def _surface_specificity_score(surface: str, record: Dict[str, Any]) -> int:
    inferred_label = str(record.get("inferred_label") or "").strip().lower()
    surface_roles = _normalize_tags(record.get("surface_roles"))
    score = 0
    if inferred_label == surface:
        score += 8
    if surface_roles == [surface]:
        score += 3
    if surface != "accent" and inferred_label == "accent":
        score -= 5
    if surface == "accent" and inferred_label in {"wall", "floor", "ceiling"}:
        score -= 2
    if len(surface_roles) > 3 and inferred_label != surface:
        score -= 1
    return score


def _score_material(surface: str, record: Dict[str, Any], scene_program: Dict[str, Any] | None) -> int:  # scores a material candidate based on tag overlap with the scene program
    surface_roles = set(_normalize_tags(record.get("surface_roles")))
    if surface not in surface_roles:
        return -10_000

    style_tags = set(_normalize_tags(record.get("style_tags")))
    color_tags = set(_normalize_tags(record.get("color_tags")))
    tone_tags = set(_normalize_tags(record.get("tone_tags")))
    scene_style_tags = _scene_style_tags(scene_program)
    scene_color_tags = _scene_color_tags(scene_program)
    scene_creative_tokens = _scene_creative_tokens(scene_program)
    record_creative_tokens = _record_creative_tokens(record)

    score = 100
    score += 9 * len(style_tags & scene_style_tags)
    score += 8 * len(color_tags & scene_color_tags)
    score += 4 * len(record_creative_tokens & scene_creative_tokens)
    score += _surface_specificity_score(surface, record)
    score += _surface_bias(surface, record)
    if surface == "wall":
        score -= _wall_washout_penalty(record, scene_program)
    if "neutral" in tone_tags:
        score += 1
    if "muted" in tone_tags:
        score += 1
    if str(record.get("source_pack") or "").strip().lower() == "materials":
        score += 1
    return score


def _compact_material_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "material_id": str(record.get("material_id") or "").strip(),
        "display_name": str(record.get("display_name") or record.get("material_name") or "").strip(),
        "surface_roles": _normalize_tags(record.get("surface_roles")),
        "color_tags": _normalize_tags(record.get("color_tags")),
        "tone_tags": _normalize_tags(record.get("tone_tags")),
        "style_tags": _normalize_tags(record.get("style_tags")),
        "material_family_tags": _normalize_tags(record.get("material_family_tags")),
        "texture_tags": _normalize_tags(record.get("texture_tags")),
        "finish_tags": _normalize_tags(record.get("finish_tags")),
        "visual_description": str(record.get("visual_description") or "").strip(),
        "preview_texture_description": str(record.get("preview_texture_description") or "").strip(),
        "preview_color_hex": _preview_color_hex(record),
        "source_pack": str(record.get("source_pack") or "").strip(),
    }


def build_surface_material_candidates(  # builds per-surface candidate lists filtered and ranked by scene affinity
    scene_program: Dict[str, Any] | None,
    *,
    limit_per_surface: int = DEFAULT_SURFACE_MATERIAL_LIMIT,
    pool_path: str | Path = DEFAULT_POOL_PATH,
) -> Dict[str, List[Dict[str, Any]]]:
    records = load_style_material_pool(pool_path, approved_only=True)
    candidates: Dict[str, List[Dict[str, Any]]] = {surface: [] for surface in SURFACE_MATERIAL_SLOTS}
    for surface in SURFACE_MATERIAL_SLOTS:
        ranked = sorted(
            records,
            key=lambda record: (
                _score_material(surface, record, scene_program),
                str(record.get("display_name") or record.get("material_name") or "").lower(),
            ),
            reverse=True,
        )
        chosen: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for record in ranked:
            if len(chosen) >= limit_per_surface:
                break
            material_id = str(record.get("material_id") or "").strip()
            if not material_id or material_id in seen_ids:
                continue
            if _score_material(surface, record, scene_program) < 0:
                continue
            chosen.append(_compact_material_record(record))
            seen_ids.add(material_id)
        candidates[surface] = chosen
    return candidates


__all__ = [
    "DEFAULT_POOL_PATH",
    "DEFAULT_SURFACE_MATERIAL_LIMIT",
    "SURFACE_MATERIAL_SLOTS",
    "build_surface_material_candidates",
    "load_style_material_pool",
    "load_style_material_pool_by_id",
]
