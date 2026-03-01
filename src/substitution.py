from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from src.pack_registry import PackRegistry


PLACEHOLDER_ASSET_ID = "placeholder_cube"

_POLY_ORDER = {"low": 0, "mid": 1, "high": 2}
_POLY_HINT_ORDER = {"low": 0, "medium": 1, "mid": 1, "high": 2}
_PERF_HINT_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class ResolvedAsset:
    resolved_asset_id: str
    resolution_type: str  # exact | substitute | placeholder
    reason: str
    coherence_checks: Dict[str, bool]
    rejected_candidate_counts: Dict[str, int]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "resolved_asset_id": self.resolved_asset_id,
            "resolution_type": self.resolution_type,
            "reason": self.reason,
            "coherence_checks": dict(self.coherence_checks),
            "rejected_candidate_counts": dict(self.rejected_candidate_counts),
        }


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _normalize_tags(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    for value in values:
        if isinstance(value, str):
            token = value.strip().lower()
            if token:
                out.append(token)
    return out


def _infer_category(tokens: Iterable[str]) -> str | None:
    token_set = set(tokens)
    if token_set & {"chair", "stool", "bench", "sofa", "seat"}:
        return "seating"
    if token_set & {"table", "desk", "counter"}:
        return "table"
    if token_set & {"lamp", "light", "spot", "chandelier"}:
        return "light"
    if token_set & {"shelf", "bookcase", "cabinet", "dresser", "wardrobe"}:
        return "storage"
    if token_set & {"plant", "decor", "vase", "frame"}:
        return "decor"
    return None


def _adjacent_poly_allowed(requested_poly: str, candidate_poly: str) -> bool:
    req_idx = _POLY_ORDER.get(requested_poly)
    cand_idx = _POLY_ORDER.get(candidate_poly)
    if req_idx is None or cand_idx is None:
        return True
    return abs(req_idx - cand_idx) <= 1


def _safe_tag_overlap(a: Iterable[str], b: Iterable[str]) -> int:
    return len(set(a) & set(b))


def _asset_pack_id_from_hint(asset_id: str, registry: PackRegistry) -> str | None:
    if asset_id in registry.assets_by_id:
        return registry.assets_by_id[asset_id]["pack_id"]

    token = asset_id.split("_", 1)[0].lower()
    if not token:
        return None

    for pack_id in sorted(registry.packs_by_id.keys()):
        if pack_id.lower().startswith(token):
            return pack_id
    return None


def _derive_requested_profile(
    requested_asset_id: str,
    requested_tags: List[str] | None,
    requested_meta: Dict[str, Any] | None,
    registry: PackRegistry,
) -> Dict[str, Any]:
    requested_meta = requested_meta or {}

    tags = _normalize_tags(requested_tags or requested_meta.get("tags") or [])
    existing = registry.assets_by_id.get(requested_asset_id)
    if existing:
        existing_asset = existing.get("asset", {})
        tags = tags or _normalize_tags(existing_asset.get("tags"))

    if not tags:
        tags = sorted(_tokenize(requested_asset_id))

    category = requested_meta.get("category")
    if not isinstance(category, str):
        category = _infer_category(tags)

    style_tags = _normalize_tags(requested_meta.get("style_tags"))
    era_tags = _normalize_tags(requested_meta.get("era_tags"))
    color_tags = _normalize_tags(requested_meta.get("color_tags"))
    visual_style = requested_meta.get("visual_style")
    if not isinstance(visual_style, str):
        visual_style = None
    poly_style = requested_meta.get("poly_style")
    if not isinstance(poly_style, str):
        poly_style = None

    return {
        "tags": tags,
        "category": category,
        "style_tags": style_tags,
        "era_tags": era_tags,
        "color_tags": color_tags,
        "visual_style": visual_style.lower() if isinstance(visual_style, str) else None,
        "poly_style": poly_style.lower() if isinstance(poly_style, str) else None,
    }


def _candidate_record(
    pack_id: str,
    pack: Dict[str, Any],
    asset: Dict[str, Any],
) -> Dict[str, Any]:
    tags = _normalize_tags(asset.get("tags"))
    label = str(asset.get("label", ""))
    tokens = set(tags) | _tokenize(label) | _tokenize(asset.get("asset_id", ""))
    category = asset.get("category")
    if not isinstance(category, str):
        category = _infer_category(tokens)

    style_tags = _normalize_tags(asset.get("style_tags"))
    era_tags = _normalize_tags(asset.get("era_tags"))
    color_tags = _normalize_tags(asset.get("color_tags"))

    visual_style = asset.get("visual_style")
    if isinstance(visual_style, str):
        visual_style = visual_style.lower()
    else:
        visual_style = None

    poly_style = asset.get("poly_style")
    if isinstance(poly_style, str):
        poly_style = poly_style.lower()
    else:
        poly_hint = str(pack.get("perf_meta", {}).get("poly_budget_hint", "")).lower()
        if poly_hint in _POLY_HINT_ORDER:
            poly_style = "mid" if poly_hint == "medium" else poly_hint
        else:
            poly_style = None

    quest_compatible = asset.get("quest_compatible")
    classification = asset.get("classification")
    if not isinstance(classification, str):
        classification = "prop"

    perf_meta = pack.get("perf_meta", {}) if isinstance(pack.get("perf_meta"), dict) else {}
    texture_tier = perf_meta.get("texture_tier", 1)
    if not isinstance(texture_tier, int):
        texture_tier = 1
    perf_hint = str(perf_meta.get("poly_budget_hint", "low")).lower()
    perf_rank = _PERF_HINT_ORDER.get(perf_hint, 1)

    return {
        "asset_id": asset.get("asset_id"),
        "pack_id": pack_id,
        "tags": tags,
        "category": category,
        "style_tags": style_tags,
        "era_tags": era_tags,
        "color_tags": color_tags,
        "visual_style": visual_style,
        "poly_style": poly_style,
        "quest_compatible": quest_compatible,
        "classification": classification,
        "texture_tier": texture_tier,
        "perf_rank": perf_rank,
    }


def _coherence_filter(
    candidates: List[Dict[str, Any]],
    requested: Dict[str, Any],
    room_theme: Dict[str, Any] | None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    room_theme = room_theme or {}
    room_style = set(_normalize_tags(room_theme.get("style_tags")))
    room_era = set(_normalize_tags(room_theme.get("era_tags")))
    room_colors = set(_normalize_tags(room_theme.get("color_tags")))

    out: List[Dict[str, Any]] = []
    rejections: Counter[str] = Counter()

    for candidate in candidates:
        if candidate.get("quest_compatible") is False:
            rejections["rejected_for_quest_compat"] += 1
            continue

        if candidate.get("classification") != "prop":
            rejections["rejected_for_classification"] += 1
            continue

        req_visual = requested.get("visual_style")
        cand_visual = candidate.get("visual_style")
        if req_visual and cand_visual and cand_visual != req_visual:
            rejections["rejected_for_style"] += 1
            continue

        req_poly = requested.get("poly_style")
        cand_poly = candidate.get("poly_style")
        if req_poly and cand_poly and not _adjacent_poly_allowed(req_poly, cand_poly):
            rejections["rejected_for_poly"] += 1
            continue

        theme_must_match = bool(room_style or room_era or room_colors)
        candidate_theme_available = bool(
            candidate.get("style_tags") or candidate.get("era_tags") or candidate.get("color_tags")
        )
        if theme_must_match and candidate_theme_available:
            overlap = (
                _safe_tag_overlap(candidate.get("style_tags", []), room_style)
                + _safe_tag_overlap(candidate.get("era_tags", []), room_era)
                + _safe_tag_overlap(candidate.get("color_tags", []), room_colors)
            )
            if overlap <= 0:
                rejections["rejected_for_theme"] += 1
                continue

        out.append(candidate)

    return out, dict(rejections)


def _score_candidate(candidate: Dict[str, Any], requested: Dict[str, Any], room_theme: Dict[str, Any] | None) -> Tuple:
    room_theme = room_theme or {}
    requested_tags = requested.get("tags", [])
    requested_category = requested.get("category")

    shared_tags = _safe_tag_overlap(candidate.get("tags", []), requested_tags)
    category_match = 1 if requested_category and candidate.get("category") == requested_category else 0

    style_overlap = _safe_tag_overlap(candidate.get("style_tags", []), requested.get("style_tags", []))
    era_overlap = _safe_tag_overlap(candidate.get("era_tags", []), requested.get("era_tags", []))
    room_style_overlap = _safe_tag_overlap(candidate.get("style_tags", []), _normalize_tags(room_theme.get("style_tags")))
    room_era_overlap = _safe_tag_overlap(candidate.get("era_tags", []), _normalize_tags(room_theme.get("era_tags")))
    style_score = style_overlap + era_overlap + room_style_overlap + room_era_overlap

    color_score = _safe_tag_overlap(candidate.get("color_tags", []), requested.get("color_tags", []))
    color_score += _safe_tag_overlap(candidate.get("color_tags", []), _normalize_tags(room_theme.get("color_tags")))

    texture_tier = int(candidate.get("texture_tier", 1))
    perf_rank = int(candidate.get("perf_rank", 1))

    # Maximize semantic coherence first, then minimize cost, then deterministic asset id.
    return (
        -shared_tags,
        -category_match,
        -style_score,
        -color_score,
        texture_tier,
        perf_rank,
        str(candidate.get("asset_id", "")),
    )


def _build_coherence_checks(
    requested: Dict[str, Any],
    selected: Dict[str, Any] | None,
    room_theme: Dict[str, Any] | None,
) -> Dict[str, bool]:
    if selected is None:
        return {
            "visual_style_match": False,
            "poly_style_match": False,
            "theme_overlap_match": False,
        }

    room_theme = room_theme or {}

    req_visual = requested.get("visual_style")
    cand_visual = selected.get("visual_style")
    visual_style_match = not req_visual or not cand_visual or req_visual == cand_visual

    req_poly = requested.get("poly_style")
    cand_poly = selected.get("poly_style")
    poly_style_match = not req_poly or not cand_poly or _adjacent_poly_allowed(req_poly, cand_poly)

    room_theme_active = bool(room_theme and any(_normalize_tags(room_theme.get(k)) for k in ("style_tags", "era_tags", "color_tags")))
    selected_theme_available = bool(
        selected.get("style_tags") or selected.get("era_tags") or selected.get("color_tags")
    )
    theme_overlap = (
        _safe_tag_overlap(selected.get("style_tags", []), _normalize_tags(room_theme.get("style_tags")))
        + _safe_tag_overlap(selected.get("era_tags", []), _normalize_tags(room_theme.get("era_tags")))
        + _safe_tag_overlap(selected.get("color_tags", []), _normalize_tags(room_theme.get("color_tags")))
    )
    theme_overlap_match = True
    if room_theme_active and selected_theme_available:
        theme_overlap_match = theme_overlap > 0

    return {
        "visual_style_match": visual_style_match,
        "poly_style_match": poly_style_match,
        "theme_overlap_match": theme_overlap_match,
    }


def resolve_asset_or_substitute(
    requested_asset_id: str,
    requested_tags: List[str] | None,
    pack_ids: List[str],
    registry: PackRegistry,
    requested_meta: Dict[str, Any] | None = None,
    room_theme: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    selected_pack_ids = [pack_id for pack_id in pack_ids if pack_id in registry.packs_by_id]
    requested_invalid_packs = bool(pack_ids) and not selected_pack_ids
    if not selected_pack_ids and not requested_invalid_packs:
        selected_pack_ids = sorted(registry.packs_by_id.keys())

    requested = _derive_requested_profile(
        requested_asset_id=requested_asset_id,
        requested_tags=requested_tags,
        requested_meta=requested_meta,
        registry=registry,
    )
    rejected_counts: Counter[str] = Counter()

    existing = registry.assets_by_id.get(requested_asset_id)
    if existing:
        existing_pack_id = str(existing.get("pack_id"))
        if requested_invalid_packs:
            rejected_counts["rejected_for_pack_selection"] += 1
        elif selected_pack_ids and existing_pack_id not in selected_pack_ids:
            rejected_counts["rejected_for_pack_selection"] += 1
        else:
            pack = registry.packs_by_id.get(existing_pack_id, {})
            candidate = _candidate_record(existing_pack_id, pack, existing.get("asset", {}))
            filtered_exact, rejected_exact = _coherence_filter([candidate], requested, room_theme)
            rejected_counts.update(rejected_exact)
            if filtered_exact:
                coherence = _build_coherence_checks(requested, candidate, room_theme)
                return ResolvedAsset(
                    resolved_asset_id=requested_asset_id,
                    resolution_type="exact",
                    reason="asset_found",
                    coherence_checks=coherence,
                    rejected_candidate_counts=dict(rejected_counts),
                ).as_dict()

    requested_pack_hint = _asset_pack_id_from_hint(requested_asset_id, registry)
    shared_pack_candidates: List[Dict[str, Any]] = []
    cross_pack_candidates: List[Dict[str, Any]] = []

    for pack_id in selected_pack_ids:
        pack = registry.packs_by_id.get(pack_id)
        if not pack:
            continue
        for asset in pack.get("assets", []):
            candidate = _candidate_record(pack_id, pack, asset)
            if requested_pack_hint and pack_id == requested_pack_hint:
                shared_pack_candidates.append(candidate)
            else:
                cross_pack_candidates.append(candidate)

    requested_tag_set = set(requested.get("tags", []))

    def _tag_gate(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not requested_tag_set:
            return candidates
        with_overlap = [
            candidate
            for candidate in candidates
            if requested_tag_set & set(candidate.get("tags", []))
        ]
        return with_overlap if with_overlap else []

    for candidate_pool in (_tag_gate(shared_pack_candidates), _tag_gate(cross_pack_candidates)):
        if not candidate_pool:
            continue

        filtered, rejected = _coherence_filter(candidate_pool, requested, room_theme)
        rejected_counts.update(rejected)
        if not filtered:
            continue

        selected = sorted(filtered, key=lambda c: _score_candidate(c, requested, room_theme))[0]
        coherence = _build_coherence_checks(requested, selected, room_theme)
        return ResolvedAsset(
            resolved_asset_id=str(selected["asset_id"]),
            resolution_type="substitute",
            reason="tagged_nearest_match",
            coherence_checks=coherence,
            rejected_candidate_counts=dict(rejected_counts),
        ).as_dict()

    return ResolvedAsset(
        resolved_asset_id=PLACEHOLDER_ASSET_ID,
        resolution_type="placeholder",
        reason="no_viable_substitute",
        coherence_checks=_build_coherence_checks(requested, None, room_theme),
        rejected_candidate_counts=dict(rejected_counts),
    ).as_dict()
