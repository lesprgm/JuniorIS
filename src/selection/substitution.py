from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple

from src.catalog.pack_registry import PackRegistry
from src.placement.geometry import canonicalize_semantic_role, semantic_role_key
from src.placement.semantic_taxonomy import substitution_family_for_tokens


PLACEHOLDER_ASSET_ID = "placeholder_cube"  # fallback asset when no viable substitute is found

_POLY_ORDER = {"low": 0, "mid": 1, "high": 2}  # hierarchy of polygon counts for downgrade comparisons
_POLY_HINT_ORDER = {"low": 0, "medium": 1, "mid": 1, "high": 2}  # alternative namings normalized for polygon hints
_PERF_HINT_ORDER = {"low": 0, "medium": 1, "high": 2}  # rank order classification for overall performance cost


@dataclass(frozen=True)
# Keep behavior deterministic so planner/runtime contracts stay stable.
class ResolvedAsset:  # immutable result of asset resolution with full audit trail
    resolved_asset_id: str
    resolution_type: str  # exact | substitute | placeholder
    reason: str
    coherence_checks: Dict[str, bool]
    rejected_candidate_counts: Dict[str, int]
    alternatives: List[str] = field(default_factory=list)
    rationale: List[str] = field(default_factory=list)
    selection_backend: str = "deterministic"
    semantic_failure_reason: str | None = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "resolved_asset_id": self.resolved_asset_id,
            "resolution_type": self.resolution_type,
            "reason": self.reason,
            "coherence_checks": dict(self.coherence_checks),
            "rejected_candidate_counts": dict(self.rejected_candidate_counts),
            "alternatives": list(self.alternatives),
            "rationale": list(self.rationale),
            "selection_backend": self.selection_backend,
            "semantic_failure_reason": self.semantic_failure_reason,
        }


def _tokenize(text: str) -> set[str]:  # extracts alphanumeric tokens for fuzzy tag matching
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


def _optional_lower_string(value: Any) -> str | None:  # safely lowercases string or drops None/non-strings
    if not isinstance(value, str):
        return None
    token = value.strip().lower()
    return token or None


def _substitution_family(tokens: Iterable[str]) -> str | None:  # maps tokens to taxonomy-backed substitute families
    return substitution_family_for_tokens(tokens) or None

def _adjacent_poly_allowed(requested_poly: str, candidate_poly: str) -> bool:  # allows +-1 poly tier mismatch (e.g., low can use mid but not high)
    req_idx = _POLY_ORDER.get(requested_poly)
    cand_idx = _POLY_ORDER.get(candidate_poly)
    if req_idx is None or cand_idx is None:
        return True
    return abs(req_idx - cand_idx) <= 1


def _safe_tag_overlap(a: Iterable[str], b: Iterable[str]) -> int:  # utility to count intersection matches between two tag arrays
    return len(set(a) & set(b))


def _asset_pack_id_from_hint(asset_id: str, registry: PackRegistry) -> str | None:  # extracts assumed pack namespace from the prefix of the asset ID if registry is missing it
    if asset_id in registry.assets_by_id:
        return registry.assets_by_id[asset_id]["pack_id"]

    token = asset_id.split("_", 1)[0].lower()
    if not token:
        return None

    for pack_id in sorted(registry.packs_by_id.keys()):
        if pack_id.lower().startswith(token):
            return pack_id
    return None


def _derive_requested_profile(  # synthesizes the expected category/tags/role for an asset when searching for alternates
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
        category = _substitution_family(tags)
    role = semantic_role_key(
        {
            "category": category,
            "role": requested_meta.get("role"),
            "label": requested_meta.get("label"),
            "asset_id": requested_asset_id,
            "requested_asset_id": requested_asset_id,
            "tags": tags,
        }
    )

    return {
        "tags": tags,
        "category": canonicalize_semantic_role(category),
        "role": role,
        "style_tags": _normalize_tags(requested_meta.get("style_tags")),
        "era_tags": _normalize_tags(requested_meta.get("era_tags")),
        "color_tags": _normalize_tags(requested_meta.get("color_tags")),
        "visual_style": _optional_lower_string(requested_meta.get("visual_style")),
        "poly_style": _optional_lower_string(requested_meta.get("poly_style")),
    }


def _candidate_record(pack_id: str, pack: Dict[str, Any], asset: Dict[str, Any]) -> Dict[str, Any]:  # flattens an asset and its pack metadata into a single substitution candidate profile
    tags = _normalize_tags(asset.get("tags"))
    label = str(asset.get("label", ""))
    tokens = set(tags) | _tokenize(label) | _tokenize(asset.get("asset_id", ""))
    category = asset.get("category")
    if not isinstance(category, str):
        category = _substitution_family(tokens)
    role = semantic_role_key(
        {
            "category": category,
            "label": label,
            "asset_id": asset.get("asset_id"),
            "tags": tags,
        }
    )

    style_tags = _normalize_tags(asset.get("style_tags"))
    era_tags = _normalize_tags(asset.get("era_tags"))
    color_tags = _normalize_tags(asset.get("color_tags"))

    visual_style = _optional_lower_string(asset.get("visual_style"))
    poly_style = _optional_lower_string(asset.get("poly_style"))
    if poly_style is None:
        poly_hint = _optional_lower_string(pack.get("perf_meta", {}).get("poly_budget_hint", ""))
        if poly_hint in _POLY_HINT_ORDER:
            poly_style = "mid" if poly_hint == "medium" else poly_hint

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
        "category": canonicalize_semantic_role(category),
        "role": role,
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


def _room_theme_sets(room_theme: Dict[str, Any] | None) -> Tuple[set[str], set[str], set[str]]:  # separates a theme definition into style, era, and color tag sets
    room_theme = room_theme or {}
    return (
        set(_normalize_tags(room_theme.get("style_tags"))),
        set(_normalize_tags(room_theme.get("era_tags"))),
        set(_normalize_tags(room_theme.get("color_tags"))),
    )


def _theme_overlap(candidate: Dict[str, Any], style_tags: Iterable[str], era_tags: Iterable[str], color_tags: Iterable[str]) -> int:  # counts total overlapping markers between an asset candidate and the room's desired theme
    return (
        _safe_tag_overlap(candidate.get("style_tags", []), style_tags)
        + _safe_tag_overlap(candidate.get("era_tags", []), era_tags)
        + _safe_tag_overlap(candidate.get("color_tags", []), color_tags)
    )


def _coherence_filter(  # gates candidates on quest compatibility, role, visual style, poly tier, and theme
    candidates: List[Dict[str, Any]],
    requested: Dict[str, Any],
    room_theme: Dict[str, Any] | None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    room_style, room_era, room_colors = _room_theme_sets(room_theme)

    out: List[Dict[str, Any]] = []
    rejections: Counter[str] = Counter()

    for candidate in candidates:
        if candidate.get("quest_compatible") is False:
            rejections["rejected_for_quest_compat"] += 1
            continue
        if candidate.get("classification") != "prop":
            rejections["rejected_for_classification"] += 1
            continue

        requested_role = requested.get("role")
        candidate_role = candidate.get("role")
        if requested_role and candidate_role and candidate_role != requested_role:
            rejections["rejected_for_role"] += 1
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
        candidate_theme_available = bool(candidate.get("style_tags") or candidate.get("era_tags") or candidate.get("color_tags"))
        if theme_must_match and candidate_theme_available:
            overlap = _theme_overlap(candidate, room_style, room_era, room_colors)
            if overlap <= 0:
                rejections["rejected_for_theme"] += 1
                continue

        out.append(candidate)

    return out, dict(rejections)


def _score_candidate(candidate: Dict[str, Any], requested: Dict[str, Any], room_theme: Dict[str, Any] | None) -> Tuple:  # multi-factor sort key: tag overlap, category, style, color, perf
    requested_tags = requested.get("tags", [])
    requested_category = requested.get("category")
    room_style, room_era, room_colors = _room_theme_sets(room_theme)

    shared_tags = _safe_tag_overlap(candidate.get("tags", []), requested_tags)
    category_match = 1 if requested_category and candidate.get("category") == requested_category else 0

    style_overlap = _safe_tag_overlap(candidate.get("style_tags", []), requested.get("style_tags", []))
    era_overlap = _safe_tag_overlap(candidate.get("era_tags", []), requested.get("era_tags", []))
    room_style_overlap = _safe_tag_overlap(candidate.get("style_tags", []), room_style)
    room_era_overlap = _safe_tag_overlap(candidate.get("era_tags", []), room_era)
    style_score = style_overlap + era_overlap + room_style_overlap + room_era_overlap

    color_score = _safe_tag_overlap(candidate.get("color_tags", []), requested.get("color_tags", []))
    color_score += _safe_tag_overlap(candidate.get("color_tags", []), room_colors)

    texture_tier = int(candidate.get("texture_tier", 1))
    perf_rank = int(candidate.get("perf_rank", 1))

    return (
        -shared_tags,
        -category_match,
        -style_score,
        -color_score,
        texture_tier,
        perf_rank,
        str(candidate.get("asset_id", "")),
    )


def _build_coherence_checks(  # verifies if the final substituted asset adhered to style/poly constraints
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

    req_visual = requested.get("visual_style")
    cand_visual = selected.get("visual_style")
    visual_style_match = not req_visual or not cand_visual or req_visual == cand_visual

    req_poly = requested.get("poly_style")
    cand_poly = selected.get("poly_style")
    poly_style_match = not req_poly or not cand_poly or _adjacent_poly_allowed(req_poly, cand_poly)

    room_style, room_era, room_colors = _room_theme_sets(room_theme)
    room_theme_active = bool(room_style or room_era or room_colors)
    selected_theme_available = bool(selected.get("style_tags") or selected.get("era_tags") or selected.get("color_tags"))
    theme_overlap = _theme_overlap(selected, room_style, room_era, room_colors)
    theme_overlap_match = True
    if room_theme_active and selected_theme_available:
        theme_overlap_match = theme_overlap > 0

    return {
        "visual_style_match": visual_style_match,
        "poly_style_match": poly_style_match,
        "theme_overlap_match": theme_overlap_match,
    }


def _resolve_pack_scope(pack_ids: List[str], registry: PackRegistry) -> Tuple[List[str], bool]:  # filters desired packs by installed manifest availability, falling back to all if invalid
    selected_pack_ids = [pack_id for pack_id in pack_ids if pack_id in registry.packs_by_id]
    requested_invalid_packs = bool(pack_ids) and not selected_pack_ids
    if not selected_pack_ids and not requested_invalid_packs:
        selected_pack_ids = sorted(registry.packs_by_id.keys())
    return selected_pack_ids, requested_invalid_packs


def _try_exact_resolution(  # attempts fast-path where requested asset is present in the specified pack scope
    requested_asset_id: str,
    selected_pack_ids: List[str],
    requested_invalid_packs: bool,
    requested: Dict[str, Any],
    room_theme: Dict[str, Any] | None,
    registry: PackRegistry,
    rejected_counts: Counter[str],
) -> Dict[str, Any] | None:
    existing = registry.assets_by_id.get(requested_asset_id)
    if not existing:
        return None

    existing_pack_id = str(existing.get("pack_id"))
    if requested_invalid_packs or (selected_pack_ids and existing_pack_id not in selected_pack_ids):
        rejected_counts["rejected_for_pack_selection"] += 1
        return None

    pack = registry.packs_by_id.get(existing_pack_id, {})
    candidate = _candidate_record(existing_pack_id, pack, existing.get("asset", {}))
    filtered_exact, rejected_exact = _coherence_filter([candidate], requested, room_theme)
    rejected_counts.update(rejected_exact)
    if not filtered_exact:
        return None

    return ResolvedAsset(
        resolved_asset_id=requested_asset_id,
        resolution_type="exact",
        reason="asset_found",
        coherence_checks=_build_coherence_checks(requested, candidate, room_theme),
        rejected_candidate_counts=dict(rejected_counts),
        rationale=["Requested asset passed deterministic pack and coherence checks."],
        selection_backend="exact",
    ).as_dict()


def _collect_candidate_pools(  # splits all known candidates into shared-pack (preferred) and cross-pack (fallback) buckets
    requested_asset_id: str,
    selected_pack_ids: List[str],
    registry: PackRegistry,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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

    return shared_pack_candidates, cross_pack_candidates


def _deterministic_substitution_selection(  # final decider which chooses highest-ranked candidate ensuring consistent output
    filtered_candidates: List[Dict[str, Any]],
    requested: Dict[str, Any],
    room_theme: Dict[str, Any] | None,
) -> Dict[str, Any]:
    if not filtered_candidates:
        return {"ok": False}

    ranked = sorted(filtered_candidates, key=lambda candidate: _score_candidate(candidate, requested, room_theme))
    selected = ranked[0]
    alternatives = [str(candidate.get("asset_id", "")) for candidate in ranked[:3] if candidate.get("asset_id")]
    return {
        "ok": True,
        "selected": selected,
        "alternatives": alternatives,
        "rationale": ["Selected the highest-ranked compile-compatible substitute after pack, tag, and coherence filtering."],
    }


def resolve_asset_or_substitute(  # main entry: tries exact match, then shared-pack substitute, then cross-pack, then placeholder
    requested_asset_id: str,
    requested_tags: List[str] | None,
    pack_ids: List[str],
    registry: PackRegistry,
    requested_meta: Dict[str, Any] | None = None,
    room_theme: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    selected_pack_ids, requested_invalid_packs = _resolve_pack_scope(pack_ids, registry)
    requested = _derive_requested_profile(
        requested_asset_id=requested_asset_id,
        requested_tags=requested_tags,
        requested_meta=requested_meta,
        registry=registry,
    )
    rejected_counts: Counter[str] = Counter()
    allow_passthrough_exact = bool((requested_meta or {}).get("allow_passthrough_exact"))

    exact_result = _try_exact_resolution(
        requested_asset_id=requested_asset_id,
        selected_pack_ids=selected_pack_ids,
        requested_invalid_packs=requested_invalid_packs,
        requested=requested,
        room_theme=room_theme,
        registry=registry,
        rejected_counts=rejected_counts,
    )
    if exact_result is not None:
        return exact_result

    if allow_passthrough_exact:
        return ResolvedAsset(
            resolved_asset_id=requested_asset_id,
            resolution_type="exact",
            reason="approved_planner_passthrough",
            coherence_checks=_build_coherence_checks(requested, None, room_theme),
            rejected_candidate_counts=dict(rejected_counts),
            rationale=["Requested asset is preserved from the approved planner pool even though it is not in the compile registry."],
            selection_backend="passthrough",
        ).as_dict()

    shared_pack_candidates, cross_pack_candidates = _collect_candidate_pools(
        requested_asset_id=requested_asset_id,
        selected_pack_ids=selected_pack_ids,
        registry=registry,
    )

    requested_tag_set = set(requested.get("tags", []))

    def _tag_gate(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not requested_tag_set:
            return candidates
        with_overlap = [candidate for candidate in candidates if requested_tag_set & set(candidate.get("tags", []))]
        return with_overlap if with_overlap else []

    for candidate_pool in (_tag_gate(shared_pack_candidates), _tag_gate(cross_pack_candidates)):
        if not candidate_pool:
            continue

        filtered, rejected = _coherence_filter(candidate_pool, requested, room_theme)
        rejected_counts.update(rejected)
        if not filtered:
            continue

        deterministic_result = _deterministic_substitution_selection(filtered, requested, room_theme)
        if deterministic_result.get("ok"):
            selected = deterministic_result["selected"]
            return ResolvedAsset(
                resolved_asset_id=str(selected["asset_id"]),
                resolution_type="substitute",
                reason="deterministic_substitute_match",
                coherence_checks=_build_coherence_checks(requested, selected, room_theme),
                rejected_candidate_counts=dict(rejected_counts),
                alternatives=deterministic_result["alternatives"],
                rationale=deterministic_result["rationale"],
                selection_backend="deterministic",
            ).as_dict()

    return ResolvedAsset(
        resolved_asset_id=PLACEHOLDER_ASSET_ID,
        resolution_type="placeholder",
        reason="no_viable_substitute",
        coherence_checks=_build_coherence_checks(requested, None, room_theme),
        rejected_candidate_counts=dict(rejected_counts),
        alternatives=[],
        rationale=["No deterministic candidate survived pack, tag, and coherence filtering."],
        selection_backend="deterministic",
    ).as_dict()
