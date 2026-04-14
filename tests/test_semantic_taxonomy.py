from __future__ import annotations

import pytest

from src.placement.semantic_taxonomy import (
    expand_semantic_aliases,
    ground_concept,
    load_semantic_taxonomy,
    role_match_tokens,
    substitution_family_for_tokens,
    validate_semantic_taxonomy,
)


@pytest.mark.parametrize(
    ("concept", "expected"),
    [
        ("nightstand", ("table", "nightstand")),
        ("dresser", ("cabinet", "dresser")),
        ("desk", ("table", "desk")),
        ("floor_lamp", ("lamp", "floor_lamp")),
        ("painting", ("decor", "wall_art")),
        ("focal_art", ("decor", "wall_art")),
        ("pillar", ("decor", "architectural_column")),
        ("column", ("decor", "architectural_column")),
    ],
)
def test_taxonomy_grounds_semantic_concepts(concept: str, expected: tuple[str, str]):
    assert ground_concept(concept) == expected


@pytest.mark.parametrize(
    ("tokens", "family"),
    [
        ({"chair"}, "seating"),
        ({"desk"}, "table"),
        ({"bookcase"}, "storage"),
        ({"lamp"}, "light"),
        ({"frame"}, "decor"),
    ],
)
def test_taxonomy_resolves_substitution_families(tokens: set[str], family: str):
    assert substitution_family_for_tokens(tokens) == family


def test_taxonomy_expands_shortlist_aliases():
    assert {"wall_art", "focal_art", "painting"}.issubset(expand_semantic_aliases({"painting"}))
    assert {"warm_lighting", "floor_lamp"}.issubset(expand_semantic_aliases({"lamp"}))


def test_taxonomy_role_match_tokens_are_data_backed():
    assert "dresser" in role_match_tokens("cabinet")
    assert "light" in role_match_tokens("lamp")


def test_taxonomy_scene_policies_are_data_backed():
    policies = load_semantic_taxonomy()["scene_policies"]
    assert "bathroom_assets" in policies
    assert "food_props" not in policies


def test_taxonomy_validation_rejects_bad_role_alias():
    payload = dict(load_semantic_taxonomy())
    payload["role_aliases"] = {**payload["role_aliases"], "bad_alias": "not_a_role"}
    with pytest.raises(ValueError):
        validate_semantic_taxonomy(payload)
