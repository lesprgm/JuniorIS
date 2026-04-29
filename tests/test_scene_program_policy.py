from __future__ import annotations

import pytest

from src.planning.scene_program_policy import (
    completion_rules,
    load_scene_program_policy,
    relation_type_default,
    validate_scene_program_policy,
)
from src.placement.scene_solver_defaults import (
    GROUP_CONSTRAINT_DEFAULTS,
    GROUP_ZONE_DEFAULTS,
    SEATING_ROLES,
)


def test_scene_program_policy_loads_contract_vocabulary():
    policy = load_scene_program_policy()
    assert "near" in policy["allowed_relations"]
    assert relation_type_default("support_on") == "support"
    assert any(rule["rule_id"] == "sleep_bedside_support" for rule in completion_rules())


def test_scene_program_policy_validation_rejects_bad_relation_default():
    payload = dict(load_scene_program_policy())
    payload["relation_type_defaults"] = {**payload["relation_type_defaults"], "not_allowed": "proximity"}
    with pytest.raises(ValueError):
        validate_scene_program_policy(payload)


def test_scene_program_policy_loads_group_defaults():
    assert "chair" in SEATING_ROLES
    assert GROUP_ZONE_DEFAULTS["reading_corner"] == "corner"
    assert GROUP_CONSTRAINT_DEFAULTS["bedside_cluster"] == "against_wall"

from src.llm.planner import _prompt_policy_text
from src.planning.scene_types import SUPPORTED_SEMANTIC_ROLES
from src.placement.semantic_taxonomy import supported_runtime_roles


def test_scene_program_policy_owns_anchor_and_material_context_tokens():
    policy = load_scene_program_policy()
    assert "reading_nook" in policy["allowed_anchor_preferences"]
    assert "gallery" in policy["bright_white_wall_context_tokens"]


def test_prompt_policy_owns_llm_examples():
    assert "bedroom" in _prompt_policy_text("intent_few_shots")
    assert "selection priors" in _prompt_policy_text("selection_few_shots")


def test_scene_types_runtime_roles_are_taxonomy_backed():
    assert SUPPORTED_SEMANTIC_ROLES == supported_runtime_roles()
