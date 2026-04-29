from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_POLICY_PATH = Path(__file__).resolve().parent / "taxonomy" / "scene_program_policy_v1.json"


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        token = str(item or "").strip().lower().replace("-", "_").replace(" ", "_")
        if token:
            out.append(token)
    return out


@lru_cache(maxsize=1)
def load_scene_program_policy(path: str | None = None) -> Dict[str, Any]:
    policy_path = Path(path) if path else DEFAULT_POLICY_PATH
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    validate_scene_program_policy(payload)
    return payload


def validate_scene_program_policy(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("scene program policy must be a JSON object")
    required = {
        "version",
        "allowed_relations",
        "allowed_relation_types",
        "relation_type_defaults",
        "allowed_constraint_strengths",
        "allowed_target_surface_types",
        "allowed_density_targets",
        "allowed_symmetry_preferences",
        "allowed_group_types",
        "allowed_group_layouts",
        "allowed_facing_rules",
        "allowed_zone_preferences",
        "allowed_group_importance",
        "allowed_focal_walls",
        "allowed_circulation_preferences",
        "allowed_empty_space_preferences",
        "allowed_slot_necessities",
        "allowed_slot_sources",
        "allowed_optional_placement_hints",
        "required_surface_material_slots",
        "allowed_budget_keys",
        "adjacency_relations",
        "spatial_position_relations",
        "near_constraint_relations",
        "allowed_anchor_preferences",
        "bright_white_wall_context_tokens",
        "seating_roles",
        "edge_biased_roles",
        "group_zone_defaults",
        "group_constraint_defaults",
        "completion_rules",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"scene program policy missing required keys: {', '.join(missing)}")
    allowed_relations = set(_string_list(payload.get("allowed_relations")))
    allowed_relation_types = set(_string_list(payload.get("allowed_relation_types")))
    if not allowed_relations or not allowed_relation_types:
        raise ValueError("scene program policy must define relation vocabularies")
    for relation, relation_type in dict(payload.get("relation_type_defaults") or {}).items():
        if str(relation).strip().lower() not in allowed_relations:
            raise ValueError(f"relation_type_defaults.{relation} is not an allowed relation")
        if str(relation_type).strip().lower() not in allowed_relation_types:
            raise ValueError(f"relation_type_defaults.{relation} maps to unsupported relation type '{relation_type}'")
    for rule in payload.get("completion_rules") or []:
        if not isinstance(rule, dict):
            raise ValueError("completion_rules entries must be objects")
        slot = rule.get("slot")
        if not isinstance(slot, dict) or not str(slot.get("slot_id") or "").strip() or not str(slot.get("concept") or "").strip():
            raise ValueError("completion_rules entries must define slot.slot_id and slot.concept")


def policy_set(key: str) -> set[str]:
    return set(_string_list(load_scene_program_policy().get(key)))


def policy_tuple(key: str) -> tuple[str, ...]:
    return tuple(_string_list(load_scene_program_policy().get(key)))



def policy_dict(key: str) -> Dict[str, str]:
    values = load_scene_program_policy().get(key)
    if not isinstance(values, dict):
        return {}
    return {
        str(item_key or "").strip().lower().replace("-", "_").replace(" ", "_"): str(item_value or "").strip().lower().replace("-", "_").replace(" ", "_")
        for item_key, item_value in values.items()
        if str(item_key or "").strip() and str(item_value or "").strip()
    }

def relation_type_default(relation: str) -> str:
    token = str(relation or "").strip().lower().replace("-", "_").replace(" ", "_")
    defaults = {str(k).strip().lower(): str(v).strip().lower() for k, v in dict(load_scene_program_policy().get("relation_type_defaults") or {}).items()}
    return defaults.get(token, "proximity")


def completion_rules() -> List[Dict[str, Any]]:
    return [dict(rule) for rule in load_scene_program_policy().get("completion_rules") or [] if isinstance(rule, dict)]
