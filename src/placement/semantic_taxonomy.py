from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

DEFAULT_TAXONOMY_PATH = Path(__file__).resolve().parent / "taxonomy" / "semantic_taxonomy_v1.json"


def _normalize_token(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")
    return "_".join(part for part in token.split("_") if part)


def _token_parts(value: Any) -> List[str]:
    return [part for part in _normalize_token(value).split("_") if part]


@lru_cache(maxsize=1)
def load_semantic_taxonomy(path: str | None = None) -> Dict[str, Any]:
    taxonomy_path = Path(path) if path else DEFAULT_TAXONOMY_PATH
    payload = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    validate_semantic_taxonomy(payload)
    return payload


def validate_semantic_taxonomy(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("semantic taxonomy must be a JSON object")
    required = {
        "version",
        "supported_runtime_roles",
        "role_aliases",
        "concept_aliases",
        "concept_grounding",
        "token_grounding_rules",
        "semantic_alias_groups",
        "substitution_families",
        "role_match_tokens",
        "decor",
        "rescue_families",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"semantic taxonomy missing required keys: {', '.join(missing)}")
    supported_roles = set(string_list(payload.get("supported_runtime_roles")))
    if not supported_roles:
        raise ValueError("semantic taxonomy must define supported_runtime_roles")
    for alias, role in dict(payload.get("role_aliases") or {}).items():
        if _normalize_token(role) not in supported_roles:
            raise ValueError(f"role_aliases.{alias} maps to unsupported role '{role}'")
    for concept, grounding in dict(payload.get("concept_grounding") or {}).items():
        if not isinstance(grounding, dict) or _normalize_token(grounding.get("runtime_role")) not in supported_roles:
            raise ValueError(f"concept_grounding.{concept} must map to a supported runtime role")
    for rule in payload.get("token_grounding_rules") or []:
        if not isinstance(rule, dict) or _normalize_token(rule.get("runtime_role")) not in supported_roles:
            raise ValueError("token_grounding_rules entries must map to supported runtime roles")
    for family, rule in dict(payload.get("rescue_families") or {}).items():
        if not isinstance(rule, dict) or not string_list(rule.get("slot_tokens")) or not string_list(rule.get("asset_tokens")):
            raise ValueError(f"rescue_families.{family} must define slot_tokens and asset_tokens")


def string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        token = _normalize_token(item)
        if token:
            out.append(token)
    return out


def supported_runtime_roles() -> set[str]:
    return set(string_list(load_semantic_taxonomy().get("supported_runtime_roles")))


def role_aliases() -> Dict[str, str]:
    return {_normalize_token(k): _normalize_token(v) for k, v in dict(load_semantic_taxonomy().get("role_aliases") or {}).items()}


def canonicalize_role_token(value: Any) -> str:
    token = _normalize_token(value)
    if not token:
        return ""
    aliases = role_aliases()
    supported = supported_runtime_roles()
    for candidate in (token, token[:-1] if token.endswith("s") else token):
        if candidate in aliases:
            return aliases[candidate]
        if candidate in supported:
            return candidate
    return ""


def _strip_context_suffix(parts: List[str]) -> List[str]:
    if parts and parts[-1] in {"area", "room", "space", "corner", "zone"}:
        return parts[:-1]
    return parts


def canonicalize_concept(value: Any) -> str:
    parts = _strip_context_suffix(_token_parts(value))
    if not parts:
        return ""
    token = "_".join(parts)
    aliases = {_normalize_token(k): _normalize_token(v) for k, v in dict(load_semantic_taxonomy().get("concept_aliases") or {}).items()}
    if token in aliases:
        return aliases[token]
    for index in range(len(parts)):
        suffix = "_".join(parts[index:])
        if suffix in aliases:
            return aliases[suffix]
    return token


def _rule_subtype(rule: Dict[str, Any], token: str, token_set: set[str], role: str) -> str:
    subtype_by_token = dict(rule.get("subtype_by_token") or {})
    for key, value in subtype_by_token.items():
        if _normalize_token(key) in token_set:
            return _normalize_token(value)
    default = _normalize_token(rule.get("default_subtype"))
    if default == "$token_except_role":
        return "" if token in {role, "storage", "cabinet"} else token
    return default


def ground_concept(concept: Any) -> Tuple[str, str]:
    token = canonicalize_concept(concept)
    if not token:
        return "", ""
    taxonomy = load_semantic_taxonomy()
    grounding = dict(taxonomy.get("concept_grounding") or {}).get(token)
    if isinstance(grounding, dict):
        return _normalize_token(grounding.get("runtime_role")), _normalize_token(grounding.get("subtype"))

    direct = canonicalize_role_token(token)
    if direct:
        return direct, ""

    token_set = set(_token_parts(token))
    for rule in taxonomy.get("token_grounding_rules") or []:
        if not isinstance(rule, dict):
            continue
        match_all = set(string_list(rule.get("match_all")))
        match_any = set(string_list(rule.get("match_any")))
        if match_all and not match_all.issubset(token_set):
            continue
        if match_any and not (match_any & token_set):
            continue
        role = _normalize_token(rule.get("runtime_role"))
        return role, _rule_subtype(rule, token, token_set, role)
    return "", ""


def expand_semantic_aliases(tokens: Iterable[str]) -> set[str]:
    normalized = {_normalize_token(token) for token in tokens if _normalize_token(token)}
    aliases: set[str] = set()
    for group in load_semantic_taxonomy().get("semantic_alias_groups") or []:
        if not isinstance(group, dict):
            continue
        triggers = set(string_list(group.get("triggers")))
        if normalized & triggers:
            aliases.update(string_list(group.get("aliases")))
    return aliases


def _tokens_from_values(values: Iterable[Any], *, expand_aliases: bool = False) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        token = _normalize_token(value)
        if not token:
            continue
        tokens.add(token)
        tokens.update(_token_parts(token))
        concept = canonicalize_concept(token)
        if concept:
            tokens.add(concept)
    return tokens | (expand_semantic_aliases(tokens) if expand_aliases else set())


def _record_tokens(record: Dict[str, Any], scalar_keys: Iterable[str], list_keys: Iterable[str]) -> set[str]:
    values: List[Any] = [record.get(key) for key in scalar_keys]
    for key in list_keys:
        raw = record.get(key)
        if isinstance(raw, list):
            values.extend(raw)
    return _tokens_from_values(values)


def rescue_family_for_slot(slot: Dict[str, Any]) -> str:
    tokens = _record_tokens(slot, ("slot_id", "concept", "subtype", "runtime_role", "runtime_role_hint"), ())
    tokens |= expand_semantic_aliases(tokens)
    for family, rule in dict(load_semantic_taxonomy().get("rescue_families") or {}).items():
        if tokens & set(string_list(dict(rule).get("slot_tokens"))):
            return _normalize_token(family)
    return ""


def rescue_family_allows_duplicate_soften(family: str) -> bool:
    rule = dict(dict(load_semantic_taxonomy().get("rescue_families") or {}).get(_normalize_token(family)) or {})
    return rule.get("duplicate_soften") is True


def asset_matches_rescue_family(asset: Dict[str, Any], family: str) -> bool:
    rule = dict(dict(load_semantic_taxonomy().get("rescue_families") or {}).get(_normalize_token(family)) or {})
    if not rule:
        return False
    tokens = _record_tokens(
        asset,
        ("asset_id", "label", "display_name", "room_role_subtype", "semantic_concept", "category"),
        ("tags", "support_surface_types"),
    )
    return bool(tokens & set(string_list(rule.get("asset_tokens"))))


def substitution_family_for_tokens(tokens: Iterable[str]) -> str:
    token_set = {_normalize_token(token) for token in tokens if _normalize_token(token)}
    for family, values in dict(load_semantic_taxonomy().get("substitution_families") or {}).items():
        if token_set & set(string_list(values)):
            return _normalize_token(family)
    return ""


def role_match_tokens(role: str) -> tuple[str, ...]:
    token = _normalize_token(role)
    values = dict(load_semantic_taxonomy().get("role_match_tokens") or {}).get(token)
    return tuple(string_list(values)) or (token,)


def decor_kinds() -> tuple[str, ...]:
    decor = load_semantic_taxonomy().get("decor") if isinstance(load_semantic_taxonomy().get("decor"), dict) else {}
    return tuple(string_list(decor.get("kinds")))


def decor_allowed_anchors() -> set[str]:
    decor = load_semantic_taxonomy().get("decor") if isinstance(load_semantic_taxonomy().get("decor"), dict) else {}
    return set(string_list(decor.get("allowed_anchors")))


def decor_anchor_aliases() -> Dict[str, str]:
    decor = load_semantic_taxonomy().get("decor") if isinstance(load_semantic_taxonomy().get("decor"), dict) else {}
    return {_normalize_token(k): _normalize_token(v) for k, v in dict(decor.get("anchor_aliases") or {}).items()}


def decor_kind_rules() -> Dict[str, Dict[str, List[str]]]:
    decor = load_semantic_taxonomy().get("decor") if isinstance(load_semantic_taxonomy().get("decor"), dict) else {}
    rules: Dict[str, Dict[str, List[str]]] = {}
    for kind, rule in dict(decor.get("kind_rules") or {}).items():
        if not isinstance(rule, dict):
            continue
        rules[_normalize_token(kind)] = {str(key): string_list(value) for key, value in rule.items()}
    return rules


def scene_policy_tokens(policy: str, key: str) -> set[str]:
    policies = load_semantic_taxonomy().get("scene_policies")
    if not isinstance(policies, dict):
        return set()
    values = dict(policies.get(_normalize_token(policy)) or {}).get(key)
    return set(string_list(values))


def tokens_match_scene_policy(policy: str, tokens: Iterable[Any]) -> bool:
    return bool({_normalize_token(token) for token in tokens if _normalize_token(token)} & scene_policy_tokens(policy, "asset_tokens"))


def scene_allows_policy(policy: str, scene_tokens: Iterable[Any], negative_tokens: Iterable[Any] = ()) -> bool:
    normalized_scene = {_normalize_token(token) for token in scene_tokens if _normalize_token(token)}
    normalized_negative = {_normalize_token(token) for token in negative_tokens if _normalize_token(token)}
    if normalized_negative & scene_policy_tokens(policy, "negative_tokens"):
        return False
    return bool(normalized_scene & scene_policy_tokens(policy, "allow_scene_tokens"))
