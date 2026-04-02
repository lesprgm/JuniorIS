from __future__ import annotations

import json
import pathlib
from typing import Any, Dict

from jsonschema import Draft7Validator

from src.stylekit_registry import load_stylekit_registry


ROOT = pathlib.Path(__file__).resolve().parents[1]
API_RESPONSE_SCHEMA_PATH = ROOT / 'schemas' / 'api_response_v0.2.schema.json'
MANIFEST_SCHEMA_PATH = ROOT / 'schemas' / 'manifest_v0.2.schema.json'

_SCHEMA_CACHE: Dict[str, Dict[str, Any]] = {}


def _load_schema(path: pathlib.Path) -> Dict[str, Any]:
    key = str(path)
    cached = _SCHEMA_CACHE.get(key)
    if cached is None:
        cached = json.loads(path.read_text(encoding='utf-8'))
        _SCHEMA_CACHE[key] = cached
    return cached


def validate_schema(payload: Dict[str, Any], schema_path: pathlib.Path) -> Dict[str, Any]:
    validator = Draft7Validator(_load_schema(schema_path))
    errors = []
    for err in sorted(validator.iter_errors(payload), key=lambda item: list(item.path)):
        path = '$'
        for part in err.path:
            path += f'[{part}]' if isinstance(part, int) else f'.{part}'
        errors.append({'path': path, 'message': err.message})
    return {'ok': not errors, 'errors': errors}


def validate_api_response_contract(payload: Dict[str, Any]) -> Dict[str, Any]:
    return validate_schema(payload, API_RESPONSE_SCHEMA_PATH)


def validate_manifest_contract(payload: Dict[str, Any]) -> Dict[str, Any]:
    return validate_schema(payload, MANIFEST_SCHEMA_PATH)


def _normalize_runtime_polish(stylekit: Dict[str, Any] | None) -> Dict[str, Any]:
    stylekit = stylekit or {}
    ambience = stylekit.get('ambience') if isinstance(stylekit.get('ambience'), dict) else None
    decals = stylekit.get('decals') if isinstance(stylekit.get('decals'), list) else []
    postfx = stylekit.get('postfx') if isinstance(stylekit.get('postfx'), dict) else None
    perf_overrides = stylekit.get('perf_overrides') if isinstance(stylekit.get('perf_overrides'), dict) else {}
    return {
        'ambience': ambience,
        'decals': decals,
        'postfx': postfx,
        'perf_overrides': perf_overrides,
    }


def _empty_stylekit_payload() -> Dict[str, Any]:
    return {
        'stylekit_id': None,
        'lighting': None,
        'palette': None,
        'skybox': None,
        'runtime_polish': _normalize_runtime_polish(None),
    }


def _normalized_contract_object(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def resolve_stylekit_runtime_payload(stylekit_id: str | None) -> Dict[str, Any]:
    if not stylekit_id:
        return _empty_stylekit_payload()

    registry = load_stylekit_registry()
    stylekit = registry.get_stylekit(stylekit_id)
    if not isinstance(stylekit, dict):
        payload = _empty_stylekit_payload()
        payload['stylekit_id'] = stylekit_id
        return payload

    return {
        'stylekit_id': stylekit_id,
        'lighting': stylekit.get('lighting'),
        'palette': stylekit.get('palette'),
        'skybox': stylekit.get('skybox'),
        'runtime_polish': _normalize_runtime_polish(stylekit),
    }


def _base_manifest_payload(payload: Dict[str, Any], *, manifest_version: str) -> Dict[str, Any]:
    return {
        'manifest_version': manifest_version,
        'generated_at_utc': payload.get('generated_at_utc'),
        'world_id': payload.get('world_id'),
        'portal_ready_at_phase': payload.get('portal_ready_at_phase', 'phase0'),
        'worldspec_version': payload.get('worldspec_version'),
        'template_id': payload.get('template_id'),
        'safe_spawn': payload.get('safe_spawn'),
        'phase_order': payload.get('phase_order') or ['phase0'],
        'phases': payload.get('phases') or {},
        'planner_backend': payload.get('planner_backend'),
        'semantic_path_status': payload.get('semantic_path_status'),
        'fallback_used': payload.get('fallback_used'),
        'fallback_reason': payload.get('fallback_reason'),
        'candidate_asset_ids': payload.get('candidate_asset_ids') or [],
        'prompt_plan': payload.get('prompt_plan') or {},
        'budgets': payload.get('budgets') or {},
        'placement_intent': _normalized_contract_object(payload.get('placement_intent')),
        'placement_plan': _normalized_contract_object(payload.get('placement_plan')),
    }


def parse_manifest_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    manifest_version = str(payload.get('manifest_version') or '0.1')
    stylekit_block = payload.get('stylekit') if isinstance(payload.get('stylekit'), dict) else {}
    runtime_polish = payload.get('runtime_polish') if isinstance(payload.get('runtime_polish'), dict) else None

    if manifest_version == '0.1':
        resolved_stylekit = resolve_stylekit_runtime_payload(stylekit_block.get('stylekit_id')) if stylekit_block else _empty_stylekit_payload()
        stylekit_payload = {
            'stylekit_id': stylekit_block.get('stylekit_id') if stylekit_block else None,
            'lighting': stylekit_block.get('lighting') if stylekit_block else resolved_stylekit.get('lighting'),
            'palette': stylekit_block.get('palette') if stylekit_block else resolved_stylekit.get('palette'),
            'skybox': stylekit_block.get('skybox') if stylekit_block else resolved_stylekit.get('skybox'),
        }
        return {
            **_base_manifest_payload(payload, manifest_version='0.2'),
            'stylekit': stylekit_payload,
            'runtime_polish': runtime_polish or resolved_stylekit.get('runtime_polish') or _normalize_runtime_polish(None),
        }

    return {
        **_base_manifest_payload(payload, manifest_version=manifest_version),
        'stylekit': stylekit_block,
        'runtime_polish': runtime_polish or _normalize_runtime_polish(None),
    }
