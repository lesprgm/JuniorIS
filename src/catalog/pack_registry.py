from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List

from jsonschema import Draft7Validator


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]  # climb from src/catalog/ to the repo root
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "pack_manifest_v0.schema.json"  # JSON schema that every pack.json must satisfy


@dataclass
# Keep behavior deterministic so planner/runtime contracts stay stable.
class PackRegistry:
    packs_by_id: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # pack_id -> full manifest dict
    assets_by_id: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # asset_id -> {pack_id, asset} lookup
    tags_index: Dict[str, List[str]] = field(default_factory=dict)  # pack_id -> tag list for search filtering
    errors: List[Dict[str, str]] = field(default_factory=list)  # accumulated loading/validation errors

    def search_packs(self, tags: List[str]) -> List[str]:  # find packs whose tags fully contain the query set
        if not tags:
            return sorted(self.packs_by_id.keys())
        required = set(tags)
        matches = []
        for pack_id, pack_tags in self.tags_index.items():
            if required.issubset(set(pack_tags)):
                matches.append(pack_id)
        return sorted(matches)


_SCHEMA_CACHE: Dict[str, Any] | None = None  # cached to avoid re-reading schema JSON on every call


def _load_schema(schema_path: pathlib.Path) -> Dict[str, Any]:  # lazy-loads and caches the pack JSON schema
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        _SCHEMA_CACHE = json.loads(schema_path.read_text(encoding="utf-8"))
    return _SCHEMA_CACHE


def _format_error_path(path_parts: List[Any]) -> str:  # converts jsonschema path tuples into dot-bracket notation like $.assets[0].role
    if not path_parts:
        return "$"
    out = "$"
    for p in path_parts:
        if isinstance(p, int):
            out += f"[{p}]"
        else:
            out += f".{p}"
    return out


def load_pack_registry(
    packs_root: str | pathlib.Path = "packs",
    schema_path: str | pathlib.Path = DEFAULT_SCHEMA_PATH,
) -> PackRegistry:
    packs_dir = pathlib.Path(packs_root)
    schema_file = pathlib.Path(schema_path)
    schema = _load_schema(schema_file)  # validate every manifest against the pack schema
    validator = Draft7Validator(schema)

    registry = PackRegistry()

    if not packs_dir.exists():
        registry.errors.append(
            {"path": str(packs_dir), "message": "packs directory does not exist"}
        )
        return registry

    manifest_paths = sorted(packs_dir.glob("**/pack.json"))  # sorted for deterministic registration order
    for manifest_path in manifest_paths:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            registry.errors.append(
                {"path": str(manifest_path), "message": f"invalid JSON: {exc}"}
            )
            continue

        schema_errors = sorted(validator.iter_errors(manifest), key=lambda e: list(e.path))
        if schema_errors:
            for err in schema_errors:
                registry.errors.append(
                    {
                        "path": f"{manifest_path}:{_format_error_path(list(err.path))}",
                        "message": err.message,
                    }
                )
            continue

        pack_id = manifest["pack_id"]  # guaranteed present after schema validation
        if pack_id in registry.packs_by_id:
            registry.errors.append(
                {
                    "path": str(manifest_path),
                    "message": f"duplicate pack_id '{pack_id}' skipped",
                }
            )
            continue

        pack_copy = dict(manifest)  # shallow copy so we can attach metadata without mutating the original
        pack_copy["_manifest_path"] = str(manifest_path)  # stash source path for debugging and substitution hints
        registry.packs_by_id[pack_id] = pack_copy
        registry.tags_index[pack_id] = list(manifest.get("tags", []))

        for asset in manifest.get("assets", []):
            asset_id = asset["asset_id"]
            if asset_id in registry.assets_by_id:
                registry.errors.append(
                    {
                        "path": str(manifest_path),
                        "message": (
                            f"duplicate asset_id '{asset_id}' skipped "
                            f"(already in pack '{registry.assets_by_id[asset_id]['pack_id']}')"
                        ),
                    }
                )
                continue

            registry.assets_by_id[asset_id] = {
                "pack_id": pack_id,
                "asset": asset,
            }

    return registry
