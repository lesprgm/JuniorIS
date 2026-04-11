from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List

from jsonschema import Draft7Validator


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "stylekit_manifest_v0.schema.json"


@dataclass
# Keep behavior deterministic so planner/runtime contracts stay stable.
class StyleKitRegistry:
    stylekits_by_id: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # stylekit_id -> full stylekit manifest dict
    tags_index: Dict[str, List[str]] = field(default_factory=dict)
    errors: List[Dict[str, str]] = field(default_factory=list)  # accumulated loading/validation errors

    def list_stylekits(self) -> List[str]:
        return sorted(self.stylekits_by_id.keys())

    def get_stylekit(self, stylekit_id: str) -> Dict[str, Any] | None:  # returns full manifest dict or None if not registered
        return self.stylekits_by_id.get(stylekit_id)

    def search_stylekits(self, tags: List[str]) -> List[str]:
        if not tags:
            return self.list_stylekits()
        required = set(tags)
        matches = []
        for stylekit_id, style_tags in self.tags_index.items():
            if required.issubset(set(style_tags)):
                matches.append(stylekit_id)
        return sorted(matches)


_SCHEMA_CACHE: Dict[str, Any] | None = None  # module-level cache so schema is read from disk only once


def _load_schema(schema_path: pathlib.Path) -> Dict[str, Any]:
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        _SCHEMA_CACHE = json.loads(schema_path.read_text(encoding="utf-8"))
    return _SCHEMA_CACHE


def _format_error_path(path_parts: List[Any]) -> str:
    if not path_parts:
        return "$"
    out = "$"
    for part in path_parts:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            out += f".{part}"
    return out


def _normalize_ambience(value: Any) -> Dict[str, Any] | None:
    if isinstance(value, str) and value.strip():
        return {"clip_id": value.strip(), "volume": 0.2}
    if isinstance(value, dict):
        clip_id = str(value.get("clip_id") or "").strip()
        if not clip_id:
            return None
        volume = value.get("volume", 0.2)
        if not isinstance(volume, (int, float)):
            volume = 0.2
        return {"clip_id": clip_id, "volume": max(0.0, min(float(volume), 1.0))}
    return None


def _normalize_decals(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    decals: List[Dict[str, Any]] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        decal_id = str(entry.get("decal_id") or "").strip()
        max_count = entry.get("max_count")
        if not decal_id or not isinstance(max_count, int) or max_count < 0:
            continue
        decals.append({"decal_id": decal_id, "max_count": max_count})
    return decals


def _normalize_postfx(value: Any) -> Dict[str, Any] | None:
    if isinstance(value, str) and value.strip():
        return {"profile_id": value.strip()}
    if isinstance(value, dict):
        profile_id = str(value.get("profile_id") or "").strip()
        if profile_id:
            return {"profile_id": profile_id}
    return None


def _normalize_stylekit_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    stylekit = dict(manifest)
    stylekit["ambience"] = _normalize_ambience(manifest.get("ambience"))
    stylekit["decals"] = _normalize_decals(manifest.get("decals"))
    stylekit["postfx"] = _normalize_postfx(manifest.get("postfx"))
    perf_overrides = manifest.get("perf_overrides")
    stylekit["perf_overrides"] = dict(perf_overrides) if isinstance(perf_overrides, dict) else {}
    return stylekit


def load_stylekit_registry(
    stylekits_root: str | pathlib.Path = "stylekits",
    schema_path: str | pathlib.Path = DEFAULT_SCHEMA_PATH,
) -> StyleKitRegistry:
    stylekits_dir = pathlib.Path(stylekits_root)
    schema_file = pathlib.Path(schema_path)
    schema = _load_schema(schema_file)
    validator = Draft7Validator(schema)

    registry = StyleKitRegistry()

    if not stylekits_dir.exists():
        registry.errors.append(
            {"path": str(stylekits_dir), "message": "stylekits directory does not exist"}
        )
        return registry

    manifest_paths = sorted(stylekits_dir.glob("**/stylekit.json"))
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

        stylekit_id = manifest["stylekit_id"]
        if stylekit_id in registry.stylekits_by_id:
            registry.errors.append(
                {
                    "path": str(manifest_path),
                    "message": f"duplicate stylekit_id '{stylekit_id}' skipped",
                }
            )
            continue

        stylekit_copy = _normalize_stylekit_manifest(manifest)
        stylekit_copy["_manifest_path"] = str(manifest_path)
        registry.stylekits_by_id[stylekit_id] = stylekit_copy
        registry.tags_index[stylekit_id] = list(manifest.get("tags", []))

    return registry
