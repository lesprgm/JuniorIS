from __future__ import annotations

import json
import pathlib
import sys
from typing import Any, Dict, List

from jsonschema import Draft7Validator


SCHEMA_PATH = pathlib.Path(__file__).resolve().parents[1] / "schemas" / "worldspec_v0.schema.json"


_SCHEMA_CACHE: Dict[str, Any] | None = None


def _load_schema() -> Dict[str, Any]:
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        _SCHEMA_CACHE = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return _SCHEMA_CACHE


def _format_error_path(path_parts: List[Any]) -> str:
    if not path_parts:
        return "$"
    out = "$"
    for p in path_parts:
        if isinstance(p, int):
            out += f"[{p}]"
        else:
            out += f".{p}"
    return out


def validate_worldspec(data: Dict[str, Any]) -> Dict[str, Any]:
    schema = _load_schema()
    validator = Draft7Validator(schema)
    errors = []
    for err in sorted(validator.iter_errors(data), key=lambda e: list(e.path)):
        errors.append(
            {
                "path": _format_error_path(list(err.path)),
                "message": err.message,
            }
        )
    return {"ok": len(errors) == 0, "errors": errors}


def _main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python3 -m src.validate_worldspec path/to/worldspec.json", file=sys.stderr)
        return 2

    input_path = pathlib.Path(sys.argv[1])
    if not input_path.exists():
        print(f"File not found: {input_path}", file=sys.stderr)
        return 2

    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON: {exc}", file=sys.stderr)
        return 2

    result = validate_worldspec(payload)
    if result["ok"]:
        print("OK: WorldSpec is valid.")
        return 0

    print("INVALID: WorldSpec failed validation.")
    for err in result["errors"]:
        print(f"- {err['path']}: {err['message']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
