from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.compiler_phase0 import compile_phase0
from src.planner import plan_worldspec

BUILD_ROOT = pathlib.Path("build")
PORTAL_READY_PHASE = "phase0"
_ERROR_RECOVERABLE = {
    "invalid_request": True,
    "planner_failed": True,
    "compile_failed": True,
    "spawn_failed": True,
    "manifest_failed": True,
    "internal_error": False,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _error(
    error_code: str,
    user_message: str,
    errors: Optional[list[dict[str, Any]]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    details = errors or []
    payload: Dict[str, Any] = {
        "ok": False,
        "error_code": error_code,
        "user_message": user_message,
        "recoverable": _ERROR_RECOVERABLE.get(error_code, False),
        "details": details,
        "errors": details,
    }
    if isinstance(extra, dict):
        payload.update(extra)
    return payload


def _write_manifest(
    build_root: pathlib.Path,
    world_id: str,
    worldspec: Dict[str, Any],
    compile_result: Dict[str, Any],
    prompt_plan: Optional[Dict[str, Any]] = None,
    planner_backend: Optional[str] = None,
    candidate_asset_ids: Optional[list[str]] = None,
) -> pathlib.Path:
    world_dir = build_root / world_id
    world_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = world_dir / "manifest.json"

    phase0_artifact = compile_result.get("phase0_artifact")
    phase0_filename = pathlib.Path(phase0_artifact).name if phase0_artifact else "phase0.json"

    manifest_payload = {
        "manifest_version": "0.1",
        "generated_at_utc": _utc_now_iso(),
        "world_id": world_id,
        "portal_ready_at_phase": PORTAL_READY_PHASE,
        "worldspec_version": worldspec.get("worldspec_version"),
        "template_id": worldspec.get("template_id"),
        "safe_spawn": compile_result.get("safe_spawn"),
        "phase_order": ["phase0"],
        "phases": {
            "phase0": {
                "artifact": phase0_filename,
                "teleportable_surfaces": compile_result.get("teleportable_surfaces", 0),
            }
        },
    }
    if isinstance(prompt_plan, dict):
        manifest_payload["prompt_plan"] = {
            "mode": prompt_plan.get("mode"),
            "strategy": prompt_plan.get("strategy"),
            "selected_prompt": prompt_plan.get("selected_prompt"),
            "selected_variant_index": prompt_plan.get("selected_variant_index"),
        }
    if isinstance(planner_backend, str) and planner_backend:
        manifest_payload["planner_backend"] = planner_backend
    if isinstance(candidate_asset_ids, list) and candidate_asset_ids:
        manifest_payload["candidate_asset_ids"] = candidate_asset_ids[:40]

    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def run_plan_and_compile(
    prompt_text: str,
    optional_seed: Optional[int] = None,
    user_prefs: Optional[Dict[str, Any]] = None,
    build_root: pathlib.Path | str = BUILD_ROOT,
) -> Dict[str, Any]:
    prompt = (prompt_text or "").strip()
    if not prompt:
        return _error(
            "invalid_request",
            "Prompt is empty.",
            [{"path": "$.prompt_text", "message": "prompt_text must be a non-empty string"}],
        )

    if optional_seed is not None and (isinstance(optional_seed, bool) or not isinstance(optional_seed, int)):
        return _error(
            "invalid_request",
            "Seed must be an integer.",
            [{"path": "$.optional_seed", "message": "optional_seed must be an integer"}],
        )

    if optional_seed is not None and (optional_seed < 0 or optional_seed > 2_147_483_647):
        return _error(
            "invalid_request",
            "Seed is outside supported range.",
            [{"path": "$.optional_seed", "message": "optional_seed must be between 0 and 2147483647"}],
        )

    normalized_prefs: Dict[str, Any] = user_prefs if isinstance(user_prefs, dict) else {}

    try:
        planner_result = plan_worldspec(prompt, seed=optional_seed, user_prefs=normalized_prefs)
    except Exception:
        return _error(
            "internal_error",
            "Unexpected planner error.",
            [{"path": "$.planner", "message": "unhandled planner exception"}],
        )
    prompt_plan = planner_result.get("prompt_plan") if isinstance(planner_result, dict) else None
    planner_backend = planner_result.get("planner_backend") if isinstance(planner_result, dict) else None
    candidate_asset_ids = (
        planner_result.get("candidate_asset_ids") if isinstance(planner_result, dict) else None
    )
    planner_extra: Dict[str, Any] = {}
    if isinstance(prompt_plan, dict):
        planner_extra["prompt_plan"] = prompt_plan
    if isinstance(planner_backend, str):
        planner_extra["planner_backend"] = planner_backend
    if isinstance(candidate_asset_ids, list):
        planner_extra["candidate_asset_ids"] = candidate_asset_ids
    planner_error_code = planner_result.get("error_code") if isinstance(planner_result, dict) else None
    if not planner_result.get("ok"):
        user_message = "Could not build a valid world plan for this prompt."
        if isinstance(planner_error_code, str) and planner_error_code.startswith("llm_"):
            user_message = "LLM planner failed for this prompt. Retry or switch planner mode."
        return _error(
            "planner_failed",
            user_message,
            planner_result.get("errors", []),
            extra={
                **planner_extra,
                "planner_error_code": planner_error_code,
            }
            if planner_extra or planner_error_code
            else None,
        )

    worldspec = planner_result.get("worldspec")
    if not isinstance(worldspec, dict):
        return _error(
            "planner_failed",
            "Planner returned an invalid world specification.",
            [{"path": "$.worldspec", "message": "planner result missing worldspec object"}],
            extra=planner_extra,
        )

    try:
        compile_result = compile_phase0(worldspec, build_root=build_root, write_artifact=True)
    except Exception:
        return _error(
            "internal_error",
            "Unexpected compiler error.",
            [{"path": "$.compile", "message": "unhandled compiler exception"}],
        )
    if not compile_result.get("ok"):
        compile_errors = compile_result.get("errors", [])
        has_spawn_error = any(
            isinstance(err, dict) and str(err.get("path", "")).startswith("$.safe_spawn")
            for err in compile_errors
        )
        return _error(
            "spawn_failed" if has_spawn_error else "compile_failed",
            "World spawn preparation failed." if has_spawn_error else "World compilation failed before destination was playable.",
            compile_errors,
        )

    world_id = str(compile_result["world_id"])
    try:
        manifest_path = _write_manifest(
            pathlib.Path(build_root),
            world_id,
            worldspec,
            compile_result,
            prompt_plan=prompt_plan if isinstance(prompt_plan, dict) else None,
            planner_backend=planner_backend if isinstance(planner_backend, str) else None,
            candidate_asset_ids=candidate_asset_ids if isinstance(candidate_asset_ids, list) else None,
        )
    except OSError as exc:
        return _error(
            "manifest_failed",
            "Could not write world manifest.",
            [{"path": "$.manifest", "message": str(exc)}],
        )
    except Exception:
        return _error(
            "internal_error",
            "Unexpected manifest error.",
            [{"path": "$.manifest", "message": "unhandled manifest exception"}],
        )

    response = {
        "ok": True,
        "world_id": world_id,
        "manifest_url": f"/build/{world_id}/manifest.json",
        "manifest_path": str(manifest_path),
        "portal_ready_at_phase": PORTAL_READY_PHASE,
        "errors": [],
    }
    if isinstance(prompt_plan, dict):
        response["prompt_plan"] = prompt_plan
        response["selected_prompt"] = prompt_plan.get("selected_prompt")
    if isinstance(planner_backend, str):
        response["planner_backend"] = planner_backend
    if isinstance(candidate_asset_ids, list):
        response["candidate_asset_ids"] = candidate_asset_ids
    return response


try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - optional dependency for API serving
    FastAPI = None
    BaseModel = object
    Field = None
    JSONResponse = None


if FastAPI is not None:
    class PlanAndCompileRequest(BaseModel):
        prompt_text: str
        optional_seed: Optional[int] = None
        user_prefs: Dict[str, Any] = Field(default_factory=dict)


    app = FastAPI(title="JuniorIS Planner/Compiler API", version="0.1")

    @app.post("/plan_and_compile")
    def plan_and_compile(payload: PlanAndCompileRequest):
        result = run_plan_and_compile(
            prompt_text=payload.prompt_text,
            optional_seed=payload.optional_seed,
            user_prefs=payload.user_prefs,
        )

        if result.get("ok"):
            return result

        code = result.get("error_code")
        status_code = (
            400
            if code == "invalid_request"
            else 422
            if code in {"planner_failed", "compile_failed", "spawn_failed"}
            else 500
        )
        return JSONResponse(content=result, status_code=status_code)


def main() -> int:
    if FastAPI is None:
        raise RuntimeError("FastAPI is not installed. Install fastapi and uvicorn to run the API server.")

    import uvicorn  # type: ignore

    uvicorn.run("src.api_server:app", host="127.0.0.1", port=8000, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
