from __future__ import annotations

import json
import pathlib
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.compilation.phase0 import compile_phase0
from src.planning.planner import plan_worldspec
from src.contracts.runtime import resolve_stylekit_runtime_payload

BUILD_ROOT = pathlib.Path("build")
PORTAL_READY_PHASE = "phase0"
API_CONTRACT_VERSION = "0.2"
_ERROR_RECOVERABLE = {
    "invalid_request": True,
    "planner_failed": True,
    "compile_failed": True,
    "spawn_failed": True,
    "manifest_failed": True,
    "internal_error": False,
}
_ERROR_RETRYABLE = {
    "invalid_request": False,
    "planner_failed": True,
    "compile_failed": True,
    "spawn_failed": True,
    "manifest_failed": True,
    "internal_error": False,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:12]}"


def _new_trace_id() -> str:
    return f"trace_{uuid.uuid4().hex[:12]}"


def _error(
    error_code: str,
    user_message: str,
    request_id: str,
    trace_id: str,
    errors: Optional[list[dict[str, Any]]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    details = errors or []
    retryable = _ERROR_RETRYABLE.get(error_code, False)
    payload: Dict[str, Any] = {
        "api_contract_version": API_CONTRACT_VERSION,
        "ok": False,
        "request_id": request_id,
        "trace_id": trace_id,
        "error_code": error_code,
        "user_message": user_message,
        "recoverable": _ERROR_RECOVERABLE.get(error_code, False),
        "retryable": retryable,
        "retry_after_ms": 1000 if retryable else None,
        "details": details,
        "errors": details,
    }
    if isinstance(extra, dict):
        payload.update(extra)
    return payload


def _planner_response_fields(planner_result: Dict[str, Any]) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    for key, expected_type in (
        ("prompt_plan", dict),
        ("planner_backend", str),
        ("candidate_asset_ids", list),
        ("semantic_receipts", dict),
        ("semantic_path_status", str),
        ("fallback_used", bool),
    ):
        value = planner_result.get(key)
        if isinstance(value, expected_type):
            fields[key] = value

    fallback_reason = planner_result.get("fallback_reason")
    if fallback_reason is None or isinstance(fallback_reason, str):
        fields["fallback_reason"] = fallback_reason
    return fields


def _planner_context(planner_result: Dict[str, Any]) -> tuple[Dict[str, Any], str | None]:
    return _planner_response_fields(planner_result), planner_result.get("error_code")


def _prompt_plan_manifest(prompt_plan: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mode": prompt_plan.get("mode"),
        "strategy": prompt_plan.get("strategy"),
        "selected_prompt": prompt_plan.get("selected_prompt"),
        "selected_variant_index": prompt_plan.get("selected_variant_index"),
    }


def _normalized_manifest_object(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _build_readiness(compile_result: Dict[str, Any]) -> Dict[str, Any]:
    # Unity only opens the portal once phase0 exists and a safe spawn was found.
    phase0_ready = bool(compile_result.get("ok"))
    safe_spawn_ready = isinstance(compile_result.get("safe_spawn"), dict)
    blocked_reasons = []
    if not phase0_ready:
        blocked_reasons.append("phase0_not_ready")
    if not safe_spawn_ready:
        blocked_reasons.append("safe_spawn_not_ready")
    portal_allowed = phase0_ready and safe_spawn_ready and not blocked_reasons
    return {
        "phase0_ready": phase0_ready,
        "safe_spawn_ready": safe_spawn_ready,
        "portal_allowed": portal_allowed,
        "blocked_reasons": blocked_reasons,
    }


def _write_manifest(
    build_root: pathlib.Path,
    world_id: str,
    worldspec: Dict[str, Any],
    compile_result: Dict[str, Any],
    prompt_plan: Optional[Dict[str, Any]] = None,
    planner_backend: Optional[str] = None,
    candidate_asset_ids: Optional[list[str]] = None,
    semantic_path_status: Optional[str] = None,
    fallback_used: Optional[bool] = None,
    fallback_reason: Optional[str] = None,
) -> pathlib.Path:
    # Keep the manifest narrow and runtime-focused so Unity does not need to
    # understand the full backend planner/compiler internals.
    world_dir = build_root / world_id
    world_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = world_dir / "manifest.json"

    phase0_artifact = compile_result.get("phase0_artifact")
    phase0_filename = pathlib.Path(phase0_artifact).name if phase0_artifact else "phase0.json"

    manifest_payload = {
        "manifest_version": "0.2",
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
        "budgets": worldspec.get("budgets", {}),
        "placement_intent": _normalized_manifest_object(worldspec.get("placement_intent")),
        "placement_plan": _normalized_manifest_object(worldspec.get("placement_plan")),
    }
    if isinstance(prompt_plan, dict):
        manifest_payload["prompt_plan"] = _prompt_plan_manifest(prompt_plan)
    if isinstance(planner_backend, str) and planner_backend:
        manifest_payload["planner_backend"] = planner_backend
    if isinstance(candidate_asset_ids, list) and candidate_asset_ids:
        manifest_payload["candidate_asset_ids"] = candidate_asset_ids[:40]
    if isinstance(semantic_path_status, str):
        manifest_payload["semantic_path_status"] = semantic_path_status
    if isinstance(fallback_used, bool):
        manifest_payload["fallback_used"] = fallback_used
    if fallback_reason is None or isinstance(fallback_reason, str):
        manifest_payload["fallback_reason"] = fallback_reason
    stylekit_payload = resolve_stylekit_runtime_payload(worldspec.get("stylekit_id"))
    manifest_payload["stylekit"] = {
        "stylekit_id": stylekit_payload.get("stylekit_id"),
        "lighting": stylekit_payload.get("lighting"),
        "palette": stylekit_payload.get("palette"),
        "skybox": stylekit_payload.get("skybox"),
    }
    manifest_payload["runtime_polish"] = stylekit_payload.get("runtime_polish", {})

    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _compile_error_response(
    compile_result: Dict[str, Any],
    request_id: str,
    trace_id: str,
    planner_extra: Dict[str, Any],
) -> Dict[str, Any]:
    compile_errors = compile_result.get("errors", [])
    has_spawn_error = any(
        isinstance(err, dict) and str(err.get("path", "")).startswith("$.safe_spawn")
        for err in compile_errors
    )
    return _error(
        "spawn_failed" if has_spawn_error else "compile_failed",
        "World spawn preparation failed."
        if has_spawn_error
        else "World compilation failed before destination was playable.",
        request_id,
        trace_id,
        compile_errors,
        extra=planner_extra if planner_extra else None,
    )


def _status_code_for_error(error_code: Any) -> int:
    if error_code == "invalid_request":
        return 400
    if error_code in {"planner_failed", "compile_failed", "spawn_failed"}:
        return 422
    return 500


def _invalid_request_error(
    request_id: str,
    trace_id: str,
    user_message: str,
    path: str,
    message: str,
) -> Dict[str, Any]:
    return _error(
        "invalid_request",
        user_message,
        request_id,
        trace_id,
        [{"path": path, "message": message}],
    )


def run_plan_and_compile(
    prompt_text: str,
    optional_seed: Optional[int] = None,
    user_prefs: Optional[Dict[str, Any]] = None,
    build_root: pathlib.Path | str = BUILD_ROOT,
) -> Dict[str, Any]:
    # Request-scoped ids start here so every downstream artifact and error
    # response can be traced back to one submission.
    request_id = _new_request_id()
    trace_id = _new_trace_id()

    prompt = (prompt_text or "").strip()
    if not prompt:
        return _invalid_request_error(
            request_id,
            trace_id,
            "Prompt is empty.",
            "$.prompt_text",
            "prompt_text must be a non-empty string",
        )

    if optional_seed is not None and (isinstance(optional_seed, bool) or not isinstance(optional_seed, int)):
        return _invalid_request_error(
            request_id,
            trace_id,
            "Seed must be an integer.",
            "$.optional_seed",
            "optional_seed must be an integer",
        )

    if optional_seed is not None and (optional_seed < 0 or optional_seed > 2_147_483_647):
        return _invalid_request_error(
            request_id,
            trace_id,
            "Seed is outside supported range.",
            "$.optional_seed",
            "optional_seed must be between 0 and 2147483647",
        )

    normalized_prefs: Dict[str, Any] = user_prefs if isinstance(user_prefs, dict) else {}

    try:
        # The planner owns semantic selection; compile_phase0 only sees a
        # validated WorldSpec contract.
        planner_result = plan_worldspec(prompt, seed=optional_seed, user_prefs=normalized_prefs)
    except Exception:
        return _error(
            "internal_error",
            "Unexpected planner error.",
            request_id,
            trace_id,
            [{"path": "$.planner", "message": "unhandled planner exception"}],
        )
    prompt_plan = planner_result.get("prompt_plan")
    planner_backend = planner_result.get("planner_backend")
    candidate_asset_ids = planner_result.get("candidate_asset_ids")
    semantic_receipts = planner_result.get("semantic_receipts")
    semantic_path_status = planner_result.get("semantic_path_status")
    fallback_used = planner_result.get("fallback_used")
    fallback_reason = planner_result.get("fallback_reason")
    planner_extra, planner_error_code = _planner_context(planner_result)
    if not planner_result.get("ok"):
        user_message = "Could not build a valid world plan for this prompt."
        if isinstance(planner_error_code, str) and planner_error_code.startswith("llm_"):
            user_message = "LLM planner failed for this prompt. Retry or switch planner mode."
        return _error(
            "planner_failed",
            user_message,
            request_id,
            trace_id,
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
            request_id,
            trace_id,
            [{"path": "$.worldspec", "message": "planner result missing worldspec object"}],
            extra=planner_extra,
        )

    try:
        compile_result = compile_phase0(worldspec, build_root=build_root, write_artifact=True)
    except Exception:
        return _error(
            "internal_error",
            "Unexpected compiler error.",
            request_id,
            trace_id,
            [{"path": "$.compile", "message": "unhandled compiler exception"}],
        )
    if not compile_result.get("ok"):
        return _compile_error_response(compile_result, request_id, trace_id, planner_extra)

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
            semantic_path_status=semantic_path_status if isinstance(semantic_path_status, str) else None,
            fallback_used=fallback_used if isinstance(fallback_used, bool) else None,
            fallback_reason=fallback_reason if fallback_reason is None or isinstance(fallback_reason, str) else None,
        )
    except OSError as exc:
        return _error(
            "manifest_failed",
            "Could not write world manifest.",
            request_id,
            trace_id,
            [{"path": "$.manifest", "message": str(exc)}],
        )
    except Exception:
        return _error(
            "internal_error",
            "Unexpected manifest error.",
            request_id,
            trace_id,
            [{"path": "$.manifest", "message": "unhandled manifest exception"}],
        )

    readiness = _build_readiness(compile_result)
    response = {
        "api_contract_version": API_CONTRACT_VERSION,
        "ok": True,
        "request_id": request_id,
        "trace_id": trace_id,
        "world_id": world_id,
        "manifest_url": f"/build/{world_id}/manifest.json",
        "manifest_path": str(manifest_path),
        "portal_ready_at_phase": PORTAL_READY_PHASE,
        "readiness": readiness,
        "errors": [],
    }
    response.update(_planner_response_fields(planner_result))
    if isinstance(prompt_plan, dict):
        response["selected_prompt"] = prompt_plan.get("selected_prompt")
    return response


try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - optional dependency for API serving
    FastAPI = None
    BaseModel = object
    Field = None
    JSONResponse = None
    StaticFiles = None
    app = None


if FastAPI is not None:
    class PlanAndCompileRequest(BaseModel):
        prompt_text: str
        optional_seed: Optional[int] = None
        user_prefs: Dict[str, Any] = Field(default_factory=dict)


    app = FastAPI(title="JuniorIS Planner/Compiler API", version=API_CONTRACT_VERSION)
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    app.mount("/build", StaticFiles(directory=str(BUILD_ROOT), check_dir=True), name="build")

    @app.get("/healthz")
    def healthz():
        return {
            "ok": True,
            "api_contract_version": API_CONTRACT_VERSION,
            "build_root": str(BUILD_ROOT),
        }

    @app.post("/plan_and_compile")
    def plan_and_compile(payload: PlanAndCompileRequest):
        result = run_plan_and_compile(
            prompt_text=payload.prompt_text,
            optional_seed=payload.optional_seed,
            user_prefs=payload.user_prefs,
        )

        if result.get("ok"):
            return result

        status_code = _status_code_for_error(result.get("error_code"))
        return JSONResponse(content=result, status_code=status_code)


def main() -> int:
    if FastAPI is None:
        raise RuntimeError("FastAPI is not installed. Install fastapi and uvicorn to run the API server.")

    import uvicorn  # type: ignore

    uvicorn.run("src.api.server:app", host="127.0.0.1", port=8000, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
