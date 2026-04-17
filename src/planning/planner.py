from __future__ import annotations

from typing import Any, Dict, List

from src.llm.planner import request_llm_design_brief, request_llm_intent, request_llm_selection
from src.catalog.pack_registry import load_pack_registry
from src.placement.geometry import canonicalize_semantic_role, map_semantic_concept_to_runtime_role
from src.planning import assets as planner_assets
from src.planning import semantics as planner_semantics
from src.planning.utils import normalize_bool as _normalize_bool, seed_from_prompt as _seed_from_prompt
from src.catalog.stylekit_registry import load_stylekit_registry
from src.runtime.decor_plan import build_decor_asset_ids_by_kind, build_decor_capabilities
from src.world.validation import validate_worldspec


"""Planner orchestration for prompt -> WorldSpec generation."""


TEMPLATE_ALLOWLIST = ["room_basic"]  # only room_basic is implemented; expand when new templates are added
DEFAULT_TEMPLATE_ID = "room_basic"  # used when user_prefs doesn't specify a template
DEFAULT_BUDGETS = {
    "max_props": 25,
    "max_props_hard": 30,
    "max_floor_objects": 8,
    "max_wall_objects": 4,
    "max_surface_objects": 4,
    "max_texture_tier": 1,
    "max_lights": 2,
    "max_clutter_weight": 5,
}
DEFAULT_PLACEMENT_MODE = "scene_graph_solver"  # deterministic solver consumes the scene graph instead of raw LLM coordinates


# Keep behavior deterministic so planner/runtime contracts stay stable.
def _error_entry(path: str, message: str) -> Dict[str, Any]:  # standardizes internal pipeline errors into a consistent structure
    return {"path": path, "message": message}


def _planner_unavailable(path: str, message: str) -> Dict[str, Any]:  # structures errors when communication with the LLM backend fails
    return {
        "ok": False,
        "error_code": "planner_unavailable",
        "planner_backend": "llm_unavailable",
        "semantic_path_status": "failed",
        "errors": [_error_entry(path, message)],
    }


def _planner_state(  # encodes debugging audit trail of steps taken during the generation process
    *,
    planner_backend: str,
    semantic_path_status: str,
) -> Dict[str, Any]:
    return {
        "planner_backend": planner_backend,
        "semantic_path_status": semantic_path_status,
    }


def _semantic_failure_backend(error_code: str) -> str:  # infers which LLM operation (intent vs selection) caused an abort
    return "llm_unavailable" if error_code.startswith("llm_") else "llm"


def _semantic_failure_result(  # produces early exit status when LLM hallucinated entirely unparseable JSON
    *,
    prompt_plan: Dict[str, Any],
    candidate_assets: List[Dict[str, Any]],
    error_code: str,
    message: str,
    errors: List[Dict[str, Any]] | None = None,
    extra_fields: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    candidate_asset_ids = [str(asset.get("asset_id")) for asset in candidate_assets[:40]]
    payload = {
        "ok": False,
        "error_code": error_code,
        **_planner_state(
            planner_backend=_semantic_failure_backend(error_code),
            semantic_path_status="failed",
        ),
        "prompt_plan": prompt_plan,
        "candidate_asset_ids": candidate_asset_ids,
        "errors": errors or [_error_entry("$.llm", message)],
    }
    if isinstance(extra_fields, dict):
        payload.update(extra_fields)
    return payload


def _invalid_prompt_mode_result(bad_mode: str) -> Dict[str, Any]:  # error output when a non-existent generation strategy was specified
    return {
        "ok": False,
        "error_code": "invalid_prompt_mode",
        **_planner_state(
            planner_backend="llm",
            semantic_path_status="failed",
        ),
        "errors": [
            _error_entry(
                "$.user_prefs.prompt_mode",
                f"Unsupported prompt_mode '{bad_mode}'. Only 'llm' is supported.",
            )
        ],
    }


def _build_worldspec(  # assembles the final WorldSpec dict from all planning outputs
    *,
    template_id: str,
    effective_seed: int,
    stylekit_id: str | None,
    pack_ids: List[str],
    placements: List[Dict[str, Any]],
    budgets: Dict[str, int],
    colors: Dict[str, str],
    planner_policy: Dict[str, Any],
    placement_intent: Dict[str, Any] | None = None,
    placement_plan: Dict[str, Any] | None = None,
    layout_program: Dict[str, Any] | None = None,
    decor_plan: Dict[str, Any] | None = None,
    surface_material_selection: Dict[str, Any] | None = None,
    optional_additions: List[Dict[str, Any]] | None = None,
    scene_context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    worldspec: Dict[str, Any] = {
        "worldspec_version": "0.1",
        "template_id": template_id,
        "seed": int(effective_seed),
        "stylekit_id": stylekit_id,
        "pack_ids": pack_ids,
        "placements": placements,
        "budgets": budgets,
        "planner_policy": planner_policy,
    }
    if colors:
        worldspec["colors"] = colors
    if isinstance(placement_intent, dict):
        worldspec["placement_intent"] = placement_intent
    if isinstance(placement_plan, dict):
        worldspec["placement_plan"] = placement_plan
    if isinstance(layout_program, dict):
        worldspec["layout_program"] = layout_program
    if isinstance(decor_plan, dict):
        worldspec["decor_plan"] = decor_plan
    if isinstance(surface_material_selection, dict):
        worldspec["surface_material_selection"] = surface_material_selection
    if isinstance(optional_additions, list):
        worldspec["optional_additions"] = optional_additions
    if isinstance(scene_context, dict):
        worldspec["scene_context"] = scene_context
    return worldspec


def _budget_overrides(user_prefs: Dict[str, Any]) -> Dict[str, int]:  # applies user-provided budget limits over the defaults
    budgets = dict(DEFAULT_BUDGETS)
    for key in (
        "max_props",
        "max_props_hard",
        "max_floor_objects",
        "max_wall_objects",
        "max_surface_objects",
        "max_texture_tier",
        "max_lights",
        "max_clutter_weight",
    ):
        value = user_prefs.get(key)
        if isinstance(value, int) and value > 0:
            budgets[key] = value
    return budgets


def _required_slot_anchor_count(scene_program: Dict[str, Any]) -> Dict[str, int]:
    semantic_slots = [
        slot
        for slot in list(scene_program.get("semantic_slots") or [])
        if isinstance(slot, dict) and str(slot.get("slot_id") or "").strip()
    ]
    role_counts: Dict[str, int] = {}
    for slot in semantic_slots:
        priority = str(slot.get("priority") or "should").strip().lower()
        if priority == "optional":
            continue
        concept = slot.get("concept") or slot.get("runtime_role_hint") or slot.get("runtime_role")
        runtime_role = canonicalize_semantic_role(
            map_semantic_concept_to_runtime_role(concept)[0]
            or slot.get("runtime_role_hint")
            or concept
        )
        if not runtime_role:
            continue
        role_counts[runtime_role] = max(
            role_counts.get(runtime_role, 0),
            max(1, int(slot.get("count") or 1)),
        )
    counts = {"floor": 0, "wall": 0, "surface": 0, "lights": 0, "clutter_weight": 0}
    for role, count in role_counts.items():
        if role == "sign":
            counts["wall"] += count
            counts["clutter_weight"] += count
            continue
        if role == "decor":
            counts["surface"] += count
            counts["clutter_weight"] += count
            continue
        counts["floor"] += count
        if role == "lamp":
            counts["lights"] += count
        counts["clutter_weight"] += 3 if role in {"sofa", "table", "bed", "cabinet", "appliance"} else 2
    return counts


def _derive_scene_budgets(
    scene_program: Dict[str, Any],
    placement_intent: Dict[str, Any],
    base_budgets: Dict[str, int],
) -> Dict[str, int]:
    del placement_intent
    budgets = dict(base_budgets)
    anchor_counts = _required_slot_anchor_count(scene_program)
    optional_policy = scene_program.get("optional_addition_policy") if isinstance(scene_program.get("optional_addition_policy"), dict) else {}
    budgets["max_floor_objects"] = max(budgets["max_floor_objects"], anchor_counts["floor"])
    budgets["max_wall_objects"] = max(budgets["max_wall_objects"], anchor_counts["wall"])
    budgets["max_surface_objects"] = max(budgets["max_surface_objects"], anchor_counts["surface"])
    budgets["max_lights"] = max(budgets["max_lights"], anchor_counts["lights"])
    explicit_clutter = optional_policy.get("max_clutter_weight")
    if isinstance(explicit_clutter, (int, float)) and int(explicit_clutter) >= 0:
        budgets["max_clutter_weight"] = max(
            budgets["max_clutter_weight"],
            anchor_counts["clutter_weight"] + int(explicit_clutter),
        )
    else:
        budgets["max_clutter_weight"] = max(budgets["max_clutter_weight"], anchor_counts["clutter_weight"])

    if bool(optional_policy.get("allow_optional_additions", True)):
        if bool(optional_policy.get("prefer_wall_accents")):
            budgets["max_wall_objects"] = min(budgets["max_wall_objects"] + 1, budgets["max_props_hard"])
        if bool(optional_policy.get("prefer_surface_accents")):
            budgets["max_surface_objects"] = min(budgets["max_surface_objects"] + 1, budgets["max_props_hard"])
        optional_count = optional_policy.get("max_count")
        if isinstance(optional_count, (int, float)) and int(optional_count) >= 0:
            budgets["max_props"] = min(
                max(budgets.get("max_props", 0), anchor_counts["floor"] + int(optional_count)),
                budgets["max_props_hard"],
            )

    derived_total = (
        budgets["max_floor_objects"]
        + budgets["max_wall_objects"]
        + budgets["max_surface_objects"]
        + budgets["max_lights"]
    )
    budgets["max_props"] = min(
        max(budgets.get("max_props", 0), derived_total),
        budgets["max_props_hard"],
    )
    return budgets


def _no_placements_result(  # error output when shortliner or placer could find zero valid objects
    *,
    planner_backend: str,
    semantic_path_status: str,
    prompt_plan: Dict[str, Any],
    chosen_assets: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "ok": False,
        "error_code": "planner_no_safe_assets",
        **_planner_state(
            planner_backend=planner_backend,
            semantic_path_status=semantic_path_status,
        ),
        "prompt_plan": prompt_plan,
        "candidate_asset_ids": [str(asset.get("asset_id")) for asset in chosen_assets[:40]],
        "errors": [_error_entry("$.placements", "Planner could not select any eligible placements.")],
    }


def _planner_bootstrap(
    prompt_text: str,
    seed: int | None,
    user_prefs: Dict[str, Any],
) -> Dict[str, Any] | Dict[str, Any]:
    valid_mode, bad_mode = planner_semantics.normalize_prompt_mode(user_prefs)
    if not valid_mode:
        return _invalid_prompt_mode_result(bad_mode)

    registry = load_pack_registry()
    style_registry = load_stylekit_registry()
    if not registry.packs_by_id:
        return _planner_unavailable(
            "$.pack_registry",
            "No valid packs loaded; planner cannot select assets.",
        )
    if not style_registry.stylekits_by_id:
        return _planner_unavailable(
            "$.stylekit_registry",
            "No valid stylekits loaded; planner cannot select style.",
        )

    prompt_plan = planner_semantics.build_prompt_plan(prompt_text, user_prefs)
    all_assets = planner_assets.collect_assets([], registry)
    bootstrap_candidates = planner_assets.build_semantic_candidate_shortlist(all_assets, prompt_text, limit=40)
    prompt_plan["candidate_count"] = len(bootstrap_candidates)
    if not bootstrap_candidates:
        return _semantic_failure_result(
            prompt_plan=prompt_plan,
            candidate_assets=all_assets,
            error_code="planner_no_safe_assets",
            message="Planner could not find any quest-safe candidate assets.",
            errors=[{"path": "$.placements", "message": "Planner could not select any eligible placements."}],
        )

    return {
        "ok": True,
        "placement_mode": DEFAULT_PLACEMENT_MODE,
        "effective_seed": seed if seed is not None else _seed_from_prompt(prompt_text or "default"),
        "template_id": DEFAULT_TEMPLATE_ID if DEFAULT_TEMPLATE_ID in TEMPLATE_ALLOWLIST else TEMPLATE_ALLOWLIST[0],
        "registry": registry,
        "style_registry": style_registry,
        "prompt_plan": prompt_plan,
        "budgets": _budget_overrides(user_prefs),
        "all_assets": all_assets,
        "bootstrap_candidates": bootstrap_candidates,
        "allowed_stylekit_ids": style_registry.list_stylekits(),
        "allowed_pack_ids": sorted(registry.packs_by_id.keys()),
    }


def _semantic_intent_stage(
    *,
    prompt_text: str,
    prompt_plan: Dict[str, Any],
    user_prefs: Dict[str, Any],
    bootstrap_candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    design_brief: Dict[str, Any] = {}
    design_brief_result = request_llm_design_brief(prompt_plan=prompt_plan, user_prefs=user_prefs)
    if design_brief_result.get("ok"):
        design_brief = dict(design_brief_result.get("design_brief") or {})

    intent_result = request_llm_intent(prompt_plan=prompt_plan, user_prefs=user_prefs, design_brief=design_brief)
    if not intent_result.get("ok"):
        return _semantic_failure_result(
            prompt_plan=prompt_plan,
            candidate_assets=bootstrap_candidates,
            error_code=str(intent_result.get("error_code") or "llm_unavailable"),
            message=str(intent_result.get("message") or "LLM planner failed."),
        )

    validated_intent = planner_semantics.validate_semantic_intent(
        intent_result.get("intent_payload", {}),
        prompt_text=prompt_text,
    )
    if not validated_intent.get("ok"):
        return _semantic_failure_result(
            prompt_plan=prompt_plan,
            candidate_assets=bootstrap_candidates,
            error_code=str(validated_intent.get("error_code") or "semantic_invalid_intent"),
            message=str(validated_intent.get("message") or "Semantic planner returned an invalid intent."),
            errors=list(validated_intent.get("errors") or []),
            extra_fields={
                "invalid_intent_payload": intent_result.get("intent_payload"),
            },
        )

    scene_program = planner_semantics.complete_scene_program(validated_intent["scene_program"], prompt_text)
    return {
        "ok": True,
        "design_brief": design_brief,
        "scene_program": scene_program,
        "intent_spec": planner_semantics.scene_program_to_intent_spec(scene_program),
        "placement_intent": planner_semantics.scene_program_to_placement_intent(
            scene_program,
            validated_intent["placement_intent"],
        ),
    }


def _semantic_selection_stage(
    *,
    prompt_text: str,
    user_prefs: Dict[str, Any],
    prompt_plan: Dict[str, Any],
    all_assets: List[Dict[str, Any]],
    allowed_stylekit_ids: List[str],
    allowed_pack_ids: List[str],
    style_registry: Any,
    registry: Any,
    budgets: Dict[str, int],
    bootstrap_candidates: List[Dict[str, Any]],
    design_brief: Dict[str, Any],
    scene_program: Dict[str, Any],
    intent_spec: Dict[str, Any],
    placement_intent: Dict[str, Any],
) -> Dict[str, Any]:
    budgets = _derive_scene_budgets(scene_program, placement_intent, budgets)
    semantic_candidates = planner_assets.build_semantic_candidate_shortlist(
        all_assets,
        prompt_text,
        limit=40,
        intent_spec=intent_spec,
        scene_program=scene_program,
    )
    prompt_plan["candidate_count"] = len(semantic_candidates)
    if not semantic_candidates:
        return _semantic_failure_result(
            prompt_plan=prompt_plan,
            candidate_assets=all_assets,
            error_code="planner_no_safe_assets",
            message="Semantic intent did not map to any quest-safe candidate assets.",
            errors=[{"path": "$.placements", "message": "Planner could not select any eligible placements."}],
        )

    surface_material_candidates = planner_semantics.build_surface_material_candidates(scene_program)
    selection_result = request_llm_selection(
        prompt_plan=prompt_plan,
        candidate_assets=semantic_candidates,
        allowed_stylekit_ids=allowed_stylekit_ids,
        allowed_pack_ids=allowed_pack_ids,
        default_budgets=budgets,
        intent_spec=intent_spec,
        scene_program=scene_program,
        placement_intent=placement_intent,
        user_prefs=user_prefs,
        design_brief=design_brief,
        stylekit_candidates=planner_semantics.build_stylekit_candidates(style_registry),
        pack_candidates=planner_semantics.build_pack_candidates(registry),
        surface_material_candidates=surface_material_candidates,
    )
    if not selection_result.get("ok"):
        return _semantic_failure_result(
            prompt_plan=prompt_plan,
            candidate_assets=semantic_candidates,
            error_code=str(selection_result.get("error_code") or "llm_unavailable"),
            message=str(selection_result.get("message") or "LLM planner failed."),
        )

    validated_plan = planner_semantics.validate_semantic_plan(
        {"design_brief": design_brief, "intent": scene_program, "selection": selection_result.get("selection", {})},
        all_assets=all_assets,
        allowed_stylekit_ids=allowed_stylekit_ids,
        allowed_pack_ids=allowed_pack_ids,
        default_budgets=budgets,
        prompt_text=prompt_text,
        placement_intent=placement_intent,
        surface_material_candidates=surface_material_candidates,
    )
    if not validated_plan.get("ok"):
        return _semantic_failure_result(
            prompt_plan=prompt_plan,
            candidate_assets=semantic_candidates,
            error_code=str(validated_plan.get("error_code") or "semantic_invalid_selection"),
            message=str(validated_plan.get("message") or "Semantic planner returned an invalid selection."),
            errors=validated_plan.get("errors"),
            extra_fields={
                key: validated_plan.get(key)
                for key in (
                    "scene_program",
                    "intent_spec",
                    "placement_intent",
                    "missing_required_slots",
                    "covered_required_slots",
                    "softened_required_slots",
                    "slot_diagnostics",
                    "selected_asset_ids",
                    "slot_asset_map",
                    "fallback_asset_ids_by_slot",
                )
                if validated_plan.get(key) is not None
            },
        )

    return {
        "ok": True,
        "budgets": validated_plan["budgets"],
        "scene_program": validated_plan["scene_program"],
        "semantic_candidates": semantic_candidates,
        "selected_prompt": validated_plan["selected_prompt"],
        "stylekit_id": validated_plan["stylekit_id"],
        "pack_ids": validated_plan["pack_ids"],
        "chosen_assets": validated_plan["assets"],
        "semantic_selection": validated_plan["semantic_selection"],
        "decor_plan": validated_plan.get("decor_plan") or {},
        "surface_material_selection": validated_plan.get("surface_material_selection") or {},
        "surface_material_candidates": surface_material_candidates,
        "optional_additions": list((validated_plan.get("semantic_selection") or {}).get("optional_additions") or []),
        "covered_required_slots": list(validated_plan.get("covered_required_slots") or []),
        "missing_required_slots": list(validated_plan.get("missing_required_slots") or []),
        "slot_diagnostics": list(validated_plan.get("slot_diagnostics") or []),
    }


def _placement_stage(
    *,
    selected_prompt: str,
    effective_seed: int,
    budgets: Dict[str, int],
    chosen_assets: List[Dict[str, Any]],
    optional_additions: List[Dict[str, Any]],
    all_assets: List[Dict[str, Any]],
    intent_spec: Dict[str, Any],
    placement_intent: Dict[str, Any],
    scene_program: Dict[str, Any],
) -> Dict[str, Any]:
    placements, placement_plan = planner_assets.build_layout_from_selected_assets(
        chosen_assets,
        prompt_text=selected_prompt,
        seed=effective_seed,
        max_props=budgets["max_props"],
        budgets=budgets,
        intent_spec=intent_spec,
        placement_intent=placement_intent,
        candidate_assets=all_assets,
        scene_program=scene_program,
    )
    placements.extend(
        planner_assets.build_optional_raw_placements(
            optional_additions,
            candidate_assets=all_assets,
            scene_program=scene_program,
        )
    )
    if placements:
        placement_plan = dict(placement_plan)
        placement_plan["placed_count"] = len(placements)
        placement_plan["optional_addition_count"] = len(optional_additions)
    return {"placements": placements, "placement_plan": placement_plan}


def _scene_context(scene_program: Dict[str, Any], scene_context_assets: List[Dict[str, Any]]) -> Dict[str, Any]:
    context = {
        "archetype": scene_program.get("archetype"),
        "scene_type": scene_program.get("scene_type"),
        "source_prompt": scene_program.get("source_prompt"),
        "design_brief": dict(scene_program.get("design_brief") or {}),
        "concept_label": scene_program.get("concept_label"),
        "creative_tags": list(scene_program.get("creative_tags") or []),
        "mood_tags": list(scene_program.get("mood_tags") or []),
        "style_descriptors": list(scene_program.get("style_descriptors") or []),
        "scene_features": list(scene_program.get("scene_features") or []),
        "execution_archetype": scene_program.get("execution_archetype"),
        "negative_constraints": list(scene_program.get("negative_constraints") or []),
        "semantic_slots": [dict(slot) for slot in scene_program.get("semantic_slots") or [] if isinstance(slot, dict)],
        "grounded_slots": [dict(slot) for slot in scene_program.get("grounded_slots") or [] if isinstance(slot, dict)],
        "focal_object_role": scene_program.get("focal_object_role"),
        "focal_wall": scene_program.get("focal_wall"),
        **build_decor_capabilities(scene_context_assets),
    }
    context["decor_asset_ids_by_kind"] = build_decor_asset_ids_by_kind(scene_context_assets, scene_context=context)
    return context


def _planner_result(
    *,
    validation: Dict[str, Any],
    worldspec: Dict[str, Any],
    prompt_plan: Dict[str, Any],
    planner_backend: str,
    semantic_path_status: str,
    placement_plan: Dict[str, Any],
    user_prefs: Dict[str, Any],
    selected_assets: List[Dict[str, Any]],
    chosen_assets: List[Dict[str, Any]],
    scene_program: Dict[str, Any],
    intent_spec: Dict[str, Any],
    placement_intent: Dict[str, Any],
    semantic_selection: Dict[str, Any],
) -> Dict[str, Any]:
    public_scene_program = planner_semantics.public_scene_program(scene_program)
    public_intent_spec = planner_semantics.public_intent_spec(intent_spec)
    result: Dict[str, Any] = {
        "ok": validation["ok"],
        "worldspec": worldspec,
        "errors": validation["errors"],
        "prompt_plan": prompt_plan,
        **_planner_state(
            planner_backend=planner_backend,
            semantic_path_status=semantic_path_status,
        ),
        "candidate_asset_ids": [str(asset.get("asset_id")) for asset in selected_assets[:40]],
        "scene_program": public_scene_program,
        "intent_spec": public_intent_spec,
        "placement_intent": placement_intent,
        "placement_plan": placement_plan,
        "covered_required_slots": list(semantic_selection.get("covered_required_slots") or []),
        "missing_required_slots": list(semantic_selection.get("missing_required_slots") or []),
        "slot_diagnostics": list(semantic_selection.get("slot_diagnostics") or []),
    }
    if _normalize_bool(user_prefs.get("include_semantic_receipts"), default=True):
        result["semantic_receipts"] = planner_semantics.semantic_receipts(
            selected_assets=selected_assets,
            candidate_assets=chosen_assets,
            intent_spec=public_intent_spec,
            scene_program=public_scene_program,
            placement_intent=placement_intent,
            semantic_selection=semantic_selection,
        )
    return result


def plan_worldspec(  # top-level orchestrator: prompt -> intent -> selection -> placement -> worldspec
    prompt_text: str,
    seed: int | None = None,
    user_prefs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    prompt_text = (prompt_text or "").strip()
    user_prefs = user_prefs or {}
    bootstrap = _planner_bootstrap(prompt_text, seed, user_prefs)
    if not bootstrap.get("ok"):
        return bootstrap

    prompt_plan = bootstrap["prompt_plan"]
    budgets = bootstrap["budgets"]
    all_assets = bootstrap["all_assets"]
    bootstrap_candidates = bootstrap["bootstrap_candidates"]

    planner_backend = "llm"
    semantic_path_status = "ok"
    intent_stage = _semantic_intent_stage(
        prompt_text=prompt_text,
        prompt_plan=prompt_plan,
        user_prefs=user_prefs,
        bootstrap_candidates=bootstrap_candidates,
    )
    if not intent_stage.get("ok"):
        return intent_stage

    selection_stage = _semantic_selection_stage(
        prompt_text=prompt_text,
        user_prefs=user_prefs,
        prompt_plan=prompt_plan,
        all_assets=all_assets,
        allowed_stylekit_ids=bootstrap["allowed_stylekit_ids"],
        allowed_pack_ids=bootstrap["allowed_pack_ids"],
        style_registry=bootstrap["style_registry"],
        registry=bootstrap["registry"],
        budgets=budgets,
        bootstrap_candidates=bootstrap_candidates,
        design_brief=intent_stage["design_brief"],
        scene_program=intent_stage["scene_program"],
        intent_spec=intent_stage["intent_spec"],
        placement_intent=intent_stage["placement_intent"],
    )
    if not selection_stage.get("ok"):
        return selection_stage

    scene_program = selection_stage["scene_program"]
    intent_spec = intent_stage["intent_spec"]
    placement_intent = intent_stage["placement_intent"]
    chosen_assets = selection_stage["chosen_assets"]
    semantic_selection = selection_stage["semantic_selection"]
    selected_prompt = selection_stage["selected_prompt"]
    prompt_plan["selected_prompt"] = selected_prompt
    prompt_plan["strategy"] = "semantic_primary"

    placement_stage = _placement_stage(
        selected_prompt=selected_prompt,
        effective_seed=bootstrap["effective_seed"],
        budgets=selection_stage["budgets"],
        chosen_assets=chosen_assets,
        optional_additions=selection_stage["optional_additions"],
        all_assets=all_assets,
        intent_spec=intent_spec,
        placement_intent=placement_intent,
        scene_program=scene_program,
    )
    placements = placement_stage["placements"]
    placement_plan = placement_stage["placement_plan"]
    if not placements:
        return _no_placements_result(
            planner_backend=planner_backend,
            semantic_path_status=semantic_path_status,
            prompt_plan=prompt_plan,
            chosen_assets=chosen_assets,
        )

    colors = planner_semantics.apply_stylekit_colors(selection_stage["stylekit_id"], bootstrap["style_registry"])
    colors = planner_semantics.apply_surface_material_colors(
        colors,
        selection_stage["surface_material_selection"],
        selection_stage["surface_material_candidates"],
    )
    planner_policy = {
        "semantic_path_status": semantic_path_status,
        "placement_mode": bootstrap["placement_mode"],
    }
    scene_context_assets = selection_stage["semantic_candidates"] or chosen_assets or all_assets
    worldspec = _build_worldspec(
        template_id=bootstrap["template_id"],
        effective_seed=bootstrap["effective_seed"],
        stylekit_id=selection_stage["stylekit_id"],
        pack_ids=selection_stage["pack_ids"],
        placements=placements,
        budgets=selection_stage["budgets"],
        colors=colors,
        planner_policy=planner_policy,
        placement_intent=placement_intent,
        placement_plan=placement_plan,
        layout_program=(placement_plan or {}).get("layout_program"),
        decor_plan=selection_stage["decor_plan"],
        surface_material_selection=selection_stage["surface_material_selection"],
        optional_additions=selection_stage["optional_additions"],
        scene_context=_scene_context(scene_program, scene_context_assets),
    )
    validation = validate_worldspec(worldspec)
    selected_asset_records = planner_assets.candidate_assets_by_ids(
        all_assets,
        [placement["asset_id"] for placement in placements],
    )
    return _planner_result(
        validation=validation,
        worldspec=worldspec,
        prompt_plan=prompt_plan,
        planner_backend=planner_backend,
        semantic_path_status=semantic_path_status,
        placement_plan=placement_plan,
        user_prefs=user_prefs,
        selected_assets=selected_asset_records,
        chosen_assets=chosen_assets,
        scene_program=scene_program,
        intent_spec=intent_spec,
        placement_intent=placement_intent,
        semantic_selection=semantic_selection,
    )
