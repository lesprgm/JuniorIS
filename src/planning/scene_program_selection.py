from __future__ import annotations

from typing import Any, Dict, List

from src.planning.scene_program_normalization import scene_program_to_intent_spec, validate_semantic_intent
from src.planning.scene_program_selection_assets import (
    _expanded_assets_from_selection,
    _normalize_surface_material_selection,
)
from src.planning.scene_program_selection_payload import (
    _ground_selection_slots,
    _normalize_pack_ids,
    _normalize_selection_budgets,
    _normalize_selection_extras,
    _normalized_group_assignments,
    _selection_coverage,
    _selection_error,
    _semantic_selection_payload,
)

def validate_semantic_plan(  # validates an LLM selection response: checks stylekit, pack_ids, asset coverage, and budgets
    llm_plan: Dict[str, Any],
    *,
    all_assets: List[Dict[str, Any]],
    allowed_stylekit_ids: List[str],
    allowed_pack_ids: List[str],
    default_budgets: Dict[str, int],
    prompt_text: str,
    placement_intent: Dict[str, Any] | None = None,
    surface_material_candidates: Dict[str, List[Dict[str, Any]]] | None = None,
) -> Dict[str, Any]:
    if not isinstance(llm_plan, dict):
        return _selection_error(
            message="Semantic planner response was not an object.",
            errors=[{"path": "$.llm", "message": "Semantic planner response was not an object."}],
        )

    raw_intent = llm_plan.get("intent")
    validated_intent = validate_semantic_intent(
        {
            "intent": raw_intent,
            "placement_intent": placement_intent,
            "design_brief": llm_plan.get("design_brief"),
        },
        prompt_text=prompt_text,
    )
    if not validated_intent.get("ok"):
        return _selection_error(
            message="Semantic planner response included an invalid intent.",
            scene_program=validated_intent.get("scene_program"),
            intent_spec=validated_intent.get("intent_spec"),
            placement_intent=validated_intent.get("placement_intent"),
            errors=list(validated_intent.get("errors") or []),
        )

    base_scene_program = validated_intent["scene_program"]
    intent_spec = validated_intent["intent_spec"]
    effective_placement_intent = validated_intent["placement_intent"]
    selection = llm_plan.get("selection") if isinstance(llm_plan.get("selection"), dict) else llm_plan
    if not isinstance(selection, dict):
        return _selection_error(
            message="Semantic planner response is missing a selection object.",
            scene_program=base_scene_program,
            intent_spec=intent_spec,
            placement_intent=effective_placement_intent,
            errors=[{"path": "$.llm.selection", "message": "Semantic planner response is missing a selection object."}],
        )

    stylekit_id = str(selection.get("stylekit_id") or "").strip()
    if not stylekit_id or stylekit_id not in allowed_stylekit_ids:
        return _selection_error(
            message="Semantic planner must choose a valid approved stylekit.",
            scene_program=base_scene_program,
            intent_spec=intent_spec,
            placement_intent=effective_placement_intent,
            errors=[{
                "path": "$.llm.selection.stylekit_id",
                "message": f"Semantic planner must choose one approved stylekit id from: {', '.join(allowed_stylekit_ids)}.",
            }],
        )

    pack_ids = _normalize_pack_ids(selection.get("pack_ids"), allowed_pack_ids)
    scene_program, slot_asset_map, normalized_fallback_asset_ids_by_slot, softened_rescue_slot_ids = _ground_selection_slots(
        selection,
        scene_program=base_scene_program,
        all_assets=all_assets,
    )
    intent_spec = scene_program_to_intent_spec(scene_program)
    group_assignments, group_assignment_errors = _normalized_group_assignments(
        selection,
        scene_program=scene_program,
        slot_asset_map=slot_asset_map,
        all_assets=all_assets,
    )

    chosen_assets, selection_coverage_errors = _expanded_assets_from_selection(
        scene_program=scene_program,
        slot_asset_map=slot_asset_map,
        group_assignments=group_assignments,
        all_assets=all_assets,
        prompt_text=prompt_text,
        softened_rescue_slot_ids=softened_rescue_slot_ids,
    )
    chosen_asset_ids = [str(asset.get("asset_id")) for asset in chosen_assets]
    budgets = _normalize_selection_budgets(selection, default_budgets)
    extras = _normalize_selection_extras(
        selection,
        intent_spec.get("confidence", 0.0),
        scene_program=scene_program,
    )
    selected_prompt = str(selection.get("selected_prompt") or prompt_text).strip() or prompt_text.strip()
    raw_selection_attempted = bool(
        (isinstance(selection.get("slot_asset_map"), dict) and selection.get("slot_asset_map"))
        or (isinstance(selection.get("group_assignments"), list) and selection.get("group_assignments"))
    )
    if not (slot_asset_map or group_assignments):
        if raw_selection_attempted:
            return _selection_error(
                message="Semantic planner selected no valid approved assets.",
                scene_program=scene_program,
                intent_spec=intent_spec,
                placement_intent=effective_placement_intent,
                error_code="semantic_unknown_assets",
                errors=[{"path": "$.llm.selection", "message": "Semantic planner selected no valid approved assets."}],
                extra={"unknown_asset_ids": []},
            )
        return _selection_error(
            message="Semantic planner did not assign assets for the scene roles.",
            scene_program=scene_program,
            intent_spec=intent_spec,
            placement_intent=effective_placement_intent,
            errors=[{"path": "$.llm.selection", "message": "Semantic planner did not assign assets for the scene roles."}],
        )

    if not chosen_assets:
        return _selection_error(
            message="Semantic planner selected no valid approved assets.",
            scene_program=scene_program,
            intent_spec=intent_spec,
            placement_intent=effective_placement_intent,
            error_code="semantic_unknown_assets",
            errors=[{"path": "$.llm.selection", "message": "Semantic planner selected no valid approved assets."}],
            extra={"unknown_asset_ids": []},
        )

    coverage = _selection_coverage(
        scene_program=scene_program,
        slot_asset_map=slot_asset_map,
        fallback_asset_ids_by_slot=normalized_fallback_asset_ids_by_slot,
        selection_coverage_errors=selection_coverage_errors,
        softened_rescue_slot_ids=softened_rescue_slot_ids,
        prompt_text=prompt_text,
    )
    if coverage["selection_coverage_errors"] or coverage["missing_required_slots"]:
        missing_tokens = sorted(set(coverage["selection_coverage_errors"] + coverage["missing_required_slots"]))
        return _selection_error(
            message="Semantic planner did not satisfy the required semantic slots.",
            scene_program=scene_program,
            intent_spec=intent_spec,
            placement_intent=effective_placement_intent,
            error_code="semantic_missing_required_slots",
            errors=[
                {"path": "$.llm.selection", "message": "Semantic planner did not satisfy the required semantic slots."},
                {"path": "$.llm.selection.slot_asset_map", "message": f"Missing or underfilled required slots: {', '.join(missing_tokens)}"},
            ] + list(group_assignment_errors or []),
            extra={
                "covered_required_slots": coverage["covered_required_slots"],
                "missing_required_slots": missing_tokens,
                "softened_required_slots": coverage.get("softened_required_slots") or [],
                "slot_diagnostics": coverage["slot_diagnostics"],
                "selected_asset_ids": chosen_asset_ids,
                "slot_asset_map": slot_asset_map,
                "fallback_asset_ids_by_slot": normalized_fallback_asset_ids_by_slot,
            },
        )

    if group_assignment_errors:
        return _selection_error(
            message="Semantic planner returned invalid group assignments.",
            scene_program=scene_program,
            intent_spec=intent_spec,
            placement_intent=effective_placement_intent,
            errors=group_assignment_errors,
        )

    surface_material_selection, surface_material_errors = _normalize_surface_material_selection(
        selection.get("surface_material_selection"),
        surface_material_candidates,
    )
    if surface_material_errors:
        return _selection_error(
            message="Semantic planner returned an invalid shell material selection.",
            scene_program=scene_program,
            intent_spec=intent_spec,
            placement_intent=effective_placement_intent,
            errors=surface_material_errors,
        )

    selection_payload = dict(selection)
    if normalized_fallback_asset_ids_by_slot:
        selection_payload["fallback_asset_ids_by_slot"] = normalized_fallback_asset_ids_by_slot
    semantic_selection = _semantic_selection_payload(
        selection=selection_payload,
        selected_prompt=selected_prompt,
        stylekit_id=stylekit_id,
        pack_ids=pack_ids,
        chosen_asset_ids=chosen_asset_ids,
        slot_asset_map=slot_asset_map,
        group_assignments=group_assignments,
        scene_program=scene_program,
        all_assets=all_assets,
        alternatives=extras["alternatives"],
        rationale=extras["rationale"],
        confidence=extras["confidence"],
        surface_material_selection=surface_material_selection,
        coverage=coverage,
    )

    return {
        "ok": True,
        "scene_program": scene_program,
        "selected_prompt": selected_prompt,
        "stylekit_id": stylekit_id,
        "pack_ids": pack_ids,
        "assets": chosen_assets,
        "asset_ids": chosen_asset_ids,
        "budgets": budgets,
        "has_unknown_asset_ids": False,
        "unknown_asset_ids": [],
        "intent_spec": intent_spec,
        "placement_intent": effective_placement_intent,
        "covered_required_slots": coverage["covered_required_slots"],
        "missing_required_slots": coverage["missing_required_slots"],
        "softened_required_slots": coverage.get("softened_required_slots") or [],
        "slot_diagnostics": coverage["slot_diagnostics"],
        "semantic_selection": semantic_selection,
        "decor_plan": semantic_selection["decor_plan"],
        "surface_material_selection": surface_material_selection,
    }
