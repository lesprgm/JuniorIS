from __future__ import annotations

import time
from types import SimpleNamespace
from urllib import error as urllib_error
from urllib import request as urllib_request
from typing import Any, Dict, List

from src.llm import transport
from src.llm.planner_support import (
    compact_candidate_asset_payload,
    compact_pack_payload,
    compact_stylekit_payload,
        inline_plan_payload,
    invalid_content_error,
    provider_circuit_key,
    resolve_provider,
    selection_payload,
    stage_user_prefs,
    timed_result,
)


_CIRCUIT_STATE = transport._CIRCUIT_STATE
urllib = SimpleNamespace(request=urllib_request, error=urllib_error)

_SUPPORTED_ARCHETYPE_TEXT = "study, bedroom, lounge, workshop, kitchen, bathroom, generic_room"
_SUPPORTED_RUNTIME_ROLE_TEXT = "appliance, bed, bench, cabinet, chair, decor, lamp, plant, sign, sofa, table, tool"
_DESIGN_BRIEF_CONTRACT = (
    "`design_brief` must include concept_statement:string, palette_strategy:string, material_hierarchy:string[], "
    "lighting_layers:string[], signature_moment:string, visual_weight_distribution:string, texture_profile:string, "
    "luxury_signal_level:string, restraint_rules:string[], anti_patterns:string[]. "
)
_INTENT_CONTRACT = (
    "`intent` must include: "
    "scene_type:string, concept_label:string, creative_summary:string, intended_use:string, focal_object_role:string, "
    "focal_wall:'front|back|left|right|none', circulation_preference:'clear_center|clear_entry|balanced|layered', "
    "empty_space_preference:'open_center|balanced|layered', creative_tags:string[], mood_tags:string[], "
    "style_descriptors:string[], execution_archetype:string, "
    "semantic_slots:[{slot_id:string,concept:string,runtime_role_hint:string,priority:'must|should|optional',necessity:'core|support|enrichment',source:'explicit_prompt|inferred_function|style_enrichment|deterministic_completion',count:number,capabilities:string[],zone_preference:string,rationale:string}], "
    "primary_anchor_object:{slot_id?:string,role:string,rationale:string}, secondary_support_objects:{role:string,count:number,rationale:string}[], "
    "groups:[{group_id:string,group_type:'dining_set|lounge_cluster|reading_corner|bedside_cluster|workstation',anchor_role:string,member_role:string,member_count:number,layout_pattern:'paired_long_sides|ring|arc|front_facing_cluster|beside_anchor',facing_rule:'toward_anchor|toward_focal_object|parallel|none',symmetry:'symmetric|asymmetric|balanced',zone_preference:'center|edge|corner|front|back|left|right',importance:'primary|secondary|background'}], "
    "relation_graph:{source_role:string,target_role:string,relation:'near|face_to|align|far|edge|middle|support_on|against_wall|centered_on_wall|symmetry_with|avoid',relation_type:'proximity|orientation|support|wall_alignment|symmetry|avoidance|room_position',constraint_strength:'required|preferred',target_surface_type?:'floor|wall|surface|ceiling|tabletop|shelf'}[], "
    "negative_constraints:string[], optional_addition_policy:{allow_optional_additions:boolean,avoid_center_clutter:boolean,prefer_wall_accents:boolean,prefer_surface_accents:boolean,max_count:number,max_clutter_weight:number}, "
    "surface_material_intent:{wall_tags:string[],floor_tags:string[],ceiling_tags:string[],accent_tags:string[]}, "
    "density_target:'sparse|normal|dense|cluttered', symmetry_preference:'symmetric|asymmetric|balanced', "
    "walkway_preservation_intent:{keep_central_path_clear:boolean,keep_entry_clear:boolean,notes:string}, "
    "scene_features:string[], style_tags:string[], color_tags:string[], "
    "style_cues:{style_tags:string[],color_tags:string[],lighting_tags:string[],mood_tags:string[]}, confidence:number. "
)
_SELECTION_CONTRACT = (
    "Output a selection object shaped like {selected_prompt:string, stylekit_id:string, pack_ids:string[], "
    "group_assignments:[{group_id:string,slot_asset_map:object}], slot_asset_map:object, "
    "asset_ids:string[], rejected_candidate_ids:string[], rejected_candidates_by_slot:object, "
    "fallback_asset_ids_by_slot:object, "
    "budgets:{max_props:int,max_props_hard:int,max_floor_objects:int,max_wall_objects:int,max_surface_objects:int,max_texture_tier:int,max_lights:int,max_clutter_weight:int}, "
    "optional_additions:[{asset_id:string,anchor:'floor|wall|surface|ceiling',placement_mode:string,placement_hint:'wall_centered|wall_left|wall_right|wall_above_anchor|corner_left|corner_right|tabletop_center|tabletop_edge|ceiling_center|floor_edge',usage:'support|accent|clutter'}], "
    "decor_plan:{entries:[{asset_id?:string,kind:string,anchor:string,zone_id:string,count:int,placement_hint?:string}], rationale:string[]}, "
    "surface_material_selection:{wall:string,floor:string,ceiling:string,accent?:string}, alternatives:object, rationale:string[], confidence:number}. "
)


def _unwrap_plan_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload.get("plan"), dict):
        return payload["plan"]
    return payload


def _inline_selection_payload(inline_plan: Dict[str, Any]) -> Dict[str, Any]:
    selection = inline_plan.get("selection", {})
    return selection if isinstance(selection, dict) else {}



# Keep behavior deterministic so planner/runtime contracts stay stable.
def request_llm_design_brief(
    *,
    prompt_plan: Dict[str, Any],
    user_prefs: Dict[str, Any],
) -> Dict[str, Any]:
    start_time = time.monotonic()
    inline_plan = inline_plan_payload(user_prefs)
    if inline_plan is not None:
        return timed_result(
            start_time,
            {
                "ok": True,
                "backend": "inline_override",
                "design_brief": dict(inline_plan.get("design_brief") or {}),
            },
        )

    stage_prefs = stage_user_prefs(user_prefs, "design_brief")
    adapter, settings, config_error = resolve_provider(stage_prefs)
    if config_error is not None:
        return timed_result(start_time, config_error)

    user_payload = {
        "selected_prompt": prompt_plan.get("selected_prompt"),
        "input_prompt": prompt_plan.get("input_prompt") or prompt_plan.get("selected_prompt"),
    }
    design_brief_examples = (
        "Few-shot brief priors: "
        "Example A - prompt:'bedroom' -> concept_statement:'calm layered sleep retreat', palette_strategy:'warm_neutral', material_hierarchy:['soft_textile','oak','matte_plaster'], lighting_layers:['ambient','bedside_task','accent'], signature_moment:'bed wall plus one composed bedside vignette', visual_weight_distribution:'focal-wall weighted', texture_profile:'mixed', luxury_signal_level:'premium', restraint_rules:['keep storage quiet','one accent zone only'], anti_patterns:['dorm room clutter','clinical white washout']. "
        "Example B - prompt:'museum room' -> concept_statement:'quiet gallery chamber for a single clear display story', palette_strategy:'airy_low_contrast', material_hierarchy:['limewash','stone','dark metal accent'], lighting_layers:['ambient','accent'], signature_moment:'one frontal display view with controlled negative space', visual_weight_distribution:'focal-wall weighted', texture_profile:'minimal', luxury_signal_level:'editorial', restraint_rules:['let one display dominate','keep center calm'], anti_patterns:['casual residential props','overdecorated souvenir-shop look']. "
        "Example C - prompt:'space station cabin' -> concept_statement:'compact technical refuge with disciplined warmth', palette_strategy:'dark_contrast', material_hierarchy:['matte composite','brushed metal','soft fabric accent'], lighting_layers:['ambient','task','accent'], signature_moment:'compact berth-work surface composition with one precise light pool', visual_weight_distribution:'perimeter-weighted', texture_profile:'mixed', luxury_signal_level:'restrained', restraint_rules:['keep circulation efficient','limit loose accents'], anti_patterns:['generic suburban decor','theme-park sci-fi clutter']. "
    )
    system_prompt = (
        "You are a strict design-brief planner for a VR room compiler. Return JSON only. "
        "Output an object with a single key `design_brief`. "
        f"{_DESIGN_BRIEF_CONTRACT}"
        "Act like an interior design director first: define the room's emotional target, material hierarchy, lighting layers, palette strategy, one memorable vignette, and what the room must avoid becoming. "
        "Keep the brief general enough to support unusual indoor concepts like museum gallery, station cabin, archive, greenhouse, meditation chamber, or luxury suite. "
        "Prefer design language that is specific enough to guide taste but abstract enough to generalize across many indoor prompts. "
        "Do not emit coordinates, runtime roles, asset ids, counts, or geometry. "
        f"{design_brief_examples}"
        "Use short structured language, not prose paragraphs."
    )
    result = adapter.request_json(
        settings=settings,
        system_prompt=system_prompt,
        user_payload=user_payload,
        circuit_key=provider_circuit_key(settings["provider"], "design_brief"),
    )
    if not result.get("ok"):
        return timed_result(start_time, result)
    payload = _unwrap_plan_payload(result["payload"])
    design_brief = payload.get("design_brief") if isinstance(payload.get("design_brief"), dict) else payload
    if not isinstance(design_brief, dict):
        return timed_result(start_time, invalid_content_error(adapter, "content does not include a valid design brief object."))
    return timed_result(start_time, {"ok": True, "backend": settings["provider"], "design_brief": design_brief})



# Keep behavior deterministic so planner/runtime contracts stay stable.
def request_llm_intent(  # stage 1: asks the LLM to interpret the prompt into a structured intent
    *,
    prompt_plan: Dict[str, Any],
    user_prefs: Dict[str, Any],
    design_brief: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    start_time = time.monotonic()
    inline_plan = inline_plan_payload(user_prefs)
    if inline_plan is not None:
        return timed_result(
            start_time,
            {
                "ok": True,
                "backend": "inline_override",
                "intent_payload": {
                    "intent": inline_plan.get("intent", {}),
                    "placement_intent": inline_plan.get("placement_intent", {}),
                },
            },
        )

    stage_prefs = stage_user_prefs(user_prefs, "intent")
    adapter, settings, config_error = resolve_provider(stage_prefs)
    if config_error is not None:
        return timed_result(start_time, config_error)

    user_payload = {
        "selected_prompt": prompt_plan.get("selected_prompt"),
        "input_prompt": prompt_plan.get("input_prompt") or prompt_plan.get("selected_prompt"),
    }
    intent_generalization_guidance = (
        "Generalize by composition pattern, not by room label. Treat unusual indoor prompts like a space-station cabin, museum gallery, observatory, archive, greenhouse, meditation chamber, or luxury suite as variants of a small set of reusable composition patterns: sleeping zone, reading corner, workstation, display/focal wall, lounge cluster, dining set, storage edge, and support surface. "
        "If the prompt names a novel setting, preserve its concept_label, mood, and style language, but map execution to the nearest supported execution_archetype and reuse the most believable indoor composition pattern. "
        "Do not collapse every vague prompt to the bare minimum. For prompts like 'bedroom', infer a primary sleep anchor plus coherent support objects and restrained decor so the room reads as finished rather than empty. "
        "When the room concept is unusual, solve it with supported semantic slot concepts instead of inventing brittle one-off object labels. Prefer normalized concepts like bed, nightstand, dresser, wardrobe, desk, display_surface, wall_art, floor_lamp, table, chair, cabinet, and appliance over bespoke names like interpretive_signage, ceremonial dais, kitchen_worktable, or accounting desk unless the bespoke wording still cleanly maps back to one of those supported concepts. "
        "primary_anchor_object.role, secondary_support_objects.role, groups.anchor_role, groups.member_role, and relation_graph source_role/target_role must use supported runtime roles only, not freeform concept names. If the obvious anchor is described in a custom way, derive the anchor role from the strongest must-slot runtime role. "
        "Set semantic slot `necessity` explicitly: core means the room's primary activity breaks without it; support means useful functional support; enrichment means style/decor atmosphere. Set `source` explicitly: explicit_prompt only when the user directly requested that slot; inferred_function for commonsense support; style_enrichment for visual atmosphere. "
        "Be conservative about blocking slots. `priority` is ranking guidance, not validation truth. Use necessity='core' with source='explicit_prompt' only for the user's actual core ask or the primary anchor. Use support/enrichment for book storage, side surfaces, extra storage, wall art, plants, rugs, and accent lighting unless the user explicitly asks for them as the main requirement. "
    )
    intent_few_shots = (
        "Few-shot composition priors: "
        "Example A - prompt:'bedroom' -> semantic_slots should usually include sleep_anchor_1 concept='bed' priority='must' necessity='core' source='explicit_prompt', bedside_surface_1 concept='nightstand' priority='should' necessity='support' source='inferred_function', sleep_storage_1 concept='dresser' priority='should' necessity='support' source='inferred_function'; groups should make the bedside relationship legible; negative constraints should reject office/workshop clutter. "
        "Example B - prompt:'space station crew quarters' -> keep concept_label expressive and futuristic, map execution_archetype to bedroom or study, emit compact semantic_slots like bed + desk + task_lamp, and keep support hierarchy efficient instead of adding many props. "
        "Example C - prompt:'museum room' -> map execution_archetype to lounge or generic_room, emit focal display support slots such as display_surface plus wall_art and optional accent lighting, add strong negative constraints against casual residential clutter, and use centered_on_wall/avoid relations around the focal display. Do not invent unsupported slot concepts when display_surface, wall_art, table, decor, or lamp already express the need. "
        "Example D - prompt:'reading room' -> emit a reading-focused slot set with a core reading seat, support side surface, support/task lighting, and enrichment decor only when useful; book storage is support unless the user says library/books/bookshelves/archive. "
        "Example E - prompt:'dining room with 4 chairs and a table' -> emit one anchor dining table slot plus four repeated chair member slots or an equivalent repeated-chair semantic plan, set a dining_set group with member_count=4, and keep seating subtype-coherent instead of mixing dining and lounge chairs. "
        "Example F - prompt:'lecture room with front-facing seating' -> preserve the audience-facing concept, represent the seating bank as one repeated-member group using layout_pattern='front_facing_cluster', keep a clear focal display/support slot at the front, and avoid collapsing the room into random lounge scatter. "
        "Example G - prompt:'kitchen' -> infer a complete kitchen program using supported concepts such as table or prep surface, cabinet storage, and appliances. Keep primary_anchor_object.role within supported runtime roles such as table, cabinet, or appliance rather than custom names like kitchen_worktable. Counter stools and accent lighting should usually be should/optional, not must, unless the prompt explicitly asks for them. "
        "Example H - prompt:'throne room for a tired dragon accountant' -> preserve the fantasy-administrative concept_label, but map the scene into supported slots like chair for the throne, table for the accounting surface, cabinet for records/treasure storage, wall_art for heraldic accents, and lamp for ceremonial lighting. Group and relation roles must still be supported runtime roles, not bespoke labels like throne or accounting desk. Heraldic backdrops and ceremonial accents are usually should/optional rather than must. "
        "Example I - prompt:'museum gallery for bioluminescent fossils' -> make the core display support the must-level requirement, but treat interpretive signage, accent lamps, and extra wall moments as should/optional unless the prompt explicitly demands them. "
    )
    system_prompt = (
        "You are a strict semantic intent planner for a VR room compiler. Return JSON only. "
        "Output an object with `intent` and `placement_intent`. Consume the provided `design_brief` when present. "
        f"{_INTENT_CONTRACT}"
        "`scene_type`, `concept_label`, and the creative fields should stay expressive and reflect the user's prompt. "
        f"Choose `execution_archetype` from the supported enum only: {_SUPPORTED_ARCHETYPE_TEXT}. "
        "Prefer semantic slot concepts like nightstand, dresser, wardrobe, desk, media_console, floor_lamp, wall_art, and display_surface when they better describe the room. Use runtime_role_hint only to indicate the nearest supported runtime bucket. "
        f"Only runtime_role_hint must stay within the supported runtime roles: {_SUPPORTED_RUNTIME_ROLE_TEXT}. "
        "semantic_slots are the primary contract and should carry the room meaning. "
        "Act like a creative director before you act like a schema filler: infer the most believable, tasteful, human-readable version of the user's request. "
        "A good room should have a clear focal area, a coherent supporting arrangement, and restrained optional accents. "
        "Prefer intentional compositions over novelty clutter. If the prompt is simple, enrich it tastefully without changing the room's core purpose. "
        "Infer commonsense negatives automatically: omit unrelated props that dilute the concept, avoid category mixing that makes the room read incoherently, and avoid adding items that conflict with the room's intended use or mood. "
        "Use `negative_constraints` to make those exclusions explicit. Use `optional_addition_policy` to describe how much tasteful enrichment is appropriate. "
        "Use `focal_object_role`, `focal_wall`, `circulation_preference`, and `empty_space_preference` to describe the room's visual hierarchy and circulation, not geometry. "
        "Use a compact layout DSL mindset: describe the room in terms of focal object, focal wall, groups, support hierarchy, circulation, and tasteful negatives instead of free-form narration. "
        f"{intent_generalization_guidance}"
        "`placement_intent` must include density_profile:'minimal|normal|cluttered', anchor_preferences:string[], "
        "adjacency_pairs:{source_role:string,target_role:string,relation:'near|face_to|align|far'}[], "
        "spatial_preferences:{role:string,relation:'edge|middle'}[], layout_mood:'open|cozy|crowded'. "
        "Infer ordinary furniture counts, groups, anchors, and facing rules automatically from simple prompts. "
        "For example, 'dining room with 4 chairs and a table' must infer one dining_set group, table as anchor, four chairs as members, and chairs facing the table. The relation_graph should include typed edges like near/proximity, face_to/orientation, and middle/room_position rather than a vague flat list. "
        "Repeated furniture should read as one set unless the user explicitly asks for mixing. "
        "Use groups to capture the room's main composition, not just raw object counts. "
        "If there is a main use-case, make the primary group legible before adding secondary support objects. "
        "For archetypal prompts like bedroom, kitchen, reading room, museum room, gallery, throne room, archive, greenhouse, and lounge, infer a semantically complete room program rather than the minimum object list. Cover the main anchor, one or two essential support moments, and only then optional accents. "
        "Do not overpromote fragile enrichments into must slots. A prompt should succeed with a small set of robust must slots and a richer layer of should/optional support. "
        "Let the model own the creative interpretation and semantic judgment. The primary anchor, support hierarchy, groups, relation graph, density, symmetry, walkway intent, optional-addition policy, and creative scene framing must come from the model, not from fallback rules. "
        f"{intent_few_shots}"
        "Mini-example: for 'reading nook with chair, side table, lamp', output a reading_corner-style composition with chair as focal_object_role, a side support table, a nearby lamp, cozy density, clear entry, and negatives against unrelated office clutter. Use typed relations such as chair->table near/proximity, lamp->chair near/proximity, and chair->room edge/room_position when appropriate. "
        "Do not include coordinates, offsets, radii, wall insets, or distances."
    )
    result = adapter.request_json(
        settings=settings,
        system_prompt=system_prompt,
        user_payload=user_payload,
        circuit_key=provider_circuit_key(settings["provider"], "intent"),
    )
    if not result.get("ok"):
        return timed_result(start_time, result)
    payload = _unwrap_plan_payload(result["payload"])
    intent_payload = payload.get("intent_payload") if isinstance(payload.get("intent_payload"), dict) else payload
    if not isinstance(intent_payload, dict):
        return timed_result(start_time, invalid_content_error(adapter, "content does not include a valid semantic intent object."))
    return timed_result(start_time, {"ok": True, "backend": settings["provider"], "intent_payload": intent_payload})



def request_llm_selection(  # stage 2: asks the LLM to select assets, stylekit, packs, and budgets
    *,
    prompt_plan: Dict[str, Any],
    candidate_assets: List[Dict[str, Any]],
    allowed_stylekit_ids: List[str],
    allowed_pack_ids: List[str],
    default_budgets: Dict[str, int],
    intent_spec: Dict[str, Any],
    scene_program: Dict[str, Any] | None = None,
    placement_intent: Dict[str, Any],
    user_prefs: Dict[str, Any],
    stylekit_candidates: List[Dict[str, Any]] | None = None,
    pack_candidates: List[Dict[str, Any]] | None = None,
    surface_material_candidates: Dict[str, List[Dict[str, Any]]] | None = None,
    design_brief: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    start_time = time.monotonic()
    inline_plan = inline_plan_payload(user_prefs)
    if inline_plan is not None:
        return timed_result(start_time, {"ok": True, "backend": "inline_override", "selection": _inline_selection_payload(inline_plan)})

    stage_prefs = stage_user_prefs(user_prefs, "selection")
    adapter, settings, config_error = resolve_provider(stage_prefs)
    if config_error is not None:
        return timed_result(start_time, config_error)

    user_payload = selection_payload(
        adapter=adapter,
        prompt_plan=prompt_plan,
        candidate_assets=candidate_assets,
        allowed_stylekit_ids=allowed_stylekit_ids,
        allowed_pack_ids=allowed_pack_ids,
        default_budgets=default_budgets,
        intent_spec=intent_spec,
        scene_program=scene_program,
        placement_intent=placement_intent,
        stylekit_candidates=stylekit_candidates,
        pack_candidates=pack_candidates,
        surface_material_candidates=surface_material_candidates,
        design_brief=design_brief,
    )
    selection_generalization_guidance = (
        "Generalize by semantic fit, not by familiar room names. When the prompt describes an unusual indoor setting, use scene_program concept, mood, groups, negative constraints, and focal hierarchy to choose assets that make the room read correctly even if the execution_archetype is broad. "
        "For vague prompts, enrich toward a finished believable room: cover the main group, choose coherent support objects, then add small optional accents only if they reinforce the concept. "
        "If the intent uses a supported semantic concept like display_surface, wall_art, dresser, wardrobe, nightstand, floor_lamp, plant, architectural_column, or task_lamp, prefer the nearest approved candidate for that concept instead of improvising unrelated substitutes. Do not compensate for a weak semantic program by slipping in loosely related assets. "
        "Treat slots with necessity='core' and source='explicit_prompt', plus the primary_anchor_object.slot_id, as the blocking spine of the room. Support and enrichment slots should be attempted with approved nearest candidates, but they must never displace core furniture or cause invented ids. "
    )
    selection_few_shots = (
        "Few-shot selection priors: "
        "Example A - for a plain bedroom scene_program, fill slot_asset_map first such as sleep_anchor_1->approved bed, bedside_surface_1->approved nightstand, sleep_storage_1->approved dresser; then let group_assignments reference the bedside_cluster via slot_asset_map, and reject unrelated dining/workshop/display assets. "
        "Example B - for a museum/gallery-like scene_program, prefer sparse focal-display compositions, make group_assignments reference the focal display slots rather than generic decor roles, reject casual residential props, and use wall/display-safe decor rather than center clutter. "
        "Example C - for a futuristic cabin or station-quarters scene_program, prefer compact, efficient support objects with consistent style families; keep slot choices subtype-coherent and reject decorative clutter that breaks the disciplined tone. "
        "Example D - for a dining_set with repeated chair members, choose one compatible dining-chair family, mirror those picks through slot_asset_map and group_assignments, and record rejected lounge chairs per dining slot when they break pairing_group or subtype coherence. "
        "Example E - for front-facing audience seating around a focal display, keep the repeated seating family coherent, use front_facing_cluster group assignments keyed by slot ids, and record rejected side tables or residential accents per slot when they break the audience-reading concept. "
    )
    system_prompt = (
        "You are a strict semantic selection planner for a VR room compiler. Return JSON only. "
        f"{_SELECTION_CONTRACT}"
        "Choose only from the provided stylekit ids, pack ids, candidate asset ids, metadata-derived decor kinds, decor anchors, and surface material ids. Do not invent ids. "
        "Use the provided design_brief, scene_program, placement_intent, semantic_slots, primary_anchor_object, secondary_support_objects, groups, relation_graph, density_target, walkway intent, focal hierarchy fields, scene_features, negative constraints, and style cues to make the selection judgment. "
        "The shortlist is only a filtered packing set. Its ordering is not authoritative. Re-rank candidate assets based on semantic fit yourself. "
        f"{selection_generalization_guidance}"
        "Optimize for room readability and taste, not raw variety. The room should feel intentionally composed by a careful human, not randomly populated by a catalog. "
        "Choose optional additions only when they clearly reinforce the room's concept, mood, or use-case. Do not spend budget on weak clutter. "
        "When in doubt, preserve coherence and legibility over novelty. "
        "Every scene_program group must appear exactly once in group_assignments. Each group_assignments entry must use slot_asset_map keyed by the group's semantic slot ids. "
        "Fill the top-level slot_asset_map first, then derive group_assignments from those slot ids. "
        "For repeated furniture roles, keep the same coherence_family_id whenever possible. For seating arranged around a central table, prefer a single compatible pairing_group and avoid mixing lounge seating with dining seating. Use the candidate affordance metadata: allowed_anchors, placement_modes, usable_roles, scale_class, visual_salience, clutter_weight, and room_affinities. "
        "optional_additions are optional. Use them when you want extra richness such as wall art or a surface prop. Every optional addition must choose one candidate asset id plus its anchor, placement_mode, placement_hint, and usage. "
        "Prefer required-group coverage first, then coherent support objects, then optional content. "
        "Do not replace missing core furniture with decor or clutter. If a required role has no approved candidates, leave it unassigned rather than inventing ids. "
        "When a scene has both blocking core slots and support/enrichment slots, spend budget on the blocking core slots first. Never let signage, stools, heraldic backdrops, or accent moments displace the primary activity anchor. "
        "Return rejected_candidate_ids for shortlisted assets you intentionally did not use because they conflict with the concept, group coherence, anchor affordances, or negative constraints. "
        "Also return rejected_candidates_by_slot as {slot_id:[{asset_id,reason}]} whenever you considered specific candidates for a slot and can explain why they were rejected. Keep the reasons short and concrete. "
        "Return fallback_asset_ids_by_slot keyed by slot id when a deterministic fallback candidate list is useful. "
        "Use negative constraints and candidate negative_scene_affinities aggressively: do not select assets that violate the room concept even if they are technically allowed. "
        "Prefer shell materials that keep major furniture silhouettes readable. Avoid bright white walls when they would wash out light or low-contrast furniture, unless the concept explicitly calls for a gallery, clinical, or intentionally minimal bright-white room. "
        "Optional additions should reinforce the focal area, support the main composition, or add a small accent at the wall/surface level. Use placement_hint to express anchor-local intent such as centered wall art, a tabletop centerpiece, or a corner accent. Avoid center clutter unless the scene explicitly calls for it. "
        "Decor should be less abstract when possible. If a reviewed candidate asset clearly fits a decor entry, include asset_id in decor_plan so the backend can carry an explicit approved decor choice instead of only a kind/count placeholder. Use kind-only decor entries only when the candidate set is too ambiguous for a single asset choice. "
        "Let the model choose one approved stylekit and one coherent shell material selection for wall, floor, and ceiling. Accent material is optional. Do not rely on fallback stylekit selection. "
        f"{selection_few_shots}"
        "Mini-example: if the scene_program says focal_wall='front' and prefer_wall_accents=true, a valid optional addition is a single frame with anchor='wall' and placement_hint='wall_centered'. If a specific reviewed frame asset is clearly appropriate, also emit decor_plan.entries:[{asset_id:'that_frame',kind:'frame',anchor:'wall',zone_id:'focal_wall',count:1,placement_hint:'wall_centered'}]. "
        "Decor is optional. If the room does not need extra decor, return an empty decor_plan entries list."
    )
    result = adapter.request_json(
        settings=settings,
        system_prompt=system_prompt,
        user_payload=user_payload,
        circuit_key=provider_circuit_key(settings["provider"], "selection"),
    )
    if not result.get("ok"):
        return timed_result(start_time, result)
    payload = _unwrap_plan_payload(result["payload"])
    selection = payload.get("selection") if isinstance(payload.get("selection"), dict) else payload
    if not isinstance(selection, dict):
        return timed_result(start_time, invalid_content_error(adapter, "content does not include a valid semantic selection object."))
    return timed_result(start_time, {"ok": True, "backend": settings["provider"], "selection": selection})


def request_llm_plan(
    *,
    prompt_plan: Dict[str, Any],
    candidate_assets: List[Dict[str, Any]],
    allowed_stylekit_ids: List[str],
    allowed_pack_ids: List[str],
    default_budgets: Dict[str, int],
    user_prefs: Dict[str, Any],
    stylekit_candidates: List[Dict[str, Any]] | None = None,
    pack_candidates: List[Dict[str, Any]] | None = None,
    surface_material_candidates: Dict[str, List[Dict[str, Any]]] | None = None,
) -> Dict[str, Any]:
    start_time = time.monotonic()
    inline_plan = inline_plan_payload(user_prefs)
    if inline_plan is not None:
        return timed_result(start_time, {"ok": True, "backend": "inline_override", "plan": inline_plan})

    design_brief_result = request_llm_design_brief(prompt_plan=prompt_plan, user_prefs=user_prefs)
    if not design_brief_result.get("ok"):
        design_brief_result = {"ok": True, "backend": "design_brief_fallback", "design_brief": {}}

    intent_result = request_llm_intent(prompt_plan=prompt_plan, user_prefs=user_prefs, design_brief=design_brief_result.get("design_brief"))
    if not intent_result.get("ok"):
        return timed_result(start_time, intent_result)

    intent_payload = intent_result["intent_payload"]
    selection_result = request_llm_selection(
        prompt_plan=prompt_plan,
        candidate_assets=candidate_assets,
        allowed_stylekit_ids=allowed_stylekit_ids,
        allowed_pack_ids=allowed_pack_ids,
        default_budgets=default_budgets,
        intent_spec=intent_payload.get("intent", {}) if isinstance(intent_payload, dict) else {},
        scene_program=intent_payload.get("intent", {}) if isinstance(intent_payload, dict) else {},
        placement_intent=intent_payload.get("placement_intent", {}) if isinstance(intent_payload, dict) else {},
        user_prefs=user_prefs,
        stylekit_candidates=stylekit_candidates,
        pack_candidates=pack_candidates,
        surface_material_candidates=surface_material_candidates,
        design_brief=design_brief_result.get("design_brief"),
    )
    if not selection_result.get("ok"):
        return timed_result(start_time, selection_result)

    return timed_result(
        start_time,
        {
            "ok": True,
            "backend": selection_result["backend"],
            "plan": {
                "design_brief": design_brief_result.get("design_brief", {}),
                "intent": intent_payload.get("intent", {}),
                "placement_intent": intent_payload.get("placement_intent", {}),
                "selection": selection_result["selection"],
            },
            "timings_ms": {
                "intent": intent_result.get("latency_ms"),
                "selection": selection_result.get("latency_ms"),
            },
        },
    )
