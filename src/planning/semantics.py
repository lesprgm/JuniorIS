from __future__ import annotations

"""Planner semantic helpers split into scene-program and support modules."""

from typing import Any, Dict, Tuple

from src.planning.scene_program import (
    complete_scene_program,
    ground_scene_program,
    normalize_intent_spec,
    normalize_scene_program,
    public_intent_spec,
    public_scene_program,
    scene_program_to_intent_spec,
    scene_program_to_placement_intent,
    validate_semantic_intent,
    validate_semantic_plan,
)
from src.planning.scene_support import (
    apply_stylekit_colors,
    apply_surface_material_colors,
    build_pack_candidates,
    build_stylekit_candidates,
    build_surface_material_candidates,
    semantic_receipts,
)
from src.planning.scene_types import (
    SUPPORTED_SEMANTIC_ROLES,
    DesignBriefSpec,
    GroundedSlotSpec,
    SceneAnchorSpec,
    SceneProgram,
    SceneRelationSpec,
    SceneSupportSpec,
    SemanticSlotSpec,
    StyleCueSpec,
    WalkwayIntentSpec,
)


# Keep behavior deterministic so planner/runtime contracts stay stable.
def normalize_prompt_mode(user_prefs: Dict[str, Any] | None) -> Tuple[bool, str]:  # validates that prompt_mode is 'llm'; returns (valid, bad_mode)
    mode = str((user_prefs or {}).get("prompt_mode") or "llm").strip().lower()
    if mode == "llm":
        return True, ""
    return False, mode


def build_prompt_plan(prompt_text: str, user_prefs: Dict[str, Any] | None) -> Dict[str, Any]:  # wraps the raw prompt into the plan structure consumed by LLM stages
    selected_prompt = str(prompt_text or "").strip()
    return {
        "mode": "llm",
        "input_prompt": selected_prompt,
        "creative_variants": [selected_prompt] if selected_prompt else [],
        "selected_variant_index": 0,
        "selected_prompt": selected_prompt,
        "strategy": "semantic_primary",
    }


__all__ = [
    "SUPPORTED_SEMANTIC_ROLES",
    "DesignBriefSpec",
    "GroundedSlotSpec",
    "SceneAnchorSpec",
    "SceneProgram",
    "SceneRelationSpec",
    "SceneSupportSpec",
    "SemanticSlotSpec",
    "StyleCueSpec",
    "WalkwayIntentSpec",
    "apply_stylekit_colors",
    "apply_surface_material_colors",
    "build_prompt_plan",
    "build_pack_candidates",
    "build_stylekit_candidates",
    "build_surface_material_candidates",
    "complete_scene_program",
    "ground_scene_program",
    "normalize_intent_spec",
    "normalize_prompt_mode",
    "normalize_scene_program",
    "public_intent_spec",
    "public_scene_program",
    "scene_program_to_intent_spec",
    "scene_program_to_placement_intent",
    "semantic_receipts",
    "validate_semantic_intent",
    "validate_semantic_plan",
]
