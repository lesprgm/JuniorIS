from __future__ import annotations

SUPPORTED_ARCHETYPES: tuple[str, ...] = (
    "study",
    "bedroom",
    "workshop",
    "lounge",
    "kitchen",
    "bathroom",
    "generic_room",
)


# Keep behavior deterministic so planner/runtime contracts stay stable.
def get_supported_archetypes() -> set[str]:  # returns the valid archetype enum set for LLM validation
    return set(SUPPORTED_ARCHETYPES)
