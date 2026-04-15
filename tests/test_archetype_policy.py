from src.planning.archetype_policy import SUPPORTED_ARCHETYPES, get_supported_archetypes


# Keep behavior deterministic so planner/runtime contracts stay stable.
def test_supported_archetypes_are_stable():
    assert SUPPORTED_ARCHETYPES == (
        "study",
        "bedroom",
        "workshop",
        "lounge",
        "kitchen",
        "bathroom",
        "generic_room",
    )


def test_get_supported_archetypes_returns_set():
    assert get_supported_archetypes() == set(SUPPORTED_ARCHETYPES)
