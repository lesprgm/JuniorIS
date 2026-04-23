from __future__ import annotations

from pathlib import Path

from src.voice import service


def _phases(plan):
    return {item["phase"]: item for item in plan["items"]}


def test_build_chatter_plan_uses_room_goblin_defaults():
    plan = service.build_chatter_plan(
        prompt_text="quiet museum gallery with pillars",
        scene_context={"concept_label": "museum gallery", "creative_tags": ["museum"]},
    )

    assert plan["ok"] is True
    assert plan["assistant_persona"] == "room_goblin"
    phases = _phases(plan)
    assert set(phases) == set(service._BUILDING_PHASES)
    assert phases["prompt_received"]["tone"] == "ack"
    assert phases["building"]["tone"] == "joke"
    assert any(
        token in phases["planning"]["text"].lower()
        for token in ("vibe", "mood board", "json", "gallery", "curation", "art")
    )
    assert "portal" in phases["portal_ready"]["text"].lower() or "world" in phases["portal_ready"]["text"].lower()
    assert "failed" in phases["build_failed"]["text"].lower() or "retry" in phases["build_failed"]["text"].lower()


def test_build_chatter_plan_remains_deterministic_for_same_prompt():
    kwargs = {
        "prompt_text": "cozy study with rainy windows and a reading chair",
        "scene_context": {"concept_label": "study", "mood_tags": ["cozy", "rain"]},
    }

    assert service.build_chatter_plan(**kwargs)["items"] == service.build_chatter_plan(**kwargs)["items"]


def test_build_chatter_plan_uses_canonical_phase_keys_only():
    plan = service.build_chatter_plan(prompt_text="small gallery", phase="portal_ready")

    assert [item["phase"] for item in plan["items"]] == ["portal_ready"]


def test_prompt_flavor_lines_trigger_for_obvious_themes():
    plan = service.build_chatter_plan(
        prompt_text="cozy cyberpunk bedroom with rain outside",
        scene_context={"mood_tags": ["cozy"], "style_descriptors": ["cyberpunk"]},
    )
    phase_text = " ".join(item["text"].lower() for item in plan["items"])

    assert any(token in phase_text for token in ("cozy", "neon", "rainy", "bedroom", "glowing", "softening", "blanket"))


def test_room_goblin_lines_are_short_and_do_not_mock_user():
    plan = service.build_chatter_plan(prompt_text="weird room with chairs")

    forbidden = ("user", "your prompt is bad", "stupid", "dumb", "corporate")
    for item in plan["items"]:
        text = item["text"]
        assert len(text.split()) <= 18
        assert all(term not in text.lower() for term in forbidden)


def test_build_tts_artifact_writes_cached_file(monkeypatch, tmp_path: Path):
    def fake_synthesize(text, *, user_prefs=None):
        del text, user_prefs
        return {
            "ok": True,
            "content_type": "audio/mpeg",
            "extension": "mp3",
            "cache_key": "abc123",
            "audio_bytes": b"fake-mp3",
            "settings": {"voice_id": "test"},
        }

    monkeypatch.setattr(service, "synthesize_tts_bytes", fake_synthesize)
    artifact = service.build_tts_artifact("hello world", build_root=tmp_path)
    expected_cache_key = service._cache_key("hello world", service.resolve_voice_settings({}))

    assert artifact["ok"] is True
    assert artifact["audio_url"] == f"/build/voice_cache/{expected_cache_key}.mp3"
    assert (tmp_path / "voice_cache" / f"{expected_cache_key}.mp3").read_bytes() == b"fake-mp3"


def test_build_tts_artifact_uses_existing_cache_without_synthesis(monkeypatch, tmp_path: Path):
    cache_dir = tmp_path / "voice_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = service._cache_key("hello world", service.resolve_voice_settings({}))
    artifact_path = cache_dir / f"{cache_key}.mp3"
    artifact_path.write_bytes(b"cached-mp3")

    def fail_synthesize(text, *, user_prefs=None):
        raise AssertionError("synthesize_tts_bytes should not be called on a cache hit")

    monkeypatch.setattr(service, "synthesize_tts_bytes", fail_synthesize)
    artifact = service.build_tts_artifact("hello world", build_root=tmp_path)

    assert artifact["ok"] is True
    assert artifact["audio_url"] == f"/build/voice_cache/{artifact_path.name}"
    assert artifact_path.read_bytes() == b"cached-mp3"
