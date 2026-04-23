from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
from typing import Any, Dict, List

import httpx


DEFAULT_ELEVENLABS_BASE_URL = "https://api.elevenlabs.io"
DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
DEFAULT_MODEL_ID = "eleven_multilingual_v2"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"
DEFAULT_STABILITY = 0.45
DEFAULT_SIMILARITY_BOOST = 0.75
DEFAULT_STYLE = 0.2
DEFAULT_SPEAKER_BOOST = True
ASSISTANT_PERSONA = "room_goblin"

_BUILDING_PHASES = (
    "prompt_received",
    "planning",
    "building",
    "long_wait",
    "portal_ready",
    "build_failed",
    "asset_missing",
    "safe_spawn_failed",
    "performance_trimmed",
)


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
    return default


def _as_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(parsed, maximum))


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip().lower() for item in value if isinstance(item, str) and str(item).strip()]


def resolve_voice_settings(user_prefs: Dict[str, Any] | None = None) -> Dict[str, Any]:
    prefs = user_prefs if isinstance(user_prefs, dict) else {}
    return {
        "enabled": _as_bool(prefs.get("enable_loading_chatter", os.getenv("JUNIORIS_ENABLE_LOADING_CHATTER", "false")), False),
        "voice_id": str(prefs.get("voice_id", os.getenv("ELEVENLABS_VOICE_ID", DEFAULT_VOICE_ID))).strip() or DEFAULT_VOICE_ID,
        "model_id": str(prefs.get("voice_model_id", os.getenv("ELEVENLABS_MODEL_ID", DEFAULT_MODEL_ID))).strip() or DEFAULT_MODEL_ID,
        "output_format": str(
            prefs.get("voice_output_format", os.getenv("ELEVENLABS_OUTPUT_FORMAT", DEFAULT_OUTPUT_FORMAT))
        ).strip()
        or DEFAULT_OUTPUT_FORMAT,
        "stability": _as_float(prefs.get("voice_stability", os.getenv("ELEVENLABS_STABILITY", DEFAULT_STABILITY)), DEFAULT_STABILITY, 0.0, 1.0),
        "similarity_boost": _as_float(
            prefs.get("voice_similarity_boost", os.getenv("ELEVENLABS_SIMILARITY_BOOST", DEFAULT_SIMILARITY_BOOST)),
            DEFAULT_SIMILARITY_BOOST,
            0.0,
            1.0,
        ),
        "style": _as_float(prefs.get("voice_style", os.getenv("ELEVENLABS_STYLE", DEFAULT_STYLE)), DEFAULT_STYLE, 0.0, 1.0),
        "use_speaker_boost": _as_bool(
            prefs.get("voice_speaker_boost", os.getenv("ELEVENLABS_SPEAKER_BOOST", str(DEFAULT_SPEAKER_BOOST).lower())),
            DEFAULT_SPEAKER_BOOST,
        ),
        "play_local": _as_bool(prefs.get("voice_play_local", False), False),
        "base_url": str(prefs.get("voice_base_url", os.getenv("ELEVENLABS_BASE_URL", DEFAULT_ELEVENLABS_BASE_URL))).strip()
        or DEFAULT_ELEVENLABS_BASE_URL,
        "api_key": str(prefs.get("voice_api_key", os.getenv("ELEVENLABS_API_KEY", ""))).strip(),
        "request_timeout_s": _as_float(
            prefs.get("voice_timeout_s", os.getenv("ELEVENLABS_TIMEOUT_S", 12.0)),
            12.0,
            2.0,
            120.0,
        ),
        "assistant_persona": ASSISTANT_PERSONA,
    }


def _deterministic_pick(options: List[str], seed: str) -> str:
    if not options:
        return ""
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return options[int(digest[:8], 16) % len(options)]


def _prompt_tokens(prompt_text: str, scene_context: Dict[str, Any] | None) -> set[str]:
    values = {str(prompt_text or "").strip().lower()}
    if isinstance(scene_context, dict):
        values.update(_string_list(scene_context.get("creative_tags")))
        values.update(_string_list(scene_context.get("mood_tags")))
        values.update(_string_list(scene_context.get("style_descriptors")))
        for key in ("concept_label", "scene_type", "execution_archetype"):
            value = str(scene_context.get(key) or "").strip().lower()
            if value:
                values.add(value)
    tokens: set[str] = set()
    for value in values:
        tokens.update(part for part in value.replace("-", " ").split() if part)
    return tokens


ROOM_GOBLIN_SYSTEM_PROMPT = (
    "You are the Room Goblin, a casual VR world-building voice assistant. "
    "Speak in short, deadpan loading-screen lines. Give real progress, but phrase it like a tired dev "
    "joking about JSON, chairs, collision, greyboxes, portals, and the vibe. Never sound corporate. "
    "Never mock the user. Do not overtalk."
)
# Future LLM fallback prompt only. The runtime chatter path is deterministic.

_PHASE_TONES = {
    "prompt_received": "ack",
    "planning": "progress",
    "building": "joke",
    "long_wait": "wait",
    "portal_ready": "completion",
    "build_failed": "error",
    "asset_missing": "fallback",
    "safe_spawn_failed": "error",
    "performance_trimmed": "fallback",
}

_ROOM_GOBLIN_PHASE_BANK: Dict[str, List[str]] = {
    "prompt_received": [
        "Got it. Turning that sentence into a room. Normal computer behavior.",
        "Okay. Asking the backend to hallucinate responsibly.",
        "Cool. Converting imagination into a legally walkable box.",
    ],
    "planning": [
        "Extracting the vibe. Please do not interrupt the vibe.",
        "Design brief forming. Basically a mood board with commitment issues.",
        "The room has a concept now. Dangerous.",
        "JSON has entered the chat. Everyone act normal.",
    ],
    "building": [
        "Placing furniture. The chair and table are learning personal space.",
        "Trying to make this look like a room and not objects that fell out of a cloud.",
        "The rug is trying to become the main character.",
        "Making sure the walls are walls and not suggestions.",
    ],
    "long_wait": [
        "Still building. Not frozen. Just emotionally involved with a chair.",
        "This one's taking a bit. The room is negotiating with geometry.",
        "Still here. The backend has entered its trust-me-bro era.",
    ],
    "portal_ready": [
        "Portal's ready. Go judge the room in person.",
        "Done. Your room exists now, which is honestly a lot.",
        "World compiled. Please enter before the furniture changes its mind.",
    ],
    "build_failed": [
        "Okay, the room failed inspection. Not emotionally. Spatially.",
        "Bad news: the room did not pass the can-a-human-stand-in-it test. Great test, honestly.",
        "The build tripped over reality. Retry is allowed.",
    ],
    "asset_missing": [
        "I don't have the exact prop, so I'm swapping in the closest approved cousin.",
        "The asset library said no, which is rude but technically allowed.",
    ],
    "safe_spawn_failed": [
        "I couldn't find a safe place to spawn you. The furniture won this round.",
        "No safe spawn point passed inspection. Very rude of geometry.",
    ],
    "performance_trimmed": [
        "The room got too powerful for the headset. Trimming the drama.",
        "Performance check got nervous. Removing a little decorative chaos.",
    ],
}

_PROMPT_FLAVOR_LINES: Dict[str, List[str]] = {
    "cozy": [
        "Adding cozy clutter. Peaceful, not wizard tax audit.",
        "Softening the vibe. Legally distinct from blanket propaganda.",
    ],
    "cyberpunk": [
        "Adding neon. Cyberpunk, not gamer dentist office.",
        "Routing cables emotionally. The future requires at least one glowing line.",
    ],
    "medieval": [
        "If a chair looks throwable, that's historical accuracy.",
        "Adding old-world weight. Nobody tell the collision system about feudalism.",
    ],
    "horror": [
        "Lighting pass. Creepy, but not lawsuit creepy.",
        "Adding tension. The furniture has been instructed not to blink.",
    ],
    "futuristic": [
        "Making it futuristic. Which legally means at least one glowing line.",
        "Adding clean sci-fi surfaces. The room is pretending fingerprints do not exist.",
    ],
    "rain": [
        "Adding rainy vibes. The room is now emotionally damp.",
        "Moisture has been implied. The floor remains legally dry.",
    ],
    "gallery": [
        "Gallery mode engaged. The empty space is calling itself curation.",
        "Leaving room for art to act important.",
    ],
    "study": [
        "Study energy detected. The furniture is about to develop opinions.",
        "Adding serious surfaces. The desk already thinks it is helping.",
    ],
    "bedroom": [
        "Bedroom detected. The bed is preparing to become the protagonist.",
        "Adding sleep logic. Very brave of furniture to imply rest.",
    ],
}

_FLAVOR_ALIASES = {
    "cozy": {"cozy", "warm", "snug", "soft"},
    "cyberpunk": {"cyberpunk", "neon", "dystopian"},
    "medieval": {"medieval", "castle", "tavern", "knight"},
    "horror": {"horror", "creepy", "haunted", "eerie", "scary"},
    "futuristic": {"futuristic", "future", "sci", "scifi", "spaceship"},
    "rain": {"rain", "rainy", "storm", "stormy"},
    "gallery": {"gallery", "museum", "exhibit", "art"},
    "study": {"study", "library", "office", "reading"},
    "bedroom": {"bedroom", "sleep", "bed", "suite"},
}


def _normalize_phase(phase: str | None) -> str | None:
    token = str(phase or "").strip().lower()
    return token if token in _BUILDING_PHASES else None


def _prompt_flavor(prompt_text: str, scene_context: Dict[str, Any] | None) -> str | None:
    tokens = _prompt_tokens(prompt_text, scene_context)
    for flavor, aliases in _FLAVOR_ALIASES.items():
        if tokens & aliases:
            return flavor
    return None


def _phase_options(phase: str, prompt_text: str, scene_context: Dict[str, Any] | None) -> List[str]:
    options = list(_ROOM_GOBLIN_PHASE_BANK.get(phase, []))
    flavor = _prompt_flavor(prompt_text, scene_context)
    if flavor and phase == "building":
        return _PROMPT_FLAVOR_LINES.get(flavor, options)
    if flavor and phase in {"planning", "long_wait"}:
        options.extend(_PROMPT_FLAVOR_LINES.get(flavor, []))
    return options


def _choose_line(phase: str, options: List[str], tone: str, seed: str) -> Dict[str, str]:
    return {
        "tone": tone,
        "text": _deterministic_pick(options, f"{seed}|{phase}|{tone}"),
    }


def _phase_lines(
    prompt_text: str,
    scene_context: Dict[str, Any] | None,
    settings: Dict[str, Any],
) -> Dict[str, Dict[str, str]]:
    seed = f"{prompt_text}|{settings['assistant_persona']}"
    lines: Dict[str, Dict[str, str]] = {}
    for phase in _BUILDING_PHASES:
        options = _phase_options(phase, prompt_text, scene_context)
        lines[phase] = _choose_line(phase, options, _PHASE_TONES[phase], seed)
    return lines


def _cache_key(text: str, settings: Dict[str, Any]) -> str:
    payload = {
        "text": text,
        "voice_id": settings["voice_id"],
        "model_id": settings["model_id"],
        "output_format": settings["output_format"],
        "stability": settings["stability"],
        "similarity_boost": settings["similarity_boost"],
        "style": settings["style"],
        "use_speaker_boost": settings["use_speaker_boost"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def build_chatter_plan(
    *,
    prompt_text: str,
    scene_context: Dict[str, Any] | None = None,
    phase: str | None = None,
    user_prefs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    settings = resolve_voice_settings(user_prefs)
    phase_lines = _phase_lines(
        prompt_text=prompt_text,
        scene_context=scene_context,
        settings=settings,
    )
    requested_phase = _normalize_phase(phase)
    requested_phases = [requested_phase] if requested_phase in _BUILDING_PHASES else list(_BUILDING_PHASES)
    items: List[Dict[str, Any]] = []
    for name in requested_phases:
        chosen = phase_lines.get(name)
        if not isinstance(chosen, dict):
            continue
        item = {
            "phase": name,
            "tone": chosen["tone"],
            "text": chosen["text"],
            "cache_key": _cache_key(chosen["text"], settings),
        }
        items.append(item)
    return {
        "ok": True,
        "voice_enabled": settings["enabled"],
        "assistant_persona": settings["assistant_persona"],
        "items": items,
    }


def _artifact_extension(output_format: str) -> str:
    token = str(output_format or "").strip().lower()
    if token.startswith("mp3"):
        return "mp3"
    if token.startswith("pcm"):
        return "pcm"
    return "bin"


def _artifact_content_type(output_format: str) -> str:
    token = str(output_format or "").strip().lower()
    if token.startswith("mp3"):
        return "audio/mpeg"
    if token.startswith("pcm"):
        return "audio/L16"
    return "application/octet-stream"


def synthesize_tts_bytes(
    text: str,
    *,
    user_prefs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    settings = resolve_voice_settings(user_prefs)
    if not str(text or "").strip():
        return {"ok": False, "error_code": "invalid_request", "message": "text must be non-empty"}
    if not settings["api_key"]:
        return {"ok": False, "error_code": "voice_unavailable", "message": "ELEVENLABS_API_KEY is not configured"}
    url = f"{settings['base_url'].rstrip('/')}/v1/text-to-speech/{settings['voice_id']}"
    params = {"output_format": settings["output_format"]}
    payload = {
        "text": text,
        "model_id": settings["model_id"],
        "voice_settings": {
            "stability": settings["stability"],
            "similarity_boost": settings["similarity_boost"],
            "style": settings["style"],
            "use_speaker_boost": settings["use_speaker_boost"],
        },
    }
    headers = {
        "xi-api-key": settings["api_key"],
        "accept": _artifact_content_type(settings["output_format"]),
        "content-type": "application/json",
    }
    try:
        with httpx.Client(timeout=settings["request_timeout_s"]) as client:
            response = client.post(url, params=params, headers=headers, json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return {
            "ok": False,
            "error_code": "voice_http_error",
            "message": f"ElevenLabs returned HTTP {exc.response.status_code}",
        }
    except httpx.HTTPError as exc:
        return {
            "ok": False,
            "error_code": "voice_transport_error",
            "message": f"ElevenLabs request failed: {exc}",
        }
    return {
        "ok": True,
        "content_type": _artifact_content_type(settings["output_format"]),
        "extension": _artifact_extension(settings["output_format"]),
        "cache_key": _cache_key(text, settings),
        "audio_bytes": response.content,
        "settings": settings,
    }


def build_tts_artifact(
    text: str,
    *,
    build_root: pathlib.Path,
    user_prefs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    settings = resolve_voice_settings(user_prefs)
    normalized_text = str(text or "").strip()
    if not normalized_text:
        return {"ok": False, "error_code": "invalid_request", "message": "text must be non-empty"}
    cache_key = _cache_key(normalized_text, settings)
    extension = _artifact_extension(settings["output_format"])
    artifact_dir = pathlib.Path(build_root) / "voice_cache"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{cache_key}.{extension}"
    if artifact_path.exists():
        return {
            "ok": True,
            "cache_key": cache_key,
            "artifact_path": str(artifact_path),
            "audio_url": f"/build/voice_cache/{artifact_path.name}",
            "content_type": _artifact_content_type(settings["output_format"]),
            "settings": settings,
        }
    synthesized = synthesize_tts_bytes(normalized_text, user_prefs=user_prefs)
    if not synthesized.get("ok"):
        return synthesized
    artifact_path.write_bytes(bytes(synthesized["audio_bytes"]))
    return {
        "ok": True,
        "cache_key": cache_key,
        "artifact_path": str(artifact_path),
        "audio_url": f"/build/voice_cache/{artifact_path.name}",
        "content_type": synthesized["content_type"],
        "settings": synthesized["settings"],
    }


def maybe_play_local_audio(artifact_path: str | pathlib.Path, *, enabled: bool) -> None:
    if not enabled:
        return
    path = pathlib.Path(artifact_path)
    if not path.exists():
        return
    command: List[str] | None = None
    if os.name == "posix" and os.uname().sysname.lower() == "darwin":
        command = ["afplay", str(path)]
    if command is None:
        return
    try:
        subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return
