from __future__ import annotations

import hashlib
import re
from typing import Any

def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))

def seed_from_prompt(prompt_text: str) -> int:
    digest = hashlib.sha256(prompt_text.strip().encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)

def normalize_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default
