from __future__ import annotations

import hashlib
import re
from typing import Any

# Keep behavior deterministic so planner/runtime contracts stay stable.
def tokenize(text: str) -> set[str]:  # splits text into lowercase alphanumeric tokens for fuzzy matching
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))

def seed_from_prompt(prompt_text: str) -> int:  # generates a deterministic seed from the prompt for reproducible placement
    digest = hashlib.sha256(prompt_text.strip().encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)

def normalize_bool(value: Any, default: bool = False) -> bool:  # coerces string/bool values into Python bools for config flags
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default
