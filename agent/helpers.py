"""Pure helpers with no LiveKit imports (safe for fast unit tests and tooling)."""

import os
from typing import Dict, List


def get_metadata_value(metadata: Dict, keys: List[str], default: str = "") -> str:
    """Get metadata value trying multiple key variations (case-insensitive)."""
    if not metadata:
        return default

    for key in keys:
        if key in metadata:
            return metadata.get(key, default)

    metadata_lower = {k.lower(): v for k, v in metadata.items()}
    for key in keys:
        if key.lower() in metadata_lower:
            return metadata_lower[key.lower()]

    return default


def is_debug_audio_enabled() -> bool:
    """Second AudioStream on the mic is for debugging only; set AGENT_DEBUG_AUDIO=1 to enable."""
    v = os.getenv("AGENT_DEBUG_AUDIO", "").strip().lower()
    return v in ("1", "true", "yes", "on")
