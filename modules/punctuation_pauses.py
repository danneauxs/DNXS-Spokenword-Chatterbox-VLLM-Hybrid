"""
Punctuation-to-Pause Conversion Module
======================================

Converts punctuation in text to [pause:Xms] tags using config-defined silence durations.
Handles boundary conflicts to prevent double pauses.
"""

from config.config import PUNCTUATION_PAUSE_MAPPING as _CONFIG_MAPPING
from config import config as _cfg


# Resolve config mapping to actual duration values
def _resolve_pause_mapping():
    """Resolve string references to actual millisecond values."""
    resolved = {}
    for punct, duration_ref in _CONFIG_MAPPING.items():
        if isinstance(duration_ref, str):
            # Get the actual value from config
            resolved[punct] = getattr(_cfg, duration_ref, 150)  # Default to 150ms
        else:
            # Already a number
            resolved[punct] = duration_ref
    return resolved


# Mapping is now resolved dynamically in add_pause_tags_to_text() for real-time GUI support
# Module-level caching removed to allow runtime spinner changes
# PUNCTUATION_PAUSE_MAPPING = _resolve_pause_mapping()  # ← Moved into function


def get_punctuation_type(punct_char: str) -> str | None:
    """
    Map punctuation character to its boundary type name.

    Args:
        punct_char: Single punctuation character

    Returns:
        str: Boundary type name ('period', 'question', etc.) or None
    """
    mapping = {
        '.': 'period',
        '?': 'question',
        '!': 'exclamation',
        ';': 'semicolon',
        ':': 'colon',
        '—': 'dash',
        '...': 'ellipsis'
    }
    return mapping.get(punct_char, None)


def add_pause_tags_to_text(text: str, boundary_type: str) -> tuple[str, str | None]:
    """
    Replace punctuation with [pause:Xms] tags and handle boundary conflicts.

    Resolves pause durations dynamically from config to support real-time 
    GUI spinner changes without app restart.

    Only replaces punctuation that is not already inside [pause... ] tags.

    Args:
        text: Input text containing punctuation
        boundary_type: Current boundary type ('period', 'paragraph', etc.)

    Returns:
        tuple: (updated_text, updated_boundary_type)
    """
    import re

    # Resolve mapping fresh on each call (picks up runtime GUI changes)
    PUNCTUATION_PAUSE_MAPPING = _resolve_pause_mapping()

    updated_text = text
    replaced_punctuations = []

    # Sort by length (longest first) to handle ellipsis before periods
    sorted_punct = sorted(PUNCTUATION_PAUSE_MAPPING.keys(), key=len, reverse=True)

    for punct in sorted_punct:
        # Find all occurrences of this punctuation that are NOT inside [pause... ] tags
        # Use regex to match punctuation not preceded by '[pause' and not followed by anything until ']'
        # This is complex, so use a different approach: temporarily replace pause tags, replace punctuation, then restore

        # Temporarily replace pause tags with placeholders
        pause_tag_pattern = r'\[pause:\d+ms\]'
        pause_placeholders = []
        temp_text = updated_text

        # Replace each pause tag with a unique placeholder
        for match in re.finditer(pause_tag_pattern, temp_text):
            placeholder = f"__PAUSE_TAG_{len(pause_placeholders)}__"
            pause_placeholders.append(match.group())
            temp_text = temp_text.replace(match.group(), placeholder, 1)

        # Now replace punctuation in the temp_text (which has no pause tags)
        if punct in temp_text:
            duration = PUNCTUATION_PAUSE_MAPPING[punct]
            pause_tag = f'[pause:{duration}ms]'
            temp_text = temp_text.replace(punct, pause_tag)
            replaced_punctuations.append(punct)

        # Restore pause tags
        for i, placeholder in enumerate(pause_placeholders):
            temp_text = temp_text.replace(f"__PAUSE_TAG_{i}__", placeholder)

        updated_text = temp_text

    # All punctuation converted to pause tags, including ending punctuation
    # Boundary detection will prevent double pauses when tags are present
    return updated_text, boundary_type


def validate_pause_tags(text: str) -> bool:
    """
    Validate that pause tags in text are properly formatted.

    Args:
        text: Text containing pause tags

    Returns:
        bool: True if all tags are valid
    """
    import re

    # Check for properly formatted tags
    tag_pattern = r'\[pause:\d+ms\]'
    invalid_tags = re.findall(r'\[pause:[^\]]*\]', text)

    for tag in invalid_tags:
        if not re.match(tag_pattern, tag):
            return False

    return True