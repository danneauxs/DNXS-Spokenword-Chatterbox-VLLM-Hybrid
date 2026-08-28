"""Insert [pause:Xms] after punctuation without sending mid-clause cuts to T3.

Duration 0 means no tag and no T3 split (Pocket comma=0 behavior). Sentence-end
marks (. ? !) are tagged only at real sentence boundaries, never inside
abbreviations or decimals. Mid-clause marks (comma, dash, ellipsis) default
off. Existing [pause:Xms] tags are left intact.
"""

from __future__ import annotations

import re

from config.config import PUNCTUATION_PAUSE_MAPPING as _CONFIG_MAPPING
from config import config as _cfg

_PAUSE_TAG_RE = re.compile(r"\[pause:\d+ms\]")
_TITLE_ABBREV = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "jr",
        "sr",
        "vs",
        "etc",
        "st",
        "no",
        "prof",
        "rev",
        "gen",
        "col",
        "sgt",
        "lt",
        "mt",
        "ft",
        "inc",
        "ltd",
        "co",
        "vol",
        "pp",
        "ch",
        "fig",
        "approx",
        "est",
    }
)
_SENTENCE_END_MARKS = (".", "?", "!")
_MID_CLAUSE_MARKS = (",", ";", ":", "—", "--", "...")


def _resolve_pause_mapping():
    """Resolve string references to actual millisecond values.

    Integer GUI overrides (including comma) are kept as-is. Durations <= 0 are
    dropped so they never become tags.
    """
    resolved = {}
    mapping = getattr(_cfg, "PUNCTUATION_PAUSE_MAPPING", _CONFIG_MAPPING) or {}
    for punct, duration_ref in mapping.items():
        if isinstance(duration_ref, str):
            ms = getattr(_cfg, duration_ref, 0)
        else:
            ms = duration_ref
        try:
            ms_int = int(ms)
        except (TypeError, ValueError):
            continue
        if ms_int > 0:
            resolved[str(punct)] = ms_int
    return resolved


def get_punctuation_type(punct_char: str) -> str | None:
    """Map punctuation character to its boundary type name.

    Args:
        punct_char: Single punctuation character.

    Returns:
        Boundary type name ('period', 'question', etc.) or None.
    """
    mapping = {
        ".": "period",
        "?": "question",
        "!": "exclamation",
        ";": "semicolon",
        ":": "colon",
        "—": "dash",
        "...": "ellipsis",
        ",": "comma",
    }
    return mapping.get(punct_char, None)


def _protect_pause_tags(text: str) -> tuple[str, list[str]]:
    """Replace existing pause tags with placeholders so they are not re-parsed.

    Args:
        text: Chunk text that may already contain [pause:Xms] or ~ markers converted.

    Returns:
        Protected text and the ordered list of original tags.
    """
    tags: list[str] = []

    def _stash(match: re.Match) -> str:
        """Store one pause tag and return a unique placeholder."""
        tags.append(match.group(0))
        return f"__PAUSE_TAG_{len(tags) - 1}__"

    return _PAUSE_TAG_RE.sub(_stash, text), tags


def _restore_pause_tags(text: str, tags: list[str]) -> str:
    """Put original pause tags back after punctuation injection."""
    out = text
    for i, tag in enumerate(tags):
        out = out.replace(f"__PAUSE_TAG_{i}__", tag)
    return out


def _word_before(text: str, index: int) -> str:
    """Return the alphabetic word immediately before index."""
    j = index - 1
    while j >= 0 and not text[j].isalnum():
        j -= 1
    end = j + 1
    while j >= 0 and text[j].isalpha():
        j -= 1
    return text[j + 1 : end].lower()


def _is_sentence_end(text: str, index: int, mark: str) -> bool:
    """Return True when this mark is a real sentence end, not Mr./3.14/U.S.

    Args:
        text: Full (pause-protected) chunk string.
        index: Start index of the punctuation mark.
        mark: The mark string (. or ? or !).

    Returns:
        True if T3 may split after this mark.
    """
    if mark == ".":
        if index > 0 and index + 1 < len(text) and text[index - 1].isdigit() and text[index + 1].isdigit():
            return False
        prev = _word_before(text, index)
        if prev in _TITLE_ABBREV or len(prev) == 1:
            return False
    after = index + len(mark)
    if after >= len(text):
        return True
    if text.startswith("__PAUSE_TAG_", after):
        return True
    ch = text[after]
    if ch.isspace():
        return True
    if ch in '"\'”’)':
        return True
    return False


def _insert_after_matches(text: str, mark: str, ms: int, predicate) -> str:
    """Replace each allowed occurrence of mark with a spaced pause tag.

    Existing tags are placeholders, so they are never duplicated.

    Args:
        text: Pause-protected chunk text.
        mark: Punctuation string to find (longest marks first).
        ms: Pause duration in milliseconds; must be > 0.
        predicate: Optional fn(text, index, mark) -> bool. None means all hits.

    Returns:
        Text with tags inserted after qualifying marks.
    """
    if ms <= 0 or not mark:
        return text
    tag = f"[pause:{ms}ms]"
    out: list[str] = []
    i = 0
    n = len(text)
    mlen = len(mark)
    while i < n:
        if text.startswith("__PAUSE_TAG_", i):
            end = text.find("__", i + 12)
            if end == -1:
                out.append(text[i:])
                break
            out.append(text[i : end + 2])
            i = end + 2
            continue
        if text.startswith(mark, i) and (predicate is None or predicate(text, i, mark)):
            i += mlen
            while out and out[-1] and out[-1][-1].isspace():
                out[-1] = out[-1][:-1]
                if not out[-1]:
                    out.pop()
            out.append(" ")

            rest_start = i
            while rest_start < n and text[rest_start].isspace():
                rest_start += 1
            if text.startswith("__PAUSE_TAG_", rest_start):
                end = text.find("__", rest_start + 12)
                if end != -1:
                    out.append(text[rest_start : end + 2])
                    out.append(" ")
                    i = end + 2
                    while i < n and text[i].isspace():
                        i += 1
                    continue
            out.append(tag)
            out.append(" ")
            i = rest_start
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def add_pause_tags_to_text(text: str, boundary_type: str) -> tuple[str, str | None]:
    """Insert pause tags after punctuation that is allowed to split T3.

    Duration 0 (Pocket comma default) leaves the mark unchanged. Sentence-end
    . ? ! only tag at real sentence boundaries. Mid-clause comma/dash/ellipsis
    only tag when their mapping is a positive millisecond value.

    Args:
        text: Input chunk text.
        boundary_type: Current boundary type; returned unchanged.

    Returns:
        (updated_text, boundary_type).
    """
    if not text:
        return text, boundary_type
    mapping = _resolve_pause_mapping()
    protected, tags = _protect_pause_tags(text)
    updated = protected
    # Longest mid-clause marks first so '...' wins over '.'.
    for mark in sorted(_MID_CLAUSE_MARKS, key=len, reverse=True):
        ms = mapping.get(mark, 0)
        if ms > 0:
            updated = _insert_after_matches(updated, mark, ms, None)
    for mark in _SENTENCE_END_MARKS:
        ms = mapping.get(mark, 0)
        if ms > 0:
            updated = _insert_after_matches(updated, mark, ms, _is_sentence_end)
    return _restore_pause_tags(updated, tags), boundary_type


def split_t3_sentences(text: str) -> list[str]:
    """Split a T3 prompt on real sentence ends (. ? !), not Dr./decimals.

    One T3 generate per sentence so vLLM speech-stop cannot kill sentence two.
    Complete micro-sentences (Yes. / I see.) stay as their own prompts.

    Args:
        text: One pause-tag-free T3 prompt, possibly several sentences.

    Returns:
        Sentence strings with terminal punctuation. One item if nothing to split.
    """
    raw = str(text or "").strip()
    if not raw:
        return []
    pieces: list[str] = []
    start = 0
    i = 0
    n = len(raw)
    while i < n:
        mark = None
        for candidate in _SENTENCE_END_MARKS:
            if raw.startswith(candidate, i):
                mark = candidate
                break
        if mark is None:
            i += 1
            continue
        if not _is_sentence_end(raw, i, mark):
            i += len(mark)
            continue
        end = i + len(mark)
        while end < n and raw[end] in "\"'”’)":
            end += 1
        piece = raw[start:end].strip()
        if piece:
            pieces.append(piece)
        i = end
        while i < n and raw[i].isspace():
            i += 1
        start = i
    tail = raw[start:].strip()
    if tail:
        pieces.append(tail)
    return pieces or [raw]


def min_speech_tokens_for_text(text: str, max_tokens: int) -> int:
    """Return vLLM min_tokens so speech-stop cannot fire before the sentence is due.

    Args:
        text: Sentence being generated.
        max_tokens: Sampling max; min_tokens must stay at or below this.

    Returns:
        Inclusive floor, 0 when the prompt is empty.
    """
    words = len(str(text or "").split())
    if words <= 0 or max_tokens <= 1:
        return 0
    try:
        per_word = int(getattr(_cfg, "T3_MIN_SPEECH_TOKENS_PER_WORD", 5) or 5)
    except Exception:
        per_word = 5
    floor = max(1, words * max(1, per_word))
    return min(floor, max_tokens - 1)


def validate_pause_tags(text: str) -> bool:
    """Return True when every [pause:...] tag uses integer milliseconds."""
    invalid_tags = re.findall(r"\[pause:[^\]]*\]", text)
    tag_pattern = r"\[pause:\d+ms\]"
    for tag in invalid_tags:
        if not re.match(tag_pattern, tag):
            return False
    return True
