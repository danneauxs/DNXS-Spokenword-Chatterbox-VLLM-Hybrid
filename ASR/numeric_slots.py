"""Replace written and numeral number expressions with opaque comparison slots."""

from __future__ import annotations

import re


NUMERIC_SLOT = "<NUM>"

_DIGIT_WORDS = (
    r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)"
)
_CARDINAL_WORDS = (
    "zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    "twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    "twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|"
    "thousand|million|billion|trillion"
)
_ORDINAL_WORDS = (
    "first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    "eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|"
    "seventeenth|eighteenth|nineteenth|twentieth|thirtieth|fortieth|"
    "fiftieth|sixtieth|seventieth|eightieth|ninetieth|hundredth|"
    "thousandth|millionth|billionth"
)
_NUMBER_WORD = rf"(?:{_CARDINAL_WORDS}|{_ORDINAL_WORDS})"
# ``and`` is part of a cardinal expression only after a scale word (for
# example, ``one hundred and five``).  A bare ``two and three`` is a list of
# two independent numbers and must keep its conjunction visible to alignment.
_SCALE_WORDS = r"(?:hundred|thousand|million|billion|trillion|hundredth|thousandth|millionth|billionth)"
_SCALE_AND_SENTINEL = "__NUM_SCALE_AND__"
_WORD_NUMBER = (
    rf"{_NUMBER_WORD}(?:(?:[\s-]+){_NUMBER_WORD}|"
    rf"(?:\s+{_SCALE_AND_SENTINEL}\s+){_NUMBER_WORD})*"
)
_NUMBER_WORD_TOKEN_RE = re.compile(rf"^{_NUMBER_WORD}$", re.IGNORECASE)
_WRITTEN_DIGIT_SEQUENCE_RE = re.compile(
    rf"\b{_DIGIT_WORDS}(?:(?:[\s-]+){_DIGIT_WORDS})+\b",
    re.IGNORECASE,
)
_WRITTEN_DIGIT_HUNDRED_TIME_RE = re.compile(
    rf"\b{_DIGIT_WORDS}(?:[\s-]+){_DIGIT_WORDS}(?:[\s-]+)hundred\b",
    re.IGNORECASE,
)
_CURRENCY_WORD = r"(?:dollars?|pounds?|euros?|yen)"

_WRITTEN_RANGE_RE = re.compile(
    rf"\b{_WORD_NUMBER}\s+to\s+{_WORD_NUMBER}\b",
    re.IGNORECASE,
)
_WRITTEN_CURRENCY_RE = re.compile(
    rf"\b{_WORD_NUMBER}\s+{_CURRENCY_WORD}\b", re.IGNORECASE
)
_WRITTEN_NUMBER_RE = re.compile(rf"\b{_WORD_NUMBER}\b", re.IGNORECASE)
_NUMERIC_CURRENCY_RE = re.compile(
    rf"(?<!\w)(?:[$£€¥]\s*\d+(?:,\d{{3}})*(?:\.\d+)?|"
    rf"\d+(?:,\d{{3}})*(?:\.\d+)?\s+{_CURRENCY_WORD})(?!\w)",
    re.IGNORECASE,
)
_NUMERIC_TIME_RE = re.compile(r"(?<!\w)\d{1,2}:\d{2}(?!\w)")
_NUMERIC_RANGE_RE = re.compile(
    r"(?<!\w)\d+(?:,\d{3})*(?:\.\d+)?\s*[-–—]\s*"
    r"\d+(?:,\d{3})*(?:\.\d+)?(?:[kKmMbB])?(?!\w)"
)
_NUMERIC_TO_RANGE_RE = re.compile(
    r"(?<!\w)\d+(?:,\d{3})*(?:\.\d+)?(?:[kKmMbB])?\s+to\s+"
    r"\d+(?:,\d{3})*(?:\.\d+)?(?:[kKmMbB])?(?!\w)",
    re.IGNORECASE,
)
_NUMERIC_ORDINAL_RE = re.compile(r"(?<!\w)\d+(?:st|nd|rd|th)(?!\w)", re.IGNORECASE)
_NUMERIC_DECADE_RE = re.compile(r"(?<!\w)\d{1,4}s(?!\w)", re.IGNORECASE)
_NUMERIC_PLAIN_RE = re.compile(r"(?<![\w<])\d+(?:,\d{3})*(?:\.\d+)?(?![\w>])")
# Digits with %, optional spaces, or the bare word "percent" must share one slot
# so "40%", "40 percent", and "40percent" compare as the same numeric value.
_NUMERIC_PERCENT_RE = re.compile(
    r"(?<!\w)\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent|per\s+cent)(?!\w)",
    re.IGNORECASE,
)
_WRITTEN_PERCENT_RE = re.compile(
    rf"\b{_WORD_NUMBER}\s+(?:per\s+cent|percent)\b",
    re.IGNORECASE,
)
# Hyphenated percent forms (``zero-percent``, ``40-percent``) must share the
# same slot as spaced ``zero percent`` / ``40 percent``.
_WRITTEN_DASH_PERCENT_RE = re.compile(
    rf"\b{_WORD_NUMBER}[-–—](?:per\s+cent|percent)\b",
    re.IGNORECASE,
)
_NUMERIC_DASH_PERCENT_RE = re.compile(
    r"(?<!\w)\d+(?:,\d{3})*(?:\.\d+)?[-–—](?:%|percent|per\s+cent)(?!\w)",
    re.IGNORECASE,
)
_WRITTEN_DECADE_RE = re.compile(
    r"\b(?:twenties|thirties|forties|fifties|sixties|seventies|"
    r"eighties|nineties)\b",
    re.IGNORECASE,
)
_ATTACHED_MEASURE_UNIT_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?P<number>\d+(?:,\d{3})*(?:\.\d+)?)"
    r"(?P<unit>"
    r"milliseconds?|seconds?|minutes?|hours?|days?|weeks?|months?|years?|"
    r"miles?|kilometers?|kilometres?|meters?|metres?|feet|foot|inches?|"
    r"liters?|litres?|gallons?|pounds?|ounces?|grams?|kilograms?|"
    r"dollars?|euros?|yen"
    r")\b",
    re.IGNORECASE,
)
# Compact run measures such as ``5K`` / ``10m`` stay number + unit letter so
# they align with written ``five K`` after numeric slotting.
_ATTACHED_LETTER_UNIT_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?P<number>\d+(?:,\d{3})*(?:\.\d+)?)"
    r"(?P<unit>[kKmMbB])\b"
)
# Prose-plus-count compounds such as ``teams-30`` are a word and a number.
_ATTACHED_WORD_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?P<word>[A-Za-z]{4,})"
    r"[-–—]"
    r"(?P<number>\d+(?:,\d{3})*(?:\.\d+)?)"
    r"(?![A-Za-z0-9])"
)


def separate_attached_measure_units(text: str) -> str:
    """Insert a boundary between a numeral and a recognized spoken unit.

    Source text occasionally omits the space in forms such as ``45minutes``.
    That is a numeric expression followed by ordinary prose, not an identifier.
    Splitting only known units preserves identifiers such as ``M8-Tron``.
    Also splits compact letter units (``5K``) and prose-count dashes
    (``teams-30``) so both sides of ASR comparison keep the same tokens.
    """
    if not text:
        return ""
    text = _ATTACHED_MEASURE_UNIT_RE.sub(r"\g<number> \g<unit>", text)
    text = _ATTACHED_LETTER_UNIT_RE.sub(r"\g<number> \g<unit>", text)
    text = _ATTACHED_WORD_NUMBER_RE.sub(r"\g<word> \g<number>", text)
    return text


# Digits plus scale word (``60 thousand``, ``60,000 thousand`` rare) share one slot
# with written ``sixty thousand`` and compact ``60,000``.
_DIGIT_SCALE_RE = re.compile(
    r"(?<!\w)\d+(?:,\d{3})*(?:\.\d+)?\s+"
    r"(?:thousand|million|billion|trillion)\b",
    re.IGNORECASE,
)
# Comma between number-word and scale (``sixty, thousand``) is punctuation noise
# from dialogue/TTS text, not a list boundary for magnitudes.
_NUMBER_COMMA_SCALE_RE = re.compile(
    rf"\b({_NUMBER_WORD})\s*,\s*(thousand|million|billion|trillion)\b",
    re.IGNORECASE,
)


def separate_numeric_compound_surfaces(text: str) -> str:
    """Split numeric-led adjective compounds before typed-ID detection.

    Numeric compounds such as ``16-year-old`` and ``twenty-one-year-old`` are
    grammatical number expressions followed by ordinary words. Splitting only
    at the boundary before the first non-number component preserves internal
    number hyphens while preventing the complete surface from being classified
    as an opaque mixed alphanumeric identifier.

    Args:
        text: Surface text before identifier and number classification.

    Returns:
        Text with numeric-to-prose hyphen boundaries converted to spaces.
    """
    if not text:
        return ""

    def split_token(match: re.Match) -> str:
        """Split one hyphenated token only after its numeric components."""
        token = match.group(0)
        parts = re.split(r"[-–—]", token)
        numeric_count = 0
        for part in parts:
            clean = part.strip(".,")
            is_digit = bool(re.fullmatch(r"\d+(?:[.,]\d+)?(?:st|nd|rd|th)?", clean, re.IGNORECASE))
            is_word = bool(_NUMBER_WORD_TOKEN_RE.fullmatch(clean))
            if not (is_digit or is_word):
                break
            numeric_count += 1
        if numeric_count == len(parts) or numeric_count == 0:
            return token
        numeric_surface = "-".join(parts[:numeric_count])
        prose_surface = " ".join(parts[numeric_count:])
        return f"{numeric_surface} {prose_surface}"

    # Candidate must begin with a number word/digit and contain a hyphen. The
    # callback performs the stricter component-level test above.
    return re.sub(
        rf"(?<!\w)(?:\d+(?:[.,]\d+)?(?:st|nd|rd|th)?|{_NUMBER_WORD})(?:[-–—][A-Za-z0-9]+)+",
        split_token,
        text,
        flags=re.IGNORECASE,
    )


def _protect_scale_conjunctions(text: str) -> str:
    """Mark scale conjunctions so list conjunctions remain separate tokens."""
    return re.sub(
        rf"\b({_SCALE_WORDS})\s+and\s+(?={_NUMBER_WORD}\b)",
        rf"\1 {_SCALE_AND_SENTINEL} ",
        text,
        flags=re.IGNORECASE,
    )


def replace_spoken_number_spans(text: str) -> str:
    """Replace every written or numeral number expression with one opaque slot.

    The comparison layer deliberately ignores the number's rendered form while
    retaining surrounding prose.  Currency expressions include their unit so a
    symbol form such as ``$12`` matches the spoken form ``twelve dollars``.
    Digit-by-digit sequences such as ``six oh five`` and spoken military hours
    such as ``zero nine hundred`` are folded into one slot, but standalone
    ``oh`` stays visible as spoken filler.
    """
    if not text:
        return ""

    # Replace compound forms before their component numerals so every spoken
    # value receives one positional slot rather than several partial slots.
    normalized = _protect_scale_conjunctions(text.lower())
    # Fuse ``sixty, thousand`` → ``sixty thousand`` before slotting.
    normalized = _NUMBER_COMMA_SCALE_RE.sub(r"\1 \2", normalized)
    # Preserve range structure. Written and numeric ``to`` forms each expose
    # two numeric operands, while numeric hyphen ranges retain their existing
    # compact single-slot behavior for identifier-like surfaces.
    normalized = _WRITTEN_RANGE_RE.sub(
        f"{NUMERIC_SLOT} to {NUMERIC_SLOT}", normalized
    )
    normalized = _NUMERIC_TO_RANGE_RE.sub(
        f"{NUMERIC_SLOT} to {NUMERIC_SLOT}", normalized
    )
    for pattern in (
        _NUMERIC_CURRENCY_RE,
        _NUMERIC_TIME_RE,
        _NUMERIC_RANGE_RE,
        _NUMERIC_ORDINAL_RE,
        _WRITTEN_CURRENCY_RE,
        _WRITTEN_DASH_PERCENT_RE,
        _WRITTEN_PERCENT_RE,
        _WRITTEN_DIGIT_HUNDRED_TIME_RE,
        _WRITTEN_DIGIT_SEQUENCE_RE,
        _DIGIT_SCALE_RE,
        _WRITTEN_NUMBER_RE,
        _WRITTEN_DECADE_RE,
        _NUMERIC_DASH_PERCENT_RE,
        _NUMERIC_PERCENT_RE,
        _NUMERIC_DECADE_RE,
        _NUMERIC_PLAIN_RE,
    ):
        normalized = pattern.sub(NUMERIC_SLOT, normalized)
    return normalized
