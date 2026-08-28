"""Detect Chapter/Part headings at chunk start for audiobook TOC ids.

Rules:
- Chapter 0 is all text before the first real Part/Chapter heading.
- A header must begin the chunk (after strip/quotes), not appear mid-prose.
- Digit, Roman, or word-number after Chapter/Part/Book all count.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_WORD_NUMBERS: Dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "twenty-one": 21,
    "twenty one": 21,
    "thirty": 30,
}

_WORD_BODY = (
    r"Twenty[-\s]?One|Thirty|Twenty|"
    r"Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|"
    r"Eleven|Twelve|Ten|One|Two|Three|Four|Five|Six|Seven|Eight|Nine"
)

_HEADER_RE = re.compile(
    rf"""
    ^\s*
    (?:
        (?P<label_chapter>Chapter|CHAPTER|Ch\.)
        \s*
        (?:
            (?P<ch_digits>\d+)
            | (?P<ch_roman>[IVXLCDM]{{1,6}})
            | (?P<ch_words>{_WORD_BODY})
        )
        |
        (?P<label_part>Part|PART|Book|BOOK)
        \s*
        (?:
            (?P<part_digits>\d+)
            | (?P<part_roman>[IVXLCDM]{{1,6}})
            | (?P<part_words>{_WORD_BODY})
        )
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_ROMAN_MAP = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8,
    "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15,
    "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20, "XXV": 25, "XXX": 30,
}


def _parse_roman(token: str) -> Optional[int]:
    """Return integer value for a common Roman numeral, or None if unknown."""
    return _ROMAN_MAP.get(token.upper())


def _parse_word_number(token: str) -> Optional[int]:
    """Return integer for a word-form number such as twenty-one."""
    key = re.sub(r"\s+", " ", token.strip().lower().replace("-", " "))
    if key in _WORD_NUMBERS:
        return _WORD_NUMBERS[key]
    return _WORD_NUMBERS.get(key.replace(" ", "-"))


def match_chapter_header(text: str) -> Optional[Dict[str, Any]]:
    """Return header info when text begins with a Part/Chapter heading.

    Args:
        text: Sentence or chunk text to inspect.

    Returns:
        Dict with title, kind (chapter|part|book), and optional number, or None.
    """
    if not text or not text.strip():
        return None
    candidate = text.lstrip()
    candidate = re.sub(r'^[\"\'“”‘’]+', "", candidate)
    match = _HEADER_RE.match(candidate)
    if not match:
        return None

    kind = "chapter"
    if match.group("label_part"):
        label = match.group("label_part")
        kind = "book" if label.lower() == "book" else "part"
        digits = match.group("part_digits")
        roman = match.group("part_roman")
        words = match.group("part_words")
        label_text = label
    else:
        digits = match.group("ch_digits")
        roman = match.group("ch_roman")
        words = match.group("ch_words")
        label_text = match.group("label_chapter")

    number: Optional[int] = None
    number_token = ""
    if digits:
        number = int(digits)
        number_token = digits
    elif roman:
        number = _parse_roman(roman)
        number_token = roman
    elif words:
        number = _parse_word_number(words)
        number_token = words

    if number_token:
        title = re.sub(r"\s+", " ", f"{label_text} {number_token}".strip())
    else:
        title = re.sub(r"\s+", " ", candidate[: match.end()].strip())

    return {
        "title": title,
        "kind": kind,
        "number": number,
        "match_end": match.end(),
    }


def assign_chapter_ids(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stamp sequential chapter_id/title on chunk dicts from heading detection.

    Chunks before the first heading stay chapter_id 0 (front matter). Each new
    heading starts the next chapter. Existing chapter_id values are kept.

    Args:
        chunks: Ordered chunk records that may already carry chapter_id.

    Returns:
        New list of chunk dicts with chapter_id and chapter_title filled in.
    """
    if chunks and any(c.get("chapter_id") is not None for c in chunks):
        out = []
        for chunk in chunks:
            row = dict(chunk)
            if row.get("chapter_id") is None:
                num = row.get("chapter_number")
                row["chapter_id"] = int(num) if num is not None else 0
            if not row.get("chapter_title"):
                cid = int(row["chapter_id"])
                row["chapter_title"] = "Front matter" if cid == 0 else f"Chapter {cid}"
            out.append(row)
        return out

    out = []
    current_id = 0
    current_title = "Front matter"
    for chunk in chunks:
        row = dict(chunk)
        header = match_chapter_header(str(row.get("text") or ""))
        if header:
            current_id += 1
            current_title = header["title"] or f"Chapter {current_id}"
        row["chapter_id"] = current_id
        row["chapter_title"] = current_title
        out.append(row)
    return out
