#!/usr/bin/env python3
"""Spoken-content comparison for ASR validation (pure functions, no torch/GUI)."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import rapidfuzz.fuzz as fuzz
from num2words import num2words

try:
    from .numeric_slots import (
        NUMERIC_SLOT,
        replace_spoken_number_spans,
        separate_attached_measure_units,
        separate_numeric_compound_surfaces,
    )
except ImportError:  # Supports direct execution from the ASR directory.
    from numeric_slots import (
        NUMERIC_SLOT,
        replace_spoken_number_spans,
        separate_attached_measure_units,
        separate_numeric_compound_surfaces,
    )

# Compiled patterns
_ORDINAL_RE = re.compile(r"^(\d+)(st|nd|rd|th)$", re.IGNORECASE)
_DIGIT_RE = re.compile(r"\b\d+\b")
_APOSTROPHE_RE = re.compile(r"['\u2019\u2018]")
_NON_ALNUM_KEEP_ID_RE = re.compile(r"[^\w\s<>]")
_WHITESPACE_RE = re.compile(r"\s+")
_BRACKET_RE = re.compile(r"[(){}\[\]]")
# TTS digital pause metadata: silence only, never spoken (must not become <NUM>).
_INLINE_PAUSE_MARKER_RE = re.compile(r"\[\d+(?:\.\d+)?s\]", re.IGNORECASE)
_DASH_RE = re.compile(r"[-–—―]+")
_DELIMITED_LETTER_SEQUENCE_RE = re.compile(
    r"(?<![A-Za-z])(?:[A-Za-z](?:[.-][A-Za-z])+)(?![A-Za-z])"
)
_ROMAN_MULTI_RE = re.compile(
    r"^(?=[ivxlcdm]{2,}$)m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$",
    re.IGNORECASE,
)
_CONTRACTION_RE = re.compile(
    r"\b([a-z0-9]+)'(t|s|re|ve|ll|d|m)\b",
    re.IGNORECASE,
)
_PHONETIC_CONTRACTION_RE = re.compile(
    r"\b(gonna|wanna|kinda|gotta|whatdya|whatya|dyou|dunno|yknow)\b",
    re.IGNORECASE,
)
_NONSTANDARD_WASNT_RE = re.compile(r"\bwas(?:e)?nt\b", re.IGNORECASE)
_YKNOW_RE = re.compile(r"\by['’]?know\b", re.IGNORECASE)
_D_CONTRACTION_WITH_FOLLOWING_RE = re.compile(
    r"\b([a-z]+)'d(?=\s+([a-z]+)\b)", re.IGNORECASE
)
_LEXICAL_POSSESSIVE_RE = re.compile(r"\b([a-z]{3,})'s\b", re.IGNORECASE)
_RAW_SPOKEN_TOKEN_RE = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)?|\d+")
_ASR_TERMINAL_DIAGNOSTIC_RE = re.compile(
    r"(?:\s*\[BLANK_AUDIO\])+(?:\s*[.?!,;:]*)$", re.IGNORECASE
)
_UPPER_ROMAN_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])([IVXLCDM]{2,})(?![A-Za-z0-9])")
_BOOK_TERM_RE = re.compile(r"\b[A-Za-z][A-Za-z']*\b")
_ACRONYM_POSSESSIVE_RE = re.compile(r"\b([A-Z]{2,})['\u2019]s\b")
_RAW_SHORT_ACRONYM_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z]{2,4})(?![A-Za-z0-9])")
_TIME_DIGIT_WORDS = {
    "zero": 0,
    "oh": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}
_TIME_TENS_WORDS = {
    "ten": 10,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
}
_TIME_SPOKEN_SEQUENCE_RE = re.compile(
    r"(?<!\w)(?P<lead>zero|oh|one|two|three|four|five|six|seven|eight|nine)\s+"
    r"(?P<hour>zero|oh|one|two|three|four|five|six|seven|eight|nine)\s+"
    r"(?P<minute>ten|twenty|thirty|forty|fifty)\s+"
    r"(?P<unit>hours?)\b",
    re.IGNORECASE,
)
_TIME_NUMERIC_TO_RE = re.compile(
    r"(?<!\w)(?P<lead>\d{1,2})\s+to\s+(?P<minute>\d{1,2})\s+"
    r"(?P<unit>hours?)\b",
    re.IGNORECASE,
)

DEFAULT_VALIDATION_CONFIG: Dict[str, Any] = {
    "pass_threshold": 0.88,
    "review_threshold": 0.72,
    "max_extra_content_words": 1,
    "max_extra_content_ratio": 0.15,
    "min_reference_coverage": 0.90,
    "phonetic_match_threshold": 0.85,
    "repeat_phrase_max_length": 6,
    "pass_tolerance_score": 0.95,
}

# Multi-character Roman only — never standalone "i" (pronoun)
ROMAN_TO_INT = {
    "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10,
    "xi": 11, "xii": 12, "xiii": 13, "xiv": 14, "xv": 15, "xvi": 16, "xvii": 17,
    "xviii": 18, "xix": 19, "xx": 20, "xxi": 21, "xxii": 22, "xxiii": 23, "xxiv": 24,
    "xxv": 25, "xxx": 30, "xl": 40, "l": 50, "lx": 60, "lxx": 70, "lxxx": 80,
    "xc": 90, "c": 100, "xiv": 14,
}
# "dc" is Washington DC / geo acronym, not Roman 600. Keep other short
# letter-strings convertible only when they are true numerals in context.
ROMAN_BLOCKLIST = {"liv", "dc"}
_ROMAN_CONTEXT_WORDS = frozenset({
    "act", "book", "chapter", "episode", "level", "part", "phase",
    "section", "season", "series", "stage", "volume", "vol",
})
_ROMAN_SUBTRACTIVE_PAIRS = frozenset({"iv", "ix", "xl", "xc", "cd", "cm"})

NEGATIONS = frozenset({"not", "no", "never", "cannot", "dont", "doesnt", "didnt", "wont", "isnt", "wasnt", "werent"})
PERSONAL_PRONOUNS = frozenset({"i", "me", "he", "him", "she", "her", "we", "us", "they", "them", "you"})
POSSESSIVE_CONTRACTION_BASES = frozenset({
    "he", "her", "here", "how", "i", "it", "she", "that", "there", "these",
    "they", "this", "those", "we", "what", "when", "where", "who", "why", "you",
})
FUNCTION_WORDS = frozenset({
    "the", "a", "an", "to", "in", "as", "of", "on", "at", "for", "by", "with",
    "from", "is", "are", "was", "were", "be", "been", "have", "has", "had",
    "do", "does", "did", "and", "or", "but", "if",
    "will", "would", "could", "should", "shall", "can", "may", "might", "must",
    "who", "whom", "whose",
    "it", "its", "he", "she", "they", "we", "you", "i", "my", "your", "his",
    "her", "their", "our", "this", "that", "these", "those", "so", "too",
})


def build_book_term_evidence(
    source_texts: Iterable[str],
    minimum_occurrences: int = 3,
) -> Dict[str, Any]:
    """Collect recurring book terms and alphabetic acronyms from source text.

    Evidence identifies terms which ASR is likely to spell inconsistently. It
    deliberately does not record a hypothesis alias or create a reusable
    per-book dictionary: the comparator still requires a clean one-to-one
    alignment in the individual sentence before accepting a substitution.
    Lowercase repeated terms may qualify when their source shape looks like a
    book term rather than ordinary prose.

    Args:
        source_texts: Original chunk text for one book.
        minimum_occurrences: Minimum source appearances needed for evidence.

    Returns:
        JSON-safe evidence keyed by normalized source term.
    """
    minimum_occurrences = max(2, int(minimum_occurrences))
    observed: Dict[str, Dict[str, Any]] = {}
    for text in source_texts:
        for match in _BOOK_TERM_RE.finditer(text or ""):
            surface = match.group(0)
            base_surface = re.sub(r"['\u2019]s$", "", surface, flags=re.IGNORECASE)
            canonical = base_surface.lower()
            is_alpha_acronym = base_surface.isalpha() and base_surface.isupper()
            lowercase_book_term = _looks_like_lowercase_book_term(base_surface)
            if (
                len(canonical) < 3
                or canonical in FUNCTION_WORDS
                or canonical in NEGATIONS
                or _is_id_like_token(base_surface)
                or _looks_numeric_word(canonical)
                or (not is_alpha_acronym and not surface[0].isupper() and not lowercase_book_term)
            ):
                continue
            entry = observed.setdefault(
                canonical,
                {
                    "surface": base_surface,
                    "occurrences": 0,
                    "capitalized_occurrences": 0,
                    "noninitial_capitalized_occurrences": 0,
                    "acronym_occurrences": 0,
                    "lowercase_unusual_occurrences": 0,
                },
            )
            entry["occurrences"] += 1
            if surface[0].isupper():
                entry["capitalized_occurrences"] += 1
            if lowercase_book_term:
                entry["lowercase_unusual_occurrences"] += 1
            if is_alpha_acronym:
                entry["acronym_occurrences"] += 1
            prefix = text[:match.start()].rstrip()
            if prefix and prefix[-1] not in ".!?":
                entry["noninitial_capitalized_occurrences"] += 1

    terms = {
        term: entry
        for term, entry in observed.items()
        if entry["occurrences"] >= minimum_occurrences
        and (
            entry["noninitial_capitalized_occurrences"] >= 1
            or entry["acronym_occurrences"] >= minimum_occurrences
            or entry["lowercase_unusual_occurrences"] >= minimum_occurrences
        )
    }
    return {
        "minimum_occurrences": minimum_occurrences,
        "terms": terms,
    }


def _looks_like_lowercase_book_term(token: str) -> bool:
    """Return True for lowercase repeated terms that look book-specific.

    This is a narrow shape heuristic for recurrent source terms such as
    ``hrum``. It excludes short function words and ordinary lowercase prose by
    requiring a low-vowel or consonant-heavy pattern.
    """
    clean = re.sub(r"[^a-z]", "", token.lower())
    if len(clean) < 4:
        return False
    if clean in FUNCTION_WORDS or clean in NEGATIONS:
        return False
    vowels = sum(char in "aeiouy" for char in clean)
    consonants = sum(char.isalpha() and char not in "aeiouy" for char in clean)
    return vowels <= 1 or (len(clean) >= 6 and consonants >= vowels + 2)


def _strict_short_acronym_tokens(
    ref_text: str,
    book_term_evidence: Optional[Dict[str, Any]],
) -> Set[str]:
    """Return short all-caps source tokens that stay non-tolerant.

    Acronym recurrence is evidence that a token matters, not permission to
    treat arbitrary ASR words as aliases. Explicitly delimited source forms
    are canonicalized before this check, so ``J-O-S`` and ``JOS`` still match.
    """
    strict_tokens = {
        match.group(1).lower()
        for match in _RAW_SHORT_ACRONYM_RE.finditer(ref_text or "")
    }
    for match in _DELIMITED_LETTER_SEQUENCE_RE.finditer(ref_text or ""):
        letters = re.sub(r"[^A-Za-z]", "", match.group(0)).lower()
        if 2 <= len(letters) <= 4:
            strict_tokens.add(letters)
    return strict_tokens


def _render_bare_two_letter_acronym(token: str) -> str:
    """Return compact letter-name rendering for a bare two-letter acronym."""
    clean = re.sub(r"[^A-Za-z]", "", token or "")
    if len(clean) != 2 or not clean.isupper():
        return ""
    return "".join(_LETTER_TO_ACRONYM_RENDERING.get(char.lower(), "") for char in clean)

_IDENTIFIER_NUMBER_WORDS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
}
_SPLIT_WORD_DIGIT_WORDS = frozenset({
    "zero", "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine",
})

# Only clear spelling/filler variants — never collapse distinct pronunciations of homographs.
# Proper-name aliases are deliberately excluded: single-token sound matching is
# generalized in ``tokens_phonetically_equal`` instead of curated per book.
SPELLING_VARIANTS = {
    "blond": "blond", "blonde": "blond",
    "toward": "toward", "towards": "toward", "tooward": "toward",
    "tward": "toward",
    "dr": "doctor",
    "mister": "mister",
    "mr": "mister",
    "mrs": "misses",
    "em": "them",
    "st": "saint",
    "alright": "all right",
    "allright": "all right",
    "onboard": "on board",
    "woulda": "would have",
    "ya": "you",
    "ha": "ha", "hah": "ha", "haw": "ha",
    "uhoh": "uh",
    "kinda": "kind of", "gonna": "going to", "wanna": "want to", "gotta": "got to",
    "gimme": "give me",
    "oh": "uh", "o": "uh", "uh": "uh", "uhh": "uh", "um": "uh", "umm": "uh",
    "er": "uh", "erm": "uh", "ah": "uh", "eh": "uh", "hmm": "uh",
    "hmmm": "uh", "mm": "uh", "mmm": "uh", "huh": "uh",
}

# Both forms are pronounced "eye" in ordinary prose. Identifier and numeric
# slots are protected before this equivalence is considered.
_SPOKEN_LETTER_HOMOPHONE_PAIRS = frozenset({frozenset({"i", "e"})})

# This title is often transcribed as the English pronoun. Keep it as a token
# equivalence rather than normalizing it, so joined spellings such as
# ``Herwylund`` can still be matched against ``Herr Wieland``.
_SPOKEN_WORD_HOMOPHONE_PAIRS = frozenset({frozenset({"herr", "her"})})

# Only these spellings are acoustically indistinguishable from a negation.
# Broad length-based matching previously allowed ``no`` to consume ``hand``.
_NEGATION_SPOKEN_EQUIVALENCE_PAIRS = frozenset({
    frozenset({"no", "know"}),
    frozenset({"no", "now"}),
    frozenset({"not", "knot"}),
})

FILLER_WORDS = frozenset({
    "uh", "uhh", "um", "umm", "er", "erm", "ah", "eh", "hmm", "hmmm",
    "mm", "mmm", "huh", "oh",
})

_CONTRACTION_TAIL = {
    "t": None,  # special: n't
    "s": "is",
    "re": "are",
    "ve": "have",
    "ll": "will",
    "d": "would",
    "m": "am",
}

_PHONETIC_MAP = {
    "gonna": "going to",
    "wanna": "want to",
    "kinda": "kind of",
    "gotta": "got to",
    "whatdya": "what do you",
    "whatya": "what do you",
    "dyou": "do you",
    "dunno": "do not know",
    "yknow": "you know",
}

_IRREGULAR_PAST_PARTICIPLES = frozenset({
    "been", "become", "begun", "bitten", "blown", "bought", "brought",
    "built", "caught", "chosen", "come", "cost", "cut", "done", "drawn",
    "driven", "drunk", "eaten", "fallen", "felt", "fought", "found",
    "given", "gone", "gotten", "grown", "had", "heard", "held", "kept", "known",
    "laid", "led", "left", "lost", "made", "meant", "met", "paid",
    "put", "read", "ridden", "run", "said", "seen", "sent", "set",
    "shown", "sold", "spoken", "spent", "stood", "stolen", "taken",
    "taught", "told", "thought", "understood", "woken", "won", "written",
})

_LETTER_NAME_TO_LETTER = {
    "ay": "a",
    "bee": "b",
    "see": "c",
    "cee": "c",
    "sea": "c",
    "dee": "d",
    "ee": "e",
    "eff": "f",
    "gee": "g",
    "aitch": "h",
    "eye": "i",
    "jay": "j",
    "kay": "k",
    "el": "l",
    "em": "m",
    "en": "n",
    "oh": "o",
    "pee": "p",
    "cue": "q",
    "are": "r",
    "ess": "s",
    "tee": "t",
    "you": "u",
    "vee": "v",
    "doubleyou": "w",
    "ex": "x",
    "why": "y",
    "zed": "z",
}

_LETTER_TO_ACRONYM_RENDERING = {
    "a": "ay",
    "b": "bee",
    "c": "see",
    "d": "dee",
    "e": "ee",
    "f": "ef",
    "g": "gee",
    "h": "aitch",
    "i": "eye",
    "j": "jay",
    "k": "kay",
    "l": "el",
    "m": "em",
    "n": "en",
    "o": "o",
    "p": "pee",
    "q": "cue",
    "r": "ar",
    "s": "ess",
    "t": "tee",
    "u": "you",
    "v": "vee",
    "w": "doubleyou",
    "x": "ex",
    "y": "why",
    "z": "zed",
}

_PLACEHOLDER_PREFIXES = ("<NUM>", "<ID")
_ID_PLACEHOLDER_RE = re.compile(r"^<ID(\d+)>$", re.IGNORECASE)
_BLOCKED_PHONETIC_PAIRS = {
    frozenset({"reed", "red"}),
    frozenset({"lyve", "liv"}),
    frozenset({"leed", "led"}),
    frozenset({"teer", "tair"}),
    frozenset({"bau", "bo"}),
    frozenset({"wynd", "wined"}),
    frozenset({"mynoot", "minit"}),
}
_RULES_PATH = Path(__file__).with_name("spoken_compare_rules.json")
_HOMOGRAPH_WHITELIST_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "asr_homograph_whitelist.json"
)
_HOMOGRAPH_WHITELIST_PAIRS: Set[Tuple[Tuple[str, ...], Tuple[str, ...]]] = set()


def _homograph_phrase_tokens(value: Any) -> Tuple[str, ...]:
    """Normalize one config phrase into the comparator's whitespace token surface.

    Args:
        value: Config value containing one editable pronunciation surface.

    Returns:
        Lowercase alphanumeric tokens, or an empty tuple for invalid input.
    """
    if not isinstance(value, str):
        return ()
    tokens = []
    for item in value.casefold().split():
        clean = re.sub(r"[^a-z0-9]", "", item)
        if clean:
            tokens.append(clean)
    return tuple(tokens)


def _homograph_pair_key(
    left: Sequence[str],
    right: Sequence[str],
) -> Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
    """Return one order-independent key for two normalized homograph forms.

    Args:
        left: First normalized pronunciation surface.
        right: Second normalized pronunciation surface.

    Returns:
        Stable pair key, or ``None`` when either surface is empty or identical.
    """
    left_key = tuple(left)
    right_key = tuple(right)
    if not left_key or not right_key or left_key == right_key:
        return None
    return (left_key, right_key) if left_key < right_key else (right_key, left_key)


def _parse_homograph_whitelist(raw_rules: Any) -> Set[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
    """Parse exact editable homograph groups into supported comparator pair keys.

    Only one-to-one, one-to-two, and one-to-three forms are admitted because
    those are the bounded alignment shapes that can preserve local content.

    Args:
        raw_rules: Decoded whitelist object or its homograph entry list.

    Returns:
        Set of order-independent normalized pronunciation pair keys.
    """
    pairs: Set[Tuple[Tuple[str, ...], Tuple[str, ...]]] = set()
    if isinstance(raw_rules, dict):
        entries = raw_rules.get("homograph_whitelist")
    else:
        entries = raw_rules
    if not isinstance(entries, list):
        return pairs
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        canonical = _homograph_phrase_tokens(entry.get("canonical"))
        equivalents = entry.get("equivalents")
        if not canonical or not isinstance(equivalents, list):
            continue
        forms = [canonical]
        forms.extend(
            phrase
            for phrase in (_homograph_phrase_tokens(item) for item in equivalents)
            if phrase
        )
        for index, left in enumerate(forms):
            for right in forms[index + 1:]:
                if {len(left), len(right)} not in ({1}, {1, 2}, {1, 3}):
                    continue
                pair = _homograph_pair_key(left, right)
                if pair is not None:
                    pairs.add(pair)
    return pairs


def _default_spoken_rules() -> Dict[str, Any]:
    """Return built-in spoken-rule defaults used when files are absent."""
    return {
        "spelling_variants": dict(SPELLING_VARIANTS),
        "phonetic_map": dict(_PHONETIC_MAP),
        "letter_name_aliases": dict(_LETTER_NAME_TO_LETTER),
        "negations": sorted(NEGATIONS),
        "blocked_phonetic_pairs": [sorted(pair) for pair in _BLOCKED_PHONETIC_PAIRS],
        "placeholder_prefixes": list(_PLACEHOLDER_PREFIXES),
        "homograph_whitelist": [],
    }


def _merge_spoken_rules(raw_rules: Any, homograph_rules: Any = None) -> Dict[str, Any]:
    """Merge comparator rules and the separate config-folder homograph whitelist.

    Args:
        raw_rules: Decoded legacy comparator-rules object.
        homograph_rules: Decoded config-folder whitelist object.

    Returns:
        Merged rule dictionary ready for module-global installation.
    """
    rules = _default_spoken_rules()
    if isinstance(homograph_rules, dict):
        rules["homograph_whitelist"] = homograph_rules.get("homograph_whitelist", [])
    if not isinstance(raw_rules, dict):
        return rules

    def merge_map(name: str) -> None:
        """Merges dictionary items from `raw_rules` into `rules`, converting keys and values to lowercase."""
        value = raw_rules.get(name)
        if isinstance(value, dict):
            for key, item in value.items():
                key_text = str(key).strip().lower()
                item_text = str(item).strip().lower()
                if key_text and item_text:
                    rules[name][key_text] = item_text

    merge_map("spelling_variants")
    merge_map("phonetic_map")
    merge_map("letter_name_aliases")

    negations = raw_rules.get("negations")
    if isinstance(negations, list):
        extra = {str(item).strip().lower() for item in negations if str(item).strip()}
        if extra:
            rules["negations"] = sorted(set(rules["negations"]) | extra)

    blocked = raw_rules.get("blocked_phonetic_pairs")
    if isinstance(blocked, list):
        extra_pairs = set()
        for item in blocked:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                left = str(item[0]).strip().lower()
                right = str(item[1]).strip().lower()
                if left and right:
                    extra_pairs.add(frozenset({left, right}))
        if extra_pairs:
            current = {frozenset(pair) for pair in rules["blocked_phonetic_pairs"]}
            rules["blocked_phonetic_pairs"] = [sorted(pair) for pair in current | extra_pairs]

    prefixes = raw_rules.get("placeholder_prefixes")
    if isinstance(prefixes, list):
        seen: List[str] = []
        for prefix in list(rules["placeholder_prefixes"]) + [str(item).strip() for item in prefixes]:
            if prefix and prefix not in seen:
                seen.append(prefix)
        if seen:
            rules["placeholder_prefixes"] = seen

    return rules


def _apply_spoken_rules(rules: Dict[str, Any]) -> None:
    """Install merged spoken rules into module globals for fast lookup."""
    global SPELLING_VARIANTS
    global _PHONETIC_MAP
    global _LETTER_NAME_TO_LETTER
    global NEGATIONS
    global _BLOCKED_PHONETIC_PAIRS
    global _PLACEHOLDER_PREFIXES
    global _HOMOGRAPH_WHITELIST_PAIRS

    SPELLING_VARIANTS = dict(rules["spelling_variants"])
    _PHONETIC_MAP = dict(rules["phonetic_map"])
    _LETTER_NAME_TO_LETTER = dict(rules["letter_name_aliases"])
    NEGATIONS = frozenset(rules["negations"])
    _BLOCKED_PHONETIC_PAIRS = {frozenset(pair) for pair in rules["blocked_phonetic_pairs"]}
    _PLACEHOLDER_PREFIXES = tuple(rules["placeholder_prefixes"])
    _HOMOGRAPH_WHITELIST_PAIRS = _parse_homograph_whitelist(rules["homograph_whitelist"])


def reload_spoken_rules() -> Dict[str, Any]:
    """Reload on-disk rules and refresh comparator lookup tables."""
    try:
        raw = json.loads(_RULES_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        raw = None
    try:
        homograph_raw = json.loads(_HOMOGRAPH_WHITELIST_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        homograph_raw = None
    rules = _merge_spoken_rules(raw, homograph_rules=homograph_raw)
    _apply_spoken_rules(rules)
    return rules


try:
    reload_spoken_rules()
except Exception:
    # Built-in defaults stay active if file is absent or malformed.
    pass


def _is_placeholder_token(token: str) -> bool:
    """Return True when token is any numeric or positional comparison slot."""
    lower = token.lower()
    return any(lower.startswith(prefix.lower()) for prefix in _PLACEHOLDER_PREFIXES)


def _placeholder_slot_index(token: str) -> Optional[int]:
    """Return positional ID slot number, excluding the generic ``<NUM>`` marker."""
    match = _ID_PLACEHOLDER_RE.fullmatch(str(token).strip())
    return int(match.group(1)) if match else None


def _collapse_identifier_sequences(normalized: str) -> str:
    """Collapse adjacent numeric slots into one format-agnostic identifier span.

    TTS text may spell an identifier one digit at a time while ASR renders the
    same spoken sequence as a compact number or hyphenated code.  Spoken
    separator words (for example ``dash`` or ``dot``) can occur between those
    slots, so a short non-prose bridge between identifier slots is consumed by
    the same contextual matcher.  Comparison must require the surrounding
    prose and identifier position, not the number of visual chunks produced by
    the two renderings.  A lone numeric slot remains separate so ordinary
    one-number alignment is unchanged.
    """
    tokens = normalized.split()
    collapsed: List[str] = []
    index = 0
    while index < len(tokens):
        if not _is_placeholder_token(tokens[index]):
            collapsed.append(tokens[index])
            index += 1
            continue

        end = index
        placeholder_count = 0
        while end < len(tokens):
            if _is_placeholder_token(tokens[end]):
                placeholder_count += 1
                end += 1
                continue
            # A bridge is structural only inside an identifier run.  Keep
            # ordinary function words, negations, and number words as prose so
            # ``3 and 4`` cannot silently erase a meaningful conjunction.
            if (
                end + 1 < len(tokens)
                and _is_placeholder_token(tokens[end + 1])
                and _is_identifier_separator_bridge(tokens[end])
            ):
                end += 1
                continue
            break
        if placeholder_count >= 2:
            collapsed.append("<IDSEQ>")
            index = end
        else:
            collapsed.append(tokens[index])
            index += 1
    return " ".join(collapsed)


def _is_identifier_separator_bridge(token: str) -> bool:
    """Return whether a token can bridge two slots in an identifier run.

    This deliberately uses token context instead of an ever-growing list of
    separator spellings.  Common prose/function words, negations, and number
    words remain protected; any short lexical bridge such as a spoken
    punctuation name is handled uniformly.
    """
    clean = _clean_context_word(token)
    if not clean or _is_placeholder_token(token):
        return False
    if clean in FUNCTION_WORDS or clean in NEGATIONS or clean in PERSONAL_PRONOUNS:
        return False
    if _looks_numeric_word(clean):
        return False
    return len(clean) <= 12


def _contains_negation_text(text: str) -> bool:
    """Return True when text contains any configured negation token."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return any(token in NEGATIONS for token in tokens)


def merge_validation_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge user overrides onto DEFAULT_VALIDATION_CONFIG without mutating defaults."""
    cfg = dict(DEFAULT_VALIDATION_CONFIG)
    if overrides:
        for key, value in overrides.items():
            if key in cfg or key in DEFAULT_VALIDATION_CONFIG:
                cfg[key] = value
            else:
                cfg[key] = value
    return cfg


def _clean_context_word(word: str) -> str:
    """Normalize a word to lowercase alphanumeric only for context checks."""
    return re.sub(r"[^a-z0-9]", "", word.lower())


def _strip_balanced_quotes(text: str) -> str:
    """Remove balanced outer quotes from a raw list candidate without touching content."""
    quote_chars = {"'", '"', "“", "”", "‘", "’"}
    raw = text.strip()
    while len(raw) >= 2 and raw[0] in quote_chars and raw[-1] in quote_chars:
        raw = raw[1:-1].strip()
    return raw


def _identifier_piece_info(token: str) -> Tuple[str, str]:
    """Return cleaned identifier text and coarse token kind for typed IDs."""
    clean = re.sub(r"[^A-Za-z0-9]", "", token)
    if not clean:
        return "", "other"
    lower = clean.lower()
    if clean.isdigit() or lower in _IDENTIFIER_NUMBER_WORDS:
        return clean, "number"
    has_letters = any(char.isalpha() for char in clean)
    has_digits = any(char.isdigit() for char in clean)
    if has_letters and has_digits:
        return clean, "mixed"
    if has_letters:
        return clean, "alpha"
    return clean, "other"


def _canonical_identifier_piece(token: str) -> str:
    """Convert one identifier token into stable lowercase alnum text."""
    clean, kind = _identifier_piece_info(token)
    if not clean:
        return ""
    if kind == "number":
        return _IDENTIFIER_NUMBER_WORDS.get(clean.lower(), clean.lower())
    return clean.lower()


def _spoken_identifier_symbol_value(token: str) -> Optional[str]:
    """Map a constrained spoken symbol rendering to identifier value text."""
    clean = _clean_context_word(token)
    if len(clean) == 1 and clean.isdigit():
        return clean
    # ASR sometimes renders spoken "eight" as the letter-name grapheme H.
    if clean == "h":
        return "8"
    if len(clean) == 1 and clean.isalpha():
        return clean.lower()
    return None


def _identifier_surface_token(token: str) -> Tuple[str, bool]:
    """Strip terminal punctuation and possessive suffix from one surface token."""
    raw = token.strip()
    raw = re.sub(r"[.?!,;:]+$", "", raw)
    possessive = raw.endswith(("'s", "’s", "'S", "’S"))
    if possessive:
        raw = raw[:-2]
    raw = re.sub(r"[.?!,;:]+$", "", raw)
    return raw, possessive


def _is_identifier_suffix_token(suffix_clean: str) -> bool:
    """Return True when a trailing token can belong to a punctuated craft ID.

    Long pure-alpha place or prose words such as ``Moscow`` must not be glued
    onto letter initialisms (``D. C. Moscow`` is geo + city, not one ID).
    Short craft suffixes such as ``Tron`` remain allowed.
    """
    if not suffix_clean or not re.fullmatch(r"[a-z0-9]+", suffix_clean):
        return False
    # Digit-bearing pieces are identifier-like (``m8``, ``x12``).
    if any(char.isdigit() for char in suffix_clean):
        return True
    # Pure letters: keep only short craft-like tails; never swallow long prose.
    return 1 < len(suffix_clean) <= 5


def _punctuated_identifier_surface(
    tokens: Sequence[str],
    start: int,
) -> Optional[Dict[str, Any]]:
    """Detect a narrow punctuation-separated identifier surface.

    The matched form is a single-letter prefix followed by a punctuated
    single-letter-or-digit component and alphabetic suffix, such as
    ``M. H.Tron`` or ``M. H.Tron's``. Ordinary prose like ``Dr. A. Smith`` is
    excluded because the prefix is not a single symbol. Geo forms such as
    ``D. C. Moscow`` are also excluded so cities are not absorbed into the ID.
    """
    if start + 1 >= len(tokens):
        return None

    prefix_raw = tokens[start].strip()
    prefix_clean = _clean_context_word(prefix_raw)
    if len(prefix_clean) != 1 or not prefix_clean.isalpha():
        return None
    if not re.search(r"[.\-]", prefix_raw):
        return None

    next_raw, next_possessive = _identifier_surface_token(tokens[start + 1])
    match = re.fullmatch(r"([A-Za-z0-9])(?:[.\-]+)([A-Za-z]+)", next_raw)
    if match:
        middle_value = _spoken_identifier_symbol_value(match.group(1))
        if middle_value is None:
            return None
        suffix = match.group(2).lower()
        if not _is_identifier_suffix_token(suffix):
            return None
        components = [prefix_clean.lower(), middle_value, suffix]
        canonical = "".join(components)
        return {
            "span_len": 2,
            "context": {
                "placeholder": "",
                "canonical_id": canonical,
                "components": components,
                "kind": "identifier",
                "words_before": [
                    _clean_context_word(token)
                    for token in tokens[max(0, start - 2):start]
                    if _clean_context_word(token)
                ],
                "words_after": [
                    _clean_context_word(token)
                    for token in tokens[start + 2:min(len(tokens), start + 4)]
                    if _clean_context_word(token)
                ],
            },
        }

    if start + 2 >= len(tokens):
        return None
    middle_raw = tokens[start + 1].strip()
    middle_clean = _clean_context_word(middle_raw)
    if len(middle_clean) != 1 or not middle_raw or not re.search(r"[.\-]", middle_raw):
        return None
    suffix_raw, _ = _identifier_surface_token(tokens[start + 2])
    suffix_clean = _clean_context_word(suffix_raw)
    if not _is_identifier_suffix_token(suffix_clean):
        return None
    if not re.fullmatch(r"[a-z]+", suffix_clean):
        return None
    # Craft tails are proper-like (``Tron``). Lowercase prose (``used``,
    # ``moscow`` after bad casing) must stay outside the ID span.
    suffix_surface = tokens[start + 2].strip()
    if not suffix_surface[:1].isupper():
        return None
    if suffix_clean in FUNCTION_WORDS or suffix_clean in NEGATIONS:
        return None
    middle_value = _spoken_identifier_symbol_value(middle_clean)
    if middle_value is None:
        return None
    components = [prefix_clean.lower(), middle_value, suffix_clean]
    canonical = "".join(components)
    return {
        "span_len": 3,
        "context": {
            "placeholder": "",
            "canonical_id": canonical,
            "components": components,
            "kind": "identifier",
            "words_before": [
                _clean_context_word(token)
                for token in tokens[max(0, start - 2):start]
                if _clean_context_word(token)
            ],
            "words_after": [
                _clean_context_word(token)
                for token in tokens[start + 3:min(len(tokens), start + 5)]
                if _clean_context_word(token)
            ],
        },
    }


def _identifier_components(tokens: Sequence[str]) -> List[str]:
    """Split identifier surfaces into comparable letters, numbers, and words."""
    components: List[str] = []
    for token in tokens:
        raw, _ = _identifier_surface_token(token)
        clean = re.sub(r"[^A-Za-z0-9]", "", raw)
        for piece in re.findall(r"[A-Za-z]+|\d+", clean):
            components.append(_canonical_identifier_piece(piece))
    return components


def _typed_identifier_span_length(tokens: Sequence[str], start: int) -> int:
    """Return span length for a typed identifier starting at ``start``."""
    punctuated = _punctuated_identifier_surface(tokens, start)
    if punctuated:
        return int(punctuated["span_len"])
    clean, kind = _identifier_piece_info(tokens[start])
    if not clean:
        return 0
    if _ORDINAL_RE.match(clean.lower()):
        return 0
    if kind == "mixed":
        # Numeric decade notation such as ``50s`` is a number expression, not
        # an identifier. Leave it for numeric-slot normalization so following
        # prose such as ``early`` cannot be swallowed into the wildcard.
        if re.fullmatch(r"\d{1,4}s", clean, re.IGNORECASE):
            return 0
        # Attached percent measures (``70percent``, ``12.5pct``) are numeric
        # values, not identifiers. Do not absorb following prose like ``water``.
        if re.fullmatch(
            r"\d+(?:,\d{3})*(?:\.\d+)?(?:percent|pct|%)",
            clean,
            re.IGNORECASE,
        ):
            return 0
        # Hyphenated ordinal eras such as ``21st-century`` are number+prose,
        # not identifiers. Leave them for numeric slot + dash splitting so
        # ``21st century`` and ``twenty first century`` share structure.
        if re.fullmatch(r"\d+(?:st|nd|rd|th)century", clean, re.IGNORECASE):
            return 0
        # Digit + single-letter unit (``5K``, ``10m``) is a measure, not an ID
        # that may swallow following prose such as ``would``.
        if re.fullmatch(r"\d+[kKmMbB]", clean):
            return 0
        # Long lowercase-led word + trailing digits (``teams-30``) is prose
        # plus a number after dash splitting, not a typed craft ID like MSV1.
        word_digits = re.fullmatch(r"([A-Za-z]+)(\d+)", clean)
        if word_digits:
            alpha = word_digits.group(1)
            if len(alpha) >= 4 and not alpha.isupper():
                return 0
            if len(alpha) >= 5:
                return 0
        if start + 1 < len(tokens):
            _, next_kind = _identifier_piece_info(tokens[start + 1])
            next_clean, _ = _identifier_piece_info(tokens[start + 1])
            if (
                next_kind == "alpha"
                and "-" not in tokens[start]
                and "'" not in tokens[start]
                and "’" not in tokens[start]
                and next_clean.lower() not in FUNCTION_WORDS
                and len(next_clean) <= 5
            ):
                # Bare mixed tokens like ``M8`` can absorb one spoken suffix word.
                return 2
        return 1
    # Spoken craft IDs like ``M eight Tron`` use a *single* letter head plus a
    # number-word plus a short suffix. Two-letter English heads (``No``, ``To``,
    # ``In``, ``On``…) produce false IDs for ordinary prose such as ``No one is``.
    if kind != "alpha" or len(clean) != 1:
        return 0
    if clean.lower() in FUNCTION_WORDS or clean.lower() in NEGATIONS:
        return 0
    if not tokens[start][:1].isalpha() or not tokens[start][:1].isupper():
        return 0
    if start + 2 >= len(tokens):
        return 0
    _, next_kind = _identifier_piece_info(tokens[start + 1])
    if next_kind != "number":
        return 0
    _, suffix_kind = _identifier_piece_info(tokens[start + 2])
    if suffix_kind != "alpha":
        return 0
    suffix_clean, _ = _identifier_piece_info(tokens[start + 2])
    # Do not treat function-word tails (``is``, ``in``, ``to``) as ID suffixes.
    if suffix_clean.lower() in FUNCTION_WORDS or suffix_clean.lower() in NEGATIONS:
        return 0
    return 3


def _spoken_numeric_code_surface(
    tokens: Sequence[str],
    start: int,
) -> Optional[Dict[str, Any]]:
    """Detect a spoken digit-plus-letter code such as ``eight cee``.

    Compact ASR output may render ``8C`` as a numeral followed by a spoken
    letter name. This recognizes that general two-component code shape while
    excluding ambiguous digit words such as ``oh`` that commonly belong to
    spoken numeric sequences.

    Args:
        tokens: Whitespace-tokenized surface text.
        start: Candidate first-token index.

    Returns:
        Identifier context and span length when the candidate is code-like;
        otherwise ``None``.
    """
    if start + 1 >= len(tokens):
        return None
    first_clean, first_kind = _identifier_piece_info(tokens[start])
    if first_kind != "number":
        return None
    second_raw, _ = _identifier_surface_token(tokens[start + 1])
    second_clean = _clean_context_word(second_raw)
    if (
        not second_clean
        or second_clean in _IDENTIFIER_NUMBER_WORDS
        or second_clean in FUNCTION_WORDS
        or second_clean in PERSONAL_PRONOUNS
        or second_clean in NEGATIONS
    ):
        return None
    letter = _spoken_letter_alias(second_clean)
    if not letter or len(letter) != 1:
        return None
    components = [_canonical_identifier_piece(first_clean), letter]
    return {
        "span_len": 2,
        "context": {
            "placeholder": "",
            "canonical_id": "".join(components),
            "components": components,
            "kind": "identifier",
            "words_before": [
                _clean_context_word(token)
                for token in tokens[max(0, start - 2):start]
                if _clean_context_word(token)
            ],
            "words_after": [
                _clean_context_word(token)
                for token in tokens[start + 2:min(len(tokens), start + 4)]
                if _clean_context_word(token)
            ],
        },
    }


def _replace_typed_identifier_spans(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Replace typed identifier spans with slots while keeping prose intact."""
    if not text:
        return "", []

    tokens = text.split()
    out_tokens: List[str] = []
    id_contexts: List[Dict[str, Any]] = []
    i = 0
    while i < len(tokens):
        spoken_code = _spoken_numeric_code_surface(tokens, i)
        if spoken_code:
            span_len = int(spoken_code["span_len"])
            placeholder = f"<TID{len(id_contexts)}>"
            out_tokens.append(placeholder)
            context = dict(spoken_code["context"])
            context["placeholder"] = placeholder
            id_contexts.append(context)
            i += span_len
            continue
        punctuated = _punctuated_identifier_surface(tokens, i)
        if punctuated:
            span_len = int(punctuated["span_len"])
            span_tokens = tokens[i:i + span_len]
            placeholder = f"<TID{len(id_contexts)}>"
            out_tokens.append(placeholder)
            context = dict(punctuated["context"])
            context["placeholder"] = placeholder
            id_contexts.append(context)
            i += span_len
            continue
        span_len = _typed_identifier_span_length(tokens, i)
        if span_len:
            span_tokens = tokens[i:i + span_len]
            placeholder = f"<TID{len(id_contexts)}>"
            out_tokens.append(placeholder)
            id_contexts.append(
                {
                    "placeholder": placeholder,
                    "canonical_id": "".join(_identifier_components(span_tokens)),
                    "components": _identifier_components(span_tokens),
                    "kind": "identifier",
                    "words_before": [
                        _clean_context_word(token)
                        for token in tokens[max(0, i - 2):i]
                        if _clean_context_word(token)
                    ],
                    "words_after": [
                        _clean_context_word(token)
                        for token in tokens[i + span_len:min(len(tokens), i + span_len + 2)]
                        if _clean_context_word(token)
                    ],
                }
            )
            i += span_len
            continue
        out_tokens.append(tokens[i])
        i += 1
    return " ".join(out_tokens), id_contexts


def _finalize_comparison_slots(
    text: str,
    typed_contexts: Sequence[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Assign one positional slot sequence across typed IDs and numeric values."""
    tokens = text.split()
    output: List[str] = []
    contexts: List[Dict[str, Any]] = []
    for index, token in enumerate(tokens):
        typed_match = re.fullmatch(r"<TID(\d+)>[^\w<>]*", token, re.IGNORECASE)
        numeric_match = re.fullmatch(r"<NUM>[^\w<>]*", token, re.IGNORECASE)
        if typed_match:
            source_index = int(typed_match.group(1))
            context = dict(typed_contexts[source_index])
            placeholder = f"<ID{len(contexts)}>"
            context["placeholder"] = placeholder
            contexts.append(context)
            output.append(placeholder)
            continue
        if numeric_match:
            placeholder = f"<ID{len(contexts)}>"
            contexts.append(
                {
                    "placeholder": placeholder,
                    "canonical_id": "numeric",
                    "kind": "numeric",
                    "components": [NUMERIC_SLOT],
                    "words_before": [
                        _clean_context_word(previous)
                        for previous in tokens[max(0, index - 2):index]
                        if not previous.startswith("<") and _clean_context_word(previous)
                    ],
                    "words_after": [
                        _clean_context_word(next_token)
                        for next_token in tokens[index + 1:min(len(tokens), index + 3)]
                        if not next_token.startswith("<") and _clean_context_word(next_token)
                    ],
                }
            )
            output.append(placeholder)
            continue
        output.append(token)
    return " ".join(output), contexts


def _is_id_like_token(token: str) -> bool:
    """Return True when token looks like an identifier, not prose or an ordinal."""
    clean = _clean_context_word(token)
    if not clean:
        return False
    if _ORDINAL_RE.match(clean):
        return False
    has_letters = any(char.isalpha() for char in clean)
    has_digits = any(char.isdigit() for char in clean)
    if has_letters and has_digits and len(clean) >= 3:
        return True
    if not has_letters and has_digits and len(clean) >= 4:
        return True
    return False


def _expand_contraction_match(match: re.Match) -> str:
    """Expand a single apostrophe contraction with token-aware rules."""
    left = match.group(1).lower()
    tail = match.group(2).lower()
    if tail == "t":
        # Special irregulars spelled without a visible n before the apostrophe
        irregular_t = {
            "can": "cannot",
            "won": "will not",
            "shan": "shall not",
            "ain": "is not",
        }
        if left in irregular_t:
            return irregular_t[left]
        # Standard n't forms: don't, isn't, weren't, ...
        if left.endswith("n") and len(left) > 1:
            stem = left[:-1]
            return f"{stem} not"
        return f"{left} not"
    mapped = _CONTRACTION_TAIL.get(tail)
    if mapped is None:
        return match.group(0).lower()
    return f"{left} {mapped}"


def _looks_like_past_participle(token: str) -> bool:
    """Return True when following context makes an apostrophe-d mean ``had``."""
    word = re.sub(r"[^a-z]", "", token.lower())
    return bool(word) and (word.endswith("ed") or word in _IRREGULAR_PAST_PARTICIPLES)


def _expand_d_contraction_with_context(match: re.Match) -> str:
    """Expand ``'d`` using participle and interrogative grammar context."""
    subject = match.group(1).lower()
    following = match.group(2).lower()
    if subject in {"how", "what", "where", "when", "why"} and following in {
        "i", "you", "we", "they", "he", "she",
    }:
        auxiliary = "did"
    else:
        auxiliary = "had" if _looks_like_past_participle(following) else "would"
    return f"{subject} {auxiliary}"


def _lexical_possessive_bases(text: str) -> Set[str]:
    """Return noun-like apostrophe-s bases from unnormalized hypothesis text.

    This records a narrow ASR spelling clue without changing normal contraction
    expansion. ``George's`` may be an ASR spelling of a book term, whereas
    ``she's`` remains a real contraction and must still expand to ``she is``.
    """
    normalized = _APOSTROPHE_RE.sub("'", text or "")
    return {
        match.group(1).lower()
        for match in _LEXICAL_POSSESSIVE_RE.finditer(normalized)
        if match.group(1).lower() not in POSSESSIVE_CONTRACTION_BASES
    }


def _strip_evidenced_book_term_possessives(
    text: str,
    book_term_tokens: Set[str],
) -> str:
    """Remove possessive suffixes only from recurring source book terms.

    A source form such as ``Hrum's`` identifies the recurring term ``Hrum``;
    expanding it to ``Hrum is`` creates a fake word for ASR to miss. Ordinary
    possessives and contractions remain unchanged because this rule requires
    dynamic evidence from repeated source text.
    """
    if not text or not book_term_tokens:
        return text
    normalized = _APOSTROPHE_RE.sub("'", text)
    return _LEXICAL_POSSESSIVE_RE.sub(
        lambda match: (
            match.group(1)
            if match.group(1).lower() in book_term_tokens
            else match.group(0)
        ),
        normalized,
    )


def _raw_token_difference(
    ref_text: str,
    hyp_text: str,
) -> Optional[Tuple[List[re.Match[str]], List[re.Match[str]], int, int, int]]:
    """Locate the sole differing raw-token span shared by two sentences.

    Returns match lists plus the shared-prefix length and exclusive end indexes
    for reference and hypothesis. Callers use this only for narrow repairs;
    more than one differing region is never collapsed into a safe equivalence.
    """
    ref_matches = list(_RAW_SPOKEN_TOKEN_RE.finditer(ref_text or ""))
    hyp_matches = list(_RAW_SPOKEN_TOKEN_RE.finditer(hyp_text or ""))
    if not ref_matches or not hyp_matches:
        return None

    ref_tokens = [match.group(0).lower().replace("\u2019", "'") for match in ref_matches]
    hyp_tokens = [match.group(0).lower().replace("\u2019", "'") for match in hyp_matches]
    prefix = 0
    while (
        prefix < len(ref_tokens)
        and prefix < len(hyp_tokens)
        and ref_tokens[prefix] == hyp_tokens[prefix]
    ):
        prefix += 1

    suffix = 0
    while (
        suffix < len(ref_tokens) - prefix
        and suffix < len(hyp_tokens) - prefix
        and ref_tokens[-(suffix + 1)] == hyp_tokens[-(suffix + 1)]
    ):
        suffix += 1
    ref_end = len(ref_tokens) - suffix if suffix else len(ref_tokens)
    hyp_end = len(hyp_tokens) - suffix if suffix else len(hyp_tokens)
    if prefix == ref_end and prefix == hyp_end:
        return None
    return ref_matches, hyp_matches, prefix, ref_end, hyp_end


def _raw_time_two_to_equivalence(
    ref_text: str,
    hyp_text: str,
) -> Tuple[Optional[Dict[str, Any]], str, str]:
    """Align a time-like ``two`` sequence with ASR's ``to`` rendering.

    In military-style time speech, a source sequence such as ``zero two
    thirty hours`` can be transcribed as ``0 to 30 hours``. The source-side
    numeric sequence and the hypothesis-side range are both required, and the
    source's middle component must literally be ``two``. Ordinary ranges such
    as ``twenty to thirty hours`` therefore remain untouched.

    Args:
        ref_text: Source text before normalization.
        hyp_text: ASR text before normalization.

    Returns:
        Evidence plus rewritten text when one guarded time variant is found;
        otherwise ``(None, ref_text, hyp_text)``.
    """
    ref_matches = list(_TIME_SPOKEN_SEQUENCE_RE.finditer(ref_text or ""))
    hyp_matches = list(_TIME_NUMERIC_TO_RE.finditer(hyp_text or ""))
    if len(ref_matches) != 1 or len(hyp_matches) != 1:
        return None, ref_text, hyp_text

    ref_match = ref_matches[0]
    hyp_match = hyp_matches[0]
    if ref_match.group("hour").lower() != "two":
        return None, ref_text, hyp_text
    lead_value = _TIME_DIGIT_WORDS[ref_match.group("lead").lower()]
    minute_value = _TIME_TENS_WORDS[ref_match.group("minute").lower()]
    if int(hyp_match.group("lead")) != lead_value:
        return None, ref_text, hyp_text
    if int(hyp_match.group("minute")) != minute_value:
        return None, ref_text, hyp_text

    ref_surface = ref_match.group(0)
    hyp_surface = hyp_match.group(0)
    rewritten_ref = (
        f"{ref_text[:ref_match.start()]}{hyp_surface}"
        f"{ref_text[ref_match.end():]}"
    )
    return (
        {
            "kind": "time_two_to_surface_variation",
            "expected": ref_surface,
            "hypothesis": hyp_surface,
        },
        rewritten_ref,
        hyp_text,
    )


def _raw_formatting_equivalence(
    ref_text: str,
    hyp_text: str,
) -> Optional[Dict[str, Any]]:
    """Recognize transcripts differing only by capitalization or punctuation.

    This intentionally retains apostrophes inside words.  It accepts only an
    identical complete raw word sequence, so punctuation changes cannot invoke
    later surface-rewrite rules and move a repeated word to another position.
    """
    ref_tokens = [
        match.group(0).lower().replace("\u2019", "'")
        for match in _RAW_SPOKEN_TOKEN_RE.finditer(ref_text or "")
    ]
    hyp_tokens = [
        match.group(0).lower().replace("\u2019", "'")
        for match in _RAW_SPOKEN_TOKEN_RE.finditer(hyp_text or "")
    ]
    if ref_tokens and ref_tokens == hyp_tokens and ref_text != hyp_text:
        return {
            "kind": "raw_formatting_equivalence",
            "expected": ref_text,
            "hypothesis": hyp_text,
        }
    return None


def _raw_normalized_equivalence(
    ref_text: str,
    hyp_text: str,
    canon_lookup: Optional[dict],
) -> Optional[Dict[str, Any]]:
    """Recognize complete equality under the ordinary safe normalizer.

    This check occurs before optional raw repair rules.  If the canonical
    spoken sequence already agrees, applying a structural rewrite adds no
    information and risks choosing the wrong repeated occurrence.
    """
    ref_normalized, _ = normalize(ref_text, canon_lookup)
    hyp_normalized, _ = normalize(hyp_text, canon_lookup)
    if ref_normalized and ref_normalized == hyp_normalized and ref_text != hyp_text:
        return {
            "kind": "raw_normalized_equivalence",
            "expected": ref_text,
            "hypothesis": hyp_text,
        }
    return None


def _raw_reduced_pronoun_equivalence(
    ref_text: str,
    hyp_text: str,
) -> Optional[Dict[str, Any]]:
    """Accept source ``'em`` when ASR writes the acoustically close ``him``.

    The complete raw token streams must otherwise agree position-for-position;
    this is a pronunciation rule for the reduced object pronoun, never a
    general ``them``/``him`` substitution.
    """
    ref_matches = list(_RAW_SPOKEN_TOKEN_RE.finditer(ref_text or ""))
    hyp_matches = list(_RAW_SPOKEN_TOKEN_RE.finditer(hyp_text or ""))
    if not ref_matches or len(ref_matches) != len(hyp_matches):
        return None
    variations: List[Tuple[str, str]] = []
    for ref_match, hyp_match in zip(ref_matches, hyp_matches):
        ref_token = ref_match.group(0).lower().replace("\u2019", "'")
        hyp_token = hyp_match.group(0).lower().replace("\u2019", "'")
        if ref_token == hyp_token:
            continue
        if (ref_token, hyp_token) != ("em", "him"):
            return None
        variations.append((ref_match.group(0), hyp_match.group(0)))
    if not variations:
        return None
    return {
        "kind": "reduced_pronoun_equivalence",
        "expected": "'em",
        "hypothesis": "him",
        "count": len(variations),
    }


def _raw_call_sign_article_equivalence(
    ref_text: str,
    hyp_text: str,
) -> Optional[Dict[str, Any]]:
    """Accept ASR's article before a capitalized two-word call sign.

    Source dialogue often writes ``You're Low Boy`` as a call sign, while ASR
    may insert an orthographic ``a`` before the same spoken title.  Require a
    second capitalized source word and an otherwise exact raw sequence; this
    cannot hide a general added article in ordinary prose.
    """
    ref_matches = list(_RAW_SPOKEN_TOKEN_RE.finditer(ref_text or ""))
    hyp_matches = list(_RAW_SPOKEN_TOKEN_RE.finditer(hyp_text or ""))
    if len(hyp_matches) != len(ref_matches) + 1:
        return None
    for hyp_index, hyp_match in enumerate(hyp_matches):
        if hyp_match.group(0).lower() != "a" or hyp_index < 1:
            continue
        ref_index = hyp_index
        if ref_index + 1 >= len(ref_matches):
            continue
        prior = ref_matches[ref_index - 1].group(0).lower().replace("\u2019", "'")
        if prior not in {"you're", "youre"}:
            continue
        title = [ref_matches[ref_index].group(0), ref_matches[ref_index + 1].group(0)]
        if not all(word[:1].isupper() for word in title):
            continue
        without_article = [
            match.group(0).lower().replace("\u2019", "'")
            for index, match in enumerate(hyp_matches)
            if index != hyp_index
        ]
        ref_tokens = [
            match.group(0).lower().replace("\u2019", "'")
            for match in ref_matches
        ]
        if without_article == ref_tokens:
            return {
                "kind": "call_sign_article_equivalence",
                "expected": " ".join(title),
                "hypothesis": f"a {title[0]} {title[1]}",
            }
    return None


def _raw_slash_name_equivalence(
    ref_text: str,
    hyp_text: str,
    canon_lookup: Optional[dict],
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Separate one slash-joined personal name when ASR speaks two words.

    A slash is occasionally used in source prose to join a surname pair.  This
    does not apply to compact codes: both source components must be alphabetic
    name-sized words, the surname must match exactly, and the full rewritten
    utterance must normalize identically to ASR.
    """
    pattern = re.compile(
        r"(?<![A-Za-z])([A-Za-z]{3,}(?:['\u2019][A-Za-z]+)?)/([A-Za-z]{3,})(?![A-Za-z])"
    )
    matches = list(pattern.finditer(ref_text or ""))
    if len(matches) != 1:
        return None, ref_text
    source = matches[0]
    left = _clean_context_word(source.group(1))
    right = _clean_context_word(source.group(2))
    hyp_tokens = list(_RAW_SPOKEN_TOKEN_RE.finditer(hyp_text or ""))
    candidates = []
    for index in range(len(hyp_tokens) - 1):
        hyp_left = _clean_context_word(hyp_tokens[index].group(0))
        hyp_right = _clean_context_word(hyp_tokens[index + 1].group(0))
        if (
            hyp_right == right
            and tokens_phonetically_equal(left, hyp_left)
        ):
            candidates.append((hyp_tokens[index].group(0), hyp_tokens[index + 1].group(0)))
    if len(candidates) != 1:
        return None, ref_text
    replacement = " ".join(candidates[0])
    rewritten = f"{ref_text[:source.start()]}{replacement}{ref_text[source.end():]}"
    rewritten_tokens = [match.group(0) for match in _RAW_SPOKEN_TOKEN_RE.finditer(rewritten)]
    hypothesis_tokens = [match.group(0) for match in _RAW_SPOKEN_TOKEN_RE.finditer(hyp_text)]
    if not rewritten_tokens or not hypothesis_tokens:
        return None, ref_text
    # Compare the remaining stream in local spans.  Whole-utterance normalize
    # can itself be affected by the slash form; a changed span is accepted only
    # when the ordinary numeric normalizer proves those local words equivalent.
    for tag, ref_start, ref_end, hyp_start, hyp_end in SequenceMatcher(
        None,
        [token.lower() for token in rewritten_tokens],
        [token.lower() for token in hypothesis_tokens],
        autojunk=False,
    ).get_opcodes():
        if tag == "equal":
            continue
        ref_span = " ".join(rewritten_tokens[ref_start:ref_end])
        hyp_span = " ".join(hypothesis_tokens[hyp_start:hyp_end])
        ref_normalized, _ = normalize(ref_span, canon_lookup)
        hyp_normalized, _ = normalize(hyp_span, canon_lookup)
        if ref_normalized and ref_normalized == hyp_normalized:
            continue
        # Event-style proper labels can be rendered with a spurious apostrophe
        # before their number (``Expo Seventy-Four`` -> ``Expo's 74``).  The
        # source label must be capitalized and the remaining number span must
        # independently normalize exactly, so ordinary possessive prose stays
        # outside this rule.
        ref_part = rewritten_tokens[ref_start:ref_end]
        hyp_part = hypothesis_tokens[hyp_start:hyp_end]
        if (
            len(ref_part) >= 2
            and len(hyp_part) >= 2
            and ref_part[0][0].isupper()
            and hyp_part[0].lower().replace("\u2019", "'")
            == f"{ref_part[0].lower()}'s"
            and normalize(" ".join(ref_part[1:]), canon_lookup)[0]
            == normalize(" ".join(hyp_part[1:]), canon_lookup)[0]
        ):
            continue
        return None, ref_text
    return (
        {
            "kind": "slash_joined_name_equivalence",
            "expected": source.group(0),
            "hypothesis": replacement,
        },
        rewritten,
    )


def _raw_spoken_code_equivalence(
    ref_text: str,
    hyp_text: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Render a spoken two-part letter-number code in ASR's compact form.

    The rule is structural: each source number must be an English cardinal
    from zero through ninety-nine, both leading letters must agree, and ASR
    must contain the same two digit values.  It cannot accept a changed code.
    """
    word_to_number = {
        re.sub(r"[- ]", "", num2words(value, lang="en").lower()): value
        for value in range(100)
    }
    source_pattern = re.compile(
        r"(?<![A-Za-z0-9])([A-Za-z])[- ]([A-Za-z-]+)\s*/\s*([A-Za-z])[- ]([A-Za-z-]+)(?![A-Za-z0-9])"
    )
    hyp_pattern = re.compile(
        r"(?<![A-Za-z0-9])([A-Za-z])(\d+)\s*[-/]\s*([A-Za-z])(\d+)(?![A-Za-z0-9])"
    )
    source_matches = list(source_pattern.finditer(ref_text or ""))
    hyp_matches = list(hyp_pattern.finditer(hyp_text or ""))
    if len(source_matches) != 1 or len(hyp_matches) != 1:
        return None, ref_text
    source = source_matches[0]
    hypothesis = hyp_matches[0]
    source_numbers = [
        word_to_number.get(re.sub(r"[- ]", "", source.group(index).lower()))
        for index in (2, 4)
    ]
    if (
        None in source_numbers
        or source.group(1).lower() != hypothesis.group(1).lower()
        or source.group(3).lower() != hypothesis.group(3).lower()
        or source_numbers != [int(hypothesis.group(2)), int(hypothesis.group(4))]
    ):
        return None, ref_text
    replacement = hypothesis.group(0)
    rewritten = f"{ref_text[:source.start()]}{replacement}{ref_text[source.end():]}"
    return (
        {
            "kind": "spoken_letter_number_code_equivalence",
            "expected": source.group(0),
            "hypothesis": replacement,
        },
        rewritten,
    )


def _raw_lowercase_n_conjunction_equivalence(
    ref_text: str,
    hyp_text: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Expand prose ``n`` to ``and`` only when every other raw token agrees."""
    ref_matches = list(_RAW_SPOKEN_TOKEN_RE.finditer(ref_text or ""))
    hyp_matches = list(_RAW_SPOKEN_TOKEN_RE.finditer(hyp_text or ""))
    if not ref_matches or len(ref_matches) != len(hyp_matches):
        return None, ref_text
    replacements: List[Tuple[int, int, str]] = []
    for ref_match, hyp_match in zip(ref_matches, hyp_matches):
        ref_token = ref_match.group(0)
        hyp_token = hyp_match.group(0)
        if ref_token.lower() == hyp_token.lower():
            continue
        if ref_token != "n" or hyp_token.lower() != "and":
            return None, ref_text
        replacements.append((ref_match.start(), ref_match.end(), hyp_token))
    if not replacements:
        return None, ref_text
    rewritten = ref_text
    for start, end, replacement in reversed(replacements):
        rewritten = f"{rewritten[:start]}{replacement}{rewritten[end:]}"
    return (
        {
            "kind": "reduced_conjunction_equivalence",
            "expected": "n",
            "hypothesis": "and",
            "count": len(replacements),
        },
        rewritten,
    )


def _raw_lexical_possessive_filler_equivalence(
    ref_text: str,
    hyp_text: str,
) -> Tuple[Optional[Dict[str, Any]], str, str]:
    """Rewrite one lexical possessive plus harmless filler ASR variation.

    A source phrase such as ``shuttle's um paint`` may be transcribed as
    ``shuttles on paint``.  Accept only this two-token difference when the
    possessive base, following content word, and every other raw token agree.
    This keeps a filler mishear from creating a fake ``is`` token while never
    accepting changed content, protected words, identifiers, or numbers.
    """
    ref_matches = list(_RAW_SPOKEN_TOKEN_RE.finditer(ref_text or ""))
    hyp_matches = list(_RAW_SPOKEN_TOKEN_RE.finditer(hyp_text or ""))
    if len(ref_matches) != len(hyp_matches) or len(ref_matches) < 3:
        return None, ref_text, hyp_text
    ref_tokens = [
        match.group(0).lower().replace("\u2019", "'") for match in ref_matches
    ]
    hyp_tokens = [
        match.group(0).lower().replace("\u2019", "'") for match in hyp_matches
    ]
    candidates: List[int] = []
    for index in range(len(ref_tokens) - 2):
        possessive = re.fullmatch(r"([a-z]{2,})'s", ref_tokens[index])
        if not possessive:
            continue
        base = possessive.group(1)
        if base in POSSESSIVE_CONTRACTION_BASES or base in NEGATIONS:
            continue
        if hyp_tokens[index] not in {base, f"{base}s"}:
            continue
        ref_filler = _clean_context_word(ref_tokens[index + 1])
        hyp_filler = _clean_context_word(hyp_tokens[index + 1])
        if (
            ref_filler not in FILLER_WORDS
            or not re.fullmatch(r"[a-z]{1,3}", hyp_filler)
            or hyp_filler in NEGATIONS
            or _looks_numeric_word(hyp_filler)
            or _is_id_like_token(hyp_tokens[index + 1])
        ):
            continue
        if _clean_context_word(ref_tokens[index + 2]) != _clean_context_word(
            hyp_tokens[index + 2]
        ):
            continue
        if any(
            ref_token != hyp_token
            for token_index, (ref_token, hyp_token) in enumerate(
                zip(ref_tokens, hyp_tokens)
            )
            if token_index not in {index, index + 1}
        ):
            continue
        candidates.append(index)
    if len(candidates) != 1:
        return None, ref_text, hyp_text
    index = candidates[0]
    replacements = [
        (ref_matches[index].span(), hyp_matches[index].group(0)),
        (ref_matches[index + 1].span(), hyp_matches[index + 1].group(0)),
    ]
    rewritten_reference = ref_text
    for (start, end), replacement in reversed(replacements):
        rewritten_reference = (
            f"{rewritten_reference[:start]}{replacement}{rewritten_reference[end:]}"
        )
    return (
        {
            "kind": "lexical_possessive_filler_variation",
            "expected": f"{ref_matches[index].group(0)} {ref_matches[index + 1].group(0)}",
            "hypothesis": f"{hyp_matches[index].group(0)} {hyp_matches[index + 1].group(0)}",
        },
        rewritten_reference,
        hyp_text,
    )


def _raw_compact_boundary_surface_equivalences(
    ref_text: str,
    hyp_text: str,
    phonetic_threshold: float = 0.85,
) -> Tuple[List[Dict[str, Any]], str, str]:
    """Align every anchored compact word-boundary variation before number slots.

    This compares only one-to-two/three raw spans whose compact surfaces are
    identical, or whose hyphen/apostrophe-bearing surface is strongly phonetic.
    It runs before number parsing so ``sixpack`` and ``six-pack`` retain their
    shared lexical surface.  Negation, digit-bearing IDs, and ambiguous matches
    remain untouched.
    """
    ref_matches = list(_RAW_SPOKEN_TOKEN_RE.finditer(ref_text or ""))
    hyp_matches = list(_RAW_SPOKEN_TOKEN_RE.finditer(hyp_text or ""))
    if not ref_matches or not hyp_matches:
        return [], ref_text, hyp_text

    def clean_span(matches: Sequence[re.Match[str]]) -> str:
        """Return one lowercase alphanumeric compact surface for a raw span."""
        return "".join(_clean_context_word(match.group(0)) for match in matches)

    def anchored(ref_start: int, ref_end: int, hyp_start: int, hyp_end: int) -> bool:
        """Require every available neighbor so repeated prose cannot cross-match."""
        left_matches = ref_start > 0 and hyp_start > 0 and (
            _clean_context_word(ref_matches[ref_start - 1].group(0))
            == _clean_context_word(hyp_matches[hyp_start - 1].group(0))
        )
        right_matches = ref_end < len(ref_matches) and hyp_end < len(hyp_matches) and (
            _clean_context_word(ref_matches[ref_end].group(0))
            == _clean_context_word(hyp_matches[hyp_end].group(0))
        )
        # A single shared neighbor is insufficient in prose with repeated words:
        # it can pair a structured span with a different occurrence farther in
        # the sentence and reorder otherwise identical speech.  At a true text
        # edge one neighbor is all that exists; everywhere else both must hold.
        has_left = ref_start > 0 and hyp_start > 0
        has_right = ref_end < len(ref_matches) and hyp_end < len(hyp_matches)
        return (
            (not has_left or left_matches)
            and (not has_right or right_matches)
            and (left_matches or right_matches)
        )

    candidates: List[Tuple[int, int, int, int, str, str]] = []
    for ref_len, hyp_len in ((1, 2), (2, 1), (1, 3), (3, 1)):
        for ref_start in range(len(ref_matches) - ref_len + 1):
            ref_end = ref_start + ref_len
            ref_span = ref_matches[ref_start:ref_end]
            for hyp_start in range(len(hyp_matches) - hyp_len + 1):
                hyp_end = hyp_start + hyp_len
                if not anchored(ref_start, ref_end, hyp_start, hyp_end):
                    continue
                hyp_span = hyp_matches[hyp_start:hyp_end]
                words = [
                    _clean_context_word(match.group(0))
                    for match in [*ref_span, *hyp_span]
                ]
                raw_surfaces = [match.group(0) for match in [*ref_span, *hyp_span]]
                if (
                    not all(words)
                    or any(word in NEGATIONS for word in words)
                    or any(_is_id_like_token(surface) for surface in raw_surfaces)
                ):
                    continue
                ref_compact = clean_span(ref_span)
                hyp_compact = clean_span(hyp_span)
                exact = ref_compact == hyp_compact
                ref_structured = (
                    "'" in ref_text[ref_span[0].start():ref_span[-1].end()]
                    or "\u2019" in ref_text[ref_span[0].start():ref_span[-1].end()]
                    or "-" in ref_text[ref_span[0].start():ref_span[-1].end()]
                )
                hyp_structured = (
                    "'" in hyp_text[hyp_span[0].start():hyp_span[-1].end()]
                    or "\u2019" in hyp_text[hyp_span[0].start():hyp_span[-1].end()]
                    or "-" in hyp_text[hyp_span[0].start():hyp_span[-1].end()]
                )
                has_hyphen = (
                    "-" in ref_text[ref_span[0].start():ref_span[-1].end()]
                    or "-" in hyp_text[hyp_span[0].start():hyp_span[-1].end()]
                )
                # A clitic must participate in the actual multi-token span;
                # otherwise ``Hamm's`` could incorrectly consume ``of hams``.
                ref_internal_apostrophe = any(
                    re.search(r"[A-Za-z]['\u2019][A-Za-z]", match.group(0))
                    and not re.search(r"['\u2019]s$", match.group(0), re.IGNORECASE)
                    for match in ref_span
                )
                hyp_internal_apostrophe = any(
                    re.search(r"[A-Za-z]['\u2019][A-Za-z]", match.group(0))
                    and not re.search(r"['\u2019]s$", match.group(0), re.IGNORECASE)
                    for match in hyp_span
                )
                apostrophe_bearing_span = any(
                    "'" in match.group(0) or "\u2019" in match.group(0)
                    for match in [*ref_span, *hyp_span]
                )
                protected_word_beside_apostrophe = any(
                    word in PERSONAL_PRONOUNS or word == "if" for word in words
                )
                structured = (
                    (ref_len > 1 and ref_structured)
                    or (hyp_len > 1 and hyp_structured)
                    or ref_internal_apostrophe
                    or hyp_internal_apostrophe
                )
                phonetic = (
                    not exact
                    and structured
                    # A fuzzy boundary rewrite must not consume a protected
                    # word beside an apostrophe-bearing contraction.  For example, it
                    # must not turn ``We aren't`` into ``aren't`` or absorb
                    # ``her`` into ``t'ward her``. Dedicated contraction and
                    # possessive rules handle those surfaces without deleting
                    # adjacent speech.
                    and not (
                        apostrophe_bearing_span and protected_word_beside_apostrophe
                    )
                    and min(len(ref_compact), len(hyp_compact)) >= 5
                    and abs(len(ref_compact) - len(hyp_compact)) <= 2
                    # Numeric values must agree exactly; ``sixpack`` may join
                    # ``six-pack``, but never a different ``five-pack``.
                    and not any(
                        compact.startswith(number_word)
                        for compact in (ref_compact, hyp_compact)
                        for number_word in _SPLIT_WORD_DIGIT_WORDS
                    )
                    and (
                        tokens_phonetically_equal(
                            ref_compact, hyp_compact, phonetic_threshold
                        )
                        or spoken_token_equivalent(
                            ref_compact, hyp_compact, phonetic_threshold
                        ) == "phonetic"
                    )
                )
                # Ordinary exact boundaries remain visible to the existing DP
                # evidence path. Raw handling is only needed before numeric
                # slotting when one side actually uses a hyphen.
                raw_exact = exact and has_hyphen
                if not raw_exact and not phonetic:
                    continue
                candidates.append((
                    ref_start, ref_end, hyp_start, hyp_end,
                    "exact_surface" if raw_exact else "structured_phonetic",
                    hyp_text[hyp_span[0].start():hyp_span[-1].end()],
                ))

    # Apply every unambiguous, non-overlapping pair.  There is intentionally no
    # per-chunk count limit: independently anchored spelling surfaces are not
    # extra spoken content.
    replacements: List[Tuple[int, int, str, Dict[str, Any]]] = []
    used_ref: Set[int] = set()
    used_hyp: Set[int] = set()
    for ref_start, ref_end, hyp_start, hyp_end, method, replacement in sorted(
        candidates,
        key=lambda item: (
            item[0],
            item[4] != "exact_surface",
            (item[1] - item[0]) + (item[3] - item[2]),
            item[2],
        ),
    ):
        if any(index in used_ref for index in range(ref_start, ref_end)) or any(
            index in used_hyp for index in range(hyp_start, hyp_end)
        ):
            continue
        start = ref_matches[ref_start].start()
        end = ref_matches[ref_end - 1].end()
        expected = ref_text[start:end]
        replacements.append((
            start,
            end,
            replacement,
            {
                "kind": "raw_compact_boundary_surface",
                "expected": expected,
                "hypothesis": replacement,
                "method": method,
            },
        ))
        used_ref.update(range(ref_start, ref_end))
        used_hyp.update(range(hyp_start, hyp_end))

    rewritten_ref = ref_text
    for start, end, replacement, _evidence in reversed(replacements):
        rewritten_ref = f"{rewritten_ref[:start]}{replacement}{rewritten_ref[end:]}"
    return (
        [evidence for _start, _end, _replacement, evidence in replacements],
        rewritten_ref,
        hyp_text,
    )


def _raw_apostrophe_s_surface_equivalence(
    ref_text: str,
    hyp_text: str,
) -> Tuple[List[Dict[str, Any]], str, str]:
    """Rewrite all anchored apostrophe-s surfaces before contraction expansion.

    ASR may write a possessive or contracted ``'s`` without the apostrophe,
    either as the bare base word or with a plural-looking final ``s``.  Exact
    preceding and following raw words anchor the rewrite, so it remains safe
    when another independent word-boundary difference shifts token positions.
    """
    ref_matches = list(_RAW_SPOKEN_TOKEN_RE.finditer(ref_text or ""))
    hyp_matches = list(_RAW_SPOKEN_TOKEN_RE.finditer(hyp_text or ""))
    candidates: List[Tuple[Dict[str, Any], int, str]] = []
    for ref_index, ref_match in enumerate(ref_matches):
        ref_surface = ref_match.group(0)
        ref_apostrophe_s = re.fullmatch(
            r"([a-z]{2,})'s",
            ref_surface.lower().replace("\u2019", "'"),
        )
        if not ref_apostrophe_s or ref_index == 0 or ref_index + 1 >= len(ref_matches):
            continue
        if _is_id_like_token(ref_surface):
            continue
        base = ref_apostrophe_s.group(1)
        ref_neighbors = (
            _clean_context_word(ref_matches[ref_index - 1].group(0)),
            _clean_context_word(ref_matches[ref_index + 1].group(0)),
        )
        if not all(ref_neighbors):
            continue
        for hyp_index, hyp_match in enumerate(hyp_matches):
            if hyp_index == 0 or hyp_index + 1 >= len(hyp_matches):
                continue
            hyp_surface = hyp_match.group(0)
            ref_comparison_surface = ref_surface.lower().replace("\u2019", "'")
            hyp_comparison_surface = hyp_surface.lower().replace("\u2019", "'")
            if hyp_comparison_surface == ref_comparison_surface:
                continue
            hyp_clean = _clean_context_word(hyp_surface)
            if (
                hyp_clean not in {base, f"{base}s"}
                and not tokens_phonetically_equal(base, hyp_clean)
            ):
                continue
            hyp_neighbors = (
                _clean_context_word(hyp_matches[hyp_index - 1].group(0)),
                _clean_context_word(hyp_matches[hyp_index + 1].group(0)),
            )
            if hyp_neighbors != ref_neighbors:
                continue
            candidates.append((
                {
                    "kind": "apostrophe_s_surface_variation",
                    "expected": ref_surface,
                    "hypothesis": hyp_surface,
                },
                ref_index,
                hyp_surface,
            ))
    by_reference: Dict[int, List[Tuple[Dict[str, Any], str]]] = {}
    for evidence, ref_index, replacement in candidates:
        by_reference.setdefault(ref_index, []).append((evidence, replacement))
    replacements: List[Tuple[int, int, str, Dict[str, Any]]] = []
    for ref_index, matches in by_reference.items():
        # Repeated surrounding words can make a raw candidate ambiguous; leave
        # those to token alignment instead of choosing an arbitrary ASR token.
        if len(matches) != 1:
            continue
        evidence, replacement = matches[0]
        start, end = ref_matches[ref_index].span()
        replacements.append((start, end, replacement, evidence))
    rewritten_ref = ref_text
    for start, end, replacement, _evidence in reversed(replacements):
        rewritten_ref = f"{rewritten_ref[:start]}{replacement}{rewritten_ref[end:]}"
    return [evidence for _start, _end, _replacement, evidence in replacements], rewritten_ref, hyp_text


def _raw_possessive_or_contraction_surface_fusion_equivalence(
    ref_text: str,
    hyp_text: str,
) -> Tuple[Optional[Dict[str, Any]], str, str]:
    """Normalize one exact apostrophe-bearing surface fusion before tokenization.

    Possessives and contractions must remain raw until their intended surface
    is recovered: expanding ``Tran's face`` to ``tran is face`` would otherwise
    prevent an exact match with ASR ``Transface``. Generic prose fusion is not
    rewritten here; it reaches dynamic-programming alignment as explicit
    boundary evidence instead.
    """
    difference = _raw_token_difference(ref_text, hyp_text)
    if difference is None:
        return None, ref_text, hyp_text
    ref_matches, hyp_matches, prefix, ref_end, hyp_end = difference
    ref_span = ref_matches[prefix:ref_end]
    hyp_span = hyp_matches[prefix:hyp_end]
    if {len(ref_span), len(hyp_span)} not in ({1, 2}, {1, 3}):
        return None, ref_text, hyp_text

    split_span = ref_span if len(ref_span) > 1 else hyp_span
    single_span = hyp_span if len(hyp_span) == 1 else ref_span
    split_surfaces = [match.group(0) for match in split_span]
    single_surface = single_span[0].group(0)
    if not any("'" in surface or "\u2019" in surface for surface in [*split_surfaces, single_surface]):
        return None, ref_text, hyp_text
    split_words = [_clean_context_word(surface) for surface in split_surfaces]
    single_word = _clean_context_word(single_surface)
    if not single_word or not all(split_words):
        return None, ref_text, hyp_text
    if any(
        _looks_numeric_word(word)
        or _is_id_like_token(surface)
        or (len(surface) == 1 and surface.isupper())
        for word, surface in zip(split_words, split_surfaces)
    ):
        return None, ref_text, hyp_text
    if (
        _looks_numeric_word(single_word)
        or _is_id_like_token(single_surface)
        or (len(single_surface) == 1 and single_surface.isupper())
    ):
        return None, ref_text, hyp_text

    compact_split = "".join(re.sub(r"[^A-Za-z]", "", surface) for surface in split_surfaces).lower()
    compact_single = re.sub(r"[^A-Za-z]", "", single_surface).lower()
    if not compact_split or compact_split != compact_single:
        return None, ref_text, hyp_text

    ref_start = ref_span[0].start()
    ref_stop = ref_span[-1].end()
    hyp_start = hyp_span[0].start()
    hyp_stop = hyp_span[-1].end()
    expected = ref_text[ref_start:ref_stop]
    hypothesis = hyp_text[hyp_start:hyp_stop]
    if len(ref_span) > 1:
        rewritten_ref = f"{ref_text[:ref_start]}{hypothesis}{ref_text[ref_stop:]}"
        return (
            {
                "kind": "exact_surface_word_fusion",
                "specialized_rule": "apostrophe_surface",
                "expected": expected,
                "hypothesis": hypothesis,
            },
            rewritten_ref,
            hyp_text,
        )
    rewritten_hyp = f"{hyp_text[:hyp_start]}{expected}{hyp_text[hyp_stop:]}"
    return (
        {
            "kind": "exact_surface_word_split",
            "specialized_rule": "apostrophe_surface",
            "expected": expected,
            "hypothesis": hypothesis,
        },
        ref_text,
        rewritten_hyp,
    )


def _raw_contraction_equivalence(
    ref_text: str,
    hyp_text: str,
) -> Tuple[Optional[Dict[str, Any]], str, str]:
    """Normalize a single safe contracted-auxiliary ASR difference.

    ASR can drop a contraction's subject (``He'd`` to ``had``), drop its
    auxiliary (``He'd`` to ``he``), or supply one source prose omits
    (``you`` to ``you've``). The rest of the raw sentence must be identical,
    and the rule accepts only personal-pronoun contractions with non-negated
    auxiliary forms.
    """
    ref_matches = list(_RAW_SPOKEN_TOKEN_RE.finditer(ref_text or ""))
    hyp_matches = list(_RAW_SPOKEN_TOKEN_RE.finditer(hyp_text or ""))
    if not ref_matches or len(ref_matches) != len(hyp_matches):
        return None, ref_text, hyp_text
    candidates: List[Tuple[Dict[str, Any], str, int]] = []
    for index, (ref_match, hyp_match) in enumerate(zip(ref_matches, hyp_matches)):
        ref_surface = ref_match.group(0)
        hyp_surface = hyp_match.group(0)
        ref_clean = _clean_context_word(ref_surface)
        hyp_clean = _clean_context_word(hyp_surface)
        if (
            not ref_clean
            or not hyp_clean
            or ref_clean in NEGATIONS
            or hyp_clean in NEGATIONS
            or _is_id_like_token(ref_surface)
            or _is_id_like_token(hyp_surface)
            or _looks_numeric_word(ref_clean)
            or _looks_numeric_word(hyp_clean)
        ):
            continue

        ref_contraction = re.fullmatch(
            r"([a-z]+)'d",
            ref_surface.lower().replace("\u2019", "'"),
        )
        if ref_contraction and ref_contraction.group(1) in PERSONAL_PRONOUNS:
            # Preserve the existing contextual 'd logic, but accept a raw
            # apostrophe-free ASR spelling only for the specific she'd -> shed
            # case that drops the apostrophe without changing surrounding text.
            if (
                ref_contraction.group(1) == "she"
                and hyp_clean == "shed"
                and all(
                    ref_token.group(0).lower().replace("\u2019", "'")
                    == hyp_token.group(0).lower().replace("\u2019", "'")
                    for token_index, (ref_token, hyp_token) in enumerate(
                        zip(ref_matches, hyp_matches)
                    )
                    if token_index != index
                )
            ):
                candidates.append((
                    {
                        "kind": "contracted_pronoun_spelling",
                        "expected": ref_surface,
                        "hypothesis": hyp_surface,
                    },
                    "hypothesis",
                    index,
                ))
                continue
            following = (
                ref_matches[index + 1].group(0)
                if index + 1 < len(ref_matches)
                else ""
            )
            expanded, _ = normalize(f"{ref_surface} {following}")
            expanded_tokens = expanded.split()
            if (
                len(expanded_tokens) >= 2
                and expanded_tokens[0] == ref_contraction.group(1)
                and expanded_tokens[1] == hyp_clean
                and hyp_clean in {"had", "would"}
            ):
                candidates.append((
                    {
                        "kind": "contracted_auxiliary_omission",
                        "expected": ref_surface,
                        "hypothesis": hyp_surface,
                    },
                    "reference",
                    index,
                ))
                continue
            if (
                len(expanded_tokens) >= 2
                and expanded_tokens[0] == ref_contraction.group(1)
                and expanded_tokens[1] in {"had", "would"}
                and hyp_clean == ref_contraction.group(1)
            ):
                candidates.append((
                    {
                        "kind": "contracted_auxiliary_drop",
                        "expected": ref_surface,
                        "hypothesis": hyp_surface,
                    },
                    "reference",
                    index,
                ))
                continue

        hyp_contraction = re.fullmatch(
            r"([a-z]+)'(re|ve|ll|d|m)",
            hyp_surface.lower().replace("\u2019", "'"),
        )
        if hyp_contraction and hyp_contraction.group(1) == ref_clean:
            candidates.append((
                {
                    "kind": "contracted_auxiliary_addition",
                    "expected": ref_surface,
                    "hypothesis": hyp_surface,
                },
                "hypothesis",
                index,
            ))

    if len(candidates) != 1:
        return None, ref_text, hyp_text
    evidence, target, index = candidates[0]
    if target == "reference":
        start, end = ref_matches[index].span()
        replacement = hyp_matches[index].group(0)
        return evidence, f"{ref_text[:start]}{replacement}{ref_text[end:]}", hyp_text
    start, end = hyp_matches[index].span()
    replacement = ref_matches[index].group(0)
    return evidence, ref_text, f"{hyp_text[:start]}{replacement}{hyp_text[end:]}"


def _raw_delimited_letter_sequence_equivalence(
    ref_text: str,
    hyp_text: str,
) -> Tuple[Optional[Dict[str, Any]], str, str]:
    """Rewrite one explicit delimited letter sequence when letters match.

    This accepts source forms such as ``J-O-S`` or ``J.O.S.`` against the
    concatenated ASR token ``JOS`` only when the raw letters match exactly.
    Bare acronyms like ``SOB`` are not touched.
    """
    difference = _raw_token_difference(ref_text, hyp_text)
    if difference is None:
        return None, ref_text, hyp_text
    ref_matches, hyp_matches, prefix, ref_end, hyp_end = difference
    ref_span = ref_matches[prefix:ref_end]
    hyp_span = hyp_matches[prefix:hyp_end]
    if len(ref_span) == len(hyp_span):
        return None, ref_text, hyp_text
    if len(ref_span) <= 1 or len(hyp_span) != 1:
        return None, ref_text, hyp_text
    ref_letters = "".join(
        re.sub(r"[^a-z]", "", match.group(0).lower()) for match in ref_span
    )
    hyp_letters = "".join(
        re.sub(r"[^a-z]", "", match.group(0).lower()) for match in hyp_span
    )
    if not ref_letters or ref_letters != hyp_letters:
        return None, ref_text, hyp_text
    if not all(len(re.sub(r"[^a-z]", "", match.group(0).lower())) == 1 for match in ref_span):
        return None, ref_text, hyp_text
    if not all(re.sub(r"[^a-z]", "", match.group(0).lower()).isalpha() for match in hyp_span):
        return None, ref_text, hyp_text
    if len(ref_letters) < 2 or len(ref_letters) > 4:
        return None, ref_text, hyp_text
    start, end = ref_matches[prefix].span()
    ref_end_span = ref_matches[ref_end - 1].span()[1]
    replacement = hyp_matches[prefix].group(0)
    return (
        {
            "kind": "letter_sequence_surface_variation",
            "expected": ref_text[start:ref_end_span],
            "hypothesis": replacement,
        },
        f"{ref_text[:start]}{replacement}{ref_text[ref_end_span:]}",
        hyp_text,
    )


def _raw_two_letter_acronym_ref_spans(
    ref_text: str,
    ref_matches: Sequence[re.Match[str]],
) -> List[Tuple[int, int, str, int, int]]:
    """Locate bare or delimited two-letter acronym spans in source text.

    Returns tuples of ``(start_token_index, end_token_index_exclusive,
    compact_uppercase, char_start, char_end)``. Bare ``XO`` is one token.
    Hyphenated ``X-O`` is two single-letter tokens joined only by letter
    delimiters so ordinary prose is never treated as an acronym.
    """
    spans: List[Tuple[int, int, str, int, int]] = []
    index = 0
    while index < len(ref_matches):
        surface = ref_matches[index].group(0)
        if surface.isalpha() and surface.isupper() and len(surface) == 2:
            start, end = ref_matches[index].span()
            spans.append((index, index + 1, surface, start, end))
            index += 1
            continue
        if (
            index + 1 < len(ref_matches)
            and surface.isalpha()
            and surface.isupper()
            and len(surface) == 1
        ):
            next_surface = ref_matches[index + 1].group(0)
            if (
                next_surface.isalpha()
                and next_surface.isupper()
                and len(next_surface) == 1
            ):
                between = ref_text[ref_matches[index].end():ref_matches[index + 1].start()]
                # Only pure letter delimiters (X-O, X.O, X O) form one acronym.
                if re.fullmatch(r"[.\-\s]*", between or ""):
                    compact = f"{surface}{next_surface}"
                    start = ref_matches[index].start()
                    end = ref_matches[index + 1].end()
                    spans.append((index, index + 2, compact, start, end))
                    index += 2
                    continue
        index += 1
    return spans


def _raw_bare_two_letter_acronym_equivalence(
    ref_text: str,
    hyp_text: str,
) -> Tuple[Optional[Dict[str, Any]], str, str]:
    """Rewrite aligned two-letter acronyms when ASR spells letter names.

    Accepts source forms such as ``XO`` or delimited ``X-O`` against the
    compact ASR form ``EXO`` or the split letter-name form ``ex o`` when the
    surrounding raw context pins the match. It can coexist with other
    independent spelling substitutions. Three-letter acronyms like ``SOB``
    are not handled here.
    """
    ref_matches = list(_RAW_SPOKEN_TOKEN_RE.finditer(ref_text or ""))
    hyp_matches = list(_RAW_SPOKEN_TOKEN_RE.finditer(hyp_text or ""))
    if not ref_matches or not hyp_matches:
        return None, ref_text, hyp_text

    replacements: List[Tuple[Tuple[int, int], str, str, str]] = []
    for ref_start_i, ref_end_i, compact, span_start, span_end in _raw_two_letter_acronym_ref_spans(
        ref_text,
        ref_matches,
    ):
        expected = _render_bare_two_letter_acronym(compact)
        if not expected:
            continue
        ref_before = (
            _clean_context_word(ref_matches[ref_start_i - 1].group(0))
            if ref_start_i > 0 else ""
        )
        ref_after = (
            _clean_context_word(ref_matches[ref_end_i].group(0))
            if ref_end_i < len(ref_matches) else ""
        )
        candidates: List[Tuple[int, int, str]] = []
        for hyp_index, hyp_match in enumerate(hyp_matches):
            if _clean_context_word(hyp_match.group(0)) != expected:
                continue
            hyp_before = (
                _clean_context_word(hyp_matches[hyp_index - 1].group(0))
                if hyp_index > 0 else ""
            )
            hyp_after = (
                _clean_context_word(hyp_matches[hyp_index + 1].group(0))
                if hyp_index + 1 < len(hyp_matches) else ""
            )
            if (ref_before and hyp_before == ref_before) or (ref_after and hyp_after == ref_after):
                start, end = hyp_match.span()
                candidates.append((start, end, hyp_match.group(0)))
        # Split letter-name ASR such as "ex o" for source X-O / XO.
        for hyp_index in range(len(hyp_matches) - 1):
            first = _clean_context_word(hyp_matches[hyp_index].group(0))
            second = _clean_context_word(hyp_matches[hyp_index + 1].group(0))
            if f"{first}{second}" != expected:
                continue
            between = hyp_text[hyp_matches[hyp_index].end():hyp_matches[hyp_index + 1].start()]
            if not re.fullmatch(r"[.\-\s]*", between or ""):
                continue
            hyp_before = (
                _clean_context_word(hyp_matches[hyp_index - 1].group(0))
                if hyp_index > 0 else ""
            )
            hyp_after = (
                _clean_context_word(hyp_matches[hyp_index + 2].group(0))
                if hyp_index + 2 < len(hyp_matches) else ""
            )
            if (ref_before and hyp_before == ref_before) or (ref_after and hyp_after == ref_after):
                start = hyp_matches[hyp_index].start()
                end = hyp_matches[hyp_index + 1].end()
                candidates.append((start, end, hyp_text[start:end]))
        if len(candidates) == 1:
            hyp_start, hyp_end, hyp_surface = candidates[0]
            replacements.append(
                ((span_start, span_end), compact, hyp_surface, expected)
            )
            _ = hyp_start, hyp_end
    if not replacements:
        return None, ref_text, hyp_text

    rewritten_ref = ref_text
    for (start, end), _ref_surface, hyp_surface, _expected in reversed(replacements):
        rewritten_ref = f"{rewritten_ref[:start]}{hyp_surface}{rewritten_ref[end:]}"
    first_ref, first_hyp, first_rendered = replacements[0][1], replacements[0][2], replacements[0][3]
    return (
        {
            "kind": "bare_two_letter_acronym_surface_variation",
            "expected": first_ref,
            "hypothesis": first_hyp,
            "rendered": first_rendered,
            "count": len(replacements),
        },
        rewritten_ref,
        hyp_text,
    )


def _raw_consonant_acronym_equivalence(
    ref_text: str,
    hyp_text: str,
) -> Tuple[Optional[Dict[str, Any]], str, str]:
    """Align a short uppercase ASR acronym with its source word skeleton.

    Some ASR decoders emit an all-caps abbreviation for a spoken name or
    invented term, such as ``Devi`` → ``DV``. Accept only one isolated raw
    token difference when the hypothesis is a short uppercase token, its
    letters exactly equal the reference consonant skeleton, and phonetic
    evidence also supports the reduction. This keeps the rule structural
    rather than naming any particular book term.

    Args:
        ref_text: Source text before normalization.
        hyp_text: ASR text before normalization.

    Returns:
        Evidence plus rewritten reference/hypothesis text when safe; otherwise
        ``(None, ref_text, hyp_text)``.
    """
    difference = _raw_token_difference(ref_text, hyp_text)
    if difference is None:
        return None, ref_text, hyp_text
    ref_matches, hyp_matches, prefix, ref_end, hyp_end = difference
    ref_span = ref_matches[prefix:ref_end]
    hyp_span = hyp_matches[prefix:hyp_end]
    if len(ref_span) != 1 or len(hyp_span) != 1:
        return None, ref_text, hyp_text

    ref_surface = ref_span[0].group(0)
    hyp_surface = hyp_span[0].group(0)
    if (
        not ref_surface.isalpha()
        or not hyp_surface.isalpha()
        or not hyp_surface.isupper()
        or not 2 <= len(hyp_surface) <= 4
        or len(ref_surface) <= len(hyp_surface)
    ):
        return None, ref_text, hyp_text
    ref_clean = ref_surface.lower()
    hyp_clean = hyp_surface.lower()
    consonants = "".join(char for char in ref_clean if char not in "aeiou")
    if consonants != hyp_clean:
        return None, ref_text, hyp_text
    if not tokens_phonetically_equal(ref_clean, hyp_clean):
        return None, ref_text, hyp_text

    start, end = ref_span[0].span()
    return (
        {
            "kind": "consonant_acronym_surface_variation",
            "expected": ref_surface,
            "hypothesis": hyp_surface,
            "rendered": hyp_clean,
        },
        f"{ref_text[:start]}{hyp_surface}{ref_text[end:]}",
        hyp_text,
    )


def _raw_single_conjunction_omission_equivalence(
    ref_text: str,
    hyp_text: str,
) -> Tuple[Optional[Dict[str, Any]], str, str]:
    """Remove one safe omitted internal conjunction before comparison.

    ASR often does not write a softly spoken ``and`` between two ordinary
    content words.  Accept only that exact one-token difference when every
    other raw token matches.  This rule never accepts a leading/trailing word,
    a negation, an identifier, a number, or another conjunction.
    """
    difference = _raw_token_difference(ref_text, hyp_text)
    if difference is None:
        return None, ref_text, hyp_text
    ref_matches, hyp_matches, prefix, ref_end, hyp_end = difference
    ref_span = ref_matches[prefix:ref_end]
    hyp_span = hyp_matches[prefix:hyp_end]
    if len(ref_span) != 1 or hyp_span or prefix == 0 or ref_end >= len(ref_matches):
        return None, ref_text, hyp_text
    conjunction = _clean_context_word(ref_span[0].group(0))
    if conjunction != "and":
        return None, ref_text, hyp_text
    neighbors = [
        _clean_context_word(ref_matches[prefix - 1].group(0)),
        _clean_context_word(ref_matches[ref_end].group(0)),
    ]
    if any(
        not word
        or word in FUNCTION_WORDS
        or word in FILLER_WORDS
        or word in NEGATIONS
        or _looks_numeric_word(word)
        or _is_id_like_token(word)
        for word in neighbors
    ):
        return None, ref_text, hyp_text
    start, end = ref_span[0].span()
    return (
        {
            "kind": "single_internal_conjunction_omission",
            "expected": ref_span[0].group(0),
            "hypothesis": "",
        },
        f"{ref_text[:start]}{ref_text[end:]}",
        hyp_text,
    )


def _raw_split_word_number_equivalence(
    ref_text: str,
    hyp_text: str,
) -> Tuple[Optional[Dict[str, Any]], str, str]:
    """Accept one ordinary word split around a spoken digit rendering.

    TTS source text may deliberately spell one word as ``estim 8`` or
    ``estim-eight`` to control pronunciation. Accept that local rendering
    against ``estimate`` when one raw span contains a single ordinary token
    and the other contains exactly two tokens whose second token is a spoken
    digit rendering. This check runs on local token replacements rather than
    requiring the whole sentence to differ in only one place, so independent
    safe substitutions can coexist.
    """
    ref_matches = list(_RAW_SPOKEN_TOKEN_RE.finditer(ref_text or ""))
    hyp_matches = list(_RAW_SPOKEN_TOKEN_RE.finditer(hyp_text or ""))
    if not ref_matches or not hyp_matches:
        return None, ref_text, hyp_text

    ref_tokens = [match.group(0).lower().replace("\u2019", "'") for match in ref_matches]
    hyp_tokens = [match.group(0).lower().replace("\u2019", "'") for match in hyp_matches]
    candidates: List[Tuple[Dict[str, Any], str, str]] = []

    for opcode in SequenceMatcher(None, ref_tokens, hyp_tokens).get_opcodes():
        tag, ref_start_idx, ref_end_idx, hyp_start_idx, hyp_end_idx = opcode
        if tag == "equal":
            continue
        ref_span = ref_tokens[ref_start_idx:ref_end_idx]
        hyp_span = hyp_tokens[hyp_start_idx:hyp_end_idx]
        if {len(ref_span), len(hyp_span)} != {1, 2}:
            continue

        single_span = ref_span if len(ref_span) == 1 else hyp_span
        split_span = hyp_span if len(ref_span) == 1 else ref_span
        single = _clean_context_word(single_span[0])
        split_words = [_clean_context_word(token) for token in split_span]
        if not single or len(single) < 4 or not all(split_words):
            continue
        number_word = split_words[1]
        if not re.fullmatch(r"\d", number_word) and number_word not in _SPLIT_WORD_DIGIT_WORDS:
            continue
        prefix_word = split_words[0]
        if (
            len(prefix_word) < 3
            or prefix_word in FUNCTION_WORDS
            or prefix_word in NEGATIONS
            or single in FUNCTION_WORDS
            or single in NEGATIONS
            or _is_id_like_token(single)
            or _is_id_like_token(prefix_word)
            or _looks_numeric_word(single)
        ):
            continue
        if number_word.isdigit():
            number_word = num2words(int(number_word), lang="en")
        joined = f"{prefix_word}{number_word}"
        if not tokens_phonetically_equal(single, joined):
            continue

        ref_start = ref_matches[ref_start_idx].start()
        ref_stop = ref_matches[ref_end_idx - 1].end()
        hyp_start = hyp_matches[hyp_start_idx].start()
        hyp_stop = hyp_matches[hyp_end_idx - 1].end()
        ref_rewritten = (
            f"{ref_text[:ref_start]}{hyp_text[hyp_start:hyp_stop]}{ref_text[ref_stop:]}"
            if len(ref_span) == 1
            else ref_text
        )
        hyp_rewritten = (
            f"{hyp_text[:hyp_start]}{ref_text[ref_start:ref_stop]}{hyp_text[hyp_stop:]}"
            if len(hyp_span) == 1
            else hyp_text
        )
        candidates.append((
            {
                "kind": "split_word_number_phonetic_equivalence",
                "expected": ref_text[ref_start:ref_stop],
                "hypothesis": hyp_text[hyp_start:hyp_stop],
                "joined": joined,
            },
            ref_rewritten,
            hyp_rewritten,
        ))

    if len(candidates) != 1:
        return None, ref_text, hyp_text
    return candidates[0]


def _raw_albeit_resegmentation_equivalence(
    ref_text: str,
    hyp_text: str,
    strict_ref_tokens: Optional[Set[str]] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Preserve the guarded ``Albeit`` ↔ ``I'll be at`` raw compatibility rule.

    This is intentionally not a generic pre-alignment rewrite. Ordinary exact
    resegmentation must remain visible to dynamic programming as a boundary
    operation. ``Albeit`` has a genuinely different compact surface after
    contraction expansion, so its existing narrow compatibility behavior stays
    separately named and auditable.
    """
    difference = _raw_token_difference(ref_text, hyp_text)
    if difference is None:
        return None, hyp_text
    ref_matches, hyp_matches, prefix, ref_end, hyp_end = difference
    ref_tokens = [match.group(0).lower().replace("\u2019", "'") for match in ref_matches]
    hyp_tokens = [match.group(0).lower().replace("\u2019", "'") for match in hyp_matches]
    ref_span = ref_tokens[prefix:ref_end]
    hyp_span = hyp_tokens[prefix:hyp_end]
    if {len(ref_span), len(hyp_span)} != {1, 3}:
        return None, hyp_text

    ref_words = [_clean_context_word(token) for token in ref_span]
    hyp_words = [_clean_context_word(token) for token in hyp_span]
    if not all([*ref_words, *hyp_words]):
        return None, hyp_text
    if strict_ref_tokens and any(token in strict_ref_tokens for token in ref_words):
        return None, hyp_text
    if not (
        (tuple(ref_words), tuple(hyp_words)) == (("albeit",), ("ill", "be", "at"))
        or (tuple(ref_words), tuple(hyp_words)) == (("ill", "be", "at"), ("albeit",))
    ):
        return None, hyp_text

    start = hyp_matches[prefix].start()
    end = hyp_matches[hyp_end - 1].end()
    ref_start = ref_matches[prefix].start()
    ref_stop = ref_matches[ref_end - 1].end()
    rewritten_hyp = f"{hyp_text[:start]}{ref_text[ref_start:ref_stop]}{hyp_text[end:]}"
    return {
        "kind": "raw_phrase_resegmentation",
        "specialized_rule": "albeit",
        "expected": ref_text[ref_start:ref_stop],
        "hypothesis": hyp_text[start:end],
    }, rewritten_hyp


def _normalized_single_adjacent_stutter_equivalence(
    ref_normalized: str,
    hyp_normalized: str,
    phonetic_threshold: float,
    strict_ref_tokens: Optional[Set[str]] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Remove one harmless adjacent duplicate while preserving all other speech.

    This accepts one extra copy of an immediately neighboring ordinary word
    only when every other non-exact token pair is already a normal spoken
    equivalence. Numbers, identifiers, negations, phrase repeats, and two or
    more duplicate insertions remain visible to the main comparison.
    """
    ref_tokens = ref_normalized.split()
    hyp_tokens = hyp_normalized.split()
    if not ref_tokens or not hyp_tokens:
        return None, hyp_normalized

    stutters: List[Dict[str, str]] = []
    for tag, ref_start, ref_end, hyp_start, hyp_end in SequenceMatcher(
        None, ref_tokens, hyp_tokens
    ).get_opcodes():
        if tag == "equal":
            continue
        if tag == "insert" and hyp_end - hyp_start == 1:
            inserted = hyp_tokens[hyp_start]
            clean = _clean_context_word(inserted)
            left = ref_tokens[ref_start - 1] if ref_start else ""
            right = ref_tokens[ref_start] if ref_start < len(ref_tokens) else ""
            if (
                clean
                and clean in {_clean_context_word(left), _clean_context_word(right)}
                and not _is_placeholder_token(inserted)
                and not _is_id_like_token(inserted)
                and not _looks_numeric_word(clean)
                and not _contains_negation_text(clean)
            ):
                stutters.append({"expected": clean, "hypothesis": clean})
                continue
        if tag == "replace" and ref_end - ref_start == hyp_end - hyp_start:
            pairs = zip(ref_tokens[ref_start:ref_end], hyp_tokens[hyp_start:hyp_end])
            if all(
                spoken_token_equivalent(
                    ref_token,
                    hyp_token,
                    phonetic_threshold,
                    strict_ref_tokens=strict_ref_tokens,
                )
                != "none"
                for ref_token, hyp_token in pairs
            ):
                continue
        return None, hyp_normalized

    if len(stutters) != 1:
        return None, hyp_normalized
    return {
        "kind": "single_adjacent_word_stutter",
        **stutters[0],
    }, ref_normalized


def _expand_contractions(text: str) -> str:
    """Expand common English contractions using boundary-aware replacement."""
    text = _APOSTROPHE_RE.sub("'", text)
    # ASR and source cleanup can remove apostrophes from otherwise ordinary
    # contractions. Normalize those spellings before expanding their meaning.
    text = _NONSTANDARD_WASNT_RE.sub("wasn't", text)
    text = _YKNOW_RE.sub("you know", text)
    text = _D_CONTRACTION_WITH_FOLLOWING_RE.sub(
        _expand_d_contraction_with_context, text
    )

    def phonetic(m: re.Match) -> str:
        """Map informal spoken contractions to expanded phrases."""
        return _PHONETIC_MAP.get(m.group(1).lower(), m.group(0).lower())

    text = _PHONETIC_CONTRACTION_RE.sub(phonetic, text)
    text = _CONTRACTION_RE.sub(_expand_contraction_match, text)
    return text


def _roman_to_int_token(token: str) -> Optional[int]:
    """Convert multi-character Roman numerals only; never convert pronoun 'i'."""
    lower = token.lower()
    if lower == "i":
        return None
    if len(lower) == 1:
        return None
    if lower in ROMAN_BLOCKLIST:
        return None
    if lower in ROMAN_TO_INT:
        return ROMAN_TO_INT[lower]
    if len(lower) >= 2 and _ROMAN_MULTI_RE.match(lower):
        total = 0
        prev = 0
        values = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
        for ch in reversed(lower):
            val = values[ch]
            if val < prev:
                total -= val
            else:
                total += val
                prev = val
        return total if total > 0 else None
    return None


def _replace_uppercase_roman_numerals(text: str) -> str:
    """Convert explicit uppercase Roman numerals before numeric slot detection."""
    def has_numeric_context(start: int) -> bool:
        """Return whether nearby prose explicitly frames a Roman numeral."""
        before = re.findall(r"[A-Za-z]+", text[:start].lower())
        after = re.findall(r"[A-Za-z]+", text[start:].lower())
        nearby = set(before[-2:]) | set(after[:2])
        return bool(nearby & _ROMAN_CONTEXT_WORDS)

    def replace(match: re.Match) -> str:
        """Keep non-Roman candidates intact while converting valid numeral tokens."""
        candidate = match.group(1)
        # Two-letter additive Roman strings are frequently uppercase acronyms
        # (for example, ``DV``). Require explicit enumeration context before
        # converting them; subtractive forms such as ``IV`` remain supported.
        if (
            len(candidate) == 2
            and candidate.lower() not in _ROMAN_SUBTRACTIVE_PAIRS
            and not has_numeric_context(match.start(1))
        ):
            return candidate
        value = _roman_to_int_token(candidate)
        return str(value) if value is not None else candidate

    return _UPPER_ROMAN_TOKEN_RE.sub(replace, text)


def _canonicalize_dialogue_surface(text: str) -> str:
    """Strip dialogue quotes and unify quote glyphs before ID/number classification.

    Chunk text often keeps an unmatched opening quote (``\"No one is…``). That
    glyph changes token shapes enough for typed-ID detection to take a different
    path than clean ASR output, inventing false slot asymmetry. Surface canon
    must run on both reference and hypothesis before any slotting.
    """
    if not text:
        return ""
    text = (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    text = text.strip()
    # Unmatched lead/trail dialogue quotes are not spoken content.
    while text and text[0] in "\"'":
        text = text[1:].lstrip()
    while text and text[-1] in "\"'":
        text = text[:-1].rstrip()
    return text


def _lock_prose_indefinite_compounds(text: str) -> str:
    """Fuse indefinite compounds so number/ID passes cannot split them.

    Closed-class English: ``no one``, ``anyone``, ``everyone``, ``someone`` and
    their spaced/hyphen ASR splits. Fusing to a single alphanumeric token keeps
    ``one`` out of numeric and typed-ID span detection while preserving prose.
    """
    if not text:
        return ""
    # Spaced / hyphenated forms first.
    text = re.sub(
        r"\b(no|any|every|some)[-\s]+one\b",
        lambda match: f"{match.group(1).lower()}one",
        text,
        flags=re.IGNORECASE,
    )
    # Already-closed forms stay one token (normalize case for stability).
    text = re.sub(
        r"\b(anyone|everyone|someone|noone)\b",
        lambda match: match.group(1).lower(),
        text,
        flags=re.IGNORECASE,
    )
    return text


def _lock_number_bearing_prose_compounds(text: str) -> str:
    """Fuse closed compounds so internal number-words are not numeric slots.

    ASR often writes ``one-time`` / ``first hand`` while the book has
    ``onetime`` / ``firsthand``. Without a lock, ``one`` and ``first`` become
    ``<NUM>`` / ``<ID*>`` on one side only and invent a hard fail for the same
    spoken compound. Run this before number span replacement.
    """
    if not text:
        return ""
    # one-time / one time / onetime → one solid prose token
    text = re.sub(r"\bone[-\s]+time\b", "onetime", text, flags=re.IGNORECASE)
    text = re.sub(r"\bonetime\b", "onetime", text, flags=re.IGNORECASE)
    # first-hand / first hand / firsthand (ordinal "first" must not slot)
    text = re.sub(r"\bfirst[-\s]+hand\b", "firsthand", text, flags=re.IGNORECASE)
    text = re.sub(r"\bfirsthand\b", "firsthand", text, flags=re.IGNORECASE)
    # second-hand / second hand / secondhand
    text = re.sub(r"\bsecond[-\s]+hand\b", "secondhand", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsecondhand\b", "secondhand", text, flags=re.IGNORECASE)
    # none the less / nonetheless (spacing-only ASR variants)
    text = re.sub(
        r"\bnone[-\s]+the[-\s]+less\b",
        "nonetheless",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bnonetheless\b", "nonetheless", text, flags=re.IGNORECASE)
    return text


def _canonicalize_number_abbreviations(text: str) -> str:
    """Map spoken number abbreviations onto the full word ``number``.

    ASR often writes ``No. 1`` / ``No 1`` for book ``Number One``. Leaving
    bare ``no`` triggers negation protection and hard-fails a pure orthography
    difference. Rewrite only when ``no`` is clearly the abbreviation before a
    numeral (``No. 1``, ``no 12``), not bare dialogue ``no``.
    """
    if not text:
        return ""
    # No.1 / No. 1 / no. 12
    text = re.sub(
        r"\bno\.\s*(\d+)\b",
        r"number \1",
        text,
        flags=re.IGNORECASE,
    )
    # No 1 / no 12 (space, no period) — still abbreviation before digits
    text = re.sub(
        r"\bno\s+(\d+)\b",
        r"number \1",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _join_spaced_dotted_letter_initialisms(text: str) -> str:
    """Join leftover dotted letter initialisms after typed craft-ID detection.

    Runs after punctuated craft IDs (``M. H. Tron``) are already slotted so those
    forms are not flattened. Handles geo/org ASR spellings with spaces such as
    ``D. C.`` and ``N. I. S.`` before Roman conversion and number slotting.
    """
    if not text:
        return ""

    def join_match(match: re.Match) -> str:
        """Collapse one dotted letter run into an uppercase acronym token."""
        letters = re.findall(r"[A-Za-z]", match.group(0))
        return "".join(letters).upper()

    # Require a period after each two-letter initial.  A bare final initial is
    # safe only in a three-or-more-letter run; otherwise ``couldn't. I`` can
    # be misread as the false initialism ``t. I`` before contractions expand.
    text = re.sub(
        r"(?<!['\u2019])(?<![A-Za-z0-9])"
        r"(?:"
        r"(?:[A-Za-z]\s*\.\s*){2,}[A-Za-z]"
        r"|(?:[A-Za-z]\s*\.\s*){2,}"
        r")"
        r"(?![A-Za-z0-9])",
        join_match,
        text,
    )
    return text


def _separate_delimited_letter_sequences(text: str) -> str:
    """Join explicitly delimited letters into one comparison token.

    Source prose can spell an initialism as ``S-O-B`` while ASR writes
    ``S.O.B.`` or ``SOB``. All forms represent the same spoken letter
    sequence, so their delimiters are removed for comparison. Ordinary words
    such as ``x-ray`` do not match this complete-letter-sequence pattern.
    Same-letter stutters such as ``I-I`` stay unjoined so dash splitting and
    repeat-collapse can accept ASR single ``I``.
    """
    if not text:
        return ""

    def separate(match: re.Match) -> str:
        """Remove internal delimiters from one explicit letter sequence."""
        surface = match.group(0)
        letters = re.sub(r"[.-]+", "", surface)
        # I-I / A-A is dialog stutter, not a two-letter acronym.
        if len(letters) == 2 and letters[0].lower() == letters[1].lower():
            return surface
        return letters

    return _DELIMITED_LETTER_SEQUENCE_RE.sub(separate, text)


def _soundex(text: str) -> str:
    """Create a compact Soundex key for secondary phonetic comparisons."""
    letters = re.sub(r"[^a-z]", "", text.lower())
    if not letters:
        return ""
    codes = {
        **dict.fromkeys("bfpv", "1"),
        **dict.fromkeys("cgjkqsxz", "2"),
        **dict.fromkeys("dt", "3"),
        "l": "4",
        **dict.fromkeys("mn", "5"),
        "r": "6",
    }
    first = letters[0].upper()
    encoded: List[str] = []
    previous = codes.get(letters[0], "")
    for letter in letters[1:]:
        code = codes.get(letter, "")
        if code and code != previous:
            encoded.append(code)
        previous = code
    return (first + "".join(encoded) + "000")[:4]


def _double_metaphone_primary(text: str) -> str:
    """
    Lightweight primary metaphone-like code for phonetic equivalence checks.

    Not a full Double Metaphone; used with spelling similarity, not alone.
    """
    s = re.sub(r"[^a-z]", "", text.lower())
    if not s:
        return ""
    # Common letter-group folds
    s = s.replace("gne", "ne").replace("kn", "n").replace("gn", "n")
    s = s.replace("ph", "f").replace("q", "k").replace("x", "ks")
    s = s.replace("ck", "k").replace("ch", "x").replace("sh", "x")
    s = s.replace("th", "0").replace("wh", "w").replace("wr", "r")
    # Drop vowels after first char
    if not s:
        return ""
    out = [s[0]]
    for ch in s[1:]:
        if ch in "aeiouy":
            continue
        if ch != out[-1]:
            out.append(ch)
    return "".join(out)[:8]


def _phonetic_skeleton(text: str) -> str:
    """Collapse text to a consonant skeleton for loose spoken-sound matching."""
    s = re.sub(r"[^a-z]", "", text.lower())
    if not s:
        return ""
    s = s.replace("kn", "n").replace("gn", "n").replace("wr", "r").replace("wh", "w")
    s = s.replace("ph", "f").replace("qu", "kw").replace("ck", "k").replace("x", "ks")
    # These are articulatory folds, not word-specific aliases. They let ASR
    # spelling variants retain the consonant shape a listener would hear.
    s = s.replace("th", "z").replace("ng", "n")
    s = re.sub(r"(?:mb|m|n)$", "n", s)
    s = re.sub(r"[aeiou]", "", s)
    s = s.replace("z", "s").replace("w", "v")
    s = re.sub(r"(.)\1+", r"\1", s)
    return s


def _short_terminal_voicing_variant(a: str, b: str) -> bool:
    """Recognize short same-initial words differing only by terminal s/z voicing."""
    if not (2 <= len(a) <= 4 and 2 <= len(b) <= 4):
        return False
    if a[0] != b[0]:
        return False
    return {a[-1], b[-1]} == {"s", "z"}


def _same_initial_sound(a: str, b: str) -> bool:
    """Return True for matching initials or the common v/w sound interchange."""
    if not a or not b:
        return False
    if a[0] == b[0]:
        return True
    return {a[0], b[0]} == {"v", "w"}


def _spoken_letter_alias(token: str) -> str:
    """Map single letters and common letter-name spellings to one alias."""
    clean = re.sub(r"[^a-z]", "", token.lower())
    if not clean:
        return ""
    if len(clean) == 1:
        return clean
    if clean in _LETTER_NAME_TO_LETTER:
        return _LETTER_NAME_TO_LETTER[clean]
    return clean


def _contains_negation_token(tokens: Sequence[str]) -> bool:
    """Return True when any token is a negation word we must not merge away."""
    for token in tokens:
        clean = re.sub(r"[^a-z]", "", token.lower())
        if clean in NEGATIONS or clean.endswith("nt") and clean in {"dont", "doesnt", "didnt", "wont", "isnt", "wasnt", "werent"}:
            return True
    return False


def tokens_phonetically_equal(a: str, b: str, threshold: float = 0.85) -> bool:
    """Return True when two tokens likely represent the same spoken sound."""
    if a == b:
        return True
    if not a or not b:
        return False
    if any(char.isspace() for char in a) or any(char.isspace() for char in b):
        return False
    # Protected tokens carry meaning that pronunciation tolerance must not hide.
    if (
        _is_placeholder_token(a)
        or _is_placeholder_token(b)
        or _is_id_like_token(a)
        or _is_id_like_token(b)
        or _contains_negation_text(a)
        or _contains_negation_text(b)
    ):
        return False
    # Distinct short homograph spellings that must NOT match each other.
    if frozenset({a, b}) in _BLOCKED_PHONETIC_PAIRS:
        return False

    alias_a = _spoken_letter_alias(a)
    alias_b = _spoken_letter_alias(b)
    if alias_a and alias_a == alias_b:
        return True

    meta_a = _double_metaphone_primary(a)
    meta_b = _double_metaphone_primary(b)
    if meta_a and meta_a == meta_b and len(meta_a) >= 2:
        # Require supporting character evidence so Soundex-alone is never enough
        if fuzz.ratio(a, b) >= 50 or abs(len(a) - len(b)) <= 2:
            return True

    # Spelling closeness for near-homophones / invented names
    ratio = fuzz.ratio(a, b) / 100.0
    if ratio >= threshold:
        return True

    skeleton_a = _phonetic_skeleton(a)
    skeleton_b = _phonetic_skeleton(b)
    if skeleton_a and skeleton_b:
        if skeleton_a == skeleton_b and (
            fuzz.ratio(a, b) >= 50
            or _short_terminal_voicing_variant(a, b)
            or _same_initial_sound(a, b)
        ):
            return True
        if (
            fuzz.ratio(a, b) >= 60
            and fuzz.ratio(skeleton_a, skeleton_b) >= 50
            and _same_initial_sound(a, b)
        ):
            return True

    # Allow multi-syllable invented names when metaphone prefixes match + high partial
    if len(a) >= 5 and len(b) >= 5:
        if meta_a[:3] == meta_b[:3] and fuzz.partial_ratio(a, b) >= 80:
            return True
    return False


def spoken_token_equivalent(
    ref: str,
    hyp: str,
    phonetic_threshold: float = 0.85,
    strict_ref_tokens: Optional[Set[str]] = None,
) -> str:
    """
    Classify token pair equivalence.

    Returns: exact | normalized | homograph | phonetic | none.

    ``strict_ref_tokens`` protects one-off short all-caps source acronyms from
    the general same-space tolerance. Recurring acronyms are supplied through
    book-term evidence and therefore are not in this set.
    """
    if ref == hyp:
        return "exact"
    if _is_placeholder_token(ref) and _is_placeholder_token(hyp):
        return "normalized"
    # Spelling variant table (safe pairs only)
    ra = SPELLING_VARIANTS.get(ref, ref)
    ha = SPELLING_VARIANTS.get(hyp, hyp)
    if ra == ha:
        return "normalized"
    if ra.replace(" ", "") == ha.replace(" ", ""):
        return "normalized"
    # Dialect / reduced pronouns: 'e → he, em → them (em also in SPELLING_VARIANTS).
    ref_c = _clean_context_word(ra)
    hyp_c = _clean_context_word(ha)
    if {ref_c, hyp_c} in {
        frozenset({"e", "he"}),
        frozenset({"em", "them"}),
        frozenset({"em", "him"}),
        frozenset({"aye", "i"}),
        frozenset({"ay", "i"}),
    }:
        return "normalized"
    # Light morphology: begged↔beg, guards↔guard (shell still checked by align).
    if _simple_morphology_equivalent(ref_c, hyp_c):
        return "normalized"
    if {ra, ha} in {
        frozenset({"i would", "i"}),
        frozenset({"i had", "i"}),
        frozenset({"i am", "i"}),
    }:
        return "normalized"
    ref_clean = _clean_context_word(ra)
    if strict_ref_tokens and ref_clean in strict_ref_tokens:
        return "none"
    hyp_clean = _clean_context_word(ha)
    homograph_pair = _homograph_pair_key((ref_clean,), (hyp_clean,))
    if (
        homograph_pair is not None
        and homograph_pair in _HOMOGRAPH_WHITELIST_PAIRS
        and not _is_placeholder_token(ref)
        and not _is_placeholder_token(hyp)
        and not _is_id_like_token(ref)
        and not _is_id_like_token(hyp)
        and not _looks_numeric_word(ref_clean)
        and not _looks_numeric_word(hyp_clean)
        and not _contains_negation_text(ra)
        and not _contains_negation_text(ha)
    ):
        return "homograph"
    if (
        frozenset({ref_clean, hyp_clean}) in _SPOKEN_LETTER_HOMOPHONE_PAIRS
        and not _is_id_like_token(ref)
        and not _is_id_like_token(hyp)
        and not _looks_numeric_word(ref_clean)
        and not _looks_numeric_word(hyp_clean)
    ):
        return "phonetic"
    if (
        frozenset({ref_clean, hyp_clean}) in _SPOKEN_WORD_HOMOPHONE_PAIRS
        and not _is_id_like_token(ref)
        and not _is_id_like_token(hyp)
        and not _looks_numeric_word(ref_clean)
        and not _looks_numeric_word(hyp_clean)
    ):
        return "phonetic"
    if ref_clean != hyp_clean and hyp_clean in PERSONAL_PRONOUNS:
        return "none"
    if _protected_short_word_ambiguity(ra, ha):
        return "ambiguous"
    if _contains_negation_text(ra) != _contains_negation_text(ha):
        return "none"
    # Phonetic near-miss: allow a wider length band only for longer content
    # words (names/homophones). Short stems such as ``cap``/``captain`` stay
    # outside this band so partial truncations do not soft-pass.
    _min_len = min(len(ref_clean), len(hyp_clean)) if ref_clean and hyp_clean else 0
    _len_delta = abs(len(ref_clean) - len(hyp_clean)) if ref_clean and hyp_clean else 99
    _max_delta = 2 if _min_len < 5 else 4
    if (
        ref_clean
        and hyp_clean
        and _min_len >= 3
        and _len_delta <= _max_delta
        and tokens_phonetically_equal(ref, hyp, phonetic_threshold)
    ):
        return "phonetic"
    # Longer content words: ASR name/homophone noise (Karus/Harris, century/sentry)
    # is acceptable quality variance when lengths are similar. Ratio bar is low
    # because ASR often invents a different spelling of the same heard name.
    if (
        ref_clean
        and hyp_clean
        and _min_len >= 5
        and _len_delta <= 4
        and ref_clean not in FUNCTION_WORDS
        and hyp_clean not in FUNCTION_WORDS
        and ref_clean not in NEGATIONS
        and hyp_clean not in NEGATIONS
        and not _looks_numeric_word(ref_clean)
        and not _looks_numeric_word(hyp_clean)
        and fuzz.ratio(ref_clean, hyp_clean) >= 50
    ):
        return "phonetic"
    # Multiword phrase vs joined form (going to vs goingto already handled)
    if fuzz.ratio(ref, hyp) >= 90:
        return "normalized"
    return "none"


def _aligned_single_token_spelling_equivalent(
    ref: str,
    hyp: str,
    phonetic_threshold: float = 0.85,
    strict_ref_tokens: Optional[Set[str]] = None,
) -> bool:
    """Accept a clean same-space one-token ASR spelling difference.

    ASR spelling is not evidence that the generated audio is wrong. A local
    ordinary token may differ by two letters when no protected material is
    involved. Sentence-level alignment still rejects real omissions, added
    speech, repetition, truncation, IDs, and numeric slots.
    """
    if not ref or not hyp:
        return False
    if any(char.isspace() for char in ref) or any(char.isspace() for char in hyp):
        return False

    ref_clean = _clean_context_word(ref)
    hyp_clean = _clean_context_word(hyp)
    if not ref_clean or not hyp_clean:
        return False
    # Structural conjunction/range words are not spelling variants. Protect
    # them here because this late same-space pass otherwise turns ``to`` ↔
    # ``and`` into a phonetic match after the numeric shapes were separated.
    if ref_clean != hyp_clean and {ref_clean, hyp_clean} <= {"and", "or", "to"}:
        return False
    if (
        _is_placeholder_token(ref)
        or _is_placeholder_token(hyp)
        or _is_id_like_token(ref)
        or _is_id_like_token(hyp)
        or _contains_negation_text(ref)
        or _contains_negation_text(hyp)
        or _contains_negation_text(ref_clean)
        or _contains_negation_text(hyp_clean)
        or _looks_numeric_word(ref_clean)
        or _looks_numeric_word(hyp_clean)
        or (strict_ref_tokens and ref_clean in strict_ref_tokens)
        or (ref_clean != hyp_clean and hyp_clean in PERSONAL_PRONOUNS)
    ):
        return False
    # Fillers only match other fillers (uh/oh/o), never random short words (on).
    if ref_clean in FILLER_WORDS or hyp_clean in FILLER_WORDS:
        return ref_clean in FILLER_WORDS and hyp_clean in FILLER_WORDS
    # Same-space spelling: short tokens stay tight; longer name-like tokens
    # may differ by a few letters (ASR confusable names). Truncations such as
    # ``captain``→``cap`` stay rejected via the length band. Unrelated long
    # words (``expected``/``wrong``) fail the ratio floor.
    _min_len = min(len(ref_clean), len(hyp_clean))
    _max_delta = 2 if _min_len < 5 else 4
    if abs(len(ref_clean) - len(hyp_clean)) > _max_delta:
        return False
    if _min_len < 2:
        return False
    if _min_len >= 5 and fuzz.ratio(ref_clean, hyp_clean) < 50:
        return False

    _ = phonetic_threshold
    return True


_SPECIAL_EXACT_BOUNDARY_RESEGMENTATIONS = frozenset({
    (("a", "round"), ("around",)),
    (("in", "to"), ("into",)),
})
_GUARDED_PHONETIC_BOUNDARY_PREFIXES = frozenset({
    "a", "an", "the", "my", "your", "his", "her", "our", "their", "for",
})
_GUARDED_TITLE_NAME_PREFIXES = frozenset({
    "captain", "commander", "doctor", "dr", "herr", "madam", "miss", "mister",
    "monsieur", "mr", "mrs", "ms", "professor", "sir",
})


def _special_exact_boundary_resegmentation_mode(
    ref_words: Sequence[str],
    hyp_words: Sequence[str],
) -> Optional[str]:
    """Return explicit compatibility evidence for retained exact boundary forms."""
    pair = (tuple(ref_words), tuple(hyp_words))
    reverse_pair = (pair[1], pair[0])
    if pair in _SPECIAL_EXACT_BOUNDARY_RESEGMENTATIONS or reverse_pair in _SPECIAL_EXACT_BOUNDARY_RESEGMENTATIONS:
        return "special_exact_surface"
    return None


def _guarded_function_boundary_phonetic_mode(
    ref_words: Sequence[str],
    hyp_words: Sequence[str],
    phonetic_threshold: float = 0.85,
) -> Optional[str]:
    """Retain narrow name-like function-prefix splits without generic fuzzy joins.

    This preserves legacy cases such as ``Maiwand`` ↔ ``my wand`` and
    ``for Carthum`` ↔ ``Forcatham``. Both spans must contain the same bounded
    one-to-two shape, the split form must begin with a named function prefix,
    and the joined forms must satisfy the existing strict phonetic comparator.
    Negation never enters this fuzzy rule; exact compact resegmentation handles
    preserved negation content separately.
    """
    if {len(ref_words), len(hyp_words)} != {1, 2}:
        return None
    if any(word in NEGATIONS for word in [*ref_words, *hyp_words]):
        return None
    split_words = ref_words if len(ref_words) == 2 else hyp_words
    single_word = hyp_words[0] if len(hyp_words) == 1 else ref_words[0]
    if (
        split_words[0] not in _GUARDED_PHONETIC_BOUNDARY_PREFIXES
        or not _is_content_word(split_words[1])
        or len(single_word) < 4
        or len(split_words[1]) < 2
    ):
        return None
    joined_split = "".join(split_words)
    if tokens_phonetically_equal(single_word, joined_split, phonetic_threshold):
        return "guarded_function_phonetic"
    return None


def _guarded_title_name_phonetic_mode(
    ref_words: Sequence[str],
    hyp_words: Sequence[str],
    strict_ref_tokens: Optional[Set[str]] = None,
    phonetic_threshold: float = 0.85,
) -> Optional[str]:
    """Retain title-plus-name fusion without reopening generic content fuzziness.

    A title, such as ``Herr``, followed by a name is a recognizable spoken
    structure. Its fused ASR rendering is allowed only when the joined form
    passes strict phonetic comparison. Ordinary content pairs stay excluded so
    ``Black goop`` cannot be accepted as ``Blackcoop``.
    """
    if {len(ref_words), len(hyp_words)} != {1, 2}:
        return None
    if any(word in NEGATIONS for word in [*ref_words, *hyp_words]):
        return None
    split_words = ref_words if len(ref_words) == 2 else hyp_words
    single_word = hyp_words[0] if len(hyp_words) == 1 else ref_words[0]
    if (
        split_words[0] not in _GUARDED_TITLE_NAME_PREFIXES
        or not _is_content_word(split_words[1])
        or len(split_words[1]) < 3
        or len(single_word) < 5
        or (strict_ref_tokens and any(word in strict_ref_tokens for word in ref_words))
    ):
        return None
    if tokens_phonetically_equal(single_word, "".join(split_words), phonetic_threshold):
        return "guarded_title_phonetic"
    return None


def _guarded_reduced_auxiliary_you_mode(
    ref_words: Sequence[str],
    hyp_words: Sequence[str],
) -> Optional[str]:
    """Match reduced ``d'you`` speech with ASR's expanded ``do you`` form.

    This is a contraction-pronunciation rule, not generic fuzzy word fusion.
    It remains bounded to the normalized ``dyou`` surface and the exact
    auxiliary-plus-pronoun expansion so unrelated content words cannot enter.
    """
    if (tuple(ref_words), tuple(hyp_words)) in {
        (("dyou",), ("do", "you")),
        (("do", "you"), ("dyou",)),
    }:
        return "guarded_reduced_auxiliary"
    return None


def _boundary_resegmentation_mode(
    ref_span: Sequence[str],
    hyp_span: Sequence[str],
    strict_ref_tokens: Optional[Set[str]] = None,
) -> Optional[str]:
    """Return match mode for a bounded prose word-boundary resegmentation.

    Word boundaries are not spoken content, so one token may match two or three
    adjacent tokens when their compact letter surfaces are identical. This is
    transcript-equivalence policy, not a claim that the alternatives share
    spelling or semantic intent. Generic fuzzy resegmentation is prohibited;
    only a separately guarded legacy function-prefix phonetic rule remains.
    Protected numeric, identifier, and acronym material stays on specialized
    comparison paths.

    Args:
        ref_span: Consecutive normalized reference tokens.
        hyp_span: Consecutive normalized hypothesis tokens.
        strict_ref_tokens: Source acronym tokens that must not be resegmented.

    Returns:
        Exact or named guarded match evidence, otherwise ``None``.
    """
    if len(ref_span) == len(hyp_span) or not ref_span or not hyp_span:
        return None
    if {len(ref_span), len(hyp_span)} not in ({1, 2}, {1, 3}):
        return None

    ref_words = [_clean_context_word(token) for token in ref_span]
    hyp_words = [_clean_context_word(token) for token in hyp_span]
    if not all(ref_words) or not all(hyp_words):
        return None
    if strict_ref_tokens and any(token in strict_ref_tokens for token in ref_words):
        return None

    protected_tokens = [*ref_span, *hyp_span]
    if any(
        _is_placeholder_token(token)
        or _is_id_like_token(token)
        or _looks_numeric_word(_clean_context_word(token))
        for token in protected_tokens
    ):
        return None

    ref_joined = "".join(ref_words)
    hyp_joined = "".join(hyp_words)
    homograph_pair = _homograph_pair_key(ref_words, hyp_words)
    if (
        homograph_pair is not None
        and homograph_pair in _HOMOGRAPH_WHITELIST_PAIRS
        and not any(word in NEGATIONS for word in [*ref_words, *hyp_words])
    ):
        return "homograph_whitelist"
    if ref_joined == hyp_joined:
        return _special_exact_boundary_resegmentation_mode(ref_words, hyp_words) or "exact_surface"
    reduced_auxiliary_mode = _guarded_reduced_auxiliary_you_mode(ref_words, hyp_words)
    if reduced_auxiliary_mode is not None:
        return reduced_auxiliary_mode
    return _guarded_function_boundary_phonetic_mode(ref_words, hyp_words)


def _local_joined_span_equivalent(
    ref_span: Sequence[str],
    hyp_span: Sequence[str],
    strict_ref_tokens: Optional[Set[str]] = None,
) -> bool:
    """Return whether local prose spans differ only by word boundaries."""
    return _boundary_resegmentation_mode(
        ref_span,
        hyp_span,
        strict_ref_tokens=strict_ref_tokens,
    ) is not None


def _expected_repeat_collapse_equivalent(
    ref_span: Sequence[str],
    hyp_token: str,
    strict_ref_tokens: Optional[Set[str]] = None,
) -> bool:
    """Accept one ASR rendering of two expected adjacent repeated words."""
    if len(ref_span) != 2:
        return False
    ref_words = [_clean_context_word(token) for token in ref_span]
    hyp_word = _clean_context_word(hyp_token)
    if not hyp_word or ref_words[0] != ref_words[1]:
        return False
    if strict_ref_tokens and ref_words[0] in strict_ref_tokens:
        return False
    if (
        ref_words[0] in NEGATIONS
        or hyp_word in NEGATIONS
        or _is_placeholder_token(hyp_token)
        or _is_id_like_token(hyp_token)
        or _looks_numeric_word(ref_words[0])
        or _looks_numeric_word(hyp_word)
    ):
        return False
    return min(len(ref_words[0]), len(hyp_word)) >= 1 and abs(
        len(ref_words[0]) - len(hyp_word)
    ) <= 2


def _modal_base_to_past_equivalent(
    ref_span: Sequence[str],
    hyp_token: str,
) -> bool:
    """Accept would/will + base verb against a simple past ASR rendering.

    Handles cases such as ``would work`` → ``worked`` where ASR compresses the
    modal and past-tenses the verb. Only regular -ed style past forms qualify.
    """
    if len(ref_span) != 2:
        return False
    modal = _clean_context_word(ref_span[0])
    base = _clean_context_word(ref_span[1])
    past = _clean_context_word(hyp_token)
    if modal not in {"will", "would"} or not base or not past:
        return False
    if (
        base in NEGATIONS
        or past in NEGATIONS
        or _is_placeholder_token(hyp_token)
        or _is_id_like_token(hyp_token)
        or _looks_numeric_word(base)
        or _looks_numeric_word(past)
    ):
        return False
    if past == f"{base}ed" or past == f"{base}d":
        return True
    if base.endswith("e") and past == f"{base}d":
        return True
    if base.endswith("y") and len(base) > 1 and past == f"{base[:-1]}ied":
        return True
    if (
        len(base) > 2
        and base[-1] not in "aeiou"
        and base[-2] in "aeiou"
        and past == f"{base}{base[-1]}ed"
    ):
        return True
    return False


def _trailing_name_fusion_equivalent(
    ref_span: Sequence[str],
    hyp_token: str,
    strict_ref_tokens: Optional[Set[str]] = None,
) -> bool:
    """Accept a two-word title/name span fused into one ASR token.

    The ASR token must retain the complete trailing source name. This prevents
    unrelated two-word prose from being absorbed while allowing forms such as
    ``Commander Nord`` rendered as ``Kwardenord``.
    """
    if len(ref_span) != 2:
        return False
    ref_words = [_clean_context_word(token) for token in ref_span]
    hyp_word = _clean_context_word(hyp_token)
    if not hyp_word or not all(ref_words):
        return False
    if strict_ref_tokens and any(token in strict_ref_tokens for token in ref_words):
        return False
    if any(token in FUNCTION_WORDS | NEGATIONS for token in ref_words):
        return False
    if any(
        _is_placeholder_token(token)
        or _is_id_like_token(token)
        or _looks_numeric_word(_clean_context_word(token))
        for token in [*ref_span, hyp_token]
    ):
        return False
    ref_joined = "".join(ref_words)
    return (
        len(ref_joined) >= 8
        and len(hyp_word) >= 5
        and min(len(ref_joined), len(hyp_word)) / max(len(ref_joined), len(hyp_word)) >= 0.65
        and hyp_word.endswith(ref_words[-1])
    )


def _accept_local_same_space_substitutions(
    alignment: Dict[str, Any],
    strict_ref_tokens: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Convert stable one-to-one spelling substitutions into neutral matches.

    This runs only after baseline alignment and only for sentences with no
    unpaired source or hypothesis tokens. It therefore cannot slide a local
    spelling tolerance across neighboring prose or hide an actual omission.
    """
    operations = list(alignment.get("operations") or [])
    if (
        alignment.get("extra_tokens")
        or alignment.get("missing_tokens")
        or not operations
    ):
        return alignment
    updated_operations: List[Dict[str, Any]] = []
    changed = False
    for operation in operations:
        updated = dict(operation)
        if operation.get("op") == "substitution" and _aligned_single_token_spelling_equivalent(
            str(operation.get("ref") or ""),
            str(operation.get("hyp") or ""),
            strict_ref_tokens=strict_ref_tokens,
        ):
            updated["op"] = "phonetic_equivalent"
            changed = True
        updated_operations.append(updated)
    if not changed:
        return alignment

    matching_ops = {
        "exact_match",
        "normalized_equivalent",
        "phonetic_equivalent",
        "ambiguous_equivalent",
        "boundary_resegmentation",
        "identifier_wildcard",
    }
    ref_count = int(alignment.get("reference_token_count") or 0)
    matched = sum(1 for operation in updated_operations if operation["op"] in matching_ops)
    covered_refs = sum(
        len(str(operation.get("ref") or "").split())
        for operation in updated_operations
        if operation["op"] in matching_ops
    )
    score_sum = 0.0
    for operation in updated_operations:
        if operation["op"] in {
            "exact_match",
            "normalized_equivalent",
            "phonetic_equivalent",
            "boundary_resegmentation",
        }:
            score_sum += len(str(operation.get("ref") or "").split())
        elif operation["op"] in {"ambiguous_equivalent", "identifier_wildcard"}:
            score_sum += 1.0
    updated_alignment = dict(alignment)
    updated_alignment.update({
        "operations": updated_operations,
        "matched_reference_tokens": matched,
        "substitutions": [],
        "coverage_score": covered_refs / ref_count if ref_count else 1.0,
        "phonetic_score": score_sum / ref_count if ref_count else 1.0,
    })
    return updated_alignment


def _protected_short_word_ambiguity(ref: str, hyp: str) -> bool:
    """Recognize negation words that ASR collapses into one short substitute.

    The rule is deliberately narrow: one token per side, no placeholders, IDs,
    numbers, whitespace, or paired negations. It only covers negation-like
    words against a short single-token ASR rendering so surrounding prose must
    still align through the normal comparator.
    """
    if not ref or not hyp:
        return False
    if any(char.isspace() for char in ref) or any(char.isspace() for char in hyp):
        return False

    ref_clean = _clean_context_word(ref)
    hyp_clean = _clean_context_word(hyp)
    if not ref_clean or not hyp_clean:
        return False
    if (
        _is_placeholder_token(ref)
        or _is_placeholder_token(hyp)
        or _is_id_like_token(ref)
        or _is_id_like_token(hyp)
        or _looks_numeric_word(ref_clean)
        or _looks_numeric_word(hyp_clean)
    ):
        return False

    ref_neg = ref_clean in NEGATIONS
    hyp_neg = hyp_clean in NEGATIONS
    if ref_neg == hyp_neg:
        return False

    return frozenset({ref_clean, hyp_clean}) in _NEGATION_SPOKEN_EQUIVALENCE_PAIRS


def _merged_split_phrase_ambiguities(
    alignment: Dict[str, Any],
    phonetic_threshold: float = 0.85,
) -> List[Dict[str, Any]]:
    """Return narrow evidence for a negation-led two-token merged/split phrase.

    The rule only accepts aligned two-token windows with no insertions,
    deletions, or substitutions anywhere in the comparison. It is intended for
    the exact style of short phrase ambiguity where a negation and a following
    short word are rendered as a compact ASR phrase.
    """
    operations = list(alignment.get("operations", []))
    if any(op.get("op") in {"insertion", "deletion", "substitution"} for op in operations):
        return []

    accepted: List[Dict[str, Any]] = []
    for idx in range(len(operations) - 1):
        left = operations[idx]
        right = operations[idx + 1]
        if left.get("op") != "ambiguous_equivalent":
            continue
        if right.get("op") not in {"exact_match", "normalized_equivalent", "phonetic_equivalent", "ambiguous_equivalent"}:
            continue

        ref_left_raw = str(left.get("ref") or "")
        ref_right_raw = str(right.get("ref") or "")
        hyp_left_raw = str(left.get("hyp") or "")
        hyp_right_raw = str(right.get("hyp") or "")
        ref_left = _clean_context_word(ref_left_raw)
        ref_right = _clean_context_word(ref_right_raw)
        hyp_left = _clean_context_word(hyp_left_raw)
        hyp_right = _clean_context_word(hyp_right_raw)
        if not ref_left or not ref_right or not hyp_left or not hyp_right:
            continue
        if ref_left not in NEGATIONS or ref_right not in FUNCTION_WORDS:
            continue
        if hyp_right not in FUNCTION_WORDS:
            continue
        if any(
            _is_placeholder_token(token)
            or _is_id_like_token(token)
            or _looks_numeric_word(_clean_context_word(token))
            for token in [ref_left_raw, ref_right_raw, hyp_left_raw, hyp_right_raw]
        ):
            continue
        if spoken_token_equivalent(ref_right_raw, hyp_right_raw, phonetic_threshold) == "none":
            continue

        accepted.append(
            {
                "slot": idx,
                "kind": "merged_split_phrase",
                "expected": {
                    "surface": f"{ref_left_raw} {ref_right_raw}",
                    "canonical": f"{ref_left} {ref_right}",
                    "components": [ref_left, ref_right],
                },
                "hypothesis": {
                    "surface": f"{hyp_left_raw} {hyp_right_raw}",
                    "canonical": f"{hyp_left} {hyp_right}",
                    "components": [hyp_left, hyp_right],
                },
                "matched": True,
                "result": "accepted",
            }
        )

    return accepted


def strip_inline_pause_markers(text: str) -> str:
    """Remove TTS ``[Xs]`` / ``[X.Ys]`` pause markers from comparison text.

    Markers are digital silence metadata, not spoken words. Leaving them in the
    book reference turns the digits into ``<NUM>`` slots that ASR never produces,
    causing false missing-content failures.
    """
    if not text:
        return text
    return _INLINE_PAUSE_MARKER_RE.sub(" ", text)


def normalize(
    text: str,
    canon_lookup: Optional[dict] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Normalize text for spoken comparison while tracking ID-like spans.

    Does not collapse adjacent repetitions. Converts ordinals/cardinals to words.
    Does not map distinct homograph pronunciations onto each other.
    """
    # canon_lookup kept for API compatibility; intentional spelling map is SPELLING_VARIANTS
    _ = canon_lookup

    if not text:
        return "", []

    original = text
    # Drop digital pause markers before any number/ID slotting.
    text = strip_inline_pause_markers(text)
    # L0: dialogue quotes must not change typed-ID / number topology vs ASR.
    text = _canonicalize_dialogue_surface(text)
    # L1: lock closed-class compounds containing number-words (no one, anyone…).
    text = _lock_prose_indefinite_compounds(text)
    # L1b: onetime / firsthand / nonetheless before any number or dash work.
    text = _lock_number_bearing_prose_compounds(text)
    # ``NIS's`` is a possessive acronym, not the contraction ``NIS is``.
    text = _ACRONYM_POSSESSIVE_RE.sub(lambda match: match.group(1), text)
    text = _APOSTROPHE_RE.sub("'", text)
    text = separate_attached_measure_units(text)
    # Numeric-led adjective compounds are number + prose (``16-year-old``),
    # not opaque mixed IDs. Split their numeric/prose boundary before the ID
    # pass; internal number hyphens such as ``twenty-one`` remain intact.
    text = separate_numeric_compound_surfaces(text)
    # Typed craft IDs (M8-Tron, M. H. Tron) before letter-initialism join so
    # punctuated craft forms are not flattened into bare acronyms.
    text, id_contexts = _replace_typed_identifier_spans(text)
    # L2: join leftover D.C. / N. I. S. style initialisms, then S-O-B forms.
    text = _join_spaced_dotted_letter_initialisms(text)
    text = _separate_delimited_letter_sequences(text)
    # Roman → digits for true numerals (Chapter XIV). Blocklist keeps DC≠600.
    text = _replace_uppercase_roman_numerals(text)
    text = text.lower()
    # Re-lock after lowercasing so spaced/hyphen forms still fuse before slots.
    text = _lock_number_bearing_prose_compounds(text)
    # No. 1 ≡ Number One before negation / number slotting sees bare "no".
    text = _canonicalize_number_abbreviations(text)
    text = replace_spoken_number_spans(text)
    text = re.sub(r"<tid(\d+)>", lambda match: f"<TID{match.group(1)}>", text)
    text, id_contexts = _finalize_comparison_slots(text, id_contexts)
    text = _BRACKET_RE.sub(" ", text)
    text = text.replace("...", " ").replace("…", " ")
    # Keep the interjection "uh-oh" distinct from generic filler so it can
    # still fail when ASR drops the first syllable or flattens the sound.
    text = re.sub(r"\buh(?:[-\s]+)oh\b", "uhoh", text)

    # Hyphens become token boundaries (preserve sang syllables for repetition analysis)
    text = _DASH_RE.sub(" ", text)
    text = _expand_contractions(text)

    tokens = text.split()
    converted: List[str] = []
    for token in tokens:
        if token.startswith("<ID"):
            converted.append(token)
            continue
        # Strip trailing punctuation stuck to ordinals before match
        bare = re.sub(r"[^\w<>]+$", "", token)
        bare = re.sub(r"^[^\w<>]+", "", bare)
        if re.fullmatch(r"[a-z]\d", bare):
            converted.append(bare[0])
            converted.append(num2words(int(bare[1]), lang="en"))
            continue
        if re.fullmatch(r"\d[a-z]", bare):
            converted.append(num2words(int(bare[0]), lang="en"))
            converted.append(bare[1])
            continue
        m = _ORDINAL_RE.match(bare)
        if m:
            try:
                converted.append(num2words(int(m.group(1)), to="ordinal"))
            except Exception:
                converted.append(bare)
            continue
        converted.append(bare if bare else token)
    text = " ".join(converted)

    def digit_to_word(match: re.Match) -> str:
        """Convert a standalone integer token to English words."""
        try:
            return num2words(int(match.group()), lang="en")
        except Exception:
            return match.group()

    text = _DIGIT_RE.sub(digit_to_word, text)
    text = _NON_ALNUM_KEEP_ID_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()

    # Apply safe spelling variants only (not homograph pronunciation collapse)
    tokens = text.split()
    out_tokens: List[str] = []
    for token in tokens:
        if token.startswith("<ID"):
            out_tokens.append(token)
            continue
        mapped = SPELLING_VARIANTS.get(token, token)
        if " " in mapped:
            out_tokens.extend(mapped.split())
        else:
            out_tokens.append(mapped)

    # Do NOT collapse repetitions — hallucination detection needs them
    normalized_prose = " ".join(out_tokens)
    # Silence unused original/ref for static checkers
    _ = original
    return normalized_prose, id_contexts


def _identifier_wildcard_span_lengths(
    hyp_tokens: Sequence[str],
    end: int,
    max_span: int = 8,
) -> List[int]:
    """Return plausible ASR token spans that can fill one reference ID/NUM slot.

    Book-side placeholders (``<ID0>`` / ``<NUM>``) are value-agnostic: ASR may
    write a digit form, a number word, or a confusable content word (``Five``
    vs ``Hive``). A single non-negation hyp token always qualifies. Longer
    spans still require an identifier-like prefix so placeholders cannot
    swallow ordinary following prose.
    """
    lengths: List[int] = []
    first = max(0, end - max_span)
    for start in range(first, end):
        span = [str(token) for token in hyp_tokens[start:end]]
        if not span or any(_is_placeholder_token(token) for token in span):
            continue
        first_token = span[0]
        first_clean = _clean_context_word(first_token)
        # One token fills the slot: content/number words only (Five↔Hive).
        # Never absorb bare function words (to/of/a) into a placeholder.
        if len(span) == 1:
            if not first_clean or first_clean in NEGATIONS:
                continue
            if first_clean in FUNCTION_WORDS and not _looks_numeric_word(first_clean):
                continue
            lengths.append(1)
            continue
        first_is_initial = len(first_token) <= 2 and first_token.isalpha()
        first_has_digit = any(char.isdigit() for char in first_token)
        if not first_is_initial and not first_has_digit:
            continue
        prefix_is_identifier_piece = all(
            (len(token) <= 2 and token.isalpha())
            or any(char.isdigit() for char in token)
            for token in span[:-1]
        )
        final_is_content = span[-1].isalpha() and len(span[-1]) > 2
        if prefix_is_identifier_piece and final_is_content:
            lengths.append(len(span))
    return lengths


def _copula_homophone_span_match(
    ref_span: Sequence[str],
    hyp_span: Sequence[str],
) -> bool:
    """Match expanded copulas to ASR homophones (they are ↔ their, we are ↔ were)."""
    ref_t = tuple(_clean_context_word(token) for token in ref_span)
    hyp_t = tuple(_clean_context_word(token) for token in hyp_span)
    if not ref_t or not hyp_t or ref_t == hyp_t:
        return False
    groups = (
        {("they", "are"), ("their",), ("there",)},
        {("we", "are"), ("were",)},
        {("you", "are"), ("your",)},
        {("what", "is"), ("whats",)},
        {("that", "is"), ("thats",)},
    )
    return any(ref_t in group and hyp_t in group for group in groups)


def _simple_morphology_equivalent(ref: str, hyp: str) -> bool:
    """Return True for simple inflection pairs ASR often flattens.

    Covers past/present and plural drops such as ``begged``/``beg`` or
    ``guards``/``guard``. Does not accept unrelated stems.
    """
    if not ref or not hyp or ref == hyp:
        return False
    longer, shorter = (ref, hyp) if len(ref) >= len(hyp) else (hyp, ref)
    if len(shorter) < 3:
        return False
    if longer == shorter + "ed" or longer == shorter + "d":
        return True
    if longer == shorter + "ing":
        return True
    if longer == shorter + "s" or longer == shorter + "es":
        return True
    # doubled consonant past: begged / beg
    if (
        len(longer) >= 4
        and longer.endswith("ed")
        and longer[:-2] == shorter + shorter[-1]
    ):
        return True
    return False


def _soft_accept_lone_content_substitution(
    substitutions: Sequence[Dict[str, Any]],
    missing_tokens: Sequence[str],
    extra_tokens: Sequence[str],
    critical: Sequence[Dict[str, str]],
    repetition: Dict[str, Any],
    operations: Optional[Sequence[Dict[str, Any]]] = None,
) -> Optional[Dict[str, str]]:
    """Soft-accept exactly one content-word swap when the sentence shell matches.

    Policy (user option 2): ASR invents different spellings for the same heard
    token (names, book terms, confusable words). When that is the *only*
    unmatched content difference, do not hard-fail. Negation/number critical
    mismatches and multi-token gaps still fail.
    """
    if repetition.get("severe") or repetition.get("is_hallucination"):
        return None
    if any(
        item.get("type") in {
            "negation",
            "missing_negation",
            "number",
            "missing_number",
            "missing_clause",
            "interjection",
        }
        for item in critical
    ):
        return None
    # Shell must match: no deletions/insertions at all (function or content).
    if missing_tokens or extra_tokens:
        return None
    content_subs = [
        sub
        for sub in substitutions
        if _is_content_word(sub.get("ref") or "")
        and _is_content_word(sub.get("hyp") or "")
    ]
    if len(content_subs) != 1:
        return None
    # Only one unmatched op overall (that single content sub).
    if len(list(substitutions)) != 1:
        return None
    # Alignment shell: other ops must be clean matches, not filler~random soft hits.
    if operations is not None:
        for op in operations:
            kind = op.get("op")
            if kind in {
                "exact_match",
                "normalized_equivalent",
                "identifier_wildcard",
                "ambiguous_equivalent",
            }:
                continue
            if kind == "substitution":
                continue
            if kind == "phonetic_equivalent":
                ref_words = str(op.get("ref") or "").split()
                hyp_words = str(op.get("hyp") or "").split()
                # Article elision joins (along a ↔ along) are shell-preserving.
                if (
                    len(ref_words) == 2
                    and len(hyp_words) == 1
                    and _clean_context_word(ref_words[0]) in {"a", "an", "the"}
                ) or (
                    len(hyp_words) == 2
                    and len(ref_words) == 1
                    and _clean_context_word(hyp_words[0]) in {"a", "an", "the"}
                ):
                    continue
                r_tok = _clean_context_word(ref_words[0] if ref_words else "")
                h_tok = _clean_context_word(hyp_words[0] if hyp_words else "")
                if r_tok in FILLER_WORDS or h_tok in FILLER_WORDS:
                    return None
                if r_tok in FUNCTION_WORDS or h_tok in FUNCTION_WORDS:
                    return None
                continue
            return None
    sub = content_subs[0]
    ref = _clean_context_word(sub.get("ref") or "")
    hyp = _clean_context_word(sub.get("hyp") or "")
    if not ref or not hyp:
        return None
    # Book-side negation changes stay hard (handled in critical). ASR writing
    # ``no`` for a long content word (gnome→no) is allowed as soft confusable.
    if ref in NEGATIONS:
        return None
    if hyp in NEGATIONS and len(ref) < 4:
        return None
    if ref in FUNCTION_WORDS or (hyp in FUNCTION_WORDS and hyp not in NEGATIONS):
        return None
    if _is_placeholder_token(sub.get("ref") or "") or _is_placeholder_token(
        sub.get("hyp") or ""
    ):
        return None
    # Content words ≥3 letters (saw/sought). Both ≤3 and non-pronoun blocks
    # short acronym noise (MSV/MSB) while allowing them/him pronouns.
    if min(len(ref), len(hyp)) < 3 and hyp not in NEGATIONS:
        return None
    if max(len(ref), len(hyp)) <= 3 and hyp not in NEGATIONS:
        if not (ref in PERSONAL_PRONOUNS and hyp in PERSONAL_PRONOUNS):
            return None
    if not _simple_morphology_equivalent(ref, hyp) and hyp not in NEGATIONS:
        ratio = fuzz.ratio(ref, hyp)
        # Unrelated stems (nis→george) stay hard fails.
        if ratio < 18:
            return None
        # Short token expanded to long letter-name soup (sob→essohbee),
        # not ordinary confusable pairs (tax→attacks).
        if len(ref) <= 3 and len(hyp) >= 8 and ratio < 60:
            return None
    return {"ref": str(sub.get("ref") or ""), "hyp": str(sub.get("hyp") or "")}


def _soft_accept_context_repaired_short_substitution(
    substitutions: Sequence[Dict[str, Any]],
    missing_tokens: Sequence[str],
    extra_tokens: Sequence[str],
    critical: Sequence[Dict[str, str]],
    repetition: Dict[str, Any],
) -> Optional[Dict[str, str]]:
    """Accept one approved short-token ASR ambiguity with an exact sentence shell.

    These pairs are listener-confirmed context repairs: a listener hears the
    correct sentence unless reading the source, while the ASR has selected a
    nearby pronoun or demonstrative.  The deliberately tiny pair set prevents
    this exception from accepting general substitutions, numbers, negations,
    omissions, additions, or repeated speech.
    """
    if missing_tokens or extra_tokens or repetition.get("severe") or repetition.get("is_hallucination"):
        return None
    if critical or len(substitutions) != 1:
        return None
    substitution = substitutions[0]
    ref = _clean_context_word(str(substitution.get("ref") or ""))
    hyp = _clean_context_word(str(substitution.get("hyp") or ""))
    if not ref or not hyp:
        return None
    if any(
        _is_placeholder_token(str(substitution.get(side) or ""))
        or _looks_numeric_word(token)
        or token in NEGATIONS
        for side, token in (("ref", ref), ("hyp", hyp))
    ):
        return None
    approved_pairs = {
        frozenset({"it", "him"}),
        frozenset({"that", "them"}),
    }
    if frozenset({ref, hyp}) not in approved_pairs:
        return None
    return {
        "ref": str(substitution.get("ref") or ""),
        "hyp": str(substitution.get("hyp") or ""),
    }


def _soft_accept_by_what_one_question(
    substitutions: Sequence[Dict[str, Any]],
    missing_tokens: Sequence[str],
    extra_tokens: Sequence[str],
    critical: Sequence[Dict[str, str]],
    repetition: Dict[str, Any],
    operations: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, str]]:
    """Accept ASR's ``By one, I don't know`` rendering of a short question.

    The rule needs the exact ``by what I`` shell and one numeric placeholder,
    so it cannot suppress arbitrary number substitutions or questions such as
    ``From what?``.  This form is a listener-confirmed question-word rendering
    rather than an accepted change to a spoken number.
    """
    if missing_tokens or extra_tokens or critical or len(substitutions) != 1:
        return None
    if repetition.get("severe") or repetition.get("is_hallucination"):
        return None
    substitution = substitutions[0]
    if _clean_context_word(str(substitution.get("ref") or "")) != "what":
        return None
    if not _is_placeholder_token(str(substitution.get("hyp") or "")):
        return None
    for index, operation in enumerate(operations):
        if operation.get("op") != "substitution":
            continue
        if operation.get("ref") != substitution.get("ref") or operation.get("hyp") != substitution.get("hyp"):
            continue
        previous = operations[index - 1] if index else {}
        following = operations[index + 1] if index + 1 < len(operations) else {}
        if (
            previous.get("op") in {"exact_match", "normalized_equivalent"}
            and _clean_context_word(str(previous.get("ref") or "")) == "by"
            and following.get("op") in {"exact_match", "normalized_equivalent"}
            and _clean_context_word(str(following.get("ref") or "")) == "i"
        ):
            return {
                "ref": str(substitution.get("ref") or ""),
                "hyp": str(substitution.get("hyp") or ""),
            }
    return None


def _soft_accept_high_coverage_listener_fuzzy(
    substitutions: Sequence[Dict[str, Any]],
    missing_tokens: Sequence[str],
    extra_tokens: Sequence[str],
    critical: Sequence[Dict[str, str]],
    repetition: Dict[str, Any],
    coverage: float,
    phonetic: float,
    ref_tokens: Sequence[str],
    ref_ids: Sequence[Dict[str, Any]],
    hyp_ids: Sequence[Dict[str, Any]],
) -> Optional[List[Dict[str, str]]]:
    """Accept tightly bounded listener-level ASR resegmentations as fuzzies.

    This is not a general low-score pass.  It covers a long sentence where ASR
    preserves at least 92 percent of the token shell but leaves one or two
    short residuals after ordinary phonetic matching.  Equal identifier counts
    are mandatory: an added or removed number remains a real failure even when
    sequence alignment happens to consume it as a wildcard.
    """
    if critical or repetition.get("severe") or repetition.get("is_hallucination"):
        return None
    if extra_tokens or len(ref_tokens) < 24:
        return None
    if coverage < 0.92 or phonetic < 0.92:
        return None
    if len(ref_ids) != len(hyp_ids):
        return None
    differences = len(substitutions) + len(missing_tokens)
    if not 1 <= differences <= 2:
        return None
    for token in [*missing_tokens, *(item.get("ref") or "" for item in substitutions), *(item.get("hyp") or "" for item in substitutions)]:
        clean = _clean_context_word(str(token))
        if not clean or clean in NEGATIONS or _is_placeholder_token(str(token)):
            return None
    return [
        {"ref": str(item.get("ref") or ""), "hyp": str(item.get("hyp") or "")}
        for item in substitutions
    ]


def align_tokens(
    ref_tokens: Sequence[str],
    hyp_tokens: Sequence[str],
    phonetic_threshold: float = 0.85,
    strict_ref_tokens: Optional[Set[str]] = None,
    book_term_tokens: Optional[Set[str]] = None,
    hyp_lexical_possessives: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Token-level sequence alignment via dynamic programming.

    Operations: exact_match, normalized_equivalent, phonetic_equivalent,
    substitution, insertion, deletion. Recurring book terms can consume a
    raw hypothesis possessive only when that possessive expanded to two tokens.
    """
    n, m = len(ref_tokens), len(hyp_tokens)
    # Rank every path by hard edits, fuzzy use, operation count, consumed span,
    # then stable candidate order. Equal edit-distance paths otherwise depend on
    # append order and can hide an exact boundary behind an older fuzzy join.
    Rank = Tuple[int, int, int, int, Tuple[int, ...]]
    INF = 10**9
    inf_rank: Rank = (INF, INF, INF, INF, (INF,))
    dp: List[List[Rank]] = [[inf_rank] * (m + 1) for _ in range(n + 1)]
    bt: List[List[Optional[str]]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = (0, 0, 0, 0, ())
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, -i, (95,) * i)
        bt[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, j, -j, (96,) * j)
        bt[0][j] = "ins"

    def ranked_candidate(
        previous_i: int,
        previous_j: int,
        kind: str,
        equivalence: str,
        hard_edits: int,
        stable_order: int,
    ) -> Tuple[Rank, str, str]:
        """Build one deterministic DP candidate from its predecessor state."""
        previous = dp[previous_i][previous_j]
        ref_span = i - previous_i
        hyp_span = j - previous_j
        fuzzy = int(equivalence in {
            "phonetic", "ambiguous", "guarded_function_phonetic", "guarded_title_phonetic",
            "guarded_reduced_auxiliary",
        })
        rank: Rank = (
            previous[0] + hard_edits,
            previous[1] + fuzzy,
            previous[2] + 1,
            previous[3] - max(ref_span, hyp_span),
            previous[4] + (stable_order,),
        )
        return rank, kind, equivalence

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # multi-token hyp packs against one ref rarely needed — 1:1 first
            eq = spoken_token_equivalent(
                ref_tokens[i - 1],
                hyp_tokens[j - 1],
                phonetic_threshold,
                strict_ref_tokens=strict_ref_tokens,
            )
            match_cost = 0 if eq != "none" else 1
            direct_order = {
                "exact": 10,
                "normalized": 11,
                "homograph": 20,
                "phonetic": 30,
                "ambiguous": 31,
            }.get(eq, 90)
            candidates = [
                ranked_candidate(i - 1, j - 1, "match" if eq != "none" else "sub", eq, match_cost, direct_order),
                ranked_candidate(i - 1, j, "del", "none", 1, 95),
                ranked_candidate(i, j - 1, "ins", "none", 1, 96),
            ]
            ref_tok = ref_tokens[i - 1]
            if _is_placeholder_token(ref_tok):
                for span_len in _identifier_wildcard_span_lengths(hyp_tokens, j):
                    candidates.append(ranked_candidate(
                        i - 1, j - span_len, f"wildcard{span_len}", "wildcard", 0, 20
                    ))
            # Copula homophones: they are ↔ their, we are ↔ were, …
            if i >= 2 and j >= 1 and _copula_homophone_span_match(
                ref_tokens[i - 2:i],
                [hyp_tokens[j - 1]],
            ):
                candidates.append(ranked_candidate(i - 2, j - 1, "copula_ref2", "phonetic", 0, 40))
            if i >= 1 and j >= 2 and _copula_homophone_span_match(
                [ref_tok],
                hyp_tokens[j - 2:j],
            ):
                candidates.append(ranked_candidate(i - 1, j - 2, "copula_hyp2", "phonetic", 0, 40))
            if i >= 2 and j >= 1:
                title_mode = _guarded_title_name_phonetic_mode(
                    ref_tokens[i - 2:i],
                    [hyp_tokens[j - 1]],
                    strict_ref_tokens=strict_ref_tokens,
                    phonetic_threshold=phonetic_threshold,
                )
                if title_mode is not None:
                    candidates.append(ranked_candidate(i - 2, j - 1, "title_name_ref2", title_mode, 0, 43))
            if i >= 1 and j >= 2:
                title_mode = _guarded_title_name_phonetic_mode(
                    [ref_tok],
                    hyp_tokens[j - 2:j],
                    strict_ref_tokens=strict_ref_tokens,
                    phonetic_threshold=phonetic_threshold,
                )
                if title_mode is not None:
                    candidates.append(ranked_candidate(i - 1, j - 2, "title_name_hyp2", title_mode, 0, 43))
            for span_len in (2, 3):
                if j < span_len:
                    continue
                boundary_mode = _boundary_resegmentation_mode(
                    [ref_tok],
                    hyp_tokens[j - span_len:j],
                    strict_ref_tokens=strict_ref_tokens,
                )
                if boundary_mode is not None:
                    boundary_order = 12 if boundary_mode in {"exact_surface", "special_exact_surface"} else 41
                    candidates.append(ranked_candidate(
                        i - 1, j - span_len, f"boundary_hyp{span_len}", boundary_mode, 0, boundary_order
                    ))
            if (
                j >= 2
                and book_term_tokens
                and _clean_context_word(ref_tok) in book_term_tokens
                and hyp_tokens[j - 1] == "is"
                and _clean_context_word(hyp_tokens[j - 2]) in (hyp_lexical_possessives or set())
                and abs(
                    len(_clean_context_word(ref_tok))
                    - len(_clean_context_word(hyp_tokens[j - 2]))
                ) <= 2
                and tokens_phonetically_equal(
                    _clean_context_word(ref_tok),
                    _clean_context_word(hyp_tokens[j - 2]),
                    phonetic_threshold,
                )
            ):
                # Only raw apostrophe-s evidence may consume the added ``is``.
                candidates.append(ranked_candidate(i - 1, j - 2, "book_possessive_hyp2", "phonetic", 0, 42))
            # One hypothesis token can join two reference words without loss.
            for span_len in (2, 3):
                if i < span_len:
                    continue
                boundary_mode = _boundary_resegmentation_mode(
                    ref_tokens[i - span_len:i],
                    [hyp_tokens[j - 1]],
                    strict_ref_tokens=strict_ref_tokens,
                )
                if boundary_mode is not None:
                    boundary_order = 12 if boundary_mode in {"exact_surface", "special_exact_surface"} else 41
                    candidates.append(ranked_candidate(
                        i - span_len, j - 1, f"boundary_ref{span_len}", boundary_mode, 0, boundary_order
                    ))

            if i >= 2:
                parts = ref_tokens[i - 2:i]
                if _expected_repeat_collapse_equivalent(
                    parts,
                    hyp_tokens[j - 1],
                    strict_ref_tokens=strict_ref_tokens,
                ):
                    candidates.append(ranked_candidate(i - 2, j - 1, "repeat_ref2", "phonetic", 0, 50))
                if _modal_base_to_past_equivalent(parts, hyp_tokens[j - 1]):
                    candidates.append(ranked_candidate(i - 2, j - 1, "modal_past_ref2", "phonetic", 0, 51))
                if _trailing_name_fusion_equivalent(
                    parts,
                    hyp_tokens[j - 1],
                    strict_ref_tokens=strict_ref_tokens,
                ):
                    candidates.append(ranked_candidate(i - 2, j - 1, "name_ref2", "phonetic", 0, 52))

            best = min(candidates, key=lambda x: x[0])
            dp[i][j] = best[0]
            bt[i][j] = best[1] + "|" + best[2]

    # Backtrack
    operations: List[Dict[str, Any]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i == 0:
            operations.append({"op": "insertion", "hyp": hyp_tokens[j - 1], "ref": None})
            j -= 1
            continue
        if j == 0:
            operations.append({"op": "deletion", "ref": ref_tokens[i - 1], "hyp": None})
            i -= 1
            continue
        tag = bt[i][j] or "sub|none"
        kind, eq = tag.split("|", 1) if "|" in tag else (tag, "none")
        if kind == "match":
            op_name = {
                "exact": "exact_match",
                "normalized": "normalized_equivalent",
                "homograph": "phonetic_equivalent",
                "phonetic": "phonetic_equivalent",
                "ambiguous": "ambiguous_equivalent",
            }.get(eq, "exact_match")
            operation = {
                "op": op_name,
                "ref": ref_tokens[i - 1],
                "hyp": hyp_tokens[j - 1],
            }
            if eq == "homograph":
                operation["specialized_rule"] = "homograph_whitelist"
            operations.append(operation)
            i -= 1
            j -= 1
        elif kind.startswith("boundary_hyp"):
            span_len = int(kind.removeprefix("boundary_hyp"))
            operations.append({
                "op": "boundary_resegmentation",
                "ref": ref_tokens[i - 1],
                "hyp": " ".join(hyp_tokens[j - span_len:j]),
                "boundary_method": eq,
                "direction": f"ref_1_to_hyp_{span_len}",
            })
            i -= 1
            j -= span_len
        elif kind.startswith("boundary_ref"):
            span_len = int(kind.removeprefix("boundary_ref"))
            operations.append({
                "op": "boundary_resegmentation",
                "ref": " ".join(ref_tokens[i - span_len:i]),
                "hyp": hyp_tokens[j - 1],
                "boundary_method": eq,
                "direction": f"ref_{span_len}_to_hyp_1",
            })
            i -= span_len
            j -= 1
        elif kind in {"book_possessive_hyp2", "copula_hyp2", "title_name_hyp2"}:
            operations.append({
                "op": "phonetic_equivalent",
                "ref": ref_tokens[i - 1],
                "hyp": " ".join(hyp_tokens[j - 2:j]),
            })
            i -= 1
            j -= 2
        elif kind in {"repeat_ref2", "name_ref2", "modal_past_ref2", "copula_ref2", "title_name_ref2"}:
            operations.append({
                "op": "phonetic_equivalent",
                "ref": " ".join(ref_tokens[i - 2:i]),
                "hyp": hyp_tokens[j - 1],
            })
            i -= 2
            j -= 1
        elif kind.startswith("wildcard"):
            span_len = int(kind.removeprefix("wildcard"))
            operations.append({
                "op": "identifier_wildcard",
                "ref": ref_tokens[i - 1],
                "hyp": " ".join(hyp_tokens[j - span_len:j]),
                "hyp_tokens": list(hyp_tokens[j - span_len:j]),
            })
            i -= 1
            j -= span_len
        elif kind == "del":
            operations.append({"op": "deletion", "ref": ref_tokens[i - 1], "hyp": None})
            i -= 1
        elif kind == "ins":
            operations.append({"op": "insertion", "ref": None, "hyp": hyp_tokens[j - 1]})
            j -= 1
        else:
            operations.append({
                "op": "substitution",
                "ref": ref_tokens[i - 1],
                "hyp": hyp_tokens[j - 1],
            })
            i -= 1
            j -= 1

    operations.reverse()

    matched = sum(
        1
        for op in operations
        if op["op"] in {
            "exact_match",
            "normalized_equivalent",
            "phonetic_equivalent",
            "ambiguous_equivalent",
            "boundary_resegmentation",
            "identifier_wildcard",
        }
    )
    # Count reference coverage by matched ref pieces
    covered_refs = 0
    for op in operations:
        if op["op"] in {
            "exact_match",
            "normalized_equivalent",
            "phonetic_equivalent",
            "ambiguous_equivalent",
            "boundary_resegmentation",
            "identifier_wildcard",
        }:
            covered_refs += len(str(op.get("ref", "")).split())
        # substitutions cover none

    extra = [op["hyp"] for op in operations if op["op"] == "insertion" and op.get("hyp")]
    missing = [op["ref"] for op in operations if op["op"] == "deletion" and op.get("ref")]
    substitutions = [
        {"ref": op["ref"], "hyp": op["hyp"]}
        for op in operations
        if op["op"] == "substitution"
    ]

    phonetic_ops = [
        op for op in operations
        if op["op"] in {
            "exact_match",
            "normalized_equivalent",
            "phonetic_equivalent",
            "ambiguous_equivalent",
            "identifier_wildcard",
        }
    ]
    phonetic_score = (len(phonetic_ops) / max(1, n)) if n else 1.0
    # Refine phonetic_score as matched weighted quality
    if n:
        score_sum = 0.0
        for op in operations:
            if op["op"] == "exact_match":
                score_sum += 1.0
            elif op["op"] == "normalized_equivalent":
                score_sum += 1.0
            elif op["op"] == "phonetic_equivalent":
                # Accepted sound-alikes and content-preserving joins are neutral.
                score_sum += len(str(op.get("ref") or "").split())
            elif op["op"] == "boundary_resegmentation":
                # Boundary changes preserve all reference spoken content.
                score_sum += len(str(op.get("ref") or "").split())
            elif op["op"] == "ambiguous_equivalent":
                score_sum += 1.0
            elif op["op"] == "identifier_wildcard":
                score_sum += 1.0
        phonetic_score = score_sum / n

    coverage = covered_refs / n if n else 1.0

    return {
        "operations": operations,
        "matched_reference_tokens": matched,
        "reference_token_count": n,
        "hypothesis_token_count": m,
        "extra_tokens": extra,
        "missing_tokens": missing,
        "substitutions": substitutions,
        "coverage_score": coverage,
        "phonetic_score": phonetic_score,
    }


def _is_content_word(token: str) -> bool:
    """Return True for non-function words used in extra-speech leniency checks."""
    lower = token.lower()
    return lower not in FUNCTION_WORDS and lower not in FILLER_WORDS and not token.startswith("<ID")


def detect_repetition(
    ref_tokens: Sequence[str],
    hyp_tokens: Sequence[str],
    max_phrase_len: int = 6,
) -> Dict[str, Any]:
    """
    Detect unexpected adjacent word/phrase repetition relative to the reference.

    Uses block comparison for phrases of length 2..max_phrase_len.
    """
    def max_adjacent_word(tokens: Sequence[str]) -> Dict[str, int]:
        """Count maximum adjacent runs of each single word."""
        if not tokens:
            return {}
        best: Dict[str, int] = {}
        cur = tokens[0]
        count = 1
        for tok in tokens[1:]:
            if tok == cur:
                count += 1
            else:
                best[cur] = max(best.get(cur, 0), count)
                cur = tok
                count = 1
        best[cur] = max(best.get(cur, 0), count)
        return best

    def max_phrase_runs(tokens: Sequence[str], n: int) -> Dict[str, int]:
        """Count maximum adjacent block repetitions of n-token phrases."""
        if len(tokens) < n * 2:
            return {}
        best: Dict[str, int] = {}
        i = 0
        while i <= len(tokens) - n:
            block = tuple(tokens[i:i + n])
            run = 1
            j = i + n
            while j <= len(tokens) - n and tuple(tokens[j:j + n]) == block:
                run += 1
                j += n
            if run >= 2:
                key = " ".join(block)
                best[key] = max(best.get(key, 0), run)
                i = j
            else:
                i += 1
        return best

    def syllable_close(a: str, b: str) -> bool:
        """True when short syllables are close (la/lal) for song-like text."""
        if a == b:
            return True
        if max(len(a), len(b)) > 5:
            return False
        if a.startswith(b) or b.startswith(a):
            return fuzz.ratio(a, b) >= 65
        return fuzz.ratio(a, b) >= 80

    def allowed_for_word(word: str, ref_tokens_local: Sequence[str], ref_adj: Dict[str, int]) -> int:
        """Compute allowed repetition count including near-syllable relatives."""
        allowed = ref_adj.get(word, 0)
        total = 0
        for t in ref_tokens_local:
            if syllable_close(t, word):
                total += 1
        adj_near = 0
        for rw, rc in ref_adj.items():
            if syllable_close(rw, word):
                adj_near = max(adj_near, rc)
        return max(allowed, total, adj_near)

    ref_words = max_adjacent_word(ref_tokens)
    hyp_words = max_adjacent_word(hyp_tokens)
    details: List[Dict[str, Any]] = []
    severe = False

    for word, hyp_count in hyp_words.items():
        allowed = allowed_for_word(word, ref_tokens, ref_words)
        if hyp_count > allowed and hyp_count >= 2:
            excess = hyp_count - allowed
            # Short sung syllables with surrounding related pronunciation — tolerate more
            if allowed >= 3 and len(word) <= 3 and excess <= 2:
                continue
            if allowed >= 4 and len(word) <= 4 and hyp_count <= allowed + 2:
                continue
            sev = "severe" if hyp_count >= 4 or excess >= 3 else ("moderate" if hyp_count == 3 else "minor")
            if (word in FUNCTION_WORDS or word in FILLER_WORDS) and hyp_count <= 2:
                sev = "tolerable"
            details.append({
                "pattern": word,
                "hyp_count": hyp_count,
                "ref_count": allowed,
                "type": "single_word",
                "severity": sev,
            })
            if sev == "severe":
                severe = True

    for n in range(2, max_phrase_len + 1):
        ref_p = max_phrase_runs(ref_tokens, n)
        hyp_p = max_phrase_runs(hyp_tokens, n)
        for phrase, hyp_count in hyp_p.items():
            parts = phrase.split()
            if parts and all(len(p) <= 3 for p in parts):
                soft = min(allowed_for_word(p, ref_tokens, ref_words) for p in parts)
                allowed = max(ref_p.get(phrase, 0), max(1, soft // max(1, n)) if soft else 0)
                # Song-line: entire hyp of short syllables matching short ref syllables
                near_ref = sum(1 for t in ref_tokens if any(syllable_close(t, p) for p in parts))
                if near_ref >= len(parts) * 2:
                    allowed = max(allowed, hyp_count)
            else:
                allowed = ref_p.get(phrase, 0)
            if hyp_count > allowed and hyp_count >= 2:
                excess = hyp_count - allowed
                if allowed >= 2 and excess <= 1 and all(len(p) <= 3 for p in parts):
                    continue
                short_reference = sum(len(token) <= 3 for token in ref_tokens)
                if parts and all(len(part) <= 3 for part in parts) and short_reference >= len(ref_tokens) - 1:
                    # Sung syllables often collapse phonetically in ASR. Keep
                    # the low similarity score, but do not call it repetition.
                    continue
                details.append({
                    "pattern": phrase,
                    "hyp_count": hyp_count,
                    "ref_count": allowed,
                    "type": "phrase",
                    "severity": "severe",
                })
                severe = True

    is_hallucination = bool(details) and any(
        d["severity"] in {"severe", "moderate", "minor"} for d in details
    )
    top = details[0] if details else None
    return {
        "is_hallucination": is_hallucination and any(d["severity"] != "tolerable" for d in details),
        "details": details,
        "severe": severe,
        "pattern": top["pattern"] if top else "",
        "count": top["hyp_count"] if top else 0,
        "ref_count": top["ref_count"] if top else 0,
        "type": top["type"] if top else "",
        "severity": top["severity"] if top else "none",
    }


def detect_truncation(
    ref_tokens: Sequence[str],
    hyp_tokens: Sequence[str],
    alignment: Optional[Dict[str, Any]] = None,
) -> dict:
    """Detect severe truncation after restoring exact boundary-equivalent words.

    Raw ASR token counts understate speech when one token represents two or
    three exact reference tokens. Count only the known deficit consumed by an
    accepted boundary operation, leaving unrelated missing content visible.
    """
    ref_words = len(ref_tokens)
    hyp_words = len(hyp_tokens)
    effective_hyp_words = hyp_words
    for operation in (alignment or {}).get("operations", []):
        if operation.get("op") != "boundary_resegmentation":
            continue
        ref_count = len(str(operation.get("ref") or "").split())
        hyp_count = len(str(operation.get("hyp") or "").split())
        effective_hyp_words += max(0, ref_count - hyp_count)
    if ref_words == 0:
        return {
            "is_truncated": False,
            "ref_words": 0,
            "hyp_words": hyp_words,
            "effective_hyp_words": effective_hyp_words,
            "ratio": 1.0,
        }
    ratio = effective_hyp_words / ref_words
    return {
        "is_truncated": ratio < 0.4,
        "ref_words": ref_words,
        "hyp_words": hyp_words,
        "effective_hyp_words": effective_hyp_words,
        "ratio": ratio,
    }


def _classify_extra_speech(
    operations: List[Dict[str, Any]],
    ref_count: int,
    cfg: Dict[str, Any],
) -> Tuple[List[str], Optional[str]]:
    """Classify unmatched insertions as prefix/internal/suffix and severity."""
    extras = [op for op in operations if op["op"] == "insertion"]
    if not extras:
        return [], None

    # Position: before first match / after last match / internal
    first_match = next(
        (i for i, op in enumerate(operations)
         if op["op"] not in {"insertion"}),
        len(operations),
    )
    last_match = max(
        (i for i, op in enumerate(operations)
         if op["op"] not in {"insertion", "deletion"}),
        default=-1,
    )

    prefix = []
    internal = []
    suffix = []
    for i, op in enumerate(operations):
        if op["op"] != "insertion":
            continue
        tok = op.get("hyp") or ""
        if i < first_match:
            prefix.append(tok)
        elif last_match >= 0 and i > last_match:
            suffix.append(tok)
        else:
            internal.append(tok)

    content_extra = [t for t in prefix + internal + suffix if _is_content_word(t)]
    max_words = int(cfg.get("max_extra_content_words", 1))
    max_ratio = float(cfg.get("max_extra_content_ratio", 0.15))
    allowed = max(max_words, int(ref_count * max_ratio)) if ref_count else max_words

    failure_type = None
    if len(content_extra) > allowed:
        if len(suffix) >= 2 and sum(1 for t in suffix if _is_content_word(t)) >= 2:
            failure_type = "unexpected_suffix"
        elif len(prefix) >= 2 and sum(1 for t in prefix if _is_content_word(t)) >= 2:
            failure_type = "unexpected_prefix"
        else:
            failure_type = "unexpected_internal_speech"
    # Consecutive content extras always fail
    run = 0
    for t in prefix + internal + suffix:
        if _is_content_word(t):
            run += 1
            if run >= 2:
                failure_type = failure_type or "unexpected_internal_speech"
                break
        else:
            run = 0

    return content_extra, failure_type


def _critical_mismatches(
    alignment: Dict[str, Any],
    ref_tokens: Sequence[str],
    hyp_tokens: Sequence[str],
    phonetic_threshold: float = 0.85,
) -> List[Dict[str, str]]:
    """Find changed negations and numbers that alter meaning materially."""
    critical: List[Dict[str, str]] = []

    for sub in alignment.get("substitutions", []):
        ref = (sub.get("ref") or "").lower()
        hyp = (sub.get("hyp") or "").lower()
        if ref != hyp and {ref, hyp} <= {"and", "or", "to"}:
            # These words carry list/range structure after numeric slotting;
            # phonetic leniency must not erase a changed numeric relationship.
            critical.append({"type": "numeric_structure", "ref": ref, "hyp": hyp})
        if ref == "uhoh" or hyp == "uhoh":
            if ref != hyp and spoken_token_equivalent(ref, hyp, phonetic_threshold) == "none":
                critical.append({"type": "interjection", "ref": ref, "hyp": hyp})
        # Polarity critical only when the *book* had a negation that changed.
        # ASR writing ``no`` for a long content word (gnome→no) is confusable
        # spelling, not a polarity edit of the source.
        if ref in NEGATIONS:
            if ref != hyp and spoken_token_equivalent(ref, hyp, phonetic_threshold) == "none":
                critical.append({"type": "negation", "ref": ref, "hyp": hyp})
        # number word tokens (cardinal/ordinal words or digits already expanded)
        ref_is_num = _looks_numeric_word(ref)
        hyp_is_num = _looks_numeric_word(hyp)
        if ref_is_num or hyp_is_num:
            if ref != hyp and spoken_token_equivalent(ref, hyp) == "none":
                critical.append({"type": "number", "ref": ref, "hyp": hyp})

    for miss in alignment.get("missing_tokens", []):
        low = (miss or "").lower()
        if low in NEGATIONS:
            critical.append({"type": "missing_negation", "ref": low, "hyp": ""})
        if _looks_numeric_word(low):
            critical.append({"type": "missing_number", "ref": low, "hyp": ""})

    # Missing two+ consecutive content words
    run: List[str] = []
    for op in alignment.get("operations", []):
        if op["op"] == "deletion" and _is_content_word(op.get("ref") or ""):
            run.append(op["ref"])
        else:
            if len(run) >= 2:
                critical.append({
                    "type": "missing_clause",
                    "ref": " ".join(run),
                    "hyp": "",
                })
            run = []
    if len(run) >= 2:
        critical.append({"type": "missing_clause", "ref": " ".join(run), "hyp": ""})

    _ = hyp_tokens
    _ = ref_tokens
    return critical


def _identifier_mismatches(
    ref_ids: Sequence[Dict[str, Any]],
    hyp_ids: Sequence[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Return typed-identifier slot mismatches by slot order."""
    mismatches: List[Dict[str, str]] = []
    for comparison in _identifier_comparisons(ref_ids, hyp_ids):
        if comparison["result"] == "mismatch":
            mismatches.append(
                {
                    "type": "identifier",
                    "ref": str(comparison["expected"].get("canonical") or ""),
                    "hyp": str(comparison["hypothesis"].get("canonical") or ""),
                }
            )
    return mismatches


def _identifier_contexts_equivalent(
    reference: Dict[str, Any],
    hypothesis: Dict[str, Any],
    phonetic_threshold: float = 0.85,
) -> bool:
    """Compare identifier components without weakening numeric component checks."""
    reference_components = reference.get("components") or [reference.get("canonical_id", "")]
    hypothesis_components = hypothesis.get("components") or [hypothesis.get("canonical_id", "")]
    if len(reference_components) != len(hypothesis_components):
        return False
    for ref_component, hyp_component in zip(reference_components, hypothesis_components):
        ref_component = str(ref_component)
        hyp_component = str(hyp_component)
        if ref_component == hyp_component:
            continue
        if ref_component.isdigit() or hyp_component.isdigit():
            return False
        if not tokens_phonetically_equal(ref_component, hyp_component, phonetic_threshold):
            return False
    return True


def _identifier_comparison_side(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Render one identifier context into compact comparison evidence."""
    if not context:
        return {
            "kind": "",
            "canonical": "",
            "components": [],
        }
    components = [str(item) for item in (context.get("components") or []) if str(item)]
    canonical = str(context.get("canonical_id") or "")
    if not components and canonical:
        components = [canonical]
    return {
        "kind": str(context.get("kind") or "identifier"),
        "canonical": canonical,
        "components": components,
    }


def _identifier_comparisons(
    ref_ids: Sequence[Dict[str, Any]],
    hyp_ids: Sequence[Dict[str, Any]],
    wildcard_slots: Optional[Set[int]] = None,
) -> List[Dict[str, Any]]:
    """Build slot-by-slot evidence for wildcard identifier comparison."""
    wildcard_slots = wildcard_slots or set()
    comparisons: List[Dict[str, Any]] = []
    shared = min(len(ref_ids), len(hyp_ids))
    for idx in range(max(len(ref_ids), len(hyp_ids))):
        ref_id = ref_ids[idx] if idx < len(ref_ids) else None
        hyp_id = hyp_ids[idx] if idx < len(hyp_ids) else None
        expected = _identifier_comparison_side(ref_id)
        hypothesis = _identifier_comparison_side(hyp_id)
        expected_kind = expected["kind"] or "identifier"
        hypothesis_kind = hypothesis["kind"] or "identifier"
        if idx >= shared:
            kind = expected_kind if ref_id is not None else hypothesis_kind
        elif expected_kind == hypothesis_kind:
            kind = expected_kind
        else:
            kind = "mixed"

        if expected_kind == hypothesis_kind == "numeric":
            result = "exempt"
            matched = True
        elif ref_id is None or hyp_id is None:
            if ref_id is not None and idx in wildcard_slots:
                hypothesis = {
                    "kind": expected_kind,
                    "canonical": "<wildcard>",
                    "components": [],
                }
                result = "matched"
                matched = True
                comparisons.append(
                    {
                        "slot": idx,
                        "kind": kind,
                        "expected": expected,
                        "hypothesis": hypothesis,
                        "matched": matched,
                        "result": result,
                    }
                )
                continue
            result = "exempt" if (expected_kind == "numeric" or hypothesis_kind == "numeric") else "mismatch"
            matched = result != "mismatch"
        elif expected_kind == hypothesis_kind:
            # Wildcard slots compare by position and kind, not by slot payload.
            result = "matched"
            matched = True
        elif expected_kind == "numeric" or hypothesis_kind == "numeric":
            result = "mismatch"
            matched = False
        else:
            result = "mismatch"
            matched = False

        comparisons.append(
            {
                "slot": idx,
                "kind": kind,
                "expected": expected,
                "hypothesis": hypothesis,
                "matched": matched,
                "result": result,
            }
        )
    return comparisons


def _identifier_score(comparisons: Sequence[Dict[str, Any]]) -> float:
    """Compute identifier score from non-exempt slot comparisons."""
    counted = [item for item in comparisons if item.get("result") != "exempt"]
    if not counted:
        return 1.0
    matched = sum(1 for item in counted if item.get("matched"))
    return matched / float(len(counted))


def _render_aligned_hypothesis(
    operations: Sequence[Dict[str, Any]],
) -> str:
    """Render hypothesis alignment with consumed ID spans as reference slots."""
    rendered: List[str] = []
    for operation in operations:
        op_name = operation.get("op")
        if op_name == "identifier_wildcard":
            rendered.append(str(operation.get("ref") or ""))
        elif op_name == "deletion":
            continue
        elif operation.get("hyp") is not None:
            rendered.extend(str(operation.get("hyp") or "").split())
    return " ".join(token for token in rendered if token)


_ALIGNED_MATCH_OPS = frozenset({
    "exact_match",
    "normalized_equivalent",
    "phonetic_equivalent",
    "ambiguous_equivalent",
    "boundary_resegmentation",
    "identifier_wildcard",
})


def _aligned_repetition_tokens(operations: Sequence[Dict[str, Any]]) -> List[str]:
    """Build hypothesis tokens for repetition checks using accepted book forms.

    Phonetic matches such as carstairs ≈ car-stairs count as the reference
    word, so a phrase the book already repeats is not a new repeated phrase.
    Insertions and substitutions keep hypothesis words so an extra copy still
    fails.

    Args:
        operations: Alignment ops from align_tokens.

    Returns:
        Hypothesis token list in book spelling where the match was accepted.
    """
    tokens: List[str] = []
    for operation in operations:
        op_name = operation.get("op")
        if op_name == "deletion":
            continue
        if op_name in _ALIGNED_MATCH_OPS:
            source = operation.get("ref") or operation.get("hyp") or ""
        else:
            source = operation.get("hyp") or ""
        tokens.extend(str(source).split())
    return [token for token in tokens if token]


def _isolated_label_list_items(text: str) -> List[Dict[str, str]]:
    """Parse an isolated comma/semicolon label list into comparable items.

    Returns empty list when text is not a narrow label-list candidate. Candidate
    lists must contain at least three single-word items, use only comma/semicolon
    separators, avoid digits, placeholders, negations, and obvious prose words.
    """
    raw = _strip_balanced_quotes(text or "")
    if not raw:
        return []

    core = re.sub(r"[.!?…]+$", "", raw).strip()
    if not core or not re.search(r"[;,]", core):
        return []
    if re.search(r"[.?!:()\[\]{}]", core):
        return []

    parts = [part.strip() for part in re.split(r"[;,]", core)]
    if len(parts) < 3 or any(not part for part in parts):
        return []

    items: List[Dict[str, str]] = []
    for part in parts:
        if any(ch.isspace() for ch in part):
            return []
        if any(ch.isdigit() for ch in part):
            return []
        if not re.fullmatch(r"[A-Za-z][A-Za-z'\-]*[A-Za-z]", part):
            return []
        clean = _clean_context_word(part)
        if len(clean) < 4 or clean in NEGATIONS or clean in FUNCTION_WORDS:
            return []
        if _looks_numeric_word(clean):
            return []
        normalized, contexts = normalize(part)
        if any(token.startswith("<ID") or token.startswith("<NUM>") for token in normalized.split()):
            return []
        if contexts:
            return []
        items.append({
            "surface": part,
            "canonical": clean,
            "components": [clean],
        })
    return items


def _accepted_list_label_equivalences(
    ref_text: str,
    hyp_text: str,
    phonetic_threshold: float = 0.85,
) -> List[Dict[str, Any]]:
    """Return accepted evidence for isolated label-list substitutions.

    The rule stays narrow: both sides must be isolated comma/semicolon-separated
    single-word lists with identical item counts, and every differing position
    must be supported by strong string or phonetic evidence.
    """
    ref_items = _isolated_label_list_items(ref_text)
    hyp_items = _isolated_label_list_items(hyp_text)
    if len(ref_items) < 3 or len(ref_items) != len(hyp_items):
        return []

    accepted: List[Dict[str, Any]] = []
    for slot, (ref_item, hyp_item) in enumerate(zip(ref_items, hyp_items)):
        ref_canonical = ref_item["canonical"]
        hyp_canonical = hyp_item["canonical"]
        if ref_canonical == hyp_canonical:
            continue
        similarity = fuzz.ratio(ref_canonical, hyp_canonical) / 100.0
        phonetic = tokens_phonetically_equal(ref_canonical, hyp_canonical, phonetic_threshold)
        length_delta = abs(len(ref_canonical) - len(hyp_canonical))
        if length_delta > 2 and not phonetic:
            return []
        accepted.append(
            {
                "slot": slot,
                "kind": "label_list",
                "expected": {
                    "surface": ref_item["surface"],
                    "canonical": ref_canonical,
                    "components": list(ref_item["components"]),
                },
                "hypothesis": {
                    "surface": hyp_item["surface"],
                    "canonical": hyp_canonical,
                    "components": list(hyp_item["components"]),
                },
                "matched": True,
                "result": "accepted",
                "similarity": round(similarity, 3),
                "length_delta": length_delta,
            }
        )
    return accepted


def _accepted_book_term_equivalences(
    alignment: Dict[str, Any],
    book_term_evidence: Optional[Dict[str, Any]],
    repetition: Dict[str, Any],
    truncation: Dict[str, Any],
    phonetic_threshold: float = 0.85,
) -> List[Dict[str, Any]]:
    """Accept clean recurring book-term ASR spellings only.

    This is intentionally stricter than general phonetic matching. Source
    evidence must identify the expected term as recurring, all surrounding
    prose must align exactly, by normalization, or through a separately
    accepted local same-space equivalence, and a one-token substitution must
    pass the shared phonetic predicate. Every remaining substitution must be
    an evidenced source term.
    """
    if not book_term_evidence or repetition.get("is_hallucination") or truncation.get("is_truncated"):
        return []
    terms = book_term_evidence.get("terms") or {}
    operations = list(alignment.get("operations") or [])
    if alignment.get("missing_tokens") or alignment.get("extra_tokens"):
        return []
    candidate_operations = [
        op
        for op in operations
        if op.get("op") == "substitution"
        or (
            op.get("op") == "phonetic_equivalent"
            and (
                (
                    len(str(op.get("ref") or "").split()) == 1
                    and len(str(op.get("hyp") or "").split()) == 1
                )
                or (
                    len(str(op.get("hyp") or "").split()) == 2
                    and str(op.get("hyp") or "").split()[-1] == "is"
                )
            )
        )
    ]
    if not terms or not candidate_operations:
        return []

    allowed_context_ops = {
        "exact_match",
        "normalized_equivalent",
        "phonetic_equivalent",
        "ambiguous_equivalent",
        "identifier_wildcard",
        "substitution",
    }
    if any(op.get("op") not in allowed_context_ops for op in operations):
        return []

    accepted: List[Dict[str, Any]] = []
    for slot, operation in enumerate(operations):
        if operation not in candidate_operations:
            continue
        ref_surface = str(operation.get("ref") or "")
        hyp_surface = str(operation.get("hyp") or "")
        ref = _clean_context_word(ref_surface)
        hyp = _clean_context_word(hyp_surface)
        evidence = terms.get(ref)
        if (
            not ref
            or not hyp
            or evidence is None
            or " " in ref_surface
            or (
                " " in hyp_surface
                and not (
                    operation.get("op") == "phonetic_equivalent"
                    and hyp_surface.endswith(" is")
                )
            )
            or _is_placeholder_token(ref_surface)
            or _is_placeholder_token(hyp_surface)
            or _is_id_like_token(ref_surface)
            or _is_id_like_token(hyp_surface)
            or ref in NEGATIONS
            or hyp in NEGATIONS
            or _looks_numeric_word(ref)
            or _looks_numeric_word(hyp)
        ):
            return []
        if " " not in ref_surface and " " not in hyp_surface:
            if abs(len(ref) - len(hyp)) > 2:
                return []
            if not tokens_phonetically_equal(ref, hyp, phonetic_threshold):
                return []
        accepted.append(
            {
                "slot": slot,
                "kind": "book_term_equivalence",
                "expected": {"surface": ref_surface, "canonical": ref},
                "hypothesis": {"surface": hyp_surface, "canonical": hyp},
                "source_occurrences": int(evidence.get("occurrences") or 0),
                "capitalized_occurrences": int(evidence.get("capitalized_occurrences") or 0),
                "acronym_occurrences": int(evidence.get("acronym_occurrences") or 0),
                "noninitial_capitalized_occurrences": int(
                    evidence.get("noninitial_capitalized_occurrences") or 0
                ),
                "matched": True,
                "result": "accepted",
            }
        )
    return accepted


def _leading_protected_word_confirmation(
    ref_text: str,
    hyp_text: str,
    canon_lookup: Optional[dict] = None,
    alignment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Flag first-position mismatches on protected leading words.

    The flag supports two-stage review for short protected starters, especially
    negation words. It never changes PASS/FAIL; it only marks cases that should
    be handed to Stage 2 for confirmation. A boundary-aware first alignment
    operation supersedes the raw first-token check because it may preserve all
    protected letters across a compound split or fusion.

    Args:
        ref_text: Original reference text.
        hyp_text: ASR transcript text.
        canon_lookup: Optional compatibility canonicalization lookup.
        alignment: Final token alignment, when available.
    """
    if alignment:
        operations = alignment.get("operations") or []
        if operations and operations[0].get("op") == "boundary_resegmentation":
            return {
                "requires_second_stage_confirmation": False,
                "second_stage_confirmation_reason": "",
            }
    ref_norm, _ = normalize(ref_text, canon_lookup)
    hyp_norm, _ = normalize(hyp_text, canon_lookup)
    ref_first = next((token for token in ref_norm.split() if token), "")
    hyp_first = next((token for token in hyp_norm.split() if token), "")
    if not ref_first or not hyp_first or ref_first == hyp_first:
        return {
            "requires_second_stage_confirmation": False,
            "second_stage_confirmation_reason": "",
        }
    if ref_first in NEGATIONS or hyp_first in NEGATIONS:
        return {
            "requires_second_stage_confirmation": True,
            "second_stage_confirmation_reason": (
                f"First protected word mismatch: {ref_first} → {hyp_first}"
            ),
        }
    return {
        "requires_second_stage_confirmation": False,
        "second_stage_confirmation_reason": "",
    }


def _looks_numeric_word(token: str) -> bool:
    """Return True if token is numeric or a common English number/ordinal word."""
    if not token:
        return False
    if token.isdigit():
        return True
    if _ORDINAL_RE.match(token):
        return True
    number_words = {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
        "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
        "thousand", "first", "second", "third", "fourth", "fifth", "sixth",
        "seventh", "eighth", "ninth", "tenth", "twentieth", "thirtieth",
    }
    return token in number_words or token.endswith("ty") and token[:-2] in {
        "twen", "thir", "for", "fif", "six", "seven", "eigh", "nine",
    }


def explain_from_alignment(result: Dict[str, Any]) -> str:
    """Build a human-readable explanation from structured comparison fields."""
    ft = result.get("failure_type") or ""
    parts: List[str] = []
    if ft == "unexpected_suffix" and result.get("extra_tokens"):
        parts.append(f'Unexpected suffix: "{" ".join(result["extra_tokens"])}"')
    elif ft == "unexpected_prefix" and result.get("extra_tokens"):
        parts.append(f'Unexpected prefix: "{" ".join(result["extra_tokens"])}"')
    elif ft == "unexpected_internal_speech" and result.get("extra_tokens"):
        parts.append(f'Unexpected extra speech: "{" ".join(result["extra_tokens"])}"')
    if result.get("missing_tokens"):
        parts.append(f'Missing content: "{" ".join(result["missing_tokens"])}"')
    for sub in result.get("substitutions", [])[:5]:
        parts.append(f"substituted: '{sub['ref']}' → '{sub['hyp']}'")
    for cm in result.get("critical_mismatches", [])[:5]:
        if cm["type"] == "number":
            parts.append(f"Changed number: \"{cm['ref']}\" → \"{cm['hyp']}\"")
        elif cm["type"] == "identifier":
            parts.append(f"Changed identifier: \"{cm['ref']}\" → \"{cm['hyp']}\"")
        elif cm["type"] == "interjection":
            parts.append(f"Changed interjection: \"{cm['ref']}\" → \"{cm['hyp']}\"")
        elif cm["type"] in {"negation", "missing_negation"}:
            parts.append(f"Changed/missing negation: \"{cm['ref']}\" → \"{cm['hyp']}\"")
        elif cm["type"] == "missing_clause":
            parts.append(f"Missing content: \"{cm['ref']}\"")
    for rep in result.get("repetition_details", [])[:3]:
        parts.append(
            f"Unexpected repetition: \"{rep['pattern']}\" repeated "
            f"{rep['hyp_count']} times; expected {rep['ref_count']}"
        )
    if result.get("truncation_warning"):
        parts.append(result["truncation_warning"])
    if not parts:
        if result.get("classification") == "PASS":
            return ""
        return "minor spoken variation or low combined confidence"
    return "; ".join(parts)


def compare_spoken(
    ref_text: str,
    hyp_text: str,
    threshold: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
    canon_lookup: Optional[dict] = None,
    book_term_evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compare reference text to ASR hypothesis on spoken-content grounds.

    Returns structured scores, binary classification (PASS/FAIL), and details.
    """
    ref_text = strip_chatterbox_pause_tags(ref_text)
    terminal_diagnostic_equivalences: List[Dict[str, Any]] = []
    diagnostic_match = _ASR_TERMINAL_DIAGNOSTIC_RE.search(hyp_text or "")
    if diagnostic_match:
        # A terminal backend annotation is not spoken content when all prior
        # transcript words still undergo ordinary comparison.
        terminal_diagnostic_equivalences.append({
            "kind": "terminal_asr_diagnostic",
            "expected": "",
            "hypothesis": diagnostic_match.group(0).strip(),
        })
        hyp_text = (hyp_text or "")[:diagnostic_match.start()].rstrip()
    cfg = merge_validation_config(config)
    if threshold is not None:
        cfg["pass_threshold"] = threshold

    phonetic_thr = float(cfg.get("phonetic_match_threshold", 0.85))
    pre_raw_equivalence = _raw_formatting_equivalence(ref_text, hyp_text)
    if pre_raw_equivalence is None:
        pre_raw_equivalence = _raw_normalized_equivalence(
            ref_text,
            hyp_text,
            canon_lookup,
        )
    if pre_raw_equivalence is None:
        pre_raw_equivalence = _raw_reduced_pronoun_equivalence(ref_text, hyp_text)
    if pre_raw_equivalence is None:
        pre_raw_equivalence = _raw_call_sign_article_equivalence(ref_text, hyp_text)
    full_raw_equivalence = pre_raw_equivalence is not None
    if full_raw_equivalence:
        # Later raw repair rules are unnecessary once the full utterance has a
        # proven safe equivalence, and can otherwise cross-match repetitions.
        hyp_text = ref_text
    slash_name_equivalence, ref_slash_name_text = _raw_slash_name_equivalence(
        ref_text,
        hyp_text,
        canon_lookup,
    )
    if slash_name_equivalence is not None:
        # The helper proved whole-utterance normalized equality.
        ref_text = ref_slash_name_text
        hyp_text = ref_slash_name_text
    spoken_code_equivalence, ref_code_text = _raw_spoken_code_equivalence(
        ref_text,
        hyp_text,
    )
    early_raw_phrase_equivalence, hyp_pre_boundary_text = (
        _raw_albeit_resegmentation_equivalence(ref_code_text, hyp_text)
    )
    if full_raw_equivalence:
        # The complete spoken surfaces already agree.  Running a local raw
        # boundary rewrite afterward can compare overlapping equal tokens and
        # manufacture text (for example, C-I-C -> ci cic).
        compact_boundary_equivalences = []
        ref_boundary_text, hyp_boundary_text = ref_code_text, hyp_pre_boundary_text
    else:
        compact_boundary_equivalences, ref_boundary_text, hyp_boundary_text = (
            _raw_compact_boundary_surface_equivalences(
                ref_code_text, hyp_pre_boundary_text, phonetic_thr
            )
        )
    time_two_to_equivalence, ref_time_text, hyp_time_text = (
        _raw_time_two_to_equivalence(ref_boundary_text, hyp_boundary_text)
    )
    possessive_filler_equivalence, ref_possessive_filler_text, hyp_possessive_filler_text = (
        _raw_lexical_possessive_filler_equivalence(ref_time_text, hyp_time_text)
    )
    apostrophe_s_equivalences, ref_apostrophe_s_text, hyp_apostrophe_s_text = (
        _raw_apostrophe_s_surface_equivalence(
            ref_possessive_filler_text,
            hyp_possessive_filler_text,
        )
    )
    reduced_conjunction_equivalence, ref_conjunction_n_text = (
        _raw_lowercase_n_conjunction_equivalence(
            ref_apostrophe_s_text,
            hyp_apostrophe_s_text,
        )
    )
    fusion_equivalence, ref_fusion_text, hyp_fusion_text = (
        _raw_possessive_or_contraction_surface_fusion_equivalence(
            ref_conjunction_n_text,
            hyp_apostrophe_s_text,
        )
    )
    letter_sequence_equivalence, ref_letter_sequence_text, hyp_letter_sequence_text = (
        _raw_delimited_letter_sequence_equivalence(
            ref_fusion_text,
            hyp_fusion_text,
        )
    )
    acronym_render_equivalence, ref_acronym_render_text, hyp_acronym_render_text = (
        _raw_bare_two_letter_acronym_equivalence(
            ref_letter_sequence_text,
            hyp_letter_sequence_text,
        )
    )
    consonant_acronym_equivalence, ref_acronym_text, hyp_acronym_text = (
        _raw_consonant_acronym_equivalence(
            ref_acronym_render_text,
            hyp_acronym_render_text,
        )
    )
    contraction_equivalence, ref_contraction_text, hyp_contraction_text = (
        _raw_contraction_equivalence(ref_acronym_text, hyp_acronym_text)
    )
    conjunction_equivalence, ref_conjunction_text, hyp_conjunction_text = (
        _raw_single_conjunction_omission_equivalence(
            ref_contraction_text,
            hyp_contraction_text,
        )
    )
    late_raw_equivalence = _raw_normalized_equivalence(
        ref_conjunction_text,
        hyp_conjunction_text,
        canon_lookup,
    )
    if late_raw_equivalence is not None:
        hyp_conjunction_text = ref_conjunction_text
    book_term_tokens = set((book_term_evidence or {}).get("terms") or {})
    strict_ref_tokens = _strict_short_acronym_tokens(ref_text, book_term_evidence)
    ref_comparison_text = _strip_evidenced_book_term_possessives(
        ref_conjunction_text,
        book_term_tokens,
    )
    split_word_number_equivalence, ref_split_text, hyp_split_text = (
        _raw_split_word_number_equivalence(ref_comparison_text, hyp_conjunction_text)
    )
    raw_phrase_equivalence, hyp_comparison_text = _raw_albeit_resegmentation_equivalence(
        ref_split_text,
        hyp_split_text,
        strict_ref_tokens=strict_ref_tokens,
    )
    ref_comparison_text = ref_split_text
    ref_norm, ref_ids = normalize(ref_comparison_text, canon_lookup)
    hyp_norm, hyp_ids = normalize(hyp_comparison_text, canon_lookup)
    stutter_equivalence, hyp_norm = _normalized_single_adjacent_stutter_equivalence(
        ref_norm,
        hyp_norm,
        phonetic_thr,
        strict_ref_tokens=strict_ref_tokens,
    )

    # Matching placeholders are wildcard slots. Collapse only adjacent numeric
    # slots so digit-by-digit book text and a compact ASR code share one span,
    # while preserving all prose on either side of that identifier.
    filtered_ref = _collapse_identifier_sequences(ref_norm)
    filtered_hyp = _collapse_identifier_sequences(hyp_norm)

    ref_tokens = filtered_ref.split() if filtered_ref else []
    hyp_tokens = filtered_hyp.split() if filtered_hyp else []

    hyp_lexical_possessives = _lexical_possessive_bases(hyp_text)
    alignment = align_tokens(
        ref_tokens,
        hyp_tokens,
        phonetic_thr,
        strict_ref_tokens=strict_ref_tokens,
        book_term_tokens=book_term_tokens,
        hyp_lexical_possessives=hyp_lexical_possessives,
    )
    alignment = _accept_local_same_space_substitutions(
        alignment,
        strict_ref_tokens=strict_ref_tokens,
    )
    boundary_resegmentation_equivalences = [
        {
            "kind": "boundary_resegmentation",
            "expected": str(operation.get("ref") or ""),
            "hypothesis": str(operation.get("hyp") or ""),
            "direction": str(operation.get("direction") or ""),
            "method": str(operation.get("boundary_method") or ""),
        }
        for operation in alignment["operations"]
        if operation.get("op") == "boundary_resegmentation"
    ]
    filtered_hyp = _render_aligned_hypothesis(alignment["operations"])
    # Count repeats on aligned tokens so hyphen splits of a book name are not
    # a new phrase; unmatched hyp words stay so extra copies still fail.
    repetition = detect_repetition(
        ref_tokens,
        _aligned_repetition_tokens(alignment["operations"]),
        int(cfg.get("repeat_phrase_max_length", 6)),
    )
    truncation = detect_truncation(ref_tokens, hyp_tokens, alignment=alignment)
    accepted_list_label_equivalences = _accepted_list_label_equivalences(
        ref_text,
        hyp_text,
        phonetic_thr,
    )
    accepted_book_term_equivalences = _accepted_book_term_equivalences(
        alignment,
        book_term_evidence,
        repetition,
        truncation,
        phonetic_thr,
    )
    accepted_list_pairs = {
        (
            item["expected"]["canonical"],
            item["hypothesis"]["canonical"],
        )
        for item in accepted_list_label_equivalences
    }
    accepted_book_pairs = {
        (
            item["expected"]["canonical"],
            item["hypothesis"]["canonical"],
        )
        for item in accepted_book_term_equivalences
    }
    filtered_alignment = dict(alignment)
    filtered_alignment["substitutions"] = [
        sub for sub in alignment["substitutions"]
        if (
            _clean_context_word(sub.get("ref") or ""),
            _clean_context_word(sub.get("hyp") or ""),
        ) not in accepted_list_pairs | accepted_book_pairs
    ]
    accepted_ambiguous_equivalences = [
        {
            "slot": idx,
            "kind": "protected_short_word",
            "expected": {
                "surface": str(op.get("ref") or ""),
                "canonical": _clean_context_word(str(op.get("ref") or "")),
                "components": [_clean_context_word(str(op.get("ref") or ""))],
            },
            "hypothesis": {
                "surface": str(op.get("hyp") or ""),
                "canonical": _clean_context_word(str(op.get("hyp") or "")),
                "components": [_clean_context_word(str(op.get("hyp") or ""))],
            },
            "matched": True,
            "result": "accepted",
        }
        for idx, op in enumerate(alignment["operations"])
        if op["op"] == "ambiguous_equivalent"
    ]
    accepted_ambiguous_equivalences.extend(
        _merged_split_phrase_ambiguities(alignment, phonetic_thr)
    )
    second_stage_confirmation = _leading_protected_word_confirmation(
        ref_text,
        hyp_text,
        canon_lookup,
        alignment=alignment,
    )

    content_extra, extra_failure = _classify_extra_speech(
        alignment["operations"], len(ref_tokens), cfg
    )
    if any(_contains_negation_text(token) for token in alignment["extra_tokens"]):
        extra_failure = extra_failure or "changed_negation"
    critical = _critical_mismatches(filtered_alignment, ref_tokens, hyp_tokens, phonetic_thr)
    wildcard_slots = set()
    for operation in alignment["operations"]:
        if operation.get("op") != "identifier_wildcard":
            continue
        slot = _placeholder_slot_index(operation.get("ref") or "")
        if slot is not None:
            wildcard_slots.add(slot)
    identifier_comparisons = _identifier_comparisons(ref_ids, hyp_ids, wildcard_slots)
    review_thr = float(cfg.get("review_threshold", 0.75))
    content_missing = [tok for tok in alignment["missing_tokens"] if _is_content_word(tok)]
    content_substitutions = [
        sub for sub in filtered_alignment["substitutions"]
        if _is_content_word(sub.get("ref") or "") or _is_content_word(sub.get("hyp") or "")
    ]
    # Option 2: exactly one content-word swap with matching shell → soft pass.
    lone_content_soft = _soft_accept_lone_content_substitution(
        filtered_alignment["substitutions"],
        alignment["missing_tokens"],
        alignment["extra_tokens"],
        critical,
        repetition,
        alignment["operations"],
    )
    if lone_content_soft is not None:
        soft_pair = (
            _clean_context_word(lone_content_soft["ref"]),
            _clean_context_word(lone_content_soft["hyp"]),
        )
        filtered_alignment["substitutions"] = [
            sub
            for sub in filtered_alignment["substitutions"]
            if (
                _clean_context_word(sub.get("ref") or ""),
                _clean_context_word(sub.get("hyp") or ""),
            )
            != soft_pair
        ]
        content_substitutions = [
            sub
            for sub in content_substitutions
            if (
                _clean_context_word(sub.get("ref") or ""),
                _clean_context_word(sub.get("hyp") or ""),
            )
            != soft_pair
        ]
    context_repaired_short_soft = _soft_accept_context_repaired_short_substitution(
        filtered_alignment["substitutions"],
        alignment["missing_tokens"],
        alignment["extra_tokens"],
        critical,
        repetition,
    )
    if context_repaired_short_soft is not None:
        context_pair = (
            _clean_context_word(context_repaired_short_soft["ref"]),
            _clean_context_word(context_repaired_short_soft["hyp"]),
        )
        filtered_alignment["substitutions"] = [
            sub
            for sub in filtered_alignment["substitutions"]
            if (
                _clean_context_word(sub.get("ref") or ""),
                _clean_context_word(sub.get("hyp") or ""),
            )
            != context_pair
        ]
        content_substitutions = [
            sub
            for sub in content_substitutions
            if (
                _clean_context_word(sub.get("ref") or ""),
                _clean_context_word(sub.get("hyp") or ""),
            )
            != context_pair
        ]
    by_what_one_soft = _soft_accept_by_what_one_question(
        filtered_alignment["substitutions"],
        alignment["missing_tokens"],
        alignment["extra_tokens"],
        critical,
        repetition,
        alignment["operations"],
    )
    if by_what_one_soft is not None:
        question_pair = (
            _clean_context_word(by_what_one_soft["ref"]),
            _clean_context_word(by_what_one_soft["hyp"]),
        )
        filtered_alignment["substitutions"] = [
            sub
            for sub in filtered_alignment["substitutions"]
            if (
                _clean_context_word(sub.get("ref") or ""),
                _clean_context_word(sub.get("hyp") or ""),
            )
            != question_pair
        ]
        content_substitutions = [
            sub
            for sub in content_substitutions
            if (
                _clean_context_word(sub.get("ref") or ""),
                _clean_context_word(sub.get("hyp") or ""),
            )
            != question_pair
        ]
    high_coverage_fuzzy_soft = _soft_accept_high_coverage_listener_fuzzy(
        filtered_alignment["substitutions"],
        alignment["missing_tokens"],
        alignment["extra_tokens"],
        critical,
        repetition,
        coverage=float(alignment["coverage_score"]),
        phonetic=float(alignment["phonetic_score"]),
        ref_tokens=ref_tokens,
        ref_ids=ref_ids,
        hyp_ids=hyp_ids,
    )
    if high_coverage_fuzzy_soft is not None:
        filtered_alignment["substitutions"] = []
        content_missing = []
        content_substitutions = []
    accepted_equivalences = []
    for operation in alignment["operations"]:
        if (
            operation["op"] != "phonetic_equivalent"
            or " " in str(operation.get("ref") or "")
            or " " in str(operation.get("hyp") or "")
        ):
            continue
        equivalence = {
            "ref": str(operation["ref"]),
            "hyp": str(operation["hyp"]),
        }
        if operation.get("specialized_rule"):
            equivalence["kind"] = str(operation["specialized_rule"])
        accepted_equivalences.append(equivalence)
    accepted_equivalences.extend(boundary_resegmentation_equivalences)
    if lone_content_soft is not None:
        accepted_equivalences.append(
            {
                "ref": lone_content_soft["ref"],
                "hyp": lone_content_soft["hyp"],
                "kind": "single_content_token_soft",
            }
        )
    if context_repaired_short_soft is not None:
        accepted_equivalences.append(
            {
                "ref": context_repaired_short_soft["ref"],
                "hyp": context_repaired_short_soft["hyp"],
                "kind": "context_repaired_short_token",
            }
        )
    if by_what_one_soft is not None:
        accepted_equivalences.append(
            {
                "ref": by_what_one_soft["ref"],
                "hyp": by_what_one_soft["hyp"],
                "kind": "context_question_word_fuzzy",
            }
        )
    if high_coverage_fuzzy_soft is not None:
        accepted_equivalences.extend(
            {
                "ref": item["ref"],
                "hyp": item["hyp"],
                "kind": "high_coverage_listener_fuzzy",
            }
            for item in high_coverage_fuzzy_soft
        )
    accepted_equivalences.extend(
        {
            "ref": item["expected"]["canonical"],
            "hyp": item["hypothesis"]["canonical"],
            "kind": "book_term_equivalence",
        }
        for item in accepted_book_term_equivalences
    )
    accepted_phrase_equivalences = [
        item
        for item in (
            *terminal_diagnostic_equivalences,
            pre_raw_equivalence,
            slash_name_equivalence,
            spoken_code_equivalence,
            *compact_boundary_equivalences,
            possessive_filler_equivalence,
            *apostrophe_s_equivalences,
            reduced_conjunction_equivalence,
            fusion_equivalence,
            letter_sequence_equivalence,
            acronym_render_equivalence,
            consonant_acronym_equivalence,
            time_two_to_equivalence,
            contraction_equivalence,
            conjunction_equivalence,
            late_raw_equivalence,
            split_word_number_equivalence,
            early_raw_phrase_equivalence,
            raw_phrase_equivalence,
            stutter_equivalence,
            *boundary_resegmentation_equivalences,
        )
        if item is not None
    ]

    coverage = float(alignment["coverage_score"])
    phonetic = float(alignment["phonetic_score"])
    if accepted_list_label_equivalences or accepted_book_term_equivalences:
        coverage = 1.0
        phonetic = 1.0
    # Lone content soft-accepts are policy matches; do not leave a low combined
    # score that fails after hard rules already cleared.
    if (
        lone_content_soft is not None
        or context_repaired_short_soft is not None
        or by_what_one_soft is not None
        or high_coverage_fuzzy_soft is not None
    ):
        coverage = max(coverage, 0.95)
        phonetic = max(phonetic, 0.95)

    # Penalties
    extra_penalty = min(1.0, len(content_extra) * 0.15)
    missing_penalty = min(1.0, len(content_missing) * 0.2)
    sub_penalty = min(1.0, len(content_substitutions) * 0.15)
    rep_penalty = 0.5 if repetition.get("severe") else (0.2 if repetition.get("is_hallucination") else 0.0)
    critical_penalty = 1.0 if critical else 0.0

    combined = max(
        0.0,
        min(1.0, 0.55 * coverage + 0.45 * phonetic)
        - extra_penalty
        - missing_penalty
        - sub_penalty
        - rep_penalty
        - critical_penalty * 0.5,
    )
    # If pure match
    if (
        not alignment["extra_tokens"]
        and not alignment["missing_tokens"]
        and not filtered_alignment["substitutions"]
        and not critical
        and not repetition.get("severe")
    ):
        combined = max(combined, min(coverage, phonetic))

    # Character fallback only as secondary signal when tokens empty-equal
    if not ref_tokens and not hyp_tokens:
        combined = 1.0
        coverage = 1.0
        phonetic = 1.0
    elif ref_tokens and not hyp_tokens:
        combined = 0.0

    failure_type = None
    hard_fail = False

    if truncation["is_truncated"]:
        hard_fail = True
        failure_type = "truncation"
    if repetition.get("severe"):
        hard_fail = True
        failure_type = failure_type or "unexpected_repetition"
    if extra_failure:
        hard_fail = True
        failure_type = failure_type or extra_failure
    if (
        (alignment["missing_tokens"] or alignment["extra_tokens"])
        and high_coverage_fuzzy_soft is None
    ):
        # Content gaps still hard-fail. Function/filler-only gaps (articles,
        # auxiliaries, soft conjunctions) are common ASR drops of correct audio
        # and must not alone force regeneration when coverage stays high.
        # Function/filler-only gaps (the/a/have/it/and) are common ASR drops of
        # correct audio. Soft-pass when no content word is missing/extra/subbed.
        # Coordinating ``or`` between content words stays hard (walk or run).
        _soft_function_gap = frozenset({
            "the", "a", "an", "to", "of", "for", "in", "on", "at", "as", "by",
            "with", "from", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "can", "may", "might", "must",
            "it", "that", "and", "who", "whom", "whose", "which",
        })
        gap_tokens = [
            _clean_context_word(tok)
            for tok in list(alignment["missing_tokens"]) + list(alignment["extra_tokens"])
            if _clean_context_word(tok)
        ]
        function_only_unmatched = (
            not content_missing
            and not content_extra
            and not content_substitutions
            and not critical
            and coverage >= 0.75
            and gap_tokens
            and all(tok in _soft_function_gap for tok in gap_tokens)
        )
        if not function_only_unmatched:
            hard_fail = True
            failure_type = failure_type or (
                "missing_speech" if alignment["missing_tokens"] else "unexpected_internal_speech"
            )
        elif function_only_unmatched:
            # Do not leave a low combined score as a soft fail after clearing hard rules.
            coverage = max(coverage, 0.92)
            phonetic = max(phonetic, 0.92)
    if critical:
        hard_fail = True
        ctype = critical[0]["type"]
        failure_type = failure_type or (
            "changed_negation" if "negation" in ctype
            else "changed_number" if "number" in ctype
            else "changed_numeric_structure" if ctype == "numeric_structure"
            else "changed_identifier" if "identifier" in ctype
            else "missing_speech"
        )
    if content_substitutions:
        hard_fail = True
        failure_type = failure_type or "unaccepted_substitution"
    if coverage < float(cfg.get("min_reference_coverage", 0.90)) and len(ref_tokens) >= 2:
        if coverage < 0.5:
            hard_fail = True
            failure_type = failure_type or "low_coverage"

    # Foreign / garbage heuristic
    if hyp_tokens and coverage < 0.3 and phonetic < 0.3 and len(hyp_tokens) >= 3:
        hard_fail = True
        failure_type = failure_type or "unrelated_speech"

    pass_thr = float(cfg.get("pass_threshold", 0.88))
    minor_clean = (
        not alignment["extra_tokens"]
        and not content_missing
        and not critical
        and not repetition.get("is_hallucination")
        and not extra_failure
        and coverage >= pass_thr
        and phonetic >= phonetic_thr
    )
    # Allow only soft substitutions (phonetic already counted as match)
    only_soft = (
        not content_missing
        and not content_extra
        and not critical
        and not repetition.get("severe")
        and not content_substitutions
        and coverage >= 0.85
        and phonetic >= 0.85
    )

    if hard_fail:
        classification = "FAIL"
        passed = False
    elif (
        context_repaired_short_soft is not None
        or by_what_one_soft is not None
        or high_coverage_fuzzy_soft is not None
    ):
        # Keep listener-review evidence visible while skipping pointless retry.
        classification = "FUZZY_ACCEPT"
        passed = True
        combined = max(combined, review_thr)
    elif combined >= pass_thr or minor_clean:
        classification = "PASS"
        passed = True
    elif only_soft and combined >= review_thr:
        classification = "PASS"
        passed = True
    elif lone_content_soft is not None:
        # Policy: one content-token ASR confusable with matching shell is a pass.
        classification = "PASS"
        passed = True
        combined = max(combined, review_thr)
    else:
        classification = "FAIL"
        passed = False
        failure_type = failure_type or "low_score"

    # Backward-compatible minor_mismatch flag for close phonetic-only diffs
    minor_mismatch = (
        classification == "PASS"
        and any(op["op"] == "phonetic_equivalent" for op in alignment["operations"])
        and not filtered_alignment["substitutions"]
        and not alignment["missing_tokens"]
        and not content_extra
    )

    truncation_warning = ""
    if truncation["is_truncated"]:
        truncation_warning = (
            f"Possible truncation: {truncation['hyp_words']} words "
            f"vs {truncation['ref_words']} expected ({truncation['ratio']:.1%})"
        )
    hallucination_warning = ""
    if repetition.get("is_hallucination"):
        hallucination_warning = (
            f"Hallucination detected: '{repetition.get('pattern')}' "
            f"repeated {repetition.get('count')} times in hypothesis "
            f"(vs {repetition.get('ref_count')} in reference, "
            f"type: {repetition.get('type')}, severity: {repetition.get('severity')})"
        )

    out: Dict[str, Any] = {
        "comparison_policy_version": "boundary-resegmentation-v2",
        "passed": passed,
        "classification": classification,
        "failure_type": failure_type or "",
        "score": combined,
        "prose_score": combined,
        "id_score": _identifier_score(identifier_comparisons),
        "requires_second_stage_confirmation": second_stage_confirmation["requires_second_stage_confirmation"],
        "second_stage_confirmation_reason": second_stage_confirmation["second_stage_confirmation_reason"],
        "coverage_score": coverage,
        "phonetic_score": phonetic,
        "ref_normalized": filtered_ref,
        "hyp_normalized": filtered_hyp,
        "extra_tokens": list(alignment["extra_tokens"]),
        "missing_tokens": list(alignment["missing_tokens"]),
        "substitutions": list(filtered_alignment["substitutions"]),
        "accepted_equivalences": accepted_equivalences,
        "accepted_phrase_equivalences": accepted_phrase_equivalences,
        "accepted_ambiguous_equivalences": accepted_ambiguous_equivalences,
        "accepted_list_label_equivalences": accepted_list_label_equivalences,
        "accepted_book_term_equivalences": accepted_book_term_equivalences,
        "critical_mismatches": critical,
        "identifier_comparisons": identifier_comparisons,
        "repetition_details": repetition.get("details", []),
        "alignment_operations": alignment["operations"],
        "minor_mismatch": minor_mismatch,
        "hallucination_warning": hallucination_warning,
        "truncation_warning": truncation_warning,
        "ref_id_keys": [c["canonical_id"] for c in ref_ids],
        "hyp_id_keys": [c["canonical_id"] for c in hyp_ids],
    }
    out["explanation"] = explain_from_alignment(out) if not passed else ""
    return chatterbox_compare_fields(out)


_CHATTERBOX_PAUSE_TAG_RE = re.compile(r"\[pause:\d+ms\]", re.IGNORECASE)


def strip_chatterbox_pause_tags(text: str) -> str:
    """Remove Chatterbox [pause:Nms] tags so they are not scored as spoken words.

    Pocket already strips [1.0s] markers. This repo inserts [pause:150ms] instead.

    Args:
        text: Source chunk text that may contain pause tags.

    Returns:
        Text with pause tags replaced by spaces.
    """
    return _CHATTERBOX_PAUSE_TAG_RE.sub(" ", str(text or ""))


def normalize_spoken(text: str, canon_lookup: Optional[dict] = None) -> str:
    """Return Pocket-normalized comparison text for Chatterbox tests and reports.

    Args:
        text: Source or ASR transcript.
        canon_lookup: Unused Pocket API compatibility argument.

    Returns:
        Normalized comparison string without identifier metadata.
    """
    return normalize(strip_chatterbox_pause_tags(text), canon_lookup)[0]


def chatterbox_compare_fields(compared: Dict[str, Any]) -> Dict[str, Any]:
    """Map Pocket compare_spoken keys onto Chatterbox ASR report fields.

    Args:
        compared: Result dict from Pocket compare_spoken.

    Returns:
        Same dict with missing_words, extra_words, extra_content_words filled.
    """
    extra = list(compared.get("extra_tokens") or compared.get("extra_words") or [])
    missing = list(compared.get("missing_tokens") or compared.get("missing_words") or [])
    compared["extra_words"] = extra
    compared["missing_words"] = missing
    compared["extra_content_words"] = compared.get("extra_content_words", len(extra))
    if compared.get("passed") and not compared.get("explanation"):
        score = float(compared.get("score") or 0.0)
        compared["explanation"] = f"Heard the intended words (score {score:.2f})."
    return compared
