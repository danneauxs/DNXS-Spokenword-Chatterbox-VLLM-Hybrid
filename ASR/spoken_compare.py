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
    )
except ImportError:  # Supports direct execution from the ASR directory.
    from numeric_slots import (
        NUMERIC_SLOT,
        replace_spoken_number_spans,
        separate_attached_measure_units,
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
_UPPER_ROMAN_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])([IVXLCDM]{2,})(?![A-Za-z0-9])")
_BOOK_TERM_RE = re.compile(r"\b[A-Za-z][A-Za-z']*\b")
_ACRONYM_POSSESSIVE_RE = re.compile(r"\b([A-Z]{2,})['\u2019]s\b")
_RAW_SHORT_ACRONYM_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z]{2,4})(?![A-Za-z0-9])")

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


def _default_spoken_rules() -> Dict[str, Any]:
    """Return built-in spoken-rule defaults used when file is absent."""
    return {
        "spelling_variants": dict(SPELLING_VARIANTS),
        "phonetic_map": dict(_PHONETIC_MAP),
        "letter_name_aliases": dict(_LETTER_NAME_TO_LETTER),
        "negations": sorted(NEGATIONS),
        "blocked_phonetic_pairs": [sorted(pair) for pair in _BLOCKED_PHONETIC_PAIRS],
        "placeholder_prefixes": list(_PLACEHOLDER_PREFIXES),
    }


def _merge_spoken_rules(raw_rules: Any) -> Dict[str, Any]:
    """Merge file-backed overrides onto built-in spoken defaults."""
    rules = _default_spoken_rules()
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

    SPELLING_VARIANTS = dict(rules["spelling_variants"])
    _PHONETIC_MAP = dict(rules["phonetic_map"])
    _LETTER_NAME_TO_LETTER = dict(rules["letter_name_aliases"])
    NEGATIONS = frozenset(rules["negations"])
    _BLOCKED_PHONETIC_PAIRS = {frozenset(pair) for pair in rules["blocked_phonetic_pairs"]}
    _PLACEHOLDER_PREFIXES = tuple(rules["placeholder_prefixes"])


def reload_spoken_rules() -> Dict[str, Any]:
    """Reload on-disk rules and refresh comparator lookup tables."""
    try:
        raw = json.loads(_RULES_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        raw = None
    rules = _merge_spoken_rules(raw)
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


def _replace_typed_identifier_spans(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Replace typed identifier spans with slots while keeping prose intact."""
    if not text:
        return "", []

    tokens = text.split()
    out_tokens: List[str] = []
    id_contexts: List[Dict[str, Any]] = []
    i = 0
    while i < len(tokens):
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


def _raw_apostrophe_s_surface_equivalence(
    ref_text: str,
    hyp_text: str,
) -> Tuple[Optional[Dict[str, Any]], str, str]:
    """Rewrite one source apostrophe-s token matched by equal raw neighbors.

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
            if hyp_clean not in {base, f"{base}s"}:
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
    if len(candidates) != 1:
        return None, ref_text, hyp_text
    evidence, ref_index, replacement = candidates[0]
    start, end = ref_matches[ref_index].span()
    return evidence, f"{ref_text[:start]}{replacement}{ref_text[end:]}", hyp_text


def _raw_exact_surface_word_fusion_equivalence(
    ref_text: str,
    hyp_text: str,
) -> Tuple[Optional[Dict[str, Any]], str, str]:
    """Accept one exact prose word-boundary fusion before normalization.

    ASR can emit two or three adjacent spoken words as one token, especially
    around possessives and contractions.  This accepts only a sole raw-token
    difference whose letters agree exactly after spaces, apostrophes, and
    hyphens are removed.  Numeric, identifier-like, and initialism spans stay
    protected because word boundaries are significant in those forms.
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


def _raw_resegmented_phrase_equivalence(
    ref_text: str,
    hyp_text: str,
    strict_ref_tokens: Optional[Set[str]] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return a safe one-to-many raw phrase rewrite for ASR word splitting.

    The source and hypothesis must be identical outside one phrase. The source
    side is exactly one ordinary word, the ASR side is two or three words, and
    their joined forms stay within the approved two-character spoken-space
    bound. This repairs forms such as ``Albeit`` read by ASR as
    ``I'll be at`` without permitting added prose, identifiers, numbers, or
    negation changes.
    """
    difference = _raw_token_difference(ref_text, hyp_text)
    if difference is None:
        return None, hyp_text
    ref_matches, hyp_matches, prefix, ref_end, hyp_end = difference
    ref_tokens = [match.group(0).lower().replace("\u2019", "'") for match in ref_matches]
    hyp_tokens = [match.group(0).lower().replace("\u2019", "'") for match in hyp_matches]
    ref_span = ref_tokens[prefix:ref_end]
    hyp_span = hyp_tokens[prefix:hyp_end]
    if len(ref_span) != 1 or len(hyp_span) not in {2, 3}:
        return None, hyp_text

    ref_word = _clean_context_word(ref_span[0])
    hyp_words = [_clean_context_word(token) for token in hyp_span]
    hyp_joined = "".join(hyp_words)
    ref_span_normalized, _ = normalize(ref_matches[prefix].group(0))
    hyp_span_normalized, _ = normalize(hyp_text[
        hyp_matches[prefix].start():hyp_matches[hyp_end - 1].end()
    ])
    if (
        not ref_word
        or not all(hyp_words)
        or len(ref_word) < 4
        or len(hyp_joined) < 4
        or (strict_ref_tokens and ref_word in strict_ref_tokens)
        or ref_word in NEGATIONS
        or any(word in NEGATIONS for word in hyp_words)
        or any(token in NEGATIONS for token in ref_span_normalized.split())
        or any(token in NEGATIONS for token in hyp_span_normalized.split())
        or _is_id_like_token(ref_span[0])
        or any(_is_id_like_token(token) for token in hyp_span)
        or _looks_numeric_word(ref_word)
        or any(_looks_numeric_word(word) for word in hyp_words)
        or abs(len(ref_word) - len(hyp_joined)) > 2
    ):
        return None, hyp_text

    start = hyp_matches[prefix].start()
    end = hyp_matches[hyp_end - 1].end()
    rewritten_hyp = f"{hyp_text[:start]}{ref_matches[prefix].group(0)}{hyp_text[end:]}"
    return {
        "kind": "raw_phrase_resegmentation",
        "expected": ref_matches[prefix].group(0),
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
    def replace(match: re.Match) -> str:
        """Keep non-Roman candidates intact while converting valid numeral tokens."""
        value = _roman_to_int_token(match.group(1))
        return str(value) if value is not None else match.group(1)

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


def _multi_token_name_match(ref: str, joined_hyp: str, threshold: float = 0.85) -> bool:
    """
    Match a long reference name against concatenated hypothesis tokens.

    Requires similar length so short words never absorb trailing content.
    """
    if len(ref) < 5 or len(joined_hyp) < 4:
        return False
    # Length must be within ~40% to absorb trailing "thank you" etc.
    longer = max(len(ref), len(joined_hyp))
    shorter = min(len(ref), len(joined_hyp))
    if shorter / longer < 0.6:
        return False
    if tokens_phonetically_equal(ref, joined_hyp, threshold):
        return True
    if fuzz.ratio(ref, joined_hyp) >= 72:
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

    Returns: exact | normalized | phonetic | none.

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


def _local_joined_span_equivalent(
    ref_span: Sequence[str],
    hyp_span: Sequence[str],
    strict_ref_tokens: Optional[Set[str]] = None,
) -> bool:
    """Accept one-to-two or two-to-one ASR resegmentation with equal space.

    The rule joins each bounded span and accepts only ordinary prose whose
    joined spellings differ by at most two characters. A source negation never
    enters this rule. A hypothesis ``no`` may participate only as part of a
    two-word rendering of one non-negated source word, such as ``nowhere`` to
    ``no way``.
    """
    if len(ref_span) == len(hyp_span) or not ref_span or not hyp_span:
        return False
    ref_words = [_clean_context_word(token) for token in ref_span]
    hyp_words = [_clean_context_word(token) for token in hyp_span]
    if not all(ref_words) or not all(hyp_words):
        return False
    if strict_ref_tokens and any(token in strict_ref_tokens for token in ref_words):
        return False
    article_compound = (
        len(ref_words) == 2
        and len(hyp_words) == 1
        and ref_words[0] == "a"
        and hyp_words[0] == f"a{ref_words[1]}"
    )
    # Exact joins such as ``in to`` ↔ ``into`` may use function words; other
    # multi-token function spans stay rejected so soft prepositions cannot hide.
    exact_function_join = "".join(ref_words) == "".join(hyp_words)
    if (
        len(ref_words) > 1
        and not all(_is_content_word(token) for token in ref_words)
        and not article_compound
        and not exact_function_join
    ):
        return False
    protected_tokens = [*ref_span, *hyp_span]
    if any(
        _is_placeholder_token(token)
        or _is_id_like_token(token)
        or _looks_numeric_word(_clean_context_word(token))
        for token in protected_tokens
    ):
        return False
    if any(token in NEGATIONS for token in ref_words):
        return False
    if any(token in NEGATIONS for token in hyp_words):
        if not (len(ref_words) == 1 and len(hyp_words) == 2):
            return False
    ref_joined = "".join(ref_words)
    hyp_joined = "".join(hyp_words)
    if len(hyp_words) > 1 and hyp_words[0] in FUNCTION_WORDS and hyp_words[0] != "no":
        # A source word may be split into a real phrase (Maiwand → my wand).
        # Permit that only when their joined sounds agree, never by length alone.
        return (
            len(ref_words) == 1
            and len(ref_joined) >= 4
            and tokens_phonetically_equal(ref_joined, hyp_joined)
        )
    return min(len(ref_joined), len(hyp_joined)) >= 3 and abs(
        len(ref_joined) - len(hyp_joined)
    ) <= 2


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
        if operation["op"] in {"exact_match", "normalized_equivalent", "phonetic_equivalent"}:
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


def _letter_join_equivalent(
    left_tokens: Sequence[str],
    right_tokens: Sequence[str],
) -> bool:
    """Return True when joined alnum letters of two spans are near-identical.

    Covers ASR fusing ``for Carthum`` → ``Forcatham`` or splitting ``Thulla``
    → ``the la`` without requiring dictionary names.
    """
    def join_letters(tokens: Sequence[str]) -> str:
        """Joins a sequence of strings into one after cleaning each."""
        return "".join(_clean_context_word(token) for token in tokens)

    left = join_letters(left_tokens)
    right = join_letters(right_tokens)
    if not left or not right:
        return False
    if left == right:
        return True
    if abs(len(left) - len(right)) > max(2, len(left) // 5):
        return False
    return fuzz.ratio(left, right) >= 82


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
    # dp cost; match=0, sub=1, ins/del=1
    INF = 10**9
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    bt: List[List[Optional[str]]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = "ins"

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
            candidates = [
                (dp[i - 1][j - 1] + match_cost, "match" if eq != "none" else "sub", eq),
                (dp[i - 1][j] + 1, "del", "none"),
                (dp[i][j - 1] + 1, "ins", "none"),
            ]
            ref_tok = ref_tokens[i - 1]
            if _is_placeholder_token(ref_tok):
                for span_len in _identifier_wildcard_span_lengths(hyp_tokens, j):
                    candidates.append(
                        (dp[i - 1][j - span_len], f"wildcard{span_len}", "wildcard")
                    )
            # Copula homophones: they are ↔ their, we are ↔ were, …
            if i >= 2 and j >= 1 and _copula_homophone_span_match(
                ref_tokens[i - 2:i],
                [hyp_tokens[j - 1]],
            ):
                candidates.append((dp[i - 2][j - 1], "copula_ref2", "phonetic"))
            if i >= 1 and j >= 2 and _copula_homophone_span_match(
                [ref_tok],
                hyp_tokens[j - 2:j],
            ):
                candidates.append((dp[i - 1][j - 2], "copula_hyp2", "phonetic"))
            # Letter-join fusion: for Carthum ↔ Forcatham, Thulla ↔ the la.
            # Never fuse across negation (not able ↛ notable).
            if (
                i >= 2
                and j >= 1
                and not _contains_negation_token(ref_tokens[i - 2:i])
                and not _contains_negation_token([hyp_tokens[j - 1]])
                and _letter_join_equivalent(ref_tokens[i - 2:i], [hyp_tokens[j - 1]])
            ):
                candidates.append((dp[i - 2][j - 1], "letters_ref2", "phonetic"))
            if (
                i >= 1
                and j >= 2
                and not _contains_negation_token([ref_tok])
                and not _contains_negation_token(hyp_tokens[j - 2:j])
                and _letter_join_equivalent([ref_tok], hyp_tokens[j - 2:j])
            ):
                candidates.append((dp[i - 1][j - 2], "letters_hyp2", "phonetic"))
            # Multi-token hyp packs one long name only when joined length is similar
            if (
                j >= 2
                and not (strict_ref_tokens and _clean_context_word(ref_tok) in strict_ref_tokens)
                and len(ref_tok) >= 5
                and _is_content_word(ref_tok)
                and all(_is_content_word(token) for token in hyp_tokens[j - 2:j])
                and not _is_placeholder_token(ref_tok)
                and not any(_is_placeholder_token(tok) for tok in hyp_tokens[j - 2:j])
                and not _contains_negation_token([ref_tok, *hyp_tokens[j - 2:j]])
            ):
                joined = hyp_tokens[j - 2] + hyp_tokens[j - 1]
                if _multi_token_name_match(ref_tok, joined, phonetic_threshold):
                    candidates.append((dp[i - 1][j - 2], "match2", "phonetic"))
            if j >= 2 and _local_joined_span_equivalent(
                [ref_tok],
                hyp_tokens[j - 2:j],
                strict_ref_tokens=strict_ref_tokens,
            ):
                candidates.append((dp[i - 1][j - 2], "local_hyp2", "phonetic"))
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
                candidates.append((dp[i - 1][j - 2], "book_possessive_hyp2", "phonetic"))
            if (
                j >= 3
                and not (strict_ref_tokens and _clean_context_word(ref_tok) in strict_ref_tokens)
                and len(ref_tok) >= 6
                and _is_content_word(ref_tok)
                and all(_is_content_word(token) for token in hyp_tokens[j - 3:j])
                and not _is_placeholder_token(ref_tok)
                and not any(_is_placeholder_token(tok) for tok in hyp_tokens[j - 3:j])
                and not _contains_negation_token([ref_tok, *hyp_tokens[j - 3:j]])
            ):
                joined3 = "".join(hyp_tokens[j - 3:j])
                if _multi_token_name_match(ref_tok, joined3, phonetic_threshold):
                    candidates.append((dp[i - 1][j - 3], "match3", "phonetic"))
            # One hypothesis token can join two reference words without loss.
            if i >= 2:
                parts = ref_tokens[i - 2:i]
                if _local_joined_span_equivalent(
                    parts,
                    [hyp_tokens[j - 1]],
                    strict_ref_tokens=strict_ref_tokens,
                ):
                    candidates.append((dp[i - 2][j - 1], "local_ref2", "phonetic"))
                if _expected_repeat_collapse_equivalent(
                    parts,
                    hyp_tokens[j - 1],
                    strict_ref_tokens=strict_ref_tokens,
                ):
                    candidates.append((dp[i - 2][j - 1], "repeat_ref2", "phonetic"))
                if _modal_base_to_past_equivalent(parts, hyp_tokens[j - 1]):
                    candidates.append((dp[i - 2][j - 1], "modal_past_ref2", "phonetic"))
                if _trailing_name_fusion_equivalent(
                    parts,
                    hyp_tokens[j - 1],
                    strict_ref_tokens=strict_ref_tokens,
                ):
                    candidates.append((dp[i - 2][j - 1], "name_ref2", "phonetic"))
                if (
                    not _contains_negation_token(parts)
                    and not any(_is_placeholder_token(tok) for tok in parts)
                    and all(len(_clean_context_word(token)) >= 2 for token in parts)
                    and all(_is_content_word(token) for token in parts)
                ):
                    joined_ref = "".join(parts)
                    if (
                        not _is_placeholder_token(hyp_tokens[j - 1])
                        and spoken_token_equivalent(joined_ref, hyp_tokens[j - 1], phonetic_threshold) != "none"
                    ):
                        candidates.append((dp[i - 2][j - 1], "match_ref2", "phonetic"))

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
                "phonetic": "phonetic_equivalent",
                "ambiguous": "ambiguous_equivalent",
            }.get(eq, "exact_match")
            operations.append({
                "op": op_name,
                "ref": ref_tokens[i - 1],
                "hyp": hyp_tokens[j - 1],
            })
            i -= 1
            j -= 1
        elif kind == "match2":
            operations.append({
                "op": "phonetic_equivalent",
                "ref": ref_tokens[i - 1],
                "hyp": " ".join(hyp_tokens[j - 2:j]),
            })
            i -= 1
            j -= 2
        elif kind == "match3":
            operations.append({
                "op": "phonetic_equivalent",
                "ref": ref_tokens[i - 1],
                "hyp": " ".join(hyp_tokens[j - 3:j]),
            })
            i -= 1
            j -= 3
        elif kind in {"local_hyp2", "book_possessive_hyp2", "copula_hyp2", "letters_hyp2"}:
            operations.append({
                "op": "phonetic_equivalent",
                "ref": ref_tokens[i - 1],
                "hyp": " ".join(hyp_tokens[j - 2:j]),
            })
            i -= 1
            j -= 2
        elif kind in {
            "local_ref2",
            "repeat_ref2",
            "name_ref2",
            "modal_past_ref2",
            "copula_ref2",
            "letters_ref2",
        }:
            operations.append({
                "op": "phonetic_equivalent",
                "ref": " ".join(ref_tokens[i - 2:i]),
                "hyp": hyp_tokens[j - 1],
            })
            i -= 2
            j -= 1
        elif kind == "match_ref2":
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


def detect_truncation(ref_tokens: Sequence[str], hyp_tokens: Sequence[str]) -> dict:
    """Detect severe truncation via word-count ratio (<40% of reference words)."""
    ref_words = len(ref_tokens)
    hyp_words = len(hyp_tokens)
    if ref_words == 0:
        return {"is_truncated": False, "ref_words": 0, "hyp_words": hyp_words, "ratio": 1.0}
    ratio = hyp_words / ref_words
    return {
        "is_truncated": ratio < 0.4,
        "ref_words": ref_words,
        "hyp_words": hyp_words,
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
        elif op_name in {"match2", "match3", "match_ref2"}:
            rendered.extend(str(operation.get("hyp") or "").split())
        elif operation.get("hyp") is not None:
            rendered.extend(str(operation.get("hyp") or "").split())
    return " ".join(token for token in rendered if token)


_ALIGNED_MATCH_OPS = frozenset({
    "exact_match",
    "normalized_equivalent",
    "phonetic_equivalent",
    "ambiguous_equivalent",
    "identifier_wildcard",
    "match2",
    "match3",
    "match_ref2",
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
) -> Dict[str, Any]:
    """Flag first-position mismatches on protected leading words.

    The flag supports two-stage review for short protected starters, especially
    negation words. It never changes PASS/FAIL; it only marks cases that should
    be handed to Stage 2 for confirmation.
    """
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
    cfg = merge_validation_config(config)
    if threshold is not None:
        cfg["pass_threshold"] = threshold

    possessive_filler_equivalence, ref_possessive_filler_text, hyp_possessive_filler_text = (
        _raw_lexical_possessive_filler_equivalence(ref_text, hyp_text)
    )
    apostrophe_s_equivalence, ref_apostrophe_s_text, hyp_apostrophe_s_text = (
        _raw_apostrophe_s_surface_equivalence(
            ref_possessive_filler_text,
            hyp_possessive_filler_text,
        )
    )
    fusion_equivalence, ref_fusion_text, hyp_fusion_text = (
        _raw_exact_surface_word_fusion_equivalence(
            ref_apostrophe_s_text,
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
    contraction_equivalence, ref_contraction_text, hyp_contraction_text = (
        _raw_contraction_equivalence(ref_acronym_render_text, hyp_acronym_render_text)
    )
    conjunction_equivalence, ref_conjunction_text, hyp_conjunction_text = (
        _raw_single_conjunction_omission_equivalence(
            ref_contraction_text,
            hyp_contraction_text,
        )
    )
    book_term_tokens = set((book_term_evidence or {}).get("terms") or {})
    strict_ref_tokens = _strict_short_acronym_tokens(ref_text, book_term_evidence)
    ref_comparison_text = _strip_evidenced_book_term_possessives(
        ref_conjunction_text,
        book_term_tokens,
    )
    split_word_number_equivalence, ref_split_text, hyp_split_text = (
        _raw_split_word_number_equivalence(ref_comparison_text, hyp_conjunction_text)
    )
    raw_phrase_equivalence, hyp_comparison_text = _raw_resegmented_phrase_equivalence(
        ref_split_text,
        hyp_split_text,
        strict_ref_tokens=strict_ref_tokens,
    )
    ref_comparison_text = ref_split_text
    phonetic_thr = float(cfg.get("phonetic_match_threshold", 0.85))
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
    filtered_hyp = _render_aligned_hypothesis(alignment["operations"])
    # Count repeats on aligned tokens so hyphen splits of a book name are not
    # a new phrase; unmatched hyp words stay so extra copies still fail.
    repetition = detect_repetition(
        ref_tokens,
        _aligned_repetition_tokens(alignment["operations"]),
        int(cfg.get("repeat_phrase_max_length", 6)),
    )
    truncation = detect_truncation(ref_tokens, hyp_tokens)
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
    accepted_equivalences = [
        {"ref": str(op["ref"]), "hyp": str(op["hyp"])}
        for op in alignment["operations"]
        if (
            op["op"] == "phonetic_equivalent"
            and " " not in str(op.get("ref") or "")
            and " " not in str(op.get("hyp") or "")
        )
    ]
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
            possessive_filler_equivalence,
            apostrophe_s_equivalence,
            fusion_equivalence,
            letter_sequence_equivalence,
            acronym_render_equivalence,
            contraction_equivalence,
            conjunction_equivalence,
            split_word_number_equivalence,
            raw_phrase_equivalence,
            stutter_equivalence,
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
