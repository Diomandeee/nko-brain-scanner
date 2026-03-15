"""
nko.phonetics — Unified N'Ko Phonetics Module
═══════════════════════════════════════════════

The bedrock phonetics layer for the N'Ko Unified Platform.
Consolidates IPA mappings, tone handling, character classification,
and Unicode utilities from 13+ scattered implementations into one
authoritative module.

Data sources:
    - data/nko-unified.json  (NKO-1.1 canonical dataset, 232 records)
    - core/audio/phoneme.py  (PhonemeMapper — IPA + tone marks)
    - core/transliteration/nko.py  (NkoHandler — full char map)
    - core/prediction/prosody_engine.py  (vowel/consonant sets, tone enums)
    - core/prediction/tts_engine.py  (PhonemeMapping, TonePattern)

Usage::

    from nko.phonetics import NKoPhonetics

    ph = NKoPhonetics()
    ph.is_nko_char('ߞ')          # True
    ph.classify('ߞ')             # 'consonant'
    ph.to_ipa('ߒߞߏ')            # 'nkɔ'
    ph.get_char_info('ߞ')        # CharInfo(char='ߞ', code='U+07DE', ...)

Author: NKO-1.2 build (pulse plan)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# Constants — N'Ko Unicode Block (U+07C0 – U+07FF)
# ═══════════════════════════════════════════════════════════════════════════════

#: Start of the N'Ko Unicode block (first digit: ߀)
NKO_BLOCK_START: int = 0x07C0

#: End of the N'Ko Unicode block (last assigned: U+07FA, block to U+07FF)
NKO_BLOCK_END: int = 0x07FF

#: Tuple for ``range()``-style checks
UNICODE_RANGE: Tuple[int, int] = (NKO_BLOCK_START, NKO_BLOCK_END)

# Sub-ranges within the block
NKO_DIGITS_RANGE: Tuple[int, int] = (0x07C0, 0x07C9)        # ߀–߉
NKO_LETTERS_RANGE: Tuple[int, int] = (0x07CA, 0x07EA)        # ߊ–ߪ
NKO_COMBINING_RANGE: Tuple[int, int] = (0x07EB, 0x07F3)      # ߫–߳
NKO_PUNCTUATION_RANGE: Tuple[int, int] = (0x07F4, 0x07FA)    # ߴ–ߺ


# ═══════════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════════

class ToneType(Enum):
    """Manding tone categories — maps to combining marks in N'Ko."""
    HIGH = auto()       # ߫  U+07EB
    LOW = auto()        # ߬  U+07EC  (labeled "Falling" in some sources)
    RISING = auto()     # ߭  U+07ED
    LONG = auto()       # ߮  U+07EE
    VERY_LONG = auto()  # ߯  U+07EF
    NASAL = auto()      # ߲  U+07F2
    NASAL_ALT = auto()  # ߳  U+07F3
    MID = auto()        # Unmarked (default)
    UNKNOWN = auto()


class CharCategory(Enum):
    """Classification categories for N'Ko codepoints."""
    VOWEL = "vowel"
    CONSONANT = "consonant"
    DIGIT = "digit"
    TONE_MARK = "tone_mark"
    COMBINING = "combining"      # nasalization etc. that aren't tone
    PUNCTUATION = "punctuation"
    OTHER = "other"


# ═══════════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CharInfo:
    """Immutable record for a single N'Ko character."""
    char: str
    codepoint: int
    code: str              # e.g. "U+07DE"
    name: str              # e.g. "Ka"
    category: CharCategory
    ipa: str               # IPA transcription (empty for non-phonemic chars)
    latin: str             # Approximate Latin transliteration
    note: str = ""         # Optional remark
    tone_type: Optional[ToneType] = None      # Set for combining tone marks
    digit_value: Optional[int] = None          # Set for digits
    punctuation_eq: Optional[str] = None       # Set for punctuation


@dataclass(frozen=True)
class Phoneme:
    """A single phoneme produced during IPA conversion."""
    symbol: str            # IPA symbol
    source_char: str       # Original N'Ko character
    audio_hint: str        # Human-readable pronunciation hint
    duration_ms: int       # Suggested duration for TTS
    tone: Optional[ToneType] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Canonical character tables (ground truth, hand-verified)
# Sourced from nko-unified.json + core/transliteration/nko.py + core/audio/phoneme.py
# ═══════════════════════════════════════════════════════════════════════════════

# Vowels: char → (ipa, name, latin)
_VOWELS: Dict[str, Tuple[str, str, str]] = {
    "ߊ": ("a",  "A",       "a"),
    "ߋ": ("o",  "O",       "o"),
    "ߌ": ("i",  "I",       "i"),
    "ߍ": ("e",  "E",       "e"),
    "ߎ": ("u",  "U",       "u"),
    "ߏ": ("ɔ",  "Open O",  "ɔ"),
    "ߐ": ("ə",  "Schwa",   "ə"),
}

# Consonants: char → (ipa, name, latin, note)
_CONSONANTS: Dict[str, Tuple[str, str, str, str]] = {
    "ߑ": ("a",   "Dagbamma",       "a",  "Rare variant"),
    "ߒ": ("n",   "N (Long Leg)",   "n",  "The N in N'Ko"),
    "ߓ": ("b",   "Ba",             "b",  ""),
    "ߔ": ("p",   "Pa",             "p",  "Primarily loan words"),
    "ߕ": ("t",   "Ta",             "t",  ""),
    "ߖ": ("dʒ",  "Ja",             "j",  ""),
    "ߗ": ("tʃ",  "Cha",            "ch", ""),
    "ߘ": ("d",   "Da",             "d",  ""),
    "ߙ": ("r",   "Ra",             "r",  ""),
    "ߚ": ("rr",  "Rra",            "rr", "Trilled"),
    "ߛ": ("s",   "Sa",             "s",  ""),
    "ߜ": ("gb",  "Gba",            "gb", "Labial-velar"),
    "ߝ": ("f",   "Fa",             "f",  ""),
    "ߞ": ("k",   "Ka",             "k",  ""),
    "ߟ": ("l",   "La",             "l",  ""),
    "ߠ": ("n̥",   "Na Woloso",      "n",  "Voiceless nasal"),
    "ߡ": ("m",   "Ma",             "m",  ""),
    "ߢ": ("ɲ",   "Nya",            "ny", ""),
    "ߣ": ("n",   "Na",             "n",  ""),
    "ߤ": ("h",   "Ha",             "h",  ""),
    "ߥ": ("w",   "Wa",             "w",  ""),
    "ߦ": ("j",   "Ya",             "y",  ""),
    "ߧ": ("ŋ",   "Nya Woloso",     "ng", "Velar nasal"),
    "ߨ": ("p",   "Pa (variant)",   "p",  "Variant"),
    "ߩ": ("r",   "Ra (variant)",   "r",  "Variant"),
    "ߪ": ("s",   "Sa (variant)",   "s",  "Variant"),
}

# Digits: char → (value, nko_name)
_DIGITS: Dict[str, Tuple[int, str]] = {
    "߀": (0, "fuyi"),
    "߁": (1, "kelen"),
    "߂": (2, "fila"),
    "߃": (3, "saba"),
    "߄": (4, "naani"),
    "߅": (5, "duuru"),
    "߆": (6, "wɔɔrɔ"),
    "߇": (7, "wolonwula"),
    "߈": (8, "segin"),
    "߉": (9, "kɔnɔntɔ"),
}

# Tone / combining marks: char → (ToneType, meaning, ipa_diacritic)
_TONE_MARKS: Dict[str, Tuple[ToneType, str, str]] = {
    "߫": (ToneType.HIGH,      "High tone",             "\u0301"),   # combining acute
    "߬": (ToneType.LOW,       "Falling tone",          "\u0300"),   # combining grave
    "߭": (ToneType.RISING,    "Rising tone",           "\u0302"),   # combining circumflex
    "߮": (ToneType.LONG,      "Long",                  "ː"),
    "߯": (ToneType.VERY_LONG, "Very long",             "ːː"),
}

_COMBINING_MARKS: Dict[str, Tuple[ToneType, str, str]] = {
    "߲": (ToneType.NASAL,     "Nasalization",          "\u0303"),   # combining tilde
    "߳": (ToneType.NASAL_ALT, "Nasalization (alt)",    "\u0303"),
}

# Punctuation: char → (meaning, latin_equivalent)
_PUNCTUATION: Dict[str, Tuple[str, str]] = {
    "߸": ("Exclamation",                "!"),
    "߹": ("Full stop",                  "."),
    "߷": ("Ellipsis",                   "..."),
    "ߺ": ("Lajanyalan (word joiner)",   "-"),
    "ߴ": ("High tone apostrophe",       "'"),
    "ߵ": ("Low tone apostrophe",        "'"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience sets (frozensets for O(1) membership tests)
# ═══════════════════════════════════════════════════════════════════════════════

VOWEL_CHARS: frozenset = frozenset(_VOWELS.keys())
CONSONANT_CHARS: frozenset = frozenset(_CONSONANTS.keys())
LETTER_CHARS: frozenset = VOWEL_CHARS | CONSONANT_CHARS
DIGIT_CHARS: frozenset = frozenset(_DIGITS.keys())
TONE_MARK_CHARS: frozenset = frozenset(_TONE_MARKS.keys())
COMBINING_CHARS: frozenset = frozenset(_COMBINING_MARKS.keys())
PUNCTUATION_CHARS: frozenset = frozenset(_PUNCTUATION.keys())
ALL_NKO_CHARS: frozenset = (
    LETTER_CHARS | DIGIT_CHARS | TONE_MARK_CHARS | COMBINING_CHARS | PUNCTUATION_CHARS
)

# IPA vowel symbols (for syllabification)
IPA_VOWELS: frozenset = frozenset({"a", "e", "i", "o", "u", "ɔ", "ə"})


# ═══════════════════════════════════════════════════════════════════════════════
# NKoPhonetics — Main public API
# ═══════════════════════════════════════════════════════════════════════════════

class NKoPhonetics:
    """
    Unified phonetics engine for N'Ko script.

    Provides:
        - Character info lookup  (``get_char_info``)
        - Unicode range checks   (``is_nko_char``, ``is_vowel``, …)
        - IPA conversion         (``to_ipa``, ``to_phonemes``)
        - Tone detection         (``get_tone``, ``strip_tones``)
        - Classification         (``classify``, ``classify_text``)
        - Data loading           (``load_data`` from nko-unified.json)

    All static tables are built at class level; ``__init__`` optionally
    loads the full JSON dataset for extended queries.

    Example::

        ph = NKoPhonetics()
        assert ph.to_ipa("ߒߞߏ") == "nkɔ"
        assert ph.classify("ߞ") == CharCategory.CONSONANT
    """

    # ─── class-level lookup tables ───────────────────────────────────────

    _char_info_cache: Dict[str, CharInfo] = {}
    _nko_to_ipa: Dict[str, str] = {}

    def __init__(self, data_path: Optional[str | Path] = None, *, load_json: bool = True):
        """
        Parameters
        ----------
        data_path : path-like, optional
            Override path to ``nko-unified.json``.  Defaults to
            ``<project>/data/nko-unified.json``.
        load_json : bool
            If *True* (default), load the canonical JSON dataset on init
            and merge any extra fields into char info records.  Set *False*
            for a lightweight, tables-only instance.
        """
        # Build internal lookup from the static tables (idempotent)
        if not NKoPhonetics._char_info_cache:
            NKoPhonetics._build_tables()

        self._json_data: Optional[dict] = None
        if load_json:
            self._json_data = self._load_json(data_path)

    # ─── table construction (once per process) ───────────────────────────

    @classmethod
    def _build_tables(cls) -> None:
        """Populate ``_char_info_cache`` and ``_nko_to_ipa`` from static data."""
        cache = cls._char_info_cache
        ipa_map = cls._nko_to_ipa

        # Vowels
        for ch, (ipa, name, latin) in _VOWELS.items():
            cp = ord(ch)
            info = CharInfo(
                char=ch, codepoint=cp, code=f"U+{cp:04X}",
                name=name, category=CharCategory.VOWEL,
                ipa=ipa, latin=latin,
            )
            cache[ch] = info
            ipa_map[ch] = ipa

        # Consonants
        for ch, (ipa, name, latin, note) in _CONSONANTS.items():
            cp = ord(ch)
            info = CharInfo(
                char=ch, codepoint=cp, code=f"U+{cp:04X}",
                name=name, category=CharCategory.CONSONANT,
                ipa=ipa, latin=latin, note=note,
            )
            cache[ch] = info
            ipa_map[ch] = ipa

        # Digits
        for ch, (value, nko_name) in _DIGITS.items():
            cp = ord(ch)
            cache[ch] = CharInfo(
                char=ch, codepoint=cp, code=f"U+{cp:04X}",
                name=nko_name, category=CharCategory.DIGIT,
                ipa="", latin=str(value), digit_value=value,
            )

        # Tone marks
        for ch, (tone, meaning, ipa_d) in _TONE_MARKS.items():
            cp = ord(ch)
            cache[ch] = CharInfo(
                char=ch, codepoint=cp, code=f"U+{cp:04X}",
                name=meaning, category=CharCategory.TONE_MARK,
                ipa=ipa_d, latin="", tone_type=tone,
            )

        # Other combining marks (nasalization)
        for ch, (tone, meaning, ipa_d) in _COMBINING_MARKS.items():
            cp = ord(ch)
            cache[ch] = CharInfo(
                char=ch, codepoint=cp, code=f"U+{cp:04X}",
                name=meaning, category=CharCategory.COMBINING,
                ipa=ipa_d, latin="", tone_type=tone,
            )

        # Punctuation
        for ch, (meaning, equiv) in _PUNCTUATION.items():
            cp = ord(ch)
            cache[ch] = CharInfo(
                char=ch, codepoint=cp, code=f"U+{cp:04X}",
                name=meaning, category=CharCategory.PUNCTUATION,
                ipa="", latin=equiv, punctuation_eq=equiv,
            )

    # ─── JSON data loading ───────────────────────────────────────────────

    def _load_json(self, path: Optional[str | Path] = None) -> Optional[dict]:
        """Load ``nko-unified.json`` and return the parsed dict."""
        if path is None:
            # Walk up from this file to find the data/ dir
            candidates = [
                Path(__file__).resolve().parent / "data" / "nko-unified.json",  # Inside package
                Path(__file__).resolve().parent.parent / "data" / "nko-unified.json",  # Dev layout
                Path.home() / "Desktop" / "NKo" / "data" / "nko-unified.json",  # Fallback
            ]
            for c in candidates:
                if c.exists():
                    path = c
                    break
        if path is None:
            return None
        path = Path(path)
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @property
    def json_data(self) -> Optional[dict]:
        """Access the full parsed JSON dataset (or *None* if not loaded)."""
        return self._json_data

    @property
    def vocabulary(self) -> List[dict]:
        """Vocabulary entries from the JSON dataset."""
        if self._json_data is None:
            return []
        return self._json_data.get("vocabulary", [])

    @property
    def proverbs(self) -> List[dict]:
        """Proverb entries from the JSON dataset."""
        if self._json_data is None:
            return []
        return self._json_data.get("proverbs", [])

    # ─── Unicode range utilities ─────────────────────────────────────────

    @staticmethod
    def is_nko_char(ch: str) -> bool:
        """Return *True* if *ch* is in the N'Ko Unicode block (U+07C0–U+07FF)."""
        if len(ch) != 1:
            return False
        cp = ord(ch)
        return NKO_BLOCK_START <= cp <= NKO_BLOCK_END

    @staticmethod
    def is_nko_text(text: str) -> bool:
        """Return *True* if every non-whitespace character is N'Ko."""
        return all(
            NKO_BLOCK_START <= ord(ch) <= NKO_BLOCK_END
            for ch in text
            if not ch.isspace()
        )

    @staticmethod
    def nko_purity(text: str) -> float:
        """Fraction of non-whitespace characters that are N'Ko (0.0–1.0)."""
        total = sum(1 for ch in text if not ch.isspace())
        if total == 0:
            return 0.0
        nko = sum(
            1 for ch in text
            if not ch.isspace() and NKO_BLOCK_START <= ord(ch) <= NKO_BLOCK_END
        )
        return nko / total

    # ─── Character info lookup ───────────────────────────────────────────

    @classmethod
    def get_char_info(cls, ch: str) -> Optional[CharInfo]:
        """
        Return a :class:`CharInfo` for *ch*, or *None* if not a known N'Ko char.
        """
        if not cls._char_info_cache:
            cls._build_tables()
        return cls._char_info_cache.get(ch)

    @classmethod
    def get_all_chars(cls) -> Dict[str, CharInfo]:
        """Return a copy of the full character info table."""
        if not cls._char_info_cache:
            cls._build_tables()
        return dict(cls._char_info_cache)

    # ─── Classification ──────────────────────────────────────────────────

    @staticmethod
    def classify(ch: str) -> CharCategory:
        """
        Classify a single character.

        Returns :attr:`CharCategory.OTHER` for non-N'Ko or unknown chars.
        """
        if ch in VOWEL_CHARS:
            return CharCategory.VOWEL
        if ch in CONSONANT_CHARS:
            return CharCategory.CONSONANT
        if ch in DIGIT_CHARS:
            return CharCategory.DIGIT
        if ch in TONE_MARK_CHARS:
            return CharCategory.TONE_MARK
        if ch in COMBINING_CHARS:
            return CharCategory.COMBINING
        if ch in PUNCTUATION_CHARS:
            return CharCategory.PUNCTUATION
        return CharCategory.OTHER

    @staticmethod
    def is_vowel(ch: str) -> bool:
        """Check if character is an N'Ko vowel."""
        return ch in VOWEL_CHARS

    @staticmethod
    def is_consonant(ch: str) -> bool:
        """Check if character is an N'Ko consonant."""
        return ch in CONSONANT_CHARS

    @staticmethod
    def is_letter(ch: str) -> bool:
        """Check if character is an N'Ko letter (vowel or consonant)."""
        return ch in LETTER_CHARS

    @staticmethod
    def is_tone_mark(ch: str) -> bool:
        """Check if character is a tone/length mark (U+07EB–U+07EF)."""
        return ch in TONE_MARK_CHARS

    @staticmethod
    def is_combining(ch: str) -> bool:
        """Check if character is any combining mark (tone + nasalization)."""
        return ch in TONE_MARK_CHARS or ch in COMBINING_CHARS

    @staticmethod
    def is_digit(ch: str) -> bool:
        """Check if character is an N'Ko digit."""
        return ch in DIGIT_CHARS

    @staticmethod
    def is_punctuation(ch: str) -> bool:
        """Check if character is N'Ko punctuation."""
        return ch in PUNCTUATION_CHARS

    # ─── Tone handling ───────────────────────────────────────────────────

    @staticmethod
    def get_tone(ch: str) -> Optional[ToneType]:
        """
        Return the :class:`ToneType` if *ch* is a tone/combining mark,
        else *None*.
        """
        if ch in _TONE_MARKS:
            return _TONE_MARKS[ch][0]
        if ch in _COMBINING_MARKS:
            return _COMBINING_MARKS[ch][0]
        return None

    @staticmethod
    def strip_tones(text: str) -> str:
        """Remove all tone and combining marks from N'Ko text."""
        return "".join(
            ch for ch in text
            if ch not in TONE_MARK_CHARS and ch not in COMBINING_CHARS
        )

    @staticmethod
    def extract_tones(text: str) -> List[Tuple[int, ToneType]]:
        """
        Return a list of ``(position, ToneType)`` for every tone mark in *text*.

        Position is the index in the original string.
        """
        result: List[Tuple[int, ToneType]] = []
        for i, ch in enumerate(text):
            tone = None
            if ch in _TONE_MARKS:
                tone = _TONE_MARKS[ch][0]
            elif ch in _COMBINING_MARKS:
                tone = _COMBINING_MARKS[ch][0]
            if tone is not None:
                result.append((i, tone))
        return result

    @staticmethod
    def has_tone_marks(text: str) -> bool:
        """Return *True* if *text* contains any tone or combining marks."""
        for ch in text:
            if ch in TONE_MARK_CHARS or ch in COMBINING_CHARS:
                return True
        return False

    # ─── IPA conversion ──────────────────────────────────────────────────

    @classmethod
    def char_to_ipa(cls, ch: str) -> str:
        """
        Convert a single N'Ko letter to its IPA symbol.

        Returns the original character unchanged if no mapping exists
        (whitespace, punctuation, non-N'Ko, etc.).
        """
        if not cls._nko_to_ipa:
            cls._build_tables()
        return cls._nko_to_ipa.get(ch, ch)

    @classmethod
    def to_ipa(cls, text: str, *, include_tones: bool = False) -> str:
        """
        Convert N'Ko text to an IPA transcription string.

        Parameters
        ----------
        text : str
            N'Ko text (may include tone marks, spaces, punctuation).
        include_tones : bool
            If *True*, append IPA diacritics for tone marks.  If *False*
            (default), tone marks are silently skipped for a clean base
            transcription.

        Returns
        -------
        str
            IPA string with spaces preserved.
        """
        if not cls._nko_to_ipa:
            cls._build_tables()

        parts: List[str] = []
        for ch in text:
            # Whitespace passthrough
            if ch.isspace():
                parts.append(ch)
                continue

            # Tone / combining marks
            if ch in _TONE_MARKS:
                if include_tones:
                    parts.append(_TONE_MARKS[ch][2])
                continue
            if ch in _COMBINING_MARKS:
                if include_tones:
                    parts.append(_COMBINING_MARKS[ch][2])
                continue

            # Punctuation — skip or pass through equivalent
            if ch in _PUNCTUATION:
                continue

            # Letter → IPA
            ipa = cls._nko_to_ipa.get(ch)
            if ipa is not None:
                parts.append(ipa)
            # Digits and unknowns — passthrough
            else:
                parts.append(ch)

        return "".join(parts)

    @classmethod
    def to_phonemes(cls, text: str) -> List[Phoneme]:
        """
        Convert N'Ko text to a list of :class:`Phoneme` objects.

        Tone marks are attached to the **preceding** phoneme.
        """
        if not cls._nko_to_ipa:
            cls._build_tables()

        phonemes: List[Phoneme] = []
        pending_tone: Optional[ToneType] = None

        for ch in text:
            # Tone / combining marks
            tone = None
            if ch in _TONE_MARKS:
                tone = _TONE_MARKS[ch][0]
            elif ch in _COMBINING_MARKS:
                tone = _COMBINING_MARKS[ch][0]

            if tone is not None:
                # Attach to the most recent phoneme
                if phonemes:
                    prev = phonemes[-1]
                    phonemes[-1] = Phoneme(
                        symbol=prev.symbol,
                        source_char=prev.source_char,
                        audio_hint=prev.audio_hint,
                        duration_ms=prev.duration_ms,
                        tone=tone,
                    )
                else:
                    pending_tone = tone
                continue

            # Skip whitespace, punctuation, digits for phoneme stream
            if ch.isspace() or ch in _PUNCTUATION or ch in _DIGITS:
                continue

            ipa = cls._nko_to_ipa.get(ch)
            if ipa is None:
                continue

            # Determine hint & duration
            if ch in _VOWELS:
                _, name, _ = _VOWELS[ch]
                hint = f'vowel "{name}"'
                dur = 120
            elif ch in _CONSONANTS:
                _, name, _, _ = _CONSONANTS[ch]
                hint = f'consonant "{name}"'
                dur = 80
            else:
                hint = ""
                dur = 100

            phonemes.append(Phoneme(
                symbol=ipa,
                source_char=ch,
                audio_hint=hint,
                duration_ms=dur,
                tone=pending_tone,
            ))
            pending_tone = None

        return phonemes

    # ─── IPA → N'Ko (reverse mapping) ───────────────────────────────────

    @classmethod
    def ipa_to_nko(cls, ipa: str) -> str:
        """
        Convert an IPA string back to N'Ko script (best-effort).

        Uses longest-match on multi-char IPA symbols (``dʒ``, ``tʃ``, ``gb``, etc.).
        """
        if not cls._nko_to_ipa:
            cls._build_tables()

        # Build reverse map with primary forms preferred
        reverse: Dict[str, str] = {}
        # Insert all, then overwrite with primary (non-variant) chars
        for ch, ipa_sym in cls._nko_to_ipa.items():
            reverse[ipa_sym] = ch
        # Ensure primary consonants win over variants
        primary_overrides = {
            "n": "ߣ", "r": "ߙ", "s": "ߛ", "p": "ߔ", "a": "ߊ",
        }
        reverse.update(primary_overrides)

        # Sort IPA keys longest first for greedy matching
        keys_sorted = sorted(reverse.keys(), key=len, reverse=True)

        result: List[str] = []
        i = 0
        while i < len(ipa):
            if ipa[i].isspace():
                result.append(ipa[i])
                i += 1
                continue
            matched = False
            for key in keys_sorted:
                if ipa[i:i + len(key)] == key:
                    result.append(reverse[key])
                    i += len(key)
                    matched = True
                    break
            if not matched:
                result.append(ipa[i])
                i += 1

        return "".join(result)

    # ─── Syllabification (simplified) ────────────────────────────────────

    @classmethod
    def syllabify_ipa(cls, ipa: str) -> List[str]:
        """
        Split an IPA string into syllables (simplified CV model).

        Breaks after each vowel nucleus.  Good enough for Manding which
        is predominantly CV/CVN structure.
        """
        syllables: List[str] = []
        current: List[str] = []

        for ch in ipa:
            if ch.isspace():
                if current:
                    syllables.append("".join(current))
                    current = []
                continue
            current.append(ch)
            if ch in IPA_VOWELS:
                syllables.append("".join(current))
                current = []

        # Leftover consonants attach to last syllable
        if current:
            if syllables:
                syllables[-1] += "".join(current)
            else:
                syllables.append("".join(current))

        return syllables

    # ─── Script detection ────────────────────────────────────────────────

    @staticmethod
    def detect_script(text: str) -> str:
        """
        Auto-detect dominant script in *text*.

        Returns one of ``"nko"``, ``"arabic"``, ``"latin"``, ``"mixed"``.
        """
        counts = {"nko": 0, "arabic": 0, "latin": 0}
        for ch in text:
            cp = ord(ch)
            if NKO_BLOCK_START <= cp <= NKO_BLOCK_END:
                counts["nko"] += 1
            elif 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
                counts["arabic"] += 1
            elif 0x0041 <= cp <= 0x005A or 0x0061 <= cp <= 0x007A:
                counts["latin"] += 1

        total = sum(counts.values())
        if total == 0:
            return "latin"

        dominant = max(counts, key=counts.get)  # type: ignore[arg-type]
        ratio = counts[dominant] / total
        if ratio < 0.6 and total > 2:
            return "mixed"
        return dominant

    # ─── Digit utilities ─────────────────────────────────────────────────

    @staticmethod
    def nko_digit_value(ch: str) -> Optional[int]:
        """Return the integer value of an N'Ko digit, or *None*."""
        entry = _DIGITS.get(ch)
        return entry[0] if entry else None

    @staticmethod
    def int_to_nko_digits(n: int) -> str:
        """Convert a non-negative integer to N'Ko digit string."""
        if n < 0:
            raise ValueError("Negative numbers not supported")
        if n == 0:
            return "߀"
        # Build reverse map
        val_to_char = {v: ch for ch, (v, _) in _DIGITS.items()}
        result: List[str] = []
        while n > 0:
            result.append(val_to_char[n % 10])
            n //= 10
        # N'Ko is RTL, but digit string representation is same order
        return "".join(reversed(result))

    # ─── Convenience / display ───────────────────────────────────────────

    @classmethod
    def pronunciation_guide(cls, text: str) -> List[Dict[str, str]]:
        """
        Return a character-by-character pronunciation guide.

        Each entry: ``{"char": …, "ipa": …, "category": …, "hint": …}``
        """
        if not cls._char_info_cache:
            cls._build_tables()

        guide: List[Dict[str, str]] = []
        for ch in text:
            info = cls._char_info_cache.get(ch)
            if info is not None:
                guide.append({
                    "char": ch,
                    "ipa": info.ipa,
                    "category": info.category.value,
                    "name": info.name,
                    "hint": info.note or info.name,
                })
            elif ch.isspace():
                guide.append({"char": ch, "ipa": " ", "category": "space", "name": "space", "hint": ""})
        return guide

    def __repr__(self) -> str:
        loaded = "json loaded" if self._json_data else "tables only"
        return f"<NKoPhonetics ({loaded}, {len(self._char_info_cache)} chars)>"


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level convenience — importable as ``from nko.phonetics import IPA``
# ═══════════════════════════════════════════════════════════════════════════════

#: Singleton instance for quick access
IPA = NKoPhonetics(load_json=False)


# ═══════════════════════════════════════════════════════════════════════════════
# __all__ — public API surface
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Main class
    "NKoPhonetics",
    # Data classes
    "CharInfo",
    "Phoneme",
    # Enums
    "ToneType",
    "CharCategory",
    # Constants
    "NKO_BLOCK_START",
    "NKO_BLOCK_END",
    "UNICODE_RANGE",
    "NKO_DIGITS_RANGE",
    "NKO_LETTERS_RANGE",
    "NKO_COMBINING_RANGE",
    "NKO_PUNCTUATION_RANGE",
    # Character sets
    "VOWEL_CHARS",
    "CONSONANT_CHARS",
    "LETTER_CHARS",
    "DIGIT_CHARS",
    "TONE_MARK_CHARS",
    "COMBINING_CHARS",
    "PUNCTUATION_CHARS",
    "ALL_NKO_CHARS",
    "IPA_VOWELS",
    # Singleton
    "IPA",
]
