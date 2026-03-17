"""
nko.transliterate — Canonical Transliteration Engine
ߒߞߏ ߛߓߍ ߦߟߊ߬ߡߊ — N'Ko Script Bridge

Unified transliteration between N'Ko, Latin, and Arabic scripts.
Consolidates implementations from:
  - core/transliteration/ (Bridge, NkoHandler, ArabicHandler, LatinHandler)
  - core/prediction/translation_bridge.py (keyboard-ai)
  - tools/telegram-bot/bridge_core.py
  - core/audio/phoneme.py (PhonemeMapper)

Architecture: IPA as phonetic intermediary for all conversions.
  N'Ko ←→ IPA ←→ Latin
  N'Ko ←→ IPA ←→ Arabic
  Arabic ←→ IPA ←→ Latin

Usage:
    from nko.transliterate import transliterate, detect_script, NkoTransliterator

    # Quick function — auto-detect source, specify target
    transliterate("ߒߞߏ", target="latin")        # → "nko"
    transliterate("salam", target="nko")           # → "ߛߊߟߊߡ"
    transliterate("سلام", target="latin")          # → "slam"

    # Script detection
    detect_script("ߒߞߏ")                          # → "nko"
    detect_script("hello")                          # → "latin"

    # Full API via class
    t = NkoTransliterator()
    t.convert("ߒߞߏ", target="latin")
    t.convert_all("ߒߞߏ")  # → {"nko": "ߒߞߏ", "latin": "nko", "arabic": "نكو"}
    t.batch(["ߒߞߏ", "ߛߟߊ߯ߡ"], target="latin")
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Union


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Script Type Enum
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Script(str, Enum):
    """Supported scripts for transliteration."""
    NKO = "nko"
    LATIN = "latin"
    ARABIC = "arabic"

    def __str__(self) -> str:
        return self.value


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Result Dataclass
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass(frozen=True)
class TranslitResult:
    """Result of a transliteration operation."""
    source_text: str
    source_script: Script
    target_text: str
    target_script: Script
    ipa: str = ""
    confidence: float = 1.0
    warnings: Tuple[str, ...] = ()

    def __str__(self) -> str:
        return self.target_text

    def __repr__(self) -> str:
        return (
            f"TranslitResult({self.source_script}→{self.target_script}: "
            f"{self.source_text!r} → {self.target_text!r})"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Character Maps — The Single Source of Truth
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── N'Ko → IPA (canonical) ────────────────────────────────────

NKO_VOWELS_TO_IPA: Dict[str, str] = {
    "ߊ": "a",       # A (U+07CA)
    "ߋ": "o",       # O (U+07CB)
    "ߌ": "i",       # I (U+07CC)
    "ߍ": "e",       # E (U+07CD)
    "ߎ": "u",       # U (U+07CE)
    "ߏ": "ɔ",       # Open O (U+07CF)
    "ߐ": "ɛ",       # Open E / Schwa (U+07D0) — corrected per Manding phonology
}

NKO_CONSONANTS_TO_IPA: Dict[str, str] = {
    "ߒ": "n",       # Syllabic N — THE N in N'Ko (U+07D2)
    "ߓ": "b",       # Ba
    "ߔ": "p",       # Pa (loan words)
    "ߕ": "t",       # Ta
    "ߖ": "dʒ",      # Ja
    "ߗ": "tʃ",      # Cha
    "ߘ": "d",       # Da
    "ߙ": "r",       # Ra
    "ߚ": "rr",      # Rra (trilled)
    "ߛ": "s",       # Sa
    "ߜ": "gb",      # Gba (labial-velar)
    "ߝ": "f",       # Fa
    "ߞ": "k",       # Ka
    "ߟ": "l",       # La
    "ߠ": "na",      # Na Woloso (syllabic)
    "ߡ": "m",       # Ma
    "ߢ": "ɲ",       # Nya (palatal nasal)
    "ߣ": "n",       # Na
    "ߤ": "h",       # Ha
    "ߥ": "w",       # Wa
    "ߦ": "j",       # Ya (IPA j = English y)
    "ߧ": "ŋ",       # Nga (velar nasal)
    "ߨ": "p",       # Pa (alternate)
    "ߩ": "r",       # Ra (alternate)
    "ߪ": "s",       # Sa (alternate)
}

NKO_DIGITS_TO_WESTERN: Dict[str, str] = {
    "߀": "0", "߁": "1", "߂": "2", "߃": "3", "߄": "4",
    "߅": "5", "߆": "6", "߇": "7", "߈": "8", "߉": "9",
}

NKO_TONE_MARKS: Dict[str, str] = {
    "߫": "́",       # High tone (combining acute) U+07EB
    "߬": "̀",       # Low tone (combining grave) U+07EC
    "߭": "̂",       # Falling tone (combining circumflex) U+07ED
    "߮": "̌",       # Rising tone (combining caron) U+07EE
    "߯": "ː",      # Long vowel (IPA length) U+07EF
}

NKO_PUNCTUATION: Dict[str, str] = {
    "߸": ",",       # Comma
    "߹": ".",       # Full stop
    "߷": "!",       # Exclamation
    "ߺ": "-",       # Lajanyalan (word joiner)
}

NKO_COMBINING: Dict[str, str] = {
    "߲": "ⁿ",      # Nasalization mark U+07F2
    "߳": "̃",       # Nasalization (combining tilde) U+07F3
}

# Full N'Ko → IPA (merged)
NKO_TO_IPA: Dict[str, str] = {
    **NKO_VOWELS_TO_IPA,
    **NKO_CONSONANTS_TO_IPA,
    **NKO_DIGITS_TO_WESTERN,
    **NKO_TONE_MARKS,
    **NKO_PUNCTUATION,
    **NKO_COMBINING,
    # Dagbamma (rare)
    "ߑ": "a",       # U+07D1
}

# ── IPA → N'Ko (reverse, with precedence) ────────────────────

_IPA_TO_NKO_RAW: Dict[str, str] = {}
for _k, _v in NKO_TO_IPA.items():
    if _v and _v not in _IPA_TO_NKO_RAW:
        _IPA_TO_NKO_RAW[_v] = _k

# Ensure primary characters win over alternates
IPA_TO_NKO: Dict[str, str] = {
    **_IPA_TO_NKO_RAW,
    "a": "ߊ",
    "n": "ߣ",
    "r": "ߙ",
    "s": "ߛ",
    "p": "ߔ",
    "g": "ߜ",  # Bambara plain g → N'Ko gba (ߜ). Whisper ASR outputs 'g' not 'gb'.
    # ── Bambara gap fixes ─────────────────────────────────────────────────────
    # z: N'Ko has no distinct z letter. Standard Bambara N'Ko orthography uses ߛ
    #    (sa) for z-sounds in loanwords (e.g. zɛrɛ 'money' → ߛߍߙߍ).
    "z": "ߛ",
    # ə (schwa): In Bambara N'Ko, schwa is written as ߐ (same glyph as ɛ).
    "ə": "ߐ",
    # ʃ: The sh sound (e.g. Bambara 'sha'). No dedicated N'Ko letter; written ߛ.
    "ʃ": "ߛ",
}

# Remove the 'na' → ߠ (Na Woloso) entry from IPA_TO_NKO.
# Na Woloso (ߠ) is a rare syllabic character, NOT used in standard Bambara text.
# The greedy 2-char 'na' match causes 'nana' → 'ߠߠ' instead of correct 'ߣߊߣߊ'.
# Sequences of n+a are handled correctly by the individual n→ߣ and a→ߊ rules.
IPA_TO_NKO.pop("na", None)

# ── IPA → Latin (readable) ───────────────────────────────────

IPA_TO_LATIN: Dict[str, str] = {
    # Vowels
    "a": "a", "e": "e", "i": "i", "o": "o", "u": "u",
    "ɔ": "ɔ", "ɛ": "ɛ", "ə": "e",
    "aː": "aa", "eː": "ee", "iː": "ii", "oː": "oo", "uː": "uu",
    # Consonants
    "b": "b", "d": "d", "f": "f", "g": "g", "h": "h",
    "k": "k", "l": "l", "m": "m", "n": "n", "p": "p",
    "r": "r", "s": "s", "t": "t", "v": "v", "w": "w",
    "z": "z", "j": "y",  # IPA j → Latin y
    # Digraphic consonants
    "dʒ": "j", "tʃ": "c", "gb": "gb", "ŋ": "ng",
    "ɲ": "ny", "rr": "rr", "na": "na",
    # IPA specials
    "ʔ": "'", "ɣ": "gh", "ʃ": "sh", "ʒ": "zh",
    "θ": "th", "ð": "dh", "x": "kh",
    "ħ": "h", "ʕ": "'", "q": "q",
    # Emphatics → simplified
    "sˤ": "s", "dˤ": "d", "tˤ": "t", "ðˤ": "dh",
    # Tone/length marks → omit in Latin by default
    "́": "", "̀": "", "̂": "", "̌": "", "ː": "",
    "̄": "", "̃": "n", "ⁿ": "n",
}

# ── Latin → IPA ──────────────────────────────────────────────

# Digraphs first (checked before single chars)
LATIN_DIGRAPHS_TO_IPA: Dict[str, str] = {
    "ch": "tʃ", "sh": "ʃ", "th": "θ", "dh": "ð",
    "gh": "ɣ", "kh": "x", "ng": "ŋ", "ny": "ɲ",
    "gb": "gb", "rr": "rr", "ph": "f", "wh": "w",
    "dj": "dʒ", "nk": "ŋk",
}

LATIN_SINGLE_TO_IPA: Dict[str, str] = {
    "a": "a", "b": "b", "c": "tʃ", "d": "d", "e": "e",
    "f": "f", "g": "g", "h": "h", "i": "i", "j": "dʒ",
    "k": "k", "l": "l", "m": "m", "n": "n", "o": "o",
    "p": "p", "q": "k", "r": "r", "s": "s", "t": "t",
    "u": "u", "v": "v", "w": "w", "x": "ks", "y": "j",
    "z": "z",
    # Bambara special vowels and consonants (used directly in Latin Bambara corpora)
    "ɔ": "ɔ", "ɛ": "ɛ", "ə": "ə",
    "ɲ": "ɲ",   # Palatal nasal — written directly in Bambara Latin (e.g. ɲɛ, ɲinini)
    "ŋ": "ŋ",   # Velar nasal — written directly in some Bambara corpora
    # Combining diacritics (tone marks) — present after NFD normalization of à á è é etc.
    # These pass through to IPA as-is; IPA_TO_NKO maps them to N'Ko tone marks (߬ ߫).
    "\u0300": "\u0300",  # combining grave accent  → ߬ low tone
    "\u0301": "\u0301",  # combining acute accent  → ߫ high tone
    "\u0302": "\u0302",  # combining circumflex    → ߭ falling tone
    "\u030C": "\u030C",  # combining caron         → ߮ rising tone
    "\u0303": "\u0303",  # combining tilde (nasal) → ߳ nasalization
    "'": "ʔ",
}

# ── Arabic → IPA ─────────────────────────────────────────────

ARABIC_TO_IPA: Dict[str, str] = {
    # Consonants
    "ء": "ʔ", "ا": "aː", "آ": "ʔaː",
    "ب": "b", "ت": "t", "ث": "θ", "ج": "dʒ",
    "ح": "ħ", "خ": "x", "د": "d", "ذ": "ð",
    "ر": "r", "ز": "z", "س": "s", "ش": "ʃ",
    "ص": "sˤ", "ض": "dˤ", "ط": "tˤ", "ظ": "ðˤ",
    "ع": "ʕ", "غ": "ɣ", "ف": "f", "ق": "q",
    "ك": "k", "ل": "l", "م": "m", "ن": "n",
    "ه": "h", "و": "w", "ي": "j", "ى": "aː",
    # Short vowels (diacritics)
    "َ": "a", "ِ": "i", "ُ": "u", "ْ": "",
    # Tanwin
    "ً": "an", "ٍ": "in", "ٌ": "un",
    # Special
    "ة": "a", "ّ": "ː",
    # Digits
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    # Punctuation
    "،": ",", "؛": ";", "؟": "?",
}

# ── IPA → Arabic ─────────────────────────────────────────────

IPA_TO_ARABIC: Dict[str, str] = {
    "b": "ب", "t": "ت", "d": "د", "r": "ر", "z": "ز",
    "s": "س", "f": "ف", "k": "ك", "l": "ل", "m": "م",
    "n": "ن", "h": "ه", "w": "و", "j": "ي", "q": "ق",
    "x": "خ", "g": "غ",
    "a": "ا", "i": "ي", "u": "و", "e": "ي", "o": "و",
    "ʔ": "ء", "ħ": "ح", "ʕ": "ع", "ɣ": "غ",
    "ʃ": "ش", "θ": "ث", "ð": "ذ",
    "dʒ": "ج", "tʃ": "ش",
    "ŋ": "نغ", "ɲ": "ني",
    "ɔ": "و", "ɛ": "ي", "ə": "ا",
    "gb": "غب",
    "rr": "ر",
    "na": "نا",
    # Emphatics
    "sˤ": "ص", "dˤ": "ض", "tˤ": "ط", "ðˤ": "ظ",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Script Detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_script(text: str) -> Script:
    """
    Auto-detect the dominant script in *text* using Unicode range voting.

    Returns Script.NKO, Script.ARABIC, or Script.LATIN (default fallback).

    Examples:
        >>> detect_script("ߒߞߏ")
        <Script.NKO: 'nko'>
        >>> detect_script("hello")
        <Script.LATIN: 'latin'>
        >>> detect_script("سلام")
        <Script.ARABIC: 'arabic'>
    """
    counts = {Script.NKO: 0, Script.ARABIC: 0, Script.LATIN: 0}
    for ch in text:
        cp = ord(ch)
        if 0x07C0 <= cp <= 0x07FF:
            counts[Script.NKO] += 1
        elif 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0xFB50 <= cp <= 0xFDFF:
            counts[Script.ARABIC] += 1
        elif (0x0041 <= cp <= 0x005A or 0x0061 <= cp <= 0x007A
              or 0x00C0 <= cp <= 0x024F  # Extended Latin
              or cp in (0x025B, 0x0254, 0x0259)):  # ɛ, ɔ, ə
            counts[Script.LATIN] += 1
    total = sum(counts.values())
    if total == 0:
        return Script.LATIN
    return max(counts, key=counts.get)  # type: ignore[arg-type]


def is_nko(ch: str) -> bool:
    """Return True if *ch* is in the N'Ko Unicode block (U+07C0–U+07FF)."""
    cp = ord(ch[0]) if ch else 0
    return 0x07C0 <= cp <= 0x07FF


def is_arabic(ch: str) -> bool:
    """Return True if *ch* is in an Arabic Unicode block."""
    cp = ord(ch[0]) if ch else 0
    return (0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F
            or 0xFB50 <= cp <= 0xFDFF)


def is_latin(ch: str) -> bool:
    """Return True if *ch* is a Latin letter (basic or extended)."""
    cp = ord(ch[0]) if ch else 0
    return (0x0041 <= cp <= 0x005A or 0x0061 <= cp <= 0x007A
            or 0x00C0 <= cp <= 0x024F
            or cp in (0x025B, 0x0254, 0x0259))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Core Transliterator Class
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class NkoTransliterator:
    """
    Canonical N'Ko transliteration engine.

    All conversions route through IPA as a phonetic intermediary:
        Source Script → IPA → Target Script

    Supports N'Ko, Latin (Manding/general), and Arabic scripts.
    """

    def __init__(self) -> None:
        # Precompile Latin digraph patterns (sorted longest-first)
        self._latin_digraphs = sorted(
            LATIN_DIGRAPHS_TO_IPA.keys(), key=len, reverse=True
        )
        # IPA multi-char tokens sorted longest-first for matching
        self._ipa_to_latin_keys = sorted(
            IPA_TO_LATIN.keys(), key=len, reverse=True
        )
        self._ipa_to_arabic_keys = sorted(
            IPA_TO_ARABIC.keys(), key=len, reverse=True
        )
        self._ipa_to_nko_keys = sorted(
            IPA_TO_NKO.keys(), key=len, reverse=True
        )

    # ── Public API ────────────────────────────────────────────

    def convert(
        self,
        text: str,
        source: Optional[Union[str, Script]] = None,
        target: Union[str, Script] = Script.LATIN,
    ) -> TranslitResult:
        """
        Convert *text* from *source* script to *target* script.

        If *source* is None, it is auto-detected.

        Args:
            text: Input text in any supported script.
            source: Source script ("nko", "latin", "arabic") or None for auto.
            target: Target script.

        Returns:
            TranslitResult with converted text, IPA, confidence, etc.
        """
        src = _coerce_script(source) if source is not None else detect_script(text)
        tgt = _coerce_script(target)

        if src == tgt:
            return TranslitResult(
                source_text=text, source_script=src,
                target_text=text, target_script=tgt,
            )

        # Step 1: source → IPA
        ipa = self._to_ipa(text, src)
        # Step 2: IPA → target
        result = self._from_ipa(ipa, tgt)

        # Confidence heuristic: ratio of successfully-mapped chars
        total = len([c for c in text if not c.isspace()])
        mapped = len([c for c in result if not c.isspace()])
        conf = min(mapped / max(total, 1), 1.0)

        return TranslitResult(
            source_text=text, source_script=src,
            target_text=result, target_script=tgt,
            ipa=ipa, confidence=round(conf, 3),
        )

    def convert_all(self, text: str, source: Optional[Union[str, Script]] = None) -> Dict[str, str]:
        """
        Convert *text* to all three scripts.

        Returns:
            Dict with keys "nko", "latin", "arabic" → converted text.
        """
        src = _coerce_script(source) if source is not None else detect_script(text)
        out: Dict[str, str] = {}
        for tgt in Script:
            if tgt == src:
                out[tgt.value] = text
            else:
                out[tgt.value] = self.convert(text, src, tgt).target_text
        return out

    def batch(
        self,
        texts: Sequence[str],
        source: Optional[Union[str, Script]] = None,
        target: Union[str, Script] = Script.LATIN,
    ) -> List[TranslitResult]:
        """
        Batch-convert a sequence of texts.

        Args:
            texts: Iterable of input strings.
            source: Shared source script or None for per-text auto-detect.
            target: Target script.

        Returns:
            List of TranslitResult, one per input text.
        """
        return [self.convert(t, source, target) for t in texts]

    def to_ipa(self, text: str, source: Optional[Union[str, Script]] = None) -> str:
        """
        Convert *text* to its IPA representation.
        """
        src = _coerce_script(source) if source is not None else detect_script(text)
        return self._to_ipa(text, src)

    def analyze(self, text: str) -> Dict:
        """
        Analyze *text* and return script composition breakdown.
        """
        counts = {"nko": 0, "arabic": 0, "latin": 0, "other": 0}
        for ch in text:
            cp = ord(ch)
            if 0x07C0 <= cp <= 0x07FF:
                counts["nko"] += 1
            elif 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
                counts["arabic"] += 1
            elif 0x0041 <= cp <= 0x005A or 0x0061 <= cp <= 0x007A:
                counts["latin"] += 1
            elif not ch.isspace():
                counts["other"] += 1
        total = sum(counts.values()) or 1
        dominant = max(counts, key=counts.get)  # type: ignore
        return {
            "counts": counts,
            "percentages": {k: round(v / total * 100, 1) for k, v in counts.items()},
            "dominant": dominant,
            "total_chars": sum(counts.values()),
        }

    # ── Internal: Source → IPA ────────────────────────────────

    def _to_ipa(self, text: str, script: Script) -> str:
        if script == Script.NKO:
            return self._nko_to_ipa(text)
        elif script == Script.ARABIC:
            return self._arabic_to_ipa(text)
        else:
            return self._latin_to_ipa(text)

    def _nko_to_ipa(self, text: str) -> str:
        parts: list[str] = []
        for ch in text:
            if ch in NKO_TO_IPA:
                parts.append(NKO_TO_IPA[ch])
            elif ch.isspace():
                parts.append(ch)
            else:
                parts.append(ch)  # Pass through unknown
        return "".join(parts)

    def _arabic_to_ipa(self, text: str) -> str:
        parts: list[str] = []
        for ch in text:
            if ch in ARABIC_TO_IPA:
                parts.append(ARABIC_TO_IPA[ch])
            elif ch.isspace():
                parts.append(ch)
            else:
                parts.append(ch)
        return "".join(parts)

    def _latin_to_ipa(self, text: str) -> str:
        parts: list[str] = []
        # NFD-normalize so that precomposed toned vowels (à á è é ì í ò ó ù ú)
        # decompose into base-letter + combining diacritic.  The combining diacritics
        # are mapped in LATIN_SINGLE_TO_IPA (pass-through) and then IPA_TO_NKO
        # converts them to N'Ko tone marks (߬ ߫ etc.).
        lower = unicodedata.normalize("NFD", text.lower())
        i = 0
        while i < len(lower):
            matched = False
            # Try digraphs (longest first)
            for dg in self._latin_digraphs:
                dg_len = len(dg)
                if lower[i:i + dg_len] == dg:
                    parts.append(LATIN_DIGRAPHS_TO_IPA[dg])
                    i += dg_len
                    matched = True
                    break
            if matched:
                continue
            ch = lower[i]
            if ch in LATIN_SINGLE_TO_IPA:
                parts.append(LATIN_SINGLE_TO_IPA[ch])
            elif ch.isspace() or ch.isdigit():
                parts.append(ch)
            else:
                parts.append(ch)
            i += 1
        return "".join(parts)

    # ── Internal: IPA → Target ────────────────────────────────

    def _from_ipa(self, ipa: str, script: Script) -> str:
        if script == Script.NKO:
            return self._ipa_to_nko(ipa)
        elif script == Script.ARABIC:
            return self._ipa_to_arabic(ipa)
        else:
            return self._ipa_to_latin(ipa)

    def _ipa_to_latin(self, ipa: str) -> str:
        parts: list[str] = []
        i = 0
        while i < len(ipa):
            matched = False
            for key in self._ipa_to_latin_keys:
                kl = len(key)
                if ipa[i:i + kl] == key:
                    parts.append(IPA_TO_LATIN[key])
                    i += kl
                    matched = True
                    break
            if not matched:
                parts.append(ipa[i])
                i += 1
        return "".join(parts)

    def _ipa_to_nko(self, ipa: str) -> str:
        parts: list[str] = []
        i = 0
        while i < len(ipa):
            matched = False
            for key in self._ipa_to_nko_keys:
                kl = len(key)
                if ipa[i:i + kl] == key:
                    parts.append(IPA_TO_NKO[key])
                    i += kl
                    matched = True
                    break
            if not matched:
                ch = ipa[i]
                if ch.isspace():
                    parts.append(ch)
                else:
                    parts.append(ch)  # Pass through
                i += 1
        return "".join(parts)

    def _ipa_to_arabic(self, ipa: str) -> str:
        parts: list[str] = []
        i = 0
        while i < len(ipa):
            matched = False
            for key in self._ipa_to_arabic_keys:
                kl = len(key)
                if ipa[i:i + kl] == key:
                    parts.append(IPA_TO_ARABIC[key])
                    i += kl
                    matched = True
                    break
            if not matched:
                ch = ipa[i]
                if ch.isspace():
                    parts.append(ch)
                elif ch.isdigit():
                    # Western → Arabic-Indic digits
                    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
                    parts.append(arabic_digits[int(ch)])
                else:
                    parts.append(ch)
                i += 1
        return "".join(parts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Module-Level Convenience Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_DEFAULT_ENGINE = NkoTransliterator()


def transliterate(
    text: str,
    source: Optional[Union[str, Script]] = None,
    target: Union[str, Script] = "latin",
) -> str:
    """
    One-call transliteration: detect source, convert to *target* script.

    This is the primary public API — import and call directly:

        from nko.transliterate import transliterate

        transliterate("ߒߞߏ")                # → "nko"
        transliterate("hello", target="nko")  # → "ߤߍߟߟߋ"

    Args:
        text: Input text.
        source: Source script (auto-detect if None).
        target: Target script (default "latin").

    Returns:
        Converted text as a plain string.
    """
    return _DEFAULT_ENGINE.convert(text, source, target).target_text


def convert(
    text: str,
    source: Optional[Union[str, Script]] = None,
    target: Union[str, Script] = "latin",
) -> TranslitResult:
    """
    Like transliterate() but returns a full TranslitResult with metadata.
    """
    return _DEFAULT_ENGINE.convert(text, source, target)


def convert_all(text: str, source: Optional[Union[str, Script]] = None) -> Dict[str, str]:
    """
    Convert *text* to all three scripts. Returns {"nko": ..., "latin": ..., "arabic": ...}.
    """
    return _DEFAULT_ENGINE.convert_all(text, source)


def batch(
    texts: Sequence[str],
    source: Optional[Union[str, Script]] = None,
    target: Union[str, Script] = "latin",
) -> List[TranslitResult]:
    """
    Batch-convert a list of strings.
    """
    return _DEFAULT_ENGINE.batch(texts, source, target)


def to_ipa(text: str, source: Optional[Union[str, Script]] = None) -> str:
    """
    Convert text to IPA representation.
    """
    return _DEFAULT_ENGINE.to_ipa(text, source)


def analyze(text: str) -> Dict:
    """
    Analyze text script composition.
    """
    return _DEFAULT_ENGINE.analyze(text)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _coerce_script(val: Union[str, Script]) -> Script:
    """Normalize a string or Script to Script enum."""
    if isinstance(val, Script):
        return val
    v = str(val).lower().strip()
    try:
        return Script(v)
    except ValueError:
        raise ValueError(
            f"Unknown script {val!r}. Expected one of: {[s.value for s in Script]}"
        ) from None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Public exports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

__all__ = [
    # Main class
    "NkoTransliterator",
    # Convenience functions
    "transliterate",
    "convert",
    "convert_all",
    "batch",
    "to_ipa",
    "analyze",
    "detect_script",
    # Helpers
    "is_nko",
    "is_arabic",
    "is_latin",
    # Types
    "Script",
    "TranslitResult",
    # Character maps (for advanced users / phonetics integration)
    "NKO_TO_IPA",
    "IPA_TO_NKO",
    "IPA_TO_LATIN",
    "IPA_TO_ARABIC",
    "ARABIC_TO_IPA",
    "NKO_VOWELS_TO_IPA",
    "NKO_CONSONANTS_TO_IPA",
    "NKO_TONE_MARKS",
    "NKO_DIGITS_TO_WESTERN",
]
