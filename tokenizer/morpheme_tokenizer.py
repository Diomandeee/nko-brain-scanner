#!/usr/bin/env python3
"""
Morpheme-Aware Tokenizer for N'Ko

Exploits N'Ko's agglutinative morphology: segments text using the
morphological analyzer first, then applies BPE *within* morpheme
boundaries. This prevents merges from crossing linguistic boundaries
(e.g., a root suffix merging with a postposition prefix).

Pipeline:
  text → morphological analysis → per-morpheme BPE → token sequence

For high-confidence words (>0.4): split at morpheme boundaries, BPE within each
For low-confidence words: fall back to standard BPE (same as current tokenizer)

Maintains separate merge tables for: roots, affixes, particles, unknown.
Uses [MORPH_SEP] token to mark morpheme boundaries (aids model learning).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add NKo project to path
_NKO_ROOT = Path.home() / "Desktop" / "NKo"
if str(_NKO_ROOT) not in sys.path:
    sys.path.insert(0, str(_NKO_ROOT))

from nko.morphology import MorphologicalAnalyzer, WordAnalysis, MorphemeType
from nko.phonetics import (
    NKoPhonetics,
    VOWEL_CHARS,
    CONSONANT_CHARS,
    TONE_MARK_CHARS,
    COMBINING_CHARS,
    ALL_NKO_CHARS,
)

NKO_START = 0x07C0
NKO_END = 0x07FF

# Manding particles (function words — tokenized as whole units)
MANDING_PARTICLES = frozenset({
    "ߓߍ", "ߕߍ", "ߞߊ", "ߦߋ", "ߟߊ", "ߣߊ", "ߘߐ",
    "ߝߍ", "ߞߊ߲", "ߡߊ", "ߣߌ߫", "ߥߊ߬", "ߘߏ", "ߡߌ߬",
})


@dataclass
class MorphToken:
    """A token produced by morpheme-aware tokenization."""
    text: str
    token_id: int
    morpheme_type: str  # "root", "suffix", "particle", "bpe", "char", "special"
    morpheme_boundary: bool = False  # True if this starts a new morpheme


class MorphemeAwareTokenizer:
    """
    Tokenizer that respects N'Ko morphological structure.

    Compared to pure BPE:
    - Roots stay as single tokens when possible
    - Affixes (suffixes, postpositions) have dedicated tokens
    - BPE merges only happen within morpheme boundaries
    - [MORPH_SEP] marks boundaries for the model to learn from
    """

    # Special tokens
    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3
    SEP_ID = 4
    MORPH_SEP_ID = 5
    SPECIAL_COUNT = 6

    def __init__(
        self,
        vocab_path: Optional[str] = None,
        confidence_threshold: float = 0.4,
        use_morph_sep: bool = True,
    ):
        self._analyzer = MorphologicalAnalyzer()
        self._phonetics = NKoPhonetics()
        self._confidence_threshold = confidence_threshold
        self._use_morph_sep = use_morph_sep

        # BPE merge tables per morpheme category
        self._root_merges: List[Tuple[str, str]] = []
        self._affix_merges: List[Tuple[str, str]] = []
        self._general_merges: List[Tuple[str, str]] = []

        # Token maps
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}

        if vocab_path and Path(vocab_path).exists():
            self._load_vocab(vocab_path)
        else:
            self._build_default_vocab()

    def _build_default_vocab(self):
        """Build vocabulary from N'Ko characters + morphemes."""
        t2i = {
            "[PAD]": self.PAD_ID,
            "[UNK]": self.UNK_ID,
            "[BOS]": self.BOS_ID,
            "[EOS]": self.EOS_ID,
            "[SEP]": self.SEP_ID,
            "[MORPH_SEP]": self.MORPH_SEP_ID,
        }
        idx = self.SPECIAL_COUNT

        # All N'Ko characters
        for cp in range(NKO_START, NKO_END + 1):
            ch = chr(cp)
            if ch not in t2i:
                t2i[ch] = idx
                idx += 1

        # Space
        t2i[" "] = idx
        idx += 1

        # Manding particles as whole tokens
        for particle in sorted(MANDING_PARTICLES):
            if particle not in t2i:
                t2i[particle] = idx
                idx += 1

        # Common morphemes from analyzer
        for morph_text in self._get_analyzer_morphemes():
            if morph_text not in t2i:
                t2i[morph_text] = idx
                idx += 1

        self._token_to_id = t2i
        self._id_to_token = {v: k for k, v in t2i.items()}

    def _get_analyzer_morphemes(self) -> List[str]:
        """Extract morphemes from the analyzer's built-in inventories."""
        morphemes = []
        seen = set()

        for source in [
            self._analyzer.subject_pronouns,
            self._analyzer.tense_markers,
            self._analyzer.postpositions,
            self._analyzer.derivation_suffixes,
        ]:
            for item in source:
                for text in [item.nko, item.latin]:
                    if text and text.strip() and text.strip() not in seen:
                        seen.add(text.strip())
                        morphemes.append(text.strip())

        return morphemes

    def _load_vocab(self, path: str):
        """Load vocabulary and merge tables from JSON."""
        with open(path) as f:
            data = json.load(f)

        self._token_to_id = data.get("token_to_id", {})
        self._id_to_token = {int(v): k for k, v in self._token_to_id.items()}

        # Load category-specific merges
        self._root_merges = [tuple(m) for m in data.get("root_merges", [])]
        self._affix_merges = [tuple(m) for m in data.get("affix_merges", [])]
        self._general_merges = [tuple(m) for m in data.get("general_merges", [])]

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    def _is_nko_text(self, text: str) -> bool:
        return any(NKO_START <= ord(ch) <= NKO_END for ch in text)

    def _split_to_units(self, text: str) -> List[str]:
        """Split text into base+tone units (tone marks attach to preceding base)."""
        units = []
        current = ""
        for ch in text:
            cp = ord(ch)
            if ch in TONE_MARK_CHARS or ch in COMBINING_CHARS or (0x07EB <= cp <= 0x07F5):
                current += ch
            else:
                if current:
                    units.append(current)
                current = ch
        if current:
            units.append(current)
        return units

    def _apply_merges(self, units: List[str], merges: List[Tuple[str, str]]) -> List[str]:
        """Apply BPE merges to a sequence of units."""
        for left, right in merges:
            new_units = []
            i = 0
            while i < len(units):
                if i < len(units) - 1 and units[i] == left and units[i + 1] == right:
                    new_units.append(left + right)
                    i += 2
                else:
                    new_units.append(units[i])
                    i += 1
            units = new_units
            if len(units) <= 1:
                break
        return units

    def _select_merges(self, morpheme_type: str) -> List[Tuple[str, str]]:
        """Select appropriate merge table for morpheme type."""
        if morpheme_type in ("root",):
            return self._root_merges if self._root_merges else self._general_merges
        elif morpheme_type in ("suffix", "prefix", "postposition", "derivation"):
            return self._affix_merges if self._affix_merges else self._general_merges
        return self._general_merges

    def _lookup_id(self, text: str) -> int:
        return self._token_to_id.get(text, self.UNK_ID)

    def _tokenize_morpheme(self, text: str, morph_type: str) -> List[MorphToken]:
        """Tokenize a single morpheme using category-appropriate BPE."""
        # Check if whole morpheme is in vocab
        tid = self._lookup_id(text)
        if tid != self.UNK_ID:
            return [MorphToken(text=text, token_id=tid, morpheme_type=morph_type)]

        # Fall back to BPE within this morpheme
        units = self._split_to_units(text)
        merges = self._select_merges(morph_type)
        units = self._apply_merges(units, merges)

        tokens = []
        for unit in units:
            tid = self._lookup_id(unit)
            tokens.append(MorphToken(
                text=unit,
                token_id=tid,
                morpheme_type="bpe" if len(unit) > 1 else "char",
            ))
        return tokens

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into morpheme-aware strings."""
        return [t.text for t in self._tokenize_internal(text)]

    def _tokenize_internal(self, text: str) -> List[MorphToken]:
        """Core tokenization producing MorphToken objects."""
        text = text.strip()
        if not text:
            return []

        words = text.split()
        all_tokens: List[MorphToken] = []

        for i, word in enumerate(words):
            if i > 0:
                all_tokens.append(MorphToken("[SEP]", self.SEP_ID, "special"))

            # Check if it's a known particle (whole-token match)
            if word in MANDING_PARTICLES or NKoPhonetics.strip_tones(word) in MANDING_PARTICLES:
                tid = self._lookup_id(word)
                if tid == self.UNK_ID:
                    clean = NKoPhonetics.strip_tones(word)
                    tid = self._lookup_id(clean)
                all_tokens.append(MorphToken(
                    text=word, token_id=tid, morpheme_type="particle",
                ))
                continue

            # Skip non-N'Ko words (pass through as character tokens)
            if not self._is_nko_text(word):
                for ch in word:
                    tid = self._lookup_id(ch)
                    all_tokens.append(MorphToken(text=ch, token_id=tid, morpheme_type="char"))
                continue

            # Morphological analysis
            analysis = self._analyzer.analyze_word(word)

            if analysis.confidence >= self._confidence_threshold and len(analysis.morphemes) > 0:
                # High confidence: tokenize at morpheme boundaries
                for j, morph in enumerate(analysis.morphemes):
                    morph_text = morph.nko if morph.nko else morph.text

                    # Add morpheme separator between morphemes
                    if j > 0 and self._use_morph_sep:
                        all_tokens.append(MorphToken(
                            "[MORPH_SEP]", self.MORPH_SEP_ID, "special",
                            morpheme_boundary=True,
                        ))

                    morph_tokens = self._tokenize_morpheme(
                        morph_text, morph.morpheme_type.value,
                    )
                    if morph_tokens:
                        morph_tokens[0].morpheme_boundary = (j > 0)
                    all_tokens.extend(morph_tokens)
            else:
                # Low confidence: standard BPE on whole word
                units = self._split_to_units(word)
                units = self._apply_merges(units, self._general_merges)
                for unit in units:
                    tid = self._lookup_id(unit)
                    all_tokens.append(MorphToken(
                        text=unit, token_id=tid,
                        morpheme_type="bpe" if len(unit) > 1 else "char",
                    ))

        return all_tokens

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """Encode text to token IDs."""
        tokens = self._tokenize_internal(text)
        ids = [t.token_id for t in tokens]
        if add_special:
            ids = [self.BOS_ID] + ids + [self.EOS_ID]
        return ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        parts = []
        for tid in token_ids:
            if tid in (self.PAD_ID, self.BOS_ID, self.EOS_ID, self.MORPH_SEP_ID):
                continue
            if tid == self.SEP_ID:
                parts.append(" ")
                continue
            parts.append(self._id_to_token.get(tid, ""))
        return "".join(parts)

    def morpheme_boundary_tokens(self, text: str) -> List[bool]:
        """Return boolean mask indicating morpheme boundaries."""
        tokens = self._tokenize_internal(text)
        return [t.morpheme_boundary for t in tokens]

    def analyze_compression(self, text: str) -> Dict:
        """Analyze tokenization compression on given text."""
        morph_tokens = self._tokenize_internal(text)
        morph_count = len(morph_tokens)

        # Compare with character-level
        nko_chars = [ch for ch in text if NKO_START <= ord(ch) <= NKO_END]
        char_count = len(nko_chars) if nko_chars else len(text)

        # Count complete syllables
        syllable_tokens = 0
        for t in morph_tokens:
            if t.morpheme_type != "special":
                has_vowel = any(ch in VOWEL_CHARS for ch in t.text)
                has_consonant = any(ch in CONSONANT_CHARS for ch in t.text)
                if has_vowel:
                    syllable_tokens += 1

        return {
            "text_length": len(text),
            "nko_char_count": len(nko_chars),
            "token_count": morph_count,
            "compression_ratio": char_count / max(morph_count, 1),
            "syllable_aligned_tokens": syllable_tokens,
            "syllable_alignment_ratio": syllable_tokens / max(morph_count, 1),
            "morpheme_types": Counter(t.morpheme_type for t in morph_tokens),
        }

    def save_vocab(self, path: str):
        """Save vocabulary and merge tables."""
        data = {
            "token_to_id": self._token_to_id,
            "vocab_size": self.vocab_size,
            "special_tokens": {
                "[PAD]": self.PAD_ID,
                "[UNK]": self.UNK_ID,
                "[BOS]": self.BOS_ID,
                "[EOS]": self.EOS_ID,
                "[SEP]": self.SEP_ID,
                "[MORPH_SEP]": self.MORPH_SEP_ID,
            },
            "root_merges": [list(m) for m in self._root_merges],
            "affix_merges": [list(m) for m in self._affix_merges],
            "general_merges": [list(m) for m in self._general_merges],
            "confidence_threshold": self._confidence_threshold,
        }
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._token_to_id)
