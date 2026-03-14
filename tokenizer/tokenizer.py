"""
NKO-aware tokenizer for LLM activation profiling.

Segments N'Ko text at morpheme boundaries using the existing
NKo Unified Platform morphology engine. Tone marks are treated
as attributes on preceding tokens, not separate tokens.

Usage::

    tok = NkoTokenizer()
    ids = tok.encode("ߒ ߓߍ߬ ߕߊ߬ߡߌ߲")
    text = tok.decode(ids)
    morphemes = tok.tokenize("ߒ ߓߍ߬ ߕߊ߬ߡߌ߲")
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

# Add NKo project to path so we can import the real modules
_NKO_ROOT = Path.home() / "Desktop" / "NKo"
if str(_NKO_ROOT) not in sys.path:
    sys.path.insert(0, str(_NKO_ROOT))

from nko.morphology import MorphologicalAnalyzer, WordAnalysis, MorphemeType
from nko.phonetics import (
    NKoPhonetics,
    ToneType,
    TONE_MARK_CHARS,
    COMBINING_CHARS,
    LETTER_CHARS,
    DIGIT_CHARS,
    ALL_NKO_CHARS,
)


# Special token IDs
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
SEP_ID = 4  # Word boundary marker
SPECIAL_COUNT = 5


@dataclass
class Token:
    """A single token with optional tone attribute."""
    text: str
    token_id: int
    tone: Optional[ToneType] = None
    morpheme_type: Optional[str] = None


class NkoTokenizer:
    """
    NKO-aware tokenizer that segments at morpheme boundaries.

    Strategy (v2 with BPE):
    1. Split text on whitespace into words
    2. Run MorphologicalAnalyzer.analyze_word() on each word
    3. For high-confidence results (>0.4), use morpheme boundaries
    4. For unknown words, apply BPE merges then fall back to char-level
    5. Tone marks attach to preceding base character (not separate tokens)

    The vocabulary is built from:
    - 5 special tokens: [PAD], [UNK], [BOS], [EOS], [SEP]
    - 64 NKO characters (U+07C0-U+07FF)
    - 512 BPE subword merges learned from N'Ko corpus
    - Common morphemes from morphological analyzer
    """

    def __init__(self, vocab_path: Optional[str] = None, use_bpe: bool = True):
        self._analyzer = MorphologicalAnalyzer()
        self._phonetics = NKoPhonetics()
        self._bpe_merges: List[Tuple[str, str]] = []

        # Try BPE vocab first, then fall back to original
        if vocab_path is None and use_bpe:
            bpe_path = Path(__file__).parent / "bpe_vocab.json"
            if bpe_path.exists():
                vocab_path = str(bpe_path)
            else:
                vocab_path = str(Path(__file__).parent / "vocab.json")
        elif vocab_path is None:
            vocab_path = str(Path(__file__).parent / "vocab.json")

        vp = Path(vocab_path)
        if vp.exists():
            with open(vp) as f:
                data = json.load(f)
            self._token_to_id: Dict[str, int] = data.get("token_to_id", {})
            self._id_to_token: Dict[int, str] = {
                int(v): k for k, v in self._token_to_id.items()
            }
            # Load BPE merges if present
            if "merges" in data:
                self._bpe_merges = [tuple(m) for m in data["merges"]]
        else:
            # Build minimal vocab from phonetics tables
            self._token_to_id, self._id_to_token = self._build_default_vocab()

    def _build_default_vocab(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build vocabulary from NKO character tables."""
        t2i: Dict[str, int] = {
            "[PAD]": PAD_ID,
            "[UNK]": UNK_ID,
            "[BOS]": BOS_ID,
            "[EOS]": EOS_ID,
            "[SEP]": SEP_ID,
        }
        idx = SPECIAL_COUNT

        # All NKO letters (vowels + consonants)
        for ch in sorted(LETTER_CHARS):
            t2i[ch] = idx
            idx += 1

        # NKO digits
        for ch in sorted(DIGIT_CHARS):
            t2i[ch] = idx
            idx += 1

        # Common morphemes from the analyzer's built-in inventories
        common_morphemes = self._extract_common_morphemes()
        for morph in common_morphemes:
            if morph not in t2i:
                t2i[morph] = idx
                idx += 1

        i2t = {v: k for k, v in t2i.items()}
        return t2i, i2t

    def _extract_common_morphemes(self) -> List[str]:
        """Extract common morphemes from the analyzer's built-in data."""
        morphemes = []

        # Subject pronouns
        for p in self._analyzer.subject_pronouns:
            if p.nko:
                morphemes.append(p.nko)
            if p.latin:
                morphemes.append(p.latin)

        # Tense markers
        for tm in self._analyzer.tense_markers:
            if tm.nko:
                morphemes.append(tm.nko)
            if tm.latin:
                morphemes.append(tm.latin)

        # Postpositions
        for pp in self._analyzer.postpositions:
            if pp.nko:
                morphemes.append(pp.nko)
            if pp.latin:
                morphemes.append(pp.latin)

        # Derivation suffixes from affix inventory
        for sfx in self._analyzer.derivation_suffixes:
            if sfx.nko:
                morphemes.append(sfx.nko)
            if sfx.latin:
                morphemes.append(sfx.latin)

        # Deduplicate while preserving order
        seen = set()
        result = []
        for m in morphemes:
            m_clean = m.strip()
            if m_clean and m_clean not in seen:
                seen.add(m_clean)
                result.append(m_clean)
        return result

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    def _strip_tones(self, text: str) -> Tuple[str, List[Tuple[int, ToneType]]]:
        """Strip tone marks and return (clean_text, tone_positions)."""
        tones = NKoPhonetics.extract_tones(text)
        clean = NKoPhonetics.strip_tones(text)
        return clean, tones

    def _lookup_id(self, token_text: str) -> int:
        """Look up token ID, falling back to character-level then UNK."""
        if token_text in self._token_to_id:
            return self._token_to_id[token_text]
        return UNK_ID

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

    def _apply_bpe(self, units: List[str]) -> List[str]:
        """Apply BPE merges to a sequence of character units."""
        if not self._bpe_merges:
            return units
        for left, right in self._bpe_merges:
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

    def _char_tokenize(self, text: str) -> List[Token]:
        """BPE-enhanced character tokenization for unknown morphemes."""
        # Split into base+tone units, then apply BPE merges
        units = self._split_to_units(text)
        units = self._apply_bpe(units)

        tokens = []
        for unit in units:
            tid = self._lookup_id(unit)
            # Extract tone from the unit
            tone_info = NKoPhonetics.extract_tones(unit)
            tone = tone_info[0][1] if tone_info else None
            tokens.append(Token(
                text=unit,
                token_id=tid,
                tone=tone,
                morpheme_type="bpe" if len(unit) > 1 else "char",
            ))
        return tokens

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize NKO text into morpheme strings (no ID conversion).

        Returns list of morpheme text segments. Tone marks are stripped
        but recorded as attributes internally.
        """
        tokens = self._tokenize_internal(text)
        return [t.text for t in tokens]

    def _tokenize_internal(self, text: str) -> List[Token]:
        """Core tokenization producing Token objects."""
        text = text.strip()
        if not text:
            return []

        words = text.split()
        all_tokens: List[Token] = []

        for i, word in enumerate(words):
            if i > 0:
                all_tokens.append(Token("[SEP]", SEP_ID, morpheme_type="special"))

            # Strip tones before analysis, record them
            clean_word, tones = self._strip_tones(word)

            # Analyze with morphology engine
            analysis = self._analyzer.analyze_word(word)

            if analysis.confidence >= 0.4 and len(analysis.morphemes) > 0:
                # Use morpheme-level tokenization
                for morph in analysis.morphemes:
                    morph_text = morph.nko if morph.nko else morph.text
                    morph_clean = NKoPhonetics.strip_tones(morph_text)

                    # Check if morpheme is in vocab
                    tid = self._lookup_id(morph_clean)
                    if tid != UNK_ID:
                        # Extract tone from original morpheme text
                        morph_tones = NKoPhonetics.extract_tones(morph_text)
                        tone = morph_tones[0][1] if morph_tones else None
                        all_tokens.append(Token(
                            text=morph_clean,
                            token_id=tid,
                            tone=tone,
                            morpheme_type=morph.morpheme_type.value,
                        ))
                    else:
                        # Fall back to character-level for this morpheme
                        all_tokens.extend(self._char_tokenize(morph_text))
            else:
                # Low confidence: character-level tokenization
                all_tokens.extend(self._char_tokenize(word))

        return all_tokens

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """
        Tokenize NKO text into token IDs.

        Parameters
        ----------
        text : str
            NKO text to tokenize.
        add_special : bool
            If True, wrap with [BOS] and [EOS] tokens.

        Returns
        -------
        List[int]
            Token IDs.
        """
        tokens = self._tokenize_internal(text)
        ids = [t.token_id for t in tokens]
        if add_special:
            ids = [BOS_ID] + ids + [EOS_ID]
        return ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Convert token IDs back to text.

        Strips special tokens ([PAD], [BOS], [EOS]) and joins
        morphemes. [SEP] tokens become spaces.
        """
        parts = []
        for tid in token_ids:
            if tid in (PAD_ID, BOS_ID, EOS_ID):
                continue
            if tid == SEP_ID:
                parts.append(" ")
                continue
            token_text = self._id_to_token.get(tid, "")
            parts.append(token_text)
        return "".join(parts)

    def compare_with_hf(self, text: str, hf_tokenizer) -> Dict:
        """
        Compare NKO morpheme tokenization with HuggingFace tokenizer.

        Returns compression metrics showing how many tokens each
        approach uses for equivalent semantic content.
        """
        nko_tokens = self.tokenize(text)
        hf_tokens = hf_tokenizer.tokenize(text)
        return {
            "text": text,
            "nko_token_count": len(nko_tokens),
            "hf_token_count": len(hf_tokens),
            "nko_tokens": nko_tokens,
            "hf_tokens": hf_tokens,
            "compression_ratio": len(hf_tokens) / max(len(nko_tokens), 1),
        }

    def batch_encode(self, texts: List[str], add_special: bool = True) -> List[List[int]]:
        """Encode a batch of texts."""
        return [self.encode(t, add_special) for t in texts]

    def get_vocab(self) -> Dict[str, int]:
        """Return the full vocabulary mapping."""
        return dict(self._token_to_id)

    def save_vocab(self, path: str) -> None:
        """Save vocabulary to JSON file."""
        data = {
            "token_to_id": self._token_to_id,
            "vocab_size": self.vocab_size,
            "special_tokens": {
                "[PAD]": PAD_ID,
                "[UNK]": UNK_ID,
                "[BOS]": BOS_ID,
                "[EOS]": EOS_ID,
                "[SEP]": SEP_ID,
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
