"""
N'Ko Syllable Codebook — Exhaustive enumeration of valid N'Ko syllables.

N'Ko has strict CV/CVN syllable structure:
  - CV:  consonant + vowel (+ optional tone)
  - CVN: consonant + vowel + nasal coda (+ optional tone)
  - V:   vowel-initial syllable (+ optional tone)
  - VN:  vowel + nasal coda (+ optional tone)

With 26 consonants, 7 vowels, ~3 nasal codas, and 5 tone marks,
the total codebook size is ~3,640 tonal syllables.

This codebook serves as the retrieval target for the joint-embedding ASR.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional

# Import from canonical NKo library
import sys
_NKO_LIB = Path.home() / "Desktop" / "NKo"
if _NKO_LIB.exists() and str(_NKO_LIB) not in sys.path:
    sys.path.insert(0, str(_NKO_LIB))

from nko.phonetics import NKoPhonetics

_phon = NKoPhonetics()


# N'Ko character sets (from phonetics.py)
VOWELS = [ch for ch in "ߊߋߌߍߎߏߐ"]  # 7 vowels
CONSONANTS = [
    ch for ch in "ߒߓߔߕߖߗߘߙߚߛߜߝߞߟߠߡߢߣߤߥߦߧߨ"
]
TONE_MARKS = [ch for ch in "߫߬߭߮߯"]  # HIGH, LOW, RISING, LONG, VERY_LONG
NASAL_CODAS = ["\u07F2", "\u07F3"]  # ߲ (nasalization), ߳ (alt nasal)
NASAL_LETTER = "ߒ"  # /n/ as syllable-final nasal


@dataclass(frozen=True)
class Syllable:
    """A single N'Ko syllable with all metadata."""
    nko: str           # N'Ko script representation
    ipa: str           # IPA pronunciation
    onset: str         # Consonant onset (empty for V/VN)
    nucleus: str       # Vowel nucleus
    coda: str          # Nasal coda (empty for CV/V)
    tone: str          # Tone mark (empty for unmarked)
    pattern: str       # CV, CVN, V, VN
    index: int         # Codebook index (0-based)


class NKoSyllableCodebook:
    """
    Exhaustive codebook of all valid N'Ko syllables.

    Used as the retrieval target for the joint-embedding ASR architecture.
    Each syllable gets a unique index for embedding lookup.
    """

    def __init__(self):
        self.syllables: List[Syllable] = []
        self._nko_to_idx: Dict[str, int] = {}
        self._ipa_to_idx: Dict[str, int] = {}
        self._build()

    def _build(self):
        """Enumerate all valid N'Ko syllables."""
        idx = 0

        # Pattern 1: V (vowel only, no tone)
        for v in VOWELS:
            ipa = _phon.char_to_ipa(v)
            syl = Syllable(
                nko=v, ipa=ipa, onset="", nucleus=v,
                coda="", tone="", pattern="V", index=idx,
            )
            self.syllables.append(syl)
            self._nko_to_idx[v] = idx
            self._ipa_to_idx[ipa] = idx
            idx += 1

        # Pattern 2: V + tone
        for v in VOWELS:
            for t in TONE_MARKS:
                nko = v + t
                v_ipa = _phon.char_to_ipa(v)
                t_ipa = _phon.char_to_ipa(t)
                ipa = v_ipa + t_ipa
                syl = Syllable(
                    nko=nko, ipa=ipa, onset="", nucleus=v,
                    coda="", tone=t, pattern="V", index=idx,
                )
                self.syllables.append(syl)
                self._nko_to_idx[nko] = idx
                self._ipa_to_idx[ipa] = idx
                idx += 1

        # Pattern 3: VN (vowel + nasal coda)
        for v in VOWELS:
            for n in NASAL_CODAS:
                nko = v + n
                ipa = _phon.char_to_ipa(v) + "\u0303"  # nasalized
                syl = Syllable(
                    nko=nko, ipa=ipa, onset="", nucleus=v,
                    coda=n, tone="", pattern="VN", index=idx,
                )
                self.syllables.append(syl)
                self._nko_to_idx[nko] = idx
                idx += 1

        # Pattern 4: VN + tone
        for v in VOWELS:
            for n in NASAL_CODAS:
                for t in TONE_MARKS:
                    nko = v + t + n  # tone before nasal in N'Ko
                    ipa = _phon.char_to_ipa(v) + "\u0303"
                    syl = Syllable(
                        nko=nko, ipa=ipa, onset="", nucleus=v,
                        coda=n, tone=t, pattern="VN", index=idx,
                    )
                    self.syllables.append(syl)
                    self._nko_to_idx[nko] = idx
                    idx += 1

        # Pattern 5: CV (consonant + vowel)
        for c in CONSONANTS:
            for v in VOWELS:
                nko = c + v
                c_ipa = _phon.char_to_ipa(c)
                v_ipa = _phon.char_to_ipa(v)
                ipa = c_ipa + v_ipa
                syl = Syllable(
                    nko=nko, ipa=ipa, onset=c, nucleus=v,
                    coda="", tone="", pattern="CV", index=idx,
                )
                self.syllables.append(syl)
                self._nko_to_idx[nko] = idx
                self._ipa_to_idx[ipa] = idx
                idx += 1

        # Pattern 6: CV + tone
        for c in CONSONANTS:
            for v in VOWELS:
                for t in TONE_MARKS:
                    nko = c + v + t
                    c_ipa = _phon.char_to_ipa(c)
                    v_ipa = _phon.char_to_ipa(v)
                    t_ipa = _phon.char_to_ipa(t)
                    ipa = c_ipa + v_ipa + t_ipa
                    syl = Syllable(
                        nko=nko, ipa=ipa, onset=c, nucleus=v,
                        coda="", tone=t, pattern="CV", index=idx,
                    )
                    self.syllables.append(syl)
                    self._nko_to_idx[nko] = idx
                    idx += 1

        # Pattern 7: CVN (consonant + vowel + nasal)
        for c in CONSONANTS:
            for v in VOWELS:
                for n in NASAL_CODAS:
                    nko = c + v + n
                    c_ipa = _phon.char_to_ipa(c)
                    v_ipa = _phon.char_to_ipa(v)
                    ipa = c_ipa + v_ipa + "\u0303"
                    syl = Syllable(
                        nko=nko, ipa=ipa, onset=c, nucleus=v,
                        coda=n, tone="", pattern="CVN", index=idx,
                    )
                    self.syllables.append(syl)
                    self._nko_to_idx[nko] = idx
                    idx += 1

        # Pattern 8: CVN + tone
        for c in CONSONANTS:
            for v in VOWELS:
                for n in NASAL_CODAS:
                    for t in TONE_MARKS:
                        nko = c + v + t + n
                        c_ipa = _phon.char_to_ipa(c)
                        v_ipa = _phon.char_to_ipa(v)
                        ipa = c_ipa + v_ipa + "\u0303"
                        syl = Syllable(
                            nko=nko, ipa=ipa, onset=c, nucleus=v,
                            coda=n, tone=t, pattern="CVN", index=idx,
                        )
                        self.syllables.append(syl)
                        self._nko_to_idx[nko] = idx
                        idx += 1

    def __len__(self) -> int:
        return len(self.syllables)

    def __getitem__(self, idx: int) -> Syllable:
        return self.syllables[idx]

    def lookup(self, nko: str) -> Optional[Syllable]:
        """Look up a syllable by its N'Ko representation."""
        idx = self._nko_to_idx.get(nko)
        return self.syllables[idx] if idx is not None else None

    def lookup_ipa(self, ipa: str) -> Optional[Syllable]:
        """Look up a syllable by its IPA representation."""
        idx = self._ipa_to_idx.get(ipa)
        return self.syllables[idx] if idx is not None else None

    def by_pattern(self, pattern: str) -> List[Syllable]:
        """Get all syllables matching a pattern (CV, CVN, V, VN)."""
        return [s for s in self.syllables if s.pattern == pattern]

    def stats(self) -> Dict:
        """Return codebook statistics."""
        patterns = {}
        for s in self.syllables:
            patterns[s.pattern] = patterns.get(s.pattern, 0) + 1
        return {
            "total": len(self.syllables),
            "patterns": patterns,
            "consonants": len(CONSONANTS),
            "vowels": len(VOWELS),
            "tones": len(TONE_MARKS),
            "nasals": len(NASAL_CODAS),
        }

    def save(self, path: str):
        """Save codebook to JSON."""
        data = {
            "stats": self.stats(),
            "syllables": [asdict(s) for s in self.syllables],
        }
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "NKoSyllableCodebook":
        """Load codebook from JSON."""
        cb = cls.__new__(cls)
        cb.syllables = []
        cb._nko_to_idx = {}
        cb._ipa_to_idx = {}
        with open(path) as f:
            data = json.load(f)
        for entry in data["syllables"]:
            syl = Syllable(**entry)
            cb.syllables.append(syl)
            cb._nko_to_idx[syl.nko] = syl.index
            if syl.ipa:
                cb._ipa_to_idx[syl.ipa] = syl.index
        return cb


if __name__ == "__main__":
    cb = NKoSyllableCodebook()
    stats = cb.stats()
    print(f"N'Ko Syllable Codebook: {stats['total']} entries")
    for pattern, count in sorted(stats["patterns"].items()):
        print(f"  {pattern}: {count}")

    # Show examples
    print("\nExamples:")
    for pattern in ["V", "CV", "CVN", "VN"]:
        examples = cb.by_pattern(pattern)[:3]
        for s in examples:
            print(f"  {s.pattern}: {s.nko} [{s.ipa}] (idx={s.index})")

    # Save
    out = Path(__file__).parent.parent / "data" / "syllable_codebook.json"
    out.parent.mkdir(exist_ok=True)
    cb.save(str(out))
    print(f"\nSaved to {out}")
