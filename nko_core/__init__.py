"""
nko_core — Unified N'Ko linguistic layer for the Brain Scanner project.

Re-exports from ~/Desktop/NKo/ canonical library:
  - phonetics: Character classification, IPA, tone marks, syllabification
  - transliterate: N'Ko ↔ Latin ↔ Arabic via IPA intermediary
  - morphology: Word decomposition, conjugation, compound detection

Usage:
    from nko_core import phonetics, transliterate, morphology
    from nko_core.phonetics import NKoPhonetics, VOWEL_CHARS, CONSONANT_CHARS
    from nko_core.transliterate import transliterate as nko_to_latin
    from nko_core.morphology import MorphologicalAnalyzer
"""

import sys
from pathlib import Path

# Add the canonical NKo library to the import path
_NKO_LIB = Path.home() / "Desktop" / "NKo"
if _NKO_LIB.exists() and str(_NKO_LIB) not in sys.path:
    sys.path.insert(0, str(_NKO_LIB))

# Re-export core modules
try:
    from nko import phonetics
    from nko import transliterate
    from nko import morphology
except ImportError as e:
    import warnings
    warnings.warn(
        f"Could not import from ~/Desktop/NKo/nko/: {e}. "
        f"Ensure the NKo library is installed at {_NKO_LIB}"
    )

# Convenience re-exports
try:
    from nko.phonetics import (
        NKoPhonetics,
        ToneType,
        CharCategory,
        CharInfo,
        Phoneme,
    )
    from nko.transliterate import (
        NkoTransliterator,
        transliterate as nko_transliterate,
    )
    from nko.morphology import (
        MorphologicalAnalyzer,
        MorphemeType,
        WordAnalysis,
        Morpheme,
    )
except ImportError:
    pass
