"""
ߒߞߏ — N'Ko Unified Platform
Digital infrastructure for 40M+ Manding language speakers

Modules:
    nko.phonetics      — IPA, tone, Unicode character analysis
    nko.transliterate   — N'Ko ↔ Latin ↔ Arabic script bridge
    nko.morphology      — Word decomposition, conjugation, noun classes
    nko.culture         — Proverbs, blessings, greetings, calendar, concepts
"""

__version__ = "0.1.0"

# ── Phonetics ────────────────────────────────────────────────
from nko.phonetics import (
    NKoPhonetics,
    CharInfo,
    Phoneme,
    ToneType,
    CharCategory,
)

# ── Transliteration ──────────────────────────────────────────
from nko.transliterate import (
    NkoTransliterator,
    transliterate,
    detect_script,
    Script,
    TranslitResult,
)

# ── Morphology ───────────────────────────────────────────────
from nko.morphology import (
    MorphologicalAnalyzer,
    VerbConjugator,
    CompoundDetector,
    WordAnalysis,
    MorphemeType,
    NounClass,
    TenseAspect,
)

# ── Culture ──────────────────────────────────────────────────
from nko.culture import (
    NKoCulture,
)

__all__ = [
    "__version__",
    # Phonetics
    "NKoPhonetics",
    "CharInfo",
    "Phoneme",
    "ToneType",
    "CharCategory",
    # Transliteration
    "NkoTransliterator",
    "transliterate",
    "detect_script",
    "Script",
    "TranslitResult",
    # Morphology
    "MorphologicalAnalyzer",
    "VerbConjugator",
    "CompoundDetector",
    "WordAnalysis",
    "MorphemeType",
    "NounClass",
    "TenseAspect",
    # Culture
    "NKoCulture",
]
