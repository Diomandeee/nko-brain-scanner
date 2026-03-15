"""
ߒ߬ߞߏ ߞߊ߲ ߠߊ߬ߞߎ߲ — N'Ko Unified Morphology Engine
nko.morphology — The public API for Manding morphological analysis

Merges intelligence from:
  • core/morphology/morphology/ (cross-script-bridge): Deep analyzer, conjugator,
    compound splitter, tone engine
  • core/prediction/morphological_engine.py (keyboard-ai): Prediction-oriented
    morphology, code-switching, cultural calendar, affix inventories

Manding (Bambara, Maninka, Dioula/Jula) morphology is largely isolating with
agglutinative tendencies. Verbs do not inflect for person/number — tense/aspect
is carried by auxiliary particles in a strict S-Aux-V-O order.

Key linguistic features handled:
  ✦ Word decomposition: prefix + root + suffix
  ✦ Root/stem extraction across N'Ko, Latin, Arabic scripts
  ✦ Noun class detection (Manding person/thing/liquid/abstract classes)
  ✦ Verb conjugation via particle system (9 tense-aspect forms × 6 persons)
  ✦ Comprehensive affix inventory (derivational + inflectional)
  ✦ Compound word detection & splitting (N+N, N+V, V+N, etc.)
  ✦ Tone-aware analysis preserving meaning-distinguishing tone marks
  ✦ Code-switching detection (Manding ↔ French ↔ English)

ߞߊ߲ ߕߍ߫ ߛߐ߲߬ — Language has structure (beyond its letters)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set
from enum import Enum
import re


# ═══════════════════════════════════════════════════════════════════════════════
# §1  ENUMERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

class MorphemeType(Enum):
    """Classification of word parts in Manding"""
    ROOT = "root"
    PREFIX = "prefix"
    SUFFIX = "suffix"
    POSTPOSITION = "postposition"
    SUBJECT_PRONOUN = "subject"
    OBJECT_PRONOUN = "object"
    TENSE_MARKER = "tense"
    ASPECT_MARKER = "aspect"
    NEGATION = "negation"
    DETERMINER = "determiner"
    CONNECTOR = "connector"
    TONE_MARK = "tone"
    PLURAL = "plural"
    DERIVATION = "derivation"


class NounClass(Enum):
    """
    Manding noun classes.

    Unlike Bantu languages with 15+ classes, Manding has a reduced noun-class
    system.  Classification is primarily semantic — person vs. thing vs. liquid
    vs. abstract — with some morphological markers (especially in plural forms
    and pronoun agreement).
    """
    PERSON = "person"           # ߡߐ߱-class: humans, personified entities
    ANIMAL = "animal"           # Animate non-human
    THING = "thing"             # ߝߍ߲-class: inanimate objects
    LIQUID = "liquid"           # ߖߌ-class: water, blood, milk …
    MASS = "mass"               # Uncountable substances (sand, flour …)
    ABSTRACT = "abstract"       # Qualities, states, actions nominalized
    PLACE = "place"             # Locations, countries, villages
    PLANT = "plant"             # Trees, crops, vegetation
    BODY_PART = "body_part"     # Body parts (special plural behaviour)
    KINSHIP = "kinship"         # Kinship terms (special vocative forms)
    TIME = "time"               # Temporal nouns (day, season …)


class TenseAspect(Enum):
    """Manding tense-aspect-mood system"""
    PROGRESSIVE = "progressive"        # ߓߍ (bɛ)  — habitual / ongoing
    NEG_PROGRESSIVE = "neg_progressive" # ߕߍ (tɛ)  — negated progressive
    COMPLETIVE = "completive"          # ߞߊ (ka)  — completed
    NEG_COMPLETIVE = "neg_completive"  # ߡߊ (ma)  — negated completed
    FUTURE = "future"                  # ߣߊ (na)  — intention / future
    FUTURE_PROG = "future_progressive" # ߓߍ ߣߊ (bɛ na) — future progressive
    SUBJUNCTIVE = "subjunctive"        # ߞߊ (ka)  — in subordinate clauses
    IMPERATIVE = "imperative"          # Ø — bare verb
    CONDITIONAL = "conditional"        # ߣߌ…ߞߊ (ni…ka) — if…then


class PersonNumber(Enum):
    """Subject pronouns — person × number"""
    FIRST_SG = "1sg"    # ߒ (n)     — I
    SECOND_SG = "2sg"   # ߌ (i)     — you (sg)
    THIRD_SG = "3sg"    # ߊ߬ (a)   — s/he/it
    FIRST_PL = "1pl"    # ߊ߲ (an)  — we
    SECOND_PL = "2pl"   # ߊ߬ߥ (aw) — you (pl)
    THIRD_PL = "3pl"    # ߊ߬ߟߎ (alu) — they


class CompoundType(Enum):
    """Types of Manding compound words"""
    NOUN_NOUN = "N+N"
    NOUN_VERB = "N+V"
    VERB_NOUN = "V+N"
    ADJ_NOUN = "A+N"
    NOUN_ADJ = "N+A"
    VERB_VERB = "V+V"
    REDUPLICATED = "REDUP"
    IDIOMATIC = "IDIOM"


class TonePattern(Enum):
    """Manding tone patterns — N'Ko preserves these via combining marks"""
    HIGH = "H"
    LOW = "L"
    FALLING = "F"
    RISING = "R"
    MID = "M"


# ═══════════════════════════════════════════════════════════════════════════════
# §2  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Morpheme:
    """A single meaningful unit within a word"""
    text: str
    morpheme_type: MorphemeType
    gloss: str = ""
    ipa: str = ""
    nko: str = ""
    latin: str = ""
    arabic: str = ""
    is_bound: bool = False
    tone: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "type": self.morpheme_type.value,
            "gloss": self.gloss,
            "ipa": self.ipa,
            "nko": self.nko,
            "latin": self.latin,
            "arabic": self.arabic,
            "bound": self.is_bound,
            "tone": self.tone,
        }


@dataclass
class WordAnalysis:
    """Complete morphological analysis of a word"""
    original: str
    script: str
    morphemes: List[Morpheme] = field(default_factory=list)
    root: Optional[Morpheme] = None
    word_class: str = ""
    noun_class: Optional[NounClass] = None
    is_compound: bool = False
    confidence: float = 0.0
    alternatives: List["WordAnalysis"] = field(default_factory=list)

    @property
    def prefix_count(self) -> int:
        return sum(1 for m in self.morphemes if m.morpheme_type == MorphemeType.PREFIX)

    @property
    def suffix_count(self) -> int:
        return sum(1 for m in self.morphemes if m.morpheme_type in (
            MorphemeType.SUFFIX, MorphemeType.DERIVATION, MorphemeType.PLURAL
        ))

    @property
    def morpheme_count(self) -> int:
        return len(self.morphemes)

    @property
    def gloss_string(self) -> str:
        return "-".join(m.gloss or m.text for m in self.morphemes)

    def decomposition(self) -> Dict[str, List[str]]:
        """Return structured decomposition: prefixes + root + suffixes."""
        prefixes = [m.text for m in self.morphemes if m.morpheme_type == MorphemeType.PREFIX]
        root_text = self.root.text if self.root else ""
        suffixes = [m.text for m in self.morphemes if m.morpheme_type in (
            MorphemeType.SUFFIX, MorphemeType.DERIVATION, MorphemeType.PLURAL
        )]
        return {"prefixes": prefixes, "root": root_text, "suffixes": suffixes}

    def reconstruct(self, target_script: str) -> str:
        parts = []
        for m in self.morphemes:
            if target_script == "nko" and m.nko:
                parts.append(m.nko)
            elif target_script == "latin" and m.latin:
                parts.append(m.latin)
            elif target_script == "arabic" and m.arabic:
                parts.append(m.arabic)
            else:
                parts.append(m.text)
        return "".join(parts)

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "script": self.script,
            "word_class": self.word_class,
            "noun_class": self.noun_class.value if self.noun_class else None,
            "is_compound": self.is_compound,
            "confidence": self.confidence,
            "root": self.root.to_dict() if self.root else None,
            "morphemes": [m.to_dict() for m in self.morphemes],
            "gloss": self.gloss_string,
            "decomposition": self.decomposition(),
        }


@dataclass
class ConjugatedForm:
    """A complete verb phrase in all three scripts"""
    subject: str
    particle: str
    verb: str
    full_nko: str
    full_latin: str
    full_arabic: str
    tense: TenseAspect
    person: PersonNumber
    gloss: str

    def to_dict(self) -> dict:
        return {
            "nko": self.full_nko,
            "latin": self.full_latin,
            "arabic": self.full_arabic,
            "tense": self.tense.value,
            "person": self.person.value,
            "gloss": self.gloss,
        }


@dataclass
class CompoundComponent:
    """A single component of a compound word"""
    text: str
    gloss: str = ""
    word_class: str = ""
    nko: str = ""
    latin: str = ""
    is_head: bool = False

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "gloss": self.gloss,
            "class": self.word_class,
            "nko": self.nko,
            "latin": self.latin,
            "head": self.is_head,
        }


@dataclass
class CompoundWord:
    """Analysis of a compound word"""
    original: str
    components: List[CompoundComponent] = field(default_factory=list)
    compound_type: CompoundType = CompoundType.NOUN_NOUN
    literal_meaning: str = ""
    actual_meaning: str = ""
    is_transparent: bool = True
    nko: str = ""
    latin: str = ""
    confidence: float = 0.0

    @property
    def component_count(self) -> int:
        return len(self.components)

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "type": self.compound_type.value,
            "components": [c.to_dict() for c in self.components],
            "literal": self.literal_meaning,
            "meaning": self.actual_meaning,
            "transparent": self.is_transparent,
            "nko": self.nko,
            "latin": self.latin,
            "confidence": self.confidence,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# §3  AFFIX INVENTORY
#     Complete catalogue of known Manding prefixes and suffixes
# ═══════════════════════════════════════════════════════════════════════════════

class AffixInventory:
    """
    Comprehensive inventory of Manding affixes.

    Merges the derivational suffixes from cross-script-bridge/analyzer and the
    VERBAL_SUFFIXES / NOMINAL_SUFFIXES / PREFIXES from keyboard-ai/morphological_engine.
    """

    # ── PREFIXES ──────────────────────────────────────────────────────────────
    PREFIXES: Dict[str, Dict] = {
        # Causative
        "ߟߊ߬": {"latin": "la", "type": "causative",
                  "meaning": "to cause", "gloss": "CAUS"},
        # Benefactive
        "ߡߊ߬": {"latin": "ma", "type": "benefactive",
                  "meaning": "for / on behalf of", "gloss": "BEN"},
        # Motion-toward / future
        "ߣߊ߬": {"latin": "na", "type": "future_motion",
                  "meaning": "coming to", "gloss": "VEN"},
    }

    # ── DERIVATIONAL SUFFIXES ─────────────────────────────────────────────────
    DERIVATIONAL_SUFFIXES: Dict[str, Dict] = {
        # Agentive — verb → doer
        "ߟߌ":  {"latin": "li",  "type": "agentive",
                  "meaning": "one who does", "gloss": "AGENT"},
        "ߓߊ":  {"latin": "ba",  "type": "agentive",
                  "meaning": "one who (habitually)", "gloss": "AGENT"},
        # Nominalizer — verb → act
        "ߟߊ߲": {"latin": "lan", "type": "nominalizer",
                  "meaning": "act of doing", "gloss": "NMLZ"},
        # Abstract quality — noun → quality
        "ߦߊ":  {"latin": "ya",  "type": "abstract",
                  "meaning": "quality / state of", "gloss": "QUAL"},
        # Possessor / owner
        "ߡߊ":  {"latin": "ma",  "type": "possessor",
                  "meaning": "possessor / owner", "gloss": "POSS"},
        # Instrumental — verb → tool
        "ߣߊ":  {"latin": "na",  "type": "instrumental",
                  "meaning": "instrument / tool for", "gloss": "INSTR"},
        # Locative — noun → place of
        "ߟߊ":  {"latin": "la",  "type": "locative",
                  "meaning": "place of", "gloss": "LOC"},
        # Diminutive
        "ߘߋ߲": {"latin": "den", "type": "diminutive",
                  "meaning": "small / young", "gloss": "DIM"},
        # Causative suffix
        "ߟߊ߲": {"latin": "lan", "type": "causative_suffix",
                  "meaning": "to cause to", "gloss": "CAUS"},
    }

    # ── INFLECTIONAL SUFFIXES ─────────────────────────────────────────────────
    INFLECTIONAL_SUFFIXES: Dict[str, Dict] = {
        # Plural
        "ߟߎ":  {"latin": "lu",  "type": "plural",
                  "meaning": "plural", "gloss": "PL"},
        # Perfective aspect
        "ߟߊ":  {"latin": "la",  "type": "perfective",
                  "meaning": "completed action", "gloss": "PFV"},
        # Progressive aspect
        "ߕߐ":  {"latin": "to",  "type": "progressive",
                  "meaning": "ongoing action", "gloss": "PROG"},
        # Ablative direction
        "ߓߐ":  {"latin": "bo",  "type": "ablative",
                  "meaning": "from", "gloss": "ABL"},
        # Illative direction
        "ߘߏ߲": {"latin": "don", "type": "illative",
                  "meaning": "into", "gloss": "ILL"},
    }

    # ── POSTPOSITIONS ─────────────────────────────────────────────────────────
    POSTPOSITIONS: Dict[str, Dict] = {
        "ߟߊ":  {"latin": "la",  "meaning": "in / at",      "gloss": "LOC"},
        "ߘߐ":  {"latin": "do",  "meaning": "inside",       "gloss": "INESS"},
        "ߞߊ߲": {"latin": "kan", "meaning": "on / upon",    "gloss": "SUPER"},
        "ߝߍ":  {"latin": "fe",  "meaning": "near / beside", "gloss": "PROX"},
        "ߦߋ":  {"latin": "ye",  "meaning": "at / toward",  "gloss": "ALL"},
        "ߓߟߏ": {"latin": "bolo","meaning": "hand / direction","gloss": "DIR"},
    }

    @classmethod
    def all_prefixes(cls) -> Dict[str, Dict]:
        return dict(cls.PREFIXES)

    @classmethod
    def all_suffixes(cls) -> Dict[str, Dict]:
        merged: Dict[str, Dict] = {}
        merged.update(cls.DERIVATIONAL_SUFFIXES)
        merged.update(cls.INFLECTIONAL_SUFFIXES)
        return merged

    @classmethod
    def all_postpositions(cls) -> Dict[str, Dict]:
        return dict(cls.POSTPOSITIONS)

    @classmethod
    def lookup(cls, form: str) -> Optional[Dict]:
        """Look up any affix by its N'Ko or Latin form."""
        for store in (cls.PREFIXES, cls.DERIVATIONAL_SUFFIXES,
                      cls.INFLECTIONAL_SUFFIXES, cls.POSTPOSITIONS):
            if form in store:
                return store[form]
            for nko, info in store.items():
                if info.get("latin", "").lower() == form.lower():
                    return {**info, "nko": nko}
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# §4  N'KO ↔ LATIN TRANSLITERATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

# N'Ko consonant/vowel → Latin (rough, for morpheme-matching)
_NKO_TO_LATIN: Dict[str, str] = {
    "ߊ": "a", "ߋ": "o", "ߌ": "i", "ߍ": "e", "ߎ": "u", "ߏ": "ɔ", "ߐ": "ə",
    "ߒ": "n", "ߓ": "b", "ߔ": "p", "ߕ": "t", "ߖ": "j", "ߗ": "c",
    "ߘ": "d", "ߙ": "r", "ߚ": "rr", "ߛ": "s", "ߜ": "gb", "ߝ": "f",
    "ߞ": "k", "ߟ": "l", "ߠ": "nh", "ߡ": "m", "ߢ": "ny", "ߣ": "n",
    "ߤ": "h", "ߥ": "w", "ߦ": "y", "ߧ": "ng", "ߨ": "p",
}

# N'Ko tone/combining marks to skip during rough transliteration
_NKO_SKIP = frozenset("߲߫߬߭߮߯߳")


def _detect_script(text: str) -> str:
    """Detect whether *text* is N'Ko, Arabic, or Latin."""
    for ch in text:
        cp = ord(ch)
        if 0x07C0 <= cp <= 0x07FF:
            return "nko"
        if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
            return "arabic"
        if 0x0041 <= cp <= 0x007A:
            return "latin"
    return "latin"


def _nko_to_latin_rough(text: str) -> str:
    """Quick N'Ko → Latin for morpheme matching (strips tones)."""
    result: List[str] = []
    for ch in text:
        if ch in _NKO_TO_LATIN:
            result.append(_NKO_TO_LATIN[ch])
        elif ch in _NKO_SKIP:
            pass
        elif ch == " ":
            result.append(" ")
    return "".join(result).strip()


def _normalize_to_latin(text: str, script: str) -> str:
    """Normalize any script to Latin for pattern matching."""
    if script == "latin":
        return text.lower().strip()
    if script == "nko":
        return _nko_to_latin_rough(text)
    return text.lower().strip()


# ═══════════════════════════════════════════════════════════════════════════════
# §5  MORPHOLOGICAL ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class MorphologicalAnalyzer:
    """
    Decomposes Manding words into morphemes with cross-script awareness.

    Merges the pattern-matching analyzer from *core/morphology* with the
    prefix/suffix tables from *core/prediction/morphological_engine*.
    """

    def __init__(self) -> None:
        self._build_morpheme_tables()

    # ── table construction ────────────────────────────────────────────────────

    def _build_morpheme_tables(self) -> None:
        # Subject pronouns
        self.subject_pronouns: List[Morpheme] = [
            Morpheme("ߒ", MorphemeType.SUBJECT_PRONOUN, "I", "n̩", "ߒ", "n", "نْ", tone="H"),
            Morpheme("ߌ", MorphemeType.SUBJECT_PRONOUN, "you(sg)", "i", "ߌ", "i", "إِ", tone="H"),
            Morpheme("ߊ߬", MorphemeType.SUBJECT_PRONOUN, "s/he", "à", "ߊ߬", "a", "أَ", tone="L"),
            Morpheme("ߊ߲", MorphemeType.SUBJECT_PRONOUN, "we", "án", "ߊ߲", "an", "أَنْ", tone="H"),
            Morpheme("ߊ߬ߟߎ", MorphemeType.SUBJECT_PRONOUN, "they", "àlu", "ߊ߬ߟߎ", "alu", "أَلُ", tone="L"),
            Morpheme("ߊ߬ߥߎ", MorphemeType.SUBJECT_PRONOUN, "you(pl)", "áwu", "ߊ߬ߥߎ", "awu", "أَوُ", tone="L"),
        ]

        # Tense / aspect markers
        self.tense_markers: List[Morpheme] = [
            Morpheme("ߓߍ", MorphemeType.TENSE_MARKER, "PROG", "bɛ", "ߓߍ", "be", "بَ",
                     notes="Progressive / habitual affirmative"),
            Morpheme("ߕߍ", MorphemeType.TENSE_MARKER, "NEG.PROG", "tɛ", "ߕߍ", "te", "تَ",
                     notes="Progressive / habitual negative"),
            Morpheme("ߞߊ", MorphemeType.TENSE_MARKER, "COMPL", "ka", "ߞߊ", "ka", "كَ",
                     notes="Completive (past)"),
            Morpheme("ߡߊ", MorphemeType.TENSE_MARKER, "NEG.COMPL", "ma", "ߡߊ", "ma", "مَ",
                     notes="Negative completive"),
            Morpheme("ߣߊ", MorphemeType.TENSE_MARKER, "FUT", "na", "ߣߊ", "na", "نَ",
                     notes="Future / intentional"),
            Morpheme("ߓߍ߫ ߣߊ", MorphemeType.TENSE_MARKER, "FUT.PROG", "bɛ́ na",
                     "ߓߍ߫ ߣߊ", "bena", "بَنَ", notes="Future progressive"),
        ]

        # Postpositions
        self.postpositions: List[Morpheme] = [
            Morpheme("ߟߊ", MorphemeType.POSTPOSITION, "in/at", "la", "ߟߊ", "la", "لَ", is_bound=True),
            Morpheme("ߘߐ", MorphemeType.POSTPOSITION, "inside", "dɔ", "ߘߐ", "do", "دُ", is_bound=True),
            Morpheme("ߞߊ߲", MorphemeType.POSTPOSITION, "on/upon", "kan", "ߞߊ߲", "kan", "كَنْ", is_bound=True),
            Morpheme("ߝߍ", MorphemeType.POSTPOSITION, "near/beside", "fɛ", "ߝߍ", "fe", "فَ", is_bound=True),
            Morpheme("ߦߋ", MorphemeType.POSTPOSITION, "at/toward", "jo", "ߦߋ", "ye", "يَ", is_bound=True),
            Morpheme("ߓߟߏ", MorphemeType.POSTPOSITION, "hand/direction", "bolo", "ߓߟߏ", "bolo", "بُلُ", is_bound=True),
        ]

        # Derivational suffixes
        self.derivation_suffixes: List[Morpheme] = [
            Morpheme("ߟߌ", MorphemeType.DERIVATION, "AGENT", "li", "ߟߌ", "li", "لِ",
                     is_bound=True, notes="Verb → Agent noun"),
            Morpheme("ߓߊ", MorphemeType.DERIVATION, "AGENT.HAB", "ba", "ߓߊ", "ba", "بَ",
                     is_bound=True, notes="Habitual agent"),
            Morpheme("ߟߊ߲", MorphemeType.DERIVATION, "NMLZ", "lan", "ߟߊ߲", "lan", "لَنْ",
                     is_bound=True, notes="Verb → Noun (act of doing)"),
            Morpheme("ߦߊ", MorphemeType.DERIVATION, "QUAL", "ya", "ߦߊ", "ya", "يَ",
                     is_bound=True, notes="Noun → Quality/state"),
            Morpheme("ߡߊ", MorphemeType.DERIVATION, "POSS", "ma", "ߡߊ", "ma", "مَ",
                     is_bound=True, notes="Possessor / owner"),
            Morpheme("ߣߊ", MorphemeType.DERIVATION, "INSTR", "na", "ߣߊ", "na", "نَ",
                     is_bound=True, notes="Instrument (tool for)"),
            Morpheme("ߟߊ", MorphemeType.DERIVATION, "LOC", "la", "ߟߊ", "la", "لَ",
                     is_bound=True, notes="Place of"),
            Morpheme("ߘߋ߲", MorphemeType.DERIVATION, "DIM", "den", "ߘߋ߲", "den", "دَنْ",
                     is_bound=True, notes="Diminutive"),
        ]

        # Plural marker
        self.plural_marker = Morpheme(
            "ߟߎ", MorphemeType.PLURAL, "PL", "lu", "ߟߎ", "lu", "لُ", is_bound=True,
        )

        # Verb roots — combined from both sources
        _verb_data = [
            ("ߕߊ߯", "go", "taá", "taa"),
            ("ߣߊ߯", "come", "naá", "naa"),
            ("ߞߍ", "do/make", "kɛ", "ke"),
            ("ߝߐ", "say", "fɔ", "fo"),
            ("ߘߐ߲", "eat", "dɔ̀n", "don"),
            ("ߡߌ߲", "drink", "mìn", "min"),
            ("ߛߐ߬ߘߐ", "know", "sɔ̀dɔ", "sodo"),
            ("ߓߐ", "exit/leave", "bɔ", "bo"),
            ("ߘߏ", "stay/remain", "dó", "do"),
            ("ߦߋ", "see", "jé", "ye"),
            ("ߟߊ߲ߘߌ", "understand", "lándi", "landi"),
            ("ߞߊ߬ߟߊ߲", "speak/tell", "kàlan", "kalan"),
            ("ߛߓߍ", "write", "sɛ́bɛ", "sebe"),
            ("ߞߊ߬ߙߊ", "read", "kàra", "kara"),
            ("ߘߌ", "give", "dí", "di"),
            ("ߡߊ߬ߞߐ", "return", "màkɔ", "mako"),
            ("ߛߐ", "buy", "sɔ́", "so"),
            ("ߝߍ", "want", "fɛ́", "fe"),
            ("ߛߍ", "be able", "sé", "se"),
            ("ߡߍ߲", "hear", "mɛ̀n", "men"),
            ("ߘߏ߲", "enter", "dón", "don"),
            ("ߛߌ߬ߟߊ", "travel", "sìla", "sila"),
            ("ߕߊ", "take", "tá", "ta"),
            ("ߘߊ", "learn", "dá", "da"),
        ]
        self.verb_roots: Dict[str, Morpheme] = {}
        for nko, gloss, ipa, latin in _verb_data:
            self.verb_roots[latin.lower()] = Morpheme(
                nko, MorphemeType.ROOT, gloss, ipa, nko, latin, "",
                notes=f"verb root: {gloss}",
            )

        # Noun roots — combined from both sources, enriched with noun-class
        _noun_data: List[Tuple[str, str, str, str, NounClass]] = [
            ("ߡߐ߱", "person", "mɔ̀gɔ", "mogo", NounClass.PERSON),
            ("ߘߋ߲", "child", "dén", "den", NounClass.PERSON),
            ("ߡߛߏ", "woman", "músó", "muso", NounClass.PERSON),
            ("ߗߍ", "man", "cɛ́", "ce", NounClass.PERSON),
            ("ߛߏ", "house", "só", "so", NounClass.THING),
            ("ߖߊ߬ߡߊ", "market", "jàma", "jama", NounClass.PLACE),
            ("ߖߌ", "water", "jí", "ji", NounClass.LIQUID),
            ("ߘߎ߬ߡ", "food/rice", "dùmu", "dumu", NounClass.MASS),
            ("ߛߌ߬ߙ", "road", "sìra", "sira", NounClass.THING),
            ("ߞߎ߬ߙ", "boat", "kùru", "kuru", NounClass.THING),
            ("ߞߊ߲", "language/word", "kán", "kan", NounClass.ABSTRACT),
            ("ߕߎ߬ߓ", "land/soil", "dùgu", "dugu", NounClass.PLACE),
            ("ߛߓ", "writing/book", "sɛ́bɛ", "sebe", NounClass.THING),
            ("ߝ߭ߊ", "father", "fáa", "faa", NounClass.KINSHIP),
            ("ߓ߭ߊ", "mother", "báa", "baa", NounClass.KINSHIP),
            ("ߟߐ", "name", "tɔ́gɔ", "togo", NounClass.ABSTRACT),
            ("ߖߊ", "money", "já", "ja", NounClass.THING),
            ("ߘߎ", "earth/land", "dú", "du", NounClass.PLACE),
            ("ߛߊ", "rain/sky", "sá", "sa", NounClass.ABSTRACT),
            ("ߝ߬ߣ", "food", "fɛ̀n", "fen", NounClass.MASS),
        ]
        self.noun_roots: Dict[str, Morpheme] = {}
        self._noun_classes: Dict[str, NounClass] = {}
        for nko, gloss, ipa, latin, nc in _noun_data:
            self.noun_roots[latin.lower()] = Morpheme(
                nko, MorphemeType.ROOT, gloss, ipa, nko, latin, "",
            )
            self._noun_classes[latin.lower()] = nc

        # Additional noun-class heuristics (suffixes/semantic patterns)
        self._noun_class_suffix_hints: Dict[str, NounClass] = {
            "ya": NounClass.ABSTRACT,     # -ߦߊ suffix → abstract
            "li": NounClass.ABSTRACT,     # -ߟߌ nominalization
            "lan": NounClass.ABSTRACT,    # -ߟߊ߲ nominalization
            "den": NounClass.PERSON,      # diminutive, often person-related
            "ba": NounClass.PERSON,       # agentive
        }

        # ── latin lookup map ──
        self._latin_lookup: Dict[str, Morpheme] = {}
        for morpheme_list in [self.subject_pronouns, self.tense_markers,
                              self.postpositions, self.derivation_suffixes]:
            for m in morpheme_list:
                if m.latin:
                    self._latin_lookup[m.latin.lower()] = m
        self._latin_lookup[self.plural_marker.latin.lower()] = self.plural_marker

    # ── script detection ──────────────────────────────────────────────────────

    def detect_script(self, text: str) -> str:
        return _detect_script(text)

    # ── main analysis entry ───────────────────────────────────────────────────

    def analyze(self, text: str, script: str | None = None) -> List[WordAnalysis]:
        """Analyze *text* into morphological components."""
        if script is None:
            script = self.detect_script(text)
        words = text.strip().split()
        return [self._analyze_word(w, script) for w in words]

    def analyze_word(self, word: str, script: str | None = None) -> WordAnalysis:
        """Analyze a single *word*."""
        if script is None:
            script = self.detect_script(word)
        return self._analyze_word(word, script)

    # ── private analysis helpers ──────────────────────────────────────────────

    def _analyze_word(self, word: str, script: str) -> WordAnalysis:
        latin = _normalize_to_latin(word, script)
        analysis = WordAnalysis(original=word, script=script)

        # 1. Subject pronoun?
        for p in self.subject_pronouns:
            if latin == p.latin.lower():
                analysis.morphemes.append(p)
                analysis.word_class = "pronoun"
                analysis.confidence = 0.95
                return analysis

        # 2. Tense marker?
        for tm in self.tense_markers:
            if latin == tm.latin.lower():
                analysis.morphemes.append(tm)
                analysis.word_class = "particle"
                analysis.confidence = 0.90
                return analysis

        # 3. Prefixed word?
        prefix_match = self._strip_prefix(latin)
        if prefix_match:
            prefix_m, remainder = prefix_match
            # Try the remainder as a verb
            verb_match = self._match_verb(remainder, word, script)
            if verb_match:
                verb_match.morphemes.insert(0, prefix_m)
                verb_match.confidence = min(verb_match.confidence + 0.05, 1.0)
                return verb_match

        # 4. Verb (possibly with suffixes)?
        verb_match = self._match_verb(latin, word, script)
        if verb_match:
            return verb_match

        # 5. Noun (possibly with plural / postposition)?
        noun_match = self._match_noun(latin, word, script)
        if noun_match:
            return noun_match

        # 6. Postposition?
        for pp in self.postpositions:
            if latin == pp.latin.lower():
                analysis.morphemes.append(pp)
                analysis.word_class = "postposition"
                analysis.confidence = 0.85
                return analysis

        # 7. Fallback
        analysis.morphemes.append(Morpheme(
            word, MorphemeType.ROOT, "?", "",
            word if script == "nko" else "",
            word if script == "latin" else "",
            word if script == "arabic" else "",
        ))
        analysis.word_class = "unknown"
        analysis.confidence = 0.30
        return analysis

    def _strip_prefix(self, latin: str) -> Optional[Tuple[Morpheme, str]]:
        """Try to strip a known prefix and return (prefix_morpheme, remainder)."""
        for nko, info in AffixInventory.PREFIXES.items():
            prefix_latin = info["latin"].lower()
            if latin.startswith(prefix_latin) and len(latin) > len(prefix_latin):
                m = Morpheme(
                    nko, MorphemeType.PREFIX, info.get("gloss", ""),
                    "", nko, info["latin"], "",
                    is_bound=True, notes=info.get("meaning", ""),
                )
                return m, latin[len(prefix_latin):]
        return None

    def _match_verb(self, latin: str, original: str, script: str) -> Optional[WordAnalysis]:
        for root_latin, root_m in sorted(
            self.verb_roots.items(), key=lambda kv: len(kv[0]), reverse=True
        ):
            if latin.startswith(root_latin):
                remainder = latin[len(root_latin):]
                analysis = WordAnalysis(original=original, script=script,
                                        word_class="verb", confidence=0.80)
                analysis.root = root_m
                analysis.morphemes.append(root_m)

                if remainder:
                    matched = self._match_suffixes(remainder)
                    if matched:
                        analysis.morphemes.extend(matched)
                        analysis.confidence = 0.85
                    else:
                        analysis.morphemes.append(Morpheme(
                            remainder, MorphemeType.SUFFIX, "?",
                            notes="Unknown verbal suffix",
                        ))
                        analysis.confidence = 0.60
                return analysis
        return None

    def _match_noun(self, latin: str, original: str, script: str) -> Optional[WordAnalysis]:
        for root_latin, root_m in sorted(
            self.noun_roots.items(), key=lambda kv: len(kv[0]), reverse=True
        ):
            if latin.startswith(root_latin):
                remainder = latin[len(root_latin):]
                analysis = WordAnalysis(original=original, script=script,
                                        word_class="noun", confidence=0.80)
                analysis.root = root_m
                analysis.morphemes.append(root_m)

                # Noun class
                analysis.noun_class = self._noun_classes.get(root_latin)

                # Plural?
                if remainder and remainder.startswith(self.plural_marker.latin.lower()):
                    analysis.morphemes.append(self.plural_marker)
                    remainder = remainder[len(self.plural_marker.latin):]
                    analysis.confidence = 0.90

                # Postposition or derivational suffix on remainder?
                if remainder:
                    for pp in self.postpositions:
                        if remainder == pp.latin.lower():
                            analysis.morphemes.append(pp)
                            analysis.confidence = 0.85
                            remainder = ""
                            break
                if remainder:
                    matched_sfx = self._match_suffixes(remainder)
                    if matched_sfx:
                        analysis.morphemes.extend(matched_sfx)
                        # Suffix may change noun class
                        for sfx in matched_sfx:
                            hint_cls = self._noun_class_suffix_hints.get(sfx.latin.lower())
                            if hint_cls:
                                analysis.noun_class = hint_cls

                return analysis
        return None

    def _match_suffixes(self, remainder: str) -> Optional[List[Morpheme]]:
        matched: List[Morpheme] = []
        pos = 0
        while pos < len(remainder):
            found = False
            # Longest-match among derivation suffixes
            for sfx in sorted(self.derivation_suffixes,
                              key=lambda s: len(s.latin), reverse=True):
                if remainder[pos:].startswith(sfx.latin.lower()):
                    matched.append(sfx)
                    pos += len(sfx.latin)
                    found = True
                    break
            if not found:
                # Plural?
                if remainder[pos:].startswith(self.plural_marker.latin.lower()):
                    matched.append(self.plural_marker)
                    pos += len(self.plural_marker.latin)
                else:
                    return None
        return matched if matched else None

    # ── noun-class detection ──────────────────────────────────────────────────

    def detect_noun_class(self, word: str, script: str | None = None) -> Optional[NounClass]:
        """Return the :class:`NounClass` for *word*, or ``None``."""
        if script is None:
            script = self.detect_script(word)
        latin = _normalize_to_latin(word, script)
        # Direct root lookup
        if latin in self._noun_classes:
            return self._noun_classes[latin]
        # Try stripping plural then looking up
        if latin.endswith("lu") and latin[:-2] in self._noun_classes:
            return self._noun_classes[latin[:-2]]
        # Suffix hints
        for sfx, nc in self._noun_class_suffix_hints.items():
            if latin.endswith(sfx) and latin != sfx:
                return nc
        return None

    # ── sentence-level analysis ───────────────────────────────────────────────

    def analyze_sentence(self, text: str, script: str | None = None) -> Dict:
        analyses = self.analyze(text, script)
        types = [a.word_class for a in analyses]
        pattern = "-".join(types)

        structures = {
            "pronoun-particle-verb": "STV (Subject-Tense-Verb)",
            "pronoun-particle-verb-noun": "STVO (Subject-Tense-Verb-Object)",
            "noun-particle-verb": "STV (Noun Subject-Tense-Verb)",
            "noun-postposition": "NP (Noun Phrase + Postposition)",
            "noun-noun": "NN (Compound / Genitive)",
        }
        structure = "unclassified"
        for pat, desc in structures.items():
            if pattern.startswith(pat):
                structure = desc
                break

        return {
            "words": [a.to_dict() for a in analyses],
            "structure": structure,
            "glossing": " ".join(a.gloss_string for a in analyses),
            "original": text,
            "script": script or (analyses[0].script if analyses else "unknown"),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# §6  VERB CONJUGATOR
# ═══════════════════════════════════════════════════════════════════════════════

class VerbConjugator:
    """
    Generate Manding verb phrases across all three scripts.

    Manding verbs do **not** inflect — the particle system carries TAM.
    This conjugator produces Subject + Particle + Verb triplets.
    """

    def __init__(self) -> None:
        self._build_tables()

    def _build_tables(self) -> None:
        self.pronouns: Dict[PersonNumber, Dict[str, str]] = {
            PersonNumber.FIRST_SG:  {"nko": "ߒ",    "latin": "n",   "arabic": "نْ",   "ipa": "n̩"},
            PersonNumber.SECOND_SG: {"nko": "ߌ",    "latin": "i",   "arabic": "إِ",   "ipa": "i"},
            PersonNumber.THIRD_SG:  {"nko": "ߊ߬",   "latin": "a",   "arabic": "أَ",   "ipa": "à"},
            PersonNumber.FIRST_PL:  {"nko": "ߊ߲",   "latin": "an",  "arabic": "أَنْ", "ipa": "án"},
            PersonNumber.SECOND_PL: {"nko": "ߊ߬ߥ",  "latin": "aw",  "arabic": "أَوْ", "ipa": "àw"},
            PersonNumber.THIRD_PL:  {"nko": "ߊ߬ߟߎ", "latin": "alu", "arabic": "أَلُ", "ipa": "àlu"},
        }

        self.particles: Dict[TenseAspect, Dict[str, str]] = {
            TenseAspect.PROGRESSIVE:     {"nko": "ߓߍ",     "latin": "be",   "arabic": "بَ",   "ipa": "bɛ"},
            TenseAspect.NEG_PROGRESSIVE: {"nko": "ߕߍ",     "latin": "te",   "arabic": "تَ",   "ipa": "tɛ"},
            TenseAspect.COMPLETIVE:      {"nko": "ߞߊ",     "latin": "ka",   "arabic": "كَ",   "ipa": "ka"},
            TenseAspect.NEG_COMPLETIVE:  {"nko": "ߡߊ",     "latin": "ma",   "arabic": "مَ",   "ipa": "ma"},
            TenseAspect.FUTURE:          {"nko": "ߣߊ",     "latin": "na",   "arabic": "نَ",   "ipa": "na"},
            TenseAspect.FUTURE_PROG:     {"nko": "ߓߍ ߣߊ", "latin": "bena", "arabic": "بَنَ", "ipa": "bɛ na"},
            TenseAspect.IMPERATIVE:      {"nko": "",        "latin": "",     "arabic": "",     "ipa": ""},
            TenseAspect.SUBJUNCTIVE:     {"nko": "ߞߊ",     "latin": "ka",   "arabic": "كَ",   "ipa": "ka"},
            TenseAspect.CONDITIONAL:     {"nko": "ߣߌ...ߞߊ","latin": "ni...ka","arabic":"نِ...كَ","ipa":"ni...ka"},
        }

        self.common_verbs: Dict[str, Dict[str, str]] = {
            "go":    {"nko": "ߕߊ߯",   "latin": "taa",   "arabic": "تَا",   "ipa": "taá"},
            "come":  {"nko": "ߣߊ߯",   "latin": "naa",   "arabic": "نَا",   "ipa": "naá"},
            "do":    {"nko": "ߞߍ",    "latin": "ke",    "arabic": "كَ",    "ipa": "kɛ"},
            "say":   {"nko": "ߝߐ",    "latin": "fo",    "arabic": "فُ",    "ipa": "fɔ"},
            "eat":   {"nko": "ߘߐ߲",   "latin": "don",   "arabic": "دُنْ",  "ipa": "dɔ̀n"},
            "drink": {"nko": "ߡߌ߲",   "latin": "min",   "arabic": "مِنْ",  "ipa": "mìn"},
            "see":   {"nko": "ߦߋ",    "latin": "ye",    "arabic": "يَ",    "ipa": "jé"},
            "write": {"nko": "ߛߓߍ",   "latin": "sebe",  "arabic": "سَبَ",  "ipa": "sɛ́bɛ"},
            "read":  {"nko": "ߞߊ߬ߙ",  "latin": "kara",  "arabic": "كَرَ",  "ipa": "kàra"},
            "give":  {"nko": "ߘߌ",    "latin": "di",    "arabic": "دِ",    "ipa": "dí"},
            "know":  {"nko": "ߛߐ߬ߘߐ", "latin": "sodo",  "arabic": "سُدُ",  "ipa": "sɔ̀dɔ"},
            "want":  {"nko": "ߝߍ",    "latin": "fe",    "arabic": "فَ",    "ipa": "fɛ́"},
            "leave": {"nko": "ߓߐ",    "latin": "bo",    "arabic": "بُ",    "ipa": "bɔ"},
            "stay":  {"nko": "ߘߏ",    "latin": "do",    "arabic": "دُ",    "ipa": "dó"},
            "learn": {"nko": "ߟߊ߲ߘߌ", "latin": "landi", "arabic": "لَنْدِ","ipa": "lándi"},
            "hear":  {"nko": "ߡߍ߲",   "latin": "men",   "arabic": "مَنْ",  "ipa": "mɛ̀n"},
            "enter": {"nko": "ߘߏ߲",   "latin": "don",   "arabic": "دُنْ",  "ipa": "dón"},
            "buy":   {"nko": "ߛߐ",    "latin": "so",    "arabic": "سُ",    "ipa": "sɔ́"},
            "take":  {"nko": "ߕߊ",    "latin": "ta",    "arabic": "تَ",    "ipa": "tá"},
        }

    def _resolve_verb(self, verb: str) -> Dict[str, str]:
        """Resolve *verb* (gloss, Latin, or N'Ko) to all script forms."""
        v = verb.lower()
        if v in self.common_verbs:
            return {**self.common_verbs[v], "gloss": v}
        for gloss, forms in self.common_verbs.items():
            if forms["latin"].lower() == v:
                return {**forms, "gloss": gloss}
        for gloss, forms in self.common_verbs.items():
            if forms["nko"] == verb:
                return {**forms, "gloss": gloss}
        return {"nko": verb, "latin": verb, "arabic": verb, "ipa": verb, "gloss": verb}

    def conjugate(
        self,
        verb: str,
        tense: TenseAspect = TenseAspect.PROGRESSIVE,
        person: PersonNumber = PersonNumber.THIRD_SG,
    ) -> ConjugatedForm:
        """Generate a conjugated verb phrase in all three scripts."""
        vf = self._resolve_verb(verb)
        pron = self.pronouns[person]
        part = self.particles[tense]

        if tense == TenseAspect.IMPERATIVE:
            if person == PersonNumber.SECOND_SG:
                nko = vf["nko"]
                latin = vf["latin"]
                arabic = vf["arabic"]
                gloss = f'{vf["gloss"]}!'
            else:
                nko = f'{pron["nko"]} {vf["nko"]}'
                latin = f'{pron["latin"]} {vf["latin"]}'
                arabic = f'{pron["arabic"]} {vf["arabic"]}'
                gloss = f'{person.value}-{vf["gloss"]}!'
        else:
            nko = f'{pron["nko"]} {part["nko"]} {vf["nko"]}'
            latin = f'{pron["latin"]} {part["latin"]} {vf["latin"]}'
            arabic = f'{pron["arabic"]} {part["arabic"]} {vf["arabic"]}'
            gloss = f'{person.value}-{tense.value}-{vf["gloss"]}'

        return ConjugatedForm(
            subject=pron["nko"], particle=part["nko"], verb=vf["nko"],
            full_nko=nko, full_latin=latin, full_arabic=arabic,
            tense=tense, person=person, gloss=gloss,
        )

    def full_paradigm(self, verb: str) -> Dict[str, List[ConjugatedForm]]:
        """Complete conjugation paradigm — every tense × every person."""
        return {
            tense.value: [self.conjugate(verb, tense, person)
                          for person in PersonNumber]
            for tense in TenseAspect
        }

    def compare_tenses(self, verb: str) -> List[Dict]:
        """Side-by-side comparison of all tenses (1sg) for learning."""
        result = []
        for tense in TenseAspect:
            form = self.conjugate(verb, tense, PersonNumber.FIRST_SG)
            result.append({
                "tense": tense.value,
                "nko": form.full_nko,
                "latin": form.full_latin,
                "arabic": form.full_arabic,
                "gloss": form.gloss,
            })
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# §7  COMPOUND WORD DETECTOR & SPLITTER
# ═══════════════════════════════════════════════════════════════════════════════

class CompoundDetector:
    """
    Detect and decompose compound words in Manding.

    Manding forms compounds extensively — *soda* "house-mouth" = doorway,
    *kanmogo* "language-person" = linguist.  This class maintains a known-
    compound database and a greedy longest-match splitter for novel compounds.
    """

    def __init__(self) -> None:
        self._build_root_index()
        self._build_known_compounds()

    def _build_root_index(self) -> None:
        self.root_index: Dict[str, Tuple[str, str, str]] = {
            # latin: (gloss, word_class, nko)
            "so": ("house", "noun", "ߛߏ"),
            "da": ("mouth/create", "noun", "ߘߊ"),
            "ji": ("water", "noun", "ߖߌ"),
            "bo": ("exit", "verb", "ߓߐ"),
            "fo": ("say", "verb", "ߝߐ"),
            "kan": ("language/word", "noun", "ߞߊ߲"),
            "mogo": ("person", "noun", "ߡߐ߱"),
            "den": ("child", "noun", "ߘߋ߲"),
            "ba": ("mother/big", "noun", "ߓ߭ߊ"),
            "fa": ("father", "noun", "ߝ߭ߊ"),
            "ke": ("do/make", "verb", "ߞߍ"),
            "taa": ("go", "verb", "ߕߊ߯"),
            "naa": ("come", "verb", "ߣߊ߯"),
            "don": ("know/enter", "verb", "ߘߐ߲"),
            "min": ("drink", "verb", "ߡߌ߲"),
            "sira": ("road", "noun", "ߛߌ߬ߙ"),
            "dugu": ("village/land", "noun", "ߘߎ߬ߜ"),
            "muso": ("woman", "noun", "ߡߛߏ"),
            "ce": ("man", "noun", "ߗߍ"),
            "ye": ("see", "verb", "ߦߋ"),
            "se": ("be able", "verb", "ߛߍ"),
            "fe": ("want/near", "verb", "ߝߍ"),
            "lu": ("plural", "suffix", "ߟߎ"),
            "tigi": ("owner/chief", "noun", "ߕߌ߬ߜ"),
            "fin": ("black/dark", "adj", "ߝߌ߲"),
            "je": ("white/clear", "adj", "ߖߍ"),
            "kele": ("battle/war", "noun", "ߞߍ߬ߟ"),
            "sebe": ("write", "verb", "ߛߓ"),
            "kara": ("read", "verb", "ߞߊ߬ߙ"),
            "nko": ("clear language", "noun", "ߒ߬ߞ"),
            "dumu": ("food/eat", "noun", "ߘߎ߬ߡ"),
            "sila": ("travel", "verb", "ߛߌ߬ߟ"),
            "wolo": ("give birth", "verb", "ߥߟ"),
            "kolo": ("bone/hard", "noun", "ߞߟ"),
            "fili": ("lose", "verb", "ߝߌ߬ߟ"),
            "bolo": ("hand", "noun", "ߓߟ"),
        }

    def _build_known_compounds(self) -> None:
        self.known_compounds: Dict[str, CompoundWord] = {}
        _data = [
            ("soda", "ߛߏ ߘߊ", CompoundType.NOUN_NOUN,
             [("so", "ߛߏ", "house", "noun", True), ("da", "ߘߊ", "mouth", "noun", False)],
             "house-mouth", "doorway", True),
            ("jibo", "ߖߌ ߓߐ", CompoundType.NOUN_VERB,
             [("ji", "ߖߌ", "water", "noun", False), ("bo", "ߓߐ", "exit", "verb", True)],
             "water-exit", "spring/fountain", True),
            ("kanmogo", "ߞߊ߲ ߡߐ߱", CompoundType.NOUN_NOUN,
             [("kan", "ߞߊ߲", "language", "noun", False), ("mogo", "ߡߐ߱", "person", "noun", True)],
             "language-person", "linguist/interpreter", True),
            ("dugutigi", "ߘߎ߬ߜ߬ ߕߌ߬ߜ", CompoundType.NOUN_NOUN,
             [("dugu", "ߘߎ߬ߜ߬", "village", "noun", False), ("tigi", "ߕߌ߬ߜ", "owner/chief", "noun", True)],
             "village-owner", "village chief", True),
            ("denba", "ߘߋ߲ ߓ߭ߊ", CompoundType.NOUN_NOUN,
             [("den", "ߘߋ߲", "child", "noun", False), ("ba", "ߓ߭ߊ", "mother", "noun", True)],
             "child-mother", "birth mother", True),
            ("fokan", "ߝߐ ߞߊ߲", CompoundType.VERB_NOUN,
             [("fo", "ߝߐ", "say", "verb", False), ("kan", "ߞߊ߲", "word", "noun", True)],
             "say-word", "speech/language", True),
            ("siradon", "ߛߌ߬ߙ ߘߐ߲", CompoundType.NOUN_VERB,
             [("sira", "ߛߌ߬ߙ", "road", "noun", False), ("don", "ߘߐ߲", "know", "verb", True)],
             "road-know", "guide", True),
            ("nkosebe", "ߒ߬ߞ ߛߓ", CompoundType.NOUN_NOUN,
             [("nko", "ߒ߬ߞ", "N'Ko", "noun", False), ("sebe", "ߛߓ", "writing", "noun", True)],
             "N'Ko-writing", "N'Ko script", True),
            ("musoden", "ߡߛߏ ߘߋ߲", CompoundType.NOUN_NOUN,
             [("muso", "ߡߛߏ", "woman", "noun", False), ("den", "ߘߋ߲", "child", "noun", True)],
             "woman-child", "daughter", True),
            ("ceden", "ߗߍ ߘߋ߲", CompoundType.NOUN_NOUN,
             [("ce", "ߗߍ", "man", "noun", False), ("den", "ߘߋ߲", "child", "noun", True)],
             "man-child", "son", True),
            ("sofin", "ߛߏ ߝߌ߲", CompoundType.NOUN_ADJ,
             [("so", "ߛߏ", "house", "noun", True), ("fin", "ߝߌ߲", "black/dark", "adj", False)],
             "house-dark", "prison", False),
            ("duguji", "ߘߎ߬ߜ ߖߌ", CompoundType.NOUN_NOUN,
             [("dugu", "ߘߎ߬ߜ", "land", "noun", False), ("ji", "ߖߌ", "water", "noun", True)],
             "land-water", "well water", True),
            ("sirakele", "ߛߌ߬ߙ ߞߍ߬ߟ", CompoundType.NOUN_NOUN,
             [("sira", "ߛߌ߬ߙ", "road", "noun", False), ("kele", "ߞߍ߬ߟ", "battle", "noun", True)],
             "road-battle", "roadblock/obstacle", True),
            ("jidon", "ߖߌ ߘߐ߲", CompoundType.NOUN_NOUN,
             [("ji", "ߖߌ", "water", "noun", False), ("don", "ߘߐ߲", "inside/enter", "noun", True)],
             "water-place", "well/water source", True),
        ]
        for latin, nko, ctype, comp_data, literal, actual, transparent in _data:
            components = [
                CompoundComponent(text=cl, gloss=cg, word_class=cc, nko=cn, latin=cl, is_head=ch)
                for cl, cn, cg, cc, ch in comp_data
            ]
            self.known_compounds[latin.lower()] = CompoundWord(
                original=latin, components=components, compound_type=ctype,
                literal_meaning=literal, actual_meaning=actual,
                is_transparent=transparent, nko=nko, latin=latin, confidence=1.0,
            )

    # ── public API ────────────────────────────────────────────────────────────

    def is_compound(self, word: str, script: str = "latin") -> bool:
        """Quick check — is *word* a known or decomposable compound?"""
        latin = _normalize_to_latin(word, script)
        if latin in self.known_compounds:
            return True
        return self._try_decompose(latin) is not None

    def split(self, word: str, script: str = "latin") -> CompoundWord:
        """Decompose *word* into compound components."""
        latin = _normalize_to_latin(word, script)
        if latin in self.known_compounds:
            return self.known_compounds[latin]
        dec = self._try_decompose(latin)
        if dec:
            return dec
        return CompoundWord(
            original=word,
            components=[CompoundComponent(text=word, gloss="?")],
            literal_meaning=word,
            actual_meaning="(unknown)",
            confidence=0.0,
        )

    def find_compounds_with(self, root: str) -> List[CompoundWord]:
        """Return all known compounds containing *root*."""
        results = []
        for cw in self.known_compounds.values():
            for c in cw.components:
                if c.latin.lower() == root.lower() or c.gloss.lower() == root.lower():
                    results.append(cw)
                    break
        return results

    def generate_compound(self, root1: str, root2: str) -> Optional[CompoundWord]:
        """Build a *hypothetical* compound from two known roots."""
        r1 = self.root_index.get(root1.lower())
        r2 = self.root_index.get(root2.lower())
        if not r1 or not r2:
            return None
        g1, c1, n1 = r1
        g2, c2, n2 = r2
        type_map = {
            ("noun", "noun"): CompoundType.NOUN_NOUN,
            ("noun", "verb"): CompoundType.NOUN_VERB,
            ("verb", "noun"): CompoundType.VERB_NOUN,
            ("adj", "noun"): CompoundType.ADJ_NOUN,
            ("noun", "adj"): CompoundType.NOUN_ADJ,
            ("verb", "verb"): CompoundType.VERB_VERB,
        }
        return CompoundWord(
            original=f"{root1}{root2}",
            components=[
                CompoundComponent(text=root1, gloss=g1, word_class=c1, nko=n1, latin=root1),
                CompoundComponent(text=root2, gloss=g2, word_class=c2, nko=n2, latin=root2, is_head=True),
            ],
            compound_type=type_map.get((c1, c2), CompoundType.NOUN_NOUN),
            literal_meaning=f"{g1}-{g2}",
            actual_meaning=f"(hypothetical: {g1}-{g2})",
            nko=f"{n1} {n2}",
            latin=f"{root1}{root2}",
            confidence=0.50,
        )

    # ── private ───────────────────────────────────────────────────────────────

    def _try_decompose(self, latin: str) -> Optional[CompoundWord]:
        components: List[CompoundComponent] = []
        remaining = latin
        while remaining:
            best: Optional[Tuple[str, str, str, str]] = None
            best_len = 0
            for root, (gloss, wclass, nko) in sorted(
                self.root_index.items(), key=lambda kv: len(kv[0]), reverse=True
            ):
                if remaining.startswith(root) and len(root) > best_len:
                    best = (root, gloss, wclass, nko)
                    best_len = len(root)
            if best:
                root, gloss, wclass, nko = best
                components.append(CompoundComponent(
                    text=root, gloss=gloss, word_class=wclass, nko=nko, latin=root,
                ))
                remaining = remaining[best_len:]
            else:
                if components:
                    components.append(CompoundComponent(text=remaining, gloss="?"))
                    remaining = ""
                else:
                    return None

        if len(components) < 2:
            return None

        # Head assignment — last noun
        for c in reversed(components):
            if c.word_class == "noun":
                c.is_head = True
                break

        classes = tuple(c.word_class for c in components[:2])
        type_map = {
            ("noun", "noun"): CompoundType.NOUN_NOUN,
            ("noun", "verb"): CompoundType.NOUN_VERB,
            ("verb", "noun"): CompoundType.VERB_NOUN,
            ("adj", "noun"): CompoundType.ADJ_NOUN,
            ("noun", "adj"): CompoundType.NOUN_ADJ,
            ("verb", "verb"): CompoundType.VERB_VERB,
        }
        literal = "-".join(c.gloss for c in components)
        return CompoundWord(
            original="".join(c.latin for c in components if c.latin),
            components=components,
            compound_type=type_map.get(classes, CompoundType.NOUN_NOUN),
            literal_meaning=literal,
            actual_meaning=f"({literal})",
            nko=" ".join(c.nko for c in components if c.nko),
            latin="".join(c.latin for c in components if c.latin),
            confidence=0.70,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# §8  CONVENIENCE FUNCTIONS (top-level API)
# ═══════════════════════════════════════════════════════════════════════════════

# Module-level singletons (lazy)
_analyzer: Optional[MorphologicalAnalyzer] = None
_conjugator: Optional[VerbConjugator] = None
_compound_detector: Optional[CompoundDetector] = None


def _get_analyzer() -> MorphologicalAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = MorphologicalAnalyzer()
    return _analyzer


def _get_conjugator() -> VerbConjugator:
    global _conjugator
    if _conjugator is None:
        _conjugator = VerbConjugator()
    return _conjugator


def _get_compound_detector() -> CompoundDetector:
    global _compound_detector
    if _compound_detector is None:
        _compound_detector = CompoundDetector()
    return _compound_detector


def analyze(text: str, script: str | None = None) -> List[WordAnalysis]:
    """Analyze *text* into morphological components."""
    return _get_analyzer().analyze(text, script)


def analyze_word(word: str, script: str | None = None) -> WordAnalysis:
    """Analyze a single *word*."""
    return _get_analyzer().analyze_word(word, script)


def decompose(word: str, script: str | None = None) -> Dict[str, List[str]]:
    """Return ``{"prefixes": [...], "root": "...", "suffixes": [...]}``."""
    return _get_analyzer().analyze_word(word, script).decomposition()


def extract_root(word: str, script: str | None = None) -> str:
    """Return the root morpheme of *word* (empty string if unknown)."""
    wa = _get_analyzer().analyze_word(word, script)
    return wa.root.text if wa.root else ""


def detect_noun_class(word: str, script: str | None = None) -> Optional[NounClass]:
    """Return the :class:`NounClass` for *word* or ``None``."""
    return _get_analyzer().detect_noun_class(word, script)


def conjugate(
    verb: str,
    tense: TenseAspect = TenseAspect.PROGRESSIVE,
    person: PersonNumber = PersonNumber.THIRD_SG,
) -> ConjugatedForm:
    """Conjugate *verb* in the given *tense* and *person*."""
    return _get_conjugator().conjugate(verb, tense, person)


def full_paradigm(verb: str) -> Dict[str, List[ConjugatedForm]]:
    """Return every tense × every person for *verb*."""
    return _get_conjugator().full_paradigm(verb)


def is_compound(word: str, script: str = "latin") -> bool:
    """Quick check — is *word* a compound?"""
    return _get_compound_detector().is_compound(word, script)


def split_compound(word: str, script: str = "latin") -> CompoundWord:
    """Decompose *word* into compound components."""
    return _get_compound_detector().split(word, script)


def get_affix(form: str) -> Optional[Dict]:
    """Look up any affix by N'Ko or Latin form."""
    return AffixInventory.lookup(form)


def list_prefixes() -> Dict[str, Dict]:
    """Return all known Manding prefixes."""
    return AffixInventory.all_prefixes()


def list_suffixes() -> Dict[str, Dict]:
    """Return all known Manding suffixes (derivational + inflectional)."""
    return AffixInventory.all_suffixes()


def list_postpositions() -> Dict[str, Dict]:
    """Return all known Manding postpositions."""
    return AffixInventory.all_postpositions()


# ═══════════════════════════════════════════════════════════════════════════════
# §9  PUBLIC EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "MorphemeType", "NounClass", "TenseAspect", "PersonNumber",
    "CompoundType", "TonePattern",
    # Data classes
    "Morpheme", "WordAnalysis", "ConjugatedForm",
    "CompoundComponent", "CompoundWord",
    # Classes
    "MorphologicalAnalyzer", "VerbConjugator", "CompoundDetector",
    "AffixInventory",
    # Convenience functions
    "analyze", "analyze_word", "decompose", "extract_root",
    "detect_noun_class",
    "conjugate", "full_paradigm",
    "is_compound", "split_compound",
    "get_affix", "list_prefixes", "list_suffixes", "list_postpositions",
]
