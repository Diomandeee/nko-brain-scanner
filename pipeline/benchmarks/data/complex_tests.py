"""
Complex Test Generator for N'Ko and Manding Language Benchmarks.

Generates advanced test cases that require deep language understanding:
- Novel word building (morphological composition)
- Sentence construction from vocabulary
- Disambiguation tests
- Error correction tests
- Proverb completion and interpretation
- Compound word building
- Dialect variation tests
- Cross-Manding cognate matching
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

from ..config import NKO_MORPHEMES, NKO_ALPHABET, BenchmarkConfig


@dataclass
class NovelWordTest:
    """Test for constructing novel N'Ko words from morphemes."""
    components: List[Dict[str, str]]  # List of morphemes with meanings
    expected_combination: str
    expected_meaning: str
    difficulty: str  # easy, medium, hard
    test_type: str = "novel_word"


@dataclass
class SentenceConstructionTest:
    """Test for constructing grammatical N'Ko sentences."""
    vocabulary: List[Dict[str, str]]  # Words with meanings
    target_meaning: str
    expected_sentence: str
    grammar_notes: List[str] = field(default_factory=list)
    difficulty: str = "medium"
    test_type: str = "sentence_construction"


@dataclass
class DisambiguationTest:
    """Test for understanding ambiguous N'Ko text."""
    nko_text: str
    interpretations: List[Dict[str, str]]  # Possible meanings
    context_hint: Optional[str] = None
    difficulty: str = "hard"
    test_type: str = "disambiguation"


@dataclass
class ErrorCorrectionTest:
    """Test for correcting errors in N'Ko text."""
    incorrect_text: str
    correct_text: str
    error_type: str  # spelling, grammar, tone, word_order
    error_description: str
    difficulty: str = "medium"
    test_type: str = "error_correction"


@dataclass
class ProverbTest:
    """Test for proverb completion and interpretation."""
    proverb_partial: str  # Beginning of proverb
    proverb_full: str  # Complete proverb
    language: str  # nko, bambara
    meaning: str
    cultural_context: Optional[str] = None
    difficulty: str = "hard"
    test_type: str = "proverb"


@dataclass
class CompoundWordTest:
    """Test for building compound words from roots."""
    root_words: List[Dict[str, str]]  # List of {word, meaning}
    expected_compound: str
    compound_meaning: str
    language: str  # nko, bambara
    difficulty: str = "medium"
    test_type: str = "compound_word"


@dataclass
class DialectVariationTest:
    """Test for recognizing dialect variations across Manding languages."""
    concept: str  # The meaning/concept
    bambara_form: str
    malinke_form: Optional[str] = None
    jula_form: Optional[str] = None
    nko_form: Optional[str] = None
    phonetic_notes: Optional[str] = None
    difficulty: str = "hard"
    test_type: str = "dialect_variation"


@dataclass
class CognateMatchTest:
    """Test for matching cognate words across Manding languages."""
    source_word: str
    source_language: str
    target_language: str
    expected_cognate: str
    shared_meaning: str
    difficulty: str = "medium"
    test_type: str = "cognate_match"


class ComplexTestGenerator:
    """
    Generates complex test cases that challenge true N'Ko understanding.
    
    These tests go beyond simple memorization to test:
    - Productive morphology (creating new words)
    - Syntactic knowledge (building sentences)
    - Semantic understanding (disambiguation)
    - Grammatical competence (error correction)
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.morphemes = NKO_MORPHEMES
        self.alphabet = NKO_ALPHABET
    
    def generate_novel_word_tests(self, count: int = 50) -> List[NovelWordTest]:
        """
        Generate tests for constructing novel N'Ko words from morphemes.
        
        Tests productive morphology - can the model combine morphemes
        to create valid words?
        """
        tests = []
        
        # Root + suffix combinations
        root_suffix_pairs = [
            # verb root + agent suffix
            {
                "components": [
                    {"morpheme": "ߞߊ߬ߙߊ߲", "meaning": "to study/learn", "type": "root"},
                    {"morpheme": "ߟߌ", "meaning": "agent/doer", "type": "suffix"},
                ],
                "expected": "ߞߊ߬ߙߊ߲߬ߟߌ",
                "meaning": "student/learner",
                "difficulty": "easy",
            },
            # verb root + place suffix
            {
                "components": [
                    {"morpheme": "ߞߍ", "meaning": "to do/make", "type": "root"},
                    {"morpheme": "ߟߊ", "meaning": "place of action", "type": "suffix"},
                ],
                "expected": "ߞߍߟߊ",
                "meaning": "workshop/place of work",
                "difficulty": "easy",
            },
            # noun + abstract quality
            {
                "components": [
                    {"morpheme": "ߢߌ߬ߡߊ", "meaning": "good/kind", "type": "root"},
                    {"morpheme": "ߦߊ", "meaning": "abstract quality", "type": "suffix"},
                ],
                "expected": "ߢߌ߬ߡߊ߬ߦߊ",
                "meaning": "goodness/kindness",
                "difficulty": "medium",
            },
            # noun + augmentative
            {
                "components": [
                    {"morpheme": "ߡߏ߬ߛߏ", "meaning": "woman", "type": "root"},
                    {"morpheme": "ߓߊ", "meaning": "big/great", "type": "suffix"},
                ],
                "expected": "ߡߏ߬ߛߏ߬ߓߊ",
                "meaning": "great woman/lady",
                "difficulty": "easy",
            },
            # Complex: prefix + root + suffix
            {
                "components": [
                    {"morpheme": "ߡߊ߬", "meaning": "causative", "type": "prefix"},
                    {"morpheme": "ߟߐ߲", "meaning": "knowledge", "type": "root"},
                    {"morpheme": "ߟߌ", "meaning": "agent", "type": "suffix"},
                ],
                "expected": "ߡߊ߬ߟߐ߲߬ߟߌ",
                "meaning": "teacher (one who causes knowledge)",
                "difficulty": "hard",
            },
        ]
        
        # Generate tests from predefined pairs
        for pair in root_suffix_pairs:
            tests.append(NovelWordTest(
                components=pair["components"],
                expected_combination=pair["expected"],
                expected_meaning=pair["meaning"],
                difficulty=pair["difficulty"],
            ))
        
        # Generate additional randomized combinations
        roots = list(self.morphemes.get("roots", {}).items())
        suffixes = list(self.morphemes.get("suffixes", {}).items())
        
        while len(tests) < count and roots and suffixes:
            root_key, root_data = random.choice(roots)
            suffix_key, suffix_data = random.choice(suffixes)
            
            # Create a plausible combination prompt
            tests.append(NovelWordTest(
                components=[
                    {"morpheme": root_key, "meaning": root_data.get("meaning", ""), "type": "root"},
                    {"morpheme": suffix_key, "meaning": suffix_data.get("meaning", ""), "type": "suffix"},
                ],
                expected_combination=f"{root_key}{suffix_key}",  # Simplified
                expected_meaning=f"{root_data.get('meaning', '')} + {suffix_data.get('meaning', '')}",
                difficulty="medium",
            ))
        
        return tests[:count]
    
    def generate_sentence_construction_tests(self, count: int = 50) -> List[SentenceConstructionTest]:
        """
        Generate tests for constructing N'Ko sentences from vocabulary.
        
        Tests syntactic knowledge - can the model arrange words
        into grammatically correct sentences?
        """
        tests = []
        
        # Predefined sentence construction tasks
        sentence_tasks = [
            {
                "vocabulary": [
                    {"nko": "ߒ", "meaning": "I"},
                    {"nko": "ߓߍ߫", "meaning": "am (present marker)"},
                    {"nko": "ߕߊ߯", "meaning": "to go"},
                    {"nko": "ߛߎ߬ߜߎ", "meaning": "market"},
                    {"nko": "ߟߊ", "meaning": "to/at (postposition)"},
                ],
                "target": "I am going to the market",
                "expected": "ߒ ߓߍ߫ ߕߊ߯ ߛߎ߬ߜߎ ߟߊ",
                "notes": ["Subject-Verb-Object-Postposition order"],
                "difficulty": "easy",
            },
            {
                "vocabulary": [
                    {"nko": "ߘߋ߲ߣߍ߲", "meaning": "children"},
                    {"nko": "ߓߍ߫", "meaning": "are (present marker)"},
                    {"nko": "ߕߎ߬ߟߐ߲", "meaning": "playing"},
                    {"nko": "ߟߎ", "meaning": "courtyard"},
                    {"nko": "ߞߣߐ߫", "meaning": "in"},
                ],
                "target": "The children are playing in the courtyard",
                "expected": "ߘߋ߲ߣߍ߲ ߓߍ߫ ߕߎ߬ߟߐ߲ ߟߎ ߞߣߐ߫",
                "notes": ["Plural noun subject", "Locative postposition"],
                "difficulty": "medium",
            },
            {
                "vocabulary": [
                    {"nko": "ߊ߬", "meaning": "he/she"},
                    {"nko": "ߦߋ߫", "meaning": "past marker"},
                    {"nko": "ߣߊ߬", "meaning": "to come"},
                    {"nko": "ߛߎ", "meaning": "yesterday"},
                ],
                "target": "He/she came yesterday",
                "expected": "ߊ߬ ߦߋ߫ ߣߊ߬ ߛߎ",
                "notes": ["Past tense construction", "Time adverb at end"],
                "difficulty": "medium",
            },
            {
                "vocabulary": [
                    {"nko": "ߡߎ߲߬", "meaning": "what"},
                    {"nko": "ߦߋ߫", "meaning": "is"},
                    {"nko": "ߌ", "meaning": "your"},
                    {"nko": "ߕߐ߮", "meaning": "name"},
                ],
                "target": "What is your name?",
                "expected": "ߌ ߕߐ߮ ߦߋ߫ ߡߎ߲߬؟",
                "notes": ["Question formation", "Possessive construction"],
                "difficulty": "easy",
            },
            {
                "vocabulary": [
                    {"nko": "ߒ", "meaning": "I"},
                    {"nko": "ߓ߬", "meaning": "love (contracted form)"},
                    {"nko": "ߌ", "meaning": "you"},
                    {"nko": "ߝߍ߬", "meaning": "with/towards"},
                ],
                "target": "I love you",
                "expected": "ߒ ߓ߬ ߌ ߝߍ߬",
                "notes": ["Verb with postposition complement"],
                "difficulty": "medium",
            },
        ]
        
        for task in sentence_tasks:
            tests.append(SentenceConstructionTest(
                vocabulary=task["vocabulary"],
                target_meaning=task["target"],
                expected_sentence=task["expected"],
                grammar_notes=task["notes"],
                difficulty=task["difficulty"],
            ))
        
        # Duplicate with variations to reach count
        while len(tests) < count:
            original = random.choice(sentence_tasks)
            tests.append(SentenceConstructionTest(
                vocabulary=original["vocabulary"],
                target_meaning=original["target"],
                expected_sentence=original["expected"],
                grammar_notes=original["notes"],
                difficulty=original["difficulty"],
            ))
        
        return tests[:count]
    
    def generate_disambiguation_tests(self, count: int = 30) -> List[DisambiguationTest]:
        """
        Generate tests for understanding ambiguous N'Ko text.
        
        Tests semantic understanding - can the model identify
        multiple valid interpretations?
        """
        tests = []
        
        # Predefined ambiguous cases
        ambiguous_cases = [
            {
                "text": "ߊ߬ ߓߍ߫ ߕߊ߯",
                "interpretations": [
                    {"meaning": "He/she is going", "context": "talking about a person"},
                    {"meaning": "It is going", "context": "talking about a thing/animal"},
                ],
                "difficulty": "easy",
            },
            {
                "text": "ߛߏ ߓߍ߫ ߖߊ߬",
                "interpretations": [
                    {"meaning": "The house is broken", "context": "building damage"},
                    {"meaning": "The city is destroyed", "context": "larger scale destruction"},
                ],
                "hint": "ߛߏ can mean both 'house' and 'city'",
                "difficulty": "medium",
            },
            {
                "text": "ߒ ߡߊ߫ ߊ߬ ߟߐ߲",
                "interpretations": [
                    {"meaning": "I didn't know it", "context": "past ignorance"},
                    {"meaning": "I don't know him/her", "context": "not acquainted"},
                    {"meaning": "I haven't learned it", "context": "education context"},
                ],
                "difficulty": "hard",
            },
        ]
        
        for case in ambiguous_cases:
            tests.append(DisambiguationTest(
                nko_text=case["text"],
                interpretations=case["interpretations"],
                context_hint=case.get("hint"),
                difficulty=case["difficulty"],
            ))
        
        while len(tests) < count:
            original = random.choice(ambiguous_cases)
            tests.append(DisambiguationTest(
                nko_text=original["text"],
                interpretations=original["interpretations"],
                context_hint=original.get("hint"),
                difficulty=original["difficulty"],
            ))
        
        return tests[:count]
    
    def generate_error_correction_tests(self, count: int = 50) -> List[ErrorCorrectionTest]:
        """
        Generate tests for correcting errors in N'Ko text.
        
        Tests grammatical competence - can the model identify
        and correct various types of errors?
        """
        tests = []
        
        # Predefined error cases
        error_cases = [
            # Word order errors
            {
                "incorrect": "ߕߊ߯ ߓߍ߫ ߒ ߛߎ߬ߜߎ ߟߊ",
                "correct": "ߒ ߓߍ߫ ߕߊ߯ ߛߎ߬ߜߎ ߟߊ",
                "type": "word_order",
                "description": "Subject should come before verb marker",
                "difficulty": "medium",
            },
            # Missing tone mark
            {
                "incorrect": "ߒ ߓߍ ߕߊ߯",
                "correct": "ߒ ߓߍ߫ ߕߊ߯",
                "type": "tone",
                "description": "Missing tone mark on verb marker ߓߍ߫",
                "difficulty": "easy",
            },
            # Wrong postposition
            {
                "incorrect": "ߒ ߓߍ߫ ߕߊ߯ ߛߎ߬ߜߎ ߞߣߐ߫",
                "correct": "ߒ ߓߍ߫ ߕߊ߯ ߛߎ߬ߜߎ ߟߊ",
                "type": "grammar",
                "description": "Use ߟߊ (to) not ߞߣߐ߫ (in) for destination",
                "difficulty": "hard",
            },
            # Spelling error (wrong character)
            {
                "incorrect": "ߛߏ ߝߏ߬ߙߏ",
                "correct": "ߛߏ ߝߐ߬ߙߐ",
                "type": "spelling",
                "description": "Incorrect vowels in word",
                "difficulty": "medium",
            },
            # Agreement error
            {
                "incorrect": "ߘߋ߲ ߓߍ߫ ߕߊ߯ (for plural)",
                "correct": "ߘߋ߲ߣߍ߲ ߓߍ߫ ߕߊ߯",
                "type": "grammar",
                "description": "Plural marker missing on noun",
                "difficulty": "medium",
            },
        ]
        
        for case in error_cases:
            tests.append(ErrorCorrectionTest(
                incorrect_text=case["incorrect"],
                correct_text=case["correct"],
                error_type=case["type"],
                error_description=case["description"],
                difficulty=case["difficulty"],
            ))
        
        while len(tests) < count:
            original = random.choice(error_cases)
            tests.append(ErrorCorrectionTest(
                incorrect_text=original["incorrect"],
                correct_text=original["correct"],
                error_type=original["type"],
                error_description=original["description"],
                difficulty=original["difficulty"],
            ))
        
        return tests[:count]
    
    def generate_proverb_tests(self, count: int = 50) -> List[ProverbTest]:
        """
        Generate proverb completion and interpretation tests.
        
        Tests cultural and linguistic understanding through traditional
        Manding proverbs in N'Ko and Bambara.
        """
        tests = []
        
        # Manding proverbs with meanings
        proverbs = [
            # Bambara proverbs
            {
                "partial": "Mɔgɔ tɛ mɔgɔ ye...",
                "full": "Mɔgɔ tɛ mɔgɔ ye, mɔgɔ de ye mɔgɔ ka fɛn ye",
                "language": "bambara",
                "meaning": "No one is useless, everyone has something of value to offer",
                "context": "Used to remind people of inherent human worth",
                "difficulty": "hard",
            },
            {
                "partial": "Hakilima tɛ...",
                "full": "Hakilima tɛ hakili kelen na",
                "language": "bambara",
                "meaning": "A wise person doesn't have just one idea (wisdom has many facets)",
                "context": "Encourages open-mindedness and multiple perspectives",
                "difficulty": "medium",
            },
            {
                "partial": "Kuma tɛ fɔ...",
                "full": "Kuma tɛ fɔ ka taa, a bɛ fɔ ka na",
                "language": "bambara",
                "meaning": "Words are not said to go away, they are said to come back",
                "context": "Warning about the consequences of speech",
                "difficulty": "hard",
            },
            {
                "partial": "Den ye furusiri ye...",
                "full": "Den ye furusiri ye, n'a tɛ ɲɛnama ye a tɛ saya ye",
                "language": "bambara",
                "meaning": "A child is like clothing - it's neither life nor death (children are precious but not everything)",
                "context": "About the importance but not absoluteness of children",
                "difficulty": "hard",
            },
            {
                "partial": "Saya bɛ...",
                "full": "Saya bɛ mɔgɔ bɛɛ bolo",
                "language": "bambara",
                "meaning": "Death comes to everyone",
                "context": "Universal truth about mortality",
                "difficulty": "easy",
            },
            # N'Ko proverbs
            {
                "partial": "ߡߐ߰ ߛߌ߫ ߕߍ߫ ߕߏ߫...",
                "full": "ߡߐ߰ ߛߌ߫ ߕߍ߫ ߕߏ߫ ߞߊ߬ ߘߎ߰ ߝߊ߲߭ ߡߍ߲ ߡߊ߬",
                "language": "nko",
                "meaning": "No one is left to own any part of the earth",
                "context": "About the impermanence of earthly possessions",
                "difficulty": "hard",
            },
            {
                "partial": "ߞߎ߲߬ߠߊ߬ߦߊ...",
                "full": "ߞߎ߲߬ߠߊ߬ߦߊ ߕߍ߫ ߕߏ߫ ߘߐ߫ ߟߊ߬ߓߊ߰ߙߊ߬ߟߌ ߘߐ߫",
                "language": "nko",
                "meaning": "Patience is the key to success",
                "context": "Encouragement to be patient in work",
                "difficulty": "medium",
            },
        ]
        
        for proverb in proverbs:
            tests.append(ProverbTest(
                proverb_partial=proverb["partial"],
                proverb_full=proverb["full"],
                language=proverb["language"],
                meaning=proverb["meaning"],
                cultural_context=proverb.get("context"),
                difficulty=proverb["difficulty"],
            ))
        
        # Pad with repeats if needed
        while len(tests) < count:
            original = random.choice(proverbs)
            tests.append(ProverbTest(
                proverb_partial=original["partial"],
                proverb_full=original["full"],
                language=original["language"],
                meaning=original["meaning"],
                cultural_context=original.get("context"),
                difficulty=original["difficulty"],
            ))
        
        return tests[:count]
    
    def generate_compound_word_tests(self, count: int = 50) -> List[CompoundWordTest]:
        """
        Generate compound word building tests.
        
        Tests morphological productivity - can the model build
        compound words from component parts?
        """
        tests = []
        
        # Compound words in Bambara and N'Ko
        compounds = [
            # Bambara compounds
            {
                "roots": [
                    {"word": "so", "meaning": "house"},
                    {"word": "ba", "meaning": "big/mother"},
                ],
                "compound": "soba",
                "meaning": "big house / main house",
                "language": "bambara",
                "difficulty": "easy",
            },
            {
                "roots": [
                    {"word": "den", "meaning": "child"},
                    {"word": "muso", "meaning": "woman/female"},
                ],
                "compound": "denmuso",
                "meaning": "girl / daughter",
                "language": "bambara",
                "difficulty": "easy",
            },
            {
                "roots": [
                    {"word": "ji", "meaning": "water"},
                    {"word": "kɔnɔ", "meaning": "inside"},
                ],
                "compound": "jikɔnɔ",
                "meaning": "underwater / in the water",
                "language": "bambara",
                "difficulty": "medium",
            },
            {
                "roots": [
                    {"word": "baara", "meaning": "work"},
                    {"word": "kɛla", "meaning": "doer"},
                ],
                "compound": "baarakɛla",
                "meaning": "worker",
                "language": "bambara",
                "difficulty": "medium",
            },
            {
                "roots": [
                    {"word": "fɔli", "meaning": "greeting"},
                    {"word": "kan", "meaning": "word/language"},
                ],
                "compound": "fɔlikan",
                "meaning": "greeting words / salutation",
                "language": "bambara",
                "difficulty": "hard",
            },
            # N'Ko compounds
            {
                "roots": [
                    {"word": "ߛߏ", "meaning": "house"},
                    {"word": "ߓߊ", "meaning": "big"},
                ],
                "compound": "ߛߏ߬ߓߊ",
                "meaning": "big house",
                "language": "nko",
                "difficulty": "easy",
            },
            {
                "roots": [
                    {"word": "ߞߊ߬ߙߊ߲", "meaning": "study/learn"},
                    {"word": "ߟߊ", "meaning": "place"},
                ],
                "compound": "ߞߊ߬ߙߊ߲߬ߟߊ",
                "meaning": "school / place of study",
                "language": "nko",
                "difficulty": "medium",
            },
            {
                "roots": [
                    {"word": "ߟߐ߲", "meaning": "knowledge"},
                    {"word": "ߕߌ߮", "meaning": "person/master"},
                ],
                "compound": "ߟߐ߲߬ߕߌ߮",
                "meaning": "scholar / knowledgeable person",
                "language": "nko",
                "difficulty": "hard",
            },
        ]
        
        for compound in compounds:
            tests.append(CompoundWordTest(
                root_words=compound["roots"],
                expected_compound=compound["compound"],
                compound_meaning=compound["meaning"],
                language=compound["language"],
                difficulty=compound["difficulty"],
            ))
        
        while len(tests) < count:
            original = random.choice(compounds)
            tests.append(CompoundWordTest(
                root_words=original["roots"],
                expected_compound=original["compound"],
                compound_meaning=original["meaning"],
                language=original["language"],
                difficulty=original["difficulty"],
            ))
        
        return tests[:count]
    
    def generate_dialect_variation_tests(self, count: int = 30) -> List[DialectVariationTest]:
        """
        Generate dialect variation tests across Manding languages.
        
        Tests understanding of dialectal differences between
        Bambara, Malinke, and Jula.
        """
        tests = []
        
        # Cross-dialect equivalents
        variations = [
            {
                "concept": "water",
                "bambara": "ji",
                "malinke": "ji",
                "jula": "ji",
                "nko": "ߖߌ",
                "notes": "Same across all variants",
                "difficulty": "easy",
            },
            {
                "concept": "to go",
                "bambara": "taa",
                "malinke": "ta",
                "jula": "taa",
                "nko": "ߕߊ߯",
                "notes": "Slight vowel length difference in Malinke",
                "difficulty": "medium",
            },
            {
                "concept": "house",
                "bambara": "so",
                "malinke": "bun",
                "jula": "so",
                "nko": "ߛߏ",
                "notes": "Malinke uses a different root",
                "difficulty": "hard",
            },
            {
                "concept": "woman",
                "bambara": "muso",
                "malinke": "musoo",
                "jula": "muso",
                "nko": "ߡߎ߬ߛߏ",
                "notes": "Malinke has vowel lengthening",
                "difficulty": "medium",
            },
            {
                "concept": "to come",
                "bambara": "na",
                "malinke": "na",
                "jula": "na",
                "nko": "ߣߊ߬",
                "notes": "Identical across variants",
                "difficulty": "easy",
            },
            {
                "concept": "to eat",
                "bambara": "dun",
                "malinke": "dɔmɔ",
                "jula": "dun",
                "nko": "ߘߎ߲",
                "notes": "Malinke has expanded form",
                "difficulty": "hard",
            },
            {
                "concept": "big/great",
                "bambara": "ba",
                "malinke": "ba",
                "jula": "ba",
                "nko": "ߓߊ",
                "notes": "Same across all variants",
                "difficulty": "easy",
            },
            {
                "concept": "child",
                "bambara": "den",
                "malinke": "din",
                "jula": "den",
                "nko": "ߘߋ߲",
                "notes": "Vowel variation in Malinke",
                "difficulty": "medium",
            },
        ]
        
        for var in variations:
            tests.append(DialectVariationTest(
                concept=var["concept"],
                bambara_form=var["bambara"],
                malinke_form=var.get("malinke"),
                jula_form=var.get("jula"),
                nko_form=var.get("nko"),
                phonetic_notes=var.get("notes"),
                difficulty=var["difficulty"],
            ))
        
        while len(tests) < count:
            original = random.choice(variations)
            tests.append(DialectVariationTest(
                concept=original["concept"],
                bambara_form=original["bambara"],
                malinke_form=original.get("malinke"),
                jula_form=original.get("jula"),
                nko_form=original.get("nko"),
                phonetic_notes=original.get("notes"),
                difficulty=original["difficulty"],
            ))
        
        return tests[:count]
    
    def generate_cognate_match_tests(self, count: int = 30) -> List[CognateMatchTest]:
        """
        Generate cognate matching tests across Manding languages.
        
        Tests the ability to recognize related words across dialects.
        """
        tests = []
        
        # Cognate pairs
        cognates = [
            {"source": "ji", "source_lang": "bambara", "target_lang": "nko", "target": "ߖߌ", "meaning": "water", "difficulty": "easy"},
            {"source": "muso", "source_lang": "bambara", "target_lang": "nko", "target": "ߡߎ߬ߛߏ", "meaning": "woman", "difficulty": "easy"},
            {"source": "den", "source_lang": "bambara", "target_lang": "nko", "target": "ߘߋ߲", "meaning": "child", "difficulty": "easy"},
            {"source": "baara", "source_lang": "bambara", "target_lang": "nko", "target": "ߓߊ߯ߙߊ", "meaning": "work", "difficulty": "medium"},
            {"source": "sɛbɛn", "source_lang": "bambara", "target_lang": "nko", "target": "ߛߓߍ߲", "meaning": "to write", "difficulty": "medium"},
            {"source": "ߛߏ", "source_lang": "nko", "target_lang": "bambara", "target": "so", "meaning": "house", "difficulty": "easy"},
            {"source": "ߕߊ߯", "source_lang": "nko", "target_lang": "bambara", "target": "taa", "meaning": "to go", "difficulty": "easy"},
            {"source": "ߟߐ߲", "source_lang": "nko", "target_lang": "bambara", "target": "lɔn", "meaning": "knowledge", "difficulty": "medium"},
            {"source": "kelen", "source_lang": "bambara", "target_lang": "malinke", "target": "kelen", "meaning": "one", "difficulty": "easy"},
            {"source": "fila", "source_lang": "bambara", "target_lang": "jula", "target": "fila", "meaning": "two", "difficulty": "easy"},
        ]
        
        for cog in cognates:
            tests.append(CognateMatchTest(
                source_word=cog["source"],
                source_language=cog["source_lang"],
                target_language=cog["target_lang"],
                expected_cognate=cog["target"],
                shared_meaning=cog["meaning"],
                difficulty=cog["difficulty"],
            ))
        
        while len(tests) < count:
            original = random.choice(cognates)
            tests.append(CognateMatchTest(
                source_word=original["source"],
                source_language=original["source_lang"],
                target_language=original["target_lang"],
                expected_cognate=original["target"],
                shared_meaning=original["meaning"],
                difficulty=original["difficulty"],
            ))
        
        return tests[:count]
    
    def generate_all_tests(self) -> Dict[str, List]:
        """
        Generate all complex test types.
        
        Returns:
            Dict with keys for each test type
        """
        samples = self.config.compositional_samples
        
        # Distribute samples across test types (8 types now)
        per_type = max(10, samples // 8)
        
        return {
            "novel_word": self.generate_novel_word_tests(per_type),
            "sentence_construction": self.generate_sentence_construction_tests(per_type),
            "disambiguation": self.generate_disambiguation_tests(per_type),
            "error_correction": self.generate_error_correction_tests(per_type),
            "proverb": self.generate_proverb_tests(per_type),
            "compound_word": self.generate_compound_word_tests(per_type),
            "dialect_variation": self.generate_dialect_variation_tests(per_type),
            "cognate_match": self.generate_cognate_match_tests(per_type),
        }
    
    def to_benchmark_format(self) -> List[Dict[str, Any]]:
        """
        Convert all tests to a unified benchmark format.
        
        Returns:
            List of test dictionaries ready for benchmarking
        """
        all_tests = self.generate_all_tests()
        
        benchmark_tests = []
        
        for test in all_tests["novel_word"]:
            benchmark_tests.append({
                "type": "novel_word",
                "difficulty": test.difficulty,
                "prompt": f"Combine these N'Ko morphemes to form a word: {[c['morpheme'] + ' (' + c['meaning'] + ')' for c in test.components]}",
                "expected": test.expected_combination,
                "expected_meaning": test.expected_meaning,
            })
        
        for test in all_tests["sentence_construction"]:
            benchmark_tests.append({
                "type": "sentence_construction",
                "difficulty": test.difficulty,
                "prompt": f"Construct a N'Ko sentence meaning '{test.target_meaning}' using: {test.vocabulary}",
                "expected": test.expected_sentence,
                "notes": test.grammar_notes,
            })
        
        for test in all_tests["disambiguation"]:
            benchmark_tests.append({
                "type": "disambiguation",
                "difficulty": test.difficulty,
                "prompt": f"What are all possible meanings of this N'Ko text: {test.nko_text}",
                "expected_interpretations": test.interpretations,
                "hint": test.context_hint,
            })
        
        for test in all_tests["error_correction"]:
            benchmark_tests.append({
                "type": "error_correction",
                "difficulty": test.difficulty,
                "prompt": f"Correct any errors in this N'Ko text: {test.incorrect_text}",
                "expected": test.correct_text,
                "error_type": test.error_type,
                "explanation": test.error_description,
            })
        
        # New test types for Manding benchmark
        for test in all_tests["proverb"]:
            benchmark_tests.append({
                "type": "proverb_completion",
                "difficulty": test.difficulty,
                "language": test.language,
                "prompt": f"Complete this {test.language.upper()} proverb: {test.proverb_partial}",
                "expected": test.proverb_full,
                "meaning": test.meaning,
                "cultural_context": test.cultural_context,
            })
        
        for test in all_tests["compound_word"]:
            root_desc = " + ".join([f"{r['word']} ({r['meaning']})" for r in test.root_words])
            benchmark_tests.append({
                "type": "compound_word",
                "difficulty": test.difficulty,
                "language": test.language,
                "prompt": f"Combine these {test.language.upper()} words into a compound: {root_desc}",
                "expected": test.expected_compound,
                "expected_meaning": test.compound_meaning,
            })
        
        for test in all_tests["dialect_variation"]:
            benchmark_tests.append({
                "type": "dialect_variation",
                "difficulty": test.difficulty,
                "prompt": f"Given Bambara '{test.bambara_form}' meaning '{test.concept}', what is the equivalent in Malinke?",
                "expected": test.malinke_form,
                "bambara": test.bambara_form,
                "jula": test.jula_form,
                "nko": test.nko_form,
                "notes": test.phonetic_notes,
            })
        
        for test in all_tests["cognate_match"]:
            benchmark_tests.append({
                "type": "cognate_match",
                "difficulty": test.difficulty,
                "prompt": f"What is the {test.target_language} cognate for {test.source_language} '{test.source_word}' (meaning: {test.shared_meaning})?",
                "source": test.source_word,
                "source_language": test.source_language,
                "target_language": test.target_language,
                "expected": test.expected_cognate,
                "meaning": test.shared_meaning,
            })
        
        return benchmark_tests


def generate_compositional_tests(config: Optional[BenchmarkConfig] = None) -> List[Dict[str, Any]]:
    """Convenience function to generate compositional tests."""
    generator = ComplexTestGenerator(config)
    return generator.to_benchmark_format()

