"""
Curriculum-Based Progressive Difficulty Testing.

Implements CEFR-aligned (A1-C2) progressive difficulty levels for
evaluating AI model comprehension across the Manding language learning journey.

Levels:
- A1: Basic greetings, numbers, common words
- A2: Simple sentences, questions, basic grammar
- B1: Complex sentences, proverbs, intermediate vocabulary
- B2: Idiomatic expressions, cultural references, formal/informal registers
- C1: Error correction, novel compositions, nuanced meanings
- C2: Proverb completion, literary translation, dialectal variations
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from ..data.manding_loader import (
    Language,
    TranslationPair,
    VocabEntry,
    MandingTestSet,
)
from ..providers.base import BaseProvider


class CEFRLevel(Enum):
    """Common European Framework of Reference levels."""
    A1 = "A1"  # Beginner
    A2 = "A2"  # Elementary
    B1 = "B1"  # Intermediate
    B2 = "B2"  # Upper Intermediate
    C1 = "C1"  # Advanced
    C2 = "C2"  # Proficient


@dataclass
class CurriculumTestItem:
    """A single curriculum test item."""
    id: str
    level: CEFRLevel
    task_type: str  # translation, fill_blank, multiple_choice, etc.
    prompt: str
    reference_answer: str
    options: List[str] = field(default_factory=list)  # For multiple choice
    hints: List[str] = field(default_factory=list)
    topic: str = "general"
    source_lang: Language = Language.NKO
    target_lang: Language = Language.ENGLISH
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class CurriculumResult:
    """Result of a curriculum test item."""
    item_id: str
    level: CEFRLevel
    task_type: str
    prompt: str
    reference_answer: str
    prediction: Optional[str] = None
    is_correct: bool = False
    partial_score: float = 0.0
    latency_ms: int = 0
    error: Optional[str] = None


# =============================================================================
# Level-specific content templates
# =============================================================================

# A1 Level: Basic greetings, numbers, common words
A1_TEMPLATES = {
    "greetings": [
        {"nko": "ߌ ߣߌ߫ ߛߍ߫", "bambara": "I ni ce", "english": "Hello", "french": "Bonjour"},
        {"nko": "ߌ ߣߌ߫ ߛߐ߯ߞߐߡߊ߫", "bambara": "I ni sogoma", "english": "Good morning", "french": "Bonjour (matin)"},
        {"nko": "ߌ ߣߌ߫ ߕߟߋ߬", "bambara": "I ni tile", "english": "Good afternoon", "french": "Bonjour (après-midi)"},
        {"nko": "ߌ ߣߌ߫ ߥߎ߬ߟߊ߫", "bambara": "I ni wula", "english": "Good evening", "french": "Bonsoir"},
    ],
    "numbers": [
        {"nko": "ߞߋߟߋ߲߫", "bambara": "kelen", "english": "one", "french": "un"},
        {"nko": "ߝߌ߬ߟߊ", "bambara": "fila", "english": "two", "french": "deux"},
        {"nko": "ߛߓߊ", "bambara": "saba", "english": "three", "french": "trois"},
        {"nko": "ߣߊ߯ߣߌ߲", "bambara": "naani", "english": "four", "french": "quatre"},
        {"nko": "ߘߎ߯ߙߎ߲", "bambara": "duuru", "english": "five", "french": "cinq"},
    ],
    "common_words": [
        {"nko": "ߛߌ߫", "bambara": "ji", "english": "water", "french": "eau"},
        {"nko": "ߘߐ߬ߞ", "bambara": "dumuni", "english": "food", "french": "nourriture"},
        {"nko": "ߡߊ߰", "bambara": "maa", "english": "person", "french": "personne"},
    ],
}

# A2 Level: Simple sentences, questions
A2_TEMPLATES = {
    "simple_sentences": [
        {"nko": "ߒ ߓߍ߫ ߓߊ߯ߙߊ ߟߊ߫", "bambara": "N bɛ baara la", "english": "I am working", "french": "Je travaille"},
        {"nko": "ߊ߬ ߓߍ߫ ߕߊ߯ߡߊ ߟߊ߫", "bambara": "A bɛ taama la", "english": "He/She is walking", "french": "Il/Elle marche"},
    ],
    "questions": [
        {"nko": "ߌ ߕߐ߮ ߘߋ߬؟", "bambara": "I tɔgɔ de?", "english": "What is your name?", "french": "Quel est ton nom?"},
        {"nko": "ߌ ߓߍ߫ ߡߌ߲߬ ߘߐ߫؟", "bambara": "I bɛ min na?", "english": "Where are you from?", "french": "D'où viens-tu?"},
    ],
}

# B1 Level: Complex sentences, proverbs
B1_TEMPLATES = {
    "complex_sentences": [
        {
            "nko": "ߣߌ߫ ߌ ߓߍ߫ ߓߊߡߊ߲ߞߊ߲ ߝߐ߫ ، ߊ߬ ߘߌ߫ ߒ ߘߌ߬ߦߊ ߞߏߛߓߍ",
            "bambara": "Ni i bɛ Bamanankan fɔ, a di n diya kosɛbɛ",
            "english": "If you speak Bambara, it makes me very happy",
            "french": "Si tu parles bambara, ça me fait très plaisir",
        },
    ],
    "proverbs": [
        {
            "text": "ߡߐ߰ ߛߌ߫ ߕߍ߫ ߕߏ߫ ߞߊ߬ ߘߎ߰ ߝߊ߲߭ ߡߍ߲ ߡߊ߬ ، ߊ߬ ߦߋ߫ ߘߐ߫ ߟߋ߬",
            "meaning": "No one is left to own any part of the earth; it owns them",
            "topic": "wisdom",
        },
    ],
}

# B2 Level: Idiomatic expressions, cultural references
B2_TEMPLATES = {
    "idioms": [
        {
            "expression": "ߘߋ߲߬ ߕߍ߫ ߘߋ߲߬ ߦߋ߫",
            "literal": "A child is not a child",
            "meaning": "Children are capable of great things",
            "usage": "cultural",
        },
    ],
    "formal_informal": [
        {
            "formal": "ߊ߬ ߘߌ߫ ߒ ߘߌ߬ߦߊ",
            "informal": "ߊ ߘ ߒ ߘ",
            "meaning": "It pleases me",
        },
    ],
}

# C1 Level: Error correction, novel compositions
C1_TEMPLATES = {
    "error_correction": [
        {
            "incorrect": "ߒ ߓߍ ߕߊ߯ߡߊ ߟߊ",  # Missing tone marks
            "correct": "ߒ ߓߍ߫ ߕߊ߯ߡߊ ߟߊ߫",
            "error_type": "tone_marks",
        },
    ],
    "composition_prompts": [
        {
            "prompt": "Write a short greeting for a formal letter in N'Ko",
            "key_elements": ["respect", "blessing", "formal register"],
        },
    ],
}

# C2 Level: Literary translation, dialectal variations
C2_TEMPLATES = {
    "literary": [
        {
            "text": "ߊߟߎ߫ ߦߋ߫ ߡߏ߬ߛߏ ߟߋ߬ ، ߊ߬ ߓߊ ߦߋ߫ ߡߏ߬ߛߏ ߟߋ߬",
            "translation": "She is a woman, and her mother is a woman",
            "literary_context": "Traditional saying about lineage",
        },
    ],
    "dialect_comparison": [
        {
            "concept": "hello",
            "bambara": "I ni ce",
            "malinke": "I ni ce",
            "jula": "I ni cɛ",
            "differences": "Minor phonetic variations",
        },
    ],
}


class CurriculumTestGenerator:
    """
    Generates curriculum-based tests at different CEFR levels.
    """
    
    def __init__(self, testset: Optional[MandingTestSet] = None):
        self.testset = testset
        self.templates = {
            CEFRLevel.A1: A1_TEMPLATES,
            CEFRLevel.A2: A2_TEMPLATES,
            CEFRLevel.B1: B1_TEMPLATES,
            CEFRLevel.B2: B2_TEMPLATES,
            CEFRLevel.C1: C1_TEMPLATES,
            CEFRLevel.C2: C2_TEMPLATES,
        }
    
    def generate_level_tests(
        self,
        level: CEFRLevel,
        count: int = 50,
        source_lang: Language = Language.NKO,
        target_lang: Language = Language.ENGLISH,
    ) -> List[CurriculumTestItem]:
        """Generate tests for a specific CEFR level."""
        tests = []
        
        if level == CEFRLevel.A1:
            tests.extend(self._generate_a1_tests(count, source_lang, target_lang))
        elif level == CEFRLevel.A2:
            tests.extend(self._generate_a2_tests(count, source_lang, target_lang))
        elif level == CEFRLevel.B1:
            tests.extend(self._generate_b1_tests(count, source_lang, target_lang))
        elif level == CEFRLevel.B2:
            tests.extend(self._generate_b2_tests(count, source_lang, target_lang))
        elif level == CEFRLevel.C1:
            tests.extend(self._generate_c1_tests(count, source_lang, target_lang))
        elif level == CEFRLevel.C2:
            tests.extend(self._generate_c2_tests(count, source_lang, target_lang))
        
        return tests[:count]
    
    def _generate_a1_tests(
        self, count: int, source_lang: Language, target_lang: Language
    ) -> List[CurriculumTestItem]:
        """Generate A1 (Beginner) level tests."""
        tests = []
        templates = self.templates[CEFRLevel.A1]
        
        # Translation tests from greetings
        for i, item in enumerate(templates.get("greetings", [])):
            source = item.get(source_lang.value.lower(), item.get("nko", ""))
            target = item.get(target_lang.value.lower(), item.get("english", ""))
            
            if source and target:
                tests.append(CurriculumTestItem(
                    id=f"a1_greeting_{i}",
                    level=CEFRLevel.A1,
                    task_type="translation",
                    prompt=f"Translate to {target_lang.value}: {source}",
                    reference_answer=target,
                    topic="greetings",
                    source_lang=source_lang,
                    target_lang=target_lang,
                ))
        
        # Number tests
        for i, item in enumerate(templates.get("numbers", [])):
            source = item.get(source_lang.value.lower(), item.get("nko", ""))
            target = item.get(target_lang.value.lower(), item.get("english", ""))
            
            if source and target:
                tests.append(CurriculumTestItem(
                    id=f"a1_number_{i}",
                    level=CEFRLevel.A1,
                    task_type="translation",
                    prompt=f"What does this number mean: {source}",
                    reference_answer=target,
                    topic="numbers",
                    source_lang=source_lang,
                    target_lang=target_lang,
                ))
        
        # Multiple choice vocabulary
        vocab_items = templates.get("common_words", []) + templates.get("numbers", [])
        for i, item in enumerate(vocab_items[:count//3]):
            source = item.get("nko", "")
            correct = item.get("english", "")
            
            if source and correct:
                # Generate distractors
                all_meanings = [v.get("english", "") for v in vocab_items if v.get("english") != correct]
                distractors = random.sample(all_meanings, min(3, len(all_meanings)))
                options = [correct] + distractors
                random.shuffle(options)
                
                tests.append(CurriculumTestItem(
                    id=f"a1_vocab_mc_{i}",
                    level=CEFRLevel.A1,
                    task_type="multiple_choice",
                    prompt=f"What does '{source}' mean?",
                    reference_answer=correct,
                    options=options,
                    topic="vocabulary",
                    source_lang=Language.NKO,
                    target_lang=Language.ENGLISH,
                ))
        
        # Supplement with testset data if available
        if self.testset and len(tests) < count:
            easy_vocab = [v for v in self.testset.nko_vocabulary if v.cefr_level == "A1"]
            for i, vocab in enumerate(easy_vocab[:count - len(tests)]):
                if vocab.meaning_primary:
                    tests.append(CurriculumTestItem(
                        id=f"a1_vocab_{i}",
                        level=CEFRLevel.A1,
                        task_type="definition",
                        prompt=f"Define this word: {vocab.word}",
                        reference_answer=vocab.meaning_primary,
                        topic="vocabulary",
                        source_lang=Language.NKO,
                        target_lang=Language.ENGLISH,
                    ))
        
        return tests
    
    def _generate_a2_tests(
        self, count: int, source_lang: Language, target_lang: Language
    ) -> List[CurriculumTestItem]:
        """Generate A2 (Elementary) level tests."""
        tests = []
        templates = self.templates[CEFRLevel.A2]
        
        # Simple sentence translations
        for i, item in enumerate(templates.get("simple_sentences", [])):
            source = item.get(source_lang.value.lower(), item.get("bambara", ""))
            target = item.get(target_lang.value.lower(), item.get("english", ""))
            
            if source and target:
                tests.append(CurriculumTestItem(
                    id=f"a2_sentence_{i}",
                    level=CEFRLevel.A2,
                    task_type="translation",
                    prompt=f"Translate this sentence: {source}",
                    reference_answer=target,
                    topic="sentences",
                    source_lang=source_lang,
                    target_lang=target_lang,
                ))
        
        # Question translations
        for i, item in enumerate(templates.get("questions", [])):
            source = item.get(source_lang.value.lower(), item.get("bambara", ""))
            target = item.get(target_lang.value.lower(), item.get("english", ""))
            
            if source and target:
                tests.append(CurriculumTestItem(
                    id=f"a2_question_{i}",
                    level=CEFRLevel.A2,
                    task_type="translation",
                    prompt=f"Translate this question: {source}",
                    reference_answer=target,
                    topic="questions",
                    source_lang=source_lang,
                    target_lang=target_lang,
                ))
        
        # Supplement with Bambara-English pairs if available
        if self.testset and len(tests) < count:
            for i, pair in enumerate(self.testset.bambara_english_pairs[:count - len(tests)]):
                if pair.bambara_text and pair.english_text:
                    tests.append(CurriculumTestItem(
                        id=f"a2_bam_eng_{i}",
                        level=CEFRLevel.A2,
                        task_type="translation",
                        prompt=f"Translate from Bambara: {pair.bambara_text}",
                        reference_answer=pair.english_text,
                        topic="bambara_english",
                        source_lang=Language.BAMBARA,
                        target_lang=Language.ENGLISH,
                    ))
        
        return tests
    
    def _generate_b1_tests(
        self, count: int, source_lang: Language, target_lang: Language
    ) -> List[CurriculumTestItem]:
        """Generate B1 (Intermediate) level tests."""
        tests = []
        templates = self.templates[CEFRLevel.B1]
        
        # Complex sentences
        for i, item in enumerate(templates.get("complex_sentences", [])):
            source = item.get(source_lang.value.lower(), item.get("bambara", ""))
            target = item.get(target_lang.value.lower(), item.get("english", ""))
            
            if source and target:
                tests.append(CurriculumTestItem(
                    id=f"b1_complex_{i}",
                    level=CEFRLevel.B1,
                    task_type="translation",
                    prompt=f"Translate this complex sentence: {source}",
                    reference_answer=target,
                    topic="complex_sentences",
                    source_lang=source_lang,
                    target_lang=target_lang,
                ))
        
        # Proverb meaning
        for i, proverb in enumerate(templates.get("proverbs", [])):
            tests.append(CurriculumTestItem(
                id=f"b1_proverb_{i}",
                level=CEFRLevel.B1,
                task_type="explanation",
                prompt=f"Explain the meaning of this proverb: {proverb['text']}",
                reference_answer=proverb["meaning"],
                topic="proverbs",
                source_lang=Language.NKO,
                target_lang=Language.ENGLISH,
            ))
        
        # Add translations from nicolingua with medium complexity
        if self.testset and len(tests) < count:
            # Filter for medium-length sentences
            medium_pairs = [
                p for p in self.testset.nko_translations
                if p.nko_text and 20 < len(p.nko_text) < 100
            ]
            
            for i, pair in enumerate(random.sample(medium_pairs, min(count - len(tests), len(medium_pairs)))):
                target_text = pair.english_text or pair.french_text or ""
                if target_text:
                    tests.append(CurriculumTestItem(
                        id=f"b1_trans_{i}",
                        level=CEFRLevel.B1,
                        task_type="translation",
                        prompt=f"Translate: {pair.nko_text}",
                        reference_answer=target_text,
                        topic="translation",
                        source_lang=Language.NKO,
                        target_lang=target_lang,
                    ))
        
        return tests
    
    def _generate_b2_tests(
        self, count: int, source_lang: Language, target_lang: Language
    ) -> List[CurriculumTestItem]:
        """Generate B2 (Upper Intermediate) level tests."""
        tests = []
        templates = self.templates[CEFRLevel.B2]
        
        # Idiom interpretation
        for i, idiom in enumerate(templates.get("idioms", [])):
            tests.append(CurriculumTestItem(
                id=f"b2_idiom_{i}",
                level=CEFRLevel.B2,
                task_type="explanation",
                prompt=f"What does this idiom mean: '{idiom['expression']}' (literal: {idiom['literal']})",
                reference_answer=idiom["meaning"],
                topic="idioms",
                hints=[f"Literal meaning: {idiom['literal']}"],
                source_lang=Language.NKO,
                target_lang=Language.ENGLISH,
            ))
        
        # Formal/informal register
        for i, item in enumerate(templates.get("formal_informal", [])):
            tests.append(CurriculumTestItem(
                id=f"b2_register_{i}",
                level=CEFRLevel.B2,
                task_type="conversion",
                prompt=f"Convert this formal expression to informal: {item['formal']}",
                reference_answer=item["informal"],
                topic="register",
                source_lang=Language.NKO,
                target_lang=Language.NKO,
                metadata={"meaning": item["meaning"]},
            ))
        
        # Cultural context questions from Bambara corpus
        if self.testset and len(tests) < count:
            cultural_pairs = [
                p for p in self.testset.bambara_french_pairs
                if p.bambara_text and len(p.bambara_text) > 50
            ]
            
            for i, pair in enumerate(random.sample(cultural_pairs, min(count - len(tests), len(cultural_pairs)))):
                tests.append(CurriculumTestItem(
                    id=f"b2_cultural_{i}",
                    level=CEFRLevel.B2,
                    task_type="translation",
                    prompt=f"Translate with cultural awareness: {pair.bambara_text}",
                    reference_answer=pair.french_text,
                    topic="cultural",
                    source_lang=Language.BAMBARA,
                    target_lang=Language.FRENCH,
                ))
        
        return tests
    
    def _generate_c1_tests(
        self, count: int, source_lang: Language, target_lang: Language
    ) -> List[CurriculumTestItem]:
        """Generate C1 (Advanced) level tests."""
        tests = []
        templates = self.templates[CEFRLevel.C1]
        
        # Error correction
        for i, item in enumerate(templates.get("error_correction", [])):
            tests.append(CurriculumTestItem(
                id=f"c1_error_{i}",
                level=CEFRLevel.C1,
                task_type="error_correction",
                prompt=f"Correct the errors in this text: {item['incorrect']}",
                reference_answer=item["correct"],
                topic="error_correction",
                source_lang=Language.NKO,
                target_lang=Language.NKO,
                metadata={"error_type": item["error_type"]},
            ))
        
        # Composition prompts
        for i, item in enumerate(templates.get("composition_prompts", [])):
            tests.append(CurriculumTestItem(
                id=f"c1_compose_{i}",
                level=CEFRLevel.C1,
                task_type="composition",
                prompt=item["prompt"],
                reference_answer=", ".join(item["key_elements"]),
                topic="composition",
                source_lang=Language.ENGLISH,
                target_lang=Language.NKO,
                hints=item["key_elements"],
            ))
        
        # Complex back-translation
        if self.testset and len(tests) < count:
            complex_pairs = [
                p for p in self.testset.nko_translations
                if p.nko_text and len(p.nko_text) > 80
            ]
            
            for i, pair in enumerate(random.sample(complex_pairs, min(count - len(tests), len(complex_pairs)))):
                target_text = pair.english_text or pair.french_text or ""
                if target_text:
                    tests.append(CurriculumTestItem(
                        id=f"c1_back_{i}",
                        level=CEFRLevel.C1,
                        task_type="back_translation",
                        prompt=f"Translate to N'Ko: {target_text}",
                        reference_answer=pair.nko_text,
                        topic="back_translation",
                        source_lang=target_lang,
                        target_lang=Language.NKO,
                    ))
        
        return tests
    
    def _generate_c2_tests(
        self, count: int, source_lang: Language, target_lang: Language
    ) -> List[CurriculumTestItem]:
        """Generate C2 (Proficient) level tests."""
        tests = []
        templates = self.templates[CEFRLevel.C2]
        
        # Literary translation
        for i, item in enumerate(templates.get("literary", [])):
            tests.append(CurriculumTestItem(
                id=f"c2_literary_{i}",
                level=CEFRLevel.C2,
                task_type="literary_translation",
                prompt=f"Translate this literary text, preserving style: {item['text']}",
                reference_answer=item["translation"],
                topic="literary",
                source_lang=Language.NKO,
                target_lang=Language.ENGLISH,
                metadata={"context": item["literary_context"]},
            ))
        
        # Dialect comparison
        for i, item in enumerate(templates.get("dialect_comparison", [])):
            tests.append(CurriculumTestItem(
                id=f"c2_dialect_{i}",
                level=CEFRLevel.C2,
                task_type="dialect_comparison",
                prompt=f"Given '{item['bambara']}' in Bambara (meaning: {item['concept']}), what is the Jula equivalent?",
                reference_answer=item["jula"],
                topic="dialects",
                source_lang=Language.BAMBARA,
                target_lang=Language.JULA,
                metadata={
                    "malinke": item["malinke"],
                    "differences": item["differences"],
                },
            ))
        
        # Proverb completion from cognates
        if self.testset and len(tests) < count:
            for i, cognate in enumerate(self.testset.nko_bambara_cognates[:count - len(tests)]):
                if cognate.nko_form and cognate.bambara_form:
                    tests.append(CurriculumTestItem(
                        id=f"c2_cognate_{i}",
                        level=CEFRLevel.C2,
                        task_type="cognate_completion",
                        prompt=f"Given N'Ko '{cognate.nko_form}' meaning '{cognate.meaning}', what is the Bambara cognate?",
                        reference_answer=cognate.bambara_form,
                        topic="cognates",
                        source_lang=Language.NKO,
                        target_lang=Language.BAMBARA,
                    ))
        
        return tests
    
    def generate_all_levels(
        self,
        samples_per_level: int = 50,
        source_lang: Language = Language.NKO,
        target_lang: Language = Language.ENGLISH,
    ) -> Dict[CEFRLevel, List[CurriculumTestItem]]:
        """Generate tests for all CEFR levels."""
        return {
            level: self.generate_level_tests(level, samples_per_level, source_lang, target_lang)
            for level in CEFRLevel
        }


class CurriculumTask:
    """
    Run curriculum-based progressive difficulty tests.
    """
    
    def __init__(self):
        self.name = "curriculum"
        self.description = "CEFR A1-C2 progressive difficulty evaluation"
    
    async def run(
        self,
        provider: BaseProvider,
        test_items: List[CurriculumTestItem],
        progress_callback: Optional[callable] = None,
    ) -> List[CurriculumResult]:
        """
        Run curriculum tests on a provider.
        
        Args:
            provider: LLM provider to test
            test_items: List of curriculum test items
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of CurriculumResult objects
        """
        results = []
        
        for i, item in enumerate(test_items):
            if progress_callback:
                progress_callback(i, len(test_items), f"curriculum_{item.level.value}")
            
            prompt = self._build_prompt(item)
            
            try:
                import time
                start_time = time.time()
                
                response = await provider.complete(prompt)
                
                latency_ms = int(response.latency_ms)
                prediction = response.completion.strip() if response.success else ""
                
                is_correct, partial_score = self._evaluate_response(
                    prediction, item.reference_answer, item.task_type
                )
                
                result = CurriculumResult(
                    item_id=item.id,
                    level=item.level,
                    task_type=item.task_type,
                    prompt=item.prompt,
                    reference_answer=item.reference_answer,
                    prediction=prediction,
                    is_correct=is_correct,
                    partial_score=partial_score,
                    latency_ms=latency_ms,
                )
                
            except Exception as e:
                result = CurriculumResult(
                    item_id=item.id,
                    level=item.level,
                    task_type=item.task_type,
                    prompt=item.prompt,
                    reference_answer=item.reference_answer,
                    error=str(e),
                )
            
            results.append(result)
        
        return results
    
    def _build_prompt(self, item: CurriculumTestItem) -> str:
        """Build prompt for a curriculum test item."""
        if item.task_type == "multiple_choice":
            options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(item.options)])
            return f"{item.prompt}\n\n{options_str}\n\nAnswer with just the letter:"
        
        elif item.task_type == "error_correction":
            return f"{item.prompt}\n\nProvide only the corrected text:"
        
        elif item.task_type == "explanation":
            return f"{item.prompt}\n\nProvide a brief explanation:"
        
        else:
            return f"{item.prompt}\n\nAnswer:"
    
    def _evaluate_response(
        self,
        prediction: str,
        reference: str,
        task_type: str,
    ) -> Tuple[bool, float]:
        """Evaluate a curriculum test response."""
        if not prediction:
            return False, 0.0
        
        pred_lower = prediction.lower().strip()
        ref_lower = reference.lower().strip()
        
        # Exact match
        if pred_lower == ref_lower:
            return True, 1.0
        
        # For multiple choice, check letter match
        if task_type == "multiple_choice":
            if len(pred_lower) == 1 and pred_lower.isalpha():
                return False, 0.0  # Can't verify without options
        
        # Word overlap for partial credit
        pred_words = set(pred_lower.split())
        ref_words = set(ref_lower.split())
        
        if not ref_words:
            return False, 0.0
        
        overlap = len(pred_words & ref_words)
        partial = overlap / len(ref_words)
        
        return partial > 0.8, partial


def calculate_curriculum_metrics(
    results: List[CurriculumResult]
) -> Dict[str, Any]:
    """
    Calculate aggregate metrics for curriculum task results.
    """
    if not results:
        return {"overall_accuracy": 0.0, "by_level": {}}
    
    valid_results = [r for r in results if r.error is None]
    
    if not valid_results:
        return {
            "overall_accuracy": 0.0,
            "by_level": {},
            "error_count": len(results),
        }
    
    # Overall metrics
    correct_count = sum(1 for r in valid_results if r.is_correct)
    avg_partial = sum(r.partial_score for r in valid_results) / len(valid_results)
    
    # Group by level
    by_level: Dict[CEFRLevel, List[CurriculumResult]] = {}
    for r in valid_results:
        if r.level not in by_level:
            by_level[r.level] = []
        by_level[r.level].append(r)
    
    level_metrics = {}
    for level, level_results in by_level.items():
        level_correct = sum(1 for r in level_results if r.is_correct)
        level_partial = sum(r.partial_score for r in level_results) / len(level_results)
        level_metrics[level.value] = {
            "accuracy": level_correct / len(level_results) * 100,
            "partial_score": level_partial * 100,
            "count": len(level_results),
        }
    
    return {
        "overall_accuracy": correct_count / len(valid_results) * 100,
        "overall_partial": avg_partial * 100,
        "count": len(valid_results),
        "error_count": len(results) - len(valid_results),
        "by_level": level_metrics,
    }

