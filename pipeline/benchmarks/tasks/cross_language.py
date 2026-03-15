"""
Cross-Language Translation Tasks for Manding Languages.

Evaluates AI models on translation between Manding language variants:
- N'Ko ↔ Bambara (script transliteration + translation)
- N'Ko ↔ Malinke
- Bambara ↔ Jula
- Script conversion (N'Ko script ↔ Latin)
- Dialect identification
- Cognate recognition
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from ..data.manding_loader import (
    Language,
    TranslationPair,
    CognatePair,
    VocabEntry,
)
from ..providers.base import BaseProvider


class CrossLanguageTaskType(Enum):
    """Types of cross-language tasks."""
    TRANSLATION = "translation"
    SCRIPT_CONVERSION = "script_conversion"
    DIALECT_IDENTIFICATION = "dialect_identification"
    COGNATE_RECOGNITION = "cognate_recognition"
    BACK_TRANSLATION = "back_translation"


@dataclass
class CrossLanguageResult:
    """Result of a cross-language task."""
    task_id: str
    task_type: CrossLanguageTaskType
    source_lang: Language
    target_lang: Language
    source_text: str
    reference_text: str
    prediction: Optional[str] = None
    is_correct: bool = False
    partial_match: float = 0.0
    latency_ms: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CrossMandingTranslationTask:
    """
    Test translation between Manding language variants.
    
    This task evaluates the model's ability to translate between:
    - N'Ko (script) ↔ Bambara (Latin)
    - N'Ko ↔ French
    - Bambara ↔ French
    - Bambara ↔ English
    """
    
    def __init__(self):
        self.name = "cross_manding_translation"
        self.description = "Translation between Manding language variants"
    
    async def run(
        self,
        provider: BaseProvider,
        translation_pairs: List[TranslationPair],
        source_lang: Language,
        target_lang: Language,
        progress_callback: Optional[callable] = None,
    ) -> List[CrossLanguageResult]:
        """
        Run cross-language translation tests.
        
        Args:
            provider: LLM provider to test
            translation_pairs: Pairs to use for testing
            source_lang: Source language
            target_lang: Target language
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of CrossLanguageResult objects
        """
        results = []
        
        # Filter pairs that have the required language texts
        valid_pairs = self._filter_pairs_for_direction(
            translation_pairs, source_lang, target_lang
        )
        
        for i, pair in enumerate(valid_pairs):
            if progress_callback:
                progress_callback(i, len(valid_pairs), "cross_translation")
            
            source_text = self._get_text_for_lang(pair, source_lang)
            reference_text = self._get_text_for_lang(pair, target_lang)
            
            if not source_text or not reference_text:
                continue
            
            # Build translation prompt
            try:
                import time
                start_time = time.time()
                
                # Use provider's translate method
                lang_code_map = {
                    Language.NKO: "nko",
                    Language.BAMBARA: "bam",
                    Language.MALINKE: "man",
                    Language.JULA: "dyu",
                    Language.ENGLISH: "en",
                    Language.FRENCH: "fr",
                }
                
                source_code = lang_code_map.get(source_lang, source_lang.value)
                target_code = lang_code_map.get(target_lang, target_lang.value)
                
                response = await provider.translate(
                    text=source_text,
                    source_lang=source_code,
                    target_lang=target_code,
                )
                
                latency_ms = int(response.latency_ms)
                prediction = response.translation.strip() if response.success else ""
                
                # Calculate match scores
                is_correct, partial_match = self._evaluate_translation(
                    prediction, reference_text
                )
                
                result = CrossLanguageResult(
                    task_id=f"{self.name}_{pair.id}",
                    task_type=CrossLanguageTaskType.TRANSLATION,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    source_text=source_text,
                    reference_text=reference_text,
                    prediction=prediction,
                    is_correct=is_correct,
                    partial_match=partial_match,
                    latency_ms=latency_ms,
                    metadata={
                        "pair_source": pair.source,
                        "text_type": pair.text_type,
                    }
                )
                
            except Exception as e:
                result = CrossLanguageResult(
                    task_id=f"{self.name}_{pair.id}",
                    task_type=CrossLanguageTaskType.TRANSLATION,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    source_text=source_text,
                    reference_text=reference_text,
                    error=str(e),
                )
            
            results.append(result)
        
        return results
    
    def _filter_pairs_for_direction(
        self,
        pairs: List[TranslationPair],
        source_lang: Language,
        target_lang: Language,
    ) -> List[TranslationPair]:
        """Filter pairs that have text for both source and target languages."""
        valid = []
        for pair in pairs:
            source_text = self._get_text_for_lang(pair, source_lang)
            target_text = self._get_text_for_lang(pair, target_lang)
            if source_text and target_text:
                valid.append(pair)
        return valid
    
    def _get_text_for_lang(
        self, pair: TranslationPair, lang: Language
    ) -> Optional[str]:
        """Get text from pair for specified language."""
        if lang == Language.NKO:
            return pair.nko_text
        elif lang == Language.BAMBARA:
            return pair.bambara_text
        elif lang == Language.ENGLISH:
            return pair.english_text
        elif lang == Language.FRENCH:
            return pair.french_text
        elif lang == pair.source_lang:
            return pair.source_text
        elif lang == pair.target_lang:
            return pair.target_text
        return None
    
    def _build_translation_prompt(
        self,
        text: str,
        source_lang: Language,
        target_lang: Language,
    ) -> str:
        """Build a translation prompt."""
        lang_names = {
            Language.NKO: "N'Ko (Manding language in N'Ko script)",
            Language.BAMBARA: "Bambara (Manding language in Latin script)",
            Language.MALINKE: "Malinke (Manding language in Latin script)",
            Language.JULA: "Jula/Dioula (Manding language in Latin script)",
            Language.ENGLISH: "English",
            Language.FRENCH: "French",
        }
        
        source_name = lang_names.get(source_lang, source_lang.value)
        target_name = lang_names.get(target_lang, target_lang.value)
        
        prompt = f"""Translate the following text from {source_name} to {target_name}.

Source text ({source_lang.value}): {text}

Provide only the translation in {target_lang.value}, nothing else.

Translation:"""
        
        return prompt
    
    def _evaluate_translation(
        self, prediction: str, reference: str
    ) -> Tuple[bool, float]:
        """Evaluate translation quality."""
        if not prediction:
            return False, 0.0
        
        # Exact match
        pred_lower = prediction.lower().strip()
        ref_lower = reference.lower().strip()
        
        if pred_lower == ref_lower:
            return True, 1.0
        
        # Word overlap
        pred_words = set(pred_lower.split())
        ref_words = set(ref_lower.split())
        
        if not ref_words:
            return False, 0.0
        
        overlap = len(pred_words & ref_words)
        partial = overlap / len(ref_words)
        
        return False, partial


class ScriptTransliterationTask:
    """
    Test N'Ko script ↔ Latin script conversion.
    
    Evaluates the model's ability to:
    - Convert N'Ko script to Latin transliteration
    - Convert Latin to N'Ko script
    """
    
    def __init__(self):
        self.name = "script_transliteration"
        self.description = "N'Ko script ↔ Latin script conversion"
    
    async def run(
        self,
        provider: BaseProvider,
        vocabulary: List[VocabEntry],
        direction: str = "nko_to_latin",  # or "latin_to_nko"
        progress_callback: Optional[callable] = None,
    ) -> List[CrossLanguageResult]:
        """Run script transliteration tests."""
        results = []
        
        # Filter vocab with both N'Ko and Latin forms
        valid_vocab = [
            v for v in vocabulary
            if v.word and v.latin_transcription
        ]
        
        for i, entry in enumerate(valid_vocab):
            if progress_callback:
                progress_callback(i, len(valid_vocab), "script_conversion")
            
            if direction == "nko_to_latin":
                source_text = entry.word
                reference_text = entry.latin_transcription
                prompt = f"""Convert this N'Ko script text to Latin transliteration.

N'Ko: {source_text}

Provide only the Latin transliteration, nothing else.

Latin:"""
            else:
                source_text = entry.latin_transcription
                reference_text = entry.word
                prompt = f"""Convert this Latin transliteration to N'Ko script.

Latin: {source_text}

Provide only the N'Ko script text, nothing else.

N'Ko:"""
            
            try:
                import time
                start_time = time.time()
                
                response = await provider.complete(prompt)
                
                latency_ms = int(response.latency_ms)
                prediction = response.completion.strip() if response.success else ""
                
                # Evaluate
                is_correct = prediction.lower() == reference_text.lower()
                partial_match = self._calculate_similarity(prediction, reference_text)
                
                result = CrossLanguageResult(
                    task_id=f"{self.name}_{entry.id}",
                    task_type=CrossLanguageTaskType.SCRIPT_CONVERSION,
                    source_lang=Language.NKO if direction == "nko_to_latin" else Language.BAMBARA,
                    target_lang=Language.BAMBARA if direction == "nko_to_latin" else Language.NKO,
                    source_text=source_text,
                    reference_text=reference_text,
                    prediction=prediction,
                    is_correct=is_correct,
                    partial_match=partial_match,
                    latency_ms=latency_ms,
                    metadata={"direction": direction},
                )
                
            except Exception as e:
                result = CrossLanguageResult(
                    task_id=f"{self.name}_{entry.id}",
                    task_type=CrossLanguageTaskType.SCRIPT_CONVERSION,
                    source_lang=Language.NKO if direction == "nko_to_latin" else Language.BAMBARA,
                    target_lang=Language.BAMBARA if direction == "nko_to_latin" else Language.NKO,
                    source_text=source_text,
                    reference_text=reference_text,
                    error=str(e),
                )
            
            results.append(result)
        
        return results
    
    def _calculate_similarity(self, pred: str, ref: str) -> float:
        """Calculate character-level similarity."""
        if not pred or not ref:
            return 0.0
        
        pred = pred.lower()
        ref = ref.lower()
        
        # Simple character overlap
        pred_chars = set(pred)
        ref_chars = set(ref)
        
        if not ref_chars:
            return 0.0
        
        overlap = len(pred_chars & ref_chars)
        return overlap / len(ref_chars)


class DialectIdentificationTask:
    """
    Test identification of Manding dialect variants.
    
    Evaluates whether the model can distinguish between:
    - Bambara
    - Malinke/Maninka
    - Jula/Dioula
    """
    
    def __init__(self):
        self.name = "dialect_identification"
        self.description = "Identify Manding dialect variants"
        
        # Sample phrases for each dialect
        self.dialect_samples = {
            "bambara": [
                ("I ni ce", "Hello (to a man)"),
                ("I ni sogoma", "Good morning"),
                ("N bɛ baara la", "I am working"),
            ],
            "malinke": [
                ("M ba di", "My mother"),
                ("A ka di", "It is good"),
            ],
            "jula": [
                ("A ye dɔn", "He/she knows"),
            ],
        }
    
    async def run(
        self,
        provider: BaseProvider,
        pairs: List[TranslationPair],
        progress_callback: Optional[callable] = None,
    ) -> List[CrossLanguageResult]:
        """Run dialect identification tests."""
        results = []
        
        # Filter Bambara pairs (we know their dialect)
        bambara_pairs = [p for p in pairs if p.source_lang == Language.BAMBARA]
        
        for i, pair in enumerate(bambara_pairs[:50]):  # Limit for this task
            if progress_callback:
                progress_callback(i, min(50, len(bambara_pairs)), "dialect_id")
            
            prompt = f"""Identify the Manding dialect/variant of the following text.
The options are: Bambara, Malinke (Maninka), or Jula (Dioula).

Text: {pair.source_text}

Respond with only the dialect name (Bambara, Malinke, or Jula):"""
            
            try:
                import time
                start_time = time.time()
                
                response = await provider.complete(prompt)
                
                latency_ms = int(response.latency_ms)
                prediction = response.completion.strip().lower() if response.success else ""
                
                # Check if correctly identified as Bambara
                is_correct = "bambara" in prediction
                
                result = CrossLanguageResult(
                    task_id=f"{self.name}_{pair.id}",
                    task_type=CrossLanguageTaskType.DIALECT_IDENTIFICATION,
                    source_lang=Language.BAMBARA,
                    target_lang=Language.BAMBARA,  # Same (identification task)
                    source_text=pair.source_text,
                    reference_text="bambara",
                    prediction=prediction,
                    is_correct=is_correct,
                    partial_match=1.0 if is_correct else 0.0,
                    latency_ms=latency_ms,
                )
                
            except Exception as e:
                result = CrossLanguageResult(
                    task_id=f"{self.name}_{pair.id}",
                    task_type=CrossLanguageTaskType.DIALECT_IDENTIFICATION,
                    source_lang=Language.BAMBARA,
                    target_lang=Language.BAMBARA,
                    source_text=pair.source_text,
                    reference_text="bambara",
                    error=str(e),
                )
            
            results.append(result)
        
        return results


class CognateRecognitionTask:
    """
    Test recognition of cognate words across Manding variants.
    
    Evaluates whether the model can match related words between:
    - N'Ko and Bambara
    - Bambara and Malinke
    """
    
    def __init__(self):
        self.name = "cognate_recognition"
        self.description = "Match cognate words across Manding variants"
    
    async def run(
        self,
        provider: BaseProvider,
        cognates: List[CognatePair],
        progress_callback: Optional[callable] = None,
    ) -> List[CrossLanguageResult]:
        """Run cognate recognition tests."""
        results = []
        
        for i, cognate in enumerate(cognates):
            if progress_callback:
                progress_callback(i, len(cognates), "cognate_recognition")
            
            if not cognate.nko_form or not cognate.bambara_form:
                continue
            
            prompt = f"""Given this N'Ko word and its meaning, identify the equivalent Bambara word.

N'Ko word: {cognate.nko_form}
Meaning: {cognate.meaning}

What is the Bambara (Latin script) equivalent? Provide only the word:"""
            
            try:
                import time
                start_time = time.time()
                
                response = await provider.complete(prompt)
                
                latency_ms = int(response.latency_ms)
                prediction = response.completion.strip().lower() if response.success else ""
                reference = cognate.bambara_form.lower()
                
                is_correct = prediction == reference
                partial_match = self._word_similarity(prediction, reference)
                
                result = CrossLanguageResult(
                    task_id=f"{self.name}_{cognate.id}",
                    task_type=CrossLanguageTaskType.COGNATE_RECOGNITION,
                    source_lang=Language.NKO,
                    target_lang=Language.BAMBARA,
                    source_text=cognate.nko_form,
                    reference_text=cognate.bambara_form,
                    prediction=prediction,
                    is_correct=is_correct,
                    partial_match=partial_match,
                    latency_ms=latency_ms,
                    metadata={"meaning": cognate.meaning},
                )
                
            except Exception as e:
                result = CrossLanguageResult(
                    task_id=f"{self.name}_{cognate.id}",
                    task_type=CrossLanguageTaskType.COGNATE_RECOGNITION,
                    source_lang=Language.NKO,
                    target_lang=Language.BAMBARA,
                    source_text=cognate.nko_form,
                    reference_text=cognate.bambara_form,
                    error=str(e),
                )
            
            results.append(result)
        
        return results
    
    def _word_similarity(self, pred: str, ref: str) -> float:
        """Calculate word similarity using character overlap."""
        if not pred or not ref:
            return 0.0
        
        pred_chars = set(pred)
        ref_chars = set(ref)
        
        if not ref_chars:
            return 0.0
        
        intersection = len(pred_chars & ref_chars)
        union = len(pred_chars | ref_chars)
        
        return intersection / union if union > 0 else 0.0


class BackTranslationTask:
    """
    Test round-trip translation quality.
    
    Translates A → B → A and evaluates preservation of meaning.
    """
    
    def __init__(self):
        self.name = "back_translation"
        self.description = "Round-trip translation evaluation"
    
    async def run(
        self,
        provider: BaseProvider,
        pairs: List[TranslationPair],
        source_lang: Language,
        pivot_lang: Language,
        progress_callback: Optional[callable] = None,
    ) -> List[CrossLanguageResult]:
        """
        Run back-translation tests.
        
        Translates: source_lang → pivot_lang → source_lang
        Then compares with original.
        """
        results = []
        
        for i, pair in enumerate(pairs):
            if progress_callback:
                progress_callback(i, len(pairs), "back_translation")
            
            source_text = pair.source_text
            
            # Step 1: Translate to pivot
            prompt_forward = f"""Translate to {pivot_lang.value}:
{source_text}

Translation:"""
            
            try:
                import time
                start_time = time.time()
                
                # Forward translation
                response1 = await provider.complete(prompt_forward)
                pivot_text = response1.completion.strip() if response1.success else ""
                
                # Back translation
                prompt_back = f"""Translate to {source_lang.value}:
{pivot_text}

Translation:"""
                
                response2 = await provider.complete(prompt_back)
                back_text = response2.completion.strip() if response2.success else ""
                
                latency_ms = int(response1.latency_ms + response2.latency_ms)
                
                # Evaluate preservation
                is_correct, partial_match = self._evaluate_preservation(
                    source_text, back_text
                )
                
                result = CrossLanguageResult(
                    task_id=f"{self.name}_{pair.id}",
                    task_type=CrossLanguageTaskType.BACK_TRANSLATION,
                    source_lang=source_lang,
                    target_lang=source_lang,  # Same (round-trip)
                    source_text=source_text,
                    reference_text=source_text,  # Original is reference
                    prediction=back_text,
                    is_correct=is_correct,
                    partial_match=partial_match,
                    latency_ms=latency_ms,
                    metadata={
                        "pivot_lang": pivot_lang.value,
                        "pivot_text": pivot_text,
                    },
                )
                
            except Exception as e:
                result = CrossLanguageResult(
                    task_id=f"{self.name}_{pair.id}",
                    task_type=CrossLanguageTaskType.BACK_TRANSLATION,
                    source_lang=source_lang,
                    target_lang=source_lang,
                    source_text=source_text,
                    reference_text=source_text,
                    error=str(e),
                )
            
            results.append(result)
        
        return results
    
    def _evaluate_preservation(
        self, original: str, back_translated: str
    ) -> Tuple[bool, float]:
        """Evaluate how well the original meaning is preserved."""
        if not back_translated:
            return False, 0.0
        
        orig_lower = original.lower().strip()
        back_lower = back_translated.lower().strip()
        
        if orig_lower == back_lower:
            return True, 1.0
        
        # Word overlap
        orig_words = set(orig_lower.split())
        back_words = set(back_lower.split())
        
        if not orig_words:
            return False, 0.0
        
        overlap = len(orig_words & back_words)
        partial = overlap / len(orig_words)
        
        return partial > 0.8, partial


def calculate_cross_language_metrics(
    results: List[CrossLanguageResult]
) -> Dict[str, Any]:
    """
    Calculate aggregate metrics for cross-language task results.
    
    Returns:
        Dict with accuracy, average partial match, etc.
    """
    if not results:
        return {"accuracy": 0.0, "partial_match": 0.0, "count": 0}
    
    valid_results = [r for r in results if r.error is None]
    
    if not valid_results:
        return {
            "accuracy": 0.0,
            "partial_match": 0.0,
            "count": len(results),
            "error_count": len(results),
        }
    
    correct_count = sum(1 for r in valid_results if r.is_correct)
    avg_partial = sum(r.partial_match for r in valid_results) / len(valid_results)
    avg_latency = sum(r.latency_ms for r in valid_results) / len(valid_results)
    
    # Group by task type
    by_type: Dict[str, List[CrossLanguageResult]] = {}
    for r in valid_results:
        key = r.task_type.value
        if key not in by_type:
            by_type[key] = []
        by_type[key].append(r)
    
    type_metrics = {}
    for task_type, type_results in by_type.items():
        type_correct = sum(1 for r in type_results if r.is_correct)
        type_partial = sum(r.partial_match for r in type_results) / len(type_results)
        type_metrics[task_type] = {
            "accuracy": type_correct / len(type_results) * 100,
            "partial_match": type_partial * 100,
            "count": len(type_results),
        }
    
    return {
        "accuracy": correct_count / len(valid_results) * 100,
        "partial_match": avg_partial * 100,
        "avg_latency_ms": avg_latency,
        "count": len(valid_results),
        "error_count": len(results) - len(valid_results),
        "by_task_type": type_metrics,
    }

