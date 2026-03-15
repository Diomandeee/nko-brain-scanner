"""
Translation Metrics for N'Ko Benchmark.

Calculates BLEU, chrF++, and other translation quality metrics.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    import sacrebleu
    HAS_SACREBLEU = True
except ImportError:
    HAS_SACREBLEU = False


@dataclass
class TranslationScores:
    """Translation quality scores."""
    bleu: float = 0.0
    bleu_1: float = 0.0
    bleu_2: float = 0.0
    bleu_3: float = 0.0
    bleu_4: float = 0.0
    brevity_penalty: float = 0.0
    chrf: float = 0.0  # chrF++
    ter: float = 0.0  # Translation Error Rate (optional)
    

class TranslationMetrics:
    """
    Calculate translation quality metrics.
    
    Metrics:
    - BLEU: Bilingual Evaluation Understudy
    - chrF++: Character n-gram F-score with word n-grams
    - TER: Translation Error Rate (optional)
    """
    
    def __init__(self):
        if not HAS_SACREBLEU:
            print("Warning: sacrebleu not installed. Install with: pip install sacrebleu")
    
    def calculate_bleu(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """
        Calculate BLEU score.
        
        Args:
            predictions: List of predicted translations
            references: List of reference translations
            
        Returns:
            Dict with BLEU scores
        """
        if not HAS_SACREBLEU:
            return {"bleu": 0.0, "bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0, "brevity_penalty": 1.0}
        
        if not predictions or not references:
            return {"bleu": 0.0, "bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0, "brevity_penalty": 1.0}
        
        try:
            # sacrebleu expects references as list of lists
            refs = [[ref] for ref in references]
            
            bleu = sacrebleu.corpus_bleu(predictions, list(zip(*refs)))
            
            return {
                "bleu": bleu.score,
                "bleu_1": bleu.precisions[0] if bleu.precisions else 0.0,
                "bleu_2": bleu.precisions[1] if len(bleu.precisions) > 1 else 0.0,
                "bleu_3": bleu.precisions[2] if len(bleu.precisions) > 2 else 0.0,
                "bleu_4": bleu.precisions[3] if len(bleu.precisions) > 3 else 0.0,
                "brevity_penalty": bleu.bp,
            }
        except Exception as e:
            print(f"BLEU calculation error: {e}")
            return {"bleu": 0.0, "bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0, "brevity_penalty": 1.0}
    
    def calculate_chrf(
        self,
        predictions: List[str],
        references: List[str],
    ) -> float:
        """
        Calculate chrF++ score.
        
        chrF++ is character-level metric that correlates well with human judgment,
        especially useful for morphologically rich languages like N'Ko.
        
        Args:
            predictions: List of predicted translations
            references: List of reference translations
            
        Returns:
            chrF++ score
        """
        if not HAS_SACREBLEU:
            return 0.0
        
        if not predictions or not references:
            return 0.0
        
        try:
            refs = [[ref] for ref in references]
            chrf = sacrebleu.corpus_chrf(predictions, list(zip(*refs)))
            return chrf.score
        except Exception as e:
            print(f"chrF calculation error: {e}")
            return 0.0
    
    def calculate_all(
        self,
        predictions: List[str],
        references: List[str],
    ) -> TranslationScores:
        """
        Calculate all translation metrics.
        
        Args:
            predictions: List of predicted translations
            references: List of reference translations
            
        Returns:
            TranslationScores with all metrics
        """
        bleu_scores = self.calculate_bleu(predictions, references)
        chrf_score = self.calculate_chrf(predictions, references)
        
        return TranslationScores(
            bleu=bleu_scores["bleu"],
            bleu_1=bleu_scores["bleu_1"],
            bleu_2=bleu_scores["bleu_2"],
            bleu_3=bleu_scores["bleu_3"],
            bleu_4=bleu_scores["bleu_4"],
            brevity_penalty=bleu_scores["brevity_penalty"],
            chrf=chrf_score,
        )
    
    def calculate_sentence_bleu(
        self,
        prediction: str,
        reference: str,
    ) -> float:
        """
        Calculate sentence-level BLEU.
        
        Args:
            prediction: Single predicted translation
            reference: Single reference translation
            
        Returns:
            Sentence BLEU score
        """
        if not HAS_SACREBLEU:
            return 0.0
        
        try:
            bleu = sacrebleu.sentence_bleu(prediction, [reference])
            return bleu.score
        except Exception:
            return 0.0


def calculate_exact_match(predictions: List[str], references: List[str]) -> float:
    """Calculate exact match accuracy."""
    if not predictions or not references:
        return 0.0
    
    matches = sum(
        1 for pred, ref in zip(predictions, references)
        if pred.strip().lower() == ref.strip().lower()
    )
    return matches / len(predictions)


def calculate_fuzzy_match(
    predictions: List[str],
    references: List[str],
    threshold: float = 0.8,
) -> float:
    """
    Calculate fuzzy match accuracy.
    
    Uses simple character overlap for fuzzy matching.
    """
    if not predictions or not references:
        return 0.0
    
    def char_overlap(s1: str, s2: str) -> float:
        """Calculate character-level Jaccard similarity."""
        set1 = set(s1.lower())
        set2 = set(s2.lower())
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)
    
    matches = sum(
        1 for pred, ref in zip(predictions, references)
        if char_overlap(pred, ref) >= threshold
    )
    return matches / len(predictions)

