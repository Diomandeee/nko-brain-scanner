"""
Accuracy Metrics for N'Ko Benchmark.

Calculates accuracy metrics for non-translation tasks.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re


@dataclass
class AccuracyScores:
    """Accuracy scores for a task."""
    exact_match: float = 0.0
    partial_match: float = 0.0
    content_overlap: float = 0.0
    response_rate: float = 0.0  # % of successful API calls


class AccuracyMetrics:
    """
    Calculate accuracy metrics for benchmark tasks.
    
    Supports:
    - Exact match (case-insensitive)
    - Partial match (contains expected)
    - Content overlap (word/character overlap)
    - Response rate (successful API calls)
    """
    
    def calculate_exact_match(
        self,
        predictions: List[str],
        references: List[str],
        case_sensitive: bool = False,
    ) -> float:
        """
        Calculate exact match accuracy.
        
        Args:
            predictions: List of predictions
            references: List of expected values
            case_sensitive: Whether to do case-sensitive matching
            
        Returns:
            Exact match accuracy (0.0 to 1.0)
        """
        if not predictions or not references:
            return 0.0
        
        if len(predictions) != len(references):
            min_len = min(len(predictions), len(references))
            predictions = predictions[:min_len]
            references = references[:min_len]
        
        matches = 0
        for pred, ref in zip(predictions, references):
            if case_sensitive:
                if pred.strip() == ref.strip():
                    matches += 1
            else:
                if pred.strip().lower() == ref.strip().lower():
                    matches += 1
        
        return matches / len(predictions)
    
    def calculate_partial_match(
        self,
        predictions: List[str],
        references: List[str],
    ) -> float:
        """
        Calculate partial match accuracy (prediction contains reference).
        
        Args:
            predictions: List of predictions
            references: List of expected values
            
        Returns:
            Partial match accuracy (0.0 to 1.0)
        """
        if not predictions or not references:
            return 0.0
        
        if len(predictions) != len(references):
            min_len = min(len(predictions), len(references))
            predictions = predictions[:min_len]
            references = references[:min_len]
        
        matches = 0
        for pred, ref in zip(predictions, references):
            pred_lower = pred.lower()
            ref_lower = ref.lower()
            
            # Check if reference content appears in prediction
            if ref_lower in pred_lower:
                matches += 1
            # Also check words
            elif any(word in pred_lower for word in ref_lower.split() if len(word) > 2):
                matches += 0.5  # Partial credit
        
        return matches / len(predictions)
    
    def calculate_word_overlap(
        self,
        predictions: List[str],
        references: List[str],
    ) -> float:
        """
        Calculate average word overlap ratio.
        
        Args:
            predictions: List of predictions
            references: List of expected values
            
        Returns:
            Average word overlap (0.0 to 1.0)
        """
        if not predictions or not references:
            return 0.0
        
        if len(predictions) != len(references):
            min_len = min(len(predictions), len(references))
            predictions = predictions[:min_len]
            references = references[:min_len]
        
        total_overlap = 0.0
        
        for pred, ref in zip(predictions, references):
            pred_words = set(re.findall(r'\w+', pred.lower()))
            ref_words = set(re.findall(r'\w+', ref.lower()))
            
            if not pred_words or not ref_words:
                continue
            
            overlap = len(pred_words & ref_words) / len(pred_words | ref_words)
            total_overlap += overlap
        
        return total_overlap / len(predictions)
    
    def calculate_response_rate(
        self,
        successes: List[bool],
    ) -> float:
        """
        Calculate response success rate.
        
        Args:
            successes: List of success flags
            
        Returns:
            Success rate (0.0 to 1.0)
        """
        if not successes:
            return 0.0
        
        return sum(1 for s in successes if s) / len(successes)
    
    def calculate_all(
        self,
        predictions: List[str],
        references: List[str],
        successes: Optional[List[bool]] = None,
    ) -> AccuracyScores:
        """
        Calculate all accuracy metrics.
        
        Args:
            predictions: List of predictions
            references: List of expected values
            successes: Optional list of success flags
            
        Returns:
            AccuracyScores with all metrics
        """
        return AccuracyScores(
            exact_match=self.calculate_exact_match(predictions, references),
            partial_match=self.calculate_partial_match(predictions, references),
            content_overlap=self.calculate_word_overlap(predictions, references),
            response_rate=self.calculate_response_rate(successes) if successes else 1.0,
        )


def calculate_nko_specific_accuracy(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Calculate N'Ko-specific accuracy metrics.
    
    Checks for:
    - Correct N'Ko script characters
    - Proper diacritical marks
    - Valid word structure
    """
    if not predictions or not references:
        return {
            "nko_char_accuracy": 0.0,
            "diacritic_accuracy": 0.0,
        }
    
    total_nko_chars = 0
    correct_nko_chars = 0
    total_diacritics = 0
    correct_diacritics = 0
    
    # N'Ko Unicode range: U+07C0 to U+07FF
    nko_pattern = re.compile(r'[\u07C0-\u07FF]')
    # Diacritical marks in N'Ko
    diacritic_pattern = re.compile(r'[\u07EB-\u07F5]')
    
    for pred, ref in zip(predictions, references):
        # Count N'Ko characters
        pred_nko = set(nko_pattern.findall(pred))
        ref_nko = set(nko_pattern.findall(ref))
        
        if ref_nko:
            total_nko_chars += len(ref_nko)
            correct_nko_chars += len(pred_nko & ref_nko)
        
        # Count diacritics
        pred_diacritics = set(diacritic_pattern.findall(pred))
        ref_diacritics = set(diacritic_pattern.findall(ref))
        
        if ref_diacritics:
            total_diacritics += len(ref_diacritics)
            correct_diacritics += len(pred_diacritics & ref_diacritics)
    
    return {
        "nko_char_accuracy": correct_nko_chars / max(1, total_nko_chars),
        "diacritic_accuracy": correct_diacritics / max(1, total_diacritics),
    }

