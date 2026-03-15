"""
Vocabulary Task for N'Ko Benchmark.

Tests AI models on N'Ko vocabulary definitions, word classes, and morphology.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from ..providers.base import BaseProvider, TaskType
from ..data.sampler import VocabularySample


@dataclass
class VocabularyResult:
    """Result of a single vocabulary test."""
    sample_id: int
    word: str
    expected_meaning: Optional[str]
    expected_class: Optional[str]
    predicted_meaning: str
    predicted_class: Optional[str]
    success: bool
    latency_ms: float
    error: Optional[str] = None


@dataclass
class VocabularyResults:
    """Aggregated results for vocabulary task."""
    results: List[VocabularyResult] = field(default_factory=list)
    total_tests: int = 0
    successful_tests: int = 0
    avg_latency_ms: float = 0.0


class VocabularyTask:
    """
    Vocabulary benchmark task.
    
    Tests AI models on:
    - Defining N'Ko words
    - Identifying word classes (noun, verb, etc.)
    - Providing usage examples
    - Understanding morphological structure
    """
    
    WEIGHT = 0.20  # 20% of overall score
    
    def __init__(self):
        self.results = VocabularyResults()
    
    async def run(
        self,
        provider: BaseProvider,
        vocabulary_samples: List[VocabularySample],
        progress_callback: Optional[callable] = None,
    ) -> VocabularyResults:
        """
        Run vocabulary benchmark.
        
        Args:
            provider: AI provider to test
            vocabulary_samples: List of vocabulary test samples
            progress_callback: Optional callback for progress updates
            
        Returns:
            VocabularyResults with all test results
        """
        self.results = VocabularyResults()
        
        total_latency = 0.0
        
        for i, sample in enumerate(vocabulary_samples):
            # Build prompt
            prompt = f"""Define this N'Ko/Bambara word: {sample.word}

Provide:
1. Primary meaning in English
2. Word class (noun, verb, adjective, adverb, pronoun, etc.)
3. 1-2 example sentences if possible
4. Any related words or variants

Be concise and accurate."""
            
            result = await provider.run_task(
                TaskType.VOCABULARY,
                {"word": sample.word, "prompt": prompt}
            )
            
            # Extract expected values
            expected_meaning = None
            if sample.definitions_en:
                expected_meaning = sample.definitions_en[0]
            
            task_result = VocabularyResult(
                sample_id=i,
                word=sample.word,
                expected_meaning=expected_meaning,
                expected_class=sample.word_class,
                predicted_meaning=result.response,
                predicted_class=None,  # Would need NLP to extract
                success=result.success,
                latency_ms=result.latency_ms,
                error=result.error,
            )
            
            self.results.results.append(task_result)
            self.results.total_tests += 1
            
            if result.success:
                self.results.successful_tests += 1
                total_latency += result.latency_ms
            
            if progress_callback:
                progress_callback(
                    task="vocabulary",
                    current=i + 1,
                    total=len(vocabulary_samples),
                )
        
        if self.results.successful_tests > 0:
            self.results.avg_latency_ms = total_latency / self.results.successful_tests
        
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for vocabulary task."""
        return {
            "task": "vocabulary",
            "weight": self.WEIGHT,
            "total_tests": self.results.total_tests,
            "successful_tests": self.results.successful_tests,
            "success_rate": self.results.successful_tests / max(1, self.results.total_tests),
            "avg_latency_ms": self.results.avg_latency_ms,
        }

