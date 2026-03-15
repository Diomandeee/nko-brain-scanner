"""
Compositional Task for N'Ko Benchmark.

Tests AI models on deep language understanding through:
- Novel word building
- Sentence construction
- Disambiguation
- Error correction
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from ..providers.base import BaseProvider, TaskType


@dataclass
class CompositionalResult:
    """Result of a single compositional test."""
    sample_id: int
    test_type: str  # novel_word, sentence_construction, disambiguation, error_correction
    prompt: str
    expected: str
    predicted: str
    success: bool
    latency_ms: float
    difficulty: str = "medium"
    error: Optional[str] = None


@dataclass
class CompositionalResults:
    """Aggregated results for compositional task."""
    results: List[CompositionalResult] = field(default_factory=list)
    total_tests: int = 0
    successful_tests: int = 0
    avg_latency_ms: float = 0.0
    
    # By test type
    novel_word_results: List[CompositionalResult] = field(default_factory=list)
    sentence_results: List[CompositionalResult] = field(default_factory=list)
    disambiguation_results: List[CompositionalResult] = field(default_factory=list)
    error_correction_results: List[CompositionalResult] = field(default_factory=list)


class CompositionalTask:
    """
    Compositional reasoning benchmark task.
    
    Tests AI models on deep language understanding:
    - Novel word building (morphological composition)
    - Sentence construction from vocabulary
    - Disambiguation of ambiguous text
    - Error correction in N'Ko text
    """
    
    WEIGHT = 0.15  # 15% of overall score
    
    def __init__(self):
        self.results = CompositionalResults()
    
    async def run(
        self,
        provider: BaseProvider,
        compositional_samples: List[Dict[str, Any]],
        progress_callback: Optional[callable] = None,
    ) -> CompositionalResults:
        """
        Run compositional reasoning benchmark.
        
        Args:
            provider: AI provider to test
            compositional_samples: List of compositional test samples
            progress_callback: Optional callback for progress updates
            
        Returns:
            CompositionalResults with all test results
        """
        self.results = CompositionalResults()
        
        total_latency = 0.0
        
        for i, sample in enumerate(compositional_samples):
            test_type = sample.get("type", "novel_word")
            prompt = sample.get("prompt", "")
            expected = sample.get("expected", "")
            difficulty = sample.get("difficulty", "medium")
            
            # Enhance prompt based on test type
            if test_type == "novel_word":
                full_prompt = f"""N'Ko Morphology Test:

{prompt}

Provide:
1. The combined word in N'Ko script
2. The meaning of the combined word
3. Brief explanation of how the morphemes combine"""
            
            elif test_type == "sentence_construction":
                full_prompt = f"""N'Ko Sentence Construction:

{prompt}

Provide:
1. The constructed N'Ko sentence
2. Verify it matches the target meaning
3. Note any grammar rules applied"""
            
            elif test_type == "disambiguation":
                full_prompt = f"""N'Ko Disambiguation:

{prompt}

List ALL possible interpretations with:
1. Each interpretation's meaning
2. The context in which each would apply
3. Any grammatical cues that distinguish them"""
            
            elif test_type == "error_correction":
                full_prompt = f"""N'Ko Error Correction:

{prompt}

Provide:
1. The corrected N'Ko text
2. What error was found
3. Brief explanation of the correction"""
            
            else:
                full_prompt = prompt
            
            result = await provider.run_task(
                TaskType.COMPOSITIONAL,
                {"prompt": full_prompt}
            )
            
            task_result = CompositionalResult(
                sample_id=i,
                test_type=test_type,
                prompt=prompt,
                expected=expected,
                predicted=result.response,
                success=result.success,
                latency_ms=result.latency_ms,
                difficulty=difficulty,
                error=result.error,
            )
            
            # Add to appropriate result list
            self.results.results.append(task_result)
            self.results.total_tests += 1
            
            if test_type == "novel_word":
                self.results.novel_word_results.append(task_result)
            elif test_type == "sentence_construction":
                self.results.sentence_results.append(task_result)
            elif test_type == "disambiguation":
                self.results.disambiguation_results.append(task_result)
            elif test_type == "error_correction":
                self.results.error_correction_results.append(task_result)
            
            if result.success:
                self.results.successful_tests += 1
                total_latency += result.latency_ms
            
            if progress_callback:
                progress_callback(
                    task="compositional",
                    test_type=test_type,
                    current=i + 1,
                    total=len(compositional_samples),
                )
        
        if self.results.successful_tests > 0:
            self.results.avg_latency_ms = total_latency / self.results.successful_tests
        
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for compositional task."""
        return {
            "task": "compositional",
            "weight": self.WEIGHT,
            "total_tests": self.results.total_tests,
            "successful_tests": self.results.successful_tests,
            "success_rate": self.results.successful_tests / max(1, self.results.total_tests),
            "avg_latency_ms": self.results.avg_latency_ms,
            "by_type": {
                "novel_word": {
                    "total": len(self.results.novel_word_results),
                    "successful": sum(1 for r in self.results.novel_word_results if r.success),
                },
                "sentence_construction": {
                    "total": len(self.results.sentence_results),
                    "successful": sum(1 for r in self.results.sentence_results if r.success),
                },
                "disambiguation": {
                    "total": len(self.results.disambiguation_results),
                    "successful": sum(1 for r in self.results.disambiguation_results if r.success),
                },
                "error_correction": {
                    "total": len(self.results.error_correction_results),
                    "successful": sum(1 for r in self.results.error_correction_results if r.success),
                },
            },
        }

