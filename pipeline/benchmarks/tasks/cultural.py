"""
Cultural Task for N'Ko Benchmark.

Tests AI models on cultural context, proverbs, idioms, and appropriate register.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from ..providers.base import BaseProvider, TaskType


@dataclass
class CulturalResult:
    """Result of a single cultural test."""
    sample_id: int
    nko_text: str
    cultural_type: str  # proverb, idiom, greeting, etc.
    expected_meaning: Optional[str]
    predicted_explanation: str
    success: bool
    latency_ms: float
    error: Optional[str] = None


@dataclass
class CulturalResults:
    """Aggregated results for cultural task."""
    results: List[CulturalResult] = field(default_factory=list)
    total_tests: int = 0
    successful_tests: int = 0
    avg_latency_ms: float = 0.0


class CulturalTask:
    """
    Cultural benchmark task.
    
    Tests AI models on:
    - Explaining Mandinka proverbs
    - Understanding idiomatic expressions
    - Appropriate register and formality
    - Cultural nuance in translations
    """
    
    WEIGHT = 0.10  # 10% of overall score
    
    def __init__(self):
        self.results = CulturalResults()
    
    async def run(
        self,
        provider: BaseProvider,
        cultural_samples: List[Dict[str, Any]],
        progress_callback: Optional[callable] = None,
    ) -> CulturalResults:
        """
        Run cultural benchmark.
        
        Args:
            provider: AI provider to test
            cultural_samples: List of cultural test samples
            progress_callback: Optional callback for progress updates
            
        Returns:
            CulturalResults with all test results
        """
        self.results = CulturalResults()
        
        total_latency = 0.0
        
        for i, sample in enumerate(cultural_samples):
            nko_text = sample.get("nko_text", "")
            cultural_type = sample.get("type", "proverb_candidate")
            
            # Build prompt based on cultural type
            if "proverb" in cultural_type:
                prompt = f"""Analyze this N'Ko text which may be a proverb or wise saying:

{nko_text}

Provide:
1. Literal translation to English
2. The deeper meaning or lesson
3. Cultural context in Manding society
4. When this might be used

Be culturally sensitive and accurate."""
            
            elif cultural_type == "greeting":
                prompt = f"""Explain this N'Ko greeting or social expression:

{nko_text}

Provide:
1. Meaning in English
2. When it's appropriate to use
3. The expected response
4. Level of formality"""
            
            else:  # General cultural expression
                prompt = f"""Explain the cultural meaning of this N'Ko text:

{nko_text}

Provide:
1. Translation to English
2. Cultural significance
3. Context of usage
4. Any related expressions"""
            
            result = await provider.run_task(
                TaskType.CULTURAL,
                {"prompt": prompt, "nko_text": nko_text}
            )
            
            task_result = CulturalResult(
                sample_id=i,
                nko_text=nko_text,
                cultural_type=cultural_type,
                expected_meaning=sample.get("english"),
                predicted_explanation=result.response,
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
                    task="cultural",
                    current=i + 1,
                    total=len(cultural_samples),
                )
        
        if self.results.successful_tests > 0:
            self.results.avg_latency_ms = total_latency / self.results.successful_tests
        
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for cultural task."""
        return {
            "task": "cultural",
            "weight": self.WEIGHT,
            "total_tests": self.results.total_tests,
            "successful_tests": self.results.successful_tests,
            "success_rate": self.results.successful_tests / max(1, self.results.total_tests),
            "avg_latency_ms": self.results.avg_latency_ms,
        }

