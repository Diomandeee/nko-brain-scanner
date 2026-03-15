"""
Translation Task for N'Ko Benchmark.

Tests translation quality between N'Ko, English, and French.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from ..providers.base import BaseProvider, TaskType
from ..data.sampler import TranslationSample


@dataclass
class TranslationTaskResult:
    """Result of a single translation test."""
    sample_id: int
    source_text: str
    source_lang: str
    target_lang: str
    expected: str
    predicted: str
    success: bool
    latency_ms: float
    error: Optional[str] = None


@dataclass
class TranslationTaskResults:
    """Aggregated results for translation task."""
    results: List[TranslationTaskResult] = field(default_factory=list)
    total_tests: int = 0
    successful_tests: int = 0
    avg_latency_ms: float = 0.0
    
    # By direction
    nko_to_en_results: List[TranslationTaskResult] = field(default_factory=list)
    nko_to_fr_results: List[TranslationTaskResult] = field(default_factory=list)
    en_to_nko_results: List[TranslationTaskResult] = field(default_factory=list)
    fr_to_nko_results: List[TranslationTaskResult] = field(default_factory=list)


class TranslationTask:
    """
    Translation benchmark task.
    
    Tests AI models on:
    - N'Ko to English translation
    - N'Ko to French translation
    - English to N'Ko translation
    - French to N'Ko translation
    """
    
    WEIGHT = 0.40  # 40% of overall score
    
    def __init__(self):
        self.results = TranslationTaskResults()
    
    async def run(
        self,
        provider: BaseProvider,
        nko_to_en: List[TranslationSample],
        nko_to_fr: List[TranslationSample],
        en_to_nko: List[TranslationSample],
        fr_to_nko: List[TranslationSample],
        progress_callback: Optional[callable] = None,
    ) -> TranslationTaskResults:
        """
        Run translation benchmark.
        
        Args:
            provider: AI provider to test
            nko_to_en: N'Ko to English test samples
            nko_to_fr: N'Ko to French test samples
            en_to_nko: English to N'Ko test samples
            fr_to_nko: French to N'Ko test samples
            progress_callback: Optional callback for progress updates
            
        Returns:
            TranslationTaskResults with all test results
        """
        self.results = TranslationTaskResults()
        
        # Run each direction
        directions = [
            ("nko", "en", nko_to_en, self.results.nko_to_en_results),
            ("nko", "fr", nko_to_fr, self.results.nko_to_fr_results),
            ("en", "nko", en_to_nko, self.results.en_to_nko_results),
            ("fr", "nko", fr_to_nko, self.results.fr_to_nko_results),
        ]
        
        total_latency = 0.0
        
        for source_lang, target_lang, samples, results_list in directions:
            for i, sample in enumerate(samples):
                # Get source and expected text based on direction
                if source_lang == "nko":
                    source_text = sample.nko_text
                    expected = sample.english if target_lang == "en" else sample.french
                elif source_lang == "en":
                    source_text = sample.english
                    expected = sample.nko_text
                else:  # fr
                    source_text = sample.french
                    expected = sample.nko_text
                
                # Run translation
                result = await provider.run_task(
                    TaskType.TRANSLATION,
                    {
                        "text": source_text,
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                    }
                )
                
                task_result = TranslationTaskResult(
                    sample_id=i,
                    source_text=source_text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    expected=expected,
                    predicted=result.response,
                    success=result.success,
                    latency_ms=result.latency_ms,
                    error=result.error,
                )
                
                results_list.append(task_result)
                self.results.results.append(task_result)
                self.results.total_tests += 1
                
                if result.success:
                    self.results.successful_tests += 1
                    total_latency += result.latency_ms
                
                if progress_callback:
                    progress_callback(
                        task="translation",
                        direction=f"{source_lang}_to_{target_lang}",
                        current=i + 1,
                        total=len(samples),
                    )
        
        # Calculate average latency
        if self.results.successful_tests > 0:
            self.results.avg_latency_ms = total_latency / self.results.successful_tests
        
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for translation task."""
        return {
            "task": "translation",
            "weight": self.WEIGHT,
            "total_tests": self.results.total_tests,
            "successful_tests": self.results.successful_tests,
            "success_rate": self.results.successful_tests / max(1, self.results.total_tests),
            "avg_latency_ms": self.results.avg_latency_ms,
            "by_direction": {
                "nko_to_en": {
                    "total": len(self.results.nko_to_en_results),
                    "successful": sum(1 for r in self.results.nko_to_en_results if r.success),
                },
                "nko_to_fr": {
                    "total": len(self.results.nko_to_fr_results),
                    "successful": sum(1 for r in self.results.nko_to_fr_results if r.success),
                },
                "en_to_nko": {
                    "total": len(self.results.en_to_nko_results),
                    "successful": sum(1 for r in self.results.en_to_nko_results if r.success),
                },
                "fr_to_nko": {
                    "total": len(self.results.fr_to_nko_results),
                    "successful": sum(1 for r in self.results.fr_to_nko_results if r.success),
                },
            },
        }

