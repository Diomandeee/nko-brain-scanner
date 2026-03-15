"""
Script Knowledge Task for N'Ko Benchmark.

Tests AI models on N'Ko script recognition, transliteration, and character knowledge.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from ..providers.base import BaseProvider, TaskType
from ..config import NKO_ALPHABET


@dataclass
class ScriptKnowledgeResult:
    """Result of a single script knowledge test."""
    sample_id: int
    nko_text: str
    task_type: str  # recognize, transliterate, identify_char
    expected: Optional[str]
    predicted: str
    success: bool
    latency_ms: float
    error: Optional[str] = None


@dataclass
class ScriptKnowledgeResults:
    """Aggregated results for script knowledge task."""
    results: List[ScriptKnowledgeResult] = field(default_factory=list)
    total_tests: int = 0
    successful_tests: int = 0
    avg_latency_ms: float = 0.0


class ScriptKnowledgeTask:
    """
    Script knowledge benchmark task.
    
    Tests AI models on:
    - N'Ko character recognition
    - Latin to N'Ko transliteration
    - N'Ko to Latin transliteration
    - Character identification and naming
    """
    
    WEIGHT = 0.15  # 15% of overall score
    
    def __init__(self):
        self.results = ScriptKnowledgeResults()
        self.alphabet = NKO_ALPHABET
    
    async def run(
        self,
        provider: BaseProvider,
        script_samples: List[Dict[str, str]],
        progress_callback: Optional[callable] = None,
    ) -> ScriptKnowledgeResults:
        """
        Run script knowledge benchmark.
        
        Args:
            provider: AI provider to test
            script_samples: List of script test samples
            progress_callback: Optional callback for progress updates
            
        Returns:
            ScriptKnowledgeResults with all test results
        """
        self.results = ScriptKnowledgeResults()
        
        total_latency = 0.0
        
        for i, sample in enumerate(script_samples):
            nko_text = sample.get("nko_text", "")
            task_type = sample.get("task", "recognize")
            
            # Build prompt based on task type
            if task_type == "recognize":
                prompt = f"""Identify and explain this N'Ko text: {nko_text}

Provide:
1. Latin transliteration
2. Meaning in English
3. Brief description of the N'Ko characters used"""
            
            elif task_type == "transliterate":
                prompt = f"""Transliterate this N'Ko text to Latin script: {nko_text}

Provide only the Latin transliteration, nothing else."""
            
            else:  # identify_char
                prompt = f"""Identify this N'Ko character: {nko_text}

Provide:
1. The character name
2. Its Latin equivalent
3. Common usage"""
            
            result = await provider.run_task(
                TaskType.SCRIPT_KNOWLEDGE,
                {"nko_text": nko_text, "prompt": prompt}
            )
            
            task_result = ScriptKnowledgeResult(
                sample_id=i,
                nko_text=nko_text,
                task_type=task_type,
                expected=sample.get("expected"),
                predicted=result.response,
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
                    task="script_knowledge",
                    current=i + 1,
                    total=len(script_samples),
                )
        
        if self.results.successful_tests > 0:
            self.results.avg_latency_ms = total_latency / self.results.successful_tests
        
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for script knowledge task."""
        return {
            "task": "script_knowledge",
            "weight": self.WEIGHT,
            "total_tests": self.results.total_tests,
            "successful_tests": self.results.successful_tests,
            "success_rate": self.results.successful_tests / max(1, self.results.total_tests),
            "avg_latency_ms": self.results.avg_latency_ms,
        }

